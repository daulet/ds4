use std::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use cuda_core::{
    DeviceBuffer, IntoResult, ManagedBuffer, ReadOnlyPageableHostMemory,
    ReadOnlyRegisteredHostMemory,
};

#[cfg(feature = "cuda-oxide-kernels")]
use crate::abi_kernels::AbiKernelModule;
use crate::allocation_policy::managed_kv_decision;
use crate::substrate::CudaOxideSubstrate;

static BACKEND: Mutex<Option<CudaOxideSubstrate>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_KERNELS: Mutex<Option<AbiKernelModule>> = Mutex::new(None);
static ABI_MODEL_RANGES: Mutex<Vec<AbiModelRange>> = Mutex::new(Vec::new());
static ABI_PAGEABLE_MODEL_RANGE: Mutex<Option<AbiPageableModelRange>> = Mutex::new(None);
static ABI_COPIED_MODEL: Mutex<Option<AbiCopiedModel>> = Mutex::new(None);
static ABI_REGISTERED_MODEL: Mutex<Option<AbiRegisteredModel>> = Mutex::new(None);
static ABI_MODEL_CONTROL: Mutex<AbiModelControl> = Mutex::new(AbiModelControl {
    model_map: 0,
    model_size: 0,
    model_fd: -1,
    model_fd_host_base: 0,
    model_file_size: 0,
    model_direct_align: 1,
    #[cfg(target_os = "linux")]
    model_direct_file: None,
});

struct AbiModelRange {
    model_map: usize,
    model_size: u64,
    offset: u64,
    bytes: u64,
    storage: AbiModelRangeStorage,
}

enum AbiModelRangeStorage {
    DeviceCopy(DeviceBuffer<u8>),
    BufferedFdDeviceCopy(DeviceBuffer<u8>),
    DirectIoFdDeviceCopy(DeviceBuffer<u8>),
    ReadOnlyRegistered {
        _registration: ReadOnlyRegisteredHostMemory<'static, u8>,
        requested_device_ptr: u64,
    },
}

impl AbiModelRange {
    fn device_ptr(&self) -> u64 {
        match &self.storage {
            AbiModelRangeStorage::DeviceCopy(buffer)
            | AbiModelRangeStorage::BufferedFdDeviceCopy(buffer)
            | AbiModelRangeStorage::DirectIoFdDeviceCopy(buffer) => buffer.cu_deviceptr(),
            AbiModelRangeStorage::ReadOnlyRegistered {
                requested_device_ptr,
                ..
            } => *requested_device_ptr,
        }
    }
}

struct AbiPageableModelRange {
    model_map: usize,
    model_size: u64,
    offset: u64,
    bytes: u64,
    pageable: ReadOnlyPageableHostMemory<'static, u8>,
}

impl AbiPageableModelRange {
    fn device_ptr(
        &self,
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
    ) -> Option<u64> {
        let end = offset.checked_add(bytes)?;
        if self.model_map != model_map as usize
            || self.model_size != model_size
            || offset < self.offset
            || end > self.offset.checked_add(self.bytes)?
        {
            return None;
        }
        self.pageable
            .cu_deviceptr()
            .checked_add(offset - self.offset)
    }
}

struct AbiRegisteredModel {
    model_map: usize,
    model_size: u64,
    _registration: ReadOnlyRegisteredHostMemory<'static, u8>,
    device_ptr: u64,
}

impl AbiRegisteredModel {
    fn matches(&self, model_map: *const c_void, model_size: u64) -> bool {
        self.model_map == model_map as usize && self.model_size == model_size
    }

    fn device_ptr(
        &self,
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
    ) -> Option<u64> {
        let end = offset.checked_add(bytes)?;
        if !self.matches(model_map, model_size) || end > model_size {
            return None;
        }
        self.device_ptr.checked_add(offset)
    }
}

struct AbiCopiedModel {
    model_map: usize,
    model_size: u64,
    copied_bytes: u64,
    device: DeviceBuffer<u8>,
}

impl AbiCopiedModel {
    fn matches(&self, model_map: *const c_void, model_size: u64) -> bool {
        self.model_map == model_map as usize && self.model_size == model_size
    }

    fn device_ptr(
        &self,
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
    ) -> Option<u64> {
        let end = offset.checked_add(bytes)?;
        if !self.matches(model_map, model_size) || end > self.copied_bytes {
            return None;
        }
        self.device.cu_deviceptr().checked_add(offset)
    }
}

struct AbiModelControl {
    model_map: usize,
    model_size: u64,
    model_fd: c_int,
    model_fd_host_base: usize,
    model_file_size: u64,
    model_direct_align: u64,
    #[cfg(target_os = "linux")]
    model_direct_file: Option<std::fs::File>,
}

enum TensorStorage {
    Device(DeviceBuffer<u8>),
    Managed(ManagedBuffer<u8>),
    View(u64),
}

#[repr(C)]
pub struct Ds4GpuTensor {
    storage: TensorStorage,
    bytes: u64,
}

impl Ds4GpuTensor {
    fn device_ptr(&self) -> u64 {
        match &self.storage {
            TensorStorage::Device(buffer) => buffer.cu_deviceptr(),
            TensorStorage::Managed(buffer) => buffer.cu_deviceptr(),
            TensorStorage::View(ptr) => *ptr,
        }
    }
}

fn status(operation: impl FnOnce() -> bool) -> c_int {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(true) => 1,
        Ok(false) | Err(_) => 0,
    }
}

fn pointer<T>(operation: impl FnOnce() -> *mut T) -> *mut T {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(ptr::null_mut())
}

fn with_backend<T>(operation: impl FnOnce(&CudaOxideSubstrate) -> Option<T>) -> Option<T> {
    let backend = BACKEND.lock().ok()?;
    operation(backend.as_ref()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_kernels<T>(
    backend: &CudaOxideSubstrate,
    operation: impl FnOnce(&AbiKernelModule) -> Option<T>,
) -> Option<T> {
    let mut kernels = ABI_KERNELS.lock().ok()?;
    if kernels.is_none() {
        *kernels = Some(AbiKernelModule::load(backend.context()).ok()?);
    }
    operation(kernels.as_ref()?)
}

fn abi_page_bounded_source(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<(&'static [u8], u64, u64)> {
    let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    if page_size == 0 || !page_size.is_power_of_two() {
        return None;
    }
    let model_start = model_map as usize;
    let model_end = model_start.checked_add(usize::try_from(model_size).ok()?)?;
    let start = model_start.checked_add(usize::try_from(offset).ok()?)?;
    let end = start.checked_add(usize::try_from(bytes).ok()?)?;
    let registered_start = start & !(page_size - 1);
    let registered_end = end.checked_add(page_size - 1)? & !(page_size - 1);
    if registered_start < model_start || registered_end > model_end {
        return None;
    }
    // SAFETY: the public C ABI requires the active model mapping to remain
    // readable and immutable until replacement or cleanup. The registered
    // slice is fully contained in that declared mapping and any CUDA guard
    // built from it is retained in ABI state until that completion boundary.
    let source = unsafe {
        std::slice::from_raw_parts(
            registered_start as *const u8,
            registered_end.checked_sub(registered_start)?,
        )
    };
    Some((
        source,
        u64::try_from(registered_start - model_start).ok()?,
        u64::try_from(start - registered_start).ok()?,
    ))
}

fn abi_registered_source(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<(&'static [u8], u64)> {
    let (source, _, device_offset) = abi_page_bounded_source(model_map, model_size, offset, bytes)?;
    Some((source, device_offset))
}

fn pageable_hmm_fallback_selected() -> bool {
    std::env::var_os("DS4_CUDA_COPY_MODEL_CHUNKED").is_some()
        && std::env::var_os("DS4_CUDA_NO_MODEL_PREFETCH").is_none()
        && std::env::var_os("DS4_CUDA_COPY_MODEL").is_none()
        && std::env::var_os("DS4_CUDA_WEIGHT_CACHE").is_none()
        && std::env::var_os("DS4_CUDA_WEIGHT_PRELOAD").is_none()
        && (std::env::var_os("DS4_CUDA_NO_MODEL_COPY").is_some()
            || std::env::var_os("DS4_CUDA_DIRECT_MODEL").is_some())
}

fn pageable_hmm_direct_read_selected() -> bool {
    std::env::var_os("DS4_CUDA_WEIGHT_CACHE").is_none()
        && std::env::var_os("DS4_CUDA_WEIGHT_PRELOAD").is_none()
}

const ABI_MODEL_COPY_CHUNK_BYTES: usize = 64 * 1024 * 1024;

fn try_register_abi_model(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
) -> bool {
    let Ok(model_len) = usize::try_from(model_size) else {
        return false;
    };
    // SAFETY: the public model-map ABI requires the active mapping to remain
    // readable and immutable until synchronized replacement or cleanup; the
    // stored registration guard preserves that CUDA-visible lifetime.
    let source = unsafe { std::slice::from_raw_parts(model_map.cast::<u8>(), model_len) };
    let Ok(registration) = backend.register_read_only_host_range(source) else {
        return false;
    };
    let device_ptr = registration.cu_deviceptr();
    let Ok(mut active) = ABI_REGISTERED_MODEL.lock() else {
        return false;
    };
    *active = Some(AbiRegisteredModel {
        model_map: model_map as usize,
        model_size,
        _registration: registration,
        device_ptr,
    });
    true
}

fn chunk_selected_model_copy_selected() -> bool {
    std::env::var_os("DS4_CUDA_COPY_MODEL_CHUNKED").is_some()
        && std::env::var_os("DS4_CUDA_NO_MODEL_COPY").is_none()
        && std::env::var_os("DS4_CUDA_DIRECT_MODEL").is_none()
        && std::env::var_os("DS4_CUDA_WEIGHT_CACHE").is_none()
        && std::env::var_os("DS4_CUDA_WEIGHT_PRELOAD").is_none()
}

fn full_model_copy_selected() -> bool {
    std::env::var_os("DS4_CUDA_COPY_MODEL").is_some_and(|value| !value.is_empty())
}

fn fd_weight_cache_selected() -> bool {
    std::env::var_os("DS4_CUDA_WEIGHT_CACHE").is_some()
        && std::env::var_os("DS4_CUDA_NO_FD_CACHE").is_none()
        && !std::env::var_os("DS4_CUDA_DIRECT_MODEL").is_some_and(|value| !value.is_empty())
}

fn buffered_fd_weight_cache_selected() -> bool {
    fd_weight_cache_selected() && std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_some()
}

fn direct_io_fd_weight_cache_selected() -> bool {
    fd_weight_cache_selected() && std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_none()
}

fn abi_model_fd(model_map: *const c_void) -> Option<c_int> {
    let control = ABI_MODEL_CONTROL.lock().ok()?;
    if control.model_fd < 0
        || (control.model_fd_host_base != 0 && control.model_fd_host_base != model_map as usize)
    {
        return None;
    }
    Some(control.model_fd)
}

fn upload_abi_buffered_fd_range(
    backend: &CudaOxideSubstrate,
    fd: c_int,
    offset: u64,
    bytes: u64,
) -> Option<DeviceBuffer<u8>> {
    let bytes = usize::try_from(bytes).ok()?;
    let mut staging = backend.pinned_zeroed::<u8>(bytes).ok()?;
    let mut done = 0usize;
    while done < bytes {
        let file_offset = offset.checked_add(u64::try_from(done).ok()?)?;
        let file_offset = libc::off_t::try_from(file_offset).ok()?;
        let result = unsafe {
            libc::pread(
                fd,
                staging.as_mut_slice()[done..].as_mut_ptr().cast(),
                bytes - done,
                file_offset,
            )
        };
        if result < 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return None;
        }
        if result == 0 {
            return None;
        }
        done = done.checked_add(usize::try_from(result).ok()?)?;
    }
    backend.upload_pinned_u8_range(&staging, 0, bytes).ok()
}

fn try_upload_abi_buffered_fd_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    offset: u64,
    bytes: u64,
) -> Option<DeviceBuffer<u8>> {
    if !buffered_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    upload_abi_buffered_fd_range(backend, abi_model_fd(model_map)?, offset, bytes)
}

#[cfg(target_os = "linux")]
fn try_upload_abi_direct_fd_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    offset: u64,
    bytes: u64,
) -> Option<(DeviceBuffer<u8>, bool)> {
    use std::os::unix::fs::FileExt;

    if !direct_io_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    let fd = abi_model_fd(model_map)?;
    let direct_device = (|| -> Option<DeviceBuffer<u8>> {
        let mut control = ABI_MODEL_CONTROL.lock().ok()?;
        let direct_file = control.model_direct_file.as_ref()?;
        let alignment = control.model_direct_align.max(1);
        let read_offset = offset - (offset % alignment);
        let payload_delta = offset.checked_sub(read_offset)?;
        let payload_end = payload_delta.checked_add(bytes)?;
        let read_bytes = payload_end.checked_add(alignment - 1)? / alignment * alignment;
        if control.model_file_size == 0
            || read_offset > control.model_file_size
            || read_bytes > control.model_file_size - read_offset
        {
            None
        } else {
            let stage_bytes = read_bytes.checked_add(alignment)?;
            let stage_bytes = usize::try_from(stage_bytes).ok()?;
            let mut staging = backend.pinned_zeroed::<u8>(stage_bytes).ok()?;
            let alignment = usize::try_from(alignment).ok()?;
            let aligned_delta = (alignment - (staging.as_ptr() as usize % alignment)) % alignment;
            let read_bytes = usize::try_from(read_bytes).ok()?;
            let direct_window =
                &mut staging.as_mut_slice()[aligned_delta..aligned_delta + read_bytes];
            if let Err(error) = direct_file.read_exact_at(direct_window, read_offset) {
                disable_abi_direct_io_after_error(&mut control, error.raw_os_error());
                return None;
            }
            let payload_delta = usize::try_from(payload_delta).ok()?;
            let bytes = usize::try_from(bytes).ok()?;
            backend
                .upload_pinned_u8_range(&staging, aligned_delta + payload_delta, bytes)
                .ok()
        }
    })();
    direct_device.map(|device| (device, true)).or_else(|| {
        upload_abi_buffered_fd_range(backend, fd, offset, bytes).map(|device| (device, false))
    })
}

#[cfg(target_os = "linux")]
fn abi_direct_io_error_disables(raw_os_error: Option<c_int>) -> bool {
    raw_os_error.is_some_and(|raw| {
        [libc::EINVAL, libc::EFAULT, libc::ENOTSUP, libc::EOPNOTSUPP].contains(&raw)
    })
}

#[cfg(target_os = "linux")]
fn disable_abi_direct_io_after_error(
    control: &mut AbiModelControl,
    raw_os_error: Option<c_int>,
) -> bool {
    if !abi_direct_io_error_disables(raw_os_error) {
        return false;
    }
    control.model_direct_file = None;
    control.model_direct_align = 1;
    true
}

#[cfg(not(target_os = "linux"))]
fn try_upload_abi_direct_fd_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    offset: u64,
    bytes: u64,
) -> Option<(DeviceBuffer<u8>, bool)> {
    if !direct_io_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    upload_abi_buffered_fd_range(backend, abi_model_fd(model_map)?, offset, bytes)
        .map(|device| (device, false))
}

fn configure_abi_model_fd(control: &mut AbiModelControl, fd: c_int) {
    control.model_fd = fd;
    control.model_file_size = 0;
    control.model_direct_align = 1;
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

        control.model_direct_file = None;
        if fd < 0 {
            return;
        }
        let direct_path = format!("/proc/self/fd/{fd}");
        if let Some(metadata) = std::fs::metadata(&direct_path)
            .ok()
            .filter(|metadata| metadata.len() > 0)
        {
            control.model_file_size = metadata.len();
            control.model_direct_align = (metadata.blksize() as u64).max(1);
        }
        if std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_none() {
            control.model_direct_file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(direct_path)
                .ok();
            if control.model_direct_file.is_some() {
                control.model_direct_align = control.model_direct_align.max(512);
            }
        }
    }
}

fn try_copy_abi_model_window(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    map_offset: u64,
    map_size: u64,
) -> bool {
    if map_size == 0 || map_offset > model_size || map_size > model_size - map_offset {
        return false;
    }
    if ABI_COPIED_MODEL.lock().ok().is_some_and(|active| {
        active
            .as_ref()
            .is_some_and(|model| model.matches(model_map, model_size))
    }) {
        return true;
    }
    if ABI_REGISTERED_MODEL.lock().ok().is_some_and(|active| {
        active
            .as_ref()
            .is_some_and(|model| model.matches(model_map, model_size))
    }) {
        return true;
    }
    let Some(copied_bytes) = map_offset.checked_add(map_size) else {
        return false;
    };
    let Ok(model_len) = usize::try_from(model_size) else {
        return false;
    };
    let Ok(copied_len) = usize::try_from(copied_bytes) else {
        return false;
    };
    let Ok(device) = backend.zeroed::<u8>(model_len) else {
        return false;
    };
    let Ok(mut staging) = backend.pinned_zeroed::<u8>(copied_len.min(ABI_MODEL_COPY_CHUNK_BYTES))
    else {
        return false;
    };
    let mut copied = 0usize;
    while copied < copied_len {
        let bytes = (copied_len - copied).min(ABI_MODEL_COPY_CHUNK_BYTES);
        // SAFETY: bounds were checked against the active public mapping and
        // each pinned transfer completes before its source buffer is reused.
        let source =
            unsafe { std::slice::from_raw_parts(model_map.cast::<u8>().add(copied), bytes) };
        staging.as_mut_slice()[..bytes].copy_from_slice(source);
        if unsafe { backend.enqueue_pinned_u8_range_async(&device, copied, &staging, 0, bytes) }
            .is_err()
            || backend.synchronize().is_err()
        {
            return false;
        }
        copied += bytes;
    }
    let Ok(mut active) = ABI_COPIED_MODEL.lock() else {
        return false;
    };
    *active = Some(AbiCopiedModel {
        model_map: model_map as usize,
        model_size,
        copied_bytes,
        device,
    });
    true
}

fn try_prefetch_abi_pageable_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> bool {
    let Some((source, range_offset, _)) =
        abi_page_bounded_source(model_map, model_size, offset, bytes)
    else {
        return false;
    };
    if !backend.pageable_memory_access().unwrap_or(false) {
        return false;
    }
    let Ok(pageable) = backend.pageable_read_only_range(source) else {
        return false;
    };
    if backend
        .prefetch_pageable_read_mostly_to_device(&pageable)
        .is_err()
    {
        return false;
    }
    let Ok(mut active) = ABI_PAGEABLE_MODEL_RANGE.lock() else {
        return false;
    };
    *active = Some(AbiPageableModelRange {
        model_map: model_map as usize,
        model_size,
        offset: range_offset,
        bytes: source.len() as u64,
        pageable,
    });
    true
}

fn with_cached_abi_model_range<T>(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    operation: impl FnOnce(u64) -> Option<T>,
) -> Option<T> {
    if model_map.is_null() || offset > model_size || bytes > model_size - offset {
        return None;
    }
    if bytes == 0 {
        let ptr = (model_map as usize).checked_add(usize::try_from(offset).ok()?)?;
        return operation(ptr as u64);
    }
    let registered_model_ptr = ABI_REGISTERED_MODEL
        .lock()
        .ok()?
        .as_ref()
        .and_then(|model| model.device_ptr(model_map, model_size, offset, bytes));
    if let Some(ptr) = registered_model_ptr {
        return operation(ptr);
    }
    let copied_ptr = ABI_COPIED_MODEL
        .lock()
        .ok()?
        .as_ref()
        .and_then(|model| model.device_ptr(model_map, model_size, offset, bytes));
    if let Some(ptr) = copied_ptr {
        return operation(ptr);
    }
    let pageable_ptr = if pageable_hmm_direct_read_selected() {
        ABI_PAGEABLE_MODEL_RANGE
            .lock()
            .ok()?
            .as_ref()
            .and_then(|range| range.device_ptr(model_map, model_size, offset, bytes))
    } else {
        None
    };
    if let Some(ptr) = pageable_ptr {
        return operation(ptr);
    }
    let end = offset.checked_add(bytes)?;
    let mut ranges = ABI_MODEL_RANGES.lock().ok()?;
    if let Some(range) = ranges.iter().find(|range| {
        range.model_map == model_map as usize
            && range.model_size == model_size
            && offset >= range.offset
            && range
                .offset
                .checked_add(range.bytes)
                .is_some_and(|range_end| end <= range_end)
    }) {
        let ptr = range.device_ptr().checked_add(offset - range.offset)?;
        return operation(ptr);
    }
    let offset = usize::try_from(offset).ok()?;
    let bytes = usize::try_from(bytes).ok()?;
    // SAFETY: the public C ABI requires `model_map` to remain readable for
    // `model_size` bytes while this operation executes; bounds were checked
    // above and the asynchronous upload is synchronized before returning.
    let source = unsafe { std::slice::from_raw_parts(model_map.cast::<u8>().add(offset), bytes) };
    let storage =
        match try_upload_abi_direct_fd_range(backend, model_map, offset as u64, bytes as u64) {
            Some((device, true)) => AbiModelRangeStorage::DirectIoFdDeviceCopy(device),
            Some((device, false)) => AbiModelRangeStorage::BufferedFdDeviceCopy(device),
            None => match try_upload_abi_buffered_fd_range(
                backend,
                model_map,
                offset as u64,
                bytes as u64,
            ) {
                Some(device) => AbiModelRangeStorage::BufferedFdDeviceCopy(device),
                None => {
                    match abi_registered_source(model_map, model_size, offset as u64, bytes as u64)
                        .and_then(|(registered_source, device_offset)| {
                            let registration = backend
                                .register_read_only_host_range(registered_source)
                                .ok()?;
                            Some(AbiModelRangeStorage::ReadOnlyRegistered {
                                requested_device_ptr: registration
                                    .cu_deviceptr()
                                    .checked_add(device_offset)?,
                                _registration: registration,
                            })
                        }) {
                        Some(storage) => storage,
                        None => AbiModelRangeStorage::DeviceCopy(backend.upload(source).ok()?),
                    }
                }
            },
        };
    backend.synchronize().ok()?;
    let ptr = match &storage {
        AbiModelRangeStorage::DeviceCopy(device)
        | AbiModelRangeStorage::BufferedFdDeviceCopy(device)
        | AbiModelRangeStorage::DirectIoFdDeviceCopy(device) => device.cu_deviceptr(),
        AbiModelRangeStorage::ReadOnlyRegistered {
            requested_device_ptr,
            ..
        } => *requested_device_ptr,
    };
    ranges.push(AbiModelRange {
        model_map: model_map as usize,
        model_size,
        offset: offset as u64,
        bytes: bytes as u64,
        storage,
    });
    operation(ptr)
}

fn allocation_len(bytes: u64) -> Option<(u64, usize)> {
    let bytes = bytes.max(1);
    Some((bytes, usize::try_from(bytes).ok()?))
}

unsafe fn tensor_ref<'a>(tensor: *const Ds4GpuTensor) -> Option<&'a Ds4GpuTensor> {
    unsafe { tensor.as_ref() }
}

fn checked_range(tensor: &Ds4GpuTensor, offset: u64, bytes: u64) -> Option<(u64, usize)> {
    if offset > tensor.bytes || bytes > tensor.bytes - offset {
        return None;
    }
    let ptr = tensor.device_ptr().checked_add(offset)?;
    Some((ptr, usize::try_from(bytes).ok()?))
}

#[no_mangle]
pub extern "C" fn ds4_gpu_init() -> c_int {
    status(|| {
        let Ok(mut backend) = BACKEND.lock() else {
            return false;
        };
        if backend.is_none() {
            let Ok(opened) = CudaOxideSubstrate::open(0) else {
                return false;
            };
            *backend = Some(opened);
        }
        true
    })
}

#[no_mangle]
pub extern "C" fn ds4_gpu_cleanup() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(mut backend) = BACKEND.lock() {
            if let Some(active) = backend.as_ref() {
                let _ = active.synchronize_device();
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut kernels) = ABI_KERNELS.lock() {
                *kernels = None;
            }
            if let Ok(mut model_ranges) = ABI_MODEL_RANGES.lock() {
                model_ranges.clear();
            }
            if let Ok(mut pageable_range) = ABI_PAGEABLE_MODEL_RANGE.lock() {
                *pageable_range = None;
            }
            if let Ok(mut copied_model) = ABI_COPIED_MODEL.lock() {
                *copied_model = None;
            }
            if let Ok(mut registered_model) = ABI_REGISTERED_MODEL.lock() {
                *registered_model = None;
            }
            if let Ok(mut control) = ABI_MODEL_CONTROL.lock() {
                control.model_map = 0;
                control.model_size = 0;
                configure_abi_model_fd(&mut control, -1);
                control.model_fd_host_base = 0;
            }
            *backend = None;
        }
    }));
}

#[no_mangle]
pub extern "C" fn ds4_gpu_tensor_alloc(bytes: u64) -> *mut Ds4GpuTensor {
    pointer(|| {
        let Some((bytes, len)) = allocation_len(bytes) else {
            return ptr::null_mut();
        };
        with_backend(|backend| {
            backend.context().bind_to_thread().ok()?;
            let ptr = unsafe { cuda_core::memory::malloc_sync(len).ok()? };
            let storage = TensorStorage::Device(unsafe {
                DeviceBuffer::from_raw_parts(ptr, len, backend.context().clone())
            });
            Some(Box::into_raw(Box::new(Ds4GpuTensor { storage, bytes })))
        })
        .unwrap_or(ptr::null_mut())
    })
}

#[no_mangle]
pub extern "C" fn ds4_gpu_tensor_alloc_managed(bytes: u64) -> *mut Ds4GpuTensor {
    pointer(|| {
        let Some((bytes, len)) = allocation_len(bytes) else {
            return ptr::null_mut();
        };
        with_backend(|backend| {
            let storage = TensorStorage::Managed(backend.managed_zeroed::<u8>(len).ok()?);
            Some(Box::into_raw(Box::new(Ds4GpuTensor { storage, bytes })))
        })
        .unwrap_or(ptr::null_mut())
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_view(
    base: *const Ds4GpuTensor,
    offset: u64,
    bytes: u64,
) -> *mut Ds4GpuTensor {
    pointer(|| {
        let Some(base) = (unsafe { tensor_ref(base) }) else {
            return ptr::null_mut();
        };
        let Some((device_ptr, _)) = checked_range(base, offset, bytes) else {
            return ptr::null_mut();
        };
        Box::into_raw(Box::new(Ds4GpuTensor {
            storage: TensorStorage::View(device_ptr),
            bytes,
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_free(tensor: *mut Ds4GpuTensor) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !tensor.is_null() {
            drop(unsafe { Box::from_raw(tensor) });
        }
    }));
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_bytes(tensor: *const Ds4GpuTensor) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        unsafe { tensor_ref(tensor) }.map_or(0, |tensor| tensor.bytes)
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_contents(tensor: *mut Ds4GpuTensor) -> *mut c_void {
    pointer(|| {
        let Some(tensor) = (unsafe { tensor_ref(tensor.cast_const()) }) else {
            return ptr::null_mut();
        };
        let synchronized =
            with_backend(|backend| Some(backend.synchronize_device().is_ok())).unwrap_or(false);
        if !synchronized {
            return ptr::null_mut();
        }
        tensor.device_ptr() as usize as *mut c_void
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_fill_f32(
    tensor: *mut Ds4GpuTensor,
    value: f32,
    count: u64,
) -> c_int {
    status(|| {
        let Some(tensor) = (unsafe { tensor_ref(tensor.cast_const()) }) else {
            return false;
        };
        let Some(bytes) = count.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if bytes > tensor.bytes {
            return false;
        }
        with_backend(|backend| {
            if count == 0 {
                return Some(true);
            }
            backend.context().bind_to_thread().ok()?;
            let len = usize::try_from(count).ok()?;
            // SAFETY: `bytes <= tensor.bytes` above bounds these D32 stores
            // within the CUDA allocation or view owned by the active context.
            let result = unsafe {
                cuda_core::sys::cuMemsetD32Async(
                    tensor.device_ptr(),
                    value.to_bits(),
                    len,
                    backend.stream().cu_stream(),
                )
            }
            .result();
            Some(result.is_ok())
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_add_tensor(
    out: *mut Ds4GpuTensor,
    a: *const Ds4GpuTensor,
    b: *const Ds4GpuTensor,
    count: u32,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(a) = (unsafe { tensor_ref(a) }) else {
            return false;
        };
        let Some(b) = (unsafe { tensor_ref(b) }) else {
            return false;
        };
        let bytes = u64::from(count) * size_of::<f32>() as u64;
        if out.bytes < bytes || a.bytes < bytes || b.bytes < bytes {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: bounds above cover each device pointer; raw launch
                // preserves current-C support for input/output aliasing.
                Some(unsafe {
                    kernels.add_tensor(
                        backend.stream(),
                        out.device_ptr(),
                        a.device_ptr(),
                        b.device_ptr(),
                        count,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_repeat_hc_tensor(
    out: *mut Ds4GpuTensor,
    row: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(row) = (unsafe { tensor_ref(row) }) else {
            return false;
        };
        let Some(count) = u64::from(n_embd).checked_mul(u64::from(n_hc)) else {
            return false;
        };
        let Some(out_bytes) = count.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let row_bytes = u64::from(n_embd) * size_of::<f32>() as u64;
        if n_embd == 0 || n_hc == 0 || out.bytes < out_bytes || row.bytes < row_bytes {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: bounds above cover each device pointer; raw launch
                // preserves current-C support for input/output aliasing.
                Some(unsafe {
                    kernels.repeat_hc_tensor(
                        backend.stream(),
                        out.device_ptr(),
                        row.device_ptr(),
                        n_embd,
                        n_hc,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_directional_steering_project_tensor(
    x: *mut Ds4GpuTensor,
    directions: *const Ds4GpuTensor,
    layer: u32,
    width: u32,
    rows: u32,
    scale: f32,
) -> c_int {
    status(|| {
        let Some(x) = (unsafe { tensor_ref(x.cast_const()) }) else {
            return false;
        };
        let Some(directions) = (unsafe { tensor_ref(directions) }) else {
            return false;
        };
        let Some(x_elements) = u64::from(width).checked_mul(u64::from(rows)) else {
            return false;
        };
        let Some(direction_elements) = u64::from(layer)
            .checked_add(1)
            .and_then(|layers| layers.checked_mul(u64::from(width)))
        else {
            return false;
        };
        let Some(x_bytes) = x_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(direction_bytes) = direction_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if width == 0
            || rows == 0
            || scale == 0.0
            || x.bytes < x_bytes
            || directions.bytes < direction_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: bounds above cover each device pointer; raw launch
                // preserves the current-C in-place tensor boundary.
                Some(unsafe {
                    kernels.directional_steering_project_tensor(
                        backend.stream(),
                        x.device_ptr(),
                        directions.device_ptr(),
                        layer,
                        width,
                        rows,
                        scale,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_swiglu_tensor(
    out: *mut Ds4GpuTensor,
    gate: *const Ds4GpuTensor,
    up: *const Ds4GpuTensor,
    count: u32,
    clamp: f32,
    weight: f32,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(gate) = (unsafe { tensor_ref(gate) }) else {
            return false;
        };
        let Some(up) = (unsafe { tensor_ref(up) }) else {
            return false;
        };
        let bytes = u64::from(count) * size_of::<f32>() as u64;
        if count == 0 || out.bytes < bytes || gate.bytes < bytes || up.bytes < bytes {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: bounds above cover each device pointer; raw launch
                // preserves current-C support for output/input aliasing.
                Some(unsafe {
                    kernels.swiglu_tensor(
                        backend.stream(),
                        out.device_ptr(),
                        gate.device_ptr(),
                        up.device_ptr(),
                        count,
                        clamp,
                        weight,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
unsafe fn rms_norm_plain_rows_impl(
    out: *mut Ds4GpuTensor,
    x: *const Ds4GpuTensor,
    n: u32,
    rows: u32,
    eps: f32,
) -> bool {
    let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
        return false;
    };
    let Some(x) = (unsafe { tensor_ref(x) }) else {
        return false;
    };
    let Some(count) = u64::from(n).checked_mul(u64::from(rows)) else {
        return false;
    };
    let Some(bytes) = count.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    if rows == 0 || out.bytes < bytes || x.bytes < bytes {
        return false;
    }
    with_backend(|backend| {
        with_abi_kernels(backend, |kernels| {
            // SAFETY: bounds above cover each device pointer; all source
            // elements are reduced before row-local stores, preserving
            // current-C support for output/input aliasing.
            Some(unsafe {
                kernels.rms_norm_plain_rows_tensor(
                    backend.stream(),
                    out.device_ptr(),
                    x.device_ptr(),
                    n,
                    rows,
                    eps,
                )
            })
        })
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_rms_norm_plain_tensor(
    out: *mut Ds4GpuTensor,
    x: *const Ds4GpuTensor,
    n: u32,
    eps: f32,
) -> c_int {
    status(|| unsafe { rms_norm_plain_rows_impl(out, x, n, 1, eps) })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_rms_norm_plain_rows_tensor(
    out: *mut Ds4GpuTensor,
    x: *const Ds4GpuTensor,
    n: u32,
    rows: u32,
    eps: f32,
) -> c_int {
    status(|| unsafe { rms_norm_plain_rows_impl(out, x, n, rows, eps) })
}

#[cfg(feature = "cuda-oxide-kernels")]
unsafe fn rms_norm_weight_rows_impl(
    out: *mut Ds4GpuTensor,
    x: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    rows: u32,
    eps: f32,
) -> bool {
    let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
        return false;
    };
    let Some(x) = (unsafe { tensor_ref(x) }) else {
        return false;
    };
    let Some(count) = u64::from(n).checked_mul(u64::from(rows)) else {
        return false;
    };
    let Some(bytes) = count.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let weight_bytes = u64::from(n) * size_of::<f32>() as u64;
    if model_map.is_null()
        || rows == 0
        || weight_offset > model_size
        || weight_bytes > model_size - weight_offset
        || out.bytes < bytes
        || x.bytes < bytes
    {
        return false;
    }
    with_backend(|backend| {
        with_cached_abi_model_range(
            backend,
            model_map,
            model_size,
            weight_offset,
            weight_bytes,
            |weight_ptr| {
                with_abi_kernels(backend, |kernels| {
                    // SAFETY: tensor and model ranges above cover every
                    // device pointer; row reductions complete before stores,
                    // preserving current-C output/input aliasing.
                    Some(unsafe {
                        kernels.rms_norm_weight_rows_tensor(
                            backend.stream(),
                            out.device_ptr(),
                            x.device_ptr(),
                            weight_ptr,
                            n,
                            rows,
                            eps,
                        )
                    })
                })
            },
        )
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_rms_norm_weight_tensor(
    out: *mut Ds4GpuTensor,
    x: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    eps: f32,
) -> c_int {
    status(|| unsafe {
        rms_norm_weight_rows_impl(out, x, model_map, model_size, weight_offset, n, 1, eps)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_rms_norm_weight_rows_tensor(
    out: *mut Ds4GpuTensor,
    x: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n: u32,
    rows: u32,
    eps: f32,
) -> c_int {
    status(|| unsafe {
        rms_norm_weight_rows_impl(out, x, model_map, model_size, weight_offset, n, rows, eps)
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_write(
    tensor: *mut Ds4GpuTensor,
    offset: u64,
    data: *const c_void,
    bytes: u64,
) -> c_int {
    status(|| {
        let Some(tensor) = (unsafe { tensor_ref(tensor.cast_const()) }) else {
            return false;
        };
        if data.is_null() {
            return false;
        }
        let Some((dst, bytes)) = checked_range(tensor, offset, bytes) else {
            return false;
        };
        with_backend(|backend| {
            if bytes == 0 {
                return Some(true);
            }
            let result = unsafe {
                cuda_core::memory::memcpy_htod_async(
                    dst,
                    data.cast::<u8>(),
                    bytes,
                    backend.stream().cu_stream(),
                )
            };
            Some(result.is_ok() && backend.synchronize().is_ok())
        })
        .unwrap_or(false)
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_read(
    tensor: *const Ds4GpuTensor,
    offset: u64,
    data: *mut c_void,
    bytes: u64,
) -> c_int {
    status(|| {
        let Some(tensor) = (unsafe { tensor_ref(tensor) }) else {
            return false;
        };
        if data.is_null() {
            return false;
        }
        let Some((src, bytes)) = checked_range(tensor, offset, bytes) else {
            return false;
        };
        with_backend(|backend| {
            if bytes == 0 {
                return Some(true);
            }
            let result = unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    data.cast::<u8>(),
                    src,
                    bytes,
                    backend.stream().cu_stream(),
                )
            };
            Some(result.is_ok() && backend.synchronize().is_ok())
        })
        .unwrap_or(false)
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_tensor_copy(
    dst: *mut Ds4GpuTensor,
    dst_offset: u64,
    src: *const Ds4GpuTensor,
    src_offset: u64,
    bytes: u64,
) -> c_int {
    status(|| {
        let Some(dst) = (unsafe { tensor_ref(dst.cast_const()) }) else {
            return false;
        };
        let Some(src) = (unsafe { tensor_ref(src) }) else {
            return false;
        };
        let Some((dst_ptr, bytes)) = checked_range(dst, dst_offset, bytes) else {
            return false;
        };
        let Some((src_ptr, _)) = checked_range(src, src_offset, bytes as u64) else {
            return false;
        };
        with_backend(|backend| {
            if bytes == 0 {
                return Some(true);
            }
            let result = unsafe {
                cuda_core::memory::memcpy_dtod_async(
                    dst_ptr,
                    src_ptr,
                    bytes,
                    backend.stream().cu_stream(),
                )
            };
            Some(result.is_ok() && backend.synchronize().is_ok())
        })
        .unwrap_or(false)
    })
}

#[no_mangle]
pub extern "C" fn ds4_gpu_begin_commands() -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn ds4_gpu_flush_commands() -> c_int {
    status(|| with_backend(|backend| Some(backend.flush_commands().is_ok())).unwrap_or(false))
}

#[no_mangle]
pub extern "C" fn ds4_gpu_end_commands() -> c_int {
    status(|| with_backend(|backend| Some(backend.end_commands().is_ok())).unwrap_or(false))
}

#[no_mangle]
pub extern "C" fn ds4_gpu_synchronize() -> c_int {
    status(|| with_backend(|backend| Some(backend.synchronize_device().is_ok())).unwrap_or(false))
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_set_model_map(model_map: *const c_void, model_size: u64) -> c_int {
    status(|| {
        if model_map.is_null() || model_size == 0 {
            return false;
        }
        with_backend(|backend| {
            let mut control = ABI_MODEL_CONTROL.lock().ok()?;
            if control.model_map == model_map as usize && control.model_size == model_size {
                return Some(true);
            }
            backend.synchronize().ok()?;
            ABI_MODEL_RANGES.lock().ok()?.clear();
            *ABI_PAGEABLE_MODEL_RANGE.lock().ok()? = None;
            *ABI_COPIED_MODEL.lock().ok()? = None;
            *ABI_REGISTERED_MODEL.lock().ok()? = None;
            control.model_map = model_map as usize;
            control.model_size = model_size;
            if control.model_fd >= 0 && control.model_fd_host_base == 0 {
                control.model_fd_host_base = model_map as usize;
            }
            if !full_model_copy_selected() {
                let _ = try_register_abi_model(backend, model_map, model_size);
            }
            Some(true)
        })
        .unwrap_or(false)
    })
}

#[no_mangle]
pub extern "C" fn ds4_gpu_set_model_fd(fd: c_int) -> c_int {
    status(|| {
        let Ok(mut control) = ABI_MODEL_CONTROL.lock() else {
            return false;
        };
        configure_abi_model_fd(&mut control, fd);
        control.model_fd_host_base = control.model_map;
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_set_model_map_range(
    model_map: *const c_void,
    model_size: u64,
    map_offset: u64,
    map_size: u64,
) -> c_int {
    if unsafe { ds4_gpu_set_model_map(model_map, model_size) } == 0 {
        return 0;
    }
    if chunk_selected_model_copy_selected()
        && with_backend(|backend| {
            Some(try_copy_abi_model_window(
                backend, model_map, model_size, map_offset, map_size,
            ))
        })
        .unwrap_or(false)
    {
        return 1;
    }
    if pageable_hmm_fallback_selected() {
        let _ = with_backend(|backend| {
            let _ = try_prefetch_abi_pageable_range(
                backend, model_map, model_size, map_offset, map_size,
            );
            Some(())
        });
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_cache_model_range(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    _label: *const c_char,
) -> c_int {
    status(|| {
        if bytes == 0 {
            return true;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(backend, model_map, model_size, offset, bytes, |_| {
                Some(true)
            })
        })
        .unwrap_or(false)
    })
}

#[no_mangle]
pub extern "C" fn ds4_gpu_should_use_managed_kv_cache(
    kv_cache_bytes: u64,
    context_bytes: u64,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        let memory = with_backend(|backend| backend.memory_capacity().ok());
        c_int::from(managed_kv_decision(kv_cache_bytes, context_bytes, memory).use_managed)
    }))
    .unwrap_or(0)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{abi_direct_io_error_disables, disable_abi_direct_io_after_error, AbiModelControl};

    #[test]
    fn public_direct_io_disable_error_classes_match_current_c_policy() {
        for raw_os_error in [libc::EINVAL, libc::EFAULT, libc::ENOTSUP, libc::EOPNOTSUPP] {
            assert!(abi_direct_io_error_disables(Some(raw_os_error)));
        }
        assert!(!abi_direct_io_error_disables(Some(libc::EIO)));
        assert!(!abi_direct_io_error_disables(None));

        let mut qualifying = AbiModelControl {
            model_map: 0,
            model_size: 0,
            model_fd: -1,
            model_fd_host_base: 0,
            model_file_size: 0,
            model_direct_align: 4096,
            model_direct_file: None,
        };
        assert!(disable_abi_direct_io_after_error(
            &mut qualifying,
            Some(libc::EINVAL)
        ));
        assert_eq!(qualifying.model_direct_align, 1);

        let mut non_qualifying = AbiModelControl {
            model_direct_align: 4096,
            ..qualifying
        };
        assert!(!disable_abi_direct_io_after_error(
            &mut non_qualifying,
            Some(libc::EIO)
        ));
        assert_eq!(non_qualifying.model_direct_align, 4096);
    }
}
