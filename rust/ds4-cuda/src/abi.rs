use std::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use cuda_core::{
    CudaEvent, DeviceBuffer, IntoResult, ManagedBuffer, PinnedHostBuffer,
    ReadOnlyPageableHostMemory, ReadOnlyRegisteredHostMemory,
};

#[cfg(feature = "cuda-oxide-kernels")]
use crate::abi_kernels::AbiKernelModule;
use crate::allocation_policy::managed_kv_decision;
use crate::substrate::CudaOxideSubstrate;

static BACKEND: Mutex<Option<CudaOxideSubstrate>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_KERNELS: Mutex<Option<AbiKernelModule>> = Mutex::new(None);
static ABI_MODEL_RANGES: Mutex<Vec<AbiModelRange>> = Mutex::new(Vec::new());
#[cfg(target_os = "linux")]
static ABI_MODEL_ARENAS: Mutex<AbiModelArenaState> = Mutex::new(AbiModelArenaState {
    arenas: Vec::new(),
    range_bytes: 0,
    progress: AbiModelLoadProgress {
        next_bytes: 0,
        last: None,
        started: false,
        tty: false,
    },
});
static ABI_PAGEABLE_MODEL_RANGE: Mutex<Option<AbiPageableModelRange>> = Mutex::new(None);
static ABI_COPIED_MODEL: Mutex<Option<AbiCopiedModel>> = Mutex::new(None);
static ABI_REGISTERED_MODEL: Mutex<Option<AbiRegisteredModel>> = Mutex::new(None);
static ABI_MODEL_RANGE_MAPPING_SUPPORTED: AtomicBool = AtomicBool::new(true);
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

#[cfg(target_os = "linux")]
struct AbiModelArena {
    device: DeviceBuffer<u8>,
    bytes: u64,
    used: u64,
}

#[cfg(target_os = "linux")]
struct AbiModelArenaState {
    arenas: Vec<AbiModelArena>,
    range_bytes: u64,
    progress: AbiModelLoadProgress,
}

#[cfg(target_os = "linux")]
struct AbiModelLoadProgress {
    next_bytes: u64,
    last: Option<std::time::Instant>,
    started: bool,
    tty: bool,
}

#[cfg(target_os = "linux")]
impl AbiModelLoadProgress {
    fn reset(&mut self) {
        self.next_bytes = 0;
        self.last = None;
        self.started = false;
        self.tty = false;
    }
}

enum AbiModelRangeStorage {
    DeviceCopy(DeviceBuffer<u8>),
    #[cfg(not(target_os = "linux"))]
    BufferedFdDeviceCopy(DeviceBuffer<u8>),
    #[cfg(not(target_os = "linux"))]
    DirectIoFdDeviceCopy(DeviceBuffer<u8>),
    #[cfg(target_os = "linux")]
    BufferedFdArenaDeviceCopy {
        requested_device_ptr: u64,
    },
    #[cfg(target_os = "linux")]
    DirectIoFdArenaDeviceCopy {
        requested_device_ptr: u64,
    },
    ReadOnlyRegistered {
        _registration: ReadOnlyRegisteredHostMemory<'static, u8>,
        requested_device_ptr: u64,
    },
}

enum AbiFdRangeResolution {
    Cached(AbiModelRangeStorage),
    BudgetFallback { requested_device_ptr: u64 },
}

impl AbiModelRange {
    fn device_ptr(&self) -> u64 {
        match &self.storage {
            AbiModelRangeStorage::DeviceCopy(buffer) => buffer.cu_deviceptr(),
            #[cfg(not(target_os = "linux"))]
            AbiModelRangeStorage::BufferedFdDeviceCopy(buffer)
            | AbiModelRangeStorage::DirectIoFdDeviceCopy(buffer) => buffer.cu_deviceptr(),
            #[cfg(target_os = "linux")]
            AbiModelRangeStorage::BufferedFdArenaDeviceCopy {
                requested_device_ptr,
            }
            | AbiModelRangeStorage::DirectIoFdArenaDeviceCopy {
                requested_device_ptr,
            } => *requested_device_ptr,
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

fn abi_range_registration_disables(error: &cuda_core::DriverError) -> bool {
    [
        cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
        cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
    ]
    .contains(&error.0)
}

fn try_register_abi_model_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<AbiModelRangeStorage> {
    if !ABI_MODEL_RANGE_MAPPING_SUPPORTED.load(Ordering::Relaxed) {
        return None;
    }
    let (registered_source, device_offset) =
        abi_registered_source(model_map, model_size, offset, bytes)?;
    match backend.register_read_only_host_range(registered_source) {
        Ok(registration) => Some(AbiModelRangeStorage::ReadOnlyRegistered {
            requested_device_ptr: registration.cu_deviceptr().checked_add(device_offset)?,
            _registration: registration,
        }),
        Err(error) => {
            if abi_range_registration_disables(&error) {
                ABI_MODEL_RANGE_MAPPING_SUPPORTED.store(false, Ordering::Relaxed);
            }
            None
        }
    }
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
const ABI_DIRECT_FD_STAGE_SLOTS: usize = 4;

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

fn abi_model_copy_chunk_bytes_from_value(value: Option<&std::ffi::CStr>) -> Option<usize> {
    let mut mb = 64_u64;
    if let Some(value) = value {
        let mut end = ptr::null_mut();
        let parsed = unsafe { libc::strtoull(value.as_ptr(), &mut end, 10) };
        if end.cast_const() != value.as_ptr() && parsed > 0 {
            mb = parsed;
        }
    }
    let bytes = mb.clamp(16, 4096).checked_mul(1024 * 1024)?;
    usize::try_from(bytes).ok()
}

fn abi_model_copy_chunk_bytes() -> Option<usize> {
    let value = std::env::var("DS4_CUDA_MODEL_COPY_CHUNK_MB")
        .ok()
        .and_then(|value| std::ffi::CString::new(value).ok());
    abi_model_copy_chunk_bytes_from_value(value.as_deref())
}

#[cfg(target_os = "linux")]
fn abi_model_arena_chunk_bytes_from_value(
    value: Option<&std::ffi::CStr>,
    need: usize,
) -> Option<usize> {
    const MIB: u64 = 1024 * 1024;
    const ROUNDING_BYTES: u64 = 256 * MIB;

    let mut mb = 1792_u64;
    if let Some(value) = value {
        let mut end = ptr::null_mut();
        let parsed = unsafe { libc::strtoull(value.as_ptr(), &mut end, 10) };
        if end.cast_const() != value.as_ptr() && parsed > 0 {
            mb = parsed;
        }
    }
    let mut bytes = mb.clamp(256, 8192).checked_mul(MIB)?;
    let need = u64::try_from(need).ok()?;
    if bytes < need {
        bytes = need
            .checked_add(ROUNDING_BYTES - 1)?
            .checked_div(ROUNDING_BYTES)?
            .checked_mul(ROUNDING_BYTES)?;
    }
    usize::try_from(bytes).ok()
}

#[cfg(target_os = "linux")]
fn abi_model_arena_chunk_bytes(need: usize) -> Option<usize> {
    let value = std::env::var("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB")
        .ok()
        .and_then(|value| std::ffi::CString::new(value).ok());
    abi_model_arena_chunk_bytes_from_value(value.as_deref(), need)
}

#[cfg(target_os = "linux")]
fn abi_model_cache_limit_bytes_from_value(value: Option<&std::ffi::CStr>) -> Option<u64> {
    let mut gb = 0_u64;
    if let Some(value) = value {
        let mut end = ptr::null_mut();
        let parsed = unsafe { libc::strtoull(value.as_ptr(), &mut end, 10) };
        if end.cast_const() != value.as_ptr() {
            gb = parsed;
        }
    }
    if gb == 0 {
        Some(u64::MAX)
    } else {
        gb.checked_mul(1024 * 1024 * 1024)
    }
}

#[cfg(target_os = "linux")]
fn abi_model_cache_limit_bytes() -> Option<u64> {
    let value = std::env::var("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB")
        .ok()
        .and_then(|value| std::ffi::CString::new(value).ok());
    abi_model_cache_limit_bytes_from_value(value.as_deref())
}

#[cfg(target_os = "linux")]
fn abi_model_discard_source_pages(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<()> {
    if std::env::var_os("DS4_CUDA_KEEP_MODEL_PAGES").is_some()
        || model_map.is_null()
        || bytes == 0
        || offset > model_size
    {
        return Some(());
    }
    let bytes = bytes.min(model_size.checked_sub(offset)?);
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page_size)
        .ok()
        .filter(|size| *size > 0)
        .unwrap_or(4096);
    let host_start = (model_map as usize).checked_add(usize::try_from(offset).ok()?)?;
    let host_end = host_start.checked_add(usize::try_from(bytes).ok()?)?;
    let page_size = usize::try_from(page_size).ok()?;
    let page_start = host_start / page_size * page_size;
    let page_end = host_end
        .checked_add(page_size.checked_sub(1)?)?
        .checked_div(page_size)?
        .checked_mul(page_size)?;
    if page_end > page_start {
        let _ = unsafe {
            libc::posix_madvise(
                page_start as *mut c_void,
                page_end - page_start,
                libc::POSIX_MADV_DONTNEED,
            )
        };
    }
    Some(())
}

#[cfg(target_os = "linux")]
fn abi_model_drop_file_pages(fd: c_int, offset: u64, bytes: u64) -> Option<()> {
    if std::env::var_os("DS4_CUDA_KEEP_MODEL_PAGES").is_some() || fd < 0 || bytes == 0 {
        return Some(());
    }
    let offset = libc::off_t::try_from(offset).ok()?;
    let bytes = libc::off_t::try_from(bytes).ok()?;
    let _ = unsafe { libc::posix_fadvise(fd, offset, bytes, libc::POSIX_FADV_DONTNEED) };
    Some(())
}

#[cfg(target_os = "linux")]
fn abi_model_load_progress_note(progress: &mut AbiModelLoadProgress, cached_bytes: u64) {
    use std::io::Write as _;

    if std::env::var_os("DS4_CUDA_WEIGHT_CACHE_VERBOSE").is_some() {
        return;
    }
    let now = std::time::Instant::now();
    if !progress.started {
        progress.started = true;
        progress.tty = unsafe { libc::isatty(libc::STDERR_FILENO) != 0 };
        progress.next_bytes = if progress.tty {
            2_u64 << 30
        } else {
            16_u64 << 30
        };
        progress.last = Some(now);
        if progress.tty {
            eprint!("ds4: CUDA loading model tensors into device cache: 0.00 GiB");
        } else {
            eprintln!("ds4: CUDA loading model tensors into device cache");
        }
    }
    let interval = if progress.tty {
        std::time::Duration::from_secs(2)
    } else {
        std::time::Duration::from_secs(10)
    };
    if cached_bytes < progress.next_bytes
        && progress
            .last
            .is_some_and(|last| now.duration_since(last) < interval)
    {
        return;
    }
    if progress.tty {
        eprint!(
            "\rds4: CUDA loading model tensors into device cache: {:.2} GiB",
            cached_bytes as f64 / 1073741824.0
        );
    } else {
        eprintln!(
            "ds4: CUDA loading model tensors {:.2} GiB cached",
            cached_bytes as f64 / 1073741824.0
        );
    }
    let _ = std::io::stderr().flush();
    progress.last = Some(now);
    let step = if progress.tty {
        2_u64 << 30
    } else {
        16_u64 << 30
    };
    while progress.next_bytes <= cached_bytes {
        progress.next_bytes = progress.next_bytes.saturating_add(step);
    }
}

#[cfg(target_os = "linux")]
fn align_abi_model_arena_bytes(bytes: u64) -> Option<u64> {
    const ALIGNMENT: u64 = 256;
    bytes
        .checked_add(ALIGNMENT - 1)?
        .checked_div(ALIGNMENT)?
        .checked_mul(ALIGNMENT)
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

fn abi_model_ptr(model_map: *const c_void, offset: u64) -> Option<u64> {
    let ptr = (model_map as usize).checked_add(usize::try_from(offset).ok()?)?;
    u64::try_from(ptr).ok()
}

fn read_abi_buffered_fd_into(fd: c_int, offset: u64, destination: &mut [u8]) -> Option<()> {
    let mut done = 0usize;
    while done < destination.len() {
        let file_offset = offset.checked_add(u64::try_from(done).ok()?)?;
        let file_offset = libc::off_t::try_from(file_offset).ok()?;
        let result = unsafe {
            libc::pread(
                fd,
                destination[done..].as_mut_ptr().cast(),
                destination.len() - done,
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
    Some(())
}

#[cfg(not(target_os = "linux"))]
fn upload_abi_buffered_fd_range(
    backend: &CudaOxideSubstrate,
    fd: c_int,
    offset: u64,
    bytes: u64,
) -> Option<DeviceBuffer<u8>> {
    let bytes = usize::try_from(bytes).ok()?;
    let mut staging = backend.pinned_zeroed::<u8>(bytes).ok()?;
    read_abi_buffered_fd_into(fd, offset, staging.as_mut_slice())?;
    backend.upload_pinned_u8_range(&staging, 0, bytes).ok()
}

#[cfg(target_os = "linux")]
fn try_upload_abi_buffered_fd_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<AbiFdRangeResolution> {
    if !buffered_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    match upload_abi_async_fd_arena_range(
        backend,
        abi_model_fd(model_map)?,
        model_map,
        model_size,
        offset,
        bytes,
        false,
    )? {
        AbiFdArenaUpload::Uploaded {
            requested_device_ptr,
            ..
        } => Some(AbiFdRangeResolution::Cached(
            AbiModelRangeStorage::BufferedFdArenaDeviceCopy {
                requested_device_ptr,
            },
        )),
        AbiFdArenaUpload::BudgetFallback => Some(AbiFdRangeResolution::BudgetFallback {
            requested_device_ptr: abi_model_ptr(model_map, offset)?,
        }),
    }
}

#[cfg(not(target_os = "linux"))]
fn try_upload_abi_buffered_fd_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    _model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<AbiFdRangeResolution> {
    if !buffered_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    upload_abi_buffered_fd_range(backend, abi_model_fd(model_map)?, offset, bytes).map(|device| {
        AbiFdRangeResolution::Cached(AbiModelRangeStorage::BufferedFdDeviceCopy(device))
    })
}

#[cfg(target_os = "linux")]
enum AbiFdArenaUpload {
    Uploaded {
        requested_device_ptr: u64,
        used_direct_io: bool,
    },
    BudgetFallback,
}

#[cfg(target_os = "linux")]
fn read_abi_direct_or_buffered_fd_stage(
    fd: c_int,
    staging: &mut PinnedHostBuffer<u8>,
    offset: u64,
    bytes: usize,
) -> Option<(usize, bool)> {
    use std::os::unix::fs::FileExt;

    let direct_payload = (|| -> Option<usize> {
        let mut control = ABI_MODEL_CONTROL.lock().ok()?;
        let direct_file = control.model_direct_file.as_ref()?;
        let alignment = control.model_direct_align.max(1);
        let read_offset = offset - (offset % alignment);
        let payload_delta = offset.checked_sub(read_offset)?;
        let payload_end = payload_delta.checked_add(u64::try_from(bytes).ok()?)?;
        let read_bytes = payload_end.checked_add(alignment - 1)? / alignment * alignment;
        if control.model_file_size == 0
            || read_offset > control.model_file_size
            || read_bytes > control.model_file_size - read_offset
        {
            None
        } else {
            let alignment = usize::try_from(alignment).ok()?;
            let aligned_delta = (alignment - (staging.as_ptr() as usize % alignment)) % alignment;
            let read_bytes = usize::try_from(read_bytes).ok()?;
            if read_bytes > staging.len().checked_sub(aligned_delta)? {
                return None;
            }
            let direct_window =
                &mut staging.as_mut_slice()[aligned_delta..aligned_delta + read_bytes];
            if let Err(error) = direct_file.read_exact_at(direct_window, read_offset) {
                disable_abi_direct_io_after_error(&mut control, error.raw_os_error());
                return None;
            }
            let payload_delta = usize::try_from(payload_delta).ok()?;
            Some(aligned_delta + payload_delta)
        }
    })();
    if let Some(payload) = direct_payload {
        return Some((payload, true));
    }
    read_abi_buffered_fd_into(fd, offset, &mut staging.as_mut_slice()[..bytes])?;
    Some((0, false))
}

#[cfg(target_os = "linux")]
fn upload_abi_async_fd_range_into(
    backend: &CudaOxideSubstrate,
    device: &DeviceBuffer<u8>,
    device_offset: usize,
    fd: c_int,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    use_direct_io: bool,
    cached_bytes_before: u64,
    progress: &mut AbiModelLoadProgress,
) -> Option<bool> {
    let bytes = usize::try_from(bytes).ok()?;
    let chunk_bytes = abi_model_copy_chunk_bytes()?;
    let alignment = ABI_MODEL_CONTROL.lock().ok()?.model_direct_align.max(1);
    let alignment = usize::try_from(alignment).ok()?;
    let stage_bytes = chunk_bytes.checked_add(alignment.checked_mul(2)?)?;
    let mut staging = Vec::with_capacity(ABI_DIRECT_FD_STAGE_SLOTS);
    let mut events: Vec<Option<CudaEvent>> = Vec::with_capacity(ABI_DIRECT_FD_STAGE_SLOTS);
    for _ in 0..ABI_DIRECT_FD_STAGE_SLOTS {
        staging.push(backend.pinned_zeroed::<u8>(stage_bytes).ok()?);
        events.push(None);
    }
    let upload_result = (|| -> Option<bool> {
        let mut copied = 0usize;
        let mut chunk_index = 0usize;
        let mut used_direct = false;
        while copied < bytes {
            let this_chunk = (bytes - copied).min(chunk_bytes);
            let slot = chunk_index % ABI_DIRECT_FD_STAGE_SLOTS;
            if let Some(event) = events[slot].take() {
                event.synchronize().ok()?;
            }
            let file_offset = offset.checked_add(u64::try_from(copied).ok()?)?;
            let (payload_offset, direct) = if use_direct_io {
                read_abi_direct_or_buffered_fd_stage(
                    fd,
                    &mut staging[slot],
                    file_offset,
                    this_chunk,
                )?
            } else {
                read_abi_buffered_fd_into(
                    fd,
                    file_offset,
                    &mut staging[slot].as_mut_slice()[..this_chunk],
                )?;
                (0, false)
            };
            unsafe {
                backend
                    .enqueue_pinned_u8_range_async(
                        device,
                        device_offset.checked_add(copied)?,
                        &staging[slot],
                        payload_offset,
                        this_chunk,
                    )
                    .ok()?;
            }
            events[slot] = Some(backend.record_event().ok()?);
            let this_chunk = u64::try_from(this_chunk).ok()?;
            abi_model_drop_file_pages(fd, file_offset, this_chunk)?;
            abi_model_discard_source_pages(model_map, model_size, file_offset, this_chunk)?;
            used_direct |= direct;
            copied = copied.checked_add(usize::try_from(this_chunk).ok()?)?;
            abi_model_load_progress_note(
                progress,
                cached_bytes_before.checked_add(u64::try_from(copied).ok()?)?,
            );
            chunk_index = chunk_index.checked_add(1)?;
        }
        Some(used_direct)
    })();
    let synchronize_ok = backend.synchronize().is_ok();
    let used_direct = upload_result?;
    if !synchronize_ok {
        return None;
    }
    Some(used_direct)
}

#[cfg(target_os = "linux")]
fn upload_abi_async_fd_arena_range(
    backend: &CudaOxideSubstrate,
    fd: c_int,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    use_direct_io: bool,
) -> Option<AbiFdArenaUpload> {
    let aligned_bytes = align_abi_model_arena_bytes(bytes)?;
    let limit = abi_model_cache_limit_bytes()?;
    let mut state = ABI_MODEL_ARENAS.lock().ok()?;
    if state.range_bytes > limit || bytes > limit - state.range_bytes {
        return Some(AbiFdArenaUpload::BudgetFallback);
    }
    if aligned_bytes > limit - state.range_bytes {
        return Some(AbiFdArenaUpload::BudgetFallback);
    }
    let reservation = state.arenas.iter().enumerate().find_map(|(index, arena)| {
        let used = align_abi_model_arena_bytes(arena.used)?;
        (used <= arena.bytes && aligned_bytes <= arena.bytes - used).then_some((index, used))
    });
    let (arena_index, device_offset) = match reservation {
        Some((index, used)) => {
            state.arenas[index].used = used + aligned_bytes;
            (index, used)
        }
        None => {
            let chunk_bytes = abi_model_arena_chunk_bytes(usize::try_from(bytes).ok()?)?;
            let device = backend.zeroed::<u8>(chunk_bytes).ok()?;
            state.arenas.push(AbiModelArena {
                device,
                bytes: u64::try_from(chunk_bytes).ok()?,
                used: aligned_bytes,
            });
            (state.arenas.len().checked_sub(1)?, 0)
        }
    };
    let cached_bytes_before = state.range_bytes;
    let AbiModelArenaState {
        arenas, progress, ..
    } = &mut *state;
    let arena = arenas.get(arena_index)?;
    let requested_device_ptr = arena.device.cu_deviceptr().checked_add(device_offset)?;
    let used_direct_io = upload_abi_async_fd_range_into(
        backend,
        &arena.device,
        usize::try_from(device_offset).ok()?,
        fd,
        model_map,
        model_size,
        offset,
        bytes,
        use_direct_io,
        cached_bytes_before,
        progress,
    )?;
    state.range_bytes = state.range_bytes.checked_add(bytes)?;
    let range_bytes = state.range_bytes;
    abi_model_load_progress_note(&mut state.progress, range_bytes);
    Some(AbiFdArenaUpload::Uploaded {
        requested_device_ptr,
        used_direct_io,
    })
}

#[cfg(target_os = "linux")]
fn try_upload_abi_direct_fd_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<AbiFdRangeResolution> {
    if !direct_io_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    match upload_abi_async_fd_arena_range(
        backend,
        abi_model_fd(model_map)?,
        model_map,
        model_size,
        offset,
        bytes,
        true,
    )? {
        AbiFdArenaUpload::Uploaded {
            requested_device_ptr,
            used_direct_io,
        } => Some(AbiFdRangeResolution::Cached(if used_direct_io {
            AbiModelRangeStorage::DirectIoFdArenaDeviceCopy {
                requested_device_ptr,
            }
        } else {
            AbiModelRangeStorage::BufferedFdArenaDeviceCopy {
                requested_device_ptr,
            }
        })),
        AbiFdArenaUpload::BudgetFallback => Some(AbiFdRangeResolution::BudgetFallback {
            requested_device_ptr: abi_model_ptr(model_map, offset)?,
        }),
    }
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
    _model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<AbiFdRangeResolution> {
    if !direct_io_fd_weight_cache_selected() || bytes == 0 {
        return None;
    }
    upload_abi_buffered_fd_range(backend, abi_model_fd(model_map)?, offset, bytes).map(|device| {
        AbiFdRangeResolution::Cached(AbiModelRangeStorage::BufferedFdDeviceCopy(device))
    })
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
    let storage = match try_upload_abi_direct_fd_range(
        backend,
        model_map,
        model_size,
        offset as u64,
        bytes as u64,
    ) {
        Some(AbiFdRangeResolution::Cached(storage)) => storage,
        Some(AbiFdRangeResolution::BudgetFallback {
            requested_device_ptr,
        }) => return operation(requested_device_ptr),
        None => match try_upload_abi_buffered_fd_range(
            backend,
            model_map,
            model_size,
            offset as u64,
            bytes as u64,
        ) {
            Some(AbiFdRangeResolution::Cached(storage)) => storage,
            Some(AbiFdRangeResolution::BudgetFallback {
                requested_device_ptr,
            }) => return operation(requested_device_ptr),
            None => {
                match try_register_abi_model_range(
                    backend,
                    model_map,
                    model_size,
                    offset as u64,
                    bytes as u64,
                ) {
                    Some(storage) => storage,
                    None => {
                        // SAFETY: the public C ABI requires `model_map` to remain readable
                        // for `model_size` bytes while this copy executes; bounds were
                        // checked above and the upload is synchronized before returning.
                        let source = unsafe {
                            std::slice::from_raw_parts(model_map.cast::<u8>().add(offset), bytes)
                        };
                        AbiModelRangeStorage::DeviceCopy(backend.upload(source).ok()?)
                    }
                }
            }
        },
    };
    backend.synchronize().ok()?;
    let ptr = match &storage {
        AbiModelRangeStorage::DeviceCopy(device) => device.cu_deviceptr(),
        #[cfg(not(target_os = "linux"))]
        AbiModelRangeStorage::BufferedFdDeviceCopy(device)
        | AbiModelRangeStorage::DirectIoFdDeviceCopy(device) => device.cu_deviceptr(),
        #[cfg(target_os = "linux")]
        AbiModelRangeStorage::BufferedFdArenaDeviceCopy {
            requested_device_ptr,
        }
        | AbiModelRangeStorage::DirectIoFdArenaDeviceCopy {
            requested_device_ptr,
        } => *requested_device_ptr,
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
            #[cfg(target_os = "linux")]
            if let Ok(mut model_arenas) = ABI_MODEL_ARENAS.lock() {
                model_arenas.arenas.clear();
                model_arenas.range_bytes = 0;
                model_arenas.progress.reset();
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
                ABI_MODEL_RANGE_MAPPING_SUPPORTED.store(true, Ordering::Relaxed);
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
            #[cfg(target_os = "linux")]
            {
                let mut model_arenas = ABI_MODEL_ARENAS.lock().ok()?;
                model_arenas.arenas.clear();
                model_arenas.range_bytes = 0;
                model_arenas.progress.reset();
            }
            *ABI_PAGEABLE_MODEL_RANGE.lock().ok()? = None;
            *ABI_COPIED_MODEL.lock().ok()? = None;
            *ABI_REGISTERED_MODEL.lock().ok()? = None;
            control.model_map = model_map as usize;
            control.model_size = model_size;
            ABI_MODEL_RANGE_MAPPING_SUPPORTED.store(true, Ordering::Relaxed);
            if control.model_fd >= 0 && control.model_fd_host_base == 0 {
                control.model_fd_host_base = model_map as usize;
            }
            if full_model_copy_selected()
                && try_copy_abi_model_window(backend, model_map, model_size, 0, model_size)
            {
                return Some(true);
            }
            let _ = try_register_abi_model(backend, model_map, model_size);
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
    use std::ffi::CString;

    use super::{
        abi_direct_io_error_disables, abi_model_arena_chunk_bytes_from_value,
        abi_model_cache_limit_bytes_from_value, abi_model_copy_chunk_bytes_from_value,
        abi_range_registration_disables, disable_abi_direct_io_after_error, AbiModelControl,
    };

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

    #[test]
    fn public_range_registration_disable_errors_match_current_c_policy() {
        for raw in [
            cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
            cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
        ] {
            assert!(abi_range_registration_disables(&cuda_core::DriverError(
                raw
            )));
        }
        assert!(!abi_range_registration_disables(&cuda_core::DriverError(
            cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY,
        )));
    }

    #[test]
    fn public_direct_io_async_chunk_override_matches_current_c_clamp() {
        let small = CString::new("8").expect("valid chunk string");
        let trailing = CString::new("32rest").expect("valid chunk string");
        let huge = CString::new("5000").expect("valid chunk string");
        let zero = CString::new("0").expect("valid chunk string");

        assert_eq!(
            abi_model_copy_chunk_bytes_from_value(Some(&small)),
            Some(16 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_copy_chunk_bytes_from_value(Some(&trailing)),
            Some(32 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_copy_chunk_bytes_from_value(Some(&huge)),
            Some(4096 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_copy_chunk_bytes_from_value(Some(&zero)),
            Some(64 * 1024 * 1024)
        );
    }

    #[test]
    fn public_fd_arena_chunk_override_matches_current_c_clamp_and_growth() {
        let small = CString::new("8").expect("valid arena string");
        let trailing = CString::new("512rest").expect("valid arena string");
        let huge = CString::new("9000").expect("valid arena string");
        let zero = CString::new("0").expect("valid arena string");

        assert_eq!(
            abi_model_arena_chunk_bytes_from_value(Some(&small), 1),
            Some(256 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_arena_chunk_bytes_from_value(Some(&trailing), 1),
            Some(512 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_arena_chunk_bytes_from_value(Some(&huge), 1),
            Some(8192 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_arena_chunk_bytes_from_value(Some(&zero), 1),
            Some(1792 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_arena_chunk_bytes_from_value(Some(&small), 300 * 1024 * 1024),
            Some(512 * 1024 * 1024)
        );
    }

    #[test]
    fn public_fd_cache_limit_override_matches_current_c_gib_policy() {
        let one = CString::new("1").expect("valid cache limit string");
        let trailing = CString::new("2rest").expect("valid cache limit string");
        let zero = CString::new("0").expect("valid cache limit string");
        let invalid = CString::new("off").expect("valid cache limit string");

        assert_eq!(
            abi_model_cache_limit_bytes_from_value(Some(&one)),
            Some(1024 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_cache_limit_bytes_from_value(Some(&trailing)),
            Some(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            abi_model_cache_limit_bytes_from_value(Some(&zero)),
            Some(u64::MAX)
        );
        assert_eq!(
            abi_model_cache_limit_bytes_from_value(Some(&invalid)),
            Some(u64::MAX)
        );
    }
}
