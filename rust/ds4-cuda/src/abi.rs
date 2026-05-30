use std::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use cuda_core::{DeviceBuffer, IntoResult, ManagedBuffer, ReadOnlyRegisteredHostMemory};

#[cfg(feature = "cuda-oxide-kernels")]
use crate::abi_kernels::AbiKernelModule;
use crate::allocation_policy::managed_kv_decision;
use crate::substrate::CudaOxideSubstrate;

static BACKEND: Mutex<Option<CudaOxideSubstrate>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_KERNELS: Mutex<Option<AbiKernelModule>> = Mutex::new(None);
static ABI_MODEL_RANGES: Mutex<Vec<AbiModelRange>> = Mutex::new(Vec::new());
static ABI_MODEL_CONTROL: Mutex<AbiModelControl> = Mutex::new(AbiModelControl {
    model_map: 0,
    model_size: 0,
    _model_fd: -1,
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
    ReadOnlyRegistered {
        _registration: ReadOnlyRegisteredHostMemory<'static, u8>,
        requested_device_ptr: u64,
    },
}

impl AbiModelRange {
    fn device_ptr(&self) -> u64 {
        match &self.storage {
            AbiModelRangeStorage::DeviceCopy(buffer) => buffer.cu_deviceptr(),
            AbiModelRangeStorage::ReadOnlyRegistered {
                requested_device_ptr,
                ..
            } => *requested_device_ptr,
        }
    }
}

struct AbiModelControl {
    model_map: usize,
    model_size: u64,
    _model_fd: c_int,
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

fn abi_registered_source(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> Option<(&'static [u8], u64)> {
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
    // slice is fully contained in that declared mapping and its guard is
    // retained in ABI_MODEL_RANGES until that completion boundary.
    let source = unsafe {
        std::slice::from_raw_parts(
            registered_start as *const u8,
            registered_end.checked_sub(registered_start)?,
        )
    };
    Some((source, u64::try_from(start - registered_start).ok()?))
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
    let registered_storage =
        abi_registered_source(model_map, model_size, offset as u64, bytes as u64).and_then(
            |(registered_source, device_offset)| {
                let registration = backend
                    .register_read_only_host_range(registered_source)
                    .ok()?;
                Some(AbiModelRangeStorage::ReadOnlyRegistered {
                    requested_device_ptr: registration.cu_deviceptr().checked_add(device_offset)?,
                    _registration: registration,
                })
            },
        );
    let storage = match registered_storage {
        Some(storage) => storage,
        None => AbiModelRangeStorage::DeviceCopy(backend.upload(source).ok()?),
    };
    backend.synchronize().ok()?;
    let ptr = match &storage {
        AbiModelRangeStorage::DeviceCopy(device) => device.cu_deviceptr(),
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
            if let Ok(mut control) = ABI_MODEL_CONTROL.lock() {
                control.model_map = 0;
                control.model_size = 0;
                control._model_fd = -1;
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
            control.model_map = model_map as usize;
            control.model_size = model_size;
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
        control._model_fd = fd;
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
    let _ = (map_offset, map_size);
    // The baseline C path records the mapping here; prefetch and residency
    // selection remain separate policy work.
    unsafe { ds4_gpu_set_model_map(model_map, model_size) }
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
