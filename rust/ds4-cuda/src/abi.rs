use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use cuda_core::{
    BlasMathMode, CudaEvent, DeviceBuffer, IntoResult, ManagedBuffer, PinnedHostBuffer,
    ProjectionConfig, ReadOnlyPageableHostMemory, ReadOnlyRegisteredHostMemory,
    StridedBatchedSgemmConfig,
};

#[cfg(feature = "cuda-oxide-kernels")]
use crate::abi_kernels::{AbiKernelModule, ABI_MOE_IQ2_GRID, ABI_MOE_IQ2_SIGNS};
use crate::allocation_policy::managed_kv_decision;
use crate::q8_policy::{
    q8_f32_cache_allowed, q8_preload_format, Q8CacheOptions, Q8CacheState, Q8PreloadFormat,
};
use crate::substrate::CudaOxideSubstrate;
#[cfg(feature = "cuda-oxide-kernels")]
use crate::{
    q8_dp4a_enabled, select_attention_output_a_path, select_attention_prefill_path,
    select_f16_pair_projection_path, select_f16_projection_path, select_f32_projection_path,
    select_q8_matmul_path, select_router_select_path, AttentionOutputADispatchOptions,
    AttentionOutputAPath, AttentionPrefillDispatchOptions, AttentionPrefillPath,
    F16PairProjectionDispatch, F16PairProjectionPath, F16ProjectionDispatch,
    Q8MatmulDispatchOptions, Q8MatmulPath, RouterSelectDispatchOptions, RouterSelectPath,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP, DS4_CUDA_ATTENTION_SCORE_CAP,
};

static BACKEND: Mutex<Option<CudaOxideSubstrate>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_KERNELS: Mutex<Option<AbiKernelModule>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_F16_ACTIVATIONS: Mutex<Option<DeviceBuffer<f16>>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_Q8_ACTIVATIONS: Mutex<Option<AbiQ8Activations>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_MOE_IQ2_TABLES: Mutex<Option<AbiMoeIq2Tables>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_ROUTED_MOE_BATCH_SCRATCH: Mutex<Option<AbiRoutedMoeBatchScratch>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_INDEXED_TOPK_SORT_SCRATCH: Mutex<Option<DeviceBuffer<i32>>> = Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_ATTENTION_OUTPUT_CUBLAS_SCRATCH: Mutex<Option<AbiAttentionOutputCublasScratch>> =
    Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_ATTENTION_PREFILL_CUBLAS_SCRATCH: Mutex<Option<AbiAttentionPrefillCublasScratch>> =
    Mutex::new(None);
#[cfg(feature = "cuda-oxide-kernels")]
static ABI_Q8_CACHE: Mutex<AbiQ8Cache> = Mutex::new(AbiQ8Cache {
    f16_ranges: Vec::new(),
    f32_ranges: Vec::new(),
    state: Q8CacheState {
        f16_cached_bytes: 0,
        f16_disabled_after_failure: false,
        f16_budget_notice_printed: false,
        optional_preload_disabled: false,
    },
});
static ABI_MODEL_RANGES: Mutex<Vec<AbiModelRange>> = Mutex::new(Vec::new());
#[cfg(target_os = "linux")]
static ABI_MODEL_ARENAS: Mutex<AbiModelArenaState> = Mutex::new(AbiModelArenaState {
    arenas: Vec::new(),
    range_bytes: 0,
    cache_full: false,
    progress: AbiModelLoadProgress {
        next_bytes: 0,
        last: None,
        started: false,
        tty: false,
    },
});
#[cfg(target_os = "linux")]
static ABI_MODEL_STAGE_POOL: Mutex<AbiModelStagePool> = Mutex::new(AbiModelStagePool {
    slots: Vec::new(),
    stage_bytes: 0,
});
static ABI_PAGEABLE_MODEL_RANGE: Mutex<Option<AbiPageableModelRange>> = Mutex::new(None);
static ABI_COPIED_MODEL: Mutex<Option<AbiCopiedModel>> = Mutex::new(None);
static ABI_REGISTERED_MODEL: Mutex<Option<AbiRegisteredModel>> = Mutex::new(None);
static ABI_MODEL_RANGE_MAPPING_SUPPORTED: AtomicBool = AtomicBool::new(true);
static ABI_QUALITY_MODE: AtomicBool = AtomicBool::new(false);
static ABI_DEFAULT_BLAS_MATH: AtomicBool = AtomicBool::new(false);
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

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiQ8F16Range {
    model_map: usize,
    offset: u64,
    weight_bytes: u64,
    in_dim: u64,
    out_dim: u64,
    device: DeviceBuffer<f16>,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiQ8F32Range {
    model_map: usize,
    offset: u64,
    weight_bytes: u64,
    in_dim: u64,
    out_dim: u64,
    device: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiQ8Cache {
    f16_ranges: Vec<AbiQ8F16Range>,
    f32_ranges: Vec<AbiQ8F32Range>,
    state: Q8CacheState,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiQ8Activations {
    quantized: DeviceBuffer<i8>,
    scales: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiMoeIq2Tables {
    grid: DeviceBuffer<u64>,
    signs: DeviceBuffer<u8>,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiRoutedMoeBatchScratch {
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    cursors: DeviceBuffer<u32>,
    sorted_pairs: DeviceBuffer<u32>,
    tile_offsets: DeviceBuffer<u32>,
    tile_total: DeviceBuffer<u32>,
    tile_experts: DeviceBuffer<u32>,
    tile_starts: DeviceBuffer<u32>,
    tile16_offsets: DeviceBuffer<u32>,
    tile16_total: DeviceBuffer<u32>,
    tile16_experts: DeviceBuffer<u32>,
    tile16_starts: DeviceBuffer<u32>,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiAttentionOutputCublasScratch {
    packed_heads: DeviceBuffer<f32>,
    transposed_weights: DeviceBuffer<f32>,
    packed_low: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda-oxide-kernels")]
struct AbiAttentionPrefillCublasScratch {
    kv: DeviceBuffer<f32>,
    q_heads: DeviceBuffer<f32>,
    keys: DeviceBuffer<f32>,
    keys_transposed: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    output_by_head: DeviceBuffer<f32>,
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
    cache_full: bool,
    progress: AbiModelLoadProgress,
}

#[cfg(target_os = "linux")]
struct AbiModelStageSlot {
    staging: PinnedHostBuffer<u8>,
    event: Option<CudaEvent>,
}

#[cfg(target_os = "linux")]
struct AbiModelStagePool {
    slots: Vec<AbiModelStageSlot>,
    stage_bytes: usize,
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

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_f16_activations<T>(
    backend: &CudaOxideSubstrate,
    elements: usize,
    operation: impl FnOnce(&mut DeviceBuffer<f16>) -> Option<T>,
) -> Option<T> {
    if elements == 0 {
        return None;
    }
    let mut activations = ABI_F16_ACTIVATIONS.lock().ok()?;
    if activations
        .as_ref()
        .map_or(true, |current| current.len() < elements)
    {
        if activations.is_some() {
            backend.synchronize().ok()?;
        }
        let bytes = elements.checked_mul(size_of::<f16>())?;
        backend.context().bind_to_thread().ok()?;
        let ptr = unsafe { cuda_core::memory::malloc_sync(bytes).ok()? };
        // SAFETY: malloc_sync allocates exactly this typed span in the
        // active context; DeviceBuffer drop pairs it with free_sync.
        *activations = Some(unsafe {
            DeviceBuffer::<f16>::from_raw_parts(ptr, elements, backend.context().clone())
        });
    }
    operation(activations.as_mut()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_q8_activations<T>(
    backend: &CudaOxideSubstrate,
    quantized_elements: usize,
    scale_elements: usize,
    operation: impl FnOnce(&mut AbiQ8Activations) -> Option<T>,
) -> Option<T> {
    if quantized_elements == 0 || scale_elements == 0 {
        return None;
    }
    let mut activations = ABI_Q8_ACTIVATIONS.lock().ok()?;
    if activations.as_ref().is_none_or(|current| {
        current.quantized.len() < quantized_elements || current.scales.len() < scale_elements
    }) {
        if activations.is_some() {
            backend.synchronize().ok()?;
        }
        *activations = Some(AbiQ8Activations {
            quantized: backend.zeroed::<i8>(quantized_elements).ok()?,
            scales: backend.zeroed::<f32>(scale_elements).ok()?,
        });
    }
    operation(activations.as_mut()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_moe_iq2_tables<T>(
    backend: &CudaOxideSubstrate,
    operation: impl FnOnce(&AbiMoeIq2Tables) -> Option<T>,
) -> Option<T> {
    let mut tables = ABI_MOE_IQ2_TABLES.lock().ok()?;
    if tables.is_none() {
        *tables = Some(AbiMoeIq2Tables {
            grid: backend.upload(&ABI_MOE_IQ2_GRID).ok()?,
            signs: backend.upload(&ABI_MOE_IQ2_SIGNS).ok()?,
        });
    }
    operation(tables.as_ref()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_routed_moe_batch_scratch<T>(
    backend: &CudaOxideSubstrate,
    pair_count: usize,
    tile_capacity: usize,
    tile16_capacity: usize,
    operation: impl FnOnce(&mut AbiRoutedMoeBatchScratch) -> Option<T>,
) -> Option<T> {
    if pair_count == 0 || tile_capacity == 0 || tile16_capacity == 0 {
        return None;
    }
    let mut scratch = ABI_ROUTED_MOE_BATCH_SCRATCH.lock().ok()?;
    if scratch.as_ref().is_none_or(|current| {
        current.sorted_pairs.len() < pair_count
            || current.tile_experts.len() < tile_capacity
            || current.tile_starts.len() < tile_capacity
            || current.tile16_experts.len() < tile16_capacity
            || current.tile16_starts.len() < tile16_capacity
    }) {
        if scratch.is_some() {
            backend.synchronize().ok()?;
        }
        *scratch = Some(AbiRoutedMoeBatchScratch {
            counts: backend.zeroed::<u32>(256).ok()?,
            offsets: backend.zeroed::<u32>(257).ok()?,
            cursors: backend.zeroed::<u32>(256).ok()?,
            sorted_pairs: backend.zeroed::<u32>(pair_count).ok()?,
            tile_offsets: backend.zeroed::<u32>(257).ok()?,
            tile_total: backend.zeroed::<u32>(1).ok()?,
            tile_experts: backend.zeroed::<u32>(tile_capacity).ok()?,
            tile_starts: backend.zeroed::<u32>(tile_capacity).ok()?,
            tile16_offsets: backend.zeroed::<u32>(257).ok()?,
            tile16_total: backend.zeroed::<u32>(1).ok()?,
            tile16_experts: backend.zeroed::<u32>(tile16_capacity).ok()?,
            tile16_starts: backend.zeroed::<u32>(tile16_capacity).ok()?,
        });
    }
    operation(scratch.as_mut()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
fn clear_abi_routed_moe_counts(
    backend: &CudaOxideSubstrate,
    counts: &DeviceBuffer<u32>,
) -> Option<()> {
    backend.context().bind_to_thread().ok()?;
    // SAFETY: `counts` is a 256-element device buffer in the active context
    // and the clear is enqueued on the same stream as the consuming kernels.
    unsafe {
        cuda_core::sys::cuMemsetD32Async(
            counts.cu_deviceptr(),
            0,
            counts.len(),
            backend.stream().cu_stream(),
        )
    }
    .result()
    .ok()
}

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_attention_output_cublas_scratch<T>(
    backend: &CudaOxideSubstrate,
    packed_head_elements: usize,
    transposed_weight_elements: usize,
    packed_low_elements: usize,
    operation: impl FnOnce(&mut AbiAttentionOutputCublasScratch) -> Option<T>,
) -> Option<T> {
    if packed_head_elements == 0 || transposed_weight_elements == 0 || packed_low_elements == 0 {
        return None;
    }
    let mut scratch = ABI_ATTENTION_OUTPUT_CUBLAS_SCRATCH.lock().ok()?;
    if scratch.as_ref().is_none_or(|current| {
        current.packed_heads.len() < packed_head_elements
            || current.transposed_weights.len() < transposed_weight_elements
            || current.packed_low.len() < packed_low_elements
    }) {
        if scratch.is_some() {
            backend.synchronize().ok()?;
        }
        *scratch = Some(AbiAttentionOutputCublasScratch {
            packed_heads: backend.zeroed::<f32>(packed_head_elements).ok()?,
            transposed_weights: backend.zeroed::<f32>(transposed_weight_elements).ok()?,
            packed_low: backend.zeroed::<f32>(packed_low_elements).ok()?,
        });
    }
    operation(scratch.as_mut()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
fn with_abi_attention_prefill_cublas_scratch<T>(
    backend: &CudaOxideSubstrate,
    kv_elements: usize,
    output_elements: usize,
    head_kv_elements: usize,
    score_elements: usize,
    operation: impl FnOnce(&mut AbiAttentionPrefillCublasScratch) -> Option<T>,
) -> Option<T> {
    if kv_elements == 0 || output_elements == 0 || head_kv_elements == 0 || score_elements == 0 {
        return None;
    }
    let mut scratch = ABI_ATTENTION_PREFILL_CUBLAS_SCRATCH.lock().ok()?;
    if scratch.as_ref().is_none_or(|current| {
        current.kv.len() < kv_elements
            || current.q_heads.len() < output_elements
            || current.keys.len() < head_kv_elements
            || current.keys_transposed.len() < head_kv_elements
            || current.scores.len() < score_elements
            || current.output_by_head.len() < output_elements
    }) {
        if scratch.is_some() {
            backend.synchronize().ok()?;
        }
        *scratch = Some(AbiAttentionPrefillCublasScratch {
            kv: backend.zeroed::<f32>(kv_elements).ok()?,
            q_heads: backend.zeroed::<f32>(output_elements).ok()?,
            keys: backend.zeroed::<f32>(head_kv_elements).ok()?,
            keys_transposed: backend.zeroed::<f32>(head_kv_elements).ok()?,
            scores: backend.zeroed::<f32>(score_elements).ok()?,
            output_by_head: backend.zeroed::<f32>(output_elements).ok()?,
        });
    }
    operation(scratch.as_mut()?)
}

#[cfg(feature = "cuda-oxide-kernels")]
fn with_abi_indexed_topk_sort_scratch<T>(
    backend: &CudaOxideSubstrate,
    elements: usize,
    operation: impl FnOnce(&mut DeviceBuffer<i32>) -> Option<T>,
) -> Option<T> {
    if elements == 0 {
        return None;
    }
    let mut scratch = ABI_INDEXED_TOPK_SORT_SCRATCH.lock().ok()?;
    if scratch
        .as_ref()
        .is_none_or(|current| current.len() < elements)
    {
        if scratch.is_some() {
            backend.synchronize().ok()?;
        }
        *scratch = Some(backend.zeroed::<i32>(elements).ok()?);
    }
    operation(scratch.as_mut()?)
}

fn abi_no_tf32_selected() -> bool {
    std::env::var_os("DS4_CUDA_NO_TF32").is_some()
}

fn update_abi_blas_math_state() {
    ABI_DEFAULT_BLAS_MATH.store(
        ABI_QUALITY_MODE.load(Ordering::Relaxed) || abi_no_tf32_selected(),
        Ordering::Relaxed,
    );
}

fn apply_abi_blas_math(blas: &cuda_core::Blas) -> bool {
    let mode = if ABI_DEFAULT_BLAS_MATH.load(Ordering::Relaxed) {
        BlasMathMode::Default
    } else {
        BlasMathMode::Tf32TensorOp
    };
    blas.set_math_mode(mode).is_ok()
}

#[cfg(feature = "cuda-oxide-kernels")]
fn abi_mib_env(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?.parse::<u64>().ok()?;
    Some(value.checked_mul(1024 * 1024).unwrap_or(u64::MAX))
}

#[cfg(feature = "cuda-oxide-kernels")]
fn abi_q8_cache_options() -> Q8CacheOptions {
    Q8CacheOptions {
        quality_mode: ABI_QUALITY_MODE.load(Ordering::Relaxed),
        no_q8_f16_cache: std::env::var_os("DS4_CUDA_NO_Q8_F16_CACHE").is_some(),
        q8_f16_all: std::env::var_os("DS4_CUDA_Q8_F16_ALL").is_some(),
        no_attention_output_f16_cache: std::env::var_os("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE")
            .is_some(),
        no_attn_q_b_f16_cache: std::env::var_os("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE").is_some(),
        attention_output_preload: std::env::var_os("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD").is_some(),
        q8_f16_limit_bytes: abi_mib_env("DS4_CUDA_Q8_F16_CACHE_MB"),
        q8_f16_reserve_bytes: abi_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB"),
        no_q8_f32_cache: std::env::var_os("DS4_CUDA_NO_Q8_F32_CACHE").is_some(),
        q8_f32_all: std::env::var_os("DS4_CUDA_Q8_F32_ALL").is_some(),
        attn_q_b_f32_cache: std::env::var_os("DS4_CUDA_ATTN_Q_B_F32_CACHE").is_some(),
        q8_f32_large: std::env::var_os("DS4_CUDA_Q8_F32_LARGE").is_some(),
        q8_f32_preload: std::env::var_os("DS4_CUDA_Q8_F32_PRELOAD").is_some(),
        weight_cache_verbose: std::env::var_os("DS4_CUDA_WEIGHT_CACHE_VERBOSE").is_some(),
    }
}

#[cfg(feature = "cuda-oxide-kernels")]
fn clear_abi_q8_converted_ranges(cache: &mut AbiQ8Cache) {
    cache.f16_ranges.clear();
    cache.f32_ranges.clear();
    let optional_preload_disabled = cache.state.optional_preload_disabled;
    cache.state = Q8CacheState {
        optional_preload_disabled,
        ..Q8CacheState::default()
    };
}

#[cfg(feature = "cuda-oxide-kernels")]
fn q8_range_matches<T>(
    ranges: &[T],
    matches: impl Fn(&T) -> (usize, u64, u64, u64, u64),
    model_map: *const c_void,
    offset: u64,
    weight_bytes: u64,
    in_dim: u64,
    out_dim: u64,
) -> bool {
    ranges
        .iter()
        .any(|range| matches(range) == (model_map as usize, offset, weight_bytes, in_dim, out_dim))
}

#[cfg(feature = "cuda-oxide-kernels")]
fn abi_q8_f16_ptr(
    model_map: *const c_void,
    offset: u64,
    weight_bytes: u64,
    in_dim: u64,
    out_dim: u64,
) -> Option<u64> {
    ABI_Q8_CACHE
        .lock()
        .ok()?
        .f16_ranges
        .iter()
        .find_map(|range| {
            ((
                range.model_map,
                range.offset,
                range.weight_bytes,
                range.in_dim,
                range.out_dim,
            ) == (model_map as usize, offset, weight_bytes, in_dim, out_dim))
                .then(|| range.device.cu_deviceptr())
        })
}

#[cfg(feature = "cuda-oxide-kernels")]
fn abi_q8_f32_ptr(
    model_map: *const c_void,
    offset: u64,
    weight_bytes: u64,
    in_dim: u64,
    out_dim: u64,
) -> Option<u64> {
    ABI_Q8_CACHE
        .lock()
        .ok()?
        .f32_ranges
        .iter()
        .find_map(|range| {
            ((
                range.model_map,
                range.offset,
                range.weight_bytes,
                range.in_dim,
                range.out_dim,
            ) == (model_map as usize, offset, weight_bytes, in_dim, out_dim))
                .then(|| range.device.cu_deviceptr())
        })
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

fn direct_model_read_selected() -> bool {
    std::env::var_os("DS4_CUDA_DIRECT_MODEL").is_some_and(|value| !value.is_empty())
}

fn fd_weight_cache_selected() -> bool {
    std::env::var_os("DS4_CUDA_NO_FD_CACHE").is_none() && !direct_model_read_selected()
}

fn buffered_fd_weight_cache_selected() -> bool {
    fd_weight_cache_selected() && std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_some()
}

fn direct_io_fd_weight_cache_selected() -> bool {
    fd_weight_cache_selected() && std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_none()
}

fn strict_fd_weight_cache_selected() -> bool {
    std::env::var_os("DS4_CUDA_STRICT_WEIGHT_CACHE").is_some()
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
        AbiFdArenaUpload::ArenaFallback => {
            if strict_fd_weight_cache_selected() {
                None
            } else {
                Some(AbiFdRangeResolution::BudgetFallback {
                    requested_device_ptr: abi_model_ptr(model_map, offset)?,
                })
            }
        }
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
    ArenaFallback,
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
    let mut stage_pool = ABI_MODEL_STAGE_POOL.lock().ok()?;
    if stage_pool.stage_bytes < stage_bytes {
        if !stage_pool.slots.is_empty() {
            backend.synchronize().ok()?;
        }
        stage_pool.slots.clear();
        stage_pool.stage_bytes = 0;
        for _ in 0..ABI_DIRECT_FD_STAGE_SLOTS {
            stage_pool.slots.push(AbiModelStageSlot {
                staging: backend.pinned_zeroed::<u8>(stage_bytes).ok()?,
                event: None,
            });
        }
        stage_pool.stage_bytes = stage_bytes;
    }
    let upload_result = (|| -> Option<bool> {
        let mut copied = 0usize;
        let mut chunk_index = 0usize;
        let mut used_direct = false;
        while copied < bytes {
            let this_chunk = (bytes - copied).min(chunk_bytes);
            let slot = chunk_index % ABI_DIRECT_FD_STAGE_SLOTS;
            let stage_slot = stage_pool.slots.get_mut(slot)?;
            if let Some(event) = stage_slot.event.take() {
                event.synchronize().ok()?;
            }
            let file_offset = offset.checked_add(u64::try_from(copied).ok()?)?;
            let (payload_offset, direct) = if use_direct_io {
                read_abi_direct_or_buffered_fd_stage(
                    fd,
                    &mut stage_slot.staging,
                    file_offset,
                    this_chunk,
                )?
            } else {
                read_abi_buffered_fd_into(
                    fd,
                    file_offset,
                    &mut stage_slot.staging.as_mut_slice()[..this_chunk],
                )?;
                (0, false)
            };
            unsafe {
                backend
                    .enqueue_pinned_u8_range_async(
                        device,
                        device_offset.checked_add(copied)?,
                        &stage_slot.staging,
                        payload_offset,
                        this_chunk,
                    )
                    .ok()?;
            }
            stage_slot.event = Some(backend.record_event().ok()?);
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
    if synchronize_ok {
        for slot in &mut stage_pool.slots {
            slot.event = None;
        }
    }
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
    if state.cache_full {
        return Some(AbiFdArenaUpload::ArenaFallback);
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
            if aligned_bytes > limit - state.range_bytes {
                return Some(AbiFdArenaUpload::ArenaFallback);
            }
            let chunk_bytes = abi_model_arena_chunk_bytes(usize::try_from(bytes).ok()?)?;
            let device = match backend.allocate_u8(chunk_bytes) {
                Ok(device) => device,
                Err(_) => {
                    state.cache_full = true;
                    return Some(AbiFdArenaUpload::ArenaFallback);
                }
            };
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
        AbiFdArenaUpload::ArenaFallback => {
            if strict_fd_weight_cache_selected() {
                None
            } else {
                Some(AbiFdRangeResolution::BudgetFallback {
                    requested_device_ptr: abi_model_ptr(model_map, offset)?,
                })
            }
        }
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
    if direct_model_read_selected() {
        let ptr = (model_map as usize).checked_add(usize::try_from(offset).ok()?)?;
        return operation(ptr as u64);
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
        drop(ranges);
        return operation(ptr);
    }
    let offset = usize::try_from(offset).ok()?;
    let bytes = usize::try_from(bytes).ok()?;
    let fd_resolution = if direct_io_fd_weight_cache_selected() {
        try_upload_abi_direct_fd_range(backend, model_map, model_size, offset as u64, bytes as u64)
    } else {
        try_upload_abi_buffered_fd_range(
            backend,
            model_map,
            model_size,
            offset as u64,
            bytes as u64,
        )
    };
    let storage = match fd_resolution {
        Some(AbiFdRangeResolution::Cached(storage)) => storage,
        Some(AbiFdRangeResolution::BudgetFallback {
            requested_device_ptr,
        }) => {
            drop(ranges);
            return operation(requested_device_ptr);
        }
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
    drop(ranges);
    operation(ptr)
}

fn abi_model_range_is_cached(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
) -> bool {
    if model_map.is_null() || offset > model_size || bytes > model_size - offset {
        return false;
    }
    if ABI_REGISTERED_MODEL.lock().ok().is_some_and(|active| {
        active.as_ref().is_some_and(|model| {
            model
                .device_ptr(model_map, model_size, offset, bytes)
                .is_some()
        })
    }) || ABI_COPIED_MODEL.lock().ok().is_some_and(|active| {
        active.as_ref().is_some_and(|model| {
            model
                .device_ptr(model_map, model_size, offset, bytes)
                .is_some()
        })
    }) {
        return true;
    }
    let Some(end) = offset.checked_add(bytes) else {
        return false;
    };
    ABI_MODEL_RANGES.lock().ok().is_some_and(|ranges| {
        ranges.iter().any(|range| {
            range.model_map == model_map as usize
                && range.model_size == model_size
                && offset >= range.offset
                && range
                    .offset
                    .checked_add(range.bytes)
                    .is_some_and(|range_end| end <= range_end)
        })
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
fn abi_q8_shape(in_dim: u64, out_dim: u64) -> Option<(u64, usize, u64)> {
    if in_dim == 0 || out_dim == 0 {
        return None;
    }
    let elements = in_dim.checked_mul(out_dim)?;
    let elements_usize = usize::try_from(elements).ok()?;
    let packed_bytes = out_dim.checked_mul(in_dim.div_ceil(32))?.checked_mul(34)?;
    Some((elements, elements_usize, packed_bytes))
}

#[cfg(feature = "cuda-oxide-kernels")]
fn cache_abi_q8_f16_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    in_dim: u64,
    out_dim: u64,
    label: &str,
    preload: bool,
) -> bool {
    let Some((elements, elements_usize, packed_bytes)) = abi_q8_shape(in_dim, out_dim) else {
        return true;
    };
    if packed_bytes > bytes {
        return true;
    }
    let Ok(mut cache) = ABI_Q8_CACHE.lock() else {
        return false;
    };
    if q8_range_matches(
        &cache.f16_ranges,
        |range| {
            (
                range.model_map,
                range.offset,
                range.weight_bytes,
                range.in_dim,
                range.out_dim,
            )
        },
        model_map,
        offset,
        bytes,
        in_dim,
        out_dim,
    ) {
        return true;
    }
    let Some(out_bytes) = elements.checked_mul(size_of::<f16>() as u64) else {
        return true;
    };
    let options = abi_q8_cache_options();
    let admission = cache.state.admit_f16_bytes(
        options,
        Some(label),
        in_dim,
        out_dim,
        out_bytes,
        backend.memory_capacity().ok(),
        preload,
    );
    if !admission.admitted {
        if preload {
            cache.state.disable_optional_preload_after_failure();
        }
        return true;
    }
    let Ok(device) = backend.zeroed::<f16>(elements_usize) else {
        cache.f16_ranges.clear();
        cache.state.disable_f16_after_failure();
        if preload {
            cache.state.disable_optional_preload_after_failure();
        }
        return true;
    };
    let launched = with_cached_abi_model_range(
        backend,
        model_map,
        model_size,
        offset,
        bytes,
        |weights_ptr| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: the public range and packed Q8 shape were checked
                // above; the retained converted buffer survives queued use.
                Some(unsafe {
                    kernels.dequant_q8_f16_tensor(
                        backend.stream(),
                        weights_ptr,
                        device.cu_deviceptr(),
                        bytes,
                        elements,
                        in_dim,
                        out_dim,
                    )
                })
            })
        },
    )
    .unwrap_or(false);
    if !launched {
        let _ = backend.synchronize();
        cache.f16_ranges.clear();
        cache.state.disable_f16_after_failure();
        if preload {
            cache.state.disable_optional_preload_after_failure();
        }
        return true;
    }
    cache.state.record_f16_success(out_bytes);
    cache.f16_ranges.push(AbiQ8F16Range {
        model_map: model_map as usize,
        offset,
        weight_bytes: bytes,
        in_dim,
        out_dim,
        device,
    });
    true
}

#[cfg(feature = "cuda-oxide-kernels")]
fn cache_abi_q8_f32_range(
    backend: &CudaOxideSubstrate,
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    in_dim: u64,
    out_dim: u64,
) -> bool {
    let Some((elements, elements_usize, packed_bytes)) = abi_q8_shape(in_dim, out_dim) else {
        return true;
    };
    if packed_bytes > bytes {
        return true;
    }
    let Ok(mut cache) = ABI_Q8_CACHE.lock() else {
        return false;
    };
    if q8_range_matches(
        &cache.f32_ranges,
        |range| {
            (
                range.model_map,
                range.offset,
                range.weight_bytes,
                range.in_dim,
                range.out_dim,
            )
        },
        model_map,
        offset,
        bytes,
        in_dim,
        out_dim,
    ) {
        return true;
    }
    let Ok(device) = backend.zeroed::<f32>(elements_usize) else {
        cache.state.disable_optional_preload_after_failure();
        return true;
    };
    let launched = with_cached_abi_model_range(
        backend,
        model_map,
        model_size,
        offset,
        bytes,
        |weights_ptr| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: the public range and packed Q8 shape were checked
                // above; the retained converted buffer survives queued use.
                Some(unsafe {
                    kernels.dequant_q8_f32_tensor(
                        backend.stream(),
                        weights_ptr,
                        device.cu_deviceptr(),
                        bytes,
                        elements,
                        in_dim,
                        out_dim,
                    )
                })
            })
        },
    )
    .unwrap_or(false);
    if !launched {
        let _ = backend.synchronize();
        cache.state.disable_optional_preload_after_failure();
        return true;
    }
    cache.f32_ranges.push(AbiQ8F32Range {
        model_map: model_map as usize,
        offset,
        weight_bytes: bytes,
        in_dim,
        out_dim,
        device,
    });
    true
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
            update_abi_blas_math_state();
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
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut activations) = ABI_F16_ACTIVATIONS.lock() {
                *activations = None;
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut activations) = ABI_Q8_ACTIVATIONS.lock() {
                *activations = None;
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut tables) = ABI_MOE_IQ2_TABLES.lock() {
                *tables = None;
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut scratch) = ABI_ROUTED_MOE_BATCH_SCRATCH.lock() {
                *scratch = None;
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut scratch) = ABI_ATTENTION_OUTPUT_CUBLAS_SCRATCH.lock() {
                *scratch = None;
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut scratch) = ABI_ATTENTION_PREFILL_CUBLAS_SCRATCH.lock() {
                *scratch = None;
            }
            #[cfg(feature = "cuda-oxide-kernels")]
            if let Ok(mut q8_cache) = ABI_Q8_CACHE.lock() {
                clear_abi_q8_converted_ranges(&mut q8_cache);
            }
            if let Ok(mut model_ranges) = ABI_MODEL_RANGES.lock() {
                model_ranges.clear();
            }
            #[cfg(target_os = "linux")]
            if let Ok(mut model_arenas) = ABI_MODEL_ARENAS.lock() {
                model_arenas.arenas.clear();
                model_arenas.range_bytes = 0;
                model_arenas.cache_full = false;
                model_arenas.progress.reset();
            }
            #[cfg(target_os = "linux")]
            if let Ok(mut stage_pool) = ABI_MODEL_STAGE_POOL.lock() {
                stage_pool.slots.clear();
                stage_pool.stage_bytes = 0;
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
pub unsafe extern "C" fn ds4_gpu_hc_split_sinkhorn_tensor(
    out: *mut Ds4GpuTensor,
    mix: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) -> c_int {
    status(|| {
        const MIX_ELEMENTS: u64 = 24;
        const SCALE_ELEMENTS: u64 = 3;

        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(mix) = (unsafe { tensor_ref(mix) }) else {
            return false;
        };
        let mix_bytes = MIX_ELEMENTS * size_of::<f32>() as u64;
        let scale_bytes = SCALE_ELEMENTS * size_of::<f32>() as u64;
        if model_map.is_null()
            || n_hc != 4
            || scale_offset > model_size
            || scale_bytes > model_size - scale_offset
            || base_offset > model_size
            || mix_bytes > model_size - base_offset
            || mix.bytes < mix_bytes
            || out.bytes < mix_bytes
        {
            return false;
        }
        let rows = (mix.bytes / mix_bytes).min(out.bytes / mix_bytes);
        let Ok(n_rows) = u32::try_from(rows) else {
            return false;
        };
        if n_rows == 0 {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                scale_offset,
                scale_bytes,
                |scale_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        base_offset,
                        mix_bytes,
                        |base_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: rows are bounded by both tensors and
                                // cached model spans cover the kernel reads.
                                Some(unsafe {
                                    kernels.hc_split_sinkhorn_tensor(
                                        backend.stream(),
                                        out.device_ptr(),
                                        mix.device_ptr(),
                                        scale_ptr,
                                        base_ptr,
                                        n_rows,
                                        sinkhorn_iters,
                                        eps,
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_hc_split_weighted_sum_tensor(
    out: *mut Ds4GpuTensor,
    split: *mut Ds4GpuTensor,
    mix: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) -> c_int {
    status(|| {
        const MIX_ELEMENTS: u64 = 24;
        const SCALE_ELEMENTS: u64 = 3;

        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(split) = (unsafe { tensor_ref(split.cast_const()) }) else {
            return false;
        };
        let Some(mix) = (unsafe { tensor_ref(mix) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(out_row_bytes) = u64::from(n_embd).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let mix_bytes = MIX_ELEMENTS * size_of::<f32>() as u64;
        let scale_bytes = SCALE_ELEMENTS * size_of::<f32>() as u64;
        if model_map.is_null()
            || n_embd == 0
            || n_hc != 4
            || out.bytes < out_row_bytes
            || out.bytes % out_row_bytes != 0
            || scale_offset > model_size
            || scale_bytes > model_size - scale_offset
            || base_offset > model_size
            || mix_bytes > model_size - base_offset
        {
            return false;
        }
        let rows = out.bytes / out_row_bytes;
        let Ok(n_rows) = u32::try_from(rows) else {
            return false;
        };
        if n_rows == 0 {
            return false;
        }
        let Some(required_mix_bytes) = rows.checked_mul(mix_bytes) else {
            return false;
        };
        let Some(required_residual_bytes) = rows
            .checked_mul(u64::from(n_hc))
            .and_then(|value| value.checked_mul(out_row_bytes))
        else {
            return false;
        };
        if mix.bytes < required_mix_bytes
            || split.bytes < required_mix_bytes
            || residual_hc.bytes < required_residual_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                scale_offset,
                scale_bytes,
                |scale_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        base_offset,
                        mix_bytes,
                        |base_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: output-derived rows and all fused
                                // input/model spans are validated above.
                                Some(unsafe {
                                    kernels.hc_split_weighted_sum_tensor(
                                        backend.stream(),
                                        out.device_ptr(),
                                        split.device_ptr(),
                                        mix.device_ptr(),
                                        residual_hc.device_ptr(),
                                        scale_ptr,
                                        base_ptr,
                                        n_embd,
                                        n_hc,
                                        n_rows,
                                        sinkhorn_iters,
                                        eps,
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn hc_split_weighted_sum_norm_fallback(
    out: *mut Ds4GpuTensor,
    norm_out: *mut Ds4GpuTensor,
    split: *mut Ds4GpuTensor,
    mix: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    norm_eps: f32,
) -> bool {
    unsafe {
        ds4_gpu_hc_split_weighted_sum_tensor(
            out,
            split,
            mix,
            residual_hc,
            model_map,
            model_size,
            scale_offset,
            base_offset,
            n_embd,
            n_hc,
            sinkhorn_iters,
            eps,
        ) != 0
            && ds4_gpu_rms_norm_weight_tensor(
                norm_out,
                out.cast_const(),
                model_map,
                model_size,
                norm_weight_offset,
                n_embd,
                norm_eps,
            ) != 0
    }
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_hc_split_weighted_sum_norm_tensor(
    out: *mut Ds4GpuTensor,
    norm_out: *mut Ds4GpuTensor,
    split: *mut Ds4GpuTensor,
    mix: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    norm_eps: f32,
) -> c_int {
    if std::env::var_os("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED").is_some() {
        return status(|| unsafe {
            hc_split_weighted_sum_norm_fallback(
                out,
                norm_out,
                split,
                mix,
                residual_hc,
                model_map,
                model_size,
                scale_offset,
                base_offset,
                norm_weight_offset,
                n_embd,
                n_hc,
                sinkhorn_iters,
                eps,
                norm_eps,
            )
        });
    }
    status(|| {
        const MIX_ELEMENTS: u64 = 24;
        const SCALE_ELEMENTS: u64 = 3;

        let Some(out_ref) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(norm_out_ref) = (unsafe { tensor_ref(norm_out.cast_const()) }) else {
            return false;
        };
        let Some(split_ref) = (unsafe { tensor_ref(split.cast_const()) }) else {
            return false;
        };
        let Some(mix_ref) = (unsafe { tensor_ref(mix) }) else {
            return false;
        };
        let Some(residual_ref) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(out_row_bytes) = u64::from(n_embd).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let mix_bytes = MIX_ELEMENTS * size_of::<f32>() as u64;
        let scale_bytes = SCALE_ELEMENTS * size_of::<f32>() as u64;
        if model_map.is_null()
            || n_embd == 0
            || n_hc != 4
            || out_ref.bytes < out_row_bytes
            || out_ref.bytes % out_row_bytes != 0
            || norm_out_ref.bytes < out_ref.bytes
            || scale_offset > model_size
            || scale_bytes > model_size - scale_offset
            || base_offset > model_size
            || mix_bytes > model_size - base_offset
            || norm_weight_offset > model_size
            || out_row_bytes > model_size - norm_weight_offset
        {
            return false;
        }
        let rows = out_ref.bytes / out_row_bytes;
        if rows != 1 {
            return unsafe {
                hc_split_weighted_sum_norm_fallback(
                    out,
                    norm_out,
                    split,
                    mix,
                    residual_hc,
                    model_map,
                    model_size,
                    scale_offset,
                    base_offset,
                    norm_weight_offset,
                    n_embd,
                    n_hc,
                    sinkhorn_iters,
                    eps,
                    norm_eps,
                )
            };
        }
        let required_residual_bytes = u64::from(n_hc) * out_row_bytes;
        if mix_ref.bytes < mix_bytes
            || split_ref.bytes < mix_bytes
            || residual_ref.bytes < required_residual_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                scale_offset,
                scale_bytes,
                |scale_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        base_offset,
                        mix_bytes,
                        |base_ptr| {
                            with_cached_abi_model_range(
                                backend,
                                model_map,
                                model_size,
                                norm_weight_offset,
                                out_row_bytes,
                                |norm_weight_ptr| {
                                    with_abi_kernels(backend, |kernels| {
                                        // SAFETY: the one-row fused branch is
                                        // selected only after all spans above.
                                        Some(unsafe {
                                            kernels.hc_split_weighted_sum_norm_tensor(
                                                backend.stream(),
                                                out_ref.device_ptr(),
                                                norm_out_ref.device_ptr(),
                                                split_ref.device_ptr(),
                                                mix_ref.device_ptr(),
                                                residual_ref.device_ptr(),
                                                scale_ptr,
                                                base_ptr,
                                                norm_weight_ptr,
                                                n_embd,
                                                n_hc,
                                                1,
                                                sinkhorn_iters,
                                                eps,
                                                norm_eps,
                                            )
                                        })
                                    })
                                },
                            )
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_output_hc_weights_tensor(
    out: *mut Ds4GpuTensor,
    pre: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    eps: f32,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(pre) = (unsafe { tensor_ref(pre) }) else {
            return false;
        };
        let Some(row_bytes) = u64::from(n_hc).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_hc == 0
            || row_bytes == 0
            || out.bytes < row_bytes
            || out.bytes % row_bytes != 0
            || pre.bytes < out.bytes
            || scale_offset > model_size
            || size_of::<f32>() as u64 > model_size - scale_offset
            || base_offset > model_size
            || row_bytes > model_size - base_offset
        {
            return false;
        }
        let Ok(n_tokens) = u32::try_from(out.bytes / row_bytes) else {
            return false;
        };
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                scale_offset,
                size_of::<f32>() as u64,
                |scale_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        base_offset,
                        row_bytes,
                        |base_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: output rows, input coverage, and
                                // cached scale/base spans are validated above.
                                Some(unsafe {
                                    kernels.output_hc_weights_tensor(
                                        backend.stream(),
                                        out.device_ptr(),
                                        pre.device_ptr(),
                                        scale_ptr,
                                        base_ptr,
                                        n_hc,
                                        n_tokens,
                                        eps,
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_embed_token_hc_tensor(
    out_hc: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n_vocab: u32,
    token: u32,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(weight_elements) = u64::from(n_vocab).checked_mul(u64::from(n_embd)) else {
            return false;
        };
        let Some(weight_bytes) = weight_elements.checked_mul(size_of::<u16>() as u64) else {
            return false;
        };
        let Some(out_elements) = u64::from(n_embd).checked_mul(u64::from(n_hc)) else {
            return false;
        };
        let Some(out_bytes) = out_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_vocab == 0
            || token >= n_vocab
            || n_embd == 0
            || n_hc == 0
            || weight_offset > model_size
            || weight_bytes > model_size - weight_offset
            || out_hc.bytes < out_bytes
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
                |weights_ptr| {
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: single-token bounds, output capacity, and
                        // cached FP16 embedding span are validated above.
                        Some(unsafe {
                            kernels.embed_token_hc_tensor(
                                backend.stream(),
                                out_hc.device_ptr(),
                                weights_ptr,
                                n_vocab,
                                token,
                                n_embd,
                                n_hc,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_embed_tokens_hc_tensor(
    out_hc: *mut Ds4GpuTensor,
    tokens: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    n_vocab: u32,
    n_tokens: u32,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(tokens) = (unsafe { tensor_ref(tokens) }) else {
            return false;
        };
        let Some(weight_elements) = u64::from(n_vocab).checked_mul(u64::from(n_embd)) else {
            return false;
        };
        let Some(weight_bytes) = weight_elements.checked_mul(size_of::<u16>() as u64) else {
            return false;
        };
        let Some(out_elements) = u64::from(n_tokens)
            .checked_mul(u64::from(n_hc))
            .and_then(|elements| elements.checked_mul(u64::from(n_embd)))
        else {
            return false;
        };
        let Some(out_bytes) = out_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(token_bytes) = u64::from(n_tokens).checked_mul(size_of::<i32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_vocab == 0
            || n_tokens == 0
            || n_embd == 0
            || n_hc == 0
            || weight_offset > model_size
            || weight_bytes > model_size - weight_offset
            || tokens.bytes < token_bytes
            || out_hc.bytes < out_bytes
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
                |weights_ptr| {
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: source/output spans and cached embedding
                        // storage are validated; invalid token IDs fall back
                        // to embedding row zero inside the kernel.
                        Some(unsafe {
                            kernels.embed_tokens_hc_tensor(
                                backend.stream(),
                                out_hc.device_ptr(),
                                tokens.device_ptr(),
                                weights_ptr,
                                n_vocab,
                                n_tokens,
                                n_embd,
                                n_hc,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_head_rms_norm_tensor(
    x: *mut Ds4GpuTensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    eps: f32,
) -> c_int {
    status(|| {
        let Some(x) = (unsafe { tensor_ref(x.cast_const()) }) else {
            return false;
        };
        let Some(rows) = u64::from(n_tok).checked_mul(u64::from(n_head)) else {
            return false;
        };
        let Some(elements) = rows.checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(bytes) = elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if n_tok == 0 || n_head == 0 || head_dim == 0 || x.bytes < bytes {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: the mutable tensor span and all launch dimensions
                // are validated above; the kernel only updates that span.
                Some(unsafe {
                    kernels.head_rms_norm_tensor(
                        backend.stream(),
                        x.device_ptr(),
                        n_tok,
                        n_head,
                        head_dim,
                        eps,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_dsv4_fp8_kv_quantize_tensor(
    x: *mut Ds4GpuTensor,
    n_tok: u32,
    head_dim: u32,
    n_rot: u32,
) -> c_int {
    status(|| {
        let Some(x) = (unsafe { tensor_ref(x.cast_const()) }) else {
            return false;
        };
        let Some(elements) = u64::from(n_tok).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(bytes) = elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if n_tok == 0 || n_rot > head_dim || x.bytes < bytes {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: the mutable tensor span and prefix/tail boundary
                // are validated above; the kernel modifies only the prefix.
                Some(unsafe {
                    kernels.dsv4_fp8_kv_quantize_tensor(
                        backend.stream(),
                        x.device_ptr(),
                        n_tok,
                        head_dim,
                        n_rot,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_dsv4_indexer_qat_tensor(
    x: *mut Ds4GpuTensor,
    n_rows: u32,
    head_dim: u32,
) -> c_int {
    status(|| {
        let Some(x) = (unsafe { tensor_ref(x.cast_const()) }) else {
            return false;
        };
        let Some(elements) = u64::from(n_rows).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(bytes) = elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if n_rows == 0 || head_dim != 128 || x.bytes < bytes {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: the mutable tensor span and the exact row width
                // required by the in-place Hadamard kernel are validated above.
                Some(unsafe {
                    kernels.dsv4_indexer_qat_tensor(
                        backend.stream(),
                        x.device_ptr(),
                        n_rows,
                        head_dim,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_dsv4_topk_mask_tensor(
    mask: *mut Ds4GpuTensor,
    topk: *const Ds4GpuTensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> c_int {
    status(|| {
        let Some(mask) = (unsafe { tensor_ref(mask.cast_const()) }) else {
            return false;
        };
        let Some(topk) = (unsafe { tensor_ref(topk) }) else {
            return false;
        };
        let Some(count) = u64::from(n_tokens).checked_mul(u64::from(n_comp)) else {
            return false;
        };
        let Some(selected_count) = u64::from(n_tokens).checked_mul(u64::from(top_k)) else {
            return false;
        };
        let Some(mask_bytes) = count.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(topk_bytes) = selected_count.checked_mul(size_of::<u32>() as u64) else {
            return false;
        };
        if n_comp == 0
            || n_tokens == 0
            || top_k == 0
            || mask.bytes < mask_bytes
            || topk.bytes < topk_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: both device spans and nonzero launch dimensions
                // are validated before submission.
                Some(unsafe {
                    kernels.dsv4_topk_mask_tensor(
                        backend.stream(),
                        mask.device_ptr(),
                        topk.device_ptr(),
                        n_comp,
                        n_tokens,
                        top_k,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_rope_tail_tensor(
    x: *mut Ds4GpuTensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos0: u32,
    n_ctx_orig: u32,
    inverse: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> c_int {
    status(|| {
        let Some(x) = (unsafe { tensor_ref(x.cast_const()) }) else {
            return false;
        };
        let Some(elements) = u64::from(n_tok)
            .checked_mul(u64::from(n_head))
            .and_then(|count| count.checked_mul(u64::from(head_dim)))
        else {
            return false;
        };
        let Some(bytes) = elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(pairs) = n_tok
            .checked_mul(n_head)
            .and_then(|rows| rows.checked_mul(n_rot / 2))
        else {
            return false;
        };
        if n_tok == 0
            || n_head == 0
            || n_rot == 0
            || n_rot > head_dim
            || n_rot & 1 != 0
            || x.bytes < bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: the full mutable tensor span, rotary width, and
                // checked nonzero pair launch are validated above.
                Some(unsafe {
                    kernels.rope_tail_tensor(
                        backend.stream(),
                        x.device_ptr(),
                        n_tok,
                        n_head,
                        head_dim,
                        n_rot,
                        pos0,
                        1,
                        n_ctx_orig,
                        inverse,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        beta_fast,
                        beta_slow,
                        pairs,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
unsafe fn store_raw_kv_impl(
    raw_cache: *mut Ds4GpuTensor,
    kv: *const Ds4GpuTensor,
    raw_cap: u32,
    pos0: u32,
    n_tokens: u32,
    head_dim: u32,
) -> c_int {
    status(|| {
        let Some(raw_cache) = (unsafe { tensor_ref(raw_cache.cast_const()) }) else {
            return false;
        };
        let Some(kv) = (unsafe { tensor_ref(kv) }) else {
            return false;
        };
        let Some(raw_elements) = u64::from(raw_cap).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(kv_elements) = u64::from(n_tokens).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(raw_bytes) = raw_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(kv_bytes) = kv_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Ok(grid_blocks) = u32::try_from(kv_elements.div_ceil(256_u64)) else {
            return false;
        };
        if raw_cap == 0
            || n_tokens == 0
            || head_dim == 0
            || grid_blocks == 0
            || raw_cache.bytes < raw_bytes
            || kv.bytes < kv_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_abi_kernels(backend, |kernels| {
                // SAFETY: source/destination spans, nonzero ring geometry,
                // and the checked launch grid are validated above.
                Some(unsafe {
                    kernels.store_raw_kv_batch_tensor(
                        backend.stream(),
                        raw_cache.device_ptr(),
                        kv.device_ptr(),
                        raw_elements,
                        kv_elements,
                        raw_cap,
                        pos0,
                        n_tokens,
                        head_dim,
                        grid_blocks,
                    )
                })
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_store_raw_kv_tensor(
    raw_cache: *mut Ds4GpuTensor,
    kv: *const Ds4GpuTensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
) -> c_int {
    unsafe { store_raw_kv_impl(raw_cache, kv, raw_cap, row, 1, head_dim) }
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_store_raw_kv_batch_tensor(
    raw_cache: *mut Ds4GpuTensor,
    kv: *const Ds4GpuTensor,
    raw_cap: u32,
    pos0: u32,
    n_tokens: u32,
    head_dim: u32,
) -> c_int {
    unsafe { store_raw_kv_impl(raw_cache, kv, raw_cap, pos0, n_tokens, head_dim) }
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_kv_fp8_store_raw_tensor(
    kv: *mut Ds4GpuTensor,
    raw_cache: *mut Ds4GpuTensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
    n_rot: u32,
) -> c_int {
    if unsafe { ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) } == 0 {
        return 0;
    }
    unsafe { ds4_gpu_store_raw_kv_tensor(raw_cache, kv.cast_const(), raw_cap, row, head_dim) }
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_compressor_store_batch_tensor(
    kv: *const Ds4GpuTensor,
    sc: *const Ds4GpuTensor,
    state_kv: *mut Ds4GpuTensor,
    state_score: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
) -> c_int {
    status(|| {
        let Some(kv) = (unsafe { tensor_ref(kv) }) else {
            return false;
        };
        let Some(sc) = (unsafe { tensor_ref(sc) }) else {
            return false;
        };
        let Some(state_kv) = (unsafe { tensor_ref(state_kv.cast_const()) }) else {
            return false;
        };
        let Some(state_score) = (unsafe { tensor_ref(state_score.cast_const()) }) else {
            return false;
        };
        let coff = if ratio == 4 { 2_u32 } else { 1_u32 };
        let Some(width) = coff.checked_mul(head_dim) else {
            return false;
        };
        let Some(state_rows) = coff.checked_mul(ratio) else {
            return false;
        };
        let Some(input_elements) = u64::from(n_tokens).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(state_elements) = u64::from(state_rows).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(ape_elements) = u64::from(width).checked_mul(u64::from(ratio)) else {
            return false;
        };
        let Some(input_bytes) = input_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(state_bytes) = state_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let ape_element_bytes = if ape_type == 1 {
            size_of::<u16>() as u64
        } else {
            size_of::<f32>() as u64
        };
        let Some(ape_bytes) = ape_elements.checked_mul(ape_element_bytes) else {
            return false;
        };
        let Ok(grid_blocks) = u32::try_from(input_elements.div_ceil(256_u64)) else {
            return false;
        };
        if model_map.is_null()
            || head_dim == 0
            || ratio == 0
            || n_tokens == 0
            || ape_type > 1
            || width == 0
            || grid_blocks == 0
            || ape_offset > model_size
            || ape_bytes > model_size - ape_offset
            || kv.bytes < input_bytes
            || sc.bytes < input_bytes
            || state_kv.bytes < state_bytes
            || state_score.bytes < state_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                ape_offset,
                ape_bytes,
                |ape_ptr| {
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: source/state spans, the selected cached APE
                        // range, geometry, and checked grid are validated above.
                        Some(unsafe {
                            kernels.compressor_store_batch_tensor(
                                backend.stream(),
                                kv.device_ptr(),
                                sc.device_ptr(),
                                state_kv.device_ptr(),
                                state_score.device_ptr(),
                                ape_ptr,
                                input_elements,
                                state_elements,
                                ape_elements,
                                ape_type,
                                head_dim,
                                ratio,
                                pos0,
                                n_tokens,
                                grid_blocks,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_compressor_update_tensor(
    kv_cur: *const Ds4GpuTensor,
    sc_cur: *const Ds4GpuTensor,
    state_kv: *mut Ds4GpuTensor,
    state_score: *mut Ds4GpuTensor,
    comp_cache: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    ratio: u32,
    pos: u32,
    comp_row: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> c_int {
    status(|| {
        let Some(kv_cur_ref) = (unsafe { tensor_ref(kv_cur) }) else {
            return false;
        };
        let Some(sc_cur_ref) = (unsafe { tensor_ref(sc_cur) }) else {
            return false;
        };
        let Some(state_kv_ref) = (unsafe { tensor_ref(state_kv.cast_const()) }) else {
            return false;
        };
        let Some(state_score_ref) = (unsafe { tensor_ref(state_score.cast_const()) }) else {
            return false;
        };
        let Some(comp_cache_ref) = (unsafe { tensor_ref(comp_cache.cast_const()) }) else {
            return false;
        };
        if model_map.is_null()
            || head_dim == 0
            || ratio == 0
            || n_rot > head_dim
            || n_rot & 1 != 0
            || ape_type > 1
            || norm_type != 0
        {
            return false;
        }
        let coff = if ratio == 4 { 2_u32 } else { 1_u32 };
        let Some(width) = coff.checked_mul(head_dim) else {
            return false;
        };
        let Some(state_rows) = coff.checked_mul(ratio) else {
            return false;
        };
        let input_elements = u64::from(width);
        let Some(state_elements) = u64::from(state_rows).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(input_bytes) = input_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(state_bytes) = state_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(ape_elements) = u64::from(width).checked_mul(u64::from(ratio)) else {
            return false;
        };
        let ape_element_bytes = if ape_type == 1 {
            size_of::<u16>() as u64
        } else {
            size_of::<f32>() as u64
        };
        let Some(ape_bytes) = ape_elements.checked_mul(ape_element_bytes) else {
            return false;
        };
        let norm_bytes = u64::from(head_dim) * size_of::<f32>() as u64;
        let emit = pos.wrapping_add(1) % ratio == 0;
        let mut row_ptr = 0_u64;
        if emit {
            let Some(comp_rows) = comp_row.checked_add(1) else {
                return false;
            };
            let Some(comp_elements) = u64::from(comp_rows).checked_mul(u64::from(head_dim)) else {
                return false;
            };
            let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
                return false;
            };
            let Some(row_offset) = u64::from(comp_row)
                .checked_mul(u64::from(head_dim))
                .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
            else {
                return false;
            };
            let Some((device_ptr, _)) = checked_range(comp_cache_ref, row_offset, norm_bytes)
            else {
                return false;
            };
            if comp_cache_ref.bytes < comp_bytes {
                return false;
            }
            row_ptr = device_ptr;
        }
        if width == 0
            || ape_offset > model_size
            || ape_bytes > model_size - ape_offset
            || norm_offset > model_size
            || norm_bytes > model_size - norm_offset
            || kv_cur_ref.bytes < input_bytes
            || sc_cur_ref.bytes < input_bytes
            || state_kv_ref.bytes < state_bytes
            || state_score_ref.bytes < state_bytes
        {
            return false;
        }
        // The current-C wrapper validates output/model spans before mutating
        // state, then always stores the current row before deciding to emit.
        if unsafe {
            ds4_gpu_compressor_store_batch_tensor(
                kv_cur,
                sc_cur,
                state_kv,
                state_score,
                model_map,
                model_size,
                ape_offset,
                ape_type,
                head_dim,
                ratio,
                pos,
                1,
            )
        } == 0
        {
            return false;
        }
        if !emit {
            return true;
        }
        with_backend(|backend| {
            let pooled = with_abi_kernels(backend, |kernels| {
                // SAFETY: the output row and updated state spans plus the
                // checked ratio-dependent geometry were validated above.
                Some(unsafe {
                    kernels.compressor_update_pool_tensor(
                        backend.stream(),
                        row_ptr,
                        state_kv_ref.device_ptr(),
                        state_score_ref.device_ptr(),
                        state_elements,
                        head_dim,
                        ratio,
                    )
                })
            })?;
            if !pooled {
                return Some(false);
            }
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                norm_offset,
                norm_bytes,
                |norm_ptr| {
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: emitted row and cached norm spans are valid;
                        // launches are sequenced after the published store.
                        if !unsafe {
                            kernels.rms_norm_weight_rows_tensor(
                                backend.stream(),
                                row_ptr,
                                row_ptr,
                                norm_ptr,
                                head_dim,
                                1,
                                rms_eps,
                            )
                        } {
                            return Some(false);
                        }
                        if n_rot == 0 {
                            // Current C reaches a zero-grid RoPE submission
                            // after store/pool/RMS and fails before shifting.
                            return Some(false);
                        }
                        if !unsafe {
                            kernels.rope_tail_tensor(
                                backend.stream(),
                                row_ptr,
                                1,
                                1,
                                head_dim,
                                n_rot,
                                pos.wrapping_add(1).wrapping_sub(ratio),
                                1,
                                n_ctx_orig,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow,
                                n_rot / 2,
                            )
                        } {
                            return Some(false);
                        }
                        if ratio == 4 {
                            let shift_elements = 4_u64 * u64::from(width);
                            let Ok(shift_blocks) = u32::try_from(shift_elements.div_ceil(256_u64))
                            else {
                                return Some(false);
                            };
                            if !unsafe {
                                kernels.compressor_shift_ratio4_tensor(
                                    backend.stream(),
                                    state_kv_ref.device_ptr(),
                                    state_score_ref.device_ptr(),
                                    state_elements,
                                    width,
                                    shift_blocks,
                                )
                            } {
                                return Some(false);
                            }
                        }
                        Some(true)
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_compressor_prefill_tensor(
    comp_cache: *mut Ds4GpuTensor,
    state_kv: *mut Ds4GpuTensor,
    state_score: *mut Ds4GpuTensor,
    kv: *const Ds4GpuTensor,
    sc: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    quantize_fp8: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> c_int {
    status(|| {
        let Some(comp_cache) = (unsafe { tensor_ref(comp_cache.cast_const()) }) else {
            return false;
        };
        let Some(state_kv) = (unsafe { tensor_ref(state_kv.cast_const()) }) else {
            return false;
        };
        let Some(state_score) = (unsafe { tensor_ref(state_score.cast_const()) }) else {
            return false;
        };
        let Some(kv) = (unsafe { tensor_ref(kv) }) else {
            return false;
        };
        let Some(sc) = (unsafe { tensor_ref(sc) }) else {
            return false;
        };
        if model_map.is_null()
            || head_dim == 0
            || ratio == 0
            || n_tokens == 0
            || n_rot > head_dim
            || n_rot & 1 != 0
            || ape_type > 1
            || norm_type != 0
        {
            return false;
        }
        let coff = if ratio == 4 { 2_u32 } else { 1_u32 };
        let Some(width) = coff.checked_mul(head_dim) else {
            return false;
        };
        let Some(state_rows) = coff.checked_mul(ratio) else {
            return false;
        };
        let n_comp = n_tokens / ratio;
        let cutoff = n_comp * ratio;
        let rem = n_tokens - cutoff;
        let Some(input_elements) = u64::from(n_tokens).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(state_elements) = u64::from(state_rows).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(comp_elements) = u64::from(n_comp).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(ape_elements) = u64::from(ratio).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(input_bytes) = input_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(state_bytes) = state_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let ape_element_bytes = if ape_type == 1 {
            size_of::<u16>() as u64
        } else {
            size_of::<f32>() as u64
        };
        let Some(ape_bytes) = ape_elements.checked_mul(ape_element_bytes) else {
            return false;
        };
        let norm_bytes = u64::from(head_dim) * size_of::<f32>() as u64;
        let Some(previous_elements) = u64::from(ratio).checked_mul(u64::from(width)) else {
            return false;
        };
        let Some(remainder_elements) = u64::from(rem).checked_mul(u64::from(width)) else {
            return false;
        };
        let Ok(previous_blocks) = u32::try_from(previous_elements.div_ceil(256_u64)) else {
            return false;
        };
        let Ok(remainder_blocks) = u32::try_from(remainder_elements.div_ceil(256_u64)) else {
            return false;
        };
        let Some(pairs) = n_comp.checked_mul(n_rot / 2) else {
            return false;
        };
        if width == 0
            || ape_offset > model_size
            || ape_bytes > model_size - ape_offset
            || norm_offset > model_size
            || norm_bytes > model_size - norm_offset
            || kv.bytes < input_bytes
            || sc.bytes < input_bytes
            || state_kv.bytes < state_bytes
            || state_score.bytes < state_bytes
            || (n_comp != 0 && comp_cache.bytes < comp_bytes)
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                ape_offset,
                ape_bytes,
                |ape_ptr| {
                    backend.context().bind_to_thread().ok()?;
                    let state_len = usize::try_from(state_elements).ok()?;
                    // SAFETY: state spans and element count were validated;
                    // fills precede every placement and output launch.
                    unsafe {
                        cuda_core::sys::cuMemsetD32Async(
                            state_kv.device_ptr(),
                            0.0_f32.to_bits(),
                            state_len,
                            backend.stream().cu_stream(),
                        )
                    }
                    .result()
                    .ok()?;
                    unsafe {
                        cuda_core::sys::cuMemsetD32Async(
                            state_score.device_ptr(),
                            f32::NEG_INFINITY.to_bits(),
                            state_len,
                            backend.stream().cu_stream(),
                        )
                    }
                    .result()
                    .ok()?;
                    let placed = with_abi_kernels(backend, |kernels| {
                        if ratio == 4 && cutoff >= ratio {
                            // SAFETY: the final completed ratio-4 block,
                            // initialized state, APE span, and grid are valid.
                            if !unsafe {
                                kernels.compressor_set_rows_tensor(
                                    backend.stream(),
                                    kv.device_ptr(),
                                    sc.device_ptr(),
                                    state_kv.device_ptr(),
                                    state_score.device_ptr(),
                                    ape_ptr,
                                    input_elements,
                                    state_elements,
                                    ape_elements,
                                    ape_type,
                                    width,
                                    ratio,
                                    pos0,
                                    cutoff - ratio,
                                    0,
                                    ratio,
                                    previous_blocks,
                                )
                            } {
                                return Some(false);
                            }
                        }
                        if rem != 0 {
                            let destination = if ratio == 4 { ratio } else { 0 };
                            // SAFETY: remainder rows, destination state bank,
                            // selected APE span, and grid are validated.
                            if !unsafe {
                                kernels.compressor_set_rows_tensor(
                                    backend.stream(),
                                    kv.device_ptr(),
                                    sc.device_ptr(),
                                    state_kv.device_ptr(),
                                    state_score.device_ptr(),
                                    ape_ptr,
                                    input_elements,
                                    state_elements,
                                    ape_elements,
                                    ape_type,
                                    width,
                                    ratio,
                                    pos0,
                                    cutoff,
                                    destination,
                                    rem,
                                    remainder_blocks,
                                )
                            } {
                                return Some(false);
                            }
                        }
                        Some(true)
                    })?;
                    if !placed || n_comp == 0 {
                        return Some(placed);
                    }
                    let pooled = with_abi_kernels(backend, |kernels| {
                        // SAFETY: input, output, selected APE span, and
                        // checked nonzero compressed-row geometry are valid.
                        Some(unsafe {
                            kernels.compressor_prefill_pool_tensor(
                                backend.stream(),
                                comp_cache.device_ptr(),
                                kv.device_ptr(),
                                sc.device_ptr(),
                                ape_ptr,
                                input_elements,
                                comp_elements,
                                ape_elements,
                                ape_type,
                                head_dim,
                                ratio,
                                pos0,
                                n_comp,
                            )
                        })
                    })?;
                    if !pooled {
                        return Some(false);
                    }
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        norm_offset,
                        norm_bytes,
                        |norm_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                if !unsafe {
                                    kernels.rms_norm_weight_rows_tensor(
                                        backend.stream(),
                                        comp_cache.device_ptr(),
                                        comp_cache.device_ptr(),
                                        norm_ptr,
                                        head_dim,
                                        n_comp,
                                        rms_eps,
                                    )
                                } {
                                    return Some(false);
                                }
                                if n_rot != 0
                                    && !unsafe {
                                        kernels.rope_tail_tensor(
                                            backend.stream(),
                                            comp_cache.device_ptr(),
                                            n_comp,
                                            1,
                                            head_dim,
                                            n_rot,
                                            pos0,
                                            ratio,
                                            n_ctx_orig,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            beta_fast,
                                            beta_slow,
                                            pairs,
                                        )
                                    }
                                {
                                    return Some(false);
                                }
                                if quantize_fp8
                                    && !unsafe {
                                        kernels.dsv4_fp8_kv_quantize_tensor(
                                            backend.stream(),
                                            comp_cache.device_ptr(),
                                            n_comp,
                                            head_dim,
                                            n_rot,
                                        )
                                    }
                                {
                                    return Some(false);
                                }
                                Some(true)
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_compressor_prefill_state_ratio4_tensor(
    state_kv: *mut Ds4GpuTensor,
    state_score: *mut Ds4GpuTensor,
    kv_tail: *const Ds4GpuTensor,
    sc_tail: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    pos0: u32,
) -> c_int {
    status(|| {
        let Some(state_kv) = (unsafe { tensor_ref(state_kv.cast_const()) }) else {
            return false;
        };
        let Some(state_score) = (unsafe { tensor_ref(state_score.cast_const()) }) else {
            return false;
        };
        let Some(kv_tail) = (unsafe { tensor_ref(kv_tail) }) else {
            return false;
        };
        let Some(sc_tail) = (unsafe { tensor_ref(sc_tail) }) else {
            return false;
        };
        let Some(width) = 2_u32.checked_mul(head_dim) else {
            return false;
        };
        let tail_elements = 4_u64 * u64::from(width);
        let state_elements = 8_u64 * u64::from(width);
        let ape_elements = 4_u64 * u64::from(width);
        let Some(tail_bytes) = tail_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(state_bytes) = state_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let ape_element_bytes = if ape_type == 1 {
            size_of::<u16>() as u64
        } else {
            size_of::<f32>() as u64
        };
        let Some(ape_bytes) = ape_elements.checked_mul(ape_element_bytes) else {
            return false;
        };
        let Ok(grid_blocks) = u32::try_from(tail_elements.div_ceil(256_u64)) else {
            return false;
        };
        if model_map.is_null()
            || head_dim == 0
            || ape_type > 1
            || width == 0
            || grid_blocks == 0
            || ape_offset > model_size
            || ape_bytes > model_size - ape_offset
            || kv_tail.bytes < tail_bytes
            || sc_tail.bytes < tail_bytes
            || state_kv.bytes < state_bytes
            || state_score.bytes < state_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                ape_offset,
                ape_bytes,
                |ape_ptr| {
                    backend.context().bind_to_thread().ok()?;
                    let state_len = usize::try_from(state_elements).ok()?;
                    // SAFETY: state spans and element count were validated above;
                    // both fills are ordered before the kernel on this stream.
                    unsafe {
                        cuda_core::sys::cuMemsetD32Async(
                            state_kv.device_ptr(),
                            0.0_f32.to_bits(),
                            state_len,
                            backend.stream().cu_stream(),
                        )
                    }
                    .result()
                    .ok()?;
                    unsafe {
                        cuda_core::sys::cuMemsetD32Async(
                            state_score.device_ptr(),
                            f32::NEG_INFINITY.to_bits(),
                            state_len,
                            backend.stream().cu_stream(),
                        )
                    }
                    .result()
                    .ok()?;
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: state/source spans, initialized state, cached
                        // APE range, and checked fixed-ratio grid are validated.
                        Some(unsafe {
                            kernels.compressor_set_rows_tensor(
                                backend.stream(),
                                kv_tail.device_ptr(),
                                sc_tail.device_ptr(),
                                state_kv.device_ptr(),
                                state_score.device_ptr(),
                                ape_ptr,
                                tail_elements,
                                state_elements,
                                ape_elements,
                                ape_type,
                                width,
                                4,
                                pos0,
                                0,
                                0,
                                4,
                                grid_blocks,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_compressor_prefill_ratio4_replay_tensor(
    comp_cache: *mut Ds4GpuTensor,
    state_kv: *mut Ds4GpuTensor,
    state_score: *mut Ds4GpuTensor,
    kv: *const Ds4GpuTensor,
    sc: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    pos0: u32,
    n_tokens: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    quantize_fp8: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> c_int {
    status(|| {
        let Some(comp_cache) = (unsafe { tensor_ref(comp_cache.cast_const()) }) else {
            return false;
        };
        let Some(state_kv) = (unsafe { tensor_ref(state_kv.cast_const()) }) else {
            return false;
        };
        let Some(state_score) = (unsafe { tensor_ref(state_score.cast_const()) }) else {
            return false;
        };
        let Some(kv) = (unsafe { tensor_ref(kv) }) else {
            return false;
        };
        let Some(sc) = (unsafe { tensor_ref(sc) }) else {
            return false;
        };
        let Some(width) = 2_u32.checked_mul(head_dim) else {
            return false;
        };
        let n_comp = n_tokens / 4;
        let Some(input_elements) = u64::from(n_tokens).checked_mul(u64::from(width)) else {
            return false;
        };
        let state_elements = 8_u64 * u64::from(width);
        let Some(comp_elements) = u64::from(n_comp).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let ape_elements = 4_u64 * u64::from(width);
        let Some(input_bytes) = input_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(state_bytes) = state_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let ape_element_bytes = if ape_type == 1 {
            size_of::<u16>() as u64
        } else {
            size_of::<f32>() as u64
        };
        let Some(ape_bytes) = ape_elements.checked_mul(ape_element_bytes) else {
            return false;
        };
        let norm_bytes = u64::from(head_dim) * size_of::<f32>() as u64;
        let tail_elements = 4_u64 * u64::from(width);
        let Ok(tail_grid_blocks) = u32::try_from(tail_elements.div_ceil(256_u64)) else {
            return false;
        };
        let Some(pairs) = n_comp.checked_mul(n_rot / 2) else {
            return false;
        };
        if model_map.is_null()
            || head_dim == 0
            || n_tokens == 0
            || n_tokens & 3 != 0
            || pos0 & 3 != 0
            || n_rot > head_dim
            || n_rot & 1 != 0
            || ape_type > 1
            || norm_type != 0
            || width == 0
            || tail_grid_blocks == 0
            || (n_rot != 0 && pairs == 0)
            || ape_offset > model_size
            || ape_bytes > model_size - ape_offset
            || norm_offset > model_size
            || norm_bytes > model_size - norm_offset
            || kv.bytes < input_bytes
            || sc.bytes < input_bytes
            || state_kv.bytes < state_bytes
            || state_score.bytes < state_bytes
            || comp_cache.bytes < comp_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                ape_offset,
                ape_bytes,
                |ape_ptr| {
                    let pooled = with_abi_kernels(backend, |kernels| {
                        // SAFETY: replay input, prior state, output, APE range,
                        // and fixed-ratio launch dimensions are validated above.
                        Some(unsafe {
                            kernels.compressor_prefill_ratio4_replay_pool_tensor(
                                backend.stream(),
                                comp_cache.device_ptr(),
                                kv.device_ptr(),
                                sc.device_ptr(),
                                state_kv.device_ptr(),
                                state_score.device_ptr(),
                                ape_ptr,
                                input_elements,
                                state_elements,
                                comp_elements,
                                ape_elements,
                                ape_type,
                                head_dim,
                                pos0,
                                n_comp,
                            )
                        })
                    })?;
                    if !pooled {
                        return Some(false);
                    }
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        norm_offset,
                        norm_bytes,
                        |norm_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: the complete replay output and
                                // selected cached norm range were validated.
                                if !unsafe {
                                    kernels.rms_norm_weight_rows_tensor(
                                        backend.stream(),
                                        comp_cache.device_ptr(),
                                        comp_cache.device_ptr(),
                                        norm_ptr,
                                        head_dim,
                                        n_comp,
                                        rms_eps,
                                    )
                                } {
                                    return Some(false);
                                }
                                if n_rot != 0
                                    && !unsafe {
                                        kernels.rope_tail_tensor(
                                            backend.stream(),
                                            comp_cache.device_ptr(),
                                            n_comp,
                                            1,
                                            head_dim,
                                            n_rot,
                                            pos0,
                                            4,
                                            n_ctx_orig,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            beta_fast,
                                            beta_slow,
                                            pairs,
                                        )
                                    }
                                {
                                    return Some(false);
                                }
                                if quantize_fp8
                                    && !unsafe {
                                        kernels.dsv4_fp8_kv_quantize_tensor(
                                            backend.stream(),
                                            comp_cache.device_ptr(),
                                            n_comp,
                                            head_dim,
                                            n_rot,
                                        )
                                    }
                                {
                                    return Some(false);
                                }
                                backend.context().bind_to_thread().ok()?;
                                let state_len = usize::try_from(state_elements).ok()?;
                                // SAFETY: state rebuild follows successful
                                // compressed output work on the same stream.
                                unsafe {
                                    cuda_core::sys::cuMemsetD32Async(
                                        state_kv.device_ptr(),
                                        0.0_f32.to_bits(),
                                        state_len,
                                        backend.stream().cu_stream(),
                                    )
                                }
                                .result()
                                .ok()?;
                                unsafe {
                                    cuda_core::sys::cuMemsetD32Async(
                                        state_score.device_ptr(),
                                        f32::NEG_INFINITY.to_bits(),
                                        state_len,
                                        backend.stream().cu_stream(),
                                    )
                                }
                                .result()
                                .ok()?;
                                // SAFETY: the final four source rows and the
                                // rebuilt state/APE spans are validated above.
                                Some(unsafe {
                                    kernels.compressor_set_rows_tensor(
                                        backend.stream(),
                                        kv.device_ptr(),
                                        sc.device_ptr(),
                                        state_kv.device_ptr(),
                                        state_score.device_ptr(),
                                        ape_ptr,
                                        input_elements,
                                        state_elements,
                                        ape_elements,
                                        ape_type,
                                        width,
                                        4,
                                        pos0,
                                        n_tokens - 4,
                                        0,
                                        4,
                                        tail_grid_blocks,
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_decode_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    comp_kv: *const Ds4GpuTensor,
    n_comp: u32,
    comp_mask: *const Ds4GpuTensor,
    use_mask: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| {
        let Some(heads) = (unsafe { tensor_ref(heads.cast_const()) }) else {
            return false;
        };
        let Some(q) = (unsafe { tensor_ref(q) }) else {
            return false;
        };
        let Some(raw_kv) = (unsafe { tensor_ref(raw_kv) }) else {
            return false;
        };
        let comp_kv = if n_comp != 0 {
            let Some(comp_kv) = (unsafe { tensor_ref(comp_kv) }) else {
                return false;
            };
            comp_kv
        } else {
            raw_kv
        };
        let comp_mask = if use_mask != 0 {
            let Some(comp_mask) = (unsafe { tensor_ref(comp_mask) }) else {
                return false;
            };
            comp_mask
        } else {
            raw_kv
        };
        let Some(output_elements) = u64::from(n_head).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(raw_elements) = u64::from(raw_cap).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(comp_elements) = u64::from(n_comp).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let sink_elements = u64::from(n_head);
        let mask_elements = u64::from(n_comp);
        let Some(output_bytes) = output_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(raw_bytes) = raw_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(sink_bytes) = sink_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(mask_bytes) = mask_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_raw == 0
            || raw_cap < n_raw
            || raw_start >= raw_cap
            || n_head == 0
            || head_dim == 0
            || sinks_offset > model_size
            || sink_bytes > model_size - sinks_offset
            || heads.bytes < output_bytes
            || q.bytes < output_bytes
            || raw_kv.bytes < raw_bytes
            || (n_comp != 0 && comp_kv.bytes < comp_bytes)
            || (use_mask != 0 && comp_mask.bytes < mask_bytes)
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                sinks_offset,
                sink_bytes,
                |sinks_ptr| {
                    with_abi_kernels(backend, |kernels| {
                        if n_comp > DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP
                        {
                            if use_mask != 0
                                || head_dim != 512
                                || std::env::var_os("DS4_CUDA_NO_WINDOW_ATTENTION").is_some()
                            {
                                return Some(false);
                            }
                            // SAFETY: this preserves current-C overflow
                            // dispatch after validating all public spans.
                            return Some(unsafe {
                                kernels.attention_decode_mixed_heads8_online_tensor(
                                    backend.stream(),
                                    heads.device_ptr(),
                                    sinks_ptr,
                                    q.device_ptr(),
                                    raw_kv.device_ptr(),
                                    comp_kv.device_ptr(),
                                    output_elements,
                                    sink_elements,
                                    raw_elements,
                                    comp_elements,
                                    1,
                                    0,
                                    n_raw,
                                    raw_cap,
                                    raw_start,
                                    n_comp,
                                    0,
                                    0,
                                    n_head,
                                    head_dim,
                                )
                            });
                        }
                        // SAFETY: score-cap-bounded generic decode and each
                        // model/tensor span have been validated above.
                        Some(unsafe {
                            kernels.attention_decode_mixed_tensor(
                                backend.stream(),
                                heads.device_ptr(),
                                sinks_ptr,
                                q.device_ptr(),
                                raw_kv.device_ptr(),
                                comp_kv.device_ptr(),
                                comp_mask.device_ptr(),
                                output_elements,
                                sink_elements,
                                raw_elements,
                                comp_elements,
                                mask_elements,
                                1,
                                0,
                                n_raw,
                                raw_cap,
                                raw_start,
                                n_comp,
                                0,
                                0,
                                use_mask,
                                n_head,
                                head_dim,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_decode_raw_batch_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    window: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| unsafe {
        attention_decode_batch_impl(
            heads,
            model_map,
            model_size,
            sinks_offset,
            q,
            raw_kv,
            ptr::null(),
            ptr::null(),
            0,
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            0,
            window,
            1,
            n_head,
            head_dim,
        )
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_decode_mixed_batch_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    comp_kv: *const Ds4GpuTensor,
    comp_mask: *const Ds4GpuTensor,
    use_comp_mask: u32,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| unsafe {
        attention_decode_batch_impl(
            heads,
            model_map,
            model_size,
            sinks_offset,
            q,
            raw_kv,
            comp_kv,
            comp_mask,
            use_comp_mask,
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_decode_batch_impl(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    comp_kv: *const Ds4GpuTensor,
    comp_mask: *const Ds4GpuTensor,
    use_comp_mask: u32,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> bool {
    let Some(heads) = (unsafe { tensor_ref(heads.cast_const()) }) else {
        return false;
    };
    let Some(q) = (unsafe { tensor_ref(q) }) else {
        return false;
    };
    let Some(raw_kv) = (unsafe { tensor_ref(raw_kv) }) else {
        return false;
    };
    let comp_kv = if n_comp != 0 {
        let Some(comp_kv) = (unsafe { tensor_ref(comp_kv) }) else {
            return false;
        };
        comp_kv
    } else {
        raw_kv
    };
    let comp_mask = if use_comp_mask != 0 {
        let Some(comp_mask) = (unsafe { tensor_ref(comp_mask) }) else {
            return false;
        };
        comp_mask
    } else {
        raw_kv
    };
    let Some(output_elements) = u64::from(n_tokens)
        .checked_mul(u64::from(n_head))
        .and_then(|value| value.checked_mul(u64::from(head_dim)))
    else {
        return false;
    };
    let Some(raw_elements) = u64::from(raw_cap).checked_mul(u64::from(head_dim)) else {
        return false;
    };
    let Some(comp_elements) = u64::from(n_comp).checked_mul(u64::from(head_dim)) else {
        return false;
    };
    let Some(mask_elements) = u64::from(n_tokens).checked_mul(u64::from(n_comp)) else {
        return false;
    };
    let sink_elements = u64::from(n_head);
    let Some(output_bytes) = output_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(raw_bytes) = raw_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(mask_bytes) = mask_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(sink_bytes) = sink_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    if model_map.is_null()
        || n_tokens == 0
        || n_raw == 0
        || raw_cap < n_raw
        || raw_start >= raw_cap
        || (n_comp != 0 && ratio == 0)
        || n_head == 0
        || head_dim == 0
        || sinks_offset > model_size
        || sink_bytes > model_size - sinks_offset
        || heads.bytes < output_bytes
        || q.bytes < output_bytes
        || raw_kv.bytes < raw_bytes
        || (n_comp != 0 && comp_kv.bytes < comp_bytes)
        || (use_comp_mask != 0 && comp_mask.bytes < mask_bytes)
    {
        return false;
    }
    with_backend(|backend| {
        with_cached_abi_model_range(
            backend,
            model_map,
            model_size,
            sinks_offset,
            sink_bytes,
            |sinks_ptr| {
                with_abi_kernels(backend, |kernels| {
                    let no_window_attention =
                        std::env::var_os("DS4_CUDA_NO_WINDOW_ATTENTION").is_some();
                    if n_comp > DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP {
                        if use_comp_mask != 0 || head_dim != 512 || no_window_attention {
                            return Some(false);
                        }
                        // SAFETY: every public span and the current-C
                        // overflow-online dispatch preconditions are checked.
                        return Some(unsafe {
                            kernels.attention_decode_mixed_heads8_online_tensor(
                                backend.stream(),
                                heads.device_ptr(),
                                sinks_ptr,
                                q.device_ptr(),
                                raw_kv.device_ptr(),
                                comp_kv.device_ptr(),
                                output_elements,
                                sink_elements,
                                raw_elements,
                                comp_elements,
                                n_tokens,
                                pos0,
                                n_raw,
                                raw_cap,
                                raw_start,
                                n_comp,
                                window,
                                ratio,
                                n_head,
                                head_dim,
                            )
                        });
                    }
                    if use_comp_mask == 0
                        && n_tokens > 1
                        && head_dim == 512
                        && !no_window_attention
                        && (std::env::var_os("DS4_CUDA_WINDOW_ATTENTION").is_some()
                            || (!ABI_QUALITY_MODE.load(Ordering::Relaxed) && n_tokens >= 128))
                    {
                        // SAFETY: this is the current-C online window branch
                        // after validating every public tensor/model span.
                        return Some(unsafe {
                            kernels.attention_decode_mixed_heads8_online_tensor(
                                backend.stream(),
                                heads.device_ptr(),
                                sinks_ptr,
                                q.device_ptr(),
                                raw_kv.device_ptr(),
                                comp_kv.device_ptr(),
                                output_elements,
                                sink_elements,
                                raw_elements,
                                comp_elements,
                                n_tokens,
                                pos0,
                                n_raw,
                                raw_cap,
                                raw_start,
                                n_comp,
                                window,
                                ratio,
                                n_head,
                                head_dim,
                            )
                        });
                    }
                    // SAFETY: generic decode is score-cap bounded and every
                    // public tensor/model span has been validated.
                    Some(unsafe {
                        kernels.attention_decode_mixed_tensor(
                            backend.stream(),
                            heads.device_ptr(),
                            sinks_ptr,
                            q.device_ptr(),
                            raw_kv.device_ptr(),
                            comp_kv.device_ptr(),
                            comp_mask.device_ptr(),
                            output_elements,
                            sink_elements,
                            raw_elements,
                            comp_elements,
                            mask_elements,
                            n_tokens,
                            pos0,
                            n_raw,
                            raw_cap,
                            raw_start,
                            n_comp,
                            window,
                            ratio,
                            use_comp_mask,
                            n_head,
                            head_dim,
                        )
                    })
                })
            },
        )
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
unsafe fn hc_weighted_sum_impl(
    out: &Ds4GpuTensor,
    residual_hc: &Ds4GpuTensor,
    weights: &Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
    weight_stride: u32,
) -> bool {
    let Some(out_bytes_per_token) = u64::from(n_embd).checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    if out_bytes_per_token == 0 || n_hc == 0 || weight_stride < n_hc {
        return false;
    }
    let n_tokens = out.bytes / out_bytes_per_token;
    let Ok(n_tokens_u32) = u32::try_from(n_tokens) else {
        return false;
    };
    if n_tokens_u32 == 0 {
        return false;
    }
    let Some(residual_elements) = n_tokens
        .checked_mul(u64::from(n_hc))
        .and_then(|value| value.checked_mul(u64::from(n_embd)))
    else {
        return false;
    };
    let Some(residual_bytes) = residual_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(weight_elements) = (n_tokens - 1)
        .checked_mul(u64::from(weight_stride))
        .and_then(|value| value.checked_add(u64::from(n_hc)))
    else {
        return false;
    };
    let Some(weight_bytes) = weight_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    if residual_hc.bytes < residual_bytes || weights.bytes < weight_bytes {
        return false;
    }
    with_backend(|backend| {
        with_abi_kernels(backend, |kernels| {
            // SAFETY: bounds above cover the token-strided residual and
            // weight accesses for the current-C weighted-sum contract.
            Some(unsafe {
                kernels.hc_weighted_sum_tensor(
                    backend.stream(),
                    out.device_ptr(),
                    residual_hc.device_ptr(),
                    weights.device_ptr(),
                    n_embd,
                    n_hc,
                    n_tokens_u32,
                    weight_stride,
                )
            })
        })
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    comp_kv: *const Ds4GpuTensor,
    topk: *const Ds4GpuTensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    top_k: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| unsafe {
        let Some(heads) = tensor_ref(heads.cast_const()) else {
            return false;
        };
        let Some(q) = tensor_ref(q) else {
            return false;
        };
        let Some(raw_kv) = tensor_ref(raw_kv) else {
            return false;
        };
        let Some(comp_kv) = tensor_ref(comp_kv) else {
            return false;
        };
        let Some(topk) = tensor_ref(topk) else {
            return false;
        };
        let Some(output_elements) = u64::from(n_tokens)
            .checked_mul(u64::from(n_head))
            .and_then(|value| value.checked_mul(u64::from(head_dim)))
        else {
            return false;
        };
        let Some(raw_elements) = u64::from(raw_cap).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(comp_elements) = u64::from(n_comp).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let Some(topk_elements) = u64::from(n_tokens).checked_mul(u64::from(top_k)) else {
            return false;
        };
        let sink_elements = u64::from(n_head);
        let Some(output_bytes) = output_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(raw_bytes) = raw_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(topk_bytes) = topk_elements.checked_mul(size_of::<i32>() as u64) else {
            return false;
        };
        let Some(sink_bytes) = sink_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_tokens == 0
            || n_raw == 0
            || raw_cap < n_raw
            || raw_start >= raw_cap
            || n_comp == 0
            || top_k == 0
            || top_k > 512
            || n_head == 0
            || head_dim == 0
            || sinks_offset > model_size
            || sink_bytes > model_size - sinks_offset
            || heads.bytes < output_bytes
            || q.bytes < output_bytes
            || raw_kv.bytes < raw_bytes
            || comp_kv.bytes < comp_bytes
            || topk.bytes < topk_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                sinks_offset,
                sink_bytes,
                |sinks_ptr| {
                    with_abi_kernels(backend, |kernels| {
                        let launch = |topk_ptr: u64| {
                            if n_tokens > 1
                                && head_dim == 512
                                && std::env::var_os("DS4_CUDA_NO_INDEXED_HEADS8").is_none()
                            {
                                if std::env::var_os("DS4_CUDA_INDEXED_TWOPASS").is_none() {
                                    kernels.attention_indexed_mixed_heads8_online_tensor(
                                        backend.stream(),
                                        heads.device_ptr(),
                                        sinks_ptr,
                                        q.device_ptr(),
                                        raw_kv.device_ptr(),
                                        comp_kv.device_ptr(),
                                        topk_ptr,
                                        output_elements,
                                        sink_elements,
                                        raw_elements,
                                        comp_elements,
                                        topk_elements,
                                        n_tokens,
                                        pos0,
                                        n_raw,
                                        raw_cap,
                                        raw_start,
                                        n_comp,
                                        top_k,
                                        window,
                                        ratio,
                                        n_head,
                                        head_dim,
                                    )
                                } else {
                                    kernels.attention_indexed_mixed_heads8_rb4_tensor(
                                        backend.stream(),
                                        heads.device_ptr(),
                                        sinks_ptr,
                                        q.device_ptr(),
                                        raw_kv.device_ptr(),
                                        comp_kv.device_ptr(),
                                        topk_ptr,
                                        output_elements,
                                        sink_elements,
                                        raw_elements,
                                        comp_elements,
                                        topk_elements,
                                        n_tokens,
                                        pos0,
                                        n_raw,
                                        raw_cap,
                                        raw_start,
                                        n_comp,
                                        top_k,
                                        window,
                                        ratio,
                                        n_head,
                                        head_dim,
                                    )
                                }
                            } else {
                                kernels.attention_indexed_mixed_tensor(
                                    backend.stream(),
                                    heads.device_ptr(),
                                    sinks_ptr,
                                    q.device_ptr(),
                                    raw_kv.device_ptr(),
                                    comp_kv.device_ptr(),
                                    topk_ptr,
                                    output_elements,
                                    sink_elements,
                                    raw_elements,
                                    comp_elements,
                                    topk_elements,
                                    n_tokens,
                                    pos0,
                                    n_raw,
                                    raw_cap,
                                    raw_start,
                                    n_comp,
                                    top_k,
                                    window,
                                    ratio,
                                    n_head,
                                    head_dim,
                                )
                            }
                        };
                        if n_tokens > 1
                            && top_k == 512
                            && std::env::var_os("DS4_CUDA_NO_INDEXED_TOPK_SORT").is_none()
                        {
                            let elements = usize::try_from(topk_elements).ok()?;
                            with_abi_indexed_topk_sort_scratch(backend, elements, |sorted| {
                                if !kernels.indexed_topk_sort_512_asc_tensor(
                                    backend.stream(),
                                    topk.device_ptr(),
                                    sorted.cu_deviceptr(),
                                    topk_elements,
                                    n_tokens,
                                ) {
                                    return Some(false);
                                }
                                Some(launch(sorted.cu_deviceptr()))
                            })
                        } else {
                            Some(launch(topk.device_ptr()))
                        }
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_prefill_raw_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    n_tokens: u32,
    window: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| {
        let Some(heads) = (unsafe { tensor_ref(heads.cast_const()) }) else {
            return false;
        };
        let Some(q) = (unsafe { tensor_ref(q) }) else {
            return false;
        };
        let Some(raw_kv) = (unsafe { tensor_ref(raw_kv) }) else {
            return false;
        };
        let Some(output_elements) = u64::from(n_tokens)
            .checked_mul(u64::from(n_head))
            .and_then(|value| value.checked_mul(u64::from(head_dim)))
        else {
            return false;
        };
        let Some(raw_elements) = u64::from(n_tokens).checked_mul(u64::from(head_dim)) else {
            return false;
        };
        let sink_elements = u64::from(n_head);
        let Some(output_bytes) = output_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(raw_bytes) = raw_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(sink_bytes) = sink_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_tokens == 0
            || n_head == 0
            || head_dim == 0
            || window > 256
            || sinks_offset > model_size
            || sink_bytes > model_size - sinks_offset
            || heads.bytes < output_bytes
            || q.bytes < output_bytes
            || raw_kv.bytes < raw_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                sinks_offset,
                sink_bytes,
                |sinks_ptr| {
                    let blas = backend.blas_handle().ok();
                    let path = select_attention_prefill_path(AttentionPrefillDispatchOptions {
                        use_comp_mask: false,
                        n_tokens,
                        head_dim,
                        cublas_ready: blas.is_some(),
                        no_cublas_attention: std::env::var_os("DS4_CUDA_NO_CUBLAS_ATTENTION")
                            .is_some(),
                        no_window_attention: std::env::var_os("DS4_CUDA_NO_WINDOW_ATTENTION")
                            .is_some(),
                        window_attention: std::env::var_os("DS4_CUDA_WINDOW_ATTENTION").is_some(),
                        quality_mode: ABI_QUALITY_MODE.load(Ordering::Relaxed),
                    });
                    match path {
                        AttentionPrefillPath::StaticHeads8Online => {
                            with_abi_kernels(backend, |kernels| {
                                Some(unsafe {
                                    kernels.attention_static_mixed_heads8_online_tensor(
                                        backend.stream(),
                                        heads.device_ptr(),
                                        sinks_ptr,
                                        q.device_ptr(),
                                        raw_kv.device_ptr(),
                                        raw_kv.device_ptr(),
                                        output_elements,
                                        sink_elements,
                                        raw_elements,
                                        raw_elements,
                                        n_tokens,
                                        0,
                                        window,
                                        1,
                                        n_head,
                                        head_dim,
                                    )
                                })
                            })
                        }
                        AttentionPrefillPath::Cublas => Some(unsafe {
                            attention_prefill_cublas_impl(
                                backend,
                                blas.as_ref()?,
                                heads,
                                sinks_ptr,
                                q,
                                raw_kv,
                                raw_kv,
                                raw_kv,
                                false,
                                n_tokens,
                                0,
                                window,
                                1,
                                n_head,
                                head_dim,
                            )
                        }),
                        AttentionPrefillPath::Generic => with_abi_kernels(backend, |kernels| {
                            Some(unsafe {
                                kernels.attention_prefill_raw_tensor(
                                    backend.stream(),
                                    heads.device_ptr(),
                                    sinks_ptr,
                                    q.device_ptr(),
                                    raw_kv.device_ptr(),
                                    output_elements,
                                    sink_elements,
                                    raw_elements,
                                    n_tokens,
                                    window,
                                    n_head,
                                    head_dim,
                                )
                            })
                        }),
                    }
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_prefill_static_mixed_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    comp_kv: *const Ds4GpuTensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| unsafe {
        attention_prefill_mixed_impl(
            heads,
            model_map,
            model_size,
            sinks_offset,
            q,
            raw_kv,
            comp_kv,
            ptr::null(),
            false,
            n_tokens,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    comp_kv: *const Ds4GpuTensor,
    comp_mask: *const Ds4GpuTensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> c_int {
    status(|| unsafe {
        attention_prefill_mixed_impl(
            heads,
            model_map,
            model_size,
            sinks_offset,
            q,
            raw_kv,
            comp_kv,
            comp_mask,
            true,
            n_tokens,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_prefill_mixed_impl(
    heads: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    sinks_offset: u64,
    q: *const Ds4GpuTensor,
    raw_kv: *const Ds4GpuTensor,
    comp_kv: *const Ds4GpuTensor,
    comp_mask: *const Ds4GpuTensor,
    use_comp_mask: bool,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> bool {
    let Some(heads) = (unsafe { tensor_ref(heads.cast_const()) }) else {
        return false;
    };
    let Some(q) = (unsafe { tensor_ref(q) }) else {
        return false;
    };
    let Some(raw_kv) = (unsafe { tensor_ref(raw_kv) }) else {
        return false;
    };
    let comp_kv = if n_comp != 0 {
        let Some(comp_kv) = (unsafe { tensor_ref(comp_kv) }) else {
            return false;
        };
        comp_kv
    } else {
        raw_kv
    };
    let comp_mask = if use_comp_mask {
        let Some(comp_mask) = (unsafe { tensor_ref(comp_mask) }) else {
            return false;
        };
        comp_mask
    } else {
        raw_kv
    };
    let Some(output_elements) = u64::from(n_tokens)
        .checked_mul(u64::from(n_head))
        .and_then(|value| value.checked_mul(u64::from(head_dim)))
    else {
        return false;
    };
    let Some(raw_elements) = u64::from(n_tokens).checked_mul(u64::from(head_dim)) else {
        return false;
    };
    let Some(comp_elements) = u64::from(n_comp).checked_mul(u64::from(head_dim)) else {
        return false;
    };
    let Some(mask_elements) = u64::from(n_tokens).checked_mul(u64::from(n_comp)) else {
        return false;
    };
    let sink_elements = u64::from(n_head);
    let Some(output_bytes) = output_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(raw_bytes) = raw_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(comp_bytes) = comp_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(mask_bytes) = mask_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(sink_bytes) = sink_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    if model_map.is_null()
        || n_tokens == 0
        || ratio == 0
        || n_head == 0
        || head_dim == 0
        || n_tokens.checked_add(n_comp).is_none()
        || sinks_offset > model_size
        || sink_bytes > model_size - sinks_offset
        || heads.bytes < output_bytes
        || q.bytes < output_bytes
        || raw_kv.bytes < raw_bytes
        || (n_comp != 0 && comp_kv.bytes < comp_bytes)
        || (use_comp_mask && comp_mask.bytes < mask_bytes)
    {
        return false;
    }
    with_backend(|backend| {
        with_cached_abi_model_range(
            backend,
            model_map,
            model_size,
            sinks_offset,
            sink_bytes,
            |sinks_ptr| {
                let blas = backend.blas_handle().ok();
                let path = select_attention_prefill_path(AttentionPrefillDispatchOptions {
                    use_comp_mask,
                    n_tokens,
                    head_dim,
                    cublas_ready: blas.is_some(),
                    no_cublas_attention: std::env::var_os("DS4_CUDA_NO_CUBLAS_ATTENTION").is_some(),
                    no_window_attention: std::env::var_os("DS4_CUDA_NO_WINDOW_ATTENTION").is_some(),
                    window_attention: std::env::var_os("DS4_CUDA_WINDOW_ATTENTION").is_some(),
                    quality_mode: ABI_QUALITY_MODE.load(Ordering::Relaxed),
                });
                match path {
                    AttentionPrefillPath::StaticHeads8Online => {
                        with_abi_kernels(backend, |kernels| {
                            Some(unsafe {
                                kernels.attention_static_mixed_heads8_online_tensor(
                                    backend.stream(),
                                    heads.device_ptr(),
                                    sinks_ptr,
                                    q.device_ptr(),
                                    raw_kv.device_ptr(),
                                    comp_kv.device_ptr(),
                                    output_elements,
                                    sink_elements,
                                    raw_elements,
                                    comp_elements,
                                    n_tokens,
                                    n_comp,
                                    window,
                                    ratio,
                                    n_head,
                                    head_dim,
                                )
                            })
                        })
                    }
                    AttentionPrefillPath::Cublas => Some(unsafe {
                        attention_prefill_cublas_impl(
                            backend,
                            blas.as_ref()?,
                            heads,
                            sinks_ptr,
                            q,
                            raw_kv,
                            comp_kv,
                            comp_mask,
                            use_comp_mask,
                            n_tokens,
                            n_comp,
                            window,
                            ratio,
                            n_head,
                            head_dim,
                        )
                    }),
                    AttentionPrefillPath::Generic => with_abi_kernels(backend, |kernels| {
                        Some(unsafe {
                            kernels.attention_prefill_mixed_tensor(
                                backend.stream(),
                                heads.device_ptr(),
                                sinks_ptr,
                                q.device_ptr(),
                                raw_kv.device_ptr(),
                                comp_kv.device_ptr(),
                                comp_mask.device_ptr(),
                                output_elements,
                                sink_elements,
                                raw_elements,
                                comp_elements,
                                mask_elements,
                                n_tokens,
                                n_comp,
                                window,
                                ratio,
                                u32::from(use_comp_mask),
                                n_head,
                                head_dim,
                            )
                        })
                    }),
                }
            },
        )
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_prefill_cublas_impl(
    backend: &CudaOxideSubstrate,
    blas: &cuda_core::Blas,
    heads: &Ds4GpuTensor,
    sinks_ptr: u64,
    q: &Ds4GpuTensor,
    raw_kv: &Ds4GpuTensor,
    comp_kv: &Ds4GpuTensor,
    comp_mask: &Ds4GpuTensor,
    use_comp_mask: bool,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> bool {
    let Some(n_keys) = n_tokens.checked_add(n_comp) else {
        return false;
    };
    let Some(kv_elements) = u64::from(n_keys).checked_mul(u64::from(head_dim)) else {
        return false;
    };
    let Some(output_elements) = u64::from(n_tokens)
        .checked_mul(u64::from(n_head))
        .and_then(|value| value.checked_mul(u64::from(head_dim)))
    else {
        return false;
    };
    let Some(head_kv_elements) = u64::from(n_head).checked_mul(kv_elements) else {
        return false;
    };
    let Some(score_elements) = u64::from(n_head)
        .checked_mul(u64::from(n_tokens))
        .and_then(|value| value.checked_mul(u64::from(n_keys)))
    else {
        return false;
    };
    let Ok(kv_count) = usize::try_from(kv_elements) else {
        return false;
    };
    let Ok(output_count) = usize::try_from(output_elements) else {
        return false;
    };
    let Ok(head_kv_count) = usize::try_from(head_kv_elements) else {
        return false;
    };
    let Ok(score_count) = usize::try_from(score_elements) else {
        return false;
    };
    let Ok(n_tokens_usize) = usize::try_from(n_tokens) else {
        return false;
    };
    let Ok(n_keys_usize) = usize::try_from(n_keys) else {
        return false;
    };
    let Ok(n_head_usize) = usize::try_from(n_head) else {
        return false;
    };
    let Ok(head_dim_usize) = usize::try_from(head_dim) else {
        return false;
    };
    if !apply_abi_blas_math(blas) {
        return false;
    }
    with_abi_attention_prefill_cublas_scratch(
        backend,
        kv_count,
        output_count,
        head_kv_count,
        score_count,
        |scratch| {
            with_abi_kernels(backend, |kernels| {
                if !unsafe {
                    kernels.attention_prefill_pack_mixed_kv_tensor(
                        backend.stream(),
                        raw_kv.device_ptr(),
                        comp_kv.device_ptr(),
                        scratch.kv.cu_deviceptr(),
                        n_tokens,
                        n_comp,
                        head_dim,
                    ) && kernels.attention_prefill_pack_q_heads_tensor(
                        backend.stream(),
                        q.device_ptr(),
                        scratch.q_heads.cu_deviceptr(),
                        n_tokens,
                        n_head,
                        head_dim,
                    ) && kernels.attention_prefill_replicate_kv_tensor(
                        backend.stream(),
                        scratch.kv.cu_deviceptr(),
                        scratch.keys.cu_deviceptr(),
                        scratch.keys_transposed.cu_deviceptr(),
                        n_keys,
                        n_head,
                        head_dim,
                    )
                } {
                    return Some(false);
                }
                let Ok(mut score_config) = StridedBatchedSgemmConfig::packed(
                    n_tokens_usize,
                    n_keys_usize,
                    head_dim_usize,
                    n_head_usize,
                ) else {
                    return Some(false);
                };
                score_config.alpha = 1.0 / (head_dim as f32).sqrt();
                if blas
                    .sgemm_strided_batched(
                        backend.stream(),
                        score_config,
                        &scratch.q_heads,
                        &scratch.keys_transposed,
                        &mut scratch.scores,
                    )
                    .is_err()
                {
                    return Some(false);
                }
                let softmax_ok = if n_comp == 0 {
                    unsafe {
                        kernels.attention_prefill_raw_softmax_tensor(
                            backend.stream(),
                            sinks_ptr,
                            scratch.scores.cu_deviceptr(),
                            n_tokens,
                            window,
                            n_keys,
                            n_head,
                        )
                    }
                } else {
                    unsafe {
                        kernels.attention_prefill_mixed_softmax_tensor(
                            backend.stream(),
                            sinks_ptr,
                            comp_mask.device_ptr(),
                            scratch.scores.cu_deviceptr(),
                            n_tokens,
                            n_comp,
                            window,
                            ratio,
                            n_keys,
                            n_head,
                            u32::from(use_comp_mask),
                        )
                    }
                };
                if !softmax_ok {
                    return Some(false);
                }
                let Ok(value_config) = StridedBatchedSgemmConfig::packed(
                    n_tokens_usize,
                    head_dim_usize,
                    n_keys_usize,
                    n_head_usize,
                ) else {
                    return Some(false);
                };
                if blas
                    .sgemm_strided_batched(
                        backend.stream(),
                        value_config,
                        &scratch.scores,
                        &scratch.keys,
                        &mut scratch.output_by_head,
                    )
                    .is_err()
                {
                    return Some(false);
                }
                Some(unsafe {
                    kernels.attention_prefill_unpack_heads_tensor(
                        backend.stream(),
                        scratch.output_by_head.cu_deviceptr(),
                        heads.device_ptr(),
                        n_tokens,
                        n_head,
                        head_dim,
                    )
                })
            })
        },
    )
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_output_low_q8_tensor(
    low: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    out_a_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    heads: *const Ds4GpuTensor,
) -> c_int {
    status(|| {
        let Some(low) = (unsafe { tensor_ref(low.cast_const()) }) else {
            return false;
        };
        let Some(heads) = (unsafe { tensor_ref(heads) }) else {
            return false;
        };
        let Some(low_dim) = u64::from(n_groups).checked_mul(rank) else {
            return false;
        };
        let Some((_weight_elements, _weight_elements_usize, weight_bytes)) =
            abi_q8_shape(group_dim, low_dim)
        else {
            return false;
        };
        let Some(head_elements) = u64::from(n_groups).checked_mul(group_dim) else {
            return false;
        };
        let Some(head_bytes) = head_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(low_bytes) = low_dim.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || group_dim == 0
            || rank == 0
            || n_groups == 0
            || out_a_offset > model_size
            || weight_bytes > model_size - out_a_offset
            || heads.bytes < head_bytes
            || low.bytes < low_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                out_a_offset,
                weight_bytes,
                |weight_ptr| {
                    let blocks = group_dim.div_ceil(32);
                    let Some(quantized_elements) = u64::from(n_groups)
                        .checked_mul(blocks)
                        .and_then(|value| value.checked_mul(32))
                    else {
                        return Some(false);
                    };
                    let Some(scale_elements) = u64::from(n_groups).checked_mul(blocks) else {
                        return Some(false);
                    };
                    let Some(quantized_elements) = usize::try_from(quantized_elements).ok() else {
                        return Some(false);
                    };
                    let Some(scale_elements) = usize::try_from(scale_elements).ok() else {
                        return Some(false);
                    };
                    with_abi_q8_activations(
                        backend,
                        quantized_elements,
                        scale_elements,
                        |activations| {
                            with_abi_kernels(backend, |kernels| {
                                if !unsafe {
                                    kernels.quantize_q8_f32_tensor(
                                        backend.stream(),
                                        heads.device_ptr(),
                                        activations.quantized.cu_deviceptr(),
                                        activations.scales.cu_deviceptr(),
                                        group_dim,
                                        blocks,
                                        u64::from(n_groups),
                                    )
                                } {
                                    return Some(false);
                                }
                                Some(unsafe {
                                    kernels.attention_output_low_q8_tensor(
                                        backend.stream(),
                                        low.device_ptr(),
                                        weight_ptr,
                                        activations.quantized.cu_deviceptr(),
                                        activations.scales.cu_deviceptr(),
                                        group_dim,
                                        rank,
                                        n_groups,
                                        1,
                                        q8_dp4a_enabled(
                                            std::env::var_os("DS4_CUDA_NO_Q8_DP4A").is_some(),
                                        ),
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
fn attention_output_a_cublas_min_tokens() -> u32 {
    let Ok(value) = std::env::var("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN") else {
        return 2;
    };
    let Ok(value) = CString::new(value) else {
        return 2;
    };
    let mut end = ptr::null_mut();
    // SAFETY: `value` is NUL-terminated and `end` is valid for `strtol`.
    let parsed = unsafe { libc::strtol(value.as_ptr(), &mut end, 10) };
    if end.cast_const() != value.as_ptr() && parsed > 1 && parsed < 4096 {
        parsed as u32
    } else {
        2
    }
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_output_a_native_impl(
    backend: &CudaOxideSubstrate,
    low: &Ds4GpuTensor,
    heads: &Ds4GpuTensor,
    weight_ptr: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    n_tokens: u32,
) -> bool {
    let blocks = group_dim.div_ceil(32);
    let Some(rows) = u64::from(n_tokens).checked_mul(u64::from(n_groups)) else {
        return false;
    };
    let Some(quantized_elements) = rows
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(32))
    else {
        return false;
    };
    let Some(scale_elements) = rows.checked_mul(blocks) else {
        return false;
    };
    let Some(quantized_elements) = usize::try_from(quantized_elements).ok() else {
        return false;
    };
    let Some(scale_elements) = usize::try_from(scale_elements).ok() else {
        return false;
    };
    with_abi_q8_activations(backend, quantized_elements, scale_elements, |activations| {
        with_abi_kernels(backend, |kernels| {
            if !unsafe {
                kernels.quantize_q8_f32_tensor(
                    backend.stream(),
                    heads.device_ptr(),
                    activations.quantized.cu_deviceptr(),
                    activations.scales.cu_deviceptr(),
                    group_dim,
                    blocks,
                    rows,
                )
            } {
                return Some(false);
            }
            Some(unsafe {
                kernels.attention_output_low_q8_tensor(
                    backend.stream(),
                    low.device_ptr(),
                    weight_ptr,
                    activations.quantized.cu_deviceptr(),
                    activations.scales.cu_deviceptr(),
                    group_dim,
                    rank,
                    n_groups,
                    n_tokens,
                    q8_dp4a_enabled(std::env::var_os("DS4_CUDA_NO_Q8_DP4A").is_some()),
                )
            })
        })
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_output_a_cublas_impl(
    backend: &CudaOxideSubstrate,
    blas: &cuda_core::Blas,
    low: &Ds4GpuTensor,
    heads: &Ds4GpuTensor,
    expanded_weight_ptr: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    n_tokens: u32,
) -> bool {
    let Some(packed_head_elements) = u64::from(n_tokens)
        .checked_mul(u64::from(n_groups))
        .and_then(|value| value.checked_mul(group_dim))
    else {
        return false;
    };
    let Some(transposed_weight_elements) = u64::from(n_groups)
        .checked_mul(rank)
        .and_then(|value| value.checked_mul(group_dim))
    else {
        return false;
    };
    let Some(packed_low_elements) = u64::from(n_tokens)
        .checked_mul(u64::from(n_groups))
        .and_then(|value| value.checked_mul(rank))
    else {
        return false;
    };
    let Some(packed_head_count) = usize::try_from(packed_head_elements).ok() else {
        return false;
    };
    let Some(transposed_weight_count) = usize::try_from(transposed_weight_elements).ok() else {
        return false;
    };
    let Some(packed_low_count) = usize::try_from(packed_low_elements).ok() else {
        return false;
    };
    let Some(n_tokens_usize) = usize::try_from(n_tokens).ok() else {
        return false;
    };
    let Some(n_groups_usize) = usize::try_from(n_groups).ok() else {
        return false;
    };
    let Some(group_dim_usize) = usize::try_from(group_dim).ok() else {
        return false;
    };
    let Some(rank_usize) = usize::try_from(rank).ok() else {
        return false;
    };
    if !apply_abi_blas_math(blas) {
        return false;
    }
    with_abi_f16_activations(backend, packed_head_count, |packed_heads_f16| {
        with_abi_attention_output_cublas_scratch(
            backend,
            packed_head_count,
            transposed_weight_count,
            packed_low_count,
            |scratch| {
                with_abi_kernels(backend, |kernels| {
                    if !unsafe {
                        kernels.attention_pack_group_heads_f16_tensor(
                            backend.stream(),
                            heads.device_ptr(),
                            packed_heads_f16.cu_deviceptr(),
                            n_tokens,
                            n_groups,
                            group_dim,
                        ) && kernels.f16_to_f32_tensor(
                            backend.stream(),
                            packed_heads_f16.cu_deviceptr(),
                            scratch.packed_heads.cu_deviceptr(),
                            packed_head_elements,
                        ) && kernels.attention_expand_group_weights_sgemm_tensor(
                            backend.stream(),
                            expanded_weight_ptr,
                            scratch.transposed_weights.cu_deviceptr(),
                            n_groups,
                            rank,
                            group_dim,
                        )
                    } {
                        return Some(false);
                    }
                    let Ok(config) = StridedBatchedSgemmConfig::packed(
                        n_tokens_usize,
                        rank_usize,
                        group_dim_usize,
                        n_groups_usize,
                    ) else {
                        return Some(false);
                    };
                    if blas
                        .sgemm_strided_batched(
                            backend.stream(),
                            config,
                            &scratch.packed_heads,
                            &scratch.transposed_weights,
                            &mut scratch.packed_low,
                        )
                        .is_err()
                    {
                        return Some(false);
                    }
                    Some(unsafe {
                        kernels.attention_unpack_group_low_tensor(
                            backend.stream(),
                            scratch.packed_low.cu_deviceptr(),
                            low.device_ptr(),
                            n_tokens,
                            n_groups,
                            rank,
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
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_attention_output_q8_batch_tensor(
    out: *mut Ds4GpuTensor,
    low: *mut Ds4GpuTensor,
    _group_tmp: *mut Ds4GpuTensor,
    _low_tmp: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    out_a_offset: u64,
    out_b_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    out_dim: u64,
    heads: *const Ds4GpuTensor,
    n_tokens: u32,
) -> c_int {
    status(|| {
        let Some(out_tensor) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(low_tensor) = (unsafe { tensor_ref(low.cast_const()) }) else {
            return false;
        };
        let Some(heads_tensor) = (unsafe { tensor_ref(heads) }) else {
            return false;
        };
        let Some(low_dim) = u64::from(n_groups).checked_mul(rank) else {
            return false;
        };
        let Some((_out_a_elements, _out_a_elements_usize, out_a_bytes)) =
            abi_q8_shape(group_dim, low_dim)
        else {
            return false;
        };
        let Some((_out_b_elements, _out_b_elements_usize, out_b_bytes)) =
            abi_q8_shape(low_dim, out_dim)
        else {
            return false;
        };
        let Some(head_bytes) = u64::from(n_tokens)
            .checked_mul(u64::from(n_groups))
            .and_then(|value| value.checked_mul(group_dim))
            .and_then(|value| value.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        let Some(low_bytes) = u64::from(n_tokens)
            .checked_mul(low_dim)
            .and_then(|value| value.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        let Some(out_bytes) = u64::from(n_tokens)
            .checked_mul(out_dim)
            .and_then(|value| value.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        if model_map.is_null()
            || group_dim == 0
            || rank == 0
            || n_groups == 0
            || out_dim == 0
            || n_tokens == 0
            || out_a_offset > model_size
            || out_b_offset > model_size
            || out_a_bytes > model_size - out_a_offset
            || out_b_bytes > model_size - out_b_offset
            || heads_tensor.bytes < head_bytes
            || low_tensor.bytes < low_bytes
            || out_tensor.bytes < out_bytes
        {
            return false;
        }
        let output_a_ok = with_backend(|backend| {
            let blas = backend.blas_handle().ok();
            let cublas_min_tokens = attention_output_a_cublas_min_tokens();
            if !ABI_QUALITY_MODE.load(Ordering::Relaxed)
                && blas.is_some()
                && n_tokens >= cublas_min_tokens
                && std::env::var_os("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A").is_none()
            {
                let _ = cache_abi_q8_f16_range(
                    backend,
                    model_map,
                    model_size,
                    out_a_offset,
                    out_a_bytes,
                    group_dim,
                    low_dim,
                    "attn_output_a",
                    false,
                );
            }
            let expanded_weight_ptr =
                abi_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim);
            let path = select_attention_output_a_path(AttentionOutputADispatchOptions {
                quality_mode: ABI_QUALITY_MODE.load(Ordering::Relaxed),
                cublas_ready: blas.is_some(),
                n_tokens,
                cublas_min_tokens,
                no_cublas_attention_output_a: std::env::var_os(
                    "DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A",
                )
                .is_some(),
                expanded_f16_ready: expanded_weight_ptr.is_some(),
            });
            match path {
                AttentionOutputAPath::CublasF16 => Some(unsafe {
                    attention_output_a_cublas_impl(
                        backend,
                        blas.as_ref()?,
                        low_tensor,
                        heads_tensor,
                        expanded_weight_ptr?,
                        group_dim,
                        rank,
                        n_groups,
                        n_tokens,
                    )
                }),
                AttentionOutputAPath::NativeQ8 => with_cached_abi_model_range(
                    backend,
                    model_map,
                    model_size,
                    out_a_offset,
                    out_a_bytes,
                    |weight_ptr| {
                        Some(unsafe {
                            attention_output_a_native_impl(
                                backend,
                                low_tensor,
                                heads_tensor,
                                weight_ptr,
                                group_dim,
                                rank,
                                n_groups,
                                n_tokens,
                            )
                        })
                    },
                ),
            }
        })
        .unwrap_or(false);
        output_a_ok
            && unsafe {
                matmul_q8_0_tensor_labeled(
                    out,
                    model_map,
                    model_size,
                    out_b_offset,
                    low_dim,
                    out_dim,
                    low.cast_const(),
                    u64::from(n_tokens),
                    "attn_output_b",
                ) != 0
            }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_hc_weighted_sum_tensor(
    out: *mut Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    weights: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(weights) = (unsafe { tensor_ref(weights) }) else {
            return false;
        };
        unsafe { hc_weighted_sum_impl(out, residual_hc, weights, n_embd, n_hc, n_hc) }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_hc_weighted_sum_split_tensor(
    out: *mut Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    split: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(split) = (unsafe { tensor_ref(split) }) else {
            return false;
        };
        let Some(weight_stride) = n_hc.checked_mul(n_hc).and_then(|comb| {
            n_hc.checked_mul(2)
                .and_then(|prefix| prefix.checked_add(comb))
        }) else {
            return false;
        };
        unsafe { hc_weighted_sum_impl(out, residual_hc, split, n_embd, n_hc, weight_stride) }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn hc_expand_impl(
    out_hc: &Ds4GpuTensor,
    block_out: &Ds4GpuTensor,
    block_add: Option<&Ds4GpuTensor>,
    has_add: bool,
    residual_hc: &Ds4GpuTensor,
    post_ptr: u64,
    post_bytes: u64,
    post_stride: u32,
    comb_ptr: u64,
    comb_bytes: u64,
    comb_stride: u32,
    n_embd: u32,
    n_hc: u32,
) -> bool {
    let Some(output_elements_per_token) = u64::from(n_embd).checked_mul(u64::from(n_hc)) else {
        return false;
    };
    let Some(output_bytes_per_token) =
        output_elements_per_token.checked_mul(size_of::<f32>() as u64)
    else {
        return false;
    };
    if output_bytes_per_token == 0 {
        return false;
    }
    let n_tokens = out_hc.bytes / output_bytes_per_token;
    let Ok(n_tokens_u32) = u32::try_from(n_tokens) else {
        return false;
    };
    if n_tokens_u32 == 0 {
        return false;
    }
    let Some(block_elements) = n_tokens.checked_mul(u64::from(n_embd)) else {
        return false;
    };
    let Some(block_bytes) = block_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(residual_elements) = n_tokens.checked_mul(output_elements_per_token) else {
        return false;
    };
    let Some(residual_bytes) = residual_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(post_elements) = (n_tokens - 1)
        .checked_mul(u64::from(post_stride))
        .and_then(|prefix| prefix.checked_add(u64::from(n_hc)))
    else {
        return false;
    };
    let Some(comb_width) = u64::from(n_hc).checked_mul(u64::from(n_hc)) else {
        return false;
    };
    let Some(comb_elements) = (n_tokens - 1)
        .checked_mul(u64::from(comb_stride))
        .and_then(|prefix| prefix.checked_add(comb_width))
    else {
        return false;
    };
    let Some(required_post_bytes) = post_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(required_comb_bytes) = comb_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let block_add = block_add.unwrap_or(block_out);
    if block_out.bytes < block_bytes
        || block_add.bytes < block_bytes
        || residual_hc.bytes < residual_bytes
        || post_bytes < required_post_bytes
        || comb_bytes < required_comb_bytes
    {
        return false;
    }
    with_backend(|backend| {
        with_abi_kernels(backend, |kernels| {
            // SAFETY: the validated spans include every token-strided post
            // and combination access and preserve current-C aliasing rules.
            Some(unsafe {
                kernels.hc_expand_tensor(
                    backend.stream(),
                    out_hc.device_ptr(),
                    block_out.device_ptr(),
                    block_add.device_ptr(),
                    residual_hc.device_ptr(),
                    post_ptr,
                    comb_ptr,
                    n_embd,
                    n_hc,
                    n_tokens_u32,
                    post_stride,
                    comb_stride,
                    has_add,
                )
            })
        })
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_hc_expand_tensor(
    out_hc: *mut Ds4GpuTensor,
    block_out: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    post: *const Ds4GpuTensor,
    comb: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(block_out) = (unsafe { tensor_ref(block_out) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(post) = (unsafe { tensor_ref(post) }) else {
            return false;
        };
        let Some(comb) = (unsafe { tensor_ref(comb) }) else {
            return false;
        };
        let Some(comb_stride) = n_hc.checked_mul(n_hc) else {
            return false;
        };
        unsafe {
            hc_expand_impl(
                out_hc,
                block_out,
                None,
                false,
                residual_hc,
                post.device_ptr(),
                post.bytes,
                n_hc,
                comb.device_ptr(),
                comb.bytes,
                comb_stride,
                n_embd,
                n_hc,
            )
        }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_hc_expand_split_tensor(
    out_hc: *mut Ds4GpuTensor,
    block_out: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    split: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(block_out) = (unsafe { tensor_ref(block_out) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(split) = (unsafe { tensor_ref(split) }) else {
            return false;
        };
        let Some(mix_hc) = n_hc.checked_mul(n_hc).and_then(|comb| {
            n_hc.checked_mul(2)
                .and_then(|prefix| prefix.checked_add(comb))
        }) else {
            return false;
        };
        let post_offset = u64::from(n_hc) * size_of::<f32>() as u64;
        let comb_offset = u64::from(n_hc) * 2 * size_of::<f32>() as u64;
        let Some((post_ptr, post_bytes)) =
            checked_range(split, post_offset, split.bytes.saturating_sub(post_offset))
        else {
            return false;
        };
        let Some((comb_ptr, comb_bytes)) =
            checked_range(split, comb_offset, split.bytes.saturating_sub(comb_offset))
        else {
            return false;
        };
        unsafe {
            hc_expand_impl(
                out_hc,
                block_out,
                None,
                false,
                residual_hc,
                post_ptr,
                post_bytes as u64,
                mix_hc,
                comb_ptr,
                comb_bytes as u64,
                mix_hc,
                n_embd,
                n_hc,
            )
        }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_hc_expand_add_split_tensor(
    out_hc: *mut Ds4GpuTensor,
    block_out: *const Ds4GpuTensor,
    block_add: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    split: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(block_out) = (unsafe { tensor_ref(block_out) }) else {
            return false;
        };
        let Some(block_add) = (unsafe { tensor_ref(block_add) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(split) = (unsafe { tensor_ref(split) }) else {
            return false;
        };
        let Some(mix_hc) = n_hc.checked_mul(n_hc).and_then(|comb| {
            n_hc.checked_mul(2)
                .and_then(|prefix| prefix.checked_add(comb))
        }) else {
            return false;
        };
        let post_offset = u64::from(n_hc) * size_of::<f32>() as u64;
        let comb_offset = u64::from(n_hc) * 2 * size_of::<f32>() as u64;
        let Some((post_ptr, post_bytes)) =
            checked_range(split, post_offset, split.bytes.saturating_sub(post_offset))
        else {
            return false;
        };
        let Some((comb_ptr, comb_bytes)) =
            checked_range(split, comb_offset, split.bytes.saturating_sub(comb_offset))
        else {
            return false;
        };
        unsafe {
            hc_expand_impl(
                out_hc,
                block_out,
                Some(block_add),
                true,
                residual_hc,
                post_ptr,
                post_bytes as u64,
                mix_hc,
                comb_ptr,
                comb_bytes as u64,
                mix_hc,
                n_embd,
                n_hc,
            )
        }
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
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_router_select_tensor(
    selected: *mut Ds4GpuTensor,
    weights: *mut Ds4GpuTensor,
    probs: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    token: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: bool,
    hash_mode: bool,
    logits: *const Ds4GpuTensor,
) -> c_int {
    status(|| {
        let Some(selected) = (unsafe { tensor_ref(selected.cast_const()) }) else {
            return false;
        };
        let Some(weights) = (unsafe { tensor_ref(weights.cast_const()) }) else {
            return false;
        };
        let Some(probs) = (unsafe { tensor_ref(probs.cast_const()) }) else {
            return false;
        };
        let Some(logits) = (unsafe { tensor_ref(logits) }) else {
            return false;
        };
        unsafe {
            router_select_impl(
                selected,
                weights,
                probs,
                model_map,
                model_size,
                bias_offset,
                hash_offset,
                hash_rows,
                token as i32,
                n_expert_groups,
                n_group_used,
                has_bias,
                hash_mode,
                logits,
                None,
                1,
            )
        }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_router_select_batch_tensor(
    selected: *mut Ds4GpuTensor,
    weights: *mut Ds4GpuTensor,
    probs: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: bool,
    hash_mode: bool,
    logits: *const Ds4GpuTensor,
    tokens: *const Ds4GpuTensor,
    n_tokens: u32,
) -> c_int {
    status(|| {
        let Some(selected) = (unsafe { tensor_ref(selected.cast_const()) }) else {
            return false;
        };
        let Some(weights) = (unsafe { tensor_ref(weights.cast_const()) }) else {
            return false;
        };
        let Some(probs) = (unsafe { tensor_ref(probs.cast_const()) }) else {
            return false;
        };
        let Some(logits) = (unsafe { tensor_ref(logits) }) else {
            return false;
        };
        let Some(tokens) = (unsafe { tensor_ref(tokens) }) else {
            return false;
        };
        unsafe {
            router_select_impl(
                selected,
                weights,
                probs,
                model_map,
                model_size,
                bias_offset,
                hash_offset,
                hash_rows,
                0,
                n_expert_groups,
                n_group_used,
                has_bias,
                hash_mode,
                logits,
                Some(tokens),
                n_tokens,
            )
        }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn router_select_impl(
    selected: &Ds4GpuTensor,
    weights: &Ds4GpuTensor,
    probs: &Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    token_scalar: i32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: bool,
    hash_mode: bool,
    logits: &Ds4GpuTensor,
    tokens: Option<&Ds4GpuTensor>,
    n_tokens: u32,
) -> bool {
    const N_EXPERT: u64 = 256;
    const TOP_K: u64 = 6;
    let Some(prob_elements) = u64::from(n_tokens).checked_mul(N_EXPERT) else {
        return false;
    };
    let Some(selected_elements) = u64::from(n_tokens).checked_mul(TOP_K) else {
        return false;
    };
    let Some(prob_bytes) = prob_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(selected_bytes) = selected_elements.checked_mul(size_of::<i32>() as u64) else {
        return false;
    };
    let Some(weight_bytes) = selected_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let token_bytes = u64::from(n_tokens) * size_of::<i32>() as u64;
    let hash_bytes = u64::from(hash_rows) * TOP_K * size_of::<i32>() as u64;
    if model_map.is_null()
        || n_tokens == 0
        || n_expert_groups > 1
        || n_group_used > 0
        || (hash_mode && hash_rows == 0)
        || logits.bytes < prob_bytes
        || probs.bytes < prob_bytes
        || selected.bytes < selected_bytes
        || weights.bytes < weight_bytes
        || tokens.is_some_and(|tokens| tokens.bytes < token_bytes)
    {
        return false;
    }

    with_backend(|backend| {
        let fallback_ptr = logits.device_ptr();
        if has_bias && !hash_mode {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                bias_offset,
                N_EXPERT * size_of::<f32>() as u64,
                |bias_ptr| unsafe {
                    launch_router_select(
                        backend,
                        selected,
                        weights,
                        probs,
                        bias_ptr,
                        fallback_ptr,
                        hash_rows,
                        token_scalar,
                        has_bias,
                        hash_mode,
                        logits,
                        tokens,
                        n_tokens,
                    )
                },
            )
        } else if hash_mode {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                hash_offset,
                hash_bytes,
                |hash_ptr| unsafe {
                    launch_router_select(
                        backend,
                        selected,
                        weights,
                        probs,
                        fallback_ptr,
                        hash_ptr,
                        hash_rows,
                        token_scalar,
                        has_bias,
                        hash_mode,
                        logits,
                        tokens,
                        n_tokens,
                    )
                },
            )
        } else {
            unsafe {
                launch_router_select(
                    backend,
                    selected,
                    weights,
                    probs,
                    fallback_ptr,
                    fallback_ptr,
                    hash_rows,
                    token_scalar,
                    has_bias,
                    hash_mode,
                    logits,
                    tokens,
                    n_tokens,
                )
            }
        }
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_router_select(
    backend: &CudaOxideSubstrate,
    selected: &Ds4GpuTensor,
    weights: &Ds4GpuTensor,
    probs: &Ds4GpuTensor,
    bias_ptr: u64,
    hash_ptr: u64,
    hash_rows: u32,
    token_scalar: i32,
    has_bias: bool,
    hash_mode: bool,
    logits: &Ds4GpuTensor,
    tokens: Option<&Ds4GpuTensor>,
    n_tokens: u32,
) -> Option<bool> {
    let tokens_ptr = tokens.map_or_else(|| logits.device_ptr(), Ds4GpuTensor::device_ptr);
    let path = select_router_select_path(RouterSelectDispatchOptions {
        no_warp_router_select: std::env::var_os("DS4_CUDA_NO_WARP_ROUTER_SELECT").is_some(),
        no_parallel_router_select: std::env::var_os("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT").is_some(),
    });
    with_abi_kernels(backend, |kernels| {
        let launched = match path {
            RouterSelectPath::WarpTopK => unsafe {
                kernels.router_select_warp_topk_tensor(
                    backend.stream(),
                    selected.device_ptr(),
                    weights.device_ptr(),
                    probs.device_ptr(),
                    bias_ptr,
                    hash_ptr,
                    logits.device_ptr(),
                    tokens_ptr,
                    n_tokens,
                    token_scalar,
                    hash_rows,
                    has_bias && !hash_mode,
                    hash_mode,
                    tokens.is_some(),
                )
            },
            RouterSelectPath::Parallel => unsafe {
                kernels.router_select_parallel_tensor(
                    backend.stream(),
                    selected.device_ptr(),
                    weights.device_ptr(),
                    probs.device_ptr(),
                    bias_ptr,
                    hash_ptr,
                    logits.device_ptr(),
                    tokens_ptr,
                    n_tokens,
                    token_scalar,
                    hash_rows,
                    has_bias && !hash_mode,
                    hash_mode,
                    tokens.is_some(),
                )
            },
            RouterSelectPath::Scalar => unsafe {
                kernels.router_select_scalar_tensor(
                    backend.stream(),
                    selected.device_ptr(),
                    weights.device_ptr(),
                    probs.device_ptr(),
                    bias_ptr,
                    hash_ptr,
                    logits.device_ptr(),
                    tokens_ptr,
                    n_tokens,
                    token_scalar,
                    hash_rows,
                    has_bias && !hash_mode,
                    hash_mode,
                    tokens.is_some(),
                )
            },
        };
        Some(launched)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_routed_moe_one_tensor(
    out: *mut Ds4GpuTensor,
    gate: *mut Ds4GpuTensor,
    up: *mut Ds4GpuTensor,
    mid: *mut Ds4GpuTensor,
    down: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: *const Ds4GpuTensor,
    weights: *const Ds4GpuTensor,
    n_expert: u32,
    clamp: f32,
    x: *const Ds4GpuTensor,
) -> c_int {
    status(|| {
        const MOE_EXPERT_COUNT: u64 = 256;
        const QK_K: u32 = 256;
        const Q8_K_BYTES: u64 = 292;

        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(gate) = (unsafe { tensor_ref(gate.cast_const()) }) else {
            return false;
        };
        let Some(up) = (unsafe { tensor_ref(up.cast_const()) }) else {
            return false;
        };
        let Some(mid) = (unsafe { tensor_ref(mid.cast_const()) }) else {
            return false;
        };
        let Some(down) = (unsafe { tensor_ref(down.cast_const()) }) else {
            return false;
        };
        let Some(selected) = (unsafe { tensor_ref(selected) }) else {
            return false;
        };
        let Some(weights) = (unsafe { tensor_ref(weights) }) else {
            return false;
        };
        let Some(x) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let q4_k = gate_type == 12 && down_type == 12;
        let Some(x_bytes) = u64::from(expert_in_dim).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(route_bytes) = u64::from(n_expert).checked_mul(size_of::<i32>() as u64) else {
            return false;
        };
        let Some(weight_bytes) = u64::from(n_expert).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(mid_elements) = u64::from(n_expert).checked_mul(u64::from(expert_mid_dim)) else {
            return false;
        };
        let Some(mid_bytes) = mid_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(down_elements) = u64::from(n_expert).checked_mul(u64::from(out_dim)) else {
            return false;
        };
        let Some(down_bytes) = down_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(out_bytes) = u64::from(out_dim).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(gate_model_bytes) = MOE_EXPERT_COUNT.checked_mul(gate_expert_bytes) else {
            return false;
        };
        let Some(down_model_bytes) = MOE_EXPERT_COUNT.checked_mul(down_expert_bytes) else {
            return false;
        };
        if model_map.is_null()
            || n_expert == 0
            || expert_in_dim % QK_K != 0
            || expert_mid_dim % QK_K != 0
            || (!q4_k && (gate_type != 16 || down_type != 10))
            || (q4_k && n_expert != 6)
            || gate_offset > model_size
            || up_offset > model_size
            || down_offset > model_size
            || gate_model_bytes > model_size - gate_offset
            || gate_model_bytes > model_size - up_offset
            || down_model_bytes > model_size - down_offset
            || x.bytes < x_bytes
            || selected.bytes < route_bytes
            || weights.bytes < weight_bytes
            || gate.bytes < mid_bytes
            || up.bytes < mid_bytes
            || mid.bytes < mid_bytes
            || down.bytes < down_bytes
            || out.bytes < out_bytes
        {
            return false;
        }
        let xq_blocks = expert_in_dim / QK_K;
        let midq_blocks = expert_mid_dim / QK_K;
        let Some(xq_bytes) = u64::from(xq_blocks).checked_mul(Q8_K_BYTES) else {
            return false;
        };
        let Some(midq_bytes) = u64::from(n_expert)
            .checked_mul(u64::from(midq_blocks))
            .and_then(|blocks| blocks.checked_mul(Q8_K_BYTES))
        else {
            return false;
        };
        let use_quantized = down.bytes >= xq_bytes && gate.bytes >= midq_bytes;

        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                gate_offset,
                gate_model_bytes,
                |gate_weights_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        up_offset,
                        gate_model_bytes,
                        |up_weights_ptr| {
                            with_cached_abi_model_range(
                                backend,
                                model_map,
                                model_size,
                                down_offset,
                                down_model_bytes,
                                |down_weights_ptr| {
                                    with_abi_moe_iq2_tables(backend, |tables| {
                                        with_abi_kernels(backend, |kernels| {
                                            if !use_quantized {
                                                if !unsafe {
                                                    kernels.moe_gate_up_mid_f32_tensor(
                                                        backend.stream(),
                                                        1,
                                                        gate.device_ptr(),
                                                        up.device_ptr(),
                                                        mid.device_ptr(),
                                                        gate_weights_ptr,
                                                        up_weights_ptr,
                                                        x.device_ptr(),
                                                        selected.device_ptr(),
                                                        weights.device_ptr(),
                                                        tables.grid.cu_deviceptr(),
                                                        tables.signs.cu_deviceptr(),
                                                        gate_model_bytes,
                                                        expert_in_dim,
                                                        expert_mid_dim,
                                                        n_expert,
                                                        gate_expert_bytes,
                                                        gate_row_bytes,
                                                        clamp,
                                                    )
                                                } {
                                                    return Some(false);
                                                }
                                                if !unsafe {
                                                    kernels.moe_down_f32_tensor(
                                                        backend.stream(),
                                                        1,
                                                        down.device_ptr(),
                                                        down_weights_ptr,
                                                        mid.device_ptr(),
                                                        selected.device_ptr(),
                                                        down_model_bytes,
                                                        expert_mid_dim,
                                                        out_dim,
                                                        n_expert,
                                                        down_expert_bytes,
                                                        down_row_bytes,
                                                    )
                                                } {
                                                    return Some(false);
                                                }
                                                return Some(unsafe {
                                                    kernels.moe_sum_tensor(
                                                        backend.stream(),
                                                        1,
                                                        out.device_ptr(),
                                                        down.device_ptr(),
                                                        out_dim,
                                                        n_expert,
                                                    )
                                                });
                                            }
                                            if !unsafe {
                                                kernels.moe_q8_k_quantize_tensor(
                                                    backend.stream(),
                                                    x.device_ptr(),
                                                    down.device_ptr(),
                                                    xq_bytes,
                                                    expert_in_dim,
                                                    1,
                                                )
                                            } {
                                                return Some(false);
                                            }
                                            let use_decode_lut = xq_blocks <= 16
                                                && std::env::var_os(
                                                    "DS4_CUDA_MOE_NO_DECODE_LUT_GATE",
                                                )
                                                .is_none();
                                            let gate_ok = if use_decode_lut {
                                                unsafe {
                                                    kernels.moe_gate_up_mid_decode_tensor(
                                                        backend.stream(),
                                                        q4_k,
                                                        std::env::var_os(
                                                            "DS4_CUDA_MOE_WRITE_GATE_UP",
                                                        )
                                                        .is_some(),
                                                        gate.device_ptr(),
                                                        up.device_ptr(),
                                                        mid.device_ptr(),
                                                        gate_weights_ptr,
                                                        up_weights_ptr,
                                                        down.device_ptr(),
                                                        selected.device_ptr(),
                                                        weights.device_ptr(),
                                                        tables.grid.cu_deviceptr(),
                                                        tables.signs.cu_deviceptr(),
                                                        gate_model_bytes,
                                                        xq_bytes,
                                                        xq_blocks,
                                                        expert_mid_dim,
                                                        n_expert,
                                                        gate_expert_bytes,
                                                        gate_row_bytes,
                                                        clamp,
                                                    )
                                                }
                                            } else {
                                                unsafe {
                                                    kernels.moe_gate_up_mid_qwarp32_tensor(
                                                        backend.stream(),
                                                        gate.device_ptr(),
                                                        up.device_ptr(),
                                                        mid.device_ptr(),
                                                        gate_weights_ptr,
                                                        up_weights_ptr,
                                                        down.device_ptr(),
                                                        selected.device_ptr(),
                                                        weights.device_ptr(),
                                                        tables.grid.cu_deviceptr(),
                                                        tables.signs.cu_deviceptr(),
                                                        gate_model_bytes,
                                                        xq_bytes,
                                                        xq_blocks,
                                                        expert_mid_dim,
                                                        n_expert,
                                                        gate_expert_bytes,
                                                        gate_row_bytes,
                                                        clamp,
                                                    )
                                                }
                                            };
                                            if !gate_ok
                                                || !unsafe {
                                                    kernels.moe_q8_k_quantize_tensor(
                                                        backend.stream(),
                                                        mid.device_ptr(),
                                                        gate.device_ptr(),
                                                        midq_bytes,
                                                        expert_mid_dim,
                                                        n_expert,
                                                    )
                                                }
                                            {
                                                return Some(false);
                                            }
                                            let direct_sum6 = n_expert == 6
                                                && std::env::var_os(
                                                    "DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6",
                                                )
                                                .is_none();
                                            if direct_sum6 {
                                                return Some(unsafe {
                                                    kernels.moe_down_sum6_tensor(
                                                        backend.stream(),
                                                        q4_k,
                                                        out.device_ptr(),
                                                        down_weights_ptr,
                                                        gate.device_ptr(),
                                                        selected.device_ptr(),
                                                        down_model_bytes,
                                                        midq_bytes,
                                                        midq_blocks,
                                                        out_dim,
                                                        down_expert_bytes,
                                                        down_row_bytes,
                                                    )
                                                });
                                            }
                                            if !unsafe {
                                                kernels.moe_down_qwarp32_tensor(
                                                    backend.stream(),
                                                    down.device_ptr(),
                                                    down_weights_ptr,
                                                    gate.device_ptr(),
                                                    selected.device_ptr(),
                                                    down_model_bytes,
                                                    midq_bytes,
                                                    midq_blocks,
                                                    out_dim,
                                                    n_expert,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                )
                                            } {
                                                return Some(false);
                                            }
                                            Some(unsafe {
                                                kernels.moe_sum_tensor(
                                                    backend.stream(),
                                                    1,
                                                    out.device_ptr(),
                                                    down.device_ptr(),
                                                    out_dim,
                                                    n_expert,
                                                )
                                            })
                                        })
                                    })
                                },
                            )
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_routed_moe_batch_tensor(
    out: *mut Ds4GpuTensor,
    gate: *mut Ds4GpuTensor,
    up: *mut Ds4GpuTensor,
    mid: *mut Ds4GpuTensor,
    down: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: *const Ds4GpuTensor,
    weights: *const Ds4GpuTensor,
    n_expert: u32,
    clamp: f32,
    x: *const Ds4GpuTensor,
    n_tokens: u32,
    mid_is_f16: *mut bool,
) -> c_int {
    if n_tokens == 1 {
        let ok = unsafe {
            ds4_gpu_routed_moe_one_tensor(
                out,
                gate,
                up,
                mid,
                down,
                model_map,
                model_size,
                gate_offset,
                up_offset,
                down_offset,
                gate_type,
                down_type,
                gate_expert_bytes,
                gate_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                out_dim,
                selected,
                weights,
                n_expert,
                clamp,
                x,
            )
        };
        if ok != 0 && !mid_is_f16.is_null() {
            unsafe { *mid_is_f16 = false };
        }
        return ok;
    }
    status(|| {
        const MOE_EXPERT_COUNT: u64 = 256;
        const QK_K: u32 = 256;
        const Q8_K_BYTES: u64 = 292;

        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(gate) = (unsafe { tensor_ref(gate.cast_const()) }) else {
            return false;
        };
        let Some(up) = (unsafe { tensor_ref(up.cast_const()) }) else {
            return false;
        };
        let Some(mid) = (unsafe { tensor_ref(mid.cast_const()) }) else {
            return false;
        };
        let Some(down) = (unsafe { tensor_ref(down.cast_const()) }) else {
            return false;
        };
        let Some(selected) = (unsafe { tensor_ref(selected) }) else {
            return false;
        };
        let Some(weights) = (unsafe { tensor_ref(weights) }) else {
            return false;
        };
        let Some(x) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let Some(pair_count) = n_tokens.checked_mul(n_expert) else {
            return false;
        };
        let Some(x_bytes) = u64::from(n_tokens)
            .checked_mul(u64::from(expert_in_dim))
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        let Some(route_bytes) = u64::from(pair_count).checked_mul(size_of::<i32>() as u64) else {
            return false;
        };
        let Some(weight_bytes) = u64::from(pair_count).checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(mid_elements) = u64::from(pair_count).checked_mul(u64::from(expert_mid_dim))
        else {
            return false;
        };
        let Some(mid_bytes) = mid_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(down_elements) = u64::from(pair_count).checked_mul(u64::from(out_dim)) else {
            return false;
        };
        let Some(down_bytes) = down_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(out_elements) = u64::from(n_tokens).checked_mul(u64::from(out_dim)) else {
            return false;
        };
        let Some(out_bytes) = out_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(gate_model_bytes) = MOE_EXPERT_COUNT.checked_mul(gate_expert_bytes) else {
            return false;
        };
        let Some(down_model_bytes) = MOE_EXPERT_COUNT.checked_mul(down_expert_bytes) else {
            return false;
        };
        if model_map.is_null()
            || n_tokens == 0
            || n_expert == 0
            || expert_in_dim % QK_K != 0
            || expert_mid_dim % QK_K != 0
            || gate_type != 16
            || down_type != 10
            || gate_offset > model_size
            || up_offset > model_size
            || down_offset > model_size
            || gate_model_bytes > model_size - gate_offset
            || gate_model_bytes > model_size - up_offset
            || down_model_bytes > model_size - down_offset
            || x.bytes < x_bytes
            || selected.bytes < route_bytes
            || weights.bytes < weight_bytes
            || gate.bytes < mid_bytes
            || up.bytes < mid_bytes
            || mid.bytes < mid_bytes
            || down.bytes < down_bytes
            || out.bytes < out_bytes
        {
            return false;
        }
        let xq_blocks = expert_in_dim / QK_K;
        let midq_blocks = expert_mid_dim / QK_K;
        let Some(xq_bytes) = u64::from(n_tokens)
            .checked_mul(u64::from(xq_blocks))
            .and_then(|blocks| blocks.checked_mul(Q8_K_BYTES))
        else {
            return false;
        };
        let Some(midq_bytes) = u64::from(pair_count)
            .checked_mul(u64::from(midq_blocks))
            .and_then(|blocks| blocks.checked_mul(Q8_K_BYTES))
        else {
            return false;
        };
        let use_quantized = down.bytes >= xq_bytes && gate.bytes >= midq_bytes;

        let launched = with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                gate_offset,
                gate_model_bytes,
                |gate_weights_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        up_offset,
                        gate_model_bytes,
                        |up_weights_ptr| {
                            with_cached_abi_model_range(
                                backend,
                                model_map,
                                model_size,
                                down_offset,
                                down_model_bytes,
                                |down_weights_ptr| {
                                    with_abi_moe_iq2_tables(backend, |tables| {
                                        with_abi_kernels(backend, |kernels| {
                                            if !use_quantized {
                                                if !unsafe {
                                                    kernels.moe_gate_up_mid_f32_tensor(
                                                        backend.stream(),
                                                        n_tokens,
                                                        gate.device_ptr(),
                                                        up.device_ptr(),
                                                        mid.device_ptr(),
                                                        gate_weights_ptr,
                                                        up_weights_ptr,
                                                        x.device_ptr(),
                                                        selected.device_ptr(),
                                                        weights.device_ptr(),
                                                        tables.grid.cu_deviceptr(),
                                                        tables.signs.cu_deviceptr(),
                                                        gate_model_bytes,
                                                        expert_in_dim,
                                                        expert_mid_dim,
                                                        n_expert,
                                                        gate_expert_bytes,
                                                        gate_row_bytes,
                                                        clamp,
                                                    )
                                                } {
                                                    return Some(false);
                                                }
                                                if !unsafe {
                                                    kernels.moe_down_f32_tensor(
                                                        backend.stream(),
                                                        n_tokens,
                                                        down.device_ptr(),
                                                        down_weights_ptr,
                                                        mid.device_ptr(),
                                                        selected.device_ptr(),
                                                        down_model_bytes,
                                                        expert_mid_dim,
                                                        out_dim,
                                                        n_expert,
                                                        down_expert_bytes,
                                                        down_row_bytes,
                                                    )
                                                } {
                                                    return Some(false);
                                                }
                                                return Some(unsafe {
                                                    kernels.moe_sum_tensor(
                                                        backend.stream(),
                                                        n_tokens,
                                                        out.device_ptr(),
                                                        down.device_ptr(),
                                                        out_dim,
                                                        n_expert,
                                                    )
                                                });
                                            }
                                            if !unsafe {
                                                kernels.moe_q8_k_quantize_tensor(
                                                    backend.stream(),
                                                    x.device_ptr(),
                                                    down.device_ptr(),
                                                    xq_bytes,
                                                    expert_in_dim,
                                                    n_tokens,
                                                )
                                            } {
                                                return Some(false);
                                            }

                                            let use_expert_tiles = std::env::var_os(
                                                "DS4_CUDA_MOE_NO_EXPERT_TILES",
                                            )
                                            .is_none();
                                            let expert_tile_m = if std::env::var_os(
                                                "DS4_CUDA_MOE_TILE4",
                                            )
                                            .is_some()
                                            {
                                                4
                                            } else {
                                                8
                                            };
                                            let use_p2_sorted =
                                                std::env::var_os("DS4_CUDA_MOE_NO_P2").is_none();
                                            let use_atomic_down = use_expert_tiles
                                                && (std::env::var_os(
                                                    "DS4_CUDA_MOE_ATOMIC_DOWN",
                                                )
                                                .is_some()
                                                    || (n_tokens >= 128
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
                                                        )
                                                        .is_none()));
                                            let use_gate_rowspan = use_expert_tiles
                                                && expert_tile_m == 8
                                                && (std::env::var_os(
                                                    "DS4_CUDA_MOE_GATE_ROW2048",
                                                )
                                                .is_some()
                                                    || std::env::var_os(
                                                        "DS4_CUDA_MOE_GATE_ROW256",
                                                    )
                                                    .is_some()
                                                    || std::env::var_os(
                                                        "DS4_CUDA_MOE_GATE_ROW128",
                                                    )
                                                    .is_some()
                                                    || (n_tokens >= 128
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_GATE_ROW2048",
                                                        )
                                                        .is_none()
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_GATE_ROW256",
                                                        )
                                                        .is_none()
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_GATE_ROW128",
                                                        )
                                                        .is_none()));
                                            let use_down_tile16 = use_atomic_down
                                                && expert_tile_m == 8
                                                && n_tokens >= 128
                                                && std::env::var_os(
                                                    "DS4_CUDA_MOE_NO_DOWN_TILE16",
                                                )
                                                .is_none();
                                            let gate_row_span = if std::env::var_os(
                                                "DS4_CUDA_MOE_GATE_ROW512",
                                            )
                                            .is_some()
                                            {
                                                512
                                            } else if std::env::var_os(
                                                "DS4_CUDA_MOE_GATE_ROW2048",
                                            )
                                            .is_some()
                                            {
                                                2048
                                            } else {
                                                1024
                                            };
                                            let down_row_span = if std::env::var_os(
                                                "DS4_CUDA_MOE_DOWN_ROW512",
                                            )
                                            .is_some()
                                            {
                                                512
                                            } else if std::env::var_os(
                                                "DS4_CUDA_MOE_DOWN_ROW1024",
                                            )
                                            .is_some()
                                            {
                                                1024
                                            } else {
                                                2048
                                            };
                                            let use_down_rowspan = use_atomic_down
                                                && expert_tile_m == 8
                                                && (std::env::var_os(
                                                    "DS4_CUDA_MOE_DOWN_ROW2048",
                                                )
                                                .is_some()
                                                    || std::env::var_os(
                                                        "DS4_CUDA_MOE_DOWN_ROW256",
                                                    )
                                                    .is_some()
                                                    || std::env::var_os(
                                                        "DS4_CUDA_MOE_DOWN_ROW128",
                                                    )
                                                    .is_some()
                                                    || std::env::var_os(
                                                        "DS4_CUDA_MOE_DOWN_ROW64",
                                                    )
                                                    .is_some()
                                                    || (use_down_tile16
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_DOWN_ROW2048",
                                                        )
                                                        .is_none()
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_DOWN_ROW256",
                                                        )
                                                        .is_none()
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_DOWN_ROW128",
                                                        )
                                                        .is_none()
                                                        && std::env::var_os(
                                                            "DS4_CUDA_MOE_NO_DOWN_ROW64",
                                                        )
                                                        .is_none()));
                                            let write_gate_up = std::env::var_os(
                                                "DS4_CUDA_MOE_WRITE_GATE_UP",
                                            )
                                            .is_some();
                                            let Some(tile_capacity) = pair_count
                                                .div_ceil(expert_tile_m)
                                                .checked_add(256)
                                            else {
                                                return Some(false);
                                            };
                                            let tile16_capacity = if use_down_tile16 {
                                                let Some(capacity) =
                                                    pair_count.div_ceil(16).checked_add(256)
                                                else {
                                                    return Some(false);
                                                };
                                                capacity
                                            } else {
                                                1
                                            };
                                            let Some(pair_count_len) =
                                                usize::try_from(pair_count).ok()
                                            else {
                                                return Some(false);
                                            };
                                            let Some(tile_capacity_len) =
                                                usize::try_from(tile_capacity).ok()
                                            else {
                                                return Some(false);
                                            };
                                            let Some(tile16_capacity_len) =
                                                usize::try_from(tile16_capacity).ok()
                                            else {
                                                return Some(false);
                                            };
                                            with_abi_routed_moe_batch_scratch(
                                                backend,
                                                pair_count_len,
                                                tile_capacity_len,
                                                tile16_capacity_len,
                                                |scratch| {
                                                    clear_abi_routed_moe_counts(
                                                        backend,
                                                        &scratch.counts,
                                                    )?;
                                                    if !unsafe {
                                                        kernels.moe_count_sorted_pairs_tensor(
                                                            backend.stream(),
                                                            pair_count,
                                                            selected.device_ptr(),
                                                            scratch.counts.cu_deviceptr(),
                                                        )
                                                    } || !unsafe {
                                                        kernels.moe_prefix_sorted_pairs_tensor(
                                                            backend.stream(),
                                                            scratch.counts.cu_deviceptr(),
                                                            scratch.offsets.cu_deviceptr(),
                                                            scratch.cursors.cu_deviceptr(),
                                                        )
                                                    } || !unsafe {
                                                        kernels.moe_scatter_sorted_pairs_tensor(
                                                            backend.stream(),
                                                            pair_count,
                                                            selected.device_ptr(),
                                                            scratch.cursors.cu_deviceptr(),
                                                            scratch.sorted_pairs.cu_deviceptr(),
                                                        )
                                                    } {
                                                        return Some(false);
                                                    }
                                                    if use_expert_tiles
                                                        && (!unsafe {
                                                            kernels
                                                                .moe_build_expert_tile_offsets_tensor(
                                                                    backend.stream(),
                                                                    expert_tile_m,
                                                                    scratch.counts.cu_deviceptr(),
                                                                    scratch
                                                                        .tile_offsets
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_total
                                                                        .cu_deviceptr(),
                                                                )
                                                        } || !unsafe {
                                                            kernels.moe_build_expert_tiles_tensor(
                                                                backend.stream(),
                                                                expert_tile_m,
                                                                tile_capacity,
                                                                scratch.counts.cu_deviceptr(),
                                                                scratch.tile_offsets.cu_deviceptr(),
                                                                scratch.tile_experts.cu_deviceptr(),
                                                                scratch.tile_starts.cu_deviceptr(),
                                                            )
                                                        })
                                                    {
                                                        return Some(false);
                                                    }
                                                    if use_expert_tiles
                                                        && use_down_tile16
                                                        && (!unsafe {
                                                            kernels
                                                                .moe_build_expert_tile_offsets_tensor(
                                                                    backend.stream(),
                                                                    16,
                                                                    scratch.counts.cu_deviceptr(),
                                                                    scratch
                                                                        .tile16_offsets
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile16_total
                                                                        .cu_deviceptr(),
                                                                )
                                                        } || !unsafe {
                                                            kernels.moe_build_expert_tiles_tensor(
                                                                backend.stream(),
                                                                16,
                                                                tile16_capacity,
                                                                scratch.counts.cu_deviceptr(),
                                                                scratch
                                                                    .tile16_offsets
                                                                    .cu_deviceptr(),
                                                                scratch
                                                                    .tile16_experts
                                                                    .cu_deviceptr(),
                                                                scratch
                                                                    .tile16_starts
                                                                    .cu_deviceptr(),
                                                            )
                                                        })
                                                    {
                                                        return Some(false);
                                                    }
                                                    let gate_ok = if use_expert_tiles
                                                        && use_gate_rowspan
                                                    {
                                                        unsafe {
                                                            kernels
                                                                .moe_gate_up_mid_expert_tile8_rowspan_tensor(
                                                                    backend.stream(),
                                                                    gate.device_ptr(),
                                                                    up.device_ptr(),
                                                                    mid.device_ptr(),
                                                                    gate_weights_ptr,
                                                                    up_weights_ptr,
                                                                    down.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    scratch.offsets.cu_deviceptr(),
                                                                    scratch.counts.cu_deviceptr(),
                                                                    scratch
                                                                        .tile_total
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_experts
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_starts
                                                                        .cu_deviceptr(),
                                                                    weights.device_ptr(),
                                                                    tables.grid.cu_deviceptr(),
                                                                    tables.signs.cu_deviceptr(),
                                                                    gate_model_bytes,
                                                                    xq_bytes,
                                                                    xq_blocks,
                                                                    expert_mid_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    tile_capacity,
                                                                    gate_row_span,
                                                                    gate_expert_bytes,
                                                                    gate_row_bytes,
                                                                    write_gate_up,
                                                                    clamp,
                                                                )
                                                        }
                                                    } else if use_expert_tiles
                                                        && expert_tile_m == 8
                                                    {
                                                        unsafe {
                                                            kernels
                                                                .moe_gate_up_mid_expert_tile8_row32_tensor(
                                                                    backend.stream(),
                                                                    gate.device_ptr(),
                                                                    up.device_ptr(),
                                                                    mid.device_ptr(),
                                                                    gate_weights_ptr,
                                                                    up_weights_ptr,
                                                                    down.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    scratch.offsets.cu_deviceptr(),
                                                                    scratch.counts.cu_deviceptr(),
                                                                    scratch
                                                                        .tile_total
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_experts
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_starts
                                                                        .cu_deviceptr(),
                                                                    weights.device_ptr(),
                                                                    tables.grid.cu_deviceptr(),
                                                                    tables.signs.cu_deviceptr(),
                                                                    gate_model_bytes,
                                                                    xq_bytes,
                                                                    xq_blocks,
                                                                    expert_mid_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    tile_capacity,
                                                                    gate_expert_bytes,
                                                                    gate_row_bytes,
                                                                    write_gate_up,
                                                                    clamp,
                                                                )
                                                        }
                                                    } else if use_expert_tiles {
                                                        unsafe {
                                                            kernels
                                                                .moe_gate_up_mid_expert_tile4_row32_tensor(
                                                                    backend.stream(),
                                                                    gate.device_ptr(),
                                                                    up.device_ptr(),
                                                                    mid.device_ptr(),
                                                                    gate_weights_ptr,
                                                                    up_weights_ptr,
                                                                    down.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    scratch.offsets.cu_deviceptr(),
                                                                    scratch.counts.cu_deviceptr(),
                                                                    scratch
                                                                        .tile_total
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_experts
                                                                        .cu_deviceptr(),
                                                                    scratch
                                                                        .tile_starts
                                                                        .cu_deviceptr(),
                                                                    weights.device_ptr(),
                                                                    tables.grid.cu_deviceptr(),
                                                                    tables.signs.cu_deviceptr(),
                                                                    gate_model_bytes,
                                                                    xq_bytes,
                                                                    xq_blocks,
                                                                    expert_mid_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    tile_capacity,
                                                                    gate_expert_bytes,
                                                                    gate_row_bytes,
                                                                    write_gate_up,
                                                                    clamp,
                                                                )
                                                        }
                                                    } else if use_p2_sorted {
                                                        unsafe {
                                                            kernels
                                                                .moe_gate_up_mid_sorted_p2_qwarp32_tensor(
                                                                    backend.stream(),
                                                                    gate.device_ptr(),
                                                                    up.device_ptr(),
                                                                    mid.device_ptr(),
                                                                    gate_weights_ptr,
                                                                    up_weights_ptr,
                                                                    down.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    selected.device_ptr(),
                                                                    weights.device_ptr(),
                                                                    tables.grid.cu_deviceptr(),
                                                                    tables.signs.cu_deviceptr(),
                                                                    gate_model_bytes,
                                                                    xq_bytes,
                                                                    xq_blocks,
                                                                    expert_mid_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    gate_expert_bytes,
                                                                    gate_row_bytes,
                                                                    clamp,
                                                                )
                                                        }
                                                    } else {
                                                        unsafe {
                                                            kernels
                                                                .moe_gate_up_mid_sorted_qwarp32_tensor(
                                                                    backend.stream(),
                                                                    gate.device_ptr(),
                                                                    up.device_ptr(),
                                                                    mid.device_ptr(),
                                                                    gate_weights_ptr,
                                                                    up_weights_ptr,
                                                                    down.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    selected.device_ptr(),
                                                                    weights.device_ptr(),
                                                                    tables.grid.cu_deviceptr(),
                                                                    tables.signs.cu_deviceptr(),
                                                                    gate_model_bytes,
                                                                    xq_bytes,
                                                                    xq_blocks,
                                                                    expert_mid_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    gate_expert_bytes,
                                                                    gate_row_bytes,
                                                                    clamp,
                                                                )
                                                        }
                                                    };
                                                    if !gate_ok
                                                        || !unsafe {
                                                            kernels.moe_q8_k_quantize_tensor(
                                                                backend.stream(),
                                                                mid.device_ptr(),
                                                                gate.device_ptr(),
                                                                midq_bytes,
                                                                expert_mid_dim,
                                                                pair_count,
                                                            )
                                                        }
                                                    {
                                                        return Some(false);
                                                    }
                                                    if use_atomic_down
                                                        && !unsafe {
                                                            kernels.moe_atomic_output_zero_tensor(
                                                                backend.stream(),
                                                                out.device_ptr(),
                                                                out_elements,
                                                            )
                                                        }
                                                    {
                                                        return Some(false);
                                                    }
                                                    let down_ptr = if use_atomic_down {
                                                        out.device_ptr()
                                                    } else {
                                                        down.device_ptr()
                                                    };
                                                    let down_ok = if use_expert_tiles {
                                                        let (
                                                            down_tile_total,
                                                            down_tile_experts,
                                                            down_tile_starts,
                                                            down_tile_capacity,
                                                        ) = if use_down_tile16 {
                                                            (
                                                                scratch
                                                                    .tile16_total
                                                                    .cu_deviceptr(),
                                                                scratch
                                                                    .tile16_experts
                                                                    .cu_deviceptr(),
                                                                scratch
                                                                    .tile16_starts
                                                                    .cu_deviceptr(),
                                                                tile16_capacity,
                                                            )
                                                        } else {
                                                            (
                                                                scratch
                                                                    .tile_total
                                                                    .cu_deviceptr(),
                                                                scratch
                                                                    .tile_experts
                                                                    .cu_deviceptr(),
                                                                scratch
                                                                    .tile_starts
                                                                    .cu_deviceptr(),
                                                                tile_capacity,
                                                            )
                                                        };
                                                        if use_down_rowspan {
                                                            unsafe {
                                                                kernels
                                                                    .moe_down_expert_tile16_rowspan_tensor(
                                                                        backend.stream(),
                                                                        down_ptr,
                                                                        down_weights_ptr,
                                                                        gate.device_ptr(),
                                                                        scratch
                                                                            .sorted_pairs
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .offsets
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .counts
                                                                            .cu_deviceptr(),
                                                                        down_tile_total,
                                                                        down_tile_experts,
                                                                        down_tile_starts,
                                                                        down_model_bytes,
                                                                        midq_bytes,
                                                                        midq_blocks,
                                                                        out_dim,
                                                                        n_expert,
                                                                        n_tokens,
                                                                        pair_count,
                                                                        down_tile_capacity,
                                                                        down_row_span,
                                                                        down_expert_bytes,
                                                                        down_row_bytes,
                                                                        use_atomic_down,
                                                                    )
                                                            }
                                                        } else if use_down_tile16 {
                                                            unsafe {
                                                                kernels
                                                                    .moe_down_expert_tile16_row32_tensor(
                                                                        backend.stream(),
                                                                        down_ptr,
                                                                        down_weights_ptr,
                                                                        gate.device_ptr(),
                                                                        scratch
                                                                            .sorted_pairs
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .offsets
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .counts
                                                                            .cu_deviceptr(),
                                                                        down_tile_total,
                                                                        down_tile_experts,
                                                                        down_tile_starts,
                                                                        down_model_bytes,
                                                                        midq_bytes,
                                                                        midq_blocks,
                                                                        out_dim,
                                                                        n_expert,
                                                                        n_tokens,
                                                                        pair_count,
                                                                        down_tile_capacity,
                                                                        down_expert_bytes,
                                                                        down_row_bytes,
                                                                        use_atomic_down,
                                                                    )
                                                            }
                                                        } else if expert_tile_m == 8 {
                                                            unsafe {
                                                                kernels
                                                                    .moe_down_expert_tile8_row32_tensor(
                                                                        backend.stream(),
                                                                        down_ptr,
                                                                        down_weights_ptr,
                                                                        gate.device_ptr(),
                                                                        scratch
                                                                            .sorted_pairs
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .offsets
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .counts
                                                                            .cu_deviceptr(),
                                                                        down_tile_total,
                                                                        down_tile_experts,
                                                                        down_tile_starts,
                                                                        down_model_bytes,
                                                                        midq_bytes,
                                                                        midq_blocks,
                                                                        out_dim,
                                                                        n_expert,
                                                                        n_tokens,
                                                                        pair_count,
                                                                        down_tile_capacity,
                                                                        down_expert_bytes,
                                                                        down_row_bytes,
                                                                        use_atomic_down,
                                                                    )
                                                            }
                                                        } else {
                                                            unsafe {
                                                                kernels
                                                                    .moe_down_expert_tile4_row32_tensor(
                                                                        backend.stream(),
                                                                        down_ptr,
                                                                        down_weights_ptr,
                                                                        gate.device_ptr(),
                                                                        scratch
                                                                            .sorted_pairs
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .offsets
                                                                            .cu_deviceptr(),
                                                                        scratch
                                                                            .counts
                                                                            .cu_deviceptr(),
                                                                        down_tile_total,
                                                                        down_tile_experts,
                                                                        down_tile_starts,
                                                                        down_model_bytes,
                                                                        midq_bytes,
                                                                        midq_blocks,
                                                                        out_dim,
                                                                        n_expert,
                                                                        n_tokens,
                                                                        pair_count,
                                                                        down_tile_capacity,
                                                                        down_expert_bytes,
                                                                        down_row_bytes,
                                                                        use_atomic_down,
                                                                    )
                                                            }
                                                        }
                                                    } else if use_p2_sorted {
                                                        unsafe {
                                                            kernels
                                                                .moe_down_sorted_p2_qwarp32_tensor(
                                                                    backend.stream(),
                                                                    down.device_ptr(),
                                                                    down_weights_ptr,
                                                                    gate.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    selected.device_ptr(),
                                                                    down_model_bytes,
                                                                    midq_bytes,
                                                                    midq_blocks,
                                                                    out_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    down_expert_bytes,
                                                                    down_row_bytes,
                                                                )
                                                        }
                                                    } else {
                                                        unsafe {
                                                            kernels
                                                                .moe_down_sorted_qwarp32_tensor(
                                                                    backend.stream(),
                                                                    down.device_ptr(),
                                                                    down_weights_ptr,
                                                                    gate.device_ptr(),
                                                                    scratch
                                                                        .sorted_pairs
                                                                        .cu_deviceptr(),
                                                                    selected.device_ptr(),
                                                                    down_model_bytes,
                                                                    midq_bytes,
                                                                    midq_blocks,
                                                                    out_dim,
                                                                    n_expert,
                                                                    pair_count,
                                                                    down_expert_bytes,
                                                                    down_row_bytes,
                                                                )
                                                        }
                                                    };
                                                    if !down_ok {
                                                        return Some(false);
                                                    }
                                                    if use_atomic_down {
                                                        Some(true)
                                                    } else {
                                                        Some(unsafe {
                                                            kernels.moe_sum_tensor(
                                                                backend.stream(),
                                                                n_tokens,
                                                                out.device_ptr(),
                                                                down.device_ptr(),
                                                                out_dim,
                                                                n_expert,
                                                            )
                                                        })
                                                    }
                                                },
                                            )
                                        })
                                    })
                                },
                            )
                        },
                    )
                },
            )
        })
        .unwrap_or(false);
        if launched && !mid_is_f16.is_null() {
            unsafe { *mid_is_f16 = false };
        }
        launched
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
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_matmul_f16_tensor(
    out: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    n_tok: u64,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(x) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let Some(weight_elements) = in_dim.checked_mul(out_dim) else {
            return false;
        };
        let Some(weight_bytes) = weight_elements.checked_mul(size_of::<u16>() as u64) else {
            return false;
        };
        let Some(x_bytes) = in_dim
            .checked_mul(n_tok)
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        let Some(out_bytes) = out_dim
            .checked_mul(n_tok)
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        if model_map.is_null()
            || in_dim == 0
            || out_dim == 0
            || n_tok == 0
            || weight_offset > model_size
            || weight_bytes > model_size - weight_offset
            || x.bytes < x_bytes
            || out.bytes < out_bytes
        {
            return false;
        }
        let path = select_f16_projection_path(F16ProjectionDispatch {
            blas_ready: n_tok > 1,
            serial_f16: std::env::var_os("DS4_CUDA_SERIAL_F16_MATMUL").is_some(),
            serial_router: std::env::var_os("DS4_CUDA_SERIAL_ROUTER").is_some(),
            no_ordered_f16_matmul: std::env::var_os("DS4_CUDA_NO_ORDERED_F16_MATMUL").is_some(),
            in_dim,
            out_dim,
            n_tokens: n_tok,
        });
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                weight_offset,
                weight_bytes,
                |weight_ptr| {
                    if path == crate::F16ProjectionPath::Blas {
                        let Ok(blas) = backend.blas_handle() else {
                            return Some(false);
                        };
                        if !apply_abi_blas_math(&blas) {
                            return Some(false);
                        }
                        let Some(weight_elements) = usize::try_from(weight_elements).ok() else {
                            return Some(false);
                        };
                        let Some(x_elements) = usize::try_from(in_dim.checked_mul(n_tok)?).ok()
                        else {
                            return Some(false);
                        };
                        let Some(out_elements) = usize::try_from(out_dim.checked_mul(n_tok)?).ok()
                        else {
                            return Some(false);
                        };
                        let Some(in_dim) = usize::try_from(in_dim).ok() else {
                            return Some(false);
                        };
                        let Some(out_dim) = usize::try_from(out_dim).ok() else {
                            return Some(false);
                        };
                        let Some(n_tok) = usize::try_from(n_tok).ok() else {
                            return Some(false);
                        };
                        return with_abi_f16_activations(backend, x_elements, |activations| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: activation bounds are validated and
                                // retained scratch storage survives queued BLAS
                                // use until a later synchronized replacement.
                                if !unsafe {
                                    kernels.f32_to_f16_tensor(
                                        backend.stream(),
                                        x.device_ptr(),
                                        activations.cu_deviceptr(),
                                        x_elements as u64,
                                    )
                                } {
                                    return Some(false);
                                }
                                // SAFETY: these wrappers borrow ABI-owned
                                // device allocations during BLAS submission;
                                // ManuallyDrop prevents duplicate frees.
                                let weights = ManuallyDrop::new(unsafe {
                                    DeviceBuffer::<f16>::from_raw_parts(
                                        weight_ptr,
                                        weight_elements,
                                        backend.context().clone(),
                                    )
                                });
                                let mut output = ManuallyDrop::new(unsafe {
                                    DeviceBuffer::<f32>::from_raw_parts(
                                        out.device_ptr(),
                                        out_elements,
                                        backend.context().clone(),
                                    )
                                });
                                Some(
                                    blas.project_f16_f32(
                                        backend.stream(),
                                        ProjectionConfig::new(in_dim, out_dim, n_tok),
                                        &weights,
                                        activations,
                                        &mut output,
                                    )
                                    .is_ok(),
                                )
                            })
                        });
                    }
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: this leaf validates tensor and cached
                        // model-weight bounds before selecting the equivalent
                        // current-C base, ordered, or serial path.
                        Some(unsafe {
                            kernels.matmul_f16_tensor(
                                backend.stream(),
                                out.device_ptr(),
                                weight_ptr,
                                x.device_ptr(),
                                in_dim,
                                out_dim,
                                n_tok,
                                path,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_matmul_f16_pair_tensor(
    out0: *mut Ds4GpuTensor,
    out1: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight0_offset: u64,
    weight1_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    n_tok: u64,
) -> c_int {
    status(|| {
        let Some(out0_tensor) = (unsafe { tensor_ref(out0.cast_const()) }) else {
            return false;
        };
        let Some(out1_tensor) = (unsafe { tensor_ref(out1.cast_const()) }) else {
            return false;
        };
        let Some(x_tensor) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let Some(weight_elements) = in_dim.checked_mul(out_dim) else {
            return false;
        };
        let Some(weight_bytes) = weight_elements.checked_mul(size_of::<u16>() as u64) else {
            return false;
        };
        let Some(x_bytes) = in_dim
            .checked_mul(n_tok)
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        let Some(out_bytes) = out_dim
            .checked_mul(n_tok)
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        if model_map.is_null()
            || in_dim == 0
            || out_dim == 0
            || n_tok == 0
            || weight0_offset > model_size
            || weight1_offset > model_size
            || weight_bytes > model_size - weight0_offset
            || weight_bytes > model_size - weight1_offset
            || x_tensor.bytes < x_bytes
            || out0_tensor.bytes < out_bytes
            || out1_tensor.bytes < out_bytes
        {
            return false;
        }
        let path = select_f16_pair_projection_path(F16PairProjectionDispatch {
            n_tokens: n_tok,
            no_f16_pair_matmul: std::env::var_os("DS4_CUDA_NO_F16_PAIR_MATMUL").is_some(),
            serial_f16: std::env::var_os("DS4_CUDA_SERIAL_F16_MATMUL").is_some(),
            serial_router: std::env::var_os("DS4_CUDA_SERIAL_ROUTER").is_some(),
            no_ordered_f16_matmul: std::env::var_os("DS4_CUDA_NO_ORDERED_F16_MATMUL").is_some(),
        });
        if path == F16PairProjectionPath::TwoIndependent {
            return unsafe {
                ds4_gpu_matmul_f16_tensor(
                    out0,
                    model_map,
                    model_size,
                    weight0_offset,
                    in_dim,
                    out_dim,
                    x,
                    n_tok,
                ) != 0
                    && ds4_gpu_matmul_f16_tensor(
                        out1,
                        model_map,
                        model_size,
                        weight1_offset,
                        in_dim,
                        out_dim,
                        x,
                        n_tok,
                    ) != 0
            };
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                weight0_offset,
                weight_bytes,
                |weight0_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        weight1_offset,
                        weight_bytes,
                        |weight1_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: bounds above cover two single-token
                                // outputs and both live cached F16 weight ranges.
                                Some(unsafe {
                                    kernels.matmul_f16_pair_ordered_chunks_tensor(
                                        backend.stream(),
                                        out0_tensor.device_ptr(),
                                        out1_tensor.device_ptr(),
                                        weight0_ptr,
                                        weight1_ptr,
                                        x_tensor.device_ptr(),
                                        in_dim,
                                        out_dim,
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_matmul_f32_tensor(
    out: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    n_tok: u64,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(x) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let Some(weight_elements) = in_dim.checked_mul(out_dim) else {
            return false;
        };
        let Some(weight_bytes) = weight_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(x_bytes) = in_dim
            .checked_mul(n_tok)
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        let Some(out_bytes) = out_dim
            .checked_mul(n_tok)
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
        else {
            return false;
        };
        if model_map.is_null()
            || in_dim == 0
            || out_dim == 0
            || n_tok == 0
            || weight_offset > model_size
            || weight_bytes > model_size - weight_offset
            || x.bytes < x_bytes
            || out.bytes < out_bytes
        {
            return false;
        }
        let path = select_f32_projection_path(n_tok > 1, n_tok);
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                weight_offset,
                weight_bytes,
                |weight_ptr| {
                    if path == crate::F32ProjectionPath::Blas {
                        let Ok(blas) = backend.blas_handle() else {
                            return Some(false);
                        };
                        if !apply_abi_blas_math(&blas) {
                            return Some(false);
                        }
                        let Some(weight_elements) = usize::try_from(weight_elements).ok() else {
                            return Some(false);
                        };
                        let Some(x_elements) = usize::try_from(in_dim.checked_mul(n_tok)?).ok()
                        else {
                            return Some(false);
                        };
                        let Some(out_elements) = usize::try_from(out_dim.checked_mul(n_tok)?).ok()
                        else {
                            return Some(false);
                        };
                        let Some(in_dim) = usize::try_from(in_dim).ok() else {
                            return Some(false);
                        };
                        let Some(out_dim) = usize::try_from(out_dim).ok() else {
                            return Some(false);
                        };
                        let Some(n_tok) = usize::try_from(n_tok).ok() else {
                            return Some(false);
                        };
                        // SAFETY: the wrappers borrow caller-owned device
                        // allocations for the duration of this queued BLAS
                        // operation; ManuallyDrop prevents duplicate frees.
                        let weights = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<f32>::from_raw_parts(
                                weight_ptr,
                                weight_elements,
                                backend.context().clone(),
                            )
                        });
                        let activations = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<f32>::from_raw_parts(
                                x.device_ptr(),
                                x_elements,
                                backend.context().clone(),
                            )
                        });
                        let mut output = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<f32>::from_raw_parts(
                                out.device_ptr(),
                                out_elements,
                                backend.context().clone(),
                            )
                        });
                        return Some(
                            blas.project_f32(
                                backend.stream(),
                                ProjectionConfig::new(in_dim, out_dim, n_tok),
                                &weights,
                                &activations,
                                &mut output,
                            )
                            .is_ok(),
                        );
                    }
                    with_abi_kernels(backend, |kernels| {
                        // SAFETY: this leaf validates the single-token tensor
                        // and cached F32 weight bounds before launching the
                        // current-C-equivalent base kernel.
                        Some(unsafe {
                            kernels.matmul_f32_tensor(
                                backend.stream(),
                                out.device_ptr(),
                                weight_ptr,
                                x.device_ptr(),
                                in_dim,
                                out_dim,
                                n_tok,
                                path,
                            )
                        })
                    })
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
unsafe fn matmul_q8_0_tensor_labeled(
    out: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    n_tok: u64,
    label: &str,
) -> c_int {
    status(|| {
        let Some(out) = (unsafe { tensor_ref(out.cast_const()) }) else {
            return false;
        };
        let Some(x) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let Some((_weight_elements, weight_elements_usize, weight_bytes)) =
            abi_q8_shape(in_dim, out_dim)
        else {
            return false;
        };
        let Some(x_elements) = in_dim.checked_mul(n_tok) else {
            return false;
        };
        let Some(out_elements) = out_dim.checked_mul(n_tok) else {
            return false;
        };
        let Some(x_bytes) = x_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(out_bytes) = out_elements.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        if model_map.is_null()
            || n_tok == 0
            || weight_offset > model_size
            || weight_bytes > model_size - weight_offset
            || x.bytes < x_bytes
            || out.bytes < out_bytes
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
                |packed_weight_ptr| {
                    let blas = (n_tok > 1).then(|| backend.blas_handle().ok()).flatten();
                    if blas.is_some()
                        && abi_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim)
                            .is_none()
                        && q8_f32_cache_allowed(
                            abi_q8_cache_options(),
                            Some(label),
                            in_dim,
                            out_dim,
                        )
                    {
                        let _ = cache_abi_q8_f32_range(
                            backend,
                            model_map,
                            model_size,
                            weight_offset,
                            weight_bytes,
                            in_dim,
                            out_dim,
                        );
                    }
                    let expanded_f32_ptr = blas.as_ref().and_then(|_| {
                        abi_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim)
                    });
                    if blas.is_some() && expanded_f32_ptr.is_none() {
                        let _ = cache_abi_q8_f16_range(
                            backend,
                            model_map,
                            model_size,
                            weight_offset,
                            weight_bytes,
                            in_dim,
                            out_dim,
                            label,
                            false,
                        );
                    }
                    let expanded_f16_ptr = blas.as_ref().and_then(|_| {
                        abi_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim)
                    });
                    let mut path = select_q8_matmul_path(Q8MatmulDispatchOptions {
                        cublas_ready: blas.is_some(),
                        expanded_f32_blas_ready: expanded_f32_ptr.is_some(),
                        expanded_f16_blas_ready: expanded_f16_ptr.is_some(),
                        n_tokens: n_tok,
                        blocks: in_dim.div_ceil(32),
                        no_batch_warp: std::env::var_os("DS4_CUDA_NO_Q8_BATCH_WARP").is_some(),
                    });
                    if path == Q8MatmulPath::ExpandedF32Blas {
                        let Some(blas) = blas.as_ref() else {
                            return Some(false);
                        };
                        if !apply_abi_blas_math(blas) {
                            return Some(false);
                        }
                        let Some(x_elements) = usize::try_from(x_elements).ok() else {
                            return Some(false);
                        };
                        let Some(out_elements) = usize::try_from(out_elements).ok() else {
                            return Some(false);
                        };
                        let Some(in_dim) = usize::try_from(in_dim).ok() else {
                            return Some(false);
                        };
                        let Some(out_dim) = usize::try_from(out_dim).ok() else {
                            return Some(false);
                        };
                        let Some(n_tok) = usize::try_from(n_tok).ok() else {
                            return Some(false);
                        };
                        let weights = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<f32>::from_raw_parts(
                                expanded_f32_ptr?,
                                weight_elements_usize,
                                backend.context().clone(),
                            )
                        });
                        let activations = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<f32>::from_raw_parts(
                                x.device_ptr(),
                                x_elements,
                                backend.context().clone(),
                            )
                        });
                        let mut output = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<f32>::from_raw_parts(
                                out.device_ptr(),
                                out_elements,
                                backend.context().clone(),
                            )
                        });
                        return Some(
                            blas.project_f32(
                                backend.stream(),
                                ProjectionConfig::new(in_dim, out_dim, n_tok),
                                &weights,
                                &activations,
                                &mut output,
                            )
                            .is_ok(),
                        );
                    }
                    if path == Q8MatmulPath::ExpandedF16Blas {
                        let Some(blas) = blas.as_ref() else {
                            return Some(false);
                        };
                        if !apply_abi_blas_math(blas) {
                            return Some(false);
                        }
                        let Some(x_elements) = usize::try_from(x_elements).ok() else {
                            return Some(false);
                        };
                        let Some(out_elements) = usize::try_from(out_elements).ok() else {
                            return Some(false);
                        };
                        let Some(in_dim) = usize::try_from(in_dim).ok() else {
                            return Some(false);
                        };
                        let Some(out_dim) = usize::try_from(out_dim).ok() else {
                            return Some(false);
                        };
                        let Some(n_tok) = usize::try_from(n_tok).ok() else {
                            return Some(false);
                        };
                        let projected =
                            with_abi_f16_activations(backend, x_elements, |activations| {
                                with_abi_kernels(backend, |kernels| {
                                    if !unsafe {
                                        kernels.f32_to_f16_tensor(
                                            backend.stream(),
                                            x.device_ptr(),
                                            activations.cu_deviceptr(),
                                            x_elements as u64,
                                        )
                                    } {
                                        return Some(false);
                                    }
                                    let weights = ManuallyDrop::new(unsafe {
                                        DeviceBuffer::<f16>::from_raw_parts(
                                            expanded_f16_ptr?,
                                            weight_elements_usize,
                                            backend.context().clone(),
                                        )
                                    });
                                    let mut output = ManuallyDrop::new(unsafe {
                                        DeviceBuffer::<f32>::from_raw_parts(
                                            out.device_ptr(),
                                            out_elements,
                                            backend.context().clone(),
                                        )
                                    });
                                    Some(
                                        blas.project_f16_f32(
                                            backend.stream(),
                                            ProjectionConfig::new(in_dim, out_dim, n_tok),
                                            &weights,
                                            activations,
                                            &mut output,
                                        )
                                        .is_ok(),
                                    )
                                })
                            })
                            .unwrap_or(false);
                        if projected {
                            return Some(true);
                        }
                        let _ = backend.synchronize();
                        if let Ok(mut cache) = ABI_Q8_CACHE.lock() {
                            cache.f16_ranges.clear();
                            cache.state.disable_f16_after_failure();
                        }
                        path = select_q8_matmul_path(Q8MatmulDispatchOptions {
                            cublas_ready: false,
                            expanded_f32_blas_ready: false,
                            expanded_f16_blas_ready: false,
                            n_tokens: n_tok as u64,
                            blocks: (in_dim as u64).div_ceil(32),
                            no_batch_warp: std::env::var_os("DS4_CUDA_NO_Q8_BATCH_WARP").is_some(),
                        });
                    }
                    let blocks = in_dim.div_ceil(32);
                    let Some(quantized_elements) = n_tok
                        .checked_mul(blocks)
                        .and_then(|value| value.checked_mul(32))
                    else {
                        return Some(false);
                    };
                    let Some(scale_elements) = n_tok.checked_mul(blocks) else {
                        return Some(false);
                    };
                    let Some(quantized_elements) = usize::try_from(quantized_elements).ok() else {
                        return Some(false);
                    };
                    let Some(scale_elements) = usize::try_from(scale_elements).ok() else {
                        return Some(false);
                    };
                    with_abi_q8_activations(
                        backend,
                        quantized_elements,
                        scale_elements,
                        |activations| {
                            with_abi_kernels(backend, |kernels| {
                                if !unsafe {
                                    kernels.quantize_q8_f32_tensor(
                                        backend.stream(),
                                        x.device_ptr(),
                                        activations.quantized.cu_deviceptr(),
                                        activations.scales.cu_deviceptr(),
                                        in_dim,
                                        blocks,
                                        n_tok,
                                    )
                                } {
                                    return Some(false);
                                }
                                Some(unsafe {
                                    kernels.matmul_q8_tensor(
                                        backend.stream(),
                                        out.device_ptr(),
                                        packed_weight_ptr,
                                        activations.quantized.cu_deviceptr(),
                                        activations.scales.cu_deviceptr(),
                                        in_dim,
                                        out_dim,
                                        n_tok,
                                        path,
                                        q8_dp4a_enabled(
                                            std::env::var_os("DS4_CUDA_NO_Q8_DP4A").is_some(),
                                        ),
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_matmul_q8_0_tensor(
    out: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    n_tok: u64,
) -> c_int {
    unsafe {
        matmul_q8_0_tensor_labeled(
            out,
            model_map,
            model_size,
            weight_offset,
            in_dim,
            out_dim,
            x,
            n_tok,
            "q8_0",
        )
    }
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_q8_pair_fused_impl(
    out0: &Ds4GpuTensor,
    out1: &Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight0_offset: u64,
    weight1_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &Ds4GpuTensor,
) -> bool {
    let Some((_weight_elements, _weight_elements_usize, weight_bytes)) =
        abi_q8_shape(in_dim, out_dim)
    else {
        return false;
    };
    let Some(x_bytes) = in_dim.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(out_bytes) = out_dim.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    if model_map.is_null()
        || weight0_offset > model_size
        || weight1_offset > model_size
        || weight_bytes > model_size - weight0_offset
        || weight_bytes > model_size - weight1_offset
        || x.bytes < x_bytes
        || out0.bytes < out_bytes
        || out1.bytes < out_bytes
    {
        return false;
    }
    with_backend(|backend| {
        with_cached_abi_model_range(
            backend,
            model_map,
            model_size,
            weight0_offset,
            weight_bytes,
            |weight0_ptr| {
                with_cached_abi_model_range(
                    backend,
                    model_map,
                    model_size,
                    weight1_offset,
                    weight_bytes,
                    |weight1_ptr| {
                        let blocks = in_dim.div_ceil(32);
                        let Some(quantized_elements) = blocks.checked_mul(32) else {
                            return Some(false);
                        };
                        let Some(quantized_elements) = usize::try_from(quantized_elements).ok()
                        else {
                            return Some(false);
                        };
                        let Some(scale_elements) = usize::try_from(blocks).ok() else {
                            return Some(false);
                        };
                        with_abi_q8_activations(
                            backend,
                            quantized_elements,
                            scale_elements,
                            |activations| {
                                with_abi_kernels(backend, |kernels| {
                                    if !unsafe {
                                        kernels.quantize_q8_f32_tensor(
                                            backend.stream(),
                                            x.device_ptr(),
                                            activations.quantized.cu_deviceptr(),
                                            activations.scales.cu_deviceptr(),
                                            in_dim,
                                            blocks,
                                            1,
                                        )
                                    } {
                                        return Some(false);
                                    }
                                    Some(unsafe {
                                        kernels.matmul_q8_pair_tensor(
                                            backend.stream(),
                                            out0.device_ptr(),
                                            out1.device_ptr(),
                                            weight0_ptr,
                                            weight1_ptr,
                                            activations.quantized.cu_deviceptr(),
                                            activations.scales.cu_deviceptr(),
                                            in_dim,
                                            out_dim,
                                            out_dim,
                                            q8_dp4a_enabled(
                                                std::env::var_os("DS4_CUDA_NO_Q8_DP4A").is_some(),
                                            ),
                                        )
                                    })
                                })
                            },
                        )
                    },
                )
            },
        )
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
    gate: *mut Ds4GpuTensor,
    up: *mut Ds4GpuTensor,
    mid: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    gate_offset: u64,
    up_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    clamp: f32,
) -> c_int {
    if std::env::var_os("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR").is_some() {
        return status(|| unsafe {
            ds4_gpu_matmul_q8_0_tensor(
                gate,
                model_map,
                model_size,
                gate_offset,
                in_dim,
                out_dim,
                x,
                1,
            ) != 0
                && ds4_gpu_matmul_q8_0_tensor(
                    up, model_map, model_size, up_offset, in_dim, out_dim, x, 1,
                ) != 0
                && ds4_gpu_swiglu_tensor(
                    mid,
                    gate.cast_const(),
                    up.cast_const(),
                    out_dim as u32,
                    clamp,
                    1.0,
                ) != 0
        });
    }
    status(|| {
        let Some(gate_tensor) = (unsafe { tensor_ref(gate.cast_const()) }) else {
            return false;
        };
        let Some(up_tensor) = (unsafe { tensor_ref(up.cast_const()) }) else {
            return false;
        };
        let Some(x_tensor) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        unsafe {
            matmul_q8_pair_fused_impl(
                gate_tensor,
                up_tensor,
                model_map,
                model_size,
                gate_offset,
                up_offset,
                in_dim,
                out_dim,
                x_tensor,
            ) && ds4_gpu_swiglu_tensor(
                mid,
                gate.cast_const(),
                up.cast_const(),
                out_dim as u32,
                clamp,
                1.0,
            ) != 0
        }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_q8_hc_expand_fused_impl(
    out_hc: &Ds4GpuTensor,
    block_out: &Ds4GpuTensor,
    block_add: Option<&Ds4GpuTensor>,
    has_add: bool,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &Ds4GpuTensor,
    residual_hc: &Ds4GpuTensor,
    split: &Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> bool {
    let Some((_weight_elements, _weight_elements_usize, weight_bytes)) =
        abi_q8_shape(in_dim, out_dim)
    else {
        return false;
    };
    let Some(hc_elements) = u64::from(n_embd).checked_mul(u64::from(n_hc)) else {
        return false;
    };
    let Some(hc_bytes) = hc_elements.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(mix_hc) = n_hc.checked_mul(n_hc).and_then(|comb| {
        n_hc.checked_mul(2)
            .and_then(|prefix| prefix.checked_add(comb))
    }) else {
        return false;
    };
    let Some(split_bytes) = u64::from(mix_hc).checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(x_bytes) = in_dim.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let Some(block_bytes) = out_dim.checked_mul(size_of::<f32>() as u64) else {
        return false;
    };
    let block_add = block_add.unwrap_or(block_out);
    if model_map.is_null()
        || in_dim == 0
        || out_dim == 0
        || n_embd == 0
        || n_hc == 0
        || out_dim != u64::from(n_embd)
        || weight_offset > model_size
        || weight_bytes > model_size - weight_offset
        || x.bytes < x_bytes
        || block_out.bytes < block_bytes
        || block_add.bytes < block_bytes
        || residual_hc.bytes < hc_bytes
        || split.bytes < split_bytes
        || out_hc.bytes < hc_bytes
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
            |packed_weight_ptr| {
                let blocks = in_dim.div_ceil(32);
                let Some(quantized_elements) = blocks.checked_mul(32) else {
                    return Some(false);
                };
                let Some(quantized_elements) = usize::try_from(quantized_elements).ok() else {
                    return Some(false);
                };
                let Some(scale_elements) = usize::try_from(blocks).ok() else {
                    return Some(false);
                };
                with_abi_q8_activations(
                    backend,
                    quantized_elements,
                    scale_elements,
                    |activations| {
                        with_abi_kernels(backend, |kernels| {
                            if !unsafe {
                                kernels.quantize_q8_f32_tensor(
                                    backend.stream(),
                                    x.device_ptr(),
                                    activations.quantized.cu_deviceptr(),
                                    activations.scales.cu_deviceptr(),
                                    in_dim,
                                    blocks,
                                    1,
                                )
                            } {
                                return Some(false);
                            }
                            Some(unsafe {
                                kernels.matmul_q8_hc_expand_tensor(
                                    backend.stream(),
                                    out_hc.device_ptr(),
                                    block_out.device_ptr(),
                                    block_add.device_ptr(),
                                    residual_hc.device_ptr(),
                                    split.device_ptr(),
                                    packed_weight_ptr,
                                    activations.quantized.cu_deviceptr(),
                                    activations.scales.cu_deviceptr(),
                                    in_dim,
                                    out_dim,
                                    n_embd,
                                    n_hc,
                                    has_add,
                                    q8_dp4a_enabled(
                                        std::env::var_os("DS4_CUDA_NO_Q8_DP4A").is_some(),
                                    ),
                                )
                            })
                        })
                    },
                )
            },
        )
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_matmul_q8_0_hc_expand_tensor(
    out_hc: *mut Ds4GpuTensor,
    block_out: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    split: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    if std::env::var_os("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED").is_some() {
        return status(|| unsafe {
            ds4_gpu_matmul_q8_0_tensor(
                block_out,
                model_map,
                model_size,
                weight_offset,
                in_dim,
                out_dim,
                x,
                1,
            ) != 0
                && ds4_gpu_hc_expand_split_tensor(
                    out_hc,
                    block_out,
                    residual_hc,
                    split,
                    n_embd,
                    n_hc,
                ) != 0
        });
    }
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(block_out) = (unsafe { tensor_ref(block_out.cast_const()) }) else {
            return false;
        };
        let Some(x) = (unsafe { tensor_ref(x) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(split) = (unsafe { tensor_ref(split) }) else {
            return false;
        };
        unsafe {
            matmul_q8_hc_expand_fused_impl(
                out_hc,
                block_out,
                None,
                false,
                model_map,
                model_size,
                weight_offset,
                in_dim,
                out_dim,
                x,
                residual_hc,
                split,
                n_embd,
                n_hc,
            )
        }
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_shared_down_hc_expand_q8_0_tensor(
    out_hc: *mut Ds4GpuTensor,
    shared_out: *mut Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    shared_mid: *const Ds4GpuTensor,
    routed_out: *const Ds4GpuTensor,
    residual_hc: *const Ds4GpuTensor,
    split: *const Ds4GpuTensor,
    n_embd: u32,
    n_hc: u32,
) -> c_int {
    if std::env::var_os("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED").is_some() {
        return status(|| unsafe {
            ds4_gpu_matmul_q8_0_tensor(
                shared_out,
                model_map,
                model_size,
                weight_offset,
                in_dim,
                out_dim,
                shared_mid,
                1,
            ) != 0
                && ds4_gpu_hc_expand_add_split_tensor(
                    out_hc,
                    shared_out,
                    routed_out,
                    residual_hc,
                    split,
                    n_embd,
                    n_hc,
                ) != 0
        });
    }
    status(|| {
        let Some(out_hc) = (unsafe { tensor_ref(out_hc.cast_const()) }) else {
            return false;
        };
        let Some(shared_out) = (unsafe { tensor_ref(shared_out.cast_const()) }) else {
            return false;
        };
        let Some(shared_mid) = (unsafe { tensor_ref(shared_mid) }) else {
            return false;
        };
        let Some(routed_out) = (unsafe { tensor_ref(routed_out) }) else {
            return false;
        };
        let Some(residual_hc) = (unsafe { tensor_ref(residual_hc) }) else {
            return false;
        };
        let Some(split) = (unsafe { tensor_ref(split) }) else {
            return false;
        };
        unsafe {
            matmul_q8_hc_expand_fused_impl(
                out_hc,
                shared_out,
                Some(routed_out),
                true,
                model_map,
                model_size,
                weight_offset,
                in_dim,
                out_dim,
                shared_mid,
                residual_hc,
                split,
                n_embd,
                n_hc,
            )
        }
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

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
    q_out: *mut Ds4GpuTensor,
    q: *const Ds4GpuTensor,
    model_map: *const c_void,
    model_size: u64,
    q_weight_offset: u64,
    q_n: u32,
    kv_out: *mut Ds4GpuTensor,
    kv: *const Ds4GpuTensor,
    kv_weight_offset: u64,
    kv_n: u32,
    rows: u32,
    eps: f32,
) -> c_int {
    if std::env::var_os("DS4_CUDA_DISABLE_QKV_RMS_FUSED").is_some() {
        return if unsafe {
            ds4_gpu_rms_norm_weight_rows_tensor(
                q_out,
                q,
                model_map,
                model_size,
                q_weight_offset,
                q_n,
                rows,
                eps,
            ) != 0
                && ds4_gpu_rms_norm_weight_rows_tensor(
                    kv_out,
                    kv,
                    model_map,
                    model_size,
                    kv_weight_offset,
                    kv_n,
                    rows,
                    eps,
                ) != 0
        } {
            1
        } else {
            0
        };
    }
    status(|| {
        let Some(q_out) = (unsafe { tensor_ref(q_out.cast_const()) }) else {
            return false;
        };
        let Some(q) = (unsafe { tensor_ref(q) }) else {
            return false;
        };
        let Some(kv_out) = (unsafe { tensor_ref(kv_out.cast_const()) }) else {
            return false;
        };
        let Some(kv) = (unsafe { tensor_ref(kv) }) else {
            return false;
        };
        let Some(q_count) = u64::from(q_n).checked_mul(u64::from(rows)) else {
            return false;
        };
        let Some(kv_count) = u64::from(kv_n).checked_mul(u64::from(rows)) else {
            return false;
        };
        let Some(q_bytes) = q_count.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let Some(kv_bytes) = kv_count.checked_mul(size_of::<f32>() as u64) else {
            return false;
        };
        let q_weight_bytes = u64::from(q_n) * size_of::<f32>() as u64;
        let kv_weight_bytes = u64::from(kv_n) * size_of::<f32>() as u64;
        if model_map.is_null()
            || q_n == 0
            || kv_n == 0
            || rows == 0
            || q_weight_offset > model_size
            || q_weight_bytes > model_size - q_weight_offset
            || kv_weight_offset > model_size
            || kv_weight_bytes > model_size - kv_weight_offset
            || q_out.bytes < q_bytes
            || q.bytes < q_bytes
            || kv_out.bytes < kv_bytes
            || kv.bytes < kv_bytes
        {
            return false;
        }
        with_backend(|backend| {
            with_cached_abi_model_range(
                backend,
                model_map,
                model_size,
                q_weight_offset,
                q_weight_bytes,
                |q_weight_ptr| {
                    with_cached_abi_model_range(
                        backend,
                        model_map,
                        model_size,
                        kv_weight_offset,
                        kv_weight_bytes,
                        |kv_weight_ptr| {
                            with_abi_kernels(backend, |kernels| {
                                // SAFETY: all tensor and cached model-weight
                                // spans are validated before the fused launch.
                                Some(unsafe {
                                    kernels.dsv4_qkv_rms_norm_rows_tensor(
                                        backend.stream(),
                                        q_out.device_ptr(),
                                        q.device_ptr(),
                                        q_weight_ptr,
                                        q_n,
                                        kv_out.device_ptr(),
                                        kv.device_ptr(),
                                        kv_weight_ptr,
                                        kv_n,
                                        rows,
                                        eps,
                                    )
                                })
                            })
                        },
                    )
                },
            )
        })
        .unwrap_or(false)
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
            #[cfg(feature = "cuda-oxide-kernels")]
            {
                let mut q8_cache = ABI_Q8_CACHE.lock().ok()?;
                clear_abi_q8_converted_ranges(&mut q8_cache);
            }
            #[cfg(target_os = "linux")]
            {
                let mut model_arenas = ABI_MODEL_ARENAS.lock().ok()?;
                model_arenas.arenas.clear();
                model_arenas.range_bytes = 0;
                model_arenas.cache_full = false;
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
                Some(abi_model_range_is_cached(
                    model_map, model_size, offset, bytes,
                ))
            })
        })
        .unwrap_or(false)
    })
}

#[cfg(feature = "cuda-oxide-kernels")]
#[no_mangle]
pub unsafe extern "C" fn ds4_gpu_cache_q8_f16_range(
    model_map: *const c_void,
    model_size: u64,
    offset: u64,
    bytes: u64,
    in_dim: u64,
    out_dim: u64,
    label: *const c_char,
) -> c_int {
    status(|| {
        if model_map.is_null() || bytes == 0 {
            return true;
        }
        if offset > model_size || bytes > model_size - offset {
            return false;
        }
        let label = if label.is_null() {
            "q8_0".into()
        } else {
            // SAFETY: the C ABI supplies a NUL-terminated optional label for
            // the duration of this synchronous policy decision.
            unsafe { CStr::from_ptr(label) }.to_string_lossy()
        };
        with_backend(|backend| {
            let format = {
                let cache = ABI_Q8_CACHE.lock().ok()?;
                q8_preload_format(
                    abi_q8_cache_options(),
                    Some(label.as_ref()),
                    in_dim,
                    out_dim,
                    cache.state,
                )
            };
            Some(match format {
                Some(Q8PreloadFormat::F16) => cache_abi_q8_f16_range(
                    backend,
                    model_map,
                    model_size,
                    offset,
                    bytes,
                    in_dim,
                    out_dim,
                    label.as_ref(),
                    true,
                ),
                Some(Q8PreloadFormat::F32) => cache_abi_q8_f32_range(
                    backend, model_map, model_size, offset, bytes, in_dim, out_dim,
                ),
                None => true,
            })
        })
        .unwrap_or(false)
    })
}

#[no_mangle]
pub extern "C" fn ds4_gpu_print_memory_report(label: *const c_char) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let label = if label.is_null() {
            "".into()
        } else {
            // SAFETY: the C ABI supplies a NUL-terminated optional label for
            // the duration of this synchronous diagnostic call.
            unsafe { CStr::from_ptr(label) }.to_string_lossy()
        };
        let _ = with_backend(|backend| {
            let memory = backend.memory_capacity().ok()?;
            eprintln!(
                "ds4: CUDA memory report {}: free {:.2} MiB total {:.2} MiB",
                label,
                memory.free_bytes as f64 / 1048576.0,
                memory.total_bytes as f64 / 1048576.0
            );
            Some(())
        });
    }));
}

#[no_mangle]
pub extern "C" fn ds4_gpu_set_quality(quality: bool) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        ABI_QUALITY_MODE.store(quality, Ordering::Relaxed);
        update_abi_blas_math_state();
    }));
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
