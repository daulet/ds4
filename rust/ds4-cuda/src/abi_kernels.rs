use std::ffi::c_void;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use cuda_core::embedded::{
    embedded_modules_from_current_exe, ArtifactPayloadKind, EmbeddedModuleError,
};
use cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir::{self, LtoirError};

const THREADS_PER_BLOCK: u32 = 256;
const ABI_KERNEL_ARTIFACT: &str = "ds4-cuda";

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn abi_add_kernel(count: u32, a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i < count as usize {
            if let Some(element) = out.get_mut(index) {
                *element = a[i] + b[i];
            }
        }
    }

    #[kernel]
    pub fn abi_repeat_hc_kernel(count: u64, n_embd: u32, row: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if (i as u64) < count {
            if let Some(element) = out.get_mut(index) {
                *element = row[i % n_embd as usize];
            }
        }
    }

    #[kernel]
    pub fn abi_directional_steering_project_kernel(
        layer: u32,
        width: u32,
        rows: u32,
        scale: f32,
        directions: &[f32],
        mut x: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows || width == 0 {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let width = width as usize;
        let x_base = row as usize * width;
        let direction_base = layer as usize * width;
        let x_ptr = x.as_mut_ptr();

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < width {
            unsafe {
                sum += *x_ptr.add(x_base + i) * directions[direction_base + i];
            }
            i += nth;
        }

        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();

        let mut stride = nth >> 1;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let coefficient = unsafe { scale * PARTIAL[0] };
        i = tid;
        while i < width {
            unsafe {
                *x.get_unchecked_mut(x_base + i) -= coefficient * directions[direction_base + i];
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn abi_swiglu_kernel(
        count: u32,
        clamp: f32,
        weight: f32,
        gate: &[f32],
        up: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= count as usize {
            return;
        }

        let mut g = gate[i];
        let mut u = up[i];
        if clamp > 1.0e-6_f32 {
            if (g.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || g > clamp {
                g = clamp;
            }
            if (u.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || u < -clamp {
                u = -clamp;
            } else if u > clamp {
                u = clamp;
            }
        }
        if let Some(element) = out.get_mut(index) {
            *element = (g / (1.0_f32 + (-g).exp())) * u * weight;
        }
    }

    #[kernel]
    pub fn abi_rms_norm_plain_kernel(
        n: u32,
        rows: u32,
        eps: f32,
        x: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let n = n as usize;
        let base = row as usize * n;

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            let value = x[base + i];
            sum += value * value;
            i += nth;
        }
        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();

        let mut stride = nth >> 1;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let scale = 1.0_f32 / (unsafe { PARTIAL[0] } / n as f32 + eps).sqrt();
        i = tid;
        while i < n {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale;
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn abi_rms_norm_weight_kernel(
        n: u32,
        rows: u32,
        eps: f32,
        x: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let n = n as usize;
        let base = row as usize * n;

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            let value = x[base + i];
            sum += value * value;
            i += nth;
        }
        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();

        let mut stride = nth >> 1;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let scale = 1.0_f32 / (unsafe { PARTIAL[0] } / n as f32 + eps).sqrt();
        i = tid;
        while i < n {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale * weight[i];
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn abi_matmul_f16_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        weights: &[f16],
        x: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        if row >= out_dim || token >= n_tok {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let weight_base = row as usize * in_dim as usize;
        let x_base = token as usize * in_dim as usize;
        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < in_dim as usize {
            sum += weights[weight_base + i] as f32 * x[x_base + i];
            i += nth;
        }
        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = nth >> 1;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(token as usize * out_dim as usize + row as usize) =
                    PARTIAL[0];
            }
        }
    }

    #[kernel]
    pub fn abi_matmul_f16_serial_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        weights: &[f16],
        x: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        if row >= out_dim || token >= n_tok || thread::threadIdx_x() != 0 {
            return;
        }
        let weight_base = row as usize * in_dim as usize;
        let x_base = token as usize * in_dim as usize;
        let mut sum = 0.0_f32;
        let mut i = 0_usize;
        while i < in_dim as usize {
            sum += weights[weight_base + i] as f32 * x[x_base + i];
            i += 1;
        }
        unsafe {
            *out.get_unchecked_mut(token as usize * out_dim as usize + row as usize) = sum;
        }
    }

    #[kernel]
    pub fn abi_matmul_f16_ordered_chunks_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        weights: &[f16],
        x: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 32> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        if row >= out_dim || token >= n_tok {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let chunk = (in_dim as usize + 31) / 32;
        let start = tid * chunk;
        let mut end = start + chunk;
        if end > in_dim as usize {
            end = in_dim as usize;
        }
        let weight_base = row as usize * in_dim as usize;
        let x_base = token as usize * in_dim as usize;
        let mut sum = 0.0_f32;
        let mut i = start;
        while i < end {
            sum += weights[weight_base + i] as f32 * x[x_base + i];
            i += 1;
        }
        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();
        if tid == 0 {
            let mut total = 0.0_f32;
            let mut lane = 0_usize;
            while lane < 32 {
                unsafe {
                    total += PARTIAL[lane];
                }
                lane += 1;
            }
            unsafe {
                *out.get_unchecked_mut(token as usize * out_dim as usize + row as usize) = total;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AbiKernelModule {
    add_kernel: CudaFunction,
    repeat_hc_kernel: CudaFunction,
    directional_steering_project_kernel: CudaFunction,
    swiglu_kernel: CudaFunction,
    rms_norm_plain_kernel: CudaFunction,
    rms_norm_weight_kernel: CudaFunction,
    matmul_f16_kernel: CudaFunction,
    matmul_f16_serial_kernel: CudaFunction,
    matmul_f16_ordered_chunks_kernel: CudaFunction,
}

impl AbiKernelModule {
    pub(crate) fn load(context: &Arc<CudaContext>) -> Result<Self, AbiKernelLoadError> {
        let module = load_abi_module(context)?;
        Ok(Self {
            add_kernel: module
                .load_function("abi_add_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            repeat_hc_kernel: module
                .load_function("abi_repeat_hc_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            directional_steering_project_kernel: module
                .load_function("abi_directional_steering_project_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            swiglu_kernel: module
                .load_function("abi_swiglu_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            rms_norm_plain_kernel: module
                .load_function("abi_rms_norm_plain_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            rms_norm_weight_kernel: module
                .load_function("abi_rms_norm_weight_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_f16_kernel: module
                .load_function("abi_matmul_f16_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_f16_serial_kernel: module
                .load_function("abi_matmul_f16_serial_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_f16_ordered_chunks_kernel: module
                .load_function("abi_matmul_f16_ordered_chunks_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
        })
    }

    pub(crate) unsafe fn add_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        a_ptr: u64,
        b_ptr: u64,
        count: u32,
    ) -> bool {
        let Some(config) = launch_config(u64::from(count)) else {
            return false;
        };
        let mut count = count;
        let mut a_ptr = a_ptr;
        let mut a_len = u64::from(count);
        let mut b_ptr = b_ptr;
        let mut b_len = u64::from(count);
        let mut out_ptr = out_ptr;
        let mut out_len = u64::from(count);
        let mut params = [
            (&mut count as *mut u32).cast::<c_void>(),
            (&mut a_ptr as *mut u64).cast::<c_void>(),
            (&mut a_len as *mut u64).cast::<c_void>(),
            (&mut b_ptr as *mut u64).cast::<c_void>(),
            (&mut b_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates every device range and holds the
        // owning CUDA context and loaded module through launch submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.add_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn repeat_hc_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        row_ptr: u64,
        n_embd: u32,
        n_hc: u32,
    ) -> bool {
        let count = u64::from(n_embd) * u64::from(n_hc);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut count = count;
        let mut n_embd = n_embd;
        let mut row_ptr = row_ptr;
        let mut row_len = u64::from(n_embd);
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut count as *mut u64).cast::<c_void>(),
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut row_ptr as *mut u64).cast::<c_void>(),
            (&mut row_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates every device range and holds the
        // owning CUDA context and loaded module through launch submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.repeat_hc_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn directional_steering_project_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        directions_ptr: u64,
        layer: u32,
        width: u32,
        rows: u32,
        scale: f32,
    ) -> bool {
        let mut threads = THREADS_PER_BLOCK;
        while threads > width && threads > 1 {
            threads >>= 1;
        }
        let config = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut layer = layer;
        let mut width = width;
        let mut rows = rows;
        let mut scale = scale;
        let mut directions_ptr = directions_ptr;
        let mut directions_len = (u64::from(layer) + 1) * u64::from(width);
        let mut x_ptr = x_ptr;
        let mut x_len = u64::from(rows) * u64::from(width);
        let mut params = [
            (&mut layer as *mut u32).cast::<c_void>(),
            (&mut width as *mut u32).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut scale as *mut f32).cast::<c_void>(),
            (&mut directions_ptr as *mut u64).cast::<c_void>(),
            (&mut directions_len as *mut u64).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates every device range and holds the
        // owning CUDA context and loaded module through launch submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.directional_steering_project_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn swiglu_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        gate_ptr: u64,
        up_ptr: u64,
        count: u32,
        clamp: f32,
        weight: f32,
    ) -> bool {
        let Some(config) = launch_config(u64::from(count)) else {
            return false;
        };
        let mut count = count;
        let mut clamp = clamp;
        let mut weight = weight;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = u64::from(count);
        let mut up_ptr = up_ptr;
        let mut up_len = u64::from(count);
        let mut out_ptr = out_ptr;
        let mut out_len = u64::from(count);
        let mut params = [
            (&mut count as *mut u32).cast::<c_void>(),
            (&mut clamp as *mut f32).cast::<c_void>(),
            (&mut weight as *mut f32).cast::<c_void>(),
            (&mut gate_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_len as *mut u64).cast::<c_void>(),
            (&mut up_ptr as *mut u64).cast::<c_void>(),
            (&mut up_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates every device range and holds the
        // owning CUDA context and loaded module through launch submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.swiglu_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn rms_norm_plain_rows_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        x_ptr: u64,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = u64::from(n) * u64::from(rows);
        let mut n = n;
        let mut rows = rows;
        let mut eps = eps;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates every device range and holds the
        // owning CUDA context and loaded module through launch submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.rms_norm_plain_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn rms_norm_weight_rows_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        x_ptr: u64,
        weight_ptr: u64,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = u64::from(n) * u64::from(rows);
        let mut n = n;
        let mut rows = rows;
        let mut eps = eps;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut weight_ptr = weight_ptr;
        let mut weight_len = u64::from(n);
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut weight_ptr as *mut u64).cast::<c_void>(),
            (&mut weight_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates every device range and retains
        // cached model weights through launch submission and later syncs.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.rms_norm_weight_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn matmul_f16_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        weight_ptr: u64,
        x_ptr: u64,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        path: crate::F16ProjectionPath,
    ) -> bool {
        let threads = match path {
            crate::F16ProjectionPath::Base => THREADS_PER_BLOCK,
            crate::F16ProjectionPath::Serial => 1,
            crate::F16ProjectionPath::OrderedChunks => 32,
            crate::F16ProjectionPath::Blas => return false,
        };
        let grid_x = match u32::try_from(out_dim) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let grid_y = match u32::try_from(n_tok) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let function = match path {
            crate::F16ProjectionPath::Base => &self.matmul_f16_kernel,
            crate::F16ProjectionPath::Serial => &self.matmul_f16_serial_kernel,
            crate::F16ProjectionPath::OrderedChunks => &self.matmul_f16_ordered_chunks_kernel,
            crate::F16ProjectionPath::Blas => return false,
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut in_dim = in_dim;
        let mut out_dim = out_dim;
        let mut n_tok = n_tok;
        let mut weight_ptr = weight_ptr;
        let mut weight_len = in_dim * out_dim;
        let mut x_ptr = x_ptr;
        let mut x_len = in_dim * n_tok;
        let mut out_ptr = out_ptr;
        let mut out_len = out_dim * n_tok;
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut out_dim as *mut u64).cast::<c_void>(),
            (&mut n_tok as *mut u64).cast::<c_void>(),
            (&mut weight_ptr as *mut u64).cast::<c_void>(),
            (&mut weight_len as *mut u64).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates single-token output/input ranges and
        // retains the model-backed F16 weight range through launch submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }
}

fn launch_config(count: u64) -> Option<LaunchConfig> {
    let blocks = count.div_ceil(u64::from(THREADS_PER_BLOCK));
    let grid_x = u32::try_from(blocks).ok()?;
    (grid_x > 0).then_some(LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn load_abi_module(context: &Arc<CudaContext>) -> Result<Arc<CudaModule>, AbiKernelLoadError> {
    let module = embedded_modules_from_current_exe()
        .map_err(AbiKernelLoadError::Embedded)?
        .into_iter()
        .find(|module| module.name() == ABI_KERNEL_ARTIFACT)
        .ok_or_else(|| {
            AbiKernelLoadError::Embedded(EmbeddedModuleError::ModuleNotFound {
                name: ABI_KERNEL_ARTIFACT.to_string(),
            })
        })?;
    let ptx = module
        .payload(ArtifactPayloadKind::Ptx)
        .ok_or_else(|| AbiKernelLoadError::Embedded(EmbeddedModuleError::NoModules))?;
    if !ptx.windows(b"__nv_".len()).any(|window| window == b"__nv_") {
        return module.load(context).map_err(AbiKernelLoadError::Embedded);
    }

    let artifact_dir = std::env::temp_dir().join(format!("ds4-cuda-abi-{}", std::process::id()));
    std::fs::create_dir_all(&artifact_dir).map_err(|source| AbiKernelLoadError::Io {
        path: artifact_dir.clone(),
        source,
    })?;
    let ptx_path = artifact_dir.join("ds4-cuda-abi.ptx");
    std::fs::write(&ptx_path, ptx).map_err(|source| AbiKernelLoadError::Io {
        path: ptx_path.clone(),
        source,
    })?;
    let cubin_path =
        ltoir::build_cubin_from_ptx_with_libdevice(&ptx_path, &link_target_arch(context)?)
            .map_err(AbiKernelLoadError::Link)?;
    let loaded = context
        .load_module_from_file(cubin_path.to_string_lossy().as_ref())
        .map_err(AbiKernelLoadError::Driver)?;
    std::fs::remove_dir_all(&artifact_dir).map_err(|source| AbiKernelLoadError::Io {
        path: artifact_dir,
        source,
    })?;
    Ok(loaded)
}

fn link_target_arch(context: &CudaContext) -> Result<String, AbiKernelLoadError> {
    if let Ok(arch) = std::env::var("CUDA_OXIDE_LINK_TARGET") {
        return Ok(arch);
    }
    let (major, minor) = context
        .compute_capability()
        .map_err(AbiKernelLoadError::Driver)?;
    Ok(format!("sm_{major}{minor}"))
}

#[derive(Debug)]
pub(crate) enum AbiKernelLoadError {
    Embedded(EmbeddedModuleError),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Link(LtoirError),
    Driver(DriverError),
}

impl fmt::Display for AbiKernelLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedded(error) => error.fmt(formatter),
            Self::Io { path, source } => {
                write!(formatter, "failed to write {}: {source}", path.display())
            }
            Self::Link(error) => error.fmt(formatter),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AbiKernelLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Embedded(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            Self::Link(error) => Some(error),
            Self::Driver(error) => Some(error),
        }
    }
}
