use std::ffi::c_void;
use std::fmt;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::Arc;

use cuda_core::embedded::{
    artifact_bundles_from_binary_path, embedded_modules_from_current_exe, ArtifactPayloadKind,
    EmbeddedModule, EmbeddedModuleError,
};
use cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig};
use cuda_device::mma::{load_a_m16n8k16, load_b_m16n8k16, mma_m16n8k16_f32_f16, zero_accumulator};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicU32},
    cuda_module, device, integer, kernel, thread, warp, DisjointSlice, DynamicSharedArray,
    SharedArray,
};
use cuda_host::ltoir::{self, LtoirError};

use crate::{IndexerScoreKernel, IndexerTopkKernel};

#[device]
unsafe extern "C" {
    fn __nv_rsqrtf(value: f32) -> f32;
}

// Device extern wrappers remain in the host DSO; satisfy that unused host edge.
#[unsafe(export_name = "__nv_rsqrtf")]
extern "C" fn host_libdevice_rsqrtf_stub(value: f32) -> f32 {
    1.0_f32 / value.sqrt()
}

const THREADS_PER_BLOCK: u32 = 256;
const ABI_KERNEL_ARTIFACT: &str = "ds4-cuda";
const ABI_ROUTER_N_EXPERT: usize = 256;
const ABI_ROUTER_TOP_K: usize = 6;
const ABI_ROUTER_ROWS_PER_WARP_BLOCK: u32 = 4;
const ABI_MOE_QK_K: usize = 256;
const ABI_MOE_IQ2_BLOCK_BYTES: u64 = 66;
const ABI_MOE_Q2_BLOCK_BYTES: u64 = 84;
const ABI_MOE_Q4_BLOCK_BYTES: u64 = 144;
const ABI_MOE_Q8_K_BLOCK_BYTES: u64 = 292;
const ABI_MOE_SORTED_EXPERTS: usize = 256;
const ABI_MOE_CACHED_GATE_MAX_BLOCKS: usize = 16;
const ABI_MOE_CACHED_DOWN_MAX_BLOCKS: usize = 8;
const ABI_INDEXER_HEAD_DIM: usize = 128;
const ABI_INDEXER_N_HEAD: usize = 64;
const ABI_INDEXER_TILE_TOKENS: usize = 16;
const ABI_INDEXER_TILE_COMPONENTS: usize = 16;
const ABI_INDEXER_MMA_K: usize = 16;
const ABI_INDEXER_MMA_N: usize = 8;
const ABI_INDEXER_DIRECT_THREADS: u32 = 128;
const ABI_INDEXER_WMMA_THREADS: u32 = 32;
const ABI_INDEXER_WMMA32_COMPONENTS: usize = 32;
const ABI_INDEXER_WMMA32_WARPS: usize = 2;
const ABI_INDEXER_WMMA32_THREADS: u32 = 64;
const ABI_INDEXER_WMMA64_COMPONENTS: usize = 64;
const ABI_INDEXER_WMMA64_WARPS: usize = 4;
const ABI_INDEXER_WMMA64_THREADS: u32 = 128;
const ABI_INDEXER_WMMA128_COMPONENTS: usize = 128;
const ABI_INDEXER_WMMA128_WARPS: usize = 8;
const ABI_INDEXER_WMMA128_THREADS: u32 = 256;
const ABI_INDEXER_TOPK_1024_SORT_N: usize = 1024;
const ABI_INDEXER_TOPK_2048_SORT_N: usize = 2048;
const ABI_INDEXER_TOPK_4096_SORT_N: usize = 4096;
const ABI_INDEXER_TOPK_8192_SORT_N: usize = 8192;
const ABI_INDEXER_TOPK_THREADS: u32 = 1024;
const ABI_INDEXER_TOPK_PACKED_THREADS: u32 = 512;
const ABI_INDEXER_TOPK_PACKED_ITEMS_PER_THREAD: u32 = 16;
const ABI_INDEXER_TOPK_PACKED_SHARED_KEY_BYTES: u32 =
    (ABI_INDEXER_TOPK_8192_SORT_N * std::mem::size_of::<u64>()) as u32;
const ABI_INDEXER_TOPK_EMPTY_KEY: u64 = 0x007f_ffff_u64 << 32;
const ABI_INDEXER_TOPK_MERGE_GROUP: u32 = 8;
pub(crate) const ABI_MOE_IQ2_SIGNS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];
pub(crate) const ABI_MOE_IQ2_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

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
    pub fn abi_hc_split_sinkhorn_kernel(
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        mut split: DisjointSlice<f32>,
    ) {
        let row = thread::index_1d().get();
        if row < n_rows as usize {
            abi_hc4_split_one(row, sinkhorn_iters, eps, mix, scale, base, &mut split);
        }
    }

    fn abi_hc4_split_one(
        row: usize,
        sinkhorn_iters: u32,
        eps: f32,
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        split: &mut DisjointSlice<f32>,
    ) {
        const N_HC: usize = 4;
        const MIX_HC: usize = 24;

        let input = row * MIX_HC;
        let mut hc = 0_usize;
        while hc < N_HC {
            let pre = mix[input + hc] * scale[0] + base[hc];
            let post = mix[input + N_HC + hc] * scale[1] + base[N_HC + hc];
            unsafe {
                *split.get_unchecked_mut(input + hc) = 1.0 / (1.0 + (-pre).exp()) + eps;
                *split.get_unchecked_mut(input + N_HC + hc) = 2.0 / (1.0 + (-post).exp());
            }
            hc += 1;
        }
        let mut combinations = [0.0_f32; 16];
        let mut source = 0_usize;
        while source < N_HC {
            let first =
                mix[input + 2 * N_HC + source * N_HC] * scale[2] + base[2 * N_HC + source * N_HC];
            let mut maximum = first;
            let mut destination = 0_usize;
            while destination < N_HC {
                let index = source * N_HC + destination;
                let value = mix[input + 2 * N_HC + index] * scale[2] + base[2 * N_HC + index];
                combinations[index] = value;
                if value > maximum {
                    maximum = value;
                }
                destination += 1;
            }
            let mut sum = 0.0_f32;
            destination = 0;
            while destination < N_HC {
                let index = source * N_HC + destination;
                let value = (combinations[index] - maximum).exp();
                combinations[index] = value;
                sum += value;
                destination += 1;
            }
            destination = 0;
            while destination < N_HC {
                let index = source * N_HC + destination;
                combinations[index] = combinations[index] / sum + eps;
                destination += 1;
            }
            source += 1;
        }
        let mut column = 0_usize;
        while column < N_HC {
            let mut sum = eps;
            let mut row_index = 0_usize;
            while row_index < N_HC {
                sum += combinations[row_index * N_HC + column];
                row_index += 1;
            }
            row_index = 0;
            while row_index < N_HC {
                let index = row_index * N_HC + column;
                combinations[index] /= sum;
                row_index += 1;
            }
            column += 1;
        }
        let mut iteration = 1_u32;
        while iteration < sinkhorn_iters {
            source = 0;
            while source < N_HC {
                let mut sum = eps;
                column = 0;
                while column < N_HC {
                    sum += combinations[source * N_HC + column];
                    column += 1;
                }
                column = 0;
                while column < N_HC {
                    let index = source * N_HC + column;
                    combinations[index] /= sum;
                    column += 1;
                }
                source += 1;
            }
            column = 0;
            while column < N_HC {
                let mut sum = eps;
                let mut row_index = 0_usize;
                while row_index < N_HC {
                    sum += combinations[row_index * N_HC + column];
                    row_index += 1;
                }
                row_index = 0;
                while row_index < N_HC {
                    let index = row_index * N_HC + column;
                    combinations[index] /= sum;
                    row_index += 1;
                }
                column += 1;
            }
            iteration += 1;
        }
        let mut index = 0_usize;
        while index < 16 {
            unsafe {
                *split.get_unchecked_mut(input + 2 * N_HC + index) = combinations[index];
            }
            index += 1;
        }
    }

    #[kernel]
    pub fn abi_hc_split_weighted_sum_fused_kernel(
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        mix: &[f32],
        residual_hc: &[f32],
        scale: &[f32],
        base: &[f32],
        mut split: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
    ) {
        const N_HC: u32 = 4;
        const MIX_HC: u64 = 24;

        let token = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if token >= n_rows || n_hc != N_HC {
            return;
        }
        if lane == 0 {
            abi_hc4_split_one(
                token as usize,
                sinkhorn_iters,
                eps,
                mix,
                scale,
                base,
                &mut split,
            );
        }
        thread::sync_threads();
        let split_ptr = split.as_mut_ptr();
        let split_base = u64::from(token) * MIX_HC;
        let mut dimension = u64::from(lane);
        while dimension < u64::from(n_embd) {
            let mut accumulator = 0.0_f32;
            let mut source_hc = 0_u64;
            while source_hc < u64::from(N_HC) {
                accumulator += residual_hc[((u64::from(token) * u64::from(N_HC) + source_hc)
                    * u64::from(n_embd)
                    + dimension) as usize]
                    * unsafe { *split_ptr.add((split_base + source_hc) as usize) };
                source_hc += 1;
            }
            unsafe {
                *out.get_unchecked_mut(
                    (u64::from(token) * u64::from(n_embd) + dimension) as usize,
                ) = accumulator;
            }
            dimension += u64::from(thread::blockDim_x());
        }
    }

    #[kernel]
    pub fn abi_hc_split_weighted_sum_norm_fused_kernel(
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
        mix: &[f32],
        residual_hc: &[f32],
        scale: &[f32],
        base: &[f32],
        norm_weight: &[f32],
        mut split: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut norm_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        const N_HC: u32 = 4;
        const MIX_HC: u64 = 24;

        let token = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if token >= n_rows || n_hc != N_HC {
            return;
        }
        if lane == 0 {
            abi_hc4_split_one(
                token as usize,
                sinkhorn_iters,
                eps,
                mix,
                scale,
                base,
                &mut split,
            );
        }
        thread::sync_threads();
        let split_ptr = split.as_mut_ptr();
        let split_base = u64::from(token) * MIX_HC;
        let mut sum = 0.0_f32;
        let mut dimension = u64::from(lane);
        while dimension < u64::from(n_embd) {
            let mut accumulator = 0.0_f32;
            let mut source_hc = 0_u64;
            while source_hc < u64::from(N_HC) {
                accumulator += residual_hc[((u64::from(token) * u64::from(N_HC) + source_hc)
                    * u64::from(n_embd)
                    + dimension) as usize]
                    * unsafe { *split_ptr.add((split_base + source_hc) as usize) };
                source_hc += 1;
            }
            unsafe {
                *out.get_unchecked_mut(
                    (u64::from(token) * u64::from(n_embd) + dimension) as usize,
                ) = accumulator;
            }
            sum += accumulator * accumulator;
            dimension += u64::from(thread::blockDim_x());
        }
        unsafe {
            PARTIAL[lane as usize] = sum;
        }
        thread::sync_threads();
        let mut stride = thread::blockDim_x() >> 1;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL[lane as usize] += PARTIAL[(lane + stride) as usize];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        let mean_square = unsafe { PARTIAL[0] } / n_embd as f32 + norm_eps;
        let norm_scale = unsafe { __nv_rsqrtf(mean_square) };
        dimension = u64::from(lane);
        while dimension < u64::from(n_embd) {
            let index = (u64::from(token) * u64::from(n_embd) + dimension) as usize;
            let value = unsafe { *out.as_mut_ptr().add(index) };
            unsafe {
                *norm_out.get_unchecked_mut(index) =
                    value * norm_scale * norm_weight[dimension as usize];
            }
            dimension += u64::from(thread::blockDim_x());
        }
    }

    #[kernel]
    pub fn abi_output_hc_weights_kernel(
        n_hc: u32,
        n_tokens: u32,
        eps: f32,
        pre: &[f32],
        scale: &[f32],
        base: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u64;
        let count = u64::from(n_tokens) * u64::from(n_hc);
        if index >= count {
            return;
        }
        let hc = (index % u64::from(n_hc)) as usize;
        let z = pre[index as usize] * scale[0] + base[hc];
        unsafe {
            *out.get_unchecked_mut(index as usize) = 1.0 / (1.0 + (-z).exp()) + eps;
        }
    }

    #[kernel]
    pub fn abi_hc_weighted_sum_kernel(
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        weight_stride: u32,
        residual_hc: &[f32],
        weights: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u64;
        if index >= u64::from(n_embd) * u64::from(n_tokens) {
            return;
        }
        let dimension = index % u64::from(n_embd);
        let token = index / u64::from(n_embd);
        let mut accumulator = 0.0_f32;
        let mut source_hc = 0_u64;
        while source_hc < u64::from(n_hc) {
            accumulator += residual_hc
                [((token * u64::from(n_hc) + source_hc) * u64::from(n_embd) + dimension) as usize]
                * weights[(token * u64::from(weight_stride) + source_hc) as usize];
            source_hc += 1;
        }
        unsafe {
            *out.get_unchecked_mut(index as usize) = accumulator;
        }
    }

    #[kernel]
    pub fn abi_hc_expand_kernel(
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        post_stride: u32,
        comb_stride: u32,
        has_add: u32,
        block_out: &[f32],
        block_add: &[f32],
        residual_hc: &[f32],
        post: &[f32],
        comb: &[f32],
        mut out_hc: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u64;
        let n_elem = u64::from(n_tokens) * u64::from(n_hc) * u64::from(n_embd);
        if index >= n_elem {
            return;
        }
        let dimension = index % u64::from(n_embd);
        let temporary = index / u64::from(n_embd);
        let destination_hc = temporary % u64::from(n_hc);
        let token = temporary / u64::from(n_hc);
        let block_index = (token * u64::from(n_embd) + dimension) as usize;
        let mut block_value = block_out[block_index];
        if has_add != 0 {
            block_value += block_add[block_index];
        }
        let mut accumulator =
            block_value * post[(token * u64::from(post_stride) + destination_hc) as usize];
        let mut source_hc = 0_u64;
        while source_hc < u64::from(n_hc) {
            accumulator += comb[(token * u64::from(comb_stride)
                + destination_hc
                + source_hc * u64::from(n_hc)) as usize]
                * residual_hc[(token * u64::from(n_hc) * u64::from(n_embd)
                    + source_hc * u64::from(n_embd)
                    + dimension) as usize];
            source_hc += 1;
        }
        unsafe {
            *out_hc.get_unchecked_mut(index as usize) = accumulator;
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

        let mean_square = unsafe { PARTIAL[0] } / n as f32 + eps;
        let scale = unsafe { __nv_rsqrtf(mean_square) };
        i = tid;
        while i < n {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale;
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn abi_head_rms_norm_kernel(
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
        mut x: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= n_tok * n_head {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let head_dim = head_dim as usize;
        let base = row as usize * head_dim;
        let x_ptr = x.as_mut_ptr();

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < head_dim {
            let value = unsafe { *x_ptr.add(base + i) };
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

        let mean_square = unsafe { PARTIAL[0] } / head_dim as f32 + eps;
        let scale = unsafe { __nv_rsqrtf(mean_square) };
        i = tid;
        while i < head_dim {
            unsafe {
                *x.get_unchecked_mut(base + i) *= scale;
            }
            i += nth;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_rope_tail_kernel(
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        pos_stride: u32,
        n_ctx_orig: u32,
        inverse: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        mut x: DisjointSlice<f32>,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let pairs_per_head = n_rot / 2;
        let pairs = n_tok * n_head * pairs_per_head;
        if gid >= pairs {
            return;
        }
        let pair = gid % pairs_per_head;
        let row = gid / pairs_per_head;
        let head = row % n_head;
        let token = row / n_head;
        let n_nope = head_dim - n_rot;
        let rot_i = pair * 2;

        let mut corr0 = 0.0_f32;
        let mut corr1 = 0.0_f32;
        if ext_factor != 0.0 {
            let denom = 2.0_f32 * freq_base.ln();
            corr0 = (n_rot as f32
                * (n_ctx_orig as f32 / (beta_fast * 2.0_f32 * 3.1415927_f32)).ln()
                / denom)
                .floor();
            corr1 = (n_rot as f32
                * (n_ctx_orig as f32 / (beta_slow * 2.0_f32 * 3.1415927_f32)).ln()
                / denom)
                .ceil();
            if corr0 < 0.0 {
                corr0 = 0.0;
            }
            if corr1 > (n_rot - 1) as f32 {
                corr1 = (n_rot - 1) as f32;
            }
        }

        let theta_extrap =
            (pos0 + token * pos_stride) as f32 * freq_base.powf(-(rot_i as f32) / n_rot as f32);
        let theta_interp = freq_scale * theta_extrap;
        let mut theta = theta_interp;
        let mut mscale = attn_factor;
        if ext_factor != 0.0 {
            let denom = if corr1 - corr0 > 0.001 {
                corr1 - corr0
            } else {
                0.001
            };
            let mut y = (pair as f32 - corr0) / denom;
            if y < 0.0 {
                y = 0.0;
            } else if y > 1.0 {
                y = 1.0;
            }
            let ramp_mix = (1.0 - y) * ext_factor;
            theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
        }
        let c = theta.cos() * mscale;
        let mut s = theta.sin() * mscale;
        if inverse != 0 {
            s = -s;
        }

        let base = ((u64::from(token) * u64::from(n_head) + u64::from(head)) * u64::from(head_dim)
            + u64::from(n_nope)
            + u64::from(rot_i)) as usize;
        let x0 = unsafe { *x.as_mut_ptr().add(base) };
        let x1 = unsafe { *x.as_mut_ptr().add(base + 1) };
        unsafe {
            *x.get_unchecked_mut(base) = x0 * c - x1 * s;
            *x.get_unchecked_mut(base + 1) = x0 * s + x1 * c;
        }
    }

    #[kernel]
    pub fn abi_store_raw_kv_batch_kernel(
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
        kv: &[f32],
        mut raw: DisjointSlice<f32>,
    ) {
        let gid = u64::from(thread::blockIdx_x()) * u64::from(thread::blockDim_x())
            + u64::from(thread::threadIdx_x());
        let count = u64::from(n_tokens) * u64::from(head_dim);
        if gid >= count {
            return;
        }
        let dimension = gid % u64::from(head_dim);
        let token = gid / u64::from(head_dim);
        let row = pos0.wrapping_add(token as u32) % raw_cap;
        unsafe {
            *raw.get_unchecked_mut((u64::from(row) * u64::from(head_dim) + dimension) as usize) =
                (kv[gid as usize] as f16) as f32;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_compressor_store_kernel(
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
    ) {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let gid = u64::from(thread::blockIdx_x()) * u64::from(thread::blockDim_x())
            + u64::from(thread::threadIdx_x());
        let count = u64::from(n_tokens) * u64::from(width);
        if gid >= count {
            return;
        }
        let token = gid / u64::from(width);
        let dimension = gid % u64::from(width);
        let phase = pos0.wrapping_add(token as u32) % ratio;
        let row = if ratio == 4 { ratio + phase } else { phase };
        let ape_index = (u64::from(phase) * u64::from(width) + dimension) as usize;
        let ape = if ape_type == 1 {
            ape_f16[ape_index] as f32
        } else {
            ape_f32[ape_index]
        };
        unsafe {
            *state_kv.get_unchecked_mut((u64::from(row) * u64::from(width) + dimension) as usize) =
                kv[gid as usize];
            *state_score
                .get_unchecked_mut((u64::from(row) * u64::from(width) + dimension) as usize) =
                sc[gid as usize] + ape;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_compressor_set_rows_kernel(
        width: u32,
        ratio: u32,
        pos0: u32,
        src0: u32,
        dst0: u32,
        rows: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
    ) {
        let gid = u64::from(thread::blockIdx_x()) * u64::from(thread::blockDim_x())
            + u64::from(thread::threadIdx_x());
        let count = u64::from(rows) * u64::from(width);
        if gid >= count {
            return;
        }
        let row = (gid / u64::from(width)) as u32;
        let dimension = gid % u64::from(width);
        let source = src0 + row;
        let destination = dst0 + row;
        let phase = pos0.wrapping_add(source) % ratio;
        let input = u64::from(source) * u64::from(width) + dimension;
        let output = u64::from(destination) * u64::from(width) + dimension;
        let ape_index = (u64::from(phase) * u64::from(width) + dimension) as usize;
        let ape = if ape_type == 1 {
            ape_f16[ape_index] as f32
        } else {
            ape_f32[ape_index]
        };
        unsafe {
            *state_kv.get_unchecked_mut(output as usize) = kv[input as usize];
            *state_score.get_unchecked_mut(output as usize) = sc[input as usize] + ape;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_compressor_prefill_ratio4_replay_pool_kernel(
        head_dim: u32,
        pos0: u32,
        n_comp: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        state_kv: &[f32],
        state_score: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut comp: DisjointSlice<f32>,
    ) {
        let dimension = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let compressed = thread::blockIdx_y();
        if dimension >= head_dim || compressed >= n_comp {
            return;
        }
        let width = 2 * head_dim;
        let mut max_score = f32::NEG_INFINITY;
        let mut row = 0_u32;
        if compressed == 0 {
            while row < 4 {
                let score = state_score[(row * width + dimension) as usize];
                if score > max_score {
                    max_score = score;
                }
                row += 1;
            }
        } else {
            let base = (compressed - 1) * 4;
            while row < 4 {
                let token = base + row;
                let phase = pos0.wrapping_add(token) % 4;
                let ape_index = (phase * width + dimension) as usize;
                let ape = if ape_type == 1 {
                    ape_f16[ape_index] as f32
                } else {
                    ape_f32[ape_index]
                };
                let score = sc[(token * width + dimension) as usize] + ape;
                if score > max_score {
                    max_score = score;
                }
                row += 1;
            }
        }
        let base = compressed * 4;
        row = 0;
        while row < 4 {
            let token = base + row;
            let phase = pos0.wrapping_add(token) % 4;
            let ape_index = (phase * width + head_dim + dimension) as usize;
            let ape = if ape_type == 1 {
                ape_f16[ape_index] as f32
            } else {
                ape_f32[ape_index]
            };
            let score = sc[(token * width + head_dim + dimension) as usize] + ape;
            if score > max_score {
                max_score = score;
            }
            row += 1;
        }

        let mut denominator = 0.0_f32;
        let mut accumulator = 0.0_f32;
        row = 0;
        if compressed == 0 {
            while row < 4 {
                let index = (row * width + dimension) as usize;
                let weight = (state_score[index] - max_score).exp();
                denominator += weight;
                accumulator += state_kv[index] * weight;
                row += 1;
            }
        } else {
            let base = (compressed - 1) * 4;
            while row < 4 {
                let token = base + row;
                let phase = pos0.wrapping_add(token) % 4;
                let ape_index = (phase * width + dimension) as usize;
                let ape = if ape_type == 1 {
                    ape_f16[ape_index] as f32
                } else {
                    ape_f32[ape_index]
                };
                let index = (token * width + dimension) as usize;
                let weight = (sc[index] + ape - max_score).exp();
                denominator += weight;
                accumulator += kv[index] * weight;
                row += 1;
            }
        }
        let base = compressed * 4;
        row = 0;
        while row < 4 {
            let token = base + row;
            let phase = pos0.wrapping_add(token) % 4;
            let ape_index = (phase * width + head_dim + dimension) as usize;
            let ape = if ape_type == 1 {
                ape_f16[ape_index] as f32
            } else {
                ape_f32[ape_index]
            };
            let index = (token * width + head_dim + dimension) as usize;
            let weight = (sc[index] + ape - max_score).exp();
            denominator += weight;
            accumulator += kv[index] * weight;
            row += 1;
        }
        unsafe {
            *comp.get_unchecked_mut((compressed * head_dim + dimension) as usize) =
                if denominator != 0.0 {
                    accumulator / denominator
                } else {
                    0.0
                };
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_compressor_prefill_pool_kernel(
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_comp: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut comp: DisjointSlice<f32>,
    ) {
        let dimension = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let compressed = thread::blockIdx_y();
        if dimension >= head_dim || compressed >= n_comp {
            return;
        }
        let width = if ratio == 4 { 2 * head_dim } else { head_dim };
        let mut max_score = f32::NEG_INFINITY;
        if ratio == 4 {
            if compressed > 0 {
                let base = (compressed - 1) * ratio;
                let mut row = 0_u32;
                while row < 4 {
                    let token = base + row;
                    let phase = pos0.wrapping_add(token) % ratio;
                    let index = (token * width + dimension) as usize;
                    let ape_index = (phase * width + dimension) as usize;
                    let ape = if ape_type == 1 {
                        ape_f16[ape_index] as f32
                    } else {
                        ape_f32[ape_index]
                    };
                    let score = sc[index] + ape;
                    if score > max_score {
                        max_score = score;
                    }
                    row += 1;
                }
            }
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < 4 {
                let token = base + row;
                let phase = pos0.wrapping_add(token) % ratio;
                let index = (token * width + head_dim + dimension) as usize;
                let ape_index = (phase * width + head_dim + dimension) as usize;
                let ape = if ape_type == 1 {
                    ape_f16[ape_index] as f32
                } else {
                    ape_f32[ape_index]
                };
                let score = sc[index] + ape;
                if score > max_score {
                    max_score = score;
                }
                row += 1;
            }
        } else {
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < ratio {
                let token = base + row;
                let phase = pos0.wrapping_add(token) % ratio;
                let index = (token * width + dimension) as usize;
                let ape_index = (phase * width + dimension) as usize;
                let ape = if ape_type == 1 {
                    ape_f16[ape_index] as f32
                } else {
                    ape_f32[ape_index]
                };
                let score = sc[index] + ape;
                if score > max_score {
                    max_score = score;
                }
                row += 1;
            }
        }

        let mut denominator = 0.0_f32;
        let mut accumulator = 0.0_f32;
        if ratio == 4 {
            if compressed > 0 {
                let base = (compressed - 1) * ratio;
                let mut row = 0_u32;
                while row < 4 {
                    let token = base + row;
                    let phase = pos0.wrapping_add(token) % ratio;
                    let index = (token * width + dimension) as usize;
                    let ape_index = (phase * width + dimension) as usize;
                    let ape = if ape_type == 1 {
                        ape_f16[ape_index] as f32
                    } else {
                        ape_f32[ape_index]
                    };
                    let weight = (sc[index] + ape - max_score).exp();
                    denominator += weight;
                    accumulator += kv[index] * weight;
                    row += 1;
                }
            }
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < 4 {
                let token = base + row;
                let phase = pos0.wrapping_add(token) % ratio;
                let index = (token * width + head_dim + dimension) as usize;
                let ape_index = (phase * width + head_dim + dimension) as usize;
                let ape = if ape_type == 1 {
                    ape_f16[ape_index] as f32
                } else {
                    ape_f32[ape_index]
                };
                let weight = (sc[index] + ape - max_score).exp();
                denominator += weight;
                accumulator += kv[index] * weight;
                row += 1;
            }
        } else {
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < ratio {
                let token = base + row;
                let phase = pos0.wrapping_add(token) % ratio;
                let index = (token * width + dimension) as usize;
                let ape_index = (phase * width + dimension) as usize;
                let ape = if ape_type == 1 {
                    ape_f16[ape_index] as f32
                } else {
                    ape_f32[ape_index]
                };
                let weight = (sc[index] + ape - max_score).exp();
                denominator += weight;
                accumulator += kv[index] * weight;
                row += 1;
            }
        }
        unsafe {
            *comp.get_unchecked_mut((compressed * head_dim + dimension) as usize) =
                if denominator != 0.0 {
                    accumulator / denominator
                } else {
                    0.0
                };
        }
    }

    #[kernel]
    pub fn abi_compressor_update_pool_kernel(
        head_dim: u32,
        ratio: u32,
        state_kv: &[f32],
        state_score: &[f32],
        mut row: DisjointSlice<f32>,
    ) {
        let dimension = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if dimension >= head_dim {
            return;
        }
        let width = if ratio == 4 { 2 * head_dim } else { head_dim };
        let mut max_score = f32::NEG_INFINITY;
        let mut candidate = 0_u32;
        if ratio == 4 {
            while candidate < 4 {
                let prior = (candidate * width + dimension) as usize;
                let active = ((ratio + candidate) * width + head_dim + dimension) as usize;
                let prior_score = state_score[prior];
                let active_score = state_score[active];
                if prior_score > max_score {
                    max_score = prior_score;
                }
                if active_score > max_score {
                    max_score = active_score;
                }
                candidate += 1;
            }
        } else {
            while candidate < ratio {
                let score = state_score[(candidate * width + dimension) as usize];
                if score > max_score {
                    max_score = score;
                }
                candidate += 1;
            }
        }

        let mut denominator = 0.0_f32;
        let mut accumulator = 0.0_f32;
        candidate = 0;
        if ratio == 4 {
            while candidate < 4 {
                let prior = (candidate * width + dimension) as usize;
                let active = ((ratio + candidate) * width + head_dim + dimension) as usize;
                let prior_weight = (state_score[prior] - max_score).exp();
                let active_weight = (state_score[active] - max_score).exp();
                denominator += prior_weight + active_weight;
                accumulator += state_kv[prior] * prior_weight + state_kv[active] * active_weight;
                candidate += 1;
            }
        } else {
            while candidate < ratio {
                let index = (candidate * width + dimension) as usize;
                let weight = (state_score[index] - max_score).exp();
                denominator += weight;
                accumulator += state_kv[index] * weight;
                candidate += 1;
            }
        }
        unsafe {
            *row.get_unchecked_mut(dimension as usize) = if denominator != 0.0 {
                accumulator / denominator
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn abi_compressor_shift_ratio4_kernel(
        width: u32,
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
    ) {
        let index = u64::from(thread::blockIdx_x()) * u64::from(thread::blockDim_x())
            + u64::from(thread::threadIdx_x());
        let half = 4_u64 * u64::from(width);
        if index >= half {
            return;
        }
        let source = (half + index) as usize;
        let destination = index as usize;
        let kv = unsafe { *state_kv.as_mut_ptr().add(source) };
        let score = unsafe { *state_score.as_mut_ptr().add(source) };
        unsafe {
            *state_kv.get_unchecked_mut(destination) = kv;
            *state_score.get_unchecked_mut(destination) = score;
            *state_kv.get_unchecked_mut(source) = kv;
            *state_score.get_unchecked_mut(source) = score;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_router_select_kernel(
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: u32,
        hash_mode: u32,
        use_token_buffer: u32,
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
    ) {
        let token_index = thread::blockIdx_x() as usize;
        if token_index >= n_tokens as usize || thread::threadIdx_x() != 0 {
            return;
        }
        let prob_base = token_index * ABI_ROUTER_N_EXPERT;
        let selected_base = token_index * ABI_ROUTER_TOP_K;
        let mut expert = 0_usize;
        while expert < ABI_ROUTER_N_EXPERT {
            unsafe {
                *probs.get_unchecked_mut(prob_base + expert) =
                    abi_router_prob(logits[prob_base + expert]);
            }
            expert += 1;
        }

        let mut chosen = [-1_i32; ABI_ROUTER_TOP_K];
        if hash_mode != 0 {
            let mut token = if use_token_buffer != 0 {
                tokens[token_index]
            } else {
                token_scalar
            };
            if token < 0 || token as u32 >= hash_rows {
                token = 0;
            }
            let hash_base = token as usize * ABI_ROUTER_TOP_K;
            let mut output = 0_usize;
            while output < ABI_ROUTER_TOP_K {
                chosen[output] = hash[hash_base + output];
                output += 1;
            }
        } else {
            expert = 0;
            while expert < ABI_ROUTER_N_EXPERT {
                let score = abi_router_prob(logits[prob_base + expert])
                    + if has_bias != 0 { bias[expert] } else { 0.0 };
                let mut output = 0_usize;
                while output < ABI_ROUTER_TOP_K {
                    let incumbent = chosen[output];
                    let better = if incumbent < 0 {
                        true
                    } else {
                        score
                            > abi_router_prob(logits[prob_base + incumbent as usize])
                                + if has_bias != 0 {
                                    bias[incumbent as usize]
                                } else {
                                    0.0
                                }
                    };
                    if better {
                        let mut shift = ABI_ROUTER_TOP_K - 1;
                        while shift > output {
                            chosen[shift] = chosen[shift - 1];
                            shift -= 1;
                        }
                        chosen[output] = expert as i32;
                        break;
                    }
                    output += 1;
                }
                expert += 1;
            }
        }

        let mut sum = 0.0_f32;
        let mut output = 0_usize;
        while output < ABI_ROUTER_TOP_K {
            let selected_expert = chosen[output];
            let probability =
                if selected_expert >= 0 && selected_expert < ABI_ROUTER_N_EXPERT as i32 {
                    abi_router_prob(logits[prob_base + selected_expert as usize])
                } else {
                    0.0
                };
            unsafe {
                *selected.get_unchecked_mut(selected_base + output) = selected_expert;
                *weights.get_unchecked_mut(selected_base + output) = probability;
            }
            sum += probability;
            output += 1;
        }
        if sum < 6.103515625e-5_f32 {
            sum = 6.103515625e-5_f32;
        }
        output = 0;
        while output < ABI_ROUTER_TOP_K {
            let probability = unsafe { *weights.as_mut_ptr().add(selected_base + output) };
            unsafe {
                *weights.get_unchecked_mut(selected_base + output) = probability / sum * 1.5;
            }
            output += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_router_select_parallel_kernel(
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: u32,
        hash_mode: u32,
        use_token_buffer: u32,
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
    ) {
        static mut SPROB: SharedArray<f32, ABI_ROUTER_N_EXPERT> = SharedArray::UNINIT;

        let token_index = thread::blockIdx_x() as usize;
        let expert = thread::threadIdx_x() as usize;
        if token_index >= n_tokens as usize || expert >= ABI_ROUTER_N_EXPERT {
            return;
        }
        let prob_base = token_index * ABI_ROUTER_N_EXPERT;
        let selected_base = token_index * ABI_ROUTER_TOP_K;
        let probability = abi_router_prob(logits[prob_base + expert]);
        unsafe {
            SPROB[expert] = probability;
            *probs.get_unchecked_mut(prob_base + expert) = probability;
        }
        thread::sync_threads();
        if expert != 0 {
            return;
        }

        let mut chosen = [-1_i32; ABI_ROUTER_TOP_K];
        if hash_mode != 0 {
            let mut token = if use_token_buffer != 0 {
                tokens[token_index]
            } else {
                token_scalar
            };
            if token < 0 || token as u32 >= hash_rows {
                token = 0;
            }
            let hash_base = token as usize * ABI_ROUTER_TOP_K;
            let mut output = 0_usize;
            while output < ABI_ROUTER_TOP_K {
                chosen[output] = hash[hash_base + output];
                output += 1;
            }
        } else {
            let mut candidate = 0_usize;
            while candidate < ABI_ROUTER_N_EXPERT {
                let score =
                    unsafe { SPROB[candidate] } + if has_bias != 0 { bias[candidate] } else { 0.0 };
                let mut output = 0_usize;
                while output < ABI_ROUTER_TOP_K {
                    let incumbent = chosen[output];
                    let better = if incumbent < 0 {
                        true
                    } else {
                        score
                            > unsafe { SPROB[incumbent as usize] }
                                + if has_bias != 0 {
                                    bias[incumbent as usize]
                                } else {
                                    0.0
                                }
                    };
                    if better {
                        let mut shift = ABI_ROUTER_TOP_K - 1;
                        while shift > output {
                            chosen[shift] = chosen[shift - 1];
                            shift -= 1;
                        }
                        chosen[output] = candidate as i32;
                        break;
                    }
                    output += 1;
                }
                candidate += 1;
            }
        }

        let mut sum = 0.0_f32;
        let mut output = 0_usize;
        while output < ABI_ROUTER_TOP_K {
            let selected_expert = chosen[output];
            let probability =
                if selected_expert >= 0 && selected_expert < ABI_ROUTER_N_EXPERT as i32 {
                    unsafe { SPROB[selected_expert as usize] }
                } else {
                    0.0
                };
            unsafe {
                *selected.get_unchecked_mut(selected_base + output) = selected_expert;
                *weights.get_unchecked_mut(selected_base + output) = probability;
            }
            sum += probability;
            output += 1;
        }
        if sum < 6.103515625e-5_f32 {
            sum = 6.103515625e-5_f32;
        }
        output = 0;
        while output < ABI_ROUTER_TOP_K {
            let probability = unsafe { *weights.as_mut_ptr().add(selected_base + output) };
            unsafe {
                *weights.get_unchecked_mut(selected_base + output) = probability / sum * 1.5;
            }
            output += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_router_select_warp_topk_kernel(
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: u32,
        hash_mode: u32,
        use_token_buffer: u32,
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
    ) {
        static mut SPROB: SharedArray<f32, { 4 * ABI_ROUTER_N_EXPERT }> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x();
        let row_in_block = thread::threadIdx_y();
        let token_index = thread::blockIdx_x() * ABI_ROUTER_ROWS_PER_WARP_BLOCK + row_in_block;
        if token_index >= n_tokens || lane >= 32 {
            return;
        }
        let prob_base = token_index as usize * ABI_ROUTER_N_EXPERT;
        let shared_base = row_in_block as usize * ABI_ROUTER_N_EXPERT;
        let selected_base = token_index as usize * ABI_ROUTER_TOP_K;
        let mut local_prob = [0.0_f32; 8];
        let mut local_score = [0.0_f32; 8];
        let mut slot = 0_usize;
        while slot < 8 {
            let expert = lane as usize + slot * 32;
            let probability = abi_router_prob(logits[prob_base + expert]);
            local_prob[slot] = probability;
            local_score[slot] = probability + if has_bias != 0 { bias[expert] } else { 0.0 };
            unsafe {
                SPROB[shared_base + expert] = probability;
                *probs.get_unchecked_mut(prob_base + expert) = probability;
            }
            slot += 1;
        }
        warp::sync_mask(u32::MAX);

        if hash_mode != 0 {
            if lane == 0 {
                let mut token = if use_token_buffer != 0 {
                    tokens[token_index as usize]
                } else {
                    token_scalar
                };
                if token < 0 || token as u32 >= hash_rows {
                    token = 0;
                }
                let hash_base = token as usize * ABI_ROUTER_TOP_K;
                let mut sum = 0.0_f32;
                let mut output = 0_usize;
                while output < ABI_ROUTER_TOP_K {
                    let selected_expert = hash[hash_base + output];
                    let probability =
                        if selected_expert >= 0 && selected_expert < ABI_ROUTER_N_EXPERT as i32 {
                            unsafe { SPROB[shared_base + selected_expert as usize] }
                        } else {
                            0.0
                        };
                    unsafe {
                        *selected.get_unchecked_mut(selected_base + output) = selected_expert;
                        *weights.get_unchecked_mut(selected_base + output) = probability;
                    }
                    sum += probability;
                    output += 1;
                }
                if sum < 6.103515625e-5_f32 {
                    sum = 6.103515625e-5_f32;
                }
                output = 0;
                while output < ABI_ROUTER_TOP_K {
                    let probability = unsafe { *weights.as_mut_ptr().add(selected_base + output) };
                    unsafe {
                        *weights.get_unchecked_mut(selected_base + output) =
                            probability / sum * 1.5;
                    }
                    output += 1;
                }
            }
            return;
        }

        let mut output_prob = [0.0_f32; ABI_ROUTER_TOP_K];
        let mut output_index = [0_u32; ABI_ROUTER_TOP_K];
        let mut output = 0_usize;
        while output < ABI_ROUTER_TOP_K {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prob = 0.0_f32;
            let mut best_index = u32::MAX;
            slot = 0;
            while slot < 8 {
                let candidate = lane + slot as u32 * 32;
                let score = local_score[slot];
                if abi_router_score_better(score, candidate, best_score, best_index) {
                    best_score = score;
                    best_prob = local_prob[slot];
                    best_index = candidate;
                }
                slot += 1;
            }
            let mut mask = 16_u32;
            while mask > 0 {
                let other_score = warp::shuffle_xor_f32(best_score, mask);
                let other_prob = warp::shuffle_xor_f32(best_prob, mask);
                let other_index = warp::shuffle_xor(best_index, mask);
                if abi_router_score_better(other_score, other_index, best_score, best_index) {
                    best_score = other_score;
                    best_prob = other_prob;
                    best_index = other_index;
                }
                mask >>= 1;
            }
            slot = 0;
            while slot < 8 {
                if lane + slot as u32 * 32 == best_index {
                    local_score[slot] = f32::NEG_INFINITY;
                }
                slot += 1;
            }
            if lane == 0 {
                output_index[output] = best_index;
                output_prob[output] = best_prob;
            }
            output += 1;
        }

        if lane == 0 {
            let mut sum = 0.0_f32;
            output = 0;
            while output < ABI_ROUTER_TOP_K {
                unsafe {
                    *selected.get_unchecked_mut(selected_base + output) =
                        output_index[output] as i32;
                    *weights.get_unchecked_mut(selected_base + output) = output_prob[output];
                }
                sum += output_prob[output];
                output += 1;
            }
            if sum < 6.103515625e-5_f32 {
                sum = 6.103515625e-5_f32;
            }
            output = 0;
            while output < ABI_ROUTER_TOP_K {
                unsafe {
                    *weights.get_unchecked_mut(selected_base + output) =
                        *weights.get_unchecked_mut(selected_base + output) / sum * 1.5;
                }
                output += 1;
            }
        }
    }

    fn abi_router_score_better(
        candidate_score: f32,
        candidate_index: u32,
        best_score: f32,
        best_index: u32,
    ) -> bool {
        candidate_score > best_score
            || (candidate_score == best_score && candidate_index < best_index)
    }

    fn abi_router_prob(logit: f32) -> f32 {
        let softplus = if logit > 20.0 {
            logit
        } else if logit < -20.0 {
            logit.exp()
        } else {
            (1.0 + logit.exp()).ln()
        };
        softplus.sqrt()
    }

    #[kernel]
    pub fn abi_moe_q8_k_quantize_kernel(
        in_dim: u32,
        n_rows: u32,
        x: &[f32],
        mut out: DisjointSlice<u8>,
    ) {
        static mut ABS_PART: SharedArray<f32, ABI_MOE_QK_K> = SharedArray::UNINIT;
        static mut VAL_PART: SharedArray<f32, ABI_MOE_QK_K> = SharedArray::UNINIT;
        static mut Q_PART: SharedArray<i32, ABI_MOE_QK_K> = SharedArray::UNINIT;
        static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut ISCALE: SharedArray<f32, 1> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let row = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        let blocks = in_dim / ABI_MOE_QK_K as u32;
        if row >= n_rows || block >= blocks || lane >= THREADS_PER_BLOCK {
            return;
        }
        let input = (row * in_dim + block * ABI_MOE_QK_K as u32 + lane) as usize;
        let value = x[input];
        let magnitude = if value < 0.0 { -value } else { value };
        unsafe {
            ABS_PART[lane as usize] = magnitude;
            VAL_PART[lane as usize] = value;
        }
        thread::sync_threads();
        let mut stride = THREADS_PER_BLOCK >> 1;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    if ABS_PART[(lane + stride) as usize] > ABS_PART[lane as usize] {
                        ABS_PART[lane as usize] = ABS_PART[(lane + stride) as usize];
                        VAL_PART[lane as usize] = VAL_PART[(lane + stride) as usize];
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        let output = ((row * blocks + block) as u64 * ABI_MOE_Q8_K_BLOCK_BYTES) as usize;
        if unsafe { ABS_PART[0] } == 0.0 {
            if lane == 0 {
                abi_moe_store_u32(&mut out, output, 0);
            }
            unsafe {
                *out.get_unchecked_mut(output + 4 + lane as usize) = 0;
            }
            if lane < 16 {
                abi_moe_store_i16(&mut out, output + 260 + lane as usize * 2, 0);
            }
            return;
        }
        if lane == 0 {
            unsafe {
                ISCALE[0] = -127.0 / VAL_PART[0];
                SCALE[0] = 1.0 / ISCALE[0];
            }
            abi_moe_store_u32(&mut out, output, unsafe { SCALE[0] }.to_bits());
        }
        thread::sync_threads();
        let quantized = abi_moe_clamp_i8(abi_moe_round_ties_even(unsafe { ISCALE[0] } * value));
        unsafe {
            Q_PART[lane as usize] = quantized as i32;
            *out.get_unchecked_mut(output + 4 + lane as usize) = quantized as u8;
        }
        thread::sync_threads();
        if lane < 16 {
            let mut sum = 0_i32;
            let mut index = lane as usize * 16;
            let end = index + 16;
            while index < end {
                sum += unsafe { Q_PART[index] };
                index += 1;
            }
            abi_moe_store_i16(&mut out, output + 260 + lane as usize * 2, sum as i16);
        }
    }

    #[kernel]
    pub fn abi_moe_count_sorted_pairs_kernel(pair_count: u32, selected: &[i32], counts: &[u32]) {
        let pair = thread::index_1d().get();
        if pair >= pair_count as usize {
            return;
        }
        let mut expert = selected[pair];
        if expert < 0 {
            expert = 0;
        }
        let counter = unsafe { &*(counts.as_ptr().add(expert as usize) as *const DeviceAtomicU32) };
        counter.fetch_add(1, AtomicOrdering::Relaxed);
    }

    #[kernel]
    pub fn abi_moe_prefix_sorted_pairs_kernel(
        counts: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut cursors: DisjointSlice<u32>,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0_u32;
        let mut expert = 0_usize;
        while expert < ABI_MOE_SORTED_EXPERTS {
            unsafe {
                *offsets.get_unchecked_mut(expert) = sum;
                *cursors.get_unchecked_mut(expert) = sum;
            }
            sum += counts[expert];
            expert += 1;
        }
        unsafe {
            *offsets.get_unchecked_mut(ABI_MOE_SORTED_EXPERTS) = sum;
        }
    }

    #[kernel]
    pub fn abi_moe_scatter_sorted_pairs_kernel(
        pair_count: u32,
        selected: &[i32],
        cursors: &[u32],
        mut sorted_pairs: DisjointSlice<u32>,
    ) {
        let pair = thread::index_1d().get();
        if pair >= pair_count as usize {
            return;
        }
        let mut expert = selected[pair];
        if expert < 0 {
            expert = 0;
        }
        let cursor = unsafe { &*(cursors.as_ptr().add(expert as usize) as *const DeviceAtomicU32) };
        let position = cursor.fetch_add(1, AtomicOrdering::Relaxed);
        unsafe {
            *sorted_pairs.get_unchecked_mut(position as usize) = pair as u32;
        }
    }

    #[kernel]
    pub fn abi_moe_build_expert_tile_offsets_kernel(
        block_m: u32,
        counts: &[u32],
        mut tile_offsets: DisjointSlice<u32>,
        mut tile_total: DisjointSlice<u32>,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0_u32;
        let mut expert = 0_usize;
        while expert < ABI_MOE_SORTED_EXPERTS {
            unsafe {
                *tile_offsets.get_unchecked_mut(expert) = sum;
            }
            sum += counts[expert].div_ceil(block_m);
            expert += 1;
        }
        unsafe {
            *tile_offsets.get_unchecked_mut(ABI_MOE_SORTED_EXPERTS) = sum;
            *tile_total.get_unchecked_mut(0) = sum;
        }
    }

    #[kernel]
    pub fn abi_moe_build_expert_tiles_kernel(
        block_m: u32,
        counts: &[u32],
        tile_offsets: &[u32],
        mut tile_experts: DisjointSlice<u32>,
        mut tile_starts: DisjointSlice<u32>,
    ) {
        let expert = thread::threadIdx_x() as usize;
        if expert >= ABI_MOE_SORTED_EXPERTS {
            return;
        }
        let tile_count = counts[expert].div_ceil(block_m);
        let offset = tile_offsets[expert];
        let mut tile = 0_u32;
        while tile < tile_count {
            unsafe {
                *tile_experts.get_unchecked_mut((offset + tile) as usize) = expert as u32;
                *tile_starts.get_unchecked_mut((offset + tile) as usize) = tile * block_m;
            }
            tile += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_expert_tile4_row32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        write_aux: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        abi_moe_gate_up_mid_expert_tile_row32(
            4,
            xq_blocks,
            expert_mid_dim,
            n_expert,
            write_aux,
            clamp,
            gate_expert_bytes,
            gate_row_bytes,
            gate_weights,
            up_weights,
            xq,
            sorted_pairs,
            offsets,
            counts,
            tile_total,
            tile_experts,
            tile_starts,
            weights,
            iq2_grid,
            iq2_signs,
            &mut gate_out,
            &mut up_out,
            &mut mid_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_expert_tile8_row32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        write_aux: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        abi_moe_gate_up_mid_expert_tile_row32(
            8,
            xq_blocks,
            expert_mid_dim,
            n_expert,
            write_aux,
            clamp,
            gate_expert_bytes,
            gate_row_bytes,
            gate_weights,
            up_weights,
            xq,
            sorted_pairs,
            offsets,
            counts,
            tile_total,
            tile_experts,
            tile_starts,
            weights,
            iq2_grid,
            iq2_signs,
            &mut gate_out,
            &mut up_out,
            &mut mid_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_expert_tile8_rowspan_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        row_span: u32,
        write_aux: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] {
            return;
        }
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        let mut row_offset = 0_u32;
        while row_offset < row_span {
            let row = thread::blockIdx_x() * row_span + row_lane + row_offset;
            if row < expert_mid_dim {
                let row_base =
                    u64::from(expert) * gate_expert_bytes + u64::from(row) * gate_row_bytes;
                let mut entry = 0_u32;
                while entry < 8 {
                    let local_pair = local_start + entry;
                    if local_pair < counts[expert as usize] {
                        let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                        let token = pair / n_expert;
                        let mut gate = 0.0_f32;
                        let mut up = 0.0_f32;
                        let mut block = lane;
                        while block < xq_blocks {
                            let q8_block = (u64::from(token) * u64::from(xq_blocks)
                                + u64::from(block))
                                as usize;
                            let packed =
                                (row_base + u64::from(block) * ABI_MOE_IQ2_BLOCK_BYTES) as usize;
                            gate += abi_moe_iq2_q8_k_dot(
                                gate_weights,
                                packed,
                                xq,
                                q8_block,
                                iq2_grid,
                                iq2_signs,
                            );
                            up += abi_moe_iq2_q8_k_dot(
                                up_weights, packed, xq, q8_block, iq2_grid, iq2_signs,
                            );
                            block += 8;
                        }
                        gate = abi_moe_quarter_warp_sum(gate);
                        up = abi_moe_quarter_warp_sum(up);
                        if lane == 0 {
                            abi_moe_apply_clamp(&mut gate, &mut up, clamp);
                            let output = (pair * expert_mid_dim + row) as usize;
                            unsafe {
                                if write_aux != 0 {
                                    *gate_out.get_unchecked_mut(output) = gate;
                                    *up_out.get_unchecked_mut(output) = up;
                                }
                                *mid_out.get_unchecked_mut(output) =
                                    (gate / (1.0 + (-gate).exp())) * up * weights[pair as usize];
                            }
                        }
                    }
                    entry += 1;
                }
            }
            row_offset += 32;
        }
    }

    macro_rules! abi_moe_iq2_signed_word {
        ($grid:expr, $signs:expr, $lane:expr) => {{
            let lane = $lane as u32;
            let bits = ($signs >> lane) as u32;
            let sign_bits =
                ((bits & 1) << 7) | ((bits & 2) << 14) | ((bits & 4) << 21) | ((bits & 8) << 28);
            let mask = integer::prmt_b32_ba98(sign_bits);
            let values = ($grid >> (8 * lane)) as u32;
            ((values ^ mask).wrapping_add(mask & 0x0101_0101)) as i32
        }};
    }

    macro_rules! abi_moe_cached_weight_load_u16 {
        ($values:expr, $offset:expr) => {{
            let values = $values.as_ptr();
            let offset = $offset;
            unsafe { *values.add(offset) as u16 | ((*values.add(offset + 1) as u16) << 8) }
        }};
    }

    #[allow(clippy::too_many_arguments, static_mut_refs)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        row_span: u32,
        write_aux: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        static mut SXQ: SharedArray<
            u8,
            { 8 * ABI_MOE_CACHED_GATE_MAX_BLOCKS * ABI_MOE_Q8_K_BLOCK_BYTES as usize },
            4,
        > = SharedArray::UNINIT;
        static mut S_IQ2_GRID: SharedArray<u64, 256> = SharedArray::UNINIT;
        static mut S_IQ2_SIGNS: SharedArray<u8, 128> = SharedArray::UNINIT;

        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] || xq_blocks as usize > ABI_MOE_CACHED_GATE_MAX_BLOCKS {
            return;
        }
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        let mut np = 0_u32;
        while np < 8 && local_start + np < counts[expert as usize] {
            np += 1;
        }
        let block_bytes = ABI_MOE_Q8_K_BLOCK_BYTES as usize;
        let thread_index = thread::threadIdx_x() as usize;
        let staged_blocks = np as usize * xq_blocks as usize;
        let mut staged_byte = thread_index;
        while staged_byte < staged_blocks * block_bytes {
            let staged_block = staged_byte / block_bytes;
            let byte_index = staged_byte - staged_block * block_bytes;
            let entry = staged_block / xq_blocks as usize;
            let block = staged_block - entry * xq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let token = pair / n_expert;
            let input_block = token as usize * xq_blocks as usize + block;
            unsafe {
                SXQ[staged_byte] = xq[input_block * block_bytes + byte_index];
            }
            staged_byte += THREADS_PER_BLOCK as usize;
        }
        if thread_index < 256 {
            unsafe {
                S_IQ2_GRID[thread_index] = iq2_grid[thread_index];
            }
        }
        if thread_index < 128 {
            unsafe {
                S_IQ2_SIGNS[thread_index] = iq2_signs[thread_index];
            }
        }
        thread::sync_threads();
        let mut row_offset = 0_u32;
        while row_offset < row_span {
            let row = thread::blockIdx_x() * row_span + row_lane + row_offset;
            if row < expert_mid_dim {
                let row_base =
                    u64::from(expert) * gate_expert_bytes + u64::from(row) * gate_row_bytes;
                let mut gate0 = 0.0_f32;
                let mut gate1 = 0.0_f32;
                let mut gate2 = 0.0_f32;
                let mut gate3 = 0.0_f32;
                let mut gate4 = 0.0_f32;
                let mut gate5 = 0.0_f32;
                let mut gate6 = 0.0_f32;
                let mut gate7 = 0.0_f32;
                let mut up0 = 0.0_f32;
                let mut up1 = 0.0_f32;
                let mut up2 = 0.0_f32;
                let mut up3 = 0.0_f32;
                let mut up4 = 0.0_f32;
                let mut up5 = 0.0_f32;
                let mut up6 = 0.0_f32;
                let mut up7 = 0.0_f32;
                let mut block = lane;
                while block < xq_blocks {
                    let packed = (row_base + u64::from(block) * ABI_MOE_IQ2_BLOCK_BYTES) as usize;
                    {
                        let weight_scale =
                            f16::from_bits(abi_moe_cached_weight_load_u16!(gate_weights, packed))
                                as f32;
                        let mut block_sum0 = 0_i32;
                        let mut block_sum1 = 0_i32;
                        let mut block_sum2 = 0_i32;
                        let mut block_sum3 = 0_i32;
                        let mut block_sum4 = 0_i32;
                        let mut block_sum5 = 0_i32;
                        let mut block_sum6 = 0_i32;
                        let mut block_sum7 = 0_i32;
                        let mut ib32 = 0_usize;
                        while ib32 < ABI_MOE_QK_K / 32 {
                            let q2 = packed + 2 + ib32 * 8;
                            let aux_g = abi_moe_cached_weight_load_u16!(gate_weights, q2) as u32
                                | ((abi_moe_cached_weight_load_u16!(gate_weights, q2 + 2) as u32)
                                    << 16);
                            let aux_s = abi_moe_cached_weight_load_u16!(gate_weights, q2 + 4)
                                as u32
                                | ((abi_moe_cached_weight_load_u16!(gate_weights, q2 + 6) as u32)
                                    << 16);
                            let multiplier = (2 * (aux_s >> 28) + 1) as i32;
                            let grid0 = unsafe { S_IQ2_GRID[(aux_g & 0xff) as usize] };
                            let signs0 = unsafe { S_IQ2_SIGNS[(aux_s & 127) as usize] };
                            let grid1 = unsafe { S_IQ2_GRID[((aux_g >> 8) & 0xff) as usize] };
                            let signs1 = unsafe { S_IQ2_SIGNS[((aux_s >> 7) & 127) as usize] };
                            let grid2 = unsafe { S_IQ2_GRID[((aux_g >> 16) & 0xff) as usize] };
                            let signs2 = unsafe { S_IQ2_SIGNS[((aux_s >> 14) & 127) as usize] };
                            let grid3 = unsafe { S_IQ2_GRID[((aux_g >> 24) & 0xff) as usize] };
                            let signs3 = unsafe { S_IQ2_SIGNS[((aux_s >> 21) & 127) as usize] };
                            let weight_word0 = abi_moe_iq2_signed_word!(grid0, signs0, 0);
                            let weight_word1 = abi_moe_iq2_signed_word!(grid0, signs0, 4);
                            let weight_word2 = abi_moe_iq2_signed_word!(grid1, signs1, 0);
                            let weight_word3 = abi_moe_iq2_signed_word!(grid1, signs1, 4);
                            let weight_word4 = abi_moe_iq2_signed_word!(grid2, signs2, 0);
                            let weight_word5 = abi_moe_iq2_signed_word!(grid2, signs2, 4);
                            let weight_word6 = abi_moe_iq2_signed_word!(grid3, signs3, 0);
                            let weight_word7 = abi_moe_iq2_signed_word!(grid3, signs3, 4);
                            let q8_index = ib32 * 32;
                            macro_rules! accumulate_entry {
                                ($entry:literal, $sum:ident) => {{
                                    if np > $entry {
                                        let q8_block =
                                            $entry as usize * xq_blocks as usize + block as usize;
                                        let mut subtotal = 0_i32;
                                        subtotal = integer::dp4a_i8(
                                            weight_word0,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word1,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 4,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word2,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 8,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word3,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 12,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word4,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 16,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word5,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 20,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word6,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 24,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word7,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 28,
                                            ),
                                            subtotal,
                                        );
                                        $sum += subtotal * multiplier;
                                    }
                                }};
                            }
                            accumulate_entry!(0, block_sum0);
                            accumulate_entry!(1, block_sum1);
                            accumulate_entry!(2, block_sum2);
                            accumulate_entry!(3, block_sum3);
                            accumulate_entry!(4, block_sum4);
                            accumulate_entry!(5, block_sum5);
                            accumulate_entry!(6, block_sum6);
                            accumulate_entry!(7, block_sum7);
                            ib32 += 1;
                        }
                        macro_rules! accumulate_scaled_entry {
                            ($entry:literal, $sum:ident, $accum:ident) => {{
                                if np > $entry {
                                    let q8_block =
                                        $entry as usize * xq_blocks as usize + block as usize;
                                    $accum += 0.125
                                        * weight_scale
                                        * abi_moe_cached_q8_scale(
                                            unsafe { SXQ.as_ptr() },
                                            q8_block,
                                        )
                                        * $sum as f32;
                                }
                            }};
                        }
                        accumulate_scaled_entry!(0, block_sum0, gate0);
                        accumulate_scaled_entry!(1, block_sum1, gate1);
                        accumulate_scaled_entry!(2, block_sum2, gate2);
                        accumulate_scaled_entry!(3, block_sum3, gate3);
                        accumulate_scaled_entry!(4, block_sum4, gate4);
                        accumulate_scaled_entry!(5, block_sum5, gate5);
                        accumulate_scaled_entry!(6, block_sum6, gate6);
                        accumulate_scaled_entry!(7, block_sum7, gate7);
                    }
                    {
                        let weight_scale =
                            f16::from_bits(abi_moe_cached_weight_load_u16!(up_weights, packed))
                                as f32;
                        let mut block_sum0 = 0_i32;
                        let mut block_sum1 = 0_i32;
                        let mut block_sum2 = 0_i32;
                        let mut block_sum3 = 0_i32;
                        let mut block_sum4 = 0_i32;
                        let mut block_sum5 = 0_i32;
                        let mut block_sum6 = 0_i32;
                        let mut block_sum7 = 0_i32;
                        let mut ib32 = 0_usize;
                        while ib32 < ABI_MOE_QK_K / 32 {
                            let q2 = packed + 2 + ib32 * 8;
                            let aux_g = abi_moe_cached_weight_load_u16!(up_weights, q2) as u32
                                | ((abi_moe_cached_weight_load_u16!(up_weights, q2 + 2) as u32)
                                    << 16);
                            let aux_s = abi_moe_cached_weight_load_u16!(up_weights, q2 + 4) as u32
                                | ((abi_moe_cached_weight_load_u16!(up_weights, q2 + 6) as u32)
                                    << 16);
                            let multiplier = (2 * (aux_s >> 28) + 1) as i32;
                            let grid0 = unsafe { S_IQ2_GRID[(aux_g & 0xff) as usize] };
                            let signs0 = unsafe { S_IQ2_SIGNS[(aux_s & 127) as usize] };
                            let grid1 = unsafe { S_IQ2_GRID[((aux_g >> 8) & 0xff) as usize] };
                            let signs1 = unsafe { S_IQ2_SIGNS[((aux_s >> 7) & 127) as usize] };
                            let grid2 = unsafe { S_IQ2_GRID[((aux_g >> 16) & 0xff) as usize] };
                            let signs2 = unsafe { S_IQ2_SIGNS[((aux_s >> 14) & 127) as usize] };
                            let grid3 = unsafe { S_IQ2_GRID[((aux_g >> 24) & 0xff) as usize] };
                            let signs3 = unsafe { S_IQ2_SIGNS[((aux_s >> 21) & 127) as usize] };
                            let weight_word0 = abi_moe_iq2_signed_word!(grid0, signs0, 0);
                            let weight_word1 = abi_moe_iq2_signed_word!(grid0, signs0, 4);
                            let weight_word2 = abi_moe_iq2_signed_word!(grid1, signs1, 0);
                            let weight_word3 = abi_moe_iq2_signed_word!(grid1, signs1, 4);
                            let weight_word4 = abi_moe_iq2_signed_word!(grid2, signs2, 0);
                            let weight_word5 = abi_moe_iq2_signed_word!(grid2, signs2, 4);
                            let weight_word6 = abi_moe_iq2_signed_word!(grid3, signs3, 0);
                            let weight_word7 = abi_moe_iq2_signed_word!(grid3, signs3, 4);
                            let q8_index = ib32 * 32;
                            macro_rules! accumulate_entry {
                                ($entry:literal, $sum:ident) => {{
                                    if np > $entry {
                                        let q8_block =
                                            $entry as usize * xq_blocks as usize + block as usize;
                                        let mut subtotal = 0_i32;
                                        subtotal = integer::dp4a_i8(
                                            weight_word0,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word1,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 4,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word2,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 8,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word3,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 12,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word4,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 16,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word5,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 20,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word6,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 24,
                                            ),
                                            subtotal,
                                        );
                                        subtotal = integer::dp4a_i8(
                                            weight_word7,
                                            abi_moe_cached_q8_word(
                                                unsafe { SXQ.as_ptr() },
                                                q8_block,
                                                q8_index + 28,
                                            ),
                                            subtotal,
                                        );
                                        $sum += subtotal * multiplier;
                                    }
                                }};
                            }
                            accumulate_entry!(0, block_sum0);
                            accumulate_entry!(1, block_sum1);
                            accumulate_entry!(2, block_sum2);
                            accumulate_entry!(3, block_sum3);
                            accumulate_entry!(4, block_sum4);
                            accumulate_entry!(5, block_sum5);
                            accumulate_entry!(6, block_sum6);
                            accumulate_entry!(7, block_sum7);
                            ib32 += 1;
                        }
                        macro_rules! accumulate_scaled_entry {
                            ($entry:literal, $sum:ident, $accum:ident) => {{
                                if np > $entry {
                                    let q8_block =
                                        $entry as usize * xq_blocks as usize + block as usize;
                                    $accum += 0.125
                                        * weight_scale
                                        * abi_moe_cached_q8_scale(
                                            unsafe { SXQ.as_ptr() },
                                            q8_block,
                                        )
                                        * $sum as f32;
                                }
                            }};
                        }
                        accumulate_scaled_entry!(0, block_sum0, up0);
                        accumulate_scaled_entry!(1, block_sum1, up1);
                        accumulate_scaled_entry!(2, block_sum2, up2);
                        accumulate_scaled_entry!(3, block_sum3, up3);
                        accumulate_scaled_entry!(4, block_sum4, up4);
                        accumulate_scaled_entry!(5, block_sum5, up5);
                        accumulate_scaled_entry!(6, block_sum6, up6);
                        accumulate_scaled_entry!(7, block_sum7, up7);
                    }
                    block += 8;
                }
                macro_rules! emit_entry {
                    ($entry:literal, $gate:ident, $up:ident) => {{
                        if np > $entry {
                            let pair = sorted_pairs
                                [(offsets[expert as usize] + local_start + $entry) as usize];
                            let mut gate_value = abi_moe_quarter_warp_sum($gate);
                            let mut up_value = abi_moe_quarter_warp_sum($up);
                            if lane == 0 {
                                abi_moe_apply_clamp(&mut gate_value, &mut up_value, clamp);
                                let output = (pair * expert_mid_dim + row) as usize;
                                unsafe {
                                    if write_aux != 0 {
                                        *gate_out.get_unchecked_mut(output) = gate_value;
                                        *up_out.get_unchecked_mut(output) = up_value;
                                    }
                                    *mid_out.get_unchecked_mut(output) = (gate_value
                                        / (1.0 + (-gate_value).exp()))
                                        * up_value
                                        * weights[pair as usize];
                                }
                            }
                        }
                    }};
                }
                emit_entry!(0, gate0, up0);
                emit_entry!(1, gate1, up1);
                emit_entry!(2, gate2, up2);
                emit_entry!(3, gate3, up3);
                emit_entry!(4, gate4, up4);
                emit_entry!(5, gate5, up5);
                emit_entry!(6, gate6, up6);
                emit_entry!(7, gate7, up7);
            }
            row_offset += 32;
        }
    }

    #[kernel]
    pub fn abi_moe_atomic_output_zero_kernel(mut output: DisjointSlice<f32>, count: u64) {
        let index = thread::index_1d().get();
        if index < count as usize {
            unsafe {
                *output.get_unchecked_mut(index) = 0.0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_expert_tile4_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        atomic_out: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        mut down_out: DisjointSlice<f32>,
    ) {
        abi_moe_down_expert_tile_row32(
            4,
            false,
            midq_blocks,
            out_dim,
            n_expert,
            atomic_out,
            down_expert_bytes,
            down_row_bytes,
            down_weights,
            midq,
            sorted_pairs,
            offsets,
            counts,
            tile_total,
            tile_experts,
            tile_starts,
            &mut down_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_expert_tile8_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        atomic_out: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        mut down_out: DisjointSlice<f32>,
    ) {
        abi_moe_down_expert_tile_row32(
            8,
            false,
            midq_blocks,
            out_dim,
            n_expert,
            atomic_out,
            down_expert_bytes,
            down_row_bytes,
            down_weights,
            midq,
            sorted_pairs,
            offsets,
            counts,
            tile_total,
            tile_experts,
            tile_starts,
            &mut down_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_expert_tile16_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        atomic_out: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        mut down_out: DisjointSlice<f32>,
    ) {
        abi_moe_down_expert_tile_row32(
            16,
            true,
            midq_blocks,
            out_dim,
            n_expert,
            atomic_out,
            down_expert_bytes,
            down_row_bytes,
            down_weights,
            midq,
            sorted_pairs,
            offsets,
            counts,
            tile_total,
            tile_experts,
            tile_starts,
            &mut down_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_expert_tile16_rowspan_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        row_span: u32,
        atomic_out: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        mut down_out: DisjointSlice<f32>,
    ) {
        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] {
            return;
        }
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        if local_start & 8 != 0 {
            return;
        }
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let mut row_offset = 0_u32;
        while row_offset < row_span {
            let row = thread::blockIdx_x() * row_span + row_lane + row_offset;
            if row < out_dim {
                let row_base =
                    u64::from(expert) * down_expert_bytes + u64::from(row) * down_row_bytes;
                let mut entry = 0_u32;
                while entry < 16 {
                    let local_pair = local_start + entry;
                    if local_pair < counts[expert as usize] {
                        let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                        let mut accumulator = 0.0_f32;
                        let mut block = lane;
                        while block < midq_blocks {
                            accumulator += abi_moe_q2_q8_k_dot(
                                down_weights,
                                (row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES) as usize,
                                midq,
                                (u64::from(pair) * u64::from(midq_blocks) + u64::from(block))
                                    as usize,
                            );
                            block += 8;
                        }
                        accumulator = abi_moe_quarter_warp_sum(accumulator);
                        if lane == 0 {
                            if atomic_out != 0 {
                                let token = pair / n_expert;
                                let output = (token * out_dim + row) as usize;
                                let cell = unsafe {
                                    &*(down_out.as_mut_ptr().add(output) as *const DeviceAtomicF32)
                                };
                                cell.fetch_add(accumulator, AtomicOrdering::Relaxed);
                            } else {
                                unsafe {
                                    *down_out.get_unchecked_mut((pair * out_dim + row) as usize) =
                                        accumulator;
                                }
                            }
                        }
                    }
                    entry += 1;
                }
            }
            row_offset += 32;
        }
    }

    macro_rules! abi_moe_down_load_u32 {
        ($values:expr, $offset:expr) => {{
            let offset = $offset;
            $values[offset] as u32
                | (($values[offset + 1] as u32) << 8)
                | (($values[offset + 2] as u32) << 16)
                | (($values[offset + 3] as u32) << 24)
        }};
    }

    macro_rules! abi_moe_down_accumulate_q2_group {
        ($weights:expr, $q:expr, $shift:expr, $q8_base:expr, $scale_index:ident, $np:expr, $midq_blocks:expr, $block:expr, $q8:expr, $entry_base:expr, $sum0:ident, $sum1:ident, $sum2:ident, $sum3:ident, $sum4:ident, $sum5:ident, $sum6:ident, $sum7:ident) => {{
            let first_scale = ($weights[$scale_index] & 0x0f) as i32;
            $scale_index += 1;
            let second_scale = ($weights[$scale_index] & 0x0f) as i32;
            $scale_index += 1;
            let first_word0 =
                ((abi_moe_down_load_u32!($weights, $q) >> $shift) & 0x0303_0303) as i32;
            let first_word1 =
                ((abi_moe_down_load_u32!($weights, $q + 4) >> $shift) & 0x0303_0303) as i32;
            let first_word2 =
                ((abi_moe_down_load_u32!($weights, $q + 8) >> $shift) & 0x0303_0303) as i32;
            let first_word3 =
                ((abi_moe_down_load_u32!($weights, $q + 12) >> $shift) & 0x0303_0303) as i32;
            let second_word0 =
                ((abi_moe_down_load_u32!($weights, $q + 16) >> $shift) & 0x0303_0303) as i32;
            let second_word1 =
                ((abi_moe_down_load_u32!($weights, $q + 20) >> $shift) & 0x0303_0303) as i32;
            let second_word2 =
                ((abi_moe_down_load_u32!($weights, $q + 24) >> $shift) & 0x0303_0303) as i32;
            let second_word3 =
                ((abi_moe_down_load_u32!($weights, $q + 28) >> $shift) & 0x0303_0303) as i32;
            macro_rules! accumulate_entry {
                ($entry:literal, $sum:ident) => {{
                    if $np > $entry_base + $entry {
                        let q8_block = ($entry_base + $entry) as usize * $midq_blocks as usize
                            + $block as usize;
                        let mut first = 0_i32;
                        first = integer::dp4a_i8(
                            first_word0,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base),
                            first,
                        );
                        first = integer::dp4a_i8(
                            first_word1,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 4),
                            first,
                        );
                        first = integer::dp4a_i8(
                            first_word2,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 8),
                            first,
                        );
                        first = integer::dp4a_i8(
                            first_word3,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 12),
                            first,
                        );
                        let mut second = 0_i32;
                        second = integer::dp4a_i8(
                            second_word0,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 16),
                            second,
                        );
                        second = integer::dp4a_i8(
                            second_word1,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 20),
                            second,
                        );
                        second = integer::dp4a_i8(
                            second_word2,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 24),
                            second,
                        );
                        second = integer::dp4a_i8(
                            second_word3,
                            abi_moe_cached_q8_word($q8, q8_block, $q8_base + 28),
                            second,
                        );
                        $sum += first_scale * first + second_scale * second;
                    }
                }};
            }
            accumulate_entry!(0, $sum0);
            accumulate_entry!(1, $sum1);
            accumulate_entry!(2, $sum2);
            accumulate_entry!(3, $sum3);
            accumulate_entry!(4, $sum4);
            accumulate_entry!(5, $sum5);
            accumulate_entry!(6, $sum6);
            accumulate_entry!(7, $sum7);
        }};
    }

    #[allow(clippy::too_many_arguments, static_mut_refs)]
    #[kernel]
    pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        row_span: u32,
        atomic_out: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        mut down_out: DisjointSlice<f32>,
    ) {
        static mut SMIDQ: SharedArray<
            u8,
            { 16 * ABI_MOE_CACHED_DOWN_MAX_BLOCKS * ABI_MOE_Q8_K_BLOCK_BYTES as usize },
            4,
        > = SharedArray::UNINIT;

        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] || midq_blocks as usize > ABI_MOE_CACHED_DOWN_MAX_BLOCKS {
            return;
        }
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        if local_start & 8 != 0 {
            return;
        }
        let mut np = 0_u32;
        while np < 16 && local_start + np < counts[expert as usize] {
            np += 1;
        }
        let block_bytes = ABI_MOE_Q8_K_BLOCK_BYTES as usize;
        let thread_index = thread::threadIdx_x() as usize;
        let staged_blocks = np as usize * midq_blocks as usize;
        let mut staged_byte = thread_index;
        while staged_byte < staged_blocks * block_bytes {
            let staged_block = staged_byte / block_bytes;
            let byte_index = staged_byte - staged_block * block_bytes;
            let entry = staged_block / midq_blocks as usize;
            let block = staged_block - entry * midq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let input_block = pair as usize * midq_blocks as usize + block;
            unsafe {
                SMIDQ[staged_byte] = midq[input_block * block_bytes + byte_index];
            }
            staged_byte += THREADS_PER_BLOCK as usize;
        }
        thread::sync_threads();
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let mut row_offset = 0_u32;
        while row_offset < row_span {
            let row = thread::blockIdx_x() * row_span + row_lane + row_offset;
            if row < out_dim {
                let row_base =
                    u64::from(expert) * down_expert_bytes + u64::from(row) * down_row_bytes;
                let mut entry_base = 0_u32;
                while entry_base < np {
                    let mut accumulator0 = 0.0_f32;
                    let mut accumulator1 = 0.0_f32;
                    let mut accumulator2 = 0.0_f32;
                    let mut accumulator3 = 0.0_f32;
                    let mut accumulator4 = 0.0_f32;
                    let mut accumulator5 = 0.0_f32;
                    let mut accumulator6 = 0.0_f32;
                    let mut accumulator7 = 0.0_f32;
                    let mut block = lane;
                    while block < midq_blocks {
                        let packed =
                            (row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES) as usize;
                        let weight_scale =
                            f16::from_bits(abi_moe_load_u16(down_weights, packed + 80)) as f32;
                        let weight_min =
                            f16::from_bits(abi_moe_load_u16(down_weights, packed + 82)) as f32;
                        let mut min_sum0 = 0_i32;
                        let mut min_sum1 = 0_i32;
                        let mut min_sum2 = 0_i32;
                        let mut min_sum3 = 0_i32;
                        let mut min_sum4 = 0_i32;
                        let mut min_sum5 = 0_i32;
                        let mut min_sum6 = 0_i32;
                        let mut min_sum7 = 0_i32;
                        let mut scale = 0_usize;
                        macro_rules! accumulate_min_entry {
                            ($entry:literal, $sum:ident, $minimum:expr) => {{
                                if np > entry_base + $entry {
                                    let q8_block = (entry_base + $entry) as usize
                                        * midq_blocks as usize
                                        + block as usize;
                                    $sum += abi_moe_cached_q8_bsum(
                                        unsafe { SMIDQ.as_ptr() },
                                        q8_block,
                                        scale,
                                    ) * $minimum;
                                }
                            }};
                        }
                        while scale < 16 {
                            let minimum = (down_weights[packed + scale] >> 4) as i32;
                            accumulate_min_entry!(0, min_sum0, minimum);
                            accumulate_min_entry!(1, min_sum1, minimum);
                            accumulate_min_entry!(2, min_sum2, minimum);
                            accumulate_min_entry!(3, min_sum3, minimum);
                            accumulate_min_entry!(4, min_sum4, minimum);
                            accumulate_min_entry!(5, min_sum5, minimum);
                            accumulate_min_entry!(6, min_sum6, minimum);
                            accumulate_min_entry!(7, min_sum7, minimum);
                            scale += 1;
                        }
                        let mut quant_sum0 = 0_i32;
                        let mut quant_sum1 = 0_i32;
                        let mut quant_sum2 = 0_i32;
                        let mut quant_sum3 = 0_i32;
                        let mut quant_sum4 = 0_i32;
                        let mut quant_sum5 = 0_i32;
                        let mut quant_sum6 = 0_i32;
                        let mut quant_sum7 = 0_i32;
                        let mut scale_index = packed;
                        let mut chunk = 0_usize;
                        macro_rules! accumulate_q2_group {
                            ($q:expr, $shift:expr, $q8_base:expr) => {{
                                abi_moe_down_accumulate_q2_group!(
                                    down_weights,
                                    $q,
                                    $shift,
                                    $q8_base,
                                    scale_index,
                                    np,
                                    midq_blocks,
                                    block,
                                    unsafe { SMIDQ.as_ptr() },
                                    entry_base,
                                    quant_sum0,
                                    quant_sum1,
                                    quant_sum2,
                                    quant_sum3,
                                    quant_sum4,
                                    quant_sum5,
                                    quant_sum6,
                                    quant_sum7
                                );
                            }};
                        }
                        while chunk < 2 {
                            let q = packed + 16 + chunk * 32;
                            let q8_base = chunk * 128;
                            accumulate_q2_group!(q, 0, q8_base);
                            accumulate_q2_group!(q, 2, q8_base + 32);
                            accumulate_q2_group!(q, 4, q8_base + 64);
                            accumulate_q2_group!(q, 6, q8_base + 96);
                            chunk += 1;
                        }
                        macro_rules! accumulate_scaled_entry {
                            ($entry:literal, $quant_sum:ident, $min_sum:ident, $accumulator:ident) => {{
                                if np > entry_base + $entry {
                                    let q8_block = (entry_base + $entry) as usize
                                        * midq_blocks as usize
                                        + block as usize;
                                    $accumulator += abi_moe_cached_q8_scale(
                                        unsafe { SMIDQ.as_ptr() },
                                        q8_block,
                                    ) * (weight_scale * $quant_sum as f32
                                        - weight_min * $min_sum as f32);
                                }
                            }};
                        }
                        accumulate_scaled_entry!(0, quant_sum0, min_sum0, accumulator0);
                        accumulate_scaled_entry!(1, quant_sum1, min_sum1, accumulator1);
                        accumulate_scaled_entry!(2, quant_sum2, min_sum2, accumulator2);
                        accumulate_scaled_entry!(3, quant_sum3, min_sum3, accumulator3);
                        accumulate_scaled_entry!(4, quant_sum4, min_sum4, accumulator4);
                        accumulate_scaled_entry!(5, quant_sum5, min_sum5, accumulator5);
                        accumulate_scaled_entry!(6, quant_sum6, min_sum6, accumulator6);
                        accumulate_scaled_entry!(7, quant_sum7, min_sum7, accumulator7);
                        block += 8;
                    }
                    macro_rules! emit_entry {
                        ($entry:literal, $accumulator:ident) => {{
                            if np > entry_base + $entry {
                                let pair = sorted_pairs[(offsets[expert as usize]
                                    + local_start
                                    + entry_base
                                    + $entry) as usize];
                                let accumulator = abi_moe_quarter_warp_sum($accumulator);
                                if lane == 0 {
                                    if atomic_out != 0 {
                                        let token = pair / n_expert;
                                        let output = (token * out_dim + row) as usize;
                                        let cell = unsafe {
                                            &*(down_out.as_mut_ptr().add(output)
                                                as *const DeviceAtomicF32)
                                        };
                                        cell.fetch_add(accumulator, AtomicOrdering::Relaxed);
                                    } else {
                                        unsafe {
                                            *down_out.get_unchecked_mut(
                                                (pair * out_dim + row) as usize,
                                            ) = accumulator;
                                        }
                                    }
                                }
                            }
                        }};
                    }
                    emit_entry!(0, accumulator0);
                    emit_entry!(1, accumulator1);
                    emit_entry!(2, accumulator2);
                    emit_entry!(3, accumulator3);
                    emit_entry!(4, accumulator4);
                    emit_entry!(5, accumulator5);
                    emit_entry!(6, accumulator6);
                    emit_entry!(7, accumulator7);
                    entry_base += 8;
                }
            }
            row_offset += 32;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_f32_kernel(
        n_tokens: u32,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        x: &[f32],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_GATE: SharedArray<f32, ABI_MOE_QK_K> = SharedArray::UNINIT;
        static mut PARTIAL_UP: SharedArray<f32, ABI_MOE_QK_K> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let pair = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        if row >= expert_mid_dim || pair >= n_tokens * n_expert || lane >= THREADS_PER_BLOCK {
            return;
        }
        let token = pair / n_expert;
        let mut expert = selected[pair as usize];
        if expert < 0 {
            expert = 0;
        }
        let blocks = expert_in_dim / ABI_MOE_QK_K as u32;
        let row_base = expert as u64 * gate_expert_bytes + u64::from(row) * gate_row_bytes;
        let x_base = token as usize * expert_in_dim as usize;
        let mut gate = 0.0_f32;
        let mut up = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let packed = row_base + u64::from(block) * ABI_MOE_IQ2_BLOCK_BYTES;
            gate += abi_moe_iq2_f32_dot(
                gate_weights,
                packed as usize,
                x,
                x_base + block as usize * ABI_MOE_QK_K,
                iq2_grid,
                iq2_signs,
            );
            up += abi_moe_iq2_f32_dot(
                up_weights,
                packed as usize,
                x,
                x_base + block as usize * ABI_MOE_QK_K,
                iq2_grid,
                iq2_signs,
            );
            block += THREADS_PER_BLOCK;
        }
        unsafe {
            PARTIAL_GATE[lane as usize] = gate;
            PARTIAL_UP[lane as usize] = up;
        }
        thread::sync_threads();
        let mut stride = THREADS_PER_BLOCK >> 1;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL_GATE[lane as usize] += PARTIAL_GATE[(lane + stride) as usize];
                    PARTIAL_UP[lane as usize] += PARTIAL_UP[(lane + stride) as usize];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if lane == 0 {
            gate = unsafe { PARTIAL_GATE[0] };
            up = unsafe { PARTIAL_UP[0] };
            abi_moe_apply_clamp(&mut gate, &mut up, clamp);
            let offset = (pair * expert_mid_dim + row) as usize;
            unsafe {
                *gate_out.get_unchecked_mut(offset) = gate;
                *up_out.get_unchecked_mut(offset) = up;
                *mid_out.get_unchecked_mut(offset) =
                    (gate / (1.0 + (-gate).exp())) * up * weights[pair as usize];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_f32_kernel(
        n_tokens: u32,
        expert_mid_dim: u32,
        out_dim: u32,
        n_expert: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        mid: &[f32],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, ABI_MOE_QK_K> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let pair = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        if row >= out_dim || pair >= n_tokens * n_expert || lane >= THREADS_PER_BLOCK {
            return;
        }
        let mut expert = selected[pair as usize];
        if expert < 0 {
            expert = 0;
        }
        let blocks = expert_mid_dim / ABI_MOE_QK_K as u32;
        let row_base = expert as u64 * down_expert_bytes + u64::from(row) * down_row_bytes;
        let mid_base = pair as usize * expert_mid_dim as usize;
        let mut accumulator = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let packed = row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES;
            accumulator += abi_moe_q2_f32_dot(
                down_weights,
                packed as usize,
                mid,
                mid_base + block as usize * ABI_MOE_QK_K,
            );
            block += THREADS_PER_BLOCK;
        }
        unsafe {
            PARTIAL[lane as usize] = accumulator;
        }
        thread::sync_threads();
        let mut stride = THREADS_PER_BLOCK >> 1;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL[lane as usize] += PARTIAL[(lane + stride) as usize];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if lane == 0 {
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = PARTIAL[0];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_qwarp32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        abi_moe_gate_up_quantized(
            true,
            false,
            xq_blocks,
            expert_mid_dim,
            n_expert,
            clamp,
            gate_expert_bytes,
            gate_row_bytes,
            gate_weights,
            up_weights,
            xq,
            selected,
            weights,
            iq2_grid,
            iq2_signs,
            &mut gate_out,
            &mut up_out,
            &mut mid_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_sorted_qwarp32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        let pair = sorted_pairs[thread::blockIdx_y() as usize];
        if row >= expert_mid_dim {
            return;
        }
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let row_base = expert as u64 * gate_expert_bytes + u64::from(row) * gate_row_bytes;
        let mut gate = 0.0_f32;
        let mut up = 0.0_f32;
        let mut block = lane;
        while block < xq_blocks {
            let q8_block = (u64::from(token) * u64::from(xq_blocks) + u64::from(block)) as usize;
            let packed = (row_base + u64::from(block) * ABI_MOE_IQ2_BLOCK_BYTES) as usize;
            gate += abi_moe_iq2_q8_k_dot(gate_weights, packed, xq, q8_block, iq2_grid, iq2_signs);
            up += abi_moe_iq2_q8_k_dot(up_weights, packed, xq, q8_block, iq2_grid, iq2_signs);
            block += 8;
        }
        gate = abi_moe_quarter_warp_sum(gate);
        up = abi_moe_quarter_warp_sum(up);
        if lane == 0 {
            abi_moe_apply_clamp(&mut gate, &mut up, clamp);
            let offset = (pair * expert_mid_dim + row) as usize;
            unsafe {
                *gate_out.get_unchecked_mut(offset) = gate;
                *up_out.get_unchecked_mut(offset) = up;
                *mid_out.get_unchecked_mut(offset) =
                    (gate / (1.0 + (-gate).exp())) * up * weights[pair as usize];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let pair_lane = (thread::threadIdx_x() >> 3) & 1;
        let row = thread::blockIdx_x() * 16 + (thread::threadIdx_x() >> 4);
        let sorted_index = thread::blockIdx_y() * 2 + pair_lane;
        if row >= expert_mid_dim || sorted_index >= pair_count {
            return;
        }
        let pair = sorted_pairs[sorted_index as usize];
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let row_base = expert as u64 * gate_expert_bytes + u64::from(row) * gate_row_bytes;
        let mut gate = 0.0_f32;
        let mut up = 0.0_f32;
        let mut block = lane;
        while block < xq_blocks {
            let q8_block = (u64::from(token) * u64::from(xq_blocks) + u64::from(block)) as usize;
            let packed = (row_base + u64::from(block) * ABI_MOE_IQ2_BLOCK_BYTES) as usize;
            gate += abi_moe_iq2_q8_k_dot(gate_weights, packed, xq, q8_block, iq2_grid, iq2_signs);
            up += abi_moe_iq2_q8_k_dot(up_weights, packed, xq, q8_block, iq2_grid, iq2_signs);
            block += 8;
        }
        gate = abi_moe_quarter_warp_sum(gate);
        up = abi_moe_quarter_warp_sum(up);
        if lane == 0 {
            abi_moe_apply_clamp(&mut gate, &mut up, clamp);
            let offset = (pair * expert_mid_dim + row) as usize;
            unsafe {
                *gate_out.get_unchecked_mut(offset) = gate;
                *up_out.get_unchecked_mut(offset) = up;
                *mid_out.get_unchecked_mut(offset) =
                    (gate / (1.0 + (-gate).exp())) * up * weights[pair as usize];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_decode_lut_qwarp32_kernel(
        write_aux: u32,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        abi_moe_gate_up_quantized(
            write_aux != 0,
            false,
            xq_blocks,
            expert_mid_dim,
            n_expert,
            clamp,
            gate_expert_bytes,
            gate_row_bytes,
            gate_weights,
            up_weights,
            xq,
            selected,
            weights,
            iq2_grid,
            iq2_signs,
            &mut gate_out,
            &mut up_out,
            &mut mid_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_gate_up_mid_decode_q4_k_qwarp32_kernel(
        write_aux: u32,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        abi_moe_gate_up_quantized(
            write_aux != 0,
            true,
            xq_blocks,
            expert_mid_dim,
            n_expert,
            clamp,
            gate_expert_bytes,
            gate_row_bytes,
            gate_weights,
            up_weights,
            xq,
            selected,
            weights,
            iq2_grid,
            iq2_signs,
            &mut gate_out,
            &mut up_out,
            &mut mid_out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        let pair = thread::blockIdx_y();
        if row >= out_dim || pair >= n_expert {
            return;
        }
        let mut expert = selected[pair as usize];
        if expert < 0 {
            expert = 0;
        }
        let row_base = expert as u64 * down_expert_bytes + u64::from(row) * down_row_bytes;
        let mut accumulator = 0.0_f32;
        let mut block = lane;
        while block < midq_blocks {
            accumulator += abi_moe_q2_q8_k_dot(
                down_weights,
                (row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES) as usize,
                midq,
                (u64::from(pair) * u64::from(midq_blocks) + u64::from(block)) as usize,
            );
            block += 8;
        }
        accumulator = abi_moe_quarter_warp_sum(accumulator);
        if lane == 0 {
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_sorted_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        let pair = sorted_pairs[thread::blockIdx_y() as usize];
        if row >= out_dim {
            return;
        }
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let row_base = expert as u64 * down_expert_bytes + u64::from(row) * down_row_bytes;
        let mut accumulator = 0.0_f32;
        let mut block = lane;
        while block < midq_blocks {
            accumulator += abi_moe_q2_q8_k_dot(
                down_weights,
                (row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES) as usize,
                midq,
                (u64::from(pair) * u64::from(midq_blocks) + u64::from(block)) as usize,
            );
            block += 8;
        }
        accumulator = abi_moe_quarter_warp_sum(accumulator);
        if lane == 0 {
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_sorted_p2_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        pair_count: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let pair_lane = (thread::threadIdx_x() >> 3) & 1;
        let row = thread::blockIdx_x() * 16 + (thread::threadIdx_x() >> 4);
        let sorted_index = thread::blockIdx_y() * 2 + pair_lane;
        if row >= out_dim || sorted_index >= pair_count {
            return;
        }
        let pair = sorted_pairs[sorted_index as usize];
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let row_base = expert as u64 * down_expert_bytes + u64::from(row) * down_row_bytes;
        let mut accumulator = 0.0_f32;
        let mut block = lane;
        while block < midq_blocks {
            accumulator += abi_moe_q2_q8_k_dot(
                down_weights,
                (row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES) as usize,
                midq,
                (u64::from(pair) * u64::from(midq_blocks) + u64::from(block)) as usize,
            );
            block += 8;
        }
        accumulator = abi_moe_quarter_warp_sum(accumulator);
        if lane == 0 {
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_sum6_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        selected: &[i32],
        mut out: DisjointSlice<f32>,
    ) {
        abi_moe_down_sum6(
            false,
            midq_blocks,
            out_dim,
            down_expert_bytes,
            down_row_bytes,
            down_weights,
            midq,
            selected,
            &mut out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_moe_down_q4_k_sum6_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        selected: &[i32],
        mut out: DisjointSlice<f32>,
    ) {
        abi_moe_down_sum6(
            true,
            midq_blocks,
            out_dim,
            down_expert_bytes,
            down_row_bytes,
            down_weights,
            midq,
            selected,
            &mut out,
        );
    }

    #[kernel]
    pub fn abi_moe_sum_kernel(
        n_tokens: u32,
        out_dim: u32,
        n_expert: u32,
        down: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index >= n_tokens * out_dim {
            return;
        }
        let token = index / out_dim;
        let row = index - token * out_dim;
        let mut accumulator = 0.0_f32;
        let mut slot = 0_u32;
        while slot < n_expert {
            accumulator += down[((token * n_expert + slot) * out_dim + row) as usize];
            slot += 1;
        }
        unsafe {
            *out.get_unchecked_mut(index as usize) = accumulator;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn abi_moe_gate_up_quantized(
        write_aux_only: bool,
        q4_k: bool,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        selected: &[i32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        gate_out: &mut DisjointSlice<f32>,
        up_out: &mut DisjointSlice<f32>,
        mid_out: &mut DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let pair = thread::blockIdx_y();
        if pair >= n_expert {
            return;
        }
        let mut expert = selected[pair as usize];
        if expert < 0 {
            expert = 0;
        }
        let mut rr = 0_u32;
        while rr < 4 {
            let row = thread::blockIdx_x() * 128 + row_lane + rr * 32;
            if row < expert_mid_dim {
                let row_base = expert as u64 * gate_expert_bytes + u64::from(row) * gate_row_bytes;
                let block_bytes = if q4_k {
                    ABI_MOE_Q4_BLOCK_BYTES
                } else {
                    ABI_MOE_IQ2_BLOCK_BYTES
                };
                let mut gate = 0.0_f32;
                let mut up = 0.0_f32;
                let mut block = lane;
                while block < xq_blocks {
                    let q8_block = block as usize;
                    let packed = (row_base + u64::from(block) * block_bytes) as usize;
                    if q4_k {
                        gate += abi_moe_q4_k_q8_k_dot(gate_weights, packed, xq, q8_block);
                        up += abi_moe_q4_k_q8_k_dot(up_weights, packed, xq, q8_block);
                    } else {
                        gate += abi_moe_iq2_q8_k_dot(
                            gate_weights,
                            packed,
                            xq,
                            q8_block,
                            iq2_grid,
                            iq2_signs,
                        );
                        up += abi_moe_iq2_q8_k_dot(
                            up_weights, packed, xq, q8_block, iq2_grid, iq2_signs,
                        );
                    }
                    block += 8;
                }
                gate = abi_moe_quarter_warp_sum(gate);
                up = abi_moe_quarter_warp_sum(up);
                if lane == 0 {
                    abi_moe_apply_clamp(&mut gate, &mut up, clamp);
                    let offset = (pair * expert_mid_dim + row) as usize;
                    unsafe {
                        if write_aux_only {
                            *gate_out.get_unchecked_mut(offset) = gate;
                            *up_out.get_unchecked_mut(offset) = up;
                        }
                        *mid_out.get_unchecked_mut(offset) =
                            (gate / (1.0 + (-gate).exp())) * up * weights[pair as usize];
                    }
                }
            }
            rr += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn abi_moe_down_sum6(
        q4_k: bool,
        midq_blocks: u32,
        out_dim: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        selected: &[i32],
        out: &mut DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        if row >= out_dim {
            return;
        }
        let block_bytes = if q4_k {
            ABI_MOE_Q4_BLOCK_BYTES
        } else {
            ABI_MOE_Q2_BLOCK_BYTES
        };
        let mut total = 0.0_f32;
        let mut slot = 0_u32;
        while slot < 6 {
            let mut expert = selected[slot as usize];
            if expert < 0 {
                expert = 0;
            }
            let row_base = expert as u64 * down_expert_bytes + u64::from(row) * down_row_bytes;
            let mut accumulator = 0.0_f32;
            let mut block = lane;
            while block < midq_blocks {
                let packed = (row_base + u64::from(block) * block_bytes) as usize;
                let q8_block =
                    (u64::from(slot) * u64::from(midq_blocks) + u64::from(block)) as usize;
                accumulator += if q4_k {
                    abi_moe_q4_k_q8_k_dot(down_weights, packed, midq, q8_block)
                } else {
                    abi_moe_q2_q8_k_dot(down_weights, packed, midq, q8_block)
                };
                block += 8;
            }
            accumulator = abi_moe_quarter_warp_sum(accumulator);
            if lane == 0 {
                total += accumulator;
            }
            slot += 1;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row as usize) = total;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn abi_moe_gate_up_mid_expert_tile_row32(
        tile_width: u32,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        write_aux: u32,
        clamp: f32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        weights: &[f32],
        iq2_grid: &[u64],
        iq2_signs: &[u8],
        gate_out: &mut DisjointSlice<f32>,
        up_out: &mut DisjointSlice<f32>,
        mid_out: &mut DisjointSlice<f32>,
    ) {
        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] {
            return;
        }
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        if row >= expert_mid_dim {
            return;
        }
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        let row_base = u64::from(expert) * gate_expert_bytes + u64::from(row) * gate_row_bytes;
        let mut entry = 0_u32;
        while entry < tile_width {
            let local_pair = local_start + entry;
            if local_pair < counts[expert as usize] {
                let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                let token = pair / n_expert;
                let mut gate = 0.0_f32;
                let mut up = 0.0_f32;
                let mut block = lane;
                while block < xq_blocks {
                    let q8_block =
                        (u64::from(token) * u64::from(xq_blocks) + u64::from(block)) as usize;
                    let packed = (row_base + u64::from(block) * ABI_MOE_IQ2_BLOCK_BYTES) as usize;
                    gate += abi_moe_iq2_q8_k_dot(
                        gate_weights,
                        packed,
                        xq,
                        q8_block,
                        iq2_grid,
                        iq2_signs,
                    );
                    up +=
                        abi_moe_iq2_q8_k_dot(up_weights, packed, xq, q8_block, iq2_grid, iq2_signs);
                    block += 8;
                }
                gate = abi_moe_quarter_warp_sum(gate);
                up = abi_moe_quarter_warp_sum(up);
                if lane == 0 {
                    abi_moe_apply_clamp(&mut gate, &mut up, clamp);
                    let output = (pair * expert_mid_dim + row) as usize;
                    unsafe {
                        if write_aux != 0 {
                            *gate_out.get_unchecked_mut(output) = gate;
                            *up_out.get_unchecked_mut(output) = up;
                        }
                        *mid_out.get_unchecked_mut(output) =
                            (gate / (1.0 + (-gate).exp())) * up * weights[pair as usize];
                    }
                }
            }
            entry += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn abi_moe_down_expert_tile_row32(
        tile_width: u32,
        require_even_tile_pair: bool,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        atomic_out: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        down_weights: &[u8],
        midq: &[u8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        down_out: &mut DisjointSlice<f32>,
    ) {
        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] {
            return;
        }
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        if row >= out_dim {
            return;
        }
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        if require_even_tile_pair && local_start & 8 != 0 {
            return;
        }
        let row_base = u64::from(expert) * down_expert_bytes + u64::from(row) * down_row_bytes;
        let mut entry = 0_u32;
        while entry < tile_width {
            let local_pair = local_start + entry;
            if local_pair < counts[expert as usize] {
                let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                let mut accumulator = 0.0_f32;
                let mut block = lane;
                while block < midq_blocks {
                    accumulator += abi_moe_q2_q8_k_dot(
                        down_weights,
                        (row_base + u64::from(block) * ABI_MOE_Q2_BLOCK_BYTES) as usize,
                        midq,
                        (u64::from(pair) * u64::from(midq_blocks) + u64::from(block)) as usize,
                    );
                    block += 8;
                }
                accumulator = abi_moe_quarter_warp_sum(accumulator);
                if lane == 0 {
                    if atomic_out != 0 {
                        let token = pair / n_expert;
                        let output = (token * out_dim + row) as usize;
                        let cell = unsafe {
                            &*(down_out.as_mut_ptr().add(output) as *const DeviceAtomicF32)
                        };
                        cell.fetch_add(accumulator, AtomicOrdering::Relaxed);
                    } else {
                        unsafe {
                            *down_out.get_unchecked_mut((pair * out_dim + row) as usize) =
                                accumulator;
                        }
                    }
                }
            }
            entry += 1;
        }
    }

    fn abi_moe_apply_clamp(gate: &mut f32, up: &mut f32, clamp: f32) {
        if clamp > 1.0e-6 {
            if *gate > clamp {
                *gate = clamp;
            }
            if *up > clamp {
                *up = clamp;
            }
            if *up < -clamp {
                *up = -clamp;
            }
        }
    }

    fn abi_moe_iq2_f32_dot(
        packed: &[u8],
        base: usize,
        x: &[f32],
        x_base: usize,
        iq2_grid: &[u64],
        iq2_signs: &[u8],
    ) -> f32 {
        let scale = f16::from_bits(abi_moe_load_u16(packed, base)) as f32;
        let mut accumulator = 0.0_f32;
        let mut ib32 = 0_usize;
        while ib32 < ABI_MOE_QK_K / 32 {
            let q2 = base + 2 + ib32 * 8;
            let aux_g = abi_moe_load_u16(packed, q2) as u32
                | ((abi_moe_load_u16(packed, q2 + 2) as u32) << 16);
            let aux_s = abi_moe_load_u16(packed, q2 + 4) as u32
                | ((abi_moe_load_u16(packed, q2 + 6) as u32) << 16);
            let dl = scale * (0.5 + (aux_s >> 28) as f32) * 0.25;
            let mut group = 0_u32;
            while group < 4 {
                let grid = iq2_grid[((aux_g >> (8 * group)) & 0xff) as usize];
                let signs = iq2_signs[((aux_s >> (7 * group)) & 127) as usize];
                let mut lane = 0_u32;
                while lane < 8 {
                    let mut value = ((grid >> (8 * lane)) & 0xff) as f32;
                    if signs & (1_u8 << lane) != 0 {
                        value = -value;
                    }
                    accumulator +=
                        dl * value * x[x_base + ib32 * 32 + group as usize * 8 + lane as usize];
                    lane += 1;
                }
                group += 1;
            }
            ib32 += 1;
        }
        accumulator
    }

    fn abi_moe_q2_f32_dot(packed: &[u8], base: usize, x: &[f32], x_base: usize) -> f32 {
        let d = f16::from_bits(abi_moe_load_u16(packed, base + 80)) as f32;
        let dmin = f16::from_bits(abi_moe_load_u16(packed, base + 82)) as f32;
        let mut accumulator = 0.0_f32;
        let mut il = 0_usize;
        while il < 16 {
            let chunk = il / 8;
            let pair = il & 1;
            let shift = ((il / 2) & 3) * 2;
            let scale = packed[base + il];
            let dl = d * (scale & 0x0f) as f32;
            let ml = dmin * (scale >> 4) as f32;
            let q = base + 16 + 32 * chunk + 16 * pair;
            let xf = x_base + chunk * 128 + ((il % 8) / 2) * 32 + pair * 16;
            let mut index = 0_usize;
            while index < 16 {
                accumulator +=
                    (dl * ((packed[q + index] >> shift) & 3) as f32 - ml) * x[xf + index];
                index += 1;
            }
            il += 1;
        }
        accumulator
    }

    fn abi_moe_iq2_q8_k_dot(
        packed: &[u8],
        base: usize,
        q8: &[u8],
        q8_block: usize,
        iq2_grid: &[u64],
        iq2_signs: &[u8],
    ) -> f32 {
        let weight_scale = f16::from_bits(abi_moe_load_u16(packed, base)) as f32;
        let mut block_sum = 0_i32;
        let mut ib32 = 0_usize;
        while ib32 < ABI_MOE_QK_K / 32 {
            let q2 = base + 2 + ib32 * 8;
            let aux_g = abi_moe_load_u16(packed, q2) as u32
                | ((abi_moe_load_u16(packed, q2 + 2) as u32) << 16);
            let aux_s = abi_moe_load_u16(packed, q2 + 4) as u32
                | ((abi_moe_load_u16(packed, q2 + 6) as u32) << 16);
            let multiplier = (2 * (aux_s >> 28) + 1) as i32;
            let mut subtotal = 0_i32;
            let mut group = 0_u32;
            while group < 4 {
                let grid = iq2_grid[((aux_g >> (8 * group)) & 0xff) as usize];
                let signs = iq2_signs[((aux_s >> (7 * group)) & 127) as usize];
                let mut lane = 0_u32;
                while lane < 8 {
                    let mut value = ((grid >> (8 * lane)) & 0xff) as i32;
                    if signs & (1_u8 << lane) != 0 {
                        value = -value;
                    }
                    subtotal += value
                        * abi_moe_q8_value(
                            q8,
                            q8_block,
                            ib32 * 32 + group as usize * 8 + lane as usize,
                        );
                    lane += 1;
                }
                group += 1;
            }
            block_sum += subtotal * multiplier;
            ib32 += 1;
        }
        0.125 * weight_scale * abi_moe_q8_scale(q8, q8_block) * block_sum as f32
    }

    fn abi_moe_q2_q8_k_dot(packed: &[u8], base: usize, q8: &[u8], q8_block: usize) -> f32 {
        let weight_scale = f16::from_bits(abi_moe_load_u16(packed, base + 80)) as f32;
        let weight_min = f16::from_bits(abi_moe_load_u16(packed, base + 82)) as f32;
        let mut min_sum = 0_i32;
        let mut scale = 0_usize;
        while scale < 16 {
            min_sum += abi_moe_q8_bsum(q8, q8_block, scale) * (packed[base + scale] >> 4) as i32;
            scale += 1;
        }
        let mut quant_sum = 0_i32;
        let mut scale_index = 0_usize;
        let mut chunk = 0_usize;
        while chunk < 2 {
            let mut shift = 0_u32;
            let mut group = 0_usize;
            while group < 4 {
                let first_scale = (packed[base + scale_index] & 0x0f) as i32;
                scale_index += 1;
                let second_scale = (packed[base + scale_index] & 0x0f) as i32;
                scale_index += 1;
                let q = base + 16 + chunk * 32;
                let q8_base = chunk * 128 + group * 32;
                let mut lane = 0_usize;
                let mut first = 0_i32;
                let mut second = 0_i32;
                while lane < 16 {
                    first += ((packed[q + lane] >> shift) & 3) as i32
                        * abi_moe_q8_value(q8, q8_block, q8_base + lane);
                    second += ((packed[q + 16 + lane] >> shift) & 3) as i32
                        * abi_moe_q8_value(q8, q8_block, q8_base + 16 + lane);
                    lane += 1;
                }
                quant_sum += first_scale * first + second_scale * second;
                shift += 2;
                group += 1;
            }
            chunk += 1;
        }
        abi_moe_q8_scale(q8, q8_block)
            * (weight_scale * quant_sum as f32 - weight_min * min_sum as f32)
    }

    fn abi_moe_q4_k_q8_k_dot(packed: &[u8], base: usize, q8: &[u8], q8_block: usize) -> f32 {
        let weight_scale = f16::from_bits(abi_moe_load_u16(packed, base)) as f32;
        let weight_min = f16::from_bits(abi_moe_load_u16(packed, base + 2)) as f32;
        let mut quant_sum = 0_i32;
        let mut min_sum = 0_i32;
        let mut group = 0_usize;
        while group < 8 {
            let scale = if group < 4 {
                packed[base + 4 + group] & 63
            } else {
                (packed[base + 4 + group + 4] & 0x0f) | ((packed[base + 4 + group - 4] >> 6) << 4)
            };
            let minimum = if group < 4 {
                packed[base + 4 + group + 4] & 63
            } else {
                (packed[base + 4 + group + 4] >> 4) | ((packed[base + 4 + group] >> 6) << 4)
            };
            min_sum += minimum as i32
                * (abi_moe_q8_bsum(q8, q8_block, 2 * group)
                    + abi_moe_q8_bsum(q8, q8_block, 2 * group + 1));
            let q4 = base + 16 + (group >> 1) * 32;
            let shift = if group & 1 == 0 { 0 } else { 4 };
            let q8_start = group * 32;
            let mut lane = 0_usize;
            let mut subtotal = 0_i32;
            while lane < 32 {
                subtotal += ((packed[q4 + lane] >> shift) & 0x0f) as i32
                    * abi_moe_q8_value(q8, q8_block, q8_start + lane);
                lane += 1;
            }
            quant_sum += scale as i32 * subtotal;
            group += 1;
        }
        abi_moe_q8_scale(q8, q8_block)
            * (weight_scale * quant_sum as f32 - weight_min * min_sum as f32)
    }

    fn abi_moe_q8_scale(q8: &[u8], block: usize) -> f32 {
        f32::from_bits(abi_moe_load_u32(
            q8,
            block * ABI_MOE_Q8_K_BLOCK_BYTES as usize,
        ))
    }

    fn abi_moe_q8_value(q8: &[u8], block: usize, index: usize) -> i32 {
        q8[block * ABI_MOE_Q8_K_BLOCK_BYTES as usize + 4 + index] as i8 as i32
    }

    fn abi_moe_q8_bsum(q8: &[u8], block: usize, index: usize) -> i32 {
        abi_moe_load_u16(
            q8,
            block * ABI_MOE_Q8_K_BLOCK_BYTES as usize + 260 + index * 2,
        ) as i16 as i32
    }

    fn abi_moe_cached_q8_scale(q8: *const u8, block: usize) -> f32 {
        f32::from_bits(abi_moe_cached_load_aligned_u32(
            q8,
            block * ABI_MOE_Q8_K_BLOCK_BYTES as usize,
        ))
    }

    fn abi_moe_cached_q8_word(q8: *const u8, block: usize, index: usize) -> i32 {
        abi_moe_cached_load_aligned_u32(q8, block * ABI_MOE_Q8_K_BLOCK_BYTES as usize + 4 + index)
            as i32
    }

    fn abi_moe_cached_q8_bsum(q8: *const u8, block: usize, index: usize) -> i32 {
        abi_moe_cached_load_u16(
            q8,
            block * ABI_MOE_Q8_K_BLOCK_BYTES as usize + 260 + index * 2,
        ) as i16 as i32
    }

    #[inline(always)]
    fn abi_moe_quarter_warp_sum(mut value: f32) -> f32 {
        value += warp::shuffle_xor_f32(value, 4);
        value += warp::shuffle_xor_f32(value, 2);
        value += warp::shuffle_xor_f32(value, 1);
        value
    }

    fn abi_moe_load_u16(values: &[u8], offset: usize) -> u16 {
        values[offset] as u16 | ((values[offset + 1] as u16) << 8)
    }

    fn abi_moe_load_u32(values: &[u8], offset: usize) -> u32 {
        values[offset] as u32
            | ((values[offset + 1] as u32) << 8)
            | ((values[offset + 2] as u32) << 16)
            | ((values[offset + 3] as u32) << 24)
    }

    fn abi_moe_cached_load_u16(values: *const u8, offset: usize) -> u16 {
        unsafe { *values.add(offset) as u16 | ((*values.add(offset + 1) as u16) << 8) }
    }

    fn abi_moe_cached_load_aligned_u32(values: *const u8, offset: usize) -> u32 {
        unsafe { *values.add(offset).cast::<u32>() }
    }

    fn abi_moe_store_u32(values: &mut DisjointSlice<u8>, offset: usize, value: u32) {
        unsafe {
            *values.get_unchecked_mut(offset) = value as u8;
            *values.get_unchecked_mut(offset + 1) = (value >> 8) as u8;
            *values.get_unchecked_mut(offset + 2) = (value >> 16) as u8;
            *values.get_unchecked_mut(offset + 3) = (value >> 24) as u8;
        }
    }

    fn abi_moe_store_i16(values: &mut DisjointSlice<u8>, offset: usize, value: i16) {
        let value = value as u16;
        unsafe {
            *values.get_unchecked_mut(offset) = value as u8;
            *values.get_unchecked_mut(offset + 1) = (value >> 8) as u8;
        }
    }

    fn abi_moe_round_ties_even(value: f32) -> i32 {
        let lower = value.floor();
        let fraction = value - lower;
        let mut rounded = lower as i32;
        if fraction > 0.5 || (fraction == 0.5 && (rounded & 1) != 0) {
            rounded += 1;
        }
        rounded
    }

    fn abi_moe_clamp_i8(value: i32) -> i8 {
        if value > 127 {
            127
        } else if value < -128 {
            -128
        } else {
            value as i8
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_decode_mixed_kernel(
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        use_comp_mask: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        static mut SCORES: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut RAW_ROWS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut SOFTMAX: SharedArray<f32, 2> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        let tid = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x();
        if token >= n_tokens || head >= n_head {
            return;
        }
        let single_all = n_tokens == 1 && ratio == 0;
        let qpos = pos0.wrapping_add(token);
        let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
        let mut visible_comp = if single_all {
            n_comp
        } else if n_comp != 0 {
            qpos.wrapping_add(1) / ratio
        } else {
            0
        };
        if visible_comp > n_comp {
            visible_comp = n_comp;
        }
        let mut raw_first = 0_u32;
        let mut raw_count = 0_u32;
        if single_all {
            raw_count = if n_raw < 256 { n_raw } else { 256 };
        } else {
            let raw_last_pos = first_raw_pos.wrapping_add(n_raw).wrapping_sub(1);
            if qpos >= first_raw_pos {
                let mut lo = first_raw_pos;
                if window != 0 && qpos.wrapping_add(1) > window {
                    let window_lo = qpos.wrapping_add(1).wrapping_sub(window);
                    if window_lo > lo {
                        lo = window_lo;
                    }
                }
                let hi = if qpos < raw_last_pos {
                    qpos
                } else {
                    raw_last_pos
                };
                if hi >= lo {
                    raw_first = lo.wrapping_sub(first_raw_pos);
                    raw_count = hi.wrapping_sub(lo).wrapping_add(1);
                    if raw_count > 256 {
                        raw_count = 256;
                    }
                }
            }
        }
        let query_base =
            ((token as usize * n_head as usize + head as usize) * head_dim as usize) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let mut raw_row = thread::threadIdx_x();
        while raw_row < raw_count {
            unsafe {
                RAW_ROWS[raw_row as usize] =
                    raw_start.wrapping_add(raw_first).wrapping_add(raw_row) % raw_cap;
            }
            raw_row += block_dim;
        }
        thread::sync_threads();

        let n_score = raw_count + visible_comp;
        let mut local_max = sinks[head as usize];
        if visible_comp == 0 || n_tokens == 1 {
            raw_row = thread::threadIdx_x();
            while raw_row < raw_count {
                let row = unsafe { RAW_ROWS[raw_row as usize] };
                let score = abi_attention_dot(q, query_base, raw_kv, row, head_dim) * scale;
                unsafe {
                    SCORES[raw_row as usize] = score;
                }
                local_max = abi_attention_maximum(local_max, score);
                raw_row += block_dim;
            }
            let mut compressed = thread::threadIdx_x();
            while compressed < visible_comp {
                let add = if use_comp_mask != 0 {
                    comp_mask[token as usize * n_comp as usize + compressed as usize]
                } else {
                    0.0
                };
                let mut score = f32::NEG_INFINITY;
                if add > -1.0e20 {
                    score = abi_attention_dot(q, query_base, comp_kv, compressed, head_dim) * scale
                        + add;
                }
                unsafe {
                    SCORES[(raw_count + compressed) as usize] = score;
                }
                local_max = abi_attention_maximum(local_max, score);
                compressed += block_dim;
            }
        } else {
            let qlane = thread::threadIdx_x() & 7;
            let qgroup = thread::threadIdx_x() >> 3;
            let mut row0 = 0_u32;
            while row0 < n_score {
                let row = row0 + qgroup;
                if row < n_score {
                    let mut add = 0.0_f32;
                    let mut valid_row = true;
                    let kv_row;
                    let kv;
                    if row < raw_count {
                        kv_row = unsafe { RAW_ROWS[row as usize] };
                        kv = raw_kv;
                    } else {
                        let compressed = row - raw_count;
                        add = if use_comp_mask != 0 {
                            comp_mask[token as usize * n_comp as usize + compressed as usize]
                        } else {
                            0.0
                        };
                        valid_row = add > -1.0e20;
                        kv_row = compressed;
                        kv = comp_kv;
                    }
                    let mut score = f32::NEG_INFINITY;
                    if valid_row {
                        let mut dot = 0.0_f32;
                        let mut dimension = qlane;
                        while dimension < head_dim {
                            dot += q[query_base + dimension as usize]
                                * kv[(kv_row * head_dim + dimension) as usize];
                            dimension += 8;
                        }
                        let mask = 0xff_u32 << (thread::threadIdx_x() & 24);
                        let mut offset = 4_u32;
                        while offset > 0 {
                            dot += warp::shuffle_xor_f32_sync(mask, dot, offset);
                            offset >>= 1;
                        }
                        score = dot * scale + add;
                    }
                    if qlane == 0 {
                        unsafe {
                            SCORES[row as usize] = score;
                        }
                    }
                }
                row0 += 32;
            }
            thread::sync_threads();
            let mut score = thread::threadIdx_x();
            while score < n_score {
                local_max = abi_attention_maximum(local_max, unsafe { SCORES[score as usize] });
                score += block_dim;
            }
        }

        unsafe {
            PARTIAL[tid] = local_max;
        }
        thread::sync_threads();
        let mut stride = block_dim >> 1;
        while stride > 0 {
            if thread::threadIdx_x() < stride {
                unsafe {
                    PARTIAL[tid] =
                        abi_attention_maximum(PARTIAL[tid], PARTIAL[tid + stride as usize]);
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                SOFTMAX[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_score = unsafe { SOFTMAX[0] };
        let mut denominator = 0.0_f32;
        let mut score = thread::threadIdx_x();
        while score < n_score {
            let probability = (unsafe { SCORES[score as usize] } - max_score).exp();
            unsafe {
                SCORES[score as usize] = probability;
            }
            denominator += probability;
            score += block_dim;
        }
        unsafe {
            PARTIAL[tid] = denominator;
        }
        thread::sync_threads();
        stride = block_dim >> 1;
        while stride > 0 {
            if thread::threadIdx_x() < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride as usize];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                SOFTMAX[1] = PARTIAL[0] + (sinks[head as usize] - max_score).exp();
            }
        }
        thread::sync_threads();
        denominator = unsafe { SOFTMAX[1] };

        if head_dim == 512 && block_dim == 256 {
            let dimension0 = thread::threadIdx_x();
            let dimension1 = dimension0 + 256;
            let mut accumulator0 = 0.0_f32;
            let mut accumulator1 = 0.0_f32;
            raw_row = 0;
            while raw_row < raw_count {
                let probability = unsafe { SCORES[raw_row as usize] };
                let row = unsafe { RAW_ROWS[raw_row as usize] };
                accumulator0 += raw_kv[(row * head_dim + dimension0) as usize] * probability;
                accumulator1 += raw_kv[(row * head_dim + dimension1) as usize] * probability;
                raw_row += 1;
            }
            let mut compressed = 0_u32;
            while compressed < visible_comp {
                let probability = unsafe { SCORES[(raw_count + compressed) as usize] };
                accumulator0 +=
                    comp_kv[(compressed * head_dim + dimension0) as usize] * probability;
                accumulator1 +=
                    comp_kv[(compressed * head_dim + dimension1) as usize] * probability;
                compressed += 1;
            }
            unsafe {
                *heads.get_unchecked_mut(query_base + dimension0 as usize) =
                    accumulator0 / denominator;
                *heads.get_unchecked_mut(query_base + dimension1 as usize) =
                    accumulator1 / denominator;
            }
        } else {
            let mut dimension = thread::threadIdx_x();
            while dimension < head_dim {
                let mut accumulator = 0.0_f32;
                raw_row = 0;
                while raw_row < raw_count {
                    let probability = unsafe { SCORES[raw_row as usize] };
                    let row = unsafe { RAW_ROWS[raw_row as usize] };
                    accumulator += raw_kv[(row * head_dim + dimension) as usize] * probability;
                    raw_row += 1;
                }
                let mut compressed = 0_u32;
                while compressed < visible_comp {
                    let probability = unsafe { SCORES[(raw_count + compressed) as usize] };
                    accumulator +=
                        comp_kv[(compressed * head_dim + dimension) as usize] * probability;
                    compressed += 1;
                }
                unsafe {
                    *heads.get_unchecked_mut(query_base + dimension as usize) =
                        accumulator / denominator;
                }
                dimension += block_dim;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_decode_mixed_heads8_online_kernel(
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
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut SOFTMAX: SharedArray<f32, 4> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        let tid = thread::threadIdx_x() as usize;
        if token >= n_tokens || head >= n_head || head_dim != 512 {
            return;
        }
        let qpos = pos0.wrapping_add(token);
        let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
        let mut raw_first = 0_u32;
        let mut raw_count = 0_u32;
        if qpos >= first_raw_pos {
            let raw_last_pos = first_raw_pos.wrapping_add(n_raw).wrapping_sub(1);
            let mut lo = first_raw_pos;
            if window != 0 && qpos.wrapping_add(1) > window {
                let window_lo = qpos.wrapping_add(1).wrapping_sub(window);
                if window_lo > lo {
                    lo = window_lo;
                }
            }
            let hi = if qpos < raw_last_pos {
                qpos
            } else {
                raw_last_pos
            };
            if hi >= lo {
                raw_first = lo.wrapping_sub(first_raw_pos);
                raw_count = hi.wrapping_sub(lo).wrapping_add(1);
                if raw_count > 256 {
                    raw_count = 256;
                }
            }
        }
        let mut comp_count = 0_u32;
        if n_comp != 0 {
            if n_tokens == 1 && ratio == 0 {
                comp_count = n_comp;
            } else if ratio != 0 {
                comp_count = qpos.wrapping_add(1) / ratio;
                if comp_count > n_comp {
                    comp_count = n_comp;
                }
            }
        }
        let query_base =
            ((token as usize * n_head as usize + head as usize) * head_dim as usize) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let dimension0 = tid;
        let dimension1 = tid + 256;
        let mut accumulator0 = 0.0_f32;
        let mut accumulator1 = 0.0_f32;
        if tid == 0 {
            unsafe {
                SOFTMAX[0] = f32::NEG_INFINITY;
                SOFTMAX[1] = 0.0;
            }
        }
        thread::sync_threads();
        let mut score_row = 0_u32;
        while score_row < raw_count + comp_count {
            let (kv, row) = if score_row < raw_count {
                (
                    raw_kv,
                    raw_start.wrapping_add(raw_first).wrapping_add(score_row) % raw_cap,
                )
            } else {
                (comp_kv, score_row - raw_count)
            };
            let row_base = (row * head_dim) as usize;
            unsafe {
                PARTIAL[tid] = q[query_base + dimension0] * kv[row_base + dimension0]
                    + q[query_base + dimension1] * kv[row_base + dimension1];
            }
            thread::sync_threads();
            let mut stride = 128_usize;
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
                let score = unsafe { PARTIAL[0] } * scale;
                let old_max = unsafe { SOFTMAX[0] };
                let next_max = abi_attention_maximum(old_max, score);
                unsafe {
                    SOFTMAX[2] = (old_max - next_max).exp();
                    SOFTMAX[3] = (score - next_max).exp();
                    SOFTMAX[1] = SOFTMAX[1] * SOFTMAX[2] + SOFTMAX[3];
                    SOFTMAX[0] = next_max;
                }
            }
            thread::sync_threads();
            let old_scale = unsafe { SOFTMAX[2] };
            let row_scale = unsafe { SOFTMAX[3] };
            accumulator0 = accumulator0 * old_scale + kv[row_base + dimension0] * row_scale;
            accumulator1 = accumulator1 * old_scale + kv[row_base + dimension1] * row_scale;
            thread::sync_threads();
            score_row += 1;
        }
        if tid == 0 {
            let sink = sinks[head as usize];
            let old_max = unsafe { SOFTMAX[0] };
            let next_max = abi_attention_maximum(old_max, sink);
            unsafe {
                SOFTMAX[2] = (old_max - next_max).exp();
                SOFTMAX[1] = SOFTMAX[1] * SOFTMAX[2] + (sink - next_max).exp();
                SOFTMAX[0] = next_max;
            }
        }
        thread::sync_threads();
        accumulator0 *= unsafe { SOFTMAX[2] };
        accumulator1 *= unsafe { SOFTMAX[2] };
        let denominator = unsafe { SOFTMAX[1] };
        unsafe {
            *heads.get_unchecked_mut(query_base + dimension0) = accumulator0 / denominator;
            *heads.get_unchecked_mut(query_base + dimension1) = accumulator1 / denominator;
        }
    }

    #[kernel]
    pub fn abi_attention_prefill_raw_kernel(
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let raw_count = if token + 1 < window {
            token + 1
        } else {
            window
        };
        let raw_start = token + 1 - raw_count;
        abi_attention_prefill_write_head(
            token, head, raw_start, raw_count, 0, 0, n_head, head_dim, sinks, q, raw_kv, raw_kv,
            raw_kv, 0, &mut heads,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_prefill_mixed_kernel(
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        use_mask: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let raw_start = if window != 0 && token + 1 > window {
            token + 1 - window
        } else {
            0
        };
        let raw_count = token + 1 - raw_start;
        let mut visible_comp = (token + 1) / ratio;
        if visible_comp > n_comp {
            visible_comp = n_comp;
        }
        abi_attention_prefill_write_head(
            token,
            head,
            raw_start,
            raw_count,
            visible_comp,
            n_comp,
            n_head,
            head_dim,
            sinks,
            q,
            raw_kv,
            comp_kv,
            comp_mask,
            use_mask,
            &mut heads,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_static_mixed_heads8_online_kernel(
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        static mut KV_SHARED: SharedArray<f32, 2048> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || head_dim != 512 {
            return;
        }
        let lane = tid & 31;
        let head = head_group * 8 + (tid >> 5);
        let valid_head = head < n_head;
        let raw_start = if window != 0 && token + 1 > window {
            token + 1 - window
        } else {
            0
        };
        let raw_count = token + 1 - raw_start;
        let mut visible_comp = if ratio == 0 { 0 } else { (token + 1) / ratio };
        if visible_comp > n_comp {
            visible_comp = n_comp;
        }
        let n_score = raw_count + visible_comp;
        let query_base =
            ((token as usize * n_head as usize + head as usize) * head_dim as usize) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let mut query = [0.0_f32; 16];
        let mut output = [0.0_f32; 16];
        if valid_head {
            let mut item = 0_u32;
            while item < 16 {
                let dimension = (item >> 2) * 128 + lane * 4 + (item & 3);
                query[item as usize] = q[query_base + dimension as usize];
                item += 1;
            }
        }
        let mut max_score = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        let mut row0 = 0_u32;
        while row0 < n_score {
            let remaining = n_score - row0;
            let rows = if remaining < 4 { remaining } else { 4 };
            let mut offset = tid;
            while offset < rows * 512 {
                let row = row0 + offset / 512;
                let dimension = offset % 512;
                let value = if row < raw_count {
                    raw_kv[((raw_start + row) * head_dim + dimension) as usize]
                } else {
                    comp_kv[((row - raw_count) * head_dim + dimension) as usize]
                };
                unsafe {
                    KV_SHARED[offset as usize] = value;
                }
                offset += thread::blockDim_x();
            }
            thread::sync_threads();
            if valid_head {
                let mut row = 0_u32;
                while row < rows {
                    let row_base = (row * 512) as usize;
                    let mut values = [0.0_f32; 16];
                    let mut dot = 0.0_f32;
                    let mut item = 0_u32;
                    while item < 16 {
                        let dimension = (item >> 2) * 128 + lane * 4 + (item & 3);
                        let value = unsafe { KV_SHARED[row_base + dimension as usize] };
                        values[item as usize] = value;
                        dot += query[item as usize] * value;
                        item += 1;
                    }
                    let mut stride = 16_u32;
                    while stride > 0 {
                        dot += warp::shuffle_xor_f32(dot, stride);
                        stride >>= 1;
                    }
                    let score = dot * scale;
                    let next_max = abi_attention_maximum(max_score, score);
                    let old_scale = (max_score - next_max).exp();
                    let row_scale = (score - next_max).exp();
                    denominator = denominator * old_scale + row_scale;
                    item = 0;
                    while item < 16 {
                        output[item as usize] =
                            output[item as usize] * old_scale + values[item as usize] * row_scale;
                        item += 1;
                    }
                    max_score = next_max;
                    row += 1;
                }
            }
            thread::sync_threads();
            row0 += rows;
        }
        if valid_head {
            let sink = sinks[head as usize];
            let next_max = abi_attention_maximum(max_score, sink);
            let old_scale = (max_score - next_max).exp();
            denominator = denominator * old_scale + (sink - next_max).exp();
            let inverse = if denominator == 0.0 {
                0.0
            } else {
                1.0 / denominator
            };
            let mut item = 0_u32;
            while item < 16 {
                let dimension = (item >> 2) * 128 + lane * 4 + (item & 3);
                unsafe {
                    *heads.get_unchecked_mut(query_base + dimension as usize) =
                        output[item as usize] * old_scale * inverse;
                }
                item += 1;
            }
        }
    }

    #[kernel]
    pub fn abi_attention_prefill_pack_mixed_kv_kernel(
        n_tokens: u32,
        n_comp: u32,
        head_dim: u32,
        raw_kv: &[f32],
        comp_kv: &[f32],
        mut dst: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = (n_tokens + n_comp) * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let row = index as u32 / head_dim;
        let value = if row < n_tokens {
            raw_kv[(row * head_dim + dimension) as usize]
        } else {
            comp_kv[((row - n_tokens) * head_dim + dimension) as usize]
        };
        unsafe {
            *dst.get_unchecked_mut(index) = value;
        }
    }

    #[kernel]
    pub fn abi_attention_prefill_pack_q_heads_kernel(
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        q: &[f32],
        mut dst: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = n_tokens * n_head * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let token_head = index as u32 / head_dim;
        let token = token_head % n_tokens;
        let head = token_head / n_tokens;
        unsafe {
            *dst.get_unchecked_mut(index) =
                q[((token * n_head + head) * head_dim + dimension) as usize];
        }
    }

    #[kernel]
    pub fn abi_attention_prefill_replicate_kv_kernel(
        n_keys: u32,
        n_head: u32,
        head_dim: u32,
        kv: &[f32],
        mut keys: DisjointSlice<f32>,
        mut keys_transposed: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = n_head * n_keys * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let row_head = index as u32 / head_dim;
        let row = row_head % n_keys;
        let head = row_head / n_keys;
        let value = kv[(row * head_dim + dimension) as usize];
        unsafe {
            *keys.get_unchecked_mut(index) = value;
            *keys_transposed
                .get_unchecked_mut(((head * head_dim + dimension) * n_keys + row) as usize) = value;
        }
    }

    #[kernel]
    pub fn abi_attention_prefill_raw_softmax_kernel(
        n_tokens: u32,
        window: u32,
        n_keys: u32,
        n_head: u32,
        sinks: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let score_base = ((head * n_tokens + token) * n_keys) as usize;
        let mut max_score = sinks[head as usize];
        let mut key = 0_u32;
        while key < n_keys {
            let valid = key <= token && (window == 0 || token - key < window);
            let score = if valid {
                unsafe { *scores.get_unchecked_mut(score_base + key as usize) }
            } else {
                -1.0e30
            };
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = score;
            }
            max_score = abi_attention_maximum(max_score, score);
            key += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        key = 0;
        while key < n_keys {
            let score = unsafe { *scores.get_unchecked_mut(score_base + key as usize) };
            let probability = if score > -1.0e20 {
                (score - max_score).exp()
            } else {
                0.0
            };
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = probability;
            }
            denominator += probability;
            key += 1;
        }
        key = 0;
        while key < n_keys {
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) /= denominator;
            }
            key += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_prefill_mixed_softmax_kernel(
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_keys: u32,
        n_head: u32,
        use_mask: u32,
        sinks: &[f32],
        comp_mask: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || ratio == 0 || thread::threadIdx_x() != 0 {
            return;
        }
        let score_base = ((head * n_tokens + token) * n_keys) as usize;
        let visible_comp = (token + 1) / ratio;
        let mut max_score = sinks[head as usize];
        let mut key = 0_u32;
        while key < n_keys {
            let mut score = -1.0e30;
            if key < n_tokens {
                if key <= token && (window == 0 || token - key < window) {
                    score = unsafe { *scores.get_unchecked_mut(score_base + key as usize) };
                }
            } else {
                let compressed = key - n_tokens;
                if compressed < n_comp && compressed < visible_comp {
                    let add = if use_mask != 0 {
                        comp_mask[(token * n_comp + compressed) as usize]
                    } else {
                        0.0
                    };
                    if add > -1.0e20 {
                        score =
                            unsafe { *scores.get_unchecked_mut(score_base + key as usize) } + add;
                    }
                }
            }
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = score;
            }
            max_score = abi_attention_maximum(max_score, score);
            key += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        key = 0;
        while key < n_keys {
            let score = unsafe { *scores.get_unchecked_mut(score_base + key as usize) };
            let probability = if score > -1.0e20 {
                (score - max_score).exp()
            } else {
                0.0
            };
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = probability;
            }
            denominator += probability;
            key += 1;
        }
        key = 0;
        while key < n_keys {
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) /= denominator;
            }
            key += 1;
        }
    }

    #[kernel]
    pub fn abi_attention_prefill_unpack_heads_kernel(
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        tmp: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = n_tokens * n_head * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let token_head = index as u32 / head_dim;
        let head = token_head % n_head;
        let token = token_head / n_head;
        unsafe {
            *heads.get_unchecked_mut(index) =
                tmp[((head * n_tokens + token) * head_dim + dimension) as usize];
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn abi_attention_prefill_write_head(
        token: u32,
        head: u32,
        raw_start: u32,
        raw_count: u32,
        visible_comp: u32,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        use_mask: u32,
        heads: &mut DisjointSlice<f32>,
    ) {
        let query_base = ((token * n_head + head) * head_dim) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let mut max_score = sinks[head as usize];
        let mut row = 0_u32;
        while row < raw_count {
            max_score = abi_attention_maximum(
                max_score,
                abi_attention_dot(q, query_base, raw_kv, raw_start + row, head_dim) * scale,
            );
            row += 1;
        }
        let mut compressed = 0_u32;
        while compressed < visible_comp {
            let add = if use_mask != 0 {
                comp_mask[(token * n_comp + compressed) as usize]
            } else {
                0.0
            };
            if add > -1.0e20 {
                max_score = abi_attention_maximum(
                    max_score,
                    abi_attention_dot(q, query_base, comp_kv, compressed, head_dim) * scale + add,
                );
            }
            compressed += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        row = 0;
        while row < raw_count {
            denominator += (abi_attention_dot(q, query_base, raw_kv, raw_start + row, head_dim)
                * scale
                - max_score)
                .exp();
            row += 1;
        }
        compressed = 0;
        while compressed < visible_comp {
            let add = if use_mask != 0 {
                comp_mask[(token * n_comp + compressed) as usize]
            } else {
                0.0
            };
            if add > -1.0e20 {
                denominator +=
                    (abi_attention_dot(q, query_base, comp_kv, compressed, head_dim) * scale + add
                        - max_score)
                        .exp();
            }
            compressed += 1;
        }
        let mut dimension = 0_u32;
        while dimension < head_dim {
            let mut accumulator = 0.0_f32;
            row = 0;
            while row < raw_count {
                let raw_row = raw_start + row;
                let weight = (abi_attention_dot(q, query_base, raw_kv, raw_row, head_dim) * scale
                    - max_score)
                    .exp();
                accumulator += raw_kv[(raw_row * head_dim + dimension) as usize] * weight;
                row += 1;
            }
            compressed = 0;
            while compressed < visible_comp {
                let add = if use_mask != 0 {
                    comp_mask[(token * n_comp + compressed) as usize]
                } else {
                    0.0
                };
                if add > -1.0e20 {
                    let weight = (abi_attention_dot(q, query_base, comp_kv, compressed, head_dim)
                        * scale
                        + add
                        - max_score)
                        .exp();
                    accumulator += comp_kv[(compressed * head_dim + dimension) as usize] * weight;
                }
                compressed += 1;
            }
            unsafe {
                *heads.get_unchecked_mut(query_base + dimension as usize) =
                    accumulator / denominator;
            }
            dimension += 1;
        }
    }

    #[kernel]
    pub fn abi_indexed_topk_sort_512_asc_kernel(
        n_tokens: u32,
        source: &[i32],
        mut sorted: DisjointSlice<i32>,
    ) {
        static mut ROWS: SharedArray<i32, 512> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= 512 {
            return;
        }
        let index = tid as usize;
        let offset = token as usize * 512 + index;
        unsafe {
            ROWS[index] = source[offset];
        }
        thread::sync_threads();
        let mut width = 2_u32;
        while width <= 512 {
            let mut stride = width >> 1;
            while stride > 0 {
                let other = tid ^ stride;
                if other > tid && other < 512 {
                    let other_index = other as usize;
                    let left = unsafe { ROWS[index] };
                    let right = unsafe { ROWS[other_index] };
                    let ascending = (tid & width) == 0;
                    if (ascending && left > right) || (!ascending && left < right) {
                        unsafe {
                            ROWS[index] = right;
                            ROWS[other_index] = left;
                        }
                    }
                }
                thread::sync_threads();
                stride >>= 1;
            }
            width <<= 1;
        }
        unsafe {
            *sorted.get_unchecked_mut(offset) = ROWS[index];
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_indexed_mixed_kernel(
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
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        mut heads: DisjointSlice<f32>,
    ) {
        static mut SCORES: SharedArray<f32, 768> = SharedArray::UNINIT;
        static mut RAW_ROWS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut COMP_ROWS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut META: SharedArray<u32, 3> = SharedArray::UNINIT;
        static mut SOFTMAX: SharedArray<f32, 2> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        let tid = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x();
        if token >= n_tokens || head >= n_head {
            return;
        }
        let qpos = pos0.wrapping_add(token);
        let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
        let mut visible_comp = n_comp;
        if ratio != 0 {
            visible_comp = qpos.wrapping_add(1) / ratio;
            if visible_comp > n_comp {
                visible_comp = n_comp;
            }
        }
        if tid == 0 {
            let mut raw_first = 0_u32;
            let mut raw_count = 0_u32;
            if n_raw != 0 {
                let raw_last_pos = first_raw_pos.wrapping_add(n_raw).wrapping_sub(1);
                if qpos >= first_raw_pos {
                    let mut lo = first_raw_pos;
                    if window != 0 && qpos.wrapping_add(1) > window {
                        let window_lo = qpos.wrapping_add(1).wrapping_sub(window);
                        if window_lo > lo {
                            lo = window_lo;
                        }
                    }
                    let hi = if qpos < raw_last_pos {
                        qpos
                    } else {
                        raw_last_pos
                    };
                    if hi >= lo {
                        raw_first = lo.wrapping_sub(first_raw_pos);
                        raw_count = hi.wrapping_sub(lo).wrapping_add(1);
                        if raw_count > 256 {
                            raw_count = 256;
                        }
                    }
                }
            }
            let mut comp_count = 0_u32;
            let mut selected = 0_u32;
            while selected < top_k && comp_count < 512 {
                let compressed = topk[(token * top_k + selected) as usize];
                if compressed >= 0 && (compressed as u32) < visible_comp {
                    unsafe {
                        COMP_ROWS[comp_count as usize] = compressed as u32;
                    }
                    comp_count += 1;
                }
                selected += 1;
            }
            unsafe {
                META[0] = raw_count;
                META[1] = raw_first;
                META[2] = comp_count;
            }
        }
        thread::sync_threads();
        let raw_count = unsafe { META[0] };
        let raw_first = unsafe { META[1] };
        let comp_count = unsafe { META[2] };
        let mut row = thread::threadIdx_x();
        while row < raw_count {
            unsafe {
                RAW_ROWS[row as usize] =
                    raw_start.wrapping_add(raw_first).wrapping_add(row) % raw_cap;
            }
            row += block_dim;
        }
        thread::sync_threads();

        let query_base =
            ((token as usize * n_head as usize + head as usize) * head_dim as usize) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let n_score = raw_count + comp_count;
        if comp_count == 0 {
            row = thread::threadIdx_x();
            while row < raw_count {
                let raw_row = unsafe { RAW_ROWS[row as usize] };
                unsafe {
                    SCORES[row as usize] =
                        abi_attention_dot(q, query_base, raw_kv, raw_row, head_dim) * scale;
                }
                row += block_dim;
            }
        } else {
            let qlane = thread::threadIdx_x() & 7;
            let qgroup = thread::threadIdx_x() >> 3;
            let mut row0 = 0_u32;
            while row0 < n_score {
                row = row0 + qgroup;
                if row < n_score {
                    let (kv, kv_row) = if row < raw_count {
                        (raw_kv, unsafe { RAW_ROWS[row as usize] })
                    } else {
                        (comp_kv, unsafe { COMP_ROWS[(row - raw_count) as usize] })
                    };
                    let mut dot = 0.0_f32;
                    let mut dimension = qlane;
                    while dimension < head_dim {
                        dot += q[query_base + dimension as usize]
                            * kv[(kv_row * head_dim + dimension) as usize];
                        dimension += 8;
                    }
                    let mask = 0xff_u32 << (thread::threadIdx_x() & 24);
                    let mut stride = 4_u32;
                    while stride > 0 {
                        dot += warp::shuffle_xor_f32_sync(mask, dot, stride);
                        stride >>= 1;
                    }
                    if qlane == 0 {
                        unsafe {
                            SCORES[row as usize] = dot * scale;
                        }
                    }
                }
                row0 += 32;
            }
        }
        thread::sync_threads();
        let mut local_max = sinks[head as usize];
        let mut score = thread::threadIdx_x();
        while score < n_score {
            local_max = abi_attention_maximum(local_max, unsafe { SCORES[score as usize] });
            score += block_dim;
        }
        unsafe {
            PARTIAL[tid] = local_max;
        }
        thread::sync_threads();
        let mut stride = block_dim >> 1;
        while stride > 0 {
            if thread::threadIdx_x() < stride {
                unsafe {
                    PARTIAL[tid] =
                        abi_attention_maximum(PARTIAL[tid], PARTIAL[tid + stride as usize]);
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                SOFTMAX[0] = PARTIAL[0];
            }
        }
        thread::sync_threads();
        let max_score = unsafe { SOFTMAX[0] };
        let mut denominator = 0.0_f32;
        score = thread::threadIdx_x();
        while score < n_score {
            let probability = (unsafe { SCORES[score as usize] } - max_score).exp();
            unsafe {
                SCORES[score as usize] = probability;
            }
            denominator += probability;
            score += block_dim;
        }
        unsafe {
            PARTIAL[tid] = denominator;
        }
        thread::sync_threads();
        stride = block_dim >> 1;
        while stride > 0 {
            if thread::threadIdx_x() < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride as usize];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                SOFTMAX[1] = PARTIAL[0] + (sinks[head as usize] - max_score).exp();
            }
        }
        thread::sync_threads();
        denominator = unsafe { SOFTMAX[1] };
        if head_dim == 512 && block_dim == 256 {
            let dimension0 = thread::threadIdx_x();
            let dimension1 = dimension0 + 256;
            let mut accumulator0 = 0.0_f32;
            let mut accumulator1 = 0.0_f32;
            row = 0;
            while row < raw_count {
                let probability = unsafe { SCORES[row as usize] };
                let raw_row = unsafe { RAW_ROWS[row as usize] };
                accumulator0 += raw_kv[(raw_row * head_dim + dimension0) as usize] * probability;
                accumulator1 += raw_kv[(raw_row * head_dim + dimension1) as usize] * probability;
                row += 1;
            }
            let mut compressed = 0_u32;
            while compressed < comp_count {
                let probability = unsafe { SCORES[(raw_count + compressed) as usize] };
                let comp_row = unsafe { COMP_ROWS[compressed as usize] };
                accumulator0 += comp_kv[(comp_row * head_dim + dimension0) as usize] * probability;
                accumulator1 += comp_kv[(comp_row * head_dim + dimension1) as usize] * probability;
                compressed += 1;
            }
            unsafe {
                *heads.get_unchecked_mut(query_base + dimension0 as usize) =
                    accumulator0 / denominator;
                *heads.get_unchecked_mut(query_base + dimension1 as usize) =
                    accumulator1 / denominator;
            }
        } else {
            let mut dimension = thread::threadIdx_x();
            while dimension < head_dim {
                let mut accumulator = 0.0_f32;
                row = 0;
                while row < raw_count {
                    let probability = unsafe { SCORES[row as usize] };
                    let raw_row = unsafe { RAW_ROWS[row as usize] };
                    accumulator += raw_kv[(raw_row * head_dim + dimension) as usize] * probability;
                    row += 1;
                }
                let mut compressed = 0_u32;
                while compressed < comp_count {
                    let probability = unsafe { SCORES[(raw_count + compressed) as usize] };
                    let comp_row = unsafe { COMP_ROWS[compressed as usize] };
                    accumulator +=
                        comp_kv[(comp_row * head_dim + dimension) as usize] * probability;
                    compressed += 1;
                }
                unsafe {
                    *heads.get_unchecked_mut(query_base + dimension as usize) =
                        accumulator / denominator;
                }
                dimension += block_dim;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_indexed_mixed_heads8_online_kernel(
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
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        mut heads: DisjointSlice<f32>,
    ) {
        static mut KV_SHARED: SharedArray<f32, 4096> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || head_dim != 512 {
            return;
        }
        let lane = tid & 31;
        let head = head_group * 16 + (tid >> 5);
        let valid_head = head < n_head;
        let qpos = pos0.wrapping_add(token);
        let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
        let raw_last_pos = first_raw_pos.wrapping_add(n_raw).wrapping_sub(1);
        let mut raw_first = 0_u32;
        let mut raw_count = 0_u32;
        if qpos >= first_raw_pos {
            let mut lo = first_raw_pos;
            if window != 0 && qpos.wrapping_add(1) > window {
                let window_lo = qpos.wrapping_add(1).wrapping_sub(window);
                if window_lo > lo {
                    lo = window_lo;
                }
            }
            let hi = if qpos < raw_last_pos {
                qpos
            } else {
                raw_last_pos
            };
            if hi >= lo {
                raw_first = lo.wrapping_sub(first_raw_pos);
                raw_count = hi.wrapping_sub(lo).wrapping_add(1);
                if raw_count > 256 {
                    raw_count = 256;
                }
            }
        }
        let mut visible_comp = n_comp;
        if ratio != 0 {
            visible_comp = qpos.wrapping_add(1) / ratio;
            if visible_comp > n_comp {
                visible_comp = n_comp;
            }
        }
        let comp_count = if top_k < visible_comp {
            top_k
        } else {
            visible_comp
        };
        let n_score = raw_count + comp_count;
        let query_base =
            ((token as usize * n_head as usize + head as usize) * head_dim as usize) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let mut query = [0.0_f32; 16];
        let mut output = [0.0_f32; 16];
        if valid_head {
            let mut item = 0_u32;
            while item < 16 {
                let dimension = (item >> 2) * 128 + lane * 4 + (item & 3);
                query[item as usize] = q[query_base + dimension as usize];
                item += 1;
            }
        }
        let mut max_score = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        let mut row0 = 0_u32;
        while row0 < n_score {
            let remaining = n_score - row0;
            let rows = if remaining < 8 { remaining } else { 8 };
            let mut offset = tid;
            while offset < rows * 512 {
                let row = row0 + offset / 512;
                let dimension = offset % 512;
                let value = if row < raw_count {
                    let raw_row = raw_start.wrapping_add(raw_first).wrapping_add(row) % raw_cap;
                    raw_kv[(raw_row * head_dim + dimension) as usize]
                } else {
                    let selected = row - raw_count;
                    let compressed = topk[(token * top_k + selected) as usize] as u32;
                    comp_kv[(compressed * head_dim + dimension) as usize]
                };
                unsafe {
                    KV_SHARED[offset as usize] = value;
                }
                offset += thread::blockDim_x();
            }
            thread::sync_threads();
            if valid_head {
                let mut row = 0_u32;
                while row < rows {
                    let row_base = (row * 512) as usize;
                    let mut values = [0.0_f32; 16];
                    let mut dot = 0.0_f32;
                    let mut item = 0_u32;
                    while item < 16 {
                        let dimension = (item >> 2) * 128 + lane * 4 + (item & 3);
                        let value = unsafe { KV_SHARED[row_base + dimension as usize] };
                        values[item as usize] = value;
                        dot += query[item as usize] * value;
                        item += 1;
                    }
                    let mut stride = 16_u32;
                    while stride > 0 {
                        dot += warp::shuffle_xor_f32(dot, stride);
                        stride >>= 1;
                    }
                    let score = dot * scale;
                    let next_max = abi_attention_maximum(max_score, score);
                    let old_scale = (max_score - next_max).exp();
                    let row_scale = (score - next_max).exp();
                    denominator = denominator * old_scale + row_scale;
                    item = 0;
                    while item < 16 {
                        output[item as usize] =
                            output[item as usize] * old_scale + values[item as usize] * row_scale;
                        item += 1;
                    }
                    max_score = next_max;
                    row += 1;
                }
            }
            thread::sync_threads();
            row0 += rows;
        }
        if valid_head {
            let sink = sinks[head as usize];
            let next_max = abi_attention_maximum(max_score, sink);
            let old_scale = (max_score - next_max).exp();
            denominator = denominator * old_scale + (sink - next_max).exp();
            let inverse = if denominator == 0.0 {
                0.0
            } else {
                1.0 / denominator
            };
            let mut item = 0_u32;
            while item < 16 {
                let dimension = (item >> 2) * 128 + lane * 4 + (item & 3);
                unsafe {
                    *heads.get_unchecked_mut(query_base + dimension as usize) =
                        output[item as usize] * old_scale * inverse;
                }
                item += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_attention_indexed_mixed_heads8_rb4_kernel(
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
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        mut heads: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        if token >= n_tokens || head_dim != 512 || thread::threadIdx_x() != 0 {
            return;
        }
        let mut local_head = 0_u32;
        while local_head < 8 {
            let head = head_group * 8 + local_head;
            if head < n_head {
                abi_attention_indexed_write_head(
                    token, head, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window,
                    ratio, n_head, head_dim, 1, sinks, q, raw_kv, comp_kv, topk, &mut heads,
                );
            }
            local_head += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn abi_attention_indexed_write_head(
        token: u32,
        head: u32,
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
        filter_entries: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        heads: &mut DisjointSlice<f32>,
    ) {
        let qpos = pos0.wrapping_add(token);
        let first_raw_pos = pos0.wrapping_add(n_tokens).wrapping_sub(n_raw);
        let raw_last_pos = first_raw_pos.wrapping_add(n_raw).wrapping_sub(1);
        let mut raw_first = 0_u32;
        let mut raw_count = 0_u32;
        if qpos >= first_raw_pos {
            let mut lo = first_raw_pos;
            if window != 0 && qpos.wrapping_add(1) > window {
                let window_lo = qpos.wrapping_add(1).wrapping_sub(window);
                if window_lo > lo {
                    lo = window_lo;
                }
            }
            let hi = if qpos < raw_last_pos {
                qpos
            } else {
                raw_last_pos
            };
            if hi >= lo {
                raw_first = lo.wrapping_sub(first_raw_pos);
                raw_count = hi.wrapping_sub(lo).wrapping_add(1);
                if raw_count > 256 {
                    raw_count = 256;
                }
            }
        }
        let mut visible_comp = n_comp;
        if ratio != 0 {
            visible_comp = qpos.wrapping_add(1) / ratio;
            if visible_comp > n_comp {
                visible_comp = n_comp;
            }
        }
        let selected_count = if filter_entries == 0 && visible_comp < top_k {
            visible_comp
        } else {
            top_k
        };
        let query_base =
            ((token as usize * n_head as usize + head as usize) * head_dim as usize) as usize;
        let scale = unsafe { __nv_rsqrtf(head_dim as f32) };
        let mut max_score = sinks[head as usize];
        let mut row = 0_u32;
        while row < raw_count {
            let raw_row = raw_start.wrapping_add(raw_first).wrapping_add(row) % raw_cap;
            max_score = abi_attention_maximum(
                max_score,
                abi_attention_dot(q, query_base, raw_kv, raw_row, head_dim) * scale,
            );
            row += 1;
        }
        let mut selected = 0_u32;
        while selected < selected_count {
            let compressed = topk[(token * top_k + selected) as usize];
            if filter_entries == 0 || (compressed >= 0 && (compressed as u32) < visible_comp) {
                max_score = abi_attention_maximum(
                    max_score,
                    abi_attention_dot(q, query_base, comp_kv, compressed as u32, head_dim) * scale,
                );
            }
            selected += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        row = 0;
        while row < raw_count {
            let raw_row = raw_start.wrapping_add(raw_first).wrapping_add(row) % raw_cap;
            denominator += (abi_attention_dot(q, query_base, raw_kv, raw_row, head_dim) * scale
                - max_score)
                .exp();
            row += 1;
        }
        selected = 0;
        while selected < selected_count {
            let compressed = topk[(token * top_k + selected) as usize];
            if filter_entries == 0 || (compressed >= 0 && (compressed as u32) < visible_comp) {
                denominator +=
                    (abi_attention_dot(q, query_base, comp_kv, compressed as u32, head_dim)
                        * scale
                        - max_score)
                        .exp();
            }
            selected += 1;
        }
        let mut dimension = 0_u32;
        while dimension < head_dim {
            let mut accumulator = 0.0_f32;
            row = 0;
            while row < raw_count {
                let raw_row = raw_start.wrapping_add(raw_first).wrapping_add(row) % raw_cap;
                let weight = (abi_attention_dot(q, query_base, raw_kv, raw_row, head_dim) * scale
                    - max_score)
                    .exp();
                accumulator += raw_kv[(raw_row * head_dim + dimension) as usize] * weight;
                row += 1;
            }
            selected = 0;
            while selected < selected_count {
                let compressed = topk[(token * top_k + selected) as usize];
                if filter_entries == 0 || (compressed >= 0 && (compressed as u32) < visible_comp) {
                    let weight =
                        (abi_attention_dot(q, query_base, comp_kv, compressed as u32, head_dim)
                            * scale
                            - max_score)
                            .exp();
                    accumulator +=
                        comp_kv[(compressed as u32 * head_dim + dimension) as usize] * weight;
                }
                selected += 1;
            }
            unsafe {
                *heads.get_unchecked_mut(query_base + dimension as usize) =
                    accumulator / denominator;
            }
            dimension += 1;
        }
    }

    fn abi_attention_dot(q: &[f32], query_base: usize, kv: &[f32], row: u32, head_dim: u32) -> f32 {
        let mut value = 0.0_f32;
        let mut dimension = 0_u32;
        while dimension < head_dim {
            value += q[query_base + dimension as usize] * kv[(row * head_dim + dimension) as usize];
            dimension += 1;
        }
        value
    }

    fn abi_attention_maximum(left: f32, right: f32) -> f32 {
        if right > left {
            right
        } else {
            left
        }
    }

    #[kernel]
    pub fn abi_fp8_kv_quantize_kernel(
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
        mut x: DisjointSlice<f32>,
    ) {
        static mut SCRATCH: SharedArray<f32, 64> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= n_tok {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let n_nope = (head_dim - n_rot) as usize;
        let base = row as usize * head_dim as usize;
        let mut off = 0_usize;
        while off < n_nope {
            let index = off + tid;
            let valid = index < n_nope;
            let value = if valid {
                unsafe { *x.as_mut_ptr().add(base + index) }
            } else {
                0.0
            };
            unsafe {
                SCRATCH[tid] = abi_absolute(value);
            }
            thread::sync_threads();
            let mut stride = 32_usize;
            while stride > 0 {
                if tid < stride {
                    let other = unsafe { SCRATCH[tid + stride] };
                    if other > unsafe { SCRATCH[tid] } {
                        unsafe {
                            SCRATCH[tid] = other;
                        }
                    }
                }
                thread::sync_threads();
                stride >>= 1;
            }
            let amax = if unsafe { SCRATCH[0] } > 1.0e-4 {
                unsafe { SCRATCH[0] }
            } else {
                1.0e-4
            };
            let scale = 2.0_f32.powf((amax / 448.0).log2().ceil());
            if valid {
                let mut scaled = value / scale;
                if scaled > 448.0 {
                    scaled = 448.0;
                } else if scaled < -448.0 {
                    scaled = -448.0;
                }
                unsafe {
                    *x.get_unchecked_mut(base + index) = abi_e4m3fn_dequant(scaled) * scale;
                }
            }
            thread::sync_threads();
            off += 64;
        }
    }

    fn abi_absolute(value: f32) -> f32 {
        if value < 0.0 {
            -value
        } else {
            value
        }
    }

    fn abi_e4m3fn_value(value: i32) -> f32 {
        let exponent = (value >> 3) & 15;
        let mantissa = value & 7;
        if exponent == 0 {
            mantissa as f32 * 0.001953125
        } else {
            (1.0 + mantissa as f32 * 0.125) * 2.0_f32.powf(exponent as f32 - 7.0)
        }
    }

    fn abi_e4m3fn_dequant(value: f32) -> f32 {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let mut magnitude = abi_absolute(value);
        if magnitude > 448.0 {
            magnitude = 448.0;
        }
        let mut lo = 0_i32;
        let mut hi = 126_i32;
        while lo < hi {
            let mid = (lo + hi + 1) >> 1;
            if abi_e4m3fn_value(mid) <= magnitude {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut best = lo;
        if best < 126 {
            let best_diff = abi_absolute(magnitude - abi_e4m3fn_value(best));
            let next_diff = abi_absolute(magnitude - abi_e4m3fn_value(best + 1));
            if next_diff < best_diff
                || (next_diff == best_diff && (best + 1) & 1 == 0 && best & 1 != 0)
            {
                best += 1;
            }
        }
        sign * abi_e4m3fn_value(best)
    }

    #[kernel]
    pub fn abi_indexer_hadamard_fp4_kernel(n_rows: u32, head_dim: u32, mut x: DisjointSlice<f32>) {
        static mut VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut MAGNITUDES: SharedArray<f32, 128> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x() as usize;
        if row >= n_rows || head_dim != 128 || tid >= 128 {
            return;
        }
        let base = row as usize * head_dim as usize;
        unsafe {
            VALUES[tid] = *x.as_mut_ptr().add(base + tid);
        }
        thread::sync_threads();

        let mut stride = 1_usize;
        while stride < 128 {
            if tid & stride == 0 {
                let pair = (tid & !(2 * stride - 1)) + (tid & (stride - 1));
                let a = unsafe { VALUES[pair] };
                let b = unsafe { VALUES[pair + stride] };
                unsafe {
                    VALUES[pair] = a + b;
                    VALUES[pair + stride] = a - b;
                }
            }
            thread::sync_threads();
            stride <<= 1;
        }

        let value = unsafe { VALUES[tid] } * 0.08838834764831845;
        let block_base = (tid >> 5) * 32;
        let lane = tid & 31;
        unsafe {
            MAGNITUDES[tid] = abi_absolute(value);
        }
        thread::sync_threads();
        stride = 16;
        while stride > 0 {
            if lane < stride {
                let other = unsafe { MAGNITUDES[block_base + lane + stride] };
                if other > unsafe { MAGNITUDES[block_base + lane] } {
                    unsafe {
                        MAGNITUDES[block_base + lane] = other;
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let amax = if unsafe { MAGNITUDES[block_base] } > 7.052966104933725e-38 {
            unsafe { MAGNITUDES[block_base] }
        } else {
            7.052966104933725e-38
        };
        let scale = 2.0_f32.powf((amax / 6.0).log2().ceil());
        let mut scaled = value / scale;
        if scaled > 6.0 {
            scaled = 6.0;
        } else if scaled < -6.0 {
            scaled = -6.0;
        }
        unsafe {
            *x.get_unchecked_mut(base + tid) = abi_e2m1fn_dequant(scaled) * scale;
        }
    }

    #[kernel]
    pub fn abi_indexer_scores_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let comp = thread::blockIdx_x();
        let token = thread::blockIdx_y();
        if comp >= n_comp || token >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        if causal != 0 {
            let visible = (pos0 + token + 1) / ratio;
            if comp >= visible {
                if tid == 0 {
                    let output = token as usize * n_comp as usize + comp as usize;
                    unsafe {
                        *scores.get_unchecked_mut(output) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
        }

        let mut total = 0.0_f32;
        let mut head = 0;
        while head < n_head {
            let q_base = (token as usize * n_head as usize + head as usize) * head_dim as usize;
            let comp_base = comp as usize * head_dim as usize;
            let mut dot = 0.0_f32;
            let mut dimension = tid;
            while dimension < head_dim as usize {
                dot += q[q_base + dimension] * index_comp[comp_base + dimension];
                dimension += 256;
            }
            unsafe {
                PARTIAL[tid] = dot;
            }
            thread::sync_threads();

            let mut stride = 128;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        PARTIAL[tid] += PARTIAL[tid + stride];
                    }
                }
                thread::sync_threads();
                stride >>= 1;
            }

            let reduced = unsafe { PARTIAL[0] };
            let positive = if (reduced.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || reduced <= 0.0_f32
            {
                0.0_f32
            } else {
                reduced
            };
            total += positive * weights[token as usize * n_head as usize + head as usize];
            thread::sync_threads();
            head += 1;
        }
        if tid == 0 {
            let output = token as usize * n_comp as usize + comp as usize;
            unsafe {
                *scores.get_unchecked_mut(output) = total * scale;
            }
        }
    }

    #[kernel]
    pub fn abi_indexer_score_one_direct_kernel(
        n_comp: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut K_ROW: SharedArray<f32, ABI_INDEXER_HEAD_DIM> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 4> = SharedArray::UNINIT;

        let comp = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if comp >= n_comp || tid >= ABI_INDEXER_DIRECT_THREADS {
            return;
        }
        let lane = tid & 31;
        let warp_id = tid >> 5;
        if causal != 0 {
            let visible = if ratio != 0 {
                (pos0 + 1) / ratio
            } else {
                n_comp
            };
            if comp >= visible {
                if tid == 0 {
                    unsafe {
                        *scores.get_unchecked_mut(comp as usize) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
        }

        unsafe {
            K_ROW[tid as usize] = index_comp[comp as usize * ABI_INDEXER_HEAD_DIM + tid as usize];
        }
        thread::sync_threads();

        let mut total = 0.0_f32;
        let mut head_group = 0_u32;
        while head_group < ABI_INDEXER_N_HEAD as u32 {
            let head = head_group + warp_id;
            let q_base = head as usize * ABI_INDEXER_HEAD_DIM + lane as usize * 4;
            let k_base = lane as usize * 4;
            let mut dot = q[q_base] * unsafe { K_ROW[k_base] }
                + q[q_base + 1] * unsafe { K_ROW[k_base + 1] }
                + q[q_base + 2] * unsafe { K_ROW[k_base + 2] }
                + q[q_base + 3] * unsafe { K_ROW[k_base + 3] };
            let mut offset = 16_u32;
            while offset > 0 {
                dot += warp::shuffle_down_f32(dot, offset);
                offset >>= 1;
            }
            if lane == 0 {
                let positive = if (dot.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || dot <= 0.0_f32 {
                    0.0_f32
                } else {
                    dot
                };
                unsafe {
                    PARTIAL[warp_id as usize] = positive * weights[head as usize] * scale;
                }
            }
            thread::sync_threads();
            if tid == 0 {
                total += unsafe { PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3] };
            }
            thread::sync_threads();
            head_group += 4;
        }
        if tid == 0 {
            unsafe {
                *scores.get_unchecked_mut(comp as usize) = total;
            }
        }
    }

    #[kernel]
    pub fn abi_indexer_scores_wmma_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut A_TILE: SharedArray<f16, { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K }, 16> =
            SharedArray::UNINIT;
        static mut B_LO_TILE: SharedArray<f16, { ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N }, 16> =
            SharedArray::UNINIT;
        static mut B_HI_TILE: SharedArray<f16, { ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N }, 16> =
            SharedArray::UNINIT;
        static mut C_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_TILE_COMPONENTS },
        > = SharedArray::UNINIT;
        static mut ACC_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_TILE_COMPONENTS },
        > = SharedArray::UNINIT;

        let tile_c = thread::blockIdx_x() as usize * ABI_INDEXER_TILE_COMPONENTS;
        let tile_t = thread::blockIdx_y() as usize * ABI_INDEXER_TILE_TOKENS;
        let tid = thread::threadIdx_x() as usize;
        if tid >= ABI_INDEXER_WMMA_THREADS as usize {
            return;
        }

        if causal != 0 {
            let tile_end = tile_t as u32 + ABI_INDEXER_TILE_TOKENS as u32;
            let last_token = if tile_end < n_tokens {
                tile_end
            } else {
                n_tokens
            };
            let max_visible = if last_token > tile_t as u32 && ratio != 0 {
                let visible = (pos0 + last_token) / ratio;
                if visible < n_comp {
                    visible
                } else {
                    n_comp
                }
            } else {
                0
            };
            if tile_c as u32 >= max_visible {
                let mut i = tid;
                while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_TILE_COMPONENTS {
                    let row = i / ABI_INDEXER_TILE_COMPONENTS;
                    let col = i % ABI_INDEXER_TILE_COMPONENTS;
                    let token = tile_t + row;
                    let comp = tile_c + col;
                    if token < n_tokens as usize && comp < n_comp as usize {
                        unsafe {
                            *scores.get_unchecked_mut(token * n_comp as usize + comp) =
                                f32::NEG_INFINITY;
                        }
                    }
                    i += ABI_INDEXER_WMMA_THREADS as usize;
                }
                return;
            }
        }

        let mut i = tid;
        while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_TILE_COMPONENTS {
            unsafe {
                ACC_TILE[i] = 0.0;
            }
            i += ABI_INDEXER_WMMA_THREADS as usize;
        }
        thread::sync_threads();

        let lane = tid & 31;
        let a_row = (lane & 7) + (lane & 8);
        let a_col = if lane & 16 == 0 { 0 } else { 8 };
        let b_row = (lane & 7) + (lane & 8);
        let mut head = 0_usize;
        while head < ABI_INDEXER_N_HEAD {
            let mut acc_lo = zero_accumulator();
            let mut acc_hi = zero_accumulator();
            let mut k0 = 0_usize;
            while k0 < ABI_INDEXER_HEAD_DIM {
                let mut a_index = tid;
                while a_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K {
                    let row = a_index / ABI_INDEXER_MMA_K;
                    let col = a_index % ABI_INDEXER_MMA_K;
                    let token = tile_t + row;
                    let value = if token < n_tokens as usize {
                        q[(token * ABI_INDEXER_N_HEAD + head) * ABI_INDEXER_HEAD_DIM + k0 + col]
                    } else {
                        0.0
                    };
                    unsafe {
                        A_TILE[a_index] = value as f16;
                    }
                    a_index += ABI_INDEXER_WMMA_THREADS as usize;
                }

                let mut b_index = tid;
                while b_index < ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N {
                    let row = b_index / ABI_INDEXER_MMA_N;
                    let col = b_index % ABI_INDEXER_MMA_N;
                    let comp_lo = tile_c + col;
                    let comp_hi = tile_c + ABI_INDEXER_MMA_N + col;
                    let dimension = k0 + row;
                    let value_lo = if comp_lo < n_comp as usize {
                        index_comp[comp_lo * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    let value_hi = if comp_hi < n_comp as usize {
                        index_comp[comp_hi * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    unsafe {
                        B_LO_TILE[b_index] = value_lo as f16;
                        B_HI_TILE[b_index] = value_hi as f16;
                    }
                    b_index += ABI_INDEXER_WMMA_THREADS as usize;
                }
                thread::sync_threads();

                let a_ptr = unsafe {
                    (&raw const A_TILE)
                        .cast::<f16>()
                        .add(a_row * ABI_INDEXER_MMA_K + a_col)
                }
                .cast::<u8>();
                let b_lo_ptr = unsafe {
                    (&raw const B_LO_TILE)
                        .cast::<f16>()
                        .add(b_row * ABI_INDEXER_MMA_N)
                }
                .cast::<u8>();
                let b_hi_ptr = unsafe {
                    (&raw const B_HI_TILE)
                        .cast::<f16>()
                        .add(b_row * ABI_INDEXER_MMA_N)
                }
                .cast::<u8>();
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                let b_lo_frag = unsafe { load_b_m16n8k16(b_lo_ptr) };
                let b_hi_frag = unsafe { load_b_m16n8k16(b_hi_ptr) };
                acc_lo = unsafe { mma_m16n8k16_f32_f16(acc_lo, a_frag, b_lo_frag) };
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                acc_hi = unsafe { mma_m16n8k16_f32_f16(acc_hi, a_frag, b_hi_frag) };
                thread::sync_threads();
                k0 += ABI_INDEXER_MMA_K;
            }

            let group_id = lane >> 2;
            let thread_in_group = lane & 3;
            let col_base = thread_in_group * 2;
            unsafe {
                C_TILE[group_id * ABI_INDEXER_TILE_COMPONENTS + col_base] = acc_lo.x();
                C_TILE[group_id * ABI_INDEXER_TILE_COMPONENTS + col_base + 1] = acc_lo.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_TILE_COMPONENTS + col_base] = acc_lo.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_TILE_COMPONENTS + col_base + 1] = acc_lo.w();
                C_TILE[group_id * ABI_INDEXER_TILE_COMPONENTS + ABI_INDEXER_MMA_N + col_base] =
                    acc_hi.x();
                C_TILE[group_id * ABI_INDEXER_TILE_COMPONENTS + ABI_INDEXER_MMA_N + col_base + 1] =
                    acc_hi.y();
                C_TILE
                    [(group_id + 8) * ABI_INDEXER_TILE_COMPONENTS + ABI_INDEXER_MMA_N + col_base] =
                    acc_hi.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_TILE_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base
                    + 1] = acc_hi.w();
            }
            thread::sync_threads();

            let mut output_index = tid;
            while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_TILE_COMPONENTS {
                let row = output_index / ABI_INDEXER_TILE_COMPONENTS;
                let col = output_index % ABI_INDEXER_TILE_COMPONENTS;
                let token = tile_t + row;
                let comp = tile_c + col;
                if token < n_tokens as usize && comp < n_comp as usize {
                    let value = unsafe { C_TILE[output_index] };
                    let positive = if (value.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || value <= 0.0
                    {
                        0.0
                    } else {
                        value
                    };
                    unsafe {
                        ACC_TILE[output_index] +=
                            positive * weights[token * ABI_INDEXER_N_HEAD + head];
                    }
                }
                output_index += ABI_INDEXER_WMMA_THREADS as usize;
            }
            thread::sync_threads();
            head += 1;
        }

        let mut output_index = tid;
        while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_TILE_COMPONENTS {
            let row = output_index / ABI_INDEXER_TILE_COMPONENTS;
            let col = output_index % ABI_INDEXER_TILE_COMPONENTS;
            let token = tile_t + row;
            let comp = tile_c + col;
            if token < n_tokens as usize && comp < n_comp as usize {
                let mut output = unsafe { ACC_TILE[output_index] } * scale;
                if causal != 0 {
                    let visible = (pos0 + token as u32 + 1) / ratio;
                    if comp as u32 >= visible {
                        output = f32::NEG_INFINITY;
                    }
                }
                unsafe {
                    *scores.get_unchecked_mut(token * n_comp as usize + comp) = output;
                }
            }
            output_index += ABI_INDEXER_WMMA_THREADS as usize;
        }
    }

    #[kernel]
    pub fn abi_indexer_scores_wmma32_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut A_TILE: SharedArray<f16, { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K }, 16> =
            SharedArray::UNINIT;
        static mut B_LO_TILE: SharedArray<
            f16,
            { ABI_INDEXER_WMMA32_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N },
            16,
        > = SharedArray::UNINIT;
        static mut B_HI_TILE: SharedArray<
            f16,
            { ABI_INDEXER_WMMA32_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N },
            16,
        > = SharedArray::UNINIT;
        static mut C_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA32_COMPONENTS },
        > = SharedArray::UNINIT;
        static mut ACC_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA32_COMPONENTS },
        > = SharedArray::UNINIT;

        let tile_c = thread::blockIdx_x() as usize * ABI_INDEXER_WMMA32_COMPONENTS;
        let tile_t = thread::blockIdx_y() as usize * ABI_INDEXER_TILE_TOKENS;
        let tid = thread::threadIdx_x() as usize;
        if tid >= ABI_INDEXER_WMMA32_THREADS as usize {
            return;
        }

        if causal != 0 {
            let tile_end = tile_t as u32 + ABI_INDEXER_TILE_TOKENS as u32;
            let last_token = if tile_end < n_tokens {
                tile_end
            } else {
                n_tokens
            };
            let max_visible = if last_token > tile_t as u32 && ratio != 0 {
                let visible = (pos0 + last_token) / ratio;
                if visible < n_comp {
                    visible
                } else {
                    n_comp
                }
            } else {
                0
            };
            if tile_c as u32 >= max_visible {
                let mut i = tid;
                while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA32_COMPONENTS {
                    let row = i / ABI_INDEXER_WMMA32_COMPONENTS;
                    let col = i % ABI_INDEXER_WMMA32_COMPONENTS;
                    let token = tile_t + row;
                    let comp = tile_c + col;
                    if token < n_tokens as usize && comp < n_comp as usize {
                        unsafe {
                            *scores.get_unchecked_mut(token * n_comp as usize + comp) =
                                f32::NEG_INFINITY;
                        }
                    }
                    i += ABI_INDEXER_WMMA32_THREADS as usize;
                }
                return;
            }
        }

        let mut i = tid;
        while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA32_COMPONENTS {
            unsafe {
                ACC_TILE[i] = 0.0;
            }
            i += ABI_INDEXER_WMMA32_THREADS as usize;
        }
        thread::sync_threads();

        let warp_id = tid >> 5;
        let lane = tid & 31;
        let a_row = (lane & 7) + (lane & 8);
        let a_col = if lane & 16 == 0 { 0 } else { 8 };
        let b_row = (lane & 7) + (lane & 8);
        let mut head = 0_usize;
        while head < ABI_INDEXER_N_HEAD {
            let mut acc_lo = zero_accumulator();
            let mut acc_hi = zero_accumulator();
            let mut k0 = 0_usize;
            while k0 < ABI_INDEXER_HEAD_DIM {
                let mut a_index = tid;
                while a_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K {
                    let row = a_index / ABI_INDEXER_MMA_K;
                    let col = a_index % ABI_INDEXER_MMA_K;
                    let token = tile_t + row;
                    let value = if token < n_tokens as usize {
                        q[(token * ABI_INDEXER_N_HEAD + head) * ABI_INDEXER_HEAD_DIM + k0 + col]
                    } else {
                        0.0
                    };
                    unsafe {
                        A_TILE[a_index] = value as f16;
                    }
                    a_index += ABI_INDEXER_WMMA32_THREADS as usize;
                }

                let mut b_index = tid;
                while b_index < ABI_INDEXER_WMMA32_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N {
                    let warp_tile = b_index / (ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N);
                    let local = b_index % (ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N);
                    let row = local / ABI_INDEXER_MMA_N;
                    let col = local % ABI_INDEXER_MMA_N;
                    let comp_lo = tile_c + warp_tile * ABI_INDEXER_TILE_COMPONENTS + col;
                    let comp_hi =
                        tile_c + warp_tile * ABI_INDEXER_TILE_COMPONENTS + ABI_INDEXER_MMA_N + col;
                    let dimension = k0 + row;
                    let value_lo = if comp_lo < n_comp as usize {
                        index_comp[comp_lo * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    let value_hi = if comp_hi < n_comp as usize {
                        index_comp[comp_hi * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    unsafe {
                        B_LO_TILE[b_index] = value_lo as f16;
                        B_HI_TILE[b_index] = value_hi as f16;
                    }
                    b_index += ABI_INDEXER_WMMA32_THREADS as usize;
                }
                thread::sync_threads();

                let a_ptr = unsafe {
                    (&raw const A_TILE)
                        .cast::<f16>()
                        .add(a_row * ABI_INDEXER_MMA_K + a_col)
                }
                .cast::<u8>();
                let b_base =
                    warp_id * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N + b_row * ABI_INDEXER_MMA_N;
                let b_lo_ptr =
                    unsafe { (&raw const B_LO_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let b_hi_ptr =
                    unsafe { (&raw const B_HI_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                let b_lo_frag = unsafe { load_b_m16n8k16(b_lo_ptr) };
                let b_hi_frag = unsafe { load_b_m16n8k16(b_hi_ptr) };
                acc_lo = unsafe { mma_m16n8k16_f32_f16(acc_lo, a_frag, b_lo_frag) };
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                acc_hi = unsafe { mma_m16n8k16_f32_f16(acc_hi, a_frag, b_hi_frag) };
                thread::sync_threads();
                k0 += ABI_INDEXER_MMA_K;
            }

            let group_id = lane >> 2;
            let thread_in_group = lane & 3;
            let col_base = warp_id * ABI_INDEXER_TILE_COMPONENTS + thread_in_group * 2;
            unsafe {
                C_TILE[group_id * ABI_INDEXER_WMMA32_COMPONENTS + col_base] = acc_lo.x();
                C_TILE[group_id * ABI_INDEXER_WMMA32_COMPONENTS + col_base + 1] = acc_lo.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA32_COMPONENTS + col_base] = acc_lo.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA32_COMPONENTS + col_base + 1] = acc_lo.w();
                C_TILE[group_id * ABI_INDEXER_WMMA32_COMPONENTS + ABI_INDEXER_MMA_N + col_base] =
                    acc_hi.x();
                C_TILE
                    [group_id * ABI_INDEXER_WMMA32_COMPONENTS + ABI_INDEXER_MMA_N + col_base + 1] =
                    acc_hi.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA32_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base] = acc_hi.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA32_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base
                    + 1] = acc_hi.w();
            }
            thread::sync_threads();

            let mut output_index = tid;
            while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA32_COMPONENTS {
                let row = output_index / ABI_INDEXER_WMMA32_COMPONENTS;
                let col = output_index % ABI_INDEXER_WMMA32_COMPONENTS;
                let token = tile_t + row;
                let comp = tile_c + col;
                if token < n_tokens as usize && comp < n_comp as usize {
                    let value = unsafe { C_TILE[output_index] };
                    let positive = if (value.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || value <= 0.0
                    {
                        0.0
                    } else {
                        value
                    };
                    unsafe {
                        ACC_TILE[output_index] +=
                            positive * weights[token * ABI_INDEXER_N_HEAD + head];
                    }
                }
                output_index += ABI_INDEXER_WMMA32_THREADS as usize;
            }
            thread::sync_threads();
            head += 1;
        }

        let mut output_index = tid;
        while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA32_COMPONENTS {
            let row = output_index / ABI_INDEXER_WMMA32_COMPONENTS;
            let col = output_index % ABI_INDEXER_WMMA32_COMPONENTS;
            let token = tile_t + row;
            let comp = tile_c + col;
            if token < n_tokens as usize && comp < n_comp as usize {
                let mut output = unsafe { ACC_TILE[output_index] } * scale;
                if causal != 0 {
                    let visible = (pos0 + token as u32 + 1) / ratio;
                    if comp as u32 >= visible {
                        output = f32::NEG_INFINITY;
                    }
                }
                unsafe {
                    *scores.get_unchecked_mut(token * n_comp as usize + comp) = output;
                }
            }
            output_index += ABI_INDEXER_WMMA32_THREADS as usize;
        }
    }

    #[kernel]
    pub fn abi_indexer_scores_wmma64_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut A_TILE: SharedArray<f16, { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K }, 16> =
            SharedArray::UNINIT;
        static mut B_LO_TILE: SharedArray<
            f16,
            { ABI_INDEXER_WMMA64_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N },
            16,
        > = SharedArray::UNINIT;
        static mut B_HI_TILE: SharedArray<
            f16,
            { ABI_INDEXER_WMMA64_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N },
            16,
        > = SharedArray::UNINIT;
        static mut C_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA64_COMPONENTS },
        > = SharedArray::UNINIT;
        static mut ACC_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA64_COMPONENTS },
        > = SharedArray::UNINIT;

        let tile_c = thread::blockIdx_x() as usize * ABI_INDEXER_WMMA64_COMPONENTS;
        let tile_t = thread::blockIdx_y() as usize * ABI_INDEXER_TILE_TOKENS;
        let tid = thread::threadIdx_x() as usize;
        if tid >= ABI_INDEXER_WMMA64_THREADS as usize {
            return;
        }

        if causal != 0 {
            let tile_end = tile_t as u32 + ABI_INDEXER_TILE_TOKENS as u32;
            let last_token = if tile_end < n_tokens {
                tile_end
            } else {
                n_tokens
            };
            let max_visible = if last_token > tile_t as u32 && ratio != 0 {
                let visible = (pos0 + last_token) / ratio;
                if visible < n_comp {
                    visible
                } else {
                    n_comp
                }
            } else {
                0
            };
            if tile_c as u32 >= max_visible {
                let mut i = tid;
                while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA64_COMPONENTS {
                    let row = i / ABI_INDEXER_WMMA64_COMPONENTS;
                    let col = i % ABI_INDEXER_WMMA64_COMPONENTS;
                    let token = tile_t + row;
                    let comp = tile_c + col;
                    if token < n_tokens as usize && comp < n_comp as usize {
                        unsafe {
                            *scores.get_unchecked_mut(token * n_comp as usize + comp) =
                                f32::NEG_INFINITY;
                        }
                    }
                    i += ABI_INDEXER_WMMA64_THREADS as usize;
                }
                return;
            }
        }

        let mut i = tid;
        while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA64_COMPONENTS {
            unsafe {
                ACC_TILE[i] = 0.0;
            }
            i += ABI_INDEXER_WMMA64_THREADS as usize;
        }
        thread::sync_threads();

        let warp_id = tid >> 5;
        let lane = tid & 31;
        let a_row = (lane & 7) + (lane & 8);
        let a_col = if lane & 16 == 0 { 0 } else { 8 };
        let b_row = (lane & 7) + (lane & 8);
        let mut head = 0_usize;
        while head < ABI_INDEXER_N_HEAD {
            let mut acc_lo = zero_accumulator();
            let mut acc_hi = zero_accumulator();
            let mut k0 = 0_usize;
            while k0 < ABI_INDEXER_HEAD_DIM {
                let mut a_index = tid;
                while a_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K {
                    let row = a_index / ABI_INDEXER_MMA_K;
                    let col = a_index % ABI_INDEXER_MMA_K;
                    let token = tile_t + row;
                    let value = if token < n_tokens as usize {
                        q[(token * ABI_INDEXER_N_HEAD + head) * ABI_INDEXER_HEAD_DIM + k0 + col]
                    } else {
                        0.0
                    };
                    unsafe {
                        A_TILE[a_index] = value as f16;
                    }
                    a_index += ABI_INDEXER_WMMA64_THREADS as usize;
                }

                let mut b_index = tid;
                while b_index < ABI_INDEXER_WMMA64_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N {
                    let warp_tile = b_index / (ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N);
                    let local = b_index % (ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N);
                    let row = local / ABI_INDEXER_MMA_N;
                    let col = local % ABI_INDEXER_MMA_N;
                    let comp_lo = tile_c + warp_tile * ABI_INDEXER_TILE_COMPONENTS + col;
                    let comp_hi =
                        tile_c + warp_tile * ABI_INDEXER_TILE_COMPONENTS + ABI_INDEXER_MMA_N + col;
                    let dimension = k0 + row;
                    let value_lo = if comp_lo < n_comp as usize {
                        index_comp[comp_lo * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    let value_hi = if comp_hi < n_comp as usize {
                        index_comp[comp_hi * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    unsafe {
                        B_LO_TILE[b_index] = value_lo as f16;
                        B_HI_TILE[b_index] = value_hi as f16;
                    }
                    b_index += ABI_INDEXER_WMMA64_THREADS as usize;
                }
                thread::sync_threads();

                let a_ptr = unsafe {
                    (&raw const A_TILE)
                        .cast::<f16>()
                        .add(a_row * ABI_INDEXER_MMA_K + a_col)
                }
                .cast::<u8>();
                let b_base =
                    warp_id * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N + b_row * ABI_INDEXER_MMA_N;
                let b_lo_ptr =
                    unsafe { (&raw const B_LO_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let b_hi_ptr =
                    unsafe { (&raw const B_HI_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                let b_lo_frag = unsafe { load_b_m16n8k16(b_lo_ptr) };
                let b_hi_frag = unsafe { load_b_m16n8k16(b_hi_ptr) };
                acc_lo = unsafe { mma_m16n8k16_f32_f16(acc_lo, a_frag, b_lo_frag) };
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                acc_hi = unsafe { mma_m16n8k16_f32_f16(acc_hi, a_frag, b_hi_frag) };
                thread::sync_threads();
                k0 += ABI_INDEXER_MMA_K;
            }

            let group_id = lane >> 2;
            let thread_in_group = lane & 3;
            let col_base = warp_id * ABI_INDEXER_TILE_COMPONENTS + thread_in_group * 2;
            unsafe {
                C_TILE[group_id * ABI_INDEXER_WMMA64_COMPONENTS + col_base] = acc_lo.x();
                C_TILE[group_id * ABI_INDEXER_WMMA64_COMPONENTS + col_base + 1] = acc_lo.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA64_COMPONENTS + col_base] = acc_lo.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA64_COMPONENTS + col_base + 1] = acc_lo.w();
                C_TILE[group_id * ABI_INDEXER_WMMA64_COMPONENTS + ABI_INDEXER_MMA_N + col_base] =
                    acc_hi.x();
                C_TILE
                    [group_id * ABI_INDEXER_WMMA64_COMPONENTS + ABI_INDEXER_MMA_N + col_base + 1] =
                    acc_hi.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA64_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base] = acc_hi.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA64_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base
                    + 1] = acc_hi.w();
            }
            thread::sync_threads();

            let mut output_index = tid;
            while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA64_COMPONENTS {
                let row = output_index / ABI_INDEXER_WMMA64_COMPONENTS;
                let col = output_index % ABI_INDEXER_WMMA64_COMPONENTS;
                let token = tile_t + row;
                let comp = tile_c + col;
                if token < n_tokens as usize && comp < n_comp as usize {
                    let value = unsafe { C_TILE[output_index] };
                    let positive = if (value.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || value <= 0.0
                    {
                        0.0
                    } else {
                        value
                    };
                    unsafe {
                        ACC_TILE[output_index] +=
                            positive * weights[token * ABI_INDEXER_N_HEAD + head];
                    }
                }
                output_index += ABI_INDEXER_WMMA64_THREADS as usize;
            }
            thread::sync_threads();
            head += 1;
        }

        let mut output_index = tid;
        while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA64_COMPONENTS {
            let row = output_index / ABI_INDEXER_WMMA64_COMPONENTS;
            let col = output_index % ABI_INDEXER_WMMA64_COMPONENTS;
            let token = tile_t + row;
            let comp = tile_c + col;
            if token < n_tokens as usize && comp < n_comp as usize {
                let mut output = unsafe { ACC_TILE[output_index] } * scale;
                if causal != 0 {
                    let visible = (pos0 + token as u32 + 1) / ratio;
                    if comp as u32 >= visible {
                        output = f32::NEG_INFINITY;
                    }
                }
                unsafe {
                    *scores.get_unchecked_mut(token * n_comp as usize + comp) = output;
                }
            }
            output_index += ABI_INDEXER_WMMA64_THREADS as usize;
        }
    }

    #[kernel]
    pub fn abi_indexer_scores_wmma128_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut A_TILE: SharedArray<f16, { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K }, 16> =
            SharedArray::UNINIT;
        static mut B_LO_TILE: SharedArray<
            f16,
            { ABI_INDEXER_WMMA128_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N },
            16,
        > = SharedArray::UNINIT;
        static mut B_HI_TILE: SharedArray<
            f16,
            { ABI_INDEXER_WMMA128_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N },
            16,
        > = SharedArray::UNINIT;
        static mut C_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA128_COMPONENTS },
        > = SharedArray::UNINIT;
        static mut ACC_TILE: SharedArray<
            f32,
            { ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA128_COMPONENTS },
        > = SharedArray::UNINIT;

        let tile_c = thread::blockIdx_x() as usize * ABI_INDEXER_WMMA128_COMPONENTS;
        let tile_t = thread::blockIdx_y() as usize * ABI_INDEXER_TILE_TOKENS;
        let tid = thread::threadIdx_x() as usize;
        if tid >= ABI_INDEXER_WMMA128_THREADS as usize {
            return;
        }

        if causal != 0 {
            let tile_end = tile_t as u32 + ABI_INDEXER_TILE_TOKENS as u32;
            let last_token = if tile_end < n_tokens {
                tile_end
            } else {
                n_tokens
            };
            let max_visible = if last_token > tile_t as u32 && ratio != 0 {
                let visible = (pos0 + last_token) / ratio;
                if visible < n_comp {
                    visible
                } else {
                    n_comp
                }
            } else {
                0
            };
            if tile_c as u32 >= max_visible {
                let mut i = tid;
                while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA128_COMPONENTS {
                    let row = i / ABI_INDEXER_WMMA128_COMPONENTS;
                    let col = i % ABI_INDEXER_WMMA128_COMPONENTS;
                    let token = tile_t + row;
                    let comp = tile_c + col;
                    if token < n_tokens as usize && comp < n_comp as usize {
                        unsafe {
                            *scores.get_unchecked_mut(token * n_comp as usize + comp) =
                                f32::NEG_INFINITY;
                        }
                    }
                    i += ABI_INDEXER_WMMA128_THREADS as usize;
                }
                return;
            }
        }

        let mut i = tid;
        while i < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA128_COMPONENTS {
            unsafe {
                ACC_TILE[i] = 0.0;
            }
            i += ABI_INDEXER_WMMA128_THREADS as usize;
        }
        thread::sync_threads();

        let warp_id = tid >> 5;
        let lane = tid & 31;
        let a_row = (lane & 7) + (lane & 8);
        let a_col = if lane & 16 == 0 { 0 } else { 8 };
        let b_row = (lane & 7) + (lane & 8);
        let mut head = 0_usize;
        while head < ABI_INDEXER_N_HEAD {
            let mut acc_lo = zero_accumulator();
            let mut acc_hi = zero_accumulator();
            let mut k0 = 0_usize;
            while k0 < ABI_INDEXER_HEAD_DIM {
                let mut a_index = tid;
                while a_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_MMA_K {
                    let row = a_index / ABI_INDEXER_MMA_K;
                    let col = a_index % ABI_INDEXER_MMA_K;
                    let token = tile_t + row;
                    let value = if token < n_tokens as usize {
                        q[(token * ABI_INDEXER_N_HEAD + head) * ABI_INDEXER_HEAD_DIM + k0 + col]
                    } else {
                        0.0
                    };
                    unsafe {
                        A_TILE[a_index] = value as f16;
                    }
                    a_index += ABI_INDEXER_WMMA128_THREADS as usize;
                }

                let mut b_index = tid;
                while b_index < ABI_INDEXER_WMMA128_WARPS * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N {
                    let warp_tile = b_index / (ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N);
                    let local = b_index % (ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N);
                    let row = local / ABI_INDEXER_MMA_N;
                    let col = local % ABI_INDEXER_MMA_N;
                    let comp_lo = tile_c + warp_tile * ABI_INDEXER_TILE_COMPONENTS + col;
                    let comp_hi =
                        tile_c + warp_tile * ABI_INDEXER_TILE_COMPONENTS + ABI_INDEXER_MMA_N + col;
                    let dimension = k0 + row;
                    let value_lo = if comp_lo < n_comp as usize {
                        index_comp[comp_lo * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    let value_hi = if comp_hi < n_comp as usize {
                        index_comp[comp_hi * ABI_INDEXER_HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    unsafe {
                        B_LO_TILE[b_index] = value_lo as f16;
                        B_HI_TILE[b_index] = value_hi as f16;
                    }
                    b_index += ABI_INDEXER_WMMA128_THREADS as usize;
                }
                thread::sync_threads();

                let a_ptr = unsafe {
                    (&raw const A_TILE)
                        .cast::<f16>()
                        .add(a_row * ABI_INDEXER_MMA_K + a_col)
                }
                .cast::<u8>();
                let b_base =
                    warp_id * ABI_INDEXER_MMA_K * ABI_INDEXER_MMA_N + b_row * ABI_INDEXER_MMA_N;
                let b_lo_ptr =
                    unsafe { (&raw const B_LO_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let b_hi_ptr =
                    unsafe { (&raw const B_HI_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                let b_lo_frag = unsafe { load_b_m16n8k16(b_lo_ptr) };
                let b_hi_frag = unsafe { load_b_m16n8k16(b_hi_ptr) };
                acc_lo = unsafe { mma_m16n8k16_f32_f16(acc_lo, a_frag, b_lo_frag) };
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                acc_hi = unsafe { mma_m16n8k16_f32_f16(acc_hi, a_frag, b_hi_frag) };
                thread::sync_threads();
                k0 += ABI_INDEXER_MMA_K;
            }

            let group_id = lane >> 2;
            let thread_in_group = lane & 3;
            let col_base = warp_id * ABI_INDEXER_TILE_COMPONENTS + thread_in_group * 2;
            unsafe {
                C_TILE[group_id * ABI_INDEXER_WMMA128_COMPONENTS + col_base] = acc_lo.x();
                C_TILE[group_id * ABI_INDEXER_WMMA128_COMPONENTS + col_base + 1] = acc_lo.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA128_COMPONENTS + col_base] = acc_lo.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA128_COMPONENTS + col_base + 1] = acc_lo.w();
                C_TILE[group_id * ABI_INDEXER_WMMA128_COMPONENTS + ABI_INDEXER_MMA_N + col_base] =
                    acc_hi.x();
                C_TILE[group_id * ABI_INDEXER_WMMA128_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base
                    + 1] = acc_hi.y();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA128_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base] = acc_hi.z();
                C_TILE[(group_id + 8) * ABI_INDEXER_WMMA128_COMPONENTS
                    + ABI_INDEXER_MMA_N
                    + col_base
                    + 1] = acc_hi.w();
            }
            thread::sync_threads();

            let mut output_index = tid;
            while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA128_COMPONENTS {
                let row = output_index / ABI_INDEXER_WMMA128_COMPONENTS;
                let col = output_index % ABI_INDEXER_WMMA128_COMPONENTS;
                let token = tile_t + row;
                let comp = tile_c + col;
                if token < n_tokens as usize && comp < n_comp as usize {
                    let value = unsafe { C_TILE[output_index] };
                    let positive = if (value.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || value <= 0.0
                    {
                        0.0
                    } else {
                        value
                    };
                    unsafe {
                        ACC_TILE[output_index] +=
                            positive * weights[token * ABI_INDEXER_N_HEAD + head];
                    }
                }
                output_index += ABI_INDEXER_WMMA128_THREADS as usize;
            }
            thread::sync_threads();
            head += 1;
        }

        let mut output_index = tid;
        while output_index < ABI_INDEXER_TILE_TOKENS * ABI_INDEXER_WMMA128_COMPONENTS {
            let row = output_index / ABI_INDEXER_WMMA128_COMPONENTS;
            let col = output_index % ABI_INDEXER_WMMA128_COMPONENTS;
            let token = tile_t + row;
            let comp = tile_c + col;
            if token < n_tokens as usize && comp < n_comp as usize {
                let mut output = unsafe { ACC_TILE[output_index] } * scale;
                if causal != 0 {
                    let visible = (pos0 + token as u32 + 1) / ratio;
                    if comp as u32 >= visible {
                        output = f32::NEG_INFINITY;
                    }
                }
                unsafe {
                    *scores.get_unchecked_mut(token * n_comp as usize + comp) = output;
                }
            }
            output_index += ABI_INDEXER_WMMA128_THREADS as usize;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        let token = thread::blockIdx_x();
        if token >= n_tokens || thread::threadIdx_x() != 0 {
            return;
        }
        let score_base = token as usize * n_comp as usize;
        let selected_base = token as usize * top_k as usize;
        let mut k = 0;
        while k < top_k {
            unsafe {
                *selected.get_unchecked_mut(selected_base + k as usize) = 0;
            }
            k += 1;
        }
        let mut comp = 0;
        while comp < n_comp {
            let value = scores[score_base + comp as usize];
            k = 0;
            while k < top_k {
                let selected_index =
                    unsafe { *selected.get_unchecked_mut(selected_base + k as usize) };
                if k >= comp || value > scores[score_base + selected_index as usize] {
                    let mut move_index = top_k - 1;
                    while move_index > k {
                        let previous = unsafe {
                            *selected.get_unchecked_mut(selected_base + move_index as usize - 1)
                        };
                        unsafe {
                            *selected.get_unchecked_mut(selected_base + move_index as usize) =
                                previous;
                        }
                        move_index -= 1;
                    }
                    unsafe {
                        *selected.get_unchecked_mut(selected_base + k as usize) = comp;
                    }
                    break;
                }
                k += 1;
            }
            comp += 1;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_1024_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_1024_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, ABI_INDEXER_TOPK_1024_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= ABI_INDEXER_TOPK_THREADS {
            return;
        }
        let index = tid as usize;
        if tid < n_comp {
            unsafe {
                VALUES[index] = scores[token as usize * n_comp as usize + index];
                INDICES[index] = tid;
            }
        } else {
            unsafe {
                VALUES[index] = f32::NEG_INFINITY;
                INDICES[index] = u32::MAX;
            }
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_1024_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                let other = tid ^ j;
                if other > tid && other < ABI_INDEXER_TOPK_1024_SORT_N as u32 {
                    let other_index = other as usize;
                    let av = unsafe { VALUES[index] };
                    let bv = unsafe { VALUES[other_index] };
                    let ai = unsafe { INDICES[index] };
                    let bi = unsafe { INDICES[other_index] };
                    let desc_half = (tid & k) == 0;
                    let b_better = bv > av || (bv == av && bi < ai);
                    let a_better = av > bv || (av == bv && ai < bi);
                    let swap = if desc_half { b_better } else { a_better };
                    if swap {
                        unsafe {
                            VALUES[index] = bv;
                            INDICES[index] = bi;
                            VALUES[other_index] = av;
                            INDICES[other_index] = ai;
                        }
                    }
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        if tid < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + index) =
                    INDICES[index];
            }
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_pow2_2048_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_2048_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, ABI_INDEXER_TOPK_2048_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < ABI_INDEXER_TOPK_2048_SORT_N as u32 {
            let index = i as usize;
            if i < n_comp {
                unsafe {
                    VALUES[index] = scores[token as usize * n_comp as usize + index];
                    INDICES[index] = i;
                }
            } else {
                unsafe {
                    VALUES[index] = f32::NEG_INFINITY;
                    INDICES[index] = u32::MAX;
                }
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_2048_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < ABI_INDEXER_TOPK_2048_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_2048_SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }
        i = tid;
        while i < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + i as usize) =
                    INDICES[i as usize];
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let index = i as usize;
            if i < n_comp {
                unsafe {
                    VALUES[index] = scores[token as usize * n_comp as usize + index];
                    INDICES[index] = i;
                }
            } else {
                unsafe {
                    VALUES[index] = f32::NEG_INFINITY;
                    INDICES[index] = u32::MAX;
                }
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }
        i = tid;
        while i < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + i as usize) =
                    INDICES[i as usize];
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_pow2_u16_8192_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_8192_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u16, ABI_INDEXER_TOPK_8192_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < ABI_INDEXER_TOPK_8192_SORT_N as u32 {
            let index = i as usize;
            if i < n_comp {
                unsafe {
                    VALUES[index] = scores[token as usize * n_comp as usize + index];
                    INDICES[index] = i as u16;
                }
            } else {
                unsafe {
                    VALUES[index] = f32::NEG_INFINITY;
                    INDICES[index] = u16::MAX;
                }
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_8192_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < ABI_INDEXER_TOPK_8192_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_8192_SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] } as u32;
                        let bi = unsafe { INDICES[other_index] } as u32;
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi as u16;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai as u16;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }
        i = tid;
        while i < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + i as usize) =
                    INDICES[i as usize] as u32;
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_8192_packed_key_equivalent_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        let keys = DynamicSharedArray::<u64>::get();
        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= ABI_INDEXER_TOPK_PACKED_THREADS {
            return;
        }
        let mut item = 0_u32;
        while item < ABI_INDEXER_TOPK_PACKED_ITEMS_PER_THREAD {
            let i = tid * ABI_INDEXER_TOPK_PACKED_ITEMS_PER_THREAD + item;
            let key = if i < n_comp {
                let value = scores[token as usize * n_comp as usize + i as usize];
                let bits = value.to_bits();
                let ordered = if (bits & 0x8000_0000) != 0 {
                    !bits
                } else {
                    bits ^ 0x8000_0000
                };
                (ordered as u64) << 32 | (u32::MAX - i) as u64
            } else {
                ABI_INDEXER_TOPK_EMPTY_KEY
            };
            unsafe {
                *keys.add(i as usize) = key;
            }
            item += 1;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_8192_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < ABI_INDEXER_TOPK_8192_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_8192_SORT_N as u32 {
                        let left = unsafe { *keys.add(i as usize) };
                        let right = unsafe { *keys.add(other as usize) };
                        let descending = (i & k) == 0;
                        let swap = if descending {
                            right > left
                        } else {
                            left > right
                        };
                        if swap {
                            unsafe {
                                *keys.add(i as usize) = right;
                                *keys.add(other as usize) = left;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_PACKED_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        if tid < top_k {
            let key = unsafe { *keys.add(tid as usize) };
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + tid as usize) =
                    u32::MAX - key as u32;
            }
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_chunk_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        candidate_stride: u32,
        scores: &[f32],
        mut scratch: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let chunk_start = chunk * ABI_INDEXER_TOPK_4096_SORT_N as u32;
        if chunk_start >= n_comp {
            return;
        }
        let remaining = n_comp - chunk_start;
        let chunk_n = if remaining < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            remaining
        } else {
            ABI_INDEXER_TOPK_4096_SORT_N as u32
        };
        let mut i = tid;
        while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let index = i as usize;
            if i < chunk_n {
                unsafe {
                    VALUES[index] =
                        scores[token as usize * n_comp as usize + (chunk_start + i) as usize];
                    INDICES[index] = chunk_start + i;
                }
            } else {
                unsafe {
                    VALUES[index] = f32::NEG_INFINITY;
                    INDICES[index] = u32::MAX;
                }
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        i = tid;
        while i < top_k {
            let out = token as usize * candidate_stride as usize
                + chunk as usize * top_k as usize
                + i as usize;
            unsafe {
                *scratch.get_unchecked_mut(out) = INDICES[i as usize];
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_tree_merge_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        n_sets: u32,
        merge_group: u32,
        candidate_offset: u32,
        candidate_stride: u32,
        out_offset: u32,
        out_stride: u32,
        scores: &[f32],
        mut scratch: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let group = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let set0 = group * merge_group;
        if set0 >= n_sets {
            return;
        }
        let remaining = n_sets - set0;
        let set_count = if remaining < merge_group {
            remaining
        } else {
            merge_group
        };
        let candidate_count = set_count * top_k;
        let mut i = tid;
        while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let mut index = u32::MAX;
            let mut value = f32::NEG_INFINITY;
            if i < candidate_count {
                let source = candidate_offset as usize
                    + token as usize * candidate_stride as usize
                    + set0 as usize * top_k as usize
                    + i as usize;
                index = unsafe { *scratch.get_unchecked_mut(source) };
                if index < n_comp {
                    value = scores[token as usize * n_comp as usize + index as usize];
                }
            }
            unsafe {
                VALUES[i as usize] = value;
                INDICES[i as usize] = index;
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        i = tid;
        while i < top_k {
            let out = out_offset as usize
                + token as usize * out_stride as usize
                + group as usize * top_k as usize
                + i as usize;
            unsafe {
                *scratch.get_unchecked_mut(out) = INDICES[i as usize];
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
    }

    #[kernel]
    pub fn abi_indexer_topk_merge_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        candidate_offset: u32,
        candidate_count: u32,
        candidate_stride: u32,
        candidates: &[u32],
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, ABI_INDEXER_TOPK_4096_SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let mut index = u32::MAX;
            let mut value = f32::NEG_INFINITY;
            if i < candidate_count {
                let source = candidate_offset as usize
                    + token as usize * candidate_stride as usize
                    + i as usize;
                index = candidates[source];
                if index < n_comp {
                    value = scores[token as usize * n_comp as usize + index as usize];
                }
            }
            unsafe {
                VALUES[i as usize] = value;
                INDICES[i as usize] = index;
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= ABI_INDEXER_TOPK_4096_SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < ABI_INDEXER_TOPK_4096_SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
                            }
                        }
                    }
                    i += ABI_INDEXER_TOPK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        i = tid;
        while i < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + i as usize) =
                    INDICES[i as usize];
            }
            i += ABI_INDEXER_TOPK_THREADS;
        }
    }

    #[kernel]
    pub fn abi_topk_mask_kernel(
        count: u64,
        n_comp: u32,
        top_k: u32,
        topk: &[u32],
        mut mask: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let gid = index.get();
        if gid as u64 >= count {
            return;
        }
        let token = gid / n_comp as usize;
        let comp = gid - token * n_comp as usize;
        let mut value = f32::NEG_INFINITY;
        let mut k = 0;
        while k < top_k {
            if topk[token * top_k as usize + k as usize] == comp as u32 {
                value = 0.0;
                break;
            }
            k += 1;
        }
        if let Some(element) = mask.get_mut(index) {
            *element = value;
        }
    }

    fn abi_e2m1fn_value(value: i32) -> f32 {
        match value & 7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        }
    }

    fn abi_e2m1fn_dequant(value: f32) -> f32 {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let mut magnitude = abi_absolute(value);
        if magnitude > 6.0 {
            magnitude = 6.0;
        }
        let mut best = 0_i32;
        let mut best_diff = magnitude;
        let mut candidate = 1_i32;
        while candidate < 8 {
            let diff = abi_absolute(magnitude - abi_e2m1fn_value(candidate));
            if diff < best_diff || (diff == best_diff && candidate & 1 == 0 && best & 1 != 0) {
                best = candidate;
                best_diff = diff;
            }
            candidate += 1;
        }
        sign * abi_e2m1fn_value(best)
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

        let mean_square = unsafe { PARTIAL[0] } / n as f32 + eps;
        let scale = unsafe { __nv_rsqrtf(mean_square) };
        i = tid;
        while i < n {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale * weight[i];
            }
            i += nth;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn abi_dsv4_qkv_rms_norm_rows_kernel(
        q_n: u32,
        kv_n: u32,
        rows: u32,
        eps: f32,
        q: &[f32],
        q_weight: &[f32],
        mut q_out: DisjointSlice<f32>,
        kv: &[f32],
        kv_weight: &[f32],
        mut kv_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let which = thread::blockIdx_y();
        if row >= rows || which > 1 {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let n = (if which == 0 { q_n } else { kv_n }) as usize;
        let base = row as usize * n;

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            let value = if which == 0 {
                q[base + i]
            } else {
                kv[base + i]
            };
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

        let mean_square = unsafe { PARTIAL[0] } / n as f32 + eps;
        let scale = unsafe { __nv_rsqrtf(mean_square) };
        i = tid;
        while i < n {
            unsafe {
                if which == 0 {
                    *q_out.get_unchecked_mut(base + i) = q[base + i] * scale * q_weight[i];
                } else {
                    *kv_out.get_unchecked_mut(base + i) = kv[base + i] * scale * kv_weight[i];
                }
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn abi_dequant_q8_0_to_f16_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
        weights: &[u8],
        mut output: DisjointSlice<f16>,
    ) {
        let index = thread::index_1d();
        let gid = index.get() as u64;
        let count = in_dim * out_dim;
        if gid >= count {
            return;
        }
        let row = gid / in_dim;
        let column = gid - row * in_dim;
        let block = column / 32;
        let lane = column - block * 32;
        let base = ((row * blocks + block) * 34) as usize;
        let scale_bits = weights[base] as u16 | ((weights[base + 1] as u16) << 8);
        let scale = f16::from_bits(scale_bits) as f32;
        let value = weights[base + 2 + lane as usize] as i8 as f32;
        if let Some(element) = output.get_mut(index) {
            *element = (scale * value) as f16;
        }
    }

    #[kernel]
    pub fn abi_dequant_q8_0_to_f32_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
        weights: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let gid = index.get() as u64;
        let count = in_dim * out_dim;
        if gid >= count {
            return;
        }
        let row = gid / in_dim;
        let column = gid - row * in_dim;
        let block = column / 32;
        let lane = column - block * 32;
        let base = ((row * blocks + block) * 34) as usize;
        let scale_bits = weights[base] as u16 | ((weights[base + 1] as u16) << 8);
        let scale = f16::from_bits(scale_bits) as f32;
        let value = weights[base + 2 + lane as usize] as i8 as f32;
        if let Some(element) = output.get_mut(index) {
            *element = scale * value;
        }
    }

    #[kernel]
    pub fn abi_quantize_q8_0_f32_kernel(
        in_dim: u64,
        blocks: u64,
        n_tok: u64,
        x: &[f32],
        mut xq: DisjointSlice<i8>,
        mut xscale: DisjointSlice<f32>,
    ) {
        static mut VALUES: SharedArray<f32, 32> = SharedArray::UNINIT;

        let block = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        let lane = thread::threadIdx_x() as usize;
        if block >= blocks || token >= n_tok {
            return;
        }
        let start = block * 32;
        let remaining = in_dim - start;
        let count = if remaining < 32 { remaining } else { 32 } as usize;
        let input_base = token as usize * in_dim as usize + start as usize;
        let value = if lane < count {
            x[input_base + lane]
        } else {
            0.0
        };
        let magnitude = if value < 0.0 { -value } else { value };
        unsafe {
            VALUES[lane] = magnitude;
        }
        thread::sync_threads();

        let mut stride = 16;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    if VALUES[lane + stride] > VALUES[lane] {
                        VALUES[lane] = VALUES[lane + stride];
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let scale = unsafe { VALUES[0] } / 127.0;
        let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        let output_base = (token * blocks + block) as usize * 32;
        if lane == 0 {
            unsafe {
                *xscale.get_unchecked_mut((token * blocks + block) as usize) = scale;
            }
        }
        let quantized = if lane < count {
            clamp_q8(round_ties_even(value * inverse))
        } else {
            0
        };
        unsafe {
            *xq.get_unchecked_mut(output_base + lane) = quantized;
        }
    }

    #[kernel]
    pub fn abi_grouped_q8_0_a_preq_warp8_kernel(
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        n_tokens: u32,
        blocks: u64,
        use_dp4a: u32,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut low: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64 * 8 + (thread::threadIdx_x() >> 5) as u64;
        let token = thread::blockIdx_y() as u64;
        let lane = (thread::threadIdx_x() & 31) as u64;
        let low_dim = u64::from(n_groups) * rank;
        if row >= low_dim || token >= u64::from(n_tokens) {
            return;
        }
        let group = row / rank;
        let row_in_group = row - group * rank;
        let xrow = token * u64::from(n_groups) + group;
        let mut acc = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let remaining = group_dim - block * 32;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = (((group * rank + row_in_group) * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = ((xrow * blocks + block) * 32) as usize;
            let dot = q8_dot(weights, weight_base, xq, xq_base, count, use_dp4a != 0);
            acc += weight_scale * xscale[(xrow * blocks + block) as usize] * dot as f32;
            block += 32;
        }
        let mut offset = 16_u32;
        while offset > 0 {
            acc += warp::shuffle_down_f32(acc, offset);
            offset >>= 1;
        }
        if lane == 0 {
            unsafe {
                *low.get_unchecked_mut(token as usize * low_dim as usize + row as usize) = acc;
            }
        }
    }

    #[kernel]
    pub fn abi_attention_pack_group_heads_f16_kernel(
        n_tokens: u32,
        n_groups: u32,
        group_dim: u64,
        heads: &[f32],
        mut packed: DisjointSlice<f16>,
    ) {
        let index = thread::index_1d().get() as u64;
        let count = u64::from(n_groups) * u64::from(n_tokens) * group_dim;
        if index >= count {
            return;
        }
        let dimension = index % group_dim;
        let quotient = index / group_dim;
        let token = quotient % u64::from(n_tokens);
        let group = quotient / u64::from(n_tokens);
        let source = (token * u64::from(n_groups) + group) * group_dim + dimension;
        unsafe {
            *packed.get_unchecked_mut(index as usize) = heads[source as usize] as f16;
        }
    }

    #[kernel]
    pub fn abi_f16_to_f32_kernel(count: u64, input: &[f16], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d().get() as u64;
        if index < count {
            unsafe {
                *output.get_unchecked_mut(index as usize) = input[index as usize] as f32;
            }
        }
    }

    #[kernel]
    pub fn abi_attention_expand_group_weights_sgemm_kernel(
        n_groups: u32,
        rank: u64,
        group_dim: u64,
        weights: &[f16],
        mut transposed: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u64;
        let count = u64::from(n_groups) * rank * group_dim;
        if index >= count {
            return;
        }
        let dimension = index % group_dim;
        let quotient = index / group_dim;
        let output_row = quotient % rank;
        let group = quotient / rank;
        let destination = (group * group_dim + dimension) * rank + output_row;
        unsafe {
            *transposed.get_unchecked_mut(destination as usize) = weights[index as usize] as f32;
        }
    }

    #[kernel]
    pub fn abi_attention_unpack_group_low_kernel(
        n_tokens: u32,
        n_groups: u32,
        rank: u64,
        packed: &[f32],
        mut low: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u64;
        let count = u64::from(n_groups) * u64::from(n_tokens) * rank;
        if index >= count {
            return;
        }
        let output_rank = index % rank;
        let quotient = index / rank;
        let token = quotient % u64::from(n_tokens);
        let group = quotient / u64::from(n_tokens);
        let low_dim = u64::from(n_groups) * rank;
        let destination = token * low_dim + group * rank + output_rank;
        unsafe {
            *low.get_unchecked_mut(destination as usize) = packed[index as usize];
        }
    }

    #[kernel]
    pub fn abi_matmul_q8_0_preq_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        blocks: u64,
        use_dp4a: u32,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        let tid = thread::threadIdx_x() as usize;
        if row >= out_dim || token >= n_tok {
            return;
        }
        let mut acc = 0.0_f32;
        let mut block = tid as u64;
        while block < blocks {
            let remaining = in_dim - block * 32;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = ((token * blocks + block) * 32) as usize;
            let dot = q8_dot(weights, weight_base, xq, xq_base, count, use_dp4a != 0);
            acc += weight_scale * xscale[(token * blocks + block) as usize] * dot as f32;
            block += 256;
        }
        unsafe {
            PARTIAL[tid] = acc;
        }
        thread::sync_threads();
        let mut stride = 128_usize;
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
    pub fn abi_embed_token_hc_kernel(
        token: u32,
        n_embd: u32,
        count: u64,
        weights: &[f16],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if (i as u64) < count {
            let dimension = i % n_embd as usize;
            if let Some(element) = out.get_mut(index) {
                *element = weights[token as usize * n_embd as usize + dimension] as f32;
            }
        }
    }

    #[kernel]
    pub fn abi_embed_tokens_hc_kernel(
        n_vocab: u32,
        n_embd: u32,
        n_hc: u32,
        count: u64,
        tokens: &[i32],
        weights: &[f16],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let gid = index.get();
        if (gid as u64) < count {
            let dimension = gid % n_embd as usize;
            let token_index = gid / n_embd as usize / n_hc as usize;
            let token = tokens[token_index];
            let token = if token < 0 || token as u32 >= n_vocab {
                0
            } else {
                token as usize
            };
            if let Some(element) = out.get_mut(index) {
                *element = weights[token * n_embd as usize + dimension] as f32;
            }
        }
    }

    #[kernel]
    pub fn abi_matmul_q8_0_preq_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
        use_dp4a: u32,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64 * 8 + (thread::threadIdx_x() >> 5) as u64;
        let lane = (thread::threadIdx_x() & 31) as u64;
        if row >= out_dim {
            return;
        }
        let mut acc = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let remaining = in_dim - block * 32;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = (block * 32) as usize;
            let dot = q8_dot(weights, weight_base, xq, xq_base, count, use_dp4a != 0);
            acc += weight_scale * xscale[block as usize] * dot as f32;
            block += 32;
        }
        let mut offset = 16_u32;
        while offset > 0 {
            acc += warp::shuffle_down_f32(acc, offset);
            offset >>= 1;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row as usize) = acc;
            }
        }
    }

    #[kernel]
    pub fn abi_matmul_q8_0_preq_batch_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        blocks: u64,
        use_dp4a: u32,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64 * 8 + (thread::threadIdx_x() >> 5) as u64;
        let token = thread::blockIdx_y() as u64;
        let lane = (thread::threadIdx_x() & 31) as u64;
        if row >= out_dim || token >= n_tok {
            return;
        }
        let mut acc = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let remaining = in_dim - block * 32;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = ((token * blocks + block) * 32) as usize;
            let dot = q8_dot(weights, weight_base, xq, xq_base, count, use_dp4a != 0);
            acc += weight_scale * xscale[(token * blocks + block) as usize] * dot as f32;
            block += 32;
        }
        let mut offset = 16_u32;
        while offset > 0 {
            acc += warp::shuffle_down_f32(acc, offset);
            offset >>= 1;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(token as usize * out_dim as usize + row as usize) = acc;
            }
        }
    }

    #[kernel]
    pub fn abi_matmul_q8_0_pair_preq_warp8_kernel(
        in_dim: u64,
        out0_dim: u64,
        out1_dim: u64,
        blocks: u64,
        use_dp4a: u32,
        weights0: &[u8],
        weights1: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out0: DisjointSlice<f32>,
        mut out1: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64 * 8 + (thread::threadIdx_x() >> 5) as u64;
        let lane = (thread::threadIdx_x() & 31) as u64;
        if row >= out0_dim && row >= out1_dim {
            return;
        }
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let remaining = in_dim - block * 32;
            let count = if remaining < 32 { remaining } else { 32 };
            let xq_base = (block * 32) as usize;
            if row < out0_dim {
                let weight_base = ((row * blocks + block) * 34) as usize;
                let scale_bits =
                    weights0[weight_base] as u16 | ((weights0[weight_base + 1] as u16) << 8);
                let weight_scale = f16::from_bits(scale_bits) as f32;
                let dot = q8_dot(weights0, weight_base, xq, xq_base, count, use_dp4a != 0);
                acc0 += weight_scale * xscale[block as usize] * dot as f32;
            }
            if row < out1_dim {
                let weight_base = ((row * blocks + block) * 34) as usize;
                let scale_bits =
                    weights1[weight_base] as u16 | ((weights1[weight_base + 1] as u16) << 8);
                let weight_scale = f16::from_bits(scale_bits) as f32;
                let dot = q8_dot(weights1, weight_base, xq, xq_base, count, use_dp4a != 0);
                acc1 += weight_scale * xscale[block as usize] * dot as f32;
            }
            block += 32;
        }
        let mut offset = 16_u32;
        while offset > 0 {
            acc0 += warp::shuffle_down_f32(acc0, offset);
            acc1 += warp::shuffle_down_f32(acc1, offset);
            offset >>= 1;
        }
        if lane == 0 {
            unsafe {
                if row < out0_dim {
                    *out0.get_unchecked_mut(row as usize) = acc0;
                }
                if row < out1_dim {
                    *out1.get_unchecked_mut(row as usize) = acc1;
                }
            }
        }
    }

    #[kernel]
    pub fn abi_matmul_q8_0_hc_expand_preq_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        n_embd: u32,
        n_hc: u32,
        blocks: u64,
        has_add: u32,
        use_dp4a: u32,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        block_add: &[f32],
        residual_hc: &[f32],
        split: &[f32],
        mut block_out: DisjointSlice<f32>,
        mut out_hc: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64 * 8 + (thread::threadIdx_x() >> 5) as u64;
        let lane = (thread::threadIdx_x() & 31) as u64;
        if row >= out_dim {
            return;
        }
        let mut acc = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            let remaining = in_dim - block * 32;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = (block * 32) as usize;
            let dot = q8_dot(weights, weight_base, xq, xq_base, count, use_dp4a != 0);
            acc += weight_scale * xscale[block as usize] * dot as f32;
            block += 32;
        }
        let mut offset = 16_u32;
        while offset > 0 {
            acc += warp::shuffle_down_f32(acc, offset);
            offset >>= 1;
        }
        if lane == 0 {
            let row = row as usize;
            unsafe {
                *block_out.get_unchecked_mut(row) = acc;
            }
            let block_value = if has_add != 0 {
                acc + block_add[row]
            } else {
                acc
            };
            let post_base = n_hc as usize;
            let combination_base = (2 * n_hc) as usize;
            let mut destination_hc = 0_u32;
            while destination_hc < n_hc {
                let mut hc_acc = block_value * split[post_base + destination_hc as usize];
                let mut source_hc = 0_u32;
                while source_hc < n_hc {
                    let combination_index = combination_base
                        + destination_hc as usize
                        + source_hc as usize * n_hc as usize;
                    let residual_index = source_hc as usize * n_embd as usize + row;
                    hc_acc += split[combination_index] * residual_hc[residual_index];
                    source_hc += 1;
                }
                unsafe {
                    *out_hc.get_unchecked_mut(destination_hc as usize * n_embd as usize + row) =
                        hc_acc;
                }
                destination_hc += 1;
            }
        }
    }

    fn q8_dot(
        weights: &[u8],
        weight_base: usize,
        xq: &[i8],
        xq_base: usize,
        count: u64,
        use_dp4a: bool,
    ) -> i32 {
        let mut dot = 0_i32;
        if use_dp4a && count == 32 {
            let mut lane = 0_usize;
            while lane < 32 {
                let weight_word = (weights[weight_base + 2 + lane] as u32
                    | (weights[weight_base + 2 + lane + 1] as u32) << 8
                    | (weights[weight_base + 2 + lane + 2] as u32) << 16
                    | (weights[weight_base + 2 + lane + 3] as u32) << 24)
                    as i32;
                let x_word = (xq[xq_base + lane] as u8 as u32
                    | (xq[xq_base + lane + 1] as u8 as u32) << 8
                    | (xq[xq_base + lane + 2] as u8 as u32) << 16
                    | (xq[xq_base + lane + 3] as u8 as u32) << 24)
                    as i32;
                dot = integer::dp4a_i8(weight_word, x_word, dot);
                lane += 4;
            }
        } else {
            let mut lane = 0_u64;
            while lane < count {
                dot += (weights[weight_base + 2 + lane as usize] as i8 as i32)
                    * xq[xq_base + lane as usize] as i32;
                lane += 1;
            }
        }
        dot
    }

    fn round_ties_even(value: f32) -> i32 {
        let lower = value.floor();
        let fraction = value - lower;
        let mut rounded = lower as i32;
        if fraction > 0.5 || (fraction == 0.5 && (rounded & 1) != 0) {
            rounded += 1;
        }
        rounded
    }

    fn clamp_q8(value: i32) -> i8 {
        if value > 127 {
            127
        } else if value < -128 {
            -128
        } else {
            value as i8
        }
    }

    #[kernel]
    pub fn abi_f32_to_f16_kernel(count: u64, x: &[f32], mut out: DisjointSlice<f16>) {
        let index = thread::index_1d();
        let offset = index.get();
        if (offset as u64) < count {
            if let Some(element) = out.get_mut(index) {
                *element = x[offset] as f16;
            }
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

    #[kernel]
    pub fn abi_matmul_f16_pair_ordered_chunks_kernel(
        in_dim: u64,
        out_dim: u64,
        weights0: &[f16],
        weights1: &[f16],
        x: &[f32],
        mut out0: DisjointSlice<f32>,
        mut out1: DisjointSlice<f32>,
    ) {
        static mut PARTIAL0: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut PARTIAL1: SharedArray<f32, 32> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        if row >= out_dim {
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
        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        let mut i = start;
        while i < end {
            let value = x[i];
            sum0 += weights0[weight_base + i] as f32 * value;
            sum1 += weights1[weight_base + i] as f32 * value;
            i += 1;
        }
        unsafe {
            PARTIAL0[tid] = sum0;
            PARTIAL1[tid] = sum1;
        }
        thread::sync_threads();
        if tid == 0 {
            let mut total0 = 0.0_f32;
            let mut total1 = 0.0_f32;
            let mut lane = 0_usize;
            while lane < 32 {
                unsafe {
                    total0 += PARTIAL0[lane];
                    total1 += PARTIAL1[lane];
                }
                lane += 1;
            }
            unsafe {
                *out0.get_unchecked_mut(row as usize) = total0;
                *out1.get_unchecked_mut(row as usize) = total1;
            }
        }
    }

    #[kernel]
    pub fn abi_matmul_f32_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        weights: &[f32],
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
            sum += weights[weight_base + i] * x[x_base + i];
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
}

#[derive(Clone, Debug)]
pub(crate) struct AbiKernelModule {
    add_kernel: CudaFunction,
    repeat_hc_kernel: CudaFunction,
    hc_split_sinkhorn_kernel: CudaFunction,
    hc_split_weighted_sum_fused_kernel: CudaFunction,
    hc_split_weighted_sum_norm_fused_kernel: CudaFunction,
    output_hc_weights_kernel: CudaFunction,
    embed_token_hc_kernel: CudaFunction,
    embed_tokens_hc_kernel: CudaFunction,
    hc_weighted_sum_kernel: CudaFunction,
    hc_expand_kernel: CudaFunction,
    directional_steering_project_kernel: CudaFunction,
    swiglu_kernel: CudaFunction,
    rms_norm_plain_kernel: CudaFunction,
    head_rms_norm_kernel: CudaFunction,
    rope_tail_kernel: CudaFunction,
    store_raw_kv_batch_kernel: CudaFunction,
    compressor_store_kernel: CudaFunction,
    compressor_set_rows_kernel: CudaFunction,
    compressor_prefill_ratio4_replay_pool_kernel: CudaFunction,
    compressor_prefill_pool_kernel: CudaFunction,
    compressor_update_pool_kernel: CudaFunction,
    compressor_shift_ratio4_kernel: CudaFunction,
    router_select_kernel: CudaFunction,
    router_select_parallel_kernel: CudaFunction,
    router_select_warp_topk_kernel: CudaFunction,
    moe_q8_k_quantize_kernel: CudaFunction,
    moe_gate_up_mid_f32_kernel: CudaFunction,
    moe_down_f32_kernel: CudaFunction,
    moe_gate_up_mid_qwarp32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_count_sorted_pairs_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_prefix_sorted_pairs_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_scatter_sorted_pairs_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_build_expert_tile_offsets_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_build_expert_tiles_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_gate_up_mid_expert_tile4_row32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_gate_up_mid_expert_tile8_row32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_gate_up_mid_expert_tile8_rowspan_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_gate_up_mid_expert_tile8_rowspan_cached_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_atomic_output_zero_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_expert_tile4_row32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_expert_tile8_row32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_expert_tile16_row32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_expert_tile16_rowspan_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_expert_tile16_rowspan_cached_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_gate_up_mid_sorted_qwarp32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_gate_up_mid_sorted_p2_qwarp32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_sorted_qwarp32_kernel: CudaFunction,
    #[allow(dead_code)]
    moe_down_sorted_p2_qwarp32_kernel: CudaFunction,
    moe_gate_up_mid_decode_lut_qwarp32_kernel: CudaFunction,
    moe_gate_up_mid_decode_q4_k_qwarp32_kernel: CudaFunction,
    moe_down_qwarp32_kernel: CudaFunction,
    moe_down_sum6_qwarp32_kernel: CudaFunction,
    moe_down_q4_k_sum6_qwarp32_kernel: CudaFunction,
    moe_sum_kernel: CudaFunction,
    attention_decode_mixed_kernel: CudaFunction,
    attention_decode_mixed_heads8_online_kernel: CudaFunction,
    attention_prefill_raw_kernel: CudaFunction,
    attention_prefill_mixed_kernel: CudaFunction,
    attention_static_mixed_heads8_online_kernel: CudaFunction,
    attention_prefill_pack_mixed_kv_kernel: CudaFunction,
    attention_prefill_pack_q_heads_kernel: CudaFunction,
    attention_prefill_replicate_kv_kernel: CudaFunction,
    attention_prefill_raw_softmax_kernel: CudaFunction,
    attention_prefill_mixed_softmax_kernel: CudaFunction,
    attention_prefill_unpack_heads_kernel: CudaFunction,
    indexed_topk_sort_512_asc_kernel: CudaFunction,
    attention_indexed_mixed_kernel: CudaFunction,
    attention_indexed_mixed_heads8_online_kernel: CudaFunction,
    attention_indexed_mixed_heads8_rb4_kernel: CudaFunction,
    fp8_kv_quantize_kernel: CudaFunction,
    indexer_hadamard_fp4_kernel: CudaFunction,
    indexer_scores_kernel: CudaFunction,
    indexer_score_one_direct_kernel: CudaFunction,
    indexer_scores_wmma_kernel: CudaFunction,
    indexer_scores_wmma32_kernel: CudaFunction,
    indexer_scores_wmma64_kernel: CudaFunction,
    indexer_scores_wmma128_kernel: CudaFunction,
    indexer_topk_kernel: CudaFunction,
    indexer_topk_1024_kernel: CudaFunction,
    indexer_topk_pow2_2048_kernel: CudaFunction,
    indexer_topk_pow2_4096_kernel: CudaFunction,
    indexer_topk_pow2_u16_8192_kernel: CudaFunction,
    indexer_topk_8192_packed_key_equivalent_kernel: CudaFunction,
    indexer_topk_chunk_pow2_4096_kernel: CudaFunction,
    indexer_topk_tree_merge_pow2_4096_kernel: CudaFunction,
    indexer_topk_merge_pow2_4096_kernel: CudaFunction,
    topk_mask_kernel: CudaFunction,
    rms_norm_weight_kernel: CudaFunction,
    dsv4_qkv_rms_norm_rows_kernel: CudaFunction,
    dequant_q8_0_to_f16_kernel: CudaFunction,
    dequant_q8_0_to_f32_kernel: CudaFunction,
    quantize_q8_0_f32_kernel: CudaFunction,
    grouped_q8_0_a_preq_warp8_kernel: CudaFunction,
    attention_pack_group_heads_f16_kernel: CudaFunction,
    f16_to_f32_kernel: CudaFunction,
    attention_expand_group_weights_sgemm_kernel: CudaFunction,
    attention_unpack_group_low_kernel: CudaFunction,
    matmul_q8_0_preq_kernel: CudaFunction,
    matmul_q8_0_preq_warp8_kernel: CudaFunction,
    matmul_q8_0_preq_batch_warp8_kernel: CudaFunction,
    matmul_q8_0_pair_preq_warp8_kernel: CudaFunction,
    matmul_q8_0_hc_expand_preq_warp8_kernel: CudaFunction,
    f32_to_f16_kernel: CudaFunction,
    matmul_f16_kernel: CudaFunction,
    matmul_f16_serial_kernel: CudaFunction,
    matmul_f16_ordered_chunks_kernel: CudaFunction,
    matmul_f16_pair_ordered_chunks_kernel: CudaFunction,
    matmul_f32_kernel: CudaFunction,
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
            hc_split_sinkhorn_kernel: module
                .load_function("abi_hc_split_sinkhorn_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            hc_split_weighted_sum_fused_kernel: module
                .load_function("abi_hc_split_weighted_sum_fused_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            hc_split_weighted_sum_norm_fused_kernel: module
                .load_function("abi_hc_split_weighted_sum_norm_fused_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            output_hc_weights_kernel: module
                .load_function("abi_output_hc_weights_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            embed_token_hc_kernel: module
                .load_function("abi_embed_token_hc_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            embed_tokens_hc_kernel: module
                .load_function("abi_embed_tokens_hc_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            hc_weighted_sum_kernel: module
                .load_function("abi_hc_weighted_sum_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            hc_expand_kernel: module
                .load_function("abi_hc_expand_kernel")
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
            head_rms_norm_kernel: module
                .load_function("abi_head_rms_norm_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            rope_tail_kernel: module
                .load_function("abi_rope_tail_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            store_raw_kv_batch_kernel: module
                .load_function("abi_store_raw_kv_batch_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            compressor_store_kernel: module
                .load_function("abi_compressor_store_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            compressor_set_rows_kernel: module
                .load_function("abi_compressor_set_rows_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            compressor_prefill_ratio4_replay_pool_kernel: module
                .load_function("abi_compressor_prefill_ratio4_replay_pool_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            compressor_prefill_pool_kernel: module
                .load_function("abi_compressor_prefill_pool_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            compressor_update_pool_kernel: module
                .load_function("abi_compressor_update_pool_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            compressor_shift_ratio4_kernel: module
                .load_function("abi_compressor_shift_ratio4_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            router_select_kernel: module
                .load_function("abi_router_select_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            router_select_parallel_kernel: module
                .load_function("abi_router_select_parallel_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            router_select_warp_topk_kernel: module
                .load_function("abi_router_select_warp_topk_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_q8_k_quantize_kernel: module
                .load_function("abi_moe_q8_k_quantize_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_f32_kernel: module
                .load_function("abi_moe_gate_up_mid_f32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_f32_kernel: module
                .load_function("abi_moe_down_f32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_qwarp32_kernel: module
                .load_function("abi_moe_gate_up_mid_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_count_sorted_pairs_kernel: module
                .load_function("abi_moe_count_sorted_pairs_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_prefix_sorted_pairs_kernel: module
                .load_function("abi_moe_prefix_sorted_pairs_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_scatter_sorted_pairs_kernel: module
                .load_function("abi_moe_scatter_sorted_pairs_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_build_expert_tile_offsets_kernel: module
                .load_function("abi_moe_build_expert_tile_offsets_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_build_expert_tiles_kernel: module
                .load_function("abi_moe_build_expert_tiles_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_expert_tile4_row32_kernel: module
                .load_function("abi_moe_gate_up_mid_expert_tile4_row32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_expert_tile8_row32_kernel: module
                .load_function("abi_moe_gate_up_mid_expert_tile8_row32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_expert_tile8_rowspan_kernel: module
                .load_function("abi_moe_gate_up_mid_expert_tile8_rowspan_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_expert_tile8_rowspan_cached_kernel: module
                .load_function("abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_atomic_output_zero_kernel: module
                .load_function("abi_moe_atomic_output_zero_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_expert_tile4_row32_kernel: module
                .load_function("abi_moe_down_expert_tile4_row32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_expert_tile8_row32_kernel: module
                .load_function("abi_moe_down_expert_tile8_row32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_expert_tile16_row32_kernel: module
                .load_function("abi_moe_down_expert_tile16_row32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_expert_tile16_rowspan_kernel: module
                .load_function("abi_moe_down_expert_tile16_rowspan_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_expert_tile16_rowspan_cached_kernel: module
                .load_function("abi_moe_down_expert_tile16_rowspan_cached_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_sorted_qwarp32_kernel: module
                .load_function("abi_moe_gate_up_mid_sorted_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_sorted_p2_qwarp32_kernel: module
                .load_function("abi_moe_gate_up_mid_sorted_p2_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_sorted_qwarp32_kernel: module
                .load_function("abi_moe_down_sorted_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_sorted_p2_qwarp32_kernel: module
                .load_function("abi_moe_down_sorted_p2_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_decode_lut_qwarp32_kernel: module
                .load_function("abi_moe_gate_up_mid_decode_lut_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_gate_up_mid_decode_q4_k_qwarp32_kernel: module
                .load_function("abi_moe_gate_up_mid_decode_q4_k_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_qwarp32_kernel: module
                .load_function("abi_moe_down_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_sum6_qwarp32_kernel: module
                .load_function("abi_moe_down_sum6_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_down_q4_k_sum6_qwarp32_kernel: module
                .load_function("abi_moe_down_q4_k_sum6_qwarp32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            moe_sum_kernel: module
                .load_function("abi_moe_sum_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_decode_mixed_kernel: module
                .load_function("abi_attention_decode_mixed_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_decode_mixed_heads8_online_kernel: module
                .load_function("abi_attention_decode_mixed_heads8_online_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_raw_kernel: module
                .load_function("abi_attention_prefill_raw_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_mixed_kernel: module
                .load_function("abi_attention_prefill_mixed_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_static_mixed_heads8_online_kernel: module
                .load_function("abi_attention_static_mixed_heads8_online_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_pack_mixed_kv_kernel: module
                .load_function("abi_attention_prefill_pack_mixed_kv_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_pack_q_heads_kernel: module
                .load_function("abi_attention_prefill_pack_q_heads_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_replicate_kv_kernel: module
                .load_function("abi_attention_prefill_replicate_kv_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_raw_softmax_kernel: module
                .load_function("abi_attention_prefill_raw_softmax_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_mixed_softmax_kernel: module
                .load_function("abi_attention_prefill_mixed_softmax_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_prefill_unpack_heads_kernel: module
                .load_function("abi_attention_prefill_unpack_heads_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexed_topk_sort_512_asc_kernel: module
                .load_function("abi_indexed_topk_sort_512_asc_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_indexed_mixed_kernel: module
                .load_function("abi_attention_indexed_mixed_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_indexed_mixed_heads8_online_kernel: module
                .load_function("abi_attention_indexed_mixed_heads8_online_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_indexed_mixed_heads8_rb4_kernel: module
                .load_function("abi_attention_indexed_mixed_heads8_rb4_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            fp8_kv_quantize_kernel: module
                .load_function("abi_fp8_kv_quantize_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_hadamard_fp4_kernel: module
                .load_function("abi_indexer_hadamard_fp4_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_scores_kernel: module
                .load_function("abi_indexer_scores_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_score_one_direct_kernel: module
                .load_function("abi_indexer_score_one_direct_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_scores_wmma_kernel: module
                .load_function("abi_indexer_scores_wmma_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_scores_wmma32_kernel: module
                .load_function("abi_indexer_scores_wmma32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_scores_wmma64_kernel: module
                .load_function("abi_indexer_scores_wmma64_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_scores_wmma128_kernel: module
                .load_function("abi_indexer_scores_wmma128_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_kernel: module
                .load_function("abi_indexer_topk_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_1024_kernel: module
                .load_function("abi_indexer_topk_1024_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_pow2_2048_kernel: module
                .load_function("abi_indexer_topk_pow2_2048_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_pow2_4096_kernel: module
                .load_function("abi_indexer_topk_pow2_4096_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_pow2_u16_8192_kernel: module
                .load_function("abi_indexer_topk_pow2_u16_8192_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_8192_packed_key_equivalent_kernel: module
                .load_function("abi_indexer_topk_8192_packed_key_equivalent_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_chunk_pow2_4096_kernel: module
                .load_function("abi_indexer_topk_chunk_pow2_4096_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_tree_merge_pow2_4096_kernel: module
                .load_function("abi_indexer_topk_tree_merge_pow2_4096_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_topk_merge_pow2_4096_kernel: module
                .load_function("abi_indexer_topk_merge_pow2_4096_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            topk_mask_kernel: module
                .load_function("abi_topk_mask_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            rms_norm_weight_kernel: module
                .load_function("abi_rms_norm_weight_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            dsv4_qkv_rms_norm_rows_kernel: module
                .load_function("abi_dsv4_qkv_rms_norm_rows_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            dequant_q8_0_to_f16_kernel: module
                .load_function("abi_dequant_q8_0_to_f16_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            dequant_q8_0_to_f32_kernel: module
                .load_function("abi_dequant_q8_0_to_f32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            quantize_q8_0_f32_kernel: module
                .load_function("abi_quantize_q8_0_f32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            grouped_q8_0_a_preq_warp8_kernel: module
                .load_function("abi_grouped_q8_0_a_preq_warp8_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_pack_group_heads_f16_kernel: module
                .load_function("abi_attention_pack_group_heads_f16_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            f16_to_f32_kernel: module
                .load_function("abi_f16_to_f32_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_expand_group_weights_sgemm_kernel: module
                .load_function("abi_attention_expand_group_weights_sgemm_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            attention_unpack_group_low_kernel: module
                .load_function("abi_attention_unpack_group_low_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_q8_0_preq_kernel: module
                .load_function("abi_matmul_q8_0_preq_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_q8_0_preq_warp8_kernel: module
                .load_function("abi_matmul_q8_0_preq_warp8_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_q8_0_preq_batch_warp8_kernel: module
                .load_function("abi_matmul_q8_0_preq_batch_warp8_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_q8_0_pair_preq_warp8_kernel: module
                .load_function("abi_matmul_q8_0_pair_preq_warp8_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_q8_0_hc_expand_preq_warp8_kernel: module
                .load_function("abi_matmul_q8_0_hc_expand_preq_warp8_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            f32_to_f16_kernel: module
                .load_function("abi_f32_to_f16_kernel")
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
            matmul_f16_pair_ordered_chunks_kernel: module
                .load_function("abi_matmul_f16_pair_ordered_chunks_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            matmul_f32_kernel: module
                .load_function("abi_matmul_f32_kernel")
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_split_sinkhorn_tensor(
        &self,
        stream: &CudaStream,
        split_ptr: u64,
        mix_ptr: u64,
        scale_ptr: u64,
        base_ptr: u64,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> bool {
        let Some(config) = launch_config(u64::from(n_rows)) else {
            return false;
        };
        let split_len = u64::from(n_rows) * 24;
        let mut n_rows = n_rows;
        let mut sinkhorn_iters = sinkhorn_iters;
        let mut eps = eps;
        let mut mix_ptr = mix_ptr;
        let mut mix_len = split_len;
        let mut scale_ptr = scale_ptr;
        let mut scale_len = 3_u64;
        let mut base_ptr = base_ptr;
        let mut base_len = 24_u64;
        let mut split_ptr = split_ptr;
        let mut split_len = split_len;
        let mut params = [
            (&mut n_rows as *mut u32).cast::<c_void>(),
            (&mut sinkhorn_iters as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut mix_ptr as *mut u64).cast::<c_void>(),
            (&mut mix_len as *mut u64).cast::<c_void>(),
            (&mut scale_ptr as *mut u64).cast::<c_void>(),
            (&mut scale_len as *mut u64).cast::<c_void>(),
            (&mut base_ptr as *mut u64).cast::<c_void>(),
            (&mut base_len as *mut u64).cast::<c_void>(),
            (&mut split_ptr as *mut u64).cast::<c_void>(),
            (&mut split_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates row spans and both cached model
        // parameter ranges before the asynchronous kernel launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.hc_split_sinkhorn_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_weighted_sum_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        residual_hc_ptr: u64,
        weights_ptr: u64,
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        weight_stride: u32,
    ) -> bool {
        let count = u64::from(n_embd) * u64::from(n_tokens);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let residual_len = count * u64::from(n_hc);
        let weights_len = (u64::from(n_tokens) - 1) * u64::from(weight_stride) + u64::from(n_hc);
        let mut n_embd = n_embd;
        let mut n_hc = n_hc;
        let mut n_tokens = n_tokens;
        let mut weight_stride = weight_stride;
        let mut residual_hc_ptr = residual_hc_ptr;
        let mut residual_hc_len = residual_len;
        let mut weights_ptr = weights_ptr;
        let mut weights_len = weights_len;
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut weight_stride as *mut u32).cast::<c_void>(),
            (&mut residual_hc_ptr as *mut u64).cast::<c_void>(),
            (&mut residual_hc_len as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates all token-strided residual and
        // weight spans through the last accessed hyperconnection.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.hc_weighted_sum_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_split_weighted_sum_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        split_ptr: u64,
        mix_ptr: u64,
        residual_hc_ptr: u64,
        scale_ptr: u64,
        base_ptr: u64,
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> bool {
        if n_rows == 0 {
            return false;
        }
        let config = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mix_len = u64::from(n_rows) * 24;
        let out_len = u64::from(n_rows) * u64::from(n_embd);
        let residual_hc_len = out_len * u64::from(n_hc);
        let mut n_embd = n_embd;
        let mut n_hc = n_hc;
        let mut n_rows = n_rows;
        let mut sinkhorn_iters = sinkhorn_iters;
        let mut eps = eps;
        let mut mix_ptr = mix_ptr;
        let mut mix_len = mix_len;
        let mut residual_hc_ptr = residual_hc_ptr;
        let mut residual_hc_len = residual_hc_len;
        let mut scale_ptr = scale_ptr;
        let mut scale_len = 3_u64;
        let mut base_ptr = base_ptr;
        let mut base_len = 24_u64;
        let mut split_ptr = split_ptr;
        let mut split_len = mix_len;
        let mut out_ptr = out_ptr;
        let mut out_len = out_len;
        let mut params = [
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut n_rows as *mut u32).cast::<c_void>(),
            (&mut sinkhorn_iters as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut mix_ptr as *mut u64).cast::<c_void>(),
            (&mut mix_len as *mut u64).cast::<c_void>(),
            (&mut residual_hc_ptr as *mut u64).cast::<c_void>(),
            (&mut residual_hc_len as *mut u64).cast::<c_void>(),
            (&mut scale_ptr as *mut u64).cast::<c_void>(),
            (&mut scale_len as *mut u64).cast::<c_void>(),
            (&mut base_ptr as *mut u64).cast::<c_void>(),
            (&mut base_len as *mut u64).cast::<c_void>(),
            (&mut split_ptr as *mut u64).cast::<c_void>(),
            (&mut split_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates output-derived rows, every input
        // span, and both cached model ranges before the fused launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.hc_split_weighted_sum_fused_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_split_weighted_sum_norm_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        norm_out_ptr: u64,
        split_ptr: u64,
        mix_ptr: u64,
        residual_hc_ptr: u64,
        scale_ptr: u64,
        base_ptr: u64,
        norm_weight_ptr: u64,
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) -> bool {
        if n_rows == 0 {
            return false;
        }
        let config = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mix_len = u64::from(n_rows) * 24;
        let out_len = u64::from(n_rows) * u64::from(n_embd);
        let residual_hc_len = out_len * u64::from(n_hc);
        let mut n_embd = n_embd;
        let mut n_hc = n_hc;
        let mut n_rows = n_rows;
        let mut sinkhorn_iters = sinkhorn_iters;
        let mut eps = eps;
        let mut norm_eps = norm_eps;
        let mut mix_ptr = mix_ptr;
        let mut mix_len = mix_len;
        let mut residual_hc_ptr = residual_hc_ptr;
        let mut residual_hc_len = residual_hc_len;
        let mut scale_ptr = scale_ptr;
        let mut scale_len = 3_u64;
        let mut base_ptr = base_ptr;
        let mut base_len = 24_u64;
        let mut norm_weight_ptr = norm_weight_ptr;
        let mut norm_weight_len = u64::from(n_embd);
        let mut split_ptr = split_ptr;
        let mut split_len = mix_len;
        let mut out_ptr = out_ptr;
        let mut out_len = out_len;
        let mut norm_out_ptr = norm_out_ptr;
        let mut norm_out_len = out_len;
        let mut params = [
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut n_rows as *mut u32).cast::<c_void>(),
            (&mut sinkhorn_iters as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut norm_eps as *mut f32).cast::<c_void>(),
            (&mut mix_ptr as *mut u64).cast::<c_void>(),
            (&mut mix_len as *mut u64).cast::<c_void>(),
            (&mut residual_hc_ptr as *mut u64).cast::<c_void>(),
            (&mut residual_hc_len as *mut u64).cast::<c_void>(),
            (&mut scale_ptr as *mut u64).cast::<c_void>(),
            (&mut scale_len as *mut u64).cast::<c_void>(),
            (&mut base_ptr as *mut u64).cast::<c_void>(),
            (&mut base_len as *mut u64).cast::<c_void>(),
            (&mut norm_weight_ptr as *mut u64).cast::<c_void>(),
            (&mut norm_weight_len as *mut u64).cast::<c_void>(),
            (&mut split_ptr as *mut u64).cast::<c_void>(),
            (&mut split_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
            (&mut norm_out_ptr as *mut u64).cast::<c_void>(),
            (&mut norm_out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer only selects this one-row fused launch after
        // validating all tensor spans and the three cached model ranges.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.hc_split_weighted_sum_norm_fused_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn output_hc_weights_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        pre_ptr: u64,
        scale_ptr: u64,
        base_ptr: u64,
        n_hc: u32,
        n_tokens: u32,
        eps: f32,
    ) -> bool {
        let count = u64::from(n_tokens) * u64::from(n_hc);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_hc = n_hc;
        let mut n_tokens = n_tokens;
        let mut eps = eps;
        let mut pre_ptr = pre_ptr;
        let mut pre_len = count;
        let mut scale_ptr = scale_ptr;
        let mut scale_len = 1_u64;
        let mut base_ptr = base_ptr;
        let mut base_len = u64::from(n_hc);
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut pre_ptr as *mut u64).cast::<c_void>(),
            (&mut pre_len as *mut u64).cast::<c_void>(),
            (&mut scale_ptr as *mut u64).cast::<c_void>(),
            (&mut scale_len as *mut u64).cast::<c_void>(),
            (&mut base_ptr as *mut u64).cast::<c_void>(),
            (&mut base_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates complete output rows, matching
        // input coverage, and both cached model ranges before launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.output_hc_weights_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn embed_token_hc_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        weights_ptr: u64,
        n_vocab: u32,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> bool {
        let count = u64::from(n_embd) * u64::from(n_hc);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut token = token;
        let mut n_embd = n_embd;
        let mut count = count;
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(n_vocab) * u64::from(n_embd);
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut token as *mut u32).cast::<c_void>(),
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut count as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates the single token, complete output
        // span, and cached FP16 embedding range before launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.embed_token_hc_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn embed_tokens_hc_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        tokens_ptr: u64,
        weights_ptr: u64,
        n_vocab: u32,
        n_tokens: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> bool {
        let count = u64::from(n_tokens) * u64::from(n_hc) * u64::from(n_embd);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_vocab = n_vocab;
        let mut n_embd = n_embd;
        let mut n_hc = n_hc;
        let mut count = count;
        let mut tokens_ptr = tokens_ptr;
        let mut tokens_len = u64::from(n_tokens);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(n_vocab) * u64::from(n_embd);
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut n_vocab as *mut u32).cast::<c_void>(),
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut count as *mut u64).cast::<c_void>(),
            (&mut tokens_ptr as *mut u64).cast::<c_void>(),
            (&mut tokens_len as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates tokens and output spans plus the
        // cached embedding range; the kernel bounds invalid IDs to row zero.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.embed_tokens_hc_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn hc_expand_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        block_out_ptr: u64,
        block_add_ptr: u64,
        residual_hc_ptr: u64,
        post_ptr: u64,
        comb_ptr: u64,
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        post_stride: u32,
        comb_stride: u32,
        has_add: bool,
    ) -> bool {
        let count = u64::from(n_tokens) * u64::from(n_hc) * u64::from(n_embd);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let block_len = u64::from(n_tokens) * u64::from(n_embd);
        let residual_len = count;
        let post_len = (u64::from(n_tokens) - 1) * u64::from(post_stride) + u64::from(n_hc);
        let comb_width = u64::from(n_hc) * u64::from(n_hc);
        let comb_len = (u64::from(n_tokens) - 1) * u64::from(comb_stride) + comb_width;
        let mut n_embd = n_embd;
        let mut n_hc = n_hc;
        let mut n_tokens = n_tokens;
        let mut post_stride = post_stride;
        let mut comb_stride = comb_stride;
        let mut has_add = u32::from(has_add);
        let mut block_out_ptr = block_out_ptr;
        let mut block_out_len = block_len;
        let mut block_add_ptr = block_add_ptr;
        let mut block_add_len = block_len;
        let mut residual_hc_ptr = residual_hc_ptr;
        let mut residual_hc_len = residual_len;
        let mut post_ptr = post_ptr;
        let mut post_len = post_len;
        let mut comb_ptr = comb_ptr;
        let mut comb_len = comb_len;
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut post_stride as *mut u32).cast::<c_void>(),
            (&mut comb_stride as *mut u32).cast::<c_void>(),
            (&mut has_add as *mut u32).cast::<c_void>(),
            (&mut block_out_ptr as *mut u64).cast::<c_void>(),
            (&mut block_out_len as *mut u64).cast::<c_void>(),
            (&mut block_add_ptr as *mut u64).cast::<c_void>(),
            (&mut block_add_len as *mut u64).cast::<c_void>(),
            (&mut residual_hc_ptr as *mut u64).cast::<c_void>(),
            (&mut residual_hc_len as *mut u64).cast::<c_void>(),
            (&mut post_ptr as *mut u64).cast::<c_void>(),
            (&mut post_len as *mut u64).cast::<c_void>(),
            (&mut comb_ptr as *mut u64).cast::<c_void>(),
            (&mut comb_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates every strided source span and the full
        // output tensor before submitting the current-C-equivalent launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.hc_expand_kernel,
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

    pub(crate) unsafe fn head_rms_norm_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) -> bool {
        let rows = u64::from(n_tok) * u64::from(n_head);
        let Ok(grid_rows) = u32::try_from(rows) else {
            return false;
        };
        let config = LaunchConfig {
            grid_dim: (grid_rows, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = rows * u64::from(head_dim);
        let mut n_tok = n_tok;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut eps = eps;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut params = [
            (&mut n_tok as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the complete mutable head buffer and
        // nonzero dimensions before submitting the in-place launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.head_rms_norm_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn dsv4_fp8_kv_quantize_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_tok, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = u64::from(n_tok) * u64::from(head_dim);
        let mut n_tok = n_tok;
        let mut head_dim = head_dim;
        let mut n_rot = n_rot;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut params = [
            (&mut n_tok as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut n_rot as *mut u32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the complete mutable tensor span and
        // excludes the invalid zero-grid launch before submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.fp8_kv_quantize_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn rope_tail_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        pos_stride: u32,
        n_ctx_orig: u32,
        inverse: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        pairs: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (pairs.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = u64::from(n_tok) * u64::from(n_head) * u64::from(head_dim);
        let mut n_tok = n_tok;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut n_rot = n_rot;
        let mut pos0 = pos0;
        let mut pos_stride = pos_stride;
        let mut n_ctx_orig = n_ctx_orig;
        let mut inverse = u32::from(inverse);
        let mut freq_base = freq_base;
        let mut freq_scale = freq_scale;
        let mut ext_factor = ext_factor;
        let mut attn_factor = attn_factor;
        let mut beta_fast = beta_fast;
        let mut beta_slow = beta_slow;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut params = [
            (&mut n_tok as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut n_rot as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut pos_stride as *mut u32).cast::<c_void>(),
            (&mut n_ctx_orig as *mut u32).cast::<c_void>(),
            (&mut inverse as *mut u32).cast::<c_void>(),
            (&mut freq_base as *mut f32).cast::<c_void>(),
            (&mut freq_scale as *mut f32).cast::<c_void>(),
            (&mut ext_factor as *mut f32).cast::<c_void>(),
            (&mut attn_factor as *mut f32).cast::<c_void>(),
            (&mut beta_fast as *mut f32).cast::<c_void>(),
            (&mut beta_slow as *mut f32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the complete mutable tensor span, rotary
        // width, and nonzero checked pair grid before launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.rope_tail_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn store_raw_kv_batch_tensor(
        &self,
        stream: &CudaStream,
        raw_ptr: u64,
        kv_ptr: u64,
        raw_elements: u64,
        kv_elements: u64,
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
        grid_blocks: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut raw_cap = raw_cap;
        let mut pos0 = pos0;
        let mut n_tokens = n_tokens;
        let mut head_dim = head_dim;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = kv_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut params = [
            (&mut raw_cap as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates source/destination spans, nonzero ring
        // geometry, and the checked nonzero grid before launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.store_raw_kv_batch_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn compressor_store_batch_tensor(
        &self,
        stream: &CudaStream,
        kv_ptr: u64,
        sc_ptr: u64,
        state_kv_ptr: u64,
        state_score_ptr: u64,
        ape_ptr: u64,
        input_elements: u64,
        state_elements: u64,
        ape_elements: u64,
        ape_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
        grid_blocks: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut head_dim = head_dim;
        let mut ratio = ratio;
        let mut pos0 = pos0;
        let mut n_tokens = n_tokens;
        let mut ape_type = ape_type;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = input_elements;
        let mut sc_ptr = sc_ptr;
        let mut sc_len = input_elements;
        let mut ape_f32_ptr = ape_ptr;
        let mut ape_f32_len = if ape_type == 0 { ape_elements } else { 0 };
        let mut ape_f16_ptr = ape_ptr;
        let mut ape_f16_len = if ape_type == 1 { ape_elements } else { 0 };
        let mut state_kv_ptr = state_kv_ptr;
        let mut state_kv_len = state_elements;
        let mut state_score_ptr = state_score_ptr;
        let mut state_score_len = state_elements;
        let mut params = [
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut ape_type as *mut u32).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut sc_ptr as *mut u64).cast::<c_void>(),
            (&mut sc_len as *mut u64).cast::<c_void>(),
            (&mut ape_f32_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f32_len as *mut u64).cast::<c_void>(),
            (&mut ape_f16_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f16_len as *mut u64).cast::<c_void>(),
            (&mut state_kv_ptr as *mut u64).cast::<c_void>(),
            (&mut state_kv_len as *mut u64).cast::<c_void>(),
            (&mut state_score_ptr as *mut u64).cast::<c_void>(),
            (&mut state_score_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates source, state, and selected cached model
        // spans plus the nonzero checked launch grid before submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.compressor_store_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn compressor_set_rows_tensor(
        &self,
        stream: &CudaStream,
        kv_ptr: u64,
        sc_ptr: u64,
        state_kv_ptr: u64,
        state_score_ptr: u64,
        ape_ptr: u64,
        input_elements: u64,
        state_elements: u64,
        ape_elements: u64,
        ape_type: u32,
        width: u32,
        ratio: u32,
        pos0: u32,
        src0: u32,
        dst0: u32,
        rows: u32,
        grid_blocks: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut width = width;
        let mut ratio = ratio;
        let mut pos0 = pos0;
        let mut src0 = src0;
        let mut dst0 = dst0;
        let mut rows = rows;
        let mut ape_type = ape_type;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = input_elements;
        let mut sc_ptr = sc_ptr;
        let mut sc_len = input_elements;
        let mut ape_f32_ptr = ape_ptr;
        let mut ape_f32_len = if ape_type == 0 { ape_elements } else { 0 };
        let mut ape_f16_ptr = ape_ptr;
        let mut ape_f16_len = if ape_type == 1 { ape_elements } else { 0 };
        let mut state_kv_ptr = state_kv_ptr;
        let mut state_kv_len = state_elements;
        let mut state_score_ptr = state_score_ptr;
        let mut state_score_len = state_elements;
        let mut params = [
            (&mut width as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut src0 as *mut u32).cast::<c_void>(),
            (&mut dst0 as *mut u32).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut ape_type as *mut u32).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut sc_ptr as *mut u64).cast::<c_void>(),
            (&mut sc_len as *mut u64).cast::<c_void>(),
            (&mut ape_f32_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f32_len as *mut u64).cast::<c_void>(),
            (&mut ape_f16_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f16_len as *mut u64).cast::<c_void>(),
            (&mut state_kv_ptr as *mut u64).cast::<c_void>(),
            (&mut state_kv_len as *mut u64).cast::<c_void>(),
            (&mut state_score_ptr as *mut u64).cast::<c_void>(),
            (&mut state_score_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the source, destination, and selected
        // model spans together with the nonzero checked launch grid.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.compressor_set_rows_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn compressor_prefill_ratio4_replay_pool_tensor(
        &self,
        stream: &CudaStream,
        comp_ptr: u64,
        kv_ptr: u64,
        sc_ptr: u64,
        state_kv_ptr: u64,
        state_score_ptr: u64,
        ape_ptr: u64,
        input_elements: u64,
        state_elements: u64,
        comp_elements: u64,
        ape_elements: u64,
        ape_type: u32,
        head_dim: u32,
        pos0: u32,
        n_comp: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (head_dim.div_ceil(THREADS_PER_BLOCK), n_comp, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut head_dim = head_dim;
        let mut pos0 = pos0;
        let mut n_comp = n_comp;
        let mut ape_type = ape_type;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = input_elements;
        let mut sc_ptr = sc_ptr;
        let mut sc_len = input_elements;
        let mut state_kv_ptr = state_kv_ptr;
        let mut state_kv_len = state_elements;
        let mut state_score_ptr = state_score_ptr;
        let mut state_score_len = state_elements;
        let mut ape_f32_ptr = ape_ptr;
        let mut ape_f32_len = if ape_type == 0 { ape_elements } else { 0 };
        let mut ape_f16_ptr = ape_ptr;
        let mut ape_f16_len = if ape_type == 1 { ape_elements } else { 0 };
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut params = [
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut ape_type as *mut u32).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut sc_ptr as *mut u64).cast::<c_void>(),
            (&mut sc_len as *mut u64).cast::<c_void>(),
            (&mut state_kv_ptr as *mut u64).cast::<c_void>(),
            (&mut state_kv_len as *mut u64).cast::<c_void>(),
            (&mut state_score_ptr as *mut u64).cast::<c_void>(),
            (&mut state_score_len as *mut u64).cast::<c_void>(),
            (&mut ape_f32_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f32_len as *mut u64).cast::<c_void>(),
            (&mut ape_f16_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f16_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the complete replay input, prior state,
        // output, and selected cached APE spans before submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.compressor_prefill_ratio4_replay_pool_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn compressor_prefill_pool_tensor(
        &self,
        stream: &CudaStream,
        comp_ptr: u64,
        kv_ptr: u64,
        sc_ptr: u64,
        ape_ptr: u64,
        input_elements: u64,
        comp_elements: u64,
        ape_elements: u64,
        ape_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_comp: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (head_dim.div_ceil(THREADS_PER_BLOCK), n_comp, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut head_dim = head_dim;
        let mut ratio = ratio;
        let mut pos0 = pos0;
        let mut n_comp = n_comp;
        let mut ape_type = ape_type;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = input_elements;
        let mut sc_ptr = sc_ptr;
        let mut sc_len = input_elements;
        let mut ape_f32_ptr = ape_ptr;
        let mut ape_f32_len = if ape_type == 0 { ape_elements } else { 0 };
        let mut ape_f16_ptr = ape_ptr;
        let mut ape_f16_len = if ape_type == 1 { ape_elements } else { 0 };
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut params = [
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut ape_type as *mut u32).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut sc_ptr as *mut u64).cast::<c_void>(),
            (&mut sc_len as *mut u64).cast::<c_void>(),
            (&mut ape_f32_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f32_len as *mut u64).cast::<c_void>(),
            (&mut ape_f16_ptr as *mut u64).cast::<c_void>(),
            (&mut ape_f16_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the complete prefill input, output, and
        // selected cached APE spans before the nonzero launch is submitted.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.compressor_prefill_pool_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn compressor_update_pool_tensor(
        &self,
        stream: &CudaStream,
        row_ptr: u64,
        state_kv_ptr: u64,
        state_score_ptr: u64,
        state_elements: u64,
        head_dim: u32,
        ratio: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (head_dim.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut head_dim = head_dim;
        let mut ratio = ratio;
        let mut state_kv_ptr = state_kv_ptr;
        let mut state_kv_len = state_elements;
        let mut state_score_ptr = state_score_ptr;
        let mut state_score_len = state_elements;
        let mut row_ptr = row_ptr;
        let mut row_len = u64::from(head_dim);
        let mut params = [
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut state_kv_ptr as *mut u64).cast::<c_void>(),
            (&mut state_kv_len as *mut u64).cast::<c_void>(),
            (&mut state_score_ptr as *mut u64).cast::<c_void>(),
            (&mut state_score_len as *mut u64).cast::<c_void>(),
            (&mut row_ptr as *mut u64).cast::<c_void>(),
            (&mut row_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates output/state spans and checked nonzero
        // geometry before the emitted update pool is submitted.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.compressor_update_pool_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn compressor_shift_ratio4_tensor(
        &self,
        stream: &CudaStream,
        state_kv_ptr: u64,
        state_score_ptr: u64,
        state_elements: u64,
        width: u32,
        grid_blocks: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut width = width;
        let mut state_kv_ptr = state_kv_ptr;
        let mut state_kv_len = state_elements;
        let mut state_score_ptr = state_score_ptr;
        let mut state_score_len = state_elements;
        let mut params = [
            (&mut width as *mut u32).cast::<c_void>(),
            (&mut state_kv_ptr as *mut u64).cast::<c_void>(),
            (&mut state_kv_len as *mut u64).cast::<c_void>(),
            (&mut state_score_ptr as *mut u64).cast::<c_void>(),
            (&mut state_score_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: fixed ratio-4 state spans and the checked shift launch
        // extent cover both the source and duplicated destination halves.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.compressor_shift_ratio4_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn router_select_scalar_tensor(
        &self,
        stream: &CudaStream,
        selected_ptr: u64,
        weights_ptr: u64,
        probs_ptr: u64,
        bias_ptr: u64,
        hash_ptr: u64,
        logits_ptr: u64,
        tokens_ptr: u64,
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: bool,
        hash_mode: bool,
        use_token_buffer: bool,
    ) -> bool {
        unsafe {
            self.router_select_tensor(
                &self.router_select_kernel,
                stream,
                selected_ptr,
                weights_ptr,
                probs_ptr,
                bias_ptr,
                hash_ptr,
                logits_ptr,
                tokens_ptr,
                n_tokens,
                token_scalar,
                hash_rows,
                has_bias,
                hash_mode,
                use_token_buffer,
                (n_tokens, 1, 1),
                (1, 1, 1),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn router_select_parallel_tensor(
        &self,
        stream: &CudaStream,
        selected_ptr: u64,
        weights_ptr: u64,
        probs_ptr: u64,
        bias_ptr: u64,
        hash_ptr: u64,
        logits_ptr: u64,
        tokens_ptr: u64,
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: bool,
        hash_mode: bool,
        use_token_buffer: bool,
    ) -> bool {
        unsafe {
            self.router_select_tensor(
                &self.router_select_parallel_kernel,
                stream,
                selected_ptr,
                weights_ptr,
                probs_ptr,
                bias_ptr,
                hash_ptr,
                logits_ptr,
                tokens_ptr,
                n_tokens,
                token_scalar,
                hash_rows,
                has_bias,
                hash_mode,
                use_token_buffer,
                (n_tokens, 1, 1),
                (THREADS_PER_BLOCK, 1, 1),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn router_select_warp_topk_tensor(
        &self,
        stream: &CudaStream,
        selected_ptr: u64,
        weights_ptr: u64,
        probs_ptr: u64,
        bias_ptr: u64,
        hash_ptr: u64,
        logits_ptr: u64,
        tokens_ptr: u64,
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: bool,
        hash_mode: bool,
        use_token_buffer: bool,
    ) -> bool {
        unsafe {
            self.router_select_tensor(
                &self.router_select_warp_topk_kernel,
                stream,
                selected_ptr,
                weights_ptr,
                probs_ptr,
                bias_ptr,
                hash_ptr,
                logits_ptr,
                tokens_ptr,
                n_tokens,
                token_scalar,
                hash_rows,
                has_bias,
                hash_mode,
                use_token_buffer,
                (n_tokens.div_ceil(ABI_ROUTER_ROWS_PER_WARP_BLOCK), 1, 1),
                (32, ABI_ROUTER_ROWS_PER_WARP_BLOCK, 1),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn router_select_tensor(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        selected_ptr: u64,
        weights_ptr: u64,
        probs_ptr: u64,
        bias_ptr: u64,
        hash_ptr: u64,
        logits_ptr: u64,
        tokens_ptr: u64,
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: bool,
        hash_mode: bool,
        use_token_buffer: bool,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> bool {
        if n_tokens == 0 {
            return false;
        }
        let Some(prob_elements) = u64::from(n_tokens).checked_mul(ABI_ROUTER_N_EXPERT as u64)
        else {
            return false;
        };
        let Some(selected_elements) = u64::from(n_tokens).checked_mul(ABI_ROUTER_TOP_K as u64)
        else {
            return false;
        };
        let mut n_tokens = n_tokens;
        let mut token_scalar = token_scalar;
        let mut hash_rows = hash_rows;
        let mut has_bias = u32::from(has_bias);
        let mut hash_mode = u32::from(hash_mode);
        let mut use_token_buffer = u32::from(use_token_buffer);
        let mut logits_ptr = logits_ptr;
        let mut logits_len = prob_elements;
        let mut bias_ptr = bias_ptr;
        let mut bias_len = if has_bias != 0 {
            ABI_ROUTER_N_EXPERT as u64
        } else {
            0
        };
        let mut hash_ptr = hash_ptr;
        let mut hash_len = if hash_mode != 0 {
            u64::from(hash_rows) * ABI_ROUTER_TOP_K as u64
        } else {
            0
        };
        let mut tokens_ptr = tokens_ptr;
        let mut tokens_len = if use_token_buffer != 0 {
            u64::from(n_tokens)
        } else {
            0
        };
        let mut selected_ptr = selected_ptr;
        let mut selected_len = selected_elements;
        let mut weights_ptr = weights_ptr;
        let mut weights_len = selected_elements;
        let mut probs_ptr = probs_ptr;
        let mut probs_len = prob_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut token_scalar as *mut i32).cast::<c_void>(),
            (&mut hash_rows as *mut u32).cast::<c_void>(),
            (&mut has_bias as *mut u32).cast::<c_void>(),
            (&mut hash_mode as *mut u32).cast::<c_void>(),
            (&mut use_token_buffer as *mut u32).cast::<c_void>(),
            (&mut logits_ptr as *mut u64).cast::<c_void>(),
            (&mut logits_len as *mut u64).cast::<c_void>(),
            (&mut bias_ptr as *mut u64).cast::<c_void>(),
            (&mut bias_len as *mut u64).cast::<c_void>(),
            (&mut hash_ptr as *mut u64).cast::<c_void>(),
            (&mut hash_len as *mut u64).cast::<c_void>(),
            (&mut tokens_ptr as *mut u64).cast::<c_void>(),
            (&mut tokens_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut probs_ptr as *mut u64).cast::<c_void>(),
            (&mut probs_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the public wrappers validate output/input spans and cached
        // optional model ranges before selecting one current-C router launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                grid_dim,
                block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn moe_q8_k_quantize_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        scratch_ptr: u64,
        scratch_bytes: u64,
        in_dim: u32,
        n_rows: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (in_dim / ABI_MOE_QK_K as u32, n_rows, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut in_dim = in_dim;
        let mut n_rows = n_rows;
        let mut x_ptr = x_ptr;
        let mut x_len = u64::from(in_dim) * u64::from(n_rows);
        let mut scratch_ptr = scratch_ptr;
        let mut scratch_len = scratch_bytes;
        let mut params = [
            (&mut in_dim as *mut u32).cast::<c_void>(),
            (&mut n_rows as *mut u32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut scratch_ptr as *mut u64).cast::<c_void>(),
            (&mut scratch_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_q8_k_quantize_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn moe_count_sorted_pairs_tensor(
        &self,
        stream: &CudaStream,
        pair_count: u32,
        selected_ptr: u64,
        counts_ptr: u64,
    ) -> bool {
        let Some(config) = launch_config(u64::from(pair_count)) else {
            return false;
        };
        let mut pair_count = pair_count;
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(pair_count);
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut params = [
            (&mut pair_count as *mut u32).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_count_sorted_pairs_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn moe_prefix_sorted_pairs_tensor(
        &self,
        stream: &CudaStream,
        counts_ptr: u64,
        offsets_ptr: u64,
        cursors_ptr: u64,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut offsets_ptr = offsets_ptr;
        let mut offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut cursors_ptr = cursors_ptr;
        let mut cursors_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut params = [
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut offsets_len as *mut u64).cast::<c_void>(),
            (&mut cursors_ptr as *mut u64).cast::<c_void>(),
            (&mut cursors_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_prefix_sorted_pairs_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn moe_scatter_sorted_pairs_tensor(
        &self,
        stream: &CudaStream,
        pair_count: u32,
        selected_ptr: u64,
        cursors_ptr: u64,
        sorted_pairs_ptr: u64,
    ) -> bool {
        let Some(config) = launch_config(u64::from(pair_count)) else {
            return false;
        };
        let mut pair_count = pair_count;
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(pair_count);
        let mut cursors_ptr = cursors_ptr;
        let mut cursors_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut params = [
            (&mut pair_count as *mut u32).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut cursors_ptr as *mut u64).cast::<c_void>(),
            (&mut cursors_len as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_scatter_sorted_pairs_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn moe_build_expert_tile_offsets_tensor(
        &self,
        stream: &CudaStream,
        block_m: u32,
        counts_ptr: u64,
        tile_offsets_ptr: u64,
        tile_total_ptr: u64,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut block_m = block_m;
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut tile_offsets_ptr = tile_offsets_ptr;
        let mut tile_offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut tile_total_ptr = tile_total_ptr;
        let mut tile_total_len = 1_u64;
        let mut params = [
            (&mut block_m as *mut u32).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut tile_offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_offsets_len as *mut u64).cast::<c_void>(),
            (&mut tile_total_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_total_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_build_expert_tile_offsets_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn moe_build_expert_tiles_tensor(
        &self,
        stream: &CudaStream,
        block_m: u32,
        tile_capacity: u32,
        counts_ptr: u64,
        tile_offsets_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut block_m = block_m;
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut tile_offsets_ptr = tile_offsets_ptr;
        let mut tile_offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut tile_experts_ptr = tile_experts_ptr;
        let mut tile_experts_len = u64::from(tile_capacity);
        let mut tile_starts_ptr = tile_starts_ptr;
        let mut tile_starts_len = u64::from(tile_capacity);
        let mut params = [
            (&mut block_m as *mut u32).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut tile_offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_offsets_len as *mut u64).cast::<c_void>(),
            (&mut tile_experts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_experts_len as *mut u64).cast::<c_void>(),
            (&mut tile_starts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_starts_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_build_expert_tiles_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_f32_tensor(
        &self,
        stream: &CudaStream,
        n_tokens: u32,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        x_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        clamp: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (expert_mid_dim, n_tokens * n_expert, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let elements = u64::from(n_tokens) * u64::from(n_expert) * u64::from(expert_mid_dim);
        let mut n_tokens = n_tokens;
        let mut expert_in_dim = expert_in_dim;
        let mut expert_mid_dim = expert_mid_dim;
        let mut n_expert = n_expert;
        let mut clamp = clamp;
        let mut gate_expert_bytes = gate_expert_bytes;
        let mut gate_row_bytes = gate_row_bytes;
        let mut gate_weights_ptr = gate_weights_ptr;
        let mut gate_weights_len = gate_weight_bytes;
        let mut up_weights_ptr = up_weights_ptr;
        let mut up_weights_len = gate_weight_bytes;
        let mut x_ptr = x_ptr;
        let mut x_len = u64::from(n_tokens) * u64::from(expert_in_dim);
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(n_tokens) * u64::from(n_expert);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(n_tokens) * u64::from(n_expert);
        let mut iq2_grid_ptr = iq2_grid_ptr;
        let mut iq2_grid_len = 256_u64;
        let mut iq2_signs_ptr = iq2_signs_ptr;
        let mut iq2_signs_len = 128_u64;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = elements;
        let mut up_ptr = up_ptr;
        let mut up_len = elements;
        let mut mid_ptr = mid_ptr;
        let mut mid_len = elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut expert_in_dim as *mut u32).cast::<c_void>(),
            (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut clamp as *mut f32).cast::<c_void>(),
            (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
            (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_weights_len as *mut u64).cast::<c_void>(),
            (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut up_weights_len as *mut u64).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
            (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
            (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
            (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
            (&mut gate_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_len as *mut u64).cast::<c_void>(),
            (&mut up_ptr as *mut u64).cast::<c_void>(),
            (&mut up_len as *mut u64).cast::<c_void>(),
            (&mut mid_ptr as *mut u64).cast::<c_void>(),
            (&mut mid_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_gate_up_mid_f32_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_f32_tensor(
        &self,
        stream: &CudaStream,
        n_tokens: u32,
        down_ptr: u64,
        down_weights_ptr: u64,
        mid_ptr: u64,
        selected_ptr: u64,
        down_weight_bytes: u64,
        expert_mid_dim: u32,
        out_dim: u32,
        n_expert: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (out_dim, n_tokens * n_expert, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut expert_mid_dim = expert_mid_dim;
        let mut out_dim = out_dim;
        let mut n_expert = n_expert;
        let mut down_expert_bytes = down_expert_bytes;
        let mut down_row_bytes = down_row_bytes;
        let mut down_weights_ptr = down_weights_ptr;
        let mut down_weights_len = down_weight_bytes;
        let mut mid_ptr = mid_ptr;
        let mut mid_len = u64::from(n_tokens) * u64::from(n_expert) * u64::from(expert_mid_dim);
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(n_tokens) * u64::from(n_expert);
        let mut down_ptr = down_ptr;
        let mut down_len = u64::from(n_tokens) * u64::from(n_expert) * u64::from(out_dim);
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
            (&mut out_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut down_row_bytes as *mut u64).cast::<c_void>(),
            (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut down_weights_len as *mut u64).cast::<c_void>(),
            (&mut mid_ptr as *mut u64).cast::<c_void>(),
            (&mut mid_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut down_ptr as *mut u64).cast::<c_void>(),
            (&mut down_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_down_f32_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_qwarp32_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_quantized_tensor(
                &self.moe_gate_up_mid_qwarp32_kernel,
                false,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                selected_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
                xq_bytes,
                xq_blocks,
                expert_mid_dim,
                n_expert,
                gate_expert_bytes,
                gate_row_bytes,
                false,
                clamp,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_sorted_qwarp32_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_sorted_tensor(
                &self.moe_gate_up_mid_sorted_qwarp32_kernel,
                false,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                sorted_pairs_ptr,
                selected_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
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
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_sorted_p2_qwarp32_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_sorted_tensor(
                &self.moe_gate_up_mid_sorted_p2_qwarp32_kernel,
                true,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                sorted_pairs_ptr,
                selected_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
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
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn moe_gate_up_mid_sorted_tensor(
        &self,
        function: &CudaFunction,
        p2: bool,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        clamp: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (
                if p2 {
                    expert_mid_dim.div_ceil(16)
                } else {
                    expert_mid_dim.div_ceil(32)
                },
                if p2 {
                    pair_count.div_ceil(2)
                } else {
                    pair_count
                },
                1,
            ),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let output_len = u64::from(pair_count) * u64::from(expert_mid_dim);
        let mut xq_blocks = xq_blocks;
        let mut expert_mid_dim = expert_mid_dim;
        let mut n_expert = n_expert;
        let mut pair_count = pair_count;
        let mut clamp = clamp;
        let mut gate_expert_bytes = gate_expert_bytes;
        let mut gate_row_bytes = gate_row_bytes;
        let mut gate_weights_ptr = gate_weights_ptr;
        let mut gate_weights_len = gate_weight_bytes;
        let mut up_weights_ptr = up_weights_ptr;
        let mut up_weights_len = gate_weight_bytes;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = xq_bytes;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(pair_count);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(pair_count);
        let mut iq2_grid_ptr = iq2_grid_ptr;
        let mut iq2_grid_len = 256_u64;
        let mut iq2_signs_ptr = iq2_signs_ptr;
        let mut iq2_signs_len = 128_u64;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = output_len;
        let mut up_ptr = up_ptr;
        let mut up_len = output_len;
        let mut mid_ptr = mid_ptr;
        let mut mid_len = output_len;
        if p2 {
            let mut params = [
                (&mut xq_blocks as *mut u32).cast::<c_void>(),
                (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
                (&mut n_expert as *mut u32).cast::<c_void>(),
                (&mut pair_count as *mut u32).cast::<c_void>(),
                (&mut clamp as *mut f32).cast::<c_void>(),
                (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_weights_len as *mut u64).cast::<c_void>(),
                (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut up_weights_len as *mut u64).cast::<c_void>(),
                (&mut xq_ptr as *mut u64).cast::<c_void>(),
                (&mut xq_len as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
                (&mut selected_ptr as *mut u64).cast::<c_void>(),
                (&mut selected_len as *mut u64).cast::<c_void>(),
                (&mut weights_ptr as *mut u64).cast::<c_void>(),
                (&mut weights_len as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
                (&mut gate_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_len as *mut u64).cast::<c_void>(),
                (&mut up_ptr as *mut u64).cast::<c_void>(),
                (&mut up_len as *mut u64).cast::<c_void>(),
                (&mut mid_ptr as *mut u64).cast::<c_void>(),
                (&mut mid_len as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                cuda_core::launch_kernel_on_stream(
                    function,
                    config.grid_dim,
                    config.block_dim,
                    0,
                    stream,
                    &mut params,
                )
            }
            .is_ok()
        } else {
            let mut params = [
                (&mut xq_blocks as *mut u32).cast::<c_void>(),
                (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
                (&mut n_expert as *mut u32).cast::<c_void>(),
                (&mut clamp as *mut f32).cast::<c_void>(),
                (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_weights_len as *mut u64).cast::<c_void>(),
                (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut up_weights_len as *mut u64).cast::<c_void>(),
                (&mut xq_ptr as *mut u64).cast::<c_void>(),
                (&mut xq_len as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
                (&mut selected_ptr as *mut u64).cast::<c_void>(),
                (&mut selected_len as *mut u64).cast::<c_void>(),
                (&mut weights_ptr as *mut u64).cast::<c_void>(),
                (&mut weights_len as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
                (&mut gate_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_len as *mut u64).cast::<c_void>(),
                (&mut up_ptr as *mut u64).cast::<c_void>(),
                (&mut up_len as *mut u64).cast::<c_void>(),
                (&mut mid_ptr as *mut u64).cast::<c_void>(),
                (&mut mid_len as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                cuda_core::launch_kernel_on_stream(
                    function,
                    config.grid_dim,
                    config.block_dim,
                    0,
                    stream,
                    &mut params,
                )
            }
            .is_ok()
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_expert_tile4_row32_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        tile_capacity: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_expert_tile_row32_tensor(
                &self.moe_gate_up_mid_expert_tile4_row32_kernel,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
                xq_bytes,
                xq_blocks,
                expert_mid_dim,
                n_expert,
                pair_count,
                tile_capacity,
                gate_expert_bytes,
                gate_row_bytes,
                write_aux,
                clamp,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_expert_tile8_row32_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        tile_capacity: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_expert_tile_row32_tensor(
                &self.moe_gate_up_mid_expert_tile8_row32_kernel,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
                xq_bytes,
                xq_blocks,
                expert_mid_dim,
                n_expert,
                pair_count,
                tile_capacity,
                gate_expert_bytes,
                gate_row_bytes,
                write_aux,
                clamp,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn moe_gate_up_mid_expert_tile_row32_tensor(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        tile_capacity: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (expert_mid_dim.div_ceil(32), tile_capacity, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let elements = u64::from(pair_count) * u64::from(expert_mid_dim);
        let mut xq_blocks = xq_blocks;
        let mut expert_mid_dim = expert_mid_dim;
        let mut n_expert = n_expert;
        let mut write_aux = u32::from(write_aux);
        let mut clamp = clamp;
        let mut gate_expert_bytes = gate_expert_bytes;
        let mut gate_row_bytes = gate_row_bytes;
        let mut gate_weights_ptr = gate_weights_ptr;
        let mut gate_weights_len = gate_weight_bytes;
        let mut up_weights_ptr = up_weights_ptr;
        let mut up_weights_len = gate_weight_bytes;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = xq_bytes;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut offsets_ptr = offsets_ptr;
        let mut offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut tile_total_ptr = tile_total_ptr;
        let mut tile_total_len = 1_u64;
        let mut tile_experts_ptr = tile_experts_ptr;
        let mut tile_experts_len = u64::from(tile_capacity);
        let mut tile_starts_ptr = tile_starts_ptr;
        let mut tile_starts_len = u64::from(tile_capacity);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(pair_count);
        let mut iq2_grid_ptr = iq2_grid_ptr;
        let mut iq2_grid_len = 256_u64;
        let mut iq2_signs_ptr = iq2_signs_ptr;
        let mut iq2_signs_len = 128_u64;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = elements;
        let mut up_ptr = up_ptr;
        let mut up_len = elements;
        let mut mid_ptr = mid_ptr;
        let mut mid_len = elements;
        let mut params = [
            (&mut xq_blocks as *mut u32).cast::<c_void>(),
            (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut write_aux as *mut u32).cast::<c_void>(),
            (&mut clamp as *mut f32).cast::<c_void>(),
            (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
            (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_weights_len as *mut u64).cast::<c_void>(),
            (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut up_weights_len as *mut u64).cast::<c_void>(),
            (&mut xq_ptr as *mut u64).cast::<c_void>(),
            (&mut xq_len as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
            (&mut offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut offsets_len as *mut u64).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut tile_total_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_total_len as *mut u64).cast::<c_void>(),
            (&mut tile_experts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_experts_len as *mut u64).cast::<c_void>(),
            (&mut tile_starts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_starts_len as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
            (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
            (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
            (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
            (&mut gate_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_len as *mut u64).cast::<c_void>(),
            (&mut up_ptr as *mut u64).cast::<c_void>(),
            (&mut up_len as *mut u64).cast::<c_void>(),
            (&mut mid_ptr as *mut u64).cast::<c_void>(),
            (&mut mid_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_expert_tile8_rowspan_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        tile_capacity: u32,
        row_span: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_expert_tile_rowspan_tensor(
                &self.moe_gate_up_mid_expert_tile8_rowspan_kernel,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
                xq_bytes,
                xq_blocks,
                expert_mid_dim,
                n_expert,
                pair_count,
                tile_capacity,
                row_span,
                gate_expert_bytes,
                gate_row_bytes,
                write_aux,
                clamp,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_expert_tile8_rowspan_cached_tensor(
        &self,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        tile_capacity: u32,
        row_span: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        unsafe {
            self.moe_gate_up_mid_expert_tile_rowspan_tensor(
                &self.moe_gate_up_mid_expert_tile8_rowspan_cached_kernel,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
                xq_bytes,
                xq_blocks,
                expert_mid_dim,
                n_expert,
                pair_count,
                tile_capacity,
                row_span,
                gate_expert_bytes,
                gate_row_bytes,
                write_aux,
                clamp,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn moe_gate_up_mid_expert_tile_rowspan_tensor(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        tile_capacity: u32,
        row_span: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        if row_span == 0 {
            return false;
        }
        let config = LaunchConfig {
            grid_dim: (expert_mid_dim.div_ceil(row_span), tile_capacity, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let elements = u64::from(pair_count) * u64::from(expert_mid_dim);
        let mut xq_blocks = xq_blocks;
        let mut expert_mid_dim = expert_mid_dim;
        let mut n_expert = n_expert;
        let mut row_span = row_span;
        let mut write_aux = u32::from(write_aux);
        let mut clamp = clamp;
        let mut gate_expert_bytes = gate_expert_bytes;
        let mut gate_row_bytes = gate_row_bytes;
        let mut gate_weights_ptr = gate_weights_ptr;
        let mut gate_weights_len = gate_weight_bytes;
        let mut up_weights_ptr = up_weights_ptr;
        let mut up_weights_len = gate_weight_bytes;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = xq_bytes;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut offsets_ptr = offsets_ptr;
        let mut offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut tile_total_ptr = tile_total_ptr;
        let mut tile_total_len = 1_u64;
        let mut tile_experts_ptr = tile_experts_ptr;
        let mut tile_experts_len = u64::from(tile_capacity);
        let mut tile_starts_ptr = tile_starts_ptr;
        let mut tile_starts_len = u64::from(tile_capacity);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(pair_count);
        let mut iq2_grid_ptr = iq2_grid_ptr;
        let mut iq2_grid_len = 256_u64;
        let mut iq2_signs_ptr = iq2_signs_ptr;
        let mut iq2_signs_len = 128_u64;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = elements;
        let mut up_ptr = up_ptr;
        let mut up_len = elements;
        let mut mid_ptr = mid_ptr;
        let mut mid_len = elements;
        let mut params = [
            (&mut xq_blocks as *mut u32).cast::<c_void>(),
            (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut row_span as *mut u32).cast::<c_void>(),
            (&mut write_aux as *mut u32).cast::<c_void>(),
            (&mut clamp as *mut f32).cast::<c_void>(),
            (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
            (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_weights_len as *mut u64).cast::<c_void>(),
            (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut up_weights_len as *mut u64).cast::<c_void>(),
            (&mut xq_ptr as *mut u64).cast::<c_void>(),
            (&mut xq_len as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
            (&mut offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut offsets_len as *mut u64).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut tile_total_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_total_len as *mut u64).cast::<c_void>(),
            (&mut tile_experts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_experts_len as *mut u64).cast::<c_void>(),
            (&mut tile_starts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_starts_len as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
            (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
            (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
            (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
            (&mut gate_ptr as *mut u64).cast::<c_void>(),
            (&mut gate_len as *mut u64).cast::<c_void>(),
            (&mut up_ptr as *mut u64).cast::<c_void>(),
            (&mut up_len as *mut u64).cast::<c_void>(),
            (&mut mid_ptr as *mut u64).cast::<c_void>(),
            (&mut mid_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_gate_up_mid_decode_tensor(
        &self,
        stream: &CudaStream,
        q4_k: bool,
        write_aux: bool,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        clamp: f32,
    ) -> bool {
        let function = if q4_k {
            &self.moe_gate_up_mid_decode_q4_k_qwarp32_kernel
        } else {
            &self.moe_gate_up_mid_decode_lut_qwarp32_kernel
        };
        unsafe {
            self.moe_gate_up_mid_quantized_tensor(
                function,
                true,
                stream,
                gate_ptr,
                up_ptr,
                mid_ptr,
                gate_weights_ptr,
                up_weights_ptr,
                xq_ptr,
                selected_ptr,
                weights_ptr,
                iq2_grid_ptr,
                iq2_signs_ptr,
                gate_weight_bytes,
                xq_bytes,
                xq_blocks,
                expert_mid_dim,
                n_expert,
                gate_expert_bytes,
                gate_row_bytes,
                write_aux,
                clamp,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn moe_gate_up_mid_quantized_tensor(
        &self,
        function: &CudaFunction,
        includes_write_aux: bool,
        stream: &CudaStream,
        gate_ptr: u64,
        up_ptr: u64,
        mid_ptr: u64,
        gate_weights_ptr: u64,
        up_weights_ptr: u64,
        xq_ptr: u64,
        selected_ptr: u64,
        weights_ptr: u64,
        iq2_grid_ptr: u64,
        iq2_signs_ptr: u64,
        gate_weight_bytes: u64,
        xq_bytes: u64,
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        write_aux: bool,
        clamp: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (expert_mid_dim.div_ceil(128), n_expert, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let elements = u64::from(n_expert) * u64::from(expert_mid_dim);
        let mut write_aux = u32::from(write_aux);
        let mut xq_blocks = xq_blocks;
        let mut expert_mid_dim = expert_mid_dim;
        let mut n_expert = n_expert;
        let mut clamp = clamp;
        let mut gate_expert_bytes = gate_expert_bytes;
        let mut gate_row_bytes = gate_row_bytes;
        let mut gate_weights_ptr = gate_weights_ptr;
        let mut gate_weights_len = gate_weight_bytes;
        let mut up_weights_ptr = up_weights_ptr;
        let mut up_weights_len = gate_weight_bytes;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = xq_bytes;
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(n_expert);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = u64::from(n_expert);
        let mut iq2_grid_ptr = iq2_grid_ptr;
        let mut iq2_grid_len = 256_u64;
        let mut iq2_signs_ptr = iq2_signs_ptr;
        let mut iq2_signs_len = 128_u64;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = elements;
        let mut up_ptr = up_ptr;
        let mut up_len = elements;
        let mut mid_ptr = mid_ptr;
        let mut mid_len = elements;
        if includes_write_aux {
            let mut params = [
                (&mut write_aux as *mut u32).cast::<c_void>(),
                (&mut xq_blocks as *mut u32).cast::<c_void>(),
                (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
                (&mut n_expert as *mut u32).cast::<c_void>(),
                (&mut clamp as *mut f32).cast::<c_void>(),
                (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_weights_len as *mut u64).cast::<c_void>(),
                (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut up_weights_len as *mut u64).cast::<c_void>(),
                (&mut xq_ptr as *mut u64).cast::<c_void>(),
                (&mut xq_len as *mut u64).cast::<c_void>(),
                (&mut selected_ptr as *mut u64).cast::<c_void>(),
                (&mut selected_len as *mut u64).cast::<c_void>(),
                (&mut weights_ptr as *mut u64).cast::<c_void>(),
                (&mut weights_len as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
                (&mut gate_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_len as *mut u64).cast::<c_void>(),
                (&mut up_ptr as *mut u64).cast::<c_void>(),
                (&mut up_len as *mut u64).cast::<c_void>(),
                (&mut mid_ptr as *mut u64).cast::<c_void>(),
                (&mut mid_len as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                cuda_core::launch_kernel_on_stream(
                    function,
                    config.grid_dim,
                    config.block_dim,
                    0,
                    stream,
                    &mut params,
                )
            }
            .is_ok()
        } else {
            let mut params = [
                (&mut xq_blocks as *mut u32).cast::<c_void>(),
                (&mut expert_mid_dim as *mut u32).cast::<c_void>(),
                (&mut n_expert as *mut u32).cast::<c_void>(),
                (&mut clamp as *mut f32).cast::<c_void>(),
                (&mut gate_expert_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_row_bytes as *mut u64).cast::<c_void>(),
                (&mut gate_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_weights_len as *mut u64).cast::<c_void>(),
                (&mut up_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut up_weights_len as *mut u64).cast::<c_void>(),
                (&mut xq_ptr as *mut u64).cast::<c_void>(),
                (&mut xq_len as *mut u64).cast::<c_void>(),
                (&mut selected_ptr as *mut u64).cast::<c_void>(),
                (&mut selected_len as *mut u64).cast::<c_void>(),
                (&mut weights_ptr as *mut u64).cast::<c_void>(),
                (&mut weights_len as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_grid_len as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_ptr as *mut u64).cast::<c_void>(),
                (&mut iq2_signs_len as *mut u64).cast::<c_void>(),
                (&mut gate_ptr as *mut u64).cast::<c_void>(),
                (&mut gate_len as *mut u64).cast::<c_void>(),
                (&mut up_ptr as *mut u64).cast::<c_void>(),
                (&mut up_len as *mut u64).cast::<c_void>(),
                (&mut mid_ptr as *mut u64).cast::<c_void>(),
                (&mut mid_len as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                cuda_core::launch_kernel_on_stream(
                    function,
                    config.grid_dim,
                    config.block_dim,
                    0,
                    stream,
                    &mut params,
                )
            }
            .is_ok()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_qwarp32_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        selected_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (out_dim.div_ceil(32), n_expert, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut midq_blocks = midq_blocks;
        let mut out_dim = out_dim;
        let mut n_expert = n_expert;
        let mut down_expert_bytes = down_expert_bytes;
        let mut down_row_bytes = down_row_bytes;
        let mut down_weights_ptr = down_weights_ptr;
        let mut down_weights_len = down_weight_bytes;
        let mut midq_ptr = midq_ptr;
        let mut midq_len = midq_bytes;
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(n_expert);
        let mut down_ptr = down_ptr;
        let mut down_len = u64::from(n_expert) * u64::from(out_dim);
        let mut params = [
            (&mut midq_blocks as *mut u32).cast::<c_void>(),
            (&mut out_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut down_row_bytes as *mut u64).cast::<c_void>(),
            (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut down_weights_len as *mut u64).cast::<c_void>(),
            (&mut midq_ptr as *mut u64).cast::<c_void>(),
            (&mut midq_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut down_ptr as *mut u64).cast::<c_void>(),
            (&mut down_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_down_qwarp32_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_sorted_qwarp32_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        selected_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        pair_count: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
    ) -> bool {
        unsafe {
            self.moe_down_sorted_tensor(
                &self.moe_down_sorted_qwarp32_kernel,
                false,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                selected_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                pair_count,
                down_expert_bytes,
                down_row_bytes,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_sorted_p2_qwarp32_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        selected_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        pair_count: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
    ) -> bool {
        unsafe {
            self.moe_down_sorted_tensor(
                &self.moe_down_sorted_p2_qwarp32_kernel,
                true,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                selected_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                pair_count,
                down_expert_bytes,
                down_row_bytes,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn moe_down_sorted_tensor(
        &self,
        function: &CudaFunction,
        p2: bool,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        selected_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        pair_count: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (
                if p2 {
                    out_dim.div_ceil(16)
                } else {
                    out_dim.div_ceil(32)
                },
                if p2 {
                    pair_count.div_ceil(2)
                } else {
                    pair_count
                },
                1,
            ),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut midq_blocks = midq_blocks;
        let mut out_dim = out_dim;
        let mut n_expert = n_expert;
        let mut pair_count = pair_count;
        let mut down_expert_bytes = down_expert_bytes;
        let mut down_row_bytes = down_row_bytes;
        let mut down_weights_ptr = down_weights_ptr;
        let mut down_weights_len = down_weight_bytes;
        let mut midq_ptr = midq_ptr;
        let mut midq_len = midq_bytes;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(pair_count);
        let mut down_ptr = down_ptr;
        let mut down_len = u64::from(pair_count) * u64::from(out_dim);
        if p2 {
            let mut params = [
                (&mut midq_blocks as *mut u32).cast::<c_void>(),
                (&mut out_dim as *mut u32).cast::<c_void>(),
                (&mut n_expert as *mut u32).cast::<c_void>(),
                (&mut pair_count as *mut u32).cast::<c_void>(),
                (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
                (&mut down_row_bytes as *mut u64).cast::<c_void>(),
                (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut down_weights_len as *mut u64).cast::<c_void>(),
                (&mut midq_ptr as *mut u64).cast::<c_void>(),
                (&mut midq_len as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
                (&mut selected_ptr as *mut u64).cast::<c_void>(),
                (&mut selected_len as *mut u64).cast::<c_void>(),
                (&mut down_ptr as *mut u64).cast::<c_void>(),
                (&mut down_len as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                cuda_core::launch_kernel_on_stream(
                    function,
                    config.grid_dim,
                    config.block_dim,
                    0,
                    stream,
                    &mut params,
                )
            }
            .is_ok()
        } else {
            let mut params = [
                (&mut midq_blocks as *mut u32).cast::<c_void>(),
                (&mut out_dim as *mut u32).cast::<c_void>(),
                (&mut n_expert as *mut u32).cast::<c_void>(),
                (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
                (&mut down_row_bytes as *mut u64).cast::<c_void>(),
                (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
                (&mut down_weights_len as *mut u64).cast::<c_void>(),
                (&mut midq_ptr as *mut u64).cast::<c_void>(),
                (&mut midq_len as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
                (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
                (&mut selected_ptr as *mut u64).cast::<c_void>(),
                (&mut selected_len as *mut u64).cast::<c_void>(),
                (&mut down_ptr as *mut u64).cast::<c_void>(),
                (&mut down_len as *mut u64).cast::<c_void>(),
            ];
            unsafe {
                cuda_core::launch_kernel_on_stream(
                    function,
                    config.grid_dim,
                    config.block_dim,
                    0,
                    stream,
                    &mut params,
                )
            }
            .is_ok()
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn moe_atomic_output_zero_tensor(
        &self,
        stream: &CudaStream,
        output_ptr: u64,
        count: u64,
    ) -> bool {
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut output_ptr = output_ptr;
        let mut output_len = count;
        let mut count = count;
        let mut params = [
            (&mut output_ptr as *mut u64).cast::<c_void>(),
            (&mut output_len as *mut u64).cast::<c_void>(),
            (&mut count as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_atomic_output_zero_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_expert_tile4_row32_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        unsafe {
            self.moe_down_expert_tile_row32_tensor(
                &self.moe_down_expert_tile4_row32_kernel,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                n_tokens,
                pair_count,
                tile_capacity,
                down_expert_bytes,
                down_row_bytes,
                atomic_out,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_expert_tile8_row32_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        unsafe {
            self.moe_down_expert_tile_row32_tensor(
                &self.moe_down_expert_tile8_row32_kernel,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                n_tokens,
                pair_count,
                tile_capacity,
                down_expert_bytes,
                down_row_bytes,
                atomic_out,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_expert_tile16_row32_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        unsafe {
            self.moe_down_expert_tile_row32_tensor(
                &self.moe_down_expert_tile16_row32_kernel,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                n_tokens,
                pair_count,
                tile_capacity,
                down_expert_bytes,
                down_row_bytes,
                atomic_out,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn moe_down_expert_tile_row32_tensor(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (out_dim.div_ceil(32), tile_capacity, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut midq_blocks = midq_blocks;
        let mut out_dim = out_dim;
        let mut n_expert = n_expert;
        let mut atomic_out = u32::from(atomic_out);
        let mut down_expert_bytes = down_expert_bytes;
        let mut down_row_bytes = down_row_bytes;
        let mut down_weights_ptr = down_weights_ptr;
        let mut down_weights_len = down_weight_bytes;
        let mut midq_ptr = midq_ptr;
        let mut midq_len = midq_bytes;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut offsets_ptr = offsets_ptr;
        let mut offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut tile_total_ptr = tile_total_ptr;
        let mut tile_total_len = 1_u64;
        let mut tile_experts_ptr = tile_experts_ptr;
        let mut tile_experts_len = u64::from(tile_capacity);
        let mut tile_starts_ptr = tile_starts_ptr;
        let mut tile_starts_len = u64::from(tile_capacity);
        let mut down_ptr = down_ptr;
        let mut down_len = if atomic_out != 0 {
            u64::from(n_tokens) * u64::from(out_dim)
        } else {
            u64::from(pair_count) * u64::from(out_dim)
        };
        let mut params = [
            (&mut midq_blocks as *mut u32).cast::<c_void>(),
            (&mut out_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut atomic_out as *mut u32).cast::<c_void>(),
            (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut down_row_bytes as *mut u64).cast::<c_void>(),
            (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut down_weights_len as *mut u64).cast::<c_void>(),
            (&mut midq_ptr as *mut u64).cast::<c_void>(),
            (&mut midq_len as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
            (&mut offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut offsets_len as *mut u64).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut tile_total_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_total_len as *mut u64).cast::<c_void>(),
            (&mut tile_experts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_experts_len as *mut u64).cast::<c_void>(),
            (&mut tile_starts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_starts_len as *mut u64).cast::<c_void>(),
            (&mut down_ptr as *mut u64).cast::<c_void>(),
            (&mut down_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_expert_tile16_rowspan_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        row_span: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        unsafe {
            self.moe_down_expert_tile_rowspan_tensor(
                &self.moe_down_expert_tile16_rowspan_kernel,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                n_tokens,
                pair_count,
                tile_capacity,
                row_span,
                down_expert_bytes,
                down_row_bytes,
                atomic_out,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_expert_tile16_rowspan_cached_tensor(
        &self,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        row_span: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        unsafe {
            self.moe_down_expert_tile_rowspan_tensor(
                &self.moe_down_expert_tile16_rowspan_cached_kernel,
                stream,
                down_ptr,
                down_weights_ptr,
                midq_ptr,
                sorted_pairs_ptr,
                offsets_ptr,
                counts_ptr,
                tile_total_ptr,
                tile_experts_ptr,
                tile_starts_ptr,
                down_weight_bytes,
                midq_bytes,
                midq_blocks,
                out_dim,
                n_expert,
                n_tokens,
                pair_count,
                tile_capacity,
                row_span,
                down_expert_bytes,
                down_row_bytes,
                atomic_out,
            )
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    unsafe fn moe_down_expert_tile_rowspan_tensor(
        &self,
        function: &CudaFunction,
        stream: &CudaStream,
        down_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        sorted_pairs_ptr: u64,
        offsets_ptr: u64,
        counts_ptr: u64,
        tile_total_ptr: u64,
        tile_experts_ptr: u64,
        tile_starts_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
        pair_count: u32,
        tile_capacity: u32,
        row_span: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        atomic_out: bool,
    ) -> bool {
        if row_span == 0 {
            return false;
        }
        let config = LaunchConfig {
            grid_dim: (out_dim.div_ceil(row_span), tile_capacity, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut midq_blocks = midq_blocks;
        let mut out_dim = out_dim;
        let mut n_expert = n_expert;
        let mut row_span = row_span;
        let mut atomic_out = u32::from(atomic_out);
        let mut down_expert_bytes = down_expert_bytes;
        let mut down_row_bytes = down_row_bytes;
        let mut down_weights_ptr = down_weights_ptr;
        let mut down_weights_len = down_weight_bytes;
        let mut midq_ptr = midq_ptr;
        let mut midq_len = midq_bytes;
        let mut sorted_pairs_ptr = sorted_pairs_ptr;
        let mut sorted_pairs_len = u64::from(pair_count);
        let mut offsets_ptr = offsets_ptr;
        let mut offsets_len = ABI_MOE_SORTED_EXPERTS as u64 + 1;
        let mut counts_ptr = counts_ptr;
        let mut counts_len = ABI_MOE_SORTED_EXPERTS as u64;
        let mut tile_total_ptr = tile_total_ptr;
        let mut tile_total_len = 1_u64;
        let mut tile_experts_ptr = tile_experts_ptr;
        let mut tile_experts_len = u64::from(tile_capacity);
        let mut tile_starts_ptr = tile_starts_ptr;
        let mut tile_starts_len = u64::from(tile_capacity);
        let mut down_ptr = down_ptr;
        let mut down_len = if atomic_out != 0 {
            u64::from(n_tokens) * u64::from(out_dim)
        } else {
            u64::from(pair_count) * u64::from(out_dim)
        };
        let mut params = [
            (&mut midq_blocks as *mut u32).cast::<c_void>(),
            (&mut out_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut row_span as *mut u32).cast::<c_void>(),
            (&mut atomic_out as *mut u32).cast::<c_void>(),
            (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut down_row_bytes as *mut u64).cast::<c_void>(),
            (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut down_weights_len as *mut u64).cast::<c_void>(),
            (&mut midq_ptr as *mut u64).cast::<c_void>(),
            (&mut midq_len as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_ptr as *mut u64).cast::<c_void>(),
            (&mut sorted_pairs_len as *mut u64).cast::<c_void>(),
            (&mut offsets_ptr as *mut u64).cast::<c_void>(),
            (&mut offsets_len as *mut u64).cast::<c_void>(),
            (&mut counts_ptr as *mut u64).cast::<c_void>(),
            (&mut counts_len as *mut u64).cast::<c_void>(),
            (&mut tile_total_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_total_len as *mut u64).cast::<c_void>(),
            (&mut tile_experts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_experts_len as *mut u64).cast::<c_void>(),
            (&mut tile_starts_ptr as *mut u64).cast::<c_void>(),
            (&mut tile_starts_len as *mut u64).cast::<c_void>(),
            (&mut down_ptr as *mut u64).cast::<c_void>(),
            (&mut down_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn moe_down_sum6_tensor(
        &self,
        stream: &CudaStream,
        q4_k: bool,
        out_ptr: u64,
        down_weights_ptr: u64,
        midq_ptr: u64,
        selected_ptr: u64,
        down_weight_bytes: u64,
        midq_bytes: u64,
        midq_blocks: u32,
        out_dim: u32,
        down_expert_bytes: u64,
        down_row_bytes: u64,
    ) -> bool {
        let function = if q4_k {
            &self.moe_down_q4_k_sum6_qwarp32_kernel
        } else {
            &self.moe_down_sum6_qwarp32_kernel
        };
        let config = LaunchConfig {
            grid_dim: (out_dim.div_ceil(32), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut midq_blocks = midq_blocks;
        let mut out_dim = out_dim;
        let mut down_expert_bytes = down_expert_bytes;
        let mut down_row_bytes = down_row_bytes;
        let mut down_weights_ptr = down_weights_ptr;
        let mut down_weights_len = down_weight_bytes;
        let mut midq_ptr = midq_ptr;
        let mut midq_len = midq_bytes;
        let mut selected_ptr = selected_ptr;
        let mut selected_len = 6_u64;
        let mut out_ptr = out_ptr;
        let mut out_len = u64::from(out_dim);
        let mut params = [
            (&mut midq_blocks as *mut u32).cast::<c_void>(),
            (&mut out_dim as *mut u32).cast::<c_void>(),
            (&mut down_expert_bytes as *mut u64).cast::<c_void>(),
            (&mut down_row_bytes as *mut u64).cast::<c_void>(),
            (&mut down_weights_ptr as *mut u64).cast::<c_void>(),
            (&mut down_weights_len as *mut u64).cast::<c_void>(),
            (&mut midq_ptr as *mut u64).cast::<c_void>(),
            (&mut midq_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                function,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn moe_sum_tensor(
        &self,
        stream: &CudaStream,
        n_tokens: u32,
        out_ptr: u64,
        down_ptr: u64,
        out_dim: u32,
        n_expert: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: ((n_tokens * out_dim).div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut out_dim = out_dim;
        let mut n_expert = n_expert;
        let mut down_ptr = down_ptr;
        let mut down_len = u64::from(n_tokens) * u64::from(out_dim) * u64::from(n_expert);
        let mut out_ptr = out_ptr;
        let mut out_len = u64::from(n_tokens) * u64::from(out_dim);
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut out_dim as *mut u32).cast::<c_void>(),
            (&mut n_expert as *mut u32).cast::<c_void>(),
            (&mut down_ptr as *mut u64).cast::<c_void>(),
            (&mut down_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.moe_sum_kernel,
                config.grid_dim,
                config.block_dim,
                0,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_decode_mixed_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        mask_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        mask_elements: u64,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        use_comp_mask: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut pos0 = pos0;
        let mut n_raw = n_raw;
        let mut raw_cap = raw_cap;
        let mut raw_start = raw_start;
        let mut n_comp = n_comp;
        let mut window = window;
        let mut ratio = ratio;
        let mut use_comp_mask = use_comp_mask;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = sink_elements;
        let mut q_ptr = q_ptr;
        let mut q_len = output_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut mask_ptr = mask_ptr;
        let mut mask_len = mask_elements;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = output_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_raw as *mut u32).cast::<c_void>(),
            (&mut raw_cap as *mut u32).cast::<c_void>(),
            (&mut raw_start as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut use_comp_mask as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
            (&mut mask_ptr as *mut u64).cast::<c_void>(),
            (&mut mask_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the public wrapper validates every model/tensor span and
        // score-cap-bounded generic launch before submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_decode_mixed_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_decode_mixed_heads8_online_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
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
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut pos0 = pos0;
        let mut n_raw = n_raw;
        let mut raw_cap = raw_cap;
        let mut raw_start = raw_start;
        let mut n_comp = n_comp;
        let mut window = window;
        let mut ratio = ratio;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = sink_elements;
        let mut q_ptr = q_ptr;
        let mut q_len = output_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = output_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_raw as *mut u32).cast::<c_void>(),
            (&mut raw_cap as *mut u32).cast::<c_void>(),
            (&mut raw_start as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: overflow dispatch requires the current-C unmasked 512-wide
        // branch and all input/output/model spans were validated by the ABI.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_decode_mixed_heads8_online_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_prefill_raw_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut window = window;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = sink_elements;
        let mut q_ptr = q_ptr;
        let mut q_len = output_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = output_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_raw_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_prefill_mixed_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        mask_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        mask_elements: u64,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        use_mask: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut n_comp = n_comp;
        let mut window = window;
        let mut ratio = ratio;
        let mut use_mask = use_mask;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = sink_elements;
        let mut q_ptr = q_ptr;
        let mut q_len = output_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut mask_ptr = mask_ptr;
        let mut mask_len = mask_elements;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = output_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut use_mask as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
            (&mut mask_ptr as *mut u64).cast::<c_void>(),
            (&mut mask_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_mixed_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_static_mixed_heads8_online_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head.div_ceil(8), 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut n_comp = n_comp;
        let mut window = window;
        let mut ratio = ratio;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = sink_elements;
        let mut q_ptr = q_ptr;
        let mut q_len = output_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = output_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_static_mixed_heads8_online_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_prefill_pack_mixed_kv_tensor(
        &self,
        stream: &CudaStream,
        raw_ptr: u64,
        comp_ptr: u64,
        dst_ptr: u64,
        n_tokens: u32,
        n_comp: u32,
        head_dim: u32,
    ) -> bool {
        let count = u64::from(n_tokens + n_comp) * u64::from(head_dim);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_tokens = n_tokens;
        let mut n_comp = n_comp;
        let mut head_dim = head_dim;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = u64::from(n_tokens) * u64::from(head_dim);
        let mut comp_ptr = comp_ptr;
        let mut comp_len = u64::from(n_comp) * u64::from(head_dim);
        let mut dst_ptr = dst_ptr;
        let mut dst_len = count;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
            (&mut dst_ptr as *mut u64).cast::<c_void>(),
            (&mut dst_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_pack_mixed_kv_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_prefill_pack_q_heads_tensor(
        &self,
        stream: &CudaStream,
        q_ptr: u64,
        dst_ptr: u64,
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let count = u64::from(n_tokens) * u64::from(n_head) * u64::from(head_dim);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_tokens = n_tokens;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut q_ptr = q_ptr;
        let mut q_len = count;
        let mut dst_ptr = dst_ptr;
        let mut dst_len = count;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut dst_ptr as *mut u64).cast::<c_void>(),
            (&mut dst_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_pack_q_heads_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_prefill_replicate_kv_tensor(
        &self,
        stream: &CudaStream,
        kv_ptr: u64,
        keys_ptr: u64,
        keys_transposed_ptr: u64,
        n_keys: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let kv_len = u64::from(n_keys) * u64::from(head_dim);
        let count = u64::from(n_head) * kv_len;
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_keys = n_keys;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = kv_len;
        let mut keys_ptr = keys_ptr;
        let mut keys_len = count;
        let mut keys_transposed_ptr = keys_transposed_ptr;
        let mut keys_transposed_len = count;
        let mut params = [
            (&mut n_keys as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut keys_ptr as *mut u64).cast::<c_void>(),
            (&mut keys_len as *mut u64).cast::<c_void>(),
            (&mut keys_transposed_ptr as *mut u64).cast::<c_void>(),
            (&mut keys_transposed_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_replicate_kv_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_prefill_raw_softmax_tensor(
        &self,
        stream: &CudaStream,
        sinks_ptr: u64,
        scores_ptr: u64,
        n_tokens: u32,
        window: u32,
        n_keys: u32,
        n_head: u32,
    ) -> bool {
        let score_len = u64::from(n_head) * u64::from(n_tokens) * u64::from(n_keys);
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut window = window;
        let mut n_keys = n_keys;
        let mut n_head = n_head;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = u64::from(n_head);
        let mut scores_ptr = scores_ptr;
        let mut scores_len = score_len;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut n_keys as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut scores_ptr as *mut u64).cast::<c_void>(),
            (&mut scores_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_raw_softmax_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_prefill_mixed_softmax_tensor(
        &self,
        stream: &CudaStream,
        sinks_ptr: u64,
        mask_ptr: u64,
        scores_ptr: u64,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_keys: u32,
        n_head: u32,
        use_mask: u32,
    ) -> bool {
        let score_len = u64::from(n_head) * u64::from(n_tokens) * u64::from(n_keys);
        let config = LaunchConfig {
            grid_dim: (n_tokens, n_head, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut n_comp = n_comp;
        let mut window = window;
        let mut ratio = ratio;
        let mut n_keys = n_keys;
        let mut n_head = n_head;
        let mut use_mask = use_mask;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = u64::from(n_head);
        let mut mask_ptr = mask_ptr;
        let mut mask_len = u64::from(n_tokens) * u64::from(n_comp);
        let mut scores_ptr = scores_ptr;
        let mut scores_len = score_len;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut n_keys as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut use_mask as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut mask_ptr as *mut u64).cast::<c_void>(),
            (&mut mask_len as *mut u64).cast::<c_void>(),
            (&mut scores_ptr as *mut u64).cast::<c_void>(),
            (&mut scores_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_mixed_softmax_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_prefill_unpack_heads_tensor(
        &self,
        stream: &CudaStream,
        tmp_ptr: u64,
        heads_ptr: u64,
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
    ) -> bool {
        let count = u64::from(n_tokens) * u64::from(n_head) * u64::from(head_dim);
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_tokens = n_tokens;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut tmp_ptr = tmp_ptr;
        let mut tmp_len = count;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = count;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut tmp_ptr as *mut u64).cast::<c_void>(),
            (&mut tmp_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_prefill_unpack_heads_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn indexed_topk_sort_512_asc_tensor(
        &self,
        stream: &CudaStream,
        source_ptr: u64,
        sorted_ptr: u64,
        elements: u64,
        n_tokens: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_tokens, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut n_tokens = n_tokens;
        let mut source_ptr = source_ptr;
        let mut source_len = elements;
        let mut sorted_ptr = sorted_ptr;
        let mut sorted_len = elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut source_ptr as *mut u64).cast::<c_void>(),
            (&mut source_len as *mut u64).cast::<c_void>(),
            (&mut sorted_ptr as *mut u64).cast::<c_void>(),
            (&mut sorted_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.indexed_topk_sort_512_asc_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_indexed_mixed_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        topk_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        topk_elements: u64,
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
    ) -> bool {
        unsafe {
            self.launch_attention_indexed(
                &self.attention_indexed_mixed_kernel,
                (n_tokens, n_head, 1),
                (THREADS_PER_BLOCK, 1, 1),
                stream,
                heads_ptr,
                sinks_ptr,
                q_ptr,
                raw_ptr,
                comp_ptr,
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
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_indexed_mixed_heads8_online_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        topk_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        topk_elements: u64,
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
    ) -> bool {
        unsafe {
            self.launch_attention_indexed(
                &self.attention_indexed_mixed_heads8_online_kernel,
                (n_tokens, (n_head + 15) / 16, 1),
                (512, 1, 1),
                stream,
                heads_ptr,
                sinks_ptr,
                q_ptr,
                raw_ptr,
                comp_ptr,
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
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_indexed_mixed_heads8_rb4_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        topk_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        topk_elements: u64,
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
    ) -> bool {
        unsafe {
            self.launch_attention_indexed(
                &self.attention_indexed_mixed_heads8_rb4_kernel,
                (n_tokens, (n_head + 7) / 8, 1),
                (THREADS_PER_BLOCK, 1, 1),
                stream,
                heads_ptr,
                sinks_ptr,
                q_ptr,
                raw_ptr,
                comp_ptr,
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
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_attention_indexed(
        &self,
        kernel: &CudaFunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        stream: &CudaStream,
        heads_ptr: u64,
        sinks_ptr: u64,
        q_ptr: u64,
        raw_ptr: u64,
        comp_ptr: u64,
        topk_ptr: u64,
        output_elements: u64,
        sink_elements: u64,
        raw_elements: u64,
        comp_elements: u64,
        topk_elements: u64,
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
    ) -> bool {
        let mut n_tokens = n_tokens;
        let mut pos0 = pos0;
        let mut n_raw = n_raw;
        let mut raw_cap = raw_cap;
        let mut raw_start = raw_start;
        let mut n_comp = n_comp;
        let mut top_k = top_k;
        let mut window = window;
        let mut ratio = ratio;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut sinks_ptr = sinks_ptr;
        let mut sinks_len = sink_elements;
        let mut q_ptr = q_ptr;
        let mut q_len = output_elements;
        let mut raw_ptr = raw_ptr;
        let mut raw_len = raw_elements;
        let mut comp_ptr = comp_ptr;
        let mut comp_len = comp_elements;
        let mut topk_ptr = topk_ptr;
        let mut topk_len = topk_elements;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = output_elements;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut pos0 as *mut u32).cast::<c_void>(),
            (&mut n_raw as *mut u32).cast::<c_void>(),
            (&mut raw_cap as *mut u32).cast::<c_void>(),
            (&mut raw_start as *mut u32).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut top_k as *mut u32).cast::<c_void>(),
            (&mut window as *mut u32).cast::<c_void>(),
            (&mut ratio as *mut u32).cast::<c_void>(),
            (&mut n_head as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut sinks_ptr as *mut u64).cast::<c_void>(),
            (&mut sinks_len as *mut u64).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut raw_ptr as *mut u64).cast::<c_void>(),
            (&mut raw_len as *mut u64).cast::<c_void>(),
            (&mut comp_ptr as *mut u64).cast::<c_void>(),
            (&mut comp_len as *mut u64).cast::<c_void>(),
            (&mut topk_ptr as *mut u64).cast::<c_void>(),
            (&mut topk_len as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(kernel, grid_dim, block_dim, 0, stream, &mut params)
        }
        .is_ok()
    }

    pub(crate) unsafe fn dsv4_indexer_qat_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        n_rows: u32,
        head_dim: u32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let count = u64::from(n_rows) * u64::from(head_dim);
        let mut n_rows = n_rows;
        let mut head_dim = head_dim;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut params = [
            (&mut n_rows as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the full mutable 128-wide row span and
        // rejects invalid grid and head dimensions before submission.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.indexer_hadamard_fp4_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn indexer_scores_tensor(
        &self,
        stream: &CudaStream,
        path: IndexerScoreKernel,
        scores_ptr: u64,
        q_ptr: u64,
        weights_ptr: u64,
        index_comp_ptr: u64,
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
        causal: bool,
    ) -> bool {
        let q_len = u64::from(n_tokens) * u64::from(n_head) * u64::from(head_dim);
        let weights_len = u64::from(n_tokens) * u64::from(n_head);
        let index_comp_len = u64::from(n_comp) * u64::from(head_dim);
        let scores_len = u64::from(n_tokens) * u64::from(n_comp);
        let mut n_comp = n_comp;
        let mut n_tokens = n_tokens;
        let mut pos0 = pos0;
        let mut n_head = n_head;
        let mut head_dim = head_dim;
        let mut ratio = ratio;
        let mut scale = scale;
        let mut causal = u32::from(causal);
        let mut q_ptr = q_ptr;
        let mut q_len = q_len;
        let mut weights_ptr = weights_ptr;
        let mut weights_len = weights_len;
        let mut index_comp_ptr = index_comp_ptr;
        let mut index_comp_len = index_comp_len;
        let mut scores_ptr = scores_ptr;
        let mut scores_len = scores_len;

        match path {
            IndexerScoreKernel::Scalar => {
                let mut params = [
                    (&mut n_comp as *mut u32).cast::<c_void>(),
                    (&mut n_tokens as *mut u32).cast::<c_void>(),
                    (&mut pos0 as *mut u32).cast::<c_void>(),
                    (&mut n_head as *mut u32).cast::<c_void>(),
                    (&mut head_dim as *mut u32).cast::<c_void>(),
                    (&mut ratio as *mut u32).cast::<c_void>(),
                    (&mut scale as *mut f32).cast::<c_void>(),
                    (&mut causal as *mut u32).cast::<c_void>(),
                    (&mut q_ptr as *mut u64).cast::<c_void>(),
                    (&mut q_len as *mut u64).cast::<c_void>(),
                    (&mut weights_ptr as *mut u64).cast::<c_void>(),
                    (&mut weights_len as *mut u64).cast::<c_void>(),
                    (&mut index_comp_ptr as *mut u64).cast::<c_void>(),
                    (&mut index_comp_len as *mut u64).cast::<c_void>(),
                    (&mut scores_ptr as *mut u64).cast::<c_void>(),
                    (&mut scores_len as *mut u64).cast::<c_void>(),
                ];
                unsafe {
                    cuda_core::launch_kernel_on_stream(
                        &self.indexer_scores_kernel,
                        (n_comp, n_tokens, 1),
                        (THREADS_PER_BLOCK, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                }
                .is_ok()
            }
            IndexerScoreKernel::DirectOne => {
                let mut params = [
                    (&mut n_comp as *mut u32).cast::<c_void>(),
                    (&mut pos0 as *mut u32).cast::<c_void>(),
                    (&mut ratio as *mut u32).cast::<c_void>(),
                    (&mut scale as *mut f32).cast::<c_void>(),
                    (&mut causal as *mut u32).cast::<c_void>(),
                    (&mut q_ptr as *mut u64).cast::<c_void>(),
                    (&mut q_len as *mut u64).cast::<c_void>(),
                    (&mut weights_ptr as *mut u64).cast::<c_void>(),
                    (&mut weights_len as *mut u64).cast::<c_void>(),
                    (&mut index_comp_ptr as *mut u64).cast::<c_void>(),
                    (&mut index_comp_len as *mut u64).cast::<c_void>(),
                    (&mut scores_ptr as *mut u64).cast::<c_void>(),
                    (&mut scores_len as *mut u64).cast::<c_void>(),
                ];
                unsafe {
                    cuda_core::launch_kernel_on_stream(
                        &self.indexer_score_one_direct_kernel,
                        (n_comp, 1, 1),
                        (ABI_INDEXER_DIRECT_THREADS, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                }
                .is_ok()
            }
            IndexerScoreKernel::Wmma
            | IndexerScoreKernel::Wmma32
            | IndexerScoreKernel::Wmma64
            | IndexerScoreKernel::Wmma128 => {
                let (kernel, components, threads) = match path {
                    IndexerScoreKernel::Wmma => (
                        &self.indexer_scores_wmma_kernel,
                        ABI_INDEXER_TILE_COMPONENTS as u32,
                        ABI_INDEXER_WMMA_THREADS,
                    ),
                    IndexerScoreKernel::Wmma32 => (
                        &self.indexer_scores_wmma32_kernel,
                        ABI_INDEXER_WMMA32_COMPONENTS as u32,
                        ABI_INDEXER_WMMA32_THREADS,
                    ),
                    IndexerScoreKernel::Wmma64 => (
                        &self.indexer_scores_wmma64_kernel,
                        ABI_INDEXER_WMMA64_COMPONENTS as u32,
                        ABI_INDEXER_WMMA64_THREADS,
                    ),
                    IndexerScoreKernel::Wmma128 => (
                        &self.indexer_scores_wmma128_kernel,
                        ABI_INDEXER_WMMA128_COMPONENTS as u32,
                        ABI_INDEXER_WMMA128_THREADS,
                    ),
                    _ => unreachable!(),
                };
                let mut params = [
                    (&mut n_comp as *mut u32).cast::<c_void>(),
                    (&mut n_tokens as *mut u32).cast::<c_void>(),
                    (&mut pos0 as *mut u32).cast::<c_void>(),
                    (&mut ratio as *mut u32).cast::<c_void>(),
                    (&mut scale as *mut f32).cast::<c_void>(),
                    (&mut causal as *mut u32).cast::<c_void>(),
                    (&mut q_ptr as *mut u64).cast::<c_void>(),
                    (&mut q_len as *mut u64).cast::<c_void>(),
                    (&mut weights_ptr as *mut u64).cast::<c_void>(),
                    (&mut weights_len as *mut u64).cast::<c_void>(),
                    (&mut index_comp_ptr as *mut u64).cast::<c_void>(),
                    (&mut index_comp_len as *mut u64).cast::<c_void>(),
                    (&mut scores_ptr as *mut u64).cast::<c_void>(),
                    (&mut scores_len as *mut u64).cast::<c_void>(),
                ];
                unsafe {
                    cuda_core::launch_kernel_on_stream(
                        kernel,
                        (
                            n_comp.div_ceil(components),
                            n_tokens.div_ceil(ABI_INDEXER_TILE_TOKENS as u32),
                            1,
                        ),
                        (threads, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                }
                .is_ok()
            }
        }
    }

    pub(crate) fn indexer_topk_packed_dynamic_shared_available(&self) -> bool {
        self.indexer_topk_8192_packed_key_equivalent_kernel
            .set_max_dynamic_shared_memory_size(ABI_INDEXER_TOPK_PACKED_SHARED_KEY_BYTES as i32)
            .is_ok()
    }

    unsafe fn launch_indexer_topk_simple(
        &self,
        kernel: &CudaFunction,
        stream: &CudaStream,
        config: LaunchConfig,
        selected_ptr: u64,
        scores_ptr: u64,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> bool {
        let mut n_comp = n_comp;
        let mut n_tokens = n_tokens;
        let mut top_k = top_k;
        let mut scores_ptr = scores_ptr;
        let mut scores_len = u64::from(n_comp) * u64::from(n_tokens);
        let mut selected_ptr = selected_ptr;
        let mut selected_len = u64::from(top_k) * u64::from(n_tokens);
        let mut params = [
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut top_k as *mut u32).cast::<c_void>(),
            (&mut scores_ptr as *mut u64).cast::<c_void>(),
            (&mut scores_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn indexer_topk_chunked_tree_tensor(
        &self,
        stream: &CudaStream,
        selected_ptr: u64,
        scores_ptr: u64,
        scratch_ptr: u64,
        scratch_len: u64,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> bool {
        let n_chunks = n_comp.div_ceil(ABI_INDEXER_TOPK_4096_SORT_N as u32);
        let Some(initial_stride) = n_chunks.checked_mul(top_k) else {
            return false;
        };
        let Some(mut required) = n_tokens.checked_mul(initial_stride) else {
            return false;
        };
        let mut n_sets = n_chunks;
        while n_sets > ABI_INDEXER_TOPK_MERGE_GROUP {
            n_sets = n_sets.div_ceil(ABI_INDEXER_TOPK_MERGE_GROUP);
            let Some(next_stride) = n_sets.checked_mul(top_k) else {
                return false;
            };
            let Some(next_size) = n_tokens.checked_mul(next_stride) else {
                return false;
            };
            let Some(next_required) = required.checked_add(next_size) else {
                return false;
            };
            required = next_required;
        }
        if scratch_len < u64::from(required) {
            return false;
        }

        let mut n_comp_arg = n_comp;
        let mut n_tokens_arg = n_tokens;
        let mut top_k_arg = top_k;
        let mut candidate_stride = n_chunks * top_k;
        let mut scores_ptr_arg = scores_ptr;
        let mut scores_len = u64::from(n_comp) * u64::from(n_tokens);
        let mut scratch_ptr_arg = scratch_ptr;
        let mut scratch_len_arg = scratch_len;
        let mut chunk_params = [
            (&mut n_comp_arg as *mut u32).cast::<c_void>(),
            (&mut n_tokens_arg as *mut u32).cast::<c_void>(),
            (&mut top_k_arg as *mut u32).cast::<c_void>(),
            (&mut candidate_stride as *mut u32).cast::<c_void>(),
            (&mut scores_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut scores_len as *mut u64).cast::<c_void>(),
            (&mut scratch_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut scratch_len_arg as *mut u64).cast::<c_void>(),
        ];
        if unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.indexer_topk_chunk_pow2_4096_kernel,
                (n_tokens, n_chunks, 1),
                (ABI_INDEXER_TOPK_THREADS, 1, 1),
                0,
                stream,
                &mut chunk_params,
            )
        }
        .is_err()
        {
            return false;
        }

        let mut n_sets = n_chunks;
        let mut current_offset = 0_u32;
        let mut current_stride = n_chunks * top_k;
        let mut current_total = n_tokens * current_stride;
        while n_sets > ABI_INDEXER_TOPK_MERGE_GROUP {
            let next_sets = n_sets.div_ceil(ABI_INDEXER_TOPK_MERGE_GROUP);
            let next_stride = next_sets * top_k;
            let next_offset = current_total;
            let mut n_comp_arg = n_comp;
            let mut n_tokens_arg = n_tokens;
            let mut top_k_arg = top_k;
            let mut n_sets_arg = n_sets;
            let mut merge_group = ABI_INDEXER_TOPK_MERGE_GROUP;
            let mut candidate_offset = current_offset;
            let mut candidate_stride = current_stride;
            let mut out_offset = next_offset;
            let mut out_stride = next_stride;
            let mut scores_ptr_arg = scores_ptr;
            let mut scores_len = u64::from(n_comp) * u64::from(n_tokens);
            let mut scratch_ptr_arg = scratch_ptr;
            let mut scratch_len_arg = scratch_len;
            let mut merge_params = [
                (&mut n_comp_arg as *mut u32).cast::<c_void>(),
                (&mut n_tokens_arg as *mut u32).cast::<c_void>(),
                (&mut top_k_arg as *mut u32).cast::<c_void>(),
                (&mut n_sets_arg as *mut u32).cast::<c_void>(),
                (&mut merge_group as *mut u32).cast::<c_void>(),
                (&mut candidate_offset as *mut u32).cast::<c_void>(),
                (&mut candidate_stride as *mut u32).cast::<c_void>(),
                (&mut out_offset as *mut u32).cast::<c_void>(),
                (&mut out_stride as *mut u32).cast::<c_void>(),
                (&mut scores_ptr_arg as *mut u64).cast::<c_void>(),
                (&mut scores_len as *mut u64).cast::<c_void>(),
                (&mut scratch_ptr_arg as *mut u64).cast::<c_void>(),
                (&mut scratch_len_arg as *mut u64).cast::<c_void>(),
            ];
            if unsafe {
                cuda_core::launch_kernel_on_stream(
                    &self.indexer_topk_tree_merge_pow2_4096_kernel,
                    (n_tokens, next_sets, 1),
                    (ABI_INDEXER_TOPK_THREADS, 1, 1),
                    0,
                    stream,
                    &mut merge_params,
                )
            }
            .is_err()
            {
                return false;
            }
            current_total += n_tokens * next_stride;
            current_offset = next_offset;
            current_stride = next_stride;
            n_sets = next_sets;
        }

        let mut n_comp_arg = n_comp;
        let mut n_tokens_arg = n_tokens;
        let mut top_k_arg = top_k;
        let mut candidate_offset = current_offset;
        let mut candidate_count = n_sets * top_k;
        let mut candidate_stride = current_stride;
        let mut scratch_ptr_arg = scratch_ptr;
        let mut scratch_len_arg = scratch_len;
        let mut scores_ptr_arg = scores_ptr;
        let mut scores_len = u64::from(n_comp) * u64::from(n_tokens);
        let mut selected_ptr_arg = selected_ptr;
        let mut selected_len = u64::from(top_k) * u64::from(n_tokens);
        let mut final_params = [
            (&mut n_comp_arg as *mut u32).cast::<c_void>(),
            (&mut n_tokens_arg as *mut u32).cast::<c_void>(),
            (&mut top_k_arg as *mut u32).cast::<c_void>(),
            (&mut candidate_offset as *mut u32).cast::<c_void>(),
            (&mut candidate_count as *mut u32).cast::<c_void>(),
            (&mut candidate_stride as *mut u32).cast::<c_void>(),
            (&mut scratch_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut scratch_len_arg as *mut u64).cast::<c_void>(),
            (&mut scores_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut scores_len as *mut u64).cast::<c_void>(),
            (&mut selected_ptr_arg as *mut u64).cast::<c_void>(),
            (&mut selected_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.indexer_topk_merge_pow2_4096_kernel,
                (n_tokens, 1, 1),
                (ABI_INDEXER_TOPK_THREADS, 1, 1),
                0,
                stream,
                &mut final_params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn indexer_topk_tensor(
        &self,
        stream: &CudaStream,
        path: IndexerTopkKernel,
        selected_ptr: u64,
        scores_ptr: u64,
        scratch: Option<(u64, u64)>,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> bool {
        let (kernel, block_dim, shared_mem_bytes) = match path {
            IndexerTopkKernel::Scalar => (&self.indexer_topk_kernel, (1, 1, 1), 0),
            IndexerTopkKernel::Topk1024 => (
                &self.indexer_topk_1024_kernel,
                (ABI_INDEXER_TOPK_THREADS, 1, 1),
                0,
            ),
            IndexerTopkKernel::Pow2U32x2048 => (
                &self.indexer_topk_pow2_2048_kernel,
                (ABI_INDEXER_TOPK_THREADS, 1, 1),
                0,
            ),
            IndexerTopkKernel::Pow2U32x4096 => (
                &self.indexer_topk_pow2_4096_kernel,
                (ABI_INDEXER_TOPK_THREADS, 1, 1),
                0,
            ),
            IndexerTopkKernel::Pow2U16x8192 => (
                &self.indexer_topk_pow2_u16_8192_kernel,
                (ABI_INDEXER_TOPK_THREADS, 1, 1),
                0,
            ),
            IndexerTopkKernel::PackedKeyEquivalent => (
                &self.indexer_topk_8192_packed_key_equivalent_kernel,
                (ABI_INDEXER_TOPK_PACKED_THREADS, 1, 1),
                ABI_INDEXER_TOPK_PACKED_SHARED_KEY_BYTES,
            ),
            IndexerTopkKernel::ChunkedTree => {
                let Some((scratch_ptr, scratch_len)) = scratch else {
                    return false;
                };
                return unsafe {
                    self.indexer_topk_chunked_tree_tensor(
                        stream,
                        selected_ptr,
                        scores_ptr,
                        scratch_ptr,
                        scratch_len,
                        n_comp,
                        n_tokens,
                        top_k,
                    )
                };
            }
        };
        unsafe {
            self.launch_indexer_topk_simple(
                kernel,
                stream,
                LaunchConfig {
                    grid_dim: (n_tokens, 1, 1),
                    block_dim,
                    shared_mem_bytes,
                },
                selected_ptr,
                scores_ptr,
                n_comp,
                n_tokens,
                top_k,
            )
        }
    }

    pub(crate) unsafe fn dsv4_topk_mask_tensor(
        &self,
        stream: &CudaStream,
        mask_ptr: u64,
        topk_ptr: u64,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> bool {
        let count = u64::from(n_tokens) * u64::from(n_comp);
        let selected_count = u64::from(n_tokens) * u64::from(top_k);
        let Ok(grid_x) = u32::try_from(
            count
                .max(selected_count)
                .div_ceil(u64::from(THREADS_PER_BLOCK)),
        ) else {
            return false;
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut count = count;
        let mut n_comp = n_comp;
        let mut top_k = top_k;
        let mut topk_ptr = topk_ptr;
        let mut topk_len = selected_count;
        let mut mask_ptr = mask_ptr;
        let mut mask_len = count;
        let mut params = [
            (&mut count as *mut u64).cast::<c_void>(),
            (&mut n_comp as *mut u32).cast::<c_void>(),
            (&mut top_k as *mut u32).cast::<c_void>(),
            (&mut topk_ptr as *mut u64).cast::<c_void>(),
            (&mut topk_len as *mut u64).cast::<c_void>(),
            (&mut mask_ptr as *mut u64).cast::<c_void>(),
            (&mut mask_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates both tensor spans and all dimensions
        // before submitting a launch that writes exactly count mask values.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.topk_mask_kernel,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn dsv4_qkv_rms_norm_rows_tensor(
        &self,
        stream: &CudaStream,
        q_out_ptr: u64,
        q_ptr: u64,
        q_weight_ptr: u64,
        q_n: u32,
        kv_out_ptr: u64,
        kv_ptr: u64,
        kv_weight_ptr: u64,
        kv_n: u32,
        rows: u32,
        eps: f32,
    ) -> bool {
        let config = LaunchConfig {
            grid_dim: (rows, 2, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let q_count = u64::from(q_n) * u64::from(rows);
        let kv_count = u64::from(kv_n) * u64::from(rows);
        let mut q_n = q_n;
        let mut kv_n = kv_n;
        let mut rows = rows;
        let mut eps = eps;
        let mut q_ptr = q_ptr;
        let mut q_len = q_count;
        let mut q_weight_ptr = q_weight_ptr;
        let mut q_weight_len = u64::from(q_n);
        let mut q_out_ptr = q_out_ptr;
        let mut q_out_len = q_count;
        let mut kv_ptr = kv_ptr;
        let mut kv_len = kv_count;
        let mut kv_weight_ptr = kv_weight_ptr;
        let mut kv_weight_len = u64::from(kv_n);
        let mut kv_out_ptr = kv_out_ptr;
        let mut kv_out_len = kv_count;
        let mut params = [
            (&mut q_n as *mut u32).cast::<c_void>(),
            (&mut kv_n as *mut u32).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut eps as *mut f32).cast::<c_void>(),
            (&mut q_ptr as *mut u64).cast::<c_void>(),
            (&mut q_len as *mut u64).cast::<c_void>(),
            (&mut q_weight_ptr as *mut u64).cast::<c_void>(),
            (&mut q_weight_len as *mut u64).cast::<c_void>(),
            (&mut q_out_ptr as *mut u64).cast::<c_void>(),
            (&mut q_out_len as *mut u64).cast::<c_void>(),
            (&mut kv_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_len as *mut u64).cast::<c_void>(),
            (&mut kv_weight_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_weight_len as *mut u64).cast::<c_void>(),
            (&mut kv_out_ptr as *mut u64).cast::<c_void>(),
            (&mut kv_out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates all four tensor spans and both cached
        // model-weight ranges before issuing the fused Q/KV grid.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.dsv4_qkv_rms_norm_rows_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn dequant_q8_f16_tensor(
        &self,
        stream: &CudaStream,
        weights_ptr: u64,
        output_ptr: u64,
        weights_len: u64,
        output_len: u64,
        in_dim: u64,
        out_dim: u64,
    ) -> bool {
        let Some(config) = launch_config(output_len) else {
            return false;
        };
        let mut in_dim = in_dim;
        let mut out_dim = out_dim;
        let mut blocks = in_dim.div_ceil(32);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = weights_len;
        let mut output_ptr = output_ptr;
        let mut output_len = output_len;
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut out_dim as *mut u64).cast::<c_void>(),
            (&mut blocks as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut output_ptr as *mut u64).cast::<c_void>(),
            (&mut output_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates packed-weight and output bounds and
        // retains both buffers through launch submission and subsequent use.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.dequant_q8_0_to_f16_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn dequant_q8_f32_tensor(
        &self,
        stream: &CudaStream,
        weights_ptr: u64,
        output_ptr: u64,
        weights_len: u64,
        output_len: u64,
        in_dim: u64,
        out_dim: u64,
    ) -> bool {
        let Some(config) = launch_config(output_len) else {
            return false;
        };
        let mut in_dim = in_dim;
        let mut out_dim = out_dim;
        let mut blocks = in_dim.div_ceil(32);
        let mut weights_ptr = weights_ptr;
        let mut weights_len = weights_len;
        let mut output_ptr = output_ptr;
        let mut output_len = output_len;
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut out_dim as *mut u64).cast::<c_void>(),
            (&mut blocks as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut output_ptr as *mut u64).cast::<c_void>(),
            (&mut output_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI layer validates packed-weight and output bounds and
        // retains both buffers through launch submission and subsequent use.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.dequant_q8_0_to_f32_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn quantize_q8_f32_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        xq_ptr: u64,
        xscale_ptr: u64,
        in_dim: u64,
        blocks: u64,
        n_tok: u64,
    ) -> bool {
        let Ok(grid_x) = u32::try_from(blocks) else {
            return false;
        };
        let Ok(grid_y) = u32::try_from(n_tok) else {
            return false;
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut in_dim = in_dim;
        let mut blocks = blocks;
        let mut n_tok = n_tok;
        let mut x_ptr = x_ptr;
        let mut x_len = in_dim * n_tok;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = n_tok * blocks * 32;
        let mut xscale_ptr = xscale_ptr;
        let mut xscale_len = n_tok * blocks;
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut blocks as *mut u64).cast::<c_void>(),
            (&mut n_tok as *mut u64).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut xq_ptr as *mut u64).cast::<c_void>(),
            (&mut xq_len as *mut u64).cast::<c_void>(),
            (&mut xscale_ptr as *mut u64).cast::<c_void>(),
            (&mut xscale_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates source activation bounds and retains both
        // Q8 scratch allocations through the subsequent consumer launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.quantize_q8_0_f32_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attention_output_low_q8_tensor(
        &self,
        stream: &CudaStream,
        low_ptr: u64,
        weight_ptr: u64,
        xq_ptr: u64,
        xscale_ptr: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        n_tokens: u32,
        use_dp4a: bool,
    ) -> bool {
        let blocks = group_dim.div_ceil(32);
        let Some(low_dim) = u64::from(n_groups).checked_mul(rank) else {
            return false;
        };
        let Ok(grid_x) = u32::try_from(low_dim.div_ceil(8)) else {
            return false;
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, n_tokens, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut group_dim = group_dim;
        let mut rank = rank;
        let mut n_groups = n_groups;
        let mut n_tokens = n_tokens;
        let mut blocks = blocks;
        let mut use_dp4a = u32::from(use_dp4a);
        let mut weight_ptr = weight_ptr;
        let mut weight_len = low_dim * blocks * 34;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = u64::from(n_tokens) * u64::from(n_groups) * blocks * 32;
        let mut xscale_ptr = xscale_ptr;
        let mut xscale_len = u64::from(n_tokens) * u64::from(n_groups) * blocks;
        let mut low_ptr = low_ptr;
        let mut low_len = u64::from(n_tokens) * low_dim;
        let mut params = [
            (&mut group_dim as *mut u64).cast::<c_void>(),
            (&mut rank as *mut u64).cast::<c_void>(),
            (&mut n_groups as *mut u32).cast::<c_void>(),
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut blocks as *mut u64).cast::<c_void>(),
            (&mut use_dp4a as *mut u32).cast::<c_void>(),
            (&mut weight_ptr as *mut u64).cast::<c_void>(),
            (&mut weight_len as *mut u64).cast::<c_void>(),
            (&mut xq_ptr as *mut u64).cast::<c_void>(),
            (&mut xq_len as *mut u64).cast::<c_void>(),
            (&mut xscale_ptr as *mut u64).cast::<c_void>(),
            (&mut xscale_len as *mut u64).cast::<c_void>(),
            (&mut low_ptr as *mut u64).cast::<c_void>(),
            (&mut low_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates packed output-A weights and output spans,
        // and retains the quantized activation buffers through this launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.grouped_q8_0_a_preq_warp8_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_pack_group_heads_f16_tensor(
        &self,
        stream: &CudaStream,
        heads_ptr: u64,
        packed_ptr: u64,
        n_tokens: u32,
        n_groups: u32,
        group_dim: u64,
    ) -> bool {
        let Some(count) = u64::from(n_groups)
            .checked_mul(u64::from(n_tokens))
            .and_then(|value| value.checked_mul(group_dim))
        else {
            return false;
        };
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_tokens = n_tokens;
        let mut n_groups = n_groups;
        let mut group_dim = group_dim;
        let mut heads_ptr = heads_ptr;
        let mut heads_len = count;
        let mut packed_ptr = packed_ptr;
        let mut packed_len = count;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_groups as *mut u32).cast::<c_void>(),
            (&mut group_dim as *mut u64).cast::<c_void>(),
            (&mut heads_ptr as *mut u64).cast::<c_void>(),
            (&mut heads_len as *mut u64).cast::<c_void>(),
            (&mut packed_ptr as *mut u64).cast::<c_void>(),
            (&mut packed_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_pack_group_heads_f16_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn f16_to_f32_tensor(
        &self,
        stream: &CudaStream,
        input_ptr: u64,
        output_ptr: u64,
        count: u64,
    ) -> bool {
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut count = count;
        let mut input_ptr = input_ptr;
        let mut input_len = count;
        let mut output_ptr = output_ptr;
        let mut output_len = count;
        let mut params = [
            (&mut count as *mut u64).cast::<c_void>(),
            (&mut input_ptr as *mut u64).cast::<c_void>(),
            (&mut input_len as *mut u64).cast::<c_void>(),
            (&mut output_ptr as *mut u64).cast::<c_void>(),
            (&mut output_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.f16_to_f32_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_expand_group_weights_sgemm_tensor(
        &self,
        stream: &CudaStream,
        weights_ptr: u64,
        transposed_ptr: u64,
        n_groups: u32,
        rank: u64,
        group_dim: u64,
    ) -> bool {
        let Some(count) = u64::from(n_groups)
            .checked_mul(rank)
            .and_then(|value| value.checked_mul(group_dim))
        else {
            return false;
        };
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_groups = n_groups;
        let mut rank = rank;
        let mut group_dim = group_dim;
        let mut weights_ptr = weights_ptr;
        let mut weights_len = count;
        let mut transposed_ptr = transposed_ptr;
        let mut transposed_len = count;
        let mut params = [
            (&mut n_groups as *mut u32).cast::<c_void>(),
            (&mut rank as *mut u64).cast::<c_void>(),
            (&mut group_dim as *mut u64).cast::<c_void>(),
            (&mut weights_ptr as *mut u64).cast::<c_void>(),
            (&mut weights_len as *mut u64).cast::<c_void>(),
            (&mut transposed_ptr as *mut u64).cast::<c_void>(),
            (&mut transposed_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_expand_group_weights_sgemm_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    pub(crate) unsafe fn attention_unpack_group_low_tensor(
        &self,
        stream: &CudaStream,
        packed_ptr: u64,
        low_ptr: u64,
        n_tokens: u32,
        n_groups: u32,
        rank: u64,
    ) -> bool {
        let Some(count) = u64::from(n_groups)
            .checked_mul(u64::from(n_tokens))
            .and_then(|value| value.checked_mul(rank))
        else {
            return false;
        };
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut n_tokens = n_tokens;
        let mut n_groups = n_groups;
        let mut rank = rank;
        let mut packed_ptr = packed_ptr;
        let mut packed_len = count;
        let mut low_ptr = low_ptr;
        let mut low_len = count;
        let mut params = [
            (&mut n_tokens as *mut u32).cast::<c_void>(),
            (&mut n_groups as *mut u32).cast::<c_void>(),
            (&mut rank as *mut u64).cast::<c_void>(),
            (&mut packed_ptr as *mut u64).cast::<c_void>(),
            (&mut packed_len as *mut u64).cast::<c_void>(),
            (&mut low_ptr as *mut u64).cast::<c_void>(),
            (&mut low_len as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.attention_unpack_group_low_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul_q8_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        weight_ptr: u64,
        xq_ptr: u64,
        xscale_ptr: u64,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        path: crate::Q8MatmulPath,
        use_dp4a: bool,
    ) -> bool {
        let blocks = in_dim.div_ceil(32);
        let Ok(grid_y) = u32::try_from(n_tok) else {
            return false;
        };
        let (function, config) = match path {
            crate::Q8MatmulPath::PrequantizedWarp8 => {
                let Ok(grid_x) = u32::try_from(out_dim.div_ceil(8)) else {
                    return false;
                };
                (
                    &self.matmul_q8_0_preq_warp8_kernel,
                    LaunchConfig {
                        grid_dim: (grid_x, 1, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
            crate::Q8MatmulPath::PrequantizedBatchWarp8 => {
                let Ok(grid_x) = u32::try_from(out_dim.div_ceil(8)) else {
                    return false;
                };
                (
                    &self.matmul_q8_0_preq_batch_warp8_kernel,
                    LaunchConfig {
                        grid_dim: (grid_x, grid_y, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
            crate::Q8MatmulPath::PrequantizedGeneric => {
                let Ok(grid_x) = u32::try_from(out_dim) else {
                    return false;
                };
                (
                    &self.matmul_q8_0_preq_kernel,
                    LaunchConfig {
                        grid_dim: (grid_x, grid_y, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
            crate::Q8MatmulPath::ExpandedF32Blas | crate::Q8MatmulPath::ExpandedF16Blas => {
                return false;
            }
        };
        let mut in_dim = in_dim;
        let mut out_dim = out_dim;
        let mut n_tok = n_tok;
        let mut blocks = blocks;
        let mut use_dp4a = u32::from(use_dp4a);
        let mut weight_ptr = weight_ptr;
        let mut weight_len = out_dim * blocks * 34;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = n_tok * blocks * 32;
        let mut xscale_ptr = xscale_ptr;
        let mut xscale_len = n_tok * blocks;
        let mut out_ptr = out_ptr;
        let mut out_len = n_tok * out_dim;
        let mut params = match path {
            crate::Q8MatmulPath::PrequantizedWarp8 => vec![
                (&mut in_dim as *mut u64).cast::<c_void>(),
                (&mut out_dim as *mut u64).cast::<c_void>(),
                (&mut blocks as *mut u64).cast::<c_void>(),
                (&mut use_dp4a as *mut u32).cast::<c_void>(),
                (&mut weight_ptr as *mut u64).cast::<c_void>(),
                (&mut weight_len as *mut u64).cast::<c_void>(),
                (&mut xq_ptr as *mut u64).cast::<c_void>(),
                (&mut xq_len as *mut u64).cast::<c_void>(),
                (&mut xscale_ptr as *mut u64).cast::<c_void>(),
                (&mut xscale_len as *mut u64).cast::<c_void>(),
                (&mut out_ptr as *mut u64).cast::<c_void>(),
                (&mut out_len as *mut u64).cast::<c_void>(),
            ],
            crate::Q8MatmulPath::PrequantizedBatchWarp8
            | crate::Q8MatmulPath::PrequantizedGeneric => vec![
                (&mut in_dim as *mut u64).cast::<c_void>(),
                (&mut out_dim as *mut u64).cast::<c_void>(),
                (&mut n_tok as *mut u64).cast::<c_void>(),
                (&mut blocks as *mut u64).cast::<c_void>(),
                (&mut use_dp4a as *mut u32).cast::<c_void>(),
                (&mut weight_ptr as *mut u64).cast::<c_void>(),
                (&mut weight_len as *mut u64).cast::<c_void>(),
                (&mut xq_ptr as *mut u64).cast::<c_void>(),
                (&mut xq_len as *mut u64).cast::<c_void>(),
                (&mut xscale_ptr as *mut u64).cast::<c_void>(),
                (&mut xscale_len as *mut u64).cast::<c_void>(),
                (&mut out_ptr as *mut u64).cast::<c_void>(),
                (&mut out_len as *mut u64).cast::<c_void>(),
            ],
            crate::Q8MatmulPath::ExpandedF32Blas | crate::Q8MatmulPath::ExpandedF16Blas => {
                return false;
            }
        };
        // SAFETY: the ABI validates output, cached packed weights, and
        // retained quantized scratch spans before selecting this native path.
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul_q8_pair_tensor(
        &self,
        stream: &CudaStream,
        out0_ptr: u64,
        out1_ptr: u64,
        weight0_ptr: u64,
        weight1_ptr: u64,
        xq_ptr: u64,
        xscale_ptr: u64,
        in_dim: u64,
        out0_dim: u64,
        out1_dim: u64,
        use_dp4a: bool,
    ) -> bool {
        let blocks = in_dim.div_ceil(32);
        let Ok(grid_x) = u32::try_from(out0_dim.max(out1_dim).div_ceil(8)) else {
            return false;
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut in_dim = in_dim;
        let mut out0_dim = out0_dim;
        let mut out1_dim = out1_dim;
        let mut blocks = blocks;
        let mut use_dp4a = u32::from(use_dp4a);
        let mut weight0_ptr = weight0_ptr;
        let mut weight0_len = out0_dim * blocks * 34;
        let mut weight1_ptr = weight1_ptr;
        let mut weight1_len = out1_dim * blocks * 34;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = blocks * 32;
        let mut xscale_ptr = xscale_ptr;
        let mut xscale_len = blocks;
        let mut out0_ptr = out0_ptr;
        let mut out0_len = out0_dim;
        let mut out1_ptr = out1_ptr;
        let mut out1_len = out1_dim;
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut out0_dim as *mut u64).cast::<c_void>(),
            (&mut out1_dim as *mut u64).cast::<c_void>(),
            (&mut blocks as *mut u64).cast::<c_void>(),
            (&mut use_dp4a as *mut u32).cast::<c_void>(),
            (&mut weight0_ptr as *mut u64).cast::<c_void>(),
            (&mut weight0_len as *mut u64).cast::<c_void>(),
            (&mut weight1_ptr as *mut u64).cast::<c_void>(),
            (&mut weight1_len as *mut u64).cast::<c_void>(),
            (&mut xq_ptr as *mut u64).cast::<c_void>(),
            (&mut xq_len as *mut u64).cast::<c_void>(),
            (&mut xscale_ptr as *mut u64).cast::<c_void>(),
            (&mut xscale_len as *mut u64).cast::<c_void>(),
            (&mut out0_ptr as *mut u64).cast::<c_void>(),
            (&mut out0_len as *mut u64).cast::<c_void>(),
            (&mut out1_ptr as *mut u64).cast::<c_void>(),
            (&mut out1_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates both packed-Q8 ranges and retains
        // prequantized scratch through this paired single-token launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.matmul_q8_0_pair_preq_warp8_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul_q8_hc_expand_tensor(
        &self,
        stream: &CudaStream,
        out_hc_ptr: u64,
        block_out_ptr: u64,
        block_add_ptr: u64,
        residual_hc_ptr: u64,
        split_ptr: u64,
        weight_ptr: u64,
        xq_ptr: u64,
        xscale_ptr: u64,
        in_dim: u64,
        out_dim: u64,
        n_embd: u32,
        n_hc: u32,
        has_add: bool,
        use_dp4a: bool,
    ) -> bool {
        let blocks = in_dim.div_ceil(32);
        let Ok(grid_x) = u32::try_from(out_dim.div_ceil(8)) else {
            return false;
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut in_dim = in_dim;
        let mut out_dim = out_dim;
        let mut n_embd = n_embd;
        let mut n_hc = n_hc;
        let mut blocks = blocks;
        let mut has_add = u32::from(has_add);
        let mut use_dp4a = u32::from(use_dp4a);
        let mut weight_ptr = weight_ptr;
        let mut weight_len = out_dim * blocks * 34;
        let mut xq_ptr = xq_ptr;
        let mut xq_len = blocks * 32;
        let mut xscale_ptr = xscale_ptr;
        let mut xscale_len = blocks;
        let mut block_add_ptr = block_add_ptr;
        let mut block_add_len = out_dim;
        let mut residual_hc_ptr = residual_hc_ptr;
        let mut residual_hc_len = u64::from(n_hc) * u64::from(n_embd);
        let mut split_ptr = split_ptr;
        let mut split_len = u64::from(2 * n_hc + n_hc * n_hc);
        let mut block_out_ptr = block_out_ptr;
        let mut block_out_len = out_dim;
        let mut out_hc_ptr = out_hc_ptr;
        let mut out_hc_len = u64::from(n_hc) * u64::from(n_embd);
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut out_dim as *mut u64).cast::<c_void>(),
            (&mut n_embd as *mut u32).cast::<c_void>(),
            (&mut n_hc as *mut u32).cast::<c_void>(),
            (&mut blocks as *mut u64).cast::<c_void>(),
            (&mut has_add as *mut u32).cast::<c_void>(),
            (&mut use_dp4a as *mut u32).cast::<c_void>(),
            (&mut weight_ptr as *mut u64).cast::<c_void>(),
            (&mut weight_len as *mut u64).cast::<c_void>(),
            (&mut xq_ptr as *mut u64).cast::<c_void>(),
            (&mut xq_len as *mut u64).cast::<c_void>(),
            (&mut xscale_ptr as *mut u64).cast::<c_void>(),
            (&mut xscale_len as *mut u64).cast::<c_void>(),
            (&mut block_add_ptr as *mut u64).cast::<c_void>(),
            (&mut block_add_len as *mut u64).cast::<c_void>(),
            (&mut residual_hc_ptr as *mut u64).cast::<c_void>(),
            (&mut residual_hc_len as *mut u64).cast::<c_void>(),
            (&mut split_ptr as *mut u64).cast::<c_void>(),
            (&mut split_len as *mut u64).cast::<c_void>(),
            (&mut block_out_ptr as *mut u64).cast::<c_void>(),
            (&mut block_out_len as *mut u64).cast::<c_void>(),
            (&mut out_hc_ptr as *mut u64).cast::<c_void>(),
            (&mut out_hc_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates the single-token packed-Q8 and HC spans
        // and retains prequantized scratch through this fused consumer.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.matmul_q8_0_hc_expand_preq_warp8_kernel,
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

    pub(crate) unsafe fn f32_to_f16_tensor(
        &self,
        stream: &CudaStream,
        x_ptr: u64,
        out_ptr: u64,
        count: u64,
    ) -> bool {
        let Some(config) = launch_config(count) else {
            return false;
        };
        let mut count = count;
        let mut x_ptr = x_ptr;
        let mut x_len = count;
        let mut out_ptr = out_ptr;
        let mut out_len = count;
        let mut params = [
            (&mut count as *mut u64).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut out_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates activation bounds and retains the scratch
        // conversion output through every queued BLAS consumer.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.f32_to_f16_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul_f16_pair_ordered_chunks_tensor(
        &self,
        stream: &CudaStream,
        out0_ptr: u64,
        out1_ptr: u64,
        weight0_ptr: u64,
        weight1_ptr: u64,
        x_ptr: u64,
        in_dim: u64,
        out_dim: u64,
    ) -> bool {
        let grid_x = match u32::try_from(out_dim) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut in_dim = in_dim;
        let mut out_dim = out_dim;
        let mut weight0_ptr = weight0_ptr;
        let mut weight0_len = in_dim * out_dim;
        let mut weight1_ptr = weight1_ptr;
        let mut weight1_len = in_dim * out_dim;
        let mut x_ptr = x_ptr;
        let mut x_len = in_dim;
        let mut out0_ptr = out0_ptr;
        let mut out0_len = out_dim;
        let mut out1_ptr = out1_ptr;
        let mut out1_len = out_dim;
        let mut params = [
            (&mut in_dim as *mut u64).cast::<c_void>(),
            (&mut out_dim as *mut u64).cast::<c_void>(),
            (&mut weight0_ptr as *mut u64).cast::<c_void>(),
            (&mut weight0_len as *mut u64).cast::<c_void>(),
            (&mut weight1_ptr as *mut u64).cast::<c_void>(),
            (&mut weight1_len as *mut u64).cast::<c_void>(),
            (&mut x_ptr as *mut u64).cast::<c_void>(),
            (&mut x_len as *mut u64).cast::<c_void>(),
            (&mut out0_ptr as *mut u64).cast::<c_void>(),
            (&mut out0_len as *mut u64).cast::<c_void>(),
            (&mut out1_ptr as *mut u64).cast::<c_void>(),
            (&mut out1_len as *mut u64).cast::<c_void>(),
        ];
        // SAFETY: the ABI validates both single-token outputs, input, and
        // cached model-weight ranges before submitting the paired launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.matmul_f16_pair_ordered_chunks_kernel,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut params,
            )
        }
        .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul_f32_tensor(
        &self,
        stream: &CudaStream,
        out_ptr: u64,
        weight_ptr: u64,
        x_ptr: u64,
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        path: crate::F32ProjectionPath,
    ) -> bool {
        if path == crate::F32ProjectionPath::Blas {
            return false;
        }
        let grid_x = match u32::try_from(out_dim) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let grid_y = match u32::try_from(n_tok) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let config = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
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
        // SAFETY: the ABI validates the single-token output, input, and
        // cached F32 model-weight range before submitting the launch.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.matmul_f32_kernel,
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
    let module = embedded_abi_modules()
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

fn embedded_abi_modules() -> Result<Vec<EmbeddedModule>, EmbeddedModuleError> {
    let modules = embedded_modules_from_current_exe()?;
    if modules
        .iter()
        .any(|module| module.name() == ABI_KERNEL_ARTIFACT)
    {
        return Ok(modules);
    }
    #[cfg(target_os = "linux")]
    if let Some(path) = abi_image_path() {
        return Ok(artifact_bundles_from_binary_path(path)?
            .into_iter()
            .filter_map(EmbeddedModule::new)
            .collect());
    }
    Ok(modules)
}

#[cfg(target_os = "linux")]
fn abi_image_path() -> Option<PathBuf> {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    let found = unsafe {
        libc::dladdr(
            load_abi_module as *const () as *const c_void,
            info.as_mut_ptr(),
        )
    };
    if found == 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) };
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(path.to_bytes())))
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
