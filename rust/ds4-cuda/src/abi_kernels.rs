use std::ffi::c_void;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use cuda_core::embedded::{
    embedded_modules_from_current_exe, ArtifactPayloadKind, EmbeddedModuleError,
};
use cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig};
use cuda_device::{cuda_module, integer, kernel, thread, warp, DisjointSlice, SharedArray};
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
        let norm_scale = 1.0_f32 / (unsafe { PARTIAL[0] } / n_embd as f32 + norm_eps).sqrt();
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

        let scale = 1.0_f32 / (unsafe { PARTIAL[0] } / head_dim as f32 + eps).sqrt();
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
    fp8_kv_quantize_kernel: CudaFunction,
    indexer_hadamard_fp4_kernel: CudaFunction,
    rms_norm_weight_kernel: CudaFunction,
    dequant_q8_0_to_f16_kernel: CudaFunction,
    dequant_q8_0_to_f32_kernel: CudaFunction,
    quantize_q8_0_f32_kernel: CudaFunction,
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
            fp8_kv_quantize_kernel: module
                .load_function("abi_fp8_kv_quantize_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            indexer_hadamard_fp4_kernel: module
                .load_function("abi_indexer_hadamard_fp4_kernel")
                .map_err(AbiKernelLoadError::Driver)?,
            rms_norm_weight_kernel: module
                .load_function("abi_rms_norm_weight_kernel")
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
        let mut pos_stride = 1_u32;
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
