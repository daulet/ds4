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
    hc_weighted_sum_kernel: CudaFunction,
    hc_expand_kernel: CudaFunction,
    directional_steering_project_kernel: CudaFunction,
    swiglu_kernel: CudaFunction,
    rms_norm_plain_kernel: CudaFunction,
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
