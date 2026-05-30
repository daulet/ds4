#![feature(f16)]

use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicF32},
    cuda_module, kernel, thread, warp, DisjointSlice, SharedArray,
};
use cuda_host::ltoir;
use ds4_cuda::{
    substrate::CudaOxideSubstrate, M14_5C2C2_SCOPE, M14_5C2C3_SCOPE, M14_5C2C4_SCOPE,
    M14_5C2C5_SCOPE, M14_5C2C6_SCOPE, M14_5C2C7_SCOPE, M14_5C2E_SCOPE,
};

const QK_K: usize = 256;
const IQ2_BLOCK_BYTES: usize = 66;
const Q2_BLOCK_BYTES: usize = 84;
const THREADS: u32 = 256;
const MODEL_EXPERTS: usize = 4;
const N_TOKENS: u32 = 4;
const N_ROUTED: u32 = 3;
const PAIR_COUNT: u32 = N_TOKENS * N_ROUTED;
const EXPERT_MID_DIM: u32 = QK_K as u32;
const OUT_DIM: u32 = 35;
const CLAMP: f32 = 0.01;
const CACHED_GATE_MAX_BLOCKS: usize = 16;
const CACHED_DOWN_MAX_BLOCKS: usize = 8;

const IQ2_GRIDS: [u64; 4] = [
    0x0808_0808_0808_0808,
    0x0808_0808_0808_082b,
    0x0808_0808_0808_1919,
    0x0808_0808_0808_2b08,
];
const IQ2_SIGNS: [u8; 4] = [0, 129, 130, 3];

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn moe_gate_up_mid_expert_tile4_row32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        route_weights: &[f32],
        iq2_grids: &[u64],
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
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        if row >= expert_mid_dim {
            return;
        }
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        let row_blocks = ((expert * expert_mid_dim + row) * xq_blocks) as usize;
        let mut entry = 0_u32;
        while entry < 4 {
            let local_pair = local_start + entry;
            if local_pair < counts[expert as usize] {
                let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                let token = pair / n_expert;
                let mut gate = 0.0_f32;
                let mut up = 0.0_f32;
                let mut block = lane;
                while block < xq_blocks {
                    let q8_block = (token * xq_blocks + block) as usize;
                    gate += dev_dot_iq2_xxs_q8_k_block(
                        gate_weights,
                        row_blocks + block as usize,
                        xq_scales,
                        xq_values,
                        q8_block,
                        iq2_grids,
                        iq2_signs,
                    );
                    up += dev_dot_iq2_xxs_q8_k_block(
                        up_weights,
                        row_blocks + block as usize,
                        xq_scales,
                        xq_values,
                        q8_block,
                        iq2_grids,
                        iq2_signs,
                    );
                    block += 8;
                }
                gate = quarter_warp_sum_f32(gate);
                up = quarter_warp_sum_f32(up);
                if lane == 0 {
                    if clamp > 1.0e-6 {
                        if gate > clamp {
                            gate = clamp;
                        }
                        if up > clamp {
                            up = clamp;
                        }
                        if up < -clamp {
                            up = -clamp;
                        }
                    }
                    let offset = (pair * expert_mid_dim + row) as usize;
                    unsafe {
                        *gate_out.get_unchecked_mut(offset) = gate;
                        *up_out.get_unchecked_mut(offset) = up;
                        *mid_out.get_unchecked_mut(offset) =
                            (gate / (1.0 + (-gate).exp())) * up * route_weights[pair as usize];
                    }
                }
            }
            entry += 1;
        }
    }

    #[kernel]
    pub fn zero_kernel(mut output: DisjointSlice<f32>, count: u32) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < count {
            unsafe {
                *output.get_unchecked_mut(index as usize) = 0.0;
            }
        }
    }

    #[kernel]
    pub fn moe_down_expert_tile4_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        atomic_out: &[f32],
        atomic_mode: u32,
        mut down_out: DisjointSlice<f32>,
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
        let row_blocks = ((expert * out_dim + row) * midq_blocks) as usize;
        let mut entry = 0_u32;
        while entry < 4 {
            let local_pair = local_start + entry;
            if local_pair < counts[expert as usize] {
                let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                let mut accumulator = 0.0_f32;
                let mut block = lane;
                while block < midq_blocks {
                    accumulator += dev_dot_q2_k_q8_k_block(
                        down_weights,
                        row_blocks + block as usize,
                        midq_scales,
                        midq_values,
                        midq_bsums,
                        (pair * midq_blocks + block) as usize,
                    );
                    block += 8;
                }
                accumulator = quarter_warp_sum_f32(accumulator);
                if lane == 0 {
                    if atomic_mode != 0 {
                        let token = pair / n_expert;
                        let offset = (token * out_dim + row) as usize;
                        let output = unsafe {
                            &*(atomic_out.as_ptr().add(offset) as *const DeviceAtomicF32)
                        };
                        output.fetch_add(accumulator, AtomicOrdering::Relaxed);
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

    #[kernel]
    pub fn moe_gate_up_mid_expert_tile8_row32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        route_weights: &[f32],
        iq2_grids: &[u64],
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
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        if row >= expert_mid_dim {
            return;
        }
        let expert = tile_experts[tile as usize];
        let local_start = tile_starts[tile as usize];
        let row_blocks = ((expert * expert_mid_dim + row) * xq_blocks) as usize;
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
                    let q8_block = (token * xq_blocks + block) as usize;
                    gate += dev_dot_iq2_xxs_q8_k_block(
                        gate_weights,
                        row_blocks + block as usize,
                        xq_scales,
                        xq_values,
                        q8_block,
                        iq2_grids,
                        iq2_signs,
                    );
                    up += dev_dot_iq2_xxs_q8_k_block(
                        up_weights,
                        row_blocks + block as usize,
                        xq_scales,
                        xq_values,
                        q8_block,
                        iq2_grids,
                        iq2_signs,
                    );
                    block += 8;
                }
                gate = quarter_warp_sum_f32(gate);
                up = quarter_warp_sum_f32(up);
                if lane == 0 {
                    if clamp > 1.0e-6 {
                        if gate > clamp {
                            gate = clamp;
                        }
                        if up > clamp {
                            up = clamp;
                        }
                        if up < -clamp {
                            up = -clamp;
                        }
                    }
                    let offset = (pair * expert_mid_dim + row) as usize;
                    unsafe {
                        *gate_out.get_unchecked_mut(offset) = gate;
                        *up_out.get_unchecked_mut(offset) = up;
                        *mid_out.get_unchecked_mut(offset) =
                            (gate / (1.0 + (-gate).exp())) * up * route_weights[pair as usize];
                    }
                }
            }
            entry += 1;
        }
    }

    #[kernel]
    pub fn moe_gate_up_mid_expert_tile8_rowspan_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        row_span: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        route_weights: &[f32],
        iq2_grids: &[u64],
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
                let row_blocks = ((expert * expert_mid_dim + row) * xq_blocks) as usize;
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
                            let q8_block = (token * xq_blocks + block) as usize;
                            gate += dev_dot_iq2_xxs_q8_k_block(
                                gate_weights,
                                row_blocks + block as usize,
                                xq_scales,
                                xq_values,
                                q8_block,
                                iq2_grids,
                                iq2_signs,
                            );
                            up += dev_dot_iq2_xxs_q8_k_block(
                                up_weights,
                                row_blocks + block as usize,
                                xq_scales,
                                xq_values,
                                q8_block,
                                iq2_grids,
                                iq2_signs,
                            );
                            block += 8;
                        }
                        gate = quarter_warp_sum_f32(gate);
                        up = quarter_warp_sum_f32(up);
                        if lane == 0 {
                            if clamp > 1.0e-6 {
                                if gate > clamp {
                                    gate = clamp;
                                }
                                if up > clamp {
                                    up = clamp;
                                }
                                if up < -clamp {
                                    up = -clamp;
                                }
                            }
                            let offset = (pair * expert_mid_dim + row) as usize;
                            unsafe {
                                *gate_out.get_unchecked_mut(offset) = gate;
                                *up_out.get_unchecked_mut(offset) = up;
                                *mid_out.get_unchecked_mut(offset) = (gate / (1.0 + (-gate).exp()))
                                    * up
                                    * route_weights[pair as usize];
                            }
                        }
                    }
                    entry += 1;
                }
            }
            row_offset += 32;
        }
    }

    #[allow(static_mut_refs)]
    #[kernel]
    pub fn moe_gate_up_mid_expert_tile8_rowspan_cached_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        row_span: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        route_weights: &[f32],
        iq2_grids: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        static mut SXQ_SCALES: SharedArray<f32, { 8 * CACHED_GATE_MAX_BLOCKS }> =
            SharedArray::UNINIT;
        static mut SXQ_VALUES: SharedArray<i8, { 8 * CACHED_GATE_MAX_BLOCKS * QK_K }> =
            SharedArray::UNINIT;
        static mut S_IQ2_GRIDS: SharedArray<u64, { IQ2_GRIDS.len() }> = SharedArray::UNINIT;
        static mut S_IQ2_SIGNS: SharedArray<u8, { IQ2_SIGNS.len() }> = SharedArray::UNINIT;

        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] || xq_blocks as usize > CACHED_GATE_MAX_BLOCKS {
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
        let thread_index = thread::threadIdx_x() as usize;
        let staged_blocks = np as usize * xq_blocks as usize;
        let mut staged_value = thread_index;
        while staged_value < staged_blocks * QK_K {
            let staged_block = staged_value / QK_K;
            let value_index = staged_value - staged_block * QK_K;
            let entry = staged_block / xq_blocks as usize;
            let block = staged_block - entry * xq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let token = pair / n_expert;
            let input_block = token as usize * xq_blocks as usize + block;
            unsafe {
                SXQ_VALUES[staged_block * QK_K + value_index] =
                    xq_values[input_block * QK_K + value_index];
            }
            staged_value += THREADS as usize;
        }
        if thread_index < staged_blocks {
            let entry = thread_index / xq_blocks as usize;
            let block = thread_index - entry * xq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let token = pair / n_expert;
            unsafe {
                SXQ_SCALES[thread_index] = xq_scales[token as usize * xq_blocks as usize + block];
            }
        }
        if thread_index < IQ2_GRIDS.len() {
            unsafe {
                S_IQ2_GRIDS[thread_index] = iq2_grids[thread_index];
            }
        }
        if thread_index < IQ2_SIGNS.len() {
            unsafe {
                S_IQ2_SIGNS[thread_index] = iq2_signs[thread_index];
            }
        }
        thread::sync_threads();
        let mut row_offset = 0_u32;
        while row_offset < row_span {
            let row = thread::blockIdx_x() * row_span + row_lane + row_offset;
            if row < expert_mid_dim {
                let row_blocks = ((expert * expert_mid_dim + row) * xq_blocks) as usize;
                let mut entry = 0_u32;
                while entry < np {
                    let pair =
                        sorted_pairs[(offsets[expert as usize] + local_start + entry) as usize];
                    let mut gate = 0.0_f32;
                    let mut up = 0.0_f32;
                    let mut block = lane;
                    while block < xq_blocks {
                        let staged_block = entry as usize * xq_blocks as usize + block as usize;
                        gate += dev_dot_iq2_xxs_q8_k_cached_block(
                            gate_weights,
                            row_blocks + block as usize,
                            unsafe { SXQ_SCALES[staged_block] },
                            unsafe { SXQ_VALUES.as_ptr() },
                            staged_block * QK_K,
                            unsafe { S_IQ2_GRIDS.as_ptr() },
                            unsafe { S_IQ2_SIGNS.as_ptr() },
                        );
                        up += dev_dot_iq2_xxs_q8_k_cached_block(
                            up_weights,
                            row_blocks + block as usize,
                            unsafe { SXQ_SCALES[staged_block] },
                            unsafe { SXQ_VALUES.as_ptr() },
                            staged_block * QK_K,
                            unsafe { S_IQ2_GRIDS.as_ptr() },
                            unsafe { S_IQ2_SIGNS.as_ptr() },
                        );
                        block += 8;
                    }
                    gate = quarter_warp_sum_f32(gate);
                    up = quarter_warp_sum_f32(up);
                    if lane == 0 {
                        if clamp > 1.0e-6 {
                            if gate > clamp {
                                gate = clamp;
                            }
                            if up > clamp {
                                up = clamp;
                            }
                            if up < -clamp {
                                up = -clamp;
                            }
                        }
                        let offset = (pair * expert_mid_dim + row) as usize;
                        unsafe {
                            *gate_out.get_unchecked_mut(offset) = gate;
                            *up_out.get_unchecked_mut(offset) = up;
                            *mid_out.get_unchecked_mut(offset) =
                                (gate / (1.0 + (-gate).exp())) * up * route_weights[pair as usize];
                        }
                    }
                    entry += 1;
                }
            }
            row_offset += 32;
        }
    }

    #[kernel]
    pub fn moe_down_expert_tile8_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        atomic_out: &[f32],
        atomic_mode: u32,
        mut down_out: DisjointSlice<f32>,
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
        let row_blocks = ((expert * out_dim + row) * midq_blocks) as usize;
        let mut entry = 0_u32;
        while entry < 8 {
            let local_pair = local_start + entry;
            if local_pair < counts[expert as usize] {
                let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                let mut accumulator = 0.0_f32;
                let mut block = lane;
                while block < midq_blocks {
                    accumulator += dev_dot_q2_k_q8_k_block(
                        down_weights,
                        row_blocks + block as usize,
                        midq_scales,
                        midq_values,
                        midq_bsums,
                        (pair * midq_blocks + block) as usize,
                    );
                    block += 8;
                }
                accumulator = quarter_warp_sum_f32(accumulator);
                if lane == 0 {
                    if atomic_mode != 0 {
                        let token = pair / n_expert;
                        let offset = (token * out_dim + row) as usize;
                        let output = unsafe {
                            &*(atomic_out.as_ptr().add(offset) as *const DeviceAtomicF32)
                        };
                        output.fetch_add(accumulator, AtomicOrdering::Relaxed);
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

    #[kernel]
    pub fn moe_down_expert_tile16_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        atomic_out: &[f32],
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
        if local_start & 8 != 0 {
            return;
        }
        let row_blocks = ((expert * out_dim + row) * midq_blocks) as usize;
        let mut entry = 0_u32;
        while entry < 16 {
            let local_pair = local_start + entry;
            if local_pair < counts[expert as usize] {
                let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                let mut accumulator = 0.0_f32;
                let mut block = lane;
                while block < midq_blocks {
                    accumulator += dev_dot_q2_k_q8_k_block(
                        down_weights,
                        row_blocks + block as usize,
                        midq_scales,
                        midq_values,
                        midq_bsums,
                        (pair * midq_blocks + block) as usize,
                    );
                    block += 8;
                }
                accumulator = quarter_warp_sum_f32(accumulator);
                if lane == 0 {
                    let token = pair / n_expert;
                    let offset = (token * out_dim + row) as usize;
                    let output =
                        unsafe { &*(atomic_out.as_ptr().add(offset) as *const DeviceAtomicF32) };
                    output.fetch_add(accumulator, AtomicOrdering::Relaxed);
                }
            }
            entry += 1;
        }
    }

    #[kernel]
    pub fn moe_down_expert_tile16_rowspan_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        row_span: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        atomic_out: &[f32],
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
                let row_blocks = ((expert * out_dim + row) * midq_blocks) as usize;
                let mut entry = 0_u32;
                while entry < 16 {
                    let local_pair = local_start + entry;
                    if local_pair < counts[expert as usize] {
                        let pair = sorted_pairs[(offsets[expert as usize] + local_pair) as usize];
                        let mut accumulator = 0.0_f32;
                        let mut block = lane;
                        while block < midq_blocks {
                            accumulator += dev_dot_q2_k_q8_k_block(
                                down_weights,
                                row_blocks + block as usize,
                                midq_scales,
                                midq_values,
                                midq_bsums,
                                (pair * midq_blocks + block) as usize,
                            );
                            block += 8;
                        }
                        accumulator = quarter_warp_sum_f32(accumulator);
                        if lane == 0 {
                            let token = pair / n_expert;
                            let offset = (token * out_dim + row) as usize;
                            let output = unsafe {
                                &*(atomic_out.as_ptr().add(offset) as *const DeviceAtomicF32)
                            };
                            output.fetch_add(accumulator, AtomicOrdering::Relaxed);
                        }
                    }
                    entry += 1;
                }
            }
            row_offset += 32;
        }
    }

    #[allow(static_mut_refs)]
    #[kernel]
    pub fn moe_down_expert_tile16_rowspan_cached_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        row_span: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
        sorted_pairs: &[u32],
        offsets: &[u32],
        counts: &[u32],
        tile_total: &[u32],
        tile_experts: &[u32],
        tile_starts: &[u32],
        atomic_out: &[f32],
    ) {
        static mut SXQ_SCALES: SharedArray<f32, { 16 * CACHED_DOWN_MAX_BLOCKS }> =
            SharedArray::UNINIT;
        static mut SXQ_VALUES: SharedArray<i8, { 16 * CACHED_DOWN_MAX_BLOCKS * QK_K }> =
            SharedArray::UNINIT;
        static mut SXQ_BSUMS: SharedArray<i32, { 16 * CACHED_DOWN_MAX_BLOCKS * 16 }> =
            SharedArray::UNINIT;

        let tile = thread::blockIdx_y();
        if tile >= tile_total[0] || midq_blocks as usize > CACHED_DOWN_MAX_BLOCKS {
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
        let thread_index = thread::threadIdx_x() as usize;
        let staged_blocks = np as usize * midq_blocks as usize;
        let mut staged_value = thread_index;
        while staged_value < staged_blocks * QK_K {
            let staged_block = staged_value / QK_K;
            let value_index = staged_value - staged_block * QK_K;
            let entry = staged_block / midq_blocks as usize;
            let block = staged_block - entry * midq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let input_block = pair as usize * midq_blocks as usize + block;
            unsafe {
                SXQ_VALUES[staged_block * QK_K + value_index] =
                    midq_values[input_block * QK_K + value_index];
            }
            staged_value += THREADS as usize;
        }
        if thread_index < staged_blocks {
            let entry = thread_index / midq_blocks as usize;
            let block = thread_index - entry * midq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let input_block = pair as usize * midq_blocks as usize + block;
            unsafe {
                SXQ_SCALES[thread_index] = midq_scales[input_block];
            }
        }
        let mut staged_sum = thread_index;
        while staged_sum < staged_blocks * 16 {
            let staged_block = staged_sum / 16;
            let sum_index = staged_sum - staged_block * 16;
            let entry = staged_block / midq_blocks as usize;
            let block = staged_block - entry * midq_blocks as usize;
            let pair =
                sorted_pairs[(offsets[expert as usize] + local_start + entry as u32) as usize];
            let input_block = pair as usize * midq_blocks as usize + block;
            unsafe {
                SXQ_BSUMS[staged_block * 16 + sum_index] = midq_bsums[input_block * 16 + sum_index];
            }
            staged_sum += THREADS as usize;
        }
        thread::sync_threads();
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let mut row_offset = 0_u32;
        while row_offset < row_span {
            let row = thread::blockIdx_x() * row_span + row_lane + row_offset;
            if row < out_dim {
                let row_blocks = ((expert * out_dim + row) * midq_blocks) as usize;
                let mut entry = 0_u32;
                while entry < np {
                    let pair =
                        sorted_pairs[(offsets[expert as usize] + local_start + entry) as usize];
                    let mut accumulator = 0.0_f32;
                    let mut block = lane;
                    while block < midq_blocks {
                        let staged_block = entry as usize * midq_blocks as usize + block as usize;
                        accumulator += dev_dot_q2_k_q8_k_cached_block(
                            down_weights,
                            row_blocks + block as usize,
                            unsafe { SXQ_SCALES[staged_block] },
                            unsafe { SXQ_VALUES.as_ptr() },
                            unsafe { SXQ_BSUMS.as_ptr() },
                            staged_block,
                        );
                        block += 8;
                    }
                    accumulator = quarter_warp_sum_f32(accumulator);
                    if lane == 0 {
                        let token = pair / n_expert;
                        let offset = (token * out_dim + row) as usize;
                        let output = unsafe {
                            &*(atomic_out.as_ptr().add(offset) as *const DeviceAtomicF32)
                        };
                        output.fetch_add(accumulator, AtomicOrdering::Relaxed);
                    }
                    entry += 1;
                }
            }
            row_offset += 32;
        }
    }

    fn dev_dot_iq2_xxs_q8_k_block(
        packed: &[u8],
        block: usize,
        q8_scales: &[f32],
        q8_values: &[i8],
        q8_block: usize,
        iq2_grids: &[u64],
        iq2_signs: &[u8],
    ) -> f32 {
        let base = block * IQ2_BLOCK_BYTES;
        let weight_scale = f16::from_bits(load_u16(packed, base)) as f32;
        let q_base = q8_block * QK_K;
        let mut block_sum = 0_i32;
        let mut ib32 = 0_usize;
        while ib32 < QK_K / 32 {
            let q2 = base + 2 + ib32 * 8;
            let aux_g = load_u16(packed, q2) as u32 | ((load_u16(packed, q2 + 2) as u32) << 16);
            let aux_s = load_u16(packed, q2 + 4) as u32 | ((load_u16(packed, q2 + 6) as u32) << 16);
            let multiplier = (2 * (aux_s >> 28) + 1) as i32;
            let mut subtotal = 0_i32;
            let mut group = 0_u32;
            while group < 4 {
                let grid = iq2_grids[((aux_g >> (8 * group)) & 0xff) as usize];
                let signs = iq2_signs[((aux_s >> (7 * group)) & 127) as usize];
                let mut lane = 0_u32;
                while lane < 8 {
                    let mut value = ((grid >> (8 * lane)) & 0xff) as i32;
                    if signs & (1_u8 << lane) != 0 {
                        value = -value;
                    }
                    subtotal += value
                        * q8_values[q_base + ib32 * 32 + group as usize * 8 + lane as usize] as i32;
                    lane += 1;
                }
                group += 1;
            }
            block_sum += subtotal * multiplier;
            ib32 += 1;
        }
        0.125 * weight_scale * q8_scales[q8_block] * block_sum as f32
    }

    fn dev_dot_iq2_xxs_q8_k_cached_block(
        packed: &[u8],
        block: usize,
        q8_scale: f32,
        q8_values: *const i8,
        q8_base: usize,
        iq2_grids: *const u64,
        iq2_signs: *const u8,
    ) -> f32 {
        let base = block * IQ2_BLOCK_BYTES;
        let weight_scale = f16::from_bits(load_u16(packed, base)) as f32;
        let mut block_sum = 0_i32;
        let mut ib32 = 0_usize;
        while ib32 < QK_K / 32 {
            let q2 = base + 2 + ib32 * 8;
            let aux_g = load_u16(packed, q2) as u32 | ((load_u16(packed, q2 + 2) as u32) << 16);
            let aux_s = load_u16(packed, q2 + 4) as u32 | ((load_u16(packed, q2 + 6) as u32) << 16);
            let multiplier = (2 * (aux_s >> 28) + 1) as i32;
            let mut subtotal = 0_i32;
            let mut group = 0_u32;
            while group < 4 {
                let grid = unsafe { *iq2_grids.add(((aux_g >> (8 * group)) & 0xff) as usize) };
                let signs = unsafe { *iq2_signs.add(((aux_s >> (7 * group)) & 127) as usize) };
                let mut lane = 0_u32;
                while lane < 8 {
                    let mut value = ((grid >> (8 * lane)) & 0xff) as i32;
                    if signs & (1_u8 << lane) != 0 {
                        value = -value;
                    }
                    subtotal += value
                        * unsafe {
                            *q8_values.add(q8_base + ib32 * 32 + group as usize * 8 + lane as usize)
                        } as i32;
                    lane += 1;
                }
                group += 1;
            }
            block_sum += subtotal * multiplier;
            ib32 += 1;
        }
        0.125 * weight_scale * q8_scale * block_sum as f32
    }

    fn dev_dot_q2_k_q8_k_block(
        packed: &[u8],
        block: usize,
        q8_scales: &[f32],
        q8_values: &[i8],
        q8_bsums: &[i32],
        q8_block: usize,
    ) -> f32 {
        let base = block * Q2_BLOCK_BYTES;
        let weight_scale = f16::from_bits(load_u16(packed, base + 80)) as f32;
        let weight_min = f16::from_bits(load_u16(packed, base + 82)) as f32;
        let q_base = q8_block * QK_K;
        let bsum_base = q8_block * 16;
        let mut min_sum = 0_i32;
        let mut scale = 0_usize;
        while scale < 16 {
            min_sum += q8_bsums[bsum_base + scale] * (packed[base + scale] >> 4) as i32;
            scale += 1;
        }
        let mut quant_sum = 0_i32;
        let mut index = 0_usize;
        let mut chunk = 0_usize;
        while chunk < 2 {
            let mut shift = 0_u32;
            let mut group = 0_usize;
            while group < 4 {
                let first_scale = (packed[base + index] & 0x0f) as i32;
                index += 1;
                let second_scale = (packed[base + index] & 0x0f) as i32;
                index += 1;
                let q = base + 16 + chunk * 32;
                let q8 = q_base + chunk * 128 + group * 32;
                let mut lane = 0_usize;
                let mut first = 0_i32;
                let mut second = 0_i32;
                while lane < 16 {
                    first += ((packed[q + lane] >> shift) & 3) as i32 * q8_values[q8 + lane] as i32;
                    second += ((packed[q + 16 + lane] >> shift) & 3) as i32
                        * q8_values[q8 + 16 + lane] as i32;
                    lane += 1;
                }
                quant_sum += first_scale * first + second_scale * second;
                shift += 2;
                group += 1;
            }
            chunk += 1;
        }
        q8_scales[q8_block] * (weight_scale * quant_sum as f32 - weight_min * min_sum as f32)
    }

    fn dev_dot_q2_k_q8_k_cached_block(
        packed: &[u8],
        block: usize,
        q8_scale: f32,
        q8_values: *const i8,
        q8_bsums: *const i32,
        q8_block: usize,
    ) -> f32 {
        let base = block * Q2_BLOCK_BYTES;
        let weight_scale = f16::from_bits(load_u16(packed, base + 80)) as f32;
        let weight_min = f16::from_bits(load_u16(packed, base + 82)) as f32;
        let q_base = q8_block * QK_K;
        let bsum_base = q8_block * 16;
        let mut min_sum = 0_i32;
        let mut scale = 0_usize;
        while scale < 16 {
            min_sum +=
                unsafe { *q8_bsums.add(bsum_base + scale) } * (packed[base + scale] >> 4) as i32;
            scale += 1;
        }
        let mut quant_sum = 0_i32;
        let mut index = 0_usize;
        let mut chunk = 0_usize;
        while chunk < 2 {
            let mut shift = 0_u32;
            let mut group = 0_usize;
            while group < 4 {
                let first_scale = (packed[base + index] & 0x0f) as i32;
                index += 1;
                let second_scale = (packed[base + index] & 0x0f) as i32;
                index += 1;
                let q = base + 16 + chunk * 32;
                let q8 = q_base + chunk * 128 + group * 32;
                let mut lane = 0_usize;
                let mut first = 0_i32;
                let mut second = 0_i32;
                while lane < 16 {
                    first += ((packed[q + lane] >> shift) & 3) as i32
                        * unsafe { *q8_values.add(q8 + lane) } as i32;
                    second += ((packed[q + 16 + lane] >> shift) & 3) as i32
                        * unsafe { *q8_values.add(q8 + 16 + lane) } as i32;
                    lane += 1;
                }
                quant_sum += first_scale * first + second_scale * second;
                shift += 2;
                group += 1;
            }
            chunk += 1;
        }
        q8_scale * (weight_scale * quant_sum as f32 - weight_min * min_sum as f32)
    }

    fn quarter_warp_sum_f32(mut value: f32) -> f32 {
        let mut offset = 4_u32;
        while offset > 0 {
            value += warp::shuffle_xor_f32(value, offset);
            offset >>= 1;
        }
        value
    }

    fn load_u16(values: &[u8], offset: usize) -> u16 {
        values[offset] as u16 | ((values[offset + 1] as u16) << 8)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tile4 = std::env::var_os("DS4_CUDA_MOE_TILE4").is_some();
    let atomic_down = std::env::var_os("DS4_CUDA_MOE_ATOMIC_DOWN").is_some();
    let tile16 = std::env::var_os("DS4_CUDA_MOE_DOWN_TILE16").is_some();
    let gate_rowspan = std::env::var_os("DS4_CUDA_MOE_GATE_ROWSPAN").is_some();
    let down_rowspan = std::env::var_os("DS4_CUDA_MOE_DOWN_ROWSPAN").is_some();
    let shared_cache = std::env::var_os("DS4_CUDA_MOE_SHARED_CACHE").is_some();
    assert!(!tile16 || (atomic_down && !tile4));
    assert!(!gate_rowspan || (!tile4 && ((!atomic_down && !tile16) || shared_cache)));
    assert!(!down_rowspan || (atomic_down && tile16 && !tile4 && (!gate_rowspan || shared_cache)));
    assert!(!shared_cache || (!tile4 && gate_rowspan && atomic_down && tile16 && down_rowspan));
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_routed_moe_tile8_row32_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let gate_values = packed_iq2_weights(3);
    let up_values = packed_iq2_weights(11);
    let down_values = packed_q2_weights(19);
    let selected_values = selected_values();
    let route_values = route_values();
    let metadata = expert_tile_metadata(&selected_values, if tile4 { 4 } else { 8 });
    let down_metadata = if tile16 {
        expert_tile_metadata(&selected_values, 16)
    } else {
        expert_tile_metadata(&selected_values, if tile4 { 4 } else { 8 })
    };
    if tile4 {
        assert!(metadata
            .tile_experts
            .iter()
            .zip(&metadata.tile_starts)
            .any(|(&expert, &start)| expert == 1 && start == 8));
    }
    if tile16 {
        assert_eq!(down_metadata.counts[1], 9);
        assert!(down_metadata
            .tile_experts
            .iter()
            .zip(&down_metadata.tile_starts)
            .any(|(&expert, &start)| expert == 1 && start == 0));
    }
    let xq = expected_quantized_rows(&input_values(), N_TOKENS);
    let expected_gate = expected_gate_up_mid(
        &gate_values,
        &up_values,
        &xq,
        &selected_values,
        &route_values,
    );

    let gate_spans = if gate_rowspan {
        vec![Some(512_u32), Some(1024_u32), Some(2048_u32)]
    } else {
        vec![None]
    };
    for row_span in gate_spans {
        let actual_gate = run_gate_up_mid(
            &substrate,
            &module,
            &gate_values,
            &up_values,
            &xq,
            &metadata,
            &route_values,
            tile4,
            row_span,
            shared_cache,
        )?;
        substrate.flush_commands()?;
        assert_close(&substrate.download(&actual_gate.gate)?, &expected_gate.gate);
        assert_close(&substrate.download(&actual_gate.up)?, &expected_gate.up);
        assert_close(&substrate.download(&actual_gate.mid)?, &expected_gate.mid);
    }

    let midq = expected_quantized_rows(&expected_gate.mid, PAIR_COUNT);
    let expected_down = expected_down(&down_values, &midq, &selected_values);
    let down_spans = if down_rowspan {
        vec![Some(512_u32), Some(1024_u32), Some(2048_u32)]
    } else {
        vec![None]
    };
    for row_span in down_spans {
        let actual_down = run_down(
            &substrate,
            &module,
            &down_values,
            &midq,
            &down_metadata,
            tile4,
            atomic_down,
            tile16,
            row_span,
            shared_cache,
        )?;
        substrate.end_commands()?;
        if atomic_down {
            assert_close(
                &substrate.download(&actual_down)?,
                &expected_atomic_down(&expected_down),
            );
        } else {
            assert_close(&substrate.download(&actual_down)?, &expected_down);
        }
    }
    if atomic_down && !tile16 {
        let alternate_metadata = expert_tile_metadata(&selected_values, if tile4 { 8 } else { 4 });
        let alternate = run_down(
            &substrate,
            &module,
            &down_values,
            &midq,
            &alternate_metadata,
            !tile4,
            true,
            false,
            None,
            false,
        )?;
        substrate.end_commands()?;
        assert_close(
            &substrate.download(&alternate)?,
            &expected_atomic_down(&expected_down),
        );
    }

    let short_tiles = ExpertTileMetadata {
        tile_experts: vec![],
        ..down_metadata
    };
    assert!(matches!(
        run_down(
            &substrate,
            &module,
            &down_values,
            &midq,
            &short_tiles,
            tile4,
            atomic_down,
            tile16,
            if down_rowspan { Some(512) } else { None },
            shared_cache,
        ),
        Err(TileProjectionError::InvalidShape)
    ));

    if shared_cache {
        println!(
            "{{\"milestone\":\"M14.5c2e\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"gate_shared_row512_matches\":true,\"gate_shared_row1024_matches\":true,\"gate_shared_row2048_matches\":true,\"down_shared_row512_matches\":true,\"down_shared_row1024_matches\":true,\"down_shared_row2048_matches\":true,\"shared_q8_input_staging_matches\":true,\"shared_iq2_lut_staging_matches\":true,\"shared_q2_bsum_staging_matches\":true,\"tile8_and_tile16_metadata_retained\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_thread_block_sync\":true,\"uses_device_atomic_f32_fetch_add\":true,\"uses_libdevice_link_path\":true,\"consumes_rowspan_projection_surface\":{},\"owns_shared_cache_specialization\":{},\"owns_gate_and_down_cached_rowspan_dispatch\":{},\"owns_hyperconnection_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2E_SCOPE.consumes_rowspan_projection_surface,
            M14_5C2E_SCOPE.owns_shared_cache_specialization,
            M14_5C2E_SCOPE.owns_gate_and_down_cached_rowspan_dispatch,
            M14_5C2E_SCOPE.owns_hyperconnection_or_runtime_graph,
            M14_5C2E_SCOPE.changes_default_route,
        );
    } else if down_rowspan {
        println!(
            "{{\"milestone\":\"M14.5c2c7\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"down_row512_matches\":true,\"down_row1024_matches\":true,\"down_row2048_matches\":true,\"partial_row_span_matches\":true,\"tile16_descriptor_metadata_retained\":true,\"token_indexed_accumulation_matches\":true,\"device_zero_before_atomic_matches\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_device_atomic_f32_fetch_add\":true,\"consumes_tile16_row32_atomic_surface\":{},\"owns_moe_down_expert_tile16_rowspan_kernel\":{},\"owns_down_row512_row1024_and_row2048_atomic_dispatch\":{},\"owns_shared_cache_specialization\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2C7_SCOPE.consumes_tile16_row32_atomic_surface,
            M14_5C2C7_SCOPE.owns_moe_down_expert_tile16_rowspan_kernel,
            M14_5C2C7_SCOPE.owns_down_row512_row1024_and_row2048_atomic_dispatch,
            M14_5C2C7_SCOPE.owns_shared_cache_specialization,
            M14_5C2C7_SCOPE.owns_q4_k_or_runtime_graph,
            M14_5C2C7_SCOPE.changes_default_route,
        );
    } else if gate_rowspan {
        println!(
            "{{\"milestone\":\"M14.5c2c6\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"gate_row512_matches\":true,\"gate_row1024_matches\":true,\"gate_row2048_matches\":true,\"partial_row_span_matches\":true,\"tile8_descriptor_metadata_retained\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_quarter_warp_shuffle_reduction\":true,\"uses_libdevice_link_path\":true,\"consumes_tile8_row32_projection_surface\":{},\"owns_moe_gate_up_mid_expert_tile8_rowspan_kernel\":{},\"owns_gate_row512_row1024_and_row2048_dispatch\":{},\"owns_down_rowspan_dispatch\":{},\"owns_shared_cache_specialization\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2C6_SCOPE.consumes_tile8_row32_projection_surface,
            M14_5C2C6_SCOPE.owns_moe_gate_up_mid_expert_tile8_rowspan_kernel,
            M14_5C2C6_SCOPE.owns_gate_row512_row1024_and_row2048_dispatch,
            M14_5C2C6_SCOPE.owns_down_rowspan_dispatch,
            M14_5C2C6_SCOPE.owns_shared_cache_specialization,
            M14_5C2C6_SCOPE.owns_q4_k_or_runtime_graph,
            M14_5C2C6_SCOPE.changes_default_route,
        );
    } else if tile16 {
        println!(
            "{{\"milestone\":\"M14.5c2c5\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"tile16_atomic_down_matches\":true,\"tile16_partial_tile_matches\":true,\"gate_tile8_metadata_retained\":true,\"token_indexed_accumulation_matches\":true,\"device_zero_before_atomic_matches\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_device_atomic_f32_fetch_add\":true,\"consumes_atomic_row32_surface\":{},\"owns_moe_down_expert_tile16_row32_kernel\":{},\"owns_tile16_atomic_down_dispatch\":{},\"owns_rowspan_dispatch\":{},\"owns_shared_cache_specialization\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2C5_SCOPE.consumes_atomic_row32_surface,
            M14_5C2C5_SCOPE.owns_moe_down_expert_tile16_row32_kernel,
            M14_5C2C5_SCOPE.owns_tile16_atomic_down_dispatch,
            M14_5C2C5_SCOPE.owns_rowspan_dispatch,
            M14_5C2C5_SCOPE.owns_shared_cache_specialization,
            M14_5C2C5_SCOPE.owns_q4_k_or_runtime_graph,
            M14_5C2C5_SCOPE.changes_default_route,
        );
    } else if atomic_down {
        println!(
            "{{\"milestone\":\"M14.5c2c4\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"tile8_atomic_down_matches\":true,\"tile4_atomic_down_matches\":true,\"token_indexed_accumulation_matches\":true,\"device_zero_before_atomic_matches\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_device_atomic_f32_fetch_add\":true,\"consumes_tile_row32_projection_surface\":{},\"owns_zero_kernel_for_atomic_down\":{},\"owns_tile4_and_tile8_row32_atomic_down_dispatch\":{},\"owns_tile16_or_rowspan_dispatch\":{},\"owns_shared_cache_specialization\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2C4_SCOPE.consumes_tile_row32_projection_surface,
            M14_5C2C4_SCOPE.owns_zero_kernel_for_atomic_down,
            M14_5C2C4_SCOPE.owns_tile4_and_tile8_row32_atomic_down_dispatch,
            M14_5C2C4_SCOPE.owns_tile16_or_rowspan_dispatch,
            M14_5C2C4_SCOPE.owns_shared_cache_specialization,
            M14_5C2C4_SCOPE.owns_q4_k_or_runtime_graph,
            M14_5C2C4_SCOPE.changes_default_route,
        );
    } else if tile4 {
        println!(
            "{{\"milestone\":\"M14.5c2c3\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"tile4_gate_up_matches\":true,\"tile4_down_matches\":true,\"three_tile_expert_matches\":true,\"partial_tile_matches\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_quarter_warp_shuffle_reduction\":true,\"uses_libdevice_link_path\":true,\"consumes_expert_tile_metadata_surface\":{},\"uses_previously_owned_q8_k_inputs\":{},\"owns_moe_gate_up_mid_expert_tile4_row32_kernel\":{},\"owns_moe_down_expert_tile4_row32_non_atomic_surface\":{},\"owns_optional_tile4_row32_projection_dispatch\":{},\"owns_atomic_down_or_rowspan_dispatch\":{},\"owns_shared_cache_specialization\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2C3_SCOPE.consumes_expert_tile_metadata_surface,
            M14_5C2C3_SCOPE.uses_previously_owned_q8_k_inputs,
            M14_5C2C3_SCOPE.owns_moe_gate_up_mid_expert_tile4_row32_kernel,
            M14_5C2C3_SCOPE.owns_moe_down_expert_tile4_row32_non_atomic_surface,
            M14_5C2C3_SCOPE.owns_optional_tile4_row32_projection_dispatch,
            M14_5C2C3_SCOPE.owns_atomic_down_or_rowspan_dispatch,
            M14_5C2C3_SCOPE.owns_shared_cache_specialization,
            M14_5C2C3_SCOPE.owns_q4_k_or_runtime_graph,
            M14_5C2C3_SCOPE.changes_default_route,
        );
    } else {
        println!(
            "{{\"milestone\":\"M14.5c2c2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"tile8_gate_up_matches\":true,\"tile8_down_matches\":true,\"multi_tile_expert_matches\":true,\"partial_tile_matches\":true,\"negative_expert_bucket_zero_matches\":true,\"invalid_shape_rejected\":true,\"uses_quarter_warp_shuffle_reduction\":true,\"uses_libdevice_link_path\":true,\"consumes_expert_tile_metadata_surface\":{},\"uses_previously_owned_q8_k_inputs\":{},\"owns_moe_gate_up_mid_expert_tile8_row32_kernel\":{},\"owns_moe_down_expert_tile8_row32_non_atomic_surface\":{},\"owns_default_tile8_row32_projection_dispatch\":{},\"owns_atomic_down_or_rowspan_dispatch\":{},\"owns_shared_cache_specialization\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2C2_SCOPE.consumes_expert_tile_metadata_surface,
            M14_5C2C2_SCOPE.uses_previously_owned_q8_k_inputs,
            M14_5C2C2_SCOPE.owns_moe_gate_up_mid_expert_tile8_row32_kernel,
            M14_5C2C2_SCOPE.owns_moe_down_expert_tile8_row32_non_atomic_surface,
            M14_5C2C2_SCOPE.owns_default_tile8_row32_projection_dispatch,
            M14_5C2C2_SCOPE.owns_atomic_down_or_rowspan_dispatch,
            M14_5C2C2_SCOPE.owns_shared_cache_specialization,
            M14_5C2C2_SCOPE.owns_q4_k_or_runtime_graph,
            M14_5C2C2_SCOPE.changes_default_route,
        );
    }
    Ok(())
}

struct QuantizedRows {
    scales: Vec<f32>,
    values: Vec<i8>,
    bsums: Vec<i32>,
}

struct GateOutput {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
}

struct ExpectedGateOutput {
    gate: Vec<f32>,
    up: Vec<f32>,
    mid: Vec<f32>,
}

struct ExpertTileMetadata {
    sorted_pairs: Vec<u32>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    tile_total: Vec<u32>,
    tile_experts: Vec<u32>,
    tile_starts: Vec<u32>,
}

fn run_gate_up_mid(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    gate_values: &[u8],
    up_values: &[u8],
    xq: &QuantizedRows,
    metadata: &ExpertTileMetadata,
    route_values: &[f32],
    tile4: bool,
    row_span: Option<u32>,
    shared_cache: bool,
) -> Result<GateOutput, TileProjectionError> {
    validate_metadata(metadata)?;
    let gate_weights = substrate.upload(gate_values)?;
    let up_weights = substrate.upload(up_values)?;
    let scales = substrate.upload(&xq.scales)?;
    let values = substrate.upload(&xq.values)?;
    let sorted_pairs = substrate.upload(&metadata.sorted_pairs)?;
    let offsets = substrate.upload(&metadata.offsets)?;
    let counts = substrate.upload(&metadata.counts)?;
    let tile_total = substrate.upload(&metadata.tile_total)?;
    let tile_experts = substrate.upload(&metadata.tile_experts)?;
    let tile_starts = substrate.upload(&metadata.tile_starts)?;
    let route_weights = substrate.upload(route_values)?;
    let grids = substrate.upload(&IQ2_GRIDS)?;
    let signs = substrate.upload(&IQ2_SIGNS)?;
    let mut gate = substrate.zeroed::<f32>((PAIR_COUNT * EXPERT_MID_DIM) as usize)?;
    let mut up = substrate.zeroed::<f32>((PAIR_COUNT * EXPERT_MID_DIM) as usize)?;
    let mut mid = substrate.zeroed::<f32>((PAIR_COUNT * EXPERT_MID_DIM) as usize)?;
    if let Some(row_span) = row_span {
        if shared_cache {
            module.moe_gate_up_mid_expert_tile8_rowspan_cached_kernel(
                substrate.stream(),
                LaunchConfig {
                    grid_dim: (EXPERT_MID_DIM.div_ceil(row_span), metadata.tile_total[0], 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                1,
                EXPERT_MID_DIM,
                N_ROUTED,
                row_span,
                CLAMP,
                &gate_weights,
                &up_weights,
                &scales,
                &values,
                &sorted_pairs,
                &offsets,
                &counts,
                &tile_total,
                &tile_experts,
                &tile_starts,
                &route_weights,
                &grids,
                &signs,
                &mut gate,
                &mut up,
                &mut mid,
            )?;
        } else {
            module.moe_gate_up_mid_expert_tile8_rowspan_kernel(
                substrate.stream(),
                LaunchConfig {
                    grid_dim: (EXPERT_MID_DIM.div_ceil(row_span), metadata.tile_total[0], 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                1,
                EXPERT_MID_DIM,
                N_ROUTED,
                row_span,
                CLAMP,
                &gate_weights,
                &up_weights,
                &scales,
                &values,
                &sorted_pairs,
                &offsets,
                &counts,
                &tile_total,
                &tile_experts,
                &tile_starts,
                &route_weights,
                &grids,
                &signs,
                &mut gate,
                &mut up,
                &mut mid,
            )?;
        }
    } else if tile4 {
        module.moe_gate_up_mid_expert_tile4_row32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (EXPERT_MID_DIM.div_ceil(32), metadata.tile_total[0], 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            EXPERT_MID_DIM,
            N_ROUTED,
            CLAMP,
            &gate_weights,
            &up_weights,
            &scales,
            &values,
            &sorted_pairs,
            &offsets,
            &counts,
            &tile_total,
            &tile_experts,
            &tile_starts,
            &route_weights,
            &grids,
            &signs,
            &mut gate,
            &mut up,
            &mut mid,
        )?;
    } else {
        module.moe_gate_up_mid_expert_tile8_row32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (EXPERT_MID_DIM.div_ceil(32), metadata.tile_total[0], 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            EXPERT_MID_DIM,
            N_ROUTED,
            CLAMP,
            &gate_weights,
            &up_weights,
            &scales,
            &values,
            &sorted_pairs,
            &offsets,
            &counts,
            &tile_total,
            &tile_experts,
            &tile_starts,
            &route_weights,
            &grids,
            &signs,
            &mut gate,
            &mut up,
            &mut mid,
        )?;
    }
    Ok(GateOutput { gate, up, mid })
}

fn run_down(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    down_values: &[u8],
    midq: &QuantizedRows,
    metadata: &ExpertTileMetadata,
    tile4: bool,
    atomic_mode: bool,
    tile16: bool,
    row_span: Option<u32>,
    shared_cache: bool,
) -> Result<DeviceBuffer<f32>, TileProjectionError> {
    validate_metadata(metadata)?;
    let down_weights = substrate.upload(down_values)?;
    let scales = substrate.upload(&midq.scales)?;
    let values = substrate.upload(&midq.values)?;
    let bsums = substrate.upload(&midq.bsums)?;
    let sorted_pairs = substrate.upload(&metadata.sorted_pairs)?;
    let offsets = substrate.upload(&metadata.offsets)?;
    let counts = substrate.upload(&metadata.counts)?;
    let tile_total = substrate.upload(&metadata.tile_total)?;
    let tile_experts = substrate.upload(&metadata.tile_experts)?;
    let tile_starts = substrate.upload(&metadata.tile_starts)?;
    let mut down = substrate.zeroed::<f32>((PAIR_COUNT * OUT_DIM) as usize)?;
    let mut atomic_output = substrate.zeroed::<f32>((N_TOKENS * OUT_DIM) as usize)?;
    if atomic_mode {
        module.zero_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: ((N_TOKENS * OUT_DIM).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut atomic_output,
            N_TOKENS * OUT_DIM,
        )?;
    }
    if let Some(row_span) = row_span {
        if shared_cache {
            module.moe_down_expert_tile16_rowspan_cached_kernel(
                substrate.stream(),
                LaunchConfig {
                    grid_dim: (OUT_DIM.div_ceil(row_span), metadata.tile_total[0], 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                1,
                OUT_DIM,
                N_ROUTED,
                row_span,
                &down_weights,
                &scales,
                &values,
                &bsums,
                &sorted_pairs,
                &offsets,
                &counts,
                &tile_total,
                &tile_experts,
                &tile_starts,
                &atomic_output,
            )?;
        } else {
            module.moe_down_expert_tile16_rowspan_kernel(
                substrate.stream(),
                LaunchConfig {
                    grid_dim: (OUT_DIM.div_ceil(row_span), metadata.tile_total[0], 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                1,
                OUT_DIM,
                N_ROUTED,
                row_span,
                &down_weights,
                &scales,
                &values,
                &bsums,
                &sorted_pairs,
                &offsets,
                &counts,
                &tile_total,
                &tile_experts,
                &tile_starts,
                &atomic_output,
            )?;
        }
    } else if tile16 {
        module.moe_down_expert_tile16_row32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (OUT_DIM.div_ceil(32), metadata.tile_total[0], 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            OUT_DIM,
            N_ROUTED,
            &down_weights,
            &scales,
            &values,
            &bsums,
            &sorted_pairs,
            &offsets,
            &counts,
            &tile_total,
            &tile_experts,
            &tile_starts,
            &atomic_output,
        )?;
    } else if tile4 {
        module.moe_down_expert_tile4_row32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (OUT_DIM.div_ceil(32), metadata.tile_total[0], 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            OUT_DIM,
            N_ROUTED,
            &down_weights,
            &scales,
            &values,
            &bsums,
            &sorted_pairs,
            &offsets,
            &counts,
            &tile_total,
            &tile_experts,
            &tile_starts,
            &atomic_output,
            atomic_mode as u32,
            &mut down,
        )?;
    } else {
        module.moe_down_expert_tile8_row32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (OUT_DIM.div_ceil(32), metadata.tile_total[0], 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            OUT_DIM,
            N_ROUTED,
            &down_weights,
            &scales,
            &values,
            &bsums,
            &sorted_pairs,
            &offsets,
            &counts,
            &tile_total,
            &tile_experts,
            &tile_starts,
            &atomic_output,
            atomic_mode as u32,
            &mut down,
        )?;
    }
    Ok(if atomic_mode { atomic_output } else { down })
}

fn validate_metadata(metadata: &ExpertTileMetadata) -> Result<(), TileProjectionError> {
    let tile_count = metadata.tile_total.first().copied().unwrap_or(0) as usize;
    if metadata.sorted_pairs.len() < PAIR_COUNT as usize
        || metadata.offsets.len() < 257
        || metadata.counts.len() < 256
        || tile_count == 0
        || metadata.tile_experts.len() < tile_count
        || metadata.tile_starts.len() < tile_count
    {
        return Err(TileProjectionError::InvalidShape);
    }
    Ok(())
}

fn expert_tile_metadata(selected: &[i32], tile_size: u32) -> ExpertTileMetadata {
    let mut counts = vec![0_u32; 256];
    let mut grouped = vec![Vec::<u32>::new(); 256];
    for (pair, &expert) in selected.iter().enumerate() {
        let expert = normalized_expert(expert);
        counts[expert] += 1;
        grouped[expert].push(pair as u32);
    }
    let mut offsets = vec![0_u32; 257];
    let mut sorted_pairs = Vec::with_capacity(selected.len());
    for expert in 0..256 {
        offsets[expert] = sorted_pairs.len() as u32;
        sorted_pairs.extend_from_slice(&grouped[expert]);
    }
    offsets[256] = sorted_pairs.len() as u32;
    let mut tile_experts = Vec::new();
    let mut tile_starts = Vec::new();
    for (expert, &count) in counts.iter().enumerate() {
        let mut start = 0_u32;
        while start < count {
            tile_experts.push(expert as u32);
            tile_starts.push(start);
            start += tile_size;
        }
    }
    ExpertTileMetadata {
        sorted_pairs,
        offsets,
        counts,
        tile_total: vec![tile_experts.len() as u32],
        tile_experts,
        tile_starts,
    }
}

fn expected_gate_up_mid(
    gate_weights: &[u8],
    up_weights: &[u8],
    xq: &QuantizedRows,
    selected: &[i32],
    route_values: &[f32],
) -> ExpectedGateOutput {
    let mut gate = vec![0.0_f32; (PAIR_COUNT * EXPERT_MID_DIM) as usize];
    let mut up = vec![0.0_f32; (PAIR_COUNT * EXPERT_MID_DIM) as usize];
    let mut mid = vec![0.0_f32; (PAIR_COUNT * EXPERT_MID_DIM) as usize];
    for pair in 0..PAIR_COUNT as usize {
        let token = pair / N_ROUTED as usize;
        let expert = normalized_expert(selected[pair]);
        for row in 0..EXPERT_MID_DIM as usize {
            let block = expert * EXPERT_MID_DIM as usize + row;
            let gate_value = iq2_q8_k_dot(gate_weights, block, xq, token).min(CLAMP);
            let up_value = iq2_q8_k_dot(up_weights, block, xq, token).clamp(-CLAMP, CLAMP);
            let offset = pair * EXPERT_MID_DIM as usize + row;
            gate[offset] = gate_value;
            up[offset] = up_value;
            mid[offset] =
                (gate_value / (1.0 + (-gate_value).exp())) * up_value * route_values[pair];
        }
    }
    ExpectedGateOutput { gate, up, mid }
}

fn expected_down(down_weights: &[u8], midq: &QuantizedRows, selected: &[i32]) -> Vec<f32> {
    let mut down = vec![0.0_f32; (PAIR_COUNT * OUT_DIM) as usize];
    for pair in 0..PAIR_COUNT as usize {
        let expert = normalized_expert(selected[pair]);
        for row in 0..OUT_DIM as usize {
            down[pair * OUT_DIM as usize + row] =
                q2_q8_k_dot(down_weights, expert * OUT_DIM as usize + row, midq, pair);
        }
    }
    down
}

fn expected_atomic_down(down: &[f32]) -> Vec<f32> {
    let mut accumulated = vec![0.0_f32; (N_TOKENS * OUT_DIM) as usize];
    for token in 0..N_TOKENS as usize {
        for expert in 0..N_ROUTED as usize {
            let pair = token * N_ROUTED as usize + expert;
            for row in 0..OUT_DIM as usize {
                accumulated[token * OUT_DIM as usize + row] += down[pair * OUT_DIM as usize + row];
            }
        }
    }
    accumulated
}

fn expected_quantized_rows(x: &[f32], n_rows: u32) -> QuantizedRows {
    let mut scales = vec![0.0_f32; n_rows as usize];
    let mut values = vec![0_i8; n_rows as usize * QK_K];
    let mut bsums = vec![0_i32; n_rows as usize * 16];
    for row in 0..n_rows as usize {
        let base = row * QK_K;
        let mut maximum = 0.0_f32;
        let mut max_value = 0.0_f32;
        for &value in &x[base..base + QK_K] {
            if value.abs() > maximum {
                maximum = value.abs();
                max_value = value;
            }
        }
        let inverse = -127.0 / max_value;
        scales[row] = 1.0 / inverse;
        for lane in 0..QK_K {
            values[base + lane] = clamp_i8(round_ties_even(inverse * x[base + lane]));
        }
        for group in 0..16 {
            bsums[row * 16 + group] = values[base + group * 16..base + group * 16 + 16]
                .iter()
                .map(|value| *value as i32)
                .sum();
        }
    }
    QuantizedRows {
        scales,
        values,
        bsums,
    }
}

fn iq2_q8_k_dot(packed: &[u8], block: usize, q8: &QuantizedRows, q8_block: usize) -> f32 {
    let base = block * IQ2_BLOCK_BYTES;
    let d = f16::from_bits(u16::from_le_bytes([packed[base], packed[base + 1]])) as f32;
    let mut block_sum = 0_i32;
    for ib32 in 0..QK_K / 32 {
        let q2 = base + 2 + ib32 * 8;
        let aux_g = u16::from_le_bytes([packed[q2], packed[q2 + 1]]) as u32
            | (u16::from_le_bytes([packed[q2 + 2], packed[q2 + 3]]) as u32) << 16;
        let aux_s = u16::from_le_bytes([packed[q2 + 4], packed[q2 + 5]]) as u32
            | (u16::from_le_bytes([packed[q2 + 6], packed[q2 + 7]]) as u32) << 16;
        let multiplier = (2 * (aux_s >> 28) + 1) as i32;
        let mut subtotal = 0_i32;
        for group in 0..4_u32 {
            let grid = IQ2_GRIDS[((aux_g >> (8 * group)) & 0xff) as usize];
            let signs = IQ2_SIGNS[((aux_s >> (7 * group)) & 127) as usize];
            for lane in 0..8_u32 {
                let mut value = ((grid >> (8 * lane)) & 0xff) as i32;
                if signs & (1 << lane) != 0 {
                    value = -value;
                }
                subtotal += value
                    * q8.values[q8_block * QK_K + ib32 * 32 + group as usize * 8 + lane as usize]
                        as i32;
            }
        }
        block_sum += subtotal * multiplier;
    }
    0.125 * d * q8.scales[q8_block] * block_sum as f32
}

fn q2_q8_k_dot(packed: &[u8], block: usize, q8: &QuantizedRows, q8_block: usize) -> f32 {
    let base = block * Q2_BLOCK_BYTES;
    let d = f16::from_bits(u16::from_le_bytes([packed[base + 80], packed[base + 81]])) as f32;
    let dmin = f16::from_bits(u16::from_le_bytes([packed[base + 82], packed[base + 83]])) as f32;
    let mut min_sum = 0_i32;
    for index in 0..16 {
        min_sum += q8.bsums[q8_block * 16 + index] * (packed[base + index] >> 4) as i32;
    }
    let mut quant_sum = 0_i32;
    let mut scale = 0_usize;
    for chunk in 0..2 {
        for group in 0..4 {
            let shift = group * 2;
            let q = base + 16 + chunk * 32;
            let q8_base = q8_block * QK_K + chunk * 128 + group * 32;
            let first = (0..16)
                .map(|lane| {
                    ((packed[q + lane] >> shift) & 3) as i32 * q8.values[q8_base + lane] as i32
                })
                .sum::<i32>();
            let second = (0..16)
                .map(|lane| {
                    ((packed[q + 16 + lane] >> shift) & 3) as i32
                        * q8.values[q8_base + 16 + lane] as i32
                })
                .sum::<i32>();
            quant_sum += (packed[base + scale] & 0x0f) as i32 * first;
            scale += 1;
            quant_sum += (packed[base + scale] & 0x0f) as i32 * second;
            scale += 1;
        }
    }
    q8.scales[q8_block] * (d * quant_sum as f32 - dmin * min_sum as f32)
}

fn packed_iq2_weights(seed: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(MODEL_EXPERTS * EXPERT_MID_DIM as usize * IQ2_BLOCK_BYTES);
    for expert in 0..MODEL_EXPERTS {
        for row in 0..EXPERT_MID_DIM as usize {
            let scale =
                (0.001953125_f32 + ((expert + row + seed) % 4) as f32 * 0.0009765625) as f16;
            packed.extend_from_slice(&scale.to_bits().to_le_bytes());
            for ib32 in 0..QK_K / 32 {
                let grid0 = (expert + row + ib32 + seed) % IQ2_GRIDS.len();
                let aux_g = grid0 as u32
                    | (((grid0 + 1) % 4) as u32) << 8
                    | (((grid0 + 2) % 4) as u32) << 16
                    | (((grid0 + 3) % 4) as u32) << 24;
                let sign0 = (row + ib32 + seed) % IQ2_SIGNS.len();
                let aux_s = sign0 as u32
                    | (((sign0 + 1) % 4) as u32) << 7
                    | (((sign0 + 2) % 4) as u32) << 14
                    | (((sign0 + 3) % 4) as u32) << 21
                    | (((expert + row + ib32 + seed) % 3) as u32) << 28;
                packed.extend_from_slice(&(aux_g as u16).to_le_bytes());
                packed.extend_from_slice(&((aux_g >> 16) as u16).to_le_bytes());
                packed.extend_from_slice(&(aux_s as u16).to_le_bytes());
                packed.extend_from_slice(&((aux_s >> 16) as u16).to_le_bytes());
            }
        }
    }
    packed
}

fn packed_q2_weights(seed: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(MODEL_EXPERTS * OUT_DIM as usize * Q2_BLOCK_BYTES);
    for expert in 0..MODEL_EXPERTS {
        for row in 0..OUT_DIM as usize {
            for scale in 0..16 {
                packed.push(
                    (1 + ((expert + row + scale + seed) % 5) as u8)
                        | (((expert * 3 + row + scale + seed) % 4) as u8) << 4,
                );
            }
            for lane in 0..64 {
                packed.push(((expert * 17 + row * 11 + lane * 7 + seed) % 256) as u8);
            }
            let d = (0.00390625_f32 + ((expert + row + seed) % 3) as f32 * 0.001953125) as f16;
            let dmin = (0.001953125_f32 + ((row + seed) % 2) as f32 * 0.0009765625) as f16;
            packed.extend_from_slice(&d.to_bits().to_le_bytes());
            packed.extend_from_slice(&dmin.to_bits().to_le_bytes());
        }
    }
    packed
}

fn selected_values() -> Vec<i32> {
    vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 2, -1, 3]
}

fn route_values() -> Vec<f32> {
    vec![
        0.48, 0.33, 0.25, 0.2, 0.15, 0.09, 0.5, 0.21, 0.17, 0.3, 0.12, 0.08,
    ]
}

fn input_values() -> Vec<f32> {
    let mut values = Vec::with_capacity((N_TOKENS as usize) * QK_K);
    for token in 0..N_TOKENS as usize {
        values.extend((0..QK_K).map(|index| {
            let magnitude = ((index * 13 + token * 17 + 5) % 29) as f32 * 0.0078125 + 0.015625;
            if (index + token) % 3 == 0 {
                -magnitude
            } else {
                magnitude
            }
        }));
        values[token * QK_K + 17 + token] = if token % 2 == 0 {
            -0.75 - token as f32 * 0.01
        } else {
            0.75 + token as f32 * 0.01
        };
    }
    values
}

fn normalized_expert(expert: i32) -> usize {
    if expert < 0 {
        0
    } else {
        expert as usize
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 5.0e-4,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
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

fn clamp_i8(value: i32) -> i8 {
    value.clamp(-128, 127) as i8
}

#[derive(Debug)]
enum TileProjectionError {
    InvalidShape,
    Driver(DriverError),
}

impl From<DriverError> for TileProjectionError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl fmt::Display for TileProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("tile row32 projection input shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TileProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
