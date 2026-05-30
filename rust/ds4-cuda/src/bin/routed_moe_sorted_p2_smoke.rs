#![feature(f16)]

use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32},
    cuda_module, kernel, thread, warp, DisjointSlice, SharedArray,
};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5C2B2_SCOPE, M14_5C2F_SCOPE};

const QK_K: usize = 256;
const IQ2_BLOCK_BYTES: usize = 66;
const Q2_BLOCK_BYTES: usize = 84;
const THREADS: u32 = 256;
const SORTED_EXPERTS: usize = 256;
const MODEL_EXPERTS: usize = 4;
const N_TOKENS: u32 = 2;
const N_ROUTED: u32 = 3;
const PAIR_COUNT: u32 = N_TOKENS * N_ROUTED;
const EXPERT_IN_DIM: u32 = QK_K as u32;
const EXPERT_MID_DIM: u32 = QK_K as u32;
const OUT_DIM: u32 = 35;
const CLAMP: f32 = 0.01;

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
    pub fn q8_k_quantize_kernel(
        in_dim: u32,
        n_rows: u32,
        x: &[f32],
        mut scales: DisjointSlice<f32>,
        mut qs: DisjointSlice<i8>,
        mut bsums: DisjointSlice<i32>,
    ) {
        static mut ABS_PART: SharedArray<f32, QK_K> = SharedArray::UNINIT;
        static mut VAL_PART: SharedArray<f32, QK_K> = SharedArray::UNINIT;
        static mut Q_PART: SharedArray<i32, QK_K> = SharedArray::UNINIT;
        static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
        static mut ISCALE: SharedArray<f32, 1> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let row = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        let blocks = in_dim / QK_K as u32;
        if row >= n_rows || block >= blocks || lane >= THREADS {
            return;
        }
        let input = (row * in_dim + block * QK_K as u32 + lane) as usize;
        let value = x[input];
        let magnitude = if value < 0.0 { -value } else { value };
        unsafe {
            ABS_PART[lane as usize] = magnitude;
            VAL_PART[lane as usize] = value;
        }
        thread::sync_threads();
        let mut stride = THREADS >> 1;
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
        let output_block = (row * blocks + block) as usize;
        if unsafe { ABS_PART[0] } == 0.0 {
            unsafe {
                if lane == 0 {
                    *scales.get_unchecked_mut(output_block) = 0.0;
                }
                *qs.get_unchecked_mut(output_block * QK_K + lane as usize) = 0;
                if lane < 16 {
                    *bsums.get_unchecked_mut(output_block * 16 + lane as usize) = 0;
                }
            }
            return;
        }
        if lane == 0 {
            unsafe {
                ISCALE[0] = -127.0 / VAL_PART[0];
                SCALE[0] = 1.0 / ISCALE[0];
                *scales.get_unchecked_mut(output_block) = SCALE[0];
            }
        }
        thread::sync_threads();
        let quantized = clamp_i8(round_ties_even(unsafe { ISCALE[0] } * value));
        unsafe {
            Q_PART[lane as usize] = quantized as i32;
            *qs.get_unchecked_mut(output_block * QK_K + lane as usize) = quantized;
        }
        thread::sync_threads();
        if lane < 16 {
            let base = lane as usize * 16;
            let mut sum = 0_i32;
            let mut index = 0_usize;
            while index < 16 {
                sum += unsafe { Q_PART[base + index] };
                index += 1;
            }
            unsafe {
                *bsums.get_unchecked_mut(output_block * 16 + lane as usize) = sum;
            }
        }
    }

    #[kernel]
    pub fn moe_count_sorted_pairs_kernel(pair_count: u32, selected: &[i32], counts: &[u32]) {
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
    pub fn moe_prefix_sorted_pairs_kernel(
        counts: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut cursors: DisjointSlice<u32>,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0_u32;
        let mut expert = 0_usize;
        while expert < SORTED_EXPERTS {
            unsafe {
                *offsets.get_unchecked_mut(expert) = sum;
                *cursors.get_unchecked_mut(expert) = sum;
            }
            sum += counts[expert];
            expert += 1;
        }
        unsafe {
            *offsets.get_unchecked_mut(SORTED_EXPERTS) = sum;
        }
    }

    #[kernel]
    pub fn moe_scatter_sorted_pairs_kernel(
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
    pub fn moe_gate_up_mid_qwarp32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        selected: &[i32],
        route_weights: &[f32],
        iq2_grids: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row_lane = thread::threadIdx_x() >> 3;
        let pair = thread::blockIdx_y();
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let mut rr = 0_u32;
        while rr < 4 {
            let row = thread::blockIdx_x() * 128 + row_lane + rr * 32;
            if row < expert_mid_dim {
                let row_blocks = ((expert as u32 * expert_mid_dim + row) * xq_blocks) as usize;
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
            rr += 1;
        }
    }

    #[kernel]
    pub fn moe_down_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() & 7;
        let row = thread::blockIdx_x() * 32 + (thread::threadIdx_x() >> 3);
        let pair = thread::blockIdx_y();
        if row >= out_dim {
            return;
        }
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let row_blocks = ((expert as u32 * out_dim + row) * midq_blocks) as usize;
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
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
            }
        }
    }

    #[kernel]
    pub fn moe_gate_up_mid_sorted_qwarp32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        sorted_pairs: &[u32],
        selected: &[i32],
        route_weights: &[f32],
        iq2_grids: &[u64],
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
        let row_blocks = ((expert as u32 * expert_mid_dim + row) * xq_blocks) as usize;
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

    #[kernel]
    pub fn moe_down_sorted_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
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
        let row_blocks = ((expert as u32 * out_dim + row) * midq_blocks) as usize;
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
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
            }
        }
    }

    #[kernel]
    pub fn moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        xq_blocks: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        pair_count: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        xq_scales: &[f32],
        xq_values: &[i8],
        sorted_pairs: &[u32],
        selected: &[i32],
        route_weights: &[f32],
        iq2_grids: &[u64],
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
        let row_blocks = ((expert as u32 * expert_mid_dim + row) * xq_blocks) as usize;
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

    #[kernel]
    pub fn moe_down_sorted_p2_qwarp32_kernel(
        midq_blocks: u32,
        out_dim: u32,
        n_expert: u32,
        pair_count: u32,
        down_weights: &[u8],
        midq_scales: &[f32],
        midq_values: &[i8],
        midq_bsums: &[i32],
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
        let row_blocks = ((expert as u32 * out_dim + row) * midq_blocks) as usize;
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
            unsafe {
                *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
            }
        }
    }

    #[kernel]
    pub fn moe_sum_kernel(
        out_dim: u32,
        n_expert: u32,
        n_tokens: u32,
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
        if value > 127 {
            127
        } else if value < -128 {
            -128
        } else {
            value as i8
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_routed_moe_sorted_p2_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let gate_values = packed_iq2_weights(3);
    let up_values = packed_iq2_weights(11);
    let down_values = packed_q2_weights(19);
    let x_values = input_values();
    let selected_values = vec![2, 1, -1, 2, 1, 3];
    let route_values = vec![0.48, 0.33, 0.25, 0.2, 0.15, 0.09];
    let gate_weights = substrate.upload(&gate_values)?;
    let up_weights = substrate.upload(&up_values)?;
    let down_weights = substrate.upload(&down_values)?;
    let x = substrate.upload(&x_values)?;
    let selected = substrate.upload(&selected_values)?;
    let route_weights = substrate.upload(&route_values)?;
    let grids = substrate.upload(&IQ2_GRIDS)?;
    let signs = substrate.upload(&IQ2_SIGNS)?;
    let qwarp_fallback = std::env::var_os("DS4_CUDA_MOE_QWARP_FALLBACK").is_some();
    let expected = expected_sorted_p2_moe(
        &gate_values,
        &up_values,
        &down_values,
        &x_values,
        &selected_values,
        &route_values,
    );

    if qwarp_fallback {
        let generic = run_generic_qwarp_single(
            &substrate,
            &module,
            &gate_weights,
            &up_weights,
            &down_weights,
            &x,
            &selected,
            &route_weights,
            &grids,
            &signs,
        )?;
        let sorted = run_sorted_p2_moe(
            &substrate,
            &module,
            &gate_weights,
            &up_weights,
            &down_weights,
            &x,
            &selected,
            &route_weights,
            &grids,
            &signs,
            true,
        )?;
        substrate.end_commands()?;
        assert_generic_qwarp_output(&substrate, &generic, &expected)?;
        assert_output(&substrate, &sorted, &expected, &selected_values)?;

        let short_selected = substrate.zeroed::<i32>(N_ROUTED as usize - 1)?;
        assert!(matches!(
            run_generic_qwarp_single(
                &substrate,
                &module,
                &gate_weights,
                &up_weights,
                &down_weights,
                &x,
                &short_selected,
                &route_weights,
                &grids,
                &signs,
            ),
            Err(MoeError::InvalidShape)
        ));

        println!(
            "{{\"milestone\":\"M14.5c2f\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"generic_single_gate_up_matches\":true,\"generic_single_down_and_sum_matches\":true,\"sorted_no_p2_gate_up_matches\":true,\"sorted_no_p2_down_and_sum_matches\":true,\"batched_q8_input_quantize_matches\":true,\"sorted_pair_metadata_consumed\":true,\"negative_expert_fallback_matches\":true,\"partial_row_tiles_match\":true,\"invalid_shape_rejected\":true,\"uses_quarter_warp_shuffle_reduction\":true,\"uses_libdevice_link_path\":true,\"uses_q8_k_activation_quantization\":{},\"owns_moe_gate_up_mid_qwarp32_kernel\":{},\"owns_moe_down_qwarp32_kernel\":{},\"owns_moe_gate_up_mid_sorted_qwarp32_kernel\":{},\"owns_moe_down_sorted_qwarp32_kernel\":{},\"owns_no_decode_lut_generic_dispatch\":{},\"owns_no_p2_sorted_batch_dispatch\":{},\"uses_moe_sum_surface\":{},\"owns_hyperconnection_or_runtime_graph\":{},\"changes_default_route\":{}}}",
            substrate.device_name()?,
            M14_5C2F_SCOPE.uses_q8_k_activation_quantization,
            M14_5C2F_SCOPE.owns_moe_gate_up_mid_qwarp32_kernel,
            M14_5C2F_SCOPE.owns_moe_down_qwarp32_kernel,
            M14_5C2F_SCOPE.owns_moe_gate_up_mid_sorted_qwarp32_kernel,
            M14_5C2F_SCOPE.owns_moe_down_sorted_qwarp32_kernel,
            M14_5C2F_SCOPE.owns_no_decode_lut_generic_dispatch,
            M14_5C2F_SCOPE.owns_no_p2_sorted_batch_dispatch,
            M14_5C2F_SCOPE.uses_moe_sum_surface,
            M14_5C2F_SCOPE.owns_hyperconnection_or_runtime_graph,
            M14_5C2F_SCOPE.changes_default_route,
        );
        return Ok(());
    }

    let output = run_sorted_p2_moe(
        &substrate,
        &module,
        &gate_weights,
        &up_weights,
        &down_weights,
        &x,
        &selected,
        &route_weights,
        &grids,
        &signs,
        false,
    )?;
    substrate.end_commands()?;
    assert_output(&substrate, &output, &expected, &selected_values)?;

    let short_selected = substrate.zeroed::<i32>(PAIR_COUNT as usize - 1)?;
    assert!(matches!(
        run_sorted_p2_moe(
            &substrate,
            &module,
            &gate_weights,
            &up_weights,
            &down_weights,
            &x,
            &short_selected,
            &route_weights,
            &grids,
            &signs,
            false,
        ),
        Err(MoeError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.5c2b2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"sorted_pair_metadata_consumed\":true,\"batched_q8_input_quantize_matches\":true,\"sorted_p2_gate_up_matches\":true,\"sorted_p2_down_matches\":true,\"per_token_sum_matches\":true,\"negative_expert_fallback_matches\":true,\"partial_pair_and_row_tiles_match\":true,\"invalid_shape_rejected\":true,\"uses_quarter_warp_shuffle_reduction\":true,\"uses_libdevice_link_path\":true,\"consumes_sorted_pair_metadata_surface\":{},\"uses_q8_k_activation_quantization\":{},\"owns_moe_gate_up_mid_sorted_p2_qwarp32_kernel\":{},\"owns_moe_down_sorted_p2_qwarp32_kernel\":{},\"owns_no_expert_tiles_p2_batch_dispatch\":{},\"uses_moe_sum_surface\":{},\"owns_expert_tile_or_atomic_down\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5C2B2_SCOPE.consumes_sorted_pair_metadata_surface,
        M14_5C2B2_SCOPE.uses_q8_k_activation_quantization,
        M14_5C2B2_SCOPE.owns_moe_gate_up_mid_sorted_p2_qwarp32_kernel,
        M14_5C2B2_SCOPE.owns_moe_down_sorted_p2_qwarp32_kernel,
        M14_5C2B2_SCOPE.owns_no_expert_tiles_p2_batch_dispatch,
        M14_5C2B2_SCOPE.uses_moe_sum_surface,
        M14_5C2B2_SCOPE.owns_expert_tile_or_atomic_down,
        M14_5C2B2_SCOPE.owns_q4_k_or_runtime_graph,
        M14_5C2B2_SCOPE.changes_default_route,
    );
    Ok(())
}

struct QuantizedRows {
    scales: DeviceBuffer<f32>,
    values: DeviceBuffer<i8>,
    bsums: DeviceBuffer<i32>,
}

struct SortedP2Output {
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    cursors: DeviceBuffer<u32>,
    sorted_pairs: DeviceBuffer<u32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}

struct ExpectedQuantizedRows {
    scales: Vec<f32>,
    values: Vec<i8>,
    bsums: Vec<i32>,
}

struct ExpectedSortedP2Output {
    gate: Vec<f32>,
    up: Vec<f32>,
    mid: Vec<f32>,
    down: Vec<f32>,
    out: Vec<f32>,
}

struct GenericQwarpOutput {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_sorted_p2_moe(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    gate_weights: &DeviceBuffer<u8>,
    up_weights: &DeviceBuffer<u8>,
    down_weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    selected: &DeviceBuffer<i32>,
    route_weights: &DeviceBuffer<f32>,
    grids: &DeviceBuffer<u64>,
    signs: &DeviceBuffer<u8>,
    qwarp_fallback: bool,
) -> Result<SortedP2Output, MoeError> {
    if gate_weights.len() < MODEL_EXPERTS * EXPERT_MID_DIM as usize * IQ2_BLOCK_BYTES
        || up_weights.len() < MODEL_EXPERTS * EXPERT_MID_DIM as usize * IQ2_BLOCK_BYTES
        || down_weights.len() < MODEL_EXPERTS * OUT_DIM as usize * Q2_BLOCK_BYTES
        || x.len() < (N_TOKENS * EXPERT_IN_DIM) as usize
        || selected.len() < PAIR_COUNT as usize
        || route_weights.len() < PAIR_COUNT as usize
        || grids.len() < IQ2_GRIDS.len()
        || signs.len() < IQ2_SIGNS.len()
    {
        return Err(MoeError::InvalidShape);
    }
    let input_q = quantize_rows(substrate, module, x, N_TOKENS)?;
    let counts = substrate.zeroed::<u32>(SORTED_EXPERTS)?;
    module.moe_count_sorted_pairs_kernel(
        substrate.stream(),
        one_dimensional_launch(PAIR_COUNT),
        PAIR_COUNT,
        selected,
        &counts,
    )?;
    let mut offsets = substrate.zeroed::<u32>(SORTED_EXPERTS + 1)?;
    let mut cursors = substrate.zeroed::<u32>(SORTED_EXPERTS)?;
    module.moe_prefix_sorted_pairs_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
        &counts,
        &mut offsets,
        &mut cursors,
    )?;
    let mut sorted_pairs = substrate.zeroed::<u32>(PAIR_COUNT as usize)?;
    module.moe_scatter_sorted_pairs_kernel(
        substrate.stream(),
        one_dimensional_launch(PAIR_COUNT),
        PAIR_COUNT,
        selected,
        &cursors,
        &mut sorted_pairs,
    )?;
    let mut gate = substrate.zeroed::<f32>((PAIR_COUNT * EXPERT_MID_DIM) as usize)?;
    let mut up = substrate.zeroed::<f32>((PAIR_COUNT * EXPERT_MID_DIM) as usize)?;
    let mut mid = substrate.zeroed::<f32>((PAIR_COUNT * EXPERT_MID_DIM) as usize)?;
    if qwarp_fallback {
        module.moe_gate_up_mid_sorted_qwarp32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (EXPERT_MID_DIM.div_ceil(32), PAIR_COUNT, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            EXPERT_IN_DIM / QK_K as u32,
            EXPERT_MID_DIM,
            N_ROUTED,
            CLAMP,
            gate_weights,
            up_weights,
            &input_q.scales,
            &input_q.values,
            &sorted_pairs,
            selected,
            route_weights,
            grids,
            signs,
            &mut gate,
            &mut up,
            &mut mid,
        )?;
    } else {
        module.moe_gate_up_mid_sorted_p2_qwarp32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (EXPERT_MID_DIM.div_ceil(16), PAIR_COUNT.div_ceil(2), 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            EXPERT_IN_DIM / QK_K as u32,
            EXPERT_MID_DIM,
            N_ROUTED,
            PAIR_COUNT,
            CLAMP,
            gate_weights,
            up_weights,
            &input_q.scales,
            &input_q.values,
            &sorted_pairs,
            selected,
            route_weights,
            grids,
            signs,
            &mut gate,
            &mut up,
            &mut mid,
        )?;
    }
    let mid_q = quantize_rows(substrate, module, &mid, PAIR_COUNT)?;
    let mut down = substrate.zeroed::<f32>((PAIR_COUNT * OUT_DIM) as usize)?;
    if qwarp_fallback {
        module.moe_down_sorted_qwarp32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (OUT_DIM.div_ceil(32), PAIR_COUNT, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            EXPERT_MID_DIM / QK_K as u32,
            OUT_DIM,
            N_ROUTED,
            down_weights,
            &mid_q.scales,
            &mid_q.values,
            &mid_q.bsums,
            &sorted_pairs,
            selected,
            &mut down,
        )?;
    } else {
        module.moe_down_sorted_p2_qwarp32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (OUT_DIM.div_ceil(16), PAIR_COUNT.div_ceil(2), 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            EXPERT_MID_DIM / QK_K as u32,
            OUT_DIM,
            N_ROUTED,
            PAIR_COUNT,
            down_weights,
            &mid_q.scales,
            &mid_q.values,
            &mid_q.bsums,
            &sorted_pairs,
            selected,
            &mut down,
        )?;
    }
    let mut out = substrate.zeroed::<f32>((N_TOKENS * OUT_DIM) as usize)?;
    module.moe_sum_kernel(
        substrate.stream(),
        one_dimensional_launch(N_TOKENS * OUT_DIM),
        OUT_DIM,
        N_ROUTED,
        N_TOKENS,
        &down,
        &mut out,
    )?;
    Ok(SortedP2Output {
        counts,
        offsets,
        cursors,
        sorted_pairs,
        gate,
        up,
        mid,
        down,
        out,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_generic_qwarp_single(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    gate_weights: &DeviceBuffer<u8>,
    up_weights: &DeviceBuffer<u8>,
    down_weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    selected: &DeviceBuffer<i32>,
    route_weights: &DeviceBuffer<f32>,
    grids: &DeviceBuffer<u64>,
    signs: &DeviceBuffer<u8>,
) -> Result<GenericQwarpOutput, MoeError> {
    if gate_weights.len() < MODEL_EXPERTS * EXPERT_MID_DIM as usize * IQ2_BLOCK_BYTES
        || up_weights.len() < MODEL_EXPERTS * EXPERT_MID_DIM as usize * IQ2_BLOCK_BYTES
        || down_weights.len() < MODEL_EXPERTS * OUT_DIM as usize * Q2_BLOCK_BYTES
        || x.len() < EXPERT_IN_DIM as usize
        || selected.len() < N_ROUTED as usize
        || route_weights.len() < N_ROUTED as usize
        || grids.len() < IQ2_GRIDS.len()
        || signs.len() < IQ2_SIGNS.len()
    {
        return Err(MoeError::InvalidShape);
    }
    let input_q = quantize_rows(substrate, module, x, 1)?;
    let mut gate = substrate.zeroed::<f32>((N_ROUTED * EXPERT_MID_DIM) as usize)?;
    let mut up = substrate.zeroed::<f32>((N_ROUTED * EXPERT_MID_DIM) as usize)?;
    let mut mid = substrate.zeroed::<f32>((N_ROUTED * EXPERT_MID_DIM) as usize)?;
    module.moe_gate_up_mid_qwarp32_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (EXPERT_MID_DIM.div_ceil(128), N_ROUTED, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        EXPERT_IN_DIM / QK_K as u32,
        EXPERT_MID_DIM,
        N_ROUTED,
        CLAMP,
        gate_weights,
        up_weights,
        &input_q.scales,
        &input_q.values,
        selected,
        route_weights,
        grids,
        signs,
        &mut gate,
        &mut up,
        &mut mid,
    )?;
    let mid_q = quantize_rows(substrate, module, &mid, N_ROUTED)?;
    let mut down = substrate.zeroed::<f32>((N_ROUTED * OUT_DIM) as usize)?;
    module.moe_down_qwarp32_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (OUT_DIM.div_ceil(32), N_ROUTED, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        EXPERT_MID_DIM / QK_K as u32,
        OUT_DIM,
        N_ROUTED,
        down_weights,
        &mid_q.scales,
        &mid_q.values,
        &mid_q.bsums,
        selected,
        &mut down,
    )?;
    let mut out = substrate.zeroed::<f32>(OUT_DIM as usize)?;
    module.moe_sum_kernel(
        substrate.stream(),
        one_dimensional_launch(OUT_DIM),
        OUT_DIM,
        N_ROUTED,
        1,
        &down,
        &mut out,
    )?;
    Ok(GenericQwarpOutput {
        gate,
        up,
        mid,
        down,
        out,
    })
}

fn quantize_rows(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    x: &DeviceBuffer<f32>,
    n_rows: u32,
) -> Result<QuantizedRows, MoeError> {
    let blocks = EXPERT_IN_DIM as usize / QK_K;
    let mut scales = substrate.zeroed::<f32>(n_rows as usize * blocks)?;
    let mut values = substrate.zeroed::<i8>(n_rows as usize * blocks * QK_K)?;
    let mut bsums = substrate.zeroed::<i32>(n_rows as usize * blocks * 16)?;
    module.q8_k_quantize_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (blocks as u32, n_rows, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        EXPERT_IN_DIM,
        n_rows,
        x,
        &mut scales,
        &mut values,
        &mut bsums,
    )?;
    Ok(QuantizedRows {
        scales,
        values,
        bsums,
    })
}

fn one_dimensional_launch(count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn expected_sorted_p2_moe(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    x: &[f32],
    selected: &[i32],
    route_weights: &[f32],
) -> ExpectedSortedP2Output {
    let input_q = expected_quantized_rows(x, N_TOKENS);
    let mut gate = vec![0.0_f32; (PAIR_COUNT * EXPERT_MID_DIM) as usize];
    let mut up = vec![0.0_f32; (PAIR_COUNT * EXPERT_MID_DIM) as usize];
    let mut mid = vec![0.0_f32; (PAIR_COUNT * EXPERT_MID_DIM) as usize];
    for pair in 0..PAIR_COUNT as usize {
        let token = pair / N_ROUTED as usize;
        let expert = normalized_expert(selected[pair]);
        for row in 0..EXPERT_MID_DIM as usize {
            let block = expert * EXPERT_MID_DIM as usize + row;
            let mut gate_value = iq2_q8_k_dot(gate_weights, block, &input_q, token);
            let mut up_value = iq2_q8_k_dot(up_weights, block, &input_q, token);
            gate_value = gate_value.min(CLAMP);
            up_value = up_value.clamp(-CLAMP, CLAMP);
            let offset = pair * EXPERT_MID_DIM as usize + row;
            gate[offset] = gate_value;
            up[offset] = up_value;
            mid[offset] =
                (gate_value / (1.0 + (-gate_value).exp())) * up_value * route_weights[pair];
        }
    }
    let mid_q = expected_quantized_rows(&mid, PAIR_COUNT);
    let mut down = vec![0.0_f32; (PAIR_COUNT * OUT_DIM) as usize];
    for pair in 0..PAIR_COUNT as usize {
        let expert = normalized_expert(selected[pair]);
        for row in 0..OUT_DIM as usize {
            down[pair * OUT_DIM as usize + row] =
                q2_q8_k_dot(down_weights, expert * OUT_DIM as usize + row, &mid_q, pair);
        }
    }
    let mut out = vec![0.0_f32; (N_TOKENS * OUT_DIM) as usize];
    for token in 0..N_TOKENS as usize {
        for row in 0..OUT_DIM as usize {
            for slot in 0..N_ROUTED as usize {
                out[token * OUT_DIM as usize + row] +=
                    down[(token * N_ROUTED as usize + slot) * OUT_DIM as usize + row];
            }
        }
    }
    ExpectedSortedP2Output {
        gate,
        up,
        mid,
        down,
        out,
    }
}

fn expected_quantized_rows(x: &[f32], n_rows: u32) -> ExpectedQuantizedRows {
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
        if maximum == 0.0 {
            continue;
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
    ExpectedQuantizedRows {
        scales,
        values,
        bsums,
    }
}

fn iq2_q8_k_dot(packed: &[u8], block: usize, q8: &ExpectedQuantizedRows, q8_block: usize) -> f32 {
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

fn q2_q8_k_dot(packed: &[u8], block: usize, q8: &ExpectedQuantizedRows, q8_block: usize) -> f32 {
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

fn input_values() -> Vec<f32> {
    let mut values = Vec::with_capacity((N_TOKENS * EXPERT_IN_DIM) as usize);
    for token in 0..N_TOKENS as usize {
        values.extend((0..QK_K).map(|index| {
            let magnitude = ((index * 13 + token * 17 + 5) % 29) as f32 * 0.0078125 + 0.015625;
            if (index + token) % 3 == 0 {
                -magnitude
            } else {
                magnitude
            }
        }));
    }
    values[17] = -0.75;
    values[QK_K + 23] = 0.875;
    values
}

fn assert_output(
    substrate: &CudaOxideSubstrate,
    actual: &SortedP2Output,
    expected: &ExpectedSortedP2Output,
    selected: &[i32],
) -> Result<(), DriverError> {
    let counts = substrate.download(&actual.counts)?;
    let offsets = substrate.download(&actual.offsets)?;
    let cursors = substrate.download(&actual.cursors)?;
    let sorted_pairs = substrate.download(&actual.sorted_pairs)?;
    assert_sorted_metadata(selected, &counts, &offsets, &cursors, &sorted_pairs);
    assert_close(&substrate.download(&actual.gate)?, &expected.gate);
    assert_close(&substrate.download(&actual.up)?, &expected.up);
    assert_close(&substrate.download(&actual.mid)?, &expected.mid);
    assert_close(&substrate.download(&actual.down)?, &expected.down);
    assert_close(&substrate.download(&actual.out)?, &expected.out);
    Ok(())
}

fn assert_generic_qwarp_output(
    substrate: &CudaOxideSubstrate,
    actual: &GenericQwarpOutput,
    expected: &ExpectedSortedP2Output,
) -> Result<(), DriverError> {
    let pair_values = (N_ROUTED * EXPERT_MID_DIM) as usize;
    let down_values = (N_ROUTED * OUT_DIM) as usize;
    assert_close(
        &substrate.download(&actual.gate)?,
        &expected.gate[..pair_values],
    );
    assert_close(
        &substrate.download(&actual.up)?,
        &expected.up[..pair_values],
    );
    assert_close(
        &substrate.download(&actual.mid)?,
        &expected.mid[..pair_values],
    );
    assert_close(
        &substrate.download(&actual.down)?,
        &expected.down[..down_values],
    );
    assert_close(
        &substrate.download(&actual.out)?,
        &expected.out[..OUT_DIM as usize],
    );
    Ok(())
}

fn assert_sorted_metadata(
    selected: &[i32],
    counts: &[u32],
    offsets: &[u32],
    cursors: &[u32],
    sorted_pairs: &[u32],
) {
    let mut expected_counts = vec![0_u32; SORTED_EXPERTS];
    for &expert in selected {
        expected_counts[normalized_expert(expert)] += 1;
    }
    assert_eq!(counts, expected_counts);
    let mut expected_offsets = vec![0_u32; SORTED_EXPERTS + 1];
    for expert in 0..SORTED_EXPERTS {
        expected_offsets[expert + 1] = expected_offsets[expert] + expected_counts[expert];
        assert_eq!(
            cursors[expert],
            expected_offsets[expert] + expected_counts[expert]
        );
    }
    assert_eq!(offsets, expected_offsets);
    let mut seen = vec![false; PAIR_COUNT as usize];
    for expert in 0..SORTED_EXPERTS {
        for position in offsets[expert] as usize..offsets[expert + 1] as usize {
            let pair = sorted_pairs[position] as usize;
            assert!(pair < PAIR_COUNT as usize);
            assert_eq!(normalized_expert(selected[pair]), expert);
            assert!(!seen[pair]);
            seen[pair] = true;
        }
    }
    assert!(seen.into_iter().all(|entry| entry));
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
enum MoeError {
    InvalidShape,
    Driver(DriverError),
}

impl From<DriverError> for MoeError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl fmt::Display for MoeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("sorted P2 routed MoE tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MoeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
