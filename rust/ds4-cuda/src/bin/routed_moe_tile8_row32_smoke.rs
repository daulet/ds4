#![feature(f16)]

use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5C2C2_SCOPE, M14_5C2C3_SCOPE};

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
    pub fn moe_down_expert_tile4_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
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
                    unsafe {
                        *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
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
    pub fn moe_down_expert_tile8_row32_kernel(
        midq_blocks: u32,
        out_dim: u32,
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
                    unsafe {
                        *down_out.get_unchecked_mut((pair * out_dim + row) as usize) = accumulator;
                    }
                }
            }
            entry += 1;
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tile4 = std::env::var_os("DS4_CUDA_MOE_TILE4").is_some();
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
    if tile4 {
        assert!(metadata
            .tile_experts
            .iter()
            .zip(&metadata.tile_starts)
            .any(|(&expert, &start)| expert == 1 && start == 8));
    }
    let xq = expected_quantized_rows(&input_values(), N_TOKENS);
    let expected_gate = expected_gate_up_mid(
        &gate_values,
        &up_values,
        &xq,
        &selected_values,
        &route_values,
    );

    let actual_gate = run_gate_up_mid(
        &substrate,
        &module,
        &gate_values,
        &up_values,
        &xq,
        &metadata,
        &route_values,
        tile4,
    )?;
    substrate.flush_commands()?;
    assert_close(&substrate.download(&actual_gate.gate)?, &expected_gate.gate);
    assert_close(&substrate.download(&actual_gate.up)?, &expected_gate.up);
    assert_close(&substrate.download(&actual_gate.mid)?, &expected_gate.mid);

    let midq = expected_quantized_rows(&expected_gate.mid, PAIR_COUNT);
    let expected_down = expected_down(&down_values, &midq, &selected_values);
    let actual_down = run_down(&substrate, &module, &down_values, &midq, &metadata, tile4)?;
    substrate.end_commands()?;
    assert_close(&substrate.download(&actual_down)?, &expected_down);

    let short_tiles = ExpertTileMetadata {
        tile_experts: vec![],
        ..metadata
    };
    assert!(matches!(
        run_down(
            &substrate,
            &module,
            &down_values,
            &midq,
            &short_tiles,
            tile4
        ),
        Err(TileProjectionError::InvalidShape)
    ));

    if tile4 {
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
    let launch = LaunchConfig {
        grid_dim: (EXPERT_MID_DIM.div_ceil(32), metadata.tile_total[0], 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    if tile4 {
        module.moe_gate_up_mid_expert_tile4_row32_kernel(
            substrate.stream(),
            launch,
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
            launch,
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
    let launch = LaunchConfig {
        grid_dim: (OUT_DIM.div_ceil(32), metadata.tile_total[0], 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    if tile4 {
        module.moe_down_expert_tile4_row32_kernel(
            substrate.stream(),
            launch,
            1,
            OUT_DIM,
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
            &mut down,
        )?;
    } else {
        module.moe_down_expert_tile8_row32_kernel(
            substrate.stream(),
            launch,
            1,
            OUT_DIM,
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
            &mut down,
        )?;
    }
    Ok(down)
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
