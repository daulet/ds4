#![feature(f16)]

use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5C1_SCOPE};

const QK_K: usize = 256;
const IQ2_BLOCK_BYTES: usize = 66;
const Q2_BLOCK_BYTES: usize = 84;
const THREADS: u32 = 256;
const MODEL_EXPERTS: usize = 4;
const ROUTED_EXPERTS: u32 = 3;
const EXPERT_IN_DIM: u32 = QK_K as u32;
const EXPERT_MID_DIM: u32 = QK_K as u32;
const OUT_DIM: u32 = 5;
const N_TOKENS: u32 = 2;
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
    pub fn moe_gate_up_mid_f32_kernel(
        n_tokens: u32,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        n_expert: u32,
        clamp: f32,
        gate_weights: &[u8],
        up_weights: &[u8],
        x: &[f32],
        selected: &[i32],
        weights: &[f32],
        iq2_grids: &[u64],
        iq2_signs: &[u8],
        mut gate_out: DisjointSlice<f32>,
        mut up_out: DisjointSlice<f32>,
        mut mid_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL_GATE: SharedArray<f32, QK_K> = SharedArray::UNINIT;
        static mut PARTIAL_UP: SharedArray<f32, QK_K> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let pair = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        if row >= expert_mid_dim || pair >= n_tokens * n_expert || lane >= THREADS {
            return;
        }
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let blocks = expert_in_dim / QK_K as u32;
        let row_blocks = ((expert as u32 * expert_mid_dim + row) * blocks) as usize;
        let x_base = (token * expert_in_dim) as usize;
        let mut gate = 0.0_f32;
        let mut up = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            gate += dev_iq2_xxs_dot_f32(
                gate_weights,
                row_blocks + block as usize,
                x,
                x_base + block as usize * QK_K,
                iq2_grids,
                iq2_signs,
            );
            up += dev_iq2_xxs_dot_f32(
                up_weights,
                row_blocks + block as usize,
                x,
                x_base + block as usize * QK_K,
                iq2_grids,
                iq2_signs,
            );
            block += THREADS;
        }
        unsafe {
            PARTIAL_GATE[lane as usize] = gate;
            PARTIAL_UP[lane as usize] = up;
        }
        thread::sync_threads();
        let mut stride = THREADS >> 1;
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
            let route_weight = weights[(token * n_expert + slot) as usize];
            unsafe {
                *gate_out.get_unchecked_mut(offset) = gate;
                *up_out.get_unchecked_mut(offset) = up;
                *mid_out.get_unchecked_mut(offset) =
                    (gate / (1.0 + (-gate).exp())) * up * route_weight;
            }
        }
    }

    #[kernel]
    pub fn moe_down_f32_kernel(
        n_tokens: u32,
        expert_mid_dim: u32,
        out_dim: u32,
        n_expert: u32,
        down_weights: &[u8],
        mid: &[f32],
        selected: &[i32],
        mut down_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, QK_K> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let pair = thread::blockIdx_y();
        let lane = thread::threadIdx_x();
        if row >= out_dim || pair >= n_tokens * n_expert || lane >= THREADS {
            return;
        }
        let token = pair / n_expert;
        let slot = pair - token * n_expert;
        let mut expert = selected[(token * n_expert + slot) as usize];
        if expert < 0 {
            expert = 0;
        }
        let blocks = expert_mid_dim / QK_K as u32;
        let row_blocks = ((expert as u32 * out_dim + row) * blocks) as usize;
        let mid_base = (pair * expert_mid_dim) as usize;
        let mut accumulator = 0.0_f32;
        let mut block = lane;
        while block < blocks {
            accumulator += dev_q2_k_dot_f32(
                down_weights,
                row_blocks + block as usize,
                mid,
                mid_base + block as usize * QK_K,
            );
            block += THREADS;
        }
        unsafe {
            PARTIAL[lane as usize] = accumulator;
        }
        thread::sync_threads();
        let mut stride = THREADS >> 1;
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

    #[kernel]
    pub fn moe_sum_kernel(
        n_tokens: u32,
        out_dim: u32,
        n_expert: u32,
        down: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        if index >= (n_tokens * out_dim) as usize {
            return;
        }
        let token = index / out_dim as usize;
        let row = index - token * out_dim as usize;
        let mut accumulator = 0.0_f32;
        let mut expert = 0_u32;
        while expert < n_expert {
            accumulator +=
                down[(token * n_expert as usize + expert as usize) * out_dim as usize + row];
            expert += 1;
        }
        unsafe {
            *out.get_unchecked_mut(index) = accumulator;
        }
    }

    fn dev_iq2_xxs_dot_f32(
        packed: &[u8],
        block: usize,
        x: &[f32],
        x_base: usize,
        iq2_grids: &[u64],
        iq2_signs: &[u8],
    ) -> f32 {
        let base = block * IQ2_BLOCK_BYTES;
        let d = f16::from_bits(load_u16(packed, base)) as f32;
        let mut accumulator = 0.0_f32;
        let mut ib32 = 0_usize;
        while ib32 < QK_K / 32 {
            let q2 = base + 2 + ib32 * 8;
            let aux_g = load_u16(packed, q2) as u32 | ((load_u16(packed, q2 + 2) as u32) << 16);
            let aux_s = load_u16(packed, q2 + 4) as u32 | ((load_u16(packed, q2 + 6) as u32) << 16);
            let dl = d * (0.5 + (aux_s >> 28) as f32) * 0.25;
            let mut half = 0_u32;
            while half < 2 {
                let mut group = 0_u32;
                while group < 2 {
                    let gi = half * 2 + group;
                    let grid_index = ((aux_g >> (8 * gi)) & 0xff) as usize;
                    let sign_index = ((aux_s >> (14 * half + 7 * group)) & 127) as usize;
                    let grid = iq2_grids[grid_index];
                    let signs = iq2_signs[sign_index];
                    let mut lane = 0_u32;
                    while lane < 8 {
                        let mut value = ((grid >> (8 * lane)) & 0xff) as f32;
                        if signs & (1_u8 << lane) != 0 {
                            value = -value;
                        }
                        accumulator += dl
                            * value
                            * x[x_base
                                + ib32 * 32
                                + half as usize * 16
                                + group as usize * 8
                                + lane as usize];
                        lane += 1;
                    }
                    group += 1;
                }
                half += 1;
            }
            ib32 += 1;
        }
        accumulator
    }

    fn dev_q2_k_dot_f32(packed: &[u8], block: usize, x: &[f32], x_base: usize) -> f32 {
        let base = block * Q2_BLOCK_BYTES;
        let d = f16::from_bits(load_u16(packed, base + 80)) as f32;
        let dmin = f16::from_bits(load_u16(packed, base + 82)) as f32;
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
            let mut lane = 0_usize;
            while lane < 16 {
                let value = dl * ((packed[q + lane] >> shift) & 3) as f32 - ml;
                accumulator += value * x[xf + lane];
                lane += 1;
            }
            il += 1;
        }
        accumulator
    }

    fn load_u16(values: &[u8], offset: usize) -> u16 {
        values[offset] as u16 | ((values[offset + 1] as u16) << 8)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module =
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_routed_moe_f32_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let gate_values = packed_iq2_weights(3);
    let up_values = packed_iq2_weights(11);
    let down_values = packed_q2_weights(19);
    let x_values = input_values();
    let selected_values = vec![0, 2, -1, 3, 1, 0];
    let route_values = vec![0.75, 0.5, 0.25, 0.6, 0.35, 0.2];
    let gate_weights = substrate.upload(&gate_values)?;
    let up_weights = substrate.upload(&up_values)?;
    let down_weights = substrate.upload(&down_values)?;
    let x = substrate.upload(&x_values)?;
    let selected = substrate.upload(&selected_values)?;
    let route_weights = substrate.upload(&route_values)?;
    let grids = substrate.upload(&IQ2_GRIDS)?;
    let signs = substrate.upload(&IQ2_SIGNS)?;

    let batch = run_moe(
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
        N_TOKENS,
        CLAMP,
    )?;
    substrate.flush_commands()?;
    let expected_batch = expected_moe(
        &gate_values,
        &up_values,
        &down_values,
        &x_values,
        &selected_values,
        &route_values,
        N_TOKENS,
        CLAMP,
    );
    assert_output(&substrate, &batch, &expected_batch)?;

    let single = run_moe(
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
        1,
        CLAMP,
    )?;
    substrate.flush_commands()?;
    let expected_single = expected_moe(
        &gate_values,
        &up_values,
        &down_values,
        &x_values,
        &selected_values,
        &route_values,
        1,
        CLAMP,
    );
    assert_output(&substrate, &single, &expected_single)?;

    let unclamped = run_moe(
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
        1,
        0.0,
    )?;
    substrate.end_commands()?;
    let expected_unclamped = expected_moe(
        &gate_values,
        &up_values,
        &down_values,
        &x_values,
        &selected_values,
        &route_values,
        1,
        0.0,
    );
    assert_output(&substrate, &unclamped, &expected_unclamped)?;
    assert!(expected_single
        .mid
        .iter()
        .zip(&expected_unclamped.mid)
        .any(|(clamped, raw)| (clamped - raw).abs() > 1.0e-5));

    let short_selected = substrate.zeroed::<i32>((N_TOKENS * ROUTED_EXPERTS - 1) as usize)?;
    assert!(matches!(
        run_moe(
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
            N_TOKENS,
            CLAMP,
        ),
        Err(MoeError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.5c1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"packed_iq2_gate_up_matches\":true,\"packed_q2_down_matches\":true,\"weighted_swiglu_matches\":true,\"expert_sum_matches\":true,\"negative_expert_fallback_matches\":true,\"single_token_surface_matches\":true,\"batch_surface_matches\":true,\"clamp_behavior_matches\":true,\"invalid_shape_rejected\":true,\"uses_shared_reduction\":true,\"uses_libdevice_link_path\":true,\"consumes_router_selection_surface\":{},\"owns_iq2_xxs_f32_gate_up_dot\":{},\"owns_q2_k_f32_down_dot\":{},\"owns_moe_gate_up_mid_f32_kernel\":{},\"owns_moe_down_f32_kernel\":{},\"owns_moe_sum_kernel\":{},\"owns_single_and_batch_f32_activation_moe_surface\":{},\"owns_q8_activation_or_optimized_moe_dispatch\":{},\"owns_hyperconnection_or_runtime_graph\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5C1_SCOPE.consumes_router_selection_surface,
        M14_5C1_SCOPE.owns_iq2_xxs_f32_gate_up_dot,
        M14_5C1_SCOPE.owns_q2_k_f32_down_dot,
        M14_5C1_SCOPE.owns_moe_gate_up_mid_f32_kernel,
        M14_5C1_SCOPE.owns_moe_down_f32_kernel,
        M14_5C1_SCOPE.owns_moe_sum_kernel,
        M14_5C1_SCOPE.owns_single_and_batch_f32_activation_moe_surface,
        M14_5C1_SCOPE.owns_q8_activation_or_optimized_moe_dispatch,
        M14_5C1_SCOPE.owns_hyperconnection_or_runtime_graph,
        M14_5C1_SCOPE.changes_default_route,
    );
    Ok(())
}

struct MoeOutput {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}

struct ExpectedMoeOutput {
    gate: Vec<f32>,
    up: Vec<f32>,
    mid: Vec<f32>,
    down: Vec<f32>,
    out: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_moe(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    gate_weights: &DeviceBuffer<u8>,
    up_weights: &DeviceBuffer<u8>,
    down_weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    selected: &DeviceBuffer<i32>,
    weights: &DeviceBuffer<f32>,
    grids: &DeviceBuffer<u64>,
    signs: &DeviceBuffer<u8>,
    n_tokens: u32,
    clamp: f32,
) -> Result<MoeOutput, MoeError> {
    let pairs = n_tokens as usize * ROUTED_EXPERTS as usize;
    let iq2_bytes = MODEL_EXPERTS * EXPERT_MID_DIM as usize * IQ2_BLOCK_BYTES;
    let q2_bytes = MODEL_EXPERTS * OUT_DIM as usize * Q2_BLOCK_BYTES;
    if n_tokens == 0
        || gate_weights.len() < iq2_bytes
        || up_weights.len() < iq2_bytes
        || down_weights.len() < q2_bytes
        || x.len() < n_tokens as usize * EXPERT_IN_DIM as usize
        || selected.len() < pairs
        || weights.len() < pairs
        || grids.len() < IQ2_GRIDS.len()
        || signs.len() < IQ2_SIGNS.len()
    {
        return Err(MoeError::InvalidShape);
    }
    let mut gate = substrate.zeroed::<f32>(pairs * EXPERT_MID_DIM as usize)?;
    let mut up = substrate.zeroed::<f32>(pairs * EXPERT_MID_DIM as usize)?;
    let mut mid = substrate.zeroed::<f32>(pairs * EXPERT_MID_DIM as usize)?;
    let mut down = substrate.zeroed::<f32>(pairs * OUT_DIM as usize)?;
    let mut out = substrate.zeroed::<f32>(n_tokens as usize * OUT_DIM as usize)?;
    module.moe_gate_up_mid_f32_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (EXPERT_MID_DIM, n_tokens * ROUTED_EXPERTS, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        n_tokens,
        EXPERT_IN_DIM,
        EXPERT_MID_DIM,
        ROUTED_EXPERTS,
        clamp,
        gate_weights,
        up_weights,
        x,
        selected,
        weights,
        grids,
        signs,
        &mut gate,
        &mut up,
        &mut mid,
    )?;
    module.moe_down_f32_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (OUT_DIM, n_tokens * ROUTED_EXPERTS, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        n_tokens,
        EXPERT_MID_DIM,
        OUT_DIM,
        ROUTED_EXPERTS,
        down_weights,
        &mid,
        selected,
        &mut down,
    )?;
    module.moe_sum_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: ((n_tokens * OUT_DIM).div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        n_tokens,
        OUT_DIM,
        ROUTED_EXPERTS,
        &down,
        &mut out,
    )?;
    Ok(MoeOutput {
        gate,
        up,
        mid,
        down,
        out,
    })
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
                let grid1 = (grid0 + 1) % IQ2_GRIDS.len();
                let grid2 = (grid0 + 2) % IQ2_GRIDS.len();
                let grid3 = (grid0 + 3) % IQ2_GRIDS.len();
                let aux_g = grid0 as u32
                    | (grid1 as u32) << 8
                    | (grid2 as u32) << 16
                    | (grid3 as u32) << 24;
                let sign0 = (row + ib32 + seed) % IQ2_SIGNS.len();
                let sign1 = (sign0 + 1) % IQ2_SIGNS.len();
                let sign2 = (sign0 + 2) % IQ2_SIGNS.len();
                let sign3 = (sign0 + 3) % IQ2_SIGNS.len();
                let aux_s = sign0 as u32
                    | (sign1 as u32) << 7
                    | (sign2 as u32) << 14
                    | (sign3 as u32) << 21
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
                let low = 1 + ((expert + row + scale + seed) % 5) as u8;
                let high = ((expert * 3 + row + scale + seed) % 4) as u8;
                packed.push(low | (high << 4));
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
    (0..N_TOKENS as usize * EXPERT_IN_DIM as usize)
        .map(|index| {
            let magnitude = ((index * 13 + 5) % 29) as f32 * 0.0078125 + 0.015625;
            if index % 3 == 0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn expected_moe(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    x: &[f32],
    selected: &[i32],
    route_weights: &[f32],
    n_tokens: u32,
    clamp: f32,
) -> ExpectedMoeOutput {
    let pairs = n_tokens as usize * ROUTED_EXPERTS as usize;
    let mut gate = vec![0.0_f32; pairs * EXPERT_MID_DIM as usize];
    let mut up = vec![0.0_f32; pairs * EXPERT_MID_DIM as usize];
    let mut mid = vec![0.0_f32; pairs * EXPERT_MID_DIM as usize];
    let mut down = vec![0.0_f32; pairs * OUT_DIM as usize];
    let mut out = vec![0.0_f32; n_tokens as usize * OUT_DIM as usize];
    for pair in 0..pairs {
        let token = pair / ROUTED_EXPERTS as usize;
        let expert = if selected[pair] < 0 {
            0
        } else {
            selected[pair] as usize
        };
        for row in 0..EXPERT_MID_DIM as usize {
            let block = expert * EXPERT_MID_DIM as usize + row;
            let mut gate_value = iq2_xxs_dot(gate_weights, block, x, token * QK_K);
            let mut up_value = iq2_xxs_dot(up_weights, block, x, token * QK_K);
            if clamp > 1.0e-6 {
                gate_value = gate_value.min(clamp);
                up_value = up_value.clamp(-clamp, clamp);
            }
            let offset = pair * EXPERT_MID_DIM as usize + row;
            gate[offset] = gate_value;
            up[offset] = up_value;
            mid[offset] =
                (gate_value / (1.0 + (-gate_value).exp())) * up_value * route_weights[pair];
        }
        for row in 0..OUT_DIM as usize {
            let block = expert * OUT_DIM as usize + row;
            down[pair * OUT_DIM as usize + row] = q2_k_dot(down_weights, block, &mid, pair * QK_K);
        }
    }
    for token in 0..n_tokens as usize {
        for row in 0..OUT_DIM as usize {
            for slot in 0..ROUTED_EXPERTS as usize {
                out[token * OUT_DIM as usize + row] +=
                    down[(token * ROUTED_EXPERTS as usize + slot) * OUT_DIM as usize + row];
            }
        }
    }
    ExpectedMoeOutput {
        gate,
        up,
        mid,
        down,
        out,
    }
}

fn iq2_xxs_dot(packed: &[u8], block: usize, x: &[f32], x_base: usize) -> f32 {
    let base = block * IQ2_BLOCK_BYTES;
    let d = f16::from_bits(u16::from_le_bytes([packed[base], packed[base + 1]])) as f32;
    let mut accumulator = 0.0_f32;
    for ib32 in 0..QK_K / 32 {
        let q2 = base + 2 + ib32 * 8;
        let aux_g = u16::from_le_bytes([packed[q2], packed[q2 + 1]]) as u32
            | (u16::from_le_bytes([packed[q2 + 2], packed[q2 + 3]]) as u32) << 16;
        let aux_s = u16::from_le_bytes([packed[q2 + 4], packed[q2 + 5]]) as u32
            | (u16::from_le_bytes([packed[q2 + 6], packed[q2 + 7]]) as u32) << 16;
        let dl = d * (0.5 + (aux_s >> 28) as f32) * 0.25;
        for half in 0..2_u32 {
            for group in 0..2_u32 {
                let gi = half * 2 + group;
                let grid = IQ2_GRIDS[((aux_g >> (8 * gi)) & 0xff) as usize];
                let signs = IQ2_SIGNS[((aux_s >> (14 * half + 7 * group)) & 127) as usize];
                for lane in 0..8_u32 {
                    let mut value = ((grid >> (8 * lane)) & 0xff) as f32;
                    if signs & (1 << lane) != 0 {
                        value = -value;
                    }
                    accumulator += dl
                        * value
                        * x[x_base
                            + ib32 * 32
                            + half as usize * 16
                            + group as usize * 8
                            + lane as usize];
                }
            }
        }
    }
    accumulator
}

fn q2_k_dot(packed: &[u8], block: usize, x: &[f32], x_base: usize) -> f32 {
    let base = block * Q2_BLOCK_BYTES;
    let d = f16::from_bits(u16::from_le_bytes([packed[base + 80], packed[base + 81]])) as f32;
    let dmin = f16::from_bits(u16::from_le_bytes([packed[base + 82], packed[base + 83]])) as f32;
    let mut accumulator = 0.0_f32;
    for il in 0..16 {
        let chunk = il / 8;
        let pair = il & 1;
        let shift = ((il / 2) & 3) * 2;
        let scale = packed[base + il];
        let dl = d * (scale & 0x0f) as f32;
        let ml = dmin * (scale >> 4) as f32;
        let q = base + 16 + 32 * chunk + 16 * pair;
        let xf = x_base + chunk * 128 + ((il % 8) / 2) * 32 + pair * 16;
        for lane in 0..16 {
            accumulator += (dl * ((packed[q + lane] >> shift) & 3) as f32 - ml) * x[xf + lane];
        }
    }
    accumulator
}

fn assert_output(
    substrate: &CudaOxideSubstrate,
    actual: &MoeOutput,
    expected: &ExpectedMoeOutput,
) -> Result<(), DriverError> {
    assert_close(&substrate.download(&actual.gate)?, &expected.gate);
    assert_close(&substrate.download(&actual.up)?, &expected.up);
    assert_close(&substrate.download(&actual.mid)?, &expected.mid);
    assert_close(&substrate.download(&actual.down)?, &expected.down);
    assert_close(&substrate.download(&actual.out)?, &expected.out);
    Ok(())
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
            Self::InvalidShape => formatter.write_str("routed MoE tensor shape is invalid"),
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
