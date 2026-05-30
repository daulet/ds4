#![feature(f16)]

use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4D8A_SCOPE};

const Q8_BLOCK_SIZE: u64 = 32;
const Q8_BLOCK_BYTES: u64 = 34;
const THREADS: u32 = 256;
const GROUP_DIM: u64 = 35;
const RANK: u64 = 3;
const N_GROUPS: u32 = 2;
const LOW_DIM: u64 = RANK * N_GROUPS as u64;
const OUT_DIM: u64 = 5;
const N_TOKENS: u32 = 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn quantize_q8_0_f32_kernel(
        in_dim: u64,
        blocks: u64,
        n_rows: u32,
        x: &[f32],
        mut xq: DisjointSlice<i8>,
        mut xscale: DisjointSlice<f32>,
    ) {
        let block = thread::blockIdx_x() as u64;
        let row = thread::blockIdx_y() as u64;
        if block >= blocks || row >= n_rows as u64 || thread::threadIdx_x() != 0 {
            return;
        }
        let start = block * Q8_BLOCK_SIZE;
        let count = minimum(Q8_BLOCK_SIZE, in_dim - start);
        let input_base = (row * in_dim + start) as usize;
        let mut maximum = 0.0_f32;
        let mut lane = 0_u64;
        while lane < count {
            let value = x[input_base + lane as usize];
            let magnitude = if value < 0.0 { -value } else { value };
            if magnitude > maximum {
                maximum = magnitude;
            }
            lane += 1;
        }
        let scale = maximum / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        unsafe {
            *xscale.get_unchecked_mut((row * blocks + block) as usize) = scale;
        }
        let output_base = ((row * blocks + block) * Q8_BLOCK_SIZE) as usize;
        lane = 0;
        while lane < Q8_BLOCK_SIZE {
            let quantized = if lane < count {
                clamp_i8(round_ties_even(x[input_base + lane as usize] * inverse))
            } else {
                0
            };
            unsafe {
                *xq.get_unchecked_mut(output_base + lane as usize) = quantized;
            }
            lane += 1;
        }
    }

    #[kernel]
    pub fn grouped_q8_0_a_preq_warp8_kernel(
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        n_tokens: u32,
        blocks: u64,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut low: DisjointSlice<f32>,
    ) {
        let output_row = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        let low_dim = n_groups as u64 * rank;
        if output_row >= low_dim || token >= n_tokens as u64 || thread::threadIdx_x() != 0 {
            return;
        }
        let group = output_row / rank;
        let row_in_group = output_row - group * rank;
        let xrow = token * n_groups as u64 + group;
        let mut accumulator = 0.0_f32;
        let mut block = 0_u64;
        while block < blocks {
            let count = minimum(Q8_BLOCK_SIZE, group_dim - block * Q8_BLOCK_SIZE);
            let weight_base =
                (((group * rank + row_in_group) * blocks + block) * Q8_BLOCK_BYTES) as usize;
            let weight_scale = f16::from_bits(
                weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8),
            ) as f32;
            let xq_base = ((xrow * blocks + block) * Q8_BLOCK_SIZE) as usize;
            let mut dot = 0_i32;
            let mut lane = 0_u64;
            while lane < count {
                dot += (weights[weight_base + 2 + lane as usize] as i8 as i32)
                    * xq[xq_base + lane as usize] as i32;
                lane += 1;
            }
            accumulator += weight_scale * xscale[(xrow * blocks + block) as usize] * dot as f32;
            block += 1;
        }
        unsafe {
            *low.get_unchecked_mut((token * low_dim + output_row) as usize) = accumulator;
        }
    }

    #[kernel]
    pub fn matmul_q8_0_preq_batch_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tokens: u32,
        blocks: u64,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let output_row = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        if output_row >= out_dim || token >= n_tokens as u64 || thread::threadIdx_x() != 0 {
            return;
        }
        let mut accumulator = 0.0_f32;
        let mut block = 0_u64;
        while block < blocks {
            let count = minimum(Q8_BLOCK_SIZE, in_dim - block * Q8_BLOCK_SIZE);
            let weight_base = ((output_row * blocks + block) * Q8_BLOCK_BYTES) as usize;
            let weight_scale = f16::from_bits(
                weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8),
            ) as f32;
            let xq_base = ((token * blocks + block) * Q8_BLOCK_SIZE) as usize;
            let mut dot = 0_i32;
            let mut lane = 0_u64;
            while lane < count {
                dot += (weights[weight_base + 2 + lane as usize] as i8 as i32)
                    * xq[xq_base + lane as usize] as i32;
                lane += 1;
            }
            accumulator += weight_scale * xscale[(token * blocks + block) as usize] * dot as f32;
            block += 1;
        }
        unsafe {
            *out.get_unchecked_mut((token * out_dim + output_row) as usize) = accumulator;
        }
    }

    fn minimum(left: u64, right: u64) -> u64 {
        if left < right {
            left
        } else {
            right
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
        "../../ds4_cuda_attention_output_q8_native_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let heads_values = values(
        (N_TOKENS as u64 * N_GROUPS as u64 * GROUP_DIM) as usize,
        17,
        -1.25,
    );
    let out_a_values = packed_weights(LOW_DIM, GROUP_DIM, 7);
    let out_b_values = packed_weights(OUT_DIM, LOW_DIM, 13);
    let expected_low = expected_low_output(&heads_values, &out_a_values, N_TOKENS);
    let expected_out = expected_output(&expected_low, &out_b_values, N_TOKENS);
    let heads = substrate.upload(&heads_values)?;
    let out_a = substrate.upload(&out_a_values)?;
    let out_b = substrate.upload(&out_b_values)?;

    let low_single = attention_output_low_q8_tensor(&substrate, &module, &heads, &out_a)?;
    substrate.flush_commands()?;
    assert_close(
        &substrate.download(&low_single)?,
        &expected_low[..LOW_DIM as usize],
    );

    let (low_batch, out_batch) =
        attention_output_q8_batch_tensor(&substrate, &module, &heads, &out_a, &out_b)?;
    substrate.end_commands()?;
    assert_close(&substrate.download(&low_batch)?, &expected_low);
    assert_close(&substrate.download(&out_batch)?, &expected_out);

    let short_heads = substrate.zeroed::<f32>((N_GROUPS as u64 * GROUP_DIM - 1) as usize)?;
    assert!(matches!(
        attention_output_low_q8_tensor(&substrate, &module, &short_heads, &out_a),
        Err(AttentionOutputQ8Error::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4d8a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"low_q8_output_matches\":true,\"batch_low_q8_output_matches\":true,\"batch_output_q8_output_matches\":true,\"grouped_projection_matches\":true,\"two_stage_projection_matches\":true,\"partial_block_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"consumes_q8_conversion_and_matmul_kernels\":{},\"owns_attention_output_low_q8_surface\":{},\"owns_attention_output_q8_batch_native_surface\":{},\"owns_attention_output_a_cublas_dispatch\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D8A_SCOPE.consumes_q8_conversion_and_matmul_kernels,
        M14_4D8A_SCOPE.owns_attention_output_low_q8_surface,
        M14_4D8A_SCOPE.owns_attention_output_q8_batch_native_surface,
        M14_4D8A_SCOPE.owns_attention_output_a_cublas_dispatch,
        M14_4D8A_SCOPE.owns_runtime_graph_integration,
        M14_4D8A_SCOPE.changes_default_route,
    );
    Ok(())
}

fn attention_output_low_q8_tensor(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    heads: &DeviceBuffer<f32>,
    out_a: &DeviceBuffer<u8>,
) -> Result<DeviceBuffer<f32>, AttentionOutputQ8Error> {
    let blocks = GROUP_DIM.div_ceil(Q8_BLOCK_SIZE);
    if heads.len() < (N_GROUPS as u64 * GROUP_DIM) as usize
        || out_a.len() < (LOW_DIM * blocks * Q8_BLOCK_BYTES) as usize
    {
        return Err(AttentionOutputQ8Error::InvalidShape);
    }
    let mut xq = substrate
        .zeroed::<i8>((N_GROUPS as u64 * blocks * Q8_BLOCK_SIZE) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    let mut xscale = substrate
        .zeroed::<f32>((N_GROUPS as u64 * blocks) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    let mut low = substrate
        .zeroed::<f32>(LOW_DIM as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    quantize_rows(
        module,
        substrate,
        heads,
        GROUP_DIM,
        N_GROUPS,
        &mut xq,
        &mut xscale,
    )?;
    grouped_projection(module, substrate, out_a, &xq, &xscale, 1, &mut low)?;
    Ok(low)
}

fn attention_output_q8_batch_tensor(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    heads: &DeviceBuffer<f32>,
    out_a: &DeviceBuffer<u8>,
    out_b: &DeviceBuffer<u8>,
) -> Result<(DeviceBuffer<f32>, DeviceBuffer<f32>), AttentionOutputQ8Error> {
    let a_blocks = GROUP_DIM.div_ceil(Q8_BLOCK_SIZE);
    let b_blocks = LOW_DIM.div_ceil(Q8_BLOCK_SIZE);
    if heads.len() < (N_TOKENS as u64 * N_GROUPS as u64 * GROUP_DIM) as usize
        || out_a.len() < (LOW_DIM * a_blocks * Q8_BLOCK_BYTES) as usize
        || out_b.len() < (OUT_DIM * b_blocks * Q8_BLOCK_BYTES) as usize
    {
        return Err(AttentionOutputQ8Error::InvalidShape);
    }
    let n_head_rows = N_TOKENS as u64 * N_GROUPS as u64;
    let mut head_q = substrate
        .zeroed::<i8>((n_head_rows * a_blocks * Q8_BLOCK_SIZE) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    let mut head_scale = substrate
        .zeroed::<f32>((n_head_rows * a_blocks) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    let mut low = substrate
        .zeroed::<f32>((N_TOKENS as u64 * LOW_DIM) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    quantize_rows(
        module,
        substrate,
        heads,
        GROUP_DIM,
        n_head_rows as u32,
        &mut head_q,
        &mut head_scale,
    )?;
    grouped_projection(
        module,
        substrate,
        out_a,
        &head_q,
        &head_scale,
        N_TOKENS,
        &mut low,
    )?;
    let mut low_q = substrate
        .zeroed::<i8>((N_TOKENS as u64 * b_blocks * Q8_BLOCK_SIZE) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    let mut low_scale = substrate
        .zeroed::<f32>((N_TOKENS as u64 * b_blocks) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    let mut out = substrate
        .zeroed::<f32>((N_TOKENS as u64 * OUT_DIM) as usize)
        .map_err(AttentionOutputQ8Error::Driver)?;
    quantize_rows(
        module,
        substrate,
        &low,
        LOW_DIM,
        N_TOKENS,
        &mut low_q,
        &mut low_scale,
    )?;
    module
        .matmul_q8_0_preq_batch_warp8_kernel(
            substrate.stream(),
            output_launch(OUT_DIM, N_TOKENS),
            LOW_DIM,
            OUT_DIM,
            N_TOKENS,
            b_blocks,
            out_b,
            &low_q,
            &low_scale,
            &mut out,
        )
        .map_err(AttentionOutputQ8Error::Driver)?;
    Ok((low, out))
}

fn quantize_rows(
    module: &kernels::LoadedModule,
    substrate: &CudaOxideSubstrate,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    n_rows: u32,
    xq: &mut DeviceBuffer<i8>,
    xscale: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOutputQ8Error> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    module
        .quantize_q8_0_f32_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (blocks as u32, n_rows, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            in_dim,
            blocks,
            n_rows,
            x,
            xq,
            xscale,
        )
        .map_err(AttentionOutputQ8Error::Driver)
}

fn grouped_projection(
    module: &kernels::LoadedModule,
    substrate: &CudaOxideSubstrate,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    n_tokens: u32,
    low: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOutputQ8Error> {
    module
        .grouped_q8_0_a_preq_warp8_kernel(
            substrate.stream(),
            output_launch(LOW_DIM, n_tokens),
            GROUP_DIM,
            RANK,
            N_GROUPS,
            n_tokens,
            GROUP_DIM.div_ceil(Q8_BLOCK_SIZE),
            weights,
            xq,
            xscale,
            low,
        )
        .map_err(AttentionOutputQ8Error::Driver)
}

fn output_launch(output_dim: u64, n_tokens: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (output_dim as u32, n_tokens, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
        .collect()
}

fn packed_weights(rows: u64, in_dim: u64, seed: u64) -> Vec<u8> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let mut packed = Vec::with_capacity((rows * blocks * Q8_BLOCK_BYTES) as usize);
    for row in 0..rows {
        for block in 0..blocks {
            let scale = (0.125_f32 + ((row + block + seed) % 7) as f32 * 0.0625) as f16;
            packed.extend_from_slice(&scale.to_bits().to_le_bytes());
            for lane in 0..Q8_BLOCK_SIZE {
                packed.push((((row * 7 + block * 11 + lane * 3 + seed) % 21) as i8 - 10) as u8);
            }
        }
    }
    packed
}

fn expected_low_output(heads: &[f32], out_a: &[u8], n_tokens: u32) -> Vec<f32> {
    let (head_q, head_scale) = quantized_rows(heads, GROUP_DIM, n_tokens as u64 * N_GROUPS as u64);
    let blocks = GROUP_DIM.div_ceil(Q8_BLOCK_SIZE);
    let mut low = vec![0.0_f32; (n_tokens as u64 * LOW_DIM) as usize];
    for token in 0..n_tokens as u64 {
        for row in 0..LOW_DIM {
            let group = row / RANK;
            let xrow = token * N_GROUPS as u64 + group;
            low[(token * LOW_DIM + row) as usize] =
                q8_dot(out_a, row, GROUP_DIM, blocks, &head_q, &head_scale, xrow);
        }
    }
    low
}

fn expected_output(low: &[f32], out_b: &[u8], n_tokens: u32) -> Vec<f32> {
    let (low_q, low_scale) = quantized_rows(low, LOW_DIM, n_tokens as u64);
    let blocks = LOW_DIM.div_ceil(Q8_BLOCK_SIZE);
    let mut out = vec![0.0_f32; (n_tokens as u64 * OUT_DIM) as usize];
    for token in 0..n_tokens as u64 {
        for row in 0..OUT_DIM {
            out[(token * OUT_DIM + row) as usize] =
                q8_dot(out_b, row, LOW_DIM, blocks, &low_q, &low_scale, token);
        }
    }
    out
}

fn quantized_rows(values: &[f32], in_dim: u64, n_rows: u64) -> (Vec<i8>, Vec<f32>) {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let mut xq = vec![0_i8; (n_rows * blocks * Q8_BLOCK_SIZE) as usize];
    let mut xscale = vec![0.0_f32; (n_rows * blocks) as usize];
    for row in 0..n_rows {
        for block in 0..blocks {
            let start = block * Q8_BLOCK_SIZE;
            let count = Q8_BLOCK_SIZE.min(in_dim - start);
            let base = (row * in_dim + start) as usize;
            let scale = values[base..base + count as usize]
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0_f32, f32::max)
                / 127.0;
            xscale[(row * blocks + block) as usize] = scale;
            let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            for lane in 0..count {
                xq[((row * blocks + block) * Q8_BLOCK_SIZE + lane) as usize] =
                    clamp_i8(round_ties_even(values[base + lane as usize] * inverse));
            }
        }
    }
    (xq, xscale)
}

fn q8_dot(
    weights: &[u8],
    row: u64,
    in_dim: u64,
    blocks: u64,
    xq: &[i8],
    xscale: &[f32],
    xrow: u64,
) -> f32 {
    let mut accumulator = 0.0_f32;
    for block in 0..blocks {
        let count = Q8_BLOCK_SIZE.min(in_dim - block * Q8_BLOCK_SIZE);
        let weight_base = ((row * blocks + block) * Q8_BLOCK_BYTES) as usize;
        let weight_scale = f16::from_bits(u16::from_le_bytes([
            weights[weight_base],
            weights[weight_base + 1],
        ])) as f32;
        let xq_base = ((xrow * blocks + block) * Q8_BLOCK_SIZE) as usize;
        let dot = (0..count as usize)
            .map(|lane| (weights[weight_base + 2 + lane] as i8 as i32) * xq[xq_base + lane] as i32)
            .sum::<i32>();
        accumulator += weight_scale * xscale[(xrow * blocks + block) as usize] * dot as f32;
    }
    accumulator
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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum AttentionOutputQ8Error {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for AttentionOutputQ8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("attention output Q8 shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionOutputQ8Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
