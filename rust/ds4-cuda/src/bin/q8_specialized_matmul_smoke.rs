#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3D3_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn matmul_q8_0_pair_preq_warp8_kernel(
        in_dim: u64,
        out0_dim: u64,
        out1_dim: u64,
        blocks: u64,
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
            let start = block * 32;
            let remaining = in_dim - start;
            let count = if remaining < 32 { remaining } else { 32 };
            let xq_base = (block * 32) as usize;
            if row < out0_dim {
                let weight_base = ((row * blocks + block) * 34) as usize;
                let scale_bits =
                    weights0[weight_base] as u16 | ((weights0[weight_base + 1] as u16) << 8);
                let weight_scale = f16::from_bits(scale_bits) as f32;
                let mut dot = 0_i32;
                let mut element = 0_u64;
                while element < count {
                    dot += (weights0[weight_base + 2 + element as usize] as i8 as i32)
                        * xq[xq_base + element as usize] as i32;
                    element += 1;
                }
                acc0 += weight_scale * xscale[block as usize] * dot as f32;
            }
            if row < out1_dim {
                let weight_base = ((row * blocks + block) * 34) as usize;
                let scale_bits =
                    weights1[weight_base] as u16 | ((weights1[weight_base + 1] as u16) << 8);
                let weight_scale = f16::from_bits(scale_bits) as f32;
                let mut dot = 0_i32;
                let mut element = 0_u64;
                while element < count {
                    dot += (weights1[weight_base + 2 + element as usize] as i8 as i32)
                        * xq[xq_base + element as usize] as i32;
                    element += 1;
                }
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
    pub fn matmul_q8_0_hc_expand_preq_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        n_embd: u32,
        n_hc: u32,
        blocks: u64,
        has_add: u32,
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
            let start = block * 32;
            let remaining = in_dim - start;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = (block * 32) as usize;
            let mut dot = 0_i32;
            let mut element = 0_u64;
            while element < count {
                dot += (weights[weight_base + 2 + element as usize] as i8 as i32)
                    * xq[xq_base + element as usize] as i32;
                element += 1;
            }
            acc += weight_scale * xscale[block as usize] * dot as f32;
            block += 32;
        }
        let mut offset = 16_u32;
        while offset > 0 {
            acc += warp::shuffle_down_f32(acc, offset);
            offset >>= 1;
        }
        if lane == 0 {
            let row_index = row as usize;
            unsafe {
                *block_out.get_unchecked_mut(row_index) = acc;
            }
            let block_value = if has_add != 0 {
                acc + block_add[row_index]
            } else {
                acc
            };
            let post_base = n_hc as usize;
            let combination_base = (2 * n_hc) as usize;
            let mut dst_hc = 0_u32;
            while dst_hc < n_hc {
                let mut hc_acc = block_value * split[post_base + dst_hc as usize];
                let mut src_hc = 0_u32;
                while src_hc < n_hc {
                    let combination_index =
                        combination_base + dst_hc as usize + src_hc as usize * n_hc as usize;
                    let residual_index = src_hc as usize * n_embd as usize + row_index;
                    hc_acc += split[combination_index] * residual_hc[residual_index];
                    src_hc += 1;
                }
                unsafe {
                    *out_hc.get_unchecked_mut(dst_hc as usize * n_embd as usize + row_index) =
                        hc_acc;
                }
                dst_hc += 1;
            }
        }
    }
}

const Q8_BLOCK_SIZE: u64 = 32;
const Q8_BLOCK_BYTES: u64 = 34;
const WARP8_THREADS: u32 = 256;
const IN_DIM: u64 = 35;
const OUT0_DIM: u64 = 10;
const OUT1_DIM: u64 = 7;
const N_HC: u32 = 4;
const N_EMBD: u32 = OUT0_DIM as u32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE);
    let xq_values = quantized_values();
    let xscale_values = vec![0.5_f32, 1.0];
    let weights0_values = packed_weights(OUT0_DIM, 0);
    let weights1_values = packed_weights(OUT1_DIM, 5);
    let expected0 = expected_matmul(&weights0_values, OUT0_DIM, &xq_values, &xscale_values);
    let expected1 = expected_matmul(&weights1_values, OUT1_DIM, &xq_values, &xscale_values);
    let weights0 = substrate.upload(&weights0_values)?;
    let weights1 = substrate.upload(&weights1_values)?;
    let xq = substrate.upload(&xq_values)?;
    let xscale = substrate.upload(&xscale_values)?;

    let mut out0 = substrate.zeroed::<f32>(OUT0_DIM as usize)?;
    let mut out1 = substrate.zeroed::<f32>(OUT1_DIM as usize)?;
    matmul_q8_pair(
        &module,
        substrate.stream(),
        &mut out0,
        &mut out1,
        &weights0,
        &weights1,
        &xq,
        &xscale,
        IN_DIM,
        OUT0_DIM,
        OUT1_DIM,
    )?;
    substrate.flush_commands()?;
    assert_close(&substrate.download(&out0)?, &expected0);
    assert_close(&substrate.download(&out1)?, &expected1);

    let residual_values = residual_hc_values();
    let split_values = split_values();
    let block_add_values = block_add_values();
    let residual_hc = substrate.upload(&residual_values)?;
    let split = substrate.upload(&split_values)?;
    let block_add = substrate.upload(&block_add_values)?;

    let mut block_out_add = substrate.zeroed::<f32>(OUT0_DIM as usize)?;
    let mut out_hc_add = substrate.zeroed::<f32>((N_HC as u64 * OUT0_DIM) as usize)?;
    matmul_q8_hc_expand(
        &module,
        substrate.stream(),
        &mut block_out_add,
        &mut out_hc_add,
        &weights0,
        &xq,
        &xscale,
        &block_add,
        &residual_hc,
        &split,
        IN_DIM,
        N_EMBD,
        N_HC,
        true,
    )?;
    substrate.flush_commands()?;
    assert_close(&substrate.download(&block_out_add)?, &expected0);
    assert_close(
        &substrate.download(&out_hc_add)?,
        &expected_hc_expand(
            &expected0,
            &block_add_values,
            &residual_values,
            &split_values,
            true,
        ),
    );

    let mut block_out_plain = substrate.zeroed::<f32>(OUT0_DIM as usize)?;
    let mut out_hc_plain = substrate.zeroed::<f32>((N_HC as u64 * OUT0_DIM) as usize)?;
    matmul_q8_hc_expand(
        &module,
        substrate.stream(),
        &mut block_out_plain,
        &mut out_hc_plain,
        &weights0,
        &xq,
        &xscale,
        &block_add,
        &residual_hc,
        &split,
        IN_DIM,
        N_EMBD,
        N_HC,
        false,
    )?;
    substrate.end_commands()?;
    assert_close(&substrate.download(&block_out_plain)?, &expected0);
    assert_close(
        &substrate.download(&out_hc_plain)?,
        &expected_hc_expand(
            &expected0,
            &block_add_values,
            &residual_values,
            &split_values,
            false,
        ),
    );

    let mut short_hc = substrate.zeroed::<f32>((N_HC as u64 * OUT0_DIM - 1) as usize)?;
    assert!(matches!(
        matmul_q8_hc_expand(
            &module,
            substrate.stream(),
            &mut block_out_plain,
            &mut short_hc,
            &weights0,
            &xq,
            &xscale,
            &block_add,
            &residual_hc,
            &split,
            IN_DIM,
            N_EMBD,
            N_HC,
            true,
        ),
        Err(Q8SpecializedMatmulError::InvalidShape)
    ));

    assert_eq!(blocks, 2);
    println!(
        "{{\"milestone\":\"M14.3d3\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"paired_unequal_width_output_matches\":true,\"hc_expand_block_output_matches\":true,\"hc_expand_add_output_matches\":true,\"hc_expand_without_add_output_matches\":true,\"partial_block_matches\":true,\"invalid_shape_rejected\":true,\"uses_warp_shuffle_reduction\":true,\"owns_matmul_q8_0_pair_preq_warp8_kernel\":{},\"owns_matmul_q8_0_hc_expand_preq_warp8_kernel\":{},\"owns_hc_expand_optional_block_add\":{},\"owns_dp4a_acceleration\":{},\"owns_q8_matmul_dispatch_policy\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3D3_SCOPE.owns_matmul_q8_0_pair_preq_warp8_kernel,
        M14_3D3_SCOPE.owns_matmul_q8_0_hc_expand_preq_warp8_kernel,
        M14_3D3_SCOPE.owns_hc_expand_optional_block_add,
        M14_3D3_SCOPE.owns_dp4a_acceleration,
        M14_3D3_SCOPE.owns_q8_matmul_dispatch_policy,
        M14_3D3_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_q8_pair(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out0: &mut DeviceBuffer<f32>,
    out1: &mut DeviceBuffer<f32>,
    weights0: &DeviceBuffer<u8>,
    weights1: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
    out0_dim: u64,
    out1_dim: u64,
) -> Result<(), Q8SpecializedMatmulError> {
    validate_q8_input(weights0, xq, xscale, in_dim, out0_dim)?;
    validate_q8_input(weights1, xq, xscale, in_dim, out1_dim)?;
    if out0.len() < out0_dim as usize || out1.len() < out1_dim as usize {
        return Err(Q8SpecializedMatmulError::InvalidShape);
    }
    module
        .matmul_q8_0_pair_preq_warp8_kernel(
            stream,
            warp8_launch(out0_dim.max(out1_dim))?,
            in_dim,
            out0_dim,
            out1_dim,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights0,
            weights1,
            xq,
            xscale,
            out0,
            out1,
        )
        .map_err(Q8SpecializedMatmulError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_q8_hc_expand(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    block_out: &mut DeviceBuffer<f32>,
    out_hc: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    block_add: &DeviceBuffer<f32>,
    residual_hc: &DeviceBuffer<f32>,
    split: &DeviceBuffer<f32>,
    in_dim: u64,
    n_embd: u32,
    n_hc: u32,
    has_add: bool,
) -> Result<(), Q8SpecializedMatmulError> {
    let out_dim = u64::from(n_embd);
    validate_q8_input(weights, xq, xscale, in_dim, out_dim)?;
    let hc_count = u64::from(n_embd)
        .checked_mul(u64::from(n_hc))
        .ok_or(Q8SpecializedMatmulError::InvalidShape)?;
    let split_count = u64::from(2 * n_hc + n_hc * n_hc);
    if n_embd == 0
        || n_hc == 0
        || block_out.len() < out_dim as usize
        || out_hc.len() < hc_count as usize
        || block_add.len() < out_dim as usize
        || residual_hc.len() < hc_count as usize
        || split.len() < split_count as usize
    {
        return Err(Q8SpecializedMatmulError::InvalidShape);
    }
    module
        .matmul_q8_0_hc_expand_preq_warp8_kernel(
            stream,
            warp8_launch(out_dim)?,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            u32::from(has_add),
            weights,
            xq,
            xscale,
            block_add,
            residual_hc,
            split,
            block_out,
            out_hc,
        )
        .map_err(Q8SpecializedMatmulError::Driver)
}

fn validate_q8_input(
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
) -> Result<(), Q8SpecializedMatmulError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let weight_bytes = out_dim
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(Q8_BLOCK_BYTES))
        .ok_or(Q8SpecializedMatmulError::InvalidShape)?;
    let quantized = blocks
        .checked_mul(Q8_BLOCK_SIZE)
        .ok_or(Q8SpecializedMatmulError::InvalidShape)?;
    if in_dim == 0
        || out_dim == 0
        || weights.len() < weight_bytes as usize
        || xq.len() < quantized as usize
        || xscale.len() < blocks as usize
    {
        return Err(Q8SpecializedMatmulError::InvalidShape);
    }
    Ok(())
}

fn warp8_launch(out_dim: u64) -> Result<LaunchConfig, Q8SpecializedMatmulError> {
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(out_dim.div_ceil(8))
                .map_err(|_| Q8SpecializedMatmulError::InvalidShape)?,
            1,
            1,
        ),
        block_dim: (WARP8_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn packed_weights(out_dim: u64, seed: usize) -> Vec<u8> {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE) as usize;
    let scales = [0.25_f32, 0.5, 1.0, 2.0];
    let mut packed = Vec::with_capacity(out_dim as usize * blocks * Q8_BLOCK_BYTES as usize);
    for row in 0..out_dim as usize {
        for block in 0..blocks {
            let scale = scales[(row + block + seed) % scales.len()] as f16;
            packed.extend_from_slice(&scale.to_bits().to_le_bytes());
            for lane in 0..32 {
                let value = ((row * 7 + block * 11 + lane * 3 + seed) % 17) as i8 - 8;
                packed.push(value as u8);
            }
        }
    }
    packed
}

fn quantized_values() -> Vec<i8> {
    let mut values = vec![0_i8; IN_DIM.div_ceil(Q8_BLOCK_SIZE) as usize * Q8_BLOCK_SIZE as usize];
    for (index, value) in values.iter_mut().enumerate() {
        *value = ((index * 5 + 3) % 31) as i8 - 15;
    }
    values[IN_DIM as usize..].fill(0);
    values
}

fn expected_matmul(weights: &[u8], out_dim: u64, xq: &[i8], xscale: &[f32]) -> Vec<f32> {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE);
    let mut output = vec![0.0_f32; out_dim as usize];
    for row in 0..out_dim {
        let mut acc = 0.0_f32;
        for block in 0..blocks {
            let count = (IN_DIM - block * Q8_BLOCK_SIZE).min(Q8_BLOCK_SIZE);
            let weight_base = ((row * blocks + block) * Q8_BLOCK_BYTES) as usize;
            let weight_scale = f16::from_bits(u16::from_le_bytes([
                weights[weight_base],
                weights[weight_base + 1],
            ])) as f32;
            let xq_base = (block * Q8_BLOCK_SIZE) as usize;
            let mut dot = 0_i32;
            for lane in 0..count as usize {
                dot += (weights[weight_base + 2 + lane] as i8 as i32) * xq[xq_base + lane] as i32;
            }
            acc += weight_scale * xscale[block as usize] * dot as f32;
        }
        output[row as usize] = acc;
    }
    output
}

fn residual_hc_values() -> Vec<f32> {
    let mut values = Vec::with_capacity(N_HC as usize * N_EMBD as usize);
    for hc in 0..N_HC {
        for row in 0..N_EMBD {
            values.push(hc as f32 * 0.25 + row as f32 * 0.05);
        }
    }
    values
}

fn split_values() -> Vec<f32> {
    let mut values = vec![0.0_f32; (2 * N_HC + N_HC * N_HC) as usize];
    values[N_HC as usize..(2 * N_HC) as usize].copy_from_slice(&[1.0, 0.5, -0.25, 0.125]);
    for src in 0..N_HC as usize {
        for dst in 0..N_HC as usize {
            values[(2 * N_HC) as usize + dst + src * N_HC as usize] =
                (src as f32 + 1.0) * (dst as f32 + 1.0) * 0.01;
        }
    }
    values
}

fn block_add_values() -> Vec<f32> {
    (0..N_EMBD).map(|row| (row as f32 + 1.0) * 0.125).collect()
}

fn expected_hc_expand(
    block_out: &[f32],
    block_add: &[f32],
    residual_hc: &[f32],
    split: &[f32],
    has_add: bool,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; N_HC as usize * N_EMBD as usize];
    for row in 0..N_EMBD as usize {
        let block_value = block_out[row] + if has_add { block_add[row] } else { 0.0 };
        for dst in 0..N_HC as usize {
            let mut acc = block_value * split[N_HC as usize + dst];
            for src in 0..N_HC as usize {
                acc += split[(2 * N_HC) as usize + dst + src * N_HC as usize]
                    * residual_hc[src * N_EMBD as usize + row];
            }
            output[dst * N_EMBD as usize + row] = acc;
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual_value - *expected_value).abs() <= 1.0e-5,
            "value mismatch at {index}: {actual_value} != {expected_value}"
        );
    }
}

#[derive(Debug)]
enum Q8SpecializedMatmulError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for Q8SpecializedMatmulError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("specialized Q8 matmul tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Q8SpecializedMatmulError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
