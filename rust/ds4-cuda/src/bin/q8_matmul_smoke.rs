#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3D2_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn matmul_q8_0_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        blocks: u64,
        weights: &[u8],
        x: &[f32],
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
            let start = block * 32;
            let remaining = in_dim - start;
            let count = if remaining < 32 { remaining } else { 32 };
            let x_base = token as usize * in_dim as usize + start as usize;
            let mut max_value = 0.0_f32;
            let mut lane = 0_u64;
            while lane < count {
                let value = x[x_base + lane as usize];
                let magnitude = if value < 0.0 { -value } else { value };
                if magnitude > max_value {
                    max_value = magnitude;
                }
                lane += 1;
            }
            let xscale = max_value / 127.0;
            let inverse = if xscale != 0.0 { 1.0 / xscale } else { 0.0 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let mut dot = 0_i32;
            lane = 0;
            while lane < count {
                let scaled = x[x_base + lane as usize] * inverse;
                let lower = scaled.floor();
                let fraction = scaled - lower;
                let mut rounded = lower as i32;
                if fraction > 0.5 || (fraction == 0.5 && (rounded & 1) != 0) {
                    rounded += 1;
                }
                if rounded > 127 {
                    rounded = 127;
                } else if rounded < -128 {
                    rounded = -128;
                }
                dot += (weights[weight_base + 2 + lane as usize] as i8 as i32) * rounded;
                lane += 1;
            }
            acc += weight_scale * xscale * dot as f32;
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
    pub fn matmul_q8_0_preq_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        blocks: u64,
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
            let start = block * 32;
            let remaining = in_dim - start;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = ((token * blocks + block) * 32) as usize;
            let mut dot = 0_i32;
            let mut lane = 0_u64;
            while lane < count {
                dot += (weights[weight_base + 2 + lane as usize] as i8 as i32)
                    * xq[xq_base + lane as usize] as i32;
                lane += 1;
            }
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
    pub fn matmul_q8_0_preq_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
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
            unsafe {
                *out.get_unchecked_mut(row as usize) = acc;
            }
        }
    }

    #[kernel]
    pub fn matmul_q8_0_preq_batch_warp8_kernel(
        in_dim: u64,
        out_dim: u64,
        n_tok: u64,
        blocks: u64,
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
            let start = block * 32;
            let remaining = in_dim - start;
            let count = if remaining < 32 { remaining } else { 32 };
            let weight_base = ((row * blocks + block) * 34) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = ((token * blocks + block) * 32) as usize;
            let mut dot = 0_i32;
            let mut element = 0_u64;
            while element < count {
                dot += (weights[weight_base + 2 + element as usize] as i8 as i32)
                    * xq[xq_base + element as usize] as i32;
                element += 1;
            }
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
}

const Q8_BLOCK_SIZE: u64 = 32;
const Q8_BLOCK_BYTES: u64 = 34;
const REDUCE_THREADS: u32 = 256;
const WARP8_THREADS: u32 = 256;
const IN_DIM: u64 = 35;
const OUT_DIM: u64 = 10;
const N_TOK: u64 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module =
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_q8_matmul_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let packed_values = packed_weights();
    let x_values = activation_values();
    let (xq_values, xscale_values) = quantized_activations(&x_values);
    let expected = expected_matmul(&packed_values, &xq_values, &xscale_values);
    let weights = substrate.upload(&packed_values)?;
    let x = substrate.upload(&x_values)?;
    let xq = substrate.upload(&xq_values)?;
    let xscale = substrate.upload(&xscale_values)?;

    let mut direct = substrate.zeroed::<f32>((N_TOK * OUT_DIM) as usize)?;
    matmul_q8_direct(
        &module,
        substrate.stream(),
        &mut direct,
        &weights,
        &x,
        IN_DIM,
        OUT_DIM,
        N_TOK,
    )?;
    substrate.flush_commands()?;
    assert_close(&substrate.download(&direct)?, &expected);

    let mut generic = substrate.zeroed::<f32>((N_TOK * OUT_DIM) as usize)?;
    matmul_q8_preq(
        &module,
        substrate.stream(),
        &mut generic,
        &weights,
        &xq,
        &xscale,
        IN_DIM,
        OUT_DIM,
        N_TOK,
    )?;
    substrate.flush_commands()?;
    assert_close(&substrate.download(&generic)?, &expected);

    let mut single_token = substrate.zeroed::<f32>(OUT_DIM as usize)?;
    matmul_q8_preq_warp8(
        &module,
        substrate.stream(),
        &mut single_token,
        &weights,
        &xq,
        &xscale,
        IN_DIM,
        OUT_DIM,
    )?;
    substrate.flush_commands()?;
    assert_close(
        &substrate.download(&single_token)?,
        &expected[..OUT_DIM as usize],
    );

    let mut batch = substrate.zeroed::<f32>((N_TOK * OUT_DIM) as usize)?;
    matmul_q8_preq_batch_warp8(
        &module,
        substrate.stream(),
        &mut batch,
        &weights,
        &xq,
        &xscale,
        IN_DIM,
        OUT_DIM,
        N_TOK,
    )?;
    substrate.end_commands()?;
    assert_close(&substrate.download(&batch)?, &expected);

    let mut short_output = substrate.zeroed::<f32>((N_TOK * OUT_DIM - 1) as usize)?;
    assert!(matches!(
        matmul_q8_preq(
            &module,
            substrate.stream(),
            &mut short_output,
            &weights,
            &xq,
            &xscale,
            IN_DIM,
            OUT_DIM,
            N_TOK,
        ),
        Err(Q8MatmulError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3d2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"direct_quantizing_output_matches\":true,\"prequantized_generic_output_matches\":true,\"prequantized_single_token_warp8_output_matches\":true,\"prequantized_batch_warp8_output_matches\":true,\"partial_block_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_matmul_q8_0_kernel\":{},\"owns_matmul_q8_0_preq_kernel\":{},\"owns_matmul_q8_0_preq_warp8_kernel\":{},\"owns_matmul_q8_0_preq_batch_warp8_kernel\":{},\"owns_dp4a_acceleration\":{},\"owns_pair_or_hc_expand_kernels\":{},\"owns_q8_matmul_dispatch_policy\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3D2_SCOPE.owns_matmul_q8_0_kernel,
        M14_3D2_SCOPE.owns_matmul_q8_0_preq_kernel,
        M14_3D2_SCOPE.owns_matmul_q8_0_preq_warp8_kernel,
        M14_3D2_SCOPE.owns_matmul_q8_0_preq_batch_warp8_kernel,
        M14_3D2_SCOPE.owns_dp4a_acceleration,
        M14_3D2_SCOPE.owns_pair_or_hc_expand_kernels,
        M14_3D2_SCOPE.owns_q8_matmul_dispatch_policy,
        M14_3D2_SCOPE.changes_default_route,
    );
    Ok(())
}

fn matmul_q8_direct(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), Q8MatmulError> {
    validate_direct_shapes(out, weights, x, in_dim, out_dim, n_tok)?;
    module
        .matmul_q8_0_kernel(
            stream,
            reduction_launch(out_dim, n_tok)?,
            in_dim,
            out_dim,
            n_tok,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights,
            x,
            out,
        )
        .map_err(Q8MatmulError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_q8_preq(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), Q8MatmulError> {
    validate_preq_shapes(out, weights, xq, xscale, in_dim, out_dim, n_tok)?;
    module
        .matmul_q8_0_preq_kernel(
            stream,
            reduction_launch(out_dim, n_tok)?,
            in_dim,
            out_dim,
            n_tok,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights,
            xq,
            xscale,
            out,
        )
        .map_err(Q8MatmulError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_q8_preq_warp8(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
) -> Result<(), Q8MatmulError> {
    validate_preq_shapes(out, weights, xq, xscale, in_dim, out_dim, 1)?;
    module
        .matmul_q8_0_preq_warp8_kernel(
            stream,
            warp8_launch(out_dim, 1)?,
            in_dim,
            out_dim,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights,
            xq,
            xscale,
            out,
        )
        .map_err(Q8MatmulError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_q8_preq_batch_warp8(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), Q8MatmulError> {
    validate_preq_shapes(out, weights, xq, xscale, in_dim, out_dim, n_tok)?;
    module
        .matmul_q8_0_preq_batch_warp8_kernel(
            stream,
            warp8_launch(out_dim, n_tok)?,
            in_dim,
            out_dim,
            n_tok,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights,
            xq,
            xscale,
            out,
        )
        .map_err(Q8MatmulError::Driver)
}

fn validate_direct_shapes(
    out: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), Q8MatmulError> {
    let expected_x = in_dim
        .checked_mul(n_tok)
        .ok_or(Q8MatmulError::InvalidShape)?;
    validate_common(out, weights, in_dim, out_dim, n_tok)?;
    if expected_x > x.len() as u64 {
        return Err(Q8MatmulError::InvalidShape);
    }
    Ok(())
}

fn validate_preq_shapes(
    out: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), Q8MatmulError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let expected_xq = n_tok
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(Q8_BLOCK_SIZE))
        .ok_or(Q8MatmulError::InvalidShape)?;
    let expected_scales = n_tok
        .checked_mul(blocks)
        .ok_or(Q8MatmulError::InvalidShape)?;
    validate_common(out, weights, in_dim, out_dim, n_tok)?;
    if expected_xq > xq.len() as u64 || expected_scales > xscale.len() as u64 {
        return Err(Q8MatmulError::InvalidShape);
    }
    Ok(())
}

fn validate_common(
    out: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), Q8MatmulError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let expected_out = out_dim
        .checked_mul(n_tok)
        .ok_or(Q8MatmulError::InvalidShape)?;
    let expected_weights = out_dim
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(Q8_BLOCK_BYTES))
        .ok_or(Q8MatmulError::InvalidShape)?;
    if in_dim == 0
        || out_dim == 0
        || n_tok == 0
        || expected_out > out.len() as u64
        || expected_weights > weights.len() as u64
    {
        return Err(Q8MatmulError::InvalidShape);
    }
    Ok(())
}

fn reduction_launch(out_dim: u64, n_tok: u64) -> Result<LaunchConfig, Q8MatmulError> {
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(out_dim).map_err(|_| Q8MatmulError::InvalidShape)?,
            u32::try_from(n_tok).map_err(|_| Q8MatmulError::InvalidShape)?,
            1,
        ),
        block_dim: (REDUCE_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn warp8_launch(out_dim: u64, n_tok: u64) -> Result<LaunchConfig, Q8MatmulError> {
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(out_dim.div_ceil(8)).map_err(|_| Q8MatmulError::InvalidShape)?,
            u32::try_from(n_tok).map_err(|_| Q8MatmulError::InvalidShape)?,
            1,
        ),
        block_dim: (WARP8_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn packed_weights() -> Vec<u8> {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE) as usize;
    let mut packed = Vec::with_capacity(OUT_DIM as usize * blocks * Q8_BLOCK_BYTES as usize);
    let scales = [0.25_f32, 0.5, 1.0, 2.0];
    for row in 0..OUT_DIM as usize {
        for block in 0..blocks {
            let scale = scales[(row + block) % scales.len()] as f16;
            packed.extend_from_slice(&scale.to_bits().to_le_bytes());
            for lane in 0..32 {
                let value = ((row * 7 + block * 11 + lane * 3) % 17) as i8 - 8;
                packed.push(value as u8);
            }
        }
    }
    packed
}

fn activation_values() -> Vec<f32> {
    let mut values = vec![0.0_f32; (N_TOK * IN_DIM) as usize];
    values[..8].copy_from_slice(&[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0, -127.0]);
    values[32..35].copy_from_slice(&[63.5, -63.5, 127.0]);
    values[35..43].copy_from_slice(&[3.5, 4.5, -3.5, -4.5, 126.0, -127.0, 1.0, -1.0]);
    values[67..70].copy_from_slice(&[-31.5, 31.5, -127.0]);
    values
}

fn quantized_activations(input: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE) as usize;
    let mut xq = vec![0_i8; N_TOK as usize * blocks * Q8_BLOCK_SIZE as usize];
    let mut xscale = vec![0.0_f32; N_TOK as usize * blocks];
    for (token, row) in input.chunks_exact(IN_DIM as usize).enumerate() {
        for (block, values) in row.chunks(32).enumerate() {
            let scale = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max) / 127.0;
            xscale[token * blocks + block] = scale;
            let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            let base = (token * blocks + block) * 32;
            for (lane, value) in values.iter().enumerate() {
                xq[base + lane] = clamp_i8(round_ties_even(*value * inverse));
            }
        }
    }
    (xq, xscale)
}

fn expected_matmul(packed: &[u8], xq: &[i8], xscale: &[f32]) -> Vec<f32> {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE);
    let mut output = vec![0.0_f32; (N_TOK * OUT_DIM) as usize];
    for token in 0..N_TOK {
        for row in 0..OUT_DIM {
            let mut acc = 0.0_f32;
            for block in 0..blocks {
                let count = (IN_DIM - block * Q8_BLOCK_SIZE).min(Q8_BLOCK_SIZE);
                let weight_base = ((row * blocks + block) * Q8_BLOCK_BYTES) as usize;
                let weight_scale = f16::from_bits(u16::from_le_bytes([
                    packed[weight_base],
                    packed[weight_base + 1],
                ])) as f32;
                let xq_base = ((token * blocks + block) * Q8_BLOCK_SIZE) as usize;
                let mut dot = 0_i32;
                for lane in 0..count as usize {
                    dot +=
                        (packed[weight_base + 2 + lane] as i8 as i32) * xq[xq_base + lane] as i32;
                }
                acc += weight_scale * xscale[(token * blocks + block) as usize] * dot as f32;
            }
            output[(token * OUT_DIM + row) as usize] = acc;
        }
    }
    output
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
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual_value - *expected_value).abs() <= 1.0e-5,
            "value mismatch at {index}: {actual_value} != {expected_value}"
        );
    }
}

#[derive(Debug)]
enum Q8MatmulError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for Q8MatmulError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("Q8 matmul tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Q8MatmulError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
