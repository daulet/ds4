#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, integer, kernel, thread, DisjointSlice};
use ds4_cuda::{
    q8_dp4a_enabled, select_q8_matmul_path, substrate::CudaOxideSubstrate, Q8MatmulDispatchOptions,
    Q8MatmulPath, M14_3D4_SCOPE,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn matmul_q8_0_preq_dp4a_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
        weights: &[u8],
        xq: &[i8],
        xscale: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
            + thread::threadIdx_x() as u64;
        if row >= out_dim {
            return;
        }
        let mut acc = 0.0_f32;
        let mut block = 0_u64;
        while block < blocks {
            let start = block * Q8_BLOCK_SIZE;
            let remaining = in_dim - start;
            let count = if remaining < Q8_BLOCK_SIZE {
                remaining
            } else {
                Q8_BLOCK_SIZE
            };
            let weight_base = ((row * blocks + block) * Q8_BLOCK_BYTES) as usize;
            let scale_bits = weights[weight_base] as u16 | ((weights[weight_base + 1] as u16) << 8);
            let weight_scale = f16::from_bits(scale_bits) as f32;
            let xq_base = (block * Q8_BLOCK_SIZE) as usize;
            let mut dot = 0_i32;
            if count == Q8_BLOCK_SIZE {
                let mut lane = 0_usize;
                while lane < Q8_BLOCK_SIZE as usize {
                    let weight_index = weight_base + 2 + lane;
                    let weight_word = (weights[weight_index] as u32
                        | (weights[weight_index + 1] as u32) << 8
                        | (weights[weight_index + 2] as u32) << 16
                        | (weights[weight_index + 3] as u32) << 24)
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
            acc += weight_scale * xscale[block as usize] * dot as f32;
            block += 1;
        }
        unsafe {
            *out.get_unchecked_mut(row as usize) = acc;
        }
    }
}

const Q8_BLOCK_SIZE: u64 = 32;
const Q8_BLOCK_BYTES: u64 = 34;
const OUT_DIM: u64 = 5;
const FULL_IN_DIM: u64 = 32;
const TAIL_IN_DIM: u64 = 35;
const THREADS: u32 = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    run_case(&substrate, &module, FULL_IN_DIM)?;
    run_case(&substrate, &module, TAIL_IN_DIM)?;
    substrate.end_commands()?;

    let batch = Q8MatmulDispatchOptions {
        cublas_ready: false,
        expanded_f32_blas_ready: false,
        expanded_f16_blas_ready: false,
        n_tokens: 2,
        blocks: 2,
        no_batch_warp: false,
    };
    assert_eq!(
        select_q8_matmul_path(Q8MatmulDispatchOptions {
            n_tokens: 1,
            ..batch
        }),
        Q8MatmulPath::PrequantizedWarp8
    );
    assert_eq!(
        select_q8_matmul_path(batch),
        Q8MatmulPath::PrequantizedBatchWarp8
    );
    assert_eq!(
        select_q8_matmul_path(Q8MatmulDispatchOptions {
            no_batch_warp: true,
            ..batch
        }),
        Q8MatmulPath::PrequantizedGeneric
    );
    assert!(q8_dp4a_enabled(false));
    assert!(!q8_dp4a_enabled(true));

    println!(
        "{{\"milestone\":\"M14.3d4\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"dp4a_full_block_output_matches\":true,\"scalar_tail_fallback_output_matches\":true,\"single_token_warp8_dispatch_matches\":true,\"batched_warp8_dispatch_matches\":true,\"generic_dispatch_matches\":true,\"dp4a_disable_policy_matches\":true,\"uses_cuda_oxide_dp4a_i8\":{},\"owns_dp4a_acceleration\":{},\"owns_q8_matmul_dispatch_policy\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3D4_SCOPE.owns_cuda_oxide_dp4a_i8_intrinsic,
        M14_3D4_SCOPE.owns_dp4a_acceleration,
        M14_3D4_SCOPE.owns_q8_matmul_dispatch_policy,
        M14_3D4_SCOPE.owns_runtime_graph_integration,
        M14_3D4_SCOPE.changes_default_route,
    );
    Ok(())
}

fn run_case(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    in_dim: u64,
) -> Result<(), Q8Dp4aError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let weights_values = packed_weights(in_dim);
    let xq_values = quantized_values(in_dim);
    let xscale_values = (0..blocks)
        .map(|block| 0.5_f32 + block as f32 * 0.25)
        .collect::<Vec<_>>();
    let expected = expected_matmul(in_dim, &weights_values, &xq_values, &xscale_values);
    let weights = substrate
        .upload(&weights_values)
        .map_err(Q8Dp4aError::Driver)?;
    let xq = substrate.upload(&xq_values).map_err(Q8Dp4aError::Driver)?;
    let xscale = substrate
        .upload(&xscale_values)
        .map_err(Q8Dp4aError::Driver)?;
    let mut out = substrate
        .zeroed::<f32>(OUT_DIM as usize)
        .map_err(Q8Dp4aError::Driver)?;
    matmul_dp4a(
        module,
        substrate.stream(),
        &mut out,
        &weights,
        &xq,
        &xscale,
        in_dim,
    )?;
    substrate.flush_commands().map_err(Q8Dp4aError::Driver)?;
    assert_close(
        &substrate.download(&out).map_err(Q8Dp4aError::Driver)?,
        &expected,
    );
    Ok(())
}

fn matmul_dp4a(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    in_dim: u64,
) -> Result<(), Q8Dp4aError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let weight_bytes = OUT_DIM
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(Q8_BLOCK_BYTES))
        .ok_or(Q8Dp4aError::InvalidShape)?;
    if in_dim == 0
        || weights.len() < weight_bytes as usize
        || xq.len() < (blocks * Q8_BLOCK_SIZE) as usize
        || xscale.len() < blocks as usize
        || out.len() < OUT_DIM as usize
    {
        return Err(Q8Dp4aError::InvalidShape);
    }
    module
        .matmul_q8_0_preq_dp4a_kernel(
            stream,
            LaunchConfig {
                grid_dim: (
                    u32::try_from(OUT_DIM.div_ceil(u64::from(THREADS))).unwrap(),
                    1,
                    1,
                ),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            in_dim,
            OUT_DIM,
            blocks,
            weights,
            xq,
            xscale,
            out,
        )
        .map_err(Q8Dp4aError::Driver)
}

fn packed_weights(in_dim: u64) -> Vec<u8> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let mut packed = Vec::with_capacity((OUT_DIM * blocks * Q8_BLOCK_BYTES) as usize);
    for row in 0..OUT_DIM {
        for block in 0..blocks {
            let scale = (0.25_f32 + (row + block) as f32 * 0.125) as f16;
            packed.extend_from_slice(&scale.to_bits().to_le_bytes());
            for lane in 0..Q8_BLOCK_SIZE {
                packed.push((((row * 7 + block * 11 + lane * 3) % 21) as i8 - 10) as u8);
            }
        }
    }
    packed
}

fn quantized_values(in_dim: u64) -> Vec<i8> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let mut values = vec![0_i8; (blocks * Q8_BLOCK_SIZE) as usize];
    for (index, value) in values.iter_mut().take(in_dim as usize).enumerate() {
        *value = ((index * 5 + 3) % 31) as i8 - 15;
    }
    values
}

fn expected_matmul(in_dim: u64, weights: &[u8], xq: &[i8], xscale: &[f32]) -> Vec<f32> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let mut output = vec![0.0_f32; OUT_DIM as usize];
    for row in 0..OUT_DIM {
        let mut acc = 0.0_f32;
        for block in 0..blocks {
            let count = (in_dim - block * Q8_BLOCK_SIZE).min(Q8_BLOCK_SIZE);
            let weight_base = ((row * blocks + block) * Q8_BLOCK_BYTES) as usize;
            let scale = f16::from_bits(u16::from_le_bytes([
                weights[weight_base],
                weights[weight_base + 1],
            ])) as f32;
            let xq_base = (block * Q8_BLOCK_SIZE) as usize;
            let mut dot = 0_i32;
            for lane in 0..count as usize {
                dot += (weights[weight_base + 2 + lane] as i8 as i32) * xq[xq_base + lane] as i32;
            }
            acc += scale * xscale[block as usize] * dot as f32;
        }
        output[row as usize] = acc;
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
enum Q8Dp4aError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for Q8Dp4aError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("Q8 DP4A tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Q8Dp4aError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
