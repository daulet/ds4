#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3C1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn matmul_f16_kernel(
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
    pub fn matmul_f32_kernel(
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

const THREADS_PER_BLOCK: u32 = 256;
const IN_DIM: u64 = 5;
const OUT_DIM: u64 = 3;
const N_TOK: u64 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let f16_weights_values = [
        f16::from_bits(0x3800), // 0.5
        f16::from_bits(0xbc00), // -1.0
        f16::from_bits(0x4000), // 2.0
        f16::from_bits(0x3400), // 0.25
        f16::from_bits(0xc200), // -3.0
        f16::from_bits(0x3c00), // 1.0
        f16::from_bits(0x3a00), // 0.75
        f16::from_bits(0xb800), // -0.5
        f16::from_bits(0x4400), // 4.0
        f16::from_bits(0xbc00), // -1.0
        f16::from_bits(0xc000), // -2.0
        f16::from_bits(0x4200), // 3.0
        f16::from_bits(0x3000), // 0.125
        f16::from_bits(0x3e00), // 1.5
        f16::from_bits(0xb400), // -0.25
    ];
    let f32_weights_values = [
        0.375_f32, -1.25, 2.125, 0.5, -3.75, 1.5, 0.625, -0.875, 4.25, -1.125, -2.5, 3.25, 0.0625,
        1.75, -0.375,
    ];
    let x_values = [1.0_f32, -2.0, 0.5, 3.0, -1.5, -0.25, 1.25, 2.5, -1.0, 0.75];
    let f16_weights = substrate.upload(&f16_weights_values)?;
    let f32_weights = substrate.upload(&f32_weights_values)?;
    let x = substrate.upload(&x_values)?;

    let mut f16_out = substrate.zeroed::<f32>((OUT_DIM * N_TOK) as usize)?;
    matmul_f16_tensor(
        &module,
        substrate.stream(),
        &mut f16_out,
        &f16_weights,
        &x,
        IN_DIM,
        OUT_DIM,
        N_TOK,
    )?;
    substrate.flush_commands()?;
    assert_close(
        &substrate.download(&f16_out)?,
        &expected_f16_matmul(&f16_weights_values, &x_values),
        1.0e-6,
    );

    let mut f32_out = substrate.zeroed::<f32>((OUT_DIM * N_TOK) as usize)?;
    matmul_f32_tensor(
        &module,
        substrate.stream(),
        &mut f32_out,
        &f32_weights,
        &x,
        IN_DIM,
        OUT_DIM,
        N_TOK,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&f32_out)?,
        &expected_f32_matmul(&f32_weights_values, &x_values),
        1.0e-6,
    );

    let mut short_out = substrate.zeroed::<f32>((OUT_DIM * N_TOK - 1) as usize)?;
    assert!(matches!(
        matmul_f32_tensor(
            &module,
            substrate.stream(),
            &mut short_out,
            &f32_weights,
            &x,
            IN_DIM,
            OUT_DIM,
            N_TOK,
        ),
        Err(DenseProjectionError::InvalidShape)
    ));
    assert!(matches!(
        matmul_f16_tensor(
            &module,
            substrate.stream(),
            &mut f16_out,
            &f16_weights,
            &x,
            0,
            OUT_DIM,
            N_TOK,
        ),
        Err(DenseProjectionError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3c1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"f16_base_projection_output_matches\":true,\"f32_base_projection_output_matches\":true,\"multi_token_stride_matches\":true,\"invalid_shape_rejected\":true,\"uses_primitive_f16_weight_loads\":true,\"owns_matmul_f16_kernel\":{},\"owns_matmul_f32_kernel\":{},\"owns_ordered_or_pair_f16_kernels\":{},\"owns_f16_or_cublas_dispatch_policy\":{},\"owns_q8_conversion_or_matmul_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3C1_SCOPE.owns_matmul_f16_kernel,
        M14_3C1_SCOPE.owns_matmul_f32_kernel,
        M14_3C1_SCOPE.owns_ordered_or_pair_f16_kernels,
        M14_3C1_SCOPE.owns_f16_or_cublas_dispatch_policy,
        M14_3C1_SCOPE.owns_q8_conversion_or_matmul_kernels,
        M14_3C1_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_f16_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<f16>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), DenseProjectionError> {
    validate_shapes(out, weights.len() as u64, x, in_dim, out_dim, n_tok)?;
    module
        .matmul_f16_kernel(
            stream,
            launch_config(out_dim, n_tok)?,
            in_dim,
            out_dim,
            n_tok,
            weights,
            x,
            out,
        )
        .map_err(DenseProjectionError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_f32_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), DenseProjectionError> {
    validate_shapes(out, weights.len() as u64, x, in_dim, out_dim, n_tok)?;
    module
        .matmul_f32_kernel(
            stream,
            launch_config(out_dim, n_tok)?,
            in_dim,
            out_dim,
            n_tok,
            weights,
            x,
            out,
        )
        .map_err(DenseProjectionError::Driver)
}

fn validate_shapes(
    out: &DeviceBuffer<f32>,
    weight_len: u64,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), DenseProjectionError> {
    let weights = in_dim
        .checked_mul(out_dim)
        .ok_or(DenseProjectionError::InvalidShape)?;
    let inputs = in_dim
        .checked_mul(n_tok)
        .ok_or(DenseProjectionError::InvalidShape)?;
    let outputs = out_dim
        .checked_mul(n_tok)
        .ok_or(DenseProjectionError::InvalidShape)?;
    if in_dim == 0
        || out_dim == 0
        || n_tok == 0
        || weights > weight_len
        || inputs > x.len() as u64
        || outputs > out.len() as u64
    {
        return Err(DenseProjectionError::InvalidShape);
    }
    Ok(())
}

fn launch_config(out_dim: u64, n_tok: u64) -> Result<LaunchConfig, DenseProjectionError> {
    let grid_x = u32::try_from(out_dim).map_err(|_| DenseProjectionError::InvalidShape)?;
    let grid_y = u32::try_from(n_tok).map_err(|_| DenseProjectionError::InvalidShape)?;
    Ok(LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn expected_f16_matmul(weights: &[f16], x: &[f32]) -> Vec<f32> {
    expected_matmul(
        &weights
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        x,
    )
}

fn expected_f32_matmul(weights: &[f32], x: &[f32]) -> Vec<f32> {
    expected_matmul(weights, x)
}

fn expected_matmul(weights: &[f32], x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity((OUT_DIM * N_TOK) as usize);
    for token in x.chunks_exact(IN_DIM as usize) {
        for row in weights.chunks_exact(IN_DIM as usize) {
            out.push(row.iter().zip(token).map(|(w, value)| w * value).sum());
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum DenseProjectionError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for DenseProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("dense projection tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DenseProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
