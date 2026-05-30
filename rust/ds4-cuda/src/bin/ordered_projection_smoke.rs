#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3C2_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn matmul_f16_serial_kernel(
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
    pub fn matmul_f16_ordered_chunks_kernel(
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
    pub fn matmul_f16_pair_ordered_chunks_kernel(
        in_dim: u64,
        out0_dim: u64,
        out1_dim: u64,
        weights0: &[f16],
        weights1: &[f16],
        x: &[f32],
        mut out0: DisjointSlice<f32>,
        mut out1: DisjointSlice<f32>,
    ) {
        static mut PARTIAL0: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut PARTIAL1: SharedArray<f32, 32> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as u64;
        if row >= out0_dim && row >= out1_dim {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let chunk = (in_dim as usize + 31) / 32;
        let start = tid * chunk;
        let mut end = start + chunk;
        if end > in_dim as usize {
            end = in_dim as usize;
        }
        let weight0_base = row as usize * in_dim as usize;
        let weight1_base = row as usize * in_dim as usize;
        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        let mut i = start;
        while i < end {
            let value = x[i];
            if row < out0_dim {
                sum0 += weights0[weight0_base + i] as f32 * value;
            }
            if row < out1_dim {
                sum1 += weights1[weight1_base + i] as f32 * value;
            }
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
                if row < out0_dim {
                    *out0.get_unchecked_mut(row as usize) = total0;
                }
                if row < out1_dim {
                    *out1.get_unchecked_mut(row as usize) = total1;
                }
            }
        }
    }
}

const ORDERED_THREADS: u32 = 32;
const IN_DIM: u64 = 37;
const OUT_DIM: u64 = 3;
const PAIR_OUT1_DIM: u64 = 2;
const N_TOK: u64 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let weights0_values = make_weights(OUT_DIM, 0);
    let weights1_values = make_weights(PAIR_OUT1_DIM, 3);
    let x_values = make_inputs(N_TOK);
    let weights0 = substrate.upload(&weights0_values)?;
    let weights1 = substrate.upload(&weights1_values)?;
    let x = substrate.upload(&x_values)?;

    let mut serial_out = substrate.zeroed::<f32>((OUT_DIM * N_TOK) as usize)?;
    matmul_f16_serial_tensor(
        &module,
        substrate.stream(),
        &mut serial_out,
        &weights0,
        &x,
        IN_DIM,
        OUT_DIM,
        N_TOK,
    )?;
    substrate.flush_commands()?;
    assert_close(
        &substrate.download(&serial_out)?,
        &expected_matmul(&weights0_values, &x_values, OUT_DIM, N_TOK),
    );

    let mut ordered_out = substrate.zeroed::<f32>(OUT_DIM as usize)?;
    matmul_f16_ordered_chunks_tensor(
        &module,
        substrate.stream(),
        &mut ordered_out,
        &weights0,
        &x,
        IN_DIM,
        OUT_DIM,
        1,
    )?;
    substrate.flush_commands()?;
    assert_close(
        &substrate.download(&ordered_out)?,
        &expected_matmul(&weights0_values, &x_values[..IN_DIM as usize], OUT_DIM, 1),
    );

    let mut pair_out0 = substrate.zeroed::<f32>(OUT_DIM as usize)?;
    let mut pair_out1 = substrate.zeroed::<f32>(PAIR_OUT1_DIM as usize)?;
    matmul_f16_pair_ordered_chunks_tensor(
        &module,
        substrate.stream(),
        &mut pair_out0,
        &mut pair_out1,
        &weights0,
        &weights1,
        &x,
        IN_DIM,
        OUT_DIM,
        PAIR_OUT1_DIM,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&pair_out0)?,
        &expected_matmul(&weights0_values, &x_values[..IN_DIM as usize], OUT_DIM, 1),
    );
    assert_close(
        &substrate.download(&pair_out1)?,
        &expected_matmul(
            &weights1_values,
            &x_values[..IN_DIM as usize],
            PAIR_OUT1_DIM,
            1,
        ),
    );

    let mut short_pair = substrate.zeroed::<f32>((PAIR_OUT1_DIM - 1) as usize)?;
    assert!(matches!(
        matmul_f16_pair_ordered_chunks_tensor(
            &module,
            substrate.stream(),
            &mut pair_out0,
            &mut short_pair,
            &weights0,
            &weights1,
            &x,
            IN_DIM,
            OUT_DIM,
            PAIR_OUT1_DIM,
        ),
        Err(OrderedProjectionError::InvalidShape)
    ));
    assert!(matches!(
        matmul_f16_ordered_chunks_tensor(
            &module,
            substrate.stream(),
            &mut ordered_out,
            &weights0,
            &x,
            0,
            OUT_DIM,
            1,
        ),
        Err(OrderedProjectionError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3c2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"serial_multi_token_output_matches\":true,\"ordered_chunk_output_matches\":true,\"paired_unequal_width_output_matches\":true,\"invalid_shape_rejected\":true,\"uses_primitive_f16_weight_loads\":true,\"owns_matmul_f16_serial_kernel\":{},\"owns_matmul_f16_ordered_chunks_kernel\":{},\"owns_matmul_f16_pair_ordered_chunks_kernel\":{},\"owns_f16_or_cublas_dispatch_policy\":{},\"owns_q8_conversion_or_matmul_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3C2_SCOPE.owns_matmul_f16_serial_kernel,
        M14_3C2_SCOPE.owns_matmul_f16_ordered_chunks_kernel,
        M14_3C2_SCOPE.owns_matmul_f16_pair_ordered_chunks_kernel,
        M14_3C2_SCOPE.owns_f16_or_cublas_dispatch_policy,
        M14_3C2_SCOPE.owns_q8_conversion_or_matmul_kernels,
        M14_3C2_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matmul_f16_serial_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<f16>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), OrderedProjectionError> {
    validate_shapes(out, weights.len() as u64, x, in_dim, out_dim, n_tok)?;
    module
        .matmul_f16_serial_kernel(
            stream,
            launch_config(out_dim, n_tok, 1)?,
            in_dim,
            out_dim,
            n_tok,
            weights,
            x,
            out,
        )
        .map_err(OrderedProjectionError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_f16_ordered_chunks_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<f16>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), OrderedProjectionError> {
    validate_shapes(out, weights.len() as u64, x, in_dim, out_dim, n_tok)?;
    module
        .matmul_f16_ordered_chunks_kernel(
            stream,
            launch_config(out_dim, n_tok, ORDERED_THREADS)?,
            in_dim,
            out_dim,
            n_tok,
            weights,
            x,
            out,
        )
        .map_err(OrderedProjectionError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn matmul_f16_pair_ordered_chunks_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out0: &mut DeviceBuffer<f32>,
    out1: &mut DeviceBuffer<f32>,
    weights0: &DeviceBuffer<f16>,
    weights1: &DeviceBuffer<f16>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out0_dim: u64,
    out1_dim: u64,
) -> Result<(), OrderedProjectionError> {
    if in_dim == 0
        || out0_dim == 0
        || out1_dim == 0
        || in_dim
            .checked_mul(out0_dim)
            .is_none_or(|needed| needed > weights0.len() as u64)
        || in_dim
            .checked_mul(out1_dim)
            .is_none_or(|needed| needed > weights1.len() as u64)
        || in_dim > x.len() as u64
        || out0_dim > out0.len() as u64
        || out1_dim > out1.len() as u64
    {
        return Err(OrderedProjectionError::InvalidShape);
    }
    let grid_x =
        u32::try_from(out0_dim.max(out1_dim)).map_err(|_| OrderedProjectionError::InvalidShape)?;
    module
        .matmul_f16_pair_ordered_chunks_kernel(
            stream,
            LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (ORDERED_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            in_dim,
            out0_dim,
            out1_dim,
            weights0,
            weights1,
            x,
            out0,
            out1,
        )
        .map_err(OrderedProjectionError::Driver)
}

fn validate_shapes(
    out: &DeviceBuffer<f32>,
    weight_len: u64,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    out_dim: u64,
    n_tok: u64,
) -> Result<(), OrderedProjectionError> {
    if in_dim == 0
        || out_dim == 0
        || n_tok == 0
        || in_dim
            .checked_mul(out_dim)
            .is_none_or(|needed| needed > weight_len)
        || in_dim
            .checked_mul(n_tok)
            .is_none_or(|needed| needed > x.len() as u64)
        || out_dim
            .checked_mul(n_tok)
            .is_none_or(|needed| needed > out.len() as u64)
    {
        return Err(OrderedProjectionError::InvalidShape);
    }
    Ok(())
}

fn launch_config(
    out_dim: u64,
    n_tok: u64,
    threads: u32,
) -> Result<LaunchConfig, OrderedProjectionError> {
    let grid_x = u32::try_from(out_dim).map_err(|_| OrderedProjectionError::InvalidShape)?;
    let grid_y = u32::try_from(n_tok).map_err(|_| OrderedProjectionError::InvalidShape)?;
    Ok(LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn make_weights(out_dim: u64, seed: usize) -> Vec<f16> {
    let bits = [
        0xc000_u16, 0xbc00, 0xb800, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000,
    ];
    (0..(out_dim * IN_DIM) as usize)
        .map(|index| f16::from_bits(bits[(index + seed) % bits.len()]))
        .collect()
}

fn make_inputs(n_tok: u64) -> Vec<f32> {
    let values = [-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    (0..(n_tok * IN_DIM) as usize)
        .map(|index| values[(index * 3 + 1) % values.len()])
        .collect()
}

fn expected_matmul(weights: &[f16], x: &[f32], out_dim: u64, n_tok: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity((out_dim * n_tok) as usize);
    for token in x.chunks_exact(IN_DIM as usize).take(n_tok as usize) {
        for row in weights.chunks_exact(IN_DIM as usize).take(out_dim as usize) {
            out.push(
                row.iter()
                    .zip(token)
                    .map(|(weight, value)| *weight as f32 * value)
                    .sum(),
            );
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum OrderedProjectionError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for OrderedProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("ordered projection tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for OrderedProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
