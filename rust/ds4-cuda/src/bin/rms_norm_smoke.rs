use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3A_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn rms_norm_plain_kernel(
        n: u32,
        rows: u32,
        eps: f32,
        x: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let n = n as usize;
        let base = row as usize * n;

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            let value = x[base + i];
            sum += value * value;
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

        let scale = 1.0_f32 / (unsafe { PARTIAL[0] } / n as f32 + eps).sqrt();
        i = tid;
        while i < n {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale;
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn rms_norm_weight_kernel(
        n: u32,
        rows: u32,
        eps: f32,
        x: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let n = n as usize;
        let base = row as usize * n;

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            let value = x[base + i];
            sum += value * value;
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

        let scale = 1.0_f32 / (unsafe { PARTIAL[0] } / n as f32 + eps).sqrt();
        i = tid;
        while i < n {
            unsafe {
                *out.get_unchecked_mut(base + i) = x[base + i] * scale * weight[i];
            }
            i += nth;
        }
    }
}

const THREADS_PER_BLOCK: u32 = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;

    // RMS normalization needs sqrt from libdevice; retain generated typed launches.
    let raw_module =
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_rms_norm_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let x_values = [
        1.0_f32, -2.0, 0.5, 4.0, -1.5, 0.25, 3.0, -0.5, 1.25, 2.5, -3.5, 0.75, 1.5, -2.25,
    ];
    let weights = [0.5_f32, 1.0, 1.5, -0.5, 0.25, 2.0, -1.0];
    let x = substrate.upload(&x_values)?;
    let weight = substrate.upload(&weights)?;
    let mut plain = substrate.zeroed::<f32>(x_values.len())?;
    let mut weighted = substrate.zeroed::<f32>(x_values.len())?;
    rms_norm_plain_rows_tensor(&module, substrate.stream(), &mut plain, &x, 7, 2, 1.0e-5)?;
    rms_norm_weight_rows_tensor(
        &module,
        substrate.stream(),
        &mut weighted,
        &x,
        &weight,
        7,
        2,
        1.0e-5,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&plain)?,
        &expected_rms_norm(&x_values, None, 7, 1.0e-5),
        1.0e-5,
    );
    assert_close(
        &substrate.download(&weighted)?,
        &expected_rms_norm(&x_values, Some(&weights), 7, 1.0e-5),
        1.0e-5,
    );

    let mut single_row = substrate.zeroed::<f32>(7)?;
    rms_norm_plain_rows_tensor(
        &module,
        substrate.stream(),
        &mut single_row,
        &x,
        7,
        1,
        1.0e-5,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&single_row)?,
        &expected_rms_norm(&x_values[..7], None, 7, 1.0e-5),
        1.0e-5,
    );

    let mut too_short = substrate.zeroed::<f32>(13)?;
    assert!(matches!(
        rms_norm_plain_rows_tensor(
            &module,
            substrate.stream(),
            &mut too_short,
            &x,
            7,
            2,
            1.0e-5
        ),
        Err(RmsNormError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"plain_rows_output_matches\":true,\"weighted_rows_output_matches\":true,\"single_row_output_matches\":true,\"shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_rms_norm_plain_kernel\":{},\"owns_rms_norm_weight_kernel\":{},\"owns_plain_and_weighted_tensor_surface\":{},\"owns_fused_qkv_and_head_norm_kernels\":{},\"owns_dense_projection_or_q8_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3A_SCOPE.owns_rms_norm_plain_kernel,
        M14_3A_SCOPE.owns_rms_norm_weight_kernel,
        M14_3A_SCOPE.owns_plain_and_weighted_tensor_surface,
        M14_3A_SCOPE.owns_fused_qkv_and_head_norm_kernels,
        M14_3A_SCOPE.owns_dense_projection_or_q8_kernels,
        M14_3A_SCOPE.changes_default_route,
    );
    Ok(())
}

fn rms_norm_plain_rows_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    n: u32,
    rows: u32,
    eps: f32,
) -> Result<(), RmsNormError> {
    validate_rows(out, x, n, rows)?;
    module
        .rms_norm_plain_kernel(stream, launch_config(rows), n, rows, eps, x, out)
        .map_err(RmsNormError::Driver)
}

fn rms_norm_weight_rows_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    n: u32,
    rows: u32,
    eps: f32,
) -> Result<(), RmsNormError> {
    validate_rows(out, x, n, rows)?;
    if n as usize > weight.len() {
        return Err(RmsNormError::InvalidShape);
    }
    module
        .rms_norm_weight_kernel(stream, launch_config(rows), n, rows, eps, x, weight, out)
        .map_err(RmsNormError::Driver)
}

fn validate_rows(
    out: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    n: u32,
    rows: u32,
) -> Result<(), RmsNormError> {
    let count = u64::from(n) * u64::from(rows);
    if n == 0 || rows == 0 || count > out.len() as u64 || count > x.len() as u64 {
        return Err(RmsNormError::InvalidShape);
    }
    Ok(())
}

fn launch_config(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn expected_rms_norm(x: &[f32], weight: Option<&[f32]>, n: usize, eps: f32) -> Vec<f32> {
    x.chunks_exact(n)
        .flat_map(|row| {
            let scale = 1.0_f32
                / (row.iter().map(|value| value * value).sum::<f32>() / n as f32 + eps).sqrt();
            row.iter()
                .enumerate()
                .map(move |(i, value)| value * scale * weight.map_or(1.0_f32, |weight| weight[i]))
        })
        .collect()
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
enum RmsNormError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for RmsNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("RMS normalization tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RmsNormError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
