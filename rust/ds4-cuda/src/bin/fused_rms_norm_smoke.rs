use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3B1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn dsv4_qkv_rms_norm_rows_kernel(
        q_n: u32,
        kv_n: u32,
        rows: u32,
        eps: f32,
        q: &[f32],
        q_weight: &[f32],
        mut q_out: DisjointSlice<f32>,
        kv: &[f32],
        kv_weight: &[f32],
        mut kv_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let which = thread::blockIdx_y();
        if row >= rows || which > 1 {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let n = (if which == 0 { q_n } else { kv_n }) as usize;
        let base = row as usize * n;

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
            let value = if which == 0 {
                q[base + i]
            } else {
                kv[base + i]
            };
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
                if which == 0 {
                    *q_out.get_unchecked_mut(base + i) = q[base + i] * scale * q_weight[i];
                } else {
                    *kv_out.get_unchecked_mut(base + i) = kv[base + i] * scale * kv_weight[i];
                }
            }
            i += nth;
        }
    }

    #[kernel]
    pub fn head_rms_norm_kernel(
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
        mut x: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= n_tok * n_head {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let head_dim = head_dim as usize;
        let base = row as usize * head_dim;
        let x_ptr = x.as_mut_ptr();

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < head_dim {
            let value = unsafe { *x_ptr.add(base + i) };
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

        let scale = 1.0_f32 / (unsafe { PARTIAL[0] } / head_dim as f32 + eps).sqrt();
        i = tid;
        while i < head_dim {
            unsafe {
                *x.get_unchecked_mut(base + i) *= scale;
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
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_fused_rms_norm_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let q_values = [1.0_f32, -2.0, 0.5, 4.0, -1.5, 0.25, 3.0, -0.5, 1.25, 2.5];
    let q_weights = [0.5_f32, 1.0, 1.5, -0.5, 0.25];
    let kv_values = [2.0_f32, -0.5, 1.5, -1.0, 3.0, 0.25];
    let kv_weights = [-1.0_f32, 0.75, 2.0];
    let q = substrate.upload(&q_values)?;
    let q_weight = substrate.upload(&q_weights)?;
    let kv = substrate.upload(&kv_values)?;
    let kv_weight = substrate.upload(&kv_weights)?;
    let mut q_out = substrate.zeroed::<f32>(q_values.len())?;
    let mut kv_out = substrate.zeroed::<f32>(kv_values.len())?;
    dsv4_qkv_rms_norm_rows_tensor(
        &module,
        substrate.stream(),
        &mut q_out,
        &q,
        &q_weight,
        5,
        &mut kv_out,
        &kv,
        &kv_weight,
        3,
        2,
        1.0e-5,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&q_out)?,
        &expected_weighted_rms_norm(&q_values, &q_weights, 5, 1.0e-5),
        1.0e-5,
    );
    assert_close(
        &substrate.download(&kv_out)?,
        &expected_weighted_rms_norm(&kv_values, &kv_weights, 3, 1.0e-5),
        1.0e-5,
    );

    let head_values = [
        1.0_f32, -2.0, 0.5, 4.0, -1.5, 0.25, 3.0, -0.5, 1.25, 2.5, -3.5, 0.75, 1.5, -2.25, 0.25,
        2.0,
    ];
    let mut heads = substrate.upload(&head_values)?;
    head_rms_norm_tensor(&module, substrate.stream(), &mut heads, 2, 2, 4, 1.0e-5)?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&heads)?,
        &expected_plain_rms_norm(&head_values, 4, 1.0e-5),
        1.0e-5,
    );

    let mut short_q_out = substrate.zeroed::<f32>(q_values.len() - 1)?;
    assert!(matches!(
        dsv4_qkv_rms_norm_rows_tensor(
            &module,
            substrate.stream(),
            &mut short_q_out,
            &q,
            &q_weight,
            5,
            &mut kv_out,
            &kv,
            &kv_weight,
            3,
            2,
            1.0e-5,
        ),
        Err(FusedRmsNormError::InvalidShape)
    ));
    assert!(matches!(
        head_rms_norm_tensor(&module, substrate.stream(), &mut heads, 2, 2, 5, 1.0e-5),
        Err(FusedRmsNormError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3b1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"qkv_fused_output_matches\":true,\"asymmetric_q_kv_widths_match\":true,\"head_in_place_output_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_dsv4_qkv_rms_norm_rows_kernel\":{},\"owns_head_rms_norm_kernel\":{},\"owns_head_rms_norm_rope_tail_kernel\":{},\"owns_qkv_fused_dispatch_policy\":{},\"owns_dense_projection_or_q8_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3B1_SCOPE.owns_dsv4_qkv_rms_norm_rows_kernel,
        M14_3B1_SCOPE.owns_head_rms_norm_kernel,
        M14_3B1_SCOPE.owns_head_rms_norm_rope_tail_kernel,
        M14_3B1_SCOPE.owns_qkv_fused_dispatch_policy,
        M14_3B1_SCOPE.owns_dense_projection_or_q8_kernels,
        M14_3B1_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dsv4_qkv_rms_norm_rows_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q_out: &mut DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    q_weight: &DeviceBuffer<f32>,
    q_n: u32,
    kv_out: &mut DeviceBuffer<f32>,
    kv: &DeviceBuffer<f32>,
    kv_weight: &DeviceBuffer<f32>,
    kv_n: u32,
    rows: u32,
    eps: f32,
) -> Result<(), FusedRmsNormError> {
    validate_weighted_rows(q_out, q, q_weight, q_n, rows)?;
    validate_weighted_rows(kv_out, kv, kv_weight, kv_n, rows)?;
    module
        .dsv4_qkv_rms_norm_rows_kernel(
            stream,
            LaunchConfig {
                grid_dim: (rows, 2, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            q_n,
            kv_n,
            rows,
            eps,
            q,
            q_weight,
            q_out,
            kv,
            kv_weight,
            kv_out,
        )
        .map_err(FusedRmsNormError::Driver)
}

fn head_rms_norm_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), FusedRmsNormError> {
    let count = u64::from(n_tok) * u64::from(n_head) * u64::from(head_dim);
    if n_tok == 0 || n_head == 0 || head_dim == 0 || count > x.len() as u64 {
        return Err(FusedRmsNormError::InvalidShape);
    }
    module
        .head_rms_norm_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tok * n_head, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            n_tok,
            n_head,
            head_dim,
            eps,
            x,
        )
        .map_err(FusedRmsNormError::Driver)
}

fn validate_weighted_rows(
    out: &DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    n: u32,
    rows: u32,
) -> Result<(), FusedRmsNormError> {
    let count = u64::from(n) * u64::from(rows);
    if n == 0
        || rows == 0
        || n as usize > weight.len()
        || count > out.len() as u64
        || count > x.len() as u64
    {
        return Err(FusedRmsNormError::InvalidShape);
    }
    Ok(())
}

fn expected_weighted_rms_norm(x: &[f32], weight: &[f32], n: usize, eps: f32) -> Vec<f32> {
    x.chunks_exact(n)
        .flat_map(|row| {
            let scale = 1.0_f32
                / (row.iter().map(|value| value * value).sum::<f32>() / n as f32 + eps).sqrt();
            row.iter()
                .enumerate()
                .map(move |(i, value)| value * scale * weight[i])
        })
        .collect()
}

fn expected_plain_rms_norm(x: &[f32], n: usize, eps: f32) -> Vec<f32> {
    x.chunks_exact(n)
        .flat_map(|row| {
            let scale = 1.0_f32
                / (row.iter().map(|value| value * value).sum::<f32>() / n as f32 + eps).sqrt();
            row.iter().map(move |value| value * scale)
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
enum FusedRmsNormError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for FusedRmsNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("fused RMS normalization tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FusedRmsNormError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
