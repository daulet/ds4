use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3B2_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn head_rms_norm_rope_tail_kernel(
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        n_ctx_orig: u32,
        inverse: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
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
        let n_rot = n_rot as usize;
        let n_nope = head_dim - n_rot;
        let base = row as usize * head_dim;
        let t = row / n_head;
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
        while i < n_nope {
            unsafe {
                *x.get_unchecked_mut(base + i) *= scale;
            }
            i += nth;
        }

        let mut corr0 = 0.0_f32;
        let mut corr1 = 0.0_f32;
        if ext_factor != 0.0 {
            let denom = 2.0_f32 * freq_base.ln();
            corr0 = (n_rot as f32
                * (n_ctx_orig as f32 / (beta_fast * 2.0_f32 * 3.1415927_f32)).ln()
                / denom)
                .floor();
            corr1 = (n_rot as f32
                * (n_ctx_orig as f32 / (beta_slow * 2.0_f32 * 3.1415927_f32)).ln()
                / denom)
                .ceil();
            if corr0 < 0.0 {
                corr0 = 0.0;
            }
            let corr1_max = (n_rot - 1) as f32;
            if corr1 > corr1_max {
                corr1 = corr1_max;
            }
        }

        let mut pair = tid;
        while pair < n_rot / 2 {
            let rot_i = pair * 2;
            let theta_extrap = (pos0 + t) as f32 * freq_base.powf(-(rot_i as f32) / n_rot as f32);
            let theta_interp = freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;
            if ext_factor != 0.0 {
                let ramp_denom = if corr1 - corr0 > 0.001 {
                    corr1 - corr0
                } else {
                    0.001
                };
                let mut y = (pair as f32 - corr0) / ramp_denom;
                if y < 0.0 {
                    y = 0.0;
                } else if y > 1.0 {
                    y = 1.0;
                }
                let ramp_mix = (1.0 - y) * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
            }
            let c = theta.cos() * mscale;
            let mut s = theta.sin() * mscale;
            if inverse != 0 {
                s = -s;
            }
            let x0 = unsafe { *x_ptr.add(base + n_nope + rot_i) } * scale;
            let x1 = unsafe { *x_ptr.add(base + n_nope + rot_i + 1) } * scale;
            unsafe {
                *x.get_unchecked_mut(base + n_nope + rot_i) = x0 * c - x1 * s;
                *x.get_unchecked_mut(base + n_nope + rot_i + 1) = x0 * s + x1 * c;
            }
            pair += nth;
        }
    }
}

const THREADS_PER_BLOCK: u32 = 256;
const N_TOK: u32 = 2;
const N_HEAD: u32 = 2;
const HEAD_DIM: u32 = 10;
const N_ROT: u32 = 6;
const POS0: u32 = 7;
const N_CTX_ORIG: u32 = 4096;
const FREQ_BASE: f32 = 100.0;
const FREQ_SCALE: f32 = 0.5;
const EXTRAP_ATTN_FACTOR: f32 = 1.15;
const BETA_FAST: f32 = 32.0;
const BETA_SLOW: f32 = 1.0;
const EPS: f32 = 1.0e-5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_head_rms_rope_tail_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;

    let values = [
        1.0_f32, -2.0, 0.5, 4.0, -1.5, 0.25, 3.0, -0.5, 1.25, 2.5, -3.5, 0.75, 1.5, -2.25, 0.25,
        2.0, 1.75, -0.25, 3.25, -1.0, 0.125, 0.5, -1.25, 2.75, -3.0, 1.25, 0.75, -0.625, 2.5, 1.0,
        -0.875, 2.25, 1.5, -2.0, 0.375, 1.875, -0.5, 2.625, -1.75, 0.875,
    ];

    let mut interpolated = substrate.upload(&values)?;
    head_rms_norm_rope_tail_tensor(
        &module,
        substrate.stream(),
        &mut interpolated,
        N_TOK,
        N_HEAD,
        HEAD_DIM,
        N_ROT,
        POS0,
        N_CTX_ORIG,
        false,
        FREQ_BASE,
        FREQ_SCALE,
        0.0,
        1.0,
        BETA_FAST,
        BETA_SLOW,
        EPS,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&interpolated)?,
        &expected_head_rms_norm_rope_tail(&values, false, 0.0, 1.0),
        2.0e-5,
    );

    let mut yarn_forward = substrate.upload(&values)?;
    head_rms_norm_rope_tail_tensor(
        &module,
        substrate.stream(),
        &mut yarn_forward,
        N_TOK,
        N_HEAD,
        HEAD_DIM,
        N_ROT,
        POS0,
        N_CTX_ORIG,
        false,
        FREQ_BASE,
        FREQ_SCALE,
        1.0,
        EXTRAP_ATTN_FACTOR,
        BETA_FAST,
        BETA_SLOW,
        EPS,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&yarn_forward)?,
        &expected_head_rms_norm_rope_tail(&values, false, 1.0, EXTRAP_ATTN_FACTOR),
        3.0e-5,
    );

    let mut yarn_inverse = substrate.upload(&values)?;
    head_rms_norm_rope_tail_tensor(
        &module,
        substrate.stream(),
        &mut yarn_inverse,
        N_TOK,
        N_HEAD,
        HEAD_DIM,
        N_ROT,
        POS0,
        N_CTX_ORIG,
        true,
        FREQ_BASE,
        FREQ_SCALE,
        1.0,
        EXTRAP_ATTN_FACTOR,
        BETA_FAST,
        BETA_SLOW,
        EPS,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&yarn_inverse)?,
        &expected_head_rms_norm_rope_tail(&values, true, 1.0, EXTRAP_ATTN_FACTOR),
        3.0e-5,
    );

    assert!(matches!(
        head_rms_norm_rope_tail_tensor(
            &module,
            substrate.stream(),
            &mut yarn_forward,
            N_TOK,
            N_HEAD,
            HEAD_DIM,
            5,
            POS0,
            N_CTX_ORIG,
            false,
            FREQ_BASE,
            FREQ_SCALE,
            1.0,
            EXTRAP_ATTN_FACTOR,
            BETA_FAST,
            BETA_SLOW,
            EPS,
        ),
        Err(HeadRmsRopeTailError::InvalidShape)
    ));
    assert!(matches!(
        head_rms_norm_rope_tail_tensor(
            &module,
            substrate.stream(),
            &mut yarn_forward,
            N_TOK,
            N_HEAD,
            HEAD_DIM,
            12,
            POS0,
            N_CTX_ORIG,
            false,
            FREQ_BASE,
            FREQ_SCALE,
            1.0,
            EXTRAP_ATTN_FACTOR,
            BETA_FAST,
            BETA_SLOW,
            EPS,
        ),
        Err(HeadRmsRopeTailError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3b2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"interpolated_rope_output_matches\":true,\"yarn_forward_output_matches\":true,\"yarn_inverse_output_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_head_rms_norm_rope_tail_kernel\":{},\"owns_yarn_rotary_math_path\":{},\"owns_standalone_rope_tail_kernel\":{},\"owns_qkv_fused_dispatch_policy\":{},\"owns_dense_projection_or_q8_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3B2_SCOPE.owns_head_rms_norm_rope_tail_kernel,
        M14_3B2_SCOPE.owns_yarn_rotary_math_path,
        M14_3B2_SCOPE.owns_standalone_rope_tail_kernel,
        M14_3B2_SCOPE.owns_qkv_fused_dispatch_policy,
        M14_3B2_SCOPE.owns_dense_projection_or_q8_kernels,
        M14_3B2_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn head_rms_norm_rope_tail_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos0: u32,
    n_ctx_orig: u32,
    inverse: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    eps: f32,
) -> Result<(), HeadRmsRopeTailError> {
    let count = u64::from(n_tok) * u64::from(n_head) * u64::from(head_dim);
    if n_tok == 0
        || n_head == 0
        || head_dim == 0
        || n_rot > head_dim
        || n_rot & 1 != 0
        || count > x.len() as u64
    {
        return Err(HeadRmsRopeTailError::InvalidShape);
    }
    module
        .head_rms_norm_rope_tail_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tok * n_head, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            n_tok,
            n_head,
            head_dim,
            n_rot,
            pos0,
            n_ctx_orig,
            u32::from(inverse),
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            eps,
            x,
        )
        .map_err(HeadRmsRopeTailError::Driver)
}

fn expected_head_rms_norm_rope_tail(
    x: &[f32],
    inverse: bool,
    ext_factor: f32,
    attn_factor: f32,
) -> Vec<f32> {
    let head_dim = HEAD_DIM as usize;
    let n_rot = N_ROT as usize;
    let n_nope = head_dim - n_rot;
    let mut result = x.to_vec();
    let (mut corr0, mut corr1) = (0.0_f32, 0.0_f32);
    if ext_factor != 0.0 {
        let denom = 2.0 * FREQ_BASE.ln();
        corr0 = (N_ROT as f32
            * (N_CTX_ORIG as f32 / (BETA_FAST * 2.0 * std::f32::consts::PI)).ln()
            / denom)
            .floor()
            .max(0.0);
        corr1 = (N_ROT as f32
            * (N_CTX_ORIG as f32 / (BETA_SLOW * 2.0 * std::f32::consts::PI)).ln()
            / denom)
            .ceil()
            .min((N_ROT - 1) as f32);
    }

    for (row, values) in result.chunks_exact_mut(head_dim).enumerate() {
        let scale = 1.0
            / (x[row * head_dim..(row + 1) * head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / HEAD_DIM as f32
                + EPS)
                .sqrt();
        for value in &mut values[..n_nope] {
            *value *= scale;
        }
        let t = row as u32 / N_HEAD;
        for pair in 0..n_rot / 2 {
            let rot_i = pair * 2;
            let theta_extrap = (POS0 + t) as f32 * FREQ_BASE.powf(-(rot_i as f32) / N_ROT as f32);
            let theta_interp = FREQ_SCALE * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;
            if ext_factor != 0.0 {
                let ramp_mix = (1.0
                    - ((pair as f32 - corr0) / (corr1 - corr0).max(0.001)).clamp(0.0, 1.0))
                    * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / FREQ_SCALE).ln();
            }
            let c = theta.cos() * mscale;
            let s = if inverse { -theta.sin() } else { theta.sin() } * mscale;
            let x0 = values[n_nope + rot_i] * scale;
            let x1 = values[n_nope + rot_i + 1] * scale;
            values[n_nope + rot_i] = x0 * c - x1 * s;
            values[n_nope + rot_i + 1] = x0 * s + x1 * c;
        }
    }
    result
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
enum HeadRmsRopeTailError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for HeadRmsRopeTailError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("head RMS RoPE-tail tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for HeadRmsRopeTailError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
