use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4A_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn rope_tail_kernel(
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        pos_stride: u32,
        n_ctx_orig: u32,
        inverse: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        mut x: DisjointSlice<f32>,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let pairs_per_head = n_rot / 2;
        let pairs = n_tok * n_head * pairs_per_head;
        if gid >= pairs {
            return;
        }
        let pair = gid % pairs_per_head;
        let row = gid / pairs_per_head;
        let head = row % n_head;
        let token = row / n_head;
        let n_nope = head_dim - n_rot;
        let rot_i = pair * 2;

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
            if corr1 > (n_rot - 1) as f32 {
                corr1 = (n_rot - 1) as f32;
            }
        }

        let theta_extrap =
            (pos0 + token * pos_stride) as f32 * freq_base.powf(-(rot_i as f32) / n_rot as f32);
        let theta_interp = freq_scale * theta_extrap;
        let mut theta = theta_interp;
        let mut mscale = attn_factor;
        if ext_factor != 0.0 {
            let denom = if corr1 - corr0 > 0.001 {
                corr1 - corr0
            } else {
                0.001
            };
            let mut y = (pair as f32 - corr0) / denom;
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

        let base = ((token * n_head + head) * head_dim + n_nope + rot_i) as usize;
        let x0 = unsafe { *x.as_mut_ptr().add(base) };
        let x1 = unsafe { *x.as_mut_ptr().add(base + 1) };
        unsafe {
            *x.get_unchecked_mut(base) = x0 * c - x1 * s;
            *x.get_unchecked_mut(base + 1) = x0 * s + x1 * c;
        }
    }

    #[kernel]
    pub fn fp8_kv_quantize_kernel(
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
        mut x: DisjointSlice<f32>,
    ) {
        static mut SCRATCH: SharedArray<f32, 64> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= n_tok {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let n_nope = (head_dim - n_rot) as usize;
        let base = row as usize * head_dim as usize;
        let mut off = 0_usize;
        while off < n_nope {
            let index = off + tid;
            let valid = index < n_nope;
            let value = if valid {
                unsafe { *x.as_mut_ptr().add(base + index) }
            } else {
                0.0
            };
            unsafe {
                SCRATCH[tid] = absolute(value);
            }
            thread::sync_threads();
            let mut stride = 32_usize;
            while stride > 0 {
                if tid < stride {
                    let other = unsafe { SCRATCH[tid + stride] };
                    if other > unsafe { SCRATCH[tid] } {
                        unsafe {
                            SCRATCH[tid] = other;
                        }
                    }
                }
                thread::sync_threads();
                stride >>= 1;
            }
            let amax = if unsafe { SCRATCH[0] } > 1.0e-4 {
                unsafe { SCRATCH[0] }
            } else {
                1.0e-4
            };
            let scale = 2.0_f32.powf((amax / 448.0).log2().ceil());
            if valid {
                let mut scaled = value / scale;
                if scaled > 448.0 {
                    scaled = 448.0;
                } else if scaled < -448.0 {
                    scaled = -448.0;
                }
                unsafe {
                    *x.get_unchecked_mut(base + index) = e4m3fn_dequant(scaled) * scale;
                }
            }
            thread::sync_threads();
            off += 64;
        }
    }

    fn absolute(value: f32) -> f32 {
        if value < 0.0 {
            -value
        } else {
            value
        }
    }

    fn e4m3fn_value(value: i32) -> f32 {
        let exponent = (value >> 3) & 15;
        let mantissa = value & 7;
        if exponent == 0 {
            mantissa as f32 * 0.001953125
        } else {
            (1.0 + mantissa as f32 * 0.125) * 2.0_f32.powf(exponent as f32 - 7.0)
        }
    }

    fn e4m3fn_dequant(value: f32) -> f32 {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let mut magnitude = absolute(value);
        if magnitude > 448.0 {
            magnitude = 448.0;
        }
        let mut lo = 0_i32;
        let mut hi = 126_i32;
        while lo < hi {
            let mid = (lo + hi + 1) >> 1;
            if e4m3fn_value(mid) <= magnitude {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut best = lo;
        if best < 126 {
            let best_diff = absolute(magnitude - e4m3fn_value(best));
            let next_diff = absolute(magnitude - e4m3fn_value(best + 1));
            if next_diff < best_diff
                || (next_diff == best_diff && (best + 1) & 1 == 0 && best & 1 != 0)
            {
                best += 1;
            }
        }
        sign * e4m3fn_value(best)
    }
}

const THREADS: u32 = 256;
const ROPE_N_TOK: u32 = 2;
const ROPE_N_HEAD: u32 = 2;
const ROPE_HEAD_DIM: u32 = 10;
const ROPE_N_ROT: u32 = 6;
const POS0: u32 = 11;
const POS_STRIDE: u32 = 3;
const N_CTX_ORIG: u32 = 4096;
const FREQ_BASE: f32 = 100.0;
const FREQ_SCALE: f32 = 0.5;
const ATTN_FACTOR: f32 = 1.15;
const BETA_FAST: f32 = 32.0;
const BETA_SLOW: f32 = 1.0;
const KV_N_TOK: u32 = 2;
const KV_HEAD_DIM: u32 = 75;
const KV_N_ROT: u32 = 6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_rope_kv_quantization_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let rope_values = rope_values();

    let mut strided = substrate.upload(&rope_values)?;
    rope_tail_tensor(&module, substrate.stream(), &mut strided, false, 0.0, 1.0)?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&strided)?,
        &expected_rope_tail(&rope_values, false, 0.0, 1.0),
        2.0e-5,
    );

    let mut yarn_inverse = substrate.upload(&rope_values)?;
    rope_tail_tensor(
        &module,
        substrate.stream(),
        &mut yarn_inverse,
        true,
        1.0,
        ATTN_FACTOR,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&yarn_inverse)?,
        &expected_rope_tail(&rope_values, true, 1.0, ATTN_FACTOR),
        3.0e-5,
    );

    let kv_values = kv_values();
    let expected_kv = expected_fp8_kv_quantize(&kv_values);
    assert!(expected_kv
        .iter()
        .zip(&kv_values)
        .enumerate()
        .any(|(index, (expected, input))| index % (KV_HEAD_DIM as usize)
            < (KV_HEAD_DIM - KV_N_ROT) as usize
            && expected != input));
    let mut quantized = substrate.upload(&kv_values)?;
    fp8_kv_quantize_tensor(&module, substrate.stream(), &mut quantized)?;
    substrate.end_commands()?;
    let actual_kv = substrate.download(&quantized)?;
    assert_close(&actual_kv, &expected_kv, 1.0e-5);
    for row in 0..KV_N_TOK as usize {
        let tail = row * KV_HEAD_DIM as usize + (KV_HEAD_DIM - KV_N_ROT) as usize;
        assert_eq!(
            &actual_kv[tail..tail + KV_N_ROT as usize],
            &kv_values[tail..tail + KV_N_ROT as usize]
        );
    }

    let rope_count = (ROPE_N_TOK * ROPE_N_HEAD * ROPE_HEAD_DIM) as usize;
    let mut undersized_rope = substrate.zeroed::<f32>(rope_count - 1)?;
    assert!(matches!(
        rope_tail_tensor(
            &module,
            substrate.stream(),
            &mut undersized_rope,
            false,
            0.0,
            1.0
        ),
        Err(RopeKvError::InvalidShape)
    ));
    let mut undersized = substrate.zeroed::<f32>(KV_HEAD_DIM as usize)?;
    assert!(matches!(
        fp8_kv_quantize_tensor(&module, substrate.stream(), &mut undersized),
        Err(RopeKvError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"pos_stride_rope_output_matches\":true,\"yarn_inverse_output_matches\":true,\"fp8_prefix_output_matches\":true,\"fp8_partial_chunk_matches\":true,\"fp8_rope_tail_preserved\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_standalone_rope_tail_kernel\":{},\"owns_fp8_kv_quantize_kernel\":{},\"owns_kv_storage_or_compressor_kernels\":{},\"owns_attention_kernels\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4A_SCOPE.owns_standalone_rope_tail_kernel,
        M14_4A_SCOPE.owns_fp8_kv_quantize_kernel,
        M14_4A_SCOPE.owns_kv_storage_or_compressor_kernels,
        M14_4A_SCOPE.owns_attention_kernels,
        M14_4A_SCOPE.owns_runtime_graph_integration,
        M14_4A_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rope_tail_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
    inverse: bool,
    ext_factor: f32,
    attn_factor: f32,
) -> Result<(), RopeKvError> {
    let count = u64::from(ROPE_N_TOK) * u64::from(ROPE_N_HEAD) * u64::from(ROPE_HEAD_DIM);
    if ROPE_N_ROT == 0
        || ROPE_N_ROT > ROPE_HEAD_DIM
        || ROPE_N_ROT & 1 != 0
        || count > x.len() as u64
    {
        return Err(RopeKvError::InvalidShape);
    }
    let pairs = ROPE_N_TOK * ROPE_N_HEAD * (ROPE_N_ROT / 2);
    module
        .rope_tail_kernel(
            stream,
            LaunchConfig {
                grid_dim: (pairs.div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            ROPE_N_TOK,
            ROPE_N_HEAD,
            ROPE_HEAD_DIM,
            ROPE_N_ROT,
            POS0,
            POS_STRIDE,
            N_CTX_ORIG,
            u32::from(inverse),
            FREQ_BASE,
            FREQ_SCALE,
            ext_factor,
            attn_factor,
            BETA_FAST,
            BETA_SLOW,
            x,
        )
        .map_err(RopeKvError::Driver)
}

fn fp8_kv_quantize_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
) -> Result<(), RopeKvError> {
    let count = u64::from(KV_N_TOK) * u64::from(KV_HEAD_DIM);
    if KV_N_ROT > KV_HEAD_DIM || count > x.len() as u64 {
        return Err(RopeKvError::InvalidShape);
    }
    module
        .fp8_kv_quantize_kernel(
            stream,
            LaunchConfig {
                grid_dim: (KV_N_TOK, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            KV_N_TOK,
            KV_HEAD_DIM,
            KV_N_ROT,
            x,
        )
        .map_err(RopeKvError::Driver)
}

fn rope_values() -> Vec<f32> {
    (0..ROPE_N_TOK * ROPE_N_HEAD * ROPE_HEAD_DIM)
        .map(|index| ((index * 17 + 5) % 43) as f32 * 0.125 - 2.25)
        .collect()
}

fn expected_rope_tail(
    values: &[f32],
    inverse: bool,
    ext_factor: f32,
    attn_factor: f32,
) -> Vec<f32> {
    let mut result = values.to_vec();
    let n_nope = (ROPE_HEAD_DIM - ROPE_N_ROT) as usize;
    let (mut corr0, mut corr1) = (0.0_f32, 0.0_f32);
    if ext_factor != 0.0 {
        let denom = 2.0 * FREQ_BASE.ln();
        corr0 = (ROPE_N_ROT as f32
            * (N_CTX_ORIG as f32 / (BETA_FAST * 2.0 * std::f32::consts::PI)).ln()
            / denom)
            .floor()
            .max(0.0);
        corr1 = (ROPE_N_ROT as f32
            * (N_CTX_ORIG as f32 / (BETA_SLOW * 2.0 * std::f32::consts::PI)).ln()
            / denom)
            .ceil()
            .min((ROPE_N_ROT - 1) as f32);
    }
    for token in 0..ROPE_N_TOK as usize {
        for head in 0..ROPE_N_HEAD as usize {
            let base = (token * ROPE_N_HEAD as usize + head) * ROPE_HEAD_DIM as usize + n_nope;
            for pair in 0..ROPE_N_ROT as usize / 2 {
                let rot_i = pair * 2;
                let theta_extrap = (POS0 + token as u32 * POS_STRIDE) as f32
                    * FREQ_BASE.powf(-(rot_i as f32) / ROPE_N_ROT as f32);
                let theta_interp = FREQ_SCALE * theta_extrap;
                let mut theta = theta_interp;
                let mut scale = attn_factor;
                if ext_factor != 0.0 {
                    let ramp_mix = (1.0
                        - ((pair as f32 - corr0) / (corr1 - corr0).max(0.001)).clamp(0.0, 1.0))
                        * ext_factor;
                    theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                    scale *= 1.0 + 0.1 * (1.0 / FREQ_SCALE).ln();
                }
                let c = theta.cos() * scale;
                let s = if inverse { -theta.sin() } else { theta.sin() } * scale;
                let x0 = result[base + rot_i];
                let x1 = result[base + rot_i + 1];
                result[base + rot_i] = x0 * c - x1 * s;
                result[base + rot_i + 1] = x0 * s + x1 * c;
            }
        }
    }
    result
}

fn kv_values() -> Vec<f32> {
    (0..KV_N_TOK * KV_HEAD_DIM)
        .map(|index| ((index * 29 + 11) % 151) as f32 * 0.09375 - 6.75)
        .collect()
}

fn expected_fp8_kv_quantize(values: &[f32]) -> Vec<f32> {
    let mut result = values.to_vec();
    let prefix = (KV_HEAD_DIM - KV_N_ROT) as usize;
    for row in result.chunks_exact_mut(KV_HEAD_DIM as usize) {
        for chunk in row[..prefix].chunks_mut(64) {
            let amax = chunk
                .iter()
                .map(|value| value.abs())
                .fold(1.0e-4_f32, f32::max);
            let scale = 2.0_f32.powf((amax / 448.0).log2().ceil());
            for value in chunk {
                *value = e4m3fn_dequant_host((*value / scale).clamp(-448.0, 448.0)) * scale;
            }
        }
    }
    result
}

fn e4m3fn_value_host(value: i32) -> f32 {
    let exponent = (value >> 3) & 15;
    let mantissa = value & 7;
    if exponent == 0 {
        mantissa as f32 * 0.001953125
    } else {
        (1.0 + mantissa as f32 * 0.125) * 2.0_f32.powf(exponent as f32 - 7.0)
    }
}

fn e4m3fn_dequant_host(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let magnitude = value.abs().min(448.0);
    let mut lo = 0_i32;
    let mut hi = 126_i32;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value_host(mid) <= magnitude {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let mut best = lo;
    if best < 126 {
        let best_diff = (magnitude - e4m3fn_value_host(best)).abs();
        let next_diff = (magnitude - e4m3fn_value_host(best + 1)).abs();
        if next_diff < best_diff || (next_diff == best_diff && (best + 1) & 1 == 0 && best & 1 != 0)
        {
            best += 1;
        }
    }
    sign * e4m3fn_value_host(best)
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
enum RopeKvError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for RopeKvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("RoPE/KV quantization tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RopeKvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
