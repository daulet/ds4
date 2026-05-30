#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4C3B_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fill_f32_kernel(n: u32, value: f32, mut x: DisjointSlice<f32>) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if index < n {
            unsafe {
                *x.get_unchecked_mut(index as usize) = value;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn compressor_set_rows_kernel(
        width: u32,
        ratio: u32,
        pos0: u32,
        src0: u32,
        dst0: u32,
        rows: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
    ) {
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let count = rows * width;
        if index >= count {
            return;
        }
        let row = index / width;
        let dimension = index % width;
        let src = src0 + row;
        let dst = dst0 + row;
        let phase = (pos0 + src) % ratio;
        let ape_index = (phase * width + dimension) as usize;
        let ape = model_scalar(ape_type, ape_index, ape_f32, ape_f16);
        unsafe {
            *state_kv.get_unchecked_mut((dst * width + dimension) as usize) =
                kv[(src * width + dimension) as usize];
            *state_score.get_unchecked_mut((dst * width + dimension) as usize) =
                sc[(src * width + dimension) as usize] + ape;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn compressor_prefill_pool_kernel(
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_comp: u32,
        replay: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        state_kv: &[f32],
        state_score: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut comp: DisjointSlice<f32>,
    ) {
        let dimension = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let compressed = thread::blockIdx_y();
        if dimension >= head_dim || compressed >= n_comp {
            return;
        }
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let mut max_score = f32::NEG_INFINITY;
        if ratio == 4 {
            if replay != 0 && compressed == 0 {
                let mut row = 0_u32;
                while row < 4 {
                    max_score = maximum(max_score, state_score[(row * width + dimension) as usize]);
                    row += 1;
                }
            } else if compressed > 0 {
                let base = (compressed - 1) * ratio;
                let mut row = 0_u32;
                while row < 4 {
                    let token = base + row;
                    let phase = (pos0 + token) % ratio;
                    max_score = maximum(
                        max_score,
                        sc[(token * width + dimension) as usize]
                            + model_scalar(
                                ape_type,
                                (phase * width + dimension) as usize,
                                ape_f32,
                                ape_f16,
                            ),
                    );
                    row += 1;
                }
            }
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < 4 {
                let token = base + row;
                let phase = (pos0 + token) % ratio;
                max_score = maximum(
                    max_score,
                    sc[(token * width + head_dim + dimension) as usize]
                        + model_scalar(
                            ape_type,
                            (phase * width + head_dim + dimension) as usize,
                            ape_f32,
                            ape_f16,
                        ),
                );
                row += 1;
            }
        } else {
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < ratio {
                let token = base + row;
                let phase = (pos0 + token) % ratio;
                max_score = maximum(
                    max_score,
                    sc[(token * width + dimension) as usize]
                        + model_scalar(
                            ape_type,
                            (phase * width + dimension) as usize,
                            ape_f32,
                            ape_f16,
                        ),
                );
                row += 1;
            }
        }

        let mut denominator = 0.0_f32;
        let mut accumulator = 0.0_f32;
        if ratio == 4 {
            if replay != 0 && compressed == 0 {
                let mut row = 0_u32;
                while row < 4 {
                    add_candidate(
                        state_kv[(row * width + dimension) as usize],
                        state_score[(row * width + dimension) as usize],
                        max_score,
                        &mut denominator,
                        &mut accumulator,
                    );
                    row += 1;
                }
            } else if compressed > 0 {
                let base = (compressed - 1) * ratio;
                let mut row = 0_u32;
                while row < 4 {
                    let token = base + row;
                    let phase = (pos0 + token) % ratio;
                    add_candidate(
                        kv[(token * width + dimension) as usize],
                        sc[(token * width + dimension) as usize]
                            + model_scalar(
                                ape_type,
                                (phase * width + dimension) as usize,
                                ape_f32,
                                ape_f16,
                            ),
                        max_score,
                        &mut denominator,
                        &mut accumulator,
                    );
                    row += 1;
                }
            }
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < 4 {
                let token = base + row;
                let phase = (pos0 + token) % ratio;
                add_candidate(
                    kv[(token * width + head_dim + dimension) as usize],
                    sc[(token * width + head_dim + dimension) as usize]
                        + model_scalar(
                            ape_type,
                            (phase * width + head_dim + dimension) as usize,
                            ape_f32,
                            ape_f16,
                        ),
                    max_score,
                    &mut denominator,
                    &mut accumulator,
                );
                row += 1;
            }
        } else {
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < ratio {
                let token = base + row;
                let phase = (pos0 + token) % ratio;
                add_candidate(
                    kv[(token * width + dimension) as usize],
                    sc[(token * width + dimension) as usize]
                        + model_scalar(
                            ape_type,
                            (phase * width + dimension) as usize,
                            ape_f32,
                            ape_f16,
                        ),
                    max_score,
                    &mut denominator,
                    &mut accumulator,
                );
                row += 1;
            }
        }
        unsafe {
            *comp.get_unchecked_mut((compressed * head_dim + dimension) as usize) =
                accumulator / denominator;
        }
    }

    #[kernel]
    pub fn rms_norm_weight_kernel(
        n: u32,
        rows: u32,
        eps: f32,
        weight: &[f32],
        mut x: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let base = row as usize * n as usize;
        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n as usize {
            let value = unsafe { *x.as_mut_ptr().add(base + i) };
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
        while i < n as usize {
            unsafe {
                let value = *x.as_mut_ptr().add(base + i);
                *x.get_unchecked_mut(base + i) = value * scale * weight[i];
            }
            i += nth;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn rope_tail_kernel(
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        pos_stride: u32,
        n_ctx_orig: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        mut x: DisjointSlice<f32>,
    ) {
        let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let pairs_per_row = n_rot / 2;
        let pairs = n_tok * pairs_per_row;
        if gid >= pairs {
            return;
        }
        let pair = gid % pairs_per_row;
        let token = gid / pairs_per_row;
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
        let mut scale = attn_factor;
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
            scale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
        }
        let c = theta.cos() * scale;
        let s = theta.sin() * scale;
        let base = (token * head_dim + n_nope + rot_i) as usize;
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
        let prefix = (head_dim - n_rot) as usize;
        let base = row as usize * head_dim as usize;
        let mut off = 0_usize;
        while off < prefix {
            let index = off + tid;
            let valid = index < prefix;
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
                if tid < stride && unsafe { SCRATCH[tid + stride] } > unsafe { SCRATCH[tid] } {
                    unsafe {
                        SCRATCH[tid] = SCRATCH[tid + stride];
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
                unsafe {
                    *x.get_unchecked_mut(base + index) = e4m3fn_dequant(value / scale) * scale;
                }
            }
            thread::sync_threads();
            off += 64;
        }
    }

    fn model_scalar(ape_type: u32, index: usize, ape_f32: &[f32], ape_f16: &[f16]) -> f32 {
        if ape_type == 1 {
            ape_f16[index] as f32
        } else {
            ape_f32[index]
        }
    }

    fn maximum(left: f32, right: f32) -> f32 {
        if right > left {
            right
        } else {
            left
        }
    }

    fn add_candidate(
        value: f32,
        score: f32,
        max_score: f32,
        denominator: &mut f32,
        accumulator: &mut f32,
    ) {
        let weight = (score - max_score).exp();
        *denominator += weight;
        *accumulator += value * weight;
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
const HEAD_DIM: u32 = 6;
const N_ROT: u32 = 4;
const RATIO3: u32 = 3;
const RATIO4: u32 = 4;
const N_CTX_ORIG: u32 = 4096;
const FREQ_BASE: f32 = 100.0;
const FREQ_SCALE: f32 = 0.5;
const EXT_FACTOR: f32 = 1.0;
const ATTN_FACTOR: f32 = 1.15;
const BETA_FAST: f32 = 32.0;
const BETA_SLOW: f32 = 1.0;
const RMS_EPS: f32 = 1.0e-5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_compressor_prefill_orchestration_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let weights_values = values(HEAD_DIM as usize, 17, 0.5);
    let weights = substrate.upload(&weights_values)?;

    let general_tokens = 7;
    let general_pos0 = 2;
    let general_width = width(RATIO3);
    let general_kv_values = values((general_tokens * general_width) as usize, 19, -1.5);
    let general_sc_values = values((general_tokens * general_width) as usize, 23, -0.875);
    let general_ape_values = values((RATIO3 * general_width) as usize, 29, -0.25);
    let general_kv = substrate.upload(&general_kv_values)?;
    let general_sc = substrate.upload(&general_sc_values)?;
    let general_ape = substrate.upload(&general_ape_values)?;
    let unused_f16 =
        substrate.upload(&vec![f16::from_bits(0); (RATIO3 * general_width) as usize])?;
    let mut general_state =
        substrate.upload(&vec![-77.0_f32; (RATIO3 * general_width) as usize])?;
    let mut general_score =
        substrate.upload(&vec![-77.0_f32; (RATIO3 * general_width) as usize])?;
    let mut general_comp =
        substrate.zeroed::<f32>(((general_tokens / RATIO3) * HEAD_DIM) as usize)?;
    compressor_prefill_tensor(
        &module,
        substrate.stream(),
        RATIO3,
        general_pos0,
        general_tokens,
        0,
        true,
        &general_kv,
        &general_sc,
        &general_ape,
        &unused_f16,
        &weights,
        &mut general_state,
        &mut general_score,
        &mut general_comp,
    )?;
    substrate.end_commands()?;
    let expected = expected_prefill(
        RATIO3,
        general_pos0,
        general_tokens,
        false,
        true,
        &general_kv_values,
        &general_sc_values,
        &general_ape_values,
        &weights_values,
        &[],
        &[],
    );
    let unquantized_general = expected_prefill(
        RATIO3,
        general_pos0,
        general_tokens,
        false,
        false,
        &general_kv_values,
        &general_sc_values,
        &general_ape_values,
        &weights_values,
        &[],
        &[],
    );
    assert_ne!(expected.2, unquantized_general.2);
    assert_close(&substrate.download(&general_state)?, &expected.0, 1.0e-6);
    assert_close(&substrate.download(&general_score)?, &expected.1, 1.0e-6);
    assert_close(&substrate.download(&general_comp)?, &expected.2, 4.0e-5);

    let ratio4_tokens = 10;
    let ratio4_pos0 = 1;
    let ratio4_width = width(RATIO4);
    let ratio4_kv_values = values((ratio4_tokens * ratio4_width) as usize, 31, -1.75);
    let ratio4_sc_values = values((ratio4_tokens * ratio4_width) as usize, 37, -1.0);
    let ratio4_ape_f16 = values((RATIO4 * ratio4_width) as usize, 41, -0.1875)
        .into_iter()
        .map(|value| value as f16)
        .collect::<Vec<_>>();
    let ratio4_ape_values = ratio4_ape_f16
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let ratio4_kv = substrate.upload(&ratio4_kv_values)?;
    let ratio4_sc = substrate.upload(&ratio4_sc_values)?;
    let ratio4_ape = substrate.upload(&ratio4_ape_f16)?;
    let unused_f32 = substrate.upload(&vec![0.0_f32; (RATIO4 * ratio4_width) as usize])?;
    let mut ratio4_state =
        substrate.upload(&vec![-77.0_f32; (2 * RATIO4 * ratio4_width) as usize])?;
    let mut ratio4_score =
        substrate.upload(&vec![-77.0_f32; (2 * RATIO4 * ratio4_width) as usize])?;
    let mut ratio4_comp =
        substrate.zeroed::<f32>(((ratio4_tokens / RATIO4) * HEAD_DIM) as usize)?;
    compressor_prefill_tensor(
        &module,
        substrate.stream(),
        RATIO4,
        ratio4_pos0,
        ratio4_tokens,
        1,
        false,
        &ratio4_kv,
        &ratio4_sc,
        &unused_f32,
        &ratio4_ape,
        &weights,
        &mut ratio4_state,
        &mut ratio4_score,
        &mut ratio4_comp,
    )?;
    substrate.end_commands()?;
    let expected = expected_prefill(
        RATIO4,
        ratio4_pos0,
        ratio4_tokens,
        false,
        false,
        &ratio4_kv_values,
        &ratio4_sc_values,
        &ratio4_ape_values,
        &weights_values,
        &[],
        &[],
    );
    assert_close(&substrate.download(&ratio4_state)?, &expected.0, 1.0e-6);
    assert_close(&substrate.download(&ratio4_score)?, &expected.1, 1.0e-6);
    assert_close(&substrate.download(&ratio4_comp)?, &expected.2, 4.0e-5);

    let replay_tokens = 8;
    let replay_pos0 = 4;
    let replay_kv_values = values((replay_tokens * ratio4_width) as usize, 43, -1.25);
    let replay_sc_values = values((replay_tokens * ratio4_width) as usize, 47, -0.75);
    let replay_state_initial = values((2 * RATIO4 * ratio4_width) as usize, 53, -2.0);
    let replay_score_initial = values((2 * RATIO4 * ratio4_width) as usize, 59, -1.125);
    let replay_kv = substrate.upload(&replay_kv_values)?;
    let replay_sc = substrate.upload(&replay_sc_values)?;
    let mut replay_state = substrate.upload(&replay_state_initial)?;
    let mut replay_score = substrate.upload(&replay_score_initial)?;
    let mut replay_comp =
        substrate.zeroed::<f32>(((replay_tokens / RATIO4) * HEAD_DIM) as usize)?;
    compressor_prefill_ratio4_replay_tensor(
        &module,
        substrate.stream(),
        replay_pos0,
        replay_tokens,
        1,
        true,
        &replay_kv,
        &replay_sc,
        &unused_f32,
        &ratio4_ape,
        &weights,
        &mut replay_state,
        &mut replay_score,
        &mut replay_comp,
    )?;
    substrate.end_commands()?;
    let expected = expected_prefill(
        RATIO4,
        replay_pos0,
        replay_tokens,
        true,
        true,
        &replay_kv_values,
        &replay_sc_values,
        &ratio4_ape_values,
        &weights_values,
        &replay_state_initial,
        &replay_score_initial,
    );
    assert_close(&substrate.download(&replay_state)?, &expected.0, 1.0e-6);
    assert_close(&substrate.download(&replay_score)?, &expected.1, 1.0e-6);
    assert_close(&substrate.download(&replay_comp)?, &expected.2, 4.0e-5);

    let tail_kv_values = values((RATIO4 * ratio4_width) as usize, 61, -1.0);
    let tail_sc_values = values((RATIO4 * ratio4_width) as usize, 67, -0.5);
    let tail_kv = substrate.upload(&tail_kv_values)?;
    let tail_sc = substrate.upload(&tail_sc_values)?;
    let mut state_only =
        substrate.upload(&vec![-88.0_f32; (2 * RATIO4 * ratio4_width) as usize])?;
    let mut score_only =
        substrate.upload(&vec![-88.0_f32; (2 * RATIO4 * ratio4_width) as usize])?;
    compressor_prefill_state_ratio4_tensor(
        &module,
        substrate.stream(),
        replay_pos0,
        1,
        &tail_kv,
        &tail_sc,
        &unused_f32,
        &ratio4_ape,
        &mut state_only,
        &mut score_only,
    )?;
    substrate.end_commands()?;
    let expected_state = expected_state_ratio4(
        replay_pos0,
        &tail_kv_values,
        &tail_sc_values,
        &ratio4_ape_values,
    );
    assert_close(&substrate.download(&state_only)?, &expected_state.0, 1.0e-6);
    assert_close(&substrate.download(&score_only)?, &expected_state.1, 1.0e-6);

    let mut short_state = substrate.zeroed::<f32>((RATIO3 * general_width - 1) as usize)?;
    let mut valid_score = substrate.zeroed::<f32>((RATIO3 * general_width) as usize)?;
    let mut valid_comp =
        substrate.zeroed::<f32>(((general_tokens / RATIO3) * HEAD_DIM) as usize)?;
    assert!(matches!(
        compressor_prefill_tensor(
            &module,
            substrate.stream(),
            RATIO3,
            general_pos0,
            general_tokens,
            0,
            false,
            &general_kv,
            &general_sc,
            &general_ape,
            &unused_f16,
            &weights,
            &mut short_state,
            &mut valid_score,
            &mut valid_comp,
        ),
        Err(CompressorPrefillError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4c3b\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"general_ratio_prefill_remainder_matches\":true,\"ratio4_prefill_state_placement_matches\":true,\"ratio4_replay_output_before_state_rebuild_matches\":true,\"ratio4_state_only_matches\":true,\"weighted_rms_rope_composition_matches\":true,\"optional_fp8_output_matches\":true,\"f16_ape_prefill_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_compressor_prefill_orchestration\":{},\"owns_ratio4_replay_orchestration\":{},\"owns_ratio4_state_only_orchestration\":{},\"owns_optional_fp8_compressed_output\":{},\"owns_attention_kernels\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4C3B_SCOPE.owns_compressor_prefill_orchestration,
        M14_4C3B_SCOPE.owns_ratio4_replay_orchestration,
        M14_4C3B_SCOPE.owns_ratio4_state_only_orchestration,
        M14_4C3B_SCOPE.owns_optional_fp8_compressed_output,
        M14_4C3B_SCOPE.owns_attention_kernels,
        M14_4C3B_SCOPE.owns_runtime_graph_integration,
        M14_4C3B_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compressor_prefill_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
    ape_type: u32,
    quantize_fp8: bool,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    weights: &DeviceBuffer<f32>,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
    comp: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    validate_prefill(
        ratio,
        n_tokens,
        ape_type,
        kv,
        sc,
        ape_f32,
        ape_f16,
        weights,
        state_kv,
        state_score,
        comp,
    )?;
    initialize_state(module, stream, state_kv, state_score)?;
    let n_comp = n_tokens / ratio;
    let cutoff = n_comp * ratio;
    let rem = n_tokens - cutoff;
    if ratio == 4 {
        if cutoff >= ratio {
            set_rows(
                module,
                stream,
                ratio,
                pos0,
                cutoff - ratio,
                0,
                ratio,
                ape_type,
                kv,
                sc,
                ape_f32,
                ape_f16,
                state_kv,
                state_score,
            )?;
        }
        if rem != 0 {
            set_rows(
                module,
                stream,
                ratio,
                pos0,
                cutoff,
                ratio,
                rem,
                ape_type,
                kv,
                sc,
                ape_f32,
                ape_f16,
                state_kv,
                state_score,
            )?;
        }
    } else if rem != 0 {
        set_rows(
            module,
            stream,
            ratio,
            pos0,
            cutoff,
            0,
            rem,
            ape_type,
            kv,
            sc,
            ape_f32,
            ape_f16,
            state_kv,
            state_score,
        )?;
    }
    if n_comp != 0 {
        pool_and_postprocess(
            module,
            stream,
            ratio,
            pos0,
            n_comp,
            false,
            ape_type,
            quantize_fp8,
            kv,
            sc,
            ape_f32,
            ape_f16,
            weights,
            state_kv,
            state_score,
            comp,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compressor_prefill_ratio4_replay_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    pos0: u32,
    n_tokens: u32,
    ape_type: u32,
    quantize_fp8: bool,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    weights: &DeviceBuffer<f32>,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
    comp: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    if n_tokens == 0 || n_tokens & 3 != 0 || pos0 & 3 != 0 {
        return Err(CompressorPrefillError::InvalidShape);
    }
    validate_prefill(
        RATIO4,
        n_tokens,
        ape_type,
        kv,
        sc,
        ape_f32,
        ape_f16,
        weights,
        state_kv,
        state_score,
        comp,
    )?;
    pool_and_postprocess(
        module,
        stream,
        RATIO4,
        pos0,
        n_tokens / RATIO4,
        true,
        ape_type,
        quantize_fp8,
        kv,
        sc,
        ape_f32,
        ape_f16,
        weights,
        state_kv,
        state_score,
        comp,
    )?;
    initialize_state(module, stream, state_kv, state_score)?;
    set_rows(
        module,
        stream,
        RATIO4,
        pos0,
        n_tokens - RATIO4,
        0,
        RATIO4,
        ape_type,
        kv,
        sc,
        ape_f32,
        ape_f16,
        state_kv,
        state_score,
    )
}

#[allow(clippy::too_many_arguments)]
fn compressor_prefill_state_ratio4_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    pos0: u32,
    ape_type: u32,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    let count = (RATIO4 * width(RATIO4)) as usize;
    if ape_type > 1
        || kv.len() < count
        || sc.len() < count
        || state_kv.len() < (2 * count)
        || state_score.len() < (2 * count)
        || (ape_type == 0 && ape_f32.len() < count)
        || (ape_type == 1 && ape_f16.len() < count)
    {
        return Err(CompressorPrefillError::InvalidShape);
    }
    initialize_state(module, stream, state_kv, state_score)?;
    set_rows(
        module,
        stream,
        RATIO4,
        pos0,
        0,
        0,
        RATIO4,
        ape_type,
        kv,
        sc,
        ape_f32,
        ape_f16,
        state_kv,
        state_score,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_prefill(
    ratio: u32,
    n_tokens: u32,
    ape_type: u32,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    weights: &DeviceBuffer<f32>,
    state_kv: &DeviceBuffer<f32>,
    state_score: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    let width = width(ratio);
    let rows = if ratio == 4 { 2 * ratio } else { ratio };
    let n_comp = if ratio == 0 { 0 } else { n_tokens / ratio };
    if ratio == 0
        || n_tokens == 0
        || ape_type > 1
        || kv.len() < (n_tokens * width) as usize
        || sc.len() < (n_tokens * width) as usize
        || state_kv.len() < (rows * width) as usize
        || state_score.len() < (rows * width) as usize
        || (ape_type == 0 && ape_f32.len() < (ratio * width) as usize)
        || (ape_type == 1 && ape_f16.len() < (ratio * width) as usize)
        || weights.len() < HEAD_DIM as usize
        || comp.len() < (n_comp * HEAD_DIM) as usize
    {
        return Err(CompressorPrefillError::InvalidShape);
    }
    Ok(())
}

fn initialize_state(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    for (value, buffer) in [(0.0_f32, state_kv), (f32::NEG_INFINITY, state_score)] {
        module
            .fill_f32_kernel(
                stream,
                LaunchConfig {
                    grid_dim: ((buffer.len() as u32).div_ceil(THREADS), 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                buffer.len() as u32,
                value,
                buffer,
            )
            .map_err(CompressorPrefillError::Driver)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn set_rows(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    ratio: u32,
    pos0: u32,
    src0: u32,
    dst0: u32,
    rows: u32,
    ape_type: u32,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    let width = width(ratio);
    module
        .compressor_set_rows_kernel(
            stream,
            LaunchConfig {
                grid_dim: ((rows * width).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            width,
            ratio,
            pos0,
            src0,
            dst0,
            rows,
            ape_type,
            kv,
            sc,
            ape_f32,
            ape_f16,
            state_kv,
            state_score,
        )
        .map_err(CompressorPrefillError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn pool_and_postprocess(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    ratio: u32,
    pos0: u32,
    n_comp: u32,
    replay: bool,
    ape_type: u32,
    quantize_fp8: bool,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    weights: &DeviceBuffer<f32>,
    state_kv: &DeviceBuffer<f32>,
    state_score: &DeviceBuffer<f32>,
    comp: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPrefillError> {
    module
        .compressor_prefill_pool_kernel(
            stream,
            LaunchConfig {
                grid_dim: (HEAD_DIM.div_ceil(THREADS), n_comp, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            HEAD_DIM,
            ratio,
            pos0,
            n_comp,
            u32::from(replay),
            ape_type,
            kv,
            sc,
            state_kv,
            state_score,
            ape_f32,
            ape_f16,
            comp,
        )
        .map_err(CompressorPrefillError::Driver)?;
    module
        .rms_norm_weight_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_comp, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            HEAD_DIM,
            n_comp,
            RMS_EPS,
            weights,
            comp,
        )
        .map_err(CompressorPrefillError::Driver)?;
    module
        .rope_tail_kernel(
            stream,
            LaunchConfig {
                grid_dim: ((n_comp * (N_ROT / 2)).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            HEAD_DIM,
            N_ROT,
            pos0,
            ratio,
            N_CTX_ORIG,
            FREQ_BASE,
            FREQ_SCALE,
            EXT_FACTOR,
            ATTN_FACTOR,
            BETA_FAST,
            BETA_SLOW,
            comp,
        )
        .map_err(CompressorPrefillError::Driver)?;
    if quantize_fp8 {
        module
            .fp8_kv_quantize_kernel(
                stream,
                LaunchConfig {
                    grid_dim: (n_comp, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                },
                n_comp,
                HEAD_DIM,
                N_ROT,
                comp,
            )
            .map_err(CompressorPrefillError::Driver)?;
    }
    Ok(())
}

fn width(ratio: u32) -> u32 {
    (if ratio == 4 { 2 } else { 1 }) * HEAD_DIM
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn expected_prefill(
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
    replay: bool,
    quantize_fp8: bool,
    kv: &[f32],
    sc: &[f32],
    ape: &[f32],
    weights: &[f32],
    prior_state_kv: &[f32],
    prior_state_score: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_comp = n_tokens / ratio;
    let state_count = ((if ratio == 4 { 2 * ratio } else { ratio }) * width(ratio)) as usize;
    let source_state_kv = if replay {
        prior_state_kv.to_vec()
    } else {
        vec![0.0_f32; state_count]
    };
    let source_state_score = if replay {
        prior_state_score.to_vec()
    } else {
        vec![f32::NEG_INFINITY; state_count]
    };
    let mut comp = expected_pool(
        ratio,
        pos0,
        n_comp,
        replay,
        kv,
        sc,
        ape,
        &source_state_kv,
        &source_state_score,
    );
    postprocess_host(&mut comp, n_comp, ratio, pos0, quantize_fp8, weights);
    let mut state_kv = vec![0.0_f32; state_count];
    let mut state_score = vec![f32::NEG_INFINITY; state_count];
    let cutoff = n_comp * ratio;
    let rem = n_tokens - cutoff;
    if replay {
        set_rows_host(
            ratio,
            pos0,
            n_tokens - ratio,
            0,
            ratio,
            kv,
            sc,
            ape,
            &mut state_kv,
            &mut state_score,
        );
    } else if ratio == 4 {
        if cutoff >= ratio {
            set_rows_host(
                ratio,
                pos0,
                cutoff - ratio,
                0,
                ratio,
                kv,
                sc,
                ape,
                &mut state_kv,
                &mut state_score,
            );
        }
        if rem != 0 {
            set_rows_host(
                ratio,
                pos0,
                cutoff,
                ratio,
                rem,
                kv,
                sc,
                ape,
                &mut state_kv,
                &mut state_score,
            );
        }
    } else if rem != 0 {
        set_rows_host(
            ratio,
            pos0,
            cutoff,
            0,
            rem,
            kv,
            sc,
            ape,
            &mut state_kv,
            &mut state_score,
        );
    }
    (state_kv, state_score, comp)
}

fn expected_state_ratio4(pos0: u32, kv: &[f32], sc: &[f32], ape: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let count = (2 * RATIO4 * width(RATIO4)) as usize;
    let mut state_kv = vec![0.0_f32; count];
    let mut state_score = vec![f32::NEG_INFINITY; count];
    set_rows_host(
        RATIO4,
        pos0,
        0,
        0,
        RATIO4,
        kv,
        sc,
        ape,
        &mut state_kv,
        &mut state_score,
    );
    (state_kv, state_score)
}

#[allow(clippy::too_many_arguments)]
fn set_rows_host(
    ratio: u32,
    pos0: u32,
    src0: u32,
    dst0: u32,
    rows: u32,
    kv: &[f32],
    sc: &[f32],
    ape: &[f32],
    state_kv: &mut [f32],
    state_score: &mut [f32],
) {
    let width = width(ratio) as usize;
    for row in 0..rows as usize {
        let src = src0 as usize + row;
        let dst = dst0 as usize + row;
        let phase = (pos0 as usize + src) % ratio as usize;
        for dimension in 0..width {
            state_kv[dst * width + dimension] = kv[src * width + dimension];
            state_score[dst * width + dimension] =
                sc[src * width + dimension] + ape[phase * width + dimension];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn expected_pool(
    ratio: u32,
    pos0: u32,
    n_comp: u32,
    replay: bool,
    kv: &[f32],
    sc: &[f32],
    ape: &[f32],
    state_kv: &[f32],
    state_score: &[f32],
) -> Vec<f32> {
    let width = width(ratio);
    let mut comp = vec![0.0_f32; (n_comp * HEAD_DIM) as usize];
    for compressed in 0..n_comp {
        for dimension in 0..HEAD_DIM {
            let mut candidates = Vec::new();
            if ratio == 4 {
                if replay && compressed == 0 {
                    for row in 0..4 {
                        candidates.push((
                            state_kv[(row * width + dimension) as usize],
                            state_score[(row * width + dimension) as usize],
                        ));
                    }
                } else if compressed > 0 {
                    for row in 0..4 {
                        let token = (compressed - 1) * ratio + row;
                        let phase = (pos0 + token) % ratio;
                        candidates.push((
                            kv[(token * width + dimension) as usize],
                            sc[(token * width + dimension) as usize]
                                + ape[(phase * width + dimension) as usize],
                        ));
                    }
                }
                for row in 0..4 {
                    let token = compressed * ratio + row;
                    let phase = (pos0 + token) % ratio;
                    candidates.push((
                        kv[(token * width + HEAD_DIM + dimension) as usize],
                        sc[(token * width + HEAD_DIM + dimension) as usize]
                            + ape[(phase * width + HEAD_DIM + dimension) as usize],
                    ));
                }
            } else {
                for row in 0..ratio {
                    let token = compressed * ratio + row;
                    let phase = (pos0 + token) % ratio;
                    candidates.push((
                        kv[(token * width + dimension) as usize],
                        sc[(token * width + dimension) as usize]
                            + ape[(phase * width + dimension) as usize],
                    ));
                }
            }
            comp[(compressed * HEAD_DIM + dimension) as usize] = softmax_pool(&candidates);
        }
    }
    comp
}

fn softmax_pool(candidates: &[(f32, f32)]) -> f32 {
    let max_score = candidates
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0_f32;
    let mut accumulator = 0.0_f32;
    for (value, score) in candidates {
        let weight = (*score - max_score).exp();
        denominator += weight;
        accumulator += value * weight;
    }
    accumulator / denominator
}

fn postprocess_host(
    comp: &mut [f32],
    n_comp: u32,
    ratio: u32,
    pos0: u32,
    quantize_fp8: bool,
    weights: &[f32],
) {
    for row in 0..n_comp as usize {
        let values = &mut comp[row * HEAD_DIM as usize..(row + 1) * HEAD_DIM as usize];
        let scale = 1.0_f32
            / (values.iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32 + RMS_EPS)
                .sqrt();
        for (value, weight) in values.iter_mut().zip(weights) {
            *value *= scale * weight;
        }
        rope_row(values, pos0 + row as u32 * ratio);
        if quantize_fp8 {
            fp8_prefix(values);
        }
    }
}

fn rope_row(row: &mut [f32], pos: u32) {
    let n_nope = (HEAD_DIM - N_ROT) as usize;
    let denom = 2.0 * FREQ_BASE.ln();
    let corr0 = (N_ROT as f32
        * (N_CTX_ORIG as f32 / (BETA_FAST * 2.0 * std::f32::consts::PI)).ln()
        / denom)
        .floor()
        .max(0.0);
    let corr1 = (N_ROT as f32
        * (N_CTX_ORIG as f32 / (BETA_SLOW * 2.0 * std::f32::consts::PI)).ln()
        / denom)
        .ceil()
        .min((N_ROT - 1) as f32);
    for pair in 0..N_ROT as usize / 2 {
        let rot_i = pair * 2;
        let theta_extrap = pos as f32 * FREQ_BASE.powf(-(rot_i as f32) / N_ROT as f32);
        let theta_interp = FREQ_SCALE * theta_extrap;
        let ramp_mix = (1.0 - ((pair as f32 - corr0) / (corr1 - corr0).max(0.001)).clamp(0.0, 1.0))
            * EXT_FACTOR;
        let theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        let scale = ATTN_FACTOR * (1.0 + 0.1 * (1.0 / FREQ_SCALE).ln());
        let c = theta.cos() * scale;
        let s = theta.sin() * scale;
        let x0 = row[n_nope + rot_i];
        let x1 = row[n_nope + rot_i + 1];
        row[n_nope + rot_i] = x0 * c - x1 * s;
        row[n_nope + rot_i + 1] = x0 * s + x1 * c;
    }
}

fn fp8_prefix(row: &mut [f32]) {
    let prefix = (HEAD_DIM - N_ROT) as usize;
    for chunk in row[..prefix].chunks_mut(64) {
        let amax = chunk
            .iter()
            .map(|value| value.abs())
            .fold(1.0e-4_f32, f32::max);
        let scale = 2.0_f32.powf((amax / 448.0).log2().ceil());
        for value in chunk {
            *value = e4m3fn_dequant_host(*value / scale) * scale;
        }
    }
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
            actual == expected || (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum CompressorPrefillError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for CompressorPrefillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("compressor prefill tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompressorPrefillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
