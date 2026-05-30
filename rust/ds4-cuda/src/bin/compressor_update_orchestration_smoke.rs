#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4C3A_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn compressor_store_kernel(
        head_dim: u32,
        ratio: u32,
        pos: u32,
        ape_type: u32,
        kv: &[f32],
        sc: &[f32],
        ape_f32: &[f32],
        ape_f16: &[f16],
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
    ) {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let dimension = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if dimension >= width {
            return;
        }
        let phase = pos % ratio;
        let row = if ratio == 4 { ratio + phase } else { phase };
        let ape_index = (phase * width + dimension) as usize;
        let ape = model_scalar(ape_type, ape_index, ape_f32, ape_f16);
        unsafe {
            *state_kv.get_unchecked_mut((row * width + dimension) as usize) =
                kv[dimension as usize];
            *state_score.get_unchecked_mut((row * width + dimension) as usize) =
                sc[dimension as usize] + ape;
        }
    }

    #[kernel]
    pub fn compressor_update_pool_kernel(
        head_dim: u32,
        ratio: u32,
        comp_offset: u32,
        state_kv: &[f32],
        state_score: &[f32],
        mut comp: DisjointSlice<f32>,
    ) {
        let dimension = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if dimension >= head_dim {
            return;
        }
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let mut max_score = f32::NEG_INFINITY;
        let mut candidate = 0_u32;
        if ratio == 4 {
            while candidate < 4 {
                max_score = maximum(
                    max_score,
                    state_score[(candidate * width + dimension) as usize],
                );
                candidate += 1;
            }
            candidate = 0;
            while candidate < 4 {
                max_score = maximum(
                    max_score,
                    state_score[((ratio + candidate) * width + head_dim + dimension) as usize],
                );
                candidate += 1;
            }
        } else {
            while candidate < ratio {
                max_score = maximum(
                    max_score,
                    state_score[(candidate * width + dimension) as usize],
                );
                candidate += 1;
            }
        }
        let mut denominator = 0.0_f32;
        let mut accumulator = 0.0_f32;
        candidate = 0;
        if ratio == 4 {
            while candidate < 4 {
                add_candidate(
                    state_kv[(candidate * width + dimension) as usize],
                    state_score[(candidate * width + dimension) as usize],
                    max_score,
                    &mut denominator,
                    &mut accumulator,
                );
                candidate += 1;
            }
            candidate = 0;
            while candidate < 4 {
                let index = ((ratio + candidate) * width + head_dim + dimension) as usize;
                add_candidate(
                    state_kv[index],
                    state_score[index],
                    max_score,
                    &mut denominator,
                    &mut accumulator,
                );
                candidate += 1;
            }
        } else {
            while candidate < ratio {
                add_candidate(
                    state_kv[(candidate * width + dimension) as usize],
                    state_score[(candidate * width + dimension) as usize],
                    max_score,
                    &mut denominator,
                    &mut accumulator,
                );
                candidate += 1;
            }
        }
        unsafe {
            *comp.get_unchecked_mut((comp_offset + dimension) as usize) = if denominator != 0.0 {
                accumulator / denominator
            } else {
                0.0
            };
        }
    }

    #[kernel]
    pub fn rms_norm_weight_kernel(
        n: u32,
        offset: u32,
        eps: f32,
        weight: &[f32],
        mut x: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let base = offset as usize;
        let n = n as usize;
        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < n {
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
        while i < n {
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
        offset: u32,
        head_dim: u32,
        n_rot: u32,
        pos: u32,
        n_ctx_orig: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        mut x: DisjointSlice<f32>,
    ) {
        let pair = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if pair >= n_rot / 2 {
            return;
        }
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

        let theta_extrap = pos as f32 * freq_base.powf(-(rot_i as f32) / n_rot as f32);
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
        let s = theta.sin() * mscale;
        let base = (offset + n_nope + rot_i) as usize;
        let x0 = unsafe { *x.as_mut_ptr().add(base) };
        let x1 = unsafe { *x.as_mut_ptr().add(base + 1) };
        unsafe {
            *x.get_unchecked_mut(base) = x0 * c - x1 * s;
            *x.get_unchecked_mut(base + 1) = x0 * s + x1 * c;
        }
    }

    #[kernel]
    pub fn compressor_shift_ratio4_kernel(
        width: u32,
        mut state_kv: DisjointSlice<f32>,
        mut state_score: DisjointSlice<f32>,
    ) {
        let index = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
            + thread::threadIdx_x() as u64;
        let half = 4_u64 * width as u64;
        if index >= half {
            return;
        }
        let kv = unsafe { *state_kv.as_mut_ptr().add((half + index) as usize) };
        let score = unsafe { *state_score.as_mut_ptr().add((half + index) as usize) };
        unsafe {
            *state_kv.get_unchecked_mut(index as usize) = kv;
            *state_score.get_unchecked_mut(index as usize) = score;
            *state_kv.get_unchecked_mut((half + index) as usize) = kv;
            *state_score.get_unchecked_mut((half + index) as usize) = score;
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

    fn model_scalar(ape_type: u32, index: usize, ape_f32: &[f32], ape_f16: &[f16]) -> f32 {
        if ape_type == 1 {
            ape_f16[index] as f32
        } else {
            ape_f32[index]
        }
    }
}

const THREADS: u32 = 256;
const HEAD_DIM: u32 = 6;
const N_ROT: u32 = 4;
const RATIO4: u32 = 4;
const RATIO3: u32 = 3;
const COMP_ROW: u32 = 1;
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
        "../../ds4_cuda_compressor_update_orchestration_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let weight_values = values(HEAD_DIM as usize, 17, 0.5);
    let weights = substrate.upload(&weight_values)?;

    let ratio4_initial = values((2 * RATIO4 * width(RATIO4)) as usize, 23, -1.75);
    let ratio4_scores = values((2 * RATIO4 * width(RATIO4)) as usize, 29, -1.125);
    let ratio4_kv_values = values(width(RATIO4) as usize, 31, -1.5);
    let ratio4_sc_values = values(width(RATIO4) as usize, 37, -0.875);
    let ratio4_ape_f16 = values((RATIO4 * width(RATIO4)) as usize, 11, -0.25)
        .into_iter()
        .map(|value| (value + 0.015625) as f16)
        .collect::<Vec<_>>();
    let ratio4_ape_values = ratio4_ape_f16
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let unused_f32 = substrate.upload(&vec![0.0_f32; (RATIO4 * width(RATIO4)) as usize])?;
    let ratio4_ape = substrate.upload(&ratio4_ape_f16)?;
    let ratio4_kv = substrate.upload(&ratio4_kv_values)?;
    let ratio4_sc = substrate.upload(&ratio4_sc_values)?;

    let comp_initial = vec![-99.0_f32; ((COMP_ROW + 1) * HEAD_DIM) as usize];
    let mut no_emit_state = substrate.upload(&ratio4_initial)?;
    let mut no_emit_score = substrate.upload(&ratio4_scores)?;
    let mut no_emit_comp = substrate.upload(&comp_initial)?;
    assert!(!compressor_update_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        RATIO4,
        6,
        COMP_ROW,
        1,
        &ratio4_kv,
        &ratio4_sc,
        &unused_f32,
        &ratio4_ape,
        &weights,
        &mut no_emit_state,
        &mut no_emit_score,
        &mut no_emit_comp,
    )?);
    substrate.end_commands()?;
    let expected = expected_update(
        &ratio4_initial,
        &ratio4_scores,
        &comp_initial,
        &ratio4_kv_values,
        &ratio4_sc_values,
        &ratio4_ape_values,
        &weight_values,
        RATIO4,
        6,
    );
    assert_close(&substrate.download(&no_emit_state)?, &expected.0, 1.0e-6);
    assert_close(&substrate.download(&no_emit_score)?, &expected.1, 1.0e-6);
    assert_eq!(substrate.download(&no_emit_comp)?, expected.2);

    let mut ratio4_state = substrate.upload(&ratio4_initial)?;
    let mut ratio4_score = substrate.upload(&ratio4_scores)?;
    let mut ratio4_comp = substrate.upload(&comp_initial)?;
    assert!(compressor_update_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        RATIO4,
        7,
        COMP_ROW,
        1,
        &ratio4_kv,
        &ratio4_sc,
        &unused_f32,
        &ratio4_ape,
        &weights,
        &mut ratio4_state,
        &mut ratio4_score,
        &mut ratio4_comp,
    )?);
    substrate.end_commands()?;
    let expected = expected_update(
        &ratio4_initial,
        &ratio4_scores,
        &comp_initial,
        &ratio4_kv_values,
        &ratio4_sc_values,
        &ratio4_ape_values,
        &weight_values,
        RATIO4,
        7,
    );
    assert_close(&substrate.download(&ratio4_state)?, &expected.0, 1.0e-6);
    assert_close(&substrate.download(&ratio4_score)?, &expected.1, 1.0e-6);
    assert_close(&substrate.download(&ratio4_comp)?, &expected.2, 4.0e-5);

    let ratio3_initial = values((RATIO3 * width(RATIO3)) as usize, 41, -1.5);
    let ratio3_scores = values((RATIO3 * width(RATIO3)) as usize, 43, -1.0);
    let ratio3_kv_values = values(width(RATIO3) as usize, 47, -1.25);
    let ratio3_sc_values = values(width(RATIO3) as usize, 53, -0.75);
    let ratio3_ape_values = values((RATIO3 * width(RATIO3)) as usize, 13, -0.125);
    let ratio3_kv = substrate.upload(&ratio3_kv_values)?;
    let ratio3_sc = substrate.upload(&ratio3_sc_values)?;
    let ratio3_ape = substrate.upload(&ratio3_ape_values)?;
    let unused_f16 =
        substrate.upload(&vec![f16::from_bits(0); (RATIO3 * width(RATIO3)) as usize])?;
    let mut ratio3_state = substrate.upload(&ratio3_initial)?;
    let mut ratio3_score = substrate.upload(&ratio3_scores)?;
    let mut ratio3_comp = substrate.upload(&comp_initial)?;
    assert!(compressor_update_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        RATIO3,
        5,
        COMP_ROW,
        0,
        &ratio3_kv,
        &ratio3_sc,
        &ratio3_ape,
        &unused_f16,
        &weights,
        &mut ratio3_state,
        &mut ratio3_score,
        &mut ratio3_comp,
    )?);
    substrate.end_commands()?;
    let expected = expected_update(
        &ratio3_initial,
        &ratio3_scores,
        &comp_initial,
        &ratio3_kv_values,
        &ratio3_sc_values,
        &ratio3_ape_values,
        &weight_values,
        RATIO3,
        5,
    );
    assert_close(&substrate.download(&ratio3_state)?, &expected.0, 1.0e-6);
    assert_close(&substrate.download(&ratio3_score)?, &expected.1, 1.0e-6);
    assert_close(&substrate.download(&ratio3_comp)?, &expected.2, 4.0e-5);

    let mut too_short = substrate.zeroed::<f32>((2 * RATIO4 * width(RATIO4) - 1) as usize)?;
    let mut valid_score = substrate.upload(&ratio4_scores)?;
    let mut valid_comp = substrate.upload(&comp_initial)?;
    assert!(matches!(
        compressor_update_tensor(
            &module,
            substrate.stream(),
            HEAD_DIM,
            RATIO4,
            7,
            COMP_ROW,
            1,
            &ratio4_kv,
            &ratio4_sc,
            &unused_f32,
            &ratio4_ape,
            &weights,
            &mut too_short,
            &mut valid_score,
            &mut valid_comp,
        ),
        Err(CompressorUpdateError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4c3a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"ratio4_no_emit_store_only_matches\":true,\"ratio4_emit_composed_output_matches\":true,\"general_ratio_emit_composed_output_matches\":true,\"weighted_rms_composition_matches\":true,\"rope_composition_matches\":true,\"ratio4_shift_after_emit_matches\":true,\"f16_ape_update_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_compressor_update_orchestration\":{},\"owns_store_pool_norm_rope_shift_sequence\":{},\"owns_compressor_prefill_orchestration\":{},\"owns_attention_kernels\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4C3A_SCOPE.owns_compressor_update_orchestration,
        M14_4C3A_SCOPE.owns_store_pool_norm_rope_shift_sequence,
        M14_4C3A_SCOPE.owns_compressor_prefill_orchestration,
        M14_4C3A_SCOPE.owns_attention_kernels,
        M14_4C3A_SCOPE.owns_runtime_graph_integration,
        M14_4C3A_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compressor_update_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    head_dim: u32,
    ratio: u32,
    pos: u32,
    comp_row: u32,
    ape_type: u32,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    weight: &DeviceBuffer<f32>,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
    comp: &mut DeviceBuffer<f32>,
) -> Result<bool, CompressorUpdateError> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let state_rows = coff * ratio;
    let emit = ratio != 0 && (pos + 1) % ratio == 0;
    if head_dim == 0
        || ratio == 0
        || N_ROT > head_dim
        || N_ROT & 1 != 0
        || ape_type > 1
        || kv.len() < width as usize
        || sc.len() < width as usize
        || state_kv.len() < (state_rows * width) as usize
        || state_score.len() < (state_rows * width) as usize
        || (ape_type == 0 && ape_f32.len() < (ratio * width) as usize)
        || (ape_type == 1 && ape_f16.len() < (ratio * width) as usize)
        || weight.len() < head_dim as usize
        || (emit && comp.len() < ((comp_row + 1) * head_dim) as usize)
    {
        return Err(CompressorUpdateError::InvalidShape);
    }
    module
        .compressor_store_kernel(
            stream,
            LaunchConfig {
                grid_dim: (width.div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            head_dim,
            ratio,
            pos,
            ape_type,
            kv,
            sc,
            ape_f32,
            ape_f16,
            state_kv,
            state_score,
        )
        .map_err(CompressorUpdateError::Driver)?;
    if !emit {
        return Ok(false);
    }
    let offset = comp_row * head_dim;
    module
        .compressor_update_pool_kernel(
            stream,
            LaunchConfig {
                grid_dim: (head_dim.div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            head_dim,
            ratio,
            offset,
            state_kv,
            state_score,
            comp,
        )
        .map_err(CompressorUpdateError::Driver)?;
    module
        .rms_norm_weight_kernel(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            head_dim,
            offset,
            RMS_EPS,
            weight,
            comp,
        )
        .map_err(CompressorUpdateError::Driver)?;
    module
        .rope_tail_kernel(
            stream,
            LaunchConfig {
                grid_dim: ((N_ROT / 2).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            offset,
            head_dim,
            N_ROT,
            pos + 1 - ratio,
            N_CTX_ORIG,
            FREQ_BASE,
            FREQ_SCALE,
            EXT_FACTOR,
            ATTN_FACTOR,
            BETA_FAST,
            BETA_SLOW,
            comp,
        )
        .map_err(CompressorUpdateError::Driver)?;
    if ratio == 4 {
        let half = 4 * width;
        module
            .compressor_shift_ratio4_kernel(
                stream,
                LaunchConfig {
                    grid_dim: (half.div_ceil(THREADS), 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                width,
                state_kv,
                state_score,
            )
            .map_err(CompressorUpdateError::Driver)?;
    }
    Ok(true)
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
fn expected_update(
    initial_kv: &[f32],
    initial_score: &[f32],
    initial_comp: &[f32],
    kv: &[f32],
    sc: &[f32],
    ape: &[f32],
    weight: &[f32],
    ratio: u32,
    pos: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let row_width = width(ratio) as usize;
    let phase = (pos % ratio) as usize;
    let state_row = if ratio == 4 {
        ratio as usize + phase
    } else {
        phase
    };
    let mut state_kv = initial_kv.to_vec();
    let mut state_score = initial_score.to_vec();
    for dimension in 0..row_width {
        let output = state_row * row_width + dimension;
        state_kv[output] = kv[dimension];
        state_score[output] = sc[dimension] + ape[phase * row_width + dimension];
    }
    let mut comp = initial_comp.to_vec();
    if (pos + 1) % ratio == 0 {
        let mut row = expected_update_pool(ratio, &state_kv, &state_score);
        let scale = 1.0_f32
            / (row.iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32 + RMS_EPS)
                .sqrt();
        for (value, weight) in row.iter_mut().zip(weight) {
            *value *= scale * weight;
        }
        rope_row(&mut row, pos + 1 - ratio);
        let offset = (COMP_ROW * HEAD_DIM) as usize;
        comp[offset..offset + HEAD_DIM as usize].copy_from_slice(&row);
        if ratio == 4 {
            let half = (4 * width(ratio)) as usize;
            for index in 0..half {
                state_kv[index] = state_kv[half + index];
                state_score[index] = state_score[half + index];
            }
        }
    }
    (state_kv, state_score, comp)
}

fn expected_update_pool(ratio: u32, state_kv: &[f32], state_score: &[f32]) -> Vec<f32> {
    let width = width(ratio);
    let mut output = vec![0.0_f32; HEAD_DIM as usize];
    for dimension in 0..HEAD_DIM {
        let mut candidates = Vec::new();
        if ratio == 4 {
            for row in 0..4 {
                candidates.push((
                    state_kv[(row * width + dimension) as usize],
                    state_score[(row * width + dimension) as usize],
                ));
                let active = ((ratio + row) * width + HEAD_DIM + dimension) as usize;
                candidates.push((state_kv[active], state_score[active]));
            }
        } else {
            for row in 0..ratio {
                candidates.push((
                    state_kv[(row * width + dimension) as usize],
                    state_score[(row * width + dimension) as usize],
                ));
            }
        }
        output[dimension as usize] = softmax_pool(&candidates);
    }
    output
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
enum CompressorUpdateError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for CompressorUpdateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("compressor update tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompressorUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
