#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4C2_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

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
                    let score = sc[(token * width + dimension) as usize]
                        + model_scalar(
                            ape_type,
                            (phase * width + dimension) as usize,
                            ape_f32,
                            ape_f16,
                        );
                    max_score = maximum(max_score, score);
                    row += 1;
                }
            }
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < 4 {
                let token = base + row;
                let phase = (pos0 + token) % ratio;
                let score = sc[(token * width + head_dim + dimension) as usize]
                    + model_scalar(
                        ape_type,
                        (phase * width + head_dim + dimension) as usize,
                        ape_f32,
                        ape_f16,
                    );
                max_score = maximum(max_score, score);
                row += 1;
            }
        } else {
            let base = compressed * ratio;
            let mut row = 0_u32;
            while row < ratio {
                let token = base + row;
                let phase = (pos0 + token) % ratio;
                let score = sc[(token * width + dimension) as usize]
                    + model_scalar(
                        ape_type,
                        (phase * width + dimension) as usize,
                        ape_f32,
                        ape_f16,
                    );
                max_score = maximum(max_score, score);
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
                    let score = sc[(token * width + dimension) as usize]
                        + model_scalar(
                            ape_type,
                            (phase * width + dimension) as usize,
                            ape_f32,
                            ape_f16,
                        );
                    add_candidate(
                        kv[(token * width + dimension) as usize],
                        score,
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
                let score = sc[(token * width + head_dim + dimension) as usize]
                    + model_scalar(
                        ape_type,
                        (phase * width + head_dim + dimension) as usize,
                        ape_f32,
                        ape_f16,
                    );
                add_candidate(
                    kv[(token * width + head_dim + dimension) as usize],
                    score,
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
                let score = sc[(token * width + dimension) as usize]
                    + model_scalar(
                        ape_type,
                        (phase * width + dimension) as usize,
                        ape_f32,
                        ape_f16,
                    );
                add_candidate(
                    kv[(token * width + dimension) as usize],
                    score,
                    max_score,
                    &mut denominator,
                    &mut accumulator,
                );
                row += 1;
            }
        }
        unsafe {
            *comp.get_unchecked_mut((compressed * head_dim + dimension) as usize) =
                if denominator != 0.0 {
                    accumulator / denominator
                } else {
                    0.0
                };
        }
    }

    #[kernel]
    pub fn compressor_update_pool_kernel(
        head_dim: u32,
        ratio: u32,
        state_kv: &[f32],
        state_score: &[f32],
        mut row: DisjointSlice<f32>,
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
            *row.get_unchecked_mut(dimension as usize) = if denominator != 0.0 {
                accumulator / denominator
            } else {
                0.0
            };
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
const HEAD_DIM: u32 = 5;
const GENERAL_RATIO: u32 = 3;
const GENERAL_COMP: u32 = 2;
const GENERAL_TOKENS: u32 = GENERAL_RATIO * GENERAL_COMP;
const RATIO4: u32 = 4;
const RATIO4_WIDTH: u32 = 2 * HEAD_DIM;
const RATIO4_COMP: u32 = 2;
const RATIO4_TOKENS: u32 = RATIO4 * RATIO4_COMP;
const POS0: u32 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_compressor_pool_shift_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;

    let general_kv_values = values((GENERAL_TOKENS * HEAD_DIM) as usize, 13, -1.0);
    let general_score_values = values((GENERAL_TOKENS * HEAD_DIM) as usize, 7, -0.75);
    let general_ape_values = values((GENERAL_RATIO * HEAD_DIM) as usize, 5, -0.25);
    let general_state_values = values((GENERAL_RATIO * HEAD_DIM) as usize, 3, -2.0);
    let general_state_scores = values((GENERAL_RATIO * HEAD_DIM) as usize, 9, -1.25);
    let general_kv = substrate.upload(&general_kv_values)?;
    let general_score = substrate.upload(&general_score_values)?;
    let general_ape = substrate.upload(&general_ape_values)?;
    let unused_f16 =
        substrate.upload(&vec![f16::from_bits(0); (RATIO4 * RATIO4_WIDTH) as usize])?;
    let general_state = substrate.upload(&general_state_values)?;
    let general_state_score = substrate.upload(&general_state_scores)?;
    let mut general_comp = substrate.zeroed::<f32>((GENERAL_COMP * HEAD_DIM) as usize)?;
    prefill_pool_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        GENERAL_RATIO,
        GENERAL_COMP,
        false,
        0,
        &general_kv,
        &general_score,
        &general_state,
        &general_state_score,
        &general_ape,
        &unused_f16,
        &mut general_comp,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&general_comp)?,
        &expected_prefill_pool(
            HEAD_DIM,
            GENERAL_RATIO,
            GENERAL_COMP,
            false,
            &general_kv_values,
            &general_score_values,
            &general_state_values,
            &general_state_scores,
            &general_ape_values,
        ),
        1.0e-5,
    );

    let ratio4_kv_values = values((RATIO4_TOKENS * RATIO4_WIDTH) as usize, 17, -1.5);
    let ratio4_score_values = values((RATIO4_TOKENS * RATIO4_WIDTH) as usize, 11, -1.0);
    let ratio4_ape_f32_values = values((RATIO4 * RATIO4_WIDTH) as usize, 19, -0.5);
    let ratio4_ape_f16_values = ratio4_ape_f32_values
        .iter()
        .map(|value| (*value + 0.015625) as f16)
        .collect::<Vec<_>>();
    let ratio4_ape_f16_as_f32 = ratio4_ape_f16_values
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let ratio4_state_values = values((2 * RATIO4 * RATIO4_WIDTH) as usize, 23, -1.75);
    let ratio4_state_scores = values((2 * RATIO4 * RATIO4_WIDTH) as usize, 29, -1.125);
    let ratio4_kv = substrate.upload(&ratio4_kv_values)?;
    let ratio4_score = substrate.upload(&ratio4_score_values)?;
    let ratio4_ape_f32 = substrate.upload(&ratio4_ape_f32_values)?;
    let ratio4_ape_f16 = substrate.upload(&ratio4_ape_f16_values)?;
    let ratio4_state = substrate.upload(&ratio4_state_values)?;
    let ratio4_state_score = substrate.upload(&ratio4_state_scores)?;

    let mut ratio4_comp = substrate.zeroed::<f32>((RATIO4_COMP * HEAD_DIM) as usize)?;
    prefill_pool_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        RATIO4,
        RATIO4_COMP,
        false,
        0,
        &ratio4_kv,
        &ratio4_score,
        &ratio4_state,
        &ratio4_state_score,
        &ratio4_ape_f32,
        &unused_f16,
        &mut ratio4_comp,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&ratio4_comp)?,
        &expected_prefill_pool(
            HEAD_DIM,
            RATIO4,
            RATIO4_COMP,
            false,
            &ratio4_kv_values,
            &ratio4_score_values,
            &ratio4_state_values,
            &ratio4_state_scores,
            &ratio4_ape_f32_values,
        ),
        1.0e-5,
    );

    let unused_f32 = substrate.upload(&vec![0.0_f32; (RATIO4 * RATIO4_WIDTH) as usize])?;
    let mut replay_comp = substrate.zeroed::<f32>((RATIO4_COMP * HEAD_DIM) as usize)?;
    prefill_pool_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        RATIO4,
        RATIO4_COMP,
        true,
        1,
        &ratio4_kv,
        &ratio4_score,
        &ratio4_state,
        &ratio4_state_score,
        &unused_f32,
        &ratio4_ape_f16,
        &mut replay_comp,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&replay_comp)?,
        &expected_prefill_pool(
            HEAD_DIM,
            RATIO4,
            RATIO4_COMP,
            true,
            &ratio4_kv_values,
            &ratio4_score_values,
            &ratio4_state_values,
            &ratio4_state_scores,
            &ratio4_ape_f16_as_f32,
        ),
        1.0e-5,
    );

    let mut general_update = substrate.zeroed::<f32>(HEAD_DIM as usize)?;
    update_pool_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        GENERAL_RATIO,
        &general_state,
        &general_state_score,
        &mut general_update,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&general_update)?,
        &expected_update_pool(
            HEAD_DIM,
            GENERAL_RATIO,
            &general_state_values,
            &general_state_scores,
        ),
        1.0e-5,
    );
    let mut ratio4_update = substrate.zeroed::<f32>(HEAD_DIM as usize)?;
    update_pool_tensor(
        &module,
        substrate.stream(),
        HEAD_DIM,
        RATIO4,
        &ratio4_state,
        &ratio4_state_score,
        &mut ratio4_update,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&ratio4_update)?,
        &expected_update_pool(HEAD_DIM, RATIO4, &ratio4_state_values, &ratio4_state_scores),
        1.0e-5,
    );

    let mut shifted_kv = substrate.upload(&ratio4_state_values)?;
    let mut shifted_score = substrate.upload(&ratio4_state_scores)?;
    shift_ratio4_tensor(
        &module,
        substrate.stream(),
        &mut shifted_kv,
        &mut shifted_score,
    )?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&shifted_kv)?,
        expected_shift_ratio4(&ratio4_state_values)
    );
    assert_eq!(
        substrate.download(&shifted_score)?,
        expected_shift_ratio4(&ratio4_state_scores)
    );

    let mut short_comp = substrate.zeroed::<f32>((RATIO4_COMP * HEAD_DIM - 1) as usize)?;
    assert!(matches!(
        prefill_pool_tensor(
            &module,
            substrate.stream(),
            HEAD_DIM,
            RATIO4,
            RATIO4_COMP,
            false,
            0,
            &ratio4_kv,
            &ratio4_score,
            &ratio4_state,
            &ratio4_state_score,
            &ratio4_ape_f32,
            &unused_f16,
            &mut short_comp,
        ),
        Err(CompressorPoolShiftError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4c2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"general_ratio_prefill_pool_matches\":true,\"ratio4_prefill_pool_matches\":true,\"ratio4_replay_pool_matches\":true,\"general_ratio_update_pool_matches\":true,\"ratio4_update_pool_matches\":true,\"ratio4_shift_matches\":true,\"f16_ape_pool_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_compressor_prefill_pool_kernel\":{},\"owns_general_and_ratio4_prefill_branches\":{},\"owns_ratio4_replay_branch\":{},\"owns_compressor_update_pool_kernel\":{},\"owns_compressor_shift_ratio4_kernel\":{},\"owns_compressor_wrapper_orchestration\":{},\"owns_attention_kernels\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4C2_SCOPE.owns_compressor_prefill_pool_kernel,
        M14_4C2_SCOPE.owns_general_and_ratio4_prefill_branches,
        M14_4C2_SCOPE.owns_ratio4_replay_branch,
        M14_4C2_SCOPE.owns_compressor_update_pool_kernel,
        M14_4C2_SCOPE.owns_compressor_shift_ratio4_kernel,
        M14_4C2_SCOPE.owns_compressor_wrapper_orchestration,
        M14_4C2_SCOPE.owns_attention_kernels,
        M14_4C2_SCOPE.owns_runtime_graph_integration,
        M14_4C2_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prefill_pool_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    head_dim: u32,
    ratio: u32,
    n_comp: u32,
    replay: bool,
    ape_type: u32,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    state_kv: &DeviceBuffer<f32>,
    state_score: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    comp: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPoolShiftError> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let tokens = ratio * n_comp;
    let state_rows = coff * ratio;
    if ratio == 0
        || ape_type > 1
        || kv.len() < (tokens * width) as usize
        || sc.len() < (tokens * width) as usize
        || state_kv.len() < (state_rows * width) as usize
        || state_score.len() < (state_rows * width) as usize
        || comp.len() < (n_comp * head_dim) as usize
        || (ape_type == 0 && ape_f32.len() < (ratio * width) as usize)
        || (ape_type == 1 && ape_f16.len() < (ratio * width) as usize)
    {
        return Err(CompressorPoolShiftError::InvalidShape);
    }
    module
        .compressor_prefill_pool_kernel(
            stream,
            LaunchConfig {
                grid_dim: (head_dim.div_ceil(THREADS), n_comp, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            head_dim,
            ratio,
            POS0,
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
        .map_err(CompressorPoolShiftError::Driver)
}

fn update_pool_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    head_dim: u32,
    ratio: u32,
    state_kv: &DeviceBuffer<f32>,
    state_score: &DeviceBuffer<f32>,
    row: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPoolShiftError> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let state_rows = coff * ratio;
    if ratio == 0
        || state_kv.len() < (state_rows * width) as usize
        || state_score.len() < (state_rows * width) as usize
        || row.len() < head_dim as usize
    {
        return Err(CompressorPoolShiftError::InvalidShape);
    }
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
            state_kv,
            state_score,
            row,
        )
        .map_err(CompressorPoolShiftError::Driver)
}

fn shift_ratio4_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
) -> Result<(), CompressorPoolShiftError> {
    let count = (2 * RATIO4 * RATIO4_WIDTH) as usize;
    if state_kv.len() < count || state_score.len() < count {
        return Err(CompressorPoolShiftError::InvalidShape);
    }
    let half = RATIO4 * RATIO4_WIDTH;
    module
        .compressor_shift_ratio4_kernel(
            stream,
            LaunchConfig {
                grid_dim: (half.div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            RATIO4_WIDTH,
            state_kv,
            state_score,
        )
        .map_err(CompressorPoolShiftError::Driver)
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn expected_prefill_pool(
    head_dim: u32,
    ratio: u32,
    n_comp: u32,
    replay: bool,
    kv: &[f32],
    sc: &[f32],
    state_kv: &[f32],
    state_score: &[f32],
    ape: &[f32],
) -> Vec<f32> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let mut comp = vec![0.0_f32; (n_comp * head_dim) as usize];
    for compressed in 0..n_comp {
        for dimension in 0..head_dim {
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
                    let base = (compressed - 1) * ratio;
                    for row in 0..4 {
                        let token = base + row;
                        let phase = (POS0 + token) % ratio;
                        candidates.push((
                            kv[(token * width + dimension) as usize],
                            sc[(token * width + dimension) as usize]
                                + ape[(phase * width + dimension) as usize],
                        ));
                    }
                }
                let base = compressed * ratio;
                for row in 0..4 {
                    let token = base + row;
                    let phase = (POS0 + token) % ratio;
                    candidates.push((
                        kv[(token * width + head_dim + dimension) as usize],
                        sc[(token * width + head_dim + dimension) as usize]
                            + ape[(phase * width + head_dim + dimension) as usize],
                    ));
                }
            } else {
                let base = compressed * ratio;
                for row in 0..ratio {
                    let token = base + row;
                    let phase = (POS0 + token) % ratio;
                    candidates.push((
                        kv[(token * width + dimension) as usize],
                        sc[(token * width + dimension) as usize]
                            + ape[(phase * width + dimension) as usize],
                    ));
                }
            }
            comp[(compressed * head_dim + dimension) as usize] = softmax_pool(&candidates);
        }
    }
    comp
}

fn expected_update_pool(
    head_dim: u32,
    ratio: u32,
    state_kv: &[f32],
    state_score: &[f32],
) -> Vec<f32> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * head_dim;
    let mut row = vec![0.0_f32; head_dim as usize];
    for dimension in 0..head_dim {
        let mut candidates = Vec::new();
        if ratio == 4 {
            for state_row in 0..4 {
                candidates.push((
                    state_kv[(state_row * width + dimension) as usize],
                    state_score[(state_row * width + dimension) as usize],
                ));
            }
            for state_row in 0..4 {
                let index = ((ratio + state_row) * width + head_dim + dimension) as usize;
                candidates.push((state_kv[index], state_score[index]));
            }
        } else {
            for state_row in 0..ratio {
                candidates.push((
                    state_kv[(state_row * width + dimension) as usize],
                    state_score[(state_row * width + dimension) as usize],
                ));
            }
        }
        row[dimension as usize] = softmax_pool(&candidates);
    }
    row
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
    if denominator != 0.0 {
        accumulator / denominator
    } else {
        0.0
    }
}

fn expected_shift_ratio4(values: &[f32]) -> Vec<f32> {
    let mut result = values.to_vec();
    let half = (RATIO4 * RATIO4_WIDTH) as usize;
    for index in 0..half {
        result[index] = values[half + index];
        result[half + index] = values[half + index];
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
enum CompressorPoolShiftError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for CompressorPoolShiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("compressor pool/shift tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompressorPoolShiftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
