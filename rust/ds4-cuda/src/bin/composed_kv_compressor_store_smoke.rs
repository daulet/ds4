#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4C1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

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

    #[kernel]
    pub fn store_raw_kv_batch_kernel(
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
        kv: &[f32],
        mut raw: DisjointSlice<f32>,
    ) {
        let gid = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
            + thread::threadIdx_x() as u64;
        let count = n_tokens as u64 * head_dim as u64;
        if gid >= count {
            return;
        }
        let dimension = gid % head_dim as u64;
        let token = gid / head_dim as u64;
        let row = (pos0 as u64 + token) % raw_cap as u64;
        unsafe {
            *raw.get_unchecked_mut((row * head_dim as u64 + dimension) as usize) =
                (kv[gid as usize] as f16) as f32;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn compressor_store_kernel(
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
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
        let gid = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
            + thread::threadIdx_x() as u64;
        let count = n_tokens as u64 * width as u64;
        if gid >= count {
            return;
        }
        let token = gid / width as u64;
        let dimension = gid % width as u64;
        let phase = (pos0 as u64 + token) % ratio as u64;
        let row = if ratio == 4 {
            ratio as u64 + phase
        } else {
            phase
        };
        let ape_index = (phase * width as u64 + dimension) as usize;
        let ape = model_scalar(ape_type, ape_index, ape_f32, ape_f16);
        unsafe {
            *state_kv.get_unchecked_mut((row * width as u64 + dimension) as usize) =
                kv[gid as usize];
            *state_score.get_unchecked_mut((row * width as u64 + dimension) as usize) =
                sc[gid as usize] + ape;
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
        let gid = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
            + thread::threadIdx_x() as u64;
        let count = rows as u64 * width as u64;
        if gid >= count {
            return;
        }
        let row = gid / width as u64;
        let dimension = gid % width as u64;
        let src = src0 as u64 + row;
        let dst = dst0 as u64 + row;
        let phase = (pos0 as u64 + src) % ratio as u64;
        let ape_index = (phase * width as u64 + dimension) as usize;
        let ape = model_scalar(ape_type, ape_index, ape_f32, ape_f16);
        unsafe {
            *state_kv.get_unchecked_mut((dst * width as u64 + dimension) as usize) =
                kv[(src * width as u64 + dimension) as usize];
            *state_score.get_unchecked_mut((dst * width as u64 + dimension) as usize) =
                sc[(src * width as u64 + dimension) as usize] + ape;
        }
    }

    fn model_scalar(ape_type: u32, index: usize, ape_f32: &[f32], ape_f16: &[f16]) -> f32 {
        if ape_type == 1 {
            ape_f16[index] as f32
        } else {
            ape_f32[index]
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
const KV_HEAD_DIM: u32 = 75;
const KV_N_ROT: u32 = 6;
const RAW_CAP: u32 = 3;
const RAW_ROW: u32 = 2;
const COMP_HEAD_DIM: u32 = 5;
const COMP_RATIO: u32 = 4;
const COMP_WIDTH: u32 = 2 * COMP_HEAD_DIM;
const COMP_STATE_ROWS: u32 = 2 * COMP_RATIO;
const COMP_POS0: u32 = 3;
const COMP_TOKENS: u32 = 3;
const SET_SRC0: u32 = 1;
const SET_DST0: u32 = 0;
const SET_ROWS: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_composed_kv_compressor_store_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;

    let kv_values = kv_values();
    let expected_kv = expected_fp8_kv_quantize(&kv_values);
    let mut kv = substrate.upload(&kv_values)?;
    let raw_initial = vec![-99.0_f32; (RAW_CAP * KV_HEAD_DIM) as usize];
    let mut raw = substrate.upload(&raw_initial)?;
    kv_fp8_store_raw_tensor(&module, substrate.stream(), &mut kv, &mut raw)?;
    substrate.end_commands()?;
    let actual_kv = substrate.download(&kv)?;
    assert_close(&actual_kv, &expected_kv, 1.0e-5);
    assert_eq!(
        substrate.download(&raw)?,
        expected_raw_store(&raw_initial, &expected_kv)
    );
    assert_eq!(
        &actual_kv[(KV_HEAD_DIM - KV_N_ROT) as usize..],
        &kv_values[(KV_HEAD_DIM - KV_N_ROT) as usize..]
    );

    let compressor_kv_values = compressor_values(17);
    let compressor_score_values = compressor_values(31);
    let ape_f32_values = ape_f32_values();
    let ape_f16_values = ape_f16_values();
    let kv_comp = substrate.upload(&compressor_kv_values)?;
    let score_comp = substrate.upload(&compressor_score_values)?;
    let ape_f32 = substrate.upload(&ape_f32_values)?;
    let ape_f16 = substrate.upload(&ape_f16_values)?;
    let unused_f32 = substrate.upload(&vec![0.0_f32; (COMP_RATIO * COMP_WIDTH) as usize])?;
    let unused_f16 =
        substrate.upload(&vec![f16::from_bits(0); (COMP_RATIO * COMP_WIDTH) as usize])?;

    let state_initial = vec![-77.0_f32; (COMP_STATE_ROWS * COMP_WIDTH) as usize];
    let mut state_kv_f32 = substrate.upload(&state_initial)?;
    let mut state_score_f32 = substrate.upload(&state_initial)?;
    compressor_store_tensor(
        &module,
        substrate.stream(),
        &kv_comp,
        &score_comp,
        &ape_f32,
        &unused_f16,
        0,
        &mut state_kv_f32,
        &mut state_score_f32,
    )?;
    substrate.end_commands()?;
    let (expected_state_kv, expected_state_score) = expected_compressor_store(
        &state_initial,
        &compressor_kv_values,
        &compressor_score_values,
        &ape_f32_values,
    );
    assert_eq!(substrate.download(&state_kv_f32)?, expected_state_kv);
    assert_close(
        &substrate.download(&state_score_f32)?,
        &expected_state_score,
        1.0e-6,
    );

    let mut state_kv_f16 = substrate.upload(&state_initial)?;
    let mut state_score_f16 = substrate.upload(&state_initial)?;
    compressor_store_tensor(
        &module,
        substrate.stream(),
        &kv_comp,
        &score_comp,
        &unused_f32,
        &ape_f16,
        1,
        &mut state_kv_f16,
        &mut state_score_f16,
    )?;
    substrate.end_commands()?;
    let ape_f16_as_f32 = ape_f16_values
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let (expected_state_kv, expected_state_score) = expected_compressor_store(
        &state_initial,
        &compressor_kv_values,
        &compressor_score_values,
        &ape_f16_as_f32,
    );
    assert_eq!(substrate.download(&state_kv_f16)?, expected_state_kv);
    assert_close(
        &substrate.download(&state_score_f16)?,
        &expected_state_score,
        1.0e-6,
    );

    let mut set_kv_f32 = substrate.upload(&state_initial)?;
    let mut set_score_f32 = substrate.upload(&state_initial)?;
    compressor_set_rows_tensor(
        &module,
        substrate.stream(),
        &kv_comp,
        &score_comp,
        &ape_f32,
        &unused_f16,
        0,
        &mut set_kv_f32,
        &mut set_score_f32,
    )?;
    substrate.end_commands()?;
    let (expected_set_kv, expected_set_score) = expected_compressor_set_rows(
        &state_initial,
        &compressor_kv_values,
        &compressor_score_values,
        &ape_f32_values,
    );
    assert_eq!(substrate.download(&set_kv_f32)?, expected_set_kv);
    assert_close(
        &substrate.download(&set_score_f32)?,
        &expected_set_score,
        1.0e-6,
    );

    let mut set_kv_f16 = substrate.upload(&state_initial)?;
    let mut set_score_f16 = substrate.upload(&state_initial)?;
    compressor_set_rows_tensor(
        &module,
        substrate.stream(),
        &kv_comp,
        &score_comp,
        &unused_f32,
        &ape_f16,
        1,
        &mut set_kv_f16,
        &mut set_score_f16,
    )?;
    substrate.end_commands()?;
    let (expected_set_kv, expected_set_score) = expected_compressor_set_rows(
        &state_initial,
        &compressor_kv_values,
        &compressor_score_values,
        &ape_f16_as_f32,
    );
    assert_eq!(substrate.download(&set_kv_f16)?, expected_set_kv);
    assert_close(
        &substrate.download(&set_score_f16)?,
        &expected_set_score,
        1.0e-6,
    );

    let mut short_raw = substrate.zeroed::<f32>((RAW_CAP * KV_HEAD_DIM - 1) as usize)?;
    assert!(matches!(
        kv_fp8_store_raw_tensor(&module, substrate.stream(), &mut kv, &mut short_raw),
        Err(ComposedKvCompressorError::InvalidShape)
    ));
    let mut short_state = substrate.zeroed::<f32>((COMP_STATE_ROWS * COMP_WIDTH - 1) as usize)?;
    let mut valid_state = substrate.zeroed::<f32>((COMP_STATE_ROWS * COMP_WIDTH) as usize)?;
    assert!(matches!(
        compressor_store_tensor(
            &module,
            substrate.stream(),
            &kv_comp,
            &score_comp,
            &ape_f32,
            &unused_f16,
            0,
            &mut short_state,
            &mut valid_state,
        ),
        Err(ComposedKvCompressorError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4c1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"composed_fp8_raw_store_output_matches\":true,\"composed_rope_tail_preserved\":true,\"compressor_store_ratio4_output_matches\":true,\"compressor_set_rows_output_matches\":true,\"f32_ape_output_matches\":true,\"f16_ape_output_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_kv_fp8_store_raw_composition\":{},\"owns_compressor_store_kernel\":{},\"owns_compressor_set_rows_kernel\":{},\"owns_f32_and_f16_ape_reads\":{},\"owns_compressor_pooling_or_shift\":{},\"owns_attention_kernels\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4C1_SCOPE.owns_kv_fp8_store_raw_composition,
        M14_4C1_SCOPE.owns_compressor_store_kernel,
        M14_4C1_SCOPE.owns_compressor_set_rows_kernel,
        M14_4C1_SCOPE.owns_f32_and_f16_ape_reads,
        M14_4C1_SCOPE.owns_compressor_pooling_or_shift,
        M14_4C1_SCOPE.owns_attention_kernels,
        M14_4C1_SCOPE.owns_runtime_graph_integration,
        M14_4C1_SCOPE.changes_default_route,
    );
    Ok(())
}

fn kv_fp8_store_raw_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kv: &mut DeviceBuffer<f32>,
    raw: &mut DeviceBuffer<f32>,
) -> Result<(), ComposedKvCompressorError> {
    if kv.len() < KV_HEAD_DIM as usize
        || raw.len() < (RAW_CAP * KV_HEAD_DIM) as usize
        || KV_N_ROT > KV_HEAD_DIM
    {
        return Err(ComposedKvCompressorError::InvalidShape);
    }
    module
        .fp8_kv_quantize_kernel(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            1,
            KV_HEAD_DIM,
            KV_N_ROT,
            kv,
        )
        .map_err(ComposedKvCompressorError::Driver)?;
    module
        .store_raw_kv_batch_kernel(
            stream,
            LaunchConfig {
                grid_dim: (KV_HEAD_DIM.div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            RAW_CAP,
            RAW_ROW,
            1,
            KV_HEAD_DIM,
            kv,
            raw,
        )
        .map_err(ComposedKvCompressorError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn compressor_store_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    ape_type: u32,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
) -> Result<(), ComposedKvCompressorError> {
    let input = (COMP_TOKENS * COMP_WIDTH) as usize;
    let state = (COMP_STATE_ROWS * COMP_WIDTH) as usize;
    let ape = (COMP_RATIO * COMP_WIDTH) as usize;
    if ape_type > 1
        || kv.len() < input
        || sc.len() < input
        || state_kv.len() < state
        || state_score.len() < state
        || (ape_type == 0 && ape_f32.len() < ape)
        || (ape_type == 1 && ape_f16.len() < ape)
    {
        return Err(ComposedKvCompressorError::InvalidShape);
    }
    module
        .compressor_store_kernel(
            stream,
            LaunchConfig {
                grid_dim: ((COMP_TOKENS * COMP_WIDTH).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            COMP_HEAD_DIM,
            COMP_RATIO,
            COMP_POS0,
            COMP_TOKENS,
            ape_type,
            kv,
            sc,
            ape_f32,
            ape_f16,
            state_kv,
            state_score,
        )
        .map_err(ComposedKvCompressorError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn compressor_set_rows_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kv: &DeviceBuffer<f32>,
    sc: &DeviceBuffer<f32>,
    ape_f32: &DeviceBuffer<f32>,
    ape_f16: &DeviceBuffer<f16>,
    ape_type: u32,
    state_kv: &mut DeviceBuffer<f32>,
    state_score: &mut DeviceBuffer<f32>,
) -> Result<(), ComposedKvCompressorError> {
    let input = ((SET_SRC0 + SET_ROWS) * COMP_WIDTH) as usize;
    let state = (COMP_STATE_ROWS * COMP_WIDTH) as usize;
    let ape = (COMP_RATIO * COMP_WIDTH) as usize;
    if ape_type > 1
        || kv.len() < input
        || sc.len() < input
        || state_kv.len() < state
        || state_score.len() < state
        || (ape_type == 0 && ape_f32.len() < ape)
        || (ape_type == 1 && ape_f16.len() < ape)
    {
        return Err(ComposedKvCompressorError::InvalidShape);
    }
    module
        .compressor_set_rows_kernel(
            stream,
            LaunchConfig {
                grid_dim: ((SET_ROWS * COMP_WIDTH).div_ceil(THREADS), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            COMP_WIDTH,
            COMP_RATIO,
            COMP_POS0,
            SET_SRC0,
            SET_DST0,
            SET_ROWS,
            ape_type,
            kv,
            sc,
            ape_f32,
            ape_f16,
            state_kv,
            state_score,
        )
        .map_err(ComposedKvCompressorError::Driver)
}

fn kv_values() -> Vec<f32> {
    (0..KV_HEAD_DIM)
        .map(|index| ((index * 29 + 11) % 151) as f32 * 0.09375 - 6.75)
        .collect()
}

fn expected_fp8_kv_quantize(values: &[f32]) -> Vec<f32> {
    let mut result = values.to_vec();
    let prefix = (KV_HEAD_DIM - KV_N_ROT) as usize;
    for chunk in result[..prefix].chunks_mut(64) {
        let amax = chunk
            .iter()
            .map(|value| value.abs())
            .fold(1.0e-4_f32, f32::max);
        let scale = 2.0_f32.powf((amax / 448.0).log2().ceil());
        for value in chunk {
            *value = e4m3fn_dequant_host((*value / scale).clamp(-448.0, 448.0)) * scale;
        }
    }
    result
}

fn expected_raw_store(initial: &[f32], kv: &[f32]) -> Vec<f32> {
    let mut result = initial.to_vec();
    for dimension in 0..KV_HEAD_DIM as usize {
        result[RAW_ROW as usize * KV_HEAD_DIM as usize + dimension] = (kv[dimension] as f16) as f32;
    }
    result
}

fn compressor_values(seed: u32) -> Vec<f32> {
    (0..COMP_TOKENS * COMP_WIDTH)
        .map(|index| ((index * seed + 9) % 73) as f32 * 0.0625 - 1.25)
        .collect()
}

fn ape_f32_values() -> Vec<f32> {
    (0..COMP_RATIO * COMP_WIDTH)
        .map(|index| ((index * 7 + 3) % 31) as f32 * 0.03125 - 0.375)
        .collect()
}

fn ape_f16_values() -> Vec<f16> {
    ape_f32_values()
        .iter()
        .enumerate()
        .map(|(index, value)| (*value + (index % 3) as f32 * 0.015625) as f16)
        .collect()
}

fn expected_compressor_store(
    initial: &[f32],
    kv: &[f32],
    sc: &[f32],
    ape: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut state_kv = initial.to_vec();
    let mut state_score = initial.to_vec();
    for token in 0..COMP_TOKENS as usize {
        let phase = (COMP_POS0 as usize + token) % COMP_RATIO as usize;
        let row = COMP_RATIO as usize + phase;
        for dimension in 0..COMP_WIDTH as usize {
            let input = token * COMP_WIDTH as usize + dimension;
            let output = row * COMP_WIDTH as usize + dimension;
            state_kv[output] = kv[input];
            state_score[output] = sc[input] + ape[phase * COMP_WIDTH as usize + dimension];
        }
    }
    (state_kv, state_score)
}

fn expected_compressor_set_rows(
    initial: &[f32],
    kv: &[f32],
    sc: &[f32],
    ape: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut state_kv = initial.to_vec();
    let mut state_score = initial.to_vec();
    for row in 0..SET_ROWS as usize {
        let src = SET_SRC0 as usize + row;
        let dst = SET_DST0 as usize + row;
        let phase = (COMP_POS0 as usize + src) % COMP_RATIO as usize;
        for dimension in 0..COMP_WIDTH as usize {
            let input = src * COMP_WIDTH as usize + dimension;
            let output = dst * COMP_WIDTH as usize + dimension;
            state_kv[output] = kv[input];
            state_score[output] = sc[input] + ape[phase * COMP_WIDTH as usize + dimension];
        }
    }
    (state_kv, state_score)
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
enum ComposedKvCompressorError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for ComposedKvCompressorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("composed KV/compressor tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ComposedKvCompressorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
