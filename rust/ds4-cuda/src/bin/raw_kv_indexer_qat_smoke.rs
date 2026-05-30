#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4B_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

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

    #[kernel]
    pub fn indexer_hadamard_fp4_kernel(n_rows: u32, head_dim: u32, mut x: DisjointSlice<f32>) {
        static mut VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut MAGNITUDES: SharedArray<f32, 128> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x() as usize;
        if row >= n_rows || head_dim != 128 || tid >= 128 {
            return;
        }
        let base = row as usize * head_dim as usize;
        unsafe {
            VALUES[tid] = *x.as_mut_ptr().add(base + tid);
        }
        thread::sync_threads();

        let mut stride = 1_usize;
        while stride < 128 {
            if tid & stride == 0 {
                let pair = (tid & !(2 * stride - 1)) + (tid & (stride - 1));
                let a = unsafe { VALUES[pair] };
                let b = unsafe { VALUES[pair + stride] };
                unsafe {
                    VALUES[pair] = a + b;
                    VALUES[pair + stride] = a - b;
                }
            }
            thread::sync_threads();
            stride <<= 1;
        }

        let value = unsafe { VALUES[tid] } * 0.08838834764831845;
        let block_base = (tid >> 5) * 32;
        let lane = tid & 31;
        unsafe {
            MAGNITUDES[tid] = absolute(value);
        }
        thread::sync_threads();
        stride = 16;
        while stride > 0 {
            if lane < stride {
                let other = unsafe { MAGNITUDES[block_base + lane + stride] };
                if other > unsafe { MAGNITUDES[block_base + lane] } {
                    unsafe {
                        MAGNITUDES[block_base + lane] = other;
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let amax = if unsafe { MAGNITUDES[block_base] } > 7.052966104933725e-38 {
            unsafe { MAGNITUDES[block_base] }
        } else {
            7.052966104933725e-38
        };
        let scale = 2.0_f32.powf((amax / 6.0).log2().ceil());
        let mut scaled = value / scale;
        if scaled > 6.0 {
            scaled = 6.0;
        } else if scaled < -6.0 {
            scaled = -6.0;
        }
        unsafe {
            *x.get_unchecked_mut(base + tid) = e2m1fn_dequant(scaled) * scale;
        }
    }

    fn absolute(value: f32) -> f32 {
        if value < 0.0 {
            -value
        } else {
            value
        }
    }

    fn e2m1fn_value(value: i32) -> f32 {
        match value & 7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        }
    }

    fn e2m1fn_dequant(value: f32) -> f32 {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let mut magnitude = absolute(value);
        if magnitude > 6.0 {
            magnitude = 6.0;
        }
        let mut best = 0_i32;
        let mut best_diff = magnitude;
        let mut candidate = 1_i32;
        while candidate < 8 {
            let diff = absolute(magnitude - e2m1fn_value(candidate));
            if diff < best_diff || (diff == best_diff && candidate & 1 == 0 && best & 1 != 0) {
                best = candidate;
                best_diff = diff;
            }
            candidate += 1;
        }
        sign * e2m1fn_value(best)
    }
}

const THREADS: u32 = 256;
const RAW_CAP: u32 = 3;
const POS0: u32 = 2;
const N_TOKENS: u32 = 2;
const RAW_HEAD_DIM: u32 = 7;
const INDEX_ROWS: u32 = 2;
const INDEX_HEAD_DIM: u32 = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_raw_kv_indexer_qat_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;

    let kv_values = raw_kv_values();
    let kv = substrate.upload(&kv_values)?;
    let initial_raw = vec![-99.0_f32; (RAW_CAP * RAW_HEAD_DIM) as usize];
    let mut raw = substrate.upload(&initial_raw)?;
    store_raw_kv_batch_tensor(&module, substrate.stream(), &mut raw, &kv)?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&raw)?,
        expected_raw_kv_store(&initial_raw, &kv_values)
    );

    let index_values = indexer_values();
    let mut indexer = substrate.upload(&index_values)?;
    indexer_qat_tensor(&module, substrate.stream(), &mut indexer)?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&indexer)?,
        &expected_indexer_qat(&index_values),
        1.0e-5,
    );

    let mut short_raw = substrate.zeroed::<f32>((RAW_CAP * RAW_HEAD_DIM - 1) as usize)?;
    assert!(matches!(
        store_raw_kv_batch_tensor(&module, substrate.stream(), &mut short_raw, &kv),
        Err(RawKvIndexerError::InvalidShape)
    ));
    let mut short_indexer = substrate.zeroed::<f32>((INDEX_ROWS * INDEX_HEAD_DIM - 1) as usize)?;
    assert!(matches!(
        indexer_qat_tensor(&module, substrate.stream(), &mut short_indexer),
        Err(RawKvIndexerError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4b\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"raw_kv_fp16_roundtrip_matches\":true,\"raw_kv_ring_wrap_matches\":true,\"indexer_hadamard_fp4_output_matches\":true,\"fp4_block_scale_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_store_raw_kv_batch_kernel\":{},\"owns_raw_kv_store_surfaces\":{},\"owns_indexer_hadamard_fp4_kernel\":{},\"owns_indexer_qat_surface\":{},\"owns_kv_fp8_store_raw_composition\":{},\"owns_compressor_kernels\":{},\"owns_attention_kernels\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4B_SCOPE.owns_store_raw_kv_batch_kernel,
        M14_4B_SCOPE.owns_raw_kv_store_surfaces,
        M14_4B_SCOPE.owns_indexer_hadamard_fp4_kernel,
        M14_4B_SCOPE.owns_indexer_qat_surface,
        M14_4B_SCOPE.owns_kv_fp8_store_raw_composition,
        M14_4B_SCOPE.owns_compressor_kernels,
        M14_4B_SCOPE.owns_attention_kernels,
        M14_4B_SCOPE.owns_runtime_graph_integration,
        M14_4B_SCOPE.changes_default_route,
    );
    Ok(())
}

fn store_raw_kv_batch_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    raw: &mut DeviceBuffer<f32>,
    kv: &DeviceBuffer<f32>,
) -> Result<(), RawKvIndexerError> {
    let count = u64::from(N_TOKENS) * u64::from(RAW_HEAD_DIM);
    if RAW_CAP == 0 || raw.len() < (RAW_CAP * RAW_HEAD_DIM) as usize || kv.len() < count as usize {
        return Err(RawKvIndexerError::InvalidShape);
    }
    module
        .store_raw_kv_batch_kernel(
            stream,
            LaunchConfig {
                grid_dim: (
                    u32::try_from(count.div_ceil(u64::from(THREADS))).unwrap(),
                    1,
                    1,
                ),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            RAW_CAP,
            POS0,
            N_TOKENS,
            RAW_HEAD_DIM,
            kv,
            raw,
        )
        .map_err(RawKvIndexerError::Driver)
}

fn indexer_qat_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
) -> Result<(), RawKvIndexerError> {
    if x.len() < (INDEX_ROWS * INDEX_HEAD_DIM) as usize {
        return Err(RawKvIndexerError::InvalidShape);
    }
    module
        .indexer_hadamard_fp4_kernel(
            stream,
            LaunchConfig {
                grid_dim: (INDEX_ROWS, 1, 1),
                block_dim: (INDEX_HEAD_DIM, 1, 1),
                shared_mem_bytes: 0,
            },
            INDEX_ROWS,
            INDEX_HEAD_DIM,
            x,
        )
        .map_err(RawKvIndexerError::Driver)
}

fn raw_kv_values() -> Vec<f32> {
    (0..N_TOKENS * RAW_HEAD_DIM)
        .map(|index| ((index * 13 + 7) % 37) as f32 * 0.0625 - 1.0)
        .collect()
}

fn expected_raw_kv_store(initial: &[f32], kv: &[f32]) -> Vec<f32> {
    let mut result = initial.to_vec();
    for token in 0..N_TOKENS as usize {
        let row = (POS0 as usize + token) % RAW_CAP as usize;
        for dimension in 0..RAW_HEAD_DIM as usize {
            result[row * RAW_HEAD_DIM as usize + dimension] =
                (kv[token * RAW_HEAD_DIM as usize + dimension] as f16) as f32;
        }
    }
    result
}

fn indexer_values() -> Vec<f32> {
    (0..INDEX_ROWS * INDEX_HEAD_DIM)
        .map(|index| ((index * 19 + 3) % 101) as f32 * 0.03125 - 1.5)
        .collect()
}

fn expected_indexer_qat(values: &[f32]) -> Vec<f32> {
    let mut result = values.to_vec();
    for row in result.chunks_exact_mut(INDEX_HEAD_DIM as usize) {
        let mut stride = 1_usize;
        while stride < INDEX_HEAD_DIM as usize {
            let mut base = 0_usize;
            while base < INDEX_HEAD_DIM as usize {
                for lane in 0..stride {
                    let a = row[base + lane];
                    let b = row[base + stride + lane];
                    row[base + lane] = a + b;
                    row[base + stride + lane] = a - b;
                }
                base += 2 * stride;
            }
            stride <<= 1;
        }
        for value in row.iter_mut() {
            *value *= 0.08838834764831845;
        }
        for chunk in row.chunks_mut(32) {
            let amax = chunk
                .iter()
                .map(|value| value.abs())
                .fold(7.052966104933725e-38_f32, f32::max);
            let scale = 2.0_f32.powf((amax / 6.0).log2().ceil());
            for value in chunk {
                *value = e2m1fn_dequant_host((*value / scale).clamp(-6.0, 6.0)) * scale;
            }
        }
    }
    result
}

fn e2m1fn_value_host(value: i32) -> f32 {
    [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(value & 7) as usize]
}

fn e2m1fn_dequant_host(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let magnitude = value.abs().min(6.0);
    let mut best = 0_i32;
    let mut best_diff = magnitude;
    for candidate in 1..8_i32 {
        let diff = (magnitude - e2m1fn_value_host(candidate)).abs();
        if diff < best_diff || (diff == best_diff && candidate & 1 == 0 && best & 1 != 0) {
            best = candidate;
            best_diff = diff;
        }
    }
    sign * e2m1fn_value_host(best)
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
enum RawKvIndexerError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for RawKvIndexerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("raw KV/indexer QAT tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RawKvIndexerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
