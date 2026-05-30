use std::{cmp::Reverse, fmt};

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, DynamicSharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D2C3_SCOPE};

const SORT_N: usize = 8192;
const ITEMS_PER_THREAD: u32 = 16;
const BLOCK_THREADS: u32 = 512;
const TOP_K: u32 = 512;
const SHARED_KEY_BYTES: u32 = (SORT_N * std::mem::size_of::<u64>()) as u32;
const EMPTY_KEY: u64 = 0x007f_ffff_u64 << 32;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_topk_8192_packed_key_equivalent_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        let keys = DynamicSharedArray::<u64>::get();
        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= BLOCK_THREADS {
            return;
        }
        let mut item = 0_u32;
        while item < ITEMS_PER_THREAD {
            let i = tid * ITEMS_PER_THREAD + item;
            let key = if i < n_comp {
                let value = scores[token as usize * n_comp as usize + i as usize];
                let bits = value.to_bits();
                let ordered = if (bits & 0x8000_0000) != 0 {
                    !bits
                } else {
                    bits ^ 0x8000_0000
                };
                (ordered as u64) << 32 | (u32::MAX - i) as u64
            } else {
                EMPTY_KEY
            };
            unsafe {
                *keys.add(i as usize) = key;
            }
            item += 1;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                let mut i = tid;
                while i < SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_N as u32 {
                        let left = unsafe { *keys.add(i as usize) };
                        let right = unsafe { *keys.add(other as usize) };
                        let descending = (i & k) == 0;
                        let swap = if descending {
                            right > left
                        } else {
                            left > right
                        };
                        if swap {
                            unsafe {
                                *keys.add(i as usize) = right;
                                *keys.add(other as usize) = left;
                            }
                        }
                    }
                    i += BLOCK_THREADS;
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        if tid < top_k {
            let key = unsafe { *keys.add(tid as usize) };
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + tid as usize) =
                    u32::MAX - key as u32;
            }
        }
    }

    pub fn opt_in_large_dynamic_shared_memory(module: &LoadedModule) -> Result<(), DriverError> {
        module
            .__indexer_topk_8192_packed_key_equivalent_kernel_function
            .set_max_dynamic_shared_memory_size(SHARED_KEY_BYTES as i32)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    kernels::opt_in_large_dynamic_shared_memory(&module)?;
    assert_eq!(EMPTY_KEY, packed_key(f32::NEG_INFINITY, u32::MAX));

    let exact_4096 = build_scores(4096);
    let exact_4096_device = substrate.upload(&exact_4096)?;
    let mut exact_4096_selected = substrate.zeroed::<u32>(TOP_K as usize)?;
    indexer_topk_packed(
        &module,
        substrate.stream(),
        &mut exact_4096_selected,
        &exact_4096_device,
        4096,
        1,
        TOP_K,
    )?;
    substrate.flush_commands()?;
    let exact_4096_output = substrate.download(&exact_4096_selected)?;
    assert_eq!(exact_4096_output, expected_topk(&exact_4096, TOP_K));
    assert_eq!(&exact_4096_output[0..4], &[0, 1, 2, 3]);

    let wide_scores = build_scores(6000);
    let wide_scores_device = substrate.upload(&wide_scores)?;
    let mut wide_selected = substrate.zeroed::<u32>(TOP_K as usize)?;
    indexer_topk_packed(
        &module,
        substrate.stream(),
        &mut wide_selected,
        &wide_scores_device,
        6000,
        1,
        TOP_K,
    )?;
    substrate.end_commands()?;
    let wide_output = substrate.download(&wide_selected)?;
    assert_eq!(wide_output, expected_topk(&wide_scores, TOP_K));
    assert!(wide_output.iter().all(|&index| index < 6000));

    assert!(matches!(
        indexer_topk_packed(
            &module,
            substrate.stream(),
            &mut wide_selected,
            &wide_scores_device,
            6000,
            1,
            TOP_K - 1,
        ),
        Err(IndexerTopkPackedError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2d2c3\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"packed_key_4096_output_matches\":true,\"packed_key_8192_range_output_matches\":true,\"ordered_float_and_index_key_matches\":true,\"dynamic_shared_launch_matches\":true,\"sentinel_output_excluded\":true,\"invalid_shape_rejected\":true,\"owns_indexer_topk_8192_packed_key_equivalent_kernel\":{},\"owns_dynamic_shared_launch_shape\":{},\"owns_cub_library_implementation\":{},\"owns_topk_dispatch_policy\":{},\"owns_chunked_topk_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2C3_SCOPE.owns_indexer_topk_8192_packed_key_equivalent_kernel,
        M14_2D2C3_SCOPE.owns_dynamic_shared_launch_shape,
        M14_2D2C3_SCOPE.owns_cub_library_implementation,
        M14_2D2C3_SCOPE.owns_topk_dispatch_policy,
        M14_2D2C3_SCOPE.owns_chunked_topk_dispatch,
        M14_2D2C3_SCOPE.changes_default_route,
    );
    Ok(())
}

fn build_scores(n_comp: u32) -> Vec<f32> {
    (0..n_comp)
        .map(|component| match component {
            0 => f32::from_bits(0x7fc0_0001),
            1 => f32::INFINITY,
            2 | 3 => 16384.0_f32,
            _ => (component % 257) as f32 - 64.0,
        })
        .collect()
}

fn ordered_float_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if (bits & 0x8000_0000) != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

fn packed_key(value: f32, index: u32) -> u64 {
    (ordered_float_key(value) as u64) << 32 | (u32::MAX - index) as u64
}

fn expected_topk(scores: &[f32], top_k: u32) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..scores.len() as u32).collect();
    indices.sort_unstable_by_key(|&index| Reverse(packed_key(scores[index as usize], index)));
    indices.truncate(top_k as usize);
    indices
}

fn indexer_topk_packed(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerTopkPackedError> {
    if n_comp == 0
        || n_comp > SORT_N as u32
        || n_tokens == 0
        || top_k != TOP_K
        || top_k > n_comp
        || u64::from(n_tokens) * u64::from(n_comp) > scores.len() as u64
        || u64::from(n_tokens) * u64::from(top_k) > selected.len() as u64
    {
        return Err(IndexerTopkPackedError::InvalidShape);
    }
    module
        .indexer_topk_8192_packed_key_equivalent_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: SHARED_KEY_BYTES,
            },
            n_comp,
            n_tokens,
            top_k,
            scores,
            selected,
        )
        .map_err(IndexerTopkPackedError::Driver)
}

#[derive(Debug)]
enum IndexerTopkPackedError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerTopkPackedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("packed-key indexer top-k shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerTopkPackedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
