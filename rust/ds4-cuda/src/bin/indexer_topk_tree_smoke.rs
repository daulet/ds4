use std::{cmp::Ordering, fmt};

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D2C4_SCOPE};

const SORT_N: usize = 4096;
const TOP_K: u32 = 512;
const BLOCK_THREADS: u32 = 1024;
const MERGE_GROUP: u32 = 8;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_topk_chunk_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        candidate_stride: u32,
        scores: &[f32],
        mut scratch: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let chunk_start = chunk * SORT_N as u32;
        if chunk_start >= n_comp {
            return;
        }
        let remaining = n_comp - chunk_start;
        let chunk_n = if remaining < SORT_N as u32 {
            remaining
        } else {
            SORT_N as u32
        };
        let mut i = tid;
        while i < SORT_N as u32 {
            let index = i as usize;
            if i < chunk_n {
                unsafe {
                    VALUES[index] =
                        scores[token as usize * n_comp as usize + (chunk_start + i) as usize];
                    INDICES[index] = chunk_start + i;
                }
            } else {
                unsafe {
                    VALUES[index] = f32::NEG_INFINITY;
                    INDICES[index] = u32::MAX;
                }
            }
            i += BLOCK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
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

        i = tid;
        while i < top_k {
            let out = token as usize * candidate_stride as usize
                + chunk as usize * top_k as usize
                + i as usize;
            unsafe {
                *scratch.get_unchecked_mut(out) = INDICES[i as usize];
            }
            i += BLOCK_THREADS;
        }
    }

    #[kernel]
    pub fn indexer_topk_tree_merge_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        n_sets: u32,
        merge_group: u32,
        candidate_offset: u32,
        candidate_stride: u32,
        out_offset: u32,
        out_stride: u32,
        scores: &[f32],
        mut scratch: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let group = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let set0 = group * merge_group;
        if set0 >= n_sets {
            return;
        }
        let remaining = n_sets - set0;
        let set_count = if remaining < merge_group {
            remaining
        } else {
            merge_group
        };
        let candidate_count = set_count * top_k;
        let mut i = tid;
        while i < SORT_N as u32 {
            let mut index = u32::MAX;
            let mut value = f32::NEG_INFINITY;
            if i < candidate_count {
                let source = candidate_offset as usize
                    + token as usize * candidate_stride as usize
                    + set0 as usize * top_k as usize
                    + i as usize;
                index = unsafe { *scratch.get_unchecked_mut(source) };
                if index < n_comp {
                    value = scores[token as usize * n_comp as usize + index as usize];
                }
            }
            unsafe {
                VALUES[i as usize] = value;
                INDICES[i as usize] = index;
            }
            i += BLOCK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
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

        i = tid;
        while i < top_k {
            let out = out_offset as usize
                + token as usize * out_stride as usize
                + group as usize * top_k as usize
                + i as usize;
            unsafe {
                *scratch.get_unchecked_mut(out) = INDICES[i as usize];
            }
            i += BLOCK_THREADS;
        }
    }

    #[kernel]
    pub fn indexer_topk_merge_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        candidate_offset: u32,
        candidate_count: u32,
        candidate_stride: u32,
        candidates: &[u32],
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < SORT_N as u32 {
            let mut index = u32::MAX;
            let mut value = f32::NEG_INFINITY;
            if i < candidate_count {
                let source = candidate_offset as usize
                    + token as usize * candidate_stride as usize
                    + i as usize;
                index = candidates[source];
                if index < n_comp {
                    value = scores[token as usize * n_comp as usize + index as usize];
                }
            }
            unsafe {
                VALUES[i as usize] = value;
                INDICES[i as usize] = index;
            }
            i += BLOCK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < SORT_N as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_N as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] };
                        let bi = unsafe { INDICES[other_index] };
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai;
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

        i = tid;
        while i < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + i as usize) =
                    INDICES[i as usize];
            }
            i += BLOCK_THREADS;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScratchLevel {
    offset: u32,
    n_sets: u32,
    stride: u32,
}

#[derive(Debug, Eq, PartialEq)]
struct ScratchPlan {
    levels: Vec<ScratchLevel>,
    total_elements: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let n_comp = SORT_N as u32 * 9 + 73;
    let n_tokens = 2;
    let plan = scratch_plan(n_comp, n_tokens, TOP_K)?;
    assert_eq!(
        plan,
        ScratchPlan {
            levels: vec![
                ScratchLevel {
                    offset: 0,
                    n_sets: 10,
                    stride: 5120,
                },
                ScratchLevel {
                    offset: 10240,
                    n_sets: 2,
                    stride: 1024,
                },
            ],
            total_elements: 12288,
        }
    );

    let score_values = build_scores(n_tokens, n_comp);
    let scores = substrate.upload(&score_values)?;
    let mut scratch = substrate.zeroed::<u32>(plan.total_elements as usize)?;
    let mut selected = substrate.zeroed::<u32>((n_tokens * TOP_K) as usize)?;
    indexer_topk_tree(
        &module,
        substrate.stream(),
        &mut selected,
        &mut scratch,
        &scores,
        n_comp,
        n_tokens,
        TOP_K,
        &plan,
    )?;
    substrate.end_commands()?;
    let output = substrate.download(&selected)?;
    for token in 0..n_tokens {
        let row = &score_values[(token * n_comp) as usize..((token + 1) * n_comp) as usize];
        let chosen = &output[(token * TOP_K) as usize..((token + 1) * TOP_K) as usize];
        assert_eq!(chosen, expected_topk(row, TOP_K));
        assert_eq!(&chosen[0..2], &[0, SORT_N as u32]);
        assert!(chosen.iter().all(|&index| index < n_comp));
    }

    let invalid = scratch_plan(TOP_K - 1, n_tokens, TOP_K);
    assert!(matches!(invalid, Err(IndexerTopkTreeError::InvalidShape)));

    println!(
        "{{\"milestone\":\"M14.2d2c4\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"chunk_output_matches\":true,\"tree_merge_output_matches\":true,\"final_merge_output_matches\":true,\"scratch_layout_matches\":true,\"multi_token_stride_matches\":true,\"partial_chunk_sentinel_excluded\":true,\"invalid_shape_rejected\":true,\"owns_indexer_topk_chunk_pow2_4096_kernel\":{},\"owns_indexer_topk_tree_merge_pow2_4096_kernel\":{},\"owns_indexer_topk_merge_pow2_4096_kernel\":{},\"owns_scratch_layout\":{},\"owns_topk_dispatch_policy\":{},\"owns_indexed_topk_sort_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2C4_SCOPE.owns_indexer_topk_chunk_pow2_4096_kernel,
        M14_2D2C4_SCOPE.owns_indexer_topk_tree_merge_pow2_4096_kernel,
        M14_2D2C4_SCOPE.owns_indexer_topk_merge_pow2_4096_kernel,
        M14_2D2C4_SCOPE.owns_scratch_layout,
        M14_2D2C4_SCOPE.owns_topk_dispatch_policy,
        M14_2D2C4_SCOPE.owns_indexed_topk_sort_dispatch,
        M14_2D2C4_SCOPE.changes_default_route,
    );
    Ok(())
}

fn scratch_plan(
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<ScratchPlan, IndexerTopkTreeError> {
    if n_comp <= SORT_N as u32 || n_tokens == 0 || top_k != TOP_K || top_k > n_comp {
        return Err(IndexerTopkTreeError::InvalidShape);
    }
    let n_chunks = n_comp.div_ceil(SORT_N as u32);
    let mut n_sets = n_chunks;
    let mut stride = n_sets
        .checked_mul(top_k)
        .ok_or(IndexerTopkTreeError::InvalidShape)?;
    let mut total_elements = n_tokens
        .checked_mul(stride)
        .ok_or(IndexerTopkTreeError::InvalidShape)?;
    let mut levels = vec![ScratchLevel {
        offset: 0,
        n_sets,
        stride,
    }];
    while n_sets > MERGE_GROUP {
        n_sets = n_sets.div_ceil(MERGE_GROUP);
        stride = n_sets
            .checked_mul(top_k)
            .ok_or(IndexerTopkTreeError::InvalidShape)?;
        let offset = total_elements;
        total_elements = total_elements
            .checked_add(
                n_tokens
                    .checked_mul(stride)
                    .ok_or(IndexerTopkTreeError::InvalidShape)?,
            )
            .ok_or(IndexerTopkTreeError::InvalidShape)?;
        levels.push(ScratchLevel {
            offset,
            n_sets,
            stride,
        });
    }
    Ok(ScratchPlan {
        levels,
        total_elements,
    })
}

fn indexer_topk_tree(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scratch: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
    plan: &ScratchPlan,
) -> Result<(), IndexerTopkTreeError> {
    if u64::from(n_tokens) * u64::from(n_comp) > scores.len() as u64
        || u64::from(n_tokens) * u64::from(top_k) > selected.len() as u64
        || plan.total_elements as usize > scratch.len()
    {
        return Err(IndexerTopkTreeError::InvalidShape);
    }
    let first = plan.levels[0];
    module
        .indexer_topk_chunk_pow2_4096_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, first.n_sets, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            n_tokens,
            top_k,
            first.stride,
            scores,
            scratch,
        )
        .map_err(IndexerTopkTreeError::Driver)?;
    for levels in plan.levels.windows(2) {
        let current = levels[0];
        let next = levels[1];
        module
            .indexer_topk_tree_merge_pow2_4096_kernel(
                stream,
                LaunchConfig {
                    grid_dim: (n_tokens, next.n_sets, 1),
                    block_dim: (BLOCK_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                },
                n_comp,
                n_tokens,
                top_k,
                current.n_sets,
                MERGE_GROUP,
                current.offset,
                current.stride,
                next.offset,
                next.stride,
                scores,
                scratch,
            )
            .map_err(IndexerTopkTreeError::Driver)?;
    }
    let last = plan.levels[plan.levels.len() - 1];
    module
        .indexer_topk_merge_pow2_4096_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            n_tokens,
            top_k,
            last.offset,
            last.n_sets * top_k,
            last.stride,
            scratch,
            scores,
            selected,
        )
        .map_err(IndexerTopkTreeError::Driver)
}

fn build_scores(n_tokens: u32, n_comp: u32) -> Vec<f32> {
    let mut scores = Vec::with_capacity((n_tokens * n_comp) as usize);
    for token in 0..n_tokens {
        for component in 0..n_comp {
            scores.push(match component {
                0 | 4096 => 16384.0,
                _ => ((component * 37 + token * 11) % 1000) as f32,
            });
        }
    }
    scores
}

fn expected_topk(scores: &[f32], top_k: u32) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..scores.len() as u32).collect();
    indices.sort_unstable_by(|&left, &right| {
        scores[right as usize]
            .partial_cmp(&scores[left as usize])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(top_k as usize);
    indices
}

#[derive(Debug)]
enum IndexerTopkTreeError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerTopkTreeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("tree indexer top-k shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerTopkTreeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
