use std::{cmp::Ordering, fmt};

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D2C1_SCOPE};

const SORT_N: usize = 1024;
const TOP_K: u32 = 512;
const BLOCK_THREADS: u32 = 1024;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_topk_1024_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_N> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, SORT_N> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= BLOCK_THREADS {
            return;
        }
        let index = tid as usize;
        if tid < n_comp {
            unsafe {
                VALUES[index] = scores[token as usize * n_comp as usize + index];
                INDICES[index] = tid;
            }
        } else {
            unsafe {
                VALUES[index] = f32::NEG_INFINITY;
                INDICES[index] = u32::MAX;
            }
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= SORT_N as u32 {
            let mut j = k >> 1;
            while j > 0 {
                let other = tid ^ j;
                if other > tid && other < SORT_N as u32 {
                    let other_index = other as usize;
                    let av = unsafe { VALUES[index] };
                    let bv = unsafe { VALUES[other_index] };
                    let ai = unsafe { INDICES[index] };
                    let bi = unsafe { INDICES[other_index] };
                    let desc_half = (tid & k) == 0;
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
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        if tid < top_k {
            unsafe {
                *selected.get_unchecked_mut(token as usize * top_k as usize + index) =
                    INDICES[index];
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let full_scores = build_scores(2, SORT_N as u32);
    let full_scores_device = substrate.upload(&full_scores)?;
    let mut full_selected = substrate.zeroed::<u32>(2 * TOP_K as usize)?;
    indexer_topk_1024(
        &module,
        substrate.stream(),
        &mut full_selected,
        &full_scores_device,
        SORT_N as u32,
        2,
        TOP_K,
    )?;
    substrate.flush_commands()?;
    let full_output = substrate.download(&full_selected)?;
    assert_eq!(
        full_output,
        expected_topk(&full_scores, 2, SORT_N as u32, TOP_K)
    );
    assert_eq!(&full_output[0..2], &[0, 1]);
    assert_eq!(&full_output[TOP_K as usize..TOP_K as usize + 2], &[0, 1]);

    let partial_n_comp = 700_u32;
    let partial_scores = build_scores(1, partial_n_comp);
    let partial_scores_device = substrate.upload(&partial_scores)?;
    let mut partial_selected = substrate.zeroed::<u32>(TOP_K as usize)?;
    indexer_topk_1024(
        &module,
        substrate.stream(),
        &mut partial_selected,
        &partial_scores_device,
        partial_n_comp,
        1,
        TOP_K,
    )?;
    substrate.end_commands()?;
    let partial_output = substrate.download(&partial_selected)?;
    assert_eq!(
        partial_output,
        expected_topk(&partial_scores, 1, partial_n_comp, TOP_K)
    );
    assert!(partial_output.iter().all(|&index| index < partial_n_comp));

    assert!(matches!(
        indexer_topk_1024(
            &module,
            substrate.stream(),
            &mut partial_selected,
            &partial_scores_device,
            partial_n_comp,
            1,
            TOP_K - 1,
        ),
        Err(IndexerTopk1024Error::InvalidShape)
    ));
    assert!(matches!(
        indexer_topk_1024(
            &module,
            substrate.stream(),
            &mut full_selected,
            &full_scores_device,
            SORT_N as u32 + 1,
            1,
            TOP_K,
        ),
        Err(IndexerTopk1024Error::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2d2c1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"topk1024_output_matches\":true,\"partial_component_output_matches\":true,\"stable_tie_order_matches\":true,\"invalid_shape_rejected\":true,\"owns_indexer_topk_1024_kernel\":{},\"owns_larger_topk_dispatch\":{},\"owns_indexed_topk_sort_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2C1_SCOPE.owns_indexer_topk_1024_kernel,
        M14_2D2C1_SCOPE.owns_larger_topk_dispatch,
        M14_2D2C1_SCOPE.owns_indexed_topk_sort_dispatch,
        M14_2D2C1_SCOPE.changes_default_route,
    );
    Ok(())
}

fn build_scores(n_tokens: u32, n_comp: u32) -> Vec<f32> {
    let mut scores = Vec::with_capacity(n_tokens as usize * n_comp as usize);
    for token in 0..n_tokens {
        for comp in 0..n_comp {
            let score = if comp < 2 {
                4096.0_f32 - token as f32
            } else {
                (comp % 127) as f32 - token as f32 * 0.25
            };
            scores.push(score);
        }
    }
    scores
}

fn expected_topk(scores: &[f32], n_tokens: u32, n_comp: u32, top_k: u32) -> Vec<u32> {
    let mut expected = Vec::with_capacity(n_tokens as usize * top_k as usize);
    for token in 0..n_tokens {
        let base = token as usize * n_comp as usize;
        let mut indices: Vec<u32> = (0..n_comp).collect();
        indices.sort_unstable_by(|&left, &right| {
            scores[base + right as usize]
                .partial_cmp(&scores[base + left as usize])
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });
        expected.extend_from_slice(&indices[..top_k as usize]);
    }
    expected
}

fn indexer_topk_1024(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerTopk1024Error> {
    if n_comp == 0
        || n_comp > SORT_N as u32
        || n_tokens == 0
        || top_k != TOP_K
        || top_k > n_comp
        || u64::from(n_tokens) * u64::from(n_comp) > scores.len() as u64
        || u64::from(n_tokens) * u64::from(top_k) > selected.len() as u64
    {
        return Err(IndexerTopk1024Error::InvalidShape);
    }
    module
        .indexer_topk_1024_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            n_tokens,
            top_k,
            scores,
            selected,
        )
        .map_err(IndexerTopk1024Error::Driver)
}

#[derive(Debug)]
enum IndexerTopk1024Error {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerTopk1024Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("1024-element indexer top-k shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerTopk1024Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
