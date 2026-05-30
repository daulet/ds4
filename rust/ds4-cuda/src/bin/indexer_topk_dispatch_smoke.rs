use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{
    select_indexer_topk_kernel, should_sort_indexed_topk, substrate::CudaOxideSubstrate,
    IndexedTopkSortOptions, IndexerTopkDispatchOptions, IndexerTopkKernel, M14_2D2C5_SCOPE,
};

const TOP_K: u32 = 512;
const BLOCK_THREADS: u32 = 512;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexed_topk_sort_512_asc_kernel(
        n_tokens: u32,
        src: &[i32],
        mut dst: DisjointSlice<i32>,
    ) {
        static mut ROWS: SharedArray<i32, { TOP_K as usize }> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= BLOCK_THREADS {
            return;
        }
        let index = tid as usize;
        let offset = token as usize * TOP_K as usize + index;
        unsafe {
            ROWS[index] = src[offset];
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= TOP_K {
            let mut j = k >> 1;
            while j > 0 {
                let other = tid ^ j;
                if other > tid && other < TOP_K {
                    let other_index = other as usize;
                    let a = unsafe { ROWS[index] };
                    let b = unsafe { ROWS[other_index] };
                    let up = (tid & k) == 0;
                    if (up && a > b) || (!up && a < b) {
                        unsafe {
                            ROWS[index] = b;
                            ROWS[other_index] = a;
                        }
                    }
                }
                thread::sync_threads();
                j >>= 1;
            }
            k <<= 1;
        }

        unsafe {
            *dst.get_unchecked_mut(offset) = ROWS[index];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let source = build_rows();
    let source_device = substrate.upload(&source)?;
    let mut sorted_device = substrate.zeroed::<i32>(source.len())?;
    indexed_topk_sort_512_asc(
        &module,
        substrate.stream(),
        &mut sorted_device,
        &source_device,
        2,
        TOP_K,
    )?;
    substrate.end_commands()?;
    let sorted = substrate.download(&sorted_device)?;
    let expected: Vec<i32> = (0..TOP_K as i32).collect();
    assert_eq!(&sorted[..TOP_K as usize], expected.as_slice());
    assert_eq!(&sorted[TOP_K as usize..], expected.as_slice());
    assert!(matches!(
        indexed_topk_sort_512_asc(
            &module,
            substrate.stream(),
            &mut sorted_device,
            &source_device,
            1,
            TOP_K,
        ),
        Err(IndexerTopkDispatchError::InvalidShape)
    ));
    assert!(sort_dispatch_gate_matches_current_c());
    assert!(topk_dispatch_priority_matches_current_c());

    println!(
        "{{\"milestone\":\"M14.2d2c5\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"indexed_sort_output_matches\":true,\"multi_token_rows_match\":true,\"sort_dispatch_gate_matches\":true,\"topk_dispatch_priority_matches\":true,\"packed_key_equivalent_selection_matches\":true,\"invalid_shape_rejected\":true,\"owns_indexed_topk_sort_512_asc_kernel\":{},\"owns_indexed_topk_sort_dispatch\":{},\"owns_topk_dispatch_policy\":{},\"uses_packed_key_equivalent_branch\":{},\"owns_cub_library_implementation\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2C5_SCOPE.owns_indexed_topk_sort_512_asc_kernel,
        M14_2D2C5_SCOPE.owns_indexed_topk_sort_dispatch,
        M14_2D2C5_SCOPE.owns_topk_dispatch_policy,
        M14_2D2C5_SCOPE.uses_packed_key_equivalent_branch,
        M14_2D2C5_SCOPE.owns_cub_library_implementation,
        M14_2D2C5_SCOPE.changes_default_route,
    );
    Ok(())
}

fn build_rows() -> Vec<i32> {
    let mut rows = Vec::with_capacity(2 * TOP_K as usize);
    rows.extend((0..TOP_K as i32).rev());
    rows.extend((0..TOP_K).map(|index| ((index * 73) % TOP_K) as i32));
    rows
}

fn sort_dispatch_gate_matches_current_c() -> bool {
    let base = IndexedTopkSortOptions {
        n_tokens: 2,
        top_k: TOP_K,
        no_indexed_topk_sort: false,
    };
    should_sort_indexed_topk(base)
        && !should_sort_indexed_topk(IndexedTopkSortOptions {
            n_tokens: 1,
            ..base
        })
        && !should_sort_indexed_topk(IndexedTopkSortOptions {
            top_k: TOP_K - 1,
            ..base
        })
        && !should_sort_indexed_topk(IndexedTopkSortOptions {
            no_indexed_topk_sort: true,
            ..base
        })
}

fn topk_dispatch_priority_matches_current_c() -> bool {
    let base = IndexerTopkDispatchOptions {
        n_comp: 1000,
        top_k: TOP_K,
        no_topk1024: false,
        no_topk2048: false,
        no_topk8192: false,
        no_topk_chunked: false,
        packed_dynamic_shared_available: true,
    };
    select_indexer_topk_kernel(base) == IndexerTopkKernel::Topk1024
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            no_topk1024: true,
            ..base
        }) == IndexerTopkKernel::Pow2U32x2048
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 4096,
            ..base
        }) == IndexerTopkKernel::PackedKeyEquivalent
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 4096,
            packed_dynamic_shared_available: false,
            ..base
        }) == IndexerTopkKernel::Pow2U32x4096
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 6000,
            ..base
        }) == IndexerTopkKernel::PackedKeyEquivalent
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 6000,
            packed_dynamic_shared_available: false,
            ..base
        }) == IndexerTopkKernel::Pow2U16x8192
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 6000,
            no_topk8192: true,
            ..base
        }) == IndexerTopkKernel::ChunkedTree
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 9000,
            ..base
        }) == IndexerTopkKernel::ChunkedTree
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 9000,
            no_topk_chunked: true,
            ..base
        }) == IndexerTopkKernel::Scalar
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            n_comp: 6000,
            no_topk2048: true,
            ..base
        }) == IndexerTopkKernel::Scalar
        && select_indexer_topk_kernel(IndexerTopkDispatchOptions {
            top_k: TOP_K - 1,
            ..base
        }) == IndexerTopkKernel::Scalar
}

fn indexed_topk_sort_512_asc(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    dst: &mut DeviceBuffer<i32>,
    src: &DeviceBuffer<i32>,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerTopkDispatchError> {
    let count = u64::from(n_tokens) * u64::from(top_k);
    if n_tokens <= 1 || top_k != TOP_K || count > src.len() as u64 || count > dst.len() as u64 {
        return Err(IndexerTopkDispatchError::InvalidShape);
    }
    module
        .indexed_topk_sort_512_asc_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_tokens,
            src,
            dst,
        )
        .map_err(IndexerTopkDispatchError::Driver)
}

#[derive(Debug)]
enum IndexerTopkDispatchError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerTopkDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("indexed top-k sort shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerTopkDispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
