use std::{cmp::Ordering, fmt};

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D2C2_SCOPE};

const SORT_2048: usize = 2048;
const SORT_4096: usize = 4096;
const SORT_8192: usize = 8192;
const TOP_K: u32 = 512;
const BLOCK_THREADS: u32 = 1024;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_topk_pow2_2048_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_2048> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, SORT_2048> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < SORT_2048 as u32 {
            let index = i as usize;
            if i < n_comp {
                unsafe {
                    VALUES[index] = scores[token as usize * n_comp as usize + index];
                    INDICES[index] = i;
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
        while k <= SORT_2048 as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < SORT_2048 as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_2048 as u32 {
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

    #[kernel]
    pub fn indexer_topk_pow2_4096_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_4096> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, SORT_4096> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < SORT_4096 as u32 {
            let index = i as usize;
            if i < n_comp {
                unsafe {
                    VALUES[index] = scores[token as usize * n_comp as usize + index];
                    INDICES[index] = i;
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
        while k <= SORT_4096 as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < SORT_4096 as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_4096 as u32 {
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

    #[kernel]
    pub fn indexer_topk_pow2_u16_8192_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<f32, SORT_8192> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u16, SORT_8192> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens {
            return;
        }
        let mut i = tid;
        while i < SORT_8192 as u32 {
            let index = i as usize;
            if i < n_comp {
                unsafe {
                    VALUES[index] = scores[token as usize * n_comp as usize + index];
                    INDICES[index] = i as u16;
                }
            } else {
                unsafe {
                    VALUES[index] = f32::NEG_INFINITY;
                    INDICES[index] = u16::MAX;
                }
            }
            i += BLOCK_THREADS;
        }
        thread::sync_threads();

        let mut k = 2_u32;
        while k <= SORT_8192 as u32 {
            let mut j = k >> 1;
            while j > 0 {
                i = tid;
                while i < SORT_8192 as u32 {
                    let other = i ^ j;
                    if other > i && other < SORT_8192 as u32 {
                        let index = i as usize;
                        let other_index = other as usize;
                        let av = unsafe { VALUES[index] };
                        let bv = unsafe { VALUES[other_index] };
                        let ai = unsafe { INDICES[index] } as u32;
                        let bi = unsafe { INDICES[other_index] } as u32;
                        let desc_half = (i & k) == 0;
                        let b_better = bv > av || (bv == av && bi < ai);
                        let a_better = av > bv || (av == bv && ai < bi);
                        let swap = if desc_half { b_better } else { a_better };
                        if swap {
                            unsafe {
                                VALUES[index] = bv;
                                INDICES[index] = bi as u16;
                                VALUES[other_index] = av;
                                INDICES[other_index] = ai as u16;
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
                    INDICES[i as usize] as u32;
            }
            i += BLOCK_THREADS;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    run_case(
        &substrate,
        &module,
        1500,
        indexer_topk_pow2_2048,
        Pow2Kind::U32_2048,
    )?;
    run_case(
        &substrate,
        &module,
        3000,
        indexer_topk_pow2_4096,
        Pow2Kind::U32_4096,
    )?;
    run_case(
        &substrate,
        &module,
        6000,
        indexer_topk_pow2_u16_8192,
        Pow2Kind::U16_8192,
    )?;
    substrate.end_commands()?;

    let scores = substrate.upload(&build_scores(2050))?;
    let mut selected = substrate.zeroed::<u32>(TOP_K as usize)?;
    assert!(matches!(
        indexer_topk_pow2_2048(
            &module,
            substrate.stream(),
            &mut selected,
            &scores,
            2050,
            1,
            TOP_K,
        ),
        Err(IndexerTopkPow2Error::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2d2c2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"topk2048_output_matches\":true,\"topk4096_output_matches\":true,\"topk8192_u16_output_matches\":true,\"stable_tie_order_matches\":true,\"partial_component_output_matches\":true,\"invalid_shape_rejected\":true,\"owns_indexer_topk_pow2_2048_kernel\":{},\"owns_indexer_topk_pow2_4096_kernel\":{},\"owns_indexer_topk_pow2_u16_8192_kernel\":{},\"owns_cub_topk_dispatch\":{},\"owns_chunked_topk_dispatch\":{},\"owns_indexed_topk_sort_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2C2_SCOPE.owns_indexer_topk_pow2_2048_kernel,
        M14_2D2C2_SCOPE.owns_indexer_topk_pow2_4096_kernel,
        M14_2D2C2_SCOPE.owns_indexer_topk_pow2_u16_8192_kernel,
        M14_2D2C2_SCOPE.owns_cub_topk_dispatch,
        M14_2D2C2_SCOPE.owns_chunked_topk_dispatch,
        M14_2D2C2_SCOPE.owns_indexed_topk_sort_dispatch,
        M14_2D2C2_SCOPE.changes_default_route,
    );
    Ok(())
}

type Pow2Launch = fn(
    &kernels::LoadedModule,
    &CudaStream,
    &mut DeviceBuffer<u32>,
    &DeviceBuffer<f32>,
    u32,
    u32,
    u32,
) -> Result<(), IndexerTopkPow2Error>;

#[derive(Clone, Copy)]
enum Pow2Kind {
    U32_2048,
    U32_4096,
    U16_8192,
}

fn run_case(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    n_comp: u32,
    launch: Pow2Launch,
    kind: Pow2Kind,
) -> Result<(), Box<dyn std::error::Error>> {
    let scores_values = build_scores(n_comp);
    let scores = substrate.upload(&scores_values)?;
    let mut selected = substrate.zeroed::<u32>(TOP_K as usize)?;
    launch(
        module,
        substrate.stream(),
        &mut selected,
        &scores,
        n_comp,
        1,
        TOP_K,
    )?;
    substrate.flush_commands()?;
    let output = substrate.download(&selected)?;
    assert_eq!(output, expected_topk(&scores_values, n_comp, TOP_K));
    assert_eq!(&output[0..2], &[0, 1]);
    assert!(output.iter().all(|&index| index < n_comp));
    match kind {
        Pow2Kind::U32_2048 => assert!(n_comp > SORT_2048 as u32 / 2),
        Pow2Kind::U32_4096 => assert!(n_comp > SORT_4096 as u32 / 2),
        Pow2Kind::U16_8192 => assert!(n_comp > SORT_8192 as u32 / 2),
    }
    Ok(())
}

fn build_scores(n_comp: u32) -> Vec<f32> {
    (0..n_comp)
        .map(|comp| {
            if comp < 2 {
                16384.0_f32
            } else {
                (comp % 257) as f32
            }
        })
        .collect()
}

fn expected_topk(scores: &[f32], n_comp: u32, top_k: u32) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..n_comp).collect();
    indices.sort_unstable_by(|&left, &right| {
        scores[right as usize]
            .partial_cmp(&scores[left as usize])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(top_k as usize);
    indices
}

fn validate_shape(
    selected: &DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
    max_comp: u32,
) -> Result<(), IndexerTopkPow2Error> {
    if n_comp == 0
        || n_comp > max_comp
        || n_tokens == 0
        || top_k != TOP_K
        || top_k > n_comp
        || u64::from(n_tokens) * u64::from(n_comp) > scores.len() as u64
        || u64::from(n_tokens) * u64::from(top_k) > selected.len() as u64
    {
        Err(IndexerTopkPow2Error::InvalidShape)
    } else {
        Ok(())
    }
}

fn indexer_topk_pow2_2048(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerTopkPow2Error> {
    validate_shape(selected, scores, n_comp, n_tokens, top_k, SORT_2048 as u32)?;
    module
        .indexer_topk_pow2_2048_kernel(
            stream,
            launch_config(n_tokens),
            n_comp,
            n_tokens,
            top_k,
            scores,
            selected,
        )
        .map_err(IndexerTopkPow2Error::Driver)
}

fn indexer_topk_pow2_4096(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerTopkPow2Error> {
    validate_shape(selected, scores, n_comp, n_tokens, top_k, SORT_4096 as u32)?;
    module
        .indexer_topk_pow2_4096_kernel(
            stream,
            launch_config(n_tokens),
            n_comp,
            n_tokens,
            top_k,
            scores,
            selected,
        )
        .map_err(IndexerTopkPow2Error::Driver)
}

fn indexer_topk_pow2_u16_8192(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerTopkPow2Error> {
    validate_shape(selected, scores, n_comp, n_tokens, top_k, SORT_8192 as u32)?;
    module
        .indexer_topk_pow2_u16_8192_kernel(
            stream,
            launch_config(n_tokens),
            n_comp,
            n_tokens,
            top_k,
            scores,
            selected,
        )
        .map_err(IndexerTopkPow2Error::Driver)
}

fn launch_config(n_tokens: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_tokens, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[derive(Debug)]
enum IndexerTopkPow2Error {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerTopkPow2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("power-of-two indexer top-k shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerTopkPow2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
