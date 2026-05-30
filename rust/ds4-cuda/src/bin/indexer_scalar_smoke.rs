use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_scores_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let comp = thread::blockIdx_x();
        let token = thread::blockIdx_y();
        if comp >= n_comp || token >= n_tokens {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        if causal != 0 {
            let visible = (pos0 + token + 1) / ratio;
            if comp >= visible {
                if tid == 0 {
                    let output = token as usize * n_comp as usize + comp as usize;
                    unsafe {
                        *scores.get_unchecked_mut(output) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
        }

        let mut total = 0.0_f32;
        let mut head = 0;
        while head < n_head {
            let q_base = (token as usize * n_head as usize + head as usize) * head_dim as usize;
            let comp_base = comp as usize * head_dim as usize;
            let mut dot = 0.0_f32;
            let mut dimension = tid;
            while dimension < head_dim as usize {
                dot += q[q_base + dimension] * index_comp[comp_base + dimension];
                dimension += 256;
            }
            unsafe {
                PARTIAL[tid] = dot;
            }
            thread::sync_threads();

            let mut stride = 128;
            while stride > 0 {
                if tid < stride {
                    unsafe {
                        PARTIAL[tid] += PARTIAL[tid + stride];
                    }
                }
                thread::sync_threads();
                stride >>= 1;
            }

            let reduced = unsafe { PARTIAL[0] };
            let positive = if (reduced.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || reduced <= 0.0_f32
            {
                0.0_f32
            } else {
                reduced
            };
            total += positive * weights[token as usize * n_head as usize + head as usize];
            thread::sync_threads();
            head += 1;
        }
        if tid == 0 {
            let output = token as usize * n_comp as usize + comp as usize;
            unsafe {
                *scores.get_unchecked_mut(output) = total * scale;
            }
        }
    }

    #[kernel]
    pub fn indexer_topk_kernel(
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
        scores: &[f32],
        mut selected: DisjointSlice<u32>,
    ) {
        let token = thread::blockIdx_x();
        if token >= n_tokens || thread::threadIdx_x() != 0 {
            return;
        }
        let score_base = token as usize * n_comp as usize;
        let selected_base = token as usize * top_k as usize;
        let mut k = 0;
        while k < top_k {
            unsafe {
                *selected.get_unchecked_mut(selected_base + k as usize) = 0;
            }
            k += 1;
        }
        let mut comp = 0;
        while comp < n_comp {
            let value = scores[score_base + comp as usize];
            k = 0;
            while k < top_k {
                let selected_index =
                    unsafe { *selected.get_unchecked_mut(selected_base + k as usize) };
                if k >= comp || value > scores[score_base + selected_index as usize] {
                    let mut move_index = top_k - 1;
                    while move_index > k {
                        let previous = unsafe {
                            *selected.get_unchecked_mut(selected_base + move_index as usize - 1)
                        };
                        unsafe {
                            *selected.get_unchecked_mut(selected_base + move_index as usize) =
                                previous;
                        }
                        move_index -= 1;
                    }
                    unsafe {
                        *selected.get_unchecked_mut(selected_base + k as usize) = comp;
                    }
                    break;
                }
                k += 1;
            }
            comp += 1;
        }
    }

    #[kernel]
    pub fn topk_mask_kernel(
        count: u64,
        n_comp: u32,
        top_k: u32,
        topk: &[u32],
        mut mask: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let gid = index.get();
        if (gid as u64) >= count {
            return;
        }
        let token = gid / n_comp as usize;
        let comp = gid - token * n_comp as usize;
        let mut value = f32::NEG_INFINITY;
        let mut k = 0;
        while k < top_k {
            if topk[token * top_k as usize + k as usize] == comp as u32 {
                value = 0.0;
                break;
            }
            k += 1;
        }
        if let Some(element) = mask.get_mut(index) {
            *element = value;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let q_values = [1.0_f32, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0];
    let weight_values = [1.0_f32, 0.5, 0.25, 1.0];
    let comp_values = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let q = substrate.upload(&q_values)?;
    let weights = substrate.upload(&weight_values)?;
    let index_comp = substrate.upload(&comp_values)?;

    let mut scores = substrate.zeroed::<f32>(6)?;
    indexer_scores(
        &module,
        substrate.stream(),
        &mut scores,
        &q,
        &weights,
        &index_comp,
        3,
        2,
        0,
        2,
        2,
        1,
        0.5,
        false,
    )?;
    substrate.flush_commands()?;
    assert_eq!(
        substrate.download(&scores)?,
        [0.5, 1.25, 1.5, 0.75, 1.0, 1.625]
    );

    let mut causal_scores = substrate.zeroed::<f32>(6)?;
    indexer_scores(
        &module,
        substrate.stream(),
        &mut causal_scores,
        &q,
        &weights,
        &index_comp,
        3,
        2,
        4,
        2,
        2,
        2,
        0.5,
        true,
    )?;
    substrate.flush_commands()?;
    assert_eq!(
        substrate.download(&causal_scores)?,
        [0.5, 1.25, f32::NEG_INFINITY, 0.75, 1.0, 1.625]
    );

    let mut selected = substrate.zeroed::<u32>(4)?;
    indexer_topk(&module, substrate.stream(), &mut selected, &scores, 3, 2, 2)?;
    substrate.flush_commands()?;
    assert_eq!(substrate.download(&selected)?, [2, 1, 2, 1]);

    let tied_scores = substrate.upload(&[2.0_f32, 2.0, 1.0, -1.0, -1.0, -2.0])?;
    let mut tied_selected = substrate.zeroed::<u32>(4)?;
    indexer_topk(
        &module,
        substrate.stream(),
        &mut tied_selected,
        &tied_scores,
        3,
        2,
        2,
    )?;
    substrate.flush_commands()?;
    assert_eq!(substrate.download(&tied_selected)?, [0, 1, 0, 1]);

    let mut mask = substrate.zeroed::<f32>(6)?;
    topk_mask(&module, substrate.stream(), &mut mask, &selected, 3, 2, 2)?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&mask)?,
        [f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0]
    );

    assert!(matches!(
        indexer_scores(
            &module,
            substrate.stream(),
            &mut scores,
            &q,
            &weights,
            &index_comp,
            3,
            2,
            4,
            2,
            2,
            0,
            0.5,
            true,
        ),
        Err(IndexerError::InvalidScoreShape)
    ));
    assert!(matches!(
        indexer_topk(&module, substrate.stream(), &mut selected, &scores, 3, 2, 4),
        Err(IndexerError::InvalidTopKShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2d1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"indexer_scores_output_matches\":true,\"causal_scores_output_matches\":true,\"indexer_topk_output_matches\":true,\"indexer_topk_tie_order_matches\":true,\"topk_mask_output_matches\":true,\"invalid_shape_rejected\":true,\"owns_indexer_scores_fallback_kernel\":{},\"owns_indexer_topk_fallback_kernel\":{},\"owns_topk_mask_tensor\":{},\"owns_optimized_indexer_dispatch\":{},\"owns_optimized_topk_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D1_SCOPE.owns_indexer_scores_fallback_kernel,
        M14_2D1_SCOPE.owns_indexer_topk_fallback_kernel,
        M14_2D1_SCOPE.owns_topk_mask_tensor,
        M14_2D1_SCOPE.owns_optimized_indexer_dispatch,
        M14_2D1_SCOPE.owns_optimized_topk_dispatch,
        M14_2D1_SCOPE.changes_default_route,
    );
    Ok(())
}

const THREADS_PER_BLOCK: u32 = 256;

#[allow(clippy::too_many_arguments)]
fn indexer_scores(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scores: &mut DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    index_comp: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    pos0: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: f32,
    causal: bool,
) -> Result<(), IndexerError> {
    let score_elements = u64::from(n_tokens) * u64::from(n_comp);
    let q_elements = u64::from(n_tokens) * u64::from(n_head) * u64::from(head_dim);
    let weight_elements = u64::from(n_tokens) * u64::from(n_head);
    let comp_elements = u64::from(n_comp) * u64::from(head_dim);
    if n_comp == 0
        || n_tokens == 0
        || n_head == 0
        || head_dim == 0
        || (causal && ratio == 0)
        || score_elements > scores.len() as u64
        || q_elements > q.len() as u64
        || weight_elements > weights.len() as u64
        || comp_elements > index_comp.len() as u64
    {
        return Err(IndexerError::InvalidScoreShape);
    }
    module
        .indexer_scores_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_comp, n_tokens, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            n_tokens,
            pos0,
            n_head,
            head_dim,
            ratio,
            scale,
            causal as u32,
            q,
            weights,
            index_comp,
            scores,
        )
        .map_err(IndexerError::Driver)
}

fn indexer_topk(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    selected: &mut DeviceBuffer<u32>,
    scores: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerError> {
    if n_comp == 0
        || n_tokens == 0
        || top_k == 0
        || top_k > n_comp
        || u64::from(n_tokens) * u64::from(n_comp) > scores.len() as u64
        || u64::from(n_tokens) * u64::from(top_k) > selected.len() as u64
    {
        return Err(IndexerError::InvalidTopKShape);
    }
    module
        .indexer_topk_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            n_tokens,
            top_k,
            scores,
            selected,
        )
        .map_err(IndexerError::Driver)
}

fn topk_mask(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    mask: &mut DeviceBuffer<f32>,
    topk: &DeviceBuffer<u32>,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> Result<(), IndexerError> {
    let count = u64::from(n_tokens) * u64::from(n_comp);
    let selected_count = u64::from(n_tokens) * u64::from(top_k);
    if n_comp == 0
        || n_tokens == 0
        || top_k == 0
        || count > mask.len() as u64
        || selected_count > topk.len() as u64
    {
        return Err(IndexerError::InvalidMaskShape);
    }
    let blocks = count
        .max(selected_count)
        .div_ceil(u64::from(THREADS_PER_BLOCK));
    let grid_x =
        u32::try_from(blocks).map_err(|_| IndexerError::GridDimensionTooLarge { blocks })?;
    module
        .topk_mask_kernel(
            stream,
            LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            count,
            n_comp,
            top_k,
            topk,
            mask,
        )
        .map_err(IndexerError::Driver)
}

#[derive(Debug)]
enum IndexerError {
    InvalidScoreShape,
    InvalidTopKShape,
    InvalidMaskShape,
    GridDimensionTooLarge { blocks: u64 },
    Driver(DriverError),
}

impl fmt::Display for IndexerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScoreShape => formatter.write_str("indexer score shape is invalid"),
            Self::InvalidTopKShape => formatter.write_str("indexer top-k shape is invalid"),
            Self::InvalidMaskShape => formatter.write_str("top-k mask shape is invalid"),
            Self::GridDimensionTooLarge { blocks } => {
                write!(formatter, "top-k mask launch requires {blocks} CUDA blocks")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidScoreShape
            | Self::InvalidTopKShape
            | Self::InvalidMaskShape
            | Self::GridDimensionTooLarge { .. } => None,
        }
    }
}
