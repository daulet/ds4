use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D2A_SCOPE};

const HEAD_DIM: usize = 128;
const N_HEAD: usize = 64;
const DIRECT_THREADS: u32 = 128;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_score_one_direct_kernel(
        n_comp: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut K_ROW: SharedArray<f32, HEAD_DIM> = SharedArray::UNINIT;
        static mut PARTIAL: SharedArray<f32, 4> = SharedArray::UNINIT;

        let comp = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if comp >= n_comp || tid >= DIRECT_THREADS {
            return;
        }
        let lane = tid & 31;
        let warp_id = tid >> 5;
        if causal != 0 {
            let visible = if ratio != 0 {
                (pos0 + 1) / ratio
            } else {
                n_comp
            };
            if comp >= visible {
                if tid == 0 {
                    unsafe {
                        *scores.get_unchecked_mut(comp as usize) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
        }

        unsafe {
            K_ROW[tid as usize] = index_comp[comp as usize * HEAD_DIM + tid as usize];
        }
        thread::sync_threads();

        let mut total = 0.0_f32;
        let mut head_group = 0_u32;
        while head_group < N_HEAD as u32 {
            let head = head_group + warp_id;
            let q_base = head as usize * HEAD_DIM + lane as usize * 4;
            let k_base = lane as usize * 4;
            let mut dot = q[q_base] * unsafe { K_ROW[k_base] }
                + q[q_base + 1] * unsafe { K_ROW[k_base + 1] }
                + q[q_base + 2] * unsafe { K_ROW[k_base + 2] }
                + q[q_base + 3] * unsafe { K_ROW[k_base + 3] };
            let mut offset = 16_u32;
            while offset > 0 {
                dot += warp::shuffle_down_f32(dot, offset);
                offset >>= 1;
            }
            if lane == 0 {
                let positive = if (dot.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || dot <= 0.0_f32 {
                    0.0_f32
                } else {
                    dot
                };
                unsafe {
                    PARTIAL[warp_id as usize] = positive * weights[head as usize] * scale;
                }
            }
            thread::sync_threads();
            if tid == 0 {
                total += unsafe { PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3] };
            }
            thread::sync_threads();
            head_group += 4;
        }
        if tid == 0 {
            unsafe {
                *scores.get_unchecked_mut(comp as usize) = total;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let mut q_values = vec![0.0_f32; N_HEAD * HEAD_DIM];
    for head in 0..N_HEAD {
        q_values[head * HEAD_DIM] = 1.0;
    }
    let weight_values = vec![1.0_f32; N_HEAD];
    let mut comp_values = vec![0.0_f32; 4 * HEAD_DIM];
    comp_values[0] = 1.0;
    comp_values[HEAD_DIM] = 2.0;
    comp_values[2 * HEAD_DIM] = -1.0;
    comp_values[3 * HEAD_DIM] = f32::NAN;
    let q = substrate.upload(&q_values)?;
    let weights = substrate.upload(&weight_values)?;
    let index_comp = substrate.upload(&comp_values)?;

    let mut scores = substrate.zeroed::<f32>(4)?;
    indexer_score_one_direct(
        &module,
        substrate.stream(),
        &mut scores,
        &q,
        &weights,
        &index_comp,
        4,
        0,
        1,
        0.5,
        false,
    )?;
    substrate.flush_commands()?;
    assert_eq!(substrate.download(&scores)?, [32.0, 64.0, 0.0, 0.0]);

    let mut causal_scores = substrate.zeroed::<f32>(4)?;
    indexer_score_one_direct(
        &module,
        substrate.stream(),
        &mut causal_scores,
        &q,
        &weights,
        &index_comp,
        4,
        4,
        2,
        0.5,
        true,
    )?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&causal_scores)?,
        [32.0, 64.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
    );

    assert!(matches!(
        indexer_score_one_direct(
            &module,
            substrate.stream(),
            &mut scores,
            &q,
            &weights,
            &index_comp,
            4,
            4,
            0,
            0.5,
            true,
        ),
        Err(IndexerDirectError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2d2a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"direct_score_output_matches\":true,\"causal_mask_output_matches\":true,\"nan_negative_clamp_matches\":true,\"invalid_shape_rejected\":true,\"owns_indexer_score_one_direct_kernel\":{},\"owns_wmma_indexer_dispatch\":{},\"owns_specialized_topk_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2A_SCOPE.owns_indexer_score_one_direct_kernel,
        M14_2D2A_SCOPE.owns_wmma_indexer_dispatch,
        M14_2D2A_SCOPE.owns_specialized_topk_dispatch,
        M14_2D2A_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn indexer_score_one_direct(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scores: &mut DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    index_comp: &DeviceBuffer<f32>,
    n_comp: u32,
    pos0: u32,
    ratio: u32,
    scale: f32,
    causal: bool,
) -> Result<(), IndexerDirectError> {
    if n_comp == 0
        || (causal && ratio == 0)
        || scores.len() < n_comp as usize
        || q.len() < N_HEAD * HEAD_DIM
        || weights.len() < N_HEAD
        || u64::from(n_comp) * HEAD_DIM as u64 > index_comp.len() as u64
    {
        return Err(IndexerDirectError::InvalidShape);
    }
    module
        .indexer_score_one_direct_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_comp, 1, 1),
                block_dim: (DIRECT_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            pos0,
            ratio,
            scale,
            causal as u32,
            q,
            weights,
            index_comp,
            scores,
        )
        .map_err(IndexerDirectError::Driver)
}

#[derive(Debug)]
enum IndexerDirectError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerDirectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("direct indexer score shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerDirectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
