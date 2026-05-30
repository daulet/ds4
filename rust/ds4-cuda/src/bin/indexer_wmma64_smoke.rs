#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::mma::{load_a_m16n8k16, load_b_m16n8k16, mma_m16n8k16_f32_f16, zero_accumulator};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2D2B2B_SCOPE};

const HEAD_DIM: usize = 128;
const N_HEAD: usize = 64;
const TILE_TOKENS: usize = 16;
const TILE_COMPONENTS: usize = 16;
const WIDE_COMPONENTS: usize = 64;
const MMA_K: usize = 16;
const MMA_N: usize = 8;
const WMMA_WARPS: usize = 4;
const WMMA_THREADS: u32 = 128;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexer_scores_wmma64_kernel(
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        ratio: u32,
        scale: f32,
        causal: u32,
        q: &[f32],
        weights: &[f32],
        index_comp: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        static mut A_TILE: SharedArray<f16, { TILE_TOKENS * MMA_K }, 16> = SharedArray::UNINIT;
        static mut B_LO_TILE: SharedArray<f16, { WMMA_WARPS * MMA_K * MMA_N }, 16> =
            SharedArray::UNINIT;
        static mut B_HI_TILE: SharedArray<f16, { WMMA_WARPS * MMA_K * MMA_N }, 16> =
            SharedArray::UNINIT;
        static mut C_TILE: SharedArray<f32, { TILE_TOKENS * WIDE_COMPONENTS }> =
            SharedArray::UNINIT;
        static mut ACC_TILE: SharedArray<f32, { TILE_TOKENS * WIDE_COMPONENTS }> =
            SharedArray::UNINIT;

        let tile_c = thread::blockIdx_x() as usize * WIDE_COMPONENTS;
        let tile_t = thread::blockIdx_y() as usize * TILE_TOKENS;
        let tid = thread::threadIdx_x() as usize;
        if tid >= WMMA_THREADS as usize {
            return;
        }

        if causal != 0 {
            let tile_end = tile_t as u32 + TILE_TOKENS as u32;
            let last_token = if tile_end < n_tokens {
                tile_end
            } else {
                n_tokens
            };
            let max_visible = if last_token > tile_t as u32 && ratio != 0 {
                let visible = (pos0 + last_token) / ratio;
                if visible < n_comp {
                    visible
                } else {
                    n_comp
                }
            } else {
                0
            };
            if tile_c as u32 >= max_visible {
                let mut i = tid;
                while i < TILE_TOKENS * WIDE_COMPONENTS {
                    let row = i / WIDE_COMPONENTS;
                    let col = i % WIDE_COMPONENTS;
                    let token = tile_t + row;
                    let comp = tile_c + col;
                    if token < n_tokens as usize && comp < n_comp as usize {
                        unsafe {
                            *scores.get_unchecked_mut(token * n_comp as usize + comp) =
                                f32::NEG_INFINITY;
                        }
                    }
                    i += WMMA_THREADS as usize;
                }
                return;
            }
        }

        let mut i = tid;
        while i < TILE_TOKENS * WIDE_COMPONENTS {
            unsafe {
                ACC_TILE[i] = 0.0;
            }
            i += WMMA_THREADS as usize;
        }
        thread::sync_threads();

        let warp_id = tid >> 5;
        let lane = tid & 31;
        let a_row = (lane & 7) + (lane & 8);
        let a_col = if lane & 16 == 0 { 0 } else { 8 };
        let b_row = (lane & 7) + (lane & 8);
        let mut head = 0_usize;
        while head < N_HEAD {
            let mut acc_lo = zero_accumulator();
            let mut acc_hi = zero_accumulator();
            let mut k0 = 0_usize;
            while k0 < HEAD_DIM {
                let mut a_index = tid;
                while a_index < TILE_TOKENS * MMA_K {
                    let row = a_index / MMA_K;
                    let col = a_index % MMA_K;
                    let token = tile_t + row;
                    let value = if token < n_tokens as usize {
                        q[(token * N_HEAD + head) * HEAD_DIM + k0 + col]
                    } else {
                        0.0
                    };
                    unsafe {
                        A_TILE[a_index] = value as f16;
                    }
                    a_index += WMMA_THREADS as usize;
                }

                let mut b_index = tid;
                while b_index < WMMA_WARPS * MMA_K * MMA_N {
                    let warp_tile = b_index / (MMA_K * MMA_N);
                    let local = b_index % (MMA_K * MMA_N);
                    let row = local / MMA_N;
                    let col = local % MMA_N;
                    let comp_lo = tile_c + warp_tile * TILE_COMPONENTS + col;
                    let comp_hi = tile_c + warp_tile * TILE_COMPONENTS + MMA_N + col;
                    let dimension = k0 + row;
                    let value_lo = if comp_lo < n_comp as usize {
                        index_comp[comp_lo * HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    let value_hi = if comp_hi < n_comp as usize {
                        index_comp[comp_hi * HEAD_DIM + dimension]
                    } else {
                        0.0
                    };
                    unsafe {
                        B_LO_TILE[b_index] = value_lo as f16;
                        B_HI_TILE[b_index] = value_hi as f16;
                    }
                    b_index += WMMA_THREADS as usize;
                }
                thread::sync_threads();

                let a_ptr = unsafe { (&raw const A_TILE).cast::<f16>().add(a_row * MMA_K + a_col) }
                    .cast::<u8>();
                let b_base = warp_id * MMA_K * MMA_N + b_row * MMA_N;
                let b_lo_ptr =
                    unsafe { (&raw const B_LO_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let b_hi_ptr =
                    unsafe { (&raw const B_HI_TILE).cast::<f16>().add(b_base) }.cast::<u8>();
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                let b_lo_frag = unsafe { load_b_m16n8k16(b_lo_ptr) };
                let b_hi_frag = unsafe { load_b_m16n8k16(b_hi_ptr) };
                acc_lo = unsafe { mma_m16n8k16_f32_f16(acc_lo, a_frag, b_lo_frag) };
                let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
                acc_hi = unsafe { mma_m16n8k16_f32_f16(acc_hi, a_frag, b_hi_frag) };
                thread::sync_threads();
                k0 += MMA_K;
            }

            let group_id = lane >> 2;
            let thread_in_group = lane & 3;
            let col_base = warp_id * TILE_COMPONENTS + thread_in_group * 2;
            unsafe {
                C_TILE[group_id * WIDE_COMPONENTS + col_base] = acc_lo.x();
                C_TILE[group_id * WIDE_COMPONENTS + col_base + 1] = acc_lo.y();
                C_TILE[(group_id + 8) * WIDE_COMPONENTS + col_base] = acc_lo.z();
                C_TILE[(group_id + 8) * WIDE_COMPONENTS + col_base + 1] = acc_lo.w();
                C_TILE[group_id * WIDE_COMPONENTS + MMA_N + col_base] = acc_hi.x();
                C_TILE[group_id * WIDE_COMPONENTS + MMA_N + col_base + 1] = acc_hi.y();
                C_TILE[(group_id + 8) * WIDE_COMPONENTS + MMA_N + col_base] = acc_hi.z();
                C_TILE[(group_id + 8) * WIDE_COMPONENTS + MMA_N + col_base + 1] = acc_hi.w();
            }
            thread::sync_threads();

            let mut output_index = tid;
            while output_index < TILE_TOKENS * WIDE_COMPONENTS {
                let row = output_index / WIDE_COMPONENTS;
                let col = output_index % WIDE_COMPONENTS;
                let token = tile_t + row;
                let comp = tile_c + col;
                if token < n_tokens as usize && comp < n_comp as usize {
                    let value = unsafe { C_TILE[output_index] };
                    let positive = if (value.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || value <= 0.0
                    {
                        0.0
                    } else {
                        value
                    };
                    unsafe {
                        ACC_TILE[output_index] += positive * weights[token * N_HEAD + head];
                    }
                }
                output_index += WMMA_THREADS as usize;
            }
            thread::sync_threads();
            head += 1;
        }

        let mut output_index = tid;
        while output_index < TILE_TOKENS * WIDE_COMPONENTS {
            let row = output_index / WIDE_COMPONENTS;
            let col = output_index % WIDE_COMPONENTS;
            let token = tile_t + row;
            let comp = tile_c + col;
            if token < n_tokens as usize && comp < n_comp as usize {
                let mut output = unsafe { ACC_TILE[output_index] } * scale;
                if causal != 0 {
                    let visible = (pos0 + token as u32 + 1) / ratio;
                    if comp as u32 >= visible {
                        output = f32::NEG_INFINITY;
                    }
                }
                unsafe {
                    *scores.get_unchecked_mut(token * n_comp as usize + comp) = output;
                }
            }
            output_index += WMMA_THREADS as usize;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let n_tokens = 2_u32;
    let n_comp = 128_u32;

    let mut q_values = vec![0.0_f32; n_tokens as usize * N_HEAD * HEAD_DIM];
    for token in 0..n_tokens as usize {
        for head in 0..N_HEAD {
            q_values[(token * N_HEAD + head) * HEAD_DIM] = token as f32 + 1.0;
        }
    }
    let mut weights_values = vec![1.0_f32; n_tokens as usize * N_HEAD];
    for head in 0..N_HEAD {
        weights_values[N_HEAD + head] = 0.25;
    }
    let mut comp_values = vec![0.0_f32; n_comp as usize * HEAD_DIM];
    for comp in 0..n_comp as usize {
        comp_values[comp * HEAD_DIM] = match comp {
            126 => -1.0,
            127 => f32::NAN,
            _ => comp as f32 + 1.0,
        };
    }
    let q = substrate.upload(&q_values)?;
    let weights = substrate.upload(&weights_values)?;
    let index_comp = substrate.upload(&comp_values)?;

    let mut scores = substrate.zeroed::<f32>((n_tokens * n_comp) as usize)?;
    indexer_scores_wmma64(
        &module,
        substrate.stream(),
        &mut scores,
        &q,
        &weights,
        &index_comp,
        n_comp,
        n_tokens,
        0,
        1,
        0.5,
        false,
    )?;
    substrate.flush_commands()?;
    assert_eq!(
        substrate.download(&scores)?,
        expected_scores(n_tokens, n_comp, 0, 1, 0.5, false)
    );

    let mut causal_scores = substrate.zeroed::<f32>((n_tokens * n_comp) as usize)?;
    indexer_scores_wmma64(
        &module,
        substrate.stream(),
        &mut causal_scores,
        &q,
        &weights,
        &index_comp,
        n_comp,
        n_tokens,
        0,
        1,
        0.5,
        true,
    )?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&causal_scores)?,
        expected_scores(n_tokens, n_comp, 0, 1, 0.5, true)
    );

    assert!(matches!(
        indexer_scores_wmma64(
            &module,
            substrate.stream(),
            &mut scores,
            &q,
            &weights,
            &index_comp,
            n_comp,
            n_tokens,
            0,
            0,
            0.5,
            true,
        ),
        Err(IndexerWmma64Error::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2d2b2b\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"wmma64_output_matches\":true,\"causal_mask_output_matches\":true,\"four_warp_tile_mapping_matches\":true,\"weighted_epilogue_matches\":true,\"nan_negative_clamp_matches\":true,\"invalid_shape_rejected\":true,\"owns_indexer_scores_wmma64_kernel\":{},\"owns_wmma128_and_dispatch_priority\":{},\"owns_specialized_topk_dispatch\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2D2B2B_SCOPE.owns_indexer_scores_wmma64_kernel,
        M14_2D2B2B_SCOPE.owns_wmma128_and_dispatch_priority,
        M14_2D2B2B_SCOPE.owns_specialized_topk_dispatch,
        M14_2D2B2B_SCOPE.changes_default_route,
    );
    Ok(())
}

fn expected_scores(
    n_tokens: u32,
    n_comp: u32,
    pos0: u32,
    ratio: u32,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut expected = Vec::with_capacity((n_tokens * n_comp) as usize);
    for token in 0..n_tokens {
        for comp in 0..n_comp {
            if causal && comp >= (pos0 + token + 1) / ratio {
                expected.push(f32::NEG_INFINITY);
            } else {
                let component = match comp {
                    126 => -1.0,
                    127 => f32::NAN,
                    _ => comp as f32 + 1.0,
                };
                let dot = (token as f32 + 1.0) * component;
                let positive = if dot.is_nan() || dot <= 0.0 { 0.0 } else { dot };
                let weight = if token == 0 { 1.0 } else { 0.25 };
                expected.push(N_HEAD as f32 * positive * weight * scale);
            }
        }
    }
    expected
}

#[allow(clippy::too_many_arguments)]
fn indexer_scores_wmma64(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scores: &mut DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    index_comp: &DeviceBuffer<f32>,
    n_comp: u32,
    n_tokens: u32,
    pos0: u32,
    ratio: u32,
    scale: f32,
    causal: bool,
) -> Result<(), IndexerWmma64Error> {
    if n_comp == 0
        || n_tokens == 0
        || (causal && ratio == 0)
        || scores.len() < (n_tokens as usize * n_comp as usize)
        || q.len() < (n_tokens as usize * N_HEAD * HEAD_DIM)
        || weights.len() < (n_tokens as usize * N_HEAD)
        || index_comp.len() < (n_comp as usize * HEAD_DIM)
    {
        return Err(IndexerWmma64Error::InvalidShape);
    }
    module
        .indexer_scores_wmma64_kernel(
            stream,
            LaunchConfig {
                grid_dim: (
                    n_comp.div_ceil(WIDE_COMPONENTS as u32),
                    n_tokens.div_ceil(TILE_TOKENS as u32),
                    1,
                ),
                block_dim: (WMMA_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            n_comp,
            n_tokens,
            pos0,
            ratio,
            scale,
            causal as u32,
            q,
            weights,
            index_comp,
            scores,
        )
        .map_err(IndexerWmma64Error::Driver)
}

#[derive(Debug)]
enum IndexerWmma64Error {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for IndexerWmma64Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("WMMA64 indexer score shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexerWmma64Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
