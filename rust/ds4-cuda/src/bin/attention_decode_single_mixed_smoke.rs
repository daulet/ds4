use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4D1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_decode_single_mixed_kernel(
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        use_mask: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let head = thread::blockIdx_x();
        if head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let query_base = (head * head_dim) as usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut max_score = sinks[head as usize];
        let mut raw_row = 0_u32;
        while raw_row < n_raw {
            let row = (raw_start + raw_row) % raw_cap;
            max_score = maximum(max_score, dot(q, query_base, raw_kv, row, head_dim) * scale);
            raw_row += 1;
        }
        let mut compressed = 0_u32;
        while compressed < n_comp {
            let add = if use_mask != 0 {
                comp_mask[compressed as usize]
            } else {
                0.0
            };
            if add > -1.0e20 {
                max_score = maximum(
                    max_score,
                    dot(q, query_base, comp_kv, compressed, head_dim) * scale + add,
                );
            }
            compressed += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        raw_row = 0;
        while raw_row < n_raw {
            let row = (raw_start + raw_row) % raw_cap;
            denominator += (dot(q, query_base, raw_kv, row, head_dim) * scale - max_score).exp();
            raw_row += 1;
        }
        compressed = 0;
        while compressed < n_comp {
            let add = if use_mask != 0 {
                comp_mask[compressed as usize]
            } else {
                0.0
            };
            if add > -1.0e20 {
                denominator += (dot(q, query_base, comp_kv, compressed, head_dim) * scale + add
                    - max_score)
                    .exp();
            }
            compressed += 1;
        }
        let mut dimension = 0_u32;
        while dimension < head_dim {
            let mut accumulator = 0.0_f32;
            raw_row = 0;
            while raw_row < n_raw {
                let row = (raw_start + raw_row) % raw_cap;
                let score = (dot(q, query_base, raw_kv, row, head_dim) * scale - max_score).exp();
                accumulator += raw_kv[(row * head_dim + dimension) as usize] * score;
                raw_row += 1;
            }
            compressed = 0;
            while compressed < n_comp {
                let add = if use_mask != 0 {
                    comp_mask[compressed as usize]
                } else {
                    0.0
                };
                if add > -1.0e20 {
                    let score = (dot(q, query_base, comp_kv, compressed, head_dim) * scale + add
                        - max_score)
                        .exp();
                    accumulator += comp_kv[(compressed * head_dim + dimension) as usize] * score;
                }
                compressed += 1;
            }
            unsafe {
                *heads.get_unchecked_mut((head * head_dim + dimension) as usize) =
                    accumulator / denominator;
            }
            dimension += 1;
        }
    }

    fn dot(q: &[f32], query_base: usize, kv: &[f32], row: u32, head_dim: u32) -> f32 {
        let mut value = 0.0_f32;
        let mut dimension = 0_u32;
        while dimension < head_dim {
            value += q[query_base + dimension as usize] * kv[(row * head_dim + dimension) as usize];
            dimension += 1;
        }
        value
    }

    fn maximum(left: f32, right: f32) -> f32 {
        if right > left {
            right
        } else {
            left
        }
    }
}

const THREADS: u32 = 256;
const N_HEAD: u32 = 2;
const HEAD_DIM: u32 = 7;
const RAW_CAP: u32 = 5;
const N_RAW: u32 = 3;
const RAW_START: u32 = 4;
const N_COMP: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_decode_single_mixed_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let sink_values = vec![-0.5_f32, 0.375];
    let q_values = values((N_HEAD * HEAD_DIM) as usize, 17, -1.0);
    let raw_values = values((RAW_CAP * HEAD_DIM) as usize, 23, -1.5);
    let comp_values = values((N_COMP * HEAD_DIM) as usize, 29, -1.25);
    let mask_values = vec![-0.25_f32, -1.0e30];
    let sinks = substrate.upload(&sink_values)?;
    let q = substrate.upload(&q_values)?;
    let raw = substrate.upload(&raw_values)?;
    let comp = substrate.upload(&comp_values)?;
    let mask = substrate.upload(&mask_values)?;

    let mut masked = substrate.zeroed::<f32>((N_HEAD * HEAD_DIM) as usize)?;
    attention_decode_heads_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &mask,
        true,
        N_COMP,
        &mut masked,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&masked)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &mask_values,
            true,
            N_COMP,
        ),
        2.0e-5,
    );

    let mut unmasked = substrate.zeroed::<f32>((N_HEAD * HEAD_DIM) as usize)?;
    attention_decode_heads_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &mask,
        false,
        N_COMP,
        &mut unmasked,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&unmasked)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &mask_values,
            false,
            N_COMP,
        ),
        2.0e-5,
    );

    let mut raw_only = substrate.zeroed::<f32>((N_HEAD * HEAD_DIM) as usize)?;
    attention_decode_heads_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &mask,
        false,
        0,
        &mut raw_only,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&raw_only)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &mask_values,
            false,
            0,
        ),
        2.0e-5,
    );

    let short_raw = substrate.zeroed::<f32>((RAW_CAP * HEAD_DIM - 1) as usize)?;
    let mut output = substrate.zeroed::<f32>((N_HEAD * HEAD_DIM) as usize)?;
    assert!(matches!(
        attention_decode_heads_tensor(
            &module,
            substrate.stream(),
            &sinks,
            &q,
            &short_raw,
            &comp,
            &mask,
            true,
            N_COMP,
            &mut output,
        ),
        Err(AttentionDecodeError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4d1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"single_token_mixed_output_matches\":true,\"ring_wrapped_raw_rows_match\":true,\"compressed_mask_matches\":true,\"sink_softmax_matches\":true,\"raw_only_output_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_attention_decode_heads_surface\":{},\"owns_single_token_ring_raw_and_compressed_attention\":{},\"owns_masked_compressed_rows\":{},\"owns_batched_or_online_decode\":{},\"owns_prefill_indexed_or_output_q8_attention\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D1_SCOPE.owns_attention_decode_heads_surface,
        M14_4D1_SCOPE.owns_single_token_ring_raw_and_compressed_attention,
        M14_4D1_SCOPE.owns_masked_compressed_rows,
        M14_4D1_SCOPE.owns_batched_or_online_decode,
        M14_4D1_SCOPE.owns_prefill_indexed_or_output_q8_attention,
        M14_4D1_SCOPE.owns_runtime_graph_integration,
        M14_4D1_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention_decode_heads_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    mask: &DeviceBuffer<f32>,
    use_mask: bool,
    n_comp: u32,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionDecodeError> {
    if sinks.len() < N_HEAD as usize
        || q.len() < (N_HEAD * HEAD_DIM) as usize
        || raw.len() < (RAW_CAP * HEAD_DIM) as usize
        || heads.len() < (N_HEAD * HEAD_DIM) as usize
        || n_comp > N_COMP
        || (n_comp != 0 && comp.len() < (n_comp * HEAD_DIM) as usize)
        || (use_mask && mask.len() < n_comp as usize)
        || N_RAW == 0
        || RAW_CAP < N_RAW
        || RAW_START >= RAW_CAP
    {
        return Err(AttentionDecodeError::InvalidShape);
    }
    module
        .attention_decode_single_mixed_kernel(
            stream,
            LaunchConfig {
                grid_dim: (N_HEAD, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_RAW,
            RAW_CAP,
            RAW_START,
            n_comp,
            u32::from(use_mask),
            N_HEAD,
            HEAD_DIM,
            sinks,
            q,
            raw,
            comp,
            mask,
            heads,
        )
        .map_err(AttentionDecodeError::Driver)
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
        .collect()
}

fn expected_attention(
    sinks: &[f32],
    q: &[f32],
    raw: &[f32],
    comp: &[f32],
    mask: &[f32],
    use_mask: bool,
    n_comp: u32,
) -> Vec<f32> {
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let mut heads = vec![0.0_f32; (N_HEAD * HEAD_DIM) as usize];
    for head in 0..N_HEAD as usize {
        let query = &q[head * HEAD_DIM as usize..(head + 1) * HEAD_DIM as usize];
        let mut rows: Vec<(&[f32], f32)> = (0..N_RAW)
            .map(|raw_row| {
                let row = (RAW_START + raw_row) % RAW_CAP;
                let value =
                    &raw[row as usize * HEAD_DIM as usize..(row as usize + 1) * HEAD_DIM as usize];
                (value, dot_host(query, value) * scale)
            })
            .collect();
        for compressed in 0..n_comp as usize {
            let add = if use_mask { mask[compressed] } else { 0.0 };
            if add > -1.0e20 {
                let value =
                    &comp[compressed * HEAD_DIM as usize..(compressed + 1) * HEAD_DIM as usize];
                rows.push((value, dot_host(query, value) * scale + add));
            }
        }
        let max_score = rows
            .iter()
            .map(|(_, score)| *score)
            .fold(sinks[head], f32::max);
        let denominator = (sinks[head] - max_score).exp()
            + rows
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
        for dimension in 0..HEAD_DIM as usize {
            heads[head * HEAD_DIM as usize + dimension] = rows
                .iter()
                .map(|(row, score)| row[dimension] * (*score - max_score).exp())
                .sum::<f32>()
                / denominator;
        }
    }
    heads
}

fn dot_host(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
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
enum AttentionDecodeError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for AttentionDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("single-token attention tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
