use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4D6_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_indexed_mixed_kernel(
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        top_k: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        mut heads: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let qpos = pos0 + token;
        let first_raw_pos = pos0 + n_tokens - n_raw;
        let mut raw_first = 0_u32;
        let mut raw_count = 0_u32;
        let raw_last_pos = first_raw_pos + n_raw - 1;
        if qpos >= first_raw_pos {
            let mut lo = first_raw_pos;
            if window != 0 && qpos + 1 > window {
                let window_lo = qpos + 1 - window;
                if window_lo > lo {
                    lo = window_lo;
                }
            }
            let hi = if qpos < raw_last_pos {
                qpos
            } else {
                raw_last_pos
            };
            if hi >= lo {
                raw_first = lo - first_raw_pos;
                raw_count = hi - lo + 1;
                if raw_count > 256 {
                    raw_count = 256;
                }
            }
        }
        let mut visible_comp = n_comp;
        if ratio != 0 {
            visible_comp = (qpos + 1) / ratio;
            if visible_comp > n_comp {
                visible_comp = n_comp;
            }
        }
        let query_base = ((token * n_head + head) * head_dim) as usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut max_score = sinks[head as usize];
        let mut row = 0_u32;
        while row < raw_count {
            let raw_row = (raw_start + raw_first + row) % raw_cap;
            max_score = maximum(
                max_score,
                dot(q, query_base, raw_kv, raw_row, head_dim) * scale,
            );
            row += 1;
        }
        let mut selected = 0_u32;
        while selected < top_k {
            let compressed = topk[(token * top_k + selected) as usize];
            if compressed >= 0 && (compressed as u32) < visible_comp {
                max_score = maximum(
                    max_score,
                    dot(q, query_base, comp_kv, compressed as u32, head_dim) * scale,
                );
            }
            selected += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        row = 0;
        while row < raw_count {
            let raw_row = (raw_start + raw_first + row) % raw_cap;
            denominator +=
                (dot(q, query_base, raw_kv, raw_row, head_dim) * scale - max_score).exp();
            row += 1;
        }
        selected = 0;
        while selected < top_k {
            let compressed = topk[(token * top_k + selected) as usize];
            if compressed >= 0 && (compressed as u32) < visible_comp {
                denominator += (dot(q, query_base, comp_kv, compressed as u32, head_dim) * scale
                    - max_score)
                    .exp();
            }
            selected += 1;
        }
        let mut dimension = 0_u32;
        while dimension < head_dim {
            let mut accumulator = 0.0_f32;
            row = 0;
            while row < raw_count {
                let raw_row = (raw_start + raw_first + row) % raw_cap;
                let score =
                    (dot(q, query_base, raw_kv, raw_row, head_dim) * scale - max_score).exp();
                accumulator += raw_kv[(raw_row * head_dim + dimension) as usize] * score;
                row += 1;
            }
            selected = 0;
            while selected < top_k {
                let compressed = topk[(token * top_k + selected) as usize];
                if compressed >= 0 && (compressed as u32) < visible_comp {
                    let score = (dot(q, query_base, comp_kv, compressed as u32, head_dim) * scale
                        - max_score)
                        .exp();
                    accumulator +=
                        comp_kv[(compressed as u32 * head_dim + dimension) as usize] * score;
                }
                selected += 1;
            }
            unsafe {
                *heads
                    .get_unchecked_mut(((token * n_head + head) * head_dim + dimension) as usize) =
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
const N_TOKENS: u32 = 3;
const POS0: u32 = 4;
const N_HEAD: u32 = 2;
const HEAD_DIM: u32 = 7;
const RAW_CAP: u32 = 6;
const N_RAW: u32 = 4;
const RAW_START: u32 = 5;
const N_COMP: u32 = 5;
const TOP_K: u32 = 5;
const WINDOW: u32 = 3;
const RATIO: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_indexed_generic_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let sink_values = vec![-0.5_f32, 0.375];
    let q_values = values((N_TOKENS * N_HEAD * HEAD_DIM) as usize, 17, -1.0);
    let raw_values = values((RAW_CAP * HEAD_DIM) as usize, 23, -1.5);
    let comp_values = values((N_COMP * HEAD_DIM) as usize, 29, -1.25);
    let topk_values = vec![1_i32, 0, -1, 4, 1, 2, -1, 0, 4, 1, 4, 1, 2, -1, 0];
    let sinks = substrate.upload(&sink_values)?;
    let q = substrate.upload(&q_values)?;
    let raw = substrate.upload(&raw_values)?;
    let comp = substrate.upload(&comp_values)?;
    let topk = substrate.upload(&topk_values)?;

    let mut indexed = substrate.zeroed::<f32>(output_len())?;
    attention_indexed_mixed_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &topk,
        TOP_K,
        RATIO,
        &mut indexed,
    )?;
    substrate.end_commands()?;
    let indexed_values = substrate.download(&indexed)?;
    assert_close(
        &indexed_values,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &topk_values,
            TOP_K,
            RATIO,
        ),
        2.0e-5,
    );

    let mut ratio_zero = substrate.zeroed::<f32>(output_len())?;
    attention_indexed_mixed_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &topk,
        TOP_K,
        0,
        &mut ratio_zero,
    )?;
    substrate.end_commands()?;
    let ratio_zero_values = substrate.download(&ratio_zero)?;
    assert_close(
        &ratio_zero_values,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &topk_values,
            TOP_K,
            0,
        ),
        2.0e-5,
    );
    assert!(indexed_values
        .iter()
        .zip(&ratio_zero_values)
        .any(|(indexed, ratio_zero)| (indexed - ratio_zero).abs() > 1.0e-5));

    let mut invalid = substrate.zeroed::<f32>(output_len())?;
    assert!(matches!(
        attention_indexed_mixed_tensor(
            &module,
            substrate.stream(),
            &sinks,
            &q,
            &raw,
            &comp,
            &topk,
            0,
            RATIO,
            &mut invalid,
        ),
        Err(AttentionIndexedError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4d6\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"indexed_output_matches\":true,\"ratio_zero_all_compressed_matches\":true,\"topk_filter_order_and_duplicates_match\":true,\"causal_window_matches\":true,\"ring_wrapped_raw_rows_match\":true,\"visible_compressed_limit_matches\":true,\"sink_softmax_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_attention_indexed_mixed_surface\":{},\"owns_generic_indexed_kernel\":{},\"owns_topk_filter_and_order_semantics\":{},\"owns_indexed_sort_or_heads8_dispatch\":{},\"owns_output_q8_attention\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D6_SCOPE.owns_attention_indexed_mixed_surface,
        M14_4D6_SCOPE.owns_generic_indexed_kernel,
        M14_4D6_SCOPE.owns_topk_filter_and_order_semantics,
        M14_4D6_SCOPE.owns_indexed_sort_or_heads8_dispatch,
        M14_4D6_SCOPE.owns_output_q8_attention,
        M14_4D6_SCOPE.owns_runtime_graph_integration,
        M14_4D6_SCOPE.changes_default_route,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention_indexed_mixed_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    top_k: u32,
    ratio: u32,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionIndexedError> {
    if sinks.len() < N_HEAD as usize
        || q.len() < output_len()
        || raw.len() < (RAW_CAP * HEAD_DIM) as usize
        || comp.len() < (N_COMP * HEAD_DIM) as usize
        || topk.len() < (N_TOKENS * top_k) as usize
        || heads.len() < output_len()
        || N_RAW == 0
        || RAW_CAP < N_RAW
        || RAW_START >= RAW_CAP
        || N_COMP == 0
        || top_k == 0
        || top_k > 512
    {
        return Err(AttentionIndexedError::InvalidShape);
    }
    module
        .attention_indexed_mixed_kernel(
            stream,
            LaunchConfig {
                grid_dim: (N_TOKENS, N_HEAD, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            POS0,
            N_RAW,
            RAW_CAP,
            RAW_START,
            N_COMP,
            top_k,
            WINDOW,
            ratio,
            N_HEAD,
            HEAD_DIM,
            sinks,
            q,
            raw,
            comp,
            topk,
            heads,
        )
        .map_err(AttentionIndexedError::Driver)
}

fn output_len() -> usize {
    (N_TOKENS * N_HEAD * HEAD_DIM) as usize
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn expected_attention(
    sinks: &[f32],
    q: &[f32],
    raw: &[f32],
    comp: &[f32],
    topk: &[i32],
    top_k: u32,
    ratio: u32,
) -> Vec<f32> {
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let first_raw_pos = POS0 + N_TOKENS - N_RAW;
    let mut heads = vec![0.0_f32; output_len()];
    for token in 0..N_TOKENS as usize {
        let qpos = POS0 + token as u32;
        let mut raw_first = 0;
        let mut raw_count = 0;
        if qpos >= first_raw_pos {
            let lo = first_raw_pos.max(qpos.saturating_add(1).saturating_sub(WINDOW));
            let hi = qpos.min(first_raw_pos + N_RAW - 1);
            if hi >= lo {
                raw_first = lo - first_raw_pos;
                raw_count = hi - lo + 1;
            }
        }
        let visible_comp = if ratio == 0 {
            N_COMP
        } else {
            ((qpos + 1) / ratio).min(N_COMP)
        };
        for head in 0..N_HEAD as usize {
            let query_base = (token * N_HEAD as usize + head) * HEAD_DIM as usize;
            let query = &q[query_base..query_base + HEAD_DIM as usize];
            let mut rows: Vec<(&[f32], f32)> = (0..raw_count)
                .map(|offset| {
                    let row = (RAW_START + raw_first + offset) % RAW_CAP;
                    let value = &raw
                        [row as usize * HEAD_DIM as usize..(row as usize + 1) * HEAD_DIM as usize];
                    (value, dot_host(query, value) * scale)
                })
                .collect();
            for selected in 0..top_k as usize {
                let compressed = topk[token * top_k as usize + selected];
                if compressed >= 0 && (compressed as u32) < visible_comp {
                    let compressed = compressed as usize;
                    let value =
                        &comp[compressed * HEAD_DIM as usize..(compressed + 1) * HEAD_DIM as usize];
                    rows.push((value, dot_host(query, value) * scale));
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
                heads[query_base + dimension] = rows
                    .iter()
                    .map(|(row, score)| row[dimension] * (*score - max_score).exp())
                    .sum::<f32>()
                    / denominator;
            }
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
enum AttentionIndexedError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for AttentionIndexedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("indexed mixed attention shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionIndexedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
