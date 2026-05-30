use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{
    select_attention_decode_path, substrate::CudaOxideSubstrate, AttentionDecodeDispatchOptions,
    AttentionDecodePath, M14_4D3_SCOPE,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_decode_heads8_online_kernel(
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head_group = thread::blockIdx_y();
        if token >= n_tokens || head_dim != 512 || thread::threadIdx_x() != 0 {
            return;
        }
        let qpos = pos0 + token;
        let first_raw_pos = pos0 + n_tokens - n_raw;
        let mut raw_first = 0_u32;
        let mut raw_count = 0_u32;
        if n_raw != 0 {
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
        }
        let mut comp_count = 0_u32;
        if n_comp != 0 {
            if n_tokens == 1 && ratio == 0 {
                comp_count = n_comp;
            } else if ratio != 0 {
                comp_count = (qpos + 1) / ratio;
                if comp_count > n_comp {
                    comp_count = n_comp;
                }
            }
        }
        let mut local_head = 0_u32;
        while local_head < 8 {
            let head = head_group * 8 + local_head;
            if head < n_head {
                write_head(
                    token, head, raw_first, raw_count, comp_count, raw_cap, raw_start, n_head,
                    head_dim, sinks, q, raw_kv, comp_kv, &mut heads,
                );
            }
            local_head += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_head(
        token: u32,
        head: u32,
        raw_first: u32,
        raw_count: u32,
        comp_count: u32,
        raw_cap: u32,
        raw_start: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        heads: &mut DisjointSlice<f32>,
    ) {
        let query_base = ((token * n_head + head) * head_dim) as usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut max_score = f32::NEG_INFINITY;
        let mut denominator = 0.0_f32;
        let mut row = 0_u32;
        while row < raw_count {
            let raw_row = (raw_start + raw_first + row) % raw_cap;
            let score = dot(q, query_base, raw_kv, raw_row, head_dim) * scale;
            let next_max = maximum(max_score, score);
            denominator = denominator * (max_score - next_max).exp() + (score - next_max).exp();
            max_score = next_max;
            row += 1;
        }
        let mut compressed = 0_u32;
        while compressed < comp_count {
            let score = dot(q, query_base, comp_kv, compressed, head_dim) * scale;
            let next_max = maximum(max_score, score);
            denominator = denominator * (max_score - next_max).exp() + (score - next_max).exp();
            max_score = next_max;
            compressed += 1;
        }
        let sink = sinks[head as usize];
        let next_max = maximum(max_score, sink);
        denominator = denominator * (max_score - next_max).exp() + (sink - next_max).exp();
        max_score = next_max;
        let mut dimension = 0_u32;
        while dimension < head_dim {
            let mut numerator = 0.0_f32;
            row = 0;
            while row < raw_count {
                let raw_row = (raw_start + raw_first + row) % raw_cap;
                let score = dot(q, query_base, raw_kv, raw_row, head_dim) * scale;
                numerator +=
                    raw_kv[(raw_row * head_dim + dimension) as usize] * (score - max_score).exp();
                row += 1;
            }
            compressed = 0;
            while compressed < comp_count {
                let score = dot(q, query_base, comp_kv, compressed, head_dim) * scale;
                numerator += comp_kv[(compressed * head_dim + dimension) as usize]
                    * (score - max_score).exp();
                compressed += 1;
            }
            unsafe {
                *heads
                    .get_unchecked_mut(((token * n_head + head) * head_dim + dimension) as usize) =
                    numerator / denominator;
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
const HEAD_DIM: u32 = 512;
const N_HEAD: u32 = 9;
const N_TOKENS: u32 = 3;
const POS0: u32 = 4;
const RAW_CAP: u32 = 6;
const N_RAW: u32 = 4;
const RAW_START: u32 = 5;
const N_COMP: u32 = 3;
const WINDOW: u32 = 3;
const RATIO: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_decode_heads8_online_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let sinks_host = values(N_HEAD as usize, 11, -0.75);
    let q_host = values((N_TOKENS * N_HEAD * HEAD_DIM) as usize, 17, -0.875);
    let raw_host = values((RAW_CAP * HEAD_DIM) as usize, 23, -1.25);
    let comp_host = values((N_COMP * HEAD_DIM) as usize, 29, -1.0);
    let sinks = substrate.upload(&sinks_host)?;
    let q = substrate.upload(&q_host)?;
    let raw = substrate.upload(&raw_host)?;
    let comp = substrate.upload(&comp_host)?;

    let batch = AttentionCase {
        n_tokens: N_TOKENS,
        pos0: POS0,
        n_raw: N_RAW,
        raw_cap: RAW_CAP,
        raw_start: RAW_START,
        n_comp: N_COMP,
        window: WINDOW,
        ratio: RATIO,
        n_head: N_HEAD,
        head_dim: HEAD_DIM,
    };
    let mut batch_output = substrate.zeroed::<f32>(batch.output_len())?;
    attention_decode_heads8_online_tensor(
        &module,
        substrate.stream(),
        &batch,
        &sinks,
        &q,
        &raw,
        &comp,
        &mut batch_output,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&batch_output)?,
        &expected_attention(&batch, &sinks_host, &q_host, &raw_host, &comp_host),
        2.0e-5,
    );

    let single = AttentionCase {
        n_tokens: 1,
        pos0: 0,
        n_raw: 1,
        raw_cap: RAW_CAP,
        raw_start: 3,
        n_comp: N_COMP,
        window: 0,
        ratio: 0,
        n_head: N_HEAD,
        head_dim: HEAD_DIM,
    };
    let mut single_output = substrate.zeroed::<f32>(single.output_len())?;
    attention_decode_heads8_online_tensor(
        &module,
        substrate.stream(),
        &single,
        &sinks,
        &q,
        &raw,
        &comp,
        &mut single_output,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&single_output)?,
        &expected_attention(&single, &sinks_host, &q_host, &raw_host, &comp_host),
        2.0e-5,
    );

    assert_eq!(
        select_attention_decode_path(AttentionDecodeDispatchOptions {
            n_tokens: 1,
            n_comp: 7937,
            use_comp_mask: false,
            head_dim: HEAD_DIM,
            no_window_attention: false,
            window_attention: false,
            quality_mode: true,
        }),
        AttentionDecodePath::Heads8OnlineOverflow
    );
    assert_eq!(
        select_attention_decode_path(AttentionDecodeDispatchOptions {
            n_tokens: N_TOKENS,
            n_comp: N_COMP,
            use_comp_mask: false,
            head_dim: HEAD_DIM,
            no_window_attention: false,
            window_attention: true,
            quality_mode: false,
        }),
        AttentionDecodePath::Heads8OnlineWindow
    );

    let mut invalid_output = substrate.zeroed::<f32>(batch.output_len())?;
    let short_q = substrate.zeroed::<f32>(batch.output_len() - 1)?;
    assert!(matches!(
        attention_decode_heads8_online_tensor(
            &module,
            substrate.stream(),
            &batch,
            &sinks,
            &short_q,
            &raw,
            &comp,
            &mut invalid_output,
        ),
        Err(AttentionOnlineError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4d3\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"heads8_online_batch_output_matches\":true,\"heads8_online_single_all_output_matches\":true,\"partial_head_group_matches\":true,\"causal_window_matches\":true,\"ring_wrapped_raw_rows_match\":true,\"visible_compressed_limit_matches\":true,\"sink_softmax_matches\":true,\"overflow_dispatch_matches\":true,\"window_dispatch_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_heads8_online_decode_kernel\":{},\"owns_decode_online_dispatch_policy\":{},\"owns_prefill_or_indexed_online_attention\":{},\"owns_output_q8_attention\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D3_SCOPE.owns_heads8_online_decode_kernel,
        M14_4D3_SCOPE.owns_decode_online_dispatch_policy,
        M14_4D3_SCOPE.owns_prefill_or_indexed_online_attention,
        M14_4D3_SCOPE.owns_output_q8_attention,
        M14_4D3_SCOPE.owns_runtime_graph_integration,
        M14_4D3_SCOPE.changes_default_route,
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct AttentionCase {
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
}

impl AttentionCase {
    const fn output_len(self) -> usize {
        (self.n_tokens * self.n_head * self.head_dim) as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_decode_heads8_online_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    case: &AttentionCase,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOnlineError> {
    if case.n_tokens == 0
        || case.n_raw == 0
        || case.raw_cap < case.n_raw
        || case.raw_start >= case.raw_cap
        || case.head_dim != 512
        || (case.n_comp != 0 && case.n_tokens != 1 && case.ratio == 0)
        || sinks.len() < case.n_head as usize
        || q.len() < case.output_len()
        || raw.len() < (case.raw_cap * case.head_dim) as usize
        || comp.len() < (case.n_comp * case.head_dim) as usize
        || heads.len() < case.output_len()
    {
        return Err(AttentionOnlineError::InvalidShape);
    }
    module
        .attention_decode_heads8_online_kernel(
            stream,
            LaunchConfig {
                grid_dim: (case.n_tokens, case.n_head.div_ceil(8), 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            case.n_tokens,
            case.pos0,
            case.n_raw,
            case.raw_cap,
            case.raw_start,
            case.n_comp,
            case.window,
            case.ratio,
            case.n_head,
            case.head_dim,
            sinks,
            q,
            raw,
            comp,
            heads,
        )
        .map_err(AttentionOnlineError::Driver)
}

fn expected_attention(
    case: &AttentionCase,
    sinks: &[f32],
    q: &[f32],
    raw: &[f32],
    comp: &[f32],
) -> Vec<f32> {
    let scale = 1.0_f32 / (case.head_dim as f32).sqrt();
    let first_raw_pos = case.pos0 + case.n_tokens - case.n_raw;
    let mut heads = vec![0.0_f32; case.output_len()];
    for token in 0..case.n_tokens as usize {
        let qpos = case.pos0 + token as u32;
        let mut raw_first = 0;
        let mut raw_count = 0;
        if qpos >= first_raw_pos {
            let mut lo = first_raw_pos;
            if case.window != 0 && qpos + 1 > case.window {
                lo = lo.max(qpos + 1 - case.window);
            }
            let hi = qpos.min(first_raw_pos + case.n_raw - 1);
            if hi >= lo {
                raw_first = lo - first_raw_pos;
                raw_count = (hi - lo + 1).min(256);
            }
        }
        let comp_count = if case.n_comp == 0 {
            0
        } else if case.n_tokens == 1 && case.ratio == 0 {
            case.n_comp
        } else {
            ((qpos + 1) / case.ratio).min(case.n_comp)
        };
        for head in 0..case.n_head as usize {
            let query_base = (token * case.n_head as usize + head) * case.head_dim as usize;
            let query = &q[query_base..query_base + case.head_dim as usize];
            let mut rows = Vec::new();
            for offset in 0..raw_count {
                let row = (case.raw_start + raw_first + offset) % case.raw_cap;
                let value = &raw[row as usize * case.head_dim as usize
                    ..(row as usize + 1) * case.head_dim as usize];
                rows.push((value, dot_host(query, value) * scale));
            }
            for compressed in 0..comp_count as usize {
                let value = &comp[compressed * case.head_dim as usize
                    ..(compressed + 1) * case.head_dim as usize];
                rows.push((value, dot_host(query, value) * scale));
            }
            let mut max_score = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            for (_, score) in &rows {
                let next_max = max_score.max(*score);
                denominator =
                    denominator * (max_score - next_max).exp() + (*score - next_max).exp();
                max_score = next_max;
            }
            let next_max = max_score.max(sinks[head]);
            denominator =
                denominator * (max_score - next_max).exp() + (sinks[head] - next_max).exp();
            max_score = next_max;
            for dimension in 0..case.head_dim as usize {
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

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.015625 + offset)
        .collect()
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
enum AttentionOnlineError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for AttentionOnlineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("heads8 online attention tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionOnlineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
