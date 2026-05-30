use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{
    select_attention_indexed_path, should_sort_indexed_topk, substrate::CudaOxideSubstrate,
    AttentionIndexedDispatchOptions, AttentionIndexedPath, IndexedTopkSortOptions, M14_4D7_SCOPE,
};

const SORTED_TOP_K: u32 = 512;
const FILTERED_TOP_K: u32 = 5;
const SORT_THREADS: u32 = 512;
const ONLINE_THREADS: u32 = 512;
const RB4_THREADS: u32 = 256;
const N_TOKENS: u32 = 3;
const POS0: u32 = 8;
const N_HEAD: u32 = 17;
const HEAD_DIM: u32 = 512;
const RAW_CAP: u32 = 6;
const N_RAW: u32 = 4;
const RAW_START: u32 = 5;
const N_COMP: u32 = 512;
const WINDOW: u32 = 3;
const RATIO: u32 = 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn indexed_topk_sort_512_asc_kernel(
        n_tokens: u32,
        src: &[i32],
        mut dst: DisjointSlice<i32>,
    ) {
        static mut ROWS: SharedArray<i32, { SORTED_TOP_K as usize }> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if token >= n_tokens || tid >= SORTED_TOP_K {
            return;
        }
        let index = tid as usize;
        let offset = token as usize * SORTED_TOP_K as usize + index;
        unsafe {
            ROWS[index] = src[offset];
        }
        thread::sync_threads();
        let mut k = 2_u32;
        while k <= SORTED_TOP_K {
            let mut j = k >> 1;
            while j > 0 {
                let other = tid ^ j;
                if other > tid && other < SORTED_TOP_K {
                    let other_index = other as usize;
                    let a = unsafe { ROWS[index] };
                    let b = unsafe { ROWS[other_index] };
                    let ascending = (tid & k) == 0;
                    if (ascending && a > b) || (!ascending && a < b) {
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

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_indexed_mixed_heads8_online_kernel(
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
        let head_group = thread::blockIdx_y();
        if token >= n_tokens || head_dim != 512 || thread::threadIdx_x() != 0 {
            return;
        }
        let mut local_head = 0_u32;
        while local_head < 16 {
            let head = head_group * 16 + local_head;
            if head < n_head {
                write_indexed_head(
                    token, head, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window,
                    ratio, n_head, head_dim, 0, sinks, q, raw_kv, comp_kv, topk, &mut heads,
                );
            }
            local_head += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_indexed_mixed_heads8_rb4_kernel(
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
        let head_group = thread::blockIdx_y();
        if token >= n_tokens || head_dim != 512 || thread::threadIdx_x() != 0 {
            return;
        }
        let mut local_head = 0_u32;
        while local_head < 8 {
            let head = head_group * 8 + local_head;
            if head < n_head {
                write_indexed_head(
                    token, head, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window,
                    ratio, n_head, head_dim, 1, sinks, q, raw_kv, comp_kv, topk, &mut heads,
                );
            }
            local_head += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_indexed_head(
        token: u32,
        head: u32,
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
        filter_entries: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        topk: &[i32],
        heads: &mut DisjointSlice<f32>,
    ) {
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
        let selected_count = if filter_entries == 0 && visible_comp < top_k {
            visible_comp
        } else {
            top_k
        };
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
        while selected < selected_count {
            let compressed = topk[(token * top_k + selected) as usize];
            if filter_entries == 0 || (compressed >= 0 && (compressed as u32) < visible_comp) {
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
        while selected < selected_count {
            let compressed = topk[(token * top_k + selected) as usize];
            if filter_entries == 0 || (compressed >= 0 && (compressed as u32) < visible_comp) {
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
            while selected < selected_count {
                let compressed = topk[(token * top_k + selected) as usize];
                if filter_entries == 0 || (compressed >= 0 && (compressed as u32) < visible_comp) {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_indexed_optimized_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let sink_values = values(N_HEAD as usize, 11, -0.75);
    let q_values = values((N_TOKENS * N_HEAD * HEAD_DIM) as usize, 17, -0.875);
    let raw_values = values((RAW_CAP * HEAD_DIM) as usize, 23, -1.25);
    let comp_values = values((N_COMP * HEAD_DIM) as usize, 29, -1.0);
    let sinks = substrate.upload(&sink_values)?;
    let q = substrate.upload(&q_values)?;
    let raw = substrate.upload(&raw_values)?;
    let comp = substrate.upload(&comp_values)?;

    assert!(should_sort_indexed_topk(IndexedTopkSortOptions {
        n_tokens: N_TOKENS,
        top_k: SORTED_TOP_K,
        no_indexed_topk_sort: false,
    }));
    assert_eq!(
        select_attention_indexed_path(AttentionIndexedDispatchOptions {
            n_tokens: N_TOKENS,
            head_dim: HEAD_DIM,
            top_k: SORTED_TOP_K,
            no_indexed_heads8: false,
            indexed_twopass: false,
        }),
        AttentionIndexedPath::Heads8Online
    );
    let unsorted_values = build_unsorted_rows();
    let unsorted = substrate.upload(&unsorted_values)?;
    let mut sorted = substrate.zeroed::<i32>(unsorted_values.len())?;
    indexed_sort_tensor(&module, substrate.stream(), &unsorted, &mut sorted)?;
    let mut online = substrate.zeroed::<f32>(output_len())?;
    attention_online_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &sorted,
        &mut online,
    )?;
    substrate.end_commands()?;
    let sorted_values = substrate.download(&sorted)?;
    assert_sorted_rows(&sorted_values);
    assert_close(
        &substrate.download(&online)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &sorted_values,
            SORTED_TOP_K,
            false,
        ),
    );

    assert_eq!(
        select_attention_indexed_path(AttentionIndexedDispatchOptions {
            n_tokens: N_TOKENS,
            head_dim: HEAD_DIM,
            top_k: FILTERED_TOP_K,
            no_indexed_heads8: false,
            indexed_twopass: true,
        }),
        AttentionIndexedPath::Heads8Rb4
    );
    let filtered_values = vec![3_i32, 0, -1, 17, 3, 4, -1, 0, 17, 2, 8, 1, 4, -1, 0];
    let filtered = substrate.upload(&filtered_values)?;
    let mut rb4 = substrate.zeroed::<f32>(output_len())?;
    attention_rb4_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &filtered,
        &mut rb4,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&rb4)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &filtered_values,
            FILTERED_TOP_K,
            true,
        ),
    );
    assert_dispatch_fallbacks();

    println!(
        "{{\"milestone\":\"M14.4d7\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"sorted_online_output_matches\":true,\"rb4_filtered_output_matches\":true,\"integrated_sort_path_matches\":true,\"partial_head_group_matches\":true,\"dispatch_priority_matches\":true,\"causal_window_matches\":true,\"ring_wrapped_raw_rows_match\":true,\"visible_compressed_limit_matches\":true,\"topk_filter_order_and_duplicates_match\":true,\"sink_softmax_matches\":true,\"uses_libdevice_link_path\":true,\"consumes_indexed_topk_sort_policy\":{},\"owns_indexed_heads8_online_kernel\":{},\"owns_indexed_heads8_rb4_kernel\":{},\"owns_indexed_attention_dispatch_policy\":{},\"owns_output_q8_attention\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D7_SCOPE.consumes_indexed_topk_sort_policy,
        M14_4D7_SCOPE.owns_indexed_heads8_online_kernel,
        M14_4D7_SCOPE.owns_indexed_heads8_rb4_kernel,
        M14_4D7_SCOPE.owns_indexed_attention_dispatch_policy,
        M14_4D7_SCOPE.owns_output_q8_attention,
        M14_4D7_SCOPE.owns_runtime_graph_integration,
        M14_4D7_SCOPE.changes_default_route,
    );
    Ok(())
}

fn build_unsorted_rows() -> Vec<i32> {
    let mut rows = Vec::with_capacity((N_TOKENS * SORTED_TOP_K) as usize);
    rows.extend((0..SORTED_TOP_K as i32).rev());
    rows.extend((0..SORTED_TOP_K).map(|index| ((index * 73) % SORTED_TOP_K) as i32));
    rows.extend((0..SORTED_TOP_K).map(|index| ((index * 181) % SORTED_TOP_K) as i32));
    rows
}

fn assert_sorted_rows(sorted: &[i32]) {
    let expected: Vec<i32> = (0..SORTED_TOP_K as i32).collect();
    for row in sorted.chunks_exact(SORTED_TOP_K as usize) {
        assert_eq!(row, expected.as_slice());
    }
}

fn assert_dispatch_fallbacks() {
    let base = AttentionIndexedDispatchOptions {
        n_tokens: N_TOKENS,
        head_dim: HEAD_DIM,
        top_k: FILTERED_TOP_K,
        no_indexed_heads8: false,
        indexed_twopass: false,
    };
    assert_eq!(
        select_attention_indexed_path(AttentionIndexedDispatchOptions {
            no_indexed_heads8: true,
            ..base
        }),
        AttentionIndexedPath::Generic
    );
    assert_eq!(
        select_attention_indexed_path(AttentionIndexedDispatchOptions {
            n_tokens: 1,
            ..base
        }),
        AttentionIndexedPath::Generic
    );
    assert_eq!(
        select_attention_indexed_path(AttentionIndexedDispatchOptions { top_k: 513, ..base }),
        AttentionIndexedPath::Generic
    );
}

fn output_len() -> usize {
    (N_TOKENS * N_HEAD * HEAD_DIM) as usize
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
        .collect()
}

fn indexed_sort_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    source: &DeviceBuffer<i32>,
    sorted: &mut DeviceBuffer<i32>,
) -> Result<(), AttentionOptimizedError> {
    if source.len() < (N_TOKENS * SORTED_TOP_K) as usize
        || sorted.len() < (N_TOKENS * SORTED_TOP_K) as usize
    {
        return Err(AttentionOptimizedError::InvalidShape);
    }
    module
        .indexed_topk_sort_512_asc_kernel(
            stream,
            LaunchConfig {
                grid_dim: (N_TOKENS, 1, 1),
                block_dim: (SORT_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            source,
            sorted,
        )
        .map_err(AttentionOptimizedError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn attention_online_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOptimizedError> {
    launch_attention(
        module,
        stream,
        sinks,
        q,
        raw,
        comp,
        topk,
        SORTED_TOP_K,
        true,
        heads,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_rb4_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOptimizedError> {
    launch_attention(
        module,
        stream,
        sinks,
        q,
        raw,
        comp,
        topk,
        FILTERED_TOP_K,
        false,
        heads,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_attention(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    top_k: u32,
    online: bool,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOptimizedError> {
    if sinks.len() < N_HEAD as usize
        || q.len() < output_len()
        || raw.len() < (RAW_CAP * HEAD_DIM) as usize
        || comp.len() < (N_COMP * HEAD_DIM) as usize
        || topk.len() < (N_TOKENS * top_k) as usize
        || heads.len() < output_len()
        || top_k == 0
        || top_k > SORTED_TOP_K
    {
        return Err(AttentionOptimizedError::InvalidShape);
    }
    let config = LaunchConfig {
        grid_dim: (
            N_TOKENS,
            if online {
                (N_HEAD + 15) / 16
            } else {
                (N_HEAD + 7) / 8
            },
            1,
        ),
        block_dim: (if online { ONLINE_THREADS } else { RB4_THREADS }, 1, 1),
        shared_mem_bytes: 0,
    };
    if online {
        module
            .attention_indexed_mixed_heads8_online_kernel(
                stream, config, N_TOKENS, POS0, N_RAW, RAW_CAP, RAW_START, N_COMP, top_k, WINDOW,
                RATIO, N_HEAD, HEAD_DIM, sinks, q, raw, comp, topk, heads,
            )
            .map_err(AttentionOptimizedError::Driver)
    } else {
        module
            .attention_indexed_mixed_heads8_rb4_kernel(
                stream, config, N_TOKENS, POS0, N_RAW, RAW_CAP, RAW_START, N_COMP, top_k, WINDOW,
                RATIO, N_HEAD, HEAD_DIM, sinks, q, raw, comp, topk, heads,
            )
            .map_err(AttentionOptimizedError::Driver)
    }
}

#[allow(clippy::too_many_arguments)]
fn expected_attention(
    sinks: &[f32],
    q: &[f32],
    raw: &[f32],
    comp: &[f32],
    topk: &[i32],
    top_k: u32,
    filter_entries: bool,
) -> Vec<f32> {
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let first_raw_pos = POS0 + N_TOKENS - N_RAW;
    let mut heads = vec![0.0_f32; output_len()];
    for token in 0..N_TOKENS as usize {
        let qpos = POS0 + token as u32;
        let lo = first_raw_pos.max(qpos.saturating_add(1).saturating_sub(WINDOW));
        let hi = qpos.min(first_raw_pos + N_RAW - 1);
        let raw_first = lo - first_raw_pos;
        let raw_count = hi - lo + 1;
        let visible_comp = ((qpos + 1) / RATIO).min(N_COMP);
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
            let selected_count = if filter_entries {
                top_k
            } else {
                top_k.min(visible_comp)
            };
            for selected in 0..selected_count as usize {
                let compressed = topk[token * top_k as usize + selected];
                if !filter_entries || (compressed >= 0 && (compressed as u32) < visible_comp) {
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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum AttentionOptimizedError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for AttentionOptimizedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("optimized indexed attention shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionOptimizedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
