use std::fmt;

use cuda_core::{
    Blas, BlasError, BlasMathMode, CudaStream, DeviceBuffer, DriverError, LaunchConfig,
    StridedBatchedSgemmConfig,
};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{
    select_attention_prefill_path, substrate::CudaOxideSubstrate, AttentionPrefillDispatchOptions,
    AttentionPrefillPath, M14_4D5_SCOPE,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_static_mixed_heads8_online_kernel(
        n_tokens: u32,
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
        let raw_start = if window != 0 && token + 1 > window {
            token + 1 - window
        } else {
            0
        };
        let raw_count = token + 1 - raw_start;
        let mut visible_comp = if ratio == 0 { 0 } else { (token + 1) / ratio };
        if visible_comp > n_comp {
            visible_comp = n_comp;
        }
        let mut lane = 0_u32;
        while lane < 8 {
            let head = head_group * 8 + lane;
            if head < n_head {
                write_online_head(
                    token,
                    head,
                    raw_start,
                    raw_count,
                    visible_comp,
                    n_head,
                    head_dim,
                    sinks,
                    q,
                    raw_kv,
                    comp_kv,
                    &mut heads,
                );
            }
            lane += 1;
        }
    }

    #[kernel]
    pub fn attention_prefill_pack_mixed_kv_kernel(
        n_tokens: u32,
        n_comp: u32,
        head_dim: u32,
        raw_kv: &[f32],
        comp_kv: &[f32],
        mut dst: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = (n_tokens + n_comp) * head_dim;
        if index >= count as usize {
            return;
        }
        let row = index as u32 / head_dim;
        let dimension = index as u32 % head_dim;
        let value = if row < n_tokens {
            raw_kv[(row * head_dim + dimension) as usize]
        } else {
            comp_kv[((row - n_tokens) * head_dim + dimension) as usize]
        };
        unsafe {
            *dst.get_unchecked_mut(index) = value;
        }
    }

    #[kernel]
    pub fn attention_prefill_pack_q_heads_kernel(
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        q: &[f32],
        mut dst: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = n_tokens * n_head * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let token_head = index as u32 / head_dim;
        let token = token_head % n_tokens;
        let head = token_head / n_tokens;
        unsafe {
            *dst.get_unchecked_mut(index) =
                q[((token * n_head + head) * head_dim + dimension) as usize];
        }
    }

    #[kernel]
    pub fn attention_prefill_replicate_kv_kernel(
        n_keys: u32,
        n_head: u32,
        head_dim: u32,
        kv: &[f32],
        mut keys: DisjointSlice<f32>,
        mut keys_transposed: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = n_head * n_keys * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let row_head = index as u32 / head_dim;
        let row = row_head % n_keys;
        let head = row_head / n_keys;
        let value = kv[(row * head_dim + dimension) as usize];
        unsafe {
            *keys.get_unchecked_mut(index) = value;
            *keys_transposed
                .get_unchecked_mut(((head * head_dim + dimension) * n_keys + row) as usize) = value;
        }
    }

    #[kernel]
    pub fn attention_prefill_raw_softmax_kernel(
        n_tokens: u32,
        window: u32,
        n_keys: u32,
        n_head: u32,
        sinks: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let score_base = ((head * n_tokens + token) * n_keys) as usize;
        let mut max_score = sinks[head as usize];
        let mut key = 0_u32;
        while key < n_keys {
            let valid = key <= token && (window == 0 || token - key < window);
            let score = if valid {
                unsafe { *scores.get_unchecked_mut(score_base + key as usize) }
            } else {
                -1.0e30
            };
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = score;
            }
            if score > max_score {
                max_score = score;
            }
            key += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        key = 0;
        while key < n_keys {
            let score = unsafe { *scores.get_unchecked_mut(score_base + key as usize) };
            let probability = if score > -1.0e20 {
                (score - max_score).exp()
            } else {
                0.0
            };
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = probability;
            }
            denominator += probability;
            key += 1;
        }
        key = 0;
        while key < n_keys {
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) /= denominator;
            }
            key += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_prefill_mixed_softmax_kernel(
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_keys: u32,
        n_head: u32,
        use_mask: u32,
        sinks: &[f32],
        comp_mask: &[f32],
        mut scores: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || ratio == 0 || thread::threadIdx_x() != 0 {
            return;
        }
        let score_base = ((head * n_tokens + token) * n_keys) as usize;
        let visible_comp = (token + 1) / ratio;
        let mut max_score = sinks[head as usize];
        let mut key = 0_u32;
        while key < n_keys {
            let mut score = -1.0e30;
            if key < n_tokens {
                if key <= token && (window == 0 || token - key < window) {
                    score = unsafe { *scores.get_unchecked_mut(score_base + key as usize) };
                }
            } else {
                let compressed = key - n_tokens;
                if compressed < n_comp && compressed < visible_comp {
                    let add = if use_mask != 0 {
                        comp_mask[(token * n_comp + compressed) as usize]
                    } else {
                        0.0
                    };
                    if add > -1.0e20 {
                        score =
                            unsafe { *scores.get_unchecked_mut(score_base + key as usize) } + add;
                    }
                }
            }
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = score;
            }
            if score > max_score {
                max_score = score;
            }
            key += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        key = 0;
        while key < n_keys {
            let score = unsafe { *scores.get_unchecked_mut(score_base + key as usize) };
            let probability = if score > -1.0e20 {
                (score - max_score).exp()
            } else {
                0.0
            };
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) = probability;
            }
            denominator += probability;
            key += 1;
        }
        key = 0;
        while key < n_keys {
            unsafe {
                *scores.get_unchecked_mut(score_base + key as usize) /= denominator;
            }
            key += 1;
        }
    }

    #[kernel]
    pub fn attention_prefill_unpack_heads_kernel(
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        tmp: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = n_tokens * n_head * head_dim;
        if index >= count as usize {
            return;
        }
        let dimension = index as u32 % head_dim;
        let token_head = index as u32 / head_dim;
        let head = token_head % n_head;
        let token = token_head / n_head;
        unsafe {
            *heads.get_unchecked_mut(index) =
                tmp[((head * n_tokens + token) * head_dim + dimension) as usize];
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_online_head(
        token: u32,
        head: u32,
        raw_start: u32,
        raw_count: u32,
        visible_comp: u32,
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
        let mut max_score = sinks[head as usize];
        let mut row = 0_u32;
        while row < raw_count {
            let score = dot(q, query_base, raw_kv, raw_start + row, head_dim) * scale;
            if score > max_score {
                max_score = score;
            }
            row += 1;
        }
        let mut compressed = 0_u32;
        while compressed < visible_comp {
            let score = dot(q, query_base, comp_kv, compressed, head_dim) * scale;
            if score > max_score {
                max_score = score;
            }
            compressed += 1;
        }
        let mut denominator = (sinks[head as usize] - max_score).exp();
        row = 0;
        while row < raw_count {
            denominator +=
                (dot(q, query_base, raw_kv, raw_start + row, head_dim) * scale - max_score).exp();
            row += 1;
        }
        compressed = 0;
        while compressed < visible_comp {
            denominator +=
                (dot(q, query_base, comp_kv, compressed, head_dim) * scale - max_score).exp();
            compressed += 1;
        }
        let mut dimension = 0_u32;
        while dimension < head_dim {
            let mut accumulator = 0.0_f32;
            row = 0;
            while row < raw_count {
                let raw_row = raw_start + row;
                let score =
                    (dot(q, query_base, raw_kv, raw_row, head_dim) * scale - max_score).exp();
                accumulator += raw_kv[(raw_row * head_dim + dimension) as usize] * score;
                row += 1;
            }
            compressed = 0;
            while compressed < visible_comp {
                let score =
                    (dot(q, query_base, comp_kv, compressed, head_dim) * scale - max_score).exp();
                accumulator += comp_kv[(compressed * head_dim + dimension) as usize] * score;
                compressed += 1;
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
}

const THREADS: u32 = 256;
const N_TOKENS: u32 = 3;
const N_HEAD: u32 = 9;
const HEAD_DIM: u32 = 512;
const N_COMP: u32 = 2;
const WINDOW: u32 = 2;
const RATIO: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_prefill_optimized_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let blas = substrate.blas_handle()?;
    blas.set_math_mode(BlasMathMode::Default)?;

    let sink_values = values(N_HEAD as usize, 7, -0.125);
    let q_values = values((N_TOKENS * N_HEAD * HEAD_DIM) as usize, 13, -0.0625);
    let raw_values = values((N_TOKENS * HEAD_DIM) as usize, 17, -0.09375);
    let comp_values = values((N_COMP * HEAD_DIM) as usize, 19, -0.078125);
    let mask_values = vec![0.0_f32, -1.0e30, 0.0625, -1.0e30, -0.03125, 0.046875];
    let sinks = substrate.upload(&sink_values)?;
    let q = substrate.upload(&q_values)?;
    let raw = substrate.upload(&raw_values)?;
    let comp = substrate.upload(&comp_values)?;
    let mask = substrate.upload(&mask_values)?;

    assert_eq!(
        select_attention_prefill_path(AttentionPrefillDispatchOptions {
            use_comp_mask: false,
            n_tokens: N_TOKENS,
            head_dim: HEAD_DIM,
            cublas_ready: true,
            no_cublas_attention: false,
            no_window_attention: false,
            window_attention: true,
            quality_mode: true,
        }),
        AttentionPrefillPath::StaticHeads8Online
    );
    let mut online_output = substrate.zeroed::<f32>(output_len())?;
    attention_static_online_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &mut online_output,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&online_output)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &[],
            false,
        ),
    );

    assert_eq!(
        select_attention_prefill_path(AttentionPrefillDispatchOptions {
            use_comp_mask: false,
            n_tokens: N_TOKENS,
            head_dim: HEAD_DIM,
            cublas_ready: true,
            no_cublas_attention: false,
            no_window_attention: true,
            window_attention: true,
            quality_mode: false,
        }),
        AttentionPrefillPath::Cublas
    );
    let raw_cublas = attention_cublas_tensor(
        &substrate, &module, &blas, &sinks, &q, &raw, &raw, &mask, 0, false,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&raw_cublas)?,
        &expected_attention(&sink_values, &q_values, &raw_values, &[], &[], false),
    );

    assert_eq!(
        select_attention_prefill_path(AttentionPrefillDispatchOptions {
            use_comp_mask: true,
            n_tokens: N_TOKENS,
            head_dim: HEAD_DIM,
            cublas_ready: true,
            no_cublas_attention: false,
            no_window_attention: false,
            window_attention: true,
            quality_mode: false,
        }),
        AttentionPrefillPath::Cublas
    );
    let masked_cublas = attention_cublas_tensor(
        &substrate, &module, &blas, &sinks, &q, &raw, &comp, &mask, N_COMP, true,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&masked_cublas)?,
        &expected_attention(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &mask_values,
            true,
        ),
    );
    assert_dispatch_fallbacks();

    println!(
        "{{\"milestone\":\"M14.4d5\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"static_heads8_online_output_matches\":true,\"raw_cublas_prefill_output_matches\":true,\"masked_mixed_cublas_prefill_output_matches\":true,\"partial_head_group_matches\":true,\"dispatch_priority_matches\":true,\"causal_window_matches\":true,\"visible_compressed_limit_matches\":true,\"compressed_mask_matches\":true,\"sink_softmax_matches\":true,\"uses_libdevice_link_path\":true,\"owns_static_heads8_online_prefill_kernel\":{},\"owns_prefill_dispatch_policy\":{},\"owns_live_cublas_prefill_pipeline\":{},\"owns_indexed_or_output_q8_attention\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D5_SCOPE.owns_static_heads8_online_prefill_kernel,
        M14_4D5_SCOPE.owns_prefill_dispatch_policy,
        M14_4D5_SCOPE.owns_live_cublas_prefill_pipeline,
        M14_4D5_SCOPE.owns_indexed_or_output_q8_attention,
        M14_4D5_SCOPE.owns_runtime_graph_integration,
        M14_4D5_SCOPE.changes_default_route,
    );
    Ok(())
}

fn output_len() -> usize {
    (N_TOKENS * N_HEAD * HEAD_DIM) as usize
}

fn flat_config(elements: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((elements + THREADS - 1) / THREADS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_static_online_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionOptimizedError> {
    module
        .attention_static_mixed_heads8_online_kernel(
            stream,
            LaunchConfig {
                grid_dim: (N_TOKENS, (N_HEAD + 7) / 8, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            N_COMP,
            WINDOW,
            RATIO,
            N_HEAD,
            HEAD_DIM,
            sinks,
            q,
            raw,
            comp,
            heads,
        )
        .map_err(AttentionOptimizedError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn attention_cublas_tensor(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    blas: &Blas,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    mask: &DeviceBuffer<f32>,
    n_comp: u32,
    use_mask: bool,
) -> Result<DeviceBuffer<f32>, AttentionOptimizedError> {
    let n_keys = N_TOKENS + n_comp;
    let kv_len = (n_keys * HEAD_DIM) as usize;
    let head_kv_len = (N_HEAD * n_keys * HEAD_DIM) as usize;
    let mut kv = substrate.zeroed::<f32>(kv_len)?;
    let mut q_heads = substrate.zeroed::<f32>(output_len())?;
    let mut keys = substrate.zeroed::<f32>(head_kv_len)?;
    let mut keys_transposed = substrate.zeroed::<f32>(head_kv_len)?;
    module.attention_prefill_pack_mixed_kv_kernel(
        substrate.stream(),
        flat_config(n_keys * HEAD_DIM),
        N_TOKENS,
        n_comp,
        HEAD_DIM,
        raw,
        comp,
        &mut kv,
    )?;
    module.attention_prefill_pack_q_heads_kernel(
        substrate.stream(),
        flat_config(N_TOKENS * N_HEAD * HEAD_DIM),
        N_TOKENS,
        N_HEAD,
        HEAD_DIM,
        q,
        &mut q_heads,
    )?;
    module.attention_prefill_replicate_kv_kernel(
        substrate.stream(),
        flat_config(N_HEAD * n_keys * HEAD_DIM),
        n_keys,
        N_HEAD,
        HEAD_DIM,
        &kv,
        &mut keys,
        &mut keys_transposed,
    )?;

    let mut scores = substrate.zeroed::<f32>((N_HEAD * N_TOKENS * n_keys) as usize)?;
    let mut score_config = StridedBatchedSgemmConfig::packed(
        N_TOKENS as usize,
        n_keys as usize,
        HEAD_DIM as usize,
        N_HEAD as usize,
    )?;
    score_config.alpha = 1.0 / (HEAD_DIM as f32).sqrt();
    blas.sgemm_strided_batched(
        substrate.stream(),
        score_config,
        &q_heads,
        &keys_transposed,
        &mut scores,
    )?;
    if n_comp == 0 {
        module.attention_prefill_raw_softmax_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (N_TOKENS, N_HEAD, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            WINDOW,
            n_keys,
            N_HEAD,
            sinks,
            &mut scores,
        )?;
    } else {
        module.attention_prefill_mixed_softmax_kernel(
            substrate.stream(),
            LaunchConfig {
                grid_dim: (N_TOKENS, N_HEAD, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            n_comp,
            WINDOW,
            RATIO,
            n_keys,
            N_HEAD,
            u32::from(use_mask),
            sinks,
            mask,
            &mut scores,
        )?;
    }
    let mut output_by_head = substrate.zeroed::<f32>(output_len())?;
    let value_config = StridedBatchedSgemmConfig::packed(
        N_TOKENS as usize,
        HEAD_DIM as usize,
        n_keys as usize,
        N_HEAD as usize,
    )?;
    blas.sgemm_strided_batched(
        substrate.stream(),
        value_config,
        &scores,
        &keys,
        &mut output_by_head,
    )?;
    let mut heads = substrate.zeroed::<f32>(output_len())?;
    module.attention_prefill_unpack_heads_kernel(
        substrate.stream(),
        flat_config(N_TOKENS * N_HEAD * HEAD_DIM),
        N_TOKENS,
        N_HEAD,
        HEAD_DIM,
        &output_by_head,
        &mut heads,
    )?;
    Ok(heads)
}

fn assert_dispatch_fallbacks() {
    let base = AttentionPrefillDispatchOptions {
        use_comp_mask: false,
        n_tokens: 128,
        head_dim: 512,
        cublas_ready: true,
        no_cublas_attention: false,
        no_window_attention: false,
        window_attention: false,
        quality_mode: false,
    };
    assert_eq!(
        select_attention_prefill_path(base),
        AttentionPrefillPath::StaticHeads8Online
    );
    assert_eq!(
        select_attention_prefill_path(AttentionPrefillDispatchOptions {
            n_tokens: 2,
            quality_mode: true,
            ..base
        }),
        AttentionPrefillPath::Cublas
    );
    assert_eq!(
        select_attention_prefill_path(AttentionPrefillDispatchOptions {
            n_tokens: 2,
            quality_mode: true,
            no_cublas_attention: true,
            ..base
        }),
        AttentionPrefillPath::Generic
    );
}

fn expected_attention(
    sinks: &[f32],
    q: &[f32],
    raw: &[f32],
    comp: &[f32],
    mask: &[f32],
    use_mask: bool,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; output_len()];
    for token in 0..N_TOKENS as usize {
        let raw_start = if WINDOW != 0 && token as u32 + 1 > WINDOW {
            token + 1 - WINDOW as usize
        } else {
            0
        };
        let raw_count = token + 1 - raw_start;
        let visible_comp = if comp.is_empty() {
            0
        } else {
            (((token as u32 + 1) / RATIO).min(N_COMP)) as usize
        };
        for head in 0..N_HEAD as usize {
            let query_base = (token * N_HEAD as usize + head) * HEAD_DIM as usize;
            let query = &q[query_base..query_base + HEAD_DIM as usize];
            let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
            let mut rows: Vec<(&[f32], f32)> = (0..raw_count)
                .map(|offset| {
                    let row = &raw[(raw_start + offset) * HEAD_DIM as usize
                        ..(raw_start + offset + 1) * HEAD_DIM as usize];
                    (row, dot_host(query, row) * scale)
                })
                .collect();
            for compressed in 0..visible_comp {
                let add = if use_mask {
                    mask[token * N_COMP as usize + compressed]
                } else {
                    0.0
                };
                if add > -1.0e20 {
                    let row =
                        &comp[compressed * HEAD_DIM as usize..(compressed + 1) * HEAD_DIM as usize];
                    rows.push((row, dot_host(query, row) * scale + add));
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
                output[query_base + dimension] = rows
                    .iter()
                    .map(|(row, score)| row[dimension] * (*score - max_score).exp())
                    .sum::<f32>()
                    / denominator;
            }
        }
    }
    output
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 3) % 37) as f32 * 0.00390625 + offset)
        .collect()
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
            (actual - expected).abs() <= 5.0e-4,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum AttentionOptimizedError {
    Driver(DriverError),
    Blas(BlasError),
}

impl From<DriverError> for AttentionOptimizedError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl From<BlasError> for AttentionOptimizedError {
    fn from(error: BlasError) -> Self {
        Self::Blas(error)
    }
}

impl fmt::Display for AttentionOptimizedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => error.fmt(formatter),
            Self::Blas(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionOptimizedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Blas(error) => Some(error),
        }
    }
}
