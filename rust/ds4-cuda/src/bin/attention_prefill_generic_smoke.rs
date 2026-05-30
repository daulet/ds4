use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_4D4_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn attention_prefill_raw_kernel(
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        mut heads: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let raw_count = if token + 1 < window {
            token + 1
        } else {
            window
        };
        let raw_start = token + 1 - raw_count;
        write_head(
            token,
            head,
            raw_start,
            raw_count,
            0,
            0,
            n_head,
            head_dim,
            sinks,
            q,
            raw_kv,
            raw_kv,
            &[],
            0,
            &mut heads,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn attention_prefill_mixed_kernel(
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
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
        let token = thread::blockIdx_x();
        let head = thread::blockIdx_y();
        if token >= n_tokens || head >= n_head || thread::threadIdx_x() != 0 {
            return;
        }
        let raw_start = if window != 0 && token + 1 > window {
            token + 1 - window
        } else {
            0
        };
        let raw_count = token + 1 - raw_start;
        let mut visible_comp = (token + 1) / ratio;
        if visible_comp > n_comp {
            visible_comp = n_comp;
        }
        write_head(
            token,
            head,
            raw_start,
            raw_count,
            visible_comp,
            n_comp,
            n_head,
            head_dim,
            sinks,
            q,
            raw_kv,
            comp_kv,
            comp_mask,
            use_mask,
            &mut heads,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_head(
        token: u32,
        head: u32,
        raw_start: u32,
        raw_count: u32,
        visible_comp: u32,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        sinks: &[f32],
        q: &[f32],
        raw_kv: &[f32],
        comp_kv: &[f32],
        comp_mask: &[f32],
        use_mask: u32,
        heads: &mut DisjointSlice<f32>,
    ) {
        let query_base = ((token * n_head + head) * head_dim) as usize;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut max_score = sinks[head as usize];
        let mut row = 0_u32;
        while row < raw_count {
            max_score = maximum(
                max_score,
                dot(q, query_base, raw_kv, raw_start + row, head_dim) * scale,
            );
            row += 1;
        }
        let mut compressed = 0_u32;
        while compressed < visible_comp {
            let add = if use_mask != 0 {
                comp_mask[(token * n_comp + compressed) as usize]
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
        row = 0;
        while row < raw_count {
            denominator +=
                (dot(q, query_base, raw_kv, raw_start + row, head_dim) * scale - max_score).exp();
            row += 1;
        }
        compressed = 0;
        while compressed < visible_comp {
            let add = if use_mask != 0 {
                comp_mask[(token * n_comp + compressed) as usize]
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
                let add = if use_mask != 0 {
                    comp_mask[(token * n_comp + compressed) as usize]
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
const N_TOKENS: u32 = 4;
const N_HEAD: u32 = 2;
const HEAD_DIM: u32 = 7;
const N_COMP: u32 = 2;
const WINDOW: u32 = 3;
const RATIO: u32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_prefill_generic_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let sink_values = vec![-0.25_f32, 0.375];
    let q_values = values((N_TOKENS * N_HEAD * HEAD_DIM) as usize, 17, -0.75);
    let raw_values = values((N_TOKENS * HEAD_DIM) as usize, 23, -1.125);
    let comp_values = values((N_COMP * HEAD_DIM) as usize, 29, -0.875);
    let mask_values = vec![0.0_f32, -1.0e30, 0.125, -1.0e30, -0.25, 0.25, 0.0, -1.0e30];
    let sinks = substrate.upload(&sink_values)?;
    let q = substrate.upload(&q_values)?;
    let raw = substrate.upload(&raw_values)?;
    let comp = substrate.upload(&comp_values)?;
    let mask = substrate.upload(&mask_values)?;

    let mut raw_output = substrate.zeroed::<f32>(output_len())?;
    attention_prefill_raw_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &mut raw_output,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&raw_output)?,
        &expected_raw(&sink_values, &q_values, &raw_values),
        2.0e-5,
    );

    let mut static_mixed = substrate.zeroed::<f32>(output_len())?;
    attention_prefill_mixed_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &mask,
        false,
        RATIO,
        &mut static_mixed,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&static_mixed)?,
        &expected_mixed(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &mask_values,
            false,
        ),
        2.0e-5,
    );

    let mut masked_mixed = substrate.zeroed::<f32>(output_len())?;
    attention_prefill_mixed_tensor(
        &module,
        substrate.stream(),
        &sinks,
        &q,
        &raw,
        &comp,
        &mask,
        true,
        RATIO,
        &mut masked_mixed,
    )?;
    substrate.end_commands()?;
    assert_close(
        &substrate.download(&masked_mixed)?,
        &expected_mixed(
            &sink_values,
            &q_values,
            &raw_values,
            &comp_values,
            &mask_values,
            true,
        ),
        2.0e-5,
    );

    let mut invalid_output = substrate.zeroed::<f32>(output_len())?;
    assert!(matches!(
        attention_prefill_mixed_tensor(
            &module,
            substrate.stream(),
            &sinks,
            &q,
            &raw,
            &comp,
            &mask,
            true,
            0,
            &mut invalid_output,
        ),
        Err(AttentionPrefillError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4d4\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"raw_prefill_output_matches\":true,\"static_mixed_prefill_output_matches\":true,\"masked_mixed_prefill_output_matches\":true,\"causal_window_matches\":true,\"visible_compressed_limit_matches\":true,\"compressed_mask_matches\":true,\"sink_softmax_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_attention_prefill_raw_surface\":{},\"owns_attention_prefill_static_mixed_surface\":{},\"owns_attention_prefill_masked_mixed_surface\":{},\"owns_generic_prefill_kernels\":{},\"owns_static_heads8_online_or_cublas_prefill_dispatch\":{},\"owns_indexed_or_output_q8_attention\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D4_SCOPE.owns_attention_prefill_raw_surface,
        M14_4D4_SCOPE.owns_attention_prefill_static_mixed_surface,
        M14_4D4_SCOPE.owns_attention_prefill_masked_mixed_surface,
        M14_4D4_SCOPE.owns_generic_prefill_kernels,
        M14_4D4_SCOPE.owns_static_heads8_online_or_cublas_prefill_dispatch,
        M14_4D4_SCOPE.owns_indexed_or_output_q8_attention,
        M14_4D4_SCOPE.owns_runtime_graph_integration,
        M14_4D4_SCOPE.changes_default_route,
    );
    Ok(())
}

fn output_len() -> usize {
    (N_TOKENS * N_HEAD * HEAD_DIM) as usize
}

fn attention_prefill_raw_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionPrefillError> {
    if WINDOW == 0
        || WINDOW > 256
        || sinks.len() < N_HEAD as usize
        || q.len() < output_len()
        || raw.len() < (N_TOKENS * HEAD_DIM) as usize
        || heads.len() < output_len()
    {
        return Err(AttentionPrefillError::InvalidShape);
    }
    module
        .attention_prefill_raw_kernel(
            stream,
            LaunchConfig {
                grid_dim: (N_TOKENS, N_HEAD, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            WINDOW,
            N_HEAD,
            HEAD_DIM,
            sinks,
            q,
            raw,
            heads,
        )
        .map_err(AttentionPrefillError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn attention_prefill_mixed_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    sinks: &DeviceBuffer<f32>,
    q: &DeviceBuffer<f32>,
    raw: &DeviceBuffer<f32>,
    comp: &DeviceBuffer<f32>,
    mask: &DeviceBuffer<f32>,
    use_mask: bool,
    ratio: u32,
    heads: &mut DeviceBuffer<f32>,
) -> Result<(), AttentionPrefillError> {
    if ratio == 0
        || sinks.len() < N_HEAD as usize
        || q.len() < output_len()
        || raw.len() < (N_TOKENS * HEAD_DIM) as usize
        || comp.len() < (N_COMP * HEAD_DIM) as usize
        || (use_mask && mask.len() < (N_TOKENS * N_COMP) as usize)
        || heads.len() < output_len()
    {
        return Err(AttentionPrefillError::InvalidShape);
    }
    module
        .attention_prefill_mixed_kernel(
            stream,
            LaunchConfig {
                grid_dim: (N_TOKENS, N_HEAD, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            N_TOKENS,
            N_COMP,
            WINDOW,
            ratio,
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
        .map_err(AttentionPrefillError::Driver)
}

fn expected_raw(sinks: &[f32], q: &[f32], raw: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; output_len()];
    for token in 0..N_TOKENS as usize {
        let raw_count = ((token as u32 + 1).min(WINDOW)) as usize;
        let raw_start = token + 1 - raw_count;
        for head in 0..N_HEAD as usize {
            write_expected(
                &mut output,
                token,
                head,
                sinks,
                q,
                raw,
                &[],
                &[],
                false,
                raw_start,
                raw_count,
                0,
            );
        }
    }
    output
}

fn expected_mixed(
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
        let visible_comp = (((token as u32 + 1) / RATIO).min(N_COMP)) as usize;
        for head in 0..N_HEAD as usize {
            write_expected(
                &mut output,
                token,
                head,
                sinks,
                q,
                raw,
                comp,
                mask,
                use_mask,
                raw_start,
                raw_count,
                visible_comp,
            );
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn write_expected(
    output: &mut [f32],
    token: usize,
    head: usize,
    sinks: &[f32],
    q: &[f32],
    raw: &[f32],
    comp: &[f32],
    mask: &[f32],
    use_mask: bool,
    raw_start: usize,
    raw_count: usize,
    visible_comp: usize,
) {
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
            let row = &comp[compressed * HEAD_DIM as usize..(compressed + 1) * HEAD_DIM as usize];
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

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 97) as f32 * 0.03125 + offset)
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
enum AttentionPrefillError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for AttentionPrefillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => {
                formatter.write_str("generic attention prefill tensor shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionPrefillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
