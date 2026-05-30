use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5A_SCOPE};

const N_EXPERT: u32 = 256;
const TOP_K: u32 = 6;
const N_TOKENS: u32 = 3;
const HASH_ROWS: u32 = 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn router_select_kernel(
        n_tokens: u32,
        token_scalar: i32,
        hash_rows: u32,
        has_bias: u32,
        hash_mode: u32,
        use_token_buffer: u32,
        logits: &[f32],
        bias: &[f32],
        hash: &[i32],
        tokens: &[i32],
        mut selected: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
    ) {
        let token_index = thread::blockIdx_x();
        if token_index >= n_tokens || thread::threadIdx_x() != 0 {
            return;
        }
        let logit_base = token_index as usize * N_EXPERT as usize;
        let selected_base = token_index as usize * TOP_K as usize;
        let mut expert = 0_u32;
        while expert < N_EXPERT {
            unsafe {
                *probs.get_unchecked_mut(logit_base + expert as usize) =
                    router_prob(logits[logit_base + expert as usize]);
            }
            expert += 1;
        }

        let mut chosen = [-1_i32; TOP_K as usize];
        if hash_mode != 0 {
            let mut token = if use_token_buffer != 0 {
                tokens[token_index as usize]
            } else {
                token_scalar
            };
            if token < 0 || token as u32 >= hash_rows {
                token = 0;
            }
            let hash_base = token as usize * TOP_K as usize;
            let mut output = 0_u32;
            while output < TOP_K {
                chosen[output as usize] = hash[hash_base + output as usize];
                output += 1;
            }
        } else {
            expert = 0;
            while expert < N_EXPERT {
                let score = router_prob(logits[logit_base + expert as usize])
                    + if has_bias != 0 {
                        bias[expert as usize]
                    } else {
                        0.0
                    };
                let mut output = 0_u32;
                while output < TOP_K {
                    let current = chosen[output as usize];
                    let better = if current < 0 {
                        true
                    } else {
                        score
                            > router_prob(logits[logit_base + current as usize])
                                + if has_bias != 0 {
                                    bias[current as usize]
                                } else {
                                    0.0
                                }
                    };
                    if better {
                        let mut shift = TOP_K - 1;
                        while shift > output {
                            chosen[shift as usize] = chosen[(shift - 1) as usize];
                            shift -= 1;
                        }
                        chosen[output as usize] = expert as i32;
                        break;
                    }
                    output += 1;
                }
                expert += 1;
            }
        }

        let mut sum = 0.0_f32;
        let mut output = 0_u32;
        while output < TOP_K {
            let selected_expert = chosen[output as usize];
            let value = if selected_expert >= 0 && selected_expert < N_EXPERT as i32 {
                router_prob(logits[logit_base + selected_expert as usize])
            } else {
                0.0
            };
            unsafe {
                *selected.get_unchecked_mut(selected_base + output as usize) = selected_expert;
                *weights.get_unchecked_mut(selected_base + output as usize) = value;
            }
            sum += value;
            output += 1;
        }
        if sum < 6.103515625e-5_f32 {
            sum = 6.103515625e-5_f32;
        }
        output = 0;
        while output < TOP_K {
            let selected_expert = chosen[output as usize];
            let value = if selected_expert >= 0 && selected_expert < N_EXPERT as i32 {
                router_prob(logits[logit_base + selected_expert as usize])
            } else {
                0.0
            };
            unsafe {
                *weights.get_unchecked_mut(selected_base + output as usize) = value / sum * 1.5;
            }
            output += 1;
        }
    }

    fn router_prob(logit: f32) -> f32 {
        let softplus = if logit > 20.0 {
            logit
        } else if logit < -20.0 {
            logit.exp()
        } else {
            (1.0 + logit.exp()).ln()
        };
        softplus.sqrt()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module =
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_router_scalar_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let logits_values = scored_logits();
    let bias_values = bias_values();
    let hash_values = vec![9, 1, 250, -1, 7, 8, 4, 3, 2, 1, 0, 255];
    let token_values = vec![-1, 1, 7];
    let logits = substrate.upload(&logits_values)?;
    let bias = substrate.upload(&bias_values)?;
    let hash = substrate.upload(&hash_values)?;
    let tokens = substrate.upload(&token_values)?;

    let scored = router_select_tensor(
        &substrate, &module, &logits, &bias, &hash, &tokens, N_TOKENS, 0, false, true, false,
    )?;
    substrate.flush_commands()?;
    assert_router_output(
        &substrate,
        &scored,
        &expected_router(
            &logits_values,
            &bias_values,
            &hash_values,
            &token_values,
            N_TOKENS,
            0,
            false,
            true,
            false,
        ),
    )?;

    let hashed = router_select_tensor(
        &substrate, &module, &logits, &bias, &hash, &tokens, N_TOKENS, 0, true, false, true,
    )?;
    substrate.flush_commands()?;
    assert_router_output(
        &substrate,
        &hashed,
        &expected_router(
            &logits_values,
            &bias_values,
            &hash_values,
            &token_values,
            N_TOKENS,
            0,
            true,
            false,
            true,
        ),
    )?;

    let scalar = router_select_tensor(
        &substrate, &module, &logits, &bias, &hash, &tokens, 1, 1, true, false, false,
    )?;
    substrate.end_commands()?;
    assert_router_output(
        &substrate,
        &scalar,
        &expected_router(
            &logits_values,
            &bias_values,
            &hash_values,
            &token_values,
            1,
            1,
            true,
            false,
            false,
        ),
    )?;

    let short_selected = substrate.zeroed::<i32>((N_TOKENS * TOP_K - 1) as usize)?;
    assert!(matches!(
        launch_router(
            &module,
            substrate.stream(),
            &logits,
            &bias,
            &hash,
            &tokens,
            N_TOKENS,
            0,
            false,
            true,
            false,
            short_selected,
        ),
        Err(RouterError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.5a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"biased_topk_output_matches\":true,\"hash_router_output_matches\":true,\"hash_invalid_token_fallback_matches\":true,\"single_token_scalar_hash_matches\":true,\"probability_transform_matches\":true,\"normalized_weight_scale_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_router_select_kernel\":{},\"owns_scalar_single_and_batch_router_surface\":{},\"owns_bias_and_hash_router_semantics\":{},\"owns_parallel_or_warp_router_dispatch\":{},\"owns_routed_moe_or_hyperconnection\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5A_SCOPE.owns_router_select_kernel,
        M14_5A_SCOPE.owns_scalar_single_and_batch_router_surface,
        M14_5A_SCOPE.owns_bias_and_hash_router_semantics,
        M14_5A_SCOPE.owns_parallel_or_warp_router_dispatch,
        M14_5A_SCOPE.owns_routed_moe_or_hyperconnection,
        M14_5A_SCOPE.owns_runtime_graph_integration,
        M14_5A_SCOPE.changes_default_route,
    );
    Ok(())
}

struct RouterOutput {
    selected: DeviceBuffer<i32>,
    weights: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
}

struct ExpectedRouterOutput {
    selected: Vec<i32>,
    weights: Vec<f32>,
    probs: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn router_select_tensor(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    hash: &DeviceBuffer<i32>,
    tokens: &DeviceBuffer<i32>,
    n_tokens: u32,
    token_scalar: i32,
    hash_mode: bool,
    has_bias: bool,
    use_token_buffer: bool,
) -> Result<RouterOutput, RouterError> {
    let selected = substrate.zeroed::<i32>((n_tokens * TOP_K) as usize)?;
    launch_router(
        module,
        substrate.stream(),
        logits,
        bias,
        hash,
        tokens,
        n_tokens,
        token_scalar,
        hash_mode,
        has_bias,
        use_token_buffer,
        selected,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_router(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    hash: &DeviceBuffer<i32>,
    tokens: &DeviceBuffer<i32>,
    n_tokens: u32,
    token_scalar: i32,
    hash_mode: bool,
    has_bias: bool,
    use_token_buffer: bool,
    mut selected: DeviceBuffer<i32>,
) -> Result<RouterOutput, RouterError> {
    if n_tokens == 0
        || logits.len() < (n_tokens * N_EXPERT) as usize
        || selected.len() < (n_tokens * TOP_K) as usize
        || (has_bias && !hash_mode && bias.len() < N_EXPERT as usize)
        || (hash_mode && hash.len() < (HASH_ROWS * TOP_K) as usize)
        || (hash_mode && use_token_buffer && tokens.len() < n_tokens as usize)
    {
        return Err(RouterError::InvalidShape);
    }
    let mut weights = DeviceBuffer::zeroed(stream, (n_tokens * TOP_K) as usize)?;
    let mut probs = DeviceBuffer::zeroed(stream, (n_tokens * N_EXPERT) as usize)?;
    module.router_select_kernel(
        stream,
        LaunchConfig {
            grid_dim: (n_tokens, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
        n_tokens,
        token_scalar,
        HASH_ROWS,
        u32::from(has_bias),
        u32::from(hash_mode),
        u32::from(use_token_buffer),
        logits,
        bias,
        hash,
        tokens,
        &mut selected,
        &mut weights,
        &mut probs,
    )?;
    Ok(RouterOutput {
        selected,
        weights,
        probs,
    })
}

fn scored_logits() -> Vec<f32> {
    let mut logits = vec![-4.0_f32; (N_TOKENS * N_EXPERT) as usize];
    for token in 0..N_TOKENS as usize {
        for expert in 0..N_EXPERT as usize {
            logits[token * N_EXPERT as usize + expert] =
                -3.0 + ((expert * 13 + token * 7) % 31) as f32 * 0.0625;
        }
        for (rank, expert) in [42_usize, 17, 3, 200, 11, 99, 7].into_iter().enumerate() {
            logits[token * N_EXPERT as usize + expert] =
                2.5 - rank as f32 * 0.125 + token as f32 * 0.03125;
        }
    }
    logits
}

fn bias_values() -> Vec<f32> {
    (0..N_EXPERT as usize)
        .map(|expert| {
            if expert == 7 {
                0.75
            } else {
                (expert % 5) as f32 * 0.001
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn expected_router(
    logits: &[f32],
    bias: &[f32],
    hash: &[i32],
    tokens: &[i32],
    n_tokens: u32,
    token_scalar: i32,
    hash_mode: bool,
    has_bias: bool,
    use_token_buffer: bool,
) -> ExpectedRouterOutput {
    let mut selected = Vec::new();
    let mut weights = Vec::new();
    let mut probs = Vec::new();
    for token_index in 0..n_tokens as usize {
        let row = &logits[token_index * N_EXPERT as usize..(token_index + 1) * N_EXPERT as usize];
        let row_probs = row
            .iter()
            .map(|value| router_prob(*value))
            .collect::<Vec<_>>();
        let chosen = if hash_mode {
            let token = if use_token_buffer {
                tokens[token_index]
            } else {
                token_scalar
            };
            let token = if token < 0 || token as u32 >= HASH_ROWS {
                0
            } else {
                token as usize
            };
            hash[token * TOP_K as usize..(token + 1) * TOP_K as usize].to_vec()
        } else {
            let mut candidates = (0..N_EXPERT as usize).collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                let left_score = row_probs[*left] + if has_bias { bias[*left] } else { 0.0 };
                let right_score = row_probs[*right] + if has_bias { bias[*right] } else { 0.0 };
                right_score.total_cmp(&left_score)
            });
            candidates[..TOP_K as usize]
                .iter()
                .map(|expert| *expert as i32)
                .collect()
        };
        let chosen_probs = chosen
            .iter()
            .map(|expert| {
                if *expert >= 0 && *expert < N_EXPERT as i32 {
                    row_probs[*expert as usize]
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let denominator = chosen_probs.iter().sum::<f32>().max(6.103515625e-5_f32);
        selected.extend(chosen);
        weights.extend(chosen_probs.iter().map(|value| value / denominator * 1.5));
        probs.extend(row_probs);
    }
    ExpectedRouterOutput {
        selected,
        weights,
        probs,
    }
}

fn router_prob(logit: f32) -> f32 {
    let softplus = if logit > 20.0 {
        logit
    } else if logit < -20.0 {
        logit.exp()
    } else {
        (1.0 + logit.exp()).ln()
    };
    softplus.sqrt()
}

fn assert_router_output(
    substrate: &CudaOxideSubstrate,
    output: &RouterOutput,
    expected: &ExpectedRouterOutput,
) -> Result<(), DriverError> {
    assert_eq!(substrate.download(&output.selected)?, expected.selected);
    assert_close(&substrate.download(&output.weights)?, &expected.weights);
    assert_close(&substrate.download(&output.probs)?, &expected.probs);
    Ok(())
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum RouterError {
    InvalidShape,
    Driver(DriverError),
}

impl From<DriverError> for RouterError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl fmt::Display for RouterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("router tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RouterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
