use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{
    select_router_select_path, substrate::CudaOxideSubstrate, RouterSelectDispatchOptions,
    RouterSelectPath, M14_5B_SCOPE,
};

const N_EXPERT: usize = 256;
const TOP_K: usize = 6;
const N_TOKENS: u32 = 5;
const HASH_ROWS: u32 = 2;
const ROWS_PER_WARP_BLOCK: u32 = 4;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn router_select_parallel_kernel(
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
        static mut SPROB: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;

        let token_index = thread::blockIdx_x() as usize;
        let expert = thread::threadIdx_x() as usize;
        if token_index >= n_tokens as usize || expert >= N_EXPERT {
            return;
        }
        let prob_base = token_index * N_EXPERT;
        let selected_base = token_index * TOP_K;
        let value = router_prob(logits[prob_base + expert]);
        unsafe {
            SPROB[expert] = value;
            *probs.get_unchecked_mut(prob_base + expert) = value;
        }
        thread::sync_threads();
        if expert != 0 {
            return;
        }

        let mut chosen = [-1_i32; TOP_K];
        if hash_mode != 0 {
            let mut token = if use_token_buffer != 0 {
                tokens[token_index]
            } else {
                token_scalar
            };
            if token < 0 || token as u32 >= hash_rows {
                token = 0;
            }
            let hash_base = token as usize * TOP_K;
            let mut output = 0_usize;
            while output < TOP_K {
                chosen[output] = hash[hash_base + output];
                output += 1;
            }
        } else {
            let mut candidate = 0_usize;
            while candidate < N_EXPERT {
                let score =
                    unsafe { SPROB[candidate] } + if has_bias != 0 { bias[candidate] } else { 0.0 };
                let mut output = 0_usize;
                while output < TOP_K {
                    let incumbent = chosen[output];
                    let better = if incumbent < 0 {
                        true
                    } else {
                        score
                            > unsafe { SPROB[incumbent as usize] }
                                + if has_bias != 0 {
                                    bias[incumbent as usize]
                                } else {
                                    0.0
                                }
                    };
                    if better {
                        let mut shift = TOP_K - 1;
                        while shift > output {
                            chosen[shift] = chosen[shift - 1];
                            shift -= 1;
                        }
                        chosen[output] = candidate as i32;
                        break;
                    }
                    output += 1;
                }
                candidate += 1;
            }
        }

        let mut sum = 0.0_f32;
        let mut output = 0_usize;
        while output < TOP_K {
            let chosen_expert = chosen[output];
            let probability = if chosen_expert >= 0 && chosen_expert < N_EXPERT as i32 {
                unsafe { SPROB[chosen_expert as usize] }
            } else {
                0.0
            };
            unsafe {
                *selected.get_unchecked_mut(selected_base + output) = chosen_expert;
                *weights.get_unchecked_mut(selected_base + output) = probability;
            }
            sum += probability;
            output += 1;
        }
        if sum < 6.103515625e-5_f32 {
            sum = 6.103515625e-5_f32;
        }
        output = 0;
        while output < TOP_K {
            let chosen_expert = chosen[output];
            let probability = if chosen_expert >= 0 && chosen_expert < N_EXPERT as i32 {
                unsafe { SPROB[chosen_expert as usize] }
            } else {
                0.0
            };
            unsafe {
                *weights.get_unchecked_mut(selected_base + output) = probability / sum * 1.5;
            }
            output += 1;
        }
    }

    #[kernel]
    pub fn router_select_warp_topk_kernel(
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
        static mut SPROB: SharedArray<f32, { ROWS_PER_WARP_BLOCK as usize * N_EXPERT }> =
            SharedArray::UNINIT;

        let lane = thread::threadIdx_x();
        let row_in_block = thread::threadIdx_y();
        let token_index = thread::blockIdx_x() * ROWS_PER_WARP_BLOCK + row_in_block;
        if token_index >= n_tokens || lane >= 32 {
            return;
        }
        let prob_base = token_index as usize * N_EXPERT;
        let shared_base = row_in_block as usize * N_EXPERT;
        let selected_base = token_index as usize * TOP_K;
        let mut local_prob = [0.0_f32; 8];
        let mut local_score = [0.0_f32; 8];
        let mut slot = 0_usize;
        while slot < 8 {
            let expert = lane as usize + slot * 32;
            let probability = router_prob(logits[prob_base + expert]);
            local_prob[slot] = probability;
            local_score[slot] = probability + if has_bias != 0 { bias[expert] } else { 0.0 };
            unsafe {
                SPROB[shared_base + expert] = probability;
                *probs.get_unchecked_mut(prob_base + expert) = probability;
            }
            slot += 1;
        }
        warp::sync_mask(u32::MAX);

        if hash_mode != 0 {
            if lane == 0 {
                let mut token = if use_token_buffer != 0 {
                    tokens[token_index as usize]
                } else {
                    token_scalar
                };
                if token < 0 || token as u32 >= hash_rows {
                    token = 0;
                }
                let hash_base = token as usize * TOP_K;
                let mut sum = 0.0_f32;
                let mut output = 0_usize;
                while output < TOP_K {
                    let chosen_expert = hash[hash_base + output];
                    let probability = if chosen_expert >= 0 && chosen_expert < N_EXPERT as i32 {
                        unsafe { SPROB[shared_base + chosen_expert as usize] }
                    } else {
                        0.0
                    };
                    unsafe {
                        *selected.get_unchecked_mut(selected_base + output) = chosen_expert;
                        *weights.get_unchecked_mut(selected_base + output) = probability;
                    }
                    sum += probability;
                    output += 1;
                }
                if sum < 6.103515625e-5_f32 {
                    sum = 6.103515625e-5_f32;
                }
                output = 0;
                while output < TOP_K {
                    let chosen_expert = hash[hash_base + output];
                    let probability = if chosen_expert >= 0 && chosen_expert < N_EXPERT as i32 {
                        unsafe { SPROB[shared_base + chosen_expert as usize] }
                    } else {
                        0.0
                    };
                    unsafe {
                        *weights.get_unchecked_mut(selected_base + output) =
                            probability / sum * 1.5;
                    }
                    output += 1;
                }
            }
            return;
        }

        let mut output_prob = [0.0_f32; TOP_K];
        let mut output_index = [0_u32; TOP_K];
        let mut output = 0_usize;
        while output < TOP_K {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prob = 0.0_f32;
            let mut best_index = u32::MAX;
            slot = 0;
            while slot < 8 {
                let candidate = lane + slot as u32 * 32;
                let score = local_score[slot];
                if router_score_better(score, candidate, best_score, best_index) {
                    best_score = score;
                    best_prob = local_prob[slot];
                    best_index = candidate;
                }
                slot += 1;
            }
            let mut mask = 16_u32;
            while mask > 0 {
                let other_score = warp::shuffle_xor_f32(best_score, mask);
                let other_prob = warp::shuffle_xor_f32(best_prob, mask);
                let other_index = warp::shuffle_xor(best_index, mask);
                if router_score_better(other_score, other_index, best_score, best_index) {
                    best_score = other_score;
                    best_prob = other_prob;
                    best_index = other_index;
                }
                mask >>= 1;
            }
            slot = 0;
            while slot < 8 {
                if lane + slot as u32 * 32 == best_index {
                    local_score[slot] = f32::NEG_INFINITY;
                }
                slot += 1;
            }
            if lane == 0 {
                output_index[output] = best_index;
                output_prob[output] = best_prob;
            }
            output += 1;
        }

        if lane == 0 {
            let mut sum = 0.0_f32;
            output = 0;
            while output < TOP_K {
                unsafe {
                    *selected.get_unchecked_mut(selected_base + output) =
                        output_index[output] as i32;
                    *weights.get_unchecked_mut(selected_base + output) = output_prob[output];
                }
                sum += output_prob[output];
                output += 1;
            }
            if sum < 6.103515625e-5_f32 {
                sum = 6.103515625e-5_f32;
            }
            output = 0;
            while output < TOP_K {
                unsafe {
                    *weights.get_unchecked_mut(selected_base + output) =
                        output_prob[output] / sum * 1.5;
                }
                output += 1;
            }
        }
    }

    fn router_score_better(
        candidate_score: f32,
        candidate_index: u32,
        best_score: f32,
        best_index: u32,
    ) -> bool {
        candidate_score > best_score
            || (candidate_score == best_score && candidate_index < best_index)
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
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_router_optimized_smoke")?;
    let module = kernels::from_module(raw_module)?;
    let logits_values = ranked_logits();
    let tie_logits_values = tie_logits();
    let bias_values = bias_values();
    let hash_values = vec![9, 1, 250, -1, 7, 8, 4, 3, 2, 1, 0, 255];
    let token_values = vec![-1, 1, 7, 0, 1];
    let logits = substrate.upload(&logits_values)?;
    let tie_logits = substrate.upload(&tie_logits_values)?;
    let bias = substrate.upload(&bias_values)?;
    let hash = substrate.upload(&hash_values)?;
    let tokens = substrate.upload(&token_values)?;

    let expected_ranked = expected_router(
        &logits_values,
        &bias_values,
        &hash_values,
        &token_values,
        N_TOKENS,
        0,
        false,
        true,
        true,
    );
    let parallel = run_router(
        &substrate,
        &module,
        KernelPath::Parallel,
        &logits,
        &bias,
        &hash,
        &tokens,
        N_TOKENS,
        0,
        false,
        true,
        true,
    )?;
    substrate.flush_commands()?;
    assert_router_output(&substrate, &parallel, &expected_ranked)?;

    let warp_ranked = run_router(
        &substrate,
        &module,
        KernelPath::WarpTopK,
        &logits,
        &bias,
        &hash,
        &tokens,
        N_TOKENS,
        0,
        false,
        true,
        true,
    )?;
    substrate.flush_commands()?;
    assert_router_output(&substrate, &warp_ranked, &expected_ranked)?;

    let warp_hash = run_router(
        &substrate,
        &module,
        KernelPath::WarpTopK,
        &logits,
        &bias,
        &hash,
        &tokens,
        N_TOKENS,
        0,
        true,
        false,
        true,
    )?;
    substrate.flush_commands()?;
    assert_router_output(
        &substrate,
        &warp_hash,
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

    let warp_ties = run_router(
        &substrate,
        &module,
        KernelPath::WarpTopK,
        &tie_logits,
        &bias,
        &hash,
        &tokens,
        N_TOKENS,
        0,
        false,
        false,
        true,
    )?;
    substrate.flush_commands()?;
    assert_router_output(
        &substrate,
        &warp_ties,
        &expected_router(
            &tie_logits_values,
            &bias_values,
            &hash_values,
            &token_values,
            N_TOKENS,
            0,
            false,
            false,
            true,
        ),
    )?;

    let single_warp = run_router(
        &substrate,
        &module,
        KernelPath::WarpTopK,
        &logits,
        &bias,
        &hash,
        &tokens,
        1,
        1,
        true,
        false,
        false,
    )?;
    substrate.end_commands()?;
    assert_router_output(
        &substrate,
        &single_warp,
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

    let short_selected = substrate.zeroed::<i32>((N_TOKENS as usize * TOP_K) - 1)?;
    assert!(matches!(
        launch_router(
            &module,
            substrate.stream(),
            KernelPath::Parallel,
            &logits,
            &bias,
            &hash,
            &tokens,
            N_TOKENS,
            0,
            false,
            true,
            true,
            short_selected,
        ),
        Err(RouterError::InvalidShape)
    ));
    assert!(dispatch_priority_matches_current_c());

    println!(
        "{{\"milestone\":\"M14.5b\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"parallel_bias_output_matches\":true,\"warp_bias_output_matches\":true,\"warp_hash_output_matches\":true,\"warp_invalid_token_fallback_matches\":true,\"warp_tie_order_matches\":true,\"warp_partial_block_matches\":true,\"single_token_warp_matches\":true,\"dispatch_priority_matches\":true,\"invalid_shape_rejected\":true,\"uses_shared_parallel_probabilities\":true,\"uses_warp_shuffle_topk\":true,\"uses_libdevice_link_path\":true,\"consumes_scalar_router_surface\":{},\"owns_router_select_parallel_kernel\":{},\"owns_router_select_warp_topk_kernel\":{},\"owns_parallel_and_warp_router_dispatch\":{},\"owns_current_c_dispatch_priority\":{},\"owns_routed_moe_or_hyperconnection\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5B_SCOPE.consumes_scalar_router_surface,
        M14_5B_SCOPE.owns_router_select_parallel_kernel,
        M14_5B_SCOPE.owns_router_select_warp_topk_kernel,
        M14_5B_SCOPE.owns_parallel_and_warp_router_dispatch,
        M14_5B_SCOPE.owns_current_c_dispatch_priority,
        M14_5B_SCOPE.owns_routed_moe_or_hyperconnection,
        M14_5B_SCOPE.owns_runtime_graph_integration,
        M14_5B_SCOPE.changes_default_route,
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum KernelPath {
    Parallel,
    WarpTopK,
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
fn run_router(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    path: KernelPath,
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
    let selected = substrate.zeroed::<i32>(n_tokens as usize * TOP_K)?;
    launch_router(
        module,
        substrate.stream(),
        path,
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
    path: KernelPath,
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
        || logits.len() < n_tokens as usize * N_EXPERT
        || selected.len() < n_tokens as usize * TOP_K
        || (has_bias && !hash_mode && bias.len() < N_EXPERT)
        || (hash_mode && hash.len() < HASH_ROWS as usize * TOP_K)
        || (hash_mode && use_token_buffer && tokens.len() < n_tokens as usize)
    {
        return Err(RouterError::InvalidShape);
    }
    let mut weights = DeviceBuffer::zeroed(stream, n_tokens as usize * TOP_K)?;
    let mut probs = DeviceBuffer::zeroed(stream, n_tokens as usize * N_EXPERT)?;
    let arguments = (
        n_tokens,
        token_scalar,
        HASH_ROWS,
        u32::from(has_bias && !hash_mode),
        u32::from(hash_mode),
        u32::from(use_token_buffer),
        logits,
        bias,
        hash,
        tokens,
        &mut selected,
        &mut weights,
        &mut probs,
    );
    match path {
        KernelPath::Parallel => module.router_select_parallel_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens, 1, 1),
                block_dim: (N_EXPERT as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            arguments.5,
            arguments.6,
            arguments.7,
            arguments.8,
            arguments.9,
            arguments.10,
            arguments.11,
            arguments.12,
        )?,
        KernelPath::WarpTopK => module.router_select_warp_topk_kernel(
            stream,
            LaunchConfig {
                grid_dim: (n_tokens.div_ceil(ROWS_PER_WARP_BLOCK), 1, 1),
                block_dim: (32, ROWS_PER_WARP_BLOCK, 1),
                shared_mem_bytes: 0,
            },
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
            arguments.5,
            arguments.6,
            arguments.7,
            arguments.8,
            arguments.9,
            arguments.10,
            arguments.11,
            arguments.12,
        )?,
    }
    Ok(RouterOutput {
        selected,
        weights,
        probs,
    })
}

fn ranked_logits() -> Vec<f32> {
    let mut logits = vec![-4.0_f32; N_TOKENS as usize * N_EXPERT];
    for token in 0..N_TOKENS as usize {
        for expert in 0..N_EXPERT {
            logits[token * N_EXPERT + expert] =
                -3.0 + ((expert * 13 + token * 7) % 31) as f32 * 0.0625;
        }
        for (rank, expert) in [42_usize, 17, 3, 200, 11, 99, 7].into_iter().enumerate() {
            logits[token * N_EXPERT + expert] = 2.5 - rank as f32 * 0.125 + token as f32 * 0.03125;
        }
    }
    logits
}

fn tie_logits() -> Vec<f32> {
    let mut logits = vec![-4.0_f32; N_TOKENS as usize * N_EXPERT];
    for token in 0..N_TOKENS as usize {
        for expert in [96_usize, 3, 65, 35, 7, 129] {
            logits[token * N_EXPERT + expert] = 3.0 + token as f32 * 0.03125;
        }
    }
    logits
}

fn bias_values() -> Vec<f32> {
    (0..N_EXPERT)
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
        let row = &logits[token_index * N_EXPERT..(token_index + 1) * N_EXPERT];
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
            hash[token * TOP_K..(token + 1) * TOP_K].to_vec()
        } else {
            let mut candidates = (0..N_EXPERT).collect::<Vec<_>>();
            candidates.sort_by(|left, right| {
                let left_score = row_probs[*left] + if has_bias { bias[*left] } else { 0.0 };
                let right_score = row_probs[*right] + if has_bias { bias[*right] } else { 0.0 };
                right_score
                    .total_cmp(&left_score)
                    .then_with(|| left.cmp(right))
            });
            candidates[..TOP_K]
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

fn dispatch_priority_matches_current_c() -> bool {
    let default = RouterSelectDispatchOptions {
        no_warp_router_select: false,
        no_parallel_router_select: false,
    };
    select_router_select_path(default) == RouterSelectPath::WarpTopK
        && select_router_select_path(RouterSelectDispatchOptions {
            no_warp_router_select: true,
            ..default
        }) == RouterSelectPath::Parallel
        && select_router_select_path(RouterSelectDispatchOptions {
            no_parallel_router_select: true,
            ..default
        }) == RouterSelectPath::Scalar
        && select_router_select_path(RouterSelectDispatchOptions {
            no_warp_router_select: true,
            no_parallel_router_select: true,
        }) == RouterSelectPath::Scalar
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
