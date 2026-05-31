#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N_EXPERT 256u
#define TOP_K 6u
#define N_TOKENS 5u
#define HASH_ROWS 2u
#define FLOAT_TOLERANCE 2.0e-5f

struct router_model {
    float bias[N_EXPERT];
    int32_t hash[HASH_ROWS][TOP_K];
};

static float router_prob(float logit) {
    float softplus;
    if (logit > 20.0f) {
        softplus = logit;
    } else if (logit < -20.0f) {
        softplus = expf(logit);
    } else {
        softplus = log1pf(expf(logit));
    }
    return sqrtf(softplus);
}

static void expected_router(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *logits,
        const float *bias,
        const int32_t *hash,
        const int32_t *tokens,
        uint32_t n_tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        int has_bias,
        int hash_mode,
        int use_token_buffer) {
    for (uint32_t row = 0; row < n_tokens; ++row) {
        const uint32_t prob_base = row * N_EXPERT;
        const uint32_t selected_base = row * TOP_K;
        for (uint32_t expert = 0; expert < N_EXPERT; ++expert) {
            probs[prob_base + expert] = router_prob(logits[prob_base + expert]);
        }
        int32_t chosen[TOP_K] = {-1, -1, -1, -1, -1, -1};
        if (hash_mode) {
            int32_t token = use_token_buffer ? tokens[row] : token_scalar;
            if (token < 0 || (uint32_t)token >= hash_rows) token = 0;
            for (uint32_t out = 0; out < TOP_K; ++out) {
                chosen[out] = hash[(uint32_t)token * TOP_K + out];
            }
        } else {
            for (uint32_t expert = 0; expert < N_EXPERT; ++expert) {
                float score = probs[prob_base + expert] + (has_bias ? bias[expert] : 0.0f);
                for (uint32_t out = 0; out < TOP_K; ++out) {
                    const int32_t current = chosen[out];
                    const float current_score = current < 0
                            ? -INFINITY
                            : probs[prob_base + (uint32_t)current] +
                                      (has_bias ? bias[(uint32_t)current] : 0.0f);
                    if (score > current_score) {
                        for (uint32_t shift = TOP_K - 1; shift > out; --shift) {
                            chosen[shift] = chosen[shift - 1];
                        }
                        chosen[out] = (int32_t)expert;
                        break;
                    }
                }
            }
        }
        float sum = 0.0f;
        for (uint32_t out = 0; out < TOP_K; ++out) {
            const int32_t expert = chosen[out];
            const float value = expert >= 0 && expert < (int32_t)N_EXPERT
                    ? probs[prob_base + (uint32_t)expert]
                    : 0.0f;
            selected[selected_base + out] = expert;
            weights[selected_base + out] = value;
            sum += value;
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (uint32_t out = 0; out < TOP_K; ++out) {
            weights[selected_base + out] = weights[selected_base + out] / sum * 1.5f;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > FLOAT_TOLERANCE) return 0;
    }
    return 1;
}

static int equal_indices(const int32_t *actual, const int32_t *expected, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) return 0;
    }
    return 1;
}

static int sentinel_intact(const int32_t *selected, const float *weights, const float *probs) {
    for (uint32_t index = 0; index < TOP_K; ++index) {
        if (selected[index] != -77 || weights[index] != 77.0f) return 0;
    }
    for (uint32_t index = 0; index < N_EXPERT; ++index) {
        if (probs[index] != 77.0f) return 0;
    }
    return 1;
}

static void make_logits(float *values) {
    for (uint32_t row = 0; row < N_TOKENS; ++row) {
        for (uint32_t expert = 0; expert < N_EXPERT; ++expert) {
            values[row * N_EXPERT + expert] =
                    -3.0f + (float)((expert * 13u + row * 7u) % 31u) * 0.0625f;
        }
        const uint32_t ranked[] = {42u, 17u, 3u, 200u, 11u, 99u, 7u};
        for (uint32_t rank = 0; rank < sizeof(ranked) / sizeof(ranked[0]); ++rank) {
            values[row * N_EXPERT + ranked[rank]] =
                    2.5f - (float)rank * 0.125f + (float)row * 0.03125f;
        }
    }
}

static int set_warp_path(void) {
    return unsetenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == 0 &&
           unsetenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == 0;
}

static int set_parallel_path(void) {
    return setenv("DS4_CUDA_NO_WARP_ROUTER_SELECT", "1", 1) == 0 &&
           unsetenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == 0;
}

static int set_scalar_path(void) {
    return setenv("DS4_CUDA_NO_WARP_ROUTER_SELECT", "1", 1) == 0 &&
           setenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT", "1", 1) == 0;
}

int main(void) {
    struct router_model model = {0};
    model.bias[7] = 0.75f;
    for (uint32_t expert = 0; expert < N_EXPERT; ++expert) {
        model.bias[expert] += (float)(expert % 5u) * 0.001f;
    }
    const int32_t hash[HASH_ROWS][TOP_K] = {{9, 1, 250, -1, 7, 8}, {4, 3, 2, 1, 0, 255}};
    for (uint32_t row = 0; row < HASH_ROWS; ++row) {
        for (uint32_t out = 0; out < TOP_K; ++out) model.hash[row][out] = hash[row][out];
    }
    const uint64_t hash_offset = offsetof(struct router_model, hash);

    float batch_logits[N_TOKENS * N_EXPERT];
    float single_logits[N_EXPERT];
    float tie_logits[N_EXPERT] = {0};
    int32_t token_values[N_TOKENS] = {-1, 1, 7, 0, 1};
    make_logits(batch_logits);
    for (uint32_t expert = 0; expert < N_EXPERT; ++expert) single_logits[expert] = batch_logits[expert];

    int32_t actual_selected[N_TOKENS * TOP_K] = {0};
    int32_t expected_selected[N_TOKENS * TOP_K] = {0};
    float actual_weights[N_TOKENS * TOP_K] = {0};
    float expected_weights[N_TOKENS * TOP_K] = {0};
    float actual_probs[N_TOKENS * N_EXPERT] = {0};
    float expected_probs[N_TOKENS * N_EXPERT] = {0};

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(&model, sizeof(model))) return 1;
    ds4_gpu_tensor *single_selected = ds4_gpu_tensor_alloc(TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *single_weights = ds4_gpu_tensor_alloc(TOP_K * sizeof(float));
    ds4_gpu_tensor *single_probs = ds4_gpu_tensor_alloc(N_EXPERT * sizeof(float));
    ds4_gpu_tensor *single_logit_tensor = ds4_gpu_tensor_alloc(N_EXPERT * sizeof(float));
    ds4_gpu_tensor *short_selected = ds4_gpu_tensor_alloc((TOP_K - 1u) * sizeof(int32_t));
    ds4_gpu_tensor *batch_selected = ds4_gpu_tensor_alloc(N_TOKENS * TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *batch_weights = ds4_gpu_tensor_alloc(N_TOKENS * TOP_K * sizeof(float));
    ds4_gpu_tensor *batch_probs = ds4_gpu_tensor_alloc(N_TOKENS * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *batch_logit_tensor = ds4_gpu_tensor_alloc(N_TOKENS * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(N_TOKENS * sizeof(int32_t));
    if (!single_selected || !single_weights || !single_probs || !single_logit_tensor ||
        !short_selected || !batch_selected || !batch_weights || !batch_probs ||
        !batch_logit_tensor || !tokens ||
        !ds4_gpu_tensor_write(single_logit_tensor, 0, single_logits, sizeof(single_logits)) ||
        !ds4_gpu_tensor_write(batch_logit_tensor, 0, batch_logits, sizeof(batch_logits)) ||
        !ds4_gpu_tensor_write(tokens, 0, token_values, sizeof(token_values))) return 2;

    expected_router(expected_selected, expected_weights, expected_probs, single_logits,
                    model.bias, &model.hash[0][0], token_values, 1, 0, HASH_ROWS, 1, 0, 0);
    if (!set_warp_path() ||
        !ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                      sizeof(model), 0, hash_offset, HASH_ROWS, 0, 0, 0, true,
                                      false, single_logit_tensor) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_read(single_weights, 0, actual_weights, TOP_K * sizeof(float)) ||
        !ds4_gpu_tensor_read(single_probs, 0, actual_probs, N_EXPERT * sizeof(float)) ||
        !equal_indices(actual_selected, expected_selected, TOP_K) ||
        !close_array(actual_weights, expected_weights, TOP_K) ||
        !close_array(actual_probs, expected_probs, N_EXPERT)) return 3;

    expected_router(expected_selected, expected_weights, expected_probs, single_logits,
                    model.bias, &model.hash[0][0], token_values, 1, 7, HASH_ROWS, 0, 1, 0);
    if (!ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                      sizeof(model), 0, hash_offset, HASH_ROWS, 7, 0, 0, false,
                                      true, single_logit_tensor) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_read(single_weights, 0, actual_weights, TOP_K * sizeof(float)) ||
        !equal_indices(actual_selected, expected_selected, TOP_K) ||
        !close_array(actual_weights, expected_weights, TOP_K)) return 4;

    expected_router(expected_selected, expected_weights, expected_probs, single_logits,
                    model.bias, &model.hash[0][0], token_values, 1, 0, HASH_ROWS, 1, 0, 0);
    if (!set_parallel_path() ||
        !ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                      sizeof(model), 0, hash_offset, HASH_ROWS, 0, 0, 0, true,
                                      false, single_logit_tensor) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_read(single_weights, 0, actual_weights, TOP_K * sizeof(float)) ||
        !equal_indices(actual_selected, expected_selected, TOP_K) ||
        !close_array(actual_weights, expected_weights, TOP_K)) return 5;

    expected_router(expected_selected, expected_weights, expected_probs, single_logits,
                    model.bias, &model.hash[0][0], token_values, 1, 1, HASH_ROWS, 0, 1, 0);
    if (!set_scalar_path() ||
        !ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                      sizeof(model), 0, hash_offset, HASH_ROWS, 1, 0, 0, false,
                                      true, single_logit_tensor) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_read(single_weights, 0, actual_weights, TOP_K * sizeof(float)) ||
        !equal_indices(actual_selected, expected_selected, TOP_K) ||
        !close_array(actual_weights, expected_weights, TOP_K)) return 6;

    expected_router(expected_selected, expected_weights, expected_probs, batch_logits,
                    model.bias, &model.hash[0][0], token_values, N_TOKENS, 0, HASH_ROWS, 1, 0,
                    1);
    if (!set_warp_path() ||
        !ds4_gpu_router_select_batch_tensor(batch_selected, batch_weights, batch_probs, &model,
                                            sizeof(model), 0, hash_offset, HASH_ROWS, 0, 0, true,
                                            false, batch_logit_tensor, tokens, N_TOKENS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(batch_selected, 0, actual_selected, sizeof(actual_selected)) ||
        !ds4_gpu_tensor_read(batch_weights, 0, actual_weights, sizeof(actual_weights)) ||
        !ds4_gpu_tensor_read(batch_probs, 0, actual_probs, sizeof(actual_probs)) ||
        !equal_indices(actual_selected, expected_selected, N_TOKENS * TOP_K) ||
        !close_array(actual_weights, expected_weights, N_TOKENS * TOP_K) ||
        !close_array(actual_probs, expected_probs, N_TOKENS * N_EXPERT)) return 7;

    expected_router(expected_selected, expected_weights, expected_probs, batch_logits,
                    model.bias, &model.hash[0][0], token_values, N_TOKENS, 0, HASH_ROWS, 0, 1,
                    1);
    if (!ds4_gpu_router_select_batch_tensor(batch_selected, batch_weights, batch_probs, &model,
                                            sizeof(model), 0, hash_offset, HASH_ROWS, 0, 0, false,
                                            true, batch_logit_tensor, tokens, N_TOKENS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(batch_selected, 0, actual_selected, sizeof(actual_selected)) ||
        !ds4_gpu_tensor_read(batch_weights, 0, actual_weights, sizeof(actual_weights)) ||
        !equal_indices(actual_selected, expected_selected, N_TOKENS * TOP_K) ||
        !close_array(actual_weights, expected_weights, N_TOKENS * TOP_K)) return 8;

    for (uint32_t index = 0; index < TOP_K; ++index) expected_selected[index] = (int32_t)index;
    if (!ds4_gpu_tensor_write(single_logit_tensor, 0, tie_logits, sizeof(tie_logits)) ||
        !ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                      sizeof(model), 0, hash_offset, HASH_ROWS, 0, 0, 0, false,
                                      false, single_logit_tensor) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !equal_indices(actual_selected, expected_selected, TOP_K)) return 9;

    for (uint32_t index = 0; index < TOP_K; ++index) {
        actual_selected[index] = -77;
        actual_weights[index] = 77.0f;
    }
    for (uint32_t index = 0; index < N_EXPERT; ++index) actual_probs[index] = 77.0f;
    if (!ds4_gpu_tensor_write(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(single_weights, 0, actual_weights, TOP_K * sizeof(float)) ||
        !ds4_gpu_tensor_write(single_probs, 0, actual_probs, N_EXPERT * sizeof(float)) ||
        ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                     hash_offset - 1u, 0, hash_offset, HASH_ROWS, 0, 0, 0,
                                     true, false, single_logit_tensor) ||
        ds4_gpu_router_select_tensor(short_selected, single_weights, single_probs, &model,
                                     sizeof(model), 0, hash_offset, HASH_ROWS, 0, 0, 0, false,
                                     false, single_logit_tensor) ||
        ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                     sizeof(model), 0, hash_offset, HASH_ROWS, 0, 2, 0, false,
                                     false, single_logit_tensor) ||
        ds4_gpu_router_select_tensor(single_selected, single_weights, single_probs, &model,
                                     sizeof(model), 0, hash_offset, 0, 0, 0, 0, false, true,
                                     single_logit_tensor) ||
        ds4_gpu_router_select_tensor(NULL, single_weights, single_probs, &model, sizeof(model), 0,
                                     hash_offset, HASH_ROWS, 0, 0, 0, false, false,
                                     single_logit_tensor) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_selected, 0, actual_selected, TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_read(single_weights, 0, actual_weights, TOP_K * sizeof(float)) ||
        !ds4_gpu_tensor_read(single_probs, 0, actual_probs, N_EXPERT * sizeof(float)) ||
        !sentinel_intact(actual_selected, actual_weights, actual_probs)) return 10;

    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(batch_logit_tensor);
    ds4_gpu_tensor_free(batch_probs);
    ds4_gpu_tensor_free(batch_weights);
    ds4_gpu_tensor_free(batch_selected);
    ds4_gpu_tensor_free(short_selected);
    ds4_gpu_tensor_free(single_logit_tensor);
    ds4_gpu_tensor_free(single_probs);
    ds4_gpu_tensor_free(single_weights);
    ds4_gpu_tensor_free(single_selected);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_bias_default_warp_matches\":true,"
         "\"single_hash_invalid_token_fallback_matches\":true,\"forced_parallel_matches\":true,"
         "\"forced_scalar_matches\":true,\"batch_warp_partial_block_matches\":true,"
         "\"batch_hash_invalid_token_fallback_matches\":true,\"tie_order_matches\":true,"
         "\"invalid_model_range_preserves_output\":true,\"short_span_rejected\":true,"
         "\"invalid_group_rejected\":true,\"invalid_hash_rows_rejected\":true,"
         "\"null_rejected\":true,\"embedded_router_kernels_loaded\":true}");
    return 0;
}
