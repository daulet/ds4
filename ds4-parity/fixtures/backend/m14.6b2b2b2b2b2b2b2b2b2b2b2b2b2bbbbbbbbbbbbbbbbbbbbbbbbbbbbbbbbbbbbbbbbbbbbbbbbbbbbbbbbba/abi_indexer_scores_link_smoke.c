#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    DIRECT_N_COMP = 4,
    FIXED_N_HEAD = 64,
    FIXED_HEAD_DIM = 128,
    WMMA_N_COMP = 3,
    WMMA_N_TOKENS = 2,
};

static int equal_values(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (actual[i] == expected[i]) continue;
        if (isinf(actual[i]) && isinf(expected[i]) &&
            signbit(actual[i]) == signbit(expected[i])) continue;
        return 0;
    }
    return 1;
}

static int run_wmma_prefill(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp) {
    const float expected[WMMA_N_COMP * WMMA_N_TOKENS] = {
        0.0f, -INFINITY, -INFINITY, 0.0f, 0.0f, -INFINITY,
    };
    float got[WMMA_N_COMP * WMMA_N_TOKENS] = {0};
    return ds4_gpu_indexer_scores_prefill_tensor(
               scores, q, weights, index_comp, WMMA_N_COMP, WMMA_N_TOKENS,
               FIXED_N_HEAD, FIXED_HEAD_DIM, 1, 0.5f) &&
           ds4_gpu_tensor_read(scores, 0, got, sizeof(got)) &&
           equal_values(got, expected, WMMA_N_COMP * WMMA_N_TOKENS);
}

int main(void) {
    const float scalar_q[4] = {1.0f, 2.0f, -1.0f, 1.0f};
    const float scalar_weights[2] = {1.0f, 0.5f};
    const float scalar_comp[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float scalar_expected[3] = {0.5f, 1.25f, 1.5f};
    float scalar_got[3] = {0};

    float direct_q[FIXED_N_HEAD * FIXED_HEAD_DIM] = {0};
    float direct_weights[FIXED_N_HEAD];
    float direct_comp[DIRECT_N_COMP * FIXED_HEAD_DIM] = {0};
    const float direct_expected[DIRECT_N_COMP] = {32.0f, 64.0f, 0.0f, 0.0f};
    float direct_got[DIRECT_N_COMP] = {0};
    for (uint32_t head = 0; head < FIXED_N_HEAD; ++head) {
        direct_q[head * FIXED_HEAD_DIM] = 1.0f;
        direct_weights[head] = 1.0f;
    }
    direct_comp[0] = 1.0f;
    direct_comp[FIXED_HEAD_DIM] = 2.0f;
    direct_comp[2 * FIXED_HEAD_DIM] = -1.0f;
    direct_comp[3 * FIXED_HEAD_DIM] = NAN;

    const float wmma_q[WMMA_N_TOKENS * FIXED_N_HEAD * FIXED_HEAD_DIM] = {0};
    float wmma_weights[WMMA_N_TOKENS * FIXED_N_HEAD];
    const float wmma_comp[WMMA_N_COMP * FIXED_HEAD_DIM] = {0};
    float decode_got[WMMA_N_COMP * WMMA_N_TOKENS] = {0};
    const float decode_expected[WMMA_N_COMP * WMMA_N_TOKENS] = {0};
    for (uint32_t i = 0; i < WMMA_N_TOKENS * FIXED_N_HEAD; ++i) {
        wmma_weights[i] = 1.0f;
    }

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *scalar_scores = ds4_gpu_tensor_alloc(sizeof(scalar_expected));
    ds4_gpu_tensor *short_scores = ds4_gpu_tensor_alloc(sizeof(scalar_expected) - sizeof(float));
    ds4_gpu_tensor *scalar_q_t = ds4_gpu_tensor_alloc(sizeof(scalar_q));
    ds4_gpu_tensor *scalar_weights_t = ds4_gpu_tensor_alloc(sizeof(scalar_weights));
    ds4_gpu_tensor *scalar_comp_t = ds4_gpu_tensor_alloc(sizeof(scalar_comp));
    ds4_gpu_tensor *direct_scores = ds4_gpu_tensor_alloc(sizeof(direct_expected));
    ds4_gpu_tensor *direct_q_t = ds4_gpu_tensor_alloc(sizeof(direct_q));
    ds4_gpu_tensor *direct_weights_t = ds4_gpu_tensor_alloc(sizeof(direct_weights));
    ds4_gpu_tensor *direct_comp_t = ds4_gpu_tensor_alloc(sizeof(direct_comp));
    ds4_gpu_tensor *wmma_scores = ds4_gpu_tensor_alloc(sizeof(decode_expected));
    ds4_gpu_tensor *wmma_q_t = ds4_gpu_tensor_alloc(sizeof(wmma_q));
    ds4_gpu_tensor *wmma_weights_t = ds4_gpu_tensor_alloc(sizeof(wmma_weights));
    ds4_gpu_tensor *wmma_comp_t = ds4_gpu_tensor_alloc(sizeof(wmma_comp));
    if (!scalar_scores || !short_scores || !scalar_q_t || !scalar_weights_t || !scalar_comp_t ||
        !direct_scores || !direct_q_t || !direct_weights_t || !direct_comp_t ||
        !wmma_scores || !wmma_q_t || !wmma_weights_t || !wmma_comp_t) return 2;

    if (!ds4_gpu_tensor_write(scalar_q_t, 0, scalar_q, sizeof(scalar_q)) ||
        !ds4_gpu_tensor_write(scalar_weights_t, 0, scalar_weights, sizeof(scalar_weights)) ||
        !ds4_gpu_tensor_write(scalar_comp_t, 0, scalar_comp, sizeof(scalar_comp)) ||
        !ds4_gpu_tensor_write(direct_q_t, 0, direct_q, sizeof(direct_q)) ||
        !ds4_gpu_tensor_write(direct_weights_t, 0, direct_weights, sizeof(direct_weights)) ||
        !ds4_gpu_tensor_write(direct_comp_t, 0, direct_comp, sizeof(direct_comp)) ||
        !ds4_gpu_tensor_write(wmma_q_t, 0, wmma_q, sizeof(wmma_q)) ||
        !ds4_gpu_tensor_write(wmma_weights_t, 0, wmma_weights, sizeof(wmma_weights)) ||
        !ds4_gpu_tensor_write(wmma_comp_t, 0, wmma_comp, sizeof(wmma_comp))) return 3;

    if (!ds4_gpu_indexer_score_one_tensor(
            scalar_scores, scalar_q_t, scalar_weights_t, scalar_comp_t, 3, 2, 2, 0.5f) ||
        !ds4_gpu_tensor_read(scalar_scores, 0, scalar_got, sizeof(scalar_got)) ||
        !equal_values(scalar_got, scalar_expected, 3)) return 4;

    if (!ds4_gpu_indexer_score_one_tensor(
            direct_scores, direct_q_t, direct_weights_t, direct_comp_t,
            DIRECT_N_COMP, FIXED_N_HEAD, FIXED_HEAD_DIM, 0.5f) ||
        !ds4_gpu_tensor_read(direct_scores, 0, direct_got, sizeof(direct_got)) ||
        !equal_values(direct_got, direct_expected, DIRECT_N_COMP)) return 5;

    ds4_gpu_set_quality(false);
    if (!run_wmma_prefill(wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t)) return 6;
    if (setenv("DS4_CUDA_NO_INDEXER_WMMA128", "1", 1) != 0 ||
        !run_wmma_prefill(wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t)) return 7;
    if (setenv("DS4_CUDA_NO_INDEXER_WMMA64", "1", 1) != 0 ||
        !run_wmma_prefill(wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t)) return 8;
    if (setenv("DS4_CUDA_NO_INDEXER_WMMA32", "1", 1) != 0 ||
        !run_wmma_prefill(wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t)) return 9;
    if (unsetenv("DS4_CUDA_NO_INDEXER_WMMA128") != 0 ||
        unsetenv("DS4_CUDA_NO_INDEXER_WMMA64") != 0 ||
        unsetenv("DS4_CUDA_NO_INDEXER_WMMA32") != 0) return 10;

    ds4_gpu_set_quality(true);
    if (!run_wmma_prefill(wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t)) return 11;
    ds4_gpu_set_quality(false);
    if (!ds4_gpu_indexer_scores_decode_batch_tensor(
            wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t,
            WMMA_N_COMP, WMMA_N_TOKENS, 2, FIXED_N_HEAD, FIXED_HEAD_DIM, 1, 0.5f) ||
        !ds4_gpu_tensor_read(wmma_scores, 0, decode_got, sizeof(decode_got)) ||
        !equal_values(decode_got, decode_expected, WMMA_N_COMP * WMMA_N_TOKENS)) return 12;

    if (ds4_gpu_indexer_score_one_tensor(
            short_scores, scalar_q_t, scalar_weights_t, scalar_comp_t, 3, 2, 2, 0.5f) ||
        ds4_gpu_indexer_scores_prefill_tensor(
            wmma_scores, wmma_q_t, wmma_weights_t, wmma_comp_t,
            WMMA_N_COMP, WMMA_N_TOKENS, FIXED_N_HEAD, FIXED_HEAD_DIM, 0, 0.5f) ||
        ds4_gpu_indexer_score_one_tensor(
            scalar_scores, NULL, scalar_weights_t, scalar_comp_t, 3, 2, 2, 0.5f) ||
        !ds4_gpu_synchronize()) return 13;

    ds4_gpu_tensor_free(wmma_comp_t);
    ds4_gpu_tensor_free(wmma_weights_t);
    ds4_gpu_tensor_free(wmma_q_t);
    ds4_gpu_tensor_free(wmma_scores);
    ds4_gpu_tensor_free(direct_comp_t);
    ds4_gpu_tensor_free(direct_weights_t);
    ds4_gpu_tensor_free(direct_q_t);
    ds4_gpu_tensor_free(direct_scores);
    ds4_gpu_tensor_free(scalar_comp_t);
    ds4_gpu_tensor_free(scalar_weights_t);
    ds4_gpu_tensor_free(scalar_q_t);
    ds4_gpu_tensor_free(short_scores);
    ds4_gpu_tensor_free(scalar_scores);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"scalar_output_matches\":true,\"direct_one_output_matches\":true,\"wmma128_prefill_matches\":true,\"wmma64_prefill_matches\":true,\"wmma32_prefill_matches\":true,\"wmma_prefill_matches\":true,\"quality_mode_scalar_fallback_matches\":true,\"decode_batch_output_matches\":true,\"short_tensor_rejected\":true,\"zero_ratio_rejected\":true,\"null_rejected\":true,\"embedded_score_kernels_loaded\":true}");
    return 0;
}
