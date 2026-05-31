#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TOP_K 512u

static const float *sort_scores;
static int use_packed_keys;

static uint64_t packed_key(float value, uint32_t index) {
    union {
        float value;
        uint32_t bits;
    } repr = {.value = value};
    const uint32_t ordered =
        (repr.bits & 0x80000000u) != 0u ? ~repr.bits : repr.bits ^ 0x80000000u;
    return ((uint64_t)ordered << 32) | (uint64_t)(UINT32_MAX - index);
}

static int compare_indices(const void *left_ptr, const void *right_ptr) {
    const uint32_t left = *(const uint32_t *)left_ptr;
    const uint32_t right = *(const uint32_t *)right_ptr;
    if (use_packed_keys) {
        const uint64_t left_key = packed_key(sort_scores[left], left);
        const uint64_t right_key = packed_key(sort_scores[right], right);
        if (left_key > right_key) return -1;
        if (left_key < right_key) return 1;
        return 0;
    }
    if (sort_scores[left] > sort_scores[right]) return -1;
    if (sort_scores[left] < sort_scores[right]) return 1;
    return left < right ? -1 : left > right;
}

static void build_scores(float *scores, uint32_t n_comp, int packed) {
    for (uint32_t comp = 0; comp < n_comp; ++comp) {
        scores[comp] = (float)((comp * 37u) % 1009u) - 100.0f;
    }
    if (packed) {
        scores[0] = NAN;
        scores[1] = INFINITY;
        scores[2] = 16384.0f;
        scores[3] = 16384.0f;
    } else {
        scores[0] = 16384.0f;
        scores[1] = 16384.0f;
    }
}

static int run_case(uint32_t n_comp, uint32_t top_k, int packed) {
    float *scores = malloc((size_t)n_comp * sizeof(float));
    uint32_t *expected = malloc((size_t)n_comp * sizeof(uint32_t));
    uint32_t *got = calloc((size_t)top_k, sizeof(uint32_t));
    ds4_gpu_tensor *scores_tensor = NULL;
    ds4_gpu_tensor *selected_tensor = NULL;
    int ok = 0;
    if (!scores || !expected || !got) goto done;
    build_scores(scores, n_comp, packed);
    for (uint32_t comp = 0; comp < n_comp; ++comp) expected[comp] = comp;
    sort_scores = scores;
    use_packed_keys = packed;
    qsort(expected, n_comp, sizeof(uint32_t), compare_indices);
    scores_tensor = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    selected_tensor = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    if (!scores_tensor || !selected_tensor ||
        !ds4_gpu_tensor_write(scores_tensor, 0, scores, (uint64_t)n_comp * sizeof(float)) ||
        !ds4_gpu_indexer_topk_tensor(selected_tensor, scores_tensor, n_comp, 1u, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(selected_tensor, 0, got, (uint64_t)top_k * sizeof(uint32_t))) {
        goto done;
    }
    for (uint32_t index = 0; index < top_k; ++index) {
        if (got[index] != expected[index]) goto done;
    }
    ok = 1;
done:
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(scores_tensor);
    free(got);
    free(expected);
    free(scores);
    return ok;
}

static int clear_options(void) {
    return unsetenv("DS4_CUDA_NO_TOPK1024") == 0 &&
           unsetenv("DS4_CUDA_NO_TOPK2048") == 0 &&
           unsetenv("DS4_CUDA_NO_TOPK8192") == 0 &&
           unsetenv("DS4_CUDA_NO_TOPK_CHUNKED") == 0;
}

int main(void) {
    float short_scores[3] = {3.0f, 2.0f, 1.0f};
    if (!ds4_gpu_init() || !clear_options()) return 1;

    if (!run_case(3u, 2u, 0)) return 2;
    if (!run_case(700u, TOP_K, 0)) return 3;
    if (setenv("DS4_CUDA_NO_TOPK1024", "1", 1) != 0 ||
        !run_case(1500u, TOP_K, 0)) return 4;
    if (!clear_options() || !run_case(4096u, TOP_K, 1)) return 5;
    if (!clear_options() || !run_case(9000u, TOP_K, 0)) return 6;
    if (setenv("DS4_CUDA_NO_TOPK1024", "1", 1) != 0 ||
        setenv("DS4_CUDA_NO_TOPK2048", "1", 1) != 0 ||
        !run_case(1500u, TOP_K, 0)) return 7;

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(sizeof(short_scores));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(2u * sizeof(uint32_t));
    ds4_gpu_tensor *short_selected = ds4_gpu_tensor_alloc(sizeof(uint32_t));
    if (!scores || !selected || !short_selected ||
        !ds4_gpu_tensor_write(scores, 0, short_scores, sizeof(short_scores))) return 8;
    if (ds4_gpu_indexer_topk_tensor(short_selected, scores, 3u, 1u, 2u) ||
        ds4_gpu_indexer_topk_tensor(selected, scores, 3u, 1u, 4u) ||
        ds4_gpu_indexer_topk_tensor(selected, scores, 0u, 1u, 2u) ||
        ds4_gpu_indexer_topk_tensor(NULL, scores, 3u, 1u, 2u) ||
        ds4_gpu_indexer_topk_tensor(selected, NULL, 3u, 1u, 2u)) return 9;

    ds4_gpu_tensor_free(short_selected);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"scalar_output_matches\":true,"
         "\"topk1024_output_matches\":true,\"topk2048_output_matches\":true,"
         "\"packed_key_output_matches\":true,\"chunked_tree_output_matches\":true,"
         "\"disabled_specialized_scalar_fallback_matches\":true,"
         "\"short_tensor_rejected\":true,\"top_k_overflow_rejected\":true,"
         "\"zero_dimension_rejected\":true,\"null_rejected\":true,"
         "\"embedded_topk_kernels_loaded\":true}");
    return 0;
}
