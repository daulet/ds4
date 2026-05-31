#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum {
    N_COMP = 5,
    N_TOKENS = 2,
    TOP_K = 2,
    MASK_COUNT = N_COMP * N_TOKENS,
    SELECTED_COUNT = TOP_K * N_TOKENS,
};

static int equal_mask(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (actual[i] == expected[i]) continue;
        if (isinf(actual[i]) && isinf(expected[i]) &&
            signbit(actual[i]) == signbit(expected[i])) continue;
        return 0;
    }
    return 1;
}

int main(void) {
    const uint32_t selected[SELECTED_COUNT] = {1, 4, 0, 3};
    const float expected[MASK_COUNT] = {
        -INFINITY, 0.0f, -INFINITY, -INFINITY, 0.0f,
        0.0f, -INFINITY, -INFINITY, 0.0f, -INFINITY,
    };
    float got[MASK_COUNT] = {0};

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *mask = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *short_topk = ds4_gpu_tensor_alloc(sizeof(selected) - sizeof(uint32_t));
    ds4_gpu_tensor *short_mask = ds4_gpu_tensor_alloc(sizeof(got) - sizeof(float));
    if (!topk || !mask || !short_topk || !short_mask) return 2;
    if (!ds4_gpu_tensor_write(topk, 0, selected, sizeof(selected)) ||
        !ds4_gpu_dsv4_topk_mask_tensor(mask, topk, N_COMP, N_TOKENS, TOP_K) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mask, 0, got, sizeof(got)) ||
        !equal_mask(got, expected, MASK_COUNT)) return 3;

    if (ds4_gpu_dsv4_topk_mask_tensor(short_mask, topk, N_COMP, N_TOKENS, TOP_K) ||
        ds4_gpu_dsv4_topk_mask_tensor(mask, short_topk, N_COMP, N_TOKENS, TOP_K) ||
        ds4_gpu_dsv4_topk_mask_tensor(mask, topk, N_COMP, 0, TOP_K) ||
        ds4_gpu_dsv4_topk_mask_tensor(mask, NULL, N_COMP, N_TOKENS, TOP_K)) return 4;

    ds4_gpu_tensor_free(short_mask);
    ds4_gpu_tensor_free(short_topk);
    ds4_gpu_tensor_free(mask);
    ds4_gpu_tensor_free(topk);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"multi_token_mask_output_matches\":true,\"selected_zero_mask_matches\":true,\"excluded_negative_infinity_mask_matches\":true,\"short_mask_rejected\":true,\"short_topk_rejected\":true,\"zero_dimension_rejected\":true,\"null_rejected\":true,\"embedded_topk_mask_kernel_loaded\":true}");
    return 0;
}
