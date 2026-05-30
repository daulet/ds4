#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N_TOKENS 2u
#define N_EMBD 4u
#define N_HC 2u
#define SPLIT_STRIDE (2u * N_HC + N_HC * N_HC)
#define OUT_COUNT (N_TOKENS * N_EMBD)
#define RESIDUAL_COUNT (N_TOKENS * N_HC * N_EMBD)
#define DIRECT_WEIGHT_COUNT (N_TOKENS * N_HC)
#define SPLIT_COUNT (N_TOKENS * SPLIT_STRIDE)

static void reference_weighted_sum(
        float out[OUT_COUNT],
        const float residual_hc[RESIDUAL_COUNT],
        const float *weights,
        uint32_t stride) {
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        for (uint32_t dimension = 0; dimension < N_EMBD; ++dimension) {
            float accumulator = 0.0f;
            for (uint32_t source_hc = 0; source_hc < N_HC; ++source_hc) {
                accumulator += residual_hc[(token * N_HC + source_hc) * N_EMBD + dimension] *
                               weights[token * stride + source_hc];
            }
            out[token * N_EMBD + dimension] = accumulator;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-6f) return 0;
    }
    return 1;
}

static int run_direct(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *residual_hc,
        ds4_gpu_tensor *weights,
        float got[OUT_COUNT]) {
    return ds4_gpu_hc_weighted_sum_tensor(out, residual_hc, weights, N_EMBD, N_HC) &&
           ds4_gpu_synchronize() &&
           ds4_gpu_tensor_read(out, 0, got, sizeof(float) * OUT_COUNT);
}

static int run_split(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *residual_hc,
        ds4_gpu_tensor *split,
        float got[OUT_COUNT]) {
    return ds4_gpu_hc_weighted_sum_split_tensor(out, residual_hc, split, N_EMBD, N_HC) &&
           ds4_gpu_synchronize() &&
           ds4_gpu_tensor_read(out, 0, got, sizeof(float) * OUT_COUNT);
}

int main(void) {
    const float residual_values[RESIDUAL_COUNT] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
        -1.0f, 2.0f, -3.0f, 4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
    };
    const float direct_weights[DIRECT_WEIGHT_COUNT] = {0.25f, 0.75f, -0.5f, 1.25f};
    const float split_values[SPLIT_COUNT] = {
        0.5f, -0.25f, 100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f,
        -1.0f, 0.5f, -50.0f, -51.0f, -52.0f, -53.0f, -54.0f, -55.0f,
    };
    float expected_direct[OUT_COUNT] = {0};
    float expected_split[OUT_COUNT] = {0};
    float got[OUT_COUNT] = {0};

    reference_weighted_sum(expected_direct, residual_values, direct_weights, N_HC);
    reference_weighted_sum(expected_split, residual_values, split_values, SPLIT_STRIDE);
    if (!ds4_gpu_init()) return 1;

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float) * OUT_COUNT);
    ds4_gpu_tensor *residual_hc = ds4_gpu_tensor_alloc(sizeof(residual_values));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sizeof(direct_weights));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(sizeof(split_values));
    ds4_gpu_tensor *short_residual =
        ds4_gpu_tensor_alloc(sizeof(float) * (RESIDUAL_COUNT - 1u));
    ds4_gpu_tensor *short_split =
        ds4_gpu_tensor_alloc(sizeof(float) * (SPLIT_STRIDE + N_HC - 1u));
    if (!out || !residual_hc || !weights || !split || !short_residual || !short_split ||
        !ds4_gpu_tensor_write(residual_hc, 0, residual_values, sizeof(residual_values)) ||
        !ds4_gpu_tensor_write(weights, 0, direct_weights, sizeof(direct_weights)) ||
        !ds4_gpu_tensor_write(split, 0, split_values, sizeof(split_values)) ||
        !ds4_gpu_tensor_write(
            short_residual, 0, residual_values, sizeof(float) * (RESIDUAL_COUNT - 1u)) ||
        !ds4_gpu_tensor_write(
            short_split, 0, split_values, sizeof(float) * (SPLIT_STRIDE + N_HC - 1u))) {
        return 2;
    }
    if (!run_direct(out, residual_hc, weights, got) ||
        !close_array(got, expected_direct, OUT_COUNT)) {
        return 3;
    }
    if (!run_split(out, residual_hc, split, got) ||
        !close_array(got, expected_split, OUT_COUNT)) {
        return 4;
    }
    if (ds4_gpu_hc_weighted_sum_tensor(out, short_residual, weights, N_EMBD, N_HC)) {
        return 5;
    }
    if (ds4_gpu_hc_weighted_sum_split_tensor(out, residual_hc, short_split, N_EMBD, N_HC)) {
        return 6;
    }
    if (ds4_gpu_hc_weighted_sum_tensor(out, residual_hc, weights, 0u, N_HC)) {
        return 7;
    }

    ds4_gpu_tensor_free(short_split);
    ds4_gpu_tensor_free(short_residual);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(residual_hc);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"direct_weighted_sum_matches\":true,"
         "\"split_stride_weighted_sum_matches\":true,\"short_residual_rejected\":true,"
         "\"short_split_rejected\":true,\"zero_shape_rejected\":true,"
         "\"embedded_hc_weighted_sum_kernel_loaded\":true}");
    return 0;
}
