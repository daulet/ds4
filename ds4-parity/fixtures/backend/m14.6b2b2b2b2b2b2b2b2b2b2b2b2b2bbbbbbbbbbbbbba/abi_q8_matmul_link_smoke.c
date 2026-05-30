#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 35u
#define OUT_DIM 10u
#define N_TOK 2u
#define BLOCKS ((IN_DIM + 31u) / 32u)
#define WEIGHT_BYTES ((uint64_t)OUT_DIM * BLOCKS * 34u)
#define OUT_ELEMENTS ((uint64_t)OUT_DIM * N_TOK)
#define X_ELEMENTS ((uint64_t)IN_DIM * N_TOK)

static void fill_packed_weights(uint8_t *weights) {
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        for (uint32_t block = 0; block < BLOCKS; ++block) {
            const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
            weights[base] = 0x00u;
            weights[base + 1u] = 0x3cu;
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                weights[base + 2u + lane] =
                    (uint8_t)((int32_t)((row * 5u + block * 7u + lane * 3u) % 19u) - 9);
            }
        }
    }
}

static void fill_activations(float *x) {
    for (uint32_t token = 0; token < N_TOK; ++token) {
        for (uint32_t column = 0; column < IN_DIM; ++column) {
            x[(uint64_t)token * IN_DIM + column] =
                (float)((int32_t)((token * 11u + column * 5u) % 21u) - 10);
        }
    }
}

static int8_t clamp_i8(int value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static void reference_native(float *out, const uint8_t *weights, const float *x, uint32_t n_tok) {
    for (uint32_t token = 0; token < n_tok; ++token) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            float total = 0.0f;
            for (uint32_t block = 0; block < BLOCKS; ++block) {
                const uint32_t start = block * 32u;
                const uint32_t count = IN_DIM - start < 32u ? IN_DIM - start : 32u;
                float maximum = 0.0f;
                for (uint32_t lane = 0; lane < count; ++lane) {
                    const float magnitude = fabsf(x[(uint64_t)token * IN_DIM + start + lane]);
                    if (magnitude > maximum) maximum = magnitude;
                }
                const float scale = maximum / 127.0f;
                const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
                const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
                int dot = 0;
                for (uint32_t lane = 0; lane < count; ++lane) {
                    const int quantized =
                        (int)nearbyintf(x[(uint64_t)token * IN_DIM + start + lane] * inverse);
                    dot += (int8_t)weights[base + 2u + lane] * clamp_i8(quantized);
                }
                total += scale * (float)dot;
            }
            out[(uint64_t)token * OUT_DIM + row] = total;
        }
    }
}

static void reference_dense(float *out, const uint8_t *weights, const float *x) {
    for (uint32_t token = 0; token < N_TOK; ++token) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            float total = 0.0f;
            for (uint32_t column = 0; column < IN_DIM; ++column) {
                const uint32_t block = column / 32u;
                const uint32_t lane = column % 32u;
                const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
                total += (float)(int8_t)weights[base + 2u + lane] *
                         x[(uint64_t)token * IN_DIM + column];
            }
            out[(uint64_t)token * OUT_DIM + row] = total;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count, float tolerance) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > tolerance) return 0;
    }
    return 1;
}

static int run_projection(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *x,
        const uint8_t *weights,
        uint32_t n_tok,
        float *values) {
    return ds4_gpu_matmul_q8_0_tensor(
               out, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x, n_tok) &&
           ds4_gpu_synchronize() &&
           ds4_gpu_tensor_read(out, 0, values, sizeof(float) * n_tok * OUT_DIM);
}

int main(void) {
    uint8_t *weights = malloc((size_t)WEIGHT_BYTES);
    float *x_values = malloc(sizeof(float) * X_ELEMENTS);
    if (!weights || !x_values) return 1;
    fill_packed_weights(weights);
    fill_activations(x_values);

    if (setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_NO_Q8_F32_CACHE", "1", 1) != 0 ||
        unsetenv("DS4_CUDA_NO_Q8_DP4A") != 0 ||
        unsetenv("DS4_CUDA_NO_Q8_BATCH_WARP") != 0 ||
        setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "0", 1) != 0) {
        return 2;
    }
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(weights, WEIGHT_BYTES)) return 3;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(float) * X_ELEMENTS);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float) * OUT_ELEMENTS);
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_values, sizeof(float) * X_ELEMENTS)) return 4;

    float native_expected[OUT_ELEMENTS] = {0};
    float dense_expected[OUT_ELEMENTS] = {0};
    float default_dp4a[OUT_ELEMENTS] = {0};
    float scalar[OUT_ELEMENTS] = {0};
    float batch_warp[OUT_ELEMENTS] = {0};
    float generic[OUT_ELEMENTS] = {0};
    float expanded_f16[OUT_ELEMENTS] = {0};
    float expanded_f32[OUT_ELEMENTS] = {0};
    reference_native(native_expected, weights, x_values, N_TOK);
    reference_dense(dense_expected, weights, x_values);

    if (!run_projection(out, x, weights, 1, default_dp4a) ||
        !close_array(default_dp4a, native_expected, OUT_DIM, 1.0e-4f)) {
        return 5;
    }
    if (setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1) != 0 ||
        !run_projection(out, x, weights, 1, scalar) ||
        !close_array(scalar, native_expected, OUT_DIM, 1.0e-4f) ||
        !close_array(scalar, default_dp4a, OUT_DIM, 1.0e-4f)) {
        return 6;
    }
    if (unsetenv("DS4_CUDA_NO_Q8_DP4A") != 0 ||
        !run_projection(out, x, weights, N_TOK, batch_warp) ||
        !close_array(batch_warp, native_expected, OUT_ELEMENTS, 1.0e-4f)) {
        return 7;
    }
    if (setenv("DS4_CUDA_NO_Q8_BATCH_WARP", "1", 1) != 0 ||
        !run_projection(out, x, weights, N_TOK, generic) ||
        !close_array(generic, native_expected, OUT_ELEMENTS, 1.0e-4f)) {
        return 8;
    }

    if (unsetenv("DS4_CUDA_NO_Q8_BATCH_WARP") != 0 ||
        unsetenv("DS4_CUDA_NO_Q8_F16_CACHE") != 0 ||
        setenv("DS4_CUDA_Q8_F16_ALL", "1", 1) != 0 ||
        !run_projection(out, x, weights, N_TOK, expanded_f16) ||
        !close_array(expanded_f16, dense_expected, OUT_ELEMENTS, 1.0e-3f)) {
        return 9;
    }
    if (unsetenv("DS4_CUDA_NO_Q8_F32_CACHE") != 0 ||
        setenv("DS4_CUDA_Q8_F32_ALL", "1", 1) != 0 ||
        !run_projection(out, x, weights, N_TOK, expanded_f32) ||
        !close_array(expanded_f32, dense_expected, OUT_ELEMENTS, 1.0e-4f)) {
        return 10;
    }
    if (ds4_gpu_matmul_q8_0_tensor(
            out, weights, WEIGHT_BYTES - 1u, 0, IN_DIM, OUT_DIM, x, N_TOK)) {
        return 11;
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(x_values);
    free(weights);
    puts("{\"c_linked_rust_staticlib\":true,\"native_single_token_dp4a_output_matches\":true,"
         "\"native_scalar_disable_output_matches\":true,\"native_batch_warp_output_matches\":true,"
         "\"native_generic_disable_output_matches\":true,\"expanded_f16_blas_output_matches\":true,"
         "\"expanded_f32_blas_output_matches\":true,\"invalid_range_rejected\":true,"
         "\"embedded_q8_matmul_kernels_loaded\":true}");
    return 0;
}
