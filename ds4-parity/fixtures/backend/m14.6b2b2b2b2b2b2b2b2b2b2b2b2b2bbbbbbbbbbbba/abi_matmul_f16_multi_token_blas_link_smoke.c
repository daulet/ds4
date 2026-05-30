#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 37u
#define OUT_DIM 3u
#define N_TOK 2u
#define WEIGHT_BYTES (OUT_DIM * IN_DIM * sizeof(uint16_t))
#define OUTPUT_COUNT (OUT_DIM * N_TOK)

static float f16_value(uint16_t bits) {
    switch (bits) {
        case 0xc400: return -4.0f;
        case 0xc000: return -2.0f;
        case 0xbc00: return -1.0f;
        case 0xb800: return -0.5f;
        case 0x3400: return 0.25f;
        case 0x3800: return 0.5f;
        case 0x3c00: return 1.0f;
        case 0x4000: return 2.0f;
        case 0x4400: return 4.0f;
        default: return 0.0f;
    }
}

static void reference_projection(
        float out[OUTPUT_COUNT],
        const uint16_t weights[OUT_DIM * IN_DIM],
        const float input[N_TOK * IN_DIM]) {
    for (uint32_t token = 0; token < N_TOK; ++token) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            float total = 0.0f;
            for (uint32_t column = 0; column < IN_DIM; ++column) {
                total += f16_value(weights[(uint64_t)row * IN_DIM + column]) *
                         input[(uint64_t)token * IN_DIM + column];
            }
            out[(uint64_t)token * OUT_DIM + row] = total;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-3f) {
            fprintf(stderr, "projection mismatch at %u: actual=%g expected=%g\n",
                    i, actual[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

static int differs(const float *left, const float *right, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(left[i] - right[i]) > 1.0e-3f) return 1;
    }
    return 0;
}

static int run_single(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *x,
        const uint16_t *model,
        uint64_t n_tok,
        float *got,
        const float *want) {
    const uint32_t count = (uint32_t)(n_tok * OUT_DIM);
    return ds4_gpu_matmul_f16_tensor(
                   out, model, 2 * WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x, n_tok) &&
           ds4_gpu_tensor_read(out, 0, got, count * sizeof(float)) &&
           close_array(got, want, count);
}

static int run_pair(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        ds4_gpu_tensor *x,
        const uint16_t *model,
        uint64_t n_tok,
        float *got0,
        float *got1,
        const float *want0,
        const float *want1) {
    const uint32_t count = (uint32_t)(n_tok * OUT_DIM);
    return ds4_gpu_matmul_f16_pair_tensor(
                   out0, out1, model, 2 * WEIGHT_BYTES, 0, WEIGHT_BYTES,
                   IN_DIM, OUT_DIM, x, n_tok) &&
           ds4_gpu_tensor_read(out0, 0, got0, count * sizeof(float)) &&
           ds4_gpu_tensor_read(out1, 0, got1, count * sizeof(float)) &&
           close_array(got0, want0, count) &&
           close_array(got1, want1, count);
}

int main(void) {
    static const uint16_t weight_bits[] = {
        0xc000, 0xbc00, 0xb800, 0x3400, 0x3800, 0x3c00, 0x4000, 0x4400,
    };
    static const float first_token_bits[] = {
        -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
    };
    uint16_t model[2 * OUT_DIM * IN_DIM];
    float x_full[N_TOK * IN_DIM];
    float x_blas[N_TOK * IN_DIM];
    float want0_blas[OUTPUT_COUNT] = {0};
    float want1_blas[OUTPUT_COUNT] = {0};
    float want0_serial[OUTPUT_COUNT] = {0};
    float got0[OUTPUT_COUNT] = {0};
    float got1[OUTPUT_COUNT] = {0};

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) {
        model[i] = weight_bits[(i + 3u) % (sizeof(weight_bits) / sizeof(weight_bits[0]))];
        model[OUT_DIM * IN_DIM + i] =
            weight_bits[(i + 6u) % (sizeof(weight_bits) / sizeof(weight_bits[0]))];
    }
    for (uint32_t i = 0; i < IN_DIM; ++i) {
        x_full[i] = first_token_bits[(i + 1u) % (sizeof(first_token_bits) / sizeof(first_token_bits[0]))];
        x_blas[i] = x_full[i];
        x_full[IN_DIM + i] = 1.0003f;
        x_blas[IN_DIM + i] = 1.0f;
    }
    reference_projection(want0_blas, model, x_blas);
    reference_projection(want1_blas, model + OUT_DIM * IN_DIM, x_blas);
    reference_projection(want0_serial, model, x_full);
    if (!differs(want0_blas + OUT_DIM, want0_serial + OUT_DIM, OUT_DIM)) return 1;

    unsetenv("DS4_CUDA_SERIAL_F16_MATMUL");
    unsetenv("DS4_CUDA_SERIAL_ROUTER");
    unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    unsetenv("DS4_CUDA_NO_F16_PAIR_MATMUL");
    if (!ds4_gpu_init()) return 2;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_full));
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(sizeof(got0));
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(sizeof(got1));
    if (!x || !out0 || !out1) return 3;
    if (!ds4_gpu_tensor_write(x, 0, x_full, sizeof(x_full)) ||
        !ds4_gpu_set_model_map(model, sizeof(model))) return 4;

    if (!run_single(out0, x, model, 1, got0, want0_blas) ||
        !run_pair(out0, out1, x, model, 1, got0, got1, want0_blas, want1_blas)) return 5;

    for (uint32_t i = 0; i < 2 * OUT_DIM * IN_DIM; ++i) model[i] = 0;
    if (!run_single(out0, x, model, N_TOK, got0, want0_blas)) return 6;
    if (!run_pair(out0, out1, x, model, N_TOK, got0, got1, want0_blas, want1_blas)) return 7;

    if (setenv("DS4_CUDA_SERIAL_F16_MATMUL", "1", 1) != 0 ||
        !run_single(out0, x, model, N_TOK, got0, want0_serial)) return 8;
    if (unsetenv("DS4_CUDA_SERIAL_F16_MATMUL") != 0 ||
        ds4_gpu_matmul_f16_tensor(out0, model, sizeof(model), 0, IN_DIM, OUT_DIM, x, 0) ||
        ds4_gpu_matmul_f16_tensor(out0, NULL, sizeof(model), 0, IN_DIM, OUT_DIM, x, N_TOK) ||
        ds4_gpu_matmul_f16_pair_tensor(
            out0, out1, model, sizeof(model), 0, sizeof(model),
            IN_DIM, OUT_DIM, x, N_TOK)) {
        return 9;
    }
    if (!ds4_gpu_synchronize()) return 10;

    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_token_predecessor_matches\":true,\"single_token_pair_predecessor_matches\":true,\"multi_token_f16_blas_output_matches\":true,\"paired_multi_token_f16_blas_delegation_matches\":true,\"cached_f16_weights_survive_blas_after_host_mutation\":true,\"f32_to_f16_activation_rounding_observed\":true,\"serial_multi_token_f32_activation_fallback_matches\":true,\"zero_token_rejected\":true,\"invalid_second_model_range_rejected\":true,\"null_model_rejected\":true,\"cuda_oxide_blas_adapter_and_conversion_kernel_loaded\":true}");
    return 0;
}
