#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 37u
#define OUT_DIM 3u
#define WEIGHT_BYTES (OUT_DIM * IN_DIM * sizeof(uint16_t))

static float f16_value(uint16_t bits) {
    switch (bits) {
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
        float out[OUT_DIM],
        const uint16_t weights[OUT_DIM * IN_DIM],
        const float x[IN_DIM]) {
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        float total = 0.0f;
        for (uint32_t column = 0; column < IN_DIM; ++column) {
            total += f16_value(weights[(uint64_t)row * IN_DIM + column]) * x[column];
        }
        out[row] = total;
    }
}

static int close_array(const float *actual, const float *expected) {
    for (uint32_t i = 0; i < OUT_DIM; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

static int run_pair(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        ds4_gpu_tensor *x,
        const uint16_t *model,
        float got0[OUT_DIM],
        float got1[OUT_DIM],
        const float want0[OUT_DIM],
        const float want1[OUT_DIM]) {
    return ds4_gpu_matmul_f16_pair_tensor(
                   out0, out1, model, 2 * WEIGHT_BYTES, 0, WEIGHT_BYTES,
                   IN_DIM, OUT_DIM, x, 1) &&
           ds4_gpu_tensor_read(out0, 0, got0, OUT_DIM * sizeof(float)) &&
           ds4_gpu_tensor_read(out1, 0, got1, OUT_DIM * sizeof(float)) &&
           close_array(got0, want0) &&
           close_array(got1, want1);
}

int main(void) {
    static const uint16_t weight_bits[] = {
        0xc000, 0xbc00, 0xb800, 0x3400, 0x3800, 0x3c00, 0x4000, 0x4400,
    };
    static const float input_bits[] = {
        -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
    };
    uint16_t model[2 * OUT_DIM * IN_DIM];
    float x_in[IN_DIM];
    float want0[OUT_DIM] = {0};
    float want1[OUT_DIM] = {0};
    float got0[OUT_DIM] = {0};
    float got1[OUT_DIM] = {0};

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) {
        model[i] = weight_bits[(i + 3u) % (sizeof(weight_bits) / sizeof(weight_bits[0]))];
        model[OUT_DIM * IN_DIM + i] =
            weight_bits[(i + 6u) % (sizeof(weight_bits) / sizeof(weight_bits[0]))];
    }
    for (uint32_t i = 0; i < IN_DIM; ++i) {
        x_in[i] = input_bits[(i + 1u) % (sizeof(input_bits) / sizeof(input_bits[0]))];
    }
    reference_projection(want0, model, x_in);
    reference_projection(want1, model + OUT_DIM * IN_DIM, x_in);

    unsetenv("DS4_CUDA_NO_F16_PAIR_MATMUL");
    unsetenv("DS4_CUDA_SERIAL_F16_MATMUL");
    unsetenv("DS4_CUDA_SERIAL_ROUTER");
    unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(sizeof(got0));
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(sizeof(got1));
    ds4_gpu_tensor *out_two = ds4_gpu_tensor_alloc(2 * sizeof(got0));
    if (!x || !out0 || !out1 || !out_two) return 2;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_set_model_map(model, sizeof(model))) return 3;

    if (!run_pair(out0, out1, x, model, got0, got1, want0, want1)) return 4;

    for (uint32_t i = 0; i < 2 * OUT_DIM * IN_DIM; ++i) model[i] = 0;
    if (setenv("DS4_CUDA_NO_F16_PAIR_MATMUL", "1", 1) != 0 ||
        !run_pair(out0, out1, x, model, got0, got1, want0, want1)) return 5;

    if (unsetenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != 0 ||
        setenv("DS4_CUDA_NO_ORDERED_F16_MATMUL", "1", 1) != 0 ||
        !run_pair(out0, out1, x, model, got0, got1, want0, want1)) return 6;

    if (unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != 0 ||
        setenv("DS4_CUDA_SERIAL_F16_MATMUL", "1", 1) != 0 ||
        !run_pair(out0, out1, x, model, got0, got1, want0, want1)) return 7;

    if (unsetenv("DS4_CUDA_SERIAL_F16_MATMUL") != 0 ||
        ds4_gpu_matmul_f16_pair_tensor(
            out_two, out_two, model, sizeof(model), 0, WEIGHT_BYTES,
            IN_DIM, OUT_DIM, x, 2) ||
        ds4_gpu_matmul_f16_pair_tensor(
            out0, out1, NULL, sizeof(model), 0, WEIGHT_BYTES,
            IN_DIM, OUT_DIM, x, 1) ||
        ds4_gpu_matmul_f16_pair_tensor(
            out0, out1, model, sizeof(model), 0, sizeof(model),
            IN_DIM, OUT_DIM, x, 1)) {
        return 8;
    }
    if (!ds4_gpu_synchronize()) return 9;

    ds4_gpu_tensor_free(out_two);
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_token_default_paired_output_matches\":true,\"single_token_no_pair_independent_output_matches\":true,\"single_token_no_ordered_independent_output_matches\":true,\"single_token_serial_independent_output_matches\":true,\"cached_pair_weights_survive_host_mutation\":true,\"multi_token_pair_blas_rejected_until_owned\":true,\"invalid_second_model_range_rejected\":true,\"null_model_rejected\":true,\"embedded_rust_kernel_module_loaded\":true}");
    return 0;
}
