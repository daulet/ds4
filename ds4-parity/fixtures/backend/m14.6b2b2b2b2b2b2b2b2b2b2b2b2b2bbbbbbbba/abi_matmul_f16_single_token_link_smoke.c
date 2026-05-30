#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 37u
#define OUT_DIM 3u

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

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

static int run_projection(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *x,
        const uint16_t *model,
        float got[OUT_DIM],
        const float want[OUT_DIM]) {
    return ds4_gpu_matmul_f16_tensor(
                   out, model, OUT_DIM * IN_DIM * sizeof(uint16_t), 0, IN_DIM, OUT_DIM, x, 1) &&
           ds4_gpu_tensor_read(out, 0, got, OUT_DIM * sizeof(float)) &&
           close_array(got, want, OUT_DIM);
}

int main(void) {
    static const uint16_t weight_bits[] = {
        0xc000, 0xbc00, 0xb800, 0x3400, 0x3800, 0x3c00, 0x4000, 0x4400,
    };
    static const float input_bits[] = {
        -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
    };
    uint16_t model[OUT_DIM * IN_DIM];
    float x_in[IN_DIM];
    float want[OUT_DIM] = {0};
    float got[OUT_DIM] = {0};

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) {
        model[i] = weight_bits[(i + 3u) % (sizeof(weight_bits) / sizeof(weight_bits[0]))];
    }
    for (uint32_t i = 0; i < IN_DIM; ++i) {
        x_in[i] = input_bits[(i + 1u) % (sizeof(input_bits) / sizeof(input_bits[0]))];
    }
    reference_projection(want, model, x_in);

    unsetenv("DS4_CUDA_SERIAL_F16_MATMUL");
    unsetenv("DS4_CUDA_SERIAL_ROUTER");
    unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *out_two = ds4_gpu_tensor_alloc(2 * sizeof(got));
    if (!x || !out || !out_two) return 2;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_set_model_map(model, sizeof(model))) return 3;

    if (!run_projection(out, x, model, got, want)) return 4;

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) model[i] = 0;
    if (setenv("DS4_CUDA_NO_ORDERED_F16_MATMUL", "1", 1) != 0 ||
        !run_projection(out, x, model, got, want)) return 5;

    if (unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != 0 ||
        setenv("DS4_CUDA_SERIAL_F16_MATMUL", "1", 1) != 0 ||
        !run_projection(out, x, model, got, want)) return 6;

    if (unsetenv("DS4_CUDA_SERIAL_F16_MATMUL") != 0 ||
        ds4_gpu_matmul_f16_tensor(out_two, model, sizeof(model), 0, IN_DIM, OUT_DIM, x, 2) ||
        ds4_gpu_matmul_f16_tensor(out, NULL, sizeof(model), 0, IN_DIM, OUT_DIM, x, 1) ||
        ds4_gpu_matmul_f16_tensor(out, model, sizeof(model), sizeof(model), IN_DIM, OUT_DIM, x, 1)) {
        return 7;
    }
    if (!ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(out_two);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_token_default_ordered_output_matches\":true,\"single_token_base_output_matches\":true,\"single_token_serial_output_matches\":true,\"cached_f16_weights_survive_host_mutation\":true,\"multi_token_blas_rejected_until_owned\":true,\"invalid_model_range_rejected\":true,\"null_model_rejected\":true,\"embedded_rust_kernel_module_loaded\":true}");
    return 0;
}
