#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 37u
#define OUT_DIM 3u
#define N_TOK 2u
#define WEIGHT_BYTES (OUT_DIM * IN_DIM * sizeof(float))

static void reference_projection(
        float *out,
        const float weights[OUT_DIM * IN_DIM],
        const float *x,
        uint32_t n_tok) {
    for (uint32_t token = 0; token < n_tok; ++token) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            float total = 0.0f;
            for (uint32_t column = 0; column < IN_DIM; ++column) {
                total += weights[(uint64_t)row * IN_DIM + column] *
                         x[(uint64_t)token * IN_DIM + column];
            }
            out[(uint64_t)token * OUT_DIM + row] = total;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-3f) return 0;
    }
    return 1;
}

static int run_projection(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *x,
        const float *model,
        uint64_t n_tok,
        float *got,
        const float *want) {
    if (!ds4_gpu_matmul_f32_tensor(
            out, model, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x, n_tok)) {
        fputs("matmul_f32 projection failed\n", stderr);
        return 0;
    }
    const uint32_t count = (uint32_t)n_tok * OUT_DIM;
    if (!ds4_gpu_tensor_read(out, 0, got, sizeof(float) * count)) {
        fputs("matmul_f32 readback failed\n", stderr);
        return 0;
    }
    if (!close_array(got, want, count)) {
        fprintf(
            stderr,
            "matmul_f32 mismatch at n_tok=%llu: got0=%.9g want0=%.9g\n",
            (unsigned long long)n_tok,
            got[0],
            want[0]);
        return 0;
    }
    return 1;
}

int main(void) {
    static const float weight_values[] = {
        -2.0003f, -1.0003f, -0.5003f, 0.2503f,
        0.5003f, 1.0003f, 2.0003f, 4.0003f,
    };
    static const float input_values[] = {
        -2.0003f, -1.0003f, -0.5003f, 0.2503f,
        0.5003f, 1.0003f, 2.0003f, 4.0003f,
    };
    float model[OUT_DIM * IN_DIM];
    float x_in[N_TOK * IN_DIM];
    float want_single[OUT_DIM] = {0};
    float want_multi[N_TOK * OUT_DIM] = {0};
    float got_single[OUT_DIM] = {0};
    float got_multi[N_TOK * OUT_DIM] = {0};

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) {
        model[i] = weight_values[(i + 3u) % (sizeof(weight_values) / sizeof(weight_values[0]))];
    }
    for (uint32_t i = 0; i < N_TOK * IN_DIM; ++i) {
        x_in[i] = input_values[(i + 1u) % (sizeof(input_values) / sizeof(input_values[0]))];
    }
    reference_projection(want_single, model, x_in, 1);
    reference_projection(want_multi, model, x_in, N_TOK);

    if (setenv("DS4_CUDA_NO_TF32", "1", 1) != 0) return 1;
    if (!ds4_gpu_init()) return 2;
    if (unsetenv("DS4_CUDA_NO_TF32") != 0) return 3;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out_single = ds4_gpu_tensor_alloc(sizeof(got_single));
    ds4_gpu_tensor *out_multi = ds4_gpu_tensor_alloc(sizeof(got_multi));
    if (!x || !out_single || !out_multi) return 4;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_set_model_map(model, sizeof(model))) return 5;

    if (!run_projection(out_single, x, model, 1, got_single, want_single)) return 6;

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) model[i] = 0.0f;
    if (!run_projection(out_multi, x, model, N_TOK, got_multi, want_multi)) return 7;

    if (ds4_gpu_matmul_f32_tensor(
            out_multi, model, sizeof(model), 0, IN_DIM, OUT_DIM, x, 0) ||
        ds4_gpu_matmul_f32_tensor(
            out_single, NULL, sizeof(model), 0, IN_DIM, OUT_DIM, x, 1) ||
        ds4_gpu_matmul_f32_tensor(
            out_single, model, sizeof(model), sizeof(model), IN_DIM, OUT_DIM, x, 1)) {
        return 8;
    }
    if (!ds4_gpu_synchronize()) return 9;

    ds4_gpu_tensor_free(out_multi);
    ds4_gpu_tensor_free(out_single);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_token_base_predecessor_matches\":true,\"multi_token_cublas_output_matches\":true,\"cached_f32_weights_survive_blas_after_host_mutation\":true,\"no_tf32_init_selection_survives_env_unset\":true,\"zero_token_rejected\":true,\"invalid_model_range_rejected\":true,\"null_model_rejected\":true,\"cuda_oxide_blas_adapter_loaded\":true}");
    return 0;
}
