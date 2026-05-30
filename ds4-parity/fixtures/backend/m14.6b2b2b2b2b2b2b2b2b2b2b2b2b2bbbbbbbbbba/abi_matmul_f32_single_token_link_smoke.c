#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define IN_DIM 37u
#define OUT_DIM 3u
#define WEIGHT_BYTES (OUT_DIM * IN_DIM * sizeof(float))

static void reference_projection(
        float out[OUT_DIM],
        const float weights[OUT_DIM * IN_DIM],
        const float x[IN_DIM]) {
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        float total = 0.0f;
        for (uint32_t column = 0; column < IN_DIM; ++column) {
            total += weights[(uint64_t)row * IN_DIM + column] * x[column];
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

static int run_projection(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *x,
        const float *model,
        float got[OUT_DIM],
        const float want[OUT_DIM]) {
    if (!ds4_gpu_matmul_f32_tensor(
            out, model, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x, 1)) {
        fputs("matmul_f32 launch failed\n", stderr);
        return 0;
    }
    if (!ds4_gpu_tensor_read(out, 0, got, sizeof(float) * OUT_DIM)) {
        fputs("matmul_f32 readback failed\n", stderr);
        return 0;
    }
    if (!close_array(got, want)) {
        fprintf(
            stderr,
            "matmul_f32 mismatch: got=[%.9g,%.9g,%.9g] want=[%.9g,%.9g,%.9g]\n",
            got[0], got[1], got[2], want[0], want[1], want[2]);
        return 0;
    }
    return 1;
}

int main(void) {
    static const float weight_values[] = {
        -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
    };
    static const float input_values[] = {
        -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
    };
    float model[OUT_DIM * IN_DIM];
    float x_in[IN_DIM];
    float want[OUT_DIM] = {0};
    float got[OUT_DIM] = {0};

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) {
        model[i] = weight_values[(i + 3u) % (sizeof(weight_values) / sizeof(weight_values[0]))];
    }
    for (uint32_t i = 0; i < IN_DIM; ++i) {
        x_in[i] = input_values[(i + 1u) % (sizeof(input_values) / sizeof(input_values[0]))];
    }
    reference_projection(want, model, x_in);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *out_two = ds4_gpu_tensor_alloc(2 * sizeof(got));
    if (!x || !out || !out_two) return 2;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_set_model_map(model, sizeof(model))) return 3;

    if (!run_projection(out, x, model, got, want)) return 4;

    for (uint32_t i = 0; i < OUT_DIM * IN_DIM; ++i) model[i] = 0.0f;
    if (!run_projection(out, x, model, got, want)) return 5;

    if (ds4_gpu_matmul_f32_tensor(
            out_two, model, sizeof(model), 0, IN_DIM, OUT_DIM, x, 2) ||
        ds4_gpu_matmul_f32_tensor(
            out, NULL, sizeof(model), 0, IN_DIM, OUT_DIM, x, 1) ||
        ds4_gpu_matmul_f32_tensor(
            out, model, sizeof(model), sizeof(model), IN_DIM, OUT_DIM, x, 1)) {
        return 6;
    }
    if (!ds4_gpu_synchronize()) return 7;

    ds4_gpu_tensor_free(out_two);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_token_base_output_matches\":true,\"cached_f32_weights_survive_host_mutation\":true,\"multi_token_blas_rejected_until_owned\":true,\"invalid_model_range_rejected\":true,\"null_model_rejected\":true,\"embedded_rust_kernel_module_loaded\":true}");
    return 0;
}
