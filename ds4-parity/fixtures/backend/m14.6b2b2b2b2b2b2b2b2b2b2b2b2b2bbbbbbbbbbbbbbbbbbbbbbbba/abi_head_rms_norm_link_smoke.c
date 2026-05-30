#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void reference_head_rms_norm(
        float *out,
        const float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        float eps) {
    const uint32_t rows = n_tok * n_head;
    for (uint32_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < head_dim; ++i) {
            const float value = x[(uint64_t)row * head_dim + i];
            sum += value * value;
        }
        const float scale = 1.0f / sqrtf(sum / (float)head_dim + eps);
        for (uint32_t i = 0; i < head_dim; ++i) {
            out[(uint64_t)row * head_dim + i] =
                x[(uint64_t)row * head_dim + i] * scale;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    const float input[16] = {
        1.0f, -2.0f, 0.5f, 4.0f,
        -1.5f, 0.25f, 3.0f, -0.5f,
        1.25f, 2.5f, -3.5f, 0.75f,
        1.5f, -2.25f, 0.5f, 3.25f,
    };
    float expected[16] = {0};
    float got[16] = {0};
    reference_head_rms_norm(expected, input, 2, 2, 4, 1.0e-5f);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_alloc(15 * sizeof(float));
    if (!x || !short_x) return 2;
    if (!ds4_gpu_tensor_write(x, 0, input, sizeof(input)) ||
        !ds4_gpu_head_rms_norm_tensor(x, 2, 2, 4, 1.0e-5f) ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, expected, 16)) return 3;
    if (ds4_gpu_head_rms_norm_tensor(short_x, 2, 2, 4, 1.0e-5f) ||
        ds4_gpu_head_rms_norm_tensor(x, 0, 2, 4, 1.0e-5f) ||
        ds4_gpu_head_rms_norm_tensor(x, 2, 0, 4, 1.0e-5f) ||
        ds4_gpu_head_rms_norm_tensor(x, 2, 2, 0, 1.0e-5f) ||
        ds4_gpu_head_rms_norm_tensor(NULL, 2, 2, 4, 1.0e-5f)) return 4;
    if (!ds4_gpu_synchronize()) return 5;

    ds4_gpu_tensor_free(short_x);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"multi_row_in_place_output_matches\":true,\"short_tensor_rejected\":true,\"zero_dimension_rejected\":true,\"null_rejected\":true,\"embedded_head_rms_norm_kernel_loaded\":true}");
    return 0;
}
