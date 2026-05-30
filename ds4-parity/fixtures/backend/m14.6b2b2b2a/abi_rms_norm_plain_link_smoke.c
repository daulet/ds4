#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void reference_plain(
        float *out,
        const float *x,
        uint32_t n,
        uint32_t rows,
        float eps) {
    for (uint32_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            const float value = x[(uint64_t)row * n + i];
            sum += value * value;
        }
        const float scale = 1.0f / sqrtf(sum / (float)n + eps);
        for (uint32_t i = 0; i < n; ++i) {
            out[(uint64_t)row * n + i] = x[(uint64_t)row * n + i] * scale;
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
    const float x_in[14] = {
        1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f,
        -0.5f, 1.25f, 2.5f, -3.5f, 0.75f, 1.5f, -2.25f,
    };
    float want[14] = {0};
    float got[14] = {0};
    reference_plain(want, x_in, 7, 2, 1.0e-5f);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc(13 * sizeof(float));
    if (!x || !out || !short_out) return 2;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 3;
    if (!ds4_gpu_rms_norm_plain_tensor(out, x, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, 7 * sizeof(float)) ||
        !close_array(got, want, 7)) return 4;
    if (!ds4_gpu_rms_norm_plain_rows_tensor(out, x, 7, 2, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 14)) return 5;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_rms_norm_plain_rows_tensor(x, x, 7, 2, 1.0e-5f) ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, want, 14)) return 6;
    if (ds4_gpu_rms_norm_plain_rows_tensor(short_out, x, 7, 2, 1.0e-5f) ||
        ds4_gpu_rms_norm_plain_rows_tensor(out, x, 7, 0, 1.0e-5f) ||
        !ds4_gpu_rms_norm_plain_tensor(out, x, 0, 1.0e-5f) ||
        ds4_gpu_rms_norm_plain_tensor(NULL, x, 7, 1.0e-5f)) return 7;
    if (!ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"embedded_libdevice_module_loaded\":true,\"plain_single_row_output_matches\":true,\"plain_rows_output_matches\":true,\"plain_alias_output_matches\":true,\"undersized_output_rejected\":true,\"zero_rows_rejected\":true,\"zero_width_preserved\":true,\"null_rejected\":true}");
    return 0;
}
