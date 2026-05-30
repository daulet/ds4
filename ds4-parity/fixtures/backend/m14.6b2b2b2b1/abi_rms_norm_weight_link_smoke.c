#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void reference_weight(
        float *out,
        const float *x,
        const float *weight,
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
            out[(uint64_t)row * n + i] =
                x[(uint64_t)row * n + i] * scale * weight[i];
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
    const float model_map[18] = {
        99.0f, -99.0f,
        0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f,
        7.0f, -7.0f,
        -1.0f, 0.5f, 0.25f, 2.0f, -0.5f, 1.5f, 1.0f,
    };
    const uint64_t first_offset = 2 * sizeof(float);
    const uint64_t second_offset = 11 * sizeof(float);
    float want_first[14] = {0};
    float want_second[14] = {0};
    float got[14] = {0};
    reference_weight(want_first, x_in, &model_map[2], 7, 2, 1.0e-5f);
    reference_weight(want_second, x_in, &model_map[11], 7, 2, 1.0e-5f);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc(13 * sizeof(float));
    if (!x || !out || !short_out) return 2;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 3;
    if (!ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, sizeof(model_map), first_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, 7 * sizeof(float)) ||
        !close_array(got, want_first, 7)) return 4;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(
            out, x, model_map, sizeof(model_map), first_offset, 7, 2, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want_first, 14)) return 5;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_rms_norm_weight_rows_tensor(
            x, x, model_map, sizeof(model_map), first_offset, 7, 2, 1.0e-5f) ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, want_first, 14)) return 6;
    if (!ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_rms_norm_weight_rows_tensor(
            out, x, model_map, sizeof(model_map), second_offset, 7, 2, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want_second, 14)) return 7;
    if (ds4_gpu_rms_norm_weight_rows_tensor(
            short_out, x, model_map, sizeof(model_map), first_offset, 7, 2, 1.0e-5f) ||
        ds4_gpu_rms_norm_weight_rows_tensor(
            out, x, model_map, sizeof(model_map), first_offset, 7, 0, 1.0e-5f) ||
        ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, sizeof(model_map), sizeof(model_map) - sizeof(float), 7, 1.0e-5f) ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, sizeof(model_map), sizeof(model_map), 0, 1.0e-5f) ||
        ds4_gpu_rms_norm_weight_tensor(
            out, x, NULL, sizeof(model_map), first_offset, 7, 1.0e-5f)) return 8;
    if (!ds4_gpu_synchronize()) return 9;

    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"embedded_libdevice_module_loaded\":true,\"weighted_single_row_output_matches\":true,\"weighted_rows_output_matches\":true,\"weighted_alias_output_matches\":true,\"alternate_weight_offset_matches\":true,\"undersized_output_rejected\":true,\"invalid_weight_range_rejected\":true,\"zero_rows_rejected\":true,\"zero_width_preserved\":true,\"null_model_rejected\":true}");
    return 0;
}
