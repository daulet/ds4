#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void reference_weighted_rows(
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
            out[(uint64_t)row * n + i] = x[(uint64_t)row * n + i] * scale * weight[i];
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
    const uint32_t q_n = 5;
    const uint32_t kv_n = 3;
    const uint32_t rows = 2;
    const float eps = 1.0e-5f;
    const float model[8] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, -1.0f, 0.75f, 2.0f};
    const float q_values[10] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f, -0.5f, 1.25f, 2.5f};
    const float kv_values[6] = {2.0f, -0.5f, 1.5f, -1.0f, 3.0f, 0.25f};
    float q_expected[10] = {0};
    float kv_expected[6] = {0};
    float q_got[10] = {0};
    float kv_got[6] = {0};
    reference_weighted_rows(q_expected, q_values, model, q_n, rows, eps);
    reference_weighted_rows(kv_expected, kv_values, model + q_n, kv_n, rows, eps);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(q_values));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(sizeof(kv_values));
    ds4_gpu_tensor *q_out = ds4_gpu_tensor_alloc(sizeof(q_values));
    ds4_gpu_tensor *kv_out = ds4_gpu_tensor_alloc(sizeof(kv_values));
    ds4_gpu_tensor *short_q_out = ds4_gpu_tensor_alloc(sizeof(q_values) - sizeof(float));
    if (!q || !kv || !q_out || !kv_out || !short_q_out) return 2;
    if (!ds4_gpu_tensor_write(q, 0, q_values, sizeof(q_values)) ||
        !ds4_gpu_tensor_write(kv, 0, kv_values, sizeof(kv_values))) return 3;

    if (!ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            q_out, q, model, sizeof(model), 0, q_n,
            kv_out, kv, q_n * sizeof(float), kv_n, rows, eps) ||
        !ds4_gpu_tensor_read(q_out, 0, q_got, sizeof(q_got)) ||
        !ds4_gpu_tensor_read(kv_out, 0, kv_got, sizeof(kv_got)) ||
        !close_array(q_got, q_expected, 10) ||
        !close_array(kv_got, kv_expected, 6)) return 4;

    if (setenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED", "1", 1) != 0) return 5;
    if (!ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            q_out, q, model, sizeof(model), 0, q_n,
            kv_out, kv, q_n * sizeof(float), kv_n, rows, eps) ||
        !ds4_gpu_tensor_read(q_out, 0, q_got, sizeof(q_got)) ||
        !ds4_gpu_tensor_read(kv_out, 0, kv_got, sizeof(kv_got)) ||
        !close_array(q_got, q_expected, 10) ||
        !close_array(kv_got, kv_expected, 6)) return 6;
    if (unsetenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") != 0) return 7;

    if (ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            short_q_out, q, model, sizeof(model), 0, q_n,
            kv_out, kv, q_n * sizeof(float), kv_n, rows, eps) ||
        ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            q_out, q, model, sizeof(model), 0, q_n,
            kv_out, kv, q_n * sizeof(float), kv_n, 0, eps) ||
        ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            q_out, NULL, model, sizeof(model), 0, q_n,
            kv_out, kv, q_n * sizeof(float), kv_n, rows, eps)) return 8;
    if (!ds4_gpu_synchronize()) return 9;

    ds4_gpu_tensor_free(short_q_out);
    ds4_gpu_tensor_free(kv_out);
    ds4_gpu_tensor_free(q_out);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(q);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"fused_output_matches\":true,\"disabled_fusion_fallback_matches\":true,\"asymmetric_q_kv_widths_match\":true,\"short_tensor_rejected\":true,\"zero_dimension_rejected\":true,\"null_rejected\":true,\"embedded_fused_qkv_rms_kernel_loaded\":true}");
    return 0;
}
