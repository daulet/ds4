#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void reference_weight(
        float *out,
        const float *x,
        const float *weight,
        uint32_t n,
        float eps) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    const float scale = 1.0f / sqrtf(sum / (float)n + eps);
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    float first_map[9] = {91.0f, 0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f, -91.0f};
    const float second_map[9] = {92.0f, -1.0f, 0.5f, 0.25f, 2.0f, -0.5f, 1.5f, 1.0f, -92.0f};
    const uint64_t offset = sizeof(float);
    const uint64_t bytes = 7 * sizeof(float);
    float want_first[7] = {0};
    float want_second[7] = {0};
    float got[7] = {0};
    reference_weight(want_first, x_in, &first_map[1], 7, 1.0e-5f);
    reference_weight(want_second, x_in, &second_map[1], 7, 1.0e-5f);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 2;

    if (!ds4_gpu_set_model_fd(-1) ||
        !ds4_gpu_set_model_map(first_map, sizeof(first_map)) ||
        !ds4_gpu_set_model_map_range(first_map, sizeof(first_map), offset, bytes) ||
        !ds4_gpu_cache_model_range(first_map, sizeof(first_map), offset, bytes, "first")) return 3;
    if (!ds4_gpu_rms_norm_weight_tensor(
            out, x, first_map, sizeof(first_map), offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want_first, 7)) return 4;

    if (!ds4_gpu_set_model_map(second_map, sizeof(second_map)) ||
        !ds4_gpu_cache_model_range(second_map, sizeof(second_map), offset, bytes, "second") ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, second_map, sizeof(second_map), offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want_second, 7)) return 5;

    for (uint32_t i = 0; i < 7; ++i) first_map[1 + i] = second_map[1 + i];
    if (!ds4_gpu_rms_norm_weight_tensor(
            out, x, first_map, sizeof(first_map), offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want_second, 7)) return 6;

    if (!ds4_gpu_cache_model_range(NULL, 0, 99, 0, NULL) ||
        ds4_gpu_cache_model_range(second_map, sizeof(second_map), sizeof(second_map), bytes, NULL) ||
        ds4_gpu_set_model_map(NULL, sizeof(second_map)) ||
        ds4_gpu_set_model_map(second_map, 0)) return 7;
    if (!ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"public_set_model_fd_accepted\":true,\"public_set_model_map_accepted\":true,\"public_set_model_map_range_accepted\":true,\"public_cache_model_range_consumed_by_weighted_rms\":true,\"mapping_switch_output_matches\":true,\"mapping_switch_releases_prior_cached_range\":true,\"zero_byte_cache_noop_preserved\":true,\"invalid_nonzero_cache_rejected\":true,\"invalid_map_rejected\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
