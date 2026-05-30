#define _POSIX_C_SOURCE 200112L

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;

static uint32_t host_register_calls = 0;

CUresult cuMemHostRegister_v2(void *ptr, size_t bytes, unsigned int flags) {
    (void)ptr;
    (void)bytes;
    (void)flags;
    host_register_calls += 1;
    return 801;
}

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

static int consume_weight(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        const float *expected) {
    float got[7] = {0};
    if (!ds4_gpu_cache_model_range(model_map, model_size, offset, 7 * sizeof(float), "full-copy") ||
        !ds4_gpu_rms_norm_weight_tensor(out, x, model_map, model_size, offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got))) {
        return 0;
    }
    return close_array(got, expected, 7);
}

int main(void) {
    const uint64_t offset = sizeof(float);
    const uint64_t bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float first_original[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float first_changed[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float second_original[7] = {2.0f, 1.5f, 1.0f, 0.5f, -0.5f, -1.0f, -1.5f};
    const float second_changed[7] = {-2.0f, -1.5f, -1.0f, -0.5f, 0.5f, 1.0f, 1.5f};
    float first_map[9] = {0};
    float second_map[9] = {0};
    float first_want[7] = {0};
    float second_want[7] = {0};
    memcpy((unsigned char *)first_map + offset, first_original, bytes);
    memcpy((unsigned char *)second_map + offset, second_original, bytes);
    reference_weight(first_want, x_in, first_original, 7, 1.0e-5f);
    reference_weight(second_want, x_in, second_original, 7, 1.0e-5f);

    if (setenv("DS4_CUDA_COPY_MODEL", "1", 1) != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_NO_MODEL_COPY") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0) return 1;

    if (!ds4_gpu_init()) return 2;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 3;

    if (!ds4_gpu_set_model_fd(-1) ||
        !ds4_gpu_set_model_map(first_map, sizeof(first_map)) ||
        !ds4_gpu_set_model_map_range(first_map, sizeof(first_map), offset, bytes) ||
        host_register_calls != 0) return 4;
    memcpy((unsigned char *)first_map + offset, first_changed, bytes);
    if (!consume_weight(out, x, first_map, sizeof(first_map), offset, first_want)) return 5;

    if (!ds4_gpu_set_model_map(second_map, sizeof(second_map)) ||
        !ds4_gpu_set_model_map_range(second_map, sizeof(second_map), offset, bytes) ||
        host_register_calls != 0) return 6;
    memcpy((unsigned char *)second_map + offset, second_changed, bytes);
    if (!consume_weight(out, x, second_map, sizeof(second_map), offset, second_want) ||
        !ds4_gpu_synchronize()) return 7;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"nonempty_copy_model_selected\":true,\"successful_full_copy_skips_registration\":true,\"host_mutation_after_map_setup_ignored\":true,\"model_replacement_copies_new_image\":true,\"cached_weighted_rms_reads_copied_image\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
