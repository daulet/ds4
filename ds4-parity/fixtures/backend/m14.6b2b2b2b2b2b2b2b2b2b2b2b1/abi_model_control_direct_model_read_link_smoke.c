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
    if (!ds4_gpu_rms_norm_weight_tensor(out, x, model_map, model_size, offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got))) {
        return 0;
    }
    return close_array(got, expected, 7);
}

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t model_size = page_size * 2;
    const uint64_t offset = sizeof(float);
    const uint64_t bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float original[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float changed[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    float original_want[7] = {0};
    float changed_want[7] = {0};
    void *allocation = NULL;
    if (posix_memalign(&allocation, page_size, model_size) != 0) return 1;
    unsigned char *model_map = (unsigned char *)allocation;
    memset(model_map, 0, model_size);
    memcpy(model_map + offset, original, bytes);
    reference_weight(original_want, x_in, original, 7, 1.0e-5f);
    reference_weight(changed_want, x_in, changed, 7, 1.0e-5f);

    if (setenv("DS4_CUDA_DIRECT_MODEL", "1", 1) != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_NO_MODEL_COPY") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0) return 2;

    if (!ds4_gpu_init()) return 3;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 4;

    if (!ds4_gpu_set_model_fd(-1) ||
        !ds4_gpu_set_model_map(model_map, model_size) ||
        host_register_calls != 1 ||
        ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, "direct-model") != 0 ||
        !consume_weight(out, x, model_map, model_size, offset, original_want) ||
        host_register_calls != 1) return 5;

    memcpy(model_map + offset, changed, bytes);
    if (ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, "direct-model") != 0 ||
        !consume_weight(out, x, model_map, model_size, offset, changed_want) ||
        host_register_calls != 1 ||
        !ds4_gpu_synchronize()) return 6;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"nonempty_direct_model_selected\":true,\"whole_map_registration_attempt_preserved\":true,\"direct_model_skips_range_registration_and_cache\":true,\"host_mutation_visible_to_direct_weighted_read\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
