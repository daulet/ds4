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

static int consume_cached_weight(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        const float *expected,
        uint32_t n) {
    float got[7] = {0};
    if (!ds4_gpu_rms_norm_weight_tensor(
                out, x, model_map, model_size, offset, n, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got))) {
        return 0;
    }
    return close_array(got, expected, n);
}

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t model_size = page_size * 3;
    const uint64_t first_offset = sizeof(float);
    const uint64_t second_offset = page_size + sizeof(float);
    const uint64_t bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float first_weights[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float second_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float reset_weights[7] = {2.0f, 1.5f, 1.0f, 0.5f, -0.5f, -1.0f, -1.5f};
    float first_want[7] = {0};
    float second_want[7] = {0};
    float reset_want[7] = {0};
    void *first_allocation = NULL;
    void *second_allocation = NULL;
    if (posix_memalign(&first_allocation, page_size, model_size) != 0 ||
        posix_memalign(&second_allocation, page_size, model_size) != 0) return 1;
    unsigned char *first_map = (unsigned char *)first_allocation;
    unsigned char *second_map = (unsigned char *)second_allocation;
    memset(first_map, 0, model_size);
    memset(second_map, 0, model_size);
    memcpy(first_map + first_offset, first_weights, sizeof(first_weights));
    memcpy(first_map + second_offset, second_weights, sizeof(second_weights));
    memcpy(second_map + first_offset, reset_weights, sizeof(reset_weights));
    reference_weight(first_want, x_in, first_weights, 7, 1.0e-5f);
    reference_weight(second_want, x_in, second_weights, 7, 1.0e-5f);
    reference_weight(reset_want, x_in, reset_weights, 7, 1.0e-5f);

    if (unsetenv("DS4_CUDA_COPY_MODEL") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_NO_MODEL_COPY") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0) return 2;

    if (!ds4_gpu_init()) return 3;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 4;

    if (!ds4_gpu_set_model_fd(-1) ||
        !ds4_gpu_set_model_map(first_map, model_size) ||
        !ds4_gpu_cache_model_range(first_map, model_size, first_offset, bytes, "disable-first") ||
        host_register_calls != 2 ||
        !consume_cached_weight(out, x, first_map, model_size, first_offset, first_want, 7) ||
        !ds4_gpu_cache_model_range(first_map, model_size, second_offset, bytes, "disable-second") ||
        host_register_calls != 2 ||
        !consume_cached_weight(out, x, first_map, model_size, second_offset, second_want, 7)) return 5;

    if (!ds4_gpu_set_model_map(second_map, model_size) ||
        !ds4_gpu_cache_model_range(second_map, model_size, first_offset, bytes, "disable-reset") ||
        host_register_calls != 4 ||
        !consume_cached_weight(out, x, second_map, model_size, first_offset, reset_want, 7) ||
        !ds4_gpu_synchronize()) return 6;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(second_allocation);
    free(first_allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"page_aligned_model_maps\":true,\"interposed_not_supported_registration\":true,\"whole_map_failure_does_not_disable_range_attempt\":true,\"first_range_failure_disables_second_range_attempt\":true,\"model_replacement_resets_range_attempt_state\":true,\"device_copy_fallback_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
