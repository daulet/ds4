#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int write_full(int fd, const void *data, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, (const char *)data + done, bytes - done, (off_t)done);
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int create_image_fd(const unsigned char *image, size_t bytes) {
    char path[] = "/tmp/ds4-default-fd-selection-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || unlink(path) != 0 || !write_full(fd, image, bytes)) {
        if (fd >= 0) close(fd);
        return -1;
    }
    return fd;
}

static int consume_weight(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        const float *expected,
        const char *label) {
    float got[7] = {0};
    if (!ds4_gpu_cache_model_range(model_map, model_size, offset, 7 * sizeof(float), label) ||
        !ds4_gpu_rms_norm_weight_tensor(out, x, model_map, model_size, offset, 7, 1.0e-5f) ||
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
    const float host_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float default_fd_weights[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float preload_fd_weights[7] = {2.0f, 1.5f, 1.0f, 0.5f, -0.5f, -1.0f, -1.5f};
    const float disabled_fd_weights[7] = {4.0f, 3.0f, 2.0f, 1.0f, -1.0f, -2.0f, -3.0f};
    float default_want[7] = {0};
    float preload_want[7] = {0};
    float disabled_want[7] = {0};
    unsigned char default_file[8192] = {0};
    unsigned char preload_file[8192] = {0};
    unsigned char disabled_file[8192] = {0};
    void *default_allocation = NULL;
    void *preload_allocation = NULL;
    void *disabled_allocation = NULL;
    int default_fd = -1;
    int preload_fd = -1;
    int disabled_fd = -1;

    if (posix_memalign(&default_allocation, page_size, model_size) != 0 ||
        posix_memalign(&preload_allocation, page_size, model_size) != 0 ||
        posix_memalign(&disabled_allocation, page_size, model_size) != 0) return 1;
    unsigned char *default_map = (unsigned char *)default_allocation;
    unsigned char *preload_map = (unsigned char *)preload_allocation;
    unsigned char *disabled_map = (unsigned char *)disabled_allocation;
    memset(default_map, 0, model_size);
    memset(preload_map, 0, model_size);
    memset(disabled_map, 0, model_size);
    memcpy(default_map + offset, host_weights, bytes);
    memcpy(preload_map + offset, host_weights, bytes);
    memcpy(disabled_map + offset, host_weights, bytes);
    memcpy(default_file + offset, default_fd_weights, bytes);
    memcpy(preload_file + offset, preload_fd_weights, bytes);
    memcpy(disabled_file + offset, disabled_fd_weights, bytes);
    reference_weight(default_want, x_in, default_fd_weights, 7, 1.0e-5f);
    reference_weight(preload_want, x_in, preload_fd_weights, 7, 1.0e-5f);
    reference_weight(disabled_want, x_in, host_weights, 7, 1.0e-5f);

    default_fd = create_image_fd(default_file, sizeof(default_file));
    preload_fd = create_image_fd(preload_file, sizeof(preload_file));
    disabled_fd = create_image_fd(disabled_file, sizeof(disabled_file));
    if (default_fd < 0 || preload_fd < 0 || disabled_fd < 0) return 2;

    if (setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1) != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "", 1) != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0) return 3;

    if (!ds4_gpu_init()) return 4;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 5;

    if (!ds4_gpu_set_model_map(default_map, model_size) ||
        !ds4_gpu_set_model_fd(default_fd) ||
        !consume_weight(out, x, default_map, model_size, offset, default_want, "default-fd") ||
        host_register_calls != 1) return 6;

    if (setenv("DS4_CUDA_WEIGHT_PRELOAD", "1", 1) != 0 ||
        !ds4_gpu_set_model_map(preload_map, model_size) ||
        !ds4_gpu_set_model_fd(preload_fd) ||
        !consume_weight(out, x, preload_map, model_size, offset, preload_want, "preload-fd") ||
        host_register_calls != 2) return 7;

    if (unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0 ||
        setenv("DS4_CUDA_NO_FD_CACHE", "1", 1) != 0 ||
        !ds4_gpu_set_model_map(disabled_map, model_size) ||
        !ds4_gpu_set_model_fd(disabled_fd) ||
        !consume_weight(out, x, disabled_map, model_size, offset, disabled_want, "disabled-fd") ||
        host_register_calls != 4 ||
        !ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(disabled_fd);
    close(preload_fd);
    close(default_fd);
    free(disabled_allocation);
    free(preload_allocation);
    free(default_allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"buffered_only_environment\":true,\"default_fd_staging_without_weight_cache\":true,\"weight_preload_does_not_suppress_fd_staging\":true,\"no_fd_cache_disables_fd_staging\":true,\"whole_map_registration_attempts_preserved\":true,\"range_registration_only_after_fd_disable\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
