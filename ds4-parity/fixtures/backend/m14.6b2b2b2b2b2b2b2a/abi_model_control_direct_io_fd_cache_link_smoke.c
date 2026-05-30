#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t model_size = page_size * 2;
    const uint64_t offset = sizeof(float);
    const uint64_t bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float file_weights[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float host_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    float want[7] = {0};
    float got[7] = {0};
    void *allocation = NULL;
    unsigned char file_image[8192] = {0};
    char file_path[] = "/tmp/ds4-direct-io-fd-cache-XXXXXX";
    int fd = -1;
    if (posix_memalign(&allocation, page_size, model_size) != 0) return 1;
    unsigned char *model_map = (unsigned char *)allocation;
    memset(model_map, 0, model_size);
    memcpy(model_map + offset, host_weights, sizeof(host_weights));
    memcpy(file_image + offset, file_weights, sizeof(file_weights));
    reference_weight(want, x_in, file_weights, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 || !write_full(fd, file_image, sizeof(file_image))) return 2;
    if (setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1) != 0 ||
        unsetenv("DS4_CUDA_NO_DIRECT_IO") != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "", 1) != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0) return 3;

    if (!ds4_gpu_init()) return 4;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 5;
    if (!ds4_gpu_set_model_fd(fd) ||
        !ds4_gpu_set_model_map(model_map, model_size) ||
        !ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, "direct-fd") ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 7)) return 6;

    memcpy(file_image + offset, host_weights, sizeof(host_weights));
    if (!write_full(fd, file_image, sizeof(file_image)) ||
        !ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, "direct-fd-repeat") ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 7) ||
        !ds4_gpu_synchronize()) return 7;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    free(allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"page_aligned_host_map\":true,\"fd_before_map_binds_host_base\":true,\"direct_io_permitted_by_environment\":true,\"fd_bytes_precede_mutated_host_map\":true,\"repeated_cache_reuses_fd_device_copy\":true,\"weighted_output_matches\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
