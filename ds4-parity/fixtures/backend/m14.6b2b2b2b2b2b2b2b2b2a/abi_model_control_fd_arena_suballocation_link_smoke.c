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

static int write_at(int fd, const void *data, size_t bytes, off_t offset) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, (const char *)data + done, bytes - done, offset + (off_t)done);
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t first_offset = sizeof(float);
    const uint64_t second_offset = page_size + sizeof(float);
    const uint64_t range_bytes = 7ull * sizeof(float);
    const uint64_t model_size = page_size * 2ull;
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float first_file_weights[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float second_file_weights[7] = {-0.5f, 0.75f, 1.25f, 0.5f, -1.25f, 1.5f, 0.25f};
    const float host_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    float first_want[7] = {0};
    float second_want[7] = {0};
    float got[7] = {0};
    void *allocation = NULL;
    unsigned char *file_image = NULL;
    char file_path[] = "/tmp/ds4-fd-arena-XXXXXX";
    int fd = -1;

    if (posix_memalign(&allocation, page_size, model_size) != 0) return 1;
    file_image = (unsigned char *)calloc(1, model_size);
    if (!file_image) return 2;
    unsigned char *model_map = (unsigned char *)allocation;
    memset(model_map, 0, model_size);
    memcpy(model_map + first_offset, host_weights, sizeof(host_weights));
    memcpy(model_map + second_offset, host_weights, sizeof(host_weights));
    memcpy(file_image + first_offset, first_file_weights, sizeof(first_file_weights));
    memcpy(file_image + second_offset, second_file_weights, sizeof(second_file_weights));
    reference_weight(first_want, x_in, first_file_weights, 7, 1.0e-5f);
    reference_weight(second_want, x_in, second_file_weights, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 || !write_at(fd, file_image, model_size, 0)) return 3;
    if (setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) != 0 ||
        setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB", "16", 1) != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "", 1) != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0) return 4;

    if (!ds4_gpu_init()) return 5;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 6;
    if (!ds4_gpu_set_model_fd(fd) ||
        !ds4_gpu_set_model_map(model_map, model_size) ||
        !ds4_gpu_cache_model_range(model_map, model_size, first_offset, range_bytes, "arena-first") ||
        !ds4_gpu_cache_model_range(model_map, model_size, second_offset, range_bytes, "arena-second") ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, first_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, first_want, 7) ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, second_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, second_want, 7)) return 7;

    if (!write_at(fd, host_weights, sizeof(host_weights), (off_t)first_offset) ||
        !write_at(fd, host_weights, sizeof(host_weights), (off_t)second_offset) ||
        !ds4_gpu_cache_model_range(model_map, model_size, first_offset, range_bytes, "arena-first-repeat") ||
        !ds4_gpu_cache_model_range(model_map, model_size, second_offset, range_bytes, "arena-second-repeat") ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, first_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, first_want, 7) ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, second_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, second_want, 7) ||
        !ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    free(file_image);
    free(allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"page_aligned_host_map\":true,\"fd_before_map_binds_host_base\":true,\"buffered_only_environment\":true,\"bounded_arena_chunk_override\":true,\"two_disjoint_fd_cache_ranges\":true,\"fd_bytes_precede_mutated_host_map\":true,\"repeated_cache_reuses_retained_ranges\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
