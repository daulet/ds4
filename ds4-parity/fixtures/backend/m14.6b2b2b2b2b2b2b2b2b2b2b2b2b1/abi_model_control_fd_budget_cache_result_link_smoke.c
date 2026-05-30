#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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
    const uint64_t limit_bytes = 1024ull * 1024ull * 1024ull;
    const uint64_t admitted_bytes = limit_bytes - 16;
    const uint64_t fallback_offset = limit_bytes;
    const uint64_t weight_bytes = 7ull * sizeof(float);
    const uint64_t model_size = fallback_offset + page_size;
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float host_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float file_weights[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    float want[7] = {0};
    float got[7] = {0};
    char file_path[] = "/tmp/ds4-fd-budget-result-XXXXXX";
    unsigned char *model_map = MAP_FAILED;
    int fd = -1;

    model_map = mmap(NULL, model_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (model_map == MAP_FAILED ||
        mprotect(model_map + fallback_offset, page_size, PROT_READ | PROT_WRITE) != 0) {
        return 1;
    }
    memcpy(model_map + fallback_offset, host_weights, sizeof(host_weights));
    reference_weight(want, x_in, host_weights, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 ||
        ftruncate(fd, (off_t)model_size) != 0 ||
        !write_at(fd, file_weights, sizeof(file_weights), (off_t)fallback_offset)) {
        return 2;
    }
    if (setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) != 0 ||
        setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB", "64", 1) != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "", 1) != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0) return 3;

    if (!ds4_gpu_init()) return 4;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 5;
    if (!ds4_gpu_set_model_map(model_map, model_size) ||
        !ds4_gpu_set_model_fd(fd) ||
        !ds4_gpu_cache_model_range(
            model_map, model_size, 0, admitted_bytes, "budget-fill") ||
        ds4_gpu_cache_model_range(
            model_map, model_size, fallback_offset, weight_bytes, "budget-fallback") != 0 ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, fallback_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 7) ||
        host_register_calls != 1 ||
        !ds4_gpu_synchronize()) return 6;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    munmap(model_map, model_size);
    puts("{\"c_linked_rust_staticlib\":true,\"default_fd_selection_without_weight_cache\":true,\"one_gib_cache_limit_selected\":true,\"near_limit_fd_range_admitted\":true,\"budget_fallback_not_reported_as_cached\":true,\"budget_fallback_consumed_by_weighted_rms\":true,\"host_bytes_precede_file_bytes_after_budget_fallback\":true,\"whole_map_registration_attempt_preserved\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
