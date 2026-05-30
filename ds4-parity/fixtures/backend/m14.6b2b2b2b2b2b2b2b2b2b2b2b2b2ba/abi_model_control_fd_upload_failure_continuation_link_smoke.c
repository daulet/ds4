#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if !defined(__linux__) && !defined(RTLD_NEXT)
#define RTLD_NEXT ((void *)-1)
#endif

typedef int CUresult;
typedef uint64_t CUdeviceptr;
typedef void *CUstream;
typedef CUresult (*cu_memcpy_htod_async_fn)(CUdeviceptr, const void *, size_t, CUstream);

enum {
    CU_ERROR_INVALID_VALUE = 1,
    CU_ERROR_NOT_SUPPORTED = 801,
};

static uint32_t host_register_calls = 0;
static uint32_t htod_calls = 0;
static uint32_t injected_fd_copy_failures = 0;
static int fail_next_htod = 0;

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream) {
    static cu_memcpy_htod_async_fn real_copy = NULL;
    htod_calls += 1;
    if (fail_next_htod) {
        fail_next_htod = 0;
        injected_fd_copy_failures += 1;
        return CU_ERROR_INVALID_VALUE;
    }
    if (!real_copy) {
        real_copy = (cu_memcpy_htod_async_fn)dlsym(RTLD_NEXT, "cuMemcpyHtoDAsync_v2");
    }
    return real_copy ? real_copy(dst, src, bytes, stream) : CU_ERROR_NOT_SUPPORTED;
}

CUresult cuMemHostRegister_v2(void *ptr, size_t bytes, unsigned int flags) {
    (void)ptr;
    (void)bytes;
    (void)flags;
    host_register_calls += 1;
    return CU_ERROR_NOT_SUPPORTED;
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
        ssize_t n = pwrite(fd, (const unsigned char *)data + done, bytes - done, offset + (off_t)done);
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t model_size = page_size * 2;
    const uint64_t weight_offset = sizeof(float);
    const uint64_t weight_bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float host_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float file_weights[7] = {3.0f, 2.5f, 2.0f, 1.5f, 1.0f, 0.5f, 0.25f};
    const float changed[7] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    float want[7] = {0};
    float got[7] = {0};
    void *allocation = NULL;
    unsigned char *model_map = NULL;
    char file_path[] = "/tmp/ds4-fd-upload-failure-XXXXXX";
    uint32_t register_calls_before_range = 0;
    uint32_t htod_calls_before_range = 0;
    int fd = -1;

    if (posix_memalign(&allocation, page_size, model_size) != 0) return 1;
    model_map = (unsigned char *)allocation;
    memset(model_map, 0, model_size);
    memcpy(model_map + weight_offset, host_weights, weight_bytes);
    reference_weight(want, x_in, host_weights, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 ||
        ftruncate(fd, (off_t)model_size) != 0 ||
        !write_at(fd, file_weights, weight_bytes, (off_t)weight_offset)) return 2;

    if (setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) != 0 ||
        unsetenv("DS4_CUDA_NO_DIRECT_IO") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0 ||
        unsetenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0) return 3;

    if (!ds4_gpu_init()) return 4;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in)) ||
        !ds4_gpu_set_model_map(model_map, model_size) ||
        !ds4_gpu_set_model_fd(fd)) return 5;

    register_calls_before_range = host_register_calls;
    htod_calls_before_range = htod_calls;
    fail_next_htod = 1;
    if (!ds4_gpu_cache_model_range(
            model_map, model_size, weight_offset, weight_bytes, "fd-upload-copy-failure") ||
        fail_next_htod != 0 ||
        injected_fd_copy_failures != 1 ||
        host_register_calls != register_calls_before_range + 1 ||
        htod_calls != htod_calls_before_range + 2 ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, weight_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 7)) return 6;

    memcpy(model_map + weight_offset, changed, weight_bytes);
    if (!ds4_gpu_rms_norm_weight_tensor(
            out, x, model_map, model_size, weight_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 7) ||
        !ds4_gpu_synchronize()) return 7;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    free(allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"direct_fd_branch_selected\":true,\"interposed_fd_async_copy_failure\":true,\"single_selected_fd_attempt_after_failure\":true,\"fd_failure_continues_to_device_copy\":true,\"registration_fallback_boundary_preserved\":true,\"cached_fallback_retains_original_host_bytes\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
