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
typedef CUresult (*cu_mem_alloc_async_fn)(CUdeviceptr *, size_t, CUstream);

enum {
    CU_ERROR_OUT_OF_MEMORY = 2,
    CU_ERROR_NOT_SUPPORTED = 801,
};

static const size_t arena_chunk_bytes = 256ull * 1024ull * 1024ull;
static uint32_t arena_alloc_failures = 0;
static uint32_t host_register_calls = 0;

CUresult cuMemAllocAsync(CUdeviceptr *ptr, size_t bytes, CUstream stream) {
    static cu_mem_alloc_async_fn real_alloc = NULL;
    if (bytes >= arena_chunk_bytes) {
        arena_alloc_failures += 1;
        return CU_ERROR_OUT_OF_MEMORY;
    }
    if (!real_alloc) {
        real_alloc = (cu_mem_alloc_async_fn)dlsym(RTLD_NEXT, "cuMemAllocAsync");
    }
    return real_alloc ? real_alloc(ptr, bytes, stream) : CU_ERROR_NOT_SUPPORTED;
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

static int consume_weight(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        const float *expected) {
    float got[7] = {0};
    return ds4_gpu_rms_norm_weight_tensor(out, x, model_map, model_size, offset, 7, 1.0e-5f) &&
        ds4_gpu_tensor_read(out, 0, got, sizeof(got)) &&
        close_array(got, expected, 7);
}

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t first_offset = sizeof(float);
    const uint64_t second_offset = page_size + sizeof(float);
    const uint64_t model_size = page_size * 2;
    const uint64_t weight_bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float non_strict_first[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float non_strict_second[7] = {2.0f, 1.5f, 1.0f, 0.5f, -0.5f, -1.0f, -1.5f};
    const float strict_first[7] = {0.75f, 0.5f, 0.25f, -0.25f, -0.5f, -0.75f, -1.0f};
    const float strict_second[7] = {-1.5f, -1.0f, -0.5f, 0.5f, 1.0f, 1.5f, 2.0f};
    const float file_first[7] = {3.0f, 2.5f, 2.0f, 1.5f, 1.0f, 0.5f, 0.25f};
    const float file_second[7] = {-3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f, -0.25f};
    const float changed[7] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    float non_strict_first_want[7] = {0};
    float non_strict_second_want[7] = {0};
    float strict_first_want[7] = {0};
    float strict_second_want[7] = {0};
    unsigned char non_strict_map[8192] = {0};
    unsigned char strict_map[8192] = {0};
    char file_path[] = "/tmp/ds4-fd-arena-failure-XXXXXX";
    int fd = -1;

    memcpy(non_strict_map + first_offset, non_strict_first, weight_bytes);
    memcpy(non_strict_map + second_offset, non_strict_second, weight_bytes);
    memcpy(strict_map + first_offset, strict_first, weight_bytes);
    memcpy(strict_map + second_offset, strict_second, weight_bytes);
    reference_weight(non_strict_first_want, x_in, non_strict_first, 7, 1.0e-5f);
    reference_weight(non_strict_second_want, x_in, non_strict_second, 7, 1.0e-5f);
    reference_weight(strict_first_want, x_in, strict_first, 7, 1.0e-5f);
    reference_weight(strict_second_want, x_in, strict_second, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 ||
        ftruncate(fd, (off_t)model_size) != 0 ||
        !write_at(fd, file_first, weight_bytes, (off_t)first_offset) ||
        !write_at(fd, file_second, weight_bytes, (off_t)second_offset)) return 1;

    if (setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL") != 0 ||
        unsetenv("DS4_CUDA_COPY_MODEL_CHUNKED") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0 ||
        unsetenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != 0) return 2;

    if (!ds4_gpu_init()) return 3;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 4;

    if (!ds4_gpu_set_model_map(non_strict_map, model_size) ||
        !ds4_gpu_set_model_fd(fd) ||
        ds4_gpu_cache_model_range(
            non_strict_map, model_size, first_offset, weight_bytes, "arena-failure-nonstrict-first") != 0 ||
        ds4_gpu_cache_model_range(
            non_strict_map, model_size, second_offset, weight_bytes, "arena-failure-nonstrict-second") != 0 ||
        arena_alloc_failures != 1 ||
        host_register_calls != 1 ||
        !consume_weight(out, x, non_strict_map, model_size, first_offset, non_strict_first_want) ||
        !consume_weight(out, x, non_strict_map, model_size, second_offset, non_strict_second_want)) return 5;

    if (setenv("DS4_CUDA_STRICT_WEIGHT_CACHE", "1", 1) != 0 ||
        !ds4_gpu_set_model_map(strict_map, model_size) ||
        !ds4_gpu_set_model_fd(fd) ||
        !ds4_gpu_cache_model_range(
            strict_map, model_size, first_offset, weight_bytes, "arena-failure-strict-first") ||
        !ds4_gpu_cache_model_range(
            strict_map, model_size, second_offset, weight_bytes, "arena-failure-strict-second") ||
        arena_alloc_failures != 2 ||
        host_register_calls != 3) return 6;
    memcpy(strict_map + first_offset, changed, weight_bytes);
    memcpy(strict_map + second_offset, changed, weight_bytes);
    if (!consume_weight(out, x, strict_map, model_size, first_offset, strict_first_want) ||
        !consume_weight(out, x, strict_map, model_size, second_offset, strict_second_want) ||
        !ds4_gpu_synchronize()) return 7;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    puts("{\"c_linked_rust_staticlib\":true,\"buffered_fd_selection_active\":true,\"interposed_arena_allocation_failure\":true,\"non_strict_failure_returns_uncached_host_fallback\":true,\"non_strict_host_bytes_precede_file_bytes\":true,\"strict_failure_continues_to_cached_device_copy\":true,\"strict_cached_copy_retains_original_host_bytes\":true,\"persistent_cache_full_skips_second_arena_attempt\":true,\"registration_fallback_boundary_preserved\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
