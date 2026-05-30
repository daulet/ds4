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
typedef void *CUevent;
typedef void *CUstream;
typedef CUresult (*cu_event_record_fn)(CUevent, CUstream);

enum {
    CU_ERROR_INVALID_VALUE = 1,
    CU_ERROR_NOT_SUPPORTED = 801,
};

static uint32_t event_record_failures = 0;
static uint32_t host_register_calls = 0;
static int fail_event_records = 0;

CUresult cuEventRecord(CUevent event, CUstream stream) {
    static cu_event_record_fn real_record = NULL;
    if (fail_event_records) {
        event_record_failures += 1;
        return CU_ERROR_INVALID_VALUE;
    }
    if (!real_record) {
        real_record = (cu_event_record_fn)dlsym(RTLD_NEXT, "cuEventRecord");
    }
    return real_record ? real_record(event, stream) : CU_ERROR_NOT_SUPPORTED;
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
    const uint64_t model_size = page_size * 3;
    const uint64_t first_offset = sizeof(float);
    const uint64_t second_offset = page_size + sizeof(float);
    const uint64_t weight_bytes = 7 * sizeof(float);
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float host_first[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    const float file_first[7] = {3.0f, 2.5f, 2.0f, 1.5f, 1.0f, 0.5f, 0.25f};
    const float host_second[7] = {-4.0f, -3.5f, -3.0f, -2.5f, -2.0f, -1.5f, -1.0f};
    const float file_second[7] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    const float changed[7] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    float want_first[7] = {0};
    float want_second[7] = {0};
    void *allocation = NULL;
    unsigned char *model_map = NULL;
    char file_path[] = "/tmp/ds4-fd-event-record-failure-XXXXXX";
    uint32_t register_calls_after_map = 0;
    int fd = -1;

    if (posix_memalign(&allocation, page_size, model_size) != 0) return 1;
    model_map = (unsigned char *)allocation;
    memset(model_map, 0, model_size);
    memcpy(model_map + first_offset, host_first, weight_bytes);
    memcpy(model_map + second_offset, host_second, weight_bytes);
    reference_weight(want_first, x_in, host_first, 7, 1.0e-5f);
    reference_weight(want_second, x_in, host_second, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 ||
        ftruncate(fd, (off_t)model_size) != 0 ||
        !write_at(fd, file_first, weight_bytes, (off_t)first_offset) ||
        !write_at(fd, file_second, weight_bytes, (off_t)second_offset)) return 2;

    if (setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) != 0 ||
        setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB", "16", 1) != 0 ||
        setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1) != 0 ||
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
    register_calls_after_map = host_register_calls;
    fail_event_records = 1;

    if (!ds4_gpu_cache_model_range(
            model_map, model_size, first_offset, weight_bytes, "fd-event-record-first") ||
        event_record_failures != 1 ||
        host_register_calls != register_calls_after_map + 1 ||
        !consume_weight(out, x, model_map, model_size, first_offset, want_first)) return 6;

    if (setenv("DS4_CUDA_STRICT_WEIGHT_CACHE", "1", 1) != 0 ||
        !ds4_gpu_cache_model_range(
            model_map, model_size, second_offset, weight_bytes, "fd-event-record-second") ||
        event_record_failures != 2 ||
        host_register_calls != register_calls_after_map + 1 ||
        !consume_weight(out, x, model_map, model_size, second_offset, want_second)) return 7;

    memcpy(model_map + first_offset, changed, weight_bytes);
    memcpy(model_map + second_offset, changed, weight_bytes);
    if (!consume_weight(out, x, model_map, model_size, first_offset, want_first) ||
        !consume_weight(out, x, model_map, model_size, second_offset, want_second) ||
        !ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    free(allocation);
    puts("{\"c_linked_rust_staticlib\":true,\"buffered_fd_selection_active\":true,\"interposed_fd_event_record_failure\":true,\"event_record_failure_retries_without_cache_full_latch\":true,\"non_strict_event_record_failure_continues_to_cached_device_copy\":true,\"strict_event_record_failure_continues_to_cached_device_copy\":true,\"first_event_record_failure_enters_registration_fallback\":true,\"subsequent_event_record_failure_respects_registration_disable\":true,\"cached_fallback_retains_original_host_bytes\":true,\"host_bytes_precede_file_bytes_after_event_record_failure\":true,\"weighted_outputs_match\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
