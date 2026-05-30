#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

static int tracked_fd = -1;
static unsigned char *tracked_map = NULL;
static uint64_t tracked_map_size = 0;
static uint64_t file_advice_calls = 0;
static uint64_t file_advice_bytes = 0;
static uint64_t mapping_advice_calls = 0;
static uint64_t mapping_advice_bytes = 0;

int posix_fadvise(int fd, off_t offset, off_t len, int advice) {
    (void)offset;
    if (fd == tracked_fd && advice == POSIX_FADV_DONTNEED) {
        file_advice_calls += 1;
        file_advice_bytes += (uint64_t)len;
    }
    return 0;
}

int posix_madvise(void *addr, size_t len, int advice) {
    const uintptr_t start = (uintptr_t)addr;
    const uintptr_t expected = (uintptr_t)tracked_map;
    if (tracked_map != NULL &&
        advice == POSIX_MADV_DONTNEED &&
        start >= expected &&
        start + len <= expected + tracked_map_size) {
        mapping_advice_calls += 1;
        mapping_advice_bytes += (uint64_t)len;
    }
    return 0;
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

static int cache_with_stderr_capture(
        unsigned char *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *label,
        char *captured,
        size_t captured_bytes) {
    FILE *stream = tmpfile();
    int saved_stderr = -1;
    int ok = 0;
    if (!stream || captured_bytes == 0) return 0;
    saved_stderr = dup(STDERR_FILENO);
    if (saved_stderr < 0 || fflush(stderr) != 0 ||
        dup2(fileno(stream), STDERR_FILENO) < 0) {
        goto done;
    }
    tracked_map = model_map;
    tracked_map_size = model_size;
    ok = ds4_gpu_set_model_map(model_map, model_size) &&
         ds4_gpu_set_model_fd(tracked_fd) &&
         ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, label) &&
         ds4_gpu_synchronize();
    if (fflush(stderr) != 0 || dup2(saved_stderr, STDERR_FILENO) < 0) {
        ok = 0;
        goto done;
    }
    close(saved_stderr);
    saved_stderr = -1;
    rewind(stream);
    const size_t read_bytes = fread(captured, 1, captured_bytes - 1, stream);
    captured[read_bytes] = '\0';
done:
    if (saved_stderr >= 0) {
        (void)dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }
    fclose(stream);
    return ok;
}

int main(void) {
    const uint64_t page_size = 4096;
    const uint64_t weight_offset = sizeof(float);
    const uint64_t copy_chunk_bytes = 16ull * 1024ull * 1024ull;
    const uint64_t cache_bytes = copy_chunk_bytes + page_size;
    const uint64_t model_size = cache_bytes;
    const float x_in[7] = {1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 0.25f, 3.0f};
    const float file_weights[7] = {0.5f, 1.0f, 1.5f, -0.5f, 0.25f, 2.0f, -1.0f};
    const float host_weights[7] = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    float want[7] = {0};
    float got[7] = {0};
    unsigned char *first_map = NULL;
    unsigned char *second_map = NULL;
    unsigned char *suppressed_map = NULL;
    char first_stderr[256] = {0};
    char second_stderr[256] = {0};
    char suppressed_stderr[256] = {0};
    char file_path[] = "/tmp/ds4-fd-source-progress-XXXXXX";
    int fd = -1;

    if (posix_memalign((void **)&first_map, page_size, model_size) != 0 ||
        posix_memalign((void **)&second_map, page_size, model_size) != 0 ||
        posix_memalign((void **)&suppressed_map, page_size, model_size) != 0) return 1;
    memset(first_map, 0, model_size);
    memset(second_map, 0, model_size);
    memset(suppressed_map, 0, model_size);
    memcpy(first_map + weight_offset, host_weights, sizeof(host_weights));
    memcpy(second_map + weight_offset, host_weights, sizeof(host_weights));
    memcpy(suppressed_map + weight_offset, host_weights, sizeof(host_weights));
    reference_weight(want, x_in, file_weights, 7, 1.0e-5f);

    fd = mkstemp(file_path);
    if (fd < 0 || unlink(file_path) != 0 || ftruncate(fd, (off_t)model_size) != 0 ||
        !write_at(fd, file_weights, sizeof(file_weights), (off_t)weight_offset)) return 2;
    tracked_fd = fd;
    if (setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1) != 0 ||
        setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB", "16", 1) != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "1", 1) != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != 0 ||
        unsetenv("DS4_CUDA_KEEP_MODEL_PAGES") != 0 ||
        unsetenv("DS4_CUDA_NO_FD_CACHE") != 0 ||
        unsetenv("DS4_CUDA_DIRECT_MODEL") != 0 ||
        unsetenv("DS4_CUDA_WEIGHT_PRELOAD") != 0) return 3;

    if (!ds4_gpu_init() || !ds4_gpu_set_model_fd(fd)) return 4;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(x_in));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 5;
    if (!cache_with_stderr_capture(
            first_map, model_size, 0, cache_bytes, "progress-first",
            first_stderr, sizeof(first_stderr)) ||
        file_advice_calls != 2 || file_advice_bytes != cache_bytes ||
        mapping_advice_calls != 2 || mapping_advice_bytes != cache_bytes ||
        strstr(first_stderr, "ds4: CUDA loading model tensors into device cache") == NULL ||
        !ds4_gpu_rms_norm_weight_tensor(
            out, x, first_map, model_size, weight_offset, 7, 1.0e-5f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want, 7)) return 6;

    if (!cache_with_stderr_capture(
            second_map, model_size, 0, cache_bytes, "progress-reset",
            second_stderr, sizeof(second_stderr)) ||
        file_advice_calls != 4 || file_advice_bytes != cache_bytes * 2 ||
        mapping_advice_calls != 4 || mapping_advice_bytes != cache_bytes * 2 ||
        strstr(second_stderr, "ds4: CUDA loading model tensors into device cache") == NULL) return 7;

    if (setenv("DS4_CUDA_KEEP_MODEL_PAGES", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE", "1", 1) != 0 ||
        !cache_with_stderr_capture(
            suppressed_map, model_size, 0, cache_bytes, "progress-suppressed",
            suppressed_stderr, sizeof(suppressed_stderr)) ||
        file_advice_calls != 4 || mapping_advice_calls != 4 ||
        strstr(suppressed_stderr, "ds4: CUDA loading model tensors") != NULL) return 8;

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    close(fd);
    free(suppressed_map);
    free(second_map);
    free(first_map);
    puts("{\"c_linked_rust_staticlib\":true,\"page_aligned_host_maps\":true,\"fd_before_map_binds_host_base\":true,\"buffered_only_environment\":true,\"multi_chunk_fd_cache_request\":true,\"source_file_advice_observed\":true,\"source_mapping_advice_observed\":true,\"non_tty_progress_message_captured\":true,\"progress_reset_on_model_replacement\":true,\"keep_pages_suppresses_advice\":true,\"verbose_suppresses_progress\":true,\"fd_bytes_precede_divergent_host_map\":true,\"weighted_output_matches\":true,\"embedded_libdevice_module_loaded\":true}");
    return 0;
}
