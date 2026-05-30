#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    RAW_CAP = 4,
    HEAD_DIM = 7,
    N_TOKENS = 2,
    RAW_COUNT = RAW_CAP * HEAD_DIM,
    KV_COUNT = N_TOKENS * HEAD_DIM,
};

static float half_roundtrip(float value) {
    return (float)((_Float16)value);
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 0.0f) return 0;
    }
    return 1;
}

static void fill_initial_raw(float *raw) {
    for (uint32_t i = 0; i < RAW_COUNT; ++i) {
        raw[i] = -77.0f - (float)i;
    }
}

int main(void) {
    const uint32_t pos0 = UINT32_MAX;
    float raw_initial[RAW_COUNT];
    float expected[RAW_COUNT];
    float got[RAW_COUNT] = {0};
    float kv[KV_COUNT];
    for (uint32_t i = 0; i < KV_COUNT; ++i) {
        kv[i] = (float)((i * 19u + 3u) % 41u) * 0.071f - 1.4f;
    }
    fill_initial_raw(raw_initial);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(sizeof(raw_initial));
    ds4_gpu_tensor *short_raw =
        ds4_gpu_tensor_alloc(sizeof(raw_initial) - sizeof(float));
    ds4_gpu_tensor *kv_tensor = ds4_gpu_tensor_alloc(sizeof(kv));
    ds4_gpu_tensor *short_kv = ds4_gpu_tensor_alloc(sizeof(kv) - sizeof(float));
    if (!raw || !short_raw || !kv_tensor || !short_kv) return 2;

    memcpy(expected, raw_initial, sizeof(expected));
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        const uint32_t row = (uint32_t)(pos0 + token) % RAW_CAP;
        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
            expected[(uint64_t)row * HEAD_DIM + d] =
                half_roundtrip(kv[(uint64_t)token * HEAD_DIM + d]);
        }
    }
    if (!ds4_gpu_tensor_write(raw, 0, raw_initial, sizeof(raw_initial))) {
        fputs("batch raw write failed\n", stderr);
        return 3;
    }
    if (!ds4_gpu_tensor_write(kv_tensor, 0, kv, sizeof(kv))) {
        fputs("batch kv write failed\n", stderr);
        return 3;
    }
    if (!ds4_gpu_store_raw_kv_batch_tensor(raw, kv_tensor, RAW_CAP, pos0,
                                           N_TOKENS, HEAD_DIM)) {
        fputs("batch raw-KV launch failed\n", stderr);
        return 3;
    }
    if (!ds4_gpu_synchronize()) {
        fputs("batch raw-KV synchronization failed\n", stderr);
        return 3;
    }
    if (!ds4_gpu_tensor_read(raw, 0, got, sizeof(got))) {
        fputs("batch raw read failed\n", stderr);
        return 3;
    }
    if (!close_array(got, expected, RAW_COUNT)) {
        for (uint32_t i = 0; i < RAW_COUNT; ++i) {
            if (got[i] != expected[i]) {
                fprintf(stderr, "raw-KV mismatch at %u: got=%f expected=%f\n",
                        i, got[i], expected[i]);
                break;
            }
        }
        return 3;
    }
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        if (got[3u * HEAD_DIM + d] != half_roundtrip(kv[d]) ||
            got[d] != half_roundtrip(kv[HEAD_DIM + d])) return 4;
    }
    for (uint32_t row = 1; row < 3; ++row) {
        if (!close_array(got + row * HEAD_DIM, raw_initial + row * HEAD_DIM,
                         HEAD_DIM)) return 5;
    }

    memcpy(expected, raw_initial, sizeof(expected));
    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
        expected[3u * HEAD_DIM + d] = half_roundtrip(kv[d]);
    }
    if (!ds4_gpu_tensor_write(raw, 0, raw_initial, sizeof(raw_initial)) ||
        !ds4_gpu_store_raw_kv_tensor(raw, kv_tensor, RAW_CAP, pos0,
                                     HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(raw, 0, got, sizeof(got)) ||
        !close_array(got, expected, RAW_COUNT)) return 6;

    if (ds4_gpu_store_raw_kv_batch_tensor(short_raw, kv_tensor, RAW_CAP, pos0,
                                          N_TOKENS, HEAD_DIM) ||
        ds4_gpu_store_raw_kv_batch_tensor(raw, short_kv, RAW_CAP, pos0,
                                          N_TOKENS, HEAD_DIM) ||
        ds4_gpu_store_raw_kv_batch_tensor(raw, kv_tensor, 0, pos0, N_TOKENS,
                                          HEAD_DIM) ||
        ds4_gpu_store_raw_kv_batch_tensor(raw, kv_tensor, RAW_CAP, pos0, 0,
                                          HEAD_DIM) ||
        ds4_gpu_store_raw_kv_batch_tensor(raw, kv_tensor, RAW_CAP, pos0,
                                          N_TOKENS, 0) ||
        ds4_gpu_store_raw_kv_tensor(short_raw, kv_tensor, RAW_CAP, pos0,
                                    HEAD_DIM) ||
        ds4_gpu_store_raw_kv_tensor(NULL, kv_tensor, RAW_CAP, pos0, HEAD_DIM) ||
        ds4_gpu_store_raw_kv_batch_tensor(raw, NULL, RAW_CAP, pos0, N_TOKENS,
                                          HEAD_DIM)) return 7;

    ds4_gpu_tensor_free(short_kv);
    ds4_gpu_tensor_free(kv_tensor);
    ds4_gpu_tensor_free(short_raw);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"batch_fp16_ring_wrap_output_matches\":true,\"uint32_position_wrap_matches\":true,\"single_row_store_matches\":true,\"untouched_rows_preserved\":true,\"zero_grid_rejected\":true,\"invalid_shape_rejected\":true,\"null_rejected\":true,\"embedded_store_raw_kv_batch_kernel_loaded\":true}");
    return 0;
}
