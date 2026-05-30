#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    N_ROWS = 2,
    HEAD_DIM = 128,
    COUNT = N_ROWS * HEAD_DIM,
};

static float e2m1fn_value(int value) {
    switch (value & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

static float e2m1fn_dequant(float value) {
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabsf(value);
    if (magnitude > 6.0f) magnitude = 6.0f;
    int best = 0;
    float best_diff = magnitude;
    for (int candidate = 1; candidate < 8; ++candidate) {
        const float diff = fabsf(magnitude - e2m1fn_value(candidate));
        if (diff < best_diff ||
            (diff == best_diff && (candidate & 1) == 0 && (best & 1) != 0)) {
            best = candidate;
            best_diff = diff;
        }
    }
    return sign * e2m1fn_value(best);
}

static void reference_indexer_qat(float *values) {
    for (uint32_t row_index = 0; row_index < N_ROWS; ++row_index) {
        float *row = values + (uint64_t)row_index * HEAD_DIM;
        for (uint32_t stride = 1; stride < HEAD_DIM; stride <<= 1) {
            for (uint32_t base = 0; base < HEAD_DIM; base += 2 * stride) {
                for (uint32_t lane = 0; lane < stride; ++lane) {
                    const float a = row[base + lane];
                    const float b = row[base + stride + lane];
                    row[base + lane] = a + b;
                    row[base + stride + lane] = a - b;
                }
            }
        }
        for (uint32_t i = 0; i < HEAD_DIM; ++i) {
            row[i] *= 0.08838834764831845f;
        }
        for (uint32_t off = 0; off < HEAD_DIM; off += 32) {
            float amax = 7.052966104933725e-38f;
            for (uint32_t i = 0; i < 32; ++i) {
                if (fabsf(row[off + i]) > amax) amax = fabsf(row[off + i]);
            }
            const float scale = powf(2.0f, ceilf(log2f(amax / 6.0f)));
            for (uint32_t i = 0; i < 32; ++i) {
                float scaled = row[off + i] / scale;
                if (scaled > 6.0f) scaled = 6.0f;
                if (scaled < -6.0f) scaled = -6.0f;
                row[off + i] = e2m1fn_dequant(scaled) * scale;
            }
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    float input[COUNT];
    float expected[COUNT];
    float got[COUNT] = {0};
    for (uint32_t i = 0; i < COUNT; ++i) {
        input[i] = (float)((i * 19u + 3u) % 101u) * 0.03125f - 1.5f;
    }
    memcpy(expected, input, sizeof(expected));
    reference_indexer_qat(expected);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_alloc(sizeof(input) - sizeof(float));
    if (!x || !short_x) return 2;
    if (!ds4_gpu_tensor_write(x, 0, input, sizeof(input)) ||
        !ds4_gpu_dsv4_indexer_qat_tensor(x, N_ROWS, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, expected, COUNT)) return 3;

    int changed = 0;
    for (uint32_t i = 0; i < COUNT; ++i) {
        if (got[i] != input[i]) changed = 1;
    }
    if (!changed) return 4;
    if (ds4_gpu_dsv4_indexer_qat_tensor(short_x, N_ROWS, HEAD_DIM) ||
        ds4_gpu_dsv4_indexer_qat_tensor(x, 0, HEAD_DIM) ||
        ds4_gpu_dsv4_indexer_qat_tensor(x, N_ROWS, HEAD_DIM - 1) ||
        ds4_gpu_dsv4_indexer_qat_tensor(NULL, N_ROWS, HEAD_DIM)) return 5;

    ds4_gpu_tensor_free(short_x);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"multi_row_indexer_hadamard_fp4_output_matches\":true,\"fp4_block_scale_matches\":true,\"short_tensor_rejected\":true,\"invalid_shape_rejected\":true,\"null_rejected\":true,\"embedded_indexer_hadamard_fp4_kernel_loaded\":true}");
    return 0;
}
