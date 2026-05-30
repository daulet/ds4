#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    N_TOK = 2,
    HEAD_DIM = 75,
    N_ROT = 6,
    COUNT = N_TOK * HEAD_DIM,
};

static float e4m3fn_value(int value) {
    const int exponent = (value >> 3) & 15;
    const int mantissa = value & 7;
    if (exponent == 0) return (float)mantissa * 0.001953125f;
    return (1.0f + (float)mantissa * 0.125f) *
           powf(2.0f, (float)exponent - 7.0f);
}

static float e4m3fn_dequant(float value) {
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabsf(value);
    if (magnitude > 448.0f) magnitude = 448.0f;
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_value(mid) <= magnitude) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(magnitude - e4m3fn_value(best));
        const float next_diff = fabsf(magnitude - e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            ++best;
        }
    }
    return sign * e4m3fn_value(best);
}

static void reference_fp8_kv_quantize(float *values) {
    const uint32_t prefix = HEAD_DIM - N_ROT;
    for (uint32_t row = 0; row < N_TOK; ++row) {
        float *out = values + (uint64_t)row * HEAD_DIM;
        for (uint32_t off = 0; off < prefix; off += 64) {
            const uint32_t size = prefix - off < 64 ? prefix - off : 64;
            float amax = 1.0e-4f;
            for (uint32_t i = 0; i < size; ++i) {
                if (fabsf(out[off + i]) > amax) amax = fabsf(out[off + i]);
            }
            const float scale = powf(2.0f, ceilf(log2f(amax / 448.0f)));
            for (uint32_t i = 0; i < size; ++i) {
                float scaled = out[off + i] / scale;
                if (scaled > 448.0f) scaled = 448.0f;
                if (scaled < -448.0f) scaled = -448.0f;
                out[off + i] = e4m3fn_dequant(scaled) * scale;
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
        input[i] = (float)((i * 29u + 11u) % 151u) * 0.09375f - 6.75f;
    }
    memcpy(expected, input, sizeof(expected));
    reference_fp8_kv_quantize(expected);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_alloc(sizeof(input) - sizeof(float));
    if (!x || !short_x) return 2;
    if (!ds4_gpu_tensor_write(x, 0, input, sizeof(input)) ||
        !ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, N_TOK, HEAD_DIM, N_ROT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, expected, COUNT)) return 3;

    int prefix_changed = 0;
    for (uint32_t row = 0; row < N_TOK; ++row) {
        for (uint32_t i = 0; i < HEAD_DIM - N_ROT; ++i) {
            const uint32_t index = row * HEAD_DIM + i;
            if (got[index] != input[index]) prefix_changed = 1;
        }
        if (!close_array(got + row * HEAD_DIM + HEAD_DIM - N_ROT,
                         input + row * HEAD_DIM + HEAD_DIM - N_ROT,
                         N_ROT)) return 4;
    }
    if (!prefix_changed) return 5;

    if (!ds4_gpu_tensor_write(x, 0, input, HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, 1, HEAD_DIM, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(x, 0, got, HEAD_DIM * sizeof(float)) ||
        !close_array(got, input, HEAD_DIM)) return 6;
    if (!ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, 1, 0, 0) ||
        !ds4_gpu_synchronize()) return 7;

    if (ds4_gpu_dsv4_fp8_kv_quantize_tensor(short_x, N_TOK, HEAD_DIM, N_ROT) ||
        ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, N_TOK, HEAD_DIM, HEAD_DIM + 1) ||
        ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, 0, HEAD_DIM, N_ROT) ||
        ds4_gpu_dsv4_fp8_kv_quantize_tensor(NULL, N_TOK, HEAD_DIM, N_ROT)) return 8;

    ds4_gpu_tensor_free(short_x);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"fp8_prefix_output_matches\":true,\"fp8_partial_chunk_matches\":true,\"fp8_rope_tail_preserved\":true,\"empty_prefix_noop_preserved\":true,\"zero_width_noop_preserved\":true,\"invalid_shape_rejected\":true,\"null_rejected\":true,\"embedded_fp8_kv_quantize_kernel_loaded\":true}");
    return 0;
}
