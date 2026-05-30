#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    HEAD_DIM = 75,
    N_ROT = 6,
    RAW_CAP = 4,
    RAW_ROW = UINT32_MAX,
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
    for (uint32_t off = 0; off < prefix; off += 64) {
        const uint32_t size = prefix - off < 64 ? prefix - off : 64;
        float amax = 1.0e-4f;
        for (uint32_t i = 0; i < size; ++i) {
            if (fabsf(values[off + i]) > amax) amax = fabsf(values[off + i]);
        }
        const float scale = powf(2.0f, ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < size; ++i) {
            float scaled = values[off + i] / scale;
            if (scaled > 448.0f) scaled = 448.0f;
            if (scaled < -448.0f) scaled = -448.0f;
            values[off + i] = e4m3fn_dequant(scaled) * scale;
        }
    }
}

static float half_roundtrip(float value) {
    return (float)(_Float16)value;
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    float input[HEAD_DIM];
    float expected_kv[HEAD_DIM];
    float got_kv[HEAD_DIM] = {0};
    float raw_initial[RAW_CAP * HEAD_DIM];
    float expected_raw[RAW_CAP * HEAD_DIM];
    float got_raw[RAW_CAP * HEAD_DIM] = {0};
    const uint32_t row = (uint32_t)RAW_ROW % RAW_CAP;
    for (uint32_t i = 0; i < HEAD_DIM; ++i) {
        input[i] = (float)((i * 29u + 11u) % 151u) * 0.09375f - 6.75f;
    }
    memcpy(expected_kv, input, sizeof(expected_kv));
    reference_fp8_kv_quantize(expected_kv);
    for (uint32_t i = 0; i < RAW_CAP * HEAD_DIM; ++i) {
        raw_initial[i] = -99.0f - (float)i;
    }
    memcpy(expected_raw, raw_initial, sizeof(expected_raw));
    for (uint32_t i = 0; i < HEAD_DIM; ++i) {
        expected_raw[(uint64_t)row * HEAD_DIM + i] = half_roundtrip(expected_kv[i]);
    }

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *short_kv = ds4_gpu_tensor_alloc(sizeof(input) - sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(sizeof(raw_initial));
    ds4_gpu_tensor *short_raw = ds4_gpu_tensor_alloc(sizeof(raw_initial) - sizeof(float));
    if (!kv || !short_kv || !raw || !short_raw) return 2;
    if (!ds4_gpu_tensor_write(kv, 0, input, sizeof(input)) ||
        !ds4_gpu_tensor_write(raw, 0, raw_initial, sizeof(raw_initial)) ||
        !ds4_gpu_kv_fp8_store_raw_tensor(kv, raw, RAW_CAP, RAW_ROW, HEAD_DIM, N_ROT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(raw, 0, got_raw, sizeof(got_raw)) ||
        !close_array(got_kv, expected_kv, HEAD_DIM) ||
        !close_array(got_raw, expected_raw, RAW_CAP * HEAD_DIM)) return 3;
    if (!close_array(got_kv + HEAD_DIM - N_ROT, input + HEAD_DIM - N_ROT, N_ROT)) return 4;

    if (!ds4_gpu_tensor_write(kv, 0, input, sizeof(input)) ||
        ds4_gpu_kv_fp8_store_raw_tensor(kv, short_raw, RAW_CAP, RAW_ROW, HEAD_DIM, N_ROT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(kv, 0, got_kv, sizeof(got_kv)) ||
        !close_array(got_kv, expected_kv, HEAD_DIM)) return 5;

    if (ds4_gpu_kv_fp8_store_raw_tensor(short_kv, raw, RAW_CAP, RAW_ROW, HEAD_DIM, N_ROT) ||
        ds4_gpu_kv_fp8_store_raw_tensor(kv, raw, RAW_CAP, RAW_ROW, HEAD_DIM, HEAD_DIM + 1) ||
        ds4_gpu_kv_fp8_store_raw_tensor(NULL, raw, RAW_CAP, RAW_ROW, HEAD_DIM, N_ROT) ||
        ds4_gpu_kv_fp8_store_raw_tensor(kv, NULL, RAW_CAP, RAW_ROW, HEAD_DIM, N_ROT)) return 6;

    ds4_gpu_tensor_free(short_raw);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(short_kv);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"composed_fp8_raw_store_output_matches\":true,\"fp8_rope_tail_preserved\":true,\"f16_raw_store_roundtrip_matches\":true,\"uint32_raw_row_wrap_matches\":true,\"raw_store_failure_retains_fp8_mutation\":true,\"invalid_shape_rejected\":true,\"null_rejected\":true,\"reuses_embedded_fp8_and_raw_store_kernels\":true}");
    return 0;
}
