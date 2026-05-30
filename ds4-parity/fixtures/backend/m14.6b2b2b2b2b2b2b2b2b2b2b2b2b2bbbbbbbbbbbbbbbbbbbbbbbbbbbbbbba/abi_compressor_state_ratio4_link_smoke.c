#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HEAD_DIM 3u
#define RATIO4 4u
#define WIDTH (2u * HEAD_DIM)
#define TAIL_ELEMENTS (RATIO4 * WIDTH)
#define STATE_ROWS 8u
#define STATE_ELEMENTS (STATE_ROWS * WIDTH)
#define APE_ELEMENTS (RATIO4 * WIDTH)
#define POS0 UINT32_MAX

static void reference_state(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape) {
    for (uint32_t index = 0; index < STATE_ELEMENTS; ++index) {
        state_kv[index] = 0.0f;
        state_score[index] = -INFINITY;
    }
    for (uint32_t row = 0; row < RATIO4; ++row) {
        const uint32_t phase = (POS0 + row) % RATIO4;
        for (uint32_t dimension = 0; dimension < WIDTH; ++dimension) {
            const uint64_t input = (uint64_t)row * WIDTH + dimension;
            state_kv[input] = kv[input];
            state_score[input] = sc[input] + ape[(uint64_t)phase * WIDTH + dimension];
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (isinf(expected[index])) {
            if (actual[index] != expected[index]) return 0;
        } else if (fabsf(actual[index] - expected[index]) > 1.0e-6f) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    float kv[TAIL_ELEMENTS];
    float sc[TAIL_ELEMENTS];
    float sentinel[STATE_ELEMENTS];
    float expected_kv[STATE_ELEMENTS];
    float expected_score[STATE_ELEMENTS];
    float got_kv[STATE_ELEMENTS];
    float got_score[STATE_ELEMENTS];
    float model_f32[2u + APE_ELEMENTS] = {0};
    _Float16 model_f16[3u + APE_ELEMENTS] = {0};
    float ape_f16[APE_ELEMENTS];
    const uint64_t ape_f32_offset = 2u * sizeof(float);
    const uint64_t ape_f16_offset = 3u * sizeof(_Float16);

    for (uint32_t index = 0; index < TAIL_ELEMENTS; ++index) {
        kv[index] = (float)((int32_t)((index * 7u + 3u) % 31u) - 15) * 0.125f;
        sc[index] = (float)((int32_t)((index * 11u + 5u) % 37u) - 18) * 0.0625f;
    }
    for (uint32_t index = 0; index < STATE_ELEMENTS; ++index) {
        sentinel[index] = 70.0f + (float)index;
    }
    for (uint32_t index = 0; index < APE_ELEMENTS; ++index) {
        model_f32[2u + index] =
                (float)((int32_t)((index * 5u + 1u) % 19u) - 9) * 0.03125f;
        model_f16[3u + index] = (_Float16)(
                (float)((int32_t)((index * 13u + 4u) % 23u) - 11) * 0.0390625f);
        ape_f16[index] = (float)model_f16[3u + index];
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model_f32, sizeof(model_f32))) return 1;
    ds4_gpu_tensor *kv_tail = ds4_gpu_tensor_alloc(sizeof(kv));
    ds4_gpu_tensor *sc_tail = ds4_gpu_tensor_alloc(sizeof(sc));
    ds4_gpu_tensor *state_kv = ds4_gpu_tensor_alloc(sizeof(sentinel));
    ds4_gpu_tensor *state_score = ds4_gpu_tensor_alloc(sizeof(sentinel));
    ds4_gpu_tensor *short_tail = ds4_gpu_tensor_alloc(sizeof(kv) - sizeof(float));
    ds4_gpu_tensor *short_state = ds4_gpu_tensor_alloc(sizeof(sentinel) - sizeof(float));
    if (!kv_tail || !sc_tail || !state_kv || !state_score || !short_tail || !short_state ||
        !ds4_gpu_tensor_write(kv_tail, 0, kv, sizeof(kv)) ||
        !ds4_gpu_tensor_write(sc_tail, 0, sc, sizeof(sc))) {
        return 2;
    }

    reference_state(expected_kv, expected_score, kv, sc, &model_f32[2]);
    if (!ds4_gpu_tensor_write(state_kv, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 3;
    }

    if (!ds4_gpu_set_model_map(model_f16, sizeof(model_f16))) return 4;
    reference_state(expected_kv, expected_score, kv, sc, ape_f16);
    if (!ds4_gpu_tensor_write(state_kv, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, model_f16,
                sizeof(model_f16), ape_f16_offset, 1u, HEAD_DIM, POS0) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 5;
    }

    if (!ds4_gpu_set_model_map(model_f32, sizeof(model_f32)) ||
        !ds4_gpu_tensor_write(state_kv, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel, sizeof(sentinel)) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), sizeof(model_f32) - sizeof(float), 0u,
                HEAD_DIM, POS0) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_kv, sentinel, STATE_ELEMENTS) ||
        !close_array(got_score, sentinel, STATE_ELEMENTS)) {
        return 6;
    }

    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
                short_state, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, short_state, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, short_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, short_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 2u, HEAD_DIM, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, UINT32_MAX, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, 0u, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                NULL, state_score, kv_tail, sc_tail, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0) ||
        ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv, state_score, kv_tail, sc_tail, NULL,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, POS0)) {
        return 7;
    }

    ds4_gpu_tensor_free(short_state);
    ds4_gpu_tensor_free(short_tail);
    ds4_gpu_tensor_free(state_score);
    ds4_gpu_tensor_free(state_kv);
    ds4_gpu_tensor_free(sc_tail);
    ds4_gpu_tensor_free(kv_tail);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"ratio4_f32_ape_state_output_matches\":true,"
         "\"ratio4_f16_ape_state_output_matches\":true,\"state_initialization_matches\":true,"
         "\"invalid_model_range_preserves_state\":true,\"invalid_shape_rejected\":true,"
         "\"checked_overflow_rejected\":true,\"null_rejected\":true,"
         "\"embedded_compressor_set_rows_kernel_loaded\":true}");
    return 0;
}
