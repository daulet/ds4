#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HEAD_DIM 3u
#define RATIO4 4u
#define WIDTH4 (2u * HEAD_DIM)
#define TOKENS4 3u
#define ROWS4 (2u * RATIO4)
#define POS4 2u
#define WRAP_RATIO 6u
#define WRAP_WIDTH HEAD_DIM
#define WRAP_TOKENS 2u
#define WRAP_ROWS WRAP_RATIO
#define WRAP_POS UINT32_MAX
#define INPUT_ELEMENTS (TOKENS4 * WIDTH4)
#define STATE_ELEMENTS (ROWS4 * WIDTH4)
#define APE_ELEMENTS (RATIO4 * WIDTH4)

static void reference_store(
        float *state_kv,
        float *state_score,
        const float *initial,
        const float *kv,
        const float *sc,
        const float *ape,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    memcpy(state_kv, initial, STATE_ELEMENTS * sizeof(float));
    memcpy(state_score, initial, STATE_ELEMENTS * sizeof(float));
    for (uint32_t token = 0; token < n_tokens; ++token) {
        const uint32_t phase = (pos0 + token) % ratio;
        const uint32_t row = ratio == 4u ? ratio + phase : phase;
        for (uint32_t dimension = 0; dimension < width; ++dimension) {
            const uint64_t input = (uint64_t)token * width + dimension;
            const uint64_t output = (uint64_t)row * width + dimension;
            state_kv[output] = kv[input];
            state_score[output] = sc[input] + ape[(uint64_t)phase * width + dimension];
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-6f) return 0;
    }
    return 1;
}

int main(void) {
    float kv[INPUT_ELEMENTS];
    float sc[INPUT_ELEMENTS];
    float initial[STATE_ELEMENTS];
    float expected_kv[STATE_ELEMENTS];
    float expected_score[STATE_ELEMENTS];
    float got_kv[STATE_ELEMENTS] = {0};
    float got_score[STATE_ELEMENTS] = {0};
    float model_f32[2u + APE_ELEMENTS] = {0};
    _Float16 model_f16[3u + APE_ELEMENTS] = {0};
    float ape_f16[APE_ELEMENTS];
    const uint64_t ape_f32_offset = 2u * sizeof(float);
    const uint64_t ape_f16_offset = 3u * sizeof(_Float16);

    for (uint32_t index = 0; index < INPUT_ELEMENTS; ++index) {
        kv[index] = (float)((int32_t)((index * 7u + 3u) % 31u) - 15) * 0.125f;
        sc[index] = (float)((int32_t)((index * 11u + 5u) % 37u) - 18) * 0.0625f;
    }
    for (uint32_t index = 0; index < STATE_ELEMENTS; ++index) {
        initial[index] = -50.0f - (float)index;
    }
    for (uint32_t index = 0; index < APE_ELEMENTS; ++index) {
        model_f32[2u + index] =
                (float)((int32_t)((index * 5u + 1u) % 19u) - 9) * 0.03125f;
        model_f16[3u + index] = (_Float16)(
                (float)((int32_t)((index * 13u + 4u) % 23u) - 11) * 0.0390625f);
        ape_f16[index] = (float)model_f16[3u + index];
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model_f32, sizeof(model_f32))) return 1;
    ds4_gpu_tensor *kv_tensor = ds4_gpu_tensor_alloc(sizeof(kv));
    ds4_gpu_tensor *sc_tensor = ds4_gpu_tensor_alloc(sizeof(sc));
    ds4_gpu_tensor *state_kv = ds4_gpu_tensor_alloc(sizeof(initial));
    ds4_gpu_tensor *state_score = ds4_gpu_tensor_alloc(sizeof(initial));
    ds4_gpu_tensor *short_kv = ds4_gpu_tensor_alloc(sizeof(kv) - sizeof(float));
    ds4_gpu_tensor *short_sc = ds4_gpu_tensor_alloc(sizeof(sc) - sizeof(float));
    ds4_gpu_tensor *short_state = ds4_gpu_tensor_alloc(sizeof(initial) - sizeof(float));
    if (!kv_tensor || !sc_tensor || !state_kv || !state_score || !short_kv || !short_sc ||
        !short_state ||
        !ds4_gpu_tensor_write(kv_tensor, 0, kv, sizeof(kv)) ||
        !ds4_gpu_tensor_write(sc_tensor, 0, sc, sizeof(sc))) {
        return 2;
    }

    reference_store(
            expected_kv, expected_score, initial, kv, sc, &model_f32[2], HEAD_DIM,
            RATIO4, POS4, TOKENS4);
    if (!ds4_gpu_tensor_write(state_kv, 0, initial, sizeof(initial)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial, sizeof(initial)) ||
        !ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 3;
    }

    if (!ds4_gpu_set_model_map(model_f16, sizeof(model_f16))) return 4;
    reference_store(
            expected_kv, expected_score, initial, kv, sc, ape_f16, HEAD_DIM,
            RATIO4, POS4, TOKENS4);
    if (!ds4_gpu_tensor_write(state_kv, 0, initial, sizeof(initial)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial, sizeof(initial)) ||
        !ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f16,
                sizeof(model_f16), ape_f16_offset, 1u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 5;
    }

    if (!ds4_gpu_set_model_map(model_f32, sizeof(model_f32))) return 6;
    reference_store(
            expected_kv, expected_score, initial, kv, sc, &model_f32[2], HEAD_DIM,
            WRAP_RATIO, WRAP_POS, WRAP_TOKENS);
    if (!ds4_gpu_tensor_write(state_kv, 0, initial, sizeof(initial)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial, sizeof(initial)) ||
        !ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, WRAP_RATIO,
                WRAP_POS, WRAP_TOKENS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 7;
    }

    if (ds4_gpu_compressor_store_batch_tensor(
                short_kv, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, short_sc, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, short_state, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, short_state, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), sizeof(model_f32) - sizeof(float), 0u,
                HEAD_DIM, RATIO4, POS4, TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 2u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, UINT32_MAX, RATIO4,
                POS4, TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, 0u, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                0u) ||
        ds4_gpu_compressor_store_batch_tensor(
                NULL, sc_tensor, state_kv, state_score, model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4) ||
        ds4_gpu_compressor_store_batch_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, NULL,
                sizeof(model_f32), ape_f32_offset, 0u, HEAD_DIM, RATIO4, POS4,
                TOKENS4)) {
        return 8;
    }

    ds4_gpu_tensor_free(short_state);
    ds4_gpu_tensor_free(short_sc);
    ds4_gpu_tensor_free(short_kv);
    ds4_gpu_tensor_free(state_score);
    ds4_gpu_tensor_free(state_kv);
    ds4_gpu_tensor_free(sc_tensor);
    ds4_gpu_tensor_free(kv_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"ratio4_f32_ape_output_matches\":true,"
         "\"ratio4_f16_ape_output_matches\":true,\"uint32_position_wrap_matches\":true,"
         "\"untouched_rows_preserved\":true,\"invalid_model_range_rejected\":true,"
         "\"invalid_shape_rejected\":true,\"checked_overflow_rejected\":true,"
         "\"null_rejected\":true,\"embedded_compressor_store_kernel_loaded\":true}");
    return 0;
}
