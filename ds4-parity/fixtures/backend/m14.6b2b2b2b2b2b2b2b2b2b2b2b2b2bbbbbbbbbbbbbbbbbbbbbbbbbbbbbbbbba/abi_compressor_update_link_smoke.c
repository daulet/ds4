#include "ds4_gpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HEAD_DIM 4u
#define N_ROT 2u
#define RATIO4 4u
#define RATIO3 3u
#define WIDTH4 (2u * HEAD_DIM)
#define WIDTH3 HEAD_DIM
#define STATE4_ELEMENTS (2u * RATIO4 * WIDTH4)
#define STATE3_ELEMENTS (RATIO3 * WIDTH3)
#define COMP_ROW 1u
#define COMP_ELEMENTS (2u * HEAD_DIM)
#define APE4_ELEMENTS (RATIO4 * WIDTH4)
#define APE3_ELEMENTS (RATIO3 * WIDTH3)
#define RMS_EPS 1.0e-5f
#define FREQ_BASE 100.0f
#define FREQ_SCALE 1.0f
#define EXT_FACTOR 0.0f
#define ATTN_FACTOR 1.0f
#define BETA_FAST 32.0f
#define BETA_SLOW 1.0f

struct model4_f32 {
    float prefix[2];
    float ape[APE4_ELEMENTS];
    float norm[HEAD_DIM];
};

struct model4_f16 {
    _Float16 prefix[3];
    _Float16 ape[APE4_ELEMENTS];
    float norm[HEAD_DIM];
};

struct model3_f32 {
    float prefix;
    float ape[APE3_ELEMENTS];
    float norm[HEAD_DIM];
};

static void reference_store_one(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape,
        uint32_t ratio,
        uint32_t pos) {
    const uint32_t width = ratio == RATIO4 ? WIDTH4 : WIDTH3;
    const uint32_t phase = pos % ratio;
    const uint32_t row = ratio == RATIO4 ? ratio + phase : phase;
    for (uint32_t dimension = 0; dimension < width; ++dimension) {
        const uint64_t output = (uint64_t)row * width + dimension;
        state_kv[output] = kv[dimension];
        state_score[output] = sc[dimension] + ape[(uint64_t)phase * width + dimension];
    }
}

static void reference_pool(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t ratio) {
    const uint32_t width = ratio == RATIO4 ? WIDTH4 : WIDTH3;
    for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
        float max_score = -INFINITY;
        if (ratio == RATIO4) {
            for (uint32_t candidate = 0; candidate < RATIO4; ++candidate) {
                const uint64_t prior = (uint64_t)candidate * width + dimension;
                const uint64_t active =
                        (uint64_t)(ratio + candidate) * width + HEAD_DIM + dimension;
                if (state_score[prior] > max_score) max_score = state_score[prior];
                if (state_score[active] > max_score) max_score = state_score[active];
            }
        } else {
            for (uint32_t candidate = 0; candidate < ratio; ++candidate) {
                const uint64_t index = (uint64_t)candidate * width + dimension;
                if (state_score[index] > max_score) max_score = state_score[index];
            }
        }
        float denominator = 0.0f;
        float accumulator = 0.0f;
        if (ratio == RATIO4) {
            for (uint32_t candidate = 0; candidate < RATIO4; ++candidate) {
                const uint64_t prior = (uint64_t)candidate * width + dimension;
                const uint64_t active =
                        (uint64_t)(ratio + candidate) * width + HEAD_DIM + dimension;
                const float prior_weight = expf(state_score[prior] - max_score);
                const float active_weight = expf(state_score[active] - max_score);
                denominator += prior_weight + active_weight;
                accumulator += state_kv[prior] * prior_weight + state_kv[active] * active_weight;
            }
        } else {
            for (uint32_t candidate = 0; candidate < ratio; ++candidate) {
                const uint64_t index = (uint64_t)candidate * width + dimension;
                const float weight = expf(state_score[index] - max_score);
                denominator += weight;
                accumulator += state_kv[index] * weight;
            }
        }
        row[dimension] = denominator != 0.0f ? accumulator / denominator : 0.0f;
    }
}

static void reference_rms(float *row, const float *norm) {
    float sum = 0.0f;
    for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
        sum += row[dimension] * row[dimension];
    }
    const float scale = 1.0f / sqrtf(sum / HEAD_DIM + RMS_EPS);
    for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
        row[dimension] *= scale * norm[dimension];
    }
}

static void reference_rope(float *row, uint32_t pos) {
    const float theta = (float)pos * FREQ_SCALE;
    const float cosine = cosf(theta) * ATTN_FACTOR;
    const float sine = sinf(theta) * ATTN_FACTOR;
    const uint32_t tail = HEAD_DIM - N_ROT;
    const float x0 = row[tail];
    const float x1 = row[tail + 1u];
    row[tail] = x0 * cosine - x1 * sine;
    row[tail + 1u] = x0 * sine + x1 * cosine;
}

static void reference_shift_ratio4(float *state_kv, float *state_score) {
    const uint32_t half = RATIO4 * WIDTH4;
    for (uint32_t index = 0; index < half; ++index) {
        state_kv[index] = state_kv[half + index];
        state_score[index] = state_score[half + index];
    }
}

static int reference_update(
        float *comp,
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape,
        const float *norm,
        uint32_t ratio,
        uint32_t pos,
        uint32_t n_rot) {
    reference_store_one(state_kv, state_score, kv, sc, ape, ratio, pos);
    if (((pos + 1u) % ratio) != 0u) return 1;
    float *row = comp + COMP_ROW * HEAD_DIM;
    reference_pool(row, state_kv, state_score, ratio);
    reference_rms(row, norm);
    if (n_rot == 0u) return 0;
    reference_rope(row, pos + 1u - ratio);
    if (ratio == RATIO4) reference_shift_ratio4(state_kv, state_score);
    return 1;
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 4.0e-4f) return 0;
    }
    return 1;
}

int main(void) {
    float kv[WIDTH4];
    float sc[WIDTH4];
    float initial_kv[STATE4_ELEMENTS];
    float initial_score[STATE4_ELEMENTS];
    float comp_sentinel[COMP_ELEMENTS];
    float expected_comp[COMP_ELEMENTS];
    float expected_kv[STATE4_ELEMENTS];
    float expected_score[STATE4_ELEMENTS];
    float got_comp[COMP_ELEMENTS];
    float got_kv[STATE4_ELEMENTS];
    float got_score[STATE4_ELEMENTS];
    float ape4_f16[APE4_ELEMENTS];
    struct model4_f32 model4_f32 = {0};
    struct model4_f16 model4_f16 = {0};
    struct model3_f32 model3_f32 = {0};
    const uint64_t ape4_f32_offset = offsetof(struct model4_f32, ape);
    const uint64_t norm4_f32_offset = offsetof(struct model4_f32, norm);
    const uint64_t ape4_f16_offset = offsetof(struct model4_f16, ape);
    const uint64_t norm4_f16_offset = offsetof(struct model4_f16, norm);
    const uint64_t ape3_f32_offset = offsetof(struct model3_f32, ape);
    const uint64_t norm3_f32_offset = offsetof(struct model3_f32, norm);

    for (uint32_t index = 0; index < WIDTH4; ++index) {
        kv[index] = (float)((int32_t)((index * 7u + 3u) % 31u) - 15) * 0.125f;
        sc[index] = (float)((int32_t)((index * 11u + 5u) % 37u) - 18) * 0.0625f;
    }
    for (uint32_t index = 0; index < STATE4_ELEMENTS; ++index) {
        initial_kv[index] =
                (float)((int32_t)((index * 3u + 2u) % 23u) - 11) * 0.078125f;
        initial_score[index] =
                (float)((int32_t)((index * 9u + 1u) % 29u) - 14) * 0.046875f;
    }
    for (uint32_t index = 0; index < COMP_ELEMENTS; ++index) {
        comp_sentinel[index] = 180.0f + (float)index;
    }
    for (uint32_t index = 0; index < APE4_ELEMENTS; ++index) {
        model4_f32.ape[index] =
                (float)((int32_t)((index * 5u + 1u) % 19u) - 9) * 0.03125f;
        model4_f16.ape[index] = (_Float16)(
                (float)((int32_t)((index * 13u + 4u) % 23u) - 11) * 0.0390625f);
        ape4_f16[index] = (float)model4_f16.ape[index];
    }
    for (uint32_t index = 0; index < APE3_ELEMENTS; ++index) {
        model3_f32.ape[index] =
                (float)((int32_t)((index * 17u + 6u) % 31u) - 15) * 0.0234375f;
    }
    for (uint32_t index = 0; index < HEAD_DIM; ++index) {
        model4_f32.norm[index] = 0.75f + (float)index * 0.125f;
        model4_f16.norm[index] = 0.625f + (float)index * 0.15625f;
        model3_f32.norm[index] = 0.6875f + (float)index * 0.09375f;
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(&model4_f32, sizeof(model4_f32))) return 1;
    ds4_gpu_tensor *kv_tensor = ds4_gpu_tensor_alloc(sizeof(kv));
    ds4_gpu_tensor *sc_tensor = ds4_gpu_tensor_alloc(sizeof(sc));
    ds4_gpu_tensor *state_kv = ds4_gpu_tensor_alloc(sizeof(initial_kv));
    ds4_gpu_tensor *state_score = ds4_gpu_tensor_alloc(sizeof(initial_score));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
    ds4_gpu_tensor *short_input = ds4_gpu_tensor_alloc((WIDTH3 - 1u) * sizeof(float));
    ds4_gpu_tensor *short_state = ds4_gpu_tensor_alloc((STATE3_ELEMENTS - 1u) * sizeof(float));
    ds4_gpu_tensor *short_comp = ds4_gpu_tensor_alloc((COMP_ELEMENTS - 1u) * sizeof(float));
    if (!kv_tensor || !sc_tensor || !state_kv || !state_score || !comp || !short_input ||
        !short_state || !short_comp ||
        !ds4_gpu_tensor_write(kv_tensor, 0, kv, sizeof(kv)) ||
        !ds4_gpu_tensor_write(sc_tensor, 0, sc, sizeof(sc))) {
        return 2;
    }

    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, initial_kv, sizeof(expected_kv));
    memcpy(expected_score, initial_score, sizeof(expected_score));
    if (!reference_update(
                expected_comp, expected_kv, expected_score, kv, sc, model4_f32.ape,
                model4_f32.norm, RATIO4, 6u, N_ROT) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, initial_kv, sizeof(initial_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial_score, sizeof(initial_score)) ||
        !ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, 6u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE4_ELEMENTS)) {
        return 3;
    }

    if (!ds4_gpu_set_model_map(&model4_f16, sizeof(model4_f16))) return 4;
    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, initial_kv, sizeof(expected_kv));
    memcpy(expected_score, initial_score, sizeof(expected_score));
    if (!reference_update(
                expected_comp, expected_kv, expected_score, kv, sc, ape4_f16,
                model4_f16.norm, RATIO4, 7u, N_ROT) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, initial_kv, sizeof(initial_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial_score, sizeof(initial_score)) ||
        !ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f16,
                sizeof(model4_f16), ape4_f16_offset, 1u, norm4_f16_offset, 0u,
                HEAD_DIM, RATIO4, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE4_ELEMENTS)) {
        return 5;
    }

    if (!ds4_gpu_set_model_map(&model3_f32, sizeof(model3_f32))) return 6;
    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, initial_kv, sizeof(expected_kv));
    memcpy(expected_score, initial_score, sizeof(expected_score));
    if (!reference_update(
                expected_comp, expected_kv, expected_score, kv, sc, model3_f32.ape,
                model3_f32.norm, RATIO3, 2u, N_ROT) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, initial_kv, sizeof(initial_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial_score, sizeof(initial_score)) ||
        !ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model3_f32,
                sizeof(model3_f32), ape3_f32_offset, 0u, norm3_f32_offset, 0u,
                HEAD_DIM, RATIO3, 2u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE4_ELEMENTS)) {
        return 7;
    }

    if (!ds4_gpu_set_model_map(&model4_f32, sizeof(model4_f32))) return 8;
    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, initial_kv, sizeof(expected_kv));
    memcpy(expected_score, initial_score, sizeof(expected_score));
    if (reference_update(
                expected_comp, expected_kv, expected_score, kv, sc, model4_f32.ape,
                model4_f32.norm, RATIO4, UINT32_MAX, 0u) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, initial_kv, sizeof(initial_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial_score, sizeof(initial_score)) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, UINT32_MAX, COMP_ROW, 0u, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE4_ELEMENTS)) {
        return 9;
    }

    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, initial_kv, sizeof(expected_kv));
    memcpy(expected_score, initial_score, sizeof(expected_score));
    if (reference_update(
                expected_comp, expected_kv, expected_score, kv, sc, model4_f32.ape,
                model4_f32.norm, RATIO4, 7u, 0u) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, initial_kv, sizeof(initial_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial_score, sizeof(initial_score)) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, 7u, COMP_ROW, 0u, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE4_ELEMENTS)) {
        return 10;
    }

    if (!ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, initial_kv, sizeof(initial_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, initial_score, sizeof(initial_score)) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), sizeof(model4_f32) - sizeof(float), 0u,
                norm4_f32_offset, 0u, HEAD_DIM, RATIO4, 7u, COMP_ROW, N_ROT,
                4096u, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, comp_sentinel, COMP_ELEMENTS) ||
        !close_array(got_kv, initial_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, initial_score, STATE4_ELEMENTS)) {
        return 11;
    }

    if (ds4_gpu_compressor_update_tensor(
                short_input, sc_tensor, state_kv, state_score, comp, &model3_f32,
                sizeof(model3_f32), ape3_f32_offset, 0u, norm3_f32_offset, 0u,
                HEAD_DIM, RATIO3, 2u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, short_state, state_score, comp, &model3_f32,
                sizeof(model3_f32), ape3_f32_offset, 0u, norm3_f32_offset, 0u,
                HEAD_DIM, RATIO3, 2u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, short_comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 1u,
                HEAD_DIM, RATIO4, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, 0u, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                UINT32_MAX, RATIO4, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, 7u, UINT32_MAX, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                NULL, sc_tensor, state_kv, state_score, comp, &model4_f32,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_update_tensor(
                kv_tensor, sc_tensor, state_kv, state_score, comp, NULL,
                sizeof(model4_f32), ape4_f32_offset, 0u, norm4_f32_offset, 0u,
                HEAD_DIM, RATIO4, 7u, COMP_ROW, N_ROT, 4096u, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS)) {
        return 12;
    }

    ds4_gpu_tensor_free(short_comp);
    ds4_gpu_tensor_free(short_state);
    ds4_gpu_tensor_free(short_input);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(state_score);
    ds4_gpu_tensor_free(state_kv);
    ds4_gpu_tensor_free(sc_tensor);
    ds4_gpu_tensor_free(kv_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"ratio4_no_emit_store_only_matches\":true,"
         "\"ratio4_emit_f16_output_matches\":true,\"general_ratio_emit_output_matches\":true,"
         "\"uint32_emit_wrap_matches\":true,\"ratio4_shift_after_emit_matches\":true,"
         "\"n_rot_zero_partial_failure_matches\":true,"
         "\"invalid_model_range_preserves_state_and_output\":true,"
         "\"invalid_shape_rejected\":true,\"checked_overflow_rejected\":true,"
         "\"null_rejected\":true,\"embedded_compressor_update_pool_kernel_loaded\":true,"
         "\"embedded_compressor_shift_ratio4_kernel_loaded\":true}");
    return 0;
}
