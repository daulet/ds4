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
#define N_TOKENS4 9u
#define N_TOKENS3 5u
#define N_COMP4 (N_TOKENS4 / RATIO4)
#define INPUT4_ELEMENTS (N_TOKENS4 * WIDTH4)
#define INPUT3_ELEMENTS (N_TOKENS3 * WIDTH3)
#define STATE4_ELEMENTS (2u * RATIO4 * WIDTH4)
#define STATE3_ELEMENTS (RATIO3 * WIDTH3)
#define COMP_ELEMENTS (N_COMP4 * HEAD_DIM)
#define APE4_ELEMENTS (RATIO4 * WIDTH4)
#define APE3_ELEMENTS (RATIO3 * WIDTH3)
#define RMS_EPS 1.0e-5f
#define FREQ_BASE 100.0f
#define FREQ_SCALE 1.0f
#define EXT_FACTOR 0.0f
#define ATTN_FACTOR 1.0f
#define BETA_FAST 32.0f
#define BETA_SLOW 1.0f

struct model4_f16 {
    _Float16 prefix[3];
    _Float16 ape[APE4_ELEMENTS];
    float norm[HEAD_DIM];
};

struct model3_f32 {
    float prefix[2];
    float ape[APE3_ELEMENTS];
    float norm[HEAD_DIM];
};

static void set_state_rows(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t source,
        uint32_t destination,
        uint32_t rows) {
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t token = source + row;
        const uint32_t phase = (pos0 + token) % ratio;
        for (uint32_t dimension = 0; dimension < width; ++dimension) {
            const uint64_t input = (uint64_t)token * width + dimension;
            const uint64_t output = (uint64_t)(destination + row) * width + dimension;
            state_kv[output] = kv[input];
            state_score[output] =
                    sc[input] + ape[(uint64_t)phase * width + dimension];
        }
    }
}

static void reference_prefill(
        float *comp,
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const float *ape,
        const float *norm,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t n_rot) {
    const uint32_t width = ratio == RATIO4 ? WIDTH4 : WIDTH3;
    const uint32_t state_rows = ratio == RATIO4 ? 2u * ratio : ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    for (uint32_t index = 0; index < state_rows * width; ++index) {
        state_kv[index] = 0.0f;
        state_score[index] = -INFINITY;
    }
    if (ratio == RATIO4) {
        if (cutoff >= ratio) {
            set_state_rows(
                    state_kv, state_score, kv, sc, ape, width, ratio, pos0,
                    cutoff - ratio, 0u, ratio);
        }
        if (rem != 0u) {
            set_state_rows(
                    state_kv, state_score, kv, sc, ape, width, ratio, pos0,
                    cutoff, ratio, rem);
        }
    } else if (rem != 0u) {
        set_state_rows(
                state_kv, state_score, kv, sc, ape, width, ratio, pos0,
                cutoff, 0u, rem);
    }
    for (uint32_t compressed = 0; compressed < n_comp; ++compressed) {
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            float values[8];
            float scores[8];
            uint32_t candidates = 0;
            if (ratio == RATIO4) {
                if (compressed != 0u) {
                    const uint32_t prior = (compressed - 1u) * ratio;
                    for (uint32_t row = 0; row < ratio; ++row) {
                        const uint32_t token = prior + row;
                        const uint32_t phase = (pos0 + token) % ratio;
                        const uint64_t input = (uint64_t)token * width + dimension;
                        values[candidates] = kv[input];
                        scores[candidates++] =
                                sc[input] + ape[(uint64_t)phase * width + dimension];
                    }
                }
                const uint32_t current = compressed * ratio;
                for (uint32_t row = 0; row < ratio; ++row) {
                    const uint32_t token = current + row;
                    const uint32_t phase = (pos0 + token) % ratio;
                    const uint32_t second = HEAD_DIM + dimension;
                    const uint64_t input = (uint64_t)token * width + second;
                    values[candidates] = kv[input];
                    scores[candidates++] =
                            sc[input] + ape[(uint64_t)phase * width + second];
                }
            } else {
                const uint32_t current = compressed * ratio;
                for (uint32_t row = 0; row < ratio; ++row) {
                    const uint32_t token = current + row;
                    const uint32_t phase = (pos0 + token) % ratio;
                    const uint64_t input = (uint64_t)token * width + dimension;
                    values[candidates] = kv[input];
                    scores[candidates++] =
                            sc[input] + ape[(uint64_t)phase * width + dimension];
                }
            }
            float maximum = -INFINITY;
            for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
                if (scores[candidate] > maximum) maximum = scores[candidate];
            }
            float denominator = 0.0f;
            float accumulator = 0.0f;
            for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
                const float weight = expf(scores[candidate] - maximum);
                denominator += weight;
                accumulator += values[candidate] * weight;
            }
            comp[(uint64_t)compressed * HEAD_DIM + dimension] =
                    denominator != 0.0f ? accumulator / denominator : 0.0f;
        }
    }
    for (uint32_t compressed = 0; compressed < n_comp; ++compressed) {
        float sum = 0.0f;
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            const float value = comp[(uint64_t)compressed * HEAD_DIM + dimension];
            sum += value * value;
        }
        const float scale = 1.0f / sqrtf(sum / HEAD_DIM + RMS_EPS);
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            comp[(uint64_t)compressed * HEAD_DIM + dimension] *=
                    scale * norm[dimension];
        }
    }
    if (n_rot != 0u) {
        for (uint32_t compressed = 0; compressed < n_comp; ++compressed) {
            const uint32_t pos = pos0 + compressed * ratio;
            const float theta = (float)pos * FREQ_SCALE;
            const float cosine = cosf(theta) * ATTN_FACTOR;
            const float sine = sinf(theta) * ATTN_FACTOR;
            const uint64_t tail =
                    (uint64_t)compressed * HEAD_DIM + (HEAD_DIM - n_rot);
            const float x0 = comp[tail];
            const float x1 = comp[tail + 1u];
            comp[tail] = x0 * cosine - x1 * sine;
            comp[tail + 1u] = x0 * sine + x1 * cosine;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (isinf(expected[index])) {
            if (actual[index] != expected[index]) return 0;
        } else if (fabsf(actual[index] - expected[index]) > 4.0e-4f) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    float kv4[INPUT4_ELEMENTS];
    float sc4[INPUT4_ELEMENTS];
    float kv3[INPUT3_ELEMENTS];
    float sc3[INPUT3_ELEMENTS];
    float sentinel_state[STATE4_ELEMENTS];
    float comp_sentinel[COMP_ELEMENTS];
    float expected_comp[COMP_ELEMENTS];
    float expected_kv[STATE4_ELEMENTS];
    float expected_score[STATE4_ELEMENTS];
    float got_comp[COMP_ELEMENTS];
    float got_kv[STATE4_ELEMENTS];
    float got_score[STATE4_ELEMENTS];
    float ape4_f16[APE4_ELEMENTS];
    struct model4_f16 model4 = {0};
    struct model3_f32 model3 = {0};
    const uint64_t ape4_offset = offsetof(struct model4_f16, ape);
    const uint64_t norm4_offset = offsetof(struct model4_f16, norm);
    const uint64_t ape3_offset = offsetof(struct model3_f32, ape);
    const uint64_t norm3_offset = offsetof(struct model3_f32, norm);

    for (uint32_t index = 0; index < INPUT4_ELEMENTS; ++index) {
        kv4[index] = (float)((int32_t)((index * 7u + 3u) % 31u) - 15) * 0.125f;
        sc4[index] = (float)((int32_t)((index * 11u + 5u) % 37u) - 18) * 0.0625f;
    }
    for (uint32_t index = 0; index < INPUT3_ELEMENTS; ++index) {
        kv3[index] = (float)((int32_t)((index * 13u + 2u) % 29u) - 14) * 0.109375f;
        sc3[index] = (float)((int32_t)((index * 17u + 1u) % 41u) - 20) * 0.046875f;
    }
    for (uint32_t index = 0; index < STATE4_ELEMENTS; ++index) {
        sentinel_state[index] = 70.0f + (float)index;
    }
    for (uint32_t index = 0; index < COMP_ELEMENTS; ++index) {
        comp_sentinel[index] = 180.0f + (float)index;
    }
    for (uint32_t index = 0; index < APE4_ELEMENTS; ++index) {
        model4.ape[index] = (_Float16)(
                (float)((int32_t)((index * 13u + 4u) % 23u) - 11) * 0.0390625f);
        ape4_f16[index] = (float)model4.ape[index];
    }
    for (uint32_t index = 0; index < APE3_ELEMENTS; ++index) {
        model3.ape[index] =
                (float)((int32_t)((index * 5u + 1u) % 19u) - 9) * 0.03125f;
    }
    for (uint32_t index = 0; index < HEAD_DIM; ++index) {
        model4.norm[index] = 0.625f + (float)index * 0.15625f;
        model3.norm[index] = 0.75f + (float)index * 0.125f;
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(&model4, sizeof(model4))) return 1;
    ds4_gpu_tensor *kv4_tensor = ds4_gpu_tensor_alloc(sizeof(kv4));
    ds4_gpu_tensor *sc4_tensor = ds4_gpu_tensor_alloc(sizeof(sc4));
    ds4_gpu_tensor *kv3_tensor = ds4_gpu_tensor_alloc(sizeof(kv3));
    ds4_gpu_tensor *sc3_tensor = ds4_gpu_tensor_alloc(sizeof(sc3));
    ds4_gpu_tensor *state_kv = ds4_gpu_tensor_alloc(sizeof(sentinel_state));
    ds4_gpu_tensor *state_score = ds4_gpu_tensor_alloc(sizeof(sentinel_state));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(sizeof(comp_sentinel));
    ds4_gpu_tensor *expected_comp_tensor = ds4_gpu_tensor_alloc(sizeof(expected_comp));
    ds4_gpu_tensor *short_input = ds4_gpu_tensor_alloc(sizeof(kv4) - sizeof(float));
    ds4_gpu_tensor *short_state = ds4_gpu_tensor_alloc(sizeof(sentinel_state) - sizeof(float));
    ds4_gpu_tensor *short_comp = ds4_gpu_tensor_alloc(sizeof(comp_sentinel) - sizeof(float));
    if (!kv4_tensor || !sc4_tensor || !kv3_tensor || !sc3_tensor || !state_kv ||
        !state_score || !comp || !expected_comp_tensor || !short_input ||
        !short_state || !short_comp ||
        !ds4_gpu_tensor_write(kv4_tensor, 0, kv4, sizeof(kv4)) ||
        !ds4_gpu_tensor_write(sc4_tensor, 0, sc4, sizeof(sc4)) ||
        !ds4_gpu_tensor_write(kv3_tensor, 0, kv3, sizeof(kv3)) ||
        !ds4_gpu_tensor_write(sc3_tensor, 0, sc3, sizeof(sc3))) {
        return 2;
    }

    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, sentinel_state, sizeof(expected_kv));
    memcpy(expected_score, sentinel_state, sizeof(expected_score));
    reference_prefill(
            expected_comp, expected_kv, expected_score, kv4, sc4, ape4_f16,
            model4.norm, RATIO4, 1u, N_TOKENS4, N_ROT);
    if (!ds4_gpu_tensor_write(expected_comp_tensor, 0, expected_comp, sizeof(expected_comp)) ||
        !ds4_gpu_dsv4_fp8_kv_quantize_tensor(expected_comp_tensor, N_COMP4, HEAD_DIM, N_ROT) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, true, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(expected_comp_tensor, 0, expected_comp, sizeof(expected_comp)) ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE4_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE4_ELEMENTS)) {
        return 3;
    }

    if (!ds4_gpu_set_model_map(&model3, sizeof(model3))) return 4;
    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, sentinel_state, sizeof(expected_kv));
    memcpy(expected_score, sentinel_state, sizeof(expected_score));
    reference_prefill(
            expected_comp, expected_kv, expected_score, kv3, sc3, model3.ape,
            model3.norm, RATIO3, 2u, N_TOKENS3, N_ROT);
    if (!ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv3_tensor, sc3_tensor, &model3,
                sizeof(model3), ape3_offset, 0u, norm3_offset, 0u, HEAD_DIM,
                RATIO3, 2u, N_TOKENS3, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE3_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE3_ELEMENTS)) {
        return 5;
    }

    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, sentinel_state, sizeof(expected_kv));
    memcpy(expected_score, sentinel_state, sizeof(expected_score));
    reference_prefill(
            expected_comp, expected_kv, expected_score, kv3, sc3, model3.ape,
            model3.norm, RATIO3, UINT32_MAX, 2u, 0u);
    if (!ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv3_tensor, sc3_tensor, &model3,
                sizeof(model3), ape3_offset, 0u, norm3_offset, 0u, HEAD_DIM,
                RATIO3, UINT32_MAX, 2u, 0u, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE3_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE3_ELEMENTS)) {
        return 6;
    }

    memcpy(expected_comp, comp_sentinel, sizeof(expected_comp));
    memcpy(expected_kv, sentinel_state, sizeof(expected_kv));
    memcpy(expected_score, sentinel_state, sizeof(expected_score));
    reference_prefill(
            expected_comp, expected_kv, expected_score, kv3, sc3, model3.ape,
            model3.norm, RATIO3, UINT32_MAX, RATIO3, 0u);
    if (!ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv3_tensor, sc3_tensor, &model3,
                sizeof(model3), ape3_offset, 0u, norm3_offset, 0u, HEAD_DIM,
                RATIO3, UINT32_MAX, RATIO3, 0u, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE3_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE3_ELEMENTS)) {
        return 7;
    }

    if (!ds4_gpu_set_model_map(&model4, sizeof(model4)) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        !ds4_gpu_tensor_write(state_kv, 0, sentinel_state, sizeof(sentinel_state)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel_state, sizeof(sentinel_state)) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), sizeof(model4) - sizeof(_Float16), 1u,
                norm4_offset, 0u, HEAD_DIM, RATIO4, 1u, N_TOKENS4, N_ROT,
                4096u, false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, comp_sentinel, COMP_ELEMENTS) ||
        !close_array(got_kv, sentinel_state, STATE4_ELEMENTS) ||
        !close_array(got_score, sentinel_state, STATE4_ELEMENTS)) {
        return 8;
    }

    if (ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, short_input, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, short_state, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                short_comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 1u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                0u, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, 0u, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u,
                UINT32_MAX, RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false,
                FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST,
                BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                NULL, state_kv, state_score, kv4_tensor, sc4_tensor, &model4,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_tensor(
                comp, state_kv, state_score, kv4_tensor, sc4_tensor, NULL,
                sizeof(model4), ape4_offset, 1u, norm4_offset, 0u, HEAD_DIM,
                RATIO4, 1u, N_TOKENS4, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS)) {
        return 9;
    }

    ds4_gpu_tensor_free(short_comp);
    ds4_gpu_tensor_free(short_state);
    ds4_gpu_tensor_free(short_input);
    ds4_gpu_tensor_free(expected_comp_tensor);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(state_score);
    ds4_gpu_tensor_free(state_kv);
    ds4_gpu_tensor_free(sc3_tensor);
    ds4_gpu_tensor_free(kv3_tensor);
    ds4_gpu_tensor_free(sc4_tensor);
    ds4_gpu_tensor_free(kv4_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"ratio4_f16_output_and_state_matches\":true,"
         "\"ratio4_optional_fp8_composition_matches\":true,"
         "\"general_ratio_f32_output_and_remainder_matches\":true,"
         "\"stride_by_ratio_rope_matches\":true,\"no_comp_rows_state_only_matches\":true,"
         "\"n_rot_zero_success_matches\":true,\"uint32_position_wrap_matches\":true,"
         "\"invalid_model_range_preserves_state_and_output\":true,"
         "\"invalid_shape_rejected\":true,\"checked_overflow_rejected\":true,"
         "\"null_rejected\":true,\"embedded_compressor_prefill_pool_kernel_loaded\":true}");
    return 0;
}
