#include "ds4_gpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HEAD_DIM 4u
#define N_ROT 2u
#define RATIO4 4u
#define WIDTH (2u * HEAD_DIM)
#define N_TOKENS 8u
#define N_COMP (N_TOKENS / RATIO4)
#define STATE_ROWS 8u
#define INPUT_ELEMENTS (N_TOKENS * WIDTH)
#define STATE_ELEMENTS (STATE_ROWS * WIDTH)
#define COMP_ELEMENTS (N_COMP * HEAD_DIM)
#define APE_ELEMENTS (RATIO4 * WIDTH)
#define POS0 0u
#define RMS_EPS 1.0e-5f
#define FREQ_BASE 100.0f
#define FREQ_SCALE 1.0f
#define EXT_FACTOR 0.0f
#define ATTN_FACTOR 1.0f
#define BETA_FAST 32.0f
#define BETA_SLOW 1.0f

struct model_f32 {
    float prefix[2];
    float ape[APE_ELEMENTS];
    float norm[HEAD_DIM];
};

struct model_f16 {
    _Float16 prefix[3];
    _Float16 ape[APE_ELEMENTS];
    float norm[HEAD_DIM];
};

static float model_ape(
        const float *ape,
        uint32_t phase,
        uint32_t dimension) {
    return ape[(uint64_t)phase * WIDTH + dimension];
}

static void reference_replay(
        float *comp,
        float *state_kv,
        float *state_score,
        const float *prior_kv,
        const float *prior_score,
        const float *kv,
        const float *sc,
        const float *ape,
        const float *norm,
        uint32_t n_rot) {
    for (uint32_t compressed = 0; compressed < N_COMP; ++compressed) {
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            float values[8];
            float scores[8];
            uint32_t candidates = 0;
            if (compressed == 0) {
                for (uint32_t row = 0; row < RATIO4; ++row) {
                    const uint64_t index = (uint64_t)row * WIDTH + dimension;
                    values[candidates] = prior_kv[index];
                    scores[candidates++] = prior_score[index];
                }
            } else {
                const uint32_t base = (compressed - 1u) * RATIO4;
                for (uint32_t row = 0; row < RATIO4; ++row) {
                    const uint32_t token = base + row;
                    const uint32_t phase = (POS0 + token) % RATIO4;
                    const uint64_t index = (uint64_t)token * WIDTH + dimension;
                    values[candidates] = kv[index];
                    scores[candidates++] =
                            sc[index] + model_ape(ape, phase, dimension);
                }
            }
            const uint32_t base = compressed * RATIO4;
            for (uint32_t row = 0; row < RATIO4; ++row) {
                const uint32_t token = base + row;
                const uint32_t phase = (POS0 + token) % RATIO4;
                const uint32_t second = HEAD_DIM + dimension;
                const uint64_t index = (uint64_t)token * WIDTH + second;
                values[candidates] = kv[index];
                scores[candidates++] = sc[index] + model_ape(ape, phase, second);
            }
            float max_score = -INFINITY;
            for (uint32_t index = 0; index < candidates; ++index) {
                if (scores[index] > max_score) max_score = scores[index];
            }
            float denominator = 0.0f;
            float accumulator = 0.0f;
            for (uint32_t index = 0; index < candidates; ++index) {
                const float weight = expf(scores[index] - max_score);
                denominator += weight;
                accumulator += values[index] * weight;
            }
            comp[(uint64_t)compressed * HEAD_DIM + dimension] =
                    denominator != 0.0f ? accumulator / denominator : 0.0f;
        }
    }
    for (uint32_t row = 0; row < N_COMP; ++row) {
        float sum = 0.0f;
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            const float value = comp[(uint64_t)row * HEAD_DIM + dimension];
            sum += value * value;
        }
        const float scale = 1.0f / sqrtf(sum / HEAD_DIM + RMS_EPS);
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            comp[(uint64_t)row * HEAD_DIM + dimension] *= scale * norm[dimension];
        }
    }
    if (n_rot != 0) {
        for (uint32_t token = 0; token < N_COMP; ++token) {
            const float theta =
                    (float)(POS0 + token * RATIO4) * powf(FREQ_BASE, 0.0f);
            const float cosine = cosf(theta);
            const float sine = sinf(theta);
            const uint64_t base = (uint64_t)token * HEAD_DIM + (HEAD_DIM - N_ROT);
            const float x0 = comp[base];
            const float x1 = comp[base + 1u];
            comp[base] = x0 * cosine - x1 * sine;
            comp[base + 1u] = x0 * sine + x1 * cosine;
        }
    }
    for (uint32_t index = 0; index < STATE_ELEMENTS; ++index) {
        state_kv[index] = 0.0f;
        state_score[index] = -INFINITY;
    }
    for (uint32_t row = 0; row < RATIO4; ++row) {
        const uint32_t token = N_TOKENS - RATIO4 + row;
        const uint32_t phase = (POS0 + token) % RATIO4;
        for (uint32_t dimension = 0; dimension < WIDTH; ++dimension) {
            const uint64_t input = (uint64_t)token * WIDTH + dimension;
            const uint64_t output = (uint64_t)row * WIDTH + dimension;
            state_kv[output] = kv[input];
            state_score[output] = sc[input] + model_ape(ape, phase, dimension);
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (isinf(expected[index])) {
            if (actual[index] != expected[index]) return 0;
        } else if (fabsf(actual[index] - expected[index]) > 3.0e-4f) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    float kv[INPUT_ELEMENTS];
    float sc[INPUT_ELEMENTS];
    float prior_kv[STATE_ELEMENTS];
    float prior_score[STATE_ELEMENTS];
    float sentinel[STATE_ELEMENTS];
    float comp_sentinel[COMP_ELEMENTS];
    float expected_comp[COMP_ELEMENTS];
    float expected_kv[STATE_ELEMENTS];
    float expected_score[STATE_ELEMENTS];
    float got_comp[COMP_ELEMENTS];
    float got_kv[STATE_ELEMENTS];
    float got_score[STATE_ELEMENTS];
    float ape_f16[APE_ELEMENTS];
    struct model_f32 model_f32 = {0};
    struct model_f16 model_f16 = {0};
    const uint64_t ape_f32_offset = offsetof(struct model_f32, ape);
    const uint64_t norm_f32_offset = offsetof(struct model_f32, norm);
    const uint64_t ape_f16_offset = offsetof(struct model_f16, ape);
    const uint64_t norm_f16_offset = offsetof(struct model_f16, norm);

    for (uint32_t index = 0; index < INPUT_ELEMENTS; ++index) {
        kv[index] = (float)((int32_t)((index * 7u + 3u) % 31u) - 15) * 0.125f;
        sc[index] = (float)((int32_t)((index * 11u + 5u) % 37u) - 18) * 0.0625f;
    }
    for (uint32_t index = 0; index < STATE_ELEMENTS; ++index) {
        prior_kv[index] = (float)((int32_t)((index * 3u + 2u) % 23u) - 11) * 0.078125f;
        prior_score[index] = (float)((int32_t)((index * 9u + 1u) % 29u) - 14) * 0.046875f;
        sentinel[index] = 80.0f + (float)index;
    }
    for (uint32_t index = 0; index < COMP_ELEMENTS; ++index) {
        comp_sentinel[index] = 180.0f + (float)index;
    }
    for (uint32_t index = 0; index < APE_ELEMENTS; ++index) {
        model_f32.ape[index] =
                (float)((int32_t)((index * 5u + 1u) % 19u) - 9) * 0.03125f;
        model_f16.ape[index] = (_Float16)(
                (float)((int32_t)((index * 13u + 4u) % 23u) - 11) * 0.0390625f);
        ape_f16[index] = (float)model_f16.ape[index];
    }
    for (uint32_t index = 0; index < HEAD_DIM; ++index) {
        model_f32.norm[index] = 0.75f + (float)index * 0.125f;
        model_f16.norm[index] = 0.625f + (float)index * 0.15625f;
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(&model_f32, sizeof(model_f32))) return 1;
    ds4_gpu_tensor *kv_tensor = ds4_gpu_tensor_alloc(sizeof(kv));
    ds4_gpu_tensor *sc_tensor = ds4_gpu_tensor_alloc(sizeof(sc));
    ds4_gpu_tensor *state_kv = ds4_gpu_tensor_alloc(sizeof(prior_kv));
    ds4_gpu_tensor *state_score = ds4_gpu_tensor_alloc(sizeof(prior_score));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(sizeof(expected_comp));
    ds4_gpu_tensor *expected_comp_tensor = ds4_gpu_tensor_alloc(sizeof(expected_comp));
    ds4_gpu_tensor *short_comp = ds4_gpu_tensor_alloc(sizeof(expected_comp) - sizeof(float));
    ds4_gpu_tensor *short_state = ds4_gpu_tensor_alloc(sizeof(prior_kv) - sizeof(float));
    ds4_gpu_tensor *short_input = ds4_gpu_tensor_alloc(sizeof(kv) - sizeof(float));
    if (!kv_tensor || !sc_tensor || !state_kv || !state_score || !comp ||
        !expected_comp_tensor || !short_comp || !short_state || !short_input ||
        !ds4_gpu_tensor_write(kv_tensor, 0, kv, sizeof(kv)) ||
        !ds4_gpu_tensor_write(sc_tensor, 0, sc, sizeof(sc))) {
        return 2;
    }

    reference_replay(
            expected_comp, expected_kv, expected_score, prior_kv, prior_score,
            kv, sc, model_f32.ape, model_f32.norm, N_ROT);
    if (!ds4_gpu_tensor_write(state_kv, 0, prior_kv, sizeof(prior_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, prior_score, sizeof(prior_score)) ||
        !ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 3;
    }

    if (!ds4_gpu_set_model_map(&model_f16, sizeof(model_f16))) return 4;
    reference_replay(
            expected_comp, expected_kv, expected_score, prior_kv, prior_score,
            kv, sc, ape_f16, model_f16.norm, N_ROT);
    if (!ds4_gpu_tensor_write(expected_comp_tensor, 0, expected_comp, sizeof(expected_comp)) ||
        !ds4_gpu_dsv4_fp8_kv_quantize_tensor(expected_comp_tensor, N_COMP, HEAD_DIM, N_ROT) ||
        !ds4_gpu_tensor_write(state_kv, 0, prior_kv, sizeof(prior_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, prior_score, sizeof(prior_score)) ||
        !ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f16,
                sizeof(model_f16), ape_f16_offset, 1u, norm_f16_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, true, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(expected_comp_tensor, 0, expected_comp, sizeof(expected_comp)) ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS) ||
        !close_array(got_kv, expected_kv, STATE_ELEMENTS) ||
        !close_array(got_score, expected_score, STATE_ELEMENTS)) {
        return 5;
    }

    if (!ds4_gpu_set_model_map(&model_f32, sizeof(model_f32))) return 6;
    reference_replay(
            expected_comp, expected_kv, expected_score, prior_kv, prior_score,
            kv, sc, model_f32.ape, model_f32.norm, 0u);
    if (!ds4_gpu_tensor_write(state_kv, 0, prior_kv, sizeof(prior_kv)) ||
        !ds4_gpu_tensor_write(state_score, 0, prior_score, sizeof(prior_score)) ||
        !ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, 0u, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !close_array(got_comp, expected_comp, COMP_ELEMENTS)) {
        return 7;
    }

    if (!ds4_gpu_tensor_write(state_kv, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_tensor_write(state_score, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_tensor_write(comp, 0, comp_sentinel, sizeof(comp_sentinel)) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), sizeof(model_f32) - sizeof(float), 0u,
                norm_f32_offset, 0u, HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u,
                false, FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                BETA_FAST, BETA_SLOW, RMS_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(comp, 0, got_comp, sizeof(got_comp)) ||
        !ds4_gpu_tensor_read(state_kv, 0, got_kv, sizeof(got_kv)) ||
        !ds4_gpu_tensor_read(state_score, 0, got_score, sizeof(got_score)) ||
        !close_array(got_comp, comp_sentinel, COMP_ELEMENTS) ||
        !close_array(got_kv, sentinel, STATE_ELEMENTS) ||
        !close_array(got_score, sentinel, STATE_ELEMENTS)) {
        return 8;
    }

    if (ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                short_comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, short_state, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, short_input, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 1u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, 1u, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS - 1u, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                UINT32_MAX, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                NULL, state_kv, state_score, kv_tensor, sc_tensor, &model_f32,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS) ||
        ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp, state_kv, state_score, kv_tensor, sc_tensor, NULL,
                sizeof(model_f32), ape_f32_offset, 0u, norm_f32_offset, 0u,
                HEAD_DIM, POS0, N_TOKENS, N_ROT, 4096u, false, FREQ_BASE,
                FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW, RMS_EPS)) {
        return 9;
    }

    ds4_gpu_tensor_free(short_input);
    ds4_gpu_tensor_free(short_state);
    ds4_gpu_tensor_free(short_comp);
    ds4_gpu_tensor_free(expected_comp_tensor);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(state_score);
    ds4_gpu_tensor_free(state_kv);
    ds4_gpu_tensor_free(sc_tensor);
    ds4_gpu_tensor_free(kv_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"ratio4_replay_f32_output_matches\":true,"
         "\"ratio4_replay_f16_output_matches\":true,\"stride4_rope_composition_matches\":true,"
         "\"optional_fp8_composition_matches\":true,\"state_rebuild_after_output_matches\":true,"
         "\"n_rot_zero_branch_matches\":true,\"invalid_model_range_preserves_output_and_state\":true,"
         "\"invalid_shape_rejected\":true,\"checked_overflow_rejected\":true,"
         "\"null_rejected\":true,\"embedded_compressor_prefill_ratio4_replay_pool_kernel_loaded\":true}");
    return 0;
}
