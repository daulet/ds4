#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define B_N_TOKENS 3u
#define B_POS0 4u
#define B_N_HEAD 2u
#define B_HEAD_DIM 4u
#define B_RAW_CAP 6u
#define B_N_RAW 4u
#define B_RAW_START 5u
#define B_N_COMP 3u
#define B_WINDOW 3u
#define B_RATIO 2u
#define B_OUTPUT_ELEMENTS (B_N_TOKENS * B_N_HEAD * B_HEAD_DIM)

#define O_N_TOKENS 2u
#define O_POS0 4u
#define O_N_HEAD 1u
#define O_HEAD_DIM 512u
#define O_RAW_CAP 2u
#define O_N_RAW 2u
#define O_RAW_START 0u
#define O_N_COMP 7937u
#define O_WINDOW 0u
#define O_RATIO 1u
#define O_OUTPUT_ELEMENTS (O_N_TOKENS * O_N_HEAD * O_HEAD_DIM)

struct batch_model {
    float prefix[3];
    float sinks[B_N_HEAD];
};

struct online_model {
    float sinks[O_N_HEAD];
};

static float dot(const float *left, const float *right, uint32_t count) {
    float result = 0.0f;
    for (uint32_t dimension = 0; dimension < count; ++dimension) {
        result += left[dimension] * right[dimension];
    }
    return result;
}

static void reference_batch(
        float *output,
        const float *sinks,
        const float *q,
        const float *raw,
        const float *comp,
        const float *mask,
        uint32_t n_comp,
        uint32_t use_mask,
        uint32_t ratio) {
    const float scale = 1.0f / sqrtf((float)B_HEAD_DIM);
    const uint32_t first_raw_pos = B_POS0 + B_N_TOKENS - B_N_RAW;
    for (uint32_t token = 0; token < B_N_TOKENS; ++token) {
        const uint32_t qpos = B_POS0 + token;
        uint32_t raw_first = 0u;
        uint32_t raw_count = 0u;
        const uint32_t raw_last_pos = first_raw_pos + B_N_RAW - 1u;
        if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (B_WINDOW != 0u && qpos + 1u > B_WINDOW) {
                const uint32_t window_lo = qpos + 1u - B_WINDOW;
                if (window_lo > lo) lo = window_lo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
        uint32_t visible_comp = n_comp ? (qpos + 1u) / ratio : 0u;
        if (visible_comp > n_comp) visible_comp = n_comp;
        for (uint32_t head = 0; head < B_N_HEAD; ++head) {
            const float *query = q + (token * B_N_HEAD + head) * B_HEAD_DIM;
            float scores[B_N_RAW + B_N_COMP];
            uint32_t rows[B_N_RAW];
            float maximum = sinks[head];
            for (uint32_t row = 0; row < raw_count; ++row) {
                rows[row] = (B_RAW_START + raw_first + row) % B_RAW_CAP;
                scores[row] = dot(query, raw + rows[row] * B_HEAD_DIM, B_HEAD_DIM) * scale;
                if (scores[row] > maximum) maximum = scores[row];
            }
            for (uint32_t compressed = 0; compressed < visible_comp; ++compressed) {
                const float add = use_mask ? mask[token * n_comp + compressed] : 0.0f;
                scores[raw_count + compressed] =
                        add > -1.0e20f
                                ? dot(query, comp + compressed * B_HEAD_DIM, B_HEAD_DIM) * scale + add
                                : -INFINITY;
                if (scores[raw_count + compressed] > maximum) {
                    maximum = scores[raw_count + compressed];
                }
            }
            float denominator = expf(sinks[head] - maximum);
            for (uint32_t row = 0; row < raw_count; ++row) {
                denominator += expf(scores[row] - maximum);
            }
            for (uint32_t compressed = 0; compressed < visible_comp; ++compressed) {
                denominator += expf(scores[raw_count + compressed] - maximum);
            }
            for (uint32_t dimension = 0; dimension < B_HEAD_DIM; ++dimension) {
                float numerator = 0.0f;
                for (uint32_t row = 0; row < raw_count; ++row) {
                    numerator += raw[rows[row] * B_HEAD_DIM + dimension] *
                            expf(scores[row] - maximum);
                }
                for (uint32_t compressed = 0; compressed < visible_comp; ++compressed) {
                    numerator += comp[compressed * B_HEAD_DIM + dimension] *
                            expf(scores[raw_count + compressed] - maximum);
                }
                output[(token * B_N_HEAD + head) * B_HEAD_DIM + dimension] =
                        numerator / denominator;
            }
        }
    }
}

static int close_array(const float *actual, const float *expected, uint64_t count, float tolerance) {
    for (uint64_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > tolerance) return 0;
    }
    return 1;
}

static int sentinel_intact(const float *values, uint64_t count, float sentinel) {
    for (uint64_t index = 0; index < count; ++index) {
        if (values[index] != sentinel) return 0;
    }
    return 1;
}

int main(void) {
    const float sentinel = 91.0f;
    struct batch_model batch_model = {{13.0f, 17.0f, 19.0f}, {-0.375f, 0.25f}};
    const uint64_t sinks_offset =
            (uint64_t)((const char *)batch_model.sinks - (const char *)&batch_model);
    float q[B_OUTPUT_ELEMENTS];
    float raw[B_RAW_CAP * B_HEAD_DIM];
    float comp[B_N_COMP * B_HEAD_DIM];
    float mask[B_N_TOKENS * B_N_COMP] = {
            -0.25f, -1.0e30f, 0.0f,
            0.125f, -0.5f, -1.0e30f,
            0.0f, 0.25f, -0.125f,
    };
    float output[B_OUTPUT_ELEMENTS];
    float expected[B_OUTPUT_ELEMENTS];
    for (uint32_t index = 0; index < B_OUTPUT_ELEMENTS; ++index) {
        q[index] = (float)((int32_t)((index * 17u + 5u) % 29u) - 14) * 0.09375f;
    }
    for (uint32_t index = 0; index < B_RAW_CAP * B_HEAD_DIM; ++index) {
        raw[index] = (float)((int32_t)((index * 23u + 3u) % 31u) - 15) * 0.109375f;
    }
    for (uint32_t index = 0; index < B_N_COMP * B_HEAD_DIM; ++index) {
        comp[index] = (float)((int32_t)((index * 19u + 1u) % 27u) - 13) * 0.125f;
    }
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(&batch_model, sizeof(batch_model))) return 1;
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(output));
    ds4_gpu_tensor *q_tensor = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *raw_tensor = ds4_gpu_tensor_alloc(sizeof(raw));
    ds4_gpu_tensor *comp_tensor = ds4_gpu_tensor_alloc(sizeof(comp));
    ds4_gpu_tensor *mask_tensor = ds4_gpu_tensor_alloc(sizeof(mask));
    ds4_gpu_tensor *short_q = ds4_gpu_tensor_alloc(sizeof(q) - sizeof(float));
    if (!heads || !q_tensor || !raw_tensor || !comp_tensor || !mask_tensor || !short_q ||
        !ds4_gpu_tensor_write(q_tensor, 0, q, sizeof(q)) ||
        !ds4_gpu_tensor_write(raw_tensor, 0, raw, sizeof(raw)) ||
        !ds4_gpu_tensor_write(comp_tensor, 0, comp, sizeof(comp)) ||
        !ds4_gpu_tensor_write(mask_tensor, 0, mask, sizeof(mask))) {
        return 2;
    }

    reference_batch(expected, batch_model.sinks, q, raw, comp, mask, B_N_COMP, 1u, B_RATIO);
    if (!ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                comp_tensor, mask_tensor, 1u, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP,
                B_RAW_START, B_N_COMP, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !close_array(output, expected, B_OUTPUT_ELEMENTS, 4.0e-4f)) {
        return 3;
    }
    reference_batch(expected, batch_model.sinks, q, raw, comp, mask, 0u, 0u, 1u);
    if (!ds4_gpu_attention_decode_raw_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START, B_WINDOW,
                B_N_HEAD, B_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !close_array(output, expected, B_OUTPUT_ELEMENTS, 4.0e-4f)) {
        return 4;
    }

    for (uint32_t index = 0; index < B_OUTPUT_ELEMENTS; ++index) output[index] = sentinel;
    if (!ds4_gpu_tensor_write(heads, 0, output, sizeof(output)) ||
        ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sizeof(batch_model), q_tensor,
                raw_tensor, comp_tensor, mask_tensor, 1u, B_N_TOKENS, B_POS0, B_N_RAW,
                B_RAW_CAP, B_RAW_START, B_N_COMP, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, short_q,
                raw_tensor, comp_tensor, mask_tensor, 1u, B_N_TOKENS, B_POS0, B_N_RAW,
                B_RAW_CAP, B_RAW_START, B_N_COMP, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor,
                raw_tensor, comp_tensor, mask_tensor, 1u, B_N_TOKENS, B_POS0, B_N_RAW,
                B_RAW_CAP, B_RAW_START, B_N_COMP, B_WINDOW, 0u, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_decode_raw_batch_heads_tensor(
                NULL, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START, B_WINDOW,
                B_N_HEAD, B_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !sentinel_intact(output, B_OUTPUT_ELEMENTS, sentinel)) {
        return 5;
    }

    struct online_model online_model = {{0.0f}};
    const uint64_t online_comp_elements = (uint64_t)O_N_COMP * O_HEAD_DIM;
    const uint64_t online_comp_bytes = online_comp_elements * sizeof(float);
    const uint64_t online_mask_elements = (uint64_t)O_N_TOKENS * O_N_COMP;
    const uint64_t online_mask_bytes = online_mask_elements * sizeof(float);
    float *online_q = (float *)calloc(O_OUTPUT_ELEMENTS, sizeof(float));
    float *online_raw = (float *)malloc((uint64_t)O_RAW_CAP * O_HEAD_DIM * sizeof(float));
    float *online_comp = (float *)malloc((size_t)online_comp_bytes);
    float *online_mask = (float *)calloc((size_t)online_mask_elements, sizeof(float));
    float online_output[O_OUTPUT_ELEMENTS];
    float online_sentinel[O_OUTPUT_ELEMENTS];
    if (!online_q || !online_raw || !online_comp || !online_mask) return 6;
    for (uint32_t index = 0; index < O_RAW_CAP * O_HEAD_DIM; ++index) online_raw[index] = 7.0f;
    for (uint64_t index = 0; index < online_comp_elements; ++index) online_comp[index] = 2.0f;
    for (uint32_t index = 0; index < O_OUTPUT_ELEMENTS; ++index) online_sentinel[index] = sentinel;
    if (!ds4_gpu_set_model_map(&online_model, sizeof(online_model))) return 7;
    ds4_gpu_tensor *online_heads = ds4_gpu_tensor_alloc(sizeof(online_output));
    ds4_gpu_tensor *online_q_tensor = ds4_gpu_tensor_alloc(sizeof(online_output));
    ds4_gpu_tensor *online_raw_tensor =
            ds4_gpu_tensor_alloc((uint64_t)O_RAW_CAP * O_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *online_comp_tensor = ds4_gpu_tensor_alloc(online_comp_bytes);
    ds4_gpu_tensor *online_mask_tensor = ds4_gpu_tensor_alloc(online_mask_bytes);
    if (!online_heads || !online_q_tensor || !online_raw_tensor || !online_comp_tensor ||
        !online_mask_tensor ||
        !ds4_gpu_tensor_write(online_q_tensor, 0, online_q, sizeof(online_output)) ||
        !ds4_gpu_tensor_write(
                online_raw_tensor, 0, online_raw,
                (uint64_t)O_RAW_CAP * O_HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(online_comp_tensor, 0, online_comp, online_comp_bytes) ||
        !ds4_gpu_tensor_write(online_mask_tensor, 0, online_mask, online_mask_bytes) ||
        !ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, NULL, 0u, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP, O_WINDOW, O_RATIO,
                O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output))) {
        return 8;
    }
    const float overflow_expected0 = 17.0f / 7.0f;
    const float overflow_expected1 = 26.0f / 9.0f;
    for (uint32_t dimension = 0; dimension < O_HEAD_DIM; ++dimension) {
        if (fabsf(online_output[dimension] - overflow_expected0) > 5.0e-4f ||
            fabsf(online_output[O_HEAD_DIM + dimension] - overflow_expected1) > 5.0e-4f) {
            return 9;
        }
    }

    if (setenv("DS4_CUDA_WINDOW_ATTENTION", "1", 1) != 0 ||
        !ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, NULL, 0u, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, 3u, 1u, O_RATIO, O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        unsetenv("DS4_CUDA_WINDOW_ATTENTION") != 0) {
        return 10;
    }
    const float window_expected = 13.0f / 5.0f;
    for (uint32_t index = 0; index < O_OUTPUT_ELEMENTS; ++index) {
        if (fabsf(online_output[index] - window_expected) > 5.0e-4f) return 11;
    }

    if (setenv("DS4_CUDA_NO_WINDOW_ATTENTION", "1", 1) != 0 ||
        !ds4_gpu_tensor_write(online_heads, 0, online_sentinel, sizeof(online_sentinel)) ||
        ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, NULL, 0u, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP, O_WINDOW, O_RATIO,
                O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        !sentinel_intact(online_output, O_OUTPUT_ELEMENTS, sentinel) ||
        unsetenv("DS4_CUDA_NO_WINDOW_ATTENTION") != 0 ||
        ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, online_mask_tensor, 1u,
                O_N_TOKENS, O_POS0, O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP,
                O_WINDOW, O_RATIO, O_N_HEAD, O_HEAD_DIM)) {
        return 12;
    }
    free(online_q);
    free(online_raw);
    free(online_comp);
    free(online_mask);
    printf("{\"c_linked_rust_staticlib\":true,\"mixed_batch_output_matches\":true,"
           "\"raw_batch_output_matches\":true,\"causal_window_matches\":true,"
           "\"visible_compressed_limit_matches\":true,\"ring_wrapped_raw_rows_match\":true,"
           "\"batched_mask_matches\":true,\"sink_softmax_matches\":true,"
           "\"overflow_online_batch_output_matches\":true,\"window_online_output_matches\":true,"
           "\"overflow_env_disable_rejected\":true,\"overflow_mask_rejected\":true,"
           "\"invalid_model_range_preserves_output\":true,\"invalid_shape_rejected\":true,"
           "\"ratio_zero_rejected\":true,\"null_rejected\":true,"
           "\"embedded_attention_decode_kernels_reused\":true}\n");
    return 0;
}
