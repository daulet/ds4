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
#define B_N_COMP 5u
#define B_TOP_K 5u
#define B_WINDOW 3u
#define B_RATIO 2u
#define B_OUTPUT_ELEMENTS (B_N_TOKENS * B_N_HEAD * B_HEAD_DIM)

#define O_N_TOKENS 2u
#define O_POS0 0u
#define O_N_HEAD 1u
#define O_HEAD_DIM 512u
#define O_RAW_CAP 2u
#define O_N_RAW 2u
#define O_RAW_START 0u
#define O_N_COMP 512u
#define O_TOP_K 512u
#define O_WINDOW 0u
#define O_RATIO 1u
#define O_OUTPUT_ELEMENTS (O_N_TOKENS * O_N_HEAD * O_HEAD_DIM)
#define R_TOP_K 5u

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

static void reference_generic(
        float *output,
        const float *sinks,
        const float *q,
        const float *raw,
        const float *comp,
        const int32_t *topk,
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
        uint32_t visible_comp = B_N_COMP;
        if (ratio != 0u) {
            visible_comp = (qpos + 1u) / ratio;
            if (visible_comp > B_N_COMP) visible_comp = B_N_COMP;
        }
        for (uint32_t head = 0; head < B_N_HEAD; ++head) {
            const float *query = q + (token * B_N_HEAD + head) * B_HEAD_DIM;
            float scores[B_N_RAW + B_TOP_K];
            uint32_t raw_rows[B_N_RAW];
            int32_t comp_rows[B_TOP_K];
            uint32_t comp_count = 0u;
            float maximum = sinks[head];
            for (uint32_t row = 0; row < raw_count; ++row) {
                raw_rows[row] = (B_RAW_START + raw_first + row) % B_RAW_CAP;
                scores[row] = dot(query, raw + raw_rows[row] * B_HEAD_DIM, B_HEAD_DIM) * scale;
                if (scores[row] > maximum) maximum = scores[row];
            }
            for (uint32_t selected = 0; selected < B_TOP_K; ++selected) {
                const int32_t compressed = topk[token * B_TOP_K + selected];
                if (compressed >= 0 && (uint32_t)compressed < visible_comp) {
                    comp_rows[comp_count] = compressed;
                    scores[raw_count + comp_count] =
                            dot(query, comp + (uint32_t)compressed * B_HEAD_DIM, B_HEAD_DIM) * scale;
                    if (scores[raw_count + comp_count] > maximum) {
                        maximum = scores[raw_count + comp_count];
                    }
                    ++comp_count;
                }
            }
            float denominator = expf(sinks[head] - maximum);
            for (uint32_t row = 0; row < raw_count + comp_count; ++row) {
                denominator += expf(scores[row] - maximum);
            }
            for (uint32_t dimension = 0; dimension < B_HEAD_DIM; ++dimension) {
                float numerator = 0.0f;
                for (uint32_t row = 0; row < raw_count; ++row) {
                    numerator += raw[raw_rows[row] * B_HEAD_DIM + dimension] *
                            expf(scores[row] - maximum);
                }
                for (uint32_t selected = 0; selected < comp_count; ++selected) {
                    numerator += comp[(uint32_t)comp_rows[selected] * B_HEAD_DIM + dimension] *
                            expf(scores[raw_count + selected] - maximum);
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

static int constant_rows_match(
        const float *output,
        float token0,
        float token1,
        float tolerance) {
    for (uint32_t dimension = 0; dimension < O_HEAD_DIM; ++dimension) {
        if (fabsf(output[dimension] - token0) > tolerance ||
            fabsf(output[O_HEAD_DIM + dimension] - token1) > tolerance) {
            return 0;
        }
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
    const int32_t topk[B_N_TOKENS * B_TOP_K] = {
            1, 0, -1, 4, 1,
            2, -1, 0, 4, 1,
            4, 1, 2, -1, 0,
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
    ds4_gpu_tensor *topk_tensor = ds4_gpu_tensor_alloc(sizeof(topk));
    ds4_gpu_tensor *short_q = ds4_gpu_tensor_alloc(sizeof(q) - sizeof(float));
    if (!heads || !q_tensor || !raw_tensor || !comp_tensor || !topk_tensor || !short_q ||
        !ds4_gpu_tensor_write(q_tensor, 0, q, sizeof(q)) ||
        !ds4_gpu_tensor_write(raw_tensor, 0, raw, sizeof(raw)) ||
        !ds4_gpu_tensor_write(comp_tensor, 0, comp, sizeof(comp)) ||
        !ds4_gpu_tensor_write(topk_tensor, 0, topk, sizeof(topk))) {
        return 2;
    }

    reference_generic(expected, batch_model.sinks, q, raw, comp, topk, B_RATIO);
    if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START,
                B_N_COMP, B_TOP_K, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !close_array(output, expected, B_OUTPUT_ELEMENTS, 4.0e-4f)) {
        return 3;
    }
    reference_generic(expected, batch_model.sinks, q, raw, comp, topk, 0u);
    if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START,
                B_N_COMP, B_TOP_K, B_WINDOW, 0u, B_N_HEAD, B_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !close_array(output, expected, B_OUTPUT_ELEMENTS, 4.0e-4f)) {
        return 4;
    }

    for (uint32_t index = 0; index < B_OUTPUT_ELEMENTS; ++index) output[index] = sentinel;
    if (!ds4_gpu_tensor_write(heads, 0, output, sizeof(output)) ||
        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sizeof(batch_model), q_tensor,
                raw_tensor, comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP,
                B_RAW_START, B_N_COMP, B_TOP_K, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, short_q, raw_tensor,
                comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START,
                B_N_COMP, B_TOP_K, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START,
                0u, B_TOP_K, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START,
                B_N_COMP, 513u, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                NULL, &batch_model, sizeof(batch_model), sinks_offset, q_tensor, raw_tensor,
                comp_tensor, topk_tensor, B_N_TOKENS, B_POS0, B_N_RAW, B_RAW_CAP, B_RAW_START,
                B_N_COMP, B_TOP_K, B_WINDOW, B_RATIO, B_N_HEAD, B_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !sentinel_intact(output, B_OUTPUT_ELEMENTS, sentinel)) {
        return 5;
    }

    struct online_model online_model = {{0.0f}};
    float *online_q = (float *)calloc(O_OUTPUT_ELEMENTS, sizeof(float));
    float *online_raw = (float *)malloc((uint64_t)O_RAW_CAP * O_HEAD_DIM * sizeof(float));
    float *online_comp = (float *)malloc((uint64_t)O_N_COMP * O_HEAD_DIM * sizeof(float));
    int32_t *online_topk = (int32_t *)malloc((uint64_t)O_N_TOKENS * O_TOP_K * sizeof(int32_t));
    float online_output[O_OUTPUT_ELEMENTS];
    const int32_t rb4_topk[O_N_TOKENS * R_TOP_K] = {
            0, -1, 1, 0, 3,
            1, 0, 1, -1, 3,
    };
    if (!online_q || !online_raw || !online_comp || !online_topk) return 6;
    for (uint32_t index = 0; index < O_RAW_CAP * O_HEAD_DIM; ++index) online_raw[index] = 7.0f;
    for (uint32_t row = 0; row < O_N_COMP; ++row) {
        for (uint32_t dimension = 0; dimension < O_HEAD_DIM; ++dimension) {
            online_comp[row * O_HEAD_DIM + dimension] = (float)(row + 1u);
        }
    }
    for (uint32_t token = 0; token < O_N_TOKENS; ++token) {
        for (uint32_t selected = 0; selected < O_TOP_K; ++selected) {
            online_topk[token * O_TOP_K + selected] = (int32_t)(O_TOP_K - selected - 1u);
        }
    }
    if (!ds4_gpu_set_model_map(&online_model, sizeof(online_model))) return 7;
    ds4_gpu_tensor *online_heads = ds4_gpu_tensor_alloc(sizeof(online_output));
    ds4_gpu_tensor *online_q_tensor = ds4_gpu_tensor_alloc(sizeof(online_output));
    ds4_gpu_tensor *online_raw_tensor =
            ds4_gpu_tensor_alloc((uint64_t)O_RAW_CAP * O_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *online_comp_tensor =
            ds4_gpu_tensor_alloc((uint64_t)O_N_COMP * O_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *online_topk_tensor =
            ds4_gpu_tensor_alloc((uint64_t)O_N_TOKENS * O_TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *rb4_topk_tensor = ds4_gpu_tensor_alloc(sizeof(rb4_topk));
    if (!online_heads || !online_q_tensor || !online_raw_tensor || !online_comp_tensor ||
        !online_topk_tensor || !rb4_topk_tensor ||
        !ds4_gpu_tensor_write(online_q_tensor, 0, online_q, sizeof(online_output)) ||
        !ds4_gpu_tensor_write(
                online_raw_tensor, 0, online_raw,
                (uint64_t)O_RAW_CAP * O_HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(
                online_comp_tensor, 0, online_comp,
                (uint64_t)O_N_COMP * O_HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(
                online_topk_tensor, 0, online_topk,
                (uint64_t)O_N_TOKENS * O_TOP_K * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(rb4_topk_tensor, 0, rb4_topk, sizeof(rb4_topk))) {
        return 8;
    }

    if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, online_topk_tensor, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP, O_TOP_K, O_WINDOW, O_RATIO,
                O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        !constant_rows_match(online_output, 8.0f / 3.0f, 17.0f / 5.0f, 5.0e-4f)) {
        return 9;
    }
    if (setenv("DS4_CUDA_NO_INDEXED_TOPK_SORT", "1", 1) != 0 ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, online_topk_tensor, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP, O_TOP_K, O_WINDOW, O_RATIO,
                O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        !constant_rows_match(online_output, 519.0f / 3.0f, 1037.0f / 5.0f, 5.0e-4f) ||
        unsetenv("DS4_CUDA_NO_INDEXED_TOPK_SORT") != 0) {
        return 10;
    }
    if (setenv("DS4_CUDA_INDEXED_TWOPASS", "1", 1) != 0 ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, rb4_topk_tensor, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP, R_TOP_K, O_WINDOW, O_RATIO,
                O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        !constant_rows_match(online_output, 9.0f / 4.0f, 19.0f / 6.0f, 5.0e-4f) ||
        unsetenv("DS4_CUDA_INDEXED_TWOPASS") != 0) {
        return 11;
    }
    if (setenv("DS4_CUDA_NO_INDEXED_HEADS8", "1", 1) != 0 ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, online_comp_tensor, rb4_topk_tensor, O_N_TOKENS, O_POS0,
                O_N_RAW, O_RAW_CAP, O_RAW_START, O_N_COMP, R_TOP_K, O_WINDOW, 0u,
                O_N_HEAD, O_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        !constant_rows_match(online_output, 15.0f / 6.0f, 23.0f / 7.0f, 5.0e-4f) ||
        unsetenv("DS4_CUDA_NO_INDEXED_HEADS8") != 0) {
        return 12;
    }
    free(online_q);
    free(online_raw);
    free(online_comp);
    free(online_topk);
    printf("{\"c_linked_rust_staticlib\":true,\"generic_indexed_output_matches\":true,"
           "\"ratio_zero_all_compressed_matches\":true,\"topk_filter_order_and_duplicates_match\":true,"
           "\"causal_window_matches\":true,\"ring_wrapped_raw_rows_match\":true,"
           "\"sink_softmax_matches\":true,\"sorted_online_output_matches\":true,"
           "\"sort_disable_gate_matches\":true,\"rb4_filtered_output_matches\":true,"
           "\"forced_generic_environment_gate_matches\":true,"
           "\"invalid_model_range_preserves_output\":true,\"invalid_shape_rejected\":true,"
           "\"null_rejected\":true,\"embedded_indexed_attention_kernels_loaded\":true}\n");
    return 0;
}
