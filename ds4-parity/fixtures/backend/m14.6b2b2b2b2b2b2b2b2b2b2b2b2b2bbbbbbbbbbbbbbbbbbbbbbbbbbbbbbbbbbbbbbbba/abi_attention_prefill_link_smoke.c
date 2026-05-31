#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define G_TOKENS 4u
#define G_HEADS 2u
#define G_DIM 7u
#define G_COMP 2u
#define G_WINDOW 3u
#define G_RATIO 2u
#define G_OUT ((uint64_t)G_TOKENS * G_HEADS * G_DIM)

#define O_TOKENS 3u
#define O_HEADS 2u
#define O_DIM 512u
#define O_COMP 2u
#define O_WINDOW 2u
#define O_RATIO 2u
#define O_OUT ((uint64_t)O_TOKENS * O_HEADS * O_DIM)

struct generic_model {
    float prefix[3];
    float sinks[G_HEADS];
};

struct optimized_model {
    float sinks[O_HEADS];
};

static float dot(const float *left, const float *right, uint32_t count) {
    float total = 0.0f;
    for (uint32_t i = 0; i < count; ++i) total += left[i] * right[i];
    return total;
}

static void reference_prefill(
        float *out,
        const float *sinks,
        const float *q,
        const float *raw,
        const float *comp,
        const float *mask,
        int raw_api,
        int use_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t token = 0; token < n_tokens; ++token) {
        uint32_t raw_start;
        uint32_t raw_count;
        if (raw_api) {
            raw_count = token + 1u < window ? token + 1u : window;
            raw_start = token + 1u - raw_count;
        } else {
            raw_start = window != 0u && token + 1u > window ? token + 1u - window : 0u;
            raw_count = token + 1u - raw_start;
        }
        uint32_t visible_comp = raw_api ? 0u : (token + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
        for (uint32_t head = 0; head < n_head; ++head) {
            const float *query = q + ((uint64_t)token * n_head + head) * head_dim;
            float maximum = sinks[head];
            for (uint32_t row = 0; row < raw_count; ++row) {
                const float score = dot(query, raw + (raw_start + row) * head_dim, head_dim) * scale;
                if (score > maximum) maximum = score;
            }
            for (uint32_t compressed = 0; compressed < visible_comp; ++compressed) {
                const float add = use_mask ? mask[token * n_comp + compressed] : 0.0f;
                if (add > -1.0e20f) {
                    const float score = dot(query, comp + compressed * head_dim, head_dim) * scale + add;
                    if (score > maximum) maximum = score;
                }
            }
            float denominator = expf(sinks[head] - maximum);
            for (uint32_t row = 0; row < raw_count; ++row) {
                denominator += expf(dot(query, raw + (raw_start + row) * head_dim, head_dim) * scale - maximum);
            }
            for (uint32_t compressed = 0; compressed < visible_comp; ++compressed) {
                const float add = use_mask ? mask[token * n_comp + compressed] : 0.0f;
                if (add > -1.0e20f) {
                    denominator += expf(dot(query, comp + compressed * head_dim, head_dim) * scale + add - maximum);
                }
            }
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                float numerator = 0.0f;
                for (uint32_t row = 0; row < raw_count; ++row) {
                    numerator += raw[(raw_start + row) * head_dim + dimension] *
                            expf(dot(query, raw + (raw_start + row) * head_dim, head_dim) * scale - maximum);
                }
                for (uint32_t compressed = 0; compressed < visible_comp; ++compressed) {
                    const float add = use_mask ? mask[token * n_comp + compressed] : 0.0f;
                    if (add > -1.0e20f) {
                        numerator += comp[compressed * head_dim + dimension] *
                                expf(dot(query, comp + compressed * head_dim, head_dim) * scale + add - maximum);
                    }
                }
                out[((uint64_t)token * n_head + head) * head_dim + dimension] =
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

static int zeros(const float *values, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (values[index] != 0.0f) return 0;
    }
    return 1;
}

static int sentinel_intact(const float *values, uint64_t count, float sentinel) {
    for (uint64_t index = 0; index < count; ++index) {
        if (values[index] != sentinel) return 0;
    }
    return 1;
}

static int set_generic_path(void) {
    return setenv("DS4_CUDA_NO_WINDOW_ATTENTION", "1", 1) == 0 &&
           setenv("DS4_CUDA_NO_CUBLAS_ATTENTION", "1", 1) == 0 &&
           setenv("DS4_CUDA_NO_TF32", "1", 1) == 0 &&
           unsetenv("DS4_CUDA_WINDOW_ATTENTION") == 0;
}

static int set_optimized_path(int online) {
    return setenv("DS4_CUDA_NO_TF32", "1", 1) == 0 &&
           unsetenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == 0 &&
           (online ? unsetenv("DS4_CUDA_NO_WINDOW_ATTENTION")
                   : setenv("DS4_CUDA_NO_WINDOW_ATTENTION", "1", 1)) == 0 &&
           (online ? setenv("DS4_CUDA_WINDOW_ATTENTION", "1", 1)
                   : unsetenv("DS4_CUDA_WINDOW_ATTENTION")) == 0;
}

int main(void) {
    struct generic_model generic_model = {{3.0f, 5.0f, 7.0f}, {-0.25f, 0.375f}};
    const uint64_t generic_sinks_offset = offsetof(struct generic_model, sinks);
    float g_q[G_OUT];
    float g_raw[G_TOKENS * G_DIM];
    float g_comp[G_COMP * G_DIM];
    float g_mask[G_TOKENS * G_COMP] = {
        0.0f, -1.0e30f, 0.125f, -1.0e30f, -0.25f, 0.25f, 0.0f, -1.0e30f
    };
    float g_actual[G_OUT] = {0};
    float g_expected[G_OUT] = {0};
    for (uint32_t index = 0; index < G_OUT; ++index) {
        g_q[index] = ((int32_t)((index * 17u + 5u) % 29u) - 14) * 0.09375f;
    }
    for (uint32_t index = 0; index < G_TOKENS * G_DIM; ++index) {
        g_raw[index] = ((int32_t)((index * 23u + 3u) % 31u) - 15) * 0.109375f;
    }
    for (uint32_t index = 0; index < G_COMP * G_DIM; ++index) {
        g_comp[index] = ((int32_t)((index * 19u + 1u) % 27u) - 13) * 0.125f;
    }
    if (!set_generic_path() || !ds4_gpu_init() ||
        !ds4_gpu_set_model_map(&generic_model, sizeof(generic_model))) return 1;
    ds4_gpu_tensor *g_heads = ds4_gpu_tensor_alloc(sizeof(g_actual));
    ds4_gpu_tensor *g_q_tensor = ds4_gpu_tensor_alloc(sizeof(g_q));
    ds4_gpu_tensor *g_raw_tensor = ds4_gpu_tensor_alloc(sizeof(g_raw));
    ds4_gpu_tensor *g_comp_tensor = ds4_gpu_tensor_alloc(sizeof(g_comp));
    ds4_gpu_tensor *g_mask_tensor = ds4_gpu_tensor_alloc(sizeof(g_mask));
    ds4_gpu_tensor *g_short_q = ds4_gpu_tensor_alloc(sizeof(g_q) - sizeof(float));
    if (!g_heads || !g_q_tensor || !g_raw_tensor || !g_comp_tensor || !g_mask_tensor || !g_short_q ||
        !ds4_gpu_tensor_write(g_q_tensor, 0, g_q, sizeof(g_q)) ||
        !ds4_gpu_tensor_write(g_raw_tensor, 0, g_raw, sizeof(g_raw)) ||
        !ds4_gpu_tensor_write(g_comp_tensor, 0, g_comp, sizeof(g_comp)) ||
        !ds4_gpu_tensor_write(g_mask_tensor, 0, g_mask, sizeof(g_mask))) return 2;

    reference_prefill(g_expected, generic_model.sinks, g_q, g_raw, g_raw, g_mask, 1, 0,
                      G_TOKENS, 0u, G_WINDOW, 1u, G_HEADS, G_DIM);
    if (!ds4_gpu_attention_prefill_raw_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset, g_q_tensor,
                g_raw_tensor, G_TOKENS, G_WINDOW, G_HEADS, G_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(g_heads, 0, g_actual, sizeof(g_actual)) ||
        !close_array(g_actual, g_expected, G_OUT, 3.0e-5f)) return 3;

    if (!ds4_gpu_attention_prefill_raw_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset, g_q_tensor,
                g_raw_tensor, G_TOKENS, 0u, G_HEADS, G_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(g_heads, 0, g_actual, sizeof(g_actual)) ||
        !zeros(g_actual, G_OUT)) return 4;

    reference_prefill(g_expected, generic_model.sinks, g_q, g_raw, g_comp, g_mask, 0, 0,
                      G_TOKENS, G_COMP, 0u, G_RATIO, G_HEADS, G_DIM);
    if (!ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset, g_q_tensor,
                g_raw_tensor, g_comp_tensor, G_TOKENS, G_COMP, 0u, G_RATIO, G_HEADS, G_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(g_heads, 0, g_actual, sizeof(g_actual)) ||
        !close_array(g_actual, g_expected, G_OUT, 3.0e-5f)) return 5;

    reference_prefill(g_expected, generic_model.sinks, g_q, g_raw, g_comp, g_mask, 0, 1,
                      G_TOKENS, G_COMP, G_WINDOW, G_RATIO, G_HEADS, G_DIM);
    if (!ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset, g_q_tensor,
                g_raw_tensor, g_comp_tensor, g_mask_tensor, G_TOKENS, G_COMP, G_WINDOW, G_RATIO,
                G_HEADS, G_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(g_heads, 0, g_actual, sizeof(g_actual)) ||
        !close_array(g_actual, g_expected, G_OUT, 3.0e-5f)) return 6;

    const float sentinel = 91.0f;
    for (uint64_t index = 0; index < G_OUT; ++index) g_actual[index] = sentinel;
    if (!ds4_gpu_tensor_write(g_heads, 0, g_actual, sizeof(g_actual)) ||
        ds4_gpu_attention_prefill_raw_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model) - 1u, generic_sinks_offset,
                g_q_tensor, g_raw_tensor, G_TOKENS, G_WINDOW, G_HEADS, G_DIM) ||
        ds4_gpu_attention_prefill_raw_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset,
                g_short_q, g_raw_tensor, G_TOKENS, G_WINDOW, G_HEADS, G_DIM) ||
        ds4_gpu_attention_prefill_raw_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset,
                g_q_tensor, g_raw_tensor, G_TOKENS, 257u, G_HEADS, G_DIM) ||
        ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                g_heads, &generic_model, sizeof(generic_model), generic_sinks_offset, g_q_tensor,
                g_raw_tensor, g_comp_tensor, G_TOKENS, G_COMP, G_WINDOW, 0u, G_HEADS, G_DIM) ||
        ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
                NULL, &generic_model, sizeof(generic_model), generic_sinks_offset, g_q_tensor,
                g_raw_tensor, g_comp_tensor, g_mask_tensor, G_TOKENS, G_COMP, G_WINDOW, G_RATIO,
                G_HEADS, G_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(g_heads, 0, g_actual, sizeof(g_actual)) ||
        !sentinel_intact(g_actual, G_OUT, sentinel)) return 7;

    struct optimized_model optimized_model = {{0.0f, 0.0f}};
    float *o_q = (float *)calloc(O_OUT, sizeof(float));
    float *o_raw = (float *)malloc((uint64_t)O_TOKENS * O_DIM * sizeof(float));
    float *o_comp = (float *)malloc((uint64_t)O_COMP * O_DIM * sizeof(float));
    float *o_mask = (float *)calloc((uint64_t)O_TOKENS * O_COMP, sizeof(float));
    float *o_actual = (float *)calloc(O_OUT, sizeof(float));
    float *o_expected = (float *)calloc(O_OUT, sizeof(float));
    if (!o_q || !o_raw || !o_comp || !o_mask || !o_actual || !o_expected) return 8;
    for (uint32_t d = 0; d < O_DIM; ++d) {
        o_raw[d] = 1.0f;
        o_raw[O_DIM + d] = 2.0f;
        o_raw[2u * O_DIM + d] = 4.0f;
        o_comp[d] = 8.0f;
        o_comp[O_DIM + d] = 16.0f;
    }
    o_mask[O_COMP] = -1.0e30f;
    if (!ds4_gpu_set_model_map(&optimized_model, sizeof(optimized_model))) return 9;
    ds4_gpu_tensor *o_heads = ds4_gpu_tensor_alloc(sizeof(float) * O_OUT);
    ds4_gpu_tensor *o_q_tensor = ds4_gpu_tensor_alloc(sizeof(float) * O_OUT);
    ds4_gpu_tensor *o_raw_tensor = ds4_gpu_tensor_alloc((uint64_t)O_TOKENS * O_DIM * sizeof(float));
    ds4_gpu_tensor *o_comp_tensor = ds4_gpu_tensor_alloc((uint64_t)O_COMP * O_DIM * sizeof(float));
    ds4_gpu_tensor *o_mask_tensor = ds4_gpu_tensor_alloc((uint64_t)O_TOKENS * O_COMP * sizeof(float));
    if (!o_heads || !o_q_tensor || !o_raw_tensor || !o_comp_tensor || !o_mask_tensor ||
        !ds4_gpu_tensor_write(o_q_tensor, 0, o_q, sizeof(float) * O_OUT) ||
        !ds4_gpu_tensor_write(o_raw_tensor, 0, o_raw, (uint64_t)O_TOKENS * O_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(o_comp_tensor, 0, o_comp, (uint64_t)O_COMP * O_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(o_mask_tensor, 0, o_mask, (uint64_t)O_TOKENS * O_COMP * sizeof(float))) return 10;

    reference_prefill(o_expected, optimized_model.sinks, o_q, o_raw, o_comp, o_mask, 0, 0,
                      O_TOKENS, O_COMP, O_WINDOW, O_RATIO, O_HEADS, O_DIM);
    if (!set_optimized_path(1) ||
        !ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                o_heads, &optimized_model, sizeof(optimized_model), 0u, o_q_tensor, o_raw_tensor,
                o_comp_tensor, O_TOKENS, O_COMP, O_WINDOW, O_RATIO, O_HEADS, O_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(o_heads, 0, o_actual, sizeof(float) * O_OUT) ||
        !close_array(o_actual, o_expected, O_OUT, 2.0e-4f)) return 11;

    if (!set_optimized_path(0) ||
        !ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                o_heads, &optimized_model, sizeof(optimized_model), 0u, o_q_tensor, o_raw_tensor,
                o_comp_tensor, O_TOKENS, O_COMP, O_WINDOW, O_RATIO, O_HEADS, O_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(o_heads, 0, o_actual, sizeof(float) * O_OUT) ||
        !close_array(o_actual, o_expected, O_OUT, 2.0e-4f)) return 12;

    reference_prefill(o_expected, optimized_model.sinks, o_q, o_raw, o_comp, o_mask, 0, 1,
                      O_TOKENS, O_COMP, O_WINDOW, O_RATIO, O_HEADS, O_DIM);
    if (!set_optimized_path(1) ||
        !ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
                o_heads, &optimized_model, sizeof(optimized_model), 0u, o_q_tensor, o_raw_tensor,
                o_comp_tensor, o_mask_tensor, O_TOKENS, O_COMP, O_WINDOW, O_RATIO, O_HEADS, O_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(o_heads, 0, o_actual, sizeof(float) * O_OUT) ||
        !close_array(o_actual, o_expected, O_OUT, 2.0e-4f)) return 13;

    if (!set_optimized_path(0) ||
        !ds4_gpu_attention_prefill_raw_heads_tensor(
                o_heads, &optimized_model, sizeof(optimized_model), 0u, o_q_tensor, o_raw_tensor,
                O_TOKENS, 0u, O_HEADS, O_DIM) ||
        !ds4_gpu_synchronize() || !ds4_gpu_tensor_read(o_heads, 0, o_actual, sizeof(float) * O_OUT)) return 14;
    reference_prefill(o_expected, optimized_model.sinks, o_q, o_raw, o_raw, o_mask, 0, 0,
                      O_TOKENS, 0u, 0u, 1u, O_HEADS, O_DIM);
    if (!close_array(o_actual, o_expected, O_OUT, 2.0e-4f) || zeros(o_actual, O_OUT)) return 15;

    ds4_gpu_tensor_free(o_mask_tensor);
    ds4_gpu_tensor_free(o_comp_tensor);
    ds4_gpu_tensor_free(o_raw_tensor);
    ds4_gpu_tensor_free(o_q_tensor);
    ds4_gpu_tensor_free(o_heads);
    ds4_gpu_tensor_free(g_short_q);
    ds4_gpu_tensor_free(g_mask_tensor);
    ds4_gpu_tensor_free(g_comp_tensor);
    ds4_gpu_tensor_free(g_raw_tensor);
    ds4_gpu_tensor_free(g_q_tensor);
    ds4_gpu_tensor_free(g_heads);
    ds4_gpu_cleanup();
    free(o_expected);
    free(o_actual);
    free(o_mask);
    free(o_comp);
    free(o_raw);
    free(o_q);
    puts("{\"c_linked_rust_staticlib\":true,\"generic_raw_output_matches\":true,"
         "\"raw_zero_window_generic_sink_only_matches\":true,"
         "\"static_mixed_zero_window_matches\":true,\"masked_mixed_output_matches\":true,"
         "\"online_static_output_matches\":true,\"cublas_static_output_matches\":true,"
         "\"masked_cublas_output_matches\":true,"
         "\"raw_zero_window_optimized_full_causal_matches\":true,"
         "\"invalid_range_preserves_output\":true,\"invalid_shape_rejected\":true,"
         "\"null_rejected\":true,\"embedded_prefill_kernels_loaded\":true}");
    return 0;
}
