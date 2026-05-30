#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_HEAD 2u
#define HEAD_DIM 4u
#define RAW_CAP 5u
#define N_RAW 3u
#define RAW_START 4u
#define N_COMP 2u
#define ONLINE_HEADS 1u
#define ONLINE_HEAD_DIM 512u
#define ONLINE_RAW_CAP 3u
#define ONLINE_N_RAW 2u
#define ONLINE_N_COMP 7937u

struct generic_model {
    float prefix[3];
    float sinks[N_HEAD];
};

struct online_model {
    float sinks[ONLINE_HEADS];
};

static float dot(const float *left, const float *right, uint32_t count) {
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index) sum += left[index] * right[index];
    return sum;
}

static void reference_generic(
        float *output,
        const float *sinks,
        const float *q,
        const float *raw,
        const float *comp,
        const float *mask,
        uint32_t n_comp,
        uint32_t use_mask) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t head = 0; head < N_HEAD; ++head) {
        const float *query = q + head * HEAD_DIM;
        float scores[N_RAW + N_COMP];
        uint32_t rows[N_RAW];
        float maximum = sinks[head];
        for (uint32_t raw_row = 0; raw_row < N_RAW; ++raw_row) {
            rows[raw_row] = (RAW_START + raw_row) % RAW_CAP;
            scores[raw_row] = dot(query, raw + rows[raw_row] * HEAD_DIM, HEAD_DIM) * scale;
            if (scores[raw_row] > maximum) maximum = scores[raw_row];
        }
        for (uint32_t compressed = 0; compressed < n_comp; ++compressed) {
            const float add = use_mask ? mask[compressed] : 0.0f;
            scores[N_RAW + compressed] =
                    add > -1.0e20f
                            ? dot(query, comp + compressed * HEAD_DIM, HEAD_DIM) * scale + add
                            : -INFINITY;
            if (scores[N_RAW + compressed] > maximum) maximum = scores[N_RAW + compressed];
        }
        float denominator = expf(sinks[head] - maximum);
        for (uint32_t raw_row = 0; raw_row < N_RAW; ++raw_row) {
            denominator += expf(scores[raw_row] - maximum);
        }
        for (uint32_t compressed = 0; compressed < n_comp; ++compressed) {
            denominator += expf(scores[N_RAW + compressed] - maximum);
        }
        for (uint32_t dimension = 0; dimension < HEAD_DIM; ++dimension) {
            float numerator = 0.0f;
            for (uint32_t raw_row = 0; raw_row < N_RAW; ++raw_row) {
                numerator += raw[rows[raw_row] * HEAD_DIM + dimension] *
                        expf(scores[raw_row] - maximum);
            }
            for (uint32_t compressed = 0; compressed < n_comp; ++compressed) {
                numerator += comp[compressed * HEAD_DIM + dimension] *
                        expf(scores[N_RAW + compressed] - maximum);
            }
            output[head * HEAD_DIM + dimension] = numerator / denominator;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count, float tolerance) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > tolerance) return 0;
    }
    return 1;
}

static int sentinel_intact(const float *values, uint32_t count, float sentinel) {
    for (uint32_t index = 0; index < count; ++index) {
        if (values[index] != sentinel) return 0;
    }
    return 1;
}

int main(void) {
    struct generic_model generic_model = {{17.0f, 18.0f, 19.0f}, {-0.375f, 0.25f}};
    float q[N_HEAD * HEAD_DIM];
    float raw[RAW_CAP * HEAD_DIM];
    float comp[N_COMP * HEAD_DIM];
    float mask[N_COMP] = {-0.125f, -1.0e30f};
    float expected[N_HEAD * HEAD_DIM];
    float output[N_HEAD * HEAD_DIM];
    const float sentinel = 91.0f;
    const uint64_t sinks_offset = (uint64_t)((const char *)generic_model.sinks -
                                               (const char *)&generic_model);
    for (uint32_t index = 0; index < N_HEAD * HEAD_DIM; ++index) {
        q[index] = (float)((int32_t)((index * 13u + 2u) % 17u) - 8) * 0.125f;
    }
    for (uint32_t index = 0; index < RAW_CAP * HEAD_DIM; ++index) {
        raw[index] = (float)((int32_t)((index * 7u + 3u) % 23u) - 11) * 0.15625f;
    }
    for (uint32_t index = 0; index < N_COMP * HEAD_DIM; ++index) {
        comp[index] = (float)((int32_t)((index * 11u + 1u) % 19u) - 9) * 0.1875f;
    }
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(&generic_model, sizeof(generic_model))) return 1;
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

    reference_generic(expected, generic_model.sinks, q, raw, comp, mask, N_COMP, 1u);
    if (!ds4_gpu_attention_decode_heads_tensor(
                heads, &generic_model, sizeof(generic_model), sinks_offset, q_tensor,
                raw_tensor, N_RAW, RAW_CAP, RAW_START, comp_tensor, N_COMP, mask_tensor,
                1u, N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !close_array(output, expected, N_HEAD * HEAD_DIM, 4.0e-4f)) {
        return 3;
    }

    reference_generic(expected, generic_model.sinks, q, raw, comp, mask, 0u, 0u);
    if (!ds4_gpu_attention_decode_heads_tensor(
                heads, &generic_model, sizeof(generic_model), sinks_offset, q_tensor,
                raw_tensor, N_RAW, RAW_CAP, RAW_START, NULL, 0u, NULL, 0u,
                N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !close_array(output, expected, N_HEAD * HEAD_DIM, 4.0e-4f)) {
        return 4;
    }

    for (uint32_t index = 0; index < N_HEAD * HEAD_DIM; ++index) output[index] = sentinel;
    if (!ds4_gpu_tensor_write(heads, 0, output, sizeof(output)) ||
        ds4_gpu_attention_decode_heads_tensor(
                heads, &generic_model, sizeof(generic_model), sizeof(generic_model), q_tensor,
                raw_tensor, N_RAW, RAW_CAP, RAW_START, comp_tensor, N_COMP, mask_tensor,
                1u, N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, output, sizeof(output)) ||
        !sentinel_intact(output, N_HEAD * HEAD_DIM, sentinel) ||
        ds4_gpu_attention_decode_heads_tensor(
                heads, &generic_model, sizeof(generic_model), sinks_offset, short_q,
                raw_tensor, N_RAW, RAW_CAP, RAW_START, comp_tensor, N_COMP, mask_tensor,
                1u, N_HEAD, HEAD_DIM) ||
        ds4_gpu_attention_decode_heads_tensor(
                NULL, &generic_model, sizeof(generic_model), sinks_offset, q_tensor,
                raw_tensor, N_RAW, RAW_CAP, RAW_START, comp_tensor, N_COMP, mask_tensor,
                1u, N_HEAD, HEAD_DIM)) {
        return 5;
    }

    struct online_model online_model = {{0.0f}};
    const uint64_t online_elements = (uint64_t)ONLINE_N_COMP * ONLINE_HEAD_DIM;
    const uint64_t online_bytes = online_elements * sizeof(float);
    float *online_q = (float *)calloc(ONLINE_HEAD_DIM, sizeof(float));
    float *online_raw = (float *)malloc((size_t)ONLINE_RAW_CAP * ONLINE_HEAD_DIM * sizeof(float));
    float *online_comp = (float *)malloc((size_t)online_bytes);
    float online_output[ONLINE_HEAD_DIM];
    float online_sentinel[ONLINE_HEAD_DIM];
    if (!online_q || !online_raw || !online_comp) return 6;
    for (uint32_t index = 0; index < ONLINE_RAW_CAP * ONLINE_HEAD_DIM; ++index) online_raw[index] = 7.0f;
    for (uint64_t index = 0; index < online_elements; ++index) online_comp[index] = 2.0f;
    for (uint32_t index = 0; index < ONLINE_HEAD_DIM; ++index) online_sentinel[index] = sentinel;
    if (!ds4_gpu_set_model_map(&online_model, sizeof(online_model))) return 7;
    ds4_gpu_tensor *online_heads = ds4_gpu_tensor_alloc(sizeof(online_output));
    ds4_gpu_tensor *online_q_tensor = ds4_gpu_tensor_alloc(ONLINE_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *online_raw_tensor =
            ds4_gpu_tensor_alloc((uint64_t)ONLINE_RAW_CAP * ONLINE_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *online_comp_tensor = ds4_gpu_tensor_alloc(online_bytes);
    if (!online_heads || !online_q_tensor || !online_raw_tensor || !online_comp_tensor ||
        !ds4_gpu_tensor_write(online_q_tensor, 0, online_q, ONLINE_HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(
                online_raw_tensor, 0, online_raw,
                (uint64_t)ONLINE_RAW_CAP * ONLINE_HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(online_comp_tensor, 0, online_comp, online_bytes) ||
        !ds4_gpu_attention_decode_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, ONLINE_N_RAW, ONLINE_RAW_CAP, 0u, online_comp_tensor,
                ONLINE_N_COMP, NULL, 0u, ONLINE_HEADS, ONLINE_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output))) {
        return 8;
    }
    const float online_expected = 2.0f * (float)ONLINE_N_COMP / ((float)ONLINE_N_COMP + 1.0f);
    for (uint32_t index = 0; index < ONLINE_HEAD_DIM; ++index) {
        if (fabsf(online_output[index] - online_expected) > 5.0e-4f) return 9;
    }
    if (setenv("DS4_CUDA_NO_WINDOW_ATTENTION", "1", 1) != 0 ||
        !ds4_gpu_tensor_write(online_heads, 0, online_sentinel, sizeof(online_sentinel)) ||
        ds4_gpu_attention_decode_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, ONLINE_N_RAW, ONLINE_RAW_CAP, 0u, online_comp_tensor,
                ONLINE_N_COMP, NULL, 0u, ONLINE_HEADS, ONLINE_HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(online_heads, 0, online_output, sizeof(online_output)) ||
        !sentinel_intact(online_output, ONLINE_HEAD_DIM, sentinel) ||
        unsetenv("DS4_CUDA_NO_WINDOW_ATTENTION") != 0 ||
        ds4_gpu_attention_decode_heads_tensor(
                online_heads, &online_model, sizeof(online_model), 0u, online_q_tensor,
                online_raw_tensor, ONLINE_N_RAW, ONLINE_RAW_CAP, 0u, online_comp_tensor,
                ONLINE_N_COMP, mask_tensor, 1u, ONLINE_HEADS, ONLINE_HEAD_DIM)) {
        return 10;
    }
    free(online_q);
    free(online_raw);
    free(online_comp);
    printf("{\"c_linked_rust_staticlib\":true,\"generic_masked_output_matches\":true,"
           "\"raw_only_ring_wrapped_output_matches\":true,\"sink_softmax_matches\":true,"
           "\"overflow_online_output_matches\":true,\"overflow_raw_visibility_matches\":true,"
           "\"overflow_env_disable_rejected\":true,\"overflow_mask_rejected\":true,"
           "\"invalid_model_range_preserves_output\":true,\"invalid_shape_rejected\":true,"
           "\"null_rejected\":true,\"embedded_attention_decode_kernels_loaded\":true}\n");
    return 0;
}
