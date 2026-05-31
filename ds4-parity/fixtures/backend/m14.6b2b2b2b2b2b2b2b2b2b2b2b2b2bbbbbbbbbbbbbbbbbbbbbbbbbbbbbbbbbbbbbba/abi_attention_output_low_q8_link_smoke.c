#include "ds4_gpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GROUP_DIM 35u
#define RANK 3u
#define N_GROUPS 2u
#define LOW_DIM ((uint64_t)RANK * N_GROUPS)
#define BLOCKS ((GROUP_DIM + 31u) / 32u)
#define WEIGHT_BYTES (LOW_DIM * BLOCKS * 34u)
#define HEAD_ELEMENTS ((uint64_t)N_GROUPS * GROUP_DIM)

struct model {
    uint8_t prefix[5];
    uint8_t weights[WEIGHT_BYTES];
};

static int8_t clamp_i8(int value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static void fill_packed_weights(uint8_t *weights) {
    for (uint32_t row = 0; row < LOW_DIM; ++row) {
        for (uint32_t block = 0; block < BLOCKS; ++block) {
            const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
            weights[base] = 0x00u;
            weights[base + 1u] = 0x3cu;
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                weights[base + 2u + lane] =
                    (uint8_t)((int32_t)((row * 7u + block * 5u + lane * 3u) % 23u) - 11);
            }
        }
    }
}

static void fill_heads(float *heads) {
    for (uint32_t group = 0; group < N_GROUPS; ++group) {
        for (uint32_t column = 0; column < GROUP_DIM; ++column) {
            heads[(uint64_t)group * GROUP_DIM + column] =
                (float)((int32_t)((group * 13u + column * 5u) % 27u) - 13);
        }
    }
}

static void reference_low(float *low, const uint8_t *weights, const float *heads) {
    for (uint32_t row = 0; row < LOW_DIM; ++row) {
        const uint32_t group = row / RANK;
        float total = 0.0f;
        for (uint32_t block = 0; block < BLOCKS; ++block) {
            const uint32_t start = block * 32u;
            const uint32_t count = GROUP_DIM - start < 32u ? GROUP_DIM - start : 32u;
            float maximum = 0.0f;
            for (uint32_t lane = 0; lane < count; ++lane) {
                const float value = heads[(uint64_t)group * GROUP_DIM + start + lane];
                const float magnitude = fabsf(value);
                if (magnitude > maximum) maximum = magnitude;
            }
            const float scale = maximum / 127.0f;
            const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
            const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
            int dot = 0;
            for (uint32_t lane = 0; lane < count; ++lane) {
                const float value = heads[(uint64_t)group * GROUP_DIM + start + lane];
                dot += (int8_t)weights[base + 2u + lane] *
                       clamp_i8((int)nearbyintf(value * inverse));
            }
            total += scale * (float)dot;
        }
        low[row] = total;
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-4f) return 0;
    }
    return 1;
}

static int run_low(
        ds4_gpu_tensor *low,
        const struct model *model,
        uint64_t model_size,
        ds4_gpu_tensor *heads,
        float *values) {
    return ds4_gpu_attention_output_low_q8_tensor(
                   low, model, model_size, offsetof(struct model, weights),
                   GROUP_DIM, RANK, N_GROUPS, heads) &&
           ds4_gpu_synchronize() &&
           ds4_gpu_tensor_read(low, 0, values, sizeof(float) * LOW_DIM);
}

int main(void) {
    struct model model = {0};
    float heads_values[HEAD_ELEMENTS];
    float expected[LOW_DIM] = {0};
    float default_output[LOW_DIM] = {0};
    float scalar_output[LOW_DIM] = {0};
    float sentinel[LOW_DIM];
    float preserved[LOW_DIM] = {0};
    fill_packed_weights(model.weights);
    fill_heads(heads_values);
    reference_low(expected, model.weights, heads_values);
    for (uint32_t index = 0; index < LOW_DIM; ++index) {
        sentinel[index] = 310.0f + (float)index;
    }

    if (unsetenv("DS4_CUDA_NO_Q8_DP4A") != 0 ||
        !ds4_gpu_init() ||
        !ds4_gpu_set_model_map(&model, sizeof(model))) {
        return 1;
    }
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(heads_values));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(sizeof(default_output));
    ds4_gpu_tensor *short_heads = ds4_gpu_tensor_alloc(sizeof(heads_values) - sizeof(float));
    ds4_gpu_tensor *short_low = ds4_gpu_tensor_alloc(sizeof(default_output) - sizeof(float));
    if (!heads || !low || !short_heads || !short_low ||
        !ds4_gpu_tensor_write(heads, 0, heads_values, sizeof(heads_values)) ||
        !run_low(low, &model, sizeof(model), heads, default_output) ||
        !close_array(default_output, expected, LOW_DIM)) {
        return 2;
    }

    if (setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1) != 0 ||
        !run_low(low, &model, sizeof(model), heads, scalar_output) ||
        !close_array(scalar_output, expected, LOW_DIM) ||
        !close_array(scalar_output, default_output, LOW_DIM)) {
        return 3;
    }

    if (!ds4_gpu_tensor_write(low, 0, sentinel, sizeof(sentinel)) ||
        ds4_gpu_attention_output_low_q8_tensor(
            low, &model, sizeof(model) - 1u, offsetof(struct model, weights),
            GROUP_DIM, RANK, N_GROUPS, heads) ||
        ds4_gpu_attention_output_low_q8_tensor(
            low, &model, sizeof(model), offsetof(struct model, weights),
            GROUP_DIM, RANK, N_GROUPS, short_heads) ||
        ds4_gpu_attention_output_low_q8_tensor(
            short_low, &model, sizeof(model), offsetof(struct model, weights),
            GROUP_DIM, RANK, N_GROUPS, heads) ||
        ds4_gpu_attention_output_low_q8_tensor(
            low, &model, sizeof(model), offsetof(struct model, weights),
            0u, RANK, N_GROUPS, heads) ||
        ds4_gpu_attention_output_low_q8_tensor(
            NULL, &model, sizeof(model), offsetof(struct model, weights),
            GROUP_DIM, RANK, N_GROUPS, heads) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(low, 0, preserved, sizeof(preserved)) ||
        !close_array(preserved, sentinel, LOW_DIM)) {
        return 4;
    }

    ds4_gpu_tensor_free(short_low);
    ds4_gpu_tensor_free(short_heads);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"low_q8_output_matches\":true,"
         "\"partial_q8_block_matches\":true,\"dp4a_environment_gate_matches\":true,"
         "\"invalid_model_range_preserves_output\":true,\"invalid_shape_rejected\":true,"
         "\"null_rejected\":true,\"embedded_grouped_q8_output_kernel_loaded\":true}");
    return 0;
}
