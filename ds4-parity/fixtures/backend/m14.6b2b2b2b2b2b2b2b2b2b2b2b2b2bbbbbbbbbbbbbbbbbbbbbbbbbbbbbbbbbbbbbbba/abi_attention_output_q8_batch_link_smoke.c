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
#define OUT_DIM 5u
#define N_TOKENS 3u
#define A_BLOCKS ((GROUP_DIM + 31u) / 32u)
#define B_BLOCKS ((LOW_DIM + 31u) / 32u)
#define OUT_A_BYTES (LOW_DIM * A_BLOCKS * 34u)
#define OUT_B_BYTES (OUT_DIM * B_BLOCKS * 34u)
#define HEAD_ELEMENTS ((uint64_t)N_TOKENS * N_GROUPS * GROUP_DIM)
#define LOW_ELEMENTS ((uint64_t)N_TOKENS * LOW_DIM)
#define OUT_ELEMENTS ((uint64_t)N_TOKENS * OUT_DIM)

struct model {
    uint8_t prefix[5];
    uint8_t out_a[OUT_A_BYTES];
    uint8_t pad[3];
    uint8_t out_b[OUT_B_BYTES];
};

static int8_t clamp_i8(int value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static void fill_packed_weights(uint8_t *weights, uint64_t rows, uint32_t blocks, uint32_t seed) {
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint64_t base = (row * blocks + block) * 34u;
            weights[base] = 0x00u;
            weights[base + 1u] = 0x3cu;
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                weights[base + 2u + lane] =
                    (uint8_t)((int32_t)((row * 7u + block * 5u + lane * 3u + seed) % 23u) - 11);
            }
        }
    }
}

static void fill_heads(float *heads) {
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        for (uint32_t group = 0; group < N_GROUPS; ++group) {
            for (uint32_t column = 0; column < GROUP_DIM; ++column) {
                const uint64_t index =
                    ((uint64_t)token * N_GROUPS + group) * GROUP_DIM + column;
                const int32_t value =
                    (int32_t)((token * 17u + group * 11u + column * 5u) % 31u) - 15;
                heads[index] = (float)value * 0.125f;
            }
        }
    }
}

static void reference_a_native(float *low, const uint8_t *weights, const float *heads) {
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        for (uint32_t row = 0; row < LOW_DIM; ++row) {
            const uint32_t group = row / RANK;
            float total = 0.0f;
            for (uint32_t block = 0; block < A_BLOCKS; ++block) {
                const uint32_t start = block * 32u;
                const uint32_t count = GROUP_DIM - start < 32u ? GROUP_DIM - start : 32u;
                const uint64_t input_base =
                    ((uint64_t)token * N_GROUPS + group) * GROUP_DIM + start;
                float maximum = 0.0f;
                for (uint32_t lane = 0; lane < count; ++lane) {
                    const float magnitude = fabsf(heads[input_base + lane]);
                    if (magnitude > maximum) maximum = magnitude;
                }
                const float scale = maximum / 127.0f;
                const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
                const uint64_t weight_base = ((uint64_t)row * A_BLOCKS + block) * 34u;
                int dot = 0;
                for (uint32_t lane = 0; lane < count; ++lane) {
                    dot += (int8_t)weights[weight_base + 2u + lane] *
                           clamp_i8((int)nearbyintf(heads[input_base + lane] * inverse));
                }
                total += scale * (float)dot;
            }
            low[(uint64_t)token * LOW_DIM + row] = total;
        }
    }
}

static void reference_a_f16_adapter(float *low, const uint8_t *weights, const float *heads) {
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        for (uint32_t row = 0; row < LOW_DIM; ++row) {
            const uint32_t group = row / RANK;
            float total = 0.0f;
            for (uint32_t column = 0; column < GROUP_DIM; ++column) {
                const uint32_t block = column / 32u;
                const uint32_t lane = column % 32u;
                const uint64_t input =
                    ((uint64_t)token * N_GROUPS + group) * GROUP_DIM + column;
                const uint64_t weight = ((uint64_t)row * A_BLOCKS + block) * 34u + 2u + lane;
                total += heads[input] * (float)(int8_t)weights[weight];
            }
            low[(uint64_t)token * LOW_DIM + row] = total;
        }
    }
}

static void reference_b_native(float *out, const uint8_t *weights, const float *low) {
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            float maximum = 0.0f;
            const uint64_t input_base = (uint64_t)token * LOW_DIM;
            for (uint32_t lane = 0; lane < LOW_DIM; ++lane) {
                const float magnitude = fabsf(low[input_base + lane]);
                if (magnitude > maximum) maximum = magnitude;
            }
            const float scale = maximum / 127.0f;
            const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
            const uint64_t weight_base = (uint64_t)row * 34u;
            int dot = 0;
            for (uint32_t lane = 0; lane < LOW_DIM; ++lane) {
                dot += (int8_t)weights[weight_base + 2u + lane] *
                       clamp_i8((int)nearbyintf(low[input_base + lane] * inverse));
            }
            out[(uint64_t)token * OUT_DIM + row] = scale * (float)dot;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint64_t count, float tolerance) {
    for (uint64_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > tolerance) return 0;
    }
    return 1;
}

static int different_array(const float *left, const float *right, uint64_t count, float tolerance) {
    for (uint64_t index = 0; index < count; ++index) {
        if (fabsf(left[index] - right[index]) > tolerance) return 1;
    }
    return 0;
}

static int finite_array(const float *values, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

static int reset_environment(void) {
    return unsetenv("DS4_CUDA_Q8_F32_ALL") == 0 &&
           unsetenv("DS4_CUDA_Q8_F16_ALL") == 0 &&
           unsetenv("DS4_CUDA_Q8_F32_LARGE") == 0 &&
           unsetenv("DS4_CUDA_ATTN_Q_B_F32_CACHE") == 0 &&
           unsetenv("DS4_CUDA_NO_Q8_F16_CACHE") == 0 &&
           unsetenv("DS4_CUDA_Q8_F16_CACHE_MB") == 0 &&
           unsetenv("DS4_CUDA_NO_Q8_DP4A") == 0 &&
           setenv("DS4_CUDA_NO_TF32", "1", 1) == 0;
}

static int run_valid(
        const struct model *model,
        const float *heads_values,
        int disable_attention_cache,
        int disable_output_a_cublas,
        const char *minimum_tokens,
        float *low_values,
        float *out_values) {
    if (!reset_environment()) return 0;
    if ((disable_attention_cache
             ? setenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE", "1", 1)
             : unsetenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE")) != 0 ||
        (disable_output_a_cublas
             ? setenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A", "1", 1)
             : unsetenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A")) != 0 ||
        (minimum_tokens
             ? setenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN", minimum_tokens, 1)
             : unsetenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN")) != 0 ||
        !ds4_gpu_init() ||
        !ds4_gpu_set_model_map(model, sizeof(*model))) {
        return 0;
    }
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(float) * HEAD_ELEMENTS);
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(sizeof(float) * LOW_ELEMENTS);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float) * OUT_ELEMENTS);
    const int ok =
        heads && low && out &&
        ds4_gpu_tensor_write(heads, 0, heads_values, sizeof(float) * HEAD_ELEMENTS) &&
        ds4_gpu_attention_output_q8_batch_tensor(
            out, low, NULL, NULL, model, sizeof(*model),
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, N_TOKENS) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(low, 0, low_values, sizeof(float) * LOW_ELEMENTS) &&
        ds4_gpu_tensor_read(out, 0, out_values, sizeof(float) * OUT_ELEMENTS);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    return ok;
}

static int run_invalid_preservation(const struct model *model, const float *heads_values) {
    float low_sentinel[LOW_ELEMENTS];
    float out_sentinel[OUT_ELEMENTS];
    float low_preserved[LOW_ELEMENTS] = {0};
    float out_preserved[OUT_ELEMENTS] = {0};
    for (uint64_t index = 0; index < LOW_ELEMENTS; ++index) {
        low_sentinel[index] = 210.0f + (float)index;
    }
    for (uint64_t index = 0; index < OUT_ELEMENTS; ++index) {
        out_sentinel[index] = 410.0f + (float)index;
    }
    if (!reset_environment() ||
        setenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE", "1", 1) != 0 ||
        !ds4_gpu_init() ||
        !ds4_gpu_set_model_map(model, sizeof(*model))) {
        return 0;
    }
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(float) * HEAD_ELEMENTS);
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(sizeof(float) * LOW_ELEMENTS);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float) * OUT_ELEMENTS);
    ds4_gpu_tensor *short_heads = ds4_gpu_tensor_alloc(sizeof(float) * HEAD_ELEMENTS - sizeof(float));
    ds4_gpu_tensor *short_low = ds4_gpu_tensor_alloc(sizeof(float) * LOW_ELEMENTS - sizeof(float));
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc(sizeof(float) * OUT_ELEMENTS - sizeof(float));
    const int ok =
        heads && low && out && short_heads && short_low && short_out &&
        ds4_gpu_tensor_write(heads, 0, heads_values, sizeof(float) * HEAD_ELEMENTS) &&
        ds4_gpu_tensor_write(low, 0, low_sentinel, sizeof(low_sentinel)) &&
        ds4_gpu_tensor_write(out, 0, out_sentinel, sizeof(out_sentinel)) &&
        !ds4_gpu_attention_output_q8_batch_tensor(
            out, low, NULL, NULL, model, offsetof(struct model, out_b) + OUT_B_BYTES - 1u,
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, N_TOKENS) &&
        !ds4_gpu_attention_output_q8_batch_tensor(
            out, low, NULL, NULL, model, sizeof(*model),
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            GROUP_DIM, RANK, N_GROUPS, OUT_DIM, short_heads, N_TOKENS) &&
        !ds4_gpu_attention_output_q8_batch_tensor(
            out, short_low, NULL, NULL, model, sizeof(*model),
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, N_TOKENS) &&
        !ds4_gpu_attention_output_q8_batch_tensor(
            short_out, low, NULL, NULL, model, sizeof(*model),
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, N_TOKENS) &&
        !ds4_gpu_attention_output_q8_batch_tensor(
            out, low, NULL, NULL, model, sizeof(*model),
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            0u, RANK, N_GROUPS, OUT_DIM, heads, N_TOKENS) &&
        !ds4_gpu_attention_output_q8_batch_tensor(
            NULL, low, NULL, NULL, model, sizeof(*model),
            offsetof(struct model, out_a), offsetof(struct model, out_b),
            GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, N_TOKENS) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(low, 0, low_preserved, sizeof(low_preserved)) &&
        ds4_gpu_tensor_read(out, 0, out_preserved, sizeof(out_preserved)) &&
        close_array(low_preserved, low_sentinel, LOW_ELEMENTS, 0.0f) &&
        close_array(out_preserved, out_sentinel, OUT_ELEMENTS, 0.0f);
    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(short_low);
    ds4_gpu_tensor_free(short_heads);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    return ok;
}

int main(void) {
    struct model model = {0};
    float heads_values[HEAD_ELEMENTS];
    float expected_native_low[LOW_ELEMENTS] = {0};
    float expected_native_out[OUT_ELEMENTS] = {0};
    float expected_cublas_low[LOW_ELEMENTS] = {0};
    float native_low[LOW_ELEMENTS] = {0};
    float native_out[OUT_ELEMENTS] = {0};
    float cached_b_low[LOW_ELEMENTS] = {0};
    float cached_b_out[OUT_ELEMENTS] = {0};
    float cublas_low[LOW_ELEMENTS] = {0};
    float cublas_out[OUT_ELEMENTS] = {0};
    float gated_low[LOW_ELEMENTS] = {0};
    float gated_out[OUT_ELEMENTS] = {0};
    float min_low[LOW_ELEMENTS] = {0};
    float min_out[OUT_ELEMENTS] = {0};

    fill_packed_weights(model.out_a, LOW_DIM, A_BLOCKS, 2u);
    fill_packed_weights(model.out_b, OUT_DIM, B_BLOCKS, 9u);
    fill_heads(heads_values);
    reference_a_native(expected_native_low, model.out_a, heads_values);
    reference_b_native(expected_native_out, model.out_b, expected_native_low);
    reference_a_f16_adapter(expected_cublas_low, model.out_a, heads_values);

    if (!run_valid(&model, heads_values, 1, 1, NULL, native_low, native_out) ||
        !close_array(native_low, expected_native_low, LOW_ELEMENTS, 1.0e-4f) ||
        !close_array(native_out, expected_native_out, OUT_ELEMENTS, 1.0e-3f)) {
        return 1;
    }
    if (!run_valid(&model, heads_values, 0, 1, NULL, cached_b_low, cached_b_out) ||
        !close_array(cached_b_low, expected_native_low, LOW_ELEMENTS, 1.0e-4f) ||
        !different_array(cached_b_out, native_out, OUT_ELEMENTS, 1.0e-3f)) {
        return 2;
    }
    if (!run_valid(&model, heads_values, 0, 0, "2", cublas_low, cublas_out) ||
        !close_array(cublas_low, expected_cublas_low, LOW_ELEMENTS, 1.0e-4f) ||
        !different_array(cublas_low, native_low, LOW_ELEMENTS, 1.0e-3f) ||
        !finite_array(cublas_out, OUT_ELEMENTS)) {
        return 3;
    }
    if (!run_valid(&model, heads_values, 0, 1, "2", gated_low, gated_out) ||
        !close_array(gated_low, expected_native_low, LOW_ELEMENTS, 1.0e-4f) ||
        !run_valid(&model, heads_values, 0, 0, "4", min_low, min_out) ||
        !close_array(min_low, expected_native_low, LOW_ELEMENTS, 1.0e-4f)) {
        return 4;
    }
    if (!run_invalid_preservation(&model, heads_values)) {
        return 5;
    }
    puts("{\"c_linked_rust_staticlib\":true,\"native_two_stage_output_matches\":true,"
         "\"partial_q8_block_matches\":true,\"cublas_a_output_matches\":true,"
         "\"cublas_a_dispatch_diverges_from_native\":true,"
         "\"cublas_a_environment_gate_matches\":true,"
         "\"cublas_a_minimum_token_gate_matches\":true,"
         "\"attention_output_cache_labels_honored\":true,"
         "\"invalid_model_range_preserves_outputs\":true,"
         "\"invalid_shape_rejected\":true,\"null_rejected\":true,"
         "\"embedded_output_adapter_kernels_loaded\":true}");
    return 0;
}
