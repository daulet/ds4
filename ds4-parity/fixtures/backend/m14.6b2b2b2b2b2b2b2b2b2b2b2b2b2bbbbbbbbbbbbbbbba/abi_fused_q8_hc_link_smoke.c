#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 35u
#define N_EMBD 4u
#define OUT_DIM N_EMBD
#define N_HC 2u
#define BLOCKS ((IN_DIM + 31u) / 32u)
#define WEIGHT_BYTES ((uint64_t)OUT_DIM * BLOCKS * 34u)
#define HC_ELEMENTS ((uint64_t)N_HC * N_EMBD)
#define MIX_HC (2u * N_HC + N_HC * N_HC)

static void fill_packed_weights(uint8_t *weights) {
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        for (uint32_t block = 0; block < BLOCKS; ++block) {
            const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
            weights[base] = 0x00u;
            weights[base + 1u] = 0x3cu;
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                weights[base + 2u + lane] =
                    (uint8_t)((int32_t)((row * 5u + block * 7u + lane * 3u) % 19u) - 9);
            }
        }
    }
}

static void fill_activations(float *x) {
    for (uint32_t column = 0; column < IN_DIM; ++column) {
        x[column] = (float)((int32_t)((column * 5u) % 21u) - 10);
    }
}

static int8_t clamp_i8(int value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static void reference_native(float out[OUT_DIM], const uint8_t *weights, const float *x) {
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        float total = 0.0f;
        for (uint32_t block = 0; block < BLOCKS; ++block) {
            const uint32_t start = block * 32u;
            const uint32_t count = IN_DIM - start < 32u ? IN_DIM - start : 32u;
            float maximum = 0.0f;
            for (uint32_t lane = 0; lane < count; ++lane) {
                const float magnitude = fabsf(x[start + lane]);
                if (magnitude > maximum) maximum = magnitude;
            }
            const float scale = maximum / 127.0f;
            const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
            const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
            int dot = 0;
            for (uint32_t lane = 0; lane < count; ++lane) {
                const int quantized = (int)nearbyintf(x[start + lane] * inverse);
                dot += (int8_t)weights[base + 2u + lane] * clamp_i8(quantized);
            }
            total += scale * (float)dot;
        }
        out[row] = total;
    }
}

static void reference_expand(
        float out[HC_ELEMENTS],
        const float block[OUT_DIM],
        const float add[OUT_DIM],
        const float residual[HC_ELEMENTS],
        const float split[MIX_HC],
        int has_add) {
    const float *post = split + N_HC;
    const float *comb = split + 2u * N_HC;
    for (uint32_t destination = 0; destination < N_HC; ++destination) {
        for (uint32_t dimension = 0; dimension < N_EMBD; ++dimension) {
            float block_value = block[dimension];
            if (has_add) block_value += add[dimension];
            float total = block_value * post[destination];
            for (uint32_t source = 0; source < N_HC; ++source) {
                total += comb[(uint64_t)source * N_HC + destination] *
                         residual[(uint64_t)source * N_EMBD + dimension];
            }
            out[(uint64_t)destination * N_EMBD + dimension] = total;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-4f) return 0;
    }
    return 1;
}

int main(void) {
    uint8_t *weights = malloc((size_t)WEIGHT_BYTES);
    float x_values[IN_DIM];
    const float routed[OUT_DIM] = {0.1f, -0.2f, 0.35f, 0.05f};
    const float residual[HC_ELEMENTS] = {0.2f, -0.3f, 0.4f, -0.1f, -0.7f, 0.8f, 0.1f, 0.6f};
    const float split[MIX_HC] = {0.0f, 0.0f, 0.5f, 1.25f, 1.0f, -0.25f, 0.4f, 0.9f};
    float expected_block[OUT_DIM] = {0};
    float expected_plain[HC_ELEMENTS] = {0};
    float expected_add[HC_ELEMENTS] = {0};
    float expected_alias[HC_ELEMENTS] = {0};
    float got_block[OUT_DIM] = {0};
    float got_hc[HC_ELEMENTS] = {0};
    if (!weights) return 1;
    fill_packed_weights(weights);
    fill_activations(x_values);
    reference_native(expected_block, weights, x_values);
    reference_expand(expected_plain, expected_block, routed, residual, split, 0);
    reference_expand(expected_add, expected_block, routed, residual, split, 1);
    reference_expand(expected_alias, expected_block, expected_block, residual, split, 1);

    unsetenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED");
    unsetenv("DS4_CUDA_NO_Q8_DP4A");
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(weights, WEIGHT_BYTES)) return 2;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_values));
    ds4_gpu_tensor *block = ds4_gpu_tensor_alloc(sizeof(got_block));
    ds4_gpu_tensor *out_hc = ds4_gpu_tensor_alloc(sizeof(got_hc));
    ds4_gpu_tensor *routed_out = ds4_gpu_tensor_alloc(sizeof(routed));
    ds4_gpu_tensor *residual_hc = ds4_gpu_tensor_alloc(sizeof(residual));
    ds4_gpu_tensor *split_tensor = ds4_gpu_tensor_alloc(sizeof(split));
    if (!x || !block || !out_hc || !routed_out || !residual_hc || !split_tensor) return 3;
    if (!ds4_gpu_tensor_write(x, 0, x_values, sizeof(x_values)) ||
        !ds4_gpu_tensor_write(routed_out, 0, routed, sizeof(routed)) ||
        !ds4_gpu_tensor_write(residual_hc, 0, residual, sizeof(residual)) ||
        !ds4_gpu_tensor_write(split_tensor, 0, split, sizeof(split))) {
        return 4;
    }
    if (!ds4_gpu_matmul_q8_0_hc_expand_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            residual_hc, split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(block, 0, got_block, sizeof(got_block)) ||
        !ds4_gpu_tensor_read(out_hc, 0, got_hc, sizeof(got_hc)) ||
        !close_array(got_block, expected_block, OUT_DIM) ||
        !close_array(got_hc, expected_plain, HC_ELEMENTS)) {
        return 5;
    }
    if (setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1) != 0 ||
        !ds4_gpu_matmul_q8_0_hc_expand_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            residual_hc, split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out_hc, 0, got_hc, sizeof(got_hc)) ||
        !close_array(got_hc, expected_plain, HC_ELEMENTS)) {
        return 6;
    }
    if (unsetenv("DS4_CUDA_NO_Q8_DP4A") != 0 ||
        setenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED", "1", 1) != 0 ||
        !ds4_gpu_matmul_q8_0_hc_expand_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            residual_hc, split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out_hc, 0, got_hc, sizeof(got_hc)) ||
        !close_array(got_hc, expected_plain, HC_ELEMENTS)) {
        return 7;
    }
    if (unsetenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") != 0 ||
        !ds4_gpu_shared_down_hc_expand_q8_0_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            routed_out, residual_hc, split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out_hc, 0, got_hc, sizeof(got_hc)) ||
        !close_array(got_hc, expected_add, HC_ELEMENTS)) {
        return 8;
    }
    if (setenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED", "1", 1) != 0 ||
        !ds4_gpu_shared_down_hc_expand_q8_0_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            routed_out, residual_hc, split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out_hc, 0, got_hc, sizeof(got_hc)) ||
        !close_array(got_hc, expected_add, HC_ELEMENTS)) {
        return 9;
    }
    if (unsetenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") != 0 ||
        !ds4_gpu_shared_down_hc_expand_q8_0_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            block, residual_hc, split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out_hc, 0, got_hc, sizeof(got_hc)) ||
        !close_array(got_hc, expected_alias, HC_ELEMENTS)) {
        return 10;
    }
    if (ds4_gpu_matmul_q8_0_hc_expand_tensor(
            out_hc, block, weights, WEIGHT_BYTES, 0, IN_DIM, OUT_DIM, x,
            residual_hc, split_tensor, N_EMBD - 1u, N_HC)) {
        return 11;
    }

    ds4_gpu_tensor_free(split_tensor);
    ds4_gpu_tensor_free(residual_hc);
    ds4_gpu_tensor_free(routed_out);
    ds4_gpu_tensor_free(out_hc);
    ds4_gpu_tensor_free(block);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(weights);
    puts("{\"c_linked_rust_staticlib\":true,\"fused_plain_dp4a_output_matches\":true,"
         "\"fused_plain_scalar_output_matches\":true,\"fallback_plain_output_matches\":true,"
         "\"fused_shared_add_output_matches\":true,\"fallback_shared_add_output_matches\":true,"
         "\"fused_shared_alias_add_output_matches\":true,\"invalid_shape_rejected\":true,"
         "\"embedded_fused_q8_hc_kernel_loaded\":true}");
    return 0;
}
