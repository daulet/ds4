#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IN_DIM 35u
#define OUT_DIM 6u
#define BLOCKS ((IN_DIM + 31u) / 32u)
#define WEIGHT_BYTES ((uint64_t)OUT_DIM * BLOCKS * 34u)
#define MODEL_BYTES (2u * WEIGHT_BYTES)
#define CLAMP 0.75f

static void fill_packed_weights(uint8_t *weights, uint32_t seed) {
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        for (uint32_t block = 0; block < BLOCKS; ++block) {
            const uint64_t base = ((uint64_t)row * BLOCKS + block) * 34u;
            weights[base] = 0x00u;
            weights[base + 1u] = 0x3cu;
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                weights[base + 2u + lane] =
                    (uint8_t)((int32_t)((seed + row * 5u + block * 7u + lane * 3u) % 19u) - 9);
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

static void reference_swiglu(
        float out[OUT_DIM],
        const float gate[OUT_DIM],
        const float up[OUT_DIM]) {
    for (uint32_t index = 0; index < OUT_DIM; ++index) {
        float g = gate[index];
        float u = up[index];
        if (g > CLAMP) g = CLAMP;
        if (u < -CLAMP) {
            u = -CLAMP;
        } else if (u > CLAMP) {
            u = CLAMP;
        }
        out[index] = (g / (1.0f + expf(-g))) * u;
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-4f) return 0;
    }
    return 1;
}

static int run_shared(
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        const uint8_t *model,
        ds4_gpu_tensor *x,
        float got_gate[OUT_DIM],
        float got_up[OUT_DIM],
        float got_mid[OUT_DIM]) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               gate, up, mid, model, MODEL_BYTES, 0, WEIGHT_BYTES, IN_DIM, OUT_DIM, x, CLAMP) &&
           ds4_gpu_synchronize() &&
           ds4_gpu_tensor_read(gate, 0, got_gate, sizeof(float) * OUT_DIM) &&
           ds4_gpu_tensor_read(up, 0, got_up, sizeof(float) * OUT_DIM) &&
           ds4_gpu_tensor_read(mid, 0, got_mid, sizeof(float) * OUT_DIM);
}

int main(void) {
    uint8_t *model = malloc((size_t)MODEL_BYTES);
    float x_values[IN_DIM];
    float expected_gate[OUT_DIM] = {0};
    float expected_up[OUT_DIM] = {0};
    float expected_mid[OUT_DIM] = {0};
    float got_gate[OUT_DIM] = {0};
    float got_up[OUT_DIM] = {0};
    float got_mid[OUT_DIM] = {0};
    if (!model) return 1;
    fill_packed_weights(model, 0u);
    fill_packed_weights(model + WEIGHT_BYTES, 11u);
    fill_activations(x_values);
    reference_native(expected_gate, model, x_values);
    reference_native(expected_up, model + WEIGHT_BYTES, x_values);
    reference_swiglu(expected_mid, expected_gate, expected_up);

    if (unsetenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") != 0 ||
        unsetenv("DS4_CUDA_NO_Q8_DP4A") != 0 ||
        !ds4_gpu_init() ||
        !ds4_gpu_set_model_map(model, MODEL_BYTES)) {
        return 2;
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_values));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(sizeof(got_gate));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(sizeof(got_up));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(sizeof(got_mid));
    if (!x || !gate || !up || !mid ||
        !ds4_gpu_tensor_write(x, 0, x_values, sizeof(x_values))) {
        return 3;
    }
    if (!run_shared(gate, up, mid, model, x, got_gate, got_up, got_mid) ||
        !close_array(got_gate, expected_gate, OUT_DIM) ||
        !close_array(got_up, expected_up, OUT_DIM) ||
        !close_array(got_mid, expected_mid, OUT_DIM)) {
        return 4;
    }
    if (setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1) != 0 ||
        !run_shared(gate, up, mid, model, x, got_gate, got_up, got_mid) ||
        !close_array(got_gate, expected_gate, OUT_DIM) ||
        !close_array(got_up, expected_up, OUT_DIM) ||
        !close_array(got_mid, expected_mid, OUT_DIM)) {
        return 5;
    }
    if (unsetenv("DS4_CUDA_NO_Q8_DP4A") != 0 ||
        setenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR", "1", 1) != 0 ||
        !run_shared(gate, up, mid, model, x, got_gate, got_up, got_mid) ||
        !close_array(got_gate, expected_gate, OUT_DIM) ||
        !close_array(got_up, expected_up, OUT_DIM) ||
        !close_array(got_mid, expected_mid, OUT_DIM)) {
        return 6;
    }
    if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
            gate, up, mid, model, MODEL_BYTES - 1u, 0, WEIGHT_BYTES, IN_DIM, OUT_DIM, x, CLAMP)) {
        return 7;
    }

    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(model);
    puts("{\"c_linked_rust_staticlib\":true,\"paired_dp4a_output_matches\":true,"
         "\"paired_scalar_output_matches\":true,\"disabled_pair_fallback_output_matches\":true,"
         "\"swiglu_clamp_output_matches\":true,\"invalid_range_rejected\":true,"
         "\"embedded_q8_pair_kernel_loaded\":true}");
    return 0;
}
