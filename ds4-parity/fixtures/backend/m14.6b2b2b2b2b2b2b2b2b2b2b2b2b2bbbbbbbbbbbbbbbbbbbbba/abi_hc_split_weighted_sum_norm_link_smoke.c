#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N_ROWS 2u
#define N_HC 4u
#define N_EMBD 9u
#define MIX_HC 24u
#define SINKHORN_ITERS 3u
#define EPS 1.0e-4f
#define NORM_EPS 1.0e-5f

static void reference_split_one(
        float out[MIX_HC],
        const float mix[MIX_HC],
        const float scale[3],
        const float base[MIX_HC]) {
    for (uint32_t hc = 0; hc < N_HC; ++hc) {
        const float pre = mix[hc] * scale[0] + base[hc];
        const float post = mix[N_HC + hc] * scale[1] + base[N_HC + hc];
        out[hc] = 1.0f / (1.0f + expf(-pre)) + EPS;
        out[N_HC + hc] = 2.0f / (1.0f + expf(-post));
    }
    float combinations[16];
    for (uint32_t source = 0; source < N_HC; ++source) {
        float maximum = -INFINITY;
        for (uint32_t destination = 0; destination < N_HC; ++destination) {
            const uint32_t index = source * N_HC + destination;
            const float value = mix[2u * N_HC + index] * scale[2] +
                                base[2u * N_HC + index];
            combinations[index] = value;
            maximum = fmaxf(maximum, value);
        }
        float sum = 0.0f;
        for (uint32_t destination = 0; destination < N_HC; ++destination) {
            const uint32_t index = source * N_HC + destination;
            combinations[index] = expf(combinations[index] - maximum);
            sum += combinations[index];
        }
        for (uint32_t destination = 0; destination < N_HC; ++destination) {
            const uint32_t index = source * N_HC + destination;
            combinations[index] = combinations[index] / sum + EPS;
        }
    }
    for (uint32_t column = 0; column < N_HC; ++column) {
        float sum = EPS;
        for (uint32_t row = 0; row < N_HC; ++row) {
            sum += combinations[row * N_HC + column];
        }
        for (uint32_t row = 0; row < N_HC; ++row) {
            combinations[row * N_HC + column] /= sum;
        }
    }
    for (uint32_t iteration = 1; iteration < SINKHORN_ITERS; ++iteration) {
        for (uint32_t row = 0; row < N_HC; ++row) {
            float sum = EPS;
            for (uint32_t column = 0; column < N_HC; ++column) {
                sum += combinations[row * N_HC + column];
            }
            for (uint32_t column = 0; column < N_HC; ++column) {
                combinations[row * N_HC + column] /= sum;
            }
        }
        for (uint32_t column = 0; column < N_HC; ++column) {
            float sum = EPS;
            for (uint32_t row = 0; row < N_HC; ++row) {
                sum += combinations[row * N_HC + column];
            }
            for (uint32_t row = 0; row < N_HC; ++row) {
                combinations[row * N_HC + column] /= sum;
            }
        }
    }
    for (uint32_t index = 0; index < 16u; ++index) {
        out[2u * N_HC + index] = combinations[index];
    }
}

static void reference_outputs(
        float split[N_ROWS * MIX_HC],
        float out[N_ROWS * N_EMBD],
        float first_norm[N_EMBD],
        const float mix[N_ROWS * MIX_HC],
        const float residual[N_ROWS * N_HC * N_EMBD],
        const float scale[3],
        const float base[MIX_HC],
        const float norm_weight[N_EMBD]) {
    for (uint32_t row = 0; row < N_ROWS; ++row) {
        reference_split_one(
            &split[row * MIX_HC], &mix[row * MIX_HC], scale, base);
        for (uint32_t dimension = 0; dimension < N_EMBD; ++dimension) {
            float accumulator = 0.0f;
            for (uint32_t hc = 0; hc < N_HC; ++hc) {
                accumulator += residual[(row * N_HC + hc) * N_EMBD + dimension] *
                               split[row * MIX_HC + hc];
            }
            out[row * N_EMBD + dimension] = accumulator;
        }
    }
    float sum = 0.0f;
    for (uint32_t dimension = 0; dimension < N_EMBD; ++dimension) {
        sum += out[dimension] * out[dimension];
    }
    const float scale_value = 1.0f / sqrtf(sum / (float)N_EMBD + NORM_EPS);
    for (uint32_t dimension = 0; dimension < N_EMBD; ++dimension) {
        first_norm[dimension] = out[dimension] * scale_value * norm_weight[dimension];
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 5.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    float mix[N_ROWS * MIX_HC];
    float residual[N_ROWS * N_HC * N_EMBD];
    float model[96] = {0};
    const uint64_t scale_offset = 1u * sizeof(float);
    const uint64_t base_offset = 5u * sizeof(float);
    const uint64_t second_scale_offset = 30u * sizeof(float);
    const uint64_t second_base_offset = 34u * sizeof(float);
    const uint64_t norm_weight_offset = 60u * sizeof(float);
    float expected_split[N_ROWS * MIX_HC] = {0};
    float expected_out[N_ROWS * N_EMBD] = {0};
    float expected_norm[N_EMBD] = {0};
    float second_expected_split[N_ROWS * MIX_HC] = {0};
    float second_expected_out[N_ROWS * N_EMBD] = {0};
    float second_expected_norm[N_EMBD] = {0};
    float got_split[N_ROWS * MIX_HC] = {0};
    float got_out[N_ROWS * N_EMBD] = {0};
    float got_norm[N_ROWS * N_EMBD] = {0};
    float sentinel[N_ROWS * N_EMBD];
    for (uint32_t index = 0; index < N_ROWS * MIX_HC; ++index) {
        mix[index] = (float)((int32_t)((index * 7u + 3u) % 19u) - 9) * 0.08f;
    }
    for (uint32_t index = 0; index < N_ROWS * N_HC * N_EMBD; ++index) {
        residual[index] =
            (float)((int32_t)((index * 11u + 5u) % 29u) - 14) * 0.025f;
    }
    for (uint32_t index = 0; index < N_ROWS * N_EMBD; ++index) {
        sentinel[index] = 17.0f + (float)index;
    }
    model[1] = 0.75f;
    model[2] = -0.5f;
    model[3] = 0.625f;
    model[30] = -0.6f;
    model[31] = 0.4f;
    model[32] = 0.8f;
    for (uint32_t index = 0; index < MIX_HC; ++index) {
        model[5u + index] = (float)((int32_t)((index * 5u + 1u) % 13u) - 6) * 0.04f;
        model[34u + index] = (float)((int32_t)((index * 3u + 2u) % 17u) - 8) * 0.03f;
    }
    for (uint32_t index = 0; index < N_EMBD; ++index) {
        model[60u + index] = 0.8f + (float)index * 0.025f;
    }
    reference_outputs(expected_split, expected_out, expected_norm, mix, residual,
                      &model[1], &model[5], &model[60]);
    reference_outputs(second_expected_split, second_expected_out, second_expected_norm,
                      mix, residual, &model[30], &model[34], &model[60]);

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model, sizeof(model))) return 1;
    ds4_gpu_tensor *mix_tensor = ds4_gpu_tensor_alloc(sizeof(mix));
    ds4_gpu_tensor *residual_tensor = ds4_gpu_tensor_alloc(sizeof(residual));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(sizeof(got_split));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got_out));
    ds4_gpu_tensor *norm_out = ds4_gpu_tensor_alloc(sizeof(got_norm));
    ds4_gpu_tensor *one_split = ds4_gpu_tensor_alloc(MIX_HC * sizeof(float));
    ds4_gpu_tensor *one_out = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    ds4_gpu_tensor *one_norm = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    ds4_gpu_tensor *short_norm = ds4_gpu_tensor_alloc((N_EMBD - 1u) * sizeof(float));
    ds4_gpu_tensor *short_residual = ds4_gpu_tensor_alloc((N_HC * N_EMBD - 1u) * sizeof(float));
    if (!mix_tensor || !residual_tensor || !split || !out || !norm_out ||
        !one_split || !one_out || !one_norm || !short_norm || !short_residual ||
        !ds4_gpu_tensor_write(mix_tensor, 0, mix, sizeof(mix)) ||
        !ds4_gpu_tensor_write(residual_tensor, 0, residual, sizeof(residual)) ||
        !ds4_gpu_tensor_write(short_residual, 0, residual,
                              (N_HC * N_EMBD - 1u) * sizeof(float))) {
        return 2;
    }
    unsetenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED");
    if (!ds4_gpu_hc_split_weighted_sum_norm_tensor(
            one_out, one_norm, one_split, mix_tensor, residual_tensor, model, sizeof(model),
            scale_offset, base_offset, norm_weight_offset, N_EMBD, N_HC,
            SINKHORN_ITERS, EPS, NORM_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(one_split, 0, got_split, MIX_HC * sizeof(float)) ||
        !ds4_gpu_tensor_read(one_out, 0, got_out, N_EMBD * sizeof(float)) ||
        !ds4_gpu_tensor_read(one_norm, 0, got_norm, N_EMBD * sizeof(float)) ||
        !close_array(got_split, expected_split, MIX_HC) ||
        !close_array(got_out, expected_out, N_EMBD) ||
        !close_array(got_norm, expected_norm, N_EMBD)) {
        return 3;
    }
    if (!ds4_gpu_tensor_write(norm_out, 0, sentinel, sizeof(sentinel)) ||
        !ds4_gpu_hc_split_weighted_sum_norm_tensor(
            out, norm_out, split, mix_tensor, residual_tensor, model, sizeof(model),
            scale_offset, base_offset, norm_weight_offset, N_EMBD, N_HC,
            SINKHORN_ITERS, EPS, NORM_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(split, 0, got_split, sizeof(got_split)) ||
        !ds4_gpu_tensor_read(out, 0, got_out, sizeof(got_out)) ||
        !ds4_gpu_tensor_read(norm_out, 0, got_norm, sizeof(got_norm)) ||
        !close_array(got_split, expected_split, N_ROWS * MIX_HC) ||
        !close_array(got_out, expected_out, N_ROWS * N_EMBD) ||
        !close_array(got_norm, expected_norm, N_EMBD) ||
        !close_array(&got_norm[N_EMBD], &sentinel[N_EMBD], N_EMBD)) {
        return 4;
    }
    if (setenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED", "1", 1) != 0 ||
        !ds4_gpu_hc_split_weighted_sum_norm_tensor(
            one_out, one_norm, one_split, mix_tensor, residual_tensor, model, sizeof(model),
            scale_offset, base_offset, norm_weight_offset, N_EMBD, N_HC,
            SINKHORN_ITERS, EPS, NORM_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(one_out, 0, got_out, N_EMBD * sizeof(float)) ||
        !ds4_gpu_tensor_read(one_norm, 0, got_norm, N_EMBD * sizeof(float)) ||
        !close_array(got_out, expected_out, N_EMBD) ||
        !close_array(got_norm, expected_norm, N_EMBD)) {
        return 5;
    }
    unsetenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED");
    if (!ds4_gpu_hc_split_weighted_sum_norm_tensor(
            one_out, one_norm, one_split, mix_tensor, residual_tensor, model, sizeof(model),
            second_scale_offset, second_base_offset, norm_weight_offset, N_EMBD, N_HC,
            SINKHORN_ITERS, EPS, NORM_EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(one_split, 0, got_split, MIX_HC * sizeof(float)) ||
        !ds4_gpu_tensor_read(one_out, 0, got_out, N_EMBD * sizeof(float)) ||
        !ds4_gpu_tensor_read(one_norm, 0, got_norm, N_EMBD * sizeof(float)) ||
        !close_array(got_split, second_expected_split, MIX_HC) ||
        !close_array(got_out, second_expected_out, N_EMBD) ||
        !close_array(got_norm, second_expected_norm, N_EMBD)) {
        return 6;
    }
    if (ds4_gpu_hc_split_weighted_sum_norm_tensor(
            one_out, short_norm, one_split, mix_tensor, residual_tensor, model, sizeof(model),
            scale_offset, base_offset, norm_weight_offset, N_EMBD, N_HC,
            SINKHORN_ITERS, EPS, NORM_EPS) ||
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            one_out, one_norm, one_split, mix_tensor, short_residual, model, sizeof(model),
            scale_offset, base_offset, norm_weight_offset, N_EMBD, N_HC,
            SINKHORN_ITERS, EPS, NORM_EPS) ||
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            one_out, one_norm, one_split, mix_tensor, residual_tensor, model, sizeof(model),
            scale_offset, base_offset, sizeof(model) - (N_EMBD - 1u) * sizeof(float),
            N_EMBD, N_HC, SINKHORN_ITERS, EPS, NORM_EPS)) {
        return 7;
    }

    ds4_gpu_tensor_free(short_residual);
    ds4_gpu_tensor_free(short_norm);
    ds4_gpu_tensor_free(one_norm);
    ds4_gpu_tensor_free(one_out);
    ds4_gpu_tensor_free(one_split);
    ds4_gpu_tensor_free(norm_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(residual_tensor);
    ds4_gpu_tensor_free(mix_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"one_row_fused_output_matches\":true,"
         "\"one_row_fused_norm_matches\":true,\"one_row_fused_split_matches\":true,"
         "\"multi_row_fallback_output_matches\":true,"
         "\"multi_row_fallback_first_norm_only_matches\":true,"
         "\"disabled_fused_fallback_matches\":true,\"alternate_parameter_range_matches\":true,"
         "\"short_norm_out_rejected\":true,\"short_residual_rejected\":true,"
         "\"invalid_norm_weight_range_rejected\":true,"
         "\"embedded_hc_split_weighted_sum_norm_fused_kernel_loaded\":true}");
    return 0;
}
