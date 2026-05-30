#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N_ROWS 2u
#define N_HC 4u
#define MIX_HC 24u
#define SINKHORN_ITERS 3u
#define EPS 1.0e-4f

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

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 3.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    float mix[N_ROWS * MIX_HC];
    float model[60] = {0};
    const uint64_t scale_offset = 1u * sizeof(float);
    const uint64_t base_offset = 5u * sizeof(float);
    const uint64_t second_scale_offset = 30u * sizeof(float);
    const uint64_t second_base_offset = 34u * sizeof(float);
    float expected[N_ROWS * MIX_HC] = {0};
    float expected_second[N_ROWS * MIX_HC] = {0};
    float got[N_ROWS * MIX_HC] = {0};
    for (uint32_t index = 0; index < N_ROWS * MIX_HC; ++index) {
        mix[index] = (float)((int32_t)((index * 7u + 3u) % 19u) - 9) * 0.08f;
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
    for (uint32_t row = 0; row < N_ROWS; ++row) {
        reference_split_one(
            &expected[row * MIX_HC], &mix[row * MIX_HC], &model[1], &model[5]);
        reference_split_one(
            &expected_second[row * MIX_HC], &mix[row * MIX_HC], &model[30], &model[34]);
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model, sizeof(model))) return 1;
    ds4_gpu_tensor *mix_tensor = ds4_gpu_tensor_alloc(sizeof(mix));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *one_row_out = ds4_gpu_tensor_alloc(MIX_HC * sizeof(float));
    ds4_gpu_tensor *short_mix = ds4_gpu_tensor_alloc((MIX_HC - 1u) * sizeof(float));
    if (!mix_tensor || !out || !one_row_out || !short_mix ||
        !ds4_gpu_tensor_write(mix_tensor, 0, mix, sizeof(mix)) ||
        !ds4_gpu_tensor_write(short_mix, 0, mix, (MIX_HC - 1u) * sizeof(float))) {
        return 2;
    }
    if (!ds4_gpu_hc_split_sinkhorn_tensor(
            out, mix_tensor, model, sizeof(model), scale_offset, base_offset,
            N_HC, SINKHORN_ITERS, EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, expected, N_ROWS * MIX_HC)) {
        return 3;
    }
    if (!ds4_gpu_hc_split_sinkhorn_tensor(
            one_row_out, mix_tensor, model, sizeof(model), scale_offset, base_offset,
            N_HC, SINKHORN_ITERS, EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(one_row_out, 0, got, MIX_HC * sizeof(float)) ||
        !close_array(got, expected, MIX_HC)) {
        return 4;
    }
    if (!ds4_gpu_hc_split_sinkhorn_tensor(
            out, mix_tensor, model, sizeof(model), second_scale_offset, second_base_offset,
            N_HC, SINKHORN_ITERS, EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, expected_second, N_ROWS * MIX_HC)) {
        return 5;
    }
    if (ds4_gpu_hc_split_sinkhorn_tensor(
            out, short_mix, model, sizeof(model), scale_offset, base_offset,
            N_HC, SINKHORN_ITERS, EPS) ||
        ds4_gpu_hc_split_sinkhorn_tensor(
            out, mix_tensor, model, sizeof(model), scale_offset, sizeof(model) - sizeof(float),
            N_HC, SINKHORN_ITERS, EPS) ||
        ds4_gpu_hc_split_sinkhorn_tensor(
            out, mix_tensor, model, sizeof(model), scale_offset, base_offset,
            3u, SINKHORN_ITERS, EPS)) {
        return 6;
    }

    ds4_gpu_tensor_free(short_mix);
    ds4_gpu_tensor_free(one_row_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mix_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"sinkhorn_split_output_matches\":true,"
         "\"shorter_output_row_floor_matches\":true,\"alternate_parameter_range_matches\":true,"
         "\"short_mix_rejected\":true,\"invalid_model_range_rejected\":true,"
         "\"invalid_hc_count_rejected\":true,\"embedded_hc_split_sinkhorn_kernel_loaded\":true}");
    return 0;
}
