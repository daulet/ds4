#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N_TOKENS 3u
#define N_HC 4u
#define EPS 1.0e-4f

static void reference_weights(
        float out[N_TOKENS * N_HC],
        const float pre[N_TOKENS * N_HC],
        const float *scale,
        const float base[N_HC]) {
    for (uint32_t index = 0; index < N_TOKENS * N_HC; ++index) {
        const uint32_t hc = index % N_HC;
        const float z = pre[index] * scale[0] + base[hc];
        out[index] = 1.0f / (1.0f + expf(-z)) + EPS;
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 5.0e-6f) return 0;
    }
    return 1;
}

int main(void) {
    float pre[N_TOKENS * N_HC];
    float model[32] = {0};
    float expected[N_TOKENS * N_HC] = {0};
    float alternate_expected[N_TOKENS * N_HC] = {0};
    float got[N_TOKENS * N_HC] = {0};
    const uint64_t scale_offset = 1u * sizeof(float);
    const uint64_t base_offset = 4u * sizeof(float);
    const uint64_t alternate_scale_offset = 12u * sizeof(float);
    const uint64_t alternate_base_offset = 16u * sizeof(float);
    for (uint32_t index = 0; index < N_TOKENS * N_HC; ++index) {
        pre[index] = (float)((int32_t)((index * 7u + 2u) % 17u) - 8) * 0.11f;
    }
    model[1] = 0.8f;
    model[4] = -0.12f;
    model[5] = 0.06f;
    model[6] = 0.15f;
    model[7] = -0.03f;
    model[12] = -0.55f;
    model[16] = 0.18f;
    model[17] = -0.07f;
    model[18] = 0.02f;
    model[19] = 0.24f;
    reference_weights(expected, pre, &model[1], &model[4]);
    reference_weights(alternate_expected, pre, &model[12], &model[16]);

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model, sizeof(model))) return 1;
    ds4_gpu_tensor *pre_tensor = ds4_gpu_tensor_alloc(sizeof(pre));
    ds4_gpu_tensor *short_pre = ds4_gpu_tensor_alloc(sizeof(pre) - sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *one_out = ds4_gpu_tensor_alloc(N_HC * sizeof(float));
    ds4_gpu_tensor *partial_out = ds4_gpu_tensor_alloc((N_HC + 1u) * sizeof(float));
    if (!pre_tensor || !short_pre || !out || !one_out || !partial_out ||
        !ds4_gpu_tensor_write(pre_tensor, 0, pre, sizeof(pre)) ||
        !ds4_gpu_tensor_write(short_pre, 0, pre, sizeof(pre) - sizeof(float))) {
        return 2;
    }
    if (!ds4_gpu_output_hc_weights_tensor(
            out, pre_tensor, model, sizeof(model), scale_offset, base_offset, N_HC, EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, expected, N_TOKENS * N_HC)) {
        return 3;
    }
    if (!ds4_gpu_output_hc_weights_tensor(
            one_out, pre_tensor, model, sizeof(model), scale_offset, base_offset, N_HC, EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(one_out, 0, got, N_HC * sizeof(float)) ||
        !close_array(got, expected, N_HC)) {
        return 4;
    }
    if (!ds4_gpu_output_hc_weights_tensor(
            out, pre_tensor, model, sizeof(model), alternate_scale_offset,
            alternate_base_offset, N_HC, EPS) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, alternate_expected, N_TOKENS * N_HC)) {
        return 5;
    }
    if (ds4_gpu_output_hc_weights_tensor(
            out, short_pre, model, sizeof(model), scale_offset, base_offset, N_HC, EPS) ||
        ds4_gpu_output_hc_weights_tensor(
            partial_out, pre_tensor, model, sizeof(model), scale_offset, base_offset, N_HC, EPS) ||
        ds4_gpu_output_hc_weights_tensor(
            out, pre_tensor, model, sizeof(model), sizeof(model), base_offset, N_HC, EPS) ||
        ds4_gpu_output_hc_weights_tensor(
            out, pre_tensor, model, sizeof(model), scale_offset,
            sizeof(model) - (N_HC - 1u) * sizeof(float), N_HC, EPS) ||
        ds4_gpu_output_hc_weights_tensor(
            out, pre_tensor, model, sizeof(model), scale_offset, base_offset, 0, EPS)) {
        return 6;
    }

    ds4_gpu_tensor_free(partial_out);
    ds4_gpu_tensor_free(one_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(short_pre);
    ds4_gpu_tensor_free(pre_tensor);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"multi_token_sigmoid_weights_match\":true,"
         "\"single_token_row_derivation_matches\":true,\"alternate_parameter_range_matches\":true,"
         "\"short_pre_rejected\":true,\"partial_output_row_rejected\":true,"
         "\"invalid_scale_range_rejected\":true,\"invalid_base_range_rejected\":true,"
         "\"zero_hc_rejected\":true,\"embedded_output_hc_weights_kernel_loaded\":true}");
    return 0;
}
