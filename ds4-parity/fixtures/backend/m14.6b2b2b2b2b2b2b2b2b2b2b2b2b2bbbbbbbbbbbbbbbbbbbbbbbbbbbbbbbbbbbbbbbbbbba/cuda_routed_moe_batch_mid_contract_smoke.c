#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256u
#define MODEL_EXPERTS 256u
#define N_TOKENS 2u
#define N_EXPERT 1u
#define OUT_DIM 5u
#define IQ2_BLOCK_BYTES 66u
#define Q2_BLOCK_BYTES 84u

struct moe_model {
    uint8_t *bytes;
    uint64_t size;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
};

struct moe_tensors {
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *down;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *weights;
    ds4_gpu_tensor *x;
};

static void store_u16(uint8_t *values, uint64_t offset, uint16_t value) {
    values[offset] = (uint8_t)value;
    values[offset + 1u] = (uint8_t)(value >> 8);
}

static int init_model(struct moe_model *model) {
    memset(model, 0, sizeof(*model));
    model->gate_row_bytes = IQ2_BLOCK_BYTES;
    model->down_row_bytes = Q2_BLOCK_BYTES;
    model->gate_expert_bytes = QK_K * model->gate_row_bytes;
    model->down_expert_bytes = OUT_DIM * model->down_row_bytes;
    model->gate_offset = 0u;
    model->up_offset = MODEL_EXPERTS * model->gate_expert_bytes;
    model->down_offset = model->up_offset + MODEL_EXPERTS * model->gate_expert_bytes;
    model->size = model->down_offset + MODEL_EXPERTS * model->down_expert_bytes;
    model->bytes = calloc((size_t)model->size, 1u);
    if (!model->bytes) return 0;

    for (uint32_t row = 0; row < QK_K; ++row) {
        store_u16(model->bytes + model->gate_offset + (uint64_t)row * IQ2_BLOCK_BYTES,
                  0u, 0x1800u);
        store_u16(model->bytes + model->up_offset + (uint64_t)row * IQ2_BLOCK_BYTES,
                  0u, 0x1800u);
    }
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        uint8_t *block = model->bytes + model->down_offset +
                         (uint64_t)row * Q2_BLOCK_BYTES;
        for (uint32_t index = 0; index < 16u; ++index) block[index] = 0x02u;
        for (uint32_t index = 16u; index < 80u; ++index) block[index] = 0xffu;
        store_u16(block, 80u, 0x1c00u);
    }
    return 1;
}

static void free_model(struct moe_model *model) {
    free(model->bytes);
    memset(model, 0, sizeof(*model));
}

static int init_tensors(struct moe_tensors *values) {
    const uint64_t mid_bytes = (uint64_t)N_TOKENS * N_EXPERT * QK_K * sizeof(float);
    const uint64_t out_bytes = (uint64_t)N_TOKENS * OUT_DIM * sizeof(float);
    memset(values, 0, sizeof(*values));
    values->out = ds4_gpu_tensor_alloc(out_bytes);
    values->gate = ds4_gpu_tensor_alloc(mid_bytes);
    values->up = ds4_gpu_tensor_alloc(mid_bytes);
    values->mid = ds4_gpu_tensor_alloc(mid_bytes);
    values->down = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * N_EXPERT * OUT_DIM * sizeof(float));
    values->selected = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * N_EXPERT * sizeof(int32_t));
    values->weights = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * N_EXPERT * sizeof(float));
    values->x = ds4_gpu_tensor_alloc((uint64_t)N_TOKENS * QK_K * sizeof(float));
    return values->out && values->gate && values->up && values->mid && values->down &&
           values->selected && values->weights && values->x;
}

static void free_tensors(struct moe_tensors *values) {
    ds4_gpu_tensor_free(values->x);
    ds4_gpu_tensor_free(values->weights);
    ds4_gpu_tensor_free(values->selected);
    ds4_gpu_tensor_free(values->down);
    ds4_gpu_tensor_free(values->mid);
    ds4_gpu_tensor_free(values->up);
    ds4_gpu_tensor_free(values->gate);
    ds4_gpu_tensor_free(values->out);
    memset(values, 0, sizeof(*values));
}

static int seed_tensors(struct moe_tensors *values) {
    const int32_t selected[N_TOKENS * N_EXPERT] = {0, 0};
    const float weights[N_TOKENS * N_EXPERT] = {0.75f, 0.25f};
    float x[N_TOKENS * QK_K];
    for (uint32_t index = 0; index < N_TOKENS * QK_K; ++index) {
        x[index] = 0.01f + (float)(index % 19u) * 0.002f;
    }
    x[17] = 0.75f;
    x[QK_K + 29u] = 0.63f;
    return ds4_gpu_tensor_write(values->selected, 0, selected, sizeof(selected)) &&
           ds4_gpu_tensor_write(values->weights, 0, weights, sizeof(weights)) &&
           ds4_gpu_tensor_write(values->x, 0, x, sizeof(x));
}

static int invoke(
        struct moe_tensors *values,
        const struct moe_model *model,
        uint64_t model_size,
        uint32_t n_tokens,
        bool *mid_is_f16) {
    return ds4_gpu_routed_moe_batch_tensor(
            values->out, values->gate, values->up, values->mid, values->down,
            model->bytes, model_size, model->gate_offset, model->up_offset,
            model->down_offset, 16u, 10u, model->gate_expert_bytes,
            model->gate_row_bytes, model->down_expert_bytes, model->down_row_bytes,
            QK_K, QK_K, OUT_DIM, values->selected, values->weights, N_EXPERT,
            0.05f, values->x, n_tokens, mid_is_f16);
}

static int nonzero_finite(const float *values, uint32_t count) {
    int nonzero = 0;
    for (uint32_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
        if (fabsf(values[index]) > 1.0e-7f) nonzero = 1;
    }
    return nonzero;
}

int main(void) {
    struct moe_model model;
    struct moe_tensors values;
    float output[N_TOKENS * OUT_DIM];
    bool success_mid_is_f16 = true;
    bool rejected_mid_is_f16 = true;

    if (!init_model(&model) || !ds4_gpu_init() || !init_tensors(&values) ||
        !seed_tensors(&values) || !ds4_gpu_set_model_map(model.bytes, model.size)) {
        return 1;
    }

    if (!invoke(&values, &model, model.size, N_TOKENS, &success_mid_is_f16) ||
        success_mid_is_f16 ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(values.out, 0, output, sizeof(output)) ||
        !nonzero_finite(output, N_TOKENS * OUT_DIM)) {
        return 2;
    }

    if (!invoke(&values, &model, model.size, N_TOKENS, NULL) ||
        !ds4_gpu_synchronize()) {
        return 3;
    }

    if (invoke(&values, &model, model.size - 1u, N_TOKENS, &rejected_mid_is_f16) ||
        !rejected_mid_is_f16 ||
        invoke(&values, &model, model.size, 0u, &rejected_mid_is_f16) ||
        !rejected_mid_is_f16) {
        return 4;
    }

    free_tensors(&values);
    ds4_gpu_cleanup();
    free_model(&model);
    puts("{\"c_linked_original_cuda\":true,\"successful_batch_reports_f32_mid\":true,"
         "\"successful_null_result_pointer_accepted\":true,"
         "\"invalid_batch_preserves_mid_precision_result\":true,"
         "\"batched_output_nonzero\":true}");
    return 0;
}
