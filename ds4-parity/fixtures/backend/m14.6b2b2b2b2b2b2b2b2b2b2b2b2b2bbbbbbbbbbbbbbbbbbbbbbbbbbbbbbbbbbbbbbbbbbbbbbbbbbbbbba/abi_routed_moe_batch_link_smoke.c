#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256u
#define MODEL_EXPERTS 256u
#define N_TOKENS 128u
#define N_EXPERT 1u
#define OUT_DIM 5u
#define IQ2_BLOCK_BYTES 66u
#define Q2_BLOCK_BYTES 84u
#define Q8_K_BYTES 292u

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
        uint8_t *block = model->bytes + model->down_offset + (uint64_t)row * Q2_BLOCK_BYTES;
        for (uint32_t index = 0; index < 16u; ++index) block[index] = 0x02u;
        for (uint32_t index = 16u; index < 80u; ++index) block[index] = 0xffu;
        store_u16(block, 80u, 0x1c00u);
    }
    return 1;
}

static int init_tensors(struct moe_tensors *values, int quantized) {
    const uint64_t pair_count = (uint64_t)N_TOKENS * N_EXPERT;
    const uint64_t mid_bytes = pair_count * QK_K * sizeof(float);
    const uint64_t out_bytes = (uint64_t)N_TOKENS * OUT_DIM * sizeof(float);
    const uint64_t down_f32_bytes = pair_count * OUT_DIM * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)N_TOKENS * Q8_K_BYTES;
    memset(values, 0, sizeof(*values));
    values->out = ds4_gpu_tensor_alloc(out_bytes);
    values->gate = ds4_gpu_tensor_alloc(mid_bytes);
    values->up = ds4_gpu_tensor_alloc(mid_bytes);
    values->mid = ds4_gpu_tensor_alloc(mid_bytes);
    values->down = ds4_gpu_tensor_alloc(quantized ? xq_bytes : down_f32_bytes);
    values->selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
    values->weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
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
    int32_t selected[N_TOKENS * N_EXPERT];
    float weights[N_TOKENS * N_EXPERT];
    float x[N_TOKENS * QK_K];
    for (uint32_t token = 0; token < N_TOKENS; ++token) {
        selected[token] = 0;
        weights[token] = 0.25f + (float)(token % 4u) * 0.125f;
        for (uint32_t column = 0; column < QK_K; ++column) {
            x[(uint64_t)token * QK_K + column] =
                0.01f + (float)((token + column) % 19u) * 0.002f;
        }
    }
    x[17] = 0.75f;
    x[(uint64_t)(N_TOKENS - 1u) * QK_K + 29u] = 0.63f;
    return ds4_gpu_tensor_write(values->selected, 0, selected, sizeof(selected)) &&
           ds4_gpu_tensor_write(values->weights, 0, weights, sizeof(weights)) &&
           ds4_gpu_tensor_write(values->x, 0, x, sizeof(x));
}

static int clear_route_env(void) {
    static const char *names[] = {
        "DS4_CUDA_MOE_NO_EXPERT_TILES", "DS4_CUDA_MOE_TILE4",
        "DS4_CUDA_MOE_NO_P2", "DS4_CUDA_MOE_ATOMIC_DOWN",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN", "DS4_CUDA_MOE_GATE_ROW2048",
        "DS4_CUDA_MOE_GATE_ROW512", "DS4_CUDA_MOE_NO_GATE_ROW2048",
        "DS4_CUDA_MOE_NO_GATE_ROW256", "DS4_CUDA_MOE_NO_GATE_ROW128",
        "DS4_CUDA_MOE_NO_DOWN_TILE16", "DS4_CUDA_MOE_DOWN_ROW2048",
        "DS4_CUDA_MOE_DOWN_ROW512", "DS4_CUDA_MOE_DOWN_ROW1024",
        "DS4_CUDA_MOE_NO_DOWN_ROW2048", "DS4_CUDA_MOE_NO_DOWN_ROW256",
        "DS4_CUDA_MOE_NO_DOWN_ROW128", "DS4_CUDA_MOE_NO_DOWN_ROW64",
    };
    for (uint32_t index = 0; index < sizeof(names) / sizeof(names[0]); ++index) {
        if (unsetenv(names[index]) != 0) return 0;
    }
    return 1;
}

static int invoke(
        struct moe_tensors *values,
        const struct moe_model *model,
        uint32_t n_tokens,
        bool *mid_is_f16) {
    return ds4_gpu_routed_moe_batch_tensor(
            values->out, values->gate, values->up, values->mid, values->down,
            model->bytes, model->size, model->gate_offset, model->up_offset,
            model->down_offset, 16u, 10u, model->gate_expert_bytes,
            model->gate_row_bytes, model->down_expert_bytes, model->down_row_bytes,
            QK_K, QK_K, OUT_DIM, values->selected, values->weights, N_EXPERT,
            0.05f, values->x, n_tokens, mid_is_f16);
}

static int invoke_nonzero(struct moe_tensors *values, const struct moe_model *model) {
    float output[N_TOKENS * OUT_DIM];
    bool mid_is_f16 = true;
    if (!invoke(values, model, N_TOKENS, &mid_is_f16) || mid_is_f16 ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(values->out, 0, output, sizeof(output))) {
        return 0;
    }
    for (uint32_t index = 0; index < N_TOKENS * OUT_DIM; ++index) {
        if (!isfinite(output[index])) return 0;
        if (fabsf(output[index]) > 1.0e-7f) return 1;
    }
    return 0;
}

int main(void) {
    struct moe_model model;
    struct moe_tensors fallback;
    struct moe_tensors quantized;
    bool rejected_mid_is_f16 = true;
    if (!init_model(&model) || !ds4_gpu_init() || !init_tensors(&fallback, 0) ||
        !init_tensors(&quantized, 1) || !seed_tensors(&fallback) ||
        !seed_tensors(&quantized) || !ds4_gpu_set_model_map(model.bytes, model.size) ||
        !clear_route_env()) {
        return 1;
    }
    if (!invoke_nonzero(&fallback, &model)) return 2;
    if (setenv("DS4_CUDA_MOE_NO_EXPERT_TILES", "1", 1) != 0 ||
        !invoke_nonzero(&quantized, &model) ||
        setenv("DS4_CUDA_MOE_NO_P2", "1", 1) != 0 ||
        !invoke_nonzero(&quantized, &model)) {
        return 3;
    }
    if (!clear_route_env() ||
        setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NO_GATE_ROW2048", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NO_GATE_ROW256", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NO_GATE_ROW128", "1", 1) != 0 ||
        !invoke_nonzero(&quantized, &model)) {
        return 4;
    }
    if (!clear_route_env() ||
        setenv("DS4_CUDA_MOE_TILE4", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_ATOMIC_DOWN", "1", 1) != 0 ||
        !invoke_nonzero(&quantized, &model)) {
        return 5;
    }
    if (!clear_route_env() || !invoke_nonzero(&quantized, &model) ||
        !invoke(&quantized, &model, N_TOKENS, NULL) || !ds4_gpu_synchronize() ||
        invoke(&quantized, &model, 0u, &rejected_mid_is_f16) ||
        !rejected_mid_is_f16) {
        return 6;
    }
    free_tensors(&quantized);
    free_tensors(&fallback);
    ds4_gpu_cleanup();
    free(model.bytes);
    puts("{\"c_linked_rust_staticlib\":true,\"f32_fallback\":true,"
         "\"sorted_p2\":true,\"sorted_no_p2\":true,\"tiled_row32\":true,"
         "\"tile4_atomic\":true,\"tile16_rowspan_atomic\":true,"
         "\"mid_is_f16_false_on_success\":true,"
         "\"failed_batch_preserves_mid_result\":true}");
    return 0;
}
