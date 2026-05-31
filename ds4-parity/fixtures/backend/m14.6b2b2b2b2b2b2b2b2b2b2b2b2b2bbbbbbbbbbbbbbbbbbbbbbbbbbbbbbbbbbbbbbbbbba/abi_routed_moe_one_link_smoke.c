#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256u
#define MODEL_EXPERTS 256u
#define N_EXPERT 6u
#define OUT_DIM 5u
#define IQ2_BLOCK_BYTES 66u
#define Q2_BLOCK_BYTES 84u
#define Q4_BLOCK_BYTES 144u
#define Q8_K_BYTES 292u
#define AUX_ELEMENTS (N_EXPERT * QK_K)
#define FLOAT_TOLERANCE 2.0e-3f

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
    uint32_t gate_type;
    uint32_t down_type;
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
    uint32_t n_expert;
    uint64_t aux_bytes;
    uint64_t down_bytes;
};

static void store_u16(uint8_t *values, uint64_t offset, uint16_t value) {
    values[offset] = (uint8_t)value;
    values[offset + 1u] = (uint8_t)(value >> 8);
}

static void fill_iq2(uint8_t *weights, uint64_t expert_bytes) {
    for (uint32_t expert = 0; expert < 4u; ++expert) {
        for (uint32_t row = 0; row < QK_K; ++row) {
            uint8_t *block = weights + (uint64_t)expert * expert_bytes +
                             (uint64_t)row * IQ2_BLOCK_BYTES;
            store_u16(block, 0u, 0x1800u);
        }
    }
}

static void fill_q2(uint8_t *weights, uint64_t expert_bytes) {
    for (uint32_t expert = 0; expert < 4u; ++expert) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            uint8_t *block = weights + (uint64_t)expert * expert_bytes +
                             (uint64_t)row * Q2_BLOCK_BYTES;
            for (uint32_t index = 0; index < 16u; ++index) block[index] = 0x02u;
            for (uint32_t index = 16u; index < 80u; ++index) block[index] = 0xffu;
            store_u16(block, 80u, 0x1c00u);
        }
    }
}

static void fill_q4(uint8_t *weights, uint64_t expert_bytes, uint32_t rows) {
    for (uint32_t expert = 0; expert < 4u; ++expert) {
        for (uint32_t row = 0; row < rows; ++row) {
            uint8_t *block = weights + (uint64_t)expert * expert_bytes +
                             (uint64_t)row * Q4_BLOCK_BYTES;
            store_u16(block, 0u, 0x1c00u);
            store_u16(block, 2u, 0u);
            for (uint32_t index = 4u; index < 16u; ++index) block[index] = 0x01u;
            for (uint32_t index = 16u; index < Q4_BLOCK_BYTES; ++index) block[index] = 0x11u;
        }
    }
}

static int init_model(struct moe_model *model, int q4_k) {
    memset(model, 0, sizeof(*model));
    model->gate_row_bytes = q4_k ? Q4_BLOCK_BYTES : IQ2_BLOCK_BYTES;
    model->down_row_bytes = q4_k ? Q4_BLOCK_BYTES : Q2_BLOCK_BYTES;
    model->gate_expert_bytes = QK_K * model->gate_row_bytes;
    model->down_expert_bytes = OUT_DIM * model->down_row_bytes;
    model->gate_offset = 0u;
    model->up_offset = MODEL_EXPERTS * model->gate_expert_bytes;
    model->down_offset = model->up_offset + MODEL_EXPERTS * model->gate_expert_bytes;
    model->size = model->down_offset + MODEL_EXPERTS * model->down_expert_bytes;
    model->gate_type = q4_k ? 12u : 16u;
    model->down_type = q4_k ? 12u : 10u;
    model->bytes = calloc((size_t)model->size, 1u);
    if (!model->bytes) return 0;
    if (q4_k) {
        fill_q4(model->bytes + model->gate_offset, model->gate_expert_bytes, QK_K);
        fill_q4(model->bytes + model->up_offset, model->gate_expert_bytes, QK_K);
        fill_q4(model->bytes + model->down_offset, model->down_expert_bytes, OUT_DIM);
    } else {
        fill_iq2(model->bytes + model->gate_offset, model->gate_expert_bytes);
        fill_iq2(model->bytes + model->up_offset, model->gate_expert_bytes);
        fill_q2(model->bytes + model->down_offset, model->down_expert_bytes);
    }
    return 1;
}

static void free_model(struct moe_model *model) {
    free(model->bytes);
    memset(model, 0, sizeof(*model));
}

static int init_tensors(struct moe_tensors *values, uint32_t n_expert, uint64_t down_bytes) {
    memset(values, 0, sizeof(*values));
    values->n_expert = n_expert;
    values->aux_bytes = (uint64_t)n_expert * QK_K * sizeof(float);
    values->down_bytes = down_bytes;
    values->out = ds4_gpu_tensor_alloc(OUT_DIM * sizeof(float));
    values->gate = ds4_gpu_tensor_alloc(values->aux_bytes);
    values->up = ds4_gpu_tensor_alloc(values->aux_bytes);
    values->mid = ds4_gpu_tensor_alloc(values->aux_bytes);
    values->down = ds4_gpu_tensor_alloc(down_bytes);
    values->selected = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(int32_t));
    values->weights = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    values->x = ds4_gpu_tensor_alloc(QK_K * sizeof(float));
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

static int seed_tensors(
        struct moe_tensors *values,
        const int32_t *selected,
        const float *weights,
        const float *x) {
    return ds4_gpu_tensor_write(values->selected, 0, selected,
                                (uint64_t)values->n_expert * sizeof(int32_t)) &&
           ds4_gpu_tensor_write(values->weights, 0, weights,
                                (uint64_t)values->n_expert * sizeof(float)) &&
           ds4_gpu_tensor_write(values->x, 0, x, QK_K * sizeof(float));
}

static int zero_aux(ds4_gpu_tensor *tensor, uint64_t bytes) {
    float zeros[AUX_ELEMENTS] = {0.0f};
    return bytes <= sizeof(zeros) && ds4_gpu_tensor_write(tensor, 0, zeros, bytes);
}

static int invoke(
        struct moe_tensors *values,
        const struct moe_model *model,
        uint64_t model_size,
        uint32_t gate_type,
        uint32_t down_type,
        uint32_t n_expert) {
    return ds4_gpu_routed_moe_one_tensor(
            values->out, values->gate, values->up, values->mid, values->down, model->bytes,
            model_size, model->gate_offset, model->up_offset, model->down_offset, gate_type,
            down_type, model->gate_expert_bytes, model->gate_row_bytes,
            model->down_expert_bytes, model->down_row_bytes, QK_K, QK_K, OUT_DIM,
            values->selected, values->weights, n_expert, 0.05f, values->x);
}

static int close_array(const float *left, const float *right, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        const float scale = fmaxf(1.0f, fabsf(right[index]));
        if (!isfinite(left[index]) || fabsf(left[index] - right[index]) > FLOAT_TOLERANCE * scale) {
            return 0;
        }
    }
    return 1;
}

static int nonzero_finite(const float *values, uint32_t count) {
    int nonzero = 0;
    for (uint32_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
        if (fabsf(values[index]) > 1.0e-7f) nonzero = 1;
    }
    return nonzero;
}

static int all_zero(const float *values, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (values[index] != 0.0f) return 0;
    }
    return 1;
}

static int packed_q8_nonzero(const uint8_t *packed) {
    float scale = 0.0f;
    memcpy(&scale, packed, sizeof(scale));
    if (!isfinite(scale) || scale == 0.0f) return 0;
    for (uint32_t index = 4u; index < 260u; ++index) {
        if (packed[index] != 0u) return 1;
    }
    return 0;
}

static int clear_env(void) {
    return unsetenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == 0 &&
           unsetenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == 0 &&
           unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP") == 0;
}

int main(void) {
    struct moe_model iq2_model;
    struct moe_model q4_model;
    struct moe_tensors quantized;
    struct moe_tensors fallback;
    const int32_t selected[N_EXPERT] = {0, 2, -1, 3, 1, 0};
    const int32_t selected_zero[N_EXPERT] = {0, 2, 0, 3, 1, 0};
    const float route_weights[N_EXPERT] = {0.48f, 0.33f, 0.25f, 0.20f, 0.15f, 0.09f};
    float x[QK_K];
    float default_out[OUT_DIM];
    float generic_gate_out[OUT_DIM];
    float generic_down_out[OUT_DIM];
    float q4_out[OUT_DIM];
    float negative_out[OUT_DIM];
    float zero_out[OUT_DIM];
    float fallback_out[OUT_DIM];
    float up_aux[AUX_ELEMENTS];
    const float sentinel[OUT_DIM] = {77.0f, 77.0f, 77.0f, 77.0f, 77.0f};
    float observed[OUT_DIM];
    uint8_t xq[Q8_K_BYTES];
    uint8_t midq[Q8_K_BYTES];
    ds4_gpu_tensor *short_selected = NULL;

    for (uint32_t index = 0; index < QK_K; ++index) {
        x[index] = 0.01f + (float)(index % 17u) * 0.001f;
    }
    x[17] = 0.75f;

    if (!init_model(&iq2_model, 0) || !init_model(&q4_model, 1) ||
        !ds4_gpu_init() || !clear_env() ||
        !init_tensors(&quantized, N_EXPERT, Q8_K_BYTES) ||
        !init_tensors(&fallback, 1u, OUT_DIM * sizeof(float)) ||
        !(short_selected = ds4_gpu_tensor_alloc((N_EXPERT - 1u) * sizeof(int32_t))) ||
        !seed_tensors(&quantized, selected, route_weights, x) ||
        !seed_tensors(&fallback, selected, route_weights, x) ||
        !ds4_gpu_set_model_map(iq2_model.bytes, iq2_model.size)) {
        return 1;
    }

    if (!zero_aux(quantized.up, quantized.aux_bytes) ||
        !invoke(&quantized, &iq2_model, iq2_model.size, 16u, 10u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, default_out, sizeof(default_out)) ||
        !ds4_gpu_tensor_read(quantized.up, 0, up_aux, sizeof(up_aux)) ||
        !ds4_gpu_tensor_read(quantized.down, 0, xq, sizeof(xq)) ||
        !ds4_gpu_tensor_read(quantized.gate, 0, midq, sizeof(midq)) ||
        !nonzero_finite(default_out, OUT_DIM) || !all_zero(up_aux, AUX_ELEMENTS) ||
        !packed_q8_nonzero(xq) || !packed_q8_nonzero(midq)) {
        return 2;
    }

    if (setenv("DS4_CUDA_MOE_WRITE_GATE_UP", "1", 1) != 0 ||
        !zero_aux(quantized.up, quantized.aux_bytes) ||
        !invoke(&quantized, &iq2_model, iq2_model.size, 16u, 10u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, observed, sizeof(observed)) ||
        !ds4_gpu_tensor_read(quantized.up, 0, up_aux, sizeof(up_aux)) ||
        !close_array(observed, default_out, OUT_DIM) || !nonzero_finite(up_aux, AUX_ELEMENTS)) {
        return 3;
    }

    if (!clear_env() || setenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE", "1", 1) != 0 ||
        !zero_aux(quantized.up, quantized.aux_bytes) ||
        !invoke(&quantized, &iq2_model, iq2_model.size, 16u, 10u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, generic_gate_out, sizeof(generic_gate_out)) ||
        !ds4_gpu_tensor_read(quantized.up, 0, up_aux, sizeof(up_aux)) ||
        !close_array(generic_gate_out, default_out, OUT_DIM) || !nonzero_finite(up_aux, AUX_ELEMENTS)) {
        return 4;
    }

    if (!clear_env() || setenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6", "1", 1) != 0 ||
        !invoke(&quantized, &iq2_model, iq2_model.size, 16u, 10u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, generic_down_out, sizeof(generic_down_out)) ||
        !close_array(generic_down_out, default_out, OUT_DIM)) {
        return 5;
    }

    if (!clear_env() || !seed_tensors(&quantized, selected_zero, route_weights, x) ||
        !invoke(&quantized, &iq2_model, iq2_model.size, 16u, 10u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, zero_out, sizeof(zero_out)) ||
        !seed_tensors(&quantized, selected, route_weights, x) ||
        !invoke(&quantized, &iq2_model, iq2_model.size, 16u, 10u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, negative_out, sizeof(negative_out)) ||
        !close_array(negative_out, zero_out, OUT_DIM)) {
        return 6;
    }

    if (!invoke(&fallback, &iq2_model, iq2_model.size, 16u, 10u, 1u) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(fallback.out, 0, fallback_out, sizeof(fallback_out)) ||
        !nonzero_finite(fallback_out, OUT_DIM)) {
        return 7;
    }

    if (!ds4_gpu_set_model_map(q4_model.bytes, q4_model.size) ||
        !invoke(&quantized, &q4_model, q4_model.size, 12u, 12u, N_EXPERT) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, q4_out, sizeof(q4_out)) ||
        !nonzero_finite(q4_out, OUT_DIM)) {
        return 8;
    }

    if (!ds4_gpu_tensor_write(quantized.out, 0, sentinel, sizeof(sentinel)) ||
        invoke(&quantized, &q4_model, q4_model.size - 1u, 12u, 12u, N_EXPERT) ||
        invoke(&quantized, &q4_model, q4_model.size, 16u, 12u, N_EXPERT) ||
        invoke(&quantized, &q4_model, q4_model.size, 12u, 12u, N_EXPERT - 1u) ||
        ds4_gpu_routed_moe_one_tensor(
                quantized.out, quantized.gate, quantized.up, quantized.mid, quantized.down,
                q4_model.bytes, q4_model.size, q4_model.gate_offset, q4_model.up_offset,
                q4_model.down_offset, 12u, 12u, q4_model.gate_expert_bytes,
                q4_model.gate_row_bytes, q4_model.down_expert_bytes, q4_model.down_row_bytes,
                QK_K, QK_K, OUT_DIM, short_selected, quantized.weights, N_EXPERT, 0.05f,
                quantized.x) ||
        ds4_gpu_routed_moe_one_tensor(
                NULL, quantized.gate, quantized.up, quantized.mid, quantized.down,
                q4_model.bytes, q4_model.size, q4_model.gate_offset, q4_model.up_offset,
                q4_model.down_offset, 12u, 12u, q4_model.gate_expert_bytes,
                q4_model.gate_row_bytes, q4_model.down_expert_bytes, q4_model.down_row_bytes,
                QK_K, QK_K, OUT_DIM, quantized.selected, quantized.weights, N_EXPERT, 0.05f,
                quantized.x) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(quantized.out, 0, observed, sizeof(observed)) ||
        !close_array(observed, sentinel, OUT_DIM)) {
        return 9;
    }

    ds4_gpu_tensor_free(short_selected);
    free_tensors(&fallback);
    free_tensors(&quantized);
    ds4_gpu_cleanup();
    free_model(&q4_model);
    free_model(&iq2_model);
    puts("{\"c_linked_rust_staticlib\":true,\"f32_fallback_nonzero\":true,"
         "\"default_iq2_q2_direct_sum_nonzero\":true,\"packed_input_q8_alias_visible\":true,"
         "\"packed_mid_q8_alias_visible\":true,\"default_aux_unwritten\":true,"
         "\"optional_gate_up_write_visible\":true,\"forced_generic_gate_matches\":true,"
         "\"forced_generic_down_matches\":true,\"q4_k_direct_sum_nonzero\":true,"
         "\"negative_expert_fallback_matches\":true,\"invalid_model_range_preserves_output\":true,"
         "\"invalid_type_rejected\":true,\"invalid_q4_group_rejected\":true,\"short_span_rejected\":true,"
         "\"null_rejected\":true,\"embedded_routed_moe_kernels_loaded\":true}");
    return 0;
}
