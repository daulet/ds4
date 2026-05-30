#include "ds4_gpu.h"

#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q8_IN_DIM 4096u
#define Q8_OUT_DIM 2048u
#define Q8_BLOCKS (Q8_IN_DIM / 32u)
#define Q8_WEIGHT_BYTES ((uint64_t)Q8_OUT_DIM * Q8_BLOCKS * 34u)
#define Q8_RANGE_COUNT 3u
#define Q8_MODEL_BYTES (Q8_RANGE_COUNT * Q8_WEIGHT_BYTES)
#define Q8_F16_BYTES ((uint64_t)Q8_IN_DIM * Q8_OUT_DIM * sizeof(uint16_t))
#define Q8_F32_BYTES ((uint64_t)Q8_IN_DIM * Q8_OUT_DIM * sizeof(float))

#define BLAS_IN_DIM 1024u
#define BLAS_OUT_DIM 4u
#define BLAS_N_TOK 2u
#define BLAS_WEIGHT_BYTES ((uint64_t)BLAS_IN_DIM * BLAS_OUT_DIM * sizeof(float))

static int memory_free(size_t *free_bytes) {
    size_t total_bytes = 0;
    return cuMemGetInfo(free_bytes, &total_bytes) == CUDA_SUCCESS;
}

static int allocation_observed(size_t before, size_t after, size_t minimum) {
    return before > after && before - after >= minimum;
}

static int memory_stable(size_t before, size_t after) {
    const size_t tolerance = 1024u * 1024u;
    return before >= after ? before - after <= tolerance : after - before <= tolerance;
}

static void fill_q8_model(uint8_t *model) {
    for (uint32_t range = 0; range < Q8_RANGE_COUNT; ++range) {
        for (uint32_t row = 0; row < Q8_OUT_DIM; ++row) {
            for (uint32_t block = 0; block < Q8_BLOCKS; ++block) {
                uint64_t base = (uint64_t)range * Q8_WEIGHT_BYTES +
                                ((uint64_t)row * Q8_BLOCKS + block) * 34u;
                model[base] = 0x00u;
                model[base + 1u] = 0x3cu;
                for (uint32_t lane = 0; lane < 32u; ++lane) {
                    model[base + 2u + lane] = (uint8_t)(1 + (lane % 7u));
                }
            }
        }
    }
}

static void reference_projection(float *out, const float *weights, const float *x) {
    for (uint32_t token = 0; token < BLAS_N_TOK; ++token) {
        for (uint32_t row = 0; row < BLAS_OUT_DIM; ++row) {
            float total = 0.0f;
            for (uint32_t column = 0; column < BLAS_IN_DIM; ++column) {
                total += weights[(uint64_t)row * BLAS_IN_DIM + column] *
                         x[(uint64_t)token * BLAS_IN_DIM + column];
            }
            out[(uint64_t)token * BLAS_OUT_DIM + row] = total;
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-1f) return 0;
    }
    return 1;
}

static int differs(const float *left, const float *right, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(left[i] - right[i]) > 1.0e-2f) return 1;
    }
    return 0;
}

static int run_blas_projection(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *x,
        const float *model,
        float *got) {
    if (!ds4_gpu_matmul_f32_tensor(
            out, model, BLAS_WEIGHT_BYTES, 0, BLAS_IN_DIM, BLAS_OUT_DIM, x, BLAS_N_TOK)) {
        return 0;
    }
    if (!ds4_gpu_synchronize()) return 0;
    return ds4_gpu_tensor_read(out, 0, got, sizeof(float) * BLAS_N_TOK * BLAS_OUT_DIM);
}

int main(void) {
    uint8_t *q8_model = calloc((size_t)Q8_MODEL_BYTES, 1);
    float *blas_model = malloc((size_t)BLAS_WEIGHT_BYTES);
    float *blas_x = malloc(sizeof(float) * BLAS_IN_DIM * BLAS_N_TOK);
    if (!q8_model || !blas_model || !blas_x) return 1;
    fill_q8_model(q8_model);
    for (uint32_t i = 0; i < BLAS_IN_DIM * BLAS_OUT_DIM; ++i) {
        blas_model[i] = 1.0003f;
    }
    for (uint32_t i = 0; i < BLAS_IN_DIM * BLAS_N_TOK; ++i) {
        blas_x[i] = 1.0003f;
    }

    if (unsetenv("DS4_CUDA_NO_TF32") != 0 ||
        unsetenv("DS4_CUDA_Q8_F32_PRELOAD") != 0 ||
        unsetenv("DS4_CUDA_Q8_F32_ALL") != 0 ||
        setenv("DS4_CUDA_Q8_F16_ALL", "1", 1) != 0 ||
        setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "0", 1) != 0) {
        return 2;
    }
    if (!ds4_gpu_init()) return 3;
    ds4_gpu_set_quality(false);
    if (!ds4_gpu_set_model_map(q8_model, Q8_MODEL_BYTES)) return 4;

    size_t before_f16 = 0;
    size_t after_f16 = 0;
    size_t after_repeat = 0;
    size_t after_quality_suppressed = 0;
    size_t after_quality_reenabled = 0;
    size_t after_f32 = 0;
    if (!memory_free(&before_f16) ||
        !ds4_gpu_cache_q8_f16_range(
            q8_model, Q8_MODEL_BYTES, 0, Q8_WEIGHT_BYTES, Q8_IN_DIM, Q8_OUT_DIM, "range0") ||
        !ds4_gpu_synchronize() ||
        !memory_free(&after_f16) ||
        !allocation_observed(before_f16, after_f16, (size_t)(Q8_F16_BYTES / 2u))) {
        return 5;
    }
    if (!ds4_gpu_cache_q8_f16_range(
            q8_model, Q8_MODEL_BYTES, 0, Q8_WEIGHT_BYTES, Q8_IN_DIM, Q8_OUT_DIM, "range0") ||
        !ds4_gpu_synchronize() ||
        !memory_free(&after_repeat) ||
        !memory_stable(after_f16, after_repeat)) {
        return 6;
    }

    ds4_gpu_set_quality(true);
    if (!ds4_gpu_cache_q8_f16_range(
            q8_model,
            Q8_MODEL_BYTES,
            Q8_WEIGHT_BYTES,
            Q8_WEIGHT_BYTES,
            Q8_IN_DIM,
            Q8_OUT_DIM,
            "range1") ||
        !ds4_gpu_synchronize() ||
        !memory_free(&after_quality_suppressed) ||
        !memory_stable(after_repeat, after_quality_suppressed)) {
        return 7;
    }
    ds4_gpu_set_quality(false);
    if (!ds4_gpu_cache_q8_f16_range(
            q8_model,
            Q8_MODEL_BYTES,
            Q8_WEIGHT_BYTES,
            Q8_WEIGHT_BYTES,
            Q8_IN_DIM,
            Q8_OUT_DIM,
            "range1") ||
        !ds4_gpu_synchronize() ||
        !memory_free(&after_quality_reenabled) ||
        !allocation_observed(
            after_quality_suppressed, after_quality_reenabled, (size_t)(Q8_F16_BYTES / 2u))) {
        return 8;
    }

    if (setenv("DS4_CUDA_Q8_F32_PRELOAD", "1", 1) != 0 ||
        setenv("DS4_CUDA_Q8_F32_ALL", "1", 1) != 0 ||
        !ds4_gpu_cache_q8_f16_range(
            q8_model,
            Q8_MODEL_BYTES,
            2u * Q8_WEIGHT_BYTES,
            Q8_WEIGHT_BYTES,
            Q8_IN_DIM,
            Q8_OUT_DIM,
            "range2") ||
        !ds4_gpu_synchronize() ||
        !memory_free(&after_f32) ||
        !allocation_observed(after_quality_reenabled, after_f32, (size_t)(Q8_F32_BYTES / 2u))) {
        return 9;
    }
    ds4_gpu_print_memory_report("q8-quality-controls");
    if (ds4_gpu_cache_q8_f16_range(q8_model, Q8_WEIGHT_BYTES, Q8_WEIGHT_BYTES, 1, 1, 1, "bad") ||
        !ds4_gpu_cache_q8_f16_range(NULL, 0, 0, 1, 1, 1, "null")) {
        return 10;
    }

    if (!ds4_gpu_set_model_map(blas_model, BLAS_WEIGHT_BYTES)) return 11;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(float) * BLAS_IN_DIM * BLAS_N_TOK);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float) * BLAS_OUT_DIM * BLAS_N_TOK);
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, blas_x, sizeof(float) * BLAS_IN_DIM * BLAS_N_TOK)) {
        return 12;
    }
    float expected[BLAS_N_TOK * BLAS_OUT_DIM] = {0};
    float fast[BLAS_N_TOK * BLAS_OUT_DIM] = {0};
    float quality[BLAS_N_TOK * BLAS_OUT_DIM] = {0};
    float no_tf32[BLAS_N_TOK * BLAS_OUT_DIM] = {0};
    reference_projection(expected, blas_model, blas_x);
    ds4_gpu_set_quality(false);
    if (!run_blas_projection(out, x, blas_model, fast)) return 13;
    ds4_gpu_set_quality(true);
    if (!run_blas_projection(out, x, blas_model, quality) ||
        !close_array(quality, expected, BLAS_N_TOK * BLAS_OUT_DIM) ||
        !differs(fast, quality, BLAS_N_TOK * BLAS_OUT_DIM)) {
        return 14;
    }
    if (setenv("DS4_CUDA_NO_TF32", "1", 1) != 0) return 15;
    ds4_gpu_set_quality(false);
    if (!run_blas_projection(out, x, blas_model, no_tf32) ||
        !close_array(no_tf32, expected, BLAS_N_TOK * BLAS_OUT_DIM) ||
        differs(no_tf32, quality, BLAS_N_TOK * BLAS_OUT_DIM)) {
        return 16;
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(blas_x);
    free(blas_model);
    free(q8_model);
    puts("{\"c_linked_rust_staticlib\":true,\"q8_f16_preload_allocation_observed\":true,\"q8_f16_exact_cache_reuse_observed\":true,\"quality_suppresses_new_q8_f16_preload\":true,\"quality_disable_reenables_q8_f16_preload\":true,\"q8_f32_optional_preload_allocation_observed\":true,\"memory_report_callable\":true,\"quality_math_mutation_changes_f32_blas_output\":true,\"no_tf32_quality_update_uses_default_math\":true,\"invalid_range_rejected\":true,\"null_optional_cache_accepted\":true,\"embedded_q8_dequant_kernels_loaded\":true}");
    return 0;
}
