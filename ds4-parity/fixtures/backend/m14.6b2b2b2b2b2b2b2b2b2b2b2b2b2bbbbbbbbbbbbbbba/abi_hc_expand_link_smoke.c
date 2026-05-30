#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N_EMBD 3u
#define N_HC 2u
#define N_TOK 2u
#define MIX_HC (2u * N_HC + N_HC * N_HC)
#define BLOCK_ELEMENTS ((uint64_t)N_TOK * N_EMBD)
#define HC_ELEMENTS ((uint64_t)N_TOK * N_HC * N_EMBD)
#define SPLIT_ELEMENTS ((uint64_t)N_TOK * MIX_HC)

static void reference_expand(
        float out[HC_ELEMENTS],
        const float block[BLOCK_ELEMENTS],
        const float add[BLOCK_ELEMENTS],
        const float residual[HC_ELEMENTS],
        const float post[N_TOK * N_HC],
        const float comb[N_TOK * N_HC * N_HC],
        int has_add) {
    for (uint32_t token = 0; token < N_TOK; ++token) {
        for (uint32_t destination = 0; destination < N_HC; ++destination) {
            for (uint32_t dimension = 0; dimension < N_EMBD; ++dimension) {
                const uint64_t block_index = (uint64_t)token * N_EMBD + dimension;
                float block_value = block[block_index];
                if (has_add) block_value += add[block_index];
                float total = block_value * post[(uint64_t)token * N_HC + destination];
                for (uint32_t source = 0; source < N_HC; ++source) {
                    total += comb[(uint64_t)token * N_HC * N_HC +
                                  (uint64_t)source * N_HC + destination] *
                             residual[(uint64_t)token * N_HC * N_EMBD +
                                      (uint64_t)source * N_EMBD + dimension];
                }
                out[(uint64_t)token * N_HC * N_EMBD +
                    (uint64_t)destination * N_EMBD + dimension] = total;
            }
        }
    }
}

static int close_array(const float *actual, const float *expected, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    const float block[BLOCK_ELEMENTS] = {0.25f, -0.5f, 1.0f, 0.75f, -1.25f, 0.5f};
    const float add[BLOCK_ELEMENTS] = {-0.1f, 0.2f, 0.4f, -0.6f, 0.3f, 0.1f};
    const float residual[HC_ELEMENTS] = {
        0.2f, -0.3f, 0.4f, -0.7f, 0.8f, 0.1f,
        0.5f, 0.6f, -0.2f, 0.9f, -0.4f, 0.3f,
    };
    const float post[N_TOK * N_HC] = {0.5f, 1.25f, -0.75f, 0.8f};
    const float comb[N_TOK * N_HC * N_HC] = {
        1.0f, -0.25f, 0.4f, 0.9f,
        -0.2f, 0.7f, 1.1f, -0.5f,
    };
    const float split[SPLIT_ELEMENTS] = {
        0.0f, 0.0f, 0.5f, 1.25f, 1.0f, -0.25f, 0.4f, 0.9f,
        0.0f, 0.0f, -0.75f, 0.8f, -0.2f, 0.7f, 1.1f, -0.5f,
    };
    float expected_plain[HC_ELEMENTS] = {0};
    float expected_add[HC_ELEMENTS] = {0};
    float expected_alias_add[HC_ELEMENTS] = {0};
    float direct[HC_ELEMENTS] = {0};
    float split_plain[HC_ELEMENTS] = {0};
    float split_add[HC_ELEMENTS] = {0};
    float alias_add[HC_ELEMENTS] = {0};
    reference_expand(expected_plain, block, add, residual, post, comb, 0);
    reference_expand(expected_add, block, add, residual, post, comb, 1);
    reference_expand(expected_alias_add, block, block, residual, post, comb, 1);

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(direct));
    ds4_gpu_tensor *block_tensor = ds4_gpu_tensor_alloc(sizeof(block));
    ds4_gpu_tensor *add_tensor = ds4_gpu_tensor_alloc(sizeof(add));
    ds4_gpu_tensor *residual_tensor = ds4_gpu_tensor_alloc(sizeof(residual));
    ds4_gpu_tensor *post_tensor = ds4_gpu_tensor_alloc(sizeof(post));
    ds4_gpu_tensor *comb_tensor = ds4_gpu_tensor_alloc(sizeof(comb));
    ds4_gpu_tensor *split_tensor = ds4_gpu_tensor_alloc(sizeof(split));
    if (!out || !block_tensor || !add_tensor || !residual_tensor ||
        !post_tensor || !comb_tensor || !split_tensor) {
        return 2;
    }
    if (!ds4_gpu_tensor_write(block_tensor, 0, block, sizeof(block)) ||
        !ds4_gpu_tensor_write(add_tensor, 0, add, sizeof(add)) ||
        !ds4_gpu_tensor_write(residual_tensor, 0, residual, sizeof(residual)) ||
        !ds4_gpu_tensor_write(post_tensor, 0, post, sizeof(post)) ||
        !ds4_gpu_tensor_write(comb_tensor, 0, comb, sizeof(comb)) ||
        !ds4_gpu_tensor_write(split_tensor, 0, split, sizeof(split))) {
        return 3;
    }
    if (!ds4_gpu_hc_expand_tensor(out, block_tensor, residual_tensor,
                                  post_tensor, comb_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out, 0, direct, sizeof(direct)) ||
        !close_array(direct, expected_plain, HC_ELEMENTS)) {
        return 4;
    }
    if (!ds4_gpu_hc_expand_split_tensor(out, block_tensor, residual_tensor,
                                        split_tensor, N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out, 0, split_plain, sizeof(split_plain)) ||
        !close_array(split_plain, expected_plain, HC_ELEMENTS)) {
        return 5;
    }
    if (!ds4_gpu_hc_expand_add_split_tensor(out, block_tensor, add_tensor,
                                            residual_tensor, split_tensor,
                                            N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out, 0, split_add, sizeof(split_add)) ||
        !close_array(split_add, expected_add, HC_ELEMENTS)) {
        return 6;
    }
    if (!ds4_gpu_hc_expand_add_split_tensor(out, block_tensor, block_tensor,
                                            residual_tensor, split_tensor,
                                            N_EMBD, N_HC) ||
        !ds4_gpu_tensor_read(out, 0, alias_add, sizeof(alias_add)) ||
        !close_array(alias_add, expected_alias_add, HC_ELEMENTS)) {
        return 7;
    }
    if (ds4_gpu_hc_expand_split_tensor(out, block_tensor, residual_tensor,
                                       split_tensor, 0, N_HC) ||
        ds4_gpu_hc_expand_tensor(NULL, block_tensor, residual_tensor,
                                 post_tensor, comb_tensor, N_EMBD, N_HC)) {
        return 8;
    }
    if (!ds4_gpu_synchronize()) return 9;

    ds4_gpu_tensor_free(split_tensor);
    ds4_gpu_tensor_free(comb_tensor);
    ds4_gpu_tensor_free(post_tensor);
    ds4_gpu_tensor_free(residual_tensor);
    ds4_gpu_tensor_free(add_tensor);
    ds4_gpu_tensor_free(block_tensor);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"direct_expand_output_matches\":true,"
         "\"split_expand_output_matches\":true,\"split_add_output_matches\":true,"
         "\"aliased_split_add_output_matches\":true,\"invalid_zero_shape_rejected\":true,"
         "\"null_output_rejected\":true,\"embedded_hc_expand_kernel_loaded\":true}");
    return 0;
}
