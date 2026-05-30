#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N_VOCAB 3u
#define N_TOKENS 3u
#define N_EMBD 3u
#define N_HC 2u

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-6f) return 0;
    }
    return 1;
}

int main(void) {
    uint16_t model[32] = {0};
    const uint64_t weight_offset = 2u * sizeof(uint16_t);
    const uint64_t alternate_weight_offset = 16u * sizeof(uint16_t);
    const int32_t tokens[N_TOKENS] = {-1, 1, 99};
    const float expected_single[N_EMBD * N_HC] = {-2.0f, 0.75f, 1.5f, -2.0f, 0.75f, 1.5f};
    const float expected_batch[N_TOKENS * N_HC * N_EMBD] = {
        0.5f, -1.0f, 2.0f, 0.5f, -1.0f, 2.0f,
        3.0f, 4.0f, -0.25f, 3.0f, 4.0f, -0.25f,
        0.5f, -1.0f, 2.0f, 0.5f, -1.0f, 2.0f,
    };
    const float expected_alternate[N_EMBD * N_HC] = {-0.5f, 0.25f, 1.0f, -0.5f, 0.25f, 1.0f};
    float got[N_TOKENS * N_HC * N_EMBD] = {0};

    model[2] = 0x3800;  /* 0.5 */
    model[3] = 0xbc00;  /* -1.0 */
    model[4] = 0x4000;  /* 2.0 */
    model[5] = 0x4200;  /* 3.0 */
    model[6] = 0x4400;  /* 4.0 */
    model[7] = 0xb400;  /* -0.25 */
    model[8] = 0xc000;  /* -2.0 */
    model[9] = 0x3a00;  /* 0.75 */
    model[10] = 0x3e00; /* 1.5 */
    model[16] = 0xb800; /* -0.5 */
    model[17] = 0x3400; /* 0.25 */
    model[18] = 0x3c00; /* 1.0 */

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model, sizeof(model))) return 1;
    ds4_gpu_tensor *single_out = ds4_gpu_tensor_alloc(sizeof(expected_single));
    ds4_gpu_tensor *short_single_out = ds4_gpu_tensor_alloc(sizeof(expected_single) - sizeof(float));
    ds4_gpu_tensor *batch_out = ds4_gpu_tensor_alloc(sizeof(expected_batch));
    ds4_gpu_tensor *short_batch_out = ds4_gpu_tensor_alloc(sizeof(expected_batch) - sizeof(float));
    ds4_gpu_tensor *tokens_tensor = ds4_gpu_tensor_alloc(sizeof(tokens));
    ds4_gpu_tensor *short_tokens = ds4_gpu_tensor_alloc(sizeof(tokens) - sizeof(int32_t));
    if (!single_out || !short_single_out || !batch_out || !short_batch_out ||
        !tokens_tensor || !short_tokens ||
        !ds4_gpu_tensor_write(tokens_tensor, 0, tokens, sizeof(tokens)) ||
        !ds4_gpu_tensor_write(short_tokens, 0, tokens, sizeof(tokens) - sizeof(int32_t))) {
        return 2;
    }

    if (!ds4_gpu_embed_token_hc_tensor(
            single_out, model, sizeof(model), weight_offset, N_VOCAB, 2u, N_EMBD, N_HC) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_out, 0, got, sizeof(expected_single)) ||
        !close_array(got, expected_single, N_EMBD * N_HC)) {
        return 3;
    }
    if (!ds4_gpu_embed_tokens_hc_tensor(
            batch_out, tokens_tensor, model, sizeof(model), weight_offset,
            N_VOCAB, N_TOKENS, N_EMBD, N_HC) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(batch_out, 0, got, sizeof(expected_batch)) ||
        !close_array(got, expected_batch, N_TOKENS * N_HC * N_EMBD)) {
        return 4;
    }
    if (!ds4_gpu_embed_token_hc_tensor(
            single_out, model, sizeof(model), alternate_weight_offset,
            N_VOCAB, 0u, N_EMBD, N_HC) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(single_out, 0, got, sizeof(expected_alternate)) ||
        !close_array(got, expected_alternate, N_EMBD * N_HC)) {
        return 5;
    }
    if (ds4_gpu_embed_token_hc_tensor(
            single_out, model, sizeof(model), weight_offset, N_VOCAB, N_VOCAB, N_EMBD, N_HC) ||
        ds4_gpu_embed_token_hc_tensor(
            short_single_out, model, sizeof(model), weight_offset, N_VOCAB, 2u, N_EMBD, N_HC) ||
        ds4_gpu_embed_tokens_hc_tensor(
            batch_out, short_tokens, model, sizeof(model), weight_offset,
            N_VOCAB, N_TOKENS, N_EMBD, N_HC) ||
        ds4_gpu_embed_tokens_hc_tensor(
            short_batch_out, tokens_tensor, model, sizeof(model), weight_offset,
            N_VOCAB, N_TOKENS, N_EMBD, N_HC) ||
        ds4_gpu_embed_tokens_hc_tensor(
            batch_out, tokens_tensor, model, sizeof(model), sizeof(model),
            N_VOCAB, N_TOKENS, N_EMBD, N_HC)) {
        return 6;
    }

    ds4_gpu_tensor_free(short_tokens);
    ds4_gpu_tensor_free(tokens_tensor);
    ds4_gpu_tensor_free(short_batch_out);
    ds4_gpu_tensor_free(batch_out);
    ds4_gpu_tensor_free(short_single_out);
    ds4_gpu_tensor_free(single_out);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"single_token_hc_replication_matches\":true,"
         "\"batched_invalid_token_fallback_matches\":true,\"alternate_embedding_range_matches\":true,"
         "\"single_invalid_token_rejected\":true,\"short_single_output_rejected\":true,"
         "\"short_batch_tokens_rejected\":true,\"short_batch_output_rejected\":true,"
         "\"invalid_embedding_range_rejected\":true,\"embedded_embedding_kernels_loaded\":true}");
    return 0;
}
