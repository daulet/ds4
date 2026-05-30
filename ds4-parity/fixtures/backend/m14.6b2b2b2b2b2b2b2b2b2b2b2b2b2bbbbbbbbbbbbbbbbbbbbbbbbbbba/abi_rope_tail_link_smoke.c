#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    N_TOK = 2,
    N_HEAD = 2,
    HEAD_DIM = 10,
    N_ROT = 6,
    POS0 = 11,
    N_CTX_ORIG = 4096,
    COUNT = N_TOK * N_HEAD * HEAD_DIM,
};

static const float FREQ_BASE = 100.0f;
static const float FREQ_SCALE = 0.5f;
static const float ATTN_FACTOR = 1.15f;
static const float BETA_FAST = 32.0f;
static const float BETA_SLOW = 1.0f;
static const float PI = 3.1415927f;

static float rope_yarn_ramp(float low, float high, uint32_t rot_i) {
    float y = ((float)(rot_i / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static void reference_rope(float *values, int inverse, float ext_factor, float attn_factor) {
    const uint32_t n_nope = HEAD_DIM - N_ROT;
    float corr0 = 0.0f;
    float corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(FREQ_BASE);
        corr0 = floorf((float)N_ROT *
                       logf((float)N_CTX_ORIG / (BETA_FAST * 2.0f * PI)) /
                       denom);
        corr1 = ceilf((float)N_ROT *
                      logf((float)N_CTX_ORIG / (BETA_SLOW * 2.0f * PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(N_ROT - 1), corr1);
    }
    for (uint32_t token = 0; token < N_TOK; ++token) {
        for (uint32_t head = 0; head < N_HEAD; ++head) {
            float *tail = values + ((uint64_t)token * N_HEAD + head) * HEAD_DIM + n_nope;
            for (uint32_t i = 0; i < N_ROT; i += 2) {
                const float theta_extrap =
                    (float)(POS0 + token) * powf(FREQ_BASE, -((float)i) / (float)N_ROT);
                const float theta_interp = FREQ_SCALE * theta_extrap;
                float theta = theta_interp;
                float scale = attn_factor;
                if (ext_factor != 0.0f) {
                    const float mix = rope_yarn_ramp(corr0, corr1, i) * ext_factor;
                    theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
                    scale *= 1.0f + 0.1f * logf(1.0f / FREQ_SCALE);
                }
                const float c = cosf(theta) * scale;
                float s = sinf(theta) * scale;
                if (inverse) s = -s;
                const float x0 = tail[i];
                const float x1 = tail[i + 1];
                tail[i] = x0 * c - x1 * s;
                tail[i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

static int close_array(const float *actual, const float *expected, uint32_t count, float tol) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > tol) return 0;
    }
    return 1;
}

int main(void) {
    float input[COUNT];
    float expected[COUNT];
    float got[COUNT] = {0};
    for (uint32_t i = 0; i < COUNT; ++i) {
        input[i] = (float)((i * 17u + 5u) % 43u) * 0.125f - 2.25f;
    }

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_alloc(sizeof(input) - sizeof(float));
    if (!x || !short_x) return 2;

    memcpy(expected, input, sizeof(expected));
    reference_rope(expected, 0, 0.0f, 1.0f);
    if (!ds4_gpu_tensor_write(x, 0, input, sizeof(input)) ||
        !ds4_gpu_rope_tail_tensor(x, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0,
                                  N_CTX_ORIG, false, FREQ_BASE, FREQ_SCALE,
                                  0.0f, 1.0f, BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, expected, COUNT, 2.0e-5f)) return 3;
    for (uint32_t row = 0; row < N_TOK * N_HEAD; ++row) {
        if (!close_array(got + row * HEAD_DIM, input + row * HEAD_DIM,
                         HEAD_DIM - N_ROT, 0.0f)) return 4;
    }

    memcpy(expected, input, sizeof(expected));
    reference_rope(expected, 1, 1.0f, ATTN_FACTOR);
    if (!ds4_gpu_tensor_write(x, 0, input, sizeof(input)) ||
        !ds4_gpu_rope_tail_tensor(x, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0,
                                  N_CTX_ORIG, true, FREQ_BASE, FREQ_SCALE,
                                  1.0f, ATTN_FACTOR, BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        !close_array(got, expected, COUNT, 3.0e-5f)) return 5;

    if (ds4_gpu_rope_tail_tensor(short_x, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0,
                                 N_CTX_ORIG, false, FREQ_BASE, FREQ_SCALE,
                                 0.0f, 1.0f, BETA_FAST, BETA_SLOW) ||
        ds4_gpu_rope_tail_tensor(x, 0, N_HEAD, HEAD_DIM, N_ROT, POS0,
                                 N_CTX_ORIG, false, FREQ_BASE, FREQ_SCALE,
                                 0.0f, 1.0f, BETA_FAST, BETA_SLOW) ||
        ds4_gpu_rope_tail_tensor(x, N_TOK, N_HEAD, HEAD_DIM, 0, POS0,
                                 N_CTX_ORIG, false, FREQ_BASE, FREQ_SCALE,
                                 0.0f, 1.0f, BETA_FAST, BETA_SLOW) ||
        ds4_gpu_rope_tail_tensor(x, N_TOK, N_HEAD, HEAD_DIM, N_ROT - 1, POS0,
                                 N_CTX_ORIG, false, FREQ_BASE, FREQ_SCALE,
                                 0.0f, 1.0f, BETA_FAST, BETA_SLOW) ||
        ds4_gpu_rope_tail_tensor(NULL, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0,
                                 N_CTX_ORIG, false, FREQ_BASE, FREQ_SCALE,
                                 0.0f, 1.0f, BETA_FAST, BETA_SLOW)) return 6;

    ds4_gpu_tensor_free(short_x);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"interpolated_rope_output_matches\":true,\"yarn_inverse_output_matches\":true,\"non_rope_prefix_preserved\":true,\"zero_pair_rejected\":true,\"invalid_shape_rejected\":true,\"null_rejected\":true,\"embedded_rope_tail_kernel_loaded\":true}");
    return 0;
}
