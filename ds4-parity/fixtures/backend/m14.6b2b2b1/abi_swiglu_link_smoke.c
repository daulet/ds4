#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int close_array(const float *actual, const float *expected, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-5f) return 0;
    }
    return 1;
}

int main(void) {
    const float gate_in[6] = {0.0f, 2.0f, -3.0f, 0.5f, NAN, -0.25f};
    const float up_in[6] = {2.0f, 4.0f, -4.0f, -0.25f, 0.5f, NAN};
    const float want_clamped[6] = {
        0.0f, 1.3796569f, 0.1600623f, -0.0583556f, 0.4598856f, 0.1231379f,
    };
    const float want_unclamped[4] = {0.0f, 5.2847824f, 0.4268329f, -0.0583556f};
    float got[6] = {0};

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(sizeof(gate_in));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(sizeof(up_in));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(gate_in));
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc(3 * sizeof(float));
    if (!gate || !up || !out || !short_out) return 2;
    if (!ds4_gpu_tensor_write(gate, 0, gate_in, sizeof(gate_in)) ||
        !ds4_gpu_tensor_write(up, 0, up_in, sizeof(up_in))) return 3;
    if (!ds4_gpu_swiglu_tensor(out, gate, up, 6, 1.5f, 0.75f) ||
        !ds4_gpu_tensor_read(out, 0, got, sizeof(got)) ||
        !close_array(got, want_clamped, 6)) return 4;
    if (!ds4_gpu_swiglu_tensor(out, gate, up, 4, 0.0f, 0.75f) ||
        !ds4_gpu_tensor_read(out, 0, got, 4 * sizeof(float)) ||
        !close_array(got, want_unclamped, 4)) return 5;
    if (!ds4_gpu_tensor_write(gate, 0, gate_in, sizeof(gate_in)) ||
        !ds4_gpu_swiglu_tensor(gate, gate, up, 6, 1.5f, 0.75f) ||
        !ds4_gpu_tensor_read(gate, 0, got, sizeof(got)) ||
        !close_array(got, want_clamped, 6)) return 6;
    if (ds4_gpu_swiglu_tensor(short_out, gate, up, 6, 1.5f, 0.75f) ||
        ds4_gpu_swiglu_tensor(out, gate, up, 0, 1.5f, 0.75f) ||
        ds4_gpu_swiglu_tensor(NULL, gate, up, 6, 1.5f, 0.75f)) return 7;
    if (!ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"embedded_libdevice_module_loaded\":true,\"swiglu_clamped_output_matches\":true,\"swiglu_unclamped_output_matches\":true,\"swiglu_alias_output_matches\":true,\"invalid_shape_rejected\":true,\"zero_count_rejected\":true,\"null_rejected\":true}");
    return 0;
}
