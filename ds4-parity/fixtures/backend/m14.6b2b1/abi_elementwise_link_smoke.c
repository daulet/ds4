#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int write_f32(ds4_gpu_tensor *tensor, const float *values, uint32_t count) {
    return ds4_gpu_tensor_write(tensor, 0, values, (uint64_t)count * sizeof(float));
}

static int read_f32(const ds4_gpu_tensor *tensor, float *values, uint32_t count) {
    return ds4_gpu_tensor_read(tensor, 0, values, (uint64_t)count * sizeof(float));
}

int main(void) {
    const float a_in[4] = {1.0f, -2.0f, 3.5f, 8.0f};
    const float b_in[4] = {4.0f, 5.0f, -1.5f, -0.5f};
    const float sum_want[4] = {5.0f, 3.0f, 2.0f, 7.5f};
    const float row_in[3] = {2.0f, -1.5f, 4.0f};
    const float repeat_want[9] = {
        2.0f, -1.5f, 4.0f, 2.0f, -1.5f, 4.0f, 2.0f, -1.5f, 4.0f,
    };
    float got4[4] = {0};
    float got9[9] = {0};

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(sizeof(a_in));
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(sizeof(b_in));
    ds4_gpu_tensor *sum = ds4_gpu_tensor_alloc(sizeof(sum_want));
    ds4_gpu_tensor *row = ds4_gpu_tensor_alloc(sizeof(row_in));
    ds4_gpu_tensor *repeated = ds4_gpu_tensor_alloc(sizeof(repeat_want));
    if (!a || !b || !sum || !row || !repeated) return 2;
    if (!write_f32(a, a_in, 4) || !write_f32(b, b_in, 4) || !write_f32(row, row_in, 3)) return 3;
    if (!ds4_gpu_add_tensor(sum, a, b, 4) || !read_f32(sum, got4, 4) ||
        memcmp(got4, sum_want, sizeof(got4)) != 0) return 4;
    if (!ds4_gpu_add_tensor(a, a, b, 4) || !read_f32(a, got4, 4) ||
        memcmp(got4, sum_want, sizeof(got4)) != 0) return 5;
    if (!ds4_gpu_repeat_hc_tensor(repeated, row, 3, 3) || !read_f32(repeated, got9, 9) ||
        memcmp(got9, repeat_want, sizeof(got9)) != 0) return 6;
    if (ds4_gpu_add_tensor(sum, a, b, 5) ||
        ds4_gpu_repeat_hc_tensor(repeated, row, 0, 3) ||
        ds4_gpu_add_tensor(NULL, a, b, 4)) return 7;
    if (!ds4_gpu_synchronize()) return 8;

    ds4_gpu_tensor_free(repeated);
    ds4_gpu_tensor_free(row);
    ds4_gpu_tensor_free(sum);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"embedded_rust_kernel_module_loaded\":true,\"add_output_matches\":true,\"add_alias_output_matches\":true,\"repeat_hc_output_matches\":true,\"invalid_shape_rejected\":true,\"null_rejected\":true}");
    return 0;
}
