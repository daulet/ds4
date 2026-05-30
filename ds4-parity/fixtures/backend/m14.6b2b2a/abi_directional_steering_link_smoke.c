#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const float directions_in[8] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 1.0f, 0.25f,
    };
    const float x_in[8] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const float want[8] = {
        0.5625f, 2.4375f, 2.125f, 3.78125f,
        3.65625f, 3.34375f, 1.3125f, 0.828125f,
    };
    float got[8] = {0};

    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *directions = ds4_gpu_tensor_alloc(sizeof(directions_in));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_in));
    ds4_gpu_tensor *short_directions = ds4_gpu_tensor_alloc(4 * sizeof(float));
    if (!directions || !x || !short_directions) return 2;
    if (!ds4_gpu_tensor_write(directions, 0, directions_in, sizeof(directions_in)) ||
        !ds4_gpu_tensor_write(x, 0, x_in, sizeof(x_in))) return 3;
    if (!ds4_gpu_directional_steering_project_tensor(x, directions, 1, 4, 2, 0.25f) ||
        !ds4_gpu_tensor_read(x, 0, got, sizeof(got)) ||
        memcmp(got, want, sizeof(got)) != 0) return 4;
    if (ds4_gpu_directional_steering_project_tensor(x, directions, 1, 4, 2, 0.0f) ||
        ds4_gpu_directional_steering_project_tensor(x, short_directions, 1, 4, 2, 0.25f) ||
        ds4_gpu_directional_steering_project_tensor(NULL, directions, 1, 4, 2, 0.25f)) return 5;
    if (!ds4_gpu_synchronize()) return 6;

    ds4_gpu_tensor_free(short_directions);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(directions);
    ds4_gpu_cleanup();
    puts("{\"c_linked_rust_staticlib\":true,\"directional_projection_matches\":true,\"zero_scale_rejected\":true,\"bounds_rejected\":true,\"null_rejected\":true}");
    return 0;
}
