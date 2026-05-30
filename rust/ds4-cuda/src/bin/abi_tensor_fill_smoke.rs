use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use ds4_cuda::abi::{
    ds4_gpu_cleanup, ds4_gpu_init, ds4_gpu_synchronize, ds4_gpu_tensor_alloc,
    ds4_gpu_tensor_alloc_managed, ds4_gpu_tensor_fill_f32, ds4_gpu_tensor_free,
    ds4_gpu_tensor_read, ds4_gpu_tensor_view, ds4_gpu_tensor_write,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_6B2A_SCOPE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device_name = CudaOxideSubstrate::open(0)?.device_name()?;
    assert_eq!(ds4_gpu_init(), 1);

    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = ds4_gpu_tensor_alloc((input.len() * size_of::<f32>()) as u64);
    assert!(!tensor.is_null());
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_write(
                tensor,
                0,
                input.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(&input) as u64,
            )
        },
        1
    );

    assert_eq!(unsafe { ds4_gpu_tensor_fill_f32(tensor, -3.5, 4) }, 1);
    let mut prefix = [0.0_f32; 6];
    read_tensor(tensor, &mut prefix);
    assert_eq!(prefix, [-3.5, -3.5, -3.5, -3.5, 5.0, 6.0]);

    let suffix = unsafe {
        ds4_gpu_tensor_view(
            tensor,
            (4 * size_of::<f32>()) as u64,
            (2 * size_of::<f32>()) as u64,
        )
    };
    assert!(!suffix.is_null());
    assert_eq!(unsafe { ds4_gpu_tensor_fill_f32(suffix, -0.0, 2) }, 1);
    let mut signed_zero = [0.0_f32; 6];
    read_tensor(tensor, &mut signed_zero);
    assert_eq!(signed_zero[4].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(signed_zero[5].to_bits(), (-0.0_f32).to_bits());

    assert_eq!(
        unsafe { ds4_gpu_tensor_fill_f32(tensor, f32::NEG_INFINITY, 6) },
        1
    );
    let mut infinite = [0.0_f32; 6];
    read_tensor(tensor, &mut infinite);
    assert_eq!(infinite, [f32::NEG_INFINITY; 6]);

    assert_eq!(unsafe { ds4_gpu_tensor_fill_f32(tensor, 7.0, 0) }, 1);
    let mut zero_count = [0.0_f32; 6];
    read_tensor(tensor, &mut zero_count);
    assert_eq!(zero_count, [f32::NEG_INFINITY; 6]);

    let managed = ds4_gpu_tensor_alloc_managed((2 * size_of::<f32>()) as u64);
    assert!(!managed.is_null());
    assert_eq!(unsafe { ds4_gpu_tensor_fill_f32(managed, 2.25, 2) }, 1);
    let mut managed_fill = [0.0_f32; 2];
    read_tensor(managed, &mut managed_fill);
    assert_eq!(managed_fill, [2.25; 2]);

    assert_eq!(unsafe { ds4_gpu_tensor_fill_f32(tensor, 0.0, 7) }, 0);
    assert_eq!(
        unsafe { ds4_gpu_tensor_fill_f32(ptr::null_mut(), 0.0, 0) },
        0
    );
    assert_eq!(ds4_gpu_synchronize(), 1);

    unsafe {
        ds4_gpu_tensor_free(managed);
        ds4_gpu_tensor_free(suffix);
        ds4_gpu_tensor_free(tensor);
    }
    ds4_gpu_cleanup();

    println!(
        "{{\"milestone\":\"M14.6b2a\",\"device_name\":{:?},\"rust_exported_tensor_fill_abi\":true,\"exported_abi_symbol_count\":{},\"exported_compute_symbol_count\":{},\"prefix_fill_matches\":true,\"view_offset_fill_matches\":true,\"signed_zero_bits_match\":true,\"negative_infinity_fill_matches\":true,\"zero_count_is_noop\":true,\"managed_fill_matches\":true,\"bounds_rejected\":true,\"null_rejected\":true,\"owns_tensor_fill_f32\":{},\"owns_graph_compute_abi\":{},\"owns_complete_ds4_gpu_abi\":{},\"changes_default_route\":{}}}",
        device_name,
        M14_6B2A_SCOPE.exported_abi_symbol_count,
        M14_6B2A_SCOPE.exported_compute_symbol_count,
        M14_6B2A_SCOPE.owns_tensor_fill_f32,
        M14_6B2A_SCOPE.owns_graph_compute_abi,
        M14_6B2A_SCOPE.owns_complete_ds4_gpu_abi,
        M14_6B2A_SCOPE.changes_default_route,
    );
    Ok(())
}

fn read_tensor(tensor: *const ds4_cuda::abi::Ds4GpuTensor, output: &mut [f32]) {
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_read(
                tensor,
                0,
                output.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of_val(output) as u64,
            )
        },
        1
    );
}
