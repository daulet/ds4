use std::ffi::c_void;
use std::ptr;

use ds4_cuda::abi::{
    ds4_gpu_begin_commands, ds4_gpu_cleanup, ds4_gpu_end_commands, ds4_gpu_flush_commands,
    ds4_gpu_init, ds4_gpu_should_use_managed_kv_cache, ds4_gpu_synchronize, ds4_gpu_tensor_alloc,
    ds4_gpu_tensor_alloc_managed, ds4_gpu_tensor_bytes, ds4_gpu_tensor_contents,
    ds4_gpu_tensor_copy, ds4_gpu_tensor_free, ds4_gpu_tensor_read, ds4_gpu_tensor_view,
    ds4_gpu_tensor_write,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_6B1_SCOPE};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device_name = CudaOxideSubstrate::open(0)?.device_name()?;
    assert_eq!(ds4_gpu_init(), 1);

    let input = [13_u8, 21, 34, 55, 89, 144, 233, 1];
    let device = ds4_gpu_tensor_alloc(16);
    assert!(!device.is_null());
    assert_eq!(unsafe { ds4_gpu_tensor_bytes(device) }, 16);
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_write(
                device,
                4,
                input.as_ptr().cast::<c_void>(),
                input.len() as u64,
            )
        },
        1
    );
    let mut readback = [0_u8; 8];
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_read(
                device,
                4,
                readback.as_mut_ptr().cast::<c_void>(),
                readback.len() as u64,
            )
        },
        1
    );
    assert_eq!(readback, input);
    assert_eq!(unsafe { ds4_gpu_tensor_copy(device, 4, device, 4, 8) }, 1);

    let copied = ds4_gpu_tensor_alloc(16);
    assert!(!copied.is_null());
    assert_eq!(unsafe { ds4_gpu_tensor_copy(copied, 2, device, 4, 8) }, 1);
    let view = unsafe { ds4_gpu_tensor_view(copied, 2, 8) };
    assert!(!view.is_null());
    let mut copied_readback = [0_u8; 8];
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_read(
                view,
                0,
                copied_readback.as_mut_ptr().cast::<c_void>(),
                copied_readback.len() as u64,
            )
        },
        1
    );
    assert_eq!(copied_readback, input);

    let managed = ds4_gpu_tensor_alloc_managed(input.len() as u64);
    assert!(!managed.is_null());
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_write(
                managed,
                0,
                input.as_ptr().cast::<c_void>(),
                input.len() as u64,
            )
        },
        1
    );
    let mut managed_readback = [0_u8; 8];
    assert_eq!(
        unsafe {
            ds4_gpu_tensor_read(
                managed,
                0,
                managed_readback.as_mut_ptr().cast::<c_void>(),
                managed_readback.len() as u64,
            )
        },
        1
    );
    assert_eq!(managed_readback, input);

    let empty = ds4_gpu_tensor_alloc(0);
    assert!(!empty.is_null());
    assert_eq!(unsafe { ds4_gpu_tensor_bytes(empty) }, 1);
    assert!(!unsafe { ds4_gpu_tensor_contents(device) }.is_null());
    assert!(unsafe { ds4_gpu_tensor_view(device, 12, 8) }.is_null());
    assert_eq!(
        unsafe { ds4_gpu_tensor_write(device, 0, ptr::null(), 0) },
        0
    );
    assert_eq!(ds4_gpu_should_use_managed_kv_cache(0, 16 * GIB), 0);
    assert_eq!(ds4_gpu_should_use_managed_kv_cache(8 * GIB, 0), 1);
    assert_eq!(ds4_gpu_begin_commands(), 1);
    assert_eq!(ds4_gpu_flush_commands(), 1);
    assert_eq!(ds4_gpu_end_commands(), 1);
    assert_eq!(ds4_gpu_synchronize(), 1);

    unsafe {
        ds4_gpu_tensor_free(empty);
        ds4_gpu_tensor_free(managed);
        ds4_gpu_tensor_free(view);
        ds4_gpu_tensor_free(copied);
        ds4_gpu_tensor_free(device);
    }
    ds4_gpu_cleanup();

    println!(
        "{{\"milestone\":\"M14.6b1\",\"device_name\":{:?},\"rust_exported_resource_abi\":true,\"exported_resource_symbol_count\":{},\"initialization_roundtrip\":true,\"device_tensor_roundtrip\":true,\"managed_tensor_roundtrip\":true,\"view_roundtrip\":true,\"device_copy_roundtrip\":true,\"self_copy_identity_matches\":true,\"zero_alloc_is_one_byte\":true,\"invalid_range_rejected\":true,\"null_write_rejected\":true,\"managed_kv_policy_matches\":true,\"command_sync_matches\":true,\"owns_initialization\":{},\"owns_tensor_storage\":{},\"owns_host_device_copies\":{},\"owns_command_synchronization\":{},\"owns_managed_kv_policy\":{},\"owns_tensor_fill_kernel\":{},\"owns_compute_abi\":{},\"owns_complete_ds4_gpu_abi\":{},\"changes_default_route\":{}}}",
        device_name,
        M14_6B1_SCOPE.exported_resource_symbol_count,
        M14_6B1_SCOPE.owns_initialization,
        M14_6B1_SCOPE.owns_tensor_storage,
        M14_6B1_SCOPE.owns_host_device_copies,
        M14_6B1_SCOPE.owns_command_synchronization,
        M14_6B1_SCOPE.owns_managed_kv_policy,
        M14_6B1_SCOPE.owns_tensor_fill_kernel,
        M14_6B1_SCOPE.owns_compute_abi,
        M14_6B1_SCOPE.owns_complete_ds4_gpu_abi,
        M14_6B1_SCOPE.changes_default_route,
    );
    Ok(())
}
