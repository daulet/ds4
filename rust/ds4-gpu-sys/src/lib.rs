#![no_std]

use core::ffi::{c_char, c_int, c_void};
use core::marker::PhantomData;

#[repr(C)]
pub struct Ds4GpuTensor {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

unsafe extern "C" {
    pub fn ds4_gpu_init() -> c_int;
    pub fn ds4_gpu_cleanup();

    pub fn ds4_gpu_tensor_alloc(bytes: u64) -> *mut Ds4GpuTensor;
    pub fn ds4_gpu_tensor_alloc_managed(bytes: u64) -> *mut Ds4GpuTensor;
    pub fn ds4_gpu_tensor_view(
        base: *const Ds4GpuTensor,
        offset: u64,
        bytes: u64,
    ) -> *mut Ds4GpuTensor;
    pub fn ds4_gpu_tensor_free(tensor: *mut Ds4GpuTensor);
    pub fn ds4_gpu_tensor_bytes(tensor: *const Ds4GpuTensor) -> u64;
    pub fn ds4_gpu_tensor_contents(tensor: *mut Ds4GpuTensor) -> *mut c_void;
    pub fn ds4_gpu_tensor_fill_f32(tensor: *mut Ds4GpuTensor, value: f32, count: u64) -> c_int;
    pub fn ds4_gpu_tensor_write(
        tensor: *mut Ds4GpuTensor,
        offset: u64,
        data: *const c_void,
        bytes: u64,
    ) -> c_int;
    pub fn ds4_gpu_tensor_read(
        tensor: *const Ds4GpuTensor,
        offset: u64,
        data: *mut c_void,
        bytes: u64,
    ) -> c_int;
    pub fn ds4_gpu_tensor_copy(
        dst: *mut Ds4GpuTensor,
        dst_offset: u64,
        src: *const Ds4GpuTensor,
        src_offset: u64,
        bytes: u64,
    ) -> c_int;

    pub fn ds4_gpu_begin_commands() -> c_int;
    pub fn ds4_gpu_flush_commands() -> c_int;
    pub fn ds4_gpu_end_commands() -> c_int;
    pub fn ds4_gpu_synchronize() -> c_int;

    pub fn ds4_gpu_set_model_map(model_map: *const c_void, model_size: u64) -> c_int;
    pub fn ds4_gpu_set_model_fd(fd: c_int) -> c_int;
    pub fn ds4_gpu_set_model_map_range(
        model_map: *const c_void,
        model_size: u64,
        map_offset: u64,
        map_size: u64,
    ) -> c_int;
    pub fn ds4_gpu_cache_model_range(
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
        label: *const c_char,
    ) -> c_int;
    pub fn ds4_gpu_cache_q8_f16_range(
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
        in_dim: u64,
        out_dim: u64,
        label: *const c_char,
    ) -> c_int;
    pub fn ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes: u64, context_bytes: u64) -> c_int;
    pub fn ds4_gpu_set_quality(quality: bool);
    pub fn ds4_gpu_print_memory_report(label: *const c_char);
}

#[cfg(test)]
mod tests {
    use super::Ds4GpuTensor;

    #[test]
    fn opaque_tensor_is_only_used_behind_pointers() {
        assert_eq!(
            core::mem::size_of::<*mut Ds4GpuTensor>(),
            core::mem::size_of::<usize>()
        );
    }
}
