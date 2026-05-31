#![cfg(all(target_os = "linux", feature = "cuda-rust-backend"))]

use core::ffi::c_void;
use core::ptr::NonNull;
use ds4_gpu::sys;

struct RawTensor {
    raw: NonNull<sys::Ds4GpuTensor>,
}

impl RawTensor {
    fn allocate_f32(values: &[f32]) -> Self {
        let bytes = std::mem::size_of_val(values) as u64;
        let raw = unsafe { sys::ds4_gpu_tensor_alloc(bytes) };
        let tensor = Self {
            raw: NonNull::new(raw).expect("Rust CUDA tensor allocation"),
        };
        assert_eq!(
            unsafe {
                sys::ds4_gpu_tensor_write(
                    tensor.raw.as_ptr(),
                    0,
                    values.as_ptr().cast::<c_void>(),
                    bytes,
                )
            },
            1
        );
        tensor
    }

    fn allocate_zeros(count: usize) -> Self {
        let raw = unsafe { sys::ds4_gpu_tensor_alloc((count * std::mem::size_of::<f32>()) as u64) };
        Self {
            raw: NonNull::new(raw).expect("Rust CUDA output allocation"),
        }
    }

    fn read_f32(&self, count: usize) -> Vec<f32> {
        let mut values = vec![0.0_f32; count];
        assert_eq!(
            unsafe {
                sys::ds4_gpu_tensor_read(
                    self.raw.as_ptr(),
                    0,
                    values.as_mut_ptr().cast::<c_void>(),
                    std::mem::size_of_val(values.as_slice()) as u64,
                )
            },
            1
        );
        values
    }
}

impl Drop for RawTensor {
    fn drop(&mut self) {
        unsafe { sys::ds4_gpu_tensor_free(self.raw.as_ptr()) };
    }
}

#[test]
fn rust_cuda_dylib_supplies_embedded_compute_abi_to_facade() {
    assert_eq!(unsafe { sys::ds4_gpu_init() }, 1);
    {
        let left = RawTensor::allocate_f32(&[1.0, -2.0, 3.5, 4.0]);
        let right = RawTensor::allocate_f32(&[0.5, 2.0, -1.5, 8.0]);
        let out = RawTensor::allocate_zeros(4);
        assert_eq!(
            unsafe {
                sys::ds4_gpu_add_tensor(out.raw.as_ptr(), left.raw.as_ptr(), right.raw.as_ptr(), 4)
            },
            1
        );
        assert_eq!(unsafe { sys::ds4_gpu_synchronize() }, 1);
        assert_eq!(out.read_f32(4), vec![1.5, 0.0, 2.0, 12.0]);
    }
    unsafe { sys::ds4_gpu_cleanup() };
}
