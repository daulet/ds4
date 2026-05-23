#![cfg(any(
    target_os = "macos",
    all(target_os = "linux", feature = "cuda-backend")
))]

use core::ffi::c_void;
use core::ptr::NonNull;
use ds4_gpu::{sys, CommandBatch, Tensor};
use std::path::Path;

struct RawTensor {
    raw: NonNull<sys::Ds4GpuTensor>,
}

impl RawTensor {
    fn allocate(bytes: usize) -> Self {
        let raw = unsafe { sys::ds4_gpu_tensor_alloc(bytes as u64) };
        Self {
            raw: NonNull::new(raw).expect("direct C tensor allocation"),
        }
    }

    fn view(&self, offset: u64, bytes: usize) -> Self {
        let raw = unsafe { sys::ds4_gpu_tensor_view(self.raw.as_ptr(), offset, bytes as u64) };
        Self {
            raw: NonNull::new(raw).expect("direct C tensor view"),
        }
    }

    fn byte_len(&self) -> u64 {
        unsafe { sys::ds4_gpu_tensor_bytes(self.raw.as_ptr()) }
    }

    fn write(&mut self, offset: u64, data: &[u8]) -> i32 {
        unsafe {
            sys::ds4_gpu_tensor_write(
                self.raw.as_ptr(),
                offset,
                data.as_ptr().cast::<c_void>(),
                data.len() as u64,
            )
        }
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> i32 {
        unsafe {
            sys::ds4_gpu_tensor_read(
                self.raw.as_ptr(),
                offset,
                out.as_mut_ptr().cast::<c_void>(),
                out.len() as u64,
            )
        }
    }

    fn fill_f32(&mut self, value: f32, count: usize) -> i32 {
        unsafe { sys::ds4_gpu_tensor_fill_f32(self.raw.as_ptr(), value, count as u64) }
    }

    fn copy_from(&mut self, src: &Self, dst_offset: u64, src_offset: u64, bytes: usize) -> i32 {
        unsafe {
            sys::ds4_gpu_tensor_copy(
                self.raw.as_ptr(),
                dst_offset,
                src.raw.as_ptr(),
                src_offset,
                bytes as u64,
            )
        }
    }
}

impl Drop for RawTensor {
    fn drop(&mut self) {
        unsafe {
            sys::ds4_gpu_tensor_free(self.raw.as_ptr());
        }
    }
}

#[test]
fn safe_tensor_wrapper_matches_direct_c_abi() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::set_current_dir(manifest_dir.join("../..")).expect("repo root cwd");

    ds4_gpu::initialize().expect("initialize DS4 GPU backend");
    {
        let input: [u8; 32] = [
            0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 123, 45, 67, 11, 22, 33, 44, 55, 66,
            77, 88, 99, 111, 122, 133, 144, 155, 166, 177,
        ];
        let patch: [u8; 8] = [201, 202, 203, 204, 205, 206, 207, 208];

        let mut safe_src = Tensor::allocate(input.len()).expect("safe src");
        let mut safe_dst = Tensor::allocate(input.len()).expect("safe dst");
        let mut raw_src = RawTensor::allocate(input.len());
        let mut raw_dst = RawTensor::allocate(input.len());

        assert_eq!(safe_src.byte_len(), raw_src.byte_len());
        assert_eq!(safe_dst.byte_len(), raw_dst.byte_len());

        safe_src.write_bytes(0, &input).expect("safe write");
        assert_eq!(raw_src.write(0, &input), 1);

        let mut safe_read = [0u8; 32];
        let mut raw_read = [0u8; 32];
        safe_src.read_bytes(0, &mut safe_read).expect("safe read");
        assert_eq!(raw_src.read(0, &mut raw_read), 1);
        assert_eq!(safe_read, input);
        assert_eq!(safe_read, raw_read);

        safe_src.fill_f32(1.25, 4).expect("safe fill");
        assert_eq!(raw_src.fill_f32(1.25, 4), 1);
        safe_src
            .read_bytes(0, &mut safe_read)
            .expect("safe read fill");
        assert_eq!(raw_src.read(0, &mut raw_read), 1);
        assert_eq!(&safe_read[..16], &raw_read[..16]);

        {
            let mut safe_view = safe_src.view(4, patch.len()).expect("safe view");
            assert_eq!(safe_view.byte_len(), patch.len() as u64);
            safe_view.write_bytes(0, &patch).expect("safe view write");
        }
        {
            let mut raw_view = raw_src.view(4, patch.len());
            assert_eq!(raw_view.byte_len(), patch.len() as u64);
            assert_eq!(raw_view.write(0, &patch), 1);
        }

        let mut safe_batch = CommandBatch::begin().expect("safe begin");
        safe_dst
            .copy_from(&safe_src, 0, 0, 16, &mut safe_batch)
            .expect("safe copy");
        safe_batch.flush().expect("safe flush");
        safe_batch.finish().expect("safe finish");
        ds4_gpu::synchronize().expect("safe synchronize");

        unsafe {
            assert_eq!(sys::ds4_gpu_begin_commands(), 1);
        }
        assert_eq!(raw_dst.copy_from(&raw_src, 0, 0, 16), 1);
        unsafe {
            assert_eq!(sys::ds4_gpu_flush_commands(), 1);
            assert_eq!(sys::ds4_gpu_end_commands(), 1);
            assert_eq!(sys::ds4_gpu_synchronize(), 1);
        }

        safe_dst
            .read_bytes(0, &mut safe_read)
            .expect("safe dst read");
        assert_eq!(raw_dst.read(0, &mut raw_read), 1);
        assert_eq!(&safe_read[..16], &raw_read[..16]);

        assert!(safe_src.write_bytes(28, &patch).is_err());
        assert_eq!(raw_src.write(28, &patch), 0);

        let mut safe_batch = CommandBatch::begin().expect("safe begin for failing copy");
        assert!(safe_dst
            .copy_from(&safe_src, 30, 0, patch.len(), &mut safe_batch)
            .is_err());
        safe_batch.finish().expect("safe finish after failing copy");
        unsafe {
            assert_eq!(sys::ds4_gpu_begin_commands(), 1);
        }
        assert_eq!(raw_dst.copy_from(&raw_src, 30, 0, patch.len()), 0);
        unsafe {
            assert_eq!(sys::ds4_gpu_end_commands(), 1);
        }
    }
    unsafe {
        ds4_gpu::cleanup();
    }
}
