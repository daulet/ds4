#![no_std]

use core::convert::TryFrom;
use core::ffi::{c_int, c_void};
use core::fmt;
use core::marker::PhantomData;
use core::ptr::NonNull;

pub use ds4_gpu_sys as sys;

pub mod decode_backend;
pub mod decode_execution;
pub mod decode_plan;
pub mod decode_runtime;
pub mod decode_trace;
pub mod graph_plan;
pub mod graph_state;
pub mod mtp_decode2_plan;
pub mod mtp_draft_plan;
pub mod mtp_frontier_plan;
pub mod mtp_plan;
pub mod mtp_suffix_plan;
pub mod prefill_plan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct GpuStatus(c_int);

impl GpuStatus {
    pub const SUCCESS: Self = Self(1);
    pub const FAILURE: Self = Self(0);

    pub const fn from_raw(raw: c_int) -> Self {
        Self(raw)
    }

    pub const fn as_raw(self) -> c_int {
        self.0
    }

    pub const fn is_ok(self) -> bool {
        self.0 != 0
    }

    pub const fn into_result(self) -> Result<(), GpuError> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(GpuError::backend_status(self.0))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuError {
    kind: GpuErrorKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuErrorKind {
    BackendStatus(c_int),
    LengthOverflow,
    InvalidRange,
    NullTensor,
}

impl GpuError {
    pub const fn backend_status(code: c_int) -> Self {
        Self {
            kind: GpuErrorKind::BackendStatus(code),
        }
    }

    pub const fn length_overflow() -> Self {
        Self {
            kind: GpuErrorKind::LengthOverflow,
        }
    }

    pub const fn invalid_range() -> Self {
        Self {
            kind: GpuErrorKind::InvalidRange,
        }
    }

    pub const fn null_tensor() -> Self {
        Self {
            kind: GpuErrorKind::NullTensor,
        }
    }

    pub const fn kind(self) -> GpuErrorKind {
        self.kind
    }

    pub const fn code(self) -> Option<c_int> {
        match self.kind {
            GpuErrorKind::BackendStatus(code) => Some(code),
            _ => None,
        }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            GpuErrorKind::BackendStatus(code) => {
                write!(f, "DS4 GPU backend returned status {code}")
            }
            GpuErrorKind::LengthOverflow => write!(f, "length does not fit DS4 GPU ABI"),
            GpuErrorKind::InvalidRange => write!(f, "range is outside model mapping"),
            GpuErrorKind::NullTensor => write!(f, "DS4 GPU backend returned null tensor"),
        }
    }
}

pub fn initialize() -> Result<(), GpuError> {
    unsafe { GpuStatus::from_raw(sys::ds4_gpu_init()).into_result() }
}

pub fn synchronize() -> Result<(), GpuError> {
    unsafe { GpuStatus::from_raw(sys::ds4_gpu_synchronize()).into_result() }
}

/// Releases the process-global backend state.
///
/// # Safety
///
/// No live `Tensor`, `TensorView`, or `CommandBatch` may exist when this is
/// called. The C backend owns global Metal/CUDA state and does not track Rust
/// handles.
pub unsafe fn cleanup() {
    sys::ds4_gpu_cleanup();
}

#[derive(Debug)]
pub struct CommandBatch {
    active: bool,
}

impl CommandBatch {
    pub fn begin() -> Result<Self, GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_begin_commands()).into_result()?;
        }
        Ok(Self { active: true })
    }

    pub fn flush(&mut self) -> Result<(), GpuError> {
        unsafe { GpuStatus::from_raw(sys::ds4_gpu_flush_commands()).into_result() }
    }

    pub fn finish(mut self) -> Result<(), GpuError> {
        if !self.active {
            return Ok(());
        }

        let result = unsafe { GpuStatus::from_raw(sys::ds4_gpu_end_commands()).into_result() };
        self.active = false;
        result
    }
}

impl Drop for CommandBatch {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                let _ = sys::ds4_gpu_end_commands();
            }
            self.active = false;
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    raw: NonNull<sys::Ds4GpuTensor>,
}

impl Tensor {
    pub fn allocate(bytes: usize) -> Result<Self, GpuError> {
        let bytes = abi_len(bytes)?;
        unsafe { Self::from_raw(sys::ds4_gpu_tensor_alloc(bytes)) }
    }

    pub fn allocate_managed(bytes: usize) -> Result<Self, GpuError> {
        let bytes = abi_len(bytes)?;
        unsafe { Self::from_raw(sys::ds4_gpu_tensor_alloc_managed(bytes)) }
    }

    pub fn byte_len(&self) -> u64 {
        tensor_byte_len(self.raw.as_ptr())
    }

    pub fn as_tensor_ref(&self) -> TensorRef<'_> {
        TensorRef {
            raw: self.raw.as_ptr(),
            _lifetime: PhantomData,
        }
    }

    pub fn as_tensor_mut(&mut self) -> TensorMut<'_> {
        TensorMut {
            raw: self.raw.as_ptr(),
            _lifetime: PhantomData,
        }
    }

    pub fn write_bytes(&mut self, offset: u64, data: &[u8]) -> Result<(), GpuError> {
        tensor_write(self.raw.as_ptr(), offset, data)
    }

    pub fn read_bytes(&self, offset: u64, out: &mut [u8]) -> Result<(), GpuError> {
        tensor_read(self.raw.as_ptr(), offset, out)
    }

    pub fn fill_f32(&mut self, value: f32, count: usize) -> Result<(), GpuError> {
        let count = abi_len(count)?;
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_tensor_fill_f32(
                self.raw.as_ptr(),
                value,
                count,
            ))
            .into_result()
        }
    }

    pub fn copy_from(
        &mut self,
        src: &Tensor,
        dst_offset: u64,
        src_offset: u64,
        bytes: usize,
        _batch: &mut CommandBatch,
    ) -> Result<(), GpuError> {
        let bytes = abi_len(bytes)?;
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_tensor_copy(
                self.raw.as_ptr(),
                dst_offset,
                src.raw.as_ptr(),
                src_offset,
                bytes,
            ))
            .into_result()
        }
    }

    pub fn view(&mut self, offset: u64, bytes: usize) -> Result<TensorView<'_>, GpuError> {
        let bytes = abi_len(bytes)?;
        let raw = unsafe { sys::ds4_gpu_tensor_view(self.raw.as_ptr(), offset, bytes) };
        Ok(TensorView {
            raw: NonNull::new(raw).ok_or_else(GpuError::null_tensor)?,
            _base: PhantomData,
        })
    }

    unsafe fn from_raw(raw: *mut sys::Ds4GpuTensor) -> Result<Self, GpuError> {
        Ok(Self {
            raw: NonNull::new(raw).ok_or_else(GpuError::null_tensor)?,
        })
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            sys::ds4_gpu_tensor_free(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct TensorView<'a> {
    raw: NonNull<sys::Ds4GpuTensor>,
    _base: PhantomData<&'a mut Tensor>,
}

impl TensorView<'_> {
    pub fn byte_len(&self) -> u64 {
        tensor_byte_len(self.raw.as_ptr())
    }

    pub fn as_tensor_ref(&self) -> TensorRef<'_> {
        TensorRef {
            raw: self.raw.as_ptr(),
            _lifetime: PhantomData,
        }
    }

    pub fn as_tensor_mut(&mut self) -> TensorMut<'_> {
        TensorMut {
            raw: self.raw.as_ptr(),
            _lifetime: PhantomData,
        }
    }

    pub fn write_bytes(&mut self, offset: u64, data: &[u8]) -> Result<(), GpuError> {
        tensor_write(self.raw.as_ptr(), offset, data)
    }

    pub fn read_bytes(&self, offset: u64, out: &mut [u8]) -> Result<(), GpuError> {
        tensor_read(self.raw.as_ptr(), offset, out)
    }

    pub fn fill_f32(&mut self, value: f32, count: usize) -> Result<(), GpuError> {
        let count = abi_len(count)?;
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_tensor_fill_f32(
                self.raw.as_ptr(),
                value,
                count,
            ))
            .into_result()
        }
    }
}

impl Drop for TensorView<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::ds4_gpu_tensor_free(self.raw.as_ptr());
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TensorRef<'a> {
    raw: *const sys::Ds4GpuTensor,
    _lifetime: PhantomData<&'a sys::Ds4GpuTensor>,
}

impl TensorRef<'_> {
    pub(crate) const fn raw(self) -> *const sys::Ds4GpuTensor {
        self.raw
    }
}

#[derive(Debug)]
pub struct TensorMut<'a> {
    raw: *mut sys::Ds4GpuTensor,
    _lifetime: PhantomData<&'a mut sys::Ds4GpuTensor>,
}

impl TensorMut<'_> {
    pub(crate) const fn raw(self) -> *mut sys::Ds4GpuTensor {
        self.raw
    }
}

fn abi_len(len: usize) -> Result<u64, GpuError> {
    u64::try_from(len).map_err(|_| GpuError::length_overflow())
}

fn tensor_byte_len(raw: *const sys::Ds4GpuTensor) -> u64 {
    unsafe { sys::ds4_gpu_tensor_bytes(raw) }
}

fn tensor_write(raw: *mut sys::Ds4GpuTensor, offset: u64, data: &[u8]) -> Result<(), GpuError> {
    let bytes = abi_len(data.len())?;
    unsafe {
        GpuStatus::from_raw(sys::ds4_gpu_tensor_write(
            raw,
            offset,
            data.as_ptr().cast::<c_void>(),
            bytes,
        ))
        .into_result()
    }
}

fn tensor_read(raw: *const sys::Ds4GpuTensor, offset: u64, out: &mut [u8]) -> Result<(), GpuError> {
    let bytes = abi_len(out.len())?;
    unsafe {
        GpuStatus::from_raw(sys::ds4_gpu_tensor_read(
            raw,
            offset,
            out.as_mut_ptr().cast::<c_void>(),
            bytes,
        ))
        .into_result()
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuErrorKind, GpuStatus};

    #[test]
    fn nonzero_status_is_success() {
        assert_eq!(GpuStatus::from_raw(7).into_result(), Ok(()));
        assert!(GpuStatus::from_raw(7).is_ok());
    }

    #[test]
    fn zero_status_is_error() {
        let err = GpuStatus::FAILURE.into_result().unwrap_err();
        assert_eq!(err.kind(), GpuErrorKind::BackendStatus(0));
        assert_eq!(err.code(), Some(0));
    }
}
