#![no_std]

use core::ffi::c_int;
use core::fmt;

pub use ds4_gpu_sys as sys;

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
            Err(GpuError { code: self.0 })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuError {
    code: c_int,
}

impl GpuError {
    pub const fn code(self) -> c_int {
        self.code
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DS4 GPU backend returned status {}", self.code)
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuError, GpuStatus};

    #[test]
    fn nonzero_status_is_success() {
        assert_eq!(GpuStatus::from_raw(7).into_result(), Ok(()));
        assert!(GpuStatus::from_raw(7).is_ok());
    }

    #[test]
    fn zero_status_is_error() {
        let err = GpuStatus::FAILURE.into_result().unwrap_err();
        assert_eq!(err, GpuError { code: 0 });
        assert_eq!(err.code(), 0);
    }
}
