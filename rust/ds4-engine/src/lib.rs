use std::error::Error;
use std::ffi::{c_char, c_float, c_int, CString, NulError};
use std::fmt;
use std::ptr::NonNull;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Metal,
    Cuda,
    Cpu,
}

impl Backend {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    const fn as_raw(self) -> c_int {
        match self {
            Self::Metal => 0,
            Self::Cuda => 1,
            Self::Cpu => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineOptions<'a> {
    pub model_path: &'a str,
    pub backend: Backend,
    pub warm_weights: bool,
    pub quality: bool,
}

impl<'a> EngineOptions<'a> {
    pub const fn new(model_path: &'a str, backend: Backend) -> Self {
        Self {
            model_path,
            backend,
            warm_weights: false,
            quality: false,
        }
    }
}

#[derive(Debug)]
pub struct Engine {
    raw: NonNull<RawEngine>,
}

impl Engine {
    pub fn open(options: &EngineOptions<'_>) -> Result<Self, EngineError> {
        let model_path = CString::new(options.model_path)?;
        let raw_options = RawEngineOptions {
            model_path: model_path.as_ptr(),
            mtp_path: std::ptr::null(),
            backend: options.backend.as_raw(),
            n_threads: 0,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            directional_steering_file: std::ptr::null(),
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            warm_weights: options.warm_weights,
            quality: options.quality,
        };
        let mut raw = std::ptr::null_mut();
        let rc = unsafe { ds4_engine_open(&mut raw, &raw_options) };
        if rc != 0 {
            return Err(EngineError::open_failed(rc));
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or(EngineError::null_engine())?,
        })
    }

    pub fn print_summary(&self) {
        unsafe {
            ds4_engine_summary(self.raw.as_ptr());
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            ds4_engine_close(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct EngineError {
    kind: EngineErrorKind,
}

#[derive(Debug)]
enum EngineErrorKind {
    InvalidString(NulError),
    OpenFailed(c_int),
    NullEngine,
}

impl EngineError {
    fn open_failed(code: c_int) -> Self {
        Self {
            kind: EngineErrorKind::OpenFailed(code),
        }
    }

    fn null_engine() -> Self {
        Self {
            kind: EngineErrorKind::NullEngine,
        }
    }
}

impl From<NulError> for EngineError {
    fn from(error: NulError) -> Self {
        Self {
            kind: EngineErrorKind::InvalidString(error),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            EngineErrorKind::InvalidString(error) => write!(f, "invalid C string: {error}"),
            EngineErrorKind::OpenFailed(code) => write!(f, "ds4_engine_open failed with {code}"),
            EngineErrorKind::NullEngine => write!(f, "ds4_engine_open returned a null engine"),
        }
    }
}

impl Error for EngineError {}

#[repr(C)]
struct RawEngine {
    _private: [u8; 0],
}

#[repr(C)]
struct RawEngineOptions {
    model_path: *const c_char,
    mtp_path: *const c_char,
    backend: c_int,
    n_threads: c_int,
    mtp_draft_tokens: c_int,
    mtp_margin: c_float,
    directional_steering_file: *const c_char,
    directional_steering_attn: c_float,
    directional_steering_ffn: c_float,
    warm_weights: bool,
    quality: bool,
}

unsafe extern "C" {
    fn ds4_engine_open(out: *mut *mut RawEngine, opt: *const RawEngineOptions) -> c_int;
    fn ds4_engine_close(engine: *mut RawEngine);
    fn ds4_engine_summary(engine: *mut RawEngine);
}

#[cfg(test)]
mod tests {
    use super::{Backend, EngineOptions, RawEngineOptions};

    #[test]
    fn backend_values_match_c_enum() {
        assert_eq!(Backend::Metal.as_raw(), 0);
        assert_eq!(Backend::Cuda.as_raw(), 1);
        assert_eq!(Backend::Cpu.as_raw(), 2);
    }

    #[test]
    fn raw_options_keeps_c_bool_layout_small() {
        assert_eq!(std::mem::size_of::<bool>(), 1);
        assert!(std::mem::size_of::<RawEngineOptions>() >= 56);
    }

    #[test]
    fn options_default_runtime_flags_match_c_cli_inspect() {
        let options = EngineOptions::new("model.gguf", Backend::Cuda);
        assert!(!options.warm_weights);
        assert!(!options.quality);
    }
}
