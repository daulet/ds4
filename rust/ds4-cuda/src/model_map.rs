use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use cuda_core::{DeviceBuffer, DriverError};

use crate::substrate::CudaOxideSubstrate;

const PROT_READ: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x02;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

#[derive(Debug)]
pub enum ModelRangeError {
    Io(std::io::Error),
    EmptyModel,
    ModelTooLarge,
    InvalidRange { offset: u64, bytes: u64, size: u64 },
    Cuda(DriverError),
    MissingCachedRange { offset: u64, bytes: u64 },
}

impl fmt::Display for ModelRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "model mapping I/O failed: {err}"),
            Self::EmptyModel => write!(f, "model file is empty"),
            Self::ModelTooLarge => write!(f, "model file is too large to mmap"),
            Self::InvalidRange {
                offset,
                bytes,
                size,
            } => write!(
                f,
                "model range offset={offset} bytes={bytes} exceeds size={size}"
            ),
            Self::Cuda(err) => write!(f, "CUDA model-range operation failed: {err}"),
            Self::MissingCachedRange { offset, bytes } => {
                write!(f, "model range offset={offset} bytes={bytes} is not cached")
            }
        }
    }
}

impl std::error::Error for ModelRangeError {}

impl From<std::io::Error> for ModelRangeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<DriverError> for ModelRangeError {
    fn from(value: DriverError) -> Self {
        Self::Cuda(value)
    }
}

#[derive(Debug)]
pub struct MappedModelFile {
    _file: File,
    ptr: NonNull<u8>,
    size: u64,
}

impl MappedModelFile {
    pub fn open(path: &Path) -> Result<Self, ModelRangeError> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        if size == 0 {
            return Err(ModelRangeError::EmptyModel);
        }
        let length = usize::try_from(size).map_err(|_| ModelRangeError::ModelTooLarge)?;
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                length,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let ptr = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            unsafe {
                let _ = munmap(ptr, length);
            }
            ModelRangeError::Io(std::io::Error::other("mmap returned null pointer"))
        })?;
        Ok(Self {
            _file: file,
            ptr,
            size,
        })
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub fn range(&self, offset: u64, bytes: u64) -> Result<&[u8], ModelRangeError> {
        if bytes == 0 || offset > self.size || bytes > self.size - offset {
            return Err(ModelRangeError::InvalidRange {
                offset,
                bytes,
                size: self.size,
            });
        }
        let offset = usize::try_from(offset).map_err(|_| ModelRangeError::ModelTooLarge)?;
        let bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        Ok(unsafe { slice::from_raw_parts(self.ptr.as_ptr().add(offset), bytes) })
    }
}

impl Drop for MappedModelFile {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr.as_ptr().cast::<c_void>(), self.size as usize);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheOutcome {
    Inserted,
    Reused,
}

struct CachedModelRange {
    offset: u64,
    bytes: u64,
    device: DeviceBuffer<u8>,
}

#[derive(Default)]
pub struct ModelRangeCache {
    ranges: Vec<CachedModelRange>,
}

impl ModelRangeCache {
    pub fn cache_range(
        &mut self,
        substrate: &CudaOxideSubstrate,
        model: &MappedModelFile,
        offset: u64,
        bytes: u64,
    ) -> Result<CacheOutcome, ModelRangeError> {
        let source = model.range(offset, bytes)?;
        if self.find(offset, bytes).is_some() {
            return Ok(CacheOutcome::Reused);
        }
        let device = substrate.upload(source)?;
        substrate.synchronize()?;
        self.ranges.push(CachedModelRange {
            offset,
            bytes,
            device,
        });
        Ok(CacheOutcome::Inserted)
    }

    pub fn readback(
        &self,
        substrate: &CudaOxideSubstrate,
        offset: u64,
        bytes: u64,
    ) -> Result<Vec<u8>, ModelRangeError> {
        let range = self
            .find(offset, bytes)
            .ok_or(ModelRangeError::MissingCachedRange { offset, bytes })?;
        Ok(substrate.download(&range.device)?)
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    fn find(&self, offset: u64, bytes: u64) -> Option<&CachedModelRange> {
        self.ranges
            .iter()
            .find(|range| range.offset == offset && range.bytes == bytes)
    }
}
