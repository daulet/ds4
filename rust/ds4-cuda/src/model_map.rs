use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::os::raw::c_int;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use cuda_core::{DeviceBuffer, DriverError, ReadOnlyRegisteredHostMemory};

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
    InvalidPageSize,
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
            Self::InvalidPageSize => write!(f, "could not determine model mapping page size"),
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
    mapped_bytes: usize,
}

impl MappedModelFile {
    pub fn open(path: &Path) -> Result<Self, ModelRangeError> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        if size == 0 {
            return Err(ModelRangeError::EmptyModel);
        }
        let mapped_bytes = usize::try_from(round_up_to_page(size, page_size()?)?)
            .map_err(|_| ModelRangeError::ModelTooLarge)?;
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                mapped_bytes,
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
                let _ = munmap(ptr, mapped_bytes);
            }
            ModelRangeError::Io(std::io::Error::other("mmap returned null pointer"))
        })?;
        Ok(Self {
            _file: file,
            ptr,
            size,
            mapped_bytes,
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

    pub fn read_file_range(&self, offset: u64, bytes: u64) -> Result<Vec<u8>, ModelRangeError> {
        self.range(offset, bytes)?;
        let bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        let mut staged = vec![0_u8; bytes];
        self._file.read_exact_at(&mut staged, offset)?;
        Ok(staged)
    }

    pub fn registered_range_layout(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<RegisteredRangeLayout, ModelRangeError> {
        self.range(offset, bytes)?;
        let page_size = page_size()?;
        let registered_offset = offset - (offset % page_size);
        let end = offset + bytes;
        let registered_end = round_up_to_page(end, page_size)?;
        Ok(RegisteredRangeLayout {
            page_size,
            registered_offset,
            registered_bytes: registered_end - registered_offset,
            device_offset: offset - registered_offset,
        })
    }

    fn registered_source(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<(RegisteredRangeLayout, &[u8]), ModelRangeError> {
        let layout = self.registered_range_layout(offset, bytes)?;
        let registered_offset = usize::try_from(layout.registered_offset)
            .map_err(|_| ModelRangeError::ModelTooLarge)?;
        let registered_bytes =
            usize::try_from(layout.registered_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        debug_assert!(registered_offset + registered_bytes <= self.mapped_bytes);
        let source = unsafe {
            slice::from_raw_parts(self.ptr.as_ptr().add(registered_offset), registered_bytes)
        };
        Ok((layout, source))
    }
}

impl Drop for MappedModelFile {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr.as_ptr().cast::<c_void>(), self.mapped_bytes);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisteredRangeLayout {
    pub page_size: u64,
    pub registered_offset: u64,
    pub registered_bytes: u64,
    pub device_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheOutcome {
    Inserted,
    Reused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelRangeStrategy {
    MmapDeviceCopy,
    FileStagedDeviceCopy,
    ReadOnlyRegisteredOrMmapDeviceCopy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegisteredRangeResolution {
    ReadOnlyMapped,
    MmapDeviceCopyFallback(DriverError),
}

enum CachedRangeStorage<'model> {
    Device(DeviceBuffer<u8>),
    ReadOnlyRegistered {
        _registration: ReadOnlyRegisteredHostMemory<'model, u8>,
        requested_device_ptr: cuda_core::sys::CUdeviceptr,
    },
}

struct CachedModelRange<'model> {
    strategy: ModelRangeStrategy,
    offset: u64,
    bytes: u64,
    storage: CachedRangeStorage<'model>,
    registered_resolution: Option<RegisteredRangeResolution>,
}

#[derive(Default)]
pub struct ModelRangeCache<'model> {
    ranges: Vec<CachedModelRange<'model>>,
}

impl<'model> ModelRangeCache<'model> {
    pub fn cache_range(
        &mut self,
        substrate: &CudaOxideSubstrate,
        model: &'model MappedModelFile,
        offset: u64,
        bytes: u64,
    ) -> Result<CacheOutcome, ModelRangeError> {
        self.cache_range_with_strategy(
            substrate,
            model,
            offset,
            bytes,
            ModelRangeStrategy::MmapDeviceCopy,
        )
    }

    pub fn cache_range_with_strategy(
        &mut self,
        substrate: &CudaOxideSubstrate,
        model: &'model MappedModelFile,
        offset: u64,
        bytes: u64,
        strategy: ModelRangeStrategy,
    ) -> Result<CacheOutcome, ModelRangeError> {
        model.range(offset, bytes)?;
        if self.find(strategy, offset, bytes).is_some() {
            return Ok(CacheOutcome::Reused);
        }
        let (storage, registered_resolution) = match strategy {
            ModelRangeStrategy::MmapDeviceCopy => (
                CachedRangeStorage::Device(substrate.upload(model.range(offset, bytes)?)?),
                None,
            ),
            ModelRangeStrategy::FileStagedDeviceCopy => {
                let staged = model.read_file_range(offset, bytes)?;
                let device = substrate.upload(&staged)?;
                substrate.synchronize()?;
                (CachedRangeStorage::Device(device), None)
            }
            ModelRangeStrategy::ReadOnlyRegisteredOrMmapDeviceCopy => {
                let (layout, source) = model.registered_source(offset, bytes)?;
                match substrate.register_read_only_host_range(source) {
                    Ok(registration) => (
                        CachedRangeStorage::ReadOnlyRegistered {
                            requested_device_ptr: registration.cu_deviceptr()
                                + layout.device_offset,
                            _registration: registration,
                        },
                        Some(RegisteredRangeResolution::ReadOnlyMapped),
                    ),
                    Err(err) => (
                        CachedRangeStorage::Device(substrate.upload(model.range(offset, bytes)?)?),
                        Some(RegisteredRangeResolution::MmapDeviceCopyFallback(err)),
                    ),
                }
            }
        };
        substrate.synchronize()?;
        self.ranges.push(CachedModelRange {
            strategy,
            offset,
            bytes,
            storage,
            registered_resolution,
        });
        Ok(CacheOutcome::Inserted)
    }

    pub fn readback(
        &self,
        substrate: &CudaOxideSubstrate,
        offset: u64,
        bytes: u64,
    ) -> Result<Vec<u8>, ModelRangeError> {
        self.readback_with_strategy(substrate, offset, bytes, ModelRangeStrategy::MmapDeviceCopy)
    }

    pub fn readback_with_strategy(
        &self,
        substrate: &CudaOxideSubstrate,
        offset: u64,
        bytes: u64,
        strategy: ModelRangeStrategy,
    ) -> Result<Vec<u8>, ModelRangeError> {
        let range = self
            .find(strategy, offset, bytes)
            .ok_or(ModelRangeError::MissingCachedRange { offset, bytes })?;
        match &range.storage {
            CachedRangeStorage::Device(device) => Ok(substrate.download(device)?),
            CachedRangeStorage::ReadOnlyRegistered {
                requested_device_ptr,
                ..
            } => {
                let bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
                Ok(unsafe { substrate.download_u8_device_ptr(*requested_device_ptr, bytes)? })
            }
        }
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn registered_resolution(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Option<RegisteredRangeResolution> {
        self.find(
            ModelRangeStrategy::ReadOnlyRegisteredOrMmapDeviceCopy,
            offset,
            bytes,
        )
        .and_then(|range| range.registered_resolution)
    }

    fn find(
        &self,
        strategy: ModelRangeStrategy,
        offset: u64,
        bytes: u64,
    ) -> Option<&CachedModelRange<'model>> {
        self.ranges.iter().find(|range| {
            range.strategy == strategy && range.offset == offset && range.bytes == bytes
        })
    }
}

fn page_size() -> Result<u64, ModelRangeError> {
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size <= 0 {
        return Err(ModelRangeError::InvalidPageSize);
    }
    Ok(size as u64)
}

fn round_up_to_page(value: u64, page_size: u64) -> Result<u64, ModelRangeError> {
    value
        .checked_add(page_size - 1)
        .map(|end| (end / page_size) * page_size)
        .ok_or(ModelRangeError::ModelTooLarge)
}
