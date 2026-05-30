use std::ffi::c_void;
use std::fmt;
use std::fs::File;
#[cfg(target_os = "linux")]
use std::fs::OpenOptions;
use std::os::raw::c_int;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::slice;
use std::time::{Duration, Instant};

use cuda_core::{
    CudaEvent, DeviceBuffer, DriverError, PinnedHostBuffer, ReadOnlyPageableHostMemory,
    ReadOnlyRegisteredHostMemory,
};

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
    InvalidStagingConfig(&'static str),
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
            Self::InvalidStagingConfig(reason) => {
                write!(f, "invalid asynchronous staging configuration: {reason}")
            }
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

    fn page_aligned_source(
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

    fn discard_source_pages(
        &self,
        offset: u64,
        bytes: u64,
        keep_source_pages: bool,
    ) -> Result<SourcePageDiscardAttempt, ModelRangeError> {
        self.range(offset, bytes)?;
        if keep_source_pages {
            return Ok(SourcePageDiscardAttempt::default());
        }
        #[cfg(target_os = "linux")]
        {
            let layout = self.registered_range_layout(offset, bytes)?;
            let file_offset =
                libc::off_t::try_from(offset).map_err(|_| ModelRangeError::ModelTooLarge)?;
            let file_bytes =
                libc::off_t::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
            let mapping_bytes = usize::try_from(layout.registered_bytes)
                .map_err(|_| ModelRangeError::ModelTooLarge)?;
            let mapping_offset = usize::try_from(layout.registered_offset)
                .map_err(|_| ModelRangeError::ModelTooLarge)?;
            unsafe {
                let _ = libc::posix_fadvise(
                    self._file.as_raw_fd(),
                    file_offset,
                    file_bytes,
                    libc::POSIX_FADV_DONTNEED,
                );
                let _ = libc::posix_madvise(
                    self.ptr.as_ptr().add(mapping_offset).cast::<c_void>(),
                    mapping_bytes,
                    libc::POSIX_MADV_DONTNEED,
                );
            }
            Ok(SourcePageDiscardAttempt {
                file_bytes: bytes,
                mapping_bytes: layout.registered_bytes,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (offset, bytes);
            Ok(SourcePageDiscardAttempt::default())
        }
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

pub struct PrefetchedPageableModelRange<'model> {
    layout: RegisteredRangeLayout,
    requested_bytes: u64,
    pageable: ReadOnlyPageableHostMemory<'model, u8>,
}

impl PrefetchedPageableModelRange<'_> {
    pub const fn layout(&self) -> RegisteredRangeLayout {
        self.layout
    }

    pub fn requested_device_ptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.pageable.cu_deviceptr() + self.layout.device_offset
    }

    pub fn readback(&self, substrate: &CudaOxideSubstrate) -> Result<Vec<u8>, ModelRangeError> {
        let bytes =
            usize::try_from(self.requested_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        Ok(unsafe { substrate.download_u8_device_ptr(self.requested_device_ptr(), bytes)? })
    }
}

pub fn prefetch_pageable_read_only_range<'model>(
    substrate: &CudaOxideSubstrate,
    model: &'model MappedModelFile,
    offset: u64,
    bytes: u64,
) -> Result<PrefetchedPageableModelRange<'model>, ModelRangeError> {
    let (layout, source) = model.page_aligned_source(offset, bytes)?;
    let pageable = substrate.pageable_read_only_range(source)?;
    substrate.prefetch_pageable_read_mostly_to_device(&pageable)?;
    Ok(PrefetchedPageableModelRange {
        layout,
        requested_bytes: bytes,
        pageable,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PinnedStagePolicy {
    Buffered,
    DirectIoOrBufferedFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PinnedStageResolution {
    Buffered,
    DirectIo {
        alignment: u64,
        read_offset: u64,
        read_bytes: u64,
        payload_offset: u64,
    },
    BufferedFallback,
}

pub struct PinnedStagedModelRange {
    device: DeviceBuffer<u8>,
    resolution: PinnedStageResolution,
}

impl PinnedStagedModelRange {
    pub const fn resolution(&self) -> PinnedStageResolution {
        self.resolution
    }

    pub fn readback(&self, substrate: &CudaOxideSubstrate) -> Result<Vec<u8>, ModelRangeError> {
        Ok(substrate.download(&self.device)?)
    }
}

pub fn stage_pinned_model_range(
    substrate: &CudaOxideSubstrate,
    model: &MappedModelFile,
    offset: u64,
    bytes: u64,
    policy: PinnedStagePolicy,
) -> Result<PinnedStagedModelRange, ModelRangeError> {
    model.range(offset, bytes)?;
    let (staging, payload_offset, resolution) = match policy {
        PinnedStagePolicy::Buffered => buffered_pinned_source(substrate, model, offset, bytes)?,
        PinnedStagePolicy::DirectIoOrBufferedFallback => {
            match try_direct_pinned_source(substrate, model, offset, bytes)? {
                Some(source) => source,
                None => {
                    let (staging, payload_offset, _) =
                        buffered_pinned_source(substrate, model, offset, bytes)?;
                    (
                        staging,
                        payload_offset,
                        PinnedStageResolution::BufferedFallback,
                    )
                }
            }
        }
    };
    let bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
    let device = substrate.upload_pinned_u8_range(&staging, payload_offset, bytes)?;
    Ok(PinnedStagedModelRange { device, resolution })
}

fn buffered_pinned_source(
    substrate: &CudaOxideSubstrate,
    model: &MappedModelFile,
    offset: u64,
    bytes: u64,
) -> Result<(PinnedHostBuffer<u8>, usize, PinnedStageResolution), ModelRangeError> {
    let bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
    let mut staging = substrate.pinned_zeroed(bytes)?;
    model._file.read_exact_at(staging.as_mut_slice(), offset)?;
    Ok((staging, 0, PinnedStageResolution::Buffered))
}

#[cfg(target_os = "linux")]
fn try_direct_pinned_source(
    substrate: &CudaOxideSubstrate,
    model: &MappedModelFile,
    offset: u64,
    bytes: u64,
) -> Result<Option<(PinnedHostBuffer<u8>, usize, PinnedStageResolution)>, ModelRangeError> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let alignment = (model._file.metadata()?.blksize() as u64).max(512);
    let read_offset = offset - (offset % alignment);
    let payload_delta = offset - read_offset;
    let payload_end = payload_delta
        .checked_add(bytes)
        .ok_or(ModelRangeError::ModelTooLarge)?;
    let read_bytes = round_up_to_alignment(payload_end, alignment)?;
    if read_offset > model.size || read_bytes > model.size - read_offset {
        return Ok(None);
    }
    let direct_path = format!("/proc/self/fd/{}", model._file.as_raw_fd());
    let direct_file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(direct_path)
    {
        Ok(file) => file,
        Err(_) => return Ok(None),
    };
    let stage_bytes = read_bytes
        .checked_add(alignment)
        .ok_or(ModelRangeError::ModelTooLarge)?;
    let stage_bytes = usize::try_from(stage_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
    let mut staging = substrate.pinned_zeroed(stage_bytes)?;
    let base = staging.as_ptr() as usize;
    let alignment_usize = usize::try_from(alignment).map_err(|_| ModelRangeError::ModelTooLarge)?;
    let aligned_delta = (alignment_usize - (base % alignment_usize)) % alignment_usize;
    let read_bytes_usize =
        usize::try_from(read_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
    let direct_window =
        &mut staging.as_mut_slice()[aligned_delta..aligned_delta + read_bytes_usize];
    if direct_file
        .read_exact_at(direct_window, read_offset)
        .is_err()
    {
        return Ok(None);
    }
    let payload_delta =
        usize::try_from(payload_delta).map_err(|_| ModelRangeError::ModelTooLarge)?;
    Ok(Some((
        staging,
        aligned_delta + payload_delta,
        PinnedStageResolution::DirectIo {
            alignment,
            read_offset,
            read_bytes,
            payload_offset: payload_delta as u64,
        },
    )))
}

#[cfg(not(target_os = "linux"))]
fn try_direct_pinned_source(
    _substrate: &CudaOxideSubstrate,
    _model: &MappedModelFile,
    _offset: u64,
    _bytes: u64,
) -> Result<Option<(PinnedHostBuffer<u8>, usize, PinnedStageResolution)>, ModelRangeError> {
    Ok(None)
}

const ASYNC_STAGE_SLOTS: usize = 4;
const ARENA_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLoadProgressMode {
    Disabled,
    NonTty,
    Tty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsyncPinnedCacheConfig {
    pub copy_chunk_bytes: u64,
    pub arena_chunk_bytes: u64,
    pub cache_limit_bytes: u64,
    pub keep_source_pages: bool,
    pub progress_mode: ModelLoadProgressMode,
}

impl AsyncPinnedCacheConfig {
    fn validate(self) -> Result<Self, ModelRangeError> {
        if self.copy_chunk_bytes == 0 {
            return Err(ModelRangeError::InvalidStagingConfig(
                "copy_chunk_bytes must be non-zero",
            ));
        }
        if self.arena_chunk_bytes == 0 {
            return Err(ModelRangeError::InvalidStagingConfig(
                "arena_chunk_bytes must be non-zero",
            ));
        }
        if self.cache_limit_bytes == 0 {
            return Err(ModelRangeError::InvalidStagingConfig(
                "cache_limit_bytes must be non-zero",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncPinnedCacheOutcome {
    Inserted,
    Reused,
    BudgetFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectIoPolicyState {
    Unavailable,
    Enabled { alignment: u64 },
    DisabledAfterError { raw_os_error: i32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsyncPinnedCacheStats {
    pub stage_slots: usize,
    pub chunks_uploaded: u64,
    pub stage_slot_reuse_waits: u64,
    pub events_recorded: u64,
    pub direct_io_chunks: u64,
    pub buffered_chunks: u64,
    pub arena_count: usize,
    pub arena_bytes: u64,
    pub range_count: usize,
    pub range_bytes: u64,
    pub budget_fallbacks: u64,
    pub exact_range_hits: u64,
    pub containing_range_hits: u64,
    pub source_file_discard_calls: u64,
    pub source_file_discard_bytes: u64,
    pub source_mapping_discard_calls: u64,
    pub source_mapping_discard_bytes: u64,
    pub progress_notes: u64,
    pub progress_messages: u64,
    pub direct_io_state: DirectIoPolicyState,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SourcePageDiscardAttempt {
    file_bytes: u64,
    mapping_bytes: u64,
}

struct ModelLoadProgress {
    mode: ModelLoadProgressMode,
    next: u64,
    last: Option<Instant>,
    started: bool,
    notes: u64,
    messages: u64,
}

impl ModelLoadProgress {
    fn new(mode: ModelLoadProgressMode) -> Self {
        Self {
            mode,
            next: 0,
            last: None,
            started: false,
            notes: 0,
            messages: 0,
        }
    }

    fn note(&mut self, cached_bytes: u64) {
        if self.mode == ModelLoadProgressMode::Disabled {
            return;
        }
        self.notes += 1;
        let now = Instant::now();
        let (step, interval) = match self.mode {
            ModelLoadProgressMode::Disabled => unreachable!(),
            ModelLoadProgressMode::NonTty => (16_u64 << 30, Duration::from_secs(10)),
            ModelLoadProgressMode::Tty => (2_u64 << 30, Duration::from_secs(2)),
        };
        if !self.started {
            self.started = true;
            self.next = step;
            self.last = Some(now);
            self.messages += 1;
            match self.mode {
                ModelLoadProgressMode::NonTty => {
                    eprintln!("ds4: CUDA loading model tensors into device cache");
                }
                ModelLoadProgressMode::Tty => {
                    eprint!("ds4: CUDA loading model tensors into device cache: 0.00 GiB");
                }
                ModelLoadProgressMode::Disabled => unreachable!(),
            }
        }
        if cached_bytes < self.next
            && now.duration_since(self.last.expect("started progress has timestamp")) < interval
        {
            return;
        }
        self.messages += 1;
        match self.mode {
            ModelLoadProgressMode::NonTty => {
                eprintln!(
                    "ds4: CUDA loading model tensors {:.2} GiB cached",
                    cached_bytes as f64 / 1_073_741_824.0
                );
            }
            ModelLoadProgressMode::Tty => {
                eprint!(
                    "\rds4: CUDA loading model tensors into device cache: {:.2} GiB",
                    cached_bytes as f64 / 1_073_741_824.0
                );
            }
            ModelLoadProgressMode::Disabled => unreachable!(),
        }
        self.last = Some(now);
        while self.next <= cached_bytes {
            let Some(next) = self.next.checked_add(step) else {
                self.next = u64::MAX;
                break;
            };
            self.next = next;
        }
    }
}

struct AsyncPinnedStageSlot {
    staging: PinnedHostBuffer<u8>,
    event: Option<CudaEvent>,
}

struct AsyncPinnedArena {
    device: DeviceBuffer<u8>,
    bytes: u64,
    used: u64,
}

struct AsyncPinnedCachedRange {
    offset: u64,
    bytes: u64,
    arena_index: usize,
    device_offset: u64,
}

struct StageRead {
    payload_offset: usize,
    direct_io: bool,
}

struct DirectIoReader {
    #[cfg(target_os = "linux")]
    direct_file: Option<File>,
    alignment: u64,
    disabled_raw_os_error: Option<i32>,
}

impl DirectIoReader {
    fn open(model: &MappedModelFile) -> Result<Self, ModelRangeError> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

            let alignment = (model._file.metadata()?.blksize() as u64).max(512);
            let direct_path = format!("/proc/self/fd/{}", model._file.as_raw_fd());
            let direct_file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(direct_path)
                .ok();
            Ok(Self {
                direct_file,
                alignment,
                disabled_raw_os_error: None,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = model;
            Ok(Self {
                alignment: 1,
                disabled_raw_os_error: None,
            })
        }
    }

    fn state(&self) -> DirectIoPolicyState {
        if let Some(raw_os_error) = self.disabled_raw_os_error {
            return DirectIoPolicyState::DisabledAfterError { raw_os_error };
        }
        #[cfg(target_os = "linux")]
        if self.direct_file.is_some() {
            return DirectIoPolicyState::Enabled {
                alignment: self.alignment,
            };
        }
        DirectIoPolicyState::Unavailable
    }

    fn stage_alignment(&self) -> u64 {
        self.alignment.max(1)
    }

    #[cfg(target_os = "linux")]
    fn record_direct_read_error(&mut self, err: &std::io::Error) {
        if direct_io_error_disables(err.raw_os_error()) {
            self.direct_file = None;
            self.disabled_raw_os_error = err.raw_os_error();
            self.alignment = 1;
        }
    }

    fn read_into(
        &mut self,
        model: &MappedModelFile,
        staging: &mut PinnedHostBuffer<u8>,
        offset: u64,
        bytes: u64,
    ) -> Result<StageRead, ModelRangeError> {
        #[cfg(target_os = "linux")]
        if self.direct_file.is_some() {
            let read_offset = offset - (offset % self.alignment);
            let payload_delta = offset - read_offset;
            let read_bytes = round_up_to_alignment(
                payload_delta
                    .checked_add(bytes)
                    .ok_or(ModelRangeError::ModelTooLarge)?,
                self.alignment,
            )?;
            let alignment =
                usize::try_from(self.alignment).map_err(|_| ModelRangeError::ModelTooLarge)?;
            let base = staging.as_ptr() as usize;
            let aligned_delta = (alignment - (base % alignment)) % alignment;
            if read_offset <= model.size
                && read_bytes <= model.size - read_offset
                && read_bytes <= (staging.len() - aligned_delta) as u64
            {
                let payload_delta =
                    usize::try_from(payload_delta).map_err(|_| ModelRangeError::ModelTooLarge)?;
                let read_bytes =
                    usize::try_from(read_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
                let result = self
                    .direct_file
                    .as_ref()
                    .expect("direct file was checked above")
                    .read_exact_at(
                        &mut staging.as_mut_slice()[aligned_delta..aligned_delta + read_bytes],
                        read_offset,
                    );
                match result {
                    Ok(()) => {
                        return Ok(StageRead {
                            payload_offset: aligned_delta + payload_delta,
                            direct_io: true,
                        });
                    }
                    Err(err) => {
                        self.record_direct_read_error(&err);
                    }
                }
            }
        }
        let bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        model
            ._file
            .read_exact_at(&mut staging.as_mut_slice()[..bytes], offset)?;
        Ok(StageRead {
            payload_offset: 0,
            direct_io: false,
        })
    }
}

#[cfg(target_os = "linux")]
fn direct_io_error_disables(raw_os_error: Option<i32>) -> bool {
    raw_os_error.is_some_and(|raw| {
        [libc::EINVAL, libc::EFAULT, libc::ENOTSUP, libc::EOPNOTSUPP].contains(&raw)
    })
}

pub struct AsyncPinnedRangeCache<'model> {
    model: &'model MappedModelFile,
    config: AsyncPinnedCacheConfig,
    direct_io: DirectIoReader,
    slots: Vec<AsyncPinnedStageSlot>,
    arenas: Vec<AsyncPinnedArena>,
    ranges: Vec<AsyncPinnedCachedRange>,
    chunks_uploaded: u64,
    stage_slot_reuse_waits: u64,
    events_recorded: u64,
    direct_io_chunks: u64,
    buffered_chunks: u64,
    range_bytes: u64,
    budget_fallbacks: u64,
    exact_range_hits: u64,
    containing_range_hits: u64,
    source_file_discard_calls: u64,
    source_file_discard_bytes: u64,
    source_mapping_discard_calls: u64,
    source_mapping_discard_bytes: u64,
    progress: ModelLoadProgress,
}

impl<'model> AsyncPinnedRangeCache<'model> {
    pub fn new(
        model: &'model MappedModelFile,
        config: AsyncPinnedCacheConfig,
    ) -> Result<Self, ModelRangeError> {
        Ok(Self {
            model,
            config: config.validate()?,
            direct_io: DirectIoReader::open(model)?,
            slots: Vec::new(),
            arenas: Vec::new(),
            ranges: Vec::new(),
            chunks_uploaded: 0,
            stage_slot_reuse_waits: 0,
            events_recorded: 0,
            direct_io_chunks: 0,
            buffered_chunks: 0,
            range_bytes: 0,
            budget_fallbacks: 0,
            exact_range_hits: 0,
            containing_range_hits: 0,
            source_file_discard_calls: 0,
            source_file_discard_bytes: 0,
            source_mapping_discard_calls: 0,
            source_mapping_discard_bytes: 0,
            progress: ModelLoadProgress::new(config.progress_mode),
        })
    }

    pub fn cache_range(
        &mut self,
        substrate: &CudaOxideSubstrate,
        offset: u64,
        bytes: u64,
    ) -> Result<AsyncPinnedCacheOutcome, ModelRangeError> {
        self.model.range(offset, bytes)?;
        if self.find_exact(offset, bytes).is_some() {
            self.exact_range_hits += 1;
            return Ok(AsyncPinnedCacheOutcome::Reused);
        }
        if self.find_containing(offset, bytes).is_some() {
            self.containing_range_hits += 1;
            return Ok(AsyncPinnedCacheOutcome::Reused);
        }
        let Some((arena_index, device_offset)) = self.reserve_range(substrate, bytes)? else {
            self.budget_fallbacks += 1;
            return Ok(AsyncPinnedCacheOutcome::BudgetFallback);
        };
        self.upload_range(substrate, arena_index, device_offset, offset, bytes)?;
        self.ranges.push(AsyncPinnedCachedRange {
            offset,
            bytes,
            arena_index,
            device_offset,
        });
        self.range_bytes += bytes;
        self.progress.note(self.range_bytes);
        Ok(AsyncPinnedCacheOutcome::Inserted)
    }

    pub fn readback(
        &self,
        substrate: &CudaOxideSubstrate,
        offset: u64,
        bytes: u64,
    ) -> Result<Vec<u8>, ModelRangeError> {
        let range = self
            .find_containing(offset, bytes)
            .ok_or(ModelRangeError::MissingCachedRange { offset, bytes })?;
        let device = &self.arenas[range.arena_index].device;
        let requested_bytes = usize::try_from(bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        let requested_device_offset = range
            .device_offset
            .checked_add(offset - range.offset)
            .ok_or(ModelRangeError::ModelTooLarge)?;
        Ok(unsafe {
            substrate.download_u8_device_ptr(
                device.cu_deviceptr() + requested_device_offset,
                requested_bytes,
            )?
        })
    }

    pub fn stats(&self) -> AsyncPinnedCacheStats {
        AsyncPinnedCacheStats {
            stage_slots: self.slots.len(),
            chunks_uploaded: self.chunks_uploaded,
            stage_slot_reuse_waits: self.stage_slot_reuse_waits,
            events_recorded: self.events_recorded,
            direct_io_chunks: self.direct_io_chunks,
            buffered_chunks: self.buffered_chunks,
            arena_count: self.arenas.len(),
            arena_bytes: self.arenas.iter().map(|arena| arena.bytes).sum(),
            range_count: self.ranges.len(),
            range_bytes: self.range_bytes,
            budget_fallbacks: self.budget_fallbacks,
            exact_range_hits: self.exact_range_hits,
            containing_range_hits: self.containing_range_hits,
            source_file_discard_calls: self.source_file_discard_calls,
            source_file_discard_bytes: self.source_file_discard_bytes,
            source_mapping_discard_calls: self.source_mapping_discard_calls,
            source_mapping_discard_bytes: self.source_mapping_discard_bytes,
            progress_notes: self.progress.notes,
            progress_messages: self.progress.messages,
            direct_io_state: self.direct_io.state(),
        }
    }

    fn find_exact(&self, offset: u64, bytes: u64) -> Option<&AsyncPinnedCachedRange> {
        self.ranges
            .iter()
            .find(|range| range.offset == offset && range.bytes == bytes)
    }

    fn find_containing(&self, offset: u64, bytes: u64) -> Option<&AsyncPinnedCachedRange> {
        let end = offset.checked_add(bytes)?;
        self.ranges.iter().find(|range| {
            let range_end = range.offset.checked_add(range.bytes);
            offset >= range.offset && range_end.is_some_and(|range_end| end <= range_end)
        })
    }

    fn reserve_range(
        &mut self,
        substrate: &CudaOxideSubstrate,
        bytes: u64,
    ) -> Result<Option<(usize, u64)>, ModelRangeError> {
        if self.range_bytes > self.config.cache_limit_bytes
            || bytes > self.config.cache_limit_bytes - self.range_bytes
        {
            return Ok(None);
        }
        let aligned = round_up_to_alignment(bytes, ARENA_ALIGNMENT)?;
        for (index, arena) in self.arenas.iter_mut().enumerate() {
            let used = round_up_to_alignment(arena.used, ARENA_ALIGNMENT)?;
            if used <= arena.bytes && aligned <= arena.bytes - used {
                arena.used = used + aligned;
                return Ok(Some((index, used)));
            }
        }
        if aligned > self.config.cache_limit_bytes - self.range_bytes {
            return Ok(None);
        }
        let arena_bytes = self.config.arena_chunk_bytes.max(aligned);
        let arena_len = usize::try_from(arena_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        let device = substrate.zeroed(arena_len)?;
        self.arenas.push(AsyncPinnedArena {
            device,
            bytes: arena_bytes,
            used: aligned,
        });
        Ok(Some((self.arenas.len() - 1, 0)))
    }

    fn upload_range(
        &mut self,
        substrate: &CudaOxideSubstrate,
        arena_index: usize,
        device_offset: u64,
        offset: u64,
        bytes: u64,
    ) -> Result<(), ModelRangeError> {
        let stage_bytes = self
            .config
            .copy_chunk_bytes
            .checked_add(
                self.direct_io
                    .stage_alignment()
                    .checked_mul(2)
                    .ok_or(ModelRangeError::ModelTooLarge)?,
            )
            .ok_or(ModelRangeError::ModelTooLarge)?;
        self.ensure_slots(substrate, stage_bytes)?;
        let device_offset =
            usize::try_from(device_offset).map_err(|_| ModelRangeError::ModelTooLarge)?;
        let device = &self.arenas[arena_index].device;
        let mut copied = 0_u64;
        let mut chunk_index = 0_u64;
        let upload_result = (|| {
            while copied < bytes {
                let chunk_bytes = (bytes - copied).min(self.config.copy_chunk_bytes);
                let slot_index = (chunk_index as usize) % ASYNC_STAGE_SLOTS;
                let slot = &mut self.slots[slot_index];
                if let Some(event) = slot.event.take() {
                    event.synchronize()?;
                    self.stage_slot_reuse_waits += 1;
                }
                let read = self.direct_io.read_into(
                    self.model,
                    &mut slot.staging,
                    offset + copied,
                    chunk_bytes,
                )?;
                let copied_usize =
                    usize::try_from(copied).map_err(|_| ModelRangeError::ModelTooLarge)?;
                let chunk_bytes =
                    usize::try_from(chunk_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
                let destination_offset = device_offset
                    .checked_add(copied_usize)
                    .ok_or(ModelRangeError::ModelTooLarge)?;
                unsafe {
                    substrate.enqueue_pinned_u8_range_async(
                        device,
                        destination_offset,
                        &slot.staging,
                        read.payload_offset,
                        chunk_bytes,
                    )?;
                }
                slot.event = Some(substrate.record_event()?);
                self.events_recorded += 1;
                self.chunks_uploaded += 1;
                if read.direct_io {
                    self.direct_io_chunks += 1;
                } else {
                    self.buffered_chunks += 1;
                }
                let discard = self.model.discard_source_pages(
                    offset + copied,
                    chunk_bytes as u64,
                    self.config.keep_source_pages,
                )?;
                if discard.file_bytes != 0 {
                    self.source_file_discard_calls += 1;
                    self.source_file_discard_bytes += discard.file_bytes;
                }
                if discard.mapping_bytes != 0 {
                    self.source_mapping_discard_calls += 1;
                    self.source_mapping_discard_bytes += discard.mapping_bytes;
                }
                copied += chunk_bytes as u64;
                self.progress.note(self.range_bytes + copied);
                chunk_index += 1;
            }
            Ok::<(), ModelRangeError>(())
        })();
        // Drain already-enqueued copies even when a later read or event operation fails.
        let synchronize_result = substrate.synchronize().map_err(ModelRangeError::from);
        for slot in &mut self.slots {
            slot.event = None;
        }
        upload_result?;
        synchronize_result?;
        Ok(())
    }

    fn ensure_slots(
        &mut self,
        substrate: &CudaOxideSubstrate,
        stage_bytes: u64,
    ) -> Result<(), ModelRangeError> {
        let stage_bytes =
            usize::try_from(stage_bytes).map_err(|_| ModelRangeError::ModelTooLarge)?;
        if self
            .slots
            .first()
            .is_some_and(|slot| slot.staging.len() >= stage_bytes)
        {
            return Ok(());
        }
        substrate.synchronize()?;
        self.slots.clear();
        for _ in 0..ASYNC_STAGE_SLOTS {
            self.slots.push(AsyncPinnedStageSlot {
                staging: substrate.pinned_zeroed(stage_bytes)?,
                event: None,
            });
        }
        Ok(())
    }
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
                let (layout, source) = model.page_aligned_source(offset, bytes)?;
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
    round_up_to_alignment(value, page_size)
}

fn round_up_to_alignment(value: u64, alignment: u64) -> Result<u64, ModelRangeError> {
    value
        .checked_add(alignment - 1)
        .map(|end| (end / alignment) * alignment)
        .ok_or(ModelRangeError::ModelTooLarge)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::fs::File;

    use super::{direct_io_error_disables, DirectIoPolicyState, DirectIoReader};

    #[test]
    fn direct_io_disable_errors_match_current_c_policy() {
        for raw_os_error in [libc::EINVAL, libc::EFAULT, libc::ENOTSUP, libc::EOPNOTSUPP] {
            assert!(direct_io_error_disables(Some(raw_os_error)));
        }
        assert!(!direct_io_error_disables(Some(libc::EIO)));
        assert!(!direct_io_error_disables(None));
    }

    #[test]
    fn direct_io_selected_error_persistently_disables_future_direct_reads() {
        let mut reader = DirectIoReader {
            direct_file: Some(File::open("/dev/null").expect("open sentinel file")),
            alignment: 4096,
            disabled_raw_os_error: None,
        };
        reader.record_direct_read_error(&std::io::Error::from_raw_os_error(libc::EINVAL));
        assert_eq!(
            reader.state(),
            DirectIoPolicyState::DisabledAfterError {
                raw_os_error: libc::EINVAL
            }
        );
        assert!(reader.direct_file.is_none());
        assert_eq!(reader.stage_alignment(), 1);
    }
}
