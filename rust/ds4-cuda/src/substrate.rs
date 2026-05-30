use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError, ManagedBuffer,
    MappedHostBuffer, MemoryAdvice, MemoryLocation, PinnedHostBuffer, ReadOnlyPageableHostMemory,
    ReadOnlyRegisteredHostMemory, RegisteredHostMemory, StreamAttachment,
};

use crate::allocation_policy::DeviceMemoryCapacity;

/// Rust-owned CUDA host resources used before DS4 kernels move off `ds4_cuda.cu`.
///
/// This type owns the CUDA context and stream through `cuda-core`. It does not
/// claim DS4 compute-kernel ownership or alter the runtime route.
#[derive(Debug)]
pub struct CudaOxideSubstrate {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaOxideSubstrate {
    pub fn open(device_ordinal: usize) -> Result<Self, DriverError> {
        let context = CudaContext::new(device_ordinal)?;
        let stream = context.new_stream()?;
        Ok(Self { context, stream })
    }

    pub fn device_name(&self) -> Result<String, DriverError> {
        self.context.device_name()
    }

    pub fn device_ordinal(&self) -> usize {
        self.context.ordinal()
    }

    pub fn memory_capacity(&self) -> Result<DeviceMemoryCapacity, DriverError> {
        let memory = self.context.memory_info()?;
        Ok(DeviceMemoryCapacity {
            free_bytes: memory.free_bytes as u64,
            total_bytes: memory.total_bytes as u64,
        })
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }

    pub fn upload<T: DeviceCopy>(&self, data: &[T]) -> Result<DeviceBuffer<T>, DriverError> {
        DeviceBuffer::from_host(&self.stream, data)
    }

    pub fn zeroed<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, DriverError> {
        DeviceBuffer::zeroed(&self.stream, len)
    }

    pub fn download<T: DeviceCopy>(&self, buffer: &DeviceBuffer<T>) -> Result<Vec<T>, DriverError> {
        buffer.to_host_vec(&self.stream)
    }

    pub fn managed_zeroed<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> Result<ManagedBuffer<T>, DriverError> {
        ManagedBuffer::zeroed(&self.context, len)
    }

    pub fn managed_from_slice<T: DeviceCopy>(
        &self,
        data: &[T],
    ) -> Result<ManagedBuffer<T>, DriverError> {
        ManagedBuffer::from_slice(&self.context, data)
    }

    pub fn prefetch_read_mostly_to_device<T: DeviceCopy>(
        &self,
        buffer: &ManagedBuffer<T>,
    ) -> Result<(), DriverError> {
        let device = MemoryLocation::Device(self.context.cu_device());
        buffer.advise(MemoryAdvice::SetReadMostly)?;
        buffer.advise(MemoryAdvice::SetPreferredLocation(device))?;
        buffer.prefetch_to(&self.stream, device)?;
        buffer.attach_to_stream(&self.stream, StreamAttachment::Single)
    }

    pub fn return_managed_to_host<T: DeviceCopy>(
        &self,
        buffer: &ManagedBuffer<T>,
    ) -> Result<(), DriverError> {
        buffer.prefetch_to(&self.stream, MemoryLocation::Host)?;
        buffer.attach_to_stream(&self.stream, StreamAttachment::Global)?;
        self.synchronize()
    }

    pub fn mapped_from_slice<T: DeviceCopy>(
        &self,
        data: &[T],
    ) -> Result<MappedHostBuffer<T>, DriverError> {
        MappedHostBuffer::from_slice(&self.context, data)
    }

    pub fn register_host_range<'a, T: DeviceCopy>(
        &self,
        data: &'a mut [T],
    ) -> Result<RegisteredHostMemory<'a, T>, DriverError> {
        RegisteredHostMemory::new(&self.context, data)
    }

    pub fn register_read_only_host_range<'a, T: DeviceCopy>(
        &self,
        data: &'a [T],
    ) -> Result<ReadOnlyRegisteredHostMemory<'a, T>, DriverError> {
        ReadOnlyRegisteredHostMemory::new(&self.context, data)
    }

    pub fn pageable_memory_access(&self) -> Result<bool, DriverError> {
        self.context.pageable_memory_access()
    }

    pub fn pageable_memory_access_uses_host_page_tables(&self) -> Result<bool, DriverError> {
        self.context.pageable_memory_access_uses_host_page_tables()
    }

    pub fn pageable_read_only_range<'a, T: DeviceCopy>(
        &self,
        data: &'a [T],
    ) -> Result<ReadOnlyPageableHostMemory<'a, T>, DriverError> {
        ReadOnlyPageableHostMemory::new(&self.context, data)
    }

    /// Applies the current-C pageable-HMM advice sequence and completes prefetch.
    ///
    /// Synchronizing here makes this opt-in proof API safe for a borrowed mmap
    /// range. Runtime queueing policy remains outside this milestone.
    pub fn prefetch_pageable_read_mostly_to_device<T: DeviceCopy>(
        &self,
        range: &ReadOnlyPageableHostMemory<'_, T>,
    ) -> Result<(), DriverError> {
        let device = MemoryLocation::Device(self.context.cu_device());
        range.advise(MemoryAdvice::SetReadMostly)?;
        range.advise(MemoryAdvice::SetPreferredLocation(device))?;
        unsafe {
            range.prefetch_to(&self.stream, device)?;
        }
        self.synchronize()
    }

    pub fn pinned_zeroed<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> Result<PinnedHostBuffer<T>, DriverError> {
        PinnedHostBuffer::zeroed(&self.context, len)
    }

    /// Enqueues a selected pinned staging range into an existing device buffer.
    ///
    /// # Safety
    ///
    /// `staging` must remain allocated and immutable until this substrate's
    /// stream reaches a completion point after this call.
    pub unsafe fn enqueue_pinned_u8_range_async(
        &self,
        device: &DeviceBuffer<u8>,
        device_offset: usize,
        staging: &PinnedHostBuffer<u8>,
        staging_offset: usize,
        bytes: usize,
    ) -> Result<(), DriverError> {
        assert!(device_offset <= device.len() && bytes <= device.len() - device_offset);
        assert!(staging_offset <= staging.len() && bytes <= staging.len() - staging_offset);
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                device.cu_deviceptr() + device_offset as u64,
                staging.as_ptr().add(staging_offset),
                bytes,
                self.stream.cu_stream(),
            )
        }
    }

    pub fn record_event(&self) -> Result<CudaEvent, DriverError> {
        self.stream.record_event(None)
    }

    /// Copies a selected range from a live pinned staging buffer and completes it.
    pub fn upload_pinned_u8_range(
        &self,
        staging: &PinnedHostBuffer<u8>,
        offset: usize,
        bytes: usize,
    ) -> Result<DeviceBuffer<u8>, DriverError> {
        let device = self.zeroed(bytes)?;
        unsafe {
            self.enqueue_pinned_u8_range_async(&device, 0, staging, offset, bytes)?;
        }
        self.synchronize()?;
        Ok(device)
    }

    /// Copies bytes from a device-readable pointer owned by a live residency guard.
    ///
    /// # Safety
    ///
    /// `ptr` must remain device-readable for `bytes` bytes until the stream
    /// synchronization performed by this call completes.
    pub unsafe fn download_u8_device_ptr(
        &self,
        ptr: cuda_core::sys::CUdeviceptr,
        bytes: usize,
    ) -> Result<Vec<u8>, DriverError> {
        let mut host = vec![0_u8; bytes];
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                ptr,
                bytes,
                self.stream.cu_stream(),
            )?;
        }
        self.synchronize()?;
        Ok(host)
    }
}
