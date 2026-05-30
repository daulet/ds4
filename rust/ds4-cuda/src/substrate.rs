use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, ManagedBuffer};

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
}
