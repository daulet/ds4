use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2A_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add_kernel(count: u32, a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i < count as usize {
            if let Some(element) = out.get_mut(index) {
                *element = a[i] + b[i];
            }
        }
    }

    #[kernel]
    pub fn repeat_hc_kernel(count: u64, n_embd: u32, row: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if (i as u64) < count {
            if let Some(element) = out.get_mut(index) {
                *element = row[i % n_embd as usize];
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let a = substrate.upload(&[1.0_f32, -2.0, 3.5, 8.0])?;
    let b = substrate.upload(&[4.0_f32, 5.0, -1.5, -0.5])?;
    let mut sum = substrate.zeroed::<f32>(4)?;
    add_tensor(&module, substrate.stream(), &a, &b, &mut sum, 4)?;
    substrate.flush_commands()?;
    assert_eq!(substrate.download(&sum)?, [5.0, 3.0, 2.0, 7.5]);

    let row = substrate.upload(&[2.0_f32, -1.5, 4.0])?;
    let mut repeated = substrate.zeroed::<f32>(9)?;
    repeat_hc_tensor(&module, substrate.stream(), &row, &mut repeated, 3, 3)?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&repeated)?,
        [2.0, -1.5, 4.0, 2.0, -1.5, 4.0, 2.0, -1.5, 4.0]
    );

    let mut too_short = substrate.zeroed::<f32>(3)?;
    assert!(matches!(
        add_tensor(&module, substrate.stream(), &a, &b, &mut too_short, 4),
        Err(ElementwiseError::BufferTooSmall)
    ));
    assert!(matches!(
        repeat_hc_tensor(&module, substrate.stream(), &row, &mut repeated, 0, 3),
        Err(ElementwiseError::InvalidRepeatShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2a\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"add_output_matches\":true,\"repeat_hc_output_matches\":true,\"add_bounds_rejected\":true,\"repeat_shape_rejected\":true,\"owns_add_tensor\":{},\"owns_repeat_hc_tensor\":{},\"owns_embedding_kernels\":{},\"owns_indexer_kernels\":{},\"owns_swiglu_and_directional_steering\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2A_SCOPE.owns_add_tensor,
        M14_2A_SCOPE.owns_repeat_hc_tensor,
        M14_2A_SCOPE.owns_embedding_kernels,
        M14_2A_SCOPE.owns_indexer_kernels,
        M14_2A_SCOPE.owns_swiglu_and_directional_steering,
        M14_2A_SCOPE.changes_default_route,
    );
    Ok(())
}

const THREADS_PER_BLOCK: u32 = 256;

fn add_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    count: u32,
) -> Result<(), ElementwiseError> {
    if count == 0
        || count as usize > a.len()
        || count as usize > b.len()
        || count as usize > out.len()
    {
        return Err(ElementwiseError::BufferTooSmall);
    }
    module
        .add_kernel(stream, launch_config(u64::from(count))?, count, a, b, out)
        .map_err(ElementwiseError::Driver)
}

fn repeat_hc_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    row: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    n_embd: u32,
    n_hc: u32,
) -> Result<(), ElementwiseError> {
    let count = u64::from(n_embd) * u64::from(n_hc);
    if n_embd == 0 || n_hc == 0 || n_embd as usize > row.len() || count > out.len() as u64 {
        return Err(ElementwiseError::InvalidRepeatShape);
    }
    module
        .repeat_hc_kernel(stream, launch_config(count)?, count, n_embd, row, out)
        .map_err(ElementwiseError::Driver)
}

fn launch_config(count: u64) -> Result<LaunchConfig, ElementwiseError> {
    let blocks = count.div_ceil(u64::from(THREADS_PER_BLOCK));
    let grid_x =
        u32::try_from(blocks).map_err(|_| ElementwiseError::GridDimensionTooLarge { blocks })?;
    Ok(LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[derive(Debug)]
enum ElementwiseError {
    BufferTooSmall,
    InvalidRepeatShape,
    GridDimensionTooLarge { blocks: u64 },
    Driver(DriverError),
}

impl fmt::Display for ElementwiseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall => formatter.write_str("add tensor buffer is too small"),
            Self::InvalidRepeatShape => formatter.write_str("repeat_hc tensor shape is invalid"),
            Self::GridDimensionTooLarge { blocks } => {
                write!(
                    formatter,
                    "elementwise launch requires {blocks} CUDA blocks"
                )
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ElementwiseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::BufferTooSmall
            | Self::InvalidRepeatShape
            | Self::GridDimensionTooLarge { .. } => None,
        }
    }
}
