use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B4_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fill_f32(count: u64, value: f32, mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        if (index.get() as u64) < count {
            if let Some(element) = output.get_mut(index) {
                *element = value;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let mut tensor = substrate.upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    fill_f32(&module, substrate.stream(), &mut tensor, -3.5, 4)?;
    substrate.flush_commands()?;
    assert_eq!(
        substrate.download(&tensor)?,
        [-3.5, -3.5, -3.5, -3.5, 5.0, 6.0]
    );

    fill_f32(
        &module,
        substrate.stream(),
        &mut tensor,
        f32::NEG_INFINITY,
        6,
    )?;
    substrate.end_commands()?;
    assert_eq!(substrate.download(&tensor)?, [f32::NEG_INFINITY; 6]);

    fill_f32(&module, substrate.stream(), &mut tensor, 7.0, 0)?;
    substrate.synchronize_device()?;
    assert_eq!(substrate.download(&tensor)?, [f32::NEG_INFINITY; 6]);

    assert!(matches!(
        fill_f32(&module, substrate.stream(), &mut tensor, 0.0, 7),
        Err(FillF32Error::CountExceedsTensor {
            count: 7,
            tensor_len: 6
        })
    ));

    println!(
        "{{\"milestone\":\"M14.1b4\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"prefix_fill_matches\":true,\"negative_infinity_fill_matches\":true,\"zero_count_is_noop\":true,\"bounds_rejected\":true,\"flush_is_context_wide\":true,\"end_is_context_wide\":true,\"synchronize_is_context_wide\":true,\"owns_tensor_fill_f32\":{},\"owns_command_synchronization\":{},\"owns_dequant_kernels\":{},\"owns_graph_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_1B4_SCOPE.owns_tensor_fill_f32,
        M14_1B4_SCOPE.owns_command_synchronization,
        M14_1B4_SCOPE.owns_dequant_kernels,
        M14_1B4_SCOPE.owns_graph_kernels,
        M14_1B4_SCOPE.changes_default_route,
    );
    Ok(())
}

const THREADS_PER_BLOCK: u32 = 256;

fn fill_f32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    tensor: &mut DeviceBuffer<f32>,
    value: f32,
    count: u64,
) -> Result<(), FillF32Error> {
    if count > tensor.len() as u64 {
        return Err(FillF32Error::CountExceedsTensor {
            count,
            tensor_len: tensor.len() as u64,
        });
    }
    if count == 0 {
        return Ok(());
    }

    let blocks = count.div_ceil(THREADS_PER_BLOCK as u64);
    let grid_x =
        u32::try_from(blocks).map_err(|_| FillF32Error::GridDimensionTooLarge { blocks })?;
    module
        .fill_f32(
            stream,
            LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            count,
            value,
            tensor,
        )
        .map_err(FillF32Error::Driver)
}

#[derive(Debug)]
enum FillF32Error {
    CountExceedsTensor { count: u64, tensor_len: u64 },
    GridDimensionTooLarge { blocks: u64 },
    Driver(DriverError),
}

impl fmt::Display for FillF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountExceedsTensor { count, tensor_len } => {
                write!(
                    formatter,
                    "fill count {count} exceeds tensor length {tensor_len}"
                )
            }
            Self::GridDimensionTooLarge { blocks } => {
                write!(formatter, "fill launch requires {blocks} CUDA blocks")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FillF32Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::CountExceedsTensor { .. } | Self::GridDimensionTooLarge { .. } => None,
        }
    }
}
