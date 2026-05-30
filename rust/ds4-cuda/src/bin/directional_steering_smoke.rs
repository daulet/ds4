use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2B1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn directional_steering_project_kernel(
        layer: u32,
        width: u32,
        rows: u32,
        scale: f32,
        directions: &[f32],
        mut x: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows || width == 0 {
            return;
        }

        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let width = width as usize;
        let x_base = row as usize * width;
        let direction_base = layer as usize * width;
        let x_ptr = x.as_mut_ptr();

        let mut sum = 0.0_f32;
        let mut i = tid;
        while i < width {
            unsafe {
                sum += *x_ptr.add(x_base + i) * directions[direction_base + i];
            }
            i += nth;
        }

        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();

        let mut stride = nth >> 1;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let coefficient = unsafe { scale * PARTIAL[0] };
        i = tid;
        while i < width {
            unsafe {
                *x.get_unchecked_mut(x_base + i) -= coefficient * directions[direction_base + i];
            }
            i += nth;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let direction_values = [1.0_f32, 0.0, 0.0, 0.0, 0.5, -0.5, 1.0, 0.25];
    let x_values = [1.0_f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
    let directions = substrate.upload(&direction_values)?;
    let mut x = substrate.upload(&x_values)?;
    directional_steering_project_tensor(
        &module,
        substrate.stream(),
        &mut x,
        &directions,
        1,
        4,
        2,
        0.25,
    )?;
    substrate.end_commands()?;
    let projected = substrate.download(&x)?;
    let projected_expected =
        expected_directional_projection(&x_values, &direction_values[4..], 4, 0.25);
    assert_close(&projected, &projected_expected, 1.0e-5);

    assert!(matches!(
        directional_steering_project_tensor(
            &module,
            substrate.stream(),
            &mut x,
            &directions,
            1,
            4,
            2,
            0.0,
        ),
        Err(DirectionalSteeringError::InvalidProjectionShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2b1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"directional_projection_matches\":true,\"directional_shape_rejected\":true,\"owns_directional_steering_project_tensor\":{},\"owns_swiglu_tensor\":{},\"owns_embedding_kernels\":{},\"owns_indexer_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2B1_SCOPE.owns_directional_steering_project_tensor,
        M14_2B1_SCOPE.owns_swiglu_tensor,
        M14_2B1_SCOPE.owns_embedding_kernels,
        M14_2B1_SCOPE.owns_indexer_kernels,
        M14_2B1_SCOPE.changes_default_route,
    );
    Ok(())
}

const THREADS_PER_BLOCK: u32 = 256;

fn directional_steering_project_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
    directions: &DeviceBuffer<f32>,
    layer: u32,
    width: u32,
    rows: u32,
    scale: f32,
) -> Result<(), DirectionalSteeringError> {
    let x_elements = u64::from(width) * u64::from(rows);
    let direction_elements = (u64::from(layer) + 1) * u64::from(width);
    if width == 0
        || rows == 0
        || scale == 0.0
        || x_elements > x.len() as u64
        || direction_elements > directions.len() as u64
    {
        return Err(DirectionalSteeringError::InvalidProjectionShape);
    }

    let mut threads = THREADS_PER_BLOCK;
    while threads > width && threads > 1 {
        threads >>= 1;
    }
    module
        .directional_steering_project_kernel(
            stream,
            LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            },
            layer,
            width,
            rows,
            scale,
            directions,
            x,
        )
        .map_err(DirectionalSteeringError::Driver)
}

fn expected_directional_projection(
    x: &[f32],
    direction: &[f32],
    width: usize,
    scale: f32,
) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|row| {
            let coefficient = scale * row.iter().zip(direction).map(|(x, d)| x * d).sum::<f32>();
            row.iter()
                .zip(direction)
                .map(move |(x, d)| x - coefficient * d)
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum DirectionalSteeringError {
    InvalidProjectionShape,
    Driver(DriverError),
}

impl fmt::Display for DirectionalSteeringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProjectionShape => {
                formatter.write_str("directional steering projection shape is invalid")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DirectionalSteeringError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidProjectionShape => None,
        }
    }
}
