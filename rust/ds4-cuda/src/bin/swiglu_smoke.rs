use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2B2_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn swiglu_kernel(
        count: u32,
        clamp: f32,
        weight: f32,
        gate: &[f32],
        up: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= count as usize {
            return;
        }

        let mut g = gate[i];
        let mut u = up[i];
        if clamp > 1.0e-6_f32 {
            if (g.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || g > clamp {
                g = clamp;
            }
            if (u.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || u < -clamp {
                u = -clamp;
            } else if u > clamp {
                u = clamp;
            }
        }
        let silu = g / (1.0_f32 + (-g).exp());
        if let Some(element) = out.get_mut(index) {
            *element = silu * u * weight;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;

    // cargo-oxide emits workspace-package artifacts at the workspace root.
    // Link PTX plus libdevice, then retain the generated typed launches.
    let raw_module = ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_swiglu_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let gate_values = [0.0_f32, 2.0, -3.0, 0.5, f32::NAN, -0.25];
    let up_values = [2.0_f32, 4.0, -4.0, -0.25, 0.5, f32::NAN];
    let gate = substrate.upload(&gate_values)?;
    let up = substrate.upload(&up_values)?;
    let mut out = substrate.zeroed::<f32>(gate_values.len())?;
    swiglu_tensor(
        &module,
        substrate.stream(),
        &mut out,
        &gate,
        &up,
        gate_values.len() as u32,
        1.5,
        0.75,
    )?;
    substrate.end_commands()?;
    let actual = substrate.download(&out)?;
    let expected = expected_swiglu(&gate_values, &up_values, 1.5, 0.75);
    assert_close(&actual, &expected, 1.0e-5);

    let mut unclamped = substrate.zeroed::<f32>(4)?;
    swiglu_tensor(
        &module,
        substrate.stream(),
        &mut unclamped,
        &gate,
        &up,
        4,
        0.0,
        0.75,
    )?;
    substrate.end_commands()?;
    let unclamped_actual = substrate.download(&unclamped)?;
    let unclamped_expected = expected_swiglu(&gate_values[..4], &up_values[..4], 0.0, 0.75);
    assert_close(&unclamped_actual, &unclamped_expected, 1.0e-5);

    let mut too_short = substrate.zeroed::<f32>(3)?;
    assert!(matches!(
        swiglu_tensor(
            &module,
            substrate.stream(),
            &mut too_short,
            &gate,
            &up,
            gate_values.len() as u32,
            1.5,
            0.75,
        ),
        Err(SwigluError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.2b2\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"swiglu_output_matches\":true,\"swiglu_unclamped_output_matches\":true,\"swiglu_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_swiglu_tensor\":{},\"owns_directional_steering_project_tensor\":{},\"owns_embedding_kernels\":{},\"owns_indexer_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2B2_SCOPE.owns_swiglu_tensor,
        M14_2B2_SCOPE.owns_directional_steering_project_tensor,
        M14_2B2_SCOPE.owns_embedding_kernels,
        M14_2B2_SCOPE.owns_indexer_kernels,
        M14_2B2_SCOPE.changes_default_route,
    );
    Ok(())
}

const THREADS_PER_BLOCK: u32 = 256;

fn swiglu_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    count: u32,
    clamp: f32,
    weight: f32,
) -> Result<(), SwigluError> {
    if count == 0
        || count as usize > out.len()
        || count as usize > gate.len()
        || count as usize > up.len()
    {
        return Err(SwigluError::InvalidShape);
    }
    module
        .swiglu_kernel(
            stream,
            LaunchConfig {
                grid_dim: (count.div_ceil(THREADS_PER_BLOCK), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            count,
            clamp,
            weight,
            gate,
            up,
            out,
        )
        .map_err(SwigluError::Driver)
}

fn expected_swiglu(gate: &[f32], up: &[f32], clamp: f32, weight: f32) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let g = if clamp > 1.0e-6_f32 && (g.is_nan() || g > clamp) {
                clamp
            } else {
                g
            };
            let u = if clamp > 1.0e-6_f32 {
                if u.is_nan() || u < -clamp {
                    -clamp
                } else if u > clamp {
                    clamp
                } else {
                    u
                }
            } else {
                u
            };
            (g / (1.0_f32 + (-g).exp())) * u * weight
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
enum SwigluError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for SwigluError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("SwiGLU tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SwigluError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape => None,
        }
    }
}
