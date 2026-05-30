#![feature(f16)]

use std::fmt;

use cuda_core::{
    BlasMathMode, CudaStream, DeviceBuffer, DriverError, LaunchConfig, ProjectionConfig,
};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use ds4_cuda::{
    select_f16_pair_projection_path, select_f16_projection_path, select_f32_projection_path,
    substrate::CudaOxideSubstrate, F16PairProjectionDispatch, F16PairProjectionPath,
    F16ProjectionDispatch, F16ProjectionPath, F32ProjectionPath, M14_3C3_SCOPE,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn f32_to_f16_kernel(count: u64, x: &[f32], mut out: DisjointSlice<f16>) {
        let index = thread::index_1d();
        let offset = index.get();
        if (offset as u64) < count {
            if let Some(element) = out.get_mut(index) {
                *element = x[offset] as f16;
            }
        }
    }
}

const IN_DIM: usize = 4;
const OUT_DIM: usize = 3;
const N_TOK: usize = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;
    let blas = substrate.blas_handle()?;
    blas.set_math_mode(BlasMathMode::Default)?;

    let weights_f16_values = [
        f16::from_bits(0x3c00),
        f16::from_bits(0x3800),
        f16::from_bits(0xbc00),
        f16::from_bits(0x4000),
        f16::from_bits(0x3400),
        f16::from_bits(0xc000),
        f16::from_bits(0x3e00),
        f16::from_bits(0xb800),
        f16::from_bits(0x4200),
        f16::from_bits(0x3c00),
        f16::from_bits(0x0000),
        f16::from_bits(0xbc00),
    ];
    let weights_f32_values = weights_f16_values.map(|value| value as f32);
    let x_values = [1.0_f32, -2.0, 0.5, 2.0, -0.5, 1.5, -1.0, 0.25];
    let expected = expected_projection(&weights_f32_values, &x_values);
    let config = ProjectionConfig::new(IN_DIM, OUT_DIM, N_TOK);

    let weights_f16 = substrate.upload(&weights_f16_values)?;
    let weights_f32 = substrate.upload(&weights_f32_values)?;
    let x_f32 = substrate.upload(&x_values)?;
    let mut x_f16 = substrate.zeroed::<f16>(x_values.len())?;
    convert_activations(
        &module,
        substrate.stream(),
        &x_f32,
        &mut x_f16,
        x_values.len() as u64,
    )?;
    substrate.flush_commands()?;
    assert_eq!(
        substrate
            .download(&x_f16)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
        x_values
    );

    assert_eq!(
        select_f16_projection_path(F16ProjectionDispatch {
            blas_ready: true,
            serial_f16: false,
            serial_router: false,
            no_ordered_f16_matmul: false,
            in_dim: IN_DIM as u64,
            out_dim: OUT_DIM as u64,
            n_tokens: N_TOK as u64,
        }),
        F16ProjectionPath::Blas
    );
    let mut f16_output = substrate.zeroed::<f32>(OUT_DIM * N_TOK)?;
    blas.project_f16_f32(
        substrate.stream(),
        config,
        &weights_f16,
        &x_f16,
        &mut f16_output,
    )?;
    substrate.flush_commands()?;
    assert_close(&substrate.download(&f16_output)?, &expected);

    assert_eq!(
        select_f32_projection_path(true, N_TOK as u64),
        F32ProjectionPath::Blas
    );
    let mut f32_output = substrate.zeroed::<f32>(OUT_DIM * N_TOK)?;
    blas.project_f32(
        substrate.stream(),
        config,
        &weights_f32,
        &x_f32,
        &mut f32_output,
    )?;
    substrate.end_commands()?;
    assert_close(&substrate.download(&f32_output)?, &expected);

    assert_dispatch_priority();
    let mut short_converted = substrate.zeroed::<f16>(x_values.len() - 1)?;
    assert!(matches!(
        convert_activations(
            &module,
            substrate.stream(),
            &x_f32,
            &mut short_converted,
            x_values.len() as u64,
        ),
        Err(BlasProjectionError::InvalidShape)
    ));
    let mut short_output = substrate.zeroed::<f32>(OUT_DIM * N_TOK - 1)?;
    assert!(blas
        .project_f32(
            substrate.stream(),
            config,
            &weights_f32,
            &x_f32,
            &mut short_output,
        )
        .is_err());

    println!(
        "{{\"milestone\":\"M14.3c3\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"activation_conversion_matches\":true,\"mixed_precision_blas_output_matches\":true,\"f32_blas_output_matches\":true,\"dispatch_priority_matches\":true,\"pair_dispatch_matches\":true,\"invalid_shape_rejected\":true,\"owns_f32_to_f16_kernel\":{},\"owns_f16_projection_dispatch_policy\":{},\"owns_f16_pair_projection_dispatch_policy\":{},\"owns_f32_projection_dispatch_policy\":{},\"owns_live_f16_and_f32_blas_paths\":{},\"owns_q8_conversion_or_matmul_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3C3_SCOPE.owns_f32_to_f16_kernel,
        M14_3C3_SCOPE.owns_f16_projection_dispatch_policy,
        M14_3C3_SCOPE.owns_f16_pair_projection_dispatch_policy,
        M14_3C3_SCOPE.owns_f32_projection_dispatch_policy,
        M14_3C3_SCOPE.owns_live_f16_and_f32_blas_paths,
        M14_3C3_SCOPE.owns_q8_conversion_or_matmul_kernels,
        M14_3C3_SCOPE.changes_default_route,
    );
    Ok(())
}

fn convert_activations(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f16>,
    count: u64,
) -> Result<(), BlasProjectionError> {
    if count == 0 || count > input.len() as u64 || count > output.len() as u64 {
        return Err(BlasProjectionError::InvalidShape);
    }
    let elements = u32::try_from(count).map_err(|_| BlasProjectionError::InvalidShape)?;
    module
        .f32_to_f16_kernel(
            stream,
            LaunchConfig::for_num_elems(elements),
            count,
            input,
            output,
        )
        .map_err(BlasProjectionError::Driver)
}

fn assert_dispatch_priority() {
    let base = F16ProjectionDispatch {
        blas_ready: false,
        serial_f16: false,
        serial_router: false,
        no_ordered_f16_matmul: false,
        in_dim: 4096,
        out_dim: 256,
        n_tokens: 1,
    };
    assert_eq!(
        select_f16_projection_path(F16ProjectionDispatch {
            serial_router: true,
            ..base
        }),
        F16ProjectionPath::Serial
    );
    assert_eq!(
        select_f16_projection_path(base),
        F16ProjectionPath::OrderedChunks
    );
    assert_eq!(
        select_f16_projection_path(F16ProjectionDispatch {
            no_ordered_f16_matmul: true,
            ..base
        }),
        F16ProjectionPath::Base
    );
    assert_eq!(
        select_f16_pair_projection_path(F16PairProjectionDispatch {
            n_tokens: 1,
            no_f16_pair_matmul: false,
            serial_f16: false,
            serial_router: false,
            no_ordered_f16_matmul: false,
        }),
        F16PairProjectionPath::PairedOrderedChunks
    );
    assert_eq!(
        select_f16_pair_projection_path(F16PairProjectionDispatch {
            n_tokens: 1,
            no_f16_pair_matmul: true,
            serial_f16: false,
            serial_router: false,
            no_ordered_f16_matmul: false,
        }),
        F16PairProjectionPath::TwoIndependent
    );
    assert_eq!(
        select_f32_projection_path(false, 2),
        F32ProjectionPath::Base
    );
}

fn expected_projection(weights: &[f32], x: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(OUT_DIM * N_TOK);
    for token in x.chunks_exact(IN_DIM) {
        for row in weights.chunks_exact(IN_DIM) {
            output.push(row.iter().zip(token).map(|(w, value)| w * value).sum());
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}

#[derive(Debug)]
enum BlasProjectionError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for BlasProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("BLAS projection tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for BlasProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
