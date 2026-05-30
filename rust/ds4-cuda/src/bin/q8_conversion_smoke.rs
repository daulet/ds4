#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_3D1_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn dequant_q8_0_to_f16_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
        weights: &[u8],
        mut output: DisjointSlice<f16>,
    ) {
        let index = thread::index_1d();
        let gid = index.get() as u64;
        let count = in_dim * out_dim;
        if gid >= count {
            return;
        }
        let row = gid / in_dim;
        let column = gid - row * in_dim;
        let block = column / 32;
        let lane = column - block * 32;
        let base = ((row * blocks + block) * 34) as usize;
        let scale_bits = weights[base] as u16 | ((weights[base + 1] as u16) << 8);
        let scale = f16::from_bits(scale_bits) as f32;
        let value = weights[base + 2 + lane as usize] as i8 as f32;
        if let Some(element) = output.get_mut(index) {
            *element = (scale * value) as f16;
        }
    }

    #[kernel]
    pub fn dequant_q8_0_to_f32_kernel(
        in_dim: u64,
        out_dim: u64,
        blocks: u64,
        weights: &[u8],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let gid = index.get() as u64;
        let count = in_dim * out_dim;
        if gid >= count {
            return;
        }
        let row = gid / in_dim;
        let column = gid - row * in_dim;
        let block = column / 32;
        let lane = column - block * 32;
        let base = ((row * blocks + block) * 34) as usize;
        let scale_bits = weights[base] as u16 | ((weights[base + 1] as u16) << 8);
        let scale = f16::from_bits(scale_bits) as f32;
        let value = weights[base + 2 + lane as usize] as i8 as f32;
        if let Some(element) = output.get_mut(index) {
            *element = scale * value;
        }
    }

    #[kernel]
    pub fn quantize_q8_0_f32_kernel(
        in_dim: u64,
        blocks: u64,
        x: &[f32],
        mut xq: DisjointSlice<i8>,
        mut xscale: DisjointSlice<f32>,
    ) {
        static mut VALUES: SharedArray<f32, 32> = SharedArray::UNINIT;

        let block = thread::blockIdx_x() as u64;
        let token = thread::blockIdx_y() as u64;
        let lane = thread::threadIdx_x() as usize;
        if block >= blocks {
            return;
        }
        let start = block * 32;
        let remaining = in_dim - start;
        let count = if remaining < 32 { remaining } else { 32 } as usize;
        let input_base = token as usize * in_dim as usize + start as usize;
        let value = if lane < count {
            x[input_base + lane]
        } else {
            0.0
        };
        let magnitude = if value < 0.0 { -value } else { value };
        unsafe {
            VALUES[lane] = magnitude;
        }
        thread::sync_threads();

        let mut stride = 16;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    if VALUES[lane + stride] > VALUES[lane] {
                        VALUES[lane] = VALUES[lane + stride];
                    }
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }

        let scale = unsafe { VALUES[0] } / 127.0;
        let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        let output_base = (token * blocks + block) as usize * 32;
        if lane == 0 {
            unsafe {
                *xscale.get_unchecked_mut((token * blocks + block) as usize) = scale;
            }
        }
        let quantized = if lane < count {
            let scaled = value * inverse;
            let lower = scaled.floor();
            let fraction = scaled - lower;
            let mut rounded = lower as i32;
            if fraction > 0.5 || (fraction == 0.5 && (rounded & 1) != 0) {
                rounded += 1;
            }
            if rounded > 127 {
                127
            } else if rounded < -128 {
                -128
            } else {
                rounded as i8
            }
        } else {
            0
        };
        unsafe {
            *xq.get_unchecked_mut(output_base + lane) = quantized;
        }
    }
}

const DEQUANT_THREADS_PER_BLOCK: u32 = 256;
const Q8_BLOCK_SIZE: u64 = 32;
const Q8_BLOCK_BYTES: u64 = 34;
const IN_DIM: u64 = 35;
const OUT_DIM: u64 = 2;
const N_TOK: u64 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;

    // Quantization uses floor for an explicit lrintf-compatible ties-even path.
    let raw_module =
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_q8_conversion_smoke")?;
    let module = kernels::from_module(raw_module)?;

    let packed_weights_values = packed_weights();
    let packed_weights = substrate.upload(&packed_weights_values)?;
    let mut f16_output = substrate.zeroed::<f16>((IN_DIM * OUT_DIM) as usize)?;
    let mut f32_output = substrate.zeroed::<f32>((IN_DIM * OUT_DIM) as usize)?;
    dequant_q8_f16_tensor(
        &module,
        substrate.stream(),
        &mut f16_output,
        &packed_weights,
        IN_DIM,
        OUT_DIM,
    )?;
    dequant_q8_f32_tensor(
        &module,
        substrate.stream(),
        &mut f32_output,
        &packed_weights,
        IN_DIM,
        OUT_DIM,
    )?;
    substrate.flush_commands()?;
    let expected_f32 = expected_dequantized_f32(&packed_weights_values, IN_DIM, OUT_DIM);
    let expected_f16 = expected_f32
        .iter()
        .map(|value| (*value as f16) as f32)
        .collect::<Vec<_>>();
    assert_eq!(
        substrate
            .download(&f16_output)?
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        expected_f16
    );
    assert_eq!(substrate.download(&f32_output)?, expected_f32);

    let x_values = activation_values();
    let x = substrate.upload(&x_values)?;
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE);
    let mut xq = substrate.zeroed::<i8>((N_TOK * blocks * Q8_BLOCK_SIZE) as usize)?;
    let mut xscale = substrate.zeroed::<f32>((N_TOK * blocks) as usize)?;
    quantize_q8_tensor(
        &module,
        substrate.stream(),
        &mut xq,
        &mut xscale,
        &x,
        IN_DIM,
        N_TOK,
    )?;
    substrate.end_commands()?;
    assert_eq!(substrate.download(&xq)?, expected_quantized(&x_values));
    assert_eq!(substrate.download(&xscale)?, vec![1.0; 4]);

    let mut short_dequant = substrate.zeroed::<f32>((IN_DIM * OUT_DIM - 1) as usize)?;
    assert!(matches!(
        dequant_q8_f32_tensor(
            &module,
            substrate.stream(),
            &mut short_dequant,
            &packed_weights,
            IN_DIM,
            OUT_DIM,
        ),
        Err(Q8ConversionError::InvalidShape)
    ));
    let mut short_scale = substrate.zeroed::<f32>((N_TOK * blocks - 1) as usize)?;
    assert!(matches!(
        quantize_q8_tensor(
            &module,
            substrate.stream(),
            &mut xq,
            &mut short_scale,
            &x,
            IN_DIM,
            N_TOK,
        ),
        Err(Q8ConversionError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.3d1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"packed_q8_f16_dequant_matches\":true,\"packed_q8_f32_dequant_matches\":true,\"activation_quantization_matches\":true,\"ties_to_even_matches_lrintf\":true,\"partial_block_padding_matches\":true,\"invalid_shape_rejected\":true,\"uses_libdevice_link_path\":true,\"owns_dequant_q8_0_to_f16_kernel\":{},\"owns_dequant_q8_0_to_f32_kernel\":{},\"owns_quantize_q8_0_f32_kernel\":{},\"owns_quantized_matmul_kernels\":{},\"owns_q8_matmul_dispatch_policy\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_3D1_SCOPE.owns_dequant_q8_0_to_f16_kernel,
        M14_3D1_SCOPE.owns_dequant_q8_0_to_f32_kernel,
        M14_3D1_SCOPE.owns_quantize_q8_0_f32_kernel,
        M14_3D1_SCOPE.owns_quantized_matmul_kernels,
        M14_3D1_SCOPE.owns_q8_matmul_dispatch_policy,
        M14_3D1_SCOPE.changes_default_route,
    );
    Ok(())
}

fn dequant_q8_f16_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    output: &mut DeviceBuffer<f16>,
    weights: &DeviceBuffer<u8>,
    in_dim: u64,
    out_dim: u64,
) -> Result<(), Q8ConversionError> {
    validate_dequant_shapes(output.len() as u64, weights, in_dim, out_dim)?;
    module
        .dequant_q8_0_to_f16_kernel(
            stream,
            dequant_launch_config(in_dim, out_dim)?,
            in_dim,
            out_dim,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights,
            output,
        )
        .map_err(Q8ConversionError::Driver)
}

fn dequant_q8_f32_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    output: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    in_dim: u64,
    out_dim: u64,
) -> Result<(), Q8ConversionError> {
    validate_dequant_shapes(output.len() as u64, weights, in_dim, out_dim)?;
    module
        .dequant_q8_0_to_f32_kernel(
            stream,
            dequant_launch_config(in_dim, out_dim)?,
            in_dim,
            out_dim,
            in_dim.div_ceil(Q8_BLOCK_SIZE),
            weights,
            output,
        )
        .map_err(Q8ConversionError::Driver)
}

fn validate_dequant_shapes(
    output_len: u64,
    weights: &DeviceBuffer<u8>,
    in_dim: u64,
    out_dim: u64,
) -> Result<(), Q8ConversionError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let count = in_dim
        .checked_mul(out_dim)
        .ok_or(Q8ConversionError::InvalidShape)?;
    let bytes = out_dim
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(Q8_BLOCK_BYTES))
        .ok_or(Q8ConversionError::InvalidShape)?;
    if in_dim == 0 || out_dim == 0 || count > output_len || bytes > weights.len() as u64 {
        return Err(Q8ConversionError::InvalidShape);
    }
    Ok(())
}

fn quantize_q8_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    xq: &mut DeviceBuffer<i8>,
    xscale: &mut DeviceBuffer<f32>,
    x: &DeviceBuffer<f32>,
    in_dim: u64,
    n_tok: u64,
) -> Result<(), Q8ConversionError> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let inputs = in_dim
        .checked_mul(n_tok)
        .ok_or(Q8ConversionError::InvalidShape)?;
    let quantized = n_tok
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(Q8_BLOCK_SIZE))
        .ok_or(Q8ConversionError::InvalidShape)?;
    let scales = n_tok
        .checked_mul(blocks)
        .ok_or(Q8ConversionError::InvalidShape)?;
    if in_dim == 0
        || n_tok == 0
        || inputs > x.len() as u64
        || quantized > xq.len() as u64
        || scales > xscale.len() as u64
    {
        return Err(Q8ConversionError::InvalidShape);
    }
    let grid_x = u32::try_from(blocks).map_err(|_| Q8ConversionError::InvalidShape)?;
    let grid_y = u32::try_from(n_tok).map_err(|_| Q8ConversionError::InvalidShape)?;
    module
        .quantize_q8_0_f32_kernel(
            stream,
            LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            in_dim,
            blocks,
            x,
            xq,
            xscale,
        )
        .map_err(Q8ConversionError::Driver)
}

fn dequant_launch_config(in_dim: u64, out_dim: u64) -> Result<LaunchConfig, Q8ConversionError> {
    let count = in_dim
        .checked_mul(out_dim)
        .ok_or(Q8ConversionError::InvalidShape)?;
    let count = u32::try_from(count).map_err(|_| Q8ConversionError::InvalidShape)?;
    Ok(LaunchConfig {
        grid_dim: (count.div_ceil(DEQUANT_THREADS_PER_BLOCK), 1, 1),
        block_dim: (DEQUANT_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn packed_weights() -> Vec<u8> {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE) as usize;
    let mut packed = Vec::with_capacity(OUT_DIM as usize * blocks * Q8_BLOCK_BYTES as usize);
    for row in 0..OUT_DIM as usize {
        for block in 0..blocks {
            let scale = match (row, block) {
                (0, 0) => f16::from_bits(0x3800),
                (0, 1) => f16::from_bits(0x3400),
                (1, 0) => f16::from_bits(0x3c00),
                _ => f16::from_bits(0x4000),
            };
            packed.extend_from_slice(&scale.to_bits().to_le_bytes());
            for lane in 0..32 {
                let value = match (row, block) {
                    (0, 0) => lane as i8 - 16,
                    (0, 1) => [2_i8, -3, 4].get(lane).copied().unwrap_or(0),
                    (1, 0) => 15 - lane as i8,
                    _ => [-4_i8, 5, -6].get(lane).copied().unwrap_or(0),
                };
                packed.push(value as u8);
            }
        }
    }
    packed
}

fn expected_dequantized_f32(packed: &[u8], in_dim: u64, out_dim: u64) -> Vec<f32> {
    let blocks = in_dim.div_ceil(Q8_BLOCK_SIZE);
    let mut output = Vec::with_capacity((in_dim * out_dim) as usize);
    for row in 0..out_dim {
        for column in 0..in_dim {
            let block = column / Q8_BLOCK_SIZE;
            let lane = column - block * Q8_BLOCK_SIZE;
            let base = ((row * blocks + block) * Q8_BLOCK_BYTES) as usize;
            let scale = f16::from_bits(u16::from_le_bytes([packed[base], packed[base + 1]]));
            let value = packed[base + 2 + lane as usize] as i8;
            output.push(scale as f32 * value as f32);
        }
    }
    output
}

fn activation_values() -> Vec<f32> {
    let mut values = vec![0.0_f32; (N_TOK * IN_DIM) as usize];
    values[..8].copy_from_slice(&[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0, -127.0]);
    values[32..35].copy_from_slice(&[63.5, -63.5, 127.0]);
    values[35..43].copy_from_slice(&[3.5, 4.5, -3.5, -4.5, 126.0, -127.0, 1.0, -1.0]);
    values[67..70].copy_from_slice(&[-31.5, 31.5, -127.0]);
    values
}

fn expected_quantized(input: &[f32]) -> Vec<i8> {
    let blocks = IN_DIM.div_ceil(Q8_BLOCK_SIZE) as usize;
    let mut output = vec![0_i8; N_TOK as usize * blocks * Q8_BLOCK_SIZE as usize];
    for (token, row) in input.chunks_exact(IN_DIM as usize).enumerate() {
        for (block, values) in row.chunks(32).enumerate() {
            let scale = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max) / 127.0;
            let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
            let base = (token * blocks + block) * 32;
            for (lane, value) in values.iter().enumerate() {
                output[base + lane] = clamp_i8(round_ties_even(*value * inverse));
            }
        }
    }
    output
}

fn round_ties_even(value: f32) -> i32 {
    let lower = value.floor();
    let fraction = value - lower;
    let mut rounded = lower as i32;
    if fraction > 0.5 || (fraction == 0.5 && (rounded & 1) != 0) {
        rounded += 1;
    }
    rounded
}

fn clamp_i8(value: i32) -> i8 {
    value.clamp(-128, 127) as i8
}

#[derive(Debug)]
enum Q8ConversionError {
    InvalidShape,
    Driver(DriverError),
}

impl fmt::Display for Q8ConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("Q8 conversion tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Q8ConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
