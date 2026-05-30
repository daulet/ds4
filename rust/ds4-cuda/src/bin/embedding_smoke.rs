#![feature(f16)]

use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_2C_SCOPE};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn embed_token_hc_kernel(
        token: u32,
        n_embd: u32,
        count: u64,
        weights: &[f16],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if (i as u64) >= count {
            return;
        }
        let embedding_index = i % n_embd as usize;
        if let Some(element) = out.get_mut(index) {
            *element = weights[token as usize * n_embd as usize + embedding_index] as f32;
        }
    }

    #[kernel]
    pub fn embed_tokens_hc_kernel(
        n_vocab: u32,
        n_embd: u32,
        n_hc: u32,
        count: u64,
        tokens: &[i32],
        weights: &[f16],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let gid = index.get();
        if (gid as u64) >= count {
            return;
        }
        let dimension = gid % n_embd as usize;
        let token_index = gid / n_embd as usize / n_hc as usize;
        let token = tokens[token_index];
        let token = if token < 0 || token as u32 >= n_vocab {
            0
        } else {
            token as usize
        };
        if let Some(element) = out.get_mut(index) {
            *element = weights[token * n_embd as usize + dimension] as f32;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let module = kernels::load(substrate.context())?;

    let weights_values = [
        f16::from_bits(0x3800), // 0.5
        f16::from_bits(0xbc00), // -1.0
        f16::from_bits(0x4000), // 2.0
        f16::from_bits(0x4200), // 3.0
        f16::from_bits(0x4400), // 4.0
        f16::from_bits(0xb400), // -0.25
        f16::from_bits(0xc000), // -2.0
        f16::from_bits(0x3a00), // 0.75
        f16::from_bits(0x3e00), // 1.5
    ];
    let weights = substrate.upload(&weights_values)?;

    let mut single_out = substrate.zeroed::<f32>(6)?;
    embed_token_hc_tensor(
        &module,
        substrate.stream(),
        &mut single_out,
        &weights,
        3,
        2,
        3,
        2,
    )?;
    substrate.flush_commands()?;
    assert_eq!(
        substrate.download(&single_out)?,
        [-2.0, 0.75, 1.5, -2.0, 0.75, 1.5]
    );

    let tokens = substrate.upload(&[-1_i32, 1, 99])?;
    let mut batch_out = substrate.zeroed::<f32>(18)?;
    embed_tokens_hc_tensor(
        &module,
        substrate.stream(),
        &mut batch_out,
        &tokens,
        &weights,
        3,
        3,
        3,
        2,
    )?;
    substrate.end_commands()?;
    assert_eq!(
        substrate.download(&batch_out)?,
        [
            0.5, -1.0, 2.0, 0.5, -1.0, 2.0, 3.0, 4.0, -0.25, 3.0, 4.0, -0.25, 0.5, -1.0, 2.0, 0.5,
            -1.0, 2.0,
        ]
    );

    let mut too_short = substrate.zeroed::<f32>(5)?;
    assert!(matches!(
        embed_token_hc_tensor(
            &module,
            substrate.stream(),
            &mut too_short,
            &weights,
            3,
            2,
            3,
            2,
        ),
        Err(EmbeddingError::InvalidShape)
    ));
    assert!(matches!(
        embed_token_hc_tensor(
            &module,
            substrate.stream(),
            &mut single_out,
            &weights,
            3,
            3,
            3,
            2,
        ),
        Err(EmbeddingError::InvalidToken)
    ));

    println!(
        "{{\"milestone\":\"M14.2c\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"embed_token_hc_output_matches\":true,\"embed_tokens_hc_output_matches\":true,\"batch_invalid_token_fallback_matches\":true,\"embedding_shape_rejected\":true,\"single_invalid_token_rejected\":true,\"owns_embed_token_hc_tensor\":{},\"owns_embed_tokens_hc_tensor\":{},\"owns_model_range_consumption\":{},\"owns_indexer_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_2C_SCOPE.owns_embed_token_hc_tensor,
        M14_2C_SCOPE.owns_embed_tokens_hc_tensor,
        M14_2C_SCOPE.owns_model_range_consumption,
        M14_2C_SCOPE.owns_indexer_kernels,
        M14_2C_SCOPE.changes_default_route,
    );
    Ok(())
}

const THREADS_PER_BLOCK: u32 = 256;

fn embed_token_hc_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<f16>,
    n_vocab: u32,
    token: u32,
    n_embd: u32,
    n_hc: u32,
) -> Result<(), EmbeddingError> {
    if token >= n_vocab {
        return Err(EmbeddingError::InvalidToken);
    }
    let weight_elements = u64::from(n_vocab) * u64::from(n_embd);
    let count = u64::from(n_embd) * u64::from(n_hc);
    if n_vocab == 0
        || n_embd == 0
        || n_hc == 0
        || weight_elements > weights.len() as u64
        || count > out.len() as u64
    {
        return Err(EmbeddingError::InvalidShape);
    }
    module
        .embed_token_hc_kernel(
            stream,
            launch_config(count)?,
            token,
            n_embd,
            count,
            weights,
            out,
        )
        .map_err(EmbeddingError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn embed_tokens_hc_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    out: &mut DeviceBuffer<f32>,
    tokens: &DeviceBuffer<i32>,
    weights: &DeviceBuffer<f16>,
    n_vocab: u32,
    n_tokens: u32,
    n_embd: u32,
    n_hc: u32,
) -> Result<(), EmbeddingError> {
    let weight_elements = u64::from(n_vocab) * u64::from(n_embd);
    let count = u64::from(n_tokens) * u64::from(n_hc) * u64::from(n_embd);
    if n_vocab == 0
        || n_tokens == 0
        || n_embd == 0
        || n_hc == 0
        || weight_elements > weights.len() as u64
        || n_tokens as usize > tokens.len()
        || count > out.len() as u64
    {
        return Err(EmbeddingError::InvalidShape);
    }
    module
        .embed_tokens_hc_kernel(
            stream,
            launch_config(count)?,
            n_vocab,
            n_embd,
            n_hc,
            count,
            tokens,
            weights,
            out,
        )
        .map_err(EmbeddingError::Driver)
}

fn launch_config(count: u64) -> Result<LaunchConfig, EmbeddingError> {
    let blocks = count.div_ceil(u64::from(THREADS_PER_BLOCK));
    let grid_x =
        u32::try_from(blocks).map_err(|_| EmbeddingError::GridDimensionTooLarge { blocks })?;
    Ok(LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[derive(Debug)]
enum EmbeddingError {
    InvalidShape,
    InvalidToken,
    GridDimensionTooLarge { blocks: u64 },
    Driver(DriverError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("embedding tensor shape is invalid"),
            Self::InvalidToken => formatter.write_str("single embedding token is out of bounds"),
            Self::GridDimensionTooLarge { blocks } => {
                write!(formatter, "embedding launch requires {blocks} CUDA blocks")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::InvalidShape | Self::InvalidToken | Self::GridDimensionTooLarge { .. } => None,
        }
    }
}
