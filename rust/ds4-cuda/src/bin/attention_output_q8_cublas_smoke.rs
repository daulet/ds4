#![feature(f16)]

use std::fmt;

use cuda_core::{
    Blas, BlasError, BlasMathMode, DeviceBuffer, DriverError, LaunchConfig,
    StridedBatchedSgemmConfig,
};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::ltoir;
use ds4_cuda::{
    select_attention_output_a_path, substrate::CudaOxideSubstrate, AttentionOutputADispatchOptions,
    AttentionOutputAPath, M14_4D8B_SCOPE,
};

const THREADS: u32 = 256;
const GROUP_DIM: u32 = 5;
const RANK: u32 = 3;
const N_GROUPS: u32 = 2;
const N_TOKENS: u32 = 3;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn attention_pack_group_heads_f16_kernel(
        n_tokens: u32,
        n_groups: u32,
        group_dim: u32,
        heads: &[f32],
        mut dst: DisjointSlice<f16>,
    ) {
        let index = thread::index_1d().get();
        let count = (n_groups * n_tokens * group_dim) as usize;
        if index >= count {
            return;
        }
        let dimension = index as u32 % group_dim;
        let quotient = index as u32 / group_dim;
        let token = quotient % n_tokens;
        let group = quotient / n_tokens;
        let source = ((token * n_groups + group) * group_dim + dimension) as usize;
        unsafe {
            *dst.get_unchecked_mut(index) = heads[source] as f16;
        }
    }

    #[kernel]
    pub fn f16_to_f32_kernel(input: &[f16], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        if index >= input.len() || index >= output.len() {
            return;
        }
        unsafe {
            *output.get_unchecked_mut(index) = input[index] as f32;
        }
    }

    #[kernel]
    pub fn attention_expand_group_weights_sgemm_kernel(
        n_groups: u32,
        rank: u32,
        group_dim: u32,
        weights: &[f16],
        mut transposed: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = (n_groups * rank * group_dim) as usize;
        if index >= count {
            return;
        }
        let dimension = index as u32 % group_dim;
        let quotient = index as u32 / group_dim;
        let output_row = quotient % rank;
        let group = quotient / rank;
        let destination = ((group * group_dim + dimension) * rank + output_row) as usize;
        unsafe {
            *transposed.get_unchecked_mut(destination) = weights[index] as f32;
        }
    }

    #[kernel]
    pub fn attention_unpack_group_low_kernel(
        n_tokens: u32,
        n_groups: u32,
        rank: u32,
        packed: &[f32],
        mut low: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let count = (n_groups * n_tokens * rank) as usize;
        if index >= count {
            return;
        }
        let output_rank = index as u32 % rank;
        let quotient = index as u32 / rank;
        let token = quotient % n_tokens;
        let group = quotient / n_tokens;
        let low_dim = n_groups * rank;
        let destination = (token * low_dim + group * rank + output_rank) as usize;
        unsafe {
            *low.get_unchecked_mut(destination) = packed[index];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_attention_output_q8_cublas_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let blas = substrate.blas_handle()?;
    blas.set_math_mode(BlasMathMode::Default)?;

    assert_dispatch_policy();

    let heads_values = values((N_TOKENS * N_GROUPS * GROUP_DIM) as usize, 17, -0.234375);
    let weight_values = values((N_GROUPS * RANK * GROUP_DIM) as usize, 11, -0.1875)
        .into_iter()
        .map(|value| value as f16)
        .collect::<Vec<_>>();
    let expected_packed = expected_packed_heads(&heads_values);
    let expected_low = expected_grouped_a(&heads_values, &weight_values);
    let heads = substrate.upload(&heads_values)?;
    let weights = substrate.upload(&weight_values)?;

    let (packed_heads, low) =
        attention_output_a_cublas_tensor(&substrate, &module, &blas, &heads, &weights)?;
    substrate.end_commands()?;
    let packed_actual = substrate
        .download(&packed_heads)?
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    assert_close(&packed_actual, &expected_packed);
    assert_close(&substrate.download(&low)?, &expected_low);

    let short_heads = substrate.zeroed::<f32>((N_TOKENS * N_GROUPS * GROUP_DIM - 1) as usize)?;
    assert!(matches!(
        attention_output_a_cublas_tensor(&substrate, &module, &blas, &short_heads, &weights,),
        Err(AttentionOutputCublasError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.4d8b\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"cublas_grouped_a_output_matches\":true,\"f16_packed_heads_match\":true,\"f16_expanded_weight_adapter_matches\":true,\"unpacked_low_layout_matches\":true,\"dispatch_priority_matches\":true,\"minimum_token_gate_matches\":true,\"fallback_without_expanded_weights_matches\":true,\"uses_live_cublas_sgemm_adapter\":true,\"uses_libdevice_link_path\":true,\"consumes_attention_output_q8_native_surface\":{},\"owns_attention_output_a_cublas_dispatch\":{},\"owns_attention_output_a_pack_unpack_kernels\":{},\"owns_live_cublas_grouped_a_pipeline\":{},\"uses_safe_sgemm_f16_rounded_adapter\":{},\"owns_runtime_graph_integration\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_4D8B_SCOPE.consumes_attention_output_q8_native_surface,
        M14_4D8B_SCOPE.owns_attention_output_a_cublas_dispatch,
        M14_4D8B_SCOPE.owns_attention_output_a_pack_unpack_kernels,
        M14_4D8B_SCOPE.owns_live_cublas_grouped_a_pipeline,
        M14_4D8B_SCOPE.uses_safe_sgemm_f16_rounded_adapter,
        M14_4D8B_SCOPE.owns_runtime_graph_integration,
        M14_4D8B_SCOPE.changes_default_route,
    );
    Ok(())
}

fn attention_output_a_cublas_tensor(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    blas: &Blas,
    heads: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f16>,
) -> Result<(DeviceBuffer<f16>, DeviceBuffer<f32>), AttentionOutputCublasError> {
    let packed_count = (N_GROUPS * N_TOKENS * GROUP_DIM) as usize;
    let weights_count = (N_GROUPS * RANK * GROUP_DIM) as usize;
    let low_count = (N_GROUPS * N_TOKENS * RANK) as usize;
    if heads.len() < packed_count || weights.len() < weights_count {
        return Err(AttentionOutputCublasError::InvalidShape);
    }

    let mut packed_heads_f16 = substrate.zeroed::<f16>(packed_count)?;
    module.attention_pack_group_heads_f16_kernel(
        substrate.stream(),
        flat_config(packed_count as u32),
        N_TOKENS,
        N_GROUPS,
        GROUP_DIM,
        heads,
        &mut packed_heads_f16,
    )?;
    let mut packed_heads_f32 = substrate.zeroed::<f32>(packed_count)?;
    module.f16_to_f32_kernel(
        substrate.stream(),
        flat_config(packed_count as u32),
        &packed_heads_f16,
        &mut packed_heads_f32,
    )?;
    let mut transposed_weights = substrate.zeroed::<f32>(weights_count)?;
    module.attention_expand_group_weights_sgemm_kernel(
        substrate.stream(),
        flat_config(weights_count as u32),
        N_GROUPS,
        RANK,
        GROUP_DIM,
        weights,
        &mut transposed_weights,
    )?;
    let mut packed_low = substrate.zeroed::<f32>(low_count)?;
    blas.sgemm_strided_batched(
        substrate.stream(),
        StridedBatchedSgemmConfig::packed(
            N_TOKENS as usize,
            RANK as usize,
            GROUP_DIM as usize,
            N_GROUPS as usize,
        )?,
        &packed_heads_f32,
        &transposed_weights,
        &mut packed_low,
    )?;
    let mut low = substrate.zeroed::<f32>(low_count)?;
    module.attention_unpack_group_low_kernel(
        substrate.stream(),
        flat_config(low_count as u32),
        N_TOKENS,
        N_GROUPS,
        RANK,
        &packed_low,
        &mut low,
    )?;
    Ok((packed_heads_f16, low))
}

fn assert_dispatch_policy() {
    let base = AttentionOutputADispatchOptions {
        quality_mode: false,
        cublas_ready: true,
        n_tokens: 3,
        cublas_min_tokens: 2,
        no_cublas_attention_output_a: false,
        expanded_f16_ready: true,
    };
    assert_eq!(
        select_attention_output_a_path(base),
        AttentionOutputAPath::CublasF16
    );
    assert_eq!(
        select_attention_output_a_path(AttentionOutputADispatchOptions {
            cublas_min_tokens: 4,
            ..base
        }),
        AttentionOutputAPath::NativeQ8
    );
    assert_eq!(
        select_attention_output_a_path(AttentionOutputADispatchOptions {
            expanded_f16_ready: false,
            ..base
        }),
        AttentionOutputAPath::NativeQ8
    );
}

fn flat_config(elements: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((elements + THREADS - 1) / THREADS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn expected_packed_heads(heads: &[f32]) -> Vec<f32> {
    let mut packed = Vec::with_capacity((N_GROUPS * N_TOKENS * GROUP_DIM) as usize);
    for group in 0..N_GROUPS as usize {
        for token in 0..N_TOKENS as usize {
            for dimension in 0..GROUP_DIM as usize {
                let source = (token * N_GROUPS as usize + group) * GROUP_DIM as usize + dimension;
                packed.push((heads[source] as f16) as f32);
            }
        }
    }
    packed
}

fn expected_grouped_a(heads: &[f32], weights: &[f16]) -> Vec<f32> {
    let mut low = vec![0.0_f32; (N_TOKENS * N_GROUPS * RANK) as usize];
    for token in 0..N_TOKENS as usize {
        for group in 0..N_GROUPS as usize {
            for output_rank in 0..RANK as usize {
                let mut sum = 0.0_f32;
                for dimension in 0..GROUP_DIM as usize {
                    let head_index =
                        (token * N_GROUPS as usize + group) * GROUP_DIM as usize + dimension;
                    let weight_index =
                        (group * RANK as usize + output_rank) * GROUP_DIM as usize + dimension;
                    sum += ((heads[head_index] as f16) as f32) * weights[weight_index] as f32;
                }
                low[(token * N_GROUPS as usize + group) * RANK as usize + output_rank] = sum;
            }
        }
    }
    low
}

fn values(count: usize, multiplier: u32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as u32 * multiplier + 5) % 41) as f32 * 0.015625 + offset)
        .collect()
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
enum AttentionOutputCublasError {
    InvalidShape,
    Driver(DriverError),
    Blas(BlasError),
}

impl From<DriverError> for AttentionOutputCublasError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl From<BlasError> for AttentionOutputCublasError {
    fn from(error: BlasError) -> Self {
        Self::Blas(error)
    }
}

impl fmt::Display for AttentionOutputCublasError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("attention output cuBLAS shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
            Self::Blas(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AttentionOutputCublasError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
            Self::Blas(error) => Some(error),
        }
    }
}
