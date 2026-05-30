use std::fmt;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5D_SCOPE};

const N_HC: u32 = 4;
const N_TOKENS: u32 = 2;
const N_EMBD: u32 = 9;
const MIX_HC: u32 = 2 * N_HC + N_HC * N_HC;
const THREADS: u32 = 256;
const SINKHORN_ITERS: u32 = 3;
const EPS: f32 = 1.0e-4;
const NORM_EPS: f32 = 1.0e-5;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn hc_split_sinkhorn_kernel(
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        mut split: DisjointSlice<f32>,
    ) {
        let row = thread::index_1d().get();
        if row < n_rows as usize {
            hc4_split_one(row, sinkhorn_iters, eps, mix, scale, base, &mut split);
        }
    }

    #[kernel]
    pub fn hc_weighted_sum_kernel(
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        weight_stride: u32,
        x: &[f32],
        weights: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index >= n_embd * n_tokens {
            return;
        }
        let dimension = index % n_embd;
        let token = index / n_embd;
        let mut accumulator = 0.0_f32;
        let mut hc = 0_u32;
        while hc < n_hc {
            accumulator += x[((token * n_hc + hc) * n_embd + dimension) as usize]
                * weights[(token * weight_stride + hc) as usize];
            hc += 1;
        }
        unsafe {
            *out.get_unchecked_mut(index as usize) = accumulator;
        }
    }

    #[kernel]
    pub fn hc_expand_kernel(
        n_embd: u32,
        n_hc: u32,
        n_tokens: u32,
        has_add: u32,
        block_out: &[f32],
        block_add: &[f32],
        residual_hc: &[f32],
        split: &[f32],
        mut out_hc: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index >= n_tokens * n_hc * n_embd {
            return;
        }
        let dimension = index % n_embd;
        let temporary = index / n_embd;
        let destination_hc = temporary % n_hc;
        let token = temporary / n_hc;
        let split_base = token * MIX_HC;
        let mut block_value = block_out[(token * n_embd + dimension) as usize];
        if has_add != 0 {
            block_value += block_add[(token * n_embd + dimension) as usize];
        }
        let mut accumulator = block_value * split[(split_base + n_hc + destination_hc) as usize];
        let mut source_hc = 0_u32;
        while source_hc < n_hc {
            accumulator += split
                [(split_base + 2 * n_hc + destination_hc + source_hc * n_hc) as usize]
                * residual_hc[((token * n_hc + source_hc) * n_embd + dimension) as usize];
            source_hc += 1;
        }
        unsafe {
            *out_hc.get_unchecked_mut(index as usize) = accumulator;
        }
    }

    #[kernel]
    pub fn hc_split_weighted_sum_fused_kernel(
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        mix: &[f32],
        residual_hc: &[f32],
        scale: &[f32],
        base: &[f32],
        mut split: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
    ) {
        let token = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if token >= n_rows || n_hc != N_HC {
            return;
        }
        if lane == 0 {
            hc4_split_one(
                token as usize,
                sinkhorn_iters,
                eps,
                mix,
                scale,
                base,
                &mut split,
            );
        }
        thread::sync_threads();
        let split_ptr = split.as_mut_ptr();
        let split_base = token as usize * MIX_HC as usize;
        let mut dimension = lane;
        while dimension < n_embd {
            let mut accumulator = 0.0_f32;
            let mut hc = 0_u32;
            while hc < N_HC {
                accumulator += residual_hc[((token * N_HC + hc) * n_embd + dimension) as usize]
                    * unsafe { *split_ptr.add(split_base + hc as usize) };
                hc += 1;
            }
            unsafe {
                *out.get_unchecked_mut((token * n_embd + dimension) as usize) = accumulator;
            }
            dimension += thread::blockDim_x();
        }
    }

    #[kernel]
    pub fn hc_split_weighted_sum_norm_fused_kernel(
        n_embd: u32,
        n_hc: u32,
        n_rows: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
        mix: &[f32],
        residual_hc: &[f32],
        scale: &[f32],
        base: &[f32],
        norm_weight: &[f32],
        mut split: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut norm_out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, { THREADS as usize }> = SharedArray::UNINIT;

        let token = thread::blockIdx_x();
        let lane = thread::threadIdx_x();
        if token >= n_rows || n_hc != N_HC {
            return;
        }
        if lane == 0 {
            hc4_split_one(
                token as usize,
                sinkhorn_iters,
                eps,
                mix,
                scale,
                base,
                &mut split,
            );
        }
        thread::sync_threads();
        let split_ptr = split.as_mut_ptr();
        let split_base = token as usize * MIX_HC as usize;
        let mut sum = 0.0_f32;
        let mut dimension = lane;
        while dimension < n_embd {
            let mut accumulator = 0.0_f32;
            let mut hc = 0_u32;
            while hc < N_HC {
                accumulator += residual_hc[((token * N_HC + hc) * n_embd + dimension) as usize]
                    * unsafe { *split_ptr.add(split_base + hc as usize) };
                hc += 1;
            }
            unsafe {
                *out.get_unchecked_mut((token * n_embd + dimension) as usize) = accumulator;
            }
            sum += accumulator * accumulator;
            dimension += thread::blockDim_x();
        }
        unsafe {
            PARTIAL[lane as usize] = sum;
        }
        thread::sync_threads();
        let mut stride = thread::blockDim_x() >> 1;
        while stride > 0 {
            if lane < stride {
                unsafe {
                    PARTIAL[lane as usize] += PARTIAL[(lane + stride) as usize];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        let norm_scale = 1.0 / (unsafe { PARTIAL[0] } / n_embd as f32 + norm_eps).sqrt();
        dimension = lane;
        while dimension < n_embd {
            let value = unsafe { *out.as_mut_ptr().add((token * n_embd + dimension) as usize) };
            unsafe {
                *norm_out.get_unchecked_mut((token * n_embd + dimension) as usize) =
                    value * norm_scale * norm_weight[dimension as usize];
            }
            dimension += thread::blockDim_x();
        }
    }

    #[kernel]
    pub fn output_hc_weights_kernel(
        n_hc: u32,
        n_tokens: u32,
        eps: f32,
        pre: &[f32],
        scale: &[f32],
        base: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index >= n_tokens * n_hc {
            return;
        }
        let hc = index % n_hc;
        let z = pre[index as usize] * scale[0] + base[hc as usize];
        unsafe {
            *out.get_unchecked_mut(index as usize) = 1.0 / (1.0 + (-z).exp()) + eps;
        }
    }

    fn hc4_split_one(
        row: usize,
        sinkhorn_iters: u32,
        eps: f32,
        mix: &[f32],
        scale: &[f32],
        base: &[f32],
        split: &mut DisjointSlice<f32>,
    ) {
        let input = row * MIX_HC as usize;
        let output = input;
        let mut hc = 0_usize;
        while hc < N_HC as usize {
            let pre = mix[input + hc] * scale[0] + base[hc];
            let post = mix[input + N_HC as usize + hc] * scale[1] + base[N_HC as usize + hc];
            unsafe {
                *split.get_unchecked_mut(output + hc) = 1.0 / (1.0 + (-pre).exp()) + eps;
                *split.get_unchecked_mut(output + N_HC as usize + hc) = 2.0 / (1.0 + (-post).exp());
            }
            hc += 1;
        }
        let mut combinations = [0.0_f32; 16];
        let mut source = 0_usize;
        while source < N_HC as usize {
            let first = mix[input + 2 * N_HC as usize + source * N_HC as usize] * scale[2]
                + base[2 * N_HC as usize + source * N_HC as usize];
            let mut maximum = first;
            let mut destination = 0_usize;
            while destination < N_HC as usize {
                let value = mix[input + 2 * N_HC as usize + source * N_HC as usize + destination]
                    * scale[2]
                    + base[2 * N_HC as usize + source * N_HC as usize + destination];
                combinations[source * N_HC as usize + destination] = value;
                if value > maximum {
                    maximum = value;
                }
                destination += 1;
            }
            let mut sum = 0.0_f32;
            destination = 0;
            while destination < N_HC as usize {
                let index = source * N_HC as usize + destination;
                let value = (combinations[index] - maximum).exp();
                combinations[index] = value;
                sum += value;
                destination += 1;
            }
            destination = 0;
            while destination < N_HC as usize {
                let index = source * N_HC as usize + destination;
                combinations[index] = combinations[index] / sum + eps;
                destination += 1;
            }
            source += 1;
        }
        let mut column = 0_usize;
        while column < N_HC as usize {
            let mut sum = eps;
            let mut row_index = 0_usize;
            while row_index < N_HC as usize {
                sum += combinations[row_index * N_HC as usize + column];
                row_index += 1;
            }
            row_index = 0;
            while row_index < N_HC as usize {
                let index = row_index * N_HC as usize + column;
                combinations[index] /= sum;
                row_index += 1;
            }
            column += 1;
        }
        let mut iteration = 1_u32;
        while iteration < sinkhorn_iters {
            source = 0;
            while source < N_HC as usize {
                let mut sum = eps;
                column = 0;
                while column < N_HC as usize {
                    sum += combinations[source * N_HC as usize + column];
                    column += 1;
                }
                column = 0;
                while column < N_HC as usize {
                    let index = source * N_HC as usize + column;
                    combinations[index] /= sum;
                    column += 1;
                }
                source += 1;
            }
            column = 0;
            while column < N_HC as usize {
                let mut sum = eps;
                let mut row_index = 0_usize;
                while row_index < N_HC as usize {
                    sum += combinations[row_index * N_HC as usize + column];
                    row_index += 1;
                }
                row_index = 0;
                while row_index < N_HC as usize {
                    let index = row_index * N_HC as usize + column;
                    combinations[index] /= sum;
                    row_index += 1;
                }
                column += 1;
            }
            iteration += 1;
        }
        let mut index = 0_usize;
        while index < 16 {
            unsafe {
                *split.get_unchecked_mut(output + 2 * N_HC as usize + index) = combinations[index];
            }
            index += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module =
        ltoir::load_kernel_module(substrate.context(), "../../ds4_cuda_hyperconnection_smoke")?;
    let module = kernels::from_module(raw_module)?;
    let mix_values = mix_values();
    let scale_values = [0.75_f32, -0.5, 0.625];
    let base_values = base_values();
    let residual_values = residual_values();
    let block_output_values = block_values(0.07);
    let add_values = block_values(-0.03);
    let direct_weight_values = direct_weights();
    let output_pre_values = output_pre_values();
    let output_scale_values = [0.9_f32];
    let output_base_values = [0.15_f32, -0.2, 0.05, 0.3];
    let norm_weight_values = norm_weights();

    let expected_split = expected_split(&mix_values, &scale_values, &base_values);
    let expected_direct =
        expected_weighted_sum(&residual_values, &direct_weight_values, N_HC as usize);
    let expected_split_sum =
        expected_weighted_sum(&residual_values, &expected_split, MIX_HC as usize);
    let expected_expanded = expected_expand(
        &block_output_values,
        &add_values,
        &residual_values,
        &expected_split,
        true,
    );
    let expected_plain_expand = expected_expand(
        &block_output_values,
        &add_values,
        &residual_values,
        &expected_split,
        false,
    );
    let expected_norm = expected_norm(&expected_split_sum, &norm_weight_values);
    let expected_output_weights = expected_output_weights(
        &output_pre_values,
        output_scale_values[0],
        &output_base_values,
    );

    let mix = substrate.upload(&mix_values)?;
    let scale = substrate.upload(&scale_values)?;
    let base = substrate.upload(&base_values)?;
    let residual = substrate.upload(&residual_values)?;
    let block = substrate.upload(&block_output_values)?;
    let add = substrate.upload(&add_values)?;
    let direct_weights = substrate.upload(&direct_weight_values)?;
    let output_pre = substrate.upload(&output_pre_values)?;
    let output_scale = substrate.upload(&output_scale_values)?;
    let output_base = substrate.upload(&output_base_values)?;
    let norm_weight = substrate.upload(&norm_weight_values)?;

    let mut split = substrate.zeroed::<f32>((N_TOKENS * MIX_HC) as usize)?;
    hc_split_sinkhorn_tensor(&module, substrate.stream(), &mix, &scale, &base, &mut split)?;
    let mut direct = substrate.zeroed::<f32>((N_TOKENS * N_EMBD) as usize)?;
    hc_weighted_sum_tensor(
        &module,
        substrate.stream(),
        &residual,
        &direct_weights,
        &mut direct,
        N_HC,
    )?;
    let mut split_sum = substrate.zeroed::<f32>((N_TOKENS * N_EMBD) as usize)?;
    hc_weighted_sum_tensor(
        &module,
        substrate.stream(),
        &residual,
        &split,
        &mut split_sum,
        MIX_HC,
    )?;
    let mut expand = substrate.zeroed::<f32>((N_TOKENS * N_HC * N_EMBD) as usize)?;
    hc_expand_tensor(
        &module,
        substrate.stream(),
        &block,
        &add,
        &residual,
        &split,
        &mut expand,
        true,
    )?;
    let mut plain_expand = substrate.zeroed::<f32>((N_TOKENS * N_HC * N_EMBD) as usize)?;
    hc_expand_tensor(
        &module,
        substrate.stream(),
        &block,
        &add,
        &residual,
        &split,
        &mut plain_expand,
        false,
    )?;
    let mut fused_split = substrate.zeroed::<f32>((N_TOKENS * MIX_HC) as usize)?;
    let mut fused_sum = substrate.zeroed::<f32>((N_TOKENS * N_EMBD) as usize)?;
    hc_split_weighted_sum_fused_tensor(
        &module,
        substrate.stream(),
        &mix,
        &residual,
        &scale,
        &base,
        &mut fused_split,
        &mut fused_sum,
    )?;
    let mut norm_split = substrate.zeroed::<f32>((N_TOKENS * MIX_HC) as usize)?;
    let mut norm_sum = substrate.zeroed::<f32>((N_TOKENS * N_EMBD) as usize)?;
    let mut norm_out = substrate.zeroed::<f32>((N_TOKENS * N_EMBD) as usize)?;
    hc_split_weighted_sum_norm_fused_tensor(
        &module,
        substrate.stream(),
        &mix,
        &residual,
        &scale,
        &base,
        &norm_weight,
        &mut norm_split,
        &mut norm_sum,
        &mut norm_out,
    )?;
    let mut output_weights = substrate.zeroed::<f32>((N_TOKENS * N_HC) as usize)?;
    output_hc_weights_tensor(
        &module,
        substrate.stream(),
        &output_pre,
        &output_scale,
        &output_base,
        &mut output_weights,
    )?;
    substrate.end_commands()?;

    assert_close(&substrate.download(&split)?, &expected_split, 2.0e-5);
    assert_close(&substrate.download(&direct)?, &expected_direct, 1.0e-5);
    assert_close(
        &substrate.download(&split_sum)?,
        &expected_split_sum,
        2.0e-5,
    );
    assert_close(&substrate.download(&expand)?, &expected_expanded, 2.0e-5);
    assert_close(
        &substrate.download(&plain_expand)?,
        &expected_plain_expand,
        2.0e-5,
    );
    assert_close(&substrate.download(&fused_split)?, &expected_split, 2.0e-5);
    assert_close(
        &substrate.download(&fused_sum)?,
        &expected_split_sum,
        2.0e-5,
    );
    assert_close(&substrate.download(&norm_split)?, &expected_split, 2.0e-5);
    assert_close(&substrate.download(&norm_sum)?, &expected_split_sum, 2.0e-5);
    assert_close(&substrate.download(&norm_out)?, &expected_norm, 3.0e-5);
    assert_close(
        &substrate.download(&output_weights)?,
        &expected_output_weights,
        1.0e-5,
    );

    let mut too_short = substrate.zeroed::<f32>((N_TOKENS * MIX_HC - 1) as usize)?;
    assert!(matches!(
        hc_split_sinkhorn_tensor(
            &module,
            substrate.stream(),
            &mix,
            &scale,
            &base,
            &mut too_short
        ),
        Err(HyperconnectionError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.5d\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"sinkhorn_split_matches\":true,\"direct_weighted_sum_matches\":true,\"split_weighted_sum_matches\":true,\"expand_add_matches\":true,\"expand_plain_matches\":true,\"fused_split_weighted_sum_matches\":true,\"fused_split_weighted_sum_norm_matches\":true,\"output_hc_weights_matches\":true,\"invalid_shape_rejected\":true,\"uses_thread_block_sync\":true,\"uses_libdevice_link_path\":true,\"owns_hc_split_sinkhorn_kernel\":{},\"owns_hc_weighted_sum_kernel\":{},\"owns_hc_expand_kernel\":{},\"owns_hc_split_weighted_sum_fused_kernel\":{},\"owns_hc_split_weighted_sum_norm_fused_kernel\":{},\"owns_output_hc_weights_kernel\":{},\"owns_shared_expert_wrapper_or_runtime_graph\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5D_SCOPE.owns_hc_split_sinkhorn_kernel,
        M14_5D_SCOPE.owns_hc_weighted_sum_kernel,
        M14_5D_SCOPE.owns_hc_expand_kernel,
        M14_5D_SCOPE.owns_hc_split_weighted_sum_fused_kernel,
        M14_5D_SCOPE.owns_hc_split_weighted_sum_norm_fused_kernel,
        M14_5D_SCOPE.owns_output_hc_weights_kernel,
        M14_5D_SCOPE.owns_shared_expert_wrapper_or_runtime_graph,
        M14_5D_SCOPE.changes_default_route,
    );
    Ok(())
}

fn launch_1d(count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_rows(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn hc_split_sinkhorn_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    mix: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    split: &mut DeviceBuffer<f32>,
) -> Result<(), HyperconnectionError> {
    if mix.len() < (N_TOKENS * MIX_HC) as usize
        || scale.len() < 3
        || base.len() < MIX_HC as usize
        || split.len() < (N_TOKENS * MIX_HC) as usize
    {
        return Err(HyperconnectionError::InvalidShape);
    }
    module
        .hc_split_sinkhorn_kernel(
            stream,
            launch_1d(N_TOKENS),
            N_TOKENS,
            SINKHORN_ITERS,
            EPS,
            mix,
            scale,
            base,
            split,
        )
        .map_err(HyperconnectionError::Driver)
}

fn hc_weighted_sum_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    residual: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    weight_stride: u32,
) -> Result<(), HyperconnectionError> {
    if residual.len() < (N_TOKENS * N_HC * N_EMBD) as usize
        || weights.len() < ((N_TOKENS - 1) * weight_stride + N_HC) as usize
        || out.len() < (N_TOKENS * N_EMBD) as usize
    {
        return Err(HyperconnectionError::InvalidShape);
    }
    module
        .hc_weighted_sum_kernel(
            stream,
            launch_1d(N_TOKENS * N_EMBD),
            N_EMBD,
            N_HC,
            N_TOKENS,
            weight_stride,
            residual,
            weights,
            out,
        )
        .map_err(HyperconnectionError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn hc_expand_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    block: &DeviceBuffer<f32>,
    add: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    split: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    has_add: bool,
) -> Result<(), HyperconnectionError> {
    if block.len() < (N_TOKENS * N_EMBD) as usize
        || add.len() < (N_TOKENS * N_EMBD) as usize
        || residual.len() < (N_TOKENS * N_HC * N_EMBD) as usize
        || split.len() < (N_TOKENS * MIX_HC) as usize
        || out.len() < (N_TOKENS * N_HC * N_EMBD) as usize
    {
        return Err(HyperconnectionError::InvalidShape);
    }
    module
        .hc_expand_kernel(
            stream,
            launch_1d(N_TOKENS * N_HC * N_EMBD),
            N_EMBD,
            N_HC,
            N_TOKENS,
            u32::from(has_add),
            block,
            add,
            residual,
            split,
            out,
        )
        .map_err(HyperconnectionError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn hc_split_weighted_sum_fused_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    mix: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    split: &mut DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), HyperconnectionError> {
    validate_fused_inputs(mix, residual, scale, base, split, out)?;
    module
        .hc_split_weighted_sum_fused_kernel(
            stream,
            launch_rows(N_TOKENS),
            N_EMBD,
            N_HC,
            N_TOKENS,
            SINKHORN_ITERS,
            EPS,
            mix,
            residual,
            scale,
            base,
            split,
            out,
        )
        .map_err(HyperconnectionError::Driver)
}

#[allow(clippy::too_many_arguments)]
fn hc_split_weighted_sum_norm_fused_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    mix: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    norm_weight: &DeviceBuffer<f32>,
    split: &mut DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    norm_out: &mut DeviceBuffer<f32>,
) -> Result<(), HyperconnectionError> {
    validate_fused_inputs(mix, residual, scale, base, split, out)?;
    if norm_weight.len() < N_EMBD as usize || norm_out.len() < (N_TOKENS * N_EMBD) as usize {
        return Err(HyperconnectionError::InvalidShape);
    }
    module
        .hc_split_weighted_sum_norm_fused_kernel(
            stream,
            launch_rows(N_TOKENS),
            N_EMBD,
            N_HC,
            N_TOKENS,
            SINKHORN_ITERS,
            EPS,
            NORM_EPS,
            mix,
            residual,
            scale,
            base,
            norm_weight,
            split,
            out,
            norm_out,
        )
        .map_err(HyperconnectionError::Driver)
}

fn validate_fused_inputs(
    mix: &DeviceBuffer<f32>,
    residual: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    split: &DeviceBuffer<f32>,
    out: &DeviceBuffer<f32>,
) -> Result<(), HyperconnectionError> {
    if mix.len() < (N_TOKENS * MIX_HC) as usize
        || residual.len() < (N_TOKENS * N_HC * N_EMBD) as usize
        || scale.len() < 3
        || base.len() < MIX_HC as usize
        || split.len() < (N_TOKENS * MIX_HC) as usize
        || out.len() < (N_TOKENS * N_EMBD) as usize
    {
        return Err(HyperconnectionError::InvalidShape);
    }
    Ok(())
}

fn output_hc_weights_tensor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    pre: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), HyperconnectionError> {
    if pre.len() < (N_TOKENS * N_HC) as usize
        || scale.is_empty()
        || base.len() < N_HC as usize
        || out.len() < (N_TOKENS * N_HC) as usize
    {
        return Err(HyperconnectionError::InvalidShape);
    }
    module
        .output_hc_weights_kernel(
            stream,
            launch_1d(N_TOKENS * N_HC),
            N_HC,
            N_TOKENS,
            EPS,
            pre,
            scale,
            base,
            out,
        )
        .map_err(HyperconnectionError::Driver)
}

fn mix_values() -> Vec<f32> {
    (0..(N_TOKENS * MIX_HC) as usize)
        .map(|index| ((index * 7 + 3) % 19) as f32 * 0.08 - 0.72)
        .collect()
}

fn base_values() -> Vec<f32> {
    (0..MIX_HC as usize)
        .map(|index| ((index * 5 + 1) % 13) as f32 * 0.04 - 0.24)
        .collect()
}

fn residual_values() -> Vec<f32> {
    (0..(N_TOKENS * N_HC * N_EMBD) as usize)
        .map(|index| ((index * 11 + 5) % 31) as f32 * 0.025 - 0.35)
        .collect()
}

fn block_values(offset: f32) -> Vec<f32> {
    (0..(N_TOKENS * N_EMBD) as usize)
        .map(|index| ((index * 3 + 2) % 17) as f32 * 0.06 - 0.3 + offset)
        .collect()
}

fn direct_weights() -> Vec<f32> {
    (0..(N_TOKENS * N_HC) as usize)
        .map(|index| ((index * 5 + 2) % 9) as f32 * 0.09 + 0.05)
        .collect()
}

fn output_pre_values() -> Vec<f32> {
    (0..(N_TOKENS * N_HC) as usize)
        .map(|index| ((index * 7 + 1) % 11) as f32 * 0.1 - 0.5)
        .collect()
}

fn norm_weights() -> Vec<f32> {
    (0..N_EMBD as usize)
        .map(|index| 0.5 + index as f32 * 0.08)
        .collect()
}

fn expected_split(mix: &[f32], scale: &[f32], base: &[f32]) -> Vec<f32> {
    let mut split = vec![0.0_f32; (N_TOKENS * MIX_HC) as usize];
    for token in 0..N_TOKENS as usize {
        let input = token * MIX_HC as usize;
        for hc in 0..N_HC as usize {
            split[input + hc] = sigmoid(mix[input + hc] * scale[0] + base[hc]) + EPS;
            split[input + N_HC as usize + hc] = 2.0
                * sigmoid(mix[input + N_HC as usize + hc] * scale[1] + base[N_HC as usize + hc]);
        }
        let mut combination = [0.0_f32; 16];
        for source in 0..N_HC as usize {
            let mut maximum = f32::NEG_INFINITY;
            for destination in 0..N_HC as usize {
                let index = source * N_HC as usize + destination;
                combination[index] = mix[input + 2 * N_HC as usize + index] * scale[2]
                    + base[2 * N_HC as usize + index];
                maximum = maximum.max(combination[index]);
            }
            let sum: f32 = (0..N_HC as usize)
                .map(|destination| {
                    let index = source * N_HC as usize + destination;
                    combination[index] = (combination[index] - maximum).exp();
                    combination[index]
                })
                .sum();
            for destination in 0..N_HC as usize {
                let index = source * N_HC as usize + destination;
                combination[index] = combination[index] / sum + EPS;
            }
        }
        normalize_columns(&mut combination);
        for _ in 1..SINKHORN_ITERS {
            normalize_rows(&mut combination);
            normalize_columns(&mut combination);
        }
        split[input + 2 * N_HC as usize..input + MIX_HC as usize].copy_from_slice(&combination);
    }
    split
}

fn normalize_rows(values: &mut [f32; 16]) {
    for row in 0..N_HC as usize {
        let sum = EPS
            + (0..N_HC as usize)
                .map(|column| values[row * N_HC as usize + column])
                .sum::<f32>();
        for column in 0..N_HC as usize {
            values[row * N_HC as usize + column] /= sum;
        }
    }
}

fn normalize_columns(values: &mut [f32; 16]) {
    for column in 0..N_HC as usize {
        let sum = EPS
            + (0..N_HC as usize)
                .map(|row| values[row * N_HC as usize + column])
                .sum::<f32>();
        for row in 0..N_HC as usize {
            values[row * N_HC as usize + column] /= sum;
        }
    }
}

fn expected_weighted_sum(residual: &[f32], weights: &[f32], stride: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; (N_TOKENS * N_EMBD) as usize];
    for token in 0..N_TOKENS as usize {
        for dimension in 0..N_EMBD as usize {
            for hc in 0..N_HC as usize {
                out[token * N_EMBD as usize + dimension] += residual
                    [(token * N_HC as usize + hc) * N_EMBD as usize + dimension]
                    * weights[token * stride + hc];
            }
        }
    }
    out
}

fn expected_expand(
    block: &[f32],
    add: &[f32],
    residual: &[f32],
    split: &[f32],
    has_add: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; (N_TOKENS * N_HC * N_EMBD) as usize];
    for token in 0..N_TOKENS as usize {
        let split_base = token * MIX_HC as usize;
        for destination in 0..N_HC as usize {
            for dimension in 0..N_EMBD as usize {
                let row = token * N_EMBD as usize + dimension;
                let mut accumulator = (block[row] + if has_add { add[row] } else { 0.0 })
                    * split[split_base + N_HC as usize + destination];
                for source in 0..N_HC as usize {
                    accumulator += split
                        [split_base + 2 * N_HC as usize + destination + source * N_HC as usize]
                        * residual[(token * N_HC as usize + source) * N_EMBD as usize + dimension];
                }
                out[(token * N_HC as usize + destination) * N_EMBD as usize + dimension] =
                    accumulator;
            }
        }
    }
    out
}

fn expected_norm(values: &[f32], weights: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; values.len()];
    for token in 0..N_TOKENS as usize {
        let row = &values[token * N_EMBD as usize..(token + 1) * N_EMBD as usize];
        let scale = 1.0
            / (row.iter().map(|value| value * value).sum::<f32>() / N_EMBD as f32 + NORM_EPS)
                .sqrt();
        for dimension in 0..N_EMBD as usize {
            out[token * N_EMBD as usize + dimension] = row[dimension] * scale * weights[dimension];
        }
    }
    out
}

fn expected_output_weights(pre: &[f32], scale: f32, base: &[f32]) -> Vec<f32> {
    pre.iter()
        .enumerate()
        .map(|(index, value)| sigmoid(*value * scale + base[index % N_HC as usize]) + EPS)
        .collect()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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
enum HyperconnectionError {
    InvalidShape,
    Driver(DriverError),
}

impl From<DriverError> for HyperconnectionError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl fmt::Display for HyperconnectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("hyperconnection tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for HyperconnectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
