use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32},
    cuda_module, kernel, thread, DisjointSlice,
};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5C2C1_SCOPE};

const N_MODEL_EXPERTS: usize = 256;
const THREADS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn moe_count_sorted_pairs_kernel(pair_count: u32, selected: &[i32], counts: &[u32]) {
        let pair = thread::index_1d().get();
        if pair >= pair_count as usize {
            return;
        }
        let mut expert = selected[pair];
        if expert < 0 {
            expert = 0;
        }
        let counter = unsafe { &*(counts.as_ptr().add(expert as usize) as *const DeviceAtomicU32) };
        counter.fetch_add(1, AtomicOrdering::Relaxed);
    }

    #[kernel]
    pub fn moe_build_expert_tile_offsets_kernel(
        block_m: u32,
        counts: &[u32],
        mut tile_offsets: DisjointSlice<u32>,
        mut tile_total: DisjointSlice<u32>,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0_u32;
        let mut expert = 0_usize;
        while expert < N_MODEL_EXPERTS {
            unsafe {
                *tile_offsets.get_unchecked_mut(expert) = sum;
            }
            sum += counts[expert].div_ceil(block_m);
            expert += 1;
        }
        unsafe {
            *tile_offsets.get_unchecked_mut(N_MODEL_EXPERTS) = sum;
            *tile_total.get_unchecked_mut(0) = sum;
        }
    }

    #[kernel]
    pub fn moe_build_expert_tiles_kernel(
        block_m: u32,
        counts: &[u32],
        tile_offsets: &[u32],
        mut tile_experts: DisjointSlice<u32>,
        mut tile_starts: DisjointSlice<u32>,
    ) {
        let expert = thread::threadIdx_x() as usize;
        if expert >= N_MODEL_EXPERTS {
            return;
        }
        let tile_count = counts[expert].div_ceil(block_m);
        let offset = tile_offsets[expert];
        let mut tile = 0_u32;
        while tile < tile_count {
            unsafe {
                *tile_experts.get_unchecked_mut((offset + tile) as usize) = expert as u32;
                *tile_starts.get_unchecked_mut((offset + tile) as usize) = tile * block_m;
            }
            tile += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_routed_moe_expert_tiles_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;
    let selected_values = selected_values();
    let selected = substrate.upload(&selected_values)?;
    let tile8 = build_expert_tiles(&substrate, &module, &selected, 8)?;
    let tile4 = build_expert_tiles(&substrate, &module, &selected, 4)?;
    substrate.end_commands()?;
    assert_tiles(&substrate, &tile8, &selected_values, 8)?;
    assert_tiles(&substrate, &tile4, &selected_values, 4)?;

    assert!(matches!(
        build_expert_tiles(&substrate, &module, &selected, 0),
        Err(TileError::InvalidBlockSize)
    ));

    println!(
        "{{\"milestone\":\"M14.5c2c1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"tile8_offsets_match\":true,\"tile8_descriptors_match\":true,\"tile4_offsets_match\":true,\"tile4_descriptors_match\":true,\"negative_expert_bucket_zero_matches\":true,\"partial_expert_tiles_match\":true,\"invalid_block_size_rejected\":true,\"uses_device_atomic_count_surface\":true,\"consumes_sorted_pair_metadata_surface\":{},\"owns_moe_build_expert_tile_offsets_kernel\":{},\"owns_moe_build_expert_tiles_kernel\":{},\"owns_tile4_and_tile8_descriptor_metadata\":{},\"owns_tile_projection_kernels\":{},\"owns_atomic_down_or_rowspan_dispatch\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5C2C1_SCOPE.consumes_sorted_pair_metadata_surface,
        M14_5C2C1_SCOPE.owns_moe_build_expert_tile_offsets_kernel,
        M14_5C2C1_SCOPE.owns_moe_build_expert_tiles_kernel,
        M14_5C2C1_SCOPE.owns_tile4_and_tile8_descriptor_metadata,
        M14_5C2C1_SCOPE.owns_tile_projection_kernels,
        M14_5C2C1_SCOPE.owns_atomic_down_or_rowspan_dispatch,
        M14_5C2C1_SCOPE.owns_q4_k_or_runtime_graph,
        M14_5C2C1_SCOPE.changes_default_route,
    );
    Ok(())
}

struct TileOutput {
    counts: DeviceBuffer<u32>,
    tile_offsets: DeviceBuffer<u32>,
    tile_total: DeviceBuffer<u32>,
    tile_experts: DeviceBuffer<u32>,
    tile_starts: DeviceBuffer<u32>,
}

fn build_expert_tiles(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    selected: &DeviceBuffer<i32>,
    block_m: u32,
) -> Result<TileOutput, TileError> {
    if block_m == 0 {
        return Err(TileError::InvalidBlockSize);
    }
    let counts = substrate.zeroed::<u32>(N_MODEL_EXPERTS)?;
    module.moe_count_sorted_pairs_kernel(
        substrate.stream(),
        one_dimensional_launch(selected.len()),
        selected.len() as u32,
        selected,
        &counts,
    )?;
    let mut tile_offsets = substrate.zeroed::<u32>(N_MODEL_EXPERTS + 1)?;
    let mut tile_total = substrate.zeroed::<u32>(1)?;
    module.moe_build_expert_tile_offsets_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
        block_m,
        &counts,
        &mut tile_offsets,
        &mut tile_total,
    )?;
    let capacity = selected.len() + N_MODEL_EXPERTS;
    let mut tile_experts = substrate.zeroed::<u32>(capacity)?;
    let mut tile_starts = substrate.zeroed::<u32>(capacity)?;
    module.moe_build_expert_tiles_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        block_m,
        &counts,
        &tile_offsets,
        &mut tile_experts,
        &mut tile_starts,
    )?;
    Ok(TileOutput {
        counts,
        tile_offsets,
        tile_total,
        tile_experts,
        tile_starts,
    })
}

fn assert_tiles(
    substrate: &CudaOxideSubstrate,
    actual: &TileOutput,
    selected: &[i32],
    block_m: u32,
) -> Result<(), DriverError> {
    let counts = substrate.download(&actual.counts)?;
    let tile_offsets = substrate.download(&actual.tile_offsets)?;
    let tile_total = substrate.download(&actual.tile_total)?;
    let tile_experts = substrate.download(&actual.tile_experts)?;
    let tile_starts = substrate.download(&actual.tile_starts)?;
    let mut expected_counts = vec![0_u32; N_MODEL_EXPERTS];
    for &expert in selected {
        expected_counts[normalized_expert(expert)] += 1;
    }
    assert_eq!(counts, expected_counts);
    let mut expected_offsets = vec![0_u32; N_MODEL_EXPERTS + 1];
    for expert in 0..N_MODEL_EXPERTS {
        expected_offsets[expert + 1] =
            expected_offsets[expert] + expected_counts[expert].div_ceil(block_m);
    }
    assert_eq!(tile_offsets, expected_offsets);
    assert_eq!(tile_total[0], expected_offsets[N_MODEL_EXPERTS]);
    for expert in 0..N_MODEL_EXPERTS {
        let begin = expected_offsets[expert];
        let end = expected_offsets[expert + 1];
        for tile in begin..end {
            assert_eq!(tile_experts[tile as usize], expert as u32);
            assert_eq!(tile_starts[tile as usize], (tile - begin) * block_m);
        }
    }
    assert!(expected_counts[0] > block_m);
    assert!(expected_counts[7] > block_m);
    assert!(!expected_counts[7].is_multiple_of(block_m));
    Ok(())
}

fn one_dimensional_launch(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((count as u32).div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn selected_values() -> Vec<i32> {
    let mut values = vec![-1; 3];
    values.extend(vec![0; 8]);
    values.extend(vec![1; 9]);
    values.extend(vec![2; 5]);
    values.extend(vec![7; 17]);
    values.push(255);
    values
}

fn normalized_expert(expert: i32) -> usize {
    if expert < 0 {
        0
    } else {
        expert as usize
    }
}

#[derive(Debug)]
enum TileError {
    InvalidBlockSize,
    Driver(DriverError),
}

impl From<DriverError> for TileError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl fmt::Display for TileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlockSize => {
                formatter.write_str("expert tile block size must be non-zero")
            }
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBlockSize => None,
            Self::Driver(error) => Some(error),
        }
    }
}
