use std::fmt;

use cuda_core::{DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32},
    cuda_module, kernel, thread, DisjointSlice,
};
use cuda_host::ltoir;
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_5C2B1_SCOPE};

const N_MODEL_EXPERTS: usize = 256;
const N_TOKENS: usize = 3;
const N_ROUTED: usize = 6;
const PAIR_COUNT: usize = N_TOKENS * N_ROUTED;
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
    pub fn moe_prefix_sorted_pairs_kernel(
        counts: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut cursors: DisjointSlice<u32>,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut sum = 0_u32;
        let mut expert = 0_usize;
        while expert < N_MODEL_EXPERTS {
            unsafe {
                *offsets.get_unchecked_mut(expert) = sum;
                *cursors.get_unchecked_mut(expert) = sum;
            }
            sum += counts[expert];
            expert += 1;
        }
        unsafe {
            *offsets.get_unchecked_mut(N_MODEL_EXPERTS) = sum;
        }
    }

    #[kernel]
    pub fn moe_scatter_sorted_pairs_kernel(
        pair_count: u32,
        selected: &[i32],
        cursors: &[u32],
        mut sorted_pairs: DisjointSlice<u32>,
    ) {
        let pair = thread::index_1d().get();
        if pair >= pair_count as usize {
            return;
        }
        let mut expert = selected[pair];
        if expert < 0 {
            expert = 0;
        }
        let cursor = unsafe { &*(cursors.as_ptr().add(expert as usize) as *const DeviceAtomicU32) };
        let position = cursor.fetch_add(1, AtomicOrdering::Relaxed);
        unsafe {
            *sorted_pairs.get_unchecked_mut(position as usize) = pair as u32;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let raw_module = ltoir::load_kernel_module(
        substrate.context(),
        "../../ds4_cuda_routed_moe_sorted_pairs_smoke",
    )?;
    let module = kernels::from_module(raw_module)?;

    let selected_values = vec![7, 2, -1, 2, 7, 4, 1, 4, 2, -1, 5, 1, 7, 0, 5, 3, 2, 3];
    let selected = substrate.upload(&selected_values)?;
    let output = build_sorted_pairs(&substrate, &module, &selected, PAIR_COUNT)?;
    substrate.end_commands()?;
    let counts = substrate.download(&output.counts)?;
    let offsets = substrate.download(&output.offsets)?;
    let cursors = substrate.download(&output.cursors)?;
    let sorted_pairs = substrate.download(&output.sorted_pairs)?;
    assert_sorted_metadata(&selected_values, &counts, &offsets, &cursors, &sorted_pairs);

    let short_selected = substrate.zeroed::<i32>(PAIR_COUNT - 1)?;
    assert!(matches!(
        build_sorted_pairs(&substrate, &module, &short_selected, PAIR_COUNT),
        Err(SortedPairsError::InvalidShape)
    ));

    println!(
        "{{\"milestone\":\"M14.5c2b1\",\"device_name\":{:?},\"rust_kernel_toolchain\":true,\"count_histogram_matches\":true,\"prefix_offsets_match\":true,\"scatter_grouping_matches\":true,\"duplicate_expert_pairs_preserved\":true,\"negative_expert_bucket_zero_matches\":true,\"atomic_cursor_completion_matches\":true,\"invalid_shape_rejected\":true,\"uses_device_atomic_fetch_add\":true,\"consumes_quantized_single_surface\":{},\"owns_moe_count_sorted_pairs_kernel\":{},\"owns_moe_prefix_sorted_pairs_kernel\":{},\"owns_moe_scatter_sorted_pairs_kernel\":{},\"owns_negative_expert_bucket_zero\":{},\"owns_sorted_pair_metadata\":{},\"owns_sorted_projection_kernels\":{},\"owns_expert_tile_or_atomic_down\":{},\"owns_q4_k_or_runtime_graph\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_5C2B1_SCOPE.consumes_quantized_single_surface,
        M14_5C2B1_SCOPE.owns_moe_count_sorted_pairs_kernel,
        M14_5C2B1_SCOPE.owns_moe_prefix_sorted_pairs_kernel,
        M14_5C2B1_SCOPE.owns_moe_scatter_sorted_pairs_kernel,
        M14_5C2B1_SCOPE.owns_negative_expert_bucket_zero,
        M14_5C2B1_SCOPE.owns_sorted_pair_metadata,
        M14_5C2B1_SCOPE.owns_sorted_projection_kernels,
        M14_5C2B1_SCOPE.owns_expert_tile_or_atomic_down,
        M14_5C2B1_SCOPE.owns_q4_k_or_runtime_graph,
        M14_5C2B1_SCOPE.changes_default_route,
    );
    Ok(())
}

struct SortedPairsOutput {
    counts: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    cursors: DeviceBuffer<u32>,
    sorted_pairs: DeviceBuffer<u32>,
}

fn build_sorted_pairs(
    substrate: &CudaOxideSubstrate,
    module: &kernels::LoadedModule,
    selected: &DeviceBuffer<i32>,
    pair_count: usize,
) -> Result<SortedPairsOutput, SortedPairsError> {
    if pair_count == 0 || selected.len() < pair_count {
        return Err(SortedPairsError::InvalidShape);
    }
    let counts = substrate.zeroed::<u32>(N_MODEL_EXPERTS)?;
    module.moe_count_sorted_pairs_kernel(
        substrate.stream(),
        one_dimensional_launch(pair_count),
        pair_count as u32,
        selected,
        &counts,
    )?;
    let mut offsets = substrate.zeroed::<u32>(N_MODEL_EXPERTS + 1)?;
    let mut cursors = substrate.zeroed::<u32>(N_MODEL_EXPERTS)?;
    module.moe_prefix_sorted_pairs_kernel(
        substrate.stream(),
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        },
        &counts,
        &mut offsets,
        &mut cursors,
    )?;
    let mut sorted_pairs = substrate.zeroed::<u32>(pair_count)?;
    module.moe_scatter_sorted_pairs_kernel(
        substrate.stream(),
        one_dimensional_launch(pair_count),
        pair_count as u32,
        selected,
        &cursors,
        &mut sorted_pairs,
    )?;
    Ok(SortedPairsOutput {
        counts,
        offsets,
        cursors,
        sorted_pairs,
    })
}

fn one_dimensional_launch(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((count as u32).div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn assert_sorted_metadata(
    selected: &[i32],
    counts: &[u32],
    offsets: &[u32],
    cursors: &[u32],
    sorted_pairs: &[u32],
) {
    let mut expected_counts = vec![0_u32; N_MODEL_EXPERTS];
    for &expert in selected {
        expected_counts[normalized_expert(expert)] += 1;
    }
    assert_eq!(counts, expected_counts);
    let mut expected_offsets = vec![0_u32; N_MODEL_EXPERTS + 1];
    for expert in 0..N_MODEL_EXPERTS {
        expected_offsets[expert + 1] = expected_offsets[expert] + expected_counts[expert];
        assert_eq!(
            cursors[expert],
            expected_offsets[expert] + expected_counts[expert]
        );
    }
    assert_eq!(offsets, expected_offsets);
    assert_eq!(offsets[N_MODEL_EXPERTS], PAIR_COUNT as u32);
    let mut seen = vec![false; PAIR_COUNT];
    for expert in 0..N_MODEL_EXPERTS {
        for position in offsets[expert] as usize..offsets[expert + 1] as usize {
            let pair = sorted_pairs[position] as usize;
            assert!(pair < PAIR_COUNT);
            assert_eq!(normalized_expert(selected[pair]), expert);
            assert!(!seen[pair]);
            seen[pair] = true;
        }
    }
    assert!(seen.into_iter().all(|entry| entry));
    assert!(counts[0] >= 2);
    assert!(counts[2] >= 3);
}

fn normalized_expert(expert: i32) -> usize {
    if expert < 0 {
        0
    } else {
        expert as usize
    }
}

#[derive(Debug)]
enum SortedPairsError {
    InvalidShape,
    Driver(DriverError),
}

impl From<DriverError> for SortedPairsError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl fmt::Display for SortedPairsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("sorted-pair tensor shape is invalid"),
            Self::Driver(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SortedPairsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidShape => None,
            Self::Driver(error) => Some(error),
        }
    }
}
