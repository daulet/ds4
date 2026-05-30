use ds4_cuda::allocation_policy::{
    format_cuda_memory_report, managed_kv_decision, DeviceMemoryCapacity, ManagedKvReason,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B3A_SCOPE};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let live_memory = substrate.memory_capacity()?;
    assert!(live_memory.total_bytes > 0);
    assert!(live_memory.free_bytes <= live_memory.total_bytes);

    let managed = substrate.managed_zeroed::<u8>(4096)?;
    assert_eq!(managed.len(), 4096);
    assert_eq!(managed.as_slice(), &[0_u8; 4096]);

    let empty = managed_kv_decision(0, 16 * GIB, None);
    assert!(!empty.use_managed);
    assert_eq!(empty.reason, ManagedKvReason::EmptyKv);

    let huge = managed_kv_decision(8 * GIB, 0, None);
    assert!(huge.use_managed);
    assert_eq!(huge.reason, ManagedKvReason::HugeKv);

    let small_context = managed_kv_decision(1, 4 * GIB, None);
    assert!(!small_context.use_managed);
    assert_eq!(small_context.reason, ManagedKvReason::SmallContext);

    let query_failure = managed_kv_decision(1, 9 * GIB, None);
    assert!(!query_failure.use_managed);
    assert_eq!(
        query_failure.reason,
        ManagedKvReason::MemoryQueryUnavailable
    );

    let fits = managed_kv_decision(
        1,
        9 * GIB,
        Some(DeviceMemoryCapacity {
            free_bytes: 24 * GIB,
            total_bytes: 32 * GIB,
        }),
    );
    assert!(!fits.use_managed);
    assert_eq!(fits.reason, ManagedKvReason::FitsDeviceMemory);

    let reserve_pressure = managed_kv_decision(
        1,
        17 * GIB,
        Some(DeviceMemoryCapacity {
            free_bytes: 24 * GIB,
            total_bytes: 32 * GIB,
        }),
    );
    assert!(reserve_pressure.use_managed);
    assert_eq!(
        reserve_pressure.reason,
        ManagedKvReason::ContextConsumesReserve
    );

    let exceeds_free = managed_kv_decision(
        1,
        9 * GIB,
        Some(DeviceMemoryCapacity {
            free_bytes: 8 * GIB,
            total_bytes: 32 * GIB,
        }),
    );
    assert!(exceeds_free.use_managed);
    assert_eq!(
        exceeds_free.reason,
        ManagedKvReason::ContextExceedsFreeMemory
    );

    let synthetic_report = format_cuda_memory_report(
        "after graph alloc",
        DeviceMemoryCapacity {
            free_bytes: 24 * GIB,
            total_bytes: 32 * GIB,
        },
    );
    assert_eq!(
        synthetic_report,
        "ds4: CUDA memory report after graph alloc: free 24576.00 MiB total 32768.00 MiB"
    );
    eprintln!("{}", format_cuda_memory_report("b3a live", live_memory));

    println!(
        "{{\"milestone\":\"M14.1b3a\",\"device_name\":{:?},\"live_memory_info_valid\":true,\"managed_allocation\":true,\"zero_kv_uses_device\":true,\"huge_kv_uses_managed\":true,\"small_context_uses_device\":true,\"memory_query_failure_uses_device\":true,\"sufficient_capacity_uses_device\":true,\"reserve_pressure_uses_managed\":true,\"context_exceeds_free_uses_managed\":true,\"memory_report_shape_matches\":true,\"owns_managed_tensor_allocation\":{},\"owns_managed_kv_selection\":{},\"owns_memory_report\":{},\"owns_q8_cache_policy\":{},\"owns_quality_mode\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_1B3A_SCOPE.owns_managed_tensor_allocation,
        M14_1B3A_SCOPE.owns_managed_kv_selection,
        M14_1B3A_SCOPE.owns_memory_report,
        M14_1B3A_SCOPE.owns_q8_cache_policy,
        M14_1B3A_SCOPE.owns_quality_mode,
        M14_1B3A_SCOPE.owns_ds4_kernels,
        M14_1B3A_SCOPE.changes_default_route,
    );
    Ok(())
}
