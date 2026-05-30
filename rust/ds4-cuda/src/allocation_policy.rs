const GIB: u64 = 1024 * 1024 * 1024;
const MIB: f64 = 1024.0 * 1024.0;
const MANAGED_KV_THRESHOLD_BYTES: u64 = 8 * GIB;
const MANAGED_CONTEXT_THRESHOLD_BYTES: u64 = 8 * GIB;
const MANAGED_MIN_RESERVE_BYTES: u64 = 8 * GIB;
const MANAGED_MAX_RESERVE_BYTES: u64 = 40 * GIB;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceMemoryCapacity {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedKvReason {
    EmptyKv,
    HugeKv,
    SmallContext,
    MemoryQueryUnavailable,
    ContextExceedsFreeMemory,
    ContextConsumesReserve,
    FitsDeviceMemory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagedKvDecision {
    pub use_managed: bool,
    pub reason: ManagedKvReason,
    pub reserve_bytes: u64,
}

pub fn managed_kv_reserve_bytes(total_bytes: u64) -> u64 {
    (total_bytes / 4).clamp(MANAGED_MIN_RESERVE_BYTES, MANAGED_MAX_RESERVE_BYTES)
}

pub fn managed_kv_decision(
    kv_cache_bytes: u64,
    context_bytes: u64,
    memory: Option<DeviceMemoryCapacity>,
) -> ManagedKvDecision {
    if kv_cache_bytes == 0 {
        return decision(false, ManagedKvReason::EmptyKv, 0);
    }
    if kv_cache_bytes >= MANAGED_KV_THRESHOLD_BYTES {
        return decision(true, ManagedKvReason::HugeKv, 0);
    }
    if context_bytes < MANAGED_CONTEXT_THRESHOLD_BYTES {
        return decision(false, ManagedKvReason::SmallContext, 0);
    }
    let Some(memory) = memory else {
        return decision(false, ManagedKvReason::MemoryQueryUnavailable, 0);
    };
    let reserve_bytes = managed_kv_reserve_bytes(memory.total_bytes);
    if context_bytes > memory.free_bytes {
        return decision(
            true,
            ManagedKvReason::ContextExceedsFreeMemory,
            reserve_bytes,
        );
    }
    if memory.free_bytes - context_bytes < reserve_bytes {
        return decision(true, ManagedKvReason::ContextConsumesReserve, reserve_bytes);
    }
    decision(false, ManagedKvReason::FitsDeviceMemory, reserve_bytes)
}

pub fn format_cuda_memory_report(label: &str, memory: DeviceMemoryCapacity) -> String {
    format!(
        "ds4: CUDA memory report {label}: free {:.2} MiB total {:.2} MiB",
        memory.free_bytes as f64 / MIB,
        memory.total_bytes as f64 / MIB
    )
}

const fn decision(
    use_managed: bool,
    reason: ManagedKvReason,
    reserve_bytes: u64,
) -> ManagedKvDecision {
    ManagedKvDecision {
        use_managed,
        reason,
        reserve_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        format_cuda_memory_report, managed_kv_decision, managed_kv_reserve_bytes,
        DeviceMemoryCapacity, ManagedKvReason, GIB,
    };

    #[test]
    fn managed_kv_policy_matches_current_c_thresholds_and_reserve() {
        assert_eq!(managed_kv_reserve_bytes(16 * GIB), 8 * GIB);
        assert_eq!(managed_kv_reserve_bytes(128 * GIB), 32 * GIB);
        assert_eq!(managed_kv_reserve_bytes(256 * GIB), 40 * GIB);
        assert_eq!(
            managed_kv_decision(0, 32 * GIB, None).reason,
            ManagedKvReason::EmptyKv
        );
        assert!(managed_kv_decision(8 * GIB, 0, None).use_managed);
        assert_eq!(
            managed_kv_decision(1, 4 * GIB, None).reason,
            ManagedKvReason::SmallContext
        );
        assert_eq!(
            managed_kv_decision(1, 9 * GIB, None).reason,
            ManagedKvReason::MemoryQueryUnavailable
        );
    }

    #[test]
    fn managed_kv_policy_uses_capacity_only_for_large_contexts() {
        let capacity = DeviceMemoryCapacity {
            free_bytes: 24 * GIB,
            total_bytes: 32 * GIB,
        };
        let fits = managed_kv_decision(1, 9 * GIB, Some(capacity));
        assert!(!fits.use_managed);
        assert_eq!(fits.reason, ManagedKvReason::FitsDeviceMemory);
        assert_eq!(fits.reserve_bytes, 8 * GIB);

        let pressure = managed_kv_decision(
            1,
            17 * GIB,
            Some(DeviceMemoryCapacity {
                free_bytes: 24 * GIB,
                total_bytes: 32 * GIB,
            }),
        );
        assert!(pressure.use_managed);
        assert_eq!(pressure.reason, ManagedKvReason::ContextConsumesReserve);

        let exceeds = managed_kv_decision(
            1,
            9 * GIB,
            Some(DeviceMemoryCapacity {
                free_bytes: 8 * GIB,
                total_bytes: 32 * GIB,
            }),
        );
        assert!(exceeds.use_managed);
        assert_eq!(exceeds.reason, ManagedKvReason::ContextExceedsFreeMemory);
    }

    #[test]
    fn memory_report_format_matches_current_c_shape() {
        assert_eq!(
            format_cuda_memory_report(
                "after graph alloc",
                DeviceMemoryCapacity {
                    free_bytes: 24 * GIB,
                    total_bytes: 32 * GIB,
                }
            ),
            "ds4: CUDA memory report after graph alloc: free 24576.00 MiB total 32768.00 MiB"
        );
    }
}
