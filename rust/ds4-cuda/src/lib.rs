pub const CUDA_OXIDE_REVISION: &str = "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostSubstrateScope {
    pub opt_in_only: bool,
    pub owns_context_and_stream: bool,
    pub owns_device_buffer_roundtrip: bool,
    pub owns_managed_buffer_lifetime: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1A_SCOPE: HostSubstrateScope = HostSubstrateScope {
    opt_in_only: true,
    owns_context_and_stream: true,
    owns_device_buffer_roundtrip: true,
    owns_managed_buffer_lifetime: true,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelResidencyScope {
    pub opt_in_only: bool,
    pub owns_managed_advice_and_prefetch: bool,
    pub owns_mapped_host_buffer: bool,
    pub owns_registered_host_range: bool,
    pub owns_complete_model_map: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B1_SCOPE: ModelResidencyScope = ModelResidencyScope {
    opt_in_only: true,
    owns_managed_advice_and_prefetch: true,
    owns_mapped_host_buffer: true,
    owns_registered_host_range: true,
    owns_complete_model_map: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelRangeCopyScope {
    pub opt_in_only: bool,
    pub owns_mapped_model_file_lifetime: bool,
    pub owns_device_range_copy_cache: bool,
    pub owns_range_cache_reuse: bool,
    pub owns_range_strategy_selection: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2A_SCOPE: ModelRangeCopyScope = ModelRangeCopyScope {
    opt_in_only: true,
    owns_mapped_model_file_lifetime: true,
    owns_device_range_copy_cache: true,
    owns_range_cache_reuse: true,
    owns_range_strategy_selection: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelRangeStrategyScope {
    pub opt_in_only: bool,
    pub owns_explicit_mmap_device_copy_strategy: bool,
    pub owns_explicit_file_staged_device_copy_strategy: bool,
    pub owns_registered_range_strategy: bool,
    pub owns_pageable_hmm_strategy: bool,
    pub owns_o_direct_staging: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2B1_SCOPE: ModelRangeStrategyScope = ModelRangeStrategyScope {
    opt_in_only: true,
    owns_explicit_mmap_device_copy_strategy: true,
    owns_explicit_file_staged_device_copy_strategy: true,
    owns_registered_range_strategy: false,
    owns_pageable_hmm_strategy: false,
    owns_o_direct_staging: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[cfg(feature = "cuda-oxide-backend")]
pub mod model_map;

#[cfg(feature = "cuda-oxide-backend")]
pub mod substrate;

#[cfg(test)]
mod tests {
    use super::{
        CUDA_OXIDE_REVISION, M14_1A_SCOPE, M14_1B1_SCOPE, M14_1B2A_SCOPE, M14_1B2B1_SCOPE,
    };

    #[test]
    fn substrate_scope_does_not_overclaim_kernel_or_route_ownership() {
        assert_eq!(
            CUDA_OXIDE_REVISION,
            "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200"
        );
        assert!(M14_1A_SCOPE.opt_in_only);
        assert!(M14_1A_SCOPE.owns_context_and_stream);
        assert!(M14_1A_SCOPE.owns_device_buffer_roundtrip);
        assert!(M14_1A_SCOPE.owns_managed_buffer_lifetime);
        assert!(!M14_1A_SCOPE.owns_ds4_kernels);
        assert!(!M14_1A_SCOPE.changes_default_route);
    }

    #[test]
    fn residency_scope_does_not_overclaim_model_map_kernel_or_route_ownership() {
        assert!(M14_1B1_SCOPE.opt_in_only);
        assert!(M14_1B1_SCOPE.owns_managed_advice_and_prefetch);
        assert!(M14_1B1_SCOPE.owns_mapped_host_buffer);
        assert!(M14_1B1_SCOPE.owns_registered_host_range);
        assert!(!M14_1B1_SCOPE.owns_complete_model_map);
        assert!(!M14_1B1_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B1_SCOPE.changes_default_route);
    }

    #[test]
    fn range_copy_scope_does_not_overclaim_strategy_kernel_or_route_ownership() {
        assert!(M14_1B2A_SCOPE.opt_in_only);
        assert!(M14_1B2A_SCOPE.owns_mapped_model_file_lifetime);
        assert!(M14_1B2A_SCOPE.owns_device_range_copy_cache);
        assert!(M14_1B2A_SCOPE.owns_range_cache_reuse);
        assert!(!M14_1B2A_SCOPE.owns_range_strategy_selection);
        assert!(!M14_1B2A_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2A_SCOPE.changes_default_route);
    }

    #[test]
    fn file_staged_scope_does_not_overclaim_pending_strategy_or_route_ownership() {
        assert!(M14_1B2B1_SCOPE.opt_in_only);
        assert!(M14_1B2B1_SCOPE.owns_explicit_mmap_device_copy_strategy);
        assert!(M14_1B2B1_SCOPE.owns_explicit_file_staged_device_copy_strategy);
        assert!(!M14_1B2B1_SCOPE.owns_registered_range_strategy);
        assert!(!M14_1B2B1_SCOPE.owns_pageable_hmm_strategy);
        assert!(!M14_1B2B1_SCOPE.owns_o_direct_staging);
        assert!(!M14_1B2B1_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2B1_SCOPE.changes_default_route);
    }
}
