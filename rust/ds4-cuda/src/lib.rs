pub const CUDA_OXIDE_REVISION: &str = "d4791b7002152af3b7f6b15a48d7f5acd7a63011";

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisteredRangeStrategyScope {
    pub opt_in_only: bool,
    pub owns_page_aligned_read_only_registration_attempt: bool,
    pub owns_mmap_device_copy_fallback_after_registration_error: bool,
    pub owns_pageable_hmm_strategy: bool,
    pub owns_o_direct_staging: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2B2_SCOPE: RegisteredRangeStrategyScope = RegisteredRangeStrategyScope {
    opt_in_only: true,
    owns_page_aligned_read_only_registration_attempt: true,
    owns_mmap_device_copy_fallback_after_registration_error: true,
    owns_pageable_hmm_strategy: false,
    owns_o_direct_staging: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageableHmmStrategyScope {
    pub opt_in_only: bool,
    pub owns_page_aligned_pageable_hmm_prefetch: bool,
    pub owns_hmm_direct_read_pointer: bool,
    pub owns_o_direct_staging: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2B3A_SCOPE: PageableHmmStrategyScope = PageableHmmStrategyScope {
    opt_in_only: true,
    owns_page_aligned_pageable_hmm_prefetch: true,
    owns_hmm_direct_read_pointer: true,
    owns_o_direct_staging: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectIoStagingScope {
    pub opt_in_only: bool,
    pub owns_pinned_file_staging: bool,
    pub owns_o_direct_open_and_aligned_read: bool,
    pub owns_buffered_read_fallback: bool,
    pub owns_asynchronous_staging_ring: bool,
    pub owns_cache_budget_policy: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2B3B1_SCOPE: DirectIoStagingScope = DirectIoStagingScope {
    opt_in_only: true,
    owns_pinned_file_staging: true,
    owns_o_direct_open_and_aligned_read: true,
    owns_buffered_read_fallback: true,
    owns_asynchronous_staging_ring: false,
    owns_cache_budget_policy: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsyncStagingPolicyScope {
    pub opt_in_only: bool,
    pub owns_four_slot_event_ring: bool,
    pub owns_direct_io_disable_after_error_policy: bool,
    pub owns_arena_range_allocation: bool,
    pub owns_range_cache_budget_fallback: bool,
    pub owns_source_page_discard_policy: bool,
    pub owns_progress_reporting: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2B3B2_SCOPE: AsyncStagingPolicyScope = AsyncStagingPolicyScope {
    opt_in_only: true,
    owns_four_slot_event_ring: true,
    owns_direct_io_disable_after_error_policy: true,
    owns_arena_range_allocation: true,
    owns_range_cache_budget_fallback: true,
    owns_source_page_discard_policy: false,
    owns_progress_reporting: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelMapClosureScope {
    pub opt_in_only: bool,
    pub owns_containing_range_reuse: bool,
    pub owns_source_page_discard_policy: bool,
    pub owns_progress_reporting: bool,
    pub owns_raii_cache_cleanup: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B2C_SCOPE: ModelMapClosureScope = ModelMapClosureScope {
    opt_in_only: true,
    owns_containing_range_reuse: true,
    owns_source_page_discard_policy: true,
    owns_progress_reporting: true,
    owns_raii_cache_cleanup: true,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationPolicyScope {
    pub opt_in_only: bool,
    pub owns_managed_tensor_allocation: bool,
    pub owns_managed_kv_selection: bool,
    pub owns_memory_report: bool,
    pub owns_q8_cache_policy: bool,
    pub owns_quality_mode: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B3A_SCOPE: AllocationPolicyScope = AllocationPolicyScope {
    opt_in_only: true,
    owns_managed_tensor_allocation: true,
    owns_managed_kv_selection: true,
    owns_memory_report: true,
    owns_q8_cache_policy: false,
    owns_quality_mode: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8QualityPolicyScope {
    pub opt_in_only: bool,
    pub owns_q8_cache_admission_policy: bool,
    pub owns_q8_cache_failure_disable_policy: bool,
    pub owns_quality_blas_selection: bool,
    pub owns_converted_q8_buffers: bool,
    pub owns_dequant_kernels: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B3B_SCOPE: Q8QualityPolicyScope = Q8QualityPolicyScope {
    opt_in_only: true,
    owns_q8_cache_admission_policy: true,
    owns_q8_cache_failure_disable_policy: true,
    owns_quality_blas_selection: true,
    owns_converted_q8_buffers: false,
    owns_dequant_kernels: false,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FillCommandScope {
    pub opt_in_only: bool,
    pub owns_tensor_fill_f32: bool,
    pub owns_command_synchronization: bool,
    pub owns_dequant_kernels: bool,
    pub owns_graph_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1B4_SCOPE: FillCommandScope = FillCommandScope {
    opt_in_only: true,
    owns_tensor_fill_f32: true,
    owns_command_synchronization: true,
    owns_dequant_kernels: false,
    owns_graph_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElementwiseKernelScope {
    pub opt_in_only: bool,
    pub owns_add_tensor: bool,
    pub owns_repeat_hc_tensor: bool,
    pub owns_embedding_kernels: bool,
    pub owns_indexer_kernels: bool,
    pub owns_swiglu_and_directional_steering: bool,
    pub changes_default_route: bool,
}

pub const M14_2A_SCOPE: ElementwiseKernelScope = ElementwiseKernelScope {
    opt_in_only: true,
    owns_add_tensor: true,
    owns_repeat_hc_tensor: true,
    owns_embedding_kernels: false,
    owns_indexer_kernels: false,
    owns_swiglu_and_directional_steering: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectionalSteeringKernelScope {
    pub opt_in_only: bool,
    pub owns_swiglu_tensor: bool,
    pub owns_directional_steering_project_tensor: bool,
    pub owns_embedding_kernels: bool,
    pub owns_indexer_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_2B1_SCOPE: DirectionalSteeringKernelScope = DirectionalSteeringKernelScope {
    opt_in_only: true,
    owns_swiglu_tensor: false,
    owns_directional_steering_project_tensor: true,
    owns_embedding_kernels: false,
    owns_indexer_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwigluKernelScope {
    pub opt_in_only: bool,
    pub owns_swiglu_tensor: bool,
    pub owns_directional_steering_project_tensor: bool,
    pub owns_embedding_kernels: bool,
    pub owns_indexer_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_2B2_SCOPE: SwigluKernelScope = SwigluKernelScope {
    opt_in_only: true,
    owns_swiglu_tensor: true,
    owns_directional_steering_project_tensor: true,
    owns_embedding_kernels: false,
    owns_indexer_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingKernelScope {
    pub opt_in_only: bool,
    pub owns_embed_token_hc_tensor: bool,
    pub owns_embed_tokens_hc_tensor: bool,
    pub owns_model_range_consumption: bool,
    pub owns_indexer_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_2C_SCOPE: EmbeddingKernelScope = EmbeddingKernelScope {
    opt_in_only: true,
    owns_embed_token_hc_tensor: true,
    owns_embed_tokens_hc_tensor: true,
    owns_model_range_consumption: false,
    owns_indexer_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerScalarKernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_scores_fallback_kernel: bool,
    pub owns_indexer_topk_fallback_kernel: bool,
    pub owns_topk_mask_tensor: bool,
    pub owns_optimized_indexer_dispatch: bool,
    pub owns_optimized_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D1_SCOPE: IndexerScalarKernelScope = IndexerScalarKernelScope {
    opt_in_only: true,
    owns_indexer_scores_fallback_kernel: true,
    owns_indexer_topk_fallback_kernel: true,
    owns_topk_mask_tensor: true,
    owns_optimized_indexer_dispatch: false,
    owns_optimized_topk_dispatch: false,
    changes_default_route: false,
};

pub mod allocation_policy;
pub mod q8_policy;

#[cfg(feature = "cuda-oxide-backend")]
pub mod model_map;

#[cfg(feature = "cuda-oxide-backend")]
pub mod substrate;

#[cfg(test)]
mod tests {
    use super::{
        CUDA_OXIDE_REVISION, M14_1A_SCOPE, M14_1B1_SCOPE, M14_1B2A_SCOPE, M14_1B2B1_SCOPE,
        M14_1B2B2_SCOPE, M14_1B2B3A_SCOPE, M14_1B2B3B1_SCOPE, M14_1B2B3B2_SCOPE, M14_1B2C_SCOPE,
        M14_1B3A_SCOPE, M14_1B3B_SCOPE, M14_1B4_SCOPE, M14_2A_SCOPE, M14_2B1_SCOPE, M14_2B2_SCOPE,
        M14_2C_SCOPE, M14_2D1_SCOPE,
    };

    #[test]
    fn substrate_scope_does_not_overclaim_kernel_or_route_ownership() {
        assert_eq!(
            CUDA_OXIDE_REVISION,
            "d4791b7002152af3b7f6b15a48d7f5acd7a63011"
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

    #[test]
    fn registered_range_scope_records_fallback_without_pending_policy_claims() {
        assert!(M14_1B2B2_SCOPE.opt_in_only);
        assert!(M14_1B2B2_SCOPE.owns_page_aligned_read_only_registration_attempt);
        assert!(M14_1B2B2_SCOPE.owns_mmap_device_copy_fallback_after_registration_error);
        assert!(!M14_1B2B2_SCOPE.owns_pageable_hmm_strategy);
        assert!(!M14_1B2B2_SCOPE.owns_o_direct_staging);
        assert!(!M14_1B2B2_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2B2_SCOPE.changes_default_route);
    }

    #[test]
    fn pageable_hmm_scope_keeps_direct_io_kernels_and_route_pending() {
        assert!(M14_1B2B3A_SCOPE.opt_in_only);
        assert!(M14_1B2B3A_SCOPE.owns_page_aligned_pageable_hmm_prefetch);
        assert!(M14_1B2B3A_SCOPE.owns_hmm_direct_read_pointer);
        assert!(!M14_1B2B3A_SCOPE.owns_o_direct_staging);
        assert!(!M14_1B2B3A_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2B3A_SCOPE.changes_default_route);
    }

    #[test]
    fn direct_io_scope_leaves_ring_budget_kernels_and_route_pending() {
        assert!(M14_1B2B3B1_SCOPE.opt_in_only);
        assert!(M14_1B2B3B1_SCOPE.owns_pinned_file_staging);
        assert!(M14_1B2B3B1_SCOPE.owns_o_direct_open_and_aligned_read);
        assert!(M14_1B2B3B1_SCOPE.owns_buffered_read_fallback);
        assert!(!M14_1B2B3B1_SCOPE.owns_asynchronous_staging_ring);
        assert!(!M14_1B2B3B1_SCOPE.owns_cache_budget_policy);
        assert!(!M14_1B2B3B1_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2B3B1_SCOPE.changes_default_route);
    }

    #[test]
    fn async_staging_scope_leaves_cleanup_progress_kernels_and_route_pending() {
        assert!(M14_1B2B3B2_SCOPE.opt_in_only);
        assert!(M14_1B2B3B2_SCOPE.owns_four_slot_event_ring);
        assert!(M14_1B2B3B2_SCOPE.owns_direct_io_disable_after_error_policy);
        assert!(M14_1B2B3B2_SCOPE.owns_arena_range_allocation);
        assert!(M14_1B2B3B2_SCOPE.owns_range_cache_budget_fallback);
        assert!(!M14_1B2B3B2_SCOPE.owns_source_page_discard_policy);
        assert!(!M14_1B2B3B2_SCOPE.owns_progress_reporting);
        assert!(!M14_1B2B3B2_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2B3B2_SCOPE.changes_default_route);
    }

    #[test]
    fn model_map_closure_keeps_kernels_and_route_pending() {
        assert!(M14_1B2C_SCOPE.opt_in_only);
        assert!(M14_1B2C_SCOPE.owns_containing_range_reuse);
        assert!(M14_1B2C_SCOPE.owns_source_page_discard_policy);
        assert!(M14_1B2C_SCOPE.owns_progress_reporting);
        assert!(M14_1B2C_SCOPE.owns_raii_cache_cleanup);
        assert!(!M14_1B2C_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B2C_SCOPE.changes_default_route);
    }

    #[test]
    fn allocation_policy_leaves_q8_quality_kernels_and_route_pending() {
        assert!(M14_1B3A_SCOPE.opt_in_only);
        assert!(M14_1B3A_SCOPE.owns_managed_tensor_allocation);
        assert!(M14_1B3A_SCOPE.owns_managed_kv_selection);
        assert!(M14_1B3A_SCOPE.owns_memory_report);
        assert!(!M14_1B3A_SCOPE.owns_q8_cache_policy);
        assert!(!M14_1B3A_SCOPE.owns_quality_mode);
        assert!(!M14_1B3A_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B3A_SCOPE.changes_default_route);
    }

    #[test]
    fn q8_quality_policy_leaves_conversion_kernels_and_route_pending() {
        assert!(M14_1B3B_SCOPE.opt_in_only);
        assert!(M14_1B3B_SCOPE.owns_q8_cache_admission_policy);
        assert!(M14_1B3B_SCOPE.owns_q8_cache_failure_disable_policy);
        assert!(M14_1B3B_SCOPE.owns_quality_blas_selection);
        assert!(!M14_1B3B_SCOPE.owns_converted_q8_buffers);
        assert!(!M14_1B3B_SCOPE.owns_dequant_kernels);
        assert!(!M14_1B3B_SCOPE.owns_ds4_kernels);
        assert!(!M14_1B3B_SCOPE.changes_default_route);
    }

    #[test]
    fn fill_command_scope_leaves_dequant_graph_kernels_and_route_pending() {
        assert!(M14_1B4_SCOPE.opt_in_only);
        assert!(M14_1B4_SCOPE.owns_tensor_fill_f32);
        assert!(M14_1B4_SCOPE.owns_command_synchronization);
        assert!(!M14_1B4_SCOPE.owns_dequant_kernels);
        assert!(!M14_1B4_SCOPE.owns_graph_kernels);
        assert!(!M14_1B4_SCOPE.changes_default_route);
    }

    #[test]
    fn elementwise_scope_leaves_remaining_m14_2_kernels_and_route_pending() {
        assert!(M14_2A_SCOPE.opt_in_only);
        assert!(M14_2A_SCOPE.owns_add_tensor);
        assert!(M14_2A_SCOPE.owns_repeat_hc_tensor);
        assert!(!M14_2A_SCOPE.owns_embedding_kernels);
        assert!(!M14_2A_SCOPE.owns_indexer_kernels);
        assert!(!M14_2A_SCOPE.owns_swiglu_and_directional_steering);
        assert!(!M14_2A_SCOPE.changes_default_route);
    }

    #[test]
    fn directional_steering_scope_leaves_swiglu_model_kernels_and_route_pending() {
        assert!(M14_2B1_SCOPE.opt_in_only);
        assert!(!M14_2B1_SCOPE.owns_swiglu_tensor);
        assert!(M14_2B1_SCOPE.owns_directional_steering_project_tensor);
        assert!(!M14_2B1_SCOPE.owns_embedding_kernels);
        assert!(!M14_2B1_SCOPE.owns_indexer_kernels);
        assert!(!M14_2B1_SCOPE.changes_default_route);
    }

    #[test]
    fn swiglu_scope_retains_directional_ownership_without_later_kernel_or_route_claims() {
        assert!(M14_2B2_SCOPE.opt_in_only);
        assert!(M14_2B2_SCOPE.owns_swiglu_tensor);
        assert!(M14_2B2_SCOPE.owns_directional_steering_project_tensor);
        assert!(!M14_2B2_SCOPE.owns_embedding_kernels);
        assert!(!M14_2B2_SCOPE.owns_indexer_kernels);
        assert!(!M14_2B2_SCOPE.changes_default_route);
    }

    #[test]
    fn embedding_scope_leaves_model_range_indexer_and_route_integration_pending() {
        assert!(M14_2C_SCOPE.opt_in_only);
        assert!(M14_2C_SCOPE.owns_embed_token_hc_tensor);
        assert!(M14_2C_SCOPE.owns_embed_tokens_hc_tensor);
        assert!(!M14_2C_SCOPE.owns_model_range_consumption);
        assert!(!M14_2C_SCOPE.owns_indexer_kernels);
        assert!(!M14_2C_SCOPE.changes_default_route);
    }

    #[test]
    fn scalar_indexer_scope_leaves_optimized_dispatch_and_route_pending() {
        assert!(M14_2D1_SCOPE.opt_in_only);
        assert!(M14_2D1_SCOPE.owns_indexer_scores_fallback_kernel);
        assert!(M14_2D1_SCOPE.owns_indexer_topk_fallback_kernel);
        assert!(M14_2D1_SCOPE.owns_topk_mask_tensor);
        assert!(!M14_2D1_SCOPE.owns_optimized_indexer_dispatch);
        assert!(!M14_2D1_SCOPE.owns_optimized_topk_dispatch);
        assert!(!M14_2D1_SCOPE.changes_default_route);
    }
}
