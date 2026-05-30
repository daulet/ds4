pub const CUDA_OXIDE_REVISION: &str = "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7";

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerDirectKernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_score_one_direct_kernel: bool,
    pub owns_wmma_indexer_dispatch: bool,
    pub owns_specialized_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2A_SCOPE: IndexerDirectKernelScope = IndexerDirectKernelScope {
    opt_in_only: true,
    owns_indexer_score_one_direct_kernel: true,
    owns_wmma_indexer_dispatch: false,
    owns_specialized_topk_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerWmmaKernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_scores_wmma_kernel: bool,
    pub owns_widened_wmma_dispatch: bool,
    pub owns_specialized_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2B1_SCOPE: IndexerWmmaKernelScope = IndexerWmmaKernelScope {
    opt_in_only: true,
    owns_indexer_scores_wmma_kernel: true,
    owns_widened_wmma_dispatch: false,
    owns_specialized_topk_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerWmma32KernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_scores_wmma32_kernel: bool,
    pub owns_wmma64_and_wmma128_dispatch: bool,
    pub owns_specialized_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2B2A_SCOPE: IndexerWmma32KernelScope = IndexerWmma32KernelScope {
    opt_in_only: true,
    owns_indexer_scores_wmma32_kernel: true,
    owns_wmma64_and_wmma128_dispatch: false,
    owns_specialized_topk_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerWmma64KernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_scores_wmma64_kernel: bool,
    pub owns_wmma128_and_dispatch_priority: bool,
    pub owns_specialized_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2B2B_SCOPE: IndexerWmma64KernelScope = IndexerWmma64KernelScope {
    opt_in_only: true,
    owns_indexer_scores_wmma64_kernel: true,
    owns_wmma128_and_dispatch_priority: false,
    owns_specialized_topk_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexerScoreKernel {
    Scalar,
    DirectOne,
    Wmma,
    Wmma32,
    Wmma64,
    Wmma128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerScoreDispatchOptions {
    pub n_tokens: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub quality_mode: bool,
    pub no_direct_one: bool,
    pub no_wmma: bool,
    pub no_wmma128: bool,
    pub no_wmma64: bool,
    pub no_wmma32: bool,
}

pub const fn select_indexer_score_kernel(
    options: IndexerScoreDispatchOptions,
) -> IndexerScoreKernel {
    if options.n_tokens == 1
        && options.head_dim == 128
        && options.n_head == 64
        && !options.no_direct_one
    {
        return IndexerScoreKernel::DirectOne;
    }
    if !options.quality_mode && options.head_dim == 128 && options.n_head == 64 && !options.no_wmma
    {
        if !options.no_wmma128 {
            IndexerScoreKernel::Wmma128
        } else if !options.no_wmma64 {
            IndexerScoreKernel::Wmma64
        } else if !options.no_wmma32 {
            IndexerScoreKernel::Wmma32
        } else {
            IndexerScoreKernel::Wmma
        }
    } else {
        IndexerScoreKernel::Scalar
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerWmma128DispatchScope {
    pub opt_in_only: bool,
    pub owns_indexer_scores_wmma128_kernel: bool,
    pub owns_indexer_score_dispatch_policy: bool,
    pub owns_specialized_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2B2C_SCOPE: IndexerWmma128DispatchScope = IndexerWmma128DispatchScope {
    opt_in_only: true,
    owns_indexer_scores_wmma128_kernel: true,
    owns_indexer_score_dispatch_policy: true,
    owns_specialized_topk_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerTopk1024KernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_topk_1024_kernel: bool,
    pub owns_larger_topk_dispatch: bool,
    pub owns_indexed_topk_sort_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2C1_SCOPE: IndexerTopk1024KernelScope = IndexerTopk1024KernelScope {
    opt_in_only: true,
    owns_indexer_topk_1024_kernel: true,
    owns_larger_topk_dispatch: false,
    owns_indexed_topk_sort_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerTopkPow2KernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_topk_pow2_2048_kernel: bool,
    pub owns_indexer_topk_pow2_4096_kernel: bool,
    pub owns_indexer_topk_pow2_u16_8192_kernel: bool,
    pub owns_cub_topk_dispatch: bool,
    pub owns_chunked_topk_dispatch: bool,
    pub owns_indexed_topk_sort_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2C2_SCOPE: IndexerTopkPow2KernelScope = IndexerTopkPow2KernelScope {
    opt_in_only: true,
    owns_indexer_topk_pow2_2048_kernel: true,
    owns_indexer_topk_pow2_4096_kernel: true,
    owns_indexer_topk_pow2_u16_8192_kernel: true,
    owns_cub_topk_dispatch: false,
    owns_chunked_topk_dispatch: false,
    owns_indexed_topk_sort_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerTopkPackedKeyKernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_topk_8192_packed_key_equivalent_kernel: bool,
    pub owns_dynamic_shared_launch_shape: bool,
    pub owns_cub_library_implementation: bool,
    pub owns_topk_dispatch_policy: bool,
    pub owns_chunked_topk_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2C3_SCOPE: IndexerTopkPackedKeyKernelScope = IndexerTopkPackedKeyKernelScope {
    opt_in_only: true,
    owns_indexer_topk_8192_packed_key_equivalent_kernel: true,
    owns_dynamic_shared_launch_shape: true,
    owns_cub_library_implementation: false,
    owns_topk_dispatch_policy: false,
    owns_chunked_topk_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerTopkTreeKernelScope {
    pub opt_in_only: bool,
    pub owns_indexer_topk_chunk_pow2_4096_kernel: bool,
    pub owns_indexer_topk_tree_merge_pow2_4096_kernel: bool,
    pub owns_indexer_topk_merge_pow2_4096_kernel: bool,
    pub owns_scratch_layout: bool,
    pub owns_topk_dispatch_policy: bool,
    pub owns_indexed_topk_sort_dispatch: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2C4_SCOPE: IndexerTopkTreeKernelScope = IndexerTopkTreeKernelScope {
    opt_in_only: true,
    owns_indexer_topk_chunk_pow2_4096_kernel: true,
    owns_indexer_topk_tree_merge_pow2_4096_kernel: true,
    owns_indexer_topk_merge_pow2_4096_kernel: true,
    owns_scratch_layout: true,
    owns_topk_dispatch_policy: false,
    owns_indexed_topk_sort_dispatch: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexerTopkKernel {
    Scalar,
    Topk1024,
    Pow2U32x2048,
    Pow2U32x4096,
    PackedKeyEquivalent,
    Pow2U16x8192,
    ChunkedTree,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerTopkDispatchOptions {
    pub n_comp: u32,
    pub top_k: u32,
    pub no_topk1024: bool,
    pub no_topk2048: bool,
    pub no_topk8192: bool,
    pub no_topk_chunked: bool,
    pub packed_dynamic_shared_available: bool,
}

pub const fn select_indexer_topk_kernel(options: IndexerTopkDispatchOptions) -> IndexerTopkKernel {
    if options.top_k == 512 && options.n_comp <= 1024 && !options.no_topk1024 {
        return IndexerTopkKernel::Topk1024;
    }
    if options.top_k == 512 && options.n_comp <= 2048 && !options.no_topk2048 {
        return IndexerTopkKernel::Pow2U32x2048;
    }
    if options.top_k == 512 && options.n_comp <= 4096 && !options.no_topk2048 {
        if options.n_comp == 4096 && options.packed_dynamic_shared_available {
            return IndexerTopkKernel::PackedKeyEquivalent;
        }
        return IndexerTopkKernel::Pow2U32x4096;
    }
    if options.top_k == 512
        && options.n_comp <= 8192
        && !options.no_topk2048
        && !options.no_topk8192
    {
        if options.n_comp > 4096 && options.packed_dynamic_shared_available {
            return IndexerTopkKernel::PackedKeyEquivalent;
        }
        return IndexerTopkKernel::Pow2U16x8192;
    }
    if options.top_k == 512 && !options.no_topk2048 && !options.no_topk_chunked {
        return IndexerTopkKernel::ChunkedTree;
    }
    IndexerTopkKernel::Scalar
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedTopkSortOptions {
    pub n_tokens: u32,
    pub top_k: u32,
    pub no_indexed_topk_sort: bool,
}

pub const fn should_sort_indexed_topk(options: IndexedTopkSortOptions) -> bool {
    options.n_tokens > 1 && options.top_k == 512 && !options.no_indexed_topk_sort
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexerTopkDispatchScope {
    pub opt_in_only: bool,
    pub owns_indexed_topk_sort_512_asc_kernel: bool,
    pub owns_indexed_topk_sort_dispatch: bool,
    pub owns_topk_dispatch_policy: bool,
    pub uses_packed_key_equivalent_branch: bool,
    pub owns_cub_library_implementation: bool,
    pub changes_default_route: bool,
}

pub const M14_2D2C5_SCOPE: IndexerTopkDispatchScope = IndexerTopkDispatchScope {
    opt_in_only: true,
    owns_indexed_topk_sort_512_asc_kernel: true,
    owns_indexed_topk_sort_dispatch: true,
    owns_topk_dispatch_policy: true,
    uses_packed_key_equivalent_branch: true,
    owns_cub_library_implementation: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormKernelScope {
    pub opt_in_only: bool,
    pub owns_rms_norm_plain_kernel: bool,
    pub owns_rms_norm_weight_kernel: bool,
    pub owns_plain_and_weighted_tensor_surface: bool,
    pub owns_fused_qkv_and_head_norm_kernels: bool,
    pub owns_dense_projection_or_q8_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_3A_SCOPE: RmsNormKernelScope = RmsNormKernelScope {
    opt_in_only: true,
    owns_rms_norm_plain_kernel: true,
    owns_rms_norm_weight_kernel: true,
    owns_plain_and_weighted_tensor_surface: true,
    owns_fused_qkv_and_head_norm_kernels: false,
    owns_dense_projection_or_q8_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusedRmsNormKernelScope {
    pub opt_in_only: bool,
    pub owns_dsv4_qkv_rms_norm_rows_kernel: bool,
    pub owns_head_rms_norm_kernel: bool,
    pub owns_head_rms_norm_rope_tail_kernel: bool,
    pub owns_qkv_fused_dispatch_policy: bool,
    pub owns_dense_projection_or_q8_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_3B1_SCOPE: FusedRmsNormKernelScope = FusedRmsNormKernelScope {
    opt_in_only: true,
    owns_dsv4_qkv_rms_norm_rows_kernel: true,
    owns_head_rms_norm_kernel: true,
    owns_head_rms_norm_rope_tail_kernel: false,
    owns_qkv_fused_dispatch_policy: false,
    owns_dense_projection_or_q8_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadRmsRopeTailKernelScope {
    pub opt_in_only: bool,
    pub owns_head_rms_norm_rope_tail_kernel: bool,
    pub owns_yarn_rotary_math_path: bool,
    pub owns_standalone_rope_tail_kernel: bool,
    pub owns_qkv_fused_dispatch_policy: bool,
    pub owns_dense_projection_or_q8_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_3B2_SCOPE: HeadRmsRopeTailKernelScope = HeadRmsRopeTailKernelScope {
    opt_in_only: true,
    owns_head_rms_norm_rope_tail_kernel: true,
    owns_yarn_rotary_math_path: true,
    owns_standalone_rope_tail_kernel: false,
    owns_qkv_fused_dispatch_policy: false,
    owns_dense_projection_or_q8_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseProjectionKernelScope {
    pub opt_in_only: bool,
    pub owns_matmul_f16_kernel: bool,
    pub owns_matmul_f32_kernel: bool,
    pub owns_ordered_or_pair_f16_kernels: bool,
    pub owns_f16_or_cublas_dispatch_policy: bool,
    pub owns_q8_conversion_or_matmul_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_3C1_SCOPE: DenseProjectionKernelScope = DenseProjectionKernelScope {
    opt_in_only: true,
    owns_matmul_f16_kernel: true,
    owns_matmul_f32_kernel: true,
    owns_ordered_or_pair_f16_kernels: false,
    owns_f16_or_cublas_dispatch_policy: false,
    owns_q8_conversion_or_matmul_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrderedProjectionKernelScope {
    pub opt_in_only: bool,
    pub owns_matmul_f16_serial_kernel: bool,
    pub owns_matmul_f16_ordered_chunks_kernel: bool,
    pub owns_matmul_f16_pair_ordered_chunks_kernel: bool,
    pub owns_f16_or_cublas_dispatch_policy: bool,
    pub owns_q8_conversion_or_matmul_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_3C2_SCOPE: OrderedProjectionKernelScope = OrderedProjectionKernelScope {
    opt_in_only: true,
    owns_matmul_f16_serial_kernel: true,
    owns_matmul_f16_ordered_chunks_kernel: true,
    owns_matmul_f16_pair_ordered_chunks_kernel: true,
    owns_f16_or_cublas_dispatch_policy: false,
    owns_q8_conversion_or_matmul_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F16ProjectionPath {
    Blas,
    Serial,
    OrderedChunks,
    Base,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct F16ProjectionDispatch {
    pub blas_ready: bool,
    pub serial_f16: bool,
    pub serial_router: bool,
    pub no_ordered_f16_matmul: bool,
    pub in_dim: u64,
    pub out_dim: u64,
    pub n_tokens: u64,
}

pub fn select_f16_projection_path(options: F16ProjectionDispatch) -> F16ProjectionPath {
    let router_shape = options.in_dim == 4096 && options.out_dim == 256 && options.n_tokens == 1;
    let serial_router = !options.serial_f16 && router_shape && options.serial_router;
    let ordered = !options.serial_f16
        && !serial_router
        && options.n_tokens == 1
        && !options.no_ordered_f16_matmul;
    if !options.serial_f16 && options.blas_ready && options.n_tokens > 1 {
        F16ProjectionPath::Blas
    } else if options.serial_f16 || serial_router {
        F16ProjectionPath::Serial
    } else if ordered {
        F16ProjectionPath::OrderedChunks
    } else {
        F16ProjectionPath::Base
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F16PairProjectionPath {
    PairedOrderedChunks,
    TwoIndependent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct F16PairProjectionDispatch {
    pub n_tokens: u64,
    pub no_f16_pair_matmul: bool,
    pub serial_f16: bool,
    pub serial_router: bool,
    pub no_ordered_f16_matmul: bool,
}

pub fn select_f16_pair_projection_path(
    options: F16PairProjectionDispatch,
) -> F16PairProjectionPath {
    if options.n_tokens != 1
        || options.no_f16_pair_matmul
        || options.serial_f16
        || options.serial_router
        || options.no_ordered_f16_matmul
    {
        F16PairProjectionPath::TwoIndependent
    } else {
        F16PairProjectionPath::PairedOrderedChunks
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32ProjectionPath {
    Blas,
    Base,
}

pub fn select_f32_projection_path(blas_ready: bool, n_tokens: u64) -> F32ProjectionPath {
    if blas_ready && n_tokens > 1 {
        F32ProjectionPath::Blas
    } else {
        F32ProjectionPath::Base
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlasProjectionKernelScope {
    pub opt_in_only: bool,
    pub owns_f32_to_f16_kernel: bool,
    pub owns_f16_projection_dispatch_policy: bool,
    pub owns_f16_pair_projection_dispatch_policy: bool,
    pub owns_f32_projection_dispatch_policy: bool,
    pub owns_live_f16_and_f32_blas_paths: bool,
    pub owns_q8_conversion_or_matmul_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_3C3_SCOPE: BlasProjectionKernelScope = BlasProjectionKernelScope {
    opt_in_only: true,
    owns_f32_to_f16_kernel: true,
    owns_f16_projection_dispatch_policy: true,
    owns_f16_pair_projection_dispatch_policy: true,
    owns_f32_projection_dispatch_policy: true,
    owns_live_f16_and_f32_blas_paths: true,
    owns_q8_conversion_or_matmul_kernels: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8ConversionKernelScope {
    pub opt_in_only: bool,
    pub owns_dequant_q8_0_to_f16_kernel: bool,
    pub owns_dequant_q8_0_to_f32_kernel: bool,
    pub owns_quantize_q8_0_f32_kernel: bool,
    pub owns_quantized_matmul_kernels: bool,
    pub owns_q8_matmul_dispatch_policy: bool,
    pub changes_default_route: bool,
}

pub const M14_3D1_SCOPE: Q8ConversionKernelScope = Q8ConversionKernelScope {
    opt_in_only: true,
    owns_dequant_q8_0_to_f16_kernel: true,
    owns_dequant_q8_0_to_f32_kernel: true,
    owns_quantize_q8_0_f32_kernel: true,
    owns_quantized_matmul_kernels: false,
    owns_q8_matmul_dispatch_policy: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8MatmulKernelScope {
    pub opt_in_only: bool,
    pub owns_matmul_q8_0_kernel: bool,
    pub owns_matmul_q8_0_preq_kernel: bool,
    pub owns_matmul_q8_0_preq_warp8_kernel: bool,
    pub owns_matmul_q8_0_preq_batch_warp8_kernel: bool,
    pub owns_dp4a_acceleration: bool,
    pub owns_pair_or_hc_expand_kernels: bool,
    pub owns_q8_matmul_dispatch_policy: bool,
    pub changes_default_route: bool,
}

pub const M14_3D2_SCOPE: Q8MatmulKernelScope = Q8MatmulKernelScope {
    opt_in_only: true,
    owns_matmul_q8_0_kernel: true,
    owns_matmul_q8_0_preq_kernel: true,
    owns_matmul_q8_0_preq_warp8_kernel: true,
    owns_matmul_q8_0_preq_batch_warp8_kernel: true,
    owns_dp4a_acceleration: false,
    owns_pair_or_hc_expand_kernels: false,
    owns_q8_matmul_dispatch_policy: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8SpecializedMatmulKernelScope {
    pub opt_in_only: bool,
    pub owns_matmul_q8_0_pair_preq_warp8_kernel: bool,
    pub owns_matmul_q8_0_hc_expand_preq_warp8_kernel: bool,
    pub owns_hc_expand_optional_block_add: bool,
    pub owns_dp4a_acceleration: bool,
    pub owns_q8_matmul_dispatch_policy: bool,
    pub changes_default_route: bool,
}

pub const M14_3D3_SCOPE: Q8SpecializedMatmulKernelScope = Q8SpecializedMatmulKernelScope {
    opt_in_only: true,
    owns_matmul_q8_0_pair_preq_warp8_kernel: true,
    owns_matmul_q8_0_hc_expand_preq_warp8_kernel: true,
    owns_hc_expand_optional_block_add: true,
    owns_dp4a_acceleration: false,
    owns_q8_matmul_dispatch_policy: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8MatmulPath {
    ExpandedF32Blas,
    ExpandedF16Blas,
    PrequantizedWarp8,
    PrequantizedBatchWarp8,
    PrequantizedGeneric,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8MatmulDispatchOptions {
    pub cublas_ready: bool,
    pub expanded_f32_blas_ready: bool,
    pub expanded_f16_blas_ready: bool,
    pub n_tokens: u64,
    pub blocks: u64,
    pub no_batch_warp: bool,
}

pub const fn select_q8_matmul_path(options: Q8MatmulDispatchOptions) -> Q8MatmulPath {
    if options.cublas_ready && options.n_tokens > 1 && options.expanded_f32_blas_ready {
        Q8MatmulPath::ExpandedF32Blas
    } else if options.cublas_ready && options.n_tokens > 1 && options.expanded_f16_blas_ready {
        Q8MatmulPath::ExpandedF16Blas
    } else if options.n_tokens == 1 {
        Q8MatmulPath::PrequantizedWarp8
    } else if !options.no_batch_warp && options.blocks <= 32 {
        Q8MatmulPath::PrequantizedBatchWarp8
    } else {
        Q8MatmulPath::PrequantizedGeneric
    }
}

pub const fn q8_dp4a_enabled(no_q8_dp4a: bool) -> bool {
    !no_q8_dp4a
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8Dp4aDispatchScope {
    pub opt_in_only: bool,
    pub owns_cuda_oxide_dp4a_i8_intrinsic: bool,
    pub owns_dp4a_acceleration: bool,
    pub owns_q8_matmul_dispatch_policy: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_3D4_SCOPE: Q8Dp4aDispatchScope = Q8Dp4aDispatchScope {
    opt_in_only: true,
    owns_cuda_oxide_dp4a_i8_intrinsic: true,
    owns_dp4a_acceleration: true,
    owns_q8_matmul_dispatch_policy: true,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RopeKvQuantizationKernelScope {
    pub opt_in_only: bool,
    pub owns_standalone_rope_tail_kernel: bool,
    pub owns_fp8_kv_quantize_kernel: bool,
    pub owns_yarn_rotary_math_path: bool,
    pub owns_kv_storage_or_compressor_kernels: bool,
    pub owns_attention_kernels: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4A_SCOPE: RopeKvQuantizationKernelScope = RopeKvQuantizationKernelScope {
    opt_in_only: true,
    owns_standalone_rope_tail_kernel: true,
    owns_fp8_kv_quantize_kernel: true,
    owns_yarn_rotary_math_path: true,
    owns_kv_storage_or_compressor_kernels: false,
    owns_attention_kernels: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawKvIndexerQatKernelScope {
    pub opt_in_only: bool,
    pub owns_store_raw_kv_batch_kernel: bool,
    pub owns_raw_kv_store_surfaces: bool,
    pub owns_indexer_hadamard_fp4_kernel: bool,
    pub owns_indexer_qat_surface: bool,
    pub owns_kv_fp8_store_raw_composition: bool,
    pub owns_compressor_kernels: bool,
    pub owns_attention_kernels: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4B_SCOPE: RawKvIndexerQatKernelScope = RawKvIndexerQatKernelScope {
    opt_in_only: true,
    owns_store_raw_kv_batch_kernel: true,
    owns_raw_kv_store_surfaces: true,
    owns_indexer_hadamard_fp4_kernel: true,
    owns_indexer_qat_surface: true,
    owns_kv_fp8_store_raw_composition: false,
    owns_compressor_kernels: false,
    owns_attention_kernels: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComposedKvCompressorStoreKernelScope {
    pub opt_in_only: bool,
    pub owns_kv_fp8_store_raw_composition: bool,
    pub owns_compressor_store_kernel: bool,
    pub owns_compressor_set_rows_kernel: bool,
    pub owns_f32_and_f16_ape_reads: bool,
    pub owns_compressor_pooling_or_shift: bool,
    pub owns_attention_kernels: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4C1_SCOPE: ComposedKvCompressorStoreKernelScope =
    ComposedKvCompressorStoreKernelScope {
        opt_in_only: true,
        owns_kv_fp8_store_raw_composition: true,
        owns_compressor_store_kernel: true,
        owns_compressor_set_rows_kernel: true,
        owns_f32_and_f16_ape_reads: true,
        owns_compressor_pooling_or_shift: false,
        owns_attention_kernels: false,
        owns_runtime_graph_integration: false,
        changes_default_route: false,
    };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressorPoolShiftKernelScope {
    pub opt_in_only: bool,
    pub owns_compressor_prefill_pool_kernel: bool,
    pub owns_general_and_ratio4_prefill_branches: bool,
    pub owns_ratio4_replay_branch: bool,
    pub owns_compressor_update_pool_kernel: bool,
    pub owns_compressor_shift_ratio4_kernel: bool,
    pub owns_compressor_wrapper_orchestration: bool,
    pub owns_attention_kernels: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4C2_SCOPE: CompressorPoolShiftKernelScope = CompressorPoolShiftKernelScope {
    opt_in_only: true,
    owns_compressor_prefill_pool_kernel: true,
    owns_general_and_ratio4_prefill_branches: true,
    owns_ratio4_replay_branch: true,
    owns_compressor_update_pool_kernel: true,
    owns_compressor_shift_ratio4_kernel: true,
    owns_compressor_wrapper_orchestration: false,
    owns_attention_kernels: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressorUpdateOrchestrationScope {
    pub opt_in_only: bool,
    pub owns_compressor_update_orchestration: bool,
    pub owns_store_pool_norm_rope_shift_sequence: bool,
    pub owns_compressor_prefill_orchestration: bool,
    pub owns_attention_kernels: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4C3A_SCOPE: CompressorUpdateOrchestrationScope = CompressorUpdateOrchestrationScope {
    opt_in_only: true,
    owns_compressor_update_orchestration: true,
    owns_store_pool_norm_rope_shift_sequence: true,
    owns_compressor_prefill_orchestration: false,
    owns_attention_kernels: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressorPrefillOrchestrationScope {
    pub opt_in_only: bool,
    pub owns_compressor_prefill_orchestration: bool,
    pub owns_ratio4_replay_orchestration: bool,
    pub owns_ratio4_state_only_orchestration: bool,
    pub owns_optional_fp8_compressed_output: bool,
    pub owns_attention_kernels: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4C3B_SCOPE: CompressorPrefillOrchestrationScope =
    CompressorPrefillOrchestrationScope {
        opt_in_only: true,
        owns_compressor_prefill_orchestration: true,
        owns_ratio4_replay_orchestration: true,
        owns_ratio4_state_only_orchestration: true,
        owns_optional_fp8_compressed_output: true,
        owns_attention_kernels: false,
        owns_runtime_graph_integration: false,
        changes_default_route: false,
    };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionDecodeSingleMixedScope {
    pub opt_in_only: bool,
    pub owns_attention_decode_heads_surface: bool,
    pub owns_single_token_ring_raw_and_compressed_attention: bool,
    pub owns_masked_compressed_rows: bool,
    pub owns_batched_or_online_decode: bool,
    pub owns_prefill_indexed_or_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D1_SCOPE: AttentionDecodeSingleMixedScope = AttentionDecodeSingleMixedScope {
    opt_in_only: true,
    owns_attention_decode_heads_surface: true,
    owns_single_token_ring_raw_and_compressed_attention: true,
    owns_masked_compressed_rows: true,
    owns_batched_or_online_decode: false,
    owns_prefill_indexed_or_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionDecodeBatchMixedScope {
    pub opt_in_only: bool,
    pub owns_attention_decode_raw_batch_surface: bool,
    pub owns_attention_decode_mixed_batch_surface: bool,
    pub owns_causal_window_and_visible_compressed_rows: bool,
    pub owns_heads8_online_decode: bool,
    pub owns_prefill_indexed_or_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D2_SCOPE: AttentionDecodeBatchMixedScope = AttentionDecodeBatchMixedScope {
    opt_in_only: true,
    owns_attention_decode_raw_batch_surface: true,
    owns_attention_decode_mixed_batch_surface: true,
    owns_causal_window_and_visible_compressed_rows: true,
    owns_heads8_online_decode: false,
    owns_prefill_indexed_or_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

pub const DS4_CUDA_ATTENTION_SCORE_CAP: u32 = 8192;
pub const DS4_CUDA_ATTENTION_RAW_SCORE_CAP: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionDecodePath {
    Generic,
    Heads8OnlineOverflow,
    Heads8OnlineWindow,
    RejectScoreBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionDecodeDispatchOptions {
    pub n_tokens: u32,
    pub n_comp: u32,
    pub use_comp_mask: bool,
    pub head_dim: u32,
    pub no_window_attention: bool,
    pub window_attention: bool,
    pub quality_mode: bool,
}

pub const fn select_attention_decode_path(
    options: AttentionDecodeDispatchOptions,
) -> AttentionDecodePath {
    if options.n_comp > DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP {
        if !options.use_comp_mask && options.head_dim == 512 && !options.no_window_attention {
            return AttentionDecodePath::Heads8OnlineOverflow;
        }
        return AttentionDecodePath::RejectScoreBuffer;
    }
    if !options.use_comp_mask
        && options.n_tokens > 1
        && options.head_dim == 512
        && !options.no_window_attention
        && (options.window_attention || (!options.quality_mode && options.n_tokens >= 128))
    {
        return AttentionDecodePath::Heads8OnlineWindow;
    }
    AttentionDecodePath::Generic
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionDecodeHeads8OnlineScope {
    pub opt_in_only: bool,
    pub owns_heads8_online_decode_kernel: bool,
    pub owns_decode_online_dispatch_policy: bool,
    pub owns_prefill_or_indexed_online_attention: bool,
    pub owns_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D3_SCOPE: AttentionDecodeHeads8OnlineScope = AttentionDecodeHeads8OnlineScope {
    opt_in_only: true,
    owns_heads8_online_decode_kernel: true,
    owns_decode_online_dispatch_policy: true,
    owns_prefill_or_indexed_online_attention: false,
    owns_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionPrefillGenericScope {
    pub opt_in_only: bool,
    pub owns_attention_prefill_raw_surface: bool,
    pub owns_attention_prefill_static_mixed_surface: bool,
    pub owns_attention_prefill_masked_mixed_surface: bool,
    pub owns_generic_prefill_kernels: bool,
    pub owns_static_heads8_online_or_cublas_prefill_dispatch: bool,
    pub owns_indexed_or_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D4_SCOPE: AttentionPrefillGenericScope = AttentionPrefillGenericScope {
    opt_in_only: true,
    owns_attention_prefill_raw_surface: true,
    owns_attention_prefill_static_mixed_surface: true,
    owns_attention_prefill_masked_mixed_surface: true,
    owns_generic_prefill_kernels: true,
    owns_static_heads8_online_or_cublas_prefill_dispatch: false,
    owns_indexed_or_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionPrefillPath {
    StaticHeads8Online,
    Cublas,
    Generic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionPrefillDispatchOptions {
    pub use_comp_mask: bool,
    pub n_tokens: u32,
    pub head_dim: u32,
    pub cublas_ready: bool,
    pub no_cublas_attention: bool,
    pub no_window_attention: bool,
    pub window_attention: bool,
    pub quality_mode: bool,
}

pub const fn select_attention_prefill_path(
    options: AttentionPrefillDispatchOptions,
) -> AttentionPrefillPath {
    if !options.use_comp_mask
        && options.n_tokens > 1
        && options.head_dim == 512
        && !options.no_window_attention
        && (options.window_attention || (!options.quality_mode && options.n_tokens >= 128))
    {
        AttentionPrefillPath::StaticHeads8Online
    } else if options.cublas_ready
        && options.n_tokens > 1
        && options.head_dim == 512
        && !options.no_cublas_attention
    {
        AttentionPrefillPath::Cublas
    } else {
        AttentionPrefillPath::Generic
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionPrefillOptimizedScope {
    pub opt_in_only: bool,
    pub owns_static_heads8_online_prefill_kernel: bool,
    pub owns_prefill_dispatch_policy: bool,
    pub owns_live_cublas_prefill_pipeline: bool,
    pub owns_indexed_or_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D5_SCOPE: AttentionPrefillOptimizedScope = AttentionPrefillOptimizedScope {
    opt_in_only: true,
    owns_static_heads8_online_prefill_kernel: true,
    owns_prefill_dispatch_policy: true,
    owns_live_cublas_prefill_pipeline: true,
    owns_indexed_or_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionIndexedGenericScope {
    pub opt_in_only: bool,
    pub owns_attention_indexed_mixed_surface: bool,
    pub owns_generic_indexed_kernel: bool,
    pub owns_topk_filter_and_order_semantics: bool,
    pub owns_indexed_sort_or_heads8_dispatch: bool,
    pub owns_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D6_SCOPE: AttentionIndexedGenericScope = AttentionIndexedGenericScope {
    opt_in_only: true,
    owns_attention_indexed_mixed_surface: true,
    owns_generic_indexed_kernel: true,
    owns_topk_filter_and_order_semantics: true,
    owns_indexed_sort_or_heads8_dispatch: false,
    owns_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionIndexedPath {
    Heads8Online,
    Heads8Rb4,
    Generic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionIndexedDispatchOptions {
    pub n_tokens: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub no_indexed_heads8: bool,
    pub indexed_twopass: bool,
}

pub const fn select_attention_indexed_path(
    options: AttentionIndexedDispatchOptions,
) -> AttentionIndexedPath {
    if options.n_tokens > 1
        && options.head_dim == 512
        && options.top_k <= 512
        && !options.no_indexed_heads8
    {
        if options.indexed_twopass {
            AttentionIndexedPath::Heads8Rb4
        } else {
            AttentionIndexedPath::Heads8Online
        }
    } else {
        AttentionIndexedPath::Generic
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionIndexedOptimizedScope {
    pub opt_in_only: bool,
    pub consumes_indexed_topk_sort_policy: bool,
    pub owns_indexed_heads8_online_kernel: bool,
    pub owns_indexed_heads8_rb4_kernel: bool,
    pub owns_indexed_attention_dispatch_policy: bool,
    pub owns_output_q8_attention: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D7_SCOPE: AttentionIndexedOptimizedScope = AttentionIndexedOptimizedScope {
    opt_in_only: true,
    consumes_indexed_topk_sort_policy: true,
    owns_indexed_heads8_online_kernel: true,
    owns_indexed_heads8_rb4_kernel: true,
    owns_indexed_attention_dispatch_policy: true,
    owns_output_q8_attention: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionOutputQ8NativeScope {
    pub opt_in_only: bool,
    pub consumes_q8_conversion_and_matmul_kernels: bool,
    pub owns_attention_output_low_q8_surface: bool,
    pub owns_attention_output_q8_batch_native_surface: bool,
    pub owns_attention_output_a_cublas_dispatch: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D8A_SCOPE: AttentionOutputQ8NativeScope = AttentionOutputQ8NativeScope {
    opt_in_only: true,
    consumes_q8_conversion_and_matmul_kernels: true,
    owns_attention_output_low_q8_surface: true,
    owns_attention_output_q8_batch_native_surface: true,
    owns_attention_output_a_cublas_dispatch: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionOutputAPath {
    NativeQ8,
    CublasF16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionOutputADispatchOptions {
    pub quality_mode: bool,
    pub cublas_ready: bool,
    pub n_tokens: u32,
    pub cublas_min_tokens: u32,
    pub no_cublas_attention_output_a: bool,
    pub expanded_f16_ready: bool,
}

pub const fn select_attention_output_a_path(
    options: AttentionOutputADispatchOptions,
) -> AttentionOutputAPath {
    if !options.quality_mode
        && options.cublas_ready
        && options.n_tokens >= options.cublas_min_tokens
        && !options.no_cublas_attention_output_a
        && options.expanded_f16_ready
    {
        AttentionOutputAPath::CublasF16
    } else {
        AttentionOutputAPath::NativeQ8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionOutputQ8CublasScope {
    pub opt_in_only: bool,
    pub consumes_attention_output_q8_native_surface: bool,
    pub owns_attention_output_a_cublas_dispatch: bool,
    pub owns_attention_output_a_pack_unpack_kernels: bool,
    pub owns_live_cublas_grouped_a_pipeline: bool,
    pub uses_safe_sgemm_f16_rounded_adapter: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_4D8B_SCOPE: AttentionOutputQ8CublasScope = AttentionOutputQ8CublasScope {
    opt_in_only: true,
    consumes_attention_output_q8_native_surface: true,
    owns_attention_output_a_cublas_dispatch: true,
    owns_attention_output_a_pack_unpack_kernels: true,
    owns_live_cublas_grouped_a_pipeline: true,
    uses_safe_sgemm_f16_rounded_adapter: true,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterScalarScope {
    pub opt_in_only: bool,
    pub owns_router_select_kernel: bool,
    pub owns_scalar_single_and_batch_router_surface: bool,
    pub owns_bias_and_hash_router_semantics: bool,
    pub owns_parallel_or_warp_router_dispatch: bool,
    pub owns_routed_moe_or_hyperconnection: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_5A_SCOPE: RouterScalarScope = RouterScalarScope {
    opt_in_only: true,
    owns_router_select_kernel: true,
    owns_scalar_single_and_batch_router_surface: true,
    owns_bias_and_hash_router_semantics: true,
    owns_parallel_or_warp_router_dispatch: false,
    owns_routed_moe_or_hyperconnection: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouterSelectPath {
    WarpTopK,
    Parallel,
    Scalar,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterSelectDispatchOptions {
    pub no_warp_router_select: bool,
    pub no_parallel_router_select: bool,
}

pub const fn select_router_select_path(options: RouterSelectDispatchOptions) -> RouterSelectPath {
    if !options.no_warp_router_select && !options.no_parallel_router_select {
        RouterSelectPath::WarpTopK
    } else if !options.no_parallel_router_select {
        RouterSelectPath::Parallel
    } else {
        RouterSelectPath::Scalar
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterOptimizedScope {
    pub opt_in_only: bool,
    pub consumes_scalar_router_surface: bool,
    pub owns_router_select_parallel_kernel: bool,
    pub owns_router_select_warp_topk_kernel: bool,
    pub owns_parallel_and_warp_router_dispatch: bool,
    pub owns_current_c_dispatch_priority: bool,
    pub owns_routed_moe_or_hyperconnection: bool,
    pub owns_runtime_graph_integration: bool,
    pub changes_default_route: bool,
}

pub const M14_5B_SCOPE: RouterOptimizedScope = RouterOptimizedScope {
    opt_in_only: true,
    consumes_scalar_router_surface: true,
    owns_router_select_parallel_kernel: true,
    owns_router_select_warp_topk_kernel: true,
    owns_parallel_and_warp_router_dispatch: true,
    owns_current_c_dispatch_priority: true,
    owns_routed_moe_or_hyperconnection: false,
    owns_runtime_graph_integration: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeF32Scope {
    pub opt_in_only: bool,
    pub consumes_router_selection_surface: bool,
    pub owns_iq2_xxs_f32_gate_up_dot: bool,
    pub owns_q2_k_f32_down_dot: bool,
    pub owns_moe_gate_up_mid_f32_kernel: bool,
    pub owns_moe_down_f32_kernel: bool,
    pub owns_moe_sum_kernel: bool,
    pub owns_single_and_batch_f32_activation_moe_surface: bool,
    pub owns_q8_activation_or_optimized_moe_dispatch: bool,
    pub owns_hyperconnection_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C1_SCOPE: RoutedMoeF32Scope = RoutedMoeF32Scope {
    opt_in_only: true,
    consumes_router_selection_surface: true,
    owns_iq2_xxs_f32_gate_up_dot: true,
    owns_q2_k_f32_down_dot: true,
    owns_moe_gate_up_mid_f32_kernel: true,
    owns_moe_down_f32_kernel: true,
    owns_moe_sum_kernel: true,
    owns_single_and_batch_f32_activation_moe_surface: true,
    owns_q8_activation_or_optimized_moe_dispatch: false,
    owns_hyperconnection_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeQuantizedSingleScope {
    pub opt_in_only: bool,
    pub consumes_f32_fallback_surface: bool,
    pub owns_q8_k_activation_quantization: bool,
    pub owns_iq2_xxs_q8_k_gate_up_decode_lut: bool,
    pub owns_q2_k_q8_k_direct_sum6_down: bool,
    pub owns_default_single_token_iq2_q2_dispatch: bool,
    pub owns_optional_gate_up_aux_write: bool,
    pub owns_batched_sorted_or_tiled_dispatch: bool,
    pub owns_q4_k_dispatch: bool,
    pub owns_hyperconnection_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2A_SCOPE: RoutedMoeQuantizedSingleScope = RoutedMoeQuantizedSingleScope {
    opt_in_only: true,
    consumes_f32_fallback_surface: true,
    owns_q8_k_activation_quantization: true,
    owns_iq2_xxs_q8_k_gate_up_decode_lut: true,
    owns_q2_k_q8_k_direct_sum6_down: true,
    owns_default_single_token_iq2_q2_dispatch: true,
    owns_optional_gate_up_aux_write: true,
    owns_batched_sorted_or_tiled_dispatch: false,
    owns_q4_k_dispatch: false,
    owns_hyperconnection_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeQ4KSingleScope {
    pub opt_in_only: bool,
    pub consumes_quantized_single_surface: bool,
    pub owns_q4_k_q8_k_dot: bool,
    pub owns_moe_gate_up_mid_decode_q4_k_qwarp32_kernel: bool,
    pub owns_moe_down_q4_k_sum6_qwarp32_kernel: bool,
    pub owns_single_token_type12_dispatch: bool,
    pub owns_hyperconnection_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2D_SCOPE: RoutedMoeQ4KSingleScope = RoutedMoeQ4KSingleScope {
    opt_in_only: true,
    consumes_quantized_single_surface: true,
    owns_q4_k_q8_k_dot: true,
    owns_moe_gate_up_mid_decode_q4_k_qwarp32_kernel: true,
    owns_moe_down_q4_k_sum6_qwarp32_kernel: true,
    owns_single_token_type12_dispatch: true,
    owns_hyperconnection_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeSortedPairsScope {
    pub opt_in_only: bool,
    pub consumes_quantized_single_surface: bool,
    pub owns_moe_count_sorted_pairs_kernel: bool,
    pub owns_moe_prefix_sorted_pairs_kernel: bool,
    pub owns_moe_scatter_sorted_pairs_kernel: bool,
    pub owns_negative_expert_bucket_zero: bool,
    pub owns_sorted_pair_metadata: bool,
    pub owns_sorted_projection_kernels: bool,
    pub owns_expert_tile_or_atomic_down: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2B1_SCOPE: RoutedMoeSortedPairsScope = RoutedMoeSortedPairsScope {
    opt_in_only: true,
    consumes_quantized_single_surface: true,
    owns_moe_count_sorted_pairs_kernel: true,
    owns_moe_prefix_sorted_pairs_kernel: true,
    owns_moe_scatter_sorted_pairs_kernel: true,
    owns_negative_expert_bucket_zero: true,
    owns_sorted_pair_metadata: true,
    owns_sorted_projection_kernels: false,
    owns_expert_tile_or_atomic_down: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeSortedP2Scope {
    pub opt_in_only: bool,
    pub consumes_sorted_pair_metadata_surface: bool,
    pub uses_q8_k_activation_quantization: bool,
    pub owns_moe_gate_up_mid_sorted_p2_qwarp32_kernel: bool,
    pub owns_moe_down_sorted_p2_qwarp32_kernel: bool,
    pub owns_no_expert_tiles_p2_batch_dispatch: bool,
    pub uses_moe_sum_surface: bool,
    pub owns_expert_tile_or_atomic_down: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2B2_SCOPE: RoutedMoeSortedP2Scope = RoutedMoeSortedP2Scope {
    opt_in_only: true,
    consumes_sorted_pair_metadata_surface: true,
    uses_q8_k_activation_quantization: true,
    owns_moe_gate_up_mid_sorted_p2_qwarp32_kernel: true,
    owns_moe_down_sorted_p2_qwarp32_kernel: true,
    owns_no_expert_tiles_p2_batch_dispatch: true,
    uses_moe_sum_surface: true,
    owns_expert_tile_or_atomic_down: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeExpertTilesScope {
    pub opt_in_only: bool,
    pub consumes_sorted_pair_metadata_surface: bool,
    pub owns_moe_build_expert_tile_offsets_kernel: bool,
    pub owns_moe_build_expert_tiles_kernel: bool,
    pub owns_tile4_and_tile8_descriptor_metadata: bool,
    pub owns_tile_projection_kernels: bool,
    pub owns_atomic_down_or_rowspan_dispatch: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C1_SCOPE: RoutedMoeExpertTilesScope = RoutedMoeExpertTilesScope {
    opt_in_only: true,
    consumes_sorted_pair_metadata_surface: true,
    owns_moe_build_expert_tile_offsets_kernel: true,
    owns_moe_build_expert_tiles_kernel: true,
    owns_tile4_and_tile8_descriptor_metadata: true,
    owns_tile_projection_kernels: false,
    owns_atomic_down_or_rowspan_dispatch: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeTile8Row32Scope {
    pub opt_in_only: bool,
    pub consumes_expert_tile_metadata_surface: bool,
    pub uses_previously_owned_q8_k_inputs: bool,
    pub owns_moe_gate_up_mid_expert_tile8_row32_kernel: bool,
    pub owns_moe_down_expert_tile8_row32_non_atomic_surface: bool,
    pub owns_default_tile8_row32_projection_dispatch: bool,
    pub owns_atomic_down_or_rowspan_dispatch: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C2_SCOPE: RoutedMoeTile8Row32Scope = RoutedMoeTile8Row32Scope {
    opt_in_only: true,
    consumes_expert_tile_metadata_surface: true,
    uses_previously_owned_q8_k_inputs: true,
    owns_moe_gate_up_mid_expert_tile8_row32_kernel: true,
    owns_moe_down_expert_tile8_row32_non_atomic_surface: true,
    owns_default_tile8_row32_projection_dispatch: true,
    owns_atomic_down_or_rowspan_dispatch: false,
    owns_shared_cache_specialization: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeTile4Row32Scope {
    pub opt_in_only: bool,
    pub consumes_expert_tile_metadata_surface: bool,
    pub uses_previously_owned_q8_k_inputs: bool,
    pub owns_moe_gate_up_mid_expert_tile4_row32_kernel: bool,
    pub owns_moe_down_expert_tile4_row32_non_atomic_surface: bool,
    pub owns_optional_tile4_row32_projection_dispatch: bool,
    pub owns_atomic_down_or_rowspan_dispatch: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C3_SCOPE: RoutedMoeTile4Row32Scope = RoutedMoeTile4Row32Scope {
    opt_in_only: true,
    consumes_expert_tile_metadata_surface: true,
    uses_previously_owned_q8_k_inputs: true,
    owns_moe_gate_up_mid_expert_tile4_row32_kernel: true,
    owns_moe_down_expert_tile4_row32_non_atomic_surface: true,
    owns_optional_tile4_row32_projection_dispatch: true,
    owns_atomic_down_or_rowspan_dispatch: false,
    owns_shared_cache_specialization: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeAtomicDownScope {
    pub opt_in_only: bool,
    pub consumes_tile_row32_projection_surface: bool,
    pub owns_device_atomic_f32_fetch_add: bool,
    pub owns_zero_kernel_for_atomic_down: bool,
    pub owns_tile4_and_tile8_row32_atomic_down_dispatch: bool,
    pub owns_tile16_or_rowspan_dispatch: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C4_SCOPE: RoutedMoeAtomicDownScope = RoutedMoeAtomicDownScope {
    opt_in_only: true,
    consumes_tile_row32_projection_surface: true,
    owns_device_atomic_f32_fetch_add: true,
    owns_zero_kernel_for_atomic_down: true,
    owns_tile4_and_tile8_row32_atomic_down_dispatch: true,
    owns_tile16_or_rowspan_dispatch: false,
    owns_shared_cache_specialization: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeTile16Row32Scope {
    pub opt_in_only: bool,
    pub consumes_atomic_row32_surface: bool,
    pub owns_moe_down_expert_tile16_row32_kernel: bool,
    pub owns_tile16_atomic_down_dispatch: bool,
    pub owns_rowspan_dispatch: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C5_SCOPE: RoutedMoeTile16Row32Scope = RoutedMoeTile16Row32Scope {
    opt_in_only: true,
    consumes_atomic_row32_surface: true,
    owns_moe_down_expert_tile16_row32_kernel: true,
    owns_tile16_atomic_down_dispatch: true,
    owns_rowspan_dispatch: false,
    owns_shared_cache_specialization: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeGateRowspanScope {
    pub opt_in_only: bool,
    pub consumes_tile8_row32_projection_surface: bool,
    pub owns_moe_gate_up_mid_expert_tile8_rowspan_kernel: bool,
    pub owns_gate_row512_row1024_and_row2048_dispatch: bool,
    pub owns_down_rowspan_dispatch: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C6_SCOPE: RoutedMoeGateRowspanScope = RoutedMoeGateRowspanScope {
    opt_in_only: true,
    consumes_tile8_row32_projection_surface: true,
    owns_moe_gate_up_mid_expert_tile8_rowspan_kernel: true,
    owns_gate_row512_row1024_and_row2048_dispatch: true,
    owns_down_rowspan_dispatch: false,
    owns_shared_cache_specialization: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeDownRowspanScope {
    pub opt_in_only: bool,
    pub consumes_tile16_row32_atomic_surface: bool,
    pub owns_moe_down_expert_tile16_rowspan_kernel: bool,
    pub owns_down_row512_row1024_and_row2048_atomic_dispatch: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_q4_k_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2C7_SCOPE: RoutedMoeDownRowspanScope = RoutedMoeDownRowspanScope {
    opt_in_only: true,
    consumes_tile16_row32_atomic_surface: true,
    owns_moe_down_expert_tile16_rowspan_kernel: true,
    owns_down_row512_row1024_and_row2048_atomic_dispatch: true,
    owns_shared_cache_specialization: false,
    owns_q4_k_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeSharedCacheScope {
    pub opt_in_only: bool,
    pub consumes_rowspan_projection_surface: bool,
    pub owns_shared_cache_specialization: bool,
    pub owns_gate_and_down_cached_rowspan_dispatch: bool,
    pub owns_hyperconnection_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2E_SCOPE: RoutedMoeSharedCacheScope = RoutedMoeSharedCacheScope {
    opt_in_only: true,
    consumes_rowspan_projection_surface: true,
    owns_shared_cache_specialization: true,
    owns_gate_and_down_cached_rowspan_dispatch: true,
    owns_hyperconnection_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedMoeQwarpFallbackScope {
    pub opt_in_only: bool,
    pub uses_q8_k_activation_quantization: bool,
    pub owns_moe_gate_up_mid_qwarp32_kernel: bool,
    pub owns_moe_down_qwarp32_kernel: bool,
    pub owns_moe_gate_up_mid_sorted_qwarp32_kernel: bool,
    pub owns_moe_down_sorted_qwarp32_kernel: bool,
    pub owns_no_decode_lut_generic_dispatch: bool,
    pub owns_no_p2_sorted_batch_dispatch: bool,
    pub uses_moe_sum_surface: bool,
    pub owns_hyperconnection_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5C2F_SCOPE: RoutedMoeQwarpFallbackScope = RoutedMoeQwarpFallbackScope {
    opt_in_only: true,
    uses_q8_k_activation_quantization: true,
    owns_moe_gate_up_mid_qwarp32_kernel: true,
    owns_moe_down_qwarp32_kernel: true,
    owns_moe_gate_up_mid_sorted_qwarp32_kernel: true,
    owns_moe_down_sorted_qwarp32_kernel: true,
    owns_no_decode_lut_generic_dispatch: true,
    owns_no_p2_sorted_batch_dispatch: true,
    uses_moe_sum_surface: true,
    owns_hyperconnection_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HyperconnectionScope {
    pub opt_in_only: bool,
    pub owns_hc_split_sinkhorn_kernel: bool,
    pub owns_hc_weighted_sum_kernel: bool,
    pub owns_hc_expand_kernel: bool,
    pub owns_hc_split_weighted_sum_fused_kernel: bool,
    pub owns_hc_split_weighted_sum_norm_fused_kernel: bool,
    pub owns_output_hc_weights_kernel: bool,
    pub owns_shared_expert_wrapper_or_runtime_graph: bool,
    pub changes_default_route: bool,
}

pub const M14_5D_SCOPE: HyperconnectionScope = HyperconnectionScope {
    opt_in_only: true,
    owns_hc_split_sinkhorn_kernel: true,
    owns_hc_weighted_sum_kernel: true,
    owns_hc_expand_kernel: true,
    owns_hc_split_weighted_sum_fused_kernel: true,
    owns_hc_split_weighted_sum_norm_fused_kernel: true,
    owns_output_hc_weights_kernel: true,
    owns_shared_expert_wrapper_or_runtime_graph: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaRoutePromotionGate {
    pub operation_families_validated: bool,
    pub production_build_still_compiles_ds4_cuda_cu: bool,
    pub rust_exports_ds4_gpu_abi: bool,
    pub runtime_graph_route_implemented: bool,
    pub can_promote_default_route: bool,
    pub can_remove_c_cuda: bool,
}

pub const M14_6A_GATE: CudaRoutePromotionGate = CudaRoutePromotionGate {
    operation_families_validated: true,
    production_build_still_compiles_ds4_cuda_cu: true,
    rust_exports_ds4_gpu_abi: false,
    runtime_graph_route_implemented: false,
    can_promote_default_route: false,
    can_remove_c_cuda: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaAbiResourceScope {
    pub exported_resource_symbol_count: u32,
    pub owns_initialization: bool,
    pub owns_tensor_storage: bool,
    pub owns_host_device_copies: bool,
    pub owns_command_synchronization: bool,
    pub owns_managed_kv_policy: bool,
    pub owns_tensor_fill_kernel: bool,
    pub owns_compute_abi: bool,
    pub owns_complete_ds4_gpu_abi: bool,
    pub changes_default_route: bool,
}

pub const M14_6B1_SCOPE: CudaAbiResourceScope = CudaAbiResourceScope {
    exported_resource_symbol_count: 16,
    owns_initialization: true,
    owns_tensor_storage: true,
    owns_host_device_copies: true,
    owns_command_synchronization: true,
    owns_managed_kv_policy: true,
    owns_tensor_fill_kernel: false,
    owns_compute_abi: false,
    owns_complete_ds4_gpu_abi: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaAbiTensorFillScope {
    pub exported_abi_symbol_count: u32,
    pub exported_compute_symbol_count: u32,
    pub owns_tensor_fill_f32: bool,
    pub owns_graph_compute_abi: bool,
    pub owns_complete_ds4_gpu_abi: bool,
    pub changes_default_route: bool,
}

pub const M14_6B2A_SCOPE: CudaAbiTensorFillScope = CudaAbiTensorFillScope {
    exported_abi_symbol_count: 17,
    exported_compute_symbol_count: 1,
    owns_tensor_fill_f32: true,
    owns_graph_compute_abi: false,
    owns_complete_ds4_gpu_abi: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaAbiElementwiseScope {
    pub exported_abi_symbol_count: u32,
    pub exported_compute_symbol_count: u32,
    pub owns_add_tensor: bool,
    pub owns_repeat_hc_tensor: bool,
    pub uses_embedded_rust_kernel_module: bool,
    pub owns_remaining_graph_compute_abi: bool,
    pub owns_complete_ds4_gpu_abi: bool,
    pub changes_default_route: bool,
}

pub const M14_6B2B1_SCOPE: CudaAbiElementwiseScope = CudaAbiElementwiseScope {
    exported_abi_symbol_count: 19,
    exported_compute_symbol_count: 3,
    owns_add_tensor: true,
    owns_repeat_hc_tensor: true,
    uses_embedded_rust_kernel_module: true,
    owns_remaining_graph_compute_abi: false,
    owns_complete_ds4_gpu_abi: false,
    changes_default_route: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaAbiDirectionalSteeringScope {
    pub exported_abi_symbol_count: u32,
    pub exported_compute_symbol_count: u32,
    pub owns_directional_steering_project_tensor: bool,
    pub owns_remaining_graph_compute_abi: bool,
    pub owns_complete_ds4_gpu_abi: bool,
    pub changes_default_route: bool,
}

pub const M14_6B2B2A_SCOPE: CudaAbiDirectionalSteeringScope = CudaAbiDirectionalSteeringScope {
    exported_abi_symbol_count: 20,
    exported_compute_symbol_count: 4,
    owns_directional_steering_project_tensor: true,
    owns_remaining_graph_compute_abi: false,
    owns_complete_ds4_gpu_abi: false,
    changes_default_route: false,
};

pub mod allocation_policy;
pub mod q8_policy;

#[cfg(feature = "cuda-oxide-backend")]
pub mod abi;

#[cfg(feature = "cuda-oxide-kernels")]
mod abi_kernels;

#[cfg(feature = "cuda-oxide-backend")]
pub mod model_map;

#[cfg(feature = "cuda-oxide-backend")]
pub mod substrate;

#[cfg(test)]
mod tests {
    use super::{
        q8_dp4a_enabled, select_attention_decode_path, select_attention_indexed_path,
        select_attention_output_a_path, select_attention_prefill_path,
        select_f16_pair_projection_path, select_f16_projection_path, select_f32_projection_path,
        select_indexer_score_kernel, select_indexer_topk_kernel, select_q8_matmul_path,
        select_router_select_path, should_sort_indexed_topk, AttentionDecodeDispatchOptions,
        AttentionDecodePath, AttentionIndexedDispatchOptions, AttentionIndexedPath,
        AttentionOutputADispatchOptions, AttentionOutputAPath, AttentionPrefillDispatchOptions,
        AttentionPrefillPath, F16PairProjectionDispatch, F16PairProjectionPath,
        F16ProjectionDispatch, F16ProjectionPath, F32ProjectionPath, IndexedTopkSortOptions,
        IndexerScoreDispatchOptions, IndexerScoreKernel, IndexerTopkDispatchOptions,
        IndexerTopkKernel, Q8MatmulDispatchOptions, Q8MatmulPath, RouterSelectDispatchOptions,
        RouterSelectPath, CUDA_OXIDE_REVISION, M14_1A_SCOPE, M14_1B1_SCOPE, M14_1B2A_SCOPE,
        M14_1B2B1_SCOPE, M14_1B2B2_SCOPE, M14_1B2B3A_SCOPE, M14_1B2B3B1_SCOPE, M14_1B2B3B2_SCOPE,
        M14_1B2C_SCOPE, M14_1B3A_SCOPE, M14_1B3B_SCOPE, M14_1B4_SCOPE, M14_2A_SCOPE, M14_2B1_SCOPE,
        M14_2B2_SCOPE, M14_2C_SCOPE, M14_2D1_SCOPE, M14_2D2A_SCOPE, M14_2D2B1_SCOPE,
        M14_2D2B2A_SCOPE, M14_2D2B2B_SCOPE, M14_2D2B2C_SCOPE, M14_2D2C1_SCOPE, M14_2D2C2_SCOPE,
        M14_2D2C3_SCOPE, M14_2D2C4_SCOPE, M14_2D2C5_SCOPE, M14_3A_SCOPE, M14_3B1_SCOPE,
        M14_3B2_SCOPE, M14_3C1_SCOPE, M14_3C2_SCOPE, M14_3C3_SCOPE, M14_3D1_SCOPE, M14_3D2_SCOPE,
        M14_3D3_SCOPE, M14_3D4_SCOPE, M14_4A_SCOPE, M14_4B_SCOPE, M14_4C1_SCOPE, M14_4C2_SCOPE,
        M14_4C3A_SCOPE, M14_4C3B_SCOPE, M14_4D1_SCOPE, M14_4D2_SCOPE, M14_4D3_SCOPE, M14_4D4_SCOPE,
        M14_4D5_SCOPE, M14_4D6_SCOPE, M14_4D7_SCOPE, M14_4D8A_SCOPE, M14_4D8B_SCOPE, M14_5A_SCOPE,
        M14_5B_SCOPE, M14_5C1_SCOPE, M14_5C2A_SCOPE, M14_5C2B1_SCOPE, M14_5C2B2_SCOPE,
        M14_5C2C1_SCOPE, M14_5C2C2_SCOPE, M14_5C2C3_SCOPE, M14_5C2C4_SCOPE, M14_5C2C5_SCOPE,
        M14_5C2C6_SCOPE, M14_5C2C7_SCOPE, M14_5C2D_SCOPE, M14_5C2E_SCOPE, M14_5C2F_SCOPE,
        M14_5D_SCOPE, M14_6A_GATE, M14_6B1_SCOPE, M14_6B2A_SCOPE, M14_6B2B1_SCOPE,
        M14_6B2B2A_SCOPE,
    };

    #[test]
    fn substrate_scope_does_not_overclaim_kernel_or_route_ownership() {
        assert_eq!(
            CUDA_OXIDE_REVISION,
            "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7"
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

    #[test]
    fn direct_indexer_scope_leaves_tensor_core_topk_and_route_pending() {
        assert!(M14_2D2A_SCOPE.opt_in_only);
        assert!(M14_2D2A_SCOPE.owns_indexer_score_one_direct_kernel);
        assert!(!M14_2D2A_SCOPE.owns_wmma_indexer_dispatch);
        assert!(!M14_2D2A_SCOPE.owns_specialized_topk_dispatch);
        assert!(!M14_2D2A_SCOPE.changes_default_route);
    }

    #[test]
    fn base_wmma_scope_leaves_widened_dispatch_topk_and_route_pending() {
        assert!(M14_2D2B1_SCOPE.opt_in_only);
        assert!(M14_2D2B1_SCOPE.owns_indexer_scores_wmma_kernel);
        assert!(!M14_2D2B1_SCOPE.owns_widened_wmma_dispatch);
        assert!(!M14_2D2B1_SCOPE.owns_specialized_topk_dispatch);
        assert!(!M14_2D2B1_SCOPE.changes_default_route);
    }

    #[test]
    fn wmma32_scope_leaves_larger_dispatch_topk_and_route_pending() {
        assert!(M14_2D2B2A_SCOPE.opt_in_only);
        assert!(M14_2D2B2A_SCOPE.owns_indexer_scores_wmma32_kernel);
        assert!(!M14_2D2B2A_SCOPE.owns_wmma64_and_wmma128_dispatch);
        assert!(!M14_2D2B2A_SCOPE.owns_specialized_topk_dispatch);
        assert!(!M14_2D2B2A_SCOPE.changes_default_route);
    }

    #[test]
    fn wmma64_scope_leaves_wmma128_priority_topk_and_route_pending() {
        assert!(M14_2D2B2B_SCOPE.opt_in_only);
        assert!(M14_2D2B2B_SCOPE.owns_indexer_scores_wmma64_kernel);
        assert!(!M14_2D2B2B_SCOPE.owns_wmma128_and_dispatch_priority);
        assert!(!M14_2D2B2B_SCOPE.owns_specialized_topk_dispatch);
        assert!(!M14_2D2B2B_SCOPE.changes_default_route);
    }

    #[test]
    fn wmma128_scope_owns_score_dispatch_but_leaves_topk_and_route_pending() {
        assert!(M14_2D2B2C_SCOPE.opt_in_only);
        assert!(M14_2D2B2C_SCOPE.owns_indexer_scores_wmma128_kernel);
        assert!(M14_2D2B2C_SCOPE.owns_indexer_score_dispatch_policy);
        assert!(!M14_2D2B2C_SCOPE.owns_specialized_topk_dispatch);
        assert!(!M14_2D2B2C_SCOPE.changes_default_route);
    }

    #[test]
    fn topk1024_scope_leaves_larger_indexed_dispatch_and_route_pending() {
        assert!(M14_2D2C1_SCOPE.opt_in_only);
        assert!(M14_2D2C1_SCOPE.owns_indexer_topk_1024_kernel);
        assert!(!M14_2D2C1_SCOPE.owns_larger_topk_dispatch);
        assert!(!M14_2D2C1_SCOPE.owns_indexed_topk_sort_dispatch);
        assert!(!M14_2D2C1_SCOPE.changes_default_route);
    }

    #[test]
    fn topk_pow2_scope_leaves_cub_chunked_indexed_dispatch_and_route_pending() {
        assert!(M14_2D2C2_SCOPE.opt_in_only);
        assert!(M14_2D2C2_SCOPE.owns_indexer_topk_pow2_2048_kernel);
        assert!(M14_2D2C2_SCOPE.owns_indexer_topk_pow2_4096_kernel);
        assert!(M14_2D2C2_SCOPE.owns_indexer_topk_pow2_u16_8192_kernel);
        assert!(!M14_2D2C2_SCOPE.owns_cub_topk_dispatch);
        assert!(!M14_2D2C2_SCOPE.owns_chunked_topk_dispatch);
        assert!(!M14_2D2C2_SCOPE.owns_indexed_topk_sort_dispatch);
        assert!(!M14_2D2C2_SCOPE.changes_default_route);
    }

    #[test]
    fn packed_topk_scope_leaves_cub_internals_dispatch_chunking_and_route_pending() {
        assert!(M14_2D2C3_SCOPE.opt_in_only);
        assert!(M14_2D2C3_SCOPE.owns_indexer_topk_8192_packed_key_equivalent_kernel);
        assert!(M14_2D2C3_SCOPE.owns_dynamic_shared_launch_shape);
        assert!(!M14_2D2C3_SCOPE.owns_cub_library_implementation);
        assert!(!M14_2D2C3_SCOPE.owns_topk_dispatch_policy);
        assert!(!M14_2D2C3_SCOPE.owns_chunked_topk_dispatch);
        assert!(!M14_2D2C3_SCOPE.changes_default_route);
    }

    #[test]
    fn tree_topk_scope_leaves_dispatch_indexed_sort_and_route_pending() {
        assert!(M14_2D2C4_SCOPE.opt_in_only);
        assert!(M14_2D2C4_SCOPE.owns_indexer_topk_chunk_pow2_4096_kernel);
        assert!(M14_2D2C4_SCOPE.owns_indexer_topk_tree_merge_pow2_4096_kernel);
        assert!(M14_2D2C4_SCOPE.owns_indexer_topk_merge_pow2_4096_kernel);
        assert!(M14_2D2C4_SCOPE.owns_scratch_layout);
        assert!(!M14_2D2C4_SCOPE.owns_topk_dispatch_policy);
        assert!(!M14_2D2C4_SCOPE.owns_indexed_topk_sort_dispatch);
        assert!(!M14_2D2C4_SCOPE.changes_default_route);
    }

    #[test]
    fn specialized_topk_dispatch_scope_closes_opt_in_policy_not_cub_or_route() {
        assert!(M14_2D2C5_SCOPE.opt_in_only);
        assert!(M14_2D2C5_SCOPE.owns_indexed_topk_sort_512_asc_kernel);
        assert!(M14_2D2C5_SCOPE.owns_indexed_topk_sort_dispatch);
        assert!(M14_2D2C5_SCOPE.owns_topk_dispatch_policy);
        assert!(M14_2D2C5_SCOPE.uses_packed_key_equivalent_branch);
        assert!(!M14_2D2C5_SCOPE.owns_cub_library_implementation);
        assert!(!M14_2D2C5_SCOPE.changes_default_route);
    }

    #[test]
    fn rms_norm_scope_leaves_fused_norm_projection_q8_and_route_pending() {
        assert!(M14_3A_SCOPE.opt_in_only);
        assert!(M14_3A_SCOPE.owns_rms_norm_plain_kernel);
        assert!(M14_3A_SCOPE.owns_rms_norm_weight_kernel);
        assert!(M14_3A_SCOPE.owns_plain_and_weighted_tensor_surface);
        assert!(!M14_3A_SCOPE.owns_fused_qkv_and_head_norm_kernels);
        assert!(!M14_3A_SCOPE.owns_dense_projection_or_q8_kernels);
        assert!(!M14_3A_SCOPE.changes_default_route);
    }

    #[test]
    fn fused_rms_norm_scope_leaves_rope_dispatch_projection_q8_and_route_pending() {
        assert!(M14_3B1_SCOPE.opt_in_only);
        assert!(M14_3B1_SCOPE.owns_dsv4_qkv_rms_norm_rows_kernel);
        assert!(M14_3B1_SCOPE.owns_head_rms_norm_kernel);
        assert!(!M14_3B1_SCOPE.owns_head_rms_norm_rope_tail_kernel);
        assert!(!M14_3B1_SCOPE.owns_qkv_fused_dispatch_policy);
        assert!(!M14_3B1_SCOPE.owns_dense_projection_or_q8_kernels);
        assert!(!M14_3B1_SCOPE.changes_default_route);
    }

    #[test]
    fn head_rms_rope_tail_scope_leaves_standalone_rope_projection_q8_and_route_pending() {
        assert!(M14_3B2_SCOPE.opt_in_only);
        assert!(M14_3B2_SCOPE.owns_head_rms_norm_rope_tail_kernel);
        assert!(M14_3B2_SCOPE.owns_yarn_rotary_math_path);
        assert!(!M14_3B2_SCOPE.owns_standalone_rope_tail_kernel);
        assert!(!M14_3B2_SCOPE.owns_qkv_fused_dispatch_policy);
        assert!(!M14_3B2_SCOPE.owns_dense_projection_or_q8_kernels);
        assert!(!M14_3B2_SCOPE.changes_default_route);
    }

    #[test]
    fn dense_projection_base_scope_leaves_ordered_blas_q8_and_route_pending() {
        assert!(M14_3C1_SCOPE.opt_in_only);
        assert!(M14_3C1_SCOPE.owns_matmul_f16_kernel);
        assert!(M14_3C1_SCOPE.owns_matmul_f32_kernel);
        assert!(!M14_3C1_SCOPE.owns_ordered_or_pair_f16_kernels);
        assert!(!M14_3C1_SCOPE.owns_f16_or_cublas_dispatch_policy);
        assert!(!M14_3C1_SCOPE.owns_q8_conversion_or_matmul_kernels);
        assert!(!M14_3C1_SCOPE.changes_default_route);
    }

    #[test]
    fn ordered_projection_scope_leaves_blas_q8_and_route_pending() {
        assert!(M14_3C2_SCOPE.opt_in_only);
        assert!(M14_3C2_SCOPE.owns_matmul_f16_serial_kernel);
        assert!(M14_3C2_SCOPE.owns_matmul_f16_ordered_chunks_kernel);
        assert!(M14_3C2_SCOPE.owns_matmul_f16_pair_ordered_chunks_kernel);
        assert!(!M14_3C2_SCOPE.owns_f16_or_cublas_dispatch_policy);
        assert!(!M14_3C2_SCOPE.owns_q8_conversion_or_matmul_kernels);
        assert!(!M14_3C2_SCOPE.changes_default_route);
    }

    #[test]
    fn blas_projection_scope_owns_dense_dispatch_without_q8_or_route_claims() {
        assert!(M14_3C3_SCOPE.opt_in_only);
        assert!(M14_3C3_SCOPE.owns_f32_to_f16_kernel);
        assert!(M14_3C3_SCOPE.owns_f16_projection_dispatch_policy);
        assert!(M14_3C3_SCOPE.owns_f16_pair_projection_dispatch_policy);
        assert!(M14_3C3_SCOPE.owns_f32_projection_dispatch_policy);
        assert!(M14_3C3_SCOPE.owns_live_f16_and_f32_blas_paths);
        assert!(!M14_3C3_SCOPE.owns_q8_conversion_or_matmul_kernels);
        assert!(!M14_3C3_SCOPE.changes_default_route);
    }

    #[test]
    fn q8_conversion_scope_leaves_quantized_matmul_and_route_pending() {
        assert!(M14_3D1_SCOPE.opt_in_only);
        assert!(M14_3D1_SCOPE.owns_dequant_q8_0_to_f16_kernel);
        assert!(M14_3D1_SCOPE.owns_dequant_q8_0_to_f32_kernel);
        assert!(M14_3D1_SCOPE.owns_quantize_q8_0_f32_kernel);
        assert!(!M14_3D1_SCOPE.owns_quantized_matmul_kernels);
        assert!(!M14_3D1_SCOPE.owns_q8_matmul_dispatch_policy);
        assert!(!M14_3D1_SCOPE.changes_default_route);
    }

    #[test]
    fn q8_matmul_scope_leaves_acceleration_specialization_and_route_pending() {
        assert!(M14_3D2_SCOPE.opt_in_only);
        assert!(M14_3D2_SCOPE.owns_matmul_q8_0_kernel);
        assert!(M14_3D2_SCOPE.owns_matmul_q8_0_preq_kernel);
        assert!(M14_3D2_SCOPE.owns_matmul_q8_0_preq_warp8_kernel);
        assert!(M14_3D2_SCOPE.owns_matmul_q8_0_preq_batch_warp8_kernel);
        assert!(!M14_3D2_SCOPE.owns_dp4a_acceleration);
        assert!(!M14_3D2_SCOPE.owns_pair_or_hc_expand_kernels);
        assert!(!M14_3D2_SCOPE.owns_q8_matmul_dispatch_policy);
        assert!(!M14_3D2_SCOPE.changes_default_route);
    }

    #[test]
    fn q8_specialized_matmul_scope_leaves_acceleration_dispatch_and_route_pending() {
        assert!(M14_3D3_SCOPE.opt_in_only);
        assert!(M14_3D3_SCOPE.owns_matmul_q8_0_pair_preq_warp8_kernel);
        assert!(M14_3D3_SCOPE.owns_matmul_q8_0_hc_expand_preq_warp8_kernel);
        assert!(M14_3D3_SCOPE.owns_hc_expand_optional_block_add);
        assert!(!M14_3D3_SCOPE.owns_dp4a_acceleration);
        assert!(!M14_3D3_SCOPE.owns_q8_matmul_dispatch_policy);
        assert!(!M14_3D3_SCOPE.changes_default_route);
    }

    #[test]
    fn q8_dp4a_dispatch_scope_leaves_runtime_route_pending() {
        assert!(M14_3D4_SCOPE.opt_in_only);
        assert!(M14_3D4_SCOPE.owns_cuda_oxide_dp4a_i8_intrinsic);
        assert!(M14_3D4_SCOPE.owns_dp4a_acceleration);
        assert!(M14_3D4_SCOPE.owns_q8_matmul_dispatch_policy);
        assert!(!M14_3D4_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_3D4_SCOPE.changes_default_route);
    }

    #[test]
    fn rope_kv_quantization_scope_leaves_storage_attention_and_route_pending() {
        assert!(M14_4A_SCOPE.opt_in_only);
        assert!(M14_4A_SCOPE.owns_standalone_rope_tail_kernel);
        assert!(M14_4A_SCOPE.owns_fp8_kv_quantize_kernel);
        assert!(M14_4A_SCOPE.owns_yarn_rotary_math_path);
        assert!(!M14_4A_SCOPE.owns_kv_storage_or_compressor_kernels);
        assert!(!M14_4A_SCOPE.owns_attention_kernels);
        assert!(!M14_4A_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4A_SCOPE.changes_default_route);
    }

    #[test]
    fn raw_kv_indexer_qat_scope_leaves_composition_attention_and_route_pending() {
        assert!(M14_4B_SCOPE.opt_in_only);
        assert!(M14_4B_SCOPE.owns_store_raw_kv_batch_kernel);
        assert!(M14_4B_SCOPE.owns_raw_kv_store_surfaces);
        assert!(M14_4B_SCOPE.owns_indexer_hadamard_fp4_kernel);
        assert!(M14_4B_SCOPE.owns_indexer_qat_surface);
        assert!(!M14_4B_SCOPE.owns_kv_fp8_store_raw_composition);
        assert!(!M14_4B_SCOPE.owns_compressor_kernels);
        assert!(!M14_4B_SCOPE.owns_attention_kernels);
        assert!(!M14_4B_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4B_SCOPE.changes_default_route);
    }

    #[test]
    fn composed_kv_compressor_store_scope_leaves_pooling_attention_and_route_pending() {
        assert!(M14_4C1_SCOPE.opt_in_only);
        assert!(M14_4C1_SCOPE.owns_kv_fp8_store_raw_composition);
        assert!(M14_4C1_SCOPE.owns_compressor_store_kernel);
        assert!(M14_4C1_SCOPE.owns_compressor_set_rows_kernel);
        assert!(M14_4C1_SCOPE.owns_f32_and_f16_ape_reads);
        assert!(!M14_4C1_SCOPE.owns_compressor_pooling_or_shift);
        assert!(!M14_4C1_SCOPE.owns_attention_kernels);
        assert!(!M14_4C1_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4C1_SCOPE.changes_default_route);
    }

    #[test]
    fn compressor_pool_shift_scope_leaves_wrappers_attention_and_route_pending() {
        assert!(M14_4C2_SCOPE.opt_in_only);
        assert!(M14_4C2_SCOPE.owns_compressor_prefill_pool_kernel);
        assert!(M14_4C2_SCOPE.owns_general_and_ratio4_prefill_branches);
        assert!(M14_4C2_SCOPE.owns_ratio4_replay_branch);
        assert!(M14_4C2_SCOPE.owns_compressor_update_pool_kernel);
        assert!(M14_4C2_SCOPE.owns_compressor_shift_ratio4_kernel);
        assert!(!M14_4C2_SCOPE.owns_compressor_wrapper_orchestration);
        assert!(!M14_4C2_SCOPE.owns_attention_kernels);
        assert!(!M14_4C2_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4C2_SCOPE.changes_default_route);
    }

    #[test]
    fn compressor_update_scope_leaves_prefill_attention_and_route_pending() {
        assert!(M14_4C3A_SCOPE.opt_in_only);
        assert!(M14_4C3A_SCOPE.owns_compressor_update_orchestration);
        assert!(M14_4C3A_SCOPE.owns_store_pool_norm_rope_shift_sequence);
        assert!(!M14_4C3A_SCOPE.owns_compressor_prefill_orchestration);
        assert!(!M14_4C3A_SCOPE.owns_attention_kernels);
        assert!(!M14_4C3A_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4C3A_SCOPE.changes_default_route);
    }

    #[test]
    fn compressor_prefill_scope_leaves_attention_and_route_pending() {
        assert!(M14_4C3B_SCOPE.opt_in_only);
        assert!(M14_4C3B_SCOPE.owns_compressor_prefill_orchestration);
        assert!(M14_4C3B_SCOPE.owns_ratio4_replay_orchestration);
        assert!(M14_4C3B_SCOPE.owns_ratio4_state_only_orchestration);
        assert!(M14_4C3B_SCOPE.owns_optional_fp8_compressed_output);
        assert!(!M14_4C3B_SCOPE.owns_attention_kernels);
        assert!(!M14_4C3B_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4C3B_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_single_decode_scope_leaves_batched_and_other_attention_pending() {
        assert!(M14_4D1_SCOPE.opt_in_only);
        assert!(M14_4D1_SCOPE.owns_attention_decode_heads_surface);
        assert!(M14_4D1_SCOPE.owns_single_token_ring_raw_and_compressed_attention);
        assert!(M14_4D1_SCOPE.owns_masked_compressed_rows);
        assert!(!M14_4D1_SCOPE.owns_batched_or_online_decode);
        assert!(!M14_4D1_SCOPE.owns_prefill_indexed_or_output_q8_attention);
        assert!(!M14_4D1_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D1_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_batch_decode_scope_leaves_online_and_other_attention_pending() {
        assert!(M14_4D2_SCOPE.opt_in_only);
        assert!(M14_4D2_SCOPE.owns_attention_decode_raw_batch_surface);
        assert!(M14_4D2_SCOPE.owns_attention_decode_mixed_batch_surface);
        assert!(M14_4D2_SCOPE.owns_causal_window_and_visible_compressed_rows);
        assert!(!M14_4D2_SCOPE.owns_heads8_online_decode);
        assert!(!M14_4D2_SCOPE.owns_prefill_indexed_or_output_q8_attention);
        assert!(!M14_4D2_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D2_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_heads8_online_scope_leaves_other_attention_and_route_pending() {
        assert!(M14_4D3_SCOPE.opt_in_only);
        assert!(M14_4D3_SCOPE.owns_heads8_online_decode_kernel);
        assert!(M14_4D3_SCOPE.owns_decode_online_dispatch_policy);
        assert!(!M14_4D3_SCOPE.owns_prefill_or_indexed_online_attention);
        assert!(!M14_4D3_SCOPE.owns_output_q8_attention);
        assert!(!M14_4D3_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D3_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_decode_online_dispatch_paths_match_current_c_priority() {
        let base = AttentionDecodeDispatchOptions {
            n_tokens: 3,
            n_comp: 4,
            use_comp_mask: false,
            head_dim: 512,
            no_window_attention: false,
            window_attention: true,
            quality_mode: false,
        };
        assert_eq!(
            select_attention_decode_path(base),
            AttentionDecodePath::Heads8OnlineWindow
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                n_tokens: 128,
                window_attention: false,
                ..base
            }),
            AttentionDecodePath::Heads8OnlineWindow
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                n_comp: 7937,
                n_tokens: 1,
                window_attention: false,
                quality_mode: true,
                ..base
            }),
            AttentionDecodePath::Heads8OnlineOverflow
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                use_comp_mask: true,
                ..base
            }),
            AttentionDecodePath::Generic
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                n_comp: 7937,
                use_comp_mask: true,
                ..base
            }),
            AttentionDecodePath::RejectScoreBuffer
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                no_window_attention: true,
                ..base
            }),
            AttentionDecodePath::Generic
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                n_tokens: 127,
                window_attention: false,
                quality_mode: false,
                ..base
            }),
            AttentionDecodePath::Generic
        );
        assert_eq!(
            select_attention_decode_path(AttentionDecodeDispatchOptions {
                n_tokens: 128,
                window_attention: false,
                quality_mode: true,
                ..base
            }),
            AttentionDecodePath::Generic
        );
    }

    #[test]
    fn attention_generic_prefill_scope_leaves_optimized_and_other_attention_pending() {
        assert!(M14_4D4_SCOPE.opt_in_only);
        assert!(M14_4D4_SCOPE.owns_attention_prefill_raw_surface);
        assert!(M14_4D4_SCOPE.owns_attention_prefill_static_mixed_surface);
        assert!(M14_4D4_SCOPE.owns_attention_prefill_masked_mixed_surface);
        assert!(M14_4D4_SCOPE.owns_generic_prefill_kernels);
        assert!(!M14_4D4_SCOPE.owns_static_heads8_online_or_cublas_prefill_dispatch);
        assert!(!M14_4D4_SCOPE.owns_indexed_or_output_q8_attention);
        assert!(!M14_4D4_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D4_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_optimized_prefill_scope_leaves_indexed_and_route_pending() {
        assert!(M14_4D5_SCOPE.opt_in_only);
        assert!(M14_4D5_SCOPE.owns_static_heads8_online_prefill_kernel);
        assert!(M14_4D5_SCOPE.owns_prefill_dispatch_policy);
        assert!(M14_4D5_SCOPE.owns_live_cublas_prefill_pipeline);
        assert!(!M14_4D5_SCOPE.owns_indexed_or_output_q8_attention);
        assert!(!M14_4D5_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D5_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_generic_indexed_scope_leaves_optimized_output_and_route_pending() {
        assert!(M14_4D6_SCOPE.opt_in_only);
        assert!(M14_4D6_SCOPE.owns_attention_indexed_mixed_surface);
        assert!(M14_4D6_SCOPE.owns_generic_indexed_kernel);
        assert!(M14_4D6_SCOPE.owns_topk_filter_and_order_semantics);
        assert!(!M14_4D6_SCOPE.owns_indexed_sort_or_heads8_dispatch);
        assert!(!M14_4D6_SCOPE.owns_output_q8_attention);
        assert!(!M14_4D6_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D6_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_optimized_indexed_scope_leaves_output_and_route_pending() {
        assert!(M14_4D7_SCOPE.opt_in_only);
        assert!(M14_4D7_SCOPE.consumes_indexed_topk_sort_policy);
        assert!(M14_4D7_SCOPE.owns_indexed_heads8_online_kernel);
        assert!(M14_4D7_SCOPE.owns_indexed_heads8_rb4_kernel);
        assert!(M14_4D7_SCOPE.owns_indexed_attention_dispatch_policy);
        assert!(!M14_4D7_SCOPE.owns_output_q8_attention);
        assert!(!M14_4D7_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D7_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_indexed_dispatch_paths_match_current_c_priority() {
        let base = AttentionIndexedDispatchOptions {
            n_tokens: 3,
            head_dim: 512,
            top_k: 512,
            no_indexed_heads8: false,
            indexed_twopass: false,
        };
        assert_eq!(
            select_attention_indexed_path(base),
            AttentionIndexedPath::Heads8Online
        );
        assert_eq!(
            select_attention_indexed_path(AttentionIndexedDispatchOptions {
                indexed_twopass: true,
                ..base
            }),
            AttentionIndexedPath::Heads8Rb4
        );
        assert_eq!(
            select_attention_indexed_path(AttentionIndexedDispatchOptions {
                no_indexed_heads8: true,
                ..base
            }),
            AttentionIndexedPath::Generic
        );
        assert_eq!(
            select_attention_indexed_path(AttentionIndexedDispatchOptions { top_k: 513, ..base }),
            AttentionIndexedPath::Generic
        );
    }

    #[test]
    fn attention_output_q8_native_scope_leaves_cublas_and_route_pending() {
        assert!(M14_4D8A_SCOPE.opt_in_only);
        assert!(M14_4D8A_SCOPE.consumes_q8_conversion_and_matmul_kernels);
        assert!(M14_4D8A_SCOPE.owns_attention_output_low_q8_surface);
        assert!(M14_4D8A_SCOPE.owns_attention_output_q8_batch_native_surface);
        assert!(!M14_4D8A_SCOPE.owns_attention_output_a_cublas_dispatch);
        assert!(!M14_4D8A_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D8A_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_output_cublas_scope_leaves_route_pending() {
        assert!(M14_4D8B_SCOPE.opt_in_only);
        assert!(M14_4D8B_SCOPE.consumes_attention_output_q8_native_surface);
        assert!(M14_4D8B_SCOPE.owns_attention_output_a_cublas_dispatch);
        assert!(M14_4D8B_SCOPE.owns_attention_output_a_pack_unpack_kernels);
        assert!(M14_4D8B_SCOPE.owns_live_cublas_grouped_a_pipeline);
        assert!(M14_4D8B_SCOPE.uses_safe_sgemm_f16_rounded_adapter);
        assert!(!M14_4D8B_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_4D8B_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_output_a_dispatch_paths_match_current_c_priority() {
        let base = AttentionOutputADispatchOptions {
            quality_mode: false,
            cublas_ready: true,
            n_tokens: 2,
            cublas_min_tokens: 2,
            no_cublas_attention_output_a: false,
            expanded_f16_ready: true,
        };
        assert_eq!(
            select_attention_output_a_path(base),
            AttentionOutputAPath::CublasF16
        );
        for native_options in [
            AttentionOutputADispatchOptions {
                quality_mode: true,
                ..base
            },
            AttentionOutputADispatchOptions {
                cublas_ready: false,
                ..base
            },
            AttentionOutputADispatchOptions {
                n_tokens: 1,
                ..base
            },
            AttentionOutputADispatchOptions {
                cublas_min_tokens: 3,
                ..base
            },
            AttentionOutputADispatchOptions {
                no_cublas_attention_output_a: true,
                ..base
            },
            AttentionOutputADispatchOptions {
                expanded_f16_ready: false,
                ..base
            },
        ] {
            assert_eq!(
                select_attention_output_a_path(native_options),
                AttentionOutputAPath::NativeQ8
            );
        }
    }

    #[test]
    fn router_scalar_scope_leaves_optimized_moe_and_route_pending() {
        assert!(M14_5A_SCOPE.opt_in_only);
        assert!(M14_5A_SCOPE.owns_router_select_kernel);
        assert!(M14_5A_SCOPE.owns_scalar_single_and_batch_router_surface);
        assert!(M14_5A_SCOPE.owns_bias_and_hash_router_semantics);
        assert!(!M14_5A_SCOPE.owns_parallel_or_warp_router_dispatch);
        assert!(!M14_5A_SCOPE.owns_routed_moe_or_hyperconnection);
        assert!(!M14_5A_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_5A_SCOPE.changes_default_route);
    }

    #[test]
    fn router_optimized_scope_leaves_moe_and_route_pending() {
        assert!(M14_5B_SCOPE.opt_in_only);
        assert!(M14_5B_SCOPE.consumes_scalar_router_surface);
        assert!(M14_5B_SCOPE.owns_router_select_parallel_kernel);
        assert!(M14_5B_SCOPE.owns_router_select_warp_topk_kernel);
        assert!(M14_5B_SCOPE.owns_parallel_and_warp_router_dispatch);
        assert!(M14_5B_SCOPE.owns_current_c_dispatch_priority);
        assert!(!M14_5B_SCOPE.owns_routed_moe_or_hyperconnection);
        assert!(!M14_5B_SCOPE.owns_runtime_graph_integration);
        assert!(!M14_5B_SCOPE.changes_default_route);
    }

    #[test]
    fn router_optimized_dispatch_paths_match_current_c_priority() {
        let default = RouterSelectDispatchOptions {
            no_warp_router_select: false,
            no_parallel_router_select: false,
        };
        assert_eq!(
            select_router_select_path(default),
            RouterSelectPath::WarpTopK
        );
        assert_eq!(
            select_router_select_path(RouterSelectDispatchOptions {
                no_warp_router_select: true,
                ..default
            }),
            RouterSelectPath::Parallel
        );
        assert_eq!(
            select_router_select_path(RouterSelectDispatchOptions {
                no_parallel_router_select: true,
                ..default
            }),
            RouterSelectPath::Scalar
        );
        assert_eq!(
            select_router_select_path(RouterSelectDispatchOptions {
                no_warp_router_select: true,
                no_parallel_router_select: true,
            }),
            RouterSelectPath::Scalar
        );
    }

    #[test]
    fn routed_moe_f32_scope_leaves_optimized_dispatch_and_route_pending() {
        assert!(M14_5C1_SCOPE.opt_in_only);
        assert!(M14_5C1_SCOPE.consumes_router_selection_surface);
        assert!(M14_5C1_SCOPE.owns_iq2_xxs_f32_gate_up_dot);
        assert!(M14_5C1_SCOPE.owns_q2_k_f32_down_dot);
        assert!(M14_5C1_SCOPE.owns_moe_gate_up_mid_f32_kernel);
        assert!(M14_5C1_SCOPE.owns_moe_down_f32_kernel);
        assert!(M14_5C1_SCOPE.owns_moe_sum_kernel);
        assert!(M14_5C1_SCOPE.owns_single_and_batch_f32_activation_moe_surface);
        assert!(!M14_5C1_SCOPE.owns_q8_activation_or_optimized_moe_dispatch);
        assert!(!M14_5C1_SCOPE.owns_hyperconnection_or_runtime_graph);
        assert!(!M14_5C1_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_quantized_single_scope_leaves_batch_q4_and_route_pending() {
        assert!(M14_5C2A_SCOPE.opt_in_only);
        assert!(M14_5C2A_SCOPE.consumes_f32_fallback_surface);
        assert!(M14_5C2A_SCOPE.owns_q8_k_activation_quantization);
        assert!(M14_5C2A_SCOPE.owns_iq2_xxs_q8_k_gate_up_decode_lut);
        assert!(M14_5C2A_SCOPE.owns_q2_k_q8_k_direct_sum6_down);
        assert!(M14_5C2A_SCOPE.owns_default_single_token_iq2_q2_dispatch);
        assert!(M14_5C2A_SCOPE.owns_optional_gate_up_aux_write);
        assert!(!M14_5C2A_SCOPE.owns_batched_sorted_or_tiled_dispatch);
        assert!(!M14_5C2A_SCOPE.owns_q4_k_dispatch);
        assert!(!M14_5C2A_SCOPE.owns_hyperconnection_or_runtime_graph);
        assert!(!M14_5C2A_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_q4_k_single_scope_leaves_hyperconnection_and_route_pending() {
        assert!(M14_5C2D_SCOPE.opt_in_only);
        assert!(M14_5C2D_SCOPE.consumes_quantized_single_surface);
        assert!(M14_5C2D_SCOPE.owns_q4_k_q8_k_dot);
        assert!(M14_5C2D_SCOPE.owns_moe_gate_up_mid_decode_q4_k_qwarp32_kernel);
        assert!(M14_5C2D_SCOPE.owns_moe_down_q4_k_sum6_qwarp32_kernel);
        assert!(M14_5C2D_SCOPE.owns_single_token_type12_dispatch);
        assert!(!M14_5C2D_SCOPE.owns_hyperconnection_or_runtime_graph);
        assert!(!M14_5C2D_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_sorted_pairs_scope_leaves_sorted_compute_and_route_pending() {
        assert!(M14_5C2B1_SCOPE.opt_in_only);
        assert!(M14_5C2B1_SCOPE.consumes_quantized_single_surface);
        assert!(M14_5C2B1_SCOPE.owns_moe_count_sorted_pairs_kernel);
        assert!(M14_5C2B1_SCOPE.owns_moe_prefix_sorted_pairs_kernel);
        assert!(M14_5C2B1_SCOPE.owns_moe_scatter_sorted_pairs_kernel);
        assert!(M14_5C2B1_SCOPE.owns_negative_expert_bucket_zero);
        assert!(M14_5C2B1_SCOPE.owns_sorted_pair_metadata);
        assert!(!M14_5C2B1_SCOPE.owns_sorted_projection_kernels);
        assert!(!M14_5C2B1_SCOPE.owns_expert_tile_or_atomic_down);
        assert!(!M14_5C2B1_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2B1_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_sorted_p2_scope_leaves_tiles_q4_and_route_pending() {
        assert!(M14_5C2B2_SCOPE.opt_in_only);
        assert!(M14_5C2B2_SCOPE.consumes_sorted_pair_metadata_surface);
        assert!(M14_5C2B2_SCOPE.uses_q8_k_activation_quantization);
        assert!(M14_5C2B2_SCOPE.owns_moe_gate_up_mid_sorted_p2_qwarp32_kernel);
        assert!(M14_5C2B2_SCOPE.owns_moe_down_sorted_p2_qwarp32_kernel);
        assert!(M14_5C2B2_SCOPE.owns_no_expert_tiles_p2_batch_dispatch);
        assert!(M14_5C2B2_SCOPE.uses_moe_sum_surface);
        assert!(!M14_5C2B2_SCOPE.owns_expert_tile_or_atomic_down);
        assert!(!M14_5C2B2_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2B2_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_expert_tiles_scope_leaves_projection_and_route_pending() {
        assert!(M14_5C2C1_SCOPE.opt_in_only);
        assert!(M14_5C2C1_SCOPE.consumes_sorted_pair_metadata_surface);
        assert!(M14_5C2C1_SCOPE.owns_moe_build_expert_tile_offsets_kernel);
        assert!(M14_5C2C1_SCOPE.owns_moe_build_expert_tiles_kernel);
        assert!(M14_5C2C1_SCOPE.owns_tile4_and_tile8_descriptor_metadata);
        assert!(!M14_5C2C1_SCOPE.owns_tile_projection_kernels);
        assert!(!M14_5C2C1_SCOPE.owns_atomic_down_or_rowspan_dispatch);
        assert!(!M14_5C2C1_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C1_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_tile8_row32_scope_leaves_atomic_and_specialized_routes_pending() {
        assert!(M14_5C2C2_SCOPE.opt_in_only);
        assert!(M14_5C2C2_SCOPE.consumes_expert_tile_metadata_surface);
        assert!(M14_5C2C2_SCOPE.uses_previously_owned_q8_k_inputs);
        assert!(M14_5C2C2_SCOPE.owns_moe_gate_up_mid_expert_tile8_row32_kernel);
        assert!(M14_5C2C2_SCOPE.owns_moe_down_expert_tile8_row32_non_atomic_surface);
        assert!(M14_5C2C2_SCOPE.owns_default_tile8_row32_projection_dispatch);
        assert!(!M14_5C2C2_SCOPE.owns_atomic_down_or_rowspan_dispatch);
        assert!(!M14_5C2C2_SCOPE.owns_shared_cache_specialization);
        assert!(!M14_5C2C2_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C2_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_tile4_row32_scope_leaves_atomic_and_cache_routes_pending() {
        assert!(M14_5C2C3_SCOPE.opt_in_only);
        assert!(M14_5C2C3_SCOPE.consumes_expert_tile_metadata_surface);
        assert!(M14_5C2C3_SCOPE.uses_previously_owned_q8_k_inputs);
        assert!(M14_5C2C3_SCOPE.owns_moe_gate_up_mid_expert_tile4_row32_kernel);
        assert!(M14_5C2C3_SCOPE.owns_moe_down_expert_tile4_row32_non_atomic_surface);
        assert!(M14_5C2C3_SCOPE.owns_optional_tile4_row32_projection_dispatch);
        assert!(!M14_5C2C3_SCOPE.owns_atomic_down_or_rowspan_dispatch);
        assert!(!M14_5C2C3_SCOPE.owns_shared_cache_specialization);
        assert!(!M14_5C2C3_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C3_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_atomic_down_scope_leaves_tile16_rowspan_and_cache_pending() {
        assert!(M14_5C2C4_SCOPE.opt_in_only);
        assert!(M14_5C2C4_SCOPE.consumes_tile_row32_projection_surface);
        assert!(M14_5C2C4_SCOPE.owns_device_atomic_f32_fetch_add);
        assert!(M14_5C2C4_SCOPE.owns_zero_kernel_for_atomic_down);
        assert!(M14_5C2C4_SCOPE.owns_tile4_and_tile8_row32_atomic_down_dispatch);
        assert!(!M14_5C2C4_SCOPE.owns_tile16_or_rowspan_dispatch);
        assert!(!M14_5C2C4_SCOPE.owns_shared_cache_specialization);
        assert!(!M14_5C2C4_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C4_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_tile16_row32_scope_leaves_rowspan_and_cache_pending() {
        assert!(M14_5C2C5_SCOPE.opt_in_only);
        assert!(M14_5C2C5_SCOPE.consumes_atomic_row32_surface);
        assert!(M14_5C2C5_SCOPE.owns_moe_down_expert_tile16_row32_kernel);
        assert!(M14_5C2C5_SCOPE.owns_tile16_atomic_down_dispatch);
        assert!(!M14_5C2C5_SCOPE.owns_rowspan_dispatch);
        assert!(!M14_5C2C5_SCOPE.owns_shared_cache_specialization);
        assert!(!M14_5C2C5_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C5_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_gate_rowspan_scope_leaves_down_rowspan_and_cache_pending() {
        assert!(M14_5C2C6_SCOPE.opt_in_only);
        assert!(M14_5C2C6_SCOPE.consumes_tile8_row32_projection_surface);
        assert!(M14_5C2C6_SCOPE.owns_moe_gate_up_mid_expert_tile8_rowspan_kernel);
        assert!(M14_5C2C6_SCOPE.owns_gate_row512_row1024_and_row2048_dispatch);
        assert!(!M14_5C2C6_SCOPE.owns_down_rowspan_dispatch);
        assert!(!M14_5C2C6_SCOPE.owns_shared_cache_specialization);
        assert!(!M14_5C2C6_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C6_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_down_rowspan_scope_leaves_cache_and_q4_pending() {
        assert!(M14_5C2C7_SCOPE.opt_in_only);
        assert!(M14_5C2C7_SCOPE.consumes_tile16_row32_atomic_surface);
        assert!(M14_5C2C7_SCOPE.owns_moe_down_expert_tile16_rowspan_kernel);
        assert!(M14_5C2C7_SCOPE.owns_down_row512_row1024_and_row2048_atomic_dispatch);
        assert!(!M14_5C2C7_SCOPE.owns_shared_cache_specialization);
        assert!(!M14_5C2C7_SCOPE.owns_q4_k_or_runtime_graph);
        assert!(!M14_5C2C7_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_shared_cache_scope_leaves_hyperconnection_and_route_pending() {
        assert!(M14_5C2E_SCOPE.opt_in_only);
        assert!(M14_5C2E_SCOPE.consumes_rowspan_projection_surface);
        assert!(M14_5C2E_SCOPE.owns_shared_cache_specialization);
        assert!(M14_5C2E_SCOPE.owns_gate_and_down_cached_rowspan_dispatch);
        assert!(!M14_5C2E_SCOPE.owns_hyperconnection_or_runtime_graph);
        assert!(!M14_5C2E_SCOPE.changes_default_route);
    }

    #[test]
    fn routed_moe_qwarp_fallback_scope_leaves_hyperconnection_and_route_pending() {
        assert!(M14_5C2F_SCOPE.opt_in_only);
        assert!(M14_5C2F_SCOPE.uses_q8_k_activation_quantization);
        assert!(M14_5C2F_SCOPE.owns_moe_gate_up_mid_qwarp32_kernel);
        assert!(M14_5C2F_SCOPE.owns_moe_down_qwarp32_kernel);
        assert!(M14_5C2F_SCOPE.owns_moe_gate_up_mid_sorted_qwarp32_kernel);
        assert!(M14_5C2F_SCOPE.owns_moe_down_sorted_qwarp32_kernel);
        assert!(M14_5C2F_SCOPE.owns_no_decode_lut_generic_dispatch);
        assert!(M14_5C2F_SCOPE.owns_no_p2_sorted_batch_dispatch);
        assert!(M14_5C2F_SCOPE.uses_moe_sum_surface);
        assert!(!M14_5C2F_SCOPE.owns_hyperconnection_or_runtime_graph);
        assert!(!M14_5C2F_SCOPE.changes_default_route);
    }

    #[test]
    fn hyperconnection_scope_leaves_shared_wrapper_and_route_pending() {
        assert!(M14_5D_SCOPE.opt_in_only);
        assert!(M14_5D_SCOPE.owns_hc_split_sinkhorn_kernel);
        assert!(M14_5D_SCOPE.owns_hc_weighted_sum_kernel);
        assert!(M14_5D_SCOPE.owns_hc_expand_kernel);
        assert!(M14_5D_SCOPE.owns_hc_split_weighted_sum_fused_kernel);
        assert!(M14_5D_SCOPE.owns_hc_split_weighted_sum_norm_fused_kernel);
        assert!(M14_5D_SCOPE.owns_output_hc_weights_kernel);
        assert!(!M14_5D_SCOPE.owns_shared_expert_wrapper_or_runtime_graph);
        assert!(!M14_5D_SCOPE.changes_default_route);
    }

    #[test]
    fn route_promotion_gate_records_production_abi_blocker() {
        assert!(M14_6A_GATE.operation_families_validated);
        assert!(M14_6A_GATE.production_build_still_compiles_ds4_cuda_cu);
        assert!(!M14_6A_GATE.rust_exports_ds4_gpu_abi);
        assert!(!M14_6A_GATE.runtime_graph_route_implemented);
        assert!(!M14_6A_GATE.can_promote_default_route);
        assert!(!M14_6A_GATE.can_remove_c_cuda);
    }

    #[test]
    fn resource_abi_scope_leaves_compute_and_route_pending() {
        assert_eq!(M14_6B1_SCOPE.exported_resource_symbol_count, 16);
        assert!(M14_6B1_SCOPE.owns_initialization);
        assert!(M14_6B1_SCOPE.owns_tensor_storage);
        assert!(M14_6B1_SCOPE.owns_host_device_copies);
        assert!(M14_6B1_SCOPE.owns_command_synchronization);
        assert!(M14_6B1_SCOPE.owns_managed_kv_policy);
        assert!(!M14_6B1_SCOPE.owns_tensor_fill_kernel);
        assert!(!M14_6B1_SCOPE.owns_compute_abi);
        assert!(!M14_6B1_SCOPE.owns_complete_ds4_gpu_abi);
        assert!(!M14_6B1_SCOPE.changes_default_route);
    }

    #[test]
    fn tensor_fill_abi_scope_leaves_graph_compute_and_route_pending() {
        assert_eq!(M14_6B2A_SCOPE.exported_abi_symbol_count, 17);
        assert_eq!(M14_6B2A_SCOPE.exported_compute_symbol_count, 1);
        assert!(M14_6B2A_SCOPE.owns_tensor_fill_f32);
        assert!(!M14_6B2A_SCOPE.owns_graph_compute_abi);
        assert!(!M14_6B2A_SCOPE.owns_complete_ds4_gpu_abi);
        assert!(!M14_6B2A_SCOPE.changes_default_route);
    }

    #[test]
    fn elementwise_abi_scope_leaves_remaining_graph_compute_and_route_pending() {
        assert_eq!(M14_6B2B1_SCOPE.exported_abi_symbol_count, 19);
        assert_eq!(M14_6B2B1_SCOPE.exported_compute_symbol_count, 3);
        assert!(M14_6B2B1_SCOPE.owns_add_tensor);
        assert!(M14_6B2B1_SCOPE.owns_repeat_hc_tensor);
        assert!(M14_6B2B1_SCOPE.uses_embedded_rust_kernel_module);
        assert!(!M14_6B2B1_SCOPE.owns_remaining_graph_compute_abi);
        assert!(!M14_6B2B1_SCOPE.owns_complete_ds4_gpu_abi);
        assert!(!M14_6B2B1_SCOPE.changes_default_route);
    }

    #[test]
    fn directional_steering_abi_scope_leaves_remaining_graph_compute_and_route_pending() {
        assert_eq!(M14_6B2B2A_SCOPE.exported_abi_symbol_count, 20);
        assert_eq!(M14_6B2B2A_SCOPE.exported_compute_symbol_count, 4);
        assert!(M14_6B2B2A_SCOPE.owns_directional_steering_project_tensor);
        assert!(!M14_6B2B2A_SCOPE.owns_remaining_graph_compute_abi);
        assert!(!M14_6B2B2A_SCOPE.owns_complete_ds4_gpu_abi);
        assert!(!M14_6B2B2A_SCOPE.changes_default_route);
    }

    #[test]
    fn attention_prefill_dispatch_paths_match_current_c_priority() {
        let base = AttentionPrefillDispatchOptions {
            use_comp_mask: false,
            n_tokens: 128,
            head_dim: 512,
            cublas_ready: true,
            no_cublas_attention: false,
            no_window_attention: false,
            window_attention: false,
            quality_mode: false,
        };
        assert_eq!(
            select_attention_prefill_path(base),
            AttentionPrefillPath::StaticHeads8Online
        );
        assert_eq!(
            select_attention_prefill_path(AttentionPrefillDispatchOptions {
                use_comp_mask: true,
                ..base
            }),
            AttentionPrefillPath::Cublas
        );
        assert_eq!(
            select_attention_prefill_path(AttentionPrefillDispatchOptions {
                no_window_attention: true,
                ..base
            }),
            AttentionPrefillPath::Cublas
        );
        assert_eq!(
            select_attention_prefill_path(AttentionPrefillDispatchOptions {
                n_tokens: 2,
                window_attention: true,
                ..base
            }),
            AttentionPrefillPath::StaticHeads8Online
        );
        assert_eq!(
            select_attention_prefill_path(AttentionPrefillDispatchOptions {
                n_tokens: 2,
                quality_mode: true,
                ..base
            }),
            AttentionPrefillPath::Cublas
        );
        assert_eq!(
            select_attention_prefill_path(AttentionPrefillDispatchOptions {
                n_tokens: 2,
                quality_mode: true,
                no_cublas_attention: true,
                ..base
            }),
            AttentionPrefillPath::Generic
        );
        assert_eq!(
            select_attention_prefill_path(AttentionPrefillDispatchOptions {
                head_dim: 256,
                ..base
            }),
            AttentionPrefillPath::Generic
        );
    }

    #[test]
    fn q8_matmul_dispatch_paths_match_current_c_priority() {
        let base = Q8MatmulDispatchOptions {
            cublas_ready: true,
            expanded_f32_blas_ready: true,
            expanded_f16_blas_ready: true,
            n_tokens: 2,
            blocks: 2,
            no_batch_warp: false,
        };
        assert_eq!(select_q8_matmul_path(base), Q8MatmulPath::ExpandedF32Blas);
        assert_eq!(
            select_q8_matmul_path(Q8MatmulDispatchOptions {
                expanded_f32_blas_ready: false,
                ..base
            }),
            Q8MatmulPath::ExpandedF16Blas
        );
        assert_eq!(
            select_q8_matmul_path(Q8MatmulDispatchOptions {
                cublas_ready: false,
                n_tokens: 1,
                ..base
            }),
            Q8MatmulPath::PrequantizedWarp8
        );
        assert_eq!(
            select_q8_matmul_path(Q8MatmulDispatchOptions {
                cublas_ready: false,
                ..base
            }),
            Q8MatmulPath::PrequantizedBatchWarp8
        );
        assert_eq!(
            select_q8_matmul_path(Q8MatmulDispatchOptions {
                cublas_ready: false,
                no_batch_warp: true,
                ..base
            }),
            Q8MatmulPath::PrequantizedGeneric
        );
        assert_eq!(
            select_q8_matmul_path(Q8MatmulDispatchOptions {
                cublas_ready: false,
                blocks: 33,
                ..base
            }),
            Q8MatmulPath::PrequantizedGeneric
        );
        assert!(q8_dp4a_enabled(false));
        assert!(!q8_dp4a_enabled(true));
    }

    #[test]
    fn dense_projection_dispatch_paths_match_current_c_priority() {
        let base = F16ProjectionDispatch {
            blas_ready: true,
            serial_f16: false,
            serial_router: false,
            no_ordered_f16_matmul: false,
            in_dim: 1024,
            out_dim: 512,
            n_tokens: 2,
        };
        assert_eq!(select_f16_projection_path(base), F16ProjectionPath::Blas);
        assert_eq!(
            select_f16_projection_path(F16ProjectionDispatch {
                serial_f16: true,
                ..base
            }),
            F16ProjectionPath::Serial
        );
        assert_eq!(
            select_f16_projection_path(F16ProjectionDispatch {
                blas_ready: false,
                serial_router: true,
                in_dim: 4096,
                out_dim: 256,
                n_tokens: 1,
                ..base
            }),
            F16ProjectionPath::Serial
        );
        assert_eq!(
            select_f16_projection_path(F16ProjectionDispatch {
                blas_ready: false,
                n_tokens: 1,
                ..base
            }),
            F16ProjectionPath::OrderedChunks
        );
        assert_eq!(
            select_f16_projection_path(F16ProjectionDispatch {
                blas_ready: false,
                no_ordered_f16_matmul: true,
                n_tokens: 1,
                ..base
            }),
            F16ProjectionPath::Base
        );
        assert_eq!(
            select_f16_pair_projection_path(F16PairProjectionDispatch {
                n_tokens: 1,
                no_f16_pair_matmul: false,
                serial_f16: false,
                serial_router: false,
                no_ordered_f16_matmul: false,
            }),
            F16PairProjectionPath::PairedOrderedChunks
        );
        assert_eq!(
            select_f16_pair_projection_path(F16PairProjectionDispatch {
                n_tokens: 2,
                no_f16_pair_matmul: false,
                serial_f16: false,
                serial_router: false,
                no_ordered_f16_matmul: false,
            }),
            F16PairProjectionPath::TwoIndependent
        );
        assert_eq!(select_f32_projection_path(true, 2), F32ProjectionPath::Blas);
        assert_eq!(select_f32_projection_path(true, 1), F32ProjectionPath::Base);
    }

    #[test]
    fn specialized_topk_dispatch_priority_matches_current_c_launch_order() {
        let base = IndexerTopkDispatchOptions {
            n_comp: 1000,
            top_k: 512,
            no_topk1024: false,
            no_topk2048: false,
            no_topk8192: false,
            no_topk_chunked: false,
            packed_dynamic_shared_available: true,
        };
        assert_eq!(
            select_indexer_topk_kernel(base),
            IndexerTopkKernel::Topk1024
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                no_topk1024: true,
                ..base
            }),
            IndexerTopkKernel::Pow2U32x2048
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 4096,
                ..base
            }),
            IndexerTopkKernel::PackedKeyEquivalent
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 4096,
                packed_dynamic_shared_available: false,
                ..base
            }),
            IndexerTopkKernel::Pow2U32x4096
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 6000,
                ..base
            }),
            IndexerTopkKernel::PackedKeyEquivalent
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 6000,
                packed_dynamic_shared_available: false,
                ..base
            }),
            IndexerTopkKernel::Pow2U16x8192
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 6000,
                no_topk8192: true,
                ..base
            }),
            IndexerTopkKernel::ChunkedTree
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 9000,
                ..base
            }),
            IndexerTopkKernel::ChunkedTree
        );
        assert_eq!(
            select_indexer_topk_kernel(IndexerTopkDispatchOptions {
                n_comp: 9000,
                no_topk_chunked: true,
                ..base
            }),
            IndexerTopkKernel::Scalar
        );
        assert!(should_sort_indexed_topk(IndexedTopkSortOptions {
            n_tokens: 2,
            top_k: 512,
            no_indexed_topk_sort: false,
        }));
        assert!(!should_sort_indexed_topk(IndexedTopkSortOptions {
            n_tokens: 1,
            top_k: 512,
            no_indexed_topk_sort: false,
        }));
    }

    #[test]
    fn indexer_score_dispatch_priority_matches_current_c_launch_order() {
        let base = IndexerScoreDispatchOptions {
            n_tokens: 2,
            n_head: 64,
            head_dim: 128,
            quality_mode: false,
            no_direct_one: false,
            no_wmma: false,
            no_wmma128: false,
            no_wmma64: false,
            no_wmma32: false,
        };
        assert_eq!(
            select_indexer_score_kernel(base),
            IndexerScoreKernel::Wmma128
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                n_tokens: 1,
                quality_mode: true,
                no_wmma: true,
                ..base
            }),
            IndexerScoreKernel::DirectOne
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                n_tokens: 1,
                no_direct_one: true,
                ..base
            }),
            IndexerScoreKernel::Wmma128
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                no_wmma128: true,
                ..base
            }),
            IndexerScoreKernel::Wmma64
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                no_wmma128: true,
                no_wmma64: true,
                ..base
            }),
            IndexerScoreKernel::Wmma32
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                no_wmma128: true,
                no_wmma64: true,
                no_wmma32: true,
                ..base
            }),
            IndexerScoreKernel::Wmma
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                no_wmma: true,
                ..base
            }),
            IndexerScoreKernel::Scalar
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                quality_mode: true,
                ..base
            }),
            IndexerScoreKernel::Scalar
        );
        assert_eq!(
            select_indexer_score_kernel(IndexerScoreDispatchOptions {
                head_dim: 64,
                ..base
            }),
            IndexerScoreKernel::Scalar
        );
    }
}
