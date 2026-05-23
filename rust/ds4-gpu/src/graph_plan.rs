//! Rust-side graph inventory for the DS4 runtime graph.
//!
//! This module does not execute backend kernels. It names the current C graph
//! tensors, backend operation facade targets, and capacity math captured by the
//! M10.2 oracle so later graph scheduling code can fail closed before calling
//! the FFI backend.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelGraphDims {
    pub q_rank: u64,
    pub shared_dim: u64,
    pub routed_mid_dim: u64,
    pub vocab_dim: u64,
}

impl ModelGraphDims {
    pub const DS4_FLASH: Self = Self {
        q_rank: N_LORA_Q as u64,
        shared_dim: N_FF_EXP as u64,
        routed_mid_dim: N_FF_EXP as u64,
        vocab_dim: N_VOCAB as u64,
    };
}

pub const N_LAYER: usize = 43;
pub const N_EMBD: u32 = 4096;
pub const N_VOCAB: u32 = 129_280;
pub const N_HEAD: u32 = 64;
pub const N_HEAD_KV: u32 = 1;
pub const N_HEAD_DIM: u32 = 512;
pub const N_VALUE_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
pub const N_OUT_GROUP: u32 = 8;
pub const N_LORA_Q: u32 = 1024;
pub const N_LORA_O: u32 = 1024;
pub const N_EXPERT: u32 = 256;
pub const N_EXPERT_USED: u32 = 6;
pub const N_EXPERT_SHARED: u32 = 1;
pub const N_FF_EXP: u32 = 2048;
pub const N_HASH_LAYER: u32 = 3;
pub const N_SWA: u32 = 128;
pub const N_INDEXER_HEAD: u32 = 64;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const N_INDEXER_TOP_K: u32 = 512;
pub const N_HC: u32 = 4;
pub const N_HC_SINKHORN_ITER: u32 = 20;
pub const RMS_EPS: f32 = 1.0e-6;
pub const HC_EPS: f32 = 1.0e-6;

const BYTES_F32: u64 = 4;
const BYTES_I32: u64 = 4;
const BYTES_U32: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerCompression {
    Dense,
    Ratio4,
    Ratio128,
}

impl LayerCompression {
    pub const fn ratio(self) -> u32 {
        match self {
            Self::Dense => 0,
            Self::Ratio4 => 4,
            Self::Ratio128 => 128,
        }
    }
}

pub const fn layer_compression(layer: usize) -> Option<LayerCompression> {
    if layer >= N_LAYER {
        None
    } else if layer < 2 {
        Some(LayerCompression::Dense)
    } else if layer % 2 == 0 {
        Some(LayerCompression::Ratio4)
    } else {
        Some(LayerCompression::Ratio128)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerCounts {
    pub dense: u32,
    pub ratio4: u32,
    pub ratio128: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerCompCapByRatio {
    pub dense: u32,
    pub ratio4: u32,
    pub ratio128: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphPlan {
    pub ctx_size: u32,
    pub prompt_len: u32,
    pub mtp_enabled: bool,
    pub prefill_cap: u32,
    pub raw_window: u32,
    pub requested_raw_cap: u32,
    pub allocated_raw_cap: u32,
    pub comp_cap: u32,
    pub layer_counts: LayerCounts,
    pub layer_comp_cap_by_ratio: LayerCompCapByRatio,
    pub ratio4_indexer_layers: u32,
    pub mtp_tensor_group: &'static str,
}

impl GraphPlan {
    pub const fn for_context(ctx_size: u32, prompt_len: u32, mtp_enabled: bool) -> Self {
        let prefill_cap = default_prefill_cap(prompt_len);
        let raw_window = raw_window_for_context(ctx_size);
        let requested_raw_cap = raw_cap_for_context(ctx_size, prefill_cap);
        let mut allocated_raw_cap = requested_raw_cap;
        if allocated_raw_cap < raw_window {
            allocated_raw_cap = raw_window;
        }
        if allocated_raw_cap > ctx_size {
            allocated_raw_cap = ctx_size;
        }
        if allocated_raw_cap == 0 {
            allocated_raw_cap = 1;
        }

        Self {
            ctx_size,
            prompt_len,
            mtp_enabled,
            prefill_cap,
            raw_window,
            requested_raw_cap,
            allocated_raw_cap,
            comp_cap: comp_cap_for_context(ctx_size),
            layer_counts: LayerCounts {
                dense: 2,
                ratio4: 21,
                ratio128: 20,
            },
            layer_comp_cap_by_ratio: LayerCompCapByRatio {
                dense: 0,
                ratio4: layer_comp_cap_for_ratio(ctx_size, 4),
                ratio128: layer_comp_cap_for_ratio(ctx_size, 128),
            },
            ratio4_indexer_layers: 21,
            mtp_tensor_group: if mtp_enabled {
                "mtp_optional_state"
            } else {
                "none"
            },
        }
    }

    pub const fn layer_comp_cap(self, compression: LayerCompression) -> u32 {
        match compression {
            LayerCompression::Dense => self.layer_comp_cap_by_ratio.dense,
            LayerCompression::Ratio4 => self.layer_comp_cap_by_ratio.ratio4,
            LayerCompression::Ratio128 => self.layer_comp_cap_by_ratio.ratio128,
        }
    }
}

pub const M102_PLAN_CASES: [GraphPlan; 4] = [
    GraphPlan::for_context(128, 128, false),
    GraphPlan::for_context(2048, 2048, false),
    GraphPlan::for_context(32768, 32768, false),
    GraphPlan::for_context(32768, 32768, true),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphPlanCaseOracle {
    pub name: &'static str,
    pub ctx_size: u32,
    pub prompt_len: u32,
    pub mtp_enabled: bool,
    pub prefill_cap: u32,
    pub raw_window: u32,
    pub requested_raw_cap: u32,
    pub allocated_raw_cap: u32,
    pub comp_cap: u32,
    pub layer_counts: LayerCounts,
    pub layer_comp_cap_by_ratio: LayerCompCapByRatio,
    pub ratio4_indexer_layers: u32,
    pub mtp_tensor_group: &'static str,
}

impl GraphPlanCaseOracle {
    pub const fn expected_plan(self) -> GraphPlan {
        GraphPlan {
            ctx_size: self.ctx_size,
            prompt_len: self.prompt_len,
            mtp_enabled: self.mtp_enabled,
            prefill_cap: self.prefill_cap,
            raw_window: self.raw_window,
            requested_raw_cap: self.requested_raw_cap,
            allocated_raw_cap: self.allocated_raw_cap,
            comp_cap: self.comp_cap,
            layer_counts: self.layer_counts,
            layer_comp_cap_by_ratio: self.layer_comp_cap_by_ratio,
            ratio4_indexer_layers: self.ratio4_indexer_layers,
            mtp_tensor_group: self.mtp_tensor_group,
        }
    }
}

macro_rules! case {
    (
        $name:literal,
        $ctx_size:literal,
        $prompt_len:literal,
        $mtp_enabled:literal,
        $prefill_cap:literal,
        $raw_window:literal,
        $requested_raw_cap:literal,
        $allocated_raw_cap:literal,
        $comp_cap:literal,
        $dense_layers:literal,
        $ratio4_layers:literal,
        $ratio128_layers:literal,
        $dense_cap:literal,
        $ratio4_cap:literal,
        $ratio128_cap:literal,
        $ratio4_indexer_layers:literal,
        $mtp_tensor_group:literal
    ) => {
        GraphPlanCaseOracle {
            name: $name,
            ctx_size: $ctx_size,
            prompt_len: $prompt_len,
            mtp_enabled: $mtp_enabled,
            prefill_cap: $prefill_cap,
            raw_window: $raw_window,
            requested_raw_cap: $requested_raw_cap,
            allocated_raw_cap: $allocated_raw_cap,
            comp_cap: $comp_cap,
            layer_counts: LayerCounts {
                dense: $dense_layers,
                ratio4: $ratio4_layers,
                ratio128: $ratio128_layers,
            },
            layer_comp_cap_by_ratio: LayerCompCapByRatio {
                dense: $dense_cap,
                ratio4: $ratio4_cap,
                ratio128: $ratio128_cap,
            },
            ratio4_indexer_layers: $ratio4_indexer_layers,
            mtp_tensor_group: $mtp_tensor_group,
        }
    };
}

pub const M102_PLAN_CASE_ORACLE: &[GraphPlanCaseOracle] = &[
    case!(
        "short_ctx128_mtp_off",
        128,
        128,
        false,
        128,
        128,
        256,
        128,
        34,
        2,
        21,
        20,
        0,
        34,
        3,
        21,
        "none"
    ),
    case!(
        "ctx2048_mtp_off",
        2048,
        2048,
        false,
        2048,
        128,
        2048,
        2048,
        514,
        2,
        21,
        20,
        0,
        514,
        18,
        21,
        "none"
    ),
    case!(
        "ctx32768_mtp_off",
        32768,
        32768,
        false,
        2048,
        128,
        2304,
        2304,
        8194,
        2,
        21,
        20,
        0,
        8194,
        258,
        21,
        "none"
    ),
    case!(
        "ctx32768_mtp_on",
        32768,
        32768,
        true,
        2048,
        128,
        2304,
        2304,
        8194,
        2,
        21,
        20,
        0,
        8194,
        258,
        21,
        "mtp_optional_state"
    ),
];

pub const fn default_prefill_cap(prompt_len: u32) -> u32 {
    if prompt_len == 0 {
        1
    } else if prompt_len > 2048 {
        2048
    } else {
        prompt_len
    }
}

pub const fn raw_window_for_context(ctx_size: u32) -> u32 {
    if ctx_size == 0 {
        1
    } else if N_SWA > ctx_size {
        ctx_size
    } else {
        N_SWA
    }
}

pub const fn raw_cap_for_context(ctx_size: u32, prefill_cap: u32) -> u32 {
    let raw_window = raw_window_for_context(ctx_size);
    let mut wanted = raw_window + prefill_cap;
    if wanted > ctx_size {
        wanted = ctx_size;
    }
    if wanted == 0 {
        wanted = 1;
    }
    wanted = align_up(wanted, 256);
    if wanted > 8192 {
        wanted = 8192;
    }
    if wanted < raw_window {
        raw_window
    } else {
        wanted
    }
}

pub const fn comp_cap_for_context(ctx_size: u32) -> u32 {
    let cap = ctx_size / 4 + 2;
    if cap < 2 {
        2
    } else {
        cap
    }
}

pub const fn layer_comp_cap_for_ratio(ctx_size: u32, ratio: u32) -> u32 {
    if ratio == 0 {
        0
    } else {
        let cap = ctx_size / ratio + 2;
        if cap < 2 {
            2
        } else {
            cap
        }
    }
}

pub const fn align_up(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendFacade {
    BackendLifecycle,
    TensorBackend,
    CommandBackend,
    ModelMapBackend,
    EmbeddingIndexerBackend,
    ProjectionNormKvBackend,
    AttentionCompressorBackend,
    RoutingMoeBackend,
    HyperConnectionOutputBackend,
}

impl BackendFacade {
    pub const fn name(self) -> &'static str {
        match self {
            Self::BackendLifecycle => "BackendLifecycle",
            Self::TensorBackend => "TensorBackend",
            Self::CommandBackend => "CommandBackend",
            Self::ModelMapBackend => "ModelMapBackend",
            Self::EmbeddingIndexerBackend => "EmbeddingIndexerBackend",
            Self::ProjectionNormKvBackend => "ProjectionNormKvBackend",
            Self::AttentionCompressorBackend => "AttentionCompressorBackend",
            Self::RoutingMoeBackend => "RoutingMoeBackend",
            Self::HyperConnectionOutputBackend => "HyperConnectionOutputBackend",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendOperationSpec {
    pub name: &'static str,
    pub facade: BackendFacade,
}

macro_rules! op {
    ($name:literal, $facade:ident) => {
        BackendOperationSpec {
            name: $name,
            facade: BackendFacade::$facade,
        }
    };
}

pub const BACKEND_OPERATIONS: &[BackendOperationSpec] = &[
    op!("ds4_gpu_init", BackendLifecycle),
    op!("ds4_gpu_cleanup", BackendLifecycle),
    op!("ds4_gpu_set_quality", BackendLifecycle),
    op!("ds4_gpu_print_memory_report", BackendLifecycle),
    op!("ds4_gpu_tensor_alloc", TensorBackend),
    op!("ds4_gpu_tensor_alloc_managed", TensorBackend),
    op!("ds4_gpu_tensor_view", TensorBackend),
    op!("ds4_gpu_tensor_free", TensorBackend),
    op!("ds4_gpu_tensor_bytes", TensorBackend),
    op!("ds4_gpu_tensor_contents", TensorBackend),
    op!("ds4_gpu_tensor_fill_f32", TensorBackend),
    op!("ds4_gpu_tensor_write", TensorBackend),
    op!("ds4_gpu_tensor_read", TensorBackend),
    op!("ds4_gpu_tensor_copy", TensorBackend),
    op!("ds4_gpu_begin_commands", CommandBackend),
    op!("ds4_gpu_flush_commands", CommandBackend),
    op!("ds4_gpu_end_commands", CommandBackend),
    op!("ds4_gpu_synchronize", CommandBackend),
    op!("ds4_gpu_set_model_map", ModelMapBackend),
    op!("ds4_gpu_set_model_fd", ModelMapBackend),
    op!("ds4_gpu_set_model_map_range", ModelMapBackend),
    op!("ds4_gpu_cache_model_range", ModelMapBackend),
    op!("ds4_gpu_cache_q8_f16_range", ModelMapBackend),
    op!("ds4_gpu_should_use_managed_kv_cache", ModelMapBackend),
    op!("ds4_gpu_embed_token_hc_tensor", EmbeddingIndexerBackend),
    op!("ds4_gpu_embed_tokens_hc_tensor", EmbeddingIndexerBackend),
    op!("ds4_gpu_indexer_score_one_tensor", EmbeddingIndexerBackend),
    op!(
        "ds4_gpu_indexer_scores_prefill_tensor",
        EmbeddingIndexerBackend
    ),
    op!(
        "ds4_gpu_indexer_scores_decode_batch_tensor",
        EmbeddingIndexerBackend
    ),
    op!("ds4_gpu_indexer_topk_tensor", EmbeddingIndexerBackend),
    op!("ds4_gpu_dsv4_topk_mask_tensor", EmbeddingIndexerBackend),
    op!("ds4_gpu_matmul_q8_0_tensor", ProjectionNormKvBackend),
    op!(
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        ProjectionNormKvBackend
    ),
    op!("ds4_gpu_matmul_f16_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_matmul_f16_pair_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_matmul_f32_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_repeat_hc_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_rms_norm_plain_tensor", ProjectionNormKvBackend),
    op!(
        "ds4_gpu_rms_norm_plain_rows_tensor",
        ProjectionNormKvBackend
    ),
    op!("ds4_gpu_rms_norm_weight_tensor", ProjectionNormKvBackend),
    op!(
        "ds4_gpu_rms_norm_weight_rows_tensor",
        ProjectionNormKvBackend
    ),
    op!(
        "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
        ProjectionNormKvBackend
    ),
    op!("ds4_gpu_head_rms_norm_tensor", ProjectionNormKvBackend),
    op!(
        "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
        ProjectionNormKvBackend
    ),
    op!("ds4_gpu_dsv4_indexer_qat_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_rope_tail_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_kv_fp8_store_raw_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_store_raw_kv_tensor", ProjectionNormKvBackend),
    op!("ds4_gpu_store_raw_kv_batch_tensor", ProjectionNormKvBackend),
    op!(
        "ds4_gpu_compressor_update_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_compressor_store_batch_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_compressor_prefill_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_compressor_prefill_ratio4_replay_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_compressor_prefill_state_ratio4_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_decode_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_prefill_raw_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_decode_raw_batch_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_decode_mixed_batch_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_prefill_static_mixed_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_prefill_masked_mixed_heads_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_output_q8_batch_tensor",
        AttentionCompressorBackend
    ),
    op!(
        "ds4_gpu_attention_output_low_q8_tensor",
        AttentionCompressorBackend
    ),
    op!("ds4_gpu_swiglu_tensor", RoutingMoeBackend),
    op!("ds4_gpu_add_tensor", RoutingMoeBackend),
    op!(
        "ds4_gpu_directional_steering_project_tensor",
        RoutingMoeBackend
    ),
    op!("ds4_gpu_router_select_tensor", RoutingMoeBackend),
    op!("ds4_gpu_router_select_batch_tensor", RoutingMoeBackend),
    op!("ds4_gpu_routed_moe_one_tensor", RoutingMoeBackend),
    op!("ds4_gpu_routed_moe_batch_tensor", RoutingMoeBackend),
    op!(
        "ds4_gpu_hc_split_sinkhorn_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_hc_weighted_sum_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_hc_weighted_sum_split_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_hc_split_weighted_sum_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_output_hc_weights_tensor",
        HyperConnectionOutputBackend
    ),
    op!("ds4_gpu_hc_expand_tensor", HyperConnectionOutputBackend),
    op!(
        "ds4_gpu_hc_expand_split_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_hc_expand_add_split_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        HyperConnectionOutputBackend
    ),
    op!(
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        HyperConnectionOutputBackend
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorOwner {
    GraphDecodeState,
    GraphPersistentKvState,
    GraphSpeculativeFrontierState,
    GraphLayerWorkState,
    GraphMtpState,
    GraphPrefillBatchState,
    GraphOptionalControlState,
}

impl TensorOwner {
    pub const fn name(self) -> &'static str {
        match self {
            Self::GraphDecodeState => "GraphDecodeState",
            Self::GraphPersistentKvState => "GraphPersistentKvState",
            Self::GraphSpeculativeFrontierState => "GraphSpeculativeFrontierState",
            Self::GraphLayerWorkState => "GraphLayerWorkState",
            Self::GraphMtpState => "GraphMtpState",
            Self::GraphPrefillBatchState => "GraphPrefillBatchState",
            Self::GraphOptionalControlState => "GraphOptionalControlState",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementType {
    F32,
    I32,
    U32,
}

impl ElementType {
    pub const fn byte_len(self) -> u64 {
        match self {
            Self::F32 => BYTES_F32,
            Self::I32 => BYTES_I32,
            Self::U32 => BYTES_U32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelDim {
    One,
    Hc,
    MixHc,
    HcPrePost,
    HcComb,
    Embd,
    QRank,
    QDim,
    HeadDim,
    CompWidthMax,
    IndexerQ,
    IndexerHead,
    LowDim,
    GroupDim,
    LoraO,
    SharedDim,
    Expert,
    ExpertUsed,
    RoutedExpertUsedMid,
    RoutedExpertUsedEmbd,
    Vocab,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorElements {
    Fixed(ModelDim),
    Prefill(ModelDim),
    PrefillCompCap,
    PrefillTopK,
    RawCache,
    LayerAttnCompCache,
    LayerIndexCompCache,
    LayerAttnState,
    LayerIndexState,
    SpecLogits,
    External,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorByteLen {
    Known(u64),
    External,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphTensorSpec {
    pub name: &'static str,
    pub owner: TensorOwner,
    pub element_type: ElementType,
    pub elements: TensorElements,
}

impl GraphTensorSpec {
    pub fn byte_len(
        self,
        plan: GraphPlan,
        dims: ModelGraphDims,
        layer: Option<usize>,
    ) -> Result<TensorByteLen, GraphPlanError<'static>> {
        let elements = match self.elements {
            TensorElements::Fixed(dim) => dim_elements(dim, dims),
            TensorElements::Prefill(dim) => u64::from(plan.prefill_cap) * dim_elements(dim, dims),
            TensorElements::PrefillCompCap => {
                u64::from(plan.prefill_cap) * u64::from(plan.comp_cap)
            }
            TensorElements::PrefillTopK => {
                let top_k = if N_INDEXER_TOP_K == 0 {
                    1
                } else {
                    N_INDEXER_TOP_K
                };
                u64::from(plan.prefill_cap) * u64::from(top_k)
            }
            TensorElements::RawCache => u64::from(plan.allocated_raw_cap) * u64::from(N_HEAD_DIM),
            TensorElements::LayerAttnCompCache => {
                let layer = layer.ok_or(GraphPlanError::MissingLayer { field: self.name })?;
                let compression = layer_compression(layer).ok_or(GraphPlanError::InvalidLayer {
                    field: self.name,
                    layer,
                })?;
                if compression == LayerCompression::Dense {
                    0
                } else {
                    u64::from(plan.layer_comp_cap(compression)) * u64::from(N_HEAD_DIM)
                }
            }
            TensorElements::LayerIndexCompCache => {
                let layer = layer.ok_or(GraphPlanError::MissingLayer { field: self.name })?;
                let compression = layer_compression(layer).ok_or(GraphPlanError::InvalidLayer {
                    field: self.name,
                    layer,
                })?;
                if compression == LayerCompression::Ratio4 {
                    u64::from(plan.layer_comp_cap(compression)) * u64::from(N_INDEXER_HEAD_DIM)
                } else {
                    0
                }
            }
            TensorElements::LayerAttnState => layer_attn_state_elements(
                self.name,
                layer.ok_or(GraphPlanError::MissingLayer { field: self.name })?,
            )?,
            TensorElements::LayerIndexState => layer_index_state_elements(
                self.name,
                layer.ok_or(GraphPlanError::MissingLayer { field: self.name })?,
            )?,
            TensorElements::SpecLogits => 16 * dims.vocab_dim,
            TensorElements::External => return Ok(TensorByteLen::External),
        };
        Ok(TensorByteLen::Known(
            elements * self.element_type.byte_len(),
        ))
    }
}

macro_rules! field {
    ($name:literal, $owner:ident, $ty:ident, $elements:expr) => {
        GraphTensorSpec {
            name: $name,
            owner: TensorOwner::$owner,
            element_type: ElementType::$ty,
            elements: $elements,
        }
    };
}

pub const GRAPH_TENSOR_FIELDS: &[GraphTensorSpec] = &[
    field!(
        "cur_hc",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "flat_hc",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "hc_mix",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::MixHc)
    ),
    field!(
        "hc_split",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::MixHc)
    ),
    field!(
        "hc_pre",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::HcPrePost)
    ),
    field!(
        "hc_post",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::HcPrePost)
    ),
    field!(
        "hc_comb",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::HcComb)
    ),
    field!(
        "attn_cur",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "attn_norm",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "qr",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::QRank)
    ),
    field!(
        "qr_norm",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::QRank)
    ),
    field!(
        "q",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::QDim)
    ),
    field!(
        "kv_raw",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::HeadDim)
    ),
    field!(
        "kv",
        GraphDecodeState,
        F32,
        TensorElements::Fixed(ModelDim::HeadDim)
    ),
    field!(
        "layer_raw_cache",
        GraphPersistentKvState,
        F32,
        TensorElements::RawCache
    ),
    field!(
        "layer_attn_comp_cache",
        GraphPersistentKvState,
        F32,
        TensorElements::LayerAttnCompCache
    ),
    field!(
        "layer_attn_state_kv",
        GraphPersistentKvState,
        F32,
        TensorElements::LayerAttnState
    ),
    field!(
        "layer_attn_state_score",
        GraphPersistentKvState,
        F32,
        TensorElements::LayerAttnState
    ),
    field!(
        "layer_index_comp_cache",
        GraphPersistentKvState,
        F32,
        TensorElements::LayerIndexCompCache
    ),
    field!(
        "layer_index_state_kv",
        GraphPersistentKvState,
        F32,
        TensorElements::LayerIndexState
    ),
    field!(
        "layer_index_state_score",
        GraphPersistentKvState,
        F32,
        TensorElements::LayerIndexState
    ),
    field!(
        "spec_attn_state_kv",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerAttnState
    ),
    field!(
        "spec_attn_state_score",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerAttnState
    ),
    field!(
        "spec_index_state_kv",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerIndexState
    ),
    field!(
        "spec_index_state_score",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerIndexState
    ),
    field!(
        "spec_prefix1_attn_state_kv",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerAttnState
    ),
    field!(
        "spec_prefix1_attn_state_score",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerAttnState
    ),
    field!(
        "spec_prefix1_index_state_kv",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerIndexState
    ),
    field!(
        "spec_prefix1_index_state_score",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::LayerIndexState
    ),
    field!(
        "spec_logits",
        GraphSpeculativeFrontierState,
        F32,
        TensorElements::SpecLogits
    ),
    field!(
        "comp_kv_cur",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::CompWidthMax)
    ),
    field!(
        "comp_sc_cur",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::CompWidthMax)
    ),
    field!(
        "indexer_q",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::IndexerQ)
    ),
    field!(
        "indexer_weights",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::IndexerHead)
    ),
    field!(
        "indexer_scores",
        GraphLayerWorkState,
        F32,
        TensorElements::PrefillCompCap
    ),
    field!(
        "comp_mask",
        GraphLayerWorkState,
        F32,
        TensorElements::PrefillCompCap
    ),
    field!(
        "comp_selected",
        GraphLayerWorkState,
        U32,
        TensorElements::PrefillTopK
    ),
    field!(
        "heads",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::QDim)
    ),
    field!(
        "attn_low",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::LowDim)
    ),
    field!(
        "attn_out",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "after_attn_hc",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "ffn_cur",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "ffn_norm",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "shared_gate",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::SharedDim)
    ),
    field!(
        "shared_up",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::SharedDim)
    ),
    field!(
        "shared_mid",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::SharedDim)
    ),
    field!(
        "shared_out",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "router_logits",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Expert)
    ),
    field!(
        "router_probs",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Expert)
    ),
    field!(
        "router_selected",
        GraphLayerWorkState,
        I32,
        TensorElements::Fixed(ModelDim::ExpertUsed)
    ),
    field!(
        "router_weights",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::ExpertUsed)
    ),
    field!(
        "routed_gate",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::RoutedExpertUsedMid)
    ),
    field!(
        "routed_up",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::RoutedExpertUsedMid)
    ),
    field!(
        "routed_mid",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::RoutedExpertUsedMid)
    ),
    field!(
        "routed_down",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::RoutedExpertUsedEmbd)
    ),
    field!(
        "routed_out",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "ffn_out",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "after_ffn_hc",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "output_pre",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::HcPrePost)
    ),
    field!(
        "output_weights",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::HcPrePost)
    ),
    field!(
        "output_embd",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "output_norm",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "logits",
        GraphLayerWorkState,
        F32,
        TensorElements::Fixed(ModelDim::Vocab)
    ),
    field!(
        "mtp_embed",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "mtp_enorm",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "mtp_eproj",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Embd)
    ),
    field!(
        "mtp_eproj_hc",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "mtp_hnorm_hc",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "mtp_hproj_hc",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "mtp_input_hc",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "mtp_state_hc",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "mtp_next_hc",
        GraphMtpState,
        F32,
        TensorElements::Fixed(ModelDim::Hc)
    ),
    field!(
        "mtp_raw_cache",
        GraphMtpState,
        F32,
        TensorElements::RawCache
    ),
    field!(
        "prefill_tokens",
        GraphPrefillBatchState,
        I32,
        TensorElements::Prefill(ModelDim::One)
    ),
    field!(
        "batch_cur_hc",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Hc)
    ),
    field!(
        "batch_next_hc",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Hc)
    ),
    field!(
        "batch_flat_hc",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Hc)
    ),
    field!(
        "batch_hc_mix",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::MixHc)
    ),
    field!(
        "batch_hc_split",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::MixHc)
    ),
    field!(
        "batch_attn_cur",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_attn_norm",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_qr",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::QRank)
    ),
    field!(
        "batch_qr_norm",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::QRank)
    ),
    field!(
        "batch_q",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::QDim)
    ),
    field!(
        "batch_kv_raw",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::HeadDim)
    ),
    field!(
        "batch_kv",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::HeadDim)
    ),
    field!(
        "batch_comp_kv",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::CompWidthMax)
    ),
    field!(
        "batch_comp_sc",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::CompWidthMax)
    ),
    field!(
        "batch_indexer_q",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::IndexerQ)
    ),
    field!(
        "batch_indexer_weights",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::IndexerHead)
    ),
    field!(
        "batch_heads",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::QDim)
    ),
    field!(
        "batch_attn_low",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::LowDim)
    ),
    field!(
        "batch_attn_out",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_group_tmp",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::GroupDim)
    ),
    field!(
        "batch_low_tmp",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::LoraO)
    ),
    field!(
        "batch_after_attn_hc",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Hc)
    ),
    field!(
        "batch_ffn_cur",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_ffn_norm",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_shared_gate",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::SharedDim)
    ),
    field!(
        "batch_shared_up",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::SharedDim)
    ),
    field!(
        "batch_shared_mid",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::SharedDim)
    ),
    field!(
        "batch_shared_out",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_router_logits",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Expert)
    ),
    field!(
        "batch_router_probs",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Expert)
    ),
    field!(
        "batch_router_selected",
        GraphPrefillBatchState,
        I32,
        TensorElements::Prefill(ModelDim::ExpertUsed)
    ),
    field!(
        "batch_router_weights",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::ExpertUsed)
    ),
    field!(
        "batch_routed_gate",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::RoutedExpertUsedMid)
    ),
    field!(
        "batch_routed_up",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::RoutedExpertUsedMid)
    ),
    field!(
        "batch_routed_mid",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::RoutedExpertUsedMid)
    ),
    field!(
        "batch_routed_down",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::RoutedExpertUsedEmbd)
    ),
    field!(
        "batch_routed_out",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "batch_ffn_out",
        GraphPrefillBatchState,
        F32,
        TensorElements::Prefill(ModelDim::Embd)
    ),
    field!(
        "directional_steering_dirs",
        GraphOptionalControlState,
        F32,
        TensorElements::External
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandBoundarySpec {
    pub name: &'static str,
    pub c_function: &'static str,
    pub begin_end_min: u32,
    pub synchronize_on_failure: bool,
}

macro_rules! boundary {
    ($name:literal, $func:literal, $min:literal, $sync:literal) => {
        CommandBoundarySpec {
            name: $name,
            c_function: $func,
            begin_end_min: $min,
            synchronize_on_failure: $sync,
        }
    };
}

pub const COMMAND_BOUNDARIES: &[CommandBoundarySpec] = &[
    boundary!(
        "optional_indexer_stage_profile",
        "metal_graph_indexer_stage_profile_boundary",
        1,
        false
    ),
    boundary!(
        "optional_layer_stage_profile",
        "metal_graph_layer_stage_profile_boundary",
        1,
        false
    ),
    boundary!(
        "optional_q_stage_profile",
        "metal_graph_q_stage_profile_boundary",
        1,
        false
    ),
    boundary!(
        "prefill_kernel_warmup",
        "metal_graph_warmup_prefill_kernels",
        1,
        false
    ),
    boundary!(
        "decode_token_logits",
        "metal_graph_eval_token_raw_swa",
        1,
        true
    ),
    boundary!(
        "decode_token_top1",
        "metal_graph_eval_token_raw_swa_top",
        1,
        true
    ),
    boundary!("mtp_draft", "metal_graph_eval_mtp_draft_from_hc", 1, true),
    boundary!(
        "layer_major_prefill",
        "metal_graph_prefill_layer_major",
        1,
        true
    ),
    boundary!(
        "prefill_row_logits",
        "metal_graph_prefill_batch_row_logits",
        1,
        true
    ),
    boundary!(
        "chunked_prefill",
        "metal_graph_prefill_chunked_range",
        1,
        true
    ),
    boundary!("mtp_suffix_tops", "metal_graph_verify_suffix_tops", 2, true),
    boundary!(
        "mtp_decode2_exact",
        "metal_graph_verify_decode2_exact",
        3,
        true
    ),
    boundary!("spec_frontier_snapshot", "spec_frontier_snapshot", 1, true),
    boundary!("spec_frontier_restore", "spec_frontier_restore", 1, true),
    boundary!(
        "spec_frontier_commit_prefix1",
        "spec_frontier_commit_prefix1",
        1,
        true
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphPlanError<'a> {
    MissingOperation(&'a str),
    MissingTensorField(&'a str),
    MissingLayer {
        field: &'static str,
    },
    InvalidLayer {
        field: &'static str,
        layer: usize,
    },
    ExternalTensorSize {
        field: &'static str,
    },
    TensorSizeMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
}

pub fn operation_facade(name: &str) -> Option<BackendFacade> {
    BACKEND_OPERATIONS
        .iter()
        .find(|op| op.name == name)
        .map(|op| op.facade)
}

pub fn tensor_spec(name: &str) -> Option<&'static GraphTensorSpec> {
    GRAPH_TENSOR_FIELDS.iter().find(|field| field.name == name)
}

pub fn command_boundary(c_function: &str) -> Option<&'static CommandBoundarySpec> {
    COMMAND_BOUNDARIES
        .iter()
        .find(|boundary| boundary.c_function == c_function)
}

pub fn validate_required_operations<'a>(required: &'a [&'a str]) -> Result<(), GraphPlanError<'a>> {
    for name in required {
        if operation_facade(name).is_none() {
            return Err(GraphPlanError::MissingOperation(name));
        }
    }
    Ok(())
}

pub fn validate_required_tensors<'a>(required: &'a [&'a str]) -> Result<(), GraphPlanError<'a>> {
    for name in required {
        if tensor_spec(name).is_none() {
            return Err(GraphPlanError::MissingTensorField(name));
        }
    }
    Ok(())
}

pub fn validate_tensor_byte_len(
    spec: GraphTensorSpec,
    plan: GraphPlan,
    dims: ModelGraphDims,
    layer: Option<usize>,
    actual: u64,
) -> Result<(), GraphPlanError<'static>> {
    match spec.byte_len(plan, dims, layer)? {
        TensorByteLen::Known(expected) if expected == actual => Ok(()),
        TensorByteLen::Known(expected) => Err(GraphPlanError::TensorSizeMismatch {
            field: spec.name,
            expected,
            actual,
        }),
        TensorByteLen::External => Err(GraphPlanError::ExternalTensorSize { field: spec.name }),
    }
}

fn dim_elements(dim: ModelDim, dims: ModelGraphDims) -> u64 {
    match dim {
        ModelDim::One => 1,
        ModelDim::Hc => u64::from(N_HC) * u64::from(N_EMBD),
        ModelDim::MixHc => 2 * u64::from(N_HC) + u64::from(N_HC) * u64::from(N_HC),
        ModelDim::HcPrePost => u64::from(N_HC),
        ModelDim::HcComb => u64::from(N_HC) * u64::from(N_HC),
        ModelDim::Embd => u64::from(N_EMBD),
        ModelDim::QRank => dims.q_rank,
        ModelDim::QDim => u64::from(N_HEAD) * u64::from(N_HEAD_DIM),
        ModelDim::HeadDim => u64::from(N_HEAD_DIM),
        ModelDim::CompWidthMax => 2 * u64::from(N_HEAD_DIM.max(N_INDEXER_HEAD_DIM)),
        ModelDim::IndexerQ => u64::from(N_INDEXER_HEAD) * u64::from(N_INDEXER_HEAD_DIM),
        ModelDim::IndexerHead => u64::from(N_INDEXER_HEAD),
        ModelDim::LowDim => u64::from(N_OUT_GROUP) * u64::from(N_LORA_O),
        ModelDim::GroupDim => u64::from(N_HEAD_DIM) * u64::from(N_HEAD / N_OUT_GROUP),
        ModelDim::LoraO => u64::from(N_LORA_O),
        ModelDim::SharedDim => dims.shared_dim,
        ModelDim::Expert => u64::from(N_EXPERT),
        ModelDim::ExpertUsed => u64::from(N_EXPERT_USED),
        ModelDim::RoutedExpertUsedMid => u64::from(N_EXPERT_USED) * dims.routed_mid_dim,
        ModelDim::RoutedExpertUsedEmbd => u64::from(N_EXPERT_USED) * u64::from(N_EMBD),
        ModelDim::Vocab => dims.vocab_dim,
    }
}

fn layer_attn_state_elements(
    field: &'static str,
    layer: usize,
) -> Result<u64, GraphPlanError<'static>> {
    match layer_compression(layer).ok_or(GraphPlanError::InvalidLayer { field, layer })? {
        LayerCompression::Dense => Ok(0),
        LayerCompression::Ratio4 => {
            let coeff = 2;
            Ok(u64::from(coeff * N_HEAD_DIM) * u64::from(coeff * 4))
        }
        LayerCompression::Ratio128 => {
            let coeff = 1;
            Ok(u64::from(coeff * N_HEAD_DIM) * u64::from(coeff * 128))
        }
    }
}

fn layer_index_state_elements(
    field: &'static str,
    layer: usize,
) -> Result<u64, GraphPlanError<'static>> {
    match layer_compression(layer).ok_or(GraphPlanError::InvalidLayer { field, layer })? {
        LayerCompression::Ratio4 => {
            let coeff = 2;
            Ok(u64::from(coeff * N_INDEXER_HEAD_DIM) * u64::from(coeff * 4))
        }
        _ => Ok(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m102_plan_cases_match_oracle() {
        assert_eq!(M102_PLAN_CASES.len(), M102_PLAN_CASE_ORACLE.len());
        for (actual, oracle) in M102_PLAN_CASES.iter().zip(M102_PLAN_CASE_ORACLE) {
            assert_eq!(*actual, oracle.expected_plan(), "{}", oracle.name);
        }
    }

    #[test]
    fn inventory_counts_match_m102_oracle() {
        assert_eq!(BACKEND_OPERATIONS.len(), 81);
        assert_eq!(GRAPH_TENSOR_FIELDS.len(), 113);
        assert_eq!(COMMAND_BOUNDARIES.len(), 15);
    }

    #[test]
    fn inventory_names_are_unique() {
        assert_unique_backend_ops();
        assert_unique_tensor_fields();
        assert_unique_boundaries();
    }

    #[test]
    fn required_operation_validator_fails_closed() {
        let err =
            validate_required_operations(&["ds4_gpu_init", "ds4_gpu_missing_tensor"]).unwrap_err();
        assert_eq!(
            err,
            GraphPlanError::MissingOperation("ds4_gpu_missing_tensor")
        );
    }

    #[test]
    fn tensor_size_mismatch_names_field() {
        let plan = GraphPlan::for_context(32768, 32768, false);
        let spec = *tensor_spec("cur_hc").expect("cur_hc spec");
        let expected = u64::from(N_HC) * u64::from(N_EMBD) * BYTES_F32;
        let err =
            validate_tensor_byte_len(spec, plan, ModelGraphDims::DS4_FLASH, None, expected + 4)
                .unwrap_err();
        assert_eq!(
            err,
            GraphPlanError::TensorSizeMismatch {
                field: "cur_hc",
                expected,
                actual: expected + 4,
            }
        );
    }

    #[test]
    fn layer_size_formulas_cover_ratio_families() {
        let plan = GraphPlan::for_context(32768, 32768, true);
        let raw = *tensor_spec("layer_raw_cache").expect("raw cache spec");
        assert_eq!(
            raw.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(0)),
            Ok(TensorByteLen::Known(
                2304 * u64::from(N_HEAD_DIM) * BYTES_F32
            ))
        );
        let attn = *tensor_spec("layer_attn_comp_cache").expect("attn comp spec");
        assert_eq!(
            attn.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(2)),
            Ok(TensorByteLen::Known(
                8194 * u64::from(N_HEAD_DIM) * BYTES_F32
            ))
        );
        assert_eq!(
            attn.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(3)),
            Ok(TensorByteLen::Known(
                258 * u64::from(N_HEAD_DIM) * BYTES_F32
            ))
        );
        let index = *tensor_spec("layer_index_comp_cache").expect("index comp spec");
        assert_eq!(
            index.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(2)),
            Ok(TensorByteLen::Known(
                8194 * u64::from(N_INDEXER_HEAD_DIM) * BYTES_F32
            ))
        );
        assert_eq!(
            index.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(3)),
            Ok(TensorByteLen::Known(0))
        );
    }

    #[test]
    fn compression_state_size_formulas_cover_ratio_families() {
        let plan = GraphPlan::for_context(32768, 32768, true);
        let attn_state = *tensor_spec("layer_attn_state_kv").expect("attn state spec");
        assert_eq!(
            attn_state.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(0)),
            Ok(TensorByteLen::Known(0))
        );
        assert_eq!(
            attn_state.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(2)),
            Ok(TensorByteLen::Known((2 * 512) * (2 * 4) * BYTES_F32))
        );
        assert_eq!(
            attn_state.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(3)),
            Ok(TensorByteLen::Known(512 * 128 * BYTES_F32))
        );

        let index_state = *tensor_spec("spec_index_state_kv").expect("index state spec");
        assert_eq!(
            index_state.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(2)),
            Ok(TensorByteLen::Known((2 * 128) * (2 * 4) * BYTES_F32))
        );
        assert_eq!(
            index_state.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(3)),
            Ok(TensorByteLen::Known(0))
        );
    }

    fn assert_unique_backend_ops() {
        for (i, left) in BACKEND_OPERATIONS.iter().enumerate() {
            for right in &BACKEND_OPERATIONS[i + 1..] {
                assert_ne!(left.name, right.name, "duplicate backend operation");
            }
        }
    }

    fn assert_unique_tensor_fields() {
        for (i, left) in GRAPH_TENSOR_FIELDS.iter().enumerate() {
            for right in &GRAPH_TENSOR_FIELDS[i + 1..] {
                assert_ne!(left.name, right.name, "duplicate graph tensor field");
            }
        }
    }

    fn assert_unique_boundaries() {
        for (i, left) in COMMAND_BOUNDARIES.iter().enumerate() {
            for right in &COMMAND_BOUNDARIES[i + 1..] {
                assert_ne!(
                    left.c_function, right.c_function,
                    "duplicate command boundary"
                );
            }
        }
    }
}
