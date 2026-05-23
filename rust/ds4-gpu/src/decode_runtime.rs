//! No-execute decode runtime-state bridge.
//!
//! This module binds the M10.5c2 graph-state allocation plan, M10.5c1 weight
//! roles, and M10.5c4a facade trace into a resolvable runtime-handle map before
//! any backend kernels are launched.

use crate::decode_backend::DEFAULT_DECODE_FACADE_OPERATIONS;
use crate::graph_plan::{
    layer_compression, GraphPlan, LayerCompression, N_HASH_LAYER, N_INDEXER_TOP_K, N_LAYER,
};
use crate::graph_state::{GraphStateAllocation, GraphStateFieldPlan, DECODE_GRAPH_STATE_FIELDS};

pub const DECODE_RUNTIME_SCHEMA: &str = "ds4.decode_runtime.v1";
pub const DECODE_RUNTIME_SCOPE: &str = "dry_run_bridge";
pub const DECODE_RUNTIME_CASE: &str = "ctx32768_mtp_off";
pub const WEIGHT_SLICE_LAYERS: &[usize] = &[0, 2, 3];

pub const BASE_WEIGHT_FIELDS: &[&str] = &[
    "token_embd",
    "output_hc_fn",
    "output_hc_scale",
    "output_hc_base",
    "output_norm",
    "output",
];

pub const COMMON_LAYER_WEIGHT_FIELDS: &[&str] = &[
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm",
    "attn_q_a",
    "attn_q_a_norm",
    "attn_q_b",
    "attn_kv",
    "attn_kv_a_norm",
    "attn_sinks",
    "attn_output_a",
    "attn_output_b",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "ffn_norm",
    "ffn_gate_inp",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
];

pub const COMPRESSED_LAYER_WEIGHT_FIELDS: &[&str] = &[
    "attn_compressor_ape",
    "attn_compressor_kv",
    "attn_compressor_gate",
    "attn_compressor_norm",
];

pub const RATIO4_INDEXER_WEIGHT_FIELDS: &[&str] = &[
    "indexer_attn_q_b",
    "indexer_proj",
    "indexer_compressor_ape",
    "indexer_compressor_kv",
    "indexer_compressor_gate",
    "indexer_compressor_norm",
];

pub const OPTIONAL_LAYER_WEIGHT_FIELDS: &[&str] = &["ffn_exp_probs_b"];
pub const HASH_LAYER_WEIGHT_FIELDS: &[&str] = &["ffn_gate_tid2eid"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeRuntimeHandle {
    pub field: &'static str,
    pub layer: Option<usize>,
    pub allocation: GraphStateAllocation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeRuntimeSummary {
    pub logical_handles: u32,
    pub initial_owned_allocations: u32,
    pub views: u32,
    pub lazy_owned: u32,
    pub external_inputs: u32,
    pub initial_layer_counters: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeLayerCounters {
    pub layer: usize,
    pub compression: LayerCompression,
    pub layer_comp_cap: u32,
    pub layer_n_comp: u32,
    pub layer_n_index_comp: u32,
    pub indexer_top_k: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeArgSource {
    State,
    Weight,
    External,
}

impl RuntimeArgSource {
    pub const fn name(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::Weight => "weight",
            Self::External => "external",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FacadeTensorArgBinding {
    pub operation: &'static str,
    pub arg: &'static str,
    pub source: RuntimeArgSource,
    pub candidates: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightPresence {
    RequiredPresent,
    Optional,
    ExpectedAbsent,
}

impl WeightPresence {
    pub const fn name(self) -> &'static str {
        match self {
            Self::RequiredPresent => "required_present",
            Self::Optional => "optional",
            Self::ExpectedAbsent => "expected_absent",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightRequirement {
    pub scope: &'static str,
    pub layer: Option<usize>,
    pub field: &'static str,
    pub presence: WeightPresence,
}

pub const FACADE_TENSOR_ARG_BINDINGS: &[FacadeTensorArgBinding] = &[
    bind_state("ds4_gpu_embed_token_hc_tensor", "out_hc", &["cur_hc"]),
    bind_state("ds4_gpu_rms_norm_plain_tensor", "out", &["flat_hc"]),
    bind_state(
        "ds4_gpu_rms_norm_plain_tensor",
        "x",
        &["cur_hc", "after_attn_hc"],
    ),
    bind_state(
        "ds4_gpu_matmul_f16_tensor",
        "out",
        &[
            "hc_mix",
            "comp_kv_cur",
            "comp_sc_cur",
            "indexer_q",
            "indexer_weights",
            "router_logits",
            "output_pre",
        ],
    ),
    bind_state(
        "ds4_gpu_matmul_f16_tensor",
        "x",
        &["flat_hc", "attn_norm", "qr_norm", "ffn_norm"],
    ),
    bind_state(
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        "out",
        &["attn_cur", "ffn_cur"],
    ),
    bind_state(
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        "norm_out",
        &["attn_norm", "ffn_norm"],
    ),
    bind_state(
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        "split",
        &["hc_split"],
    ),
    bind_state(
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        "mix",
        &["hc_mix"],
    ),
    bind_state(
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        "residual_hc",
        &["cur_hc", "after_attn_hc"],
    ),
    bind_state("ds4_gpu_rms_norm_weight_tensor", "out", &["output_norm"]),
    bind_state("ds4_gpu_rms_norm_weight_tensor", "x", &["output_embd"]),
    bind_state(
        "ds4_gpu_matmul_q8_0_tensor",
        "out",
        &["qr", "kv_raw", "q", "logits"],
    ),
    bind_state(
        "ds4_gpu_matmul_q8_0_tensor",
        "x",
        &["attn_norm", "qr_norm", "output_norm"],
    ),
    bind_state(
        "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
        "q_out",
        &["qr_norm"],
    ),
    bind_state("ds4_gpu_dsv4_qkv_rms_norm_rows_tensor", "q", &["qr"]),
    bind_state("ds4_gpu_dsv4_qkv_rms_norm_rows_tensor", "kv_out", &["kv"]),
    bind_state("ds4_gpu_dsv4_qkv_rms_norm_rows_tensor", "kv", &["kv_raw"]),
    bind_state("ds4_gpu_head_rms_norm_tensor", "x", &["q"]),
    bind_state(
        "ds4_gpu_rope_tail_tensor",
        "x",
        &["q", "kv", "indexer_q", "heads"],
    ),
    bind_state("ds4_gpu_kv_fp8_store_raw_tensor", "kv", &["kv"]),
    bind_state(
        "ds4_gpu_kv_fp8_store_raw_tensor",
        "raw_cache",
        &["layer_raw_cache"],
    ),
    bind_state("ds4_gpu_matmul_f16_pair_tensor", "out_a", &["comp_kv_cur"]),
    bind_state("ds4_gpu_matmul_f16_pair_tensor", "out_b", &["comp_sc_cur"]),
    bind_state("ds4_gpu_matmul_f16_pair_tensor", "x", &["attn_norm"]),
    bind_state(
        "ds4_gpu_compressor_update_tensor",
        "kv_cur",
        &["comp_kv_cur"],
    ),
    bind_state(
        "ds4_gpu_compressor_update_tensor",
        "sc_cur",
        &["comp_sc_cur"],
    ),
    bind_state(
        "ds4_gpu_compressor_update_tensor",
        "state_kv",
        &["layer_attn_state_kv", "layer_index_state_kv"],
    ),
    bind_state(
        "ds4_gpu_compressor_update_tensor",
        "state_score",
        &["layer_attn_state_score", "layer_index_state_score"],
    ),
    bind_state(
        "ds4_gpu_compressor_update_tensor",
        "comp_cache",
        &["layer_attn_comp_cache", "layer_index_comp_cache"],
    ),
    bind_state(
        "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
        "x",
        &["layer_attn_comp_cache"],
    ),
    bind_state(
        "ds4_gpu_dsv4_indexer_qat_tensor",
        "x",
        &["layer_index_comp_cache", "indexer_q"],
    ),
    bind_state(
        "ds4_gpu_indexer_score_one_tensor",
        "scores",
        &["indexer_scores"],
    ),
    bind_state("ds4_gpu_indexer_score_one_tensor", "q", &["indexer_q"]),
    bind_state(
        "ds4_gpu_indexer_score_one_tensor",
        "weights",
        &["indexer_weights"],
    ),
    bind_state(
        "ds4_gpu_indexer_score_one_tensor",
        "index_comp",
        &["layer_index_comp_cache"],
    ),
    bind_state(
        "ds4_gpu_indexer_topk_tensor",
        "selected",
        &["comp_selected"],
    ),
    bind_state("ds4_gpu_indexer_topk_tensor", "scores", &["indexer_scores"]),
    bind_state(
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        "heads",
        &["heads"],
    ),
    bind_state(
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        "q",
        &["q"],
    ),
    bind_state(
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        "raw_kv",
        &["layer_raw_cache"],
    ),
    bind_state(
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        "comp_kv",
        &["layer_attn_comp_cache"],
    ),
    bind_state(
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        "topk",
        &["comp_selected"],
    ),
    bind_state("ds4_gpu_attention_decode_heads_tensor", "heads", &["heads"]),
    bind_state("ds4_gpu_attention_decode_heads_tensor", "q", &["q"]),
    bind_state(
        "ds4_gpu_attention_decode_heads_tensor",
        "raw_kv",
        &["layer_raw_cache"],
    ),
    bind_state(
        "ds4_gpu_attention_decode_heads_tensor",
        "comp_kv",
        &["layer_attn_comp_cache"],
    ),
    bind_state(
        "ds4_gpu_attention_decode_heads_tensor",
        "comp_mask",
        &["comp_mask"],
    ),
    bind_state(
        "ds4_gpu_attention_output_low_q8_tensor",
        "low",
        &["attn_low"],
    ),
    bind_state(
        "ds4_gpu_attention_output_low_q8_tensor",
        "heads",
        &["heads"],
    ),
    bind_state(
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "out_hc",
        &["after_attn_hc", "after_ffn_hc"],
    ),
    bind_state(
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "block_out",
        &["attn_out", "shared_out"],
    ),
    bind_state(
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "x",
        &["attn_low", "shared_mid"],
    ),
    bind_state(
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "residual_hc",
        &["cur_hc", "after_attn_hc"],
    ),
    bind_state(
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "split",
        &["hc_split"],
    ),
    bind_state(
        "ds4_gpu_router_select_tensor",
        "selected",
        &["router_selected"],
    ),
    bind_state(
        "ds4_gpu_router_select_tensor",
        "weights",
        &["router_weights"],
    ),
    bind_state("ds4_gpu_router_select_tensor", "probs", &["router_probs"]),
    bind_state("ds4_gpu_router_select_tensor", "logits", &["router_logits"]),
    bind_state("ds4_gpu_routed_moe_one_tensor", "out", &["routed_out"]),
    bind_state("ds4_gpu_routed_moe_one_tensor", "gate", &["routed_gate"]),
    bind_state("ds4_gpu_routed_moe_one_tensor", "up", &["routed_up"]),
    bind_state("ds4_gpu_routed_moe_one_tensor", "mid", &["routed_mid"]),
    bind_weight(
        "ds4_gpu_routed_moe_one_tensor",
        "experts",
        &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
    ),
    bind_state(
        "ds4_gpu_routed_moe_one_tensor",
        "selected",
        &["router_selected"],
    ),
    bind_state(
        "ds4_gpu_routed_moe_one_tensor",
        "weights",
        &["router_weights"],
    ),
    bind_state("ds4_gpu_routed_moe_one_tensor", "x", &["ffn_norm"]),
    bind_state(
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        "gate",
        &["shared_gate"],
    ),
    bind_state(
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        "up",
        &["shared_up"],
    ),
    bind_state(
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        "mid",
        &["shared_mid"],
    ),
    bind_state(
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        "x",
        &["ffn_norm"],
    ),
    bind_state(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "out_hc",
        &["after_ffn_hc"],
    ),
    bind_state(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "shared_out",
        &["shared_out"],
    ),
    bind_state(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "shared_mid",
        &["shared_mid"],
    ),
    bind_state(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "routed_out",
        &["routed_out"],
    ),
    bind_state(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "residual_hc",
        &["after_attn_hc"],
    ),
    bind_state(
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "split",
        &["hc_split"],
    ),
    bind_state(
        "ds4_gpu_output_hc_weights_tensor",
        "out",
        &["output_weights"],
    ),
    bind_state("ds4_gpu_output_hc_weights_tensor", "pre", &["output_pre"]),
    bind_state("ds4_gpu_hc_weighted_sum_tensor", "out", &["output_embd"]),
    bind_state("ds4_gpu_hc_weighted_sum_tensor", "residual_hc", &["cur_hc"]),
    bind_state(
        "ds4_gpu_hc_weighted_sum_tensor",
        "weights",
        &["output_weights"],
    ),
];

const fn bind_state(
    operation: &'static str,
    arg: &'static str,
    candidates: &'static [&'static str],
) -> FacadeTensorArgBinding {
    FacadeTensorArgBinding {
        operation,
        arg,
        source: RuntimeArgSource::State,
        candidates,
    }
}

const fn bind_weight(
    operation: &'static str,
    arg: &'static str,
    candidates: &'static [&'static str],
) -> FacadeTensorArgBinding {
    FacadeTensorArgBinding {
        operation,
        arg,
        source: RuntimeArgSource::Weight,
        candidates,
    }
}

pub fn for_each_runtime_handle(
    plan: GraphPlan,
    dims: crate::graph_plan::ModelGraphDims,
    mut f: impl FnMut(DecodeRuntimeHandle),
) {
    for field in DECODE_GRAPH_STATE_FIELDS {
        match field.instances {
            crate::graph_state::GraphTensorInstances::Single => {
                emit_handle(*field, plan, dims, None, &mut f);
            }
            crate::graph_state::GraphTensorInstances::PerLayer => {
                for layer in 0..N_LAYER {
                    emit_handle(*field, plan, dims, Some(layer), &mut f);
                }
            }
        }
    }
}

pub fn runtime_summary(
    plan: GraphPlan,
    dims: crate::graph_plan::ModelGraphDims,
) -> DecodeRuntimeSummary {
    let mut summary = DecodeRuntimeSummary {
        logical_handles: 0,
        initial_owned_allocations: 0,
        views: 0,
        lazy_owned: 0,
        external_inputs: 0,
        initial_layer_counters: N_LAYER as u32,
    };
    for_each_runtime_handle(plan, dims, |handle| {
        summary.logical_handles += 1;
        if handle.allocation.initially_allocated {
            summary.initial_owned_allocations += 1;
        }
        match handle.allocation.storage {
            crate::graph_state::GraphTensorStorage::View { .. } => summary.views += 1,
            crate::graph_state::GraphTensorStorage::LazyOwned => summary.lazy_owned += 1,
            crate::graph_state::GraphTensorStorage::External => summary.external_inputs += 1,
            crate::graph_state::GraphTensorStorage::Owned => {}
        }
    });
    summary
}

fn emit_handle(
    field: GraphStateFieldPlan,
    plan: GraphPlan,
    dims: crate::graph_plan::ModelGraphDims,
    layer: Option<usize>,
    f: &mut impl FnMut(DecodeRuntimeHandle),
) {
    let allocation = field
        .initial_allocation(plan, dims, layer)
        .expect("runtime bridge field is backed by graph plan");
    f(DecodeRuntimeHandle {
        field: field.name,
        layer,
        allocation,
    });
}

pub fn layer_counters(plan: GraphPlan, layer: usize) -> Option<RuntimeLayerCounters> {
    let compression = layer_compression(layer)?;
    Some(RuntimeLayerCounters {
        layer,
        compression,
        layer_comp_cap: plan.layer_comp_cap(compression),
        layer_n_comp: 0,
        layer_n_index_comp: 0,
        indexer_top_k: N_INDEXER_TOP_K,
    })
}

pub fn facade_tensor_arg_binding(
    operation: &str,
    arg: &str,
) -> Option<&'static FacadeTensorArgBinding> {
    FACADE_TENSOR_ARG_BINDINGS
        .iter()
        .find(|binding| binding.operation == operation && binding.arg == arg)
}

pub fn layer_weight_presence(layer: usize, field: &str) -> Option<WeightPresence> {
    let compression = layer_compression(layer)?;
    if contains(COMMON_LAYER_WEIGHT_FIELDS, field) {
        return Some(WeightPresence::RequiredPresent);
    }
    if contains(COMPRESSED_LAYER_WEIGHT_FIELDS, field) {
        return Some(if compression == LayerCompression::Dense {
            WeightPresence::ExpectedAbsent
        } else {
            WeightPresence::RequiredPresent
        });
    }
    if contains(RATIO4_INDEXER_WEIGHT_FIELDS, field) {
        return Some(if compression == LayerCompression::Ratio4 {
            WeightPresence::RequiredPresent
        } else {
            WeightPresence::ExpectedAbsent
        });
    }
    if contains(HASH_LAYER_WEIGHT_FIELDS, field) {
        return Some(if layer < N_HASH_LAYER as usize {
            WeightPresence::RequiredPresent
        } else {
            WeightPresence::ExpectedAbsent
        });
    }
    if contains(OPTIONAL_LAYER_WEIGHT_FIELDS, field) {
        return Some(WeightPresence::Optional);
    }
    None
}

pub fn default_decode_facade_bindings_cover_table() -> bool {
    for operation in DEFAULT_DECODE_FACADE_OPERATIONS {
        let mut i = 0usize;
        while i < operation.tensor_args.len() {
            if facade_tensor_arg_binding(operation.operation, operation.tensor_args[i]).is_none() {
                return false;
            }
            i += 1;
        }
    }
    true
}

const fn contains(values: &[&str], needle: &str) -> bool {
    let mut i = 0usize;
    while i < values.len() {
        if str_eq(values[i], needle) {
            return true;
        }
        i += 1;
    }
    false
}

const fn str_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut i = 0usize;
    while i < left.len() {
        if left[i] != right[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_plan::ModelGraphDims;

    #[test]
    fn runtime_handles_match_graph_state_shape() {
        let plan = GraphPlan::for_context(32768, 32768, false);
        let summary = runtime_summary(plan, ModelGraphDims::DS4_FLASH);
        assert_eq!(summary.logical_handles, 349);
        assert_eq!(summary.initial_owned_allocations, 272);
        assert_eq!(summary.views, 3);
        assert_eq!(summary.lazy_owned, 1);
        assert_eq!(summary.external_inputs, 1);
        assert_eq!(summary.initial_layer_counters, N_LAYER as u32);
    }

    #[test]
    fn runtime_counters_cover_dense_ratio4_and_ratio128_layers() {
        let plan = GraphPlan::for_context(32768, 32768, false);
        let dense = layer_counters(plan, 0).expect("dense");
        assert_eq!(dense.compression, LayerCompression::Dense);
        assert_eq!(dense.layer_comp_cap, 0);
        let ratio4 = layer_counters(plan, 2).expect("ratio4");
        assert_eq!(ratio4.compression, LayerCompression::Ratio4);
        assert_eq!(ratio4.layer_comp_cap, 8194);
        assert_eq!(ratio4.layer_n_comp, 0);
        assert_eq!(ratio4.layer_n_index_comp, 0);
        let ratio128 = layer_counters(plan, 3).expect("ratio128");
        assert_eq!(ratio128.compression, LayerCompression::Ratio128);
        assert_eq!(ratio128.layer_comp_cap, 258);
    }

    #[test]
    fn weight_presence_matches_slice_layers() {
        assert_eq!(
            layer_weight_presence(0, "attn_compressor_ape"),
            Some(WeightPresence::ExpectedAbsent)
        );
        assert_eq!(
            layer_weight_presence(2, "attn_compressor_ape"),
            Some(WeightPresence::RequiredPresent)
        );
        assert_eq!(
            layer_weight_presence(2, "indexer_proj"),
            Some(WeightPresence::RequiredPresent)
        );
        assert_eq!(
            layer_weight_presence(3, "indexer_proj"),
            Some(WeightPresence::ExpectedAbsent)
        );
        assert_eq!(
            layer_weight_presence(2, "ffn_gate_tid2eid"),
            Some(WeightPresence::RequiredPresent)
        );
        assert_eq!(
            layer_weight_presence(3, "ffn_gate_tid2eid"),
            Some(WeightPresence::ExpectedAbsent)
        );
        assert_eq!(
            layer_weight_presence(0, "ffn_exp_probs_b"),
            Some(WeightPresence::Optional)
        );
    }

    #[test]
    fn facade_tensor_args_are_bound() {
        assert!(default_decode_facade_bindings_cover_table());
        assert_eq!(
            facade_tensor_arg_binding("ds4_gpu_routed_moe_one_tensor", "experts")
                .map(|binding| binding.source),
            Some(RuntimeArgSource::Weight)
        );
    }

    #[test]
    fn facade_tensor_arg_bindings_have_unique_keys() {
        for i in 0..FACADE_TENSOR_ARG_BINDINGS.len() {
            for j in i + 1..FACADE_TENSOR_ARG_BINDINGS.len() {
                let left = FACADE_TENSOR_ARG_BINDINGS[i];
                let right = FACADE_TENSOR_ARG_BINDINGS[j];
                assert!(
                    left.operation != right.operation || left.arg != right.arg,
                    "duplicate binding {}:{}",
                    left.operation,
                    left.arg
                );
            }
        }
    }
}
