//! Dry-run one-token decode execution trace.
//!
//! This module expands the M10.5b decode plan into the default M10.5c3 facade
//! call tape and cache-counter transitions without calling backend kernels.

use crate::decode_backend::{
    DecodeFacadeOperation, ExistingDecodeOperation, DEFAULT_DECODE_FACADE_OPERATIONS,
    EXISTING_DECODE_OPERATIONS,
};
use crate::decode_plan::{
    compressed_rows_after, compressed_rows_before, DecodePlanCaseOracle, DecodeTokenPlan,
    M105B_DECODE_CASE_ORACLE,
};
use crate::graph_plan::{layer_compression, LayerCompression, N_INDEXER_TOP_K, N_LAYER};

pub const DECODE_TRACE_SCHEMA: &str = "ds4.decode_trace.v1";
pub const DECODE_TRACE_SCOPE: &str = "dry_run";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeTraceEventKind {
    Stage,
    Facade,
    Existing,
    State,
}

impl DecodeTraceEventKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Stage => "stage",
            Self::Facade => "facade",
            Self::Existing => "existing",
            Self::State => "state",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeTraceState {
    pub compression: &'static str,
    pub pos: Option<u32>,
    pub raw_row: Option<u32>,
    pub n_raw: Option<u32>,
    pub raw_start: Option<u32>,
    pub comp_before: Option<u32>,
    pub comp_after: Option<u32>,
    pub index_before: Option<u32>,
    pub index_after: Option<u32>,
    pub emit_compressed_row: bool,
    pub indexed_attention: bool,
    pub attention_operation: &'static str,
}

impl DecodeTraceState {
    pub const NONE: Self = Self {
        compression: "none",
        pos: None,
        raw_row: None,
        n_raw: None,
        raw_start: None,
        comp_before: None,
        comp_after: None,
        index_before: None,
        index_after: None,
        emit_compressed_row: false,
        indexed_attention: false,
        attention_operation: "",
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeTraceEvent {
    pub index: u32,
    pub kind: DecodeTraceEventKind,
    pub stage: &'static str,
    pub layer: Option<u32>,
    pub operation: &'static str,
    pub method: &'static str,
    pub tensor_args: &'static [&'static str],
    pub state: DecodeTraceState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeTraceSummary {
    pub events: u32,
    pub stage_markers: u32,
    pub facade_calls: u32,
    pub existing_calls: u32,
    pub state_events: u32,
    pub layers: u32,
    pub dense_layers: u32,
    pub ratio4_layers: u32,
    pub ratio128_layers: u32,
    pub compressed_emit_layers: u32,
    pub indexed_attention_layers: u32,
    pub split_flushes: u32,
    pub output_head_calls: u32,
    pub read_logits_calls: u32,
    pub synchronize_on_failure_calls: u32,
}

impl DecodeTraceSummary {
    pub const EMPTY: Self = Self {
        events: 0,
        stage_markers: 0,
        facade_calls: 0,
        existing_calls: 0,
        state_events: 0,
        layers: 0,
        dense_layers: 0,
        ratio4_layers: 0,
        ratio128_layers: 0,
        compressed_emit_layers: 0,
        indexed_attention_layers: 0,
        split_flushes: 0,
        output_head_calls: 0,
        read_logits_calls: 0,
        synchronize_on_failure_calls: 0,
    };

    fn observe(&mut self, event: DecodeTraceEvent) {
        self.events += 1;
        match event.kind {
            DecodeTraceEventKind::Stage => self.stage_markers += 1,
            DecodeTraceEventKind::Facade => self.facade_calls += 1,
            DecodeTraceEventKind::Existing => self.existing_calls += 1,
            DecodeTraceEventKind::State => self.state_events += 1,
        }
        if event.kind == DecodeTraceEventKind::Existing {
            if event.operation == "ds4_gpu_flush_commands" {
                self.split_flushes += 1;
            } else if event.operation == "ds4_gpu_tensor_read" {
                self.read_logits_calls += 1;
            } else if event.operation == "ds4_gpu_synchronize" {
                self.synchronize_on_failure_calls += 1;
            }
        }
        if event.stage == "output_head" && event.kind == DecodeTraceEventKind::Facade {
            self.output_head_calls += 1;
        }
        if event.kind == DecodeTraceEventKind::State {
            self.layers += 1;
            match event.state.compression {
                "dense" => self.dense_layers += 1,
                "ratio4" => self.ratio4_layers += 1,
                "ratio128" => self.ratio128_layers += 1,
                _ => {}
            }
            if event.state.emit_compressed_row {
                self.compressed_emit_layers += 1;
            }
            if event.state.indexed_attention {
                self.indexed_attention_layers += 1;
            }
        }
    }
}

pub fn decode_trace_case(name: &str) -> Option<DecodePlanCaseOracle> {
    M105B_DECODE_CASE_ORACLE
        .iter()
        .copied()
        .find(|case| case.name == name)
}

pub fn trace_summary(case: DecodePlanCaseOracle) -> DecodeTraceSummary {
    let mut summary = DecodeTraceSummary::EMPTY;
    for_each_trace_event(case, |event| summary.observe(event));
    summary
}

pub fn for_each_trace_event(case: DecodePlanCaseOracle, mut f: impl FnMut(DecodeTraceEvent)) {
    let mut builder = TraceBuilder::new(&mut f);
    let plan = case.computed();

    builder.stage("begin_commands", None);
    builder.existing("begin_commands", None, "ds4_gpu_begin_commands");
    builder.stage("embed_token_hc", None);
    builder.facade("embed_token_hc", None, "ds4_gpu_embed_token_hc_tensor");
    builder.stage("decode_layers", None);
    for layer in 0..N_LAYER {
        emit_layer_trace(&mut builder, case.input.pos, plan, layer);
        if plan.command.flush_after_layer == Some(layer as u32) {
            builder.stage("split_flush", Some(layer as u32));
            builder.existing("split_flush", Some(layer as u32), "ds4_gpu_flush_commands");
        }
    }
    if plan.command.read_logits_after_end {
        builder.stage("output_head", None);
        emit_output_head_trace(&mut builder);
    }
    builder.stage("end_commands", None);
    builder.existing("end_commands", None, "ds4_gpu_end_commands");
    if plan.command.read_logits_after_end {
        builder.stage("read_logits", None);
        builder.existing("read_logits", None, "ds4_gpu_tensor_read");
    }
    if plan.command.synchronize_on_failure {
        builder.stage("synchronize_on_failure", None);
        builder.existing("synchronize_on_failure", None, "ds4_gpu_synchronize");
    }
}

fn emit_layer_trace(
    builder: &mut TraceBuilder<'_, impl FnMut(DecodeTraceEvent)>,
    pos: u32,
    plan: DecodeTokenPlan,
    layer: usize,
) {
    let layer = layer as u32;
    builder.state(
        "decode_layers",
        layer,
        layer_state(pos, plan, layer as usize),
    );
    builder.stage("attn_hc_pre", Some(layer));
    builder.facade("attn_hc_pre", Some(layer), "ds4_gpu_rms_norm_plain_tensor");
    builder.facade("attn_hc_pre", Some(layer), "ds4_gpu_matmul_f16_tensor");
    builder.facade(
        "attn_hc_pre",
        Some(layer),
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
    );
    builder.stage("attn_norm", Some(layer));
    builder.stage("q_path", Some(layer));
    builder.facade("q_path", Some(layer), "ds4_gpu_matmul_q8_0_tensor");
    builder.facade("q_path", Some(layer), "ds4_gpu_matmul_q8_0_tensor");
    builder.facade(
        "q_path",
        Some(layer),
        "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
    );
    builder.facade("q_path", Some(layer), "ds4_gpu_matmul_q8_0_tensor");
    builder.facade("q_path", Some(layer), "ds4_gpu_head_rms_norm_tensor");
    builder.facade("q_path", Some(layer), "ds4_gpu_rope_tail_tensor");
    builder.stage("kv_path", Some(layer));
    builder.facade("kv_path", Some(layer), "ds4_gpu_rope_tail_tensor");
    builder.facade("kv_path", Some(layer), "ds4_gpu_kv_fp8_store_raw_tensor");
    builder.stage("compressor_indexer", Some(layer));
    emit_compressor_indexer_trace(builder, pos, plan, layer);
    builder.stage("attention", Some(layer));
    let attention_operation = layer_state(pos, plan, layer as usize).attention_operation;
    builder.facade("attention", Some(layer), attention_operation);
    builder.facade("attention", Some(layer), "ds4_gpu_rope_tail_tensor");
    builder.stage("attn_output", Some(layer));
    builder.facade(
        "attn_output",
        Some(layer),
        "ds4_gpu_attention_output_low_q8_tensor",
    );
    builder.facade(
        "attn_output",
        Some(layer),
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
    );
    builder.stage("attn_hc_post", Some(layer));
    builder.stage("ffn_hc_pre", Some(layer));
    builder.facade("ffn_hc_pre", Some(layer), "ds4_gpu_rms_norm_plain_tensor");
    builder.facade("ffn_hc_pre", Some(layer), "ds4_gpu_matmul_f16_tensor");
    builder.facade(
        "ffn_hc_pre",
        Some(layer),
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
    );
    builder.stage("ffn_norm", Some(layer));
    builder.stage("router", Some(layer));
    builder.facade("router", Some(layer), "ds4_gpu_matmul_f16_tensor");
    builder.facade("router", Some(layer), "ds4_gpu_router_select_tensor");
    builder.stage("routed_moe", Some(layer));
    builder.facade("routed_moe", Some(layer), "ds4_gpu_routed_moe_one_tensor");
    builder.stage("shared_gate_up", Some(layer));
    builder.facade(
        "shared_gate_up",
        Some(layer),
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
    );
    builder.stage("shared_down", Some(layer));
    builder.facade(
        "shared_down",
        Some(layer),
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
    );
    builder.stage("ffn_hc_post", Some(layer));
}

fn emit_compressor_indexer_trace(
    builder: &mut TraceBuilder<'_, impl FnMut(DecodeTraceEvent)>,
    pos: u32,
    plan: DecodeTokenPlan,
    layer: u32,
) {
    let state = layer_state(pos, plan, layer as usize);
    if state.compression == "dense" {
        return;
    }
    builder.facade(
        "compressor_indexer",
        Some(layer),
        "ds4_gpu_matmul_f16_pair_tensor",
    );
    builder.facade(
        "compressor_indexer",
        Some(layer),
        "ds4_gpu_compressor_update_tensor",
    );
    if state.emit_compressed_row {
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
        );
    }
    if state.compression != "ratio4" {
        return;
    }
    builder.facade(
        "compressor_indexer",
        Some(layer),
        "ds4_gpu_matmul_f16_pair_tensor",
    );
    builder.facade(
        "compressor_indexer",
        Some(layer),
        "ds4_gpu_compressor_update_tensor",
    );
    if state.emit_compressed_row {
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_dsv4_indexer_qat_tensor",
        );
    }
    if state.indexed_attention {
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_matmul_f16_tensor",
        );
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_rope_tail_tensor",
        );
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_dsv4_indexer_qat_tensor",
        );
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_matmul_f16_tensor",
        );
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_indexer_score_one_tensor",
        );
        builder.facade(
            "compressor_indexer",
            Some(layer),
            "ds4_gpu_indexer_topk_tensor",
        );
    }
}

fn emit_output_head_trace(builder: &mut TraceBuilder<'_, impl FnMut(DecodeTraceEvent)>) {
    builder.facade("output_head", None, "ds4_gpu_rms_norm_plain_tensor");
    builder.facade("output_head", None, "ds4_gpu_matmul_f16_tensor");
    builder.facade("output_head", None, "ds4_gpu_output_hc_weights_tensor");
    builder.facade("output_head", None, "ds4_gpu_hc_weighted_sum_tensor");
    builder.facade("output_head", None, "ds4_gpu_rms_norm_weight_tensor");
    builder.facade("output_head", None, "ds4_gpu_matmul_q8_0_tensor");
}

pub fn layer_state(pos: u32, plan: DecodeTokenPlan, layer: usize) -> DecodeTraceState {
    match layer_compression(layer).expect("valid DS4 layer") {
        LayerCompression::Dense => DecodeTraceState {
            compression: "dense",
            pos: Some(pos),
            raw_row: Some(plan.raw_row),
            n_raw: Some(plan.n_raw),
            raw_start: Some(plan.raw_start),
            attention_operation: "ds4_gpu_attention_decode_heads_tensor",
            ..DecodeTraceState::NONE
        },
        LayerCompression::Ratio4 => compressed_state(
            "ratio4",
            pos,
            4,
            plan,
            plan.ratio4_cache.comp_before,
            plan.ratio4_cache.comp_after,
            Some((
                compressed_rows_before(pos, 4),
                compressed_rows_after(pos, 4),
            )),
        ),
        LayerCompression::Ratio128 => compressed_state(
            "ratio128",
            pos,
            128,
            plan,
            plan.ratio128_cache.comp_before,
            plan.ratio128_cache.comp_after,
            None,
        ),
    }
}

fn compressed_state(
    compression: &'static str,
    pos: u32,
    ratio: u32,
    plan: DecodeTokenPlan,
    comp_before: u32,
    comp_after: u32,
    index_transition: Option<(u32, u32)>,
) -> DecodeTraceState {
    debug_assert_eq!(comp_before, compressed_rows_before(pos, ratio));
    debug_assert_eq!(comp_after, compressed_rows_after(pos, ratio));
    let emit_compressed_row = compressed_row_emits(pos, ratio);
    debug_assert_eq!(emit_compressed_row, comp_after != comp_before);
    let indexed_attention = index_transition.is_some() && comp_after > N_INDEXER_TOP_K;
    DecodeTraceState {
        compression,
        pos: Some(pos),
        raw_row: Some(plan.raw_row),
        n_raw: Some(plan.n_raw),
        raw_start: Some(plan.raw_start),
        comp_before: Some(comp_before),
        comp_after: Some(comp_after),
        index_before: index_transition.map(|(before, _)| before),
        index_after: index_transition.map(|(_, after)| after),
        emit_compressed_row,
        indexed_attention,
        attention_operation: if indexed_attention {
            "ds4_gpu_attention_indexed_mixed_batch_heads_tensor"
        } else {
            "ds4_gpu_attention_decode_heads_tensor"
        },
    }
}

fn compressed_row_emits(pos: u32, ratio: u32) -> bool {
    ratio != 0 && pos % ratio == ratio - 1
}

struct TraceBuilder<'a, F>
where
    F: FnMut(DecodeTraceEvent),
{
    next_index: u32,
    f: &'a mut F,
}

impl<'a, F> TraceBuilder<'a, F>
where
    F: FnMut(DecodeTraceEvent),
{
    fn new(f: &'a mut F) -> Self {
        Self { next_index: 0, f }
    }

    fn stage(&mut self, stage: &'static str, layer: Option<u32>) {
        let index = self.next();
        self.emit(DecodeTraceEvent {
            index,
            kind: DecodeTraceEventKind::Stage,
            stage,
            layer,
            operation: "stage_marker",
            method: "",
            tensor_args: &[],
            state: DecodeTraceState::NONE,
        });
    }

    fn state(&mut self, stage: &'static str, layer: u32, state: DecodeTraceState) {
        let index = self.next();
        self.emit(DecodeTraceEvent {
            index,
            kind: DecodeTraceEventKind::State,
            stage,
            layer: Some(layer),
            operation: "layer_state",
            method: "",
            tensor_args: &[],
            state,
        });
    }

    fn facade(&mut self, stage: &'static str, layer: Option<u32>, operation: &'static str) {
        let spec = facade_spec(operation).expect("default decode facade op");
        let index = self.next();
        self.emit(DecodeTraceEvent {
            index,
            kind: DecodeTraceEventKind::Facade,
            stage,
            layer,
            operation: spec.operation,
            method: spec.method,
            tensor_args: spec.tensor_args,
            state: DecodeTraceState::NONE,
        });
    }

    fn existing(&mut self, stage: &'static str, layer: Option<u32>, operation: &'static str) {
        let spec = existing_spec(operation).expect("existing decode op");
        let index = self.next();
        self.emit(DecodeTraceEvent {
            index,
            kind: DecodeTraceEventKind::Existing,
            stage,
            layer,
            operation: spec.operation,
            method: spec.wrapper,
            tensor_args: &[],
            state: DecodeTraceState::NONE,
        });
    }

    fn next(&mut self) -> u32 {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn emit(&mut self, event: DecodeTraceEvent) {
        (self.f)(event);
    }
}

fn facade_spec(operation: &str) -> Option<DecodeFacadeOperation> {
    DEFAULT_DECODE_FACADE_OPERATIONS
        .iter()
        .copied()
        .find(|spec| spec.operation == operation)
}

fn existing_spec(operation: &str) -> Option<ExistingDecodeOperation> {
    EXISTING_DECODE_OPERATIONS
        .iter()
        .copied()
        .find(|spec| spec.operation == operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_cases_match_m105b_oracle_cases() {
        assert_eq!(M105B_DECODE_CASE_ORACLE.len(), 5);
        assert!(decode_trace_case("ratio_emit_boundary").is_some());
        assert!(decode_trace_case("missing").is_none());
    }

    #[test]
    fn trace_summary_preserves_layer_and_command_shape() {
        let first = decode_trace_case("first_token_ctx32768").expect("case");
        let summary = trace_summary(first);
        assert_eq!(summary.layers, N_LAYER as u32);
        assert_eq!(summary.dense_layers, 2);
        assert_eq!(summary.ratio4_layers, 21);
        assert_eq!(summary.ratio128_layers, 20);
        assert_eq!(summary.split_flushes, 1);
        assert_eq!(summary.output_head_calls, 6);
        assert_eq!(summary.read_logits_calls, 1);
        assert_eq!(summary.synchronize_on_failure_calls, 1);

        let no_logits = decode_trace_case("no_logits_no_split").expect("case");
        let summary = trace_summary(no_logits);
        assert_eq!(summary.split_flushes, 0);
        assert_eq!(summary.output_head_calls, 0);
        assert_eq!(summary.read_logits_calls, 0);
    }

    #[test]
    fn cache_transition_edges_match_decode_plan_cases() {
        let ratio_emit_case = decode_trace_case("ratio_emit_boundary").expect("case");
        let ratio_emit = ratio_emit_case.computed();
        let ratio4 = layer_state(ratio_emit_case.input.pos, ratio_emit, 2);
        assert_eq!(ratio4.compression, "ratio4");
        assert_eq!(ratio4.pos, Some(127));
        assert_eq!(ratio4.comp_before, Some(31));
        assert_eq!(ratio4.comp_after, Some(32));
        assert_eq!(ratio4.index_before, Some(31));
        assert_eq!(ratio4.index_after, Some(32));
        assert!(ratio4.emit_compressed_row);
        assert!(!ratio4.indexed_attention);

        let ratio128 = layer_state(ratio_emit_case.input.pos, ratio_emit, 3);
        assert_eq!(ratio128.compression, "ratio128");
        assert_eq!(ratio128.comp_before, Some(0));
        assert_eq!(ratio128.comp_after, Some(1));
        assert_eq!(ratio128.index_before, None);
        assert!(ratio128.emit_compressed_row);

        let short_case = decode_trace_case("short_decode_after_prefill").expect("case");
        let short = short_case.computed();
        let non_emit = layer_state(short_case.input.pos, short, 2);
        assert_eq!(non_emit.comp_before, Some(5));
        assert_eq!(non_emit.comp_after, Some(5));
        assert_eq!(non_emit.index_before, Some(5));
        assert_eq!(non_emit.index_after, Some(5));
        assert!(!non_emit.emit_compressed_row);

        let long_case = decode_trace_case("long_indexed_decode").expect("case");
        let long = long_case.computed();
        let indexed = layer_state(long_case.input.pos, long, 2);
        assert_eq!(indexed.index_before, Some(838));
        assert_eq!(indexed.index_after, Some(838));
        assert!(!indexed.emit_compressed_row);
        assert!(indexed.indexed_attention);
        assert_eq!(
            indexed.attention_operation,
            "ds4_gpu_attention_indexed_mixed_batch_heads_tensor"
        );
    }

    #[test]
    fn trace_covers_every_default_facade_operation_across_cases() {
        for spec in DEFAULT_DECODE_FACADE_OPERATIONS {
            let mut found = false;
            for case in M105B_DECODE_CASE_ORACLE {
                for_each_trace_event(*case, |event| {
                    if event.operation == spec.operation {
                        found = true;
                    }
                });
            }
            assert!(found, "missing {}", spec.operation);
        }
    }
}
