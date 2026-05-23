//! No-execute one-token decode plan for the DS4 runtime graph.
//!
//! This module mirrors the release `metal_graph_eval_token_raw_swa` scheduling
//! shape without calling backend kernels. It keeps the cache counters and
//! command-boundary decisions measurable before M10.5c starts executing the
//! plan through FFI.

use crate::graph_plan::{GraphPlan, LayerCounts, N_INDEXER_TOP_K, N_LAYER, N_SWA};

const N_LAYER_U32: u32 = N_LAYER as u32;
const RATIO4_LAYERS: u32 = 21;
const RATIO128_LAYERS: u32 = 20;

pub const DECODE_TOKEN_STAGE_ORDER: &[&str] = &[
    "begin_commands",
    "embed_token_hc",
    "decode_layers",
    "split_flush",
    "output_head",
    "end_commands",
    "read_logits",
    "synchronize_on_failure",
];

pub const DECODE_LAYER_STAGE_ORDER: &[&str] = &[
    "attn_hc_pre",
    "attn_norm",
    "q_path",
    "kv_path",
    "compressor_indexer",
    "attention",
    "attn_output",
    "attn_hc_post",
    "ffn_hc_pre",
    "ffn_norm",
    "router",
    "routed_moe",
    "shared_gate_up",
    "shared_down",
    "ffn_hc_post",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodePlanInput {
    pub ctx_size: u32,
    pub prompt_len: u32,
    pub mtp_enabled: bool,
    pub pos: u32,
    pub need_logits: bool,
    pub allow_split_flush: bool,
    pub split_after_layers: u32,
}

impl DecodePlanInput {
    pub const fn graph_plan(self) -> GraphPlan {
        GraphPlan::for_context(self.ctx_size, self.prompt_len, self.mtp_enabled)
    }

    pub const fn decode_plan(self) -> DecodeTokenPlan {
        DecodeTokenPlan::for_token(
            self.graph_plan(),
            self.pos,
            self.need_logits,
            self.allow_split_flush,
            self.split_after_layers,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeCommandPlan {
    pub begin_end_pairs: u32,
    pub flush_after_layer: Option<u32>,
    pub read_logits_after_end: bool,
    pub synchronize_on_failure: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeCachePlan {
    pub comp_before: u32,
    pub comp_after: u32,
    pub emit_layers: u32,
    pub indexed_attention_layers: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeTokenPlan {
    pub raw_window: u32,
    pub raw_cap: u32,
    pub raw_row: u32,
    pub n_raw: u32,
    pub raw_start: u32,
    pub command: DecodeCommandPlan,
    pub layer_counts: LayerCounts,
    pub ratio4_cache: DecodeCachePlan,
    pub ratio128_cache: DecodeCachePlan,
}

impl DecodeTokenPlan {
    pub const fn for_token(
        graph: GraphPlan,
        pos: u32,
        need_logits: bool,
        allow_split_flush: bool,
        split_after_layers: u32,
    ) -> Self {
        let raw_cap = graph.allocated_raw_cap;
        let n_raw = raw_span_for_batch(graph.raw_window, raw_cap, pos, 1);
        let ratio4_before = compressed_rows_before(pos, 4);
        let ratio4_after = compressed_rows_after(pos, 4);
        let ratio128_before = compressed_rows_before(pos, 128);
        let ratio128_after = compressed_rows_after(pos, 128);
        Self {
            raw_window: graph.raw_window,
            raw_cap,
            raw_row: raw_row(pos, raw_cap),
            n_raw,
            raw_start: raw_start_for_span(pos, n_raw, raw_cap),
            command: DecodeCommandPlan {
                begin_end_pairs: 1,
                flush_after_layer: split_flush_layer(allow_split_flush, split_after_layers),
                read_logits_after_end: need_logits,
                synchronize_on_failure: true,
            },
            layer_counts: graph.layer_counts,
            ratio4_cache: DecodeCachePlan {
                comp_before: ratio4_before,
                comp_after: ratio4_after,
                emit_layers: if ratio4_after != ratio4_before {
                    RATIO4_LAYERS
                } else {
                    0
                },
                indexed_attention_layers: if ratio4_after > N_INDEXER_TOP_K {
                    RATIO4_LAYERS
                } else {
                    0
                },
            },
            ratio128_cache: DecodeCachePlan {
                comp_before: ratio128_before,
                comp_after: ratio128_after,
                emit_layers: if ratio128_after != ratio128_before {
                    RATIO128_LAYERS
                } else {
                    0
                },
                indexed_attention_layers: 0,
            },
        }
    }
}

pub const fn raw_span_for_batch(raw_window: u32, raw_cap: u32, pos0: u32, n_tokens: u32) -> u32 {
    if raw_cap == 0 || n_tokens == 0 {
        return 0;
    }
    let window = if raw_window == 0 { N_SWA } else { raw_window };
    let last_pos = pos0 + n_tokens - 1;
    let mut needed = n_tokens as u64;
    if window != 0 {
        needed += if n_tokens == 1 {
            (window - 1) as u64
        } else {
            window as u64
        };
    }
    let available = last_pos as u64 + 1;
    if needed > available {
        needed = available;
    }
    if needed > raw_cap as u64 {
        needed = raw_cap as u64;
    }
    needed as u32
}

pub const fn raw_start_for_span(last_pos: u32, n_raw: u32, raw_cap: u32) -> u32 {
    if raw_cap == 0 || n_raw == 0 {
        0
    } else {
        (last_pos + 1 - n_raw) % raw_cap
    }
}

pub const fn raw_row(pos: u32, raw_cap: u32) -> u32 {
    if raw_cap == 0 {
        0
    } else {
        pos % raw_cap
    }
}

pub const fn compressed_rows_before(pos: u32, ratio: u32) -> u32 {
    if ratio == 0 {
        0
    } else {
        pos / ratio
    }
}

pub const fn compressed_rows_after(pos: u32, ratio: u32) -> u32 {
    if ratio == 0 {
        0
    } else if (pos + 1) % ratio == 0 {
        pos / ratio + 1
    } else {
        pos / ratio
    }
}

pub const fn split_flush_layer(allow_split_flush: bool, split_after_layers: u32) -> Option<u32> {
    if allow_split_flush && split_after_layers != 0 && split_after_layers <= N_LAYER_U32 {
        Some(split_after_layers - 1)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodePlanCaseOracle {
    pub name: &'static str,
    pub input: DecodePlanInput,
    pub expected: DecodeTokenPlan,
}

impl DecodePlanCaseOracle {
    pub const fn computed(self) -> DecodeTokenPlan {
        self.input.decode_plan()
    }
}

macro_rules! case {
    (
        $name:literal,
        $ctx_size:literal,
        $prompt_len:literal,
        $mtp_enabled:literal,
        $pos:literal,
        $need_logits:literal,
        $allow_split_flush:literal,
        $split_after_layers:literal,
        $raw_window:literal,
        $raw_cap:literal,
        $raw_row:literal,
        $n_raw:literal,
        $raw_start:literal,
        $flush_after_layer:expr,
        $dense_layers:literal,
        $ratio4_layers:literal,
        $ratio128_layers:literal,
        $ratio4_before:literal,
        $ratio4_after:literal,
        $ratio4_emit_layers:literal,
        $ratio4_indexed_layers:literal,
        $ratio128_before:literal,
        $ratio128_after:literal,
        $ratio128_emit_layers:literal
    ) => {
        DecodePlanCaseOracle {
            name: $name,
            input: DecodePlanInput {
                ctx_size: $ctx_size,
                prompt_len: $prompt_len,
                mtp_enabled: $mtp_enabled,
                pos: $pos,
                need_logits: $need_logits,
                allow_split_flush: $allow_split_flush,
                split_after_layers: $split_after_layers,
            },
            expected: DecodeTokenPlan {
                raw_window: $raw_window,
                raw_cap: $raw_cap,
                raw_row: $raw_row,
                n_raw: $n_raw,
                raw_start: $raw_start,
                command: DecodeCommandPlan {
                    begin_end_pairs: 1,
                    flush_after_layer: $flush_after_layer,
                    read_logits_after_end: $need_logits,
                    synchronize_on_failure: true,
                },
                layer_counts: LayerCounts {
                    dense: $dense_layers,
                    ratio4: $ratio4_layers,
                    ratio128: $ratio128_layers,
                },
                ratio4_cache: DecodeCachePlan {
                    comp_before: $ratio4_before,
                    comp_after: $ratio4_after,
                    emit_layers: $ratio4_emit_layers,
                    indexed_attention_layers: $ratio4_indexed_layers,
                },
                ratio128_cache: DecodeCachePlan {
                    comp_before: $ratio128_before,
                    comp_after: $ratio128_after,
                    emit_layers: $ratio128_emit_layers,
                    indexed_attention_layers: 0,
                },
            },
        }
    };
}

pub const M105B_DECODE_CASE_ORACLE: &[DecodePlanCaseOracle] = &[
    case!(
        "first_token_ctx32768",
        32768,
        1,
        false,
        0,
        true,
        true,
        4,
        128,
        256,
        0,
        1,
        0,
        Some(3),
        2,
        21,
        20,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ),
    case!(
        "short_decode_after_prefill",
        32768,
        21,
        false,
        21,
        true,
        true,
        4,
        128,
        256,
        21,
        22,
        0,
        Some(3),
        2,
        21,
        20,
        5,
        5,
        0,
        0,
        0,
        0,
        0
    ),
    case!(
        "ratio_emit_boundary",
        32768,
        128,
        false,
        127,
        true,
        true,
        4,
        128,
        256,
        127,
        128,
        0,
        Some(3),
        2,
        21,
        20,
        31,
        32,
        21,
        0,
        0,
        1,
        20
    ),
    case!(
        "long_indexed_decode",
        32768,
        3353,
        false,
        3353,
        true,
        true,
        4,
        128,
        2304,
        1049,
        128,
        922,
        Some(3),
        2,
        21,
        20,
        838,
        838,
        0,
        21,
        26,
        26,
        0
    ),
    case!(
        "no_logits_no_split",
        32768,
        21,
        false,
        21,
        false,
        false,
        0,
        128,
        256,
        21,
        22,
        0,
        None,
        2,
        21,
        20,
        5,
        5,
        0,
        0,
        0,
        0,
        0
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_plan::{layer_compression, LayerCompression};

    #[test]
    fn decode_case_oracles_match_computed_plan() {
        for case in M105B_DECODE_CASE_ORACLE {
            assert_eq!(case.computed(), case.expected, "{}", case.name);
        }
    }

    #[test]
    fn decode_layer_counts_match_ds4_compression_layout() {
        let mut dense = 0;
        let mut ratio4 = 0;
        let mut ratio128 = 0;
        for layer in 0..N_LAYER {
            match layer_compression(layer).unwrap() {
                LayerCompression::Dense => dense += 1,
                LayerCompression::Ratio4 => ratio4 += 1,
                LayerCompression::Ratio128 => ratio128 += 1,
            }
        }
        assert_eq!(
            LayerCounts {
                dense,
                ratio4,
                ratio128
            },
            LayerCounts {
                dense: 2,
                ratio4: 21,
                ratio128: 20
            }
        );
    }

    #[test]
    fn raw_span_matches_c_decode_edges() {
        assert_eq!(raw_span_for_batch(128, 256, 0, 1), 1);
        assert_eq!(raw_span_for_batch(128, 256, 21, 1), 22);
        assert_eq!(raw_span_for_batch(128, 256, 127, 1), 128);
        assert_eq!(raw_span_for_batch(128, 2304, 3353, 1), 128);
        assert_eq!(raw_start_for_span(3353, 128, 2304), 922);
        assert_eq!(raw_row(3353, 2304), 1049);
    }

    #[test]
    fn indexed_attention_threshold_matches_c_strict_greater_than() {
        assert_eq!(compressed_rows_after(2047, 4), N_INDEXER_TOP_K);
        assert_eq!(compressed_rows_after(2051, 4), N_INDEXER_TOP_K + 1);
        let graph = GraphPlan::for_context(32768, 2048, false);
        assert_eq!(
            DecodeTokenPlan::for_token(graph, 2047, true, true, 4)
                .ratio4_cache
                .indexed_attention_layers,
            0
        );
        assert_eq!(
            DecodeTokenPlan::for_token(graph, 2051, true, true, 4)
                .ratio4_cache
                .indexed_attention_layers,
            21
        );
    }

    #[test]
    fn stage_orders_anchor_current_c_boundaries() {
        assert_eq!(
            DECODE_TOKEN_STAGE_ORDER,
            [
                "begin_commands",
                "embed_token_hc",
                "decode_layers",
                "split_flush",
                "output_head",
                "end_commands",
                "read_logits",
                "synchronize_on_failure"
            ]
        );
        assert_eq!(DECODE_LAYER_STAGE_ORDER.len(), 15);
        assert_eq!(DECODE_LAYER_STAGE_ORDER[0], "attn_hc_pre");
        assert_eq!(DECODE_LAYER_STAGE_ORDER[14], "ffn_hc_post");
    }
}
