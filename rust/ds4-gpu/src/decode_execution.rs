//! Model-backed decode execution preflight contracts.
//!
//! This module names the real-model checks that must pass on B300 before the
//! Rust one-token scheduler starts launching the full decode trace.

use crate::graph_plan::{layer_compression, LayerCompression};

pub const DECODE_EXECUTION_PREFLIGHT_SCHEMA: &str = "ds4.decode_execution_preflight.v1";
pub const DECODE_EXECUTION_PREFLIGHT_SCOPE: &str = "model_backed_b300_preflight";
pub const DECODE_EXECUTION_PREFLIGHT_CASE: &str = "short_decode_logits";
pub const DECODE_EXECUTION_PREFLIGHT_LAYERS: &[usize] = &[0, 2, 3];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeCheckpointTarget {
    pub name: &'static str,
    pub stage: &'static str,
    pub boundary: &'static str,
    pub tensor: &'static str,
    pub layer: Option<usize>,
    pub hash_policy: &'static str,
}

pub const PREFLIGHT_CHECKPOINT_TARGETS: &[DecodeCheckpointTarget] = &[
    DecodeCheckpointTarget {
        name: "short_decode_logits",
        stage: "decode",
        boundary: "metal_graph_eval_token_raw_swa",
        tensor: "logits",
        layer: None,
        hash_policy: "exact",
    },
    DecodeCheckpointTarget {
        name: "short_decode_layer2_attn_comp_cache",
        stage: "compressed-kv",
        boundary: "metal_graph_eval_token_raw_swa",
        tensor: "layer_attn_comp_cache",
        layer: Some(2),
        hash_policy: "exact",
    },
    DecodeCheckpointTarget {
        name: "short_decode_layer2_index_comp_cache",
        stage: "compressed-kv",
        boundary: "metal_graph_eval_token_raw_swa",
        tensor: "layer_index_comp_cache",
        layer: Some(2),
        hash_policy: "exact",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentativeTensorPlan {
    pub field: &'static str,
    pub layer: Option<usize>,
}

pub const REPRESENTATIVE_TENSORS: &[RepresentativeTensorPlan] = &[
    RepresentativeTensorPlan {
        field: "cur_hc",
        layer: None,
    },
    RepresentativeTensorPlan {
        field: "logits",
        layer: None,
    },
    RepresentativeTensorPlan {
        field: "layer_raw_cache",
        layer: Some(0),
    },
    RepresentativeTensorPlan {
        field: "layer_attn_comp_cache",
        layer: Some(2),
    },
    RepresentativeTensorPlan {
        field: "layer_index_comp_cache",
        layer: Some(2),
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeExecutionLayerCoverage {
    pub dense: bool,
    pub ratio4: bool,
    pub ratio128: bool,
}

impl DecodeExecutionLayerCoverage {
    pub const fn covers_default_decode(self) -> bool {
        self.dense && self.ratio4 && self.ratio128
    }
}

pub fn preflight_layer_coverage() -> DecodeExecutionLayerCoverage {
    let mut coverage = DecodeExecutionLayerCoverage {
        dense: false,
        ratio4: false,
        ratio128: false,
    };
    for layer in DECODE_EXECUTION_PREFLIGHT_LAYERS {
        match layer_compression(*layer).expect("preflight layer in range") {
            LayerCompression::Dense => coverage.dense = true,
            LayerCompression::Ratio4 => coverage.ratio4 = true,
            LayerCompression::Ratio128 => coverage.ratio128 = true,
        }
    }
    coverage
}

#[cfg(test)]
mod tests {
    use super::{
        preflight_layer_coverage, DecodeCheckpointTarget, PREFLIGHT_CHECKPOINT_TARGETS,
        REPRESENTATIVE_TENSORS,
    };

    #[test]
    fn preflight_layers_cover_decode_compression_modes() {
        assert!(preflight_layer_coverage().covers_default_decode());
    }

    #[test]
    fn preflight_targets_cover_m10_4_short_decode_checkpoints() {
        assert!(target("short_decode_logits").is_some());
        assert_eq!(
            target("short_decode_layer2_attn_comp_cache")
                .expect("attn comp target")
                .layer,
            Some(2)
        );
        assert_eq!(
            target("short_decode_layer2_index_comp_cache")
                .expect("index comp target")
                .hash_policy,
            "exact"
        );
    }

    #[test]
    fn representative_tensors_include_checkpoint_outputs() {
        assert!(REPRESENTATIVE_TENSORS
            .iter()
            .any(|tensor| tensor.field == "logits" && tensor.layer.is_none()));
        assert!(REPRESENTATIVE_TENSORS
            .iter()
            .any(|tensor| tensor.field == "layer_attn_comp_cache" && tensor.layer == Some(2)));
        assert!(REPRESENTATIVE_TENSORS
            .iter()
            .any(|tensor| tensor.field == "layer_index_comp_cache" && tensor.layer == Some(2)));
    }

    fn target(name: &str) -> Option<&'static DecodeCheckpointTarget> {
        PREFLIGHT_CHECKPOINT_TARGETS
            .iter()
            .find(|target| target.name == name)
    }
}
