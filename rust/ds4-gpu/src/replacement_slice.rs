//! Bounded backend replacement slice descriptors.
//!
//! M12.4 does not switch the runtime route. It records the first Rust-owned
//! replacement slice boundary and keeps unsupported backends fail-closed until
//! the later route gate.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementSliceSpec {
    pub schema: &'static str,
    pub milestone: &'static str,
    pub status: &'static str,
    pub id: &'static str,
    pub operation_family: &'static str,
    pub fixture_id: &'static str,
    pub operation: &'static str,
    pub method: &'static str,
    pub rust_module: &'static str,
    pub facade_replay: &'static str,
    pub tensor_fixture_manifest: &'static str,
    pub comparator: &'static str,
    pub output_fields: &'static [&'static str],
    pub supported_backends: &'static [&'static str],
    pub unsupported_backends: &'static [&'static str],
    pub runtime_route_change: bool,
    pub general_backend_replacement: bool,
    pub kernel_replacement: bool,
    pub next_required_gate: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedReplacementBackend<'a> {
    pub requested: &'a str,
}

impl<'a> UnsupportedReplacementBackend<'a> {
    pub const fn new(requested: &'a str) -> Self {
        Self { requested }
    }
}

pub const FIRST_BACKEND_REPLACEMENT_SLICE: ReplacementSliceSpec = ReplacementSliceSpec {
    schema: "ds4.backend_replacement_slice.v1",
    milestone: "M12.4",
    status: "first-replacement-slice",
    id: "m12.4-embedding-and-indexer-embed-token-hc",
    operation_family: "embedding_and_indexer",
    fixture_id: "first_kernel_embed_token_hc",
    operation: "ds4_gpu_embed_token_hc_tensor",
    method: "embed_token_hc",
    rust_module: "rust/ds4-gpu/src/replacement_slice.rs",
    facade_replay: "ds4-parity/baselines/backend/m12.3/facade-replay.json",
    tensor_fixture_manifest: "ds4-parity/baselines/backend/m12.2/manifest.json",
    comparator: "ds4-parity/compare_decode_first_kernel_oracle.py",
    output_fields: &["cur_hc"],
    supported_backends: &["cuda-b300"],
    unsupported_backends: &["cpu", "metal", "runtime-default-route"],
    runtime_route_change: false,
    general_backend_replacement: false,
    kernel_replacement: false,
    next_required_gate: "M12.5 Runtime Backend Route Gate",
};

pub const BATCHED_EMBEDDING_REPLACEMENT_SLICE: ReplacementSliceSpec = ReplacementSliceSpec {
    schema: "ds4.backend_replacement_slice.v1",
    milestone: "M13.2",
    status: "batched-embedding-replacement-slice",
    id: "m13.2-embedding-and-indexer-embed-tokens-hc",
    operation_family: "embedding_and_indexer",
    fixture_id: "m13.1-embed-tokens-hc",
    operation: "ds4_gpu_embed_tokens_hc_tensor",
    method: "embed_tokens_hc",
    rust_module: "rust/ds4-gpu/src/replacement_slice.rs",
    facade_replay: "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    tensor_fixture_manifest:
        "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    comparator: "ds4-parity/compare_prefill_whole_short.py",
    output_fields: &["after_layer42_hc", "logits"],
    supported_backends: &["cuda-b300"],
    unsupported_backends: &["cpu", "metal", "runtime-default-route"],
    runtime_route_change: false,
    general_backend_replacement: false,
    kernel_replacement: false,
    next_required_gate: "M13.3 Indexed Decode Selection Replacement Slice",
};

pub const INDEXER_SCORE_ONE_REPLACEMENT_SLICE: ReplacementSliceSpec = ReplacementSliceSpec {
    schema: "ds4.backend_replacement_slice.v1",
    milestone: "M13.3",
    status: "indexed-decode-selection-replacement-slice",
    id: "m13.3-embedding-and-indexer-indexer-score-one",
    operation_family: "embedding_and_indexer",
    fixture_id: "m13.1-indexer-score-one",
    operation: "ds4_gpu_indexer_score_one_tensor",
    method: "indexer_score_one",
    rust_module: "rust/ds4-gpu/src/replacement_slice.rs",
    facade_replay: "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    tensor_fixture_manifest:
        "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    comparator: "ds4-parity/compare_decode_long_indexed_attention.py",
    output_fields: &["layer2_indexer_scores"],
    supported_backends: &["cuda-b300"],
    unsupported_backends: &["cpu", "metal", "runtime-default-route"],
    runtime_route_change: false,
    general_backend_replacement: false,
    kernel_replacement: false,
    next_required_gate: "M13.4 Batch Indexer Fixture Gap Closure",
};

pub const INDEXER_TOPK_REPLACEMENT_SLICE: ReplacementSliceSpec = ReplacementSliceSpec {
    schema: "ds4.backend_replacement_slice.v1",
    milestone: "M13.3",
    status: "indexed-decode-selection-replacement-slice",
    id: "m13.3-embedding-and-indexer-indexer-topk",
    operation_family: "embedding_and_indexer",
    fixture_id: "m13.1-indexer-topk",
    operation: "ds4_gpu_indexer_topk_tensor",
    method: "indexer_topk",
    rust_module: "rust/ds4-gpu/src/replacement_slice.rs",
    facade_replay: "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    tensor_fixture_manifest:
        "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    comparator: "ds4-parity/compare_decode_long_indexed_attention.py",
    output_fields: &["layer2_comp_selected"],
    supported_backends: &["cuda-b300"],
    unsupported_backends: &["cpu", "metal", "runtime-default-route"],
    runtime_route_change: false,
    general_backend_replacement: false,
    kernel_replacement: false,
    next_required_gate: "M13.4 Batch Indexer Fixture Gap Closure",
};

pub const INDEXED_DECODE_SELECTION_REPLACEMENT_SLICES: &[ReplacementSliceSpec] = &[
    INDEXER_SCORE_ONE_REPLACEMENT_SLICE,
    INDEXER_TOPK_REPLACEMENT_SLICE,
];

pub const BACKEND_REPLACEMENT_SLICES: &[ReplacementSliceSpec] = &[
    FIRST_BACKEND_REPLACEMENT_SLICE,
    BATCHED_EMBEDDING_REPLACEMENT_SLICE,
    INDEXER_SCORE_ONE_REPLACEMENT_SLICE,
    INDEXER_TOPK_REPLACEMENT_SLICE,
];

pub const fn first_backend_replacement_slice() -> &'static ReplacementSliceSpec {
    &FIRST_BACKEND_REPLACEMENT_SLICE
}

pub const fn batched_embedding_replacement_slice() -> &'static ReplacementSliceSpec {
    &BATCHED_EMBEDDING_REPLACEMENT_SLICE
}

pub const fn indexer_score_one_replacement_slice() -> &'static ReplacementSliceSpec {
    &INDEXER_SCORE_ONE_REPLACEMENT_SLICE
}

pub const fn indexer_topk_replacement_slice() -> &'static ReplacementSliceSpec {
    &INDEXER_TOPK_REPLACEMENT_SLICE
}

pub const fn indexed_decode_selection_replacement_slices() -> &'static [ReplacementSliceSpec] {
    INDEXED_DECODE_SELECTION_REPLACEMENT_SLICES
}

pub const fn replacement_slices() -> &'static [ReplacementSliceSpec] {
    BACKEND_REPLACEMENT_SLICES
}

pub fn replacement_slice_by_id(id: &str) -> Option<&'static ReplacementSliceSpec> {
    for spec in replacement_slices() {
        if str_eq(spec.id, id)
            || str_eq(spec.fixture_id, id)
            || str_eq(spec.method, id)
            || str_eq(spec.operation, id)
        {
            return Some(spec);
        }
    }
    match id {
        "first" | "m12.4" | "M12.4" => Some(first_backend_replacement_slice()),
        "batched-embedding" | "m13.2" | "M13.2" => Some(batched_embedding_replacement_slice()),
        "indexed-score-one" | "indexer-score-one" | "m13.3-score-one" | "M13.3-score-one" => {
            Some(indexer_score_one_replacement_slice())
        }
        "indexed-topk" | "indexer-topk" | "m13.3-topk" | "M13.3-topk" => {
            Some(indexer_topk_replacement_slice())
        }
        _ => None,
    }
}

pub fn ensure_supported_backend<'a>(
    spec: &ReplacementSliceSpec,
    backend: &'a str,
) -> Result<(), UnsupportedReplacementBackend<'a>> {
    if contains(spec.supported_backends, backend) {
        Ok(())
    } else {
        Err(UnsupportedReplacementBackend::new(backend))
    }
}

const fn contains(values: &[&str], needle: &str) -> bool {
    let mut index = 0;
    while index < values.len() {
        if str_eq(values[index], needle) {
            return true;
        }
        index += 1;
    }
    false
}

const fn str_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m12_4_slice_stays_bounded() {
        let spec = first_backend_replacement_slice();
        assert_eq!(spec.milestone, "M12.4");
        assert_eq!(spec.status, "first-replacement-slice");
        assert_eq!(spec.operation_family, "embedding_and_indexer");
        assert_eq!(spec.operation, "ds4_gpu_embed_token_hc_tensor");
        assert!(!spec.runtime_route_change);
        assert!(!spec.general_backend_replacement);
        assert!(!spec.kernel_replacement);
    }

    #[test]
    fn unsupported_backends_fail_closed() {
        let spec = first_backend_replacement_slice();
        assert_eq!(ensure_supported_backend(spec, "cuda-b300"), Ok(()));
        assert_eq!(
            ensure_supported_backend(spec, "cpu"),
            Err(UnsupportedReplacementBackend::new("cpu"))
        );
        assert_eq!(
            ensure_supported_backend(spec, "runtime-default-route"),
            Err(UnsupportedReplacementBackend::new("runtime-default-route"))
        );
    }

    #[test]
    fn m13_2_batched_embedding_slice_stays_bounded() {
        let spec = batched_embedding_replacement_slice();
        assert_eq!(spec.milestone, "M13.2");
        assert_eq!(spec.status, "batched-embedding-replacement-slice");
        assert_eq!(spec.operation_family, "embedding_and_indexer");
        assert_eq!(spec.operation, "ds4_gpu_embed_tokens_hc_tensor");
        assert_eq!(spec.method, "embed_tokens_hc");
        assert_eq!(spec.output_fields, &["after_layer42_hc", "logits"]);
        assert!(!spec.runtime_route_change);
        assert!(!spec.general_backend_replacement);
        assert!(!spec.kernel_replacement);
    }

    #[test]
    fn replacement_slice_selection_accepts_milestone_aliases() {
        assert_eq!(
            replacement_slice_by_id("m12.4").map(|spec| spec.id),
            Some("m12.4-embedding-and-indexer-embed-token-hc")
        );
        assert_eq!(
            replacement_slice_by_id("batched-embedding").map(|spec| spec.id),
            Some("m13.2-embedding-and-indexer-embed-tokens-hc")
        );
        assert!(replacement_slice_by_id("missing").is_none());
    }

    #[test]
    fn m13_3_indexed_decode_slices_stay_bounded() {
        let score = indexer_score_one_replacement_slice();
        assert_eq!(score.milestone, "M13.3");
        assert_eq!(score.status, "indexed-decode-selection-replacement-slice");
        assert_eq!(score.operation, "ds4_gpu_indexer_score_one_tensor");
        assert_eq!(score.method, "indexer_score_one");
        assert_eq!(score.output_fields, &["layer2_indexer_scores"]);
        assert!(!score.runtime_route_change);
        assert!(!score.general_backend_replacement);
        assert!(!score.kernel_replacement);

        let topk = indexer_topk_replacement_slice();
        assert_eq!(topk.milestone, "M13.3");
        assert_eq!(topk.status, "indexed-decode-selection-replacement-slice");
        assert_eq!(topk.operation, "ds4_gpu_indexer_topk_tensor");
        assert_eq!(topk.method, "indexer_topk");
        assert_eq!(topk.output_fields, &["layer2_comp_selected"]);
        assert!(!topk.runtime_route_change);
        assert!(!topk.general_backend_replacement);
        assert!(!topk.kernel_replacement);
    }

    #[test]
    fn m13_3_selection_requires_explicit_slice() {
        assert_eq!(
            replacement_slice_by_id("indexer-score-one").map(|spec| spec.id),
            Some("m13.3-embedding-and-indexer-indexer-score-one")
        );
        assert_eq!(
            replacement_slice_by_id("ds4_gpu_indexer_topk_tensor").map(|spec| spec.id),
            Some("m13.3-embedding-and-indexer-indexer-topk")
        );
        assert!(replacement_slice_by_id("m13.3").is_none());
        assert_eq!(indexed_decode_selection_replacement_slices().len(), 2);
    }
}
