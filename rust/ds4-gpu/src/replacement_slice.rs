//! Bounded backend replacement slice descriptors.
//!
//! M12.4 does not switch the runtime route. It records the first Rust-owned
//! replacement slice boundary and keeps unsupported backends fail-closed until
//! the later route gate.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementSliceSpec {
    pub schema: &'static str,
    pub milestone: &'static str,
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

pub const fn first_backend_replacement_slice() -> &'static ReplacementSliceSpec {
    &FIRST_BACKEND_REPLACEMENT_SLICE
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
}
