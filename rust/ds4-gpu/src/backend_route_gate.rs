//! Explicit runtime backend route gate for the first replacement slice.
//!
//! M12.5 keeps the default runtime path on the current backend. The replacement
//! slice is available only through an opt-in route descriptor so closure and
//! removal decisions can stay blocked until M12.6.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeBackendRouteGateSpec {
    pub schema: &'static str,
    pub milestone: &'static str,
    pub id: &'static str,
    pub status: &'static str,
    pub route_selector: &'static str,
    pub default_route: &'static str,
    pub opt_in_route: &'static str,
    pub selected_slice_id: &'static str,
    pub operation_family: &'static str,
    pub operation: &'static str,
    pub method: &'static str,
    pub replacement_slice_artifact: &'static str,
    pub runtime_graph_route: &'static str,
    pub graph_backend: &'static str,
    pub supported_backends: &'static [&'static str],
    pub unsupported_backends: &'static [&'static str],
    pub validation_artifacts: &'static [&'static str],
    pub quality_gates: &'static [&'static str],
    pub benchmark_policy: &'static str,
    pub default_route_unchanged: bool,
    pub replacement_route_opt_in: bool,
    pub default_route_replacement_active: bool,
    pub general_backend_replacement: bool,
    pub kernel_replacement: bool,
    pub next_required_gate: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeBackendRoute {
    CurrentBackend,
    ReplacementSlice,
}

impl RuntimeBackendRoute {
    pub const fn name(self) -> &'static str {
        match self {
            Self::CurrentBackend => "current-backend",
            Self::ReplacementSlice => "replacement-slice",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeBackendRouteDecision<'a> {
    pub route: RuntimeBackendRoute,
    pub backend: &'a str,
    pub replacement_active: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeBackendRouteError<'a> {
    UnsupportedRoute { requested: &'a str },
    UnsupportedBackend { requested: &'a str },
}

pub const FIRST_BACKEND_RUNTIME_ROUTE_GATE: RuntimeBackendRouteGateSpec =
    RuntimeBackendRouteGateSpec {
        schema: "ds4.backend_runtime_route_gate.v1",
        milestone: "M12.5",
        id: "m12.5-runtime-backend-route-gate",
        status: "runtime-route-gate",
        route_selector: "--runtime-backend-route",
        default_route: "current-backend",
        opt_in_route: "replacement-slice",
        selected_slice_id: "m12.4-embedding-and-indexer-embed-token-hc",
        operation_family: "embedding_and_indexer",
        operation: "ds4_gpu_embed_token_hc_tensor",
        method: "embed_token_hc",
        replacement_slice_artifact: "ds4-parity/baselines/backend/m12.4/replacement-slice.json",
        runtime_graph_route: "graph",
        graph_backend: "cuda",
        supported_backends: &["cuda-b300"],
        unsupported_backends: &["cpu", "metal", "runtime-default-route"],
        validation_artifacts: &[
            "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
            "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
            "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
            "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
        ],
        quality_gates: &[
            "official-vectors",
            "long-context",
            "tool-server",
            "same-session-benchmark",
        ],
        benchmark_policy: "same-session-current-c-parity",
        default_route_unchanged: true,
        replacement_route_opt_in: true,
        default_route_replacement_active: false,
        general_backend_replacement: false,
        kernel_replacement: false,
        next_required_gate: "M12.6 Backend Replacement Closure And Removal Decision",
    };

pub const fn first_backend_runtime_route_gate() -> &'static RuntimeBackendRouteGateSpec {
    &FIRST_BACKEND_RUNTIME_ROUTE_GATE
}

pub fn parse_runtime_backend_route(value: &str) -> Option<RuntimeBackendRoute> {
    match value {
        "current-backend" | "current" | "default" | "off" => {
            Some(RuntimeBackendRoute::CurrentBackend)
        }
        "replacement-slice" | "m12.4-replacement-slice" => {
            Some(RuntimeBackendRoute::ReplacementSlice)
        }
        _ => None,
    }
}

pub fn route_decision<'a>(
    spec: &RuntimeBackendRouteGateSpec,
    route: &'a str,
    backend: &'a str,
) -> Result<RuntimeBackendRouteDecision<'a>, RuntimeBackendRouteError<'a>> {
    match parse_runtime_backend_route(route) {
        Some(RuntimeBackendRoute::CurrentBackend) => Ok(RuntimeBackendRouteDecision {
            route: RuntimeBackendRoute::CurrentBackend,
            backend,
            replacement_active: false,
        }),
        Some(RuntimeBackendRoute::ReplacementSlice) => {
            if contains(spec.supported_backends, backend) {
                Ok(RuntimeBackendRouteDecision {
                    route: RuntimeBackendRoute::ReplacementSlice,
                    backend,
                    replacement_active: true,
                })
            } else {
                Err(RuntimeBackendRouteError::UnsupportedBackend { requested: backend })
            }
        }
        None => Err(RuntimeBackendRouteError::UnsupportedRoute { requested: route }),
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
    fn route_gate_stays_opt_in() {
        let spec = first_backend_runtime_route_gate();
        assert_eq!(spec.milestone, "M12.5");
        assert_eq!(spec.default_route, "current-backend");
        assert_eq!(spec.opt_in_route, "replacement-slice");
        assert!(!spec.default_route_replacement_active);
        assert!(spec.default_route_unchanged);
        assert!(spec.replacement_route_opt_in);
        assert!(!spec.general_backend_replacement);
        assert!(!spec.kernel_replacement);
    }

    #[test]
    fn replacement_route_requires_supported_backend() {
        let spec = first_backend_runtime_route_gate();
        assert_eq!(
            route_decision(spec, "replacement-slice", "cuda-b300"),
            Ok(RuntimeBackendRouteDecision {
                route: RuntimeBackendRoute::ReplacementSlice,
                backend: "cuda-b300",
                replacement_active: true,
            })
        );
        assert_eq!(
            route_decision(spec, "current-backend", "cuda-b300"),
            Ok(RuntimeBackendRouteDecision {
                route: RuntimeBackendRoute::CurrentBackend,
                backend: "cuda-b300",
                replacement_active: false,
            })
        );
        assert_eq!(
            route_decision(spec, "replacement-slice", "cpu"),
            Err(RuntimeBackendRouteError::UnsupportedBackend { requested: "cpu" })
        );
        assert_eq!(
            route_decision(spec, "target-stream", "cuda-b300"),
            Err(RuntimeBackendRouteError::UnsupportedRoute {
                requested: "target-stream",
            })
        );
    }
}
