#!/usr/bin/env python3
"""Validate the M13.5 expanded embedding/indexer route closure."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "ds4-parity/baselines/backend/m13.5/expanded-route-gate.json"
CLOSURE = ROOT / "ds4-parity/baselines/backend/m13.5/expanded-route-closure.json"
M12_4_SLICE = ROOT / "ds4-parity/baselines/backend/m12.4/replacement-slice.json"
M12_5_GATE = ROOT / "ds4-parity/baselines/backend/m12.5/runtime-route-gate.json"
M13_0_DECISION = ROOT / "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json"
M13_1_MATRIX = ROOT / "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json"
M13_2_SLICE = ROOT / "ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json"
M13_3_SLICES = ROOT / "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json"
M13_4_BUNDLE = ROOT / "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json"
M10_9C = ROOT / "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json"
M10_9D = ROOT / "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json"
M10_9E = ROOT / "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json"
M10_9F = ROOT / "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json"
RUST_MODULE = ROOT / "rust/ds4-gpu/src/backend_route_gate.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_SUPPORTED = ["cuda-b300"]
EXPECTED_UNSUPPORTED = ["cpu", "metal", "runtime-default-route"]
EXPECTED_VALIDATION_ARTIFACTS = [
    "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
    "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
    "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
    "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
    "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json",
]
EXPECTED_QUALITY_GATES = [
    "official-vectors",
    "long-context",
    "tool-server",
    "same-session-benchmark",
    "batch-indexer-fixture-closure",
]
EXPECTED_OPERATIONS = [
    {
        "operation": "ds4_gpu_embed_token_hc_tensor",
        "method": "embed_token_hc",
        "source_stage": "M12.4",
        "status": "route-gated-rust-slice",
        "route_role": "rust-replacement-slice",
        "source_artifact": "ds4-parity/baselines/backend/m12.4/replacement-slice.json",
    },
    {
        "operation": "ds4_gpu_embed_tokens_hc_tensor",
        "method": "embed_tokens_hc",
        "source_stage": "M13.2",
        "status": "opt-in-rust-slice",
        "route_role": "rust-replacement-slice",
        "source_artifact": "ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json",
    },
    {
        "operation": "ds4_gpu_indexer_score_one_tensor",
        "method": "indexer_score_one",
        "source_stage": "M13.3",
        "status": "opt-in-rust-slice",
        "route_role": "rust-replacement-slice",
        "source_artifact": "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json",
    },
    {
        "operation": "ds4_gpu_indexer_topk_tensor",
        "method": "indexer_topk",
        "source_stage": "M13.3",
        "status": "opt-in-rust-slice",
        "route_role": "rust-replacement-slice",
        "source_artifact": "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json",
    },
    {
        "operation": "ds4_gpu_indexer_scores_prefill_tensor",
        "method": "indexer_scores_prefill",
        "source_stage": "M13.4",
        "status": "fixture-covered-sidecar",
        "route_role": "current-backend-sidecar",
        "source_artifact": "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json",
    },
    {
        "operation": "ds4_gpu_indexer_scores_decode_batch_tensor",
        "method": "indexer_scores_decode_batch",
        "source_stage": "M13.4",
        "status": "fixture-covered-sidecar",
        "route_role": "current-backend-sidecar",
        "source_artifact": "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json",
    },
    {
        "operation": "ds4_gpu_dsv4_topk_mask_tensor",
        "method": "dsv4_topk_mask",
        "source_stage": "M13.4",
        "status": "fixture-covered-sidecar",
        "route_role": "current-backend-sidecar",
        "source_artifact": "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json",
    },
]


@dataclass
class Report:
    checks: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    artifacts = {
        "gate": load_json(GATE),
        "closure": load_json(CLOSURE),
        "m12_4": load_json(M12_4_SLICE),
        "m12_5": load_json(M12_5_GATE),
        "m13_0": load_json(M13_0_DECISION),
        "m13_1": load_json(M13_1_MATRIX),
        "m13_2": load_json(M13_2_SLICE),
        "m13_3": load_json(M13_3_SLICES),
        "m13_4": load_json(M13_4_BUNDLE),
        "m10_9c": load_json(M10_9C),
        "m10_9d": load_json(M10_9D),
        "m10_9e": load_json(M10_9E),
        "m10_9f": load_json(M10_9F),
    }
    texts = {
        "rust": read_text(RUST_MODULE),
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, artifacts, texts, run_commands=not args.no_commands)
    if args.negative_test:
        run_negative_tests(report, artifacts, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
    *,
    run_commands: bool,
) -> None:
    gate = artifacts["gate"]
    closure = artifacts["closure"]
    validate_gate(report, gate)
    validate_closure(report, gate, closure)
    validate_operation_matrix(report, closure, artifacts)
    validate_source_artifacts(report, closure, artifacts)
    validate_runtime_artifacts(report, gate, artifacts)
    validate_rust_module(report, gate, texts["rust"])
    if run_commands:
        validate_rust_emitter(report, gate, artifacts["m12_5"])
        validate_route_decisions(report)
        run_dependency_checkers(report)
    validate_static_wiring(report, texts)


def validate_gate(report: Report, gate: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_runtime_route_gate.v1",
        "milestone": "M13.5",
        "status": "expanded-route-gate",
        "id": "m13.5-expanded-embedding-indexer-route-gate",
        "route_selector": "--runtime-backend-route",
        "default_route": "current-backend",
        "opt_in_route": "expanded-embedding-indexer",
        "selected_slice_id": "m13.5-expanded-embedding-indexer-route",
        "operation_family": "embedding_and_indexer",
        "operation": "embedding_and_indexer_expanded_route",
        "method": "expanded_embedding_indexer_route",
        "replacement_slice_artifact": "ds4-parity/baselines/backend/m13.5/expanded-route-closure.json",
        "runtime_graph_route": "graph",
        "graph_backend": "cuda",
        "benchmark_policy": "same-session-current-c-parity",
        "next_required_gate": "post-M13 roadmap decision",
        "route_check": "not-requested",
    }
    for key, value in expected.items():
        report.check(gate.get(key) == value, f"gate {key} drift")
    report.check(gate.get("supported_backends") == EXPECTED_SUPPORTED, "gate supported backend drift")
    report.check(gate.get("unsupported_backends") == EXPECTED_UNSUPPORTED, "gate unsupported backend drift")
    report.check(
        gate.get("validation_artifacts") == EXPECTED_VALIDATION_ARTIFACTS,
        "gate validation artifact drift",
    )
    report.check(gate.get("quality_gates") == EXPECTED_QUALITY_GATES, "gate quality gate drift")
    report.check(gate.get("default_route_unchanged") is True, "default route changed")
    report.check(gate.get("replacement_route_opt_in") is True, "expanded route not opt-in")
    report.check(
        gate.get("default_route_replacement_active") is False,
        "default route replacement became active",
    )
    report.check(gate.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(gate.get("kernel_replacement") is False, "kernel replacement overclaim")
    for key in ["replacement_slice_artifact", "validation_artifacts"]:
        values = gate.get(key)
        if isinstance(values, str):
            values = [values]
        report.check(isinstance(values, list), f"gate artifact path field invalid: {key}")
        if isinstance(values, list):
            for value in values:
                report.check(isinstance(value, str) and (ROOT / value).exists(), f"gate path missing: {value}")


def validate_closure(report: Report, gate: dict[str, Any], closure: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.expanded_embedding_indexer_route_closure.v1",
        "milestone": "M13.5",
        "parent": "M13",
        "previous_stage": "M13.4",
        "next_stage": "post-M13-roadmap-decision",
        "status": "expanded-route-closure-no-removal",
        "id": "m13.5-expanded-embedding-indexer-route-closure",
        "operation_family": "embedding_and_indexer",
    }
    for key, value in expected.items():
        report.check(closure.get(key) == value, f"closure {key} drift")
    source_artifacts = closure.get("source_artifacts")
    report.check(isinstance(source_artifacts, dict), "closure source artifacts missing")
    if isinstance(source_artifacts, dict):
        for key, value in source_artifacts.items():
            report.check(isinstance(value, str) and (ROOT / value).exists(), f"source artifact missing: {key}")
    policy = require_dict(report, closure.get("claim_policy"), "claim_policy")
    report.check(policy.get("runtime_route_change") is True, "M13.5 must record route gate change")
    report.check(policy.get("default_route_unchanged") is True, "default route policy drift")
    report.check(policy.get("default_route_replacement_active") is False, "default route overclaim")
    report.check(policy.get("replacement_route_opt_in_only") is True, "opt-in policy drift")
    report.check(policy.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(policy.get("kernel_replacement") is False, "kernel replacement overclaim")
    report.check(policy.get("removals_allowed") is False, "closure removal overclaim")
    report.check(policy.get("current_backend_retained_as_oracle") is True, "oracle retention drift")
    report.check(policy.get("current_backend_retained_as_sidecar") is True, "sidecar retention drift")
    report.check(
        policy.get("fixture_only_operations_use_current_backend_sidecar") is True,
        "fixture-only sidecar policy drift",
    )
    report.check(policy.get("raw_tensor_bodies_committed") is False, "raw tensor body overclaim")
    route_gate = require_dict(report, closure.get("route_gate"), "route_gate")
    for key in [
        "route_selector",
        "default_route",
        "opt_in_route",
        "selected_slice_id",
        "runtime_graph_route",
        "graph_backend",
        "supported_backends",
        "unsupported_backends",
    ]:
        report.check(route_gate.get(key) == gate.get(key), f"route gate mirror drift: {key}")
    validate_removal_criteria(report, closure)
    summary = require_dict(report, closure.get("summary"), "summary")
    report.check(summary.get("expanded_route_status") == "opt-in-only", "expanded route summary drift")
    report.check(summary.get("default_route") == "current-backend", "summary default route drift")
    report.check(summary.get("removals_allowed") is False, "summary removal overclaim")
    report.check(summary.get("next_required_gate") == "post-M13 roadmap decision", "summary next gate drift")
    decision = require_dict(report, closure.get("overall_decision"), "overall_decision")
    report.check(
        decision.get("decision") == "expanded_route_opt_in_current_backend_sidecar_retained",
        "overall decision drift",
    )
    report.check(decision.get("removals_allowed") is False, "overall removal overclaim")


def validate_removal_criteria(report: Report, closure: dict[str, Any]) -> None:
    criteria = closure.get("removal_criteria")
    report.check(isinstance(criteria, list) and len(criteria) == 5, "removal criteria count drift")
    if not isinstance(criteria, list):
        return
    statuses = [item.get("status") for item in criteria if isinstance(item, dict)]
    report.check("not-satisfied" in statuses, "removal criteria must keep a hard blocker")
    report.check(all(status != "satisfied" for status in statuses), "removal criterion overclaim")
    for item in criteria:
        report.check(isinstance(item, dict) and item.get("criterion"), "removal criterion malformed")
        report.check(isinstance(item, dict) and item.get("reason"), "removal criterion reason missing")


def validate_operation_matrix(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    matrix = closure.get("operation_route_matrix")
    report.check(isinstance(matrix, list) and len(matrix) == len(EXPECTED_OPERATIONS), "operation count drift")
    if not isinstance(matrix, list):
        return
    by_operation = {item.get("operation"): item for item in matrix if isinstance(item, dict)}
    report.check(list(by_operation) == [item["operation"] for item in EXPECTED_OPERATIONS], "operation order drift")
    for expected in EXPECTED_OPERATIONS:
        operation = expected["operation"]
        row = by_operation.get(operation)
        report.check(row is not None, f"operation missing: {operation}")
        if row is None:
            continue
        for key in ["method", "source_stage", "status", "route_role", "source_artifact"]:
            report.check(row.get(key) == expected[key], f"{operation}: {key} drift")
        report.check((ROOT / expected["source_artifact"]).exists(), f"{operation}: source artifact missing")
        report.check(row.get("default_route_replacement_active") is False, f"{operation}: default route overclaim")
        if expected["route_role"] == "rust-replacement-slice":
            report.check(row.get("route") == "expanded-embedding-indexer", f"{operation}: route drift")
            report.check(row.get("replacement_slice_available") is True, f"{operation}: replacement slice missing")
            report.check(row.get("current_backend_sidecar") is False, f"{operation}: sidecar overclaim")
        else:
            report.check(row.get("route") == "current-backend-sidecar", f"{operation}: sidecar route drift")
            report.check(row.get("replacement_slice_available") is False, f"{operation}: replacement overclaim")
            report.check(row.get("current_backend_sidecar") is True, f"{operation}: sidecar missing")
    summary = require_dict(report, closure.get("summary"), "summary")
    rust_ops = [item["operation"] for item in EXPECTED_OPERATIONS if item["route_role"] == "rust-replacement-slice"]
    sidecar_ops = [item["operation"] for item in EXPECTED_OPERATIONS if item["route_role"] == "current-backend-sidecar"]
    report.check(summary.get("rust_replacement_slice_operations") == rust_ops, "summary Rust operation drift")
    report.check(summary.get("current_backend_sidecar_operations") == sidecar_ops, "summary sidecar operation drift")
    validate_operation_sources(report, by_operation, artifacts)


def validate_operation_sources(
    report: Report,
    by_operation: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    first_op = artifacts["m12_4"].get("operation")
    report.check(first_op == "ds4_gpu_embed_token_hc_tensor", "M12.4 first op drift")
    report.check(first_op in by_operation, "M12.4 first op missing from closure")
    matrix_ops = [item.get("operation") for item in artifacts["m13_1"].get("matrix", []) if isinstance(item, dict)]
    expected_remaining = [item["operation"] for item in EXPECTED_OPERATIONS[1:]]
    report.check(set(matrix_ops) == set(expected_remaining), "M13.1 remaining operation drift")
    report.check(
        set(artifacts["m13_0"].get("remaining_operations_from_m12_6", [])) == set(expected_remaining),
        "M13.0 remaining op drift",
    )
    report.check(artifacts["m13_2"].get("operation") in by_operation, "M13.2 operation missing from closure")
    m13_3_ops = [
        item.get("operation")
        for item in artifacts["m13_3"].get("slices", [])
        if isinstance(item, dict)
    ]
    report.check(
        m13_3_ops == ["ds4_gpu_indexer_score_one_tensor", "ds4_gpu_indexer_topk_tensor"],
        "M13.3 operation drift",
    )
    for operation in m13_3_ops:
        report.check(operation in by_operation, f"M13.3 operation missing from closure: {operation}")
    fixture_ops = [
        item.get("operation")
        for item in artifacts["m13_4"].get("fixtures", [])
        if isinstance(item, dict)
    ]
    expected_sidecars = [item["operation"] for item in EXPECTED_OPERATIONS if item["route_role"] == "current-backend-sidecar"]
    report.check(fixture_ops == expected_sidecars, "M13.4 fixture operation drift")
    for operation in fixture_ops:
        row = by_operation.get(operation, {})
        report.check(row.get("route_role") == "current-backend-sidecar", f"{operation}: sidecar role missing")


def validate_source_artifacts(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    sources = require_dict(report, closure.get("source_artifacts"), "source_artifacts")
    expected_sources = {
        "expansion_decision": "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json",
        "expansion_matrix": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
        "batched_embedding_slice": "ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json",
        "indexed_decode_slices": "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json",
        "batch_indexer_fixture_bundle": "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json",
        "expanded_route_gate": "ds4-parity/baselines/backend/m13.5/expanded-route-gate.json",
        "runtime_official_vectors": "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
        "runtime_long_context": "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
        "runtime_tool_server": "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
        "runtime_benchmark_closure": "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
    }
    report.check(sources == expected_sources, "closure source artifact drift")
    report.check(artifacts["m13_0"].get("next_stage") == "M13.1", "M13.0 stage drift")
    report.check(artifacts["m13_1"].get("next_stage") == "M13.2", "M13.1 stage drift")
    report.check(artifacts["m13_2"].get("next_required_gate") == "M13.3 Indexed Decode Selection Replacement Slice", "M13.2 next gate drift")
    report.check(artifacts["m13_3"].get("next_required_gate") == "M13.4 Batch Indexer Fixture Gap Closure", "M13.3 next gate drift")
    report.check(artifacts["m13_4"].get("next_stage") == "M13.5", "M13.4 next stage drift")


def validate_runtime_artifacts(
    report: Report,
    gate: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    for key, milestone in [
        ("m10_9c", "M10.9c"),
        ("m10_9d", "M10.9d"),
        ("m10_9e", "M10.9e"),
        ("m10_9f", "M10.9f"),
    ]:
        data = artifacts[key]
        report.check(data.get("milestone") == milestone, f"{milestone}: milestone drift")
        report.check(data.get("runtime_graph_route") == gate.get("runtime_graph_route"), f"{milestone}: route drift")
        report.check(data.get("backend") == gate.get("graph_backend"), f"{milestone}: backend drift")
        model = require_dict(report, data.get("model"), f"{milestone}.model")
        report.check(model.get("sha256") == model.get("expected_sha256"), f"{milestone}: model sha drift")
    quality_gates = artifacts["m10_9f"].get("quality_gates")
    report.check(isinstance(quality_gates, list), "M10.9f quality gates missing")
    if isinstance(quality_gates, list):
        for gate_item in quality_gates:
            report.check(isinstance(gate_item, dict) and gate_item.get("ok") is True, "M10.9f quality gate failed")
    performance = require_dict(report, artifacts["m10_9f"].get("performance"), "M10.9f.performance")
    report.check(performance.get("same_session_current_c") == "pass", "same-session benchmark drift")
    claim = require_dict(report, artifacts["m10_9f"].get("claim_boundary"), "M10.9f.claim_boundary")
    report.check(claim.get("backend_replacement") is False, "M10.9f backend replacement overclaim")
    bundle_policy = require_dict(report, artifacts["m13_4"].get("claim_policy"), "M13.4.claim_policy")
    report.check(bundle_policy.get("current_backend_retained_as_sidecar") is True, "M13.4 sidecar policy drift")


def validate_rust_module(report: Report, gate: dict[str, Any], rust_source: str) -> None:
    for needle in [
        "EXPANDED_EMBEDDING_INDEXER_RUNTIME_ROUTE_GATE",
        "BACKEND_RUNTIME_ROUTE_GATES",
        "expanded_embedding_indexer_runtime_route_gate",
        "backend_runtime_route_gates",
        "runtime_route_gate_by_id",
        "ExpandedEmbeddingIndexer",
        "expanded-embedding-indexer",
        "m13.5-expanded-route",
        "route.name()",
        "spec.opt_in_route",
        gate["id"],
        gate["selected_slice_id"],
        gate["replacement_slice_artifact"],
        gate["next_required_gate"],
    ]:
        report.check(needle in rust_source, f"Rust route gate missing {needle}")


def validate_rust_emitter(report: Report, gate: dict[str, Any], m12_5_gate: dict[str, Any]) -> None:
    default = run_json(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-backend-route-gate", "--quiet"],
        expected_code=0,
    )
    report.check(default == m12_5_gate, "default Rust route gate emitter drift")
    expanded = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--gate",
            "expanded-embedding-indexer",
        ],
        expected_code=0,
    )
    report.check(expanded == gate, "expanded Rust route gate emitter drift")
    alias = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--gate",
            "M13.5",
        ],
        expected_code=0,
    )
    report.check(alias == gate, "expanded Rust route gate alias drift")


def validate_route_decisions(report: Report) -> None:
    expanded = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--gate",
            "expanded-embedding-indexer",
            "--route",
            "expanded-embedding-indexer",
            "--backend",
            "cuda-b300",
        ],
        expected_code=0,
    )
    report.check(expanded.get("route_check") == "supported", "expanded route check drift")
    report.check(expanded.get("checked_route") == "expanded-embedding-indexer", "expanded route identity drift")
    report.check(expanded.get("checked_backend") == "cuda-b300", "expanded backend identity drift")
    report.check(expanded.get("decision_replacement_active") is True, "expanded route inactive")
    first_gate_rejects = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--gate",
            "first",
            "--route",
            "expanded-embedding-indexer",
            "--backend",
            "cuda-b300",
        ],
        expected_code=2,
    )
    report.check(first_gate_rejects.get("route_check") == "unsupported-route", "first gate route rejection drift")
    report.check(
        first_gate_rejects.get("requested_route") == "expanded-embedding-indexer",
        "first gate rejected route identity drift",
    )
    unsupported_backend = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--gate",
            "expanded-embedding-indexer",
            "--route",
            "expanded-embedding-indexer",
            "--backend",
            "cpu",
        ],
        expected_code=3,
    )
    report.check(unsupported_backend.get("route_check") == "unsupported-backend", "expanded backend rejection drift")
    report.check(unsupported_backend.get("requested_backend") == "cpu", "expanded backend identity drift")


def run_dependency_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/check_backend_runtime_route_gate.py", "--negative-test"],
        ["ds4-parity/check_backend_expansion_decision.py", "--negative-test"],
        ["ds4-parity/check_backend_expansion_matrix.py", "--negative-test"],
        ["ds4-parity/check_backend_batched_embedding_slice.py", "--negative-test"],
        ["ds4-parity/check_backend_indexed_decode_slice.py", "--negative-test"],
        ["ds4-parity/check_backend_batch_indexer_fixtures.py", "--negative-test"],
        ["ds4-parity/run_runtime_graph_official_vectors.py", "--negative-test"],
        ["ds4-parity/run_runtime_graph_long_context.py", "--negative-test"],
        ["ds4-parity/run_tool_call_quality.py", "--negative-test"],
        ["ds4-parity/run_runtime_graph_bench.py", "--negative-test"],
    ]
    for command in commands:
        proc = subprocess.run([sys.executable, *command], cwd=ROOT, text=True, capture_output=True)
        report.check(proc.returncode == 0, f"{command[0]} failed: {proc.stderr or proc.stdout}")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M13.5 Expanded embedding/indexer route closure" in texts["report"], "unified report wiring missing")
    report.check("check_backend_expanded_route_closure.py" in texts["report"], "report checker path missing")
    report.check("Validate the M13.5 Expanded embedding/indexer route closure" in texts["readme"], "README wiring missing")
    report.check("M13.5: Embedding/Indexer Route Gate And Closure" in texts["roadmap"], "roadmap M13.5 missing")
    report.check("expanded-route-closure.json" in texts["roadmap"], "roadmap closure artifact missing")
    report.check("#### M13.5: Embedding/Indexer Route Gate And Closure" in texts["todo"], "TODO M13.5 missing")
    report.check("expanded-route-closure.json" in texts["todo"], "TODO closure artifact missing")
    report.check("Earlier M13.5 Embedding/Indexer Route Gate And Closure" in texts["status"], "status M13.5 previous item missing")
    report.check("Active item: post-M13 roadmap decision" in texts["status"], "post-M13 active item missing")


def run_negative_tests(
    report: Report,
    artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
) -> None:
    mutations = [
        ("default route active", lambda obj: mutate_nested(obj, ["gate", "default_route_replacement_active"], True)),
        ("route not opt-in", lambda obj: mutate_nested(obj, ["gate", "replacement_route_opt_in"], False)),
        ("removals allowed", lambda obj: mutate_nested(obj, ["closure", "claim_policy", "removals_allowed"], True)),
        ("fixture sidecar role overclaim", mutate_fixture_sidecar_role),
        ("missing operation", remove_last_operation),
        ("wrong next stage", lambda obj: mutate_nested(obj, ["closure", "next_stage"], "M14")),
        ("missing source artifact", lambda obj: mutate_nested(obj, ["closure", "source_artifacts", "batch_indexer_fixture_bundle"], "missing.json")),
    ]
    for name, mutate in mutations:
        mutated = copy.deepcopy(artifacts)
        mutate(mutated)
        mutated_report = Report()
        validate(mutated_report, mutated, texts, run_commands=False)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_nested(obj: dict[str, Any], path: list[str], value: Any) -> None:
    current: Any = obj
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def mutate_fixture_sidecar_role(obj: dict[str, dict[str, Any]]) -> None:
    matrix = obj["closure"]["operation_route_matrix"]
    for row in matrix:
        if row.get("operation") == "ds4_gpu_indexer_scores_prefill_tensor":
            row["route_role"] = "rust-replacement-slice"
            row["replacement_slice_available"] = True
            row["current_backend_sidecar"] = False
            return


def remove_last_operation(obj: dict[str, dict[str, Any]]) -> None:
    obj["closure"]["operation_route_matrix"] = obj["closure"]["operation_route_matrix"][:-1]


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label} must be object")
    return obj if isinstance(obj, dict) else {}


def run_json(command: list[str], *, expected_code: int) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != expected_code:
        raise SystemExit(
            f"{' '.join(command)}: expected exit {expected_code}, got {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{' '.join(command)}: invalid JSON output: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{' '.join(command)}: expected JSON object")
    return data


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Expanded route closure: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
