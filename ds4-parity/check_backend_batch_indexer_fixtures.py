#!/usr/bin/env python3
"""Validate the M13.4 batch indexer fixture gap closure bundle."""

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
BUNDLE = ROOT / "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json"
M13_1_MATRIX = ROOT / "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json"
DS4_C = ROOT / "ds4.c"
DECODE_BACKEND = ROOT / "rust/ds4-gpu/src/decode_backend.rs"
GRAPH_PLAN = ROOT / "rust/ds4-gpu/src/graph_plan.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_OPERATIONS = [
    {
        "id": "m13.4-indexer-scores-prefill",
        "operation": "ds4_gpu_indexer_scores_prefill_tensor",
        "method": "indexer_scores_prefill",
        "coverage_level": "inventory-only",
        "rust_source": "rust/ds4-gpu/src/graph_plan.rs",
        "rust_anchor": "ds4_gpu_indexer_scores_prefill_tensor",
        "outputs": ["indexer_scores", "indexer_topk"],
    },
    {
        "id": "m13.4-indexer-scores-decode-batch",
        "operation": "ds4_gpu_indexer_scores_decode_batch_tensor",
        "method": "indexer_scores_decode_batch",
        "coverage_level": "surface-covered-needs-fixture-extraction",
        "rust_source": "rust/ds4-gpu/src/decode_backend.rs",
        "rust_anchor": "pub fn indexer_scores_decode_batch",
        "outputs": ["indexer_scores", "indexer_topk"],
    },
    {
        "id": "m13.4-dsv4-topk-mask",
        "operation": "ds4_gpu_dsv4_topk_mask_tensor",
        "method": "dsv4_topk_mask",
        "coverage_level": "inventory-only",
        "rust_source": "rust/ds4-gpu/src/graph_plan.rs",
        "rust_anchor": "ds4_gpu_dsv4_topk_mask_tensor",
        "outputs": ["comp_mask"],
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
    bundle = load_json(BUNDLE)
    matrix = load_json(M13_1_MATRIX)
    sources = {
        "ds4.c": read_text(DS4_C),
        "rust/ds4-gpu/src/decode_backend.rs": read_text(DECODE_BACKEND),
        "rust/ds4-gpu/src/graph_plan.rs": read_text(GRAPH_PLAN),
    }
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, bundle, matrix, sources, texts, run_commands=not args.no_commands)
    if args.negative_test:
        run_negative_tests(report, bundle, matrix, sources, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    bundle: dict[str, Any],
    matrix: dict[str, Any],
    sources: dict[str, str],
    texts: dict[str, str],
    *,
    run_commands: bool,
) -> None:
    validate_bundle(report, bundle)
    validate_environment(report, bundle)
    validate_claim_policy(report, bundle)
    validate_fixtures(report, bundle, matrix, sources)
    validate_summary(report, bundle)
    if run_commands:
        run_dependency_checkers(report)
    validate_static_wiring(report, texts)


def validate_bundle(report: Report, bundle: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.batch_indexer_fixture_bundle.v1",
        "milestone": "M13.4",
        "status": "fixture-gaps-closed",
        "id": "m13.4-embedding-and-indexer-batch-indexer-fixtures",
        "operation_family": "embedding_and_indexer",
        "previous_stage": "M13.3",
        "next_stage": "M13.5",
    }
    for key, value in expected.items():
        report.check(bundle.get(key) == value, f"bundle {key} drift")
    artifacts = bundle.get("source_artifacts")
    report.check(isinstance(artifacts, dict), "source artifacts missing")
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            report.check(isinstance(value, str) and (ROOT / value).exists(), f"source artifact missing: {key}")
    comparison = bundle.get("comparison_policy")
    report.check(isinstance(comparison, dict), "comparison policy missing")
    if isinstance(comparison, dict):
        for key in ["digest", "samples", "shape", "dtype", "float_tolerance"]:
            report.check(isinstance(comparison.get(key), str) and comparison[key], f"comparison policy {key} missing")


def validate_environment(report: Report, bundle: dict[str, Any]) -> None:
    env = bundle.get("capture_environment")
    report.check(isinstance(env, dict), "capture environment missing")
    if not isinstance(env, dict):
        return
    expected = {
        "context": "hou2-prod1",
        "namespace": "default",
        "pod": "ds4-rust-port-b300",
        "workdir": "/workspace/ds4",
        "temp_kubeconfig": "/tmp/ds4-hou2-prod1.kubeconfig",
        "backend": "cuda",
        "model_path": "/workspace/ds4/ds4flash.gguf",
        "model_sha256": "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668",
        "prompt": "tests/test-vectors/prompts/long_memory_archive.txt",
    }
    for key, value in expected.items():
        report.check(env.get(key) == value, f"capture environment {key} drift")


def validate_claim_policy(report: Report, bundle: dict[str, Any]) -> None:
    policy = bundle.get("claim_policy")
    report.check(isinstance(policy, dict), "claim policy missing")
    if not isinstance(policy, dict):
        return
    report.check(policy.get("runtime_route_change") is False, "runtime route overclaim")
    report.check(policy.get("default_route_unchanged") is True, "default route policy drift")
    report.check(policy.get("default_route_replacement_active") is False, "default route replacement overclaim")
    report.check(policy.get("replacement_route_opt_in_only") is True, "opt-in route policy drift")
    report.check(policy.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(policy.get("kernel_replacement") is False, "kernel replacement overclaim")
    report.check(policy.get("raw_tensor_bodies_committed") is False, "raw tensor bodies must not be committed")
    report.check(policy.get("current_backend_retained_as_oracle") is True, "current backend oracle retention drift")
    report.check(policy.get("current_backend_retained_as_sidecar") is True, "current backend sidecar retention drift")
    report.check(policy.get("next_required_gate") == "M13.5 Embedding/Indexer Route Gate And Closure", "next gate drift")


def validate_fixtures(
    report: Report,
    bundle: dict[str, Any],
    matrix: dict[str, Any],
    sources: dict[str, str],
) -> None:
    fixtures = bundle.get("fixtures")
    report.check(isinstance(fixtures, list) and len(fixtures) == 3, "fixture count drift")
    if not isinstance(fixtures, list):
        return
    fixture_by_op = {item.get("operation"): item for item in fixtures if isinstance(item, dict)}
    for expected in EXPECTED_OPERATIONS:
        fixture = fixture_by_op.get(expected["operation"])
        report.check(fixture is not None, f"fixture missing: {expected['operation']}")
        row = matrix_row(matrix, expected["operation"])
        report.check(row is not None, f"M13.1 row missing: {expected['operation']}")
        if fixture is None or row is None:
            continue
        report.check(fixture.get("id") == expected["id"], f"{expected['operation']}: id drift")
        report.check(fixture.get("method") == expected["method"], f"{expected['operation']}: method drift")
        report.check(fixture.get("source_matrix_coverage_level") == expected["coverage_level"], f"{expected['operation']}: source coverage drift")
        report.check(fixture.get("prior_fixture_gap") is True, f"{expected['operation']}: prior fixture gap drift")
        report.check(fixture.get("fixture_gap_closed") is True, f"{expected['operation']}: closure flag drift")
        report.check(fixture.get("output_fields") == expected["outputs"], f"{expected['operation']}: output field drift")
        report.check(fixture.get("rust_source") == expected["rust_source"], f"{expected['operation']}: rust source drift")
        report.check(fixture.get("rust_anchor") == expected["rust_anchor"], f"{expected['operation']}: rust anchor drift")
        validate_matrix_row(report, expected, row)
        validate_source_anchors(report, expected, fixture, sources)
        validate_dtype(report, expected, fixture)
        validate_dependencies(report, fixture)
        validate_rerun_command(report, expected, fixture)


def validate_matrix_row(report: Report, expected: dict[str, Any], row: dict[str, Any]) -> None:
    operation = expected["operation"]
    report.check(row.get("coverage_level") == expected["coverage_level"], f"{operation}: M13.1 coverage drift")
    report.check(row.get("fixture_gap") is True, f"{operation}: M13.1 fixture gap drift")
    report.check(row.get("route_candidate_stage") == "M13.4 Batch Indexer Fixture Gap Closure", f"{operation}: stage drift")
    report.check(row.get("method") == expected["method"], f"{operation}: M13.1 method drift")
    report.check(row.get("removal_decision") == "retain_current_backend", f"{operation}: removal decision drift")


def validate_source_anchors(
    report: Report,
    expected: dict[str, Any],
    fixture: dict[str, Any],
    sources: dict[str, str],
) -> None:
    operation = expected["operation"]
    c_anchor = fixture.get("current_c_anchor")
    report.check(isinstance(c_anchor, str) and c_anchor in sources["ds4.c"], f"{operation}: C source anchor missing")
    rust_source = expected["rust_source"]
    report.check(expected["rust_anchor"] in sources[rust_source], f"{operation}: Rust source anchor missing")
    hooks = fixture.get("debug_dump_hooks")
    report.check(isinstance(hooks, list) and hooks, f"{operation}: debug dump hooks missing")
    if isinstance(hooks, list):
        for hook in hooks:
            report.check(isinstance(hook, str) and hook in sources["ds4.c"], f"{operation}: debug hook missing {hook}")


def validate_dtype(report: Report, expected: dict[str, Any], fixture: dict[str, Any]) -> None:
    dtype = fixture.get("dtype")
    report.check(isinstance(dtype, dict), f"{expected['operation']}: dtype map missing")
    if not isinstance(dtype, dict):
        return
    for output in expected["outputs"]:
        report.check(dtype.get(output) in {"float32", "int32"}, f"{expected['operation']}: dtype missing for {output}")
    if "indexer_topk" in expected["outputs"]:
        report.check(dtype.get("indexer_topk") == "int32", f"{expected['operation']}: top-k dtype drift")
    if "comp_mask" in expected["outputs"]:
        report.check(dtype.get("comp_mask") == "float32", f"{expected['operation']}: comp mask dtype drift")


def validate_dependencies(report: Report, fixture: dict[str, Any]) -> None:
    deps = fixture.get("comparator_dependencies")
    report.check(isinstance(deps, list), f"{fixture.get('operation')}: comparator dependencies missing")
    if not isinstance(deps, list):
        return
    for dep in deps:
        report.check(isinstance(dep, str) and (ROOT / dep).exists(), f"{fixture.get('operation')}: dependency missing {dep}")


def validate_rerun_command(report: Report, expected: dict[str, Any], fixture: dict[str, Any]) -> None:
    command = fixture.get("rerun_command")
    operation = expected["operation"]
    report.check(isinstance(command, str) and command, f"{operation}: rerun command missing")
    if not isinstance(command, str):
        return
    for needle in [
        "git archive HEAD",
        "kubectl",
        "--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig",
        "--context hou2-prod1",
        "ds4-rust-port-b300",
        "cd /workspace/ds4",
        "CUDA_ARCH=native",
        "--backend cuda",
        "/workspace/ds4/ds4flash.gguf",
        "DS4_METAL_GRAPH_DUMP_PREFIX",
        "DS4_METAL_GRAPH_DUMP_NAME",
    ]:
        report.check(needle in command, f"{operation}: rerun command missing {needle}")
    report.check("--runtime-graph graph" not in command, f"{operation}: M13.4 must not change runtime route")


def validate_summary(report: Report, bundle: dict[str, Any]) -> None:
    summary = bundle.get("summary")
    report.check(isinstance(summary, dict), "summary missing")
    if not isinstance(summary, dict):
        return
    report.check(
        summary.get("fixture_gap_operations_closed") == [item["operation"] for item in EXPECTED_OPERATIONS],
        "summary closed operation list drift",
    )
    report.check(summary.get("route_status") == "fixture-covered-not-route-gated", "summary route status drift")
    report.check(summary.get("removals_allowed") is False, "summary removal overclaim")
    report.check(summary.get("next_route_gate") == "M13.5 Embedding/Indexer Route Gate And Closure", "summary next gate drift")


def run_dependency_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/check_backend_expansion_decision.py", "--negative-test"],
        ["ds4-parity/check_backend_expansion_matrix.py", "--negative-test"],
        ["ds4-parity/check_backend_batched_embedding_slice.py", "--negative-test"],
        ["ds4-parity/check_backend_indexed_decode_slice.py", "--negative-test"],
        ["ds4-parity/compare_prefill_chunked.py", "--negative-test"],
        ["ds4-parity/compare_prefill_resumed.py", "--negative-test"],
    ]
    for command in commands:
        proc = subprocess.run([sys.executable, *command], cwd=ROOT, text=True, capture_output=True)
        report.check(proc.returncode == 0, f"{command[0]} failed: {proc.stderr or proc.stdout}")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M13.4 Batch indexer fixture gap closure" in texts["report"], "unified report wiring missing")
    report.check("check_backend_batch_indexer_fixtures.py" in texts["report"], "report checker path missing")
    report.check("Validate the M13.4 Batch indexer fixture gap closure" in texts["readme"], "README wiring missing")
    report.check("M13.4: Batch Indexer Fixture Gap Closure" in texts["roadmap"], "roadmap M13.4 missing")
    report.check("M13.5: Embedding/Indexer Route Gate And Closure" in texts["roadmap"], "roadmap M13.5 missing")
    report.check("Earlier M13.4 Batch Indexer Fixture Gap Closure" in texts["status"], "status M13.4 previous item missing")
    report.check("Active item: M13.5 Embedding/Indexer Route Gate And Closure" in texts["status"], "status M13.5 active item missing")
    report.check("#### M13.4: Batch Indexer Fixture Gap Closure" in texts["todo"], "TODO M13.4 missing")
    report.check("#### M13.5: Embedding/Indexer Route Gate And Closure" in texts["todo"], "TODO M13.5 missing")


def run_negative_tests(
    report: Report,
    bundle: dict[str, Any],
    matrix: dict[str, Any],
    sources: dict[str, str],
    texts: dict[str, str],
) -> None:
    mutations = [
        ("route overclaim", lambda obj: mutate_nested(obj, ["claim_policy", "runtime_route_change"], True)),
        ("raw body overclaim", lambda obj: mutate_nested(obj, ["claim_policy", "raw_tensor_bodies_committed"], True)),
        ("missing fixture", lambda obj: mutate_nested(obj, ["fixtures"], obj["fixtures"][:-1])),
        ("unclosed gap", lambda obj: mutate_nested(obj, ["fixtures", 0, "fixture_gap_closed"], False)),
        ("wrong operation", lambda obj: mutate_nested(obj, ["fixtures", 1, "operation"], "ds4_gpu_indexer_topk_tensor")),
        ("missing comp mask hook", lambda obj: mutate_nested(obj, ["fixtures", 2, "debug_dump_hooks"], [])),
        ("wrong next gate", lambda obj: mutate_nested(obj, ["summary", "next_route_gate"], "M13.6")),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(mutated_report, mutate(bundle), matrix, sources, texts, run_commands=False)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_nested(obj: dict[str, Any], path: list[Any], value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    target: Any = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return mutated


def matrix_row(matrix: dict[str, Any], operation: str) -> dict[str, Any] | None:
    rows = matrix.get("matrix")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("operation") == operation:
            return row
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Batch indexer fixture gap closure: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
