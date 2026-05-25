#!/usr/bin/env python3
"""Validate the M12.6 backend replacement closure and removal decision."""

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
CLOSURE = ROOT / "ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json"
M12_1 = ROOT / "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json"
M12_2 = ROOT / "ds4-parity/baselines/backend/m12.2/manifest.json"
M12_3 = ROOT / "ds4-parity/baselines/backend/m12.3/facade-replay.json"
M12_4 = ROOT / "ds4-parity/baselines/backend/m12.4/replacement-slice.json"
M12_5 = ROOT / "ds4-parity/baselines/backend/m12.5/runtime-route-gate.json"
M10_9C = ROOT / "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json"
M10_9D = ROOT / "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json"
M10_9E = ROOT / "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json"
M10_9F = ROOT / "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

ACCEPTED_ACTIVE_ITEMS = [
    "Active item: post-M12 roadmap decision",
    "Active item: M13",
    "Active item: post-M13 roadmap decision",
]
EXPECTED_FAMILIES = [
    "backend_lifecycle",
    "tensor_lifetime",
    "command_buffers",
    "model_mapping",
    "embedding_and_indexer",
    "dense_norm_rope_kv",
    "compressor_attention",
    "routing_moe",
    "hc_output",
]
EXPECTED_CRITERIA = [
    "rust_default_cli_server_flows",
    "official_vector_and_server_tests",
    "long_context_and_tool_quality",
    "old_code_no_longer_reference",
    "docs_and_build_entrypoints",
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
    closure = load_json(CLOSURE)
    inventory = load_json(M12_1)
    manifest = load_json(M12_2)
    replay = load_json(M12_3)
    slice_artifact = load_json(M12_4)
    route_gate = load_json(M12_5)
    runtime_artifacts = {
        "M10.9c": load_json(M10_9C),
        "M10.9d": load_json(M10_9D),
        "M10.9e": load_json(M10_9E),
        "M10.9f": load_json(M10_9F),
    }
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(
        report,
        closure,
        inventory,
        manifest,
        replay,
        slice_artifact,
        route_gate,
        runtime_artifacts,
        texts,
        run_commands=not args.no_commands,
    )
    if args.negative_test:
        run_negative_tests(
            report,
            closure,
            inventory,
            manifest,
            replay,
            slice_artifact,
            route_gate,
            runtime_artifacts,
            texts,
        )
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    closure: dict[str, Any],
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    replay: dict[str, Any],
    slice_artifact: dict[str, Any],
    route_gate: dict[str, Any],
    runtime_artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
    *,
    run_commands: bool,
) -> None:
    validate_closure_root(report, closure)
    validate_sources(report, closure)
    validate_claim_policy(report, closure, route_gate)
    validate_removal_criteria(report, closure)
    validate_operation_decisions(report, closure, inventory, manifest, replay, slice_artifact, route_gate)
    validate_runtime_evidence(report, closure, runtime_artifacts)
    if run_commands:
        run_dependency_checkers(report)
    validate_static_wiring(report, texts)


def validate_closure_root(report: Report, closure: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_replacement_closure.v1",
        "source": "m12.6-backend-replacement-closure",
        "milestone": "M12.6",
        "parent": "M12",
        "previous_stage": "M12.5",
        "next_stage": "post-M12-roadmap-decision",
        "status": "closure-no-removal",
    }
    for key, value in expected.items():
        report.check(closure.get(key) == value, f"closure {key} drift")
    overall = require_dict(report, closure.get("overall_decision"), "overall_decision")
    report.check(overall.get("decision") == "retain_current_backend_and_oracles", "overall decision drift")
    report.check(overall.get("removals_allowed") is False, "overall removals became allowed")
    next_work = overall.get("next_required_work")
    report.check(isinstance(next_work, list) and len(next_work) == 3, "next required work drift")


def validate_sources(report: Report, closure: dict[str, Any]) -> None:
    sources = require_dict(report, closure.get("source_artifacts"), "source_artifacts")
    expected = {
        "boundary_inventory": "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json",
        "operation_fixture_manifest": "ds4-parity/baselines/backend/m12.2/manifest.json",
        "facade_replay": "ds4-parity/baselines/backend/m12.3/facade-replay.json",
        "replacement_slice": "ds4-parity/baselines/backend/m12.4/replacement-slice.json",
        "runtime_route_gate": "ds4-parity/baselines/backend/m12.5/runtime-route-gate.json",
        "runtime_graph_closure": "ds4-parity/baselines/graph/m10.9a/runtime-graph-closure-matrix.json",
        "official_vectors": "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
        "long_context": "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
        "tool_server": "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
        "benchmark_closure": "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
    }
    report.check(sources == expected, "source artifact map drift")
    for key, path in expected.items():
        report.check((ROOT / path).exists(), f"source artifact missing: {key}")


def validate_claim_policy(report: Report, closure: dict[str, Any], route_gate: dict[str, Any]) -> None:
    policy = require_dict(report, closure.get("claim_policy"), "claim_policy")
    report.check(policy.get("default_route_replacement_active") is False, "default route replacement active")
    report.check(policy.get("replacement_route_opt_in_only") is True, "replacement route not opt-in only")
    report.check(policy.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(policy.get("kernel_replacement") is False, "kernel replacement overclaim")
    report.check(policy.get("removals_allowed") is False, "removals allowed")
    report.check(policy.get("current_backend_retained_as_oracle") is True, "current backend oracle retention drift")
    report.check(policy.get("current_backend_retained_as_sidecar") is True, "current backend sidecar drift")
    report.check(
        route_gate.get("default_route_replacement_active") == policy.get("default_route_replacement_active"),
        "route gate default activity drift",
    )
    report.check(route_gate.get("replacement_route_opt_in") is True, "route gate opt-in drift")
    report.check(route_gate.get("general_backend_replacement") is False, "route gate replacement overclaim")
    report.check(route_gate.get("kernel_replacement") is False, "route gate kernel overclaim")


def validate_removal_criteria(report: Report, closure: dict[str, Any]) -> None:
    criteria = closure.get("removal_criteria")
    report.check(isinstance(criteria, list), "removal criteria missing")
    if not isinstance(criteria, list):
        return
    by_id = {item.get("id"): item for item in criteria if isinstance(item, dict)}
    report.check(list(by_id) == EXPECTED_CRITERIA, "removal criteria id drift")
    report.check(by_id["rust_default_cli_server_flows"].get("status") == "partial", "CLI/server criterion drift")
    report.check(
        by_id["official_vector_and_server_tests"].get("status") == "satisfied_for_graph_route",
        "official/server criterion drift",
    )
    report.check(
        by_id["long_context_and_tool_quality"].get("status") == "satisfied_for_b300_graph_route",
        "long/tool criterion drift",
    )
    report.check(by_id["old_code_no_longer_reference"].get("status") == "not_satisfied", "oracle criterion drift")
    report.check(by_id["docs_and_build_entrypoints"].get("status") == "partial", "docs criterion drift")
    for blocked_id in ["rust_default_cli_server_flows", "old_code_no_longer_reference", "docs_and_build_entrypoints"]:
        report.check("removal_blocker" in by_id[blocked_id], f"{blocked_id}: blocker missing")


def validate_operation_decisions(
    report: Report,
    closure: dict[str, Any],
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    replay: dict[str, Any],
    slice_artifact: dict[str, Any],
    route_gate: dict[str, Any],
) -> None:
    decisions = closure.get("operation_family_decisions")
    report.check(isinstance(decisions, list), "operation family decisions missing")
    if not isinstance(decisions, list):
        return
    by_name = {item.get("name"): item for item in decisions if isinstance(item, dict)}
    report.check(list(by_name) == EXPECTED_FAMILIES, "operation family order drift")

    inventory_families = [item.get("name") for item in inventory.get("operation_families", [])]
    report.check(inventory_families == EXPECTED_FAMILIES, "inventory family drift")
    fixture_families = {item.get("operation_family") for item in manifest.get("fixtures", [])}
    replay_families = {item.get("operation_family") for item in replay.get("replays", [])}
    report.check(fixture_families == replay_families, "fixture/replay family drift")
    report.check(fixture_families <= set(EXPECTED_FAMILIES), "fixture family missing from closure")

    for name, decision in by_name.items():
        report.check(decision.get("removal_decision") == "retain_current_backend", f"{name}: removal decision drift")
        report.check(decision.get("replacement_status") != "fully_replaced", f"{name}: full replacement overclaim")
        report.check(decision.get("runtime_route_status") != "default-route", f"{name}: default route overclaim")

    embedding = by_name.get("embedding_and_indexer", {})
    replaced = embedding.get("replaced_operations")
    report.check(replaced == [slice_artifact.get("operation")], "embedding replaced operation drift")
    report.check(route_gate.get("operation") == slice_artifact.get("operation"), "route/slice operation drift")
    report.check(embedding.get("runtime_route_status") == "opt-in-only", "embedding route status drift")
    report.check(embedding.get("replacement_status") == "single_operation_route_gated", "embedding status drift")

    inventory_embedding = family_by_name(inventory, "embedding_and_indexer")
    expected_remaining = [
        op for op in inventory_embedding.get("operations", []) if op != slice_artifact.get("operation")
    ]
    report.check(embedding.get("remaining_operations") == expected_remaining, "embedding remaining operation drift")
    report.check(len(expected_remaining) == 6, "embedding remaining operation count drift")

    for name in ["dense_norm_rope_kv", "compressor_attention", "routing_moe", "hc_output"]:
        decision = by_name.get(name, {})
        report.check(decision.get("coverage") == "m12.2_fixture_m12.3_replay", f"{name}: fixture coverage drift")
        report.check(decision.get("replacement_status") == "fixture_only", f"{name}: replacement status drift")
    for name in ["backend_lifecycle", "tensor_lifetime", "command_buffers", "model_mapping"]:
        decision = by_name.get(name, {})
        report.check(decision.get("coverage") == "inventory_only", f"{name}: inventory coverage drift")
        report.check(decision.get("replacement_status") == "not_replaced", f"{name}: replacement status drift")


def validate_runtime_evidence(
    report: Report,
    closure: dict[str, Any],
    runtime_artifacts: dict[str, dict[str, Any]],
) -> None:
    sources = require_dict(report, closure.get("source_artifacts"), "source_artifacts")
    expected_paths = {
        "M10.9c": sources.get("official_vectors"),
        "M10.9d": sources.get("long_context"),
        "M10.9e": sources.get("tool_server"),
        "M10.9f": sources.get("benchmark_closure"),
    }
    for milestone, data in runtime_artifacts.items():
        report.check(data.get("milestone") == milestone, f"{milestone}: milestone drift")
        report.check(data.get("runtime_graph_route") == "graph", f"{milestone}: route drift")
        report.check(data.get("backend") == "cuda", f"{milestone}: backend drift")
        report.check(expected_paths[milestone] is not None, f"{milestone}: source path missing")

    benchmark = runtime_artifacts["M10.9f"]
    performance = require_dict(report, benchmark.get("performance"), "M10.9f.performance")
    report.check(performance.get("same_session_current_c") == "pass", "same-session benchmark policy drift")
    report.check(performance.get("same_session_regressions") == [], "same-session regression drift")
    claim_boundary = require_dict(report, benchmark.get("claim_boundary"), "M10.9f.claim_boundary")
    report.check(claim_boundary.get("backend_replacement") is False, "benchmark replacement overclaim")
    quality_gates = benchmark.get("quality_gates")
    report.check(isinstance(quality_gates, list) and len(quality_gates) == 5, "quality gate count drift")
    if isinstance(quality_gates, list):
        for gate in quality_gates:
            report.check(isinstance(gate, dict) and gate.get("ok") is True, "quality gate failure")


def run_dependency_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/check_backend_boundary_inventory.py", "--negative-test"],
        ["ds4-parity/check_backend_operation_fixtures.py", "--negative-test"],
        ["ds4-parity/check_backend_facade_replay.py", "--negative-test"],
        ["ds4-parity/check_backend_replacement_slice.py", "--negative-test"],
        ["ds4-parity/check_backend_runtime_route_gate.py", "--negative-test"],
    ]
    for command in commands:
        proc = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        report.check(proc.returncode == 0, f"{command[0]} failed: {proc.stderr or proc.stdout}")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M12.6 Backend replacement closure" in texts["report"], "unified report wiring missing")
    report.check("check_backend_replacement_closure.py" in texts["report"], "report checker path missing")
    report.check("Validate the M12.6 Backend replacement closure" in texts["readme"], "README wiring missing")
    report.check("M12.6: Backend Replacement Closure And Removal Decision" in texts["roadmap"], "roadmap M12.6 missing")
    report.check("- Status: complete." in texts["roadmap"], "roadmap M12.6 complete status missing")
    report.check(
        any(item in texts["status"] for item in ACCEPTED_ACTIVE_ITEMS),
        "status terminal active item missing",
    )
    report.check("Earlier M12.6 Backend Replacement Closure And Removal Decision" in texts["status"], "status M12.6 previous item missing")
    report.check("#### M12.6: Backend Replacement Closure And Removal Decision" in texts["todo"], "TODO M12.6 missing")


def run_negative_tests(
    report: Report,
    closure: dict[str, Any],
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    replay: dict[str, Any],
    slice_artifact: dict[str, Any],
    route_gate: dict[str, Any],
    runtime_artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
) -> None:
    mutations = [
        ("allow removals", lambda obj: mutate_nested(obj, ["claim_policy", "removals_allowed"], True)),
        (
            "default route replacement active",
            lambda obj: mutate_nested(obj, ["claim_policy", "default_route_replacement_active"], True),
        ),
        ("overall removal allowed", lambda obj: mutate_nested(obj, ["overall_decision", "removals_allowed"], True)),
        ("missing family", remove_last_family),
        ("embedding full replacement", mutate_embedding_full_replacement),
        ("old code criterion satisfied", mutate_old_code_satisfied),
        ("wrong route gate source", lambda obj: mutate_nested(obj, ["source_artifacts", "runtime_route_gate"], "missing.json")),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(
            mutated_report,
            mutate(closure),
            inventory,
            manifest,
            replay,
            slice_artifact,
            route_gate,
            runtime_artifacts,
            texts,
            run_commands=False,
        )
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_nested(closure: dict[str, Any], path: list[str], value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(closure)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return mutated


def remove_last_family(closure: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(closure)
    mutated["operation_family_decisions"] = mutated["operation_family_decisions"][:-1]
    return mutated


def mutate_embedding_full_replacement(closure: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(closure)
    for item in mutated["operation_family_decisions"]:
        if item.get("name") == "embedding_and_indexer":
            item["replacement_status"] = "fully_replaced"
            item["remaining_operations"] = []
    return mutated


def mutate_old_code_satisfied(closure: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(closure)
    for item in mutated["removal_criteria"]:
        if item.get("id") == "old_code_no_longer_reference":
            item["status"] = "satisfied"
            item.pop("removal_blocker", None)
    return mutated


def family_by_name(inventory: dict[str, Any], name: str) -> dict[str, Any]:
    for item in inventory.get("operation_families", []):
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label} must be object")
    return obj if isinstance(obj, dict) else {}


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
    print(f"Backend replacement closure: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
