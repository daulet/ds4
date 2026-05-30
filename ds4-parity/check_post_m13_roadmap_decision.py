#!/usr/bin/env python3
"""Validate the post-M13 roadmap decision and completion boundary."""

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
DECISION = ROOT / "ds4-parity/baselines/roadmap/post-m13/post-m13-roadmap-decision.json"
M13_0 = ROOT / "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json"
M13_1 = ROOT / "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json"
M13_2 = ROOT / "ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json"
M13_3 = ROOT / "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json"
M13_4 = ROOT / "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json"
M13_5_GATE = ROOT / "ds4-parity/baselines/backend/m13.5/expanded-route-gate.json"
M13_5_CLOSURE = ROOT / "ds4-parity/baselines/backend/m13.5/expanded-route-closure.json"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
LESSONS = ROOT / ".memory/lessons.md"
PROTOCOL = ROOT / ".memory/protocol.md"

EXPECTED_SOURCE_ARTIFACTS = {
    "backend_expansion_decision": "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json",
    "embedding_indexer_matrix": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
    "batched_embedding_slice": "ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json",
    "indexed_decode_slices": "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json",
    "batch_indexer_fixture_bundle": "ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json",
    "expanded_route_gate": "ds4-parity/baselines/backend/m13.5/expanded-route-gate.json",
    "expanded_route_closure": "ds4-parity/baselines/backend/m13.5/expanded-route-closure.json",
    "runtime_official_vectors": "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
    "runtime_long_context": "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
    "runtime_tool_server": "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
    "runtime_benchmark_closure": "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
}
EXPECTED_COMPLETED_STAGES = [
    "M13.0 Backend Expansion Decision",
    "M13.1 Embedding/Indexer Expansion Fixture Matrix",
    "M13.2 Batched Embedding Replacement Slice",
    "M13.3 Indexed Decode Selection Replacement Slice",
    "M13.4 Batch Indexer Fixture Gap Closure",
    "M13.5 Embedding/Indexer Route Gate And Closure",
]
EXPECTED_OPEN_DECISIONS = [
    "Whether the Rust CLI/server should initially call into a C `ds4_engine` shim or wait until graph orchestration is ported.",
    "How much of the current CPU reference path should be preserved in Rust.",
    "Whether GGUF tooling should join the same workspace or stay as separate C and Python utilities until the runtime port stabilizes.",
    "How to version KV persistence if Rust needs to change the on-disk structure.",
    "Which intermediate tensor checks are worth keeping as permanent diagnostics.",
    "What numeric tolerances should be used per backend and per operation family.",
    "What speed regression threshold is acceptable for each backend milestone.",
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
        "decision": load_json(DECISION),
        "m13_0": load_json(M13_0),
        "m13_1": load_json(M13_1),
        "m13_2": load_json(M13_2),
        "m13_3": load_json(M13_3),
        "m13_4": load_json(M13_4),
        "m13_5_gate": load_json(M13_5_GATE),
        "m13_5_closure": load_json(M13_5_CLOSURE),
    }
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
        "lessons": read_text(LESSONS),
        "protocol": read_text(PROTOCOL),
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
    decision = artifacts["decision"]
    validate_decision_artifact(report, decision)
    validate_source_artifacts(report, decision, artifacts)
    validate_m13_chain(report, artifacts)
    validate_removal_boundary(report, decision, artifacts)
    validate_static_wiring(report, decision, texts)
    if run_commands:
        run_dependency_checkers(report)


def validate_decision_artifact(report: Report, decision: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.post_m13_roadmap_decision.v1",
        "milestone": "post-M13",
        "status": "roadmap-complete-no-active-implementation-item",
        "id": "post-m13-roadmap-decision",
    }
    for key, value in expected.items():
        report.check(decision.get(key) == value, f"decision {key} drift")
    report.check(decision.get("source_artifacts") == EXPECTED_SOURCE_ARTIFACTS, "source artifact drift")
    report.check(decision.get("completed_stages") == EXPECTED_COMPLETED_STAGES, "completed stage drift")
    inner = require_dict(report, decision.get("decision"), "decision")
    expected_decision = {
        "active_item_before_decision": "post-M13 roadmap decision",
        "active_item_after_decision": "none",
        "roadmap_scope_completed": "RUST_PORT_ROADMAP.md through M13.5",
        "next_implementation_stage_selected": False,
        "default_route_promotion_allowed": False,
        "c_host_removal_allowed": False,
        "gpu_backend_removal_allowed": False,
        "general_backend_replacement_claim_allowed": False,
        "kernel_replacement_claim_allowed": False,
    }
    for key, value in expected_decision.items():
        report.check(inner.get(key) == value, f"decision boundary drift: {key}")
    report.check(isinstance(inner.get("reason"), str) and "current-backend" in inner["reason"], "decision reason missing boundary")
    report.check(decision.get("deferred_open_decisions") == EXPECTED_OPEN_DECISIONS, "open decision drift")
    policy = require_dict(report, decision.get("future_work_policy"), "future_work_policy")
    for key in [
        "requires_new_roadmap",
        "must_start_with_oracle",
        "must_keep_current_backend_oracles_until_default_route_is_proven",
        "must_not_remove_current_backend_sidecars_from_m13_5",
    ]:
        report.check(policy.get(key) is True, f"future work policy missing: {key}")
    overall = require_dict(report, decision.get("overall_decision"), "overall_decision")
    report.check(overall.get("decision") == "roadmap_complete_through_m13_defer_new_scope", "overall decision drift")
    report.check(overall.get("remaining_work") == "future roadmap only", "remaining work overclaim")
    report.check(overall.get("removals_allowed") is False, "overall removal overclaim")


def validate_source_artifacts(
    report: Report,
    decision: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    sources = decision.get("source_artifacts")
    report.check(isinstance(sources, dict), "source artifacts missing")
    if isinstance(sources, dict):
        for key, value in sources.items():
            report.check(isinstance(value, str) and (ROOT / value).exists(), f"source artifact missing: {key}")
    milestones = {
        "m13_0": "M13.0",
        "m13_1": "M13.1",
        "m13_2": "M13.2",
        "m13_3": "M13.3",
        "m13_4": "M13.4",
        "m13_5_gate": "M13.5",
        "m13_5_closure": "M13.5",
    }
    for key, milestone in milestones.items():
        report.check(artifacts[key].get("milestone") == milestone, f"{key} milestone drift")


def validate_m13_chain(report: Report, artifacts: dict[str, dict[str, Any]]) -> None:
    report.check(artifacts["m13_0"].get("next_stage") == "M13.1", "M13.0 next stage drift")
    report.check(artifacts["m13_1"].get("next_stage") == "M13.2", "M13.1 next stage drift")
    report.check(
        artifacts["m13_2"].get("next_required_gate") == "M13.3 Indexed Decode Selection Replacement Slice",
        "M13.2 next gate drift",
    )
    report.check(
        artifacts["m13_3"].get("next_required_gate") == "M13.4 Batch Indexer Fixture Gap Closure",
        "M13.3 next gate drift",
    )
    report.check(artifacts["m13_4"].get("next_stage") == "M13.5", "M13.4 next stage drift")
    report.check(
        artifacts["m13_5_gate"].get("next_required_gate") == "post-M13 roadmap decision",
        "M13.5 gate next decision drift",
    )
    closure = artifacts["m13_5_closure"]
    report.check(closure.get("next_stage") == "post-M13-roadmap-decision", "M13.5 closure next stage drift")
    summary = require_dict(report, closure.get("summary"), "M13.5.summary")
    report.check(summary.get("next_required_gate") == "post-M13 roadmap decision", "M13.5 summary next gate drift")


def validate_removal_boundary(
    report: Report,
    decision: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    closure_policy = require_dict(report, artifacts["m13_5_closure"].get("claim_policy"), "M13.5.claim_policy")
    report.check(closure_policy.get("default_route_replacement_active") is False, "M13.5 default route overclaim")
    report.check(closure_policy.get("general_backend_replacement") is False, "M13.5 backend overclaim")
    report.check(closure_policy.get("kernel_replacement") is False, "M13.5 kernel overclaim")
    report.check(closure_policy.get("removals_allowed") is False, "M13.5 removal overclaim")
    report.check(closure_policy.get("current_backend_retained_as_oracle") is True, "M13.5 oracle retention drift")
    report.check(closure_policy.get("current_backend_retained_as_sidecar") is True, "M13.5 sidecar retention drift")
    criteria = decision.get("removal_criteria_evaluation")
    report.check(isinstance(criteria, list) and len(criteria) == 5, "removal criteria evaluation drift")
    if isinstance(criteria, list):
        statuses = [item.get("status") for item in criteria if isinstance(item, dict)]
        report.check("not-satisfied" in statuses, "removal criteria need an unmet blocker")
        report.check(all(status != "satisfied" for status in statuses), "removal criterion overclaim")
        for item in criteria:
            report.check(isinstance(item, dict) and item.get("reason"), "removal criterion reason missing")


def validate_static_wiring(report: Report, decision: dict[str, Any], texts: dict[str, str]) -> None:
    path = "ds4-parity/baselines/roadmap/post-m13/post-m13-roadmap-decision.json"
    report.check("Post-M13 Roadmap Decision" in texts["roadmap"], "roadmap post-M13 section missing")
    report.check(path in texts["roadmap"], "roadmap decision artifact missing")
    report.check("Post-M13 Roadmap Decision" in texts["todo"], "TODO post-M13 section missing")
    report.check(path in texts["todo"], "TODO decision artifact missing")
    report.check("Validate the post-M13 roadmap decision" in texts["readme"], "README post-M13 wiring missing")
    report.check("check_post_m13_roadmap_decision.py" in texts["readme"], "README checker path missing")
    report.check("Post-M13 roadmap decision" in texts["report"], "unified report post-M13 item missing")
    report.check("check_post_m13_roadmap_decision.py" in texts["report"], "report checker path missing")
    report.check("Earlier post-M13 roadmap decision." in texts["status"], "status previous post-M13 marker missing")
    report.check("Active debugging ledger: none" in texts["status"], "debugging ledger not closed")
    report.check("Use `.memory/lessons.md`" in texts["protocol"], "protocol lessons rule missing")
    report.check("DS4 Rust Port Lessons" in texts["lessons"], "lessons document missing")
    roadmap_text = normalize_ws(texts["roadmap"])
    for open_decision in decision.get("deferred_open_decisions", []):
        report.check(normalize_ws(open_decision) in roadmap_text, f"roadmap open decision missing: {open_decision}")


def run_dependency_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/check_backend_expanded_route_closure.py", "--negative-test"],
        ["ds4-parity/check_backend_batch_indexer_fixtures.py", "--negative-test"],
        ["ds4-parity/check_backend_runtime_route_gate.py", "--negative-test"],
    ]
    for command in commands:
        proc = subprocess.run([sys.executable, *command], cwd=ROOT, text=True, capture_output=True)
        report.check(proc.returncode == 0, f"{command[0]} failed: {proc.stderr or proc.stdout}")


def run_negative_tests(
    report: Report,
    artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
) -> None:
    mutations = [
        ("next stage selected", lambda obj: mutate_nested(obj, ["decision", "decision", "next_implementation_stage_selected"], True)),
        ("default route promotion", lambda obj: mutate_nested(obj, ["decision", "decision", "default_route_promotion_allowed"], True)),
        ("removals allowed", lambda obj: mutate_nested(obj, ["decision", "overall_decision", "removals_allowed"], True)),
        ("missing source artifact", lambda obj: mutate_nested(obj, ["decision", "source_artifacts", "expanded_route_closure"], "missing.json")),
        ("wrong completed stages", lambda obj: mutate_nested(obj, ["decision", "completed_stages"], obj["decision"]["completed_stages"][:-1])),
        ("dropped open decision", lambda obj: mutate_nested(obj, ["decision", "deferred_open_decisions"], obj["decision"]["deferred_open_decisions"][:-1])),
        ("M13.5 removal overclaim", lambda obj: mutate_nested(obj, ["m13_5_closure", "claim_policy", "removals_allowed"], True)),
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


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label} must be object")
    return obj if isinstance(obj, dict) else {}


def normalize_ws(value: str) -> str:
    return " ".join(value.split())


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
    print(f"Post-M13 roadmap decision: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
