#!/usr/bin/env python3
"""Validate the Rust CUDA promotion benchmark performance blocker leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-promotion-benchmark-performance-blocker.json"
PREDECESSOR = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba/rust-cuda-backend-identity-log-compatibility-repair.json"


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


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} missing")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, label: str) -> list[Any]:
    report.check(isinstance(value, list), f"{label} missing")
    return value if isinstance(value, list) else []


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    texts = {
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(encoding="utf-8"),
        "todo": (ROOT / ".memory/TODO.md").read_text(encoding="utf-8"),
        "status": (ROOT / ".memory/status.md").read_text(encoding="utf-8"),
        "readme": (ROOT / "ds4-parity/README.md").read_text(encoding="utf-8"),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(encoding="utf-8"),
    }
    report = Report()
    validate(report, fixture, predecessor, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, predecessor, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA promotion benchmark performance blocker: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: Report,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_promotion_benchmark_performance_blocker.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-blocked-same-session-performance-regression", "status drift")
    validate_boundary(report, fixture, predecessor)
    validate_execution(report, fixture)
    validate_decision(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_boundary(report: Report, fixture: dict[str, Any], predecessor: dict[str, Any]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("benchmark_evidence_captured", True),
        ("candidate_processes_completed", True),
        ("candidate_backend_identity_markers_present", True),
        ("retained_quality_gates_passed", True),
        ("same_session_performance_gate_passed", False),
        ("default_current_c_route_preserved", True),
        ("runtime_route_promoted", False),
        ("c_host_removal_allowed", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    contract = require_dict(report, fixture.get("rerun_contract"), "rerun_contract")
    predecessor_execution = require_dict(report, predecessor.get("b300_execution"), "predecessor execution")
    report.check(
        contract.get("shared_library_sha256") == predecessor_execution.get("shared_library_sha256"),
        "benchmark did not reuse repaired DSO",
    )
    report.check(contract.get("engine_feature") == "cuda-rust-backend", "engine feature drift")
    report.check(contract.get("runtime_graph_route") == "graph", "graph route drift")
    report.check(contract.get("capture_used_prebuilt_candidate") is True, "prebuilt capture boundary missing")
    report.check(
        contract.get("legacy_wrapper_build_metadata_expected_to_fail") is True,
        "wrapper metadata boundary missing",
    )


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("wall_elapsed_seconds", 903),
        ("summary_sha256", "c62eee8b9f9c335db55841044f86619fe08b072249be3c07a1aba87a8d749180"),
        ("legacy_wrapper_primary_check_count", 366),
        ("legacy_wrapper_negative_check_count", 8),
        ("legacy_wrapper_exit_code", 1),
        ("gpu_utilization_percent_after_capture", 0),
        ("gpu_memory_mib_after_capture", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    report.check("--no-build" in execution.get("legacy_wrapper_expected_metadata_failure", ""), "metadata failure explanation missing")
    quality = require_dict(report, execution.get("quality_gates"), "quality_gates")
    report.check(quality == {f"M10.9{suffix}": True for suffix in ["a", "b", "c", "d", "e"]}, "quality gate drift")
    for label in ["candidate_backend_identity_marker_present", "current_c_backend_identity_marker_present"]:
        markers = require_dict(report, execution.get(label), label)
        report.check(markers == {"short": True, "long": True}, f"{label} drift")
    validate_rows(report, execution)
    performance = require_dict(report, execution.get("performance"), "performance")
    for key, expected in [
        ("same_session_max_regression", 0.05),
        ("same_session_status", "regression"),
        ("measured_throughput_field_count", 14),
        ("same_session_regression_count", 14),
    ]:
        report.check(performance.get(key) == expected, f"performance drift: {key}")
    report.check(performance.get("prefill_ratio_to_current_c_range") == [0.0607, 0.0718], "prefill ratio drift")
    report.check(performance.get("decode_ratio_to_current_c_range") == [0.4213, 0.4433], "decode ratio drift")


def validate_rows(report: Report, execution: dict[str, Any]) -> None:
    candidate = require_dict(report, execution.get("candidate_rows"), "candidate_rows")
    current_c = require_dict(report, execution.get("current_c_rows"), "current_c_rows")
    expected_contexts = {
        "b300-short.csv": [2048, 4096, 6144, 8192],
        "b300-long.csv": [16384, 24576, 32768],
    }
    regression_count = 0
    for name, contexts in expected_contexts.items():
        candidate_rows = require_list(report, candidate.get(name), f"candidate.{name}")
        current_rows = require_list(report, current_c.get(name), f"current_c.{name}")
        report.check([row.get("ctx_tokens") for row in candidate_rows] == contexts, f"{name} candidate shape drift")
        report.check([row.get("ctx_tokens") for row in current_rows] == contexts, f"{name} current-C shape drift")
        for candidate_row, current_row in zip(candidate_rows, current_rows):
            for static_key in ["ctx_tokens", "prefill_tokens", "gen_tokens", "kvcache_bytes"]:
                report.check(candidate_row.get(static_key) == current_row.get(static_key), f"{name} workload drift: {static_key}")
            for field in ["prefill_tps", "gen_tps"]:
                threshold = current_row.get(field, 0) * 0.95
                if candidate_row.get(field, 0) < threshold:
                    regression_count += 1
    report.check(regression_count == 14, "same-session regression count not reproduced from rows")


def validate_decision(report: Report, fixture: dict[str, Any]) -> None:
    decision = require_dict(report, fixture.get("decision"), "decision")
    for key, expected in [
        ("default_route", "retain-current-c"),
        ("rust_cuda_dso_promotion", "blocked"),
        ("c_cuda_removal", "blocked"),
        ("c_host_removal", "not-in-scope-and-not-allowed"),
    ]:
        report.check(decision.get(key) == expected, f"decision drift: {key}")
    report.check("all measured throughput fields" in decision.get("blocker", ""), "performance blocker missing")
    report.check("performance gap" in decision.get("required_next_work", ""), "next work missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Promotion Benchmark Performance Blocker"
    checker = "check_cuda_rust_promotion_benchmark_performance_blocker.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_comparator_negative_test_passed") is True, "comparator validation missing")
    report.check(validation.get("unified_report_passed") == 264, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified fail count drift")
    review = require_dict(report, fixture.get("review"), "review")
    for key in ["pre_execution", "follow_on_decision", "final"]:
        report.check(review.get(key) == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", f"{key} review evidence missing")


def run_negative_tests(
    report: Report,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("performance promotion overclaim", lambda value: value["ownership"].update({"same_session_performance_gate_passed": True})),
        ("route promotion overclaim", lambda value: value["decision"].update({"rust_cuda_dso_promotion": "passed"})),
        ("hidden regression", lambda value: value["b300_execution"]["candidate_rows"]["b300-short.csv"][0].update({"prefill_tps": 1460.83})),
        ("lost marker", lambda value: value["b300_execution"]["candidate_backend_identity_marker_present"].update({"long": False})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, predecessor, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
