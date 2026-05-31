#!/usr/bin/env python3
"""Validate the Rust CUDA cached MoE dispatch performance repair leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-cached-moe-dispatch-performance-repair.json"


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
    texts = {
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(encoding="utf-8"),
        "todo": (ROOT / ".memory/TODO.md").read_text(encoding="utf-8"),
        "status": (ROOT / ".memory/status.md").read_text(encoding="utf-8"),
        "readme": (ROOT / "ds4-parity/README.md").read_text(encoding="utf-8"),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(encoding="utf-8"),
    }
    report = Report()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA cached MoE dispatch performance repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_cached_moe_dispatch_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-bounded-moe-improvement-official-vectors",
        "status drift",
    )
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_decision(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_rust_cuda_moe_dispatch", True),
        ("uses_existing_cached_kernel_surfaces", True),
        ("changes_kernel_implementation", False),
        ("bounded_performance_improvement_observed", True),
        ("official_vector_gate_preserved", True),
        ("same_session_promotion_gate_reexecuted", False),
        ("default_current_c_route_preserved", True),
        ("runtime_route_promoted", False),
        ("c_host_removal_allowed", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("source") == "rust/ds4-cuda/src/abi.rs", "source drift")
    report.check(implementation.get("public_entry") == "ds4_gpu_routed_moe_batch_tensor", "entry drift")
    abi = texts["abi"].split('pub unsafe extern "C" fn ds4_gpu_routed_moe_batch_tensor(', 1)[1]
    report.check("&& xq_blocks <= 16" in abi, "bounded gate-cache dispatch missing")
    report.check(
        ".moe_gate_up_mid_expert_tile8_rowspan_cached_tensor(" in abi,
        "cached gate row-span call missing",
    )
    report.check(
        ".moe_gate_up_mid_expert_tile8_rowspan_tensor(" in abi,
        "uncached gate fallback missing",
    )
    report.check("if use_down_rowspan && midq_blocks <= 8" in abi, "bounded down-cache dispatch missing")
    report.check(
        ".moe_down_expert_tile16_rowspan_cached_tensor(" in abi,
        "cached down row-span call missing",
    )
    report.check(
        ".moe_down_expert_tile16_rowspan_tensor(" in abi,
        "uncached down fallback missing",
    )


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("pre_repair_shared_library_sha256", "223542460727037720a65a43961692ff9b42bbd75ddabea16344eb1094e69903"),
        ("repaired_shared_library_sha256", "0588303b251de67fdb9963436c4b6bb5acf8f11f0478e83f771e68a46873a174"),
        ("official_vector_binary_sha256", "15dca8e084e787bc7beceee2e705142ccbfd1a63d16260582469fa083572b9fa"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    profile = require_dict(report, execution.get("bounded_profile"), "bounded_profile")
    before = require_dict(report, profile.get("pre_repair_rust"), "pre_repair_rust")
    after = require_dict(report, profile.get("repaired_rust"), "repaired_rust")
    current_c = require_dict(report, profile.get("current_c_reference"), "current_c_reference")
    effect = require_dict(report, profile.get("repair_effect"), "repair_effect")
    report.check(before.get("routed_moe_ms") == 16515.705, "pre-repair MoE profile drift")
    report.check(after.get("routed_moe_ms") == 12225.556, "repaired MoE profile drift")
    report.check(after.get("prefill_tps") == 105.98, "repaired throughput drift")
    report.check(current_c.get("routed_moe_ms") == 925.079, "current-C profile drift")
    report.check(effect.get("rust_prefill_tps_increase_percent") == 21.6, "prefill improvement drift")
    report.check(effect.get("rust_routed_moe_time_reduction_percent") == 26.0, "MoE improvement drift")
    report.check(effect.get("remaining_rust_to_current_c_routed_moe_ratio") == 13.22, "remaining MoE gap drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official_vector_probe")
    for key, expected in [
        ("runtime_graph_route", "graph"),
        ("comparator_check_count", 1958),
        ("negative_check_count", 8),
        ("case_count", 5),
        ("exercised_case_count", 4),
        ("selected_step_count", 13),
        ("selected_match_count", 13),
        ("passed", True),
    ]:
        report.check(official.get(key) == expected, f"official-vector evidence drift: {key}")
    report.check(
        require_list(report, official.get("known_skipped_cases"), "known_skipped_cases")
        == [{"id": "long_memory_archive", "reason": "API/official graph mismatch"}],
        "known skip drift",
    )


def validate_decision(report: Report, fixture: dict[str, Any]) -> None:
    decision = require_dict(report, fixture.get("decision"), "decision")
    for key, expected in [
        ("default_route", "retain-current-c"),
        ("rust_cuda_dso_promotion", "blocked"),
        ("c_cuda_removal", "blocked"),
        ("c_host_removal", "not-in-scope-and-not-allowed"),
    ]:
        report.check(decision.get(key) == expected, f"decision drift: {key}")
    report.check("does not close" in decision.get("bounded_repair_result", ""), "remaining gap missing")
    report.check("remaining slow Rust CUDA kernel families" in decision.get("required_next_work", ""), "next work missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Cached MoE Dispatch Performance Repair"
    checker = "check_cuda_rust_cached_moe_dispatch_performance_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 feature test count drift")
    report.check(validation.get("unified_report_passed") == 265, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified fail count drift")
    review = require_dict(report, fixture.get("review"), "review")
    for key in ["pre_implementation", "final"]:
        report.check(review.get(key) == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", f"{key} review evidence missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("performance improvement hidden", lambda value: value["b300_execution"]["bounded_profile"]["repaired_rust"].update({"routed_moe_ms": 16515.705})),
        ("official correctness overclaim", lambda value: value["b300_execution"]["official_vector_probe"].update({"selected_match_count": 12})),
        ("route promotion overclaim", lambda value: value["decision"].update({"rust_cuda_dso_promotion": "passed"})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
