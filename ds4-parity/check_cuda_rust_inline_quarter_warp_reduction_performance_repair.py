#!/usr/bin/env python3
"""Validate the Rust CUDA inline quarter-warp reduction performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-inline-quarter-warp-reduction-performance-repair.json"


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


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "kernels": (ROOT / "rust/ds4-cuda/src/abi_kernels.rs").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA inline quarter-warp reduction repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_inline_quarter_warp_reduction_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-inline-quarter-warp-reduction-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_quarter_warp_helper_inlining", True),
        ("retains_reduction_tree_arithmetic", True),
        ("retains_gate_dp4a_topology", True),
        ("retains_down_dp4a_topology", True),
        ("retains_rowspan_policy", True),
        ("changes_default_current_c_route", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts["kernels"])
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], kernels: str) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    helper = kernels.split("fn abi_moe_quarter_warp_sum", 1)[1].split("fn abi_moe_load_u16", 1)[0]
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("helper", "abi_moe_quarter_warp_sum"),
        ("annotation", "#[inline(always)]"),
        ("parent_gate_dp4a_sites", 128),
        ("repaired_gate_dp4a_sites", 128),
        ("parent_gate_prmt_sites", 16),
        ("repaired_gate_prmt_sites", 16),
        ("parent_gate_quarter_warp_call_sites", 16),
        ("repaired_gate_quarter_warp_call_sites", 0),
        ("parent_gate_shuffle_sites", 0),
        ("repaired_gate_shuffle_sites", 16),
        ("parent_gate_call_uni_sites", 24),
        ("repaired_gate_call_uni_sites", 8),
        ("parent_gate_ld_local_sites", 0),
        ("repaired_gate_ld_local_sites", 0),
        ("parent_gate_st_local_sites", 0),
        ("repaired_gate_st_local_sites", 0),
        ("parent_down_dp4a_sites", 256),
        ("repaired_down_dp4a_sites", 256),
        ("parent_down_quarter_warp_call_sites", 8),
        ("repaired_down_quarter_warp_call_sites", 0),
        ("parent_down_shuffle_sites", 0),
        ("repaired_down_shuffle_sites", 8),
        ("parent_down_call_uni_sites", 8),
        ("repaired_down_call_uni_sites", 0),
        ("parent_down_ld_local_sites", 0),
        ("repaired_down_ld_local_sites", 0),
        ("parent_down_st_local_sites", 0),
        ("repaired_down_st_local_sites", 0),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check("#[inline(always)]\n    fn abi_moe_quarter_warp_sum" in kernels, "inline annotation missing")
    report.check("warp::shuffle_xor_f32(value, offset)" in helper, "quarter-warp shuffle tree missing")
    report.check("offset >>= 1;" in helper, "quarter-warp reduction stride missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "65faf1bff05339dd754e230e211ef97de1e2765cc788edaf1c53b94853e2ed59"),
        ("parent_ptx_sha256", "496c8f743ac6fc9202642cfdc934263f7f35ba4ec364fbf8089fc95ba7abc48b"),
        ("repaired_shared_library_sha256", "8ab594e8d203a744255f81898e4c212750105cbe3d5949935dd6044d9d622534"),
        ("repaired_ptx_sha256", "465374de2525cd2e805f183ff6c3ae6dff5478c8cda68fc921ba14f1da898456"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_confirmation_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, prefill, gateup, down, total in [
        (parent, "parent", 244.90, 720.977, 518.667, 1256.016),
        (repaired, "repaired", 248.12, 714.922, 512.140, 1244.513),
        (current_c, "current-C", None, 517.818, 393.200, 924.283),
    ]:
        if prefill is not None:
            report.check(profile.get("prefill_tps") == prefill, f"{label} prefill drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.008),
        ("down_speedup_over_parent", 1.013),
        ("total_speedup_over_parent", 1.009),
        ("gateup_reduction_percent_over_parent", 0.84),
        ("down_reduction_percent_over_parent", 1.26),
        ("total_reduction_percent_over_parent", 0.92),
        ("prefill_increase_percent_over_parent", 1.31),
        ("repaired_total_ratio_to_current_c", 1.35),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    rejected = require_dict(report, execution.get("rejected_preceding_vsub4_probe"), "rejected vsub4 probe")
    report.check(rejected.get("vsub4_sites") == 16, "rejected vsub4 site count drift")
    report.check("ptxas reports argument mismatch" in rejected.get("decision", ""), "rejected vsub4 reason missing")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "6e48cf952686294f986227991f1c186b3f3992397660cdab121a1f47c01474e7",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Inline Quarter-Warp Reduction Performance Repair"
    checker = "check_cuda_rust_inline_quarter_warp_reduction_performance_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README wiring missing")
    report.check(checker in texts["report"], "report wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    decision = require_dict(report, fixture.get("decision"), "decision")
    report.check(decision.get("default_route") == "retain-current-c", "default route drift")
    report.check(decision.get("rust_cuda_dso_promotion") == "blocked", "promotion drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test-count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 test-count drift")
    report.check(validation.get("unified_report_passed") == 278, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(
        review.get("pre_implementation") == "RISKS_IDENTIFIED_AND_COVERED_BY_PTX_AND_FULL_DEVICE_VALIDATION",
        "pre-review missing",
    )
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost inline", lambda value: value["implementation"].update({"repaired_gate_quarter_warp_call_sites": 16})),
        ("lost down inline", lambda value: value["implementation"].update({"repaired_down_shuffle_sites": 0})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_confirmation_profile"].update({"total_ms": 1256.016})),
        ("promotion overclaim", lambda value: value["decision"].update({"rust_cuda_dso_promotion": "passed"})),
        ("official mismatch", lambda value: value["b300_execution"]["official_vector_probe"].update({"passed": False})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
