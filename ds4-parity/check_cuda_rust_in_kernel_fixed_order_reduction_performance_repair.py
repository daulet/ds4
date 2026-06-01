#!/usr/bin/env python3
"""Validate the Rust CUDA in-kernel fixed-order reduction performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-in-kernel-fixed-order-reduction-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA in-kernel fixed-order reduction repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_in_kernel_fixed_order_reduction_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-in-kernel-fixed-order-reduction-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_reduction_expansion", True),
        ("retains_fixed_order_arithmetic", True),
        ("retains_aligned_q8_staging", True),
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
    macro = kernels.split("macro_rules! abi_moe_quarter_warp_sum_inline", 1)[1]
    macro = macro.split("#[allow(clippy::too_many_arguments, static_mut_refs)]", 1)[0]
    gate = kernels.split("pub fn abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel", 1)[1]
    gate = gate.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel", 1)[0]
    down = kernels.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel", 1)[1]
    down = down.split("pub fn abi_moe_gate_up_mid_f32_kernel", 1)[0]
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("macro", "abi_moe_quarter_warp_sum_inline"),
        ("fixed_shuffle_offsets", [4, 2, 1]),
        ("parent_gate_dp4a_sites", 128),
        ("repaired_gate_dp4a_sites", 128),
        ("parent_gate_reduction_call_sites", 16),
        ("repaired_gate_reduction_call_sites", 0),
        ("parent_gate_shuffle_sites", 0),
        ("repaired_gate_shuffle_sites", 48),
        ("parent_gate_call_uni_sites", 24),
        ("repaired_gate_call_uni_sites", 8),
        ("parent_gate_ld_local_sites", 0),
        ("repaired_gate_ld_local_sites", 0),
        ("parent_gate_st_local_sites", 0),
        ("repaired_gate_st_local_sites", 0),
        ("parent_down_dp4a_sites", 256),
        ("repaired_down_dp4a_sites", 256),
        ("parent_down_reduction_call_sites", 8),
        ("repaired_down_reduction_call_sites", 0),
        ("parent_down_shuffle_sites", 0),
        ("repaired_down_shuffle_sites", 24),
        ("parent_down_call_uni_sites", 8),
        ("repaired_down_call_uni_sites", 0),
        ("parent_down_ld_local_sites", 0),
        ("repaired_down_ld_local_sites", 0),
        ("parent_down_st_local_sites", 0),
        ("repaired_down_st_local_sites", 0),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    for offset in [4, 2, 1]:
        report.check(f"warp::shuffle_xor_f32(value, {offset})" in macro, f"macro offset {offset} missing")
    report.check(gate.count("abi_moe_quarter_warp_sum_inline!(") == 2, "gate macro use drift")
    report.check(down.count("abi_moe_quarter_warp_sum_inline!(") == 1, "down macro use drift")
    report.check("abi_moe_quarter_warp_sum($gate)" not in gate, "gate retained reduction calls")
    report.check("abi_moe_quarter_warp_sum($accumulator)" not in down, "down retained reduction calls")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-06-01"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "a09d4975d1ba6a35cd2d23500ce08c63b2037fd44c49c719acda1b0c98a9815e"),
        ("parent_ptx_sha256", "8660a38acf70582a182f503df48e8890722bbd47599cdb916ef7c87fa6b5f606"),
        ("repaired_shared_library_sha256", "76a3b57a0f59656944be9401ad008c2ea412b25c5ad1b3d293303110e7d4ada2"),
        ("repaired_ptx_sha256", "0304718144f8a00e425b388ca4ee2003240c6c4db9d11ef97d50d2ab102c5f5d"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_confirmation_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, prefill, gateup, down, total in [
        (parent, "parent", 312.37, 618.611, 457.720, 1092.625),
        (repaired, "repaired", 251.35, 604.403, 443.172, 1063.907),
        (current_c, "current-C", None, 517.818, 393.200, 924.283),
    ]:
        if prefill is not None:
            report.check(profile.get("prefill_tps") == prefill, f"{label} prefill drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.024),
        ("down_speedup_over_parent", 1.033),
        ("total_speedup_over_parent", 1.027),
        ("gateup_reduction_percent_over_parent", 2.30),
        ("down_reduction_percent_over_parent", 3.18),
        ("total_reduction_percent_over_parent", 2.63),
        ("repaired_gateup_ratio_to_current_c", 1.17),
        ("repaired_down_ratio_to_current_c", 1.13),
        ("repaired_total_ratio_to_current_c", 1.15),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "638d1cbf55105bfab3b827224fdf877a754756df0493244ba743d5a0f4a090af",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA In-Kernel Fixed-Order Reduction Performance Repair"
    checker = "check_cuda_rust_in_kernel_fixed_order_reduction_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 281, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_UNAVAILABLE_NOT_LOGGED_IN", "pre-review drift")
    report.check(review.get("final") == "CLAUDE_REVIEW_UNAVAILABLE_NOT_LOGGED_IN", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost gate expansion", lambda value: value["implementation"].update({"repaired_gate_shuffle_sites": 0})),
        ("lost down expansion", lambda value: value["implementation"].update({"repaired_down_call_uni_sites": 8})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_confirmation_profile"].update({"total_ms": 1092.625})),
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
