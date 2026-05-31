#!/usr/bin/env python3
"""Validate the Rust CUDA cached-down Q2-group expansion performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-down-q2-group-expansion-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA down Q2-group expansion repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_down_q2_group_expansion_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-down-q2-group-expanded-route-blocked", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_down_q2_group_shape", True),
        ("expands_inner_q2_groups", True),
        ("preserves_outer_chunk_order", True),
        ("retains_down_load_expansion", True),
        ("retains_gate_accumulation_order", True),
        ("retains_multi_pair_down_dp4a", True),
        ("changes_default_dispatch", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
        ("default_current_c_route_preserved", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts["kernels"])
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], kernels: str) -> None:
    group_macro = kernels.split("macro_rules! abi_moe_down_accumulate_q2_group", 1)[1].split(
        "#[allow(clippy::too_many_arguments, static_mut_refs)]", 1
    )[0]
    down = kernels.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel(", 1)[1].split(
        "pub fn abi_moe_gate_up_mid_f32_kernel(", 1
    )[0]
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("macro_rules! abi_moe_down_load_u32" in kernels, "retained down load macro missing")
    report.check("macro_rules! abi_moe_down_accumulate_q2_group" in kernels, "Q2-group macro missing")
    report.check(group_macro.count("abi_moe_down_load_u32!(") == 8, "Q2-group load shape drift")
    report.check(group_macro.count("integer::dp4a_i8(") == 8, "Q2-group DP4A shape drift")
    report.check(down.count("abi_moe_down_accumulate_q2_group!(") == 4, "fixed Q2-group calls drift")
    report.check("while chunk < 2" in down, "ordered outer chunk loop missing")
    report.check("while group < 4" not in down, "dynamic Q2-group loop retained")
    for key, expected in [
        ("fixed_group_invocation_count_per_dynamic_chunk", 4),
        ("dynamic_outer_chunk_count", 2),
        ("parent_ptx_dp4a_site_count", 8),
        ("parent_ptx_local_store_count", 51),
        ("parent_ptx_local_load_count", 6),
        ("parent_ptx_call_instruction_count", 1),
        ("repaired_ptx_dp4a_site_count", 32),
        ("repaired_ptx_local_store_count", 54),
        ("repaired_ptx_local_load_count", 9),
        ("repaired_ptx_call_instruction_count", 1),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_profiled_shared_library_sha256", "53717e57162ad2e0eedff267d3f22fa4a8246c5c065f7ad899ef544bd9894e91"),
        ("repaired_profiled_shared_library_sha256", "f0e72d3aa68715171e8caf2673048ad3e16cb432fae3267da4811454bbd1f856"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    ptx = require_dict(report, execution.get("ptx_shape"), "PTX shape")
    for key, expected in [
        ("parent_function_sha256", "57dad6696abbc392f73de19db9815354f494a971d876df6b1ce7a2560f969c7a"),
        ("repaired_function_sha256", "b1f4318ab13b04e0cd135ae9f96d84c84de5425d301dab33e447eae786e17f0f"),
        ("parent_dp4a_site_count", 8),
        ("repaired_dp4a_site_count", 32),
        ("parent_local_store_count", 51),
        ("repaired_local_store_count", 54),
        ("parent_local_load_count", 6),
        ("repaired_local_load_count", 9),
    ]:
        report.check(ptx.get(key) == expected, f"PTX attribution drift: {key}")
    parent = require_dict(report, execution.get("parent_rebuild_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_rebuild_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, gateup, down, total in [
        (parent, "parent", 1534.338, 1116.221, 2666.890),
        (repaired, "repaired", 1533.245, 905.109, 2454.643),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("down_speedup_over_parent", 1.233),
        ("total_speedup_over_parent", 1.086),
        ("down_reduction_percent_over_parent", 18.91),
        ("total_reduction_percent_over_parent", 7.96),
        ("repaired_gateup_ratio_to_current_c", 2.96),
        ("repaired_down_ratio_to_current_c", 2.30),
        ("repaired_total_ratio_to_current_c", 2.66),
        ("remaining_primary_bottleneck", "cached-gateup-and-down-native-codegen-gap"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "cc7096f2aa530fb67c0e7f9d73175c83344f294aa9788e04294b2c09b77bd503",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Down Q2-Group Expansion Performance Repair"
    checker = "check_cuda_rust_down_q2_group_expansion_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 268, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost static groups", lambda value: value["implementation"].update({"repaired_ptx_dp4a_site_count": 8})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_rebuild_profile"].update({"down_ms": 1116.221})),
        ("dispatch overclaim", lambda value: value["ownership"].update({"changes_default_dispatch": True})),
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
