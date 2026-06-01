#!/usr/bin/env python3
"""Validate the Rust CUDA cached Q8 aligned-pair load performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-cached-q8-aligned-pair-load-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA cached Q8 aligned-pair load repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_cached_q8_aligned_pair_load_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-cached-q8-aligned-pair-load-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_q8_scratch_layout", True),
        ("uses_aligned_shared_pair_loads", True),
        ("retains_global_q8_layout", True),
        ("retains_aligned_q8_staging", True),
        ("retains_fixed_order_reduction", True),
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
    gate = kernels.split("pub fn abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel", 1)[1]
    gate = gate.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel", 1)[0]
    down = kernels.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel", 1)[1]
    down = down.split("pub fn abi_moe_gate_up_mid_f32_kernel", 1)[0]
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("global_q8_block_bytes", 292),
        ("cached_q8_data_offset_bytes", 4),
        ("cached_q8_slot_bytes", 296),
        ("shared_alignment_bytes", 8),
        ("paired_load_helper", "abi_moe_cached_load_aligned_u64"),
        ("parent_gate_ld_shared_u32_sites", 128),
        ("repaired_gate_ld_shared_u32_sites", 0),
        ("parent_gate_ld_shared_u64_sites", 8),
        ("repaired_gate_ld_shared_u64_sites", 72),
        ("parent_gate_dp4a_sites", 128),
        ("repaired_gate_dp4a_sites", 128),
        ("parent_down_ld_shared_u32_sites", 256),
        ("repaired_down_ld_shared_u32_sites", 0),
        ("parent_down_ld_shared_u64_sites", 0),
        ("repaired_down_ld_shared_u64_sites", 128),
        ("parent_down_dp4a_sites", 256),
        ("repaired_down_dp4a_sites", 256),
        ("repaired_gate_ld_local_sites", 0),
        ("repaired_gate_st_local_sites", 0),
        ("repaired_down_ld_local_sites", 0),
        ("repaired_down_st_local_sites", 0),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check("const ABI_MOE_Q8_K_BLOCK_BYTES: u64 = 292;" in kernels, "global Q8 ABI drift")
    report.check("const ABI_MOE_CACHED_Q8_DATA_OFFSET: usize" in kernels, "cached Q8 offset missing")
    report.check("const ABI_MOE_CACHED_Q8_BLOCK_BYTES: usize" in kernels, "cached Q8 slot missing")
    report.check("fn abi_moe_cached_load_aligned_u64" in kernels, "aligned pair helper missing")
    report.check("*values.add(offset).cast::<u64>()" in kernels, "aligned pair read missing")
    report.check("fn abi_moe_cached_q8_word" not in kernels, "old cached word consumer retained")
    report.check("abi_moe_cached_q8_pair(" in gate, "gate pair consumer missing")
    report.check("macro_rules! abi_moe_down_accumulate_q2_group" in kernels, "down Q2 macro missing")
    report.check("abi_moe_cached_q8_pair($q8, q8_block" in kernels, "down pair consumer missing")
    report.check("\n            8,\n" in gate, "gate cached alignment missing")
    report.check("\n            8,\n" in down, "down cached alignment missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-06-01"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "76a3b57a0f59656944be9401ad008c2ea412b25c5ad1b3d293303110e7d4ada2"),
        ("parent_ptx_sha256", "0304718144f8a00e425b388ca4ee2003240c6c4db9d11ef97d50d2ab102c5f5d"),
        ("repaired_shared_library_sha256", "a4b72fe595bbc4fa95d90472383190e48f900c55b287deaaa8c1af82359ff64b"),
        ("repaired_ptx_sha256", "80e4b361be9e643175f65c3a085df9deb206ae94481622473e037dc2c0eeb39a"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_confirmation_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, gateup, down, total in [
        (parent, "parent", 604.757, 443.318, 1064.391),
        (repaired, "repaired", 583.429, 430.144, 1029.873),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.037),
        ("down_speedup_over_parent", 1.031),
        ("total_speedup_over_parent", 1.034),
        ("gateup_reduction_percent_over_parent", 3.53),
        ("down_reduction_percent_over_parent", 2.97),
        ("total_reduction_percent_over_parent", 3.24),
        ("repaired_gateup_ratio_to_current_c", 1.13),
        ("repaired_down_ratio_to_current_c", 1.09),
        ("repaired_total_ratio_to_current_c", 1.11),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "32cba47bbf71e5d55f33d4e545c54007e2381ec5349f405978dea32c5443e51b",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Cached Q8 Aligned-Pair Load Performance Repair"
    checker = "check_cuda_rust_cached_q8_aligned_pair_load_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 282, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review drift")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review drift")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost pair width", lambda value: value["implementation"].update({"repaired_down_ld_shared_u64_sites": 0})),
        ("lost padded alignment", lambda value: value["implementation"].update({"shared_alignment_bytes": 4})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_confirmation_profile"].update({"total_ms": 1064.391})),
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
