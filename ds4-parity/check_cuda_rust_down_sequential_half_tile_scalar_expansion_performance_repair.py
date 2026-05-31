#!/usr/bin/env python3
"""Validate the Rust CUDA down sequential half-tile scalar expansion repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-down-sequential-half-tile-scalar-expansion-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA down sequential half-tile scalar expansion repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema")
        == "ds4.cuda_rust_down_sequential_half_tile_scalar_expansion_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status")
        == "b300-pass-rust-down-sequential-half-tile-scalar-expansion-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_down_accumulation_schedule", True),
        ("processes_sequential_half_tiles", True),
        ("uses_named_scalar_down_accumulators", True),
        ("retains_aligned_cached_q8_loads", True),
        ("retains_q2_arithmetic", True),
        ("retains_gate_kernel_codegen", True),
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
    down = kernels.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel", 1)[1]
    down = down.split("pub fn abi_moe_gate_up_mid_f32_kernel", 1)[0]
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("scalar_half_tile_entries", 8),
        ("rejected_full_tile_entries", 16),
        ("parent_gate_kernel_ptx_sha256", "3cb04ca64e08e76854813332ce1ee421483134ea4adbff071d9bad01de8122a1"),
        ("repaired_gate_kernel_ptx_sha256", "3cb04ca64e08e76854813332ce1ee421483134ea4adbff071d9bad01de8122a1"),
        ("parent_down_dp4a_sites", 32),
        ("repaired_down_dp4a_sites", 256),
        ("parent_down_ld_local_sites", 9),
        ("repaired_down_ld_local_sites", 0),
        ("parent_down_st_local_sites", 54),
        ("repaired_down_st_local_sites", 0),
        ("parent_down_b32_register_bound", 370),
        ("repaired_down_b32_register_bound", 1530),
        ("rejected_full_tile_down_dp4a_sites", 512),
        ("rejected_full_tile_down_b32_register_bound", 2797),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check("let mut entry_base = 0_u32;" in down, "half-tile entry base missing")
    report.check("while entry_base < np {" in down, "half-tile traversal missing")
    report.check("entry_base += 8;" in down, "half-tile stride missing")
    report.check(down.count("let mut accumulator") == 8, "down scalar accumulator count drift")
    report.check(down.count("let mut min_sum") == 8, "down minimum scalar count drift")
    report.check(down.count("let mut quant_sum") == 8, "down quant scalar count drift")
    report.check("let mut accumulators = [0.0_f32; 16];" not in down, "array accumulator path retained")
    report.check("let mut min_sums = [0_i32; 16];" not in down, "array minimum path retained")
    report.check("let mut quant_sums = [0_i32; 16];" not in down, "array quant path retained")
    report.check("abi_moe_cached_load_aligned_u32" in kernels, "aligned Q8 parent repair missing")
    report.check("accumulate_entry!(7, $sum7);" in kernels, "fixed eight-entry Q2 group expansion missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "ccc6dc7fe28acb3edd9d11c22198f1bd2eed7a451e1a7bbc67b1d3c4a3d3dfbc"),
        ("parent_ptx_sha256", "e97c1c1f18d3329625fe8ea737aea11873db6335609d1a569d1693b832addb6b"),
        ("repaired_shared_library_sha256", "959f7d4c262a9efbf528a4b53261834a598b01941c68d8c3027e6e452fdfa275"),
        ("repaired_ptx_sha256", "df8e1b5eceb9ebb7d626404deecdbc4785f48e7b773e0246f26297a3aba838af"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_confirmation_profile"), "repaired profile")
    for profile, label, prefill, gateup, down, total in [
        (parent, "parent", 241.19, 752.737, 632.769, 1401.795),
        (repaired, "repaired", 244.16, 752.848, 518.360, 1287.615),
    ]:
        report.check(profile.get("prefill_tps") == prefill, f"{label} prefill drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("interleaved_attribution"), "attribution")
    for key, expected in [
        ("down_speedup_over_parent", 1.221),
        ("total_speedup_over_parent", 1.089),
        ("down_reduction_percent_over_parent", 18.08),
        ("total_reduction_percent_over_parent", 8.15),
        ("prefill_increase_percent_over_parent", 1.23),
        ("repaired_gateup_ratio_to_current_c", 1.45),
        ("repaired_down_ratio_to_current_c", 1.32),
        ("repaired_total_ratio_to_current_c", 1.39),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    rejected = require_dict(report, execution.get("rejected_full_tile_scalar_probe"), "rejected full-tile probe")
    report.check(rejected.get("down_ms") == 912.140, "rejected full-tile down drift")
    report.check(rejected.get("total_ms") == 1681.666, "rejected full-tile total drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "a6467387e86968bbdcf198cf6b28fc2476f4f3d4591be136edbf7864504f924c",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Down Sequential Half-Tile Scalar Expansion Performance Repair"
    checker = "check_cuda_rust_down_sequential_half_tile_scalar_expansion_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 276, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-implementation review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost scalar half", lambda value: value["implementation"].update({"scalar_half_tile_entries": 16})),
        ("lost spill removal", lambda value: value["implementation"].update({"repaired_down_ld_local_sites": 9})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_confirmation_profile"].update({"down_ms": 632.769})),
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
