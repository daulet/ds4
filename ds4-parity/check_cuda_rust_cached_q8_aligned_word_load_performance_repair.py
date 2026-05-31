#!/usr/bin/env python3
"""Validate the Rust CUDA cached Q8 aligned-word load performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-cached-q8-aligned-word-load-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA cached Q8 aligned-word load repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_cached_q8_aligned_word_load_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-cached-q8-aligned-word-load-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_q8_load_contract", True),
        ("aligns_cached_gate_staging", True),
        ("aligns_cached_down_staging", True),
        ("uses_aligned_q8_scale_and_word_loads", True),
        ("retains_q8_bsum_byte_loads", True),
        ("retains_dp4a_topology", True),
        ("retains_corrected_down_arithmetic", True),
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
        ("q8_block_bytes", 292),
        ("shared_alignment_bytes", 4),
        ("aligned_load_helper", "abi_moe_cached_load_aligned_u32"),
        ("parent_gate_dp4a_sites", 128),
        ("repaired_gate_dp4a_sites", 128),
        ("parent_gate_ld_shared_u8_sites", 584),
        ("repaired_gate_ld_shared_u8_sites", 8),
        ("repaired_gate_ld_shared_u32_sites", 128),
        ("parent_down_dp4a_sites", 32),
        ("repaired_down_dp4a_sites", 32),
        ("parent_down_ld_shared_u8_sites", 134),
        ("repaired_down_ld_shared_u8_sites", 2),
        ("repaired_down_ld_shared_u32_sites", 32),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check("static mut SXQ: SharedArray<" in gate and "\n            4,\n" in gate, "gate staging alignment missing")
    report.check("static mut SMIDQ: SharedArray<" in down and "\n            4,\n" in down, "down staging alignment missing")
    report.check("fn abi_moe_cached_load_aligned_u32" in kernels, "aligned Q8 load helper missing")
    report.check("*values.add(offset).cast::<u32>()" in kernels, "aligned Q8 word read missing")
    report.check("abi_moe_cached_load_u32(" not in kernels, "byte-built Q8 u32 helper retained")
    report.check(
        kernels.count("abi_moe_cached_load_aligned_u32(") == 3,
        "aligned helper must be limited to definition, scale, and word reads",
    )
    report.check("abi_moe_cached_q8_bsum" in down and "abi_moe_cached_load_u16(" in kernels, "Q8 bsum contract missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "cc4e745dfaffbcb1672118bc33b2c8795c438adba4d9bb8f3f220ed01ea3368c"),
        ("parent_ptx_sha256", "2b35a748f0653cc39145b418fe9087d4cd37e4750a596f95e27d2a7ea29871bf"),
        ("repaired_shared_library_sha256", "ccc6dc7fe28acb3edd9d11c22198f1bd2eed7a451e1a7bbc67b1d3c4a3d3dfbc"),
        ("repaired_ptx_sha256", "e97c1c1f18d3329625fe8ea737aea11873db6335609d1a569d1693b832addb6b"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_profile"), "repaired profile")
    for profile, label, gateup, down, total in [
        (parent, "parent", 1255.901, 860.766, 2132.947),
        (repaired, "repaired", 752.829, 632.945, 1402.260),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("adjacent_attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.668),
        ("down_speedup_over_parent", 1.360),
        ("total_speedup_over_parent", 1.521),
        ("gateup_reduction_percent_over_parent", 40.06),
        ("down_reduction_percent_over_parent", 26.47),
        ("total_reduction_percent_over_parent", 34.26),
        ("prefill_increase_percent_over_parent", 3.23),
        ("repaired_gateup_ratio_to_current_c", 1.45),
        ("repaired_down_ratio_to_current_c", 1.61),
        ("repaired_total_ratio_to_current_c", 1.52),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    rejected = require_dict(report, execution.get("rejected_fast_swiglu_probe"), "rejected fast SwiGLU probe")
    report.check(rejected.get("gateup_ms") == 1256.183, "rejected fast SwiGLU gate/up drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "9533e25ae0154cc27e693cb246115482794a6f037a40daf3779eaa97c07b634d",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Cached Q8 Aligned-Word Load Performance Repair"
    checker = "check_cuda_rust_cached_q8_aligned_word_load_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 275, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-implementation review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost alignment", lambda value: value["implementation"].update({"shared_alignment_bytes": 1})),
        ("lost aligned load codegen", lambda value: value["implementation"].update({"repaired_gate_ld_shared_u8_sites": 584})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_profile"].update({"total_ms": 2132.947})),
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
