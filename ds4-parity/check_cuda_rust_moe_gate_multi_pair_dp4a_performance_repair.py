#!/usr/bin/env python3
"""Validate the Rust CUDA cached routed-MoE gate multi-pair DP4A repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-moe-gate-multi-pair-dp4a-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA MoE gate multi-pair DP4A repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_moe_gate_multi_pair_dp4a_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-gateup-accelerated-down-open", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_gate_kernel_math", True),
        ("adds_multi_pair_iq2_dp4a", True),
        ("reuses_weight_decode_across_pairs", True),
        ("changes_cached_down_kernel_math", False),
        ("changes_default_dispatch", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
        ("default_current_c_route_preserved", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, texts["kernels"], texts["report"])
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, kernels: str, report_text: str) -> None:
    gate = kernels.split("pub fn abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel(", 1)[1].split(
        "pub fn abi_moe_atomic_output_zero_kernel(", 1
    )[0]
    gate_accumulators_present = "let mut gate = [0.0_f32; 8];" in gate or all(
        f"let mut gate{entry} = 0.0_f32;" in gate for entry in range(8)
    )
    up_accumulators_present = "let mut up = [0.0_f32; 8];" in gate or all(
        f"let mut up{entry} = 0.0_f32;" in gate for entry in range(8)
    )
    report.check(gate_accumulators_present, "gate multi-pair accumulators missing")
    report.check(up_accumulators_present, "up multi-pair accumulators missing")
    report.check(gate.count("integer::dp4a_i8(") in (4, 16), "gate/up DP4A call layout drift")
    report.check(
        "fn abi_moe_iq2_signed_word" in kernels or "macro_rules! abi_moe_iq2_signed_word" in kernels,
        "IQ2 packed sign computation missing",
    )
    paired_successor = "check_cuda_rust_cached_q8_aligned_pair_load_performance_repair.py" in report_text
    if paired_successor:
        report.check("fn abi_moe_cached_q8_pair" in kernels, "registered cached q8 pair successor missing")
        report.check("fn abi_moe_cached_q8_word" not in kernels, "retired cached q8 word helper retained")
    else:
        report.check("fn abi_moe_cached_q8_word" in kernels, "cached q8 word helper missing")
    report.check("fn abi_moe_iq2_q8_k_cached_dot(" not in kernels, "scalar cached IQ2 helper retained")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("repaired_profiled_shared_library_sha256", "926e67cb31746f434b994c020dfeaab9e1d124194d80bf371b599df832aaf032"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_rust_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_rust_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for profile, label, gateup, down, total in [
        (parent, "parent", 8217.639, 3998.630, 12232.557),
        (repaired, "repaired", 2792.292, 3993.670, 6802.279),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    for key, expected in [
        ("gateup_speedup_over_parent", 2.94),
        ("total_speedup_over_parent", 1.80),
        ("prefill_gain_percent_over_parent", 57.9),
        ("repaired_down_ratio_to_current_c", 10.16),
        ("remaining_primary_bottleneck", "cached-down-q2-multi-pair-dot"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    rejected = require_dict(report, execution.get("rejected_probes"), "rejected probes")
    report.check(rejected.get("per_pair_dp4a_total_ms") == 13489.699, "rejected per-pair probe drift")
    report.check("sm_100a is not a recognized" in rejected.get("sm_100a_target_build", ""), "target blocker missing")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Multi-Pair Gate DP4A Performance Repair"
    checker = "check_cuda_rust_moe_gate_multi_pair_dp4a_performance_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README wiring missing")
    report.check(checker in texts["report"], "report wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    decision = require_dict(report, fixture.get("decision"), "decision")
    report.check(decision.get("default_route") == "retain-current-c", "default route drift")
    report.check(decision.get("rust_cuda_dso_promotion") == "blocked", "promotion drift")
    report.check("cached down Q2" in decision.get("next_scoped_repair", ""), "next repair missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test-count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 test-count drift")
    report.check(validation.get("unified_report_passed") == 267, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost speedup", lambda value: value["b300_execution"]["repaired_rust_profile"].update({"gateup_ms": 8217.639})),
        ("down overclaim", lambda value: value["ownership"].update({"changes_cached_down_kernel_math": True})),
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
