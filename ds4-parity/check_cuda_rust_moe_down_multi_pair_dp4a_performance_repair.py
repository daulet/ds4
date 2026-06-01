#!/usr/bin/env python3
"""Validate the Rust CUDA cached routed-MoE down multi-pair DP4A repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-moe-down-multi-pair-dp4a-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA MoE down multi-pair DP4A repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_moe_down_multi_pair_dp4a_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-down-accelerated-gateup-open", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_down_kernel_math", True),
        ("adds_multi_pair_q2_dp4a", True),
        ("reuses_weight_decode_across_pairs", True),
        ("inherits_cached_gate_repair", True),
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
    q2_group = ""
    if "macro_rules! abi_moe_down_accumulate_q2_group" in kernels:
        q2_group = kernels.split("macro_rules! abi_moe_down_accumulate_q2_group", 1)[1].split(
            "#[allow(clippy::too_many_arguments, static_mut_refs)]", 1
        )[0]
    down = kernels.split("pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel(", 1)[1].split(
        "pub fn abi_moe_gate_up_mid_f32_kernel(", 1
    )[0]
    report.check(gate.count("integer::dp4a_i8(") in (4, 16), "retained gate/up DP4A layout drift")
    scalar_half_layout = (
        "let mut entry_base = 0_u32;" in down
        and down.count("let mut accumulator") == 8
        and down.count("let mut min_sum") == 8
        and down.count("let mut quant_sum") == 8
    )
    report.check(
        "let mut accumulators = [0.0_f32; 16];" in down or scalar_half_layout,
        "down multi-pair accumulators missing",
    )
    report.check(
        "let mut min_sums = [0_i32; 16];" in down or scalar_half_layout,
        "down minimum correction missing",
    )
    report.check(
        "let mut quant_sums = [0_i32; 16];" in down or scalar_half_layout,
        "down quant accumulators missing",
    )
    report.check(
        down.count("integer::dp4a_i8(") + q2_group.count("integer::dp4a_i8(") == 8,
        "down DP4A call layout drift",
    )
    paired_successor = "check_cuda_rust_cached_q8_aligned_pair_load_performance_repair.py" in report_text
    if paired_successor:
        report.check("fn abi_moe_cached_q8_pair" in kernels, "registered cached q8 pair successor missing")
        report.check("fn abi_moe_cached_q8_word" not in kernels, "retired cached q8 word helper retained")
    else:
        report.check("fn abi_moe_cached_q8_word" in kernels, "cached q8 word helper missing")
    report.check("fn abi_moe_q2_q8_k_cached_dot(" not in kernels, "scalar cached Q2 helper retained")
    report.check("fn abi_moe_cached_q8_value(" not in kernels, "scalar cached q8 value helper retained")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("repaired_profiled_shared_library_sha256", "e26a3d3364367a2cc71dc6fb99be048b3a282c727d3321e15f41d14fbb8d6443"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_rust_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_rust_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for profile, label, gateup, down, total in [
        (parent, "parent", 2792.292, 3993.670, 6802.279),
        (repaired, "repaired", 2795.404, 1126.795, 3938.489),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    for key, expected in [
        ("down_speedup_over_parent", 3.54),
        ("total_speedup_over_parent", 1.73),
        ("prefill_gain_percent_over_parent", 10.45),
        ("repaired_gateup_ratio_to_current_c", 5.40),
        ("repaired_down_ratio_to_current_c", 2.87),
        ("repaired_total_ratio_to_current_c", 4.26),
        ("remaining_primary_bottleneck", "cached-gateup-iq2-residual-gap"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(official.get("summary_sha256") == "c45998e97a1cd5fcb13785bf869bafb7346eb2c4667563c8a296a7abe3b9a03f", "official summary hash drift")
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Multi-Pair Down DP4A Performance Repair"
    checker = "check_cuda_rust_moe_down_multi_pair_dp4a_performance_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README wiring missing")
    report.check(checker in texts["report"], "report wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    decision = require_dict(report, fixture.get("decision"), "decision")
    report.check(decision.get("default_route") == "retain-current-c", "default route drift")
    report.check(decision.get("rust_cuda_dso_promotion") == "blocked", "promotion drift")
    report.check("cached gate/up IQ2" in decision.get("next_scoped_repair", ""), "next repair missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test-count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 test-count drift")
    report.check(validation.get("unified_report_passed") == 268, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost speedup", lambda value: value["b300_execution"]["repaired_rust_profile"].update({"down_ms": 3993.670})),
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
