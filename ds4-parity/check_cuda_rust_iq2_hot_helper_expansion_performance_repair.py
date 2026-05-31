#!/usr/bin/env python3
"""Validate the Rust CUDA IQ2 hot-helper expansion performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-iq2-hot-helper-expansion-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA IQ2 hot-helper expansion repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_iq2_hot_helper_expansion_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-iq2-hot-helper-expanded-route-blocked", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("expands_iq2_signed_word_at_hot_sites", True),
        ("retains_branchless_iq2_transform", True),
        ("retains_multi_pair_gate_dp4a", True),
        ("retains_multi_pair_down_dp4a", True),
        ("changes_default_dispatch", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
        ("default_current_c_route_preserved", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    kernels = texts["kernels"]
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("macro_rules! abi_moe_iq2_signed_word" in kernels, "IQ2 expansion macro missing")
    report.check("fn abi_moe_iq2_signed_word(" not in kernels, "hot IQ2 device helper retained")
    report.check(kernels.count("abi_moe_iq2_signed_word!(") in (4, 16), "IQ2 hot expansion count drift")
    report.check(
        "0_u32.wrapping_sub" in kernels or "integer::prmt_b32_ba98(sign_bits)" in kernels,
        "branchless byte-mask construction or PRMT successor missing",
    )
    report.check(".wrapping_add(mask & 0x0101_0101)" in kernels, "packed transform missing")
    report.check(implementation.get("hot_invocation_count") == 4, "fixture invocation count drift")
    report.check(
        implementation.get("retained_transform") == "(values ^ mask).wrapping_add(mask & 0x0101_0101)",
        "fixture transform drift",
    )


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_profiled_shared_library_sha256", "383fe12843109a33719bcbac5e38ec1b22ea95f9e45647b2ecc47c3f88a79f01"),
        ("repaired_profiled_shared_library_sha256", "506b1a9daaa326a650b95e1450683bf2fd677ac8439a3f6d99068a35adfcc13b"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    ptx = require_dict(report, execution.get("ptx_call_boundary"), "ptx call boundary")
    for key, expected in [
        ("parent_hot_helper_reference_count", 4),
        ("repaired_hot_helper_reference_count", 0),
        ("repaired_remaining_call_uni_count", 3),
    ]:
        report.check(ptx.get(key) == expected, f"PTX evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_rebuild_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_rebuild_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, gateup, down, total in [
        (parent, "parent", 2691.585, 1127.143, 3835.101),
        (repaired, "repaired", 2205.067, 1128.197, 3349.623),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("profiled_layer_count", 43) == 43, f"{label} layer-count drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.22),
        ("total_speedup_over_parent", 1.14),
        ("prefill_gain_percent_over_parent", 3.81),
        ("repaired_gateup_ratio_to_current_c", 4.26),
        ("repaired_down_ratio_to_current_c", 2.87),
        ("repaired_total_ratio_to_current_c", 3.62),
        ("remaining_primary_bottleneck", "cached-gateup-iq2-arithmetic-and-intrinsic-gap"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "3a55e7fd5e725f80c5e3abb806dd6d88a4f3fc4de479068acca5f4a349ce35f6",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA IQ2 Hot-Helper Expansion Performance Repair"
    checker = "check_cuda_rust_iq2_hot_helper_expansion_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 270, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("hot call retained", lambda value: value["b300_execution"]["ptx_call_boundary"].update({"repaired_hot_helper_reference_count": 1})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_rebuild_profile"].update({"gateup_ms": 2691.585})),
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
