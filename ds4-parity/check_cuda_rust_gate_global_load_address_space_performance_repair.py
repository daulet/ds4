#!/usr/bin/env python3
"""Validate the Rust CUDA gate global-load address-space performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-gate-global-load-address-space-performance-repair.json"
CUDA_OXIDE_REVISION = "ae721dc95912a918f182d13b7ca55281aa29d8f9"


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
        "manifest": (ROOT / "rust/ds4-cuda/Cargo.toml").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA gate global-load address-space repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_gate_global_load_address_space_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-gate-global-load-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_gate_global_load_address_space", True),
        ("changes_down_global_load_address_space", False),
        ("retains_private_padded_q8_layout", True),
        ("retains_paired_shared_staging", True),
        ("retains_fixed_order_reduction", True),
        ("retains_gate_dp4a_topology", True),
        ("retains_down_dp4a_topology", True),
        ("changes_default_current_c_route", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("cuda_oxide_revision", CUDA_OXIDE_REVISION),
        ("gate_global_u16_load_sites", 10),
        ("gate_global_u32_load_sites", 3),
        ("parent_gate_global_load_sites", 0),
        ("repaired_gate_global_load_sites", 13),
        ("parent_down_global_load_sites", 0),
        ("repaired_down_global_load_sites", 0),
        ("repaired_gate_st_shared_u64_sites", 2),
        ("repaired_down_st_shared_u64_sites", 1),
        ("repaired_gate_dp4a_sites", 128),
        ("repaired_down_dp4a_sites", 256),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check(texts["manifest"].count(CUDA_OXIDE_REVISION) == 3, "cuda-oxide revision pin drift")
    kernels = texts["kernels"]
    report.check("memory::load_global_u16" in kernels, "gate global u16 intrinsic missing")
    report.check("memory::load_global_u32" in kernels, "gate global u32 intrinsic missing")
    report.check("abi_moe_global_load_u16($values, $offset)" in kernels, "gate weight load not global")
    report.check("abi_moe_global_load_aligned_u32(xq.as_ptr(), input_offset)" in kernels, "gate Q8 tail load not global")
    report.check("macro_rules! abi_moe_down_load_u32" in kernels, "down load scope missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "B300 execution")
    for key, expected in [
        ("date_utc", "2026-06-01"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "5c06a124db2c5a0854ae8aebca63a870ebd07032ea9bb81f7c16e0379b497de6"),
        ("parent_ptx_sha256", "92e32bdacd74a7aac3cbf443bef79d9c54cbb193c68f063b5ef6be0a430572e9"),
        ("repaired_shared_library_sha256", "8a8b4e2a8f6e7a3751b85c47c2fffa5a009a83c01365a6a6bdd5a83375c0f5c3"),
        ("repaired_ptx_sha256", "55c1aea6973a6313299bb5bd4b9b708595a93f138052387766f79baff72c920a"),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_confirmation_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, gateup, down, total in [
        (parent, "parent", 567.730, 421.883, 1005.846),
        (repaired, "repaired", 519.560, 421.707, 957.414),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.093),
        ("down_speedup_over_parent", 1.000),
        ("total_speedup_over_parent", 1.051),
        ("gateup_reduction_percent_over_parent", 8.48),
        ("down_reduction_percent_over_parent", 0.04),
        ("total_reduction_percent_over_parent", 4.82),
        ("repaired_gateup_ratio_to_current_c", 1.004),
        ("repaired_down_ratio_to_current_c", 1.073),
        ("repaired_total_ratio_to_current_c", 1.036),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    rejected = require_dict(report, execution.get("rejected_down_global_load_probe"), "rejected probe")
    report.check(rejected.get("down_ms") == 536.311, "rejected down regression drift")
    report.check(rejected.get("retained") is False, "rejected down probe overclaim")
    feature = require_dict(report, execution.get("feature_test_probe"), "feature test")
    report.check(feature.get("test_count") == 176 and feature.get("passed") is True, "feature test drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "42de5425a5febffe2074e3fa78008e65a11080f979390429bf64ea7200fe0cf0",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Gate Global-Load Address-Space Performance Repair"
    checker = "check_cuda_rust_gate_global_load_address_space_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 284, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review drift")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review drift")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost gate global loads", lambda value: value["implementation"].update({"repaired_gate_global_load_sites": 0})),
        ("globalized down regression", lambda value: value["ownership"].update({"changes_down_global_load_address_space": True})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_confirmation_profile"].update({"total_ms": 1005.846})),
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
