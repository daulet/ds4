#!/usr/bin/env python3
"""Validate the opt-in Rust CUDA routed-MoE phase-profile parity leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-moe-phase-profile-parity.json"


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
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
        "substrate": (ROOT / "rust/ds4-cuda/src/substrate.rs").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA MoE phase-profile parity: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_moe_phase_profile_parity.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-gate-down-bottleneck-isolated", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("adds_opt_in_rust_moe_profile", True),
        ("matches_current_c_profile_schema", True),
        ("changes_kernel_math", False),
        ("changes_default_dispatch", False),
        ("profile_requires_environment_flag", True),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
        ("default_current_c_route_preserved", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, texts: dict[str, str]) -> None:
    abi = texts["abi"]
    substrate = texts["substrate"]
    batch = abi.split('pub unsafe extern "C" fn ds4_gpu_routed_moe_batch_tensor(', 1)[1]
    one = abi.split('pub unsafe extern "C" fn ds4_gpu_routed_moe_one_tensor(', 1)[1].split(
        'pub unsafe extern "C" fn ds4_gpu_routed_moe_batch_tensor(', 1
    )[0]
    report.check("struct AbiRoutedMoeProfile" in abi, "profile type missing")
    report.check('std::env::var_os("DS4_CUDA_MOE_PROFILE").is_none()' in abi, "opt-in flag missing")
    report.check("let mut profile = AbiRoutedMoeProfile::new(backend);" in batch, "batch profiler missing")
    report.check("let mut profile = AbiRoutedMoeProfile::new(backend);" not in one, "single-token route changed")
    for index in range(7):
        report.check(f"profile.record(backend, {index});" in abi, f"profile boundary {index} missing")
    report.check("profile.report(n_tokens, pair_count);" in batch, "profile report missing")
    report.check("CUDA MoE profile tokens={n_tokens} pairs={pair_count}" in abi, "profile schema missing")
    report.check("pub fn new_timing_event" in substrate, "timing event constructor missing")
    report.check("CUevent_flags_enum_CU_EVENT_DEFAULT" in substrate, "timing flag missing")
    report.check("pub fn record_event" in substrate, "existing non-timing event surface missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("profiled_shared_library_sha256", "fd29d851b9d35a599f035913e6f8cb985641e90636537b714075675f9d98bd68"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    rust = require_dict(report, execution.get("rust_profile"), "rust_profile")
    current_c = require_dict(report, execution.get("current_c_profile"), "current_c_profile")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for profile, label, total, gateup, down in [
        (rust, "rust", 12232.557, 8217.639, 3998.630),
        (current_c, "current_c", 924.283, 517.818, 393.200),
    ]:
        report.check(profile.get("profiled_layer_count") == 43, f"{label} layer-count drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
    for key, expected in [
        ("rust_to_current_c_total_ratio", 13.23),
        ("rust_to_current_c_gateup_ratio", 15.87),
        ("rust_to_current_c_down_ratio", 10.17),
        ("rust_gateup_share_percent", 67.2),
        ("rust_down_share_percent", 32.7),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official_vector_probe")
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")
    decision = require_dict(report, fixture.get("decision"), "decision")
    report.check(decision.get("default_route") == "retain-current-c", "default route drift")
    report.check(decision.get("rust_cuda_dso_promotion") == "blocked", "promotion drift")
    report.check("packed multi-pair DP4A" in decision.get("next_scoped_repair", ""), "next repair missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA MoE Phase-Profile Parity"
    checker = "check_cuda_rust_moe_phase_profile_parity.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README wiring missing")
    report.check(checker in texts["report"], "report wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 test count drift")
    report.check(validation.get("unified_report_passed") == 266, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("hidden gate bottleneck", lambda value: value["b300_execution"]["rust_profile"].update({"gateup_ms": 517.818})),
        ("promotion overclaim", lambda value: value["decision"].update({"rust_cuda_dso_promotion": "passed"})),
        ("official correctness overclaim", lambda value: value["b300_execution"]["official_vector_probe"].update({"passed": False})),
        ("next-stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
