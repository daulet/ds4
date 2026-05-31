#!/usr/bin/env python3
"""Validate the Rust CUDA reciprocal-square-root route correctness repair leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Long-Prefill Performance And C CUDA Removal Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-rsqrt-route-correctness-repair.json"


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
    print(f"{MILESTONE} Rust CUDA rsqrt route correctness repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_rsqrt_route_correctness_repair.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-short-correctness-full-route-performance-blocked",
        "status drift",
    )
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_rust_cuda_numeric_primitive", True),
        ("changes_public_abi_surface", False),
        ("default_current_c_route_preserved", True),
        ("short_official_vector_correctness_closed", True),
        ("full_official_vector_gate_closed", False),
        ("runtime_route_promoted", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("source") == "rust/ds4-cuda/src/abi_kernels.rs", "source drift")
    report.check(implementation.get("current_c_primitive") == "rsqrtf", "current-C primitive drift")
    report.check(implementation.get("rust_device_primitive") == "__nv_rsqrtf", "device primitive drift")
    kernels = texts["kernels"]
    for marker in [
        'fn __nv_rsqrtf(value: f32) -> f32;',
        '#[unsafe(export_name = "__nv_rsqrtf")]',
        "host_libdevice_rsqrtf_stub",
        "let norm_scale = unsafe { __nv_rsqrtf(mean_square) };",
        "let scale = unsafe { __nv_rsqrtf(mean_square) };",
        "let scale = unsafe { __nv_rsqrtf(head_dim as f32) };",
    ]:
        report.check(marker in kernels, f"rsqrt implementation marker missing: {marker}")
    report.check(kernels.count("__nv_rsqrtf(") == 10, "expected nine device rsqrt call sites plus declaration")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("shared_library_sha256", "c64dda39104483ee28a25babef8c140d8e59a83d1a2e889fd6e4c44eeeda9ddd"),
        ("device_libdevice_reference_present", True),
        ("host_libdevice_symbol_resolved", True),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    reproducer = require_dict(report, execution.get("two_step_reproducer"), "two_step_reproducer")
    report.check(reproducer.get("case") == "short_code_completion", "reproducer case drift")
    report.check(reproducer.get("wall_elapsed_seconds") == 54, "reproducer runtime drift")
    steps = reproducer.get("selected_steps", [])
    report.check(len(steps) == 2, "reproducer steps missing")
    if len(steps) == 2:
        report.check(steps[1].get("expected_selected_hex") == "63", "reproducer oracle drift")
        report.check(steps[1].get("selected_hex") == "63", "reproducer output drift")
        report.check(steps[1].get("selected_matches_expected") is True, "reproducer correctness lost")
    short = require_dict(report, execution.get("short_vector_probe"), "short_vector_probe")
    for key, expected in [
        ("case_count", 3),
        ("step_count", 9),
        ("selected_match_count", 9),
        ("all_selected_match", True),
    ]:
        report.check(short.get(key) == expected, f"short-vector evidence drift: {key}")
    full = require_dict(report, execution.get("full_official_vector_probe"), "full_official_vector_probe")
    report.check(full.get("completed") is False, "full gate overclaim")
    report.check(full.get("observation_seconds_at_least", 0) >= 900, "full gate duration under threshold")
    report.check(full.get("gpu_utilization_percent_while_observed") == 100, "full gate utilization drift")
    report.check(full.get("terminated_after_bound") is True, "full gate termination missing")
    report.check(full.get("gpu_memory_mib_after_termination") == 0, "B300 cleanup evidence drift")
    smokes = require_dict(report, execution.get("affected_public_abi_smokes"), "affected_public_abi_smokes")
    for key in [
        "plain_rms_norm_passed",
        "weighted_rms_norm_passed",
        "head_rms_norm_passed",
        "fused_qkv_rms_rows_passed",
        "attention_decode_heads_passed",
    ]:
        report.check(smokes.get(key) is True, f"affected ABI smoke drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Reciprocal-Square-Root Route Correctness Repair"
    checker = "check_cuda_rust_rsqrt_route_correctness_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 feature test count drift")
    report.check(validation.get("unified_report_passed") == 260, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified fail count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("short correctness hidden", lambda value: value["b300_execution"]["short_vector_probe"].update({"selected_match_count": 8})),
        ("full gate overclaim", lambda value: value["ownership"].update({"full_official_vector_gate_closed": True})),
        ("route promotion", lambda value: value["ownership"].update({"runtime_route_promoted": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
