#!/usr/bin/env python3
"""Validate the corrected Rust CUDA cached-gate row-span 256 default repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-gate-rowspan-256-default-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA gate row-span 256 default repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_gate_rowspan_256_default_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-corrected-rust-gate-rowspan-256-default-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_opt_in_rust_gate_rowspan_default", True),
        ("selects_row256_without_override", True),
        ("honors_existing_row256_control", True),
        ("honors_existing_row128_control", True),
        ("retains_explicit_row512_control", True),
        ("retains_explicit_row2048_control", True),
        ("builds_on_corrected_down_arithmetic", True),
        ("changes_cuda_kernel_body", False),
        ("changes_default_current_c_route", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts["abi"])
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], abi: str) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    span_block = abi.split("let gate_row_span = if", 1)[1].split("let down_row_span", 1)[0]
    for control in [
        "DS4_CUDA_MOE_GATE_ROW512",
        "DS4_CUDA_MOE_GATE_ROW2048",
        "DS4_CUDA_MOE_GATE_ROW256",
        "DS4_CUDA_MOE_GATE_ROW128",
    ]:
        report.check(f'"{control}"' in span_block, f"explicit control missing: {control}")
    report.check(
        "} else {\n                                                256\n                                            };" in span_block,
        "repaired no-override row span missing",
    )
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi.rs"),
        ("policy_expression", "gate_row_span"),
        ("corrected_parent_no_override_row_span", 512),
        ("repaired_no_override_row_span", 256),
        ("cuda_kernel_source_changed", False),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("candidate_benchmark_binary_sha256", "908f184bf5175d6dc84105a571d6fc7f33c109006c6108e8f3625555186cd9ae"),
        ("corrected_parent_shared_library_sha256", "b47c4c17773b279264ef92fccf70b2e719cd331834fe7fdd63b68ab081f33709"),
        ("repaired_shared_library_sha256", "0f835118466070058b1aaf1488262ba8121af707e272ac61ecbc5c8adffa509f"),
        ("parent_and_repaired_ptx_sha256", "3da678adc63c955636ab363b489f8c3f43d15601f73ca661f4ae779bc8517b8e"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("corrected_parent_default_profile"), "corrected parent profile")
    retained = require_dict(report, execution.get("repaired_default_profile"), "repaired profile")
    control = require_dict(report, execution.get("repaired_explicit_row512_profile"), "same-DSO row-512 control")
    report.check(parent.get("row_span") == 512 and parent.get("gateup_ms") == 1463.339, "corrected parent drift")
    for profile, label, row_span, gateup, total in [
        (retained, "repaired", 256, 1447.708, 2324.995),
        (control, "same-DSO control", 512, 1462.508, 2339.976),
    ]:
        report.check(profile.get("row_span") == row_span, f"{label} row-span drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    probes = execution.get("measurement_probe_profiles")
    report.check(isinstance(probes, list) and len(probes) == 4, "measurement probes missing")
    if isinstance(probes, list) and len(probes) == 4:
        report.check(probes[0].get("gateup_ms") == 1462.051, "probe control drift")
        report.check(probes[1].get("gateup_ms") == 1448.505, "first row-256 drift")
        report.check(probes[2].get("gateup_ms") == 1449.764, "row-128 drift")
        report.check(probes[3].get("gateup_ms") == 1448.424, "second row-256 drift")
    attribution = require_dict(report, execution.get("same_dso_attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_row512_control", 1.010),
        ("total_speedup_over_row512_control", 1.006),
        ("gateup_reduction_percent_over_row512_control", 1.01),
        ("total_reduction_percent_over_row512_control", 0.64),
        ("repaired_gateup_ratio_to_current_c", 2.80),
        ("repaired_down_ratio_to_current_c", 2.19),
        ("repaired_total_ratio_to_current_c", 2.52),
        ("remaining_primary_bottleneck", "cached-gateup-native-instruction-gap"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "21de915bd04acf4a280f7ae65b5bee8ea613845b45e04e3ca4914e600fa0b682",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Gate Row-Span 256 Default Performance Repair"
    checker = "check_cuda_rust_gate_rowspan_256_default_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 273, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("measurement_pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "measurement review missing")
    report.check(review.get("retention_pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "retention review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("policy evidence regression", lambda value: value["implementation"].update({"repaired_no_override_row_span": 512})),
        ("lost same-DSO speedup", lambda value: value["b300_execution"]["repaired_default_profile"].update({"gateup_ms": 1462.508})),
        ("corrected parent hidden", lambda value: value["ownership"].update({"builds_on_corrected_down_arithmetic": False})),
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
