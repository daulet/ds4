#!/usr/bin/env python3
"""Validate the Rust CUDA cached-gate row-span default performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-gate-rowspan-default-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA gate row-span default repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_gate_rowspan_default_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-opt-in-rust-gate-rowspan-default-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_opt_in_rust_gate_rowspan_default", True),
        ("selects_row512_without_override", True),
        ("retains_explicit_row512_control", True),
        ("retains_explicit_row2048_control", True),
        ("changes_cuda_kernel_body", False),
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
    span_block = texts["abi"].split("let gate_row_span = if", 1)[1].split("let down_row_span", 1)[0]
    report.check('"DS4_CUDA_MOE_GATE_ROW512"' in span_block, "explicit row-512 control missing")
    report.check('"DS4_CUDA_MOE_GATE_ROW2048"' in span_block, "explicit row-2048 control missing")
    report.check("1024" not in span_block, "parent no-override row span retained")
    report.check(
        "} else {\n                                                512\n                                            };" in span_block,
        "repaired no-override row span missing",
    )
    report.check(
        "macro_rules! abi_moe_down_accumulate_q2_group" in texts["kernels"],
        "cached-down predecessor expansion missing",
    )
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi.rs"),
        ("policy_expression", "gate_row_span"),
        ("parent_no_override_row_span", 1024),
        ("repaired_no_override_row_span", 512),
        ("cuda_kernel_source_changed", False),
        ("retained_cached_gate_kernel", "abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel"),
        ("retained_cached_down_predecessor", "abi_moe_down_accumulate_q2_group"),
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
        ("parent_profiled_shared_library_sha256", "f0e72d3aa68715171e8caf2673048ad3e16cb432fae3267da4811454bbd1f856"),
        ("repaired_profiled_shared_library_sha256", "f198f32a20679409416b929dedc4fa0a4c10688b6a34788604a390bbcd7e8673"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_default_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_default_profile"), "repaired profile")
    rejected = require_dict(report, execution.get("parent_row2048_rejected_profile"), "row-2048 rejected profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, row_span, gateup, down, total in [
        (parent, "parent", 1024, 1533.245, 905.109, 2454.643),
        (repaired, "repaired", 512, 1482.510, 908.823, 2408.000),
    ]:
        report.check(profile.get("row_span") == row_span, f"{label} row-span drift")
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    overrides = execution.get("parent_row512_override_profiles")
    report.check(isinstance(overrides, list) and len(overrides) == 2, "row-512 override repeats missing")
    if isinstance(overrides, list) and len(overrides) == 2:
        report.check(overrides[0].get("gateup_ms") == 1483.300, "first row-512 repeat drift")
        report.check(overrides[1].get("gateup_ms") == 1482.803, "second row-512 repeat drift")
    report.check(rejected.get("gateup_ms") == 1678.049, "row-2048 rejection gate/up drift")
    report.check(rejected.get("total_ms") == 2600.058, "row-2048 rejection total drift")
    report.check(current_c.get("gateup_ms") == 517.818, "current-C gate/up drift")
    report.check(current_c.get("total_ms") == 924.283, "current-C total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent_default", 1.034),
        ("total_speedup_over_parent_default", 1.019),
        ("gateup_reduction_percent_over_parent_default", 3.31),
        ("total_reduction_percent_over_parent_default", 1.90),
        ("repaired_gateup_ratio_to_current_c", 2.86),
        ("repaired_down_ratio_to_current_c", 2.31),
        ("repaired_total_ratio_to_current_c", 2.61),
        ("remaining_primary_bottleneck", "cached-gateup-and-down-native-codegen-gap"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    probes = require_dict(report, execution.get("rejected_kernel_probes"), "rejected kernel probes")
    gate_probe = require_dict(report, probes.get("paired_gate_ib32_expansion"), "rejected gate probe")
    down_probe = require_dict(report, probes.get("fixed_down_outer_chunk_expansion"), "rejected down probe")
    report.check(gate_probe.get("total_ms") == 2469.199, "rejected gate probe drift")
    report.check(down_probe.get("total_ms") == 2527.024, "rejected down probe drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "f77bce031b0f59593f2baa3cdfc94a5386be0bb88412cb885ef9ab3307d0317a",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Gate Row-Span Default Performance Repair"
    checker = "check_cuda_rust_gate_rowspan_default_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 269, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("policy evidence regression", lambda value: value["implementation"].update({"repaired_no_override_row_span": 1024})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_default_profile"].update({"gateup_ms": 1533.245})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_current_c_route": True})),
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
