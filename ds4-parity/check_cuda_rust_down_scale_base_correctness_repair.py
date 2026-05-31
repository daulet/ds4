#!/usr/bin/env python3
"""Validate the Rust CUDA cached-down Q2 scale-base correctness repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-down-scale-base-correctness-repair.json"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE.lower()}/abi_routed_moe_cached_down_scale_base_link_smoke.c"


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
        "harness": HARNESS.read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA down scale-base correctness repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_down_scale_base_correctness_repair.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-cached-down-scale-base-corrected-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_down_scale_addressing", True),
        ("adds_public_cached_down_regression_harness", True),
        ("preserves_cached_down_dp4a_topology", True),
        ("preserves_gate_kernel", True),
        ("preserves_dispatch_policy", True),
        ("marks_prior_cached_down_timings_as_pre_correction_provenance", True),
        ("official_vector_gate_preserved", True),
        ("changes_default_current_c_route", False),
        ("runtime_route_promoted", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    down = texts["kernels"].split(
        "pub fn abi_moe_down_expert_tile16_rowspan_cached_kernel(", 1
    )[1].split("pub fn abi_moe_gate_up_mid_f32_kernel(", 1)[0]
    report.check("let mut scale_index = packed;" in down, "packed-relative low-scale cursor missing")
    report.check("let mut scale_index = 0_usize;" not in down, "base-zero low-scale cursor retained")
    report.check("down_weights[packed + scale]" in down, "packed-relative minimum path missing")
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("kernel", "abi_moe_down_expert_tile16_rowspan_cached_kernel"),
        ("previous_scale_cursor", "0_usize"),
        ("repaired_scale_cursor", "packed"),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    for marker in [
        "#define N_TOKENS 128u",
        "#define OUT_DIM 2u",
        "block[index] = (uint8_t)(row + 1u)",
        "clear_route_env()",
        "ds4_gpu_routed_moe_batch_tensor(",
        "fabsf(row1 - 2.0f * row0)",
        "active_row_scale_ratio_matches",
    ]:
        report.check(marker in texts["harness"], f"public-route harness marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("buggy_parent_shared_library_sha256", "109503f9476b2fa29e1753cf3d207167c12a9c59a28077b1d40808b696def47a"),
        ("repaired_shared_library_sha256", "b47c4c17773b279264ef92fccf70b2e719cd331834fe7fdd63b68ab081f33709"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("buggy_parent_profile"), "buggy parent profile")
    repaired = require_dict(report, execution.get("repaired_profile"), "repaired profile")
    report.check(parent.get("down_ms") == 838.492, "buggy-parent down provenance drift")
    report.check(parent.get("total_ms") == 2316.378, "buggy-parent total provenance drift")
    report.check(repaired.get("down_ms") == 860.566, "repaired down profile drift")
    report.check(repaired.get("total_ms") == 2340.580, "repaired total profile drift")
    effect = require_dict(report, execution.get("correction_effect"), "correction effect")
    report.check(effect.get("down_ms_increase_from_buggy_parent") == 22.074, "corrected down delta drift")
    report.check(effect.get("total_ms_increase_from_buggy_parent") == 24.202, "corrected total delta drift")
    public = require_dict(report, execution.get("public_route_regression_probe"), "public regression probe")
    for key, expected in [
        ("staticlib_sha256", "c4c69a70dd221988db94fb4dc478656b990c5bca12e374268ff7e351bc4a7ed9"),
        ("executable_sha256", "84fbb4a66a65fe8b6d6930b45cf415147450071984b40cd6e31605de0edb3bcf"),
        ("stdout_sha256", "3d2bcbf35ae41b2697516b37cdb8e2001e1c6550bcf3e9002f21ef9a4b85aa1a"),
        ("public_entry", "ds4_gpu_routed_moe_batch_tensor"),
        ("n_tokens", 128),
        ("passed", True),
    ]:
        report.check(public.get(key) == expected, f"public regression evidence drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "d815d16c7aec7721c3c52c27d26899363d97ca5b3112d4765ef2a6f2eb33a3a2",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Down Scale-Base Correctness Repair"
    checker = "check_cuda_rust_down_scale_base_correctness_repair.py"
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
    report.check(validation.get("public_route_regression_probe_passed") is True, "public regression pass missing")
    report.check(validation.get("unified_report_passed") == 272, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("repair hidden", lambda value: value["implementation"].update({"repaired_scale_cursor": "0_usize"})),
        ("public witness hidden", lambda value: value["b300_execution"]["public_route_regression_probe"].update({"passed": False})),
        ("prior timing overclaim", lambda value: value["ownership"].update({"marks_prior_cached_down_timings_as_pre_correction_provenance": False})),
        ("route overclaim", lambda value: value["decision"].update({"rust_cuda_dso_promotion": "passed"})),
        ("official mismatch", lambda value: value["b300_execution"]["official_vector_probe"].update({"passed": False})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
