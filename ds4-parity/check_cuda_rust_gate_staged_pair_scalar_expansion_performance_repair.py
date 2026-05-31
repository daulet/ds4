#!/usr/bin/env python3
"""Validate the Rust CUDA cached-gate staged-pair scalar expansion repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-gate-staged-pair-scalar-expansion-performance-repair.json"


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
    print(f"{MILESTONE} Rust CUDA gate staged-pair scalar expansion repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_gate_staged_pair_scalar_expansion_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-gate-staged-pair-scalar-expansion-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_cached_gate_kernel_body", True),
        ("expands_staged_pair_entry_dimension", True),
        ("uses_named_scalar_accumulators", True),
        ("retains_iq2_weight_arithmetic", True),
        ("retains_cached_down_kernel", True),
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
    for key, expected in [
        ("source", "rust/ds4-cuda/src/abi_kernels.rs"),
        ("kernel", "abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel"),
        ("parent_dp4a_sites", 16),
        ("repaired_dp4a_sites", 128),
        ("parent_local_store_sites", 36),
        ("parent_local_load_sites", 8),
        ("repaired_local_store_sites", 0),
        ("repaired_local_load_sites", 0),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check("let mut gate = [0.0_f32; 8];" not in gate, "gate array retained in repaired kernel")
    report.check("let mut up = [0.0_f32; 8];" not in gate, "up array retained in repaired kernel")
    report.check("let mut block_sums = [0_i32; 8];" not in gate, "block-sum array retained in repaired kernel")
    for entry in range(8):
        report.check(f"let mut gate{entry} = 0.0_f32;" in gate, f"gate scalar missing: {entry}")
        report.check(f"let mut up{entry} = 0.0_f32;" in gate, f"up scalar missing: {entry}")
        report.check(
            gate.count(f"accumulate_entry!({entry}, block_sum{entry});") == 2,
            f"staged-pair expansion missing: {entry}",
        )
        report.check(f"emit_entry!({entry}, gate{entry}, up{entry});" in gate, f"output scalar missing: {entry}")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "0f835118466070058b1aaf1488262ba8121af707e272ac61ecbc5c8adffa509f"),
        ("parent_ptx_sha256", "3da678adc63c955636ab363b489f8c3f43d15601f73ca661f4ae779bc8517b8e"),
        ("repaired_shared_library_sha256", "cc4e745dfaffbcb1672118bc33b2c8795c438adba4d9bb8f3f220ed01ea3368c"),
        ("repaired_ptx_sha256", "2b35a748f0653cc39145b418fe9087d4cd37e4750a596f95e27d2a7ea29871bf"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_resident_repeat_profile"), "repaired profile")
    for profile, label, gateup, total in [
        (parent, "parent", 1448.003, 2325.307),
        (repaired, "repaired", 1256.146, 2133.521),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("adjacent_attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.153),
        ("total_speedup_over_parent", 1.090),
        ("gateup_reduction_percent_over_parent", 13.25),
        ("total_reduction_percent_over_parent", 8.25),
        ("prefill_increase_percent_over_parent", 3.19),
        ("repaired_total_ratio_to_current_c", 2.31),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "b2aaa8af7f7d9e854179a85f72bb2fd774207ecb0bf8c76f5e40353ce837b331",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Gate Staged-Pair Scalar Expansion Performance Repair"
    checker = "check_cuda_rust_gate_staged_pair_scalar_expansion_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 274, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-implementation review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost DP4A exposure", lambda value: value["implementation"].update({"repaired_dp4a_sites": 16})),
        ("lost local-memory reduction", lambda value: value["implementation"].update({"repaired_local_store_sites": 36})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_resident_repeat_profile"].update({"gateup_ms": 1448.003})),
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
