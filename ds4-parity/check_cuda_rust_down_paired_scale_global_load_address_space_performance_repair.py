#!/usr/bin/env python3
"""Validate the Rust CUDA down paired-scale global-load address-space repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-down-paired-scale-global-load-address-space-performance-repair.json"
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
    print(f"{MILESTONE} Rust CUDA down paired-scale global-load repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_down_paired_scale_global_load_address_space_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-down-paired-scale-global-load-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("retains_down_metadata_global_load_address_space", True),
        ("changes_down_paired_scale_global_load_address_space", True),
        ("changes_down_packed_q2_word_address_space", False),
        ("changes_down_q8_staging_address_space", False),
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
        ("q2_block_bytes", 84),
        ("scale_pair_alignment_bytes", 2),
        ("parent_down_global_u16_load_sites", 2),
        ("repaired_down_global_u16_load_sites", 6),
        ("repaired_down_global_u32_load_sites", 0),
        ("parent_down_byte_load_sites", 41),
        ("repaired_down_byte_load_sites", 33),
        ("parent_down_shared_load_sites", 152),
        ("repaired_down_shared_load_sites", 152),
        ("parent_down_shared_store_sites", 2),
        ("repaired_down_shared_store_sites", 2),
        ("parent_down_dp4a_sites", 256),
        ("repaired_down_dp4a_sites", 256),
        ("parent_down_local_sites", 0),
        ("repaired_down_local_sites", 0),
        ("parent_down_b32_register_bound", 1540),
        ("repaired_down_b32_register_bound", 1536),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check(texts["manifest"].count(CUDA_OXIDE_REVISION) == 3, "cuda-oxide revision pin drift")
    kernels = texts["kernels"]
    report.check(
        "let packed_scales = abi_moe_global_load_u16($weights, $scale_index);" in kernels,
        "paired scale load missing",
    )
    report.check(
        "let first_scale = (packed_scales & 0x000f) as i32;" in kernels,
        "first paired scale decode missing",
    )
    report.check(
        "let second_scale = ((packed_scales >> 8) & 0x000f) as i32;" in kernels,
        "second paired scale decode missing",
    )
    report.check("$scale_index += 2;" in kernels, "paired scale advancement missing")
    report.check(
        kernels.count("abi_moe_global_load_u16(down_weights, packed +") == 2,
        "metadata global-load predecessor lost",
    )
    report.check(
        "abi_moe_global_load_aligned_u32($values.as_ptr(), $offset)" not in kernels,
        "packed Q2 u32 regression retained",
    )


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "B300 execution")
    for key, expected in [
        ("date_utc", "2026-06-01"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "39aa45d4a45cf81867c9b25d74afc30a5ad25f20552ec006bca8da9951271820"),
        ("repaired_shared_library_sha256", "b1ecf8387e4861481230900d67e5d8a2eef62258c151d4b36304e7b0e5a95e22"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    for key, gateup, down, total in [
        ("initial_parent_profile", 521.993, 420.440, 958.863),
        ("initial_repaired_profile", 521.843, 401.116, 939.293),
        ("confirmation_parent_profile", 521.754, 420.188, 958.364),
        ("confirmation_repaired_profile", 521.879, 401.120, 939.422),
    ]:
        profile = require_dict(report, execution.get(key), key)
        report.check(profile.get("profiled_layer_count") == 43, f"{key} layer-count drift")
        report.check(profile.get("gateup_ms") == gateup, f"{key} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{key} down drift")
        report.check(profile.get("total_ms") == total, f"{key} total drift")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    report.check(current_c.get("down_ms") == 393.200, "current-C down drift")
    report.check(current_c.get("total_ms") == 924.283, "current-C total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("initial_down_reduction_percent_over_parent", 4.60),
        ("initial_total_reduction_percent_over_parent", 2.04),
        ("confirmation_down_reduction_percent_over_parent", 4.54),
        ("confirmation_total_reduction_percent_over_parent", 1.98),
        ("repaired_gateup_ratio_to_current_c", 1.008),
        ("repaired_down_ratio_to_current_c", 1.020),
        ("repaired_total_ratio_to_current_c", 1.016),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    feature = require_dict(report, execution.get("feature_test_probe"), "feature test")
    report.check(feature.get("test_count") == 176 and feature.get("passed") is True, "feature test drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "c85ff9e5c85dc610f60ed4b26a1c05bd7b7604ba0d26c56d5b6fa2829c69e1df",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Down Paired-Scale Global-Load Address-Space Performance Repair"
    checker = "check_cuda_rust_down_paired_scale_global_load_address_space_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 286, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review drift")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review drift")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost paired scale loads", lambda value: value["implementation"].update({"repaired_down_global_u16_load_sites": 2})),
        ("retained packed Q2 regression", lambda value: value["ownership"].update({"changes_down_packed_q2_word_address_space": True})),
        ("lost confirmed speedup", lambda value: value["b300_execution"]["confirmation_repaired_profile"].update({"down_ms": 420.188})),
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
