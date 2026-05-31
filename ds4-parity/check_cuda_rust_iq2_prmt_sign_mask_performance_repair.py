#!/usr/bin/env python3
"""Validate the Rust CUDA IQ2 PRMT sign-mask performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-iq2-prmt-sign-mask-performance-repair.json"
CURRENT_REVISION = "1000e653df60a7814fa996d146e3823d0a364280"


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
        "cargo": (ROOT / "rust/ds4-cuda/Cargo.toml").read_text(encoding="utf-8"),
        "lock": (ROOT / "Cargo.lock").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA IQ2 PRMT sign-mask repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_iq2_prmt_sign_mask_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-rust-iq2-prmt-sign-mask-route-blocked", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("advances_cuda_oxide_for_prmt_intrinsic", True),
        ("changes_iq2_sign_mask_construction", True),
        ("retains_packed_twos_complement_application", True),
        ("retains_gate_dp4a_topology", True),
        ("retains_cached_down_codegen_metrics", True),
        ("retains_rowspan_policy", True),
        ("changes_default_current_c_route", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts)
    validate_equivalence(report, fixture, texts["kernels"])
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    kernels = texts["kernels"]
    helper = kernels.split("macro_rules! abi_moe_iq2_signed_word", 1)[1].split(
        "macro_rules! abi_moe_cached_weight_load_u16", 1
    )[0]
    for key, expected in [
        ("parent_cuda_oxide_revision", "485bdd86fc1c900ad15ebd421b3b187619fe0903"),
        ("repaired_cuda_oxide_revision", CURRENT_REVISION),
        ("intrinsic", "integer::prmt_b32_ba98"),
        ("ptx_operation", "prmt.b32"),
        ("ptx_selector", "0xba98"),
        ("parent_gate_prmt_sites", 0),
        ("repaired_gate_prmt_sites", 16),
        ("parent_gate_b32_register_bound", 1030),
        ("repaired_gate_b32_register_bound", 982),
        ("parent_gate_dp4a_sites", 128),
        ("repaired_gate_dp4a_sites", 128),
        ("parent_down_dp4a_sites", 256),
        ("repaired_down_dp4a_sites", 256),
        ("parent_down_b32_register_bound", 1530),
        ("repaired_down_b32_register_bound", 1530),
    ]:
        report.check(implementation.get(key) == expected, f"implementation evidence drift: {key}")
    report.check(f'rev = "{CURRENT_REVISION}"' in texts["cargo"], "current cuda-oxide manifest pin missing")
    report.check(f"#{CURRENT_REVISION}" in texts["lock"], "current cuda-oxide lock pin missing")
    report.check("let sign_bits =" in helper, "PRMT sign-bit preparation missing")
    report.check("integer::prmt_b32_ba98(sign_bits)" in helper, "PRMT sign-mask call missing")
    report.check("0_u32.wrapping_sub" not in helper, "scalar sign-mask expansion retained")
    report.check(".wrapping_add(mask & 0x0101_0101)" in helper, "packed negation contract missing")


def validate_equivalence(report: Report, fixture: dict[str, Any], kernels: str) -> None:
    proof = require_dict(report, fixture.get("equivalence_proof"), "equivalence proof")
    signs_match = re.search(r"ABI_MOE_IQ2_SIGNS: \[u8; 128\] = \[(.*?)\];", kernels, re.S)
    grids_match = re.search(r"ABI_MOE_IQ2_GRID: \[u64; 256\] = \[(.*?)\];", kernels, re.S)
    report.check(signs_match is not None, "sign table missing")
    report.check(grids_match is not None, "grid table missing")
    if signs_match is None or grids_match is None:
        return
    signs = [int(value, 0) for value in re.findall(r"0x[0-9A-Fa-f]+|\b\d+\b", signs_match.group(1))]
    grids = [
        int(value, 0)
        for value in re.findall(r"0x[0-9A-Fa-f]+|\b\d+\b", grids_match.group(1).replace("_", ""))
    ]
    mismatches = 0
    for grid in grids:
        for sign in signs:
            for lane in (0, 4):
                bits = sign >> lane
                sign_bits = (
                    ((bits & 1) << 7)
                    | ((bits & 2) << 14)
                    | ((bits & 4) << 21)
                    | ((bits & 8) << 28)
                )
                mask = 0
                for element in range(4):
                    mask |= (0xFF if sign_bits & (0x80 << (8 * element)) else 0) << (8 * element)
                packed = (((grid >> (8 * lane)) & 0xFFFFFFFF) ^ mask) + (mask & 0x01010101)
                scalar = 0
                for element in range(4):
                    value = (grid >> (8 * (lane + element))) & 0xFF
                    value = (-value if sign & (1 << (lane + element)) else value) & 0xFF
                    scalar |= value << (8 * element)
                mismatches += (packed & 0xFFFFFFFF) != scalar
    report.check(len(signs) == proof.get("live_sign_entry_count"), "sign table count drift")
    report.check(len(grids) == proof.get("live_grid_entry_count"), "grid table count drift")
    report.check(len(grids) * len(signs) * 2 == proof.get("checked_case_count"), "proof case count drift")
    report.check(mismatches == proof.get("mismatch_count"), "PRMT table equivalence mismatch")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_compiler_backend_sha256", "2b09580d620522eb63ad7082ea1e719443bcef78032cc93d3b410abf515c69da"),
        ("repaired_compiler_backend_sha256", "833c8487d4086735ca1c90216d47d21c64bf399511d96d40e6d6007c8edff63e"),
        ("parent_shared_library_sha256", "959f7d4c262a9efbf528a4b53261834a598b01941c68d8c3027e6e452fdfa275"),
        ("parent_ptx_sha256", "df8e1b5eceb9ebb7d626404deecdbc4785f48e7b773e0246f26297a3aba838af"),
        ("repaired_shared_library_sha256", "65faf1bff05339dd754e230e211ef97de1e2765cc788edaf1c53b94853e2ed59"),
        ("repaired_ptx_sha256", "496c8f743ac6fc9202642cfdc934263f7f35ba4ec364fbf8089fc95ba7abc48b"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_control_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_confirmation_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    for profile, label, gateup, down, total in [
        (parent, "parent", 752.988, 518.526, 1287.763),
        (repaired, "repaired", 721.447, 518.779, 1256.699),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.044),
        ("total_speedup_over_parent", 1.025),
        ("gateup_reduction_percent_over_parent", 4.19),
        ("total_reduction_percent_over_parent", 2.41),
        ("repaired_gateup_ratio_to_current_c", 1.39),
        ("repaired_down_ratio_to_current_c", 1.32),
        ("repaired_total_ratio_to_current_c", 1.36),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    rejected = require_dict(report, execution.get("rejected_preceding_scheduling_probes"), "rejected probes")
    gate_half = require_dict(report, rejected.get("four_entry_gate_up_scalar_halves"), "gate half probe")
    down_quarter = require_dict(report, rejected.get("four_entry_down_scalar_quarters"), "down quarter probe")
    report.check(gate_half.get("gateup_ms") == 807.985, "rejected gate-half timing drift")
    report.check(down_quarter.get("down_ms") == 642.851, "rejected down-quarter timing drift")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(official.get("summary_sha256") == "d9f242e37bf92b4b2eb3d4bb36faae8d4afcc6c8f433271b561e9461aeeb0979", "official summary hash drift")
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA IQ2 PRMT Sign-Mask Performance Repair"
    checker = "check_cuda_rust_iq2_prmt_sign_mask_performance_repair.py"
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
    report.check(validation.get("unified_report_passed") == 277, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost PRMT", lambda value: value["implementation"].update({"repaired_gate_prmt_sites": 0})),
        ("proof mismatch", lambda value: value["equivalence_proof"].update({"mismatch_count": 1})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_confirmation_profile"].update({"gateup_ms": 752.988})),
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
