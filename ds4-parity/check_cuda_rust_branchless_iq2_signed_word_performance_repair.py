#!/usr/bin/env python3
"""Validate the Rust CUDA branchless IQ2 signed-word performance repair."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-branchless-iq2-signed-word-performance-repair.json"


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
        "cuda": (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8"),
        "cargo": (ROOT / "rust/ds4-cuda/Cargo.toml").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA branchless IQ2 signed-word repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_branchless_iq2_signed_word_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-branchless-iq2-retained-route-blocked", "status drift")
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_iq2_signed_word_helper", True),
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
    if "macro_rules! abi_moe_iq2_signed_word" in kernels:
        helper = kernels.split("macro_rules! abi_moe_iq2_signed_word", 1)[1].split(
            "#[allow(clippy::too_many_arguments, static_mut_refs)]", 1
        )[0]
    else:
        helper = kernels.split("fn abi_moe_iq2_signed_word(", 1)[1].split(
            "fn abi_moe_cached_q8_bsum(", 1
        )[0]
    report.check("while element" not in helper, "conditional byte loop retained")
    report.check(
        "0_u32.wrapping_sub" in helper or "integer::prmt_b32_ba98(sign_bits)" in helper,
        "branchless byte-mask construction or PRMT successor missing",
    )
    report.check(".wrapping_add(mask & 0x0101_0101)" in helper, "packed two's-complement transform missing")
    report.check("__vcmpne4" in texts["cuda"], "current-C packed compare evidence missing")
    report.check("__vsub4" in texts["cuda"], "current-C packed subtract evidence missing")
    report.check('rev = "ae721dc95912a918f182d13b7ca55281aa29d8f9"' in texts["cargo"], "cuda-oxide pin drift")
    proof = require_dict(report, fixture.get("equivalence_proof"), "equivalence proof")
    signs_match = re.search(r"ABI_MOE_IQ2_SIGNS: \[u8; 128\] = \[(.*?)\];", kernels, re.S)
    grids_match = re.search(r"ABI_MOE_IQ2_GRID: \[u64; 256\] = \[(.*?)\];", kernels, re.S)
    report.check(signs_match is not None, "sign table missing")
    report.check(grids_match is not None, "grid table missing")
    if signs_match is None or grids_match is None:
        return
    signs = [int(value, 0) for value in re.findall(r"0x[0-9A-Fa-f]+|\b\d+\b", signs_match.group(1))]
    grid_body = grids_match.group(1).replace("_", "")
    grids = [int(value, 0) for value in re.findall(r"0x[0-9A-Fa-f]+|\b\d+\b", grid_body)]
    report.check(len(signs) == proof.get("live_sign_entry_count"), "sign table count drift")
    report.check(len(grids) == proof.get("live_grid_entry_count"), "grid table count drift")
    minimum = min((grid >> (8 * lane)) & 0xff for grid in grids for lane in range(8))
    mismatches = 0
    for grid in grids:
        for sign in signs:
            for lane in (0, 4):
                scalar = 0
                for element in range(4):
                    source_lane = lane + element
                    value = (grid >> (8 * source_lane)) & 0xff
                    signed = (-value if sign & (1 << source_lane) else value) & 0xff
                    scalar |= signed << (8 * element)
                bits = sign >> lane
                mask = (
                    (-(bits & 1) & 0x000000FF)
                    | (-((bits >> 1) & 1) & 0x0000FF00)
                    | (-((bits >> 2) & 1) & 0x00FF0000)
                    | (-((bits >> 3) & 1) & 0xFF000000)
                )
                values = (grid >> (8 * lane)) & 0xFFFFFFFF
                packed = ((values ^ mask) + (mask & 0x01010101)) & 0xFFFFFFFF
                mismatches += scalar != packed
    report.check(minimum == proof.get("minimum_grid_byte"), "minimum grid byte drift")
    report.check(mismatches == proof.get("mismatch_count"), "table equivalence mismatch")
    report.check(len(grids) * len(signs) * 2 == proof.get("checked_case_count"), "proof case count drift")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("repaired_profiled_shared_library_sha256", "383fe12843109a33719bcbac5e38ec1b22ea95f9e45647b2ecc47c3f88a79f01"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    parent = require_dict(report, execution.get("parent_rebuild_profile"), "parent profile")
    repaired = require_dict(report, execution.get("repaired_rebuild_profile"), "repaired profile")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for profile, label, gateup, down, total in [
        (parent, "parent", 2796.005, 1127.705, 3940.005),
        (repaired, "repaired", 2691.558, 1125.908, 3833.824),
        (current_c, "current-C", 517.818, 393.200, 924.283),
    ]:
        report.check(profile.get("gateup_ms") == gateup, f"{label} gate/up drift")
        report.check(profile.get("down_ms") == down, f"{label} down drift")
        report.check(profile.get("total_ms") == total, f"{label} total drift")
    for key, expected in [
        ("gateup_speedup_over_parent", 1.04),
        ("total_speedup_over_parent", 1.03),
        ("prefill_gain_percent_over_parent", 1.38),
        ("repaired_gateup_ratio_to_current_c", 5.20),
        ("repaired_down_ratio_to_current_c", 2.86),
        ("repaired_total_ratio_to_current_c", 4.15),
        ("remaining_primary_bottleneck", "cached-gateup-iq2-instruction-gap"),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(official.get("summary_sha256") == "44258442d0ebfa546be03133b666407659e8aca42cd1b1f06bcdc37a8ca81906", "official summary hash drift")
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Branchless IQ2 Signed-Word Performance Repair"
    checker = "check_cuda_rust_branchless_iq2_signed_word_performance_repair.py"
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
    report.check(validation.get("unified_report_skipped") == 45, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("proof mismatch", lambda value: value["equivalence_proof"].update({"mismatch_count": 1})),
        ("lost speedup", lambda value: value["b300_execution"]["repaired_rebuild_profile"].update({"gateup_ms": 2796.005})),
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
