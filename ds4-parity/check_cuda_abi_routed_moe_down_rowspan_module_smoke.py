#!/usr/bin/env python3
"""Validate retained down row-span kernel in the Rust CUDA ABI module."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-down-rowspan-module-smoke.json"
REGRESSION_HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba/abi_routed_moe_one_link_smoke.c"


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


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "cuda": (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8"),
        "lib": (ROOT / "rust/ds4-cuda/src/lib.rs").read_text(encoding="utf-8"),
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
        "kernels": (ROOT / "rust/ds4-cuda/src/abi_kernels.rs").read_text(encoding="utf-8"),
        "prior_smoke": (ROOT / "rust/ds4-cuda/src/bin/routed_moe_tile8_row32_smoke.rs").read_text(encoding="utf-8"),
        "harness": REGRESSION_HARNESS.read_text(encoding="utf-8"),
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
    status = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA embedded down row-span ABI module: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_routed_moe_down_rowspan_module_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-embedded-down-rowspan-module", "status drift")
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    for marker in [
        "moe_down_expert_tile16_rowspan_kernel<512><<<",
        "moe_down_expert_tile16_rowspan_kernel<1024><<<",
        "moe_down_expert_tile16_row2048_kernel<<<",
        "const uint32_t down_row_span =",
        "const uint32_t use_down_row2048 =",
    ]:
        report.check(marker in texts["cuda"], f"current-C route marker missing: {marker}")
    for marker in [
        "pub fn moe_down_expert_tile16_rowspan_kernel",
        "M14_5C2C7_SCOPE.owns_moe_down_expert_tile16_rowspan_kernel",
        "M14_5C2C7_SCOPE.owns_down_row512_row1024_and_row2048_atomic_dispatch",
    ]:
        report.check(marker in texts["prior_smoke"], f"prior proof marker missing: {marker}")
    validate_scope(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_scope(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 74),
        ("exported_compute_symbol_count", 72),
        ("embedded_kernel_count", 89),
        ("consumes_gate_rowspan_module", True),
        ("reuses_tile16_expert_tile_metadata", True),
        ("embeds_down_tile16_rowspan_kernel", True),
        ("owns_down_rowspan_atomic_functional_surface", True),
        ("owns_shared_cache_or_public_batch_dispatch", False),
        ("owns_routed_moe_batch_tensor", False),
        ("owns_complete_routed_moe_abi", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    report.check(len(symbols) >= 74, "published Rust ABI exports disappeared")
    report.check("ds4_gpu_routed_moe_one_tensor" in symbols, "published single-token export missing")
    report.check("ds4_gpu_routed_moe_one_tensor" in symbols, "published single-token export missing")
    for marker in [
        "pub struct CudaAbiRoutedMoeDownRowspanModuleScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "embedded_kernel_count: 89",
        "embeds_down_tile16_rowspan_kernel: true",
        "owns_down_rowspan_atomic_functional_surface: true",
        "owns_shared_cache_or_public_batch_dispatch: false",
        "owns_routed_moe_batch_tensor: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    entry = "abi_moe_down_expert_tile16_rowspan_kernel"
    report.check(entry in implementation.get("embedded_kernel_entries", []), f"evidence entry missing: {entry}")
    report.check(f"pub fn {entry}" in texts["kernels"], f"embedded kernel missing: {entry}")
    report.check(f'load_function("{entry}")' in texts["kernels"], f"loader entry missing: {entry}")
    report.check("while row_offset < row_span" in texts["kernels"], "row-span loop missing")
    report.check("if local_start & 8 != 0" in texts["kernels"], "tile16 alignment guard missing")
    report.check("cell.fetch_add(accumulator, AtomicOrdering::Relaxed)" in texts["kernels"], "atomic behavior missing")
    report.check("shared-cache" in implementation.get("remaining_compute_boundary", ""), "shared-cache boundary missing")
    report.check("does not claim the shared-memory staging" in implementation.get("optimization_boundary", ""), "optimization boundary missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("target", "sm_80"),
        ("local_library_test_count", 160),
        ("feature_release_test_count", 167),
        ("staticlib_export_count", 74),
        ("embedded_kernel_count", 89),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "embedded_routed_moe_kernels_loaded",
        "single_token_public_regression_remains_passing",
        "down_rowspan_entry_generated",
        "tile16_alignment_and_atomic_semantics_retained",
        "functional_non_cached_surface_only",
    ]:
        report.check(observed.get(key) is True, f"observed module smoke drift: {key}")
    report.check("embedded_routed_moe_kernels_loaded" in texts["harness"], "regression harness marker missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 160),
        ("b300_feature_release_test_count", 167),
        ("cuda_abi_comparators_passed", 74),
        ("unified_report_passed", 247),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-down-rowspan-module-smoke.json"
    checker = "check_cuda_abi_routed_moe_down_rowspan_module_smoke.py"
    item = f"{MILESTONE}: Embedded Down Rowspan Routed MoE ABI Module"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(path in texts["roadmap"], "roadmap fixture missing")
    report.check(path in texts["todo"], "TODO fixture missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("down entry missing", lambda value: value["implementation"]["embedded_kernel_entries"].clear()),
        ("down ownership missing", lambda value: value["ownership"].update({"owns_down_rowspan_atomic_functional_surface": False})),
        ("cache overclaim", lambda value: value["ownership"].update({"owns_shared_cache_or_public_batch_dispatch": True})),
        ("kernel count mismatch", lambda value: value["ownership"].update({"embedded_kernel_count": 88})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
