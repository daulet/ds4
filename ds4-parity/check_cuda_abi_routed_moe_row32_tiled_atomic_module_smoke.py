#!/usr/bin/env python3
"""Validate retained row32 tiled and atomic-down kernels in the Rust CUDA ABI module."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-row32-tiled-atomic-module-smoke.json"
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
    print(f"{MILESTONE} Rust CUDA embedded row32 tiled atomic ABI module: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_routed_moe_row32_tiled_atomic_module_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-embedded-row32-tiled-atomic-module",
        "status drift",
    )
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    for marker in [
        "moe_gate_up_mid_expert_tile4_row32_kernel<<<",
        "moe_gate_up_mid_expert_tile8_row32_kernel<<<",
        "moe_down_expert_tile4_row32_kernel<<<",
        "moe_down_expert_tile8_row32_kernel<<<",
        "const uint32_t use_atomic_down =",
        "zero_kernel<<<",
        "const uint32_t use_down_tile16 =",
        "const uint32_t use_down_row2048 =",
    ]:
        report.check(marker in texts["cuda"], f"current-C route marker missing: {marker}")
    for marker in [
        "pub fn moe_gate_up_mid_expert_tile4_row32_kernel",
        "pub fn moe_down_expert_tile4_row32_kernel",
        "pub fn moe_gate_up_mid_expert_tile8_row32_kernel",
        "pub fn moe_down_expert_tile8_row32_kernel",
        "pub fn zero_kernel",
        "M14_5C2C4_SCOPE.owns_tile4_and_tile8_row32_atomic_down_dispatch",
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
        ("embedded_kernel_count", 86),
        ("consumes_expert_tile_metadata_module", True),
        ("embeds_tile4_and_tile8_gate_projection_kernels", True),
        ("embeds_tile4_and_tile8_down_projection_kernels", True),
        ("embeds_row32_atomic_zero_kernel", True),
        ("owns_row32_atomic_down_surface", True),
        ("owns_tile16_or_rowspan_batch_dispatch", False),
        ("owns_routed_moe_batch_tensor", False),
        ("owns_complete_routed_moe_abi", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    report.check(len(symbols) == 74, "Rust ABI export count drift")
    report.check("ds4_gpu_routed_moe_one_tensor" in symbols, "published single-token export missing")
    report.check("ds4_gpu_routed_moe_batch_tensor" not in symbols, "public batched route exported early")
    for marker in [
        "pub struct CudaAbiRoutedMoeRow32TiledAtomicModuleScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "embedded_kernel_count: 86",
        "embeds_tile4_and_tile8_gate_projection_kernels: true",
        "embeds_tile4_and_tile8_down_projection_kernels: true",
        "embeds_row32_atomic_zero_kernel: true",
        "owns_row32_atomic_down_surface: true",
        "owns_tile16_or_rowspan_batch_dispatch: false",
        "owns_routed_moe_batch_tensor: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    entries = implementation.get("embedded_kernel_entries", [])
    expected_entries = [
        "abi_moe_gate_up_mid_expert_tile4_row32_kernel",
        "abi_moe_gate_up_mid_expert_tile8_row32_kernel",
        "abi_moe_atomic_output_zero_kernel",
        "abi_moe_down_expert_tile4_row32_kernel",
        "abi_moe_down_expert_tile8_row32_kernel",
    ]
    for entry in expected_entries:
        report.check(entry in entries, f"evidence entry missing: {entry}")
        report.check(f"pub fn {entry}" in texts["kernels"], f"embedded kernel missing: {entry}")
        report.check(f'load_function("{entry}")' in texts["kernels"], f"loader entry missing: {entry}")
    report.check("DeviceAtomicF32" in texts["kernels"], "device float atomic import missing")
    report.check("fetch_add(accumulator, AtomicOrdering::Relaxed)" in texts["kernels"], "atomic down operation missing")
    report.check("tile16" in implementation.get("remaining_compute_boundary", ""), "tile16 boundary missing")
    report.check("row-span" in implementation.get("remaining_compute_boundary", ""), "row-span boundary missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("target", "sm_80"),
        ("local_library_test_count", 157),
        ("feature_release_test_count", 164),
        ("staticlib_export_count", 74),
        ("embedded_kernel_count", 86),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "embedded_routed_moe_kernels_loaded",
        "single_token_public_regression_remains_passing",
        "tile4_gate_and_down_entries_generated",
        "tile8_gate_and_down_entries_generated",
        "atomic_output_zero_entry_generated",
        "device_atomic_f32_path_lowered",
    ]:
        report.check(observed.get(key) is True, f"observed module smoke drift: {key}")
    report.check("embedded_routed_moe_kernels_loaded" in texts["harness"], "regression harness marker missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 157),
        ("b300_feature_release_test_count", 164),
        ("cuda_abi_comparators_passed", 71),
        ("unified_report_passed", 244),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-row32-tiled-atomic-module-smoke.json"
    checker = "check_cuda_abi_routed_moe_row32_tiled_atomic_module_smoke.py"
    item = f"{MILESTONE}: Embedded Row32 Tiled And Atomic Routed MoE ABI Module"
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
        ("atomic zero entry missing", lambda value: value["implementation"]["embedded_kernel_entries"].pop(2)),
        ("row32 atomic ownership missing", lambda value: value["ownership"].update({"owns_row32_atomic_down_surface": False})),
        ("wider route overclaim", lambda value: value["ownership"].update({"owns_tile16_or_rowspan_batch_dispatch": True})),
        ("kernel count mismatch", lambda value: value["ownership"].update({"embedded_kernel_count": 85})),
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
