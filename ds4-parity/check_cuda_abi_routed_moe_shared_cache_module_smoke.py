#!/usr/bin/env python3
"""Validate retained shared-cache kernels in the Rust CUDA ABI module."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-shared-cache-module-smoke.json"
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
    print(f"{MILESTONE} Rust CUDA embedded shared-cache ABI module: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_routed_moe_shared_cache_module_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-embedded-shared-cache-module", "status drift")
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    for marker in [
        "__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(",
        "__global__ static void moe_down_expert_tile16_rowspan_kernel(",
        "__shared__ cuda_block_q8_K sxq[8][16];",
        "__shared__ uint64_t s_iq2_grid[256];",
        "__shared__ uint8_t s_iq2_signs[128];",
        "__shared__ cuda_block_q8_K sxq[16][8];",
        "__syncthreads();",
    ]:
        report.check(marker in texts["cuda"], f"current-C cache marker missing: {marker}")
    for marker in [
        "pub fn moe_gate_up_mid_expert_tile8_rowspan_cached_kernel",
        "pub fn moe_down_expert_tile16_rowspan_cached_kernel",
        "M14_5C2E_SCOPE.owns_shared_cache_specialization",
        "M14_5C2E_SCOPE.owns_gate_and_down_cached_rowspan_dispatch",
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
        ("embedded_kernel_count", 91),
        ("consumes_down_rowspan_module", True),
        ("embeds_gate_rowspan_cached_kernel", True),
        ("embeds_down_rowspan_cached_kernel", True),
        ("owns_cached_gate_and_down_rowspan_surface", True),
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
        "pub struct CudaAbiRoutedMoeSharedCacheModuleScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "embedded_kernel_count: 91",
        "embeds_gate_rowspan_cached_kernel: true",
        "embeds_down_rowspan_cached_kernel: true",
        "owns_cached_gate_and_down_rowspan_surface: true",
        "owns_routed_moe_batch_tensor: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    for entry in [
        "abi_moe_gate_up_mid_expert_tile8_rowspan_cached_kernel",
        "abi_moe_down_expert_tile16_rowspan_cached_kernel",
    ]:
        report.check(entry in implementation.get("embedded_kernel_entries", []), f"evidence entry missing: {entry}")
        report.check(f"pub fn {entry}" in texts["kernels"], f"embedded kernel missing: {entry}")
        report.check(f'load_function("{entry}")' in texts["kernels"], f"loader entry missing: {entry}")
    for marker in [
        "ABI_MOE_CACHED_GATE_MAX_BLOCKS: usize = 16",
        "ABI_MOE_CACHED_DOWN_MAX_BLOCKS: usize = 8",
        "thread::sync_threads();",
        "S_IQ2_GRID",
        "abi_moe_iq2_q8_k_cached_dot",
        "abi_moe_q2_q8_k_cached_dot",
    ]:
        report.check(marker in texts["kernels"], f"cached kernel marker missing: {marker}")
    report.check("xq_blocks <= 16" in implementation.get("cache_boundary", ""), "gate cache boundary missing")
    report.check("midq_blocks <= 8" in implementation.get("cache_boundary", ""), "down cache boundary missing")
    report.check("does not export ds4_gpu_routed_moe_batch_tensor" in implementation.get("remaining_compute_boundary", ""), "public batch boundary missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("target", "sm_80"),
        ("local_library_test_count", 161),
        ("feature_release_test_count", 168),
        ("staticlib_export_count", 74),
        ("embedded_kernel_count", 91),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "embedded_routed_moe_kernels_loaded",
        "single_token_public_regression_remains_passing",
        "gate_cached_rowspan_entry_generated",
        "down_cached_rowspan_entry_generated",
        "packed_q8_and_iq2_shared_cache_surface_embedded",
        "bounded_cache_capacity_retained",
    ]:
        report.check(observed.get(key) is True, f"observed module smoke drift: {key}")
    report.check("embedded_routed_moe_kernels_loaded" in texts["harness"], "regression harness marker missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 161),
        ("b300_feature_release_test_count", 168),
        ("cuda_abi_comparators_passed", 75),
        ("unified_report_passed", 248),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-shared-cache-module-smoke.json"
    checker = "check_cuda_abi_routed_moe_shared_cache_module_smoke.py"
    item = f"{MILESTONE}: Embedded Shared Cache Routed MoE ABI Module"
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
        ("cached entries missing", lambda value: value["implementation"]["embedded_kernel_entries"].clear()),
        ("cached ownership missing", lambda value: value["ownership"].update({"owns_cached_gate_and_down_rowspan_surface": False})),
        ("batch overclaim", lambda value: value["ownership"].update({"owns_routed_moe_batch_tensor": True})),
        ("kernel count mismatch", lambda value: value["ownership"].update({"embedded_kernel_count": 89})),
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
