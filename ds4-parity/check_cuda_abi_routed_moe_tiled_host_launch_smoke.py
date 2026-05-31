#!/usr/bin/env python3
"""Validate internal tiled routed-MoE host launch methods in the Rust ABI module."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-tiled-host-launch-smoke.json"
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


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} missing")
    return value if isinstance(value, dict) else {}


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
    print(f"{MILESTONE} Rust CUDA tiled host launch methods: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_routed_moe_tiled_host_launch_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-tiled-host-launch-methods", "status drift")
    for marker in [
        "const uint32_t use_expert_tiles = use_sorted_pairs",
        "const uint32_t use_atomic_down = use_expert_tiles",
        "const uint32_t use_gate_row2048 = use_expert_tiles",
        "const uint32_t use_down_row2048 = use_atomic_down",
        "moe_build_expert_tile_offsets_kernel<<<",
        "moe_build_expert_tiles_kernel<<<",
        "moe_gate_up_mid_expert_tile8_rowspan_kernel<",
        "moe_down_expert_tile16_rowspan_kernel<",
        "zero_kernel<<<",
    ]:
        report.check(marker in texts["cuda"], f"current-C tiled marker missing: {marker}")
    validate_scope(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_scope(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 74),
        ("exported_compute_symbol_count", 72),
        ("embedded_kernel_count", 91),
        ("consumes_sorted_host_launch_methods", True),
        ("adds_expert_tile_metadata_launch_methods", True),
        ("adds_row32_tile_launch_methods", True),
        ("adds_tile16_launch_methods", True),
        ("adds_rowspan_launch_methods", True),
        ("adds_cached_launch_methods", True),
        ("adds_atomic_output_zero_launch_method", True),
        ("owns_batched_quantized_orchestration", False),
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
        "pub struct CudaAbiRoutedMoeExpertTileHostLaunchScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "adds_expert_tile_metadata_launch_methods: true",
        "adds_rowspan_launch_methods: true",
        "adds_cached_launch_methods: true",
        "adds_atomic_output_zero_launch_method: true",
        "owns_batched_quantized_orchestration: false",
        "owns_routed_moe_batch_tensor: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    methods = implementation.get("host_launch_methods", [])
    for method in [
        "moe_build_expert_tile_offsets_tensor",
        "moe_build_expert_tiles_tensor",
        "moe_gate_up_mid_expert_tile4_row32_tensor",
        "moe_gate_up_mid_expert_tile8_row32_tensor",
        "moe_gate_up_mid_expert_tile8_rowspan_tensor",
        "moe_gate_up_mid_expert_tile8_rowspan_cached_tensor",
        "moe_atomic_output_zero_tensor",
        "moe_down_expert_tile4_row32_tensor",
        "moe_down_expert_tile8_row32_tensor",
        "moe_down_expert_tile16_row32_tensor",
        "moe_down_expert_tile16_rowspan_tensor",
        "moe_down_expert_tile16_rowspan_cached_tensor",
    ]:
        report.check(method in methods, f"evidence method missing: {method}")
        report.check(f"fn {method}(" in texts["kernels"], f"host launch method missing: {method}")
    for marker in [
        "let mut tile_experts_len = u64::from(tile_capacity);",
        "let mut tile_starts_len = u64::from(tile_capacity);",
        "grid_dim: (expert_mid_dim.div_ceil(32), tile_capacity, 1)",
        "grid_dim: (out_dim.div_ceil(32), tile_capacity, 1)",
        "grid_dim: (expert_mid_dim.div_ceil(row_span), tile_capacity, 1)",
        "grid_dim: (out_dim.div_ceil(row_span), tile_capacity, 1)",
        "if row_span == 0 {",
        "let mut down_len = if atomic_out != 0 {",
        "u64::from(n_tokens) * u64::from(out_dim)",
        "u64::from(pair_count) * u64::from(out_dim)",
    ]:
        report.check(marker in texts["kernels"], f"host method shape missing: {marker}")
    report.check("caller-provided validated scratch" in implementation.get("scratch_boundary", ""), "scratch boundary missing")
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
        ("local_library_test_count", 164),
        ("feature_release_test_count", 171),
        ("staticlib_export_count", 74),
        ("embedded_kernel_count", 91),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "single_token_public_regression_remains_passing",
        "expert_tile_launch_methods_compile",
        "rowspan_cached_launch_methods_compile",
        "atomic_down_launch_methods_compile",
        "public_batch_route_not_yet_invoked",
    ]:
        report.check(observed.get(key) is True, f"observed module smoke drift: {key}")
    report.check("embedded_routed_moe_kernels_loaded" in texts["harness"], "regression harness marker missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 164),
        ("b300_feature_release_test_count", 171),
        ("cuda_abi_comparators_passed", 78),
        ("unified_report_passed", 251),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    checker = "check_cuda_abi_routed_moe_tiled_host_launch_smoke.py"
    item = f"{MILESTONE}: Tiled Routed MoE Host Launch Methods"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(MILESTONE in texts["status"], "status item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("methods missing", lambda value: value["implementation"]["host_launch_methods"].clear()),
        ("atomic launch claim missing", lambda value: value["ownership"].update({"adds_atomic_output_zero_launch_method": False})),
        ("batch overclaim", lambda value: value["ownership"].update({"owns_routed_moe_batch_tensor": True})),
        ("orchestration overclaim", lambda value: value["ownership"].update({"owns_batched_quantized_orchestration": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
