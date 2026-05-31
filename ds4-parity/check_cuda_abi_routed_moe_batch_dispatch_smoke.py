#!/usr/bin/env python3
"""Validate the public Rust CUDA batched routed-MoE ABI dispatch leaf."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-batch-dispatch-smoke.json"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_routed_moe_batch_link_smoke.c"


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
    status = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA public batch dispatch: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_routed_moe_batch_dispatch_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-public-batch-dispatch", "status drift")
    for marker in [
        "static int routed_moe_launch(",
        'extern "C" int ds4_gpu_routed_moe_batch_tensor',
        "const uint32_t use_sorted_pairs = n_tokens > 1u;",
        "const uint32_t use_expert_tiles = use_sorted_pairs",
        "const uint32_t use_atomic_down = use_expert_tiles",
        "const uint32_t use_down_tile16 = use_atomic_down",
    ]:
        report.check(marker in texts["cuda"], f"current-C dispatch marker missing: {marker}")
    validate_scope(report, fixture, texts)
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_scope(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 75),
        ("exported_compute_symbol_count", 73),
        ("embedded_kernel_count", 91),
        ("consumes_tiled_host_launch_methods", True),
        ("owns_routed_moe_batch_tensor", True),
        ("preserves_single_token_delegation", True),
        ("preserves_f32_mid_result_contract", True),
        ("owns_f32_sorted_tiled_and_atomic_dispatch", True),
        ("owns_complete_routed_moe_abi", True),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    report.check(len(symbols) == 75, "Rust ABI export count drift")
    report.check("ds4_gpu_routed_moe_one_tensor" in symbols, "single-token export missing")
    report.check("ds4_gpu_routed_moe_batch_tensor" in symbols, "public batch export missing")
    for marker in [
        "pub struct CudaAbiRoutedMoeBatchScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "owns_routed_moe_batch_tensor: true",
        "owns_complete_routed_moe_abi: true",
        "owns_remaining_graph_compute_abi: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("public_export") == "ds4_gpu_routed_moe_batch_tensor", "public export evidence drift")
    report.check("Persistent mutex-guarded DeviceBuffer scratch" in implementation.get("scratch_boundary", ""), "scratch evidence missing")
    for route in ["f32_fallback", "sorted_p2", "sorted_no_p2", "tiled_row32", "tile4_atomic", "tile16_rowspan_atomic"]:
        report.check(route in implementation.get("validated_routes", []), f"route evidence missing: {route}")
    for marker in [
        "static ABI_ROUTED_MOE_BATCH_SCRATCH",
        "struct AbiRoutedMoeBatchScratch",
        "fn with_abi_routed_moe_batch_scratch",
        "fn clear_abi_routed_moe_counts",
        "pub unsafe extern \"C\" fn ds4_gpu_routed_moe_batch_tensor",
        "ds4_gpu_routed_moe_one_tensor(",
        "kernels.moe_gate_up_mid_f32_tensor(",
        ".moe_gate_up_mid_sorted_p2_qwarp32_tensor(",
        ".moe_gate_up_mid_expert_tile8_rowspan_tensor(",
        ".moe_down_expert_tile16_rowspan_tensor(",
        "kernels.moe_atomic_output_zero_tensor(",
        "*mid_is_f16 = false",
    ]:
        report.check(marker in texts["abi"], f"batch implementation marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("target", "sm_80"),
        ("local_library_test_count", 165),
        ("feature_release_test_count", 172),
        ("staticlib_export_count", 75),
        ("embedded_kernel_count", 91),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "single_token_public_regression_remains_passing",
        "f32_fallback",
        "sorted_p2",
        "sorted_no_p2",
        "tiled_row32",
        "tile4_atomic",
        "tile16_rowspan_atomic",
        "mid_is_f16_false_on_success",
        "failed_batch_preserves_mid_result",
    ]:
        report.check(observed.get(key) is True, f"observed dispatch drift: {key}")
    for marker in [
        "ds4_gpu_routed_moe_batch_tensor(",
        "N_TOKENS 128u",
        "DS4_CUDA_MOE_NO_EXPERT_TILES",
        "DS4_CUDA_MOE_TILE4",
        "DS4_CUDA_MOE_ATOMIC_DOWN",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
        "failed_batch_preserves_mid_result",
    ]:
        report.check(marker in texts["harness"], f"linked harness marker missing: {marker}")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 165),
        ("b300_feature_release_test_count", 172),
        ("cuda_abi_comparators_passed", 79),
        ("unified_report_passed", 252),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    checker = "check_cuda_abi_routed_moe_batch_dispatch_smoke.py"
    item = f"{MILESTONE}: Public Batched Routed MoE ABI Dispatch"
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
        ("public ownership removed", lambda value: value["ownership"].update({"owns_routed_moe_batch_tensor": False})),
        ("route evidence removed", lambda value: value["implementation"]["validated_routes"].clear()),
        ("B300 atomic path removed", lambda value: value["b300_execution"]["observed"].update({"tile16_rowspan_atomic": False})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
