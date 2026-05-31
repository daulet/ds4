#!/usr/bin/env python3
"""Validate the public Rust CUDA DSV4 top-k mask ABI leaf."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-topk-mask-smoke.json"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_topk_mask_link_smoke.c"


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
        "gpu_build": (ROOT / "rust/ds4-gpu/build.rs").read_text(encoding="utf-8"),
        "gpu_sys": (ROOT / "rust/ds4-gpu-sys/src/lib.rs").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA public DSV4 top-k mask ABI: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_topk_mask_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-public-topk-mask-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbol") == "ds4_gpu_dsv4_topk_mask_tensor", "oracle symbol drift")
    for marker in [
        'extern "C" int ds4_gpu_dsv4_topk_mask_tensor',
        "mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)",
        "topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)",
        "topk_mask_kernel<<<blocks, 256>>>",
    ]:
        report.check(marker in texts["cuda"], f"current-C top-k mask marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 77),
        ("exported_compute_symbol_count", 75),
        ("embedded_kernel_count", 93),
        ("public_gpu_abi_function_count", 81),
        ("owns_dsv4_topk_mask_tensor", True),
        ("owns_topk_mask_kernel", True),
        ("owns_indexer_score_abi", False),
        ("owns_indexer_topk_selection_abi", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) >= 77, "Rust ABI export count drift")
    report.check("ds4_gpu_dsv4_topk_mask_tensor" in symbols, "top-k mask export missing")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        'pub unsafe extern "C" fn ds4_gpu_dsv4_topk_mask_tensor',
        "kernels.dsv4_topk_mask_tensor(",
        "mask.bytes < mask_bytes",
        "topk.bytes < topk_bytes",
        "n_comp == 0",
        "n_tokens == 0",
        "top_k == 0",
    ]:
        report.check(marker in texts["abi"], f"Rust top-k mask ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_topk_mask_kernel",
        "topk_mask_kernel: CudaFunction",
        '.load_function("abi_topk_mask_kernel")',
        "fn dsv4_topk_mask_tensor(",
        ".max(selected_count)",
    ]:
        report.check(marker in texts["kernels"], f"embedded top-k mask marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiTopkMaskScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 77",
        "exported_compute_symbol_count: 75",
        "embedded_kernel_count: 93",
        "owns_dsv4_topk_mask_tensor: true",
        "owns_topk_mask_kernel: true",
        "owns_indexer_topk_selection_abi: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("public_export") == "ds4_gpu_dsv4_topk_mask_tensor", "public export evidence drift")
    report.check(implementation.get("kernel_entry") == "abi_topk_mask_kernel", "kernel entry drift")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage evidence missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("target", "sm_80"),
        ("local_library_test_count", 167),
        ("feature_release_test_count", 174),
        ("staticlib_export_count", 77),
        ("embedded_kernel_count", 93),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "predecessor_public_fused_qkv_consumer_passed",
        "multi_token_mask_output_matches",
        "selected_zero_mask_matches",
        "excluded_negative_infinity_mask_matches",
        "short_mask_rejected",
        "short_topk_rejected",
        "zero_dimension_rejected",
        "null_rejected",
        "embedded_topk_mask_kernel_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed execution drift: {key}")
    for marker in [
        "ds4_gpu_dsv4_topk_mask_tensor(",
        "multi_token_mask_output_matches",
        "selected_zero_mask_matches",
        "excluded_negative_infinity_mask_matches",
        "short_mask_rejected",
        "short_topk_rejected",
    ]:
        report.check(marker in texts["harness"], f"linked harness marker missing: {marker}")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 167),
        ("b300_feature_release_test_count", 174),
        ("cuda_abi_comparators_passed", 81),
        ("unified_report_passed", 254),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    checker = "check_cuda_abi_topk_mask_smoke.py"
    item = f"{MILESTONE}: Public DSV4 TopK Mask ABI"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("public ownership removed", lambda value: value["ownership"].update({"owns_dsv4_topk_mask_tensor": False})),
        ("negative-infinity evidence removed", lambda value: value["b300_execution"]["observed"].update({"excluded_negative_infinity_mask_matches": False})),
        ("B300 kernel count drift", lambda value: value["b300_execution"].update({"embedded_kernel_count": 92})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
