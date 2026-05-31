#!/usr/bin/env python3
"""Validate the Rust CUDA public compressor update ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-compressor-update-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_compressor_update_link_smoke.c"
GPU_BUILD = ROOT / "rust/ds4-gpu/build.rs"
GPU_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"


@dataclass
class ReportState:
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
        "cuda_c": CUDA_C.read_text(encoding="utf-8"),
        "lib": CUDA_LIB.read_text(encoding="utf-8"),
        "abi": CUDA_ABI.read_text(encoding="utf-8"),
        "kernels": CUDA_KERNELS.read_text(encoding="utf-8"),
        "harness": HARNESS.read_text(encoding="utf-8"),
        "gpu_build": GPU_BUILD.read_text(encoding="utf-8"),
        "gpu_sys": GPU_SYS.read_text(encoding="utf-8"),
        "roadmap": ROADMAP.read_text(encoding="utf-8"),
        "todo": TODO.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
        "readme": README.read_text(encoding="utf-8"),
        "report": REPORT.read_text(encoding="utf-8"),
    }
    report = ReportState()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA public compressor update ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_compressor_update_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-compressor-update-abi",
        "status drift",
    )
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["ds4_gpu_compressor_update_tensor"], "oracle symbols drift")
    for marker in [
        'extern "C" int ds4_gpu_compressor_update_tensor(',
        "const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;",
        "ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,",
        "if (!emit) return 1;",
        "compressor_update_pool_kernel<<<",
        "ds4_gpu_rms_norm_weight_rows_tensor",
        "ds4_gpu_rope_tail_tensor",
        "compressor_shift_ratio4_kernel<<<",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C update marker missing: {marker}")
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 61),
        ("exported_compute_symbol_count", 38),
        ("public_gpu_abi_function_count", 81),
        ("consumes_cached_model_ranges", True),
        ("owns_compressor_update_tensor", True),
        ("owns_compressor_update_pool_kernel", True),
        ("owns_compressor_shift_ratio4_kernel", True),
        ("reuses_store_norm_rope_kernels", True),
        ("owns_general_prefill", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 71, "Rust ABI export implementation count drift")
    report.check("ds4_gpu_compressor_update_tensor" in symbols, "update export missing")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        'pub unsafe extern "C" fn ds4_gpu_compressor_update_tensor',
        "let emit = pos.wrapping_add(1) % ratio == 0;",
        "ds4_gpu_compressor_store_batch_tensor(",
        "kernels.compressor_update_pool_tensor(",
        "kernels.rms_norm_weight_rows_tensor(",
        "if n_rot == 0 {",
        "kernels.rope_tail_tensor(",
        "kernels.compressor_shift_ratio4_tensor(",
        "pub fn abi_compressor_update_pool_kernel",
        "pub fn abi_compressor_shift_ratio4_kernel",
    ]:
        target = texts["kernels"] if marker.startswith("pub fn abi_") else texts["abi"]
        report.check(marker in target, f"Rust update marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiCompressorUpdateScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 61",
        "exported_compute_symbol_count: 38",
        "owns_compressor_update_tensor: true",
        "owns_general_prefill: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(
        implementation.get("embedded_kernel_entries")
        == ["abi_compressor_update_pool_kernel", "abi_compressor_shift_ratio4_kernel"],
        "kernel ownership drift",
    )
    for kernel in ["abi_compressor_store_kernel", "abi_rms_norm_weight_kernel", "abi_rope_tail_kernel"]:
        report.check(kernel in implementation.get("reused_embedded_kernels", []), f"kernel reuse missing: {kernel}")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage path missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 145),
        ("feature_release_test_count", 152),
        ("staticlib_export_count", 61),
        ("embedded_kernel_count", 38),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "ratio4_no_emit_store_only_matches",
        "ratio4_emit_f16_output_matches",
        "general_ratio_emit_output_matches",
        "uint32_emit_wrap_matches",
        "ratio4_shift_after_emit_matches",
        "n_rot_zero_partial_failure_matches",
        "invalid_model_range_preserves_state_and_output",
        "invalid_shape_rejected",
        "checked_overflow_rejected",
        "null_rejected",
        "embedded_compressor_update_pool_kernel_loaded",
        "embedded_compressor_shift_ratio4_kernel_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 55, "predecessor count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 55, "warning count drift")
    for marker in [
        "#define RATIO4 4u",
        "#define RATIO3 3u",
        "reference_update(",
        "ds4_gpu_compressor_update_tensor(",
        "ratio4_no_emit_store_only_matches",
        "n_rot_zero_partial_failure_matches",
        "checked_overflow_rejected",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("emitted row" in value and "overflow" in value for value in risks), "overflow boundary risk missing")
    report.check(any("non-emission" in value and "duplication" in value for value in risks), "ordering risk missing")
    report.check(any("zero-rotation" in value for value in risks), "zero-RoPE timing risk missing")
    report.check(any("arbitrary general ratios" in value for value in risks), "general ratio risk missing")
    report.check(any("route promotion" in value for value in risks), "remaining-compute risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-compressor-update-smoke.json"
    checker = "check_cuda_abi_compressor_update_smoke.py"
    item = f"{MILESTONE}: Public Compressor Update ABI"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active remainder status missing",
    )
    report.check(
        fixture.get("review", {}).get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S",
        "pre-implementation review evidence missing",
    )
    report.check(
        fixture.get("review", {}).get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S",
        "final review timeout evidence missing",
    )
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("no-emit state-only behavior failure", lambda value: value["b300_execution"]["observed"].update({"ratio4_no_emit_store_only_matches": False})),
        ("zero-RoPE timing failure", lambda value: value["b300_execution"]["observed"].update({"n_rot_zero_partial_failure_matches": False})),
        ("general prefill overclaim", lambda value: value["ownership"].update({"owns_general_prefill": True})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
