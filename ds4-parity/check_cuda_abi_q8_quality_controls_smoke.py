#!/usr/bin/env python3
"""Validate the Rust CUDA public Q8 preload and quality-controls ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-q8-quality-controls-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_q8_quality_controls_link_smoke.c"
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
    print(f"{MILESTONE} Rust CUDA public Q8 preload and quality-controls ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_q8_quality_controls_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-q8-quality-controls-abi",
        "status drift",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == ["ds4_gpu_cache_q8_f16_range", "ds4_gpu_print_memory_report", "ds4_gpu_set_quality"],
        "oracle symbol drift",
    )
    for marker in [
        'extern "C" int ds4_gpu_cache_q8_f16_range',
        "cuda_q8_f16_preload_allowed",
        "cuda_q8_f32_ptr",
        "cuda_q8_f16_ptr",
        'extern "C" void ds4_gpu_print_memory_report',
        'extern "C" void ds4_gpu_set_quality',
        "g_quality_mode = quality ? 1 : 0",
        "CUBLAS_TF32_TENSOR_OP_MATH",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C controls oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 35),
        ("exported_compute_symbol_count", 12),
        ("public_gpu_abi_function_count", 81),
        ("consumes_multi_token_dense_blas_projection", True),
        ("owns_cache_q8_f16_range", True),
        ("owns_q8_f16_converted_buffers", True),
        ("owns_q8_f32_optional_preload", True),
        ("owns_quality_mode_mutation", True),
        ("owns_memory_report", True),
        ("owns_q8_matmul_compute_abi", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) >= 74, "published Rust ABI exports disappeared")
    for symbol in ["ds4_gpu_cache_q8_f16_range", "ds4_gpu_print_memory_report", "ds4_gpu_set_quality"]:
        report.check(symbol in symbols, f"public control export missing: {symbol}")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "static ABI_Q8_CACHE: Mutex<AbiQ8Cache>",
        "static ABI_QUALITY_MODE: AtomicBool",
        "static ABI_DEFAULT_BLAS_MATH: AtomicBool",
        "fn abi_q8_cache_options()",
        "fn cache_abi_q8_f16_range(",
        "fn cache_abi_q8_f32_range(",
        "q8_preload_format(",
        "clear_abi_q8_converted_ranges",
        "pub unsafe extern \"C\" fn ds4_gpu_cache_q8_f16_range",
        "pub extern \"C\" fn ds4_gpu_print_memory_report",
        "pub extern \"C\" fn ds4_gpu_set_quality",
        "update_abi_blas_math_state();",
        "apply_abi_blas_math(&blas)",
    ]:
        report.check(marker in texts["abi"], f"Rust public controls marker missing: {marker}")
    for marker in [
        "pub fn abi_dequant_q8_0_to_f16_kernel",
        "pub fn abi_dequant_q8_0_to_f32_kernel",
        "dequant_q8_0_to_f16_kernel: CudaFunction",
        "dequant_q8_0_to_f32_kernel: CudaFunction",
        'load_function("abi_dequant_q8_0_to_f16_kernel")',
        'load_function("abi_dequant_q8_0_to_f32_kernel")',
        "fn dequant_q8_f16_tensor(",
        "fn dequant_q8_f32_tensor(",
    ]:
        report.check(marker in texts["kernels"], f"embedded Q8 conversion marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiQ8QualityControlsScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 35",
        "owns_cache_q8_f16_range: true",
        "owns_q8_f16_converted_buffers: true",
        "owns_q8_f32_optional_preload: true",
        "owns_quality_mode_mutation: true",
        "owns_memory_report: true",
        "owns_q8_matmul_compute_abi: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("ABI_Q8_CACHE" in implementation.get("converted_cache_path", ""), "converted-cache path missing")
    report.check("abi_dequant_q8_0_to_f16_kernel" in implementation.get("kernel_path", ""), "kernel path missing")
    report.check("ABI_QUALITY_MODE" in implementation.get("quality_path", ""), "quality path missing")
    report.check("Q8 matmul" in implementation.get("remaining_compute_boundary", ""), "remaining Q8 compute boundary missing")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "artifact retention missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 125),
        ("feature_release_test_count", 132),
        ("staticlib_export_count", 35),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "q8_f16_preload_allocation_observed",
        "q8_f16_exact_cache_reuse_observed",
        "quality_suppresses_new_q8_f16_preload",
        "quality_disable_reenables_q8_f16_preload",
        "q8_f32_optional_preload_allocation_observed",
        "memory_report_callable",
        "quality_math_mutation_changes_f32_blas_output",
        "no_tf32_quality_update_uses_default_math",
        "invalid_range_rejected",
        "null_optional_cache_accepted",
        "embedded_q8_dequant_kernels_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 16, "predecessor regression count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 16, "predecessor warning count drift")
    for marker in [
        "ds4_gpu_cache_q8_f16_range(",
        "ds4_gpu_print_memory_report(",
        "ds4_gpu_set_quality(true)",
        "ds4_gpu_set_quality(false)",
        "DS4_CUDA_Q8_F32_PRELOAD",
        "quality_math_mutation_changes_f32_blas_output",
        "no_tf32_quality_update_uses_default_math",
        "embedded_q8_dequant_kernels_loaded",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("Q8 matmul" in value for value in risks), "Q8 matmul boundary missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-q8-quality-controls-smoke.json"
    checker = "check_cuda_abi_q8_quality_controls_smoke.py"
    item = f"{MILESTONE}: Public Q8 Cache And Quality Controls ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(item in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Runtime Route Promotion And C CUDA Removal Policy"
        in texts["status"],
        "active remainder status missing",
    )
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("F16 allocation removed", lambda value: value["b300_execution"]["observed"].update({"q8_f16_preload_allocation_observed": False})),
        ("quality suppression removed", lambda value: value["b300_execution"]["observed"].update({"quality_suppresses_new_q8_f16_preload": False})),
        ("quality mutation removed", lambda value: value["ownership"].update({"owns_quality_mode_mutation": False})),
        ("Q8 matmul overclaim", lambda value: value["ownership"].update({"owns_q8_matmul_compute_abi": True})),
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
