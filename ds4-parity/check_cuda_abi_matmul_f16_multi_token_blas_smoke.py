#!/usr/bin/env python3
"""Validate the Rust CUDA public multi-token F16 BLAS projection ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-matmul-f16-multi-token-blas-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_matmul_f16_multi_token_blas_link_smoke.c"
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
    print(f"{MILESTONE} Rust CUDA public multi-token F16 BLAS projection ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_matmul_f16_multi_token_blas_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-multi-token-f16-blas-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols") == ["ds4_gpu_matmul_f16_tensor", "ds4_gpu_matmul_f16_pair_tensor"],
        "oracle symbol drift",
    )
    for marker in [
        'extern "C" int ds4_gpu_matmul_f16_tensor',
        "DS4_CUDA_SERIAL_F16_MATMUL",
        "g_cublas_ready && n_tok > 1",
        "f32_to_f16_kernel<<<",
        "cublasGemmEx(g_cublas",
        'extern "C" int ds4_gpu_matmul_f16_pair_tensor',
        "return ds4_gpu_matmul_f16_tensor(out0",
        "CUBLAS_TF32_TENSOR_OP_MATH",
        "DS4_CUDA_NO_TF32",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C F16 BLAS oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 32),
        ("exported_compute_symbol_count", 12),
        ("public_gpu_abi_function_count", 81),
        ("consumes_single_token_f16_projection", True),
        ("consumes_paired_single_token_f16_projection", True),
        ("owns_matmul_f16_multi_token_blas_tensor", True),
        ("owns_paired_multi_token_blas_delegation", True),
        ("owns_reusable_f32_to_f16_activation_scratch", True),
        ("owns_default_f16_blas_math_selection", True),
        ("owns_quality_mode_mutation", False),
        ("owns_q8_f16_cache_hook", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 58, "Rust ABI export implementation count drift")
    report.check("ds4_gpu_matmul_f16_tensor" in symbols, "F16 public export missing")
    report.check("ds4_gpu_matmul_f16_pair_tensor" in symbols, "paired F16 public export missing")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "static ABI_F16_ACTIVATIONS: Mutex<Option<DeviceBuffer<f16>>>",
        "fn with_abi_f16_activations",
        "backend.synchronize().ok()?",
        "cuda_core::memory::malloc_sync(bytes)",
        "pub unsafe extern \"C\" fn ds4_gpu_matmul_f16_tensor",
        "n_tok == 0",
        "blas_ready: n_tok > 1",
        "backend.blas_handle()",
        "apply_abi_blas_math(&blas)",
        "BlasMathMode::Tf32TensorOp",
        "with_cached_abi_model_range(",
        "DeviceBuffer::<f16>::from_raw_parts(",
        "kernels.f32_to_f16_tensor(",
        "blas.project_f16_f32(",
        "pub unsafe extern \"C\" fn ds4_gpu_matmul_f16_pair_tensor",
        "F16PairProjectionPath::TwoIndependent",
        "ds4_gpu_matmul_f16_tensor(",
    ]:
        report.check(marker in texts["abi"], f"Rust F16 BLAS ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_f32_to_f16_kernel",
        "f32_to_f16_kernel: CudaFunction",
        'load_function("abi_f32_to_f16_kernel")',
        "fn f32_to_f16_tensor(",
        "pub fn abi_matmul_f16_kernel",
        "pub fn abi_matmul_f16_pair_ordered_chunks_kernel",
    ]:
        report.check(marker in texts["kernels"], f"embedded F16 BLAS kernel marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiF16BlasProjectionScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBA_SCOPE",
        "owns_matmul_f16_multi_token_blas_tensor: true",
        "owns_paired_multi_token_blas_delegation: true",
        "owns_reusable_f32_to_f16_activation_scratch: true",
        "owns_default_f16_blas_math_selection: true",
        "owns_quality_mode_mutation: false",
        "owns_q8_f16_cache_hook: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("with_cached_abi_model_range" in implementation.get("weight_range_path", ""), "weight range path missing")
    report.check("with_abi_f16_activations" in implementation.get("scratch_lifetime_path", ""), "scratch lifetime path missing")
    report.check("project_f16_f32" in implementation.get("owned_dispatch_boundary", ""), "owned dispatch boundary missing")
    report.check("Q8/F16" in implementation.get("remaining_compute_boundary", ""), "remaining Q8/F16 boundary missing")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "artifact retention missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 124),
        ("feature_release_test_count", 131),
        ("staticlib_export_count", 32),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "single_token_predecessor_matches",
        "single_token_pair_predecessor_matches",
        "multi_token_f16_blas_output_matches",
        "paired_multi_token_f16_blas_delegation_matches",
        "cached_f16_weights_survive_blas_after_host_mutation",
        "f32_to_f16_activation_rounding_observed",
        "serial_multi_token_f32_activation_fallback_matches",
        "zero_token_rejected",
        "invalid_second_model_range_rejected",
        "null_model_rejected",
        "cuda_oxide_blas_adapter_and_conversion_kernel_loaded",
        "historical_single_token_f16_negative_witnesses_superseded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 15, "predecessor regression count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 15, "predecessor warning count drift")
    for marker in [
        "ds4_gpu_matmul_f16_tensor(",
        "ds4_gpu_matmul_f16_pair_tensor(",
        "x_full[IN_DIM + i] = 1.0003f",
        "x_blas[IN_DIM + i] = 1.0f",
        "model[i] = 0",
        'setenv("DS4_CUDA_SERIAL_F16_MATMUL", "1", 1)',
        "multi_token_f16_blas_output_matches",
        "paired_multi_token_f16_blas_delegation_matches",
        "f32_to_f16_activation_rounding_observed",
        "zero_token_rejected",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("historical" in value for value in risks), "superseded-witness note missing")
    report.check(any("q8" in value.lower() for value in risks), "q8 boundary missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-matmul-f16-multi-token-blas-smoke.json"
    checker = "check_cuda_abi_matmul_f16_multi_token_blas_smoke.py"
    item = f"{MILESTONE}: Public Multi-Token F16 BLAS Projection ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(item in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("BLAS output removed", lambda value: value["b300_execution"]["observed"].update({"multi_token_f16_blas_output_matches": False})),
        ("paired delegation removed", lambda value: value["b300_execution"]["observed"].update({"paired_multi_token_f16_blas_delegation_matches": False})),
        ("conversion rounding removed", lambda value: value["b300_execution"]["observed"].update({"f32_to_f16_activation_rounding_observed": False})),
        ("quality overclaim", lambda value: value["ownership"].update({"owns_quality_mode_mutation": True})),
        ("q8 overclaim", lambda value: value["ownership"].update({"owns_q8_f16_cache_hook": True})),
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
