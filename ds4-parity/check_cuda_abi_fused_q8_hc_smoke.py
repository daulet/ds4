#!/usr/bin/env python3
"""Validate the Rust CUDA public fused Q8 hyperconnection ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-fused-q8-hc-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_fused_q8_hc_link_smoke.c"
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
    print(f"{MILESTONE} Rust CUDA public fused Q8 HC ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_fused_q8_hc_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-fused-q8-hc-abi",
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
        == [
            "ds4_gpu_matmul_q8_0_hc_expand_tensor",
            "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        ],
        "oracle symbols drift",
    )
    for marker in [
        "matmul_q8_0_hc_expand_preq_warp8_kernel",
        "cuda_matmul_q8_0_hc_expand_tensor_labeled",
        'extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor',
        'extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor',
        "DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED",
        "cuda_q8_use_dp4a()",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C fused Q8 HC marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 41),
        ("exported_compute_symbol_count", 18),
        ("public_gpu_abi_function_count", 81),
        ("consumes_q8_matmul_and_hc_expand_abi", True),
        ("owns_matmul_q8_0_hc_expand_tensor", True),
        ("owns_shared_down_hc_expand_q8_0_tensor", True),
        ("owns_fused_q8_hc_kernel", True),
        ("owns_disabled_fusion_fallback", True),
        ("owns_dp4a_and_scalar_selection", True),
        ("owns_internal_pair_consumer", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) >= 74, "published Rust ABI exports disappeared")
    for symbol in [
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
    ]:
        report.check(symbol in symbols, f"fused Q8 HC public export missing: {symbol}")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "unsafe fn matmul_q8_hc_expand_fused_impl(",
        "pub unsafe extern \"C\" fn ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "pub unsafe extern \"C\" fn ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED",
        "ds4_gpu_hc_expand_split_tensor(",
        "ds4_gpu_hc_expand_add_split_tensor(",
    ]:
        report.check(marker in texts["abi"], f"Rust fused Q8 HC ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_matmul_q8_0_hc_expand_preq_warp8_kernel",
        "matmul_q8_0_hc_expand_preq_warp8_kernel: CudaFunction",
        '.load_function("abi_matmul_q8_0_hc_expand_preq_warp8_kernel")',
        "fn matmul_q8_hc_expand_tensor(",
        "q8_dot(",
    ]:
        report.check(marker in texts["kernels"], f"embedded fused Q8 HC marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiFusedQ8HcScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 41",
        "exported_compute_symbol_count: 18",
        "owns_fused_q8_hc_kernel: true",
        "owns_internal_pair_consumer: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(
        implementation.get("kernel_entry") == "abi_matmul_q8_0_hc_expand_preq_warp8_kernel",
        "kernel entry drift",
    )
    report.check("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED" in implementation.get("fallback_path", ""), "fallback path missing")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage path missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 128),
        ("feature_release_test_count", 135),
        ("staticlib_export_count", 41),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "fused_plain_dp4a_output_matches",
        "fused_plain_scalar_output_matches",
        "fallback_plain_output_matches",
        "fused_shared_add_output_matches",
        "fallback_shared_add_output_matches",
        "fused_shared_alias_add_output_matches",
        "invalid_shape_rejected",
        "embedded_fused_q8_hc_kernel_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 38, "predecessor count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 38, "warning count drift")
    for marker in [
        "ds4_gpu_matmul_q8_0_hc_expand_tensor(",
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor(",
        'setenv("DS4_CUDA_NO_Q8_DP4A", "1", 1)',
        'setenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED", "1", 1)',
        "fused_shared_alias_add_output_matches",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("internal Q8 pair" in value for value in risks), "internal pair boundary missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-fused-q8-hc-smoke.json"
    checker = "check_cuda_abi_fused_q8_hc_smoke.py"
    item = f"{MILESTONE}: Public Fused Q8 Hyperconnection Consumers ABI"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Default-Route Promotion And C CUDA Removal Execution"
        in texts["status"],
        "active remainder status missing",
    )
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("DP4A result removed", lambda value: value["b300_execution"]["observed"].update({"fused_plain_dp4a_output_matches": False})),
        ("fallback result removed", lambda value: value["b300_execution"]["observed"].update({"fallback_plain_output_matches": False})),
        ("alias result removed", lambda value: value["b300_execution"]["observed"].update({"fused_shared_alias_add_output_matches": False})),
        ("pair overclaim", lambda value: value["ownership"].update({"owns_internal_pair_consumer": True})),
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
