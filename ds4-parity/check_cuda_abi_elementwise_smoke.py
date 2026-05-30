#!/usr/bin/env python3
"""Validate the M14.6b2b1 Rust CUDA embedded elementwise ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b1/abi-elementwise-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b1/abi_elementwise_link_smoke.c"
GPU_BUILD = ROOT / "rust/ds4-gpu/build.rs"
GPU_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_SYMBOLS = [
    "ds4_gpu_add_tensor",
    "ds4_gpu_begin_commands",
    "ds4_gpu_cleanup",
    "ds4_gpu_end_commands",
    "ds4_gpu_flush_commands",
    "ds4_gpu_init",
    "ds4_gpu_repeat_hc_tensor",
    "ds4_gpu_should_use_managed_kv_cache",
    "ds4_gpu_synchronize",
    "ds4_gpu_tensor_alloc",
    "ds4_gpu_tensor_alloc_managed",
    "ds4_gpu_tensor_bytes",
    "ds4_gpu_tensor_contents",
    "ds4_gpu_tensor_copy",
    "ds4_gpu_tensor_fill_f32",
    "ds4_gpu_tensor_free",
    "ds4_gpu_tensor_read",
    "ds4_gpu_tensor_view",
    "ds4_gpu_tensor_write",
]


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
    args = parse_args(argv)
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
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.6b2b1 Rust CUDA embedded elementwise ABI smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_elementwise_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b1", "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-embedded-elementwise-abi",
        "status drift",
    )
    report.check(fixture.get("exported_symbols") == EXPECTED_SYMBOLS, "exported symbol drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["ds4_gpu_add_tensor", "ds4_gpu_repeat_hc_tensor"], "oracle symbol drift")
    for marker in [
        'extern "C" int ds4_gpu_add_tensor',
        "add_kernel<<<(n + 255) / 256, 256>>>",
        'extern "C" int ds4_gpu_repeat_hc_tensor',
        "repeat_hc_kernel<<<(n + 255) / 256, 256>>>",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 19),
        ("exported_compute_symbol_count", 3),
        ("public_gpu_abi_function_count", 81),
        ("owns_tensor_fill_f32", True),
        ("owns_add_tensor", True),
        ("owns_repeat_hc_tensor", True),
        ("uses_embedded_rust_kernel_module", True),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = sorted(set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"])))
    report.check(symbols == EXPECTED_SYMBOLS, "Rust ABI symbol implementation drift")
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(set(EXPECTED_SYMBOLS) <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "pub unsafe extern \"C\" fn ds4_gpu_add_tensor",
        "pub unsafe extern \"C\" fn ds4_gpu_repeat_hc_tensor",
        "with_abi_kernels(backend",
        "preserves current-C support for input/output aliasing",
    ]:
        report.check(marker in texts["abi"], f"Rust ABI implementation marker missing: {marker}")
    for marker in [
        'const ABI_KERNEL_ARTIFACT: &str = "ds4-cuda";',
        "#[cuda_module]",
        "pub fn abi_add_kernel",
        "pub fn abi_repeat_hc_kernel",
        "cuda_core::launch_kernel_on_stream",
    ]:
        report.check(marker in texts["kernels"], f"embedded kernel marker missing: {marker}")
    report.check("DeviceBuffer" not in texts["kernels"], "FFI kernel path reintroduced transient typed ownership")
    for marker in [
        "pub const M14_6B2B1_SCOPE",
        "exported_abi_symbol_count: 19",
        "exported_compute_symbol_count: 3",
        "owns_add_tensor: true",
        "owns_repeat_hc_tensor: true",
        "uses_embedded_rust_kernel_module: true",
        "owns_remaining_graph_compute_abi: false",
        "owns_complete_ds4_gpu_abi: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("artifact_name") == "ds4-cuda", "artifact module name drift")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "artifact retention requirement missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 89),
        ("feature_test_count", 91),
        ("generated_artifact_target", "sm_80"),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    report.check("--features cuda-oxide-kernels" in execution.get("staticlib_build_command", ""), "staticlib build command drift")
    report.check("--whole-archive" in execution.get("c_link_command", ""), "C link command retention drift")
    report.check("-lcuda" in execution.get("c_link_command", ""), "C link command CUDA driver drift")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "embedded_rust_kernel_module_loaded",
        "add_output_matches",
        "add_alias_output_matches",
        "repeat_hc_output_matches",
        "invalid_shape_rejected",
        "null_rejected",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "ds4_gpu_add_tensor(sum, a, b, 4)",
        "ds4_gpu_add_tensor(a, a, b, 4)",
        "ds4_gpu_repeat_hc_tensor(repeated, row, 3, 3)",
        "ds4_gpu_add_tensor(NULL, a, b, 4)",
        "embedded_rust_kernel_module_loaded",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("retaining symbol" in value for value in risks), "retention risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b1/abi-elementwise-smoke.json"
    checker = "check_cuda_abi_elementwise_smoke.py"
    item = "M14.6b2b1: Rust CUDA Elementwise ABI Module"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.6b2b2 Remaining Rust CUDA Kernel ABI Assembly" in texts["status"], "active item missing")
    report.check("M14.6b2b1 Rust CUDA Elementwise ABI Module" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2b2 Remaining Rust CUDA Kernel ABI Assembly",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("missing add export", lambda value: value["exported_symbols"].remove("ds4_gpu_add_tensor")),
        ("remaining compute overclaim", lambda value: value["ownership"].update({"owns_remaining_graph_compute_abi": True})),
        ("route promotion overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("alias failure", lambda value: value["b300_execution"]["observed"].update({"add_alias_output_matches": False})),
        ("artifact load failure", lambda value: value["b300_execution"]["observed"].update({"embedded_rust_kernel_module_loaded": False})),
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
