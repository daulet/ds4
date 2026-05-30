#!/usr/bin/env python3
"""Validate the M14.3d1 Rust CUDA Q8 conversion kernel smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.3d1/q8-conversion-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/q8_conversion_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

DEPENDENCY_REVISION = "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide dequant_q8_0_to_f16_kernel launch proof",
    "executable-local cuda-oxide dequant_q8_0_to_f32_kernel launch proof",
    "executable-local cuda-oxide quantize_q8_0_f32_kernel launch proof",
]
EXPECTED_NOT_CLAIMED = [
    "Q8 quantized matmul kernels or their dispatch policy",
    "runtime graph integration, default CUDA route, or C CUDA removal",
]


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
    args = parse_args(argv)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "cargo": CARGO.read_text(encoding="utf-8"),
        "lock": LOCK.read_text(encoding="utf-8"),
        "lib": CRATE_LIB.read_text(encoding="utf-8"),
        "smoke": SMOKE.read_text(encoding="utf-8"),
        "cuda": CUDA_SOURCE.read_text(encoding="utf-8"),
        "roadmap": ROADMAP.read_text(encoding="utf-8"),
        "todo": TODO.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
        "readme": README.read_text(encoding="utf-8"),
        "report": REPORT.read_text(encoding="utf-8"),
    }
    report = Report()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.q8_conversion_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.3d1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{DEPENDENCY_REVISION}"' in texts["cargo"], "current crate dependency revision pin missing")
    report.check(f"#{DEPENDENCY_REVISION}" in texts["lock"], "current lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-q8-conversion-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void dequant_q8_0_to_f16_kernel",
        "__global__ static void dequant_q8_0_to_f32_kernel",
        "__global__ static void quantize_q8_0_f32_kernel",
        "__hmul",
        "__half2float",
        "lrintf",
        "matmul_q8_0_preq_kernel",
        "cuda_matmul_q8_0_tensor_labeled",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_dequant_q8_0_to_f16_kernel", True),
        ("owns_dequant_q8_0_to_f32_kernel", True),
        ("owns_quantize_q8_0_f32_kernel", True),
        ("owns_quantized_matmul_kernels", False),
        ("owns_q8_matmul_dispatch_policy", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_3D1_SCOPE",
        "owns_dequant_q8_0_to_f16_kernel: true",
        "owns_dequant_q8_0_to_f32_kernel: true",
        "owns_quantize_q8_0_f32_kernel: true",
        "owns_quantized_matmul_kernels: false",
        "owns_q8_matmul_dispatch_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 46, "feature test count drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-q8-conversion-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a device target")
    report.check("CUDA_OXIDE_LINK_TARGET" not in command, "smoke command forces a link target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.3d1",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "packed_q8_f16_dequant_matches": True,
        "packed_q8_f32_dequant_matches": True,
        "activation_quantization_matches": True,
        "ties_to_even_matches_lrintf": True,
        "partial_block_padding_matches": True,
        "invalid_shape_rejected": True,
        "uses_libdevice_link_path": True,
        "owns_dequant_q8_0_to_f16_kernel": True,
        "owns_dequant_q8_0_to_f32_kernel": True,
        "owns_quantize_q8_0_f32_kernel": True,
        "owns_quantized_matmul_kernels": False,
        "owns_q8_matmul_dispatch_policy": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 Q8 conversion result drift")
    for marker in [
        "#![feature(f16)]",
        "pub fn dequant_q8_0_to_f16_kernel",
        "pub fn dequant_q8_0_to_f32_kernel",
        "pub fn quantize_q8_0_f32_kernel",
        "weights[base] as u16 | ((weights[base + 1] as u16) << 8)",
        "SharedArray<f32, 32>",
        "let lower = scaled.floor();",
        "fraction == 0.5 && (rounded & 1) != 0",
        "ltoir::load_kernel_module",
        "Err(Q8ConversionError::InvalidShape)",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.3d1/q8-conversion-kernel-smoke.json"
    checker = "check_q8_conversion_kernel_smoke.py"
    item = "M14.3d1: Q8 Dequantization And Activation Quantization Kernels"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.3d2" in texts["status"], "next active M14.3 stage missing")
    report.check("M14.3d1 Q8 Dequantization And Activation Quantization Kernels" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("F16 dequant result absent", lambda value: value["b300_execution"]["stdout"].update({"packed_q8_f16_dequant_matches": False})),
        ("F32 dequant result absent", lambda value: value["b300_execution"]["stdout"].update({"packed_q8_f32_dequant_matches": False})),
        ("quantize result absent", lambda value: value["b300_execution"]["stdout"].update({"activation_quantization_matches": False})),
        ("rounding result absent", lambda value: value["b300_execution"]["stdout"].update({"ties_to_even_matches_lrintf": False})),
        ("padding result absent", lambda value: value["b300_execution"]["stdout"].update({"partial_block_padding_matches": False})),
        ("matmul overclaim", lambda value: value["ownership"].update({"owns_quantized_matmul_kernels": True})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.3d1 Q8 conversion kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
