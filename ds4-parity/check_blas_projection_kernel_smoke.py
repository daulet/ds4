#!/usr/bin/env python3
"""Validate the M14.3c3 Rust CUDA BLAS projection and conversion smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.3c3/blas-projection-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/blas_projection_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_DEPENDENCY_REVISION = "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7"
CURRENT_DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide f32_to_f16_kernel launch proof",
    "cuda-core DS4-layout F16/F32 BLAS projection execution proof",
    "current-C dense F16/F32 and paired F16 dispatch policy selection",
]
EXPECTED_NOT_CLAIMED = [
    "Q8 conversion or quantized matmul kernels",
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
    report.check(fixture.get("schema") == "ds4.blas_projection_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.3c3", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == RECORDED_DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "current crate dependency revision pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "current lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-blas-projection-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void f32_to_f16_kernel",
        "ds4_gpu_matmul_f16_tensor",
        "ds4_gpu_matmul_f16_pair_tensor",
        "ds4_gpu_matmul_f32_tensor",
        "DS4_CUDA_SERIAL_F16_MATMUL",
        "DS4_CUDA_SERIAL_ROUTER",
        "DS4_CUDA_NO_ORDERED_F16_MATMUL",
        "DS4_CUDA_NO_F16_PAIR_MATMUL",
        "cublasGemmEx",
        "cublasSgemm",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_f32_to_f16_kernel", True),
        ("owns_f16_projection_dispatch_policy", True),
        ("owns_f16_pair_projection_dispatch_policy", True),
        ("owns_f32_projection_dispatch_policy", True),
        ("owns_live_f16_and_f32_blas_paths", True),
        ("owns_q8_conversion_or_matmul_kernels", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_3C3_SCOPE",
        "owns_f32_to_f16_kernel: true",
        "owns_f16_projection_dispatch_policy: true",
        "owns_f16_pair_projection_dispatch_policy: true",
        "owns_f32_projection_dispatch_policy: true",
        "owns_live_f16_and_f32_blas_paths: true",
        "owns_q8_conversion_or_matmul_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 45, "feature test count drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-blas-projection-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a device target")
    report.check("CUDA_OXIDE_LINK_TARGET" not in command, "smoke command forces a link target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.3c3",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "activation_conversion_matches": True,
        "mixed_precision_blas_output_matches": True,
        "f32_blas_output_matches": True,
        "dispatch_priority_matches": True,
        "pair_dispatch_matches": True,
        "invalid_shape_rejected": True,
        "owns_f32_to_f16_kernel": True,
        "owns_f16_projection_dispatch_policy": True,
        "owns_f16_pair_projection_dispatch_policy": True,
        "owns_f32_projection_dispatch_policy": True,
        "owns_live_f16_and_f32_blas_paths": True,
        "owns_q8_conversion_or_matmul_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 BLAS projection result drift")
    for marker in [
        "#![feature(f16)]",
        "pub fn f32_to_f16_kernel",
        "*element = x[offset] as f16",
        "ProjectionConfig::new",
        "blas.project_f16_f32(",
        "blas.project_f32(",
        "select_f16_projection_path(",
        "select_f16_pair_projection_path(",
        "select_f32_projection_path(",
        "Err(BlasProjectionError::InvalidShape)",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.3c3/blas-projection-kernel-smoke.json"
    checker = "check_blas_projection_kernel_smoke.py"
    item = "M14.3c3: F16 And F32 BLAS Dispatch And Activation Conversion"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "M14.3d1 Q8 Dequantization And Activation Quantization Kernels" in texts["status"],
        "successor M14.3 stage evidence missing",
    )
    report.check("M14.3c3 F16 And F32 BLAS Dispatch And Activation Conversion" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("conversion result absent", lambda value: value["b300_execution"]["stdout"].update({"activation_conversion_matches": False})),
        ("mixed BLAS result absent", lambda value: value["b300_execution"]["stdout"].update({"mixed_precision_blas_output_matches": False})),
        ("F32 BLAS result absent", lambda value: value["b300_execution"]["stdout"].update({"f32_blas_output_matches": False})),
        ("dispatch result absent", lambda value: value["b300_execution"]["stdout"].update({"dispatch_priority_matches": False})),
        ("pair result absent", lambda value: value["b300_execution"]["stdout"].update({"pair_dispatch_matches": False})),
        ("Q8 overclaim", lambda value: value["ownership"].update({"owns_q8_conversion_or_matmul_kernels": True})),
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
    print(f"M14.3c3 BLAS projection kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
