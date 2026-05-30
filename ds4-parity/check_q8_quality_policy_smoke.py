#!/usr/bin/env python3
"""Validate the M14.1b3b Rust Q8 admission and quality-mode policy smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b3b/q8-quality-policy-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
POLICY = ROOT / "rust/ds4-cuda/src/q8_policy.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/q8_quality_policy_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_REVISION = "aabe10dc4fa0086375104458909e222d1ac1cfe3"
CURRENT_REVISION = "d4791b7002152af3b7f6b15a48d7f5acd7a63011"
EXPECTED_RUST_OWNED = [
    "cuda-oxide typed cuBLAS math-mode selection",
    "Q8/F16 cache eligibility, preload, and budget policy",
    "Q8/F16 disable-after-failure state policy",
    "Q8/F32 optional preload selection policy",
]
EXPECTED_NOT_CLAIMED = [
    "Q8 converted device-buffer allocation or cached pointer reuse",
    "Q8 converted device-buffer synchronization and release on failure",
    "dequant Q8 conversion kernels",
    "DS4 compute kernels",
    "runtime graph or default CUDA route",
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
        "policy": POLICY.read_text(encoding="utf-8"),
        "substrate": SUBSTRATE.read_text(encoding="utf-8"),
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
    report.check(fixture.get("schema") == "ds4.q8_quality_policy_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b3b", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == RECORDED_REVISION, "cuda-oxide revision drift")
    report.check(
        oxide.get("new_api") == "Blas::set_math_mode(BlasMathMode)",
        "cuBLAS math-mode API drift",
    )
    report.check(f'rev = "{CURRENT_REVISION}"' in texts["cargo"], "current crate revision pin missing")
    report.check(f"#{CURRENT_REVISION}" in texts["lock"], "current lockfile revision pin missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "cuda_q8_f16_cache_reserve_bytes",
        "cuda_q8_f16_cache_allowed",
        "cuda_q8_f16_preload_allowed",
        "cuda_q8_f16_cache_has_budget",
        "cuda_q8_f16_cache_disable_after_failure",
        "cuda_q8_f32_cache_allowed",
        "ds4_gpu_set_quality",
        "CUBLAS_TF32_TENSOR_OP_MATH",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_q8_cache_admission_policy", True),
        ("owns_q8_cache_failure_disable_policy", True),
        ("owns_quality_blas_selection", True),
        ("owns_converted_q8_buffers", False),
        ("owns_dequant_kernels", False),
        ("owns_ds4_kernels", False),
        ("changes_default_route", False),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_1B3B_SCOPE",
        "owns_q8_cache_admission_policy: true",
        "owns_q8_cache_failure_disable_policy: true",
        "owns_quality_blas_selection: true",
        "owns_converted_q8_buffers: false",
        "owns_dequant_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    for marker in [
        "pub fn quality_blas_math_policy",
        "pub fn apply_quality_blas_policy",
        "pub fn q8_f16_cache_reserve_bytes",
        "pub fn q8_f16_cache_allowed",
        "pub fn q8_f16_preload_allowed",
        "pub fn q8_f32_cache_allowed",
        "pub fn q8_preload_format",
        "pub fn admit_f16_bytes",
        "pub fn disable_f16_after_failure",
        "pub fn disable_optional_preload_after_failure",
    ]:
        report.check(marker in texts["policy"], f"Q8 policy marker missing: {marker}")
    report.check("pub fn blas_handle" in texts["substrate"], "BLAS handle wiring missing")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check("--test blas" not in execution.get("cuda_oxide_test_command", ""), "fork validation was reduced")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    report.check("--bin ds4-cuda-q8-quality-policy-smoke" in execution.get("command", ""), "smoke command missing")
    expected = {
        "milestone": "M14.1b3b",
        "device_name": "NVIDIA B300 SXM6 AC",
        "tf32_fast_mode_applied": True,
        "quality_default_math_applied": True,
        "no_tf32_default_math_applied": True,
        "f16_admission_policy": True,
        "attention_output_preload_suppression": True,
        "f16_budget_rejection": True,
        "f16_disable_after_failure": True,
        "f32_preload_policy": True,
        "optional_preload_disable_after_failure": True,
        "owns_q8_cache_admission_policy": True,
        "owns_q8_cache_failure_disable_policy": True,
        "owns_quality_blas_selection": True,
        "owns_converted_q8_buffers": False,
        "owns_dequant_kernels": False,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 Q8/quality-policy result drift")
    for marker in [
        "apply_quality_blas_policy",
        "BlasMathPolicy::Tf32TensorOp",
        "Q8F16AdmissionReason::BudgetExhausted",
        "disable_optional_preload_after_failure",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.1b3b/q8-quality-policy-smoke.json"
    checker = "check_q8_quality_policy_smoke.py"
    report.check("M14.1b3b: Q8 Cache And Quality Policy" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b3b: Q8 Cache And Quality Policy" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.1b4 Fill Kernel And Command Lifetime" in texts["status"]
        or "Active item: M14.1c Substrate Route Closure Gate" in texts["status"]
        or "Active item: M14.2" in texts["status"],
        "next active stage missing",
    )
    report.check("M14.1b3b Q8 Cache And Quality Policy" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("TF32 mode not exercised", lambda value: value["b300_execution"]["stdout"].update({"tf32_fast_mode_applied": False})),
        ("Q8 failure policy absent", lambda value: value["b300_execution"]["stdout"].update({"f16_disable_after_failure": False})),
        ("converted buffer overclaim", lambda value: value["ownership"].update({"owns_converted_q8_buffers": True})),
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
    print(f"M14.1b3b Q8/quality policy smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
