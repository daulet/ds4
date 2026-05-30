#!/usr/bin/env python3
"""Validate the M14.3b2 Rust CUDA head RMS normalization and RoPE-tail smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.3b2/head-rms-rope-tail-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/head_rms_rope_tail_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_DEPENDENCY_REVISION = "e9c0d677104751179985098f02212ff044d3ec22"
CURRENT_DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide head_rms_norm_rope_tail_kernel launch proof",
    "combined head RMS normalization and RoPE-tail tensor validation",
    "YARN interpolation and inverse rotation numeric validation",
]
EXPECTED_NOT_CLAIMED = [
    "standalone rope_tail_kernel and M14.4 RoPE or attention family",
    "environment-controlled fused-QKV fallback dispatch policy",
    "dense projection or Q8 conversion and matmul kernels",
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
    report.check(fixture.get("schema") == "ds4.head_rms_rope_tail_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.3b2", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == RECORDED_DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "current crate dependency revision pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "current lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-head-rms-rope-tail-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void head_rms_norm_rope_tail_kernel",
        "const uint32_t n_nope = head_dim - n_rot;",
        "rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor",
        "if (inverse) s = -s;",
        'extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor',
        "head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>",
        "__global__ static void rope_tail_kernel",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_head_rms_norm_rope_tail_kernel", True),
        ("owns_yarn_rotary_math_path", True),
        ("owns_standalone_rope_tail_kernel", False),
        ("owns_qkv_fused_dispatch_policy", False),
        ("owns_dense_projection_or_q8_kernels", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_3B2_SCOPE",
        "owns_head_rms_norm_rope_tail_kernel: true",
        "owns_yarn_rotary_math_path: true",
        "owns_standalone_rope_tail_kernel: false",
        "owns_qkv_fused_dispatch_policy: false",
        "owns_dense_projection_or_q8_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 41, "feature test count drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-head-rms-rope-tail-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a device target")
    report.check("CUDA_OXIDE_LINK_TARGET" not in command, "smoke command forces a link target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.3b2",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "interpolated_rope_output_matches": True,
        "yarn_forward_output_matches": True,
        "yarn_inverse_output_matches": True,
        "invalid_shape_rejected": True,
        "uses_libdevice_link_path": True,
        "owns_head_rms_norm_rope_tail_kernel": True,
        "owns_yarn_rotary_math_path": True,
        "owns_standalone_rope_tail_kernel": False,
        "owns_qkv_fused_dispatch_policy": False,
        "owns_dense_projection_or_q8_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 head RMS RoPE-tail result drift")
    for marker in [
        "pub fn head_rms_norm_rope_tail_kernel",
        "freq_base.powf(",
        "freq_base.ln()",
        ".floor()",
        ".ceil()",
        ".cos()",
        ".sin()",
        "grid_dim: (n_tok * n_head, 1, 1)",
        "n_rot & 1 != 0",
        "Err(HeadRmsRopeTailError::InvalidShape)",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.3b2/head-rms-rope-tail-kernel-smoke.json"
    checker = "check_head_rms_rope_tail_kernel_smoke.py"
    item = "M14.3b2: Head RMS Norm Rope Tail Kernel"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "M14.3c1 Base F16 And F32 Projection Kernels" in texts["status"],
        "successor M14.3 stage evidence missing",
    )
    report.check("M14.3b2 Head RMS Norm Rope Tail Kernel" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("interpolated result absent", lambda value: value["b300_execution"]["stdout"].update({"interpolated_rope_output_matches": False})),
        ("YARN result absent", lambda value: value["b300_execution"]["stdout"].update({"yarn_forward_output_matches": False})),
        ("inverse result absent", lambda value: value["b300_execution"]["stdout"].update({"yarn_inverse_output_matches": False})),
        ("standalone RoPE overclaim", lambda value: value["ownership"].update({"owns_standalone_rope_tail_kernel": True})),
        ("dispatch overclaim", lambda value: value["ownership"].update({"owns_qkv_fused_dispatch_policy": True})),
        ("projection overclaim", lambda value: value["ownership"].update({"owns_dense_projection_or_q8_kernels": True})),
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
    print(f"M14.3b2 head RMS RoPE-tail kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
