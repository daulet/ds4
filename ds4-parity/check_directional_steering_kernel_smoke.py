#!/usr/bin/env python3
"""Validate the M14.2b1 Rust CUDA directional-steering kernel smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.2b1/directional-steering-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/directional_steering_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_DEPENDENCY_REVISION = "aabe10dc4fa0086375104458909e222d1ac1cfe3"
CURRENT_DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
TOOL_REVISION = "981e3244a107d84d807cfb087793269c477cc764"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide directional_steering_project_kernel launch proof",
    "current-C-shaped in-place row projection with SharedArray reduction and synchronization",
    "current-C-shaped directional projection shape validation",
]
EXPECTED_NOT_CLAIMED = [
    "swiglu_kernel or ds4_gpu_swiglu_tensor",
    "embedding and model-range kernels",
    "indexer and top-k kernels",
    "runtime graph integration or default CUDA route",
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
    report.check(fixture.get("schema") == "ds4.directional_steering_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.2b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == RECORDED_DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("tool_revision") == TOOL_REVISION, "cargo-oxide tool revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "current crate dependency revision pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "current lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-directional-steering-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_blocker(report, fixture)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void directional_steering_project_kernel",
        "__shared__ float partial[256]",
        "__syncthreads();",
        "ds4_gpu_directional_steering_project_tensor",
        "directional_steering_project_kernel<<<rows, nth>>>",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_directional_steering_project_tensor", True),
        ("owns_swiglu_tensor", False),
        ("owns_embedding_kernels", False),
        ("owns_indexer_kernels", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_2B1_SCOPE",
        "owns_directional_steering_project_tensor: true",
        "owns_swiglu_tensor: false",
        "owns_embedding_kernels: false",
        "owns_indexer_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-directional-steering-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a device target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.2b1",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "directional_projection_matches": True,
        "directional_shape_rejected": True,
        "owns_directional_steering_project_tensor": True,
        "owns_swiglu_tensor": False,
        "owns_embedding_kernels": False,
        "owns_indexer_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 directional steering result drift")
    for marker in [
        "#[cuda_module]",
        "pub fn directional_steering_project_kernel",
        "SharedArray<f32, 256>",
        "thread::sync_threads()",
        "directional_steering_project_tensor(",
        "Err(DirectionalSteeringError::InvalidProjectionShape)",
        "M14_2B1_SCOPE.owns_swiglu_tensor",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_blocker(report: Report, fixture: dict[str, Any]) -> None:
    blocker = require_dict(report, fixture.get("swiglu_blocker"), "swiglu_blocker")
    report.check(blocker.get("stage") == "M14.2b2", "SwiGLU follow-up stage drift")
    report.check("f32::exp()" in blocker.get("attempted_operation", ""), "SwiGLU attempted operation missing")
    report.check("__nv_expf" in blocker.get("blocking_boundary", ""), "libdevice boundary missing")
    report.check("opaque-pointer" in blocker.get("blocking_boundary", ""), "NVVM pointer boundary missing")
    report.check("parse expected type" in blocker.get("blocking_error", ""), "NVVM error evidence missing")
    report.check("before claiming SwiGLU ownership" in blocker.get("required_resolution", ""), "SwiGLU gate missing")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.2b1/directional-steering-kernel-smoke.json"
    checker = "check_directional_steering_kernel_smoke.py"
    report.check("M14.2b1: Directional Steering Projection Kernel" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.2b1: Directional Steering Projection Kernel" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14." in texts["status"], "next active stage missing")
    report.check("M14.2b1 Directional Steering Projection Kernel" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        (
            "directional result absent",
            lambda value: value["b300_execution"]["stdout"].update({"directional_projection_matches": False}),
        ),
        ("SwiGLU overclaim", lambda value: value["ownership"].update({"owns_swiglu_tensor": True})),
        ("blocker omitted", lambda value: value.pop("swiglu_blocker")),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("portable target lost", lambda value: value["b300_execution"].update({"backend_selected_target": "sm_103"})),
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
    print(f"M14.2b1 directional steering kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
