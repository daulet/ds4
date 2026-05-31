#!/usr/bin/env python3
"""Validate the M14.2b2 Rust CUDA SwiGLU kernel and libdevice-link smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.2b2/swiglu-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/swiglu_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_DEPENDENCY_REVISION = "d4791b7002152af3b7f6b15a48d7f5acd7a63011"
CURRENT_DEPENDENCY_REVISION = "1000e653df60a7814fa996d146e3823d0a364280"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide swiglu_kernel launch proof",
    "current-C-shaped clamp, SiLU exponential, weight, and bounds semantics",
    "portable PTX plus context-targeted libdevice cubin loading path",
]
EXPECTED_NOT_CLAIMED = [
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
    report.check(fixture.get("schema") == "ds4.swiglu_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.2b2", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == RECORDED_DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "crate dependency revision pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-swiglu-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_repair(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void swiglu_kernel",
        "g = fminf(g, clamp);",
        "u = fminf(fmaxf(u, -clamp), clamp);",
        "float s = g / (1.0f + expf(-g));",
        "ds4_gpu_swiglu_tensor",
        "swiglu_kernel<<<(n + 255) / 256, 256>>>",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_swiglu_tensor", True),
        ("owns_directional_steering_project_tensor", True),
        ("owns_embedding_kernels", False),
        ("owns_indexer_kernels", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_2B2_SCOPE",
        "owns_swiglu_tensor: true",
        "owns_directional_steering_project_tensor: true",
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
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-swiglu-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a compile target")
    report.check("CUDA_OXIDE_LINK_TARGET" not in command, "smoke command forces a link target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    report.check(execution.get("linked_cubin_target") == "sm_103", "B300 linked target drift")
    expected = {
        "milestone": "M14.2b2",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "swiglu_output_matches": True,
        "swiglu_unclamped_output_matches": True,
        "swiglu_shape_rejected": True,
        "uses_libdevice_link_path": True,
        "owns_swiglu_tensor": True,
        "owns_directional_steering_project_tensor": True,
        "owns_embedding_kernels": False,
        "owns_indexer_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 SwiGLU result drift")
    for marker in [
        "#[cuda_module]",
        "pub fn swiglu_kernel",
        "(g.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || g > clamp",
        "(u.to_bits() & 0x7fff_ffff) > 0x7f80_0000 || u < -clamp",
        "(-g).exp()",
        "ltoir::load_kernel_module",
        "../../ds4_cuda_swiglu_smoke",
        "swiglu_tensor(",
        "Err(SwigluError::InvalidShape)",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_repair(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    repair = require_dict(report, fixture.get("libdevice_repair"), "libdevice_repair")
    report.check(repair.get("cuda_oxide_revision") == RECORDED_DEPENDENCY_REVISION, "repair revision drift")
    report.check("__nv_expf" in repair.get("resolved_boundary", ""), "resolved exp boundary missing")
    report.check("portable PTX" in repair.get("resolved_boundary", ""), "portable PTX boundary missing")
    report.check("sm_103" in repair.get("resolved_boundary", ""), "linked target boundary missing")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.2b2/swiglu-kernel-smoke.json"
    checker = "check_swiglu_kernel_smoke.py"
    report.check("M14.2b2: SwiGLU Libdevice Path" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.2b2: SwiGLU Libdevice Path" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14." in texts["status"], "next active stage missing")
    report.check("M14.2b2 SwiGLU Libdevice Path" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("SwiGLU result absent", lambda value: value["b300_execution"]["stdout"].update({"swiglu_output_matches": False})),
        ("unclamped result absent", lambda value: value["b300_execution"]["stdout"].update({"swiglu_unclamped_output_matches": False})),
        ("libdevice path absent", lambda value: value["b300_execution"]["stdout"].update({"uses_libdevice_link_path": False})),
        ("embedding overclaim", lambda value: value["ownership"].update({"owns_embedding_kernels": True})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("linked target lost", lambda value: value["b300_execution"].update({"linked_cubin_target": "sm_80"})),
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
    print(f"M14.2b2 SwiGLU kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
