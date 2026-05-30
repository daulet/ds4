#!/usr/bin/env python3
"""Validate the M14.1b4 Rust fill-kernel and command-lifetime smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b4/fill-command-lifetime-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/fill_lifetime_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

DEPENDENCY_REVISION = "aabe10dc4fa0086375104458909e222d1ac1cfe3"
TOOL_REVISION = "981e3244a107d84d807cfb087793269c477cc764"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide fill_f32 kernel launch proof",
    "current-C-shaped fill count, zero-count, and bounds semantics",
    "no-op begin plus context-wide flush, end, and explicit synchronize completion wrappers",
]
EXPECTED_NOT_CLAIMED = [
    "library embedded-kernel artifact retention or runtime graph integration",
    "dequant Q8 conversion kernels",
    "DS4 graph compute kernels",
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
    report.check(fixture.get("schema") == "ds4.fill_command_lifetime_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b4", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == DEPENDENCY_REVISION, "cuda-oxide dependency revision drift")
    report.check(oxide.get("tool_revision") == TOOL_REVISION, "cargo-oxide tool revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(oxide.get("module_form") == "executable-local #[cuda_module]", "kernel module boundary drift")
    report.check(f'rev = "{DEPENDENCY_REVISION}"' in texts["cargo"], "crate dependency revision pin missing")
    report.check(f"#{DEPENDENCY_REVISION}" in texts["lock"], "lockfile dependency revision pin missing")
    report.check('cuda-oxide-kernels = ["cuda-oxide-backend", "dep:cuda-device", "dep:cuda-host"]' in texts["cargo"], "kernel feature missing")
    report.check('name = "ds4-cuda-fill-lifetime-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "ds4_gpu_tensor_fill_f32",
        "fill_f32_kernel<<<(count + 255u) / 256u, 256>>>",
        "__global__ static void fill_f32_kernel",
        "ds4_gpu_begin_commands",
        "ds4_gpu_flush_commands",
        "ds4_gpu_end_commands",
        "ds4_gpu_synchronize",
        "cudaDeviceSynchronize",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_tensor_fill_f32", True),
        ("owns_command_synchronization", True),
        ("owns_dequant_kernels", False),
        ("owns_graph_kernels", False),
        ("changes_default_route", False),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_1B4_SCOPE",
        "owns_tensor_fill_f32: true",
        "owns_command_synchronization: true",
        "owns_dequant_kernels: false",
        "owns_graph_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    failure = require_dict(report, execution.get("failed_before_tool_fix"), "failed_before_tool_fix")
    report.check(failure.get("target") == "sm_103", "pre-fix target failure drift")
    report.check("PTX JIT compilation failed" in failure.get("runtime_error", ""), "pre-fix JIT failure missing")
    report.check("cargo test -p cargo-oxide" in execution.get("cuda_oxide_test_command", ""), "fork validation command missing")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    report.check("--features cuda-oxide-kernels" in execution.get("command", ""), "kernel feature command missing")
    report.check("--bin ds4-cuda-fill-lifetime-smoke" in execution.get("command", ""), "smoke command missing")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.1b4",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "begin_is_noop": True,
        "prefix_fill_matches": True,
        "negative_infinity_fill_matches": True,
        "zero_count_is_noop": True,
        "bounds_rejected": True,
        "flush_is_context_wide": True,
        "end_is_context_wide": True,
        "synchronize_is_context_wide": True,
        "owns_tensor_fill_f32": True,
        "owns_command_synchronization": True,
        "owns_dequant_kernels": False,
        "owns_graph_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 fill/command-lifetime result drift")
    for marker in [
        "#[cuda_module]",
        "#[kernel]",
        "pub fn fill_f32(count: u64",
        "thread::index_1d()",
        "const THREADS_PER_BLOCK: u32 = 256",
        "count.div_ceil(THREADS_PER_BLOCK as u64)",
        "f32::NEG_INFINITY",
        "Err(FillF32Error::CountExceedsTensor",
        "substrate.begin_commands()",
    ]:
        report.check(marker in texts["smoke"], f"kernel smoke marker missing: {marker}")
    for marker in [
        "pub fn context(&self)",
        "pub fn stream(&self)",
        "pub fn begin_commands(&self)",
        "pub fn flush_commands(&self)",
        "pub fn end_commands(&self)",
        "pub fn synchronize_device(&self)",
    ]:
        report.check(marker in texts["substrate"], f"substrate command marker missing: {marker}")
    report.check(texts["substrate"].count("self.context.synchronize()") >= 3, "context-wide synchronization wiring missing")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.1b4/fill-command-lifetime-smoke.json"
    checker = "check_fill_command_lifetime_smoke.py"
    report.check("M14.1b4: Fill Kernel And Command Lifetime" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b4: Fill Kernel And Command Lifetime" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.1c Substrate Route Closure Gate" in texts["status"]
        or "Active item: M14.2" in texts["status"],
        "next active stage missing",
    )
    report.check("M14.1b4 Fill Kernel And Command Lifetime" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("kernel execution absent", lambda value: value["b300_execution"]["stdout"].update({"rust_kernel_toolchain": False})),
        ("begin no-op absent", lambda value: value["b300_execution"]["stdout"].update({"begin_is_noop": False})),
        ("prefix fill absent", lambda value: value["b300_execution"]["stdout"].update({"prefix_fill_matches": False})),
        ("portable target lost", lambda value: value["b300_execution"].update({"backend_selected_target": "sm_103"})),
        ("dequant overclaim", lambda value: value["ownership"].update({"owns_dequant_kernels": True})),
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
    print(f"M14.1b4 fill/command-lifetime smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
