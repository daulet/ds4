#!/usr/bin/env python3
"""Validate the M14.1a cuda-oxide host-substrate B300 smoke contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json"
WORKSPACE_CARGO = ROOT / "Cargo.toml"
LOCKFILE = ROOT / "Cargo.lock"
CRATE_CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/substrate_smoke.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

FIXTURE_REVISION = "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200"
CURRENT_REVISION = "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7"
EXPECTED_RUST_OWNED = [
    "CUDA primary context RAII",
    "CUDA non-blocking stream RAII",
    "device-buffer host-to-device and device-to-host roundtrip",
    "zeroed device-buffer allocation and readback",
    "managed-buffer allocation and host-visible lifetime",
]
EXPECTED_NOT_CLAIMED = [
    "DS4 compute kernels",
    "arbitrary tensor fill kernel",
    "model-map cache or prefetch policy",
    "runtime graph route",
    "default CUDA route",
    "ds4_cuda.cu removal",
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
        "workspace": WORKSPACE_CARGO.read_text(encoding="utf-8"),
        "lock": LOCKFILE.read_text(encoding="utf-8"),
        "cargo": CRATE_CARGO.read_text(encoding="utf-8"),
        "lib": CRATE_LIB.read_text(encoding="utf-8"),
        "substrate": SUBSTRATE.read_text(encoding="utf-8"),
        "smoke": SMOKE.read_text(encoding="utf-8"),
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
    report.check(fixture.get("schema") == "ds4.cuda_oxide_substrate_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1a", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    validate_dependency(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_dependency(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == FIXTURE_REVISION, "cuda-oxide fixture revision drift")
    report.check(oxide.get("dependency") == "cuda-core", "cuda-core dependency drift")
    report.check(oxide.get("feature") == "cuda-oxide-backend", "feature name drift")
    report.check('"rust/ds4-cuda"' in texts["workspace"], "workspace omits ds4-cuda")
    report.check('name = "ds4-cuda"' in texts["cargo"], "crate manifest missing")
    report.check(f'rev = "{CURRENT_REVISION}"' in texts["cargo"], "current crate revision pin missing")
    report.check(
        'cuda-oxide-backend = ["dep:cuda-core", "dep:libc"]' in texts["cargo"],
        "current feature wiring missing",
    )
    report.check(f"#{CURRENT_REVISION}" in texts["lock"], "Cargo.lock omits current cuda-oxide revision")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    report.check(ownership.get("opt_in_only") is True, "substrate is no longer opt-in")
    report.check(ownership.get("owns_ds4_kernels") is False, "kernel ownership overclaim")
    report.check(ownership.get("changes_default_route") is False, "route ownership overclaim")
    report.check(ownership.get("retains_current_c_cuda_oracle") is True, "current-C oracle dropped")
    for marker in [
        "pub const M14_1A_SCOPE",
        "owns_ds4_kernels: false",
        "changes_default_route: false",
        'cfg(feature = "cuda-oxide-backend")',
    ]:
        report.check(marker in texts["lib"], f"Rust scope marker missing: {marker}")
    for marker in [
        "CudaContext::new",
        "context.new_stream",
        "DeviceBuffer::from_host",
        "DeviceBuffer::zeroed",
        "ManagedBuffer::zeroed",
        "buffer.to_host_vec",
    ]:
        report.check(marker in texts["substrate"], f"substrate operation missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("cuda_toolkit") == "13.2", "CUDA toolkit drift")
    report.check(execution.get("rust_toolchain") == "nightly-2026-04-03", "Rust toolchain drift")
    report.check("--features cuda-oxide-backend" in execution.get("command", ""), "feature smoke command missing")
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    expected = {
        "milestone": "M14.1a",
        "cuda_oxide_substrate": True,
        "device_ordinal": 0,
        "device_name": "NVIDIA B300 SXM6 AC",
        "device_roundtrip": True,
        "zeroed_roundtrip": True,
        "managed_lifetime": True,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    report.check(stdout == expected, "B300 smoke result drift")
    for marker in ["CudaOxideSubstrate::open", "device_roundtrip", "zeroed_roundtrip", "managed_lifetime"]:
        report.check(marker in texts["smoke"], f"smoke binary marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json"
    checker = "check_cuda_oxide_substrate_smoke.py"
    report.check("M14.1a: Host Substrate Buffer Roundtrip" in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1a: Host Substrate Buffer Roundtrip" in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14." in texts["status"],
        "next active stage missing",
    )
    report.check("M14.1a Host Substrate Buffer Roundtrip" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("kernel ownership overclaim", lambda value: value["ownership"].update({"owns_ds4_kernels": True})),
        ("route ownership overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("failed roundtrip", lambda value: value["b300_execution"]["stdout"].update({"device_roundtrip": False})),
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
    print(f"M14.1a cuda-oxide substrate smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
