#!/usr/bin/env python3
"""Validate the M14.1b1 bounded cuda-oxide model-residency smoke contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json"
CRATE_CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/model_residency_smoke.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

FIXTURE_REVISION = "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200"
CURRENT_REVISION = "0ec61156a7c5d65802402898b7a197bfff266d31"
MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
MODEL_SIZE = 86720111488
WINDOW_BYTES = 4096
EXPECTED_RUST_OWNED = [
    "managed model-window allocation with read-mostly advice",
    "managed model-window preferred-device prefetch and stream attachment",
    "mapped host model-window allocation with device-visible pointer",
    "registered caller-owned model-window lifetime with device-visible pointer",
]
EXPECTED_NOT_CLAIMED = [
    "complete GGUF mapping or model-range cache ownership",
    "model file descriptor or direct-I/O ownership",
    "Q8/F16 range-cache policy",
    "managed KV policy, quality mode, or memory reporting",
    "DS4 compute kernels or tensor fill kernel",
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
    report.check(fixture.get("schema") == "ds4.model_residency_handles_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    validate_dependency(report, fixture, texts)
    validate_model_window(report, fixture)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_dependency(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == FIXTURE_REVISION, "cuda-oxide fixture revision drift")
    report.check(oxide.get("dependency") == "cuda-core", "cuda-core dependency drift")
    report.check(oxide.get("feature") == "cuda-oxide-backend", "feature name drift")
    report.check(f'rev = "{CURRENT_REVISION}"' in texts["cargo"], "current crate revision pin missing")
    report.check(
        'cuda-oxide-backend = ["dep:cuda-core", "dep:libc"]' in texts["cargo"],
        "current feature wiring missing",
    )


def validate_model_window(report: Report, fixture: dict[str, Any]) -> None:
    model = require_dict(report, fixture.get("model_window"), "model_window")
    report.check(model.get("path") == "/workspace/ds4/ds4flash.gguf", "model path drift")
    report.check(model.get("sha256") == MODEL_SHA256, "model hash drift")
    report.check(model.get("model_size") == MODEL_SIZE, "model size drift")
    report.check(model.get("window_offset") == 0, "model-window offset drift")
    report.check(model.get("window_bytes") == WINDOW_BYTES, "model-window size drift")
    identity = require_dict(report, model.get("identity_verification"), "model_window.identity_verification")
    report.check(identity.get("command") == "sha256sum /workspace/ds4/ds4flash.gguf", "model hash command drift")
    report.check(identity.get("stdout", "").startswith(MODEL_SHA256), "model hash output drift")
    report.check(identity.get("passed") is True, "model hash was not verified")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    report.check(ownership.get("opt_in_only") is True, "residency path is no longer opt-in")
    report.check(ownership.get("owns_complete_model_map") is False, "complete model-map overclaim")
    report.check(ownership.get("owns_ds4_kernels") is False, "kernel ownership overclaim")
    report.check(ownership.get("changes_default_route") is False, "route ownership overclaim")
    report.check(ownership.get("retains_current_c_cuda_oracle") is True, "current-C oracle dropped")
    for marker in [
        "pub const M14_1B1_SCOPE",
        "owns_complete_model_map: false",
        "owns_ds4_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"Rust scope marker missing: {marker}")
    for marker in [
        "ManagedBuffer::from_slice",
        "MemoryAdvice::SetReadMostly",
        "MemoryAdvice::SetPreferredLocation",
        "buffer.prefetch_to",
        "StreamAttachment::Single",
        "MappedHostBuffer::from_slice",
        "RegisteredHostMemory::new",
    ]:
        report.check(marker in texts["substrate"], f"residency operation missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("cuda_toolkit") == "13.2", "CUDA toolkit drift")
    report.check(execution.get("rust_toolchain") == "nightly-2026-04-03", "Rust toolchain drift")
    report.check("--bin ds4-cuda-model-residency-smoke" in execution.get("command", ""), "smoke command missing")
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    expected = {
        "milestone": "M14.1b1",
        "device_name": "NVIDIA B300 SXM6 AC",
        "model_size": MODEL_SIZE,
        "model_window_bytes": WINDOW_BYTES,
        "managed_advice_prefetch": True,
        "mapped_device_pointer": True,
        "registered_host_pointer": True,
        "owns_complete_model_map": False,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    report.check(stdout == expected, "B300 residency result drift")
    for marker in [
        "MODEL_WINDOW_BYTES",
        "prefetch_read_mostly_to_device",
        "return_managed_to_host",
        "mapped_from_slice",
        "register_host_range",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json"
    checker = "check_model_residency_handles_smoke.py"
    report.check("M14.1b1: Bounded Model Residency Handles" in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b1: Bounded Model Residency Handles" in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.1b" in texts["status"], "next active stage missing")
    report.check("M14.1b1 Bounded Model Residency Handles" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("complete model-map overclaim", lambda value: value["ownership"].update({"owns_complete_model_map": True})),
        ("kernel ownership overclaim", lambda value: value["ownership"].update({"owns_ds4_kernels": True})),
        ("failed registered pointer", lambda value: value["b300_execution"]["stdout"].update({"registered_host_pointer": False})),
        ("failed model identity", lambda value: value["model_window"]["identity_verification"].update({"passed": False})),
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
    print(f"M14.1b1 model residency handles smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
