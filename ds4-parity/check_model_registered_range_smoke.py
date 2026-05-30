#!/usr/bin/env python3
"""Validate the M14.1b2b2 Rust-owned registered range selection smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b2b2/model-registered-range-smoke.json"
CRATE_CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
MODEL_MAP = ROOT / "rust/ds4-cuda/src/model_map.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/model_registered_range_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

FIXTURE_REVISION = "b938480882f208045bc36ecf29da1ec5531d55ba"
CURRENT_REVISION = "aabe10dc4fa0086375104458909e222d1ac1cfe3"
MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
MODEL_SIZE = 86720111488
EXPECTED_RUST_OWNED = [
    "page-aligned read-only mapped-host registration attempt for an unaligned requested range",
    "cuda-oxide immutable registration guard or explicit CUDA unsupported resolution",
    "mmap-sourced device-copy fallback after a read-only registration error",
    "strategy-keyed cache entry and exact requested-range fallback readback",
]
EXPECTED_NOT_CLAIMED = [
    "successful zero-copy registered mapping on B300",
    "cross-range disable policy after unsupported registration",
    "pageable HMM advice or prefetch selection",
    "O_DIRECT, asynchronous staging-ring, or cache-budget policy",
    "model-range consumption by DS4 compute kernels",
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
        "model_map": MODEL_MAP.read_text(encoding="utf-8"),
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
    report.check(fixture.get("schema") == "ds4.model_registered_range_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b2b2", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == FIXTURE_REVISION, "cuda-oxide fixture revision drift")
    report.check(f'rev = "{CURRENT_REVISION}"' in texts["cargo"], "current crate revision pin missing")
    report.check(oxide.get("feature") == "cuda-oxide-backend", "feature drift")
    validate_oracle(report, fixture, texts)
    validate_model_range(report, fixture)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "cudaHostRegister((void *)reg_addr",
        "cudaHostRegisterMapped | cudaHostRegisterReadOnly",
        "cudaErrorNotSupported",
        "cudaMemcpy((char *)dev + done",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_model_range(report: Report, fixture: dict[str, Any]) -> None:
    model = require_dict(report, fixture.get("model_range"), "model_range")
    report.check(model.get("path") == "/workspace/ds4/ds4flash.gguf", "model path drift")
    report.check(model.get("sha256") == MODEL_SHA256, "model hash drift")
    report.check(model.get("model_size") == MODEL_SIZE, "model size drift")
    report.check(model.get("range_offset") == 13, "range offset drift")
    report.check(model.get("range_bytes") == 4096, "range bytes drift")
    identity = require_dict(report, model.get("identity_verification"), "identity_verification")
    report.check(identity.get("stdout", "").startswith(MODEL_SHA256), "model hash output drift")
    report.check(identity.get("passed") is True, "model hash was not verified")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_page_aligned_read_only_registration_attempt", True),
        ("owns_mmap_device_copy_fallback_after_registration_error", True),
        ("owns_pageable_hmm_strategy", False),
        ("owns_o_direct_staging", False),
        ("owns_ds4_kernels", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_1B2B2_SCOPE",
        "owns_page_aligned_read_only_registration_attempt: true",
        "owns_mmap_device_copy_fallback_after_registration_error: true",
        "owns_pageable_hmm_strategy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    for marker in [
        "pub struct RegisteredRangeLayout",
        "registered_range_layout",
        "ReadOnlyRegisteredOrMmapDeviceCopy",
        "RegisteredRangeResolution::MmapDeviceCopyFallback",
    ]:
        report.check(marker in texts["model_map"], f"model-map marker missing: {marker}")
    for marker in ["ReadOnlyRegisteredHostMemory", "register_read_only_host_range"]:
        report.check(marker in texts["substrate"], f"substrate marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("cuda_toolkit") == "13.2", "CUDA toolkit drift")
    report.check("--bin ds4-cuda-model-registered-range-smoke" in execution.get("command", ""), "smoke command missing")
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    expected = {
        "milestone": "M14.1b2b2",
        "device_name": "NVIDIA B300 SXM6 AC",
        "model_size": MODEL_SIZE,
        "range_offset": 13,
        "range_bytes": 4096,
        "registration_page_size": 4096,
        "registration_offset": 0,
        "registration_bytes": 8192,
        "registration_device_offset": 13,
        "read_only_registration_attempted": True,
        "read_only_registration_supported": False,
        "read_only_registration_error_code": 801,
        "mmap_device_copy_fallback": True,
        "fallback_readback_matches": True,
        "strategy_cache_reused": True,
        "owns_page_aligned_read_only_registration_attempt": True,
        "owns_mmap_device_copy_fallback_after_registration_error": True,
        "owns_pageable_hmm_strategy": False,
        "owns_o_direct_staging": False,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    report.check(stdout == expected, "B300 registered range result drift")
    for marker in [
        "ReadOnlyRegisteredOrMmapDeviceCopy",
        "RegisteredRangeResolution::MmapDeviceCopyFallback",
        "registration_device_offset",
        "fallback_readback_matches",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.1b2b2/model-registered-range-smoke.json"
    checker = "check_model_registered_range_smoke.py"
    report.check("M14.1b2b2: Registered Range Strategy" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b2b2: Registered Range Strategy" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.1b2c Model Map Cache Closure" in texts["status"]
        or "Active item: M14.1b3 Allocation And Quality Policy" in texts["status"]
        or "Active item: M14.1b3b Q8 Cache And Quality Policy" in texts["status"]
        or "Active item: M14.1b4 Fill Kernel And Command Lifetime" in texts["status"]
        or "Active item: M14.1c Substrate Route Closure Gate" in texts["status"]
        or "Active item: M14.2" in texts["status"],
        "next active stage missing",
    )
    report.check("M14.1b2b2 Registered Range Strategy" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("zero-copy overclaim", lambda value: value["b300_execution"]["stdout"].update({"read_only_registration_supported": True})),
        ("wrong CUDA error", lambda value: value["b300_execution"]["stdout"].update({"read_only_registration_error_code": 1})),
        ("missing fallback", lambda value: value["b300_execution"]["stdout"].update({"mmap_device_copy_fallback": False})),
        ("readback mismatch", lambda value: value["b300_execution"]["stdout"].update({"fallback_readback_matches": False})),
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
    print(f"M14.1b2b2 model registered range smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
