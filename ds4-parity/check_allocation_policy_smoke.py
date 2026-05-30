#!/usr/bin/env python3
"""Validate the M14.1b3a Rust-owned managed-KV and memory-report policy smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b3a/allocation-policy-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
POLICY = ROOT / "rust/ds4-cuda/src/allocation_policy.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/allocation_policy_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

FIXTURE_REVISION = "0ec61156a7c5d65802402898b7a197bfff266d31"
CURRENT_REVISION = "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7"
EXPECTED_RUST_OWNED = [
    "cuda-oxide live device memory-capacity query",
    "managed tensor allocation through existing Rust substrate",
    "managed-KV threshold and reserve policy decisions",
    "CUDA memory-report output formatting",
]
EXPECTED_NOT_CLAIMED = [
    "Q8/F16 or Q8/F32 converted range-cache allocation",
    "quality-mode cuBLAS math configuration",
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
    report.check(fixture.get("schema") == "ds4.allocation_policy_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b3a", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == FIXTURE_REVISION, "cuda-oxide fixture revision drift")
    report.check(oxide.get("new_api") == "CudaContext::memory_info", "memory query API drift")
    report.check(f'rev = "{CURRENT_REVISION}"' in texts["cargo"], "crate current revision pin missing")
    report.check(f"#{CURRENT_REVISION}" in texts["lock"], "lockfile current revision pin missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "ds4_gpu_tensor_alloc_managed",
        "cuda_managed_kv_reserve_bytes",
        "ds4_gpu_should_use_managed_kv_cache",
        "cudaMemGetInfo",
        "ds4_gpu_print_memory_report",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_managed_tensor_allocation", True),
        ("owns_managed_kv_selection", True),
        ("owns_memory_report", True),
        ("owns_q8_cache_policy", False),
        ("owns_quality_mode", False),
        ("owns_ds4_kernels", False),
        ("changes_default_route", False),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_1B3A_SCOPE",
        "owns_managed_kv_selection: true",
        "owns_memory_report: true",
        "owns_q8_cache_policy: false",
        "owns_quality_mode: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    for marker in [
        "pub fn managed_kv_reserve_bytes",
        "pub fn managed_kv_decision",
        "ManagedKvReason::MemoryQueryUnavailable",
        "ManagedKvReason::ContextConsumesReserve",
        "pub fn format_cuda_memory_report",
    ]:
        report.check(marker in texts["policy"], f"allocation policy marker missing: {marker}")
    report.check("self.context.memory_info()" in texts["substrate"], "live memory query wiring missing")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check("--test residency" not in execution.get("cuda_oxide_test_command", ""), "fork validation was reduced")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    report.check("--bin ds4-cuda-allocation-policy-smoke" in execution.get("command", ""), "smoke command missing")
    report.check(
        execution.get("stderr_prefix") == "ds4: CUDA memory report b3a live: free ",
        "memory-report output marker drift",
    )
    expected = {
        "milestone": "M14.1b3a",
        "device_name": "NVIDIA B300 SXM6 AC",
        "live_memory_info_valid": True,
        "managed_allocation": True,
        "zero_kv_uses_device": True,
        "huge_kv_uses_managed": True,
        "small_context_uses_device": True,
        "memory_query_failure_uses_device": True,
        "sufficient_capacity_uses_device": True,
        "reserve_pressure_uses_managed": True,
        "context_exceeds_free_uses_managed": True,
        "memory_report_shape_matches": True,
        "owns_managed_tensor_allocation": True,
        "owns_managed_kv_selection": True,
        "owns_memory_report": True,
        "owns_q8_cache_policy": False,
        "owns_quality_mode": False,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 allocation-policy result drift")
    for marker in [
        "substrate.memory_capacity()",
        "managed_zeroed",
        "ManagedKvReason::ContextConsumesReserve",
        "format_cuda_memory_report",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.1b3a/allocation-policy-smoke.json"
    checker = "check_allocation_policy_smoke.py"
    report.check("M14.1b3a: Managed KV And Memory Report Policy" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b3a: Managed KV And Memory Report Policy" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.1b3b Q8 Cache And Quality Policy" in texts["status"]
        or "Active item: M14.1b4 Fill Kernel And Command Lifetime" in texts["status"]
        or "Active item: M14.1c Substrate Route Closure Gate" in texts["status"]
        or "Active item: M14." in texts["status"],
        "successor active stage missing",
    )
    report.check("M14.1b3a Managed KV And Memory Report Policy" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("live memory query absent", lambda value: value["b300_execution"]["stdout"].update({"live_memory_info_valid": False})),
        ("reserve-pressure selection absent", lambda value: value["b300_execution"]["stdout"].update({"reserve_pressure_uses_managed": False})),
        ("quality ownership overclaim", lambda value: value["ownership"].update({"owns_quality_mode": True})),
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
    print(f"M14.1b3a allocation policy smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
