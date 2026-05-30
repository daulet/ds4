#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b1 direct-I/O error-disable ABI smoke."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b1/abi-model-control-direct-io-error-disable-smoke.json"
PUBLIC_SUCCESS = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2a/abi-model-control-direct-io-fd-cache-smoke.json"
LOWER_ASYNC = ROOT / "ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2a/abi_model_control_direct_io_fd_cache_link_smoke.c"
GPU_BUILD = ROOT / "rust/ds4-gpu/build.rs"
GPU_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"


@dataclass
class ReportState:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    public_success = json.loads(PUBLIC_SUCCESS.read_text(encoding="utf-8"))
    lower_async = json.loads(LOWER_ASYNC.read_text(encoding="utf-8"))
    texts = {
        "cuda_c": CUDA_C.read_text(encoding="utf-8"),
        "lib": CUDA_LIB.read_text(encoding="utf-8"),
        "abi": CUDA_ABI.read_text(encoding="utf-8"),
        "harness": HARNESS.read_text(encoding="utf-8"),
        "gpu_build": GPU_BUILD.read_text(encoding="utf-8"),
        "gpu_sys": GPU_SYS.read_text(encoding="utf-8"),
        "roadmap": ROADMAP.read_text(encoding="utf-8"),
        "todo": TODO.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
        "readme": README.read_text(encoding="utf-8"),
        "report": REPORT.read_text(encoding="utf-8"),
    }
    report = ReportState()
    validate(report, fixture, public_success, lower_async, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, public_success, lower_async, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"M14.6b2b2b2b2b2b2b2b1 Rust CUDA direct-I/O error-disable ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    public_success: dict[str, Any],
    lower_async: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_direct_io_error_disable_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-direct-io-error-disable-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, public_success, lower_async, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["cuda_model_stage_read", "ds4_gpu_set_model_fd", "cuda_model_range_ptr_from_fd"], "oracle symbols drift")
    for marker in [
        "const int direct_errno = errno;",
        "direct_errno == EINVAL",
        "direct_errno == EFAULT",
        "direct_errno == ENOTSUP",
        "direct_errno == EOPNOTSUPP",
        "(void)close(g_model_direct_fd);",
        "g_model_direct_fd = -1;",
        "g_model_direct_align = 1;",
        "return cuda_pread_full(g_model_fd, stage, bytes, offset);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C error-disable marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_direct_io_disable_after_selected_error", True),
        ("owns_current_c_direct_io_disable_error_classes", True),
        ("owns_live_public_error_observation", False),
        ("owns_async_fd_staging_ring", False),
        ("owns_fd_cache_budget_policy", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 53, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "fn abi_direct_io_error_disables(raw_os_error: Option<c_int>) -> bool",
        "[libc::EINVAL, libc::EFAULT, libc::ENOTSUP, libc::EOPNOTSUPP]",
        "fn disable_abi_direct_io_after_error(",
        "disable_abi_direct_io_after_error(&mut control, error.raw_os_error())",
        "control.model_direct_file = None;",
        "control.model_direct_align = 1;",
        "read_abi_buffered_fd_into(fd, offset",
        "fn public_direct_io_disable_error_classes_match_current_c_policy()",
    ]:
        report.check(marker in texts["abi"], f"Rust public error-disable marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiDirectIoErrorDisableScope",
        "pub const M14_6B2B2B2B2B2B2B2B1_SCOPE",
        "owns_direct_io_disable_after_selected_error: true",
        "owns_current_c_direct_io_disable_error_classes: true",
        "owns_live_public_error_observation: false",
        "owns_async_fd_staging_ring: false",
        "owns_fd_cache_budget_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check("AsyncPinnedRangeCache" not in texts["abi"], "async fd staging overclaim in public ABI")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(
    report: ReportState,
    fixture: dict[str, Any],
    public_success: dict[str, Any],
    lower_async: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 101),
        ("feature_release_test_count", 104),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "public_error_class_policy_test_passed",
        "lower_level_error_disable_policy_tests_passed",
        "c_linked_direct_enabled_success_regression_passed",
        "staticlib_export_count_unchanged",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("public_direct_read_error_live_exercised") is False, "live public error branch overclaim")
    public_observed = require_dict(
        report,
        require_dict(report, public_success.get("b300_execution"), "public success execution").get("observed"),
        "public success observed",
    )
    report.check(public_observed.get("fd_bytes_precede_mutated_host_map") is True, "public fd-cache success evidence missing")
    lower_stdout = require_dict(
        report,
        require_dict(report, lower_async.get("b300_execution"), "lower async execution").get("stdout"),
        "lower async stdout",
    )
    report.check(lower_stdout.get("direct_io_disable_after_error_policy_present") is True, "lower-level error policy evidence missing")
    report.check(lower_stdout.get("direct_io_error_branch_live_exercised") is False, "lower-level live error overclaim drift")
    report.check("direct_io_selected" not in texts["harness"], "public success harness overclaims direct selector state")
    risks = fixture.get("integration_risks", [])
    report.check(any("does not claim a live B300 public request" in value for value in risks), "live-public caveat missing")
    report.check(any("asynchronous staging" in value for value in risks), "async/budget risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b1/abi-model-control-direct-io-error-disable-smoke.json"
    checker = "check_cuda_abi_model_control_direct_io_error_disable_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b1: Direct-I/O Error Disable ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy" in texts["status"],
        "active item missing",
    )
    report.check("M14.6b2b2b2b2b2b2b2b1 Direct-I/O Error Disable ABI" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2b2b2b2b2b2b2b2 Public Async Staging And Residual Cache Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    public_success: dict[str, Any],
    lower_async: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("disable state missing", lambda value: value["ownership"].update({"owns_direct_io_disable_after_selected_error": False})),
        ("error classes missing", lambda value: value["ownership"].update({"owns_current_c_direct_io_disable_error_classes": False})),
        ("live public error overclaim", lambda value: value["ownership"].update({"owns_live_public_error_observation": True})),
        ("async ring overclaim", lambda value: value["ownership"].update({"owns_async_fd_staging_ring": True})),
        ("policy test missing", lambda value: value["b300_execution"]["observed"].update({"public_error_class_policy_test_passed": False})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, public_success, lower_async, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
