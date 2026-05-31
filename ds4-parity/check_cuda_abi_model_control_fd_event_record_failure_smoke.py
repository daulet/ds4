#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba fd event-record failure ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba/abi-model-control-fd-event-record-failure-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba/abi_model_control_fd_event_record_failure_link_smoke.c"
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
    texts = {
        "cuda_c": CUDA_C.read_text(encoding="utf-8"),
        "lib": CUDA_LIB.read_text(encoding="utf-8"),
        "abi": CUDA_ABI.read_text(encoding="utf-8"),
        "substrate": SUBSTRATE.read_text(encoding="utf-8"),
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
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    state = "PASS" if report.ok else "FAIL"
    print(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba Rust CUDA public fd event-record "
        f"failure ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_fd_event_record_failure_smoke.v1",
        "schema drift",
    )
    report.check(
        fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba",
        "milestone drift",
    )
    report.check(
        fixture.get("status") == "b300-pass-staticlib-fd-event-record-failure-abi",
        "status drift",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols") == ["cuda_model_range_ptr_from_fd", "cuda_model_range_ptr"],
        "oracle symbols drift",
    )
    for marker in [
        "err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);",
        "ds4: CUDA model staging record failed for %s",
        "return NULL;",
        "const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);",
        "if (fd_ptr) return fd_ptr;",
        "if (g_model_range_mapping_supported) {",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C event-record marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_fd_event_record_failure_continuation", True),
        ("owns_strict_independent_event_record_failure_continuation", True),
        ("owns_live_retried_event_record_failure_observation", True),
        ("owns_remaining_event_wait_final_sync_failure_selection", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 74, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    upload_fn = texts["abi"].split("fn upload_abi_async_fd_range_into", maxsplit=1)[1].split(
        "fn upload_abi_async_fd_arena_range", maxsplit=1
    )[0]
    range_fn = texts["abi"].split("fn with_cached_abi_model_range", maxsplit=1)[1].split(
        "fn abi_model_range_is_cached", maxsplit=1
    )[0]
    for marker in [
        "stage_slot.event = Some(backend.record_event().ok()?);",
        "let upload_result = (|| -> Option<bool> {",
        "let used_direct = upload_result?;",
    ]:
        report.check(marker in upload_fn, f"Rust event-record propagation marker missing: {marker}")
    report.check(
        "pub fn record_event(&self) -> Result<CudaEvent, DriverError>" in texts["substrate"]
        and "self.stream.record_event(None)" in texts["substrate"],
        "DS4 cuda-oxide event adapter boundary missing",
    )
    for marker in [
        "let fd_resolution = if direct_io_fd_weight_cache_selected() {",
        "let storage = match fd_resolution {",
        "None => {",
        "match try_register_abi_model_range(",
        "AbiModelRangeStorage::DeviceCopy(backend.upload(source).ok()?)",
    ]:
        report.check(marker in range_fn, f"Rust post-event-record continuation marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiFdEventRecordFailureScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBA_SCOPE",
        "owns_fd_event_record_failure_continuation: true",
        "owns_strict_independent_event_record_failure_continuation: true",
        "owns_live_retried_event_record_failure_observation: true",
        "owns_remaining_event_wait_final_sync_failure_selection: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 117),
        ("feature_release_test_count", 124),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    failure = require_dict(report, execution.get("forced_failure"), "forced_failure")
    for key, expected in [
        ("interposed_symbol", "cuEventRecord"),
        ("event_record_failure_error", 1),
        ("registration_failure_error", 801),
    ]:
        report.check(failure.get(key) == expected, f"forced failure drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "buffered_fd_selection_active",
        "interposed_fd_event_record_failure",
        "event_record_failure_retries_without_cache_full_latch",
        "non_strict_event_record_failure_continues_to_cached_device_copy",
        "strict_event_record_failure_continues_to_cached_device_copy",
        "first_event_record_failure_enters_registration_fallback",
        "subsequent_event_record_failure_respects_registration_disable",
        "cached_fallback_retains_original_host_bytes",
        "host_bytes_precede_file_bytes_after_event_record_failure",
        "weighted_outputs_match",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "fd_read_failure_regression_passed",
        "fd_stage_allocation_failure_regression_passed",
        "fd_stage_pool_reuse_regression_passed",
        "fd_upload_failure_continuation_regression_passed",
        "fd_arena_failure_regression_passed",
        "fd_budget_cache_result_regression_passed",
        "default_fd_regression_passed",
        "direct_io_async_staging_regression_passed",
        "registration_disable_regression_passed",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "CUresult cuEventRecord(",
        'dlsym(RTLD_NEXT, "cuEventRecord")',
        "fail_event_records = 1;",
        "event_record_failures != 1",
        "event_record_failures != 2",
        "host_register_calls != register_calls_after_map + 1",
        'setenv("DS4_CUDA_STRICT_WEIGHT_CACHE", "1", 1)',
        "event_record_failure_retries_without_cache_full_latch",
        "subsequent_event_record_failure_respects_registration_disable",
        "cached_fallback_retains_original_host_bytes",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("forwards cuEventRecord" in value for value in risks), "event record boundary missing")
    report.check(any("range-registration disable" in value for value in risks), "registration-disable boundary missing")
    report.check(any("event-wait" in value for value in risks), "remaining event boundary missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = (
        "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba/"
        "abi-model-control-fd-event-record-failure-smoke.json"
    )
    checker = "check_cuda_abi_model_control_fd_event_record_failure_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba: Public Fd Event Record Failure Continuation ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba Public Fd Event Record Failure"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbb Remaining Graph Compute And Route Promotion Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("event-record ownership missing", lambda value: value["ownership"].update({"owns_fd_event_record_failure_continuation": False})),
        ("strict-independent ownership missing", lambda value: value["ownership"].update({"owns_strict_independent_event_record_failure_continuation": False})),
        ("event retry observation missing", lambda value: value["b300_execution"]["observed"].update({"event_record_failure_retries_without_cache_full_latch": False})),
        ("cached host result missing", lambda value: value["b300_execution"]["observed"].update({"cached_fallback_retains_original_host_bytes": False})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
