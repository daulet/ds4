#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b1 buffered fd async staging ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b1/abi-model-control-buffered-fd-async-staging-smoke.json"
LOWER_ASYNC = ROOT / "ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json"
DIRECT_ASYNC = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2a/abi-model-control-direct-io-async-staging-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b1/abi_model_control_buffered_fd_async_staging_link_smoke.c"
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
    lower_async = json.loads(LOWER_ASYNC.read_text(encoding="utf-8"))
    direct_async = json.loads(DIRECT_ASYNC.read_text(encoding="utf-8"))
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
    validate(report, fixture, lower_async, direct_async, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, lower_async, direct_async, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"M14.6b2b2b2b2b2b2b2b2b1 Rust CUDA buffered fd async staging ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    lower_async: dict[str, Any],
    direct_async: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_buffered_fd_async_staging_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-buffered-fd-async-staging-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, lower_async, direct_async, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == ["cuda_model_copy_chunk_bytes", "cuda_model_stage_pool_alloc", "cuda_model_stage_read", "cuda_model_range_ptr_from_fd"],
        "oracle symbols drift",
    )
    for marker in [
        'getenv("DS4_CUDA_MODEL_COPY_CHUNK_MB")',
        "static int cuda_model_stage_read",
        "return cuda_pread_full(g_model_fd, stage, bytes, offset);",
        "for (size_t i = 0; i < 4; i++)",
        "cudaEventSynchronize(g_model_stage_event[bi])",
        "cudaMemcpyAsync(dev + copied, payload",
        "cudaEventRecord(g_model_stage_event[bi]",
        "cudaStreamSynchronize(g_model_upload_stream)",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C buffered async marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_buffered_only_fd_async_staging", True),
        ("reuses_four_slot_event_ring", True),
        ("owns_fd_cache_budget_policy", False),
        ("owns_source_page_and_progress_policy", False),
        ("owns_remaining_model_control_selection", False),
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
    for marker in [
        "fn upload_abi_async_fd_range_into(",
        "match upload_abi_async_fd_arena_range(\n"
        "        backend,\n"
        "        abi_model_fd(model_map)?,\n"
        "        model_map,\n"
        "        model_size,\n"
        "        offset,\n"
        "        bytes,\n"
        "        false,\n"
        "    )? {",
        "let (payload_offset, direct) = if use_direct_io",
        "read_abi_buffered_fd_into(",
        "let mut stage_pool = ABI_MODEL_STAGE_POOL.lock().ok()?;",
        "event.synchronize().ok()?;",
        "enqueue_pinned_u8_range_async(",
        "stage_slot.event = Some(backend.record_event().ok()?);",
    ]:
        report.check(marker in texts["abi"], f"Rust buffered async marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiBufferedFdAsyncStagingScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B1_SCOPE",
        "owns_buffered_only_fd_async_staging: true",
        "reuses_four_slot_event_ring: true",
        "owns_fd_cache_budget_policy: false",
        "owns_source_page_and_progress_policy: false",
        "owns_remaining_model_control_selection: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(
    report: ReportState,
    fixture: dict[str, Any],
    lower_async: dict[str, Any],
    direct_async: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 103),
        ("feature_release_test_count", 107),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    request = require_dict(report, execution.get("public_request"), "public_request")
    for key, expected in [
        ("copy_chunk_bytes", 16777216),
        ("requested_chunks", 5),
        ("requested_cache_bytes", 83886080),
    ]:
        report.check(request.get(key) == expected, f"public request drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "page_aligned_host_map",
        "fd_before_map_binds_host_base",
        "buffered_only_environment",
        "multi_chunk_fd_cache_request",
        "fd_bytes_precede_mutated_host_map",
        "repeated_cache_reuses_device_copy",
        "weighted_output_matches",
        "embedded_libdevice_module_loaded",
        "direct_enabled_async_regression_passed",
        "staticlib_export_count_unchanged",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    lower_stdout = require_dict(
        report,
        require_dict(report, lower_async.get("b300_execution"), "lower async execution").get("stdout"),
        "lower async stdout",
    )
    for key, expected in [
        ("stage_slots", 4),
        ("chunks_uploaded", 7),
        ("stage_slot_reuse_waits", 2),
        ("events_recorded", 7),
    ]:
        report.check(lower_stdout.get(key) == expected, f"lower async baseline drift: {key}")
    direct_observed = require_dict(
        report,
        require_dict(report, direct_async.get("b300_execution"), "direct async execution").get("observed"),
        "direct async observed",
    )
    report.check(direct_observed.get("multi_chunk_fd_cache_request") is True, "direct async regression evidence missing")
    for marker in [
        'setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1)',
        'setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1)',
        'setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB", "16", 1)',
        "const uint64_t cache_bytes = chunk_bytes * 5ull;",
        "ds4_gpu_cache_model_range(model_map, model_size, offset, cache_bytes",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    report.check("events_recorded" not in texts["harness"], "public harness overclaims unobservable event state")
    risks = fixture.get("integration_risks", [])
    report.check(any("does not independently expose event counts" in value for value in risks), "observability caveat missing")
    report.check(any("range-arena cache budgets" in value for value in risks), "remaining policy risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b1/abi-model-control-buffered-fd-async-staging-smoke.json"
    checker = "check_cuda_abi_model_control_buffered_fd_async_staging_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b1: Public Buffered Fd Async Staging ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy" in texts["status"],
        "active item missing",
    )
    report.check("M14.6b2b2b2b2b2b2b2b2b1 Public Buffered Fd Async Staging ABI" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2b2b2b2b2b2b2b2b2 Arena Budget And Residual Model-Control Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    lower_async: dict[str, Any],
    direct_async: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("buffered async ownership missing", lambda value: value["ownership"].update({"owns_buffered_only_fd_async_staging": False})),
        ("slot reuse ownership missing", lambda value: value["ownership"].update({"reuses_four_slot_event_ring": False})),
        ("budget overclaim", lambda value: value["ownership"].update({"owns_fd_cache_budget_policy": True})),
        ("selection overclaim", lambda value: value["ownership"].update({"owns_remaining_model_control_selection": True})),
        ("buffered observation missing", lambda value: value["b300_execution"]["observed"].update({"buffered_only_environment": False})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, lower_async, direct_async, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
