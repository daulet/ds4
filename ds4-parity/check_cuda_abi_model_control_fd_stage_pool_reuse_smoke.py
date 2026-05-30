#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba fd stage-pool reuse ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba/abi-model-control-fd-stage-pool-reuse-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba/abi_model_control_fd_stage_pool_reuse_link_smoke.c"
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
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba Rust CUDA public fd stage-pool "
        f"reuse ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_fd_stage_pool_reuse_smoke.v1",
        "schema drift",
    )
    report.check(
        fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba",
        "milestone drift",
    )
    report.check(
        fixture.get("status") == "b300-pass-staticlib-fd-stage-pool-reuse-abi",
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
        oracle.get("symbols")
        == ["cuda_model_stage_pool_alloc", "cuda_model_range_ptr_from_fd", "ds4_gpu_cleanup"],
        "oracle symbols drift",
    )
    for marker in [
        "static void *g_model_stage_raw[4];",
        "static cudaEvent_t g_model_stage_event[4];",
        "static uint64_t g_model_stage_bytes;",
        "static int cuda_model_stage_pool_alloc(uint64_t bytes) {",
        "if (g_model_stage_bytes >= bytes) return 1;",
        "if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;",
        "(void)cudaFreeHost(g_model_stage_raw[i]);",
        "g_model_stage_bytes = 0;",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C stage-pool marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_retained_four_slot_fd_stage_pool", True),
        ("owns_stage_pool_cleanup_boundary", True),
        ("owns_live_second_range_reuse_observation", True),
        ("owns_live_armed_second_allocation_suppression_observation", True),
        ("owns_remaining_failure_selection", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 29, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    upload_fn = texts["abi"].split("fn upload_abi_async_fd_range_into", maxsplit=1)[1].split(
        "fn upload_abi_async_fd_arena_range", maxsplit=1
    )[0]
    cleanup_fn = texts["abi"].split('pub extern "C" fn ds4_gpu_cleanup()', maxsplit=1)[1].split(
        'pub extern "C" fn ds4_gpu_tensor_alloc', maxsplit=1
    )[0]
    model_map_fn = texts["abi"].split('pub unsafe extern "C" fn ds4_gpu_set_model_map(', maxsplit=1)[1].split(
        'pub extern "C" fn ds4_gpu_set_model_fd', maxsplit=1
    )[0]
    for marker in [
        "static ABI_MODEL_STAGE_POOL: Mutex<AbiModelStagePool>",
        "struct AbiModelStageSlot",
        "struct AbiModelStagePool",
        "let mut stage_pool = ABI_MODEL_STAGE_POOL.lock().ok()?;",
        "if stage_pool.stage_bytes < stage_bytes {",
        "stage_pool.slots.clear();",
        "stage_pool.slots.push(AbiModelStageSlot {",
        "let stage_slot = stage_pool.slots.get_mut(slot)?;",
        "if synchronize_ok {",
        "slot.event = None;",
    ]:
        report.check(marker in texts["abi"] or marker in upload_fn, f"Rust stage-pool marker missing: {marker}")
    report.check("ABI_MODEL_STAGE_POOL.lock()" in cleanup_fn, "cleanup does not release stage pool")
    report.check("ABI_MODEL_STAGE_POOL.lock()" not in model_map_fn, "model replacement unexpectedly clears stage pool")
    for marker in [
        "pub struct CudaAbiFdStagePoolReuseScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBA_SCOPE",
        "owns_retained_four_slot_fd_stage_pool: true",
        "owns_stage_pool_cleanup_boundary: true",
        "owns_live_second_range_reuse_observation: true",
        "owns_live_armed_second_allocation_suppression_observation: true",
        "owns_remaining_failure_selection: false",
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
        ("local_library_test_count", 114),
        ("feature_release_test_count", 121),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    forced = require_dict(report, execution.get("forced_boundary"), "forced_boundary")
    for key, expected in [
        ("interposed_symbol", "cuMemAllocHost_v2"),
        ("armed_second_range_allocation_error", 2),
        ("registration_failure_error", 801),
    ]:
        report.check(forced.get(key) == expected, f"forced boundary drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "buffered_fd_selection_active",
        "four_slot_stage_pool_created_once",
        "second_range_reuses_stage_pool",
        "armed_second_allocation_failure_not_triggered",
        "fd_bytes_win_after_pool_reuse",
        "cached_ranges_retain_original_fd_bytes",
        "registration_fallback_not_entered_for_second_range",
        "weighted_outputs_match",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "fd_upload_failure_continuation_regression_passed",
        "fd_arena_failure_regression_passed",
        "fd_budget_cache_result_regression_passed",
        "default_fd_regression_passed",
        "direct_io_async_staging_regression_passed",
        "registration_disable_regression_passed",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "CUresult cuMemAllocHost_v2(",
        'dlsym(RTLD_NEXT, "cuMemAllocHost_v2")',
        "fail_future_stage_alloc = 1;",
        "pinned_alloc_calls != 4",
        "injected_second_range_stage_alloc_failures != 0",
        'setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1)',
        "second_range_reuses_stage_pool",
        "cached_ranges_retain_original_fd_bytes",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("cuMemAllocHost_v2" in value for value in risks), "allocation reuse boundary missing")
    report.check(any("remain separate observations" in value for value in risks), "remaining failure boundary missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = (
        "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba/"
        "abi-model-control-fd-stage-pool-reuse-smoke.json"
    )
    checker = "check_cuda_abi_model_control_fd_stage_pool_reuse_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba: Public Fd Stage Pool Reuse ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbb Remaining Residual Failure Selection Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba Public Fd Stage Pool Reuse"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbb Remaining Residual Failure Selection Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("retained pool ownership missing", lambda value: value["ownership"].update({"owns_retained_four_slot_fd_stage_pool": False})),
        ("cleanup boundary missing", lambda value: value["ownership"].update({"owns_stage_pool_cleanup_boundary": False})),
        ("reuse observation missing", lambda value: value["b300_execution"]["observed"].update({"second_range_reuses_stage_pool": False})),
        ("allocation suppression missing", lambda value: value["b300_execution"]["observed"].update({"armed_second_allocation_failure_not_triggered": False})),
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
