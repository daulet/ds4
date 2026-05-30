#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a fd-arena failure ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-arena-failure-selection-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a/abi_model_control_fd_arena_failure_selection_link_smoke.c"
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
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public fd-arena "
        f"failure-selection ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_fd_arena_failure_selection_smoke.v1",
        "schema drift",
    )
    report.check(
        fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a",
        "milestone drift",
    )
    report.check(
        fixture.get("status") == "b300-pass-staticlib-fd-arena-failure-selection-abi",
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
        == ["cuda_model_arena_alloc", "cuda_model_range_ptr_from_fd", "ds4_gpu_cache_model_range"],
        "oracle symbols drift",
    )
    for marker in [
        "if (g_model_cache_full) return NULL;",
        "g_model_cache_full = 1;",
        'if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;',
        "return cuda_model_ptr(model_map, offset);",
        "return cuda_model_range_is_cached(model_map, offset, bytes);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C arena failure marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_non_strict_arena_failure_host_fallback", True),
        ("owns_strict_arena_failure_continuation", True),
        ("owns_persistent_arena_cache_full_state", True),
        ("owns_live_interposed_arena_failure_observation", True),
        ("owns_aligned_arena_budget_failure_routing", True),
        ("owns_remaining_failure_selection", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 59, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "cache_full: bool,",
        "fn strict_fd_weight_cache_selected() -> bool",
        'std::env::var_os("DS4_CUDA_STRICT_WEIGHT_CACHE").is_some()',
        "AbiFdArenaUpload::ArenaFallback",
        "if state.cache_full {",
        "state.cache_full = true;",
        "if aligned_bytes > limit - state.range_bytes",
        "backend.allocate_u8(chunk_bytes)",
        "model_arenas.cache_full = false;",
    ]:
        report.check(marker in texts["abi"], f"Rust arena failure marker missing: {marker}")
    upload_fn = texts["abi"].split("fn upload_abi_async_fd_arena_range", maxsplit=1)[1]
    report.check(
        upload_fn.index("bytes > limit - state.range_bytes")
        < upload_fn.index("if state.cache_full")
        < upload_fn.index("let reservation")
        < upload_fn.index("if aligned_bytes > limit - state.range_bytes"),
        "raw budget, cache-full, reusable-arena, and aligned-budget selection order drift",
    )
    for marker in [
        "pub fn allocate_u8(&self, len: usize)",
        "cuda_core::memory::malloc_async",
        "DeviceBuffer::from_raw_parts",
    ]:
        report.check(marker in texts["substrate"], f"substrate allocation marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiFdArenaFailureSelectionScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2A_SCOPE",
        "owns_non_strict_arena_failure_host_fallback: true",
        "owns_strict_arena_failure_continuation: true",
        "owns_persistent_arena_cache_full_state: true",
        "owns_live_interposed_arena_failure_observation: true",
        "owns_aligned_arena_budget_failure_routing: true",
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
        ("local_library_test_count", 112),
        ("feature_release_test_count", 119),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    failure = require_dict(report, execution.get("forced_failure"), "forced_failure")
    for key, expected in [
        ("interposed_symbol", "cuMemAllocAsync"),
        ("arena_chunk_bytes", 268435456),
        ("arena_failure_error", 2),
        ("registration_failure_error", 801),
    ]:
        report.check(failure.get(key) == expected, f"forced failure drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "buffered_fd_selection_active",
        "interposed_arena_allocation_failure",
        "non_strict_failure_returns_uncached_host_fallback",
        "non_strict_host_bytes_precede_file_bytes",
        "strict_failure_continues_to_cached_device_copy",
        "strict_cached_copy_retains_original_host_bytes",
        "persistent_cache_full_skips_second_arena_attempt",
        "registration_fallback_boundary_preserved",
        "weighted_outputs_match",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "fd_arena_suballocation_regression_passed",
        "fd_budget_cache_result_regression_passed",
        "default_fd_regression_passed",
        "registration_disable_regression_passed",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "CUresult cuMemAllocAsync(",
        'dlsym(RTLD_NEXT, "cuMemAllocAsync")',
        "CU_ERROR_OUT_OF_MEMORY",
        "arena_alloc_failures != 1",
        "arena_alloc_failures != 2",
        'setenv("DS4_CUDA_STRICT_WEIGHT_CACHE", "1", 1)',
        "non_strict_failure_returns_uncached_host_fallback",
        "strict_failure_continues_to_cached_device_copy",
        "persistent_cache_full_skips_second_arena_attempt",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("cuMemAllocAsync" in value for value in risks), "driver allocation boundary missing")
    report.check(any("not separately forced" in value for value in risks), "aligned-budget boundary missing")
    report.check(any("staging allocation/read/copy" in value for value in risks), "remaining failure risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = (
        "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a/"
        "abi-model-control-fd-arena-failure-selection-smoke.json"
    )
    checker = "check_cuda_abi_model_control_fd_arena_failure_selection_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a: Public Fd Arena Failure Selection ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a Public Fd Arena Failure Selection ABI"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2b Remaining Residual Failure Selection Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("non-strict ownership missing", lambda value: value["ownership"].update({"owns_non_strict_arena_failure_host_fallback": False})),
        ("strict ownership missing", lambda value: value["ownership"].update({"owns_strict_arena_failure_continuation": False})),
        ("cache-full ownership missing", lambda value: value["ownership"].update({"owns_persistent_arena_cache_full_state": False})),
        ("strict output missing", lambda value: value["b300_execution"]["observed"].update({"strict_failure_continues_to_cached_device_copy": False})),
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
