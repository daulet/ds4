#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b1 public fd budget ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-cache-budget-smoke.json"
LOWER_BUDGET = ROOT / "ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi_model_control_fd_cache_budget_link_smoke.c"
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
    lower_budget = json.loads(LOWER_BUDGET.read_text(encoding="utf-8"))
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
    validate(report, fixture, lower_budget, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, lower_budget, texts)
    state = "PASS" if report.ok else "FAIL"
    print(
        "M14.6b2b2b2b2b2b2b2b2b2b1 Rust CUDA public fd cache budget "
        f"ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    lower_budget: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_fd_cache_budget_smoke.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-fd-cache-budget-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, lower_budget, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == ["cuda_model_cache_limit_bytes", "cuda_model_arena_alloc", "cuda_model_range_ptr_from_fd", "cuda_model_ptr"],
        "oracle symbols drift",
    )
    for marker in [
        'getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB")',
        "if (gb == 0) return UINT64_MAX;",
        "if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes)",
        "return cuda_model_ptr(model_map, offset);",
        "if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;",
        "g_model_range_bytes += bytes;",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C budget marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_fd_cache_budget_policy", True),
        ("owns_cache_limit_gib_override", True),
        ("owns_uncached_budget_fallback_pointer", True),
        ("owns_live_rejected_transfer_observation", True),
        ("owns_public_fallback_compute_observation", False),
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
        "struct AbiModelArenaState",
        "range_bytes: u64,",
        "enum AbiFdRangeResolution",
        "fn abi_model_cache_limit_bytes_from_value(",
        "if state.range_bytes > limit || bytes > limit - state.range_bytes",
        "if aligned_bytes > limit - state.range_bytes",
        "state.range_bytes = state.range_bytes.checked_add(bytes)?;",
        "AbiFdRangeResolution::BudgetFallback",
        "return operation(requested_device_ptr);",
        "fn abi_model_range_is_cached(",
        "Some(abi_model_range_is_cached(",
        "fn public_fd_cache_limit_override_matches_current_c_gib_policy()",
    ]:
        report.check(marker in texts["abi"], f"Rust budget marker missing: {marker}")
    cached_fn = texts["abi"].split("fn with_cached_abi_model_range", maxsplit=1)[1]
    report.check(
        cached_fn.index("AbiFdRangeResolution::BudgetFallback")
        < cached_fn.index("let source = unsafe"),
        "budget fallback must resolve before host source slice construction",
    )
    for marker in [
        "pub struct CudaAbiFdCacheBudgetScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B1_SCOPE",
        "owns_fd_cache_budget_policy: true",
        "owns_cache_limit_gib_override: true",
        "owns_uncached_budget_fallback_pointer: true",
        "owns_live_rejected_transfer_observation: true",
        "owns_source_page_and_progress_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(
    report: ReportState,
    fixture: dict[str, Any],
    lower_budget: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 105),
        ("feature_release_test_count", 111),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    request = require_dict(report, execution.get("public_request"), "public_request")
    for key, expected in [
        ("cache_limit_bytes", 1073741824),
        ("admitted_range_bytes", 28),
        ("rejected_range_bytes", 1073741824),
        ("rejected_source_access", "PROT_NONE"),
        ("fd_readable_bytes", 4096),
    ]:
        report.check(request.get(key) == expected, f"public request drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "page_aligned_sparse_host_map",
        "fd_before_map_binds_host_base",
        "buffered_only_environment",
        "one_gib_cache_limit_selected",
        "small_fd_range_admitted",
        "oversized_budget_fallback_not_reported_as_cached",
        "rejected_source_pages_unreadable",
        "admitted_fd_cache_retained_after_file_mutation",
        "weighted_output_matches",
        "budget_fallback_compute_not_claimed",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "fd_arena_regression_passed",
        "buffered_async_regression_passed",
        "direct_io_async_regression_passed",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    lower_stdout = require_dict(
        report,
        require_dict(report, lower_budget.get("b300_execution"), "lower budget execution").get("stdout"),
        "lower budget stdout",
    )
    for key, expected in [
        ("cache_limit_bytes", 28672),
        ("budget_fallbacks", 1),
        ("budget_fallback_not_cached", True),
        ("owns_range_cache_budget_fallback", True),
    ]:
        report.check(lower_stdout.get(key) == expected, f"lower budget baseline drift: {key}")
    for marker in [
        "PROT_NONE",
        'setenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB", "1", 1)',
        'setenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "256", 1)',
        'setenv("DS4_CUDA_COPY_MODEL", "", 1)',
        "const uint64_t rejected_bytes = limit_bytes;",
        "budget-rejected-repeat",
        "oversized_budget_fallback_not_reported_as_cached",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("leaving compute through" in value for value in risks), "fallback compute caveat missing")
    report.check(any("source-page discard/progress" in value for value in risks), "remaining policy risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-cache-budget-smoke.json"
    checker = "check_cuda_abi_model_control_fd_cache_budget_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b1: Public Fd Cache Budget Fallback ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b1 Public Fd Cache Budget Fallback ABI" in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2 Source-Page Progress And Residual Model-Control Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    lower_budget: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("budget ownership missing", lambda value: value["ownership"].update({"owns_fd_cache_budget_policy": False})),
        ("fallback ownership missing", lambda value: value["ownership"].update({"owns_uncached_budget_fallback_pointer": False})),
        ("compute overclaim", lambda value: value["ownership"].update({"owns_public_fallback_compute_observation": True})),
        ("rejection observation missing", lambda value: value["b300_execution"]["observed"].update({"oversized_budget_fallback_not_reported_as_cached": False})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, lower_budget, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
