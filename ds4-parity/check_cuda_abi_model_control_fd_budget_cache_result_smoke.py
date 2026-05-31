#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 fd-budget cache-result ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-budget-cache-result-smoke.json"
PREDECESSOR = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-cache-budget-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b1/abi_model_control_fd_budget_cache_result_link_smoke.c"
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
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
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
    validate(report, fixture, predecessor, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, predecessor, texts)
    state = "PASS" if report.ok else "FAIL"
    print(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public fd-budget "
        f"cache-result ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_fd_budget_cache_result_smoke.v1",
        "schema drift",
    )
    report.check(
        fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1",
        "milestone drift",
    )
    report.check(
        fixture.get("status") == "b300-pass-staticlib-fd-budget-cache-result-abi",
        "status drift",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, predecessor, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == [
            "ds4_gpu_cache_model_range",
            "cuda_model_range_ptr_from_fd",
            "cuda_model_range_is_cached",
            "cuda_model_arena_alloc",
        ],
        "oracle symbols drift",
    )
    for marker in [
        "return cuda_model_ptr(model_map, offset);",
        'if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;',
        "if (g_model_cache_full) return NULL;",
        "return cuda_model_range_is_cached(model_map, offset, bytes);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C cache-result marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_uncached_fd_budget_cache_result", True),
        ("owns_live_budget_fallback_compute_observation", True),
        ("owns_arena_allocation_failure_selection", False),
        ("owns_remaining_failure_selection", False),
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
        "fn with_cached_abi_model_range",
        "AbiFdRangeResolution::BudgetFallback",
        "return operation(requested_device_ptr);",
        "fn abi_model_range_is_cached(",
        "Some(abi_model_range_is_cached(",
    ]:
        report.check(marker in texts["abi"], f"Rust cache-result marker missing: {marker}")
    cached_fn = texts["abi"].split("fn with_cached_abi_model_range", maxsplit=1)[1]
    report.check(cached_fn.count("drop(ranges);") >= 3, "range mutex releases missing before callbacks")
    report.check(
        re.search(r"drop\(ranges\);\s+return operation\(requested_device_ptr\);", cached_fn)
        is not None,
        "budget fallback callback must run after range mutex release",
    )
    report.check(
        re.search(r"drop\(ranges\);\s+operation\(ptr\)", cached_fn) is not None,
        "inserted range callback must run after range mutex release",
    )
    for marker in [
        "pub struct CudaAbiFdBudgetCacheResultScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B1_SCOPE",
        "owns_uncached_fd_budget_cache_result: true",
        "owns_live_budget_fallback_compute_observation: true",
        "owns_arena_allocation_failure_selection: false",
        "owns_remaining_failure_selection: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(
    report: ReportState,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 111),
        ("feature_release_test_count", 118),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    request = require_dict(report, execution.get("public_request"), "public_request")
    for key, expected in [
        ("cache_limit_bytes", 1073741824),
        ("admitted_range_bytes", 1073741808),
        ("fallback_offset", 1073741824),
        ("fallback_weight_bytes", 28),
        ("whole_map_registration_error", 801),
    ]:
        report.check(request.get(key) == expected, f"public request drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "default_fd_selection_without_weight_cache",
        "one_gib_cache_limit_selected",
        "near_limit_fd_range_admitted",
        "budget_fallback_not_reported_as_cached",
        "budget_fallback_consumed_by_weighted_rms",
        "host_bytes_precede_file_bytes_after_budget_fallback",
        "whole_map_registration_attempt_preserved",
        "weighted_outputs_match",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "corrected_fd_budget_regression_passed",
        "default_fd_regression_passed",
        "direct_model_regression_passed",
        "pageable_hmm_regression_passed",
        "full_model_copy_regression_passed",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    predecessor_observed = require_dict(
        report,
        require_dict(report, predecessor.get("b300_execution"), "predecessor execution").get("observed"),
        "predecessor observed",
    )
    report.check(
        predecessor_observed.get("oversized_budget_fallback_not_reported_as_cached") is True,
        "corrected predecessor cache result missing",
    )
    for marker in [
        "CUresult cuMemHostRegister_v2(",
        "return 801;",
        "const uint64_t admitted_bytes = limit_bytes - 16;",
        'unsetenv("DS4_CUDA_WEIGHT_CACHE")',
        '"budget-fill"',
        '"budget-fallback"',
        "ds4_gpu_rms_norm_weight_tensor(",
        "host_register_calls != 1",
        "budget_fallback_not_reported_as_cached",
        "host_bytes_precede_file_bytes_after_budget_fallback",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("near-one-GiB" in value for value in risks), "one-GiB live-boundary caveat missing")
    report.check(any("DS4_CUDA_STRICT_WEIGHT_CACHE" in value for value in risks), "strict-cache boundary missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = (
        "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b1/"
        "abi-model-control-fd-budget-cache-result-smoke.json"
    )
    checker = "check_cuda_abi_model_control_fd_budget_cache_result_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1: Public Fd Budget Fallback Cache-Result ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 Public Fd Budget Fallback Cache-Result ABI"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2 Remaining Residual Failure Selection Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("cache-result ownership missing", lambda value: value["ownership"].update({"owns_uncached_fd_budget_cache_result": False})),
        ("compute observation missing", lambda value: value["ownership"].update({"owns_live_budget_fallback_compute_observation": False})),
        ("arena overclaim", lambda value: value["ownership"].update({"owns_arena_allocation_failure_selection": True})),
        ("host-byte observation missing", lambda value: value["b300_execution"]["observed"].update({"host_bytes_precede_file_bytes_after_budget_fallback": False})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, predecessor, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
