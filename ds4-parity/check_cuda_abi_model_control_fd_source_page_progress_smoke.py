#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2a public fd page/progress ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-source-page-progress-smoke.json"
LOWER_POLICY = ROOT / "ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2a/abi_model_control_fd_source_page_progress_link_smoke.c"
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
    lower_policy = json.loads(LOWER_POLICY.read_text(encoding="utf-8"))
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
    validate(report, fixture, lower_policy, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, lower_policy, texts)
    state = "PASS" if report.ok else "FAIL"
    print(
        "M14.6b2b2b2b2b2b2b2b2b2b2a Rust CUDA public fd source-page "
        f"progress ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    lower_policy: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_fd_source_page_progress_smoke.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2a", "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-fd-source-page-progress-abi",
        "status drift",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, lower_policy, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == [
            "cuda_model_drop_file_pages",
            "cuda_model_discard_source_pages",
            "cuda_model_load_progress_note",
            "cuda_model_range_ptr_from_fd",
        ],
        "oracle symbols drift",
    )
    for marker in [
        "static void cuda_model_discard_source_pages(",
        'getenv("DS4_CUDA_KEEP_MODEL_PAGES")',
        "posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED)",
        "static void cuda_model_drop_file_pages(",
        "posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED)",
        "static void cuda_model_load_progress_note(uint64_t cached_bytes)",
        'getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL',
        "cuda_model_drop_file_pages(offset + copied, n);",
        "cuda_model_load_progress_note(g_model_range_bytes + copied);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C page/progress marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_fd_source_file_discard_advice", True),
        ("owns_source_mapping_discard_advice", True),
        ("owns_non_tty_progress_reporting", True),
        ("owns_verbose_progress_suppression", True),
        ("owns_synchronized_progress_reset", True),
        ("owns_physical_page_eviction_observation", False),
        ("owns_tty_progress_refresh_observation", False),
        ("owns_remaining_model_control_selection", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 45, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "struct AbiModelLoadProgress",
        "fn abi_model_discard_source_pages(",
        "fn abi_model_drop_file_pages(",
        "fn abi_model_load_progress_note(",
        "libc::posix_madvise(",
        "libc::posix_fadvise(",
        "abi_model_drop_file_pages(fd, file_offset, this_chunk)?;",
        "abi_model_discard_source_pages(model_map, model_size, file_offset, this_chunk)?;",
        "model_arenas.progress.reset();",
    ]:
        report.check(marker in texts["abi"], f"Rust page/progress marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiFdSourcePageProgressScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2A_SCOPE",
        "owns_fd_source_file_discard_advice: true",
        "owns_source_mapping_discard_advice: true",
        "owns_non_tty_progress_reporting: true",
        "owns_verbose_progress_suppression: true",
        "owns_synchronized_progress_reset: true",
        "owns_remaining_model_control_selection: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(
    report: ReportState,
    fixture: dict[str, Any],
    lower_policy: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 106),
        ("feature_release_test_count", 112),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    request = require_dict(report, execution.get("public_request"), "public_request")
    for key, expected in [
        ("copy_chunk_bytes", 16777216),
        ("cache_bytes", 16781312),
        ("chunks_per_admitted_upload", 2),
        ("ordinary_admitted_uploads", 2),
        ("suppressed_admitted_uploads", 1),
    ]:
        report.check(request.get(key) == expected, f"public request drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "page_aligned_host_maps",
        "fd_before_map_binds_host_base",
        "buffered_only_environment",
        "multi_chunk_fd_cache_request",
        "source_file_advice_observed",
        "source_mapping_advice_observed",
        "non_tty_progress_message_captured",
        "progress_reset_on_model_replacement",
        "keep_pages_suppresses_advice",
        "verbose_suppresses_progress",
        "fd_bytes_precede_divergent_host_map",
        "weighted_output_matches",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "budget_regression_passed",
        "fd_arena_regression_passed",
        "buffered_async_regression_passed",
        "direct_io_async_regression_passed",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    lower_stdout = require_dict(
        report,
        require_dict(report, lower_policy.get("b300_execution"), "lower policy execution").get("stdout"),
        "lower policy stdout",
    )
    for key, expected in [
        ("source_file_discard_calls", 2),
        ("source_mapping_discard_calls", 2),
        ("progress_notes", 3),
        ("progress_messages", 1),
        ("keep_source_pages_suppresses_advice", True),
        ("disabled_progress_suppresses_messages", True),
    ]:
        report.check(lower_stdout.get(key) == expected, f"lower policy baseline drift: {key}")
    for marker in [
        "int posix_fadvise(",
        "int posix_madvise(",
        'setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB", "16", 1)',
        'unsetenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")',
        'unsetenv("DS4_CUDA_KEEP_MODEL_PAGES")',
        'setenv("DS4_CUDA_KEEP_MODEL_PAGES", "1", 1)',
        'setenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE", "1", 1)',
        "cache_with_stderr_capture(",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("does not claim physical source-page eviction" in value for value in risks), "advice boundary missing")
    report.check(any("does not claim TTY refresh rendering" in value for value in risks), "TTY boundary missing")
    report.check(any("residual model-control selection" in value for value in risks), "remaining selection risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-source-page-progress-smoke.json"
    checker = "check_cuda_abi_model_control_fd_source_page_progress_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2a: Public Fd Source-Page And Progress ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2a Public Fd Source-Page And Progress ABI"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b Residual Model-Control Selection Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    lower_policy: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("file advice ownership missing", lambda value: value["ownership"].update({"owns_fd_source_file_discard_advice": False})),
        ("mapping advice observation missing", lambda value: value["b300_execution"]["observed"].update({"source_mapping_advice_observed": False})),
        ("eviction overclaim", lambda value: value["ownership"].update({"owns_physical_page_eviction_observation": True})),
        ("residual selection overclaim", lambda value: value["ownership"].update({"owns_remaining_model_control_selection": True})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, lower_policy, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
