#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2a public pageable HMM fallback ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2a/abi-model-control-pageable-hmm-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2a/abi_model_control_pageable_hmm_link_smoke.c"
GPU_BUILD = ROOT / "rust/ds4-gpu/build.rs"
GPU_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_SYMBOLS = [
    "ds4_gpu_add_tensor",
    "ds4_gpu_begin_commands",
    "ds4_gpu_cache_model_range",
    "ds4_gpu_cleanup",
    "ds4_gpu_directional_steering_project_tensor",
    "ds4_gpu_end_commands",
    "ds4_gpu_flush_commands",
    "ds4_gpu_init",
    "ds4_gpu_repeat_hc_tensor",
    "ds4_gpu_rms_norm_plain_rows_tensor",
    "ds4_gpu_rms_norm_plain_tensor",
    "ds4_gpu_rms_norm_weight_rows_tensor",
    "ds4_gpu_rms_norm_weight_tensor",
    "ds4_gpu_set_model_fd",
    "ds4_gpu_set_model_map",
    "ds4_gpu_set_model_map_range",
    "ds4_gpu_should_use_managed_kv_cache",
    "ds4_gpu_swiglu_tensor",
    "ds4_gpu_synchronize",
    "ds4_gpu_tensor_alloc",
    "ds4_gpu_tensor_alloc_managed",
    "ds4_gpu_tensor_bytes",
    "ds4_gpu_tensor_contents",
    "ds4_gpu_tensor_copy",
    "ds4_gpu_tensor_fill_f32",
    "ds4_gpu_tensor_free",
    "ds4_gpu_tensor_read",
    "ds4_gpu_tensor_view",
    "ds4_gpu_tensor_write",
]


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
    args = parse_args(argv)
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
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.6b2b2b2b2b2a Rust CUDA pageable HMM fallback ABI smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_pageable_hmm_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2a", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-pageable-hmm-fallback-abi", "status drift")
    report.check(fixture.get("exported_symbols") == EXPECTED_SYMBOLS, "exported symbol drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == ["cuda_model_prefetch_range", "ds4_gpu_set_model_map_range", "cuda_model_range_ptr"],
        "oracle symbols drift",
    )
    for marker in [
        "cuda_model_prefetch_range",
        'getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL',
        'getenv("DS4_CUDA_NO_MODEL_COPY") != NULL',
        "cudaDevAttrPageableMemoryAccess",
        "cudaMemAdviseSetReadMostly",
        "cudaMemAdviseSetPreferredLocation",
        "cudaMemPrefetchAsync",
        "g_model_hmm_direct = 1",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C HMM oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_environment_selected_pageable_hmm_fallback", True),
        ("owns_prefetched_window_direct_pointer", True),
        ("owns_chunked_full_model_copy", False),
        ("owns_fd_backed_staging_policy", False),
        ("owns_cross_range_registration_disable_policy", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = sorted(set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"])))
    report.check(symbols == EXPECTED_SYMBOLS, "Rust ABI symbol implementation drift")
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(set(EXPECTED_SYMBOLS) <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "static ABI_PAGEABLE_MODEL_RANGE",
        "struct AbiPageableModelRange",
        "ReadOnlyPageableHostMemory<'static, u8>",
        "fn pageable_hmm_fallback_selected()",
        'std::env::var_os("DS4_CUDA_COPY_MODEL_CHUNKED").is_some()',
        'std::env::var_os("DS4_CUDA_NO_MODEL_COPY").is_some()',
        "fn pageable_hmm_direct_read_selected()",
        "fn try_prefetch_abi_pageable_range(",
        "prefetch_pageable_read_mostly_to_device(&pageable)",
        ".and_then(|range| range.device_ptr(model_map, model_size, offset, bytes))",
        "*ABI_PAGEABLE_MODEL_RANGE.lock().ok()? = None",
    ]:
        report.check(marker in texts["abi"], f"Rust public HMM marker missing: {marker}")
    for marker in [
        "pub const M14_6B2B2B2B2B2A_SCOPE",
        "owns_environment_selected_pageable_hmm_fallback: true",
        "owns_prefetched_window_direct_pointer: true",
        "owns_chunked_full_model_copy: false",
        "owns_fd_backed_staging_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    for marker in ["AsyncPinnedRangeCache", "stage_pinned_model_range"]:
        report.check(marker not in texts["abi"], f"fd staging overclaim in public ABI: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("ds4_gpu_set_model_map_range" in implementation.get("range_path", ""), "range path missing")
    report.check("ReadOnlyPageableHostMemory" in implementation.get("lifetime_boundary", ""), "lifetime boundary missing")
    report.check("chunked full-model copy" in implementation.get("remaining_policy_boundary", ""), "chunk copy boundary missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 96),
        ("feature_release_test_count", 98),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    probe = require_dict(report, execution.get("pageable_hmm_probe"), "pageable_hmm_probe")
    for key, expected in [
        ("range_offset", 13),
        ("range_bytes", 4096),
        ("pageable_memory_access", True),
        ("pageable_memory_access_uses_host_page_tables", False),
        ("prefetch_page_size", 4096),
        ("pageable_prefetch", True),
        ("prefetch_synchronized_for_smoke", True),
        ("hmm_direct_readback_matches", True),
    ]:
        report.check(probe.get(key) == expected, f"pageable HMM probe drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "pageable_hmm_fallback_environment_selected",
        "page_aligned_model_map",
        "prefetched_direct_read_not_reported_as_cached",
        "prefetched_window_consumed_by_weighted_rms",
        "weighted_output_matches",
        "embedded_libdevice_module_loaded",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        'setenv("DS4_CUDA_COPY_MODEL_CHUNKED", "1", 1)',
        'setenv("DS4_CUDA_NO_MODEL_COPY", "1", 1)',
        'unsetenv("DS4_CUDA_NO_MODEL_PREFETCH")',
        "ds4_gpu_set_model_map_range(model_map, model_size, offset, bytes)",
        "ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, \"pageable-hmm\") != 0",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("deterministic current-C" in value for value in risks), "deterministic subset risk missing")
    report.check(any("prefetched direct-pointer route" in value for value in risks), "window scope risk missing")
    report.check(any("fd-backed" in value for value in risks), "fd staging risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2a/abi-model-control-pageable-hmm-smoke.json"
    checker = "check_cuda_abi_model_control_pageable_hmm_smoke.py"
    item = "M14.6b2b2b2b2b2a: Pageable HMM Fallback ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbb Remaining Residual Failure Selection Policy"
        in texts["status"],
        "active item missing",
    )
    report.check("M14.6b2b2b2b2b2a Pageable HMM Fallback ABI" in texts["status"], "status evidence missing")
    report.check("M14.6b2b2b2b2b2b1 Chunk-Selected Model Copy ABI" in texts["status"], "successor status missing")
    report.check(
        "M14.6b2b2b2b2b2b2a Whole-Map Registration Precedence ABI" in texts["status"],
        "registration successor status missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b1 Buffered Fd-Backed Weight Cache ABI" in texts["status"],
        "buffered fd successor status missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b Chunked Copy And Fd-Backed Model-Control Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("HMM fallback missing", lambda value: value["ownership"].update({"owns_environment_selected_pageable_hmm_fallback": False})),
        ("direct pointer missing", lambda value: value["b300_execution"]["observed"].update({"prefetched_window_consumed_by_weighted_rms": False})),
        ("chunk copy overclaim", lambda value: value["ownership"].update({"owns_chunked_full_model_copy": True})),
        ("fd staging overclaim", lambda value: value["ownership"].update({"owns_fd_backed_staging_policy": True})),
        ("route promotion overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
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
