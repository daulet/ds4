#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2a direct-I/O fd-cache public ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2a/abi-model-control-direct-io-fd-cache-smoke.json"
DIRECT_PROBE = ROOT / "ds4-parity/baselines/backend/m14.1b2b3b1/model-direct-io-smoke.json"
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
CURRENT_SUCCESSOR_SYMBOLS = ["ds4_gpu_matmul_f16_tensor", "ds4_gpu_matmul_f16_pair_tensor", "ds4_gpu_matmul_f32_tensor"]


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
    direct_probe = json.loads(DIRECT_PROBE.read_text(encoding="utf-8"))
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
    validate(report, fixture, direct_probe, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, direct_probe, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"M14.6b2b2b2b2b2b2b2a Rust CUDA direct-I/O fd cache ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    direct_probe: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_direct_io_fd_cache_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2a", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-direct-io-fd-cache-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, direct_probe, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == ["ds4_gpu_set_model_fd", "cuda_model_range_ptr", "cuda_model_range_ptr_from_fd", "cuda_model_stage_read"],
        "oracle symbols drift",
    )
    for marker in [
        'extern "C" int ds4_gpu_set_model_fd',
        'if (getenv("DS4_CUDA_NO_DIRECT_IO") == NULL)',
        "O_RDONLY | O_DIRECT",
        "g_model_direct_align",
        "g_model_file_size",
        "static int cuda_model_stage_read",
        "return cuda_pread_full(g_model_fd, stage, bytes, offset);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C direct-I/O oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_direct_io_fd_reopen_policy", True),
        ("owns_aligned_direct_pinned_read", True),
        ("owns_buffered_fallback_when_direct_unavailable", True),
        ("owns_persistent_direct_io_disable_after_error", False),
        ("owns_async_fd_staging_ring", False),
        ("owns_fd_cache_budget_policy", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = sorted(set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"])))
    report.check(
        symbols == sorted(EXPECTED_SYMBOLS + CURRENT_SUCCESSOR_SYMBOLS),
        "Rust ABI symbol implementation drift",
    )
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(set(EXPECTED_SYMBOLS) <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "DirectIoFdDeviceCopy",
        "fn direct_io_fd_weight_cache_selected() -> bool",
        'std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_none()',
        "fn try_upload_abi_direct_fd_range(",
        "model_direct_file",
        'format!("/proc/self/fd/{fd}")',
        ".custom_flags(libc::O_DIRECT)",
        "read_exact_at(direct_window, read_offset)",
        "read_abi_buffered_fd_into(fd, offset",
        "configure_abi_model_fd(&mut control, fd)",
    ]:
        report.check(marker in texts["abi"], f"Rust direct-I/O marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiDirectIoFdCacheScope",
        "pub const M14_6B2B2B2B2B2B2B2A_SCOPE",
        "owns_direct_io_fd_reopen_policy: true",
        "owns_aligned_direct_pinned_read: true",
        "owns_buffered_fallback_when_direct_unavailable: true",
        "owns_persistent_direct_io_disable_after_error: false",
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
    direct_probe: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 100),
        ("feature_release_test_count", 102),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "page_aligned_host_map",
        "fd_before_map_binds_host_base",
        "direct_io_permitted_by_environment",
        "fd_bytes_precede_mutated_host_map",
        "repeated_cache_reuses_fd_device_copy",
        "weighted_output_matches",
        "embedded_libdevice_module_loaded",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    probe = require_dict(report, execution.get("direct_selection_probe"), "direct_selection_probe")
    baseline_stdout = require_dict(report, require_dict(report, direct_probe.get("b300_execution"), "probe execution").get("stdout"), "probe stdout")
    for key, expected in [
        ("direct_io_selected", True),
        ("direct_io_alignment", 4096),
        ("direct_io_read_offset", 0),
        ("direct_io_read_bytes", 8192),
        ("direct_io_payload_offset", 13),
        ("direct_io_readback_matches", True),
        ("tail_buffered_fallback", True),
        ("tail_fallback_readback_matches", True),
    ]:
        report.check(probe.get(key) == expected, f"direct probe fixture drift: {key}")
        report.check(baseline_stdout.get(key) == expected, f"direct probe baseline drift: {key}")
    for marker in [
        'unsetenv("DS4_CUDA_NO_DIRECT_IO")',
        'setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1)',
        "ds4_gpu_set_model_fd(fd)",
        "ds4_gpu_set_model_map(model_map, model_size)",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    report.check("direct_io_selected" not in texts["harness"], "public harness overclaims unobservable direct selection")
    risks = fixture.get("integration_risks", [])
    report.check(any("does not independently expose" in value for value in risks), "public observability caveat missing")
    report.check(any("persistent direct-read error disablement" in value for value in risks), "residual policy risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2a/abi-model-control-direct-io-fd-cache-smoke.json"
    checker = "check_cuda_abi_model_control_direct_io_fd_cache_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2a: Direct-I/O Fd Cache ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy" in texts["status"],
        "active item missing",
    )
    report.check("M14.6b2b2b2b2b2b2b2a Direct-I/O Fd Cache ABI" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2b2b2b2b2b2b2b Direct-I/O Residual Failure And Cache Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    direct_probe: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("direct fd reopen missing", lambda value: value["ownership"].update({"owns_direct_io_fd_reopen_policy": False})),
        ("aligned direct read missing", lambda value: value["ownership"].update({"owns_aligned_direct_pinned_read": False})),
        ("error disable overclaim", lambda value: value["ownership"].update({"owns_persistent_direct_io_disable_after_error": True})),
        ("async ring overclaim", lambda value: value["ownership"].update({"owns_async_fd_staging_ring": True})),
        ("public output mismatch", lambda value: value["b300_execution"]["observed"].update({"fd_bytes_precede_mutated_host_map": False})),
        ("direct selection mismatch", lambda value: value["b300_execution"]["direct_selection_probe"].update({"direct_io_selected": False})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, direct_probe, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
