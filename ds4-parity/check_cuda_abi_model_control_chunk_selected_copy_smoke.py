#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b1 public chunk-selected model-copy ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b1/abi-model-control-chunk-selected-copy-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b1/abi_model_control_chunk_selected_copy_link_smoke.c"
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
    print(f"M14.6b2b2b2b2b2b1 Rust CUDA chunk-selected model-copy ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_chunk_selected_copy_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b1", "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-chunk-selected-copy-abi",
        "status drift",
    )
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
        == ["cuda_model_copy_chunked", "ds4_gpu_set_model_map_range", "cuda_model_range_ptr"],
        "oracle symbols drift",
    )
    for marker in [
        "static int cuda_model_copy_chunked(",
        'getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL',
        'getenv("DS4_CUDA_NO_MODEL_COPY") != NULL',
        "g_model_device_owned = 1",
        "if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C copy oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_chunk_selected_device_image", True),
        ("owns_bounded_pinned_copy_transfers", True),
        ("owns_whole_map_registration_precedence", False),
        ("owns_copy_allocation_failure_to_hmm_fallback", False),
        ("owns_model_copy_chunk_override_policy", False),
        ("owns_fd_backed_staging_policy", False),
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
        "static ABI_COPIED_MODEL",
        "struct AbiCopiedModel",
        "fn matches(&self, model_map: *const c_void, model_size: u64) -> bool",
        "fn chunk_selected_model_copy_selected()",
        "fn try_copy_abi_model_window(",
        "const ABI_MODEL_COPY_CHUNK_BYTES: usize = 64 * 1024 * 1024;",
        "backend.pinned_zeroed::<u8>",
        "backend.enqueue_pinned_u8_range_async(&device, copied, &staging, 0, bytes)",
        ".and_then(|model| model.device_ptr(model_map, model_size, offset, bytes))",
        "*ABI_COPIED_MODEL.lock().ok()? = None",
    ]:
        report.check(marker in texts["abi"], f"Rust chunk-selected copy marker missing: {marker}")
    for marker in [
        "pub const M14_6B2B2B2B2B2B1_SCOPE",
        "owns_chunk_selected_device_image: true",
        "owns_bounded_pinned_copy_transfers: true",
        "owns_whole_map_registration_precedence: false",
        "owns_copy_allocation_failure_to_hmm_fallback: false",
        "owns_model_copy_chunk_override_policy: false",
        "owns_fd_backed_staging_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check("AsyncPinnedRangeCache" not in texts["abi"], "fd-backed cache overclaim in public ABI")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 97),
        ("feature_release_test_count", 99),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "chunk_selected_device_image_retained",
        "repeated_map_range_reuses_device_image",
        "host_mutation_after_map_range_ignored",
        "cached_weighted_rms_reads_copied_image",
        "weighted_output_matches",
        "embedded_libdevice_module_loaded",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        'setenv("DS4_CUDA_COPY_MODEL_CHUNKED", "1", 1)',
        'unsetenv("DS4_CUDA_NO_MODEL_COPY")',
        'unsetenv("DS4_CUDA_DIRECT_MODEL")',
        "ds4_gpu_set_model_map_range(model_map, sizeof(model_map), offset, bytes)",
        "model_map[1 + i] = changed[i]",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    report.check(
        texts["harness"].count("ds4_gpu_set_model_map_range(model_map, sizeof(model_map), offset, bytes)") == 2,
        "C-linked harness does not exercise repeated map-range reuse",
    )
    risks = fixture.get("integration_risks", [])
    report.check(any("allocation" in value for value in risks), "copy failure risk missing")
    report.check(any("whole-map registration" in value for value in risks), "registration precedence risk missing")
    report.check(any("DS4_CUDA_MODEL_COPY_CHUNK_MB" in value for value in risks), "copy knob risk missing")
    report.check(any("fd-backed" in value for value in risks), "fd staging risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b1/abi-model-control-chunk-selected-copy-smoke.json"
    checker = "check_cuda_abi_model_control_chunk_selected_copy_smoke.py"
    item = "M14.6b2b2b2b2b2b1: Chunk-Selected Model Copy ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2b Remaining Residual Failure Selection Policy"
        in texts["status"],
        "active item missing",
    )
    report.check("M14.6b2b2b2b2b2b1 Chunk-Selected Model Copy ABI" in texts["status"], "status evidence missing")
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
        == "M14.6b2b2b2b2b2b2 Fd-Backed And Remaining Model-Control Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("device image missing", lambda value: value["ownership"].update({"owns_chunk_selected_device_image": False})),
        ("pinned transfer missing", lambda value: value["ownership"].update({"owns_bounded_pinned_copy_transfers": False})),
        ("registration precedence overclaim", lambda value: value["ownership"].update({"owns_whole_map_registration_precedence": True})),
        ("failure fallback overclaim", lambda value: value["ownership"].update({"owns_copy_allocation_failure_to_hmm_fallback": True})),
        ("fd staging overclaim", lambda value: value["ownership"].update({"owns_fd_backed_staging_policy": True})),
        ("output mismatch", lambda value: value["b300_execution"]["observed"].update({"weighted_output_matches": False})),
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
