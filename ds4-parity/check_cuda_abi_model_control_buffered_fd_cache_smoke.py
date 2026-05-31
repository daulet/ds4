#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b1 buffered fd-backed model-cache ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b1/abi-model-control-buffered-fd-cache-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b1/abi_model_control_buffered_fd_cache_link_smoke.c"
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
CURRENT_SUCCESSOR_SYMBOLS = ["ds4_gpu_matmul_f16_tensor", "ds4_gpu_matmul_f16_pair_tensor", "ds4_gpu_matmul_f32_tensor", "ds4_gpu_cache_q8_f16_range", "ds4_gpu_print_memory_report", "ds4_gpu_set_quality", "ds4_gpu_matmul_q8_0_tensor", "ds4_gpu_hc_expand_tensor", "ds4_gpu_hc_expand_split_tensor", "ds4_gpu_hc_expand_add_split_tensor", "ds4_gpu_matmul_q8_0_hc_expand_tensor", "ds4_gpu_shared_down_hc_expand_q8_0_tensor", "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor", "ds4_gpu_hc_weighted_sum_tensor", "ds4_gpu_hc_weighted_sum_split_tensor", "ds4_gpu_hc_split_sinkhorn_tensor", "ds4_gpu_hc_split_weighted_sum_tensor", "ds4_gpu_hc_split_weighted_sum_norm_tensor", "ds4_gpu_output_hc_weights_tensor", "ds4_gpu_embed_token_hc_tensor", "ds4_gpu_embed_tokens_hc_tensor", "ds4_gpu_head_rms_norm_tensor", "ds4_gpu_dsv4_fp8_kv_quantize_tensor", "ds4_gpu_dsv4_indexer_qat_tensor", "ds4_gpu_rope_tail_tensor", "ds4_gpu_store_raw_kv_tensor", "ds4_gpu_store_raw_kv_batch_tensor", "ds4_gpu_kv_fp8_store_raw_tensor", "ds4_gpu_compressor_store_batch_tensor", "ds4_gpu_compressor_prefill_state_ratio4_tensor", "ds4_gpu_compressor_prefill_ratio4_replay_tensor", "ds4_gpu_compressor_update_tensor", "ds4_gpu_compressor_prefill_tensor", "ds4_gpu_attention_decode_heads_tensor", "ds4_gpu_attention_decode_raw_batch_heads_tensor", "ds4_gpu_attention_decode_mixed_batch_heads_tensor", "ds4_gpu_attention_indexed_mixed_batch_heads_tensor", "ds4_gpu_attention_output_low_q8_tensor", "ds4_gpu_attention_output_q8_batch_tensor", "ds4_gpu_attention_prefill_raw_heads_tensor", "ds4_gpu_attention_prefill_static_mixed_heads_tensor", "ds4_gpu_attention_prefill_masked_mixed_heads_tensor"]


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
    print(f"M14.6b2b2b2b2b2b2b1 Rust CUDA buffered fd cache ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_buffered_fd_cache_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-staticlib-buffered-fd-cache-abi", "status drift")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
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
        'if (getenv("DS4_CUDA_NO_FD_CACHE") == NULL)',
        "static const char *cuda_model_range_ptr_from_fd(",
        "if (g_model_fd < 0 || bytes == 0) return NULL;",
        "g_model_fd_host_base",
        "return cuda_pread_full(g_model_fd, stage, bytes, offset);",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C buffered fd oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_buffered_fd_weight_cache_path", True),
        ("owns_model_fd_host_base_binding", True),
        ("owns_direct_io_fd_reopen_policy", False),
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
        "BufferedFdDeviceCopy",
        "fn fd_weight_cache_selected() -> bool",
        "fn buffered_fd_weight_cache_selected() -> bool",
        'std::env::var_os("DS4_CUDA_NO_DIRECT_IO").is_some()',
        'std::env::var_os("DS4_CUDA_NO_FD_CACHE").is_none()',
        "!direct_model_read_selected()",
        "fn try_upload_abi_buffered_fd_range(",
        "libc::pread(",
        "Some(libc::EINTR)",
        "backend.upload_pinned_u8_range(&staging, 0, bytes)",
        "model_fd_host_base",
        "control.model_fd_host_base = model_map as usize",
        "control.model_fd_host_base = control.model_map",
    ]:
        report.check(marker in texts["abi"], f"Rust buffered fd marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiBufferedFdCacheScope",
        "pub const M14_6B2B2B2B2B2B2B1_SCOPE",
        "owns_buffered_fd_weight_cache_path: true",
        "owns_model_fd_host_base_binding: true",
        "owns_direct_io_fd_reopen_policy: false",
        "owns_async_fd_staging_ring: false",
        "owns_fd_cache_budget_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check("AsyncPinnedRangeCache" not in texts["abi"], "async fd staging overclaim in public ABI")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 99),
        ("feature_release_test_count", 101),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "page_aligned_host_map",
        "fd_before_map_binds_host_base",
        "buffered_fd_weight_cache_selected",
        "fd_bytes_precede_mutated_host_map",
        "repeated_cache_reuses_fd_device_copy",
        "weighted_output_matches",
        "embedded_libdevice_module_loaded",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "mkstemp",
        "pwrite",
        'setenv("DS4_CUDA_WEIGHT_CACHE", "1", 1)',
        'setenv("DS4_CUDA_NO_DIRECT_IO", "1", 1)',
        'unsetenv("DS4_CUDA_NO_FD_CACHE")',
        "ds4_gpu_set_model_fd(fd)",
        "ds4_gpu_set_model_map(model_map, model_size)",
        "memcpy(model_map + offset, host_weights",
        "memcpy(file_image + offset, file_weights",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    report.check(
        texts["harness"].count("ds4_gpu_cache_model_range(model_map, model_size, offset, bytes") == 2,
        "C-linked harness does not exercise cached fd range reuse",
    )
    risks = fixture.get("integration_risks", [])
    report.check(any("whole-map registration rejection" in value for value in risks), "registration precedence caveat missing")
    report.check(any("direct-I/O" in value for value in risks), "direct I/O risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b1/abi-model-control-buffered-fd-cache-smoke.json"
    checker = "check_cuda_abi_model_control_buffered_fd_cache_smoke.py"
    item = "M14.6b2b2b2b2b2b2b1: Buffered Fd-Backed Weight Cache ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy" in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b1 Buffered Fd-Backed Weight Cache ABI" in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2b2b2b2b2b2b2 Direct-I/O And Residual Model-Control Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("buffered fd path missing", lambda value: value["ownership"].update({"owns_buffered_fd_weight_cache_path": False})),
        ("fd binding missing", lambda value: value["ownership"].update({"owns_model_fd_host_base_binding": False})),
        ("direct IO overclaim", lambda value: value["ownership"].update({"owns_direct_io_fd_reopen_policy": True})),
        ("async ring overclaim", lambda value: value["ownership"].update({"owns_async_fd_staging_ring": True})),
        ("budget overclaim", lambda value: value["ownership"].update({"owns_fd_cache_budget_policy": True})),
        ("fd output mismatch", lambda value: value["b300_execution"]["observed"].update({"fd_bytes_precede_mutated_host_map": False})),
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
