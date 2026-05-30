#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2a Rust CUDA basic model-control ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2a/abi-model-control-device-copy-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2a/abi_model_control_device_copy_link_smoke.c"
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
CURRENT_SUCCESSOR_SYMBOLS = ["ds4_gpu_matmul_f16_tensor", "ds4_gpu_matmul_f16_pair_tensor", "ds4_gpu_matmul_f32_tensor", "ds4_gpu_cache_q8_f16_range", "ds4_gpu_print_memory_report", "ds4_gpu_set_quality", "ds4_gpu_matmul_q8_0_tensor", "ds4_gpu_hc_expand_tensor", "ds4_gpu_hc_expand_split_tensor", "ds4_gpu_hc_expand_add_split_tensor", "ds4_gpu_matmul_q8_0_hc_expand_tensor", "ds4_gpu_shared_down_hc_expand_q8_0_tensor", "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor", "ds4_gpu_hc_weighted_sum_tensor", "ds4_gpu_hc_weighted_sum_split_tensor", "ds4_gpu_hc_split_sinkhorn_tensor", "ds4_gpu_hc_split_weighted_sum_tensor", "ds4_gpu_hc_split_weighted_sum_norm_tensor", "ds4_gpu_output_hc_weights_tensor", "ds4_gpu_embed_token_hc_tensor", "ds4_gpu_embed_tokens_hc_tensor", "ds4_gpu_head_rms_norm_tensor", "ds4_gpu_dsv4_fp8_kv_quantize_tensor", "ds4_gpu_dsv4_indexer_qat_tensor", "ds4_gpu_rope_tail_tensor", "ds4_gpu_store_raw_kv_tensor", "ds4_gpu_store_raw_kv_batch_tensor", "ds4_gpu_kv_fp8_store_raw_tensor", "ds4_gpu_compressor_store_batch_tensor", "ds4_gpu_compressor_prefill_state_ratio4_tensor", "ds4_gpu_compressor_prefill_ratio4_replay_tensor", "ds4_gpu_compressor_update_tensor", "ds4_gpu_compressor_prefill_tensor", "ds4_gpu_attention_decode_heads_tensor"]


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
    print(f"M14.6b2b2b2b2a Rust CUDA basic model-control device-copy ABI smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_model_control_device_copy_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2a", "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-basic-model-control-device-copy-abi",
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
        == [
            "ds4_gpu_set_model_map",
            "ds4_gpu_set_model_fd",
            "ds4_gpu_set_model_map_range",
            "ds4_gpu_cache_model_range",
        ],
        "oracle symbols drift",
    )
    for marker in [
        'extern "C" int ds4_gpu_set_model_map',
        'extern "C" int ds4_gpu_set_model_fd',
        'extern "C" int ds4_gpu_set_model_map_range',
        'extern "C" int ds4_gpu_cache_model_range',
        "cuda_model_range_release_all();",
        "cuda_model_range_ptr(model_map, offset, bytes",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_set_model_map_device_copy_reset", True),
        ("owns_set_model_fd_abi_acceptance", True),
        ("owns_set_model_map_range_baseline", True),
        ("owns_cache_model_range_device_copy", True),
        ("owns_registered_hmm_direct_io_policy", False),
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
        "static ABI_MODEL_CONTROL",
        "static ABI_MODEL_RANGES",
        "pub unsafe extern \"C\" fn ds4_gpu_set_model_map",
        "pub extern \"C\" fn ds4_gpu_set_model_fd",
        "pub unsafe extern \"C\" fn ds4_gpu_set_model_map_range",
        "pub unsafe extern \"C\" fn ds4_gpu_cache_model_range",
        "backend.synchronize().ok()?",
        "ABI_MODEL_RANGES.lock().ok()?.clear()",
        "with_cached_abi_model_range(backend, model_map, model_size, offset, bytes",
    ]:
        report.check(marker in texts["abi"], f"Rust model-control marker missing: {marker}")
    for marker in [
        "pub const M14_6B2B2B2B2A_SCOPE",
        "exported_abi_symbol_count: 29",
        "owns_set_model_map_device_copy_reset: true",
        "owns_set_model_fd_abi_acceptance: true",
        "owns_set_model_map_range_baseline: true",
        "owns_cache_model_range_device_copy: true",
        "owns_registered_hmm_direct_io_policy: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    for marker in ["AsyncPinnedRangeCache", "prefetch_pageable_read_only_range", "stage_pinned_model_range"]:
        report.check(marker not in texts["abi"], f"residency policy overclaim in public ABI: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("ABI_MODEL_CONTROL plus ABI_MODEL_RANGES" in implementation.get("range_path", ""), "range path missing")
    report.check("ds4_gpu_rms_norm_weight_tensor" in implementation.get("linked_compute_consumer", ""), "compute consumer missing")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "artifact retention missing")
    report.check("registered mapped-host" in implementation.get("remaining_residency_policy_boundary", ""), "policy boundary missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 94),
        ("feature_release_test_count", 96),
        ("generated_artifact_target", "sm_80"),
        ("linked_cubin_target", "sm_103"),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    report.check("std::f32::<impl f32>::exp" in execution.get("non_release_feature_test_blocker", ""), "non-release blocker missing")
    report.check("--release --features cuda-oxide-kernels" in execution.get("staticlib_build_command", ""), "staticlib build drift")
    report.check("--whole-archive" in execution.get("c_link_command", ""), "C link retention drift")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "public_set_model_fd_accepted",
        "public_set_model_map_accepted",
        "public_set_model_map_range_accepted",
        "public_cache_model_range_consumed_by_weighted_rms",
        "mapping_switch_output_matches",
        "mapping_switch_releases_prior_cached_range",
        "zero_byte_cache_noop_preserved",
        "invalid_nonzero_cache_rejected",
        "invalid_map_rejected",
        "embedded_libdevice_module_loaded",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "ds4_gpu_set_model_fd(-1)",
        "ds4_gpu_set_model_map(first_map",
        "ds4_gpu_set_model_map_range(first_map",
        "ds4_gpu_cache_model_range(first_map",
        "first_map[1 + i] = second_map[1 + i]",
        "ds4_gpu_cache_model_range(NULL, 0, 99, 0, NULL)",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("public controls currently own only" in value for value in risks), "residency policy risk missing")
    report.check(any("set_model_fd" in value for value in risks), "fd staging risk missing")
    report.check(any("retaining symbol" in value for value in risks), "retention risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2a/abi-model-control-device-copy-smoke.json"
    checker = "check_cuda_abi_model_control_device_copy_smoke.py"
    item = "M14.6b2b2b2b2a: Basic Model-Control Device-Copy ABI Export"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy" in texts["status"],
        "active item missing",
    )
    report.check("M14.6b2b2b2b2a Basic Model-Control Device-Copy ABI Export" in texts["status"], "status evidence missing")
    report.check(
        "M14.6b2b2b2b2b1 Registered Attempt And Device-Copy Fallback ABI" in texts["status"],
        "successor status missing",
    )
    report.check("M14.6b2b2b2b2b2a Pageable HMM Fallback ABI" in texts["status"], "HMM successor status missing")
    report.check("M14.6b2b2b2b2b2b1 Chunk-Selected Model Copy ABI" in texts["status"], "chunk-copy successor status missing")
    report.check(
        "M14.6b2b2b2b2b2b2a Whole-Map Registration Precedence ABI" in texts["status"],
        "whole-map registration successor status missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b1 Buffered Fd-Backed Weight Cache ABI" in texts["status"],
        "buffered fd successor status missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2b2b2b2b Registered HMM And Fd-Backed Model-Control Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("missing cache export", lambda value: value["exported_symbols"].remove("ds4_gpu_cache_model_range")),
        ("residency policy overclaim", lambda value: value["ownership"].update({"owns_registered_hmm_direct_io_policy": True})),
        ("route promotion overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("cache invalidation failure", lambda value: value["b300_execution"]["observed"].update({"mapping_switch_releases_prior_cached_range": False})),
        ("invalid map acceptance", lambda value: value["b300_execution"]["observed"].update({"invalid_map_rejected": False})),
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
