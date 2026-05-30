#!/usr/bin/env python3
"""Validate the M14.6b1 Rust CUDA resource ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b1/abi-resource-smoke.json"
CUDA_CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/abi_resource_smoke.rs"
GPU_CARGO = ROOT / "rust/ds4-gpu/Cargo.toml"
GPU_BUILD = ROOT / "rust/ds4-gpu/build.rs"
GPU_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_SYMBOLS = [
    "ds4_gpu_begin_commands",
    "ds4_gpu_cleanup",
    "ds4_gpu_end_commands",
    "ds4_gpu_flush_commands",
    "ds4_gpu_init",
    "ds4_gpu_should_use_managed_kv_cache",
    "ds4_gpu_synchronize",
    "ds4_gpu_tensor_alloc",
    "ds4_gpu_tensor_alloc_managed",
    "ds4_gpu_tensor_bytes",
    "ds4_gpu_tensor_contents",
    "ds4_gpu_tensor_copy",
    "ds4_gpu_tensor_free",
    "ds4_gpu_tensor_read",
    "ds4_gpu_tensor_view",
    "ds4_gpu_tensor_write",
]
CURRENT_SUCCESSOR_SYMBOLS = [
    "ds4_gpu_tensor_fill_f32",
    "ds4_gpu_add_tensor",
    "ds4_gpu_repeat_hc_tensor",
    "ds4_gpu_directional_steering_project_tensor",
    "ds4_gpu_swiglu_tensor",
    "ds4_gpu_rms_norm_plain_rows_tensor",
    "ds4_gpu_rms_norm_plain_tensor",
    "ds4_gpu_rms_norm_weight_rows_tensor",
    "ds4_gpu_rms_norm_weight_tensor",
    "ds4_gpu_cache_model_range",
    "ds4_gpu_set_model_fd",
    "ds4_gpu_set_model_map",
    "ds4_gpu_set_model_map_range",
    "ds4_gpu_matmul_f16_tensor",
    "ds4_gpu_matmul_f16_pair_tensor", "ds4_gpu_matmul_f32_tensor", "ds4_gpu_cache_q8_f16_range", "ds4_gpu_print_memory_report", "ds4_gpu_set_quality", "ds4_gpu_matmul_q8_0_tensor",
    "ds4_gpu_hc_expand_tensor",
    "ds4_gpu_hc_expand_split_tensor",
    "ds4_gpu_hc_expand_add_split_tensor",
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
        "cargo": CUDA_CARGO.read_text(encoding="utf-8"),
        "lib": CUDA_LIB.read_text(encoding="utf-8"),
        "abi": CUDA_ABI.read_text(encoding="utf-8"),
        "smoke": SMOKE.read_text(encoding="utf-8"),
        "gpu_cargo": GPU_CARGO.read_text(encoding="utf-8"),
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
    print(f"M14.6b1 Rust CUDA resource ABI smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_resource_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass-partial-production-abi", "status drift")
    report.check(fixture.get("exported_symbols") == EXPECTED_SYMBOLS, "exported symbol fixture drift")
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_resource_symbol_count", 16),
        ("public_gpu_abi_function_count", 81),
        ("owns_initialization", True),
        ("owns_tensor_storage", True),
        ("owns_host_device_copies", True),
        ("owns_command_synchronization", True),
        ("owns_managed_kv_policy", True),
        ("owns_tensor_fill_kernel", False),
        ("owns_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = sorted(set(re.findall(r"pub (?:unsafe )?extern \"C\" fn (ds4_gpu_[A-Za-z0-9_]+)", texts["abi"])))
    report.check(
        symbols == sorted(EXPECTED_SYMBOLS + CURRENT_SUCCESSOR_SYMBOLS),
        "Rust ABI symbol successor progression drift",
    )
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(set(EXPECTED_SYMBOLS) <= ffi_symbols, "resource symbols do not match public GPU ABI")
    for marker in [
        'crate-type = ["rlib", "staticlib"]',
        'name = "ds4-cuda-abi-resource-smoke"',
        'required-features = ["cuda-oxide-backend"]',
    ]:
        report.check(marker in texts["cargo"], f"Cargo marker missing: {marker}")
    for marker in [
        "pub mod abi;",
        "pub const M14_6B1_SCOPE",
        "exported_resource_symbol_count: 16",
        "owns_compute_abi: false",
        "owns_complete_ds4_gpu_abi: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check("ds4-cuda" not in texts["gpu_cargo"], "production GPU crate prematurely links Rust CUDA")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("feature_test_count", 89),
        ("local_library_test_count", 87),
        ("local_cuda_feature_build_blocker", "/usr/local/cuda/include/cuda.h"),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    report.check("--bin ds4-cuda-abi-resource-smoke" in execution.get("smoke_command", ""), "smoke command drift")
    report.check("target/debug/libds4_cuda.a" in execution.get("staticlib_command", ""), "staticlib command drift")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key, expected in [
        ("milestone", "M14.6b1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("rust_exported_resource_abi", True),
        ("exported_resource_symbol_count", 16),
        ("initialization_roundtrip", True),
        ("device_tensor_roundtrip", True),
        ("managed_tensor_roundtrip", True),
        ("view_roundtrip", True),
        ("device_copy_roundtrip", True),
        ("self_copy_identity_matches", True),
        ("zero_alloc_is_one_byte", True),
        ("invalid_range_rejected", True),
        ("null_write_rejected", True),
        ("managed_kv_policy_matches", True),
        ("command_sync_matches", True),
        ("owns_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
    ]:
        report.check(observed.get(key) == expected, f"observed smoke drift: {key}")
    for marker in [
        "ds4_gpu_tensor_alloc_managed",
        "ds4_gpu_tensor_copy",
        "ds4_gpu_should_use_managed_kv_cache",
        "M14_6B1_SCOPE.owns_complete_ds4_gpu_abi",
    ]:
        report.check(marker in texts["smoke"], f"smoke source marker missing: {marker}")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b1/abi-resource-smoke.json"
    checker = "check_cuda_abi_resource_smoke.py"
    item = "M14.6b1: Rust CUDA Resource ABI Exports"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        any(
            marker in texts["status"]
            for marker in [
                "Active item: M14.6b2 Rust CUDA Compute ABI Assembly",
                "Active item: M14.6b2b2 Remaining Rust CUDA Kernel ABI Assembly",
                "M14.6b2b2a Directional Steering ABI Export",
                "M14.6b2b2b1 SwiGLU Libdevice ABI Export",
                "M14.6b2b2b2a Plain RMS Norm ABI Export",
                "M14.6b2b2b2b1 Weighted RMS Device-Copy ABI Export",
            ]
        ),
        "active item missing",
    )
    report.check("M14.6b1 Rust CUDA Resource ABI Exports" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage") == "M14.6b2 Rust CUDA Compute ABI Assembly",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("complete ABI overclaim", lambda value: value["ownership"].update({"owns_complete_ds4_gpu_abi": True})),
        ("route promotion overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("missing exported symbol", lambda value: value["exported_symbols"].pop()),
        ("failed device copy", lambda value: value["b300_execution"]["observed"].update({"device_copy_roundtrip": False})),
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
