#!/usr/bin/env python3
"""Validate the Rust CUDA public attention-prefill ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-attention-prefill-smoke.json"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_attention_prefill_link_smoke.c"


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
        "cuda_c": (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8"),
        "lib": (ROOT / "rust/ds4-cuda/src/lib.rs").read_text(encoding="utf-8"),
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
        "kernels": (ROOT / "rust/ds4-cuda/src/abi_kernels.rs").read_text(encoding="utf-8"),
        "harness": HARNESS.read_text(encoding="utf-8"),
        "gpu_build": (ROOT / "rust/ds4-gpu/build.rs").read_text(encoding="utf-8"),
        "gpu_sys": (ROOT / "rust/ds4-gpu-sys/src/lib.rs").read_text(encoding="utf-8"),
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(encoding="utf-8"),
        "todo": (ROOT / ".memory/TODO.md").read_text(encoding="utf-8"),
        "status": (ROOT / ".memory/status.md").read_text(encoding="utf-8"),
        "readme": (ROOT / "ds4-parity/README.md").read_text(encoding="utf-8"),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(encoding="utf-8"),
    }
    report = ReportState()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA public attention-prefill ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_attention_prefill_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-attention-prefill-abi",
        "status drift",
    )
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == [
            "ds4_gpu_attention_prefill_raw_heads_tensor",
            "ds4_gpu_attention_prefill_static_mixed_heads_tensor",
            "ds4_gpu_attention_prefill_masked_mixed_heads_tensor",
        ],
        "oracle symbols drift",
    )
    for marker in [
        'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(',
        "static int attention_prefill_mixed_launch(",
        'extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(',
        'extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(',
        'getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL',
        'getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL',
    ]:
        report.check(marker in texts["cuda_c"], f"current-C prefill marker missing: {marker}")
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 71),
        ("exported_compute_symbol_count", 59),
        ("public_gpu_abi_function_count", 81),
        ("consumes_cached_model_ranges", True),
        ("uses_retained_cublas_scratch", True),
        ("owns_attention_prefill_raw_heads_tensor", True),
        ("owns_attention_prefill_static_mixed_heads_tensor", True),
        ("owns_attention_prefill_masked_mixed_heads_tensor", True),
        ("preserves_generic_online_and_cublas_dispatch", True),
        ("owns_complete_attention_abi", True),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 74, "Rust ABI export implementation count drift")
    for symbol in oracle_symbols():
        report.check(symbol in symbols, f"public prefill export missing: {symbol}")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "attention_prefill_cublas_impl(",
        "select_attention_prefill_path(AttentionPrefillDispatchOptions",
        "ABI_ATTENTION_PREFILL_CUBLAS_SCRATCH",
        "ds4_gpu_attention_prefill_raw_heads_tensor",
        "ds4_gpu_attention_prefill_static_mixed_heads_tensor",
        "ds4_gpu_attention_prefill_masked_mixed_heads_tensor",
    ]:
        report.check(marker in texts["abi"], f"Rust ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_attention_prefill_raw_kernel",
        "pub fn abi_attention_prefill_mixed_kernel",
        "pub fn abi_attention_static_mixed_heads8_online_kernel",
        "pub fn abi_attention_prefill_pack_mixed_kv_kernel",
        "pub fn abi_attention_prefill_pack_q_heads_kernel",
        "pub fn abi_attention_prefill_replicate_kv_kernel",
        "pub fn abi_attention_prefill_raw_softmax_kernel",
        "pub fn abi_attention_prefill_mixed_softmax_kernel",
        "pub fn abi_attention_prefill_unpack_heads_kernel",
        '.load_function("abi_attention_prefill_unpack_heads_kernel")',
    ]:
        report.check(marker in texts["kernels"], f"Rust kernel marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiAttentionPrefillScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 71",
        "exported_compute_symbol_count: 59",
        "owns_complete_attention_abi: true",
        "owns_remaining_graph_compute_abi: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    entries = implementation.get("embedded_kernel_entries", [])
    for marker in [
        "abi_attention_prefill_raw_kernel",
        "abi_attention_prefill_mixed_kernel",
        "abi_attention_static_mixed_heads8_online_kernel",
        "abi_attention_prefill_pack_mixed_kv_kernel",
        "abi_attention_prefill_pack_q_heads_kernel",
        "abi_attention_prefill_replicate_kv_kernel",
        "abi_attention_prefill_raw_softmax_kernel",
        "abi_attention_prefill_mixed_softmax_kernel",
        "abi_attention_prefill_unpack_heads_kernel",
    ]:
        report.check(marker in entries, f"embedded kernel entry missing: {marker}")
    report.check("safe SGEMM" in implementation.get("adapter_boundary", ""), "SGEMM adapter boundary missing")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage path missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 152),
        ("feature_release_test_count", 159),
        ("staticlib_export_count", 71),
        ("embedded_kernel_count", 59),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "generic_raw_output_matches",
        "raw_zero_window_generic_sink_only_matches",
        "static_mixed_zero_window_matches",
        "masked_mixed_output_matches",
        "online_static_output_matches",
        "cublas_static_output_matches",
        "masked_cublas_output_matches",
        "raw_zero_window_optimized_full_causal_matches",
        "invalid_range_preserves_output",
        "invalid_shape_rejected",
        "null_rejected",
        "embedded_prefill_kernels_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 62, "predecessor count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 62, "warning count drift")
    for marker in [
        "#define G_TOKENS 4u",
        "#define O_DIM 512u",
        "ds4_gpu_attention_prefill_raw_heads_tensor(",
        "ds4_gpu_attention_prefill_static_mixed_heads_tensor(",
        "ds4_gpu_attention_prefill_masked_mixed_heads_tensor(",
        'setenv("DS4_CUDA_WINDOW_ATTENTION"',
        "raw_zero_window_optimized_full_causal_matches",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("zero-window" in value for value in risks), "raw zero-window risk missing")
    report.check(any("safe SGEMM" in value for value in risks), "SGEMM adapter risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-attention-prefill-smoke.json"
    checker = "check_cuda_abi_attention_prefill_smoke.py"
    item = f"{MILESTONE}: Public Attention Prefill ABI"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active remainder status missing")
    report.check(
        fixture.get("review", {}).get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S",
        "pre-implementation review evidence missing",
    )
    report.check(
        fixture.get("review", {}).get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S",
        "final review timeout evidence missing",
    )
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("numeric mismatch", lambda value: value["b300_execution"]["observed"].update({"generic_raw_output_matches": False})),
        ("zero-window mismatch", lambda value: value["b300_execution"]["observed"].update({"raw_zero_window_optimized_full_causal_matches": False})),
        ("ownership overclaim", lambda value: value["ownership"].update({"owns_remaining_graph_compute_abi": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def oracle_symbols() -> list[str]:
    return [
        "ds4_gpu_attention_prefill_raw_heads_tensor",
        "ds4_gpu_attention_prefill_static_mixed_heads_tensor",
        "ds4_gpu_attention_prefill_masked_mixed_heads_tensor",
    ]


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
