#!/usr/bin/env python3
"""Validate the Rust CUDA public low-Q8 attention output ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-attention-output-low-q8-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_attention_output_low_q8_link_smoke.c"
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
    texts = {
        "cuda_c": CUDA_C.read_text(encoding="utf-8"),
        "lib": CUDA_LIB.read_text(encoding="utf-8"),
        "abi": CUDA_ABI.read_text(encoding="utf-8"),
        "kernels": CUDA_KERNELS.read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA public low-Q8 attention output ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_attention_output_low_q8_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-attention-output-low-q8-abi",
        "status drift",
    )
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["ds4_gpu_attention_output_low_q8_tensor"], "oracle symbols drift")
    for marker in [
        'extern "C" int ds4_gpu_attention_output_low_q8_tensor(',
        "quantize_q8_0_f32_kernel<<<qgrid, 32>>>",
        "grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>",
        "const int use_dp4a = cuda_q8_use_dp4a();",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C low-Q8 marker missing: {marker}")
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 67),
        ("exported_compute_symbol_count", 46),
        ("public_gpu_abi_function_count", 81),
        ("consumes_cached_model_ranges", True),
        ("reuses_q8_activation_scratch", True),
        ("owns_attention_output_low_q8_tensor", True),
        ("owns_grouped_q8_output_a_kernel", True),
        ("owns_attention_output_q8_batch_tensor", False),
        ("owns_remaining_attention_abi", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 74, "Rust ABI export implementation count drift")
    report.check("ds4_gpu_attention_output_low_q8_tensor" in symbols, "low-Q8 export missing")
    report.check("ds4_gpu_attention_output_q8_batch_tensor" in symbols, "batch output successor missing")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        'pub unsafe extern "C" fn ds4_gpu_attention_output_low_q8_tensor',
        "with_cached_abi_model_range(",
        "with_abi_q8_activations(",
        "kernels.quantize_q8_f32_tensor(",
        "kernels.attention_output_low_q8_tensor(",
        'std::env::var_os("DS4_CUDA_NO_Q8_DP4A").is_some()',
    ]:
        report.check(marker in texts["abi"], f"Rust ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_quantize_q8_0_f32_kernel",
        "pub fn abi_grouped_q8_0_a_preq_warp8_kernel",
        "grouped_q8_0_a_preq_warp8_kernel: CudaFunction",
        '.load_function("abi_grouped_q8_0_a_preq_warp8_kernel")',
        "pub(crate) unsafe fn attention_output_low_q8_tensor(",
    ]:
        report.check(marker in texts["kernels"], f"Rust kernel marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiAttentionOutputLowQ8Scope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 67",
        "exported_compute_symbol_count: 46",
        "reuses_q8_activation_scratch: true",
        "owns_attention_output_low_q8_tensor: true",
        "owns_attention_output_q8_batch_tensor: false",
        "owns_remaining_attention_abi: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(
        implementation.get("embedded_kernel_entries")
        == ["abi_quantize_q8_0_f32_kernel", "abi_grouped_q8_0_a_preq_warp8_kernel"],
        "embedded low-Q8 kernel entries drift",
    )
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage path missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 150),
        ("feature_release_test_count", 157),
        ("staticlib_export_count", 67),
        ("embedded_kernel_count", 46),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "low_q8_output_matches",
        "partial_q8_block_matches",
        "dp4a_environment_gate_matches",
        "invalid_model_range_preserves_output",
        "invalid_shape_rejected",
        "null_rejected",
        "embedded_grouped_q8_output_kernel_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 60, "predecessor count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 60, "warning count drift")
    for marker in [
        "#define GROUP_DIM 35u",
        "ds4_gpu_attention_output_low_q8_tensor(",
        'setenv("DS4_CUDA_NO_Q8_DP4A"',
        "partial_q8_block_matches",
        "invalid_model_range_preserves_output",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("retained Q8 scratch" in value for value in risks), "Q8 scratch risk missing")
    report.check(any("ds4_gpu_attention_output_q8_batch_tensor" in value for value in risks), "batch deferral risk missing")
    report.check(any("prefill attention" in value for value in risks), "prefill deferral risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-attention-output-low-q8-smoke.json"
    checker = "check_cuda_abi_attention_output_low_q8_smoke.py"
    item = f"{MILESTONE}: Public Low-Q8 Attention Output ABI"
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
        ("numeric mismatch", lambda value: value["b300_execution"]["observed"].update({"low_q8_output_matches": False})),
        ("DP4A gate mismatch", lambda value: value["b300_execution"]["observed"].update({"dp4a_environment_gate_matches": False})),
        ("batch output overclaim", lambda value: value["ownership"].update({"owns_attention_output_q8_batch_tensor": True})),
        ("attention overclaim", lambda value: value["ownership"].update({"owns_remaining_attention_abi": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
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
