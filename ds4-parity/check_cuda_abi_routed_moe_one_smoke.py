#!/usr/bin/env python3
"""Validate the Rust CUDA public single-token routed-MoE ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-one-smoke.json"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_routed_moe_one_link_smoke.c"


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
        "gpu_h": (ROOT / "ds4_gpu.h").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA public single-token routed-MoE ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_routed_moe_one_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-single-token-routed-moe-abi",
        "status drift",
    )
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["ds4_gpu_routed_moe_one_tensor"], "oracle symbols drift")
    for marker in [
        "static int routed_moe_launch(",
        'extern "C" int ds4_gpu_routed_moe_one_tensor(',
        "moe_gate_up_mid_f32_kernel<<<",
        "moe_gate_up_mid_decode_lut_qwarp32_kernel<<<",
        "moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<",
        "moe_down_sum6_qwarp32_kernel<<<",
        "moe_down_q4K_sum6_qwarp32_kernel<<<",
        'getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL',
        'getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL',
    ]:
        report.check(marker in texts["cuda_c"], f"current-C routed-MoE marker missing: {marker}")
    validate_contract_gap(report, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_contract_gap(report: ReportState, texts: dict[str, str]) -> None:
    batch_c = re.search(
        r'extern "C" int ds4_gpu_routed_moe_batch_tensor\((.*?)\) \{',
        texts["cuda_c"],
        re.S,
    )
    report.check(batch_c is not None, "current-C batched routed-MoE definition missing")
    c_args = batch_c.group(1) if batch_c else ""
    report.check("mid_is_f16" not in c_args, "current-C batch unexpectedly implements mid_is_f16")
    report.check("bool                   *mid_is_f16" in texts["gpu_h"], "public header gap marker missing")
    report.check("mid_is_f16: *mut bool" in texts["gpu_sys"], "Rust FFI gap marker missing")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 74),
        ("exported_compute_symbol_count", 72),
        ("public_gpu_abi_function_count", 81),
        ("consumes_cached_model_ranges", True),
        ("owns_routed_moe_one_tensor", True),
        ("preserves_f32_quantized_q4_k_and_environment_dispatch", True),
        ("preserves_packed_q8_k_public_scratch_aliasing", True),
        ("defers_batch_mid_is_f16_contract_gap", True),
        ("owns_routed_moe_batch_tensor", False),
        ("owns_complete_routed_moe_abi", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 74, "Rust ABI export implementation count drift")
    report.check("ds4_gpu_routed_moe_one_tensor" in symbols, "public routed-MoE export missing")
    report.check("ds4_gpu_routed_moe_batch_tensor" not in symbols, "batched routed-MoE exported early")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        'pub unsafe extern "C" fn ds4_gpu_routed_moe_one_tensor',
        '"DS4_CUDA_MOE_NO_DECODE_LUT_GATE"',
        '"DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6"',
        '"DS4_CUDA_MOE_WRITE_GATE_UP"',
        "with_abi_moe_iq2_tables",
        "ABI_MOE_IQ2_TABLES",
        "Q8_K_BYTES",
    ]:
        report.check(marker in texts["abi"], f"Rust ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_moe_q8_k_quantize_kernel",
        "pub fn abi_moe_gate_up_mid_f32_kernel",
        "pub fn abi_moe_gate_up_mid_qwarp32_kernel",
        "pub fn abi_moe_gate_up_mid_decode_lut_qwarp32_kernel",
        "pub fn abi_moe_gate_up_mid_decode_q4_k_qwarp32_kernel",
        "pub fn abi_moe_down_qwarp32_kernel",
        "pub fn abi_moe_down_sum6_qwarp32_kernel",
        "pub fn abi_moe_down_q4_k_sum6_qwarp32_kernel",
        "pub fn abi_moe_sum_kernel",
        "ABI_MOE_IQ2_GRID",
        "ABI_MOE_IQ2_SIGNS",
    ]:
        report.check(marker in texts["kernels"], f"Rust kernel marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiRoutedMoeOneScope",
        "M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 74",
        "exported_compute_symbol_count: 72",
        "owns_routed_moe_one_tensor: true",
        "defers_batch_mid_is_f16_contract_gap: true",
        "owns_routed_moe_batch_tensor: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    entries = implementation.get("embedded_kernel_entries", [])
    for marker in [
        "abi_moe_q8_k_quantize_kernel",
        "abi_moe_gate_up_mid_f32_kernel",
        "abi_moe_down_f32_kernel",
        "abi_moe_gate_up_mid_qwarp32_kernel",
        "abi_moe_gate_up_mid_decode_lut_qwarp32_kernel",
        "abi_moe_gate_up_mid_decode_q4_k_qwarp32_kernel",
        "abi_moe_down_qwarp32_kernel",
        "abi_moe_down_sum6_qwarp32_kernel",
        "abi_moe_down_q4_k_sum6_qwarp32_kernel",
        "abi_moe_sum_kernel",
    ]:
        report.check(marker in entries, f"embedded kernel entry missing: {marker}")
    report.check("mid_is_f16" in implementation.get("remaining_compute_boundary", ""), "batch gap missing")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage path missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 154),
        ("feature_release_test_count", 161),
        ("staticlib_export_count", 74),
        ("embedded_kernel_count", 72),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "f32_fallback_nonzero",
        "default_iq2_q2_direct_sum_nonzero",
        "packed_input_q8_alias_visible",
        "packed_mid_q8_alias_visible",
        "default_aux_unwritten",
        "optional_gate_up_write_visible",
        "forced_generic_gate_matches",
        "forced_generic_down_matches",
        "q4_k_direct_sum_nonzero",
        "negative_expert_fallback_matches",
        "invalid_model_range_preserves_output",
        "invalid_type_rejected",
        "invalid_q4_group_rejected",
        "short_span_rejected",
        "null_rejected",
        "embedded_routed_moe_kernels_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 64, "predecessor count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 64, "warning count drift")
    for marker in [
        "ds4_gpu_routed_moe_one_tensor(",
        'setenv("DS4_CUDA_MOE_WRITE_GATE_UP"',
        'setenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE"',
        'setenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6"',
        "packed_q8_nonzero",
        "short_selected",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("mid_is_f16" in value for value in risks), "batch ABI gap risk missing")
    report.check(any("Q4_K" in value for value in risks), "Q4 environment risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-routed-moe-one-smoke.json"
    checker = "check_cuda_abi_routed_moe_one_smoke.py"
    item = f"{MILESTONE}: Public Single-Token Routed MoE ABI"
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
        ("fallback mismatch", lambda value: value["b300_execution"]["observed"].update({"f32_fallback_nonzero": False})),
        ("scratch alias mismatch", lambda value: value["b300_execution"]["observed"].update({"packed_mid_q8_alias_visible": False})),
        ("batch ownership overclaim", lambda value: value["ownership"].update({"owns_routed_moe_batch_tensor": True})),
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
