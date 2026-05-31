#!/usr/bin/env python3
"""Validate the Rust CUDA public standalone RoPE ABI smoke."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-rope-tail-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
CUDA_KERNELS = ROOT / "rust/ds4-cuda/src/abi_kernels.rs"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/abi_rope_tail_link_smoke.c"
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
    print(f"{MILESTONE} Rust CUDA public standalone RoPE ABI smoke: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_abi_rope_tail_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-public-rope-tail-abi",
        "status drift",
    )
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["ds4_gpu_rope_tail_tensor"], "oracle symbols drift")
    for marker in [
        "__global__ static void rope_tail_kernel(",
        'extern "C" int ds4_gpu_rope_tail_tensor',
        "rope_tail_kernel<<<(pairs + 255) / 256, 256>>>",
        "rope_yarn_ramp_dev",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C RoPE marker missing: {marker}")
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 54),
        ("exported_compute_symbol_count", 31),
        ("public_gpu_abi_function_count", 81),
        ("owns_rope_tail_tensor", True),
        ("owns_rope_tail_kernel", True),
        ("owns_yarn_rotary_math_path", True),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 74, "Rust ABI export implementation count drift")
    report.check("ds4_gpu_rope_tail_tensor" in symbols, "standalone RoPE export missing")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        'pub unsafe extern "C" fn ds4_gpu_rope_tail_tensor',
        "n_rot == 0",
        "kernels.rope_tail_tensor(",
    ]:
        report.check(marker in texts["abi"], f"Rust RoPE ABI marker missing: {marker}")
    for marker in [
        "pub fn abi_rope_tail_kernel",
        "rope_tail_kernel: CudaFunction",
        '.load_function("abi_rope_tail_kernel")',
        "fn rope_tail_tensor(",
        ".ln()",
        ".cos()",
        "let base = ((u64::from(token) * u64::from(n_head) + u64::from(head))",
    ]:
        report.check(marker in texts["kernels"], f"embedded RoPE marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiRopeTailScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2B2B2BBBBBBBBBBBBBBBBBBBBBBBBBBBA_SCOPE",
        "exported_abi_symbol_count: 54",
        "exported_compute_symbol_count: 31",
        "owns_rope_tail_tensor: true",
        "owns_rope_tail_kernel: true",
        "owns_yarn_rotary_math_path: true",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("kernel_entry") == "abi_rope_tail_kernel", "kernel entry drift")
    report.check("--whole-archive" in implementation.get("linkage_requirement", ""), "linkage path missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 139),
        ("feature_release_test_count", 146),
        ("staticlib_export_count", 54),
        ("embedded_kernel_count", 32),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "interpolated_rope_output_matches",
        "yarn_inverse_output_matches",
        "non_rope_prefix_preserved",
        "zero_pair_rejected",
        "invalid_shape_rejected",
        "null_rejected",
        "embedded_rope_tail_kernel_loaded",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    report.check(observed.get("predecessor_c_linked_regression_consumers_passed") == 49, "predecessor count drift")
    report.check(observed.get("predecessor_relink_executable_stack_warning_count") == 49, "warning count drift")
    for marker in [
        "ds4_gpu_rope_tail_tensor(x, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0,",
        "ds4_gpu_rope_tail_tensor(short_x, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0,",
        "interpolated_rope_output_matches",
        "yarn_inverse_output_matches",
        "embedded_rope_tail_kernel_loaded",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("route promotion" in value for value in risks), "remaining-compute risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/abi-rope-tail-smoke.json"
    checker = "check_cuda_abi_rope_tail_smoke.py"
    item = f"{MILESTONE}: Public Standalone RoPE ABI"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active remainder status missing",
    )
    report.check(
        fixture.get("review", {}).get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S",
        "pre-implementation review evidence missing",
    )
    report.check(
        fixture.get("review", {}).get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S",
        "final review evidence missing",
    )
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("interpolated output failure", lambda value: value["b300_execution"]["observed"].update({"interpolated_rope_output_matches": False})),
        ("YaRN inverse failure", lambda value: value["b300_execution"]["observed"].update({"yarn_inverse_output_matches": False})),
        ("kernel ownership removed", lambda value: value["ownership"].update({"owns_rope_tail_kernel": False})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
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
