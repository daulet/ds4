#!/usr/bin/env python3
"""Validate the CUDA public batched routed-MoE mid-precision contract repair."""

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
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
MILESTONE_DIR = MILESTONE.lower()
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE_DIR}/cuda-routed-moe-batch-mid-contract-smoke.json"
HARNESS = ROOT / f"ds4-parity/fixtures/backend/{MILESTONE_DIR}/cuda_routed_moe_batch_mid_contract_smoke.c"


@dataclass
class Report:
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
        "cuda": (ROOT / "ds4_cuda.cu").read_text(encoding="utf-8"),
        "gpu_h": (ROOT / "ds4_gpu.h").read_text(encoding="utf-8"),
        "gpu_sys": (ROOT / "rust/ds4-gpu-sys/src/lib.rs").read_text(encoding="utf-8"),
        "metal": (ROOT / "ds4_metal.m").read_text(encoding="utf-8"),
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
        "harness": HARNESS.read_text(encoding="utf-8"),
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(encoding="utf-8"),
        "todo": (ROOT / ".memory/TODO.md").read_text(encoding="utf-8"),
        "status": (ROOT / ".memory/status.md").read_text(encoding="utf-8"),
        "readme": (ROOT / "ds4-parity/README.md").read_text(encoding="utf-8"),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(encoding="utf-8"),
    }
    report = Report()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    status = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} CUDA batched routed-MoE mid contract repair: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_routed_moe_batch_mid_contract_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-original-cuda-public-contract-repair", "status drift")
    validate_contract(report, fixture, texts)
    validate_scope(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_contract(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(oracle.get("symbols") == ["ds4_gpu_routed_moe_batch_tensor"], "oracle symbol drift")
    batch = re.search(
        r'extern "C" int ds4_gpu_routed_moe_batch_tensor\((.*?)\) \{(.*?)\n\}',
        texts["cuda"],
        re.S,
    )
    report.check(batch is not None, "CUDA batched routed-MoE definition missing")
    args = batch.group(1) if batch else ""
    body = batch.group(2) if batch else ""
    report.check("bool *mid_is_f16" in args, "CUDA batch result pointer missing")
    report.check("const int ok = routed_moe_launch(" in body, "CUDA batch launch delegation missing")
    report.check("if (ok && mid_is_f16) *mid_is_f16 = false;" in body, "CUDA F32 result report missing")
    report.check("bool                   *mid_is_f16" in texts["gpu_h"], "header result contract missing")
    report.check("mid_is_f16: *mut bool" in texts["gpu_sys"], "Rust FFI result contract missing")
    report.check("if (mid_is_f16) *mid_is_f16 = use_mid_f16;" in texts["metal"], "Metal result contract missing")


def validate_scope(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("repairs_current_cuda_mid_is_f16_result", True),
        ("reports_f32_mid_on_success", True),
        ("preserves_failed_call_result_sentinel", True),
        ("owns_routed_moe_batch_tensor_in_rust", False),
        ("owns_batched_routed_moe_scheduling_in_rust", False),
        ("changes_default_route", False),
        ("removes_c_cuda", False),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    rust_exports = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    report.check(len(rust_exports) >= 74, "published Rust ABI exports disappeared")
    report.check("ds4_gpu_routed_moe_one_tensor" in rust_exports, "published single-token export missing")
    report.check("ds4_gpu_routed_moe_one_tensor" in rust_exports, "published single-token export missing")
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check("Rust still does not export" in implementation.get("remaining_compute_boundary", ""), "batch deferral missing")
    report.check("original ds4_cuda.cu" in implementation.get("linkage_requirement", ""), "original CUDA linkage proof missing")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("linked_cubin_target", "sm_103"),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_original_cuda",
        "successful_batch_reports_f32_mid",
        "successful_null_result_pointer_accepted",
        "invalid_batch_preserves_mid_precision_result",
        "batched_output_nonzero",
    ]:
        report.check(observed.get(key) is True, f"observed contract drift: {key}")
    for marker in [
        "ds4_gpu_routed_moe_batch_tensor(",
        "bool success_mid_is_f16 = true;",
        "bool rejected_mid_is_f16 = true;",
        "N_TOKENS, NULL",
        "model.size - 1u",
        "n_tokens, mid_is_f16",
    ]:
        report.check(marker in texts["harness"], f"harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("Rust public batched" in risk for risk in risks), "Rust batch boundary risk missing")
    report.check(any("retained CUDA oracle" in risk for risk in risks), "oracle repair risk missing")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 154),
        ("cuda_abi_comparators_passed", 68),
        ("unified_report_passed", 241),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = f"ds4-parity/baselines/backend/{MILESTONE_DIR}/cuda-routed-moe-batch-mid-contract-smoke.json"
    checker = "check_cuda_routed_moe_batch_mid_contract_smoke.py"
    item = f"{MILESTONE}: CUDA Batched Routed MoE Mid Precision Contract Repair"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report wiring missing")
    report.check(NEXT_STAGE in texts["status"], "next active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    report.check(fixture.get("review", {}).get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review evidence missing")
    report.check(fixture.get("review", {}).get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review evidence missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("F32 report missing", lambda value: value["b300_execution"]["observed"].update({"successful_batch_reports_f32_mid": False})),
        ("Rust batch overclaim", lambda value: value["ownership"].update({"owns_routed_moe_batch_tensor_in_rust": True})),
        ("failed result mutation lost", lambda value: value["ownership"].update({"preserves_failed_call_result_sentinel": False})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
