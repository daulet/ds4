#!/usr/bin/env python3
"""Validate the opt-in Rust CUDA production-facade shared-library link leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Long-Prefill Performance And C CUDA Removal Policy"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-production-facade-link.json"


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


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} missing")
    return value if isinstance(value, dict) else {}


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "cuda_cargo": (ROOT / "rust/ds4-cuda/Cargo.toml").read_text(encoding="utf-8"),
        "kernels": (ROOT / "rust/ds4-cuda/src/abi_kernels.rs").read_text(encoding="utf-8"),
        "gpu_cargo": (ROOT / "rust/ds4-gpu/Cargo.toml").read_text(encoding="utf-8"),
        "gpu_build": (ROOT / "rust/ds4-gpu/build.rs").read_text(encoding="utf-8"),
        "facade_test": (ROOT / "rust/ds4-gpu/tests/rust_cuda_backend_abi.rs").read_text(encoding="utf-8"),
        "engine": (ROOT / "rust/ds4-engine/src/lib.rs").read_text(encoding="utf-8"),
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
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA production facade shared-library link: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_production_facade_link.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-opt-in-rust-cuda-production-facade-dylib",
        "status drift",
    )
    validate_ownership(report, fixture, texts)
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("public_gpu_abi_function_count", 81),
        ("embedded_kernel_count", 108),
        ("current_c_oracle_preserved", True),
        ("staticlib_c_consumer_still_requires_whole_archive", True),
        ("production_facade_opt_in", True),
        ("production_facade_uses_rust_cuda_dylib", True),
        ("production_facade_uses_rust_cuda_staticlib", False),
        ("changes_default_route", False),
        ("runtime_route_promoted", False),
        ("removes_ds4_cuda_cu", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    report.check("cuda-rust-backend = [\"cuda-backend\"]" in texts["gpu_cargo"], "opt-in feature missing")
    report.check(
        "crate-type = [\"rlib\", \"staticlib\", \"cdylib\"]" in texts["cuda_cargo"],
        "Rust CUDA shared-library artifact missing",
    )
    report.check(
        "--runtime-graph graph is not implemented yet" in texts["engine"],
        "default graph route unexpectedly promoted",
    )


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    for key, expected in [
        ("feature", "cuda-rust-backend"),
        ("artifact_environment", "DS4_CUDA_RUST_DYLIB"),
        ("artifact_name", "libds4_cuda.so"),
    ]:
        report.check(implementation.get(key) == expected, f"implementation drift: {key}")
    rust_branch = texts["gpu_build"].split("fn build_linux_rust_cuda_backend()", 1)[1].split(
        "fn rerun_for_backend_sources", 1
    )[0]
    current_c_branch = texts["gpu_build"].split("fn build_linux_cuda_backend()", 1)[1].split(
        "fn build_linux_rust_cuda_backend", 1
    )[0]
    for marker in [
        'CARGO_FEATURE_CUDA_RUST_BACKEND',
        'DS4_CUDA_RUST_DYLIB',
        'Some("libds4_cuda.so")',
        'rustc-link-lib=dylib=ds4_cuda',
        '-Wl,-rpath,',
    ]:
        report.check(marker in texts["gpu_build"], f"facade build marker missing: {marker}")
    report.check("compile_c_linux" in rust_branch, "Rust facade no longer compiles host facade")
    report.check("compile_cuda" not in rust_branch, "Rust facade unexpectedly compiles current C CUDA")
    report.check("ds4_cuda.cu" not in rust_branch, "Rust facade unexpectedly names current C CUDA")
    report.check("compile_cuda" in current_c_branch, "current-C build route removed")
    for marker in [
        "artifact_bundles_from_binary_path",
        "fn embedded_abi_modules()",
        "embedded_modules_from_current_exe()",
        "libc::dladdr",
        "fn abi_image_path()",
    ]:
        report.check(marker in texts["kernels"], f"module-loader marker missing: {marker}")
    for marker in [
        "rust_cuda_dylib_supplies_embedded_compute_abi_to_facade",
        "sys::ds4_gpu_add_tensor",
        "vec![1.5, 0.0, 2.0, 12.0]",
    ]:
        report.check(marker in texts["facade_test"], f"facade integration marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("target", "sm_80"),
        ("shared_library_export_count", 81),
        ("embedded_kernel_count", 108),
        ("local_library_test_count", 169),
        ("feature_release_test_count", 176),
        ("facade_integration_test_count", 3),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "shared_library_has_public_abi",
        "shared_library_embeds_add_kernel_ptx",
        "shared_library_resolves_cuda_driver",
        "fresh_target_facade_link_passed",
        "existing_backend_abi_passed",
        "existing_model_map_abi_passed",
        "embedded_add_kernel_passed",
        "staticlib_topk_regression_passed",
        "staticlib_gnu_stack_warning_remains",
    ]:
        report.check(observed.get(key) is True, f"observed execution drift: {key}")
    validation = require_dict(report, fixture.get("validation"), "validation")
    for key, expected in [
        ("local_ds4_cuda_library_test_count", 169),
        ("local_ds4_gpu_library_test_count", 83),
        ("b300_feature_release_test_count", 176),
        ("previous_cuda_abi_comparators_passed", 83),
        ("unified_report_passed", 257),
        ("unified_report_skipped", 45),
        ("unified_report_failed", 0),
    ]:
        report.check(validation.get(key) == expected, f"validation evidence drift: {key}")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    checker = "check_cuda_rust_production_facade_link.py"
    item = f"{MILESTONE}: Opt-In Rust CUDA Production Facade Shared-Library Link"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("default route promoted", lambda value: value["ownership"].update({"changes_default_route": True})),
        ("dylib execution removed", lambda value: value["b300_execution"]["observed"].update({"embedded_add_kernel_passed": False})),
        ("C archive regression removed", lambda value: value["b300_execution"]["observed"].update({"staticlib_topk_regression_passed": False})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
