#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2b1 registration-disable ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-registration-disable-smoke.json"
PREDECESSOR = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b1/abi-model-control-registered-fallback-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b1/abi_model_control_registration_disable_link_smoke.c"
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
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
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
    validate(report, fixture, predecessor, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, predecessor, texts)
    state = "PASS" if report.ok else "FAIL"
    print(
        "M14.6b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public registration-disable "
        f"ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: ReportState,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_registration_disable_smoke.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2b1", "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-registration-disable-abi",
        "status drift",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, predecessor, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols") == ["cuda_model_range_ptr", "ds4_gpu_set_model_map"],
        "oracle symbols drift",
    )
    for marker in [
        "static int g_model_range_mapping_supported = 1;",
        "if (g_model_range_mapping_supported) {",
        "if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;",
        "g_model_range_mapping_supported = 1;",
        "cudaHostRegister((void *)model_map",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C selection marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_cross_range_registration_disable_policy", True),
        ("owns_current_c_disable_error_classes", True),
        ("owns_model_replacement_registration_reset", True),
        ("owns_live_public_attempt_count_observation", True),
        ("owns_successful_zero_copy_registration_observation", False),
        ("owns_remaining_failure_selection", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 63, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "static ABI_MODEL_RANGE_MAPPING_SUPPORTED: AtomicBool = AtomicBool::new(true);",
        "fn abi_range_registration_disables(",
        "fn try_register_abi_model_range(",
        "ABI_MODEL_RANGE_MAPPING_SUPPORTED.store(false, Ordering::Relaxed);",
        "ABI_MODEL_RANGE_MAPPING_SUPPORTED.store(true, Ordering::Relaxed);",
    ]:
        report.check(marker in texts["abi"], f"Rust registration-disable marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiResidualRegistrationDisableScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B1_SCOPE",
        "owns_cross_range_registration_disable_policy: true",
        "owns_current_c_disable_error_classes: true",
        "owns_model_replacement_registration_reset: true",
        "owns_live_public_attempt_count_observation: true",
        "owns_remaining_failure_selection: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(
    report: ReportState,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 107),
        ("feature_release_test_count", 114),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    request = require_dict(
        report,
        execution.get("interposed_registration_request"),
        "interposed_registration_request",
    )
    for key, expected in [
        ("injected_error_code", 801),
        ("first_map_whole_registration_attempts", 1),
        ("first_map_range_registration_attempts_before_disable", 1),
        ("first_map_suppressed_disjoint_range_attempts", 1),
        ("replacement_map_whole_registration_attempts", 1),
        ("replacement_map_range_registration_attempts_after_reset", 1),
        ("total_observed_registration_attempts", 4),
    ]:
        report.check(request.get(key) == expected, f"public request drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "page_aligned_model_maps",
        "interposed_not_supported_registration",
        "whole_map_failure_does_not_disable_range_attempt",
        "first_range_failure_disables_second_range_attempt",
        "model_replacement_resets_range_attempt_state",
        "device_copy_fallback_outputs_match",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "registered_fallback_regression_passed",
        "whole_registration_regression_passed",
        "fd_budget_regression_passed",
        "fd_source_page_progress_regression_passed",
        "temporary_link_artifacts_cleaned",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    predecessor_ownership = require_dict(report, predecessor.get("ownership"), "predecessor ownership")
    report.check(
        predecessor_ownership.get("owns_cross_range_registration_disable_policy") is False,
        "predecessor pending boundary drift",
    )
    predecessor_probe = require_dict(
        report,
        require_dict(report, predecessor.get("b300_execution"), "predecessor execution").get(
            "registration_probe"
        ),
        "predecessor probe",
    )
    report.check(predecessor_probe.get("read_only_registration_error_code") == 801, "native B300 rejection evidence drift")
    for marker in [
        "CUresult cuMemHostRegister_v2(",
        "return 801;",
        "host_register_calls != 2",
        "host_register_calls != 4",
        "ds4_gpu_cache_model_range(first_map, model_size, second_offset",
        "ds4_gpu_set_model_map(second_map, model_size)",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    risks = fixture.get("integration_risks", [])
    report.check(any("does not claim a successful zero-copy" in value for value in risks), "zero-copy boundary missing")
    report.check(any("native B300 rejection" in value for value in risks), "native rejection context missing")
    report.check(any("remaining model-control failure selection" in value for value in risks), "remaining selection risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-registration-disable-smoke.json"
    checker = "check_cuda_abi_model_control_registration_disable_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2b1: Public Cross-Range Registration Disable ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2b1 Public Cross-Range Registration Disable ABI"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2 Remaining Residual Failure Selection Policy",
        "next stage drift",
    )


def run_negative_tests(
    report: ReportState,
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("disable ownership missing", lambda value: value["ownership"].update({"owns_cross_range_registration_disable_policy": False})),
        ("same-map suppression missing", lambda value: value["b300_execution"]["observed"].update({"first_range_failure_disables_second_range_attempt": False})),
        ("reset observation missing", lambda value: value["b300_execution"]["observed"].update({"model_replacement_resets_range_attempt_state": False})),
        ("zero-copy overclaim", lambda value: value["ownership"].update({"owns_successful_zero_copy_registration_observation": True})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, predecessor, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
