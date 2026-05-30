#!/usr/bin/env python3
"""Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2a full-model copy ABI smoke."""

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
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-full-model-copy-smoke.json"
CUDA_C = ROOT / "ds4_cuda.cu"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2a/abi_model_control_full_model_copy_link_smoke.c"
FD_BUDGET_HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi_model_control_fd_cache_budget_link_smoke.c"
FD_PROGRESS_HARNESS = ROOT / "ds4-parity/fixtures/backend/m14.6b2b2b2b2b2b2b2b2b2b2a/abi_model_control_fd_source_page_progress_link_smoke.c"
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
        "harness": HARNESS.read_text(encoding="utf-8"),
        "fd_budget": FD_BUDGET_HARNESS.read_text(encoding="utf-8"),
        "fd_progress": FD_PROGRESS_HARNESS.read_text(encoding="utf-8"),
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
    print(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public full-model copy "
        f"ABI smoke: {state} ({report.checks} checks)"
    )
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_abi_model_control_full_model_copy_smoke.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == "M14.6b2b2b2b2b2b2b2b2b2b2b2a", "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-staticlib-full-model-copy-abi",
        "status drift",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, fixture, texts)


def validate_oracle(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("oracle"), "oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "oracle source drift")
    report.check(
        oracle.get("symbols")
        == ["ds4_gpu_set_model_map", "cuda_model_range_ptr", "cuda_model_range_is_cached"],
        "oracle symbols drift",
    )
    for marker in [
        'const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");',
        "if (copy_env && copy_env[0])",
        "cudaMalloc(&dev, (size_t)model_size)",
        "cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice)",
        "g_model_device_owned = 1;",
        "cudaHostRegister((void *)model_map",
        "if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);",
        "if (g_model_device_owned || g_model_registered) return 1;",
    ]:
        report.check(marker in texts["cuda_c"], f"current-C full-copy marker missing: {marker}")


def validate_ownership(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("exported_abi_symbol_count", 29),
        ("exported_compute_symbol_count", 9),
        ("public_gpu_abi_function_count", 81),
        ("owns_nonempty_full_model_copy_selection", True),
        ("owns_retained_full_model_device_image", True),
        ("owns_copy_failure_registration_continuation", True),
        ("owns_live_copy_failure_observation", False),
        ("owns_remaining_failure_selection", False),
        ("owns_remaining_graph_compute_abi", False),
        ("owns_complete_ds4_gpu_abi", False),
        ("changes_default_route", False),
        ("production_build_still_compiles_ds4_cuda_cu", True),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    symbols = set(re.findall(r'pub (?:unsafe )?extern "C" fn (ds4_gpu_[A-Za-z0-9_]+)', texts["abi"]))
    ffi_symbols = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"]))
    report.check(len(symbols) == 32, "Rust ABI export implementation count drift")
    report.check(len(ffi_symbols) == 81, "public GPU ABI function count drift")
    report.check(symbols <= ffi_symbols, "Rust exports do not match public GPU ABI")
    for marker in [
        "fn full_model_copy_selected() -> bool",
        'std::env::var_os("DS4_CUDA_COPY_MODEL").is_some_and(|value| !value.is_empty())',
        "fn try_copy_abi_model_window(",
        "if full_model_copy_selected()",
        "&& try_copy_abi_model_window(backend, model_map, model_size, 0, model_size)",
        "let _ = try_register_abi_model(backend, model_map, model_size);",
    ]:
        report.check(marker in texts["abi"], f"Rust full-copy marker missing: {marker}")
    for marker in [
        "pub struct CudaAbiFullModelCopySelectionScope",
        "pub const M14_6B2B2B2B2B2B2B2B2B2B2B2A_SCOPE",
        "owns_nonempty_full_model_copy_selection: true",
        "owns_retained_full_model_device_image: true",
        "owns_copy_failure_registration_continuation: true",
        "owns_live_copy_failure_observation: false",
        "owns_remaining_failure_selection: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    report.check('.arg("ds4_cuda.cu")' in texts["gpu_build"], "current C CUDA link marker missing")


def validate_execution(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-30"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("local_library_test_count", 108),
        ("feature_release_test_count", 115),
    ]:
        report.check(execution.get(key) == expected, f"execution drift: {key}")
    observed = require_dict(report, execution.get("observed"), "observed")
    for key in [
        "c_linked_rust_staticlib",
        "nonempty_copy_model_selected",
        "successful_full_copy_skips_registration",
        "host_mutation_after_map_setup_ignored",
        "model_replacement_copies_new_image",
        "cached_weighted_rms_reads_copied_image",
        "weighted_outputs_match",
        "embedded_libdevice_module_loaded",
        "staticlib_export_count_unchanged",
        "whole_registration_regression_passed",
        "registration_disable_regression_passed",
        "chunk_selected_copy_regression_passed",
        "fd_budget_regression_passed",
        "fd_source_page_progress_regression_passed",
    ]:
        report.check(observed.get(key) is True, f"observed smoke drift: {key}")
    for marker in [
        "CUresult cuMemHostRegister_v2(",
        "return 801;",
        'setenv("DS4_CUDA_COPY_MODEL", "1", 1)',
        "host_register_calls != 0",
        "memcpy((unsigned char *)first_map + offset, first_changed, bytes)",
        "ds4_gpu_set_model_map(second_map, sizeof(second_map))",
        "ds4_gpu_rms_norm_weight_tensor(",
    ]:
        report.check(marker in texts["harness"], f"C-linked harness marker missing: {marker}")
    for name in ["fd_budget", "fd_progress"]:
        report.check(
            'setenv("DS4_CUDA_COPY_MODEL", "", 1)' in texts[name],
            f"{name} regression still selects full-model copy instead of fd staging",
        )
    risks = fixture.get("integration_risks", [])
    report.check(any("source-backed only" in value for value in risks), "live copy-failure caveat missing")
    report.check(any("fd-only predecessor" in value for value in risks), "fd regression correction caveat missing")
    report.check(any("remaining model-control" in value for value in risks), "remaining selection risk missing")
    report.check(any("executable-stack" in value for value in risks), "linker warning risk missing")


def validate_wiring(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    fixture_path = (
        "ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2a/"
        "abi-model-control-full-model-copy-smoke.json"
    )
    checker = "check_cuda_abi_model_control_full_model_copy_smoke.py"
    item = "M14.6b2b2b2b2b2b2b2b2b2b2b2a: Public Full-Model Copy Selection ABI"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbb Remaining Graph Compute And Route Promotion Policy"
        in texts["status"],
        "active item missing",
    )
    report.check(
        "M14.6b2b2b2b2b2b2b2b2b2b2b2a Public Full-Model Copy Selection ABI"
        in texts["status"],
        "status evidence missing",
    )
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(
        fixture.get("next_required_stage")
        == "M14.6b2b2b2b2b2b2b2b2b2b2b2b Remaining Residual Failure Selection Policy",
        "next stage drift",
    )


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("full-copy ownership missing", lambda value: value["ownership"].update({"owns_nonempty_full_model_copy_selection": False})),
        ("retained image missing", lambda value: value["b300_execution"]["observed"].update({"host_mutation_after_map_setup_ignored": False})),
        ("registration unexpectedly used", lambda value: value["b300_execution"]["observed"].update({"successful_full_copy_skips_registration": False})),
        ("live failure overclaim", lambda value: value["ownership"].update({"owns_live_copy_failure_observation": True})),
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
