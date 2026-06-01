#!/usr/bin/env python3
"""Validate the Rust CUDA persistent cuBLAS handle performance repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Rust CUDA Graph Benchmark Performance Repair"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-persistent-cublas-handle-performance-repair.json"


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
        "substrate": (ROOT / "rust/ds4-cuda/src/substrate.rs").read_text(encoding="utf-8"),
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA persistent cuBLAS handle repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_persistent_cublas_handle_performance_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-rust-persistent-cublas-handle-route-blocked",
        "status drift",
    )
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("retains_one_blas_handle_per_substrate", True),
        ("reuses_handle_across_projection_and_attention_calls", True),
        ("retains_existing_kernel_dispatch", True),
        ("changes_cuda_oxide_revision", False),
        ("changes_default_current_c_route", False),
        ("official_vector_gate_preserved", True),
        ("runtime_route_promoted", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(
        implementation.get("source") == "rust/ds4-cuda/src/substrate.rs",
        "implementation source drift",
    )
    report.check(
        implementation.get("serialized_owner") == "rust/ds4-cuda/src/abi.rs:BACKEND",
        "serialized owner drift",
    )
    substrate = texts["substrate"]
    for marker in [
        "use std::sync::{Arc, OnceLock};",
        "blas: OnceLock<Blas>,",
        "unsafe impl Send for CudaOxideSubstrate {}",
        "pub fn blas_handle(&self) -> Result<&Blas, BlasError> {",
        "if let Some(blas) = self.blas.get()",
        "let blas = Blas::new(&self.context)?;",
        "let _ = self.blas.set(blas);",
    ]:
        report.check(marker in substrate, f"substrate marker missing: {marker}")
    report.check(
        "pub fn blas_handle(&self) -> Result<Blas, BlasError> {" not in substrate,
        "per-operation cuBLAS handle construction signature retained",
    )
    abi = texts["abi"]
    report.check(
        "static BACKEND: Mutex<Option<CudaOxideSubstrate>> = Mutex::new(None);" in abi,
        "backend serialization owner missing",
    )
    report.check(
        "let backend = BACKEND.lock().ok()?;\n    operation(backend.as_ref()?)" in abi,
        "backend lock lifetime changed",
    )


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "B300 execution")
    for key, expected in [
        ("date_utc", "2026-06-01"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("parent_shared_library_sha256", "1bfe1d95896f23d22a3b8b03b85753cfe5d20af2780a8c939c6e969dfadcabed"),
        ("repaired_shared_library_sha256", "53898bdfc5ae12faa17ef614359c9f981eb5224dedef0237f7d1b32887315caf"),
        ("gpu_utilization_percent_after_completed_probe", 0),
        ("gpu_memory_mib_after_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    for key, tps, decode_tps, csv_sha in [
        ("parent_route_probe", 259.64, 14.93, "61dd54452ce559a6f9477c8ba902d0039c2de313e535e4418dcbb68f887d826d"),
        ("repaired_route_probe", 468.40, 14.84, "1cffee2ac30e4ca9ffcf480ae3033f268592c032a70fc051bc65a86e4243f71a"),
    ]:
        probe = require_dict(report, execution.get(key), key)
        report.check(probe.get("prefill_tps") == tps, f"{key} prefill drift")
        report.check(probe.get("decode_tps") == decode_tps, f"{key} decode drift")
        report.check(probe.get("kvcache_bytes") == 52184460, f"{key} cache drift")
        report.check(probe.get("csv_sha256") == csv_sha, f"{key} CSV hash drift")
    for key, total, attention, output_proj in [
        ("parent_profile_probe", 6434.948, 1554.880, 642.935),
        ("repaired_profile_probe", 2714.401, 1354.844, 191.542),
    ]:
        probe = require_dict(report, execution.get(key), key)
        report.check(probe.get("profiled_layer_count") == 43, f"{key} layer-count drift")
        report.check(probe.get("prefill_total_ms") == total, f"{key} total drift")
        report.check(probe.get("attention_ms") == attention, f"{key} attention drift")
        report.check(probe.get("output_projection_ms") == output_proj, f"{key} output drift")
    current_c = require_dict(report, execution.get("current_c_reference"), "current-C reference")
    report.check(current_c.get("route_prefill_tps") == 1400.45, "current-C route drift")
    report.check(current_c.get("attention_ms") == 191.404, "current-C attention drift")
    report.check(current_c.get("output_projection_ms") == 59.191, "current-C output drift")
    attribution = require_dict(report, execution.get("attribution"), "attribution")
    for key, expected in [
        ("repaired_route_prefill_speedup_over_parent", 1.804),
        ("repaired_profile_total_speedup_over_parent", 2.371),
        ("repaired_profile_total_reduction_percent_over_parent", 57.82),
        ("repaired_route_prefill_ratio_to_current_c", 0.334),
        ("repaired_attention_ratio_to_current_c", 7.078),
        ("repaired_output_projection_ratio_to_current_c", 3.236),
    ]:
        report.check(attribution.get(key) == expected, f"attribution drift: {key}")
    official = require_dict(report, execution.get("official_vector_probe"), "official vectors")
    report.check(
        official.get("summary_sha256") == "5f97ed9cc8dcd9b8387a494a56f9f8cb179f09a42ccfe21a51fc97b9b3da2300",
        "official summary hash drift",
    )
    report.check(official.get("comparator_check_count") == 1958, "official check-count drift")
    report.check(official.get("negative_check_count") == 8, "official negative-count drift")
    report.check(official.get("passed") is True, "official pass missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Persistent cuBLAS Handle Performance Repair"
    checker = "check_cuda_rust_persistent_cublas_handle_performance_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README wiring missing")
    report.check(checker in texts["report"], "report wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    decision = require_dict(report, fixture.get("decision"), "decision")
    report.check(decision.get("default_route") == "retain-current-c", "default route drift")
    report.check(decision.get("rust_cuda_dso_promotion") == "blocked", "promotion drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test-count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 test-count drift")
    report.check(validation.get("unified_report_passed") == 287, "unified pass-count drift")
    report.check(validation.get("unified_report_skipped") == 50, "unified skip-count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure-count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review drift")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review drift")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("lost retained handle", lambda value: value["ownership"].update({"retains_one_blas_handle_per_substrate": False})),
        ("lost route improvement", lambda value: value["b300_execution"]["repaired_route_probe"].update({"prefill_tps": 259.64})),
        ("attention blocker omitted", lambda value: value["b300_execution"]["attribution"].update({"repaired_attention_ratio_to_current_c": 1.0})),
        ("promotion overclaim", lambda value: value["decision"].update({"rust_cuda_dso_promotion": "passed"})),
        ("official mismatch", lambda value: value["b300_execution"]["official_vector_probe"].update({"passed": False})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
