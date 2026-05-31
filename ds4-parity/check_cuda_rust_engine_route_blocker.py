#!/usr/bin/env python3
"""Validate the opt-in Rust CUDA engine runtime blocker probe leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Default-Route Promotion And C CUDA Removal Execution"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-engine-route-blocker.json"


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
        "engine_cargo": (ROOT / "rust/ds4-engine/Cargo.toml").read_text(encoding="utf-8"),
        "engine_build": (ROOT / "rust/ds4-engine/build.rs").read_text(encoding="utf-8"),
        "engine_lib": (ROOT / "rust/ds4-engine/src/lib.rs").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA engine route blocker probe: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_engine_route_blocker.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "blocked-short-official-vector-mismatch-and-full-timeout",
        "status drift",
    )
    validate_ownership(report, fixture, texts)
    validate_build(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("engine_route_opt_in", True),
        ("engine_feature", "cuda-rust-backend"),
        ("engine_links_rust_cuda_dylib", True),
        ("engine_compiles_ds4_cuda_cu", False),
        ("default_current_c_route_preserved", True),
        ("graph_selector_behavior_unchanged", True),
        ("runtime_route_promoted", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    report.check("cuda-rust-backend = []" in texts["engine_cargo"], "engine opt-in feature missing")
    report.check(
        "--runtime-graph graph is not implemented yet" in texts["engine_lib"],
        "runtime graph selector unexpectedly promoted",
    )


def validate_build(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    for key, expected in [
        ("crate", "rust/ds4-engine"),
        ("artifact_environment", "DS4_CUDA_RUST_DYLIB"),
        ("artifact_name", "libds4_cuda.so"),
        ("host_sources", ["ds4.c", "ds4_kvstore.c"]),
    ]:
        report.check(implementation.get(key) == expected, f"implementation drift: {key}")
    rust_selector = texts["engine_build"].split(
        'if env::var_os("CARGO_FEATURE_CUDA_RUST_BACKEND").is_some()', 1
    )[1].split("let cuda_home", 1)[0]
    rust_link = texts["engine_build"].split("fn link_linux_rust_cuda", 1)[1].split(
        "fn link_cpu", 1
    )[0]
    current_c = texts["engine_build"].split("let cuda_home", 1)[1].split(
        "compile_c(&repo_root, &out_dir, \"ds4.c\", &ds4_obj, true)", 1
    )[0]
    for marker in [
        "compile_c(&repo_root, &out_dir, \"ds4.c\", &ds4_obj, false)",
        "compile_c(&repo_root, &out_dir, \"ds4_kvstore.c\", &kvstore_obj, false)",
        "link_linux_rust_cuda",
    ]:
        report.check(marker in rust_selector, f"Rust engine selector marker missing: {marker}")
    report.check("compile_cuda" not in rust_selector, "Rust engine selector unexpectedly compiles CUDA C")
    for marker in [
        "DS4_CUDA_RUST_DYLIB",
        "rustc-link-lib=dylib=ds4_cuda",
        "-Wl,-rpath,",
        "rustc-link-lib=dylib=cuda",
    ]:
        report.check(marker in rust_link, f"Rust engine linker marker missing: {marker}")
    report.check("compile_cuda" in current_c, "current-C CUDA route removed")
    report.check('"ds4_cuda.cu"' in current_c, "current-C CUDA source marker removed")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    link = require_dict(report, execution.get("link_probe"), "link_probe")
    report.check(link.get("libds4_cuda_resolved") is True, "Rust DSO was not resolved")
    report.check(link.get("libcuda_resolved") is True, "CUDA driver was not resolved")
    one_step = require_dict(report, execution.get("one_step_probe"), "one_step_probe")
    for key, expected in [
        ("case", "short_italian_fact"),
        ("step", 0),
        ("elapsed_seconds", 47),
        ("selected_matches_expected", True),
        ("expected_selected_hex", "416461"),
        ("selected_hex", "416461"),
    ]:
        report.check(one_step.get(key) == expected, f"one-step evidence drift: {key}")
    short = require_dict(report, execution.get("short_vector_probe"), "short_vector_probe")
    for key, expected in [
        ("elapsed_seconds", 180),
        ("case_count", 3),
        ("step_count", 9),
        ("selected_match_count", 8),
    ]:
        report.check(short.get(key) == expected, f"short-vector evidence drift: {key}")
    mismatch = require_dict(report, short.get("mismatch"), "short_vector_probe.mismatch")
    for key, expected in [
        ("case", "short_code_completion"),
        ("step", 1),
        ("expected_selected_hex", "63"),
        ("selected_hex", "43"),
    ]:
        report.check(mismatch.get(key) == expected, f"mismatch evidence drift: {key}")
    full = require_dict(report, execution.get("full_official_vector_probe"), "full_official_vector_probe")
    report.check(full.get("completed") is False, "full official-vector overclaim")
    report.check(full.get("observation_seconds_at_least") == 900, "full timeout bound drift")
    report.check(full.get("terminated_after_bound") is True, "full timeout termination missing")
    report.check(full.get("gpu_memory_mib_after_termination") == 0, "B300 cleanup evidence drift")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    checker = "check_cuda_rust_engine_route_blocker.py"
    item = f"{MILESTONE}: Opt-In Rust CUDA Engine Route Blocker Probe"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_engine_library_test_count") == 13, "local tests drift")
    report.check(validation.get("unified_report_passed") == 258, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified fail count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("route promotion", lambda value: value["ownership"].update({"runtime_route_promoted": True})),
        ("mismatch hidden", lambda value: value["b300_execution"]["short_vector_probe"].update({"selected_match_count": 9})),
        ("full gate overclaim", lambda value: value["b300_execution"]["full_official_vector_probe"].update({"completed": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
