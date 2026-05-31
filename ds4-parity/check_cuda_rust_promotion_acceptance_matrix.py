#!/usr/bin/env python3
"""Validate the Rust CUDA default-route promotion acceptance contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Default-Route Promotion And C CUDA Removal Execution"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-promotion-acceptance-matrix.json"


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


def require_list(report: Report, value: Any, label: str) -> list[Any]:
    report.check(isinstance(value, list), f"{label} missing")
    return value if isinstance(value, list) else []


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "engine_build": (ROOT / "rust/ds4-engine/build.rs").read_text(encoding="utf-8"),
        "engine_lib": (ROOT / "rust/ds4-engine/src/lib.rs").read_text(encoding="utf-8"),
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(encoding="utf-8"),
        "todo": (ROOT / ".memory/TODO.md").read_text(encoding="utf-8"),
        "status": (ROOT / ".memory/status.md").read_text(encoding="utf-8"),
        "readme": (ROOT / "ds4-parity/README.md").read_text(encoding="utf-8"),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(encoding="utf-8"),
    }
    m109 = {
        stage: json.loads((ROOT / path).read_text(encoding="utf-8"))
        for stage, path in {
            "M10.9c": "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
            "M10.9d": "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
            "M10.9f": "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
        }.items()
    }
    report = Report()
    validate(report, fixture, texts, m109)
    if args.negative_test:
        run_negative_tests(report, fixture, texts, m109)
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA promotion acceptance matrix: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(
    report: Report,
    fixture: dict[str, Any],
    texts: dict[str, str],
    m109: dict[str, dict[str, Any]],
) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_promotion_acceptance_matrix.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "contract-defined-transient-kubernetes-exec-authorization-recovered",
        "status drift",
    )
    validate_route_boundary(report, fixture, texts)
    validate_prior_evidence_partition(report, fixture, m109)
    validate_matrix(report, fixture)
    validate_blocker(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_route_boundary(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("engine_feature", "cuda-rust-backend"),
        ("rust_cuda_dylib_required", True),
        ("rust_cuda_dylib_environment", "DS4_CUDA_RUST_DYLIB"),
        ("feature_compiles_c_host_engine", True),
        ("feature_compiles_ds4_cuda_cu", False),
        ("current_c_default_route_preserved", True),
        ("runtime_route_promoted", False),
        ("c_host_removal_allowed", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    selector = texts["engine_build"].split(
        'if env::var_os("CARGO_FEATURE_CUDA_RUST_BACKEND").is_some()', 1
    )[1].split("let cuda_home", 1)[0]
    report.check('"ds4.c"' in selector and '"ds4_kvstore.c"' in selector, "C host boundary missing")
    report.check("link_linux_rust_cuda" in selector, "Rust CUDA DSO link missing")
    report.check("compile_cuda" not in selector and '"ds4_cuda.cu"' not in selector, "Rust route compiles C CUDA")
    report.check(
        "--runtime-graph graph is not implemented yet" in texts["engine_lib"],
        "regular CLI graph fail-closed boundary missing",
    )


def validate_prior_evidence_partition(
    report: Report, fixture: dict[str, Any], m109: dict[str, dict[str, Any]]
) -> None:
    evidence = require_dict(report, fixture.get("existing_evidence_partition"), "existing_evidence_partition")
    official = require_dict(report, evidence.get("rust_cuda_official_vectors"), "rust_cuda_official_vectors")
    report.check(official.get("status") == "passed", "official-vector predecessor status drift")
    report.check(official.get("selected_match_count") == 13, "official-vector selected match drift")
    historical = require_dict(report, evidence.get("m10_9_artifacts"), "m10_9_artifacts")
    report.check(historical.get("status") == "not-rust-cuda-dylib-evidence", "M10.9 claim boundary drift")
    for stage, summary in m109.items():
        build = summary.get("build", {})
        if stage == "M10.9d":
            command = build.get("rust", {}).get("command", [])
        else:
            command = build.get("command", [])
        text = " ".join(command)
        report.check("--features" not in text and "cuda-rust-backend" not in text, f"{stage} feature claim drift")


def validate_matrix(report: Report, fixture: dict[str, Any]) -> None:
    matrix = require_list(report, fixture.get("acceptance_matrix"), "acceptance_matrix")
    entries = {item.get("id"): item for item in matrix if isinstance(item, dict)}
    expected = {
        "default_cli": "pending-live-rerun",
        "official_vectors": "passed-predecessor",
        "long_context": "pending-live-rerun",
        "server_tool_quality": "pending-live-rerun",
        "benchmark": "pending-live-rerun",
        "default_route_promotion": "blocked-pending-runtime-gates",
        "c_host_removal": "blocked-by-host-engine-boundary",
        "c_cuda_removal": "blocked-pending-runtime-gates",
    }
    report.check(set(entries) == set(expected), "acceptance gate set drift")
    for gate, state in expected.items():
        report.check(entries.get(gate, {}).get("status") == state, f"acceptance status drift: {gate}")
    contract = require_dict(report, fixture.get("rerun_contract"), "rerun_contract")
    report.check(contract.get("required_feature") == "cuda-rust-backend", "required feature drift")
    env = require_dict(report, contract.get("build_environment"), "build_environment")
    report.check("libds4_cuda.so" in env.get("DS4_CUDA_RUST_DYLIB", ""), "DSO rerun path missing")
    binaries = set(contract.get("required_binaries", []))
    report.check(
        binaries
        == {
            "ds4-cli-one-shot-rs",
            "ds4-runtime-long-context-rs",
            "ds4-server-runtime-rs",
            "ds4-runtime-graph-bench-rs",
        },
        "required binaries drift",
    )


def validate_blocker(report: Report, fixture: dict[str, Any]) -> None:
    blocker = require_dict(report, fixture.get("infrastructure_blocker"), "infrastructure_blocker")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("pod_phase", "Running"),
        ("workspace_pvc_mounted_at", "/workspace"),
        ("api_read_succeeds", True),
        ("exec_succeeds", False),
        ("exec_restored_before_commit", True),
        ("gpu_validation_started", False),
        ("pod_replacement_not_attempted", True),
    ]:
        report.check(blocker.get(key) == expected, f"infrastructure evidence drift: {key}")
    report.check("nodes, subresource=proxy" in blocker.get("exec_error_contains", ""), "exec authorization error missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Promotion Acceptance Matrix And Rerun Contract"
    checker = "check_cuda_rust_promotion_acceptance_matrix.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README wiring missing")
    report.check(checker in texts["report"], "unified report wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_comparator_negative_test_passed") is True, "local validation missing")
    report.check(validation.get("unified_report_passed") == 262, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(
    report: Report,
    fixture: dict[str, Any],
    texts: dict[str, str],
    m109: dict[str, dict[str, Any]],
) -> None:
    for label, mutate in [
        ("runtime promotion overclaim", lambda value: value["ownership"].update({"runtime_route_promoted": True})),
        ("host removal overclaim", lambda value: value["acceptance_matrix"][6].update({"status": "passed"})),
        ("unrun long-context claim", lambda value: value["acceptance_matrix"][2].update({"status": "passed"})),
        ("lost infrastructure blocker", lambda value: value["infrastructure_blocker"].update({"exec_succeeds": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts, m109)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
