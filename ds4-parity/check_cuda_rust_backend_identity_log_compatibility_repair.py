#!/usr/bin/env python3
"""Validate the Rust CUDA backend identity-log compatibility repair leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Remaining Rust CUDA Promotion Acceptance And Route Decision"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-backend-identity-log-compatibility-repair.json"


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
        "abi": (ROOT / "rust/ds4-cuda/src/abi.rs").read_text(encoding="utf-8"),
        "substrate": (ROOT / "rust/ds4-cuda/src/substrate.rs").read_text(encoding="utf-8"),
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
    print(f"{MILESTONE} Rust CUDA backend identity-log compatibility repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_backend_identity_log_compatibility_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-cli-long-context-server-tool", "status drift")
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_rust_cuda_backend_identity_logging", True),
        ("changes_kernel_math", False),
        ("changes_runtime_route", False),
        ("default_current_c_route_preserved", True),
        ("default_cli_gate_closed", True),
        ("long_context_gate_closed", True),
        ("server_tool_quality_gate_closed", True),
        ("benchmark_gate_closed", False),
        ("runtime_route_promoted", False),
        ("c_host_removal_allowed", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(
        implementation.get("stderr_format")
        == "ds4: CUDA backend initialized on {name} (sm_{major}{minor})",
        "identity format drift",
    )
    report.check(
        "pub fn compute_capability(&self) -> Result<(i32, i32), DriverError>" in texts["substrate"],
        "compute-capability substrate query missing",
    )
    init = texts["abi"].split('pub extern "C" fn ds4_gpu_init()', 1)[1].split(
        'pub extern "C" fn ds4_gpu_cleanup()', 1
    )[0]
    report.check("opened.device_name()" in init, "device-name query missing from init")
    report.check("opened.compute_capability()" in init, "compute-capability query missing from init")
    report.check(
        'eprintln!("ds4: CUDA backend initialized on {name} (sm_{major}{minor})");' in init,
        "backend identity log missing from init",
    )
    report.check(
        init.index("if let (Ok(name), Ok((major, minor)))") < init.index("*backend = Some(opened);"),
        "identity log is not scoped to first successful initialization",
    )


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    pre = require_dict(report, fixture.get("pre_repair_blocker"), "pre_repair_blocker")
    report.check(pre.get("server_tool_validator_status") == "failed-only-rust-stderr-missing-b300-marker", "pre-repair server blocker drift")
    report.check("omitted" in pre.get("root_cause", ""), "pre-repair root cause missing")
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("shared_library_sha256", "223542460727037720a65a43961692ff9b42bbd75ddabea16344eb1094e69903"),
        ("engine_feature", "cuda-rust-backend"),
        ("gpu_utilization_percent_after_each_completed_probe", 0),
        ("gpu_memory_mib_after_each_completed_probe", 0),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    cli = require_dict(report, execution.get("default_cli"), "default_cli")
    for key, expected in [
        ("runtime_graph_route", "target-stream"),
        ("comparator_check_count", 144),
        ("negative_check_count", 5),
        ("passed", True),
    ]:
        report.check(cli.get(key) == expected, f"default CLI drift: {key}")
    long_context = require_dict(report, execution.get("long_context"), "long_context")
    for key, expected in [
        ("runtime_graph_route", "graph"),
        ("capture_kind", "retained-candidate-command-direct"),
        ("b300_identity_marker_present", True),
        ("prompt_tokens", 30474),
        ("completion_tokens", 76),
        ("cache_read_tokens", 0),
        ("cache_write_tokens", 30474),
        ("expected_fact_count", 16),
        ("matched_fact_count", 16),
        ("passed", True),
    ]:
        report.check(long_context.get(key) == expected, f"long-context drift: {key}")
    server = require_dict(report, execution.get("server_tool_quality"), "server_tool_quality")
    for key, expected in [
        ("runtime_graph_route", "graph"),
        ("comparator_check_count", 167),
        ("negative_check_count", 8),
        ("passed", True),
    ]:
        report.check(server.get(key) == expected, f"server/tool drift: {key}")
    cases = require_list(report, server.get("cases"), "server_tool_quality.cases")
    report.check([case.get("id") for case in cases if isinstance(case, dict)] == ["fast", "exact"], "server/tool case drift")
    for case in cases:
        item = require_dict(report, case, "server/tool case")
        report.check(item.get("finish_reason") == "tool_calls", "tool finish reason drift")
        report.check(item.get("tool_name") == "list_files", "tool name drift")
        report.check(item.get("arguments") == '{"path":"."}', "tool arguments drift")
        report.check(item.get("b300_identity_marker_present") is True, "tool B300 marker missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Backend Identity Log Compatibility Repair"
    checker = "check_cuda_rust_backend_identity_log_compatibility_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test count drift")
    report.check(validation.get("local_format_check_passed") is True, "format validation missing")
    report.check(validation.get("unified_report_passed") == 263, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified failure count drift")
    review = require_dict(report, fixture.get("review"), "review")
    for key in ["pre_execution", "follow_on_identity_log_fix", "final"]:
        report.check(review.get(key) == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", f"{key} review evidence missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("marker lost", lambda value: value["b300_execution"]["long_context"].update({"b300_identity_marker_present": False})),
        ("tool validation overclaim", lambda value: value["b300_execution"]["server_tool_quality"]["cases"][0].update({"tool_name": "wrong"})),
        ("benchmark overclaim", lambda value: value["ownership"].update({"benchmark_gate_closed": True})),
        ("route promotion overclaim", lambda value: value["ownership"].update({"runtime_route_promoted": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
