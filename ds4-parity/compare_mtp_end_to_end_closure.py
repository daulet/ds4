#!/usr/bin/env python3
"""Validate the M10.8g4b MTP end-to-end closure decision."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json"
BRANCH_DECISION = ROOT / "ds4-parity/baselines/graph/m10.8g4a/support-branch-decision.json"
STREAM_CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json"
RUNTIME_BLOCKER = ROOT / "ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json"

SCHEMA = "ds4.mtp_end_to_end_stream_closure.v1"
SOURCE = "b300-mtp-end-to-end-stream-closure"
MILESTONE = "M10.8g4b"
SELECTED_BRANCH = "support_absent_blocker_closure"
BLOCKER = "blocked_missing_mtp_model"
NEXT_STAGE = "M10.9"
BASE_MODEL = "/workspace/ds4/ds4flash.gguf"
BASE_MODEL_SIZE = 86_720_111_488
MISSING_MTP = "/workspace/ds4/missing-mtp.gguf"


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected JSON object")
    return obj


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def build_summary(
    branch_decision: dict[str, Any],
    stream_contract: dict[str, Any],
    runtime_blocker: dict[str, Any],
) -> dict[str, Any]:
    support = branch_decision.get("support_artifacts") or {}
    runtime = runtime_blocker.get("runtime") or {}
    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "inputs": {
            "branch_decision": rel(BRANCH_DECISION),
            "stream_contract": rel(STREAM_CONTRACT),
            "runtime_missing_support_summary": rel(RUNTIME_BLOCKER),
        },
        "selected_branch": branch_decision.get("selected_branch"),
        "closure_status": BLOCKER,
        "next_stage": NEXT_STAGE,
        "support_present_comparator": {
            "status": "not_run",
            "reason": "support_artifact_absent",
            "required_branch": "support_present_end_to_end_comparator",
        },
        "explicit_blocker": {
            "result": BLOCKER,
            "reason": "no B300 MTP support GGUF is present",
            "target_model_path": support.get("base_model_path"),
            "target_model_bytes": support.get("base_model_bytes"),
            "expected_mtp_path": support.get("expected_mtp_path"),
            "mtp_candidates": support.get("mtp_candidates"),
            "candidate_search_stdout": support.get("candidate_search_stdout"),
            "stream_case": "b300_missing_mtp_support_model",
            "runtime_guard_case": "b300_missing_mtp_support_runtime_blocker",
            "target_stream_visibility": runtime.get("target_stream_visibility"),
            "accepted_stream_delta": runtime.get("accepted_stream_delta"),
            "checkpoint_delta": runtime.get("checkpoint_delta"),
            "logits_source": runtime.get("logits_source"),
            "cache_kvc_visibility": runtime.get("cache_kvc_visibility"),
            "fallback": runtime.get("fallback"),
            "error": runtime.get("error"),
        },
        "claim_policy": branch_decision.get("claim_policy"),
        "closure_statement": (
            "M10.8g closes with an explicit support-artifact blocker; "
            "MTP-enabled current-C versus Rust stream parity is not claimed."
        ),
    }


def validate(
    summary: dict[str, Any],
    branch_decision: dict[str, Any],
    stream_contract: dict[str, Any],
    runtime_blocker: dict[str, Any],
) -> Report:
    report = Report()
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == SOURCE, "summary source drift")
    report.check(summary.get("milestone") == MILESTONE, "summary milestone drift")
    validate_inputs(report, summary.get("inputs"))
    report.check(summary.get("selected_branch") == SELECTED_BRANCH, "selected branch drift")
    report.check(summary.get("closure_status") == BLOCKER, "closure status drift")
    report.check(summary.get("next_stage") == NEXT_STAGE, "next stage drift")
    validate_support_present_status(report, summary.get("support_present_comparator"))
    validate_explicit_blocker(report, summary.get("explicit_blocker"))
    validate_claim_policy(report, summary.get("claim_policy"))
    validate_upstream(report, branch_decision, stream_contract, runtime_blocker)
    validate_static_wiring(report)
    return report


def validate_inputs(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "inputs missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("branch_decision") == rel(BRANCH_DECISION), "branch decision input drift")
    report.check(value.get("stream_contract") == rel(STREAM_CONTRACT), "stream contract input drift")
    report.check(
        value.get("runtime_missing_support_summary") == rel(RUNTIME_BLOCKER),
        "runtime blocker input drift",
    )


def validate_support_present_status(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "support-present comparator status missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("status") == "not_run", "support-present comparator status drift")
    report.check(value.get("reason") == "support_artifact_absent", "support-present reason drift")
    report.check(
        value.get("required_branch") == "support_present_end_to_end_comparator",
        "support-present required branch drift",
    )


def validate_explicit_blocker(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "explicit blocker missing")
    if not isinstance(value, dict):
        return
    expected = {
        "result": BLOCKER,
        "reason": "no B300 MTP support GGUF is present",
        "target_model_path": BASE_MODEL,
        "target_model_bytes": BASE_MODEL_SIZE,
        "expected_mtp_path": MISSING_MTP,
        "mtp_candidates": [],
        "candidate_search_stdout": "mtp_candidates=\n",
        "stream_case": "b300_missing_mtp_support_model",
        "runtime_guard_case": "b300_missing_mtp_support_runtime_blocker",
        "target_stream_visibility": "blocked_before_stream",
        "accepted_stream_delta": "blocked_before_stream",
        "checkpoint_delta": "0",
        "logits_source": "none",
        "cache_kvc_visibility": "none",
        "fallback": BLOCKER,
        "error": BLOCKER,
    }
    for key, expected_value in expected.items():
        report.check(value.get(key) == expected_value, f"explicit blocker {key} drift")


def validate_claim_policy(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "claim policy missing")
    if not isinstance(value, dict):
        return
    report.check(
        value.get("mtp_enabled_parity") == "blocked_until_support_gguf_exists",
        "MTP-enabled parity policy drift",
    )
    report.check(value.get("allowed_closure") == "explicit_support_artifact_blocker", "closure policy drift")
    report.check(value.get("must_not_report_as") == ["MTP-off pass", "MTP-enabled parity"], "forbidden claims drift")


def validate_upstream(
    report: Report,
    branch_decision: dict[str, Any],
    stream_contract: dict[str, Any],
    runtime_blocker: dict[str, Any],
) -> None:
    report.check(branch_decision.get("selected_branch") == SELECTED_BRANCH, "branch decision selected branch drift")
    report.check(branch_decision.get("next_stage") == MILESTONE, "branch decision next stage drift")
    support = branch_decision.get("support_artifacts")
    report.check(isinstance(support, dict), "branch decision support artifacts missing")
    if isinstance(support, dict):
        report.check(support.get("candidate_count") == 0, "branch decision candidate count drift")
        report.check(support.get("availability") == BLOCKER, "branch decision availability drift")
    policy = branch_decision.get("claim_policy")
    report.check(policy == {
        "mtp_enabled_parity": "blocked_until_support_gguf_exists",
        "allowed_closure": "explicit_support_artifact_blocker",
        "must_not_report_as": ["MTP-off pass", "MTP-enabled parity"],
    }, "branch decision claim policy drift")

    stream_b300 = stream_contract.get("b300")
    report.check(isinstance(stream_b300, dict), "M10.8g1 B300 metadata missing")
    if isinstance(stream_b300, dict):
        report.check(stream_b300.get("candidate_count") == 0, "M10.8g1 candidate count drift")
        report.check(stream_b300.get("availability") == BLOCKER, "M10.8g1 availability drift")
    stream_cases = {
        item.get("id"): item for item in stream_contract.get("stream_cases", []) if isinstance(item, dict)
    }
    stream_case = stream_cases.get("b300_missing_mtp_support_model")
    report.check(isinstance(stream_case, dict), "M10.8g1 missing stream blocker case")
    if isinstance(stream_case, dict):
        report.check(stream_case.get("accepted_stream_delta") == "blocked_before_stream", "stream blocker delta drift")
        report.check(stream_case.get("checkpoint_delta") == "0", "stream blocker checkpoint drift")
        report.check(stream_case.get("cache_kvc_visibility") == "none", "stream blocker cache/KVC drift")
        report.check(stream_case.get("error") == BLOCKER, "stream blocker error drift")

    runtime = runtime_blocker.get("runtime")
    report.check(isinstance(runtime, dict), "M10.8g3c runtime blocker missing")
    if isinstance(runtime, dict):
        report.check(runtime.get("target_stream_visibility") == "blocked_before_stream", "runtime visibility drift")
        report.check(runtime.get("checkpoint_delta") == "0", "runtime checkpoint drift")
        report.check(runtime.get("cache_kvc_visibility") == "none", "runtime cache/KVC drift")
        report.check(runtime.get("error") == BLOCKER, "runtime error drift")


def validate_static_wiring(report: Report) -> None:
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()
    todo = (ROOT / ".memory/TODO.md").read_text()
    report.check("compare_mtp_end_to_end_closure.py" in run_report, "unified report missing comparator")
    report.check("M10.8g4b B300 MTP end-to-end closure rerun" in run_report, "B300 rerun hook missing")
    report.check("compare_mtp_end_to_end_closure.py --negative-test" in readme, "README missing comparator command")
    report.check("M10.8g4b: B300 End-To-End Blocker Or Support Comparator Closure" in todo, "TODO missing M10.8g4b")


def run_negative_tests(
    summary: dict[str, Any],
    branch_decision: dict[str, Any],
    stream_contract: dict[str, Any],
    runtime_blocker: dict[str, Any],
) -> Report:
    report = Report()
    mutations = [
        (
            "branch flips to support-present",
            lambda data, _branch, _stream, _runtime: data.update(
                {"selected_branch": "support_present_end_to_end_comparator"}
            ),
        ),
        (
            "support comparator claims pass",
            lambda data, _branch, _stream, _runtime: data["support_present_comparator"].update({"status": "pass"}),
        ),
        (
            "blocker result removed",
            lambda data, _branch, _stream, _runtime: data["explicit_blocker"].update({"result": "none"}),
        ),
        (
            "candidate appears upstream",
            lambda _data, branch, _stream, _runtime: branch["support_artifacts"].update(
                {"mtp_candidates": ["/workspace/ds4/draft.gguf"], "candidate_count": 1}
            ),
        ),
        (
            "runtime blocker drift",
            lambda _data, _branch, _stream, runtime: runtime["runtime"].update(
                {"target_stream_visibility": "target_only"}
            ),
        ),
        (
            "claim policy drift",
            lambda data, _branch, _stream, _runtime: data["claim_policy"].update(
                {"mtp_enabled_parity": "complete"}
            ),
        ),
        (
            "next stage drift",
            lambda data, _branch, _stream, _runtime: data.update({"next_stage": "M10.8g4b"}),
        ),
    ]
    for name, mutate in mutations:
        summary_copy = copy.deepcopy(summary)
        branch_copy = copy.deepcopy(branch_decision)
        stream_copy = copy.deepcopy(stream_contract)
        runtime_copy = copy.deepcopy(runtime_blocker)
        mutate(summary_copy, branch_copy, stream_copy, runtime_copy)
        result = validate(summary_copy, branch_copy, stream_copy, runtime_copy)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--branch-decision", type=Path, default=BRANCH_DECISION)
    parser.add_argument("--stream-contract", type=Path, default=STREAM_CONTRACT)
    parser.add_argument("--runtime-blocker", type=Path, default=RUNTIME_BLOCKER)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        branch_decision = load_json(args.branch_decision)
        stream_contract = load_json(args.stream_contract)
        runtime_blocker = load_json(args.runtime_blocker)
        summary = (
            build_summary(branch_decision, stream_contract, runtime_blocker)
            if args.write_summary
            else load_json(args.summary)
        )
    except Exception as exc:
        print(f"MTP end-to-end closure: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(summary, branch_decision, stream_contract, runtime_blocker)
    if not report.ok:
        print("MTP end-to-end closure: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    if args.write_summary:
        write_json(args.write_summary, summary)
    print(f"MTP end-to-end closure: PASS, {report.checks} checks")
    if args.negative_test:
        negative = run_negative_tests(summary, branch_decision, stream_contract, runtime_blocker)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("MTP end-to-end closure negative tests: PASS, 7 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
