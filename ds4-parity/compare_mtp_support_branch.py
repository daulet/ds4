#!/usr/bin/env python3
"""Validate the M10.8g4a B300 MTP support-artifact branch decision."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.8g4a/support-branch-decision.json"
STREAM_CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json"
RUNTIME_BLOCKER = ROOT / "ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json"

SCHEMA = "ds4.mtp_support_branch_decision.v1"
SOURCE = "b300-mtp-support-artifact-branch-decision"
MILESTONE = "M10.8g4a"
B300_CONTEXT = "hou2-prod1"
B300_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
BASE_MODEL = "/workspace/ds4/ds4flash.gguf"
BASE_MODEL_SIZE = 86_720_111_488
MISSING_MTP = "/workspace/ds4/missing-mtp.gguf"
CANDIDATE_GLOBS = ["*mtp*.gguf", "*draft*.gguf"]
SELECTED_BRANCH = "support_absent_blocker_closure"
NEXT_STAGE = "M10.8g4b"


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


def find_support_candidates(workdir: Path) -> list[str]:
    candidates: list[str] = []
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(workdir)
        except ValueError:
            continue
        if len(relative.parts) > 3:
            continue
        lower = path.name.lower()
        if lower.endswith(".gguf") and (("mtp" in lower) or ("draft" in lower)):
            candidates.append(path.as_posix())
    return sorted(candidates)


def format_candidate_stdout(candidates: list[str]) -> str:
    return f"mtp_candidates={' '.join(candidates)}\n"


def build_live_summary(workdir: Path) -> dict[str, Any]:
    base_path = Path(BASE_MODEL)
    missing_path = Path(MISSING_MTP)
    candidates = find_support_candidates(workdir)
    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "b300": {
            "context": B300_CONTEXT,
            "namespace": "default",
            "pod": B300_POD,
            "workdir": B300_WORKDIR,
        },
        "inputs": {
            "stream_contract": rel(STREAM_CONTRACT),
            "runtime_missing_support_summary": rel(RUNTIME_BLOCKER),
        },
        "support_artifacts": {
            "base_model_path": BASE_MODEL,
            "base_model_exists": base_path.exists(),
            "base_model_bytes": base_path.stat().st_size if base_path.exists() else None,
            "expected_mtp_path": MISSING_MTP,
            "expected_mtp_path_exists": missing_path.exists(),
            "candidate_globs": CANDIDATE_GLOBS,
            "candidate_max_depth": 3,
            "mtp_candidates": candidates,
            "candidate_count": len(candidates),
            "candidate_search_stdout": format_candidate_stdout(candidates),
            "availability": "blocked_missing_mtp_model" if not candidates else "support_candidates_present",
        },
        "selected_branch": SELECTED_BRANCH if not candidates else "support_present_end_to_end_comparator",
        "next_stage": NEXT_STAGE,
        "claim_policy": {
            "mtp_enabled_parity": "blocked_until_support_gguf_exists",
            "allowed_closure": "explicit_support_artifact_blocker",
            "must_not_report_as": ["MTP-off pass", "MTP-enabled parity"],
        },
    }


def validate(summary: dict[str, Any], stream_contract: dict[str, Any], runtime_blocker: dict[str, Any]) -> Report:
    report = Report()
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == SOURCE, "summary source drift")
    report.check(summary.get("milestone") == MILESTONE, "summary milestone drift")
    validate_b300(report, summary.get("b300"))
    validate_inputs(report, summary.get("inputs"))
    validate_support_artifacts(report, summary.get("support_artifacts"))
    validate_branch(report, summary)
    validate_upstream_blockers(report, stream_contract, runtime_blocker)
    validate_static_wiring(report)
    return report


def validate_b300(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "b300 metadata missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("context") == B300_CONTEXT, "B300 context drift")
    report.check(value.get("namespace") == "default", "B300 namespace drift")
    report.check(value.get("pod") == B300_POD, "B300 pod drift")
    report.check(value.get("workdir") == B300_WORKDIR, "B300 workdir drift")


def validate_inputs(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "inputs missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("stream_contract") == rel(STREAM_CONTRACT), "stream contract input drift")
    report.check(
        value.get("runtime_missing_support_summary") == rel(RUNTIME_BLOCKER),
        "runtime blocker input drift",
    )


def validate_support_artifacts(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "support artifact metadata missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("base_model_path") == BASE_MODEL, "base model path drift")
    report.check(value.get("base_model_exists") is True, "base model should exist")
    report.check(value.get("base_model_bytes") == BASE_MODEL_SIZE, "base model size drift")
    report.check(value.get("expected_mtp_path") == MISSING_MTP, "missing-MTP path drift")
    report.check(value.get("expected_mtp_path_exists") is False, "missing-MTP path unexpectedly exists")
    report.check(value.get("candidate_globs") == CANDIDATE_GLOBS, "candidate globs drift")
    report.check(value.get("candidate_max_depth") == 3, "candidate max depth drift")
    report.check(value.get("mtp_candidates") == [], "support candidates unexpectedly present")
    report.check(value.get("candidate_count") == 0, "support candidate count drift")
    report.check(value.get("candidate_search_stdout") == "mtp_candidates=\n", "candidate stdout drift")
    report.check(value.get("availability") == "blocked_missing_mtp_model", "availability drift")


def validate_branch(report: Report, summary: dict[str, Any]) -> None:
    report.check(summary.get("selected_branch") == SELECTED_BRANCH, "selected branch drift")
    report.check(summary.get("next_stage") == NEXT_STAGE, "next stage drift")
    policy = summary.get("claim_policy")
    report.check(isinstance(policy, dict), "claim policy missing")
    if not isinstance(policy, dict):
        return
    report.check(
        policy.get("mtp_enabled_parity") == "blocked_until_support_gguf_exists",
        "MTP-enabled parity policy drift",
    )
    report.check(policy.get("allowed_closure") == "explicit_support_artifact_blocker", "closure policy drift")
    report.check(policy.get("must_not_report_as") == ["MTP-off pass", "MTP-enabled parity"], "forbidden claims drift")


def validate_upstream_blockers(
    report: Report,
    stream_contract: dict[str, Any],
    runtime_blocker: dict[str, Any],
) -> None:
    b300 = stream_contract.get("b300")
    report.check(isinstance(b300, dict), "M10.8g1 B300 metadata missing")
    if isinstance(b300, dict):
        report.check(b300.get("candidate_count") == 0, "M10.8g1 candidate count drift")
        report.check(b300.get("availability") == "blocked_missing_mtp_model", "M10.8g1 availability drift")
    mtp = (stream_contract.get("support_artifacts") or {}).get("mtp")
    report.check(isinstance(mtp, dict), "M10.8g1 support artifact metadata missing")
    if isinstance(mtp, dict):
        report.check(mtp.get("available") is False, "M10.8g1 support availability drift")
        report.check(mtp.get("candidate_count") == 0, "M10.8g1 support candidate drift")

    artifacts = runtime_blocker.get("support_artifacts")
    runtime = runtime_blocker.get("runtime")
    report.check(isinstance(artifacts, dict), "M10.8g3c support artifact metadata missing")
    report.check(isinstance(runtime, dict), "M10.8g3c runtime metadata missing")
    if isinstance(artifacts, dict):
        report.check(artifacts.get("candidate_count") == 0, "M10.8g3c candidate count drift")
        report.check(artifacts.get("mtp_candidates") == [], "M10.8g3c support candidates drift")
        report.check(artifacts.get("availability") == "blocked_missing_mtp_model", "M10.8g3c availability drift")
    if isinstance(runtime, dict):
        report.check(runtime.get("target_stream_visibility") == "blocked_before_stream", "M10.8g3c visibility drift")
        report.check(runtime.get("checkpoint_delta") == "0", "M10.8g3c checkpoint drift")
        report.check(runtime.get("cache_kvc_visibility") == "none", "M10.8g3c cache/KVC drift")
        report.check(runtime.get("error") == "blocked_missing_mtp_model", "M10.8g3c error drift")


def validate_static_wiring(report: Report) -> None:
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()
    todo = (ROOT / ".memory/TODO.md").read_text()
    report.check("compare_mtp_support_branch.py" in run_report, "unified report missing comparator")
    report.check("M10.8g4a B300 MTP support branch decision rerun" in run_report, "B300 rerun hook missing")
    report.check("compare_mtp_support_branch.py --negative-test" in readme, "README missing comparator command")
    report.check("M10.8g4a: B300 Support-Artifact Branch Decision" in todo, "TODO missing M10.8g4a")


def run_negative_tests(
    summary: dict[str, Any],
    stream_contract: dict[str, Any],
    runtime_blocker: dict[str, Any],
) -> Report:
    report = Report()
    mutations = [
        (
            "support candidate appears",
            lambda data, _stream, _runtime: data["support_artifacts"].update(
                {"mtp_candidates": ["/workspace/ds4/draft.gguf"], "candidate_count": 1}
            ),
        ),
        (
            "branch drift",
            lambda data, _stream, _runtime: data.update(
                {"selected_branch": "support_present_end_to_end_comparator"}
            ),
        ),
        (
            "claim policy drift",
            lambda data, _stream, _runtime: data["claim_policy"].update({"mtp_enabled_parity": "complete"}),
        ),
        (
            "M10.8g1 availability drift",
            lambda _data, stream, _runtime: stream["b300"].update({"availability": "available"}),
        ),
        (
            "M10.8g3c runtime drift",
            lambda _data, _stream, runtime: runtime["runtime"].update({"target_stream_visibility": "target_only"}),
        ),
        (
            "input drift",
            lambda data, _stream, _runtime: data["inputs"].update({"runtime_missing_support_summary": "missing.json"}),
        ),
    ]
    for name, mutate in mutations:
        summary_copy = copy.deepcopy(summary)
        stream_copy = copy.deepcopy(stream_contract)
        runtime_copy = copy.deepcopy(runtime_blocker)
        mutate(summary_copy, stream_copy, runtime_copy)
        result = validate(summary_copy, stream_copy, runtime_copy)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--stream-contract", type=Path, default=STREAM_CONTRACT)
    parser.add_argument("--runtime-blocker", type=Path, default=RUNTIME_BLOCKER)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        stream_contract = load_json(args.stream_contract)
        runtime_blocker = load_json(args.runtime_blocker)
        summary = build_live_summary(args.workdir) if args.live else load_json(args.summary)
    except Exception as exc:
        print(f"MTP support branch decision: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(summary, stream_contract, runtime_blocker)
    if not report.ok:
        print("MTP support branch decision: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    if args.write_summary:
        write_json(args.write_summary, summary)
    print(f"MTP support branch decision: PASS, {report.checks} checks")
    if args.negative_test:
        negative = run_negative_tests(summary, stream_contract, runtime_blocker)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("MTP support branch decision negative tests: PASS, 6 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
