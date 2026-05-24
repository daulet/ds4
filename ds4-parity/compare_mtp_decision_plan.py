#!/usr/bin/env python3
"""Compare the Rust MTP decision planner against the M10.8a contract."""

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
CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json"
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-decision-plan.rs"

EXPECTED_SCHEMA = "ds4.rust_mtp_decision_plan.v1"
EXPECTED_SOURCE = "rust-model-free-mtp-decision-planner"
COMPARE_KEYS = (
    "path",
    "frontier_ops",
    "accepted_suffix",
    "checkpoint_action",
    "logits_source",
    "mtp_n_raw_keep",
    "fallback",
)


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


def run_rust_plan() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-mtp-decision-plan",
            "--quiet",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def validate(candidate: dict[str, Any], contract: dict[str, Any]) -> Report:
    report = Report()
    report.check(candidate.get("schema") == EXPECTED_SCHEMA, "schema drift")
    report.check(candidate.get("source") == EXPECTED_SOURCE, "source drift")
    report.check(
        candidate.get("oracle_path")
        == "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json",
        "oracle path drift",
    )

    contract_cases = named_cases(report, contract.get("decision_cases"), "contract")
    rust_cases = named_cases(report, candidate.get("cases"), "rust")
    report.check(set(rust_cases) == set(contract_cases), "case id set drift")
    report.check(list(rust_cases) == list(contract_cases), "case order drift")

    for case_id, rust_case in rust_cases.items():
        contract_case = contract_cases.get(case_id)
        if contract_case is None:
            continue
        for key in COMPARE_KEYS:
            report.check(
                rust_case.get(key) == contract_case.get(key),
                f"{case_id}.{key}: expected {contract_case.get(key)!r}, got {rust_case.get(key)!r}",
            )
        check_fail_closed(report, case_id, rust_case)
    static_checks(report, contract_cases)
    return report


def named_cases(report: Report, value: Any, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    report.check(isinstance(value, list), f"{label}.cases must be a list")
    if not isinstance(value, list):
        return result
    for index, item in enumerate(value):
        report.check(isinstance(item, dict), f"{label}.cases[{index}] must be an object")
        if not isinstance(item, dict):
            continue
        case_id = item.get("id")
        report.check(isinstance(case_id, str), f"{label}.cases[{index}].id missing")
        if not isinstance(case_id, str):
            continue
        report.check(case_id not in result, f"{label}.cases duplicate {case_id}")
        result[case_id] = item
    return result


def check_fail_closed(report: Report, case_id: str, rust_case: dict[str, Any]) -> None:
    expected_fail_closed = {
        "b300_missing_mtp_support_model",
        "mtp_disabled_after_first_token",
        "first_draft_miss",
        "margin_skip_single_target_replay",
        "exact_decode2_failure_restore_then_sequential",
        "suffix_restore_replay_accept",
        "suffix_failure_restore_or_error",
        "sequential_safety_fallback",
    }
    expected = case_id in expected_fail_closed
    report.check(rust_case.get("fail_closed") is expected, f"{case_id}.fail_closed drift")


def static_checks(report: Report, contract_cases: dict[str, dict[str, Any]]) -> None:
    rust_source = RUST_SOURCE.read_text()
    rust_bin = RUST_BIN.read_text()
    lib_source = (ROOT / "rust/ds4-gpu/src/lib.rs").read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()

    report.check("pub mod mtp_plan;" in lib_source, "mtp_plan module not exported")
    report.check("pub enum MtpScenario" in rust_source, "MtpScenario enum missing")
    report.check("pub const MTP_SCENARIOS" in rust_source, "MTP_SCENARIOS missing")
    report.check("pub const fn plan_scenario" in rust_source, "plan_scenario missing")
    report.check("fn write_json_string" in rust_bin, "planner bin JSON escaping helper missing")
    report.check("compare_mtp_decision_plan.py" in run_report, "unified report missing M10.8b comparator")
    report.check("compare_mtp_decision_plan.py" in readme, "README missing M10.8b comparator")
    for case_id in contract_cases:
        report.check(case_id in rust_source, f"Rust planner source missing case {case_id}")


def run_negative_tests(candidate: dict[str, Any], contract: dict[str, Any]) -> int:
    mutations = [
        ("schema drift", lambda c: c.__setitem__("schema", "wrong")),
        ("missing rust case", lambda c: c["cases"].pop()),
        ("path drift", lambda c: find_case(c, "exact_decode2_full_accept").__setitem__("path", "wrong")),
        (
            "accepted suffix drift",
            lambda c: find_case(c, "exact_decode2_full_accept").__setitem__("accepted_suffix", 1),
        ),
        (
            "frontier op drift",
            lambda c: find_case(c, "suffix_prefix1_accept").__setitem__("frontier_ops", ["keep_accepted"]),
        ),
        (
            "fail-closed drift",
            lambda c: find_case(c, "b300_missing_mtp_support_model").__setitem__("fail_closed", False),
        ),
    ]
    passed = 0
    for name, mutate in mutations:
        candidate_copy = copy.deepcopy(candidate)
        mutate(candidate_copy)
        report = validate(candidate_copy, contract)
        if report.ok:
            print(f"negative-test failed to detect mutation: {name}", file=sys.stderr)
        else:
            passed += 1
    if passed != len(mutations):
        return 1
    print(f"Rust MTP decision planner negative tests: PASS, {passed} mutations")
    return 0


def find_case(candidate: dict[str, Any], case_id: str) -> dict[str, Any]:
    for item in candidate["cases"]:
        if item.get("id") == case_id:
            return item
    raise KeyError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        candidate = load_json(args.candidate) if args.candidate else run_rust_plan()
        contract = load_json(args.contract)
    except Exception as exc:
        print(f"Rust MTP decision planner comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(candidate, contract)
    if not report.ok:
        print("Rust MTP decision planner comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "Rust MTP decision planner comparator: PASS, "
        f"{len(candidate['cases'])} cases, {report.checks} checks"
    )
    if args.negative_test:
        return run_negative_tests(candidate, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
