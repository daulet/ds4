#!/usr/bin/env python3
"""Compare the Rust MTP stream outcome plan against the M10.8g1 contract."""

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
CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json"
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_stream_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-stream-plan.rs"

EXPECTED_SCHEMA = "ds4.rust_mtp_stream_plan.v1"
EXPECTED_SOURCE = "rust-model-free-mtp-stream-outcome-planner"
COMPARE_KEYS = (
    "source_case",
    "path",
    "accepted_suffix",
    "accepted_stream_delta",
    "checkpoint_delta",
    "logits_source",
    "frontier_ops",
    "mtp_n_raw_keep",
    "cache_kvc_visibility",
    "fallback",
    "error",
    "live_status",
)
EXPECTED_SUBPLANS = {
    "b300_missing_mtp_support_model": [
        "mtp_plan:b300_missing_mtp_support_model",
        "mtp_draft_plan:b300_missing_mtp_support_model",
        "mtp_decode2_plan:b300_missing_mtp_support_model",
        "mtp_suffix_plan:b300_missing_mtp_support_model",
        "mtp_frontier_plan:b300_missing_mtp_support_model",
    ],
    "mtp_disabled_after_first_token": ["mtp_plan:mtp_disabled_after_first_token"],
    "first_draft_miss": [
        "mtp_plan:first_draft_miss",
        "mtp_draft_plan:first_draft_from_current_hc",
    ],
    "margin_skip_single_target_replay": [
        "mtp_plan:margin_skip_single_target_replay",
        "mtp_draft_plan:first_draft_from_current_hc",
    ],
    "exact_decode2_full_accept": [
        "mtp_plan:exact_decode2_full_accept",
        "mtp_decode2_plan:exact_decode2_full_accept",
        "mtp_frontier_plan:snapshot_compressed_attn_frontier",
    ],
    "exact_decode2_prefix1_accept": [
        "mtp_plan:exact_decode2_prefix1_accept",
        "mtp_decode2_plan:exact_decode2_prefix1_accept",
        "mtp_frontier_plan:snapshot_compressed_attn_frontier",
        "mtp_frontier_plan:prefix1_commit_compressed_attn_frontier",
        "mtp_frontier_plan:prefix1_commit_ratio4_index_frontier",
    ],
    "exact_decode2_failure_restore_then_sequential": [
        "mtp_plan:exact_decode2_failure_restore_then_sequential",
        "mtp_decode2_plan:exact_decode2_failure_restore_then_sequential",
        "mtp_frontier_plan:restore_compressed_attn_frontier",
        "mtp_frontier_plan:restore_ratio4_index_frontier",
    ],
    "suffix_full_accept": [
        "mtp_plan:suffix_full_accept",
        "mtp_suffix_plan:suffix_full_accept",
    ],
    "suffix_prefix1_accept": [
        "mtp_plan:suffix_prefix1_accept",
        "mtp_suffix_plan:suffix_prefix1_accept",
        "mtp_frontier_plan:prefix1_commit_compressed_attn_frontier",
        "mtp_frontier_plan:prefix1_commit_ratio4_index_frontier",
    ],
    "suffix_restore_replay_accept": [
        "mtp_plan:suffix_restore_replay_accept",
        "mtp_suffix_plan:suffix_restore_replay_accept",
        "mtp_frontier_plan:snapshot_compressed_attn_frontier",
        "mtp_frontier_plan:restore_compressed_attn_frontier",
        "mtp_frontier_plan:restore_ratio4_index_frontier",
    ],
    "suffix_failure_restore_or_error": [
        "mtp_plan:suffix_failure_restore_or_error",
        "mtp_suffix_plan:suffix_failure_restore_or_error",
        "mtp_frontier_plan:restore_compressed_attn_frontier",
        "mtp_frontier_plan:restore_ratio4_index_frontier",
    ],
    "sequential_safety_fallback": ["mtp_plan:sequential_safety_fallback"],
}
SUBPLAN_SOURCES = {
    "mtp_plan": ROOT / "rust/ds4-gpu/src/mtp_plan.rs",
    "mtp_draft_plan": ROOT / "rust/ds4-gpu/src/mtp_draft_plan.rs",
    "mtp_decode2_plan": ROOT / "rust/ds4-gpu/src/mtp_decode2_plan.rs",
    "mtp_suffix_plan": ROOT / "rust/ds4-gpu/src/mtp_suffix_plan.rs",
    "mtp_frontier_plan": ROOT / "rust/ds4-gpu/src/mtp_frontier_plan.rs",
}


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
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-stream-plan", "--quiet"],
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
        == "ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json",
        "oracle path drift",
    )

    contract_cases = named_cases(report, contract.get("stream_cases"), "contract")
    rust_cases = named_cases(report, candidate.get("cases"), "rust")
    report.check(list(rust_cases) == list(contract_cases), "case order drift")
    for case_id, rust_case in rust_cases.items():
        contract_case = contract_cases.get(case_id)
        if contract_case is None:
            report.check(False, f"unexpected case {case_id}")
            continue
        for key in COMPARE_KEYS:
            report.check(
                normalize_value(rust_case.get(key)) == normalize_value(contract_case.get(key)),
                f"{case_id}.{key}: expected {contract_case.get(key)!r}, got {rust_case.get(key)!r}",
            )
        expected_subplans = EXPECTED_SUBPLANS.get(case_id)
        report.check(expected_subplans is not None, f"missing expected subplans for {case_id}")
        report.check(
            rust_case.get("selected_subplans") == expected_subplans,
            f"{case_id}.selected_subplans drift",
        )
    check_subplan_links(report, rust_cases)
    static_checks(report, contract_cases)
    return report


def normalize_value(value: Any) -> Any:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value


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
        if isinstance(case_id, str):
            result[case_id] = item
    return result


def check_subplan_links(report: Report, rust_cases: dict[str, dict[str, Any]]) -> None:
    source_text = {name: path.read_text() for name, path in SUBPLAN_SOURCES.items()}
    for case_id, case in rust_cases.items():
        subplans = case.get("selected_subplans")
        report.check(isinstance(subplans, list), f"{case_id}.selected_subplans must be a list")
        if not isinstance(subplans, list):
            continue
        for subplan in subplans:
            report.check(isinstance(subplan, str), f"{case_id}.selected_subplans item must be string")
            if not isinstance(subplan, str):
                continue
            module, _, subcase = subplan.partition(":")
            report.check(module in source_text and bool(subcase), f"{case_id}: malformed subplan {subplan!r}")
            if module in source_text and subcase:
                report.check(
                    subcase in source_text[module],
                    f"{case_id}: {subplan!r} missing from {module}",
                )


def static_checks(report: Report, contract_cases: dict[str, dict[str, Any]]) -> None:
    rust_source = RUST_SOURCE.read_text()
    rust_bin = RUST_BIN.read_text()
    lib_source = (ROOT / "rust/ds4-gpu/src/lib.rs").read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()

    for snippet in [
        "pub mod mtp_stream_plan;",
        "pub struct MtpStreamOutcomePlan",
        "pub const MTP_STREAM_OUTCOME_CASES",
        "stream_case_by_id",
    ]:
        report.check(snippet in rust_source + lib_source, f"Rust stream source missing {snippet!r}")
    for snippet in ["ds4.rust_mtp_stream_plan.v1", "fn write_json_string"]:
        report.check(snippet in rust_bin, f"Rust stream bin missing {snippet!r}")
    for case_id in contract_cases:
        report.check(case_id in rust_source, f"Rust stream plan missing case {case_id}")
    for snippet in ["compare_mtp_stream_plan.py", "M10.8g2 Rust MTP stream outcome planner"]:
        report.check(snippet in run_report, f"unified report missing {snippet!r}")
    report.check(
        "compare_mtp_stream_plan.py --negative-test" in readme,
        "README missing M10.8g2 command",
    )


def run_negative_tests(candidate: dict[str, Any], contract: dict[str, Any]) -> Report:
    report = Report()
    mutations = [
        ("schema", lambda data: data.update({"schema": "drift"})),
        ("missing case", lambda data: data["cases"].pop(1)),
        (
            "subplan drift",
            lambda data: mutate_case(data, "exact_decode2_prefix1_accept", "selected_subplans", []),
        ),
        (
            "stream delta drift",
            lambda data: mutate_case(data, "suffix_full_accept", "accepted_stream_delta", "first_token"),
        ),
        (
            "checkpoint drift",
            lambda data: mutate_case(data, "exact_decode2_full_accept", "checkpoint_delta", "2"),
        ),
        (
            "frontier drift",
            lambda data: mutate_case(data, "suffix_restore_replay_accept", "frontier_ops", ["keep_accepted"]),
        ),
        (
            "cache visibility drift",
            lambda data: mutate_case(data, "sequential_safety_fallback", "cache_kvc_visibility", "speculative"),
        ),
        (
            "blocker drift",
            lambda data: mutate_case(data, "b300_missing_mtp_support_model", "live_status", "available"),
        ),
    ]
    for name, mutate in mutations:
        mutated = copy.deepcopy(candidate)
        mutate(mutated)
        result = validate(mutated, contract)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def mutate_case(data: dict[str, Any], case_id: str, key: str, value: Any) -> None:
    for case in data["cases"]:
        if case["id"] == case_id:
            case[key] = value
            return
    raise AssertionError(case_id)


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
        print(f"Rust MTP stream plan comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(candidate, contract)
    if not report.ok:
        print("Rust MTP stream plan comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "Rust MTP stream plan comparator: PASS, "
        f"{len(candidate['cases'])} cases, {report.checks} checks"
    )
    if args.negative_test:
        negative = run_negative_tests(candidate, contract)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Rust MTP stream plan negative tests: PASS, 8 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
