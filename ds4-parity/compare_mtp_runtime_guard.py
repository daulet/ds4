#!/usr/bin/env python3
"""Compare the Rust MTP runtime guard plan against M10.8g2 stream outcomes."""

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
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_runtime_guard_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-runtime-guard-plan.rs"
STREAM_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_stream_plan.rs"

EXPECTED_SCHEMA = "ds4.rust_mtp_runtime_guard_plan.v1"
EXPECTED_SOURCE = "rust-model-free-mtp-runtime-guard-planner"
EXPECTED_CASES = [
    "engine_options_default_mtp_off",
    "one_shot_runtime_mtp_off",
    "interactive_runtime_mtp_off",
    "server_runtime_mtp_off",
    "argmax_session_runtime_non_mtp",
    "first_draft_miss_no_drift",
    "b300_missing_mtp_support_runtime_blocker",
]
STREAM_KEYS = (
    "accepted_stream_delta",
    "checkpoint_delta",
    "logits_source",
    "mtp_n_raw_keep",
    "cache_kvc_visibility",
    "fallback",
    "error",
    "live_status",
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


def run_json(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    obj = json.loads(proc.stdout)
    if not isinstance(obj, dict):
        raise TypeError("expected JSON object")
    return obj


def run_guard_plan() -> dict[str, Any]:
    return run_json(["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-runtime-guard-plan", "--quiet"])


def run_stream_plan() -> dict[str, Any]:
    return run_json(["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-stream-plan", "--quiet"])


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected JSON object")
    return obj


def validate(candidate: dict[str, Any], stream: dict[str, Any]) -> Report:
    report = Report()
    report.check(candidate.get("schema") == EXPECTED_SCHEMA, "schema drift")
    report.check(candidate.get("source") == EXPECTED_SOURCE, "source drift")
    report.check(
        candidate.get("oracle") == "M10.8g2 unavailable stream outcomes plus runtime source anchors",
        "oracle drift",
    )
    guard_cases = named_cases(report, candidate.get("cases"), "runtime_guard")
    stream_cases = named_cases(report, stream.get("cases"), "stream")
    report.check(list(guard_cases) == EXPECTED_CASES, "runtime guard case order drift")

    for case_id, case in guard_cases.items():
        stream_case_name = case.get("source_stream_case")
        report.check(isinstance(stream_case_name, str), f"{case_id}.source_stream_case missing")
        stream_case = stream_cases.get(stream_case_name)
        report.check(stream_case is not None, f"{case_id}: source stream case missing")
        if stream_case is not None:
            for key in STREAM_KEYS:
                report.check(
                    normalize(case.get(key)) == normalize(stream_case.get(key)),
                    f"{case_id}.{key}: does not match {stream_case_name}",
                )
        report.check(
            case.get("selected_stream_plan") == f"mtp_stream_plan:{stream_case_name}",
            f"{case_id}.selected_stream_plan drift",
        )
        check_case_invariants(report, case_id, case)
        check_source_anchors(report, case_id, case)

    static_checks(report)
    return report


def normalize(value: Any) -> Any:
    if isinstance(value, int):
        return str(value)
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


def check_case_invariants(report: Report, case_id: str, case: dict[str, Any]) -> None:
    if case_id == "b300_missing_mtp_support_runtime_blocker":
        report.check(case.get("target_stream_visibility") == "blocked_before_stream", f"{case_id}.visibility drift")
        report.check(case.get("accepted_stream_delta") == "blocked_before_stream", f"{case_id}.stream drift")
        report.check(case.get("checkpoint_delta") == "0", f"{case_id}.checkpoint drift")
        report.check(case.get("error") == "blocked_missing_mtp_model", f"{case_id}.error drift")
        return
    report.check(case.get("accepted_stream_delta") == "first_token", f"{case_id}.stream drift")
    report.check(case.get("checkpoint_delta") == "1", f"{case_id}.checkpoint drift")
    report.check(case.get("mtp_n_raw_keep") in (0, "0"), f"{case_id}.mtp_n_raw drift")
    report.check(case.get("error") == "none", f"{case_id}.error drift")
    report.check(
        str(case.get("target_stream_visibility", "")).startswith("target_only"),
        f"{case_id}.visibility drift",
    )


def check_source_anchors(report: Report, case_id: str, case: dict[str, Any]) -> None:
    anchors = case.get("source_anchors")
    report.check(isinstance(anchors, list) and anchors, f"{case_id}.source_anchors missing")
    if not isinstance(anchors, list):
        return
    for anchor in anchors:
        report.check(isinstance(anchor, str), f"{case_id}.source_anchor must be a string")
        if not isinstance(anchor, str):
            continue
        path_text, sep, snippet = anchor.partition("::")
        report.check(bool(sep) and bool(path_text) and bool(snippet), f"{case_id}: malformed anchor {anchor!r}")
        if not sep:
            continue
        path = ROOT / path_text
        report.check(path.is_file(), f"{case_id}: missing anchor file {path_text}")
        if path.is_file():
            report.check(snippet in path.read_text(), f"{case_id}: missing anchor snippet {anchor!r}")

    if case_id == "argmax_session_runtime_non_mtp":
        for rel in [
            "rust/ds4-engine/src/bin/ds4-argmax-runtime-rs.rs",
            "rust/ds4-engine/src/bin/ds4-session-runtime-rs.rs",
        ]:
            text = (ROOT / rel).read_text()
            report.check("--mtp" not in text, f"{case_id}: {rel} unexpectedly exposes --mtp")


def static_checks(report: Report) -> None:
    rust_source = RUST_SOURCE.read_text()
    rust_bin = RUST_BIN.read_text()
    lib_source = (ROOT / "rust/ds4-gpu/src/lib.rs").read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()
    stream_source = STREAM_SOURCE.read_text()

    for snippet in [
        "pub mod mtp_runtime_guard_plan;",
        "pub struct MtpRuntimeGuardPlan",
        "pub const MTP_RUNTIME_GUARD_CASES",
        "runtime_guard_case_matches_stream",
    ]:
        report.check(snippet in rust_source + lib_source, f"Rust guard source missing {snippet!r}")
    for snippet in ["ds4.rust_mtp_runtime_guard_plan.v1", "fn write_json_string"]:
        report.check(snippet in rust_bin, f"Rust guard bin missing {snippet!r}")
    for stream_case in [
        "b300_missing_mtp_support_model",
        "mtp_disabled_after_first_token",
        "first_draft_miss",
    ]:
        report.check(stream_case in stream_source, f"M10.8g2 stream source missing {stream_case}")
    for snippet in ["compare_mtp_runtime_guard.py", "M10.8g3a Rust MTP runtime guard plan"]:
        report.check(snippet in run_report, f"unified report missing {snippet!r}")
    report.check(
        "compare_mtp_runtime_guard.py --negative-test" in readme,
        "README missing M10.8g3a command",
    )


def run_negative_tests(candidate: dict[str, Any], stream: dict[str, Any]) -> Report:
    report = Report()
    mutations = [
        ("schema", lambda data: data.update({"schema": "drift"})),
        ("missing case", lambda data: data["cases"].pop(2)),
        (
            "stream case drift",
            lambda data: mutate_case(data, "first_draft_miss_no_drift", "source_stream_case", "suffix_full_accept"),
        ),
        (
            "stream delta drift",
            lambda data: mutate_case(data, "one_shot_runtime_mtp_off", "accepted_stream_delta", "first_token + draft"),
        ),
        (
            "checkpoint drift",
            lambda data: mutate_case(data, "server_runtime_mtp_off", "checkpoint_delta", "2"),
        ),
        (
            "anchor drift",
            lambda data: mutate_case(data, "engine_options_default_mtp_off", "source_anchors", ["rust/ds4-engine/src/lib.rs::missing anchor"]),
        ),
        (
            "blocker drift",
            lambda data: mutate_case(data, "b300_missing_mtp_support_runtime_blocker", "error", "none"),
        ),
    ]
    for name, mutate in mutations:
        mutated = copy.deepcopy(candidate)
        mutate(mutated)
        result = validate(mutated, stream)
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
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        candidate = load_json(args.candidate) if args.candidate else run_guard_plan()
        stream = load_json(args.stream) if args.stream else run_stream_plan()
    except Exception as exc:
        print(f"Rust MTP runtime guard comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(candidate, stream)
    if not report.ok:
        print("Rust MTP runtime guard comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "Rust MTP runtime guard comparator: PASS, "
        f"{len(candidate['cases'])} cases, {report.checks} checks"
    )
    if args.negative_test:
        negative = run_negative_tests(candidate, stream)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Rust MTP runtime guard negative tests: PASS, 7 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
