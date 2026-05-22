#!/usr/bin/env python3
"""Compare the Rust decode stop-policy port against the M6.6a C oracle."""

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
BASELINE = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.6a" / "current-c.json"

RESULT_FIELDS = (
    "finish_reason",
    "completion_tokens",
    "raw_text_hex",
    "visible_text_hex",
    "reasoning_hex",
    "streamed_text_hex",
    "session_invalidation_required",
    "transcript_eos_appended",
)

BOUNDARY_FIELDS = {
    "stop_boundary": ("pos", "len"),
    "tool_boundary": ("saw_start", "saw_end", "tool_call_count"),
}

API_FIELDS = (
    "openai_finish_reason",
    "anthropic_stop_reason",
    "responses_status",
    "responses_item_status",
    "responses_incomplete_reason",
)

STREAM_STEP_FIELDS = (
    "step",
    "text_len",
    "stream_safe_len",
    "delta_hex",
    "held_tail_hex",
    "hit_stop",
    "stop_pos",
    "stop_len",
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


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-decode-policy-dump-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def check_equal(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_cases(expected_cases: list[Any], got_cases: list[Any], report: Report) -> None:
    report.check(len(expected_cases) == len(got_cases), "case count drift")
    expected_names: list[str] = []
    got_names: list[str] = []
    for idx, (expected_raw, got_raw) in enumerate(zip(expected_cases, got_cases)):
        expected = require_dict(report, expected_raw, f"expected.cases[{idx}]")
        got = require_dict(report, got_raw, f"rust.cases[{idx}]")
        name = expected.get("name")
        got_name = got.get("name")
        report.check(isinstance(name, str) and bool(name), f"expected.cases[{idx}].name invalid")
        report.check(isinstance(got_name, str) and bool(got_name), f"rust.cases[{idx}].name invalid")
        if isinstance(name, str):
            expected_names.append(name)
        if isinstance(got_name, str):
            got_names.append(got_name)
        check_equal(report, name, got_name, f"cases[{idx}].name")
        compare_request(report, expected.get("request"), got.get("request"), f"cases[{idx}].request")
        compare_schedule(report, expected.get("schedule"), got.get("schedule"), f"cases[{idx}].schedule")
        compare_result(report, expected.get("result"), got.get("result"), f"cases[{idx}].result")
    report.check(expected_names == got_names, "case order drift")
    report.check(len(got_names) == len(set(got_names)), "duplicate Rust case names")


def compare_request(report: Report, expected_raw: Any, got_raw: Any, path: str) -> None:
    expected = require_dict(report, expected_raw, f"{path}.expected")
    got = require_dict(report, got_raw, f"{path}.rust")
    for key in ("surface", "api", "kind", "stream", "has_tools", "max_tokens", "stops"):
        check_equal(report, expected.get(key), got.get(key), f"{path}.{key}")


def compare_schedule(report: Report, expected_raw: Any, got_raw: Any, path: str) -> None:
    expected = require_list(report, expected_raw, f"{path}.expected")
    got = require_list(report, got_raw, f"{path}.rust")
    report.check(len(expected) == len(got), f"{path}: length drift")
    for idx, (expected_item_raw, got_item_raw) in enumerate(zip(expected, got)):
        expected_item = require_dict(report, expected_item_raw, f"{path}[{idx}].expected")
        got_item = require_dict(report, got_item_raw, f"{path}[{idx}].rust")
        for key in ("index", "eos", "text_hex"):
            check_equal(report, expected_item.get(key), got_item.get(key), f"{path}[{idx}].{key}")


def compare_result(report: Report, expected_raw: Any, got_raw: Any, path: str) -> None:
    expected = require_dict(report, expected_raw, f"{path}.expected")
    got = require_dict(report, got_raw, f"{path}.rust")
    for key in RESULT_FIELDS:
        check_equal(report, expected.get(key), got.get(key), f"{path}.{key}")
    for field, keys in BOUNDARY_FIELDS.items():
        expected_boundary = require_dict(report, expected.get(field), f"{path}.expected.{field}")
        got_boundary = require_dict(report, got.get(field), f"{path}.rust.{field}")
        for key in keys:
            check_equal(report, expected_boundary.get(key), got_boundary.get(key), f"{path}.{field}.{key}")
    expected_api = require_dict(report, expected.get("api_finish"), f"{path}.expected.api_finish")
    got_api = require_dict(report, got.get("api_finish"), f"{path}.rust.api_finish")
    for key in API_FIELDS:
        check_equal(report, expected_api.get(key), got_api.get(key), f"{path}.api_finish.{key}")

    expected_steps = require_list(report, expected.get("stream_steps"), f"{path}.expected.stream_steps")
    got_steps = require_list(report, got.get("stream_steps"), f"{path}.rust.stream_steps")
    report.check(len(expected_steps) == len(got_steps), f"{path}.stream_steps length drift")
    for idx, (expected_step_raw, got_step_raw) in enumerate(zip(expected_steps, got_steps)):
        expected_step = require_dict(report, expected_step_raw, f"{path}.expected.stream_steps[{idx}]")
        got_step = require_dict(report, got_step_raw, f"{path}.rust.stream_steps[{idx}]")
        for key in STREAM_STEP_FIELDS:
            check_equal(report, expected_step.get(key), got_step.get(key), f"{path}.stream_steps[{idx}].{key}")


def compare_dumps(expected_raw: Any, got_raw: Any) -> Report:
    report = Report()
    expected = require_dict(report, expected_raw, "expected")
    got = require_dict(report, got_raw, "rust")
    report.check(expected.get("schema") == "ds4.decode_policy_oracle.v1", "C schema mismatch")
    report.check(got.get("schema") == "ds4.rust_decode_policy_oracle.v1", "Rust schema mismatch")
    check_equal(report, expected.get("model"), got.get("model"), "model")
    expected_cases = require_list(report, expected.get("cases"), "expected.cases")
    got_cases = require_list(report, got.get("cases"), "rust.cases")
    compare_cases(expected_cases, got_cases, report)
    return report


def run_negative_tests(expected: Any, got: Any) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("finish drift", ["cases", 0, "result", "finish_reason"], "length"),
        ("request drift", ["cases", 1, "request", "max_tokens"], 99),
        ("schedule drift", ["cases", 4, "schedule", 1, "text_hex"], "00"),
        ("visible drift", ["cases", 2, "result", "visible_text_hex"], "00"),
        ("held-tail drift", ["cases", 5, "result", "stream_steps", 0, "held_tail_hex"], ""),
        ("invalidation drift", ["cases", 6, "result", "session_invalidation_required"], False),
        ("stop boundary drift", ["cases", 8, "result", "stop_boundary", "pos"], -1),
        ("tool boundary drift", ["cases", 9, "result", "tool_boundary", "tool_call_count"], 0),
        ("API mapping drift", ["cases", 10, "result", "api_finish", "responses_status"], "completed"),
        ("case coverage drift", ["cases"], got["cases"][:-1]),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(got)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = compare_dumps(expected, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--rust-dump", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--write-rust-dump", type=Path)
    args = parser.parse_args()

    expected = load_json(args.baseline)
    if args.rust_dump:
        got = load_json(args.rust_dump)
    else:
        code, stdout, stderr = run_rust_dump()
        if code != 0:
            print("rust decode policy dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        got = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    compare = compare_dumps(expected, got)
    print_report("decode policy C/Rust comparator", compare)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("decode policy C/Rust negative tests", negative)

    return 0 if compare.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
