#!/usr/bin/env python3
"""Compare the Rust KV header/policy port against the M7.2 C oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.2" / "current-c.json"
SCORE_ABS_TOLERANCE = 1e-12


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
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-kv-policy-dump-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def compare_number(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(isinstance(expected, (int, float)), f"{path}: expected number invalid")
    report.check(isinstance(got, (int, float)), f"{path}: Rust number invalid")
    if not isinstance(expected, (int, float)) or not isinstance(got, (int, float)):
        return
    if path.endswith(".score"):
        if math.isinf(expected) or math.isinf(got):
            report.check(math.isinf(expected) and math.isinf(got), f"{path}: infinity drift")
            return
        report.check(
            math.isclose(float(expected), float(got), rel_tol=0.0, abs_tol=SCORE_ABS_TOLERANCE),
            f"{path}: score drift {expected!r} != {got!r}",
        )
        return
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_value(report: Report, expected: Any, got: Any, path: str) -> None:
    if isinstance(expected, dict):
        got_dict = require_dict(report, got, path)
        report.check(list(expected) == list(got_dict), f"{path}: key order or coverage drift")
        for key, expected_value in expected.items():
            if key in got_dict:
                compare_value(report, expected_value, got_dict[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        report.check(isinstance(got, list), f"{path}: expected array")
        got_list = got if isinstance(got, list) else []
        report.check(len(expected) == len(got_list), f"{path}: length drift")
        for idx, (expected_item, got_item) in enumerate(zip(expected, got_list)):
            compare_value(report, expected_item, got_item, f"{path}[{idx}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        report.check(expected == got, f"{path}: {expected!r} != {got!r}")
        return
    if isinstance(expected, (int, float)):
        compare_number(report, expected, got, path)
        return
    report.check(False, f"{path}: unsupported value type {type(expected).__name__}")


def compare_dumps(expected_raw: Any, got_raw: Any) -> Report:
    report = Report()
    expected = require_dict(report, expected_raw, "expected")
    got = require_dict(report, got_raw, "rust")
    report.check(expected.get("schema") == "ds4.kv_policy_oracle.v1", "C schema mismatch")
    report.check(got.get("schema") == "ds4.rust_kv_policy_oracle.v1", "Rust schema mismatch")
    report.check(expected.get("source") == "current-c-kvstore-no-model", "C source mismatch")
    report.check(got.get("source") == "rust-kvstore-no-model", "Rust source mismatch")
    for key, expected_value in expected.items():
        if key in {"schema", "source"}:
            continue
        report.check(key in got, f"rust.{key}: missing section")
        if key in got:
            compare_value(report, expected_value, got[key], key)
    rust_extra = [key for key in got if key not in expected]
    report.check(not rust_extra, f"rust: unexpected sections {rust_extra!r}")
    return report


def run_negative_tests(expected: Any, got: Any) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("schema drift", ["schema"], "ds4.kv_policy_oracle.v1"),
        ("reason drift", ["reason_codes", 2, "code"], -1),
        ("header byte drift", ["header_cases", 0, "header_hex"], "00"),
        ("store length drift", ["policy_cases", "store_len", 2, "store_len"], 4096),
        ("eviction score drift", ["policy_cases", "eviction_score", 0, "score"], 99.0),
        ("prefix selected-token drift", ["policy_cases", "find_text_prefix", 0, "selected_tokens"], 1),
        ("M0.5 row drift", ["m0_5_header_fixture", "expected_rows", 0, "tokens"], 1),
        ("case coverage drift", ["policy_cases", "chat_anchor"], got["policy_cases"]["chat_anchor"][:-1]),
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
            print("rust KV policy dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        got = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    compare = compare_dumps(expected, got)
    print_report("KV policy C/Rust comparator", compare)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("KV policy C/Rust negative tests", negative)

    return 0 if compare.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
