#!/usr/bin/env python3
"""Compare the Rust KV trailer port against the M7.4b C oracle."""

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
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.4b" / "current-c.json"


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
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-kv-trailer-dump-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


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
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_dumps(expected_raw: Any, got_raw: Any) -> Report:
    report = Report()
    expected = require_dict(report, expected_raw, "expected")
    got = require_dict(report, got_raw, "rust")
    report.check(expected.get("schema") == "ds4.kv_trailer_oracle.v1", "C schema mismatch")
    report.check(got.get("schema") == "ds4.rust_kv_trailer_oracle.v1", "Rust schema mismatch")
    report.check(expected.get("source") == "current-c-server-kv-trailer-no-model", "C source mismatch")
    report.check(got.get("source") == "rust-server-kv-trailer-no-model", "Rust source mismatch")
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
        ("schema drift", ["schema"], "ds4.kv_trailer_oracle.v1"),
        ("trailer byte drift", ["tool_map_cases", 1, "trailer_hex"], "00"),
        ("entry order drift", ["tool_map_cases", 4, "decoded", "entries", 0, "id"], "call_a"),
        ("wanted load drift", ["tool_map_cases", 4, "load_wanted_count"], 2),
        ("extension key-kind drift", ["extension_flag_cases", 4, "key_kind"], "thinking-visible"),
        ("malformed error drift", ["malformed_cases", 8, "decoded", "error"], "truncated-id"),
        ("partial load drift", ["malformed_cases", 9, "load_count"], 0),
        ("case coverage drift", ["tool_map_cases"], got["tool_map_cases"][:-1]),
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
            print("rust KV trailer dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        got = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    compare = compare_dumps(expected, got)
    print_report("KV trailer C/Rust comparator", compare)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("KV trailer C/Rust negative tests", negative)

    return 0 if compare.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
