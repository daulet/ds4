#!/usr/bin/env python3
"""Compare the Rust DSV4 session payload reader against the M7.5 C oracle."""

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
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.5" / "current-c.json"


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
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-session-payload-dump-rs"],
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


def compare_structural(expected_raw: Any, got_raw: Any) -> Report:
    report = Report()
    expected = require_dict(report, expected_raw, "expected")
    got = require_dict(report, got_raw, "rust")
    report.check(
        expected.get("schema") == "ds4.session_payload_shape_structural.v1",
        "C structural schema mismatch",
    )
    report.check(
        got.get("schema") == "ds4.rust_session_payload_shape_structural.v1",
        "Rust structural schema mismatch",
    )
    report.check(
        expected.get("source") == "current-c-session-payload-no-model",
        "C structural source mismatch",
    )
    report.check(
        got.get("source") == "rust-session-payload-no-model",
        "Rust structural source mismatch",
    )
    for key, expected_value in expected.items():
        if key in {"schema", "source"}:
            continue
        report.check(key in got, f"rust.{key}: missing section")
        if key in got:
            compare_value(report, expected_value, got[key], key)
    rust_extra = [key for key in got if key not in expected]
    report.check(not rust_extra, f"rust: unexpected sections {rust_extra!r}")
    return report


def check_m0_5_fixture_contract(baseline: dict[str, Any]) -> Report:
    report = Report()
    records = baseline.get("m0_5_payload_records", [])
    report.check(
        isinstance(records, list) and len(records) == 3,
        "M0.5 payload record coverage drift",
    )
    for record in records if isinstance(records, list) else []:
        path = f"m0_5_payload_records[{record.get('file', '?')}]"
        report.check(
            record.get("raw_kv_committed") is False,
            f"{path}: raw KV should remain hash-only",
        )
        report.check(
            record.get("payload_bytes", 0) > 1_000_000,
            f"{path}: expected hash-only payload size",
        )
        report.check(
            record.get("size_matches_payload") is True,
            f"{path}: KVC size formula drift",
        )
        expected = (
            48
            + 4
            + record.get("rendered_text_bytes", 0)
            + record.get("payload_bytes", 0)
            + record.get("trailer_bytes", 0)
        )
        report.check(expected == record.get("size_bytes"), f"{path}: size_bytes drift")
    refresh = require_dict(report, baseline.get("b300_refresh"), "b300_refresh")
    commands = refresh.get("commands", [])
    all_commands = "\n".join(
        item.get("command", "") for item in commands if isinstance(item, dict)
    )
    report.check(
        "/tmp/ds4-hou2-prod1.kubeconfig" in all_commands,
        "B300 commands missing temp kubeconfig",
    )
    report.check(
        "--context hou2-prod1" in all_commands,
        "B300 commands missing explicit context",
    )
    report.check(
        "/workspace/ds4/ds4flash.gguf" in all_commands,
        "B300 commands missing model path",
    )
    return report


def run_negative_tests(baseline: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    structural = baseline["structural"]
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("magic byte drift", ["constants", "magic_bytes_hex"], "00000000"),
        ("layout drift", ["fixed_model_layout", "n_vocab"], 1),
        ("ratio drift", ["compress_ratio_by_layer", 2], 128),
        ("header rejection drift", ["header_rejection_cases", 1, "code"], "ok"),
        ("body rejection drift", ["body_probe_cases", 1, "code"], "ok"),
        (
            "payload size drift",
            ["size_case", "payload_bytes"],
            structural["size_case"]["payload_bytes"] + 4,
        ),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(got)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = compare_structural(structural, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")

    bad_baseline = copy.deepcopy(baseline)
    bad_baseline["m0_5_payload_records"][0]["payload_bytes"] += 1
    result = check_m0_5_fixture_contract(bad_baseline)
    report.check(not result.ok, "negative test failed to catch M0.5 size drift")

    bad_baseline = copy.deepcopy(baseline)
    for command in bad_baseline["b300_refresh"]["commands"]:
        command["command"] = "kubectl get pods"
    result = check_m0_5_fixture_contract(bad_baseline)
    report.check(not result.ok, "negative test failed to catch B300 command drift")
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
    parser.add_argument("--write-rust-dump", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    if args.rust_dump:
        got = load_json(args.rust_dump)
    else:
        code, stdout, stderr = run_rust_dump()
        if code != 0:
            print("rust session payload dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        got = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    structural = compare_structural(baseline.get("structural"), got)
    print_report("Session payload C/Rust structural comparator", structural)
    fixture = check_m0_5_fixture_contract(baseline)
    print_report("Session payload M0.5 fixture contract", fixture)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(baseline, got)
        print_report("Session payload negative tests", negative)

    return 0 if structural.ok and fixture.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
