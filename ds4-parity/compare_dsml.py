#!/usr/bin/env python3
"""Compare Rust DSML formatting/parsing output against the M5.6a C oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "dsml" / "m5.6a"
BASELINE_C = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"

FORMAT_FIELDS = ("name", "kind", "rendered")
PARSE_FIELDS = (
    "name",
    "require_thinking_closed",
    "input",
    "parse_ok",
    "content",
    "reasoning",
    "raw_dsml",
    "calls",
    "response_parse_ok",
    "response_recovered",
    "response_finish",
    "response_error",
    "response_content",
    "response_reasoning",
    "response_raw_dsml",
    "response_calls",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-dsml-dump"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path} must be an object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path} must be an array")
    return obj if isinstance(obj, list) else []


def check_dump_schema(obj: dict[str, Any]) -> Report:
    report = Report()
    report.check(obj.get("schema") == "ds4.dsml_oracle.v1", "schema mismatch")
    format_cases = require_list(report, obj.get("format_cases"), "format_cases")
    parse_cases = require_list(report, obj.get("parse_cases"), "parse_cases")
    report.check(len(format_cases) >= 5, "expected at least five format cases")
    report.check(len(parse_cases) >= 10, "expected at least ten parse cases")

    format_names: set[str] = set()
    for idx, raw_case in enumerate(format_cases):
        case = require_dict(report, raw_case, f"format_cases[{idx}]")
        name = case.get("name")
        report.check(isinstance(name, str) and bool(name), f"format_cases[{idx}].name invalid")
        if isinstance(name, str):
            report.check(name not in format_names, f"duplicate format case {name}")
            format_names.add(name)
        for field_name in FORMAT_FIELDS:
            report.check(field_name in case, f"format case {name} missing {field_name}")
        report.check(case.get("kind") in {"tool_calls", "tool_result"}, f"format case {name} kind invalid")
        report.check(isinstance(case.get("rendered"), str), f"format case {name} rendered must be string")
        if case.get("kind") == "tool_result":
            report.check(isinstance(case.get("input"), str), f"tool result case {name} missing input")

    parse_names: set[str] = set()
    for idx, raw_case in enumerate(parse_cases):
        case = require_dict(report, raw_case, f"parse_cases[{idx}]")
        name = case.get("name")
        report.check(isinstance(name, str) and bool(name), f"parse_cases[{idx}].name invalid")
        if isinstance(name, str):
            report.check(name not in parse_names, f"duplicate parse case {name}")
            parse_names.add(name)
        for field_name in PARSE_FIELDS:
            report.check(field_name in case, f"parse case {name} missing {field_name}")
        report.check(isinstance(case.get("require_thinking_closed"), bool), f"{name}.require_thinking_closed invalid")
        report.check(isinstance(case.get("input"), str), f"{name}.input invalid")
        report.check(isinstance(case.get("parse_ok"), bool), f"{name}.parse_ok invalid")
        report.check(isinstance(case.get("response_parse_ok"), bool), f"{name}.response_parse_ok invalid")
        report.check(isinstance(case.get("response_recovered"), bool), f"{name}.response_recovered invalid")
        report.check(isinstance(case.get("calls"), list), f"{name}.calls invalid")
        report.check(isinstance(case.get("response_calls"), list), f"{name}.response_calls invalid")
    return report


def first_diff(expected: Any, got: Any, path: str = "root") -> str | None:
    if type(expected) is not type(got):
        return f"{path}: type {type(expected).__name__} != {type(got).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        got_keys = set(got)
        if expected_keys != got_keys:
            return f"{path}: keys {sorted(expected_keys)} != {sorted(got_keys)}"
        for key in sorted(expected):
            diff = first_diff(expected[key], got[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(got):
            return f"{path}: len {len(expected)} != {len(got)}"
        for idx, (expected_item, got_item) in enumerate(zip(expected, got)):
            diff = first_diff(expected_item, got_item, f"{path}[{idx}]")
            if diff:
                return diff
        return None
    if expected != got:
        return f"{path}: {expected!r} != {got!r}"
    return None


def compare_dumps(expected: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    expected_schema = check_dump_schema(expected)
    got_schema = check_dump_schema(got)
    report.check(expected_schema.ok, "C DSML oracle schema invalid: " + "; ".join(expected_schema.errors[:3]))
    report.check(got_schema.ok, "Rust DSML dump schema invalid: " + "; ".join(got_schema.errors[:3]))

    diff = first_diff(expected, got)
    report.check(diff is None, f"Rust DSML dump drift: {diff}")
    return report


def check_manifest(manifest_path: Path, baseline_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(manifest_path), "manifest")
    report.check(manifest.get("schema") == "ds4.dsml_baseline.v1", "manifest schema mismatch")
    report.check(manifest.get("milestone") == "M5.6a", "manifest milestone mismatch")
    current = require_dict(report, manifest.get("dumps", {}).get("current_c"), "manifest.dumps.current_c")
    report.check(current.get("path") == baseline_path.name, "manifest current-c path mismatch")
    report.check(current.get("size_bytes") == baseline_path.stat().st_size, "manifest current-c size drift")
    report.check(current.get("sha256") == sha256_file(baseline_path), "manifest current-c sha256 drift")
    return report


def run_negative_tests(original: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("format rendered drift", ["format_cases", 0, "rendered"], "changed"),
        ("parse finish drift", ["parse_cases", 0, "response_finish"], "stop"),
        ("raw DSML drift", ["parse_cases", 0, "raw_dsml"], "changed"),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(original)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = compare_dumps(original, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")

    bad = copy.deepcopy(original)
    bad["parse_cases"] = bad["parse_cases"][1:]
    result = compare_dumps(original, bad)
    report.check(not result.ok, "negative test failed to catch missing parse case")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "current-c.json"
        baseline.write_text(json.dumps(original, ensure_ascii=False))
        manifest = {
            "schema": "ds4.dsml_baseline.v1",
            "milestone": "M5.6a",
            "dumps": {
                "current_c": {
                    "path": "current-c.json",
                    "size_bytes": baseline.stat().st_size + 1,
                    "sha256": sha256_file(baseline),
                }
            },
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        manifest_report = check_manifest(manifest_path, baseline)
        report.check(not manifest_report.ok, "negative test failed to catch manifest size drift")

    return report


def merge(prefix: str, dst: Report, src: Report) -> None:
    dst.checks += src.checks
    dst.errors.extend(f"{prefix}: {err}" for err in src.errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", nargs="?", type=Path, default=BASELINE_C)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    baseline = require_dict(report, load_json(args.baseline), "baseline")
    merge("baseline schema", report, check_dump_schema(baseline))
    if args.manifest:
        merge("manifest", report, check_manifest(args.manifest, args.baseline))

    rc, stdout, stderr = run_rust_dump()
    report.check(rc == 0, f"Rust DSML dump failed rc={rc} stderr={stderr.strip()}")
    if rc == 0:
        try:
            rust = require_dict(report, json.loads(stdout), "rust")
            merge("compare", report, compare_dumps(baseline, rust))
        except json.JSONDecodeError as exc:
            report.check(False, f"Rust DSML dump did not emit JSON: {exc}")

    if args.negative_test:
        merge("negative", report, run_negative_tests(baseline))

    if report.ok:
        print(f"DSML comparison: PASS, {report.checks} checks")
        return 0
    print(f"DSML comparison: FAIL, {report.checks} checks", file=sys.stderr)
    for err in report.errors:
        print(f" - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
