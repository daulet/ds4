#!/usr/bin/env python3
"""Compare Rust agent DSML streaming parser output against the M5.6b C oracle."""

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
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "dsml" / "m5.6b"
BASELINE_C = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"

SNAPSHOT_FIELDS = {
    "state",
    "search_len",
    "search_tail_hex",
    "raw_len",
    "raw_hex",
    "parse_pos",
    "param_name",
    "param_is_string",
    "param_value_start",
    "current",
    "calls",
    "error",
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
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-agent-dsml-dump"],
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


def check_call(report: Report, obj: Any, path: str) -> None:
    call = require_dict(report, obj, path)
    report.check(call.get("name") is None or isinstance(call.get("name"), str), f"{path}.name invalid")
    args = require_list(report, call.get("args"), f"{path}.args")
    for idx, raw_arg in enumerate(args):
        arg = require_dict(report, raw_arg, f"{path}.args[{idx}]")
        report.check(isinstance(arg.get("name"), str), f"{path}.args[{idx}].name invalid")
        report.check(isinstance(arg.get("value"), str), f"{path}.args[{idx}].value invalid")
        report.check(isinstance(arg.get("is_string"), bool), f"{path}.args[{idx}].is_string invalid")


def check_snapshot(report: Report, obj: Any, path: str, step: bool) -> None:
    snap = require_dict(report, obj, path)
    expected = set(SNAPSHOT_FIELDS)
    if step:
        expected |= {"chunk_index", "offset", "len"}
    report.check(set(snap) == expected, f"{path} fields drift")
    report.check(snap.get("state") in {"search", "structural", "param_value", "done", "error"}, f"{path}.state invalid")
    for field_name in ("search_len", "raw_len", "parse_pos", "param_value_start"):
        report.check(isinstance(snap.get(field_name), int), f"{path}.{field_name} invalid")
    for field_name in ("search_tail_hex", "raw_hex", "error"):
        report.check(isinstance(snap.get(field_name), str), f"{path}.{field_name} invalid")
    report.check(snap.get("param_name") is None or isinstance(snap.get("param_name"), str), f"{path}.param_name invalid")
    report.check(isinstance(snap.get("param_is_string"), bool), f"{path}.param_is_string invalid")
    check_call(report, snap.get("current"), f"{path}.current")
    calls = require_list(report, snap.get("calls"), f"{path}.calls")
    for idx, call in enumerate(calls):
        check_call(report, call, f"{path}.calls[{idx}]")
    if step:
        for field_name in ("chunk_index", "offset", "len"):
            report.check(isinstance(snap.get(field_name), int), f"{path}.{field_name} invalid")


def check_dump_schema(obj: dict[str, Any]) -> Report:
    report = Report()
    report.check(obj.get("schema") == "ds4.agent_dsml_oracle.v1", "schema mismatch")
    cases = require_list(report, obj.get("cases"), "cases")
    report.check(len(cases) >= 13, "expected at least thirteen cases")
    names: set[str] = set()
    for case_idx, raw_case in enumerate(cases):
        case = require_dict(report, raw_case, f"cases[{case_idx}]")
        name = case.get("name")
        report.check(isinstance(name, str) and bool(name), f"cases[{case_idx}].name invalid")
        if isinstance(name, str):
            report.check(name not in names, f"duplicate case {name}")
            names.add(name)
        report.check(isinstance(case.get("input"), str), f"case {name}.input invalid")
        schedules = require_list(report, case.get("schedules"), f"case {name}.schedules")
        report.check(len(schedules) >= 2, f"case {name} expected at least whole and one-byte schedules")
        schedule_names: set[str] = set()
        for sched_idx, raw_schedule in enumerate(schedules):
            schedule = require_dict(report, raw_schedule, f"case {name}.schedules[{sched_idx}]")
            sched_name = schedule.get("name")
            report.check(isinstance(sched_name, str) and bool(sched_name), f"case {name} schedule name invalid")
            if isinstance(sched_name, str):
                report.check(sched_name not in schedule_names, f"case {name} duplicate schedule {sched_name}")
                schedule_names.add(sched_name)
            steps = require_list(report, schedule.get("steps"), f"case {name}.{sched_name}.steps")
            report.check(len(steps) >= 1, f"case {name}.{sched_name} has no steps")
            for step_idx, step in enumerate(steps):
                check_snapshot(report, step, f"case {name}.{sched_name}.steps[{step_idx}]", True)
            check_snapshot(report, schedule.get("final"), f"case {name}.{sched_name}.final", False)
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
    report.check(expected_schema.ok, "C agent DSML oracle schema invalid: " + "; ".join(expected_schema.errors[:3]))
    report.check(got_schema.ok, "Rust agent DSML dump schema invalid: " + "; ".join(got_schema.errors[:3]))
    diff = first_diff(expected, got)
    report.check(diff is None, f"Rust agent DSML dump drift: {diff}")
    return report


def check_manifest(manifest_path: Path, baseline_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(manifest_path), "manifest")
    report.check(manifest.get("schema") == "ds4.agent_dsml_baseline.v1", "manifest schema mismatch")
    report.check(manifest.get("milestone") == "M5.6b", "manifest milestone mismatch")
    current = require_dict(report, manifest.get("dumps", {}).get("current_c"), "manifest.dumps.current_c")
    report.check(current.get("path") == baseline_path.name, "manifest current-c path mismatch")
    report.check(current.get("size_bytes") == baseline_path.stat().st_size, "manifest current-c size drift")
    report.check(current.get("sha256") == sha256_file(baseline_path), "manifest current-c sha256 drift")
    return report


def run_negative_tests(original: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("final state drift", ["cases", 0, "schedules", 0, "final", "state"], "error"),
        ("raw buffer drift", ["cases", 0, "schedules", 0, "final", "raw_hex"], "00"),
        ("completed call drift", ["cases", 0, "schedules", 0, "final", "calls", 0, "name"], "other"),
        ("step offset drift", ["cases", 0, "schedules", 1, "steps", 0, "offset"], 99),
        ("error drift", ["cases", 4, "schedules", 0, "final", "error"], "other error"),
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
    bad["cases"] = bad["cases"][1:]
    result = compare_dumps(original, bad)
    report.check(not result.ok, "negative test failed to catch missing case")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "current-c.json"
        baseline.write_text(json.dumps(original, ensure_ascii=False))
        manifest = {
            "schema": "ds4.agent_dsml_baseline.v1",
            "milestone": "M5.6b",
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
    report.check(rc == 0, f"Rust agent DSML dump failed rc={rc} stderr={stderr.strip()}")
    if rc == 0:
        try:
            rust = require_dict(report, json.loads(stdout), "rust")
            merge("compare", report, compare_dumps(baseline, rust))
        except json.JSONDecodeError as exc:
            report.check(False, f"Rust agent DSML dump did not emit JSON: {exc}")

    if args.negative_test:
        merge("negative", report, run_negative_tests(baseline))

    if report.ok:
        print(f"agent DSML comparison: PASS, {report.checks} checks")
        return 0
    print(f"agent DSML comparison: FAIL, {report.checks} checks", file=sys.stderr)
    for err in report.errors:
        print(f" - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
