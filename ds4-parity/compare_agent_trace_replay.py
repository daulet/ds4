#!/usr/bin/env python3
"""Compare Rust agent trace replay fixtures against the M11.1 C oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.1"
BASELINE_C = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"
SCHEMA = "ds4.agent_trace_replay_oracle.v1"
MANIFEST_SCHEMA = "ds4.agent_trace_replay_baseline.v1"
MILESTONE = "M11.1"
SOURCE = "current-c-agent-trace-oracle"
REQUIRED_NORMALIZED_FIELDS = {"timestamp", "cwd", "duration_ms", "pid", "session_sha"}


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


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path} must be an object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path} must be an array")
    return obj if isinstance(obj, list) else []


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


def parse_dsml_tool_sequence(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for invoke in re.finditer(
        r"<｜DSML｜invoke name=\"([^\"]+)\">(.*?)</｜DSML｜invoke\s*(?:｜)?\s*>",
        text,
        flags=re.S,
    ):
        args: list[dict[str, Any]] = []
        body = invoke.group(2)
        for param in re.finditer(
            r"<｜DSML｜parameter name=\"([^\"]+)\" string=\"(true|false)\">(.*?)</｜DSML｜parameter\s*(?:｜)?\s*>",
            body,
            flags=re.S,
        ):
            args.append(
                {
                    "name": param.group(1),
                    "value": param.group(3),
                    "is_string": param.group(2) == "true",
                }
            )
        calls.append({"name": invoke.group(1), "args": args})
    return calls


def compare_tool_sequences(report: Report, case: dict[str, Any], path: str) -> None:
    expected = require_list(report, case.get("expected", {}).get("tool_sequence"), f"{path}.expected.tool_sequence")
    events = require_list(report, case.get("model_events"), f"{path}.model_events")
    parsed: list[dict[str, Any]] = []
    for event in events:
        raw_event = require_dict(report, event, f"{path}.model_events[]")
        round_id = raw_event.get("round")
        text = raw_event.get("text")
        report.check(isinstance(round_id, int), f"{path}.model_events[].round invalid")
        report.check(isinstance(text, str), f"{path}.model_events[].text invalid")
        if isinstance(round_id, int) and isinstance(text, str):
            for call in parse_dsml_tool_sequence(text):
                parsed.append({"round": round_id, **call})
    report.check(parsed == expected, f"{path}: parsed DSML tool sequence drift")


def validate_case(report: Report, case: Any, path: str) -> None:
    obj = require_dict(report, case, path)
    case_id = obj.get("id")
    report.check(isinstance(case_id, str) and bool(case_id), f"{path}.id invalid")

    fixture = require_dict(report, obj.get("fixture"), f"{path}.fixture")
    report.check(fixture.get("kind") in {"scripted_model", "session_commands"}, f"{path}.fixture.kind invalid")
    report.check(fixture.get("cwd") == "<WORKSPACE>", f"{path}.fixture.cwd must be normalized")
    report.check(isinstance(fixture.get("ctx_size"), int) and fixture.get("ctx_size") > 0, f"{path}.fixture.ctx_size invalid")
    report.check(fixture.get("think_mode") in {"none", "high", "max"}, f"{path}.fixture.think_mode invalid")

    inputs = require_list(report, obj.get("inputs"), f"{path}.inputs")
    report.check(bool(inputs), f"{path}.inputs must not be empty")
    for idx, raw_input in enumerate(inputs):
        item = require_dict(report, raw_input, f"{path}.inputs[{idx}]")
        report.check(item.get("type") in {"user", "command"}, f"{path}.inputs[{idx}].type invalid")
        report.check(isinstance(item.get("text"), str) and item.get("text"), f"{path}.inputs[{idx}].text invalid")

    expected = require_dict(report, obj.get("expected"), f"{path}.expected")
    roles = require_list(report, expected.get("transcript_roles"), f"{path}.expected.transcript_roles")
    report.check(roles and roles[0] == "system", f"{path}.expected transcript must start with system")
    report.check(all(role in {"system", "user", "assistant", "tool"} for role in roles), f"{path}.expected transcript role invalid")
    report.check(isinstance(expected.get("final_visible_output"), str), f"{path}.expected.final_visible_output invalid")

    tool_stubs = require_list(report, obj.get("tool_stubs"), f"{path}.tool_stubs")
    compare_tool_sequences(report, obj, path)

    if case_id == "single_tool_round":
        report.check(len(tool_stubs) == 1, f"{path}: single tool fixture should have one stub")
        stub = require_dict(report, tool_stubs[0] if tool_stubs else None, f"{path}.tool_stubs[0]")
        report.check(stub.get("name") == "list", f"{path}.tool_stubs[0].name drift")
        report.check(stub.get("args") == expected.get("tool_sequence", [{}])[0].get("args"), f"{path}.tool_stubs[0].args drift")
        report.check(roles == ["system", "user", "assistant", "tool", "assistant"], f"{path}.transcript_roles drift")
    elif case_id == "session_switching_commands":
        report.check(not tool_stubs, f"{path}: session command fixture should not have tool stubs")
        ops = require_list(report, expected.get("session_operations"), f"{path}.expected.session_operations")
        commands = [require_dict(report, op, f"{path}.expected.session_operations[]").get("command") for op in ops]
        report.check(commands == ["save", "list", "switch", "history", "new"], f"{path}.session command order drift")
        report.check(any(op.get("session") == "<SESSION:alpha>" for op in ops if isinstance(op, dict)), f"{path}.session id not normalized")


def validate_root(obj: dict[str, Any]) -> Report:
    report = Report()
    report.check(obj.get("schema") == SCHEMA, "schema mismatch")
    report.check(obj.get("milestone") == MILESTONE, "milestone mismatch")
    report.check(obj.get("source") == SOURCE, "source mismatch")

    normalization = require_dict(report, obj.get("normalization"), "normalization")
    report.check(normalization.get("path_root") == "<WORKSPACE>", "normalization.path_root drift")
    fields = set(require_list(report, normalization.get("fields"), "normalization.fields"))
    report.check(REQUIRED_NORMALIZED_FIELDS <= fields, "normalization fields missing")
    rules = require_list(report, normalization.get("rules"), "normalization.rules")
    report.check(len(rules) >= 3, "normalization rules incomplete")

    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    for forbidden in ("/Users/", "/workspace/ds4", "duration_ms\":", "pid\":", "timestamp\":"):
        report.check(forbidden not in raw, f"un-normalized field/path present: {forbidden}")
    report.check("<WORKSPACE>" in raw, "workspace marker missing")
    report.check("<SESSION:alpha>" in raw, "session marker missing")

    cases = require_list(report, obj.get("cases"), "cases")
    ids: set[str] = set()
    for idx, case in enumerate(cases):
        case_obj = require_dict(report, case, f"cases[{idx}]")
        case_id = case_obj.get("id")
        if isinstance(case_id, str):
            report.check(case_id not in ids, f"duplicate case id {case_id}")
            ids.add(case_id)
        validate_case(report, case_obj, f"cases[{idx}]")
    report.check(ids == {"single_tool_round", "session_switching_commands"}, "case coverage drift")
    return report


def check_manifest(manifest_path: Path, baseline_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(manifest_path), "manifest")
    report.check(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    report.check(manifest.get("milestone") == MILESTONE, "manifest milestone mismatch")
    current = require_dict(report, manifest.get("dumps", {}).get("current_c"), "manifest.dumps.current_c")
    report.check(current.get("path") == baseline_path.name, "manifest current-c path mismatch")
    report.check(current.get("size_bytes") == baseline_path.stat().st_size, "manifest current-c size drift")
    report.check(current.get("sha256") == sha256_file(baseline_path), "manifest current-c sha256 drift")
    commands = require_list(report, manifest.get("validation"), "manifest.validation")
    report.check(any("compare_agent_trace_replay.py" in cmd for cmd in commands if isinstance(cmd, str)), "manifest missing comparator command")
    return report


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-agent-trace-replay-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_negative_tests(original: dict[str, Any]) -> Report:
    report = Report()

    def expect_failure(label: str, mutate) -> None:
        bad = copy.deepcopy(original)
        mutate(bad)
        result = validate_root(bad)
        report.check(not result.ok, f"negative test failed to catch {label}")

    expect_failure("schema drift", lambda obj: obj.__setitem__("schema", "wrong"))
    expect_failure("normalization field drift", lambda obj: obj["normalization"]["fields"].remove("pid"))
    expect_failure("raw path drift", lambda obj: obj["cases"][0]["fixture"].__setitem__("cwd", "/workspace/ds4"))
    expect_failure("tool name drift", lambda obj: obj["cases"][0]["model_events"][0].__setitem__("text", obj["cases"][0]["model_events"][0]["text"].replace("name=\"list\"", "name=\"read\"")))
    expect_failure("transcript role drift", lambda obj: obj["cases"][0]["expected"]["transcript_roles"].__setitem__(3, "assistant"))
    expect_failure("session command order drift", lambda obj: obj["cases"][1]["expected"]["session_operations"].reverse())

    bad = copy.deepcopy(original)
    bad["cases"] = bad["cases"][:1]
    result = validate_root(bad)
    report.check(not result.ok, "negative test failed to catch missing case")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "current-c.json"
        baseline.write_text(json.dumps(original, ensure_ascii=False))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "milestone": MILESTONE,
            "dumps": {
                "current_c": {
                    "path": "current-c.json",
                    "size_bytes": baseline.stat().st_size,
                    "sha256": "0" * 64,
                }
            },
            "validation": ["python3 ds4-parity/compare_agent_trace_replay.py --negative-test"],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        manifest_report = check_manifest(manifest_path, baseline)
        report.check(not manifest_report.ok, "negative test failed to catch manifest sha drift")

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
    merge("baseline", report, validate_root(baseline))
    if args.manifest:
        merge("manifest", report, check_manifest(args.manifest, args.baseline))

    rc, stdout, stderr = run_rust_dump()
    report.check(rc == 0, f"Rust agent trace replay dump failed rc={rc} stderr={stderr.strip()}")
    if rc == 0:
        try:
            rust = require_dict(report, json.loads(stdout), "rust")
            merge("rust schema", report, validate_root(rust))
            diff = first_diff(baseline, rust)
            report.check(diff is None, f"Rust agent trace replay drift: {diff}")
        except json.JSONDecodeError as exc:
            report.check(False, f"Rust agent trace replay dump did not emit JSON: {exc}")

    if args.negative_test:
        merge("negative", report, run_negative_tests(baseline))

    if report.ok:
        print(f"agent trace replay comparison: PASS, {report.checks} checks")
        return 0
    print(f"agent trace replay comparison: FAIL, {report.checks} checks", file=sys.stderr)
    for err in report.errors:
        print(f" - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
