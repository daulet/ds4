#!/usr/bin/env python3
"""Validate the M11.2 Rust agent rendered-context replay summary."""

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
M11_1_BASELINE = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.1" / "current-c.json"
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.2"
BASELINE = BASELINE_DIR / "rendered-context.json"
MANIFEST = BASELINE_DIR / "manifest.json"
SCHEMA = "ds4.agent_rendered_context_replay.v1"
MANIFEST_SCHEMA = "ds4.agent_rendered_context_baseline.v1"
MILESTONE = "M11.2"
SOURCE = "rust-agent-rendered-context-replay"
SYSTEM_PROMPT = "You are a helpful coding assistant running inside ds4-agent."


EXPECTED_MARKERS = {
    "single_tool_round": {
        "begin_sentence": 1,
        "user": 2,
        "assistant": 2,
        "end_sentence": 2,
        "tool_result": 1,
        "dsml_tool_calls": 1,
    },
    "session_switching_commands": {
        "begin_sentence": 1,
        "user": 1,
        "assistant": 1,
        "end_sentence": 1,
        "tool_result": 0,
        "dsml_tool_calls": 0,
    },
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


def m11_1_cases(m11_1: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("id")): case
        for case in m11_1.get("cases", [])
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    }


def final_output_for_context(case: dict[str, Any]) -> str:
    events = [event for event in case.get("model_events", []) if isinstance(event, dict)]
    if not events:
        return ""
    text = events[-1].get("text")
    return text if isinstance(text, str) else ""


def validate_case(report: Report, case: Any, oracle_case: dict[str, Any], path: str) -> None:
    obj = require_dict(report, case, path)
    case_id = obj.get("id")
    report.check(case_id in EXPECTED_MARKERS, f"{path}.id unexpected")
    report.check(obj.get("replay_source") == "M11.1", f"{path}.replay_source drift")
    report.check(obj.get("think_mode") == "none", f"{path}.think_mode drift")

    expected_roles = oracle_case.get("expected", {}).get("transcript_roles")
    roles = require_list(report, obj.get("message_roles"), f"{path}.message_roles")
    report.check(roles == expected_roles, f"{path}.message_roles drift from M11.1 fixture")

    markers = require_dict(report, obj.get("markers"), f"{path}.markers")
    expected_markers = EXPECTED_MARKERS.get(str(case_id), {})
    for key, expected in expected_markers.items():
        report.check(markers.get(key) == expected, f"{path}.markers.{key} drift")

    prompt = obj.get("prompt_text")
    report.check(isinstance(prompt, str) and bool(prompt), f"{path}.prompt_text invalid")
    if not isinstance(prompt, str):
        prompt = ""

    for marker in ("<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"):
        report.check(marker in prompt, f"{path}.prompt_text missing {marker}")
    report.check(SYSTEM_PROMPT in prompt, f"{path}.prompt_text missing agent system prompt")

    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    for forbidden in ("/Users/", "/workspace/ds4", "duration_ms", "pid", "timestamp"):
        report.check(forbidden not in raw, f"{path} contains unnormalized field/path {forbidden}")

    expected_final = final_output_for_context(oracle_case)
    report.check(obj.get("final_visible_output") == expected_final, f"{path}.final_visible_output drift")
    report.check(obj.get("contains_final_visible_output") is True, f"{path}.contains_final_visible_output drift")
    report.check(expected_final in prompt, f"{path}.prompt_text missing final visible output")

    if case_id == "single_tool_round":
        tool_dsml = oracle_case["model_events"][0]["text"]
        tool_output = oracle_case["tool_stubs"][0]["output"]
        report.check(obj.get("raw_tool_dsml_preserved") is True, f"{path}.raw_tool_dsml_preserved drift")
        report.check(tool_dsml in prompt, f"{path}.prompt_text missing raw DSML block")
        report.check("<tool_result>" in prompt and "</tool_result>" in prompt, f"{path}.tool result tags missing")
        report.check(tool_output in prompt, f"{path}.prompt_text missing tool stub output")
    elif case_id == "session_switching_commands":
        report.check(obj.get("raw_tool_dsml_preserved") is False, f"{path}.raw_tool_dsml_preserved drift")
        for command in ("/save", "/list", "/switch", "/history", "/new"):
            report.check(command not in prompt, f"{path}.prompt_text leaked session command {command}")
        report.check("<tool_result>" not in prompt, f"{path}.prompt_text has unexpected tool result")
        report.check("<｜DSML｜tool_calls>" not in prompt, f"{path}.prompt_text has unexpected DSML")


def validate_root(obj: dict[str, Any], m11_1: dict[str, Any]) -> Report:
    report = Report()
    report.check(obj.get("schema") == SCHEMA, "schema mismatch")
    report.check(obj.get("milestone") == MILESTONE, "milestone mismatch")
    report.check(obj.get("source") == SOURCE, "source mismatch")
    report.check("M11.1 current-C trace replay fixture" in str(obj.get("oracle", "")), "oracle description drift")

    cases = require_list(report, obj.get("cases"), "cases")
    by_id = m11_1_cases(m11_1)
    ids: set[str] = set()
    for idx, raw_case in enumerate(cases):
        case = require_dict(report, raw_case, f"cases[{idx}]")
        case_id = case.get("id")
        if isinstance(case_id, str):
            report.check(case_id not in ids, f"duplicate case {case_id}")
            ids.add(case_id)
            validate_case(report, case, by_id.get(case_id, {}), f"cases[{idx}]")
    report.check(ids == set(EXPECTED_MARKERS), "case coverage drift")
    return report


def check_manifest(manifest_path: Path, baseline_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(manifest_path), "manifest")
    report.check(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    report.check(manifest.get("milestone") == MILESTONE, "manifest milestone mismatch")
    current = require_dict(report, manifest.get("dumps", {}).get("rendered_context"), "manifest.dumps.rendered_context")
    report.check(current.get("path") == baseline_path.name, "manifest rendered context path mismatch")
    report.check(current.get("size_bytes") == baseline_path.stat().st_size, "manifest rendered context size drift")
    report.check(current.get("sha256") == sha256_file(baseline_path), "manifest rendered context sha256 drift")
    oracle = require_dict(report, manifest.get("oracles", {}).get("m11_1_current_c"), "manifest.oracles.m11_1_current_c")
    report.check(oracle.get("path") == "../m11.1/current-c.json", "manifest M11.1 oracle path drift")
    oracle_path = (manifest_path.parent / str(oracle.get("path", ""))).resolve()
    report.check(oracle_path.exists(), "manifest M11.1 oracle path missing")
    if oracle_path.exists():
        report.check(oracle.get("sha256") == sha256_file(oracle_path), "manifest M11.1 oracle sha256 drift")
    commands = require_list(report, manifest.get("validation"), "manifest.validation")
    report.check(any("compare_agent_rendered_context.py" in cmd for cmd in commands if isinstance(cmd, str)), "manifest missing comparator command")
    return report


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-agent-rendered-context-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_negative_tests(original: dict[str, Any], m11_1: dict[str, Any]) -> Report:
    report = Report()

    def expect_failure(label: str, mutate) -> None:
        bad = copy.deepcopy(original)
        mutate(bad)
        result = validate_root(bad, m11_1)
        report.check(not result.ok, f"negative test failed to catch {label}")

    expect_failure("schema drift", lambda obj: obj.__setitem__("schema", "wrong"))
    expect_failure("marker drift", lambda obj: obj["cases"][0]["markers"].__setitem__("tool_result", 0))
    expect_failure("role drift", lambda obj: obj["cases"][0]["message_roles"].__setitem__(3, "assistant"))
    expect_failure("final output drift", lambda obj: obj["cases"][0].__setitem__("final_visible_output", "wrong"))
    expect_failure("raw DSML removal", lambda obj: obj["cases"][0].__setitem__("prompt_text", obj["cases"][0]["prompt_text"].replace("<｜DSML｜tool_calls>", "")))
    expect_failure("session command leak", lambda obj: obj["cases"][1].__setitem__("prompt_text", obj["cases"][1]["prompt_text"] + "/save"))
    expect_failure("raw path leak", lambda obj: obj["cases"][1].__setitem__("prompt_text", obj["cases"][1]["prompt_text"] + "/workspace/ds4"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "rendered-context.json"
        baseline.write_text(json.dumps(original, ensure_ascii=False))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "milestone": MILESTONE,
            "dumps": {
                "rendered_context": {
                    "path": "rendered-context.json",
                    "size_bytes": baseline.stat().st_size,
                    "sha256": "0" * 64,
                }
            },
            "oracles": {"m11_1_current_c": {"path": "../m11.1/current-c.json"}},
            "validation": ["python3 ds4-parity/compare_agent_rendered_context.py --negative-test"],
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
    parser.add_argument("baseline", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--m11-1-baseline", type=Path, default=M11_1_BASELINE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    m11_1 = require_dict(report, load_json(args.m11_1_baseline), "m11_1")
    baseline = require_dict(report, load_json(args.baseline), "baseline")
    merge("baseline", report, validate_root(baseline, m11_1))
    if args.manifest:
        merge("manifest", report, check_manifest(args.manifest, args.baseline))

    rc, stdout, stderr = run_rust_dump()
    report.check(rc == 0, f"Rust rendered-context dump failed rc={rc} stderr={stderr.strip()}")
    if rc == 0:
        try:
            rust = require_dict(report, json.loads(stdout), "rust")
            merge("rust schema", report, validate_root(rust, m11_1))
            diff = first_diff(baseline, rust)
            report.check(diff is None, f"Rust rendered-context replay drift: {diff}")
        except json.JSONDecodeError as exc:
            report.check(False, f"Rust rendered-context dump did not emit JSON: {exc}")

    if args.negative_test:
        merge("negative", report, run_negative_tests(baseline, m11_1))

    if report.ok:
        print(f"agent rendered-context comparison: PASS, {report.checks} checks")
        return 0
    print(f"agent rendered-context comparison: FAIL, {report.checks} checks", file=sys.stderr)
    for err in report.errors:
        print(f" - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
