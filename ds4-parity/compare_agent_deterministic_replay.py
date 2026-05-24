#!/usr/bin/env python3
"""Validate the M11.3 deterministic agent tool/session replay summary."""

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
M11_2_BASELINE = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.2" / "rendered-context.json"
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.3"
BASELINE = BASELINE_DIR / "deterministic-replay.json"
MANIFEST = BASELINE_DIR / "manifest.json"
SCHEMA = "ds4.agent_deterministic_replay.v1"
MANIFEST_SCHEMA = "ds4.agent_deterministic_replay_baseline.v1"
MILESTONE = "M11.3"
SOURCE = "rust-agent-deterministic-replay"
EXPECTED_CASES = {"single_tool_round", "session_switching_commands"}


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


def index_cases(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("id")): case
        for case in root.get("cases", [])
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    }


def command_inputs(oracle_case: dict[str, Any]) -> list[str]:
    inputs = oracle_case.get("inputs", [])
    return [
        item["text"]
        for item in inputs
        if isinstance(item, dict) and item.get("type") == "command" and isinstance(item.get("text"), str)
    ]


def stripped_operation(op: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in op.items() if key not in {"step", "input"}}


def validate_single_tool_case(
    report: Report,
    case: dict[str, Any],
    oracle_case: dict[str, Any],
    rendered_case: dict[str, Any],
    path: str,
) -> None:
    expected = require_dict(report, oracle_case.get("expected"), f"{path}.oracle.expected")
    tool_replay = require_dict(report, case.get("tool_replay"), f"{path}.tool_replay")
    expected_sequence = expected.get("tool_sequence")
    report.check(tool_replay.get("tool_sequence") == expected_sequence, f"{path}.tool_sequence drift")

    stubs = require_list(report, tool_replay.get("stubs"), f"{path}.tool_replay.stubs")
    oracle_stubs = require_list(report, oracle_case.get("tool_stubs"), f"{path}.oracle.tool_stubs")
    report.check(len(stubs) == 1, f"{path}.tool_replay.stubs count drift")
    report.check(len(oracle_stubs) == 1, f"{path}.oracle tool stub count drift")
    if stubs and oracle_stubs:
        stub = require_dict(report, stubs[0], f"{path}.tool_replay.stubs[0]")
        oracle_stub = require_dict(report, oracle_stubs[0], f"{path}.oracle.tool_stubs[0]")
        stub_core = {key: stub.get(key) for key in ("round", "name", "args", "output")}
        report.check(stub_core == oracle_stub, f"{path}.tool stub drift from M11.1")
        report.check(stub.get("inserted_role") == "tool", f"{path}.tool inserted role drift")
        report.check(stub.get("inserted_after_round") == oracle_stub.get("round"), f"{path}.tool insertion round drift")

        messages = require_list(report, tool_replay.get("tool_result_messages"), f"{path}.tool_result_messages")
        report.check(messages == [oracle_stub.get("output")], f"{path}.tool result message drift")
        prompt = rendered_case.get("prompt_text")
        report.check(isinstance(prompt, str), f"{path}.M11.2 prompt missing")
        if isinstance(prompt, str):
            report.check(str(oracle_stub.get("output")) in prompt, f"{path}.M11.2 prompt missing tool output")
            report.check("<tool_result>" in prompt, f"{path}.M11.2 prompt missing tool-result tag")

    session_replay = require_dict(report, case.get("session_replay"), f"{path}.session_replay")
    report.check(session_replay.get("operations") == [], f"{path}.session operations must be empty")
    report.check(case.get("final_visible_output") == expected.get("final_visible_output"), f"{path}.final output drift")
    report.check(case.get("final_visible_output") == rendered_case.get("final_visible_output"), f"{path}.M11.2 final output drift")
    report.check(case.get("final_output_source") == "model_event_round_1", f"{path}.final output source drift")
    report.check(tool_replay.get("rendered_context_case") == "single_tool_round", f"{path}.rendered context case drift")
    report.check(tool_replay.get("rendered_context_contains_tool_result") is True, f"{path}.rendered tool result flag drift")


def validate_session_case(
    report: Report,
    case: dict[str, Any],
    oracle_case: dict[str, Any],
    rendered_case: dict[str, Any],
    path: str,
) -> None:
    expected = require_dict(report, oracle_case.get("expected"), f"{path}.oracle.expected")
    tool_replay = require_dict(report, case.get("tool_replay"), f"{path}.tool_replay")
    report.check(tool_replay.get("tool_sequence") == [], f"{path}.tool sequence must be empty")
    report.check(tool_replay.get("stubs") == [], f"{path}.tool stubs must be empty")
    report.check(tool_replay.get("tool_result_messages") == [], f"{path}.tool messages must be empty")

    expected_inputs = command_inputs(oracle_case)
    actual_inputs = require_list(report, case.get("command_inputs"), f"{path}.command_inputs")
    report.check(actual_inputs == expected_inputs, f"{path}.command input drift")

    session_replay = require_dict(report, case.get("session_replay"), f"{path}.session_replay")
    report.check(session_replay.get("normalized_sessions") == ["<SESSION:alpha>"], f"{path}.normalized session drift")
    operations = require_list(report, session_replay.get("operations"), f"{path}.session_replay.operations")
    expected_ops = require_list(report, expected.get("session_operations"), f"{path}.oracle.expected.session_operations")
    report.check(len(operations) == len(expected_ops), f"{path}.session operation count drift")
    for idx, raw_op in enumerate(operations):
        op = require_dict(report, raw_op, f"{path}.session_replay.operations[{idx}]")
        report.check(op.get("step") == idx, f"{path}.session op step drift at {idx}")
        if idx < len(expected_inputs):
            report.check(op.get("input") == expected_inputs[idx], f"{path}.session op input drift at {idx}")
        if idx < len(expected_ops):
            report.check(stripped_operation(op) == expected_ops[idx], f"{path}.session op drift at {idx}")

    model_events = require_list(report, oracle_case.get("model_events"), f"{path}.oracle.model_events")
    model_before = model_events[-1].get("text") if model_events and isinstance(model_events[-1], dict) else None
    report.check(case.get("model_visible_output_before_commands") == model_before, f"{path}.model visible output drift")
    report.check(
        case.get("model_visible_output_before_commands") == rendered_case.get("final_visible_output"),
        f"{path}.M11.2 model output drift",
    )
    prompt = rendered_case.get("prompt_text")
    report.check(isinstance(prompt, str), f"{path}.M11.2 prompt missing")
    if isinstance(prompt, str):
        for command in expected_inputs:
            report.check(command not in prompt, f"{path}.M11.2 prompt leaked command {command}")

    report.check(case.get("final_visible_output") == expected.get("final_visible_output"), f"{path}.final command output drift")
    report.check(case.get("final_output_source") == "session_command_new", f"{path}.final output source drift")


def validate_case(
    report: Report,
    case: Any,
    oracle_cases: dict[str, dict[str, Any]],
    rendered_cases: dict[str, dict[str, Any]],
    path: str,
) -> str | None:
    obj = require_dict(report, case, path)
    case_id = obj.get("id")
    report.check(case_id in EXPECTED_CASES, f"{path}.id unexpected")
    if not isinstance(case_id, str):
        return None

    sources = set(require_list(report, obj.get("replay_sources"), f"{path}.replay_sources"))
    report.check({"M11.1", "M11.2"} <= sources, f"{path}.replay sources incomplete")
    roles = require_list(report, obj.get("transcript_roles"), f"{path}.transcript_roles")
    oracle_case = oracle_cases.get(case_id, {})
    rendered_case = rendered_cases.get(case_id, {})
    expected_roles = oracle_case.get("expected", {}).get("transcript_roles")
    report.check(roles == expected_roles, f"{path}.transcript roles drift from M11.1")
    report.check(roles == rendered_case.get("message_roles"), f"{path}.transcript roles drift from M11.2")

    if case_id == "single_tool_round":
        validate_single_tool_case(report, obj, oracle_case, rendered_case, path)
    elif case_id == "session_switching_commands":
        validate_session_case(report, obj, oracle_case, rendered_case, path)
    return case_id


def validate_root(obj: dict[str, Any], m11_1: dict[str, Any], m11_2: dict[str, Any]) -> Report:
    report = Report()
    report.check(obj.get("schema") == SCHEMA, "schema mismatch")
    report.check(obj.get("milestone") == MILESTONE, "milestone mismatch")
    report.check(obj.get("source") == SOURCE, "source mismatch")
    report.check("M11.1 current-C trace replay fixture" in str(obj.get("oracle", "")), "oracle missing M11.1")
    report.check("M11.2 rendered context artifact" in str(obj.get("oracle", "")), "oracle missing M11.2")
    report.check(obj.get("live_execution") is False, "live execution flag drift")
    report.check(obj.get("model_sampling") is False, "model sampling flag drift")

    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    for forbidden in ("/Users/", "/workspace/ds4", "duration_ms", "pid", "timestamp", "session_sha"):
        report.check(forbidden not in raw, f"un-normalized field/path present: {forbidden}")

    oracle_cases = index_cases(m11_1)
    rendered_cases = index_cases(m11_2)
    cases = require_list(report, obj.get("cases"), "cases")
    ids: set[str] = set()
    for idx, raw_case in enumerate(cases):
        case_id = validate_case(report, raw_case, oracle_cases, rendered_cases, f"cases[{idx}]")
        if case_id:
            report.check(case_id not in ids, f"duplicate case {case_id}")
            ids.add(case_id)
    report.check(ids == EXPECTED_CASES, "case coverage drift")
    return report


def check_referenced_file(report: Report, manifest_path: Path, node: dict[str, Any], expected_path: str, label: str) -> None:
    report.check(node.get("path") == expected_path, f"manifest {label} path drift")
    ref_path = (manifest_path.parent / str(node.get("path", ""))).resolve()
    report.check(ref_path.exists(), f"manifest {label} path missing")
    if ref_path.exists():
        report.check(node.get("sha256") == sha256_file(ref_path), f"manifest {label} sha256 drift")


def check_manifest(manifest_path: Path, baseline_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(manifest_path), "manifest")
    report.check(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    report.check(manifest.get("milestone") == MILESTONE, "manifest milestone mismatch")
    current = require_dict(report, manifest.get("dumps", {}).get("deterministic_replay"), "manifest.dumps.deterministic_replay")
    report.check(current.get("path") == baseline_path.name, "manifest deterministic replay path mismatch")
    report.check(current.get("size_bytes") == baseline_path.stat().st_size, "manifest deterministic replay size drift")
    report.check(current.get("sha256") == sha256_file(baseline_path), "manifest deterministic replay sha256 drift")
    oracles = require_dict(report, manifest.get("oracles"), "manifest.oracles")
    check_referenced_file(
        report,
        manifest_path,
        require_dict(report, oracles.get("m11_1_current_c"), "manifest.oracles.m11_1_current_c"),
        "../m11.1/current-c.json",
        "M11.1 oracle",
    )
    check_referenced_file(
        report,
        manifest_path,
        require_dict(report, oracles.get("m11_2_rendered_context"), "manifest.oracles.m11_2_rendered_context"),
        "../m11.2/rendered-context.json",
        "M11.2 oracle",
    )
    commands = require_list(report, manifest.get("validation"), "manifest.validation")
    report.check(any("compare_agent_deterministic_replay.py" in cmd for cmd in commands if isinstance(cmd, str)), "manifest missing comparator command")
    return report


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-agent-deterministic-replay-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_negative_tests(original: dict[str, Any], m11_1: dict[str, Any], m11_2: dict[str, Any]) -> Report:
    report = Report()

    def expect_failure(label: str, mutate) -> None:
        bad = copy.deepcopy(original)
        mutate(bad)
        result = validate_root(bad, m11_1, m11_2)
        report.check(not result.ok, f"negative test failed to catch {label}")

    expect_failure("schema drift", lambda obj: obj.__setitem__("schema", "wrong"))
    expect_failure("live execution claim", lambda obj: obj.__setitem__("live_execution", True))
    expect_failure("tool sequence drift", lambda obj: obj["cases"][0]["tool_replay"]["tool_sequence"].clear())
    expect_failure("tool output drift", lambda obj: obj["cases"][0]["tool_replay"]["stubs"][0].__setitem__("output", "wrong"))
    expect_failure("rendered tool flag drift", lambda obj: obj["cases"][0]["tool_replay"].__setitem__("rendered_context_contains_tool_result", False))
    expect_failure("session command input drift", lambda obj: obj["cases"][1]["command_inputs"].__setitem__(0, "/wrong"))
    expect_failure("session operation order drift", lambda obj: obj["cases"][1]["session_replay"]["operations"].reverse())
    expect_failure("final command output drift", lambda obj: obj["cases"][1].__setitem__("final_visible_output", "wrong"))
    expect_failure("raw path leak", lambda obj: obj["cases"][1]["session_replay"].__setitem__("raw_path", "/workspace/ds4"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "deterministic-replay.json"
        baseline.write_text(json.dumps(original, ensure_ascii=False))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "milestone": MILESTONE,
            "dumps": {
                "deterministic_replay": {
                    "path": "deterministic-replay.json",
                    "size_bytes": baseline.stat().st_size,
                    "sha256": "0" * 64,
                }
            },
            "oracles": {
                "m11_1_current_c": {"path": "../m11.1/current-c.json", "sha256": "0" * 64},
                "m11_2_rendered_context": {"path": "../m11.2/rendered-context.json", "sha256": "0" * 64},
            },
            "validation": ["python3 ds4-parity/compare_agent_deterministic_replay.py --negative-test"],
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
    parser.add_argument("--m11-2-baseline", type=Path, default=M11_2_BASELINE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    m11_1 = require_dict(report, load_json(args.m11_1_baseline), "m11_1")
    m11_2 = require_dict(report, load_json(args.m11_2_baseline), "m11_2")
    baseline = require_dict(report, load_json(args.baseline), "baseline")
    merge("baseline", report, validate_root(baseline, m11_1, m11_2))
    if args.manifest:
        merge("manifest", report, check_manifest(args.manifest, args.baseline))

    rc, stdout, stderr = run_rust_dump()
    report.check(rc == 0, f"Rust deterministic replay failed rc={rc} stderr={stderr.strip()}")
    if rc == 0:
        try:
            rust = require_dict(report, json.loads(stdout), "rust")
            merge("rust schema", report, validate_root(rust, m11_1, m11_2))
            diff = first_diff(baseline, rust)
            report.check(diff is None, f"Rust deterministic replay drift: {diff}")
        except json.JSONDecodeError as exc:
            report.check(False, f"Rust deterministic replay did not emit JSON: {exc}")

    if args.negative_test:
        merge("negative", report, run_negative_tests(baseline, m11_1, m11_2))

    if report.ok:
        print(f"agent deterministic replay comparison: PASS, {report.checks} checks")
        return 0
    print(f"agent deterministic replay comparison: FAIL, {report.checks} checks", file=sys.stderr)
    for err in report.errors:
        print(f" - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
