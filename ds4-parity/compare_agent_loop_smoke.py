#!/usr/bin/env python3
"""Validate the M11.4 Rust agent no-model loop smoke summary."""

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
M11_3_BASELINE = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.3" / "deterministic-replay.json"
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "agent" / "m11.4"
BASELINE = BASELINE_DIR / "loop-smoke.json"
MANIFEST = BASELINE_DIR / "manifest.json"
SCHEMA = "ds4.agent_loop_smoke.v1"
MANIFEST_SCHEMA = "ds4.agent_loop_smoke_baseline.v1"
MILESTONE = "M11.4"
SOURCE = "rust-agent-loop-smoke"
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


def operation_without_replay_fields(op: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in op.items() if key not in {"phase", "step", "active_session"}}


def validate_single_tool_case(
    report: Report,
    case: dict[str, Any],
    m11_1_case: dict[str, Any],
    m11_2_case: dict[str, Any],
    m11_3_case: dict[str, Any],
    path: str,
) -> None:
    expected = require_dict(report, m11_1_case.get("expected"), f"{path}.M11.1.expected")
    steps = require_list(report, case.get("loop_steps"), f"{path}.loop_steps")
    phases = [step.get("phase") for step in steps if isinstance(step, dict)]
    report.check(
        phases == ["render_prompt", "parse_model_event", "tool_replay", "render_after_tool", "final_model_event"],
        f"{path}.loop phase drift",
    )

    report.check(case.get("parsed_tool_sequence") == expected.get("tool_sequence"), f"{path}.parsed tool sequence drift from M11.1")
    report.check(
        case.get("parsed_tool_sequence") == m11_3_case.get("tool_replay", {}).get("tool_sequence"),
        f"{path}.parsed tool sequence drift from M11.3",
    )
    report.check(case.get("final_transcript_roles") == expected.get("transcript_roles"), f"{path}.final roles drift from M11.1")
    report.check(case.get("final_transcript_roles") == m11_3_case.get("transcript_roles"), f"{path}.final roles drift from M11.3")
    report.check(case.get("final_visible_output") == expected.get("final_visible_output"), f"{path}.final output drift from M11.1")
    report.check(case.get("final_visible_output") == m11_3_case.get("final_visible_output"), f"{path}.final output drift from M11.3")

    if len(steps) >= 5:
        render_before = require_dict(report, steps[0], f"{path}.loop_steps[0]")
        parse_step = require_dict(report, steps[1], f"{path}.loop_steps[1]")
        tool_step = require_dict(report, steps[2], f"{path}.loop_steps[2]")
        render_after = require_dict(report, steps[3], f"{path}.loop_steps[3]")
        final_step = require_dict(report, steps[4], f"{path}.loop_steps[4]")

        report.check(render_before.get("prompt_has_user_marker") is True, f"{path}.initial prompt user marker drift")
        report.check(render_before.get("prompt_has_tool_result") is False, f"{path}.initial prompt has premature tool result")
        report.check(parse_step.get("parser_state") == "done", f"{path}.parser state drift")
        report.check(parse_step.get("parsed_tool_calls") == 1, f"{path}.parsed tool-call count drift")
        report.check(parse_step.get("raw_dsml_preserved") is True, f"{path}.raw DSML preservation drift")

        stub = require_dict(report, m11_3_case.get("tool_replay", {}).get("stubs", [None])[0], f"{path}.M11.3.stub")
        tool_core = {key: tool_step.get(key) for key in ("round", "name", "args", "output")}
        stub_core = {key: stub.get(key) for key in ("round", "name", "args", "output")}
        report.check(tool_core == stub_core, f"{path}.tool replay drift from M11.3")
        report.check(tool_step.get("source") == "deterministic_stub", f"{path}.tool source drift")
        report.check(tool_step.get("inserted_role") == "tool", f"{path}.tool inserted role drift")
        report.check(tool_step.get("live_tool_execution") is False, f"{path}.live tool execution claim")
        report.check(render_after.get("prompt_has_tool_result") is True, f"{path}.post-tool prompt missing tool result")
        report.check(render_after.get("prompt_has_tool_output") is True, f"{path}.post-tool prompt missing tool output")
        report.check(final_step.get("visible") == expected.get("final_visible_output"), f"{path}.final model event drift")

        prompt = m11_2_case.get("prompt_text")
        report.check(isinstance(prompt, str) and str(stub.get("output")) in prompt, f"{path}.M11.2 prompt/tool drift")


def validate_session_case(
    report: Report,
    case: dict[str, Any],
    m11_1_case: dict[str, Any],
    m11_2_case: dict[str, Any],
    m11_3_case: dict[str, Any],
    path: str,
) -> None:
    expected = require_dict(report, m11_1_case.get("expected"), f"{path}.M11.1.expected")
    steps = require_list(report, case.get("loop_steps"), f"{path}.loop_steps")
    phases = [step.get("phase") for step in steps if isinstance(step, dict)]
    report.check(phases == ["model_event"] + ["session_command"] * 5, f"{path}.loop phase drift")

    report.check(case.get("saved_sessions") == ["<SESSION:alpha>"], f"{path}.saved sessions drift")
    report.check(case.get("active_session") == "<SESSION:new>", f"{path}.active session drift")
    report.check(case.get("final_transcript_roles") == ["system"], f"{path}.new-session transcript drift")
    report.check(case.get("final_visible_output") == expected.get("final_visible_output"), f"{path}.final output drift from M11.1")
    report.check(case.get("final_visible_output") == m11_3_case.get("final_visible_output"), f"{path}.final output drift from M11.3")

    if steps:
        model_step = require_dict(report, steps[0], f"{path}.loop_steps[0]")
        model_events = require_list(report, m11_1_case.get("model_events"), f"{path}.M11.1.model_events")
        model_visible = model_events[-1].get("text") if model_events and isinstance(model_events[-1], dict) else None
        report.check(model_step.get("visible") == model_visible, f"{path}.model event visible drift")
        report.check(model_step.get("visible") == m11_2_case.get("final_visible_output"), f"{path}.M11.2 visible drift")
        report.check(model_step.get("visible") == m11_3_case.get("model_visible_output_before_commands"), f"{path}.M11.3 visible drift")
        report.check(model_step.get("transcript_roles") == expected.get("transcript_roles"), f"{path}.model transcript role drift")

    m11_3_ops = require_list(report, m11_3_case.get("session_replay", {}).get("operations"), f"{path}.M11.3.operations")
    report.check(len(steps) == len(m11_3_ops) + 1, f"{path}.session loop length drift")
    for idx, expected_op in enumerate(m11_3_ops):
        if idx + 1 >= len(steps):
            continue
        actual = require_dict(report, steps[idx + 1], f"{path}.loop_steps[{idx + 1}]")
        expected_core = {key: value for key, value in expected_op.items() if key != "step"}
        report.check(actual.get("step") == idx + 1, f"{path}.session step drift at {idx}")
        report.check(operation_without_replay_fields(actual) == expected_core, f"{path}.session command drift at {idx}")

    prompt = m11_2_case.get("prompt_text")
    report.check(isinstance(prompt, str), f"{path}.M11.2 prompt missing")
    if isinstance(prompt, str):
        for command in m11_3_case.get("command_inputs", []):
            report.check(command not in prompt, f"{path}.M11.2 prompt leaked command {command}")


def validate_case(
    report: Report,
    case: Any,
    m11_1_cases: dict[str, dict[str, Any]],
    m11_2_cases: dict[str, dict[str, Any]],
    m11_3_cases: dict[str, dict[str, Any]],
    path: str,
) -> str | None:
    obj = require_dict(report, case, path)
    case_id = obj.get("id")
    report.check(case_id in EXPECTED_CASES, f"{path}.id unexpected")
    if not isinstance(case_id, str):
        return None

    sources = set(require_list(report, obj.get("replay_sources"), f"{path}.replay_sources"))
    report.check({"M11.1", "M11.2", "M11.3"} <= sources, f"{path}.replay sources incomplete")
    if case_id == "single_tool_round":
        validate_single_tool_case(report, obj, m11_1_cases.get(case_id, {}), m11_2_cases.get(case_id, {}), m11_3_cases.get(case_id, {}), path)
    elif case_id == "session_switching_commands":
        validate_session_case(report, obj, m11_1_cases.get(case_id, {}), m11_2_cases.get(case_id, {}), m11_3_cases.get(case_id, {}), path)
    return case_id


def validate_root(obj: dict[str, Any], m11_1: dict[str, Any], m11_2: dict[str, Any], m11_3: dict[str, Any]) -> Report:
    report = Report()
    report.check(obj.get("schema") == SCHEMA, "schema mismatch")
    report.check(obj.get("milestone") == MILESTONE, "milestone mismatch")
    report.check(obj.get("source") == SOURCE, "source mismatch")
    oracle = str(obj.get("oracle", ""))
    report.check("M11.1 trace fixture" in oracle, "oracle missing M11.1")
    report.check("M11.2 rendered context" in oracle, "oracle missing M11.2")
    report.check("M11.3 deterministic replay" in oracle, "oracle missing M11.3")
    report.check(obj.get("model_sampling") is False, "model sampling claim drift")
    report.check(obj.get("live_tool_execution") is False, "live tool execution claim drift")
    report.check("deferred" in str(obj.get("manual_smoke", "")), "manual smoke claim is not deferred")

    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    for forbidden in ("/Users/", "/workspace/ds4", "duration_ms", "pid", "timestamp", "session_sha"):
        report.check(forbidden not in raw, f"un-normalized field/path present: {forbidden}")

    m11_1_cases = index_cases(m11_1)
    m11_2_cases = index_cases(m11_2)
    m11_3_cases = index_cases(m11_3)
    cases = require_list(report, obj.get("cases"), "cases")
    ids: set[str] = set()
    for idx, raw_case in enumerate(cases):
        case_id = validate_case(report, raw_case, m11_1_cases, m11_2_cases, m11_3_cases, f"cases[{idx}]")
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
    current = require_dict(report, manifest.get("dumps", {}).get("loop_smoke"), "manifest.dumps.loop_smoke")
    report.check(current.get("path") == baseline_path.name, "manifest loop smoke path mismatch")
    report.check(current.get("size_bytes") == baseline_path.stat().st_size, "manifest loop smoke size drift")
    report.check(current.get("sha256") == sha256_file(baseline_path), "manifest loop smoke sha256 drift")
    oracles = require_dict(report, manifest.get("oracles"), "manifest.oracles")
    check_referenced_file(report, manifest_path, require_dict(report, oracles.get("m11_1_current_c"), "manifest.oracles.m11_1_current_c"), "../m11.1/current-c.json", "M11.1 oracle")
    check_referenced_file(report, manifest_path, require_dict(report, oracles.get("m11_2_rendered_context"), "manifest.oracles.m11_2_rendered_context"), "../m11.2/rendered-context.json", "M11.2 oracle")
    check_referenced_file(report, manifest_path, require_dict(report, oracles.get("m11_3_deterministic_replay"), "manifest.oracles.m11_3_deterministic_replay"), "../m11.3/deterministic-replay.json", "M11.3 oracle")
    commands = require_list(report, manifest.get("validation"), "manifest.validation")
    report.check(any("compare_agent_loop_smoke.py" in cmd for cmd in commands if isinstance(cmd, str)), "manifest missing comparator command")
    return report


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-agent-loop-smoke-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_negative_tests(original: dict[str, Any], m11_1: dict[str, Any], m11_2: dict[str, Any], m11_3: dict[str, Any]) -> Report:
    report = Report()

    def expect_failure(label: str, mutate) -> None:
        bad = copy.deepcopy(original)
        mutate(bad)
        result = validate_root(bad, m11_1, m11_2, m11_3)
        report.check(not result.ok, f"negative test failed to catch {label}")

    expect_failure("schema drift", lambda obj: obj.__setitem__("schema", "wrong"))
    expect_failure("model sampling claim", lambda obj: obj.__setitem__("model_sampling", True))
    expect_failure("manual smoke overclaim", lambda obj: obj.__setitem__("manual_smoke", "passed live manual smoke"))
    expect_failure("parser state drift", lambda obj: obj["cases"][0]["loop_steps"][1].__setitem__("parser_state", "search"))
    expect_failure("tool output drift", lambda obj: obj["cases"][0]["loop_steps"][2].__setitem__("output", "wrong"))
    expect_failure("post-tool render drift", lambda obj: obj["cases"][0]["loop_steps"][3].__setitem__("prompt_has_tool_result", False))
    expect_failure("session order drift", lambda obj: obj["cases"][1]["loop_steps"].reverse())
    expect_failure("active session raw drift", lambda obj: obj["cases"][1].__setitem__("active_session", "/workspace/ds4"))
    expect_failure("final output drift", lambda obj: obj["cases"][1].__setitem__("final_visible_output", "wrong"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "loop-smoke.json"
        baseline.write_text(json.dumps(original, ensure_ascii=False))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "milestone": MILESTONE,
            "dumps": {
                "loop_smoke": {
                    "path": "loop-smoke.json",
                    "size_bytes": baseline.stat().st_size,
                    "sha256": "0" * 64,
                }
            },
            "oracles": {
                "m11_1_current_c": {"path": "../m11.1/current-c.json", "sha256": "0" * 64},
                "m11_2_rendered_context": {"path": "../m11.2/rendered-context.json", "sha256": "0" * 64},
                "m11_3_deterministic_replay": {"path": "../m11.3/deterministic-replay.json", "sha256": "0" * 64},
            },
            "validation": ["python3 ds4-parity/compare_agent_loop_smoke.py --negative-test"],
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
    parser.add_argument("--m11-3-baseline", type=Path, default=M11_3_BASELINE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    m11_1 = require_dict(report, load_json(args.m11_1_baseline), "m11_1")
    m11_2 = require_dict(report, load_json(args.m11_2_baseline), "m11_2")
    m11_3 = require_dict(report, load_json(args.m11_3_baseline), "m11_3")
    baseline = require_dict(report, load_json(args.baseline), "baseline")
    merge("baseline", report, validate_root(baseline, m11_1, m11_2, m11_3))
    if args.manifest:
        merge("manifest", report, check_manifest(args.manifest, args.baseline))

    rc, stdout, stderr = run_rust_dump()
    report.check(rc == 0, f"Rust agent loop smoke failed rc={rc} stderr={stderr.strip()}")
    if rc == 0:
        try:
            rust = require_dict(report, json.loads(stdout), "rust")
            merge("rust schema", report, validate_root(rust, m11_1, m11_2, m11_3))
            diff = first_diff(baseline, rust)
            report.check(diff is None, f"Rust agent loop smoke drift: {diff}")
        except json.JSONDecodeError as exc:
            report.check(False, f"Rust agent loop smoke did not emit JSON: {exc}")

    if args.negative_test:
        merge("negative", report, run_negative_tests(baseline, m11_1, m11_2, m11_3))

    if report.ok:
        print(f"agent loop smoke comparison: PASS, {report.checks} checks")
        return 0
    print(f"agent loop smoke comparison: FAIL, {report.checks} checks", file=sys.stderr)
    for err in report.errors:
        print(f" - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
