#!/usr/bin/env python3
"""Validate the M6.6a current-C decode stop-policy oracle dump."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.6a" / "current-c.json"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.6a" / "manifest.json"

REQUIRED_CASES = {
    "cli_eos_stop",
    "cli_max_tokens_length",
    "server_openai_eos_stop",
    "server_openai_max_tokens_length",
    "server_openai_user_stop_sequence",
    "server_openai_stream_holds_stop_tail",
    "server_openai_stream_stop_hit_discards_tail",
    "server_openai_stream_holds_partial_utf8",
    "server_openai_stop_mid_utf8_boundary",
    "server_openai_tool_call_boundary",
    "server_responses_length_mapping",
    "server_anthropic_tool_mapping",
    "agent_eos_stop",
    "agent_max_tokens_length",
}

EXPECTED = {
    "cli_eos_stop": {
        "finish": "stop",
        "visible": "636c692068656c6c6f",
        "completion": 1,
    },
    "cli_max_tokens_length": {
        "finish": "length",
        "visible": "6162",
        "completion": 2,
    },
    "server_openai_eos_stop": {
        "finish": "stop",
        "visible": "7365727665722068656c6c6f",
        "completion": 1,
        "openai": "stop",
    },
    "server_openai_max_tokens_length": {
        "finish": "length",
        "visible": "6f6e652074776f",
        "completion": 2,
        "openai": "length",
    },
    "server_openai_user_stop_sequence": {
        "finish": "stop",
        "visible": "616e7377657220",
        "completion": 2,
        "invalidates": True,
        "stop_pos": 7,
        "stop_len": 4,
    },
    "server_openai_stream_holds_stop_tail": {
        "finish": "stop",
        "visible": "68656c6c6f203c2f",
        "streamed": "68656c6c6f203c2f",
        "held_tail_step0": "6c6f203c2f",
    },
    "server_openai_stream_stop_hit_discards_tail": {
        "finish": "stop",
        "visible": "70726520",
        "streamed": "70726520",
        "completion": 2,
        "invalidates": True,
        "stop_pos": 4,
        "stop_len": 4,
        "held_tail_step0": "205354",
        "hit_stop_step1": True,
    },
    "server_openai_stream_holds_partial_utf8": {
        "finish": "stop",
        "visible": "e282ac206f6b",
        "streamed": "e282ac206f6b",
        "held_tail_step0": "e282",
    },
    "server_openai_stop_mid_utf8_boundary": {
        "finish": "stop",
        "visible": "e2",
        "streamed": "e2",
        "completion": 2,
        "invalidates": True,
        "stop_pos": 1,
        "stop_len": 4,
        "held_tail_step0": "e25354",
        "hit_stop_step1": True,
    },
    "server_openai_tool_call_boundary": {
        "finish": "tool_calls",
        "visible": "492077696c6c2063616c6c2e",
        "tool_calls": 1,
        "openai": "tool_calls",
    },
    "server_responses_length_mapping": {
        "finish": "length",
        "responses_status": "incomplete",
        "responses_item_status": "incomplete",
        "responses_incomplete_reason": "max_tokens",
    },
    "server_anthropic_tool_mapping": {
        "finish": "tool_calls",
        "anthropic": "tool_use",
        "tool_calls": 1,
    },
    "agent_eos_stop": {
        "finish": "stop",
        "visible": "6167656e742068656c6c6f",
        "completion": 1,
        "agent_eos": True,
    },
    "agent_max_tokens_length": {
        "finish": "length",
        "visible": "7879",
        "completion": 2,
        "agent_eos": True,
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


def require_dict(report: Report, value: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{path}: expected object")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, path: str) -> list[Any]:
    report.check(isinstance(value, list), f"{path}: expected array")
    return value if isinstance(value, list) else []


def check_hex(report: Report, value: Any, path: str) -> None:
    report.check(isinstance(value, str), f"{path}: expected hex string")
    if isinstance(value, str):
        report.check(len(value) % 2 == 0, f"{path}: odd hex length")
        try:
            bytes.fromhex(value)
        except ValueError:
            report.check(False, f"{path}: invalid hex")


def check_request(report: Report, request: Any, path: str) -> None:
    obj = require_dict(report, request, path)
    report.check(obj.get("surface") in {"cli", "server", "agent"}, f"{path}.surface invalid")
    report.check(obj.get("api") in {"openai", "anthropic", "responses"}, f"{path}.api invalid")
    report.check(obj.get("kind") in {"chat", "completion"}, f"{path}.kind invalid")
    report.check(isinstance(obj.get("stream"), bool), f"{path}.stream invalid")
    report.check(isinstance(obj.get("has_tools"), bool), f"{path}.has_tools invalid")
    report.check(isinstance(obj.get("max_tokens"), int) and obj["max_tokens"] >= 0, f"{path}.max_tokens invalid")
    stops = require_list(report, obj.get("stops"), f"{path}.stops")
    for idx, stop in enumerate(stops):
        report.check(isinstance(stop, str) and stop, f"{path}.stops[{idx}] invalid")


def check_schedule(report: Report, schedule: Any, path: str) -> None:
    items = require_list(report, schedule, path)
    report.check(bool(items), f"{path}: empty schedule")
    for idx, raw in enumerate(items):
        item = require_dict(report, raw, f"{path}[{idx}]")
        report.check(item.get("index") == idx, f"{path}[{idx}].index drift")
        report.check(isinstance(item.get("eos"), bool), f"{path}[{idx}].eos invalid")
        check_hex(report, item.get("text_hex"), f"{path}[{idx}].text_hex")


def check_stream_steps(report: Report, steps: Any, path: str) -> dict[int, dict[str, Any]]:
    items = require_list(report, steps, path)
    by_step: dict[int, dict[str, Any]] = {}
    for idx, raw in enumerate(items):
        item = require_dict(report, raw, f"{path}[{idx}]")
        step = item.get("step")
        report.check(isinstance(step, int), f"{path}[{idx}].step invalid")
        if isinstance(step, int):
            by_step[step] = item
        report.check(isinstance(item.get("text_len"), int), f"{path}[{idx}].text_len invalid")
        report.check(isinstance(item.get("stream_safe_len"), int), f"{path}[{idx}].stream_safe_len invalid")
        check_hex(report, item.get("delta_hex"), f"{path}[{idx}].delta_hex")
        check_hex(report, item.get("held_tail_hex"), f"{path}[{idx}].held_tail_hex")
        report.check(isinstance(item.get("hit_stop"), bool), f"{path}[{idx}].hit_stop invalid")
        report.check(isinstance(item.get("stop_pos"), int), f"{path}[{idx}].stop_pos invalid")
        report.check(isinstance(item.get("stop_len"), int), f"{path}[{idx}].stop_len invalid")
    return by_step


def check_result(report: Report, result: Any, name: str, path: str) -> None:
    obj = require_dict(report, result, path)
    expected = EXPECTED[name]
    report.check(obj.get("finish_reason") == expected["finish"], f"{name}: finish drift")
    report.check(obj.get("completion_tokens") == expected.get("completion", obj.get("completion_tokens")), f"{name}: completion drift")
    check_hex(report, obj.get("raw_text_hex"), f"{path}.raw_text_hex")
    check_hex(report, obj.get("visible_text_hex"), f"{path}.visible_text_hex")
    check_hex(report, obj.get("reasoning_hex"), f"{path}.reasoning_hex")
    check_hex(report, obj.get("streamed_text_hex"), f"{path}.streamed_text_hex")
    report.check(obj.get("visible_text_hex") == expected.get("visible", obj.get("visible_text_hex")), f"{name}: visible text drift")
    if expected.get("tool_calls", 0):
        raw = obj.get("raw_text_hex")
        visible = obj.get("visible_text_hex")
        report.check(isinstance(raw, str) and isinstance(visible, str) and raw.startswith(visible), f"{name}: raw text no longer starts with visible text")
        report.check(isinstance(raw, str) and "3cefbd9c44534d4cefbd9c746f6f6c5f63616c6c733e" in raw, f"{name}: raw DSML block missing")
    else:
        report.check(obj.get("raw_text_hex") == expected.get("visible", obj.get("raw_text_hex")), f"{name}: raw text drift")
    if "streamed" in expected:
        report.check(obj.get("streamed_text_hex") == expected["streamed"], f"{name}: streamed text drift")
    report.check(obj.get("session_invalidation_required") is expected.get("invalidates", False), f"{name}: invalidation drift")
    report.check(obj.get("transcript_eos_appended") is expected.get("agent_eos", False), f"{name}: transcript EOS drift")

    stop = require_dict(report, obj.get("stop_boundary"), f"{path}.stop_boundary")
    report.check(stop.get("pos") == expected.get("stop_pos", -1), f"{name}: stop pos drift")
    report.check(stop.get("len") == expected.get("stop_len", 0), f"{name}: stop len drift")

    tool = require_dict(report, obj.get("tool_boundary"), f"{path}.tool_boundary")
    report.check(tool.get("tool_call_count") == expected.get("tool_calls", 0), f"{name}: tool count drift")
    if expected.get("tool_calls", 0):
        report.check(tool.get("saw_start") is True, f"{name}: missing tool start")
        report.check(tool.get("saw_end") is True, f"{name}: missing tool end")

    api = require_dict(report, obj.get("api_finish"), f"{path}.api_finish")
    if "openai" in expected:
        report.check(api.get("openai_finish_reason") == expected["openai"], f"{name}: openai finish drift")
    if "anthropic" in expected:
        report.check(api.get("anthropic_stop_reason") == expected["anthropic"], f"{name}: anthropic stop drift")
    if "responses_status" in expected:
        report.check(api.get("responses_status") == expected["responses_status"], f"{name}: responses status drift")
        report.check(api.get("responses_item_status") == expected["responses_item_status"], f"{name}: responses item status drift")
        report.check(api.get("responses_incomplete_reason") == expected["responses_incomplete_reason"], f"{name}: responses incomplete reason drift")

    steps = check_stream_steps(report, obj.get("stream_steps"), f"{path}.stream_steps")
    if "held_tail_step0" in expected:
        first = steps.get(0, {})
        report.check(first.get("held_tail_hex") == expected["held_tail_step0"], f"{name}: held tail drift")
    if expected.get("hit_stop_step1"):
        second = steps.get(1, {})
        report.check(second.get("hit_stop") is True, f"{name}: step 1 did not hit stop")
        report.check(second.get("held_tail_hex") == "", f"{name}: stop-hit held tail was not discarded")


def check_case(report: Report, raw: Any, path: str) -> str | None:
    case = require_dict(report, raw, path)
    name = case.get("name")
    report.check(isinstance(name, str) and bool(name), f"{path}.name invalid")
    if not isinstance(name, str):
        return None
    report.check(name in REQUIRED_CASES, f"{path}.name unknown: {name}")
    report.check(isinstance(case.get("source"), str) and bool(case.get("source")), f"{path}.source invalid")
    check_request(report, case.get("request"), f"{path}.request")
    check_schedule(report, case.get("schedule"), f"{path}.schedule")
    if name in EXPECTED:
        check_result(report, case.get("result"), name, f"{path}.result")
    return name


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.decode_policy_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-decode-stop-policy", "source mismatch")
    report.check(root.get("model") == "no model is loaded for this oracle", "model marker mismatch")
    cases = require_list(report, root.get("cases"), "cases")
    names: list[str] = []
    for idx, raw_case in enumerate(cases):
        name = check_case(report, raw_case, f"cases[{idx}]")
        if name:
            names.append(name)
    report.check(set(names) == REQUIRED_CASES, "case coverage drift")
    report.check(len(names) == len(set(names)), "duplicate case names")
    return report


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    manifest = load_json(path)
    report.check(manifest.get("schema") == "ds4.decode_policy_manifest.v1", "manifest schema mismatch")
    artifact_info = manifest.get("artifact")
    if not isinstance(artifact_info, dict):
        report.check(False, "manifest artifact missing")
        return report
    report.check(artifact_info.get("path") == str(artifact.relative_to(ROOT)), "manifest artifact path drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    report.check(artifact_info.get("bytes") == artifact.stat().st_size, "manifest artifact size drift")
    commands = manifest.get("validation_commands")
    report.check(isinstance(commands, list) and bool(commands), "manifest validation commands missing")
    return report


def run_negative_tests(obj: Any) -> Report:
    report = Report()
    indexes = {
        case.get("name"): idx
        for idx, case in enumerate(obj.get("cases", []))
        if isinstance(case, dict)
    }
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("finish drift", ["cases", indexes["cli_eos_stop"], "result", "finish_reason"], "length"),
        ("visible drift", ["cases", indexes["server_openai_eos_stop"], "result", "visible_text_hex"], "00"),
        ("stop invalidation drift", ["cases", indexes["server_openai_user_stop_sequence"], "result", "session_invalidation_required"], False),
        ("held-tail drift", ["cases", indexes["server_openai_stream_holds_stop_tail"], "result", "stream_steps", 0, "held_tail_hex"], ""),
        ("streaming stop flush drift", ["cases", indexes["server_openai_stream_stop_hit_discards_tail"], "result", "streamed_text_hex"], "7072652053544f50206166746572"),
        ("mid-utf8 stop drift", ["cases", indexes["server_openai_stop_mid_utf8_boundary"], "result", "stream_steps", 1, "held_tail_hex"], "53544f50"),
        ("raw DSML drift", ["cases", indexes["server_openai_tool_call_boundary"], "result", "raw_text_hex"], EXPECTED["server_openai_tool_call_boundary"]["visible"]),
        ("tool finish drift", ["cases", indexes["server_openai_tool_call_boundary"], "result", "tool_boundary", "tool_call_count"], 0),
        ("responses mapping drift", ["cases", indexes["server_responses_length_mapping"], "result", "api_finish", "responses_status"], "completed"),
        ("case coverage drift", ["cases"], obj["cases"][:-1]),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(obj)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = check_dump(bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    obj = load_json(args.baseline)
    schema = check_dump(obj)
    print_report("decode policy schema", schema)

    manifest = Report()
    if args.manifest.exists():
        manifest = check_manifest(args.manifest, args.baseline)
        print_report("decode policy manifest", manifest)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(obj)
        print_report("decode policy negative tests", negative)

    return 0 if schema.ok and manifest.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
