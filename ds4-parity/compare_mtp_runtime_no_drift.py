#!/usr/bin/env python3
"""Check MTP-off runtime target-stream no-drift artifacts against guard rows."""

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
CLI_ORACLE = ROOT / "ds4-parity/baselines/cli/m8.12a/current-c.json"
RUNTIME_SUMMARY = ROOT / "ds4-parity/baselines/kv/m9.8f5/runtime-rust-b300-summary.json"
M05_RESPONSES = ROOT / "ds4-parity/baselines/kv-artifacts/m0.5/responses"

EXPECTED_CLI_CASES = {
    "greedy_inline_nothink": {
        "stdout_sha256": "862550215bb33a4e6f591f4c1c52fcd03dc98022f1acad87ce807f7d58b8c03c",
        "stdout_bytes": 5,
        "tokens": "2",
        "guard_case": "one_shot_runtime_mtp_off",
    },
    "prompt_file_think": {
        "stdout_sha256": "e566cf8e60978ac10a300c2503a68e03edd6d162fb11e2496057d57346660af0",
        "stdout_bytes": 8,
        "tokens": "2",
        "guard_case": "one_shot_runtime_mtp_off",
    },
    "think_max_downgrade": {
        "stdout_sha256": "c36d4b240ac10dcf300ba8d9d5aafc33957b8c1976c7f5ae2b26e654701e1b74",
        "stdout_bytes": 3,
        "tokens": "1",
        "guard_case": "one_shot_runtime_mtp_off",
    },
}

EXPECTED_SERVER_CASES = {
    "seed_miss": {
        "guard_case": "server_runtime_mtp_off",
        "finish": "length",
        "content": "I notice",
        "prompt_tokens": 550,
        "cached_tokens": 0,
        "cache_write_tokens": 550,
        "cache_source": "none",
        "disk_cached_tokens": 0,
    },
    "seed_restore": {
        "guard_case": "server_runtime_mtp_off",
        "finish": "length",
        "content": "I notice",
        "prompt_tokens": 550,
        "cached_tokens": 550,
        "cache_write_tokens": 0,
        "cache_source": "disk-text",
        "disk_cached_tokens": 550,
    },
    "continuation_restore": {
        "guard_case": "server_runtime_mtp_off",
        "finish": "stop",
        "content": "kv continued",
        "prompt_tokens": 561,
        "cached_tokens": 552,
        "cache_write_tokens": 9,
        "cache_source": "disk-text",
        "disk_cached_tokens": 552,
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected JSON object")
    return obj


def run_guard_plan() -> dict[str, Any]:
    proc = subprocess.run(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-runtime-guard-plan", "--quiet"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def validate(
    cli_oracle: dict[str, Any],
    runtime_summary: dict[str, Any],
    guard_plan: dict[str, Any],
) -> Report:
    report = Report()
    guard_cases = named_cases(report, guard_plan.get("cases"), "guard")
    validate_guard(report, guard_cases)
    validate_cli_oracle(report, cli_oracle, guard_cases)
    validate_server_runtime_summary(report, runtime_summary, guard_cases)
    validate_static_report_wiring(report)
    return report


def named_cases(report: Report, value: Any, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    report.check(isinstance(value, list), f"{label}.cases must be a list")
    if not isinstance(value, list):
        return result
    for index, item in enumerate(value):
        report.check(isinstance(item, dict), f"{label}.cases[{index}] must be an object")
        if not isinstance(item, dict):
            continue
        case_id = item.get("id") or item.get("name")
        report.check(isinstance(case_id, str), f"{label}.cases[{index}].id missing")
        if isinstance(case_id, str):
            result[case_id] = item
    return result


def validate_guard(report: Report, guard_cases: dict[str, dict[str, Any]]) -> None:
    for guard_id in ["one_shot_runtime_mtp_off", "server_runtime_mtp_off"]:
        case = guard_cases.get(guard_id)
        report.check(case is not None, f"missing guard case {guard_id}")
        if not case:
            continue
        report.check(case.get("source_stream_case") == "mtp_disabled_after_first_token", f"{guard_id}.source drift")
        report.check(case.get("accepted_stream_delta") == "first_token", f"{guard_id}.stream drift")
        report.check(case.get("checkpoint_delta") == "1", f"{guard_id}.checkpoint drift")
        report.check(case.get("logits_source") == "target first-token logits", f"{guard_id}.logits drift")
        report.check(case.get("mtp_n_raw_keep") in (0, "0"), f"{guard_id}.mtp_n_raw drift")
        report.check(
            str(case.get("target_stream_visibility", "")).startswith("target_only"),
            f"{guard_id}.visibility drift",
        )
        report.check(case.get("error") == "none", f"{guard_id}.error drift")


def validate_cli_oracle(
    report: Report,
    cli_oracle: dict[str, Any],
    guard_cases: dict[str, dict[str, Any]],
) -> None:
    report.check(cli_oracle.get("schema") == "ds4.cli_generation_oracle.v1", "CLI oracle schema drift")
    report.check(cli_oracle.get("source") == "current-c-cli-one-shot-generation", "CLI oracle source drift")
    cli_cases = named_cases(report, cli_oracle.get("cases"), "cli_oracle")
    for case_id, expected in EXPECTED_CLI_CASES.items():
        case = cli_cases.get(case_id)
        report.check(case is not None, f"missing CLI oracle case {case_id}")
        guard = guard_cases.get(expected["guard_case"])
        report.check(guard is not None, f"{case_id}: missing guard {expected['guard_case']}")
        if not case:
            continue
        argv = [str(arg) for arg in case.get("argv", [])]
        report.check("--mtp" not in argv, f"{case_id}: MTP path unexpectedly present")
        report.check("--mtp-draft" not in argv, f"{case_id}: MTP draft override unexpectedly present")
        report.check("--temp" in argv and value_after(argv, "--temp") == "0", f"{case_id}: not a target argmax case")
        report.check(value_after(argv, "--tokens") == expected["tokens"], f"{case_id}: token count drift")
        report.check(case.get("exit_code") == 0, f"{case_id}: exit code drift")
        stdout = case.get("stdout", {})
        report.check(isinstance(stdout, dict), f"{case_id}: stdout missing")
        if isinstance(stdout, dict):
            report.check(stdout.get("sha256") == expected["stdout_sha256"], f"{case_id}: stdout hash drift")
            report.check(stdout.get("bytes") == expected["stdout_bytes"], f"{case_id}: stdout byte count drift")
        anchors = case.get("stderr_anchors", [])
        report.check(isinstance(anchors, list), f"{case_id}: stderr anchors missing")
        if isinstance(anchors, list):
            report.check("ds4: context buffers" in anchors, f"{case_id}: missing context stderr anchor")
            report.check("backend=cuda" in anchors, f"{case_id}: missing backend stderr anchor")


def validate_server_runtime_summary(
    report: Report,
    runtime_summary: dict[str, Any],
    guard_cases: dict[str, dict[str, Any]],
) -> None:
    report.check(runtime_summary.get("schema") == "ds4.runtime_kv_replay_summary.v1", "runtime summary schema drift")
    report.check(runtime_summary.get("source") == "rust-runtime-b300-replay", "runtime summary source drift")
    report.check(runtime_summary.get("milestone") == "M9.8f5", "runtime summary milestone drift")
    server_cases = named_cases(report, runtime_summary.get("cases"), "runtime_summary")
    for case_id, expected in EXPECTED_SERVER_CASES.items():
        case = server_cases.get(case_id)
        report.check(case is not None, f"missing Rust runtime summary case {case_id}")
        guard = guard_cases.get(expected["guard_case"])
        report.check(guard is not None, f"{case_id}: missing guard {expected['guard_case']}")
        if not case:
            continue
        current_c = load_json(M05_RESPONSES / f"{case_id}.json")
        current_message = ((current_c.get("choices") or [{}])[0].get("message") or {})
        usage = current_c.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        report.check(case.get("content") == current_message.get("content"), f"{case_id}: Rust/current-C content drift")
        report.check(case.get("finish") == (current_c.get("choices") or [{}])[0].get("finish_reason"), f"{case_id}: finish drift")
        for key in [
            "content",
            "finish",
            "prompt_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "cache_source",
            "disk_cached_tokens",
        ]:
            report.check(case.get(key) == expected[key], f"{case_id}.{key}: expected drift")
        report.check(case.get("prompt_tokens") == usage.get("prompt_tokens"), f"{case_id}: prompt token drift")
        report.check(case.get("cached_tokens") == details.get("cached_tokens"), f"{case_id}: cached token drift")
        report.check(
            case.get("cache_write_tokens") == details.get("cache_write_tokens"),
            f"{case_id}: cache write token drift",
        )
        report.check(usage.get("completion_tokens") == 2, f"{case_id}: completion token drift")

    validate_ledger_cases(report, runtime_summary)


def validate_ledger_cases(report: Report, runtime_summary: dict[str, Any]) -> None:
    ledger_cases = named_cases(report, runtime_summary.get("ledger_cases"), "ledger")
    for case_id, expected in EXPECTED_SERVER_CASES.items():
        ledger = ledger_cases.get(case_id)
        report.check(ledger is not None, f"missing ledger case {case_id}")
        if not ledger:
            continue
        for key in ["prompt_tokens", "cached_tokens", "cache_write_tokens", "cache_source", "disk_cached_tokens"]:
            report.check(ledger.get(key) == expected[key], f"{case_id}.ledger.{key}: drift")
        events = ledger.get("events")
        report.check(isinstance(events, list) and events, f"{case_id}: ledger events missing")
        if isinstance(events, list) and events:
            names = [event.get("name") for event in events if isinstance(event, dict)]
            report.check("cache_decision" in names, f"{case_id}: cache decision event missing")
            report.check("maybe_store_continued" in names, f"{case_id}: continued-store probe missing")


def validate_static_report_wiring(report: Report) -> None:
    run_cli = (ROOT / "ds4-parity/run_cli_parity_report.py").read_text()
    run_server = (ROOT / "ds4-parity/run_server_parity_report.py").read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()
    for snippet in [
        "Rust one-shot runtime comparator",
        "compare_cli_one_shot_runtime.py",
        "ds4-cli-one-shot-rs",
    ]:
        report.check(snippet in run_cli, f"CLI runtime report missing {snippet!r}")
    for snippet in [
        "M9.8f5 B300 Rust runtime replay summary",
        "check_runtime_kv_replay_summary.py",
    ]:
        report.check(snippet in run_server, f"server runtime report missing {snippet!r}")
    for snippet in ["compare_mtp_runtime_no_drift.py", "M10.8g3b MTP runtime no-drift comparator"]:
        report.check(snippet in run_report, f"unified report missing {snippet!r}")
    report.check(
        "compare_mtp_runtime_no_drift.py --negative-test" in readme,
        "README missing M10.8g3b command",
    )


def value_after(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def run_negative_tests(
    cli_oracle: dict[str, Any],
    runtime_summary: dict[str, Any],
    guard_plan: dict[str, Any],
) -> Report:
    report = Report()
    mutations = [
        ("CLI adds MTP path", lambda cli, _runtime, _guard: cli["cases"][0]["argv"].extend(["--mtp", "draft.gguf"])),
        ("CLI stdout hash drift", lambda cli, _runtime, _guard: mutate_cli_case(cli, "greedy_inline_nothink", ["stdout", "sha256"], "0" * 64)),
        ("Rust runtime content drift", lambda _cli, runtime, _guard: mutate_runtime_case(runtime, "seed_miss", "content", "wrong")),
        ("Rust runtime cache drift", lambda _cli, runtime, _guard: mutate_runtime_case(runtime, "seed_restore", "cached_tokens", 0)),
        ("guard source drift", lambda _cli, _runtime, guard: mutate_guard_case(guard, "server_runtime_mtp_off", "source_stream_case", "first_draft_miss")),
        ("guard visibility drift", lambda _cli, _runtime, guard: mutate_guard_case(guard, "one_shot_runtime_mtp_off", "target_stream_visibility", "speculative")),
    ]
    for name, mutate in mutations:
        cli_copy = copy.deepcopy(cli_oracle)
        runtime_copy = copy.deepcopy(runtime_summary)
        guard_copy = copy.deepcopy(guard_plan)
        mutate(cli_copy, runtime_copy, guard_copy)
        result = validate(cli_copy, runtime_copy, guard_copy)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def mutate_cli_case(data: dict[str, Any], case_id: str, path: list[str], value: Any) -> None:
    for case in data["cases"]:
        if case.get("id") == case_id:
            target = case
            for item in path[:-1]:
                target = target[item]
            target[path[-1]] = value
            return
    raise AssertionError(case_id)


def mutate_runtime_case(data: dict[str, Any], case_id: str, key: str, value: Any) -> None:
    for case in data["cases"]:
        if case.get("name") == case_id:
            case[key] = value
            return
    raise AssertionError(case_id)


def mutate_guard_case(data: dict[str, Any], case_id: str, key: str, value: Any) -> None:
    for case in data["cases"]:
        if case.get("id") == case_id:
            case[key] = value
            return
    raise AssertionError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-oracle", type=Path, default=CLI_ORACLE)
    parser.add_argument("--runtime-summary", type=Path, default=RUNTIME_SUMMARY)
    parser.add_argument("--guard-plan", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        cli_oracle = load_json(args.cli_oracle)
        runtime_summary = load_json(args.runtime_summary)
        guard_plan = load_json(args.guard_plan) if args.guard_plan else run_guard_plan()
    except Exception as exc:
        print(f"MTP runtime no-drift comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(cli_oracle, runtime_summary, guard_plan)
    if not report.ok:
        print("MTP runtime no-drift comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "MTP runtime no-drift comparator: PASS, "
        f"{len(EXPECTED_CLI_CASES)} CLI cases, {len(EXPECTED_SERVER_CASES)} server cases, "
        f"{report.checks} checks"
    )
    if args.negative_test:
        negative = run_negative_tests(cli_oracle, runtime_summary, guard_plan)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("MTP runtime no-drift negative tests: PASS, 6 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
