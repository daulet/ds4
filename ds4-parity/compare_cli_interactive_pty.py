#!/usr/bin/env python3
"""Compare the Rust interactive CLI PTY surface against the M8.14 oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import check_cli_interactive_dump as oracle


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.14" / "current-c.json"
DEFAULT_CANDIDATE = ROOT / "target" / "debug" / "ds4-cli-interactive-rs"
READ_PROMPT = "ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt"
NEXT_PROMPT = "Answer with one short noun: glacier."
TIMING = "ds4: prefill: <rate> t/s, generation: <rate> t/s"
FORBIDDEN_TRANSCRIPT = (
    "perplexity",
    "imatrix",
    "--dump-logprobs",
    "diagnostic run completed",
    "M8.13",
    "M8.15a",
    "M8.15c interactive implementation supports",
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def cases(obj: Any) -> list[dict[str, Any]]:
    root = obj if isinstance(obj, dict) else {}
    raw_cases = root.get("cases") if isinstance(root.get("cases"), list) else []
    return [case for case in raw_cases if isinstance(case, dict)]


def case_by_id(obj: Any, case_id: str) -> dict[str, Any]:
    for case in cases(obj):
        if case.get("id") == case_id:
            return case
    raise KeyError(case_id)


def capture_candidate(binary: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_interactive_rust_pty.v1",
        "binary": str(binary),
        "cases": [oracle.capture_case(binary, case) for case in oracle.CASES],
        "normalization": {
            "redraw": "linenoise per-keystroke redraw frames are ignored; committed prompts and responses are compared",
            "timing": "startup seconds and prefill/generation rates are normalized by the M8.14 oracle helper",
        },
    }


def extract_turn(text: str, marker: str) -> bytes:
    if marker not in text:
        raise ValueError(f"missing marker {marker!r}")
    segment = text.split(marker, 1)[1]
    if TIMING not in segment:
        raise ValueError(f"missing timing after {marker!r}")
    segment = segment.split(TIMING, 1)[0]
    lines: list[str] = []
    for line in segment.splitlines():
        if not line or line.startswith("processing ") or line.startswith("ds4> "):
            continue
        lines.append(line)
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def expected_turns(baseline: Any) -> dict[str, bytes]:
    text = str(case_by_id(baseline, "command_suite").get("normalized_transcript", ""))
    return {
        "read": extract_turn(text, f"ds4> /read {READ_PROMPT}\n"),
        "direct": extract_turn(text, f"ds4> {NEXT_PROMPT}\n"),
    }


def normalized(case: dict[str, Any]) -> str:
    return str(case.get("normalized_transcript", ""))


def compare_case(report: Report, expected_case: dict[str, Any], candidate_case: dict[str, Any]) -> None:
    case_id = str(expected_case.get("id"))
    text = normalized(candidate_case)
    expected_script = expected_case.get("script")
    report.check(candidate_case.get("id") == case_id, f"{case_id}: id drift")
    report.check(candidate_case.get("argv") == expected_case.get("argv"), f"{case_id}: argv drift")
    report.check(candidate_case.get("script") == expected_script, f"{case_id}: script drift")
    report.check(candidate_case.get("exit_code") == expected_case.get("expected_exit_code"), f"{case_id}: exit code drift")
    report.check("Commands:" in text, f"{case_id}: missing startup help")
    report.check("ds4> " in text, f"{case_id}: missing prompt marker")
    report.check("backend=cuda" in text, f"{case_id}: missing CUDA backend anchor")

    if isinstance(expected_script, list):
        for step in expected_script:
            if not isinstance(step, str) or step in {"", oracle.CTRL_C}:
                continue
            report.check(f"ds4> {step}" in text, f"{case_id}: missing committed prompt {step!r}")

    for forbidden in FORBIDDEN_TRANSCRIPT:
        report.check(forbidden not in text, f"{case_id}: forbidden transcript marker {forbidden!r}")

    if case_id == "command_suite":
        for anchor in (
            "/help          Show this help.",
            "Thinking mode: high.",
            "ds4: warning: /think-max needs --ctx >= 393216; ctx=128 uses normal thinking instead",
            "Thinking mode: high (ctx below 393216).",
            "Thinking mode: none.",
            "ds4: context buffers",
            "ds4: unknown command: /definitely-unknown",
            "ds4: type /help for commands",
            "processing 12 input tokens",
        ):
            report.check(anchor in text, f"{case_id}: missing anchor {anchor!r}")
        report.check(text.count(TIMING) == 2, f"{case_id}: timing count drift")
    elif case_id == "ctrl_c_at_prompt":
        report.check(text.count(TIMING) == 0, f"{case_id}: unexpected timing output")


def compare_dump(baseline: Any, candidate: Any) -> Report:
    report = Report()
    preconditions = oracle.check_dump(baseline)
    report.check(preconditions.ok, "current-C interactive oracle preconditions failed")
    for error in preconditions.errors:
        report.check(False, f"oracle precondition: {error}")

    candidate_cases = {case.get("id"): case for case in cases(candidate)}
    expected_cases = {case.get("id"): case for case in cases(baseline)}
    report.check(set(candidate_cases) == set(expected_cases), "case id set drift")
    for case_id, expected_case in expected_cases.items():
        candidate_case = candidate_cases.get(case_id)
        report.check(isinstance(candidate_case, dict), f"{case_id}: missing candidate case")
        if isinstance(candidate_case, dict):
            compare_case(report, expected_case, candidate_case)

    expected = expected_turns(baseline)
    command_suite = candidate_cases.get("command_suite", {})
    if isinstance(command_suite, dict):
        text = normalized(command_suite)
        for label, marker in (
            ("read", f"ds4> /read {READ_PROMPT}\n"),
            ("direct", f"ds4> {NEXT_PROMPT}\n"),
        ):
            try:
                actual = extract_turn(text, marker)
            except ValueError as exc:
                report.check(False, str(exc))
                actual = b""
            report.check(actual == expected[label], f"{label} generated bytes drift")
            report.check(sha256_bytes(actual) == sha256_bytes(expected[label]), f"{label} generated sha drift")

    for case in candidate_cases.values():
        if isinstance(case, dict):
            text = normalized(case).encode("utf-8")
            report.check(
                case.get("normalized_transcript_sha256") == sha256_bytes(text),
                f"{case.get('id')}: normalized transcript sha drift",
            )
    return report


def run_negative_tests(baseline: Any, candidate: Any) -> Report:
    report = Report()

    def expect_failure(name: str, mutate: Callable[[Any], None]) -> None:
        mutated = copy.deepcopy(candidate)
        mutate(mutated)
        result = compare_dump(baseline, mutated)
        report.check(not result.ok, f"negative test did not fail: {name}")

    expect_failure(
        "read generated byte drift",
        lambda root: case_by_id(root, "command_suite").__setitem__(
            "normalized_transcript",
            case_by_id(root, "command_suite")["normalized_transcript"].replace("Gl\n", "X\n", 1),
        ),
    )
    expect_failure(
        "direct generated byte drift",
        lambda root: case_by_id(root, "command_suite").__setitem__(
            "normalized_transcript",
            case_by_id(root, "command_suite")["normalized_transcript"].replace("Ice\n", "X\n", 1),
        ),
    )
    expect_failure(
        "unknown command drift",
        lambda root: case_by_id(root, "command_suite").__setitem__(
            "normalized_transcript",
            case_by_id(root, "command_suite")["normalized_transcript"].replace(
                "ds4: unknown command: /definitely-unknown\n", "", 1
            ),
        ),
    )
    expect_failure("exit code drift", lambda root: case_by_id(root, "ctrl_c_at_prompt").__setitem__("exit_code", 9))
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--candidate-binary", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--write-candidate", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    if args.candidate_artifact:
        candidate = load_json(args.candidate_artifact)
    else:
        candidate = capture_candidate(args.candidate_binary)
    if args.write_candidate:
        write_json(args.write_candidate, candidate)

    report = compare_dump(baseline, candidate)
    print_report("CLI interactive PTY comparator", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(baseline, candidate)
        print_report("CLI interactive PTY negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
