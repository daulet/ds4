#!/usr/bin/env python3
"""Compare Rust reusable interactive runtime output against the M8.14 oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import check_cli_interactive_dump as oracle


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.14" / "current-c.json"
DEFAULT_CANDIDATE = ROOT / "target" / "debug" / "ds4-interactive-runtime-rs"
READ_PROMPT = "ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt"
NEXT_PROMPT = "Answer with one short noun: glacier."
B300_MODEL = "/workspace/ds4/ds4flash.gguf"
TURN_RE = re.compile(
    rb"<<<ds4-rs-turn:(?P<label>[a-z]+)>>>\n(?P<body>.*?)<<<ds4-rs-end:(?P=label)>>>\n",
    re.DOTALL,
)
FORBIDDEN_STDERR = (
    b"ds4>",
    b"perplexity",
    b"imatrix",
    b"--dump-logprobs",
    b"diagnostic run completed",
    b"M8.13",
    b"M8.15c",
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


@dataclass(frozen=True)
class CandidateOutput:
    argv: tuple[str, ...]
    exit_code: int
    stdout: bytes
    stderr: bytes


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def cases(obj: Any) -> list[dict[str, Any]]:
    root = obj if isinstance(obj, dict) else {}
    raw_cases = root.get("cases") if isinstance(root.get("cases"), list) else []
    return [case for case in raw_cases if isinstance(case, dict)]


def command_suite(obj: Any) -> dict[str, Any]:
    for case in cases(obj):
        if case.get("id") == "command_suite":
            return case
    raise KeyError("command_suite")


def extract_expected_turns(obj: Any) -> dict[str, bytes]:
    text = str(command_suite(obj).get("normalized_transcript", ""))
    read_marker = f"ds4> /read {READ_PROMPT}\n\n"
    if read_marker not in text:
        raise SystemExit("M8.14 baseline missing /read marker")
    read_segment = text.split(read_marker, 1)[1].split(
        "\nds4: prefill: <rate> t/s, generation: <rate> t/s",
        1,
    )[0]
    if read_segment and not read_segment.endswith("\n"):
        read_segment += "\n"

    direct_marker = f"ds4> {NEXT_PROMPT}\n\n"
    if direct_marker not in text:
        raise SystemExit("M8.14 baseline missing direct prompt marker")
    direct_segment = text.split(direct_marker, 1)[1].split(
        "\nds4: prefill: <rate> t/s, generation: <rate> t/s",
        1,
    )[0]
    direct_lines = [
        line
        for line in direct_segment.splitlines()
        if line and not line.startswith("processing ")
    ]
    return {
        "read": read_segment.encode("utf-8"),
        "direct": ("\n".join(direct_lines) + "\n").encode("utf-8"),
    }


def parse_turns(stdout: bytes) -> dict[str, bytes]:
    turns: dict[str, bytes] = {}
    for match in TURN_RE.finditer(stdout):
        turns[match.group("label").decode("ascii")] = match.group("body")
    return turns


def run_candidate(binary: Path) -> CandidateOutput:
    if not binary.is_file():
        raise SystemExit(f"missing candidate binary: {binary}")
    argv = (
        str(binary),
        "--cuda",
        "-m",
        B300_MODEL,
        "--ctx",
        "128",
        "--tokens",
        "1",
        "--temp",
        "0",
        "--nothink",
        "--read-prompt-file",
        READ_PROMPT,
        "--next-prompt",
        NEXT_PROMPT,
    )
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    proc = subprocess.run(
        list(argv),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return CandidateOutput(argv=argv, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def compare_dump(obj: Any, output: CandidateOutput) -> Report:
    report = Report()
    preconditions = oracle.check_dump(obj)
    report.check(preconditions.ok, "current-C interactive oracle preconditions failed")
    for error in preconditions.errors:
        report.check(False, f"oracle precondition: {error}")

    expected = extract_expected_turns(obj)
    actual = parse_turns(output.stdout)
    report.check(output.exit_code == 0, "candidate exit code drift")
    report.check(set(actual) == {"read", "direct"}, f"candidate turn labels drift: {sorted(actual)}")
    for label, expected_bytes in expected.items():
        actual_bytes = actual.get(label, b"")
        report.check(actual_bytes == expected_bytes, f"{label} turn bytes drift")
        report.check(sha256_bytes(actual_bytes) == sha256_bytes(expected_bytes), f"{label} turn sha drift")

    stderr_text = output.stderr.decode("utf-8", errors="replace")
    normalized = oracle.normalize_transcript(output.stderr)
    for anchor in (
        "ds4: context buffers",
        "backend=cuda",
        "processing 12 input tokens",
        "ds4: prefill: <rate> t/s, generation: <rate> t/s",
    ):
        report.check(anchor in normalized, f"candidate stderr missing anchor {anchor!r}")
    report.check(stderr_text.count("ds4: prefill:") == 2, "candidate timing count drift")
    for forbidden in FORBIDDEN_STDERR:
        report.check(forbidden not in output.stderr, f"candidate stderr entered forbidden path {forbidden!r}")
    return report


def run_negative_tests(obj: Any, output: CandidateOutput) -> Report:
    report = Report()

    def expect_failure(name: str, mutate_obj: Callable[[Any], None] | None = None, output_override: CandidateOutput | None = None) -> None:
        candidate = copy.deepcopy(obj)
        if mutate_obj is not None:
            mutate_obj(candidate)
        try:
            sub = compare_dump(candidate, output_override or output)
            failed = not sub.ok
        except SystemExit:
            failed = True
        report.check(failed, f"negative test did not fail: {name}")

    expect_failure(
        "read turn drift",
        output_override=CandidateOutput(output.argv, output.exit_code, output.stdout.replace(b"Gl\n", b"X\n", 1), output.stderr),
    )
    expect_failure(
        "direct turn drift",
        output_override=CandidateOutput(output.argv, output.exit_code, output.stdout.replace(b"Ice\n", b"X\n", 1), output.stderr),
    )
    expect_failure(
        "timing anchor drift",
        output_override=CandidateOutput(output.argv, output.exit_code, output.stdout, b"missing timing\n"),
    )
    expect_failure("baseline prompt drift", lambda root: command_suite(root).__setitem__("normalized_transcript", "wrong\n"))
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--candidate-binary", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    obj = load_json(args.artifact)
    output = run_candidate(args.candidate_binary)
    report = compare_dump(obj, output)
    print_report("CLI interactive runtime comparator", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(obj, output)
        print_report("CLI interactive runtime negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
