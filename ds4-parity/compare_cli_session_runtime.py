#!/usr/bin/env python3
"""Compare Rust session runtime output against the M8.12a seeded current-C case."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import check_cli_generation_dump as oracle


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.12a" / "current-c.json"
DEFAULT_CANDIDATE = ROOT / "target" / "debug" / "ds4-session-runtime-rs"
SESSION_CASE_ID = "seeded_sampling_nothink"
FORBIDDEN_STDERR = (
    b"ds4>",
    b"perplexity",
    b"imatrix",
    b"--dump-logprobs",
    b"diagnostic run completed",
    b"M8.9b inspect implementation",
    b"M8.13a argmax runtime",
    b"requires --seed",
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
    case_id: str
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


def unb64(report: Report, value: Any, label: str) -> bytes:
    report.check(isinstance(value, str), f"{label}: expected base64 string")
    if not isinstance(value, str):
        return b""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        report.check(False, f"{label}: invalid base64: {exc}")
        return b""


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, label: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{label}: expected array")
    return obj if isinstance(obj, list) else []


def session_case(obj: Any) -> dict[str, Any] | None:
    root = obj if isinstance(obj, dict) else {}
    cases = root.get("cases") if isinstance(root.get("cases"), list) else []
    for case in cases:
        if isinstance(case, dict) and case.get("id") == SESSION_CASE_ID:
            return case
    return None


def run_candidate(binary: Path, raw_case: dict[str, Any]) -> CandidateOutput:
    argv_obj = raw_case.get("argv")
    if not isinstance(argv_obj, list):
        raise SystemExit(f"{SESSION_CASE_ID}: baseline argv is missing")
    argv = tuple([str(binary), *(str(arg) for arg in argv_obj)])
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
    return CandidateOutput(
        case_id=SESSION_CASE_ID,
        argv=argv,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def capture_candidate(binary: Path, obj: Any) -> CandidateOutput:
    if not binary.is_file():
        raise SystemExit(f"missing candidate binary: {binary}")
    case = session_case(obj)
    if case is None:
        raise SystemExit(f"missing {SESSION_CASE_ID} in baseline")
    return run_candidate(binary, case)


def compare_dump(obj: Any, output: CandidateOutput) -> Report:
    report = Report()
    preconditions = oracle.check_dump(obj)
    report.check(preconditions.ok, "current-C generation oracle preconditions failed")
    for error in preconditions.errors:
        report.check(False, f"oracle precondition: {error}")

    case = require_dict(report, session_case(obj), SESSION_CASE_ID)
    report.check(case.get("id") == SESSION_CASE_ID, "session case id drift")
    report.check(case.get("seed") == 12345, "session seed drift")
    report.check(output.case_id == case.get("id"), "candidate id drift")
    report.check(output.exit_code == case.get("exit_code"), "exit code drift")

    stdout_obj = require_dict(report, case.get("stdout"), "session.stdout")
    expected_stdout = unb64(report, stdout_obj.get("base64"), "session.stdout.base64")
    report.check(output.stdout == expected_stdout, "stdout bytes drift")
    report.check(stdout_obj.get("sha256") == sha256_bytes(output.stdout), "stdout sha drift")
    report.check(stdout_obj.get("bytes") == len(output.stdout), "stdout size drift")
    report.check((len(output.stdout) == 0) == bool(case.get("stdout_empty")), "stdout empty drift")

    stderr_text = output.stderr.decode("utf-8", errors="replace")
    normalized = oracle.normalize_stderr(output.stderr)
    for anchor in require_list(report, case.get("stderr_anchors"), "session.stderr_anchors"):
        report.check(isinstance(anchor, str), "stderr anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor in stderr_text, f"stderr missing anchor {anchor!r}")
    for anchor in require_list(
        report,
        case.get("normalized_stderr_anchors"),
        "session.normalized_stderr_anchors",
    ):
        report.check(isinstance(anchor, str), "normalized stderr anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor in normalized, f"normalized stderr missing anchor {anchor!r}")
    for forbidden in FORBIDDEN_STDERR:
        report.check(forbidden not in output.stderr, f"stderr entered forbidden path {forbidden!r}")
    return report


def run_negative_tests(obj: Any, output: CandidateOutput) -> Report:
    report = Report()

    def expect_failure(name: str, path: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        sub = compare_dump(candidate, output)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("stdout bytes drift", ["cases", 3, "stdout", "base64"], b64(b"wrong\n"))
    expect_failure("stdout hash drift", ["cases", 3, "stdout", "sha256"], "0" * 64)
    expect_failure("seed drift", ["cases", 3, "seed"], 54321)
    expect_failure("exit code drift", ["cases", 3, "exit_code"], 99)
    expect_failure(
        "normalized stderr anchor drift",
        ["cases", 3, "normalized_stderr_anchors"],
        ["missing normalized anchor"],
    )
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
    output = capture_candidate(args.candidate_binary, obj)
    report = compare_dump(obj, output)
    print_report("CLI session runtime comparator", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(obj, output)
        print_report("CLI session runtime negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
