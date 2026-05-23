#!/usr/bin/env python3
"""Compare Rust inspect runtime output against the M8.8 current-C oracle."""

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

import check_cli_inspect_dump as oracle


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.8" / "current-c.json"
DEFAULT_CANDIDATE = ROOT / "target" / "debug" / "ds4-inspect-runtime-rs"
FORBIDDEN_STDERR = (
    b"ds4: context buffers",
    b"think-max",
    b"input tokens:",
    b"decode failed",
    b"perplexity",
    b"imatrix",
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


def case_backend(argv: list[Any]) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--backend" and idx + 1 < len(argv):
            value = argv[idx + 1]
            if isinstance(value, str):
                return value
        if arg == "--cuda":
            return "cuda"
        if arg == "--metal":
            return "metal"
        if arg == "--cpu":
            return "cpu"
    raise ValueError(f"case argv has no backend: {argv!r}")


def run_candidate(
    binary: Path,
    model_path: str,
    raw_case: dict[str, Any],
    use_case_argv: bool,
) -> CandidateOutput:
    if use_case_argv:
        raw_argv = require_list(Report(), raw_case.get("argv"), f"{raw_case.get('id')}.argv")
        argv = tuple([str(binary), *(str(arg) for arg in raw_argv)])
    else:
        backend = case_backend(require_list(Report(), raw_case.get("argv"), f"{raw_case.get('id')}.argv"))
        argv = (
            str(binary),
            "--backend",
            backend,
            "--model",
            model_path,
            "--inspect",
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
    return CandidateOutput(
        case_id=str(raw_case.get("id")),
        argv=argv,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def capture_candidates(binary: Path, obj: Any, use_case_argv: bool) -> dict[str, CandidateOutput]:
    root = obj if isinstance(obj, dict) else {}
    model = root.get("model") if isinstance(root.get("model"), dict) else {}
    model_path = model.get("path")
    if not isinstance(model_path, str):
        raise SystemExit("baseline model.path is missing")
    if not binary.is_file():
        raise SystemExit(f"missing candidate binary: {binary}")
    outputs: dict[str, CandidateOutput] = {}
    cases = root.get("cases") if isinstance(root.get("cases"), list) else []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            continue
        output = run_candidate(binary, model_path, raw_case, use_case_argv)
        outputs[output.case_id] = output
    return outputs


def check_candidate_case(
    report: Report,
    raw_case: Any,
    output: CandidateOutput | None,
    refs: dict[str, dict[str, Any]],
    label: str,
) -> None:
    case = require_dict(report, raw_case, label)
    case_id = case.get("id")
    report.check(output is not None, f"{label}.candidate missing")
    if output is None:
        return
    report.check(output.case_id == case_id, f"{case_id}.candidate id drift")
    report.check(output.exit_code == case.get("exit_code"), f"{case_id}.exit_code drift")
    stdout_obj = require_dict(report, case.get("stdout"), f"{case_id}.stdout")
    expected_stdout = unb64(report, stdout_obj.get("base64"), f"{case_id}.stdout.base64")
    report.check(output.stdout == expected_stdout, f"{case_id}.stdout bytes drift")
    report.check(stdout_obj.get("sha256") == sha256_bytes(output.stdout), f"{case_id}.stdout sha drift")
    summary = oracle.parse_summary(output.stdout)
    report.check(summary == case.get("summary"), f"{case_id}.summary drift")
    for anchor in require_list(report, case.get("stdout_anchors"), f"{case_id}.stdout_anchors"):
        report.check(isinstance(anchor, str), f"{case_id}.stdout anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor.encode("utf-8") in output.stdout, f"{case_id}.stdout missing anchor {anchor!r}")
    for anchor in require_list(report, case.get("stderr_anchors"), f"{case_id}.stderr_anchors"):
        report.check(isinstance(anchor, str), f"{case_id}.stderr anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor.encode("utf-8") in output.stderr, f"{case_id}.stderr missing anchor {anchor!r}")
    for forbidden in FORBIDDEN_STDERR:
        report.check(forbidden not in output.stderr, f"{case_id}.stderr entered unexpected path {forbidden!r}")
    reference_id = case.get("same_stdout_as")
    if reference_id is not None:
        ref = refs.get(reference_id)
        report.check(ref is not None, f"{case_id}.same_stdout_as reference missing")
        if ref is not None:
            ref_stdout = require_dict(report, ref.get("stdout"), f"{case_id}.reference.stdout")
            report.check(stdout_obj.get("sha256") == ref_stdout.get("sha256"), f"{case_id}.reference stdout drift")


def compare_dump(obj: Any, outputs: dict[str, CandidateOutput]) -> Report:
    report = Report()
    preconditions = oracle.check_dump(obj)
    report.check(preconditions.ok, "current-C inspect oracle preconditions failed")
    for error in preconditions.errors:
        report.check(False, f"oracle precondition: {error}")
    root = require_dict(report, obj, "root")
    cases = require_list(report, root.get("cases"), "cases")
    refs = {case["id"]: case for case in cases if isinstance(case, dict) and isinstance(case.get("id"), str)}
    for idx, raw_case in enumerate(cases):
        case_id = raw_case.get("id") if isinstance(raw_case, dict) else None
        output = outputs.get(case_id) if isinstance(case_id, str) else None
        check_candidate_case(report, raw_case, output, refs, f"cases[{idx}]")
    return report


def run_negative_tests(obj: Any, outputs: dict[str, CandidateOutput]) -> Report:
    report = Report()

    def expect_failure(name: str, path: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        sub = compare_dump(candidate, outputs)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("summary drift", ["cases", 0, "summary", "model"], "Wrong Model")
    expect_failure("stdout drift", ["cases", 0, "stdout", "base64"], b64(b"wrong\n"))
    expect_failure("stderr anchor drift", ["cases", 0, "stderr_anchors"], [])
    expect_failure("exit code drift", ["cases", 0, "exit_code"], 99)
    expect_failure("reference stdout drift", ["cases", 1, "same_stdout_as"], None)
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
    parser.add_argument("--use-case-argv", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    obj = load_json(args.artifact)
    outputs = capture_candidates(args.candidate_binary, obj, args.use_case_argv)
    report = compare_dump(obj, outputs)
    label = "CLI inspect comparator" if args.use_case_argv else "CLI inspect runtime comparator"
    print_report(label, report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(obj, outputs)
        print_report("CLI inspect runtime negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
