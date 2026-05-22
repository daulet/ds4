#!/usr/bin/env python3
"""Compare the Rust CLI parser surface against the M8.2 current-C oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.2" / "current-c.json"
RUST_BIN = ROOT / "target" / "debug" / "ds4-cli-parse-rs"

HELP_ANCHORS = (
    "Usage: ds4",
    "Invocation modes:",
    "Model and runtime:",
    "Prompt and generation:",
    "Interactive commands:",
    "Diagnostics:",
    "--dump-logprobs FILE",
    "--imatrix-dataset FILE",
    "--head-test",
    "Normal CLI commands:",
    "-h, --help",
)

NO_MODEL_MARKERS = (
    "failed to open model",
    "context buffers",
    "Metal device",
    "CUDA backend",
    "loading model",
    "parser-only implementation reached model-backed path",
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

    def merge(self, other: "Report") -> None:
        self.checks += other.checks
        self.errors.extend(other.errors)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def check_baseline(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "baseline")
    report.check(root.get("schema") == "ds4.cli_parse_oracle.v1", "baseline schema mismatch")
    report.check(root.get("source") == "current-c-cli-parse", "baseline source mismatch")
    cases = require_list(report, root.get("cases"), "baseline.cases")
    seen: set[str] = set()
    for idx, raw in enumerate(cases):
        case = require_dict(report, raw, f"baseline.cases[{idx}]")
        case_id = case.get("id")
        report.check(isinstance(case_id, str) and bool(case_id), f"baseline.cases[{idx}].id invalid")
        if isinstance(case_id, str):
            report.check(case_id not in seen, f"duplicate case id {case_id}")
            seen.add(case_id)
        report.check(isinstance(case.get("argv"), list), f"{case_id}.argv invalid")
        report.check(isinstance(case.get("exit_code"), int), f"{case_id}.exit_code invalid")
        stdout = case.get("stdout")
        stderr = case.get("stderr")
        report.check(isinstance(stdout, str), f"{case_id}.stdout invalid")
        report.check(isinstance(stderr, str), f"{case_id}.stderr invalid")
        if isinstance(stdout, str):
            report.check(case.get("stdout_bytes") == len(stdout.encode("utf-8")), f"{case_id}.stdout_bytes drift")
            report.check(case.get("stdout_sha256") == sha256_text(stdout), f"{case_id}.stdout_sha256 drift")
        if isinstance(stderr, str):
            report.check(case.get("stderr_bytes") == len(stderr.encode("utf-8")), f"{case_id}.stderr_bytes drift")
            report.check(case.get("stderr_sha256") == sha256_text(stderr), f"{case_id}.stderr_sha256 drift")
    return report


def build_rust_binary() -> None:
    proc = subprocess.run(
        ["cargo", "build", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-cli-parse-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(proc.returncode)


def run_rust(argv: list[str], rust_bin: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    return subprocess.run(
        [str(rust_bin), *argv],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def error_anchors(stderr: str) -> list[str]:
    if not stderr:
        return []
    anchors = [stderr.splitlines()[0]]
    if "Usage: ds4" in stderr:
        anchors.append("Usage: ds4")
    if "ds4: valid backends are:" in stderr:
        anchors.append("ds4: valid backends are: metal, cuda, cpu")
    return anchors


def compare_case(report: Report, case: dict[str, Any], rust_bin: Path) -> None:
    case_id = str(case.get("id"))
    argv = case.get("argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        report.check(False, f"{case_id}.argv invalid")
        return
    proc = run_rust(argv, rust_bin)
    report.check(proc.returncode == case.get("exit_code"), f"{case_id}.exit_code drift: expected {case.get('exit_code')} got {proc.returncode}")
    c_stdout = case.get("stdout") if isinstance(case.get("stdout"), str) else ""
    c_stderr = case.get("stderr") if isinstance(case.get("stderr"), str) else ""
    report.check(bool(proc.stdout) == bool(c_stdout), f"{case_id}.stdout emptiness drift")
    report.check(bool(proc.stderr) == bool(c_stderr), f"{case_id}.stderr emptiness drift")
    if c_stdout:
        for anchor in HELP_ANCHORS:
            report.check(anchor in proc.stdout, f"{case_id}.stdout missing help anchor {anchor!r}")
    if c_stderr:
        for anchor in error_anchors(c_stderr):
            report.check(anchor in proc.stderr, f"{case_id}.stderr missing anchor {anchor!r}")
    combined = proc.stdout + proc.stderr
    for marker in NO_MODEL_MARKERS:
        report.check(marker not in combined, f"{case_id}: unexpected model-load marker {marker!r}")


def compare_rust(obj: Any, rust_bin: Path) -> Report:
    report = Report()
    root = require_dict(report, obj, "baseline")
    cases = require_list(report, root.get("cases"), "baseline.cases")
    for raw in cases:
        case = require_dict(report, raw, "case")
        compare_case(report, case, rust_bin)
    return report


def run_negative_tests(obj: Any, rust_bin: Path) -> Report:
    report = Report()

    def expect_failure(name: str, mutation: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in mutation[:-1]:
            target = target[part]
        target[mutation[-1]] = value
        baseline = check_baseline(candidate)
        rust = compare_rust(candidate, rust_bin)
        report.check(not (baseline.ok and rust.ok), f"negative test did not fail: {name}")

    expect_failure("baseline stdout sha drift", ["cases", 0, "stdout_sha256"], "0" * 64)
    expect_failure("rust exit mismatch", ["cases", 2, "exit_code"], 0)
    expect_failure("rust stdout mismatch", ["cases", 0, "stdout"], "")
    expect_failure("rust stderr anchor mismatch", ["cases", 5, "stderr"], "ds4: invalid backend: vulkan\n")
    expect_failure("case argv mismatch", ["cases", 4, "argv"], ["--different"])
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--rust-bin", type=Path, default=RUST_BIN)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        build_rust_binary()
    obj = load_json(args.baseline)
    baseline_report = check_baseline(obj)
    print_report("CLI parse C fixture preconditions", baseline_report)
    rust_report = compare_rust(obj, args.rust_bin)
    print_report("CLI parse C/Rust comparator", rust_report)
    ok = baseline_report.ok and rust_report.ok
    if args.negative_test:
        negative_report = run_negative_tests(obj, args.rust_bin)
        print_report("CLI parse negative tests", negative_report)
        ok = ok and negative_report.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
