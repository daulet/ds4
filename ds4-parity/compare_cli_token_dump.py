#!/usr/bin/env python3
"""Compare the Rust CLI --dump-tokens path against the M8.4 current-C oracle."""

from __future__ import annotations

import argparse
import ast
import base64
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
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.4" / "current-c.json"
RUST_BIN = ROOT / "target" / "debug" / "ds4-cli-token-dump-rs"
TOKENIZER_FIXTURE = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.3" / "tokenizer.gguf"
EXPECTED_TOKENIZER_SHA256 = "b1e0d128bde9ea996fee335c9662e93707d2a68decaeb47a8dc5fb902bdbb025"
EXPECTED_TOKENIZER_SIZE = 4722720


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def decode_b64(report: Report, value: Any, path: str) -> bytes:
    report.check(isinstance(value, str), f"{path}: expected base64 string")
    if not isinstance(value, str):
        return b""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        report.check(False, f"{path}: invalid base64: {exc}")
        return b""


def parse_token_ids(stdout: bytes) -> list[int]:
    first_line = stdout.split(b"\n", 1)[0]
    parsed = ast.literal_eval(first_line.decode("ascii"))
    if not isinstance(parsed, list) or not all(isinstance(item, int) and item >= 0 for item in parsed):
        raise ValueError(f"invalid token id line: {first_line!r}")
    return parsed


def token_ids_sha256(token_ids: list[int]) -> str:
    blob = bytearray()
    for token_id in token_ids:
        blob.extend(f"{token_id}\n".encode("ascii"))
    return sha256_bytes(bytes(blob))


def check_tokenizer_fixture(path: Path) -> Report:
    report = Report()
    report.check(path.is_file(), f"missing tokenizer fixture: {path}")
    if path.is_file():
        report.check(path.stat().st_size == EXPECTED_TOKENIZER_SIZE, "tokenizer fixture size drift")
        report.check(sha256_file(path) == EXPECTED_TOKENIZER_SHA256, "tokenizer fixture sha drift")
    return report


def check_baseline(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "baseline")
    report.check(root.get("schema") == "ds4.cli_token_dump_oracle.v1", "baseline schema mismatch")
    report.check(root.get("source") == "current-c-cli-token-dump", "baseline source mismatch")
    cases = require_list(report, root.get("cases"), "baseline.cases")
    seen: set[str] = set()
    for idx, raw in enumerate(cases):
        case = require_dict(report, raw, f"baseline.cases[{idx}]")
        case_id = case.get("id")
        report.check(isinstance(case_id, str) and bool(case_id), f"baseline.cases[{idx}].id invalid")
        if isinstance(case_id, str):
            report.check(case_id not in seen, f"duplicate case id {case_id}")
            seen.add(case_id)
        report.check(case.get("exit_code") == 0, f"{case_id}.exit_code must be 0")
        report.check(case.get("warning_category") == "none", f"{case_id}.warning_category drift")
        argv = require_list(report, case.get("argv"), f"{case_id}.argv")
        report.check("--dump-tokens" in argv, f"{case_id}.argv missing --dump-tokens")
        stdout = decode_b64(report, case.get("stdout_base64"), f"{case_id}.stdout_base64")
        stderr = decode_b64(report, case.get("stderr_base64"), f"{case_id}.stderr_base64")
        report.check(case.get("stdout_bytes") == len(stdout), f"{case_id}.stdout_bytes drift")
        report.check(case.get("stderr_bytes") == len(stderr), f"{case_id}.stderr_bytes drift")
        report.check(case.get("stdout_sha256") == sha256_bytes(stdout), f"{case_id}.stdout_sha256 drift")
        report.check(case.get("stderr_sha256") == sha256_bytes(stderr), f"{case_id}.stderr_sha256 drift")
        report.check(stderr == b"", f"{case_id}.stderr must be empty")
        try:
            parsed_ids = parse_token_ids(stdout)
        except Exception as exc:
            report.check(False, f"{case_id}.stdout token id line invalid: {exc}")
            parsed_ids = []
        token_ids = require_list(report, case.get("token_ids"), f"{case_id}.token_ids")
        report.check(token_ids == parsed_ids, f"{case_id}.token_ids drift")
        report.check(case.get("token_count") == len(parsed_ids), f"{case_id}.token_count drift")
        report.check(case.get("token_ids_sha256") == token_ids_sha256(parsed_ids), f"{case_id}.token_ids_sha256 drift")
    return report


def build_rust_binary() -> None:
    proc = subprocess.run(
        ["cargo", "build", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-cli-token-dump-rs"],
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


def rust_argv(c_argv: list[str], tokenizer_fixture: Path) -> list[str]:
    argv = list(c_argv)
    idx = 0
    while idx < len(argv):
        if argv[idx] in {"-m", "--model"}:
            if idx + 1 >= len(argv):
                raise ValueError(f"{argv[idx]} missing value")
            argv[idx + 1] = str(tokenizer_fixture)
            return argv
        idx += 1
    return [*argv, "-m", str(tokenizer_fixture)]


def run_rust(argv: list[str], rust_bin: Path) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    return subprocess.run(
        [str(rust_bin), *argv],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )


def compare_case(report: Report, case: dict[str, Any], rust_bin: Path, tokenizer_fixture: Path) -> None:
    case_id = str(case.get("id"))
    argv = case.get("argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        report.check(False, f"{case_id}.argv invalid")
        return
    try:
        rust_args = rust_argv(argv, tokenizer_fixture)
    except ValueError as exc:
        report.check(False, f"{case_id}.argv invalid: {exc}")
        return
    proc = run_rust(rust_args, rust_bin)
    c_stdout = decode_b64(report, case.get("stdout_base64"), f"{case_id}.stdout_base64")
    c_stderr = decode_b64(report, case.get("stderr_base64"), f"{case_id}.stderr_base64")
    report.check(proc.returncode == case.get("exit_code"), f"{case_id}.exit_code drift: expected {case.get('exit_code')} got {proc.returncode}")
    report.check(proc.stdout == c_stdout, f"{case_id}.stdout bytes drift")
    report.check(proc.stderr == c_stderr, f"{case_id}.stderr bytes drift")
    if proc.stdout == c_stdout:
        try:
            parsed_ids = parse_token_ids(proc.stdout)
        except Exception as exc:
            report.check(False, f"{case_id}.rust stdout token ids invalid: {exc}")
        else:
            report.check(parsed_ids == case.get("token_ids"), f"{case_id}.rust token ids drift")


def compare_rust(obj: Any, rust_bin: Path, tokenizer_fixture: Path) -> Report:
    report = Report()
    root = require_dict(report, obj, "baseline")
    cases = require_list(report, root.get("cases"), "baseline.cases")
    for idx, raw in enumerate(cases):
        case = require_dict(report, raw, f"baseline.cases[{idx}]")
        compare_case(report, case, rust_bin, tokenizer_fixture)
    return report


def run_negative_tests(obj: Any, rust_bin: Path, tokenizer_fixture: Path) -> Report:
    report = Report()

    def expect_failure(name: str, mutation: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in mutation[:-1]:
            target = target[part]
        target[mutation[-1]] = value
        baseline = check_baseline(candidate)
        rust = compare_rust(candidate, rust_bin, tokenizer_fixture)
        report.check(not (baseline.ok and rust.ok), f"negative test did not fail: {name}")

    expect_failure("baseline stdout sha drift", ["cases", 0, "stdout_sha256"], "0" * 64)
    expect_failure("baseline token id drift", ["cases", 0, "token_ids", 0], 99999999)
    expect_failure("rust exit mismatch", ["cases", 0, "exit_code"], 2)
    expect_failure("rust stdout mismatch", ["cases", 0, "stdout_base64"], base64.b64encode(b"bad\n").decode("ascii"))
    expect_failure("rust argv prompt mismatch", ["cases", 0, "argv", 4], "different prompt")
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
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_FIXTURE)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        build_rust_binary()
    obj = load_json(args.baseline)

    tokenizer_report = check_tokenizer_fixture(args.tokenizer)
    print_report("CLI token dump tokenizer fixture", tokenizer_report)
    baseline_report = check_baseline(obj)
    print_report("CLI token dump C fixture preconditions", baseline_report)
    rust_report = compare_rust(obj, args.rust_bin, args.tokenizer)
    print_report("CLI token dump C/Rust comparator", rust_report)
    ok = tokenizer_report.ok and baseline_report.ok and rust_report.ok
    if args.negative_test:
        negative_report = run_negative_tests(obj, args.rust_bin, args.tokenizer)
        print_report("CLI token dump negative tests", negative_report)
        ok = ok and negative_report.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
