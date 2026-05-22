#!/usr/bin/env python3
"""Validate the M8.2 current-C CLI parse/error oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.2" / "current-c.json"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.2" / "manifest.json"


@dataclass(frozen=True)
class CliCase:
    case_id: str
    argv: tuple[str, ...]
    exit_code: int
    stdout_contains: tuple[str, ...] = ()
    stderr_contains: tuple[str, ...] = ()
    stdout_empty: bool = False
    stderr_empty: bool = False


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

CASES: tuple[CliCase, ...] = (
    CliCase("help_long", ("--help",), 0, stdout_contains=HELP_ANCHORS, stderr_empty=True),
    CliCase("help_short", ("-h",), 0, stdout_contains=HELP_ANCHORS, stderr_empty=True),
    CliCase("missing_prompt_value", ("-p",), 2, stderr_contains=("ds4: missing value for -p",), stdout_empty=True),
    CliCase("missing_backend_value", ("--backend",), 2, stderr_contains=("ds4: missing value for --backend",), stdout_empty=True),
    CliCase(
        "unknown_option",
        ("--definitely-unknown",),
        2,
        stderr_contains=("ds4: unknown option: --definitely-unknown", "Usage: ds4"),
        stdout_empty=True,
    ),
    CliCase(
        "invalid_backend",
        ("--backend", "vulkan"),
        2,
        stderr_contains=("ds4: invalid backend: vulkan", "ds4: valid backends are: metal, cuda, cpu"),
        stdout_empty=True,
    ),
    CliCase("invalid_ctx_zero", ("--ctx", "0"), 2, stderr_contains=("ds4: invalid value for --ctx: 0",), stdout_empty=True),
    CliCase("invalid_tokens_text", ("--tokens", "abc"), 2, stderr_contains=("ds4: invalid value for --tokens: abc",), stdout_empty=True),
    CliCase("invalid_temp_nan", ("--temp", "nan"), 2, stderr_contains=("ds4: invalid value for --temp: nan",), stdout_empty=True),
    CliCase("invalid_top_p_range", ("--top-p", "1.5"), 2, stderr_contains=("ds4: invalid value for --top-p: 1.5",), stdout_empty=True),
    CliCase("invalid_min_p_range", ("--min-p", "-0.1"), 2, stderr_contains=("ds4: invalid value for --min-p: -0.1",), stdout_empty=True),
    CliCase("invalid_threads_zero", ("--threads", "0"), 2, stderr_contains=("ds4: invalid value for --threads: 0",), stdout_empty=True),
    CliCase(
        "duplicate_prompt_sources",
        ("-p", "hello", "--prompt-file", "does-not-matter"),
        2,
        stderr_contains=("ds4: specify only one prompt source",),
        stdout_empty=True,
    ),
    CliCase(
        "missing_prompt_file",
        ("--prompt-file", "ds4-parity/baselines/cli/m8.2/missing-prompt.txt"),
        2,
        stderr_contains=("ds4: failed to open prompt file: ds4-parity/baselines/cli/m8.2/missing-prompt.txt",),
        stdout_empty=True,
    ),
    CliCase("server_deprecation", ("--server",), 2, stderr_contains=("ds4: use ds4-server for the HTTP server",), stdout_empty=True),
    CliCase(
        "removed_metal_graph_generate",
        ("--metal-graph-generate",),
        2,
        stderr_contains=("ds4: --metal-graph-generate was removed; --metal is the graph path",),
        stdout_empty=True,
    ),
    CliCase(
        "dump_tokens_requires_prompt",
        ("--dump-tokens",),
        2,
        stderr_contains=("ds4: --dump-tokens requires -p or --prompt-file",),
        stdout_empty=True,
    ),
    CliCase(
        "imatrix_out_requires_dataset",
        ("--imatrix-out", "out.dat"),
        2,
        stderr_contains=("ds4: --imatrix-out requires --imatrix-dataset",),
        stdout_empty=True,
    ),
    CliCase(
        "imatrix_dataset_requires_out",
        ("--imatrix-dataset", "dataset.txt"),
        2,
        stderr_contains=("ds4: --imatrix-dataset requires --imatrix-out",),
        stdout_empty=True,
    ),
    CliCase(
        "perplexity_rejects_prompt",
        ("--perplexity-file", "text.txt", "-p", "hello"),
        2,
        stderr_contains=("ds4: --perplexity-file does not use -p/--prompt-file",),
        stdout_empty=True,
    ),
)

NO_MODEL_MARKERS = (
    "failed to open model",
    "context buffers",
    "Metal device",
    "CUDA backend",
    "loading model",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def capture_case(binary: Path, case: CliCase) -> dict[str, Any]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    proc = subprocess.run(
        [str(binary), *case.argv],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return {
        "id": case.case_id,
        "argv": list(case.argv),
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_bytes": len(proc.stdout.encode("utf-8")),
        "stderr_bytes": len(proc.stderr.encode("utf-8")),
        "stdout_sha256": sha256_text(proc.stdout),
        "stderr_sha256": sha256_text(proc.stderr),
    }


def capture_baseline(binary: Path) -> dict[str, Any]:
    if not binary.is_file():
        raise SystemExit(f"missing CLI binary: {binary}; run `arch -arm64 make ds4` first")
    return {
        "schema": "ds4.cli_parse_oracle.v1",
        "source": "current-c-cli-parse",
        "binary": "./ds4",
        "cases": [capture_case(binary, case) for case in CASES],
    }


def expected_by_id() -> dict[str, CliCase]:
    return {case.case_id: case for case in CASES}


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def check_case(report: Report, raw: Any, expected: CliCase, path: str) -> None:
    case = require_dict(report, raw, path)
    report.check(case.get("id") == expected.case_id, f"{path}.id drift")
    report.check(case.get("argv") == list(expected.argv), f"{expected.case_id}.argv drift")
    report.check(case.get("exit_code") == expected.exit_code, f"{expected.case_id}.exit_code drift")
    stdout = case.get("stdout")
    stderr = case.get("stderr")
    report.check(isinstance(stdout, str), f"{expected.case_id}.stdout invalid")
    report.check(isinstance(stderr, str), f"{expected.case_id}.stderr invalid")
    if not isinstance(stdout, str):
        stdout = ""
    if not isinstance(stderr, str):
        stderr = ""
    report.check(case.get("stdout_bytes") == len(stdout.encode("utf-8")), f"{expected.case_id}.stdout_bytes drift")
    report.check(case.get("stderr_bytes") == len(stderr.encode("utf-8")), f"{expected.case_id}.stderr_bytes drift")
    report.check(case.get("stdout_sha256") == sha256_text(stdout), f"{expected.case_id}.stdout_sha256 drift")
    report.check(case.get("stderr_sha256") == sha256_text(stderr), f"{expected.case_id}.stderr_sha256 drift")
    if expected.stdout_empty:
        report.check(stdout == "", f"{expected.case_id}.stdout should be empty")
    if expected.stderr_empty:
        report.check(stderr == "", f"{expected.case_id}.stderr should be empty")
    for anchor in expected.stdout_contains:
        report.check(anchor in stdout, f"{expected.case_id}.stdout missing anchor {anchor!r}")
    for anchor in expected.stderr_contains:
        report.check(anchor in stderr, f"{expected.case_id}.stderr missing anchor {anchor!r}")
    combined = stdout + stderr
    for marker in NO_MODEL_MARKERS:
        report.check(marker not in combined, f"{expected.case_id}: unexpected model-load marker {marker!r}")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.cli_parse_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-cli-parse", "source mismatch")
    report.check(root.get("binary") == "./ds4", "binary drift")
    cases = require_list(report, root.get("cases"), "cases")
    expected = expected_by_id()
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    report.check(ids == [case.case_id for case in CASES], "case order or coverage drift")
    report.check(len(ids) == len(set(ids)), "duplicate case ids")
    for idx, raw_case in enumerate(cases):
        case_id = raw_case.get("id") if isinstance(raw_case, dict) else None
        exp = expected.get(case_id)
        if exp is None:
            report.check(False, f"cases[{idx}].id unexpected {case_id!r}")
            continue
        check_case(report, raw_case, exp, f"cases[{idx}]")
    return report


def build_manifest(artifact: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_parse_manifest.v1",
        "milestone": "M8.2",
        "oracle": "current C no-model CLI parser and early error surface",
        "artifact": {
            "path": "current-c.json",
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "capture_commands": [
            "arch -arm64 make ds4",
            "python3 ds4-parity/check_cli_parse_dump.py --write-baseline ds4-parity/baselines/cli/m8.2/current-c.json --write-manifest ds4-parity/baselines/cli/m8.2/manifest.json",
            "python3 ds4-parity/check_cli_parse_dump.py ds4-parity/baselines/cli/m8.2/current-c.json --manifest ds4-parity/baselines/cli/m8.2/manifest.json --negative-test",
        ],
    }


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    root = require_dict(report, load_json(path), "manifest")
    report.check(root.get("schema") == "ds4.cli_parse_manifest.v1", "manifest schema mismatch")
    report.check(root.get("milestone") == "M8.2", "manifest milestone mismatch")
    artifact_info = require_dict(report, root.get("artifact"), "manifest.artifact")
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("size_bytes") == artifact.stat().st_size, "manifest artifact size drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    commands = "\n".join(require_list(report, root.get("capture_commands"), "manifest.capture_commands"))
    for required in (
        "arch -arm64 make ds4",
        "--write-baseline ds4-parity/baselines/cli/m8.2/current-c.json",
        "--negative-test",
    ):
        report.check(required in commands, f"manifest capture command missing {required}")
    return report


def run_negative_tests(obj: Any, manifest_path: Path | None, artifact_path: Path) -> Report:
    report = Report()

    def expect_failure(name: str, path: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        sub = check_dump(candidate)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("exit code drift", ["cases", 0, "exit_code"], 2)
    expect_failure("argv drift", ["cases", 3, "argv"], ["--other"])
    expect_failure("help anchor drift", ["cases", 0, "stdout"], "")
    expect_failure("stderr anchor drift", ["cases", 5, "stderr"], "ds4: invalid backend: vulkan\n")
    expect_failure("sha drift", ["cases", 2, "stderr_sha256"], "0" * 64)
    expect_failure("case coverage drift", ["cases"], obj["cases"][:-1])

    if manifest_path is not None:
        manifest = load_json(manifest_path)
        manifest["artifact"]["sha256"] = "0" * 64
        tmp = Report()
        tmp.check(manifest.get("artifact", {}).get("sha256") == sha256_file(artifact_path), "manifest sha drift")
        report.check(not tmp.ok, "negative test did not fail: manifest sha drift")

    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "ds4")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if args.write_baseline:
        baseline = capture_baseline(args.binary)
        write_json(args.write_baseline, baseline)
        if args.write_manifest:
            write_json(args.write_manifest, build_manifest(args.write_baseline))
        return 0
    if args.write_manifest:
        write_json(args.write_manifest, build_manifest(args.artifact))
        return 0

    obj = load_json(args.artifact)
    dump_report = check_dump(obj)
    print_report("CLI parse oracle", dump_report)
    ok = dump_report.ok

    manifest_path = args.manifest
    if manifest_path is None and args.artifact.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.artifact)
        print_report("CLI parse manifest", manifest_report)
        ok = ok and manifest_report.ok

    if args.negative_test:
        negative_report = run_negative_tests(obj, manifest_path, args.artifact)
        print_report("CLI parse negative tests", negative_report)
        ok = ok and negative_report.ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
