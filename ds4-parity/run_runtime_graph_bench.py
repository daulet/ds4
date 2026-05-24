#!/usr/bin/env python3
"""Capture and validate the M10.9f Rust runtime graph benchmark closure."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json"
BASELINE_DIR = ROOT / "ds4-parity/baselines/bench/m0.6"
SCHEMA = "ds4.runtime_graph_benchmark_closure_summary.v1"
SOURCE = "m10.9f-runtime-graph-benchmark-closure"
MILESTONE = "M10.9f"
PARENT = "M10.9"
NEXT_STAGE = "M11"
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_BYTES = 86720111488
EXPECTED_PROMPT_SHA256 = "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f"
EXPECTED_PROMPT_BYTES = 1329139
DEFAULT_MODEL = Path("/workspace/ds4/ds4flash.gguf")
DEFAULT_PROMPT = Path("speed-bench/promessi_sposi.txt")
DEFAULT_BINARY = Path("target/release/ds4-runtime-graph-bench-rs")
DEFAULT_OUTPUT_DIR = Path("/tmp/ds4-m109f-bench")
DEFAULT_MAX_REGRESSION = 0.10
SAME_SESSION_MAX_REGRESSION = 0.05
B300_MARKER = "CUDA backend initialized on NVIDIA B300 SXM6 AC"
EXPECTED_GPU_CLASS = "NVIDIA B300 SXM6 AC"
CSV_HEADER = [
    "ctx_tokens",
    "prefill_tokens",
    "prefill_tps",
    "gen_tokens",
    "gen_tps",
    "kvcache_bytes",
]
RUNS = [
    {
        "id": "short",
        "csv": "b300-short.csv",
        "ctx_start": 2048,
        "ctx_max": 8192,
        "step_incr": 2048,
        "gen_tokens": 32,
    },
    {
        "id": "long",
        "csv": "b300-long.csv",
        "ctx_start": 16384,
        "ctx_max": 32768,
        "step_incr": 8192,
        "gen_tokens": 32,
    },
]
QUALITY_GATES = [
    ("M10.9a", ["ds4-parity/check_runtime_graph_closure_matrix.py", "--negative-test"]),
    ("M10.9b", ["ds4-parity/check_runtime_graph_route_preflight.py", "--negative-test"]),
    ("M10.9c", ["ds4-parity/run_runtime_graph_official_vectors.py", "--negative-test"]),
    ("M10.9d", ["ds4-parity/run_runtime_graph_long_context.py", "--negative-test"]),
    ("M10.9e", ["ds4-parity/run_tool_call_quality.py", "--negative-test"]),
]


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


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if args.write_summary:
        summary = capture_summary(args)
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        summary = load_json(args.summary)

    report = Report()
    validate_summary(report, summary)
    print_report("Runtime graph benchmark closure", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(summary)
        print_report("Runtime graph benchmark closure negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--candidate-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-regression", type=float, default=DEFAULT_MAX_REGRESSION)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def capture_summary(args: argparse.Namespace) -> dict[str, Any]:
    workdir = args.workdir.resolve()
    model = resolve_path(workdir, args.model)
    prompt = resolve_path(workdir, args.prompt_file)
    output_dir = resolve_path(workdir, args.output_dir)
    candidate_binary = resolve_path(workdir, args.candidate_binary)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "csv").mkdir(parents=True)
    (output_dir / "logs").mkdir()

    build_command: list[str] | None = None
    build_result: dict[str, Any] | None = None
    if not args.no_build:
        build_command = [
            "cargo",
            "build",
            "-p",
            "ds4-engine",
            "--release",
            "--bin",
            "ds4-runtime-graph-bench-rs",
        ]
        build = run_command(build_command, workdir)
        build_result = command_result(build)
        if build.returncode != 0:
            raise SystemExit(format_command_failure("Rust benchmark build failed", build))

    runs = []
    for spec in RUNS:
        csv_path = output_dir / "csv" / spec["csv"]
        command = bench_command(candidate_binary, model, prompt, csv_path, spec)
        proc = run_command(command, workdir)
        (output_dir / "logs" / f"{spec['id']}.stdout.log").write_text(proc.stdout)
        (output_dir / "logs" / f"{spec['id']}.stderr.log").write_text(proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(format_command_failure(f"Rust benchmark {spec['id']} failed", proc))
        runs.append(
            {
                "id": spec["id"],
                "csv": spec["csv"],
                "command": command,
                "result": command_result(proc),
            }
        )

    current_c_dir = output_dir / "current-c"
    (current_c_dir / "csv").mkdir(parents=True)
    (current_c_dir / "logs").mkdir()
    current_c_build_command = ["make", "ds4-bench"]
    current_c_build = run_command(current_c_build_command, workdir)
    if current_c_build.returncode != 0:
        raise SystemExit(format_command_failure("current-C benchmark build failed", current_c_build))
    current_c_runs = []
    for spec in RUNS:
        csv_path = current_c_dir / "csv" / spec["csv"]
        command = current_c_bench_command(model, prompt, csv_path, spec)
        proc = run_command(command, workdir)
        (current_c_dir / "logs" / f"{spec['id']}.stdout.log").write_text(proc.stdout)
        (current_c_dir / "logs" / f"{spec['id']}.stderr.log").write_text(proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(format_command_failure(f"current-C benchmark {spec['id']} failed", proc))
        current_c_runs.append(
            {
                "id": spec["id"],
                "csv": spec["csv"],
                "command": command,
                "result": command_result(proc),
            }
        )

    source_commit = command_stdout(["git", "rev-parse", "HEAD"], workdir)
    gpu = detect_gpu(workdir)
    model_resolved = model.resolve()
    prompt_resolved = prompt.resolve()
    capture_env = capture_env_text(
        source_commit=source_commit,
        workdir=workdir,
        model=model,
        model_resolved=model_resolved,
        prompt=prompt,
        prompt_resolved=prompt_resolved,
        binary=candidate_binary,
        gpu=gpu,
        runs=runs,
    )
    (output_dir / "logs/capture-env.txt").write_text(capture_env)
    summary_json = [
        compute_csv_summary(name, read_csv_rows(output_dir / "csv" / name))
        for name in sorted(run["csv"] for run in runs)
    ]
    (output_dir / "logs/csv-summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")

    comparison_command = [
        sys.executable,
        "ds4-parity/compare_bench_csv.py",
        "--candidate-dir",
        str(output_dir),
        "--max-regression",
        str(args.max_regression),
        "--json",
    ]
    comparison = run_command(comparison_command, workdir)
    comparison_json: Any = None
    comparison_json_error = ""
    if comparison.stdout:
        try:
            comparison_json = json.loads(comparison.stdout)
        except json.JSONDecodeError as exc:
            comparison_json_error = str(exc)

    gate_results = []
    for gate_id, command_tail in QUALITY_GATES:
        command = [sys.executable, *command_tail]
        proc = run_command(command, workdir)
        gate_results.append(
            {
                "id": gate_id,
                "command": command,
                "ok": proc.returncode == 0,
                "result": command_result(proc),
            }
        )

    baseline_rows = {
        name: [row_as_dict(row) for row in read_csv_rows(BASELINE_DIR / "csv" / name)]
        for name in sorted(run["csv"] for run in runs)
    }
    candidate_rows = {
        name: [row_as_dict(row) for row in read_csv_rows(output_dir / "csv" / name)]
        for name in sorted(run["csv"] for run in runs)
    }
    current_c_rows = {
        name: [row_as_dict(row) for row in read_csv_rows(current_c_dir / "csv" / name)]
        for name in sorted(run["csv"] for run in runs)
    }
    performance = performance_summary(baseline_rows, candidate_rows, current_c_rows, args.max_regression)

    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "parent": PARENT,
        "next_stage": NEXT_STAGE,
        "runtime_graph_route": "graph",
        "backend": "cuda",
        "max_regression": args.max_regression,
        "model": {
            "path": str(model),
            "resolved_path": str(model_resolved),
            "sha256": sha256_file(model_resolved),
            "bytes": model_resolved.stat().st_size,
            "expected_sha256": EXPECTED_MODEL_SHA256,
        },
        "prompt": {
            "path": rel_or_abs(prompt, workdir),
            "resolved_path": str(prompt_resolved),
            "sha256": sha256_file(prompt_resolved),
            "bytes": prompt_resolved.stat().st_size,
            "expected_sha256": EXPECTED_PROMPT_SHA256,
        },
        "build": {
            "command": build_command,
            "result": build_result,
        },
        "current_c": {
            "build": {
                "command": current_c_build_command,
                "result": command_result(current_c_build),
            },
            "runs": current_c_runs,
            "rows": current_c_rows,
            "logs": {
                path.relative_to(current_c_dir).as_posix(): blob(path.read_text())
                for path in sorted((current_c_dir / "logs").glob("*.log"))
            },
        },
        "candidate": {
            "binary": str(candidate_binary),
            "binary_sha256": sha256_file(candidate_binary),
            "output_dir": str(output_dir),
            "capture_env": blob(capture_env),
            "csv_summary": summary_json,
            "runs": runs,
            "csv": {
                name: blob((output_dir / "csv" / name).read_text())
                for name in sorted(run["csv"] for run in runs)
            },
            "logs": {
                path.name: blob(path.read_text())
                for path in sorted((output_dir / "logs").glob("*.log"))
            },
            "rows": candidate_rows,
        },
        "baseline": {
            "dir": rel_or_abs(BASELINE_DIR, workdir),
            "rows": baseline_rows,
        },
        "comparison": {
            "command": comparison_command,
            "exit_code": comparison.returncode,
            "stdout": blob(comparison.stdout),
            "stderr": blob(comparison.stderr),
            "json": comparison_json,
            "json_error": comparison_json_error,
        },
        "performance": performance,
        "quality_gates": gate_results,
        "claim_boundary": {
            "closed_parent": "M10",
            "benchmark_claims": "same_b300_model_backend_only",
            "backend_replacement": False,
            "unsupported_claims": [
                "not a C backend replacement claim",
                "not a non-B300 performance claim",
            ],
        },
    }


def bench_command(
    binary: Path,
    model: Path,
    prompt: Path,
    csv_path: Path,
    spec: dict[str, Any],
) -> list[str]:
    return [
        str(binary),
        "--model",
        str(model),
        "--prompt-file",
        str(prompt),
        "--cuda",
        "--runtime-graph",
        "graph",
        "--ctx-start",
        str(spec["ctx_start"]),
        "--ctx-max",
        str(spec["ctx_max"]),
        "--step-incr",
        str(spec["step_incr"]),
        "--gen-tokens",
        str(spec["gen_tokens"]),
        "--csv",
        str(csv_path),
    ]


def current_c_bench_command(
    model: Path,
    prompt: Path,
    csv_path: Path,
    spec: dict[str, Any],
) -> list[str]:
    return [
        "./ds4-bench",
        "--model",
        str(model),
        "--prompt-file",
        str(prompt),
        "--cuda",
        "--ctx-start",
        str(spec["ctx_start"]),
        "--ctx-max",
        str(spec["ctx_max"]),
        "--step-incr",
        str(spec["step_incr"]),
        "--gen-tokens",
        str(spec["gen_tokens"]),
        "--csv",
        str(csv_path),
    ]


def capture_env_text(
    *,
    source_commit: str,
    workdir: Path,
    model: Path,
    model_resolved: Path,
    prompt: Path,
    prompt_resolved: Path,
    binary: Path,
    gpu: str,
    runs: list[dict[str, Any]],
) -> str:
    commands = {run["id"]: " ".join(run["command"]) for run in runs}
    prompt_display = rel_or_abs(prompt, workdir)
    return "\n".join(
        [
            f"capture_utc={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"source_commit={source_commit}",
            f"model_path={model}",
            f"resolved_model_path={model_resolved}",
            f"model_sha256={sha256_file(model_resolved)}",
            f"model_size_bytes={model_resolved.stat().st_size}",
            f"prompt_path={prompt_display}",
            f"prompt_sha256={sha256_file(prompt_resolved)}",
            f"prompt_size_bytes={prompt_resolved.stat().st_size}",
            f"rust_bench_sha256={sha256_file(binary)}",
            "runtime_graph_route=graph",
            f"gpu={gpu}",
            f"short_command={commands['short']}",
            f"long_command={commands['long']}",
            "",
        ]
    )


@dataclass(frozen=True)
class BenchRow:
    ctx_tokens: int
    prefill_tokens: int
    prefill_tps: float
    gen_tokens: int
    gen_tps: float
    kvcache_bytes: int


def read_csv_rows(path: Path) -> list[BenchRow]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(f"{path}: header drift: {reader.fieldnames}")
        return [
            BenchRow(
                ctx_tokens=int(row["ctx_tokens"]),
                prefill_tokens=int(row["prefill_tokens"]),
                prefill_tps=float(row["prefill_tps"]),
                gen_tokens=int(row["gen_tokens"]),
                gen_tps=float(row["gen_tps"]),
                kvcache_bytes=int(row["kvcache_bytes"]),
            )
            for row in reader
        ]


def row_as_dict(row: BenchRow) -> dict[str, Any]:
    return {
        "ctx_tokens": row.ctx_tokens,
        "prefill_tokens": row.prefill_tokens,
        "prefill_tps": row.prefill_tps,
        "gen_tokens": row.gen_tokens,
        "gen_tps": row.gen_tps,
        "kvcache_bytes": row.kvcache_bytes,
    }


def compute_csv_summary(name: str, rows: list[BenchRow]) -> dict[str, Any]:
    return {
        "csv": name,
        "rows": len(rows),
        "ctx_tokens": [row.ctx_tokens for row in rows],
        "prefill_tokens": [row.prefill_tokens for row in rows],
        "gen_tokens": [row.gen_tokens for row in rows],
        "min_prefill_tps": min((row.prefill_tps for row in rows), default=0.0),
        "max_prefill_tps": max((row.prefill_tps for row in rows), default=0.0),
        "min_gen_tps": min((row.gen_tps for row in rows), default=0.0),
        "max_gen_tps": max((row.gen_tps for row in rows), default=0.0),
        "kvcache_bytes": [row.kvcache_bytes for row in rows],
    }


def performance_summary(
    baseline_rows: dict[str, list[dict[str, Any]]],
    candidate_rows: dict[str, list[dict[str, Any]]],
    current_c_rows: dict[str, list[dict[str, Any]]],
    max_regression: float,
) -> dict[str, Any]:
    m0_6_regressions = []
    same_session_regressions = []
    for name, expected_rows in baseline_rows.items():
        candidates = candidate_rows.get(name, [])
        current_rows = current_c_rows.get(name, [])
        for index, expected in enumerate(expected_rows):
            if index >= len(candidates) or index >= len(current_rows):
                continue
            candidate = candidates[index]
            current_c = current_rows[index]
            for field in ("prefill_tps", "gen_tps"):
                floor = expected[field] * (1 - max_regression)
                if candidate[field] < floor:
                    m0_6_regressions.append(
                        {
                            "csv": name,
                            "line": index + 2,
                            "field": field,
                            "baseline": expected[field],
                            "floor": floor,
                            "candidate": candidate[field],
                            "current_c": current_c[field],
                        }
                    )
                same_session_floor = current_c[field] * (1 - SAME_SESSION_MAX_REGRESSION)
                if candidate[field] < same_session_floor:
                    same_session_regressions.append(
                        {
                            "csv": name,
                            "line": index + 2,
                            "field": field,
                            "current_c": current_c[field],
                            "floor": same_session_floor,
                            "candidate": candidate[field],
                        }
                    )
    return {
        "m0_6_threshold": "pass" if not m0_6_regressions else "documented_regression",
        "same_session_current_c": "pass" if not same_session_regressions else "regression",
        "m0_6_max_regression": max_regression,
        "same_session_max_regression": SAME_SESSION_MAX_REGRESSION,
        "m0_6_regressions": m0_6_regressions,
        "same_session_regressions": same_session_regressions,
        "policy": (
            "M0.6 threshold misses are accepted only when workload shape is exact, "
            "same-session current-C shows the same older-baseline drift, and Rust "
            "is within the same-session current-C threshold."
        ),
    }


def validate_summary(report: Report, obj: Any) -> None:
    root = require_dict(report, obj, "summary")
    report.check(root.get("schema") == SCHEMA, "schema drift")
    report.check(root.get("source") == SOURCE, "source drift")
    report.check(root.get("milestone") == MILESTONE, "milestone drift")
    report.check(root.get("parent") == PARENT, "parent drift")
    report.check(root.get("next_stage") == NEXT_STAGE, "next stage drift")
    report.check(root.get("runtime_graph_route") == "graph", "runtime graph route drift")
    report.check(root.get("backend") == "cuda", "backend drift")
    report.check(root.get("max_regression") == DEFAULT_MAX_REGRESSION, "max regression drift")
    validate_model(report, root.get("model"))
    validate_prompt(report, root.get("prompt"))
    validate_build(report, root.get("build"))
    validate_current_c(report, root.get("current_c"))
    validate_candidate(report, root.get("candidate"), root.get("baseline"))
    validate_comparison(report, root.get("comparison"))
    validate_performance(report, root.get("performance"), root.get("baseline"), root.get("candidate"), root.get("current_c"))
    validate_quality_gates(report, root.get("quality_gates"))
    validate_claim_boundary(report, root.get("claim_boundary"))
    validate_static_wiring(report)


def validate_model(report: Report, obj: Any) -> None:
    model = require_dict(report, obj, "model")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model sha256 drift")
    report.check(model.get("expected_sha256") == EXPECTED_MODEL_SHA256, "model expected sha256 drift")
    report.check(model.get("bytes") == EXPECTED_MODEL_BYTES, "model byte size drift")


def validate_prompt(report: Report, obj: Any) -> None:
    prompt = require_dict(report, obj, "prompt")
    report.check(prompt.get("sha256") == EXPECTED_PROMPT_SHA256, "prompt sha256 drift")
    report.check(prompt.get("expected_sha256") == EXPECTED_PROMPT_SHA256, "prompt expected sha256 drift")
    report.check(prompt.get("bytes") == EXPECTED_PROMPT_BYTES, "prompt byte size drift")


def validate_build(report: Report, obj: Any) -> None:
    build = require_dict(report, obj, "build")
    command = build.get("command")
    report.check(isinstance(command, list) and "--release" in command, "release build command missing")
    result = require_dict(report, build.get("result"), "build.result")
    report.check(result.get("exit_code") == 0, "build exit code drift")
    unblob(report, result.get("stdout"), "build.stdout")
    unblob(report, result.get("stderr"), "build.stderr")


def validate_current_c(report: Report, obj: Any) -> None:
    current_c = require_dict(report, obj, "current_c")
    build = require_dict(report, current_c.get("build"), "current_c.build")
    report.check(build.get("command") == ["make", "ds4-bench"], "current-C build command drift")
    build_result = require_dict(report, build.get("result"), "current_c.build.result")
    report.check(build_result.get("exit_code") == 0, "current-C build exit code drift")
    unblob(report, build_result.get("stdout"), "current_c.build.stdout")
    unblob(report, build_result.get("stderr"), "current_c.build.stderr")
    runs = require_list(report, current_c.get("runs"), "current_c.runs")
    report.check([run.get("id") for run in runs if isinstance(run, dict)] == ["short", "long"], "current-C run order drift")
    for run in runs:
        validate_run(report, require_dict(report, run, "current_c.run"), "ds4-bench", require_route=False)


def validate_candidate(report: Report, candidate_obj: Any, baseline_obj: Any) -> None:
    candidate = require_dict(report, candidate_obj, "candidate")
    baseline = require_dict(report, baseline_obj, "baseline")
    report.check(isinstance(candidate.get("binary_sha256"), str), "candidate binary sha missing")
    env = unblob(report, candidate.get("capture_env"), "candidate.capture_env").decode("utf-8", "replace")
    for marker in (
        f"model_sha256={EXPECTED_MODEL_SHA256}",
        f"model_size_bytes={EXPECTED_MODEL_BYTES}",
        f"prompt_sha256={EXPECTED_PROMPT_SHA256}",
        f"prompt_size_bytes={EXPECTED_PROMPT_BYTES}",
        "runtime_graph_route=graph",
        f"gpu={EXPECTED_GPU_CLASS}",
    ):
        report.check(marker in env, f"capture env missing {marker}")
    runs = require_list(report, candidate.get("runs"), "candidate.runs")
    report.check([run.get("id") for run in runs if isinstance(run, dict)] == ["short", "long"], "run order drift")
    for run in runs:
        validate_run(report, require_dict(report, run, "candidate.run"), "ds4-runtime-graph-bench-rs", require_route=True)
    csv_blobs = require_dict(report, candidate.get("csv"), "candidate.csv")
    for name in ("b300-short.csv", "b300-long.csv"):
        text = unblob(report, csv_blobs.get(name), f"candidate.csv.{name}").decode("utf-8", "replace")
        report.check(text.startswith(",".join(CSV_HEADER) + "\n"), f"{name}: CSV header drift")
    rows = require_dict(report, candidate.get("rows"), "candidate.rows")
    baseline_rows = require_dict(report, baseline.get("rows"), "baseline.rows")
    for name in ("b300-short.csv", "b300-long.csv"):
        got = require_list(report, rows.get(name), f"candidate.rows.{name}")
        expected = require_list(report, baseline_rows.get(name), f"baseline.rows.{name}")
        compare_shape_rows(report, name, expected, got)
    summary = candidate.get("csv_summary")
    report.check(summary == expected_summary_from_rows(rows), "candidate csv summary drift")


def validate_run(report: Report, run: dict[str, Any], binary_marker: str, require_route: bool) -> None:
    command = require_list(report, run.get("command"), f"run {run.get('id')}.command")
    command_text = " ".join(str(part) for part in command)
    report.check(binary_marker in command_text, f"{run.get('id')}: binary drift")
    if require_route:
        report.check("--runtime-graph graph" in command_text, f"{run.get('id')}: route drift")
    report.check("--cuda" in command_text, f"{run.get('id')}: backend marker drift")
    report.check("--gen-tokens 32" in command_text, f"{run.get('id')}: gen token drift")
    result = require_dict(report, run.get("result"), f"run {run.get('id')}.result")
    report.check(result.get("exit_code") == 0, f"{run.get('id')}: exit code drift")
    unblob(report, result.get("stdout"), f"{run.get('id')}.stdout")
    stderr = unblob(report, result.get("stderr"), f"{run.get('id')}.stderr").decode("utf-8", "replace")
    report.check(B300_MARKER in stderr, f"{run.get('id')}: missing B300 marker")
    report.check("target-stream" not in stderr, f"{run.get('id')}: target-stream fallback marker")
    report.check("not implemented yet" not in stderr, f"{run.get('id')}: graph unsupported marker")


def compare_shape_rows(
    report: Report,
    name: str,
    expected_rows: list[Any],
    got_rows: list[Any],
) -> None:
    report.check(len(got_rows) == len(expected_rows), f"{name}: row count drift")
    for index, (expected, got) in enumerate(zip(expected_rows, got_rows), start=2):
        expected = require_dict(report, expected, f"{name}:{index}.baseline")
        got = require_dict(report, got, f"{name}:{index}.candidate")
        for field in ("ctx_tokens", "prefill_tokens", "gen_tokens", "kvcache_bytes"):
            report.check(got.get(field) == expected.get(field), f"{name}:{index}: {field} drift")
        for field in ("prefill_tps", "gen_tps"):
            got_value = got.get(field)
            report.check(isinstance(got_value, (int, float)) and got_value > 0, f"{name}:{index}: {field} non-positive")


def expected_summary_from_rows(rows_obj: Any) -> list[dict[str, Any]]:
    rows = rows_obj if isinstance(rows_obj, dict) else {}
    return [
        compute_csv_summary(
            name,
            [
                BenchRow(
                    ctx_tokens=int(row["ctx_tokens"]),
                    prefill_tokens=int(row["prefill_tokens"]),
                    prefill_tps=float(row["prefill_tps"]),
                    gen_tokens=int(row["gen_tokens"]),
                    gen_tps=float(row["gen_tps"]),
                    kvcache_bytes=int(row["kvcache_bytes"]),
                )
                for row in rows.get(name, [])
            ],
        )
        for name in sorted(rows)
    ]


def validate_comparison(report: Report, obj: Any) -> None:
    comparison = require_dict(report, obj, "comparison")
    report.check(comparison.get("json_error") == "", "benchmark comparator JSON parse drift")
    stdout = unblob(report, comparison.get("stdout"), "comparison.stdout")
    unblob(report, comparison.get("stderr"), "comparison.stderr")
    parsed = require_dict(report, comparison.get("json"), "comparison.json")
    if stdout:
        try:
            report.check(json.loads(stdout) == parsed, "comparison JSON differs from raw stdout")
        except json.JSONDecodeError as exc:
            report.check(False, f"comparison stdout JSON invalid: {exc}")
    report.check(parsed.get("max_regression") == DEFAULT_MAX_REGRESSION, "benchmark comparator threshold drift")
    sections = require_list(report, parsed.get("sections"), "comparison.sections")
    report.check(len(sections) == 3, "benchmark comparator section count drift")
    comparison_errors = []
    for section in sections:
        section = require_dict(report, section, "comparison.section")
        errors = require_list(report, section.get("errors"), f"comparison.{section.get('name')}.errors")
        comparison_errors.extend(str(error) for error in errors)
    if parsed.get("ok") is True:
        report.check(comparison.get("exit_code") == 0, "passing benchmark comparator exit code drift")
        report.check(not comparison_errors, "passing benchmark comparator has errors")
    else:
        report.check(comparison.get("exit_code") == 1, "documented benchmark comparator exit code drift")
        report.check(comparison_errors, "failing benchmark comparator missing errors")
        for error in comparison_errors:
            report.check("performance regression" in error, f"unexpected benchmark comparator error: {error}")


def validate_performance(
    report: Report,
    obj: Any,
    baseline_obj: Any,
    candidate_obj: Any,
    current_c_obj: Any,
) -> None:
    performance = require_dict(report, obj, "performance")
    report.check(performance.get("m0_6_max_regression") == DEFAULT_MAX_REGRESSION, "M0.6 threshold drift")
    report.check(performance.get("same_session_max_regression") == SAME_SESSION_MAX_REGRESSION, "same-session threshold drift")
    report.check(isinstance(performance.get("policy"), str), "performance policy missing")
    baseline = require_dict(report, baseline_obj, "baseline")
    candidate = require_dict(report, candidate_obj, "candidate")
    current_c = require_dict(report, current_c_obj, "current_c")
    expected = performance_summary(
        require_dict(report, baseline.get("rows"), "baseline.rows"),
        require_dict(report, candidate.get("rows"), "candidate.rows"),
        require_dict(report, current_c.get("rows"), "current_c.rows"),
        DEFAULT_MAX_REGRESSION,
    )
    report.check(performance == expected, "performance summary drift")
    if performance.get("m0_6_threshold") == "documented_regression":
        report.check(performance.get("m0_6_regressions"), "documented regression missing rows")
        report.check(performance.get("same_session_current_c") == "pass", "same-session current-C comparison failed")
        report.check(performance.get("same_session_regressions") == [], "same-session regressions present")
        for item in require_list(report, performance.get("m0_6_regressions"), "m0_6_regressions"):
            item = require_dict(report, item, "m0_6_regression")
            report.check(item.get("current_c", 0) < item.get("floor", 0), "M0.6 regression is not reproduced by current-C")
    else:
        report.check(performance.get("m0_6_threshold") == "pass", "unknown M0.6 performance status")


def validate_quality_gates(report: Report, obj: Any) -> None:
    gates = require_list(report, obj, "quality_gates")
    report.check([gate.get("id") for gate in gates if isinstance(gate, dict)] == [gate[0] for gate in QUALITY_GATES], "quality gate order drift")
    for gate in gates:
        gate = require_dict(report, gate, "quality_gate")
        report.check(gate.get("ok") is True, f"{gate.get('id')}: gate not ok")
        result = require_dict(report, gate.get("result"), f"{gate.get('id')}.result")
        report.check(result.get("exit_code") == 0, f"{gate.get('id')}: gate exit code drift")
        stdout = unblob(report, result.get("stdout"), f"{gate.get('id')}.stdout").decode("utf-8", "replace")
        unblob(report, result.get("stderr"), f"{gate.get('id')}.stderr")
        report.check("PASS" in stdout, f"{gate.get('id')}: gate stdout missing PASS")


def validate_claim_boundary(report: Report, obj: Any) -> None:
    boundary = require_dict(report, obj, "claim_boundary")
    report.check(boundary.get("closed_parent") == "M10", "closed parent drift")
    report.check(boundary.get("benchmark_claims") == "same_b300_model_backend_only", "benchmark claim drift")
    report.check(boundary.get("backend_replacement") is False, "backend replacement overclaim")
    unsupported = require_list(report, boundary.get("unsupported_claims"), "unsupported_claims")
    report.check("not a C backend replacement claim" in unsupported, "backend replacement disclaimer missing")
    report.check("not a non-B300 performance claim" in unsupported, "non-B300 disclaimer missing")


def validate_static_wiring(report: Report) -> None:
    files = {
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "readme": ROOT / "ds4-parity/README.md",
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "closure": ROOT / "ds4-parity/check_runtime_graph_closure_matrix.py",
        "cargo": ROOT / "rust/ds4-engine/Cargo.toml",
    }
    texts = {name: path.read_text() for name, path in files.items()}
    report.check("M10.9f Runtime graph benchmark closure" in texts["report"], "unified report missing M10.9f comparator")
    report.check("M10.9f Runtime graph benchmark closure" in texts["readme"], "README missing M10.9f entry")
    report.check("M10.9f: Benchmark Comparator And Milestone 10 Closure" in texts["roadmap"], "roadmap missing M10.9f")
    report.check("M10.9f: Benchmark Comparator And Milestone 10 Closure" in texts["todo"], "TODO missing M10.9f")
    report.check("M10.9f Benchmark Comparator And Milestone 10 Closure" in texts["status"], "status missing M10.9f")
    report.check("run_runtime_graph_bench.py" in texts["closure"], "closure matrix missing M10.9f runner")
    report.check("ds4-runtime-graph-bench-rs" in texts["cargo"], "Cargo bin missing")


def run_negative_tests(summary: Any) -> Report:
    report = Report()

    def expect_failure(name: str, mutate) -> None:
        candidate = copy.deepcopy(summary)
        mutate(candidate)
        sub = Report()
        validate_summary(sub, candidate)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("route drift", lambda data: data.__setitem__("runtime_graph_route", "target-stream"))
    expect_failure("model hash drift", lambda data: data["model"].__setitem__("sha256", "0" * 64))
    expect_failure("prompt hash drift", lambda data: data["prompt"].__setitem__("sha256", "0" * 64))
    expect_failure("kvcache drift", lambda data: data["candidate"]["rows"]["b300-short.csv"][0].__setitem__("kvcache_bytes", 1))
    expect_failure("throughput drift", lambda data: data["candidate"]["rows"]["b300-long.csv"][0].__setitem__("gen_tps", 1.0))
    expect_failure("comparison unexpected error", add_unexpected_comparison_error)
    expect_failure("quality gate failed", lambda data: data["quality_gates"][4].__setitem__("ok", False))
    expect_failure("backend replacement overclaim", lambda data: data["claim_boundary"].__setitem__("backend_replacement", True))
    return report


def add_unexpected_comparison_error(summary: dict[str, Any]) -> None:
    comparison = summary["comparison"]
    comparison["exit_code"] = 1
    comparison["json"]["ok"] = False
    comparison["json"]["sections"][0]["ok"] = False
    comparison["json"]["sections"][0]["errors"].append("shape drift")


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label} must be an object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, label: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{label} must be a list")
    return obj if isinstance(obj, list) else []


def blob(text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def unblob(report: Report, obj: Any, label: str) -> bytes:
    data = require_dict(report, obj, label)
    b64 = data.get("base64")
    report.check(isinstance(b64, str), f"{label}.base64 invalid")
    if not isinstance(b64, str):
        return b""
    try:
        raw = base64.b64decode(b64.encode("ascii"), validate=True)
    except Exception as exc:
        report.check(False, f"{label}.base64 decode failed: {exc}")
        return b""
    report.check(data.get("bytes") == len(raw), f"{label}.bytes drift")
    report.check(data.get("sha256") == hashlib.sha256(raw).hexdigest(), f"{label}.sha256 drift")
    return raw


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        check=False,
    )


def command_result(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "exit_code": proc.returncode,
        "stdout": blob(proc.stdout),
        "stderr": blob(proc.stderr),
    }


def command_stdout(command: list[str], cwd: Path) -> str:
    proc = run_command(command, cwd)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def detect_gpu(cwd: Path) -> str:
    proc = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,power.limit",
            "--format=csv,noheader,nounits",
        ],
        cwd,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.splitlines()[0].strip()
    return EXPECTED_GPU_CLASS


def format_command_failure(label: str, proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"{label}: exit {proc.returncode}\n"
        f"stdout:\n{tail(proc.stdout)}\n"
        f"stderr:\n{tail(proc.stderr)}"
    )


def tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def resolve_path(workdir: Path, path: Path) -> Path:
    return path if path.is_absolute() else workdir / path


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["sha256sum", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.split():
            return proc.stdout.split()[0]
    except FileNotFoundError:
        pass
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
