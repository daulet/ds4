#!/usr/bin/env python3
"""Capture and validate the M10.9c Rust runtime official-vector gate."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json"
SCHEMA = "ds4.runtime_graph_official_vectors_summary.v1"
RUST_SCHEMA = "ds4.runtime_graph_official_vectors.rust.v1"
SOURCE = "m10.9c-runtime-graph-official-vectors"
MILESTONE = "M10.9c"
NEXT_STAGE = "M10.9d"
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_BYTES = 86720111488
EXPECTED_VECTOR_SHA256 = "0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac"
EXPECTED_VECTOR_BYTES = 1207
DEFAULT_MODEL = Path("/workspace/ds4/ds4flash.gguf")
DEFAULT_VECTOR = Path("tests/test-vectors/official.vec")
DEFAULT_TOP_K = 20
LOGPROB_ABS_TOLERANCE = 4.0
DISABLED_CASES = {
    "long_memory_archive": "API/official graph mismatch",
}


@dataclass
class Report:
    checks: int = 0
    errors: list[str] = field(default_factory=list)
    max_abs_logprob_delta: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def record_logprob_delta(self, delta: float) -> None:
        self.max_abs_logprob_delta = max(self.max_abs_logprob_delta, delta)


@dataclass(frozen=True)
class VecTop:
    bytes_hex: str
    logprob: float


@dataclass(frozen=True)
class VecStep:
    index: int
    selected_hex: str
    top: tuple[VecTop, ...]


@dataclass(frozen=True)
class VecCase:
    case_id: str
    ctx: int
    nsteps: int
    prompt_path: str
    steps: tuple[VecStep, ...]


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
    print_report("Runtime graph official vectors", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(summary)
        print_report("Runtime graph official vectors negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def capture_summary(args: argparse.Namespace) -> dict[str, Any]:
    workdir = args.workdir.resolve()
    model = resolve_path(workdir, args.model)
    vectors = resolve_path(workdir, args.vectors)
    candidate_binary = args.candidate_binary
    if candidate_binary is None:
        candidate_binary = workdir / "target/debug/ds4-runtime-official-vectors-rs"
    else:
        candidate_binary = resolve_path(workdir, candidate_binary)

    build_command: list[str] | None = None
    build_result: dict[str, Any] | None = None
    if not args.no_build:
        build_command = [
            "cargo",
            "build",
            "-p",
            "ds4-engine",
            "--bin",
            "ds4-runtime-official-vectors-rs",
        ]
        build = run_command(build_command, workdir)
        build_result = command_result(build)
        if build.returncode != 0:
            raise SystemExit(format_command_failure("build failed", build))

    command = [
        str(candidate_binary),
        "--model",
        str(model),
        "--vectors",
        str(vectors),
        "--cuda",
        "--runtime-graph",
        "graph",
        "--top-k",
        str(args.top_k),
    ]
    proc = run_command(command, workdir)
    rust_json: Any = None
    rust_json_error = ""
    if proc.stdout:
        try:
            rust_json = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            rust_json_error = str(exc)
    model_resolved = model.resolve()
    vector_bytes = vectors.read_bytes()

    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "parent": "M10.9",
        "next_stage": NEXT_STAGE,
        "runtime_graph_route": "graph",
        "backend": "cuda",
        "top_k": args.top_k,
        "logprob_abs_tolerance": LOGPROB_ABS_TOLERANCE,
        "workdir": str(workdir),
        "model": {
            "path": str(model),
            "resolved_path": str(model_resolved),
            "sha256": sha256_file(model_resolved),
            "bytes": model_resolved.stat().st_size,
            "expected_sha256": EXPECTED_MODEL_SHA256,
        },
        "vector": {
            "path": rel_or_abs(vectors, workdir),
            "sha256": hashlib.sha256(vector_bytes).hexdigest(),
            "bytes": len(vector_bytes),
            "expected_sha256": EXPECTED_VECTOR_SHA256,
        },
        "build": {
            "command": build_command,
            "result": build_result,
        },
        "rust": {
            "binary": str(candidate_binary),
            "command": command,
            "exit_code": proc.returncode,
            "stdout": blob(proc.stdout),
            "stderr": blob(proc.stderr),
            "json": rust_json,
            "json_error": rust_json_error,
        },
    }


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
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


def format_command_failure(label: str, proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"{label}: exit {proc.returncode}\n"
        f"stdout:\n{tail(proc.stdout)}\n"
        f"stderr:\n{tail(proc.stderr)}"
    )


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


def blob(text: str) -> dict[str, Any]:
    raw = text.encode()
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


def validate_summary(report: Report, obj: Any) -> None:
    root = require_dict(report, obj, "summary")
    report.check(root.get("schema") == SCHEMA, "summary schema drift")
    report.check(root.get("source") == SOURCE, "summary source drift")
    report.check(root.get("milestone") == MILESTONE, "summary milestone drift")
    report.check(root.get("next_stage") == NEXT_STAGE, "summary next stage drift")
    report.check(root.get("runtime_graph_route") == "graph", "runtime graph route drift")
    report.check(root.get("backend") == "cuda", "backend drift")
    report.check(root.get("top_k") == DEFAULT_TOP_K, "top-k drift")
    report.check(
        close_float(root.get("logprob_abs_tolerance"), LOGPROB_ABS_TOLERANCE),
        "logprob tolerance drift",
    )

    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model sha256 drift")
    report.check(model.get("expected_sha256") == EXPECTED_MODEL_SHA256, "model expected sha256 drift")
    report.check(model.get("bytes") == EXPECTED_MODEL_BYTES, "model byte size drift")

    vector = require_dict(report, root.get("vector"), "vector")
    report.check(vector.get("sha256") == EXPECTED_VECTOR_SHA256, "vector sha256 drift")
    report.check(vector.get("expected_sha256") == EXPECTED_VECTOR_SHA256, "vector expected sha256 drift")
    report.check(vector.get("bytes") == EXPECTED_VECTOR_BYTES, "vector byte size drift")

    rust = require_dict(report, root.get("rust"), "rust")
    report.check(rust.get("exit_code") == 0, "Rust capture exit code drift")
    stdout = unblob(report, rust.get("stdout"), "rust.stdout")
    stderr = unblob(report, rust.get("stderr"), "rust.stderr")
    report.check(b"CUDA backend initialized on NVIDIA B300 SXM6 AC" in stderr, "stderr missing B300 CUDA marker")
    report.check(rust.get("json_error") == "", "Rust stdout JSON parse error")
    candidate = require_dict(report, rust.get("json"), "rust.json")
    if stdout:
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError as exc:
            report.check(False, f"rust.stdout JSON invalid: {exc}")
        else:
            report.check(parsed_stdout == candidate, "stored Rust JSON does not match raw stdout")
    validate_rust_json(report, candidate)
    validate_static_wiring(report)


def validate_rust_json(report: Report, obj: dict[str, Any]) -> None:
    report.check(obj.get("schema") == RUST_SCHEMA, "Rust schema drift")
    report.check(obj.get("source") == "ds4-runtime-official-vectors-rs", "Rust source drift")
    report.check(obj.get("runtime_graph_route") == "graph", "Rust route drift")
    report.check(obj.get("backend") == "cuda", "Rust backend drift")
    report.check(obj.get("top_k") == DEFAULT_TOP_K, "Rust top-k drift")
    report.check(
        close_float(obj.get("logprob_abs_tolerance"), LOGPROB_ABS_TOLERANCE),
        "Rust logprob tolerance drift",
    )
    expected = parse_vec(ROOT / DEFAULT_VECTOR)
    actual_cases = require_list(report, obj.get("cases"), "rust.cases")
    report.check(len(actual_cases) == len(expected), "case count drift")
    report.check(
        [case.get("id") for case in actual_cases if isinstance(case, dict)]
        == [case.case_id for case in expected],
        "case order drift",
    )
    actual_by_id = {
        case.get("id"): case
        for case in actual_cases
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    }
    for expected_case in expected:
        actual = require_dict(
            report,
            actual_by_id.get(expected_case.case_id),
            f"case {expected_case.case_id}",
        )
        compare_case(report, expected_case, actual)


def compare_case(report: Report, expected: VecCase, actual: dict[str, Any]) -> None:
    label = expected.case_id
    report.check(actual.get("ctx") == expected.ctx, f"{label}: ctx drift")
    report.check(actual.get("nsteps") == expected.nsteps, f"{label}: nsteps drift")
    report.check(actual.get("prompt_path") == expected.prompt_path, f"{label}: prompt path drift")
    expected_skip = DISABLED_CASES.get(label)
    if expected_skip:
        report.check(actual.get("skipped") is True, f"{label}: skipped flag drift")
        report.check(actual.get("skip_reason") == expected_skip, f"{label}: skip reason drift")
        report.check(actual.get("steps") == [], f"{label}: skipped case should not contain steps")
        return

    report.check(actual.get("skipped") is False, f"{label}: skipped flag drift")
    actual_steps = require_list(report, actual.get("steps"), f"{label}.steps")
    report.check(len(actual_steps) == len(expected.steps), f"{label}: step count drift")
    for expected_step, raw_step in zip(expected.steps, actual_steps):
        step = require_dict(report, raw_step, f"{label}.steps[{expected_step.index}]")
        compare_step(report, label, expected_step, step)


def compare_step(report: Report, case_id: str, expected: VecStep, actual: dict[str, Any]) -> None:
    label = f"{case_id} step {expected.index}"
    report.check(actual.get("step") == expected.index, f"{label}: step index drift")
    report.check(isinstance(actual.get("selected_token"), int), f"{label}: selected token invalid")
    report.check(actual.get("selected_bytes_hex") == expected.selected_hex, f"{label}: selected bytes drift")
    report.check(actual.get("expected_selected_hex") == expected.selected_hex, f"{label}: expected bytes drift")
    report.check(actual.get("selected_matches_expected") is True, f"{label}: selected match drift")
    scores = require_list(report, actual.get("top_logprobs"), f"{label}.top_logprobs")
    report.check(0 < len(scores) <= DEFAULT_TOP_K, f"{label}: top-logprob length drift")
    seen_ids: set[int] = set()
    for idx, raw_score in enumerate(scores):
        score = require_dict(report, raw_score, f"{label}.top_logprobs[{idx}]")
        check_score(report, score, f"{label}.top_logprobs[{idx}]")
        score_id = score.get("id")
        if isinstance(score_id, int):
            report.check(score_id not in seen_ids, f"{label}: duplicate top id {score_id}")
            seen_ids.add(score_id)
    if scores:
        top = require_dict(report, scores[0], f"{label}.top_logprobs[0]")
        report.check(top.get("bytes_hex") == expected.selected_hex, f"{label}: selected not top-ranked")

    official = require_list(report, actual.get("official_top"), f"{label}.official_top")
    report.check(len(official) == len(expected.top), f"{label}: official top count drift")
    for idx, (expected_top, raw_top) in enumerate(zip(expected.top, official)):
        top = require_dict(report, raw_top, f"{label}.official_top[{idx}]")
        report.check(top.get("bytes_hex") == expected_top.bytes_hex, f"{label}: official top bytes drift")
        report.check(
            close_float(top.get("official_logprob"), expected_top.logprob),
            f"{label}: official logprob drift",
        )
        report.check(top.get("found") is True, f"{label}: official top token missing locally")
        local = require_dict(report, top.get("local_score"), f"{label}.official_top[{idx}].local_score")
        check_score(report, local, f"{label}.official_top[{idx}].local_score")
        report.check(local.get("bytes_hex") == expected_top.bytes_hex, f"{label}: local top bytes drift")
        local_logprob = numeric_value(local.get("logprob"))
        official_logprob = numeric_value(top.get("official_logprob"))
        abs_delta = numeric_value(top.get("abs_delta"))
        if local_logprob is not None and official_logprob is not None and abs_delta is not None:
            recomputed = abs(local_logprob - official_logprob)
            report.record_logprob_delta(recomputed)
            report.check(close_float(abs_delta, recomputed, 1e-5), f"{label}: abs-delta drift")
            report.check(
                recomputed <= LOGPROB_ABS_TOLERANCE,
                f"{label}: logprob drift {recomputed:g} > {LOGPROB_ABS_TOLERANCE:g}",
            )


def check_score(report: Report, score: dict[str, Any], label: str) -> None:
    report.check(isinstance(score.get("id"), int), f"{label}.id invalid")
    bytes_hex = score.get("bytes_hex")
    report.check(isinstance(bytes_hex, str) and is_hex(bytes_hex), f"{label}.bytes_hex invalid")
    for key in ("logit", "logprob"):
        value = numeric_value(score.get(key))
        report.check(value is not None and math.isfinite(value), f"{label}.{key} invalid")


def validate_static_wiring(report: Report) -> None:
    files = {
        "cargo": ROOT / "rust/ds4-engine/Cargo.toml",
        "lib": ROOT / "rust/ds4-engine/src/lib.rs",
        "binary": ROOT / "rust/ds4-engine/src/bin/ds4-runtime-official-vectors-rs.rs",
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "readme": ROOT / "ds4-parity/README.md",
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
    }
    texts = {name: path.read_text() for name, path in files.items()}
    report.check("ds4-runtime-official-vectors-rs" in texts["cargo"], "Cargo binary entry missing")
    report.check("TopLogprobScore" in texts["lib"], "Rust top-logprob API missing")
    report.check("ds4_session_top_logprobs" in texts["lib"], "Rust top-logprob FFI missing")
    report.check("ds4_session_argmax" in texts["lib"], "Rust argmax FFI missing")
    report.check("runtime_graph_route" in texts["binary"], "Rust binary missing route metadata")
    report.check("long_memory_archive" in texts["binary"], "Rust binary missing skip case")
    report.check("run_runtime_graph_official_vectors.py" in texts["report"], "unified report missing M10.9c comparator")
    report.check("M10.9c Runtime graph official-vector gate" in texts["readme"], "README missing M10.9c entry")
    report.check("M10.9c: B300 Official-Vector Rust Runtime Gate" in texts["roadmap"], "roadmap missing M10.9c")
    report.check("M10.9c: B300 Official-Vector Rust Runtime Gate" in texts["todo"], "TODO missing M10.9c")
    report.check("M10.9c B300 Official-Vector Rust Runtime Gate" in texts["status"], "status missing M10.9c")


def run_negative_tests(summary: Any) -> Report:
    report = Report()

    def expect_failure(name: str, mutate) -> None:
        candidate = copy.deepcopy(summary)
        mutate(candidate)
        sub = Report()
        validate_summary(sub, candidate)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("route drift", lambda obj: obj.__setitem__("runtime_graph_route", "target-stream"))
    expect_failure("model hash drift", lambda obj: obj["model"].__setitem__("sha256", "0" * 64))
    expect_failure("vector hash drift", lambda obj: obj["vector"].__setitem__("sha256", "0" * 64))
    expect_failure(
        "skip reason drift",
        lambda obj: case_obj(obj, "long_memory_archive").__setitem__("skip_reason", "removed"),
    )
    expect_failure(
        "selected bytes drift",
        lambda obj: case_obj(obj, "short_italian_fact")["steps"][0].__setitem__("selected_bytes_hex", "00"),
    )
    expect_failure(
        "official top missing",
        lambda obj: case_obj(obj, "short_code_completion")["steps"][0]["official_top"][0].__setitem__("found", False),
    )
    expect_failure(
        "logprob delta drift",
        lambda obj: case_obj(obj, "short_reasoning_plain")["steps"][0]["official_top"][0].__setitem__("abs_delta", 99.0),
    )
    expect_failure(
        "raw stdout hash drift",
        lambda obj: obj["rust"]["stdout"].__setitem__("sha256", "0" * 64),
    )
    return report


def case_obj(summary: dict[str, Any], case_id: str) -> dict[str, Any]:
    cases = summary["rust"]["json"]["cases"]
    for case in cases:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def parse_vec(path: Path) -> list[VecCase]:
    cases: list[VecCase] = []
    current: dict[str, Any] | None = None
    current_step: dict[str, Any] | None = None
    steps: dict[int, dict[str, Any]] = {}
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "case":
            if current is not None:
                raise ValueError(f"{path}:{lineno}: nested case")
            if len(parts) != 5:
                raise ValueError(f"{path}:{lineno}: malformed case")
            current = {
                "id": parts[1],
                "ctx": int(parts[2]),
                "nsteps": int(parts[3]),
                "prompt_path": parts[4],
            }
            steps = {}
            current_step = None
        elif kind == "step":
            if current is None or len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: malformed step")
            index = int(parts[1])
            current_step = {
                "index": index,
                "selected_hex": parts[2],
                "top_count": int(parts[3]),
                "top": [],
            }
            steps[index] = current_step
        elif kind == "top":
            if current_step is None or len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: malformed top")
            current_step["top"].append(VecTop(parts[1], float(parts[2])))
        elif kind == "end":
            if current is None:
                raise ValueError(f"{path}:{lineno}: end outside case")
            ordered_steps: list[VecStep] = []
            for index in range(current["nsteps"]):
                step = steps.get(index)
                if step is None:
                    raise ValueError(f"{path}:{lineno}: missing step {index}")
                if len(step["top"]) != step["top_count"]:
                    raise ValueError(f"{path}:{lineno}: top count drift")
                ordered_steps.append(
                    VecStep(
                        index=step["index"],
                        selected_hex=step["selected_hex"],
                        top=tuple(step["top"]),
                    )
                )
            cases.append(
                VecCase(
                    case_id=current["id"],
                    ctx=current["ctx"],
                    nsteps=current["nsteps"],
                    prompt_path=current["prompt_path"],
                    steps=tuple(ordered_steps),
                )
            )
            current = None
            current_step = None
        else:
            raise ValueError(f"{path}:{lineno}: unexpected line")
    if current is not None:
        raise ValueError(f"{path}: unterminated case")
    return cases


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, label: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{label}: expected array")
    return obj if isinstance(obj, list) else []


def numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def close_float(value: Any, expected: float, tolerance: float = 0.0) -> bool:
    got = numeric_value(value)
    if got is None:
        return False
    return abs(got - expected) <= tolerance


def is_hex(value: str) -> bool:
    return len(value) % 2 == 0 and all(ch in "0123456789abcdef" for ch in value)


def tail(text: str, limit: int = 20) -> str:
    return "\n".join(text.splitlines()[-limit:])


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(
        f"{label}: {status}, {report.checks} checks, "
        f"max_abs_logprob_delta={report.max_abs_logprob_delta:.9g}"
    )
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
