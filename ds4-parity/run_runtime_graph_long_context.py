#!/usr/bin/env python3
"""Capture and validate the M10.9d Rust runtime long-context gate."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json"
SCHEMA = "ds4.runtime_graph_long_context_summary.v1"
RUST_SCHEMA = "ds4.runtime_graph_long_context.rust.v1"
SOURCE = "m10.9d-runtime-graph-long-context"
MILESTONE = "M10.9d"
NEXT_STAGE = "M10.9e"
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_BYTES = 86720111488
EXPECTED_PROMPT_SHA256 = "29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666"
EXPECTED_PROMPT_BYTES = 140309
DEFAULT_MODEL = Path("/workspace/ds4/ds4flash.gguf")
DEFAULT_PROMPT = Path("tests/long_context_story_prompt.txt")
DEFAULT_CTX = 100000
DEFAULT_MAX_TOKENS = 350
DEFAULT_SEED = 12345
B300_MARKER = "CUDA backend initialized on NVIDIA B300 SXM6 AC"
SCORE_SURFACE_POLICY = {
    "generated_text": "behavioral fact-recall exact",
    "token_scores": "not captured; nondeterministic score surface excluded",
}
FACTS = (
    ("Bob", 34),
    ("Alice", 52),
    ("Clara", 71),
    ("Diego", 93),
    ("Elena", 16),
    ("Felix", 88),
    ("Greta", 47),
    ("Hugo", 29),
    ("Iris", 64),
    ("Jonas", 12),
    ("Kira", 81),
    ("Leo", 39),
    ("Marta", 76),
    ("Nadia", 23),
    ("Owen", 58),
    ("Priya", 97),
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
    print_report("Runtime graph long-context", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(summary)
        print_report("Runtime graph long-context negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    parser.add_argument("--tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    candidate_binary = args.candidate_binary
    if candidate_binary is None:
        candidate_binary = workdir / "target/debug/ds4-runtime-long-context-rs"
    else:
        candidate_binary = resolve_path(workdir, candidate_binary)

    rust_build_command: list[str] | None = None
    rust_build_result: dict[str, Any] | None = None
    c_build_command: list[str] | None = None
    c_build_result: dict[str, Any] | None = None
    if not args.no_build:
        rust_build_command = [
            "cargo",
            "build",
            "-p",
            "ds4-engine",
            "--bin",
            "ds4-runtime-long-context-rs",
        ]
        rust_build = run_command(rust_build_command, workdir)
        rust_build_result = command_result(rust_build)
        if rust_build.returncode != 0:
            raise SystemExit(format_command_failure("Rust build failed", rust_build))

        c_build_command = ["make", "ds4_test"]
        c_build = run_command(c_build_command, workdir)
        c_build_result = command_result(c_build)
        if c_build.returncode != 0:
            raise SystemExit(format_command_failure("current-C ds4_test build failed", c_build))

    current_c_env = {
        "DS4_TEST_MODEL": str(model),
        "DS4_TEST_LONG_PROMPT": str(prompt),
    }
    current_c_command = ["./ds4_test", "--long-context"]
    current_c = run_command(current_c_command, workdir, env_updates=current_c_env)

    rust_command = [
        str(candidate_binary),
        "--model",
        str(model),
        "--prompt-file",
        str(prompt),
        "--cuda",
        "--runtime-graph",
        "graph",
        "--ctx",
        str(args.ctx),
        "--tokens",
        str(args.tokens),
        "--seed",
        str(args.seed),
    ]
    rust = run_command(rust_command, workdir)
    rust_json: Any = None
    rust_json_error = ""
    if rust.stdout:
        try:
            rust_json = json.loads(rust.stdout)
        except json.JSONDecodeError as exc:
            rust_json_error = str(exc)

    model_resolved = model.resolve()
    prompt_resolved = prompt.resolve()
    prompt_bytes = prompt_resolved.read_bytes()

    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "parent": "M10.9",
        "next_stage": NEXT_STAGE,
        "runtime_graph_route": "graph",
        "backend": "cuda",
        "settings": {
            "ctx": args.ctx,
            "max_tokens": args.tokens,
            "seed": args.seed,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "min_p": 0,
        },
        "score_surface_policy": SCORE_SURFACE_POLICY,
        "workdir": str(workdir),
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
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "bytes": len(prompt_bytes),
            "expected_sha256": EXPECTED_PROMPT_SHA256,
        },
        "build": {
            "rust": {
                "command": rust_build_command,
                "result": rust_build_result,
            },
            "current_c": {
                "command": c_build_command,
                "result": c_build_result,
            },
        },
        "current_c": {
            "command": current_c_command,
            "env": current_c_env,
            "exit_code": current_c.returncode,
            "stdout": blob(current_c.stdout),
            "stderr": blob(current_c.stderr),
        },
        "rust": {
            "binary": str(candidate_binary),
            "command": rust_command,
            "exit_code": rust.returncode,
            "stdout": blob(rust.stdout),
            "stderr": blob(rust.stderr),
            "json": rust_json,
            "json_error": rust_json_error,
        },
    }


def run_command(
    command: list[str],
    cwd: Path,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    if env_updates:
        env.update(env_updates)
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


def validate_summary(report: Report, obj: Any) -> None:
    root = require_dict(report, obj, "summary")
    report.check(root.get("schema") == SCHEMA, "summary schema drift")
    report.check(root.get("source") == SOURCE, "summary source drift")
    report.check(root.get("milestone") == MILESTONE, "summary milestone drift")
    report.check(root.get("next_stage") == NEXT_STAGE, "summary next stage drift")
    report.check(root.get("runtime_graph_route") == "graph", "runtime graph route drift")
    report.check(root.get("backend") == "cuda", "backend drift")
    report.check(root.get("settings") == expected_settings(), "generation settings drift")
    report.check(root.get("score_surface_policy") == SCORE_SURFACE_POLICY, "score surface policy drift")

    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model sha256 drift")
    report.check(model.get("expected_sha256") == EXPECTED_MODEL_SHA256, "model expected sha256 drift")
    report.check(model.get("bytes") == EXPECTED_MODEL_BYTES, "model byte size drift")

    prompt = require_dict(report, root.get("prompt"), "prompt")
    report.check(prompt.get("sha256") == EXPECTED_PROMPT_SHA256, "prompt sha256 drift")
    report.check(prompt.get("expected_sha256") == EXPECTED_PROMPT_SHA256, "prompt expected sha256 drift")
    report.check(prompt.get("bytes") == EXPECTED_PROMPT_BYTES, "prompt byte size drift")

    validate_build(report, root.get("build"))
    validate_current_c(report, root.get("current_c"))
    rust = validate_rust_capture(report, root.get("rust"))
    validate_rust_json(report, rust)
    validate_static_wiring(report)


def expected_settings() -> dict[str, Any]:
    return {
        "ctx": DEFAULT_CTX,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "seed": DEFAULT_SEED,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "min_p": 0,
    }


def validate_build(report: Report, obj: Any) -> None:
    build = require_dict(report, obj, "build")
    for key in ("rust", "current_c"):
        entry = require_dict(report, build.get(key), f"build.{key}")
        command = entry.get("command")
        report.check(isinstance(command, list) and command, f"build.{key}.command invalid")
        result = require_dict(report, entry.get("result"), f"build.{key}.result")
        report.check(result.get("exit_code") == 0, f"build.{key}.exit_code drift")
        unblob(report, result.get("stdout"), f"build.{key}.stdout")
        unblob(report, result.get("stderr"), f"build.{key}.stderr")


def validate_current_c(report: Report, obj: Any) -> None:
    current_c = require_dict(report, obj, "current_c")
    report.check(current_c.get("exit_code") == 0, "current-C long-context exit code drift")
    env = require_dict(report, current_c.get("env"), "current_c.env")
    report.check(isinstance(env.get("DS4_TEST_MODEL"), str), "current-C model env missing")
    report.check(isinstance(env.get("DS4_TEST_LONG_PROMPT"), str), "current-C prompt env missing")
    stdout = unblob(report, current_c.get("stdout"), "current_c.stdout").decode("utf-8", "replace")
    stderr = unblob(report, current_c.get("stderr"), "current_c.stderr").decode("utf-8", "replace")
    report.check("ds4 tests: ok" in stdout, "current-C stdout missing pass marker")
    report.check("long-context:" in stderr, "current-C stderr missing long-context section")
    report.check("long-context" in stderr and "OK" in stderr, "current-C stderr missing long-context OK marker")
    report.check(B300_MARKER in stderr, "current-C stderr missing B300 CUDA marker")
    for marker in (
        "missing assignment",
        "wrong assignment",
        "long-context: ERR",
        "ds4 tests:",
        "TEST failed",
    ):
        report.check(marker not in stderr, f"current-C stderr has failure marker: {marker}")


def validate_rust_capture(report: Report, obj: Any) -> dict[str, Any]:
    rust = require_dict(report, obj, "rust")
    report.check(rust.get("exit_code") == 0, "Rust long-context exit code drift")
    stdout = unblob(report, rust.get("stdout"), "rust.stdout")
    stderr = unblob(report, rust.get("stderr"), "rust.stderr").decode("utf-8", "replace")
    report.check(B300_MARKER in stderr, "Rust stderr missing B300 CUDA marker")
    report.check("not implemented yet" not in stderr, "Rust stderr reports graph route not implemented")
    report.check("target-stream" not in stderr, "Rust stderr reports target-stream fallback")
    report.check(rust.get("json_error") == "", "Rust stdout JSON parse error")
    candidate = require_dict(report, rust.get("json"), "rust.json")
    if stdout:
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError as exc:
            report.check(False, f"rust.stdout JSON invalid: {exc}")
        else:
            report.check(parsed_stdout == candidate, "stored Rust JSON does not match raw stdout")
    return candidate


def validate_rust_json(report: Report, obj: dict[str, Any]) -> None:
    report.check(obj.get("schema") == RUST_SCHEMA, "Rust schema drift")
    report.check(obj.get("source") == "ds4-runtime-long-context-rs", "Rust source drift")
    report.check(obj.get("runtime_graph_route") == "graph", "Rust route drift")
    report.check(obj.get("backend") == "cuda", "Rust backend drift")
    report.check(obj.get("ctx") == DEFAULT_CTX, "Rust ctx drift")
    report.check(obj.get("max_tokens") == DEFAULT_MAX_TOKENS, "Rust max-token drift")
    report.check(obj.get("seed") == DEFAULT_SEED, "Rust seed drift")
    report.check(obj.get("temperature") == 0, "Rust temperature drift")
    report.check(obj.get("top_k") == 0, "Rust top-k drift")
    report.check(obj.get("top_p") == 1, "Rust top-p drift")
    report.check(obj.get("min_p") == 0, "Rust min-p drift")
    report.check(obj.get("exit_code") == 0, "Rust generated exit code drift")
    prompt_tokens = obj.get("prompt_tokens")
    completion_tokens = obj.get("completion_tokens")
    report.check(isinstance(prompt_tokens, int) and prompt_tokens > 30000, "Rust prompt token count too low")
    report.check(
        isinstance(completion_tokens, int) and 0 < completion_tokens <= DEFAULT_MAX_TOKENS,
        "Rust completion token count drift",
    )
    report.check(obj.get("finish_reason") in {"length", "stop"}, "Rust finish reason drift")
    report.check(obj.get("cache_read_tokens") == 0, "Rust cache-read accounting drift")
    report.check(obj.get("cache_write_tokens") == prompt_tokens, "Rust cache-write accounting drift")
    report.check(obj.get("live_tokens_before") == 0, "Rust live token probe drift")
    report.check(obj.get("live_prompt_common") == 0, "Rust live prompt-common probe drift")
    text = obj.get("generated_text")
    report.check(isinstance(text, str) and bool(text.strip()), "Rust generated text missing")
    if isinstance(text, str):
        validate_facts(report, text)


def validate_facts(report: Report, text: str) -> None:
    for name, value in FACTS:
        ok, message = output_has_fact(text, name, value)
        report.check(ok, message)


def output_has_fact(text: str, name: str, expected: int) -> tuple[bool, str]:
    name_len = len(name)
    pos = 0
    saw_wrong_assignment = False
    wrong_value = -1
    while True:
        idx = text.find(name, pos)
        if idx < 0:
            break
        before = text[idx - 1] if idx > 0 else ""
        after_idx = idx + name_len
        after = text[after_idx] if after_idx < len(text) else ""
        before_ok = idx == 0 or is_name_boundary(before)
        after_ok = is_name_boundary(after) or after in {" ", "\t", "="}
        if before_ok and after_ok:
            parsed = parse_assignment_value(text, after_idx)
            if parsed is not None:
                if parsed == expected:
                    return True, ""
                saw_wrong_assignment = True
                wrong_value = parsed
        pos = idx + name_len
    if saw_wrong_assignment:
        return False, f"generated text wrong assignment for {name}: got {wrong_value} expected {expected}"
    return False, f"generated text missing assignment for {name}={expected}"


def is_name_boundary(ch: str) -> bool:
    return ch == "" or not (ch.isalnum() or ch == "_")


def parse_assignment_value(text: str, pos: int) -> int | None:
    while pos < len(text) and text[pos] in {" ", "\t"}:
        pos += 1
    if pos >= len(text) or text[pos] != "=":
        return None
    pos += 1
    while pos < len(text) and text[pos] in {" ", "\t"}:
        pos += 1
    if pos >= len(text) or not text[pos].isdigit():
        return None
    value = 0
    while pos < len(text) and text[pos].isdigit():
        value = value * 10 + int(text[pos])
        pos += 1
    return value


def validate_static_wiring(report: Report) -> None:
    files = {
        "cargo": ROOT / "rust/ds4-engine/Cargo.toml",
        "binary": ROOT / "rust/ds4-engine/src/bin/ds4-runtime-long-context-rs.rs",
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "readme": ROOT / "ds4-parity/README.md",
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
    }
    texts = {name: path.read_text() for name, path in files.items()}
    report.check("ds4-runtime-long-context-rs" in texts["cargo"], "Cargo binary entry missing")
    report.check(RUST_SCHEMA in texts["binary"], "Rust binary missing schema metadata")
    report.check("ServerGenerationOptions" in texts["binary"], "Rust binary missing generation options")
    report.check("DEFAULT_CTX_SIZE" in texts["binary"], "Rust binary missing context default")
    report.check("cache_write_tokens" in texts["binary"], "Rust binary missing cache accounting")
    report.check("run_runtime_graph_long_context.py" in texts["report"], "unified report missing M10.9d comparator")
    report.check("M10.9d Runtime graph long-context gate" in texts["readme"], "README missing M10.9d entry")
    report.check("M10.9d: B300 Long-Context Rust Runtime Gate" in texts["roadmap"], "roadmap missing M10.9d")
    report.check("M10.9d: B300 Long-Context Rust Runtime Gate" in texts["todo"], "TODO missing M10.9d")
    report.check("M10.9d B300 Long-Context Rust Runtime Gate" in texts["status"], "status missing M10.9d")


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
    expect_failure("prompt hash drift", lambda obj: obj["prompt"].__setitem__("sha256", "0" * 64))
    expect_failure("current-C exit drift", lambda obj: obj["current_c"].__setitem__("exit_code", 1))
    expect_failure("Rust prompt token count too low", lambda obj: update_rust_json(obj, {"prompt_tokens": 64}))
    expect_failure("Rust fact drift", lambda obj: update_rust_json(obj, {"generated_text": wrong_fact_text()}))
    expect_failure("Rust fallback marker", add_rust_fallback_marker)
    expect_failure("cache accounting drift", lambda obj: update_rust_json(obj, {"cache_read_tokens": 12}))
    return report


def wrong_fact_text() -> str:
    return "\n".join(f"{name} = {value + (1 if name == 'Bob' else 0)}" for name, value in FACTS)


def update_rust_json(summary: dict[str, Any], updates: dict[str, Any]) -> None:
    rust = summary["rust"]
    candidate = rust["json"]
    candidate.update(updates)
    rust["stdout"] = blob(json.dumps(candidate, indent=2) + "\n")


def add_rust_fallback_marker(summary: dict[str, Any]) -> None:
    rust = summary["rust"]
    raw = base64.b64decode(rust["stderr"]["base64"]).decode("utf-8", "replace")
    rust["stderr"] = blob(raw + "\nds4: runtime graph route not implemented yet; target-stream fallback\n")


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def tail(text: str, limit: int = 20) -> str:
    return "\n".join(text.splitlines()[-limit:])


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
