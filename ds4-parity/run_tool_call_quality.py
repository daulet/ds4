#!/usr/bin/env python3
"""Run the Rust server-runtime tool-call quality check.

This mirrors the C `./ds4_test --tool-call-quality` surface at the HTTP
runtime boundary: run fast and exact/quality engines, send the deterministic
tool-call request, classify the parsed OpenAI response, and preserve raw
artifacts for any drift investigation.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json"
SCHEMA = "ds4.runtime_graph_tool_server_summary.v1"
SOURCE = "m10.9e-runtime-graph-tool-server"
MILESTONE = "M10.9e"
NEXT_STAGE = "M10.9f"
DEFAULT_SERVER_BIN = ROOT / "target" / "debug" / "ds4-server-runtime-rs"
DEFAULT_CURRENT_C_BIN = ROOT / "ds4_test"
DEFAULT_OUT_DIR = ROOT / "target" / "tool-call-quality"
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_BYTES = 86720111488
EXPECTED_TOOL_NAME = "list_files"
EXPECTED_ARGUMENTS = {"path": "."}
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.05
DEFAULT_RUNTIME_GRAPH_ROUTE = "graph"
B300_MARKER = "CUDA backend initialized on NVIDIA B300 SXM6 AC"


QUALITY_REQUEST: dict[str, Any] = {
    "model": "deepseek-v4-flash",
    "messages": [
        {
            "role": "user",
            "content": "List the files in the current directory. Use the provided tool; do not answer in prose.",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list.",
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "think": False,
    "temperature": 0,
    "seed": 123,
    "max_tokens": 256,
    "stream": False,
}


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    quality: bool


@dataclass
class CaseResult:
    case_id: str
    quality: bool
    ok: bool
    category: str
    command: list[str]
    http_status: int | None
    finish_reason: str | None
    tool_name: str | None
    arguments: str | None
    artifact_dir: str
    request_path: str
    response_path: str
    headers_path: str
    trace_path: str
    stdout_path: str
    stderr_path: str
    detail: str = ""


@dataclass(frozen=True)
class HttpResult:
    status: int
    reason: str
    headers: list[tuple[str, str]]
    body: bytes


class QualityError(RuntimeError):
    pass


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


def compact_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def quality_cases(selection: str) -> list[QualityCase]:
    cases = [
        QualityCase("fast", False),
        QualityCase("exact", True),
    ]
    if selection == "both":
        return cases
    return [case for case in cases if case.case_id == selection]


def server_command(
    server_bin: Path,
    model: Path,
    backend: str,
    runtime_graph_route: str,
    host: str,
    port: int,
    trace_path: Path,
    ctx: int,
    quality: bool,
) -> list[str]:
    command = [
        str(server_bin),
        "-m",
        str(model),
        "--backend",
        backend,
        "--runtime-graph",
        runtime_graph_route,
        "--host",
        host,
        "--port",
        str(port),
        "--trace",
        str(trace_path),
        "--ctx",
        str(ctx),
    ]
    if quality:
        command.append("--quality")
    return command


def request_json() -> dict[str, Any]:
    return copy.deepcopy(QUALITY_REQUEST)


def wait_ready(
    host: str, port: int, proc: subprocess.Popen[bytes], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise QualityError(
                f"server exited before ready with status {proc.returncode}"
            )
        try:
            result = http_request(host, port, "GET", "/v1/models", None)
            if result.status == 200:
                return
            last_error = f"GET /v1/models returned HTTP {result.status}"
        except OSError as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise QualityError(f"server did not become ready: {last_error}")


def http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes | None,
) -> HttpResult:
    conn = http.client.HTTPConnection(host, port, timeout=30)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        return HttpResult(
            status=response.status,
            reason=response.reason,
            headers=response.getheaders(),
            body=payload,
        )
    finally:
        conn.close()


def write_headers(path: Path, result: HttpResult) -> None:
    lines = [f"HTTP/1.1 {result.status} {result.reason}"]
    lines.extend(f"{name}: {value}" for name, value in result.headers)
    path.write_text("\n".join(lines) + "\n")


def classify_response(
    status: int, body: bytes
) -> tuple[bool, str, str | None, str | None, str | None, str]:
    if status != 200:
        return False, "http_error", None, None, None, f"HTTP {status}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return False, "invalid_json", None, None, None, str(exc)
    if not isinstance(payload, dict):
        return (
            False,
            "invalid_json",
            None,
            None,
            None,
            "top-level response is not an object",
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "missing_choice", None, None, None, "missing choices[0]"
    choice = choices[0]
    if not isinstance(choice, dict):
        return False, "missing_choice", None, None, None, "choices[0] is not an object"
    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        return (
            False,
            "missing_message",
            finish_reason,
            None,
            None,
            "missing assistant message",
        )
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False, "missing_tool_call", finish_reason, None, None, "no tool calls"
    first = tool_calls[0]
    if not isinstance(first, dict):
        return (
            False,
            "missing_tool_call",
            finish_reason,
            None,
            None,
            "tool_calls[0] is not an object",
        )
    function = first.get("function")
    if not isinstance(function, dict):
        return (
            False,
            "missing_function",
            finish_reason,
            None,
            None,
            "missing function payload",
        )
    tool_name = function.get("name")
    arguments = function.get("arguments")
    if tool_name != EXPECTED_TOOL_NAME:
        return (
            False,
            "wrong_tool_name",
            finish_reason,
            as_optional_str(tool_name),
            as_optional_str(arguments),
            f"expected {EXPECTED_TOOL_NAME}",
        )
    if not isinstance(arguments, str):
        return (
            False,
            "invalid_arguments",
            finish_reason,
            tool_name,
            None,
            "arguments is not a string",
        )
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return False, "invalid_arguments", finish_reason, tool_name, arguments, str(exc)
    if parsed_arguments != EXPECTED_ARGUMENTS:
        return (
            False,
            "wrong_arguments",
            finish_reason,
            tool_name,
            arguments,
            f"expected {EXPECTED_ARGUMENTS}",
        )
    if finish_reason != "tool_calls":
        return (
            False,
            "wrong_finish_reason",
            as_optional_str(finish_reason),
            tool_name,
            arguments,
            "expected tool_calls",
        )
    return True, "ok", finish_reason, tool_name, arguments, ""


def as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def run_case(
    case: QualityCase,
    server_bin: Path,
    model: Path,
    backend: str,
    runtime_graph_route: str,
    host: str,
    port: int,
    ctx: int,
    out_dir: Path,
    ready_timeout: float,
) -> CaseResult:
    case_dir = out_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    request_path = case_dir / "request.json"
    response_path = case_dir / "response.json"
    headers_path = case_dir / "headers.txt"
    trace_path = case_dir / "trace.txt"
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    request_body = compact_json(request_json())
    request_path.write_bytes(request_body + b"\n")
    command = server_command(
        server_bin,
        model,
        backend,
        runtime_graph_route,
        host,
        port,
        trace_path,
        ctx,
        case.quality,
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        try:
            wait_ready(host, port, proc, ready_timeout)
            http_result = http_request(
                host, port, "POST", "/v1/chat/completions", request_body
            )
            response_path.write_bytes(http_result.body)
            write_headers(headers_path, http_result)
            ok, category, finish, tool_name, arguments, detail = classify_response(
                http_result.status, http_result.body
            )
            return CaseResult(
                case_id=case.case_id,
                quality=case.quality,
                ok=ok,
                category=category,
                command=command,
                http_status=http_result.status,
                finish_reason=finish,
                tool_name=tool_name,
                arguments=arguments,
                artifact_dir=str(case_dir),
                request_path=str(request_path),
                response_path=str(response_path),
                headers_path=str(headers_path),
                trace_path=str(trace_path),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                detail=detail,
            )
        except Exception as exc:
            return CaseResult(
                case_id=case.case_id,
                quality=case.quality,
                ok=False,
                category="runner_error",
                command=command,
                http_status=None,
                finish_reason=None,
                tool_name=None,
                arguments=None,
                artifact_dir=str(case_dir),
                request_path=str(request_path),
                response_path=str(response_path),
                headers_path=str(headers_path),
                trace_path=str(trace_path),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                detail=str(exc),
            )
        finally:
            stop_process(proc)


def stop_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def write_summary(
    out_dir: Path, results: list[CaseResult], args: argparse.Namespace
) -> dict[str, Any]:
    payload = {
        "oracle": "./ds4_test --tool-call-quality",
        "rust_runner": "ds4-server-runtime-rs",
        "model": str(args.model),
        "backend": args.backend,
        "runtime_graph_route": args.runtime_graph,
        "ctx": args.ctx,
        "request_controls": {
            "temperature": QUALITY_REQUEST["temperature"],
            "top_k": DEFAULT_TOP_K,
            "top_p": DEFAULT_TOP_P,
            "min_p": DEFAULT_MIN_P,
            "seed": QUALITY_REQUEST["seed"],
            "max_tokens": QUALITY_REQUEST["max_tokens"],
            "stream": QUALITY_REQUEST["stream"],
            "sampling_note": "top_k/top_p/min_p use C and Rust OpenAI defaults omitted from the request body",
            "expected_tool_name": EXPECTED_TOOL_NAME,
            "expected_arguments": EXPECTED_ARGUMENTS,
        },
        "ok": all(result.ok for result in results),
        "cases": [asdict(result) for result in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "summary.txt").write_text(format_report(results) + "\n")
    return payload


def write_artifact_summary(
    path: Path,
    out_dir: Path,
    results: list[CaseResult],
    args: argparse.Namespace,
    current_c: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    model = args.model.resolve()
    request_body = compact_json(request_json())
    current_c_env = {"DS4_TEST_MODEL": str(args.model)}
    payload = {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "parent": "M10.9",
        "next_stage": NEXT_STAGE,
        "runtime_graph_route": args.runtime_graph,
        "backend": args.backend,
        "ctx": args.ctx,
        "oracle": "./ds4_test --tool-call-quality",
        "rust_runner": "ds4-server-runtime-rs",
        "model": {
            "path": str(args.model),
            "resolved_path": str(model),
            "sha256": sha256_file(model),
            "bytes": model.stat().st_size,
            "expected_sha256": EXPECTED_MODEL_SHA256,
        },
        "request": {
            "body": blob_bytes(request_body),
            "controls": {
                "temperature": QUALITY_REQUEST["temperature"],
                "top_k": DEFAULT_TOP_K,
                "top_p": DEFAULT_TOP_P,
                "min_p": DEFAULT_MIN_P,
                "seed": QUALITY_REQUEST["seed"],
                "max_tokens": QUALITY_REQUEST["max_tokens"],
                "stream": QUALITY_REQUEST["stream"],
                "expected_tool_name": EXPECTED_TOOL_NAME,
                "expected_arguments": EXPECTED_ARGUMENTS,
            },
        },
        "current_c": {
            "command": [str(args.current_c_bin), "--tool-call-quality"],
            "env": current_c_env,
            "exit_code": current_c.returncode,
            "stdout": blob(current_c.stdout),
            "stderr": blob(current_c.stderr),
        },
        "rust": {
            "server_bin": str(args.server_bin),
            "out_dir": str(out_dir),
            "ok": all(result.ok for result in results),
            "cases": [artifact_case(result) for result in results],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def artifact_case(result: CaseResult) -> dict[str, Any]:
    response_json: Any = None
    response_error = ""
    response_bytes = Path(result.response_path).read_bytes() if Path(result.response_path).exists() else b""
    if response_bytes:
        try:
            response_json = json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            response_error = str(exc)
    return {
        **asdict(result),
        "response_json": response_json,
        "response_json_error": response_error,
        "artifacts": {
            "request": blob_file(Path(result.request_path)),
            "response": blob_bytes(response_bytes),
            "headers": blob_file(Path(result.headers_path)),
            "trace": blob_file(Path(result.trace_path)),
            "stdout": blob_file(Path(result.stdout_path)),
            "stderr": blob_file(Path(result.stderr_path)),
        },
    }


def run_current_c(args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["DS4_TEST_MODEL"] = str(args.model)
    return subprocess.run(
        [str(args.current_c_bin), "--tool-call-quality"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        check=False,
    )


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


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
    return blob_bytes(text.encode("utf-8"))


def blob_file(path: Path) -> dict[str, Any]:
    return blob_bytes(path.read_bytes() if path.exists() else b"")


def blob_bytes(raw: bytes) -> dict[str, Any]:
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


def format_report(results: list[CaseResult]) -> str:
    lines = ["DS4 Rust tool-call quality report"]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(
            f"[{status}] {result.case_id} category={result.category} "
            f"http={result.http_status} tool={result.tool_name} finish={result.finish_reason}"
        )
        if result.detail:
            lines.append(f"  detail: {result.detail}")
        lines.append(f"  artifacts: {result.artifact_dir}")
    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    lines.append(f"summary: {passed} passed, {failed} failed")
    return "\n".join(lines)


def response_payload() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": EXPECTED_TOOL_NAME,
                                "arguments": json.dumps(
                                    EXPECTED_ARGUMENTS, separators=(",", ":")
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def response_body(mutator: Any | None = None) -> bytes:
    payload = response_payload()
    if mutator is not None:
        mutator(payload)
    return compact_json(payload)


def first_function(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["choices"][0]["message"]["tool_calls"][0]["function"]


def run_self_test() -> int:
    cases = [
        ("ok", classify_response(200, response_body())[0:2], (True, "ok")),
        ("http_error", classify_response(503, b"{}")[0:2], (False, "http_error")),
        ("invalid_json", classify_response(200, b"{")[0:2], (False, "invalid_json")),
        (
            "missing_choice",
            classify_response(200, compact_json({"choices": []}))[0:2],
            (False, "missing_choice"),
        ),
        (
            "missing_message",
            classify_response(
                200,
                response_body(lambda payload: payload["choices"][0].pop("message")),
            )[0:2],
            (False, "missing_message"),
        ),
        (
            "missing_tool_call",
            classify_response(
                200,
                response_body(
                    lambda payload: payload["choices"][0]["message"].pop("tool_calls")
                ),
            )[0:2],
            (False, "missing_tool_call"),
        ),
        (
            "missing_function",
            classify_response(
                200,
                response_body(
                    lambda payload: payload["choices"][0]["message"]["tool_calls"][
                        0
                    ].pop("function")
                ),
            )[0:2],
            (False, "missing_function"),
        ),
        (
            "wrong_tool_name",
            classify_response(
                200,
                response_body(
                    lambda payload: first_function(payload).__setitem__(
                        "name", "wrong_tool"
                    )
                ),
            )[0:2],
            (False, "wrong_tool_name"),
        ),
        (
            "invalid_arguments",
            classify_response(
                200,
                response_body(
                    lambda payload: first_function(payload).__setitem__(
                        "arguments", "{"
                    )
                ),
            )[0:2],
            (False, "invalid_arguments"),
        ),
        (
            "wrong_arguments",
            classify_response(
                200,
                response_body(
                    lambda payload: first_function(payload).__setitem__(
                        "arguments", json.dumps({"path": "/tmp"}, separators=(",", ":"))
                    )
                ),
            )[0:2],
            (False, "wrong_arguments"),
        ),
        (
            "wrong_finish_reason",
            classify_response(
                200,
                response_body(
                    lambda payload: payload["choices"][0].__setitem__(
                        "finish_reason", "stop"
                    )
                ),
            )[0:2],
            (False, "wrong_finish_reason"),
        ),
    ]
    request = compact_json(request_json()).decode("utf-8")
    cases.append(
        (
            "seed_control",
            (('"seed":123' in request), "seed_control"),
            (True, "seed_control"),
        )
    )
    for label, actual, expected in cases:
        if actual != expected:
            print(
                f"self-test case {label} failed: expected {expected}, got {actual}",
                file=sys.stderr,
            )
            return 1
    print("self-test: PASS")
    return 0


def validate_summary(report: Report, obj: Any) -> None:
    root = require_dict(report, obj, "summary")
    report.check(root.get("schema") == SCHEMA, "summary schema drift")
    report.check(root.get("source") == SOURCE, "summary source drift")
    report.check(root.get("milestone") == MILESTONE, "summary milestone drift")
    report.check(root.get("next_stage") == NEXT_STAGE, "summary next stage drift")
    report.check(root.get("runtime_graph_route") == DEFAULT_RUNTIME_GRAPH_ROUTE, "runtime graph route drift")
    report.check(root.get("backend") == "cuda", "backend drift")
    report.check(root.get("ctx") == 32768, "context length drift")

    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model sha256 drift")
    report.check(model.get("expected_sha256") == EXPECTED_MODEL_SHA256, "model expected sha256 drift")
    report.check(model.get("bytes") == EXPECTED_MODEL_BYTES, "model byte size drift")

    request = require_dict(report, root.get("request"), "request")
    request_body = unblob(report, request.get("body"), "request.body")
    report.check(request_body == compact_json(request_json()), "request body drift")
    controls = require_dict(report, request.get("controls"), "request.controls")
    report.check(controls == expected_controls(), "request controls drift")

    validate_current_c(report, root.get("current_c"))
    rust = require_dict(report, root.get("rust"), "rust")
    report.check(rust.get("ok") is True, "Rust aggregate ok drift")
    cases = require_list(report, rust.get("cases"), "rust.cases")
    report.check([case.get("case_id") for case in cases if isinstance(case, dict)] == ["fast", "exact"], "case order drift")
    for expected_id, expected_quality in (("fast", False), ("exact", True)):
        case = next(
            (case for case in cases if isinstance(case, dict) and case.get("case_id") == expected_id),
            None,
        )
        validate_case(report, require_dict(report, case, f"case {expected_id}"), expected_id, expected_quality)
    validate_static_wiring(report)


def expected_controls() -> dict[str, Any]:
    return {
        "temperature": QUALITY_REQUEST["temperature"],
        "top_k": DEFAULT_TOP_K,
        "top_p": DEFAULT_TOP_P,
        "min_p": DEFAULT_MIN_P,
        "seed": QUALITY_REQUEST["seed"],
        "max_tokens": QUALITY_REQUEST["max_tokens"],
        "stream": QUALITY_REQUEST["stream"],
        "expected_tool_name": EXPECTED_TOOL_NAME,
        "expected_arguments": EXPECTED_ARGUMENTS,
    }


def validate_current_c(report: Report, obj: Any) -> None:
    current_c = require_dict(report, obj, "current_c")
    report.check(current_c.get("exit_code") == 0, "current-C tool quality exit code drift")
    env = require_dict(report, current_c.get("env"), "current_c.env")
    report.check(isinstance(env.get("DS4_TEST_MODEL"), str), "current-C model env missing")
    stdout = unblob(report, current_c.get("stdout"), "current_c.stdout").decode("utf-8", "replace")
    stderr = unblob(report, current_c.get("stderr"), "current_c.stderr").decode("utf-8", "replace")
    report.check("ds4 tests: ok" in stdout, "current-C stdout missing pass marker")
    report.check("tool-call-quality:" in stderr, "current-C stderr missing tool-call section")
    report.check("tool-call quality fast path" in stderr, "current-C stderr missing fast path marker")
    report.check("tool-call quality exact path" in stderr, "current-C stderr missing exact path marker")
    report.check(B300_MARKER in stderr, "current-C stderr missing B300 CUDA marker")
    for marker in ("tool-call-quality: ERR", "missing", "wrong", "failure(s)", "TEST failed"):
        report.check(marker not in stderr, f"current-C stderr has failure marker: {marker}")


def validate_case(report: Report, case: dict[str, Any], case_id: str, quality: bool) -> None:
    report.check(case.get("quality") is quality, f"{case_id}: quality flag drift")
    report.check(case.get("ok") is True, f"{case_id}: ok drift")
    report.check(case.get("category") == "ok", f"{case_id}: category drift")
    report.check(case.get("http_status") == 200, f"{case_id}: HTTP status drift")
    report.check(case.get("finish_reason") == "tool_calls", f"{case_id}: finish reason drift")
    report.check(case.get("tool_name") == EXPECTED_TOOL_NAME, f"{case_id}: tool name drift")
    arguments = case.get("arguments")
    report.check(isinstance(arguments, str), f"{case_id}: arguments missing")
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            report.check(False, f"{case_id}: arguments JSON invalid: {exc}")
        else:
            report.check(parsed_arguments == EXPECTED_ARGUMENTS, f"{case_id}: arguments drift")
    command = require_list(report, case.get("command"), f"{case_id}.command")
    report.check("--runtime-graph" in command, f"{case_id}: runtime graph selector missing")
    if "--runtime-graph" in command:
        report.check(command[command.index("--runtime-graph") + 1] == DEFAULT_RUNTIME_GRAPH_ROUTE, f"{case_id}: runtime graph command drift")
    report.check("--backend" in command and "cuda" in command, f"{case_id}: backend command drift")
    report.check(("--quality" in command) is quality, f"{case_id}: quality command drift")

    artifacts = require_dict(report, case.get("artifacts"), f"{case_id}.artifacts")
    request = unblob(report, artifacts.get("request"), f"{case_id}.request")
    response = unblob(report, artifacts.get("response"), f"{case_id}.response")
    headers = unblob(report, artifacts.get("headers"), f"{case_id}.headers").decode("utf-8", "replace")
    trace = unblob(report, artifacts.get("trace"), f"{case_id}.trace").decode("utf-8", "replace")
    unblob(report, artifacts.get("stdout"), f"{case_id}.stdout")
    stderr = unblob(report, artifacts.get("stderr"), f"{case_id}.stderr").decode("utf-8", "replace")
    report.check(request == compact_json(request_json()) + b"\n", f"{case_id}: raw request drift")
    report.check(headers.startswith("HTTP/1.1 200"), f"{case_id}: headers missing HTTP 200")
    ok, category, finish, tool_name, args, detail = classify_response(200, response)
    report.check(ok and category == "ok", f"{case_id}: raw response classification drift: {category} {detail}")
    report.check(finish == "tool_calls", f"{case_id}: raw response finish drift")
    report.check(tool_name == EXPECTED_TOOL_NAME, f"{case_id}: raw response tool drift")
    report.check(args == arguments, f"{case_id}: raw response arguments mismatch")
    for marker in (
        "stream: 0\n",
        "tools: 1\n",
        "cache_source:",
        "--- runtime cache ledger ---",
        "finish: tool_calls\n",
        "dsml_start: 1\n",
        "dsml_end: 1\n",
        "tool_call[0]:\n",
        "name: list_files\n",
        "arguments:\n{\"path\": \".\"}\n",
    ):
        report.check(marker in trace, f"{case_id}: trace missing marker {marker!r}")
    report.check(B300_MARKER in stderr, f"{case_id}: stderr missing B300 CUDA marker")
    report.check("not implemented yet" not in stderr, f"{case_id}: graph route not implemented")
    report.check("target-stream" not in stderr, f"{case_id}: target-stream fallback marker present")


def validate_static_wiring(report: Report) -> None:
    files = {
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "readme": ROOT / "ds4-parity/README.md",
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
    }
    texts = {name: path.read_text() for name, path in files.items()}
    report.check("M10.9e Runtime graph tool/server gate" in texts["report"], "unified report missing M10.9e comparator")
    report.check("M10.9e Runtime graph tool/server gate" in texts["readme"], "README missing M10.9e entry")
    report.check("M10.9e: Tool-Call Quality And Server Replay Rust Runtime Gate" in texts["roadmap"], "roadmap missing M10.9e")
    report.check("M10.9e: Tool-Call Quality And Server Replay Rust Runtime Gate" in texts["todo"], "TODO missing M10.9e")
    report.check("M10.9e Tool-Call Quality And Server Replay Rust Runtime Gate" in texts["status"], "status missing M10.9e")


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
    expect_failure("request control drift", lambda obj: obj["request"]["controls"].__setitem__("seed", 456))
    expect_failure("current-C exit drift", lambda obj: obj["current_c"].__setitem__("exit_code", 1))
    expect_failure("case order drift", lambda obj: obj["rust"]["cases"].reverse())
    expect_failure("HTTP status drift", lambda obj: case_obj(obj, "fast").__setitem__("http_status", 500))
    expect_failure("tool name drift", lambda obj: update_case_response(obj, "exact", "wrong_tool", json.dumps(EXPECTED_ARGUMENTS, separators=(",", ":"))))
    expect_failure("trace marker drift", lambda obj: remove_trace_marker(obj, "fast", "finish: tool_calls\n"))
    return report


def update_case_response(summary: dict[str, Any], case_id: str, tool_name: str, arguments: str) -> None:
    case = case_obj(summary, case_id)
    case["tool_name"] = tool_name
    case["arguments"] = arguments
    payload = response_payload()
    function = first_function(payload)
    function["name"] = tool_name
    function["arguments"] = arguments
    raw = compact_json(payload)
    case["response_json"] = payload
    case["artifacts"]["response"] = blob_bytes(raw)


def remove_trace_marker(summary: dict[str, Any], case_id: str, marker: str) -> None:
    case = case_obj(summary, case_id)
    raw = base64.b64decode(case["artifacts"]["trace"]["base64"]).decode("utf-8", "replace")
    case["artifacts"]["trace"] = blob(raw.replace(marker, ""))


def case_obj(summary: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in summary["rust"]["cases"]:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, label: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{label}: expected array")
    return obj if isinstance(obj, list) else []


def print_validation_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--current-c-bin", type=Path, default=DEFAULT_CURRENT_C_BIN)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=["metal", "cuda", "cpu"], default="cuda")
    parser.add_argument("--runtime-graph", default=DEFAULT_RUNTIME_GRAPH_ROUTE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--case", choices=["fast", "exact", "both"], default="both")
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.model is None:
        summary = load_json(args.summary)
        report = Report()
        validate_summary(report, summary)
        print_validation_report("Runtime graph tool/server", report)
        ok = report.ok
        if args.negative_test:
            negative = run_negative_tests(summary)
            print_validation_report("Runtime graph tool/server negative tests", negative)
            ok = ok and negative.ok
        return 0 if ok else 1
    if not args.server_bin.is_file():
        raise SystemExit(f"missing server binary: {args.server_bin}")
    if args.write_summary and not args.current_c_bin.is_file():
        raise SystemExit(f"missing current-C binary: {args.current_c_bin}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(
            case,
            args.server_bin,
            args.model,
            args.backend,
            args.runtime_graph,
            args.host,
            args.port + idx,
            args.ctx,
            args.out_dir,
            args.ready_timeout,
        )
        for idx, case in enumerate(quality_cases(args.case))
    ]
    write_summary(args.out_dir, results, args)
    artifact_ok = True
    if args.write_summary:
        current_c = run_current_c(args)
        artifact = write_artifact_summary(args.write_summary, args.out_dir, results, args, current_c)
        report = Report()
        validate_summary(report, artifact)
        print_validation_report("Runtime graph tool/server", report)
        artifact_ok = report.ok
        if args.negative_test:
            negative = run_negative_tests(artifact)
            print_validation_report("Runtime graph tool/server negative tests", negative)
            artifact_ok = artifact_ok and negative.ok
    print(format_report(results))
    return 0 if all(result.ok for result in results) and artifact_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
