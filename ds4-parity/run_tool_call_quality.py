#!/usr/bin/env python3
"""Run the Rust server-runtime tool-call quality check.

This mirrors the C `./ds4_test --tool-call-quality` surface at the HTTP
runtime boundary: run fast and exact/quality engines, send the deterministic
tool-call request, classify the parsed OpenAI response, and preserve raw
artifacts for any drift investigation.
"""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_BIN = ROOT / "target" / "debug" / "ds4-server-runtime-rs"
DEFAULT_OUT_DIR = ROOT / "target" / "tool-call-quality"
EXPECTED_TOOL_NAME = "list_files"
EXPECTED_ARGUMENTS = {"path": "."}
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.05


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
        server_bin, model, backend, host, port, trace_path, ctx, case.quality
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=["metal", "cuda", "cpu"], default="cuda")
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
        raise SystemExit("--model is required unless --self-test is used")
    if not args.server_bin.is_file():
        raise SystemExit(f"missing server binary: {args.server_bin}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(
            case,
            args.server_bin,
            args.model,
            args.backend,
            args.host,
            args.port + idx,
            args.ctx,
            args.out_dir,
            args.ready_timeout,
        )
        for idx, case in enumerate(quality_cases(args.case))
    ]
    write_summary(args.out_dir, results, args)
    print(format_report(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
