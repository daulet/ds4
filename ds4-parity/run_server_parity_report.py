#!/usr/bin/env python3
"""Run the local DS4 Milestone 9 server/runtime parity report."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 600

KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
KUBE_CONTEXT = "hou2-prod1"
KUBE_NAMESPACE = "default"
KUBE_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
B300_MODEL = (
    "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@dataclass
class ReportItem:
    name: str
    kind: str
    command: list[str] | None = None
    status: str = "PENDING"
    exit_code: int | None = None
    summary: str = ""
    reason: str = ""
    rerun_command: str = ""
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "SKIP"}


class ServerParityReport:
    def __init__(self, root: Path, timeout: int) -> None:
        self.root = root.resolve()
        self.timeout = timeout
        self.items: list[ReportItem] = []

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def run(self) -> None:
        for name, command in model_free_commands():
            item = ReportItem(name=name, kind="model-free-server-test", command=command)
            self.items.append(item)
            self.run_command(item)
        for name, command in comparator_commands():
            item = ReportItem(name=name, kind="server-comparator", command=command)
            self.items.append(item)
            self.run_command(item)
        self.items.extend(b300_skip_items())

    def run_command(self, item: ReportItem) -> None:
        assert item.command is not None
        try:
            proc = subprocess.run(
                item.command,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            item.status = "FAIL"
            item.reason = f"command not found: {exc.filename}"
            item.rerun_command = shell_join(item.command)
            return
        except subprocess.TimeoutExpired as exc:
            item.status = "FAIL"
            item.reason = f"timed out after {self.timeout}s"
            item.rerun_command = shell_join(item.command)
            item.stdout_tail = tail_lines(exc.stdout or "")
            item.stderr_tail = tail_lines(exc.stderr or "")
            return

        item.exit_code = proc.returncode
        item.status = "PASS" if proc.returncode == 0 else "FAIL"
        item.rerun_command = shell_join(item.command)
        item.stdout_tail = tail_lines(proc.stdout)
        item.stderr_tail = tail_lines(proc.stderr)
        item.summary = extract_summary(proc.stdout) or extract_summary(proc.stderr)
        if item.status == "PASS" and item.kind == "model-free-server-test":
            passed = extract_passed_test_count(proc.stdout) or extract_passed_test_count(
                proc.stderr
            )
            if passed is None:
                item.status = "FAIL"
                item.reason = "could not parse cargo test pass count"
            elif passed == 0:
                item.status = "FAIL"
                item.reason = "cargo test filter matched zero tests"
        if proc.returncode != 0:
            item.reason = f"exit status {proc.returncode}"

    def report_text(self) -> str:
        lines = [
            "DS4 Milestone 9 server/runtime parity report",
            f"root: {self.root}",
            f"timeout_seconds: {self.timeout}",
        ]
        for item in self.items:
            command = shell_join(item.command) if item.command else item.rerun_command
            lines.extend(
                [
                    f"[{item.status}] {item.name}",
                    f"  kind: {item.kind}",
                    f"  command: {command}",
                ]
            )
            if item.exit_code is not None:
                lines.append(f"  exit_code: {item.exit_code}")
            if item.summary:
                lines.append(f"  summary: {item.summary}")
            if item.reason:
                lines.append(f"  reason: {item.reason}")
            if item.status == "SKIP" and item.rerun_command:
                lines.append(f"  rerun: {item.rerun_command}")
            if item.status == "FAIL":
                append_tail(lines, "stdout_tail", item.stdout_tail)
                append_tail(lines, "stderr_tail", item.stderr_tail)
        passed = sum(1 for item in self.items if item.status == "PASS")
        skipped = sum(1 for item in self.items if item.status == "SKIP")
        failed = sum(1 for item in self.items if item.status == "FAIL")
        lines.append(f"summary: {passed} passed, {skipped} skipped, {failed} failed")
        return "\n".join(lines) + "\n"

    def report_json(self) -> str:
        payload = {
            "root": str(self.root),
            "ok": self.ok,
            "timeout_seconds": self.timeout,
            "items": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "status": item.status,
                    "command": item.command,
                    "exit_code": item.exit_code,
                    "summary": item.summary,
                    "reason": item.reason,
                    "rerun_command": item.rerun_command,
                    "stdout_tail": item.stdout_tail,
                    "stderr_tail": item.stderr_tail,
                }
                for item in self.items
            ],
        }
        return json.dumps(payload, indent=2) + "\n"


def model_free_commands() -> list[tuple[str, list[str]]]:
    return [
        (
            "Rust runtime server route/cache tests",
            [
                "cargo",
                "test",
                "-p",
                "ds4-engine",
                "--bin",
                "ds4-server-runtime-rs",
            ],
        ),
        (
            "Rust server request parsing and prompt tests",
            ["cargo", "test", "-p", "ds4-gguf", "--lib", "server_chat"],
        ),
        (
            "Rust server response formatter tests",
            ["cargo", "test", "-p", "ds4-gguf", "--lib", "server_response"],
        ),
        (
            "Rust server HTTP framing tests",
            ["cargo", "test", "-p", "ds4-gguf", "--lib", "server_http"],
        ),
        (
            "Rust server no-model dispatch tests",
            ["cargo", "test", "-p", "ds4-gguf", "--lib", "server_no_model"],
        ),
        (
            "Rust no-model socket route tests",
            ["cargo", "test", "-p", "ds4-gguf", "--test", "no_model_server"],
        ),
    ]


def comparator_commands() -> list[tuple[str, list[str]]]:
    return [
        (
            "M0.4/M0.5 server and KV artifact comparator",
            [sys.executable, "ds4-parity/compare_server_kv.py"],
        ),
        (
            "M0.4/M0.5 server and KV artifact negative tests",
            [sys.executable, "ds4-parity/compare_server_kv.py", "--negative-test"],
        ),
        (
            "M9 runtime KV replay policy comparator",
            [sys.executable, "ds4-parity/compare_kv_replay.py", "--negative-test"],
        ),
        (
            "M9.8f5 B300 Rust runtime replay summary and M10.7d2 ledger contract",
            [sys.executable, "ds4-parity/check_runtime_kv_replay_summary.py", "--negative-test"],
        ),
    ]


def b300_skip_items() -> list[ReportItem]:
    return [
        ReportItem(
            name="B300 Rust runtime M0.4 server replay refresh",
            kind="b300-rust-runtime",
            status="SKIP",
            reason="model-backed server replay requires the B300 pod and model",
            rerun_command=b300_exec(b300_m04_runtime_replay_script()),
        ),
        ReportItem(
            name="B300 Rust runtime M0.5 KV replay refresh",
            kind="b300-rust-runtime",
            status="SKIP",
            reason=(
                "model-backed three-lifetime KV replay requires the B300 pod; "
                "the checked-in M9.8f5 summary records the latest successful run"
            ),
            rerun_command=b300_exec(b300_m05_runtime_replay_script()),
        ),
        ReportItem(
            name="B300 ds4_test --server oracle refresh",
            kind="b300-current-c-oracle",
            status="SKIP",
            reason=(
                "`ds4_test --server` is model-backed and C-harness-only; run it "
                "on B300 when refreshing the current-C oracle artifacts"
            ),
            rerun_command=b300_exec(
                "cd /workspace/ds4 && make ds4_test && "
                f"DS4_TEST_MODEL={B300_MODEL} ./ds4_test --server"
            ),
        ),
    ]


def b300_rust_env() -> str:
    return "PATH=/tmp/cargo/bin:$PATH CARGO_HOME=/tmp/cargo RUSTUP_HOME=/tmp/rustup"


def b300_runtime_build_command() -> str:
    return (
        f"{b300_rust_env()} cargo build -p ds4-engine "
        "--bin ds4-server-runtime-rs"
    )


def b300_runtime_server_command(
    *,
    root: str,
    port: str,
    tokens: int,
    trace: str,
    kv_dir: str,
) -> str:
    return (
        f"{b300_rust_env()} target/debug/ds4-server-runtime-rs "
        f"-m {B300_MODEL} --cuda --ctx 32768 -n {tokens} "
        f"--host 127.0.0.1 --port {port} "
        f"--trace {trace} --kv-disk-dir {kv_dir} "
        "--kv-disk-space-mb 512 --kv-cache-min-tokens 512 "
        "--kv-cache-cold-max-tokens 30000 "
        "--kv-cache-continued-interval-tokens 0"
    )


def b300_m04_runtime_replay_script() -> str:
    root = "/tmp/ds4-m99-server-m04"
    server = b300_runtime_server_command(
        root=root,
        port="18080",
        tokens=64,
        trace=f"{root}/traces/server.trace",
        kv_dir=f"{root}/kv",
    )
    requests = [
        ("chat_basic", "json"),
        ("chat_stream", "sse"),
        ("chat_tool_call", "json"),
        ("chat_thinking_disabled", "json"),
        ("chat_cache_seed", "json"),
        ("chat_cache_continuation", "json"),
    ]
    replay_steps = [
        (
            "("
            "code=$(curl -sS -w '%{http_code}' "
            f"-o {root}/responses/models.json http://127.0.0.1:18080/v1/models); "
            f"echo models http_code=$code >>{root}/logs/replay.log; "
            'test "$code" = 200'
            ")"
        )
    ]
    for name, extension in requests:
        output = f"{root}/responses/{name}.{'sse' if extension == 'sse' else 'json'}"
        fixture = f"ds4-parity/baselines/server-fixtures/m0.4/{name}.json"
        replay_steps.append(
            (
                "("
                "code=$(curl -sS -w '%{http_code}' "
                f"-o {output} http://127.0.0.1:18080/v1/chat/completions "
                "-H 'Content-Type: application/json' "
                f"--data-binary @{fixture}); "
                f"echo {name} http_code=$code response={output} >>{root}/logs/replay.log; "
                'test "$code" = 200'
                ")"
            )
        )
    json_checks = [f"python3 -m json.tool {root}/responses/models.json >/dev/null"]
    json_checks.extend(
        f"python3 -m json.tool {root}/responses/{name}.json >/dev/null"
        for name, extension in requests
        if extension == "json"
    )
    steps = [
        "set -e",
        "cd /workspace/ds4",
        f"rm -rf {root}",
        f"mkdir -p {root}/responses {root}/traces {root}/kv {root}/logs",
        b300_runtime_build_command(),
        f"{server} >{root}/logs/server.log 2>&1 & pid=$!",
        "trap 'kill $pid 2>/dev/null || true' EXIT",
        "sleep 5",
        " && ".join(replay_steps),
        "kill $pid",
        "(wait $pid || true)",
        "trap - EXIT",
        " && ".join(json_checks),
        f"grep -q 'data: \\[DONE\\]' {root}/responses/chat_stream.sse",
        f"sha256sum {root}/responses/* {root}/traces/server.trace",
    ]
    return " && ".join(steps)


def b300_m05_runtime_replay_script() -> str:
    root = "/tmp/ds4-m99-server-m05"
    replay_script = (
        "run_case() { "
        "name=$1; fixture=$2; port=$3; "
        f"{b300_runtime_server_command(root=root, port='${port}', tokens=16, trace=f'{root}/traces/${{name}}.trace', kv_dir=f'{root}/kv')} "
        f">{root}/logs/${{name}}.server.log 2>&1 & pid=$!; "
        "trap 'kill $pid 2>/dev/null || true' EXIT; "
        "sleep 5; "
        "code=$(curl -sS -w '%{http_code}' "
        f"-o {root}/responses/${{name}}.json http://127.0.0.1:${{port}}/v1/chat/completions "
        "-H 'Content-Type: application/json' "
        "--data-binary @${fixture}); "
        f"echo ${{name}} http_code=$code >>{root}/logs/replay.log; "
        "kill $pid; wait $pid || true; "
        "trap - EXIT; "
        'test "$code" = 200; '
        "}; "
        "run_case seed_miss ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json 18210 && "
        "run_case seed_restore ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json 18211 && "
        "run_case continuation_restore ds4-parity/baselines/kv-fixtures/m0.5/kv_continuation.json 18212"
    )
    checks = [
        f"python3 -m json.tool {root}/responses/seed_miss.json >/dev/null",
        f"python3 -m json.tool {root}/responses/seed_restore.json >/dev/null",
        f"python3 -m json.tool {root}/responses/continuation_restore.json >/dev/null",
        f"sha256sum {root}/kv/*.kv",
    ]
    steps = [
        "set -e",
        "cd /workspace/ds4",
        f"rm -rf {root}",
        f"mkdir -p {root}/responses {root}/traces {root}/kv {root}/logs",
        b300_runtime_build_command(),
        replay_script,
        " && ".join(checks),
    ]
    return " && ".join(steps)


def b300_exec(command: str) -> str:
    return shell_join(
        [
            "kubectl",
            "--kubeconfig",
            KUBECONFIG,
            "--context",
            KUBE_CONTEXT,
            "-n",
            KUBE_NAMESPACE,
            "exec",
            KUBE_POD,
            "--",
            "sh",
            "-lc",
            command,
        ]
    )


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def tail_lines(text: str, limit: int = 20) -> list[str]:
    return text.splitlines()[-limit:]


def append_tail(lines: list[str], label: str, tail: list[str]) -> None:
    if not tail:
        return
    lines.append(f"  {label}:")
    for line in tail:
        lines.append(f"    {line}")


def extract_summary(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.startswith("summary:"):
            return line
        if "test result:" in line:
            return line.strip()
    return ""


def extract_passed_test_count(text: str) -> int | None:
    for line in reversed(text.splitlines()):
        match = re.search(r"test result: ok\. (\d+) passed;", line)
        if match:
            return int(match.group(1))
    return None


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (default: parent of ds4-parity/)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-command timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    report = ServerParityReport(args.root, args.timeout)
    report.run()
    sys.stdout.write(report.report_json() if args.json else report.report_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
