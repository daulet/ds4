#!/usr/bin/env python3
"""Run the Milestone 8 CLI parity report.

The report executes local CLI artifact validators and no-model Rust
comparators. Model-backed current-C refreshes and Rust runtime/PTY comparators
are recorded as B300 skips with exact rerun commands so the local report passes
without `/workspace/ds4/ds4flash.gguf`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
KUBE_CONTEXT = "hou2-prod1"
KUBE_NAMESPACE = "default"
KUBE_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
DEFAULT_TIMEOUT_SECONDS = 600
CARGO_ENV = "CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup PATH=/tmp/ds4-cargo/bin:$PATH"


@dataclass
class ReportItem:
    name: str
    milestone: str
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


class CliParityReport:
    def __init__(self, root: Path, timeout: int) -> None:
        self.root = root.resolve()
        self.timeout = timeout
        self.items: list[ReportItem] = []

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def run(self) -> None:
        for name, milestone, command in local_commands():
            item = ReportItem(name=name, milestone=milestone, kind="local-cli", command=command)
            self.items.append(item)
            self.run_command(item)
        self.items.extend(b300_skip_items(self.root))

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
        if proc.returncode != 0:
            item.reason = f"exit status {proc.returncode}"

    def report_text(self) -> str:
        lines = [
            "DS4 Milestone 8 CLI parity report",
            f"root: {self.root}",
            f"timeout_seconds: {self.timeout}",
        ]
        for item in self.items:
            command = shell_join(item.command) if item.command else item.rerun_command
            lines.extend(
                [
                    f"[{item.status}] {item.milestone} {item.name}",
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
                    "milestone": item.milestone,
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


def local_commands() -> list[tuple[str, str, list[str]]]:
    py = sys.executable
    return [
        (
            "current-C parse/error oracle validator",
            "M8.2",
            [
                py,
                "ds4-parity/check_cli_parse_dump.py",
                "ds4-parity/baselines/cli/m8.2/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.2/manifest.json",
                "--negative-test",
            ],
        ),
        ("Rust parser comparator", "M8.3", [py, "ds4-parity/compare_cli_parse.py", "--negative-test"]),
        (
            "current-C token dump oracle validator",
            "M8.4",
            [
                py,
                "ds4-parity/check_cli_token_dump.py",
                "ds4-parity/baselines/cli/m8.4/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.4/manifest.json",
                "--negative-test",
            ],
        ),
        ("Rust token dump comparator", "M8.5", [py, "ds4-parity/compare_cli_token_dump.py", "--negative-test"]),
        (
            "current-C diagnostics oracle validator",
            "M8.6",
            [
                py,
                "ds4-parity/check_cli_diagnostics_dump.py",
                "ds4-parity/baselines/cli/m8.6/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.6/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "current-C inspect oracle validator",
            "M8.8",
            [
                py,
                "ds4-parity/check_cli_inspect_dump.py",
                "ds4-parity/baselines/cli/m8.8/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.8/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "current-C one-shot generation oracle validator",
            "M8.12a",
            [
                py,
                "ds4-parity/check_cli_generation_dump.py",
                "ds4-parity/baselines/cli/m8.12a/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.12a/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "current-C runtime-controls oracle validator",
            "M8.12b",
            [
                py,
                "ds4-parity/check_cli_runtime_controls_dump.py",
                "ds4-parity/baselines/cli/m8.12b/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.12b/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "current-C interactive PTY oracle validator",
            "M8.14",
            [
                py,
                "ds4-parity/check_cli_interactive_dump.py",
                "ds4-parity/baselines/cli/m8.14/current-c.json",
                "--manifest",
                "ds4-parity/baselines/cli/m8.14/manifest.json",
                "--negative-test",
            ],
        ),
    ]


def b300_skip_items(root: Path) -> list[ReportItem]:
    return [
        manifest_skip(root, "M8.4", "current-C token dump oracle refresh", "m8.4"),
        manifest_skip(root, "M8.6", "current-C diagnostics oracle refresh", "m8.6"),
        manifest_skip(root, "M8.8", "current-C inspect oracle refresh", "m8.8"),
        manifest_skip(root, "M8.12a", "current-C one-shot generation oracle refresh", "m8.12a"),
        manifest_skip(root, "M8.12b", "current-C runtime-controls oracle refresh", "m8.12b"),
        manifest_skip(root, "M8.14", "current-C interactive PTY oracle refresh", "m8.14"),
        b300_runtime_skip(
            "M8.9b",
            "Rust inspect runtime comparator",
            "ds4-inspect-runtime-rs",
            "ds4-parity/compare_cli_inspect_runtime.py ds4-parity/baselines/cli/m8.8/current-c.json "
            "--candidate-binary target/debug/ds4-inspect-runtime-rs --negative-test",
        ),
        b300_runtime_skip(
            "M8.13a",
            "Rust argmax runtime comparator",
            "ds4-argmax-runtime-rs",
            "ds4-parity/compare_cli_argmax_runtime.py ds4-parity/baselines/cli/m8.12a/current-c.json "
            "--candidate-binary target/debug/ds4-argmax-runtime-rs --negative-test",
        ),
        b300_runtime_skip(
            "M8.13b",
            "Rust session runtime comparator",
            "ds4-session-runtime-rs",
            "ds4-parity/compare_cli_session_runtime.py ds4-parity/baselines/cli/m8.12a/current-c.json "
            "--candidate-binary target/debug/ds4-session-runtime-rs --negative-test",
        ),
        b300_runtime_skip(
            "M8.13c",
            "Rust one-shot runtime comparator",
            "ds4-cli-one-shot-rs",
            "ds4-parity/compare_cli_one_shot_runtime.py ds4-parity/baselines/cli/m8.12a/current-c.json "
            "--candidate-binary target/debug/ds4-cli-one-shot-rs --negative-test",
        ),
        b300_runtime_skip(
            "M8.13d",
            "Rust runtime-controls comparator",
            "ds4-cli-one-shot-rs",
            "ds4-parity/compare_cli_runtime_controls_runtime.py ds4-parity/baselines/cli/m8.12b/current-c.json "
            "--candidate-binary target/debug/ds4-cli-one-shot-rs --negative-test",
        ),
        b300_runtime_skip(
            "M8.15a",
            "Rust interactive runtime comparator",
            "ds4-interactive-runtime-rs",
            "ds4-parity/compare_cli_interactive_runtime.py ds4-parity/baselines/cli/m8.14/current-c.json "
            "--candidate-binary target/debug/ds4-interactive-runtime-rs --negative-test",
        ),
        b300_runtime_skip(
            "M8.15c",
            "Rust interactive PTY comparator",
            "ds4-cli-interactive-rs",
            "ds4-parity/compare_cli_interactive_pty.py ds4-parity/baselines/cli/m8.14/current-c.json "
            "--candidate-binary target/debug/ds4-cli-interactive-rs "
            "--write-candidate /tmp/ds4-m8.15c-rust-pty.json --negative-test",
        ),
    ]


def manifest_skip(root: Path, milestone: str, name: str, directory: str) -> ReportItem:
    manifest = root / "ds4-parity" / "baselines" / "cli" / directory / "manifest.json"
    rerun_command = ""
    try:
        obj = json.loads(manifest.read_text())
        commands = obj.get("capture_commands")
        if isinstance(commands, list):
            rerun_command = " && ".join(str(command) for command in commands)
    except OSError:
        pass
    return ReportItem(
        name=name,
        milestone=milestone,
        kind="b300-cli-refresh",
        status="SKIP",
        reason="model-backed current-C CLI refresh is not executed by the local report",
        rerun_command=rerun_command or f"missing capture_commands in {manifest}",
    )


def b300_runtime_skip(milestone: str, name: str, binary: str, comparator: str) -> ReportItem:
    script = (
        f"{CARGO_ENV} CUDA_ARCH=native cargo build -p ds4-engine --bin {binary} && "
        f"{CARGO_ENV} python3 {comparator}"
    )
    return ReportItem(
        name=name,
        milestone=milestone,
        kind="b300-rust-runtime",
        status="SKIP",
        reason="model-backed Rust CLI runtime comparator requires the B300 model/workdir",
        rerun_command=b300_exec(script),
    )


def b300_exec(script: str) -> str:
    command = [
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
        f"set -e; cd {B300_WORKDIR}; {script}",
    ]
    return shell_join(command)


def extract_summary(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.startswith("summary:") or ": PASS," in line or ": FAIL," in line:
            return line
    return ""


def tail_lines(text: str, limit: int = 12) -> list[str]:
    return text.splitlines()[-limit:]


def append_tail(lines: list[str], label: str, tail: list[str]) -> None:
    if not tail:
        return
    lines.append(f"  {label}:")
    for line in tail:
        lines.append(f"    {line}")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of ds4-parity/)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="timeout per executed command (default: 600)",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        print("--timeout-seconds must be positive", file=sys.stderr)
        return 2
    report = CliParityReport(root=args.root, timeout=args.timeout_seconds)
    report.run()
    sys.stdout.write(report.report_json() if args.json else report.report_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
