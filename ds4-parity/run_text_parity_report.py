#!/usr/bin/env python3
"""Run the local DS4 Milestone 5 text parity report."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 600

M52_MANIFEST = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.2" / "manifest.json"
M53_MANIFEST = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.3" / "manifest.json"


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


class TextParityReport:
    def __init__(self, root: Path, timeout: int) -> None:
        self.root = root.resolve()
        self.timeout = timeout
        self.items: list[ReportItem] = []

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def run(self) -> None:
        for name, command in comparator_commands():
            item = ReportItem(name=name, kind="text-comparator", command=command)
            self.items.append(item)
            self.run_command(item)
        self.items.extend(refresh_skip_items())

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
            "DS4 Milestone 5 text parity report",
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


def comparator_commands() -> list[tuple[str, list[str]]]:
    return [
        (
            "M5.2 current-C token/prompt oracle schema",
            [
                sys.executable,
                "ds4-parity/check_tokenization_dump.py",
                "ds4-parity/baselines/tokenization/m5.2/current-c.json",
                "--manifest",
                "ds4-parity/baselines/tokenization/m5.2/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "M5.3-M5.5 Rust tokenizer and prompt parity",
            [
                sys.executable,
                "ds4-parity/compare_tokenizer_text.py",
                "--manifest",
                "ds4-parity/baselines/tokenization/m5.3/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "M5.6a server DSML formatting and parse parity",
            [
                sys.executable,
                "ds4-parity/compare_dsml.py",
                "--manifest",
                "ds4-parity/baselines/dsml/m5.6a/manifest.json",
                "--negative-test",
            ],
        ),
        (
            "M5.6b agent DSML streaming parser parity",
            [
                sys.executable,
                "ds4-parity/compare_agent_dsml.py",
                "--manifest",
                "ds4-parity/baselines/dsml/m5.6b/manifest.json",
                "--negative-test",
            ],
        ),
    ]


def refresh_skip_items() -> list[ReportItem]:
    return [
        ReportItem(
            name="B300 M5.2 token/prompt oracle refresh",
            kind="b300-refresh",
            status="SKIP",
            reason="model-backed token/prompt oracle refresh requires the B300 q2-imatrix model",
            rerun_command=manifest_refresh_command(M52_MANIFEST),
        ),
        ReportItem(
            name="B300 M5.3 tokenizer fixture refresh",
            kind="b300-refresh",
            status="SKIP",
            reason="tokenizer fixture extraction refresh requires the B300 q2-imatrix model",
            rerun_command=manifest_refresh_command(M53_MANIFEST),
        ),
    ]


def manifest_refresh_command(path: Path) -> str:
    try:
        manifest = json.loads(path.read_text())
    except OSError as exc:
        return f"<missing manifest {path}: {exc}>"
    commands = manifest.get("refresh_commands", [])
    if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
        return f"<manifest {path} has no refresh_commands list>"
    return " && ".join(commands)


def extract_summary(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.startswith("summary:"):
            return line
        lowered = line.lower()
        if "pass" in lowered or "fail" in lowered:
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
        default=ROOT,
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
    report = TextParityReport(root=args.root, timeout=args.timeout_seconds)
    report.run()
    sys.stdout.write(report.report_json() if args.json else report.report_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
