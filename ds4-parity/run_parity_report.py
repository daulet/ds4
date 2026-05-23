#!/usr/bin/env python3
"""Run the local DS4 parity report for Milestone 1.

The report has two jobs:

* run local no-model C checks that are available in this workspace;
* run the committed artifact comparators from M1.2 through M1.5, M4.6, M5.7,
  M6.7, M7.9, M9.9, and M10.2.

Model-backed B300 oracle refreshes are intentionally skipped by default.  A
skip is allowed only when the report gives the missing requirement and an exact
rerun command that preserves the temporary kubeconfig and explicit context
workflow used for the captured baselines.
"""

from __future__ import annotations

import argparse
import json
import platform
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
B300_MODEL = "/workspace/ds4/ds4flash.gguf"
DEFAULT_TIMEOUT_SECONDS = 600


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


class ParityReport:
    def __init__(self, root: Path, skip_local_oracles: bool, timeout: int) -> None:
        self.root = root.resolve()
        self.skip_local_oracles = skip_local_oracles
        self.timeout = timeout
        self.items: list[ReportItem] = []

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def run(self) -> None:
        self.run_local_oracles()
        self.run_comparators()
        self.add_b300_skips()

    def run_local_oracles(self) -> None:
        commands = local_oracle_commands()
        for name, command in commands:
            item = ReportItem(name=name, kind="local-oracle", command=command)
            self.items.append(item)
            if self.skip_local_oracles:
                item.status = "SKIP"
                item.reason = "local oracle execution disabled by --skip-local-oracles"
                item.rerun_command = shell_join(command)
                continue
            self.run_command(item)

    def run_comparators(self) -> None:
        commands = [
            (
                "M1.2 static baseline verifier",
                [sys.executable, "ds4-parity/verify_baselines.py"],
            ),
            (
                "M1.3 server/KV artifact comparator",
                [sys.executable, "ds4-parity/compare_server_kv.py"],
            ),
            (
                "M1.4 logprob numeric comparator",
                [sys.executable, "ds4-parity/compare_logprob_numeric.py"],
            ),
            (
                "M1.5 benchmark CSV comparator",
                [sys.executable, "ds4-parity/compare_bench_csv.py"],
            ),
            (
                "M4.6 metadata baseline comparator",
                [sys.executable, "ds4-parity/compare_metadata_baseline.py", "--negative-test"],
            ),
            (
                "M5.7 text parity report",
                [sys.executable, "ds4-parity/run_text_parity_report.py"],
            ),
            (
                "M6.7 sampling/logprob parity report",
                [sys.executable, "ds4-parity/run_sampling_parity_report.py"],
            ),
            (
                "M7.9 KV/snapshot parity report",
                [sys.executable, "ds4-parity/run_kv_parity_report.py"],
            ),
            (
                "M8.16 CLI parity report",
                [sys.executable, "ds4-parity/run_cli_parity_report.py"],
            ),
            (
                "M9.9 server/runtime parity report",
                [sys.executable, "ds4-parity/run_server_parity_report.py"],
            ),
            (
                "M10.2 graph plan inventory oracle",
                [sys.executable, "ds4-parity/check_graph_plan_inventory.py"],
            ),
        ]
        for name, command in commands:
            item = ReportItem(name=name, kind="comparator", command=command)
            self.items.append(item)
            self.run_command(item)

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
        if proc.returncode != 0 and not item.reason:
            item.reason = f"exit status {proc.returncode}"

    def add_b300_skips(self) -> None:
        for item in b300_skip_items():
            self.items.append(item)

    def report_text(self) -> str:
        lines = [
            "DS4 unified parity report",
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
        lines.append(
            f"summary: {passed} passed, {skipped} skipped, {failed} failed"
        )
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


def local_oracle_commands() -> list[tuple[str, list[str]]]:
    prefix = ["arch", "-arm64"] if platform.system() == "Darwin" else []
    return [
        ("local no-model clean", prefix + ["make", "clean"]),
        ("local no-model build ds4_test", prefix + ["make", "ds4_test"]),
        ("local no-model ds4_test --server", prefix + ["./ds4_test", "--server"]),
        (
            "local no-model ds4_test --metal-kernels",
            prefix + ["./ds4_test", "--metal-kernels"],
        ),
        (
            "local no-model make cuda-regression",
            prefix + ["make", "cuda-regression"],
        ),
    ]


def b300_skip_items() -> list[ReportItem]:
    return [
        ReportItem(
            name="B300 model-backed M0.3 logprob oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason="model-backed B300 rerun is not executed by the local report",
            rerun_command=b300_exec(
                "make ds4_test && "
                f"DS4_TEST_MODEL={B300_MODEL} "
                "DS4_TEST_VECTOR_FILE=tests/test-vectors/official.vec "
                "./ds4_test --logprob-vectors"
            ),
        ),
        ReportItem(
            name="B300 model-backed M0.4 server trace oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "server replay refresh requires this B300 server start command "
                "plus the fixture replay order recorded in the M0.4 replay log"
            ),
            rerun_command=b300_exec(
                "make clean ds4-server && "
                "./ds4-server -m /workspace/ds4/ds4flash.gguf --cuda --ctx 32768 "
                "--tokens 64 --host 127.0.0.1 --port 18080 "
                "--trace ds4-parity/baselines/server-traces/m0.4/traces/server.trace "
                "--kv-disk-dir ds4-parity/baselines/server-traces/m0.4/kv "
                "--kv-disk-space-mb 512 --kv-cache-min-tokens 512 "
                "--kv-cache-cold-max-tokens 30000 "
                "--kv-cache-continued-interval-tokens 0"
            ),
        ),
        ReportItem(
            name="B300 model-backed M0.5 KV restore oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "KV replay refresh requires three server lifetimes; this command "
                "starts the first lifetime with concrete port/server labels"
            ),
            rerun_command=b300_exec(
                "PORT=18081 SERVER=server-a; "
                "make clean ds4-server && "
                "./ds4-server -m /workspace/ds4/ds4flash.gguf --cuda --ctx 32768 "
                "--tokens 16 --host 127.0.0.1 --port ${PORT} "
                "--trace ds4-parity/baselines/kv-artifacts/m0.5/traces/${SERVER}.trace "
                "--kv-disk-dir ds4-parity/baselines/kv-artifacts/m0.5/kv "
                "--kv-disk-space-mb 512 --kv-cache-min-tokens 512 "
                "--kv-cache-cold-max-tokens 30000 "
                "--kv-cache-continued-interval-tokens 0"
            ),
        ),
        ReportItem(
            name="B300 model-backed M0.6 benchmark oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason="benchmark refresh requires the B300 model and same GPU class",
            rerun_command=b300_exec(
                "make clean ds4-bench && "
                "./ds4-bench -m /workspace/ds4/ds4flash.gguf --cuda "
                "--prompt-file speed-bench/promessi_sposi.txt "
                "--ctx-start 2048 --ctx-max 8192 --step-incr 2048 "
                "--gen-tokens 32 --csv ds4-parity/baselines/bench/m0.6/csv/b300-short.csv && "
                "./ds4-bench -m /workspace/ds4/ds4flash.gguf --cuda "
                "--prompt-file speed-bench/promessi_sposi.txt "
                "--ctx-start 16384 --ctx-max 32768 --step-incr 8192 "
                "--gen-tokens 32 --csv ds4-parity/baselines/bench/m0.6/csv/b300-long.csv"
            ),
        ),
        ReportItem(
            name="B300 model-backed M4.6 metadata baseline refresh",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "metadata baseline refresh requires the B300 q2-imatrix model; "
                "the committed manifest records source-file copy commands and artifact hashes"
            ),
            rerun_command=b300_metadata_refresh_command(),
        ),
    ]


def b300_metadata_refresh_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    copy_commands = [
        shell_join(prefix + ["cp", "ds4.c", f"{KUBE_POD}:{B300_WORKDIR}/ds4.c"]),
        shell_join(prefix + ["cp", "ds4.h", f"{KUBE_POD}:{B300_WORKDIR}/ds4.h"]),
        shell_join(prefix + ["cp", "ds4_metadata_dump.c", f"{KUBE_POD}:{B300_WORKDIR}/ds4_metadata_dump.c"]),
    ]
    capture = b300_exec(
        "make clean ds4-metadata-dump CUDA_ARCH=native && "
        f"./ds4-metadata-dump -m {B300_MODEL} -o /tmp/ds4-metadata-m4.6-c.json && "
        "wc -c /tmp/ds4-metadata-m4.6-c.json && "
        "sha256sum /tmp/ds4-metadata-m4.6-c.json && "
        "python3 ds4-parity/check_metadata_dump.py /tmp/ds4-metadata-m4.6-c.json --negative-test"
    )
    copy_back = shell_join(
        prefix + ["cp", f"{KUBE_POD}:/tmp/ds4-metadata-m4.6-c.json", "/tmp/ds4-metadata-m4.6-c.json"]
    )
    return " && ".join([*copy_commands, capture, copy_back])


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
        if line.startswith("summary:"):
            return line
    return ""


def tail_lines(text: str, limit: int = 12) -> list[str]:
    lines = text.splitlines()
    return lines[-limit:]


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
        "--skip-local-oracles",
        action="store_true",
        help="skip local no-model C checks and print rerun commands",
    )
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
    report = ParityReport(
        root=args.root,
        skip_local_oracles=args.skip_local_oracles,
        timeout=args.timeout_seconds,
    )
    report.run()
    sys.stdout.write(report.report_json() if args.json else report.report_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
