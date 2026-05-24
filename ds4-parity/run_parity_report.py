#!/usr/bin/env python3
"""Run the local DS4 parity report for Milestone 1.

The report has two jobs:

* run local no-model C checks that are available in this workspace;
* run the committed artifact comparators from M1.2 through M1.5, M4.6, M5.7,
  M6.7, M7.9, M9.9, M10.2, M10.3, M10.4, M10.5a, M10.5b, M10.5c1, M10.5c2,
  M10.5c3, M10.5c4a, M10.5c4b, M10.5c4c1, M10.5c4c2a, and
  M10.5c4c2b1, M10.5c4c2b2a, M10.5c4c2b2b1,
  M10.5c4c2b2b2a, M10.5c4c2b2b2b1, M10.5c4c2b2b2b2a,
  M10.5c4c2b2b2b2b1, M10.5c4c2b2b2b2b2a,
  M10.5c4c2b2b2b2b2b1, M10.5c4c2b2b2b2b2b2a,
  M10.5c4c2b2b2b2b2b2b1, M10.5c4c2b2b2b2b2b2b2a,
  M10.5c4c2b2b2b2b2b2b2b1, M10.5c4c2b2b2b2b2b2b2b2a,
  M10.5c4c2b2b2b2b2b2b2b2b, M10.5c4d1, M10.5c4d2,
  M10.5c4d3, M10.5c4d4, M10.6a, M10.6b, M10.6c, M10.6d, M10.7a,
  M10.7b, M10.7c1, M10.7c2, M10.7c3a, M10.7c3b, M10.7c3c, M10.7c3d,
  M10.7d3a, M10.7d3b, and M10.7d3c1.

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
            (
                "M10.3 Rust graph plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_plan_rust.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.4 graph checkpoint oracle",
                [
                    sys.executable,
                    "ds4-parity/check_graph_checkpoint_oracle.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5a Rust GPU sys ABI comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_gpu_sys_abi.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5b Rust decode plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_plan_rust.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c1 Rust structured weight table comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_rust_weight_table.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c2 Rust graph state comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_state_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c3 Rust decode backend facade comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_backend_facade.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4a Rust decode trace comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_trace.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4b Rust decode runtime bridge comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_runtime_bridge.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c1 Rust CUDA backend smoke contract",
                [
                    sys.executable,
                    "ds4-parity/compare_b300_rust_backend_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2a Rust decode model-map bridge comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_model_map_bridge.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b1 Rust decode execution preflight comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_execution_preflight.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2a Rust full decode state allocation comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_state_allocation.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b1 Rust first decode kernel comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_first_kernel.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2a Rust first-kernel current-C oracle comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_first_kernel_oracle.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b1 Rust layer-0 attention HC-pre comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer0_attn_hc_pre.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2a Rust layer-0 QKV/RoPE comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer0_qkv_rope.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b1 Rust layer-0 attention-output comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer0_attn_output.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2a Rust layer-0 FFN-output comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer0_ffn_output.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b1 Rust layer-0 output-head comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer0_output_head.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2a Rust two-layer output-head comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_two_layer_output_head.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b1 Rust layer-2 compressor-state comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer2_compressor_state.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b2a Rust layer-2 attention-output comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer2_attn_output.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b2b1 Rust layer-2 FFN-output comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer2_ffn_output.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b2b2a Rust layer-3 ratio-128 FFN-output comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer3_ffn_output.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b2b2b1 Rust layer-4 post-ratio128 ratio-4/indexer FFN-output comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_layer4_ffn_output.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b2b2b2a Rust all-layer final-HC comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_all_layer_final_hc.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4c2b2b2b2b2b2b2b2b2b Rust full output-head comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_full_output_head.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4d1 Rust short continuation output-head comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_short_continuation_output_head.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4d2 Rust ratio-boundary output-head comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_ratio_boundary_output_head.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4d3 Rust long indexed attention comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_long_indexed_attention.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.5c4d4 Rust directional-steering decode comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_decode_directional_steering.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.6a Rust prefill scheduling plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_prefill_plan_rust.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.6b Rust whole-prefill short comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_prefill_whole_short.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.6c Rust chunked-prefill comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_prefill_chunked.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.6d Rust resumed-prefill comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_prefill_resumed.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7a Rust graph-session payload layout comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_session_payload_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7b Rust graph-session payload reader/writer comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_session_payload_rw.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7c1 Rust restore payload header comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_restore_payload_header_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7c2 Rust raw graph payload import comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_payload_raw_import.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7c3a Rust raw graph snapshot import comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_snapshot_raw_import.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7c3b Rust graph restore target comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_restore_target_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7c3c Rust graph restore readback comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_restore_readback.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7d3b Rust graph restore frontier projection comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_graph_restore_next_token.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7d3a Rust graph restore frontier contract",
                [
                    sys.executable,
                    "ds4-parity/check_graph_restore_frontier_contract.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.7d3c1 Rust post-restore KVC decision contract",
                [
                    sys.executable,
                    "ds4-parity/check_post_restore_kvc_decision_contract.py",
                    "--negative-test",
                ],
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
            name="M10.5c4c1 B300 Rust CUDA backend smoke rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust CUDA backend smoke requires the B300 pod, CUDA toolchain, "
                "and a Rust toolchain bootstrap in that pod"
            ),
            rerun_command=b300_rust_backend_smoke_command(),
        ),
        ReportItem(
            name="M10.5c4c2a B300 Rust model-map backend smoke rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust model-map backend smoke requires the B300 pod and "
                "feature-gated CUDA backend linkage"
            ),
            rerun_command=b300_rust_model_map_smoke_command(),
        ),
        ReportItem(
            name="M10.5c4c2b1 B300 Rust decode execution preflight rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust decode execution preflight requires the B300 pod, the "
                "real q2-imatrix GGUF, and feature-gated CUDA backend linkage"
            ),
            rerun_command=b300_rust_decode_preflight_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2a B300 Rust decode state allocation rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust decode state allocation requires the B300 pod and "
                "feature-gated CUDA backend linkage"
            ),
            rerun_command=b300_rust_decode_state_allocation_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b1 B300 Rust first decode kernel rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust first decode kernel requires the B300 pod, the real "
                "q2-imatrix GGUF, and feature-gated CUDA backend linkage"
            ),
            rerun_command=b300_rust_decode_first_kernel_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2a B300 first-kernel current-C oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "First-kernel current-C oracle comparison requires the B300 "
                "pod, the real q2-imatrix GGUF, the current-C helper, and "
                "feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_first_kernel_current_c_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b1 B300 layer-0 attention HC-pre oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-0 attention HC-pre current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer0_attn_hc_pre_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2a B300 layer-0 QKV/RoPE oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-0 QKV/RoPE current-C oracle comparison requires the "
                "B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer0_qkv_rope_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b1 B300 layer-0 attention-output oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-0 attention-output current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer0_attn_output_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2a B300 layer-0 FFN-output oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-0 FFN-output current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer0_ffn_output_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b1 B300 layer-0 output-head oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-0 output-head current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer0_output_head_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2a B300 two-layer output-head oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Two-layer output-head current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_two_layer_output_head_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b1 B300 layer-2 compressor-state oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-2 compressor-state current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer2_compressor_state_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b2a B300 layer-2 attention-output oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-2 attention-output current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer2_attn_output_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b2b1 B300 layer-2 FFN-output oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-2 FFN-output current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer2_ffn_output_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b2b2a B300 layer-3 ratio-128 FFN-output oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-3 ratio-128 FFN-output current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_layer3_ffn_output_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b2b2b1 B300 layer-4 post-ratio128 ratio-4/indexer FFN-output oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Layer-4 post-ratio128 ratio-4/indexer FFN-output current-C "
                "oracle comparison requires the B300 pod, the real q2-imatrix "
                "GGUF, the current-C helper, and feature-gated Rust CUDA "
                "backend linkage"
            ),
            rerun_command=b300_layer4_ffn_output_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b2b2b2a B300 all-layer final-HC oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "All-layer final-HC current-C oracle comparison requires the "
                "B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_all_layer_final_hc_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4c2b2b2b2b2b2b2b2b2b B300 full output-head oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Full output-head/logits current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_full_output_head_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4d1 B300 short continuation output-head oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Short continuation output-head/logits current-C oracle "
                "comparison requires the B300 pod, the real q2-imatrix GGUF, "
                "the current-C helper, and feature-gated Rust CUDA backend "
                "linkage"
            ),
            rerun_command=b300_short_continuation_output_head_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4d2 B300 ratio-boundary output-head oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Ratio-boundary output-head/logits current-C oracle comparison "
                "requires the B300 pod, the real q2-imatrix GGUF, the current-C "
                "helper, and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_ratio_boundary_output_head_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4d3 B300 long indexed attention oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Long indexed-attention current-C oracle comparison requires "
                "the B300 pod, the real q2-imatrix GGUF, the current-C helper, "
                "and feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_long_indexed_attention_oracle_command(),
        ),
        ReportItem(
            name="M10.5c4d4 B300 directional-steering decode oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Directional-steering decode current-C oracle comparison "
                "requires the B300 pod, the real q2-imatrix GGUF, "
                "dir-steering/out/verbosity.f32, the current-C helper, and "
                "feature-gated Rust CUDA backend linkage"
            ),
            rerun_command=b300_directional_steering_decode_oracle_command(),
        ),
        ReportItem(
            name="M10.6b B300 whole-prefill short oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Whole-prefill short-prompt current-C oracle comparison "
                "requires the B300 pod, the real q2-imatrix GGUF, the current-C "
                "helper, the short prompt fixture, and feature-gated Rust CUDA "
                "backend linkage"
            ),
            rerun_command=b300_prefill_whole_short_oracle_command(),
        ),
        ReportItem(
            name="M10.6c B300 chunked-prefill oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Chunked-prefill current-C oracle comparison requires the "
                "B300 pod, the real q2-imatrix GGUF, the long prompt fixture, "
                "the deterministic CUDA MoE mode, and feature-gated Rust CUDA "
                "backend linkage"
            ),
            rerun_command=b300_prefill_chunked_oracle_command(),
        ),
        ReportItem(
            name="M10.6d B300 resumed-prefill oracle rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Resumed-prefill current-C oracle comparison requires the "
                "B300 pod, the real q2-imatrix GGUF, the long prompt fixture, "
                "the deterministic CUDA MoE mode, and feature-gated Rust CUDA "
                "backend linkage"
            ),
            rerun_command=b300_prefill_resumed_oracle_command(),
        ),
        ReportItem(
            name="M10.7c2 B300 Rust raw graph payload import rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Raw graph payload import requires the B300 pod because the "
                "M7.8 disk payload bodies are hash-only and remain in "
                "/workspace/ds4"
            ),
            rerun_command=b300_rust_raw_graph_payload_import_command(),
        ),
        ReportItem(
            name="M10.7c3a B300 Rust raw graph snapshot import rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Raw graph snapshot import requires the B300 pod because the "
                "M7.8 memory snapshot bodies are hash-only and must be "
                "materialized in /workspace/ds4"
            ),
            rerun_command=b300_rust_raw_graph_snapshot_import_command(),
        ),
        ReportItem(
            name="M10.7c3c B300 Rust graph restore readback rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Graph restore readback requires the B300 pod because the "
                "M7.8 disk payload and memory snapshot raw bodies are "
                "hash-only and remain in /workspace/ds4"
            ),
            rerun_command=b300_rust_graph_restore_readback_command(),
        ),
        ReportItem(
            name="M10.7d3b B300 Rust graph restore frontier projection rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Graph restore frontier projection requires the B300 pod "
                "because the M7.8 disk payload and memory snapshot raw bodies "
                "are hash-only and remain in /workspace/ds4"
            ),
            rerun_command=b300_rust_graph_restore_next_token_command(),
        ),
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


def b300_rust_backend_smoke_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "rustup default stable && "
        "CUDA_ARCH=native cargo test -p ds4-gpu --features cuda-backend "
        "--test backend_abi -- --nocapture"
    )
    return f"{source_refresh} && {smoke}"


def b300_rust_model_map_smoke_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "CUDA_ARCH=native cargo test -p ds4-gpu --features cuda-backend "
        "--test model_map_abi -- --nocapture"
    )
    return f"{source_refresh} && {smoke}"


def b300_rust_decode_preflight_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-exec-preflight --quiet -- "
        f"--model {B300_MODEL} "
        "--model-sha256 efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668 "
        "> /tmp/ds4-c2b1-preflight.json && "
        "python3 ds4-parity/compare_decode_execution_preflight.py "
        "--candidate /tmp/ds4-c2b1-preflight.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_rust_decode_state_allocation_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-state-alloc --quiet "
        "> /tmp/ds4-c2b2a-state-allocation.json && "
        "python3 ds4-parity/compare_decode_state_allocation.py "
        "--candidate /tmp/ds4-c2b2a-state-allocation.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_rust_decode_first_kernel_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-first-kernel --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b1-first-kernel.json && "
        "python3 ds4-parity/compare_decode_first_kernel.py "
        "--candidate /tmp/ds4-c2b2b1-first-kernel.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_first_kernel_current_c_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-first-kernel-oracle-dump CUDA_ARCH=native && "
        f"./ds4-first-kernel-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2a-first-kernel-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-first-kernel --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2a-first-kernel-rust.json && "
        "python3 ds4-parity/compare_decode_first_kernel_oracle.py "
        "--oracle /tmp/ds4-c2b2b2a-first-kernel-oracle.json "
        "--candidate /tmp/ds4-c2b2b2a-first-kernel-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer0_attn_hc_pre_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer0-attn-hc-pre-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer0-attn-hc-pre-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b1-layer0-attn-hc-pre-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer0-attn-hc-pre --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b1-layer0-attn-hc-pre-rust.json && "
        "python3 ds4-parity/compare_decode_layer0_attn_hc_pre.py "
        "--oracle /tmp/ds4-c2b2b2b1-layer0-attn-hc-pre-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b1-layer0-attn-hc-pre-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer0_qkv_rope_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer0-qkv-rope-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer0-qkv-rope-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2a-layer0-qkv-rope-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer0-qkv-rope --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2a-layer0-qkv-rope-rust.json && "
        "python3 ds4-parity/compare_decode_layer0_qkv_rope.py "
        "--oracle /tmp/ds4-c2b2b2b2a-layer0-qkv-rope-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2a-layer0-qkv-rope-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer0_attn_output_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer0-attn-output-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer0-attn-output-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b1-layer0-attn-output-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer0-attn-output --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b1-layer0-attn-output-rust.json && "
        "python3 ds4-parity/compare_decode_layer0_attn_output.py "
        "--oracle /tmp/ds4-c2b2b2b2b1-layer0-attn-output-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b1-layer0-attn-output-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer0_ffn_output_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer0-ffn-output-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer0-ffn-output-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2a-layer0-ffn-output-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer0-ffn-output --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2a-layer0-ffn-output-rust.json && "
        "python3 ds4-parity/compare_decode_layer0_ffn_output.py "
        "--oracle /tmp/ds4-c2b2b2b2b2a-layer0-ffn-output-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2a-layer0-ffn-output-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer0_output_head_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer0-output-head-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer0-output-head-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b1-layer0-output-head-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer0-output-head --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b1-layer0-output-head-rust.json && "
        "python3 ds4-parity/compare_decode_layer0_output_head.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b1-layer0-output-head-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b1-layer0-output-head-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_two_layer_output_head_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-two-layer-output-head-oracle-dump CUDA_ARCH=native && "
        f"./ds4-two-layer-output-head-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2a-two-layer-output-head-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-two-layer-output-head --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2a-two-layer-output-head-rust.json && "
        "python3 ds4-parity/compare_decode_two_layer_output_head.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2a-two-layer-output-head-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2a-two-layer-output-head-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer2_compressor_state_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer2-compressor-state-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer2-compressor-state-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b1-layer2-compressor-state-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer2-compressor-state --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b1-layer2-compressor-state-rust.json && "
        "python3 ds4-parity/compare_decode_layer2_compressor_state.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b1-layer2-compressor-state-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b1-layer2-compressor-state-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer2_attn_output_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer2-attn-output-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer2-attn-output-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b2a-layer2-attn-output-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer2-attn-output --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b2a-layer2-attn-output-rust.json && "
        "python3 ds4-parity/compare_decode_layer2_attn_output.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b2a-layer2-attn-output-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b2a-layer2-attn-output-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer2_ffn_output_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer2-ffn-output-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer2-ffn-output-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b2b1-layer2-ffn-output-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer2-ffn-output --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b2b1-layer2-ffn-output-rust.json && "
        "python3 ds4-parity/compare_decode_layer2_ffn_output.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b2b1-layer2-ffn-output-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b2b1-layer2-ffn-output-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer3_ffn_output_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer3-ffn-output-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer3-ffn-output-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b2b2a-layer3-ffn-output-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer3-ffn-output --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b2b2a-layer3-ffn-output-rust.json && "
        "python3 ds4-parity/compare_decode_layer3_ffn_output.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b2b2a-layer3-ffn-output-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b2b2a-layer3-ffn-output-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_layer4_ffn_output_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-layer4-ffn-output-oracle-dump CUDA_ARCH=native && "
        f"./ds4-layer4-ffn-output-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-layer4-ffn-output --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-rust.json && "
        "python3 ds4-parity/compare_decode_layer4_ffn_output.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_all_layer_final_hc_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-all-layer-final-hc-oracle-dump CUDA_ARCH=native && "
        f"./ds4-all-layer-final-hc-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b2b2b2a-all-layer-final-hc-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-all-layer-final-hc --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b2b2b2a-all-layer-final-hc-rust.json && "
        "python3 ds4-parity/compare_decode_all_layer_final_hc.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b2b2b2a-all-layer-final-hc-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b2b2b2a-all-layer-final-hc-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_full_output_head_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-full-output-head-oracle-dump CUDA_ARCH=native && "
        f"./ds4-full-output-head-oracle-dump -m {B300_MODEL} --token 0 "
        "-o /tmp/ds4-c2b2b2b2b2b2b2b2b2b-full-output-head-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-full-output-head --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c2b2b2b2b2b2b2b2b2b-full-output-head-rust.json && "
        "python3 ds4-parity/compare_decode_full_output_head.py "
        "--oracle /tmp/ds4-c2b2b2b2b2b2b2b2b2b-full-output-head-oracle.json "
        "--candidate /tmp/ds4-c2b2b2b2b2b2b2b2b2b-full-output-head-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_short_continuation_output_head_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-short-continuation-output-head-oracle-dump CUDA_ARCH=native && "
        f"./ds4-short-continuation-output-head-oracle-dump -m {B300_MODEL} "
        "-o /tmp/ds4-c4d1-short-continuation-output-head-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-short-continuation-output-head --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c4d1-short-continuation-output-head-rust.json && "
        "python3 ds4-parity/compare_decode_short_continuation_output_head.py "
        "--oracle /tmp/ds4-c4d1-short-continuation-output-head-oracle.json "
        "--candidate /tmp/ds4-c4d1-short-continuation-output-head-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_ratio_boundary_output_head_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "make ds4-ratio-boundary-output-head-oracle-dump CUDA_ARCH=native && "
        f"./ds4-ratio-boundary-output-head-oracle-dump -m {B300_MODEL} "
        "-o /tmp/ds4-c4d2-ratio-boundary-output-head-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-ratio-boundary-output-head --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c4d2-ratio-boundary-output-head-rust.json && "
        "python3 ds4-parity/compare_decode_ratio_boundary_output_head.py "
        "--oracle /tmp/ds4-c4d2-ratio-boundary-output-head-oracle.json "
        "--candidate /tmp/ds4-c4d2-ratio-boundary-output-head-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_long_indexed_attention_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "rm -f ds4.o ds4_cuda.o ds4_long_indexed_attention_oracle_dump*.o "
        "ds4-long-indexed-attention-oracle-dump && "
        "make ds4-long-indexed-attention-oracle-dump CUDA_ARCH=native && "
        f"./ds4-long-indexed-attention-oracle-dump -m {B300_MODEL} "
        "-o /tmp/ds4-c4d3-long-indexed-attention-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-long-indexed-attention --quiet -- "
        f"--model {B300_MODEL} "
        "> /tmp/ds4-c4d3-long-indexed-attention-rust.json && "
        "python3 ds4-parity/compare_decode_long_indexed_attention.py "
        "--oracle /tmp/ds4-c4d3-long-indexed-attention-oracle.json "
        "--candidate /tmp/ds4-c4d3-long-indexed-attention-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_directional_steering_decode_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    smoke = b300_exec(
        "rm -f ds4.o ds4_cuda.o ds4_directional_steering_oracle_dump*.o "
        "ds4-directional-steering-oracle-dump && "
        "make ds4-directional-steering-oracle-dump CUDA_ARCH=native && "
        f"./ds4-directional-steering-oracle-dump -m {B300_MODEL} "
        "--dir-steering-file dir-steering/out/verbosity.f32 "
        "--dir-steering-attn 0.5 --dir-steering-ffn 0.25 "
        "-o /tmp/ds4-c4d4-directional-steering-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-decode-full-output-head --quiet -- "
        f"--model {B300_MODEL} "
        "--dir-steering-file dir-steering/out/verbosity.f32 "
        "--dir-steering-attn 0.5 --dir-steering-ffn 0.25 "
        "> /tmp/ds4-c4d4-directional-steering-rust.json && "
        "python3 ds4-parity/compare_decode_directional_steering.py "
        "--oracle /tmp/ds4-c4d4-directional-steering-oracle.json "
        "--candidate /tmp/ds4-c4d4-directional-steering-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_prefill_whole_short_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    prompt = "tests/test-vectors/prompts/short_italian_fact.txt"
    smoke = b300_exec(
        "make ds4-prefill-whole-short-oracle-dump CUDA_ARCH=native && "
        f"./ds4-prefill-whole-short-oracle-dump --model {B300_MODEL} "
        f"--prompt {prompt} --backend cuda "
        "--output /tmp/ds4-m106b-prefill-whole-short-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-prefill-whole-short --quiet -- "
        f"--model {B300_MODEL} --prompt {prompt} "
        "> /tmp/ds4-m106b-prefill-whole-short-rust.json && "
        "python3 ds4-parity/compare_prefill_whole_short.py "
        "--oracle /tmp/ds4-m106b-prefill-whole-short-oracle.json "
        "--candidate /tmp/ds4-m106b-prefill-whole-short-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_prefill_chunked_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    prompt = "tests/test-vectors/prompts/long_memory_archive.txt"
    smoke = b300_exec(
        "export DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 && "
        "make ds4-prefill-whole-short-oracle-dump CUDA_ARCH=native && "
        f"./ds4-prefill-whole-short-oracle-dump --model {B300_MODEL} "
        f"--prompt {prompt} --limit-tokens 2052 --backend cuda "
        "--output /tmp/ds4-m106c-prefill-chunked-2052-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-prefill-whole-short --quiet -- "
        f"--model {B300_MODEL} --prompt {prompt} --limit-tokens 2052 "
        "> /tmp/ds4-m106c-prefill-chunked-2052-rust.json && "
        "python3 ds4-parity/compare_prefill_chunked.py "
        "--oracle /tmp/ds4-m106c-prefill-chunked-2052-oracle.json "
        "--candidate /tmp/ds4-m106c-prefill-chunked-2052-rust.json && "
        f"./ds4-prefill-whole-short-oracle-dump --model {B300_MODEL} "
        f"--prompt {prompt} --backend cuda "
        "--output /tmp/ds4-m106c-prefill-chunked-long-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-prefill-whole-short --quiet -- "
        f"--model {B300_MODEL} --prompt {prompt} "
        "> /tmp/ds4-m106c-prefill-chunked-long-rust.json && "
        "python3 ds4-parity/compare_prefill_chunked.py "
        "--oracle /tmp/ds4-m106c-prefill-chunked-long-oracle.json "
        "--candidate /tmp/ds4-m106c-prefill-chunked-long-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_prefill_resumed_oracle_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    prompt = "tests/test-vectors/prompts/long_memory_archive.txt"
    smoke = b300_exec(
        "export DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 && "
        "make ds4-prefill-whole-short-oracle-dump CUDA_ARCH=native && "
        f"./ds4-prefill-whole-short-oracle-dump --model {B300_MODEL} "
        f"--prompt {prompt} --limit-tokens 512 --resume-prefix-tokens 512 --backend cuda "
        "--output /tmp/ds4-m106d-prefill-resumed-cache-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-prefill-whole-short --quiet -- "
        f"--model {B300_MODEL} --prompt {prompt} --limit-tokens 512 "
        "--resume-prefix-tokens 512 "
        "> /tmp/ds4-m106d-prefill-resumed-cache-rust.json && "
        "python3 ds4-parity/compare_prefill_resumed.py "
        "--oracle /tmp/ds4-m106d-prefill-resumed-cache-oracle.json "
        "--candidate /tmp/ds4-m106d-prefill-resumed-cache-rust.json && "
        f"./ds4-prefill-whole-short-oracle-dump --model {B300_MODEL} "
        f"--prompt {prompt} --limit-tokens 514 --resume-prefix-tokens 512 --backend cuda "
        "--output /tmp/ds4-m106d-prefill-resumed-decode-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-prefill-whole-short --quiet -- "
        f"--model {B300_MODEL} --prompt {prompt} --limit-tokens 514 "
        "--resume-prefix-tokens 512 "
        "> /tmp/ds4-m106d-prefill-resumed-decode-rust.json && "
        "python3 ds4-parity/compare_prefill_resumed.py "
        "--oracle /tmp/ds4-m106d-prefill-resumed-decode-oracle.json "
        "--candidate /tmp/ds4-m106d-prefill-resumed-decode-rust.json && "
        f"./ds4-prefill-whole-short-oracle-dump --model {B300_MODEL} "
        f"--prompt {prompt} --limit-tokens 2337 --resume-prefix-tokens 1537 --backend cuda "
        "--output /tmp/ds4-m106d-prefill-resumed-chunked-oracle.json && "
        "CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend "
        "--bin ds4-prefill-whole-short --quiet -- "
        f"--model {B300_MODEL} --prompt {prompt} --limit-tokens 2337 "
        "--resume-prefix-tokens 1537 "
        "> /tmp/ds4-m106d-prefill-resumed-chunked-rust.json && "
        "python3 ds4-parity/compare_prefill_resumed.py "
        "--oracle /tmp/ds4-m106d-prefill-resumed-chunked-oracle.json "
        "--candidate /tmp/ds4-m106d-prefill-resumed-chunked-rust.json"
    )
    return f"{source_refresh} && {smoke}"


def b300_rust_raw_graph_payload_import_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    summary = "/tmp/ds4-m107c2-raw-import.json"
    smoke = b300_exec(
        "python3 ds4-parity/compare_graph_payload_raw_import.py "
        f"--live --workdir {B300_WORKDIR} --write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/kv/m10.7c2/rust-b300-raw-import.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_rust_raw_graph_snapshot_import_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    current_c = "/tmp/ds4-m107c3a-current-c-with-snapshots.json"
    summary = "/tmp/ds4-m107c3a-snapshot-raw-import.json"
    raw_dir = "ds4-parity/baselines/kv/m7.8/raw"
    smoke = b300_exec(
        "make ds4-restore-dump CUDA_ARCH=native && "
        f"mkdir -p {raw_dir} && "
        "./ds4-restore-dump --backend cuda "
        f"-m {B300_MODEL} "
        "--model-sha256 efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668 "
        "--seed-prompt ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_prompt.txt "
        "--seed-assistant ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_assistant.txt "
        "--continuation-user ds4-parity/baselines/kv-fixtures/m7.8/restore_continuation_user.txt "
        f"--payload-dir {raw_dir} --snapshot-dir {raw_dir} -o {current_c} && "
        f"python3 ds4-parity/check_restore_dump.py {current_c} --negative-test && "
        "python3 ds4-parity/compare_graph_snapshot_raw_import.py "
        f"--live --workdir {B300_WORKDIR} --write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/kv/m10.7c3a/rust-b300-snapshot-raw-import.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_rust_graph_restore_readback_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    summary = "/tmp/ds4-m107c3c-restore-readback.json"
    smoke = b300_exec(
        "python3 ds4-parity/compare_graph_restore_readback.py "
        f"--live --workdir {B300_WORKDIR} --write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/kv/m10.7c3c/rust-b300-restore-readback.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_rust_graph_restore_next_token_command() -> str:
    prefix = [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]
    source_refresh = (
        "git archive HEAD | "
        + shell_join(prefix + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR])
    )
    summary = "/tmp/ds4-m107d3b-restore-next-token.json"
    smoke = b300_exec(
        "CUDA_ARCH=native python3 ds4-parity/compare_graph_restore_next_token.py "
        f"--live --workdir {B300_WORKDIR} --write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


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
