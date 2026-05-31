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
  M10.7d3a, M10.7d3b, M10.7d3c1, M10.7d3c2, M10.8a, M10.8b, M10.8c,
  M10.8d, M10.8e, M10.8f, M10.8g1, M10.8g2, M10.8g3a, M10.8g3b,
  M10.8g3c, M10.8g4a, M10.8g4b, M10.9a, M10.9b, M10.9c, M10.9d,
  M10.9e, M10.9f, M11.1, M11.2, M11.3, M11.4, M12.1, M12.2, M12.3, M12.4,
  M12.5, M12.6, M13.0, M13.1, M13.2, M13.3, M13.4, M13.5, the
  post-M13 roadmap decision, M14.0, M14.1a, M14.1b1, M14.1b2a,
  M14.1b2b1, and M14.1b2b2.

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
            (
                "M10.7d3c2 Rust post-restore KVC file smoke comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_post_restore_kvc_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8a MTP state-machine contract",
                [
                    sys.executable,
                    "ds4-parity/check_mtp_state_machine_contract.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8b Rust MTP decision planner comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_decision_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8c Rust MTP draft orchestration plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_draft_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8d Rust MTP exact-N=2 verifier plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_decode2_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8e Rust MTP suffix verifier plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_suffix_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8f Rust MTP frontier mutation plan comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_frontier_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g1 MTP stream parity contract",
                [
                    sys.executable,
                    "ds4-parity/check_mtp_stream_parity_contract.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g2 Rust MTP stream outcome planner",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_stream_plan.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g3a Rust MTP runtime guard plan",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_runtime_guard.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g3b MTP runtime no-drift comparator",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_runtime_no_drift.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g3c B300 MTP missing-support runtime smoke",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_runtime_missing_support.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g4a B300 MTP support branch decision",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_support_branch.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.8g4b B300 MTP end-to-end closure",
                [
                    sys.executable,
                    "ds4-parity/compare_mtp_end_to_end_closure.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.9a Runtime graph closure matrix",
                [
                    sys.executable,
                    "ds4-parity/check_runtime_graph_closure_matrix.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.9b Runtime graph route preflight",
                [
                    sys.executable,
                    "ds4-parity/check_runtime_graph_route_preflight.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.9c Runtime graph official-vector gate",
                [
                    sys.executable,
                    "ds4-parity/run_runtime_graph_official_vectors.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.9d Runtime graph long-context gate",
                [
                    sys.executable,
                    "ds4-parity/run_runtime_graph_long_context.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.9e Runtime graph tool/server gate",
                [
                    sys.executable,
                    "ds4-parity/run_tool_call_quality.py",
                    "--negative-test",
                ],
            ),
            (
                "M10.9f Runtime graph benchmark closure",
                [
                    sys.executable,
                    "ds4-parity/run_runtime_graph_bench.py",
                    "--negative-test",
                ],
            ),
            (
                "M11.1 Agent trace replay oracle",
                [
                    sys.executable,
                    "ds4-parity/compare_agent_trace_replay.py",
                    "--negative-test",
                ],
            ),
            (
                "M11.2 Agent rendered-context replay",
                [
                    sys.executable,
                    "ds4-parity/compare_agent_rendered_context.py",
                    "--negative-test",
                ],
            ),
            (
                "M11.3 Agent deterministic tool/session replay",
                [
                    sys.executable,
                    "ds4-parity/compare_agent_deterministic_replay.py",
                    "--negative-test",
                ],
            ),
            (
                "M11.4 Agent no-model loop smoke",
                [
                    sys.executable,
                    "ds4-parity/compare_agent_loop_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M12.1 Backend boundary inventory",
                [
                    sys.executable,
                    "ds4-parity/check_backend_boundary_inventory.py",
                    "--negative-test",
                ],
            ),
            (
                "M12.2 Backend operation tensor fixtures",
                [
                    sys.executable,
                    "ds4-parity/check_backend_operation_fixtures.py",
                    "--negative-test",
                ],
            ),
            (
                "M12.3 Backend facade replay harness",
                [
                    sys.executable,
                    "ds4-parity/check_backend_facade_replay.py",
                    "--negative-test",
                ],
            ),
            (
                "M12.4 Backend replacement slice",
                [
                    sys.executable,
                    "ds4-parity/check_backend_replacement_slice.py",
                    "--negative-test",
                ],
            ),
            (
                "M12.5 Backend runtime route gate",
                [
                    sys.executable,
                    "ds4-parity/check_backend_runtime_route_gate.py",
                    "--negative-test",
                ],
            ),
            (
                "M12.6 Backend replacement closure",
                [
                    sys.executable,
                    "ds4-parity/check_backend_replacement_closure.py",
                    "--negative-test",
                ],
            ),
            (
                "M13.0 Backend expansion decision",
                [
                    sys.executable,
                    "ds4-parity/check_backend_expansion_decision.py",
                    "--negative-test",
                ],
            ),
            (
                "M13.1 Backend expansion matrix",
                [
                    sys.executable,
                    "ds4-parity/check_backend_expansion_matrix.py",
                    "--negative-test",
                ],
            ),
            (
                "M13.2 Batched embedding replacement slice",
                [
                    sys.executable,
                    "ds4-parity/check_backend_batched_embedding_slice.py",
                    "--negative-test",
                ],
            ),
            (
                "M13.3 Indexed decode selection replacement slice",
                [
                    sys.executable,
                    "ds4-parity/check_backend_indexed_decode_slice.py",
                    "--negative-test",
                ],
            ),
            (
                "M13.4 Batch indexer fixture gap closure",
                [
                    sys.executable,
                    "ds4-parity/check_backend_batch_indexer_fixtures.py",
                    "--negative-test",
                ],
            ),
            (
                "M13.5 Expanded embedding/indexer route closure",
                [
                    sys.executable,
                    "ds4-parity/check_backend_expanded_route_closure.py",
                    "--negative-test",
                ],
            ),
            (
                "Post-M13 roadmap decision",
                [
                    sys.executable,
                    "ds4-parity/check_post_m13_roadmap_decision.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.0 CUDA Rust ownership inventory",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_ownership_inventory.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1a cuda-oxide host substrate smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_oxide_substrate_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b1 cuda-oxide model residency handles smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_residency_handles_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2a cuda-oxide model range copy smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_range_copy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2b1 cuda-oxide model range strategy smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_range_strategy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2b2 cuda-oxide model registered range smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_registered_range_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2b3a cuda-oxide model pageable HMM smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_pageable_hmm_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2b3b1 cuda-oxide model direct-I/O smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_direct_io_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2b3b2 cuda-oxide model async staging smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_async_staging_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b2c cuda-oxide model-map cache closure smoke",
                [
                    sys.executable,
                    "ds4-parity/check_model_map_closure_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b3a cuda-oxide allocation policy smoke",
                [
                    sys.executable,
                    "ds4-parity/check_allocation_policy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b3b cuda-oxide Q8/quality policy smoke",
                [
                    sys.executable,
                    "ds4-parity/check_q8_quality_policy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1b4 cuda-oxide fill/command-lifetime smoke",
                [
                    sys.executable,
                    "ds4-parity/check_fill_command_lifetime_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.1c cuda-oxide substrate route closure",
                [
                    sys.executable,
                    "ds4-parity/check_substrate_route_closure.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2a cuda-oxide add/repeat elementwise kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_elementwise_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2b1 cuda-oxide directional steering kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_directional_steering_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2b2 cuda-oxide SwiGLU libdevice kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_swiglu_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2c cuda-oxide embedding kernel-pair smoke",
                [
                    sys.executable,
                    "ds4-parity/check_embedding_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d1 cuda-oxide scalar indexer selection kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_scalar_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2a cuda-oxide direct-one indexer score kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_direct_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2b1 cuda-oxide base WMMA indexer score kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_wmma_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2b2a cuda-oxide WMMA32 indexer score kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_wmma32_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2b2b cuda-oxide WMMA64 indexer score kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_wmma64_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2b2c cuda-oxide WMMA128 indexer score and dispatch smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_wmma128_dispatch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2c1 cuda-oxide 1024-element bitonic top-k kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_topk1024_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2c2 cuda-oxide power-of-two top-k kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_topk_pow2_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2c3 cuda-oxide packed-key top-k equivalent smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_topk_packed_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2c4 cuda-oxide chunk/tree top-k kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_topk_tree_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2d2c5 cuda-oxide indexed-sort and top-k dispatch smoke",
                [
                    sys.executable,
                    "ds4-parity/check_indexer_topk_dispatch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.2e cuda-oxide kernel-family closure",
                [
                    sys.executable,
                    "ds4-parity/check_m14_2_kernel_closure.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3a cuda-oxide plain and weighted RMS norm smoke",
                [
                    sys.executable,
                    "ds4-parity/check_rms_norm_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3b1 cuda-oxide fused QKV and head RMS norm smoke",
                [
                    sys.executable,
                    "ds4-parity/check_fused_rms_norm_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3b2 cuda-oxide head RMS RoPE-tail smoke",
                [
                    sys.executable,
                    "ds4-parity/check_head_rms_rope_tail_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3c1 cuda-oxide base F16 and F32 projection smoke",
                [
                    sys.executable,
                    "ds4-parity/check_dense_projection_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3c2 cuda-oxide ordered and serial F16 projection smoke",
                [
                    sys.executable,
                    "ds4-parity/check_ordered_projection_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3c3 cuda-oxide BLAS projection and conversion smoke",
                [
                    sys.executable,
                    "ds4-parity/check_blas_projection_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3d1 cuda-oxide Q8 conversion kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_q8_conversion_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3d2 cuda-oxide Q8 matmul kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_q8_matmul_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3d3 cuda-oxide Q8 specialized matmul kernel smoke",
                [
                    sys.executable,
                    "ds4-parity/check_q8_specialized_matmul_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.3d4 cuda-oxide Q8 DP4A dispatch smoke",
                [
                    sys.executable,
                    "ds4-parity/check_q8_dp4a_dispatch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4a cuda-oxide RoPE tail and FP8 KV quantization smoke",
                [
                    sys.executable,
                    "ds4-parity/check_rope_kv_quantization_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4b cuda-oxide raw KV storage and indexer QAT smoke",
                [
                    sys.executable,
                    "ds4-parity/check_raw_kv_indexer_qat_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4c1 cuda-oxide composed KV and compressor-store smoke",
                [
                    sys.executable,
                    "ds4-parity/check_composed_kv_compressor_store_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4c2 cuda-oxide compressor pool and ratio-4 shift smoke",
                [
                    sys.executable,
                    "ds4-parity/check_compressor_pool_shift_kernel_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4c3a cuda-oxide compressor update orchestration smoke",
                [
                    sys.executable,
                    "ds4-parity/check_compressor_update_orchestration_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4c3b cuda-oxide compressor prefill orchestration smoke",
                [
                    sys.executable,
                    "ds4-parity/check_compressor_prefill_orchestration_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d1 cuda-oxide single-token mixed attention decode smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_decode_single_mixed_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d2 cuda-oxide generic batched mixed attention decode smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_decode_batch_mixed_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d3 cuda-oxide heads8 online attention decode smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_decode_heads8_online_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d4 cuda-oxide generic attention prefill smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_prefill_generic_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d5 cuda-oxide optimized attention prefill smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_prefill_optimized_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d6 cuda-oxide generic indexed mixed attention smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_indexed_generic_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d7 cuda-oxide optimized indexed attention smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_indexed_optimized_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d8a cuda-oxide native output-Q8 attention smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_output_q8_native_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.4d8b cuda-oxide cuBLAS output-Q8 attention smoke",
                [
                    sys.executable,
                    "ds4-parity/check_attention_output_q8_cublas_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5a cuda-oxide scalar router smoke",
                [
                    sys.executable,
                    "ds4-parity/check_router_scalar_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5b cuda-oxide optimized router smoke",
                [
                    sys.executable,
                    "ds4-parity/check_router_optimized_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c1 cuda-oxide routed MoE F32 fallback smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_f32_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2a cuda-oxide quantized single-token routed MoE smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_quantized_single_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2b1 cuda-oxide routed MoE sorted-pair metadata smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_sorted_pairs_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2b2 cuda-oxide sorted-P2 routed MoE smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_sorted_p2_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c1 cuda-oxide routed MoE expert-tile metadata smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_expert_tiles_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c2 cuda-oxide routed MoE tile8 row32 smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_tile8_row32_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c3 cuda-oxide routed MoE tile4 row32 smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_tile4_row32_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c4 cuda-oxide routed MoE atomic-down smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_atomic_down_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c5 cuda-oxide routed MoE tile16 row32 smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_tile16_row32_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c6 cuda-oxide routed MoE gate-rowspan smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_gate_rowspan_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2c7 cuda-oxide routed MoE down-rowspan smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_down_rowspan_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2d cuda-oxide single-token Q4_K routed MoE smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_q4_k_single_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2e cuda-oxide routed MoE shared-cache smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_shared_cache_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5c2f cuda-oxide routed MoE qwarp fallback smoke",
                [
                    sys.executable,
                    "ds4-parity/check_routed_moe_qwarp_fallback_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.5d cuda-oxide hyperconnection smoke",
                [
                    sys.executable,
                    "ds4-parity/check_hyperconnection_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6a CUDA route promotion blocker",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_route_promotion_gate.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b1 Rust CUDA resource ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_resource_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2a Rust CUDA tensor-fill ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_tensor_fill_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b1 Rust CUDA embedded elementwise ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_elementwise_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2a Rust CUDA directional-steering ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_directional_steering_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b1 Rust CUDA SwiGLU libdevice ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_swiglu_libdevice_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2a Rust CUDA plain RMS ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_plain_rms_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b1 Rust CUDA weighted RMS device-copy ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_weighted_rms_device_copy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2a Rust CUDA basic model-control device-copy ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_device_copy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b1 Rust CUDA registered-attempt device-copy fallback ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_registered_fallback_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2a Rust CUDA pageable HMM fallback ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_pageable_hmm_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b1 Rust CUDA chunk-selected model-copy ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_chunk_selected_copy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2a Rust CUDA whole-map registration precedence ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_whole_registration_precedence_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b1 Rust CUDA buffered fd cache ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_buffered_fd_cache_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2a Rust CUDA direct-I/O fd cache ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_direct_io_fd_cache_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b1 Rust CUDA direct-I/O error-disable ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_direct_io_error_disable_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2a Rust CUDA direct-I/O async staging ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_direct_io_async_staging_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b1 Rust CUDA buffered fd async staging ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_buffered_fd_async_staging_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2a Rust CUDA public fd arena suballocation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_arena_suballocation_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b1 Rust CUDA public fd cache budget ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_cache_budget_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2a Rust CUDA public fd source-page progress ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_source_page_progress_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public registration-disable ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_registration_disable_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public full-model copy ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_full_model_copy_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public direct-model read ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_direct_model_read_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public default-fd selection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_default_fd_selection_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public fd-budget cache-result ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_budget_cache_result_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public fd-arena failure-selection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_arena_failure_selection_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba Rust CUDA public fd-upload failure-continuation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_upload_failure_continuation_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba Rust CUDA public fd stage-pool reuse ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_stage_pool_reuse_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba Rust CUDA public fd stage-allocation failure-continuation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_stage_allocation_failure_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba Rust CUDA public fd-read failure-continuation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_read_failure_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba Rust CUDA public fd event-record failure-continuation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_event_record_failure_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba Rust CUDA public fd event-wait failure-continuation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_event_wait_failure_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba Rust CUDA public fd final-sync failure-continuation ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_model_control_fd_final_sync_failure_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba Rust CUDA public single-token F16 projection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_matmul_f16_single_token_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbba Rust CUDA public single-token paired F16 projection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_matmul_f16_pair_single_token_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbba Rust CUDA public single-token F32 projection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_matmul_f32_single_token_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbba Rust CUDA public multi-token F32 BLAS projection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_matmul_f32_multi_token_blas_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba Rust CUDA public multi-token F16 BLAS projection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_matmul_f16_multi_token_blas_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba Rust CUDA public Q8 preload and quality-controls ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_q8_quality_controls_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba Rust CUDA public Q8 matmul ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_q8_matmul_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba Rust CUDA public hyperconnection expansion ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_hc_expand_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba Rust CUDA public fused Q8 hyperconnection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_fused_q8_hc_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba Rust CUDA public shared gate/up Q8 SwiGLU ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_shared_gate_up_swiglu_q8_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba Rust CUDA public hyperconnection weighted-sum ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_hc_weighted_sum_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba Rust CUDA public hyperconnection split-Sinkhorn ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_hc_split_sinkhorn_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba Rust CUDA public hyperconnection split weighted-sum ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_hc_split_weighted_sum_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba Rust CUDA public hyperconnection split weighted-sum norm ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_hc_split_weighted_sum_norm_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba Rust CUDA public output hyperconnection weights ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_output_hc_weights_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public embedding hyperconnection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_embedding_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public head RMS norm ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_head_rms_norm_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public FP8 KV quantization ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_fp8_kv_quantize_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public indexer QAT ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_indexer_qat_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public standalone RoPE ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_rope_tail_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public raw KV storage ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_raw_kv_storage_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public composed FP8 raw KV storage ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_composed_kv_fp8_raw_store_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public compressor batch-store ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_compressor_store_batch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public compressor ratio-4 state ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_compressor_state_ratio4_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public compressor ratio-4 replay ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_compressor_replay_ratio4_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public compressor update ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_compressor_update_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public general compressor prefill ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_compressor_prefill_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public single-token attention decode heads ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_attention_decode_heads_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public batched attention decode ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_attention_decode_batch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public indexed batched attention ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_attention_indexed_batch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public low-Q8 attention output ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_attention_output_low_q8_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public batched Q8 attention output ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_attention_output_q8_batch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public attention-prefill ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_attention_prefill_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public router-selection ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_router_selection_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public single-token routed-MoE ABI smoke",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_one_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba CUDA batched routed-MoE mid contract repair",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_routed_moe_batch_mid_contract_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded sorted routed-MoE ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_sorted_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded expert-tile metadata ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_expert_tile_metadata_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded row32 tiled atomic ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_row32_tiled_atomic_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded tile16 row32 atomic ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_tile16_row32_atomic_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded gate row-span ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_gate_rowspan_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded down row-span ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_down_rowspan_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA embedded shared-cache ABI module",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_shared_cache_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA batched F32 routed-MoE ABI prerequisite",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_batched_f32_module_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA sorted routed-MoE host launch methods",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_sorted_host_launch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA tiled routed-MoE host launch methods",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_tiled_host_launch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public batched routed-MoE ABI dispatch",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_routed_moe_batch_dispatch_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public fused QKV RMS rows ABI",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_fused_qkv_rms_rows_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public DSV4 top-k mask ABI",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_topk_mask_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public indexer score-dispatch ABI",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_indexer_scores_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA public indexer top-k dispatch ABI",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_abi_indexer_topk_smoke.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA production facade shared-library link",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_production_facade_link.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA engine route blocker probe",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_engine_route_blocker.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA generic decode-attention parallel repair",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_generic_decode_attention_parallel_repair.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA reciprocal-square-root route correctness repair",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_rsqrt_route_correctness_repair.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA long-route attention parallel repair",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_long_route_attention_parallel_repair.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA promotion acceptance matrix",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_promotion_acceptance_matrix.py",
                    "--negative-test",
                ],
            ),
            (
                "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust CUDA backend identity-log compatibility repair",
                [
                    sys.executable,
                    "ds4-parity/check_cuda_rust_backend_identity_log_compatibility_repair.py",
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
            name="M10.7d3c2 B300 Rust post-restore KVC file smoke rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Post-restore KVC file smoke requires the B300 pod because "
                "the M7.8 disk payload and memory snapshot raw bodies are "
                "hash-only and remain in /workspace/ds4"
            ),
            rerun_command=b300_rust_post_restore_kvc_smoke_command(),
        ),
        ReportItem(
            name="M10.8a B300 MTP support-artifact availability rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "MTP-enabled smoke is blocked until a B300 MTP support GGUF "
                "is present; this check verifies the expected missing support "
                "artifact and candidate search result"
            ),
            rerun_command=b300_mtp_availability_command(),
        ),
        ReportItem(
            name="M10.8g3c B300 MTP missing-support runtime smoke rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "MTP-enabled stream parity is blocked until a B300 MTP support "
                "GGUF is present; this rerun proves the Rust runtime fails "
                "closed before stream mutation when the configured support "
                "artifact is absent"
            ),
            rerun_command=b300_mtp_missing_support_runtime_command(),
        ),
        ReportItem(
            name="M10.8g4a B300 MTP support branch decision rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "MTP-enabled stream parity is blocked until a B300 MTP support "
                "GGUF is present; this rerun refreshes the branch decision "
                "that selects the final support comparator or explicit blocker"
            ),
            rerun_command=b300_mtp_support_branch_decision_command(),
        ),
        ReportItem(
            name="M10.8g4b B300 MTP end-to-end closure rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "MTP-enabled stream parity is blocked until a B300 MTP support "
                "GGUF is present; this rerun refreshes the branch decision and "
                "final explicit blocker closure"
            ),
            rerun_command=b300_mtp_end_to_end_closure_command(),
        ),
        ReportItem(
            name="M10.9a B300 runtime graph fixture-readiness rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "M10.9 model-backed runtime graph gates require the B300 model, "
                "official-vector fixture, benchmark prompt, and committed M0.6 "
                "benchmark CSV fixtures"
            ),
            rerun_command=b300_runtime_graph_fixture_readiness_command(),
        ),
        ReportItem(
            name="M10.9c B300 Rust runtime official-vector rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust runtime official-vector comparison requires the B300 pod, "
                "the q2-imatrix GGUF, CUDA linkage, and raw Rust stdout/stderr "
                "capture"
            ),
            rerun_command=b300_runtime_graph_official_vectors_command(),
        ),
        ReportItem(
            name="M10.9d B300 Rust runtime long-context rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust runtime long-context comparison requires the B300 pod, "
                "the q2-imatrix GGUF, CUDA linkage, current-C long-context "
                "pass/fail evidence, and raw Rust stdout/stderr capture"
            ),
            rerun_command=b300_runtime_graph_long_context_command(),
        ),
        ReportItem(
            name="M10.9e B300 Rust runtime tool/server rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust runtime tool/server comparison requires the B300 pod, "
                "the q2-imatrix GGUF, CUDA linkage, current-C tool-call "
                "quality evidence, raw Rust responses, traces, and logs"
            ),
            rerun_command=b300_runtime_graph_tool_server_command(),
        ),
        ReportItem(
            name="M10.9f B300 Rust runtime benchmark closure rerun",
            kind="b300-oracle",
            status="SKIP",
            reason=(
                "Rust runtime benchmark closure requires the B300 pod, "
                "the q2-imatrix GGUF, CUDA linkage, M0.6 benchmark CSV "
                "baseline, Rust graph runtime CSVs, and quality-gate evidence"
            ),
            rerun_command=b300_runtime_graph_benchmark_closure_command(),
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


def b300_rust_post_restore_kvc_smoke_command() -> str:
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
    summary = "/tmp/ds4-m107d3c2-post-restore-kvc.json"
    smoke = b300_exec(
        "python3 ds4-parity/compare_post_restore_kvc_smoke.py "
        f"--live --workdir {B300_WORKDIR} "
        "--output-dir /tmp/ds4-m107d3c2-kvc "
        f"--write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/kv/m10.7d3/rust-b300-post-restore-kvc.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_mtp_availability_command() -> str:
    return b300_exec(
        "ls -l /workspace/ds4/ds4flash.gguf; "
        "readlink -f /workspace/ds4/ds4flash.gguf; "
        "test ! -e /workspace/ds4/missing-mtp.gguf; "
        "candidates=$(find /workspace/ds4 -maxdepth 3 -type f "
        "\\( -iname '*mtp*.gguf' -o -iname '*draft*.gguf' \\) -print | sort); "
        "printf 'mtp_candidates=%s\\n' \"$candidates\"; "
        "test -z \"$candidates\""
    )


def b300_mtp_missing_support_runtime_command() -> str:
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
    summary = "/tmp/ds4-m108g3c-missing-support-runtime.json"
    smoke = b300_exec(
        "export PATH=/tmp/cargo/bin:$PATH CARGO_HOME=/tmp/cargo RUSTUP_HOME=/tmp/rustup; "
        "CUDA_ARCH=native cargo build -p ds4-engine --bin ds4-cli-one-shot-rs && "
        "CUDA_ARCH=native python3 ds4-parity/compare_mtp_runtime_missing_support.py "
        f"--live --workdir {B300_WORKDIR} "
        "--candidate-binary target/debug/ds4-cli-one-shot-rs "
        f"--write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_mtp_support_branch_decision_command() -> str:
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
    summary = "/tmp/ds4-m108g4a-support-branch-decision.json"
    smoke = b300_exec(
        "python3 ds4-parity/compare_mtp_support_branch.py "
        f"--live --workdir {B300_WORKDIR} "
        f"--write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.8g4a/support-branch-decision.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_mtp_end_to_end_closure_command() -> str:
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
    branch = "/tmp/ds4-m108g4a-support-branch-decision.json"
    summary = "/tmp/ds4-m108g4b-end-to-end-closure.json"
    smoke = b300_exec(
        "python3 ds4-parity/compare_mtp_support_branch.py "
        f"--live --workdir {B300_WORKDIR} "
        f"--write-summary {branch} --negative-test && "
        "python3 ds4-parity/compare_mtp_end_to_end_closure.py "
        f"--branch-decision {branch} --write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_runtime_graph_fixture_readiness_command() -> str:
    return b300_exec(
        "target=$(readlink -f /workspace/ds4/ds4flash.gguf); "
        "printf 'resolved_model=%s\\n' \"$target\"; "
        "stat -c 'resolved_model_bytes=%s' \"$target\"; "
        "sha256sum tests/test-vectors/official.vec speed-bench/promessi_sposi.txt; "
        "test -f ds4-parity/baselines/bench/m0.6/csv/b300-short.csv; "
        "test -f ds4-parity/baselines/bench/m0.6/csv/b300-long.csv; "
        "python3 -m json.tool ds4-parity/baselines/bench/m0.6/logs/csv-summary.json >/dev/null; "
        "printf 'm109_fixture_probe=ok\\n'"
    )


def b300_runtime_graph_official_vectors_command() -> str:
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
    summary = "/tmp/ds4-m109c-official-vectors.json"
    smoke = b300_exec(
        "CUDA_ARCH=native python3 ds4-parity/run_runtime_graph_official_vectors.py "
        f"--workdir {B300_WORKDIR} "
        f"--model {B300_MODEL} "
        f"--write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_runtime_graph_long_context_command() -> str:
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
    summary = "/tmp/ds4-m109d-long-context.json"
    smoke = b300_exec(
        "CUDA_ARCH=native python3 ds4-parity/run_runtime_graph_long_context.py "
        f"--workdir {B300_WORKDIR} "
        f"--model {B300_MODEL} "
        f"--write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_runtime_graph_tool_server_command() -> str:
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
    summary = "/tmp/ds4-m109e-tool-server.json"
    smoke = b300_exec(
        "CUDA_ARCH=native make ds4_test && "
        "CUDA_ARCH=native cargo build -p ds4-engine --bin ds4-server-runtime-rs && "
        "CUDA_ARCH=native python3 ds4-parity/run_tool_call_quality.py "
        "--server-bin target/debug/ds4-server-runtime-rs "
        f"--model {B300_MODEL} "
        "--backend cuda --runtime-graph graph "
        "--out-dir /tmp/ds4-m109e-tool-call-quality "
        f"--write-summary {summary} --ready-timeout 360 --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
        ]
    )
    return f"{source_refresh} && {smoke} && {copy_back}"


def b300_runtime_graph_benchmark_closure_command() -> str:
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
    summary = "/tmp/ds4-m109f-benchmark-closure.json"
    smoke = b300_exec(
        "CUDA_ARCH=native python3 ds4-parity/run_runtime_graph_bench.py "
        f"--workdir {B300_WORKDIR} --model {B300_MODEL} "
        "--output-dir /tmp/ds4-m109f-bench "
        f"--write-summary {summary} --negative-test"
    )
    copy_back = shell_join(
        prefix
        + [
            "cp",
            f"{KUBE_POD}:{summary}",
            "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
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
