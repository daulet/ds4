#!/usr/bin/env python3
"""Validate the M10.9 runtime graph closure matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.9a/runtime-graph-closure-matrix.json"

SCHEMA = "ds4.runtime_graph_closure_matrix.v1"
SOURCE = "m10.9-runtime-graph-closure-matrix"
MILESTONE = "M10.9a"
PARENT = "M10.9"
NEXT_STAGE = "M10.9b"
MODEL_PATH = "/workspace/ds4/ds4flash.gguf"
RESOLVED_MODEL = (
    "/workspace/ds4/gguf/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)
MODEL_BYTES = 86_720_111_488
MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
OFFICIAL_VEC_SHA256 = "0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac"
BENCH_PROMPT_SHA256 = "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f"


def m10_9b_rerun() -> str:
    return (
        "python3 ds4-parity/check_runtime_graph_route_preflight.py "
        "--write-summary ds4-parity/baselines/graph/m10.9b/"
        "runtime-graph-route-preflight.json --negative-test"
    )


def m10_9c_rerun() -> str:
    return (
        "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig "
        "--context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- "
        "tar -xf - -C /workspace/ds4 && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; "
        "CUDA_ARCH=native python3 ds4-parity/run_runtime_graph_official_vectors.py "
        "--workdir /workspace/ds4 --model /workspace/ds4/ds4flash.gguf "
        "--write-summary /tmp/ds4-m109c-official-vectors.json --negative-test' && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default cp ds4-rust-port-b300:/tmp/ds4-m109c-official-vectors.json "
        "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json"
    )


def m10_9d_rerun() -> str:
    return (
        "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig "
        "--context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- "
        "tar -xf - -C /workspace/ds4 && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; "
        "CUDA_ARCH=native python3 ds4-parity/run_runtime_graph_long_context.py "
        "--workdir /workspace/ds4 --model /workspace/ds4/ds4flash.gguf "
        "--write-summary /tmp/ds4-m109d-long-context.json --negative-test' && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default cp ds4-rust-port-b300:/tmp/ds4-m109d-long-context.json "
        "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json"
    )


def m10_9e_rerun() -> str:
    return (
        "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig "
        "--context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- "
        "tar -xf - -C /workspace/ds4 && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; "
        "CUDA_ARCH=native make ds4_test && "
        "CUDA_ARCH=native cargo build -p ds4-engine --bin ds4-server-runtime-rs && "
        "CUDA_ARCH=native python3 ds4-parity/run_tool_call_quality.py "
        "--server-bin target/debug/ds4-server-runtime-rs "
        "--model /workspace/ds4/ds4flash.gguf --backend cuda "
        "--runtime-graph graph "
        "--out-dir /tmp/ds4-m109e-tool-call-quality --ready-timeout 360 "
        "--write-summary /tmp/ds4-m109e-tool-server.json --negative-test' && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default cp ds4-rust-port-b300:/tmp/ds4-m109e-tool-server.json "
        "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json"
    )


def m10_9f_rerun() -> str:
    return (
        "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig "
        "--context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- "
        "tar -xf - -C /workspace/ds4 && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; "
        "CUDA_ARCH=native python3 ds4-parity/run_runtime_graph_bench.py "
        "--workdir /workspace/ds4 --model /workspace/ds4/ds4flash.gguf "
        "--output-dir /tmp/ds4-m109f-bench --write-summary "
        "/tmp/ds4-m109f-benchmark-closure.json --negative-test' && "
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 "
        "-n default cp ds4-rust-port-b300:/tmp/ds4-m109f-benchmark-closure.json "
        "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json"
    )


GATES: list[dict[str, Any]] = [
    {
        "id": "M10.9b",
        "name": "Rust Runtime Graph Route Switch And Preflight",
        "oracle": [
            "current Rust runtime target-stream path",
            "M10.5 through M10.8 graph scheduling contracts",
            "current C graph runtime option behavior",
        ],
        "fixtures": [
            "one-shot runtime options",
            "interactive runtime options",
            "server runtime options",
            "CUDA and non-CUDA backend selectors",
            "unsupported graph route cases",
            "cache/KVC preflight cases",
        ],
        "comparator": "ds4-parity/check_runtime_graph_route_preflight.py",
        "artifact": "ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json",
        "rerun_command": m10_9b_rerun(),
        "acceptance": [
            "graph route selection is explicit",
            "default runtime behavior is unchanged",
            "unsupported graph-runtime cases fail before stream or cache mutation",
        ],
        "drift_policy": [
            "option names exact",
            "fail-closed categories exact",
            "output bytes exact",
            "checkpoint/cache/KVC deltas exact",
            "logs and timings normalized",
        ],
        "claim_boundary": "no model-backed Rust graph parity claim",
    },
    {
        "id": "M10.9c",
        "name": "B300 Official-Vector Rust Runtime Gate",
        "oracle": [
            "M0.3 current-C ./ds4_test --logprob-vectors baseline",
            "M6 numeric tolerance policy",
            "M10.9b Rust runtime graph route",
        ],
        "fixtures": [
            MODEL_PATH,
            "tests/test-vectors/official.vec",
            "CUDA backend",
            "deterministic generation settings",
            "captured Rust runtime vector output",
        ],
        "comparator": "ds4-parity/run_runtime_graph_official_vectors.py",
        "artifact": "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
        "rerun_command": m10_9c_rerun(),
        "acceptance": [
            "selected greedy tokens match current-C baseline exactly",
            "numeric scores stay within M6 tolerance",
            "current-C skipped rows remain explicit",
        ],
        "drift_policy": [
            "selected token IDs exact",
            "fixture hash exact",
            "model hash exact",
            "backend exact",
            "score tolerances follow M6 policy",
        ],
        "claim_boundary": "official-vector gate only, not full Milestone 10 closure",
    },
    {
        "id": "M10.9d",
        "name": "B300 Long-Context Rust Runtime Gate",
        "oracle": [
            "current-C ./ds4_test --long-context baseline behavior",
            "M7/M10 long-context graph checkpoint contracts",
            "M10.9c official-vector runtime evidence",
        ],
        "fixtures": [
            "long-context fact-recall prompt",
            MODEL_PATH,
            "CUDA backend",
            "graph route selector",
            "retained Rust stdout/stderr/log artifacts",
        ],
        "comparator": "ds4-parity/run_runtime_graph_long_context.py",
        "artifact": "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
        "rerun_command": m10_9d_rerun(),
        "acceptance": [
            "Rust graph runtime completes long-context gate",
            "no fallback to C host route",
            "nondeterministic score surfaces are classified explicitly",
        ],
        "drift_policy": [
            "command line exact",
            "model/backend identity exact",
            "context length exact",
            "pass/fail markers exact",
            "floating score surfaces use tolerance or nondeterminism labels",
        ],
        "claim_boundary": "long-context gate only, not benchmark or tool quality",
    },
    {
        "id": "M10.9e",
        "name": "Tool-Call Quality And Server Replay Rust Runtime Gate",
        "oracle": [
            "current-C ./ds4_test --tool-call-quality",
            "M9 server/runtime replay",
            "ds4-parity/run_tool_call_quality.py classifier",
        ],
        "fixtures": [
            "B300 Rust server runtime binary",
            "OpenAI tool-call request fixture",
            "M9 server/cache request fixtures",
            "trace output",
            "cache/KVC directories",
            "retained raw responses",
        ],
        "comparator": "ds4-parity/run_tool_call_quality.py",
        "artifact": "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
        "rerun_command": m10_9e_rerun(),
        "acceptance": [
            "tool-call classification passes through Rust graph runtime",
            "server/runtime parity remains green",
            "cache/KVC behavior matches M9 replay contracts",
        ],
        "drift_policy": [
            "HTTP status exact",
            "response schema exact",
            "tool name and arguments exact",
            "trace/cache ledger markers exact",
            "request IDs and timings normalized",
        ],
        "claim_boundary": "tool/server gate only, not benchmark closure",
    },
    {
        "id": "M10.9f",
        "name": "Benchmark Comparator And Milestone 10 Closure",
        "oracle": [
            "M0.6 ds4-bench short/long CSV baseline",
            "M10.9c through M10.9e quality gates",
            "same B300 model/backend identity",
        ],
        "fixtures": [
            "speed-bench/promessi_sposi.txt",
            MODEL_PATH,
            "CUDA backend",
            "Rust graph runtime benchmark CSVs",
            "capture metadata",
        ],
        "comparator": "ds4-parity/run_runtime_graph_bench.py",
        "artifact": "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
        "rerun_command": m10_9f_rerun(),
        "acceptance": [
            "benchmark workload shape matches exactly",
            "throughput regression beyond threshold is documented",
            "all M10.9 quality gates are green",
            "Milestone 10 closure does not claim backend replacement",
        ],
        "drift_policy": [
            "CSV schema exact",
            "prompt hash exact",
            "model hash exact",
            "backend exact",
            "context frontiers exact",
            "generation-token counts exact",
            "kvcache_bytes exact",
            "throughput uses M0.6 regression threshold",
        ],
        "claim_boundary": "final Milestone 10 closure, not backend replacement",
    },
]


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


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected JSON object")
    return obj


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def build_summary() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "parent": PARENT,
        "next_stage": NEXT_STAGE,
        "last_completed": "M10.8g4b",
        "inputs": {
            "roadmap": "RUST_PORT_ROADMAP.md",
            "todo": ".memory/TODO.md",
            "status": ".memory/status.md",
            "unified_report": "ds4-parity/run_parity_report.py",
            "readme": "ds4-parity/README.md",
            "m10_8g4b_closure": "ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json",
        },
        "b300_fixture_inventory": {
            "context": "hou2-prod1",
            "namespace": "default",
            "pod": "ds4-rust-port-b300",
            "temp_kubeconfig": "/tmp/ds4-hou2-prod1.kubeconfig",
            "model_path": MODEL_PATH,
            "resolved_model_path": RESOLVED_MODEL,
            "model_sha256": MODEL_SHA256,
            "model_bytes": MODEL_BYTES,
            "official_vec": {
                "path": "tests/test-vectors/official.vec",
                "sha256": OFFICIAL_VEC_SHA256,
            },
            "bench_prompt": {
                "path": "speed-bench/promessi_sposi.txt",
                "sha256": BENCH_PROMPT_SHA256,
            },
            "benchmark_csvs": [
                "ds4-parity/baselines/bench/m0.6/csv/b300-short.csv",
                "ds4-parity/baselines/bench/m0.6/csv/b300-long.csv",
            ],
        },
        "gates": copy.deepcopy(GATES),
        "claim_policy": {
            "runtime_graph_parity": "blocked_until_m10.9b_through_m10.9f_pass",
            "backend_replacement": "not_claimed_by_milestone_10",
            "benchmark_claims": "same_b300_model_backend_only",
            "must_not_report_as": [
                "backend replacement",
                "full runtime graph parity before M10.9f",
                "model-free proof of model-backed quality",
            ],
        },
    }


def validate(summary: dict[str, Any]) -> Report:
    report = Report()
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == SOURCE, "summary source drift")
    report.check(summary.get("milestone") == MILESTONE, "summary milestone drift")
    report.check(summary.get("parent") == PARENT, "summary parent drift")
    report.check(summary.get("next_stage") == NEXT_STAGE, "next stage drift")
    report.check(summary.get("last_completed") == "M10.8g4b", "last completed milestone drift")
    validate_inputs(report, summary.get("inputs"))
    validate_b300_inventory(report, summary.get("b300_fixture_inventory"))
    validate_gates(report, summary.get("gates"))
    validate_claim_policy(report, summary.get("claim_policy"))
    validate_static_files(report)
    validate_static_wiring(report)
    return report


def validate_inputs(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "inputs missing")
    if not isinstance(value, dict):
        return
    expected = build_summary()["inputs"]
    for key, expected_value in expected.items():
        report.check(value.get(key) == expected_value, f"input {key} drift")


def validate_b300_inventory(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "B300 fixture inventory missing")
    if not isinstance(value, dict):
        return
    expected = build_summary()["b300_fixture_inventory"]
    for key in ("context", "namespace", "pod", "temp_kubeconfig"):
        report.check(value.get(key) == expected[key], f"B300 {key} drift")
    for key in ("model_path", "resolved_model_path", "model_sha256", "model_bytes"):
        report.check(value.get(key) == expected[key], f"B300 model {key} drift")
    for nested in ("official_vec", "bench_prompt"):
        got = value.get(nested)
        report.check(isinstance(got, dict), f"B300 {nested} missing")
        if isinstance(got, dict):
            report.check(got == expected[nested], f"B300 {nested} drift")
    report.check(value.get("benchmark_csvs") == expected["benchmark_csvs"], "benchmark CSV path drift")


def validate_gates(report: Report, value: Any) -> None:
    report.check(isinstance(value, list), "gates missing")
    if not isinstance(value, list):
        return
    expected_ids = [gate["id"] for gate in GATES]
    got_ids = [gate.get("id") for gate in value if isinstance(gate, dict)]
    report.check(got_ids == expected_ids, "gate order/id drift")
    report.check(len(value) == len(GATES), "gate count drift")
    for idx, expected in enumerate(GATES):
        report.check(idx < len(value), f"gate {expected['id']} missing")
        if idx >= len(value) or not isinstance(value[idx], dict):
            continue
        gate = value[idx]
        for key, expected_value in expected.items():
            report.check(gate.get(key) == expected_value, f"{expected['id']} {key} drift")
        report.check(gate.get("artifact", "").startswith("ds4-parity/baselines/graph/"), f"{expected['id']} artifact outside graph baselines")
        report.check("kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig" in gate.get("rerun_command", "") or expected["id"] == "M10.9b", f"{expected['id']} missing temp kubeconfig rerun")
        report.check("claim_boundary" in gate, f"{expected['id']} missing claim boundary")


def validate_claim_policy(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "claim policy missing")
    if not isinstance(value, dict):
        return
    expected = build_summary()["claim_policy"]
    report.check(value == expected, "claim policy drift")


def validate_static_files(report: Report) -> None:
    required = [
        "ds4-parity/baselines/logs/m0.3-b300-logprob-vectors.log",
        "ds4-parity/baselines/bench/m0.6/csv/b300-short.csv",
        "ds4-parity/baselines/bench/m0.6/csv/b300-long.csv",
        "ds4-parity/baselines/bench/m0.6/logs/csv-summary.json",
        "ds4-parity/run_server_parity_report.py",
        "ds4-parity/run_tool_call_quality.py",
        "ds4-parity/run_runtime_graph_bench.py",
        "ds4-parity/compare_bench_csv.py",
        "ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json",
    ]
    for path in required:
        report.check((ROOT / path).exists(), f"required file missing: {path}")


def validate_static_wiring(report: Report) -> None:
    texts = {
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(),
        "todo": (ROOT / ".memory/TODO.md").read_text(),
        "status": (ROOT / ".memory/status.md").read_text(),
        "readme": (ROOT / "ds4-parity/README.md").read_text(),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(),
    }
    report.check("M10.9a: Runtime Graph Closure Matrix And Rerun Contract" in texts["roadmap"], "roadmap M10.9a missing")
    report.check("M10.9f: Benchmark Comparator And Milestone 10 Closure" in texts["roadmap"], "roadmap M10.9f missing")
    report.check("M10.9a: Runtime Graph Closure Matrix And Rerun Contract" in texts["todo"], "TODO M10.9a missing")
    status_has_m10_9a = (
        "Active item: M10.9a Runtime Graph Closure Matrix And Rerun Contract" in texts["status"]
        or "Earlier M10.9a Runtime Graph Closure Matrix And Rerun Contract" in texts["status"]
    )
    status_has_expected_active = (
        "Active item: M10.9a Runtime Graph Closure Matrix And Rerun Contract" in texts["status"]
        or "Active item: M10.9b Rust Runtime Graph Route Switch And Preflight" in texts["status"]
        or "Active item: M10.9c B300 Official-Vector Rust Runtime Gate" in texts["status"]
        or "Active item: M10.9d B300 Long-Context Rust Runtime Gate" in texts["status"]
        or "Active item: M10.9e Tool-Call Quality And Server Replay Rust Runtime Gate" in texts["status"]
        or "Active item: M10.9f Benchmark Comparator And Milestone 10 Closure" in texts["status"]
        or "Active item: M11 Agent Trace Replay" in texts["status"]
        or "Active item: M11.1 Agent Trace Replay Oracle And Fixture Contract" in texts["status"]
        or "Active item: M11.2 Rust Agent Rendered Context Replay" in texts["status"]
        or "Active item: M11.3 Deterministic Tool Stub And Session Command Replay" in texts["status"]
        or "Active item: M11.4 Rust Agent Loop And Manual Smoke" in texts["status"]
        or "Active item: M12 Backend Replacement Parity Split Planning" in texts["status"]
        or "Active item: M12.1 Backend Boundary Inventory And Claim Matrix" in texts["status"]
        or "Active item: M12.2 Operation Tensor Fixture Capture" in texts["status"]
        or "Active item: M12.3 Rust Backend Facade Parity Harness" in texts["status"]
        or "Active item: M12.4 First Backend Replacement Slice" in texts["status"]
        or "Active item: M12.5 Runtime Backend Route Gate" in texts["status"]
        or "Active item: M12.6 Backend Replacement Closure And Removal Decision" in texts["status"]
        or "Active item: post-M12 roadmap decision" in texts["status"]
        or "Active item: M13" in texts["status"]
    )
    report.check(status_has_m10_9a, "status M10.9a missing")
    report.check(status_has_expected_active, "status active M10.9a/M10.9b/M10.9c/M10.9d/M10.9e/M10.9f/M11 missing")
    report.check("check_runtime_graph_closure_matrix.py --negative-test" in texts["readme"], "README matrix command missing")
    report.check("M10.9a Runtime graph closure matrix" in texts["report"], "unified report M10.9a missing")
    report.check("M10.9a B300 runtime graph fixture-readiness rerun" in texts["report"], "B300 fixture-readiness rerun missing")


def run_negative_tests(summary: dict[str, Any]) -> Report:
    report = Report()
    mutations = [
        ("drop a gate", lambda data: data["gates"].pop()),
        ("gate order drift", lambda data: data["gates"].__setitem__(0, copy.deepcopy(data["gates"][1]))),
        ("claim policy overclaims parity", lambda data: data["claim_policy"].update({"runtime_graph_parity": "complete"})),
        ("backend replacement claimed", lambda data: data["claim_policy"].update({"backend_replacement": "complete"})),
        ("model byte drift", lambda data: data["b300_fixture_inventory"].update({"model_bytes": 92})),
        ("official vec hash drift", lambda data: data["b300_fixture_inventory"]["official_vec"].update({"sha256": "bad"})),
        ("missing rerun command", lambda data: data["gates"][2].update({"rerun_command": ""})),
        ("wrong next stage", lambda data: data.update({"next_stage": "M10.9c"})),
    ]
    for name, mutate in mutations:
        candidate = copy.deepcopy(summary)
        mutate(candidate)
        result = validate(candidate)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        summary = build_summary() if args.write_summary else load_json(args.summary)
    except Exception as exc:
        print(f"Runtime graph closure matrix: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(summary)
    if not report.ok:
        print("Runtime graph closure matrix: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    if args.write_summary:
        write_json(args.write_summary, summary)
    print(f"Runtime graph closure matrix: PASS, {report.checks} checks")
    if args.negative_test:
        negative = run_negative_tests(summary)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Runtime graph closure matrix negative tests: PASS, 8 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
