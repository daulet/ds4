#!/usr/bin/env python3
"""Compare the Rust MTP frontier mutation plan against current-C anchors."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
M108A_CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json"
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_frontier_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-frontier-plan.rs"
EVIDENCE_PATHS = [
    ROOT / "ds4-parity/baselines/kv/m10.7c3c/rust-b300-restore-readback.json",
    ROOT / "ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json",
]

FRONTIER_COUNTERS = ["n_comp", "n_index_comp", "mtp_n_raw"]

EXPECTED_CASES = {
    "b300_missing_mtp_support_model": {
        "source_function": "none",
        "ratio_family": "none",
        "saved_counters": [],
        "counter_updates": [],
        "tensor_copies": [],
        "mtp_n_raw_action": "none",
        "invisible_rows_policy": "blocked_missing_mtp_model",
        "failure_action": "blocked_missing_mtp_model",
        "live_status": "blocked_missing_mtp_model",
    },
    "snapshot_dense_layer_counters_only": {
        "source_function": "spec_frontier_snapshot",
        "ratio_family": "ratio0",
        "saved_counters": FRONTIER_COUNTERS,
        "counter_updates": [
            "f.n_comp = g.layer_n_comp",
            "f.n_index_comp = g.layer_n_index_comp",
            "f.mtp_n_raw = g.mtp_n_raw",
        ],
        "tensor_copies": [],
        "mtp_n_raw_action": "save",
        "invisible_rows_policy": "none",
        "failure_action": "spec_frontier_free_on_failure",
        "live_status": "blocked_missing_mtp_model",
    },
    "snapshot_compressed_attn_frontier": {
        "source_function": "spec_frontier_snapshot",
        "ratio_family": "ratio4_or_ratio128",
        "saved_counters": FRONTIER_COUNTERS,
        "counter_updates": [
            "f.n_comp = g.layer_n_comp",
            "f.n_index_comp = g.layer_n_index_comp",
            "f.mtp_n_raw = g.mtp_n_raw",
        ],
        "tensor_copies": [
            "spec_attn_state_kv <- layer_attn_state_kv",
            "spec_attn_state_score <- layer_attn_state_score",
        ],
        "mtp_n_raw_action": "save",
        "invisible_rows_policy": "none",
        "failure_action": "spec_frontier_free_on_failure",
        "live_status": "blocked_missing_mtp_model",
    },
    "snapshot_ratio4_index_frontier": {
        "source_function": "spec_frontier_snapshot",
        "ratio_family": "ratio4",
        "saved_counters": FRONTIER_COUNTERS,
        "counter_updates": [
            "f.n_comp = g.layer_n_comp",
            "f.n_index_comp = g.layer_n_index_comp",
            "f.mtp_n_raw = g.mtp_n_raw",
        ],
        "tensor_copies": [
            "spec_index_state_kv <- layer_index_state_kv",
            "spec_index_state_score <- layer_index_state_score",
        ],
        "mtp_n_raw_action": "save",
        "invisible_rows_policy": "none",
        "failure_action": "spec_frontier_free_on_failure",
        "live_status": "blocked_missing_mtp_model",
    },
    "restore_compressed_attn_frontier": {
        "source_function": "spec_frontier_restore",
        "ratio_family": "ratio4_or_ratio128",
        "saved_counters": FRONTIER_COUNTERS,
        "counter_updates": [
            "g.layer_n_comp = f.n_comp",
            "g.layer_n_index_comp = f.n_index_comp",
            "g.mtp_n_raw = f.mtp_n_raw",
        ],
        "tensor_copies": [
            "layer_attn_state_kv <- spec_attn_state_kv",
            "layer_attn_state_score <- spec_attn_state_score",
        ],
        "mtp_n_raw_action": "restore",
        "invisible_rows_policy": "append_only_rows_may_remain_beyond_restored_counters",
        "failure_action": "return_false_after_synchronize",
        "live_status": "blocked_missing_mtp_model",
    },
    "restore_ratio4_index_frontier": {
        "source_function": "spec_frontier_restore",
        "ratio_family": "ratio4",
        "saved_counters": FRONTIER_COUNTERS,
        "counter_updates": [
            "g.layer_n_comp = f.n_comp",
            "g.layer_n_index_comp = f.n_index_comp",
            "g.mtp_n_raw = f.mtp_n_raw",
        ],
        "tensor_copies": [
            "layer_index_state_kv <- spec_index_state_kv",
            "layer_index_state_score <- spec_index_state_score",
        ],
        "mtp_n_raw_action": "restore",
        "invisible_rows_policy": "append_only_rows_may_remain_beyond_restored_counters",
        "failure_action": "return_false_after_synchronize",
        "live_status": "blocked_missing_mtp_model",
    },
    "prefix1_commit_compressed_attn_frontier": {
        "source_function": "spec_frontier_commit_prefix1",
        "ratio_family": "ratio4_or_ratio128",
        "saved_counters": ["spec_prefix1_n_comp"],
        "counter_updates": ["g.layer_n_comp = g.spec_prefix1_n_comp"],
        "tensor_copies": [
            "layer_attn_state_kv <- spec_prefix1_attn_state_kv",
            "layer_attn_state_score <- spec_prefix1_attn_state_score",
        ],
        "mtp_n_raw_action": "unchanged",
        "invisible_rows_policy": "second speculative row may remain invisible",
        "failure_action": "return_false_after_synchronize",
        "live_status": "blocked_missing_mtp_model",
    },
    "prefix1_commit_ratio4_index_frontier": {
        "source_function": "spec_frontier_commit_prefix1",
        "ratio_family": "ratio4",
        "saved_counters": ["spec_prefix1_n_index_comp"],
        "counter_updates": ["g.layer_n_index_comp = g.spec_prefix1_n_index_comp"],
        "tensor_copies": [
            "layer_index_state_kv <- spec_prefix1_index_state_kv",
            "layer_index_state_score <- spec_prefix1_index_state_score",
        ],
        "mtp_n_raw_action": "unchanged",
        "invisible_rows_policy": "second speculative row may remain invisible",
        "failure_action": "return_false_after_synchronize",
        "live_status": "blocked_missing_mtp_model",
    },
}


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


def run_rust_plan() -> dict[str, Any]:
    proc = subprocess.run(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-frontier-plan", "--quiet"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def validate(candidate: dict[str, Any], contract: dict[str, Any]) -> Report:
    report = Report()
    report.check(candidate.get("schema") == "ds4.rust_mtp_frontier_plan.v1", "schema drift")
    report.check(
        candidate.get("source") == "rust-model-free-mtp-frontier-orchestration",
        "source drift",
    )
    report.check(
        candidate.get("oracle") == "spec_frontier_snapshot_restore_prefix1",
        "oracle drift",
    )
    cases = named_cases(report, candidate.get("cases"))
    report.check(list(cases) == list(EXPECTED_CASES), "case order drift")
    for case_id, expected in EXPECTED_CASES.items():
        case = cases.get(case_id)
        if case is None:
            report.check(False, f"missing case {case_id}")
            continue
        for key, expected_value in expected.items():
            report.check(
                case.get(key) == expected_value,
                f"{case_id}.{key}: expected {expected_value!r}, got {case.get(key)!r}",
            )
    check_contract_links(report, contract)
    static_checks(report)
    return report


def named_cases(report: Report, value: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    report.check(isinstance(value, list), "cases must be a list")
    if not isinstance(value, list):
        return result
    for index, item in enumerate(value):
        report.check(isinstance(item, dict), f"cases[{index}] must be an object")
        if not isinstance(item, dict):
            continue
        case_id = item.get("id")
        report.check(isinstance(case_id, str), f"cases[{index}].id missing")
        if isinstance(case_id, str):
            result[case_id] = item
    return result


def check_contract_links(report: Report, contract: dict[str, Any]) -> None:
    mtp = contract.get("support_artifacts", {}).get("mtp", {})
    report.check(mtp.get("available") is False, "M10.8a MTP support availability drift")
    anchors = {
        item.get("name"): item
        for item in contract.get("function_anchors", [])
        if isinstance(item, dict)
    }
    for name in [
        "spec_frontier_snapshot",
        "spec_frontier_restore",
        "spec_frontier_commit_prefix1",
    ]:
        anchor = anchors.get(name, {})
        snippets = anchor.get("required_snippets", [])
        report.check(anchor.get("name") == name, f"M10.8a missing anchor {name}")
        report.check(isinstance(snippets, list) and bool(snippets), f"M10.8a {name} snippets empty")


def static_checks(report: Report) -> None:
    c_source = (ROOT / "ds4.c").read_text()
    graph_plan = (ROOT / "rust/ds4-gpu/src/graph_plan.rs").read_text()
    rust_source = RUST_SOURCE.read_text()
    rust_bin = RUST_BIN.read_text()
    lib_source = (ROOT / "rust/ds4-gpu/src/lib.rs").read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()

    for snippet in [
        "uint32_t n_comp[DS4_N_LAYER];",
        "uint32_t n_index_comp[DS4_N_LAYER];",
        "uint32_t mtp_n_raw;",
        "static void spec_frontier_free(ds4_spec_frontier *f)",
        "memset(f, 0, sizeof(*f));",
        "static bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s)",
        "f->mtp_n_raw = g->mtp_n_raw;",
        "f->n_comp[il] = g->layer_n_comp[il];",
        "f->n_index_comp[il] = g->layer_n_index_comp[il];",
        "if (ratio == 0) continue;",
        "ds4_gpu_tensor_copy(g->spec_attn_state_kv[il], 0,",
        "ds4_gpu_tensor_copy(g->spec_index_state_kv[il], 0,",
        "spec_frontier_free(f);",
        "static bool spec_frontier_restore(ds4_spec_frontier *f, ds4_session *s)",
        "g->mtp_n_raw = f->mtp_n_raw;",
        "g->layer_n_comp[il] = f->n_comp[il];",
        "g->layer_n_index_comp[il] = f->n_index_comp[il];",
        "ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,",
        "ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,",
        "static bool spec_frontier_commit_prefix1(ds4_session *s)",
        "g->layer_n_comp[il] = g->spec_prefix1_n_comp[il];",
        "ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,",
        "g->layer_n_index_comp[il] = g->spec_prefix1_n_index_comp[il];",
        "ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,",
    ]:
        report.check(snippet in c_source, f"ds4.c missing frontier anchor {snippet!r}")
    for name in [
        "spec_frontier_snapshot",
        "spec_frontier_restore",
        "spec_frontier_commit_prefix1",
    ]:
        boundary_index = graph_plan.find(f'"{name}"')
        report.check(boundary_index >= 0, f"graph_plan missing {name} boundary")
        boundary_block = graph_plan[boundary_index : boundary_index + 180]
        report.check(name in boundary_block, f"graph_plan {name} boundary target drift")
    for path in EVIDENCE_PATHS:
        text = path.read_text()
        for snippet in ['"mtp_n_raw"', '"layer_n_comp"', '"layer_n_index_comp"']:
            report.check(snippet in text, f"{path.name} missing evidence {snippet}")
    for snippet in [
        "pub mod mtp_frontier_plan;",
        "pub const MTP_FRONTIER_MUTATION_CASES",
        "FRONTIER_COUNTERS",
        "prefix1_commit_ratio4_index_frontier",
    ]:
        report.check(snippet in rust_source + lib_source, f"Rust source missing {snippet!r}")
    for snippet in ["fn write_json_string", "MTP_FRONTIER_MUTATION_CASES"]:
        report.check(snippet in rust_bin, f"Rust bin missing {snippet!r}")
    for snippet in ["compare_mtp_frontier_plan.py", "M10.8f Rust MTP frontier mutation plan"]:
        report.check(snippet in run_report, f"unified report missing {snippet!r}")
    report.check("compare_mtp_frontier_plan.py --negative-test" in readme, "README missing M10.8f command")


def run_negative_tests(candidate: dict[str, Any], contract: dict[str, Any]) -> Report:
    report = Report()
    mutations = [
        ("schema", lambda data: data.update({"schema": "drift"})),
        ("missing case", lambda data: data["cases"].pop(1)),
        (
            "snapshot counter drift",
            lambda data: mutate_case(
                data,
                "snapshot_dense_layer_counters_only",
                "counter_updates",
                ["f.n_comp = g.layer_n_comp"],
            ),
        ),
        (
            "snapshot copy drift",
            lambda data: mutate_case(
                data,
                "snapshot_compressed_attn_frontier",
                "tensor_copies",
                ["spec_attn_state_kv <- layer_attn_state_kv"],
            ),
        ),
        (
            "raw restore drift",
            lambda data: mutate_case(
                data,
                "restore_compressed_attn_frontier",
                "mtp_n_raw_action",
                "save",
            ),
        ),
        (
            "prefix invisible policy drift",
            lambda data: mutate_case(
                data,
                "prefix1_commit_compressed_attn_frontier",
                "invisible_rows_policy",
                "none",
            ),
        ),
        (
            "ratio4 index drift",
            lambda data: mutate_case(
                data,
                "prefix1_commit_ratio4_index_frontier",
                "counter_updates",
                [],
            ),
        ),
        (
            "live blocker drift",
            lambda data: mutate_case(
                data,
                "b300_missing_mtp_support_model",
                "live_status",
                "available",
            ),
        ),
    ]
    for name, mutate in mutations:
        mutated = copy.deepcopy(candidate)
        mutate(mutated)
        result = validate(mutated, contract)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def mutate_case(data: dict[str, Any], case_id: str, key: str, value: Any) -> None:
    for case in data["cases"]:
        if case["id"] == case_id:
            case[key] = value
            return
    raise AssertionError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    candidate = run_rust_plan()
    contract = load_json(M108A_CONTRACT)
    report = validate(candidate, contract)
    if not report.ok:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "Rust MTP frontier plan comparator: "
        f"PASS, {len(EXPECTED_CASES)} cases, {report.checks} checks"
    )
    if args.negative_test:
        negative = run_negative_tests(candidate, contract)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Rust MTP frontier plan negative tests: PASS, 8 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
