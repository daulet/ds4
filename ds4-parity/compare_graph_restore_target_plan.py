#!/usr/bin/env python3
"""Compare Rust graph restore target mapping against the C graph restore path."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import compare_graph_payload_raw_import as raw_import


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.8" / "current-c.json"
RUST_SCHEMA = "ds4.rust_graph_restore_target_plan.v1"
EXPECTED_CASES = [
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
]
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected object")
    return data


def run_rust_dump() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-session-payload-dump-rs",
            "--",
            "--restore-target-plan",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise TypeError("Rust restore target dump did not return an object")
    return data


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def cases_by_id(report: Report, cases: Any, label: str) -> dict[str, dict[str, Any]]:
    report.check(isinstance(cases, list), f"{label}.cases must be a list")
    if isinstance(cases, list):
        report.check(len(cases) == len(EXPECTED_CASES), f"{label}.case count drift")
    out: dict[str, dict[str, Any]] = {}
    for case in cases if isinstance(cases, list) else []:
        report.check(isinstance(case, dict), f"{label}.case must be object")
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        report.check(isinstance(case_id, str), f"{label}.case id must be string")
        if isinstance(case_id, str):
            out[case_id] = case
    report.check(list(out) == EXPECTED_CASES, f"{label}.case order drift")
    return out


def compare_value(report: Report, expected: Any, got: Any, path: str) -> None:
    if isinstance(expected, dict):
        got_dict = require_dict(report, got, path)
        report.check(list(expected) == list(got_dict), f"{path}: key order or coverage drift")
        for key, expected_value in expected.items():
            if key in got_dict:
                compare_value(report, expected_value, got_dict[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        report.check(isinstance(got, list), f"{path}: expected array")
        got_list = got if isinstance(got, list) else []
        report.check(len(expected) == len(got_list), f"{path}: array length drift")
        for idx, expected_value in enumerate(expected):
            if idx < len(got_list):
                compare_value(report, expected_value, got_list[idx], f"{path}[{idx}]")
        return
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def payload_bytes(case: dict[str, Any]) -> int:
    if case.get("kind") == "disk-payload":
        return int(case["payload_bytes"])
    if case.get("kind") == "memory-snapshot":
        return int(case["snapshot_bytes"])
    raise ValueError(f"{case.get('id')}: unknown restore kind {case.get('kind')!r}")


def layer_counts(token_count: int) -> tuple[list[int], list[int]]:
    n_comp: list[int] = []
    n_index_comp: list[int] = []
    for layer in range(raw_import.N_LAYER):
        ratio = raw_import.compress_ratio(layer)
        count = raw_import.compressed_rows(token_count, ratio)
        n_comp.append(count)
        n_index_comp.append(count if ratio == 4 else 0)
    return n_comp, n_index_comp


def expected_layer(layer: int, raw_rows: int, n_comp: list[int], n_index_comp: list[int]) -> dict[str, Any]:
    ratio = raw_import.compress_ratio(layer)
    raw_bytes = raw_rows * raw_import.N_HEAD_DIM * 4
    if ratio == 0:
        attn = {
            "n_comp": 0,
            "comp_cache_bytes": 0,
            "state_kv_bytes": 0,
            "state_score_bytes": 0,
            "targets": [],
        }
    else:
        state_bytes = raw_import.layer_attn_state_bytes(ratio)
        attn = {
            "n_comp": n_comp[layer],
            "comp_cache_bytes": n_comp[layer] * raw_import.N_HEAD_DIM * 4,
            "state_kv_bytes": state_bytes,
            "state_score_bytes": state_bytes,
            "targets": [
                "g->layer_attn_comp_cache[layer]",
                "g->layer_attn_state_kv[layer]",
                "g->layer_attn_state_score[layer]",
            ],
        }
    if ratio == 4:
        state_bytes = raw_import.layer_index_state_bytes(ratio)
        index = {
            "n_index_comp": n_index_comp[layer],
            "comp_cache_bytes": n_index_comp[layer] * raw_import.N_INDEXER_HEAD_DIM * 4,
            "state_kv_bytes": state_bytes,
            "state_score_bytes": state_bytes,
            "targets": [
                "g->layer_index_comp_cache[layer]",
                "g->layer_index_state_kv[layer]",
                "g->layer_index_state_score[layer]",
            ],
        }
    else:
        index = {
            "n_index_comp": 0,
            "comp_cache_bytes": 0,
            "state_kv_bytes": 0,
            "state_score_bytes": 0,
            "targets": [],
        }
    return {
        "layer": layer,
        "ratio": ratio,
        "raw": {
            "target": "g->layer_raw_cache[layer]",
            "bytes": raw_bytes,
        },
        "attn": attn,
        "index": index,
    }


def expected_case(case: dict[str, Any]) -> dict[str, Any]:
    header = raw_import.decode_header(str(case["header_prefix_hex"]))
    token_count = header["prompt_tokens"]
    raw_rows = header["raw_live_rows"]
    raw_first_pos = token_count - raw_rows
    raw_last_pos = token_count - 1
    n_comp, n_index_comp = layer_counts(token_count)
    physical_rows = [(raw_first_pos + idx) % header["raw_cap"] for idx in range(raw_rows)]
    ratio4_layers = sum(1 for layer in range(raw_import.N_LAYER) if raw_import.compress_ratio(layer) == 4)
    ratio128_layers = sum(1 for layer in range(raw_import.N_LAYER) if raw_import.compress_ratio(layer) == 128)
    return {
        "id": case["id"],
        "kind": case["kind"],
        "prompt_case": case["prompt_case"],
        "ctx": header["ctx"],
        "prompt_tokens": token_count,
        "payload_bytes": payload_bytes(case),
        "checkpoint": {
            "target": "s->checkpoint",
            "source": "payload token u32 stream",
            "tokens": token_count,
            "bytes": token_count * 4,
            "commit": "replace-after-success",
        },
        "logits": {
            "target": "s->logits",
            "source": "payload logits f32 stream",
            "bytes": raw_import.N_VOCAB * 4,
        },
        "count_tables": {
            "n_comp_source": "payload n_comp table",
            "n_index_comp_source": "payload n_index_comp table",
            "bytes_each": raw_import.N_LAYER * 4,
            "post_restore_targets": ["g->layer_n_comp", "g->layer_n_index_comp"],
        },
        "raw_ring": {
            "target": "g->layer_raw_cache[layer]",
            "source_order": "logical-position-order",
            "row_bytes": raw_import.N_HEAD_DIM * 4,
            "rows_per_layer": raw_rows,
            "first_pos": raw_first_pos,
            "last_pos": raw_last_pos,
            "physical_rows": physical_rows,
        },
        "layer_summary": {
            "layer_count": raw_import.N_LAYER,
            "raw_layer_spans": raw_import.N_LAYER,
            "attn_comp_layers": ratio4_layers + ratio128_layers,
            "ratio4_layers": ratio4_layers,
            "ratio128_layers": ratio128_layers,
            "index_layers": ratio4_layers,
        },
        "layers": [
            expected_layer(layer, raw_rows, n_comp, n_index_comp)
            for layer in range(raw_import.N_LAYER)
        ],
        "post_restore_state": {
            "checkpoint_valid": True,
            "mtp_draft_valid": False,
            "mtp_n_raw": 0,
            "layer_n_comp": n_comp,
            "layer_n_index_comp": n_index_comp,
        },
    }


def validate_oracle(report: Report, oracle: dict[str, Any]) -> list[dict[str, Any]]:
    report.check(oracle.get("schema") == "ds4.restore_oracle.v1", "oracle schema drift")
    report.check(oracle.get("source") == "current-c-b300-restore", "oracle source drift")
    report.check(oracle.get("model_path") == "/workspace/ds4/ds4flash.gguf", "oracle model path drift")
    report.check(oracle.get("model_sha256") == EXPECTED_MODEL_SHA256, "oracle model sha drift")
    cases = cases_by_id(report, oracle.get("cases"), "oracle")
    expected = []
    for case_id in EXPECTED_CASES:
        case = cases.get(case_id, {})
        path = f"oracle.{case_id}"
        report.check(case.get("raw_payload_committed") is False, f"{path}.raw payload must stay hash-only")
        report.check(case.get("reference", {}).get("selected_token") == case.get("restored", {}).get("selected_token"), f"{path}.selected token mismatch")
        try:
            expected.append(expected_case(case))
        except Exception as exc:
            report.check(False, f"{path}.expected failed: {exc}")
    return expected


def validate_candidate(report: Report, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    report.check(candidate.get("schema") == RUST_SCHEMA, "candidate schema drift")
    report.check(
        candidate.get("source") == "rust-graph-restore-target-plan-no-tensor-writes",
        "candidate source drift",
    )
    report.check(
        candidate.get("oracle") == "ds4-parity/baselines/kv/m7.8/current-c.json",
        "candidate oracle path drift",
    )
    report.check(candidate.get("model_path") == "/workspace/ds4/ds4flash.gguf", "candidate model path drift")
    report.check(candidate.get("model_sha256") == EXPECTED_MODEL_SHA256, "candidate model sha drift")
    report.check(candidate.get("restore_order_source") == "ds4_session_load_payload graph path", "candidate restore source drift")
    report.check("hash-only" in str(candidate.get("raw_body_policy")), "candidate raw-body policy drift")
    return list(cases_by_id(report, candidate.get("cases"), "candidate").values())


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "readme": ROOT / "ds4-parity" / "README.md",
        "report": ROOT / "ds4-parity" / "run_parity_report.py",
        "rust_bin": ROOT / "rust" / "ds4-gguf" / "src" / "bin" / "ds4-session-payload-dump-rs.rs",
        "c": ROOT / "ds4.c",
    }
    text = {name: path.read_text() for name, path in files.items()}
    status_flat = " ".join(text["status"].split())
    report.check("M10.7c3b: Rust Graph Restore Target Mapping Contract" in text["roadmap"], "roadmap M10.7c3b missing")
    report.check("M10.7c3b: Rust Graph Restore Target Mapping Contract" in text["todo"], "TODO M10.7c3b missing")
    report.check("M10.7c3b Rust Graph Restore Target Mapping Contract" in status_flat, "status M10.7c3b missing")
    report.check("compare_graph_restore_target_plan.py" in text["readme"], "README restore target command missing")
    report.check("M10.7c3b Rust graph restore target comparator" in text["report"], "unified report M10.7c3b missing")
    report.check("--restore-target-plan" in text["rust_bin"], "Rust restore target plan flag missing")
    for snippet in (
        "g->layer_raw_cache[il]",
        "g->layer_attn_comp_cache[il]",
        "g->layer_index_comp_cache[il]",
        "g->layer_n_comp[il] = n_comp[il]",
        "g->mtp_n_raw = 0",
    ):
        report.check(snippet in text["c"], f"C restore source missing {snippet}")


def validate_pair(oracle: dict[str, Any], candidate: dict[str, Any]) -> Report:
    report = Report()
    expected_cases = validate_oracle(report, oracle)
    got_cases = validate_candidate(report, candidate)
    for expected, got in zip(expected_cases, got_cases, strict=False):
        compare_value(report, expected, got, f"case.{expected.get('id')}")
    static_checks(report)
    return report


def run_negative_tests(oracle: dict[str, Any], candidate: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("case order drift", ["cases", 0, "id"], "snapshot_seed"),
        ("raw physical row drift", ["cases", 0, "raw_ring", "physical_rows", 0], -1),
        ("attn target drift", ["cases", 0, "layers", 2, "attn", "targets", 0], "wrong"),
        ("index bytes drift", ["cases", 0, "layers", 2, "index", "comp_cache_bytes"], 1),
        ("post counter drift", ["cases", 2, "post_restore_state", "layer_n_comp", 2], -1),
        ("mtp raw drift", ["cases", 0, "post_restore_state", "mtp_n_raw"], 1),
        ("source drift", ["restore_order_source"], "other"),
        ("policy drift", ["raw_body_policy"], "raw bodies required"),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(candidate)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = validate_pair(oracle, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        oracle = load_json(args.oracle)
        candidate = load_json(args.candidate) if args.candidate else run_rust_dump()
    except Exception as exc:
        print(f"restore target comparator: FAIL: {exc}")
        return 1

    report = validate_pair(oracle, candidate)
    print_report("Restore target comparator", report)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(oracle, candidate)
        print_report("Restore target negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
