#!/usr/bin/env python3
"""Compare Rust graph-session payload layout planning against current C."""

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
C_SCHEMA = "ds4.graph_session_payload_plan_oracle.v1"
RUST_SCHEMA = "ds4.rust_graph_session_payload_plan.v1"
EXPECTED_CASES = [
    "short_checkpoint_tokens3",
    "continued_frontier_tokens924",
    "prefill_cap_cross_tokens2052",
    "raw_ring_wrap_tokens2305",
    "near_context_tokens32767",
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


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def run_json(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def run_c_dump() -> dict[str, Any]:
    return run_json([str(ROOT / "ds4-session-payload-dump"), "--graph-plan"])


def run_rust_dump() -> dict[str, Any]:
    return run_json(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-session-payload-dump-rs",
            "--",
            "--graph-plan",
        ]
    )


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


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
        report.check(len(expected) == len(got_list), f"{path}: length drift")
        for idx, (expected_item, got_item) in enumerate(zip(expected, got_list)):
            compare_value(report, expected_item, got_item, f"{path}[{idx}]")
        return
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def comparable(obj: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in obj.items() if key not in {"schema", "source"}}


def cases_by_name(report: Report, obj: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    cases = obj.get("cases")
    report.check(isinstance(cases, list), f"{label}.cases must be a list")
    out: dict[str, dict[str, Any]] = {}
    for case in cases if isinstance(cases, list) else []:
        report.check(isinstance(case, dict), f"{label}.cases entry must be object")
        if not isinstance(case, dict):
            continue
        name = case.get("name")
        report.check(isinstance(name, str), f"{label}.case name must be string")
        if isinstance(name, str):
            out[name] = case
    report.check(list(out) == EXPECTED_CASES, f"{label}.case order drift")
    return out


def validate_shape(report: Report, obj: dict[str, Any], schema: str, label: str) -> None:
    report.check(obj.get("schema") == schema, f"{label} schema drift")
    report.check(
        obj.get("scope") == "graph-session-payload-layout",
        f"{label} scope drift",
    )
    constants = require_dict(report, obj.get("constants"), f"{label}.constants")
    expected_constants = {
        "magic_u32": 878072644,
        "version": 1,
        "u32_fields": 13,
        "header_bytes": 52,
        "io_chunk_bytes": 8 * 1024 * 1024,
        "n_layer": 43,
        "n_head_dim": 512,
        "n_indexer_head_dim": 128,
        "n_vocab": 129_280,
        "n_swa": 128,
    }
    for key, expected in expected_constants.items():
        report.check(constants.get(key) == expected, f"{label}.constants.{key} drift")
    cases = cases_by_name(report, obj, label)
    for name in EXPECTED_CASES:
        case = cases.get(name, {})
        validate_case(report, case, f"{label}.{name}")


def validate_case(report: Report, case: dict[str, Any], path: str) -> None:
    token_count = case.get("token_count")
    raw_cap = case.get("raw_cap")
    raw_live = case.get("raw_live_rows")
    report.check(case.get("ctx_size") == 32_768, f"{path}.ctx_size drift")
    report.check(case.get("prefill_cap") == 2048, f"{path}.prefill_cap drift")
    report.check(raw_cap == 2304, f"{path}.raw_cap drift")
    report.check(case.get("raw_window") == 128, f"{path}.raw_window drift")
    report.check(case.get("comp_cap") == 8194, f"{path}.comp_cap drift")
    if isinstance(token_count, int) and isinstance(raw_live, int):
        expected_live = min(token_count, 128)
        report.check(raw_live == expected_live, f"{path}.raw_live_rows drift")
        report.check(case.get("raw_first_pos") == token_count - expected_live, f"{path}.raw_first_pos drift")
        report.check(case.get("raw_last_pos") == token_count - 1, f"{path}.raw_last_pos drift")
    if isinstance(token_count, int) and isinstance(raw_cap, int):
        report.check(case.get("raw_first_phys") == case.get("raw_first_pos") % raw_cap, f"{path}.raw_first_phys drift")
        report.check(case.get("raw_last_phys") == case.get("raw_last_pos") % raw_cap, f"{path}.raw_last_phys drift")
        report.check(case.get("ratio4_rows") == token_count // 4, f"{path}.ratio4_rows drift")
        report.check(case.get("ratio128_rows") == token_count // 128, f"{path}.ratio128_rows drift")

    sections = require_dict(report, case.get("section_bytes"), f"{path}.section_bytes")
    section_sum = sum(int(value) for value in sections.values() if isinstance(value, int))
    report.check(section_sum == case.get("payload_bytes"), f"{path}.payload byte total drift")
    samples = case.get("layer_samples")
    report.check(isinstance(samples, list) and len(samples) == 4, f"{path}.layer sample coverage drift")
    by_layer = {sample.get("layer"): sample for sample in samples if isinstance(sample, dict)}
    for layer, ratio in ((0, 0), (2, 4), (3, 128), (42, 4)):
        sample = require_dict(report, by_layer.get(layer), f"{path}.layer{layer}")
        report.check(sample.get("ratio") == ratio, f"{path}.layer{layer}.ratio drift")
        report.check(sample.get("raw_first_phys") == case.get("raw_first_phys"), f"{path}.layer{layer}.raw_first drift")
        report.check(sample.get("raw_last_phys") == case.get("raw_last_phys"), f"{path}.layer{layer}.raw_last drift")
        expected_n_comp = 0 if ratio == 0 or not isinstance(token_count, int) else token_count // ratio
        expected_n_index = expected_n_comp if ratio == 4 else 0
        report.check(sample.get("n_comp") == expected_n_comp, f"{path}.layer{layer}.n_comp drift")
        report.check(sample.get("n_index_comp") == expected_n_index, f"{path}.layer{layer}.n_index drift")

    if case.get("name") == "raw_ring_wrap_tokens2305":
        report.check(case.get("raw_first_phys") == 2177, "raw wrap first physical row drift")
        report.check(case.get("raw_last_phys") == 0, "raw wrap last physical row drift")
    if case.get("name") == "near_context_tokens32767":
        report.check(case.get("payload_bytes", 0) > 100_000_000, "near-context payload should stay large")


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "readme": ROOT / "ds4-parity/README.md",
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "c_helper": ROOT / "ds4_session_payload_dump.c",
        "c_core": ROOT / "ds4.c",
        "rust": ROOT / "rust/ds4-gguf/src/session_payload.rs",
        "rust_bin": ROOT / "rust/ds4-gguf/src/bin/ds4-session-payload-dump-rs.rs",
    }
    text = {name: path.read_text() for name, path in files.items()}
    report.check("M10.7a: Rust Graph Session Payload Layout Plan" in text["roadmap"], "roadmap M10.7a split missing")
    report.check("raw_ring_wrap_tokens2305" in text["todo"], "TODO raw-ring fixture missing")
    report.check("compare_graph_session_payload_plan.py" in text["readme"], "README command missing")
    report.check("M10.7a Rust graph-session payload layout comparator" in text["report"], "unified report entry missing")
    report.check("--graph-plan" in text["c_helper"], "C helper graph-plan flag missing")
    report.check("ds4_dump_graph_session_payload_plan_json" in text["c_core"], "C graph payload plan oracle missing")
    report.check("GRAPH_PAYLOAD_FIXTURES" in text["rust"], "Rust graph payload fixtures missing")
    report.check("--graph-plan" in text["rust_bin"], "Rust graph-plan flag missing")


def validate_pair(expected: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    validate_shape(report, expected, C_SCHEMA, "c")
    validate_shape(report, got, RUST_SCHEMA, "rust")
    compare_value(report, comparable(expected), comparable(got), "payload_plan")
    static_checks(report)
    return report


def run_negative_tests(expected: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("body order drift", ["body_order", 0], "checkpoint_tokens"),
        ("header byte drift", ["constants", "header_bytes"], 56),
        ("raw ring mapping drift", ["cases", 3, "raw_last_phys"], 1),
        ("payload byte drift", ["cases", 4, "payload_bytes"], got["cases"][4]["payload_bytes"] + 4),
        ("ratio4 row drift", ["cases", 2, "ratio4_rows"], 1),
        ("section byte drift", ["cases", 1, "section_bytes", "raw_rows"], 1),
        ("layer sample drift", ["cases", 2, "layer_samples", 1, "n_comp"], 1),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(got)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = validate_pair(expected, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        expected = load_json(args.oracle) if args.oracle else run_c_dump()
        got = load_json(args.candidate) if args.candidate else run_rust_dump()
    except Exception as exc:
        print(f"graph session payload plan comparator: FAIL: {exc}")
        return 1

    report = validate_pair(expected, got)
    print_report("Graph session payload C/Rust plan comparator", report)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("Graph session payload negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
