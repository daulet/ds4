#!/usr/bin/env python3
"""Compare Rust graph-session payload read/write probes against current C."""

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
C_SCHEMA = "ds4.graph_session_payload_rw_oracle.v1"
RUST_SCHEMA = "ds4.rust_graph_session_payload_rw.v1"
EXPECTED_CASES = [
    "valid_short_graph_payload",
    "valid_raw_wrap_graph_payload",
    "trailing_payload_bytes",
    "truncated_tensor_body",
    "n_comp_over_cap",
    "n_index_comp_over_cap",
    "raw_live_rows_not_expected",
    "ctx_too_large",
    "layer_count_mismatch",
    "prefill_cap_mismatch",
    "comp_cap_too_large",
]
EXPECTED_CODES = {
    "valid_short_graph_payload": "ok",
    "valid_raw_wrap_graph_payload": "ok",
    "trailing_payload_bytes": "trailing-payload-bytes",
    "truncated_tensor_body": "truncated-payload",
    "n_comp_over_cap": "invalid-compressed-row-count",
    "n_index_comp_over_cap": "invalid-indexer-row-count",
    "raw_live_rows_not_expected": "raw-ring-mismatch",
    "ctx_too_large": "context-fit",
    "layer_count_mismatch": "layout-mismatch",
    "prefill_cap_mismatch": "chunk-layout-mismatch",
    "comp_cap_too_large": "compressed-cap-too-large",
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def run_c_dump() -> dict[str, Any]:
    return run_json([str(ROOT / "ds4-session-payload-dump"), "--graph-probe"])


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
            "--graph-probe",
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


def validate_dump(report: Report, obj: dict[str, Any], schema: str, label: str) -> None:
    report.check(obj.get("schema") == schema, f"{label}.schema drift")
    report.check(
        obj.get("scope") == "graph-session-payload-read-write",
        f"{label}.scope drift",
    )
    runtime = require_dict(report, obj.get("runtime"), f"{label}.runtime")
    expected_runtime = {
        "ctx_size": 32_768,
        "prefill_cap": 2048,
        "raw_cap": 2304,
        "raw_window": 128,
        "comp_cap": 8194,
    }
    for key, expected in expected_runtime.items():
        report.check(runtime.get(key) == expected, f"{label}.runtime.{key} drift")

    cases = cases_by_name(report, obj, label)
    for name in EXPECTED_CASES:
        case = cases.get(name, {})
        expected_code = EXPECTED_CODES[name]
        report.check(case.get("code") == expected_code, f"{label}.{name}.code drift")
        report.check(case.get("ok") is (expected_code == "ok"), f"{label}.{name}.ok drift")
        report.check(isinstance(case.get("payload_bytes"), int), f"{label}.{name}.payload_bytes missing")
        report.check(isinstance(case.get("fnv1a64"), str) and len(case.get("fnv1a64", "")) == 16, f"{label}.{name}.fnv drift")
        if expected_code == "ok":
            validate_parsed(report, require_dict(report, case.get("parsed"), f"{label}.{name}.parsed"), f"{label}.{name}")
        else:
            report.check("parsed" not in case, f"{label}.{name}.parsed should be absent on rejection")


def validate_parsed(report: Report, parsed: dict[str, Any], path: str) -> None:
    token_count = parsed.get("token_count")
    raw_first_pos = parsed.get("raw_first_pos")
    raw_last_pos = parsed.get("raw_last_pos")
    raw_first_phys = parsed.get("raw_first_phys")
    raw_last_phys = parsed.get("raw_last_phys")
    if isinstance(token_count, int):
        expected_live = min(token_count, 128)
        report.check(raw_first_pos == token_count - expected_live, f"{path}.raw_first_pos drift")
        report.check(raw_last_pos == token_count - 1, f"{path}.raw_last_pos drift")
        if isinstance(raw_first_pos, int):
            report.check(raw_first_phys == raw_first_pos % 2304, f"{path}.raw_first_phys drift")
        if isinstance(raw_last_pos, int):
            report.check(raw_last_phys == raw_last_pos % 2304, f"{path}.raw_last_phys drift")
        report.check(parsed.get("ratio4_rows") == token_count // 4, f"{path}.ratio4_rows drift")
        report.check(parsed.get("ratio128_rows") == token_count // 128, f"{path}.ratio128_rows drift")
        report.check(parsed.get("layer2_n_index_comp") == token_count // 4, f"{path}.layer2 index drift")
    sections = require_dict(report, parsed.get("section_bytes"), f"{path}.section_bytes")
    section_sum = sum(int(value) for value in sections.values() if isinstance(value, int))
    report.check(section_sum == parsed.get("payload_bytes"), f"{path}.section total drift")


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
    report.check("M10.7b: Rust Graph Session Payload Reader And Writer" in text["roadmap"], "roadmap M10.7b missing")
    report.check("M10.7b: Rust Graph Session Payload Reader And Writer" in text["todo"], "TODO M10.7b missing")
    report.check("compare_graph_session_payload_rw.py" in text["readme"], "README graph payload RW command missing")
    report.check("M10.7b Rust graph-session payload reader/writer comparator" in text["report"], "unified report M10.7b entry missing")
    report.check("--graph-probe" in text["c_helper"], "C helper graph-probe flag missing")
    report.check("ds4_dump_graph_session_payload_probe_json" in text["c_core"], "C graph payload probe missing")
    report.check("read_graph_payload" in text["rust"], "Rust graph payload reader missing")
    report.check("--graph-probe" in text["rust_bin"], "Rust graph-probe flag missing")


def validate_pair(expected: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    validate_dump(report, expected, C_SCHEMA, "c")
    validate_dump(report, got, RUST_SCHEMA, "rust")
    compare_value(report, comparable(expected), comparable(got), "graph_payload_rw")
    static_checks(report)
    return report


def run_negative_tests(expected: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("runtime raw cap drift", ["runtime", "raw_cap"], 128),
        ("valid digest drift", ["cases", 0, "fnv1a64"], "0000000000000000"),
        ("wrap physical row drift", ["cases", 1, "parsed", "raw_last_phys"], 1),
        ("trailing code drift", ["cases", 2, "code"], "ok"),
        ("payload byte drift", ["cases", 1, "payload_bytes"], got["cases"][1]["payload_bytes"] + 4),
        ("section byte drift", ["cases", 0, "parsed", "section_bytes", "raw_rows"], 1),
        ("index row drift", ["cases", 1, "parsed", "layer2_n_index_comp"], 0),
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
        print(f"graph session payload reader/writer comparator: FAIL: {exc}")
        return 1

    report = validate_pair(expected, got)
    print_report("Graph session payload reader/writer comparator", report)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("Graph session payload reader/writer negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
