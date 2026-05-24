#!/usr/bin/env python3
"""Compare Rust restore payload header planning against the M7.8 restore oracle."""

from __future__ import annotations

import argparse
import copy
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.8" / "current-c.json"
RUST_SCHEMA = "ds4.rust_restore_payload_header_plan.v1"
EXPECTED_CASES = [
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
]
HEADER_FIELDS = (
    "magic",
    "version",
    "ctx",
    "prefill_cap",
    "raw_cap",
    "raw_window",
    "comp_cap",
    "prompt_tokens",
    "n_layer",
    "n_head_dim",
    "n_indexer_head_dim",
    "n_vocab",
    "raw_live_rows",
)


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
            "--restore-header-plan",
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
        raise TypeError("Rust restore header dump did not return an object")
    return data


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def decode_header(hex_value: str) -> dict[str, int]:
    header = bytes.fromhex(hex_value)
    if len(header) != 52:
        raise ValueError(f"header prefix must be 52 bytes, got {len(header)}")
    values = struct.unpack("<13I", header)
    return dict(zip(HEADER_FIELDS, values, strict=True))


def payload_bytes(case: dict[str, Any]) -> int:
    if case.get("kind") == "disk-payload":
        return int(case["payload_bytes"])
    if case.get("kind") == "memory-snapshot":
        return int(case["snapshot_bytes"])
    raise ValueError(f"{case.get('id')}: unknown restore kind {case.get('kind')!r}")


def normalized_oracle_case(case: dict[str, Any]) -> dict[str, Any]:
    header = decode_header(str(case["header_prefix_hex"]))
    token_count = header["prompt_tokens"]
    return {
        "id": case["id"],
        "kind": case["kind"],
        "prompt_case": case["prompt_case"],
        "ctx": header["ctx"],
        "prompt_tokens": token_count,
        "raw_payload_committed": case["raw_payload_committed"],
        "header_prefix_hex": case["header_prefix_hex"],
        "payload_bytes": payload_bytes(case),
        "graph": {
            "prefill_cap": header["prefill_cap"],
            "raw_cap": header["raw_cap"],
            "raw_window": header["raw_window"],
            "comp_cap": header["comp_cap"],
            "raw_live_rows": header["raw_live_rows"],
            "ratio4_rows": token_count // 4,
            "ratio128_rows": token_count // 128,
        },
    }


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
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def validate_oracle(report: Report, oracle: dict[str, Any]) -> list[dict[str, Any]]:
    report.check(oracle.get("schema") == "ds4.restore_oracle.v1", "oracle schema drift")
    report.check(oracle.get("source") == "current-c-b300-restore", "oracle source drift")
    report.check(oracle.get("model_path") == "/workspace/ds4/ds4flash.gguf", "oracle model path drift")
    report.check(
        oracle.get("model_sha256") == "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668",
        "oracle model sha drift",
    )
    cases = cases_by_id(report, oracle.get("cases"), "oracle")
    normalized = []
    for case_id in EXPECTED_CASES:
        case = cases.get(case_id, {})
        path = f"oracle.{case_id}"
        report.check(case.get("raw_payload_committed") is False, f"{path}.raw payload must stay hash-only")
        if case.get("kind") == "disk-payload":
            report.check(isinstance(case.get("payload_sha256"), str), f"{path}.payload sha missing")
        if case.get("kind") == "memory-snapshot":
            report.check(isinstance(case.get("snapshot_sha256"), str), f"{path}.snapshot sha missing")
        report.check(case.get("reference", {}).get("selected_token") == case.get("restored", {}).get("selected_token"), f"{path}.selected token mismatch")
        try:
            normalized.append(normalized_oracle_case(case))
        except Exception as exc:
            report.check(False, f"{path}.normalize failed: {exc}")
    return normalized


def validate_candidate(report: Report, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    report.check(candidate.get("schema") == RUST_SCHEMA, "candidate schema drift")
    report.check(
        candidate.get("source") == "rust-restore-payload-header-plan-no-raw-bodies",
        "candidate source drift",
    )
    report.check(
        candidate.get("oracle") == "ds4-parity/baselines/kv/m7.8/current-c.json",
        "candidate oracle path drift",
    )
    report.check(candidate.get("model_path") == "/workspace/ds4/ds4flash.gguf", "candidate model path drift")
    report.check(
        candidate.get("model_sha256") == "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668",
        "candidate model sha drift",
    )
    report.check("hash-only" in str(candidate.get("raw_body_policy")), "candidate raw-body policy drift")
    return list(cases_by_id(report, candidate.get("cases"), "candidate").values())


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "readme": ROOT / "ds4-parity/README.md",
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "rust_bin": ROOT / "rust/ds4-gguf/src/bin/ds4-session-payload-dump-rs.rs",
        "rust": ROOT / "rust/ds4-gguf/src/session_payload.rs",
    }
    text = {name: path.read_text() for name, path in files.items()}
    report.check(
        "M10.7c1: Rust Restore Payload Header Contract" in text["roadmap"],
        "roadmap M10.7c1 missing",
    )
    report.check(
        "M10.7c1: Rust Restore Payload Header Contract" in text["todo"],
        "TODO M10.7c1 missing",
    )
    status_flat = " ".join(text["status"].split())
    report.check(
        "M10.7c1 Rust Restore Payload Header Contract" in status_flat,
        "status M10.7c1 evidence missing",
    )
    report.check(
        "compare_restore_payload_header_plan.py" in text["readme"],
        "README restore header command missing",
    )
    report.check(
        "M10.7c1 Rust restore payload header comparator" in text["report"],
        "unified report M10.7c1 missing",
    )
    report.check("--restore-header-plan" in text["rust_bin"], "Rust restore header plan flag missing")
    report.check("restore_header_contract_matches_m78_payload_sizes" in text["rust"], "Rust M7.8 header contract test missing")


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
        ("header byte drift", ["cases", 0, "header_prefix_hex"], "00" * 52),
        ("payload byte drift", ["cases", 2, "payload_bytes"], candidate["cases"][2]["payload_bytes"] + 4),
        ("raw committed drift", ["cases", 0, "raw_payload_committed"], True),
        ("ratio4 row drift", ["cases", 2, "graph", "ratio4_rows"], 1),
        ("model sha drift", ["model_sha256"], "0" * 64),
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
        print(f"restore payload header comparator: FAIL: {exc}")
        return 1

    report = validate_pair(oracle, candidate)
    print_report("Restore payload header comparator", report)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(oracle, candidate)
        print_report("Restore payload header negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
