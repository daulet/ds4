#!/usr/bin/env python3
"""Compare Rust raw graph snapshot import against the M7.8 snapshot oracle."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import compare_graph_payload_raw_import as raw_import


ROOT = raw_import.ROOT
ORACLE = raw_import.ORACLE
SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c3a" / "rust-b300-snapshot-raw-import.json"
SCHEMA = "ds4.rust_graph_snapshot_raw_import.v1"
EXPECTED_CASES = ["snapshot_seed", "snapshot_continuation"]
SNAPSHOT_RAW_DIR = "ds4-parity/baselines/kv/m7.8/raw"


def snapshot_oracle_cases(
    report: raw_import.Report,
    oracle: dict[str, Any],
) -> list[dict[str, Any]]:
    report.check(oracle.get("schema") == "ds4.restore_oracle.v1", "oracle schema drift")
    report.check(oracle.get("source") == "current-c-b300-restore", "oracle source drift")
    cases = oracle.get("cases")
    report.check(isinstance(cases, list), "oracle cases must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases if isinstance(cases, list) else []:
        if isinstance(case, dict) and case.get("kind") == "memory-snapshot":
            case_id = case.get("id")
            if isinstance(case_id, str):
                by_id[case_id] = case
    report.check(list(by_id) == EXPECTED_CASES, "oracle memory snapshot case order drift")

    out: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASES:
        if case_id not in by_id:
            continue
        case = dict(by_id[case_id])
        case["snapshot_file"] = f"{SNAPSHOT_RAW_DIR}/{case_id}.dsv4"
        out.append(case)
    return out


def rust_probe_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "payload_file": case["snapshot_file"],
    }


def expected_parsed(case: dict[str, Any]) -> dict[str, Any]:
    payload_case = dict(case)
    payload_case["payload_bytes"] = case["snapshot_bytes"]
    return raw_import.expected_parsed(payload_case)


def build_live_summary(oracle: dict[str, Any], workdir: Path) -> dict[str, Any]:
    report = raw_import.Report()
    oracle_cases = snapshot_oracle_cases(report, oracle)
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    rust = raw_import.run_rust_probe([rust_probe_case(case) for case in oracle_cases], workdir)
    rust_cases = raw_import.cases_by_id(raw_import.Report(), rust.get("cases"), "rust")
    cases = []
    for case in oracle_cases:
        snapshot_path = workdir / str(case["snapshot_file"])
        snapshot_sha256 = raw_import.sha256_file(snapshot_path)
        rust_case = rust_cases.get(str(case["id"]), {})
        cases.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "prompt_case": case["prompt_case"],
                "snapshot_file": case["snapshot_file"],
                "snapshot_bytes": int(case["snapshot_bytes"]),
                "snapshot_sha256": snapshot_sha256,
                "oracle_snapshot_sha256": case["snapshot_sha256"],
                "snapshot_sha256_matches_oracle": snapshot_sha256 == case["snapshot_sha256"],
                "rust": rust_case,
            }
        )
    return {
        "schema": SCHEMA,
        "source": "rust-b300-raw-snapshot-import",
        "oracle": "ds4-parity/baselines/kv/m7.8/current-c.json",
        "raw_body_policy": "hash-only; raw memory snapshot bodies remain on B300 and are not committed",
        "b300": {
            "kube_context": "hou2-prod1",
            "pod": "ds4-rust-port-b300",
            "workdir": "/workspace/ds4",
        },
        "rust_probe": {
            "schema": rust.get("schema"),
            "source": rust.get("source"),
            "runtime": rust.get("runtime"),
        },
        "cases": cases,
    }


def validate_summary(
    oracle: dict[str, Any],
    summary: dict[str, Any],
    *,
    raw_root: Path | None = None,
) -> raw_import.Report:
    report = raw_import.Report()
    oracle_cases = snapshot_oracle_cases(report, oracle)
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == "rust-b300-raw-snapshot-import", "summary source drift")
    report.check(
        summary.get("oracle") == "ds4-parity/baselines/kv/m7.8/current-c.json",
        "summary oracle path drift",
    )
    report.check("hash-only" in str(summary.get("raw_body_policy")), "raw-body policy drift")
    b300 = raw_import.require_dict(report, summary.get("b300"), "summary.b300")
    report.check(b300.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(b300.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(b300.get("workdir") == "/workspace/ds4", "B300 workdir drift")
    rust_probe = raw_import.require_dict(report, summary.get("rust_probe"), "summary.rust_probe")
    report.check(rust_probe.get("schema") == raw_import.RUST_SCHEMA, "Rust probe schema drift")
    report.check(rust_probe.get("source") == "rust-graph-payload-file-probe", "Rust probe source drift")
    cases = raw_import.cases_by_id(report, summary.get("cases"), "summary")
    report.check(list(cases) == EXPECTED_CASES, "summary case order drift")

    for oracle_case in oracle_cases:
        case_id = str(oracle_case["id"])
        summary_case = cases.get(case_id, {})
        path = f"case.{case_id}"
        report.check(summary_case.get("kind") == "memory-snapshot", f"{path}.kind drift")
        report.check(summary_case.get("prompt_case") == oracle_case.get("prompt_case"), f"{path}.prompt_case drift")
        report.check(summary_case.get("snapshot_file") == oracle_case.get("snapshot_file"), f"{path}.snapshot_file drift")
        report.check(summary_case.get("snapshot_bytes") == oracle_case.get("snapshot_bytes"), f"{path}.snapshot_bytes drift")
        report.check(raw_import.is_sha256_hex(summary_case.get("snapshot_sha256")), f"{path}.snapshot_sha invalid")
        report.check(summary_case.get("oracle_snapshot_sha256") == oracle_case.get("snapshot_sha256"), f"{path}.oracle_snapshot_sha drift")
        report.check(
            summary_case.get("snapshot_sha256_matches_oracle")
            == (summary_case.get("snapshot_sha256") == oracle_case.get("snapshot_sha256")),
            f"{path}.snapshot sha match flag drift",
        )
        if raw_root is not None:
            raw_path = raw_root / str(oracle_case["snapshot_file"])
            report.check(raw_path.is_file(), f"{path}.raw file missing")
            if raw_path.is_file():
                report.check(raw_path.stat().st_size == int(oracle_case["snapshot_bytes"]), f"{path}.raw size drift")
                report.check(raw_import.sha256_file(raw_path) == summary_case.get("snapshot_sha256"), f"{path}.raw sha drift")
        rust = raw_import.require_dict(report, summary_case.get("rust"), f"{path}.rust")
        report.check(rust.get("id") == case_id, f"{path}.rust.id drift")
        report.check(rust.get("path") == oracle_case.get("snapshot_file"), f"{path}.rust.path drift")
        report.check(rust.get("payload_bytes") == oracle_case.get("snapshot_bytes"), f"{path}.rust.payload_bytes drift")
        fnv = str(rust.get("fnv1a64", ""))
        report.check(len(fnv) == 16 and all(c in "0123456789abcdef" for c in fnv), f"{path}.rust.fnv drift")
        report.check(rust.get("ok") is True, f"{path}.rust.ok drift")
        report.check(rust.get("code") == "ok", f"{path}.rust.code drift")
        report.check(rust.get("error") == "", f"{path}.rust.error drift")
        raw_import.compare_value(report, expected_parsed(oracle_case), rust.get("parsed"), f"{path}.rust.parsed")
    static_checks(report)
    return report


def static_checks(report: raw_import.Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "readme": ROOT / "ds4-parity" / "README.md",
        "report": ROOT / "ds4-parity" / "run_parity_report.py",
        "dumper": ROOT / "ds4_restore_dump.c",
    }
    text = {name: path.read_text() for name, path in files.items()}
    status_flat = " ".join(text["status"].split())
    report.check("M10.7c3a: Rust Memory Snapshot Raw Body Import Smoke" in text["roadmap"], "roadmap M10.7c3a missing")
    report.check("M10.7c3a: Rust Memory Snapshot Raw Body Import Smoke" in text["todo"], "TODO M10.7c3a missing")
    report.check("M10.7c3a Rust Memory Snapshot Raw Body Import Smoke" in status_flat, "status M10.7c3a missing")
    report.check("compare_graph_snapshot_raw_import.py" in text["readme"], "README snapshot import command missing")
    report.check("M10.7c3a Rust raw graph snapshot import comparator" in text["report"], "unified report M10.7c3a missing")
    report.check("M10.7c3a B300 Rust raw graph snapshot import rerun" in text["report"], "B300 snapshot import rerun missing")
    report.check("--snapshot-dir" in text["dumper"], "ds4-restore-dump snapshot-dir flag missing")


def run_negative_tests(oracle: dict[str, Any], summary: dict[str, Any]) -> raw_import.Report:
    report = raw_import.Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("case order drift", ["cases", 0, "id"], "snapshot_continuation"),
        ("snapshot sha format drift", ["cases", 0, "snapshot_sha256"], "0" * 64),
        ("oracle snapshot sha drift", ["cases", 0, "oracle_snapshot_sha256"], "0" * 64),
        (
            "snapshot sha match flag drift",
            ["cases", 0, "snapshot_sha256_matches_oracle"],
            not summary["cases"][0]["snapshot_sha256_matches_oracle"],
        ),
        ("snapshot byte drift", ["cases", 1, "snapshot_bytes"], summary["cases"][1]["snapshot_bytes"] + 4),
        ("rust ok drift", ["cases", 0, "rust", "ok"], False),
        ("rust parsed token drift", ["cases", 1, "rust", "parsed", "token_count"], 1),
        ("section byte drift", ["cases", 0, "rust", "parsed", "section_bytes", "raw_rows"], 1),
        ("policy drift", ["raw_body_policy"], "raw bodies committed"),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(summary)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = validate_summary(oracle, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--live", action="store_true", help="run the Rust file probe against raw snapshot files")
    parser.add_argument("--workdir", type=Path, default=ROOT, help="repo/workdir containing raw snapshot files for --live")
    parser.add_argument("--write-summary", type=Path, help="write the live Rust import summary JSON")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        oracle = raw_import.load_json(args.oracle)
        summary = build_live_summary(oracle, args.workdir) if args.live else raw_import.load_json(args.summary)
        if args.write_summary:
            raw_import.write_json(args.write_summary, summary)
    except Exception as exc:
        print(f"raw graph snapshot import comparator: FAIL: {exc}")
        return 1

    raw_root = args.workdir if args.live else None
    report = validate_summary(oracle, summary, raw_root=raw_root)
    raw_import.print_report("Raw graph snapshot import comparator", report)

    negative = raw_import.Report()
    if args.negative_test:
        negative = run_negative_tests(oracle, summary)
        raw_import.print_report("Raw graph snapshot import negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
