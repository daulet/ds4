#!/usr/bin/env python3
"""Compare Rust raw graph payload import against the M7.8 disk payload oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.8" / "current-c.json"
SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c2" / "rust-b300-raw-import.json"
SCHEMA = "ds4.rust_graph_payload_raw_import.v1"
RUST_SCHEMA = "ds4.rust_graph_payload_file_probe.v1"
EXPECTED_CASES = ["disk_seed_payload", "disk_continuation_payload"]
HEADER_BYTES = 52
N_LAYER = 43
N_HEAD_DIM = 512
N_INDEXER_HEAD_DIM = 128
N_VOCAB = 129_280
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and any(ch != "0" for ch in value)
        and all(ch in "0123456789abcdef" for ch in value)
    )


def decode_header(hex_value: str) -> dict[str, int]:
    header = bytes.fromhex(hex_value)
    if len(header) != HEADER_BYTES:
        raise ValueError(f"header prefix must be {HEADER_BYTES} bytes, got {len(header)}")
    values = struct.unpack("<13I", header)
    return dict(zip(HEADER_FIELDS, values, strict=True))


def compress_ratio(layer: int) -> int:
    if layer < 2:
        return 0
    return 4 if layer & 1 == 0 else 128


def compressed_rows(token_count: int, ratio: int) -> int:
    return 0 if ratio == 0 else token_count // ratio


def layer_attn_state_bytes(ratio: int) -> int:
    coff = 2 if ratio == 4 else 1
    return coff * N_HEAD_DIM * coff * ratio * 4


def layer_index_state_bytes(ratio: int) -> int:
    coff = 2 if ratio == 4 else 1
    return coff * N_INDEXER_HEAD_DIM * coff * ratio * 4


def expected_sections(header: dict[str, int]) -> dict[str, int]:
    sections = {
        "header": HEADER_BYTES,
        "tokens": header["prompt_tokens"] * 4,
        "logits": N_VOCAB * 4,
        "attn_counts": N_LAYER * 4,
        "index_counts": N_LAYER * 4,
        "raw_rows": 0,
        "attn_compressed_rows": 0,
        "attn_state": 0,
        "indexer_compressed_rows": 0,
        "indexer_state": 0,
    }
    for layer in range(N_LAYER):
        sections["raw_rows"] += header["raw_live_rows"] * N_HEAD_DIM * 4
        ratio = compress_ratio(layer)
        if ratio == 0:
            continue
        n_comp = compressed_rows(header["prompt_tokens"], ratio)
        sections["attn_compressed_rows"] += n_comp * N_HEAD_DIM * 4
        sections["attn_state"] += 2 * layer_attn_state_bytes(ratio)
        if ratio == 4:
            sections["indexer_compressed_rows"] += n_comp * N_INDEXER_HEAD_DIM * 4
            sections["indexer_state"] += 2 * layer_index_state_bytes(ratio)
    return sections


def expected_parsed(case: dict[str, Any]) -> dict[str, Any]:
    header = decode_header(str(case["header_prefix_hex"]))
    token_count = header["prompt_tokens"]
    raw_first_pos = token_count - header["raw_live_rows"]
    raw_last_pos = token_count - 1
    return {
        "token_count": token_count,
        "raw_first_pos": raw_first_pos,
        "raw_last_pos": raw_last_pos,
        "raw_first_phys": raw_first_pos % header["raw_cap"],
        "raw_last_phys": raw_last_pos % header["raw_cap"],
        "payload_bytes": int(case["payload_bytes"]),
        "ratio4_rows": compressed_rows(token_count, 4),
        "ratio128_rows": compressed_rows(token_count, 128),
        "layer2_n_index_comp": compressed_rows(token_count, 4),
        "section_bytes": expected_sections(header),
    }


def disk_oracle_cases(report: Report, oracle: dict[str, Any]) -> list[dict[str, Any]]:
    report.check(oracle.get("schema") == "ds4.restore_oracle.v1", "oracle schema drift")
    report.check(oracle.get("source") == "current-c-b300-restore", "oracle source drift")
    cases = oracle.get("cases")
    report.check(isinstance(cases, list), "oracle cases must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases if isinstance(cases, list) else []:
        if isinstance(case, dict) and case.get("kind") == "disk-payload":
            case_id = case.get("id")
            if isinstance(case_id, str):
                by_id[case_id] = case
    report.check(list(by_id) == EXPECTED_CASES, "oracle disk payload case order drift")
    return [by_id[case_id] for case_id in EXPECTED_CASES if case_id in by_id]


def run_rust_probe(oracle_cases: list[dict[str, Any]], workdir: Path) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ds4-gguf",
        "--bin",
        "ds4-session-payload-dump-rs",
        "--",
    ]
    for case in oracle_cases:
        command.extend(["--graph-file-probe", f"{case['id']}:{case['payload_file']}"])
    proc = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise TypeError("Rust graph file probe did not return an object")
    return data


def build_live_summary(oracle: dict[str, Any], workdir: Path) -> dict[str, Any]:
    report = Report()
    oracle_cases = disk_oracle_cases(report, oracle)
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    rust = run_rust_probe(oracle_cases, workdir)
    rust_cases = cases_by_id(Report(), rust.get("cases"), "rust")
    cases = []
    for case in oracle_cases:
        payload_path = workdir / str(case["payload_file"])
        payload_sha256 = sha256_file(payload_path)
        rust_case = rust_cases.get(str(case["id"]), {})
        cases.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "prompt_case": case["prompt_case"],
                "payload_file": case["payload_file"],
                "payload_bytes": int(case["payload_bytes"]),
                "payload_sha256": payload_sha256,
                "oracle_payload_sha256": case["payload_sha256"],
                "payload_sha256_matches_oracle": payload_sha256 == case["payload_sha256"],
                "rust": rust_case,
            }
        )
    return {
        "schema": SCHEMA,
        "source": "rust-b300-raw-payload-import",
        "oracle": "ds4-parity/baselines/kv/m7.8/current-c.json",
        "raw_body_policy": "hash-only; raw restore bodies remain on B300 and are not committed",
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


def cases_by_id(report: Report, cases: Any, label: str) -> dict[str, dict[str, Any]]:
    report.check(isinstance(cases, list), f"{label}.cases must be a list")
    out: dict[str, dict[str, Any]] = {}
    for case in cases if isinstance(cases, list) else []:
        report.check(isinstance(case, dict), f"{label}.case must be object")
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        report.check(isinstance(case_id, str), f"{label}.case id must be string")
        if isinstance(case_id, str):
            out[case_id] = case
    return out


def require_dict(report: Report, value: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{path}: expected object")
    return value if isinstance(value, dict) else {}


def compare_value(report: Report, expected: Any, got: Any, path: str) -> None:
    if isinstance(expected, dict):
        got_dict = require_dict(report, got, path)
        report.check(list(expected) == list(got_dict), f"{path}: key order or coverage drift")
        for key, expected_value in expected.items():
            if key in got_dict:
                compare_value(report, expected_value, got_dict[key], f"{path}.{key}")
        return
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def validate_summary(
    oracle: dict[str, Any],
    summary: dict[str, Any],
    *,
    raw_root: Path | None = None,
) -> Report:
    report = Report()
    oracle_cases = disk_oracle_cases(report, oracle)
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == "rust-b300-raw-payload-import", "summary source drift")
    report.check(
        summary.get("oracle") == "ds4-parity/baselines/kv/m7.8/current-c.json",
        "summary oracle path drift",
    )
    report.check("hash-only" in str(summary.get("raw_body_policy")), "raw-body policy drift")
    b300 = require_dict(report, summary.get("b300"), "summary.b300")
    report.check(b300.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(b300.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(b300.get("workdir") == "/workspace/ds4", "B300 workdir drift")
    rust_probe = require_dict(report, summary.get("rust_probe"), "summary.rust_probe")
    report.check(rust_probe.get("schema") == RUST_SCHEMA, "Rust probe schema drift")
    report.check(rust_probe.get("source") == "rust-graph-payload-file-probe", "Rust probe source drift")
    cases = cases_by_id(report, summary.get("cases"), "summary")
    report.check(list(cases) == EXPECTED_CASES, "summary case order drift")

    for oracle_case in oracle_cases:
        case_id = str(oracle_case["id"])
        summary_case = cases.get(case_id, {})
        path = f"case.{case_id}"
        report.check(summary_case.get("kind") == "disk-payload", f"{path}.kind drift")
        report.check(summary_case.get("prompt_case") == oracle_case.get("prompt_case"), f"{path}.prompt_case drift")
        report.check(summary_case.get("payload_file") == oracle_case.get("payload_file"), f"{path}.payload_file drift")
        report.check(summary_case.get("payload_bytes") == oracle_case.get("payload_bytes"), f"{path}.payload_bytes drift")
        report.check(is_sha256_hex(summary_case.get("payload_sha256")), f"{path}.payload_sha invalid")
        report.check(summary_case.get("oracle_payload_sha256") == oracle_case.get("payload_sha256"), f"{path}.oracle_payload_sha drift")
        report.check(
            summary_case.get("payload_sha256_matches_oracle")
            == (summary_case.get("payload_sha256") == oracle_case.get("payload_sha256")),
            f"{path}.payload sha match flag drift",
        )
        if raw_root is not None:
            raw_path = raw_root / str(oracle_case["payload_file"])
            if raw_path.exists():
                report.check(raw_path.stat().st_size == int(oracle_case["payload_bytes"]), f"{path}.raw size drift")
                report.check(sha256_file(raw_path) == summary_case.get("payload_sha256"), f"{path}.raw sha drift")
        rust = require_dict(report, summary_case.get("rust"), f"{path}.rust")
        report.check(rust.get("id") == case_id, f"{path}.rust.id drift")
        report.check(rust.get("path") == oracle_case.get("payload_file"), f"{path}.rust.path drift")
        report.check(rust.get("payload_bytes") == oracle_case.get("payload_bytes"), f"{path}.rust.payload_bytes drift")
        fnv = str(rust.get("fnv1a64", ""))
        report.check(len(fnv) == 16 and all(c in "0123456789abcdef" for c in fnv), f"{path}.rust.fnv drift")
        report.check(rust.get("ok") is True, f"{path}.rust.ok drift")
        report.check(rust.get("code") == "ok", f"{path}.rust.code drift")
        report.check(rust.get("error") == "", f"{path}.rust.error drift")
        compare_value(report, expected_parsed(oracle_case), rust.get("parsed"), f"{path}.rust.parsed")
    static_checks(report)
    return report


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "readme": ROOT / "ds4-parity/README.md",
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "rust_bin": ROOT / "rust/ds4-gguf/src/bin/ds4-session-payload-dump-rs.rs",
    }
    text = {name: path.read_text() for name, path in files.items()}
    status_flat = " ".join(text["status"].split())
    report.check("M10.7c2: Rust Disk KV Payload Byte Import Smoke" in text["roadmap"], "roadmap M10.7c2 missing")
    report.check("M10.7c2: Rust Disk KV Payload Byte Import Smoke" in text["todo"], "TODO M10.7c2 missing")
    report.check("M10.7c2 Rust Disk KV Payload Byte Import Smoke" in status_flat, "status M10.7c2 missing")
    report.check("compare_graph_payload_raw_import.py" in text["readme"], "README raw import command missing")
    report.check("M10.7c2 Rust raw graph payload import comparator" in text["report"], "unified report M10.7c2 missing")
    report.check("M10.7c2 B300 Rust raw graph payload import rerun" in text["report"], "B300 raw import rerun missing")
    report.check("--graph-file-probe" in text["rust_bin"], "Rust graph file probe flag missing")


def run_negative_tests(oracle: dict[str, Any], summary: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("case order drift", ["cases", 0, "id"], "disk_continuation_payload"),
        ("payload sha format drift", ["cases", 0, "payload_sha256"], "0" * 64),
        ("oracle payload sha drift", ["cases", 0, "oracle_payload_sha256"], "0" * 64),
        (
            "payload sha match flag drift",
            ["cases", 0, "payload_sha256_matches_oracle"],
            not summary["cases"][0]["payload_sha256_matches_oracle"],
        ),
        ("payload byte drift", ["cases", 1, "payload_bytes"], summary["cases"][1]["payload_bytes"] + 4),
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


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--live", action="store_true", help="run the Rust file probe against raw payload files")
    parser.add_argument("--workdir", type=Path, default=ROOT, help="repo/workdir containing raw payload files for --live")
    parser.add_argument("--write-summary", type=Path, help="write the live Rust import summary JSON")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        oracle = load_json(args.oracle)
        summary = build_live_summary(oracle, args.workdir) if args.live else load_json(args.summary)
        if args.write_summary:
            write_json(args.write_summary, summary)
    except Exception as exc:
        print(f"raw graph payload import comparator: FAIL: {exc}")
        return 1

    raw_root = args.workdir if args.live else None
    report = validate_summary(oracle, summary, raw_root=raw_root)
    print_report("Raw graph payload import comparator", report)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(oracle, summary)
        print_report("Raw graph payload import negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
