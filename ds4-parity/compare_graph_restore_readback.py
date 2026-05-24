#!/usr/bin/env python3
"""Compare Rust graph restore tensor readback against the M7.8 raw bodies."""

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
DISK_SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c2" / "rust-b300-raw-import.json"
SNAPSHOT_SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c3a" / "rust-b300-snapshot-raw-import.json"
SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c3c" / "rust-b300-restore-readback.json"
SCHEMA = "ds4.rust_graph_restore_readback_summary.v1"
RUST_SCHEMA = "ds4.rust_graph_restore_readback.v1"
EXPECTED_CASES = [
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
]
SAMPLE_LAYERS = [0, 2, 3, 42]


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
        raise TypeError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def compare_value(report: Report, expected: Any, got: Any, path: str) -> None:
    if isinstance(expected, dict):
        got_dict = require_dict(report, got, path)
        report.check(list(expected) == list(got_dict), f"{path}: key order or coverage drift")
        for key, expected_value in expected.items():
            if key in got_dict:
                compare_value(report, expected_value, got_dict[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        got_list = require_list(report, got, path)
        report.check(len(expected) == len(got_list), f"{path}: array length drift")
        for idx, expected_value in enumerate(expected):
            if idx < len(got_list):
                compare_value(report, expected_value, got_list[idx], f"{path}[{idx}]")
        return
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def cases_by_id(report: Report, cases: Any, label: str) -> dict[str, dict[str, Any]]:
    got = require_list(report, cases, f"{label}.cases")
    report.check(len(got) == len(EXPECTED_CASES), f"{label}.case count drift")
    out: dict[str, dict[str, Any]] = {}
    for case in got:
        report.check(isinstance(case, dict), f"{label}.case must be object")
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        report.check(isinstance(case_id, str), f"{label}.case id must be string")
        if isinstance(case_id, str):
            out[case_id] = case
    report.check(list(out) == EXPECTED_CASES, f"{label}.case order drift")
    return out


def restore_oracle_cases(report: Report, oracle: dict[str, Any]) -> list[dict[str, Any]]:
    report.check(oracle.get("schema") == "ds4.restore_oracle.v1", "oracle schema drift")
    report.check(oracle.get("source") == "current-c-b300-restore", "oracle source drift")
    by_id: dict[str, dict[str, Any]] = {}
    for case in require_list(report, oracle.get("cases"), "oracle.cases"):
        if isinstance(case, dict):
            case_id = case.get("id")
            if isinstance(case_id, str) and case_id in EXPECTED_CASES:
                by_id[case_id] = case
    report.check(list(by_id) == EXPECTED_CASES, "oracle restore case order drift")
    return [normalize_oracle_case(by_id[case_id]) for case_id in EXPECTED_CASES if case_id in by_id]


def normalize_oracle_case(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("kind") == "disk-payload":
        return {
            **case,
            "raw_file": case["payload_file"],
            "raw_bytes": int(case["payload_bytes"]),
            "oracle_raw_sha256": case["payload_sha256"],
            "payload_bytes": int(case["payload_bytes"]),
        }
    if case.get("kind") == "memory-snapshot":
        raw_file = f"ds4-parity/baselines/kv/m7.8/raw/{case['id']}.dsv4"
        return {
            **case,
            "raw_file": raw_file,
            "raw_bytes": int(case["snapshot_bytes"]),
            "oracle_raw_sha256": case["snapshot_sha256"],
            "payload_bytes": int(case["snapshot_bytes"]),
        }
    raise ValueError(f"{case.get('id')}: unsupported restore kind {case.get('kind')!r}")


def run_rust_readback(oracle_cases: list[dict[str, Any]], workdir: Path) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ds4-gpu",
        "--features",
        "cuda-backend",
        "--bin",
        "ds4-graph-restore-readback",
        "--",
    ]
    for case in oracle_cases:
        command.extend(["--case", f"{case['id']}:{case['raw_file']}"])
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
        raise TypeError("Rust restore readback did not return an object")
    return data


def build_live_summary(oracle: dict[str, Any], workdir: Path) -> dict[str, Any]:
    report = Report()
    oracle_cases = restore_oracle_cases(report, oracle)
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    rust = run_rust_readback(oracle_cases, workdir)
    rust_cases = cases_by_id(Report(), rust.get("cases"), "rust")
    cases = []
    for case in oracle_cases:
        raw_path = workdir / str(case["raw_file"])
        raw_sha256 = raw_import.sha256_file(raw_path)
        rust_case = rust_cases.get(str(case["id"]), {})
        cases.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "prompt_case": case["prompt_case"],
                "raw_file": case["raw_file"],
                "raw_bytes": case["raw_bytes"],
                "raw_sha256": raw_sha256,
                "oracle_raw_sha256": case["oracle_raw_sha256"],
                "raw_sha256_matches_oracle": raw_sha256 == case["oracle_raw_sha256"],
                "rust": rust_case,
            }
        )
    return {
        "schema": SCHEMA,
        "source": "rust-b300-graph-restore-readback",
        "oracle": "ds4-parity/baselines/kv/m7.8/current-c.json",
        "raw_body_policy": "hash-only; raw disk payload and memory snapshot bodies remain on B300 and are not committed",
        "b300": {
            "kube_context": "hou2-prod1",
            "pod": "ds4-rust-port-b300",
            "workdir": "/workspace/ds4",
        },
        "rust_readback": {
            "schema": rust.get("schema"),
            "source": rust.get("source"),
            "runtime": rust.get("runtime"),
        },
        "cases": cases,
    }


def source_metadata(disk_summary: dict[str, Any], snapshot_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for case in disk_summary.get("cases", []):
        if isinstance(case, dict):
            rust = case.get("rust") if isinstance(case.get("rust"), dict) else {}
            out[str(case.get("id"))] = {
                "raw_sha256": case.get("payload_sha256"),
                "oracle_raw_sha256": case.get("oracle_payload_sha256"),
                "raw_sha256_matches_oracle": case.get("payload_sha256_matches_oracle"),
                "file_fnv1a64": rust.get("fnv1a64"),
            }
    for case in snapshot_summary.get("cases", []):
        if isinstance(case, dict):
            rust = case.get("rust") if isinstance(case.get("rust"), dict) else {}
            out[str(case.get("id"))] = {
                "raw_sha256": case.get("snapshot_sha256"),
                "oracle_raw_sha256": case.get("oracle_snapshot_sha256"),
                "raw_sha256_matches_oracle": case.get("snapshot_sha256_matches_oracle"),
                "file_fnv1a64": rust.get("fnv1a64"),
            }
    return out


def validate_summary(
    oracle: dict[str, Any],
    summary: dict[str, Any],
    disk_summary: dict[str, Any],
    snapshot_summary: dict[str, Any],
) -> Report:
    report = Report()
    oracle_cases = restore_oracle_cases(report, oracle)
    metadata = source_metadata(disk_summary, snapshot_summary)

    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == "rust-b300-graph-restore-readback", "summary source drift")
    report.check(
        summary.get("oracle") == "ds4-parity/baselines/kv/m7.8/current-c.json",
        "summary oracle path drift",
    )
    report.check("hash-only" in str(summary.get("raw_body_policy")), "raw-body policy drift")
    b300 = require_dict(report, summary.get("b300"), "summary.b300")
    report.check(b300.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(b300.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(b300.get("workdir") == "/workspace/ds4", "B300 workdir drift")
    rust_readback = require_dict(report, summary.get("rust_readback"), "summary.rust_readback")
    report.check(rust_readback.get("schema") == RUST_SCHEMA, "Rust readback schema drift")
    report.check(rust_readback.get("source") == "rust-graph-restore-readback", "Rust readback source drift")
    runtime = require_dict(report, rust_readback.get("runtime"), "summary.rust_readback.runtime")
    report.check(runtime.get("ctx") == 32768, "Rust readback runtime ctx drift")
    report.check(runtime.get("kind") == "default-graph", "Rust readback runtime kind drift")
    report.check(runtime.get("backend") == "ds4-gpu", "Rust readback runtime backend drift")

    cases = cases_by_id(report, summary.get("cases"), "summary")
    for oracle_case in oracle_cases:
        case_id = str(oracle_case["id"])
        summary_case = cases.get(case_id, {})
        expected_meta = metadata.get(case_id, {})
        path = f"case.{case_id}"
        report.check(summary_case.get("kind") == oracle_case.get("kind"), f"{path}.kind drift")
        report.check(summary_case.get("prompt_case") == oracle_case.get("prompt_case"), f"{path}.prompt_case drift")
        report.check(summary_case.get("raw_file") == oracle_case.get("raw_file"), f"{path}.raw_file drift")
        report.check(summary_case.get("raw_bytes") == oracle_case.get("raw_bytes"), f"{path}.raw_bytes drift")
        report.check(raw_import.is_sha256_hex(summary_case.get("raw_sha256")), f"{path}.raw_sha256 invalid")
        report.check(summary_case.get("raw_sha256") == expected_meta.get("raw_sha256"), f"{path}.raw_sha256 source drift")
        report.check(
            summary_case.get("oracle_raw_sha256") == expected_meta.get("oracle_raw_sha256"),
            f"{path}.oracle_raw_sha256 drift",
        )
        report.check(
            summary_case.get("raw_sha256_matches_oracle") == expected_meta.get("raw_sha256_matches_oracle"),
            f"{path}.raw sha match flag drift",
        )
        rust = require_dict(report, summary_case.get("rust"), f"{path}.rust")
        validate_rust_case(report, oracle_case, rust, expected_meta, path)
    static_checks(report)
    return report


def validate_rust_case(
    report: Report,
    oracle_case: dict[str, Any],
    rust: dict[str, Any],
    expected_meta: dict[str, Any],
    path: str,
) -> None:
    report.check(rust.get("id") == oracle_case.get("id"), f"{path}.rust.id drift")
    report.check(rust.get("path") == oracle_case.get("raw_file"), f"{path}.rust.path drift")
    report.check(rust.get("payload_bytes") == oracle_case.get("raw_bytes"), f"{path}.rust.payload_bytes drift")
    report.check(rust.get("file_fnv1a64") == expected_meta.get("file_fnv1a64"), f"{path}.rust.file_fnv drift")
    report.check(rust.get("ok") is True, f"{path}.rust.ok drift")
    report.check(rust.get("error") == "", f"{path}.rust.error drift")
    compare_value(report, raw_import.expected_parsed(oracle_case), rust.get("parsed"), f"{path}.rust.parsed")
    readback = require_dict(report, rust.get("readback"), f"{path}.rust.readback")
    validate_readback(report, oracle_case, readback, path)


def validate_readback(report: Report, oracle_case: dict[str, Any], readback: dict[str, Any], path: str) -> None:
    parsed = raw_import.expected_parsed(oracle_case)
    sections = parsed["section_bytes"]
    expected_digest_bytes = {
        "checkpoint": sections["tokens"],
        "logits": sections["logits"],
        "attn_counts": sections["attn_counts"],
        "index_counts": sections["index_counts"],
        "raw_rows": sections["raw_rows"],
        "attn_compressed_rows": sections["attn_compressed_rows"],
        "attn_state_kv": sections["attn_state"] // 2,
        "attn_state_score": sections["attn_state"] // 2,
        "indexer_compressed_rows": sections["indexer_compressed_rows"],
        "index_state_kv": sections["indexer_state"] // 2,
        "index_state_score": sections["indexer_state"] // 2,
    }
    for key, byte_count in expected_digest_bytes.items():
        validate_digest(report, readback.get(key), byte_count, f"{path}.rust.readback.{key}")

    samples = require_list(report, readback.get("samples"), f"{path}.rust.readback.samples")
    report.check([sample.get("layer") for sample in samples if isinstance(sample, dict)] == SAMPLE_LAYERS, f"{path}.sample layer drift")
    header = raw_import.decode_header(str(oracle_case["header_prefix_hex"]))
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        layer = sample.get("layer")
        if not isinstance(layer, int):
            report.check(False, f"{path}.sample layer invalid")
            continue
        validate_sample(report, header, layer, sample, f"{path}.sample[{layer}]")

    state = require_dict(report, readback.get("post_restore_state"), f"{path}.rust.readback.post_restore_state")
    n_comp, n_index_comp = layer_counts(header["prompt_tokens"])
    expected_state = {
        "checkpoint_valid": True,
        "mtp_draft_valid": False,
        "mtp_n_raw": 0,
        "layer_n_comp": n_comp,
        "layer_n_index_comp": n_index_comp,
    }
    compare_value(report, expected_state, state, f"{path}.rust.readback.post_restore_state")


def validate_sample(report: Report, header: dict[str, int], layer: int, sample: dict[str, Any], path: str) -> None:
    ratio = raw_import.compress_ratio(layer)
    report.check(sample.get("ratio") == ratio, f"{path}.ratio drift")
    raw_bytes = header["raw_live_rows"] * raw_import.N_HEAD_DIM * 4
    validate_digest(report, sample.get("raw"), raw_bytes, f"{path}.raw")
    if ratio == 0:
        for key in (
            "attn_compressed_rows",
            "attn_state_kv",
            "attn_state_score",
            "indexer_compressed_rows",
            "index_state_kv",
            "index_state_score",
        ):
            report.check(sample.get(key) is None, f"{path}.{key} should be null for dense layer")
        return
    n_comp = raw_import.compressed_rows(header["prompt_tokens"], ratio)
    validate_digest(
        report,
        sample.get("attn_compressed_rows"),
        n_comp * raw_import.N_HEAD_DIM * 4,
        f"{path}.attn_compressed_rows",
    )
    validate_digest(report, sample.get("attn_state_kv"), raw_import.layer_attn_state_bytes(ratio), f"{path}.attn_state_kv")
    validate_digest(report, sample.get("attn_state_score"), raw_import.layer_attn_state_bytes(ratio), f"{path}.attn_state_score")
    if ratio == 4:
        validate_digest(
            report,
            sample.get("indexer_compressed_rows"),
            n_comp * raw_import.N_INDEXER_HEAD_DIM * 4,
            f"{path}.indexer_compressed_rows",
        )
        validate_digest(report, sample.get("index_state_kv"), raw_import.layer_index_state_bytes(ratio), f"{path}.index_state_kv")
        validate_digest(report, sample.get("index_state_score"), raw_import.layer_index_state_bytes(ratio), f"{path}.index_state_score")
    else:
        for key in ("indexer_compressed_rows", "index_state_kv", "index_state_score"):
            report.check(sample.get(key) is None, f"{path}.{key} should be null for ratio-128 layer")


def validate_digest(report: Report, obj: Any, bytes_expected: int, path: str) -> None:
    digest = require_dict(report, obj, path)
    report.check(digest.get("bytes") == bytes_expected, f"{path}.bytes drift")
    source = digest.get("source_fnv1a64")
    readback = digest.get("readback_fnv1a64")
    report.check(is_fnv_hex(source), f"{path}.source_fnv invalid")
    report.check(is_fnv_hex(readback), f"{path}.readback_fnv invalid")
    report.check(source == readback, f"{path}.source/readback FNV drift")
    report.check(digest.get("matched") is True, f"{path}.matched drift")


def is_fnv_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 16 and all(ch in "0123456789abcdef" for ch in value)


def layer_counts(token_count: int) -> tuple[list[int], list[int]]:
    n_comp: list[int] = []
    n_index_comp: list[int] = []
    for layer in range(raw_import.N_LAYER):
        ratio = raw_import.compress_ratio(layer)
        count = raw_import.compressed_rows(token_count, ratio)
        n_comp.append(count)
        n_index_comp.append(count if ratio == 4 else 0)
    return n_comp, n_index_comp


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory" / "TODO.md",
        "status": ROOT / ".memory" / "status.md",
        "readme": ROOT / "ds4-parity" / "README.md",
        "report": ROOT / "ds4-parity" / "run_parity_report.py",
        "rust_bin": ROOT / "rust" / "ds4-gpu" / "src" / "bin" / "ds4-graph-restore-readback.rs",
        "c_core": ROOT / "ds4.c",
    }
    text = {name: path.read_text() for name, path in files.items()}
    status_flat = " ".join(text["status"].split())
    report.check("M10.7c3c: Rust Graph Tensor Restore Readback Smoke" in text["roadmap"], "roadmap M10.7c3c missing")
    report.check("M10.7c3c: Rust Graph Tensor Restore Readback Smoke" in text["todo"], "TODO M10.7c3c missing")
    report.check("M10.7c3c Rust Graph Tensor Restore Readback Smoke" in status_flat, "status M10.7c3c missing")
    report.check("compare_graph_restore_readback.py" in text["readme"], "README restore readback command missing")
    report.check("M10.7c3c Rust graph restore readback comparator" in text["report"], "unified report M10.7c3c missing")
    report.check("M10.7c3c B300 Rust graph restore readback rerun" in text["report"], "B300 restore readback rerun missing")
    report.check("ds4-graph-restore-readback" in text["rust_bin"], "Rust restore readback bin marker missing")
    for snippet in (
        "layer_raw_cache",
        "layer_attn_comp_cache",
        "layer_index_comp_cache",
        "post_restore_state",
    ):
        report.check(snippet in text["rust_bin"], f"Rust restore readback {snippet} marker missing")
    for snippet in (
        "payload_read_tensor_span",
        "g->layer_raw_cache[il]",
        "g->layer_attn_comp_cache[il]",
        "g->layer_index_comp_cache[il]",
    ):
        report.check(snippet in text["c_core"], f"C restore snippet missing: {snippet}")


def run_negative_tests(
    oracle: dict[str, Any],
    summary: dict[str, Any],
    disk_summary: dict[str, Any],
    snapshot_summary: dict[str, Any],
) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("raw sha drift", ["cases", 0, "raw_sha256"], "0" * 64),
        ("rust fnv drift", ["cases", 0, "rust", "file_fnv1a64"], "0" * 16),
        ("raw readback flag drift", ["cases", 0, "rust", "readback", "raw_rows", "matched"], False),
        ("logits readback fnv drift", ["cases", 0, "rust", "readback", "logits", "readback_fnv1a64"], "f" * 16),
        ("counter drift", ["cases", 1, "rust", "readback", "post_restore_state", "layer_n_comp", 2], 0),
        ("sample byte drift", ["cases", 2, "rust", "readback", "samples", 1, "raw", "bytes"], 1),
        ("runtime drift", ["rust_readback", "runtime", "backend"], "not-ds4-gpu"),
        ("policy drift", ["raw_body_policy"], "raw bodies committed"),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(summary)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = validate_summary(oracle, bad, disk_summary, snapshot_summary)
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
    parser.add_argument("--disk-summary", type=Path, default=DISK_SUMMARY)
    parser.add_argument("--snapshot-summary", type=Path, default=SNAPSHOT_SUMMARY)
    parser.add_argument("--live", action="store_true", help="run the Rust GPU readback over raw B300 files")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    oracle = load_json(args.oracle)
    disk_summary = load_json(args.disk_summary)
    snapshot_summary = load_json(args.snapshot_summary)
    if args.live:
        summary = build_live_summary(oracle, args.workdir)
        if args.write_summary:
            write_json(args.write_summary, summary)
    else:
        summary = load_json(args.summary)

    report = validate_summary(oracle, summary, disk_summary, snapshot_summary)
    print_report("Graph restore readback comparator", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(oracle, summary, disk_summary, snapshot_summary)
        print_report("Graph restore readback negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
