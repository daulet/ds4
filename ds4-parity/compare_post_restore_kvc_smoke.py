#!/usr/bin/env python3
"""Compare the M10.7d3c2 B300 post-restore KVC file smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/kv/m10.7d3/rust-b300-post-restore-kvc.json"
CONTRACT = ROOT / "ds4-parity/baselines/kv/m10.7d3/post-restore-kvc-decision-contract.json"
RESTORE_SUMMARY = ROOT / "ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json"
RAW_DIR = Path("ds4-parity/baselines/kv/m7.8/raw")
SCHEMA = "ds4.rust_post_restore_kvc_smoke_summary.v1"
RUST_SCHEMA = "ds4.rust_post_restore_kvc_smoke.v1"
EXPECTED_CASES = (
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
)
TEXT_KEYS = {
    "disk_seed_payload": {
        "path": "ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/0ab2314538b11686a11e296b7f697651fbd17e60.txt",
        "source": "m0.5 seed rendered cache key",
    },
    "snapshot_seed": {
        "path": "ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/0ab2314538b11686a11e296b7f697651fbd17e60.txt",
        "source": "m0.5 seed rendered cache key",
    },
    "disk_continuation_payload": {
        "path": "ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.txt",
        "source": "m0.5 continuation shutdown rendered cache key",
    },
    "snapshot_continuation": {
        "path": "ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.txt",
        "source": "m0.5 continuation shutdown rendered cache key",
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def named_items(report: Report, items: Any, label: str, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(require_list(report, items, label)):
        report.check(isinstance(item, dict), f"{label}[{index}]: expected object")
        if not isinstance(item, dict):
            continue
        name = item.get(key)
        report.check(isinstance(name, str) and bool(name), f"{label}[{index}].{key}: expected string")
        if isinstance(name, str) and name:
            report.check(name not in out, f"{label}: duplicate {key} {name!r}")
            out[name] = item
    return out


def compare_value(report: Report, expected: Any, got: Any, path: str) -> None:
    if isinstance(expected, dict):
        got_dict = require_dict(report, got, path)
        report.check(list(expected) == list(got_dict), f"{path}: key order or coverage drift")
        for key, expected_value in expected.items():
            if key in got_dict:
                compare_value(report, expected_value, got_dict[key], f"{path}.{key}")
        return
    report.check(expected == got, f"{path}: expected {expected!r}, got {got!r}")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def is_sha1(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def is_fnv(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 16 and all(ch in "0123456789abcdef" for ch in value)


def run_rust_smoke(workdir: Path, output_dir: Path) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ds4-gguf",
        "--bin",
        "ds4-post-restore-kvc-smoke",
        "--",
        "--output-dir",
        str(output_dir),
    ]
    for case_id in EXPECTED_CASES:
        text_path = TEXT_KEYS[case_id]["path"]
        payload_path = RAW_DIR / f"{case_id}.dsv4"
        command.extend(["--case", f"{case_id}:{payload_path.as_posix()}:{text_path}"])
    proc = subprocess.run(command, cwd=workdir, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise TypeError("Rust post-restore KVC smoke did not return an object")
    return data


def build_live_summary(workdir: Path, output_dir: Path) -> dict[str, Any]:
    rust = run_rust_smoke(workdir, output_dir)
    rust_cases = named_items(Report(), rust.get("cases"), "rust.cases", "id")
    cases = []
    for case_id in EXPECTED_CASES:
        payload_path = RAW_DIR / f"{case_id}.dsv4"
        text_path = Path(TEXT_KEYS[case_id]["path"])
        cases.append(
            {
                "id": case_id,
                "raw_file": payload_path.as_posix(),
                "raw_sha256": sha256_file(workdir / payload_path),
                "text_key": {
                    "path": text_path.as_posix(),
                    "source": TEXT_KEYS[case_id]["source"],
                    "sha1": sha1_file(workdir / text_path),
                    "sha256": sha256_file(workdir / text_path),
                    "bytes": file_size(workdir / text_path),
                },
                "rust": rust_cases.get(case_id, {}),
            }
        )
    return {
        "schema": SCHEMA,
        "source": "rust-b300-post-restore-kvc-smoke",
        "contract_path": rel(CONTRACT),
        "restore_summary_path": rel(RESTORE_SUMMARY),
        "raw_body_policy": "hash-only raw graph payload bodies remain on the B300; committed summary stores payload digests and KVC wrapper metadata",
        "text_key_policy": "rendered cache-key text files are committed M0.5 current-C artifacts selected by prompt family",
        "b300": {
            "kube_context": "hou2-prod1",
            "pod": "ds4-rust-port-b300",
            "workdir": "/workspace/ds4",
        },
        "rust_smoke": {
            "schema": rust.get("schema"),
            "source": rust.get("source"),
            "runtime": rust.get("runtime"),
        },
        "cases": cases,
    }


def validate_summary(
    summary: dict[str, Any],
    contract: dict[str, Any],
    restore_summary: dict[str, Any],
) -> Report:
    report = Report()
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == "rust-b300-post-restore-kvc-smoke", "summary source drift")
    report.check(summary.get("contract_path") == rel(CONTRACT), "contract path drift")
    report.check(summary.get("restore_summary_path") == rel(RESTORE_SUMMARY), "restore summary path drift")
    report.check("hash-only raw graph payload bodies" in str(summary.get("raw_body_policy")), "raw-body policy drift")
    report.check("M0.5 current-C artifacts" in str(summary.get("text_key_policy")), "text-key policy drift")
    validate_b300(report, summary.get("b300"))
    validate_rust_smoke_meta(report, summary.get("rust_smoke"))

    contract_cases = named_items(report, contract.get("post_restore_cases"), "contract.post_restore_cases", "name")
    restore_cases = named_items(report, restore_summary.get("cases"), "restore_summary.cases", "id")
    summary_cases = named_items(report, summary.get("cases"), "summary.cases", "id")
    report.check(tuple(summary_cases) == EXPECTED_CASES, "summary case order drift")
    for case_id in EXPECTED_CASES:
        path = f"cases.{case_id}"
        contract_case = contract_cases.get(case_id, {})
        restore_case = restore_cases.get(case_id, {})
        summary_case = summary_cases.get(case_id, {})
        validate_case(report, path, case_id, summary_case, contract_case, restore_case)
    static_checks(report)
    return report


def validate_b300(report: Report, obj: Any) -> None:
    b300 = require_dict(report, obj, "summary.b300")
    report.check(b300.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(b300.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(b300.get("workdir") == "/workspace/ds4", "B300 workdir drift")


def validate_rust_smoke_meta(report: Report, obj: Any) -> None:
    meta = require_dict(report, obj, "summary.rust_smoke")
    report.check(meta.get("schema") == RUST_SCHEMA, "Rust smoke schema drift")
    report.check(meta.get("source") == "rust-post-restore-kvc-smoke", "Rust smoke source drift")
    runtime = require_dict(report, meta.get("runtime"), "summary.rust_smoke.runtime")
    report.check(runtime.get("ctx") == 32768, "Rust smoke runtime ctx drift")
    report.check(runtime.get("kind") == "default-graph-payload", "Rust smoke runtime kind drift")
    report.check(runtime.get("kvc_writer") == "ds4_gguf::kv_policy", "Rust smoke KVC writer drift")


def validate_case(
    report: Report,
    path: str,
    case_id: str,
    summary_case: dict[str, Any],
    contract_case: dict[str, Any],
    restore_case: dict[str, Any],
) -> None:
    report.check(summary_case.get("raw_file") == contract_case.get("raw_file"), f"{path}.raw_file contract drift")
    report.check(summary_case.get("raw_file") == restore_case.get("raw_file"), f"{path}.raw_file restore drift")
    report.check(is_sha256(summary_case.get("raw_sha256")), f"{path}.raw_sha256 invalid")
    report.check(summary_case.get("raw_sha256") == restore_case.get("raw_sha256"), f"{path}.raw_sha256 restore drift")
    validate_text_key(report, path, case_id, summary_case.get("text_key"))
    rust = require_dict(report, summary_case.get("rust"), f"{path}.rust")
    report.check(rust.get("id") == case_id, f"{path}.rust.id drift")
    report.check(rust.get("payload_path") == contract_case.get("raw_file"), f"{path}.rust.payload_path drift")
    text_path = TEXT_KEYS[case_id]["path"]
    report.check(rust.get("text_path") == text_path, f"{path}.rust.text_path drift")
    validate_parsed(report, path, rust.get("parsed"), contract_case, restore_case)
    validate_text(report, path, rust.get("text"), summary_case.get("text_key"))
    validate_decisions(report, path, rust.get("decisions"), contract_case)
    validate_kvc(report, path, rust.get("kvc"), contract_case, restore_case, summary_case.get("text_key"))


def validate_text_key(report: Report, path: str, case_id: str, obj: Any) -> None:
    text_key = require_dict(report, obj, f"{path}.text_key")
    expected_path = TEXT_KEYS[case_id]["path"]
    local_path = ROOT / expected_path
    report.check(text_key.get("path") == expected_path, f"{path}.text_key.path drift")
    report.check(text_key.get("source") == TEXT_KEYS[case_id]["source"], f"{path}.text_key.source drift")
    report.check(text_key.get("sha1") == sha1_file(local_path), f"{path}.text_key.sha1 drift")
    report.check(text_key.get("sha256") == sha256_file(local_path), f"{path}.text_key.sha256 drift")
    report.check(text_key.get("bytes") == file_size(local_path), f"{path}.text_key.bytes drift")


def validate_parsed(
    report: Report,
    path: str,
    obj: Any,
    contract_case: dict[str, Any],
    restore_case: dict[str, Any],
) -> None:
    parsed = require_dict(report, obj, f"{path}.parsed")
    restore_parsed = require_dict(
        report,
        require_dict(report, restore_case.get("rust"), f"{path}.restore.rust").get("parsed"),
        f"{path}.restore.parsed",
    )
    for field in (
        "token_count",
        "raw_first_pos",
        "raw_last_pos",
        "raw_first_phys",
        "raw_last_phys",
        "payload_bytes",
        "ratio4_rows",
        "ratio128_rows",
        "layer2_n_index_comp",
        "section_bytes",
    ):
        compare_value(report, restore_parsed.get(field), parsed.get(field), f"{path}.parsed.{field}")
    report.check(parsed.get("token_count") == contract_case.get("restored_tokens"), f"{path}.restored token drift")
    report.check(parsed.get("payload_bytes") == contract_case.get("payload_bytes"), f"{path}.payload byte drift")


def validate_text(report: Report, path: str, obj: Any, text_key_obj: Any) -> None:
    text = require_dict(report, obj, f"{path}.text")
    text_key = require_dict(report, text_key_obj, f"{path}.text_key")
    report.check(text.get("bytes") == text_key.get("bytes"), f"{path}.text.bytes drift")
    report.check(text.get("sha1") == text_key.get("sha1"), f"{path}.text.sha1 drift")
    report.check(is_fnv(text.get("fnv1a64")), f"{path}.text.fnv invalid")


def validate_decisions(report: Report, path: str, obj: Any, contract_case: dict[str, Any]) -> None:
    decisions = require_dict(report, obj, f"{path}.decisions")
    report.check(decisions.get("loaded_frontier") == contract_case.get("restored_tokens"), f"{path}.loaded frontier drift")
    compare_value(report, contract_case.get("current_live_skip"), decisions.get("current_live_skip"), f"{path}.current_live_skip")
    compare_value(report, contract_case.get("next_continued_store"), decisions.get("next_continued_store"), f"{path}.next_continued_store")
    compare_value(report, contract_case.get("already_stored_boundary"), decisions.get("already_stored_boundary"), f"{path}.already_stored_boundary")
    header = expected_header(contract_case)
    compare_value(report, header, decisions.get("shutdown_write_header"), f"{path}.shutdown_write_header")


def validate_kvc(
    report: Report,
    path: str,
    obj: Any,
    contract_case: dict[str, Any],
    restore_case: dict[str, Any],
    text_key_obj: Any,
) -> None:
    kvc = require_dict(report, obj, f"{path}.kvc")
    text_key = require_dict(report, text_key_obj, f"{path}.text_key")
    expected_name = f"{text_key.get('sha1')}.kv"
    payload_bytes = contract_case.get("payload_bytes")
    text_bytes = text_key.get("bytes")
    expected_size = 48 + 4 + int(text_bytes or 0) + int(payload_bytes or 0)
    restore_rust = require_dict(report, restore_case.get("rust"), f"{path}.restore.rust")
    report.check(kvc.get("file_name") == expected_name, f"{path}.kvc.file_name drift")
    report.check(kvc.get("file_size") == expected_size, f"{path}.kvc.file_size drift")
    report.check(is_fnv(kvc.get("file_fnv1a64")), f"{path}.kvc.file_fnv invalid")
    compare_value(report, expected_header(contract_case), kvc.get("header"), f"{path}.kvc.header")
    report.check(kvc.get("text_bytes") == text_bytes, f"{path}.kvc.text_bytes drift")
    report.check(kvc.get("text_sha1") == text_key.get("sha1"), f"{path}.kvc.text_sha1 drift")
    report.check(kvc.get("payload_bytes") == payload_bytes, f"{path}.kvc.payload_bytes drift")
    report.check(kvc.get("payload_fnv1a64") == restore_rust.get("file_fnv1a64"), f"{path}.payload fnv restore drift")
    report.check(kvc.get("trailer_bytes") == 0, f"{path}.trailer byte drift")
    readback = require_dict(report, kvc.get("readback"), f"{path}.kvc.readback")
    report.check(readback.get("file_size") == kvc.get("file_size"), f"{path}.readback file size drift")
    compare_value(report, kvc.get("header"), readback.get("header"), f"{path}.readback.header")
    report.check(readback.get("text_bytes") == kvc.get("text_bytes"), f"{path}.readback text bytes drift")
    report.check(readback.get("text_sha1") == kvc.get("text_sha1"), f"{path}.readback text sha drift")
    report.check(readback.get("payload_bytes") == kvc.get("payload_bytes"), f"{path}.readback payload bytes drift")
    report.check(readback.get("payload_fnv1a64") == kvc.get("payload_fnv1a64"), f"{path}.readback payload fnv drift")
    report.check(readback.get("trailer_bytes") == 0, f"{path}.readback trailer drift")


def expected_header(contract_case: dict[str, Any]) -> dict[str, Any]:
    header = contract_case.get("shutdown_write_header", {})
    return {
        "quant_bits": header.get("quant_bits"),
        "reason_name": header.get("reason_name"),
        "reason": header.get("reason"),
        "ext_flags": header.get("ext_flags"),
        "tokens": header.get("tokens"),
        "hits": header.get("hits"),
        "ctx_size": header.get("ctx_size"),
        "created_at": 0,
        "last_used": 0,
        "payload_bytes": header.get("payload_bytes"),
    }


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "readme": ROOT / "ds4-parity/README.md",
        "report": ROOT / "ds4-parity/run_parity_report.py",
        "cargo": ROOT / "rust/ds4-gguf/Cargo.toml",
        "rust_bin": ROOT / "rust/ds4-gguf/src/bin/ds4-post-restore-kvc-smoke.rs",
    }
    texts = {name: path.read_text() for name, path in files.items()}
    required = {
        "roadmap": "M10.7d3c2: B300 Restored Payload KVC File Smoke",
        "todo": "M10.7d3c2: B300 Restored Payload KVC File Smoke",
        "status": "M10.7d3c2 B300 Restored Payload KVC File Smoke",
        "readme": "compare_post_restore_kvc_smoke.py",
        "report": "M10.7d3c2 Rust post-restore KVC file smoke comparator",
        "cargo": "ds4-post-restore-kvc-smoke",
        "rust_bin": RUST_SCHEMA,
    }
    for name, snippet in required.items():
        report.check(snippet in texts[name], f"static_checks.{name}: missing {snippet!r}")


def run_negative_tests(summary: dict[str, Any], contract: dict[str, Any], restore_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def expect_failure(label: str, mutator: Callable[[dict[str, Any]], None]) -> None:
        bad = copy.deepcopy(summary)
        mutator(bad)
        if validate_summary(bad, contract, restore_summary).ok:
            errors.append(f"negative test did not fail: {label}")

    expect_failure("raw sha drift", lambda s: s["cases"][0].__setitem__("raw_sha256", "0" * 64))
    expect_failure("text sha drift", lambda s: s["cases"][1]["text_key"].__setitem__("sha1", "0" * 40))
    expect_failure("parsed token drift", lambda s: s["cases"][2]["rust"]["parsed"].__setitem__("token_count", 560))
    expect_failure("continued target drift", lambda s: s["cases"][0]["rust"]["decisions"]["next_continued_store"].__setitem__("target", 0))
    expect_failure("shutdown reason drift", lambda s: s["cases"][3]["rust"]["kvc"]["header"].__setitem__("reason", 1))
    expect_failure("file size drift", lambda s: s["cases"][2]["rust"]["kvc"].__setitem__("file_size", 1))
    expect_failure("payload fnv drift", lambda s: s["cases"][1]["rust"]["kvc"]["readback"].__setitem__("payload_fnv1a64", "0" * 16))
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--restore-summary", type=Path, default=RESTORE_SUMMARY)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ds4-m107d3c2-kvc"))
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        contract = load_json(args.contract)
        restore_summary = load_json(args.restore_summary)
        if args.live:
            summary = build_live_summary(args.workdir, args.output_dir)
            if args.write_summary:
                write_json(args.write_summary, summary)
        else:
            summary = load_json(args.summary)
    except (OSError, TypeError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = validate_summary(summary, contract, restore_summary)
    if not report.ok:
        for error in report.errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.negative_test:
        errors = run_negative_tests(summary, contract, restore_summary)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
    print(f"post-restore KVC smoke comparator: PASS, {len(EXPECTED_CASES)} cases, {report.checks} checks")
    if args.negative_test:
        print("post-restore KVC smoke negative tests: PASS, 7 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
