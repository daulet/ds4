#!/usr/bin/env python3
"""Validate the M7.5 session payload shape oracle."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.5" / "current-c.json"
M05 = ROOT / "ds4-parity" / "baselines" / "kv-artifacts" / "m0.5"
KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
KUBE_CONTEXT = "hou2-prod1"
KUBE_NAMESPACE = "default"
KUBE_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
B300_MODEL = "/workspace/ds4/ds4flash.gguf"
B300_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
KVC_FIXED_HEADER_BYTES = 48
KVC_TEXT_LEN_BYTES = 4


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def run_c_dump(root: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [str(root / "ds4-session-payload-dump")],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        raise RuntimeError("ds4-session-payload-dump failed")
    return json.loads(proc.stdout)


def parse_key_value_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key] = value
    return out


def parse_sha_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            out[Path(parts[1]).name] = parts[0]
    return out


def parse_artifact_sha(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            prefix = "ds4-parity/baselines/"
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
            out[rel] = parts[0]
    return out


def load_m05_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    m05 = root / "ds4-parity" / "baselines" / "kv-artifacts" / "m0.5"
    logs = m05 / "logs"
    raw_hashes = parse_sha_file(logs / "kv-file-sha256.txt")
    normalized_hashes = parse_sha_file(logs / "kv-file-normalized-sha256.txt")
    artifact_hashes = parse_artifact_sha(logs / "artifact-sha256.txt")
    capture_env = parse_key_value_file(logs / "capture-env.txt")

    records: list[dict[str, Any]] = []
    with (logs / "kv-header.tsv").open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            file_name = row["file"]
            rendered_text_rel = f"kv-artifacts/m0.5/rendered-text/{Path(file_name).stem}.txt"
            payload_bytes = int(row["payload_bytes"])
            rendered_text_bytes = int(row["rendered_text_bytes"])
            trailer_bytes = int(row["trailer_bytes"])
            size_bytes = int(row["size_bytes"])
            expected_size = (
                KVC_FIXED_HEADER_BYTES
                + KVC_TEXT_LEN_BYTES
                + rendered_text_bytes
                + payload_bytes
                + trailer_bytes
            )
            records.append(
                {
                    "file": file_name,
                    "raw_kv_committed": False,
                    "raw_kv_path": f"ds4-parity/baselines/kv-artifacts/m0.5/kv/{file_name}",
                    "raw_kv_sha256": raw_hashes[file_name],
                    "timestamp_normalized_sha256": normalized_hashes[file_name],
                    "rendered_text_sha256": row["rendered_text_sha256"],
                    "rendered_text_artifact_sha256": artifact_hashes[rendered_text_rel],
                    "reason": int(row["reason"]),
                    "reason_name": row["reason_name"],
                    "quant": int(row["quant"]),
                    "ext_flags": int(row["ext_flags"]),
                    "tokens": int(row["tokens"]),
                    "hits": int(row["hits"]),
                    "ctx": int(row["ctx"]),
                    "payload_bytes": payload_bytes,
                    "rendered_text_bytes": rendered_text_bytes,
                    "trailer_bytes": trailer_bytes,
                    "size_bytes": size_bytes,
                    "expected_size_bytes": expected_size,
                    "size_matches_payload": expected_size == size_bytes,
                    "hash_policy": "raw KV files exceed 1 MiB and are represented by full and timestamp-normalized SHA256 records",
                }
            )

    evidence = {
        "capture_env": {
            "source_commit": capture_env["source_commit"],
            "model_path": capture_env["model_path"],
            "resolved_model_path": capture_env["resolved_model_path"],
            "model_sha256": capture_env["model_sha256"],
            "model_size_bytes": int(capture_env["model_size_bytes"]),
            "ds4_server_sha256": capture_env["ds4_server_sha256"],
            "gpu": capture_env["gpu"],
            "server_command": capture_env["server_command"],
        },
        "artifact_hashes": {
            "kv_header_tsv": artifact_hashes["kv-artifacts/m0.5/logs/kv-header.tsv"],
            "kv_file_sha256_log": artifact_hashes["kv-artifacts/m0.5/logs/kv-file-sha256.txt"],
            "kv_file_normalized_sha256_log": artifact_hashes["kv-artifacts/m0.5/logs/kv-file-normalized-sha256.txt"],
            "capture_env_log": artifact_hashes["kv-artifacts/m0.5/logs/capture-env.txt"],
            "replay_log": artifact_hashes["kv-artifacts/m0.5/logs/replay.log"],
            "cache_decisions_log": artifact_hashes["kv-artifacts/m0.5/logs/cache-decisions.txt"],
        },
        "artifact_hash_check": {
            "kv_header_tsv": sha256_file(logs / "kv-header.tsv"),
            "kv_file_sha256_log": sha256_file(logs / "kv-file-sha256.txt"),
            "kv_file_normalized_sha256_log": sha256_file(logs / "kv-file-normalized-sha256.txt"),
            "capture_env_log": sha256_file(logs / "capture-env.txt"),
            "replay_log": sha256_file(logs / "replay.log"),
            "cache_decisions_log": sha256_file(logs / "cache-decisions.txt"),
        },
    }
    return records, evidence


def kubectl_prefix() -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        KUBECONFIG,
        "--context",
        KUBE_CONTEXT,
        "-n",
        KUBE_NAMESPACE,
    ]


def b300_exec(script: str) -> str:
    return shell_join(
        kubectl_prefix()
        + [
            "exec",
            KUBE_POD,
            "--",
            "sh",
            "-lc",
            f"set -e; cd {B300_WORKDIR}; {script}",
        ]
    )


def b300_refresh_commands() -> list[dict[str, str]]:
    source_refresh = (
        "git archive HEAD | "
        + shell_join(
            kubectl_prefix()
            + ["exec", "-i", KUBE_POD, "--", "tar", "-xf", "-", "-C", B300_WORKDIR]
        )
    )
    copy_fixtures = shell_join(
        kubectl_prefix()
        + [
            "cp",
            "ds4-parity/baselines/kv-fixtures/m0.5",
            f"{KUBE_POD}:{B300_WORKDIR}/ds4-parity/baselines/kv-fixtures/m0.5",
        ]
    )
    server_template = (
        "./ds4-server -m /workspace/ds4/ds4flash.gguf --cuda --ctx 32768 "
        "--tokens 16 --host 127.0.0.1 --port ${PORT} "
        "--trace ds4-parity/baselines/kv-artifacts/m0.5/traces/${SERVER}.trace "
        "--kv-disk-dir ds4-parity/baselines/kv-artifacts/m0.5/kv "
        "--kv-disk-space-mb 512 --kv-cache-min-tokens 512 "
        "--kv-cache-cold-max-tokens 30000 --kv-cache-continued-interval-tokens 0"
    )
    curl_seed = (
        "curl -sS -o ds4-parity/baselines/kv-artifacts/m0.5/responses/${RESPONSE}.json "
        "-w '%{http_code}' http://127.0.0.1:${PORT}/v1/chat/completions "
        "-H 'Content-Type: application/json' "
        "--data-binary @ds4-parity/baselines/kv-fixtures/m0.5/${FIXTURE}.json"
    )
    return [
        {"name": "refresh_source", "command": source_refresh},
        {"name": "copy_fixtures", "command": copy_fixtures},
        {
            "name": "build_server",
            "command": b300_exec("make clean ds4-server && sha256sum ds4-server && file ds4-server"),
        },
        {
            "name": "server_a_seed_miss",
            "command": b300_exec(
                "PORT=18081 SERVER=server-a RESPONSE=seed_miss FIXTURE=kv_seed; "
                f"{server_template} >/tmp/ds4-m05-${{SERVER}}.log 2>&1 & "
                "pid=$!; sleep 5; "
                f"{curl_seed}; "
                "kill $pid; wait $pid || true"
            ),
        },
        {
            "name": "server_b_seed_restore",
            "command": b300_exec(
                "PORT=18082 SERVER=server-b RESPONSE=seed_restore FIXTURE=kv_seed; "
                f"{server_template} >/tmp/ds4-m05-${{SERVER}}.log 2>&1 & "
                "pid=$!; sleep 5; "
                f"{curl_seed}; "
                "kill $pid; wait $pid || true"
            ),
        },
        {
            "name": "server_c_continuation_restore",
            "command": b300_exec(
                "PORT=18083 SERVER=server-c RESPONSE=continuation_restore FIXTURE=kv_continuation; "
                f"{server_template} >/tmp/ds4-m05-${{SERVER}}.log 2>&1 & "
                "pid=$!; sleep 5; "
                f"{curl_seed}; "
                "kill $pid; wait $pid || true"
            ),
        },
        {
            "name": "parse_and_hash_artifacts",
            "command": b300_exec(
                "python3 -m json.tool ds4-parity/baselines/kv-artifacts/m0.5/responses/seed_miss.json >/dev/null && "
                "python3 -m json.tool ds4-parity/baselines/kv-artifacts/m0.5/responses/seed_restore.json >/dev/null && "
                "python3 -m json.tool ds4-parity/baselines/kv-artifacts/m0.5/responses/continuation_restore.json >/dev/null && "
                "sha256sum ds4-parity/baselines/kv-artifacts/m0.5/kv/*.kv"
            ),
        },
    ]


def build_current(root: Path) -> dict[str, Any]:
    structural = run_c_dump(root)
    records, evidence = load_m05_records(root)
    return {
        "schema": "ds4.session_payload_shape_oracle.v1",
        "source": "current-c-session-payload-no-model-plus-m0.5-records",
        "structural": structural,
        "m0_5_payload_records": records,
        "m0_5_evidence": evidence,
        "b300_refresh": {
            "kubeconfig": KUBECONFIG,
            "context": KUBE_CONTEXT,
            "namespace": KUBE_NAMESPACE,
            "pod": KUBE_POD,
            "workdir": B300_WORKDIR,
            "model_path": B300_MODEL,
            "model_sha256": B300_MODEL_SHA256,
            "commands": b300_refresh_commands(),
        },
    }


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


def structural_invariants(report: Report, baseline: dict[str, Any]) -> None:
    report.check(baseline.get("schema") == "ds4.session_payload_shape_oracle.v1", "baseline schema mismatch")
    structural = require_dict(report, baseline.get("structural"), "structural")
    constants = require_dict(report, structural.get("constants"), "structural.constants")
    report.check(constants.get("magic_bytes_hex") == "44535634", "DSV4 little-endian bytes drift")
    report.check(constants.get("u32_fields") == 13, "payload header field count drift")
    report.check(constants.get("header_bytes") == 52, "payload header byte size drift")
    layout = require_dict(report, structural.get("fixed_model_layout"), "structural.fixed_model_layout")
    report.check(layout.get("n_layer") == 43, "fixed layer count drift")
    report.check(layout.get("n_head_dim") == 512, "fixed head dim drift")
    report.check(layout.get("n_indexer_head_dim") == 128, "fixed indexer dim drift")
    report.check(layout.get("n_vocab") == 129280, "fixed vocab drift")
    ratios = structural.get("compress_ratio_by_layer")
    report.check(isinstance(ratios, list) and len(ratios) == 43, "compress ratio coverage drift")

    size_case = require_dict(report, structural.get("size_case"), "structural.size_case")
    sections = require_dict(report, size_case.get("section_bytes"), "structural.size_case.section_bytes")
    total = sum(int(sections.get(name, 0)) for name in sections)
    report.check(total == size_case.get("payload_bytes"), "section byte total does not match payload_bytes")

    body_codes = {case["name"]: case["code"] for case in structural.get("body_probe_cases", [])}
    report.check(body_codes.get("valid_cpu_payload") == "ok", "valid CPU payload no longer loads")
    report.check(body_codes.get("trailing_payload_bytes") == "trailing-payload-bytes", "trailing payload rejection drift")
    report.check(body_codes.get("n_comp_over_cap") == "invalid-compressed-row-count", "n_comp rejection drift")
    report.check(body_codes.get("n_index_comp_over_cap") == "invalid-indexer-row-count", "n_index rejection drift")


def m05_invariants(report: Report, baseline: dict[str, Any]) -> None:
    records = baseline.get("m0_5_payload_records", [])
    report.check(isinstance(records, list) and len(records) == 3, "M0.5 payload record coverage drift")
    for record in records if isinstance(records, list) else []:
        path = f"m0_5_payload_records[{record.get('file', '?')}]"
        report.check(record.get("raw_kv_committed") is False, f"{path}: raw KV file should remain hash-only")
        report.check(record.get("payload_bytes", 0) > 1_000_000, f"{path}: expected hash-only payload size")
        report.check(record.get("size_matches_payload") is True, f"{path}: KVC size does not match payload/text/trailer")
        expected = (
            KVC_FIXED_HEADER_BYTES
            + KVC_TEXT_LEN_BYTES
            + record.get("rendered_text_bytes", 0)
            + record.get("payload_bytes", 0)
            + record.get("trailer_bytes", 0)
        )
        report.check(expected == record.get("size_bytes"), f"{path}: size formula drift")

    evidence = require_dict(report, baseline.get("m0_5_evidence"), "m0_5_evidence")
    capture = require_dict(report, evidence.get("capture_env"), "m0_5_evidence.capture_env")
    report.check(capture.get("model_sha256") == B300_MODEL_SHA256, "M0.5 model hash drift")
    hashes = require_dict(report, evidence.get("artifact_hashes"), "m0_5_evidence.artifact_hashes")
    checks = require_dict(report, evidence.get("artifact_hash_check"), "m0_5_evidence.artifact_hash_check")
    for key, value in hashes.items():
        report.check(checks.get(key) == value, f"M0.5 artifact hash mismatch for {key}")


def b300_invariants(report: Report, baseline: dict[str, Any]) -> None:
    refresh = require_dict(report, baseline.get("b300_refresh"), "b300_refresh")
    report.check(refresh.get("kubeconfig") == KUBECONFIG, "B300 kubeconfig drift")
    report.check(refresh.get("context") == KUBE_CONTEXT, "B300 context drift")
    report.check(refresh.get("namespace") == KUBE_NAMESPACE, "B300 namespace drift")
    report.check(refresh.get("pod") == KUBE_POD, "B300 pod drift")
    report.check(refresh.get("model_path") == B300_MODEL, "B300 model path drift")
    report.check(refresh.get("model_sha256") == B300_MODEL_SHA256, "B300 model hash drift")
    commands = refresh.get("commands", [])
    report.check(isinstance(commands, list) and len(commands) >= 7, "B300 refresh command coverage drift")
    for item in commands if isinstance(commands, list) else []:
        command = item.get("command", "")
        report.check(KUBECONFIG in command, f"{item.get('name')}: missing temp kubeconfig")
        report.check(f"--context {KUBE_CONTEXT}" in command, f"{item.get('name')}: missing explicit context")
        report.check(KUBE_POD in command, f"{item.get('name')}: missing pod")
    all_commands = "\n".join(item.get("command", "") for item in commands if isinstance(item, dict))
    report.check("FIXTURE=kv_seed" in all_commands, "B300 refresh missing seed fixture")
    report.check("FIXTURE=kv_continuation" in all_commands, "B300 refresh missing continuation fixture")
    report.check(B300_MODEL in all_commands, "B300 refresh missing model path")


def validate_candidate(expected: dict[str, Any], got: dict[str, Any]) -> Report:
    report = Report()
    compare_value(report, expected, got, "baseline")
    structural_invariants(report, got)
    m05_invariants(report, got)
    b300_invariants(report, got)
    return report


def run_negative_tests(expected: dict[str, Any]) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("payload version drift", ["structural", "constants", "version"], 2),
        ("header case drift", ["structural", "header_rejection_cases", 1, "code"], "ok"),
        ("body case drift", ["structural", "body_probe_cases", 1, "code"], "ok"),
        ("payload byte drift", ["m0_5_payload_records", 0, "payload_bytes"], expected["m0_5_payload_records"][0]["payload_bytes"] + 1),
        ("raw KV committed drift", ["m0_5_payload_records", 0, "raw_kv_committed"], True),
        ("artifact hash drift", ["m0_5_evidence", "artifact_hashes", "kv_header_tsv"], "0" * 64),
        ("B300 context drift", ["b300_refresh", "context"], "default"),
        ("B300 command drift", ["b300_refresh", "commands", 0, "command"], "kubectl get pods"),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(expected)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = validate_candidate(expected, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--write-baseline", type=Path)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    current = build_current(root)
    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(current, indent=2) + "\n")
        return 0

    expected = load_json(args.baseline)
    report = validate_candidate(current, expected)
    print_report("Session payload shape oracle", report)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(current)
        print_report("Session payload shape negative tests", negative)

    return 0 if report.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
