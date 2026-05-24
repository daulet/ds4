#!/usr/bin/env python3
"""Compare Rust graph restore next-token state against a same-capture C oracle."""

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
import compare_graph_restore_readback as readback_cmp
import check_graph_restore_frontier_contract as frontier_cmp


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.8" / "current-c.json"
DISK_SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c2" / "rust-b300-raw-import.json"
SNAPSHOT_SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c3a" / "rust-b300-snapshot-raw-import.json"
READBACK_SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c3c" / "rust-b300-restore-readback.json"
SUMMARY = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7c3d" / "rust-b300-restore-next-token.json"
FRONTIER_CONTRACT = ROOT / "ds4-parity" / "baselines" / "kv" / "m10.7d3" / "restore-frontier-contract.json"
SCHEMA = "ds4.rust_graph_restore_next_token_summary.v1"
RUST_SCHEMA = "ds4.rust_graph_restore_next_token.v1"
RUST_READBACK_SCHEMA = "ds4.rust_graph_restore_readback.v1"
EXPECTED_CASES = [
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
]
MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
DEFAULT_B300_MODEL = Path("/workspace/ds4/ds4flash.gguf")
RAW_DIR = Path("ds4-parity/baselines/kv/m7.8/raw")
SEED_PROMPT = Path("ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_prompt.txt")
SEED_ASSISTANT = Path("ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_assistant.txt")
CONTINUATION_USER = Path("ds4-parity/baselines/kv-fixtures/m7.8/restore_continuation_user.txt")


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
    return readback_cmp.restore_oracle_cases(report, oracle)


def run_current_c_restore(
    workdir: Path,
    output: Path,
    model: Path,
) -> dict[str, Any]:
    raw_dir = workdir / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    make = subprocess.run(
        ["make", "ds4-restore-dump", "CUDA_ARCH=native"],
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if make.returncode != 0:
        raise RuntimeError(make.stderr.strip() or make.stdout.strip())
    command = [
        "./ds4-restore-dump",
        "--backend",
        "cuda",
        "-m",
        str(model),
        "--model-sha256",
        MODEL_SHA256,
        "--seed-prompt",
        str(SEED_PROMPT),
        "--seed-assistant",
        str(SEED_ASSISTANT),
        "--continuation-user",
        str(CONTINUATION_USER),
        "--payload-dir",
        str(RAW_DIR),
        "--snapshot-dir",
        str(RAW_DIR),
        "-o",
        str(output),
    ]
    proc = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return load_json(output if output.is_absolute() else workdir / output)


def run_rust_next_token(oracle_cases: list[dict[str, Any]], workdir: Path) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ds4-gpu",
        "--features",
        "cuda-backend",
        "--bin",
        "ds4-graph-restore-next-token",
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
        raise TypeError("Rust restore next-token smoke did not return an object")
    return data


def build_live_summary(
    shape_oracle: dict[str, Any],
    workdir: Path,
    current_c_output: Path,
    model: Path,
) -> dict[str, Any]:
    current_c = run_current_c_restore(workdir, current_c_output, model)
    report = Report()
    shape_cases = restore_oracle_cases(report, shape_oracle)
    current_cases = restore_oracle_cases(report, current_c)
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    rust = run_rust_next_token(current_cases, workdir)
    rust_readback = readback_cmp.run_rust_readback(current_cases, workdir)
    rust_cases = cases_by_id(Report(), rust.get("cases"), "rust")
    readback_cases = cases_by_id(Report(), rust_readback.get("cases"), "rust_readback")
    shape_by_id = {str(case["id"]): case for case in shape_cases}
    current_by_id = {str(case["id"]): case for case in current_cases}
    cases = []
    for case_id in EXPECTED_CASES:
        shape_case = shape_by_id[case_id]
        current_case = current_by_id[case_id]
        raw_path = workdir / str(current_case["raw_file"])
        raw_sha256 = raw_import.sha256_file(raw_path)
        cases.append(
            {
                "id": case_id,
                "kind": shape_case["kind"],
                "prompt_case": shape_case["prompt_case"],
                "raw_file": current_case["raw_file"],
                "raw_bytes": current_case["raw_bytes"],
                "raw_sha256": raw_sha256,
                "current_c_raw_sha256": current_case["oracle_raw_sha256"],
                "m7_8_oracle_raw_sha256": shape_case["oracle_raw_sha256"],
                "raw_sha256_matches_current_c": raw_sha256 == current_case["oracle_raw_sha256"],
                "raw_sha256_matches_m7_8_oracle": raw_sha256 == shape_case["oracle_raw_sha256"],
                "current_c": {
                    "selected_token": current_case["restored"]["selected_token"],
                    "selected_bytes_hex": current_case["restored"]["selected_bytes_hex"],
                    "top_logprobs": current_case["restored"]["top_logprobs"],
                },
                "rust_readback": readback_cases.get(case_id, {}),
                "rust": rust_cases.get(case_id, {}),
            }
        )
    return {
        "schema": SCHEMA,
        "source": "rust-b300-graph-restore-next-token",
        "shape_oracle": "ds4-parity/baselines/kv/m7.8/current-c.json",
        "readback_oracle": "same-capture rust ds4-graph-restore-readback plus committed M10.7c3c validation",
        "raw_body_policy": "hash-only; raw disk payload and memory snapshot bodies remain on B300 and are not committed; top-logprob scores compare against the same-capture current-C restore oracle",
        "b300": {
            "kube_context": "hou2-prod1",
            "pod": "ds4-rust-port-b300",
            "workdir": "/workspace/ds4",
        },
        "current_c_restore": {
            "schema": current_c.get("schema"),
            "source": current_c.get("source"),
            "backend": current_c.get("backend"),
            "top_k": current_c.get("top_k"),
            "score_abs_tolerance": current_c.get("score_abs_tolerance"),
            "model_sha256": current_c.get("model_sha256"),
            "capture_output": str(current_c_output),
        },
        "rust_next_token": {
            "schema": rust.get("schema"),
            "source": rust.get("source"),
            "runtime": rust.get("runtime"),
            "top_k": rust.get("top_k"),
        },
        "rust_readback": {
            "schema": rust_readback.get("schema"),
            "source": rust_readback.get("source"),
            "runtime": rust_readback.get("runtime"),
        },
        "cases": cases,
    }


def validate_summary(
    shape_oracle: dict[str, Any],
    summary: dict[str, Any],
    committed_readback_summary: dict[str, Any],
    disk_summary: dict[str, Any],
    snapshot_summary: dict[str, Any],
    frontier_contract: dict[str, Any],
) -> Report:
    report = Report()
    shape_cases = restore_oracle_cases(report, shape_oracle)
    committed_readback_report = readback_cmp.validate_summary(
        shape_oracle,
        committed_readback_summary,
        disk_summary,
        snapshot_summary,
    )
    report.checks += committed_readback_report.checks
    report.errors.extend(f"M10.7c3c committed readback evidence: {err}" for err in committed_readback_report.errors)

    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == "rust-b300-graph-restore-next-token", "summary source drift")
    report.check(
        summary.get("shape_oracle") == "ds4-parity/baselines/kv/m7.8/current-c.json",
        "summary shape oracle path drift",
    )
    report.check("same-capture" in str(summary.get("readback_oracle")), "summary readback oracle drift")
    report.check("same-capture current-C restore oracle" in str(summary.get("raw_body_policy")), "raw-body policy drift")
    b300 = require_dict(report, summary.get("b300"), "summary.b300")
    report.check(b300.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(b300.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(b300.get("workdir") == "/workspace/ds4", "B300 workdir drift")

    current_c_meta = require_dict(report, summary.get("current_c_restore"), "summary.current_c_restore")
    report.check(current_c_meta.get("schema") == "ds4.restore_oracle.v1", "current-C schema drift")
    report.check(current_c_meta.get("source") == "current-c-b300-restore", "current-C source drift")
    report.check(current_c_meta.get("backend") == "cuda", "current-C backend drift")
    report.check(current_c_meta.get("top_k") == 20, "current-C top_k drift")
    report.check(current_c_meta.get("score_abs_tolerance") == 1e-5, "current-C tolerance drift")
    report.check(current_c_meta.get("model_sha256") == MODEL_SHA256, "current-C model sha drift")

    rust_meta = require_dict(report, summary.get("rust_next_token"), "summary.rust_next_token")
    report.check(rust_meta.get("schema") == RUST_SCHEMA, "Rust next-token schema drift")
    report.check(rust_meta.get("source") == "rust-graph-restore-next-token", "Rust next-token source drift")
    report.check(rust_meta.get("top_k") == int(current_c_meta.get("top_k", 20)), "Rust next-token top_k drift")
    validate_runtime(report, rust_meta.get("runtime"), "summary.rust_next_token.runtime")

    readback_meta = require_dict(report, summary.get("rust_readback"), "summary.rust_readback")
    report.check(readback_meta.get("schema") == RUST_READBACK_SCHEMA, "Rust readback schema drift")
    report.check(readback_meta.get("source") == "rust-graph-restore-readback", "Rust readback source drift")
    validate_runtime(report, readback_meta.get("runtime"), "summary.rust_readback.runtime")

    summary_cases = cases_by_id(report, summary.get("cases"), "summary")
    frontier_cases = frontier_contract_cases(report, frontier_contract)
    policy_options = require_dict(report, frontier_contract.get("policy_options"), "frontier_contract.policy_options")
    already_stored_probe = frontier_policy_probe(report, frontier_contract, "already_stored_boundary_skips")
    shape_by_id = {str(case["id"]): case for case in shape_cases}
    tolerance = float(current_c_meta.get("score_abs_tolerance", 1e-5))
    for case_id in EXPECTED_CASES:
        shape_case = shape_by_id.get(case_id, {})
        path = f"case.{case_id}"
        summary_case = summary_cases.get(case_id, {})
        report.check(summary_case.get("kind") == shape_case.get("kind"), f"{path}.kind drift")
        report.check(summary_case.get("prompt_case") == shape_case.get("prompt_case"), f"{path}.prompt_case drift")
        report.check(summary_case.get("raw_file") == shape_case.get("raw_file"), f"{path}.raw_file drift")
        report.check(summary_case.get("raw_bytes") == shape_case.get("raw_bytes"), f"{path}.raw_bytes drift")
        report.check(raw_import.is_sha256_hex(summary_case.get("raw_sha256")), f"{path}.raw_sha256 invalid")
        report.check(summary_case.get("raw_sha256") == summary_case.get("current_c_raw_sha256"), f"{path}.raw/current-C sha drift")
        report.check(
            summary_case.get("m7_8_oracle_raw_sha256") == shape_case.get("oracle_raw_sha256"),
            f"{path}.m7.8 raw sha drift",
        )
        report.check(summary_case.get("raw_sha256_matches_current_c") is True, f"{path}.current-C raw sha flag drift")
        report.check(isinstance(summary_case.get("raw_sha256_matches_m7_8_oracle"), bool), f"{path}.m7.8 sha flag invalid")
        current_c = require_dict(report, summary_case.get("current_c"), f"{path}.current_c")
        rust_readback = require_dict(report, summary_case.get("rust_readback"), f"{path}.rust_readback")
        rust = require_dict(report, summary_case.get("rust"), f"{path}.rust")
        validate_rust_readback_case(report, shape_case, rust_readback, rust, path)
        validate_rust_next_token_case(report, shape_case, current_c, rust, rust_readback, tolerance, path)
        validate_frontier_projection(
            report,
            frontier_cases.get(case_id, {}),
            policy_options,
            already_stored_probe,
            rust,
            path,
        )
    static_checks(report)
    return report


def validate_runtime(report: Report, runtime_obj: Any, path: str) -> None:
    runtime = require_dict(report, runtime_obj, path)
    report.check(runtime.get("ctx") == 32768, f"{path}.ctx drift")
    report.check(runtime.get("kind") == "default-graph", f"{path}.kind drift")
    report.check(runtime.get("backend") == "ds4-gpu", f"{path}.backend drift")


def validate_rust_readback_case(
    report: Report,
    shape_case: dict[str, Any],
    rust_readback: dict[str, Any],
    rust_next: dict[str, Any],
    path: str,
) -> None:
    report.check(rust_readback.get("id") == shape_case.get("id"), f"{path}.readback.id drift")
    report.check(rust_readback.get("path") == shape_case.get("raw_file"), f"{path}.readback.path drift")
    report.check(rust_readback.get("payload_bytes") == shape_case.get("raw_bytes"), f"{path}.readback.payload_bytes drift")
    report.check(rust_readback.get("ok") is True, f"{path}.readback.ok drift")
    report.check(rust_readback.get("error") == "", f"{path}.readback.error drift")
    report.check(rust_readback.get("file_fnv1a64") == rust_next.get("file_fnv1a64"), f"{path}.readback file fnv drift")
    compare_value(report, raw_import.expected_parsed(shape_case), rust_readback.get("parsed"), f"{path}.readback.parsed")
    readback_cmp.validate_readback(report, shape_case, rust_readback.get("readback"), f"{path}.readback")


def validate_rust_next_token_case(
    report: Report,
    shape_case: dict[str, Any],
    current_c: dict[str, Any],
    rust: dict[str, Any],
    rust_readback: dict[str, Any],
    tolerance: float,
    path: str,
) -> None:
    report.check(rust.get("id") == shape_case.get("id"), f"{path}.rust.id drift")
    report.check(rust.get("path") == shape_case.get("raw_file"), f"{path}.rust.path drift")
    report.check(rust.get("payload_bytes") == shape_case.get("raw_bytes"), f"{path}.rust.payload_bytes drift")
    report.check(rust.get("file_fnv1a64") == rust_readback.get("file_fnv1a64"), f"{path}.rust.file_fnv drift")
    report.check(rust.get("ok") is True, f"{path}.rust.ok drift")
    report.check(rust.get("error") == "", f"{path}.rust.error drift")
    compare_value(report, raw_import.expected_parsed(shape_case), rust.get("parsed"), f"{path}.rust.parsed")

    next_token = require_dict(report, rust.get("next_token"), f"{path}.rust.next_token")
    report.check(current_c.get("selected_token") == first_top_id(current_c), f"{path}.current-C selected/top drift")
    report.check(next_token.get("cache_source") == "restored-graph-payload", f"{path}.cache source drift")
    report.check(next_token.get("next_token_source") == "restored-session-logits", f"{path}.next-token source drift")
    report.check(next_token.get("graph_restored") is True, f"{path}.graph_restored drift")
    report.check(next_token.get("checkpoint_tokens") == shape_case.get("prompt_tokens"), f"{path}.checkpoint token count drift")
    report.check(is_fnv_hex(next_token.get("checkpoint_fnv1a64")), f"{path}.checkpoint fnv invalid")
    report.check(is_fnv_hex(next_token.get("logits_fnv1a64")), f"{path}.logits fnv invalid")
    readback = rust_readback.get("readback") if isinstance(rust_readback.get("readback"), dict) else {}
    checkpoint = readback.get("checkpoint") if isinstance(readback.get("checkpoint"), dict) else {}
    logits = readback.get("logits") if isinstance(readback.get("logits"), dict) else {}
    report.check(next_token.get("checkpoint_fnv1a64") == checkpoint.get("readback_fnv1a64"), f"{path}.checkpoint fnv readback drift")
    report.check(next_token.get("logits_fnv1a64") == logits.get("readback_fnv1a64"), f"{path}.logits fnv readback drift")
    report.check(next_token.get("selected_token") == current_c.get("selected_token"), f"{path}.selected token drift")
    validate_top_logprobs(report, current_c, next_token, tolerance, path)
    validate_post_restore_state(report, shape_case, next_token, path)


def first_top_id(obj: dict[str, Any]) -> Any:
    top = obj.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        return top[0].get("id")
    return None


def validate_top_logprobs(
    report: Report,
    current_c: dict[str, Any],
    next_token: dict[str, Any],
    tolerance: float,
    path: str,
) -> None:
    expected = require_list(report, current_c.get("top_logprobs"), f"{path}.current_c.top_logprobs")
    got = require_list(report, next_token.get("top_logprobs"), f"{path}.rust.top_logprobs")
    report.check(len(got) == len(expected), f"{path}.top-logprob count drift")
    for idx, expected_score in enumerate(expected):
        if idx >= len(got):
            continue
        got_score = require_dict(report, got[idx], f"{path}.rust.top_logprobs[{idx}]")
        expected_score = require_dict(report, expected_score, f"{path}.current_c.top_logprobs[{idx}]")
        report.check(got_score.get("id") == expected_score.get("id"), f"{path}.top[{idx}].id drift")
        for field in ("logit", "logprob"):
            got_value = got_score.get(field)
            expected_value = expected_score.get(field)
            report.check(isinstance(got_value, (int, float)), f"{path}.top[{idx}].{field} missing")
            report.check(isinstance(expected_value, (int, float)), f"{path}.current_c.top[{idx}].{field} missing")
            if isinstance(got_value, (int, float)) and isinstance(expected_value, (int, float)):
                delta = abs(float(got_value) - float(expected_value))
                report.check(delta <= tolerance, f"{path}.top[{idx}].{field} drift {delta:g} > {tolerance:g}")


def validate_post_restore_state(
    report: Report,
    shape_case: dict[str, Any],
    next_token: dict[str, Any],
    path: str,
) -> None:
    state = require_dict(report, next_token.get("post_restore_state"), f"{path}.post_restore_state")
    n_comp, n_index_comp = readback_cmp.layer_counts(int(shape_case["prompt_tokens"]))
    expected_state = {
        "checkpoint_valid": True,
        "mtp_draft_valid": False,
        "mtp_n_raw": 0,
        "layer_n_comp": n_comp,
        "layer_n_index_comp": n_index_comp,
    }
    compare_value(report, expected_state, state, f"{path}.post_restore_state")


def frontier_contract_cases(
    report: Report,
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    report.check(contract.get("schema") == frontier_cmp.SCHEMA, "frontier contract schema drift")
    report.check(contract.get("milestone") == "M10.7d3a", "frontier contract milestone drift")
    cases = require_list(report, contract.get("restore_frontier_cases"), "frontier_contract.restore_frontier_cases")
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            report.check(False, "frontier contract case must be object")
            continue
        name = case.get("name")
        report.check(isinstance(name, str), "frontier contract case name missing")
        if isinstance(name, str):
            out[name] = case
    report.check(list(out) == EXPECTED_CASES, "frontier contract case order drift")
    return out


def frontier_policy_probe(
    report: Report,
    contract: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    probes = require_list(report, contract.get("policy_probes"), "frontier_contract.policy_probes")
    for probe in probes:
        if isinstance(probe, dict) and probe.get("name") == name:
            return probe
    report.check(False, f"frontier contract policy probe missing: {name}")
    return {}


def validate_frontier_projection(
    report: Report,
    contract_case: dict[str, Any],
    policy_options: dict[str, Any],
    already_stored_probe: dict[str, Any],
    rust: dict[str, Any],
    path: str,
) -> None:
    next_token = require_dict(report, rust.get("next_token"), f"{path}.rust.next_token")
    projection = require_dict(report, next_token.get("frontier_projection"), f"{path}.frontier_projection")
    report.check(projection.get("source") == "restored-token-count", f"{path}.frontier.source drift")
    compare_value(report, policy_options, projection.get("policy"), f"{path}.frontier.policy")
    report.check(
        projection.get("loaded_frontier") == contract_case.get("loaded_frontier"),
        f"{path}.frontier.loaded_frontier drift",
    )
    compare_value(
        report,
        contract_case.get("current_live_skip"),
        projection.get("current_live_skip"),
        f"{path}.frontier.current_live_skip",
    )
    compare_value(
        report,
        contract_case.get("next_continued_store"),
        projection.get("next_continued_store"),
        f"{path}.frontier.next_continued_store",
    )
    expected_already_stored = {
        "frontier_before": already_stored_probe.get("frontier_before"),
        "live_tokens": already_stored_probe.get("live_tokens"),
        "target": already_stored_probe.get("target"),
    }
    compare_value(
        report,
        expected_already_stored,
        projection.get("already_stored_boundary"),
        f"{path}.frontier.already_stored_boundary",
    )
    compare_value(
        report,
        contract_case.get("post_restore_shutdown"),
        projection.get("post_restore_shutdown"),
        f"{path}.frontier.post_restore_shutdown",
    )


def is_fnv_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 16 and all(ch in "0123456789abcdef" for ch in value)


def static_checks(report: Report) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory" / "TODO.md",
        "status": ROOT / ".memory" / "status.md",
        "readme": ROOT / "ds4-parity" / "README.md",
        "report": ROOT / "ds4-parity" / "run_parity_report.py",
        "rust_bin": ROOT / "rust" / "ds4-gpu" / "src" / "bin" / "ds4-graph-restore-next-token.rs",
        "c_fixture": ROOT / "ds4_restore_dump.c",
    }
    text = {name: path.read_text() for name, path in files.items()}
    status_flat = " ".join(text["status"].split())
    report.check("M10.7c3d: Rust Graph Tensor Restore Next-Token Smoke" in text["roadmap"], "roadmap M10.7c3d missing")
    report.check("M10.7c3d: Rust Graph Tensor Restore Next-Token Smoke" in text["todo"], "TODO M10.7c3d missing")
    report.check("M10.7c3d Rust Graph Tensor Restore Next-Token Smoke" in status_flat, "status M10.7c3d missing")
    report.check("compare_graph_restore_next_token.py" in text["readme"], "README restore next-token command missing")
    report.check("same-capture current-C restore oracle" in text["readme"], "README same-capture policy missing")
    report.check("M10.7d3b Rust graph restore frontier projection comparator" in text["report"], "unified report M10.7d3b missing")
    report.check("M10.7d3b B300 Rust graph restore frontier projection rerun" in text["report"], "B300 restore frontier projection rerun missing")
    for snippet in (
        "restored-graph-payload",
        "restored-session-logits",
        "top_logprobs",
        "frontier_projection",
        "layer_raw_cache",
    ):
        report.check(snippet in text["rust_bin"], f"Rust next-token marker missing: {snippet}")
    for snippet in (
        "load_payload_file(restored",
        "ds4_session_load_snapshot(restored",
        "capture_state(engine, restored",
        "ds4_session_top_logprobs",
    ):
        report.check(snippet in text["c_fixture"], f"C restore fixture snippet missing: {snippet}")


def run_negative_tests(
    shape_oracle: dict[str, Any],
    summary: dict[str, Any],
    committed_readback_summary: dict[str, Any],
    disk_summary: dict[str, Any],
    snapshot_summary: dict[str, Any],
    frontier_contract: dict[str, Any],
) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("raw sha drift", ["cases", 0, "raw_sha256"], "0" * 64),
        ("current-C raw sha flag drift", ["cases", 0, "raw_sha256_matches_current_c"], False),
        ("selected token drift", ["cases", 0, "rust", "next_token", "selected_token"], -1),
        ("top order drift", ["cases", 1, "rust", "next_token", "top_logprobs", 0, "id"], -1),
        ("logprob drift", ["cases", 2, "rust", "next_token", "top_logprobs", 0, "logprob"], 0.0),
        ("cache source drift", ["cases", 0, "rust", "next_token", "cache_source"], "cold-prefill"),
        ("graph restored drift", ["cases", 1, "rust", "next_token", "graph_restored"], False),
        ("counter drift", ["cases", 3, "rust", "next_token", "post_restore_state", "layer_n_comp", 2], 0),
        ("readback evidence drift", ["cases", 0, "rust_readback", "readback", "logits", "matched"], False),
        ("runtime drift", ["rust_next_token", "runtime", "backend"], "not-ds4-gpu"),
        ("policy drift", ["raw_body_policy"], "raw bodies committed"),
        (
            "frontier projection drift",
            [
                "cases",
                0,
                "rust",
                "next_token",
                "frontier_projection",
                "next_continued_store",
                "target",
            ],
            0,
        ),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(summary)
        target: Any = bad
        try:
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
        except (KeyError, IndexError, TypeError):
            report.check(False, f"negative test mutation path missing for {label}")
            continue
        result = validate_summary(
            shape_oracle,
            bad,
            committed_readback_summary,
            disk_summary,
            snapshot_summary,
            frontier_contract,
        )
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
    parser.add_argument("--readback-summary", type=Path, default=READBACK_SUMMARY)
    parser.add_argument("--disk-summary", type=Path, default=DISK_SUMMARY)
    parser.add_argument("--snapshot-summary", type=Path, default=SNAPSHOT_SUMMARY)
    parser.add_argument("--frontier-contract", type=Path, default=FRONTIER_CONTRACT)
    parser.add_argument("--live", action="store_true", help="recapture C restore and run Rust GPU restore over the same B300 raw files")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_B300_MODEL)
    parser.add_argument("--current-c-output", type=Path, default=Path("/tmp/ds4-m107c3d-current-c-restore.json"))
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    shape_oracle = load_json(args.oracle)
    committed_readback_summary = load_json(args.readback_summary)
    disk_summary = load_json(args.disk_summary)
    snapshot_summary = load_json(args.snapshot_summary)
    frontier_contract = load_json(args.frontier_contract)
    if args.live:
        summary = build_live_summary(shape_oracle, args.workdir, args.current_c_output, args.model)
        if args.write_summary:
            write_json(args.write_summary, summary)
    else:
        summary = load_json(args.summary)

    report = validate_summary(
        shape_oracle,
        summary,
        committed_readback_summary,
        disk_summary,
        snapshot_summary,
        frontier_contract,
    )
    print_report("Graph restore next-token comparator", report)
    ok = report.ok
    if args.negative_test:
        negative = run_negative_tests(
            shape_oracle,
            summary,
            committed_readback_summary,
            disk_summary,
            snapshot_summary,
            frontier_contract,
        )
        print_report("Graph restore next-token negative tests", negative)
        ok = ok and negative.ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
