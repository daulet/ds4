#!/usr/bin/env python3
"""Validate the M10.8a model-free MTP state-machine contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json"
)
CLI_RUNTIME_CONTROLS = ROOT / "ds4-parity/baselines/cli/m8.12b/current-c.json"

SCHEMA = "ds4.mtp_state_machine_contract.v1"
EXPECTED_MILESTONE = "M10.8a"
EXPECTED_SOURCE = "current-c-model-free-mtp-state-machine-contract"
EXPECTED_B300 = {
    "context": "hou2-prod1",
    "namespace": "default",
    "pod": "ds4-rust-port-b300",
    "workdir": "/workspace/ds4",
    "base_model_path": "/workspace/ds4/ds4flash.gguf",
    "base_model_symlink_target": "/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
    "expected_mtp_path": "/workspace/ds4/missing-mtp.gguf",
    "candidate_globs": ["*mtp*.gguf", "*draft*.gguf"],
    "candidate_max_depth": 3,
    "candidate_count": 0,
    "checked_utc": "2026-05-24",
    "availability": "blocked_missing_mtp_model",
}
EXPECTED_ENV_GATES = {
    "spec_disable_callers": "DS4_MTP_SPEC_DISABLE",
    "probe": "DS4_MTP_PROBE",
    "strict": "DS4_MTP_STRICT",
    "batch_verify_override": "DS4_MTP_BATCH_VERIFY",
    "capture_prefix1": "DS4_MTP_CAPTURE_PREFIX1",
    "exact_replay_debug": "DS4_MTP_EXACT_REPLAY",
    "force_snapshot": "DS4_MTP_FORCE_SNAPSHOT",
    "full_logits": "DS4_MTP_FULL_LOGITS",
    "min_margin": "DS4_MTP_MIN_MARGIN",
    "conf_log": "DS4_MTP_CONF_LOG",
    "spec_log": "DS4_MTP_SPEC_LOG",
}
EXPECTED_BOUNDARIES = {
    "mtp_draft": 'boundary!("mtp_draft", "metal_graph_eval_mtp_draft_from_hc", 1, true)',
    "mtp_suffix_tops": 'boundary!("mtp_suffix_tops", "metal_graph_verify_suffix_tops", 2, true)',
    "mtp_decode2_exact": '"metal_graph_verify_decode2_exact"',
    "spec_frontier_snapshot": 'boundary!("spec_frontier_snapshot", "spec_frontier_snapshot", 1, true)',
    "spec_frontier_restore": 'boundary!("spec_frontier_restore", "spec_frontier_restore", 1, true)',
    "spec_frontier_commit_prefix1": '"spec_frontier_commit_prefix1"',
}
EXPECTED_CASES = {
    "b300_missing_mtp_support_model": {
        "path": "availability_blocker",
        "frontier_ops": [],
        "accepted_suffix": 0,
        "logits_source": "none",
        "mtp_n_raw_keep": 0,
        "fallback": "blocked_missing_mtp_model",
    },
    "mtp_disabled_after_first_token": {
        "path": "guard",
        "frontier_ops": [],
        "accepted_suffix": 0,
        "logits_source": "target first-token logits",
        "mtp_n_raw_keep": 0,
        "fallback": "return first-token accept",
    },
    "first_draft_miss": {
        "path": "draft_miss",
        "frontier_ops": [],
        "accepted_suffix": 0,
        "logits_source": "target first-token logits",
        "mtp_n_raw_keep": 0,
        "fallback": "skip speculative work",
    },
    "margin_skip_single_target_replay": {
        "path": "margin_skip",
        "frontier_ops": ["keep_accepted"],
        "accepted_suffix": 1,
        "logits_source": "target decode logits for drafts[0]",
        "mtp_n_raw_keep": 1,
        "fallback": "margin-skip",
    },
    "exact_decode2_full_accept": {
        "path": "exact_decode2",
        "frontier_ops": ["snapshot", "keep_accepted"],
        "accepted_suffix": 2,
        "logits_source": "decode2 logits1",
        "mtp_n_raw_keep": 2,
        "fallback": "none",
    },
    "exact_decode2_prefix1_accept": {
        "path": "exact_decode2",
        "frontier_ops": ["snapshot", "commit_prefix1", "keep_accepted"],
        "accepted_suffix": 1,
        "logits_source": "decode2 logits0",
        "mtp_n_raw_keep": 1,
        "fallback": "none",
    },
    "exact_decode2_failure_restore_then_sequential": {
        "path": "exact_decode2_failure",
        "frontier_ops": ["snapshot", "restore"],
        "accepted_suffix": "verified_by_sequential_fallback",
        "logits_source": "sequential target decode",
        "mtp_n_raw_keep": "verified_by_sequential_fallback",
        "fallback": "sequential safety fallback",
    },
    "suffix_full_accept": {
        "path": "suffix_verifier",
        "frontier_ops": ["keep_accepted"],
        "accepted_suffix": "draft_n",
        "logits_source": "spec logits row draft_n - 1",
        "mtp_n_raw_keep": "draft_n",
        "fallback": "none",
    },
    "suffix_prefix1_accept": {
        "path": "suffix_verifier",
        "frontier_ops": ["commit_prefix1", "keep_accepted"],
        "accepted_suffix": 1,
        "logits_source": "spec logits row 0",
        "mtp_n_raw_keep": 1,
        "fallback": "none",
    },
    "suffix_restore_replay_accept": {
        "path": "suffix_verifier_replay",
        "frontier_ops": ["snapshot", "restore", "keep_accepted"],
        "accepted_suffix": "commit_drafts",
        "logits_source": "target replay logits",
        "mtp_n_raw_keep": "commit_drafts",
        "fallback": "target replay",
    },
    "suffix_failure_restore_or_error": {
        "path": "suffix_verifier_failure",
        "frontier_ops": ["restore_or_error"],
        "accepted_suffix": "verified_by_sequential_fallback_or_error",
        "logits_source": "sequential target decode or none on hard error",
        "mtp_n_raw_keep": "verified_by_sequential_fallback_or_zero",
        "fallback": "sequential safety fallback or MTP verifier failed",
    },
    "sequential_safety_fallback": {
        "path": "sequential_fallback",
        "frontier_ops": ["keep_accepted"],
        "accepted_suffix": "verified",
        "logits_source": "normal target decode logits",
        "mtp_n_raw_keep": "verified",
        "fallback": "target sequential verifier",
    },
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def validate(contract: dict[str, Any], cli_controls: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expect_value(errors, contract, "schema", SCHEMA)
    expect_value(errors, contract, "milestone", EXPECTED_MILESTONE)
    expect_value(errors, contract, "source", EXPECTED_SOURCE)
    check_source_files(errors, contract)
    check_b300(errors, contract)
    check_support_artifacts(errors, contract, cli_controls)
    check_env_gates(errors, contract)
    check_state_fields(errors, contract)
    check_function_anchors(errors, contract)
    check_command_boundaries(errors, contract)
    check_decision_cases(errors, contract)
    return errors


def check_source_files(errors: list[str], contract: dict[str, Any]) -> None:
    expected = [
        "ds4.c",
        "ds4.h",
        "rust/ds4-gpu/src/graph_plan.rs",
        rel(CLI_RUNTIME_CONTROLS),
    ]
    got = contract.get("source_files")
    if got != expected:
        errors.append(f"source_files: expected {expected!r}, got {got!r}")


def check_b300(errors: list[str], contract: dict[str, Any]) -> None:
    b300 = expect_object(errors, contract.get("b300"), "b300")
    if not b300:
        return
    for key, expected in EXPECTED_B300.items():
        got = b300.get(key)
        if got != expected:
            errors.append(f"b300.{key}: expected {expected!r}, got {got!r}")


def check_support_artifacts(
    errors: list[str],
    contract: dict[str, Any],
    cli_controls: dict[str, Any],
) -> None:
    support = expect_object(errors, contract.get("support_artifacts"), "support_artifacts")
    mtp = expect_object(errors, support.get("mtp"), "support_artifacts.mtp")
    cli_support = expect_object(errors, cli_controls.get("support_artifacts"), "cli.support_artifacts")
    cli_mtp = expect_object(errors, cli_support.get("mtp"), "cli.support_artifacts.mtp")
    expected_path = EXPECTED_B300["expected_mtp_path"]
    if mtp.get("path") != expected_path:
        errors.append(f"support_artifacts.mtp.path: expected {expected_path!r}, got {mtp.get('path')!r}")
    if mtp.get("available") is not False:
        errors.append("support_artifacts.mtp.available: expected false")
    if mtp.get("candidate_count") != 0:
        errors.append(f"support_artifacts.mtp.candidate_count: expected 0, got {mtp.get('candidate_count')!r}")
    if "no MTP GGUF" not in str(mtp.get("blocker")):
        errors.append("support_artifacts.mtp.blocker: missing no-MTP blocker")
    for key in ("path", "available", "blocker"):
        if mtp.get(key) != cli_mtp.get(key):
            errors.append(
                f"support_artifacts.mtp.{key}: contract {mtp.get(key)!r} "
                f"differs from M8.12b {cli_mtp.get(key)!r}"
            )


def check_env_gates(errors: list[str], contract: dict[str, Any]) -> None:
    gates = expect_object(errors, contract.get("environment_gates"), "environment_gates")
    if gates != EXPECTED_ENV_GATES:
        errors.append("environment_gates drift")
    ds4 = read_text("ds4.c", errors)
    for key, env in EXPECTED_ENV_GATES.items():
        if key == "spec_disable_callers":
            continue
        if env not in ds4:
            errors.append(f"ds4.c: missing env gate {env}")
    for caller in ("ds4_cli.c", "ds4_server.c"):
        text = read_text(caller, errors)
        if "DS4_MTP_SPEC_DISABLE" not in text:
            errors.append(f"{caller}: missing DS4_MTP_SPEC_DISABLE caller gate")


def check_state_fields(errors: list[str], contract: dict[str, Any]) -> None:
    fields = expect_object(errors, contract.get("state_fields"), "state_fields")
    expected = {
        "checkpoint",
        "target_logits",
        "draft_valid",
        "draft_token",
        "draft_logits",
        "mtp_raw_frontier",
        "compressed_frontier",
        "spec_frontier_tensors",
        "prefix1_frontier_tensors",
    }
    if set(fields) != expected:
        errors.append(f"state_fields: expected {sorted(expected)}, got {sorted(fields)}")


def check_function_anchors(errors: list[str], contract: dict[str, Any]) -> None:
    anchors = expect_named_objects(errors, contract, "function_anchors", "name")
    expected_names = {
        "ds4_engine_has_mtp",
        "ds4_engine_mtp_draft_tokens",
        "metal_graph_eval_mtp_draft_from_hc",
        "metal_graph_eval_mtp_draft",
        "metal_graph_verify_decode2_exact",
        "metal_graph_verify_suffix_tops",
        "spec_frontier_snapshot",
        "spec_frontier_restore",
        "spec_frontier_commit_prefix1",
        "ds4_session_decode_speculative",
    }
    if set(anchors) != expected_names:
        errors.append(f"function_anchors: expected {sorted(expected_names)}, got {sorted(anchors)}")
    cache: dict[str, str] = {}
    for name, anchor in anchors.items():
        file_value = anchor.get("file")
        if not isinstance(file_value, str):
            errors.append(f"function_anchors.{name}.file: expected string")
            continue
        text = cache.setdefault(file_value, read_text(file_value, errors))
        snippets = anchor.get("required_snippets")
        if not isinstance(snippets, list) or not snippets:
            errors.append(f"function_anchors.{name}.required_snippets: expected non-empty list")
            continue
        for snippet in snippets:
            if not isinstance(snippet, str) or not snippet:
                errors.append(f"function_anchors.{name}: invalid snippet {snippet!r}")
            elif snippet not in text:
                errors.append(f"function_anchors.{name}: missing source snippet {snippet!r}")


def check_command_boundaries(errors: list[str], contract: dict[str, Any]) -> None:
    boundaries = contract.get("command_boundaries")
    expected_names = list(EXPECTED_BOUNDARIES)
    if boundaries != expected_names:
        errors.append(f"command_boundaries: expected {expected_names!r}, got {boundaries!r}")
    graph_plan = read_text("rust/ds4-gpu/src/graph_plan.rs", errors)
    for name, snippet in EXPECTED_BOUNDARIES.items():
        if snippet not in graph_plan:
            errors.append(f"graph_plan.rs: missing command boundary {name}")


def check_decision_cases(errors: list[str], contract: dict[str, Any]) -> None:
    cases = expect_named_objects(errors, contract, "decision_cases", "id")
    if set(cases) != set(EXPECTED_CASES):
        errors.append(f"decision_cases: expected {sorted(EXPECTED_CASES)}, got {sorted(cases)}")
    function_names = {
        item.get("name")
        for item in expect_list(errors, contract, "function_anchors")
        if isinstance(item, dict)
    }
    for case_id, expected in EXPECTED_CASES.items():
        case = cases.get(case_id)
        if case is None:
            continue
        for key, value in expected.items():
            if case.get(key) != value:
                errors.append(f"decision_cases.{case_id}.{key}: expected {value!r}, got {case.get(key)!r}")
        conditions = case.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            errors.append(f"decision_cases.{case_id}.conditions: expected non-empty list")
        checkpoint_action = case.get("checkpoint_action")
        if not isinstance(checkpoint_action, str) or not checkpoint_action:
            errors.append(f"decision_cases.{case_id}.checkpoint_action: expected non-empty string")
        required = case.get("required_functions")
        if not isinstance(required, list):
            errors.append(f"decision_cases.{case_id}.required_functions: expected list")
        else:
            for function in required:
                if function in {"sample_argmax", "metal_graph_eval_token_raw_swa", "metal_graph_eval_token_raw_swa_top", "ds4_session_eval", "metal_graph_read_spec_logits_row"}:
                    continue
                if function not in function_names:
                    errors.append(f"decision_cases.{case_id}: unknown required function {function!r}")
    check_case_relationships(errors, cases)


def check_case_relationships(errors: list[str], cases: dict[str, dict[str, Any]]) -> None:
    exact_full = cases.get("exact_decode2_full_accept", {})
    exact_prefix = cases.get("exact_decode2_prefix1_accept", {})
    suffix_prefix = cases.get("suffix_prefix1_accept", {})
    suffix_replay = cases.get("suffix_restore_replay_accept", {})
    if exact_full.get("accepted_suffix") != 2 or exact_full.get("logits_source") != "decode2 logits1":
        errors.append("exact_decode2_full_accept must accept two suffix tokens from logits1")
    if "commit_prefix1" not in exact_prefix.get("frontier_ops", []):
        errors.append("exact_decode2_prefix1_accept must commit prefix1")
    if "commit_prefix1" not in suffix_prefix.get("frontier_ops", []):
        errors.append("suffix_prefix1_accept must commit prefix1")
    if "restore" not in suffix_replay.get("frontier_ops", []):
        errors.append("suffix_restore_replay_accept must restore before replay")
    if cases.get("b300_missing_mtp_support_model", {}).get("accepted_suffix") != 0:
        errors.append("missing MTP support case must not create accepted suffix tokens")


def read_text(path: str, errors: list[str]) -> str:
    full = ROOT / path
    try:
        return full.read_text()
    except OSError as exc:
        errors.append(f"{path}: unable to read: {exc}")
        return ""


def expect_value(errors: list[str], obj: dict[str, Any], key: str, expected: Any) -> None:
    got = obj.get(key)
    if got != expected:
        errors.append(f"{key}: expected {expected!r}, got {got!r}")


def expect_object(errors: list[str], obj: Any, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        errors.append(f"{label}: expected object")
        return {}
    return obj


def expect_list(errors: list[str], obj: dict[str, Any], key: str) -> list[Any]:
    value = obj.get(key)
    if not isinstance(value, list):
        errors.append(f"{key}: expected list")
        return []
    return value


def expect_named_objects(
    errors: list[str],
    obj: dict[str, Any],
    key: str,
    name_key: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(expect_list(errors, obj, key)):
        if not isinstance(item, dict):
            errors.append(f"{key}[{index}]: expected object")
            continue
        name = item.get(name_key)
        if not isinstance(name, str) or not name:
            errors.append(f"{key}[{index}].{name_key}: expected non-empty string")
            continue
        if name in result:
            errors.append(f"{key}: duplicate {name_key} {name!r}")
        result[name] = item
    return result


def run_negative_tests(contract: dict[str, Any], cli_controls: dict[str, Any]) -> int:
    mutations = [
        ("schema drift", lambda c: c.__setitem__("schema", "wrong")),
        ("mtp availability drift", lambda c: c["support_artifacts"]["mtp"].__setitem__("available", True)),
        ("missing case", lambda c: c["decision_cases"].pop()),
        (
            "exact full accept count drift",
            lambda c: find_case(c, "exact_decode2_full_accept").__setitem__("accepted_suffix", 1),
        ),
        (
            "suffix prefix action drift",
            lambda c: find_case(c, "suffix_prefix1_accept").__setitem__("frontier_ops", ["keep_accepted"]),
        ),
        (
            "missing source anchor",
            lambda c: c["function_anchors"][0].__setitem__("required_snippets", ["not in source"]),
        ),
        (
            "command boundary drift",
            lambda c: c.__setitem__("command_boundaries", c["command_boundaries"][:-1]),
        ),
    ]
    failures = 0
    for name, mutate in mutations:
        candidate = copy.deepcopy(contract)
        mutate(candidate)
        errors = validate(candidate, cli_controls)
        if errors:
            failures += 1
        else:
            print(f"negative-test failed to detect mutation: {name}", file=sys.stderr)
    if failures != len(mutations):
        return 1
    print(f"MTP state-machine contract negative tests: PASS, {failures} mutations")
    return 0


def find_case(contract: dict[str, Any], case_id: str) -> dict[str, Any]:
    for item in contract["decision_cases"]:
        if item.get("id") == case_id:
            return item
    raise KeyError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--cli-controls", type=Path, default=CLI_RUNTIME_CONTROLS)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        contract = load_json(args.contract)
        cli_controls = load_json(args.cli_controls)
    except Exception as exc:
        print(f"MTP state-machine contract: FAIL: {exc}", file=sys.stderr)
        return 1

    errors = validate(contract, cli_controls)
    if errors:
        print("MTP state-machine contract: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "MTP state-machine contract: PASS, "
        f"{len(contract['decision_cases'])} cases, "
        f"{len(contract['function_anchors'])} source anchors"
    )
    if args.negative_test:
        return run_negative_tests(contract, cli_controls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
