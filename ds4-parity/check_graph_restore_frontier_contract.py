#!/usr/bin/env python3
"""Validate the M10.7d3a graph restore continued-frontier contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "ds4-parity/baselines/kv/m10.7d3/restore-frontier-contract.json"
)
DEFAULT_RESTORE_SUMMARY = (
    ROOT / "ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json"
)
DEFAULT_POLICY_ORACLE = ROOT / "ds4-parity/baselines/kv/m7.2/current-c.json"

SCHEMA = "ds4.graph_restore_frontier_contract.v1"
EXPECTED_CASES = (
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
)
EXPECTED_POLICY_PROBES = (
    "restored_seed_frontier_reenables_next_boundary",
    "restored_continuation_frontier_reenables_next_boundary",
    "already_stored_boundary_skips",
)
EXPECTED_POLICY_OPTIONS = {
    "min_tokens": 512,
    "cold_max_tokens": 30000,
    "continued_interval_tokens": 10000,
    "boundary_trim_tokens": 32,
    "boundary_align_tokens": 2048,
    "continued_step_tokens": 10240,
}
EXPECTED_REASON_CODES = {
    "cold": 1,
    "continued": 2,
    "shutdown": 4,
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def validate(
    contract: dict[str, Any],
    restore_summary: dict[str, Any],
    policy_oracle: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expect_value(errors, contract, "schema", SCHEMA)
    expect_value(errors, contract, "milestone", "M10.7d3a")
    expect_value(errors, contract, "source", "model-free-graph-restore-frontier-contract")
    expect_value(errors, contract, "restore_summary_path", rel(DEFAULT_RESTORE_SUMMARY))
    expect_value(errors, contract, "policy_oracle_path", rel(DEFAULT_POLICY_ORACLE))

    policy_options = expect_object(errors, contract, "policy_options")
    if policy_options:
        check_exact_object(errors, "policy_options", policy_options, EXPECTED_POLICY_OPTIONS)
        validate_policy_options(errors, policy_options, policy_oracle)

    reason_codes = expect_object(errors, contract, "reason_codes")
    if reason_codes:
        check_exact_object(errors, "reason_codes", reason_codes, EXPECTED_REASON_CODES)
        validate_reason_codes(errors, reason_codes, policy_oracle)

    validate_policy_oracle_references(errors, policy_oracle)
    restore_cases = restore_cases_by_id(errors, restore_summary)
    header_rows = m0_5_header_rows(errors, policy_oracle)

    probes = expect_named_objects(errors, contract, "policy_probes", "name")
    if probes and policy_options and reason_codes:
        check_names(errors, "policy_probes", probes, EXPECTED_POLICY_PROBES)
        validate_policy_probes(errors, probes, policy_options, reason_codes)

    cases = expect_named_objects(errors, contract, "restore_frontier_cases", "name")
    if cases and policy_options and reason_codes:
        check_names(errors, "restore_frontier_cases", cases, EXPECTED_CASES)
        for name in EXPECTED_CASES:
            case = cases.get(name)
            restore_case = restore_cases.get(name)
            if case is None or restore_case is None:
                continue
            validate_restore_case(
                errors,
                case,
                restore_case,
                policy_options,
                reason_codes,
                header_rows,
            )
    static_checks(errors)
    return errors


def validate_policy_options(
    errors: list[str],
    policy_options: dict[str, Any],
    policy_oracle: dict[str, Any],
) -> None:
    defaults = policy_oracle.get("defaults")
    if not isinstance(defaults, dict):
        errors.append("policy_oracle.defaults: expected object")
        return
    for key, expected in EXPECTED_POLICY_OPTIONS.items():
        if key == "continued_step_tokens":
            got = continued_step(policy_options)
        else:
            got = defaults.get(key)
        if got != expected:
            errors.append(f"policy_options.{key}: expected {expected!r}, got {got!r}")


def validate_reason_codes(
    errors: list[str],
    reason_codes: dict[str, Any],
    policy_oracle: dict[str, Any],
) -> None:
    oracle_codes = {}
    for item in expect_list(errors, policy_oracle, "reason_codes"):
        if isinstance(item, dict) and isinstance(item.get("input"), str):
            oracle_codes[item["input"]] = item.get("code")
    for name, expected in reason_codes.items():
        if oracle_codes.get(name) != expected:
            errors.append(
                f"reason_codes.{name}: oracle expected {oracle_codes.get(name)!r}, got {expected!r}"
            )


def validate_policy_oracle_references(
    errors: list[str],
    policy_oracle: dict[str, Any],
) -> None:
    policy_cases = expect_object(errors, policy_oracle, "policy_cases")
    transitions = named_items(
        errors,
        expect_list(errors, policy_cases, "continued_frontier_transitions"),
        "policy_cases.continued_frontier_transitions",
        "name",
    )
    expected_transition_events = {
        "disk_restore_records_loaded_frontier": [
            {
                "op": "record_disk_load",
                "tokens": 552,
                "frontier": 552,
                "target_probe": 10240,
                "target": 10240,
            }
        ],
        "suppress_already_stored_skip": [
            {
                "op": "suppress",
                "tokens": 10240,
                "old_frontier": -1,
                "frontier": 10240,
                "target_probe": 10240,
                "target": 0,
            },
            {
                "op": "restore_suppressed",
                "restore_old_frontier": -1,
                "restore_suppressed_tokens": 10240,
                "frontier": 10240,
                "target_probe": 10240,
                "target": 0,
            },
        ],
        "suppress_fresh_frontier": [
            {
                "op": "suppress",
                "tokens": 10240,
                "old_frontier": 0,
                "frontier": 10240,
                "target_probe": 10240,
                "target": 0,
            },
            {
                "op": "restore_suppressed",
                "restore_old_frontier": 0,
                "restore_suppressed_tokens": 10240,
                "frontier": 0,
                "target_probe": 10240,
                "target": 10240,
            },
        ],
    }
    for name, expected_events in expected_transition_events.items():
        transition = transitions.get(name)
        if transition is None:
            errors.append(f"continued_frontier_transitions: missing {name!r}")
            continue
        if transition.get("events") != expected_events:
            errors.append(f"continued_frontier_transitions.{name}.events drift")


def restore_cases_by_id(
    errors: list[str],
    restore_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expect_value(errors, restore_summary, "schema", "ds4.rust_graph_restore_next_token_summary.v1")
    cases = expect_named_objects(errors, restore_summary, "cases", "id")
    check_names(errors, "restore_summary.cases", cases, EXPECTED_CASES)
    return cases


def m0_5_header_rows(
    errors: list[str],
    policy_oracle: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    fixture = expect_object(errors, policy_oracle, "m0_5_header_fixture")
    rows = named_items(
        errors,
        expect_list(errors, fixture, "expected_rows"),
        "m0_5_header_fixture.expected_rows",
        "file",
    )
    expected_rows = {
        "0ab2314538b11686a11e296b7f697651fbd17e60.kv": ("cold", 1, 550),
        "a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv": ("shutdown", 4, 552),
    }
    for file_name, (reason_name, reason, tokens) in expected_rows.items():
        row = rows.get(file_name)
        if row is None:
            errors.append(f"m0_5_header_fixture: missing {file_name}")
            continue
        if (row.get("reason_name"), row.get("reason"), row.get("tokens")) != (
            reason_name,
            reason,
            tokens,
        ):
            errors.append(f"m0_5_header_fixture.{file_name}: reason/tokens drift")
    return rows


def validate_policy_probes(
    errors: list[str],
    probes: dict[str, dict[str, Any]],
    policy_options: dict[str, Any],
    reason_codes: dict[str, Any],
) -> None:
    for name, probe in probes.items():
        target = continued_store_target(
            policy_options,
            int_field(errors, probe, "live_tokens", f"policy_probes.{name}"),
            int_field(errors, probe, "frontier_before", f"policy_probes.{name}"),
        )
        if probe.get("target") != target:
            errors.append(f"policy_probes.{name}.target: expected {target}, got {probe.get('target')!r}")
        reason_name = probe.get("reason_name")
        if reason_name is not None and probe.get("reason") != reason_codes.get(reason_name):
            errors.append(f"policy_probes.{name}.reason code drift")


def validate_restore_case(
    errors: list[str],
    case: dict[str, Any],
    restore_case: dict[str, Any],
    policy_options: dict[str, Any],
    reason_codes: dict[str, Any],
    header_rows: dict[str, dict[str, Any]],
) -> None:
    name = str(case.get("name"))
    path = f"restore_frontier_cases.{name}"
    rust = expect_object(errors, restore_case, "rust")
    next_token = expect_object(errors, rust, "next_token")
    restored_tokens = next_token.get("checkpoint_tokens")
    for key in ("prompt_case", "kind"):
        if case.get(key) != restore_case.get(key):
            errors.append(f"{path}.{key}: expected restore summary value {restore_case.get(key)!r}, got {case.get(key)!r}")
    if case.get("restored_tokens") != restored_tokens:
        errors.append(f"{path}.restored_tokens: expected {restored_tokens!r}, got {case.get('restored_tokens')!r}")
    if case.get("loaded_frontier") != restored_tokens:
        errors.append(f"{path}.loaded_frontier: expected restored token count {restored_tokens!r}")

    current_live_skip = expect_object(errors, case, "current_live_skip")
    if current_live_skip:
        validate_target(
            errors,
            f"{path}.current_live_skip",
            current_live_skip,
            policy_options,
            int(case.get("loaded_frontier", 0)),
        )
        if current_live_skip.get("reason") != "restored-position-unaligned":
            errors.append(f"{path}.current_live_skip.reason drift")

    next_store = expect_object(errors, case, "next_continued_store")
    if next_store:
        validate_target(
            errors,
            f"{path}.next_continued_store",
            next_store,
            policy_options,
            int_field(errors, next_store, "frontier_before", f"{path}.next_continued_store"),
        )
        if next_store.get("reason_name") != "continued":
            errors.append(f"{path}.next_continued_store.reason_name drift")
        if next_store.get("reason") != reason_codes.get("continued"):
            errors.append(f"{path}.next_continued_store.reason code drift")

    shutdown = expect_object(errors, case, "post_restore_shutdown")
    if shutdown:
        if shutdown.get("reason_name") != "shutdown":
            errors.append(f"{path}.post_restore_shutdown.reason_name drift")
        if shutdown.get("reason") != reason_codes.get("shutdown"):
            errors.append(f"{path}.post_restore_shutdown.reason code drift")
        if shutdown.get("tokens_source") != "restored-session-position":
            errors.append(f"{path}.post_restore_shutdown.tokens_source drift")

    validate_header_reference(errors, path, case, "cold_store_reference", header_rows)
    validate_header_reference(errors, path, case, "restored_disk_reference", header_rows)


def validate_header_reference(
    errors: list[str],
    case_path: str,
    case: dict[str, Any],
    key: str,
    header_rows: dict[str, dict[str, Any]],
) -> None:
    value = case.get(key)
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{case_path}.{key}: expected object")
        return
    file_name = value.get("file")
    row = header_rows.get(file_name)
    if row is None:
        errors.append(f"{case_path}.{key}.file: missing M0.5 header row {file_name!r}")
        return
    for field in ("reason_name", "reason", "tokens"):
        if value.get(field) != row.get(field):
            errors.append(
                f"{case_path}.{key}.{field}: expected M0.5 row value {row.get(field)!r}, got {value.get(field)!r}"
            )


def validate_target(
    errors: list[str],
    path: str,
    data: dict[str, Any],
    policy_options: dict[str, Any],
    frontier_before: int,
) -> None:
    live_tokens = int_field(errors, data, "live_tokens", path)
    expected = continued_store_target(policy_options, live_tokens, frontier_before)
    if data.get("target") != expected:
        errors.append(f"{path}.target: expected {expected}, got {data.get('target')!r}")


def continued_step(policy_options: dict[str, Any]) -> int:
    interval = int(policy_options.get("continued_interval_tokens", 0))
    if interval <= 0:
        return 0
    align = int(policy_options.get("boundary_align_tokens", 0))
    if align > 0:
        step = ((interval + align - 1) // align) * align
        return step if step > 0 else align
    return interval


def continued_store_target(
    policy_options: dict[str, Any],
    live_tokens: int,
    frontier_before: int,
) -> int:
    step = continued_step(policy_options)
    if step <= 0:
        return 0
    if live_tokens < int(policy_options.get("min_tokens", 0)):
        return 0
    if live_tokens % step != 0:
        return 0
    if live_tokens <= frontier_before:
        return 0
    return live_tokens


def static_checks(errors: list[str]) -> None:
    files = {
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "readme": ROOT / "ds4-parity/README.md",
        "report": ROOT / "ds4-parity/run_parity_report.py",
    }
    texts = {name: path.read_text() for name, path in files.items()}
    required = {
        "roadmap": "M10.7d3a: Graph Restore Frontier Contract",
        "todo": "M10.7d3a: Graph Restore Frontier Contract",
        "status": "M10.7d3a Graph Restore Frontier Contract",
        "readme": "check_graph_restore_frontier_contract.py",
        "report": "M10.7d3a Rust graph restore frontier contract",
    }
    for name, snippet in required.items():
        if snippet not in texts[name]:
            errors.append(f"static_checks.{name}: missing {snippet!r}")


def expect_object(errors: list[str], data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if isinstance(value, dict):
        return value
    errors.append(f"{key}: expected object")
    return {}


def expect_list(errors: list[str], data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if isinstance(value, list):
        return value
    errors.append(f"{key}: expected list")
    return []


def expect_value(errors: list[str], data: dict[str, Any], key: str, expected: Any) -> None:
    got = data.get(key)
    if got != expected:
        errors.append(f"{key}: expected {expected!r}, got {got!r}")


def expect_named_objects(
    errors: list[str],
    data: dict[str, Any],
    key: str,
    name_key: str,
) -> dict[str, dict[str, Any]]:
    return named_items(errors, expect_list(errors, data, key), key, name_key)


def named_items(
    errors: list[str],
    items: list[Any],
    label: str,
    name_key: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"{label}[{index}]: expected object")
            continue
        name = item.get(name_key)
        if not isinstance(name, str) or not name:
            errors.append(f"{label}[{index}].{name_key}: expected non-empty string")
            continue
        if name in out:
            errors.append(f"{label}: duplicate {name_key} {name!r}")
            continue
        out[name] = item
    return out


def check_names(
    errors: list[str],
    label: str,
    actual: dict[str, dict[str, Any]],
    expected: tuple[str, ...],
) -> None:
    if tuple(actual) != expected:
        errors.append(f"{label}: expected names {expected!r}, got {tuple(actual)!r}")


def check_exact_object(
    errors: list[str],
    label: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if set(actual) != set(expected):
        errors.append(f"{label}: expected keys {sorted(expected)}, got {sorted(actual)}")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            errors.append(f"{label}.{key}: expected {expected_value!r}, got {actual.get(key)!r}")


def int_field(errors: list[str], data: dict[str, Any], key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    errors.append(f"{path}.{key}: expected int")
    return 0


def run_negative_tests(
    contract: dict[str, Any],
    restore_summary: dict[str, Any],
    policy_oracle: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    def expect_failure(label: str, mutator: Any) -> None:
        bad_contract = copy.deepcopy(contract)
        bad_restore = copy.deepcopy(restore_summary)
        bad_policy = copy.deepcopy(policy_oracle)
        mutator(bad_contract, bad_restore, bad_policy)
        if not validate(bad_contract, bad_restore, bad_policy):
            errors.append(f"negative test did not fail: {label}")

    expect_failure(
        "restored token drift",
        lambda c, _r, _p: c["restore_frontier_cases"][0].__setitem__("restored_tokens", 551),
    )
    expect_failure(
        "loaded frontier drift",
        lambda c, _r, _p: c["restore_frontier_cases"][2].__setitem__("loaded_frontier", 552),
    )
    expect_failure(
        "continued target drift",
        lambda c, _r, _p: c["restore_frontier_cases"][0]["next_continued_store"].__setitem__(
            "target", 0
        ),
    )
    expect_failure(
        "already-stored skip drift",
        lambda c, _r, _p: c["policy_probes"][2].__setitem__("target", 10240),
    )
    expect_failure(
        "reason drift",
        lambda c, _r, _p: c["restore_frontier_cases"][1]["post_restore_shutdown"].__setitem__(
            "reason", 1
        ),
    )
    expect_failure(
        "restore summary drift",
        lambda _c, r, _p: r["cases"][0]["rust"]["next_token"].__setitem__(
            "checkpoint_tokens", 551
        ),
    )
    expect_failure(
        "policy oracle transition drift",
        lambda _c, _r, p: p["policy_cases"]["continued_frontier_transitions"][7]["events"][
            0
        ].__setitem__("target", 0),
    )
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--restore-summary", type=Path, default=DEFAULT_RESTORE_SUMMARY)
    parser.add_argument("--policy-oracle", type=Path, default=DEFAULT_POLICY_ORACLE)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        contract = load_json(args.contract)
        restore_summary = load_json(args.restore_summary)
        policy_oracle = load_json(args.policy_oracle)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    errors = validate(contract, restore_summary, policy_oracle)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.negative_test:
        negative_errors = run_negative_tests(contract, restore_summary, policy_oracle)
        if negative_errors:
            for error in negative_errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
    print("graph restore frontier contract: PASS, 4 cases, 3 policy probes")
    if args.negative_test:
        print("graph restore frontier contract negative tests: PASS, 7 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
