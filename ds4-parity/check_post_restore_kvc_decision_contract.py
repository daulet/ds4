#!/usr/bin/env python3
"""Validate the M10.7d3c1 post-restore KVC decision contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "ds4-parity/baselines/kv/m10.7d3/post-restore-kvc-decision-contract.json"
)
DEFAULT_RESTORE_SUMMARY = (
    ROOT / "ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json"
)
DEFAULT_FRONTIER_CONTRACT = (
    ROOT / "ds4-parity/baselines/kv/m10.7d3/restore-frontier-contract.json"
)
DEFAULT_RUNTIME_SUMMARY = ROOT / "ds4-parity/baselines/kv/m9.8f5/runtime-rust-b300-summary.json"
DEFAULT_RUNTIME_LEDGER_CONTRACT = (
    ROOT / "ds4-parity/baselines/kv/m10.7d2/runtime-ledger-contract.json"
)
DEFAULT_KVC_FILE_ORACLE = ROOT / "ds4-parity/baselines/kv/m7.4a/current-c.json"

SCHEMA = "ds4.post_restore_kvc_decision_contract.v1"
EXPECTED_CASES = (
    "disk_seed_payload",
    "snapshot_seed",
    "disk_continuation_payload",
    "snapshot_continuation",
)
EXPECTED_RUNTIME_REFERENCES = (
    "seed_restore_skips_continued",
    "continuation_restore_skips_continued",
    "shutdown_reason_header_reference",
)


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
    frontier_contract: dict[str, Any],
    runtime_summary: dict[str, Any],
    runtime_ledger_contract: dict[str, Any],
    kvc_file_oracle: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expect_value(errors, contract, "schema", SCHEMA)
    expect_value(errors, contract, "milestone", "M10.7d3c1")
    expect_value(errors, contract, "source", "model-free-post-restore-kvc-decision-contract")
    expect_value(errors, contract, "restore_summary_path", rel(DEFAULT_RESTORE_SUMMARY))
    expect_value(errors, contract, "frontier_contract_path", rel(DEFAULT_FRONTIER_CONTRACT))
    expect_value(errors, contract, "runtime_summary_path", rel(DEFAULT_RUNTIME_SUMMARY))
    expect_value(
        errors,
        contract,
        "runtime_ledger_contract_path",
        rel(DEFAULT_RUNTIME_LEDGER_CONTRACT),
    )
    expect_value(errors, contract, "kvc_file_oracle_path", rel(DEFAULT_KVC_FILE_ORACLE))

    validate_artifact_headers(errors, restore_summary, frontier_contract, runtime_summary, runtime_ledger_contract, kvc_file_oracle)
    validate_policy_and_reasons(errors, contract, frontier_contract)
    validate_kvc_header_contract(errors, contract, kvc_file_oracle)
    validate_runtime_references(errors, contract, runtime_summary, runtime_ledger_contract)
    validate_post_restore_cases(errors, contract, restore_summary, frontier_contract)
    static_checks(errors)
    return errors


def validate_artifact_headers(
    errors: list[str],
    restore_summary: dict[str, Any],
    frontier_contract: dict[str, Any],
    runtime_summary: dict[str, Any],
    runtime_ledger_contract: dict[str, Any],
    kvc_file_oracle: dict[str, Any],
) -> None:
    expect_value(errors, restore_summary, "schema", "ds4.rust_graph_restore_next_token_summary.v1")
    expect_value(errors, restore_summary, "source", "rust-b300-graph-restore-next-token")
    expect_value(errors, frontier_contract, "schema", "ds4.graph_restore_frontier_contract.v1")
    expect_value(errors, frontier_contract, "milestone", "M10.7d3a")
    expect_value(errors, runtime_summary, "schema", "ds4.runtime_kv_replay_summary.v1")
    expect_value(errors, runtime_summary, "milestone", "M9.8f5")
    expect_value(errors, runtime_ledger_contract, "schema", "ds4.runtime_kv_replay_ledger_contract.v1")
    expect_value(errors, runtime_ledger_contract, "milestone", "M10.7d2b")
    expect_value(errors, kvc_file_oracle, "schema", "ds4.kvc_file_oracle.v1")
    expect_value(errors, kvc_file_oracle, "source", "current-c-kvstore-file-no-model")


def validate_policy_and_reasons(
    errors: list[str],
    contract: dict[str, Any],
    frontier_contract: dict[str, Any],
) -> None:
    policy = expect_object(errors, contract, "policy_options")
    frontier_policy = expect_object(errors, frontier_contract, "policy_options")
    if policy and frontier_policy and policy != frontier_policy:
        errors.append("policy_options: drift from M10.7d3a frontier contract")
    reason_codes = expect_object(errors, contract, "reason_codes")
    frontier_reasons = expect_object(errors, frontier_contract, "reason_codes")
    expected_reasons = {"cold": 1, "continued": 2, "shutdown": 4}
    if reason_codes and reason_codes != expected_reasons:
        errors.append("reason_codes: drift from expected KVC reason constants")
    if reason_codes and frontier_reasons:
        for key, expected in expected_reasons.items():
            if frontier_reasons.get(key) != expected:
                errors.append(f"frontier_contract.reason_codes.{key}: drift")


def validate_kvc_header_contract(
    errors: list[str],
    contract: dict[str, Any],
    kvc_file_oracle: dict[str, Any],
) -> None:
    header = expect_object(errors, contract, "kvc_header_contract")
    constants = expect_object(errors, kvc_file_oracle, "constants")
    if header.get("fixed_header_bytes") != constants.get("fixed_header"):
        errors.append("kvc_header_contract.fixed_header_bytes: drift from M7.4a oracle")
    expected = {
        "quant_bits": 2,
        "ext_flags": 0,
        "hits_for_new_file": 0,
        "ctx_size": 32768,
        "fixed_header_bytes": 48,
        "text_length_bytes": 4,
        "trailer_bytes": 0,
        "file_size_formula": "fixed_header_bytes + text_length_bytes + rendered_text_bytes + payload_bytes",
    }
    check_exact_object(errors, "kvc_header_contract", header, expected)
    shutdown_case = named_items(
        errors,
        expect_list(errors, kvc_file_oracle, "cases"),
        "kvc_file_oracle.cases",
        "name",
    ).get("empty_text_thinking_trailer")
    if shutdown_case is None:
        errors.append("kvc_file_oracle.cases: missing shutdown fixture")
        return
    input_obj = expect_object(errors, shutdown_case, "input")
    if input_obj.get("reason") != 4:
        errors.append("kvc_file_oracle shutdown reason code drift")


def validate_runtime_references(
    errors: list[str],
    contract: dict[str, Any],
    runtime_summary: dict[str, Any],
    runtime_ledger_contract: dict[str, Any],
) -> None:
    refs = named_items(
        errors,
        expect_list(errors, contract, "runtime_restore_references"),
        "runtime_restore_references",
        "name",
    )
    check_names(errors, "runtime_restore_references", refs, EXPECTED_RUNTIME_REFERENCES)
    summary_cases = named_items(errors, expect_list(errors, runtime_summary, "ledger_cases"), "runtime_summary.ledger_cases", "name")
    contract_cases = named_items(errors, expect_list(errors, runtime_ledger_contract, "cases"), "runtime_ledger_contract.cases", "name")
    for ref_name in ("seed_restore_skips_continued", "continuation_restore_skips_continued"):
        ref = refs.get(ref_name)
        if ref is None:
            continue
        runtime_case_name = ref.get("runtime_case")
        summary_case = summary_cases.get(runtime_case_name)
        ledger_case = contract_cases.get(runtime_case_name)
        if summary_case is None or ledger_case is None:
            errors.append(f"runtime_restore_references.{ref_name}: missing runtime case {runtime_case_name!r}")
            continue
        for field in ("cache_source", "cached_tokens", "cache_write_tokens", "disk_cached_tokens", "kv_write_file"):
            if ref.get(field) != summary_case.get(field):
                errors.append(f"runtime_restore_references.{ref_name}.{field}: summary drift")
            if ref.get(field) != ledger_case.get(field):
                errors.append(f"runtime_restore_references.{ref_name}.{field}: ledger contract drift")
        maybe = expect_object(errors, ref, "maybe_store_continued")
        expected_event = find_event(summary_case, "maybe_store_continued")
        contract_event = find_event(ledger_case, "maybe_store_continued")
        for source, event in (("summary", expected_event), ("ledger contract", contract_event)):
            if event is None:
                errors.append(f"runtime_restore_references.{ref_name}: missing maybe_store_continued in {source}")
                continue
            for field in ("tokens", "frontier_before", "frontier_after", "success"):
                if maybe.get(field) != event.get(field):
                    errors.append(f"runtime_restore_references.{ref_name}.maybe_store_continued.{field}: {source} drift")
    header_ref = refs.get("shutdown_reason_header_reference")
    if header_ref:
        headers = named_items(errors, expect_list(errors, runtime_summary, "kv_headers"), "runtime_summary.kv_headers", "file")
        row = headers.get(header_ref.get("runtime_header_file"))
        if row is None:
            errors.append("runtime_restore_references.shutdown_reason_header_reference: missing runtime header row")
        else:
            for field in ("reason_name", "reason", "ctx_size"):
                row_key = "ctx" if field == "ctx_size" else field
                if header_ref.get(field) != row.get(row_key):
                    errors.append(f"runtime_restore_references.shutdown_reason_header_reference.{field}: runtime header drift")


def validate_post_restore_cases(
    errors: list[str],
    contract: dict[str, Any],
    restore_summary: dict[str, Any],
    frontier_contract: dict[str, Any],
) -> None:
    cases = named_items(errors, expect_list(errors, contract, "post_restore_cases"), "post_restore_cases", "name")
    check_names(errors, "post_restore_cases", cases, EXPECTED_CASES)
    restore_cases = named_items(errors, expect_list(errors, restore_summary, "cases"), "restore_summary.cases", "id")
    frontier_cases = named_items(errors, expect_list(errors, frontier_contract, "restore_frontier_cases"), "frontier_contract.restore_frontier_cases", "name")
    already_probe = named_items(errors, expect_list(errors, frontier_contract, "policy_probes"), "frontier_contract.policy_probes", "name").get("already_stored_boundary_skips", {})
    header_defaults = expect_object(errors, contract, "kvc_header_contract")
    reason_codes = expect_object(errors, contract, "reason_codes")
    for name in EXPECTED_CASES:
        case = cases.get(name)
        restore_case = restore_cases.get(name)
        frontier_case = frontier_cases.get(name)
        if case is None or restore_case is None or frontier_case is None:
            continue
        path = f"post_restore_cases.{name}"
        for field in ("prompt_case", "kind", "raw_file"):
            restore_field = "raw_file" if field == "raw_file" else field
            if case.get(field) != restore_case.get(restore_field):
                errors.append(f"{path}.{field}: restore summary drift")
        rust = expect_object(errors, restore_case, "rust")
        parsed = expect_object(errors, rust, "parsed")
        next_token = expect_object(errors, expect_object(errors, rust, "next_token"), "frontier_projection")
        restored_tokens = int_value(errors, parsed, "token_count", f"{path}.parsed")
        payload_bytes = int_value(errors, parsed, "payload_bytes", f"{path}.parsed")
        if case.get("restored_tokens") != restored_tokens:
            errors.append(f"{path}.restored_tokens: restore summary drift")
        if case.get("payload_bytes") != payload_bytes:
            errors.append(f"{path}.payload_bytes: restore summary drift")
        compare_object(errors, case.get("current_live_skip"), frontier_case.get("current_live_skip"), f"{path}.current_live_skip")
        compare_object(errors, case.get("next_continued_store"), frontier_case.get("next_continued_store"), f"{path}.next_continued_store")
        compare_object(errors, case.get("already_stored_boundary"), {
            "frontier_before": already_probe.get("frontier_before"),
            "live_tokens": already_probe.get("live_tokens"),
            "target": already_probe.get("target"),
        }, f"{path}.already_stored_boundary")
        compare_object(errors, case.get("current_live_skip"), next_token.get("current_live_skip"), f"{path}.projection.current_live_skip")
        compare_object(errors, case.get("next_continued_store"), next_token.get("next_continued_store"), f"{path}.projection.next_continued_store")
        compare_object(errors, case.get("already_stored_boundary"), next_token.get("already_stored_boundary"), f"{path}.projection.already_stored_boundary")
        validate_shutdown_header(errors, path, case, header_defaults, reason_codes)


def validate_shutdown_header(
    errors: list[str],
    path: str,
    case: dict[str, Any],
    header_defaults: dict[str, Any],
    reason_codes: dict[str, Any],
) -> None:
    header = expect_object(errors, case, "shutdown_write_header")
    expected = {
        "quant_bits": header_defaults.get("quant_bits"),
        "reason_name": "shutdown",
        "reason": reason_codes.get("shutdown"),
        "ext_flags": header_defaults.get("ext_flags"),
        "tokens": case.get("restored_tokens"),
        "hits": header_defaults.get("hits_for_new_file"),
        "ctx_size": header_defaults.get("ctx_size"),
        "payload_bytes": case.get("payload_bytes"),
    }
    check_exact_object(errors, f"{path}.shutdown_write_header", header, expected)


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
        "roadmap": "M10.7d3c1: Post-Restore KVC Decision Contract",
        "todo": "M10.7d3c1: Post-Restore KVC Decision Contract",
        "status": "M10.7d3c1 Post-Restore KVC Decision Contract",
        "readme": "check_post_restore_kvc_decision_contract.py",
        "report": "M10.7d3c1 Rust post-restore KVC decision contract",
    }
    for name, snippet in required.items():
        if snippet not in texts[name]:
            errors.append(f"static_checks.{name}: missing {snippet!r}")


def find_event(case: dict[str, Any], name: str) -> dict[str, Any] | None:
    for event in case.get("events", []):
        if isinstance(event, dict) and event.get("name") == name:
            return event
    return None


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
    if list(actual) != list(expected):
        errors.append(f"{label}: key order or coverage drift")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            errors.append(f"{label}.{key}: expected {expected_value!r}, got {actual.get(key)!r}")


def compare_object(errors: list[str], actual: Any, expected: Any, label: str) -> None:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        errors.append(f"{label}: expected objects")
        return
    check_exact_object(errors, label, actual, expected)


def int_value(errors: list[str], data: dict[str, Any], key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    errors.append(f"{path}.{key}: expected int")
    return 0


def run_negative_tests(
    contract: dict[str, Any],
    restore_summary: dict[str, Any],
    frontier_contract: dict[str, Any],
    runtime_summary: dict[str, Any],
    runtime_ledger_contract: dict[str, Any],
    kvc_file_oracle: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    def mutate_runtime_maybe_store_success(runtime: dict[str, Any]) -> None:
        cases = named_items(
            [],
            runtime.get("ledger_cases", []),
            "runtime_summary.ledger_cases",
            "name",
        )
        case = cases["seed_restore"]
        event = find_event(case, "maybe_store_continued")
        if event is None:
            raise KeyError("seed_restore.maybe_store_continued")
        event["success"] = True

    def mutate_case_field(contract_data: dict[str, Any], case_name: str, field: str, value: Any) -> None:
        cases = named_items(
            [],
            contract_data.get("post_restore_cases", []),
            "post_restore_cases",
            "name",
        )
        cases[case_name][field] = value

    def mutate_case_nested_field(
        contract_data: dict[str, Any],
        case_name: str,
        object_name: str,
        field: str,
        value: Any,
    ) -> None:
        cases = named_items(
            [],
            contract_data.get("post_restore_cases", []),
            "post_restore_cases",
            "name",
        )
        nested = cases[case_name][object_name]
        if isinstance(nested, dict):
            nested[field] = value

    def mutate_restore_projection(
        restore: dict[str, Any],
        case_id: str,
        object_name: str,
        field: str,
        value: Any,
    ) -> None:
        cases = named_items([], restore.get("cases", []), "restore_summary.cases", "id")
        projection = cases[case_id]["rust"]["next_token"]["frontier_projection"]
        nested = projection[object_name]
        if isinstance(nested, dict):
            nested[field] = value

    def expect_failure(label: str, mutator: Callable[..., None]) -> None:
        bad_contract = copy.deepcopy(contract)
        bad_restore = copy.deepcopy(restore_summary)
        bad_frontier = copy.deepcopy(frontier_contract)
        bad_runtime = copy.deepcopy(runtime_summary)
        bad_ledger = copy.deepcopy(runtime_ledger_contract)
        bad_kvc = copy.deepcopy(kvc_file_oracle)
        mutator(bad_contract, bad_restore, bad_frontier, bad_runtime, bad_ledger, bad_kvc)
        if not validate(bad_contract, bad_restore, bad_frontier, bad_runtime, bad_ledger, bad_kvc):
            errors.append(f"negative test did not fail: {label}")

    expect_failure(
        "restored token drift",
        lambda c, *_: mutate_case_field(c, "disk_seed_payload", "restored_tokens", 551),
    )
    expect_failure(
        "payload byte drift",
        lambda c, *_: mutate_case_field(c, "disk_continuation_payload", "payload_bytes", 1),
    )
    expect_failure(
        "shutdown reason drift",
        lambda c, *_: mutate_case_nested_field(c, "snapshot_seed", "shutdown_write_header", "reason", 1),
    )
    expect_failure(
        "continued skip drift",
        lambda c, *_: mutate_case_nested_field(c, "disk_seed_payload", "current_live_skip", "target", 550),
    )
    expect_failure(
        "already-stored skip drift",
        lambda c, *_: mutate_case_nested_field(
            c,
            "snapshot_continuation",
            "already_stored_boundary",
            "target",
            10240,
        ),
    )
    expect_failure(
        "runtime restored skip drift",
        lambda _c, _r, _f, runtime, *_: mutate_runtime_maybe_store_success(runtime),
    )
    expect_failure(
        "frontier projection drift",
        lambda _c, restore, *_: mutate_restore_projection(
            restore,
            "disk_seed_payload",
            "next_continued_store",
            "target",
            0,
        ),
    )
    expect_failure(
        "KVC fixed header drift",
        lambda _c, _r, _f, _runtime, _ledger, kvc: kvc["constants"].__setitem__("fixed_header", 52),
    )
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--restore-summary", type=Path, default=DEFAULT_RESTORE_SUMMARY)
    parser.add_argument("--frontier-contract", type=Path, default=DEFAULT_FRONTIER_CONTRACT)
    parser.add_argument("--runtime-summary", type=Path, default=DEFAULT_RUNTIME_SUMMARY)
    parser.add_argument("--runtime-ledger-contract", type=Path, default=DEFAULT_RUNTIME_LEDGER_CONTRACT)
    parser.add_argument("--kvc-file-oracle", type=Path, default=DEFAULT_KVC_FILE_ORACLE)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        contract = load_json(args.contract)
        restore_summary = load_json(args.restore_summary)
        frontier_contract = load_json(args.frontier_contract)
        runtime_summary = load_json(args.runtime_summary)
        runtime_ledger_contract = load_json(args.runtime_ledger_contract)
        kvc_file_oracle = load_json(args.kvc_file_oracle)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    errors = validate(
        contract,
        restore_summary,
        frontier_contract,
        runtime_summary,
        runtime_ledger_contract,
        kvc_file_oracle,
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.negative_test:
        negative_errors = run_negative_tests(
            contract,
            restore_summary,
            frontier_contract,
            runtime_summary,
            runtime_ledger_contract,
            kvc_file_oracle,
        )
        if negative_errors:
            for error in negative_errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
    print("post-restore KVC decision contract: PASS, 4 cases, 3 runtime references")
    if args.negative_test:
        print("post-restore KVC decision contract negative tests: PASS, 8 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
