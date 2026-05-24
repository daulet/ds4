#!/usr/bin/env python3
"""Validate the M9.8f5 Rust runtime KV replay summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "ds4-parity/baselines/kv/m9.8f5/runtime-rust-b300-summary.json"
DEFAULT_LEDGER_CONTRACT = (
    ROOT / "ds4-parity/baselines/kv/m10.7d2/runtime-ledger-contract.json"
)

EXPECTED_CASES: dict[str, dict[str, Any]] = {
    "seed_miss": {
        "fixture": "ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json",
        "finish": "length",
        "content": "I notice",
        "prompt_tokens": 550,
        "cached_tokens": 0,
        "cache_write_tokens": 550,
        "cache_source": "none",
        "disk_cached_tokens": 0,
    },
    "seed_restore": {
        "fixture": "ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json",
        "finish": "length",
        "content": "I notice",
        "prompt_tokens": 550,
        "cached_tokens": 550,
        "cache_write_tokens": 0,
        "cache_source": "disk-text",
        "disk_cached_tokens": 550,
        "disk_cache_file": "0ab2314538b11686a11e296b7f697651fbd17e60.kv",
    },
    "continuation_restore": {
        "fixture": "ds4-parity/baselines/kv-fixtures/m0.5/kv_continuation.json",
        "finish": "stop",
        "content": "kv continued",
        "prompt_tokens": 561,
        "cached_tokens": 552,
        "cache_write_tokens": 9,
        "cache_source": "disk-text",
        "disk_cached_tokens": 552,
        "disk_cache_file": "a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv",
    },
}

EXPECTED_FLAGS = {
    "ctx": 32768,
    "tokens": 16,
    "kv_disk_space_mb": 512,
    "kv_cache_min_tokens": 512,
    "kv_cache_cold_max_tokens": 30000,
    "kv_cache_continued_interval_tokens": 0,
}

EXPECTED_KV_HEADERS: dict[str, dict[str, Any]] = {
    "0ab2314538b11686a11e296b7f697651fbd17e60.kv": {
        "reason": 1,
        "reason_name": "cold",
        "tokens": 550,
        "hits": 1,
        "ctx": 32768,
        "payload_bytes": 31526948,
        "rendered_text_bytes": 2520,
        "size_bytes": 31529520,
    },
    "4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.kv": {
        "reason": 4,
        "reason_name": "shutdown",
        "tokens": 563,
        "hits": 0,
        "ctx": 32768,
        "payload_bytes": 31688280,
        "rendered_text_bytes": 2632,
        "size_bytes": 31690964,
    },
    "a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv": {
        "reason": 4,
        "reason_name": "shutdown",
        "tokens": 552,
        "hits": 1,
        "ctx": 32768,
        "payload_bytes": 31580716,
        "rendered_text_bytes": 2528,
        "size_bytes": 31583296,
    },
}

EXPECTED_LEDGER_CASES: dict[str, dict[str, Any]] = {
    "seed_miss": {
        "fixture": "ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json",
        "cache_source": "none",
        "prompt_tokens": 550,
        "cached_tokens": 0,
        "cache_write_tokens": 550,
        "disk_cached_tokens": 0,
        "disk_cache_file": None,
        "kv_write_file": "0ab2314538b11686a11e296b7f697651fbd17e60.kv",
        "kv_write_reason_name": "cold",
        "kv_write_tokens": 550,
        "events": [
            {
                "name": "reset_continued_frontier",
                "tokens": 0,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": True,
            },
            {
                "name": "cache_decision",
                "cache_source": "none",
                "tokens": 550,
                "cached_tokens": 0,
                "cache_write_tokens": 550,
                "disk_cached_tokens": 0,
                "frontier_before": 0,
                "frontier_after": 0,
            },
            {
                "name": "suppress_continued_store",
                "tokens": 550,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": False,
            },
            {
                "name": "maybe_store_continued",
                "reason": "continued",
                "tokens": 550,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": False,
            },
            {
                "name": "store_live_prefix",
                "reason": "cold",
                "tokens": 550,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": True,
            },
            {
                "name": "note_store",
                "tokens": 550,
                "frontier_before": 0,
                "frontier_after": 550,
                "success": True,
            },
        ],
    },
    "seed_restore": {
        "fixture": "ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json",
        "cache_source": "disk-text",
        "prompt_tokens": 550,
        "cached_tokens": 550,
        "cache_write_tokens": 0,
        "disk_cached_tokens": 550,
        "disk_cache_file": "0ab2314538b11686a11e296b7f697651fbd17e60.kv",
        "kv_write_file": None,
        "kv_write_reason_name": None,
        "kv_write_tokens": None,
        "events": [
            {
                "name": "reset_continued_frontier",
                "tokens": 0,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": True,
            },
            {
                "name": "cache_decision",
                "cache_source": "disk-text",
                "tokens": 550,
                "cached_tokens": 550,
                "cache_write_tokens": 0,
                "disk_cached_tokens": 550,
                "frontier_before": 550,
                "frontier_after": 550,
            },
            {
                "name": "maybe_store_continued",
                "reason": "continued",
                "tokens": 550,
                "frontier_before": 550,
                "frontier_after": 550,
                "success": False,
            },
        ],
    },
    "continuation_restore": {
        "fixture": "ds4-parity/baselines/kv-fixtures/m0.5/kv_continuation.json",
        "cache_source": "disk-text",
        "prompt_tokens": 561,
        "cached_tokens": 552,
        "cache_write_tokens": 9,
        "disk_cached_tokens": 552,
        "disk_cache_file": "a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv",
        "kv_write_file": None,
        "kv_write_reason_name": None,
        "kv_write_tokens": None,
        "events": [
            {
                "name": "reset_continued_frontier",
                "tokens": 0,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": True,
            },
            {
                "name": "cache_decision",
                "cache_source": "disk-text",
                "tokens": 561,
                "cached_tokens": 552,
                "cache_write_tokens": 9,
                "disk_cached_tokens": 552,
                "frontier_before": 552,
                "frontier_after": 552,
            },
            {
                "name": "maybe_store_continued",
                "reason": "continued",
                "tokens": 561,
                "frontier_before": 552,
                "frontier_after": 552,
                "success": False,
            },
        ],
    },
    "memory_token_continuation": {
        "fixture": "ds4-parity/baselines/server-fixtures/m0.4/chat_cache_continuation.json",
        "cache_source": "memory-token",
        "prompt_tokens": 50,
        "cached_tokens": 41,
        "cache_write_tokens": 9,
        "disk_cached_tokens": 0,
        "disk_cache_file": None,
        "kv_write_file": None,
        "kv_write_reason_name": None,
        "kv_write_tokens": None,
        "events": [
            {
                "name": "cache_decision",
                "cache_source": "memory-token",
                "tokens": 50,
                "cached_tokens": 41,
                "cache_write_tokens": 9,
                "disk_cached_tokens": 0,
                "frontier_before": 0,
                "frontier_after": 0,
            },
            {
                "name": "maybe_store_continued",
                "reason": "continued",
                "tokens": 50,
                "frontier_before": 0,
                "frontier_after": 0,
                "success": False,
            },
        ],
    },
}

EXPECTED_B300_LEDGER_CASE_NAMES = (
    "seed_miss",
    "seed_restore",
    "continuation_restore",
)
EXPECTED_B300_LEDGER_CASES: dict[str, dict[str, Any]] = {
    name: EXPECTED_LEDGER_CASES[name] for name in EXPECTED_B300_LEDGER_CASE_NAMES
}
EXPECTED_B300_TRACE_EVENT_COUNTS = {
    "seed_miss": 8,
    "seed_restore": 5,
    "continuation_restore": 6,
}
EXPECTED_B300_TRACE_EVENT_NAMES = {
    "seed_miss": [
        "reset_continued_frontier",
        "cache_decision",
        "suppress_continued_store",
        "maybe_store_continued",
        "store_live_prefix",
        "note_store",
        "maybe_store_continued",
        "maybe_store_continued",
    ],
    "seed_restore": [
        "reset_continued_frontier",
        "cache_decision",
        "maybe_store_continued",
        "maybe_store_continued",
        "maybe_store_continued",
    ],
    "continuation_restore": [
        "reset_continued_frontier",
        "cache_decision",
        "maybe_store_continued",
        "maybe_store_continued",
        "maybe_store_continued",
        "maybe_store_continued",
    ],
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def validate(summary_path: Path, ledger_contract_path: Path) -> list[str]:
    errors: list[str] = []
    data = load_json(summary_path, errors)
    ledger_contract = load_json(ledger_contract_path, errors)
    if not isinstance(data, dict):
        return errors or ["summary root is not an object"]
    if not isinstance(ledger_contract, dict):
        return errors or ["ledger contract root is not an object"]
    validate_data(errors, data, ledger_contract, summary_path)
    return errors


def validate_data(
    errors: list[str],
    data: dict[str, Any],
    ledger_contract: dict[str, Any],
    summary_path: Path,
) -> None:
    expect_value(errors, data, "schema", "ds4.runtime_kv_replay_summary.v1")
    expect_value(errors, data, "milestone", "M9.8f5")
    expect_value(errors, data, "source", "rust-runtime-b300-replay")

    b300 = expect_object(errors, data, "b300")
    if b300:
        expect_value(errors, b300, "kube_context", "hou2-prod1")
        expect_value(errors, b300, "pod", "ds4-rust-port-b300")
        expect_value(errors, b300, "workdir", "/workspace/ds4")
        expect_value(
            errors,
            b300,
            "model",
            "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
        )

    flags = expect_object(errors, data, "server_flags")
    if flags:
        for key, expected in EXPECTED_FLAGS.items():
            expect_value(errors, flags, key, expected)

    cases = expect_named_objects(errors, data, "cases", "name")
    if cases:
        check_named_records(errors, "cases", cases, EXPECTED_CASES)

    headers = expect_named_objects(errors, data, "kv_headers", "file")
    if headers:
        check_named_records(errors, "kv_headers", headers, EXPECTED_KV_HEADERS)

    summary_ledger_cases = expect_named_objects(errors, data, "ledger_cases", "name")
    if summary_ledger_cases:
        check_named_records(
            errors,
            "summary_ledger_cases",
            summary_ledger_cases,
            EXPECTED_B300_LEDGER_CASES,
        )
        validate_ledger_cases_against_summary(
            errors,
            "summary_ledger_cases",
            summary_ledger_cases,
            cases,
            headers,
            EXPECTED_B300_LEDGER_CASE_NAMES,
        )
        validate_summary_ledger_trace_metadata(errors, summary_ledger_cases)

    validate_ledger_contract(errors, ledger_contract, summary_path, cases, headers)


def validate_ledger_contract(
    errors: list[str],
    contract: dict[str, Any],
    summary_path: Path,
    summary_cases: dict[str, dict[str, Any]],
    headers: dict[str, dict[str, Any]],
) -> None:
    expect_value(errors, contract, "schema", "ds4.runtime_kv_replay_ledger_contract.v1")
    expect_value(errors, contract, "milestone", "M10.7d2b")
    expect_value(errors, contract, "source", "model-free-runtime-ledger-contract")
    expect_value(errors, contract, "summary_path", rel(summary_path))
    cases = expect_named_objects(errors, contract, "cases", "name")
    if not cases:
        return
    check_exact_named_records(errors, "ledger_cases", cases, EXPECTED_LEDGER_CASES)
    validate_ledger_cases_against_summary(
        errors,
        "ledger_cases",
        cases,
        summary_cases,
        headers,
        EXPECTED_B300_LEDGER_CASE_NAMES,
    )


def validate_ledger_cases_against_summary(
    errors: list[str],
    label: str,
    ledger_cases: dict[str, dict[str, Any]],
    summary_cases: dict[str, dict[str, Any]],
    headers: dict[str, dict[str, Any]],
    case_names: tuple[str, ...],
) -> None:
    for name in case_names:
        expected = ledger_cases.get(name)
        actual = summary_cases.get(name)
        if expected is None or actual is None:
            continue
        for key in (
            "fixture",
            "cache_source",
            "prompt_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "disk_cached_tokens",
            "disk_cache_file",
        ):
            if expected.get(key) != actual.get(key):
                errors.append(
                    f"{label}.{name}.{key}: expected summary value "
                    f"{actual.get(key)!r}, got {expected.get(key)!r}"
                )
    for name, case in ledger_cases.items():
        write_file = case.get("kv_write_file")
        if write_file is None:
            continue
        header = headers.get(write_file)
        if header is None:
            errors.append(f"{label}.{name}.kv_write_file: missing header {write_file!r}")
            continue
        if case.get("kv_write_reason_name") != header.get("reason_name"):
            errors.append(f"{label}.{name}.kv_write_reason_name drift")
        if case.get("kv_write_tokens") != header.get("tokens"):
            errors.append(f"{label}.{name}.kv_write_tokens drift")


def validate_summary_ledger_trace_metadata(
    errors: list[str],
    ledger_cases: dict[str, dict[str, Any]],
) -> None:
    for name in EXPECTED_B300_LEDGER_CASE_NAMES:
        case = ledger_cases.get(name)
        if case is None:
            continue
        expect_value(errors, case, "trace_file", f"traces/{name}.trace")
        expect_value(
            errors,
            case,
            "trace_event_count",
            EXPECTED_B300_TRACE_EVENT_COUNTS[name],
        )
        expect_value(
            errors,
            case,
            "trace_event_names",
            EXPECTED_B300_TRACE_EVENT_NAMES[name],
        )


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        errors.append(f"failed to read {path}: {exc}")
    except json.JSONDecodeError as exc:
        errors.append(f"failed to parse {path}: {exc}")
    return None


def expect_object(errors: list[str], data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if isinstance(value, dict):
        return value
    errors.append(f"{key}: expected object")
    return {}


def expect_value(
    errors: list[str],
    data: dict[str, Any],
    key: str,
    expected: Any,
) -> None:
    actual = data.get(key)
    if actual != expected:
        errors.append(f"{key}: expected {expected!r}, got {actual!r}")


def expect_named_objects(
    errors: list[str],
    data: dict[str, Any],
    key: str,
    name_key: str,
) -> dict[str, dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        errors.append(f"{key}: expected list")
        return {}
    named: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{key}[{index}]: expected object")
            continue
        name = item.get(name_key)
        if not isinstance(name, str) or not name:
            errors.append(f"{key}[{index}].{name_key}: expected non-empty string")
            continue
        if name in named:
            errors.append(f"{key}: duplicate {name_key} {name!r}")
            continue
        named[name] = item
    return named


def check_named_records(
    errors: list[str],
    label: str,
    actual: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> None:
    actual_names = set(actual)
    expected_names = set(expected)
    for name in sorted(expected_names - actual_names):
        errors.append(f"{label}: missing {name!r}")
    for name in sorted(actual_names - expected_names):
        errors.append(f"{label}: unexpected {name!r}")
    for name in sorted(expected_names & actual_names):
        record = actual[name]
        for key, expected_value in expected[name].items():
            actual_value = record.get(key)
            if actual_value != expected_value:
                errors.append(
                    f"{label}.{name}.{key}: expected {expected_value!r}, "
                    f"got {actual_value!r}"
                )


def check_exact_named_records(
    errors: list[str],
    label: str,
    actual: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> None:
    check_named_records(errors, label, actual, expected)
    for name, expected_record in expected.items():
        record = actual.get(name)
        if record is None:
            continue
        expected_keys = set(expected_record) | {"name"}
        if set(record) != expected_keys:
            errors.append(
                f"{label}.{name}: expected keys {sorted(expected_keys)}, "
                f"got {sorted(record)}"
            )


def run_negative_tests(summary_path: Path, ledger_contract_path: Path) -> list[str]:
    errors: list[str] = []
    load_errors: list[str] = []
    summary = load_json(summary_path, load_errors)
    contract = load_json(ledger_contract_path, load_errors)
    if load_errors:
        return load_errors
    if not isinstance(summary, dict) or not isinstance(contract, dict):
        return ["negative tests require object summary and ledger contract"]

    def expect_failure(label: str, mutator: Any) -> None:
        mutated_summary = copy_json(summary)
        mutated_contract = copy_json(contract)
        mutator(mutated_summary, mutated_contract)
        trial_errors: list[str] = []
        validate_data(trial_errors, mutated_summary, mutated_contract, summary_path)
        if not trial_errors:
            errors.append(f"negative test did not fail: {label}")

    expect_failure(
        "missing ledger event",
        lambda _summary, contract: contract["cases"][0]["events"].pop(),
    )
    expect_failure(
        "ledger event order drift",
        lambda _summary, contract: contract["cases"][0]["events"].reverse(),
    )
    expect_failure(
        "frontier transition drift",
        lambda _summary, contract: contract["cases"][0]["events"][2].__setitem__(
            "frontier_after", 550
        ),
    )
    expect_failure(
        "memory-token source drift",
        lambda _summary, contract: contract["cases"][3].__setitem__("cache_source", "none"),
    )
    expect_failure(
        "summary cross-check drift",
        lambda summary, _contract: summary["cases"][1].__setitem__("cached_tokens", 549),
    )
    expect_failure(
        "summary ledger event drift",
        lambda summary, _contract: summary["ledger_cases"][0]["events"][1].__setitem__(
            "cache_source", "disk-text"
        ),
    )
    return errors


def copy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="runtime replay summary JSON to validate",
    )
    parser.add_argument(
        "--ledger-contract",
        type=Path,
        default=DEFAULT_LEDGER_CONTRACT,
        help="runtime replay ledger contract JSON to validate",
    )
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="run checker self-tests that mutate summary and ledger fields",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    errors = validate(args.summary, args.ledger_contract)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.negative_test:
        negative_errors = run_negative_tests(args.summary, args.ledger_contract)
        if negative_errors:
            for error in negative_errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
    print(
        "summary: runtime KV replay summary passed, "
        "3 cases, 3 kv headers, 3 summary ledger cases, 4 contract ledger cases"
    )
    if args.negative_test:
        print("summary: runtime KV replay negative tests passed, 6 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
