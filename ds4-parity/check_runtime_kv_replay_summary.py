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


def validate(summary_path: Path) -> list[str]:
    errors: list[str] = []
    data = load_json(summary_path, errors)
    if not isinstance(data, dict):
        return errors or ["summary root is not an object"]

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

    return errors


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="runtime replay summary JSON to validate",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    errors = validate(args.summary)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("summary: runtime KV replay summary passed, 3 cases, 3 kv headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
