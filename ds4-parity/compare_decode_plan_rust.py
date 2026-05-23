#!/usr/bin/env python3
"""Compare Rust M10.5b decode planning against the current-C oracle."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "ds4-parity/baselines/graph/m10.5b/decode-plan-oracle.json"
DEFAULT_RUST = ROOT / "rust/ds4-gpu/src/decode_plan.rs"


CASE_KEYS = [
    "name",
    "ctx_size",
    "prompt_len",
    "mtp_enabled",
    "pos",
    "need_logits",
    "allow_split_flush",
    "split_after_layers",
    "raw_window",
    "raw_cap",
    "raw_row",
    "n_raw",
    "raw_start",
    "flush_after_layer",
    "dense_layers",
    "ratio4_layers",
    "ratio128_layers",
    "ratio4_comp_before",
    "ratio4_comp_after",
    "ratio4_emit_layers",
    "ratio4_indexed_layers",
    "ratio128_comp_before",
    "ratio128_comp_after",
    "ratio128_emit_layers",
]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    oracle = load_json(args.oracle)
    rust_source = read_text(args.rust_source)
    if args.negative_test:
        return run_negative_tests(oracle, rust_source)

    errors = validate(oracle, rust_source)
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    elif errors:
        print_errors(errors)
    else:
        print(
            "Rust decode plan comparator: "
            f"{len(oracle.get('cases', []))} cases, "
            f"{len(oracle.get('token_stage_order', []))} token stages, "
            f"{len(oracle.get('layer_stage_order', []))} layer stages"
        )
    return 1 if errors else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--rust-source", type=Path, default=DEFAULT_RUST)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def validate(oracle: dict[str, Any], rust_source: str) -> list[str]:
    errors: list[str] = []
    expect_value(errors, oracle, "schema", "ds4.decode_plan.v1")

    compare_list(
        errors,
        "token_stage_order",
        expect_string_list(errors, oracle, "token_stage_order"),
        rust_string_array(rust_source, "DECODE_TOKEN_STAGE_ORDER"),
    )
    compare_list(
        errors,
        "layer_stage_order",
        expect_string_list(errors, oracle, "layer_stage_order"),
        rust_string_array(rust_source, "DECODE_LAYER_STAGE_ORDER"),
    )

    expected_cases = expected_case_map(errors, oracle)
    rust_cases = rust_case_map(rust_source)
    compare_case_maps(errors, expected_cases, rust_cases)
    return errors


def expect_value(errors: list[str], obj: dict[str, Any], key: str, expected: Any) -> None:
    got = obj.get(key)
    if got != expected:
        errors.append(f"{key}: expected {expected!r}, got {got!r}")


def expect_string_list(errors: list[str], obj: dict[str, Any], key: str) -> list[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{key}: expected string list")
        return []
    return value


def rust_string_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"pub const {re.escape(name)}:.*?=\s*&\[(.*?)\];",
        source,
        flags=re.S,
    )
    if match is None:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def expected_case_map(errors: list[str], oracle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = oracle.get("cases")
    if not isinstance(cases, list):
        errors.append("cases: expected list")
        return {}
    out: dict[str, dict[str, Any]] = {}
    layer_counts = oracle.get("layer_counts", {})
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"cases[{index}]: expected object")
            continue
        name = case.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"cases[{index}].name: expected non-empty string")
            continue
        normalized = normalize_case(case, layer_counts)
        if name in out:
            errors.append(f"cases: duplicate {name!r}")
        out[name] = normalized
    return out


def normalize_case(case: dict[str, Any], layer_counts: Any) -> dict[str, Any]:
    out = {key: case.get(key) for key in CASE_KEYS if key != "name"}
    if isinstance(layer_counts, dict):
        out["dense_layers"] = layer_counts.get("dense")
        out["ratio4_layers"] = layer_counts.get("ratio4")
        out["ratio128_layers"] = layer_counts.get("ratio128")
    return out


def rust_case_map(source: str) -> dict[str, dict[str, Any]]:
    match = re.search(
        r"pub const M105B_DECODE_CASE_ORACLE:.*?=\s*&\[(.*?)\];",
        source,
        flags=re.S,
    )
    if match is None:
        return {}
    body = match.group(1)
    case_pattern = re.compile(
        r'case!\(\s*"([^"]+)"\s*,'
        + r"\s*([0-9]+)\s*," * 2
        + r"\s*(true|false)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*(true|false)\s*,"
        + r"\s*(true|false)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*(Some\([0-9]+\)|None)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*,"
        + r"\s*([0-9]+)\s*\)",
        flags=re.S,
    )
    out: dict[str, dict[str, Any]] = {}
    for fields in case_pattern.findall(body):
        values = list(fields)
        name = values[0]
        parsed: dict[str, Any] = {}
        for key, value in zip(CASE_KEYS[1:], values[1:], strict=True):
            if value == "true":
                parsed[key] = True
            elif value == "false":
                parsed[key] = False
            elif value == "None":
                parsed[key] = None
            elif value.startswith("Some("):
                parsed[key] = int(value[5:-1])
            else:
                parsed[key] = int(value)
        out[name] = parsed
    return out


def compare_list(errors: list[str], label: str, expected: list[str], got: list[str]) -> None:
    if expected != got:
        errors.append(f"{label}: expected {expected!r}, got {got!r}")


def compare_case_maps(
    errors: list[str],
    expected: dict[str, dict[str, Any]],
    got: dict[str, dict[str, Any]],
) -> None:
    for name in sorted(expected.keys() - got.keys()):
        errors.append(f"missing Rust decode plan case {name}")
    for name in sorted(got.keys() - expected.keys()):
        errors.append(f"unexpected Rust decode plan case {name}")
    for name in sorted(expected.keys() & got.keys()):
        if expected[name] != got[name]:
            errors.append(f"{name}: expected {expected[name]!r}, got {got[name]!r}")


def run_negative_tests(oracle: dict[str, Any], rust_source: str) -> int:
    baseline_errors = validate(oracle, rust_source)
    if baseline_errors:
        print_errors(baseline_errors)
        return 1

    mutated_oracle = copy.deepcopy(oracle)
    mutated_oracle["cases"][3]["ratio4_indexed_layers"] = 0
    tests = [
        ("missing token stage", oracle, rust_source.replace('"split_flush",\n', "", 1)),
        ("rust raw-start drift", oracle, rust_source.replace("922,\n        Some(3),", "923,\n        Some(3),", 1)),
        ("oracle indexed-layer drift", mutated_oracle, rust_source),
    ]
    unexpected_passes: list[str] = []
    for label, test_oracle, test_source in tests:
        errors = validate(test_oracle, test_source)
        if errors:
            print(f"negative {label}: detected {len(errors)} decode-plan error(s)")
        else:
            print(f"negative {label}: mutation was not detected")
            unexpected_passes.append(label)
    return 1 if unexpected_passes else 0


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
