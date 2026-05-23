#!/usr/bin/env python3
"""Compare Rust graph-plan inventory against the M10.2 current-C oracle."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
DEFAULT_RUST = ROOT / "rust/ds4-gpu/src/graph_plan.rs"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    oracle = load_json(args.oracle)
    rust_source = args.rust_source.read_text()
    if args.negative_test:
        return run_negative_tests(oracle, rust_source)

    errors = validate(oracle, rust_source)
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    elif errors:
        print_errors(errors)
    else:
        counts = inventory_counts(oracle)
        print(
            "Rust graph plan comparator: "
            f"{counts['operations']} operations, "
            f"{counts['tensor_fields']} tensor fields, "
            f"{counts['command_boundaries']} command boundaries, "
            f"{counts['graph_plan_cases']} plan cases"
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


def validate(oracle: dict[str, Any], rust_source: str) -> list[str]:
    errors: list[str] = []
    expected_ops = expected_operations(oracle)
    rust_ops = rust_operations(rust_source)
    compare_maps(errors, "operations", expected_ops, rust_ops)

    expected_fields = expected_tensor_fields(oracle)
    rust_fields = rust_tensor_fields(rust_source)
    compare_maps(errors, "tensor fields", expected_fields, rust_fields)

    expected_boundaries = expected_command_boundaries(oracle)
    rust_boundaries = rust_command_boundaries(rust_source)
    compare_boundary_maps(errors, expected_boundaries, rust_boundaries)

    expected_plan_cases = expected_graph_plan_cases(oracle)
    rust_plan_cases = rust_graph_plan_cases(rust_source)
    compare_plan_case_maps(errors, expected_plan_cases, rust_plan_cases)
    return errors


def expected_operations(oracle: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in oracle.get("operation_groups", []):
        facade = group["rust_facade"]
        for operation in group["operations"]:
            out[operation] = facade
    return out


def rust_operations(source: str) -> dict[str, str]:
    return {
        name: facade
        for name, facade in re.findall(
            r'op!\(\s*"([^"]+)"\s*,\s*([A-Za-z0-9_]+)\s*\)',
            source,
            flags=re.S,
        )
    }


def expected_tensor_fields(oracle: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in oracle.get("tensor_owner_groups", []):
        owner = group["owner"]
        for field in group["fields"]:
            out[field] = owner
    return out


def rust_tensor_fields(source: str) -> dict[str, str]:
    return {
        name: owner
        for name, owner in re.findall(
            r'field!\(\s*"([^"]+)"\s*,\s*([A-Za-z0-9_]+)\s*,',
            source,
            flags=re.S,
        )
    }


def expected_command_boundaries(oracle: dict[str, Any]) -> dict[str, tuple[str, int, bool]]:
    out: dict[str, tuple[str, int, bool]] = {}
    for boundary in oracle.get("command_boundaries", []):
        out[boundary["c_function"]] = (
            boundary["name"],
            boundary["begin_end_min"],
            boundary["synchronize_on_failure"],
        )
    return out


def rust_command_boundaries(source: str) -> dict[str, tuple[str, int, bool]]:
    out: dict[str, tuple[str, int, bool]] = {}
    for name, c_function, begin_end_min, sync in re.findall(
        r'boundary!\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*([0-9]+)\s*,\s*(true|false)\s*\)',
        source,
        flags=re.S,
    ):
        out[c_function] = (name, int(begin_end_min), sync == "true")
    return out


def expected_graph_plan_cases(oracle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        case["name"]: normalize_graph_plan_case(case)
        for case in oracle.get("graph_plan_cases", [])
        if isinstance(case, dict)
    }


def rust_graph_plan_cases(source: str) -> dict[str, dict[str, Any]]:
    match = re.search(
        r"pub const M102_PLAN_CASE_ORACLE:.*?=\s*&\[(.*?)\];",
        source,
        flags=re.S,
    )
    if match is None:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for fields in re.findall(
        r'case!\(\s*"([^"]+)"\s*,'
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*(true|false)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r"\s*([0-9]+)\s*,"
        r'\s*"([^"]+)"\s*\)',
        match.group(1),
        flags=re.S,
    ):
        (
            name,
            ctx_size,
            prompt_len,
            mtp_enabled,
            prefill_cap,
            raw_window,
            requested_raw_cap,
            allocated_raw_cap,
            comp_cap,
            dense_layers,
            ratio4_layers,
            ratio128_layers,
            dense_cap,
            ratio4_cap,
            ratio128_cap,
            ratio4_indexer_layers,
            mtp_tensor_group,
        ) = fields
        out[name] = {
            "ctx_size": int(ctx_size),
            "prompt_len": int(prompt_len),
            "mtp_enabled": mtp_enabled == "true",
            "prefill_cap": int(prefill_cap),
            "raw_window": int(raw_window),
            "requested_raw_cap": int(requested_raw_cap),
            "allocated_raw_cap": int(allocated_raw_cap),
            "comp_cap": int(comp_cap),
            "layer_counts": {
                "dense": int(dense_layers),
                "ratio4": int(ratio4_layers),
                "ratio128": int(ratio128_layers),
            },
            "layer_comp_cap_by_ratio": {
                "dense": int(dense_cap),
                "ratio4": int(ratio4_cap),
                "ratio128": int(ratio128_cap),
            },
            "ratio4_indexer_layers": int(ratio4_indexer_layers),
            "mtp_tensor_group": mtp_tensor_group,
        }
    return out


def normalize_graph_plan_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "ctx_size": int(case["ctx_size"]),
        "prompt_len": int(case["prompt_len"]),
        "mtp_enabled": bool(case["mtp_enabled"]),
        "prefill_cap": int(case["prefill_cap"]),
        "raw_window": int(case["raw_window"]),
        "requested_raw_cap": int(case["requested_raw_cap"]),
        "allocated_raw_cap": int(case["allocated_raw_cap"]),
        "comp_cap": int(case["comp_cap"]),
        "layer_counts": {
            "dense": int(case["layer_counts"]["dense"]),
            "ratio4": int(case["layer_counts"]["ratio4"]),
            "ratio128": int(case["layer_counts"]["ratio128"]),
        },
        "layer_comp_cap_by_ratio": {
            "dense": int(case["layer_comp_cap_by_ratio"]["dense"]),
            "ratio4": int(case["layer_comp_cap_by_ratio"]["ratio4"]),
            "ratio128": int(case["layer_comp_cap_by_ratio"]["ratio128"]),
        },
        "ratio4_indexer_layers": int(case["ratio4_indexer_layers"]),
        "mtp_tensor_group": str(case["mtp_tensor_group"]),
    }


def compare_maps(
    errors: list[str],
    label: str,
    expected: dict[str, str],
    actual: dict[str, str],
) -> None:
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"{label}: missing {name!r}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"{label}: unexpected {name!r}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            errors.append(
                f"{label}.{name}: expected target {expected[name]!r}, got {actual[name]!r}"
            )


def compare_boundary_maps(
    errors: list[str],
    expected: dict[str, tuple[str, int, bool]],
    actual: dict[str, tuple[str, int, bool]],
) -> None:
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"command boundaries: missing {name!r}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"command boundaries: unexpected {name!r}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            errors.append(
                f"command boundaries.{name}: expected {expected[name]!r}, got {actual[name]!r}"
            )


def compare_plan_case_maps(
    errors: list[str],
    expected: dict[str, dict[str, Any]],
    actual: dict[str, dict[str, Any]],
) -> None:
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"plan cases: missing {name!r}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"plan cases: unexpected {name!r}")
    for name in sorted(set(expected) & set(actual)):
        expected_flat = flatten_case(expected[name])
        actual_flat = flatten_case(actual[name])
        for key in sorted(expected_flat):
            if expected_flat[key] != actual_flat.get(key):
                errors.append(
                    f"plan cases.{name}.{key}: expected {expected_flat[key]!r}, "
                    f"got {actual_flat.get(key)!r}"
                )


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in case.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                out[f"{key}.{nested_key}"] = nested_value
        else:
            out[key] = value
    return out


def run_negative_tests(oracle: dict[str, Any], rust_source: str) -> int:
    positive = validate(oracle, rust_source)
    if positive:
        print_errors(positive)
        return 1

    mutations = [
        (
            "missing operation",
            lambda s: replace_once(
                s,
                '"ds4_gpu_attention_decode_heads_tensor"',
                '"ds4_gpu_attention_decode_heads_tensor_missing"',
            ),
        ),
        (
            "wrong operation facade",
            lambda s: replace_once(
                s,
                'op!("ds4_gpu_init", BackendLifecycle)',
                'op!("ds4_gpu_init", TensorBackend)',
            ),
        ),
        (
            "missing tensor field",
            lambda s: replace_once(s, '"layer_raw_cache"', '"layer_raw_cache_missing"'),
        ),
        ("wrong tensor owner", mutate_tensor_owner),
        (
            "missing command boundary",
            lambda s: replace_once(
                s,
                '"metal_graph_verify_decode2_exact"',
                '"metal_graph_verify_decode2_exact_missing"',
            ),
        ),
        (
            "wrong command boundary minimum",
            lambda s: replace_once(
                s,
                'boundary!("mtp_suffix_tops", "metal_graph_verify_suffix_tops", 2, true)',
                'boundary!("mtp_suffix_tops", "metal_graph_verify_suffix_tops", 1, true)',
            ),
        ),
        (
            "wrong command boundary sync",
            lambda s: replace_once(
                s,
                'boundary!("mtp_suffix_tops", "metal_graph_verify_suffix_tops", 2, true)',
                'boundary!("mtp_suffix_tops", "metal_graph_verify_suffix_tops", 2, false)',
            ),
        ),
        ("wrong plan scalar", mutate_plan_scalar),
        (
            "missing plan case",
            lambda s: replace_once(s, '"ctx32768_mtp_on"', '"ctx32768_mtp_on_missing"'),
        ),
    ]
    passed = 0
    for name, mutate in mutations:
        try:
            mutated = mutate(rust_source)
        except ValueError as exc:
            print(f"negative test {name!r}: {exc}", file=sys.stderr)
            continue
        errors = validate(copy.deepcopy(oracle), mutated)
        if errors:
            passed += 1
        else:
            print(f"negative test {name!r}: expected validation failure", file=sys.stderr)
    if passed != len(mutations):
        return 1
    print(f"Rust graph plan negative test: {passed} mutations failed as expected")
    return 0


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"mutation target not found: {old!r}")
    return source.replace(old, new, 1)


def mutate_tensor_owner(source: str) -> str:
    mutated, count = re.subn(
        r'(field!\(\s*"layer_raw_cache"\s*,\s*)GraphPersistentKvState',
        lambda match: f"{match.group(1)}GraphDecodeState",
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise ValueError("mutation target not found: layer_raw_cache owner")
    return mutated


def mutate_plan_scalar(source: str) -> str:
    mutated, count = re.subn(
        r'(case!\(\s*"ctx2048_mtp_off"\s*,\s*2048\s*,\s*2048\s*,'
        r"\s*false\s*,\s*2048\s*,\s*128\s*,\s*2048\s*,\s*2048\s*,\s*)514",
        lambda match: f"{match.group(1)}515",
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise ValueError("mutation target not found: ctx2048_mtp_off comp_cap")
    return mutated


def inventory_counts(oracle: dict[str, Any]) -> dict[str, int]:
    return {
        "operations": sum(len(group["operations"]) for group in oracle["operation_groups"]),
        "tensor_fields": sum(len(group["fields"]) for group in oracle["tensor_owner_groups"]),
        "command_boundaries": len(oracle["command_boundaries"]),
        "graph_plan_cases": len(oracle["graph_plan_cases"]),
    }


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"Rust graph plan comparator: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
