#!/usr/bin/env python3
"""Compare Rust ds4-gpu-sys declarations against the graph ABI oracle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
DEFAULT_HEADER = ROOT / "ds4_gpu.h"
DEFAULT_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"


@dataclass(frozen=True)
class AbiSignature:
    return_type: str
    params: tuple[str, ...]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    oracle = load_json(args.oracle)
    header_source = read_text(args.header)
    rust_source = read_text(args.rust_sys)

    if args.negative_test:
        return run_negative_tests(oracle, header_source, rust_source)

    errors = validate(oracle, header_source, rust_source)
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    elif errors:
        print_errors(errors)
    else:
        names = expected_operations(oracle)
        c_sigs = parse_c_header(header_source)
        total_params = sum(len(c_sigs[name].params) for name in names)
        print(
            "Rust GPU sys ABI comparator: "
            f"{len(names)} declarations, {total_params} parameters"
        )
    return 1 if errors else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--rust-sys", type=Path, default=DEFAULT_SYS)
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


def validate(oracle: dict[str, Any], header_source: str, rust_source: str) -> list[str]:
    errors: list[str] = []
    expected = expected_operations(oracle)
    c_sigs, c_duplicates = parse_c_header_with_duplicates(header_source)
    rust_sigs, rust_duplicates = parse_rust_sys_with_duplicates(rust_source)

    for name in sorted(c_duplicates):
        errors.append(f"duplicate C declaration for {name}")
    for name in sorted(rust_duplicates):
        errors.append(f"duplicate Rust sys declaration for {name}")

    expected_set = set(expected)
    for name in sorted(rust_sigs):
        if name not in expected_set:
            errors.append(f"unexpected Rust sys declaration outside M10.2 oracle: {name}")

    for name in expected:
        c_sig = c_sigs.get(name)
        rust_sig = rust_sigs.get(name)
        if c_sig is None:
            errors.append(f"missing C header declaration for oracle operation {name}")
            continue
        if rust_sig is None:
            errors.append(f"missing Rust sys declaration for {name}")
            continue
        if rust_sig.return_type != c_sig.return_type:
            errors.append(
                f"{name}: return type mismatch: C {c_sig.return_type}, "
                f"Rust {rust_sig.return_type}"
            )
        if rust_sig.params != c_sig.params:
            errors.append(
                f"{name}: parameter types mismatch: C {list(c_sig.params)}, "
                f"Rust {list(rust_sig.params)}"
            )
    return errors


def expected_operations(oracle: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group in oracle.get("operation_groups", []):
        for operation in group.get("operations", []):
            if not isinstance(operation, str):
                raise SystemExit("operation_groups contains a non-string operation")
            names.append(operation)
    if len(names) != len(set(names)):
        raise SystemExit("operation_groups contains duplicate operations")
    return names


def parse_c_header(source: str) -> dict[str, AbiSignature]:
    sigs, _duplicates = parse_c_header_with_duplicates(source)
    return sigs


def parse_c_header_with_duplicates(source: str) -> tuple[dict[str, AbiSignature], set[str]]:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    pattern = re.compile(
        r"(?P<ret>(?:[A-Za-z_][A-Za-z0-9_]*|void)\s*(?:\*)?)\s*"
        r"(?P<name>ds4_gpu_[A-Za-z0-9_]+)\s*"
        r"\((?P<params>.*?)\);",
        flags=re.S,
    )
    sigs: dict[str, AbiSignature] = {}
    duplicates: set[str] = set()
    for match in pattern.finditer(source):
        name = match.group("name")
        if name in sigs:
            duplicates.add(name)
        params: list[str] = []
        for param in split_c_params(match.group("params")):
            normalized = normalize_c_param(param)
            if normalized is not None:
                params.append(normalized)
        sigs[name] = AbiSignature(
            return_type=normalize_c_type(match.group("ret")),
            params=tuple(params),
        )
    return sigs, duplicates


def split_c_params(params: str) -> list[str]:
    params = params.strip()
    if not params or params == "void":
        return []
    return [param.strip() for param in params.split(",")]


def normalize_c_param(param: str) -> str | None:
    compact = compact_pointer_type(param)
    if compact == "void":
        return None
    tokens = compact.split()
    if len(tokens) < 2:
        raise SystemExit(f"cannot parse C parameter: {param!r}")
    return normalize_c_type(" ".join(tokens[:-1]))


def normalize_c_type(type_text: str) -> str:
    compact = compact_pointer_type(type_text)
    c_to_rust = {
        "void": "void",
        "int": "c_int",
        "uint64_t": "u64",
        "uint32_t": "u32",
        "float": "f32",
        "bool": "bool",
        "bool *": "*mut bool",
        "void *": "*mut c_void",
        "const void *": "*const c_void",
        "const char *": "*const c_char",
        "ds4_gpu_tensor *": "*mut Ds4GpuTensor",
        "const ds4_gpu_tensor *": "*const Ds4GpuTensor",
    }
    try:
        return c_to_rust[compact]
    except KeyError as exc:
        raise SystemExit(f"unsupported C ABI type: {type_text!r}") from exc


def parse_rust_sys_with_duplicates(source: str) -> tuple[dict[str, AbiSignature], set[str]]:
    source = "\n".join(extract_rust_extern_blocks(source))
    pattern = re.compile(
        r"pub\s+fn\s+(?P<name>ds4_gpu_[A-Za-z0-9_]+)\s*"
        r"\((?P<params>.*?)\)\s*"
        r"(?:->\s*(?P<ret>[^;]+))?;",
        flags=re.S,
    )
    sigs: dict[str, AbiSignature] = {}
    duplicates: set[str] = set()
    for match in pattern.finditer(source):
        name = match.group("name")
        if name in sigs:
            duplicates.add(name)
        sigs[name] = AbiSignature(
            return_type=normalize_rust_type(match.group("ret") or "void"),
            params=tuple(parse_rust_params(match.group("params"))),
        )
    return sigs, duplicates


def extract_rust_extern_blocks(source: str) -> list[str]:
    return re.findall(r'unsafe\s+extern\s+"C"\s*\{(.*?)\n\}', source, flags=re.S)


def parse_rust_params(params: str) -> list[str]:
    out: list[str] = []
    for raw in params.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"cannot parse Rust parameter: {item!r}")
        _name, type_text = item.split(":", 1)
        out.append(normalize_rust_type(type_text))
    return out


def normalize_rust_type(type_text: str) -> str:
    return compact_pointer_type(type_text)


def compact_pointer_type(type_text: str) -> str:
    compact = " ".join(type_text.replace("*", " * ").split())
    return compact.replace("* const", "*const").replace("* mut", "*mut")


def run_negative_tests(oracle: dict[str, Any], header_source: str, rust_source: str) -> int:
    baseline_errors = validate(oracle, header_source, rust_source)
    if baseline_errors:
        print_errors(baseline_errors)
        return 1

    tests = [
        (
            "missing rust declaration",
            header_source,
            remove_rust_function(rust_source, "ds4_gpu_attention_decode_heads_tensor"),
        ),
        (
            "rust parameter type drift",
            header_source,
            mutate_rust_function(
                rust_source,
                "ds4_gpu_attention_decode_heads_tensor",
                "raw_start: u32,",
                "raw_start: u64,",
            ),
        ),
        (
            "header parameter type drift",
            header_source.replace(
                "uint32_t                raw_start,",
                "uint64_t                raw_start,",
                1,
            ),
            rust_source,
        ),
    ]

    unexpected_passes: list[str] = []
    for label, header, rust in tests:
        errors = validate(oracle, header, rust)
        if errors:
            print(f"negative {label}: detected {len(errors)} ABI error(s)")
        else:
            print(f"negative {label}: mutation was not detected")
            unexpected_passes.append(label)
    return 1 if unexpected_passes else 0


def remove_rust_function(source: str, name: str) -> str:
    pattern = re.compile(
        rf"\n\s*pub\s+fn\s+{re.escape(name)}\s*\(.*?\)\s*(?:->\s*[^;]+)?;\n",
        flags=re.S,
    )
    mutated, count = pattern.subn("\n", source, count=1)
    if count != 1:
        raise SystemExit(f"failed to remove Rust function {name}")
    return mutated


def mutate_rust_function(source: str, name: str, before: str, after: str) -> str:
    pattern = re.compile(
        rf"(?P<fn>\n\s*pub\s+fn\s+{re.escape(name)}\s*\(.*?\)\s*(?:->\s*[^;]+)?;\n)",
        flags=re.S,
    )
    match = pattern.search(source)
    if match is None:
        raise SystemExit(f"failed to find Rust function {name}")
    block = match.group("fn")
    mutated = block.replace(before, after, 1)
    if mutated == block:
        raise SystemExit(f"failed to mutate Rust function {name}")
    return source[: match.start("fn")] + mutated + source[match.end("fn") :]


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
