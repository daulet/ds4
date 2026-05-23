#!/usr/bin/env python3
"""Compare the Rust decode backend facade against the default C decode path."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import compare_gpu_sys_abi


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
HEADER = ROOT / "ds4_gpu.h"
RUST_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
RUST_FACADE = ROOT / "rust/ds4-gpu/src/decode_backend.rs"
RUST_LIB = ROOT / "rust/ds4-gpu/src/lib.rs"
RUST_GPU_SRC = ROOT / "rust/ds4-gpu/src"


DEFAULT_DECODE_OPERATIONS = [
    "ds4_gpu_embed_token_hc_tensor",
    "ds4_gpu_rms_norm_plain_tensor",
    "ds4_gpu_matmul_f16_tensor",
    "ds4_gpu_hc_split_weighted_sum_norm_tensor",
    "ds4_gpu_rms_norm_weight_tensor",
    "ds4_gpu_matmul_q8_0_tensor",
    "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
    "ds4_gpu_head_rms_norm_tensor",
    "ds4_gpu_rope_tail_tensor",
    "ds4_gpu_kv_fp8_store_raw_tensor",
    "ds4_gpu_matmul_f16_pair_tensor",
    "ds4_gpu_compressor_update_tensor",
    "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
    "ds4_gpu_dsv4_indexer_qat_tensor",
    "ds4_gpu_indexer_score_one_tensor",
    "ds4_gpu_indexer_topk_tensor",
    "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
    "ds4_gpu_attention_decode_heads_tensor",
    "ds4_gpu_attention_output_low_q8_tensor",
    "ds4_gpu_matmul_q8_0_hc_expand_tensor",
    "ds4_gpu_router_select_tensor",
    "ds4_gpu_routed_moe_one_tensor",
    "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
    "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
    "ds4_gpu_output_hc_weights_tensor",
    "ds4_gpu_hc_weighted_sum_tensor",
]

EXISTING_DECODE_OPERATIONS = [
    "ds4_gpu_begin_commands",
    "ds4_gpu_flush_commands",
    "ds4_gpu_end_commands",
    "ds4_gpu_synchronize",
    "ds4_gpu_tensor_read",
    "ds4_gpu_tensor_view",
    "ds4_gpu_tensor_free",
]


@dataclass
class FacadeSpec:
    operation: str
    method: str
    tensor_args: list[str]


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


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


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


def validate(
    report: Report,
    oracle: dict[str, Any],
    header_source: str,
    rust_sys_source: str,
    rust_facade_source: str,
    rust_lib_source: str,
    rust_gpu_sources: dict[str, str],
    required_operations: list[str] = DEFAULT_DECODE_OPERATIONS,
) -> None:
    oracle_ops = expected_operations_from_oracle(oracle)
    for operation in required_operations + EXISTING_DECODE_OPERATIONS:
        report.check(operation in oracle_ops, f"{operation} missing from M10.2 oracle")

    abi_errors = compare_gpu_sys_abi.validate(oracle, header_source, rust_sys_source)
    report.check(not abi_errors, "M10.5a ABI comparator has errors")
    for error in abi_errors[:5]:
        report.errors.append(f"ABI: {error}")

    facade_specs = parse_facade_specs(rust_facade_source)
    facade_by_operation = {spec.operation: spec for spec in facade_specs}
    report.check(
        len(facade_by_operation) == len(facade_specs),
        "facade operation table contains duplicate operations",
    )
    compare_operation_list(
        report,
        "default decode facade operations",
        required_operations,
        [spec.operation for spec in facade_specs],
    )

    method_names: set[str] = set()
    for operation in required_operations:
        spec = facade_by_operation.get(operation)
        if spec is None:
            continue
        report.check(spec.method not in method_names, f"duplicate facade method {spec.method}")
        method_names.add(spec.method)
        signature_args = method_tensor_args(rust_facade_source, spec.method)
        report.check(
            signature_args == spec.tensor_args,
            f"{spec.method}: tensor args expected {spec.tensor_args!r}, got {signature_args!r}",
        )
        report.check(
            f"sys::{operation}(" in rust_facade_source,
            f"{spec.method}: missing raw sys call for {operation}",
        )

    existing_specs = parse_existing_specs(rust_facade_source)
    compare_operation_list(
        report,
        "existing decode operations",
        EXISTING_DECODE_OPERATIONS,
        [operation for operation, _wrapper in existing_specs],
    )
    for operation in EXISTING_DECODE_OPERATIONS:
        report.check(
            f"sys::{operation}(" in rust_lib_source,
            f"existing wrapper missing lib.rs sys call for {operation}",
        )

    for path, source in rust_gpu_sources.items():
        if path in {"lib.rs", "decode_backend.rs"}:
            continue
        report.check(
            "sys::ds4_gpu_" not in source,
            f"{path}: raw backend call outside facade/lifecycle module",
        )


def expected_operations_from_oracle(oracle: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for group in oracle.get("operation_groups", []):
        for operation in group.get("operations", []):
            if isinstance(operation, str):
                out.add(operation)
    return out


def parse_facade_specs(source: str) -> list[FacadeSpec]:
    body = const_array_body(source, "DEFAULT_DECODE_FACADE_OPERATIONS")
    specs: list[FacadeSpec] = []
    for item in re.findall(r"DecodeFacadeOperation\s*\{(.*?)\n\s*\},", body, flags=re.S):
        operation = require_string_field(item, "operation")
        method = require_string_field(item, "method")
        tensor_match = re.search(r"tensor_args:\s*&\[(.*?)\]", item, flags=re.S)
        if tensor_match is None:
            raise SystemExit(f"facade spec for {operation} missing tensor_args")
        tensor_args = re.findall(r'"([^"]+)"', tensor_match.group(1))
        specs.append(FacadeSpec(operation=operation, method=method, tensor_args=tensor_args))
    return specs


def parse_existing_specs(source: str) -> list[tuple[str, str]]:
    body = const_array_body(source, "EXISTING_DECODE_OPERATIONS")
    specs: list[tuple[str, str]] = []
    for item in re.findall(r"ExistingDecodeOperation\s*\{(.*?)\n\s*\},", body, flags=re.S):
        specs.append((require_string_field(item, "operation"), require_string_field(item, "wrapper")))
    return specs


def require_string_field(item: str, field: str) -> str:
    match = re.search(rf'{field}:\s*"([^"]+)"', item)
    if match is None:
        raise SystemExit(f"missing {field} in item: {item}")
    return match.group(1)


def const_array_body(source: str, name: str) -> str:
    match = re.search(
        rf"pub const {re.escape(name)}:.*?=\s*&\[(.*?)\n\];",
        source,
        flags=re.S,
    )
    if match is None:
        raise SystemExit(f"missing Rust const array {name}")
    return match.group(1)


def method_tensor_args(source: str, method: str) -> list[str]:
    params = method_params(source, method)
    out: list[str] = []
    for param in split_top_level_commas(params):
        param = param.strip()
        if not param or param == "self":
            continue
        if ":" not in param:
            continue
        name, ty = [part.strip() for part in param.split(":", 1)]
        if "TensorRef" in ty or "TensorMut" in ty:
            out.append(name)
    return out


def method_params(source: str, method: str) -> str:
    match = re.search(rf"pub fn {re.escape(method)}\s*\(", source)
    if match is None:
        raise SystemExit(f"missing Rust method {method}")
    start = match.end()
    depth = 1
    i = start
    while i < len(source):
        char = source[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start:i]
        i += 1
    raise SystemExit(f"unterminated Rust method signature {method}")


def split_top_level_commas(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    for i, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth = max(0, angle_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "," and angle_depth == 0 and paren_depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def compare_operation_list(
    report: Report,
    label: str,
    expected: list[str],
    got: list[str],
) -> None:
    report.check(got == expected, f"{label}: expected {expected!r}, got {got!r}")


def rust_gpu_sources() -> dict[str, str]:
    return {path.name: read_text(path) for path in sorted(RUST_GPU_SRC.glob("*.rs"))}


def run_negative_tests(
    report: Report,
    oracle: dict[str, Any],
    header_source: str,
    rust_sys_source: str,
    rust_facade_source: str,
    rust_lib_source: str,
    rust_gpu_source_map: dict[str, str],
) -> None:
    mutations = [
        (
            "missing facade operation",
            rust_facade_source.replace(
                '    DecodeFacadeOperation {\n'
                '        operation: "ds4_gpu_attention_decode_heads_tensor",\n'
                '        method: "attention_decode_heads",\n'
                '        tensor_args: &["heads", "q", "raw_kv", "comp_kv", "comp_mask"],\n'
                "    },\n",
                "",
                1,
            ),
        ),
        (
            "tensor argument order drift",
            rust_facade_source.replace(
                'tensor_args: &["heads", "q", "raw_kv", "comp_kv", "comp_mask"],',
                'tensor_args: &["heads", "raw_kv", "q", "comp_kv", "comp_mask"],',
                1,
            ),
        ),
        (
            "missing sys call",
            rust_facade_source.replace(
                "sys::ds4_gpu_router_select_tensor(",
                "sys::ds4_gpu_router_select_tensor_removed(",
                1,
            ),
        ),
    ]
    for name, mutated_facade_source in mutations:
        mutated_report = Report()
        validate(
            mutated_report,
            oracle,
            header_source,
            rust_sys_source,
            mutated_facade_source,
            rust_lib_source,
            rust_gpu_source_map,
        )
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Rust decode backend facade comparator: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(argv)

    report = Report()
    oracle = load_json(ORACLE)
    header_source = read_text(HEADER)
    rust_sys_source = read_text(RUST_SYS)
    rust_facade_source = read_text(RUST_FACADE)
    rust_lib_source = read_text(RUST_LIB)
    rust_gpu_source_map = rust_gpu_sources()

    validate(
        report,
        oracle,
        header_source,
        rust_sys_source,
        rust_facade_source,
        rust_lib_source,
        rust_gpu_source_map,
    )
    if args.negative_test:
        run_negative_tests(
            report,
            oracle,
            header_source,
            rust_sys_source,
            rust_facade_source,
            rust_lib_source,
            rust_gpu_source_map,
        )

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
