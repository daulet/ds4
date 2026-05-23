#!/usr/bin/env python3
"""Check the Rust DS4 structured weight table against flat bindings and C fields."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import compare_tensor_bindings as tensor_fixture


ROOT = Path(__file__).resolve().parents[1]
N_LAYER = 43


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


def run_rust_dump(base: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            "--validate-ds4-layout",
            str(base),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def c_struct_fields(struct_name: str) -> list[str]:
    text = (ROOT / "ds4.c").read_text()
    match = re.search(
        rf"typedef struct\s*\{{(?P<body>[^}}]*)\}}\s*{re.escape(struct_name)};",
        text,
    )
    if not match:
        raise RuntimeError(f"missing C struct {struct_name}")
    return re.findall(r"ds4_tensor\s+\*(\w+);", match.group("body"))


def flatten_weight_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(table.get("base", []))
    for layer in table.get("layers", []):
        out.extend(layer.get("fields", []))
    return out


def normalize_binding(binding: dict[str, Any]) -> dict[str, Any]:
    out = {
        "role": binding.get("role"),
        "present": binding.get("present"),
    }
    if binding.get("present"):
        out.update(
            {
                "name": binding.get("name"),
                "type": binding.get("type"),
                "type_name": binding.get("type_name"),
                "ndim": binding.get("ndim"),
                "dims": binding.get("dims"),
                "elements": binding.get("elements"),
                "bytes": binding.get("bytes"),
                "rel_offset": binding.get("rel_offset"),
                "abs_offset": binding.get("abs_offset"),
            }
        )
    return out


def field_map(layer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry.get("field"): entry for entry in layer.get("fields", [])}


def field_present(fields: dict[str, dict[str, Any]], name: str) -> bool | None:
    entry = fields.get(name)
    if not isinstance(entry, dict):
        return None
    return entry.get("present")


def validate_weight_table(report: Report, obj: dict[str, Any]) -> None:
    table = obj.get("weight_table")
    report.check(isinstance(table, dict), "weight_table missing")
    if not isinstance(table, dict):
        return
    report.check(table.get("schema") == "ds4.weights.v1", "weight table schema drift")

    base = table.get("base")
    layers = table.get("layers")
    report.check(isinstance(base, list), "weight_table.base must be a list")
    report.check(isinstance(layers, list), "weight_table.layers must be a list")
    if not isinstance(base, list) or not isinstance(layers, list):
        return

    c_base_fields = c_struct_fields("ds4_weights")
    c_layer_fields = c_struct_fields("ds4_layer_weights")
    report.check([entry.get("field") for entry in base] == c_base_fields, "base field order differs from C ds4_weights")
    report.check(len(layers) == N_LAYER, "layer count differs from DS4_N_LAYER")

    for idx, layer in enumerate(layers):
        report.check(layer.get("index") == idx, f"layer {idx} index mismatch")
        fields = layer.get("fields")
        report.check(isinstance(fields, list), f"layer {idx} fields must be a list")
        if isinstance(fields, list):
            report.check(
                [entry.get("field") for entry in fields] == c_layer_fields,
                f"layer {idx} field order differs from C ds4_layer_weights",
            )

    flat_from_table = [normalize_binding(entry) for entry in flatten_weight_table(table)]
    flat_from_dump = [normalize_binding(entry) for entry in obj.get("bound_tensors", [])]
    report.check(flat_from_table == flat_from_dump, "structured weight table does not flatten to bound_tensors")

    if len(layers) == N_LAYER:
        layer0 = field_map(layers[0])
        layer2 = field_map(layers[2])
        layer3 = field_map(layers[3])
        report.check(field_present(layer0, "attn_compressor_ape") is False, "dense layer compressor must be absent")
        report.check(field_present(layer2, "attn_compressor_ape") is True, "ratio-4 compressor must be present")
        report.check(field_present(layer2, "indexer_proj") is True, "ratio-4 indexer must be present")
        report.check(field_present(layer3, "attn_compressor_ape") is True, "ratio-128 compressor must be present")
        report.check(field_present(layer3, "indexer_proj") is False, "ratio-128 indexer must be absent")
        report.check(field_present(layer2, "ffn_gate_tid2eid") is True, "hash-layer tid2eid must be present")
        report.check(field_present(layer3, "ffn_gate_tid2eid") is False, "post-hash-layer tid2eid must be absent")


def mutate_removed_layer(obj: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    mutated["weight_table"]["layers"].pop()
    return mutated


def mutate_removed_field(obj: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    fields = mutated["weight_table"]["layers"][2]["fields"]
    mutated["weight_table"]["layers"][2]["fields"] = [
        entry for entry in fields if entry.get("field") != "indexer_proj"
    ]
    return mutated


def mutate_presence(obj: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    for entry in mutated["weight_table"]["layers"][0]["fields"]:
        if entry.get("field") == "attn_compressor_ape":
            entry["present"] = True
            entry["name"] = "unexpected"
            entry["type"] = 0
            entry["dims"] = [1]
            entry["bytes"] = 4
            break
    return mutated


def run_negative_tests(report: Report, obj: dict[str, Any]) -> None:
    mutations = [
        ("removed-layer", mutate_removed_layer),
        ("removed-field", mutate_removed_field),
        ("presence-drift", mutate_presence),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate_weight_table(mutated_report, mutate(obj))
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Rust weight table comparator: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    with tempfile.TemporaryDirectory(prefix="ds4-rust-weight-table-") as tmp:
        base_path = Path(tmp) / "base.gguf"
        tensor_fixture.write_gguf(base_path, tensor_fixture.base_tensors(), include_metadata=True)
        obj = run_rust_dump(base_path)
    validate_weight_table(report, obj)
    if args.negative_test:
        run_negative_tests(report, obj)

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
