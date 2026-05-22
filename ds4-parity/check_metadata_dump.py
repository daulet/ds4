#!/usr/bin/env python3
"""Schema checks for current-C DS4 metadata dumps."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_METADATA_KEYS = {
    "general.name",
    "general.architecture",
    "deepseek4.block_count",
    "deepseek4.embedding_length",
    "deepseek4.vocab_size",
    "deepseek4.attention.head_count",
    "deepseek4.attention.compress_ratios",
    "deepseek4.rope.scaling.original_context_length",
    "deepseek4.expert_count",
    "deepseek4.hyper_connection.count",
    "deepseek4.swiglu_clamp_exp",
}

REQUIRED_BOUND_ROLES = {
    "base.token_embd",
    "base.output",
    "base.layer.0.attn_norm",
    "base.layer.0.attn_q_a",
    "base.layer.0.ffn_gate_exps",
    "base.layer.42.attn_norm",
    "base.layer.42.ffn_down_shexp",
}


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


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} must be an object")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, label: str) -> list[Any]:
    report.check(isinstance(value, list), f"{label} must be an array")
    return value if isinstance(value, list) else []


def check_int(report: Report, obj: dict[str, Any], key: str, label: str) -> None:
    value = obj.get(key)
    report.check(isinstance(value, int) and value >= 0, f"{label}.{key} must be a nonnegative integer")


def check_tensor_ref(report: Report, obj: dict[str, Any], label: str) -> None:
    report.check(isinstance(obj.get("name"), str) and obj["name"], f"{label}.name missing")
    check_int(report, obj, "type", label)
    report.check(isinstance(obj.get("type_name"), str) and obj["type_name"], f"{label}.type_name missing")
    check_int(report, obj, "ndim", label)
    dims = require_list(report, obj.get("dims"), f"{label}.dims")
    report.check(len(dims) == obj.get("ndim"), f"{label}.dims length must equal ndim")
    for idx, dim in enumerate(dims):
        report.check(isinstance(dim, int) and dim > 0, f"{label}.dims[{idx}] must be positive")
    for key in ("elements", "bytes", "rel_offset", "abs_offset"):
        check_int(report, obj, key, label)


def check_metadata_dump(path: Path) -> Report:
    report = Report()
    root = require_dict(report, load_json(path), "root")

    report.check(root.get("schema") == "ds4.metadata.v1", "schema must be ds4.metadata.v1")
    report.check(root.get("source") == "current-c-loader", "source must be current-c-loader")

    model = require_dict(report, root.get("model"), "model")
    report.check(isinstance(model.get("path"), str) and model["path"], "model.path missing")
    for key in (
        "size",
        "gguf_version",
        "metadata_count",
        "tensor_count",
        "alignment",
        "tensor_data_offset",
    ):
        check_int(report, model, key, "model")
    report.check(model.get("gguf_version") == 3, "model.gguf_version must be 3")

    validation = require_dict(report, root.get("validation"), "validation")
    report.check(validation.get("config") == "passed", "validation.config must be passed")
    report.check(validation.get("weights") == "passed", "validation.weights must be passed")
    report.check(validation.get("mtp_weights") in {"passed", "skipped"}, "validation.mtp_weights invalid")

    selected = require_list(report, root.get("selected_metadata"), "selected_metadata")
    selected_keys: set[str] = set()
    for idx, entry_value in enumerate(selected):
        entry = require_dict(report, entry_value, f"selected_metadata[{idx}]")
        key = entry.get("key")
        selected_keys.add(key) if isinstance(key, str) else None
        report.check(isinstance(key, str) and key, f"selected_metadata[{idx}].key missing")
        report.check(isinstance(entry.get("type"), str) and entry["type"], f"selected_metadata[{idx}].type missing")
        if entry.get("type") == "array":
            check_int(report, entry, "len", f"selected_metadata[{idx}]")
            values = require_list(report, entry.get("values"), f"selected_metadata[{idx}].values")
            report.check(len(values) == entry.get("len"), f"selected_metadata[{idx}].values length mismatch")
        else:
            report.check("value" in entry, f"selected_metadata[{idx}].value missing")
    for key in sorted(REQUIRED_METADATA_KEYS):
        report.check(key in selected_keys, f"missing selected metadata key: {key}")

    tensor_types = require_list(report, root.get("tensor_types"), "tensor_types")
    type_count_sum = 0
    for idx, entry_value in enumerate(tensor_types):
        entry = require_dict(report, entry_value, f"tensor_types[{idx}]")
        check_int(report, entry, "type", f"tensor_types[{idx}]")
        report.check(isinstance(entry.get("type_name"), str) and entry["type_name"], f"tensor_types[{idx}].type_name missing")
        check_int(report, entry, "count", f"tensor_types[{idx}]")
        check_int(report, entry, "bytes", f"tensor_types[{idx}]")
        type_count_sum += entry["count"] if isinstance(entry.get("count"), int) else 0

    tensors = require_list(report, root.get("tensors"), "tensors")
    report.check(len(tensors) == model.get("tensor_count"), "tensors length must equal model.tensor_count")
    report.check(type_count_sum == len(tensors), "tensor_types counts must sum to tensor count")
    seen_tensor_names: set[str] = set()
    for idx, tensor_value in enumerate(tensors):
        tensor = require_dict(report, tensor_value, f"tensors[{idx}]")
        check_int(report, tensor, "index", f"tensors[{idx}]")
        report.check(tensor.get("index") == idx, f"tensors[{idx}].index mismatch")
        check_tensor_ref(report, tensor, f"tensors[{idx}]")
        name = tensor.get("name")
        if isinstance(name, str):
            report.check(name not in seen_tensor_names, f"duplicate tensor name: {name}")
            seen_tensor_names.add(name)

    bound = require_list(report, root.get("bound_tensors"), "bound_tensors")
    bound_roles: set[str] = set()
    for idx, entry_value in enumerate(bound):
        entry = require_dict(report, entry_value, f"bound_tensors[{idx}]")
        role = entry.get("role")
        bound_roles.add(role) if isinstance(role, str) else None
        report.check(isinstance(role, str) and role, f"bound_tensors[{idx}].role missing")
        report.check(isinstance(entry.get("present"), bool), f"bound_tensors[{idx}].present missing")
        if entry.get("present"):
            check_tensor_ref(report, entry, f"bound_tensors[{idx}]")
            name = entry.get("name")
            if isinstance(name, str):
                report.check(name in seen_tensor_names, f"bound tensor references unknown tensor: {name}")
    for role in sorted(REQUIRED_BOUND_ROLES):
        report.check(role in bound_roles, f"missing bound tensor role: {role}")

    return report


def run_negative(path: Path) -> Report:
    report = Report()
    original = load_json(path)
    with tempfile.TemporaryDirectory(prefix="ds4-metadata-negative-") as tmp:
        tmp_path = Path(tmp)

        bad_count = tmp_path / "bad-count.json"
        mutated = json.loads(json.dumps(original))
        mutated["model"]["tensor_count"] = mutated["model"]["tensor_count"] + 1
        bad_count.write_text(json.dumps(mutated))
        report.check(not check_metadata_dump(bad_count).ok, "negative tensor_count drift was not detected")

        bad_role = tmp_path / "bad-role.json"
        mutated = json.loads(json.dumps(original))
        mutated["bound_tensors"] = [
            item for item in mutated["bound_tensors"] if item.get("role") != "base.token_embd"
        ]
        bad_role.write_text(json.dumps(mutated))
        report.check(not check_metadata_dump(bad_role).ok, "negative missing bound role was not detected")

        bad_key = tmp_path / "bad-key.json"
        mutated = json.loads(json.dumps(original))
        mutated["selected_metadata"] = [
            item for item in mutated["selected_metadata"] if item.get("key") != "deepseek4.block_count"
        ]
        bad_key.write_text(json.dumps(mutated))
        report.check(not check_metadata_dump(bad_key).ok, "negative missing metadata key was not detected")

        shutil.rmtree(tmp_path, ignore_errors=True)
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = check_metadata_dump(args.dump)
    print_report("metadata schema", report)
    if not report.ok:
        return 1

    if args.negative_test:
        negative = run_negative(args.dump)
        print_report("metadata negative tests", negative)
        if not negative.ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
