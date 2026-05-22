#!/usr/bin/env python3
"""Compare committed supported-model metadata baselines and candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from check_metadata_dump import check_metadata_dump


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "metadata" / "m4.6"
BASELINE_C = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"
NORMALIZED_MODEL_PATH = "<normalized-model-path>"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_dump(obj: dict[str, Any]) -> dict[str, Any]:
    model = dict(obj.get("model", {}))
    if "path" in model:
        model["path"] = NORMALIZED_MODEL_PATH
    return {
        "schema": obj.get("schema"),
        "model": model,
        "validation": obj.get("validation"),
        "selected_metadata": obj.get("selected_metadata"),
        "tensor_types": obj.get("tensor_types"),
        "tensors": obj.get("tensors"),
        "bound_tensors": obj.get("bound_tensors"),
    }


def first_diff(expected: Any, actual: Any, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: missing keys {missing}, extra keys {extra}"
        for key in sorted(expected):
            if expected[key] != actual[key]:
                return first_diff(expected[key], actual[key], f"{path}.{key}")
        return f"{path}: objects differ"
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for idx, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                return first_diff(left, right, f"{path}[{idx}]")
        return f"{path}: arrays differ"
    return f"{path}: {expected!r} != {actual!r}"


def compare_equal(report: Report, label: str, expected: Any, actual: Any) -> None:
    if expected == actual:
        report.check(True, f"{label} matches")
    else:
        report.check(False, f"{label} drift: {first_diff(expected, actual)}")


def metadata_entry(obj: dict[str, Any], key: str) -> dict[str, Any]:
    for entry in obj["selected_metadata"]:
        if entry.get("key") == key:
            return entry
    raise KeyError(key)


def tensor_entry(obj: dict[str, Any], name: str) -> dict[str, Any]:
    for tensor in obj["tensors"]:
        if tensor.get("name") == name:
            return tensor
    raise KeyError(name)


def bound_entry(obj: dict[str, Any], role: str) -> dict[str, Any]:
    for bound in obj["bound_tensors"]:
        if bound.get("role") == role:
            return bound
    raise KeyError(role)


def assert_drift_detected(
    report: Report,
    label: str,
    baseline: dict[str, Any],
    mutation: Any,
) -> None:
    mutated = copy.deepcopy(baseline)
    mutation(mutated)
    report.check(
        normalize_dump(mutated) != normalize_dump(baseline),
        f"negative {label} drift was not detected",
    )


def run_negative_tests(report: Report, baseline: dict[str, Any]) -> None:
    path_mutation = copy.deepcopy(baseline)
    path_mutation["model"]["path"] = "/tmp/different/workspace/model.gguf"
    compare_equal(
        report,
        "normalized model path mutation",
        normalize_dump(baseline),
        normalize_dump(path_mutation),
    )

    assert_drift_detected(
        report,
        "scalar metadata",
        baseline,
        lambda obj: metadata_entry(obj, "deepseek4.embedding_length").__setitem__("value", 4097),
    )
    assert_drift_detected(
        report,
        "array metadata",
        baseline,
        lambda obj: metadata_entry(obj, "deepseek4.attention.compress_ratios")["values"].__setitem__(2, 128),
    )
    assert_drift_detected(
        report,
        "tensor shape",
        baseline,
        lambda obj: tensor_entry(obj, "token_embd.weight")["dims"].__setitem__(0, 4097),
    )
    assert_drift_detected(
        report,
        "tensor type",
        baseline,
        lambda obj: tensor_entry(obj, "token_embd.weight").__setitem__("type", 0),
    )
    assert_drift_detected(
        report,
        "binding",
        baseline,
        lambda obj: bound_entry(obj, "base.token_embd").__setitem__("name", "output.weight"),
    )
    assert_drift_detected(
        report,
        "offset",
        baseline,
        lambda obj: tensor_entry(obj, "token_embd.weight").__setitem__(
            "abs_offset",
            tensor_entry(obj, "token_embd.weight")["abs_offset"] + 32,
        ),
    )


def check_manifest(report: Report, manifest_path: Path, baseline_path: Path) -> None:
    manifest = load_json(manifest_path)
    report.check(manifest.get("schema") == "ds4.metadata_baseline.v1", "manifest schema mismatch")
    current_c = manifest.get("dumps", {}).get("current_c", {})
    report.check(
        current_c.get("sha256") == sha256_file(baseline_path),
        "current-c dump sha256 does not match manifest",
    )
    report.check(
        current_c.get("size_bytes") == baseline_path.stat().st_size,
        "current-c dump size does not match manifest",
    )
    model = manifest.get("model", {})
    report.check(model.get("size_bytes") == 86720111488, "model size mismatch")
    report.check(
        model.get("sha256") == "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668",
        "model sha256 mismatch",
    )
    refresh_commands = manifest.get("refresh_commands")
    report.check(isinstance(refresh_commands, list) and len(refresh_commands) >= 2, "refresh commands missing")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    sections = 1 if report.ok else 0
    print(f"metadata baseline comparison: {status}, {report.checks} checks")
    print(f"summary: {sections}/1 sections passed, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE_C)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--candidate-c", type=Path)
    parser.add_argument("--candidate-rust", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    baseline = load_json(args.baseline)
    schema_report = check_metadata_dump(args.baseline)
    report.check(schema_report.ok, "baseline schema check failed")
    for error in schema_report.errors:
        report.errors.append(f"baseline schema: {error}")

    check_manifest(report, args.manifest, args.baseline)
    baseline_norm = normalize_dump(baseline)
    compare_equal(report, "baseline self-compare", baseline_norm, normalize_dump(load_json(args.baseline)))

    if args.negative_test:
        run_negative_tests(report, baseline)

    if args.candidate_c:
        candidate_c = load_json(args.candidate_c)
        compare_equal(report, f"C candidate {args.candidate_c}", baseline_norm, normalize_dump(candidate_c))

    if args.candidate_rust:
        candidate_rust = load_json(args.candidate_rust)
        compare_equal(
            report,
            f"Rust candidate {args.candidate_rust}",
            baseline_norm,
            normalize_dump(candidate_rust),
        )

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
