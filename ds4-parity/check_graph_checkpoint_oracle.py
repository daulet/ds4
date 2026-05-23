#!/usr/bin/env python3
"""Validate the M10.4 current-C graph checkpoint oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "ds4-parity/baselines/graph/m10.4/current-c.json"
DEFAULT_MANIFEST = ROOT / "ds4-parity/baselines/graph/m10.4/manifest.json"
SCHEMA = "ds4.graph_checkpoint_oracle.v1"

REQUIRED_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "short_prefill_logits": ("output-head", "metal_graph_prefill_layer_major"),
    "short_decode_logits": ("decode", "metal_graph_eval_token_raw_swa"),
    "short_decode_layer2_attn_comp_cache": ("compressed-kv", "metal_graph_eval_token_raw_swa"),
    "short_decode_layer2_index_comp_cache": ("compressed-kv", "metal_graph_eval_token_raw_swa"),
    "long_chunked_prefill_logits": ("prefill", "metal_graph_prefill_chunked_range"),
    "long_chunked_prefill_layer2_attn_comp_cache": ("compressed-kv", "metal_graph_prefill_chunked_range"),
    "long_chunked_prefill_layer2_index_comp_cache": ("compressed-kv", "metal_graph_prefill_chunked_range"),
    "cache_continuation_prefill_logits": ("cache-continuation", "metal_graph_prefill_chunked_range"),
    "cache_continuation_layer2_attn_comp_cache": ("compressed-kv", "metal_graph_prefill_chunked_range"),
    "cache_continuation_layer2_index_comp_cache": ("compressed-kv", "metal_graph_prefill_chunked_range"),
}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    baseline = load_json(args.baseline)
    candidate = baseline if args.candidate is None else load_json(args.candidate)
    manifest = load_json(args.manifest) if args.manifest.exists() else None

    if args.negative_test:
        errors = validate_dump(baseline, manifest)
        if errors:
            print_errors(errors)
            return 1
        return run_negative_tests(baseline)

    errors = validate_dump(candidate, manifest)
    errors.extend(compare_dumps(baseline, candidate))
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    elif errors:
        print_errors(errors)
    else:
        checkpoints = candidate["checkpoints"]
        exact = sum(1 for c in checkpoints if c.get("hash_policy") == "exact")
        tolerant = sum(1 for c in checkpoints if c.get("hash_policy") == "tolerance")
        print(
            "Graph checkpoint oracle: "
            f"{len(checkpoints)} checkpoints, {exact} exact hashes, {tolerant} tolerant logits"
        )
    return 1 if errors else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--json", action="store_true")
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


def validate_dump(obj: dict[str, Any], manifest: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    check(obj.get("schema") == SCHEMA, errors, "schema drift")
    check(obj.get("source") == "current-c-b300-graph-checkpoints", errors, "source drift")
    check(obj.get("backend") in {"cuda", "metal"}, errors, "backend must be graph-capable")
    check(isinstance(obj.get("ctx_size"), int) and obj["ctx_size"] > 0, errors, "ctx_size invalid")
    check(isinstance(obj.get("short_prompt_tokens"), int) and obj["short_prompt_tokens"] > 0, errors, "short prompt token count invalid")
    check(isinstance(obj.get("long_prompt_tokens"), int) and obj["long_prompt_tokens"] > obj["short_prompt_tokens"], errors, "long prompt token count invalid")

    checkpoints = obj.get("checkpoints")
    if not isinstance(checkpoints, list):
        errors.append("checkpoints must be a list")
        return errors
    by_name: dict[str, dict[str, Any]] = {}
    for i, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            errors.append(f"checkpoints[{i}] must be an object")
            continue
        name = checkpoint.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"checkpoints[{i}].name invalid")
            continue
        if name in by_name:
            errors.append(f"duplicate checkpoint {name!r}")
        by_name[name] = checkpoint
        validate_checkpoint(errors, checkpoint)

    for name, (stage, boundary) in REQUIRED_CHECKPOINTS.items():
        checkpoint = by_name.get(name)
        if checkpoint is None:
            errors.append(f"missing required checkpoint {name!r}")
            continue
        check(checkpoint.get("stage") == stage, errors, f"{name}.stage drift")
        check(checkpoint.get("boundary") == boundary, errors, f"{name}.boundary drift")

    skips = obj.get("skips")
    check(isinstance(skips, list), errors, "skips must be a list")
    skip_names = {skip.get("name") for skip in skips if isinstance(skip, dict)}
    if "mtp_verify_decode2_logits" not in by_name:
        check("mtp_verify_decode2_exact" in skip_names, errors, "missing MTP verifier checkpoint or skip")

    if manifest is not None:
        check(manifest.get("schema") == "ds4.graph_checkpoint_manifest.v1", errors, "manifest schema drift")
        check(manifest.get("baseline") == "current-c.json", errors, "manifest baseline drift")
        commands = manifest.get("commands")
        check(isinstance(commands, list) and commands, errors, "manifest commands missing")
    return errors


def validate_checkpoint(errors: list[str], checkpoint: dict[str, Any]) -> None:
    name = str(checkpoint.get("name"))
    for key in ("stage", "boundary", "fixture", "tensor", "dtype", "hash_policy", "sha256"):
        check(isinstance(checkpoint.get(key), str) and checkpoint[key], errors, f"{name}.{key} invalid")
    check(checkpoint.get("dtype") == "f32", errors, f"{name}.dtype must be f32")
    check(checkpoint.get("hash_policy") in {"exact", "tolerance"}, errors, f"{name}.hash_policy invalid")
    check(is_sha256(checkpoint.get("sha256")), errors, f"{name}.sha256 invalid")
    shape = checkpoint.get("shape")
    element_count = checkpoint.get("element_count")
    byte_count = checkpoint.get("bytes")
    check(isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int), errors, f"{name}.shape invalid")
    check(isinstance(element_count, int) and element_count > 0, errors, f"{name}.element_count invalid")
    check(isinstance(byte_count, int) and byte_count == element_count * 4, errors, f"{name}.bytes invalid")
    if isinstance(shape, list) and shape:
        check(shape[0] == element_count, errors, f"{name}.shape/element_count drift")
    check(isinstance(checkpoint.get("offset"), int) and checkpoint["offset"] >= 0, errors, f"{name}.offset invalid")
    check(checkpoint.get("row") is None or isinstance(checkpoint.get("row"), int), errors, f"{name}.row invalid")
    check(checkpoint.get("layer") is None or isinstance(checkpoint.get("layer"), int), errors, f"{name}.layer invalid")
    tolerance = checkpoint.get("f32_tolerance")
    check(isinstance(tolerance, (int, float)) and tolerance >= 0, errors, f"{name}.f32_tolerance invalid")
    validate_top(errors, name, checkpoint.get("top"))
    validate_samples(errors, name, checkpoint.get("samples"), element_count)
    validate_counters(errors, name, checkpoint.get("counters"))


def validate_top(errors: list[str], name: str, top: Any) -> None:
    if not isinstance(top, dict):
        errors.append(f"{name}.top invalid")
        return
    check(top.get("index") is None or isinstance(top.get("index"), int), errors, f"{name}.top.index invalid")
    check(top.get("value") is None or is_number(top.get("value")), errors, f"{name}.top.value invalid")


def validate_samples(errors: list[str], name: str, samples: Any, element_count: Any) -> None:
    if not isinstance(samples, list) or not samples:
        errors.append(f"{name}.samples invalid")
        return
    seen: set[int] = set()
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"{name}.samples[{i}] invalid")
            continue
        index = sample.get("index")
        value = sample.get("value")
        check(isinstance(index, int) and index >= 0, errors, f"{name}.samples[{i}].index invalid")
        if isinstance(index, int) and isinstance(element_count, int):
            check(index < element_count, errors, f"{name}.samples[{i}].index out of range")
            check(index not in seen, errors, f"{name}.samples[{i}].index duplicated")
            seen.add(index)
        check(is_number(value), errors, f"{name}.samples[{i}].value invalid")


def validate_counters(errors: list[str], name: str, counters: Any) -> None:
    if not isinstance(counters, dict):
        errors.append(f"{name}.counters invalid")
        return
    for key in ("checkpoint_len", "prefill_cap", "raw_cap", "raw_window", "comp_cap", "mtp_n_raw"):
        check(isinstance(counters.get(key), int) and counters[key] >= 0, errors, f"{name}.counters.{key} invalid")
    if "layer" in name:
        for key in ("layer_comp_cap", "layer_n_comp", "layer_n_index_comp"):
            check(isinstance(counters.get(key), int) and counters[key] >= 0, errors, f"{name}.counters.{key} invalid")


def compare_dumps(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("schema", "source", "model_sha256", "backend", "ctx_size"):
        if baseline.get(key) != candidate.get(key):
            errors.append(f"{key}: expected {baseline.get(key)!r}, got {candidate.get(key)!r}")

    expected = checkpoints_by_name(baseline)
    actual = checkpoints_by_name(candidate)
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"checkpoints: missing {name!r}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"checkpoints: unexpected {name!r}")
    for name in sorted(set(expected) & set(actual)):
        compare_checkpoint(errors, name, expected[name], actual[name])
    return errors


def compare_checkpoint(
    errors: list[str],
    name: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    exact_fields = (
        "stage",
        "boundary",
        "fixture",
        "tensor",
        "dtype",
        "shape",
        "element_count",
        "bytes",
        "offset",
        "row",
        "layer",
        "hash_policy",
        "counters",
    )
    for key in exact_fields:
        if expected.get(key) != actual.get(key):
            errors.append(f"{name}.{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}")
    if expected.get("hash_policy") == "exact" and expected.get("sha256") != actual.get("sha256"):
        errors.append(f"{name}.sha256: expected {expected.get('sha256')!r}, got {actual.get('sha256')!r}")
    compare_top(errors, name, expected, actual)
    compare_samples(errors, name, expected, actual)


def compare_top(errors: list[str], name: str, expected: dict[str, Any], actual: dict[str, Any]) -> None:
    expected_top = expected.get("top", {})
    actual_top = actual.get("top", {})
    if expected_top.get("index") != actual_top.get("index"):
        errors.append(f"{name}.top.index: expected {expected_top.get('index')!r}, got {actual_top.get('index')!r}")
    compare_float(errors, f"{name}.top.value", expected_top.get("value"), actual_top.get("value"), tolerance_for(expected))


def compare_samples(errors: list[str], name: str, expected: dict[str, Any], actual: dict[str, Any]) -> None:
    expected_samples = {sample["index"]: sample["value"] for sample in expected.get("samples", [])}
    actual_samples = {sample["index"]: sample["value"] for sample in actual.get("samples", []) if isinstance(sample, dict) and "index" in sample}
    if set(expected_samples) != set(actual_samples):
        errors.append(f"{name}.samples.indices: expected {sorted(expected_samples)}, got {sorted(actual_samples)}")
        return
    tolerance = tolerance_for(expected)
    for index in sorted(expected_samples):
        compare_float(errors, f"{name}.samples[{index}]", expected_samples[index], actual_samples[index], tolerance)


def compare_float(errors: list[str], label: str, expected: Any, actual: Any, tolerance: float) -> None:
    if not is_number(expected) or not is_number(actual):
        if expected != actual:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}")
        return
    if math.isfinite(float(expected)) and math.isfinite(float(actual)):
        if abs(float(expected) - float(actual)) > tolerance:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}, tolerance {tolerance}")
    elif expected != actual:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def run_negative_tests(baseline: dict[str, Any]) -> int:
    mutations = [
        ("missing checkpoint", remove_checkpoint),
        ("boundary drift", mutate_boundary),
        ("counter drift", mutate_counter),
        ("exact hash drift", mutate_exact_hash),
        ("sample drift", mutate_sample),
        ("missing mtp skip", mutate_mtp_skip),
    ]
    passed = 0
    for name, mutate in mutations:
        candidate = mutate(copy.deepcopy(baseline))
        errors = validate_dump(candidate)
        errors.extend(compare_dumps(baseline, candidate))
        if errors:
            passed += 1
        else:
            print(f"negative test {name!r}: expected validation failure", file=sys.stderr)
    if passed != len(mutations):
        return 1
    print(f"Graph checkpoint negative test: {passed} mutations failed as expected")
    return 0


def remove_checkpoint(obj: dict[str, Any]) -> dict[str, Any]:
    obj["checkpoints"] = [c for c in obj["checkpoints"] if c.get("name") != "short_decode_logits"]
    return obj


def mutate_boundary(obj: dict[str, Any]) -> dict[str, Any]:
    checkpoints_by_name(obj)["long_chunked_prefill_logits"]["boundary"] = "metal_graph_prefill_layer_major"
    return obj


def mutate_counter(obj: dict[str, Any]) -> dict[str, Any]:
    checkpoints_by_name(obj)["short_decode_layer2_attn_comp_cache"]["counters"]["layer_n_comp"] += 1
    return obj


def mutate_exact_hash(obj: dict[str, Any]) -> dict[str, Any]:
    checkpoints_by_name(obj)["short_decode_layer2_index_comp_cache"]["sha256"] = "0" * 64
    return obj


def mutate_sample(obj: dict[str, Any]) -> dict[str, Any]:
    checkpoint = checkpoints_by_name(obj)["cache_continuation_prefill_logits"]
    checkpoint["samples"][0]["value"] = float(checkpoint["samples"][0]["value"]) + checkpoint["f32_tolerance"] + 1.0
    return obj


def mutate_mtp_skip(obj: dict[str, Any]) -> dict[str, Any]:
    if "mtp_verify_decode2_logits" not in checkpoints_by_name(obj):
        obj["skips"] = []
    else:
        checkpoints_by_name(obj)["mtp_verify_decode2_logits"]["stage"] = "wrong"
    return obj


def checkpoints_by_name(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {checkpoint["name"]: checkpoint for checkpoint in obj.get("checkpoints", []) if isinstance(checkpoint, dict) and isinstance(checkpoint.get("name"), str)}


def tolerance_for(checkpoint: dict[str, Any]) -> float:
    if checkpoint.get("hash_policy") == "exact":
        return 0.0
    return float(checkpoint.get("f32_tolerance", 0.0))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"Graph checkpoint oracle: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
