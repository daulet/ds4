#!/usr/bin/env python3
"""Validate the Rust first real-model decode kernel contract for M10.5c4c2b2b1."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "bin": ROOT / "rust/ds4-gpu/src/bin/ds4-decode-first-kernel.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

SCHEMA = "ds4.decode_first_kernel.v1"
CASE = "embed_token_hc_token0"
MODEL_SIZE = 86720111488
TENSOR_DATA_OFFSET = 5333824
TOKEN_EMBD_OFFSET = 77928033088
TOKEN_EMBD_BYTES = 1059061760
FNV1A64 = "f76512db41f80c4d"
SAMPLES = {
    0: -0.107421875,
    1: -0.019897461,
    8192: -0.107421875,
    16382: 0.22558594,
    16383: -0.17285156,
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


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    texts = {name: path.read_text() for name, path in FILES.items()}
    if args.negative_test:
        return run_negative_tests(texts)

    report = Report()
    validate_static(report, texts)
    if args.candidate is not None:
        validate_candidate(report, load_json(args.candidate))

    if report.ok:
        print(f"Rust decode first-kernel contract: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="B300 JSON emitted by ds4-decode-first-kernel")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    bin_text = texts["bin"]
    for needle, message in (
        (SCHEMA, "first-kernel schema missing"),
        (CASE, "first-kernel case missing"),
        ("parse_gguf_allowing_missing_tensor_data", "GGUF prefix parser missing"),
        ("bind_ds4_weights", "DS4 weight binding missing"),
        ("mmap(", "full model mmap missing"),
        ("set_model_fd", "model fd bridge missing"),
        ("set_model_map_range", "model map range bridge missing"),
        ("DecodeBackend::new", "decode backend facade missing"),
        ("CommandBatch::begin", "command batch begin missing"),
        (".embed_token_hc(", "embed_token_hc facade call missing"),
        ("weights.token_embd.abs_offset", "token embedding offset missing"),
        ("Tensor::allocate", "cur_hc allocation missing"),
        ("read_bytes", "cur_hc readback missing"),
        ("fnv1a64(&bytes)", "cur_hc full-buffer digest missing"),
        ("synchronize", "backend synchronization missing"),
        ("BackendGuard", "cleanup guard missing"),
        ("ds4_gpu::cleanup", "backend cleanup missing"),
    ):
        report.check(needle in bin_text, message)

    report_text = texts["report"]
    report.check("M10.5c4c2b2b1 Rust first decode kernel comparator" in report_text, "unified report comparator missing")
    report.check("M10.5c4c2b2b1 B300 Rust first decode kernel rerun" in report_text, "B300 first-kernel skip missing")
    report.check("--bin ds4-decode-first-kernel" in report_text, "B300 first-kernel command missing binary")
    report.check("--candidate /tmp/ds4-c2b2b1-first-kernel.json" in report_text, "B300 first-kernel candidate validation missing")

    report.check("M10.5c4c2b2b1 Rust first decode kernel" in texts["readme"], "README entry missing")
    report.check("M10.5c4c2b2b1: Rust First Decode Kernel Execution" in texts["roadmap"], "roadmap first-kernel split missing")
    report.check("M10.5c4c2b2b2b: Rust One-Token Decode B300 Execution" in texts["roadmap"], "roadmap decode remainder missing")
    report.check("M10.5c4c2b2b1: Rust First Decode Kernel Execution" in texts["todo"], "TODO first-kernel split missing")
    report.check("M10.5c4c2b2b2b: Rust One-Token Decode B300 Execution" in texts["todo"], "TODO decode remainder missing")


def validate_candidate(report: Report, obj: dict[str, Any]) -> None:
    report.check(obj.get("schema") == SCHEMA, "candidate schema drift")
    report.check(obj.get("case") == CASE, "candidate case drift")

    model = obj.get("model")
    report.check(isinstance(model, dict), "candidate model missing")
    if isinstance(model, dict):
        report.check(model.get("mapped_size") == MODEL_SIZE, "candidate model size drift")
        report.check(model.get("tensor_count") == 1328, "candidate tensor count drift")
        report.check(model.get("tensor_data_offset") == TENSOR_DATA_OFFSET, "candidate tensor-data offset drift")
        report.check(model.get("bound_layers") == 43, "candidate bound layer count drift")

    operation = obj.get("operation")
    report.check(isinstance(operation, dict), "candidate operation missing")
    if isinstance(operation, dict):
        report.check(operation.get("name") == "ds4_gpu_embed_token_hc_tensor", "candidate operation name drift")
        report.check(operation.get("method") == "embed_token_hc", "candidate method drift")
        report.check(operation.get("command_batch") is True, "candidate command batch missing")
        report.check(operation.get("synchronized") is True, "candidate synchronize missing")
        report.check(operation.get("token") == 0, "candidate token drift")
        report.check(operation.get("n_vocab") == 129280, "candidate vocab drift")
        report.check(operation.get("n_embd") == 4096, "candidate n_embd drift")
        report.check(operation.get("n_hc") == 4, "candidate n_hc drift")

    weight = obj.get("weight")
    report.check(isinstance(weight, dict), "candidate weight missing")
    if isinstance(weight, dict):
        report.check(weight.get("role") == "base.token_embd", "candidate weight role drift")
        report.check(weight.get("abs_offset") == TOKEN_EMBD_OFFSET, "candidate token embedding offset drift")
        report.check(weight.get("bytes") == TOKEN_EMBD_BYTES, "candidate token embedding bytes drift")
        report.check(weight.get("type") == 1, "candidate token embedding type drift")
        report.check(weight.get("type_name") == "f16", "candidate token embedding type name drift")

    output = obj.get("output")
    report.check(isinstance(output, dict), "candidate output missing")
    if isinstance(output, dict):
        report.check(output.get("field") == "cur_hc", "candidate output field drift")
        report.check(output.get("bytes") == 65536, "candidate output byte drift")
        report.check(output.get("elements") == 16384, "candidate output element drift")
        report.check(output.get("nonzero_elements") == 16384, "candidate output nonzero count drift")
        report.check(output.get("fnv1a64") == FNV1A64, "candidate output FNV digest drift")
        validate_samples(report, output.get("samples"))


def validate_samples(report: Report, samples: Any) -> None:
    report.check(isinstance(samples, list), "candidate samples missing")
    if not isinstance(samples, list):
        return
    by_index = {entry.get("index"): entry.get("value") for entry in samples if isinstance(entry, dict)}
    report.check(set(SAMPLES) <= set(by_index), "candidate sample set incomplete")
    for index, expected in SAMPLES.items():
        value = by_index.get(index)
        report.check(isinstance(value, (int, float)) and math.isfinite(value), f"sample {index} is not finite")
        if isinstance(value, (int, float)) and math.isfinite(value):
            report.check(abs(float(value) - expected) <= 1e-6, f"sample {index} value drift")


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read candidate {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"candidate {path}: expected JSON object")
    return obj


def run_negative_tests(texts: dict[str, str]) -> int:
    static_mutations = [
        ("remove schema", "bin", SCHEMA, "ds4.decode_first_kernel.removed"),
        ("remove command batch", "bin", "CommandBatch::begin", "CommandBatch_removed::begin"),
        ("remove facade call", "bin", ".embed_token_hc(", ".embed_token_hc_removed("),
        ("remove readback", "bin", "read_bytes", "read_removed"),
        ("remove digest", "bin", "fnv1a64(&bytes)", "fnv1a64_removed(&bytes)"),
        ("remove b300 candidate check", "report", "--candidate /tmp/ds4-c2b2b1-first-kernel.json", ""),
        ("remove roadmap split", "roadmap", "M10.5c4c2b2b1: Rust First Decode Kernel Execution", "M10.5c4c2b2b1 removed"),
    ]
    failures: list[str] = []
    for label, key, needle, replacement in static_mutations:
        mutated = copy.deepcopy(texts)
        if needle not in mutated[key]:
            failures.append(f"{label}: mutation needle not found")
            continue
        mutated[key] = mutated[key].replace(needle, replacement)
        report = Report()
        validate_static(report, mutated)
        if report.ok:
            failures.append(f"{label}: validation unexpectedly passed")

    candidate_mutations = [
        ("weight offset", mutate_weight_offset),
        ("nonzero count", mutate_nonzero_count),
        ("output digest", mutate_output_digest),
        ("sample value", mutate_sample_value),
    ]
    for label, mutate in candidate_mutations:
        report = Report()
        validate_candidate(report, mutate(valid_candidate()))
        if report.ok:
            failures.append(f"{label}: candidate validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(static_mutations) + len(candidate_mutations)} mutations rejected")
    return 0


def valid_candidate() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "case": CASE,
        "model": {
            "mapped_size": MODEL_SIZE,
            "header_bytes_read": 8388608,
            "tensor_count": 1328,
            "tensor_data_offset": TENSOR_DATA_OFFSET,
            "bound_layers": 43,
        },
        "operation": {
            "name": "ds4_gpu_embed_token_hc_tensor",
            "method": "embed_token_hc",
            "command_batch": True,
            "synchronized": True,
            "token": 0,
            "n_vocab": 129280,
            "n_embd": 4096,
            "n_hc": 4,
        },
        "weight": {
            "role": "base.token_embd",
            "abs_offset": TOKEN_EMBD_OFFSET,
            "bytes": TOKEN_EMBD_BYTES,
            "type": 1,
            "type_name": "f16",
        },
        "output": {
            "field": "cur_hc",
            "bytes": 65536,
            "elements": 16384,
            "nonzero_elements": 16384,
            "fnv1a64": FNV1A64,
            "samples": [{"index": index, "value": value} for index, value in SAMPLES.items()],
        },
    }


def mutate_weight_offset(obj: dict[str, Any]) -> dict[str, Any]:
    obj["weight"]["abs_offset"] += 32
    return obj


def mutate_nonzero_count(obj: dict[str, Any]) -> dict[str, Any]:
    obj["output"]["nonzero_elements"] -= 1
    return obj


def mutate_output_digest(obj: dict[str, Any]) -> dict[str, Any]:
    obj["output"]["fnv1a64"] = "0123456789abcdef"
    return obj


def mutate_sample_value(obj: dict[str, Any]) -> dict[str, Any]:
    obj["output"]["samples"][0]["value"] += 0.01
    return obj


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
