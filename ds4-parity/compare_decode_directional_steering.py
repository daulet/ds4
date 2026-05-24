#!/usr/bin/env python3
"""Compare Rust directional-steering decode readback against current-C."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "c_helper": ROOT / "ds4_directional_steering_oracle_dump.c",
    "ds4_c": ROOT / "ds4.c",
    "ds4_h": ROOT / "ds4.h",
    "makefile": ROOT / "Makefile",
    "decode_backend": ROOT / "rust/ds4-gpu/src/decode_backend.rs",
    "rust_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-decode-full-output-head.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

ORACLE_SCHEMA = "ds4.directional_steering_decode_oracle.v1"
CANDIDATE_SCHEMA = "ds4.decode_directional_steering.v1"
CASE = "token0_layer0_directional_steering_full_output_head"
MODEL_SIZE = 86720111488
TENSOR_DATA_OFFSET = 5333824
STEERING_BYTES = 704512
STEERING_ELEMENTS = 176128
STEERING_ATTN = 0.5
STEERING_FFN = 0.25
STEERING_FNV1A64 = "960514fa6e7884ca"
OUTPUTS = {
    "layer0_attn_out": 4096,
    "layer0_after_attn_hc": 16384,
    "layer0_ffn_out": 4096,
    "layer0_after_ffn_hc": 16384,
    "after_layer42_hc": 16384,
    "output_pre": 4,
    "output_weights": 4,
    "output_embd": 4096,
    "output_norm": 4096,
    "logits": 129280,
}
EXPECTED_FNV1A64 = {
    "layer0_attn_out": "68356dba6c067ffa",
    "layer0_after_attn_hc": "f1c47bcde7bdec38",
    "layer0_ffn_out": "7c8abeae9af7cc84",
    "layer0_after_ffn_hc": "db94a9015d610f1b",
    "after_layer42_hc": "7b8a60690319eff8",
    "output_pre": "5b6b7ffd274f62b2",
    "output_weights": "42a754df67d85acf",
    "output_embd": "c1e3490b198cf968",
    "output_norm": "53be42a180587d23",
    "logits": "8caf00d359fba4f1",
}
EXPECTED_OPERATION = {
    "token": 0,
    "first_layer": 0,
    "last_layer": 42,
    "position": 0,
    "decoded_layers": 43,
    "ctx_size": 32768,
    "prefill_cap": 2048,
    "raw_cap": 2304,
    "raw_window": 128,
    "raw_row": 0,
    "raw_start": 0,
    "n_raw": 1,
    "n_comp": 0,
    "n_selected": 0,
    "use_mask": 0,
    "emit_compressed_row": 0,
    "n_vocab": 129280,
    "vocab_dim": 129280,
    "n_embd": 4096,
    "n_hc": 4,
    "hc_dim": 16384,
    "steering_layer": 0,
    "directional_steering_bytes": STEERING_BYTES,
    "directional_steering_elements": STEERING_ELEMENTS,
}
EXPECTED_FLOAT_OPERATION = {
    "rms_eps": 1e-6,
    "hc_eps": 1e-6,
    "directional_steering_attn": STEERING_ATTN,
    "directional_steering_ffn": STEERING_FFN,
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FNV = re.compile(r"^[0-9a-f]{16}$")


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
    if args.oracle is not None or args.candidate is not None:
        if args.oracle is None or args.candidate is None:
            raise SystemExit("--oracle and --candidate must be provided together")
        validate_pair(report, load_json(args.oracle), load_json(args.candidate))

    if report.ok:
        print(f"Rust directional-steering decode comparator: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, help="JSON from ds4-directional-steering-oracle-dump")
    parser.add_argument("--candidate", type=Path, help="JSON from ds4-decode-full-output-head with steering")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    c_helper = texts["c_helper"]
    for needle, message in (
        ("ds4_dump_directional_steering_decode_oracle_json", "C helper does not call steering oracle"),
        ("--dir-steering-file", "C helper steering file flag missing"),
        ("--dir-steering-attn", "C helper steering attention flag missing"),
        ("--dir-steering-ffn", "C helper steering FFN flag missing"),
    ):
        report.check(needle in c_helper, message)

    ds4_c = texts["ds4_c"]
    for needle, message in (
        ("ds4_dump_directional_steering_decode_oracle_json", "C steering oracle implementation missing"),
        ("metal_graph_load_directional_steering", "C oracle does not load steering directions"),
        ("ds4_gpu_tensor_copy(layer0_attn_out_tensor", "C oracle does not checkpoint layer0 attn_out"),
        ("ds4_gpu_tensor_copy(layer0_ffn_out_tensor", "C oracle does not checkpoint layer0 ffn_out"),
        (ORACLE_SCHEMA, "C steering oracle schema missing"),
        ("\"layer0_attn_out\"", "C oracle layer0 attn output missing"),
        ("\"layer0_after_ffn_hc\"", "C oracle layer0 FFN HC output missing"),
        ("directional_steering_attn", "C oracle steering scale metadata missing"),
    ):
        report.check(needle in ds4_c, message)

    report.check(
        "ds4_dump_directional_steering_decode_oracle_json" in texts["ds4_h"],
        "header declaration for steering oracle missing",
    )
    makefile = texts["makefile"]
    report.check("ds4-directional-steering-oracle-dump:" in makefile, "Makefile steering target missing")
    report.check("ds4_directional_steering_oracle_dump_cpu.o" in makefile, "CPU steering helper object missing")

    decode_backend = texts["decode_backend"]
    for needle, message in (
        ("DIRECTIONAL_STEERING_DECODE_FACADE_OPERATIONS", "steering facade operation list missing"),
        ("attention_output_q8_batch", "attention output batch wrapper missing"),
        ("pub fn directional_steering_project(", "directional steering wrapper missing"),
        ("pub fn add(", "add wrapper missing"),
        ("hc_expand_split", "HC expand split wrapper missing"),
    ):
        report.check(needle in decode_backend, message)

    rust_bin = texts["rust_bin"]
    for needle, message in (
        (CANDIDATE_SCHEMA, "Rust candidate steering schema missing"),
        ("--dir-steering-file", "Rust candidate steering file flag missing"),
        (".attention_output_q8_batch(", "Rust candidate does not use unfused attention output"),
        (".directional_steering_project(", "Rust candidate does not call steering projection"),
        (".add(", "Rust candidate does not materialize FFN output"),
        (".hc_expand_split(", "Rust candidate does not use split HC expansion"),
        ("layer0_ffn_out", "Rust candidate layer0 FFN checkpoint missing"),
        ("directional_steering_fnv1a64", "Rust candidate steering file hash metadata missing"),
    ):
        report.check(needle in rust_bin, message)

    report_text = texts["report"]
    report.check(
        "M10.5c4d4 Rust directional-steering decode comparator" in report_text,
        "unified report comparator missing",
    )
    report.check(
        "b300_directional_steering_decode_oracle_command" in report_text,
        "unified report B300 steering rerun command missing",
    )
    readme = texts["readme"]
    report.check(
        "M10.5c4d4" in readme and "compare_decode_directional_steering.py" in readme,
        "README steering entry missing",
    )
    report.check(
        "M10.5c4d4: Rust Directional-Steering Decode Coverage" in texts["roadmap"]
        and "attention scale `0.5` and FFN scale" in texts["roadmap"],
        "roadmap steering fixture not pinned",
    )
    report.check(
        "M10.5c4d4: Rust Directional-Steering Decode Coverage" in texts["todo"]
        and "post-steer FFN HC expansion" in texts["todo"],
        "TODO steering item not pinned",
    )


def validate_pair(report: Report, oracle: dict[str, Any], candidate: dict[str, Any]) -> None:
    validate_common(report, oracle, "oracle", ORACLE_SCHEMA)
    validate_common(report, candidate, "candidate", CANDIDATE_SCHEMA)
    report.check(oracle.get("source") == "current-c", "oracle source drift")
    validate_operation_pair(report, oracle.get("operation"), candidate.get("operation"))
    validate_outputs_pair(report, oracle.get("outputs"), candidate.get("outputs"))


def validate_common(report: Report, obj: dict[str, Any], label: str, schema: str) -> None:
    report.check(obj.get("schema") == schema, f"{label} schema drift")
    report.check(obj.get("case") == CASE, f"{label} case drift")
    model = obj.get("model")
    report.check(isinstance(model, dict), f"{label} model missing")
    if isinstance(model, dict):
        report.check(model.get("mapped_size") == MODEL_SIZE, f"{label} model size drift")
        report.check(model.get("tensor_count") == 1328, f"{label} tensor count drift")
        report.check(model.get("tensor_data_offset") == TENSOR_DATA_OFFSET, f"{label} tensor-data offset drift")
        report.check(model.get("bound_layers") == 43, f"{label} bound layer count drift")

    operation = obj.get("operation")
    report.check(isinstance(operation, dict), f"{label} operation missing")
    if isinstance(operation, dict):
        for key, value in EXPECTED_OPERATION.items():
            report.check(operation.get(key) == value, f"{label} operation {key} drift")
        for key, value in EXPECTED_FLOAT_OPERATION.items():
            report.check(abs(float(operation.get(key, 0.0)) - value) <= 1e-7, f"{label} operation {key} drift")
        report.check(operation.get("directional_steering_attn_enabled") is True, f"{label} attention steering disabled")
        report.check(operation.get("directional_steering_ffn_enabled") is True, f"{label} FFN steering disabled")
        report.check(
            operation.get("directional_steering_fnv1a64") == STEERING_FNV1A64,
            f"{label} steering FNV drift",
        )
        report.check("dir-steering/out/verbosity.f32" in str(operation.get("directional_steering_file")), f"{label} steering path drift")

    outputs = obj.get("outputs")
    report.check(isinstance(outputs, dict), f"{label} outputs missing")
    if isinstance(outputs, dict):
        report.check(set(outputs) == set(OUTPUTS), f"{label} output tensor set drift")
        for field, elements in OUTPUTS.items():
            validate_output(report, outputs.get(field), label, field, elements)


def validate_output(report: Report, output: Any, label: str, field: str, elements: int) -> None:
    report.check(isinstance(output, dict), f"{label} {field} output missing")
    if not isinstance(output, dict):
        return
    report.check(output.get("field") == field, f"{label} {field} field drift")
    report.check(output.get("elements") == elements, f"{label} {field} element drift")
    report.check(output.get("bytes") == elements * 4, f"{label} {field} byte drift")
    report.check(isinstance(output.get("nonzero_elements"), int), f"{label} {field} nonzero count missing")
    report.check(is_fnv(output.get("fnv1a64")), f"{label} {field} FNV digest invalid")
    if label == "oracle":
        report.check(is_sha256(output.get("sha256")), f"oracle {field} sha256 invalid")
    expected = EXPECTED_FNV1A64.get(field)
    if expected is not None:
        report.check(output.get("fnv1a64") == expected, f"{label} {field} FNV digest drift")
    validate_samples(report, output.get("samples"), label, field, elements)


def validate_operation_pair(report: Report, oracle_operation: Any, candidate_operation: Any) -> None:
    report.check(isinstance(oracle_operation, dict), "oracle operation missing")
    report.check(isinstance(candidate_operation, dict), "candidate operation missing")
    if not isinstance(oracle_operation, dict) or not isinstance(candidate_operation, dict):
        return
    for key in EXPECTED_OPERATION:
        report.check(
            oracle_operation.get(key) == candidate_operation.get(key),
            f"candidate operation {key} does not match current-C oracle",
        )
    for key in EXPECTED_FLOAT_OPERATION:
        report.check(
            abs(float(oracle_operation.get(key, 0.0)) - float(candidate_operation.get(key, 0.0))) <= 1e-7,
            f"candidate operation {key} does not match current-C oracle",
        )
    for key in (
        "directional_steering_fnv1a64",
        "directional_steering_attn_enabled",
        "directional_steering_ffn_enabled",
    ):
        report.check(
            oracle_operation.get(key) == candidate_operation.get(key),
            f"candidate operation {key} does not match current-C oracle",
        )


def validate_outputs_pair(report: Report, oracle_outputs: Any, candidate_outputs: Any) -> None:
    report.check(isinstance(oracle_outputs, dict), "oracle outputs missing")
    report.check(isinstance(candidate_outputs, dict), "candidate outputs missing")
    if not isinstance(oracle_outputs, dict) or not isinstance(candidate_outputs, dict):
        return
    for field in OUTPUTS:
        oracle = oracle_outputs.get(field)
        candidate = candidate_outputs.get(field)
        if not isinstance(oracle, dict) or not isinstance(candidate, dict):
            continue
        report.check(
            oracle.get("fnv1a64") == candidate.get("fnv1a64"),
            f"candidate {field} FNV digest does not match current-C oracle",
        )
        report.check(
            oracle.get("nonzero_elements") == candidate.get("nonzero_elements"),
            f"candidate {field} nonzero count does not match current-C oracle",
        )
        report.check(
            samples_match(oracle.get("samples"), candidate.get("samples")),
            f"candidate {field} samples do not match current-C oracle",
        )


def validate_samples(report: Report, samples: Any, label: str, field: str, elements: int) -> None:
    report.check(isinstance(samples, list), f"{label} {field} samples missing")
    if not isinstance(samples, list):
        return
    by_index = sample_map(samples)
    expected_indices = set(sample_indices(elements))
    report.check(expected_indices <= set(by_index), f"{label} {field} sample set incomplete")
    for index in expected_indices:
        value = by_index.get(index)
        report.check(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"{label} {field} sample {index} is not finite",
        )


def sample_indices(elements: int) -> list[int]:
    raw = [0, 1, elements // 2, elements - 2 if elements > 1 else 0, elements - 1]
    out: list[int] = []
    for index in raw:
        if 0 <= index < elements and index not in out:
            out.append(index)
    return out


def sample_map(samples: Any) -> dict[int, int | float]:
    if not isinstance(samples, list):
        return {}
    out: dict[int, int | float] = {}
    for entry in samples:
        if isinstance(entry, dict) and isinstance(entry.get("index"), int):
            value = entry.get("value")
            if isinstance(value, (int, float)):
                out[entry["index"]] = value
    return out


def samples_match(left: Any, right: Any) -> bool:
    left_map = sample_map(left)
    right_map = sample_map(right)
    if set(left_map) != set(right_map):
        return False
    return all(
        math.isclose(float(value), float(right_map[index]), rel_tol=1e-6, abs_tol=1e-6)
        for index, value in left_map.items()
    )


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def is_fnv(value: Any) -> bool:
    return isinstance(value, str) and FNV.fullmatch(value) is not None


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return obj


def run_negative_tests(texts: dict[str, str]) -> int:
    static_mutations = [
        ("remove C helper call", "c_helper", "ds4_dump_directional_steering_decode_oracle_json", "removed"),
        ("remove C steering load", "ds4_c", "metal_graph_load_directional_steering", "removed_load"),
        ("remove C checkpoint", "ds4_c", "ds4_gpu_tensor_copy(layer0_ffn_out_tensor", "removed_copy("),
        ("remove Rust schema", "rust_bin", CANDIDATE_SCHEMA, "removed.schema"),
        ("remove Rust steering projection", "rust_bin", ".directional_steering_project(", ".removed("),
        ("remove facade wrapper", "decode_backend", "pub fn directional_steering_project", "pub fn removed_project"),
        ("remove report comparator", "report", "M10.5c4d4 Rust directional-steering decode comparator", "removed comparator"),
        ("remove roadmap fixture", "roadmap", "attention scale `0.5` and FFN scale", "attention scale removed"),
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

    pair_mutations = [
        ("candidate digest", mutate_candidate_digest),
        ("candidate sample value", mutate_candidate_sample),
        ("candidate tensor size", mutate_candidate_tensor_size),
        ("candidate steering hash", mutate_candidate_steering_hash),
        ("candidate layer count", mutate_candidate_layer_count),
    ]
    for label, mutate in pair_mutations:
        report = Report()
        oracle, candidate = valid_pair()
        mutate(candidate)
        validate_pair(report, oracle, candidate)
        if report.ok:
            failures.append(f"{label}: paired validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(static_mutations) + len(pair_mutations)} mutations rejected")
    return 0


def valid_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    oracle = valid_common(ORACLE_SCHEMA)
    oracle["source"] = "current-c"
    candidate = valid_common(CANDIDATE_SCHEMA)
    oracle["outputs"] = valid_outputs(include_sha=True)
    candidate["outputs"] = valid_outputs(include_sha=False)
    return oracle, candidate


def valid_common(schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "case": CASE,
        "model": {
            "mapped_size": MODEL_SIZE,
            "tensor_count": 1328,
            "tensor_data_offset": TENSOR_DATA_OFFSET,
            "bound_layers": 43,
        },
        "operation": {
            **EXPECTED_OPERATION,
            **EXPECTED_FLOAT_OPERATION,
            "directional_steering_file": "/workspace/ds4/dir-steering/out/verbosity.f32",
            "directional_steering_fnv1a64": "1234567890abcdef",
            "directional_steering_attn_enabled": True,
            "directional_steering_ffn_enabled": True,
        },
    }


def valid_outputs(include_sha: bool) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for idx, (field, elements) in enumerate(OUTPUTS.items()):
        output = {
            "field": field,
            "bytes": elements * 4,
            "elements": elements,
            "nonzero_elements": elements,
            "fnv1a64": f"{idx + 1:016x}",
            "samples": [{"index": sample, "value": float(idx + sample) / 10.0} for sample in sample_indices(elements)],
        }
        if include_sha:
            output["sha256"] = f"{idx:064x}"
        outputs[field] = output
    return outputs


def mutate_candidate_digest(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["layer0_attn_out"]["fnv1a64"] = "0000000000000000"


def mutate_candidate_sample(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["logits"]["samples"][0]["value"] += 0.01


def mutate_candidate_tensor_size(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["layer0_ffn_out"]["elements"] = 1


def mutate_candidate_steering_hash(candidate: dict[str, Any]) -> None:
    candidate["operation"]["directional_steering_fnv1a64"] = "0000000000000000"


def mutate_candidate_layer_count(candidate: dict[str, Any]) -> None:
    candidate["operation"]["decoded_layers"] = 42


def print_errors(errors: list[str]) -> None:
    print("Rust directional-steering comparator failures:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
