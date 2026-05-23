#!/usr/bin/env python3
"""Compare Rust first-kernel readback against the current-C embedding oracle."""

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
    "c_helper": ROOT / "ds4_first_kernel_oracle_dump.c",
    "ds4_c": ROOT / "ds4.c",
    "ds4_h": ROOT / "ds4.h",
    "makefile": ROOT / "Makefile",
    "rust_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-decode-first-kernel.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

ORACLE_SCHEMA = "ds4.first_kernel_oracle.v1"
CANDIDATE_SCHEMA = "ds4.decode_first_kernel.v1"
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
SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        print(f"Rust first-kernel current-C oracle comparator: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, help="JSON from ds4-first-kernel-oracle-dump")
    parser.add_argument("--candidate", type=Path, help="JSON from ds4-decode-first-kernel")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    c_helper = texts["c_helper"]
    for needle, message in (
        ("ds4_dump_first_kernel_oracle_json", "C helper does not call exported oracle dump"),
        ("--token", "C helper token flag missing"),
        ("--model", "C helper model flag missing"),
        ("--output", "C helper output flag missing"),
    ):
        report.check(needle in c_helper, message)

    ds4_c = texts["ds4_c"]
    for needle, message in (
        ("ds4_dump_first_kernel_oracle_json", "C oracle dump implementation missing"),
        ("model_open(&model", "C oracle does not use current model loader"),
        ("config_validate_model(&model)", "C oracle does not validate model config"),
        ("weights_bind(&weights, &model)", "C oracle does not bind current C weights"),
        ("embed_token_f16(&model, &weights", "C oracle does not use current embedding helper"),
        ("hc_from_plain_embedding(cur_hc", "C oracle does not use current HC broadcast helper"),
        ("ds4_sha256_hex(cur_hc", "C oracle output sha256 missing"),
        ("ds4_fnv1a64_bytes(cur_hc", "C oracle output FNV digest missing"),
        (ORACLE_SCHEMA, "C oracle schema missing"),
    ):
        report.check(needle in ds4_c, message)

    report.check(
        "int ds4_dump_first_kernel_oracle_json" in texts["ds4_h"],
        "header declaration for first-kernel oracle missing",
    )

    makefile = texts["makefile"]
    report.check("ds4-first-kernel-oracle-dump:" in makefile, "Makefile helper target missing")
    report.check("ds4_first_kernel_oracle_dump_cpu.o" in makefile, "CPU helper object missing")

    rust_bin = texts["rust_bin"]
    report.check("fnv1a64(&bytes)" in rust_bin, "Rust full-buffer FNV digest missing")
    report.check("fnv1a64" in rust_bin, "Rust JSON FNV field missing")

    report_text = texts["report"]
    report.check(
        "M10.5c4c2b2b2a Rust first-kernel current-C oracle comparator" in report_text,
        "unified report comparator missing",
    )
    report.check(
        "M10.5c4c2b2b2a B300 first-kernel current-C oracle rerun" in report_text,
        "unified report B300 skip missing",
    )
    report.check(
        "ds4-first-kernel-oracle-dump" in report_text and "compare_decode_first_kernel_oracle.py" in report_text,
        "B300 oracle rerun command missing helper/comparator",
    )

    report.check("M10.5c4c2b2b2a Rust first-kernel current-C oracle" in texts["readme"], "README entry missing")
    report.check("M10.5c4c2b2b2a: Rust First-Kernel Current-C Oracle Comparator" in texts["roadmap"], "roadmap split missing")
    report.check("M10.5c4c2b2b2b: Rust One-Token Decode B300 Execution" in texts["roadmap"], "roadmap remainder split missing")
    report.check("M10.5c4c2b2b2a: Rust First-Kernel Current-C Oracle Comparator" in texts["todo"], "TODO split missing")
    report.check("M10.5c4c2b2b2b: Rust One-Token Decode B300 Execution" in texts["todo"], "TODO remainder split missing")


def validate_pair(report: Report, oracle: dict[str, Any], candidate: dict[str, Any]) -> None:
    validate_common(report, oracle, "oracle", ORACLE_SCHEMA)
    validate_common(report, candidate, "candidate", CANDIDATE_SCHEMA)
    report.check(oracle.get("source") == "current-c", "oracle source drift")

    oracle_output = oracle.get("output") if isinstance(oracle.get("output"), dict) else {}
    candidate_output = candidate.get("output") if isinstance(candidate.get("output"), dict) else {}
    report.check(
        oracle_output.get("fnv1a64") == candidate_output.get("fnv1a64"),
        "candidate cur_hc FNV digest does not match current-C oracle",
    )
    report.check(
        oracle_output.get("nonzero_elements") == candidate_output.get("nonzero_elements"),
        "candidate nonzero count does not match current-C oracle",
    )
    report.check(
        samples_match(oracle_output.get("samples"), candidate_output.get("samples")),
        "candidate samples do not match current-C oracle",
    )


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
        report.check(operation.get("token") == 0, f"{label} token drift")
        report.check(operation.get("n_vocab") == 129280, f"{label} vocab drift")
        report.check(operation.get("n_embd") == 4096, f"{label} n_embd drift")
        report.check(operation.get("n_hc") == 4, f"{label} n_hc drift")

    weight = obj.get("weight")
    report.check(isinstance(weight, dict), f"{label} weight missing")
    if isinstance(weight, dict):
        report.check(weight.get("role") == "base.token_embd", f"{label} weight role drift")
        report.check(weight.get("abs_offset") == TOKEN_EMBD_OFFSET, f"{label} token embedding offset drift")
        report.check(weight.get("bytes") == TOKEN_EMBD_BYTES, f"{label} token embedding bytes drift")
        report.check(weight.get("type") == 1, f"{label} token embedding type drift")
        report.check(weight.get("type_name") == "f16", f"{label} token embedding type name drift")

    output = obj.get("output")
    report.check(isinstance(output, dict), f"{label} output missing")
    if isinstance(output, dict):
        report.check(output.get("field") == "cur_hc", f"{label} output field drift")
        report.check(output.get("bytes") == 65536, f"{label} output byte drift")
        report.check(output.get("elements") == 16384, f"{label} output element drift")
        report.check(output.get("nonzero_elements") == 16384, f"{label} output nonzero count drift")
        report.check(output.get("fnv1a64") == FNV1A64, f"{label} output FNV digest drift")
        if label == "oracle":
            report.check(is_sha256(output.get("sha256")), "oracle output sha256 invalid")
        validate_samples(report, output.get("samples"), label)


def validate_samples(report: Report, samples: Any, label: str) -> None:
    report.check(isinstance(samples, list), f"{label} samples missing")
    if not isinstance(samples, list):
        return
    by_index = sample_map(samples)
    report.check(set(SAMPLES) <= set(by_index), f"{label} sample set incomplete")
    for index, expected in SAMPLES.items():
        value = by_index.get(index)
        report.check(isinstance(value, (int, float)) and math.isfinite(value), f"{label} sample {index} is not finite")
        if isinstance(value, (int, float)) and math.isfinite(value):
            report.check(abs(float(value) - expected) <= 1e-6, f"{label} sample {index} value drift")


def sample_map(samples: Any) -> dict[int, float]:
    if not isinstance(samples, list):
        return {}
    out: dict[int, float] = {}
    for entry in samples:
        if isinstance(entry, dict) and isinstance(entry.get("index"), int):
            value = entry.get("value")
            if isinstance(value, (int, float)):
                out[entry["index"]] = float(value)
    return out


def samples_match(left: Any, right: Any) -> bool:
    left_map = sample_map(left)
    right_map = sample_map(right)
    if set(left_map) != set(right_map):
        return False
    for index, left_value in left_map.items():
        right_value = right_map[index]
        if abs(float(left_value) - float(right_value)) > 1e-6:
            return False
    return True


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


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
        ("remove C embed helper call", "ds4_c", "embed_token_f16(&model, &weights", "embed_token_removed(&model, &weights"),
        ("remove C HC broadcast helper call", "ds4_c", "hc_from_plain_embedding(cur_hc", "hc_from_plain_removed(cur_hc"),
        ("remove Make target", "makefile", "ds4-first-kernel-oracle-dump:", "ds4-first-kernel-oracle-removed:"),
        ("remove Rust FNV field", "rust_bin", "fnv1a64(&bytes)", "fnv1a64_removed(&bytes)"),
        (
            "remove B300 oracle command",
            "report",
            "M10.5c4c2b2b2a B300 first-kernel current-C oracle rerun",
            "M10.5c4c2b2b2a B300 removed",
        ),
        (
            "remove roadmap split",
            "roadmap",
            "M10.5c4c2b2b2a: Rust First-Kernel Current-C Oracle Comparator",
            "M10.5c4c2b2b2a removed",
        ),
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
        ("candidate FNV drift", lambda o, c: c["output"].__setitem__("fnv1a64", "fedcba9876543210")),
        ("candidate sample drift", lambda o, c: c["output"]["samples"][0].__setitem__("value", -0.5)),
        ("oracle token drift", lambda o, c: o["operation"].__setitem__("token", 1)),
        ("candidate output shape drift", lambda o, c: c["output"].__setitem__("bytes", 32768)),
    ]
    for label, mutate in pair_mutations:
        oracle = valid_oracle()
        candidate = valid_candidate()
        mutate(oracle, candidate)
        report = Report()
        validate_pair(report, oracle, candidate)
        if report.ok:
            failures.append(f"{label}: pair validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(static_mutations) + len(pair_mutations)} mutations rejected")
    return 0


def valid_oracle() -> dict[str, Any]:
    obj = valid_common(ORACLE_SCHEMA)
    obj["source"] = "current-c"
    obj["operation"]["name"] = "current_c_embed_token_f16_hc_from_plain_embedding"
    obj["operation"]["method"] = "embed_token_f16+hc_from_plain_embedding"
    obj["output"]["sha256"] = "a" * 64
    return obj


def valid_candidate() -> dict[str, Any]:
    obj = valid_common(CANDIDATE_SCHEMA)
    obj["operation"]["name"] = "ds4_gpu_embed_token_hc_tensor"
    obj["operation"]["method"] = "embed_token_hc"
    obj["operation"]["command_batch"] = True
    obj["operation"]["synchronized"] = True
    return obj


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


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
