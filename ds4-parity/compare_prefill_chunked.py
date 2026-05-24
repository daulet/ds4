#!/usr/bin/env python3
"""Compare Rust chunked-prefill readback against the current-C chunked oracle."""

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
    "c_helper": ROOT / "ds4_prefill_whole_short_oracle_dump.c",
    "ds4_c": ROOT / "ds4.c",
    "ds4_h": ROOT / "ds4.h",
    "makefile": ROOT / "Makefile",
    "rust_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-prefill-whole-short.rs",
    "decode_backend": ROOT / "rust/ds4-gpu/src/decode_backend.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

SCHEMA = "ds4.prefill_chunked.v1"
ORACLE_SCHEMA = "ds4.prefill_chunked_oracle.v1"

EXPECTED_OPERATION = {
    "long_memory_archive_2052_chunked_prefill": {
        "fixture": "long_memory_archive_2052",
        "prompt_tokens": 2052,
        "chunk_count": 2,
        "chunks": [(0, 2048, 2048), (2048, 4, 2052)],
        "output_abs_pos": 2051,
        "output_row": 3,
        "raw_row": 2051,
        "raw_start": 1924,
        "n_raw": 128,
        "layer2_n_comp": 513,
        "layer2_n_index_comp": 513,
        "layer5_n_comp": 16,
        "layer42_n_comp": 513,
        "layer42_n_index_comp": 513,
    },
    "long_memory_archive_chunked_prefill": {
        "fixture": "long_memory_archive",
        "prompt_tokens": 3353,
        "chunk_count": 2,
        "chunks": [(0, 2048, 2048), (2048, 1305, 3353)],
        "output_abs_pos": 3352,
        "output_row": 1304,
        "raw_row": 1048,
        "raw_start": 921,
        "n_raw": 128,
        "layer2_n_comp": 838,
        "layer2_n_index_comp": 838,
        "layer5_n_comp": 26,
        "layer42_n_comp": 838,
        "layer42_n_index_comp": 838,
    },
}

OUTPUTS = {
    "after_layer42_hc": 16384,
    "output_pre": 4,
    "output_weights": 4,
    "output_embd": 4096,
    "output_norm": 4096,
    "logits": 129280,
    "layer2_raw_cache_row": 512,
    "layer2_attn_comp_row4": 512,
    "layer2_index_comp_row4": 128,
    "layer5_attn_state_kv": 65536,
    "layer5_attn_state_score": 65536,
    "layer42_raw_cache_row": 512,
    "layer42_attn_comp_row4": 512,
    "layer42_index_comp_row4": 128,
    "layer42_attn_state_kv": 8192,
    "layer42_index_state_kv": 2048,
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
    if args.oracle is not None or args.candidate is not None:
        if args.candidate is None:
            raise SystemExit("--candidate is required when --oracle is provided")
        candidate = load_json(args.candidate)
        validate_candidate(report, candidate)
        if args.oracle is not None:
            validate_pair(report, load_json(args.oracle), candidate)

    if report.ok:
        print(f"Rust chunked-prefill current-C oracle comparator: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, help="JSON from ds4-prefill-whole-short-oracle-dump")
    parser.add_argument("--candidate", type=Path, help="JSON from ds4-prefill-whole-short")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    c_helper = texts["c_helper"]
    for needle, message in (
        ("ds4_dump_prefill_whole_short_oracle_json", "C helper does not call prefill oracle dump"),
        ("--limit-tokens", "C helper token-limit flag missing"),
        ("--output", "C helper output flag missing"),
    ):
        report.check(needle in c_helper, message)

    ds4_c = texts["ds4_c"]
    for needle, message in (
        (ORACLE_SCHEMA, "C chunked oracle schema missing"),
        ("long_memory_archive_2052_chunked_prefill", "C 2052-token chunked case missing"),
        ("long_memory_archive_chunked_prefill", "C full chunked case missing"),
        ("\\\"chunk_count\\\"", "C chunked oracle does not emit chunk count"),
        ("\\\"chunks\\\"", "C chunked oracle does not emit chunk schedule"),
        ("\\\"output_abs_pos\\\"", "C chunked oracle does not emit absolute output position"),
        ("metal_graph_prefill_chunked_range", "C oracle does not use production chunked prefill"),
    ):
        report.check(needle in ds4_c, message)
    report.check(
        "int ds4_dump_prefill_whole_short_oracle_json" in texts["ds4_h"],
        "header declaration for prefill oracle missing",
    )
    report.check("ds4-prefill-whole-short-oracle-dump:" in texts["makefile"], "Makefile helper target missing")

    rust_bin = texts["rust_bin"]
    for needle, message in (
        (SCHEMA, "Rust chunked schema missing"),
        ("CHUNKED_2052_CASE", "2052-token chunked case missing"),
        ("--limit-tokens", "token-limit fixture flag missing"),
        ("prefill_chunks", "chunk schedule helper missing"),
        ("raw_span_for_batch", "raw span helper not used"),
        ("raw_start_for_span", "raw-start helper not used"),
        ("compressor_prefill_ratio4_replay", "ratio4 replay path missing"),
        ("indexer_scores_decode_batch", "decode-batch indexer scoring missing"),
        ("attention_decode_raw_batch_heads", "nonzero raw batch attention missing"),
        ("attention_decode_mixed_batch_heads", "nonzero mixed batch attention missing"),
    ):
        report.check(needle in rust_bin, message)

    decode_backend = texts["decode_backend"]
    for needle, message in (
        ("ds4_gpu_compressor_prefill_ratio4_replay_tensor", "ratio4 replay wrapper missing"),
        ("ds4_gpu_indexer_scores_decode_batch_tensor", "indexer decode-batch wrapper missing"),
        ("ds4_gpu_attention_decode_raw_batch_heads_tensor", "raw batch attention wrapper missing"),
        ("ds4_gpu_attention_decode_mixed_batch_heads_tensor", "mixed batch attention wrapper missing"),
    ):
        report.check(needle in decode_backend, message)

    report.check(
        "M10.6c Rust chunked-prefill comparator" in texts["report"],
        "unified report entry missing",
    )
    report.check("compare_prefill_chunked.py" in texts["readme"], "README command missing")
    report.check(
        "M10.6c: Rust Cold Chunked-Prefill Execution" in texts["roadmap"],
        "roadmap M10.6c section missing",
    )
    report.check(
        "M10.6c: Rust Cold Chunked-Prefill Execution" in texts["todo"],
        "TODO M10.6c section missing",
    )


def validate_candidate(report: Report, candidate: Any) -> None:
    report.check(isinstance(candidate, dict), "candidate JSON root must be an object")
    if not isinstance(candidate, dict):
        return
    report.check(candidate.get("schema") == SCHEMA, "candidate schema drift")
    case = candidate.get("case")
    report.check(case in EXPECTED_OPERATION, "candidate case is not an M10.6c chunked case")
    if case not in EXPECTED_OPERATION:
        return

    operation = candidate.get("operation")
    report.check(isinstance(operation, dict), "candidate operation missing")
    if isinstance(operation, dict):
        expected = EXPECTED_OPERATION[case]
        report.check(operation.get("boundary") == "chunked_prefill", "boundary drift")
        for key, value in expected.items():
            if key == "chunks":
                got = [
                    (chunk.get("start"), chunk.get("n_tokens"), chunk.get("end"))
                    for chunk in operation.get("chunks", [])
                    if isinstance(chunk, dict)
                ]
                report.check(got == value, "chunk schedule drift")
            else:
                report.check(operation.get(key) == value, f"operation {key} drift")
        report.check(operation.get("prefill_cap") == 2048, "prefill cap drift")
        report.check(operation.get("raw_cap") == 2304, "raw cap drift")
        report.check(operation.get("raw_window") == 128, "raw window drift")

    outputs = candidate.get("outputs")
    report.check(isinstance(outputs, dict), "candidate outputs missing")
    if isinstance(outputs, dict):
        report.check(set(outputs) == set(OUTPUTS), "candidate output tensor set drift")
        for field, elements in OUTPUTS.items():
            validate_output(report, outputs.get(field), field, elements)


def validate_oracle(report: Report, oracle: Any) -> None:
    report.check(isinstance(oracle, dict), "oracle JSON root must be an object")
    if not isinstance(oracle, dict):
        return
    report.check(oracle.get("schema") == ORACLE_SCHEMA, "oracle schema drift")
    report.check(oracle.get("source") == "current-c", "oracle source drift")
    case = oracle.get("case")
    report.check(case in EXPECTED_OPERATION, "oracle case is not an M10.6c chunked case")
    if case not in EXPECTED_OPERATION:
        return

    operation = oracle.get("operation")
    report.check(isinstance(operation, dict), "oracle operation missing")
    if isinstance(operation, dict):
        validate_operation_shape(report, operation, case, "oracle")

    outputs = oracle.get("outputs")
    report.check(isinstance(outputs, dict), "oracle outputs missing")
    if isinstance(outputs, dict):
        report.check(set(outputs) == set(OUTPUTS), "oracle output tensor set drift")
        for field, elements in OUTPUTS.items():
            validate_output(report, outputs.get(field), field, elements, "oracle")


def validate_pair(report: Report, oracle: Any, candidate: Any) -> None:
    validate_oracle(report, oracle)
    if not isinstance(oracle, dict) or not isinstance(candidate, dict):
        return
    report.check(oracle.get("case") == candidate.get("case"), "candidate case does not match oracle")
    oracle_operation = oracle.get("operation")
    candidate_operation = candidate.get("operation")
    if isinstance(oracle_operation, dict) and isinstance(candidate_operation, dict):
        for key in EXPECTED_OPERATION.get(candidate.get("case"), {}):
            if key == "chunks":
                report.check(
                    normalize_chunks(oracle_operation.get("chunks")) == normalize_chunks(candidate_operation.get("chunks")),
                    "candidate chunks do not match current-C oracle",
                )
            else:
                report.check(
                    oracle_operation.get(key) == candidate_operation.get(key),
                    f"candidate operation {key} does not match current-C oracle",
                )
        for key in ("prefill_cap", "raw_cap", "raw_window"):
            report.check(
                oracle_operation.get(key) == candidate_operation.get(key),
                f"candidate operation {key} does not match current-C oracle",
            )

    oracle_weights = oracle.get("weights")
    candidate_weights = candidate.get("weights")
    if isinstance(oracle_weights, dict) and isinstance(candidate_weights, dict):
        report.check(set(oracle_weights) == set(candidate_weights), "candidate weight set does not match oracle")
        for key, oracle_weight in oracle_weights.items():
            candidate_weight = candidate_weights.get(key)
            if not isinstance(oracle_weight, dict) or not isinstance(candidate_weight, dict):
                report.check(False, f"{key} weight missing in oracle or candidate")
                continue
            for field in ("role", "abs_offset", "bytes", "type", "type_name"):
                report.check(
                    oracle_weight.get(field) == candidate_weight.get(field),
                    f"candidate {key} weight {field} does not match current-C oracle",
                )

    oracle_outputs = oracle.get("outputs")
    candidate_outputs = candidate.get("outputs")
    if isinstance(oracle_outputs, dict) and isinstance(candidate_outputs, dict):
        for field in OUTPUTS:
            oracle_output = oracle_outputs.get(field)
            candidate_output = candidate_outputs.get(field)
            if not isinstance(oracle_output, dict) or not isinstance(candidate_output, dict):
                report.check(False, f"{field} missing in oracle or candidate")
                continue
            report.check(
                oracle_output.get("fnv1a64") == candidate_output.get("fnv1a64"),
                f"candidate {field} FNV digest does not match current-C oracle",
            )
            report.check(
                oracle_output.get("nonzero_elements") == candidate_output.get("nonzero_elements"),
                f"candidate {field} nonzero count does not match current-C oracle",
            )
            report.check(
                samples_match(oracle_output.get("samples"), candidate_output.get("samples")),
                f"candidate {field} samples do not match current-C oracle",
            )


def validate_operation_shape(report: Report, operation: dict[str, Any], case: str, label: str) -> None:
    expected = EXPECTED_OPERATION[case]
    report.check(operation.get("boundary") == "chunked_prefill", f"{label} boundary drift")
    for key, value in expected.items():
        if key == "chunks":
            report.check(normalize_chunks(operation.get("chunks")) == value, f"{label} chunk schedule drift")
        else:
            report.check(operation.get(key) == value, f"{label} operation {key} drift")
    report.check(operation.get("prefill_cap") == 2048, f"{label} prefill cap drift")
    report.check(operation.get("raw_cap") == 2304, f"{label} raw cap drift")
    report.check(operation.get("raw_window") == 128, f"{label} raw window drift")


def validate_output(report: Report, output: Any, field: str, elements: int, label: str = "candidate") -> None:
    report.check(isinstance(output, dict), f"{field} output missing")
    if not isinstance(output, dict):
        return
    report.check(output.get("field") == field, f"{label} {field} field drift")
    report.check(output.get("elements") == elements, f"{label} {field} element drift")
    report.check(output.get("bytes") == elements * 4, f"{label} {field} byte drift")
    report.check(isinstance(output.get("fnv1a64"), str), f"{label} {field} FNV digest missing")
    report.check(isinstance(output.get("nonzero_elements"), int), f"{label} {field} nonzero count missing")
    samples = output.get("samples")
    report.check(isinstance(samples, list) and samples, f"{label} {field} samples missing")


def normalize_chunks(chunks: Any) -> list[tuple[Any, Any, Any]]:
    if not isinstance(chunks, list):
        return []
    return [
        (chunk.get("start"), chunk.get("n_tokens"), chunk.get("end"))
        for chunk in chunks
        if isinstance(chunk, dict)
    ]


def samples_match(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    left_map = sample_map(left)
    right_map = sample_map(right)
    if set(left_map) != set(right_map):
        return False
    for index, left_value in left_map.items():
        right_value = right_map[index]
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            if not math.isclose(float(left_value), float(right_value), abs_tol=1.0e-6):
                return False
        elif left_value != right_value:
            return False
    return True


def sample_map(samples: list[Any]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for sample in samples:
        if isinstance(sample, dict):
            out[sample.get("index")] = sample.get("value")
    return out


def run_negative_tests(texts: dict[str, str]) -> int:
    failures: list[str] = []
    cases = [
        ("remove chunked schema", "rust_bin", SCHEMA, "schema"),
        (
            "remove replay wrapper",
            "decode_backend",
            "ds4_gpu_compressor_prefill_ratio4_replay_tensor",
            "ratio4 replay",
        ),
        ("remove report entry", "report", "M10.6c Rust chunked-prefill comparator", "report"),
    ]
    for name, key, needle, expected_error in cases:
        mutated = dict(texts)
        mutated[key] = mutated[key].replace(needle, "")
        report = Report()
        validate_static(report, mutated)
        if not any(expected_error in error for error in report.errors):
            failures.append(f"{name}: expected an error containing {expected_error!r}")

    candidate = valid_candidate()
    candidate["operation"]["chunks"][1]["n_tokens"] = 1304
    report = Report()
    validate_candidate(report, candidate)
    if not any("chunk schedule" in error for error in report.errors):
        failures.append("candidate chunk schedule mutation was not detected")

    candidate = valid_candidate()
    oracle = valid_oracle()
    candidate["outputs"]["logits"]["fnv1a64"] = "f" * 16
    report = Report()
    validate_pair(report, oracle, candidate)
    if not any("logits FNV" in error for error in report.errors):
        failures.append("candidate-oracle logits mutation was not detected")

    if failures:
        print_errors(failures)
        return 1
    print(f"Rust chunked-prefill negative tests: {len(cases) + 2} mutations rejected")
    return 0


def valid_candidate() -> dict[str, Any]:
    outputs = {
        field: {
            "field": field,
            "bytes": elements * 4,
            "elements": elements,
            "nonzero_elements": 1,
            "fnv1a64": "0" * 16,
            "samples": [
                {"index": 0, "value": 0.0},
                {"index": 1, "value": 0.0},
                {"index": elements // 2, "value": 0.0},
                {"index": elements - 2, "value": 0.0},
                {"index": elements - 1, "value": 0.0},
            ],
        }
        for field, elements in OUTPUTS.items()
    }
    return {
        "schema": SCHEMA,
        "case": "long_memory_archive_chunked_prefill",
        "operation": {
            "boundary": "chunked_prefill",
            "fixture": "long_memory_archive",
            "prompt_tokens": 3353,
            "chunk_count": 2,
            "chunks": [
                {"start": 0, "n_tokens": 2048, "end": 2048},
                {"start": 2048, "n_tokens": 1305, "end": 3353},
            ],
            "output_abs_pos": 3352,
            "output_row": 1304,
            "prefill_cap": 2048,
            "raw_cap": 2304,
            "raw_window": 128,
            "raw_row": 1048,
            "raw_start": 921,
            "n_raw": 128,
            "layer2_n_comp": 838,
            "layer2_n_index_comp": 838,
            "layer5_n_comp": 26,
            "layer42_n_comp": 838,
            "layer42_n_index_comp": 838,
        },
        "outputs": outputs,
    }


def valid_oracle() -> dict[str, Any]:
    oracle = copy.deepcopy(valid_candidate())
    oracle["schema"] = ORACLE_SCHEMA
    oracle["source"] = "current-c"
    return oracle


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def print_errors(errors: list[str]) -> None:
    print("Rust chunked-prefill comparator failures:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
