#!/usr/bin/env python3
"""Compare Rust resumed-prefill readback against the current-C session oracle."""

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
    "rust_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-prefill-whole-short.rs",
    "decode_backend": ROOT / "rust/ds4-gpu/src/decode_backend.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

CANDIDATE_SCHEMA = "ds4.prefill_resumed.v1"
ORACLE_SCHEMA = "ds4.prefill_resumed_oracle.v1"

EXPECTED_OPERATION = {
    "long_memory_archive_exact_prefix_cache_hit": {
        "boundary": "cache_hit",
        "fixture": "long_memory_archive_prefix_512",
        "prompt_tokens": 512,
        "prefix_tokens": 512,
        "suffix_tokens": 0,
        "checkpoint_tokens_before": 512,
        "checkpoint_tokens_after": 512,
        "resume_min_tokens": 4,
        "decode_tokens": 0,
        "chunk_count": 0,
        "chunks": [],
        "prefix_prefill_layer_calls": 43,
        "prefill_layer_calls": 0,
        "decode_layer_calls": 0,
        "output_abs_pos": 511,
        "output_row": 511,
        "raw_row": 511,
        "raw_start": 384,
        "n_raw": 128,
        "layer2_n_comp": 128,
        "layer2_n_index_comp": 128,
        "layer5_n_comp": 4,
        "layer42_n_comp": 128,
        "layer42_n_index_comp": 128,
    },
    "long_memory_archive_short_resume_decode_suffix": {
        "boundary": "decode_suffix",
        "fixture": "long_memory_archive_512_to_514",
        "prompt_tokens": 514,
        "prefix_tokens": 512,
        "suffix_tokens": 2,
        "checkpoint_tokens_before": 512,
        "checkpoint_tokens_after": 514,
        "resume_min_tokens": 4,
        "decode_tokens": 2,
        "chunk_count": 0,
        "chunks": [],
        "prefix_prefill_layer_calls": 43,
        "prefill_layer_calls": 0,
        "decode_layer_calls": 86,
        "output_abs_pos": 513,
        "output_row": 0,
        "raw_row": 513,
        "raw_start": 386,
        "n_raw": 128,
        "layer2_n_comp": 128,
        "layer2_n_index_comp": 128,
        "layer5_n_comp": 4,
        "layer42_n_comp": 128,
        "layer42_n_index_comp": 128,
    },
    "long_memory_archive_resume_chunked_boundary": {
        "boundary": "resumed_chunked_prefill",
        "fixture": "long_memory_archive_1537_to_2337",
        "prompt_tokens": 2337,
        "prefix_tokens": 1537,
        "suffix_tokens": 800,
        "checkpoint_tokens_before": 1537,
        "checkpoint_tokens_after": 2337,
        "resume_min_tokens": 4,
        "decode_tokens": 0,
        "chunk_count": 2,
        "chunks": [(1537, 511, 2048), (2048, 289, 2337)],
        "prefix_prefill_layer_calls": 43,
        "prefill_layer_calls": 86,
        "decode_layer_calls": 0,
        "output_abs_pos": 2336,
        "output_row": 288,
        "raw_row": 32,
        "raw_start": 2209,
        "n_raw": 128,
        "layer2_n_comp": 584,
        "layer2_n_index_comp": 584,
        "layer5_n_comp": 18,
        "layer42_n_comp": 584,
        "layer42_n_index_comp": 584,
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
        print(f"Rust resumed-prefill current-C oracle comparator: {report.checks} checks")
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
        ("--resume-prefix-tokens", "C helper resume-prefix flag missing"),
        ("resume_prefix_tokens", "C helper does not pass resume-prefix tokens"),
        ("--limit-tokens", "C helper token-limit flag missing"),
    ):
        report.check(needle in c_helper, message)

    ds4_c = texts["ds4_c"]
    for needle, message in (
        (ORACLE_SCHEMA, "C resumed oracle schema missing"),
        ("ds4_session_sync(s, &resume_prefix", "C oracle does not sync prefix first"),
        ("long_memory_archive_exact_prefix_cache_hit", "C exact-prefix resumed case missing"),
        ("long_memory_archive_short_resume_decode_suffix", "C decode-suffix resumed case missing"),
        ("long_memory_archive_resume_chunked_boundary", "C chunked resumed case missing"),
        ("metal_graph_resume_prefill_min_tokens", "C oracle does not report resume threshold"),
        ("\\\"checkpoint_tokens_before\\\"", "C oracle does not emit checkpoint start"),
        ("\\\"decode_tokens\\\"", "C oracle does not emit decode token count"),
    ):
        report.check(needle in ds4_c, message)
    report.check(
        "resume_prefix_tokens" in texts["ds4_h"],
        "header declaration for resumed prefill oracle missing",
    )

    rust_bin = texts["rust_bin"]
    for needle, message in (
        (CANDIDATE_SCHEMA, "Rust resumed schema missing"),
        ("--resume-prefix-tokens", "Rust resume-prefix flag missing"),
        ("ExecutionRoute::CacheHit", "Rust cache-hit route missing"),
        ("ExecutionRoute::DecodeSuffix", "Rust decode-suffix route missing"),
        ("ExecutionRoute::ResumedChunked", "Rust resumed-chunked route missing"),
        ("execute_decode_suffix", "Rust decode suffix execution helper missing"),
        ("execute_prefill_chunks", "Rust prefill chunk execution helper missing"),
        ("indexer_score_one", "Rust decode suffix indexed attention scoring missing"),
        ("attention_indexed_mixed_batch_heads", "Rust decode suffix indexed attention missing"),
        ("RESUME_PREFILL_MIN_TOKENS", "Rust resume threshold constant missing"),
    ):
        report.check(needle in rust_bin, message)

    report.check(
        "M10.6d Rust resumed-prefill comparator" in texts["report"],
        "unified report entry missing",
    )
    report.check("compare_prefill_resumed.py" in texts["readme"], "README command missing")
    report.check(
        "M10.6d: Rust Resumed-Suffix Prefill Execution" in texts["roadmap"],
        "roadmap M10.6d section missing",
    )
    report.check(
        "long_memory_archive_1537_to_2337" in texts["todo"],
        "TODO M10.6d concrete fixture missing",
    )
    report.check("indexer_score_one" in texts["decode_backend"], "decode backend indexer wrapper missing")


def validate_candidate(report: Report, candidate: Any) -> None:
    validate_root(report, candidate, CANDIDATE_SCHEMA, "candidate")


def validate_oracle(report: Report, oracle: Any) -> None:
    validate_root(report, oracle, ORACLE_SCHEMA, "oracle")
    if isinstance(oracle, dict):
        report.check(oracle.get("source") == "current-c", "oracle source drift")


def validate_root(report: Report, obj: Any, schema: str, label: str) -> None:
    report.check(isinstance(obj, dict), f"{label} JSON root must be an object")
    if not isinstance(obj, dict):
        return
    report.check(obj.get("schema") == schema, f"{label} schema drift")
    case = obj.get("case")
    report.check(case in EXPECTED_OPERATION, f"{label} case is not an M10.6d resumed case")
    if case not in EXPECTED_OPERATION:
        return
    operation = obj.get("operation")
    report.check(isinstance(operation, dict), f"{label} operation missing")
    if isinstance(operation, dict):
        validate_operation_shape(report, operation, case, label)
    outputs = obj.get("outputs")
    report.check(isinstance(outputs, dict), f"{label} outputs missing")
    if isinstance(outputs, dict):
        report.check(set(outputs) == set(OUTPUTS), f"{label} output tensor set drift")
        for field, elements in OUTPUTS.items():
            validate_output(report, outputs.get(field), field, elements, label)


def validate_pair(report: Report, oracle: Any, candidate: Any) -> None:
    validate_oracle(report, oracle)
    if not isinstance(oracle, dict) or not isinstance(candidate, dict):
        return
    report.check(oracle.get("case") == candidate.get("case"), "candidate case does not match oracle")
    oracle_operation = oracle.get("operation")
    candidate_operation = candidate.get("operation")
    if isinstance(oracle_operation, dict) and isinstance(candidate_operation, dict):
        expected = EXPECTED_OPERATION.get(candidate.get("case"), {})
        for key in expected:
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
    for key, value in expected.items():
        if key == "chunks":
            report.check(normalize_chunks(operation.get("chunks")) == value, f"{label} chunk schedule drift")
        else:
            report.check(operation.get(key) == value, f"{label} operation {key} drift")
    report.check(operation.get("prefill_cap") == 2048, f"{label} prefill cap drift")
    report.check(operation.get("raw_cap") == 2304, f"{label} raw cap drift")
    report.check(operation.get("raw_window") == 128, f"{label} raw window drift")


def validate_output(report: Report, output: Any, field: str, elements: int, label: str) -> None:
    report.check(isinstance(output, dict), f"{label} {field} output missing")
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
        ("remove resumed schema", "rust_bin", CANDIDATE_SCHEMA, "schema"),
        ("remove C resume flag", "c_helper", "--resume-prefix-tokens", "resume-prefix"),
        ("remove report entry", "report", "M10.6d Rust resumed-prefill comparator", "report"),
    ]
    for name, key, needle, expected_error in cases:
        mutated = dict(texts)
        mutated[key] = mutated[key].replace(needle, "")
        report = Report()
        validate_static(report, mutated)
        if not any(expected_error in error for error in report.errors):
            failures.append(f"{name}: expected an error containing {expected_error!r}")

    candidate = valid_candidate("long_memory_archive_resume_chunked_boundary")
    candidate["operation"]["chunks"][0]["n_tokens"] = 512
    report = Report()
    validate_candidate(report, candidate)
    if not any("chunk schedule" in error for error in report.errors):
        failures.append("candidate resumed chunk mutation was not detected")

    candidate = valid_candidate("long_memory_archive_short_resume_decode_suffix")
    candidate["operation"]["decode_tokens"] = 1
    report = Report()
    validate_candidate(report, candidate)
    if not any("decode_tokens" in error for error in report.errors):
        failures.append("candidate decode-token mutation was not detected")

    candidate = valid_candidate("long_memory_archive_exact_prefix_cache_hit")
    oracle = valid_oracle("long_memory_archive_exact_prefix_cache_hit")
    candidate["outputs"]["logits"]["fnv1a64"] = "f" * 16
    report = Report()
    validate_pair(report, oracle, candidate)
    if not any("logits FNV" in error for error in report.errors):
        failures.append("candidate-oracle logits mutation was not detected")

    if failures:
        print_errors(failures)
        return 1
    print(f"Rust resumed-prefill negative tests: {len(cases) + 3} mutations rejected")
    return 0


def valid_candidate(case: str) -> dict[str, Any]:
    expected = EXPECTED_OPERATION[case]
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
    operation = dict(expected)
    operation["chunks"] = [
        {"start": start, "n_tokens": n_tokens, "end": end}
        for start, n_tokens, end in expected["chunks"]
    ]
    operation["prefill_cap"] = 2048
    operation["raw_cap"] = 2304
    operation["raw_window"] = 128
    return {
        "schema": CANDIDATE_SCHEMA,
        "case": case,
        "operation": operation,
        "outputs": outputs,
    }


def valid_oracle(case: str) -> dict[str, Any]:
    oracle = copy.deepcopy(valid_candidate(case))
    oracle["schema"] = ORACLE_SCHEMA
    oracle["source"] = "current-c"
    return oracle


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def print_errors(errors: list[str]) -> None:
    print("Rust resumed-prefill comparator failures:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
