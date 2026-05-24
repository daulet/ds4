#!/usr/bin/env python3
"""Compare Rust M10.6a prefill scheduling plan against current-C behavior."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "prefill_plan": ROOT / "rust/ds4-gpu/src/prefill_plan.rs",
    "prefill_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-prefill-plan.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

SCHEMA = "ds4.prefill_plan.v1"
SCOPE = "m10.6a"
EXPECTED_CASES: dict[str, dict[str, Any]] = {
    "cold_whole_prompt_22": {
        "input": {
            "ctx_size": 32768,
            "prompt_len": 22,
            "start": 0,
            "n_tokens": 22,
            "checkpoint_valid": False,
        },
        "route": "whole_layer_major",
        "prefill_cap": 22,
        "raw_cap": 256,
        "chunk_cap": 22,
        "first_chunk_tokens": 22,
        "chunk_count": 1,
        "final_output_batch_row": 21,
        "output_absolute_pos": 21,
        "progress_point_count": 0,
        "layer_batch_calls": 43,
        "chunks": [{"start": 0, "tokens": 22}],
        "progress_points": [],
    },
    "cold_whole_prefill_cap_boundary": {
        "input": {
            "ctx_size": 32768,
            "prompt_len": 2048,
            "start": 0,
            "n_tokens": 2048,
            "checkpoint_valid": False,
        },
        "route": "whole_layer_major",
        "prefill_cap": 2048,
        "raw_cap": 2304,
        "chunk_cap": 2048,
        "first_chunk_tokens": 2048,
        "chunk_count": 1,
        "final_output_batch_row": 2047,
        "output_absolute_pos": 2047,
        "progress_point_count": 0,
        "layer_batch_calls": 43,
        "chunks": [{"start": 0, "tokens": 2048}],
        "progress_points": [],
    },
    "cold_chunked_2052_crosses_prefill_cap": {
        "input": {
            "ctx_size": 32768,
            "prompt_len": 2052,
            "start": 0,
            "n_tokens": 2052,
            "checkpoint_valid": False,
        },
        "route": "chunked_range",
        "prefill_cap": 2048,
        "raw_cap": 2304,
        "chunk_cap": 2048,
        "first_chunk_tokens": 2048,
        "chunk_count": 2,
        "final_output_batch_row": 3,
        "output_absolute_pos": 2051,
        "progress_point_count": 3,
        "layer_batch_calls": 86,
        "chunks": [{"start": 0, "tokens": 2048}, {"start": 2048, "tokens": 4}],
        "progress_points": [0, 2048, 2052],
    },
    "resume_suffix_aligns_to_prefill_boundary": {
        "input": {
            "ctx_size": 32768,
            "prompt_len": 4096,
            "start": 1537,
            "n_tokens": 800,
            "checkpoint_valid": True,
        },
        "route": "chunked_range",
        "prefill_cap": 2048,
        "raw_cap": 2304,
        "chunk_cap": 2048,
        "first_chunk_tokens": 511,
        "chunk_count": 2,
        "final_output_batch_row": 288,
        "output_absolute_pos": 2336,
        "progress_point_count": 3,
        "layer_batch_calls": 86,
        "chunks": [{"start": 1537, "tokens": 511}, {"start": 2048, "tokens": 289}],
        "progress_points": [1537, 2048, 2337],
    },
    "resume_short_suffix_uses_decode": {
        "input": {
            "ctx_size": 32768,
            "prompt_len": 4096,
            "start": 512,
            "n_tokens": 2,
            "checkpoint_valid": True,
        },
        "route": "decode_suffix",
        "prefill_cap": 2048,
        "raw_cap": 2304,
        "chunk_cap": 0,
        "first_chunk_tokens": 0,
        "chunk_count": 0,
        "final_output_batch_row": None,
        "output_absolute_pos": None,
        "progress_point_count": 0,
        "layer_batch_calls": 86,
        "chunks": [],
        "progress_points": [],
    },
    "checkpoint_exact_prefix_cache_hit": {
        "input": {
            "ctx_size": 32768,
            "prompt_len": 4096,
            "start": 4096,
            "n_tokens": 0,
            "checkpoint_valid": True,
        },
        "route": "cache_hit",
        "prefill_cap": 2048,
        "raw_cap": 2304,
        "chunk_cap": 0,
        "first_chunk_tokens": 0,
        "chunk_count": 0,
        "final_output_batch_row": None,
        "output_absolute_pos": None,
        "progress_point_count": 0,
        "layer_batch_calls": 0,
        "chunks": [],
        "progress_points": [],
    },
}
ROUTE_NAMES = {
    "WholeLayerMajor": "whole_layer_major",
    "ChunkedRange": "chunked_range",
    "DecodeSuffix": "decode_suffix",
    "CacheHit": "cache_hit",
    "Invalid": "invalid",
}


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    texts = {name: path.read_text() for name, path in FILES.items()}
    if args.negative_test:
        return run_negative_tests(texts)

    errors: list[str] = []
    validate_static(errors, texts)
    if args.candidate is not None:
        validate_candidate(errors, load_json(args.candidate))

    if errors:
        print_errors(errors)
    else:
        chunks = sum(len(case["chunks"]) for case in EXPECTED_CASES.values())
        progress = sum(len(case["progress_points"]) for case in EXPECTED_CASES.values())
        suffix = "" if args.candidate is None else ", candidate JSON checked"
        print(
            "Rust prefill plan comparator: "
            f"{len(EXPECTED_CASES)} cases, {chunks} chunks, "
            f"{progress} progress points{suffix}"
        )
    return 1 if errors else 0


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="JSON from ds4-prefill-plan")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(errors: list[str], texts: dict[str, str]) -> None:
    source = texts["prefill_plan"]
    for needle, message in (
        (SCHEMA, "prefill plan schema missing"),
        ("RESUME_PREFILL_MIN_TOKENS: u32 = 4", "resume prefill threshold drift"),
        ("GraphPlan::for_context", "prefill plan does not use graph capacity plan"),
        ("input.n_tokens < RESUME_PREFILL_MIN_TOKENS", "resume decode threshold check missing"),
        ("prefill_cap - boundary_offset", "absolute prefill-boundary alignment missing"),
        ("N_LAYER as u32", "layer batch-call accounting missing"),
    ):
        if needle not in source:
            errors.append(message)

    cases = parse_rust_cases(source)
    compare_case_maps(errors, cases)

    prefill_bin = texts["prefill_bin"]
    for needle, message in (
        ("PREFILL_PLAN_SCHEMA", "prefill plan JSON bin schema missing"),
        ("computed", "prefill plan JSON bin computed section missing"),
        ("expected", "prefill plan JSON bin expected section missing"),
    ):
        if needle not in prefill_bin:
            errors.append(message)

    for key, needle, message in (
        ("report", "M10.6a Rust prefill scheduling plan comparator", "unified report entry missing"),
        ("readme", "compare_prefill_plan_rust.py", "README prefill comparator docs missing"),
        ("roadmap", "M10.6a: Rust Prefill Scheduling Plan", "roadmap split missing M10.6a"),
        ("todo", "M10.6a: Rust Prefill Scheduling Plan", "TODO split missing M10.6a"),
    ):
        if needle not in texts[key]:
            errors.append(message)


def parse_rust_cases(source: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r'case!\(\s*"(?P<name>[^"]+)"\s*,'
        r"\s*(?P<ctx_size>[0-9]+)\s*,"
        r"\s*(?P<prompt_len>[0-9]+)\s*,"
        r"\s*(?P<start>[0-9]+)\s*,"
        r"\s*(?P<n_tokens>[0-9]+)\s*,"
        r"\s*(?P<checkpoint_valid>true|false)\s*,"
        r"\s*(?P<route>[A-Za-z0-9_]+)\s*,"
        r"\s*(?P<prefill_cap>[0-9]+)\s*,"
        r"\s*(?P<raw_cap>[0-9]+)\s*,"
        r"\s*(?P<chunk_cap>[0-9]+)\s*,"
        r"\s*(?P<first_chunk_tokens>[0-9]+)\s*,"
        r"\s*(?P<chunk_count>[0-9]+)\s*,"
        r"\s*(?P<final_row>Some\([0-9]+\)|None)\s*,"
        r"\s*(?P<output_pos>Some\([0-9]+\)|None)\s*,"
        r"\s*(?P<progress_point_count>[0-9]+)\s*,"
        r"\s*(?P<layer_batch_calls>[0-9]+)\s*,"
        r"\s*\[(?P<chunks>[^\]]*)\]\s*,"
        r"\s*\[(?P<progress>[^\]]*)\]\s*\)",
        flags=re.S,
    )
    for match in pattern.finditer(source):
        route = ROUTE_NAMES.get(match.group("route"), f"<unknown:{match.group('route')}>")
        out[match.group("name")] = {
            "input": {
                "ctx_size": int(match.group("ctx_size")),
                "prompt_len": int(match.group("prompt_len")),
                "start": int(match.group("start")),
                "n_tokens": int(match.group("n_tokens")),
                "checkpoint_valid": match.group("checkpoint_valid") == "true",
            },
            "route": route,
            "prefill_cap": int(match.group("prefill_cap")),
            "raw_cap": int(match.group("raw_cap")),
            "chunk_cap": int(match.group("chunk_cap")),
            "first_chunk_tokens": int(match.group("first_chunk_tokens")),
            "chunk_count": int(match.group("chunk_count")),
            "final_output_batch_row": parse_option_u32(match.group("final_row")),
            "output_absolute_pos": parse_option_u32(match.group("output_pos")),
            "progress_point_count": int(match.group("progress_point_count")),
            "layer_batch_calls": int(match.group("layer_batch_calls")),
            "chunks": parse_chunks(match.group("chunks")),
            "progress_points": [int(value) for value in re.findall(r"[0-9]+", match.group("progress"))],
        }
    return out


def parse_option_u32(value: str) -> int | None:
    if value == "None":
        return None
    match = re.fullmatch(r"Some\(([0-9]+)\)", value)
    if match is None:
        raise ValueError(value)
    return int(match.group(1))


def parse_chunks(value: str) -> list[dict[str, int]]:
    return [
        {"start": int(start), "tokens": int(tokens)}
        for start, tokens in re.findall(r"\(([0-9]+)\s*,\s*([0-9]+)\)", value)
    ]


def compare_case_maps(errors: list[str], cases: dict[str, dict[str, Any]]) -> None:
    if set(cases) != set(EXPECTED_CASES):
        errors.append(f"case set drift: expected {sorted(EXPECTED_CASES)}, got {sorted(cases)}")
        return
    for name, expected in EXPECTED_CASES.items():
        got = cases[name]
        for key, expected_value in expected.items():
            if got.get(key) != expected_value:
                errors.append(f"{name}.{key}: expected {expected_value!r}, got {got.get(key)!r}")


def validate_candidate(errors: list[str], candidate: dict[str, Any]) -> None:
    if candidate.get("schema") != SCHEMA:
        errors.append("candidate schema drift")
    if candidate.get("scope") != SCOPE:
        errors.append("candidate scope drift")
    cases = candidate.get("cases")
    if not isinstance(cases, list):
        errors.append("candidate cases missing")
        return
    by_name = {case.get("name"): case for case in cases if isinstance(case, dict)}
    if set(by_name) != set(EXPECTED_CASES):
        errors.append(f"candidate case set drift: expected {sorted(EXPECTED_CASES)}, got {sorted(by_name)}")
        return
    for name, expected in EXPECTED_CASES.items():
        case = by_name[name]
        if case.get("input") != expected["input"]:
            errors.append(f"candidate {name} input drift")
        computed = normalize_candidate_plan(case.get("computed"))
        expected_json = normalize_candidate_plan(case.get("expected"))
        if computed != expected_json:
            errors.append(f"candidate {name} computed plan does not match embedded expected plan")
        for key, expected_value in expected.items():
            if key == "input":
                continue
            if computed.get(key) != expected_value:
                errors.append(f"candidate {name}.{key}: expected {expected_value!r}, got {computed.get(key)!r}")


def normalize_candidate_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "route": value.get("route"),
        "prefill_cap": value.get("prefill_cap"),
        "raw_cap": value.get("raw_cap"),
        "chunk_cap": value.get("chunk_cap"),
        "first_chunk_tokens": value.get("first_chunk_tokens"),
        "chunk_count": value.get("chunk_count"),
        "final_output_batch_row": value.get("final_output_batch_row"),
        "output_absolute_pos": value.get("output_absolute_pos"),
        "progress_point_count": value.get("progress_point_count"),
        "layer_batch_calls": value.get("layer_batch_calls"),
        "chunks": value.get("chunks"),
        "progress_points": value.get("progress_points"),
    }


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
        ("remove schema", "prefill_plan", SCHEMA, "removed.schema"),
        ("change resume threshold", "prefill_plan", "RESUME_PREFILL_MIN_TOKENS: u32 = 4", "RESUME_PREFILL_MIN_TOKENS: u32 = 5"),
        ("remove boundary alignment", "prefill_plan", "prefill_cap - boundary_offset", "prefill_cap"),
        ("change resumed first chunk", "prefill_plan", "(1537, 511)", "(1537, 512)"),
        ("change decode suffix route", "prefill_plan", "DecodeSuffix,\n        2048,", "ChunkedRange,\n        2048,"),
        ("remove report entry", "report", "M10.6a Rust prefill scheduling plan comparator", "removed comparator"),
        ("remove roadmap split", "roadmap", "M10.6a: Rust Prefill Scheduling Plan", "removed split"),
    ]
    failures: list[str] = []
    for label, key, needle, replacement in static_mutations:
        mutated = copy.deepcopy(texts)
        if needle not in mutated[key]:
            failures.append(f"{label}: mutation needle not found")
            continue
        mutated[key] = mutated[key].replace(needle, replacement, 1)
        errors: list[str] = []
        validate_static(errors, mutated)
        if not errors:
            failures.append(f"{label}: validation unexpectedly passed")

    candidate_mutations = [
        ("candidate route", lambda obj: obj["cases"][0]["computed"].update({"route": "chunked_range"})),
        ("candidate chunk", lambda obj: obj["cases"][3]["computed"]["chunks"][0].update({"tokens": 512})),
        ("candidate progress", lambda obj: obj["cases"][2]["computed"].update({"progress_points": [0, 2052]})),
    ]
    for label, mutate in candidate_mutations:
        candidate = valid_candidate()
        mutate(candidate)
        errors = []
        validate_static(errors, texts)
        validate_candidate(errors, candidate)
        if not errors:
            failures.append(f"{label}: validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(static_mutations) + len(candidate_mutations)} mutations rejected")
    return 0


def valid_candidate() -> dict[str, Any]:
    cases = []
    for name, expected in EXPECTED_CASES.items():
        plan = {key: value for key, value in expected.items() if key != "input"}
        cases.append(
            {
                "name": name,
                "input": copy.deepcopy(expected["input"]),
                "computed": copy.deepcopy(plan),
                "expected": copy.deepcopy(plan),
            }
        )
    return {"schema": SCHEMA, "scope": SCOPE, "cases": cases}


def print_errors(errors: list[str]) -> None:
    print("Rust prefill plan comparator failures:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
