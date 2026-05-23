#!/usr/bin/env python3
"""Validate the Rust full decode state allocation contract for M10.5c4c2b2a."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "bin": ROOT / "rust/ds4-gpu/src/bin/ds4-decode-state-alloc.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

SCHEMA = "ds4.decode_state_allocation.v1"
CASE = "ctx32768_mtp_off"
SUMMARY = {
    "logical_instances": 349,
    "initial_owned_allocations": 272,
    "initial_owned_bytes": 806175248,
    "views_created": 3,
    "lazy_owned_deferred": 1,
    "external_inputs": 1,
    "zero_full_capacity_fills": 105,
    "zero_state_fills": 62,
    "negative_infinity_fills": 62,
}
REQUIRED_ALLOCATIONS = {
    (None, "comp_mask"): (67125248, "unspecified"),
    (None, "indexer_scores"): (67125248, "unspecified"),
    (2, "layer_attn_comp_cache"): (16781312, "zero_full_capacity"),
}
REQUIRED_VIEWS = {
    "hc_pre": ("hc_split", None, 0, 16),
    "hc_post": ("hc_split", None, 16, 16),
    "hc_comb": ("hc_split", None, 32, 64),
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
        print(f"Rust decode state allocation contract: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="B300 JSON emitted by ds4-decode-state-alloc")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    bin_text = texts["bin"]
    for needle, message in (
        (SCHEMA, "allocation schema missing"),
        ('CASE: &str = "ctx32768_mtp_off"', "allocation case drift"),
        ("GraphPlan::for_context(32768, 32768, false)", "default decode graph plan missing"),
        ("ModelGraphDims::DS4_FLASH", "DS4 Flash graph dimensions missing"),
        ("DECODE_GRAPH_STATE_FIELDS", "graph-state field table missing"),
        ("GraphTensorInstances::PerLayer", "per-layer field expansion missing"),
        ("ds4_gpu::graph_plan::N_LAYER", "layer-count expansion missing"),
        ("initially_allocated", "initial allocation gate missing"),
        ("Tensor::allocate", "owned tensor allocation missing"),
        ("fill_f32(0.0", "zero-fill initialization missing"),
        ("fill_f32(f32::NEG_INFINITY", "negative-infinity initialization missing"),
        ("GraphTensorStorage::View", "view storage handling missing"),
        (".view(", "backend tensor view creation missing"),
        ("summary.initial_owned_bytes += bytes", "owned byte accounting missing"),
        ("BackendGuard", "backend cleanup guard missing"),
        ("ds4_gpu::cleanup", "backend cleanup call missing"),
    ):
        report.check(needle in bin_text, message)
    report_text = texts["report"]
    report.check("M10.5c4c2b2a Rust full decode state allocation comparator" in report_text, "unified report comparator missing")
    report.check("M10.5c4c2b2a B300 Rust decode state allocation rerun" in report_text, "B300 allocation skip missing")
    report.check("--bin ds4-decode-state-alloc" in report_text, "B300 allocation command missing binary")
    report.check("--candidate /tmp/ds4-c2b2a-state-allocation.json" in report_text, "B300 allocation candidate validation missing")

    report.check("M10.5c4c2b2a Rust full decode state allocation" in texts["readme"], "README entry missing")
    report.check("M10.5c4c2b2a: Rust Full Decode State Allocation" in texts["roadmap"], "roadmap allocation split missing")
    report.check("M10.5c4c2b2b: Rust One-Token Decode B300 Execution" in texts["roadmap"], "roadmap decode remainder missing")
    report.check("M10.5c4c2b2a: Rust Full Decode State Allocation" in texts["todo"], "TODO allocation split missing")
    report.check("M10.5c4c2b2b: Rust One-Token Decode B300 Execution" in texts["todo"], "TODO decode remainder missing")


def validate_candidate(report: Report, obj: dict[str, Any]) -> None:
    report.check(obj.get("schema") == SCHEMA, "candidate schema drift")
    report.check(obj.get("case") == CASE, "candidate case drift")
    report.check(obj.get("ctx_size") == 32768, "candidate ctx_size drift")
    report.check(obj.get("prompt_len") == 32768, "candidate prompt_len drift")

    summary = obj.get("summary")
    report.check(isinstance(summary, dict), "candidate summary missing")
    if isinstance(summary, dict):
        for key, expected in SUMMARY.items():
            report.check(summary.get(key) == expected, f"candidate summary {key} drift")

    allocations = obj.get("largest_allocations")
    report.check(isinstance(allocations, list) and len(allocations) >= 3, "candidate largest allocations missing")
    if isinstance(allocations, list):
        by_key = {
            (entry.get("layer"), entry.get("field")): entry
            for entry in allocations
            if isinstance(entry, dict)
        }
        for key, (bytes_, fill) in REQUIRED_ALLOCATIONS.items():
            entry = by_key.get(key)
            report.check(isinstance(entry, dict), f"candidate allocation {key} missing")
            if isinstance(entry, dict):
                report.check(entry.get("bytes") == bytes_, f"candidate allocation {key} byte drift")
                report.check(entry.get("fill") == fill, f"candidate allocation {key} fill drift")

    views = obj.get("views")
    report.check(isinstance(views, list), "candidate views missing")
    if isinstance(views, list):
        by_field = {
            entry.get("field"): entry
            for entry in views
            if isinstance(entry, dict)
        }
        report.check(set(REQUIRED_VIEWS) <= set(by_field), "candidate view set incomplete")
        for field, (base, layer, offset, bytes_) in REQUIRED_VIEWS.items():
            entry = by_field.get(field)
            if isinstance(entry, dict):
                report.check(entry.get("base") == base, f"candidate view {field} base drift")
                report.check(entry.get("layer") == layer, f"candidate view {field} layer drift")
                report.check(entry.get("offset_bytes") == offset, f"candidate view {field} offset drift")
                report.check(entry.get("bytes") == bytes_, f"candidate view {field} byte drift")


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
        ("remove schema", "bin", SCHEMA, "ds4.decode_state_allocation.removed"),
        ("remove allocation", "bin", "Tensor::allocate", "Tensor_removed::allocate"),
        ("remove view", "bin", "GraphTensorStorage::View", "GraphTensorStorage::RemovedView"),
        ("remove neg-inf fill", "bin", "fill_f32(f32::NEG_INFINITY", "fill_f32(0.0"),
        ("remove b300 candidate check", "report", "--candidate /tmp/ds4-c2b2a-state-allocation.json", ""),
        ("remove roadmap split", "roadmap", "M10.5c4c2b2a: Rust Full Decode State Allocation", "M10.5c4c2b2a removed"),
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
        ("summary byte count", mutate_summary_bytes),
        ("required allocation", mutate_required_allocation),
        ("view offset", mutate_view_offset),
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
        "ctx_size": 32768,
        "prompt_len": 32768,
        "summary": copy.deepcopy(SUMMARY),
        "largest_allocations": [
            {"field": "comp_mask", "layer": None, "bytes": 67125248, "fill": "unspecified"},
            {"field": "indexer_scores", "layer": None, "bytes": 67125248, "fill": "unspecified"},
            {"field": "layer_attn_comp_cache", "layer": 2, "bytes": 16781312, "fill": "zero_full_capacity"},
        ],
        "views": [
            {"field": "hc_pre", "base": "hc_split", "layer": None, "offset_bytes": 0, "bytes": 16},
            {"field": "hc_post", "base": "hc_split", "layer": None, "offset_bytes": 16, "bytes": 16},
            {"field": "hc_comb", "base": "hc_split", "layer": None, "offset_bytes": 32, "bytes": 64},
        ],
    }


def mutate_summary_bytes(obj: dict[str, Any]) -> dict[str, Any]:
    obj["summary"]["initial_owned_bytes"] -= 4
    return obj


def mutate_required_allocation(obj: dict[str, Any]) -> dict[str, Any]:
    obj["largest_allocations"] = [
        entry
        for entry in obj["largest_allocations"]
        if entry.get("field") != "layer_attn_comp_cache"
    ]
    return obj


def mutate_view_offset(obj: dict[str, Any]) -> dict[str, Any]:
    obj["views"][1]["offset_bytes"] = 0
    return obj


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
