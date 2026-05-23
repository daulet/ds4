#!/usr/bin/env python3
"""Compare the Rust decode graph-state plan against the M10.2 owner oracle."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
CASE_NAME = "ctx32768_mtp_off"
DECODE_OWNERS = {
    "GraphDecodeState",
    "GraphPersistentKvState",
    "GraphLayerWorkState",
    "GraphOptionalControlState",
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


def run_rust_plan() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-graph-state-plan",
            "--quiet",
            "--",
            "--case",
            CASE_NAME,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def load_oracle() -> dict[str, Any]:
    return json.loads(ORACLE.read_text())


def first_case(obj: dict[str, Any]) -> dict[str, Any]:
    cases = obj.get("cases")
    if not isinstance(cases, list) or not cases:
        return {}
    return cases[0]


def unique_fields_for_owner(allocations: list[dict[str, Any]], owner: str) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for entry in allocations:
        if entry.get("owner") != owner:
            continue
        field_name = entry.get("field")
        if isinstance(field_name, str) and field_name not in seen:
            fields.append(field_name)
            seen.add(field_name)
    return fields


def allocation(
    allocations: list[dict[str, Any]],
    field_name: str,
    layer: int | None = None,
) -> dict[str, Any]:
    for entry in allocations:
        if entry.get("field") == field_name and entry.get("layer") == layer:
            return entry
    return {}


def validate_plan(report: Report, obj: dict[str, Any], oracle: dict[str, Any]) -> None:
    report.check(obj.get("schema") == "ds4.graph_state.v1", "schema drift")
    report.check(obj.get("scope") == "decode", "scope drift")
    case = first_case(obj)
    report.check(case.get("name") == CASE_NAME, "case name drift")

    allocations = case.get("allocations")
    summary = case.get("summary")
    report.check(isinstance(allocations, list), "allocations must be a list")
    report.check(isinstance(summary, dict), "summary must be an object")
    if not isinstance(allocations, list) or not isinstance(summary, dict):
        return

    expected_excluded = [
        group["owner"]
        for group in oracle["tensor_owner_groups"]
        if group["owner"] not in DECODE_OWNERS
    ]
    report.check(obj.get("excluded_owners") == expected_excluded, "excluded owner list drift")

    for group in oracle["tensor_owner_groups"]:
        owner = group["owner"]
        if owner in DECODE_OWNERS:
            got = unique_fields_for_owner(allocations, owner)
            report.check(got == group["fields"], f"{owner} fields differ from M10.2 oracle")

    expected_summary = {
        "logical_instances": 349,
        "initial_owned_allocations": 272,
        "initial_owned_bytes": 806175248,
        "views": 3,
        "lazy_owned": 1,
        "external_inputs": 1,
        "zero_full_capacity_fills": 105,
        "zero_state_fills": 62,
        "negative_infinity_fills": 62,
    }
    for key, expected in expected_summary.items():
        report.check(summary.get(key) == expected, f"summary {key} drift")

    report.check(
        allocation(allocations, "hc_pre").get("storage") == "view"
        and allocation(allocations, "hc_pre").get("view_base") == "hc_split"
        and allocation(allocations, "hc_pre").get("view_offset_bytes") == 0,
        "hc_pre view drift",
    )
    report.check(allocation(allocations, "hc_pre").get("bytes") == 16, "hc_pre view extent drift")
    report.check(
        allocation(allocations, "hc_post").get("storage") == "view"
        and allocation(allocations, "hc_post").get("view_base") == "hc_split"
        and allocation(allocations, "hc_post").get("view_offset_bytes") == 16,
        "hc_post view drift",
    )
    report.check(allocation(allocations, "hc_post").get("bytes") == 16, "hc_post view extent drift")
    report.check(
        allocation(allocations, "hc_comb").get("storage") == "view"
        and allocation(allocations, "hc_comb").get("view_base") == "hc_split"
        and allocation(allocations, "hc_comb").get("view_offset_bytes") == 32,
        "hc_comb view drift",
    )
    report.check(allocation(allocations, "hc_comb").get("bytes") == 64, "hc_comb view extent drift")
    report.check(
        allocation(allocations, "ffn_out").get("storage") == "lazy_owned"
        and allocation(allocations, "ffn_out").get("initially_allocated") is False,
        "ffn_out lazy allocation drift",
    )
    report.check(
        allocation(allocations, "directional_steering_dirs").get("storage") == "external",
        "directional steering external-input drift",
    )

    report.check(
        allocation(allocations, "layer_raw_cache", 0).get("bytes") == 4_718_592,
        "raw cache byte size drift",
    )
    report.check(
        allocation(allocations, "layer_attn_comp_cache", 2).get("bytes") == 16_781_312,
        "ratio-4 attention cache byte size drift",
    )
    report.check(
        allocation(allocations, "layer_attn_comp_cache", 3).get("bytes") == 528_384,
        "ratio-128 attention cache byte size drift",
    )
    report.check(
        allocation(allocations, "layer_index_comp_cache", 2).get("bytes") == 4_195_328,
        "ratio-4 index cache byte size drift",
    )
    report.check(
        allocation(allocations, "layer_index_comp_cache", 3).get("bytes") == 0
        and allocation(allocations, "layer_index_comp_cache", 3).get("initially_allocated") is False,
        "ratio-128 index cache absence drift",
    )


def run_negative_tests(report: Report, obj: dict[str, Any], oracle: dict[str, Any]) -> None:
    mutations = [
        ("summary", mutate_summary),
        ("field", mutate_field),
        ("view", mutate_view),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate_plan(mutated_report, mutate(obj), oracle)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_summary(obj: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    mutated["cases"][0]["summary"]["zero_full_capacity_fills"] -= 1
    return mutated


def mutate_field(obj: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    mutated["cases"][0]["allocations"] = [
        entry
        for entry in mutated["cases"][0]["allocations"]
        if not (entry.get("field") == "layer_index_comp_cache" and entry.get("layer") == 2)
    ]
    return mutated


def mutate_view(obj: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(obj)
    for entry in mutated["cases"][0]["allocations"]:
        if entry.get("field") == "hc_post":
            entry["view_offset_bytes"] = 0
            break
    return mutated


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Rust graph state comparator: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    oracle = load_oracle()
    obj = run_rust_plan()
    validate_plan(report, obj, oracle)
    if args.negative_test:
        run_negative_tests(report, obj, oracle)

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
