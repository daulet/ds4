#!/usr/bin/env python3
"""Validate the M13.1 embedding/indexer expansion fixture matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json"
DECISION = ROOT / "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json"
M12_6 = ROOT / "ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json"
M12_1 = ROOT / "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json"
M10_2 = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_REMAINING = [
    "ds4_gpu_embed_tokens_hc_tensor",
    "ds4_gpu_indexer_score_one_tensor",
    "ds4_gpu_indexer_scores_prefill_tensor",
    "ds4_gpu_indexer_scores_decode_batch_tensor",
    "ds4_gpu_indexer_topk_tensor",
    "ds4_gpu_dsv4_topk_mask_tensor",
]
PAIR_READY = [
    "ds4_gpu_embed_tokens_hc_tensor",
    "ds4_gpu_indexer_score_one_tensor",
    "ds4_gpu_indexer_topk_tensor",
]
FIXTURE_GAPS = [
    "ds4_gpu_indexer_scores_prefill_tensor",
    "ds4_gpu_indexer_scores_decode_batch_tensor",
    "ds4_gpu_dsv4_topk_mask_tensor",
]
EXPECTED_LEVELS = {
    "pair-comparator-ready",
    "surface-covered-needs-fixture-extraction",
    "inventory-only",
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
    matrix = load_json(MATRIX)
    decision = load_json(DECISION)
    closure = load_json(M12_6)
    inventory = load_json(M12_1)
    graph = load_json(M10_2)
    sources = {
        "ds4.c": read_text(ROOT / "ds4.c"),
        "rust/ds4-gpu/src/decode_backend.rs": read_text(ROOT / "rust/ds4-gpu/src/decode_backend.rs"),
        "rust/ds4-gpu/src/graph_plan.rs": read_text(ROOT / "rust/ds4-gpu/src/graph_plan.rs"),
    }
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, matrix, decision, closure, inventory, graph, sources, texts)
    if args.negative_test:
        run_negative_tests(report, matrix, decision, closure, inventory, graph, sources, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    matrix: dict[str, Any],
    decision: dict[str, Any],
    closure: dict[str, Any],
    inventory: dict[str, Any],
    graph: dict[str, Any],
    sources: dict[str, str],
    texts: dict[str, str],
) -> None:
    validate_artifact(report, matrix)
    validate_against_decision(report, matrix, decision)
    validate_against_closure(report, matrix, closure)
    validate_against_inventories(report, matrix, inventory, graph)
    validate_rows(report, matrix, sources)
    validate_summary(report, matrix)
    validate_static_wiring(report, texts)


def validate_artifact(report: Report, matrix: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_expansion_matrix.v1",
        "source": "m13.1-embedding-indexer-expansion-matrix",
        "milestone": "M13.1",
        "parent": "M13",
        "previous_stage": "M13.0",
        "next_stage": "M13.2",
        "status": "fixture-matrix",
        "operation_family": "embedding_and_indexer",
    }
    for key, value in expected.items():
        report.check(matrix.get(key) == value, f"matrix {key} drift")

    artifacts = require_dict(report, matrix.get("source_artifacts"), "source_artifacts")
    expected_artifacts = {
        "expansion_decision": "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json",
        "backend_replacement_closure": "ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json",
        "boundary_inventory": "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json",
        "graph_plan_inventory": "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json",
    }
    report.check(artifacts == expected_artifacts, "source artifact map drift")
    for value in expected_artifacts.values():
        report.check((ROOT / value).exists(), f"source artifact missing: {value}")

    policy = require_dict(report, matrix.get("claim_policy"), "claim_policy")
    expected_policy = {
        "runtime_route_change": False,
        "default_route_unchanged": True,
        "replacement_route_opt_in_only": True,
        "default_route_replacement_active": False,
        "general_backend_replacement": False,
        "kernel_replacement": False,
        "removals_allowed": False,
        "current_backend_retained_as_oracle": True,
        "current_backend_retained_as_sidecar": True,
    }
    for key, value in expected_policy.items():
        report.check(policy.get(key) == value, f"claim policy {key} drift")

    levels = require_dict(report, matrix.get("coverage_levels"), "coverage_levels")
    report.check(set(levels) == EXPECTED_LEVELS, "coverage level set drift")


def validate_against_decision(report: Report, matrix: dict[str, Any], decision: dict[str, Any]) -> None:
    report.check(decision.get("milestone") == "M13.0", "decision milestone drift")
    report.check(decision.get("decision") == "broaden_embedding_and_indexer_route", "decision route drift")
    report.check(decision.get("remaining_operations_from_m12_6") == EXPECTED_REMAINING, "decision operation list drift")
    rows = rows_by_operation(matrix)
    decision_rows = decision.get("operation_coverage")
    report.check(isinstance(decision_rows, list), "decision operation coverage missing")
    if isinstance(decision_rows, list):
        for item in decision_rows:
            if not isinstance(item, dict):
                report.check(False, "decision operation coverage item invalid")
                continue
            operation = item.get("operation")
            row = rows.get(operation)
            report.check(row is not None, f"matrix missing decision operation {operation}")
            if row is None:
                continue
            report.check(row.get("route_candidate_stage") == item.get("next_stage"), f"{operation}: next stage drift from M13.0")
            if item.get("coverage") == "inventory_only":
                report.check(row.get("coverage_level") == "inventory-only", f"{operation}: inventory coverage drift")
            if item.get("comparators"):
                report.check(row.get("comparators") == item.get("comparators"), f"{operation}: comparator drift from M13.0")


def validate_against_closure(report: Report, matrix: dict[str, Any], closure: dict[str, Any]) -> None:
    report.check(closure.get("milestone") == "M12.6", "closure milestone drift")
    embedding = family_by_name(closure.get("operation_family_decisions"), "embedding_and_indexer")
    report.check(embedding is not None, "closure embedding family missing")
    if embedding is None:
        return
    report.check(embedding.get("replaced_operations") == ["ds4_gpu_embed_token_hc_tensor"], "closure replaced op drift")
    report.check(embedding.get("remaining_operations") == EXPECTED_REMAINING, "closure remaining op drift")
    report.check(operations(matrix) == embedding.get("remaining_operations"), "matrix operations do not match closure")
    report.check(embedding.get("removal_decision") == "retain_current_backend", "closure removal decision drift")


def validate_against_inventories(
    report: Report,
    matrix: dict[str, Any],
    inventory: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    for label, data, key in [
        ("M12.1", inventory, "operation_families"),
        ("M10.2", graph, "operation_groups"),
    ]:
        family = family_by_name(data.get(key), "embedding_and_indexer")
        report.check(family is not None, f"{label} embedding/indexer family missing")
        if family is None:
            continue
        family_operations = family.get("operations")
        report.check(isinstance(family_operations, list), f"{label} operation list missing")
        if isinstance(family_operations, list):
            for operation in operations(matrix):
                report.check(operation in family_operations, f"{label} missing operation {operation}")


def validate_rows(report: Report, matrix: dict[str, Any], sources: dict[str, str]) -> None:
    rows = require_list(report, matrix.get("matrix"), "matrix rows")
    report.check([row.get("operation") for row in rows if isinstance(row, dict)] == EXPECTED_REMAINING, "matrix row order drift")
    for row in rows:
        if not isinstance(row, dict):
            report.check(False, "matrix row is not an object")
            continue
        operation = row.get("operation")
        report.check(operation in EXPECTED_REMAINING, f"unexpected operation {operation}")
        report.check(row.get("coverage_level") in EXPECTED_LEVELS, f"{operation}: invalid coverage level")
        report.check(row.get("route_status") == "not-route-gated", f"{operation}: route status overclaim")
        report.check(row.get("removal_decision") == "retain_current_backend", f"{operation}: removal decision drift")
        source_path = row.get("current_c_source")
        source = sources.get(source_path)
        report.check(source is not None, f"{operation}: current-C source missing")
        if source is not None:
            report.check(row.get("current_c_anchor") in source, f"{operation}: current-C anchor missing")
        rust_source_path = row.get("rust_source")
        rust_source = sources.get(rust_source_path)
        report.check(rust_source is not None, f"{operation}: Rust source missing")
        if rust_source is not None:
            report.check(row.get("rust_anchor") in rust_source, f"{operation}: Rust anchor missing")
            if row.get("rust_safe_wrapper") is True:
                report.check(f'operation: "{operation}"' in rust_source, f"{operation}: facade operation missing")
                report.check(str(row.get("method")) in rust_source, f"{operation}: Rust method missing")
        comparators = require_list(report, row.get("comparators"), f"{operation}.comparators")
        for comparator in comparators:
            report.check(isinstance(comparator, str) and (ROOT / comparator).exists(), f"{operation}: comparator missing")
        if operation in PAIR_READY:
            report.check(row.get("coverage_level") == "pair-comparator-ready", f"{operation}: pair-ready level drift")
            report.check(row.get("fixture_gap") is False, f"{operation}: unexpected fixture gap")
            report.check(comparators, f"{operation}: pair-ready comparator missing")
            report.check(row.get("covered_outputs"), f"{operation}: covered outputs missing")
        if operation in FIXTURE_GAPS:
            report.check(row.get("fixture_gap") is True, f"{operation}: fixture gap not recorded")
            report.check(row.get("route_candidate_stage") == "M13.4 Batch Indexer Fixture Gap Closure", f"{operation}: gap stage drift")
        if row.get("coverage_level") == "inventory-only":
            report.check(comparators == [], f"{operation}: inventory-only comparator overclaim")
            report.check(row.get("covered_outputs") == [], f"{operation}: inventory-only output overclaim")


def validate_summary(report: Report, matrix: dict[str, Any]) -> None:
    summary = require_dict(report, matrix.get("summary"), "summary")
    report.check(summary.get("total_remaining_operations") == 6, "summary total drift")
    report.check(summary.get("pair_comparator_ready") == PAIR_READY, "summary pair-ready drift")
    report.check(summary.get("fixture_gap_operations") == FIXTURE_GAPS, "summary fixture-gap drift")
    report.check(summary.get("next_route_candidate") == "M13.2 Batched Embedding Replacement Slice", "summary next route drift")
    report.check(summary.get("next_gap_closure") == "M13.4 Batch Indexer Fixture Gap Closure", "summary next gap drift")
    report.check(summary.get("removals_allowed") is False, "summary removals allowed")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M13.1 Backend expansion matrix" in texts["report"], "unified report wiring missing")
    report.check("check_backend_expansion_matrix.py" in texts["report"], "report checker path missing")
    report.check("Validate the M13.1 Backend expansion matrix" in texts["readme"], "README wiring missing")
    report.check("M13.1: Embedding/Indexer Expansion Fixture Matrix" in texts["roadmap"], "roadmap M13.1 missing")
    report.check("M13.2: Batched Embedding Replacement Slice" in texts["roadmap"], "roadmap M13.2 missing")
    report.check("Earlier M13.1 Embedding/Indexer Expansion Fixture Matrix" in texts["status"], "status M13.1 previous item missing")
    report.check("Active item: M13" in texts["status"], "status M13 active item missing")
    report.check("#### M13.1: Embedding/Indexer Expansion Fixture Matrix" in texts["todo"], "TODO M13.1 missing")
    report.check("#### M13.2: Batched Embedding Replacement Slice" in texts["todo"], "TODO M13.2 missing")


def run_negative_tests(
    report: Report,
    matrix: dict[str, Any],
    decision: dict[str, Any],
    closure: dict[str, Any],
    inventory: dict[str, Any],
    graph: dict[str, Any],
    sources: dict[str, str],
    texts: dict[str, str],
) -> None:
    mutations = [
        ("missing operation", remove_last_row),
        ("route overclaim", lambda obj: mutate_nested(obj, ["claim_policy", "runtime_route_change"], True)),
        ("default route active", lambda obj: mutate_nested(obj, ["claim_policy", "default_route_replacement_active"], True)),
        ("removals allowed", lambda obj: mutate_nested(obj, ["claim_policy", "removals_allowed"], True)),
        ("pair-ready marked gap", mutate_pair_ready_to_gap),
        ("inventory op overclaims comparator", mutate_inventory_overclaim),
        ("wrong current-C anchor", lambda obj: mutate_nested(obj, ["matrix", 0, "current_c_anchor"], "missing_anchor")),
        ("wrong next stage", lambda obj: mutate_nested(obj, ["matrix", 1, "route_candidate_stage"], "M13.4 Batch Indexer Fixture Gap Closure")),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(mutated_report, mutate(matrix), decision, closure, inventory, graph, sources, texts)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def remove_last_row(matrix: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(matrix)
    mutated["matrix"] = mutated["matrix"][:-1]
    mutated["summary"]["fixture_gap_operations"] = mutated["summary"]["fixture_gap_operations"][:-1]
    return mutated


def mutate_pair_ready_to_gap(matrix: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(matrix)
    mutated["matrix"][0]["coverage_level"] = "inventory-only"
    mutated["matrix"][0]["fixture_gap"] = True
    mutated["matrix"][0]["comparators"] = []
    return mutated


def mutate_inventory_overclaim(matrix: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(matrix)
    mutated["matrix"][2]["coverage_level"] = "pair-comparator-ready"
    mutated["matrix"][2]["fixture_gap"] = False
    mutated["matrix"][2]["comparators"] = ["ds4-parity/compare_prefill_whole_short.py"]
    mutated["matrix"][2]["covered_outputs"] = ["fake_output"]
    return mutated


def mutate_nested(matrix: dict[str, Any], path: list[Any], value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(matrix)
    target: Any = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return mutated


def operations(matrix: dict[str, Any]) -> list[Any]:
    rows = matrix.get("matrix")
    if not isinstance(rows, list):
        return []
    return [row.get("operation") for row in rows if isinstance(row, dict)]


def rows_by_operation(matrix: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    rows = matrix.get("matrix")
    if not isinstance(rows, list):
        return {}
    return {row.get("operation"): row for row in rows if isinstance(row, dict)}


def family_by_name(families: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(families, list):
        return None
    for family in families:
        if isinstance(family, dict) and family.get("name") == name:
            return family
    return None


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} missing")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, label: str) -> list[Any]:
    report.check(isinstance(value, list), f"{label} missing")
    return value if isinstance(value, list) else []


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Backend expansion matrix: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
