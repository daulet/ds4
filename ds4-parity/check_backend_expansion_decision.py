#!/usr/bin/env python3
"""Validate the M13.0 backend expansion decision."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json"
M12_1_INVENTORY = ROOT / "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json"
M12_6_CLOSURE = ROOT / "ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json"
M10_2_GRAPH = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
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
EXPECTED_STAGE_NAMES = {
    "M13.1": "Embedding/Indexer Expansion Fixture Matrix",
    "M13.2": "Batched Embedding Replacement Slice",
    "M13.3": "Indexed Decode Selection Replacement Slice",
    "M13.4": "Batch Indexer Fixture Gap Closure",
    "M13.5": "Embedding/Indexer Route Gate And Closure",
}
EXPECTED_COMPARATORS = {
    "embed_tokens_hc": "ds4-parity/compare_prefill_whole_short.py",
    "indexer_decode": "ds4-parity/compare_decode_long_indexed_attention.py",
    "chunked_prefill": "ds4-parity/compare_prefill_chunked.py",
    "resumed_prefill": "ds4-parity/compare_prefill_resumed.py",
}
INVENTORY_ONLY = {
    "ds4_gpu_indexer_scores_prefill_tensor",
    "ds4_gpu_dsv4_topk_mask_tensor",
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
    decision = load_json(DECISION)
    inventory = load_json(M12_1_INVENTORY)
    closure = load_json(M12_6_CLOSURE)
    graph = load_json(M10_2_GRAPH)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, decision, inventory, closure, graph, texts)
    if args.negative_test:
        run_negative_tests(report, decision, inventory, closure, graph, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    decision: dict[str, Any],
    inventory: dict[str, Any],
    closure: dict[str, Any],
    graph: dict[str, Any],
    texts: dict[str, str],
) -> None:
    validate_artifact(report, decision)
    validate_source_artifacts(report, decision)
    validate_against_m12_6(report, decision, closure)
    validate_against_inventories(report, decision, inventory, graph)
    validate_operation_coverage(report, decision)
    validate_stage_plan(report, decision)
    validate_static_wiring(report, texts)


def validate_artifact(report: Report, decision: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_expansion_decision.v1",
        "source": "m13.0-backend-expansion-decision",
        "milestone": "M13.0",
        "parent": "M13",
        "previous_stage": "M12.6",
        "next_stage": "M13.1",
        "status": "split-planned",
        "decision": "broaden_embedding_and_indexer_route",
    }
    for key, value in expected.items():
        report.check(decision.get(key) == value, f"decision {key} drift")

    policy = require_dict(report, decision.get("selection_policy"), "selection_policy")
    expected_policy = {
        "chosen_operation_family": "embedding_and_indexer",
        "broaden_existing_route": True,
        "start_new_family": False,
        "default_route_unchanged": True,
        "replacement_route_opt_in_only": True,
        "general_backend_replacement": False,
        "kernel_replacement": False,
        "removals_allowed": False,
        "current_backend_retained_as_oracle": True,
        "current_backend_retained_as_sidecar": True,
    }
    for key, value in expected_policy.items():
        report.check(policy.get(key) == value, f"selection policy {key} drift")
    reason = policy.get("reason")
    report.check(isinstance(reason, str) and "M12.6" in reason, "selection reason must cite M12.6")

    overall = require_dict(report, decision.get("overall_decision"), "overall_decision")
    report.check(
        overall.get("active_next_item") == "M13.1 Embedding/Indexer Expansion Fixture Matrix",
        "overall active next item drift",
    )
    report.check(overall.get("removals_allowed") is False, "overall decision allowed removals")


def validate_source_artifacts(report: Report, decision: dict[str, Any]) -> None:
    artifacts = require_dict(report, decision.get("source_artifacts"), "source_artifacts")
    expected = {
        "boundary_inventory": "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json",
        "operation_fixture_manifest": "ds4-parity/baselines/backend/m12.2/manifest.json",
        "facade_replay": "ds4-parity/baselines/backend/m12.3/facade-replay.json",
        "replacement_slice": "ds4-parity/baselines/backend/m12.4/replacement-slice.json",
        "runtime_route_gate": "ds4-parity/baselines/backend/m12.5/runtime-route-gate.json",
        "backend_replacement_closure": "ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json",
        "graph_plan_inventory": "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json",
    }
    report.check(artifacts == expected, "source artifact map drift")
    for value in expected.values():
        report.check((ROOT / value).exists(), f"source artifact missing: {value}")

    comparators = require_dict(report, decision.get("source_comparators"), "source_comparators")
    report.check(comparators == EXPECTED_COMPARATORS, "source comparator map drift")
    for value in EXPECTED_COMPARATORS.values():
        report.check((ROOT / value).exists(), f"source comparator missing: {value}")


def validate_against_m12_6(report: Report, decision: dict[str, Any], closure: dict[str, Any]) -> None:
    report.check(closure.get("milestone") == "M12.6", "closure milestone drift")
    claim_policy = require_dict(report, closure.get("claim_policy"), "closure claim_policy")
    report.check(claim_policy.get("removals_allowed") is False, "M12.6 allowed removals")
    report.check(
        claim_policy.get("default_route_replacement_active") is False,
        "M12.6 default route active drift",
    )
    embedding = family_by_name(closure.get("operation_family_decisions"), "embedding_and_indexer")
    report.check(embedding is not None, "M12.6 embedding/indexer decision missing")
    if embedding is None:
        return
    report.check(
        embedding.get("replacement_status") == "single_operation_route_gated",
        "M12.6 embedding/indexer replacement status drift",
    )
    report.check(
        embedding.get("replaced_operations") == ["ds4_gpu_embed_token_hc_tensor"],
        "M12.6 replaced operation drift",
    )
    report.check(embedding.get("remaining_operations") == EXPECTED_REMAINING, "M12.6 remaining operation drift")
    report.check(
        decision.get("remaining_operations_from_m12_6") == embedding.get("remaining_operations"),
        "decision remaining operations do not match M12.6",
    )
    required_work = require_list(report, closure.get("overall_decision", {}).get("next_required_work"), "next_required_work")
    report.check(
        any("embedding/indexer" in str(item) for item in required_work),
        "M12.6 did not point at embedding/indexer broadening",
    )


def validate_against_inventories(
    report: Report,
    decision: dict[str, Any],
    inventory: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    inventory_family = family_by_name(inventory.get("operation_families"), "embedding_and_indexer")
    graph_family = family_by_name(graph.get("operation_groups"), "embedding_and_indexer")
    for label, family in [("M12.1", inventory_family), ("M10.2", graph_family)]:
        report.check(family is not None, f"{label} embedding/indexer family missing")
        if family is None:
            continue
        operations = family.get("operations")
        report.check(isinstance(operations, list), f"{label} operations missing")
        if isinstance(operations, list):
            report.check("ds4_gpu_embed_token_hc_tensor" in operations, f"{label} first slice op missing")
            for operation in decision.get("remaining_operations_from_m12_6", []):
                report.check(operation in operations, f"{label} missing remaining operation {operation}")


def validate_operation_coverage(report: Report, decision: dict[str, Any]) -> None:
    coverage = require_list(report, decision.get("operation_coverage"), "operation_coverage")
    operations = [item.get("operation") for item in coverage if isinstance(item, dict)]
    report.check(operations == EXPECTED_REMAINING, "operation coverage order drift")
    stage_names = stage_name_by_id(decision)
    for item in coverage:
        if not isinstance(item, dict):
            report.check(False, "operation coverage item is not an object")
            continue
        operation = item.get("operation")
        report.check(operation in EXPECTED_REMAINING, f"unexpected operation {operation}")
        next_stage = item.get("next_stage")
        report.check(isinstance(next_stage, str) and next_stage[:5] in stage_names, f"{operation}: next stage missing")
        if isinstance(next_stage, str) and next_stage[:5] in stage_names:
            report.check(
                stage_names[next_stage[:5]] in next_stage,
                f"{operation}: next stage name drift",
            )
        comparators = require_list(report, item.get("comparators"), f"{operation}.comparators")
        if operation in INVENTORY_ONLY:
            report.check(item.get("coverage") == "inventory_only", f"{operation}: inventory-only status drift")
            report.check(comparators == [], f"{operation}: inventory-only op must not claim comparator coverage")
            report.check(next_stage == "M13.4 Batch Indexer Fixture Gap Closure", f"{operation}: gap stage drift")
        else:
            report.check(item.get("coverage") != "inventory_only", f"{operation}: covered op lost comparator status")
            report.check(comparators, f"{operation}: comparator list missing")
            for comparator in comparators:
                report.check(comparator in EXPECTED_COMPARATORS.values(), f"{operation}: unexpected comparator")
                report.check((ROOT / comparator).exists(), f"{operation}: comparator path missing")
        report.check(isinstance(item.get("oracle"), str) and "current-C" in item["oracle"], f"{operation}: current-C oracle missing")


def validate_stage_plan(report: Report, decision: dict[str, Any]) -> None:
    stage_plan = require_list(report, decision.get("stage_plan"), "stage_plan")
    stages = [item.get("stage") for item in stage_plan if isinstance(item, dict)]
    report.check(stages == list(EXPECTED_STAGE_NAMES), "stage order drift")
    for item in stage_plan:
        if not isinstance(item, dict):
            report.check(False, "stage plan item is not an object")
            continue
        stage = item.get("stage")
        report.check(item.get("name") == EXPECTED_STAGE_NAMES.get(stage), f"{stage}: stage name drift")
        for key in ["goal", "oracle", "fixture", "comparator", "acceptance", "drift_policy"]:
            value = item.get(key)
            report.check(isinstance(value, str) and value, f"{stage}: {key} missing")
        report.check("removal" not in str(item.get("goal", "")).lower(), f"{stage}: goal should not be removal")
    m13_5 = next((item for item in stage_plan if isinstance(item, dict) and item.get("stage") == "M13.5"), {})
    report.check("default route remains current-backend" in m13_5.get("acceptance", ""), "M13.5 default route guard missing")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M13.0 Backend expansion decision" in texts["report"], "unified report wiring missing")
    report.check("check_backend_expansion_decision.py" in texts["report"], "report checker path missing")
    report.check("Validate the M13.0 Backend expansion decision" in texts["readme"], "README wiring missing")
    report.check("Milestone 13: Backend Replacement Expansion" in texts["roadmap"], "roadmap M13 missing")
    report.check("M13.0: Backend Expansion Decision" in texts["roadmap"], "roadmap M13.0 missing")
    report.check("M13.1: Embedding/Indexer Expansion Fixture Matrix" in texts["roadmap"], "roadmap M13.1 missing")
    report.check("Earlier M13.0 Backend Expansion Decision" in texts["status"], "status M13.0 previous item missing")
    report.check(
        "Active item: M13.1 Embedding/Indexer Expansion Fixture Matrix" in texts["status"],
        "status M13.1 active item missing",
    )
    report.check("#### M13.0: Backend Expansion Decision" in texts["todo"], "TODO M13.0 missing")
    report.check("#### M13.1: Embedding/Indexer Expansion Fixture Matrix" in texts["todo"], "TODO M13.1 missing")


def run_negative_tests(
    report: Report,
    decision: dict[str, Any],
    inventory: dict[str, Any],
    closure: dict[str, Any],
    graph: dict[str, Any],
    texts: dict[str, str],
) -> None:
    mutations = [
        ("new family selected", mutate_new_family),
        ("removals allowed", lambda obj: mutate_nested(obj, ["selection_policy", "removals_allowed"], True)),
        (
            "default route replacement active",
            lambda obj: mutate_nested(obj, ["selection_policy", "default_route_unchanged"], False),
        ),
        ("missing remaining operation", remove_remaining_operation),
        ("covered op loses comparator", mutate_covered_to_inventory_only),
        ("inventory op overclaims comparator", mutate_inventory_overclaim),
        ("missing M13.1 stage", remove_m13_1_stage),
        ("wrong active next item", lambda obj: mutate_nested(obj, ["overall_decision", "active_next_item"], "M13.2")),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(mutated_report, mutate(decision), inventory, closure, graph, texts)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_new_family(decision: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(decision)
    mutated["decision"] = "start_dense_norm_rope_kv_route"
    mutated["selection_policy"]["chosen_operation_family"] = "dense_norm_rope_kv"
    mutated["selection_policy"]["broaden_existing_route"] = False
    mutated["selection_policy"]["start_new_family"] = True
    return mutated


def remove_remaining_operation(decision: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(decision)
    mutated["remaining_operations_from_m12_6"] = mutated["remaining_operations_from_m12_6"][:-1]
    mutated["operation_coverage"] = mutated["operation_coverage"][:-1]
    return mutated


def mutate_covered_to_inventory_only(decision: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(decision)
    mutated["operation_coverage"][0]["coverage"] = "inventory_only"
    mutated["operation_coverage"][0]["comparators"] = []
    return mutated


def mutate_inventory_overclaim(decision: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(decision)
    mutated["operation_coverage"][-1]["coverage"] = "fake_pair_comparator"
    mutated["operation_coverage"][-1]["comparators"] = ["ds4-parity/compare_prefill_whole_short.py"]
    mutated["operation_coverage"][-1]["next_stage"] = "M13.2 Batched Embedding Replacement Slice"
    return mutated


def remove_m13_1_stage(decision: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(decision)
    mutated["stage_plan"] = mutated["stage_plan"][1:]
    return mutated


def mutate_nested(decision: dict[str, Any], path: list[str], value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(decision)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return mutated


def stage_name_by_id(decision: dict[str, Any]) -> dict[str, str]:
    stage_plan = decision.get("stage_plan")
    if not isinstance(stage_plan, list):
        return {}
    return {
        str(item.get("stage")): str(item.get("name"))
        for item in stage_plan
        if isinstance(item, dict)
    }


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
    print(f"Backend expansion decision: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
