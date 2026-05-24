#!/usr/bin/env python3
"""Validate the M12.1 backend boundary inventory and claim matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json"
GRAPH_ORACLE = ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

ALLOWED_OWNER_STATES = {"current-c", "ffi-wrapped", "rust-planned", "rust-owned"}
ALLOWED_PLATFORMS = {"macos-metal", "linux-cuda"}
FORBIDDEN_CLAIM_WORDS = ("remove-current", "default replacement", "rust-owned kernel")


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
    inventory = load_json(INVENTORY)
    graph_oracle = load_json(GRAPH_ORACLE)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    if args.negative_test:
        return run_negative_tests(inventory, graph_oracle, texts)

    report = Report()
    validate(report, inventory, graph_oracle, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def validate(
    report: Report,
    inventory: dict[str, Any],
    graph_oracle: dict[str, Any],
    texts: dict[str, str],
) -> None:
    report.check(inventory.get("schema") == "ds4.backend_boundary_inventory.v1", "schema drift")
    report.check(inventory.get("milestone") == "M12.1", "milestone drift")
    report.check(inventory.get("status") == "inventory-only", "M12.1 must remain inventory-only")
    report.check(inventory.get("next_stage") == "M12.2", "next stage must be M12.2")
    validate_source_artifacts(report, inventory)
    validate_routes(report, inventory)
    validate_b300_commands(report, inventory, texts)
    validate_claim_policy(report, inventory)
    validate_operation_families(report, inventory, graph_oracle)
    validate_static_wiring(report, texts)


def validate_source_artifacts(report: Report, inventory: dict[str, Any]) -> None:
    artifacts = inventory.get("source_artifacts")
    report.check(isinstance(artifacts, list) and len(artifacts) >= 8, "source artifacts missing")
    if not isinstance(artifacts, list):
        return

    ids: set[str] = set()
    required_ids = {
        "m10_2_graph_plan_inventory",
        "backend_header",
        "rust_sys",
        "rust_safe_facade",
        "rust_gpu_build",
        "m10_5c4c1_backend_abi_smoke",
        "m10_5c4c1_contract_checker",
        "m10_9a_runtime_graph_closure",
        "m10_9f_runtime_benchmark_closure",
    }
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            report.check(False, "source artifact entry is not an object")
            continue
        artifact_id = artifact.get("id")
        path_text = artifact.get("path")
        role = artifact.get("role")
        expected_sha = artifact.get("sha256")
        report.check(isinstance(artifact_id, str) and artifact_id, "source artifact id missing")
        report.check(artifact_id not in ids, f"duplicate source artifact id {artifact_id}")
        if isinstance(artifact_id, str):
            ids.add(artifact_id)
        report.check(isinstance(role, str) and role, f"{artifact_id}: role missing")
        report.check(isinstance(path_text, str) and path_text, f"{artifact_id}: path missing")
        report.check(isinstance(expected_sha, str) and len(expected_sha) == 64, f"{artifact_id}: sha256 missing")
        if not isinstance(path_text, str) or not isinstance(expected_sha, str):
            continue
        path = ROOT / path_text
        report.check(path.exists(), f"{artifact_id}: {path_text} does not exist")
        if path.exists():
            report.check(sha256_file(path) == expected_sha, f"{artifact_id}: sha256 drift for {path_text}")
    report.check(required_ids <= ids, f"source artifacts missing required ids: {sorted(required_ids - ids)}")


def validate_routes(report: Report, inventory: dict[str, Any]) -> None:
    routes = inventory.get("route_selectors")
    report.check(isinstance(routes, dict), "route_selectors missing")
    if not isinstance(routes, dict):
        return
    report.check(routes.get("cli_backends") == ["metal", "cuda", "cpu"], "CLI backend selector drift")
    report.check(
        routes.get("runtime_graph_routes") == ["target-stream", "off", "graph"],
        "runtime graph route selector drift",
    )
    report.check(routes.get("graph_backend_requires") == ["metal", "cuda"], "graph backend requirement drift")
    report.check("fail-closed" in str(routes.get("cpu_graph_route")), "CPU graph route must fail closed")


def validate_b300_commands(
    report: Report,
    inventory: dict[str, Any],
    texts: dict[str, str],
) -> None:
    commands = inventory.get("b300_rerun_commands")
    report.check(isinstance(commands, list) and len(commands) >= 3, "B300 rerun commands missing")
    if not isinstance(commands, list):
        return
    required_ids = {
        "m10_5c4c1_cuda_backend_smoke",
        "m10_9a_fixture_readiness_probe",
        "m10_9f_benchmark_closure",
    }
    got_ids: set[str] = set()
    for command in commands:
        if not isinstance(command, dict):
            report.check(False, "B300 command entry is not an object")
            continue
        command_id = command.get("id")
        source_function = command.get("source_function")
        command_text = command.get("command")
        if isinstance(command_id, str):
            got_ids.add(command_id)
        report.check(isinstance(command_id, str) and command_id, "B300 command id missing")
        report.check(isinstance(source_function, str) and source_function, f"{command_id}: source function missing")
        report.check(isinstance(command_text, str) and command_text, f"{command_id}: command missing")
        if not isinstance(command_text, str):
            continue
        for needle in [
            "kubectl",
            "--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig",
            "--context hou2-prod1",
            "-n default",
            "ds4-rust-port-b300",
            "cd /workspace/ds4",
        ]:
            report.check(needle in command_text, f"{command_id}: command missing {needle}")
        if command_id != "m10_9a_fixture_readiness_probe":
            report.check("git archive HEAD" in command_text, f"{command_id}: source refresh missing")
            report.check("CUDA_ARCH=native" in command_text, f"{command_id}: CUDA_ARCH missing")
        if isinstance(source_function, str) and source_function.startswith("b300_"):
            report.check(source_function in texts["report"], f"{command_id}: report source function missing")
    report.check(required_ids <= got_ids, f"B300 command ids missing: {sorted(required_ids - got_ids)}")


def validate_claim_policy(report: Report, inventory: dict[str, Any]) -> None:
    policy = inventory.get("claim_policy")
    report.check(isinstance(policy, dict), "claim policy missing")
    if not isinstance(policy, dict):
        return
    report.check(policy.get("backend_replacement") == "not_claimed", "backend replacement overclaim")
    report.check(policy.get("kernel_replacement") == "not_claimed", "kernel replacement overclaim")
    report.check(policy.get("removals_allowed") is False, "removal must not be allowed in M12.1")
    report.check(policy.get("first_replacement_slice") == "blocked_until_m12.4", "replacement slice gate drift")
    report.check(policy.get("closure_removal") == "blocked_until_m12.6", "closure removal gate drift")
    forbidden = policy.get("must_not_report_as")
    report.check(isinstance(forbidden, list), "must_not_report_as missing")
    if isinstance(forbidden, list):
        for phrase in ["backend replacement", "C/CUDA/Metal removal", "Rust-owned kernel coverage"]:
            report.check(phrase in forbidden, f"must_not_report_as missing {phrase}")


def validate_operation_families(
    report: Report,
    inventory: dict[str, Any],
    graph_oracle: dict[str, Any],
) -> None:
    families = inventory.get("operation_families")
    oracle_groups = graph_oracle.get("operation_groups")
    report.check(isinstance(families, list), "operation families missing")
    report.check(isinstance(oracle_groups, list), "oracle operation groups missing")
    if not isinstance(families, list) or not isinstance(oracle_groups, list):
        return
    report.check(len(families) == len(oracle_groups), "operation family count drift")

    artifact_ids = {
        artifact.get("id")
        for artifact in inventory.get("source_artifacts", [])
        if isinstance(artifact, dict)
    }
    for index, oracle_group in enumerate(oracle_groups):
        if not isinstance(oracle_group, dict):
            report.check(False, f"oracle group {index} is not an object")
            continue
        if index >= len(families):
            continue
        family = families[index]
        if not isinstance(family, dict):
            report.check(False, f"operation family {index} is not an object")
            continue
        name = family.get("name")
        report.check(name == oracle_group.get("name"), f"family {index} name drift")
        report.check(family.get("operations") == oracle_group.get("operations"), f"{name}: operation list drift")
        owner_state = family.get("owner_state")
        report.check(owner_state in ALLOWED_OWNER_STATES, f"{name}: invalid owner_state {owner_state!r}")
        report.check(owner_state != "rust-owned", f"{name}: M12.1 must not claim rust-owned kernels")
        platforms = family.get("required_platforms")
        report.check(isinstance(platforms, list) and platforms, f"{name}: required platforms missing")
        if isinstance(platforms, list):
            report.check(all(platform in ALLOWED_PLATFORMS for platform in platforms), f"{name}: unsupported platform claim")
            report.check("linux-cuda" in platforms, f"{name}: B300 CUDA platform missing")
        report.check(isinstance(family.get("model_requirement"), str) and family["model_requirement"], f"{name}: model requirement missing")
        report.check(family.get("fixture_source") in artifact_ids, f"{name}: fixture source missing from artifacts")
        comparator = family.get("comparator")
        report.check(isinstance(comparator, str) and comparator.startswith("ds4-parity/"), f"{name}: comparator path missing")
        if isinstance(comparator, str):
            report.check((ROOT / comparator).exists(), f"{name}: comparator {comparator} does not exist")
        claim = family.get("claim_boundary")
        report.check(isinstance(claim, str) and "no-removal" in claim, f"{name}: no-removal boundary missing")
        if isinstance(claim, str):
            lower_claim = claim.lower()
            for word in FORBIDDEN_CLAIM_WORDS:
                report.check(word not in lower_claim, f"{name}: forbidden claim wording {word!r}")
        drift = family.get("drift_policy")
        report.check(isinstance(drift, list) and len(drift) >= 2, f"{name}: drift policy incomplete")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M12.1 Backend boundary inventory" in texts["readme"], "README M12.1 section missing")
    report.check(
        "M12.1 Backend boundary inventory" in texts["report"],
        "unified parity report M12.1 item missing",
    )
    report.check("ds4-parity/check_backend_boundary_inventory.py" in texts["report"], "report command missing checker")
    report.check("M12.1: Backend Boundary Inventory And Claim Matrix" in texts["roadmap"], "roadmap M12.1 missing")
    report.check("Status: complete." in texts["todo"], "TODO does not record a completed milestone")
    report.check(
        "Earlier M12.1 Backend Boundary Inventory And Claim Matrix" in texts["status"],
        "status missing M12.1 completion",
    )
    report.check(
        "Active item: M12.2 Operation Tensor Fixture Capture" in texts["status"]
        or "Active item: M12.3 Rust Backend Facade Parity Harness" in texts["status"]
        or "Active item: M12.4 First Backend Replacement Slice" in texts["status"]
        or "Active item: M12.5 Runtime Backend Route Gate" in texts["status"],
        "status active item must advance to M12.2 or later",
    )


def run_negative_tests(
    inventory: dict[str, Any],
    graph_oracle: dict[str, Any],
    texts: dict[str, str],
) -> int:
    mutations: list[tuple[str, dict[str, Any]]] = []

    missing_family = copy.deepcopy(inventory)
    missing_family["operation_families"] = missing_family["operation_families"][:-1]
    mutations.append(("missing operation family", missing_family))

    operation_drift = copy.deepcopy(inventory)
    operation_drift["operation_families"][1]["operations"] = operation_drift["operation_families"][1]["operations"][:-1]
    mutations.append(("operation list drift", operation_drift))

    overclaim = copy.deepcopy(inventory)
    overclaim["operation_families"][0]["owner_state"] = "rust-owned"
    mutations.append(("rust-owned kernel overclaim", overclaim))

    removal = copy.deepcopy(inventory)
    removal["claim_policy"]["removals_allowed"] = True
    mutations.append(("removal allowed", removal))

    missing_context = copy.deepcopy(inventory)
    command = missing_context["b300_rerun_commands"][0]["command"]
    missing_context["b300_rerun_commands"][0]["command"] = command.replace(
        "--context hou2-prod1",
        "--context removed",
    )
    mutations.append(("B300 context drift", missing_context))

    bad_hash = copy.deepcopy(inventory)
    bad_hash["source_artifacts"][1]["sha256"] = "0" * 64
    mutations.append(("source hash drift", bad_hash))

    failures: list[str] = []
    for name, mutated in mutations:
        report = Report()
        validate(report, mutated, graph_oracle, texts)
        if report.ok:
            failures.append(f"{name}: validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(mutations)} mutations rejected")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Backend boundary inventory: {status}, {report.checks} checks")
    if report.errors:
        print_errors(report.errors)


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
