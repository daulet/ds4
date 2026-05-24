#!/usr/bin/env python3
"""Validate the M12.3 Rust backend facade replay harness."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import compare_decode_backend_facade


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "ds4-parity/baselines/backend/m12.3/facade-replay.json"
M12_2_MANIFEST = ROOT / "ds4-parity/baselines/backend/m12.2/manifest.json"
M12_1_INVENTORY = ROOT / "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json"
RUST_FACADE = ROOT / "rust/ds4-gpu/src/decode_backend.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_FIXTURES = [
    "first_kernel_embed_token_hc",
    "layer0_qkv_rope",
    "layer0_attention_output",
    "layer0_ffn_router_moe",
    "full_output_head_logits",
]
ALLOWED_BINDING_SOURCES = {"state", "weight", "external"}
ALLOWED_BINDING_MODES = {"mut", "ref", "optional_ref"}


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
    replay = load_json(REPLAY)
    manifest = load_json(M12_2_MANIFEST)
    inventory = load_json(M12_1_INVENTORY)
    facade_source = read_text(RUST_FACADE)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, replay, manifest, inventory, facade_source, texts)
    if args.negative_test:
        run_negative_tests(report, replay, manifest, inventory, facade_source, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    replay: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    facade_source: str,
    texts: dict[str, str],
) -> None:
    report.check(replay.get("schema") == "ds4.backend_facade_replay.v1", "schema drift")
    report.check(replay.get("milestone") == "M12.3", "milestone drift")
    report.check(replay.get("previous_stage") == "M12.2", "previous stage drift")
    report.check(replay.get("next_stage") == "M12.4", "next stage drift")
    report.check(replay.get("status") == "facade-replay-harness", "status drift")
    validate_oracles(report, replay)
    validate_claim_policy(report, replay)
    validate_error_policy(report, replay)
    validate_comparison_policy(report, replay)
    validate_replays(report, replay, manifest, inventory, facade_source)
    validate_static_wiring(report, texts)


def validate_oracles(report: Report, replay: dict[str, Any]) -> None:
    oracle = replay.get("oracle")
    report.check(isinstance(oracle, dict), "oracle block missing")
    if not isinstance(oracle, dict):
        return
    expected = {
        "operation_fixture_manifest": "ds4-parity/baselines/backend/m12.2/manifest.json",
        "backend_boundary_inventory": "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json",
        "rust_facade_source": "rust/ds4-gpu/src/decode_backend.rs",
        "facade_contract_comparator": "ds4-parity/compare_decode_backend_facade.py",
        "runtime_bridge_comparator": "ds4-parity/compare_decode_runtime_bridge.py",
    }
    for key, value in expected.items():
        got = oracle.get(key)
        report.check(got == value, f"oracle {key} drift")
        if isinstance(got, str) and got.startswith("ds4-parity/") or isinstance(got, str) and got.startswith("rust/"):
            report.check((ROOT / got).exists(), f"oracle path missing: {got}")


def validate_claim_policy(report: Report, replay: dict[str, Any]) -> None:
    policy = replay.get("claim_policy")
    report.check(isinstance(policy, dict), "claim policy missing")
    if not isinstance(policy, dict):
        return
    report.check(policy.get("backend_replacement") == "not_claimed", "backend replacement overclaim")
    report.check(policy.get("kernel_replacement") == "not_claimed", "kernel replacement overclaim")
    report.check(policy.get("runtime_route_change") is False, "runtime route change overclaim")
    report.check(policy.get("route_gate") == "blocked_until_m12.6", "route gate drift")
    report.check(policy.get("next_required_gate") == "M12.4 First Backend Replacement Slice", "next required gate drift")


def validate_error_policy(report: Report, replay: dict[str, Any]) -> None:
    policy = replay.get("error_policy")
    report.check(isinstance(policy, dict), "error policy missing")
    if not isinstance(policy, dict):
        return
    report.check(policy.get("return_type") == "Result<(), GpuError>", "return type drift")
    report.check(
        policy.get("backend_status_mapping") == "GpuStatus::from_raw(...).into_result()",
        "backend status mapping drift",
    )
    kinds = policy.get("accepted_error_kinds")
    report.check(kinds == ["BackendStatus", "NullTensor"], "accepted error kinds drift")
    report.check(policy.get("panic_policy") == "no panic path in selected facade wrappers", "panic policy drift")


def validate_comparison_policy(report: Report, replay: dict[str, Any]) -> None:
    policy = replay.get("comparison_policy")
    report.check(isinstance(policy, dict), "comparison policy missing")
    if not isinstance(policy, dict):
        return
    for key in ["call_order", "operation_set", "tensor_bindings", "output_comparison", "synchronization"]:
        report.check(isinstance(policy.get(key), str) and policy[key], f"comparison policy {key} missing")


def validate_replays(
    report: Report,
    replay: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    facade_source: str,
) -> None:
    replays = replay.get("replays")
    report.check(isinstance(replays, list) and len(replays) == len(EXPECTED_FIXTURES), "replay count drift")
    if not isinstance(replays, list):
        return
    report.check([entry.get("fixture_id") for entry in replays] == EXPECTED_FIXTURES, "fixture order drift")

    fixture_by_id = {fixture.get("id"): fixture for fixture in manifest.get("fixtures", []) if isinstance(fixture, dict)}
    families, inventory_operations = inventory_sets(inventory)
    facade_specs = compare_decode_backend_facade.parse_facade_specs(facade_source)
    spec_by_operation = {spec.operation: spec for spec in facade_specs}
    known_methods = {spec.method for spec in facade_specs}

    for entry in replays:
        if not isinstance(entry, dict):
            report.check(False, "replay entry is not an object")
            continue
        fixture_id = entry.get("fixture_id")
        fixture = fixture_by_id.get(fixture_id)
        report.check(fixture is not None, f"{fixture_id}: missing M12.2 fixture")
        if fixture is None:
            continue
        validate_replay_entry(
            report,
            entry,
            fixture,
            families,
            inventory_operations,
            spec_by_operation,
            known_methods,
            facade_source,
        )


def validate_replay_entry(
    report: Report,
    entry: dict[str, Any],
    fixture: dict[str, Any],
    families: set[str],
    inventory_operations: set[str],
    spec_by_operation: dict[str, compare_decode_backend_facade.FacadeSpec],
    known_methods: set[str],
    facade_source: str,
) -> None:
    fixture_id = entry.get("fixture_id")
    report.check(entry.get("operation_family") == fixture.get("operation_family"), f"{fixture_id}: family drift")
    report.check(entry.get("operation_family") in families, f"{fixture_id}: family outside M12.1 inventory")
    report.check(entry.get("case") == fixture.get("case"), f"{fixture_id}: case drift")
    report.check(entry.get("comparator") == fixture.get("comparator"), f"{fixture_id}: comparator drift")
    report.check(entry.get("output_fields") == fixture.get("output_fields"), f"{fixture_id}: output field drift")
    report.check(entry.get("candidate_binary") in fixture.get("rerun_command", ""), f"{fixture_id}: candidate binary drift")
    validate_source_order(report, entry, known_methods)
    validate_synchronization(report, entry, fixture)
    validate_output_fields(report, fixture)

    calls = entry.get("calls")
    report.check(isinstance(calls, list) and calls, f"{fixture_id}: calls missing")
    if not isinstance(calls, list):
        return
    report.check([call.get("ordinal") for call in calls] == list(range(1, len(calls) + 1)), f"{fixture_id}: ordinals drift")
    operations = [call.get("operation") for call in calls]
    report.check(unique_preserving_order(operations) == fixture.get("operations"), f"{fixture_id}: operation order drift")
    report.check(all(operation in inventory_operations for operation in operations), f"{fixture_id}: operation outside inventory")
    for call in calls:
        validate_call(report, fixture_id, call, spec_by_operation, facade_source)


def validate_source_order(
    report: Report,
    entry: dict[str, Any],
    known_methods: set[str],
) -> None:
    fixture_id = entry.get("fixture_id")
    source = entry.get("candidate_source")
    report.check(isinstance(source, str) and source, f"{fixture_id}: candidate source missing")
    if not isinstance(source, str):
        return
    path = ROOT / source
    report.check(path.exists(), f"{fixture_id}: candidate source path missing")
    if not path.exists():
        return
    actual_methods = facade_method_calls(read_text(path), known_methods)
    expected_methods = [call.get("method") for call in entry.get("calls", []) if isinstance(call, dict)]
    report.check(is_subsequence(expected_methods, actual_methods), f"{fixture_id}: source method order drift")


def validate_synchronization(report: Report, entry: dict[str, Any], fixture: dict[str, Any]) -> None:
    fixture_id = entry.get("fixture_id")
    sync = entry.get("synchronization")
    report.check(isinstance(sync, dict), f"{fixture_id}: synchronization missing")
    if not isinstance(sync, dict):
        return
    report.check(sync.get("command_batch") is True, f"{fixture_id}: command batch drift")
    report.check(sync.get("synchronized") is True, f"{fixture_id}: synchronized drift")
    report.check(isinstance(sync.get("sync_point"), str) and sync["sync_point"], f"{fixture_id}: sync point missing")
    candidate = load_json(ROOT / fixture["candidate"]["path"])
    operation = candidate.get("operation", {})
    report.check(operation.get("command_batch") is True, f"{fixture_id}: candidate command batch drift")
    report.check(operation.get("synchronized") is True, f"{fixture_id}: candidate synchronized drift")


def validate_output_fields(report: Report, fixture: dict[str, Any]) -> None:
    fixture_id = fixture.get("id")
    candidate = load_json(ROOT / fixture["candidate"]["path"])
    got = output_fields(candidate)
    for field in fixture.get("output_fields", []):
        report.check(field in got, f"{fixture_id}: candidate output field missing {field}")


def validate_call(
    report: Report,
    fixture_id: Any,
    call: Any,
    spec_by_operation: dict[str, compare_decode_backend_facade.FacadeSpec],
    facade_source: str,
) -> None:
    report.check(isinstance(call, dict), f"{fixture_id}: call is not an object")
    if not isinstance(call, dict):
        return
    operation = call.get("operation")
    method = call.get("method")
    spec = spec_by_operation.get(operation)
    report.check(spec is not None, f"{fixture_id}: facade spec missing for {operation}")
    if spec is None:
        return
    report.check(method == spec.method, f"{fixture_id}: method drift for {operation}")
    report.check(call.get("tensor_args") == spec.tensor_args, f"{fixture_id}: tensor args drift for {operation}")
    validate_bindings(report, fixture_id, operation, spec.tensor_args, call.get("bindings"))
    validate_error_propagation(report, fixture_id, operation, spec.method, call.get("error_kinds"), facade_source)


def validate_bindings(
    report: Report,
    fixture_id: Any,
    operation: Any,
    tensor_args: list[str],
    bindings: Any,
) -> None:
    report.check(isinstance(bindings, dict), f"{fixture_id}: bindings missing for {operation}")
    if not isinstance(bindings, dict):
        return
    report.check(list(bindings) == tensor_args, f"{fixture_id}: binding key order drift for {operation}")
    for arg in tensor_args:
        binding = bindings.get(arg)
        report.check(isinstance(binding, dict), f"{fixture_id}: binding missing for {operation}:{arg}")
        if not isinstance(binding, dict):
            continue
        report.check(binding.get("source") in ALLOWED_BINDING_SOURCES, f"{fixture_id}: binding source drift for {operation}:{arg}")
        report.check(isinstance(binding.get("name"), str) and binding["name"], f"{fixture_id}: binding name missing for {operation}:{arg}")
        report.check(binding.get("mode") in ALLOWED_BINDING_MODES, f"{fixture_id}: binding mode drift for {operation}:{arg}")


def validate_error_propagation(
    report: Report,
    fixture_id: Any,
    operation: str,
    method: str,
    error_kinds: Any,
    facade_source: str,
) -> None:
    report.check(isinstance(error_kinds, list) and "BackendStatus" in error_kinds, f"{fixture_id}: backend status error missing")
    if operation == "ds4_gpu_attention_decode_heads_tensor":
        report.check(error_kinds == ["BackendStatus", "NullTensor"], f"{fixture_id}: optional attention error drift")
    else:
        report.check(error_kinds == ["BackendStatus"], f"{fixture_id}: unexpected local error kind for {operation}")
    body = method_body(facade_source, method)
    report.check(f"sys::{operation}(" in body, f"{fixture_id}: raw sys call missing for {operation}")
    report.check("GpuStatus::from_raw" in body, f"{fixture_id}: status wrapper missing for {operation}")
    report.check(".into_result()" in body, f"{fixture_id}: error mapping missing for {operation}")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M12.3 Backend facade replay harness" in texts["report"], "unified report wiring missing")
    report.check("check_backend_facade_replay.py" in texts["report"], "report checker path missing")
    report.check("Validate the M12.3 Backend facade replay harness" in texts["readme"], "README wiring missing")
    report.check("M12.3: Rust Backend Facade Parity Harness" in texts["roadmap"], "roadmap M12.3 missing")
    report.check("- Status: complete." in texts["roadmap"], "roadmap complete status missing")
    report.check("#### M12.4: First Backend Replacement Slice" in texts["roadmap"], "roadmap M12.4 missing")
    report.check("- Status: active." in texts["roadmap"], "roadmap M12.4 active status missing")
    report.check(
        "Active item: M12.4 First Backend Replacement Slice" in texts["status"]
        or "Active item: M12.5 Runtime Backend Route Gate" in texts["status"]
        or "Active item: M12.6 Backend Replacement Closure And Removal Decision" in texts["status"],
        "status active item missing",
    )
    report.check("Earlier M12.3 Rust Backend Facade Parity Harness" in texts["status"], "status previous item missing")
    report.check("#### M12.3: Rust Backend Facade Parity Harness" in texts["todo"], "TODO M12.3 missing")
    report.check("#### M12.4: First Backend Replacement Slice" in texts["todo"], "TODO M12.4 missing")


def run_negative_tests(
    report: Report,
    replay: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    facade_source: str,
    texts: dict[str, str],
) -> None:
    mutations = [
        ("runtime route overclaim", mutate_route_claim),
        ("missing binding", mutate_missing_binding),
        ("method drift", mutate_method),
        ("call order drift", mutate_call_order),
        ("sync drift", mutate_sync),
        ("output field drift", mutate_output_field),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(mutated_report, mutate(replay), manifest, inventory, facade_source, texts)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_route_claim(replay: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(replay)
    mutated["claim_policy"]["runtime_route_change"] = True
    return mutated


def mutate_missing_binding(replay: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(replay)
    del mutated["replays"][2]["calls"][1]["bindings"]["raw_kv"]
    return mutated


def mutate_method(replay: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(replay)
    mutated["replays"][1]["calls"][6]["method"] = "dsv4_qkv_rms_norm_rows_removed"
    return mutated


def mutate_call_order(replay: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(replay)
    calls = mutated["replays"][1]["calls"]
    calls[0], calls[1] = calls[1], calls[0]
    return mutated


def mutate_sync(replay: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(replay)
    mutated["replays"][4]["synchronization"]["synchronized"] = False
    return mutated


def mutate_output_field(replay: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(replay)
    mutated["replays"][4]["output_fields"][-1] = "missing_logits"
    return mutated


def inventory_sets(inventory: dict[str, Any]) -> tuple[set[str], set[str]]:
    families: set[str] = set()
    operations: set[str] = set()
    for family in inventory.get("operation_families", []):
        if not isinstance(family, dict):
            continue
        name = family.get("name")
        if isinstance(name, str):
            families.add(name)
        for operation in family.get("operations", []):
            if isinstance(operation, str):
                operations.add(operation)
    return families, operations


def facade_method_calls(source: str, known_methods: set[str]) -> list[str]:
    calls = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)
    return [call for call in calls if call in known_methods]


def is_subsequence(expected: list[Any], actual: list[Any]) -> bool:
    pos = 0
    for item in actual:
        if pos < len(expected) and expected[pos] == item:
            pos += 1
    return pos == len(expected)


def unique_preserving_order(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def output_fields(obj: dict[str, Any]) -> set[str]:
    if isinstance(obj.get("outputs"), dict):
        return set(obj["outputs"])
    output = obj.get("output")
    if isinstance(output, dict) and isinstance(output.get("field"), str):
        return {output["field"]}
    return set()


def method_body(source: str, method: str) -> str:
    match = re.search(rf"pub fn {re.escape(method)}\s*\(", source)
    if match is None:
        return ""
    brace = source.find("{", match.end())
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    return ""


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


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Backend facade replay harness: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
