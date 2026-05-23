#!/usr/bin/env python3
"""Compare the Rust decode runtime-state bridge against graph, weight, and trace oracles."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import compare_decode_backend_facade
import compare_rust_weight_table
import compare_tensor_bindings as tensor_fixture


ROOT = Path(__file__).resolve().parents[1]
RUST_FACADE = ROOT / "rust/ds4-gpu/src/decode_backend.rs"
CASE_NAME = "ctx32768_mtp_off"
N_LAYER = 43
N_INDEXER_TOP_K = 512


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


def run_json(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def run_runtime_bridge() -> dict[str, Any]:
    return run_json(["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-decode-runtime-bridge", "--quiet"])


def run_graph_state_plan() -> dict[str, Any]:
    return run_json(
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
        ]
    )


def run_decode_trace() -> dict[str, Any]:
    return run_json(["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-decode-trace", "--quiet"])


def run_weight_dump() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ds4-runtime-bridge-weights-") as tmp:
        base_path = Path(tmp) / "base.gguf"
        tensor_fixture.write_gguf(base_path, tensor_fixture.base_tensors(), include_metadata=True)
        return compare_rust_weight_table.run_rust_dump(base_path)


def first_case(obj: dict[str, Any]) -> dict[str, Any]:
    case = obj.get("case")
    return case if isinstance(case, dict) else {}


def allocation_key(entry: dict[str, Any]) -> tuple[str | None, int | None]:
    return entry.get("field"), entry.get("layer")


def handle_map(handles: list[dict[str, Any]]) -> dict[tuple[str | None, int | None], dict[str, Any]]:
    return {allocation_key(entry): entry for entry in handles}


def graph_allocation_map(graph_case: dict[str, Any]) -> dict[tuple[str | None, int | None], dict[str, Any]]:
    allocations = graph_case.get("allocations", [])
    if not isinstance(allocations, list):
        return {}
    return handle_map(allocations)


def compress_ratio(layer: int) -> int:
    if layer < 2:
        return 0
    return 4 if layer % 2 == 0 else 128


def validate(
    report: Report,
    runtime: dict[str, Any],
    graph: dict[str, Any],
    trace: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    report.check(runtime.get("schema") == "ds4.decode_runtime.v1", "schema drift")
    report.check(runtime.get("scope") == "dry_run_bridge", "scope drift")
    case = first_case(runtime)
    report.check(case.get("name") == CASE_NAME, "case name drift")
    report.check(case.get("ctx_size") == 32768, "ctx_size drift")
    report.check(case.get("prompt_len") == 32768, "prompt_len drift")
    report.check(case.get("mtp_enabled") is False, "mtp flag drift")

    graph_case = (graph.get("cases") or [{}])[0]
    validate_handles(report, case, graph_case)
    validate_counters(report, case)
    validate_facade_bindings(report, case, trace)
    validate_weight_requirements(report, case, weights)


def validate_handles(report: Report, case: dict[str, Any], graph_case: dict[str, Any]) -> None:
    handles = case.get("handles")
    summary = case.get("summary")
    graph_allocs = graph_allocation_map(graph_case)
    report.check(isinstance(handles, list), "handles must be a list")
    report.check(isinstance(summary, dict), "summary must be an object")
    if not isinstance(handles, list) or not isinstance(summary, dict):
        return
    report.check(len(handles) == len(graph_allocs), "handle count differs from graph-state plan")
    got = handle_map(handles)
    report.check(set(got) == set(graph_allocs), "handle field/layer set differs from graph-state plan")

    comparable = [
        "owner",
        "storage",
        "view_base",
        "view_offset_bytes",
        "initial_fill",
        "bytes",
        "initially_allocated",
    ]
    for key, expected in graph_allocs.items():
        entry = got.get(key, {})
        for field in comparable:
            report.check(entry.get(field) == expected.get(field), f"{key}: handle {field} drift")

    graph_summary = graph_case.get("summary", {})
    report.check(summary.get("logical_handles") == graph_summary.get("logical_instances"), "logical handle summary drift")
    report.check(
        summary.get("initial_owned_allocations") == graph_summary.get("initial_owned_allocations"),
        "owned allocation summary drift",
    )
    report.check(summary.get("views") == graph_summary.get("views"), "view summary drift")
    report.check(summary.get("lazy_owned") == graph_summary.get("lazy_owned"), "lazy summary drift")
    report.check(summary.get("external_inputs") == graph_summary.get("external_inputs"), "external summary drift")
    report.check(summary.get("initial_layer_counters") == N_LAYER, "layer counter summary drift")


def validate_counters(report: Report, case: dict[str, Any]) -> None:
    counters = case.get("initial_layer_counters")
    report.check(isinstance(counters, list), "initial_layer_counters must be a list")
    if not isinstance(counters, list):
        return
    report.check(len(counters) == N_LAYER, "counter layer count drift")
    by_layer = {entry.get("layer"): entry for entry in counters}
    for layer in range(N_LAYER):
        entry = by_layer.get(layer, {})
        ratio = compress_ratio(layer)
        expected_compression = "dense" if ratio == 0 else f"ratio{ratio}"
        report.check(entry.get("compression") == expected_compression, f"layer {layer}: compression drift")
        report.check(entry.get("layer_n_comp") == 0, f"layer {layer}: initial n_comp drift")
        report.check(entry.get("layer_n_index_comp") == 0, f"layer {layer}: initial n_index_comp drift")
        report.check(entry.get("indexer_top_k") == N_INDEXER_TOP_K, f"layer {layer}: indexer top-k drift")
        if ratio == 0:
            report.check(entry.get("layer_comp_cap") == 0, f"layer {layer}: dense comp cap drift")
        elif ratio == 4:
            report.check(entry.get("layer_comp_cap") == 8194, f"layer {layer}: ratio4 comp cap drift")
        else:
            report.check(entry.get("layer_comp_cap") == 258, f"layer {layer}: ratio128 comp cap drift")


def validate_facade_bindings(report: Report, case: dict[str, Any], trace: dict[str, Any]) -> None:
    handles = case.get("handles", [])
    bindings = case.get("facade_arg_bindings")
    requirements = case.get("weight_requirements", [])
    report.check(isinstance(bindings, list), "facade_arg_bindings must be a list")
    if not isinstance(bindings, list):
        return
    handle_fields = {entry.get("field") for entry in handles if isinstance(entry, dict)}
    weight_fields = {entry.get("field") for entry in requirements if isinstance(entry, dict)}
    binding_keys = [(entry.get("operation"), entry.get("arg")) for entry in bindings]
    report.check(len(binding_keys) == len(set(binding_keys)), "duplicate facade arg binding keys")
    binding_map = {(entry.get("operation"), entry.get("arg")): entry for entry in bindings}

    facade_specs = compare_decode_backend_facade.parse_facade_specs(RUST_FACADE.read_text())
    for spec in facade_specs:
        if spec.operation not in compare_decode_backend_facade.DEFAULT_DECODE_OPERATIONS:
            continue
        for arg in spec.tensor_args:
            binding = binding_map.get((spec.operation, arg))
            report.check(binding is not None, f"{spec.operation}:{arg} missing runtime binding")
            if binding is not None:
                validate_binding_candidates(report, binding, handle_fields, weight_fields)

    cases = trace.get("cases", [])
    report.check(isinstance(cases, list), "trace cases must be a list")
    if isinstance(cases, list):
        for trace_case in cases:
            for event in trace_case.get("events", []):
                if event.get("kind") != "facade":
                    continue
                operation = event.get("operation")
                for arg in event.get("tensor_args", []):
                    report.check((operation, arg) in binding_map, f"trace arg has no runtime binding: {operation}:{arg}")

    routed_experts = binding_map.get(("ds4_gpu_routed_moe_one_tensor", "experts"), {})
    report.check(routed_experts.get("source") == "weight", "routed experts must resolve through weights")
    report.check(
        routed_experts.get("candidates") == ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
        "routed experts weight candidate drift",
    )


def validate_binding_candidates(
    report: Report,
    binding: dict[str, Any],
    handle_fields: set[Any],
    weight_fields: set[Any],
) -> None:
    candidates = binding.get("candidates")
    report.check(isinstance(candidates, list) and len(candidates) > 0, f"{binding}: empty candidates")
    if not isinstance(candidates, list):
        return
    source = binding.get("source")
    if source == "state":
        for candidate in candidates:
            report.check(candidate in handle_fields, f"{binding.get('operation')}:{binding.get('arg')} unknown state {candidate}")
    elif source == "weight":
        for candidate in candidates:
            report.check(candidate in weight_fields, f"{binding.get('operation')}:{binding.get('arg')} unknown weight {candidate}")
    else:
        report.check(source == "external", f"{binding.get('operation')}:{binding.get('arg')} source drift")


def validate_weight_requirements(report: Report, case: dict[str, Any], weights: dict[str, Any]) -> None:
    requirements = case.get("weight_requirements")
    report.check(isinstance(requirements, list), "weight_requirements must be a list")
    if not isinstance(requirements, list):
        return
    bound = weights.get("bound_tensors", [])
    role_map = {entry.get("role"): entry for entry in bound if isinstance(entry, dict)}
    report.check(len(requirements) == 111, "weight requirement count drift")
    roles = [entry.get("role") for entry in requirements]
    report.check(len(roles) == len(set(roles)), "duplicate weight requirement role")
    for entry in requirements:
        role = entry.get("role")
        binding = role_map.get(role)
        report.check(binding is not None, f"missing M10.5c1 weight role {role}")
        if binding is None:
            continue
        presence = entry.get("presence")
        present = binding.get("present")
        if presence == "required_present":
            report.check(present is True, f"{role}: required role absent")
            validate_present_weight(report, role, binding)
        elif presence == "expected_absent":
            report.check(present is False, f"{role}: expected absent role is present")
        elif presence == "optional":
            if present is True:
                validate_present_weight(report, role, binding)
        else:
            report.check(False, f"{role}: unknown presence {presence}")

    assert_role(report, role_map, "base.token_embd", True, "f16")
    assert_role(report, role_map, "base.output", True, "q8_0")
    assert_role(report, role_map, "base.layer.0.ffn_gate_tid2eid", True, "i32")
    assert_role(report, role_map, "base.layer.2.indexer_proj", True, "f16")
    assert_role(report, role_map, "base.layer.3.indexer_proj", False, None)
    assert_role(report, role_map, "base.layer.3.attn_compressor_ape", True, "f16")


def validate_present_weight(report: Report, role: Any, binding: dict[str, Any]) -> None:
    report.check(isinstance(binding.get("abs_offset"), int), f"{role}: abs_offset missing")
    report.check(isinstance(binding.get("type"), int), f"{role}: type missing")
    report.check(isinstance(binding.get("type_name"), str), f"{role}: type_name missing")
    report.check(isinstance(binding.get("bytes"), int) and binding.get("bytes") > 0, f"{role}: bytes missing")


def assert_role(
    report: Report,
    role_map: dict[Any, dict[str, Any]],
    role: str,
    present: bool,
    type_name: str | None,
) -> None:
    binding = role_map.get(role, {})
    report.check(binding.get("present") is present, f"{role}: representative presence drift")
    if present:
        report.check(binding.get("type_name") == type_name, f"{role}: representative type drift")
        report.check(isinstance(binding.get("abs_offset"), int), f"{role}: representative offset missing")


def run_negative_tests(
    report: Report,
    runtime: dict[str, Any],
    graph: dict[str, Any],
    trace: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    mutations = [
        ("summary", mutate_summary),
        ("handle", mutate_handle),
        ("binding", mutate_binding),
        ("weight-role", mutate_weight_role),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(mutated_report, mutate(runtime), graph, trace, weights)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(runtime)
    mutated["case"]["summary"]["logical_handles"] -= 1
    return mutated


def mutate_handle(runtime: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(runtime)
    mutated["case"]["handles"] = [
        entry
        for entry in mutated["case"]["handles"]
        if not (entry.get("field") == "layer_index_comp_cache" and entry.get("layer") == 2)
    ]
    return mutated


def mutate_binding(runtime: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(runtime)
    for entry in mutated["case"]["facade_arg_bindings"]:
        if entry.get("operation") == "ds4_gpu_attention_decode_heads_tensor" and entry.get("arg") == "raw_kv":
            entry["candidates"] = ["missing_raw_cache"]
            return mutated
    raise AssertionError("raw_kv binding not found")


def mutate_weight_role(runtime: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(runtime)
    for entry in mutated["case"]["weight_requirements"]:
        if entry.get("role") == "base.layer.2.indexer_proj":
            entry["presence"] = "expected_absent"
            return mutated
    raise AssertionError("indexer weight requirement not found")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Rust decode runtime bridge comparator: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(argv)

    runtime = run_runtime_bridge()
    graph = run_graph_state_plan()
    trace = run_decode_trace()
    weights = run_weight_dump()
    report = Report()
    validate(report, runtime, graph, trace, weights)
    if args.negative_test:
        run_negative_tests(report, runtime, graph, trace, weights)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
