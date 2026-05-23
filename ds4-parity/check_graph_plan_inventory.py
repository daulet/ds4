#!/usr/bin/env python3
"""Validate the M10.2 graph plan and GPU operation inventory oracle."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = (
    ROOT / "ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json"
)

REQUIRED_CONSTANTS = [
    "DS4_N_LAYER",
    "DS4_N_EMBD",
    "DS4_N_VOCAB",
    "DS4_N_HEAD",
    "DS4_N_HEAD_KV",
    "DS4_N_HEAD_DIM",
    "DS4_N_VALUE_DIM",
    "DS4_N_ROT",
    "DS4_N_OUT_GROUP",
    "DS4_N_LORA_Q",
    "DS4_N_LORA_O",
    "DS4_N_EXPERT",
    "DS4_N_EXPERT_USED",
    "DS4_N_EXPERT_SHARED",
    "DS4_N_FF_EXP",
    "DS4_N_HASH_LAYER",
    "DS4_N_SWA",
    "DS4_N_INDEXER_HEAD",
    "DS4_N_INDEXER_HEAD_DIM",
    "DS4_N_INDEXER_TOP_K",
    "DS4_N_HC",
    "DS4_N_HC_SINKHORN_ITER",
]

REQUIRED_BOUNDARY_FUNCTIONS = [
    "metal_graph_indexer_stage_profile_boundary",
    "metal_graph_layer_stage_profile_boundary",
    "metal_graph_q_stage_profile_boundary",
    "metal_graph_warmup_prefill_kernels",
    "metal_graph_eval_token_raw_swa",
    "metal_graph_eval_token_raw_swa_top",
    "metal_graph_eval_mtp_draft_from_hc",
    "metal_graph_prefill_layer_major",
    "metal_graph_prefill_batch_row_logits",
    "metal_graph_prefill_chunked_range",
    "metal_graph_verify_suffix_tops",
    "metal_graph_verify_decode2_exact",
    "spec_frontier_snapshot",
    "spec_frontier_restore",
    "spec_frontier_commit_prefix1",
]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.negative_test:
        return run_negative_tests(args.oracle)

    oracle = load_json(args.oracle)
    errors = validate_oracle(oracle)
    if args.json:
        print(json.dumps(report_payload(oracle, errors), indent=2, sort_keys=True))
    elif errors:
        print_errors(errors)
    else:
        counts = report_counts(oracle)
        print(
            "graph plan inventory oracle: "
            f"{counts['operations']} operations, "
            f"{counts['tensor_fields']} tensor fields, "
            f"{counts['plan_cases']} plan cases, "
            f"{counts['command_boundaries']} command boundaries"
        )
    return 1 if errors else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=DEFAULT_ORACLE,
        help="graph plan inventory JSON fixture to validate",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable validation report",
    )
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="mutate the fixture in memory and require validation failures",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return value


def validate_oracle(oracle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source = read_source()
    source_clean = strip_c_comments(source)
    header = read_header()
    header_clean = strip_c_comments(header)

    expect_value(errors, oracle, "schema", "ds4.graph_plan_inventory.v1")
    expect_value(errors, oracle, "milestone", "M10.2")

    constants = extract_model_constants(source_clean)
    validate_constants(errors, oracle.get("model_constants"), constants)
    validate_compression_source(errors, source_clean)
    validate_cap_source(errors, source_clean)
    validate_assumptions(errors, oracle.get("assumptions"))

    declared_ops = extract_gpu_names(header_clean)
    called_ops = extract_gpu_names(source_clean)
    validate_operation_groups(errors, oracle.get("operation_groups"), declared_ops, called_ops)

    graph_fields = extract_graph_tensor_fields(source_clean)
    tensor_owner_names = validate_tensor_groups(
        errors,
        oracle.get("tensor_owner_groups"),
        graph_fields,
    )

    validate_plan_cases(
        errors,
        oracle.get("graph_plan_cases"),
        constants,
        tensor_owner_names,
    )
    validate_command_boundaries(errors, oracle.get("command_boundaries"), source_clean)
    return errors


def read_source() -> str:
    return (ROOT / "ds4.c").read_text()


def read_header() -> str:
    return (ROOT / "ds4_gpu.h").read_text()


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def expect_value(
    errors: list[str],
    data: dict[str, Any],
    key: str,
    expected: Any,
) -> None:
    actual = data.get(key)
    if actual != expected:
        errors.append(f"{key}: expected {expected!r}, got {actual!r}")


def extract_model_constants(source: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    for name, value in re.findall(r"\b(DS4_N_[A-Z0-9_]+)\s*=\s*([0-9]+)\s*,", source):
        constants[name] = int(value)
    return constants


def validate_constants(
    errors: list[str],
    fixture: Any,
    constants: dict[str, int],
) -> None:
    if not isinstance(fixture, dict):
        errors.append("model_constants: expected object")
        return
    for name in REQUIRED_CONSTANTS:
        if name not in constants:
            errors.append(f"source constants: missing {name}")
            continue
        if fixture.get(name) != constants[name]:
            errors.append(
                f"model_constants.{name}: expected {constants[name]!r}, "
                f"got {fixture.get(name)!r}"
            )
    extra = sorted(set(fixture) - set(REQUIRED_CONSTANTS))
    for name in extra:
        errors.append(f"model_constants.{name}: unexpected constant in fixture")


def validate_compression_source(errors: list[str], source: str) -> None:
    body = extract_function_body(source, "ds4_layer_compress_ratio")
    if not body:
        errors.append("ds4_layer_compress_ratio: function body not found")
        return
    required_fragments = [
        "if (il < 2) return 0;",
        "return (il & 1u) == 0 ? 4u : 128u;",
    ]
    for fragment in required_fragments:
        if fragment not in body:
            errors.append(
                "ds4_layer_compress_ratio: missing expected source fragment "
                f"{fragment!r}"
            )


def validate_cap_source(errors: list[str], source: str) -> None:
    expected_fragments = {
        "ds4_default_prefill_cap_for_prompt": [
            'const char *env = getenv("DS4_METAL_PREFILL_CHUNK");',
            "} else if (prompt_len > 2048) {",
            "cap = 2048u;",
            "if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;",
        ],
        "metal_graph_raw_cap_for_context": [
            "wanted = align_up(wanted, 256u);",
            "if (wanted > 8192u) wanted = 8192u;",
            'const char *env = getenv("DS4_METAL_GRAPH_RAW_CAP");',
            "if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;",
        ],
        "metal_graph_alloc_raw_cap": [
            "if (raw_cap > ctx_size) raw_cap = ctx_size;",
            "g->comp_cap = ctx_size / min_ratio + 2u;",
            "g->layer_comp_cap[il] = ctx_size / ratio + 2u;",
        ],
    }
    for function, fragments in expected_fragments.items():
        body = extract_function_body(source, function)
        if not body:
            errors.append(f"{function}: function body not found")
            continue
        for fragment in fragments:
            if fragment not in body:
                errors.append(f"{function}: missing expected source fragment {fragment!r}")


def validate_assumptions(errors: list[str], fixture: Any) -> None:
    if not isinstance(fixture, dict):
        errors.append("assumptions: expected object")
        return
    env = fixture.get("environment")
    if not isinstance(env, dict):
        errors.append("assumptions.environment: expected object")
        return
    expected = {
        "DS4_METAL_PREFILL_CHUNK": "unset",
        "DS4_METAL_GRAPH_RAW_CAP": "unset",
    }
    for key, value in expected.items():
        if env.get(key) != value:
            errors.append(
                f"assumptions.environment.{key}: expected {value!r}, "
                f"got {env.get(key)!r}"
            )


def extract_gpu_names(text: str) -> set[str]:
    return set(re.findall(r"\b(ds4_gpu_[A-Za-z0-9_]+)\s*\(", text))


def validate_operation_groups(
    errors: list[str],
    fixture: Any,
    declared_ops: set[str],
    called_ops: set[str],
) -> None:
    groups = expect_list(errors, fixture, "operation_groups")
    assigned: dict[str, tuple[str, str]] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"operation_groups[{index}]: expected object")
            continue
        name = expect_nonempty_str(errors, group, f"operation_groups[{index}]", "name")
        facade = expect_nonempty_str(
            errors, group, f"operation_groups[{index}]", "rust_facade"
        )
        operations = expect_str_list(
            errors, group.get("operations"), f"operation_groups[{index}].operations"
        )
        for op in operations:
            prev = assigned.get(op)
            if prev:
                errors.append(
                    f"operation {op}: assigned to both {prev[0]} and {name}"
                )
            assigned[op] = (name, facade)

    assigned_ops = set(assigned)
    for op in sorted(declared_ops - assigned_ops):
        errors.append(f"operation_groups: ds4_gpu.h operation {op} is unassigned")
    for op in sorted(assigned_ops - declared_ops):
        errors.append(f"operation_groups: fixture operation {op} is not in ds4_gpu.h")

    for op in sorted(called_ops - declared_ops):
        errors.append(f"ds4.c: calls undeclared GPU operation {op}")
    for op in sorted((called_ops & declared_ops) - assigned_ops):
        errors.append(f"ds4.c: called GPU operation {op} has no Rust facade target")


def extract_graph_tensor_fields(source: str) -> set[str]:
    match = re.search(
        r"typedef\s+struct\s*\{(?P<body>.*?)\}\s*ds4_gpu_graph\s*;",
        source,
        flags=re.S,
    )
    if not match:
        return set()
    body = match.group("body")
    for line in body.splitlines():
        if "ds4_gpu_tensor *" in line and "," in line.split(";", 1)[0]:
            raise SystemExit(
                "unsupported ds4_gpu_graph tensor declaration style: "
                f"{line.strip()}"
            )
    return set(
        re.findall(
            r"\bds4_gpu_tensor\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\s*\[[^\]]+\])?\s*;",
            body,
        )
    )


def validate_tensor_groups(
    errors: list[str],
    fixture: Any,
    graph_fields: set[str],
) -> set[str]:
    if not graph_fields:
        errors.append("ds4_gpu_graph: no tensor fields parsed")
        return set()
    groups = expect_list(errors, fixture, "tensor_owner_groups")
    assigned: dict[str, tuple[str, str]] = {}
    group_names: set[str] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"tensor_owner_groups[{index}]: expected object")
            continue
        name = expect_nonempty_str(errors, group, f"tensor_owner_groups[{index}]", "name")
        owner = expect_nonempty_str(
            errors, group, f"tensor_owner_groups[{index}]", "owner"
        )
        if name in group_names:
            errors.append(f"tensor_owner_groups: duplicate group {name!r}")
        group_names.add(name)
        fields = expect_str_list(
            errors, group.get("fields"), f"tensor_owner_groups[{index}].fields"
        )
        for field in fields:
            prev = assigned.get(field)
            if prev:
                errors.append(
                    f"tensor field {field}: assigned to both {prev[0]} and {name}"
                )
            assigned[field] = (name, owner)

    assigned_fields = set(assigned)
    for field in sorted(graph_fields - assigned_fields):
        errors.append(f"tensor_owner_groups: ds4_gpu_graph field {field} is unassigned")
    for field in sorted(assigned_fields - graph_fields):
        errors.append(
            f"tensor_owner_groups: fixture field {field} is not in ds4_gpu_graph"
        )
    return group_names


def validate_plan_cases(
    errors: list[str],
    fixture: Any,
    constants: dict[str, int],
    tensor_owner_group_names: set[str],
) -> None:
    cases = expect_list(errors, fixture, "graph_plan_cases")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"graph_plan_cases[{index}]: expected object")
            continue
        label = expect_nonempty_str(errors, case, f"graph_plan_cases[{index}]", "name")
        if label in seen:
            errors.append(f"graph_plan_cases: duplicate case {label!r}")
        seen.add(label)
        ctx_size = expect_int(errors, case, f"graph_plan_cases[{index}]", "ctx_size")
        prompt_len = expect_int(errors, case, f"graph_plan_cases[{index}]", "prompt_len")
        mtp_enabled = case.get("mtp_enabled")
        if not isinstance(mtp_enabled, bool):
            errors.append(f"graph_plan_cases[{index}].mtp_enabled: expected bool")
            continue
        if ctx_size is None or prompt_len is None:
            continue
        mtp_tensor_group = case.get("mtp_tensor_group")
        if mtp_tensor_group != "none" and mtp_tensor_group not in tensor_owner_group_names:
            errors.append(
                f"graph_plan_cases.{label}.mtp_tensor_group: "
                f"unknown tensor owner group {mtp_tensor_group!r}"
            )
        expected = compute_plan_case(
            constants=constants,
            ctx_size=ctx_size,
            prompt_len=prompt_len,
            mtp_enabled=mtp_enabled,
        )
        for key, expected_value in expected.items():
            actual = case.get(key)
            if actual != expected_value:
                errors.append(
                    f"graph_plan_cases.{label}.{key}: expected {expected_value!r}, "
                    f"got {actual!r}"
                )

    required_contexts = {(128, False), (2048, False), (32768, False), (32768, True)}
    actual_contexts = {
        (case.get("ctx_size"), case.get("mtp_enabled"))
        for case in cases
        if isinstance(case, dict)
    }
    for ctx in sorted(required_contexts - actual_contexts):
        errors.append(
            "graph_plan_cases: missing required context "
            f"ctx_size={ctx[0]} mtp_enabled={ctx[1]}"
        )


def compute_plan_case(
    *,
    constants: dict[str, int],
    ctx_size: int,
    prompt_len: int,
    mtp_enabled: bool,
) -> dict[str, Any]:
    n_layer = constants["DS4_N_LAYER"]
    raw_window_limit = constants["DS4_N_SWA"]
    ratios = [layer_compress_ratio(layer) for layer in range(n_layer)]
    ratio4_layers = [layer for layer, ratio in enumerate(ratios) if ratio == 4]
    ratio128_layers = [layer for layer, ratio in enumerate(ratios) if ratio == 128]
    dense_layers = [layer for layer, ratio in enumerate(ratios) if ratio == 0]

    prefill_cap = default_prefill_cap(prompt_len)
    raw_window = raw_window_limit if raw_window_limit <= ctx_size else ctx_size
    if raw_window == 0:
        raw_window = 1
    requested_raw_cap = metal_graph_raw_cap_for_context(
        ctx_size=ctx_size,
        prefill_cap=prefill_cap,
        raw_window_limit=raw_window_limit,
    )
    allocated_raw_cap = requested_raw_cap
    if allocated_raw_cap < raw_window:
        allocated_raw_cap = raw_window
    if allocated_raw_cap > ctx_size:
        allocated_raw_cap = ctx_size
    if allocated_raw_cap == 0:
        allocated_raw_cap = 1

    min_ratio = min(ratio for ratio in ratios if ratio != 0)
    comp_cap = ctx_size // min_ratio + 2
    if comp_cap < 2:
        comp_cap = 2

    def layer_cap(ratio: int) -> int:
        if ratio == 0:
            return 0
        cap = ctx_size // ratio + 2
        return cap if cap >= 2 else 2

    return {
        "prefill_cap": prefill_cap,
        "raw_window": raw_window,
        "requested_raw_cap": requested_raw_cap,
        "allocated_raw_cap": allocated_raw_cap,
        "comp_cap": comp_cap,
        "layer_counts": {
            "dense": len(dense_layers),
            "ratio4": len(ratio4_layers),
            "ratio128": len(ratio128_layers),
        },
        "layer_comp_cap_by_ratio": {
            "dense": layer_cap(0),
            "ratio4": layer_cap(4),
            "ratio128": layer_cap(128),
        },
        "ratio4_indexer_layers": len(ratio4_layers),
        "mtp_tensor_group": "mtp_optional_state" if mtp_enabled else "none",
    }


def layer_compress_ratio(layer: int) -> int:
    if layer < 2:
        return 0
    return 4 if layer % 2 == 0 else 128


def default_prefill_cap(prompt_len: int) -> int:
    if prompt_len <= 0:
        return 1
    cap = prompt_len
    if prompt_len > 2048:
        cap = 2048
    if cap == 0:
        cap = 1
    if cap > prompt_len:
        cap = prompt_len
    return cap


def metal_graph_raw_cap_for_context(
    *,
    ctx_size: int,
    prefill_cap: int,
    raw_window_limit: int,
) -> int:
    raw_window = raw_window_limit
    if raw_window > ctx_size:
        raw_window = ctx_size
    if raw_window == 0:
        raw_window = 1

    wanted = raw_window + prefill_cap
    if wanted > ctx_size:
        wanted = ctx_size
    if wanted == 0:
        wanted = 1
    wanted = align_up(wanted, 256)
    if wanted > 8192:
        wanted = 8192
    raw_cap = wanted
    if raw_cap < raw_window:
        raw_cap = raw_window
    return raw_cap


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def validate_command_boundaries(
    errors: list[str],
    fixture: Any,
    source: str,
) -> None:
    boundaries = expect_list(errors, fixture, "command_boundaries")
    by_function: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(boundaries):
        if not isinstance(entry, dict):
            errors.append(f"command_boundaries[{index}]: expected object")
            continue
        fn = expect_nonempty_str(errors, entry, f"command_boundaries[{index}]", "c_function")
        if fn in by_function:
            errors.append(f"command_boundaries: duplicate function {fn!r}")
        by_function[fn] = entry
        body = extract_function_body(source, fn)
        if not body:
            errors.append(f"command_boundaries.{fn}: function body not found")
            continue
        begin_count = body.count("ds4_gpu_begin_commands(")
        end_count = body.count("ds4_gpu_end_commands(")
        min_pairs = entry.get("begin_end_min")
        if not isinstance(min_pairs, int) or min_pairs < 1:
            errors.append(f"command_boundaries.{fn}.begin_end_min: expected positive int")
            continue
        if begin_count < min_pairs:
            errors.append(
                f"command_boundaries.{fn}: expected at least {min_pairs} begin "
                f"calls, got {begin_count}"
            )
        if end_count < min_pairs:
            errors.append(
                f"command_boundaries.{fn}: expected at least {min_pairs} end "
                f"calls, got {end_count}"
            )
        if entry.get("synchronize_on_failure") is True and "ds4_gpu_synchronize(" not in body:
            errors.append(
                f"command_boundaries.{fn}: expected synchronize failure path"
            )

    for fn in REQUIRED_BOUNDARY_FUNCTIONS:
        if fn not in by_function:
            errors.append(f"command_boundaries: missing required function {fn}")


def extract_function_body(source: str, name: str) -> str:
    match = re.search(
        r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{",
        source,
        flags=re.S,
    )
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    for pos in range(start, len(source)):
        char = source[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    return ""


def expect_list(errors: list[str], value: Any, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    errors.append(f"{label}: expected list")
    return []


def expect_str_list(errors: list[str], value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{label}: expected list")
        return []
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{label}[{index}]: expected non-empty string")
            continue
        out.append(item)
    return out


def expect_nonempty_str(
    errors: list[str],
    data: dict[str, Any],
    label: str,
    key: str,
) -> str:
    value = data.get(key)
    if isinstance(value, str) and value:
        return value
    errors.append(f"{label}.{key}: expected non-empty string")
    return ""


def expect_int(
    errors: list[str],
    data: dict[str, Any],
    label: str,
    key: str,
) -> int | None:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    errors.append(f"{label}.{key}: expected int")
    return None


def run_negative_tests(path: Path) -> int:
    oracle = load_json(path)
    positive_errors = validate_oracle(oracle)
    if positive_errors:
        print_errors(positive_errors)
        return 1

    mutations = [
        ("missing operation facade", remove_operation),
        ("missing graph tensor owner", remove_tensor_field),
        ("raw-cap drift", mutate_raw_cap),
    ]
    failures = 0
    for name, mutate in mutations:
        candidate = copy.deepcopy(oracle)
        mutate(candidate)
        errors = validate_oracle(candidate)
        if errors:
            failures += 1
        else:
            print(f"negative test {name!r}: expected validation failure", file=sys.stderr)
    if failures != len(mutations):
        return 1
    print(f"graph plan inventory negative test: {failures} mutations failed as expected")
    return 0


def remove_operation(oracle: dict[str, Any]) -> None:
    for group in oracle["operation_groups"]:
        ops = group["operations"]
        if "ds4_gpu_attention_decode_heads_tensor" in ops:
            ops.remove("ds4_gpu_attention_decode_heads_tensor")
            return
    raise AssertionError("negative fixture operation not found")


def remove_tensor_field(oracle: dict[str, Any]) -> None:
    for group in oracle["tensor_owner_groups"]:
        fields = group["fields"]
        if "layer_raw_cache" in fields:
            fields.remove("layer_raw_cache")
            return
    raise AssertionError("negative fixture tensor field not found")


def mutate_raw_cap(oracle: dict[str, Any]) -> None:
    oracle["graph_plan_cases"][0]["allocated_raw_cap"] = 999


def report_payload(oracle: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "ok": not errors,
        "errors": errors,
        "counts": report_counts(oracle),
    }


def report_counts(oracle: dict[str, Any]) -> dict[str, int]:
    return {
        "operations": sum(
            len(group.get("operations", []))
            for group in oracle.get("operation_groups", [])
            if isinstance(group, dict)
        ),
        "tensor_fields": sum(
            len(group.get("fields", []))
            for group in oracle.get("tensor_owner_groups", [])
            if isinstance(group, dict)
        ),
        "plan_cases": len(oracle.get("graph_plan_cases", [])),
        "command_boundaries": len(oracle.get("command_boundaries", [])),
    }


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"graph plan inventory: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
