#!/usr/bin/env python3
"""Compare Rust whole-prefill short-prompt readback against the current-C GPU oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "c_helper": ROOT / "ds4_prefill_whole_short_oracle_dump.c",
    "ds4_c": ROOT / "ds4.c",
    "ds4_h": ROOT / "ds4.h",
    "makefile": ROOT / "Makefile",
    "rust_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-prefill-whole-short.rs",
    "decode_backend": ROOT / "rust/ds4-gpu/src/decode_backend.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

ORACLE_SCHEMA = "ds4.prefill_whole_short_oracle.v1"
CANDIDATE_SCHEMA = "ds4.prefill_whole_short.v1"
CASE = "short_italian_fact_whole_prefill"
MODEL_SIZE = 86720111488
TENSOR_DATA_OFFSET = 5333824
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
EXPECTED_FNV1A64 = {
    "after_layer42_hc": "ae4b628a67fc7c26",
    "output_pre": "9241941912d230e4",
    "output_weights": "2fe0938b532e023c",
    "output_embd": "6206299d936928fa",
    "output_norm": "c9134b4723b48974",
    "logits": "7877e120ea5c8ffc",
    "layer2_raw_cache_row": "3459d367bb2f78b8",
    "layer2_attn_comp_row4": "1f4db5ca80581fb1",
    "layer2_index_comp_row4": "30318362509afec2",
    "layer5_attn_state_kv": "204d08567e0bfd60",
    "layer5_attn_state_score": "6810b4c1208c5c1c",
    "layer42_raw_cache_row": "c614486040bde402",
    "layer42_attn_comp_row4": "0924a430db895ed1",
    "layer42_index_comp_row4": "3bc03835e8f6a462",
    "layer42_attn_state_kv": "3609955d5fe9ddbe",
    "layer42_index_state_kv": "7de59c091bcc8f80",
}
WEIGHT_ROLES = {
    "token_embd": "base.token_embd",
    "output_hc_fn": "base.output_hc_fn",
    "output_hc_scale": "base.output_hc_scale",
    "output_hc_base": "base.output_hc_base",
    "output_norm": "base.output_norm",
    "output": "base.output",
}
EXPECTED_WEIGHTS = {
    "token_embd": (77928033088, 1059061760, 1, "f16"),
    "output_hc_fn": (86157337440, 131072, 1, "f16"),
    "output_hc_scale": (86157468512, 4, 0, "f32"),
    "output_hc_base": (86157337408, 16, 0, "f32"),
    "output_norm": (86720095104, 16384, 0, "f32"),
    "output": (86157468544, 562626560, 8, "q8_0"),
}
EXPECTED_OPERATION = {
    "boundary": "whole_prefill",
    "fixture": "short_italian_fact",
    "prompt_tokens": 21,
    "output_row": 20,
    "first_layer": 0,
    "last_layer": 42,
    "prefill_layer_calls": 43,
    "dense_layers": 2,
    "ratio4_layers": 21,
    "ratio128_layers": 20,
    "ctx_size": 32768,
    "plan_prompt_len": 32768,
    "prefill_cap": 2048,
    "raw_cap": 2304,
    "raw_window": 128,
    "raw_row": 20,
    "raw_start": 0,
    "n_raw": 21,
    "n_vocab": 129280,
    "vocab_dim": 129280,
    "n_embd": 4096,
    "n_hc": 4,
    "hc_dim": 16384,
    "output_pre_dim": 4,
    "output_embd_dim": 4096,
    "head_dim": 512,
    "indexer_head_dim": 128,
    "layer2_comp_cap": 8194,
    "layer2_n_comp": 5,
    "layer2_n_index_comp": 5,
    "layer5_comp_cap": 258,
    "layer5_n_comp": 0,
    "layer42_comp_cap": 8194,
    "layer42_n_comp": 5,
    "layer42_n_index_comp": 5,
}
EXPECTED_FLOAT_OPERATION = {
    "rms_eps": 1e-6,
    "hc_eps": 1e-6,
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        if args.oracle is None or args.candidate is None:
            raise SystemExit("--oracle and --candidate must be provided together")
        validate_pair(report, load_json(args.oracle), load_json(args.candidate))

    if report.ok:
        print(f"Rust whole-prefill short current-C oracle comparator: {report.checks} checks")
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
        ("ds4_dump_prefill_whole_short_oracle_json", "C helper does not call whole-prefill oracle dump"),
        ("--model", "C helper model flag missing"),
        ("--prompt", "C helper prompt flag missing"),
        ("--output", "C helper output flag missing"),
    ):
        report.check(needle in c_helper, message)

    ds4_c = texts["ds4_c"]
    for needle, message in (
        ("ds4_dump_prefill_whole_short_oracle_json", "C oracle dump implementation missing"),
        ("graph_checkpoint_prompt_tokens", "C oracle does not use the chat prompt tokenization path"),
        ("metal_graph_prefill_layer_major", "C oracle does not use production layer-major prefill"),
        ("short_italian_fact_whole_prefill", "C oracle case missing"),
        ("\\\"prompt_tokens\\\"", "C oracle does not emit prompt token count"),
        ("\\\"output_row\\\"", "C oracle does not emit output row"),
        ("\"layer2_raw_cache_row\"", "C oracle does not emit layer2 raw cache row"),
        ("\"layer2_attn_comp_row4\"", "C oracle does not emit layer2 compressed attention row"),
        ("\"layer2_index_comp_row4\"", "C oracle does not emit layer2 compressed index row"),
        ("\"layer5_attn_state_kv\"", "C oracle does not emit layer5 attention state"),
        ("\"layer5_attn_state_score\"", "C oracle does not emit layer5 score state"),
        ("\"layer42_raw_cache_row\"", "C oracle does not emit layer42 raw cache row"),
        ("\"layer42_attn_comp_row4\"", "C oracle does not emit layer42 compressed attention row"),
        ("\"layer42_index_comp_row4\"", "C oracle does not emit layer42 compressed index row"),
        ("\"layer42_attn_state_kv\"", "C oracle does not emit layer42 attention state"),
        ("\"layer42_index_state_kv\"", "C oracle does not emit layer42 index state"),
        ("\"after_layer42_hc\"", "C oracle does not emit final HC output"),
        ("\"logits\"", "C oracle does not emit logits"),
        (ORACLE_SCHEMA, "C oracle schema missing"),
    ):
        report.check(needle in ds4_c, message)

    report.check(
        "int ds4_dump_prefill_whole_short_oracle_json" in texts["ds4_h"],
        "header declaration for whole-prefill short oracle missing",
    )
    makefile = texts["makefile"]
    report.check("ds4-prefill-whole-short-oracle-dump:" in makefile, "Makefile helper target missing")
    report.check("ds4_prefill_whole_short_oracle_dump_cpu.o" in makefile, "CPU helper object missing")

    rust_bin = texts["rust_bin"]
    for needle, message in (
        (CANDIDATE_SCHEMA, "Rust candidate schema missing"),
        (CASE, "Rust candidate case missing"),
        ("render_chat_prompt_text", "Rust candidate does not render chat prompt text"),
        ("Ds4Tokenizer::from_gguf", "Rust candidate does not use GGUF tokenizer"),
        ("GraphPlan::for_context(CTX_SIZE, CTX_SIZE, false)", "graph plan raw-cap setup missing"),
        (".embed_tokens_hc(", "Rust candidate does not embed prompt tokens through the safe facade"),
        ("for layer in 0..N_LAYER", "Rust candidate does not loop all layers"),
        ("execute_prefill_layer(", "Rust prefill layer execution helper missing"),
        ("std::mem::swap(&mut state.cur_hc, &mut state.after_ffn_hc)", "Rust HC buffer swap missing"),
        (".rms_norm_plain_rows(", "batch RMS facade call missing"),
        (".store_raw_kv_batch(", "batch raw KV store facade call missing"),
        (".compressor_prefill(", "batch compressor prefill facade call missing"),
        (".compressor_prefill_state_ratio4(", "ratio4 compressor state refresh missing"),
        (".attention_prefill_raw_heads(", "raw prefill attention facade call missing"),
        (".attention_prefill_static_mixed_heads(", "static mixed prefill attention facade call missing"),
        (".router_select_batch(", "batch router facade call missing"),
        (".routed_moe_batch(", "batch routed MoE facade call missing"),
        (".hc_expand_add_split(", "batch HC expand-add facade call missing"),
        ("read_tensor_output(\"after_layer42_hc\"", "final HC readback missing"),
        ("read_tensor_output(\"logits\"", "logits readback missing"),
        ("\"layer42_index_state_kv\"", "layer42 index state readback missing"),
        ("fnv1a64(&data)", "Rust full-buffer FNV digest missing"),
        ("BackendGuard", "cleanup guard missing"),
    ):
        report.check(needle in rust_bin, message)

    decode_backend = texts["decode_backend"]
    for needle, message in (
        ("PREFILL_FACADE_OPERATIONS", "prefill facade operation inventory missing"),
        ("pub fn embed_tokens_hc", "embed_tokens_hc safe wrapper missing"),
        ("pub fn attention_prefill_static_mixed_heads", "static mixed prefill wrapper missing"),
        ("pub fn routed_moe_batch", "routed MoE batch wrapper missing"),
        ("prefill_facade_operations_are_pinned", "prefill facade operation pin test missing"),
    ):
        report.check(needle in decode_backend, message)

    report_text = texts["report"]
    report.check(
        "M10.6b Rust whole-prefill short comparator" in report_text,
        "unified report comparator missing",
    )
    report.check(
        "M10.6b B300 whole-prefill short oracle rerun" in report_text,
        "unified report B300 skip missing",
    )
    report.check(
        "ds4-prefill-whole-short-oracle-dump" in report_text
        and "compare_prefill_whole_short.py" in report_text,
        "B300 oracle rerun command missing helper/comparator",
    )

    readme = texts["readme"]
    report.check(
        "M10.6b" in readme
        and "compare_prefill_whole_short.py" in readme
        and "ds4-prefill-whole-short-oracle-dump" in readme,
        "README entry missing",
    )
    report.check(
        "M10.6b: Rust Whole-Prefill Short Execution" in texts["roadmap"],
        "roadmap whole-prefill short item missing",
    )
    report.check(
        "M10.6b: Rust Whole-Prefill Short Execution" in texts["todo"],
        "TODO whole-prefill short item missing",
    )


def validate_pair(report: Report, oracle: dict[str, Any], candidate: dict[str, Any]) -> None:
    validate_common(report, oracle, "oracle", ORACLE_SCHEMA)
    validate_common(report, candidate, "candidate", CANDIDATE_SCHEMA)
    report.check(oracle.get("source") == "current-c", "oracle source drift")
    validate_operation_pair(report, oracle.get("operation"), candidate.get("operation"))
    validate_weights_pair(report, oracle.get("weights"), candidate.get("weights"))
    validate_outputs_pair(report, oracle.get("outputs"), candidate.get("outputs"))


def validate_common(report: Report, obj: dict[str, Any], label: str, schema: str) -> None:
    report.check(obj.get("schema") == schema, f"{label} schema drift")
    report.check(obj.get("case") == CASE, f"{label} case drift")

    model = obj.get("model")
    report.check(isinstance(model, dict), f"{label} model missing")
    if isinstance(model, dict):
        report.check(model.get("mapped_size") == MODEL_SIZE, f"{label} model size drift")
        report.check(model.get("tensor_count") == 1328, f"{label} tensor count drift")
        report.check(model.get("tensor_data_offset") == TENSOR_DATA_OFFSET, f"{label} tensor-data offset drift")
        report.check(model.get("bound_layers") == 43, f"{label} bound layer count drift")

    operation = obj.get("operation")
    report.check(isinstance(operation, dict), f"{label} operation missing")
    if isinstance(operation, dict):
        for key, value in EXPECTED_OPERATION.items():
            report.check(operation.get(key) == value, f"{label} operation {key} drift")
        for key, value in EXPECTED_FLOAT_OPERATION.items():
            report.check(abs(float(operation.get(key, 0.0)) - value) <= 1e-12, f"{label} operation {key} drift")

    outputs = obj.get("outputs")
    report.check(isinstance(outputs, dict), f"{label} outputs missing")
    if isinstance(outputs, dict):
        report.check(set(outputs) == set(OUTPUTS), f"{label} output tensor set drift")
        for field, elements in OUTPUTS.items():
            validate_output(report, outputs.get(field), label, field, elements)


def validate_output(report: Report, output: Any, label: str, field: str, elements: int) -> None:
    report.check(isinstance(output, dict), f"{label} {field} output missing")
    if not isinstance(output, dict):
        return
    report.check(output.get("field") == field, f"{label} {field} field drift")
    report.check(output.get("elements") == elements, f"{label} {field} element drift")
    report.check(output.get("bytes") == elements * 4, f"{label} {field} byte drift")
    report.check(isinstance(output.get("nonzero_elements"), int), f"{label} {field} nonzero count missing")
    report.check(isinstance(output.get("fnv1a64"), str), f"{label} {field} FNV digest missing")
    if label == "oracle":
        report.check(is_sha256(output.get("sha256")), f"oracle {field} sha256 invalid")
    expected = EXPECTED_FNV1A64.get(field)
    if expected is not None:
        report.check(output.get("fnv1a64") == expected, f"{label} {field} FNV digest drift")
    validate_samples(report, output.get("samples"), label, field, elements)


def validate_weights_pair(report: Report, oracle_weights: Any, candidate_weights: Any) -> None:
    report.check(isinstance(oracle_weights, dict), "oracle weights missing")
    report.check(isinstance(candidate_weights, dict), "candidate weights missing")
    if not isinstance(oracle_weights, dict) or not isinstance(candidate_weights, dict):
        return
    expected_keys = set(WEIGHT_ROLES)
    report.check(set(oracle_weights) == expected_keys, "oracle weight set drift")
    report.check(set(candidate_weights) == expected_keys, "candidate weight set drift")
    for key, role in WEIGHT_ROLES.items():
        oracle = oracle_weights.get(key)
        candidate = candidate_weights.get(key)
        report.check(isinstance(oracle, dict), f"oracle {key} weight missing")
        report.check(isinstance(candidate, dict), f"candidate {key} weight missing")
        if not isinstance(oracle, dict) or not isinstance(candidate, dict):
            continue
        report.check(oracle.get("role") == role, f"oracle {key} role drift")
        report.check(candidate.get("role") == role, f"candidate {key} role drift")
        for field in ("abs_offset", "bytes", "type", "type_name"):
            report.check(oracle.get(field) == candidate.get(field), f"{key} weight {field} mismatch")
        offset, size, type_id, type_name = EXPECTED_WEIGHTS[key]
        report.check(oracle.get("abs_offset") == offset, f"oracle {key} offset drift")
        report.check(oracle.get("bytes") == size, f"oracle {key} byte-size drift")
        report.check(oracle.get("type") == type_id, f"oracle {key} type drift")
        report.check(oracle.get("type_name") == type_name, f"oracle {key} type-name drift")


def validate_operation_pair(report: Report, oracle_operation: Any, candidate_operation: Any) -> None:
    report.check(isinstance(oracle_operation, dict), "oracle operation missing")
    report.check(isinstance(candidate_operation, dict), "candidate operation missing")
    if not isinstance(oracle_operation, dict) or not isinstance(candidate_operation, dict):
        return
    for key in EXPECTED_OPERATION:
        report.check(
            oracle_operation.get(key) == candidate_operation.get(key),
            f"candidate operation {key} does not match current-C oracle",
        )
    for key in EXPECTED_FLOAT_OPERATION:
        report.check(
            abs(float(oracle_operation.get(key, 0.0)) - float(candidate_operation.get(key, 0.0))) <= 1e-12,
            f"candidate operation {key} does not match current-C oracle",
        )


def validate_outputs_pair(report: Report, oracle_outputs: Any, candidate_outputs: Any) -> None:
    report.check(isinstance(oracle_outputs, dict), "oracle outputs missing")
    report.check(isinstance(candidate_outputs, dict), "candidate outputs missing")
    if not isinstance(oracle_outputs, dict) or not isinstance(candidate_outputs, dict):
        return
    for field in OUTPUTS:
        oracle = oracle_outputs.get(field)
        candidate = candidate_outputs.get(field)
        if not isinstance(oracle, dict) or not isinstance(candidate, dict):
            continue
        report.check(
            oracle.get("fnv1a64") == candidate.get("fnv1a64"),
            f"candidate {field} FNV digest does not match current-C oracle",
        )
        report.check(
            oracle.get("nonzero_elements") == candidate.get("nonzero_elements"),
            f"candidate {field} nonzero count does not match current-C oracle",
        )
        report.check(
            samples_match(oracle.get("samples"), candidate.get("samples")),
            f"candidate {field} samples do not match current-C oracle",
        )


def validate_samples(report: Report, samples: Any, label: str, field: str, elements: int) -> None:
    report.check(isinstance(samples, list), f"{label} {field} samples missing")
    if not isinstance(samples, list):
        return
    by_index = sample_map(samples)
    expected_indices = set(sample_indices(elements))
    report.check(expected_indices <= set(by_index), f"{label} {field} sample set incomplete")
    for index in expected_indices:
        value = by_index.get(index)
        report.check(
            json_float_value(value),
            f"{label} {field} sample {index} is not finite",
        )


def sample_indices(elements: int) -> list[int]:
    raw = [0, 1, elements // 2, elements - 2 if elements > 1 else 0, elements - 1]
    out: list[int] = []
    for index in raw:
        if 0 <= index < elements and index not in out:
            out.append(index)
    return out


def sample_map(samples: Any) -> dict[int, int | float | str]:
    if not isinstance(samples, list):
        return {}
    out: dict[int, int | float | str] = {}
    for entry in samples:
        if isinstance(entry, dict) and isinstance(entry.get("index"), int):
            value = entry.get("value")
            if json_float_value(value):
                out[entry["index"]] = value
    return out


def samples_match(left: Any, right: Any) -> bool:
    left_map = sample_map(left)
    right_map = sample_map(right)
    if set(left_map) != set(right_map):
        return False
    for index, value in left_map.items():
        other = right_map[index]
        if isinstance(value, str) or isinstance(other, str):
            if value != other:
                return False
            continue
        if not math.isclose(float(value), float(other), rel_tol=1e-6, abs_tol=1e-6):
            return False
    return True


def json_float_value(value: Any) -> bool:
    return (isinstance(value, (int, float)) and math.isfinite(value)) or value in {"nan", "inf", "-inf"}


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


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
        ("remove schema", "rust_bin", CANDIDATE_SCHEMA, "ds4.prefill_whole_short.removed"),
        ("remove C prefill call", "ds4_c", "metal_graph_prefill_layer_major", "prefill_layer_major_removed"),
        ("remove C prompt tokenization", "ds4_c", "graph_checkpoint_prompt_tokens", "checkpoint_prompt_tokens_removed"),
        ("remove Rust prompt render", "rust_bin", "render_chat_prompt_text", "render_prompt_text_removed"),
        ("remove Rust layer loop", "rust_bin", "for layer in 0..N_LAYER", "for layer in 0..1"),
        ("remove Rust compressor prefill", "rust_bin", ".compressor_prefill(", ".compressor_prefill_removed("),
        ("remove Rust static attention", "rust_bin", ".attention_prefill_static_mixed_heads(", ".attention_prefill_static_mixed_heads_removed("),
        ("remove Rust logits read", "rust_bin", "read_tensor_output(\"logits\"", "read_tensor_output(\"removed\""),
        ("remove report comparator", "report", "compare_prefill_whole_short.py", "compare_prefill_whole_short_removed.py"),
        ("remove roadmap item", "roadmap", "M10.6b: Rust Whole-Prefill Short Execution", "M10.6b removed"),
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

    pair_mutations = [
        ("candidate digest", mutate_candidate_digest),
        ("candidate sample value", mutate_candidate_sample),
        ("candidate tensor size", mutate_candidate_tensor_size),
        ("candidate weight offset", mutate_candidate_weight_offset),
        ("candidate prompt token count", mutate_candidate_prompt_tokens),
    ]
    for label, mutate in pair_mutations:
        report = Report()
        oracle, candidate = valid_pair()
        mutate(candidate)
        validate_pair(report, oracle, candidate)
        if report.ok:
            failures.append(f"{label}: paired validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(static_mutations) + len(pair_mutations)} mutations rejected")
    return 0


def valid_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    oracle = valid_common(ORACLE_SCHEMA)
    oracle["source"] = "current-c"
    candidate = valid_common(CANDIDATE_SCHEMA)
    weights = {
        key: {
            "role": role,
            "abs_offset": EXPECTED_WEIGHTS[key][0],
            "bytes": EXPECTED_WEIGHTS[key][1],
            "type": EXPECTED_WEIGHTS[key][2],
            "type_name": EXPECTED_WEIGHTS[key][3],
        }
        for key, role in WEIGHT_ROLES.items()
    }
    oracle["weights"] = copy.deepcopy(weights)
    candidate["weights"] = copy.deepcopy(weights)
    oracle["outputs"] = valid_outputs(include_sha=True)
    candidate["outputs"] = valid_outputs(include_sha=False)
    return oracle, candidate


def valid_common(schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "case": CASE,
        "model": {
            "mapped_size": MODEL_SIZE,
            "header_bytes_read": 8388608,
            "tensor_count": 1328,
            "tensor_data_offset": TENSOR_DATA_OFFSET,
            "bound_layers": 43,
        },
        "operation": {
            "name": "unit-test",
            "method": "unit-test",
            "command_batch": True,
            "synchronized": True,
            **EXPECTED_OPERATION,
            **EXPECTED_FLOAT_OPERATION,
        },
    }


def valid_outputs(include_sha: bool) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for field, elements in OUTPUTS.items():
        entry = {
            "field": field,
            "bytes": elements * 4,
            "elements": elements,
            "nonzero_elements": min(elements, 7),
            "fnv1a64": EXPECTED_FNV1A64[field],
            "samples": [{"index": idx, "value": float(idx + 1)} for idx in sample_indices(elements)],
        }
        if include_sha:
            entry["sha256"] = "0" * 64
        outputs[field] = entry
    return outputs


def mutate_candidate_digest(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["logits"]["fnv1a64"] = "f" * 16


def mutate_candidate_sample(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["output_norm"]["samples"][0]["value"] += 1.0


def mutate_candidate_tensor_size(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["after_layer42_hc"]["elements"] -= 1


def mutate_candidate_weight_offset(candidate: dict[str, Any]) -> None:
    candidate["weights"]["output"]["abs_offset"] += 32


def mutate_candidate_prompt_tokens(candidate: dict[str, Any]) -> None:
    candidate["operation"]["prompt_tokens"] += 1


def print_errors(errors: list[str]) -> None:
    print("Rust whole-prefill short current-C oracle comparator failed:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
