#!/usr/bin/env python3
"""Compare Rust layer-0 FFN-output readback against the current-C GPU oracle."""

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
    "c_helper": ROOT / "ds4_layer0_ffn_output_oracle_dump.c",
    "ds4_c": ROOT / "ds4.c",
    "ds4_h": ROOT / "ds4.h",
    "makefile": ROOT / "Makefile",
    "rust_bin": ROOT / "rust/ds4-gpu/src/bin/ds4-decode-layer0-ffn-output.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

ORACLE_SCHEMA = "ds4.layer0_ffn_output_oracle.v1"
CANDIDATE_SCHEMA = "ds4.decode_layer0_ffn_output.v1"
CASE = "token0_layer0_ffn_output"
MODEL_SIZE = 86720111488
TENSOR_DATA_OFFSET = 5333824
OUTPUTS = {
    "after_attn_hc": (16384, "f32"),
    "ffn_cur": (4096, "f32"),
    "ffn_norm": (4096, "f32"),
    "router_logits": (256, "f32"),
    "router_probs": (256, "f32"),
    "router_selected": (6, "i32"),
    "router_weights": (6, "f32"),
    "routed_mid": (12288, "f32"),
    "routed_out": (4096, "f32"),
    "shared_mid": (2048, "f32"),
    "shared_out": (4096, "f32"),
    "after_ffn_hc": (16384, "f32"),
}
WEIGHT_ROLES = {
    "token_embd": "base.token_embd",
    "hc_attn_fn": "base.layer.0.hc_attn_fn",
    "hc_attn_scale": "base.layer.0.hc_attn_scale",
    "hc_attn_base": "base.layer.0.hc_attn_base",
    "attn_norm": "base.layer.0.attn_norm",
    "attn_q_a": "base.layer.0.attn_q_a",
    "attn_q_a_norm": "base.layer.0.attn_q_a_norm",
    "attn_q_b": "base.layer.0.attn_q_b",
    "attn_kv": "base.layer.0.attn_kv",
    "attn_kv_a_norm": "base.layer.0.attn_kv_a_norm",
    "attn_sinks": "base.layer.0.attn_sinks",
    "attn_output_a": "base.layer.0.attn_output_a",
    "attn_output_b": "base.layer.0.attn_output_b",
    "hc_ffn_fn": "base.layer.0.hc_ffn_fn",
    "hc_ffn_scale": "base.layer.0.hc_ffn_scale",
    "hc_ffn_base": "base.layer.0.hc_ffn_base",
    "ffn_norm": "base.layer.0.ffn_norm",
    "ffn_gate_inp": "base.layer.0.ffn_gate_inp",
    "ffn_gate_exps": "base.layer.0.ffn_gate_exps",
    "ffn_up_exps": "base.layer.0.ffn_up_exps",
    "ffn_down_exps": "base.layer.0.ffn_down_exps",
    "ffn_gate_shexp": "base.layer.0.ffn_gate_shexp",
    "ffn_up_shexp": "base.layer.0.ffn_up_shexp",
    "ffn_down_shexp": "base.layer.0.ffn_down_shexp",
}
OPTIONAL_WEIGHT_ROLES = {
    "ffn_exp_probs_b": "base.layer.0.ffn_exp_probs_b",
}
EXPECTED_FNV1A64 = {
    "after_attn_hc": "ad09657ac6584898",
    "ffn_cur": "6a4fadf124b872b9",
    "ffn_norm": "51f4215200d2855c",
    "router_logits": "ea0d089c828257f3",
    "router_probs": "8435f2b23e429e02",
    "router_selected": "6028192a0e6c3c3e",
    "router_weights": "0a7ff588f5caa574",
    "routed_mid": "a51a0c8b6f39b89a",
    "routed_out": "507a5d29b2e806e9",
    "shared_mid": "8fb3b60df337c136",
    "shared_out": "3f90851fbe0be24c",
    "after_ffn_hc": "3d49316c93ce351f",
}
EXPECTED_WEIGHTS = {
    "token_embd": (77928033088, 1059061760, 1, "f16"),
    "hc_attn_fn": (79129609376, 786432, 1, "f16"),
    "hc_attn_scale": (79130395808, 12, 0, "f32"),
    "hc_attn_base": (79129609280, 96, 0, "f32"),
    "attn_norm": (79100740672, 16384, 0, "f32"),
    "attn_q_a": (79060632640, 4456448, 8, "q8_0"),
    "attn_q_a_norm": (78987097152, 4096, 0, "f32"),
    "attn_q_b": (79065089088, 35651584, 8, "q8_0"),
    "attn_kv": (78987101248, 2228224, 8, "q8_0"),
    "attn_kv_a_norm": (78987095104, 2048, 0, "f32"),
    "attn_sinks": (78987094848, 256, 0, "f32"),
    "attn_output_a": (78989329472, 35651584, 8, "q8_0"),
    "attn_output_b": (79024981056, 35651584, 8, "q8_0"),
    "hc_ffn_fn": (79130395936, 786432, 1, "f16"),
    "hc_ffn_scale": (79131182368, 12, 0, "f32"),
    "hc_ffn_base": (79130395840, 96, 0, "f32"),
    "ffn_norm": (79129592896, 16384, 0, "f32"),
    "ffn_gate_inp": (79100757056, 2097152, 1, "f16"),
    "ffn_gate_exps": (14641984, 553648128, 16, "iq2_xxs"),
    "ffn_up_exps": (1272933184, 553648128, 16, "iq2_xxs"),
    "ffn_down_exps": (568290112, 704643072, 10, "q2_k"),
    "ffn_gate_shexp": (79102854208, 8912896, 8, "q8_0"),
    "ffn_up_shexp": (79120680000, 8912896, 8, "q8_0"),
    "ffn_down_shexp": (79111767104, 8912896, 8, "q8_0"),
}
EXPECTED_OPERATION = {
    "gate_row_bytes": 1056,
    "gate_expert_bytes": 2162688,
    "down_row_bytes": 672,
    "down_expert_bytes": 2752512,
    "router_has_bias": False,
    "router_hash_mode": True,
    "router_hash_rows": 129280,
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
        print(f"Rust layer-0 FFN-output current-C oracle comparator: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, help="JSON from ds4-layer0-ffn-output-oracle-dump")
    parser.add_argument("--candidate", type=Path, help="JSON from ds4-decode-layer0-ffn-output")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    c_helper = texts["c_helper"]
    for needle, message in (
        ("ds4_dump_layer0_ffn_output_oracle_json", "C helper does not call exported oracle dump"),
        ("--token", "C helper token flag missing"),
        ("--model", "C helper model flag missing"),
        ("--output", "C helper output flag missing"),
    ):
        report.check(needle in c_helper, message)

    ds4_c = texts["ds4_c"]
    for needle, message in (
        ("ds4_dump_layer0_ffn_output_oracle_json", "C oracle dump implementation missing"),
        ("ds4_gpu_set_model_fd(model.fd)", "C oracle does not bridge the model fd"),
        ("ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_tensor", "C oracle does not build attention HC-pre prefix"),
        ("ds4_gpu_attention_output_low_q8_tensor(attn_low_tensor", "C oracle does not compute attention output prefix"),
        ("ds4_gpu_hc_split_weighted_sum_norm_tensor(ffn_cur_tensor", "C oracle does not build FFN HC-pre output"),
        ("ds4_gpu_router_select_tensor(router_selected_tensor", "C oracle does not run router selection"),
        ("ds4_gpu_routed_moe_one_tensor(routed_out_tensor", "C oracle does not run routed MoE"),
        ("ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate_tensor", "C oracle does not run shared SwiGLU"),
        ("ds4_gpu_shared_down_hc_expand_q8_0_tensor(after_ffn_hc_tensor", "C oracle does not run shared down HC expansion"),
        ("layer0_attn_oracle_write_i32_output(fp, \"router_selected\"", "C oracle does not emit router_selected as i32"),
        ("routed_mid", "C oracle routed_mid output missing"),
        (ORACLE_SCHEMA, "C oracle schema missing"),
    ):
        report.check(needle in ds4_c, message)

    report.check(
        "int ds4_dump_layer0_ffn_output_oracle_json" in texts["ds4_h"],
        "header declaration for layer-0 FFN-output oracle missing",
    )

    makefile = texts["makefile"]
    report.check("ds4-layer0-ffn-output-oracle-dump:" in makefile, "Makefile helper target missing")
    report.check("ds4_layer0_ffn_output_oracle_dump_cpu.o" in makefile, "CPU helper object missing")

    rust_bin = texts["rust_bin"]
    for needle, message in (
        (CANDIDATE_SCHEMA, "Rust candidate schema missing"),
        (CASE, "Rust candidate case missing"),
        ("GraphPlan::for_context(CTX_SIZE, CTX_SIZE, false)", "graph plan raw-cap setup missing"),
        ("bind_ds4_weights", "DS4 weight binding missing"),
        ("set_model_fd", "model fd bridge missing"),
        ("set_model_map_range", "model map range bridge missing"),
        ("CommandBatch::begin", "command batch begin missing"),
        (".attention_output_low_q8(", "attention-output prefix facade call missing"),
        (".matmul_q8_0_hc_expand(", "attention-output HC expand facade call missing"),
        (".hc_split_weighted_sum_norm(", "FFN HC-pre fused facade call missing"),
        (".router_select(", "router-select facade call missing"),
        (".routed_moe_one(", "routed MoE facade call missing"),
        (".shared_gate_up_swiglu_q8_0(", "shared SwiGLU facade call missing"),
        (".shared_down_hc_expand_q8_0(", "shared down HC expand facade call missing"),
        ("read_tensor_i32_output(", "router_selected i32 readback missing"),
        ("tensor_nbytes(", "expert row-byte calculation missing"),
        ("Output::I32", "mixed f32/i32 output writer missing"),
        ("fnv1a64(&data)", "Rust full-buffer FNV digest missing"),
        ("BackendGuard", "cleanup guard missing"),
    ):
        report.check(needle in rust_bin, message)

    report_text = texts["report"]
    report.check(
        "M10.5c4c2b2b2b2b2a Rust layer-0 FFN-output comparator" in report_text,
        "unified report comparator missing",
    )
    report.check(
        "M10.5c4c2b2b2b2b2a B300 layer-0 FFN-output oracle rerun" in report_text,
        "unified report B300 skip missing",
    )
    report.check(
        "ds4-layer0-ffn-output-oracle-dump" in report_text
        and "compare_decode_layer0_ffn_output.py" in report_text,
        "B300 oracle rerun command missing helper/comparator",
    )

    report.check("M10.5c4c2b2b2b2b2a Rust layer-0 FFN-output" in texts["readme"], "README entry missing")
    report.check(
        "M10.5c4c2b2b2b2b2a: Rust Layer-0 FFN Output B300 Execution" in texts["roadmap"],
        "roadmap FFN-output split missing",
    )
    report.check(
        "M10.5c4c2b2b2b2b2b: Rust One-Token Decode B300 Execution" in texts["roadmap"],
        "roadmap remainder split missing",
    )
    report.check(
        "M10.5c4c2b2b2b2b2a: Rust Layer-0 FFN Output B300 Execution" in texts["todo"],
        "TODO FFN-output split missing",
    )
    report.check(
        "M10.5c4c2b2b2b2b2b: Rust One-Token Decode B300 Execution" in texts["todo"],
        "TODO remainder split missing",
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
        expected = {
            "token": 0,
            "layer": 0,
            "position": 0,
            "ctx_size": 32768,
            "prefill_cap": 2048,
            "raw_cap": 2304,
            "raw_window": 128,
            "raw_row": 0,
            "raw_start": 0,
            "n_raw": 1,
            "n_comp": 0,
            "use_mask": 0,
            "n_vocab": 129280,
            "n_embd": 4096,
            "n_hc": 4,
            "q_rank": 1024,
            "q_dim": 32768,
            "head_dim": 512,
            "n_head": 64,
            "n_head_kv": 1,
            "n_rot": 64,
            "n_groups": 8,
            "group_heads": 8,
            "group_dim": 4096,
            "rank": 1024,
            "low_dim": 8192,
            "hc_mix_dim": 24,
            "shared_dim": 2048,
            "expert_in_dim": 4096,
            "expert_mid_dim": 2048,
            "down_in_dim": 2048,
            "routed_out_dim": 4096,
            "n_expert": 256,
            "n_expert_used": 6,
            "router_n_expert_groups": 0,
            "router_n_group_used": 0,
        }
        for key, value in expected.items():
            report.check(operation.get(key) == value, f"{label} operation {key} drift")
        for key, value in EXPECTED_OPERATION.items():
            report.check(operation.get(key) == value, f"{label} operation {key} drift")
        for key, value in {
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "rope_ext_factor": 0.0,
            "rope_attn_factor": 1.0,
            "rope_yarn_beta_fast": 32.0,
            "rope_yarn_beta_slow": 1.0,
            "swiglu_clamp_exp": 10.0,
            "rms_eps": 1e-6,
            "hc_eps": 1e-6,
        }.items():
            report.check(abs(float(operation.get(key, 0.0)) - value) <= 1e-12, f"{label} operation {key} drift")

    outputs = obj.get("outputs")
    report.check(isinstance(outputs, dict), f"{label} outputs missing")
    if isinstance(outputs, dict):
        report.check(set(outputs) == set(OUTPUTS), f"{label} output tensor set drift")
        for field, (elements, dtype) in OUTPUTS.items():
            validate_output(report, outputs.get(field), label, field, elements, dtype)


def validate_output(report: Report, output: Any, label: str, field: str, elements: int, dtype: str) -> None:
    report.check(isinstance(output, dict), f"{label} {field} output missing")
    if not isinstance(output, dict):
        return
    report.check(output.get("field") == field, f"{label} {field} field drift")
    if dtype == "i32":
        report.check(output.get("dtype") == "i32", f"{label} {field} dtype drift")
    report.check(output.get("elements") == elements, f"{label} {field} element drift")
    report.check(output.get("bytes") == elements * 4, f"{label} {field} byte drift")
    report.check(isinstance(output.get("nonzero_elements"), int), f"{label} {field} nonzero count missing")
    report.check(isinstance(output.get("fnv1a64"), str), f"{label} {field} FNV digest missing")
    if label == "oracle":
        report.check(is_sha256(output.get("sha256")), f"oracle {field} sha256 invalid")
    expected = EXPECTED_FNV1A64.get(field)
    if expected is not None:
        report.check(output.get("fnv1a64") == expected, f"{label} {field} FNV digest drift")
    validate_samples(report, output.get("samples"), label, field, elements, dtype)


def validate_weights_pair(report: Report, oracle_weights: Any, candidate_weights: Any) -> None:
    report.check(isinstance(oracle_weights, dict), "oracle weights missing")
    report.check(isinstance(candidate_weights, dict), "candidate weights missing")
    if not isinstance(oracle_weights, dict) or not isinstance(candidate_weights, dict):
        return
    expected_keys = set(WEIGHT_ROLES)
    allowed_keys = expected_keys | set(OPTIONAL_WEIGHT_ROLES)
    oracle_keys = set(oracle_weights)
    candidate_keys = set(candidate_weights)
    report.check(expected_keys <= oracle_keys <= allowed_keys, "oracle weight set drift")
    report.check(expected_keys <= candidate_keys <= allowed_keys, "candidate weight set drift")
    report.check(oracle_keys == candidate_keys, "candidate weight set does not match current-C oracle")
    roles = {**WEIGHT_ROLES, **OPTIONAL_WEIGHT_ROLES}
    for key in sorted(oracle_keys | candidate_keys):
        role = roles.get(key)
        report.check(role is not None, f"{key} weight role unknown")
        oracle = oracle_weights.get(key)
        candidate = candidate_weights.get(key)
        report.check(isinstance(oracle, dict), f"oracle {key} weight missing")
        report.check(isinstance(candidate, dict), f"candidate {key} weight missing")
        if role is None or not isinstance(oracle, dict) or not isinstance(candidate, dict):
            continue
        report.check(oracle.get("role") == role, f"oracle {key} role drift")
        report.check(candidate.get("role") == role, f"candidate {key} role drift")
        for field in ("abs_offset", "bytes", "type", "type_name"):
            report.check(oracle.get(field) == candidate.get(field), f"{key} weight {field} mismatch")
        expected = EXPECTED_WEIGHTS.get(key)
        if expected is not None:
            offset, size, type_id, type_name = expected
            report.check(oracle.get("abs_offset") == offset, f"oracle {key} offset drift")
            report.check(oracle.get("bytes") == size, f"oracle {key} byte-size drift")
            report.check(oracle.get("type") == type_id, f"oracle {key} type drift")
            report.check(oracle.get("type_name") == type_name, f"oracle {key} type-name drift")


def validate_operation_pair(report: Report, oracle_operation: Any, candidate_operation: Any) -> None:
    report.check(isinstance(oracle_operation, dict), "oracle operation missing")
    report.check(isinstance(candidate_operation, dict), "candidate operation missing")
    if not isinstance(oracle_operation, dict) or not isinstance(candidate_operation, dict):
        return
    for key in (
        "gate_row_bytes",
        "gate_expert_bytes",
        "down_row_bytes",
        "down_expert_bytes",
        "router_has_bias",
        "router_hash_mode",
        "router_hash_rows",
    ):
        report.check(
            oracle_operation.get(key) == candidate_operation.get(key),
            f"candidate operation {key} does not match current-C oracle",
        )


def validate_outputs_pair(report: Report, oracle_outputs: Any, candidate_outputs: Any) -> None:
    report.check(isinstance(oracle_outputs, dict), "oracle outputs missing")
    report.check(isinstance(candidate_outputs, dict), "candidate outputs missing")
    if not isinstance(oracle_outputs, dict) or not isinstance(candidate_outputs, dict):
        return
    for field, (_elements, dtype) in OUTPUTS.items():
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
            samples_match(oracle.get("samples"), candidate.get("samples"), dtype),
            f"candidate {field} samples do not match current-C oracle",
        )


def validate_samples(report: Report, samples: Any, label: str, field: str, elements: int, dtype: str) -> None:
    report.check(isinstance(samples, list), f"{label} {field} samples missing")
    if not isinstance(samples, list):
        return
    by_index = sample_map(samples)
    expected_indices = set(sample_indices(elements))
    report.check(expected_indices <= set(by_index), f"{label} {field} sample set incomplete")
    for index in expected_indices:
        value = by_index.get(index)
        if dtype == "i32":
            report.check(isinstance(value, int), f"{label} {field} sample {index} is not i32")
        else:
            report.check(isinstance(value, (int, float)) and math.isfinite(value), f"{label} {field} sample {index} is not finite")


def sample_indices(elements: int) -> list[int]:
    raw = [0, 1, elements // 2, elements - 2 if elements > 1 else 0, elements - 1]
    out: list[int] = []
    for index in raw:
        if 0 <= index < elements and index not in out:
            out.append(index)
    return out


def sample_map(samples: Any) -> dict[int, int | float]:
    if not isinstance(samples, list):
        return {}
    out: dict[int, int | float] = {}
    for entry in samples:
        if isinstance(entry, dict) and isinstance(entry.get("index"), int):
            value = entry.get("value")
            if isinstance(value, (int, float)):
                out[entry["index"]] = value
    return out


def samples_match(left: Any, right: Any, dtype: str) -> bool:
    left_map = sample_map(left)
    right_map = sample_map(right)
    if set(left_map) != set(right_map):
        return False
    for index, left_value in left_map.items():
        right_value = right_map[index]
        if dtype == "i32":
            if left_value != right_value:
                return False
        elif abs(float(left_value) - float(right_value)) > 1e-6:
            return False
    return True


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
        ("remove schema", "rust_bin", CANDIDATE_SCHEMA, "ds4.decode_layer0_ffn_output.removed"),
        ("remove C router", "ds4_c", "ds4_gpu_router_select_tensor(router_selected_tensor", "ds4_gpu_router_select_removed(router_selected_tensor"),
        ("remove C routed MoE", "ds4_c", "ds4_gpu_routed_moe_one_tensor(routed_out_tensor", "ds4_gpu_routed_moe_one_removed(routed_out_tensor"),
        ("remove C shared SwiGLU", "ds4_c", "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate_tensor", "ds4_gpu_shared_gate_up_removed(shared_gate_tensor"),
        ("remove C shared down", "ds4_c", "ds4_gpu_shared_down_hc_expand_q8_0_tensor(after_ffn_hc_tensor", "ds4_gpu_shared_down_removed(after_ffn_hc_tensor"),
        ("remove Rust router", "rust_bin", ".router_select(", ".router_select_removed("),
        ("remove Rust routed MoE", "rust_bin", ".routed_moe_one(", ".routed_moe_one_removed("),
        ("remove Rust i32 read", "rust_bin", "read_tensor_i32_output(", "read_tensor_i32_output_removed("),
        ("remove B300 candidate check", "report", "compare_decode_layer0_ffn_output.py", "compare_decode_layer0_ffn_output_removed.py"),
        (
            "remove roadmap split",
            "roadmap",
            "M10.5c4c2b2b2b2b2a: Rust Layer-0 FFN Output B300 Execution",
            "M10.5c4c2b2b2b2b2a removed",
        ),
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
        ("candidate f32 sample value", mutate_candidate_sample),
        ("candidate i32 router selection", mutate_candidate_router_selected),
        ("candidate tensor size", mutate_candidate_tensor_size),
        ("candidate weight offset", mutate_candidate_weight_offset),
        ("candidate router flag", mutate_candidate_router_flag),
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
            "abs_offset": EXPECTED_WEIGHTS.get(key, (1000 + idx, 0, 0, ""))[0],
            "bytes": EXPECTED_WEIGHTS.get(key, (0, 2000 + idx, 0, ""))[1],
            "type": EXPECTED_WEIGHTS.get(key, (0, 0, 1 if "fn" in key else 8, ""))[2],
            "type_name": EXPECTED_WEIGHTS.get(key, (0, 0, 0, "f16" if "fn" in key else "q8_0"))[3],
        }
        for idx, (key, role) in enumerate({**WEIGHT_ROLES, **OPTIONAL_WEIGHT_ROLES}.items())
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
            "tensor_count": 1328,
            "tensor_data_offset": TENSOR_DATA_OFFSET,
            "bound_layers": 43,
        },
        "operation": {
            "token": 0,
            "layer": 0,
            "position": 0,
            "ctx_size": 32768,
            "prefill_cap": 2048,
            "raw_cap": 2304,
            "raw_window": 128,
            "raw_row": 0,
            "raw_start": 0,
            "n_raw": 1,
            "n_comp": 0,
            "use_mask": 0,
            "n_vocab": 129280,
            "n_embd": 4096,
            "n_hc": 4,
            "q_rank": 1024,
            "q_dim": 32768,
            "head_dim": 512,
            "n_head": 64,
            "n_head_kv": 1,
            "n_rot": 64,
            "n_groups": 8,
            "group_heads": 8,
            "group_dim": 4096,
            "rank": 1024,
            "low_dim": 8192,
            "hc_mix_dim": 24,
            "shared_dim": 2048,
            "expert_in_dim": 4096,
            "expert_mid_dim": 2048,
            "down_in_dim": 2048,
            "routed_out_dim": 4096,
            "n_expert": 256,
            "n_expert_used": 6,
            **EXPECTED_OPERATION,
            "router_n_expert_groups": 0,
            "router_n_group_used": 0,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
            "rope_ext_factor": 0.0,
            "rope_attn_factor": 1.0,
            "rope_yarn_beta_fast": 32.0,
            "rope_yarn_beta_slow": 1.0,
            "swiglu_clamp_exp": 10.0,
            "rms_eps": 1e-6,
            "hc_eps": 1e-6,
        },
    }


def valid_outputs(include_sha: bool) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for idx, (field, (elements, dtype)) in enumerate(OUTPUTS.items()):
        digest = EXPECTED_FNV1A64.get(field, f"{idx + 1:016x}")
        if dtype == "i32":
            samples = [{"index": sample, "value": idx + sample} for sample in sample_indices(elements)]
        else:
            samples = [{"index": sample, "value": float(idx + sample) / 10.0} for sample in sample_indices(elements)]
        output = {
            "field": field,
            "bytes": elements * 4,
            "elements": elements,
            "nonzero_elements": elements,
            "fnv1a64": digest,
            "samples": samples,
        }
        if dtype == "i32":
            output["dtype"] = "i32"
        if include_sha:
            output["sha256"] = f"{idx:064x}"
        outputs[field] = output
    return outputs


def mutate_candidate_digest(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["after_ffn_hc"]["fnv1a64"] = "0000000000000000"


def mutate_candidate_sample(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["routed_out"]["samples"][0]["value"] += 0.01


def mutate_candidate_router_selected(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["router_selected"]["samples"][0]["value"] += 1


def mutate_candidate_tensor_size(candidate: dict[str, Any]) -> None:
    candidate["outputs"]["shared_mid"]["elements"] = 1


def mutate_candidate_weight_offset(candidate: dict[str, Any]) -> None:
    candidate["weights"]["ffn_down_shexp"]["abs_offset"] += 4


def mutate_candidate_router_flag(candidate: dict[str, Any]) -> None:
    candidate["operation"]["router_has_bias"] = not candidate["operation"]["router_has_bias"]


def print_errors(errors: list[str]) -> None:
    print("Rust layer-0 FFN-output comparator failures:")
    for error in errors:
        print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
