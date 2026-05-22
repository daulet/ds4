#!/usr/bin/env python3
"""Compare C and Rust DS4 tensor binding/layout validation on synthetic GGUFs."""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
N_LAYER = 43
N_EMBD = 4096
N_VOCAB = 129280
N_HEAD = 64
N_HEAD_DIM = 512
N_OUT_GROUP = 8
N_LORA_Q = 1024
N_LORA_O = 1024
N_EXPERT = 256
N_EXPERT_USED = 6
N_FF_EXP = 2048
N_HASH_LAYER = 3
N_INDEXER_HEAD = 64
N_INDEXER_HEAD_DIM = 128
N_HC = 4
T_F32 = 0
T_F16 = 1
T_Q8_0 = 8
T_Q2_K = 10
T_Q4_K = 12
T_IQ2_XXS = 16
T_I32 = 26


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


Tensor = tuple[str, list[int], int]
Mutation = Callable[[list[Tensor], list[Tensor]], None]


def run_command(args: list[str], *, stdout: Path | None = None) -> subprocess.CompletedProcess[str]:
    if stdout is None:
        return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    with stdout.open("w") as f:
        return subprocess.run(args, cwd=ROOT, text=True, stdout=f, stderr=subprocess.PIPE)


def write_string(buf: bytearray, value: str) -> None:
    data = value.encode()
    buf.extend(struct.pack("<Q", len(data)))
    buf.extend(data)


def write_metadata_string(buf: bytearray, key: str, value: str) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 8))
    write_string(buf, value)


def write_metadata_u32(buf: bytearray, key: str, value: int) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 4))
    buf.extend(struct.pack("<I", value))


def write_metadata_u64(buf: bytearray, key: str, value: int) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 10))
    buf.extend(struct.pack("<Q", value))


def write_metadata_f32(buf: bytearray, key: str, value: float) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 6))
    buf.extend(struct.pack("<f", value))


def write_metadata_bool(buf: bytearray, key: str, value: bool) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 7))
    buf.extend(struct.pack("<B", 1 if value else 0))


def write_metadata_u32_array(buf: bytearray, key: str, values: list[int]) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 9))
    buf.extend(struct.pack("<I", 4))
    buf.extend(struct.pack("<Q", len(values)))
    for value in values:
        buf.extend(struct.pack("<I", value))


def write_metadata_f32_array(buf: bytearray, key: str, values: list[float]) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 9))
    buf.extend(struct.pack("<I", 6))
    buf.extend(struct.pack("<Q", len(values)))
    for value in values:
        buf.extend(struct.pack("<f", value))


def write_tensor(buf: bytearray, tensor: Tensor) -> None:
    name, dims, type_id = tensor
    write_string(buf, name)
    buf.extend(struct.pack("<I", len(dims)))
    for dim in dims:
        buf.extend(struct.pack("<Q", dim))
    buf.extend(struct.pack("<I", type_id))
    buf.extend(struct.pack("<Q", 0))


def compress_ratio(layer: int) -> int:
    if layer < 2:
        return 0
    return 4 if layer % 2 == 0 else 128


def metadata_entries() -> list[tuple[str, Any]]:
    return [
        ("string:general.name", "tensor binding fixture"),
        ("string:general.architecture", "deepseek4"),
        ("u32:general.alignment", 32),
        ("u32:deepseek4.block_count", 43),
        ("u32:deepseek4.embedding_length", N_EMBD),
        ("u32:deepseek4.vocab_size", N_VOCAB),
        ("u32:deepseek4.attention.head_count", N_HEAD),
        ("u32:deepseek4.attention.head_count_kv", 1),
        ("u32:deepseek4.attention.key_length", N_HEAD_DIM),
        ("u32:deepseek4.attention.value_length", 512),
        ("u32:deepseek4.rope.dimension_count", 64),
        ("u32:deepseek4.attention.q_lora_rank", N_LORA_Q),
        ("u32:deepseek4.attention.output_lora_rank", N_LORA_O),
        ("u32:deepseek4.attention.output_group_count", N_OUT_GROUP),
        ("u32:deepseek4.expert_count", N_EXPERT),
        ("u32:deepseek4.expert_used_count", N_EXPERT_USED),
        ("u32:deepseek4.expert_feed_forward_length", N_FF_EXP),
        ("u32:deepseek4.expert_shared_count", 1),
        ("u32:deepseek4.hash_layer_count", N_HASH_LAYER),
        ("u32:deepseek4.expert_group_count", 0),
        ("u32:deepseek4.expert_group_used_count", 0),
        ("u32:deepseek4.attention.sliding_window", 128),
        ("u32:deepseek4.attention.indexer.head_count", N_INDEXER_HEAD),
        ("u32:deepseek4.attention.indexer.key_length", N_INDEXER_HEAD_DIM),
        ("u32:deepseek4.attention.indexer.top_k", 512),
        ("u32:deepseek4.hyper_connection.count", N_HC),
        ("u32:deepseek4.hyper_connection.sinkhorn_iterations", 20),
        ("array_u32:deepseek4.attention.compress_ratios", [compress_ratio(i) for i in range(N_LAYER)]),
        ("array_f32:deepseek4.swiglu_clamp_exp", [10.0] * N_LAYER),
        ("u64:deepseek4.rope.scaling.original_context_length", 65536),
        ("f32:deepseek4.rope.freq_base", 10000.0),
        ("f32:deepseek4.rope.scaling.factor", 16.0),
        ("f32:deepseek4.rope.scaling.yarn_beta_fast", 32.0),
        ("f32:deepseek4.rope.scaling.yarn_beta_slow", 1.0),
        ("f32:deepseek4.attention.compress_rope_freq_base", 160000.0),
        ("f32:deepseek4.expert_weights_scale", 1.5),
        ("f32:deepseek4.attention.layer_norm_rms_epsilon", 0.000001),
        ("f32:deepseek4.hyper_connection.epsilon", 0.000001),
        ("bool:deepseek4.expert_weights_norm", True),
    ]


def write_metadata(buf: bytearray) -> None:
    for spec, value in metadata_entries():
        kind, key = spec.split(":", 1)
        if kind == "string":
            write_metadata_string(buf, key, value)
        elif kind == "u32":
            write_metadata_u32(buf, key, value)
        elif kind == "u64":
            write_metadata_u64(buf, key, value)
        elif kind == "f32":
            write_metadata_f32(buf, key, value)
        elif kind == "bool":
            write_metadata_bool(buf, key, value)
        elif kind == "array_u32":
            write_metadata_u32_array(buf, key, value)
        elif kind == "array_f32":
            write_metadata_f32_array(buf, key, value)
        else:
            raise ValueError(kind)


def add_layer_tensors(tensors: list[Tensor], layer: int) -> None:
    hc_dim = N_EMBD * N_HC
    hc_mix_dim = 2 * N_HC + N_HC * N_HC
    q_dim = N_HEAD * N_HEAD_DIM
    out_low_dim = N_OUT_GROUP * N_LORA_O
    ratio = compress_ratio(layer)
    p = f"blk.{layer}"
    tensors.extend(
        [
            (f"{p}.hc_attn_fn.weight", [hc_dim, hc_mix_dim], T_F16),
            (f"{p}.hc_attn_scale.weight", [3], T_F32),
            (f"{p}.hc_attn_base.weight", [hc_mix_dim], T_F32),
            (f"{p}.attn_norm.weight", [N_EMBD], T_F32),
            (f"{p}.attn_q_a.weight", [N_EMBD, N_LORA_Q], T_Q8_0),
            (f"{p}.attn_q_a_norm.weight", [N_LORA_Q], T_F32),
            (f"{p}.attn_q_b.weight", [N_LORA_Q, q_dim], T_Q8_0),
            (f"{p}.attn_kv.weight", [N_EMBD, N_HEAD_DIM], T_Q8_0),
            (f"{p}.attn_kv_a_norm.weight", [N_HEAD_DIM], T_F32),
            (f"{p}.attn_sinks.weight", [N_HEAD], T_F32),
            (f"{p}.attn_output_a.weight", [N_HEAD_DIM * (N_HEAD // N_OUT_GROUP), out_low_dim], T_Q8_0),
            (f"{p}.attn_output_b.weight", [out_low_dim, N_EMBD], T_Q8_0),
        ]
    )
    if ratio != 0:
        comp_width = (2 if ratio == 4 else 1) * N_HEAD_DIM
        tensors.extend(
            [
                (f"{p}.attn_compressor_ape.weight", [comp_width, ratio], T_F16),
                (f"{p}.attn_compressor_kv.weight", [N_EMBD, comp_width], T_F16),
                (f"{p}.attn_compressor_gate.weight", [N_EMBD, comp_width], T_F16),
                (f"{p}.attn_compressor_norm.weight", [N_HEAD_DIM], T_F32),
            ]
        )
    if ratio == 4:
        index_q_dim = N_INDEXER_HEAD * N_INDEXER_HEAD_DIM
        index_width = 2 * N_INDEXER_HEAD_DIM
        tensors.extend(
            [
                (f"{p}.indexer.attn_q_b.weight", [N_LORA_Q, index_q_dim], T_F16),
                (f"{p}.indexer.proj.weight", [N_EMBD, N_INDEXER_HEAD], T_F16),
                (f"{p}.indexer_compressor_ape.weight", [index_width, ratio], T_F16),
                (f"{p}.indexer_compressor_kv.weight", [N_EMBD, index_width], T_F16),
                (f"{p}.indexer_compressor_gate.weight", [N_EMBD, index_width], T_F16),
                (f"{p}.indexer_compressor_norm.weight", [N_INDEXER_HEAD_DIM], T_F32),
            ]
        )
    tensors.extend(
        [
            (f"{p}.hc_ffn_fn.weight", [hc_dim, hc_mix_dim], T_F16),
            (f"{p}.hc_ffn_scale.weight", [3], T_F32),
            (f"{p}.hc_ffn_base.weight", [hc_mix_dim], T_F32),
            (f"{p}.ffn_norm.weight", [N_EMBD], T_F32),
            (f"{p}.ffn_gate_inp.weight", [N_EMBD, N_EXPERT], T_F16),
            (f"{p}.ffn_gate_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], T_IQ2_XXS),
            (f"{p}.ffn_up_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], T_IQ2_XXS),
            (f"{p}.ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT], T_Q4_K),
            (f"{p}.ffn_gate_shexp.weight", [N_EMBD, N_FF_EXP], T_Q8_0),
            (f"{p}.ffn_up_shexp.weight", [N_EMBD, N_FF_EXP], T_Q8_0),
            (f"{p}.ffn_down_shexp.weight", [N_FF_EXP, N_EMBD], T_Q8_0),
        ]
    )
    if layer == 0:
        tensors.append((f"{p}.exp_probs_b.bias", [N_EXPERT], T_F32))
    if layer < N_HASH_LAYER:
        tensors.append((f"{p}.ffn_gate_tid2eid.weight", [N_EXPERT_USED, N_VOCAB], T_I32))


def base_tensors() -> list[Tensor]:
    hc_dim = N_EMBD * N_HC
    tensors = [
        ("token_embd.weight", [N_EMBD, N_VOCAB], T_F16),
        ("output_hc_base.weight", [N_HC], T_F32),
        ("output_hc_fn.weight", [hc_dim, N_HC], T_F16),
        ("output_hc_scale.weight", [1], T_F32),
        ("output_norm.weight", [N_EMBD], T_F32),
        ("output.weight", [N_EMBD, N_VOCAB], T_Q8_0),
    ]
    for layer in range(N_LAYER):
        add_layer_tensors(tensors, layer)
    return tensors


def mtp_tensors() -> list[Tensor]:
    hc_dim = N_EMBD * N_HC
    hc_mix_dim = 2 * N_HC + N_HC * N_HC
    q_dim = N_HEAD * N_HEAD_DIM
    out_low_dim = N_OUT_GROUP * N_LORA_O
    tensors = [
        ("mtp.0.hc_head_base.weight", [N_HC], T_F32),
        ("mtp.0.hc_head_fn.weight", [hc_dim, N_HC], T_F32),
        ("mtp.0.hc_head_scale.weight", [1], T_F32),
        ("mtp.0.e_proj.weight", [N_EMBD, N_EMBD], T_Q8_0),
        ("mtp.0.h_proj.weight", [N_EMBD, N_EMBD], T_Q8_0),
        ("mtp.0.enorm.weight", [N_EMBD], T_F32),
        ("mtp.0.hnorm.weight", [N_EMBD], T_F32),
        ("mtp.0.norm.weight", [N_EMBD], T_F32),
        ("mtp.0.hc_attn_fn.weight", [hc_dim, hc_mix_dim], T_F32),
        ("mtp.0.hc_attn_scale.weight", [3], T_F32),
        ("mtp.0.hc_attn_base.weight", [hc_mix_dim], T_F32),
        ("mtp.0.attn_norm.weight", [N_EMBD], T_F32),
        ("mtp.0.attn_q_a.weight", [N_EMBD, N_LORA_Q], T_Q8_0),
        ("mtp.0.attn_q_a_norm.weight", [N_LORA_Q], T_F32),
        ("mtp.0.attn_q_b.weight", [N_LORA_Q, q_dim], T_Q8_0),
        ("mtp.0.attn_kv.weight", [N_EMBD, N_HEAD_DIM], T_Q8_0),
        ("mtp.0.attn_kv_a_norm.weight", [N_HEAD_DIM], T_F32),
        ("mtp.0.attn_sinks.weight", [N_HEAD], T_F32),
        ("mtp.0.attn_output_a.weight", [N_HEAD_DIM * (N_HEAD // N_OUT_GROUP), out_low_dim], T_Q8_0),
        ("mtp.0.attn_output_b.weight", [out_low_dim, N_EMBD], T_Q8_0),
        ("mtp.0.hc_ffn_fn.weight", [hc_dim, hc_mix_dim], T_F32),
        ("mtp.0.hc_ffn_scale.weight", [3], T_F32),
        ("mtp.0.hc_ffn_base.weight", [hc_mix_dim], T_F32),
        ("mtp.0.ffn_norm.weight", [N_EMBD], T_F32),
        ("mtp.0.ffn_gate_inp.weight", [N_EMBD, N_EXPERT], T_F32),
        ("mtp.0.exp_probs_b.bias", [N_EXPERT], T_F32),
        ("mtp.0.ffn_gate_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], T_IQ2_XXS),
        ("mtp.0.ffn_up_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], T_IQ2_XXS),
        ("mtp.0.ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT], T_Q4_K),
        ("mtp.0.ffn_gate_shexp.weight", [N_EMBD, N_FF_EXP], T_Q8_0),
        ("mtp.0.ffn_up_shexp.weight", [N_EMBD, N_FF_EXP], T_Q8_0),
        ("mtp.0.ffn_down_shexp.weight", [N_FF_EXP, N_EMBD], T_Q8_0),
    ]
    return tensors


def write_gguf(path: Path, tensors: list[Tensor], include_metadata: bool) -> None:
    buf = bytearray()
    entries = metadata_entries() if include_metadata else []
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", len(tensors)))
    buf.extend(struct.pack("<Q", len(entries)))
    if include_metadata:
        write_metadata(buf)
    for tensor in tensors:
        write_tensor(buf, tensor)
    while len(buf) % 32 != 0:
        buf.append(0)
    path.write_bytes(buf)


def replace_tensor(tensors: list[Tensor], name: str, dims: list[int] | None = None, type_id: int | None = None) -> None:
    for idx, tensor in enumerate(tensors):
        if tensor[0] == name:
            tensors[idx] = (name, dims if dims is not None else tensor[1], type_id if type_id is not None else tensor[2])
            return
    raise KeyError(name)


def remove_tensor(tensors: list[Tensor], name: str) -> None:
    tensors[:] = [tensor for tensor in tensors if tensor[0] != name]


def run_c(base: Path, mtp: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            str(ROOT / "ds4-metadata-dump"),
            "--validate-layout-only",
            "-m",
            str(base),
            "--mtp",
            str(mtp),
            "-o",
            str(output),
        ]
    )


def run_rust(base: Path, mtp: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            "--validate-ds4-layout",
            "--mtp",
            str(mtp),
            str(base),
        ],
        stdout=output,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def normalize_dump(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {
            key: obj["model"][key]
            for key in (
                "size",
                "gguf_version",
                "metadata_count",
                "tensor_count",
                "alignment",
                "tensor_data_offset",
            )
        },
        "validation": obj["validation"],
        "selected_metadata": obj["selected_metadata"],
        "tensor_types": obj["tensor_types"],
        "tensors": obj["tensors"],
        "bound_tensors": obj["bound_tensors"],
    }


def normalize_error(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    text = lines[-1] if lines else ""
    text = text.removeprefix("Error: ")
    if "required tensor is missing:" in text:
        return "missing:" + text.rsplit(":", 1)[1].strip()
    match = re.search(r"tensor (.+) has type (.+), expected (.+)$", text)
    if match:
        return "type:" + match.group(1) + ":" + match.group(3)
    match = re.search(r"tensor (.+) has (\d+) dimensions, expected (\d+)", text)
    if match:
        return "ndim:" + match.group(1) + ":" + match.group(3)
    match = re.search(r"tensor (.+) has dim\[(\d+)\]=(\d+), expected (\d+)", text)
    if match:
        return "dim:" + match.group(1) + ":" + match.group(2) + ":" + match.group(4)
    return text


def compare_pass(report: Report, work: Path, name: str, mutation: Mutation | None = None) -> None:
    base = base_tensors()
    mtp = mtp_tensors()
    if mutation:
        mutation(base, mtp)
    base_path = work / f"{name}-base.gguf"
    mtp_path = work / f"{name}-mtp.gguf"
    c_path = work / f"{name}-c.json"
    rust_path = work / f"{name}-rust.json"
    write_gguf(base_path, base, include_metadata=True)
    write_gguf(mtp_path, mtp, include_metadata=False)
    c_result = run_c(base_path, mtp_path, c_path)
    rust_result = run_rust(base_path, mtp_path, rust_path)
    report.check(c_result.returncode == 0, f"C rejected {name}: {c_result.stderr.strip()}")
    report.check(rust_result.returncode == 0, f"Rust rejected {name}: {rust_result.stderr.strip()}")
    if c_result.returncode == 0 and rust_result.returncode == 0:
        report.check(normalize_dump(load_json(c_path)) == normalize_dump(load_json(rust_path)), f"{name} dumps differ")


def compare_fail(report: Report, work: Path, name: str, mutation: Mutation) -> None:
    base = base_tensors()
    mtp = mtp_tensors()
    mutation(base, mtp)
    base_path = work / f"{name}-base.gguf"
    mtp_path = work / f"{name}-mtp.gguf"
    write_gguf(base_path, base, include_metadata=True)
    write_gguf(mtp_path, mtp, include_metadata=False)
    c_result = run_c(base_path, mtp_path, work / f"{name}-c.json")
    rust_result = run_rust(base_path, mtp_path, work / f"{name}-rust.json")
    report.check(c_result.returncode != 0, f"C accepted {name}")
    report.check(rust_result.returncode != 0, f"Rust accepted {name}")
    if c_result.returncode != 0 and rust_result.returncode != 0:
        c_error = normalize_error(c_result.stderr)
        rust_error = normalize_error(rust_result.stderr)
        report.check(c_error == rust_error, f"{name} mismatch: C={c_error} Rust={rust_error}")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"tensor binding comparison: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    with tempfile.TemporaryDirectory(prefix="ds4-tensor-bindings-") as tmp:
        work = Path(tmp)
        compare_pass(report, work, "baseline")
        if args.negative_test:
            compare_fail(report, work, "missing-required-base", lambda base, mtp: remove_tensor(base, "token_embd.weight"))
            compare_fail(report, work, "wrong-type-base", lambda base, mtp: replace_tensor(base, "output.weight", type_id=T_F16))
            compare_fail(report, work, "wrong-dim-base", lambda base, mtp: replace_tensor(base, "blk.0.attn_kv.weight", dims=[N_EMBD, N_HEAD_DIM + 1]))
            compare_fail(report, work, "wrong-optional-type", lambda base, mtp: replace_tensor(base, "blk.0.exp_probs_b.bias", type_id=T_Q8_0))
            compare_fail(report, work, "wrong-routed-type", lambda base, mtp: replace_tensor(base, "blk.0.ffn_gate_exps.weight", type_id=T_Q8_0))
            compare_fail(report, work, "routed-type-mismatch", lambda base, mtp: replace_tensor(base, "blk.0.ffn_up_exps.weight", type_id=T_Q2_K))
            compare_fail(report, work, "missing-compressor", lambda base, mtp: remove_tensor(base, "blk.2.attn_compressor_norm.weight"))
            compare_fail(report, work, "missing-indexer", lambda base, mtp: remove_tensor(base, "blk.2.indexer.proj.weight"))
            compare_fail(report, work, "wrong-plain-mtp-type", lambda base, mtp: replace_tensor(mtp, "mtp.0.hc_head_fn.weight", type_id=T_Q8_0))
            compare_fail(report, work, "missing-required-mtp", lambda base, mtp: remove_tensor(mtp, "mtp.0.exp_probs_b.bias"))

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
