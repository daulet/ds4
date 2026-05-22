#!/usr/bin/env python3
"""Compare C and Rust DS4 metadata validation on synthetic GGUF fixtures."""

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


Entry = tuple[str, str, Any]
Mutation = Callable[[list[Entry]], None]


def run_command(args: list[str], *, stdout: Path | None = None) -> subprocess.CompletedProcess[str]:
    if stdout is None:
        return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    with stdout.open("w") as f:
        return subprocess.run(args, cwd=ROOT, text=True, stdout=f, stderr=subprocess.PIPE)


def write_string(buf: bytearray, value: str) -> None:
    data = value.encode()
    buf.extend(struct.pack("<Q", len(data)))
    buf.extend(data)


def write_entry(buf: bytearray, entry: Entry) -> None:
    kind, key, value = entry
    write_string(buf, key)
    if kind == "string":
        buf.extend(struct.pack("<I", 8))
        write_string(buf, value)
    elif kind == "u32":
        buf.extend(struct.pack("<I", 4))
        buf.extend(struct.pack("<I", value))
    elif kind == "u64":
        buf.extend(struct.pack("<I", 10))
        buf.extend(struct.pack("<Q", value))
    elif kind == "i32":
        buf.extend(struct.pack("<I", 5))
        buf.extend(struct.pack("<i", value))
    elif kind == "f32":
        buf.extend(struct.pack("<I", 6))
        buf.extend(struct.pack("<f", value))
    elif kind == "f64":
        buf.extend(struct.pack("<I", 12))
        buf.extend(struct.pack("<d", value))
    elif kind == "bool":
        buf.extend(struct.pack("<I", 7))
        buf.extend(struct.pack("<B", 1 if value else 0))
    elif kind == "array_u32":
        buf.extend(struct.pack("<I", 9))
        buf.extend(struct.pack("<I", 4))
        buf.extend(struct.pack("<Q", len(value)))
        for item in value:
            buf.extend(struct.pack("<I", item))
    elif kind == "array_i32":
        buf.extend(struct.pack("<I", 9))
        buf.extend(struct.pack("<I", 5))
        buf.extend(struct.pack("<Q", len(value)))
        for item in value:
            buf.extend(struct.pack("<i", item))
    elif kind == "array_f32":
        buf.extend(struct.pack("<I", 9))
        buf.extend(struct.pack("<I", 6))
        buf.extend(struct.pack("<Q", len(value)))
        for item in value:
            buf.extend(struct.pack("<f", item))
    elif kind == "array_f64":
        buf.extend(struct.pack("<I", 9))
        buf.extend(struct.pack("<I", 12))
        buf.extend(struct.pack("<Q", len(value)))
        for item in value:
            buf.extend(struct.pack("<d", item))
    else:
        raise ValueError(f"unknown metadata entry kind: {kind}")


def compress_ratios() -> list[int]:
    return [0 if layer < 2 else (4 if layer % 2 == 0 else 128) for layer in range(N_LAYER)]


def base_entries() -> list[Entry]:
    return [
        ("string", "general.name", "ds4 validation fixture"),
        ("string", "general.architecture", "deepseek4"),
        ("u32", "general.alignment", 32),
        ("u32", "deepseek4.block_count", 43),
        ("u32", "deepseek4.embedding_length", 4096),
        ("u32", "deepseek4.vocab_size", 129280),
        ("u32", "deepseek4.attention.head_count", 64),
        ("u32", "deepseek4.attention.head_count_kv", 1),
        ("u32", "deepseek4.attention.key_length", 512),
        ("u32", "deepseek4.attention.value_length", 512),
        ("u32", "deepseek4.rope.dimension_count", 64),
        ("u32", "deepseek4.attention.q_lora_rank", 1024),
        ("u32", "deepseek4.attention.output_lora_rank", 1024),
        ("u32", "deepseek4.attention.output_group_count", 8),
        ("u32", "deepseek4.expert_count", 256),
        ("u32", "deepseek4.expert_used_count", 6),
        ("u32", "deepseek4.expert_feed_forward_length", 2048),
        ("u32", "deepseek4.expert_shared_count", 1),
        ("u32", "deepseek4.hash_layer_count", 3),
        ("u32", "deepseek4.expert_group_count", 0),
        ("u32", "deepseek4.expert_group_used_count", 0),
        ("u32", "deepseek4.attention.sliding_window", 128),
        ("u32", "deepseek4.attention.indexer.head_count", 64),
        ("u32", "deepseek4.attention.indexer.key_length", 128),
        ("u32", "deepseek4.attention.indexer.top_k", 512),
        ("u32", "deepseek4.hyper_connection.count", 4),
        ("u32", "deepseek4.hyper_connection.sinkhorn_iterations", 20),
        ("array_u32", "deepseek4.attention.compress_ratios", compress_ratios()),
        ("array_f32", "deepseek4.swiglu_clamp_exp", [10.0] * N_LAYER),
        ("u64", "deepseek4.rope.scaling.original_context_length", 65536),
        ("f32", "deepseek4.rope.freq_base", 10000.0),
        ("f32", "deepseek4.rope.scaling.factor", 16.0),
        ("f32", "deepseek4.rope.scaling.yarn_beta_fast", 32.0),
        ("f32", "deepseek4.rope.scaling.yarn_beta_slow", 1.0),
        ("f32", "deepseek4.attention.compress_rope_freq_base", 160000.0),
        ("f32", "deepseek4.expert_weights_scale", 1.5),
        ("f32", "deepseek4.attention.layer_norm_rms_epsilon", 0.000001),
        ("f32", "deepseek4.hyper_connection.epsilon", 0.000001),
        ("bool", "deepseek4.expert_weights_norm", True),
    ]


def make_fixture(path: Path, mutation: Mutation | None = None) -> None:
    entries = base_entries()
    if mutation:
        mutation(entries)

    buf = bytearray()
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", len(entries)))
    for entry in entries:
        write_entry(buf, entry)
    while len(buf) % 32 != 0:
        buf.append(0)
    path.write_bytes(buf)


def replace(entries: list[Entry], key: str, kind: str, value: Any) -> None:
    for idx, entry in enumerate(entries):
        if entry[1] == key:
            entries[idx] = (kind, key, value)
            return
    raise KeyError(key)


def remove(entries: list[Entry], key: str) -> None:
    entries[:] = [entry for entry in entries if entry[1] != key]


def run_c(fixture: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            str(ROOT / "ds4-metadata-dump"),
            "--validate-config-only",
            "-m",
            str(fixture),
            "-o",
            str(output),
        ]
    )


def run_rust(fixture: Path, output: Path) -> subprocess.CompletedProcess[str]:
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
            "--validate-ds4-metadata",
            str(fixture),
        ],
        stdout=output,
    )


def normalize_error(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    text = lines[-1] if lines else ""
    text = text.removeprefix("Error: ")
    if "required metadata key is missing:" in text:
        return "missing:" + text.rsplit(":", 1)[1].strip()
    if "metadata key has a non-integer type:" in text:
        return "non-integer:" + text.rsplit(":", 1)[1].strip()
    match = re.search(r"metadata key has a non-float type \d+: (.+)$", text)
    if match:
        return "non-float:" + match.group(1)
    match = re.search(r"expected ([^=]+)=", text)
    if match:
        return "expected:" + match.group(1)
    if "compress_ratios is shorter" in text:
        return "short-array:deepseek4.attention.compress_ratios"
    if "swiglu_clamp_exp is shorter" in text:
        return "short-array:deepseek4.swiglu_clamp_exp"
    if "metadata array contains a negative value" in text:
        return "negative-array"
    match = re.search(r"compression ratio at layer (\d+):", text)
    if match:
        return "compress-ratio:" + match.group(1)
    return text


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def compare_pass(report: Report, work: Path, name: str, mutation: Mutation | None = None) -> None:
    fixture = work / f"{name}.gguf"
    c_output = work / f"{name}-c.json"
    rust_output = work / f"{name}-rust.json"
    make_fixture(fixture, mutation)

    c_result = run_c(fixture, c_output)
    rust_result = run_rust(fixture, rust_output)
    report.check(c_result.returncode == 0, f"C rejected {name}: {c_result.stderr.strip()}")
    report.check(rust_result.returncode == 0, f"Rust rejected {name}: {rust_result.stderr.strip()}")
    if c_result.returncode == 0 and rust_result.returncode == 0:
        report.check(
            load_json(c_output)["validation"]["config"] == "passed",
            f"C did not mark config validation passed for {name}",
        )
        report.check(
            load_json(rust_output)["validation"]["config"] == "passed",
            f"Rust did not mark config validation passed for {name}",
        )


def compare_fail(report: Report, work: Path, name: str, mutation: Mutation) -> None:
    fixture = work / f"{name}.gguf"
    make_fixture(fixture, mutation)

    c_result = run_c(fixture, work / f"{name}-c.json")
    rust_result = run_rust(fixture, work / f"{name}-rust.json")
    report.check(c_result.returncode != 0, f"C accepted {name}")
    report.check(rust_result.returncode != 0, f"Rust accepted {name}")
    if c_result.returncode != 0 and rust_result.returncode != 0:
        c_error = normalize_error(c_result.stderr)
        rust_error = normalize_error(rust_result.stderr)
        report.check(c_error == rust_error, f"{name} mismatch: C={c_error} Rust={rust_error}")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"metadata validation comparison: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    with tempfile.TemporaryDirectory(prefix="ds4-metadata-validation-") as tmp:
        work = Path(tmp)
        compare_pass(report, work, "baseline")
        compare_pass(
            report,
            work,
            "coerced-types",
            lambda entries: (
                replace(entries, "deepseek4.rope.scaling.original_context_length", "u32", 65536),
                replace(entries, "deepseek4.rope.freq_base", "u32", 10000),
                replace(entries, "deepseek4.rope.scaling.factor", "f64", 16.0),
                replace(entries, "deepseek4.rope.scaling.yarn_beta_slow", "i32", 1),
                replace(entries, "deepseek4.attention.compress_ratios", "array_i32", compress_ratios()),
                replace(entries, "deepseek4.swiglu_clamp_exp", "array_f64", [10.0] * N_LAYER),
            ),
        )
        if args.negative_test:
            compare_fail(report, work, "missing-key", lambda entries: remove(entries, "deepseek4.vocab_size"))
            compare_fail(
                report,
                work,
                "wrong-u32-type",
                lambda entries: replace(entries, "deepseek4.attention.head_count", "string", "64"),
            )
            compare_fail(
                report,
                work,
                "wrong-u32-value",
                lambda entries: replace(entries, "deepseek4.embedding_length", "u32", 4097),
            )
            compare_fail(
                report,
                work,
                "short-compress-ratios",
                lambda entries: replace(
                    entries,
                    "deepseek4.attention.compress_ratios",
                    "array_u32",
                    compress_ratios()[:-1],
                ),
            )
            compare_fail(
                report,
                work,
                "negative-compress-ratio",
                lambda entries: replace(
                    entries,
                    "deepseek4.attention.compress_ratios",
                    "array_i32",
                    [0, 0, -4] + compress_ratios()[3:],
                ),
            )
            compare_fail(
                report,
                work,
                "wrong-compress-ratio",
                lambda entries: replace(
                    entries,
                    "deepseek4.attention.compress_ratios",
                    "array_u32",
                    [0, 0, 128] + compress_ratios()[3:],
                ),
            )
            compare_fail(
                report,
                work,
                "short-swiglu",
                lambda entries: replace(
                    entries, "deepseek4.swiglu_clamp_exp", "array_f32", [10.0] * (N_LAYER - 1)
                ),
            )
            compare_fail(
                report,
                work,
                "float-outside-tolerance",
                lambda entries: replace(entries, "deepseek4.rope.freq_base", "f32", 10000.02),
            )
            compare_fail(
                report,
                work,
                "wrong-u64-type",
                lambda entries: replace(
                    entries, "deepseek4.rope.scaling.original_context_length", "string", "65536"
                ),
            )
            compare_fail(
                report,
                work,
                "wrong-f32-type",
                lambda entries: replace(entries, "deepseek4.expert_weights_scale", "string", "1.5"),
            )
            compare_fail(
                report,
                work,
                "wrong-bool-value",
                lambda entries: replace(entries, "deepseek4.expert_weights_norm", "bool", False),
            )

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
