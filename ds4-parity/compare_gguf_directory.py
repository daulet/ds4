#!/usr/bin/env python3
"""Compare C and Rust GGUF directory dumps on synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def write_metadata_u16(buf: bytearray, key: str, value: int) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 2))
    buf.extend(struct.pack("<H", value))


def write_metadata_u64(buf: bytearray, key: str, value: int) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 10))
    buf.extend(struct.pack("<Q", value))


def write_metadata_f32(buf: bytearray, key: str, value: float) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 6))
    buf.extend(struct.pack("<f", value))


def write_metadata_f64(buf: bytearray, key: str, value: float) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 12))
    buf.extend(struct.pack("<d", value))


def write_metadata_u32_array(buf: bytearray, key: str, values: list[int]) -> None:
    write_string(buf, key)
    buf.extend(struct.pack("<I", 9))
    buf.extend(struct.pack("<I", 4))
    buf.extend(struct.pack("<Q", len(values)))
    for value in values:
        buf.extend(struct.pack("<I", value))


def write_tensor(buf: bytearray, name: str, dims: list[int], type_id: int, rel_offset: int) -> None:
    write_string(buf, name)
    buf.extend(struct.pack("<I", len(dims)))
    for dim in dims:
        buf.extend(struct.pack("<Q", dim))
    buf.extend(struct.pack("<I", type_id))
    buf.extend(struct.pack("<Q", rel_offset))


def make_fixture(path: Path) -> None:
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 2))
    buf.extend(struct.pack("<Q", 11))

    write_metadata_string(buf, "general.name", "directory fixture")
    write_metadata_string(buf, "general.architecture", "deepseek4")
    write_metadata_u32(buf, "general.alignment", 48)
    write_metadata_u32(buf, "deepseek4.context_length", 1024)
    write_metadata_u32(buf, "deepseek4.block_count", 2)
    write_metadata_u16(buf, "deepseek4.vocab_size", 32000)
    write_metadata_f32(buf, "deepseek4.rope.freq_base", 1000000.0)
    write_metadata_f64(buf, "deepseek4.rope.scaling.factor", 40.0)
    write_metadata_u64(buf, "deepseek4.rope.scaling.original_context_length", 65536)
    write_metadata_f32(buf, "deepseek4.hyper_connection.epsilon", 0.000001)
    write_metadata_u32_array(buf, "deepseek4.attention.compress_ratios", [0, 4])

    write_tensor(buf, "tok.weight", [4], 0, 0)
    write_tensor(buf, "quant.weight", [33], 8, 16)

    while len(buf) % 48 != 0:
        buf.append(0)
    buf.extend(bytes(84))
    path.write_bytes(buf)


def make_truncated_metadata_fixture(path: Path) -> None:
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", 1))
    write_string(buf, "general.name")
    buf.extend(struct.pack("<I", 8))
    path.write_bytes(buf)


def make_offset_overflow_fixture(path: Path) -> None:
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 1))
    buf.extend(struct.pack("<Q", 0))
    write_tensor(buf, "overflow.weight", [4], 0, (1 << 64) - 1)
    path.write_bytes(buf)


def make_unknown_type_fixture(path: Path) -> None:
    buf = bytearray()
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 1))
    buf.extend(struct.pack("<Q", 0))
    write_tensor(buf, "unknown.weight", [4], 31, 0)
    path.write_bytes(buf)


def load_json(path: Path) -> Any:
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
        "selected_metadata": obj["selected_metadata"],
        "tensor_types": obj["tensor_types"],
        "tensors": obj["tensors"],
        "bound_tensors": obj["bound_tensors"],
    }


def compare_fixture(report: Report, work: Path) -> None:
    fixture = work / "fixture.gguf"
    c_dump = work / "c.json"
    rust_dump = work / "rust.json"
    make_fixture(fixture)

    c_result = run_command(
        [
            str(ROOT / "ds4-metadata-dump"),
            "--directory-only",
            "-m",
            str(fixture),
            "-o",
            str(c_dump),
        ]
    )
    report.check(c_result.returncode == 0, f"C dump failed: {c_result.stderr.strip()}")

    rust_result = run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            str(fixture),
        ],
        stdout=rust_dump,
    )
    report.check(rust_result.returncode == 0, f"Rust dump failed: {rust_result.stderr.strip()}")
    if c_result.returncode != 0 or rust_result.returncode != 0:
        return

    c_obj = normalize_dump(load_json(c_dump))
    rust_obj = normalize_dump(load_json(rust_dump))
    report.check(c_obj == rust_obj, "C and Rust directory dumps differ")


def compare_unknown_tensor_type(report: Report, work: Path) -> None:
    fixture = work / "unknown-type.gguf"
    c_dump = work / "unknown-c.json"
    rust_dump = work / "unknown-rust.json"
    make_unknown_type_fixture(fixture)

    c_result = run_command(
        [
            str(ROOT / "ds4-metadata-dump"),
            "--directory-only",
            "-m",
            str(fixture),
            "-o",
            str(c_dump),
        ]
    )
    report.check(c_result.returncode == 0, f"C rejected unknown tensor type: {c_result.stderr.strip()}")

    rust_result = run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            str(fixture),
        ],
        stdout=rust_dump,
    )
    report.check(rust_result.returncode == 0, f"Rust rejected unknown tensor type: {rust_result.stderr.strip()}")
    if c_result.returncode != 0 or rust_result.returncode != 0:
        return

    c_obj = normalize_dump(load_json(c_dump))
    rust_obj = normalize_dump(load_json(rust_dump))
    report.check(c_obj == rust_obj, "unknown tensor type dumps differ")


def compare_negative(report: Report, work: Path) -> None:
    fixture = work / "negative.gguf"
    make_fixture(fixture)
    data = bytearray(fixture.read_bytes())
    data[0] = 0
    fixture.write_bytes(data)

    c_result = run_command(
        [str(ROOT / "ds4-metadata-dump"), "--directory-only", "-m", str(fixture)]
    )
    rust_result = run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            str(fixture),
        ]
    )
    report.check(c_result.returncode != 0, "C accepted invalid magic")
    report.check(rust_result.returncode != 0, "Rust accepted invalid magic")

    make_truncated_metadata_fixture(fixture)
    c_result = run_command(
        [str(ROOT / "ds4-metadata-dump"), "--directory-only", "-m", str(fixture)]
    )
    rust_result = run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            str(fixture),
        ]
    )
    report.check(c_result.returncode != 0, "C accepted truncated metadata")
    report.check(rust_result.returncode != 0, "Rust accepted truncated metadata")

    make_fixture(fixture)
    data = fixture.read_bytes()[:-8]
    fixture.write_bytes(data)
    c_result = run_command(
        [str(ROOT / "ds4-metadata-dump"), "--directory-only", "-m", str(fixture)]
    )
    rust_result = run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            str(fixture),
        ]
    )
    report.check(c_result.returncode != 0, "C accepted out-of-file tensor data")
    report.check(rust_result.returncode != 0, "Rust accepted out-of-file tensor data")

    make_offset_overflow_fixture(fixture)
    c_result = run_command(
        [str(ROOT / "ds4-metadata-dump"), "--directory-only", "-m", str(fixture)]
    )
    rust_result = run_command(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-gguf-dump",
            "--quiet",
            "--",
            str(fixture),
        ]
    )
    report.check(c_result.returncode != 0, "C accepted tensor offset overflow")
    report.check(rust_result.returncode != 0, "Rust accepted tensor offset overflow")


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = Report()
    with tempfile.TemporaryDirectory(prefix="ds4-gguf-directory-") as tmp:
        work = Path(tmp)
        compare_fixture(report, work)
        compare_unknown_tensor_type(report, work)
        if args.negative_test:
            compare_negative(report, work)

    print_report("gguf directory comparison", report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
