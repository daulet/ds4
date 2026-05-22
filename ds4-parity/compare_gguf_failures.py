#!/usr/bin/env python3
"""Compare C and Rust rejection boundaries for malformed GGUF fixtures."""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import compare_gguf_directory as directory
import compare_metadata_validation as metadata
import compare_tensor_bindings as bindings


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


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    expected: str
    build: Callable[[Path], tuple[Path, Path | None]]


def run_command(args: list[str], *, stdout: Path | None = None) -> subprocess.CompletedProcess[str]:
    if stdout is None:
        return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    with stdout.open("w") as f:
        return subprocess.run(args, cwd=ROOT, text=True, stdout=f, stderr=subprocess.PIPE)


def write_header(buf: bytearray, n_tensors: int, n_metadata: int, version: int = 3) -> None:
    buf.extend(struct.pack("<I", 0x4655_4747))
    buf.extend(struct.pack("<I", version))
    buf.extend(struct.pack("<Q", n_tensors))
    buf.extend(struct.pack("<Q", n_metadata))


def build_directory_fixture(work: Path, name: str) -> Path:
    path = work / f"{name}.gguf"
    directory.make_fixture(path)
    return path


def build_invalid_magic(work: Path) -> tuple[Path, None]:
    path = build_directory_fixture(work, "invalid-magic")
    data = bytearray(path.read_bytes())
    data[0] = 0
    path.write_bytes(data)
    return path, None


def build_unsupported_version(work: Path) -> tuple[Path, None]:
    path = build_directory_fixture(work, "unsupported-version")
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 4, 2)
    path.write_bytes(data)
    return path, None


def build_truncated_metadata(work: Path) -> tuple[Path, None]:
    path = work / "truncated-metadata.gguf"
    directory.make_truncated_metadata_fixture(path)
    return path, None


def build_unknown_metadata_type(work: Path) -> tuple[Path, None]:
    path = work / "unknown-metadata-type.gguf"
    buf = bytearray()
    write_header(buf, n_tensors=0, n_metadata=1)
    directory.write_string(buf, "general.name")
    buf.extend(struct.pack("<I", 99))
    path.write_bytes(buf)
    return path, None


def build_bad_tensor_dimension(work: Path) -> tuple[Path, None]:
    path = work / "bad-tensor-dimension.gguf"
    buf = bytearray()
    write_header(buf, n_tensors=1, n_metadata=0)
    directory.write_tensor(buf, "bad.weight", [], 0, 0)
    path.write_bytes(buf)
    return path, None


def build_tensor_data_outside_file(work: Path) -> tuple[Path, None]:
    path = build_directory_fixture(work, "tensor-data-outside-file")
    path.write_bytes(path.read_bytes()[:-8])
    return path, None


def build_tensor_offset_overflow(work: Path) -> tuple[Path, None]:
    path = work / "tensor-offset-overflow.gguf"
    directory.make_offset_overflow_fixture(path)
    return path, None


def build_metadata_case(
    work: Path,
    name: str,
    mutation: Callable[[list[metadata.Entry]], None],
) -> tuple[Path, None]:
    path = work / f"{name}.gguf"
    metadata.make_fixture(path, mutation)
    return path, None


def build_missing_required_metadata(work: Path) -> tuple[Path, None]:
    return build_metadata_case(
        work,
        "missing-required-metadata",
        lambda entries: metadata.remove(entries, "deepseek4.block_count"),
    )


def build_wrong_metadata_type(work: Path) -> tuple[Path, None]:
    return build_metadata_case(
        work,
        "wrong-metadata-type",
        lambda entries: metadata.replace(
            entries,
            "deepseek4.rope.scaling.original_context_length",
            "string",
            "65536",
        ),
    )


def build_bad_array_length(work: Path) -> tuple[Path, None]:
    return build_metadata_case(
        work,
        "bad-array-length",
        lambda entries: metadata.replace(
            entries,
            "deepseek4.attention.compress_ratios",
            "array_u32",
            [0],
        ),
    )


def build_unsupported_tensor_type(work: Path) -> tuple[Path, Path]:
    base = bindings.base_tensors()
    mtp = bindings.mtp_tensors()
    bindings.replace_tensor(base, "output.weight", type_id=31)
    base_path = work / "unsupported-tensor-type-base.gguf"
    mtp_path = work / "unsupported-tensor-type-mtp.gguf"
    bindings.write_gguf(base_path, base, include_metadata=True)
    bindings.write_gguf(mtp_path, mtp, include_metadata=False)
    return base_path, mtp_path


def c_command(case: Case, path: Path, mtp_path: Path | None) -> list[str]:
    dump = str(ROOT / "ds4-metadata-dump")
    if case.mode == "directory":
        return [dump, "--directory-only", "-m", str(path)]
    if case.mode == "config":
        return [dump, "--validate-config-only", "-m", str(path)]
    if case.mode == "layout":
        assert mtp_path is not None
        return [dump, "--validate-layout-only", "-m", str(path), "--mtp", str(mtp_path)]
    raise AssertionError(case.mode)


def rust_command(case: Case, path: Path, mtp_path: Path | None) -> list[str]:
    command = [
        "cargo",
        "run",
        "-p",
        "ds4-gguf",
        "--bin",
        "ds4-gguf-dump",
        "--quiet",
        "--",
    ]
    if case.mode == "config":
        command.append("--validate-ds4-metadata")
    elif case.mode == "layout":
        assert mtp_path is not None
        command.extend(["--validate-ds4-layout", "--mtp", str(mtp_path)])
    command.append(str(path))
    return command


def normalize_error(proc: subprocess.CompletedProcess[str]) -> str:
    text = proc.stderr or proc.stdout
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    last = last.removeprefix("Error: ")

    if "model is not a GGUF file" in last:
        return "invalid-magic"
    if "only GGUF v3 is supported" in last:
        return "unsupported-version"
    if "truncated GGUF file" in last:
        return "truncated"
    if "unknown GGUF metadata type" in last:
        return "unknown-metadata-type"
    if "tensor has an unsupported number of dimensions" in last:
        return "bad-tensor-dimension"
    if "tensor points outside GGUF file" in last:
        return "tensor-data-outside-file"
    if "tensor offset overflow" in last:
        return "tensor-offset-overflow"
    if "required metadata key is missing:" in last:
        return "missing-metadata:" + last.rsplit(":", 1)[1].strip()
    if "metadata key has a non-integer type:" in last:
        return "wrong-metadata-type:" + last.rsplit(":", 1)[1].strip()
    if "compress_ratios is shorter" in last:
        return "bad-array-length:deepseek4.attention.compress_ratios"

    match = re.search(r"tensor (.+) has type .+, expected .+$", last)
    if match:
        return "unsupported-tensor-type:" + match.group(1)

    return last


def cases() -> list[Case]:
    return [
        Case("invalid-magic", "directory", "invalid-magic", build_invalid_magic),
        Case("unsupported-version", "directory", "unsupported-version", build_unsupported_version),
        Case("truncated-metadata", "directory", "truncated", build_truncated_metadata),
        Case("unknown-metadata-type", "directory", "unknown-metadata-type", build_unknown_metadata_type),
        Case("bad-tensor-dimension", "directory", "bad-tensor-dimension", build_bad_tensor_dimension),
        Case("tensor-data-outside-file", "directory", "tensor-data-outside-file", build_tensor_data_outside_file),
        Case("tensor-offset-overflow", "directory", "tensor-offset-overflow", build_tensor_offset_overflow),
        Case(
            "missing-required-metadata",
            "config",
            "missing-metadata:deepseek4.block_count",
            build_missing_required_metadata,
        ),
        Case(
            "wrong-metadata-type",
            "config",
            "wrong-metadata-type:deepseek4.rope.scaling.original_context_length",
            build_wrong_metadata_type,
        ),
        Case(
            "bad-array-length",
            "config",
            "bad-array-length:deepseek4.attention.compress_ratios",
            build_bad_array_length,
        ),
        Case(
            "unsupported-tensor-type",
            "layout",
            "unsupported-tensor-type:output.weight",
            build_unsupported_tensor_type,
        ),
    ]


def compare_case(report: Report, work: Path, case: Case) -> None:
    case_dir = work / case.name
    case_dir.mkdir()
    path, mtp_path = case.build(case_dir)

    c_result = run_command(c_command(case, path, mtp_path))
    rust_result = run_command(rust_command(case, path, mtp_path))
    report.check(c_result.returncode != 0, f"C accepted {case.name}")
    report.check(rust_result.returncode != 0, f"Rust accepted {case.name}")
    if c_result.returncode == 0 or rust_result.returncode == 0:
        return

    c_error = normalize_error(c_result)
    rust_error = normalize_error(rust_result)
    report.check(c_error == case.expected, f"C {case.name}: {c_error} != {case.expected}")
    report.check(rust_error == case.expected, f"Rust {case.name}: {rust_error} != {case.expected}")
    report.check(c_error == rust_error, f"{case.name} mismatch: C={c_error} Rust={rust_error}")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    sections = 1 if report.ok else 0
    print(f"gguf negative fixture comparison: {status}, {report.checks} checks")
    print(f"summary: {sections}/1 sections passed, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()

    all_cases = cases()
    if args.list_cases:
        for case in all_cases:
            print(f"{case.name}\t{case.mode}\t{case.expected}")
        return 0

    report = Report()
    with tempfile.TemporaryDirectory(prefix="ds4-gguf-failures-") as tmp:
        work = Path(tmp)
        for case in all_cases:
            compare_case(report, work, case)

    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
