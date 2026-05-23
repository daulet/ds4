#!/usr/bin/env python3
"""Validate the M10.5c4c2a Rust decode model-map bridge contract."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "sys": ROOT / "rust/ds4-gpu-sys/src/lib.rs",
    "backend": ROOT / "rust/ds4-gpu/src/decode_backend.rs",
    "test": ROOT / "rust/ds4-gpu/tests/model_map_abi.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
}
MODEL_MAP_OPERATIONS = [
    "ds4_gpu_set_model_map",
    "ds4_gpu_set_model_fd",
    "ds4_gpu_set_model_map_range",
    "ds4_gpu_cache_model_range",
    "ds4_gpu_cache_q8_f16_range",
]


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
    validate(report, texts)
    if report.ok:
        print(f"Rust decode model-map bridge contract: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, texts: dict[str, str]) -> None:
    sys_text = texts["sys"]
    backend = texts["backend"]
    test = texts["test"]
    report_text = texts["report"]

    for operation in MODEL_MAP_OPERATIONS:
        report.check(re.search(rf"pub fn {operation}\(", sys_text) is not None, f"{operation} missing from sys ABI")
        report.check(operation in backend, f"{operation} missing from safe backend bridge")

    for wrapper in [
        "set_model_map",
        "set_model_fd",
        "set_model_map_range",
        "cache_model_range",
        "cache_q8_f16_range",
    ]:
        report.check(re.search(rf"pub fn {wrapper}\(", backend) is not None, f"{wrapper} wrapper missing")

    report.check("MODEL_MAP_BACKEND_OPERATIONS" in backend, "model-map operation table missing")
    report.check('wrapper: "set_model_map_range"' in backend, "set_model_map_range not listed in operation table")
    report.check(
        '#[cfg(all(target_os = "linux", feature = "cuda-backend"))]\npub fn cache_model_range' in backend,
        "cache_model_range must be Linux CUDA feature-gated",
    )
    report.check(
        '#[cfg(all(target_os = "linux", feature = "cuda-backend"))]\npub fn cache_q8_f16_range' in backend,
        "cache_q8_f16_range must be Linux CUDA feature-gated",
    )
    report.check("label.map_or(ptr::null(), CStr::as_ptr)" in backend, "optional C label handling drift")
    report.check("validate_model_range(model, map_offset, map_size)?" in backend, "map-range wrapper must validate before FFI")
    report.check("validate_model_cache_range(model, offset, bytes)?" in backend, "cache-range wrappers must validate before FFI")
    report.check("if bytes == 0 {\n        Ok(())" in backend, "zero-byte cache range should follow C no-op")
    report.check("GpuError::invalid_range()" in backend, "model range validation must return invalid_range")
    report.check("GpuStatus::from_raw(sys::ds4_gpu_set_model_map_range" in backend, "set_model_map_range status bridge drift")
    report.check("GpuStatus::from_raw(sys::ds4_gpu_cache_model_range" in backend, "cache_model_range status bridge drift")

    report.check('feature = "cuda-backend"' in test, "B300 model-map test must require cuda-backend feature")
    report.check('target_os = "linux"' in test, "B300 model-map test must be Linux-gated")
    report.check('target_os = "macos"' not in test, "model-map ABI smoke must not run on Metal Vec mappings")
    report.check("model_map_wrappers_accept_fd_and_mapped_ranges" in test, "model-map ABI test missing")
    report.check("set_model_fd(file.as_raw_fd())" in test, "model fd path not tested")
    report.check("set_model_map_range(model, 64, 512)" in test, "model map range path not tested")
    report.check("cache_model_range(model, 128, 256" in test, "CUDA cache model range path not tested")
    report.check("cache_q8_f16_range(model, 256, 512" in test, "CUDA q8/f16 cache path not tested")
    report.check("zero-byte cache range follows C no-op" in test, "zero-byte cache no-op path not tested")
    report.check("bytes.len() as u64 + 1" in test, "out-of-range failure path not tested")

    report.check("M10.5c4c2a Rust decode model-map bridge comparator" in report_text, "unified comparator entry missing")
    report.check("M10.5c4c2a B300 Rust model-map backend smoke rerun" in report_text, "B300 model-map smoke skip missing")
    report.check("--test model_map_abi" in report_text, "B300 model-map smoke command missing")
    report.check("--features cuda-backend" in report_text, "B300 model-map smoke feature missing")

    readme = texts["readme"]
    report.check("M10.5c4c2a Rust decode model-map bridge" in readme, "README missing M10.5c4c2a entry")


def run_negative_tests(texts: dict[str, str]) -> int:
    mutations = [
        ("missing safe wrapper", "backend", "pub fn set_model_map_range", "fn set_model_map_range"),
        (
            "cache wrapper loses linux feature cfg",
            "backend",
            '#[cfg(all(target_os = "linux", feature = "cuda-backend"))]\npub fn cache_model_range',
            "pub fn cache_model_range",
        ),
        ("range validation removed", "backend", "validate_model_range(model, map_offset, map_size)?", ""),
        ("test loses fd path", "test", "set_model_fd(file.as_raw_fd())", "set_model_fd(-1)"),
        ("test loses cache path", "test", "cache_model_range(model, 128, 256", "cache_model_range(model, 0, 0"),
        ("report loses b300 command", "report", "--test model_map_abi", "--test missing_model_map"),
    ]
    failures: list[str] = []
    for label, key, needle, replacement in mutations:
        mutated = copy.deepcopy(texts)
        if needle not in mutated[key]:
            failures.append(f"{label}: mutation needle not found")
            continue
        mutated[key] = mutated[key].replace(needle, replacement)
        report = Report()
        validate(report, mutated)
        if report.ok:
            failures.append(f"{label}: validation unexpectedly passed")

    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(mutations)} mutations rejected")
    return 0


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
