#!/usr/bin/env python3
"""Validate the M10.5c4c1 Rust CUDA backend smoke contract."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "cargo": ROOT / "rust/ds4-gpu/Cargo.toml",
    "build": ROOT / "rust/ds4-gpu/build.rs",
    "backend_abi": ROOT / "rust/ds4-gpu/tests/backend_abi.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
}


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
        print(f"B300 Rust backend smoke contract: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, texts: dict[str, str]) -> None:
    cargo = texts["cargo"]
    report.check("[features]" in cargo, "ds4-gpu Cargo.toml missing features table")
    report.check("default = []" in cargo, "ds4-gpu default feature set drift")
    report.check("cuda-backend = []" in cargo, "ds4-gpu cuda-backend feature missing")

    build = texts["build"]
    report.check("CARGO_FEATURE_CUDA_BACKEND" in build, "Linux CUDA build must be feature-gated")
    report.check('"linux" if env::var_os("CARGO_FEATURE_CUDA_BACKEND").is_some()' in build, "Linux CUDA gate drift")
    report.check("build_linux_cuda_backend" in build, "Linux CUDA backend builder missing")
    report.check("ds4_cuda.cu" in build, "CUDA source not tracked by build.rs")
    report.check("ds4_iq2_tables_cuda.inc" in build, "CUDA include not tracked by build.rs")
    report.check("CUDA_HOME" in build, "CUDA_HOME env handling missing")
    report.check("CUDA_ARCH" in build, "CUDA_ARCH env handling missing")
    report.check("NVCC" in build, "NVCC env handling missing")
    report.check("-arch={arch}" in build, "nvcc architecture flag handling missing")
    report.check("cudart" in build, "cudart link missing")
    report.check("cublas" in build, "cublas link missing")
    report.check("stdc++" in build, "C++ runtime link missing")
    report.check("static=ds4_backend" in build, "static backend link missing")
    report.check('CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")' in build, "compiler arch wrapper must be macOS-only")

    backend_abi = texts["backend_abi"]
    report.check('target_os = "macos"' in backend_abi, "macOS backend ABI coverage removed")
    report.check('target_os = "linux"' in backend_abi, "Linux backend ABI cfg missing")
    report.check('feature = "cuda-backend"' in backend_abi, "Linux backend ABI must require cuda-backend feature")

    report_text = texts["report"]
    report.check("M10.5c4c1 B300 Rust CUDA backend smoke rerun" in report_text, "B300 smoke skip item missing")
    report.check("rustup default stable" in report_text, "B300 smoke command missing Rust toolchain bootstrap")
    report.check("--features cuda-backend" in report_text, "B300 smoke command missing cuda-backend feature")
    report.check("--test backend_abi" in report_text, "B300 smoke command missing backend ABI test")
    report.check("CUDA_ARCH=native" in report_text, "B300 smoke command missing CUDA_ARCH=native")

    readme = texts["readme"]
    report.check("M10.5c4c1 Rust CUDA backend smoke" in readme, "README missing M10.5c4c1 entry")


def run_negative_tests(texts: dict[str, str]) -> int:
    mutations = [
        ("missing feature", "cargo", "cuda-backend = []", ""),
        ("ungated linux build", "build", "CARGO_FEATURE_CUDA_BACKEND", "CARGO_FEATURE_REMOVED"),
        ("missing cuda source", "build", "ds4_cuda.cu", "ds4_cuda_missing.cu"),
        ("linux arch wrapper", "build", 'CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")', 'CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")'),
        ("test cfg loses feature", "backend_abi", 'feature = "cuda-backend"', 'feature = "removed"'),
        ("skip command loses feature", "report", "--features cuda-backend", "--features removed"),
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
