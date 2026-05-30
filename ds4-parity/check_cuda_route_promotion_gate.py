#!/usr/bin/env python3
"""Validate the M14.6a CUDA route promotion blocker."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.6a/production-route-blocker.json"
GPU_CARGO = ROOT / "rust/ds4-gpu/Cargo.toml"
GPU_BUILD = ROOT / "rust/ds4-gpu/build.rs"
GPU_SYS = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
CUDA_CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
CUDA_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
CUDA_ABI = ROOT / "rust/ds4-cuda/src/abi.rs"
ENGINE = ROOT / "rust/ds4-engine/src/lib.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_BLOCKERS = [
    "rust/ds4-gpu/build.rs still compiles and archives ds4_cuda.cu for the Linux CUDA backend",
    "rust/ds4-gpu does not depend on rust/ds4-cuda and rust/ds4-cuda does not export the ds4_gpu ABI",
    "rust/ds4-engine still rejects the runtime graph route instead of executing a Rust CUDA backend",
]


@dataclass
class ReportState:
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
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "gpu_cargo": GPU_CARGO.read_text(encoding="utf-8"),
        "gpu_build": GPU_BUILD.read_text(encoding="utf-8"),
        "gpu_sys": GPU_SYS.read_text(encoding="utf-8"),
        "cuda_cargo": CUDA_CARGO.read_text(encoding="utf-8"),
        "cuda_lib": CUDA_LIB.read_text(encoding="utf-8"),
        "cuda_abi": CUDA_ABI.read_text(encoding="utf-8"),
        "engine": ENGINE.read_text(encoding="utf-8"),
        "roadmap": ROADMAP.read_text(encoding="utf-8"),
        "todo": TODO.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
        "readme": README.read_text(encoding="utf-8"),
        "report": REPORT.read_text(encoding="utf-8"),
    }
    report = ReportState()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.6a CUDA route promotion blocker: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_route_promotion_gate.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.6a", "milestone drift")
    report.check(
        fixture.get("status") == "blocked-production-rust-abi-not-assembled",
        "status drift",
    )
    validate_boundary(report, fixture, texts)
    validate_decision(report, fixture, texts)
    validate_validation(report, fixture)
    validate_wiring(report, texts)


def validate_boundary(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    boundary = require_dict(report, fixture.get("production_boundary"), "production_boundary")
    for key, expected in [
        ("runtime_gpu_crate", "rust/ds4-gpu"),
        ("rust_kernel_crate", "rust/ds4-cuda"),
        ("ffi_crate", "rust/ds4-gpu-sys"),
        ("production_build_still_compiles_ds4_cuda_cu", True),
        ("production_gpu_depends_on_ds4_cuda", False),
        ("rust_cuda_library_exports_ds4_gpu_abi", False),
        ("rust_cuda_kernel_modules_are_executable_local", True),
        ("runtime_graph_route_implemented", False),
        ("public_gpu_abi_function_count", 81),
        ("cuda_only_exported_helper_count", 2),
    ]:
        report.check(boundary.get(key) == expected, f"boundary drift: {key}")
    for marker in [
        'repo_root.join("ds4_cuda.cu")',
        "compile_cuda(&repo_root, &out_dir, &cuda_obj)",
        ".arg(&cuda_obj)",
        '.arg("ds4_cuda.cu")',
    ]:
        report.check(marker in texts["gpu_build"], f"production CUDA build marker missing: {marker}")
    report.check("ds4-cuda" not in texts["gpu_cargo"], "production GPU crate unexpectedly depends on ds4-cuda")
    report.check('cuda-backend = []' in texts["gpu_cargo"], "production CUDA feature marker missing")
    ffi_exports = re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["gpu_sys"])
    report.check(len(set(ffi_exports)) == 81, "Rust FFI function count drift")
    report.check("[[bin]]" in texts["cuda_cargo"], "cuda-oxide executable-local proof markers missing")
    report.check(
        'crate-type = ["rlib", "staticlib"]' in texts["cuda_cargo"],
        "Rust CUDA resource ABI successor staticlib missing",
    )
    report.check("pub mod abi;" in texts["cuda_lib"], "Rust CUDA resource ABI successor module missing")
    report.check(
        'pub extern "C" fn ds4_gpu_init' in texts["cuda_abi"],
        "Rust CUDA resource ABI successor init export missing",
    )
    report.check(
        "owns_complete_ds4_gpu_abi: false" in texts["cuda_lib"],
        "partial resource ABI unexpectedly claims complete ABI",
    )
    report.check(
        "--runtime-graph graph is not implemented yet" in texts["engine"],
        "runtime graph rejection marker missing",
    )


def validate_decision(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    decision = require_dict(report, fixture.get("decision"), "decision")
    report.check(decision.get("operation_families_validated") is True, "operation completion drift")
    report.check(decision.get("default_route_promotion_allowed") is False, "default route overclaim")
    report.check(decision.get("c_cuda_removal_allowed") is False, "C CUDA removal overclaim")
    report.check(decision.get("blockers") == EXPECTED_BLOCKERS, "blocker list drift")
    report.check(
        decision.get("next_required_stage") == "M14.6b Rust CUDA ABI Backend Assembly",
        "next required stage drift",
    )
    for marker in [
        "pub const M14_6A_GATE",
        "operation_families_validated: true",
        "production_build_still_compiles_ds4_cuda_cu: true",
        "rust_exports_ds4_gpu_abi: false",
        "runtime_graph_route_implemented: false",
        "can_promote_default_route: false",
        "can_remove_c_cuda: false",
    ]:
        report.check(marker in texts["cuda_lib"], f"Rust route-gate marker missing: {marker}")


def validate_validation(report: ReportState, fixture: dict[str, Any]) -> None:
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("date_utc") == "2026-05-30", "validation date drift")
    report.check(validation.get("local_test_count") == 86, "local test count drift")
    report.check(validation.get("b300_test_count") == 88, "B300 test count drift")
    report.check(
        validation.get("local_cuda_kernel_build_blocker") == "/usr/local/cuda/include/cuda.h",
        "local CUDA build blocker drift",
    )


def validate_wiring(report: ReportState, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.6a/production-route-blocker.json"
    checker = "check_cuda_route_promotion_gate.py"
    item = "M14.6a: Production Route Linkage Blocker"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        any(
            marker in texts["status"]
            for marker in [
                "Active item: M14.6b2 Rust CUDA Compute ABI Assembly",
                "Active item: M14.6b2b2 Remaining Rust CUDA Kernel ABI Assembly",
                "M14.6b2b2a Directional Steering ABI Export",
                "M14.6b2b2b1 SwiGLU Libdevice ABI Export",
                "M14.6b2b2b2a Plain RMS Norm ABI Export",
                "M14.6b2b2b2b1 Weighted RMS Device-Copy ABI Export",
                "M14.6b2b2b2b2a Basic Model-Control Device-Copy ABI Export",
                "M14.6b2b2b2b2b1 Registered Attempt And Device-Copy Fallback ABI",
                "M14.6b2b2b2b2b2a Pageable HMM Fallback ABI",
                "M14.6b2b2b2b2b2b1 Chunk-Selected Model Copy ABI",
                "M14.6b2b2b2b2b2b2a Whole-Map Registration Precedence ABI",
            ]
        ),
        "active stage missing",
    )
    report.check("M14.6a Production Route Linkage Blocker" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: ReportState, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("promoted default route", lambda value: value["decision"].update({"default_route_promotion_allowed": True})),
        ("removed C CUDA", lambda value: value["decision"].update({"c_cuda_removal_allowed": True})),
        ("missing ABI blocker", lambda value: value["decision"]["blockers"].pop(1)),
        ("false runtime readiness", lambda value: value["production_boundary"].update({"runtime_graph_route_implemented": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = ReportState()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: ReportState, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
