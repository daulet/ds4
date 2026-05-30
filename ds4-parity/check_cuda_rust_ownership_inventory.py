#!/usr/bin/env python3
"""Validate the M14.0 CUDA-to-Rust ownership inventory and stage contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
GPU_HEADER = ROOT / "ds4_gpu.h"
RUST_FFI = ROOT / "rust/ds4-gpu-sys/src/lib.rs"
RUST_BUILD = ROOT / "rust/ds4-gpu/build.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_SCHEMA = "ds4.cuda_rust_ownership_inventory.v1"
EXPECTED_MILESTONE = "M14.0"
EXPECTED_STAGES = ["M14.1", "M14.2", "M14.3", "M14.4", "M14.5", "M14.6"]
EXPECTED_DS4_HASHES = {
    "ds4_cuda.cu": "d819ff2ad0945b58057b1b0ff95f1d17550f0e12fd639b3e20dbfddd8431c0c7",
    "ds4_gpu.h": "317349630b134b86cdb7cd293f267357b969f5899bb6c55bce7cc7aab9187554",
    "rust/ds4-gpu-sys/src/lib.rs": "4ba693fa05f4f5d13759d94705ca84cb2c4975754b95134651f566cd0df512a1",
    "rust/ds4-gpu/build.rs": "ba411f880038856702541185790994cf888e1c3cf54ac42a57a5319fc8f94ee7",
}
EXPECTED_CUDA_OXIDE_REVISION = "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200"
EXPECTED_CAPABILITIES = {
    "rust_kernel_codegen_and_launch",
    "device_memory_stream_event_and_residency_raii",
    "cublas_sgemm_and_strided_batched_sgemm",
    "low_precision_storage_and_deterministic_topk_primitives",
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
    inventory = load_json(INVENTORY)
    texts = {
        "cuda": read_text(CUDA_SOURCE),
        "header": read_text(GPU_HEADER),
        "ffi": read_text(RUST_FFI),
        "build": read_text(RUST_BUILD),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
        "readme": read_text(README),
        "report": read_text(REPORT),
    }
    report = Report()
    validate(report, inventory, texts)
    if args.negative_test:
        run_negative_tests(report, inventory, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, inventory: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(inventory.get("schema") == EXPECTED_SCHEMA, "inventory schema drift")
    report.check(inventory.get("milestone") == EXPECTED_MILESTONE, "inventory milestone drift")
    report.check(inventory.get("status") == "scope-and-adoption-contract", "inventory status drift")
    validate_source_snapshot(report, inventory, texts)
    validate_cuda_oxide_snapshot(report, inventory)
    validate_stages(report, inventory)
    validate_abi_ownership(report, inventory, texts)
    validate_kernel_ownership(report, inventory, texts)
    validate_claim_policy(report, inventory)
    validate_static_wiring(report, texts)


def validate_source_snapshot(
    report: Report, inventory: dict[str, Any], texts: dict[str, str]
) -> None:
    snapshot = require_dict(report, inventory.get("source_snapshot"), "source_snapshot")
    source_hashes = require_dict(report, snapshot.get("ds4_files"), "source_snapshot.ds4_files")
    report.check(source_hashes == EXPECTED_DS4_HASHES, "DS4 source hash inventory drift")
    for path, expected in EXPECTED_DS4_HASHES.items():
        report.check(sha256(ROOT / path) == expected, f"source changed without inventory refresh: {path}")
    report.check(
        "compile_cuda(&repo_root, &out_dir, &cuda_obj)" in texts["build"],
        "Rust build no longer compiles ds4_cuda.cu",
    )
    report.check('println!("cargo:rustc-link-lib=dylib=cudart");' in texts["build"], "cudart link marker missing")
    report.check('println!("cargo:rustc-link-lib=dylib=cublas");' in texts["build"], "cublas link marker missing")


def validate_cuda_oxide_snapshot(report: Report, inventory: dict[str, Any]) -> None:
    snapshot = require_dict(report, inventory.get("cuda_oxide_snapshot"), "cuda_oxide_snapshot")
    report.check(snapshot.get("revision") == EXPECTED_CUDA_OXIDE_REVISION, "cuda-oxide revision drift")
    report.check(snapshot.get("ref") == "main", "cuda-oxide ref drift")
    report.check(snapshot.get("verified_local_head_matches_remote_main") is True, "cuda-oxide provenance not verified")
    capabilities = snapshot.get("inspected_capabilities")
    report.check(isinstance(capabilities, list), "cuda-oxide capabilities missing")
    if isinstance(capabilities, list):
        actual = {item.get("capability") for item in capabilities if isinstance(item, dict)}
        report.check(actual == EXPECTED_CAPABILITIES, "cuda-oxide capability inventory drift")
        for item in capabilities:
            report.check(isinstance(item.get("evidence"), list) and bool(item["evidence"]), "capability evidence missing")
            report.check(isinstance(item.get("used_by_stages"), list) and bool(item["used_by_stages"]), "capability stage use missing")
    constraints = snapshot.get("adoption_constraints")
    report.check(isinstance(constraints, list) and len(constraints) == 3, "cuda-oxide constraint drift")


def validate_stages(report: Report, inventory: dict[str, Any]) -> None:
    stages = inventory.get("stages")
    report.check(isinstance(stages, list), "stage plan missing")
    if not isinstance(stages, list):
        return
    actual = [stage.get("stage") for stage in stages if isinstance(stage, dict)]
    report.check(actual == EXPECTED_STAGES, "stage order drift")
    for stage in stages:
        report.check(isinstance(stage.get("title"), str) and bool(stage["title"]), "stage title missing")
        for key in ["goal", "oracle", "comparator", "acceptance"]:
            report.check(isinstance(stage.get(key), str) and bool(stage[key]), f"stage {key} missing")


def validate_abi_ownership(
    report: Report, inventory: dict[str, Any], texts: dict[str, str]
) -> None:
    header_functions = set(re.findall(r"\b(ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["header"]))
    cuda_exports = set(
        re.findall(r'extern\s+"C"\s+[\s\S]*?\b(ds4_gpu_[A-Za-z0-9_]+)\s*\(', texts["cuda"])
    )
    rust_ffi = set(re.findall(r"pub fn (ds4_gpu_[A-Za-z0-9_]+)\s*\(", texts["ffi"]))
    counts = require_dict(report, require_dict(report, inventory.get("source_snapshot"), "source_snapshot").get("counts"), "counts")
    report.check(len(header_functions) == counts.get("public_header_functions"), "public header function count drift")
    report.check(len(cuda_exports) == counts.get("cuda_exported_functions"), "CUDA export count drift")
    report.check(len(rust_ffi) == counts.get("rust_ffi_functions"), "Rust FFI function count drift")
    report.check(rust_ffi == header_functions, "Rust FFI no longer exactly mirrors public ds4_gpu.h functions")
    cuda_only = sorted(cuda_exports - header_functions)
    report.check(
        cuda_only == inventory["source_snapshot"].get("cuda_only_exported_helpers"),
        "CUDA-only exported helper inventory drift",
    )
    report.check(not (header_functions - cuda_exports), "public GPU ABI declaration lacks CUDA implementation")
    families = inventory.get("abi_families")
    report.check(isinstance(families, list), "ABI family assignment missing")
    assigned = flatten_family_symbols(report, families, "ABI")
    report.check(assigned == cuda_exports, "not every CUDA exported function has exactly one ownership stage")
    if isinstance(families, list):
        report.check(
            [family.get("stage") for family in families] == EXPECTED_STAGES[:-1],
            "ABI ownership family order drift",
        )


def validate_kernel_ownership(
    report: Report, inventory: dict[str, Any], texts: dict[str, str]
) -> None:
    kernels = set(
        re.findall(
            r"__global__\s+static\s+(?:DS4_CUDA_UNUSED\s+)?(?:[A-Za-z_][A-Za-z0-9_:* ]*\s+)?([A-Za-z0-9_]+_kernel)\s*\(",
            texts["cuda"],
        )
    )
    counts = inventory["source_snapshot"]["counts"]
    report.check(len(kernels) == counts.get("unique_cuda_kernels"), "CUDA kernel count drift")
    families = inventory.get("kernel_families")
    report.check(isinstance(families, list), "kernel family assignment missing")
    assigned = flatten_family_symbols(report, families, "kernel")
    report.check(assigned == kernels, "not every CUDA kernel has exactly one ownership stage")
    if isinstance(families, list):
        report.check(
            [family.get("stage") for family in families] == EXPECTED_STAGES[:-1],
            "kernel ownership family order drift",
        )


def validate_claim_policy(report: Report, inventory: dict[str, Any]) -> None:
    policy = require_dict(report, inventory.get("claim_policy"), "claim_policy")
    report.check(policy.get("rust_cuda_operation_ownership_active") is False, "M14.0 operation ownership overclaim")
    report.check(policy.get("rust_cuda_default_route_active") is False, "M14.0 default route overclaim")
    report.check(policy.get("ds4_cuda_removal_allowed") is False, "M14.0 removal overclaim")
    report.check(policy.get("current_c_cuda_retained_as_oracle") is True, "current C CUDA oracle retention drift")
    report.check(
        policy.get("next_stage") == "M14.1 cuda-oxide Substrate And Tensor Residency",
        "next ownership stage drift",
    )


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json"
    checker = "check_cuda_rust_ownership_inventory.py"
    report.check("Milestone 14: Rust CUDA Ownership Via cuda-oxide" in texts["roadmap"], "roadmap M14 section missing")
    report.check(fixture in texts["roadmap"], "roadmap M14 fixture missing")
    report.check("M14.0: CUDA Rust Ownership Inventory And Adoption Contract" in texts["todo"], "TODO M14.0 item missing")
    report.check(fixture in texts["todo"], "TODO M14 fixture missing")
    report.check(
        "Active item: M14." in texts["status"],
        "status next active M14 stage missing",
    )
    report.check("M14.0 CUDA Rust Ownership Inventory And Adoption Contract" in texts["status"], "status M14.0 evidence missing")
    report.check(checker in texts["readme"], "README M14 checker wiring missing")
    report.check(checker in texts["report"], "unified report M14 checker wiring missing")


def flatten_family_symbols(report: Report, families: Any, name: str) -> set[str]:
    if not isinstance(families, list):
        return set()
    values: list[str] = []
    for family in families:
        report.check(isinstance(family, dict), f"{name} family malformed")
        if not isinstance(family, dict):
            continue
        report.check(family.get("stage") in EXPECTED_STAGES[:-1], f"{name} family stage invalid")
        symbols = family.get("symbols")
        report.check(isinstance(symbols, list) and bool(symbols), f"{name} family symbols missing")
        if isinstance(symbols, list):
            values.extend(symbol for symbol in symbols if isinstance(symbol, str))
    report.check(len(values) == len(set(values)), f"{name} symbol assigned more than once")
    return set(values)


def run_negative_tests(report: Report, inventory: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("missing ABI assignment", lambda item: item["abi_families"][0]["symbols"].pop()),
        ("missing kernel assignment", lambda item: item["kernel_families"][0]["symbols"].pop()),
        ("removal overclaim", lambda item: item["claim_policy"].update({"ds4_cuda_removal_allowed": True})),
    ]:
        candidate = copy.deepcopy(inventory)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.0 CUDA Rust ownership inventory: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
