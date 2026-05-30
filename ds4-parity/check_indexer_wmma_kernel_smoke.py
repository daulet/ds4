#!/usr/bin/env python3
"""Validate the M14.2d2b1 Rust CUDA base WMMA indexer score kernel smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.2d2b1/indexer-wmma-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/indexer_wmma_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_DEPENDENCY_REVISION = "d4791b7002152af3b7f6b15a48d7f5acd7a63011"
CURRENT_DEPENDENCY_REVISION = "d8ccb4174e0a92b1b80424c1c7258b29a07e4bb7"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide indexer_scores_wmma_kernel launch proof",
    "current-C-shaped base 16-component tensor-core tile and causal masking semantics",
]
EXPECTED_NOT_CLAIMED = [
    "widened WMMA indexer score dispatch",
    "specialized top-k dispatch",
    "runtime graph integration or default CUDA route",
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
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "cargo": CARGO.read_text(encoding="utf-8"),
        "lock": LOCK.read_text(encoding="utf-8"),
        "lib": CRATE_LIB.read_text(encoding="utf-8"),
        "smoke": SMOKE.read_text(encoding="utf-8"),
        "cuda": CUDA_SOURCE.read_text(encoding="utf-8"),
        "roadmap": ROADMAP.read_text(encoding="utf-8"),
        "todo": TODO.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
        "readme": README.read_text(encoding="utf-8"),
        "report": REPORT.read_text(encoding="utf-8"),
    }
    report = Report()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.indexer_wmma_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.2d2b1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == RECORDED_DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "crate dependency revision pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-indexer-wmma-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void indexer_scores_wmma_kernel",
        "__shared__ __half a_sh[16 * 128];",
        "__shared__ __half b_sh[16 * 128];",
        "wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>",
        "wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>",
        "wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);",
        "fmaxf(c_sh[i], 0.0f) * w",
        "indexer_scores_wmma_kernel<<<grid, 32>>>",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_indexer_scores_wmma_kernel", True),
        ("owns_widened_wmma_dispatch", False),
        ("owns_specialized_topk_dispatch", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_2D2B1_SCOPE",
        "owns_indexer_scores_wmma_kernel: true",
        "owns_widened_wmma_dispatch: false",
        "owns_specialized_topk_dispatch: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-indexer-wmma-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a device target")
    report.check("CUDA_OXIDE_LINK_TARGET" not in command, "smoke command forces a link target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.2d2b1",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "base_wmma_output_matches": True,
        "causal_mask_output_matches": True,
        "two_mma_column_halves_match": True,
        "weighted_epilogue_matches": True,
        "nan_negative_clamp_matches": True,
        "invalid_shape_rejected": True,
        "owns_indexer_scores_wmma_kernel": True,
        "owns_widened_wmma_dispatch": False,
        "owns_specialized_topk_dispatch": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 base WMMA result drift")
    for marker in [
        "#![feature(f16)]",
        "pub fn indexer_scores_wmma_kernel",
        "SharedArray<f16, { TILE_TOKENS * MMA_K }, 16>",
        "load_a_m16n8k16",
        "load_b_m16n8k16",
        "mma_m16n8k16_f32_f16",
        "(value.to_bits() & 0x7fff_ffff) > 0x7f80_0000",
        "weighted_epilogue_matches",
        "nan_negative_clamp_matches",
        "two_mma_column_halves_match",
        "Err(IndexerWmmaError::InvalidShape)",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.2d2b1/indexer-wmma-kernel-smoke.json"
    checker = "check_indexer_wmma_kernel_smoke.py"
    report.check("M14.2d2b1: Base Tensor-Core Indexer Score Kernel" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.2d2b1: Base Tensor-Core Indexer Score Kernel" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14." in texts["status"], "next active stage missing")
    report.check("M14.2d2b1 Base Tensor-Core Indexer Score Kernel" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("WMMA result absent", lambda value: value["b300_execution"]["stdout"].update({"base_wmma_output_matches": False})),
        ("causal result absent", lambda value: value["b300_execution"]["stdout"].update({"causal_mask_output_matches": False})),
        ("two-half result absent", lambda value: value["b300_execution"]["stdout"].update({"two_mma_column_halves_match": False})),
        ("weighted result absent", lambda value: value["b300_execution"]["stdout"].update({"weighted_epilogue_matches": False})),
        ("clamp result absent", lambda value: value["b300_execution"]["stdout"].update({"nan_negative_clamp_matches": False})),
        ("widened overclaim", lambda value: value["ownership"].update({"owns_widened_wmma_dispatch": True})),
        ("top-k overclaim", lambda value: value["ownership"].update({"owns_specialized_topk_dispatch": True})),
        ("route overclaim", lambda value: value["ownership"].update({"changes_default_route": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.2d2b1 base WMMA indexer kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
