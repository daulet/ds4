#!/usr/bin/env python3
"""Validate the M14.2d2c2 Rust CUDA power-of-two top-k kernel smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.2d2c2/indexer-topk-pow2-kernel-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/indexer_topk_pow2_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

RECORDED_DEPENDENCY_REVISION = "d4791b7002152af3b7f6b15a48d7f5acd7a63011"
CURRENT_DEPENDENCY_REVISION = "1000e653df60a7814fa996d146e3823d0a364280"
EXPECTED_RUST_OWNED = [
    "executable-local cuda-oxide 2048 and 4096 u32-index bitonic kernel launch proof",
    "executable-local cuda-oxide 8192 u16-index fallback kernel launch proof",
    "current-C-shaped descending ordering and lower-index tie semantics",
]
EXPECTED_NOT_CLAIMED = [
    "CUB top-k selection or dispatch",
    "chunked top-k dispatch",
    "indexed ascending top-k sort dispatch",
    "runtime graph integration or default CUDA route",
    "C CUDA removal",
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
    report.check(fixture.get("schema") == "ds4.indexer_topk_pow2_kernel_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.2d2c2", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == RECORDED_DEPENDENCY_REVISION, "dependency revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-kernels", "kernel feature drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "crate dependency revision pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "lockfile dependency revision pin missing")
    report.check('name = "ds4-cuda-indexer-topk-pow2-smoke"' in texts["cargo"], "smoke binary wiring missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>",
        "indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>",
        "indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>",
        "__shared__ uint32_t idxs[SORT_N];",
        "__shared__ uint16_t idxs[SORT_N];",
        "topk_score_better(bv, bi, av, ai)",
        'getenv("DS4_CUDA_NO_TOPK8192") == NULL',
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_indexer_topk_pow2_2048_kernel", True),
        ("owns_indexer_topk_pow2_4096_kernel", True),
        ("owns_indexer_topk_pow2_u16_8192_kernel", True),
        ("owns_cub_topk_dispatch", False),
        ("owns_chunked_topk_dispatch", False),
        ("owns_indexed_topk_sort_dispatch", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_2D2C2_SCOPE",
        "owns_indexer_topk_pow2_2048_kernel: true",
        "owns_indexer_topk_pow2_4096_kernel: true",
        "owns_indexer_topk_pow2_u16_8192_kernel: true",
        "owns_cub_topk_dispatch: false",
        "owns_chunked_topk_dispatch: false",
        "owns_indexed_topk_sort_dispatch: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 34, "feature test count drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-indexer-topk-pow2-smoke" in command, "smoke command missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "smoke command forces a device target")
    report.check("CUDA_OXIDE_LINK_TARGET" not in command, "smoke command forces a link target")
    report.check(execution.get("backend_selected_target") == "sm_80", "portable backend target drift")
    expected = {
        "milestone": "M14.2d2c2",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "topk2048_output_matches": True,
        "topk4096_output_matches": True,
        "topk8192_u16_output_matches": True,
        "stable_tie_order_matches": True,
        "partial_component_output_matches": True,
        "invalid_shape_rejected": True,
        "owns_indexer_topk_pow2_2048_kernel": True,
        "owns_indexer_topk_pow2_4096_kernel": True,
        "owns_indexer_topk_pow2_u16_8192_kernel": True,
        "owns_cub_topk_dispatch": False,
        "owns_chunked_topk_dispatch": False,
        "owns_indexed_topk_sort_dispatch": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 power-of-two top-k result drift")
    for marker in [
        "pub fn indexer_topk_pow2_2048_kernel",
        "pub fn indexer_topk_pow2_4096_kernel",
        "pub fn indexer_topk_pow2_u16_8192_kernel",
        "static mut INDICES: SharedArray<u32, SORT_2048>",
        "static mut INDICES: SharedArray<u32, SORT_4096>",
        "static mut INDICES: SharedArray<u16, SORT_8192>",
        "let b_better = bv > av || (bv == av && bi < ai);",
        "topk8192_u16_output_matches",
        "Err(IndexerTopkPow2Error::InvalidShape)",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.2d2c2/indexer-topk-pow2-kernel-smoke.json"
    checker = "check_indexer_topk_pow2_kernel_smoke.py"
    item = "M14.2d2c2: Power-Of-Two Top-K Kernels"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14." in texts["status"], "top-k active stage missing")
    report.check("M14.2d2c2 Power-Of-Two Top-K Kernels" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("2048 result absent", lambda value: value["b300_execution"]["stdout"].update({"topk2048_output_matches": False})),
        ("4096 result absent", lambda value: value["b300_execution"]["stdout"].update({"topk4096_output_matches": False})),
        ("8192 result absent", lambda value: value["b300_execution"]["stdout"].update({"topk8192_u16_output_matches": False})),
        ("tie result absent", lambda value: value["b300_execution"]["stdout"].update({"stable_tie_order_matches": False})),
        ("8192 claim absent", lambda value: value["ownership"].update({"owns_indexer_topk_pow2_u16_8192_kernel": False})),
        ("CUB overclaim", lambda value: value["ownership"].update({"owns_cub_topk_dispatch": True})),
        ("chunked overclaim", lambda value: value["ownership"].update({"owns_chunked_topk_dispatch": True})),
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
    print(f"M14.2d2c2 power-of-two top-k kernel smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
