#!/usr/bin/env python3
"""Validate the M14.5c2c3 Rust CUDA routed MoE tile4 row32 smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.5c2c3/routed-moe-tile4-row32-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/routed_moe_tile8_row32_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
EXPECTED_OWNED = [
    "executable-local functional tile4 row32 IQ2-XXS/Q8_K gate/up projection proof",
    "executable-local functional tile4 row32 Q2_K/Q8_K non-atomic down projection proof",
    "optional DS4_CUDA_MOE_TILE4 row32 projection dispatch over expert-tile metadata",
]
EXPECTED_NOT_CLAIMED = [
    "shared-cache optimization, atomic-down, tile16, or rowspan dispatch",
    "Q4_K, hyperconnection, runtime graph integration, default CUDA route, or C CUDA removal",
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
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.5c2c3 routed MoE tile4 row32 smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.routed_moe_tile4_row32_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.5c2c3", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == DEPENDENCY_REVISION, "revision drift")
    report.check(f'rev = "{DEPENDENCY_REVISION}"' in texts["cargo"], "dependency pin missing")
    report.check(f"#{DEPENDENCY_REVISION}" in texts["lock"], "lock pin missing")
    report.check('name = "ds4-cuda-routed-moe-tile8-row32-smoke"' in texts["cargo"], "binary missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(",
        "__global__ static void moe_down_expert_tile4_row32_kernel(",
        'const uint32_t expert_tile_m = getenv("DS4_CUDA_MOE_TILE4") ? 4u : 8u;',
        "moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>",
        "moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>",
        "down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim drift")
    for key, expected in [
        ("opt_in_only", True),
        ("consumes_expert_tile_metadata_surface", True),
        ("uses_previously_owned_q8_k_inputs", True),
        ("owns_moe_gate_up_mid_expert_tile4_row32_kernel", True),
        ("owns_moe_down_expert_tile4_row32_non_atomic_surface", True),
        ("owns_optional_tile4_row32_projection_dispatch", True),
        ("owns_atomic_down_or_rowspan_dispatch", False),
        ("owns_shared_cache_specialization", False),
        ("owns_q4_k_or_runtime_graph", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_5C2C3_SCOPE",
        "owns_moe_gate_up_mid_expert_tile4_row32_kernel: true",
        "owns_moe_down_expert_tile4_row32_non_atomic_surface: true",
        "owns_optional_tile4_row32_projection_dispatch: true",
        "owns_atomic_down_or_rowspan_dispatch: false",
        "owns_shared_cache_specialization: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 79, "feature test count drift")
    report.check(execution.get("backend_selected_target") == "sm_80", "target drift")
    command = execution.get("command", "")
    report.check("DS4_CUDA_MOE_TILE4=1" in command, "tile4 selector missing")
    report.check("--features cuda-oxide-kernels" in command, "kernel command missing")
    report.check("--bin ds4-cuda-routed-moe-tile8-row32-smoke" in command, "smoke command missing")
    expected = {
        "milestone": "M14.5c2c3",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "tile4_gate_up_matches": True,
        "tile4_down_matches": True,
        "three_tile_expert_matches": True,
        "partial_tile_matches": True,
        "negative_expert_bucket_zero_matches": True,
        "invalid_shape_rejected": True,
        "uses_quarter_warp_shuffle_reduction": True,
        "uses_libdevice_link_path": True,
        "consumes_expert_tile_metadata_surface": True,
        "uses_previously_owned_q8_k_inputs": True,
        "owns_moe_gate_up_mid_expert_tile4_row32_kernel": True,
        "owns_moe_down_expert_tile4_row32_non_atomic_surface": True,
        "owns_optional_tile4_row32_projection_dispatch": True,
        "owns_atomic_down_or_rowspan_dispatch": False,
        "owns_shared_cache_specialization": False,
        "owns_q4_k_or_runtime_graph": False,
        "changes_default_route": False,
    }
    report.check(require_dict(report, execution.get("stdout"), "stdout") == expected, "stdout drift")
    for marker in [
        "pub fn moe_gate_up_mid_expert_tile4_row32_kernel",
        "pub fn moe_down_expert_tile4_row32_kernel",
        'std::env::var_os("DS4_CUDA_MOE_TILE4")',
        "expert_tile_metadata(&selected_values, if tile4 { 4 } else { 8 })",
        "while entry < 4",
        "quarter_warp_sum_f32",
        "expected_gate_up_mid",
        "expected_down",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.5c2c3/routed-moe-tile4-row32-smoke.json"
    checker = "check_routed_moe_tile4_row32_smoke.py"
    item = "M14.5c2c3: Tile4 Row32 Projection"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.5c2c7 Down Tile16 Rowspan Projection" in texts["status"], "next active missing")
    report.check(item.replace(":", "") in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("gate output absent", lambda value: value["b300_execution"]["stdout"].update({"tile4_gate_up_matches": False})),
        ("down output absent", lambda value: value["b300_execution"]["stdout"].update({"tile4_down_matches": False})),
        ("atomic down overclaim", lambda value: value["ownership"].update({"owns_atomic_down_or_rowspan_dispatch": True})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
