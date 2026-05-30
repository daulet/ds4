#!/usr/bin/env python3
"""Validate the M14.5c1 Rust CUDA routed MoE F32-activation fallback smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.5c1/routed-moe-f32-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/routed_moe_f32_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
IQ2_TABLES = ROOT / "ds4_iq2_tables_cuda.inc"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
EXPECTED_OWNED = [
    "executable-local packed IQ2-XXS gate/up and Q2_K down decode proof",
    "current-C F32-activation gate/up, down, and expert-sum fallback kernels",
    "single-token and batched routed MoE fallback layout with clamp and negative-expert behavior",
]
EXPECTED_NOT_CLAIMED = [
    "Q8 activation, optimized routed-MoE dispatch, or Q4_K routed-MoE path",
    "hyperconnection, runtime graph integration, default CUDA route, or C CUDA removal",
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
        "iq2_tables": IQ2_TABLES.read_text(encoding="utf-8"),
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
    print(f"M14.5c1 routed MoE F32 fallback smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.routed_moe_f32_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.5c1", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == DEPENDENCY_REVISION, "revision drift")
    report.check(f'rev = "{DEPENDENCY_REVISION}"' in texts["cargo"], "dependency pin missing")
    report.check(f"#{DEPENDENCY_REVISION}" in texts["lock"], "lock pin missing")
    report.check('name = "ds4-cuda-routed-moe-f32-smoke"' in texts["cargo"], "binary missing")
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "__device__ static float dev_iq2_xxs_dot_f32(",
        "__device__ static float dev_q2_K_dot_f32(",
        "__global__ static void moe_gate_up_mid_f32_kernel(",
        "__global__ static void moe_down_f32_kernel(",
        "__global__ static void moe_sum_kernel(",
        "if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes)",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")
    for marker in [
        "__device__ __constant__ uint8_t cuda_ksigns_iq2xs[128]",
        "0x0808080808080808",
        "0x080808080808082b",
        "0x0808080808081919",
        "0x0808080808082b08",
    ]:
        report.check(marker in texts["iq2_tables"], f"IQ2 lookup marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_OWNED, "owned scope drift")
    report.check(
        ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED,
        "non-claim scope drift",
    )
    for key, expected in [
        ("opt_in_only", True),
        ("consumes_router_selection_surface", True),
        ("owns_iq2_xxs_f32_gate_up_dot", True),
        ("owns_q2_k_f32_down_dot", True),
        ("owns_moe_gate_up_mid_f32_kernel", True),
        ("owns_moe_down_f32_kernel", True),
        ("owns_moe_sum_kernel", True),
        ("owns_single_and_batch_f32_activation_moe_surface", True),
        ("owns_q8_activation_or_optimized_moe_dispatch", False),
        ("owns_hyperconnection_or_runtime_graph", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_5C1_SCOPE",
        "owns_iq2_xxs_f32_gate_up_dot: true",
        "owns_q2_k_f32_down_dot: true",
        "owns_moe_gate_up_mid_f32_kernel: true",
        "owns_moe_down_f32_kernel: true",
        "owns_moe_sum_kernel: true",
        "owns_q8_activation_or_optimized_moe_dispatch: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 73, "feature test count drift")
    report.check(execution.get("backend_selected_target") == "sm_80", "target drift")
    report.check(execution.get("uses_libdevice_link_path") is True, "libdevice proof missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel command missing")
    report.check("--bin ds4-cuda-routed-moe-f32-smoke" in command, "smoke command missing")
    expected = {
        "milestone": "M14.5c1",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "packed_iq2_gate_up_matches": True,
        "packed_q2_down_matches": True,
        "weighted_swiglu_matches": True,
        "expert_sum_matches": True,
        "negative_expert_fallback_matches": True,
        "single_token_surface_matches": True,
        "batch_surface_matches": True,
        "clamp_behavior_matches": True,
        "invalid_shape_rejected": True,
        "uses_shared_reduction": True,
        "uses_libdevice_link_path": True,
        "consumes_router_selection_surface": True,
        "owns_iq2_xxs_f32_gate_up_dot": True,
        "owns_q2_k_f32_down_dot": True,
        "owns_moe_gate_up_mid_f32_kernel": True,
        "owns_moe_down_f32_kernel": True,
        "owns_moe_sum_kernel": True,
        "owns_single_and_batch_f32_activation_moe_surface": True,
        "owns_q8_activation_or_optimized_moe_dispatch": False,
        "owns_hyperconnection_or_runtime_graph": False,
        "changes_default_route": False,
    }
    report.check(require_dict(report, execution.get("stdout"), "stdout") == expected, "stdout drift")
    for marker in [
        "pub fn moe_gate_up_mid_f32_kernel",
        "pub fn moe_down_f32_kernel",
        "pub fn moe_sum_kernel",
        "fn dev_iq2_xxs_dot_f32",
        "fn dev_q2_k_dot_f32",
        "const IQ2_GRIDS",
        "const IQ2_SIGNS",
        "static mut PARTIAL_GATE: SharedArray",
        "negative_expert_fallback_matches",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.5c1/routed-moe-f32-smoke.json"
    checker = "check_routed_moe_f32_smoke.py"
    item = "M14.5c1: Packed F32-Activation Routed MoE Fallback"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.5c2 Quantized Routed MoE Dispatch" in texts["status"],
        "next active stage missing",
    )
    report.check(item.replace(":", "") in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        (
            "packed down execution absent",
            lambda value: value["b300_execution"]["stdout"].update({"packed_q2_down_matches": False}),
        ),
        (
            "clamp behavior absent",
            lambda value: value["b300_execution"]["stdout"].update({"clamp_behavior_matches": False}),
        ),
        (
            "optimized MoE overclaim",
            lambda value: value["ownership"].update(
                {"owns_q8_activation_or_optimized_moe_dispatch": True}
            ),
        ),
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
