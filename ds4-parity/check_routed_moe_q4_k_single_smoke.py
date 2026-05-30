#!/usr/bin/env python3
"""Validate the M14.5c2d Rust CUDA single-token Q4_K routed MoE smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.5c2d/routed-moe-q4-k-single-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/routed_moe_quantized_single_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
EXPECTED_OWNED = [
    "executable-local Q4_K/Q8_K packed dot proof reusing Q8_K activation quantization",
    "single-token type-12 gate/up projection and direct six-expert Q4_K down output",
    "optional gate/up auxiliary write behavior and negative-expert fallback",
]
EXPECTED_NOT_CLAIMED = [
    "shared-cache expert-tile specialization",
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
    print(f"M14.5c2d single-token Q4_K routed MoE smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.routed_moe_q4_k_single_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.5c2d", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == DEPENDENCY_REVISION, "revision drift")
    report.check(f'rev = "{DEPENDENCY_REVISION}"' in texts["cargo"], "dependency pin missing")
    report.check(f"#{DEPENDENCY_REVISION}" in texts["lock"], "lock pin missing")
    report.check(
        'name = "ds4-cuda-routed-moe-quantized-single-smoke"' in texts["cargo"],
        "binary missing",
    )
    validate_oracle(report, fixture, texts)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "dev_q4_K_get_scale_min(",
        "dev_dot_q4_K_q8_K_block(",
        "__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(",
        "__global__ static void moe_down_q4K_sum6_qwarp32_kernel(",
        "const int q4k_path = (gate_type == 12u && down_type == 12u);",
        "if (q4k_path && (n_tokens != 1u || n_expert != 6u)) return 0;",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim drift")
    for key, expected in [
        ("opt_in_only", True),
        ("consumes_quantized_single_surface", True),
        ("owns_q4_k_q8_k_dot", True),
        ("owns_moe_gate_up_mid_decode_q4_k_qwarp32_kernel", True),
        ("owns_moe_down_q4_k_sum6_qwarp32_kernel", True),
        ("owns_single_token_type12_dispatch", True),
        ("owns_hyperconnection_or_runtime_graph", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_5C2D_SCOPE",
        "owns_q4_k_q8_k_dot: true",
        "owns_moe_gate_up_mid_decode_q4_k_qwarp32_kernel: true",
        "owns_moe_down_q4_k_sum6_qwarp32_kernel: true",
        "owns_single_token_type12_dispatch: true",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 84, "feature test count drift")
    report.check(execution.get("backend_selected_target") == "sm_80", "target drift")
    report.check(execution.get("uses_libdevice_link_path") is True, "libdevice proof missing")
    command = execution.get("command", "")
    report.check("DS4_CUDA_MOE_Q4_K=1" in command, "Q4_K selector missing")
    report.check("--features cuda-oxide-kernels" in command, "kernel command missing")
    expected = {
        "milestone": "M14.5c2d",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "q4_k_gate_up_matches": True,
        "q4_k_direct_sum6_down_matches": True,
        "single_token_type12_dispatch_matches": True,
        "optional_gate_up_write_matches": True,
        "negative_expert_fallback_matches": True,
        "q8_k_input_quantize_reused": True,
        "q8_k_mid_quantize_reused": True,
        "zero_quantize_matches": True,
        "invalid_shape_rejected": True,
        "uses_quarter_warp_shuffle_reduction": True,
        "uses_libdevice_link_path": True,
        "consumes_quantized_single_surface": True,
        "owns_q4_k_q8_k_dot": True,
        "owns_moe_gate_up_mid_decode_q4_k_qwarp32_kernel": True,
        "owns_moe_down_q4_k_sum6_qwarp32_kernel": True,
        "owns_single_token_type12_dispatch": True,
        "owns_hyperconnection_or_runtime_graph": False,
        "changes_default_route": False,
    }
    report.check(require_dict(report, execution.get("stdout"), "stdout") == expected, "stdout drift")
    for marker in [
        "pub fn moe_gate_up_mid_decode_q4_k_qwarp32_kernel",
        "pub fn moe_down_q4_k_sum6_qwarp32_kernel",
        "fn dev_dot_q4_k_q8_k_block",
        'std::env::var_os("DS4_CUDA_MOE_Q4_K")',
        "packed_q4_k_weights",
        "M14_5C2D_SCOPE",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.5c2d/routed-moe-q4-k-single-smoke.json"
    checker = "check_routed_moe_q4_k_single_smoke.py"
    item = "M14.5c2d: Single-Token Q4_K Routed MoE"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.6 CUDA Route Promotion And C CUDA Removal Gate" in texts["status"],
        "next active missing",
    )
    report.check(item.replace(":", "") in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("Q4_K gate/up absent", lambda value: value["b300_execution"]["stdout"].update({"q4_k_gate_up_matches": False})),
        ("Q4_K direct down absent", lambda value: value["b300_execution"]["stdout"].update({"q4_k_direct_sum6_down_matches": False})),
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
