#!/usr/bin/env python3
"""Validate the M14.4c3a Rust CUDA compressor update orchestration smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.4c3a/compressor-update-orchestration-smoke.json"
CARGO = ROOT / "rust/ds4-cuda/Cargo.toml"
LOCK = ROOT / "Cargo.lock"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/compressor_update_orchestration_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

DEPENDENCY_REVISION = "485bdd86fc1c900ad15ebd421b3b187619fe0903"
CURRENT_DEPENDENCY_REVISION = "ae721dc95912a918f182d13b7ca55281aa29d8f9"
EXPECTED_OWNED = [
    "executable-local cuda-oxide compressor update orchestration launch proof",
    "non-emitting store-only and emitting store, pool, weighted RMS, RoPE, shift sequence validation",
    "ratio-4 F16 APE, YARN RoPE, comp-row offset, and general-ratio emitted numeric validation",
]
EXPECTED_NOT_CLAIMED = [
    "compressor prefill or replay wrapper orchestration",
    "FP8 compressed-cache composition or attention kernels",
    "runtime graph integration, default CUDA route, or C CUDA removal",
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
    print(f"M14.4c3a compressor update orchestration smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.compressor_update_orchestration_smoke.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == "M14.4c3a", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("dependency_revision") == DEPENDENCY_REVISION, "revision drift")
    report.check(f'rev = "{CURRENT_DEPENDENCY_REVISION}"' in texts["cargo"], "dependency pin missing")
    report.check(f"#{CURRENT_DEPENDENCY_REVISION}" in texts["lock"], "lock pin missing")
    report.check(
        'name = "ds4-cuda-compressor-update-orchestration-smoke"' in texts["cargo"],
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
        'extern "C" int ds4_gpu_compressor_update_tensor(',
        "if (!emit) return 1;",
        "compressor_update_pool_kernel<<<",
        "ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,",
        "ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,",
        "compressor_shift_ratio4_kernel<<<",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_OWNED, "owned scope drift")
    report.check(
        ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED,
        "non-claim scope drift",
    )
    for key, expected in [
        ("opt_in_only", True),
        ("owns_compressor_update_orchestration", True),
        ("owns_store_pool_norm_rope_shift_sequence", True),
        ("owns_compressor_prefill_orchestration", False),
        ("owns_attention_kernels", False),
        ("owns_runtime_graph_integration", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_4C3A_SCOPE",
        "owns_compressor_update_orchestration: true",
        "owns_store_pool_norm_rope_shift_sequence: true",
        "owns_compressor_prefill_orchestration: false",
        "owns_attention_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("test_count") == 55, "feature test count drift")
    report.check(execution.get("backend_selected_target") == "sm_80", "target drift")
    report.check(execution.get("uses_libdevice_link_path") is True, "libdevice proof missing")
    command = execution.get("command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel command missing")
    report.check(
        "--bin ds4-cuda-compressor-update-orchestration-smoke" in command,
        "smoke command missing",
    )
    expected = {
        "milestone": "M14.4c3a",
        "device_name": "NVIDIA B300 SXM6 AC",
        "rust_kernel_toolchain": True,
        "ratio4_no_emit_store_only_matches": True,
        "ratio4_emit_composed_output_matches": True,
        "general_ratio_emit_composed_output_matches": True,
        "weighted_rms_composition_matches": True,
        "rope_composition_matches": True,
        "ratio4_shift_after_emit_matches": True,
        "f16_ape_update_matches": True,
        "invalid_shape_rejected": True,
        "uses_libdevice_link_path": True,
        "owns_compressor_update_orchestration": True,
        "owns_store_pool_norm_rope_shift_sequence": True,
        "owns_compressor_prefill_orchestration": False,
        "owns_attention_kernels": False,
        "owns_runtime_graph_integration": False,
        "changes_default_route": False,
    }
    report.check(require_dict(report, execution.get("stdout"), "stdout") == expected, "stdout drift")
    for marker in [
        "pub fn compressor_store_kernel",
        "pub fn compressor_update_pool_kernel",
        "pub fn rms_norm_weight_kernel",
        "pub fn rope_tail_kernel",
        "pub fn compressor_shift_ratio4_kernel",
        "fn compressor_update_tensor(",
        "expected_update(",
        "COMP_ROW",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.4c3a/compressor-update-orchestration-smoke.json"
    checker = "check_compressor_update_orchestration_smoke.py"
    item = "M14.4c3a: Compressor Update Orchestration"
    report.check(item in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check(item in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("M14.4c3b Compressor Prefill And Replay Orchestration" in texts["status"], "successor evidence missing")
    report.check(item.replace(":", "") in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        (
            "no-emit output absent",
            lambda value: value["b300_execution"]["stdout"].update({"ratio4_no_emit_store_only_matches": False}),
        ),
        (
            "rope output absent",
            lambda value: value["b300_execution"]["stdout"].update({"rope_composition_matches": False}),
        ),
        (
            "prefill overclaim",
            lambda value: value["ownership"].update({"owns_compressor_prefill_orchestration": True}),
        ),
        (
            "attention overclaim",
            lambda value: value["ownership"].update({"owns_attention_kernels": True}),
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
