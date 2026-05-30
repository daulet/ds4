#!/usr/bin/env python3
"""Validate the M14.1c cuda-oxide substrate closure and route policy."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "ds4-parity/baselines/backend/m14.1c/substrate-route-closure.json"
INVENTORY = ROOT / "ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json"
SUBSTRATE = ROOT / "rust/ds4-cuda/src/substrate.rs"
FILL_SMOKE = ROOT / "rust/ds4-cuda/src/bin/fill_lifetime_smoke.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_SOURCES = {
    "ownership_inventory": "ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json",
    "host_substrate": "ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json",
    "model_residency": "ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json",
    "model_range_copy": "ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json",
    "model_range_strategy": "ds4-parity/baselines/backend/m14.1b2b1/model-range-strategy-smoke.json",
    "registered_range": "ds4-parity/baselines/backend/m14.1b2b2/model-registered-range-smoke.json",
    "pageable_hmm": "ds4-parity/baselines/backend/m14.1b2b3a/model-pageable-hmm-smoke.json",
    "direct_io": "ds4-parity/baselines/backend/m14.1b2b3b1/model-direct-io-smoke.json",
    "async_staging": "ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json",
    "model_map_closure": "ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json",
    "allocation_policy": "ds4-parity/baselines/backend/m14.1b3a/allocation-policy-smoke.json",
    "q8_quality_policy": "ds4-parity/baselines/backend/m14.1b3b/q8-quality-policy-smoke.json",
    "fill_command_lifetime": "ds4-parity/baselines/backend/m14.1b4/fill-command-lifetime-smoke.json",
}
EXPECTED_MILESTONES = {
    "host_substrate": "M14.1a",
    "model_residency": "M14.1b1",
    "model_range_copy": "M14.1b2a",
    "model_range_strategy": "M14.1b2b1",
    "registered_range": "M14.1b2b2",
    "pageable_hmm": "M14.1b2b3a",
    "direct_io": "M14.1b2b3b1",
    "async_staging": "M14.1b2b3b2",
    "model_map_closure": "M14.1b2c",
    "allocation_policy": "M14.1b3a",
    "q8_quality_policy": "M14.1b3b",
    "fill_command_lifetime": "M14.1b4",
}
DEFERRED_TO_M14_3 = [
    "ds4_gpu_cache_q8_f16_range",
    "dequant_q8_0_to_f16_kernel",
    "dequant_q8_0_to_f32_kernel",
]
EXPECTED_CLOSED_CAPABILITIES = [
    "context, stream, device and managed allocation substrate",
    "model-window and staged range-cache lifetime policy",
    "managed-KV, memory-report, and Q8 admission or quality policy",
    "fill_f32 launch and current-C-shaped command completion surface",
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
    closure = load_json(CLOSURE)
    inventory = load_json(INVENTORY)
    artifacts = {
        name: load_json(ROOT / path)
        for name, path in EXPECTED_SOURCES.items()
        if name != "ownership_inventory"
    }
    texts = {
        "substrate": read_text(SUBSTRATE),
        "fill_smoke": read_text(FILL_SMOKE),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
        "readme": read_text(README),
        "report": read_text(REPORT),
    }
    report = Report()
    validate(report, closure, inventory, artifacts, texts)
    if args.negative_test:
        run_negative_tests(report, closure, inventory, artifacts, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    closure: dict[str, Any],
    inventory: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
) -> None:
    validate_root(report, closure)
    validate_sources(report, closure, artifacts)
    validate_inventory_refresh(report, closure, inventory)
    validate_non_claims(report, closure, artifacts)
    validate_claim_policy(report, closure)
    validate_b300_contract(report, closure, artifacts)
    validate_static_wiring(report, texts)


def validate_root(report: Report, closure: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.cuda_oxide_substrate_closure.v1",
        "milestone": "M14.1c",
        "status": "closure-no-route-promotion",
        "next_stage": "M14.2 Embedding Indexer And Elementwise Kernels",
    }
    for key, value in expected.items():
        report.check(closure.get(key) == value, f"closure {key} drift")
    report.check(
        closure.get("closed_resource_capabilities") == EXPECTED_CLOSED_CAPABILITIES,
        "closed capability boundary drift",
    )


def validate_sources(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    report.check(closure.get("source_artifacts") == EXPECTED_SOURCES, "source artifact map drift")
    for name, path in EXPECTED_SOURCES.items():
        report.check((ROOT / path).exists(), f"source artifact missing: {name}")
    for name, milestone in EXPECTED_MILESTONES.items():
        artifact = artifacts[name]
        ownership = require_dict(report, artifact.get("ownership"), f"{name}.ownership")
        report.check(artifact.get("milestone") == milestone, f"{name}: milestone drift")
        report.check(artifact.get("status") == "b300-pass", f"{name}: status drift")
        report.check(ownership.get("opt_in_only") is True, f"{name}: opt-in ownership drift")
        report.check(ownership.get("changes_default_route") is False, f"{name}: route overclaim")
        report.check(
            ownership.get("retains_current_c_cuda_oracle") is True,
            f"{name}: current-C oracle retention drift",
        )


def validate_inventory_refresh(
    report: Report,
    closure: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    refresh = require_dict(report, closure.get("inventory_refresh"), "inventory_refresh")
    report.check(refresh.get("reassigned_to_m14_3") == DEFERRED_TO_M14_3, "deferred assignment list drift")
    report.check(refresh.get("m14_1_kernel_symbols") == ["fill_f32_kernel"], "M14.1 kernel closure drift")
    m14_1_abi = family_symbols(inventory, "abi_families", "M14.1")
    m14_3_abi = family_symbols(inventory, "abi_families", "M14.3")
    m14_1_kernels = family_symbols(inventory, "kernel_families", "M14.1")
    m14_3_kernels = family_symbols(inventory, "kernel_families", "M14.3")
    report.check("ds4_gpu_cache_q8_f16_range" not in m14_1_abi, "Q8 cache still assigned to M14.1")
    report.check("ds4_gpu_cache_q8_f16_range" in m14_3_abi, "Q8 cache not assigned to M14.3")
    report.check(m14_1_kernels == {"fill_f32_kernel"}, "M14.1 kernel assignment not closed to fill")
    for kernel in DEFERRED_TO_M14_3[1:]:
        report.check(kernel not in m14_1_kernels, f"{kernel}: still assigned to M14.1")
        report.check(kernel in m14_3_kernels, f"{kernel}: not assigned to M14.3")


def validate_non_claims(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    non_claims = require_dict(report, closure.get("verified_non_claims"), "verified_non_claims")
    q8_expected = {"owns_converted_q8_buffers": False, "owns_dequant_kernels": False}
    fill_expected = {"owns_dequant_kernels": False, "owns_graph_kernels": False}
    report.check(non_claims.get("q8_quality_policy") == q8_expected, "Q8 non-claim drift")
    report.check(non_claims.get("fill_command_lifetime") == fill_expected, "fill non-claim drift")
    q8 = artifacts["q8_quality_policy"]["ownership"]
    fill = artifacts["fill_command_lifetime"]["ownership"]
    for key, expected in q8_expected.items():
        report.check(q8.get(key) is expected, f"Q8 artifact overclaim: {key}")
    for key, expected in fill_expected.items():
        report.check(fill.get(key) is expected, f"fill artifact overclaim: {key}")


def validate_claim_policy(report: Report, closure: dict[str, Any]) -> None:
    policy = require_dict(report, closure.get("claim_policy"), "claim_policy")
    for key, expected in [
        ("m14_1_substrate_available_to_following_kernel_stages", True),
        ("opt_in_only", True),
        ("default_route_replacement_active", False),
        ("ds4_cuda_removal_allowed", False),
        ("current_c_cuda_retained_as_oracle", True),
        ("q8_conversion_and_dequant_deferred_to_m14_3", True),
    ]:
        report.check(policy.get(key) is expected, f"claim policy drift: {key}")


def validate_b300_contract(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    rerun = require_dict(report, closure.get("b300_rerun_contract"), "b300_rerun_contract")
    fill_execution = artifacts["fill_command_lifetime"]["b300_execution"]
    report.check(rerun.get("device_name") == "NVIDIA B300 SXM6 AC", "B300 device drift")
    report.check("--features cuda-oxide-backend" in rerun.get("host_feature_command", ""), "host rerun command missing")
    fill_command = rerun.get("fill_kernel_command", "")
    report.check("--features cuda-oxide-kernels" in fill_command, "fill rerun feature missing")
    report.check("--bin ds4-cuda-fill-lifetime-smoke" in fill_command, "fill rerun binary missing")
    report.check("CUDA_OXIDE_TARGET" not in fill_command, "fill rerun forces a device target")
    report.check(rerun.get("backend_selected_target") == "sm_80", "portable backend target drift")
    report.check(rerun.get("fill_stdout") == fill_execution.get("stdout"), "fill rerun output drift")
    report.check(rerun["fill_stdout"].get("begin_is_noop") is True, "begin-command no-op not proved")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.1c/substrate-route-closure.json"
    checker = "check_substrate_route_closure.py"
    report.check("pub fn begin_commands(&self) {}" in texts["substrate"], "begin command facade missing")
    report.check("substrate.begin_commands();" in texts["fill_smoke"], "begin command smoke invocation missing")
    report.check("M14.1c: Substrate Route Closure Gate" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1c: Substrate Route Closure Gate" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.2" in texts["status"], "next active stage missing")
    report.check("M14.1c Substrate Route Closure Gate" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(
    report: Report,
    closure: dict[str, Any],
    inventory: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    texts: dict[str, str],
) -> None:
    for label, mutate in [
        ("route promoted", lambda item: item["claim_policy"].update({"default_route_replacement_active": True})),
        ("removal allowed", lambda item: item["claim_policy"].update({"ds4_cuda_removal_allowed": True})),
        ("Q8 reassignment omitted", lambda item: item["inventory_refresh"].update({"reassigned_to_m14_3": []})),
        ("Q8 conversion overclaimed", lambda item: item["verified_non_claims"]["q8_quality_policy"].update({"owns_converted_q8_buffers": True})),
        ("begin no-op evidence lost", lambda item: item["b300_rerun_contract"]["fill_stdout"].update({"begin_is_noop": False})),
    ]:
        candidate = copy.deepcopy(closure)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, inventory, artifacts, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


def family_symbols(inventory: dict[str, Any], family_name: str, stage: str) -> set[str]:
    for family in inventory.get(family_name, []):
        if family.get("stage") == stage:
            return set(family.get("symbols", []))
    return set()


def require_dict(report: Report, value: Any, name: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{name} must be an object")
    return value if isinstance(value, dict) else {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"M14.1c substrate route closure: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
