#!/usr/bin/env python3
"""Validate the M14.2e cuda-oxide kernel-family closure and non-claims."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "ds4-parity/baselines/backend/m14.2e/kernel-ownership-closure.json"
INVENTORY = ROOT / "ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

EXPECTED_SOURCES = {
    "ownership_inventory": "ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json",
    "elementwise": "ds4-parity/baselines/backend/m14.2a/elementwise-kernel-smoke.json",
    "directional_steering": "ds4-parity/baselines/backend/m14.2b1/directional-steering-kernel-smoke.json",
    "swiglu": "ds4-parity/baselines/backend/m14.2b2/swiglu-kernel-smoke.json",
    "embedding": "ds4-parity/baselines/backend/m14.2c/embedding-kernel-smoke.json",
    "scalar_indexer": "ds4-parity/baselines/backend/m14.2d1/indexer-scalar-kernel-smoke.json",
    "direct_indexer": "ds4-parity/baselines/backend/m14.2d2a/indexer-direct-kernel-smoke.json",
    "wmma": "ds4-parity/baselines/backend/m14.2d2b1/indexer-wmma-kernel-smoke.json",
    "wmma32": "ds4-parity/baselines/backend/m14.2d2b2a/indexer-wmma32-kernel-smoke.json",
    "wmma64": "ds4-parity/baselines/backend/m14.2d2b2b/indexer-wmma64-kernel-smoke.json",
    "wmma128_dispatch": "ds4-parity/baselines/backend/m14.2d2b2c/indexer-wmma128-dispatch-smoke.json",
    "topk1024": "ds4-parity/baselines/backend/m14.2d2c1/indexer-topk1024-kernel-smoke.json",
    "topk_pow2": "ds4-parity/baselines/backend/m14.2d2c2/indexer-topk-pow2-kernel-smoke.json",
    "topk_packed": "ds4-parity/baselines/backend/m14.2d2c3/indexer-topk-packed-kernel-smoke.json",
    "topk_tree": "ds4-parity/baselines/backend/m14.2d2c4/indexer-topk-tree-kernel-smoke.json",
    "topk_dispatch": "ds4-parity/baselines/backend/m14.2d2c5/indexer-topk-dispatch-smoke.json",
}
EXPECTED_MILESTONES = {
    "elementwise": "M14.2a",
    "directional_steering": "M14.2b1",
    "swiglu": "M14.2b2",
    "embedding": "M14.2c",
    "scalar_indexer": "M14.2d1",
    "direct_indexer": "M14.2d2a",
    "wmma": "M14.2d2b1",
    "wmma32": "M14.2d2b2a",
    "wmma64": "M14.2d2b2b",
    "wmma128_dispatch": "M14.2d2b2c",
    "topk1024": "M14.2d2c1",
    "topk_pow2": "M14.2d2c2",
    "topk_packed": "M14.2d2c3",
    "topk_tree": "M14.2d2c4",
    "topk_dispatch": "M14.2d2c5",
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
    closure = load_json(CLOSURE)
    inventory = load_json(INVENTORY)
    artifacts = {
        name: load_json(ROOT / path)
        for name, path in EXPECTED_SOURCES.items()
        if name != "ownership_inventory"
    }
    texts = {
        "cuda": read_text(CUDA_SOURCE),
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
    validate_inventory_refresh(report, closure, inventory, texts["cuda"])
    validate_equivalent_implementation(report, closure, artifacts)
    validate_claim_policy(report, closure)
    validate_b300_contract(report, closure, artifacts)
    validate_static_wiring(report, texts)


def validate_root(report: Report, closure: dict[str, Any]) -> None:
    for key, expected in [
        ("schema", "ds4.cuda_oxide_m14_2_kernel_closure.v1"),
        ("milestone", "M14.2e"),
        ("status", "closure-no-route-promotion"),
        ("next_stage", "M14.3 Dense Projection Quantization And Norm Kernels"),
    ]:
        report.check(closure.get(key) == expected, f"closure {key} drift")
    report.check(len(closure.get("closed_kernel_capabilities", [])) == 4, "closed capability boundary drift")


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
    cuda_source: str,
) -> None:
    refresh = require_dict(report, closure.get("inventory_refresh"), "inventory_refresh")
    report.check(refresh.get("reassigned_to_m14_5") == ["zero_kernel"], "zero-kernel reassignment drift")
    m14_2 = family_symbols(inventory, "kernel_families", "M14.2")
    m14_5 = family_symbols(inventory, "kernel_families", "M14.5")
    report.check(set(refresh.get("m14_2_kernel_symbols", [])) == m14_2, "M14.2 kernel closure list drift")
    report.check("zero_kernel" not in m14_2, "zero kernel remains assigned to M14.2")
    report.check("zero_kernel" in m14_5, "zero kernel is not assigned to M14.5")
    report.check("zero_kernel<<<(n + 255u) / 256u, 256>>>" in cuda_source, "current-C zero launch missing")
    report.check("routed_moe atomic zero launch" in cuda_source, "routed-MoE zero ownership marker missing")


def validate_equivalent_implementation(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    equivalent = require_dict(report, closure.get("equivalent_implementation"), "equivalent_implementation")
    packed = artifacts["topk_packed"]["ownership"]
    dispatch = artifacts["topk_dispatch"]["ownership"]
    report.check(equivalent.get("current_c_kernel") == "indexer_topk_8192_cub_kernel", "CUB kernel identity drift")
    report.check(
        equivalent.get("rust_kernel") == "indexer_topk_8192_packed_key_equivalent_kernel",
        "packed equivalent identity drift",
    )
    report.check(
        equivalent.get("preserves_ordered_float_and_lower_index_key_semantics") is True,
        "packed-key semantic proof missing",
    )
    report.check(equivalent.get("owns_cub_library_implementation") is False, "CUB overclaim")
    report.check(packed.get("owns_cub_library_implementation") is False, "packed artifact CUB overclaim")
    report.check(dispatch.get("owns_cub_library_implementation") is False, "dispatch artifact CUB overclaim")


def validate_claim_policy(report: Report, closure: dict[str, Any]) -> None:
    policy = require_dict(report, closure.get("claim_policy"), "claim_policy")
    for key, expected in [
        ("m14_2_kernels_available_to_following_operation_stages", True),
        ("opt_in_only", True),
        ("default_route_replacement_active", False),
        ("ds4_cuda_removal_allowed", False),
        ("current_c_cuda_retained_as_oracle", True),
        ("zero_kernel_deferred_to_routed_moe_m14_5", True),
        ("cub_library_implementation_owned", False),
    ]:
        report.check(policy.get(key) is expected, f"claim policy drift: {key}")


def validate_b300_contract(
    report: Report,
    closure: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    rerun = require_dict(report, closure.get("b300_rerun_contract"), "b300_rerun_contract")
    latest = artifacts["topk_dispatch"]["b300_execution"]
    report.check(rerun.get("device_name") == "NVIDIA B300 SXM6 AC", "B300 device drift")
    report.check(rerun.get("host_feature_test_count") == 38, "B300 test count drift")
    report.check("--features cuda-oxide-backend" in rerun.get("host_feature_command", ""), "host command missing")
    command = rerun.get("latest_kernel_command", "")
    report.check("--features cuda-oxide-kernels" in command, "kernel feature command missing")
    report.check("--bin ds4-cuda-indexer-topk-dispatch-smoke" in command, "kernel binary missing")
    report.check("CUDA_OXIDE_TARGET" not in command, "kernel command forces a device target")
    report.check(rerun.get("backend_selected_target") == "sm_80", "portable backend target drift")
    report.check(rerun.get("latest_kernel_stdout") == latest.get("stdout"), "latest B300 output drift")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.2e/kernel-ownership-closure.json"
    checker = "check_m14_2_kernel_closure.py"
    report.check("M14.2e: M14.2 Kernel Closure Gate" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.2e: M14.2 Kernel Closure Gate" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.3 Dense Projection Quantization And Norm Kernels" in texts["status"], "next active stage missing")
    report.check("M14.2e M14.2 Kernel Closure Gate" in texts["status"], "status evidence missing")
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
        ("zero deferral omitted", lambda item: item["inventory_refresh"].update({"reassigned_to_m14_5": []})),
        ("CUB overclaim", lambda item: item["equivalent_implementation"].update({"owns_cub_library_implementation": True})),
        ("B300 output lost", lambda item: item["b300_rerun_contract"]["latest_kernel_stdout"].update({"topk_dispatch_priority_matches": False})),
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
    print(f"M14.2e CUDA kernel closure: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
