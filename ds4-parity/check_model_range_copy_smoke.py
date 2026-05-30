#!/usr/bin/env python3
"""Validate the M14.1b2a Rust-owned model mmap/device-range copy smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
MODEL_MAP = ROOT / "rust/ds4-cuda/src/model_map.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/model_range_copy_smoke.rs"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

REVISION = "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200"
MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
MODEL_SIZE = 86720111488
RANGE_BYTES = 4096
EXPECTED_RUST_OWNED = [
    "model file and mmap lifetime",
    "bounds-checked model range selection",
    "CUDA device-buffer copy cache for one selected model range",
    "exact copied-range readback and cache-entry reuse",
]
EXPECTED_NOT_CLAIMED = [
    "registered, HMM, or direct-I/O range-strategy selection",
    "whole-model registration or whole-model device copy",
    "Q8/F16 range-cache conversion policy",
    "model-range consumption by DS4 compute kernels",
    "runtime graph or default CUDA route",
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
        "lib": CRATE_LIB.read_text(encoding="utf-8"),
        "model_map": MODEL_MAP.read_text(encoding="utf-8"),
        "smoke": SMOKE.read_text(encoding="utf-8"),
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
    report.check(fixture.get("schema") == "ds4.model_range_copy_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b2a", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == REVISION, "cuda-oxide revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-backend", "feature drift")
    validate_model_range(report, fixture)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_model_range(report: Report, fixture: dict[str, Any]) -> None:
    model = require_dict(report, fixture.get("model_range"), "model_range")
    report.check(model.get("path") == "/workspace/ds4/ds4flash.gguf", "model path drift")
    report.check(model.get("sha256") == MODEL_SHA256, "model hash drift")
    report.check(model.get("model_size") == MODEL_SIZE, "model size drift")
    report.check(model.get("range_offset") == 0, "range offset drift")
    report.check(model.get("range_bytes") == RANGE_BYTES, "range size drift")
    identity = require_dict(report, model.get("identity_verification"), "model_range.identity_verification")
    report.check(identity.get("command") == "sha256sum /workspace/ds4/ds4flash.gguf", "hash command drift")
    report.check(identity.get("stdout", "").startswith(MODEL_SHA256), "hash output drift")
    report.check(identity.get("passed") is True, "model identity was not verified")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    report.check(ownership.get("opt_in_only") is True, "range-copy path is no longer opt-in")
    report.check(ownership.get("owns_mapped_model_file_lifetime") is True, "mmap lifetime claim drift")
    report.check(ownership.get("owns_device_range_copy_cache") is True, "range-copy claim drift")
    report.check(ownership.get("owns_range_strategy_selection") is False, "strategy ownership overclaim")
    report.check(ownership.get("owns_ds4_kernels") is False, "kernel ownership overclaim")
    report.check(ownership.get("changes_default_route") is False, "route ownership overclaim")
    report.check(ownership.get("retains_current_c_cuda_oracle") is True, "current-C oracle dropped")
    for marker in [
        "pub const M14_1B2A_SCOPE",
        "owns_range_strategy_selection: false",
        "owns_ds4_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"Rust scope marker missing: {marker}")
    for marker in [
        "pub struct MappedModelFile",
        "fn mmap",
        "fn munmap",
        "pub struct ModelRangeCache",
        "model.range(offset, bytes)",
        "CachedRangeStorage::Device(substrate.upload(model.range(offset, bytes)?)?)",
        "substrate.synchronize()",
        "CacheOutcome::Reused",
    ]:
        report.check(marker in texts["model_map"], f"range-cache marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("cuda_toolkit") == "13.2", "CUDA toolkit drift")
    report.check(execution.get("rust_toolchain") == "nightly-2026-04-03", "Rust toolchain drift")
    report.check("--bin ds4-cuda-model-range-copy-smoke" in execution.get("command", ""), "smoke command missing")
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    expected = {
        "milestone": "M14.1b2a",
        "device_name": "NVIDIA B300 SXM6 AC",
        "model_size": MODEL_SIZE,
        "range_offset": 0,
        "range_bytes": RANGE_BYTES,
        "bounds_rejected": True,
        "range_copy_readback": True,
        "range_cache_reused": True,
        "owns_mapped_model_file_lifetime": True,
        "owns_device_range_copy_cache": True,
        "owns_range_strategy_selection": False,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    report.check(stdout == expected, "B300 model-range result drift")
    for marker in [
        "MappedModelFile::open",
        "model.range(model.size() + 1, 1).is_err()",
        "CacheOutcome::Inserted",
        "CacheOutcome::Reused",
        "cache.readback",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture_path = "ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json"
    checker = "check_model_range_copy_smoke.py"
    report.check("M14.1b2a: Owned Mmap Device Range Copy" in texts["roadmap"], "roadmap item missing")
    report.check(fixture_path in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b2a: Owned Mmap Device Range Copy" in texts["todo"], "TODO item missing")
    report.check(fixture_path in texts["todo"], "TODO fixture missing")
    report.check(
        "Active item: M14.1b2b" in texts["status"]
        or "Active item: M14.1b2c Model Map Cache Closure" in texts["status"],
        "next active stage missing",
    )
    report.check("M14.1b2a Owned Mmap Device Range Copy" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("strategy overclaim", lambda value: value["ownership"].update({"owns_range_strategy_selection": True})),
        ("readback failure", lambda value: value["b300_execution"]["stdout"].update({"range_copy_readback": False})),
        ("cache reuse failure", lambda value: value["b300_execution"]["stdout"].update({"range_cache_reused": False})),
        ("model identity failure", lambda value: value["model_range"]["identity_verification"].update({"passed": False})),
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
    print(f"M14.1b2a model range copy smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
