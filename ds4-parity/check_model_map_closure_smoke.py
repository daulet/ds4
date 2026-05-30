#!/usr/bin/env python3
"""Validate the M14.1b2c Rust-owned model-map cache closure smoke."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json"
CRATE_LIB = ROOT / "rust/ds4-cuda/src/lib.rs"
MODEL_MAP = ROOT / "rust/ds4-cuda/src/model_map.rs"
SMOKE = ROOT / "rust/ds4-cuda/src/bin/model_map_closure_smoke.rs"
CUDA_SOURCE = ROOT / "ds4_cuda.cu"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"

REVISION = "361300ea643688eea87eaa215d9a62a5e74a30e6"
MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
MODEL_SIZE = 86720111488
EXPECTED_RUST_OWNED = [
    "contained cached-range reuse and interior CUDA readback",
    "Linux POSIX source-page discard advisory call policy after staged chunks",
    "explicit non-TTY model-load progress emission and suppression policy",
    "RAII cache lifetime and new-cache reset state",
]
EXPECTED_NOT_CLAIMED = [
    "physical eviction of source pages after advisory calls",
    "default runtime environment or terminal-mode selection wiring",
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
    report.check(fixture.get("schema") == "ds4.model_map_closure_smoke.v1", "schema drift")
    report.check(fixture.get("milestone") == "M14.1b2c", "milestone drift")
    report.check(fixture.get("status") == "b300-pass", "B300 smoke status drift")
    oxide = require_dict(report, fixture.get("cuda_oxide"), "cuda_oxide")
    report.check(oxide.get("revision") == REVISION, "cuda-oxide revision drift")
    report.check(oxide.get("feature") == "cuda-oxide-backend", "feature drift")
    validate_oracle(report, fixture, texts)
    validate_model_range(report, fixture)
    validate_ownership(report, fixture, texts)
    validate_execution(report, fixture, texts)
    validate_wiring(report, texts)


def validate_oracle(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    oracle = require_dict(report, fixture.get("current_c_oracle"), "current_c_oracle")
    report.check(oracle.get("source") == "ds4_cuda.cu", "current-C source drift")
    for marker in [
        "cuda_model_range_ptr",
        "g_model_range_by_offset",
        "cuda_model_drop_file_pages",
        "POSIX_FADV_DONTNEED",
        "cuda_model_discard_source_pages",
        "POSIX_MADV_DONTNEED",
        "cuda_model_load_progress_note",
        "cuda_model_range_release_all",
    ]:
        report.check(marker in texts["cuda"], f"current-C oracle marker missing: {marker}")


def validate_model_range(report: Report, fixture: dict[str, Any]) -> None:
    model = require_dict(report, fixture.get("model_range"), "model_range")
    report.check(model.get("path") == "/workspace/ds4/ds4flash.gguf", "model path drift")
    report.check(model.get("sha256") == MODEL_SHA256, "model hash drift")
    report.check(model.get("model_size") == MODEL_SIZE, "model size drift")
    for key, expected in [
        ("cached_offset", 13),
        ("cached_bytes", 8192),
        ("contained_offset", 4116),
        ("contained_bytes", 257),
    ]:
        report.check(model.get(key) == expected, f"model range drift: {key}")
    identity = require_dict(report, model.get("identity_verification"), "identity_verification")
    report.check(identity.get("stdout", "").startswith(MODEL_SHA256), "model hash output drift")
    report.check(identity.get("passed") is True, "model hash was not verified")


def validate_ownership(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    report.check(ownership.get("rust_owned_in_this_stage") == EXPECTED_RUST_OWNED, "owned scope drift")
    report.check(ownership.get("not_claimed_in_this_stage") == EXPECTED_NOT_CLAIMED, "non-claim scope drift")
    for key, expected in [
        ("opt_in_only", True),
        ("owns_containing_range_reuse", True),
        ("owns_source_page_discard_policy", True),
        ("owns_progress_reporting", True),
        ("owns_raii_cache_cleanup", True),
        ("owns_ds4_kernels", False),
        ("changes_default_route", False),
        ("retains_current_c_cuda_oracle", True),
    ]:
        report.check(ownership.get(key) is expected, f"ownership drift: {key}")
    for marker in [
        "pub const M14_1B2C_SCOPE",
        "owns_containing_range_reuse: true",
        "owns_source_page_discard_policy: true",
        "owns_progress_reporting: true",
        "owns_raii_cache_cleanup: true",
        "owns_ds4_kernels: false",
        "changes_default_route: false",
    ]:
        report.check(marker in texts["lib"], f"scope marker missing: {marker}")
    for marker in [
        "pub enum ModelLoadProgressMode",
        "fn discard_source_pages",
        "libc::posix_fadvise",
        "libc::POSIX_FADV_DONTNEED",
        "libc::posix_madvise",
        "libc::POSIX_MADV_DONTNEED",
        "fn find_containing",
        "self.progress.note",
        "source_file_discard_calls",
        "source_mapping_discard_calls",
    ]:
        report.check(marker in texts["model_map"], f"model-map marker missing: {marker}")


def validate_execution(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    report.check(execution.get("kube_context") == "hou2-prod1", "B300 context drift")
    report.check(execution.get("pod") == "ds4-rust-port-b300", "B300 pod drift")
    report.check(execution.get("node") == "c1v17-b300n1-nic1", "B300 node drift")
    report.check(execution.get("cuda_toolkit") == "13.2", "CUDA toolkit drift")
    report.check("--features cuda-oxide-backend" in execution.get("test_command", ""), "feature test command missing")
    report.check("--bin ds4-cuda-model-map-closure-smoke" in execution.get("command", ""), "smoke command missing")
    report.check(
        execution.get("stderr_markers") == ["ds4: CUDA loading model tensors into device cache"],
        "non-TTY progress marker drift",
    )
    expected = {
        "milestone": "M14.1b2c",
        "device_name": "NVIDIA B300 SXM6 AC",
        "model_size": MODEL_SIZE,
        "cached_offset": 13,
        "cached_bytes": 8192,
        "contained_offset": 4116,
        "contained_bytes": 257,
        "chunks_uploaded": 2,
        "exact_range_hits": 1,
        "containing_range_hits": 1,
        "contained_range_reused": True,
        "contained_readback_matches": True,
        "source_file_discard_calls": 2,
        "source_file_discard_bytes": 8192,
        "source_mapping_discard_calls": 2,
        "source_mapping_discard_bytes": 16384,
        "progress_notes": 3,
        "progress_messages": 1,
        "non_tty_progress_initial_message": True,
        "fresh_cache_starts_empty": True,
        "keep_source_pages_suppresses_advice": True,
        "disabled_progress_suppresses_messages": True,
        "owns_containing_range_reuse": True,
        "owns_source_page_discard_policy": True,
        "owns_progress_reporting": True,
        "owns_raii_cache_cleanup": True,
        "owns_ds4_kernels": False,
        "changes_default_route": False,
    }
    stdout = require_dict(report, execution.get("stdout"), "b300_execution.stdout")
    report.check(stdout == expected, "B300 model-map closure result drift")
    for marker in [
        "AsyncPinnedRangeCache",
        "ModelLoadProgressMode::NonTty",
        "contained_readback_matches",
        "keep_source_pages_suppresses_advice",
        "fresh_cache_starts_empty",
    ]:
        report.check(marker in texts["smoke"], f"smoke marker missing: {marker}")


def validate_wiring(report: Report, texts: dict[str, str]) -> None:
    fixture = "ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json"
    checker = "check_model_map_closure_smoke.py"
    report.check("M14.1b2c: Model Map Cache Closure" in texts["roadmap"], "roadmap item missing")
    report.check(fixture in texts["roadmap"], "roadmap fixture missing")
    report.check("M14.1b2c: Model Map Cache Closure" in texts["todo"], "TODO item missing")
    report.check(fixture in texts["todo"], "TODO fixture missing")
    report.check("Active item: M14.1b3 Allocation And Quality Policy" in texts["status"], "next active stage missing")
    report.check("M14.1b2c Model Map Cache Closure" in texts["status"], "status evidence missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("contained reuse absent", lambda value: value["b300_execution"]["stdout"].update({"containing_range_hits": 0})),
        ("readback mismatch", lambda value: value["b300_execution"]["stdout"].update({"contained_readback_matches": False})),
        ("file advice absent", lambda value: value["b300_execution"]["stdout"].update({"source_file_discard_calls": 0})),
        ("mapping advice absent", lambda value: value["b300_execution"]["stdout"].update({"source_mapping_discard_calls": 0})),
        ("progress absent", lambda value: value["b300_execution"]["stdout"].update({"progress_messages": 0})),
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
    print(f"M14.1b2c model-map cache closure smoke: {status} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
