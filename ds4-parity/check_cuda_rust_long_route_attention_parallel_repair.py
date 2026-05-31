#!/usr/bin/env python3
"""Validate the Rust CUDA long-route attention parallel repair leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Default-Route Promotion And C CUDA Removal Acceptance"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-long-route-attention-parallel-repair.json"


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


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} missing")
    return value if isinstance(value, dict) else {}


def function_text(source: str, marker: str, next_marker: str) -> str:
    start = source.index(marker)
    end = source.index(next_marker, start)
    return source[start:end]


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(list(argv))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    texts = {
        "kernels": (ROOT / "rust/ds4-cuda/src/abi_kernels.rs").read_text(encoding="utf-8"),
        "roadmap": (ROOT / "RUST_PORT_ROADMAP.md").read_text(encoding="utf-8"),
        "todo": (ROOT / ".memory/TODO.md").read_text(encoding="utf-8"),
        "status": (ROOT / ".memory/status.md").read_text(encoding="utf-8"),
        "readme": (ROOT / "ds4-parity/README.md").read_text(encoding="utf-8"),
        "report": (ROOT / "ds4-parity/run_parity_report.py").read_text(encoding="utf-8"),
    }
    report = Report()
    validate(report, fixture, texts)
    if args.negative_test:
        run_negative_tests(report, fixture, texts)
    state = "PASS" if report.ok else "FAIL"
    print(f"{MILESTONE} Rust CUDA long-route attention parallel repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(fixture.get("schema") == "ds4.cuda_rust_long_route_attention_parallel_repair.v1", "schema drift")
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(fixture.get("status") == "b300-pass-full-official-vector-route", "status drift")
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_rust_cuda_kernel_parallelism", True),
        ("changes_public_abi_surface", False),
        ("default_current_c_route_preserved", True),
        ("short_official_vector_correctness_preserved", True),
        ("long_code_audit_correctness_closed", True),
        ("full_official_vector_gate_closed", True),
        ("runtime_route_promoted", False),
        ("c_cuda_removal_allowed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("source") == "rust/ds4-cuda/src/abi_kernels.rs", "source drift")
    report.check(implementation.get("dispatch_policy_changed") is False, "dispatch overclaim")
    kernels = texts["kernels"]
    static_prefill = function_text(kernels, "pub fn abi_attention_static_mixed_heads8_online_kernel(", "pub fn abi_attention_prefill_pack_mixed_kv_kernel(")
    generic_indexed = function_text(kernels, "pub fn abi_attention_indexed_mixed_kernel(", "pub fn abi_attention_indexed_mixed_heads8_online_kernel(")
    online_indexed = function_text(kernels, "pub fn abi_attention_indexed_mixed_heads8_online_kernel(", "pub fn abi_attention_indexed_mixed_heads8_rb4_kernel(")
    for text, marker, shared in [
        (static_prefill, "KV_SHARED: SharedArray<f32, 2048>", "static prefill"),
        (generic_indexed, "SCORES: SharedArray<f32, 768>", "generic indexed"),
        (online_indexed, "KV_SHARED: SharedArray<f32, 4096>", "online indexed"),
    ]:
        report.check(marker in text, f"{shared} staging marker missing")
        report.check("threadIdx_x() != 0" not in text, f"{shared} remains lane-zero serialized")
    report.check("COMP_ROWS: SharedArray<u32, 512>" in generic_indexed, "ordered generic comp-row staging missing")
    report.check("warp::shuffle_xor_f32_sync" in generic_indexed, "generic indexed warp score reduction missing")
    report.check("warp::shuffle_xor_f32" in static_prefill, "static prefill warp reduction missing")
    report.check("warp::shuffle_xor_f32" in online_indexed, "online indexed warp reduction missing")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("pod", "ds4-rust-port-b300"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("shared_library_sha256", "299ec502207cece357323516d7c1273d3658917878503c201a97557a4eb79fad"),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    before = require_dict(report, execution.get("pre_repair_long_profile"), "pre_repair_long_profile")
    report.check(before.get("completed") is False, "pre-repair blocker missing")
    report.check(before.get("timeout_seconds") == 120, "pre-repair timeout drift")
    report.check("kv_path" in before.get("last_completed_stage", ""), "pre-repair attention boundary missing")
    static = require_dict(report, execution.get("static_prefill_only_probe"), "static_prefill_only_probe")
    report.check(static.get("first_chunk_completed") is True, "static prefill progress missing")
    report.check(static.get("sample_suffix_static_attention_ms", 999.0) < 25.0, "static prefill performance evidence missing")
    report.check("indexer_setup" in static.get("last_completed_stage", ""), "indexed prefill boundary missing")
    prefill = require_dict(report, execution.get("prefill_without_generic_indexed_decode_probe"), "prefill_without_generic_indexed_decode_probe")
    report.check(prefill.get("prefill_completed") is True, "indexed prefill progress missing")
    report.check(prefill.get("decode_completed") is False, "generic decode blocker missing")
    long_probe = require_dict(report, execution.get("long_code_audit_probe"), "long_code_audit_probe")
    for key, expected in [
        ("step_count", 4),
        ("selected_match_count", 4),
        ("all_selected_match", True),
        ("wall_elapsed_seconds", 89),
        ("gpu_memory_mib_after_process", 0),
    ]:
        report.check(long_probe.get(key) == expected, f"long probe drift: {key}")
    full = require_dict(report, execution.get("full_official_vector_probe"), "full_official_vector_probe")
    for key, expected in [
        ("completed", True),
        ("wall_elapsed_seconds", 101),
        ("case_count", 5),
        ("exercised_case_count", 4),
        ("selected_step_count", 13),
        ("selected_match_count", 13),
        ("all_selected_match", True),
        ("gpu_memory_mib_after_process", 0),
    ]:
        report.check(full.get(key) == expected, f"full-vector evidence drift: {key}")
    skips = full.get("known_skipped_cases", [])
    report.check(skips == [{"id": "long_memory_archive", "reason": "API/official graph mismatch"}], "known skip drift")
    smokes = require_dict(report, execution.get("affected_public_abi_smokes"), "affected_public_abi_smokes")
    report.check(smokes.get("attention_prefill_passed") is True, "prefill smoke missing")
    report.check(smokes.get("attention_indexed_batch_passed") is True, "indexed smoke missing")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Long-Route Attention Parallel Repair"
    checker = "check_cuda_rust_long_route_attention_parallel_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 feature test count drift")
    report.check(validation.get("unified_report_passed") == 261, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified fail count drift")
    review = require_dict(report, fixture.get("review"), "review")
    for key in ["pre_implementation", "follow_on_indexed_prefill", "follow_on_generic_indexed_decode", "final"]:
        report.check(review.get(key) == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", f"{key} review evidence missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("full selected token failure", lambda value: value["b300_execution"]["full_official_vector_probe"].update({"selected_match_count": 12})),
        ("pre-repair blocker hidden", lambda value: value["b300_execution"]["pre_repair_long_profile"].update({"completed": True})),
        ("route promotion overclaim", lambda value: value["ownership"].update({"runtime_route_promoted": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
