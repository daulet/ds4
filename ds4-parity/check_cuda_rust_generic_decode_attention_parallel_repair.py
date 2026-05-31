#!/usr/bin/env python3
"""Validate the Rust CUDA generic mixed decode-attention parallel repair leaf."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"
NEXT_STAGE = "M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Default-Route Promotion And C CUDA Removal Execution"
FIXTURE = ROOT / f"ds4-parity/baselines/backend/{MILESTONE.lower()}/rust-cuda-generic-decode-attention-parallel-repair.json"


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
    print(f"{MILESTONE} Rust CUDA generic decode-attention parallel repair: {state} ({report.checks} checks)")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 0 if report.ok else 1


def validate(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    report.check(
        fixture.get("schema") == "ds4.cuda_rust_generic_decode_attention_parallel_repair.v1",
        "schema drift",
    )
    report.check(fixture.get("milestone") == MILESTONE, "milestone drift")
    report.check(
        fixture.get("status") == "b300-pass-performance-repair-correctness-still-blocked",
        "status drift",
    )
    validate_implementation(report, fixture, texts)
    validate_execution(report, fixture)
    validate_wiring(report, fixture, texts)


def validate_implementation(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    ownership = require_dict(report, fixture.get("ownership"), "ownership")
    for key, expected in [
        ("changes_generic_decode_attention_kernel", True),
        ("changes_dispatch_surface", False),
        ("changes_public_abi_surface", False),
        ("default_current_c_route_preserved", True),
        ("runtime_route_promoted", False),
        ("c_cuda_removal_allowed", False),
        ("official_vector_correctness_closed", False),
    ]:
        report.check(ownership.get(key) == expected, f"ownership drift: {key}")
    implementation = require_dict(report, fixture.get("implementation"), "implementation")
    report.check(implementation.get("source") == "rust/ds4-cuda/src/abi_kernels.rs", "source drift")
    report.check(implementation.get("kernel") == "abi_attention_decode_mixed_kernel", "kernel drift")
    kernel = texts["kernels"].split("pub fn abi_attention_decode_mixed_kernel(", 1)[1].split(
        "pub fn abi_attention_decode_mixed_heads8_online_kernel(", 1
    )[0]
    for marker in [
        "static mut SCORES: SharedArray<f32, 8192>",
        "static mut RAW_ROWS: SharedArray<u32, 256>",
        "static mut PARTIAL: SharedArray<f32, 256>",
        "static mut SOFTMAX: SharedArray<f32, 2>",
        "warp::shuffle_xor_f32_sync",
        "dimension1 = dimension0 + 256",
        "thread::sync_threads()",
    ]:
        report.check(marker in kernel, f"parallel kernel marker missing: {marker}")
    report.check("thread::threadIdx_x() != 0" not in kernel, "serialized lane-zero gate restored")


def validate_execution(report: Report, fixture: dict[str, Any]) -> None:
    execution = require_dict(report, fixture.get("b300_execution"), "b300_execution")
    for key, expected in [
        ("date_utc", "2026-05-31"),
        ("kube_context", "hou2-prod1"),
        ("namespace", "default"),
        ("pod", "ds4-rust-port-b300"),
        ("node", "c1v17-b300n1-nic1"),
        ("device_name", "NVIDIA B300 SXM6 AC"),
        ("shared_library_sha256", "1bbbd217e00d77f12f33dfec72a688346d316e5baca600c5d9d5eb2c780cc7ab"),
    ]:
        report.check(execution.get(key) == expected, f"B300 evidence drift: {key}")
    before = require_dict(report, execution.get("before_profile"), "before_profile")
    report.check(before.get("decode_position") == 27, "pre-repair decode position drift")
    report.check(before.get("attention_layer_samples_ms", {}).get("layer_0") == 471.787, "pre-repair layer 0 timing drift")
    report.check(before.get("attention_layer_samples_ms", {}).get("layer_2") == 589.625, "pre-repair layer 2 timing drift")
    after = require_dict(report, execution.get("after_profile"), "after_profile")
    report.check(after.get("wall_elapsed_seconds") == 48, "post-repair wall timing drift")
    attention = require_dict(report, after.get("attention"), "after_profile.attention")
    for key, expected in [("layer_count", 43), ("minimum_ms", 0.08), ("maximum_ms", 0.134), ("sum_ms", 4.578)]:
        report.check(attention.get(key) == expected, f"post-repair attention timing drift: {key}")
    steps = after.get("selected_steps", [])
    report.check(len(steps) == 2, "two-step output evidence missing")
    if len(steps) == 2:
        report.check(steps[0].get("selected_matches_expected") is True, "step zero unexpectedly diverged")
        report.check(steps[1].get("expected_selected_hex") == "63", "step one oracle drift")
        report.check(steps[1].get("selected_hex") == "43", "step one blocker drift")
        report.check(steps[1].get("selected_matches_expected") is False, "correctness overclaim")
    smoke = require_dict(report, execution.get("public_attention_abi_smoke"), "public_attention_abi_smoke")
    for key in [
        "linked_shared_library",
        "generic_masked_output_matches",
        "raw_only_ring_wrapped_output_matches",
        "sink_softmax_matches",
        "overflow_online_output_matches",
        "invalid_inputs_rejected",
        "embedded_attention_decode_kernels_loaded",
    ]:
        report.check(smoke.get(key) is True, f"public ABI smoke drift: {key}")
    blocker = require_dict(report, execution.get("remaining_correctness_blocker"), "remaining_correctness_blocker")
    diffs = require_dict(report, blocker.get("paired_tensor_max_abs_difference"), "paired tensor differences")
    report.check(diffs.get("q_lora") == 0.0, "low-rank input equality drift")
    report.check(diffs.get("KVraw") == 0.0, "KV raw equality drift")
    report.check(diffs.get("Qraw") == 0.00000476837158203125, "Q raw boundary drift")
    report.check(diffs.get("Qcur") == 0.0001509338617324829, "Q current drift")
    report.check(diffs.get("attn_out") == 0.0011451244354248047, "attention output drift")


def validate_wiring(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    item = f"{MILESTONE}: Rust CUDA Generic Decode Attention Parallel Repair"
    checker = "check_cuda_rust_generic_decode_attention_parallel_repair.py"
    for target, label in [("roadmap", "roadmap"), ("todo", "TODO"), ("status", "status")]:
        report.check(item in texts[target], f"{label} item missing")
    report.check(checker in texts["readme"], "README checker wiring missing")
    report.check(checker in texts["report"], "unified report checker wiring missing")
    report.check(f"Active item: {NEXT_STAGE}" in texts["status"], "active stage missing")
    report.check(fixture.get("next_required_stage") == NEXT_STAGE, "next stage drift")
    validation = require_dict(report, fixture.get("validation"), "validation")
    report.check(validation.get("local_ds4_cuda_library_test_count") == 169, "local test count drift")
    report.check(validation.get("b300_feature_release_test_count") == 176, "B300 feature test count drift")
    report.check(validation.get("unified_report_passed") == 259, "unified pass count drift")
    report.check(validation.get("unified_report_skipped") == 45, "unified skip count drift")
    report.check(validation.get("unified_report_failed") == 0, "unified fail count drift")
    review = require_dict(report, fixture.get("review"), "review")
    report.check(review.get("pre_implementation") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "pre-review missing")
    report.check(review.get("final") == "CLAUDE_REVIEW_TIMEOUT_AFTER_60S", "final review missing")


def run_negative_tests(report: Report, fixture: dict[str, Any], texts: dict[str, str]) -> None:
    for label, mutate in [
        ("route promotion", lambda value: value["ownership"].update({"runtime_route_promoted": True})),
        ("timing overclaim", lambda value: value["b300_execution"]["after_profile"]["attention"].update({"maximum_ms": 589.625})),
        ("correctness overclaim", lambda value: value["b300_execution"]["after_profile"]["selected_steps"][1].update({"selected_matches_expected": True})),
        ("next stage drift", lambda value: value.update({"next_required_stage": "wrong"})),
    ]:
        candidate = copy.deepcopy(fixture)
        mutate(candidate)
        negative = Report()
        validate(negative, candidate, texts)
        report.check(not negative.ok, f"negative test did not reject {label}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
