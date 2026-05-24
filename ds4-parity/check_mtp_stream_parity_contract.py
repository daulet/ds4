#!/usr/bin/env python3
"""Validate the M10.8g1 model-free MTP stream parity contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json"
)
M108A_CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json"

SCHEMA = "ds4.mtp_stream_parity_contract.v1"
EXPECTED_MILESTONE = "M10.8g1"
EXPECTED_SOURCE = "current-c-model-free-mtp-stream-parity-contract"
EXPECTED_SOURCE_FILES = [
    "ds4.c",
    "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json",
]
EXPECTED_B300 = {
    "context": "hou2-prod1",
    "namespace": "default",
    "pod": "ds4-rust-port-b300",
    "workdir": "/workspace/ds4",
    "base_model_path": "/workspace/ds4/ds4flash.gguf",
    "expected_mtp_path": "/workspace/ds4/missing-mtp.gguf",
    "candidate_globs": ["*mtp*.gguf", "*draft*.gguf"],
    "candidate_max_depth": 3,
    "candidate_count": 0,
    "checked_utc": "2026-05-24",
    "availability": "blocked_missing_mtp_model",
}
EXPECTED_INVARIANTS = {
    "target_stream_owner": "target model logits define every visible token",
    "first_token_commit": "ds4_session_eval commits first_token before speculative suffix work",
    "unavailable_mtp": "disabled or unavailable MTP returns the first-token accept only",
    "first_draft_guard": "drafts[0] must match target first-token logits before suffix verification",
    "verified_prefix_only": "only target-verified draft prefixes may be appended to the visible stream",
    "rollback": "misses and verifier failures restore checkpoint/frontier state before fallback or error",
    "mtp_raw_frontier": "DS4_MTP_KEEP_ACCEPTED hides future MTP raw rows by counter rewind",
    "cache_kvc_visibility": "cache and KVC accounting observe only visible target checkpoint state",
    "support_blocker": "B300 MTP-enabled stream parity remains blocked until a support GGUF exists",
}
EXPECTED_CASES = {
    "b300_missing_mtp_support_model": {
        "source_case": "b300_missing_mtp_support_model",
        "path": "availability_blocker",
        "trigger": "B300 support-artifact search finds no MTP GGUF",
        "accepted_suffix": 0,
        "accepted_stream_delta": "blocked_before_stream",
        "checkpoint_delta": "0",
        "logits_source": "none",
        "frontier_ops": [],
        "mtp_n_raw_keep": 0,
        "cache_kvc_visibility": "none",
        "fallback": "blocked_missing_mtp_model",
        "error": "blocked_missing_mtp_model",
        "live_status": "blocked_missing_mtp_model",
    },
    "mtp_disabled_after_first_token": {
        "source_case": "mtp_disabled_after_first_token",
        "path": "guard",
        "trigger": "!e->mtp_ready || !s->mtp_draft_valid || e->mtp_draft_tokens <= 1",
        "accepted_suffix": 0,
        "accepted_stream_delta": "first_token",
        "checkpoint_delta": "1",
        "logits_source": "target first-token logits",
        "frontier_ops": [],
        "mtp_n_raw_keep": 0,
        "cache_kvc_visibility": "first_token checkpoint only",
        "fallback": "return first-token accept",
        "error": "none",
        "live_status": "model_free",
    },
    "first_draft_miss": {
        "source_case": "first_draft_miss",
        "path": "draft_miss",
        "trigger": "sample_argmax(s->logits) != drafts[0]",
        "accepted_suffix": 0,
        "accepted_stream_delta": "first_token",
        "checkpoint_delta": "1",
        "logits_source": "target first-token logits",
        "frontier_ops": [],
        "mtp_n_raw_keep": 0,
        "cache_kvc_visibility": "first_token checkpoint only",
        "fallback": "skip speculative work",
        "error": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "margin_skip_single_target_replay": {
        "source_case": "margin_skip_single_target_replay",
        "path": "margin_skip",
        "trigger": "!strict_mtp && draft_n == 2 && margin < threshold",
        "accepted_suffix": 1,
        "accepted_stream_delta": "first_token + drafts[0]",
        "checkpoint_delta": "2",
        "logits_source": "target decode logits for drafts[0]",
        "frontier_ops": ["keep_accepted"],
        "mtp_n_raw_keep": 1,
        "cache_kvc_visibility": "two-token target checkpoint",
        "fallback": "margin-skip",
        "error": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "exact_decode2_full_accept": {
        "source_case": "exact_decode2_full_accept",
        "path": "exact_decode2",
        "trigger": "strict N=2 verifier accepts both draft tokens",
        "accepted_suffix": 2,
        "accepted_stream_delta": "first_token + drafts[0..1]",
        "checkpoint_delta": "3",
        "logits_source": "decode2 logits1",
        "frontier_ops": ["snapshot", "keep_accepted"],
        "mtp_n_raw_keep": 2,
        "cache_kvc_visibility": "three-token target checkpoint",
        "fallback": "none",
        "error": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "exact_decode2_prefix1_accept": {
        "source_case": "exact_decode2_prefix1_accept",
        "path": "exact_decode2",
        "trigger": "strict N=2 verifier accepts drafts[0] only",
        "accepted_suffix": 1,
        "accepted_stream_delta": "first_token + drafts[0]",
        "checkpoint_delta": "2",
        "logits_source": "decode2 logits0",
        "frontier_ops": ["snapshot", "commit_prefix1", "keep_accepted"],
        "mtp_n_raw_keep": 1,
        "cache_kvc_visibility": "two-token target checkpoint",
        "fallback": "none",
        "error": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "exact_decode2_failure_restore_then_sequential": {
        "source_case": "exact_decode2_failure_restore_then_sequential",
        "path": "exact_decode2_failure",
        "trigger": "strict N=2 verifier or prefix commit fails",
        "accepted_suffix": "verified_by_sequential_fallback",
        "accepted_stream_delta": "first_token + sequentially verified drafts",
        "checkpoint_delta": "1 + verified_by_sequential_fallback",
        "logits_source": "sequential target decode",
        "frontier_ops": ["snapshot", "restore"],
        "mtp_n_raw_keep": "verified_by_sequential_fallback",
        "cache_kvc_visibility": "sequential target checkpoint only",
        "fallback": "sequential safety fallback",
        "error": "none unless sequential decode fails",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_full_accept": {
        "source_case": "suffix_full_accept",
        "path": "suffix_verifier",
        "trigger": "microbatch suffix verifier accepts draft_n tokens",
        "accepted_suffix": "draft_n",
        "accepted_stream_delta": "first_token + drafts[0..draft_n-1]",
        "checkpoint_delta": "1 + draft_n",
        "logits_source": "spec logits row draft_n - 1",
        "frontier_ops": ["keep_accepted"],
        "mtp_n_raw_keep": "draft_n",
        "cache_kvc_visibility": "verified suffix checkpoint only",
        "fallback": "none",
        "error": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_prefix1_accept": {
        "source_case": "suffix_prefix1_accept",
        "path": "suffix_verifier",
        "trigger": "microbatch suffix verifier accepts drafts[0] with prefix1 capture",
        "accepted_suffix": 1,
        "accepted_stream_delta": "first_token + drafts[0]",
        "checkpoint_delta": "2",
        "logits_source": "spec logits row 0",
        "frontier_ops": ["commit_prefix1", "keep_accepted"],
        "mtp_n_raw_keep": 1,
        "cache_kvc_visibility": "two-token target checkpoint",
        "fallback": "none",
        "error": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_restore_replay_accept": {
        "source_case": "suffix_restore_replay_accept",
        "path": "suffix_verifier_replay",
        "trigger": "microbatch verifier accepts a prefix that requires restore/replay",
        "accepted_suffix": "commit_drafts",
        "accepted_stream_delta": "first_token + drafts[0..commit_drafts-1]",
        "checkpoint_delta": "1 + commit_drafts",
        "logits_source": "target replay logits",
        "frontier_ops": ["snapshot", "restore", "keep_accepted"],
        "mtp_n_raw_keep": "commit_drafts",
        "cache_kvc_visibility": "restored then replayed target checkpoint only",
        "fallback": "target replay",
        "error": "none unless replay fails",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_failure_restore_or_error": {
        "source_case": "suffix_failure_restore_or_error",
        "path": "suffix_verifier_failure",
        "trigger": "microbatch verifier, replay, or logits read fails",
        "accepted_suffix": "verified_by_sequential_fallback_or_error",
        "accepted_stream_delta": "first_token + sequential fallback or hard error",
        "checkpoint_delta": "1 + verified_by_sequential_fallback_or_error",
        "logits_source": "sequential target decode or none on hard error",
        "frontier_ops": ["restore_or_error"],
        "mtp_n_raw_keep": "verified_by_sequential_fallback_or_zero",
        "cache_kvc_visibility": "restored checkpoint, sequential fallback, or invalidated session",
        "fallback": "sequential safety fallback or MTP verifier failed",
        "error": "MTP verifier failed if mutated state lacks a snapshot",
        "live_status": "blocked_missing_mtp_model",
    },
    "sequential_safety_fallback": {
        "source_case": "sequential_safety_fallback",
        "path": "sequential_fallback",
        "trigger": "fast verifier cannot prove the suffix",
        "accepted_suffix": "verified",
        "accepted_stream_delta": "first_token + verified drafts",
        "checkpoint_delta": "1 + verified",
        "logits_source": "normal target decode logits",
        "frontier_ops": ["keep_accepted"],
        "mtp_n_raw_keep": "verified",
        "cache_kvc_visibility": "sequential target checkpoint only",
        "fallback": "target sequential verifier",
        "error": "none unless target decode or logits readback fails",
        "live_status": "blocked_missing_mtp_model",
    },
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def validate(contract: dict[str, Any], state_contract: dict[str, Any]) -> Report:
    report = Report()
    report.check(contract.get("schema") == SCHEMA, "schema drift")
    report.check(contract.get("milestone") == EXPECTED_MILESTONE, "milestone drift")
    report.check(contract.get("source") == EXPECTED_SOURCE, "source drift")
    report.check(contract.get("source_files") == EXPECTED_SOURCE_FILES, "source_files drift")
    check_b300(report, contract, state_contract)
    report.check(
        contract.get("stream_invariants") == EXPECTED_INVARIANTS,
        "stream_invariants drift",
    )
    check_function_anchors(report, contract)
    check_stream_cases(report, contract, state_contract)
    check_wiring(report)
    return report


def check_b300(
    report: Report,
    contract: dict[str, Any],
    state_contract: dict[str, Any],
) -> None:
    b300 = contract.get("b300")
    report.check(isinstance(b300, dict), "b300 must be an object")
    if isinstance(b300, dict):
        for key, expected in EXPECTED_B300.items():
            report.check(
                b300.get(key) == expected,
                f"b300.{key}: expected {expected!r}, got {b300.get(key)!r}",
            )
    mtp = contract.get("support_artifacts", {}).get("mtp", {})
    state_mtp = state_contract.get("support_artifacts", {}).get("mtp", {})
    report.check(mtp.get("path") == EXPECTED_B300["expected_mtp_path"], "MTP path drift")
    report.check(mtp.get("available") is False, "MTP availability drift")
    report.check(mtp.get("candidate_count") == 0, "MTP candidate count drift")
    report.check("no MTP GGUF" in str(mtp.get("blocker")), "MTP blocker text drift")
    for key in ["path", "available", "candidate_count", "blocker"]:
        report.check(
            mtp.get(key) == state_mtp.get(key),
            f"M10.8a support_artifacts.mtp.{key} drift",
        )


def check_function_anchors(report: Report, contract: dict[str, Any]) -> None:
    source = (ROOT / "ds4.c").read_text()
    anchors = contract.get("function_anchors")
    report.check(isinstance(anchors, list), "function_anchors must be a list")
    if not isinstance(anchors, list):
        return
    for anchor in anchors:
        report.check(isinstance(anchor, dict), "function anchor must be an object")
        if not isinstance(anchor, dict):
            continue
        name = anchor.get("name")
        snippets = anchor.get("required_snippets")
        report.check(isinstance(name, str) and bool(name), "function anchor name missing")
        report.check(isinstance(snippets, list) and bool(snippets), f"{name}: snippets empty")
        if isinstance(snippets, list):
            for snippet in snippets:
                report.check(isinstance(snippet, str), f"{name}: snippet is not a string")
                if isinstance(snippet, str):
                    report.check(snippet in source, f"ds4.c missing {name} anchor {snippet!r}")


def check_stream_cases(
    report: Report,
    contract: dict[str, Any],
    state_contract: dict[str, Any],
) -> None:
    state_cases = {
        item.get("id"): item
        for item in state_contract.get("decision_cases", [])
        if isinstance(item, dict)
    }
    cases = contract.get("stream_cases")
    report.check(isinstance(cases, list), "stream_cases must be a list")
    if not isinstance(cases, list):
        return
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(cases):
        report.check(isinstance(item, dict), f"stream_cases[{index}] must be an object")
        if not isinstance(item, dict):
            continue
        case_id = item.get("id")
        report.check(isinstance(case_id, str), f"stream_cases[{index}].id missing")
        if isinstance(case_id, str):
            by_id[case_id] = item
    report.check(list(by_id) == list(EXPECTED_CASES), "stream case order drift")
    for case_id, expected in EXPECTED_CASES.items():
        case = by_id.get(case_id)
        if case is None:
            report.check(False, f"missing stream case {case_id}")
            continue
        for key, expected_value in expected.items():
            report.check(
                case.get(key) == expected_value,
                f"{case_id}.{key}: expected {expected_value!r}, got {case.get(key)!r}",
            )
        source_case_id = expected["source_case"]
        state_case = state_cases.get(source_case_id)
        report.check(state_case is not None, f"M10.8a missing source case {source_case_id}")
        if state_case is None:
            continue
        for key in ["path", "frontier_ops", "accepted_suffix", "logits_source", "mtp_n_raw_keep", "fallback"]:
            report.check(
                case.get(key) == state_case.get(key),
                f"{case_id}.{key} differs from M10.8a {source_case_id}",
            )


def check_wiring(report: Report) -> None:
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()
    for snippet in [
        "check_mtp_stream_parity_contract.py",
        "M10.8g1 MTP stream parity contract",
    ]:
        report.check(snippet in run_report, f"unified report missing {snippet!r}")
    report.check(
        "check_mtp_stream_parity_contract.py --negative-test" in readme,
        "README missing M10.8g1 command",
    )


def run_negative_tests(contract: dict[str, Any], state_contract: dict[str, Any]) -> Report:
    report = Report()
    mutations = [
        ("schema", lambda data: data.update({"schema": "drift"})),
        ("missing case", lambda data: data["stream_cases"].pop(1)),
        (
            "first token drift",
            lambda data: mutate_case(data, "mtp_disabled_after_first_token", "accepted_stream_delta", "none"),
        ),
        (
            "checkpoint drift",
            lambda data: mutate_case(data, "exact_decode2_full_accept", "checkpoint_delta", "2"),
        ),
        (
            "logits drift",
            lambda data: mutate_case(data, "suffix_full_accept", "logits_source", "target logits"),
        ),
        (
            "frontier drift",
            lambda data: mutate_case(data, "suffix_restore_replay_accept", "frontier_ops", ["keep_accepted"]),
        ),
        (
            "raw frontier drift",
            lambda data: mutate_case(data, "sequential_safety_fallback", "mtp_n_raw_keep", "draft_n"),
        ),
        (
            "blocker drift",
            lambda data: data["support_artifacts"]["mtp"].update({"available": True}),
        ),
    ]
    for name, mutate in mutations:
        mutated = copy.deepcopy(contract)
        mutate(mutated)
        result = validate(mutated, state_contract)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def mutate_case(data: dict[str, Any], case_id: str, key: str, value: Any) -> None:
    for case in data["stream_cases"]:
        if case["id"] == case_id:
            case[key] = value
            return
    raise AssertionError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", nargs="?", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    contract = load_json(args.contract)
    state_contract = load_json(M108A_CONTRACT)
    report = validate(contract, state_contract)
    if not report.ok:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "M10.8g1 MTP stream parity contract: "
        f"PASS, {len(EXPECTED_CASES)} cases, {report.checks} checks"
    )
    if args.negative_test:
        negative = run_negative_tests(contract, state_contract)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("M10.8g1 MTP stream parity negative tests: PASS, 8 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
