#!/usr/bin/env python3
"""Compare the Rust MTP suffix verifier plan against current-C anchors."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
M108A_CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json"
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_suffix_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-suffix-plan.rs"

EXPECTED_CASES = {
    "b300_missing_mtp_support_model": {
        "source_functions": [],
        "command_boundaries": [],
        "verifier_input": "none",
        "capture_prefix1": "none",
        "snapshot_requirement": "none",
        "commit_rule": "blocked_missing_mtp_model",
        "readbacks": [],
        "frontier_ops": [],
        "checkpoint_action": "no session is created",
        "accepted_suffix": "0",
        "logits_source": "none",
        "mtp_n_raw_keep": "0",
        "fallback": "blocked_missing_mtp_model",
        "failure_action": "blocked_missing_mtp_model",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_full_accept": {
        "source_functions": [
            "metal_graph_verify_suffix_tops",
            "metal_graph_read_spec_logits_row",
        ],
        "command_boundaries": ["mtp_suffix_tops"],
        "verifier_input": "checkpoint with all draft tokens appended",
        "capture_prefix1": "branch-computed capture_prefix1",
        "snapshot_requirement": "optional when snapshot_required is true",
        "commit_rule": "commit_drafts == draft_n",
        "readbacks": ["row_tops[0..draft_n-2]", "spec_logits[draft_n - 1]"],
        "frontier_ops": ["keep_accepted"],
        "checkpoint_action": "checkpoint already contains all draft tokens",
        "accepted_suffix": "draft_n",
        "logits_source": "spec logits row draft_n - 1",
        "mtp_n_raw_keep": "draft_n",
        "fallback": "none",
        "failure_action": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_prefix1_accept": {
        "source_functions": [
            "metal_graph_verify_suffix_tops",
            "spec_frontier_commit_prefix1",
            "metal_graph_read_spec_logits_row",
        ],
        "command_boundaries": ["mtp_suffix_tops", "spec_frontier_commit_prefix1"],
        "verifier_input": "checkpoint with drafts[0] and drafts[1] appended",
        "capture_prefix1": "draft_n == 2 && capture_prefix1",
        "snapshot_requirement": "snapshot not required unless forced",
        "commit_rule": "commit_drafts == 1",
        "readbacks": ["row_tops[0]", "spec_logits[0]"],
        "frontier_ops": ["commit_prefix1", "keep_accepted"],
        "checkpoint_action": "reset to start then push drafts[0]",
        "accepted_suffix": "1",
        "logits_source": "spec logits row 0",
        "mtp_n_raw_keep": "1",
        "fallback": "none",
        "failure_action": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_restore_replay_accept": {
        "source_functions": [
            "spec_frontier_snapshot",
            "metal_graph_verify_suffix_tops",
            "spec_frontier_restore",
            "metal_graph_eval_token_raw_swa",
            "metal_graph_read_spec_logits_row",
        ],
        "command_boundaries": [
            "spec_frontier_snapshot",
            "mtp_suffix_tops",
            "spec_frontier_restore",
        ],
        "verifier_input": "checkpoint with all draft tokens appended",
        "capture_prefix1": "!capture_prefix1 or replay-required suffix",
        "snapshot_requirement": "snapshot_required",
        "commit_rule": "commit_drafts < draft_n",
        "readbacks": ["target replay logits", "or spec_logits[commit_drafts - 1]"],
        "frontier_ops": ["snapshot", "restore", "keep_accepted"],
        "checkpoint_action": "restore to start then replay accepted drafts",
        "accepted_suffix": "commit_drafts",
        "logits_source": "target replay logits",
        "mtp_n_raw_keep": "commit_drafts",
        "fallback": "target replay",
        "failure_action": "restore_before_replay",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_exact_replay_debug_accept": {
        "source_functions": [
            "spec_frontier_snapshot",
            "metal_graph_verify_suffix_tops",
            "spec_frontier_restore",
            "metal_graph_eval_token_raw_swa",
        ],
        "command_boundaries": [
            "spec_frontier_snapshot",
            "mtp_suffix_tops",
            "spec_frontier_restore",
        ],
        "verifier_input": "checkpoint with all draft tokens appended",
        "capture_prefix1": "any",
        "snapshot_requirement": "required by DS4_MTP_EXACT_REPLAY",
        "commit_rule": "exact_replay_debug && have_frontier",
        "readbacks": ["target replay logits"],
        "frontier_ops": ["snapshot", "restore", "keep_accepted"],
        "checkpoint_action": "restore to start then exact-replay committed drafts",
        "accepted_suffix": "commit_drafts",
        "logits_source": "target replay logits",
        "mtp_n_raw_keep": "commit_drafts",
        "fallback": "exact replay debug",
        "failure_action": "restore_failure_falls_through",
        "live_status": "blocked_missing_mtp_model",
    },
    "suffix_failure_restore_or_error": {
        "source_functions": ["metal_graph_verify_suffix_tops", "spec_frontier_restore"],
        "command_boundaries": ["mtp_suffix_tops", "spec_frontier_restore"],
        "verifier_input": "checkpoint reset to start after verifier attempt",
        "capture_prefix1": "any",
        "snapshot_requirement": "restore if have_frontier; error if verifier mutated without frontier",
        "commit_rule": "verifier, prefix commit, replay, or logits read failed",
        "readbacks": [],
        "frontier_ops": ["restore_or_error"],
        "checkpoint_action": "reset to start; restore if snapshot exists; otherwise error if verifier mutated",
        "accepted_suffix": "verified_by_sequential_fallback_or_error",
        "logits_source": "sequential target decode or none on hard error",
        "mtp_n_raw_keep": "verified_by_sequential_fallback_or_zero",
        "fallback": "sequential safety fallback or MTP verifier failed",
        "failure_action": "restore_or_error",
        "live_status": "blocked_missing_mtp_model",
    },
}

CONTRACT_CASES = [
    "suffix_full_accept",
    "suffix_prefix1_accept",
    "suffix_restore_replay_accept",
    "suffix_failure_restore_or_error",
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


def run_rust_plan() -> dict[str, Any]:
    proc = subprocess.run(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-suffix-plan", "--quiet"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def validate(candidate: dict[str, Any], contract: dict[str, Any]) -> Report:
    report = Report()
    report.check(candidate.get("schema") == "ds4.rust_mtp_suffix_plan.v1", "schema drift")
    report.check(
        candidate.get("source") == "rust-model-free-mtp-suffix-orchestration",
        "source drift",
    )
    report.check(candidate.get("oracle") == "metal_graph_verify_suffix_tops", "oracle drift")
    cases = named_cases(report, candidate.get("cases"))
    report.check(list(cases) == list(EXPECTED_CASES), "case order drift")
    for case_id, expected in EXPECTED_CASES.items():
        case = cases.get(case_id)
        if case is None:
            report.check(False, f"missing case {case_id}")
            continue
        for key, expected_value in expected.items():
            report.check(
                case.get(key) == expected_value,
                f"{case_id}.{key}: expected {expected_value!r}, got {case.get(key)!r}",
            )
    check_contract_links(report, contract)
    static_checks(report)
    return report


def named_cases(report: Report, value: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    report.check(isinstance(value, list), "cases must be a list")
    if not isinstance(value, list):
        return result
    for index, item in enumerate(value):
        report.check(isinstance(item, dict), f"cases[{index}] must be an object")
        if not isinstance(item, dict):
            continue
        case_id = item.get("id")
        report.check(isinstance(case_id, str), f"cases[{index}].id missing")
        if isinstance(case_id, str):
            result[case_id] = item
    return result


def check_contract_links(report: Report, contract: dict[str, Any]) -> None:
    mtp = contract.get("support_artifacts", {}).get("mtp", {})
    report.check(mtp.get("available") is False, "M10.8a MTP support availability drift")
    report.check(mtp.get("candidate_count") == 0, "M10.8a MTP candidate count drift")
    contract_cases = {
        item.get("id"): item
        for item in contract.get("decision_cases", [])
        if isinstance(item, dict)
    }
    for case_id in CONTRACT_CASES:
        case = contract_cases.get(case_id, {})
        expected = EXPECTED_CASES[case_id]
        for key in ["frontier_ops", "checkpoint_action", "logits_source", "fallback"]:
            report.check(
                case.get(key) == expected[key],
                f"M10.8a {case_id}.{key} drift",
            )
        report.check(
            str(case.get("accepted_suffix")) == expected["accepted_suffix"],
            f"M10.8a {case_id}.accepted_suffix drift",
        )
        report.check(
            str(case.get("mtp_n_raw_keep")) == expected["mtp_n_raw_keep"],
            f"M10.8a {case_id}.mtp_n_raw_keep drift",
        )
    report.check(
        contract_cases.get("b300_missing_mtp_support_model", {}).get("fallback")
        == "blocked_missing_mtp_model",
        "missing-support fallback drift",
    )


def static_checks(report: Report) -> None:
    c_source = (ROOT / "ds4.c").read_text()
    graph_plan = (ROOT / "rust/ds4-gpu/src/graph_plan.rs").read_text()
    rust_source = RUST_SOURCE.read_text()
    rust_bin = RUST_BIN.read_text()
    lib_source = (ROOT / "rust/ds4-gpu/src/lib.rs").read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()

    for snippet in [
        "static bool metal_graph_verify_suffix_tops(",
        "const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;",
        "metal_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens)",
        "metal_graph_encode_layer_batch(g,",
        "metal_graph_encode_output_head_batch(g,",
        "ds4_gpu_indexer_topk_tensor(g->comp_selected,",
        "ds4_gpu_tensor_read(g->comp_selected,",
        "ds4_gpu_tensor_read(g->spec_logits,",
        "static bool metal_graph_read_spec_logits_row(",
        "row >= g->prefill_cap",
    ]:
        report.check(snippet in c_source, f"ds4.c missing suffix function anchor {snippet!r}")
    for snippet in [
        "const bool capture_prefix1 =",
        'getenv("DS4_MTP_CAPTURE_PREFIX1")',
        'const bool exact_replay_debug = getenv("DS4_MTP_EXACT_REPLAY") != NULL;',
        "const bool snapshot_required =",
        'getenv("DS4_MTP_FORCE_SNAPSHOT") != NULL;',
        "if (snapshot_required) {",
        "have_frontier = spec_frontier_snapshot(&frontier, s);",
        "for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);",
        "ok = metal_graph_verify_suffix_tops(&s->graph,",
        "int commit_drafts = 1;",
        "if (row_tops[i - 1] != drafts[i]) break;",
        "if (exact_replay_debug && have_frontier) {",
        "ok = metal_graph_eval_token_raw_swa(&s->graph,",
        "if (commit_drafts == draft_n) {",
        "(uint32_t)(draft_n - 1)",
        "if (draft_n == 2 && commit_drafts == 1 && capture_prefix1) {",
        "ok = spec_frontier_commit_prefix1(s);",
        "metal_graph_read_spec_logits_row(&s->graph, 0, row_logits)",
        "ok = have_frontier && spec_frontier_restore(&frontier, s);",
        "if (ok && draft_n == 2 && commit_drafts == 1) {",
        "for (int i = 0; i < commit_drafts; i++) token_vec_push(&s->checkpoint, drafts[i]);",
        "(uint32_t)(commit_drafts - 1)",
        "if (have_frontier) {",
        "(void)spec_frontier_restore(&frontier, s);",
        'snprintf(err, errlen, "MTP verifier failed");',
    ]:
        report.check(snippet in c_source, f"ds4.c missing suffix branch anchor {snippet!r}")
    boundary_index = graph_plan.find('"mtp_suffix_tops"')
    report.check(boundary_index >= 0, "graph_plan missing mtp_suffix_tops boundary")
    boundary_block = graph_plan[boundary_index : boundary_index + 160]
    for snippet in ['"metal_graph_verify_suffix_tops"', "2,", "true"]:
        report.check(
            snippet in boundary_block,
            f"graph_plan mtp_suffix_tops boundary missing {snippet!r}",
        )
    for snippet in [
        "pub mod mtp_suffix_plan;",
        "pub const MTP_SUFFIX_ORCHESTRATION_CASES",
        "MTP_SUFFIX_BASE_FUNCTIONS",
        "suffix_exact_replay_debug_accept",
        "fn write_json_string",
        "compare_mtp_suffix_plan.py",
    ]:
        found = (
            snippet in lib_source
            or snippet in rust_source
            or snippet in rust_bin
            or snippet in run_report
            or snippet in readme
        )
        report.check(found, f"missing static Rust/report anchor {snippet!r}")


def run_negative_tests(candidate: dict[str, Any], contract: dict[str, Any]) -> int:
    mutations = [
        ("schema drift", lambda c: c.__setitem__("schema", "wrong")),
        ("missing case", lambda c: c["cases"].pop()),
        (
            "full logits row drift",
            lambda c: find_case(c, "suffix_full_accept").__setitem__(
                "logits_source", "spec logits row 0"
            ),
        ),
        (
            "prefix frontier drift",
            lambda c: find_case(c, "suffix_prefix1_accept").__setitem__(
                "frontier_ops", ["keep_accepted"]
            ),
        ),
        (
            "replay restore drift",
            lambda c: find_case(c, "suffix_restore_replay_accept").__setitem__(
                "frontier_ops", ["keep_accepted"]
            ),
        ),
        (
            "exact replay snapshot drift",
            lambda c: find_case(c, "suffix_exact_replay_debug_accept").__setitem__(
                "snapshot_requirement", "none"
            ),
        ),
        (
            "failure drift",
            lambda c: find_case(c, "suffix_failure_restore_or_error").__setitem__(
                "failure_action", "ignore"
            ),
        ),
        (
            "live blocker drift",
            lambda c: find_case(c, "b300_missing_mtp_support_model").__setitem__(
                "live_status", "executed"
            ),
        ),
    ]
    passed = 0
    for name, mutate in mutations:
        candidate_copy = copy.deepcopy(candidate)
        mutate(candidate_copy)
        report = validate(candidate_copy, contract)
        if report.ok:
            print(f"negative-test failed to detect mutation: {name}", file=sys.stderr)
        else:
            passed += 1
    if passed != len(mutations):
        return 1
    print(f"Rust MTP suffix plan negative tests: PASS, {passed} mutations")
    return 0


def find_case(candidate: dict[str, Any], case_id: str) -> dict[str, Any]:
    for item in candidate["cases"]:
        if item.get("id") == case_id:
            return item
    raise KeyError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--contract", type=Path, default=M108A_CONTRACT)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        candidate = load_json(args.candidate) if args.candidate else run_rust_plan()
        contract = load_json(args.contract)
    except Exception as exc:
        print(f"Rust MTP suffix plan comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(candidate, contract)
    if not report.ok:
        print("Rust MTP suffix plan comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "Rust MTP suffix plan comparator: PASS, "
        f"{len(candidate['cases'])} cases, {report.checks} checks"
    )
    if args.negative_test:
        return run_negative_tests(candidate, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
