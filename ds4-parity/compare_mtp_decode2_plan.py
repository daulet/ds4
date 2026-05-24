#!/usr/bin/env python3
"""Compare the Rust MTP exact-N=2 verifier plan against current-C anchors."""

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
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_decode2_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-decode2-plan.rs"

EXPECTED_CASES = {
    "b300_missing_mtp_support_model": {
        "source_function": "none",
        "command_boundary": "none",
        "target_tokens": [],
        "start_source": "none",
        "decode_command_steps": [],
        "readbacks": [],
        "frontier_ops": [],
        "accept_condition": "blocked_missing_mtp_model",
        "accepted_suffix": "0",
        "checkpoint_action": "no session is created",
        "logits_source": "none",
        "mtp_n_raw_keep": "0",
        "failure_action": "blocked_missing_mtp_model",
        "live_status": "blocked_missing_mtp_model",
    },
    "exact_decode2_full_accept": {
        "source_function": "metal_graph_verify_decode2_exact",
        "command_boundary": "mtp_decode2_exact",
        "target_tokens": ["drafts[0]", "drafts[1]"],
        "start_source": "checkpoint.len",
        "decode_command_steps": [
            "row_view_batch_cur_hc_0",
            "row_view_batch_cur_hc_1",
            "row_view_batch_next_hc_0",
            "row_view_batch_next_hc_1",
            "embed_token0_hc",
            "embed_token1_hc",
            "save_cur_after_and_capture_flag",
            "enable_spec_capture_prefix1",
            "begin_decode2_commands",
            "for_each_layer_decode_token0_at_start",
            "capture_prefix1_attn_state",
            "capture_prefix1_index_state",
            "for_each_layer_decode_token1_at_start_plus_one",
            "swap_token0_cur_next_hc",
            "swap_token1_cur_next_hc",
            "end_decode2_commands",
            "restore_cur_after_and_capture_flag",
            "output_head_token0",
            "top1_readback_token0",
            "optional_logits0_readback",
            "output_head_token1",
            "logits1_readback",
            "free_decode2_row_views",
        ],
        "readbacks": ["top0", "optional_logits0", "logits1"],
        "frontier_ops": ["snapshot", "keep_accepted"],
        "accept_condition": "row0_top == drafts[1]",
        "accepted_suffix": "2",
        "checkpoint_action": "push drafts[0] and drafts[1]",
        "logits_source": "decode2 logits1",
        "mtp_n_raw_keep": "2",
        "failure_action": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "exact_decode2_prefix1_accept": {
        "source_function": "metal_graph_verify_decode2_exact",
        "command_boundary": "mtp_decode2_exact",
        "target_tokens": ["drafts[0]", "drafts[1]"],
        "start_source": "checkpoint.len",
        "decode_command_steps": "same_as_full_accept",
        "readbacks": ["top0", "optional_logits0", "logits1"],
        "frontier_ops": ["snapshot", "commit_prefix1", "keep_accepted"],
        "accept_condition": "row0_top != drafts[1] && decode2 ok",
        "accepted_suffix": "1",
        "checkpoint_action": "reset to start then push drafts[0]",
        "logits_source": "decode2 logits0",
        "mtp_n_raw_keep": "1",
        "failure_action": "none",
        "live_status": "blocked_missing_mtp_model",
    },
    "exact_decode2_failure_restore_then_sequential": {
        "source_function": "metal_graph_verify_decode2_exact",
        "command_boundary": "mtp_decode2_exact",
        "target_tokens": ["drafts[0]", "drafts[1]"],
        "start_source": "checkpoint.len",
        "decode_command_steps": "same_as_full_accept",
        "readbacks": ["top0", "optional_logits0", "logits1"],
        "frontier_ops": ["snapshot", "restore"],
        "accept_condition": "decode2 failed or prefix1 commit failed",
        "accepted_suffix": "verified_by_sequential_fallback",
        "checkpoint_action": "reset to start before sequential fallback",
        "logits_source": "sequential target decode",
        "mtp_n_raw_keep": "verified_by_sequential_fallback",
        "failure_action": "restore_pre_verifier_frontier",
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


def run_rust_plan() -> dict[str, Any]:
    proc = subprocess.run(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-decode2-plan", "--quiet"],
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
    report.check(candidate.get("schema") == "ds4.rust_mtp_decode2_plan.v1", "schema drift")
    report.check(
        candidate.get("source") == "rust-model-free-mtp-decode2-orchestration",
        "source drift",
    )
    report.check(candidate.get("oracle") == "metal_graph_verify_decode2_exact", "oracle drift")
    cases = named_cases(report, candidate.get("cases"))
    report.check(list(cases) == list(EXPECTED_CASES), "case order drift")
    for case_id, expected in EXPECTED_CASES.items():
        case = cases.get(case_id)
        if case is None:
            report.check(False, f"missing case {case_id}")
            continue
        for key, expected_value in expected.items():
            if key == "decode_command_steps" and expected_value == "same_as_full_accept":
                expected_value = EXPECTED_CASES["exact_decode2_full_accept"][
                    "decode_command_steps"
                ]
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
    for case_id in [
        "exact_decode2_full_accept",
        "exact_decode2_prefix1_accept",
        "exact_decode2_failure_restore_then_sequential",
    ]:
        case = contract_cases.get(case_id, {})
        expected = EXPECTED_CASES[case_id]
        for key in ["frontier_ops", "checkpoint_action", "logits_source"]:
            report.check(
                case.get(key) == expected[key],
                f"M10.8a {case_id}.{key} drift",
            )
        expected_fallback = {
            "exact_decode2_full_accept": "none",
            "exact_decode2_prefix1_accept": "none",
            "exact_decode2_failure_restore_then_sequential": "sequential safety fallback",
        }[case_id]
        report.check(
            case.get("fallback") == expected_fallback,
            f"M10.8a {case_id}.fallback drift",
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
        "static bool metal_graph_verify_decode2_exact(",
        "token0,",
        "token1,",
        "uint32_t               start,",
        "ds4_gpu_tensor *cur0 = metal_graph_tensor_row_view(g->batch_cur_hc, 0, hc_dim);",
        "ds4_gpu_tensor *cur1 = metal_graph_tensor_row_view(g->batch_cur_hc, 1, hc_dim);",
        "ds4_gpu_embed_token_hc_tensor(cur0,",
        "ds4_gpu_embed_token_hc_tensor(cur1,",
        "const bool saved_capture = g->spec_capture_prefix1;",
        "g->spec_capture_prefix1 = true;",
        "if (ok) ok = ds4_gpu_begin_commands() != 0;",
        "const uint32_t pos0 = start;",
        "const uint32_t pos1 = start + 1u;",
        "g->cur_hc = cur0;",
        "g->after_ffn_hc = next0;",
        "metal_graph_encode_decode_layer(g,",
        "metal_graph_capture_prefix1_attn_state(g, il)",
        "metal_graph_capture_prefix1_index_state(g, il)",
        "g->cur_hc = cur1;",
        "g->after_ffn_hc = next1;",
        "ds4_gpu_tensor *tmp = cur0; cur0 = next0; next0 = tmp;",
        "tmp = cur1; cur1 = next1; next1 = tmp;",
        "if (ok) ok = ds4_gpu_end_commands() != 0;",
        "g->spec_capture_prefix1 = saved_capture;",
        "metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);",
        "ds4_gpu_indexer_topk_tensor(g->comp_selected,",
        "ds4_gpu_tensor_read(g->comp_selected, 0, top0, sizeof(*top0))",
        "ds4_gpu_tensor_read(g->logits,",
        "logits0,",
        "logits1,",
        "ds4_gpu_tensor_free(next1);",
        "ds4_gpu_tensor_free(cur0);",
    ]:
        report.check(snippet in c_source, f"ds4.c missing decode2 anchor {snippet!r}")
    for snippet in [
        "const bool use_decode2_exact =",
        'draft_n == 2 && strict_mtp && getenv("DS4_MTP_BATCH_VERIFY") == NULL;',
        "bool have_frontier = spec_frontier_snapshot(&frontier, s);",
        "ok = metal_graph_verify_decode2_exact(&s->graph,",
        "drafts[0],",
        "drafts[1],",
        "if (ok && row0_top == drafts[1]) {",
        "memcpy(s->logits, row_logits",
        "token_vec_push(&s->checkpoint, drafts[0]);",
        "token_vec_push(&s->checkpoint, drafts[1]);",
        "DS4_MTP_KEEP_ACCEPTED(2);",
        "s->checkpoint.len = start;",
        "ok = spec_frontier_commit_prefix1(s);",
        "memcpy(s->logits, row0_logits",
        "DS4_MTP_KEEP_ACCEPTED(1);",
        "(void)spec_frontier_restore(&frontier, s);",
    ]:
        report.check(snippet in c_source, f"ds4.c missing session decode2 anchor {snippet!r}")
    boundary_index = graph_plan.find('"mtp_decode2_exact"')
    report.check(boundary_index >= 0, "graph_plan missing mtp_decode2_exact boundary")
    boundary_block = graph_plan[boundary_index : boundary_index + 180]
    for snippet in ['"metal_graph_verify_decode2_exact"', "3,", "true"]:
        report.check(
            snippet in boundary_block,
            f"graph_plan mtp_decode2_exact boundary missing {snippet!r}",
        )
    for snippet in [
        "pub mod mtp_decode2_plan;",
        "pub const MTP_DECODE2_ORCHESTRATION_CASES",
        "MTP_DECODE2_COMMAND_STEPS",
        "exact_decode2_failure_restore_then_sequential",
        "fn write_json_string",
        "compare_mtp_decode2_plan.py",
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
            "target order drift",
            lambda c: find_case(c, "exact_decode2_full_accept").__setitem__(
                "target_tokens", ["drafts[1]", "drafts[0]"]
            ),
        ),
        (
            "logits row drift",
            lambda c: find_case(c, "exact_decode2_prefix1_accept").__setitem__(
                "logits_source", "decode2 logits1"
            ),
        ),
        (
            "frontier op drift",
            lambda c: find_case(c, "exact_decode2_prefix1_accept").__setitem__(
                "frontier_ops", ["snapshot", "keep_accepted"]
            ),
        ),
        (
            "restore drift",
            lambda c: find_case(c, "exact_decode2_failure_restore_then_sequential").__setitem__(
                "failure_action", "continue_without_restore"
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
    print(f"Rust MTP decode2 plan negative tests: PASS, {passed} mutations")
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
        print(f"Rust MTP decode2 plan comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(candidate, contract)
    if not report.ok:
        print("Rust MTP decode2 plan comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "Rust MTP decode2 plan comparator: PASS, "
        f"{len(candidate['cases'])} cases, {report.checks} checks"
    )
    if args.negative_test:
        return run_negative_tests(candidate, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
