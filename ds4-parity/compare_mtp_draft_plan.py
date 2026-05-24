#!/usr/bin/env python3
"""Compare the Rust MTP draft orchestration plan against current-C anchors."""

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
RUST_SOURCE = ROOT / "rust/ds4-gpu/src/mtp_draft_plan.rs"
RUST_BIN = ROOT / "rust/ds4-gpu/src/bin/ds4-mtp-draft-plan.rs"

EXPECTED_CASES = {
    "b300_missing_mtp_support_model": {
        "source_function": "none",
        "command_boundary": "none",
        "prev_hc": "none",
        "out_hc": "none",
        "token_source": "none",
        "pos_source": "none",
        "logits_role": "none",
        "top_id_role": "none",
        "command_steps": [],
        "readbacks": [],
        "mtp_n_raw_transition": "none",
        "saved_state": [],
        "failure_action": "blocked_missing_mtp_model",
        "live_status": "blocked_missing_mtp_model",
    },
    "first_draft_from_current_hc": {
        "source_function": "metal_graph_eval_mtp_draft",
        "command_boundary": "mtp_draft",
        "prev_hc": "cur_hc",
        "out_hc": "mtp_state_hc",
        "token_source": "accepted target token",
        "pos_source": "checkpoint.len - 1",
        "logits_role": "optional_full_logits",
        "top_id_role": "required_for_draft_token",
        "command_steps": [
            "begin_commands",
            "embed_token_hc",
            "rms_norm_embed",
            "matmul_e_proj",
            "repeat_e_proj_hc",
            "rms_norm_prev_hc",
            "matmul_h_proj",
            "add_mtp_input_hc",
            "encode_decode_layer_mtp_block",
            "set_cur_hc_to_out_hc",
            "encode_output_head_mtp",
            "optional_top1_indexer",
            "end_commands",
        ],
        "readbacks": ["optional_logits", "optional_top_id"],
        "mtp_n_raw_transition": "increment_if_less_than_raw_window",
        "saved_state": ["cur_hc", "after_ffn_hc"],
        "failure_action": "leave_mtp_draft_valid_false_and_keep_target_decode",
        "live_status": "blocked_missing_mtp_model",
    },
    "recursive_draft_state_to_next": {
        "source_function": "metal_graph_eval_mtp_draft_from_hc",
        "command_boundary": "mtp_draft",
        "prev_hc": "mtp_state_hc",
        "out_hc": "mtp_next_hc",
        "token_source": "previous draft token",
        "pos_source": "checkpoint.len + draft_n - 1",
        "logits_role": "optional_need_logits",
        "top_id_role": "required_for_next_draft",
        "command_steps": "same_as_first",
        "readbacks": ["optional_logits", "optional_top_id"],
        "mtp_n_raw_transition": "increment_if_less_than_raw_window",
        "saved_state": ["cur_hc", "after_ffn_hc"],
        "failure_action": "return_current_accepted_prefix",
        "live_status": "blocked_missing_mtp_model",
    },
    "recursive_draft_next_to_state": {
        "source_function": "metal_graph_eval_mtp_draft_from_hc",
        "command_boundary": "mtp_draft",
        "prev_hc": "mtp_next_hc",
        "out_hc": "mtp_state_hc",
        "token_source": "previous draft token",
        "pos_source": "checkpoint.len + draft_n - 1",
        "logits_role": "optional_need_logits",
        "top_id_role": "required_for_next_draft",
        "command_steps": "same_as_first",
        "readbacks": ["optional_logits", "optional_top_id"],
        "mtp_n_raw_transition": "increment_if_less_than_raw_window",
        "saved_state": ["cur_hc", "after_ffn_hc"],
        "failure_action": "return_current_accepted_prefix",
        "live_status": "blocked_missing_mtp_model",
    },
    "draft_failure_restores_saved_graph_state": {
        "source_function": "metal_graph_eval_mtp_draft_from_hc",
        "command_boundary": "mtp_draft",
        "prev_hc": "any_prev_hc",
        "out_hc": "any_out_hc",
        "token_source": "draft input token",
        "pos_source": "draft input position",
        "logits_role": "not_committed_on_failure",
        "top_id_role": "not_committed_on_failure",
        "command_steps": ["synchronize_after_failure"],
        "readbacks": [],
        "mtp_n_raw_transition": "no_increment_on_failure",
        "saved_state": ["cur_hc", "after_ffn_hc"],
        "failure_action": "restore_cur_hc_and_after_ffn_hc",
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
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-draft-plan", "--quiet"],
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
    report.check(candidate.get("schema") == "ds4.rust_mtp_draft_plan.v1", "schema drift")
    report.check(candidate.get("source") == "rust-model-free-mtp-draft-orchestration", "source drift")
    report.check(candidate.get("oracle") == "metal_graph_eval_mtp_draft_from_hc", "oracle drift")
    cases = named_cases(report, candidate.get("cases"))
    report.check(list(cases) == list(EXPECTED_CASES), "case order drift")
    for case_id, expected in EXPECTED_CASES.items():
        case = cases.get(case_id)
        if case is None:
            report.check(False, f"missing case {case_id}")
            continue
        for key, expected_value in expected.items():
            if key == "command_steps" and expected_value == "same_as_first":
                expected_value = EXPECTED_CASES["first_draft_from_current_hc"]["command_steps"]
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
    report.check(
        contract_cases.get("b300_missing_mtp_support_model", {}).get("fallback")
        == "blocked_missing_mtp_model",
        "missing-support fallback drift",
    )
    report.check(
        contract_cases.get("first_draft_miss", {}).get("accepted_suffix") == 0,
        "first-draft miss accepted suffix drift",
    )


def static_checks(report: Report) -> None:
    c_source = (ROOT / "ds4.c").read_text()
    graph_plan = (ROOT / "rust/ds4-gpu/src/graph_plan.rs").read_text()
    rust_source = RUST_SOURCE.read_text()
    rust_bin = RUST_BIN.read_text()
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()

    for snippet in [
        "static bool metal_graph_eval_mtp_draft_from_hc(",
        "uint32_t n_raw = g->mtp_n_raw + 1u;",
        "ds4_gpu_begin_commands()",
        "ds4_gpu_embed_token_hc_tensor",
        "ds4_gpu_rms_norm_weight_tensor",
        "ds4_gpu_matmul_q8_0_tensor",
        "ds4_gpu_repeat_hc_tensor",
        "ds4_gpu_rms_norm_weight_rows_tensor",
        "ds4_gpu_add_tensor(g->mtp_input_hc",
        "metal_graph_encode_decode_layer",
        "metal_graph_encode_output_head_mtp",
        "ds4_gpu_indexer_topk_tensor",
        "ds4_gpu_end_commands()",
        "ds4_gpu_tensor_read(g->logits",
        "ds4_gpu_tensor_read(g->comp_selected",
        "if (ok && g->mtp_n_raw < g->raw_window) g->mtp_n_raw++;",
        "(void)ds4_gpu_synchronize();",
        "g->cur_hc = saved_cur;",
        "g->after_ffn_hc = saved_after;",
        "static bool metal_graph_eval_mtp_draft(",
        "g->cur_hc,",
        "g->mtp_state_hc,",
    ]:
        report.check(snippet in c_source, f"ds4.c missing draft anchor {snippet!r}")
    report.check(
        'boundary!("mtp_draft", "metal_graph_eval_mtp_draft_from_hc", 1, true)'
        in graph_plan,
        "graph_plan missing mtp_draft boundary",
    )
    for snippet in [
        "pub mod mtp_draft_plan;",
        "pub const MTP_DRAFT_ORCHESTRATION_CASES",
        "MTP_DRAFT_COMMAND_STEPS",
        "draft_failure_restores_saved_graph_state",
        "fn write_json_string",
        "compare_mtp_draft_plan.py",
    ]:
        found = snippet in rust_source or snippet in rust_bin or snippet in run_report or snippet in readme or snippet in (ROOT / "rust/ds4-gpu/src/lib.rs").read_text()
        report.check(found, f"missing static Rust/report anchor {snippet!r}")


def run_negative_tests(candidate: dict[str, Any], contract: dict[str, Any]) -> int:
    mutations = [
        ("schema drift", lambda c: c.__setitem__("schema", "wrong")),
        ("missing case", lambda c: c["cases"].pop()),
        (
            "command step drift",
            lambda c: find_case(c, "first_draft_from_current_hc").__setitem__("command_steps", []),
        ),
        (
            "hc role drift",
            lambda c: find_case(c, "recursive_draft_state_to_next").__setitem__("out_hc", "mtp_state_hc"),
        ),
        (
            "raw transition drift",
            lambda c: find_case(c, "draft_failure_restores_saved_graph_state").__setitem__(
                "mtp_n_raw_transition", "increment_if_less_than_raw_window"
            ),
        ),
        (
            "live blocker drift",
            lambda c: find_case(c, "b300_missing_mtp_support_model").__setitem__("live_status", "executed"),
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
    print(f"Rust MTP draft plan negative tests: PASS, {passed} mutations")
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
        print(f"Rust MTP draft plan comparator: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(candidate, contract)
    if not report.ok:
        print("Rust MTP draft plan comparator: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(
        "Rust MTP draft plan comparator: PASS, "
        f"{len(candidate['cases'])} cases, {report.checks} checks"
    )
    if args.negative_test:
        return run_negative_tests(candidate, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
