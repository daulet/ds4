#!/usr/bin/env python3
"""Validate the M6.2 fixed-logits C sampling/logprob oracle dump."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.2" / "current-c.json"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.2" / "manifest.json"

REQUIRED_SAMPLING_CASES = {
    "greedy_tie_first_max",
    "non_finite_logits",
    "full_vocab_min_p",
    "full_vocab_top_p",
    "top_p_clamped_zero",
    "negative_min_p_clamped",
    "top_k_filter",
    "top_k_capped_to_vocab",
    "seeded_rng_draw",
    "request_cli_default_ds4_cli_c",
    "request_openai_chat_default_ds4_server_c",
    "request_openai_responses_default_ds4_server_c",
    "request_anthropic_default_ds4_server_c",
    "request_agent_default_ds4_agent_c",
    "request_thinking_default_ds4_server_c",
    "request_dsml_structural_greedy_ds4_server_c",
}

REQUEST_CASE_SOURCES = {
    "request_cli_default_ds4_cli_c": "ds4_cli.c",
    "request_openai_chat_default_ds4_server_c": "ds4_server.c",
    "request_openai_responses_default_ds4_server_c": "ds4_server.c",
    "request_anthropic_default_ds4_server_c": "ds4_server.c",
    "request_agent_default_ds4_agent_c": "ds4_agent.c",
    "request_thinking_default_ds4_server_c": "ds4_server.c",
    "request_dsml_structural_greedy_ds4_server_c": "ds4_server.c",
}

REQUIRED_LOGPROB_CASES = {
    "top_logprobs_sparse",
    "top_logprobs_tie_order",
    "top_logprobs_nonfinite",
}

SPECIAL_FLOATS = {"nan", "inf", "-inf"}


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


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_float_value(value: Any) -> bool:
    return isinstance(value, (int, float)) or value in SPECIAL_FLOATS


def approx(value: Any, expected: float, tolerance: float = 1e-6) -> bool:
    return isinstance(value, (int, float)) and abs(float(value) - expected) <= tolerance


def float_rank(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return None


def require_dict(report: Report, value: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{path}: expected object")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, path: str) -> list[Any]:
    report.check(isinstance(value, list), f"{path}: expected array")
    return value if isinstance(value, list) else []


def check_float(report: Report, value: Any, path: str) -> None:
    report.check(is_float_value(value), f"{path}: expected number or special float string")


def check_params(report: Report, params: Any, path: str) -> None:
    obj = require_dict(report, params, path)
    for key in ("temperature", "top_p", "min_p"):
        check_float(report, obj.get(key), f"{path}.{key}")
    report.check(isinstance(obj.get("top_k"), int), f"{path}.top_k invalid")
    report.check(isinstance(obj.get("seed"), int), f"{path}.seed invalid")


def check_logits(report: Report, logits: Any, path: str) -> None:
    items = require_list(report, logits, path)
    for idx, raw in enumerate(items):
        item = require_dict(report, raw, f"{path}[{idx}]")
        report.check(item.get("id") == idx, f"{path}[{idx}].id drift")
        check_float(report, item.get("value"), f"{path}[{idx}].value")


def check_sampling_case(report: Report, raw: Any, path: str) -> None:
    case = require_dict(report, raw, path)
    name = case.get("name")
    report.check(isinstance(name, str) and bool(name), f"{path}.name invalid")
    source = case.get("source")
    report.check(isinstance(source, str) and bool(source), f"{path}.source invalid")
    if name in REQUEST_CASE_SOURCES and isinstance(source, str):
        report.check(
            REQUEST_CASE_SOURCES[name] in source,
            f"{path}.source missing {REQUEST_CASE_SOURCES[name]}",
        )
    report.check(isinstance(case.get("n_vocab"), int) and case["n_vocab"] > 0, f"{path}.n_vocab invalid")
    check_params(report, case.get("params"), f"{path}.params")
    effective = require_dict(report, case.get("effective"), f"{path}.effective")
    report.check(isinstance(effective.get("top_k"), int), f"{path}.effective.top_k invalid")
    for key in ("top_p", "min_p"):
        check_float(report, effective.get(key), f"{path}.effective.{key}")
    check_logits(report, case.get("logits"), f"{path}.logits")
    report.check(isinstance(case.get("selected"), int), f"{path}.selected invalid")
    report.check(isinstance(case.get("actual_selected"), int), f"{path}.actual_selected invalid")
    report.check(case.get("selected") == case.get("actual_selected"), f"{path}.selected != actual_selected")
    report.check(case.get("matches_actual") is True, f"{path}.matches_actual not true")
    for key in ("rng_before", "rng_after", "actual_rng_after", "finite_count", "filtered_count"):
        report.check(isinstance(case.get(key), int), f"{path}.{key} invalid")
    report.check(case.get("rng_after") == case.get("actual_rng_after"), f"{path}.rng_after drift")
    report.check(isinstance(case.get("greedy"), bool), f"{path}.greedy invalid")
    for key in ("max_logit", "sum", "filtered_sum", "rng_unit"):
        check_float(report, case.get(key), f"{path}.{key}")
    candidates = require_list(report, case.get("filtered_candidates"), f"{path}.filtered_candidates")
    report.check(len(candidates) == case.get("filtered_count"), f"{path}.filtered_count drift")
    for idx, raw_candidate in enumerate(candidates):
        candidate = require_dict(report, raw_candidate, f"{path}.filtered_candidates[{idx}]")
        report.check(isinstance(candidate.get("id"), int), f"{path}.filtered_candidates[{idx}].id invalid")
        for key in ("logit", "weight", "normalized_prob"):
            check_float(report, candidate.get(key), f"{path}.filtered_candidates[{idx}].{key}")


def check_score(report: Report, raw: Any, path: str) -> None:
    score = require_dict(report, raw, path)
    report.check(isinstance(score.get("id"), int), f"{path}.id invalid")
    check_float(report, score.get("logit"), f"{path}.logit")
    check_float(report, score.get("logprob"), f"{path}.logprob")


def check_logprob_case(report: Report, raw: Any, path: str) -> None:
    case = require_dict(report, raw, path)
    report.check(isinstance(case.get("name"), str) and bool(case.get("name")), f"{path}.name invalid")
    report.check(isinstance(case.get("source"), str) and "ds4.c" in case["source"], f"{path}.source invalid")
    report.check(isinstance(case.get("n_vocab"), int) and case["n_vocab"] > 0, f"{path}.n_vocab invalid")
    check_float(report, case.get("background_logit"), f"{path}.background_logit")
    report.check(isinstance(case.get("top_k"), int) and case["top_k"] > 0, f"{path}.top_k invalid")
    report.check(isinstance(case.get("returned"), int), f"{path}.returned invalid")
    check_logits(report, case.get("logits"), f"{path}.logits")
    scores = require_list(report, case.get("scores"), f"{path}.scores")
    report.check(len(scores) == case.get("top_k"), f"{path}.scores length drift")
    for idx, score in enumerate(scores):
        check_score(report, score, f"{path}.scores[{idx}]")
        if idx > 0 and isinstance(score, dict) and isinstance(scores[idx - 1], dict):
            prev = float_rank(scores[idx - 1].get("logit"))
            cur = float_rank(score.get("logit"))
            if prev is not None and cur is not None:
                report.check(prev >= cur, f"{path}.scores[{idx}] ordering drift")
    queries = require_list(report, case.get("token_logprobs"), f"{path}.token_logprobs")
    report.check(len(queries) >= 1, f"{path}.token_logprobs empty")
    for idx, raw_query in enumerate(queries):
        query = require_dict(report, raw_query, f"{path}.token_logprobs[{idx}]")
        report.check(isinstance(query.get("token"), int), f"{path}.token_logprobs[{idx}].token invalid")
        report.check(isinstance(query.get("ok"), bool), f"{path}.token_logprobs[{idx}].ok invalid")
        check_score(report, query.get("score"), f"{path}.token_logprobs[{idx}].score")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.sampling_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-fixed-logits", "source mismatch")
    report.check(root.get("n_vocab_full") == 129280, "n_vocab_full drift")
    defaults = require_dict(report, root.get("defaults"), "defaults")
    report.check(approx(defaults.get("temperature"), 1.0), "defaults.temperature drift")
    report.check(defaults.get("top_k") == 0, "defaults.top_k drift")
    report.check(approx(defaults.get("top_p"), 1.0), "defaults.top_p drift")
    report.check(approx(defaults.get("min_p"), 0.05), "defaults.min_p drift")

    sampling_cases = require_list(report, root.get("sampling_cases"), "sampling_cases")
    sampling_names = {case.get("name") for case in sampling_cases if isinstance(case, dict)}
    report.check(REQUIRED_SAMPLING_CASES <= sampling_names, "sampling case coverage drift")
    report.check(len(sampling_names) == len(sampling_cases), "duplicate sampling case names")
    for idx, case in enumerate(sampling_cases):
        check_sampling_case(report, case, f"sampling_cases[{idx}]")

    logprob_cases = require_list(report, root.get("logprob_cases"), "logprob_cases")
    logprob_names = {case.get("name") for case in logprob_cases if isinstance(case, dict)}
    report.check(REQUIRED_LOGPROB_CASES <= logprob_names, "logprob case coverage drift")
    report.check(len(logprob_names) == len(logprob_cases), "duplicate logprob case names")
    for idx, case in enumerate(logprob_cases):
        check_logprob_case(report, case, f"logprob_cases[{idx}]")
    return report


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    manifest = load_json(path)
    report.check(manifest.get("schema") == "ds4.sampling_manifest.v1", "manifest schema mismatch")
    artifact_info = manifest.get("artifact")
    if not isinstance(artifact_info, dict):
        report.check(False, "manifest artifact missing")
        return report
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("size") == artifact.stat().st_size, "manifest artifact size drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha256 drift")
    commands = manifest.get("refresh_commands")
    report.check(isinstance(commands, list) and len(commands) >= 3, "manifest refresh commands missing")
    if isinstance(commands, list):
        joined = " && ".join(str(command) for command in commands)
        report.check("ds4-sampling-dump" in joined, "manifest refresh missing ds4-sampling-dump")
        report.check("check_sampling_dump.py" in joined, "manifest refresh missing checker")
    return report


def merge(dst: Report, prefix: str, src: Report) -> None:
    dst.checks += src.checks
    dst.errors.extend(f"{prefix}: {error}" for error in src.errors)


def run_negative(obj: dict[str, Any], manifest_path: Path | None, artifact_path: Path) -> Report:
    report = Report()

    def expect_failure(label: str, mutate: Any) -> None:
        mutated = copy.deepcopy(obj)
        mutate(mutated)
        result = check_dump(mutated)
        report.check(not result.ok, f"negative test failed to catch {label}")

    expect_failure("selected-token", lambda o: o["sampling_cases"][0].__setitem__("selected", -99))
    expect_failure("missing-request-case", lambda o: o.__setitem__("sampling_cases", o["sampling_cases"][:-1]))
    expect_failure(
        "candidate-list",
        lambda o: o["sampling_cases"][2].__setitem__(
            "filtered_candidates",
            o["sampling_cases"][2]["filtered_candidates"][:-1],
        ),
    )
    expect_failure("logprob-score-order", lambda o: o["logprob_cases"][0]["scores"].reverse())
    expect_failure("token-logprob", lambda o: o["logprob_cases"][0]["token_logprobs"][0].__setitem__("ok", "yes"))

    if manifest_path is not None:
        manifest = load_json(manifest_path)
        manifest["artifact"]["sha256"] = "0" * 64
        tmp_report = Report()
        tmp_report.check(manifest.get("artifact", {}).get("sha256") == sha256_file(artifact_path), "manifest sha256")
        report.check(not tmp_report.ok, "negative test failed to catch manifest sha256")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    obj = load_json(args.dump)
    report = check_dump(obj)
    print_report("sampling schema", report)
    if not report.ok:
        return 1

    manifest_path = args.manifest
    if manifest_path is None and args.dump == BASELINE and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest = check_manifest(manifest_path, args.dump)
        print_report("sampling manifest", manifest)
        if not manifest.ok:
            return 1

    if args.negative_test:
        negative = run_negative(obj, manifest_path, args.dump)
        print_report("sampling negative tests", negative)
        if not negative.ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
