#!/usr/bin/env python3
"""Compare the Rust sampler/logprob port against the M6.2 C oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.2" / "current-c.json"

ORDINARY_ABS_TOLERANCE = 1e-5
SENTINEL_REL_TOLERANCE = 1e-6
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


def run_rust_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-sampling-dump-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def float_value(value: Any) -> float | str | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value in SPECIAL_FLOATS:
        return value
    return None


def compare_float(report: Report, expected: Any, got: Any, path: str) -> None:
    ev = float_value(expected)
    gv = float_value(got)
    report.check(ev is not None, f"{path}: expected value is not numeric")
    report.check(gv is not None, f"{path}: Rust value is not numeric")
    if ev is None or gv is None:
        return
    if isinstance(ev, str) or isinstance(gv, str):
        report.check(ev == gv, f"{path}: special float drift {expected!r} != {got!r}")
        return
    if math.isnan(ev) or math.isnan(gv):
        report.check(math.isnan(ev) and math.isnan(gv), f"{path}: nan drift")
        return
    if abs(ev) >= 1e20 or abs(gv) >= 1e20:
        scale = max(abs(ev), abs(gv), 1.0)
        report.check(
            abs(ev - gv) <= scale * SENTINEL_REL_TOLERANCE,
            f"{path}: sentinel drift {expected!r} != {got!r}",
        )
        return
    report.check(
        abs(ev - gv) <= ORDINARY_ABS_TOLERANCE,
        f"{path}: numeric drift {expected!r} != {got!r}",
    )


def compare_int(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(isinstance(expected, int), f"{path}: expected int invalid")
    report.check(isinstance(got, int), f"{path}: Rust int invalid")
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_bool(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(isinstance(expected, bool), f"{path}: expected bool invalid")
    report.check(isinstance(got, bool), f"{path}: Rust bool invalid")
    report.check(expected is got, f"{path}: {expected!r} != {got!r}")


def by_name(items: list[Any], path: str, report: Report) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, raw in enumerate(items):
        item = require_dict(report, raw, f"{path}[{idx}]")
        name = item.get("name")
        report.check(isinstance(name, str) and bool(name), f"{path}[{idx}].name invalid")
        if isinstance(name, str):
            report.check(name not in out, f"{path}: duplicate case {name}")
            out[name] = item
    return out


def compare_params(report: Report, expected: dict[str, Any], got: dict[str, Any], path: str) -> None:
    for key in ("temperature", "top_p", "min_p"):
        compare_float(report, expected.get(key), got.get(key), f"{path}.{key}")
    for key in ("top_k", "seed"):
        compare_int(report, expected.get(key), got.get(key), f"{path}.{key}")


def compare_logits(report: Report, expected: Any, got: Any, path: str) -> None:
    expected_items = require_list(report, expected, path)
    got_items = require_list(report, got, path)
    report.check(len(expected_items) == len(got_items), f"{path}: length drift")
    for idx, (expected_raw, got_raw) in enumerate(zip(expected_items, got_items)):
        expected_item = require_dict(report, expected_raw, f"{path}[{idx}].expected")
        got_item = require_dict(report, got_raw, f"{path}[{idx}].rust")
        compare_int(report, expected_item.get("id"), got_item.get("id"), f"{path}[{idx}].id")
        compare_float(report, expected_item.get("value"), got_item.get("value"), f"{path}[{idx}].value")


def compare_sampling_case(
    report: Report,
    expected: dict[str, Any],
    got: dict[str, Any],
    path: str,
) -> None:
    compare_int(report, expected.get("n_vocab"), got.get("n_vocab"), f"{path}.n_vocab")
    compare_params(
        report,
        require_dict(report, expected.get("params"), f"{path}.expected.params"),
        require_dict(report, got.get("params"), f"{path}.rust.params"),
        f"{path}.params",
    )
    expected_effective = require_dict(report, expected.get("effective"), f"{path}.expected.effective")
    got_effective = require_dict(report, got.get("effective"), f"{path}.rust.effective")
    compare_int(report, expected_effective.get("top_k"), got_effective.get("top_k"), f"{path}.effective.top_k")
    for key in ("top_p", "min_p"):
        compare_float(report, expected_effective.get(key), got_effective.get(key), f"{path}.effective.{key}")
    compare_logits(report, expected.get("logits"), got.get("logits"), f"{path}.logits")
    for key in ("selected", "actual_selected", "rng_before", "rng_after", "actual_rng_after", "finite_count", "filtered_count"):
        compare_int(report, expected.get(key), got.get(key), f"{path}.{key}")
    compare_bool(report, expected.get("matches_actual"), got.get("matches_actual"), f"{path}.matches_actual")
    compare_bool(report, expected.get("greedy"), got.get("greedy"), f"{path}.greedy")
    for key in ("max_logit", "sum", "filtered_sum", "rng_unit"):
        compare_float(report, expected.get(key), got.get(key), f"{path}.{key}")

    expected_candidates = require_list(report, expected.get("filtered_candidates"), f"{path}.expected.filtered_candidates")
    got_candidates = require_list(report, got.get("filtered_candidates"), f"{path}.rust.filtered_candidates")
    report.check(len(expected_candidates) == len(got_candidates), f"{path}.filtered_candidates length drift")
    for idx, (expected_raw, got_raw) in enumerate(zip(expected_candidates, got_candidates)):
        expected_candidate = require_dict(report, expected_raw, f"{path}.expected.filtered_candidates[{idx}]")
        got_candidate = require_dict(report, got_raw, f"{path}.rust.filtered_candidates[{idx}]")
        compare_int(report, expected_candidate.get("id"), got_candidate.get("id"), f"{path}.filtered_candidates[{idx}].id")
        for key in ("logit", "weight", "normalized_prob"):
            compare_float(
                report,
                expected_candidate.get(key),
                got_candidate.get(key),
                f"{path}.filtered_candidates[{idx}].{key}",
            )


def compare_score(report: Report, expected: Any, got: Any, path: str) -> None:
    expected_score = require_dict(report, expected, f"{path}.expected")
    got_score = require_dict(report, got, f"{path}.rust")
    compare_int(report, expected_score.get("id"), got_score.get("id"), f"{path}.id")
    compare_float(report, expected_score.get("logit"), got_score.get("logit"), f"{path}.logit")
    compare_float(report, expected_score.get("logprob"), got_score.get("logprob"), f"{path}.logprob")


def compare_logprob_case(
    report: Report,
    expected: dict[str, Any],
    got: dict[str, Any],
    path: str,
) -> None:
    for key in ("n_vocab", "top_k", "returned"):
        compare_int(report, expected.get(key), got.get(key), f"{path}.{key}")
    compare_float(report, expected.get("background_logit"), got.get("background_logit"), f"{path}.background_logit")
    compare_logits(report, expected.get("logits"), got.get("logits"), f"{path}.logits")

    expected_scores = require_list(report, expected.get("scores"), f"{path}.expected.scores")
    got_scores = require_list(report, got.get("scores"), f"{path}.rust.scores")
    report.check(len(expected_scores) == len(got_scores), f"{path}.scores length drift")
    for idx, (expected_score, got_score) in enumerate(zip(expected_scores, got_scores)):
        compare_score(report, expected_score, got_score, f"{path}.scores[{idx}]")

    expected_queries = require_list(report, expected.get("token_logprobs"), f"{path}.expected.token_logprobs")
    got_queries = require_list(report, got.get("token_logprobs"), f"{path}.rust.token_logprobs")
    report.check(len(expected_queries) == len(got_queries), f"{path}.token_logprobs length drift")
    for idx, (expected_raw, got_raw) in enumerate(zip(expected_queries, got_queries)):
        expected_query = require_dict(report, expected_raw, f"{path}.expected.token_logprobs[{idx}]")
        got_query = require_dict(report, got_raw, f"{path}.rust.token_logprobs[{idx}]")
        compare_int(report, expected_query.get("token"), got_query.get("token"), f"{path}.token_logprobs[{idx}].token")
        compare_bool(report, expected_query.get("ok"), got_query.get("ok"), f"{path}.token_logprobs[{idx}].ok")
        compare_score(report, expected_query.get("score"), got_query.get("score"), f"{path}.token_logprobs[{idx}].score")


def compare_dumps(expected: Any, got: Any) -> Report:
    report = Report()
    expected_root = require_dict(report, expected, "expected")
    got_root = require_dict(report, got, "rust")
    report.check(expected_root.get("schema") == "ds4.sampling_oracle.v1", "C schema mismatch")
    report.check(got_root.get("schema") == "ds4.rust_sampling_oracle.v1", "Rust schema mismatch")
    compare_int(report, expected_root.get("n_vocab_full"), got_root.get("n_vocab_full"), "n_vocab_full")
    for key in ("temperature", "top_p", "min_p"):
        compare_float(report, expected_root.get("defaults", {}).get(key), got_root.get("defaults", {}).get(key), f"defaults.{key}")
    compare_int(report, expected_root.get("defaults", {}).get("top_k"), got_root.get("defaults", {}).get("top_k"), "defaults.top_k")

    expected_sampling = by_name(require_list(report, expected_root.get("sampling_cases"), "expected.sampling_cases"), "expected.sampling_cases", report)
    got_sampling = by_name(require_list(report, got_root.get("sampling_cases"), "rust.sampling_cases"), "rust.sampling_cases", report)
    report.check(list(expected_sampling) == list(got_sampling), "sampling case order or coverage drift")
    for name, expected_case in expected_sampling.items():
        got_case = got_sampling.get(name)
        report.check(got_case is not None, f"{name}: missing Rust sampling case")
        if got_case is not None:
            compare_sampling_case(report, expected_case, got_case, f"sampling_cases.{name}")

    expected_logprob = by_name(require_list(report, expected_root.get("logprob_cases"), "expected.logprob_cases"), "expected.logprob_cases", report)
    got_logprob = by_name(require_list(report, got_root.get("logprob_cases"), "rust.logprob_cases"), "rust.logprob_cases", report)
    report.check(list(expected_logprob) == list(got_logprob), "logprob case order or coverage drift")
    for name, expected_case in expected_logprob.items():
        got_case = got_logprob.get(name)
        report.check(got_case is not None, f"{name}: missing Rust logprob case")
        if got_case is not None:
            compare_logprob_case(report, expected_case, got_case, f"logprob_cases.{name}")
    return report


def run_negative_tests(expected: Any, got: Any) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("selected token drift", ["sampling_cases", 0, "selected"], -99),
        ("rng drift", ["sampling_cases", 1, "rng_after"], 123),
        ("candidate-list drift", ["sampling_cases", 2, "filtered_candidates"], []),
        ("logprob drift", ["logprob_cases", 0, "scores", 0, "logprob"], 99.0),
        ("request coverage drift", ["sampling_cases"], got["sampling_cases"][:-1]),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(got)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = compare_dumps(expected, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--rust-dump", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--write-rust-dump", type=Path)
    args = parser.parse_args()

    expected = load_json(args.baseline)
    if args.rust_dump:
        got = load_json(args.rust_dump)
    else:
        code, stdout, stderr = run_rust_dump()
        if code != 0:
            print("rust sampling dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        got = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    compare = compare_dumps(expected, got)
    print_report("sampling C/Rust comparator", compare)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("sampling C/Rust negative tests", negative)

    return 0 if compare.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
