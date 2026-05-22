#!/usr/bin/env python3
"""Compare Rust sampler/logprob output over M6.4 model logits against current C."""

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
BASELINE = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.4" / "current-c.json"
LOGITS = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.4" / "logits.f32le"
TOKENIZER = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.3" / "tokenizer.gguf"

ORDINARY_ABS_TOLERANCE = 1e-5
SPECIAL_FLOATS = {"nan", "inf", "-inf"}


@dataclass
class Report:
    checks: int = 0
    errors: list[str] = field(default_factory=list)
    max_logit_delta: float = 0.0
    max_logprob_delta: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def record_delta(self, kind: str, delta: float) -> None:
        if kind == "logit":
            self.max_logit_delta = max(self.max_logit_delta, delta)
        elif kind == "logprob":
            self.max_logprob_delta = max(self.max_logprob_delta, delta)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def run_rust_dump(logits: Path, tokenizer: Path, top_k: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-model-logits-dump-rs",
            "--",
            "--logits",
            str(logits),
            "--tokenizer",
            str(tokenizer),
            "--top-k",
            str(top_k),
        ],
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


def compare_int(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(isinstance(expected, int), f"{path}: expected int invalid")
    report.check(isinstance(got, int), f"{path}: Rust int invalid")
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_str(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(isinstance(expected, str), f"{path}: expected string invalid")
    report.check(isinstance(got, str), f"{path}: Rust string invalid")
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_bool(report: Report, expected: Any, got: Any, path: str) -> None:
    report.check(isinstance(expected, bool), f"{path}: expected bool invalid")
    report.check(isinstance(got, bool), f"{path}: Rust bool invalid")
    report.check(expected is got, f"{path}: {expected!r} != {got!r}")


def float_value(value: Any) -> float | str | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value in SPECIAL_FLOATS:
        return value
    return None


def compare_float(
    report: Report,
    expected: Any,
    got: Any,
    path: str,
    kind: str,
) -> None:
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
    delta = abs(ev - gv)
    report.record_delta(kind, delta)
    report.check(
        delta <= ORDINARY_ABS_TOLERANCE,
        f"{path}: numeric drift {expected!r} != {got!r}",
    )


@dataclass
class ExpectedStep:
    global_index: int
    case_id: str
    case_step: int
    step: dict[str, Any]


def expected_steps(oracle: Any, report: Report) -> list[ExpectedStep]:
    root = require_dict(report, oracle, "expected")
    report.check(root.get("schema") == "ds4.session_logits_oracle.v1", "C schema mismatch")
    steps: list[ExpectedStep] = []
    for case_idx, raw_case in enumerate(require_list(report, root.get("cases"), "expected.cases")):
        case = require_dict(report, raw_case, f"expected.cases[{case_idx}]")
        case_id = case.get("id")
        report.check(isinstance(case_id, str) and bool(case_id), f"expected.cases[{case_idx}].id invalid")
        skipped = case.get("skipped")
        compare_bool(report, skipped, skipped, f"expected.cases[{case_idx}].skipped")
        case_steps = require_list(report, case.get("steps"), f"expected.cases[{case_idx}].steps")
        if skipped:
            report.check(not case_steps, f"{case_id}: skipped case should not carry logits steps")
            continue
        for step_idx, raw_step in enumerate(case_steps):
            step = require_dict(report, raw_step, f"{case_id}.steps[{step_idx}]")
            compare_int(report, step_idx, step.get("step"), f"{case_id}.steps[{step_idx}].step")
            compare_bool(
                report,
                True,
                step.get("selected_matches_expected"),
                f"{case_id}.steps[{step_idx}].selected_matches_expected",
            )
            steps.append(
                ExpectedStep(
                    global_index=len(steps),
                    case_id=str(case_id),
                    case_step=step_idx,
                    step=step,
                )
            )
    return steps


def compare_score(
    report: Report,
    expected: Any,
    got: Any,
    path: str,
) -> None:
    expected_score = require_dict(report, expected, f"{path}.expected")
    got_score = require_dict(report, got, f"{path}.rust")
    compare_int(report, expected_score.get("id"), got_score.get("id"), f"{path}.id")
    compare_str(report, expected_score.get("bytes_hex"), got_score.get("bytes_hex"), f"{path}.bytes_hex")
    compare_float(report, expected_score.get("logit"), got_score.get("logit"), f"{path}.logit", "logit")
    compare_float(
        report,
        expected_score.get("logprob"),
        got_score.get("logprob"),
        f"{path}.logprob",
        "logprob",
    )


def compare_step(
    report: Report,
    expected: ExpectedStep,
    got: dict[str, Any],
    path: str,
) -> None:
    step = expected.step
    compare_int(report, expected.global_index, got.get("index"), f"{path}.index")
    compare_int(report, step.get("logits_offset"), got.get("logits_offset"), f"{path}.logits_offset")
    compare_int(report, step.get("logits_bytes"), got.get("logits_bytes"), f"{path}.logits_bytes")
    compare_int(report, step.get("selected_token"), got.get("selected_token"), f"{path}.selected_token")
    compare_str(report, step.get("selected_bytes_hex"), got.get("selected_bytes_hex"), f"{path}.selected_bytes_hex")
    compare_str(
        report,
        step.get("expected_selected_hex"),
        got.get("selected_bytes_hex"),
        f"{path}.expected_selected_hex",
    )

    expected_scores = require_list(report, step.get("top_logprobs"), f"{path}.expected.top_logprobs")
    got_scores = require_list(report, got.get("top_logprobs"), f"{path}.rust.top_logprobs")
    report.check(len(expected_scores) == len(got_scores), f"{path}.top_logprobs length drift")
    compare_int(report, len(expected_scores), got.get("top_k_returned"), f"{path}.top_k_returned")
    if expected_scores and got_scores:
        expected_top = require_dict(report, expected_scores[0], f"{path}.expected.top_logprobs[0]")
        got_top = require_dict(report, got_scores[0], f"{path}.rust.top_logprobs[0]")
        compare_int(report, step.get("selected_token"), expected_top.get("id"), f"{path}.expected.selected_is_top")
        compare_int(report, got.get("selected_token"), got_top.get("id"), f"{path}.rust.selected_is_top")
        compare_str(
            report,
            step.get("selected_bytes_hex"),
            expected_top.get("bytes_hex"),
            f"{path}.expected.selected_bytes_is_top",
        )
        compare_str(
            report,
            got.get("selected_bytes_hex"),
            got_top.get("bytes_hex"),
            f"{path}.rust.selected_bytes_is_top",
        )
    for score_idx, (expected_score, got_score) in enumerate(zip(expected_scores, got_scores)):
        compare_score(report, expected_score, got_score, f"{path}.top_logprobs[{score_idx}]")


def compare_dumps(expected: Any, got: Any) -> Report:
    report = Report()
    steps = expected_steps(expected, report)
    got_root = require_dict(report, got, "rust")
    report.check(got_root.get("schema") == "ds4.rust_model_logits_slices.v1", "Rust schema mismatch")
    compare_int(report, expected.get("top_k"), got_root.get("top_k"), "top_k")
    compare_int(report, expected.get("n_vocab_full", 129280), got_root.get("n_vocab_full"), "n_vocab_full")
    compare_int(report, len(steps), got_root.get("slice_count"), "slice_count")

    slices = require_list(report, got_root.get("slices"), "rust.slices")
    report.check(len(steps) == len(slices), "model-backed step coverage drift")
    for expected_step, got_raw in zip(steps, slices):
        got_step = require_dict(report, got_raw, f"rust.slices[{expected_step.global_index}]")
        compare_step(
            report,
            expected_step,
            got_step,
            f"{expected_step.case_id}.steps[{expected_step.case_step}]",
        )
    return report


def set_path(root: Any, path: list[str | int], value: Any) -> None:
    target = root
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def run_negative_tests(expected: Any, got: Any) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("selected token drift", ["slices", 0, "selected_token"], -99),
        ("selected byte drift", ["slices", 0, "selected_bytes_hex"], "00"),
        ("top id drift", ["slices", 1, "top_logprobs", 0, "id"], -1),
        ("top byte drift", ["slices", 1, "top_logprobs", 0, "bytes_hex"], "00"),
        ("logprob drift", ["slices", 2, "top_logprobs", 0, "logprob"], 99.0),
        ("coverage drift", ["slices"], got["slices"][:-1]),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(got)
        set_path(bad, path, value)
        result = compare_dumps(expected, bad)
        report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(
        f"{label}: {status}, {report.checks} checks, "
        f"max_abs_logit_delta={report.max_logit_delta:.9g}, "
        f"max_abs_logprob_delta={report.max_logprob_delta:.9g}"
    )
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--logits", type=Path, default=LOGITS)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--rust-dump", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--write-rust-dump", type=Path)
    args = parser.parse_args()

    expected = load_json(args.baseline)
    top_k = int(expected.get("top_k", 20))
    if args.rust_dump:
        got = load_json(args.rust_dump)
    else:
        code, stdout, stderr = run_rust_dump(args.logits, args.tokenizer, top_k)
        if code != 0:
            print("rust model logits dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        got = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    compare = compare_dumps(expected, got)
    print_report("model logits C/Rust comparator", compare)

    negative = Report()
    if args.negative_test:
        negative = run_negative_tests(expected, got)
        print_report("model logits C/Rust negative tests", negative)

    return 0 if compare.ok and negative.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
