#!/usr/bin/env python3
"""Validate the M10.8g3c B300 missing-MTP runtime smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json"
CURRENT_C = ROOT / "ds4-parity/baselines/cli/m8.12b/current-c.json"
STREAM_CONTRACT = ROOT / "ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json"

SCHEMA = "ds4.rust_mtp_missing_support_runtime_smoke.v1"
SOURCE = "rust-b300-mtp-missing-support-runtime-smoke"
MILESTONE = "M10.8g3c"
GUARD_CASE = "b300_missing_mtp_support_runtime_blocker"
STREAM_CASE = "b300_missing_mtp_support_model"
CURRENT_C_CASE = "mtp_missing_model"
B300_CONTEXT = "hou2-prod1"
B300_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
BASE_MODEL = "/workspace/ds4/ds4flash.gguf"
BASE_MODEL_SIZE = 86_720_111_488
MISSING_MTP = "/workspace/ds4/missing-mtp.gguf"
CANDIDATE_GLOBS = ["*mtp*.gguf", "*draft*.gguf"]
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
STDERR_TEXT = (
    "ds4: context buffers 23.39 MiB (ctx=128, backend=cuda, prefill_chunk=128, "
    "raw_kv_rows=256, compressed_kv_rows=34)\n"
    "ds4: cannot open model '/workspace/ds4/missing-mtp.gguf': No such file or directory\n"
)
STDERR_SHA256 = "826268e476a14b68cf733c113b9a8517c9c3209988de7dbb3bbd98e7f64f444a"
ARGV = [
    "--cuda",
    "-m",
    BASE_MODEL,
    "--mtp",
    MISSING_MTP,
    "--mtp-draft",
    "2",
    "--mtp-margin",
    "3",
    "--ctx",
    "128",
    "--tokens",
    "1",
    "--temp",
    "0",
    "--nothink",
    "-p",
    "Answer with one short noun: glacier.",
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected JSON object")
    return obj


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_guard_plan(workdir: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-mtp-runtime-guard-plan", "--quiet"],
        cwd=workdir,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    obj = json.loads(proc.stdout)
    if not isinstance(obj, dict):
        raise TypeError("runtime guard plan must be a JSON object")
    return obj


def named_cases(report: Report, value: Any, label: str, key: str = "id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    report.check(isinstance(value, list), f"{label} must be a list")
    if not isinstance(value, list):
        return result
    for index, item in enumerate(value):
        report.check(isinstance(item, dict), f"{label}[{index}] must be an object")
        if not isinstance(item, dict):
            continue
        case_id = item.get(key)
        report.check(isinstance(case_id, str) and bool(case_id), f"{label}[{index}].{key} missing")
        if isinstance(case_id, str) and case_id:
            result[case_id] = item
    return result


def current_c_case(current_c: dict[str, Any]) -> dict[str, Any]:
    for case in current_c.get("cases", []):
        if isinstance(case, dict) and case.get("id") == CURRENT_C_CASE:
            return case
    raise KeyError(CURRENT_C_CASE)


def build_live_summary(workdir: Path, candidate_binary: Path) -> dict[str, Any]:
    binary = candidate_binary if candidate_binary.is_absolute() else workdir / candidate_binary
    proc = subprocess.run(
        [str(binary), *ARGV],
        cwd=workdir,
        check=False,
        capture_output=True,
    )
    candidates = find_support_candidates(workdir)
    base_path = Path(BASE_MODEL)
    missing_path = Path(MISSING_MTP)
    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "guard_case": GUARD_CASE,
        "stream_case": STREAM_CASE,
        "current_c_baseline": "ds4-parity/baselines/cli/m8.12b/current-c.json:mtp_missing_model",
        "b300": {
            "context": B300_CONTEXT,
            "namespace": "default",
            "pod": B300_POD,
            "workdir": B300_WORKDIR,
        },
        "support_artifacts": {
            "base_model_path": BASE_MODEL,
            "base_model_exists": base_path.exists(),
            "base_model_bytes": base_path.stat().st_size if base_path.exists() else None,
            "expected_mtp_path": MISSING_MTP,
            "expected_mtp_path_exists": missing_path.exists(),
            "candidate_globs": CANDIDATE_GLOBS,
            "candidate_max_depth": 3,
            "mtp_candidates": candidates,
            "candidate_count": len(candidates),
            "candidate_search_stdout": format_candidate_stdout(candidates),
            "availability": "blocked_missing_mtp_model",
        },
        "runtime": {
            "binary": rel_to_workdir(workdir, binary),
            "argv": ARGV,
            "exit_code": proc.returncode,
            "expected_exit_code": 1,
            "stdout": {
                "base64": "",
                "bytes": len(proc.stdout),
                "sha256": sha256_bytes(proc.stdout),
            },
            "stderr": {
                "bytes": len(proc.stderr),
                "sha256": sha256_bytes(proc.stderr),
                "text": proc.stderr.decode("utf-8", errors="replace"),
            },
            "stderr_anchors": [
                "backend=cuda",
                "ds4: cannot open model '/workspace/ds4/missing-mtp.gguf': No such file or directory",
            ],
            "target_stream_visibility": "blocked_before_stream",
            "accepted_stream_delta": "blocked_before_stream",
            "checkpoint_delta": "0",
            "logits_source": "none",
            "mtp_n_raw_keep": 0,
            "cache_kvc_visibility": "none",
            "fallback": "blocked_missing_mtp_model",
            "error": "blocked_missing_mtp_model",
        },
    }


def rel_to_workdir(workdir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workdir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def find_support_candidates(workdir: Path) -> list[str]:
    candidates: list[str] = []
    max_depth = 3
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(workdir)
        except ValueError:
            continue
        if len(relative.parts) > max_depth:
            continue
        lower = path.name.lower()
        if (("mtp" in lower) or ("draft" in lower)) and lower.endswith(".gguf"):
            candidates.append(path.as_posix())
    return sorted(candidates)


def format_candidate_stdout(candidates: list[str]) -> str:
    return f"mtp_candidates={' '.join(candidates)}\n"


def validate(
    summary: dict[str, Any],
    guard_plan: dict[str, Any],
    current_c: dict[str, Any],
    stream_contract: dict[str, Any],
) -> Report:
    report = Report()
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == SOURCE, "summary source drift")
    report.check(summary.get("milestone") == MILESTONE, "summary milestone drift")
    report.check(summary.get("guard_case") == GUARD_CASE, "summary guard case drift")
    report.check(summary.get("stream_case") == STREAM_CASE, "summary stream case drift")
    validate_b300(report, summary.get("b300"))
    validate_support_artifacts(report, summary.get("support_artifacts"))
    validate_runtime(report, summary.get("runtime"), current_c_case(current_c))
    validate_guard_linkage(report, summary, guard_plan, stream_contract)
    validate_static_wiring(report)
    return report


def validate_b300(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "b300 metadata missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("context") == B300_CONTEXT, "B300 context drift")
    report.check(value.get("namespace") == "default", "B300 namespace drift")
    report.check(value.get("pod") == B300_POD, "B300 pod drift")
    report.check(value.get("workdir") == B300_WORKDIR, "B300 workdir drift")


def validate_support_artifacts(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "support artifact metadata missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("base_model_path") == BASE_MODEL, "base model path drift")
    report.check(value.get("base_model_exists") is True, "base model should exist on B300")
    report.check(value.get("base_model_bytes") == BASE_MODEL_SIZE, "base model size drift")
    report.check(value.get("expected_mtp_path") == MISSING_MTP, "missing-MTP path drift")
    report.check(value.get("expected_mtp_path_exists") is False, "missing-MTP path unexpectedly exists")
    report.check(value.get("candidate_globs") == CANDIDATE_GLOBS, "candidate globs drift")
    report.check(value.get("candidate_max_depth") == 3, "candidate max-depth drift")
    report.check(value.get("mtp_candidates") == [], "MTP support candidates unexpectedly present")
    report.check(value.get("candidate_count") == 0, "MTP support candidate count drift")
    report.check(value.get("candidate_search_stdout") == "mtp_candidates=\n", "candidate search output drift")
    report.check(value.get("availability") == "blocked_missing_mtp_model", "availability drift")


def validate_runtime(report: Report, value: Any, current_c: dict[str, Any]) -> None:
    report.check(isinstance(value, dict), "runtime summary missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("binary") == "target/debug/ds4-cli-one-shot-rs", "runtime binary drift")
    report.check(value.get("argv") == ARGV, "runtime argv drift")
    report.check("--mtp" in value.get("argv", []), "runtime argv missing --mtp")
    report.check("--mtp-draft" in value.get("argv", []), "runtime argv missing --mtp-draft")
    report.check(value.get("exit_code") == 1, "runtime exit code drift")
    report.check(value.get("expected_exit_code") == 1, "runtime expected exit code drift")
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    report.check(stdout == {"base64": "", "bytes": 0, "sha256": EMPTY_SHA256}, "runtime stdout drift")
    report.check(isinstance(stderr, dict), "runtime stderr missing")
    if isinstance(stderr, dict):
        report.check(stderr.get("bytes") == len(STDERR_TEXT.encode()), "runtime stderr byte drift")
        report.check(stderr.get("sha256") == STDERR_SHA256, "runtime stderr hash drift")
        report.check(stderr.get("text") == STDERR_TEXT, "runtime stderr text drift")
    report.check(value.get("stderr_anchors") == current_c.get("stderr_anchors"), "stderr anchor drift")
    for key in [
        "target_stream_visibility",
        "accepted_stream_delta",
        "checkpoint_delta",
        "logits_source",
        "cache_kvc_visibility",
        "fallback",
        "error",
    ]:
        report.check(value.get(key) is not None, f"runtime {key} missing")
    report.check(value.get("target_stream_visibility") == "blocked_before_stream", "runtime visibility drift")
    report.check(value.get("accepted_stream_delta") == "blocked_before_stream", "runtime stream drift")
    report.check(value.get("checkpoint_delta") == "0", "runtime checkpoint drift")
    report.check(value.get("logits_source") == "none", "runtime logits source drift")
    report.check(value.get("mtp_n_raw_keep") in (0, "0"), "runtime mtp_n_raw drift")
    report.check(value.get("cache_kvc_visibility") == "none", "runtime cache/KVC drift")
    report.check(value.get("fallback") == "blocked_missing_mtp_model", "runtime fallback drift")
    report.check(value.get("error") == "blocked_missing_mtp_model", "runtime error drift")
    report.check(current_c.get("availability") == "blocked_missing_mtp_model", "current-C availability drift")
    report.check(current_c.get("exit_code") == value.get("exit_code"), "current-C exit code mismatch")
    report.check(current_c.get("stdout") == value.get("stdout"), "current-C stdout mismatch")
    report.check(current_c.get("stderr_normalized") == STDERR_TEXT, "current-C stderr text drift")
    report.check(current_c.get("stderr_normalized_sha256") == STDERR_SHA256, "current-C stderr hash drift")


def validate_guard_linkage(
    report: Report,
    summary: dict[str, Any],
    guard_plan: dict[str, Any],
    stream_contract: dict[str, Any],
) -> None:
    guard_cases = named_cases(report, guard_plan.get("cases"), "guard.cases")
    guard = guard_cases.get(GUARD_CASE)
    report.check(guard is not None, f"missing guard case {GUARD_CASE}")
    stream_cases = named_cases(report, stream_contract.get("stream_cases"), "stream.stream_cases")
    stream = stream_cases.get(STREAM_CASE)
    report.check(stream is not None, f"missing stream case {STREAM_CASE}")
    runtime = summary.get("runtime") if isinstance(summary.get("runtime"), dict) else {}
    if guard:
        report.check(guard.get("source_stream_case") == STREAM_CASE, "guard source stream drift")
        for key in [
            "accepted_stream_delta",
            "checkpoint_delta",
            "logits_source",
            "cache_kvc_visibility",
            "fallback",
            "error",
            "target_stream_visibility",
        ]:
            report.check(runtime.get(key) == guard.get(key), f"runtime/guard {key} drift")
        report.check(runtime.get("mtp_n_raw_keep") in (guard.get("mtp_n_raw_keep"), str(guard.get("mtp_n_raw_keep"))), "runtime/guard mtp_n_raw drift")
        report.check(guard.get("live_status") == "blocked_missing_mtp_model", "guard live-status drift")
    if stream:
        for key in [
            "accepted_stream_delta",
            "checkpoint_delta",
            "logits_source",
            "cache_kvc_visibility",
            "fallback",
            "error",
        ]:
            report.check(runtime.get(key) == stream.get(key), f"runtime/stream {key} drift")
        report.check(stream.get("live_status") == "blocked_missing_mtp_model", "stream live-status drift")


def validate_static_wiring(report: Report) -> None:
    run_report = (ROOT / "ds4-parity/run_parity_report.py").read_text()
    readme = (ROOT / "ds4-parity/README.md").read_text()
    status = (ROOT / ".memory/TODO.md").read_text()
    report.check("compare_mtp_runtime_missing_support.py" in run_report, "unified report missing comparator")
    report.check("M10.8g3c B300 MTP missing-support runtime smoke rerun" in run_report, "B300 rerun hook missing")
    report.check("compare_mtp_runtime_missing_support.py --negative-test" in readme, "README missing comparator command")
    report.check("M10.8g3c: B300 Missing-Support Runtime Smoke" in status, "TODO missing M10.8g3c stage")


def run_negative_tests(
    summary: dict[str, Any],
    guard_plan: dict[str, Any],
    current_c: dict[str, Any],
    stream_contract: dict[str, Any],
) -> Report:
    report = Report()
    mutations = [
        ("runtime exit drift", lambda data, _guard: data["runtime"].update({"exit_code": 0})),
        ("stdout drift", lambda data, _guard: data["runtime"]["stdout"].update({"bytes": 1})),
        ("stderr hash drift", lambda data, _guard: data["runtime"]["stderr"].update({"sha256": "0" * 64})),
        (
            "candidate appears",
            lambda data, _guard: data["support_artifacts"].update(
                {"mtp_candidates": ["/workspace/ds4/draft.gguf"], "candidate_count": 1}
            ),
        ),
        ("stream visibility drift", lambda data, _guard: data["runtime"].update({"target_stream_visibility": "target_only"})),
        (
            "cache visibility drift",
            lambda data, _guard: data["runtime"].update({"cache_kvc_visibility": "first_token checkpoint only"}),
        ),
        (
            "guard source drift",
            lambda _data, guard: mutate_guard(guard, "source_stream_case", "mtp_disabled_after_first_token"),
        ),
    ]
    for name, mutate in mutations:
        summary_copy = copy.deepcopy(summary)
        guard_copy = copy.deepcopy(guard_plan)
        mutate(summary_copy, guard_copy)
        result = validate(summary_copy, guard_copy, current_c, stream_contract)
        report.check(not result.ok, f"negative mutation did not fail: {name}")
    return report


def mutate_guard(guard_plan: dict[str, Any], key: str, value: Any) -> None:
    for case in guard_plan["cases"]:
        if case.get("id") == GUARD_CASE:
            case[key] = value
            return
    raise AssertionError(GUARD_CASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--guard-plan", type=Path)
    parser.add_argument("--current-c", type=Path, default=CURRENT_C)
    parser.add_argument("--stream-contract", type=Path, default=STREAM_CONTRACT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--candidate-binary", type=Path, default=Path("target/debug/ds4-cli-one-shot-rs"))
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        current_c = load_json(args.current_c)
        stream_contract = load_json(args.stream_contract)
        guard_plan = load_json(args.guard_plan) if args.guard_plan else run_guard_plan(args.workdir)
        summary = (
            build_live_summary(args.workdir, args.candidate_binary)
            if args.live
            else load_json(args.summary)
        )
    except Exception as exc:
        print(f"MTP runtime missing-support smoke: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(summary, guard_plan, current_c, stream_contract)
    if not report.ok:
        print("MTP runtime missing-support smoke: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    if args.write_summary:
        write_json(args.write_summary, summary)
    print(f"MTP runtime missing-support smoke: PASS, {report.checks} checks")
    if args.negative_test:
        negative = run_negative_tests(summary, guard_plan, current_c, stream_contract)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("MTP runtime missing-support negative tests: PASS, 7 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
