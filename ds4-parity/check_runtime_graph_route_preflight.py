#!/usr/bin/env python3
"""Validate the M10.9b Rust runtime graph route selector preflight."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json"
SCHEMA = "ds4.runtime_graph_route_preflight.v1"
SOURCE = "m10.9b-runtime-graph-route-preflight"
MILESTONE = "M10.9b"
NEXT_STAGE = "M10.9c"
UNSUPPORTED_CODE = 99
MISSING_MODEL = "/tmp/ds4-m109b-missing-model.gguf"
SERVER_CACHE_DIR = "/tmp/ds4-m109b-route-preflight-kv"
PROMPT_FILE = "ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt"


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


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    binary: str
    argv: list[str]
    route: str
    category: str
    backend: str
    expected_exit: int
    cache_dir: str | None = None


CASES = [
    CaseSpec(
        "cli_one_shot_default_missing_model",
        "ds4-cli-one-shot-rs",
        [
            "--cpu",
            "-m",
            MISSING_MODEL,
            "-p",
            "Answer with one short noun: glacier.",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
        ],
        "target-stream",
        "target_stream_missing_model",
        "cpu",
        1,
    ),
    CaseSpec(
        "cli_one_shot_disabled_route_missing_model",
        "ds4-cli-one-shot-rs",
        [
            "--runtime-graph",
            "off",
            "--cpu",
            "-m",
            MISSING_MODEL,
            "-p",
            "Answer with one short noun: glacier.",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
        ],
        "target-stream",
        "disabled_route_missing_model",
        "cpu",
        1,
    ),
    CaseSpec(
        "cli_one_shot_graph_unsupported",
        "ds4-cli-one-shot-rs",
        [
            "--runtime-graph",
            "graph",
            "--cuda",
            "-m",
            MISSING_MODEL,
            "-p",
            "Answer with one short noun: glacier.",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
        ],
        "graph",
        "unsupported_graph_route",
        "cuda",
        UNSUPPORTED_CODE,
    ),
    CaseSpec(
        "cli_interactive_graph_unsupported",
        "ds4-cli-interactive-rs",
        [
            "--runtime-graph",
            "graph",
            "--cpu",
            "-m",
            MISSING_MODEL,
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
        ],
        "graph",
        "unsupported_graph_route",
        "cpu",
        UNSUPPORTED_CODE,
    ),
    CaseSpec(
        "direct_interactive_graph_unsupported",
        "ds4-interactive-runtime-rs",
        [
            "--runtime-graph",
            "graph",
            "--cuda",
            "-m",
            MISSING_MODEL,
            "--read-prompt-file",
            PROMPT_FILE,
            "--next-prompt",
            "Answer with one short noun: glacier.",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
        ],
        "graph",
        "unsupported_graph_route",
        "cuda",
        UNSUPPORTED_CODE,
    ),
    CaseSpec(
        "server_target_stream_missing_model",
        "ds4-server-runtime-rs",
        [
            "--runtime-graph",
            "target-stream",
            "--cpu",
            "-m",
            MISSING_MODEL,
            "--port",
            "18109",
            "--tokens",
            "1",
        ],
        "target-stream",
        "target_stream_missing_model",
        "cpu",
        1,
    ),
    CaseSpec(
        "server_graph_unsupported_with_cache",
        "ds4-server-runtime-rs",
        [
            "--runtime-graph",
            "graph",
            "--cuda",
            "-m",
            MISSING_MODEL,
            "--port",
            "18110",
            "--tokens",
            "1",
            "--kv-disk-dir",
            SERVER_CACHE_DIR,
        ],
        "graph",
        "unsupported_graph_route",
        "cuda",
        UNSUPPORTED_CODE,
        SERVER_CACHE_DIR,
    ),
    CaseSpec(
        "server_invalid_route",
        "ds4-server-runtime-rs",
        ["--runtime-graph", "fallback", "--cpu", "-m", MISSING_MODEL],
        "invalid",
        "invalid_selector",
        "cpu",
        2,
    ),
]


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


def blob(data: bytes, include_text: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "base64": base64.b64encode(data).decode("ascii"),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    if include_text:
        result["text"] = data.decode("utf-8", errors="replace")
    return result


def build_binaries(workdir: Path) -> None:
    command = [
        "cargo",
        "build",
        "-p",
        "ds4-engine",
        "--bin",
        "ds4-cli-one-shot-rs",
        "--bin",
        "ds4-cli-interactive-rs",
        "--bin",
        "ds4-interactive-runtime-rs",
        "--bin",
        "ds4-server-runtime-rs",
    ]
    proc = subprocess.run(command, cwd=workdir, text=True, capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def build_summary(workdir: Path, candidate_dir: Path, do_build: bool) -> dict[str, Any]:
    if do_build:
        build_binaries(workdir)
    if Path(MISSING_MODEL).exists():
        raise RuntimeError(f"missing-model fixture unexpectedly exists: {MISSING_MODEL}")
    if Path(SERVER_CACHE_DIR).exists():
        shutil.rmtree(SERVER_CACHE_DIR)

    cases = []
    for spec in CASES:
        binary = candidate_dir / spec.binary
        if not binary.is_file():
            raise RuntimeError(f"missing candidate binary: {binary}")
        if spec.cache_dir and Path(spec.cache_dir).exists():
            shutil.rmtree(spec.cache_dir)
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        proc = subprocess.run(
            [str(binary), *spec.argv],
            cwd=workdir,
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
        cache_dir_created = Path(spec.cache_dir).exists() if spec.cache_dir else None
        if spec.cache_dir and cache_dir_created:
            shutil.rmtree(spec.cache_dir)
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        cases.append(
            {
                "id": spec.case_id,
                "binary": f"target/debug/{spec.binary}",
                "argv": spec.argv,
                "route": spec.route,
                "category": spec.category,
                "backend": spec.backend,
                "exit_code": proc.returncode,
                "expected_exit_code": spec.expected_exit,
                "stdout": blob(proc.stdout, include_text=False),
                "stderr": blob(proc.stderr),
                "stderr_anchors": stderr_anchors(spec, stderr_text),
                "model_open_attempted": "cannot open model" in stderr_text,
                "stream_visibility": "blocked_before_stream"
                if spec.category in {"unsupported_graph_route", "invalid_selector"}
                else "no_generation_missing_model",
                "checkpoint_delta": "0",
                "cache_kvc_visibility": "none",
                "cache_dir": spec.cache_dir,
                "cache_dir_created": cache_dir_created,
            }
        )

    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "milestone": MILESTONE,
        "next_stage": NEXT_STAGE,
        "route_selector": {
            "options": ["--runtime-graph", "--runtime-graph-route"],
            "default": "target-stream",
            "disabled_alias": "off",
            "supported_values": ["target-stream", "off", "graph"],
            "unsupported_route": "graph",
            "unsupported_exit_code": UNSUPPORTED_CODE,
        },
        "fixtures": {
            "missing_model": MISSING_MODEL,
            "server_cache_dir": SERVER_CACHE_DIR,
            "prompt_file": PROMPT_FILE,
        },
        "cases": cases,
        "acceptance": {
            "graph_route_selection": "explicit",
            "default_behavior": "target-stream route unchanged",
            "unsupported_route": "fails before model open, stream output, or cache directory creation",
            "claim_boundary": "no model-backed Rust graph parity claim",
        },
    }


def stderr_anchors(spec: CaseSpec, stderr_text: str) -> list[str]:
    if spec.category == "unsupported_graph_route":
        return [
            f"{spec.binary}: --runtime-graph graph is not implemented yet",
            "use --runtime-graph target-stream",
        ]
    if spec.category == "invalid_selector":
        return [
            "invalid runtime graph route: fallback",
            "valid runtime graph routes are: target-stream, off, graph",
        ]
    anchors = ["cannot open model", MISSING_MODEL]
    if "backend=cpu" in stderr_text:
        anchors.append("backend=cpu")
    return anchors


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
        report.check(isinstance(case_id, str) and bool(case_id), f"cases[{index}].id missing")
        if isinstance(case_id, str):
            result[case_id] = item
    return result


def validate(summary: dict[str, Any]) -> Report:
    report = Report()
    report.check(summary.get("schema") == SCHEMA, "summary schema drift")
    report.check(summary.get("source") == SOURCE, "summary source drift")
    report.check(summary.get("milestone") == MILESTONE, "summary milestone drift")
    report.check(summary.get("next_stage") == NEXT_STAGE, "summary next-stage drift")
    validate_selector(report, summary.get("route_selector"))
    validate_fixtures(report, summary.get("fixtures"))
    cases = named_cases(report, summary.get("cases"))
    report.check(set(cases) == {case.case_id for case in CASES}, "case id set drift")
    for spec in CASES:
        case = cases.get(spec.case_id)
        report.check(case is not None, f"missing case {spec.case_id}")
        if case:
            validate_case(report, spec, case)
    validate_acceptance(report, summary.get("acceptance"))
    validate_static_wiring(report)
    return report


def validate_selector(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "route selector missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("options") == ["--runtime-graph", "--runtime-graph-route"], "route option drift")
    report.check(value.get("default") == "target-stream", "route default drift")
    report.check(value.get("disabled_alias") == "off", "route disabled alias drift")
    report.check(value.get("supported_values") == ["target-stream", "off", "graph"], "route values drift")
    report.check(value.get("unsupported_route") == "graph", "unsupported route drift")
    report.check(value.get("unsupported_exit_code") == UNSUPPORTED_CODE, "unsupported exit code drift")


def validate_fixtures(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "fixtures missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("missing_model") == MISSING_MODEL, "missing model fixture drift")
    report.check(value.get("server_cache_dir") == SERVER_CACHE_DIR, "server cache fixture drift")
    report.check(value.get("prompt_file") == PROMPT_FILE, "prompt file fixture drift")


def validate_case(report: Report, spec: CaseSpec, case: dict[str, Any]) -> None:
    report.check(case.get("binary") == f"target/debug/{spec.binary}", f"{spec.case_id}.binary drift")
    report.check(case.get("argv") == spec.argv, f"{spec.case_id}.argv drift")
    report.check(case.get("route") == spec.route, f"{spec.case_id}.route drift")
    report.check(case.get("category") == spec.category, f"{spec.case_id}.category drift")
    report.check(case.get("backend") == spec.backend, f"{spec.case_id}.backend drift")
    report.check(case.get("exit_code") == spec.expected_exit, f"{spec.case_id}.exit drift")
    report.check(case.get("expected_exit_code") == spec.expected_exit, f"{spec.case_id}.expected exit drift")
    validate_blob(report, case.get("stdout"), f"{spec.case_id}.stdout")
    validate_blob(report, case.get("stderr"), f"{spec.case_id}.stderr")
    stdout = case.get("stdout") if isinstance(case.get("stdout"), dict) else {}
    stderr = case.get("stderr") if isinstance(case.get("stderr"), dict) else {}
    stderr_text = str(stderr.get("text", ""))
    report.check(stdout.get("bytes") == 0, f"{spec.case_id}.stdout not empty")
    for anchor in case.get("stderr_anchors", []):
        report.check(isinstance(anchor, str), f"{spec.case_id}.anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor in stderr_text, f"{spec.case_id}.stderr missing {anchor!r}")
    report.check(case.get("checkpoint_delta") == "0", f"{spec.case_id}.checkpoint drift")
    report.check(case.get("cache_kvc_visibility") == "none", f"{spec.case_id}.cache drift")
    if spec.category == "unsupported_graph_route":
        report.check(case.get("model_open_attempted") is False, f"{spec.case_id}.model open attempted")
        report.check("cannot open model" not in stderr_text, f"{spec.case_id}.missing-model leak")
        report.check(case.get("stream_visibility") == "blocked_before_stream", f"{spec.case_id}.stream drift")
        report.check(case.get("stderr_anchors") == stderr_anchors(spec, stderr_text), f"{spec.case_id}.anchors drift")
    elif spec.category == "invalid_selector":
        report.check(case.get("model_open_attempted") is False, f"{spec.case_id}.model open attempted")
        report.check(case.get("stream_visibility") == "blocked_before_stream", f"{spec.case_id}.stream drift")
        report.check(case.get("stderr_anchors") == stderr_anchors(spec, stderr_text), f"{spec.case_id}.anchors drift")
    else:
        report.check(case.get("model_open_attempted") is True, f"{spec.case_id}.model open missing")
        report.check("runtime graph route is not implemented" not in stderr_text, f"{spec.case_id}.route blocker leak")
        report.check(case.get("stream_visibility") == "no_generation_missing_model", f"{spec.case_id}.stream drift")
    if spec.cache_dir:
        report.check(case.get("cache_dir") == spec.cache_dir, f"{spec.case_id}.cache dir drift")
        report.check(case.get("cache_dir_created") is False, f"{spec.case_id}.cache dir was created")


def validate_blob(report: Report, value: Any, label: str) -> None:
    report.check(isinstance(value, dict), f"{label} missing")
    if not isinstance(value, dict):
        return
    raw = value.get("base64")
    report.check(isinstance(raw, str), f"{label}.base64 missing")
    if isinstance(raw, str):
        try:
            data = base64.b64decode(raw.encode("ascii"), validate=True)
        except Exception as exc:
            report.check(False, f"{label}.base64 invalid: {exc}")
            data = b""
        report.check(value.get("bytes") == len(data), f"{label}.bytes drift")
        report.check(value.get("sha256") == sha256_bytes(data), f"{label}.sha drift")


def validate_acceptance(report: Report, value: Any) -> None:
    report.check(isinstance(value, dict), "acceptance missing")
    if not isinstance(value, dict):
        return
    report.check(value.get("graph_route_selection") == "explicit", "selection acceptance drift")
    report.check(value.get("default_behavior") == "target-stream route unchanged", "default acceptance drift")
    report.check(
        value.get("unsupported_route")
        == "fails before model open, stream output, or cache directory creation",
        "unsupported acceptance drift",
    )
    report.check(value.get("claim_boundary") == "no model-backed Rust graph parity claim", "claim boundary drift")


def validate_static_wiring(report: Report) -> None:
    files = {
        "engine": ROOT / "rust/ds4-engine/src/lib.rs",
        "cli_parse": ROOT / "rust/ds4-gguf/src/cli_parse.rs",
        "one_shot": ROOT / "rust/ds4-engine/src/bin/ds4-cli-one-shot-rs.rs",
        "interactive": ROOT / "rust/ds4-engine/src/bin/ds4-cli-interactive-rs.rs",
        "direct_interactive": ROOT / "rust/ds4-engine/src/bin/ds4-interactive-runtime-rs.rs",
        "server": ROOT / "rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs",
        "run_report": ROOT / "ds4-parity/run_parity_report.py",
        "readme": ROOT / "ds4-parity/README.md",
        "todo": ROOT / ".memory/TODO.md",
        "status": ROOT / ".memory/status.md",
        "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
        "matrix": ROOT / "ds4-parity/baselines/graph/m10.9a/runtime-graph-closure-matrix.json",
    }
    text = {name: path.read_text() for name, path in files.items()}
    report.check("pub enum RuntimeGraphRoute" in text["engine"], "engine route enum missing")
    report.check("RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE: i32 = 99" in text["engine"], "unsupported code missing")
    report.check('"off" | "disabled"' in text["engine"], "disabled alias missing")
    report.check("runtime_graph_route: CliRuntimeGraphRoute" in text["cli_parse"], "CLI parse route field missing")
    report.check('"--runtime-graph" | "--runtime-graph-route"' in text["cli_parse"], "CLI parse route option missing")
    for key in ["one_shot", "interactive", "direct_interactive", "server"]:
        report.check(".fail_closed(" in text[key], f"{key} fail-closed call missing")
    report.check("runtime_graph_route: RuntimeGraphRoute" in text["server"], "server route config missing")
    report.check("check_runtime_graph_route_preflight.py" in text["run_report"], "unified report entry missing")
    report.check("M10.9b Runtime graph route preflight" in text["run_report"], "unified report label missing")
    report.check("check_runtime_graph_route_preflight.py --negative-test" in text["readme"], "README command missing")
    report.check("M10.9b: Rust Runtime Graph Route Switch And Preflight" in text["roadmap"], "roadmap stage missing")
    todo_m109b = section(text["todo"], "#### M10.9b:", "#### M10.9c:")
    report.check("- Status: complete" in todo_m109b, "TODO M10.9b complete marker missing")
    report.check("M10.9c B300 Official-Vector Rust Runtime Gate" in text["status"], "status did not advance to M10.9c")
    matrix = json.loads(text["matrix"])
    gate = next((item for item in matrix.get("gates", []) if item.get("id") == "M10.9b"), {})
    report.check(gate.get("comparator") == "ds4-parity/check_runtime_graph_route_preflight.py", "matrix comparator drift")
    report.check(gate.get("artifact") == "ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json", "matrix artifact drift")


def section(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    body = text.split(start, 1)[1]
    return body.split(end, 1)[0] if end in body else body


def run_negative_tests(summary: dict[str, Any]) -> Report:
    report = Report()

    def expect_failure(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        candidate = copy.deepcopy(summary)
        mutate(candidate)
        result = validate(candidate)
        report.check(not result.ok, f"negative mutation did not fail: {name}")

    expect_failure("schema drift", lambda data: data.update({"schema": "bad"}))
    expect_failure(
        "unsupported code drift",
        lambda data: data["route_selector"].update({"unsupported_exit_code": 0}),
    )
    expect_failure(
        "graph exit drift",
        lambda data: case(data, "cli_one_shot_graph_unsupported").update({"exit_code": 0}),
    )
    expect_failure(
        "graph attempted model open",
        lambda data: case(data, "server_graph_unsupported_with_cache").update({"model_open_attempted": True}),
    )
    expect_failure(
        "cache dir created",
        lambda data: case(data, "server_graph_unsupported_with_cache").update({"cache_dir_created": True}),
    )
    expect_failure(
        "target missing model skipped",
        lambda data: case(data, "cli_one_shot_default_missing_model").update({"model_open_attempted": False}),
    )
    expect_failure(
        "stdout nonempty",
        lambda data: case(data, "server_invalid_route")["stdout"].update({"bytes": 1}),
    )
    expect_failure(
        "claim boundary drift",
        lambda data: data["acceptance"].update({"claim_boundary": "full runtime graph parity"}),
    )
    return report


def case(data: dict[str, Any], case_id: str) -> dict[str, Any]:
    for item in data["cases"]:
        if item.get("id") == case_id:
            return item
    raise AssertionError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--workdir", type=Path, default=ROOT)
    parser.add_argument("--candidate-dir", type=Path, default=Path("target/debug"))
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.write_summary:
            candidate_dir = args.candidate_dir
            if not candidate_dir.is_absolute():
                candidate_dir = args.workdir / candidate_dir
            summary = build_summary(args.workdir, candidate_dir, not args.no_build)
            write_json(args.write_summary, summary)
        else:
            summary = load_json(args.summary)
    except Exception as exc:
        print(f"Runtime graph route preflight: FAIL: {exc}", file=sys.stderr)
        return 1

    report = validate(summary)
    if not report.ok:
        print("Runtime graph route preflight: FAIL")
        for error in report.errors:
            print(f"- {error}")
        return 1
    print(f"Runtime graph route preflight: PASS, {report.checks} checks")
    if args.negative_test:
        negative = run_negative_tests(summary)
        if not negative.ok:
            for error in negative.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Runtime graph route preflight negative tests: PASS, 8 mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
