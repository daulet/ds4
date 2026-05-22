#!/usr/bin/env python3
"""Validate the M8.6 current-C CLI logprob/perplexity diagnostic oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.6"
FIXTURE_DIR = ROOT / "ds4-parity" / "baselines" / "cli-fixtures" / "m8.6"
BASELINE = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"

EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_SIZE = 86720111488
SCORE_ABS_TOLERANCE = 1e-5
B300_KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
B300_CONTEXT = "hou2-prod1"
B300_NAMESPACE = "default"
B300_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
B300_MODEL = "/workspace/ds4/ds4flash.gguf"

PERPLEXITY_RE = re.compile(
    rb"tokens=(?P<tokens>\d+) scored=(?P<scored>\d+) "
    rb"nll=(?P<nll>[0-9.eE+-]+) avg_nll=(?P<avg_nll>[0-9.eE+-]+) ppl=(?P<ppl>[0-9.eE+-]+)\n"
)


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    kind: str
    argv: tuple[str, ...]
    exit_code: int
    output_file: str | None = None
    expected_steps: int = 0
    expected_top_k: int = 0
    stderr_anchors: tuple[str, ...] = ()
    stdout_kind: str = "empty"


CASES: tuple[DiagnosticCase, ...] = (
    DiagnosticCase(
        "logprobs_inline_top3",
        "dump_logprobs",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "512",
            "--tokens",
            "2",
            "--logprobs-top-k",
            "3",
            "--dump-logprobs",
            "/tmp/ds4-cli-m8.6/logprobs_inline_top3.json",
            "-p",
            "M8.6 logprob inline prompt.",
        ),
        0,
        output_file="/tmp/ds4-cli-m8.6/logprobs_inline_top3.json",
        expected_steps=2,
        expected_top_k=3,
        stderr_anchors=("ds4: context buffers",),
    ),
    DiagnosticCase(
        "logprobs_prompt_file_top5",
        "dump_logprobs",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "512",
            "--tokens",
            "1",
            "--logprobs-top-k",
            "5",
            "--dump-logprobs",
            "/tmp/ds4-cli-m8.6/logprobs_prompt_file_top5.json",
            "--prompt-file",
            "ds4-parity/baselines/cli-fixtures/m8.6/logprob_prompt_file.txt",
        ),
        0,
        output_file="/tmp/ds4-cli-m8.6/logprobs_prompt_file_top5.json",
        expected_steps=1,
        expected_top_k=5,
        stderr_anchors=("ds4: context buffers",),
    ),
    DiagnosticCase(
        "logprobs_bad_output_path",
        "dump_logprobs_error",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "512",
            "--tokens",
            "1",
            "--logprobs-top-k",
            "3",
            "--dump-logprobs",
            "/tmp/ds4-cli-m8.6/missing-dir/out.json",
            "-p",
            "M8.6 bad output path.",
        ),
        1,
        expected_steps=0,
        expected_top_k=0,
        stderr_anchors=("ds4: context buffers", "ds4: failed to open --dump-logprobs file:"),
    ),
    DiagnosticCase(
        "perplexity_basic",
        "perplexity",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "64",
            "--tokens",
            "4",
            "--perplexity-file",
            "ds4-parity/baselines/cli-fixtures/m8.6/perplexity_text.txt",
        ),
        0,
        stdout_kind="perplexity",
        stderr_anchors=("ds4: context buffers", "ds4: perplexity scored 4/4"),
    ),
)


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(report: Report, value: Any, label: str) -> bytes:
    report.check(isinstance(value, str), f"{label}: expected base64 string")
    if not isinstance(value, str):
        return b""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        report.check(False, f"{label}: invalid base64: {exc}")
        return b""


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, label: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{label}: expected array")
    return obj if isinstance(obj, list) else []


def capture_bytes(data: bytes) -> dict[str, Any]:
    return {
        "base64": b64(data),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def run_case(binary: Path, case: DiagnosticCase) -> dict[str, Any]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    if case.output_file:
        Path(case.output_file).unlink(missing_ok=True)
    proc = subprocess.run(
        [str(binary), *case.argv],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    output = None
    if case.output_file and Path(case.output_file).exists():
        raw = Path(case.output_file).read_bytes()
        output_json = json.loads(raw.decode("utf-8"))
        output = {
            "path": case.output_file,
            "raw": capture_bytes(raw),
            "json": output_json,
            "summary": summarize_logprob_json(output_json),
        }
    captured = {
        "id": case.case_id,
        "kind": case.kind,
        "argv": list(case.argv),
        "exit_code": proc.returncode,
        "stdout": capture_bytes(proc.stdout),
        "stderr": capture_bytes(proc.stderr),
        "stderr_anchors": list(case.stderr_anchors),
        "stdout_kind": case.stdout_kind,
        "expected_steps": case.expected_steps,
        "expected_top_k": case.expected_top_k,
        "output_file": output,
    }
    if case.stdout_kind == "perplexity":
        captured["perplexity"] = parse_perplexity_stdout(proc.stdout)
    return captured


def capture_baseline(binary: Path, model_sha256: str) -> dict[str, Any]:
    if not binary.is_file():
        raise SystemExit(f"missing CLI binary: {binary}; build ds4 first")
    model = Path(B300_MODEL)
    if not model.is_file():
        raise SystemExit(f"missing model: {B300_MODEL}")
    Path("/tmp/ds4-cli-m8.6").mkdir(parents=True, exist_ok=True)
    shutil.rmtree("/tmp/ds4-cli-m8.6/missing-dir", ignore_errors=True)
    return {
        "schema": "ds4.cli_diagnostics_oracle.v1",
        "source": "current-c-cli-diagnostics",
        "binary": "./ds4",
        "model": {
            "path": B300_MODEL,
            "size": model.stat().st_size,
            "sha256": model_sha256,
        },
        "numeric_policy": {
            "score_abs_tolerance_for_future_comparators": SCORE_ABS_TOLERANCE,
            "source": "M6 model-logits comparator ordinary absolute tolerance",
        },
        "cases": [run_case(binary, case) for case in CASES],
    }


def summarize_logprob_json(obj: Any) -> dict[str, Any]:
    root = obj if isinstance(obj, dict) else {}
    steps = root.get("steps") if isinstance(root.get("steps"), list) else []
    return {
        "prompt_tokens": root.get("prompt_tokens"),
        "ctx": root.get("ctx"),
        "top_k": root.get("top_k"),
        "steps": len(steps),
        "selected_ids": [
            step.get("selected", {}).get("id") for step in steps if isinstance(step, dict)
        ],
    }


def parse_perplexity_stdout(stdout: bytes) -> dict[str, Any]:
    match = PERPLEXITY_RE.fullmatch(stdout)
    if not match:
        return {"parse_ok": False}
    out: dict[str, Any] = {"parse_ok": True}
    for key in ("tokens", "scored"):
        out[key] = int(match.group(key))
    for key in ("nll", "avg_nll", "ppl"):
        out[key] = float(match.group(key))
    return out


def check_bytes(report: Report, raw: Any, label: str) -> bytes:
    obj = require_dict(report, raw, label)
    data = unb64(report, obj.get("base64"), f"{label}.base64")
    report.check(obj.get("bytes") == len(data), f"{label}.bytes drift")
    report.check(obj.get("sha256") == sha256_bytes(data), f"{label}.sha256 drift")
    return data


def check_token(report: Report, raw: Any, label: str) -> None:
    token = require_dict(report, raw, label)
    report.check(isinstance(token.get("id"), int) and token["id"] >= 0, f"{label}.id invalid")
    text = token.get("text")
    raw_bytes = require_list(report, token.get("bytes"), f"{label}.bytes")
    report.check(isinstance(text, str), f"{label}.text invalid")
    for idx, byte in enumerate(raw_bytes):
        report.check(isinstance(byte, int) and 0 <= byte <= 255, f"{label}.bytes[{idx}] invalid")


def check_logprob_json(report: Report, obj: Any, expected: DiagnosticCase, label: str) -> None:
    root = require_dict(report, obj, label)
    report.check(root.get("source") == "ds4", f"{label}.source drift")
    report.check(root.get("ctx") == argv_int(expected.argv, "--ctx"), f"{label}.ctx drift")
    report.check(root.get("top_k") == expected.expected_top_k, f"{label}.top_k drift")
    report.check(isinstance(root.get("prompt_tokens"), int) and root["prompt_tokens"] > 0, f"{label}.prompt_tokens invalid")
    steps = require_list(report, root.get("steps"), f"{label}.steps")
    report.check(len(steps) == expected.expected_steps, f"{label}.steps length drift")
    for idx, raw_step in enumerate(steps):
        step = require_dict(report, raw_step, f"{label}.steps[{idx}]")
        report.check(step.get("step") == idx, f"{label}.steps[{idx}].step drift")
        check_token(report, step.get("selected"), f"{label}.steps[{idx}].selected")
        top = require_list(report, step.get("top_logprobs"), f"{label}.steps[{idx}].top_logprobs")
        report.check(0 < len(top) <= expected.expected_top_k, f"{label}.steps[{idx}].top_logprobs length drift")
        seen: set[int] = set()
        for score_idx, raw_score in enumerate(top):
            score = require_dict(report, raw_score, f"{label}.steps[{idx}].top_logprobs[{score_idx}]")
            check_token(report, score.get("token"), f"{label}.steps[{idx}].top_logprobs[{score_idx}].token")
            token_obj = score.get("token")
            token_id = token_obj.get("id") if isinstance(token_obj, dict) else None
            if isinstance(token_id, int):
                report.check(token_id not in seen, f"{label}.steps[{idx}].duplicate top id {token_id}")
                seen.add(token_id)
            for key in ("logit", "logprob"):
                value = score.get(key)
                report.check(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label}.steps[{idx}].top_logprobs[{score_idx}].{key} invalid")


def argv_int(argv: tuple[str, ...], opt: str) -> int | None:
    for idx, arg in enumerate(argv):
        if arg == opt and idx + 1 < len(argv):
            return int(argv[idx + 1])
    return None


def check_perplexity(report: Report, raw: Any, label: str) -> None:
    obj = require_dict(report, raw, label)
    report.check(obj.get("parse_ok") is True, f"{label}.parse_ok drift")
    tokens = obj.get("tokens")
    scored = obj.get("scored")
    report.check(isinstance(tokens, int) and tokens > 32, f"{label}.tokens invalid")
    report.check(scored == 4, f"{label}.scored drift")
    nll = obj.get("nll")
    avg = obj.get("avg_nll")
    ppl = obj.get("ppl")
    for key, value in (("nll", nll), ("avg_nll", avg), ("ppl", ppl)):
        report.check(isinstance(value, float) and math.isfinite(value) and value > 0.0, f"{label}.{key} invalid")
    if isinstance(nll, float) and isinstance(avg, float):
        report.check(abs(avg - nll / 4.0) <= 5e-9, f"{label}.avg_nll does not match nll/scored")
    if isinstance(avg, float) and isinstance(ppl, float):
        report.check(abs(ppl - math.exp(avg)) <= 5e-7, f"{label}.ppl does not match exp(avg_nll)")


def check_case(report: Report, raw: Any, expected: DiagnosticCase, label: str) -> None:
    case = require_dict(report, raw, label)
    report.check(case.get("id") == expected.case_id, f"{label}.id drift")
    report.check(case.get("kind") == expected.kind, f"{expected.case_id}.kind drift")
    report.check(case.get("argv") == list(expected.argv), f"{expected.case_id}.argv drift")
    report.check(case.get("exit_code") == expected.exit_code, f"{expected.case_id}.exit_code drift")
    stdout = check_bytes(report, case.get("stdout"), f"{expected.case_id}.stdout")
    stderr = check_bytes(report, case.get("stderr"), f"{expected.case_id}.stderr")
    report.check(case.get("stderr_anchors") == list(expected.stderr_anchors), f"{expected.case_id}.stderr_anchors drift")
    for anchor in expected.stderr_anchors:
        report.check(anchor.encode("utf-8") in stderr, f"{expected.case_id}.stderr missing anchor {anchor!r}")
    if expected.stdout_kind == "empty":
        report.check(stdout == b"", f"{expected.case_id}.stdout should be empty")
    if expected.stdout_kind == "perplexity":
        check_perplexity(report, case.get("perplexity"), f"{expected.case_id}.perplexity")
    output = case.get("output_file")
    if expected.output_file is None:
        report.check(output is None, f"{expected.case_id}.output_file should be null")
    else:
        out = require_dict(report, output, f"{expected.case_id}.output_file")
        report.check(out.get("path") == expected.output_file, f"{expected.case_id}.output path drift")
        raw_file = check_bytes(report, out.get("raw"), f"{expected.case_id}.output_file.raw")
        parsed = json.loads(raw_file.decode("utf-8"))
        report.check(out.get("json") == parsed, f"{expected.case_id}.output_file.json drift")
        check_logprob_json(report, parsed, expected, f"{expected.case_id}.output_file.json")
        report.check(out.get("summary") == summarize_logprob_json(parsed), f"{expected.case_id}.summary drift")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.cli_diagnostics_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-cli-diagnostics", "source mismatch")
    report.check(root.get("binary") == "./ds4", "binary drift")
    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("path") == B300_MODEL, "model.path drift")
    report.check(model.get("size") == EXPECTED_MODEL_SIZE, "model.size drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 drift")
    policy = require_dict(report, root.get("numeric_policy"), "numeric_policy")
    report.check(policy.get("score_abs_tolerance_for_future_comparators") == SCORE_ABS_TOLERANCE, "numeric tolerance drift")
    cases = require_list(report, root.get("cases"), "cases")
    report.check([case.get("id") for case in cases if isinstance(case, dict)] == [case.case_id for case in CASES], "case order or coverage drift")
    expected = {case.case_id: case for case in CASES}
    for idx, raw_case in enumerate(cases):
        case_id = raw_case.get("id") if isinstance(raw_case, dict) else None
        exp = expected.get(case_id)
        if exp is None:
            report.check(False, f"cases[{idx}].id unexpected {case_id!r}")
            continue
        check_case(report, raw_case, exp, f"cases[{idx}]")
    return report


def build_manifest(artifact: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_diagnostics_manifest.v1",
        "milestone": "M8.6",
        "oracle": "current C CLI --dump-logprobs and --perplexity-file diagnostics",
        "artifact": {
            "path": "current-c.json",
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "b300": {
            "context": B300_CONTEXT,
            "namespace": B300_NAMESPACE,
            "pod": B300_POD,
            "workdir": B300_WORKDIR,
            "kubeconfig": B300_KUBECONFIG,
        },
        "model": {
            "link_path": B300_MODEL,
            "size_bytes": EXPECTED_MODEL_SIZE,
            "sha256": EXPECTED_MODEL_SHA256,
        },
        "capture_commands": [
            f"git archive HEAD | kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} exec -i {B300_POD} -- tar -xf - -C {B300_WORKDIR}",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} exec {B300_POD} -- sh -lc 'set -e; cd {B300_WORKDIR}; make ds4 CUDA_ARCH=native; python3 ds4-parity/check_cli_diagnostics_dump.py --write-baseline ds4-parity/baselines/cli/m8.6/current-c.json --write-manifest ds4-parity/baselines/cli/m8.6/manifest.json --binary ./ds4; python3 ds4-parity/check_cli_diagnostics_dump.py ds4-parity/baselines/cli/m8.6/current-c.json --manifest ds4-parity/baselines/cli/m8.6/manifest.json --negative-test'",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} cp {B300_POD}:{B300_WORKDIR}/ds4-parity/baselines/cli/m8.6/current-c.json ds4-parity/baselines/cli/m8.6/current-c.json",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} cp {B300_POD}:{B300_WORKDIR}/ds4-parity/baselines/cli/m8.6/manifest.json ds4-parity/baselines/cli/m8.6/manifest.json",
        ],
        "normalization": {
            "stderr": "context/progress stderr is retained as captured; future comparators may anchor categories instead of byte-comparing timing-free progress",
            "logprob_scores": "selected token order and top-logprob order are exact; future Rust score values use the M6 ordinary absolute tolerance",
            "perplexity": "stdout scalar fields are parsed from the current C printf format",
        },
    }


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    root = require_dict(report, load_json(path), "manifest")
    report.check(root.get("schema") == "ds4.cli_diagnostics_manifest.v1", "manifest schema mismatch")
    report.check(root.get("milestone") == "M8.6", "manifest milestone mismatch")
    artifact_info = require_dict(report, root.get("artifact"), "manifest.artifact")
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("size_bytes") == artifact.stat().st_size, "manifest artifact size drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    commands = "\n".join(require_list(report, root.get("capture_commands"), "manifest.capture_commands"))
    for required in (
        "git archive HEAD",
        "make ds4 CUDA_ARCH=native",
        "--write-baseline ds4-parity/baselines/cli/m8.6/current-c.json",
        "--negative-test",
    ):
        report.check(required in commands, f"manifest capture command missing {required}")
    return report


def run_negative_tests(obj: Any, manifest_path: Path | None, artifact_path: Path) -> Report:
    report = Report()

    def expect_failure(name: str, path: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        sub = check_dump(candidate)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("case coverage drift", ["cases"], obj["cases"][:-1])
    expect_failure("stdout sha drift", ["cases", 3, "stdout", "sha256"], "0" * 64)
    expect_failure("logprob selected drift", ["cases", 0, "output_file", "json", "steps", 0, "selected", "id"], -1)
    expect_failure("perplexity scored drift", ["cases", 3, "perplexity", "scored"], 99)
    expect_failure("stderr anchor drift", ["cases", 2, "stderr_anchors"], [])
    expect_failure("numeric policy drift", ["numeric_policy", "score_abs_tolerance_for_future_comparators"], 0.1)
    if manifest_path is not None:
        manifest = load_json(manifest_path)
        manifest["artifact"]["sha256"] = "0" * 64
        tmp = Report()
        tmp.check(manifest.get("artifact", {}).get("sha256") == sha256_file(artifact_path), "manifest sha drift")
        report.check(not tmp.ok, "negative test did not fail: manifest sha drift")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("./ds4"))
    parser.add_argument("--model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if args.write_baseline:
        baseline = capture_baseline(args.binary if args.binary.is_absolute() else ROOT / args.binary, args.model_sha256)
        write_json(args.write_baseline, baseline)
        if args.write_manifest:
            write_json(args.write_manifest, build_manifest(args.write_baseline))
        return 0
    if args.write_manifest:
        write_json(args.write_manifest, build_manifest(args.artifact))
        return 0

    obj = load_json(args.artifact)
    dump_report = check_dump(obj)
    print_report("CLI diagnostics oracle", dump_report)
    ok = dump_report.ok
    manifest_path = args.manifest
    if manifest_path is None and args.artifact.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.artifact)
        print_report("CLI diagnostics manifest", manifest_report)
        ok = ok and manifest_report.ok
    if args.negative_test:
        negative_report = run_negative_tests(obj, manifest_path, args.artifact)
        print_report("CLI diagnostics negative tests", negative_report)
        ok = ok and negative_report.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
