#!/usr/bin/env python3
"""Validate the M8.12a current-C CLI one-shot generation oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.12a"
FIXTURE_DIR = ROOT / "ds4-parity" / "baselines" / "cli-fixtures" / "m8.12a"
BASELINE = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"

EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_SIZE = 86720111488
B300_KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
B300_CONTEXT = "hou2-prod1"
B300_NAMESPACE = "default"
B300_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
B300_MODEL = "/workspace/ds4/ds4flash.gguf"

TIMING_RE = re.compile(r"ds4: prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s")
STARTUP_RE = re.compile(r"in [0-9.]+s")
FORBIDDEN_STDERR = (
    b"ds4>",
    b"perplexity",
    b"imatrix",
    b"--dump-logprobs",
    b"diagnostic run completed",
)


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    argv_prefix: tuple[str, ...]
    exit_code: int = 0
    prompt_text: str | None = None
    fixture_name: str | None = None
    seed: int | None = None
    mode: str = "greedy"
    stdout_empty: bool = False
    stderr_anchors: tuple[str, ...] = ()
    normalized_stderr_anchors: tuple[str, ...] = ()

    def prompt_bytes(self) -> bytes:
        if self.prompt_text is not None:
            return self.prompt_text.encode("utf-8")
        if self.fixture_name is None:
            raise ValueError(f"{self.case_id}: missing prompt")
        return fixture_path(self.fixture_name).read_bytes()

    def prompt_args(self) -> tuple[str, ...]:
        if self.prompt_text is not None:
            return ("-p", self.prompt_text)
        if self.fixture_name is None:
            raise ValueError(f"{self.case_id}: missing prompt")
        return ("--prompt-file", fixture_relpath(self.fixture_name))


CASES: tuple[GenerationCase, ...] = (
    GenerationCase(
        "greedy_inline_nothink",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "128",
            "--tokens",
            "2",
            "--temp",
            "0",
            "--nothink",
        ),
        prompt_text="Answer with one short noun: glacier.",
        mode="greedy_inline_nothink",
        stderr_anchors=("ds4: context buffers", "backend=cuda", "ds4: using GPU graph generation"),
    ),
    GenerationCase(
        "prompt_file_think",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "128",
            "--tokens",
            "2",
            "--temp",
            "0",
            "--think",
        ),
        fixture_name="prompt_file.txt",
        mode="greedy_prompt_file_think",
        stderr_anchors=("ds4: context buffers", "backend=cuda", "ds4: using GPU graph generation"),
    ),
    GenerationCase(
        "think_max_downgrade",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "128",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--think-max",
        ),
        prompt_text="Say one number.",
        mode="think_max_downgrade",
        stderr_anchors=(
            "ds4: context buffers",
            "backend=cuda",
            "ds4: warning: --think-max needs --ctx >= 393216; ctx=128 uses normal thinking instead",
        ),
    ),
    GenerationCase(
        "seeded_sampling_nothink",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "128",
            "--tokens",
            "2",
            "--temp",
            "0.7",
            "--top-p",
            "0.9",
            "--min-p",
            "0.05",
            "--seed",
            "12345",
            "--nothink",
        ),
        prompt_text="Answer with one adjective: ice.",
        seed=12345,
        mode="seeded_sampling_nothink",
        stderr_anchors=("ds4: context buffers", "backend=cuda"),
        normalized_stderr_anchors=("ds4: prefill: <rate> t/s, generation: <rate> t/s",),
    ),
    GenerationCase(
        "ctx_too_small",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "16",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
        ),
        exit_code=1,
        prompt_text=(
            "This prompt is intentionally long enough to exceed a tiny "
            "sixteen-token context window after chat wrapping."
        ),
        mode="context_too_small",
        stdout_empty=True,
        stderr_anchors=(
            "ds4: context buffers",
            "backend=cuda",
            "ds4: using GPU graph generation",
            "ds4: prompt is empty or exceeds context size",
        ),
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


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / name


def fixture_relpath(name: str) -> str:
    return f"ds4-parity/baselines/cli-fixtures/m8.12a/{name}"


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


def normalize_stderr(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace")
    text = STARTUP_RE.sub("in <seconds>s", text)
    text = TIMING_RE.sub("ds4: prefill: <rate> t/s, generation: <rate> t/s", text)
    return text


def resolve_binary(binary: Path) -> Path:
    return binary if binary.is_absolute() else ROOT / binary


def capture_case(binary: Path, case: GenerationCase) -> dict[str, Any]:
    prompt = case.prompt_bytes()
    argv = (*case.argv_prefix, *case.prompt_args())
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    proc = subprocess.run(
        [str(resolve_binary(binary)), *argv],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    stderr_normalized = normalize_stderr(proc.stderr)
    return {
        "id": case.case_id,
        "mode": case.mode,
        "argv": list(argv),
        "prompt_file": fixture_relpath(case.fixture_name) if case.fixture_name else None,
        "prompt_base64": b64(prompt),
        "prompt_bytes": len(prompt),
        "prompt_sha256": sha256_bytes(prompt),
        "seed": case.seed,
        "exit_code": proc.returncode,
        "expected_exit_code": case.exit_code,
        "stdout": capture_bytes(proc.stdout),
        "stderr": capture_bytes(proc.stderr),
        "stderr_normalized": stderr_normalized,
        "stderr_normalized_sha256": sha256_bytes(stderr_normalized.encode("utf-8")),
        "stderr_anchors": list(case.stderr_anchors),
        "normalized_stderr_anchors": list(case.normalized_stderr_anchors),
        "stdout_empty": case.stdout_empty,
    }


def capture_baseline(binary: Path, model_sha256: str) -> dict[str, Any]:
    binary_path = resolve_binary(binary)
    if not binary_path.is_file():
        raise SystemExit(f"missing CLI binary: {binary_path}; build ds4 first")
    model = Path(B300_MODEL)
    if not model.is_file():
        raise SystemExit(f"missing model: {B300_MODEL}")
    return {
        "schema": "ds4.cli_generation_oracle.v1",
        "source": "current-c-cli-one-shot-generation",
        "binary": "./ds4",
        "model": {
            "path": B300_MODEL,
            "size_bytes": model.stat().st_size,
            "sha256": model_sha256,
        },
        "b300": {
            "context": B300_CONTEXT,
            "namespace": B300_NAMESPACE,
            "pod": B300_POD,
            "workdir": B300_WORKDIR,
            "kubeconfig": B300_KUBECONFIG,
        },
        "cases": [capture_case(binary, case) for case in CASES],
        "normalization": {
            "stderr": "startup seconds and throughput rates are normalized; backend identity, warnings, generation path, and error categories are exact",
            "stdout": "generated bytes and hashes are exact for committed current-C fixture",
        },
    }


def case_by_id() -> dict[str, GenerationCase]:
    return {case.case_id: case for case in CASES}


def check_dump(obj: Any) -> Report:
    report = Report()
    data = require_dict(report, obj, "root")
    report.check(data.get("schema") == "ds4.cli_generation_oracle.v1", "schema drift")
    model = require_dict(report, data.get("model"), "model")
    report.check(model.get("path") == B300_MODEL, "model.path drift")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "model.size_bytes drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 drift")

    cases = require_list(report, data.get("cases"), "cases")
    expected = case_by_id()
    report.check(len(cases) == len(expected), "case count drift")
    seen: set[str] = set()
    stdout_by_id: dict[str, bytes] = {}
    for index, item in enumerate(cases):
        case_obj = require_dict(report, item, f"cases[{index}]")
        case_id = case_obj.get("id")
        report.check(isinstance(case_id, str), f"cases[{index}].id missing")
        if not isinstance(case_id, str):
            continue
        seen.add(case_id)
        expected_case = expected.get(case_id)
        report.check(expected_case is not None, f"{case_id}: unexpected case")
        if expected_case is None:
            continue

        report.check(case_obj.get("mode") == expected_case.mode, f"{case_id}: mode drift")
        report.check(case_obj.get("argv") == list((*expected_case.argv_prefix, *expected_case.prompt_args())), f"{case_id}: argv drift")
        report.check(case_obj.get("prompt_sha256") == sha256_bytes(expected_case.prompt_bytes()), f"{case_id}: prompt hash drift")
        report.check(case_obj.get("prompt_bytes") == len(expected_case.prompt_bytes()), f"{case_id}: prompt size drift")
        report.check(case_obj.get("seed") == expected_case.seed, f"{case_id}: seed drift")
        report.check(case_obj.get("exit_code") == expected_case.exit_code, f"{case_id}: exit code drift")
        report.check(case_obj.get("expected_exit_code") == expected_case.exit_code, f"{case_id}: expected exit code drift")

        stdout = require_dict(report, case_obj.get("stdout"), f"{case_id}.stdout")
        stderr = require_dict(report, case_obj.get("stderr"), f"{case_id}.stderr")
        stdout_bytes = unb64(report, stdout.get("base64"), f"{case_id}.stdout.base64")
        stderr_bytes = unb64(report, stderr.get("base64"), f"{case_id}.stderr.base64")
        stdout_by_id[case_id] = stdout_bytes
        report.check(stdout.get("bytes") == len(stdout_bytes), f"{case_id}: stdout byte count drift")
        report.check(stdout.get("sha256") == sha256_bytes(stdout_bytes), f"{case_id}: stdout sha drift")
        report.check(stderr.get("bytes") == len(stderr_bytes), f"{case_id}: stderr byte count drift")
        report.check(stderr.get("sha256") == sha256_bytes(stderr_bytes), f"{case_id}: stderr sha drift")
        report.check((len(stdout_bytes) == 0) == expected_case.stdout_empty, f"{case_id}: stdout empty policy drift")

        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        normalized = normalize_stderr(stderr_bytes)
        report.check(case_obj.get("stderr_normalized") == normalized, f"{case_id}: normalized stderr drift")
        report.check(
            case_obj.get("stderr_normalized_sha256") == sha256_bytes(normalized.encode("utf-8")),
            f"{case_id}: normalized stderr sha drift",
        )
        for anchor in expected_case.stderr_anchors:
            report.check(anchor in stderr_text, f"{case_id}: missing stderr anchor {anchor!r}")
        for anchor in expected_case.normalized_stderr_anchors:
            report.check(anchor in normalized, f"{case_id}: missing normalized stderr anchor {anchor!r}")
        for forbidden in FORBIDDEN_STDERR:
            report.check(forbidden not in stderr_bytes, f"{case_id}: forbidden stderr marker {forbidden!r}")

    report.check(set(expected) == seen, f"case id drift expected={sorted(expected)} got={sorted(seen)}")
    if "greedy_inline_nothink" in stdout_by_id and "seeded_sampling_nothink" in stdout_by_id:
        report.check(
            stdout_by_id["greedy_inline_nothink"] != stdout_by_id["seeded_sampling_nothink"],
            "seeded sampling should not collapse to the greedy inline stdout fixture",
        )
    return report


def check_manifest(manifest: Any, baseline_path: Path) -> Report:
    report = Report()
    obj = require_dict(report, manifest, "manifest")
    report.check(obj.get("schema") == "ds4.cli_generation_manifest.v1", "manifest schema drift")
    report.check(obj.get("milestone") == "M8.12a", "manifest milestone drift")
    artifact = require_dict(report, obj.get("artifact"), "manifest.artifact")
    report.check(artifact.get("path") == "current-c.json", "manifest artifact path drift")
    if baseline_path.exists():
        report.check(artifact.get("size_bytes") == baseline_path.stat().st_size, "manifest artifact size drift")
        report.check(artifact.get("sha256") == sha256_file(baseline_path), "manifest artifact sha drift")
    model = require_dict(report, obj.get("model"), "manifest.model")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "manifest model size drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "manifest model sha drift")
    commands = require_list(report, obj.get("capture_commands"), "manifest.capture_commands")
    report.check(len(commands) >= 4, "manifest capture commands missing")
    joined = "\n".join(str(command) for command in commands)
    for required in (
        "git archive HEAD",
        "--context hou2-prod1",
        "make ds4 CUDA_ARCH=native",
        "check_cli_generation_dump.py",
        "--negative-test",
    ):
        report.check(required in joined, f"manifest capture command missing {required!r}")
    return report


def make_manifest(baseline_path: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_generation_manifest.v1",
        "milestone": "M8.12a",
        "oracle": "current C CLI one-shot generation transcripts",
        "artifact": {
            "path": "current-c.json",
            "size_bytes": baseline_path.stat().st_size,
            "sha256": sha256_file(baseline_path),
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
            "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- tar -xf - -C /workspace/ds4",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; make ds4 CUDA_ARCH=native; python3 ds4-parity/check_cli_generation_dump.py --write-baseline ds4-parity/baselines/cli/m8.12a/current-c.json --write-manifest ds4-parity/baselines/cli/m8.12a/manifest.json --binary ./ds4; python3 ds4-parity/check_cli_generation_dump.py ds4-parity/baselines/cli/m8.12a/current-c.json --manifest ds4-parity/baselines/cli/m8.12a/manifest.json --negative-test'",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/cli/m8.12a/current-c.json ds4-parity/baselines/cli/m8.12a/current-c.json",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/cli/m8.12a/manifest.json ds4-parity/baselines/cli/m8.12a/manifest.json",
        ],
        "normalization": {
            "stderr": "startup seconds and throughput rates are normalized; backend identity, warnings, generation path, and error categories are exact",
            "stdout": "generated bytes and hashes are exact for committed current-C fixture",
        },
    }


def run_negative_test(obj: Any) -> Report:
    report = Report()

    def expect_failure(label: str, mutator: Any) -> None:
        candidate = copy.deepcopy(obj)
        mutator(candidate)
        result = check_dump(candidate)
        report.check(not result.ok, f"negative test failed to detect {label}")

    expect_failure("stdout hash drift", lambda o: o["cases"][0]["stdout"].__setitem__("sha256", "0" * 64))
    expect_failure("stderr category drift", lambda o: o["cases"][2].__setitem__("stderr", o["cases"][0]["stderr"]))
    expect_failure("prompt hash drift", lambda o: o["cases"][1].__setitem__("prompt_sha256", "0" * 64))
    expect_failure("exit code drift", lambda o: o["cases"][4].__setitem__("exit_code", 0))
    expect_failure("seeded sample collapse", lambda o: o["cases"][3]["stdout"].__setitem__("base64", o["cases"][0]["stdout"]["base64"]))
    return report


def print_report(name: str, report: Report) -> int:
    if report.ok:
        print(f"{name}: PASS, {report.checks} checks")
        return 0
    print(f"{name}: FAIL, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--binary", type=Path, default=Path("./ds4"))
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if args.write_baseline:
        obj = capture_baseline(args.binary, args.model_sha256)
        write_json(args.write_baseline, obj)
        if args.write_manifest:
            write_json(args.write_manifest, make_manifest(args.write_baseline))
        return 0

    obj = load_json(args.baseline)
    rc = print_report("CLI generation oracle", check_dump(obj))
    if args.manifest and args.manifest.exists():
        rc |= print_report("CLI generation manifest", check_manifest(load_json(args.manifest), args.baseline))
    if args.negative_test:
        rc |= print_report("CLI generation negative tests", run_negative_test(obj))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
