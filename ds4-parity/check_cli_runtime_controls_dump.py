#!/usr/bin/env python3
"""Validate the M8.12b current-C CLI runtime-control oracle."""

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
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.12b"
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
STEERING_PATH = "dir-steering/out/verbosity.f32"
STEERING_SHA256 = "6414573b7d88822e16e6fe5972386ef2f1e51fc8502fe5849c4a611afad50cdd"
STEERING_SIZE = 704512
MISSING_MTP_PATH = "/workspace/ds4/missing-mtp.gguf"

TIMING_RE = re.compile(r"ds4: prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s")
STARTUP_RE = re.compile(r"in [0-9.]+s")
WARM_RE = re.compile(r"warmed tensor pages in [0-9.]+s")
FORBIDDEN_STDERR = (
    b"ds4>",
    b"perplexity",
    b"imatrix",
    b"--dump-logprobs",
    b"diagnostic run completed",
)


@dataclass(frozen=True)
class RuntimeCase:
    case_id: str
    argv: tuple[str, ...]
    exit_code: int = 0
    stdout_empty: bool = False
    support_artifact: str | None = None
    availability: str = "executed"
    stderr_anchors: tuple[str, ...] = ()
    normalized_stderr_anchors: tuple[str, ...] = ()


PROMPT = "Answer with one short noun: glacier."

CASES: tuple[RuntimeCase, ...] = (
    RuntimeCase(
        "backend_name_cuda_quality_threads",
        (
            "--backend",
            "cuda",
            "-m",
            B300_MODEL,
            "--ctx",
            "128",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--quality",
            "-t",
            "2",
            "--nothink",
            "-p",
            PROMPT,
        ),
        stderr_anchors=("ds4: context buffers", "backend=cuda", "ds4: using GPU graph generation"),
    ),
    RuntimeCase(
        "warm_weights",
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
            "--warm-weights",
            "--nothink",
            "-p",
            PROMPT,
        ),
        stderr_anchors=("ds4: warming mapped tensor pages: 80.76 GiB", "checksum=", "ds4: using GPU graph generation"),
        normalized_stderr_anchors=("ds4: warmed tensor pages in <seconds>s",),
    ),
    RuntimeCase(
        "directional_steering",
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
            "--dir-steering-file",
            STEERING_PATH,
            "--dir-steering-ffn",
            "0.25",
            "--dir-steering-attn",
            "0",
            "--nothink",
            "-p",
            PROMPT,
        ),
        support_artifact=STEERING_PATH,
        stderr_anchors=(
            "ds4: directional steering enabled: dir-steering/out/verbosity.f32 attn=0 ffn=0.25",
            "ds4: using GPU graph generation",
        ),
    ),
    RuntimeCase(
        "backend_metal_error",
        (
            "--backend",
            "metal",
            "-m",
            B300_MODEL,
            "--ctx",
            "128",
            "--tokens",
            "1",
            "--temp",
            "0",
            "--nothink",
            "-p",
            PROMPT,
        ),
        exit_code=1,
        stdout_empty=True,
        availability="blocked",
        stderr_anchors=("backend=metal", "ds4: Metal backend requested but this build is linked with CUDA, not Metal"),
    ),
    RuntimeCase(
        "mtp_missing_model",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--mtp",
            MISSING_MTP_PATH,
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
            PROMPT,
        ),
        exit_code=1,
        stdout_empty=True,
        support_artifact=MISSING_MTP_PATH,
        availability="blocked_missing_mtp_model",
        stderr_anchors=("backend=cuda", "ds4: cannot open model '/workspace/ds4/missing-mtp.gguf': No such file or directory"),
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
    return {"base64": b64(data), "bytes": len(data), "sha256": sha256_bytes(data)}


def normalize_stderr(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace")
    text = STARTUP_RE.sub("in <seconds>s", text)
    text = WARM_RE.sub("warmed tensor pages in <seconds>s", text)
    text = TIMING_RE.sub("ds4: prefill: <rate> t/s, generation: <rate> t/s", text)
    return text


def resolve_binary(binary: Path) -> Path:
    return binary if binary.is_absolute() else ROOT / binary


def capture_case(binary: Path, case: RuntimeCase) -> dict[str, Any]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    proc = subprocess.run(
        [str(resolve_binary(binary)), *case.argv],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    normalized = normalize_stderr(proc.stderr)
    return {
        "id": case.case_id,
        "argv": list(case.argv),
        "exit_code": proc.returncode,
        "expected_exit_code": case.exit_code,
        "stdout": capture_bytes(proc.stdout),
        "stderr": capture_bytes(proc.stderr),
        "stderr_normalized": normalized,
        "stderr_normalized_sha256": sha256_bytes(normalized.encode("utf-8")),
        "stdout_empty": case.stdout_empty,
        "support_artifact": case.support_artifact,
        "availability": case.availability,
        "stderr_anchors": list(case.stderr_anchors),
        "normalized_stderr_anchors": list(case.normalized_stderr_anchors),
    }


def support_artifacts() -> dict[str, Any]:
    steering = ROOT / STEERING_PATH
    return {
        "directional_steering": {
            "path": STEERING_PATH,
            "size_bytes": steering.stat().st_size if steering.exists() else None,
            "sha256": sha256_file(steering) if steering.exists() else None,
        },
        "mtp": {
            "path": MISSING_MTP_PATH,
            "available": Path(MISSING_MTP_PATH).exists(),
            "blocker": "no MTP GGUF is present in the B300 workspace",
        },
    }


def capture_baseline(binary: Path, model_sha256: str) -> dict[str, Any]:
    binary_path = resolve_binary(binary)
    if not binary_path.is_file():
        raise SystemExit(f"missing CLI binary: {binary_path}; build ds4 first")
    model = Path(B300_MODEL)
    if not model.is_file():
        raise SystemExit(f"missing model: {B300_MODEL}")
    return {
        "schema": "ds4.cli_runtime_controls_oracle.v1",
        "source": "current-c-cli-one-shot-runtime-controls",
        "binary": "./ds4",
        "model": {"path": B300_MODEL, "size_bytes": model.stat().st_size, "sha256": model_sha256},
        "b300": {
            "context": B300_CONTEXT,
            "namespace": B300_NAMESPACE,
            "pod": B300_POD,
            "workdir": B300_WORKDIR,
            "kubeconfig": B300_KUBECONFIG,
        },
        "support_artifacts": support_artifacts(),
        "cases": [capture_case(binary, case) for case in CASES],
        "normalization": {
            "stderr": "startup seconds, warm-weight seconds, and throughput rates are normalized; runtime-control categories are exact",
            "stdout": "generated bytes and hashes are exact for executed transcript cases",
        },
    }


def case_by_id() -> dict[str, RuntimeCase]:
    return {case.case_id: case for case in CASES}


def check_support_artifacts(report: Report, data: dict[str, Any]) -> None:
    support = require_dict(report, data.get("support_artifacts"), "support_artifacts")
    steering = require_dict(report, support.get("directional_steering"), "support_artifacts.directional_steering")
    report.check(steering.get("path") == STEERING_PATH, "steering path drift")
    report.check(steering.get("size_bytes") == STEERING_SIZE, "steering size drift")
    report.check(steering.get("sha256") == STEERING_SHA256, "steering sha drift")
    mtp = require_dict(report, support.get("mtp"), "support_artifacts.mtp")
    report.check(mtp.get("path") == MISSING_MTP_PATH, "mtp path drift")
    report.check(mtp.get("available") is False, "mtp availability drift")
    report.check("no MTP GGUF" in str(mtp.get("blocker")), "mtp blocker drift")


def check_dump(obj: Any) -> Report:
    report = Report()
    data = require_dict(report, obj, "root")
    report.check(data.get("schema") == "ds4.cli_runtime_controls_oracle.v1", "schema drift")
    model = require_dict(report, data.get("model"), "model")
    report.check(model.get("path") == B300_MODEL, "model.path drift")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "model.size_bytes drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 drift")
    check_support_artifacts(report, data)

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
        report.check(case_obj.get("argv") == list(expected_case.argv), f"{case_id}: argv drift")
        report.check(case_obj.get("exit_code") == expected_case.exit_code, f"{case_id}: exit code drift")
        report.check(case_obj.get("expected_exit_code") == expected_case.exit_code, f"{case_id}: expected exit code drift")
        report.check(case_obj.get("stdout_empty") == expected_case.stdout_empty, f"{case_id}: stdout empty policy drift")
        report.check(case_obj.get("support_artifact") == expected_case.support_artifact, f"{case_id}: support artifact drift")
        report.check(case_obj.get("availability") == expected_case.availability, f"{case_id}: availability drift")

        stdout = require_dict(report, case_obj.get("stdout"), f"{case_id}.stdout")
        stderr = require_dict(report, case_obj.get("stderr"), f"{case_id}.stderr")
        stdout_bytes = unb64(report, stdout.get("base64"), f"{case_id}.stdout.base64")
        stderr_bytes = unb64(report, stderr.get("base64"), f"{case_id}.stderr.base64")
        stdout_by_id[case_id] = stdout_bytes
        report.check(stdout.get("bytes") == len(stdout_bytes), f"{case_id}: stdout byte count drift")
        report.check(stdout.get("sha256") == sha256_bytes(stdout_bytes), f"{case_id}: stdout sha drift")
        report.check(stderr.get("bytes") == len(stderr_bytes), f"{case_id}: stderr byte count drift")
        report.check(stderr.get("sha256") == sha256_bytes(stderr_bytes), f"{case_id}: stderr sha drift")
        report.check((len(stdout_bytes) == 0) == expected_case.stdout_empty, f"{case_id}: stdout empty drift")

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
    for case_id in ("backend_metal_error", "mtp_missing_model"):
        if case_id in stdout_by_id:
            report.check(stdout_by_id[case_id] == b"", f"{case_id}: blocked case emitted stdout")
    return report


def check_manifest(manifest: Any, baseline_path: Path) -> Report:
    report = Report()
    obj = require_dict(report, manifest, "manifest")
    report.check(obj.get("schema") == "ds4.cli_runtime_controls_manifest.v1", "manifest schema drift")
    report.check(obj.get("milestone") == "M8.12b", "manifest milestone drift")
    artifact = require_dict(report, obj.get("artifact"), "manifest.artifact")
    report.check(artifact.get("path") == "current-c.json", "manifest artifact path drift")
    if baseline_path.exists():
        report.check(artifact.get("size_bytes") == baseline_path.stat().st_size, "manifest artifact size drift")
        report.check(artifact.get("sha256") == sha256_file(baseline_path), "manifest artifact sha drift")
    model = require_dict(report, obj.get("model"), "manifest.model")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "manifest model size drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "manifest model sha drift")
    commands = require_list(report, obj.get("capture_commands"), "manifest.capture_commands")
    joined = "\n".join(str(command) for command in commands)
    for required in (
        "git archive HEAD",
        "--context hou2-prod1",
        "make ds4 CUDA_ARCH=native",
        "check_cli_runtime_controls_dump.py",
        "--negative-test",
    ):
        report.check(required in joined, f"manifest capture command missing {required!r}")
    return report


def make_manifest(baseline_path: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_runtime_controls_manifest.v1",
        "milestone": "M8.12b",
        "oracle": "current C CLI one-shot runtime-control transcripts",
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
        "model": {"link_path": B300_MODEL, "size_bytes": EXPECTED_MODEL_SIZE, "sha256": EXPECTED_MODEL_SHA256},
        "capture_commands": [
            "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- tar -xf - -C /workspace/ds4",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; make ds4 CUDA_ARCH=native; python3 ds4-parity/check_cli_runtime_controls_dump.py --write-baseline ds4-parity/baselines/cli/m8.12b/current-c.json --write-manifest ds4-parity/baselines/cli/m8.12b/manifest.json --binary ./ds4; python3 ds4-parity/check_cli_runtime_controls_dump.py ds4-parity/baselines/cli/m8.12b/current-c.json --manifest ds4-parity/baselines/cli/m8.12b/manifest.json --negative-test'",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/cli/m8.12b/current-c.json ds4-parity/baselines/cli/m8.12b/current-c.json",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/cli/m8.12b/manifest.json ds4-parity/baselines/cli/m8.12b/manifest.json",
        ],
        "normalization": {
            "stderr": "startup seconds, warm-weight seconds, and throughput rates are normalized; runtime-control categories are exact",
            "stdout": "generated bytes and hashes are exact for executed transcript cases",
        },
    }


def run_negative_test(obj: Any) -> Report:
    report = Report()

    def expect_failure(label: str, mutator: Any) -> None:
        candidate = copy.deepcopy(obj)
        mutator(candidate)
        result = check_dump(candidate)
        report.check(not result.ok, f"negative test failed to detect {label}")

    expect_failure("steering artifact hash drift", lambda o: o["support_artifacts"]["directional_steering"].__setitem__("sha256", "0" * 64))
    expect_failure("warm-weight stderr drift", lambda o: o["cases"][1].__setitem__("stderr", o["cases"][0]["stderr"]))
    expect_failure("directional steering stderr drift", lambda o: o["cases"][2].__setitem__("stderr", o["cases"][0]["stderr"]))
    expect_failure("metal backend exit drift", lambda o: o["cases"][3].__setitem__("exit_code", 0))
    expect_failure("mtp availability drift", lambda o: o["support_artifacts"]["mtp"].__setitem__("available", True))
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
    rc = print_report("CLI runtime-controls oracle", check_dump(obj))
    if args.manifest and args.manifest.exists():
        rc |= print_report("CLI runtime-controls manifest", check_manifest(load_json(args.manifest), args.baseline))
    if args.negative_test:
        rc |= print_report("CLI runtime-controls negative tests", run_negative_test(obj))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
