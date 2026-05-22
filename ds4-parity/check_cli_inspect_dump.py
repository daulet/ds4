#!/usr/bin/env python3
"""Validate the M8.8 current-C CLI --inspect oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.8"
BASELINE = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"

EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_SIZE = 86720111488
EXPECTED_SUMMARY = {
    "model": "DeepSeek V4 Flash",
    "arch": "deepseek4",
    "gguf_version": 3,
    "metadata_keys": 62,
    "tensors": 1328,
    "layers": 43,
    "attention": {"heads": 64, "kv_heads": 1, "head_dim": 512, "swa": 128},
    "indexer": {"heads": 64, "head_dim": 128, "top_k": 512},
    "experts": {"count": 256, "used": 6, "groups": 0, "groups_used": 0},
    "file_size": "80.76 GiB",
    "tensor_bytes": "80.76 GiB",
    "logical_parameters": "284.33 B",
    "tensor_types": [
        {"type": "f32", "count": 492, "size": "0.00 GiB"},
        {"type": "f16", "count": 359, "size": "2.04 GiB"},
        {"type": "q8_0", "count": 345, "size": "6.15 GiB"},
        {"type": "q2_k", "count": 43, "size": "28.22 GiB"},
        {"type": "iq2_xxs", "count": 86, "size": "44.34 GiB"},
        {"type": "i32", "count": 3, "size": "0.01 GiB"},
    ],
}
B300_KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
B300_CONTEXT = "hou2-prod1"
B300_NAMESPACE = "default"
B300_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
B300_MODEL = "/workspace/ds4/ds4flash.gguf"

GGUF_RE = re.compile(r"^gguf:  v(?P<version>\d+), (?P<metadata>\d+) metadata keys, (?P<tensors>\d+) tensors$")
ATTENTION_RE = re.compile(r"^attention: heads=(?P<heads>\d+) kv_heads=(?P<kv>\d+) head_dim=(?P<dim>\d+) swa=(?P<swa>\d+)$")
INDEXER_RE = re.compile(r"^indexer: heads=(?P<heads>\d+) head_dim=(?P<dim>\d+) top_k=(?P<top>\d+)$")
EXPERTS_RE = re.compile(r"^experts: count=(?P<count>\d+) used=(?P<used>\d+) groups=(?P<groups>\d+) groups_used=(?P<groups_used>\d+)$")
TENSOR_TYPE_RE = re.compile(r"^\s+(?P<type>\S+)\s+(?P<count>\d+) tensors, (?P<size>[0-9.]+ GiB)$")


@dataclass(frozen=True)
class InspectCase:
    case_id: str
    argv: tuple[str, ...]
    exit_code: int = 0
    stdout_anchors: tuple[str, ...] = (
        "model: DeepSeek V4 Flash",
        "arch:  deepseek4",
        "tensor types:",
    )
    stderr_anchors: tuple[str, ...] = (
        "ds4: CUDA backend initialized on NVIDIA B300 SXM6 AC (sm_103)",
        "ds4: CUDA loading model tensors into device cache",
        "ds4: CUDA startup model cache prepared 80.76 GiB",
        "ds4: cuda backend initialized for graph diagnostics",
    )
    same_stdout_as: str | None = None


CASES: tuple[InspectCase, ...] = (
    InspectCase(
        "cuda_inspect",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--inspect",
        ),
    ),
    InspectCase(
        "cuda_inspect_prompt_controls_ignored",
        (
            "--cuda",
            "-m",
            B300_MODEL,
            "--inspect",
            "--ctx",
            "64",
            "--tokens",
            "1",
            "--think-max",
            "-p",
            "M8.8 inspect must not generate from this prompt.",
        ),
        same_stdout_as="cuda_inspect",
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


def parse_summary(stdout: bytes) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError:
        return {"parse_ok": False}
    lines = text.splitlines()
    try:
        summary = {
            "parse_ok": True,
            "model": lines[0].removeprefix("model: "),
            "arch": lines[1].removeprefix("arch:  "),
            "layers": int(lines[3].removeprefix("layers: ")),
            "file_size": lines[7].removeprefix("file size: "),
            "tensor_bytes": lines[8].removeprefix("tensor bytes described by GGUF: "),
            "logical_parameters": lines[9].removeprefix("logical parameters: "),
        }
        gguf = GGUF_RE.fullmatch(lines[2])
        attention = ATTENTION_RE.fullmatch(lines[4])
        indexer = INDEXER_RE.fullmatch(lines[5])
        experts = EXPERTS_RE.fullmatch(lines[6])
        if not gguf or not attention or not indexer or not experts or lines[10] != "tensor types:":
            return {"parse_ok": False}
        summary.update(
            {
                "gguf_version": int(gguf.group("version")),
                "metadata_keys": int(gguf.group("metadata")),
                "tensors": int(gguf.group("tensors")),
                "attention": {
                    "heads": int(attention.group("heads")),
                    "kv_heads": int(attention.group("kv")),
                    "head_dim": int(attention.group("dim")),
                    "swa": int(attention.group("swa")),
                },
                "indexer": {
                    "heads": int(indexer.group("heads")),
                    "head_dim": int(indexer.group("dim")),
                    "top_k": int(indexer.group("top")),
                },
                "experts": {
                    "count": int(experts.group("count")),
                    "used": int(experts.group("used")),
                    "groups": int(experts.group("groups")),
                    "groups_used": int(experts.group("groups_used")),
                },
                "tensor_types": [],
            }
        )
        for idx, line in enumerate(lines[11:]):
            match = TENSOR_TYPE_RE.fullmatch(line)
            if not match:
                return {"parse_ok": False, "bad_tensor_type_line": idx + 11}
            summary["tensor_types"].append(
                {
                    "type": match.group("type"),
                    "count": int(match.group("count")),
                    "size": match.group("size"),
                }
            )
        return summary
    except (IndexError, ValueError):
        return {"parse_ok": False}


def resolve_binary(binary: Path) -> Path:
    return binary if binary.is_absolute() else ROOT / binary


def run_case(binary: Path, case: InspectCase) -> dict[str, Any]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    proc = subprocess.run(
        [str(binary), *case.argv],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return {
        "id": case.case_id,
        "argv": list(case.argv),
        "exit_code": proc.returncode,
        "stdout": capture_bytes(proc.stdout),
        "stderr": capture_bytes(proc.stderr),
        "stdout_anchors": list(case.stdout_anchors),
        "stderr_anchors": list(case.stderr_anchors),
        "summary": parse_summary(proc.stdout),
        "same_stdout_as": case.same_stdout_as,
        "dispatch_evidence": {
            "inspect_skips_context_log": "ds4_cli.c main calls log_context_memory and cli_warn_think_max_downgraded only when !cfg.inspect",
            "inspect_branch": "ds4_cli.c main calls ds4_engine_summary(engine) when cfg.inspect before generation/repl/perplexity/imatrix branches",
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
        "schema": "ds4.cli_inspect_oracle.v1",
        "source": "current-c-cli-inspect",
        "binary": "./ds4",
        "model": {
            "path": B300_MODEL,
            "size": model.stat().st_size,
            "sha256": model_sha256,
        },
        "source_evidence": {
            "cli_dispatch": "ds4_cli.c main skips prompt/context diagnostics when cfg.inspect, opens ds4_engine, then calls ds4_engine_summary(engine)",
            "summary_entry": "ds4.c ds4_engine_summary calls model_summary(&e->model)",
            "summary_output": "ds4.c model_summary prints model metadata, tensor bytes, logical parameters, and tensor type counts",
        },
        "cases": [run_case(binary_path, case) for case in CASES],
    }


def check_bytes(report: Report, raw: Any, label: str) -> bytes:
    obj = require_dict(report, raw, label)
    data = unb64(report, obj.get("base64"), f"{label}.base64")
    report.check(obj.get("bytes") == len(data), f"{label}.bytes drift")
    report.check(obj.get("sha256") == sha256_bytes(data), f"{label}.sha256 drift")
    return data


def check_summary(report: Report, raw: Any, label: str) -> None:
    summary = require_dict(report, raw, label)
    report.check(summary.get("parse_ok") is True, f"{label}.parse_ok drift")
    for key, expected in EXPECTED_SUMMARY.items():
        report.check(summary.get(key) == expected, f"{label}.{key} drift")


def check_case(report: Report, raw: Any, expected: InspectCase, refs: dict[str, dict[str, Any]], label: str) -> None:
    case = require_dict(report, raw, label)
    report.check(case.get("id") == expected.case_id, f"{label}.id drift")
    report.check(case.get("argv") == list(expected.argv), f"{expected.case_id}.argv drift")
    report.check(case.get("exit_code") == expected.exit_code, f"{expected.case_id}.exit_code drift")
    stdout = check_bytes(report, case.get("stdout"), f"{expected.case_id}.stdout")
    stderr = check_bytes(report, case.get("stderr"), f"{expected.case_id}.stderr")
    report.check(stdout.endswith(b"\n"), f"{expected.case_id}.stdout should end with newline")
    report.check(case.get("stdout_anchors") == list(expected.stdout_anchors), f"{expected.case_id}.stdout_anchors drift")
    for anchor in expected.stdout_anchors:
        report.check(anchor.encode("utf-8") in stdout, f"{expected.case_id}.stdout missing anchor {anchor!r}")
    report.check(case.get("stderr_anchors") == list(expected.stderr_anchors), f"{expected.case_id}.stderr_anchors drift")
    for anchor in expected.stderr_anchors:
        report.check(anchor.encode("utf-8") in stderr, f"{expected.case_id}.stderr missing anchor {anchor!r}")
    for forbidden in (
        b"ds4: context buffers",
        b"think-max",
        b"input tokens:",
        b"decode failed",
        b"perplexity",
        b"imatrix",
    ):
        report.check(forbidden not in stderr, f"{expected.case_id}.stderr entered unexpected path {forbidden!r}")
    check_summary(report, case.get("summary"), f"{expected.case_id}.summary")
    dispatch = require_dict(report, case.get("dispatch_evidence"), f"{expected.case_id}.dispatch_evidence")
    for key in ("inspect_skips_context_log", "inspect_branch"):
        report.check(isinstance(dispatch.get(key), str) and dispatch[key], f"{expected.case_id}.dispatch_evidence.{key} missing")
    report.check(case.get("same_stdout_as") == expected.same_stdout_as, f"{expected.case_id}.same_stdout_as drift")
    if expected.same_stdout_as is not None:
        ref = refs.get(expected.same_stdout_as)
        report.check(ref is not None, f"{expected.case_id}.stdout reference missing")
        if ref is not None:
            ref_stdout = require_dict(report, ref.get("stdout"), f"{expected.case_id}.ref.stdout")
            stdout_obj = require_dict(report, case.get("stdout"), f"{expected.case_id}.stdout.obj")
            report.check(stdout_obj.get("sha256") == ref_stdout.get("sha256"), f"{expected.case_id}.stdout differs from reference")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.cli_inspect_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-cli-inspect", "source mismatch")
    report.check(root.get("binary") == "./ds4", "binary drift")
    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("path") == B300_MODEL, "model.path drift")
    report.check(model.get("size") == EXPECTED_MODEL_SIZE, "model.size drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 drift")
    evidence = require_dict(report, root.get("source_evidence"), "source_evidence")
    for key in ("cli_dispatch", "summary_entry", "summary_output"):
        report.check(isinstance(evidence.get(key), str) and evidence[key], f"source_evidence.{key} missing")
    cases = require_list(report, root.get("cases"), "cases")
    report.check([case.get("id") for case in cases if isinstance(case, dict)] == [case.case_id for case in CASES], "case order or coverage drift")
    refs = {case["id"]: case for case in cases if isinstance(case, dict) and isinstance(case.get("id"), str)}
    expected = {case.case_id: case for case in CASES}
    for idx, raw_case in enumerate(cases):
        case_id = raw_case.get("id") if isinstance(raw_case, dict) else None
        exp = expected.get(case_id)
        if exp is None:
            report.check(False, f"cases[{idx}].id unexpected {case_id!r}")
            continue
        check_case(report, raw_case, exp, refs, f"cases[{idx}]")
    return report


def build_manifest(artifact: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_inspect_manifest.v1",
        "milestone": "M8.8",
        "oracle": "current C CLI --inspect model summary output",
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
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} exec {B300_POD} -- sh -lc 'set -e; cd {B300_WORKDIR}; make ds4 CUDA_ARCH=native; python3 ds4-parity/check_cli_inspect_dump.py --write-baseline ds4-parity/baselines/cli/m8.8/current-c.json --write-manifest ds4-parity/baselines/cli/m8.8/manifest.json --binary ./ds4; python3 ds4-parity/check_cli_inspect_dump.py ds4-parity/baselines/cli/m8.8/current-c.json --manifest ds4-parity/baselines/cli/m8.8/manifest.json --negative-test'",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} cp {B300_POD}:{B300_WORKDIR}/ds4-parity/baselines/cli/m8.8/current-c.json ds4-parity/baselines/cli/m8.8/current-c.json",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} cp {B300_POD}:{B300_WORKDIR}/ds4-parity/baselines/cli/m8.8/manifest.json ds4-parity/baselines/cli/m8.8/manifest.json",
        ],
        "normalization": {
            "stdout": "summary section anchors, parsed model metadata, tensor type counts, and stdout hash are retained",
            "stderr": "CUDA backend identity and cache-progress anchors are retained; startup timing may vary in future comparators",
            "dispatch": "--inspect must not emit context-buffer, think-max, generation, perplexity, or imatrix progress logs",
        },
    }


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    root = require_dict(report, load_json(path), "manifest")
    report.check(root.get("schema") == "ds4.cli_inspect_manifest.v1", "manifest schema mismatch")
    report.check(root.get("milestone") == "M8.8", "manifest milestone mismatch")
    artifact_info = require_dict(report, root.get("artifact"), "manifest.artifact")
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("size_bytes") == artifact.stat().st_size, "manifest artifact size drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    model = require_dict(report, root.get("model"), "manifest.model")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "manifest model sha drift")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "manifest model size drift")
    commands = "\n".join(require_list(report, root.get("capture_commands"), "manifest.capture_commands"))
    for required in (
        "git archive HEAD",
        "make ds4 CUDA_ARCH=native",
        "--write-baseline ds4-parity/baselines/cli/m8.8/current-c.json",
        "--negative-test",
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1",
    ):
        report.check(required in commands, f"manifest capture command missing {required}")
    normalization = require_dict(report, root.get("normalization"), "manifest.normalization")
    for key in ("stdout", "stderr", "dispatch"):
        report.check(isinstance(normalization.get(key), str) and normalization[key], f"manifest normalization missing {key}")
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
    expect_failure("model sha drift", ["model", "sha256"], "0" * 64)
    expect_failure("summary model drift", ["cases", 0, "summary", "model"], "Wrong Model")
    expect_failure("tensor type drift", ["cases", 0, "summary", "tensor_types", 0, "count"], 999)
    expect_failure("stderr anchor drift", ["cases", 0, "stderr_anchors"], [])
    expect_failure("unexpected path stderr", ["cases", 0, "stderr", "base64"], b64(b"ds4: context buffers\n"))
    expect_failure("reference stdout drift", ["cases", 1, "stdout", "sha256"], "0" * 64)
    if manifest_path is not None:
        manifest = load_json(manifest_path)
        manifest["artifact"]["sha256"] = "0" * 64
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(manifest, tmp)
            tmp.write("\n")
        try:
            sub = check_manifest(tmp_path, artifact_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        report.check(not sub.ok, "negative test did not fail: manifest sha drift")
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
        baseline = capture_baseline(args.binary, args.model_sha256)
        write_json(args.write_baseline, baseline)
        if args.write_manifest:
            write_json(args.write_manifest, build_manifest(args.write_baseline))
        return 0
    if args.write_manifest:
        write_json(args.write_manifest, build_manifest(args.artifact))
        return 0

    obj = load_json(args.artifact)
    dump_report = check_dump(obj)
    print_report("CLI inspect oracle", dump_report)
    ok = dump_report.ok
    manifest_path = args.manifest
    if manifest_path is None and args.artifact.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.artifact)
        print_report("CLI inspect manifest", manifest_report)
        ok = ok and manifest_report.ok
    if args.negative_test:
        negative_report = run_negative_tests(obj, manifest_path, args.artifact)
        print_report("CLI inspect negative tests", negative_report)
        ok = ok and negative_report.ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
