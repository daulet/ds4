#!/usr/bin/env python3
"""Validate the M8.4 current-C CLI --dump-tokens oracle."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.4"
FIXTURE_DIR = ROOT / "ds4-parity" / "baselines" / "cli-fixtures" / "m8.4"
BASELINE = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"

EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_SIZE = 86720111488
INLINE_PROMPT = 'CLI token oracle prompt: cache slots 17 and JSON {"k":1}.'
SYSTEM_PROMPT = "System prompt must not affect dump-tokens."
B300_KUBECONFIG = "/tmp/ds4-hou2-prod1.kubeconfig"
B300_CONTEXT = "hou2-prod1"
B300_NAMESPACE = "default"
B300_POD = "ds4-rust-port-b300"
B300_WORKDIR = "/workspace/ds4"
B300_MODEL = "/workspace/ds4/ds4flash.gguf"


@dataclass(frozen=True)
class TokenDumpCase:
    case_id: str
    prompt_source: str
    prompt_text: str | None = None
    fixture_name: str | None = None
    extra_args: tuple[str, ...] = ()
    think_control: str = "default"
    reference_case_id: str | None = None
    rendered_chat_passthrough: bool = False

    def prompt_bytes(self) -> bytes:
        if self.prompt_text is not None:
            return self.prompt_text.encode("utf-8")
        if self.fixture_name is None:
            raise ValueError(f"{self.case_id}: missing prompt fixture")
        return fixture_path(self.fixture_name).read_bytes()

    def prompt_args(self) -> tuple[str, ...]:
        if self.prompt_text is not None:
            return ("-p", self.prompt_text)
        if self.fixture_name is None:
            raise ValueError(f"{self.case_id}: missing prompt fixture")
        return ("--prompt-file", fixture_relpath(self.fixture_name))


CASES: tuple[TokenDumpCase, ...] = (
    TokenDumpCase("inline_prompt", "argv:-p", prompt_text=INLINE_PROMPT),
    TokenDumpCase("prompt_file", "file:prompt_file.txt", fixture_name="prompt_file.txt"),
    TokenDumpCase(
        "rendered_chat_passthrough",
        "file:rendered_chat.txt",
        fixture_name="rendered_chat.txt",
        rendered_chat_passthrough=True,
    ),
    TokenDumpCase(
        "custom_system_ignored",
        "argv:-p",
        prompt_text=INLINE_PROMPT,
        extra_args=("--system", SYSTEM_PROMPT),
        reference_case_id="inline_prompt",
    ),
    TokenDumpCase(
        "empty_system_ignored",
        "argv:-p",
        prompt_text=INLINE_PROMPT,
        extra_args=("--system", ""),
        reference_case_id="inline_prompt",
    ),
    TokenDumpCase(
        "think_ignored",
        "argv:-p",
        prompt_text=INLINE_PROMPT,
        extra_args=("--think",),
        think_control="high",
        reference_case_id="inline_prompt",
    ),
    TokenDumpCase(
        "think_max_low_ctx_ignored",
        "argv:-p",
        prompt_text=INLINE_PROMPT,
        extra_args=("--think-max", "--ctx", "32768"),
        think_control="max_ctx_below_threshold",
        reference_case_id="inline_prompt",
    ),
    TokenDumpCase(
        "think_max_high_ctx_ignored",
        "argv:-p",
        prompt_text=INLINE_PROMPT,
        extra_args=("--think-max", "--ctx", "393216"),
        think_control="max_ctx_above_threshold",
        reference_case_id="inline_prompt",
    ),
    TokenDumpCase(
        "nothink_ignored",
        "argv:-p",
        prompt_text=INLINE_PROMPT,
        extra_args=("--nothink",),
        think_control="none",
        reference_case_id="inline_prompt",
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
    return f"ds4-parity/baselines/cli-fixtures/m8.4/{name}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(report: Report, value: Any, label: str) -> bytes:
    report.check(isinstance(value, str), f"{label} must be base64 string")
    if not isinstance(value, str):
        return b""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        report.check(False, f"{label} invalid base64: {exc}")
        return b""


def token_ids_sha256(token_ids: list[int]) -> str:
    blob = bytearray()
    for token_id in token_ids:
        blob.extend(f"{token_id}\n".encode("ascii"))
    return sha256_bytes(bytes(blob))


def parse_token_ids(stdout: bytes) -> list[int]:
    first_line = stdout.split(b"\n", 1)[0]
    try:
        parsed = ast.literal_eval(first_line.decode("ascii"))
    except Exception as exc:
        raise ValueError(f"failed to parse token id line {first_line!r}: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, int) and item >= 0 for item in parsed):
        raise ValueError(f"token id line must be a list of nonnegative ints: {first_line!r}")
    return parsed


def resolve_binary(binary: Path) -> Path:
    return binary if binary.is_absolute() else ROOT / binary


def capture_case(binary: Path, model: str, case: TokenDumpCase) -> dict[str, Any]:
    prompt = case.prompt_bytes()
    argv = ("--dump-tokens", "-m", model, *case.prompt_args(), *case.extra_args)
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
    if proc.returncode == 0:
        token_ids = parse_token_ids(proc.stdout)
    else:
        token_ids = []
    prompt_file = fixture_relpath(case.fixture_name) if case.fixture_name is not None else None
    return {
        "id": case.case_id,
        "argv": list(argv),
        "prompt_source": case.prompt_source,
        "prompt_file": prompt_file,
        "prompt_utf8": prompt.decode("utf-8"),
        "prompt_base64": b64(prompt),
        "prompt_bytes": len(prompt),
        "prompt_sha256": sha256_bytes(prompt),
        "think_control": case.think_control,
        "warning_category": "none",
        "dump_tokens_prompt_builder": "not_used",
        "dump_tokens_think_effect": "ignored_before_prompt_build",
        "rendered_chat_passthrough": case.rendered_chat_passthrough,
        "reference_case_id": case.reference_case_id,
        "exit_code": proc.returncode,
        "stdout_base64": b64(proc.stdout),
        "stdout_bytes": len(proc.stdout),
        "stdout_sha256": sha256_bytes(proc.stdout),
        "stderr_base64": b64(proc.stderr),
        "stderr_bytes": len(proc.stderr),
        "stderr_sha256": sha256_bytes(proc.stderr),
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
    }


def capture_baseline(binary: Path, model: str, model_sha256: str) -> dict[str, Any]:
    binary_path = resolve_binary(binary)
    if not binary_path.is_file():
        raise SystemExit(f"missing CLI binary: {binary_path}; build ds4 first")
    model_path = Path(model)
    if not model_path.is_file():
        raise SystemExit(f"missing model: {model}")
    model_size = model_path.stat().st_size
    return {
        "schema": "ds4.cli_token_dump_oracle.v1",
        "source": "current-c-cli-token-dump",
        "binary": "./ds4",
        "model": {
            "path": model,
            "size": model_size,
            "sha256": model_sha256,
        },
        "source_evidence": {
            "dump_tokens_entry": "ds4_cli.c main calls ds4_dump_text_tokenization(cfg.engine.model_path, cfg.gen.prompt, stdout) before build_prompt and cli_warn_think_max_downgraded",
            "tokenizer_entry": "ds4.c ds4_dump_text_tokenization calls tokenize_rendered_chat_vocab and dump_tokens_fp",
        },
        "cases": [capture_case(binary, model, case) for case in CASES],
    }


def expected_by_id() -> dict[str, TokenDumpCase]:
    return {case.case_id: case for case in CASES}


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, label: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{label}: expected array")
    return obj if isinstance(obj, list) else []


def is_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def check_case(report: Report, raw: Any, expected: TokenDumpCase, refs: dict[str, dict[str, Any]], label: str) -> None:
    case = require_dict(report, raw, label)
    report.check(case.get("id") == expected.case_id, f"{label}.id drift")
    report.check(case.get("prompt_source") == expected.prompt_source, f"{expected.case_id}.prompt_source drift")
    report.check(case.get("think_control") == expected.think_control, f"{expected.case_id}.think_control drift")
    report.check(case.get("warning_category") == "none", f"{expected.case_id}.warning_category drift")
    report.check(case.get("dump_tokens_prompt_builder") == "not_used", f"{expected.case_id}.prompt_builder drift")
    report.check(
        case.get("dump_tokens_think_effect") == "ignored_before_prompt_build",
        f"{expected.case_id}.think_effect drift",
    )
    report.check(
        case.get("rendered_chat_passthrough") == expected.rendered_chat_passthrough,
        f"{expected.case_id}.rendered_chat_passthrough drift",
    )
    report.check(case.get("reference_case_id") == expected.reference_case_id, f"{expected.case_id}.reference drift")
    report.check(case.get("exit_code") == 0, f"{expected.case_id}.exit_code drift")

    expected_argv = ["--dump-tokens", "-m", B300_MODEL, *expected.prompt_args(), *expected.extra_args]
    report.check(case.get("argv") == expected_argv, f"{expected.case_id}.argv drift")

    prompt = unb64(report, case.get("prompt_base64"), f"{expected.case_id}.prompt_base64")
    expected_prompt = expected.prompt_bytes()
    report.check(prompt == expected_prompt, f"{expected.case_id}.prompt bytes drift")
    report.check(case.get("prompt_utf8") == expected_prompt.decode("utf-8"), f"{expected.case_id}.prompt_utf8 drift")
    report.check(case.get("prompt_bytes") == len(prompt), f"{expected.case_id}.prompt_bytes drift")
    report.check(case.get("prompt_sha256") == sha256_bytes(prompt), f"{expected.case_id}.prompt_sha256 drift")
    if expected.fixture_name is not None:
        rel = fixture_relpath(expected.fixture_name)
        fixture = ROOT / rel
        report.check(case.get("prompt_file") == rel, f"{expected.case_id}.prompt_file drift")
        report.check(fixture.exists(), f"{expected.case_id}.fixture missing: {rel}")
        if fixture.exists():
            report.check(case.get("prompt_sha256") == sha256_file(fixture), f"{expected.case_id}.fixture sha drift")
    else:
        report.check(case.get("prompt_file") is None, f"{expected.case_id}.prompt_file must be null")

    stdout = unb64(report, case.get("stdout_base64"), f"{expected.case_id}.stdout_base64")
    stderr = unb64(report, case.get("stderr_base64"), f"{expected.case_id}.stderr_base64")
    report.check(case.get("stdout_bytes") == len(stdout), f"{expected.case_id}.stdout_bytes drift")
    report.check(case.get("stderr_bytes") == len(stderr), f"{expected.case_id}.stderr_bytes drift")
    report.check(case.get("stdout_sha256") == sha256_bytes(stdout), f"{expected.case_id}.stdout_sha256 drift")
    report.check(case.get("stderr_sha256") == sha256_bytes(stderr), f"{expected.case_id}.stderr_sha256 drift")
    report.check(stderr == b"", f"{expected.case_id}.stderr should be empty")
    report.check(stdout.endswith(b"\n"), f"{expected.case_id}.stdout should end with newline")

    try:
        parsed_ids = parse_token_ids(stdout)
    except ValueError as exc:
        report.check(False, f"{expected.case_id}.stdout token id line invalid: {exc}")
        parsed_ids = []
    token_ids = require_list(report, case.get("token_ids"), f"{expected.case_id}.token_ids")
    report.check(token_ids == parsed_ids, f"{expected.case_id}.token_ids drift")
    report.check(case.get("token_count") == len(parsed_ids), f"{expected.case_id}.token_count drift")
    report.check(len(parsed_ids) > 0, f"{expected.case_id}.token_count should be nonzero")
    report.check(case.get("token_ids_sha256") == token_ids_sha256(parsed_ids), f"{expected.case_id}.token_ids_sha256 drift")

    if expected.reference_case_id is not None:
        ref = refs.get(expected.reference_case_id)
        report.check(ref is not None, f"{expected.case_id}.reference missing")
        if ref is not None:
            report.check(case.get("stdout_sha256") == ref.get("stdout_sha256"), f"{expected.case_id}.stdout differs from reference")
            report.check(case.get("token_ids") == ref.get("token_ids"), f"{expected.case_id}.token ids differ from reference")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.cli_token_dump_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-cli-token-dump", "source mismatch")
    report.check(root.get("binary") == "./ds4", "binary drift")
    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("path") == B300_MODEL, "model.path drift")
    report.check(model.get("size") == EXPECTED_MODEL_SIZE, "model.size drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 drift")
    evidence = require_dict(report, root.get("source_evidence"), "source_evidence")
    for key in ("dump_tokens_entry", "tokenizer_entry"):
        report.check(isinstance(evidence.get(key), str) and evidence[key], f"source_evidence.{key} missing")

    cases = require_list(report, root.get("cases"), "cases")
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    report.check(ids == [case.case_id for case in CASES], "case order or coverage drift")
    report.check(len(ids) == len(set(ids)), "duplicate case ids")
    refs = {case["id"]: case for case in cases if isinstance(case, dict) and isinstance(case.get("id"), str)}
    expected = expected_by_id()
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
        "schema": "ds4.cli_token_dump_manifest.v1",
        "milestone": "M8.4",
        "oracle": "current C CLI --dump-tokens prompt ingestion and raw token dump",
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
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} exec {B300_POD} -- sh -lc 'set -e; cd {B300_WORKDIR}; make ds4 CUDA_ARCH=native; python3 ds4-parity/check_cli_token_dump.py --write-baseline ds4-parity/baselines/cli/m8.4/current-c.json --write-manifest ds4-parity/baselines/cli/m8.4/manifest.json --binary ./ds4 --model {B300_MODEL}; python3 ds4-parity/check_cli_token_dump.py ds4-parity/baselines/cli/m8.4/current-c.json --manifest ds4-parity/baselines/cli/m8.4/manifest.json --negative-test'",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} cp {B300_POD}:{B300_WORKDIR}/ds4-parity/baselines/cli/m8.4/current-c.json ds4-parity/baselines/cli/m8.4/current-c.json",
            f"kubectl --kubeconfig {B300_KUBECONFIG} --context {B300_CONTEXT} -n {B300_NAMESPACE} cp {B300_POD}:{B300_WORKDIR}/ds4-parity/baselines/cli/m8.4/manifest.json ds4-parity/baselines/cli/m8.4/manifest.json",
        ],
        "normalization": {
            "model.path": "B300 link path is retained; comparisons may normalize only the model path string",
            "stdout": "raw stdout bytes, token IDs, and token pieces are exact",
            "stderr": "must be empty for every captured case",
            "thinking_controls": "--system, --think, --think-max, --ctx, and --nothink are expected to have no dump-token effect because main exits before build_prompt and cli_warn_think_max_downgraded",
        },
    }


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    root = require_dict(report, load_json(path), "manifest")
    report.check(root.get("schema") == "ds4.cli_token_dump_manifest.v1", "manifest schema mismatch")
    report.check(root.get("milestone") == "M8.4", "manifest milestone mismatch")
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
        "--write-baseline ds4-parity/baselines/cli/m8.4/current-c.json",
        "--negative-test",
        "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1",
    ):
        report.check(required in commands, f"manifest capture command missing {required}")
    normalization = require_dict(report, root.get("normalization"), "manifest.normalization")
    report.check("thinking_controls" in normalization, "manifest normalization missing thinking_controls")
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
    expect_failure("prompt sha drift", ["cases", 0, "prompt_sha256"], "0" * 64)
    expect_failure("stdout sha drift", ["cases", 0, "stdout_sha256"], "0" * 64)
    expect_failure("token id drift", ["cases", 0, "token_ids", 0], 99999999)
    expect_failure("warning category drift", ["cases", 6, "warning_category"], "think-max-downgraded")
    expect_failure("ignored control output drift", ["cases", 3, "stdout_sha256"], "0" * 64)
    expect_failure("model sha drift", ["model", "sha256"], "0" * 64)

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
    parser.add_argument("--model", default=B300_MODEL)
    parser.add_argument("--model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if args.write_baseline:
        baseline = capture_baseline(args.binary, args.model, args.model_sha256)
        write_json(args.write_baseline, baseline)
        if args.write_manifest:
            write_json(args.write_manifest, build_manifest(args.write_baseline))
        return 0
    if args.write_manifest:
        write_json(args.write_manifest, build_manifest(args.artifact))
        return 0

    obj = load_json(args.artifact)
    dump_report = check_dump(obj)
    print_report("CLI token dump oracle", dump_report)
    ok = dump_report.ok

    manifest_path = args.manifest
    if manifest_path is None and args.artifact.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.artifact)
        print_report("CLI token dump manifest", manifest_report)
        ok = ok and manifest_report.ok

    if args.negative_test:
        negative_report = run_negative_tests(obj, manifest_path, args.artifact)
        print_report("CLI token dump negative tests", negative_report)
        ok = ok and negative_report.ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
