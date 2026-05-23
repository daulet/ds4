#!/usr/bin/env python3
"""Validate the M8.14 current-C interactive CLI PTY oracle."""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import tempfile
import termios
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "cli" / "m8.14"
FIXTURE_DIR = ROOT / "ds4-parity" / "baselines" / "cli-fixtures" / "m8.14"
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

READ_PROMPT = "ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt"
CTRL_C = "<CTRL-C>"

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMING_RE = re.compile(r"ds4: prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s")
STARTUP_RE = re.compile(r"in [0-9.]+s")
FORBIDDEN_TRANSCRIPT = (
    "perplexity",
    "imatrix",
    "--dump-logprobs",
    "diagnostic run completed",
    "M8.13",
)


@dataclass(frozen=True)
class InteractiveCase:
    case_id: str
    argv: tuple[str, ...]
    script: tuple[str, ...]
    exit_code: int = 0
    availability: str = "executed"
    anchors: tuple[str, ...] = ()
    normalized_anchors: tuple[str, ...] = ()
    min_prompt_markers: int = 1
    timeout_s: float = 180.0


BASE_ARGS = (
    "--cuda",
    "-m",
    B300_MODEL,
    "--ctx",
    "128",
    "--tokens",
    "1",
    "--temp",
    "0",
    "--nothink",
)

CASES: tuple[InteractiveCase, ...] = (
    InteractiveCase(
        "command_suite",
        BASE_ARGS,
        (
            "",
            "/help",
            "/think",
            "/think-max",
            "/nothink",
            "/ctx 128",
            f"/read {READ_PROMPT}",
            "/definitely-unknown",
            "Answer with one short noun: glacier.",
            "/quit",
        ),
        anchors=(
            "Commands:",
            "/help          Show this help.",
            "Thinking mode: high.",
            "ds4: warning: /think-max needs --ctx >= 393216; ctx=128 uses normal thinking instead",
            "Thinking mode: high (ctx below 393216).",
            "Thinking mode: none.",
            "ds4: unknown command: /definitely-unknown",
            "ds4: type /help for commands",
            "ds4> ",
            "backend=cuda",
        ),
        normalized_anchors=("ds4: prefill: <rate> t/s, generation: <rate> t/s",),
        min_prompt_markers=9,
    ),
    InteractiveCase(
        "ctrl_c_at_prompt",
        BASE_ARGS,
        (
            CTRL_C,
            "/quit",
        ),
        anchors=("Commands:", "ds4> "),
        min_prompt_markers=2,
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


def normalize_transcript(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = TIMING_RE.sub("ds4: prefill: <rate> t/s, generation: <rate> t/s", text)
    text = STARTUP_RE.sub("in <seconds>s", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).rstrip() + "\n"


def read_available(master_fd: int, transcript: bytearray, deadline: float) -> None:
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.05)
        if not ready:
            return
        try:
            chunk = os.read(master_fd, 8192)
        except OSError:
            return
        if not chunk:
            return
        transcript.extend(chunk)


def set_pty_size(master_fd: int, rows: int = 24, cols: int = 120) -> None:
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)


def wait_for_repl_ready(
    master_fd: int,
    pid: int,
    returncode: list[int | None],
    transcript: bytearray,
    deadline: float,
) -> None:
    while time.monotonic() < deadline:
        read_available(master_fd, transcript, deadline)
        normalized = normalize_transcript(bytes(transcript))
        if "Ctrl+C         Stop generation and return to the prompt." in normalized or "ds4> " in normalized:
            return
        if poll_child(pid, returncode) is not None:
            read_available(master_fd, transcript, deadline)
            return
        time.sleep(0.05)
    tail = normalize_transcript(bytes(transcript))[-2000:]
    raise TimeoutError(f"timed out waiting for interactive help banner; transcript tail:\n{tail}")


def wait_for_condition(
    master_fd: int,
    pid: int,
    returncode: list[int | None],
    transcript: bytearray,
    deadline: float,
    label: str,
    predicate: Any,
) -> None:
    while time.monotonic() < deadline:
        read_available(master_fd, transcript, deadline)
        normalized = normalize_transcript(bytes(transcript))
        if predicate(normalized):
            time.sleep(0.2)
            read_available(master_fd, transcript, deadline)
            return
        if poll_child(pid, returncode) is not None:
            read_available(master_fd, transcript, deadline)
            return
        time.sleep(0.05)
    tail = normalize_transcript(bytes(transcript))[-2000:]
    raise TimeoutError(f"timed out waiting for {label}; transcript tail:\n{tail}")


def wait_after_step(
    master_fd: int,
    pid: int,
    returncode: list[int | None],
    transcript: bytearray,
    deadline: float,
    step: str,
) -> None:
    if step == CTRL_C or step == "":
        time.sleep(0.3)
        read_available(master_fd, transcript, deadline)
    elif step == "/help":
        wait_for_condition(master_fd, pid, returncode, transcript, deadline, step, lambda text: text.count("Commands:") >= 2)
    elif step == "/think":
        wait_for_condition(master_fd, pid, returncode, transcript, deadline, step, lambda text: "Thinking mode: high." in text)
    elif step == "/think-max":
        wait_for_condition(
            master_fd,
            pid,
            returncode,
            transcript,
            deadline,
            step,
            lambda text: "Thinking mode: high (ctx below 393216)." in text,
        )
    elif step == "/nothink":
        wait_for_condition(master_fd, pid, returncode, transcript, deadline, step, lambda text: "Thinking mode: none." in text)
    elif step.startswith("/ctx"):
        wait_for_condition(master_fd, pid, returncode, transcript, deadline, step, lambda text: text.count("ds4: context buffers") >= 2)
    elif step.startswith("/read"):
        wait_for_condition(
            master_fd,
            pid,
            returncode,
            transcript,
            deadline,
            step,
            lambda text: text.count("ds4: prefill: <rate> t/s, generation: <rate> t/s") >= 1,
        )
    elif step.startswith("/"):
        wait_for_condition(master_fd, pid, returncode, transcript, deadline, step, lambda text: "ds4: unknown command:" in text)
    else:
        wait_for_condition(
            master_fd,
            pid,
            returncode,
            transcript,
            deadline,
            "model-backed prompt",
            lambda text: text.count("ds4: prefill: <rate> t/s, generation: <rate> t/s") >= 2,
        )


def poll_child(pid: int, returncode: list[int | None]) -> int | None:
    if returncode[0] is not None:
        return returncode[0]
    try:
        got_pid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        returncode[0] = 0
        return returncode[0]
    if got_pid == 0:
        return None
    returncode[0] = os.waitstatus_to_exitcode(status)
    return returncode[0]


def wait_for_exit(
    master_fd: int,
    pid: int,
    returncode: list[int | None],
    transcript: bytearray,
    deadline: float,
) -> int:
    while time.monotonic() < deadline:
        read_available(master_fd, transcript, deadline)
        rc = poll_child(pid, returncode)
        if rc is not None:
            read_available(master_fd, transcript, deadline)
            return rc
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        rc = poll_child(pid, returncode)
        return 0 if rc is None else rc
    stop_deadline = time.monotonic() + 5.0
    while time.monotonic() < stop_deadline:
        read_available(master_fd, transcript, stop_deadline)
        rc = poll_child(pid, returncode)
        if rc is not None:
            return rc
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        rc = poll_child(pid, returncode)
        return 0 if rc is None else rc
    stop_deadline = time.monotonic() + 5.0
    while time.monotonic() < stop_deadline:
        read_available(master_fd, transcript, stop_deadline)
        rc = poll_child(pid, returncode)
        if rc is not None:
            tail = normalize_transcript(bytes(transcript))[-2000:]
            raise TimeoutError(f"interactive process did not exit before timeout; transcript tail:\n{tail}")
        time.sleep(0.05)
    tail = normalize_transcript(bytes(transcript))[-2000:]
    raise TimeoutError(f"interactive process ignored SIGKILL; transcript tail:\n{tail}")


def capture_case(binary: Path, case: InteractiveCase) -> dict[str, Any]:
    binary_path = binary if binary.is_absolute() else ROOT / binary
    if not binary_path.is_file():
        raise SystemExit(f"missing CLI binary: {binary_path}; build ds4 first")

    transcript = bytearray()
    home = tempfile.mkdtemp(prefix="ds4-cli-pty-home-")
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "TERM": "xterm", "HOME": home})
    pid, master_fd = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.execvpe(str(binary_path), [str(binary_path), *case.argv], env)

    set_pty_size(master_fd)
    returncode: list[int | None] = [None]
    try:
        deadline = time.monotonic() + case.timeout_s
        wait_for_repl_ready(master_fd, pid, returncode, transcript, deadline)
        time.sleep(0.5)
        read_available(master_fd, transcript, deadline)

        for step in case.script:
            if step == CTRL_C:
                os.write(master_fd, b"\x03")
            else:
                os.write(master_fd, step.encode("utf-8") + b"\r")
            if step in {"/quit", "/exit"}:
                break
            wait_after_step(master_fd, pid, returncode, transcript, deadline, step)

        exit_code = wait_for_exit(master_fd, pid, returncode, transcript, deadline)
    finally:
        os.close(master_fd)
        shutil.rmtree(home, ignore_errors=True)

    raw = bytes(transcript)
    normalized = normalize_transcript(raw)
    return {
        "id": case.case_id,
        "argv": list(case.argv),
        "script": list(case.script),
        "exit_code": exit_code,
        "expected_exit_code": case.exit_code,
        "availability": case.availability,
        "transcript": {
            "base64": b64(raw),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        },
        "normalized_transcript": normalized,
        "normalized_transcript_sha256": sha256_bytes(normalized.encode("utf-8")),
        "anchors": list(case.anchors),
        "normalized_anchors": list(case.normalized_anchors),
        "min_prompt_markers": case.min_prompt_markers,
    }


def capture_baseline(binary: Path, model_sha256: str) -> dict[str, Any]:
    model = Path(B300_MODEL)
    if not model.is_file():
        raise SystemExit(f"missing model: {B300_MODEL}")
    read_fixture = fixture_path("read_prompt.txt")
    if not read_fixture.is_file():
        raise SystemExit(f"missing fixture: {read_fixture}")
    return {
        "schema": "ds4.cli_interactive_oracle.v1",
        "source": "current-c-cli-interactive-pty",
        "binary": "./ds4",
        "model": {"path": B300_MODEL, "size_bytes": model.stat().st_size, "sha256": model_sha256},
        "b300": {
            "context": B300_CONTEXT,
            "namespace": B300_NAMESPACE,
            "pod": B300_POD,
            "workdir": B300_WORKDIR,
            "kubeconfig": B300_KUBECONFIG,
        },
        "fixtures": {
            READ_PROMPT: {
                "bytes": read_fixture.stat().st_size,
                "sha256": sha256_file(read_fixture),
            }
        },
        "cases": [capture_case(binary, case) for case in CASES],
        "normalization": {
            "pty": "ANSI CSI escapes are stripped; CR redraws become LF-delimited lines; trailing spaces are removed",
            "timing": "startup seconds and prefill/generation rates are normalized",
        },
    }


def expected_by_id() -> dict[str, InteractiveCase]:
    return {case.case_id: case for case in CASES}


def check_case(report: Report, raw_case: Any, expected: InteractiveCase, label: str) -> None:
    case = require_dict(report, raw_case, label)
    case_id = expected.case_id
    report.check(case.get("id") == case_id, f"{label}.id drift")
    report.check(case.get("argv") == list(expected.argv), f"{case_id}.argv drift")
    report.check(case.get("script") == list(expected.script), f"{case_id}.script drift")
    report.check(case.get("exit_code") == expected.exit_code, f"{case_id}.exit code drift")
    report.check(case.get("expected_exit_code") == expected.exit_code, f"{case_id}.expected exit code drift")
    report.check(case.get("availability") == expected.availability, f"{case_id}.availability drift")
    report.check(case.get("min_prompt_markers") == expected.min_prompt_markers, f"{case_id}.prompt marker policy drift")

    transcript = require_dict(report, case.get("transcript"), f"{case_id}.transcript")
    raw = unb64(report, transcript.get("base64"), f"{case_id}.transcript.base64")
    normalized = normalize_transcript(raw)
    report.check(transcript.get("bytes") == len(raw), f"{case_id}.transcript byte count drift")
    report.check(transcript.get("sha256") == sha256_bytes(raw), f"{case_id}.transcript sha drift")
    report.check(case.get("normalized_transcript") == normalized, f"{case_id}.normalized transcript drift")
    report.check(
        case.get("normalized_transcript_sha256") == sha256_bytes(normalized.encode("utf-8")),
        f"{case_id}.normalized transcript sha drift",
    )
    report.check(normalized.count("ds4> ") >= expected.min_prompt_markers, f"{case_id}.prompt marker count drift")

    for anchor in require_list(report, case.get("anchors"), f"{case_id}.anchors"):
        report.check(isinstance(anchor, str), f"{case_id}.anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor in normalized, f"{case_id}.missing anchor {anchor!r}")
    for anchor in require_list(report, case.get("normalized_anchors"), f"{case_id}.normalized_anchors"):
        report.check(isinstance(anchor, str), f"{case_id}.normalized anchor invalid")
        if isinstance(anchor, str):
            report.check(anchor in normalized, f"{case_id}.missing normalized anchor {anchor!r}")
    for forbidden in FORBIDDEN_TRANSCRIPT:
        report.check(forbidden not in normalized, f"{case_id}.forbidden transcript marker {forbidden!r}")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.cli_interactive_oracle.v1", "schema drift")
    model = require_dict(report, root.get("model"), "model")
    report.check(model.get("path") == B300_MODEL, "model.path drift")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "model.size_bytes drift")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 drift")
    fixtures = require_dict(report, root.get("fixtures"), "fixtures")
    read_fixture = require_dict(report, fixtures.get(READ_PROMPT), READ_PROMPT)
    if fixture_path("read_prompt.txt").exists():
        report.check(read_fixture.get("bytes") == fixture_path("read_prompt.txt").stat().st_size, "read prompt size drift")
        report.check(read_fixture.get("sha256") == sha256_file(fixture_path("read_prompt.txt")), "read prompt sha drift")

    cases = require_list(report, root.get("cases"), "cases")
    expected = expected_by_id()
    report.check(len(cases) == len(expected), "case count drift")
    seen: set[str] = set()
    for idx, item in enumerate(cases):
        case = require_dict(report, item, f"cases[{idx}]")
        case_id = case.get("id")
        report.check(isinstance(case_id, str), f"cases[{idx}].id missing")
        if not isinstance(case_id, str):
            continue
        seen.add(case_id)
        expected_case = expected.get(case_id)
        report.check(expected_case is not None, f"{case_id}: unexpected case")
        if expected_case is not None:
            check_case(report, case, expected_case, f"cases[{idx}]")
    report.check(set(expected) == seen, f"case id drift expected={sorted(expected)} got={sorted(seen)}")
    return report


def check_manifest(manifest: Any, baseline_path: Path) -> Report:
    report = Report()
    obj = require_dict(report, manifest, "manifest")
    report.check(obj.get("schema") == "ds4.cli_interactive_manifest.v1", "manifest schema drift")
    report.check(obj.get("milestone") == "M8.14", "manifest milestone drift")
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
        "--context hou2-prod1",
        "make ds4 CUDA_ARCH=native",
        "check_cli_interactive_dump.py",
        "--negative-test",
    ):
        report.check(required in joined, f"manifest capture command missing {required!r}")
    return report


def make_manifest(baseline_path: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.cli_interactive_manifest.v1",
        "milestone": "M8.14",
        "oracle": "current C interactive CLI PTY transcripts",
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
            "base64 < ds4-parity/check_cli_interactive_dump.py | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- sh -lc 'base64 -d > /workspace/ds4/ds4-parity/check_cli_interactive_dump.py'",
            "base64 < ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- sh -lc 'mkdir -p /workspace/ds4/ds4-parity/baselines/cli-fixtures/m8.14 && base64 -d > /workspace/ds4/ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt'",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; make ds4 CUDA_ARCH=native; python3 ds4-parity/check_cli_interactive_dump.py --write-baseline ds4-parity/baselines/cli/m8.14/current-c.json --write-manifest ds4-parity/baselines/cli/m8.14/manifest.json --binary ./ds4; python3 ds4-parity/check_cli_interactive_dump.py ds4-parity/baselines/cli/m8.14/current-c.json --manifest ds4-parity/baselines/cli/m8.14/manifest.json --negative-test'",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/cli/m8.14/current-c.json ds4-parity/baselines/cli/m8.14/current-c.json",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/cli/m8.14/manifest.json ds4-parity/baselines/cli/m8.14/manifest.json",
        ],
        "normalization": {
            "pty": "ANSI CSI escapes are stripped; CR redraws become LF-delimited lines; trailing spaces are removed",
            "timing": "startup seconds and prefill/generation rates are normalized",
        },
    }


def run_negative_test(obj: Any) -> Report:
    report = Report()

    def expect_failure(label: str, mutator: Any) -> None:
        candidate = copy.deepcopy(obj)
        mutator(candidate)
        result = check_dump(candidate)
        report.check(not result.ok, f"negative test failed to detect {label}")

    expect_failure("model hash drift", lambda o: o["model"].__setitem__("sha256", "0" * 64))
    expect_failure("normalized transcript drift", lambda o: o["cases"][0].__setitem__("normalized_transcript", "wrong\n"))
    expect_failure("transcript hash drift", lambda o: o["cases"][0]["transcript"].__setitem__("sha256", "0" * 64))
    expect_failure("anchor drift", lambda o: o["cases"][0].__setitem__("anchors", ["missing anchor"]))
    expect_failure("exit code drift", lambda o: o["cases"][1].__setitem__("exit_code", 9))
    expect_failure("fixture hash drift", lambda o: o["fixtures"][READ_PROMPT].__setitem__("sha256", "0" * 64))
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
    rc = print_report("CLI interactive oracle", check_dump(obj))
    if args.manifest and args.manifest.exists():
        rc |= print_report("CLI interactive manifest", check_manifest(load_json(args.manifest), args.baseline))
    if args.negative_test:
        rc |= print_report("CLI interactive negative tests", run_negative_test(obj))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
