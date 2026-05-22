#!/usr/bin/env python3
"""Validate the M7.8 current-C restore oracle fixture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.8" / "current-c.json"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.8" / "manifest.json"
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_CASES = {
    "disk_seed_payload": ("disk-payload", "seed"),
    "snapshot_seed": ("memory-snapshot", "seed"),
    "disk_continuation_payload": ("disk-payload", "continuation"),
    "snapshot_continuation": ("memory-snapshot", "continuation"),
}
TOP_K = 20
SCORE_TOLERANCE = 1e-5


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def check_hex(report: Report, value: Any, path: str, length: int | None = None) -> None:
    ok = isinstance(value, str) and all(ch in "0123456789abcdef" for ch in value)
    if length is not None:
        ok = ok and isinstance(value, str) and len(value) == length
    report.check(ok, f"{path}: invalid hex")


def float_value(value: Any) -> float | str | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value in {"nan", "inf", "-inf"}:
        return value
    return None


def close_float(expected: Any, got: Any, tolerance: float = SCORE_TOLERANCE) -> bool:
    ev = float_value(expected)
    gv = float_value(got)
    if ev is None or gv is None:
        return False
    if isinstance(ev, str) or isinstance(gv, str):
        return ev == gv
    if math.isnan(ev) or math.isnan(gv):
        return math.isnan(ev) and math.isnan(gv)
    if abs(ev) >= 1e20 or abs(gv) >= 1e20:
        return abs(ev - gv) <= max(abs(ev), abs(gv), 1.0) * 1e-6
    return abs(ev - gv) <= tolerance


def trimmed_sha(path: Path) -> str:
    data = path.read_bytes()
    data = data.rstrip(b"\r\n")
    return sha256_bytes(data)


def check_score(report: Report, raw: Any, path: str) -> dict[str, Any]:
    score = require_dict(report, raw, path)
    report.check(isinstance(score.get("id"), int), f"{path}.id invalid")
    check_hex(report, score.get("bytes_hex"), f"{path}.bytes_hex")
    report.check(float_value(score.get("logit")) is not None, f"{path}.logit invalid")
    report.check(float_value(score.get("logprob")) is not None, f"{path}.logprob invalid")
    return score


def check_state(report: Report, raw: Any, path: str) -> dict[str, Any]:
    state = require_dict(report, raw, path)
    report.check(isinstance(state.get("selected_token"), int), f"{path}.selected_token invalid")
    check_hex(report, state.get("selected_bytes_hex"), f"{path}.selected_bytes_hex")
    scores = require_list(report, state.get("top_logprobs"), f"{path}.top_logprobs")
    report.check(len(scores) == TOP_K, f"{path}.top_logprobs length drift")
    seen: set[int] = set()
    for idx, raw_score in enumerate(scores):
        score = check_score(report, raw_score, f"{path}.top_logprobs[{idx}]")
        score_id = score.get("id")
        if isinstance(score_id, int):
            report.check(score_id not in seen, f"{path}.top_logprobs duplicate id {score_id}")
            seen.add(score_id)
    if scores:
        top = require_dict(report, scores[0], f"{path}.top_logprobs[0]")
        report.check(
            top.get("id") == state.get("selected_token"),
            f"{path}.selected token is not top score",
        )
        report.check(
            top.get("bytes_hex") == state.get("selected_bytes_hex"),
            f"{path}.selected bytes are not top score bytes",
        )
    return state


def check_case(report: Report, raw: Any, path: str) -> None:
    case = require_dict(report, raw, path)
    case_id = case.get("id")
    report.check(case_id in EXPECTED_CASES, f"{path}.id coverage drift")
    expected_kind, expected_prompt = EXPECTED_CASES.get(case_id, (None, None))
    report.check(case.get("kind") == expected_kind, f"{case_id}.kind drift")
    report.check(case.get("prompt_case") == expected_prompt, f"{case_id}.prompt_case drift")
    report.check(case.get("ctx") == 32768, f"{case_id}.ctx drift")
    report.check(isinstance(case.get("prompt_tokens"), int) and case["prompt_tokens"] > 0, f"{case_id}.prompt_tokens invalid")
    report.check(case.get("raw_payload_committed") is False, f"{case_id}.raw_payload_committed must stay false")
    check_hex(report, case.get("header_prefix_hex"), f"{case_id}.header_prefix_hex", 104)
    if expected_kind == "disk-payload":
        report.check(isinstance(case.get("payload_file"), str), f"{case_id}.payload_file invalid")
        report.check(isinstance(case.get("payload_bytes"), int) and case["payload_bytes"] > 1_000_000, f"{case_id}.payload_bytes invalid")
        check_hex(report, case.get("payload_sha256"), f"{case_id}.payload_sha256", 64)
    elif expected_kind == "memory-snapshot":
        report.check(isinstance(case.get("snapshot_bytes"), int) and case["snapshot_bytes"] > 1_000_000, f"{case_id}.snapshot_bytes invalid")
        report.check(isinstance(case.get("snapshot_cap"), int) and case["snapshot_cap"] >= case["snapshot_bytes"], f"{case_id}.snapshot_cap invalid")
        check_hex(report, case.get("snapshot_sha256"), f"{case_id}.snapshot_sha256", 64)

    reference = check_state(report, case.get("reference"), f"{case_id}.reference")
    restored = check_state(report, case.get("restored"), f"{case_id}.restored")
    comparison = require_dict(report, case.get("comparison"), f"{case_id}.comparison")
    report.check(comparison.get("selected_match") is True, f"{case_id}.selected_match drift")
    report.check(comparison.get("top_order_match") is True, f"{case_id}.top_order_match drift")
    report.check(reference.get("selected_token") == restored.get("selected_token"), f"{case_id}.selected token mismatch")
    report.check(reference.get("selected_bytes_hex") == restored.get("selected_bytes_hex"), f"{case_id}.selected bytes mismatch")
    expected_logit_delta = 0.0
    expected_logprob_delta = 0.0
    ref_scores = reference.get("top_logprobs", [])
    got_scores = restored.get("top_logprobs", [])
    for idx, (ref_score, got_score) in enumerate(zip(ref_scores, got_scores)):
        ref_obj = require_dict(report, ref_score, f"{case_id}.reference.top_logprobs[{idx}]")
        got_obj = require_dict(report, got_score, f"{case_id}.restored.top_logprobs[{idx}]")
        report.check(ref_obj.get("id") == got_obj.get("id"), f"{case_id}.top_logprobs[{idx}].id drift")
        report.check(ref_obj.get("bytes_hex") == got_obj.get("bytes_hex"), f"{case_id}.top_logprobs[{idx}].bytes drift")
        if isinstance(ref_obj.get("logit"), (int, float)) and isinstance(got_obj.get("logit"), (int, float)):
            expected_logit_delta = max(expected_logit_delta, abs(float(ref_obj["logit"]) - float(got_obj["logit"])))
        if isinstance(ref_obj.get("logprob"), (int, float)) and isinstance(got_obj.get("logprob"), (int, float)):
            expected_logprob_delta = max(expected_logprob_delta, abs(float(ref_obj["logprob"]) - float(got_obj["logprob"])))
    report.check(close_float(comparison.get("max_abs_logit_delta"), expected_logit_delta), f"{case_id}.max_abs_logit_delta mismatch")
    report.check(close_float(comparison.get("max_abs_logprob_delta"), expected_logprob_delta), f"{case_id}.max_abs_logprob_delta mismatch")
    if isinstance(comparison.get("max_abs_logit_delta"), (int, float)):
        report.check(float(comparison["max_abs_logit_delta"]) <= SCORE_TOLERANCE, f"{case_id}.logit delta exceeds tolerance")
    if isinstance(comparison.get("max_abs_logprob_delta"), (int, float)):
        report.check(float(comparison["max_abs_logprob_delta"]) <= SCORE_TOLERANCE, f"{case_id}.logprob delta exceeds tolerance")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.restore_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-b300-restore", "source mismatch")
    report.check(root.get("model") == "deepseek-v4-flash", "model drift")
    report.check(root.get("model_sha256") == EXPECTED_MODEL_SHA256, "model sha drift")
    report.check(root.get("backend") == "cuda", "backend drift")
    report.check(root.get("top_k") == TOP_K, "top_k drift")
    report.check(root.get("score_abs_tolerance") == SCORE_TOLERANCE, "score tolerance drift")
    fixtures = require_dict(report, root.get("fixtures"), "fixtures")
    for name in ("seed_prompt", "seed_assistant", "continuation_user"):
        fixture = require_dict(report, fixtures.get(name), f"fixtures.{name}")
        fixture_path = ROOT / str(fixture.get("path", ""))
        report.check(fixture_path.is_file(), f"fixtures.{name}.path missing")
        report.check(fixture.get("trim_trailing_newline") is True, f"fixtures.{name}.trim policy drift")
        if fixture_path.is_file():
            report.check(trimmed_sha(fixture_path) == fixture.get("sha256"), f"fixtures.{name}.sha drift")
    cases = require_list(report, root.get("cases"), "cases")
    names = [case.get("id") for case in cases if isinstance(case, dict)]
    report.check(set(names) == set(EXPECTED_CASES), "case coverage drift")
    report.check(len(names) == len(set(names)), "duplicate case ids")
    for idx, raw_case in enumerate(cases):
        check_case(report, raw_case, f"cases[{idx}]")
    return report


def build_manifest(artifact: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.restore_manifest.v1",
        "milestone": "M7.8",
        "oracle": "current C B300 disk payload and in-memory session snapshot restore",
        "capture": {
            "kubeconfig": "/tmp/ds4-hou2-prod1.kubeconfig",
            "context": "hou2-prod1",
            "namespace": "default",
            "pod": "ds4-rust-port-b300",
            "workdir": "/workspace/ds4",
            "model_path": "/workspace/ds4/ds4flash.gguf",
            "model_sha256": EXPECTED_MODEL_SHA256,
        },
        "artifacts": {
            "current_c": {
                "path": "current-c.json",
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        },
        "refresh_commands": [
            "cp ~/.kube/config /tmp/ds4-hou2-prod1.kubeconfig && chmod 600 /tmp/ds4-hou2-prod1.kubeconfig",
            "git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- tar -xf - -C /workspace/ds4",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default exec ds4-rust-port-b300 -- sh -lc 'set -e; cd /workspace/ds4; make ds4-restore-dump CUDA_ARCH=native; mkdir -p ds4-parity/baselines/kv/m7.8 ds4-parity/baselines/kv/m7.8/raw; ./ds4-restore-dump --backend cuda -m /workspace/ds4/ds4flash.gguf --model-sha256 efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668 --seed-prompt ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_prompt.txt --seed-assistant ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_assistant.txt --continuation-user ds4-parity/baselines/kv-fixtures/m7.8/restore_continuation_user.txt --payload-dir ds4-parity/baselines/kv/m7.8/raw -o ds4-parity/baselines/kv/m7.8/current-c.json; python3 ds4-parity/check_restore_dump.py ds4-parity/baselines/kv/m7.8/current-c.json --negative-test'",
            "kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 -n default cp ds4-rust-port-b300:/workspace/ds4/ds4-parity/baselines/kv/m7.8/current-c.json ds4-parity/baselines/kv/m7.8/current-c.json",
            "python3 ds4-parity/check_restore_dump.py ds4-parity/baselines/kv/m7.8/current-c.json --manifest ds4-parity/baselines/kv/m7.8/manifest.json --negative-test",
        ],
    }


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    manifest = load_json(path)
    root = require_dict(report, manifest, "manifest")
    report.check(root.get("schema") == "ds4.restore_manifest.v1", "manifest schema mismatch")
    artifact_info = require_dict(report, require_dict(report, root.get("artifacts"), "manifest.artifacts").get("current_c"), "manifest.artifacts.current_c")
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    report.check(artifact_info.get("size") == artifact.stat().st_size, "manifest artifact size drift")
    commands = "\n".join(require_list(report, root.get("refresh_commands"), "manifest.refresh_commands"))
    for required in (
        "/tmp/ds4-hou2-prod1.kubeconfig",
        "--context hou2-prod1",
        "make ds4-restore-dump CUDA_ARCH=native",
        "--payload-dir ds4-parity/baselines/kv/m7.8/raw",
        "python3 ds4-parity/check_restore_dump.py",
    ):
        report.check(required in commands, f"manifest refresh command missing {required}")
    return report


def run_negative_tests(obj: Any) -> Report:
    report = Report()

    def expect_failure(name: str, path: list[str | int], value: Any) -> None:
        candidate = copy.deepcopy(obj)
        target: Any = candidate
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        sub = check_dump(candidate)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure("selected drift", ["cases", 0, "restored", "selected_token"], -1)
    expect_failure("top id drift", ["cases", 1, "restored", "top_logprobs", 0, "id"], -1)
    expect_failure("delta drift", ["cases", 2, "comparison", "max_abs_logprob_delta"], 1.0)
    expect_failure("raw payload committed", ["cases", 0, "raw_payload_committed"], True)
    expect_failure("fixture sha drift", ["fixtures", "seed_prompt", "sha256"], "0" * 64)
    expect_failure("case coverage drift", ["cases"], obj["cases"][:-1])
    return report


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    if args.write_manifest:
        write_json(args.write_manifest, build_manifest(args.artifact))
        return 0

    obj = load_json(args.artifact)
    dump_report = check_dump(obj)
    print_report("restore oracle schema", dump_report)
    ok = dump_report.ok

    manifest_path = args.manifest
    if manifest_path is None and args.artifact.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.artifact)
        print_report("restore manifest", manifest_report)
        ok = ok and manifest_report.ok

    if args.negative_test:
        negative_report = run_negative_tests(obj)
        print_report("restore oracle negative tests", negative_report)
        ok = ok and negative_report.ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
