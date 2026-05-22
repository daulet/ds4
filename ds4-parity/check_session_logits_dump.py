#!/usr/bin/env python3
"""Validate the M6.4 current-C model-backed session logits fixture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "sampling" / "m6.4"
BASELINE = BASELINE_DIR / "current-c.json"
LOGITS = BASELINE_DIR / "logits.f32le"
MANIFEST = BASELINE_DIR / "manifest.json"

EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_CASES = {
    "short_italian_fact",
    "short_code_completion",
    "short_reasoning_plain",
    "long_memory_archive",
    "long_code_audit",
}
EXPECTED_SKIPPED = {"long_memory_archive", "long_code_audit"}
N_VOCAB = 129_280
LOGITS_BYTES = N_VOCAB * 4


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


def check_float_value(report: Report, value: Any, path: str) -> None:
    report.check(isinstance(value, (int, float)) or value in {"nan", "inf", "-inf"}, f"{path}: invalid float")


def check_score(report: Report, raw: Any, path: str) -> None:
    score = require_dict(report, raw, path)
    report.check(isinstance(score.get("id"), int), f"{path}.id invalid")
    check_hex(report, score.get("bytes_hex"), f"{path}.bytes_hex")
    check_float_value(report, score.get("logit"), f"{path}.logit")
    check_float_value(report, score.get("logprob"), f"{path}.logprob")


def close_float(expected: Any, got: float, tolerance: float = 1e-5) -> bool:
    if not isinstance(expected, (int, float)):
        return False
    if abs(float(expected)) >= 1e20 or abs(got) >= 1e20:
        return abs(float(expected) - got) <= max(abs(float(expected)), abs(got), 1.0) * 1e-6
    return abs(float(expected) - got) <= tolerance


def compute_top_scores(logits: list[float], k: int = 20) -> list[tuple[int, float, float]]:
    top: list[tuple[int, float]] = []
    max_logit = -1.0e30
    for idx, value in enumerate(logits):
        if not math.isfinite(value):
            continue
        if value > max_logit:
            max_logit = value
        insert_at = None
        for pos, (_, top_logit) in enumerate(top):
            if value > top_logit:
                insert_at = pos
                break
        if insert_at is None:
            if len(top) < k:
                top.append((idx, value))
        else:
            top.insert(insert_at, (idx, value))
            del top[k:]
    total = sum(math.exp(value - max_logit) for value in logits if math.isfinite(value))
    logsum = max_logit + math.log(total)
    return [(idx, value, value - logsum) for idx, value in top[:k]]


def check_step_logits(report: Report, segment: bytes, step: dict[str, Any], path: str) -> None:
    if len(segment) != LOGITS_BYTES:
        report.check(False, f"{path}: bad logits segment size")
        return
    logits = [value[0] for value in struct.iter_unpack("<f", segment)]
    selected = 0
    best_logit = -1.0e30
    for idx, value in enumerate(logits):
        if value > best_logit:
            selected = idx
            best_logit = value
    report.check(selected == step.get("selected_token"), f"{path}: selected token does not match logits argmax")
    expected_scores = require_list(report, step.get("top_logprobs"), f"{path}.top_logprobs")
    computed_scores = compute_top_scores(logits, len(expected_scores))
    report.check(len(computed_scores) == len(expected_scores), f"{path}: computed top count drift")
    for idx, (computed, raw_score) in enumerate(zip(computed_scores, expected_scores)):
        score = require_dict(report, raw_score, f"{path}.top_logprobs[{idx}]")
        token, logit, logprob = computed
        report.check(score.get("id") == token, f"{path}.top_logprobs[{idx}].id does not match logits")
        report.check(close_float(score.get("logit"), logit), f"{path}.top_logprobs[{idx}].logit does not match logits")
        report.check(
            close_float(score.get("logprob"), logprob),
            f"{path}.top_logprobs[{idx}].logprob does not match logits",
        )


def check_dump(obj: Any, logits_path: Path) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.session_logits_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-b300-session-logits", "source mismatch")
    report.check(root.get("model") == "deepseek-v4-flash", "model drift")
    report.check(root.get("model_sha256") == EXPECTED_MODEL_SHA256, "model sha256 drift")
    report.check(root.get("backend") == "cuda", "backend drift")
    report.check(root.get("logits_format") == "f32le", "logits format drift")
    report.check(root.get("top_k") == 20, "top_k drift")

    cases = require_list(report, root.get("cases"), "cases")
    names = [case.get("id") for case in cases if isinstance(case, dict)]
    report.check(set(names) == EXPECTED_CASES, "case coverage drift")
    report.check(len(names) == len(set(names)), "duplicate case ids")

    blob = logits_path.read_bytes() if logits_path.is_file() else b""
    report.check(logits_path.is_file(), f"missing logits blob: {logits_path}")
    used_ranges: list[tuple[int, int, str]] = []

    for idx, raw_case in enumerate(cases):
        case = require_dict(report, raw_case, f"cases[{idx}]")
        case_id = case.get("id")
        report.check(isinstance(case_id, str) and bool(case_id), f"cases[{idx}].id invalid")
        report.check(isinstance(case.get("ctx"), int) and case["ctx"] > 0, f"{case_id}.ctx invalid")
        report.check(isinstance(case.get("nsteps"), int) and case["nsteps"] > 0, f"{case_id}.nsteps invalid")
        report.check(isinstance(case.get("prompt_path"), str), f"{case_id}.prompt_path invalid")
        skipped = case.get("skipped")
        report.check(isinstance(skipped, bool), f"{case_id}.skipped invalid")
        if case_id in EXPECTED_SKIPPED:
            report.check(skipped is True, f"{case_id}: expected skip")
            report.check(isinstance(case.get("skip_reason"), str) and bool(case["skip_reason"]), f"{case_id}.skip_reason invalid")
            report.check(case.get("steps") == [], f"{case_id}: skipped case should have no steps")
            continue
        report.check(skipped is False, f"{case_id}: unexpected skip")
        check_hex(report, case.get("prompt_sha256"), f"{case_id}.prompt_sha256", 64)
        prompt_path = ROOT / str(case.get("prompt_path", ""))
        report.check(prompt_path.is_file(), f"{case_id}.prompt_path missing")
        if prompt_path.is_file():
            report.check(
                sha256_file(prompt_path) == case.get("prompt_sha256"),
                f"{case_id}.prompt_sha256 drift",
            )
        report.check(isinstance(case.get("prompt_tokens"), int) and case["prompt_tokens"] > 0, f"{case_id}.prompt_tokens invalid")
        steps = require_list(report, case.get("steps"), f"{case_id}.steps")
        report.check(len(steps) == case.get("nsteps"), f"{case_id}: step count drift")
        for step_idx, raw_step in enumerate(steps):
            step = require_dict(report, raw_step, f"{case_id}.steps[{step_idx}]")
            report.check(step.get("step") == step_idx, f"{case_id}.steps[{step_idx}].step drift")
            report.check(isinstance(step.get("selected_token"), int), f"{case_id}.steps[{step_idx}].selected_token invalid")
            check_hex(report, step.get("selected_bytes_hex"), f"{case_id}.steps[{step_idx}].selected_bytes_hex")
            check_hex(report, step.get("expected_selected_hex"), f"{case_id}.steps[{step_idx}].expected_selected_hex")
            report.check(step.get("selected_matches_expected") is True, f"{case_id}.steps[{step_idx}].selected mismatch")
            report.check(
                step.get("selected_bytes_hex") == step.get("expected_selected_hex"),
                f"{case_id}.steps[{step_idx}].selected bytes drift",
            )
            offset = step.get("logits_offset")
            size = step.get("logits_bytes")
            report.check(isinstance(offset, int) and offset >= 0, f"{case_id}.steps[{step_idx}].logits_offset invalid")
            report.check(size == LOGITS_BYTES, f"{case_id}.steps[{step_idx}].logits_bytes drift")
            check_hex(report, step.get("logits_sha256"), f"{case_id}.steps[{step_idx}].logits_sha256", 64)
            if isinstance(offset, int) and isinstance(size, int) and offset >= 0:
                end = offset + size
                report.check(end <= len(blob), f"{case_id}.steps[{step_idx}]: logits range outside blob")
                if end <= len(blob):
                    segment = blob[offset:end]
                    digest = sha256_bytes(segment)
                    report.check(digest == step.get("logits_sha256"), f"{case_id}.steps[{step_idx}]: logits sha drift")
                    check_step_logits(report, segment, step, f"{case_id}.steps[{step_idx}]")
                    used_ranges.append((offset, end, f"{case_id}:{step_idx}"))
            top_logprobs = require_list(report, step.get("top_logprobs"), f"{case_id}.steps[{step_idx}].top_logprobs")
            report.check(len(top_logprobs) == 20, f"{case_id}.steps[{step_idx}].top_logprobs length drift")
            ids: set[int] = set()
            for score_idx, score in enumerate(top_logprobs):
                score_obj = require_dict(report, score, f"{case_id}.steps[{step_idx}].top_logprobs[{score_idx}]")
                check_score(report, score_obj, f"{case_id}.steps[{step_idx}].top_logprobs[{score_idx}]")
                score_id = score_obj.get("id")
                if isinstance(score_id, int):
                    report.check(score_id not in ids, f"{case_id}.steps[{step_idx}]: duplicate top id {score_id}")
                    ids.add(score_id)
            official_top = require_list(report, step.get("official_top"), f"{case_id}.steps[{step_idx}].official_top")
            report.check(len(official_top) >= 1, f"{case_id}.steps[{step_idx}].official_top empty")
            for top_idx, raw_top in enumerate(official_top):
                top = require_dict(report, raw_top, f"{case_id}.steps[{step_idx}].official_top[{top_idx}]")
                check_hex(report, top.get("bytes_hex"), f"{case_id}.steps[{step_idx}].official_top[{top_idx}].bytes_hex")
                check_float_value(report, top.get("official_logprob"), f"{case_id}.steps[{step_idx}].official_top[{top_idx}].official_logprob")
                report.check(top.get("found") is True, f"{case_id}.steps[{step_idx}].official_top[{top_idx}] missing locally")
                check_score(report, top.get("local_score"), f"{case_id}.steps[{step_idx}].official_top[{top_idx}].local_score")
                delta_path = f"{case_id}.steps[{step_idx}].official_top[{top_idx}].abs_delta"
                delta = top.get("abs_delta")
                check_float_value(report, delta, delta_path)
                if isinstance(delta, (int, float)):
                    report.check(float(delta) <= 4.0, f"{delta_path}: exceeds capture tolerance")
                    local = require_dict(report, top.get("local_score"), f"{case_id}.steps[{step_idx}].official_top[{top_idx}].local_score")
                    official = top.get("official_logprob")
                    local_logprob = local.get("logprob")
                    if isinstance(official, (int, float)) and isinstance(local_logprob, (int, float)):
                        report.check(
                            close_float(delta, abs(float(local_logprob) - float(official))),
                            f"{delta_path}: does not match local/official logprob delta",
                        )

    used_ranges.sort()
    for idx, (start, end, label) in enumerate(used_ranges):
        report.check(start == idx * LOGITS_BYTES, f"{label}: non-contiguous logits offset")
        report.check(end == (idx + 1) * LOGITS_BYTES, f"{label}: logits range size drift")
    report.check(len(blob) == len(used_ranges) * LOGITS_BYTES, "logits blob has unused bytes")
    return report


def check_manifest(path: Path, json_path: Path, logits_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(path), "manifest")
    report.check(manifest.get("schema") == "ds4.session_logits_manifest.v1", "manifest schema mismatch")
    report.check(manifest.get("milestone") == "M6.4", "manifest milestone mismatch")
    artifacts = require_dict(report, manifest.get("artifacts"), "manifest.artifacts")
    current = require_dict(report, artifacts.get("current_c"), "manifest.artifacts.current_c")
    logits = require_dict(report, artifacts.get("logits_blob"), "manifest.artifacts.logits_blob")
    report.check(current.get("path") == "current-c.json", "current-c path drift")
    report.check(current.get("size") == json_path.stat().st_size, "current-c size drift")
    report.check(current.get("sha256") == sha256_file(json_path), "current-c sha drift")
    report.check(logits.get("path") == "logits.f32le", "logits path drift")
    report.check(logits.get("size") == logits_path.stat().st_size, "logits size drift")
    report.check(logits.get("sha256") == sha256_file(logits_path), "logits sha drift")
    report.check(logits.get("format") == "f32le", "logits format drift")
    report.check(logits.get("n_vocab") == N_VOCAB, "logits n_vocab drift")
    expected_steps = logits_path.stat().st_size // LOGITS_BYTES if logits_path.stat().st_size % LOGITS_BYTES == 0 else -1
    report.check(logits.get("steps") == expected_steps, "logits step count drift")
    commands = manifest.get("refresh_commands")
    report.check(isinstance(commands, list) and len(commands) >= 4, "refresh commands missing")
    if isinstance(commands, list):
        joined = "\n".join(str(command) for command in commands)
        for required in (
            "--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig",
            "--context hou2-prod1",
            "ds4-logits-dump",
            "check_session_logits_dump.py",
        ):
            report.check(required in joined, f"refresh commands missing {required}")
    return report


def run_negative_tests(original: dict[str, Any], logits: Path, manifest: Path) -> Report:
    report = Report()
    mutations: list[tuple[str, list[str | int], Any]] = [
        ("selected mismatch", ["cases", 0, "steps", 0, "selected_matches_expected"], False),
        ("selected token vs logits drift", ["cases", 0, "steps", 0, "selected_token"], -1),
        ("prompt hash drift", ["cases", 0, "prompt_sha256"], "0" * 64),
        ("logits sha drift", ["cases", 0, "steps", 0, "logits_sha256"], "0" * 64),
        ("missing top score", ["cases", 0, "steps", 0, "top_logprobs"], []),
        ("official delta drift", ["cases", 0, "steps", 0, "official_top", 0, "abs_delta"], 99.0),
        ("unexpected skip", ["cases", 0, "skipped"], True),
    ]
    for label, path, value in mutations:
        bad = copy.deepcopy(original)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        result = check_dump(bad, logits)
        report.check(not result.ok, f"negative test failed to catch {label}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_mutations: list[tuple[str, list[str | int], Any]] = [
            ("manifest current sha drift", ["artifacts", "current_c", "sha256"], "0" * 64),
            ("manifest logits sha drift", ["artifacts", "logits_blob", "sha256"], "0" * 64),
            ("manifest logits size drift", ["artifacts", "logits_blob", "size"], -1),
            ("manifest refresh commands missing", ["refresh_commands"], []),
        ]
        for label, path, value in manifest_mutations:
            bad_manifest = tmp_path / f"{label.replace(' ', '_')}.json"
            data = load_json(manifest)
            target: Any = data
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            bad_manifest.write_text(json.dumps(data))
            result = check_manifest(bad_manifest, BASELINE, logits)
            report.check(not result.ok, f"negative test failed to catch {label}")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, nargs="?", default=BASELINE)
    parser.add_argument("--logits", type=Path, default=LOGITS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    obj = load_json(args.dump)
    dump_report = check_dump(obj, args.logits)
    print_report("session logits schema", dump_report)
    manifest_report = check_manifest(args.manifest, args.dump, args.logits)
    print_report("session logits manifest", manifest_report)
    negative_report = Report()
    if args.negative_test:
        negative_report = run_negative_tests(obj, args.logits, args.manifest)
        print_report("session logits negative tests", negative_report)
    return 0 if dump_report.ok and manifest_report.ok and negative_report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
