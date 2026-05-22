#!/usr/bin/env python3
"""Compare Rust KV replay decisions against committed current-C trace artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.7" / "current-c.json"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.7" / "manifest.json"
M04 = ROOT / "ds4-parity" / "baselines" / "server-traces" / "m0.4"
M05 = ROOT / "ds4-parity" / "baselines" / "kv-artifacts" / "m0.5"
M05_FIXTURES = ROOT / "ds4-parity" / "baselines" / "kv-fixtures" / "m0.5"

M5_PRECONDITION_PATHS = [
    "ds4-parity/baselines/tokenization/m5.2/current-c.json",
    "ds4-parity/baselines/tokenization/m5.2/manifest.json",
    "ds4-parity/baselines/dsml/m5.6a/current-c.json",
    "ds4-parity/baselines/dsml/m5.6a/manifest.json",
    "ds4-parity/baselines/dsml/m5.6b/current-c.json",
    "ds4-parity/baselines/dsml/m5.6b/manifest.json",
]

FIXTURE_HASH_PATHS = [
    "ds4-parity/baselines/server-traces/m0.4/traces/server.trace",
    "ds4-parity/baselines/server-traces/m0.4/responses/chat_tool_call.json",
    "ds4-parity/baselines/server-traces/m0.4/responses/chat_cache_seed.json",
    "ds4-parity/baselines/server-traces/m0.4/responses/chat_cache_continuation.json",
    "ds4-parity/baselines/server-fixtures/m0.4/chat_cache_seed.json",
    "ds4-parity/baselines/server-fixtures/m0.4/chat_cache_continuation.json",
    "ds4-parity/baselines/kv-artifacts/m0.5/logs/cache-decisions.txt",
    "ds4-parity/baselines/kv-artifacts/m0.5/logs/kv-header.tsv",
    "ds4-parity/baselines/kv-artifacts/m0.5/responses/seed_miss.json",
    "ds4-parity/baselines/kv-artifacts/m0.5/responses/seed_restore.json",
    "ds4-parity/baselines/kv-artifacts/m0.5/responses/continuation_restore.json",
    "ds4-parity/baselines/kv-artifacts/m0.5/traces/server-a.trace",
    "ds4-parity/baselines/kv-artifacts/m0.5/traces/server-b.trace",
    "ds4-parity/baselines/kv-artifacts/m0.5/traces/server-c.trace",
    "ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/0ab2314538b11686a11e296b7f697651fbd17e60.txt",
    "ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/a0cac6ff193696ccb5d7e9ae151d7255d39cf161.txt",
    "ds4-parity/baselines/kv-fixtures/m0.5/kv_seed.json",
    "ds4-parity/baselines/kv-fixtures/m0.5/kv_continuation.json",
    "ds4-parity/baselines/kv/m7.2/current-c.json",
] + M5_PRECONDITION_PATHS

RUST_REPLAY_FIELDS = [
    "name",
    "prompt_tokens",
    "live_tokens_before",
    "live_prompt_common",
    "memory_token_reusable",
    "memory_miss_reason",
    "cache_source",
    "cached_tokens",
    "disk_cached_tokens",
    "cache_write_tokens",
    "disk_cache_file",
    "disk_cache_reason_name",
    "disk_cache_reason_code",
    "disk_cache_ext_flags",
    "disk_cache_key_kind",
    "rendered_text_sha256",
    "rendered_text_bytes",
    "rendered_prompt_sha256",
    "rendered_prompt_bytes",
    "effective_suffix_hex",
    "effective_suffix_bytes",
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


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_dict(report: Report, obj: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{path}: expected object")
    return obj if isinstance(obj, dict) else {}


def require_list(report: Report, obj: Any, path: str) -> list[Any]:
    report.check(isinstance(obj, list), f"{path}: expected array")
    return obj if isinstance(obj, list) else []


def int_value(value: str) -> int:
    return int(value.strip())


def key_kind(ext_flags: int | None) -> str | None:
    if ext_flags is None:
        return None
    if ext_flags & 2:
        return "responses-visible"
    if ext_flags & 4:
        return "thinking-visible"
    return "token-text"


def reason_name_to_code(name: str | None) -> int | None:
    return {
        "cold": 1,
        "continued": 2,
        "evict": 3,
        "shutdown": 4,
        "agent-system": 5,
        "agent-session": 6,
    }.get(name) if name is not None else None


def parse_requests(path: Path) -> dict[int, str]:
    text = path.read_text()
    requests: dict[int, str] = {}
    pattern = re.compile(r"===== request (\d+) [^\n]* =====\n(.*?)===== end request \1 =====", re.S)
    for match in pattern.finditer(text):
        requests[int(match.group(1))] = match.group(2)
    return requests


def section(block: str, title: str) -> str:
    marker = f"--- {title} ---\n"
    if marker not in block:
        return ""
    rest = block.split(marker, 1)[1]
    next_marker = rest.find("\n--- ")
    if next_marker >= 0:
        return rest[:next_marker].strip("\n")
    return rest.strip("\n")


def parse_header_fields(block: str) -> dict[str, Any]:
    head = block.split("--- cache decision ---", 1)[0]
    result: dict[str, Any] = {}
    for raw in head.splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {
            "prompt_tokens",
            "effective_prompt_tokens",
            "cached_tokens",
            "max_tokens",
            "stream",
            "tools",
        }:
            result[key] = int_value(value)
        elif key in {"kind", "model", "think_mode"}:
            result[key] = value
    return result


def parse_cache_decision(block: str) -> dict[str, Any]:
    raw = section(block, "cache decision")
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("tool_replay:"):
            match = re.search(r"mem=(\d+) disk=(\d+) canonical=(\d+) missing_ids=(\d+)", line)
            if match:
                result["tool_replay"] = {
                    "mem": int(match.group(1)),
                    "disk": int(match.group(2)),
                    "canonical": int(match.group(3)),
                    "missing_ids": int(match.group(4)),
                }
            continue
        if line.startswith("token_window:"):
            result["token_window_range"] = line.split(":", 1)[1].strip()
            window = "\n".join(
                item.rstrip()
                for item in raw.split("token_window:", 1)[1].splitlines()[1:]
                if item.strip()
            )
            result["token_window_sha256"] = sha256_bytes(window.encode())
            result["token_window_lines"] = len(window.splitlines()) if window else 0
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {
            "live_tokens_before",
            "prompt_tokens",
            "live_prompt_common",
            "memory_token_reusable",
            "cached_tokens",
            "disk_cached_tokens",
            "first_mismatch_token",
        }:
            result[key] = int_value(value)
        elif key == "disk_cache_file":
            result[key] = Path(value).name
        else:
            result[key] = value
    result.setdefault("tool_replay", {"mem": 0, "disk": 0, "canonical": 0, "missing_ids": 0})
    return result


def rendered_prompt(block: str) -> bytes:
    return section(block, "rendered prompt").encode()


def parsed_message_fields(block: str) -> dict[str, Any]:
    raw = section(block, "parsed message")
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"generated_tokens", "dsml_start", "dsml_end"}:
            result[key] = int_value(value)
        elif key == "finish":
            result[key] = value
    return result


def response_summary(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    choice = obj["choices"][0]
    usage = obj["usage"]
    details = usage["prompt_tokens_details"]
    message = choice["message"]
    return {
        "finish_reason": choice["finish_reason"],
        "content": message.get("content", ""),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cached_tokens": details["cached_tokens"],
        "cache_write_tokens": details["cache_write_tokens"],
    }


def parse_kv_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with (M05 / "logs" / "kv-header.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"file", "sha256", "normalized_sha256", "rendered_text_sha256", "magic", "reason_name"}:
                    normalized[key] = value
                else:
                    normalized[key] = int(value)
            rows[normalized["file"]] = normalized
    return rows


def build_replay_case(
    name: str,
    artifact_family: str,
    fixture: str,
    response_path: Path,
    block: str,
    kv_rows: dict[str, dict[str, Any]],
    prefix_prompt: bytes | None = None,
) -> dict[str, Any]:
    header = parse_header_fields(block)
    cache = parse_cache_decision(block)
    prompt = rendered_prompt(block)
    parsed = parsed_message_fields(block)
    response = response_summary(response_path)

    disk_file = cache.get("disk_cache_file")
    row = kv_rows.get(disk_file) if isinstance(disk_file, str) else None
    rendered_text = None
    if isinstance(disk_file, str):
        rendered_path = M05 / "rendered-text" / f"{disk_file.removesuffix('.kv')}.txt"
        rendered_text = rendered_path.read_bytes()

    suffix: bytes | None = None
    if rendered_text is not None:
        if prompt.startswith(rendered_text):
            suffix = prompt[len(rendered_text) :]
    elif prefix_prompt is not None and prompt.startswith(prefix_prompt):
        suffix = prompt[len(prefix_prompt) :]

    case: dict[str, Any] = {
        "name": name,
        "artifact_family": artifact_family,
        "fixture": fixture,
        "response": rel(response_path),
        "model": header.get("model"),
        "prompt_tokens": cache.get("prompt_tokens"),
        "effective_prompt_tokens": header.get("effective_prompt_tokens"),
        "live_tokens_before": cache.get("live_tokens_before"),
        "live_prompt_common": cache.get("live_prompt_common"),
        "memory_token_reusable": bool(cache.get("memory_token_reusable")),
        "memory_miss_reason": cache.get("memory_miss_reason"),
        "tool_replay": cache.get("tool_replay"),
        "cache_source": cache.get("cache_source"),
        "cached_tokens": cache.get("cached_tokens"),
        "disk_cached_tokens": cache.get("disk_cached_tokens"),
        "cache_write_tokens": response["cache_write_tokens"],
        "disk_cache_file": disk_file if isinstance(disk_file, str) else None,
        "disk_cache_reason_name": row.get("reason_name") if row else None,
        "disk_cache_reason_code": row.get("reason") if row else None,
        "disk_cache_ext_flags": row.get("ext_flags") if row else None,
        "disk_cache_key_kind": key_kind(row.get("ext_flags")) if row else None,
        "rendered_text_sha256": row.get("rendered_text_sha256") if row else None,
        "rendered_text_bytes": row.get("rendered_text_bytes") if row else None,
        "rendered_prompt_sha256": sha256_bytes(prompt),
        "rendered_prompt_bytes": len(prompt),
        "effective_suffix_hex": suffix.hex() if suffix is not None else None,
        "effective_suffix_bytes": len(suffix) if suffix is not None else None,
        "response_finish_reason": response["finish_reason"],
        "response_content": response["content"],
        "response_prompt_tokens": response["prompt_tokens"],
        "response_cached_tokens": response["cached_tokens"],
        "response_cache_write_tokens": response["cache_write_tokens"],
        "parsed_finish": parsed.get("finish"),
        "parsed_generated_tokens": parsed.get("generated_tokens"),
        "parsed_dsml_start": parsed.get("dsml_start"),
        "parsed_dsml_end": parsed.get("dsml_end"),
    }
    if "first_mismatch_token" in cache:
        case["first_mismatch_token"] = cache["first_mismatch_token"]
    if "token_window_sha256" in cache:
        case["token_window_sha256"] = cache["token_window_sha256"]
        case["token_window_lines"] = cache["token_window_lines"]
        case["token_window_range"] = cache["token_window_range"]
    return case


def fixture_hashes() -> list[dict[str, str]]:
    records = []
    for path in FIXTURE_HASH_PATHS:
        role = "m5-precondition" if path in M5_PRECONDITION_PATHS else "replay-fixture"
        records.append({"path": path, "sha256": sha256_file(ROOT / path), "role": role})
    return records


def build_current_c_oracle() -> dict[str, Any]:
    kv_rows = parse_kv_rows()
    m05_blocks = {
        "server-a": parse_requests(M05 / "traces" / "server-a.trace")[1],
        "server-b": parse_requests(M05 / "traces" / "server-b.trace")[1],
        "server-c": parse_requests(M05 / "traces" / "server-c.trace")[1],
    }
    m04_requests = parse_requests(M04 / "traces" / "server.trace")
    m04_seed_prompt = rendered_prompt(m04_requests[5])

    replay_cases = [
        build_replay_case(
            "m0_5_seed_miss",
            "m0.5",
            rel(M05_FIXTURES / "kv_seed.json"),
            M05 / "responses" / "seed_miss.json",
            m05_blocks["server-a"],
            kv_rows,
        ),
        build_replay_case(
            "m0_5_seed_restore",
            "m0.5",
            rel(M05_FIXTURES / "kv_seed.json"),
            M05 / "responses" / "seed_restore.json",
            m05_blocks["server-b"],
            kv_rows,
        ),
        build_replay_case(
            "m0_5_continuation_restore",
            "m0.5",
            rel(M05_FIXTURES / "kv_continuation.json"),
            M05 / "responses" / "continuation_restore.json",
            m05_blocks["server-c"],
            kv_rows,
        ),
        build_replay_case(
            "m0_4_tool_call",
            "m0.4",
            "server-trace-request-3",
            M04 / "responses" / "chat_tool_call.json",
            m04_requests[3],
            kv_rows,
        ),
        build_replay_case(
            "m0_4_cache_seed",
            "m0.4",
            rel(ROOT / "ds4-parity" / "baselines" / "server-fixtures" / "m0.4" / "chat_cache_seed.json"),
            M04 / "responses" / "chat_cache_seed.json",
            m04_requests[5],
            kv_rows,
        ),
        build_replay_case(
            "m0_4_cache_continuation",
            "m0.4",
            rel(ROOT / "ds4-parity" / "baselines" / "server-fixtures" / "m0.4" / "chat_cache_continuation.json"),
            M04 / "responses" / "chat_cache_continuation.json",
            m04_requests[6],
            kv_rows,
            prefix_prompt=m04_seed_prompt,
        ),
    ]
    tool_response = load_json(M04 / "responses" / "chat_tool_call.json")
    tool_call = tool_response["choices"][0]["message"]["tool_calls"][0]
    args = tool_call["function"]["arguments"].encode()
    tool_block = parsed_message_fields(m04_requests[3])
    return {
        "schema": "ds4.kv_replay_oracle.v1",
        "source": "current-c-log-artifacts",
        "fixture_hashes": fixture_hashes(),
        "replay_cases": replay_cases,
        "dsml_tool_call_cases": [
            {
                "name": "m0_4_tool_call",
                "dsml_start": tool_block.get("dsml_start"),
                "dsml_end": tool_block.get("dsml_end"),
                "tool_call_count": 1,
                "tool_call_id": tool_call["id"],
                "tool_call_name": tool_call["function"]["name"],
                "tool_call_arguments_sha256": sha256_bytes(args),
                "tool_call_arguments_bytes": len(args),
            }
        ],
    }


def run_rust_replay_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-kv-replay-dump-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_rust_policy_dump() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ds4-gguf", "--bin", "ds4-kv-policy-dump-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def compare_value(report: Report, expected: Any, got: Any, path: str) -> None:
    if isinstance(expected, dict):
        got_dict = require_dict(report, got, path)
        report.check(list(expected) == list(got_dict), f"{path}: key order or coverage drift")
        for key, expected_value in expected.items():
            if key in got_dict:
                compare_value(report, expected_value, got_dict[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        report.check(isinstance(got, list), f"{path}: expected array")
        got_list = got if isinstance(got, list) else []
        report.check(len(expected) == len(got_list), f"{path}: length drift")
        for idx, (expected_item, got_item) in enumerate(zip(expected, got_list)):
            compare_value(report, expected_item, got_item, f"{path}[{idx}]")
        return
    report.check(expected == got, f"{path}: {expected!r} != {got!r}")


def compare_current_c(baseline: dict[str, Any], current: dict[str, Any]) -> Report:
    report = Report()
    report.check(baseline.get("schema") == "ds4.kv_replay_oracle.v1", "baseline schema mismatch")
    report.check(current.get("schema") == "ds4.kv_replay_oracle.v1", "current schema mismatch")
    baseline_hashes = {item["path"]: item for item in baseline.get("fixture_hashes", [])}
    current_hashes = {item["path"]: item for item in current.get("fixture_hashes", [])}
    report.check(set(baseline_hashes) == set(current_hashes), "fixture hash coverage drift")
    for path, expected in baseline_hashes.items():
        got = current_hashes.get(path)
        if got is None:
            continue
        label = "M5 fixture precondition drift" if expected.get("role") == "m5-precondition" else "fixture precondition drift"
        report.check(expected.get("sha256") == got.get("sha256"), f"{label}: {path}")
        report.check(expected.get("role") == got.get("role"), f"fixture role drift: {path}")

    for key in ("replay_cases", "dsml_tool_call_cases"):
        compare_value(report, baseline.get(key), current.get(key), key)
    return report


def by_name(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    return {item.get("name"): item for item in items if isinstance(item, dict) and isinstance(item.get("name"), str)}


def compare_rust_replay(baseline: dict[str, Any], rust: dict[str, Any]) -> Report:
    report = Report()
    report.check(rust.get("schema") == "ds4.rust_kv_replay_oracle.v1", "Rust replay schema mismatch")
    report.check(rust.get("source") == "rust-kv-replay-no-model", "Rust replay source mismatch")
    expected_cases = by_name(baseline.get("replay_cases"))
    rust_cases = by_name(rust.get("replay_cases"))
    report.check(set(expected_cases) == set(rust_cases), "Rust replay case coverage drift")
    for name, got in rust_cases.items():
        expected = expected_cases.get(name, {})
        for field in RUST_REPLAY_FIELDS:
            report.check(field in got, f"Rust {name}.{field}: missing")
            report.check(got.get(field) == expected.get(field), f"Rust {name}.{field} drift")
        reason_name = got.get("disk_cache_reason_name")
        report.check(
            got.get("disk_cache_reason_code") == reason_name_to_code(reason_name),
            f"Rust {name}.reason code/name mismatch",
        )

    compare_value(
        report,
        baseline.get("dsml_tool_call_cases"),
        rust.get("dsml_tool_call_cases"),
        "dsml_tool_call_cases",
    )
    return report


def check_rust_policy_precondition(baseline: dict[str, Any], rust_policy: dict[str, Any]) -> Report:
    report = Report()
    report.check(rust_policy.get("schema") == "ds4.rust_kv_policy_oracle.v1", "Rust policy schema mismatch")
    fixture = require_dict(report, rust_policy.get("m0_5_header_fixture"), "m0_5_header_fixture")
    rows = by_name([
        {
            **row,
            "name": row.get("file"),
        }
        for row in fixture.get("expected_rows", [])
        if isinstance(row, dict)
    ])
    disk_cases = [
        case for case in baseline.get("replay_cases", []) if case.get("disk_cache_file") is not None
    ]
    for case in disk_cases:
        name = case["disk_cache_file"]
        row = rows.get(name)
        report.check(row is not None, f"Rust policy M0.5 row missing: {name}")
        if row is None:
            continue
        for field, case_key in (
            ("tokens", "disk_cached_tokens"),
            ("reason_name", "disk_cache_reason_name"),
            ("reason", "disk_cache_reason_code"),
            ("ext_flags", "disk_cache_ext_flags"),
            ("rendered_text_bytes", "rendered_text_bytes"),
        ):
            report.check(row.get(field) == case.get(case_key), f"Rust policy {name}.{field} drift")
    return report


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def build_manifest(artifact: Path) -> dict[str, Any]:
    return {
        "schema": "ds4.kv_replay_manifest.v1",
        "artifact": {
            "path": "current-c.json",
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "oracle": "current C M0.4/M0.5 replay traces with M5 prompt fixture preconditions",
        "validation": [
            "python3 -m py_compile ds4-parity/compare_kv_replay.py",
            "python3 ds4-parity/compare_kv_replay.py --negative-test",
            "cargo test -p ds4-gguf kv_policy",
            "cargo test -p ds4-gguf --bin ds4-kv-replay-dump-rs",
            "cargo test --workspace",
            "git diff --check",
        ],
    }


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    manifest = load_json(path)
    root = require_dict(report, manifest, "manifest")
    report.check(root.get("schema") == "ds4.kv_replay_manifest.v1", "manifest schema mismatch")
    artifact_info = require_dict(report, root.get("artifact"), "manifest.artifact")
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    report.check(artifact_info.get("size_bytes") == artifact.stat().st_size, "manifest artifact size drift")
    return report


def run_negative_tests(baseline: dict[str, Any], current: dict[str, Any], rust: dict[str, Any]) -> Report:
    report = Report()

    def expect_rust_failure(name: str, path: list[str | int], value: Any) -> None:
        bad = copy.deepcopy(rust)
        target: Any = bad
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        sub = compare_rust_replay(baseline, bad)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_rust_failure("cache source drift", ["replay_cases", 1, "cache_source"], "none")
    expect_rust_failure("cached token drift", ["replay_cases", 2, "cached_tokens"], 550)
    expect_rust_failure("suffix byte drift", ["replay_cases", 5, "effective_suffix_hex"], "00")
    expect_rust_failure("DSML argument drift", ["dsml_tool_call_cases", 0, "tool_call_arguments_sha256"], "00")

    bad_baseline = copy.deepcopy(baseline)
    for item in bad_baseline["fixture_hashes"]:
        if item["role"] == "m5-precondition":
            item["sha256"] = "0" * 64
            break
    sub = compare_current_c(bad_baseline, current)
    report.check(not sub.ok, "negative test did not fail: M5 precondition drift")

    bad_baseline = copy.deepcopy(baseline)
    bad_baseline["replay_cases"][0]["cache_write_tokens"] = 1
    sub = compare_current_c(bad_baseline, current)
    report.check(not sub.ok, "negative test did not fail: C replay drift")
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rust-dump", type=Path)
    parser.add_argument("--rust-policy-dump", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--write-rust-dump", type=Path)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    current = build_current_c_oracle()
    if args.write_baseline:
        write_json(args.write_baseline, current)
        manifest_path = args.write_manifest
        if manifest_path is not None:
            write_json(manifest_path, build_manifest(args.write_baseline))
        return 0

    baseline = load_json(args.baseline)
    current_report = compare_current_c(baseline, current)
    print_report("KV replay C fixture preconditions", current_report)
    ok = current_report.ok

    if args.rust_dump:
        rust = load_json(args.rust_dump)
    else:
        code, stdout, stderr = run_rust_replay_dump()
        if code != 0:
            print("rust KV replay dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        rust = json.loads(stdout)
        if args.write_rust_dump:
            args.write_rust_dump.write_text(stdout)

    rust_report = compare_rust_replay(baseline, rust)
    print_report("KV replay C/Rust comparator", rust_report)
    ok = ok and rust_report.ok

    if args.rust_policy_dump:
        rust_policy = load_json(args.rust_policy_dump)
    else:
        code, stdout, stderr = run_rust_policy_dump()
        if code != 0:
            print("rust KV policy precondition dump: FAIL")
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            return 1
        rust_policy = json.loads(stdout)
    policy_report = check_rust_policy_precondition(baseline, rust_policy)
    print_report("KV replay Rust policy precondition", policy_report)
    ok = ok and policy_report.ok

    manifest_path = args.manifest
    if manifest_path is None and args.baseline.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.baseline)
        print_report("KV replay manifest", manifest_report)
        ok = ok and manifest_report.ok

    if args.negative_test:
        negative_report = run_negative_tests(baseline, current, rust)
        print_report("KV replay negative tests", negative_report)
        ok = ok and negative_report.ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
