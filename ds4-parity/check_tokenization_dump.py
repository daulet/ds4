#!/usr/bin/env python3
"""Schema checks for current-C tokenization and prompt oracle dumps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.2"
BASELINE_C = BASELINE_DIR / "current-c.json"
MANIFEST = BASELINE_DIR / "manifest.json"
EXPECTED_MODEL_SHA256 = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
EXPECTED_MODEL_SIZE = 86720111488
EXPECTED_TOKEN_COUNT = 129280
EXPECTED_TOKEN_BYTES_SHA256 = "c92251fc634ff01cc6767d2e3ce1d368e72b5f02b647983d4410eb0c46693fa3"
EXPECTED_MERGE_COUNT = 127741
EXPECTED_MERGE_PAIRS_SHA256 = "8100a9693dc10b8aad79abbe20b172545ff5e1e6051e0705cc91e73b88e3751f"
EXPECTED_LITERAL_SPECIAL_COUNT = 863

EXPECTED_SPECIALS = {
    "bos": {"text": "<｜begin▁of▁sentence｜>", "id": 0},
    "eos": {"text": "<｜end▁of▁sentence｜>", "id": 1},
    "user": {"text": "<｜User｜>", "id": 128803},
    "assistant": {"text": "<｜Assistant｜>", "id": 128804},
    "think_start": {"text": "<think>", "id": 128821},
    "think_end": {"text": "</think>", "id": 128822},
    "dsml": {"text": "｜DSML｜", "id": 128825},
}

REQUIRED_SPECIALS = set(EXPECTED_SPECIALS)

EXPECTED_SERVER_SEMANTICS = {
    "m0.4/chat_basic": {"think_mode": "none", "has_tools": False, "prompt_preserves_reasoning": False},
    "m0.4/chat_stream": {"think_mode": "none", "has_tools": False, "prompt_preserves_reasoning": False},
    "m0.4/chat_tool_call": {"think_mode": "none", "has_tools": True, "prompt_preserves_reasoning": True},
    "m0.4/chat_thinking_disabled": {"think_mode": "none", "has_tools": False, "prompt_preserves_reasoning": False},
    "m0.4/chat_cache_seed": {"think_mode": "none", "has_tools": False, "prompt_preserves_reasoning": False},
    "m0.4/chat_cache_continuation": {
        "think_mode": "none",
        "has_tools": False,
        "prompt_preserves_reasoning": False,
    },
    "builtin_thinking_max_developer": {
        "think_mode": "max",
        "has_tools": False,
        "prompt_preserves_reasoning": False,
    },
    "builtin_function_result": {"think_mode": "high", "has_tools": False, "prompt_preserves_reasoning": True},
    "builtin_empty_tools_arrays": {"think_mode": "high", "has_tools": False, "prompt_preserves_reasoning": False},
}

REQUIRED_SERVER_CASES = {
    "m0.4/chat_basic",
    "m0.4/chat_stream",
    "m0.4/chat_tool_call",
    "m0.4/chat_thinking_disabled",
    "m0.4/chat_cache_seed",
    "m0.4/chat_cache_continuation",
    "builtin_thinking_max_developer",
    "builtin_function_result",
    "builtin_empty_tools_arrays",
}

REQUIRED_CLI_CASES = {
    "cli_basic_high",
    "cli_developer_max",
    "cli_tool_function_none",
}

EXPECTED_CLI_OPS = {
    "cli_basic_high": ["begin", "append_message", "append_message", "assistant_prefix"],
    "cli_developer_max": ["begin", "max_effort_prefix", "append_message", "append_message", "assistant_prefix"],
    "cli_tool_function_none": [
        "begin",
        "append_message",
        "append_message",
        "append_message",
        "append_message",
        "assistant_prefix",
    ],
}


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


def is_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def require_dict(report: Report, value: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{label} must be an object")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, label: str) -> list[Any]:
    report.check(isinstance(value, list), f"{label} must be an array")
    return value if isinstance(value, list) else []


def require_str(report: Report, value: Any, label: str) -> str:
    report.check(isinstance(value, str), f"{label} must be a string")
    return value if isinstance(value, str) else ""


def check_nonneg_int(report: Report, value: Any, label: str) -> None:
    report.check(isinstance(value, int) and value >= 0, f"{label} must be a nonnegative integer")


def check_token(report: Report, value: Any, label: str) -> None:
    token = require_dict(report, value, label)
    report.check(isinstance(token.get("id"), int) and token["id"] >= 0, f"{label}.id invalid")
    require_str(report, token.get("text"), f"{label}.text")
    raw = require_list(report, token.get("bytes"), f"{label}.bytes")
    for idx, byte in enumerate(raw):
        report.check(isinstance(byte, int) and 0 <= byte <= 255, f"{label}.bytes[{idx}] invalid")


def check_tokens(report: Report, case: dict[str, Any], label: str) -> None:
    tokens = require_list(report, case.get("tokens"), f"{label}.tokens")
    report.check(case.get("token_count") == len(tokens), f"{label}.token_count must match tokens length")
    report.check(len(tokens) > 0, f"{label}.tokens must not be empty")
    ids_blob = bytearray()
    pieces_blob = bytearray()
    for idx, token in enumerate(tokens):
        check_token(report, token, f"{label}.tokens[{idx}]")
        if isinstance(token, dict) and isinstance(token.get("id"), int):
            token_id = token["id"]
            ids_blob.extend(f"{token_id}\n".encode())
            raw = token.get("bytes")
            if isinstance(raw, list) and all(isinstance(b, int) and 0 <= b <= 255 for b in raw):
                piece = bytes(raw)
                pieces_blob.extend(f"{token_id}:{len(piece)}:".encode())
                pieces_blob.extend(piece)
                pieces_blob.extend(b"\n")
    report.check(case.get("token_ids_sha256") == sha256_bytes(bytes(ids_blob)), f"{label}.token_ids_sha256 drift")
    report.check(case.get("token_pieces_sha256") == sha256_bytes(bytes(pieces_blob)), f"{label}.token_pieces_sha256 drift")


def check_text_like_case(report: Report, case_value: Any, label: str, sha_key: str = "input_sha256") -> None:
    case = require_dict(report, case_value, label)
    require_str(report, case.get("name"), f"{label}.name")
    text = require_str(report, case.get("input"), f"{label}.input")
    report.check(case.get(sha_key) == sha256_bytes(text.encode()), f"{label}.{sha_key} drift")
    check_tokens(report, case, label)


def check_tokenizer(report: Report, root: dict[str, Any]) -> None:
    tok = require_dict(report, root.get("tokenizer"), "tokenizer")
    report.check(tok.get("token_count") == EXPECTED_TOKEN_COUNT, "tokenizer.token_count mismatch")
    report.check(tok.get("token_bytes_sha256") == EXPECTED_TOKEN_BYTES_SHA256, "tokenizer.token_bytes_sha256 mismatch")
    report.check(tok.get("merge_count") == EXPECTED_MERGE_COUNT, "tokenizer.merge_count mismatch")
    report.check(tok.get("merge_pairs_sha256") == EXPECTED_MERGE_PAIRS_SHA256, "tokenizer.merge_pairs_sha256 mismatch")
    report.check(isinstance(tok.get("canonical_hash_format"), str) and "u64le" in tok["canonical_hash_format"], "tokenizer canonical format missing")

    loaded = set(require_list(report, tok.get("loaded_metadata"), "tokenizer.loaded_metadata"))
    report.check({"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}.issubset(loaded), "tokenizer loaded metadata mismatch")

    specials = require_list(report, tok.get("special_token_at"), "tokenizer.special_token_at")
    names: set[str] = set()
    ids: set[int] = set()
    for idx, entry_value in enumerate(specials):
        entry = require_dict(report, entry_value, f"tokenizer.special_token_at[{idx}]")
        name = require_str(report, entry.get("name"), f"tokenizer.special_token_at[{idx}].name")
        text = require_str(report, entry.get("text"), f"tokenizer.special_token_at[{idx}].text")
        token_id = entry.get("id")
        report.check(isinstance(token_id, int) and token_id >= 0, f"tokenizer.special_token_at[{idx}].id invalid")
        names.add(name)
        if isinstance(token_id, int):
            ids.add(token_id)
        if name in EXPECTED_SPECIALS:
            expected = EXPECTED_SPECIALS[name]
            report.check(token_id == expected["id"], f"special_token_at {name} id mismatch")
            report.check(text == expected["text"], f"special_token_at {name} text mismatch")
    report.check(names == REQUIRED_SPECIALS, f"special_token_at names mismatch: {sorted(names)}")
    report.check(len(specials) == len(EXPECTED_SPECIALS), "special_token_at count mismatch")
    report.check(len(ids) == len(specials), "special_token_at IDs must be unique")

    literal = require_list(report, tok.get("literal_special_tokens"), "tokenizer.literal_special_tokens")
    report.check(len(literal) == EXPECTED_LITERAL_SPECIAL_COUNT, "literal_special_tokens count mismatch")
    literal_ids: set[int] = set()
    for idx, entry_value in enumerate(literal):
        entry = require_dict(report, entry_value, f"tokenizer.literal_special_tokens[{idx}]")
        token_id = entry.get("id")
        report.check(isinstance(token_id, int) and token_id >= 0, f"literal_special_tokens[{idx}].id invalid")
        text = require_str(report, entry.get("bytes"), f"literal_special_tokens[{idx}].bytes")
        report.check("｜" in text, f"literal_special_tokens[{idx}] missing fullwidth bar")
        if isinstance(token_id, int):
            report.check(token_id not in literal_ids, f"duplicate literal special id {token_id}")
            literal_ids.add(token_id)


def check_server_case(report: Report, case_value: Any, label: str) -> None:
    case = require_dict(report, case_value, label)
    name = require_str(report, case.get("name"), f"{label}.name")
    report.check(case.get("endpoint") == "/v1/chat/completions", f"{label}.endpoint mismatch")
    report.check(case.get("parse_ok") is True, f"{label}.parse_ok must be true")
    report.check(is_hex_sha256(case.get("request_body_sha256")), f"{label}.request_body_sha256 invalid")
    check_nonneg_int(report, case.get("request_body_bytes"), f"{label}.request_body_bytes")
    fixture = case.get("fixture")
    if isinstance(fixture, str):
        fixture_path = ROOT / fixture
        report.check(fixture_path.exists(), f"{label}.fixture missing: {fixture}")
        if fixture_path.exists():
            report.check(case.get("request_body_sha256") == sha256_file(fixture_path), f"{label}.request_body_sha256 drift")
            report.check(case.get("request_body_bytes") == fixture_path.stat().st_size, f"{label}.request_body_bytes drift")
    prompt = require_str(report, case.get("prompt_text"), f"{label}.prompt_text")
    report.check(case.get("prompt_sha256") == sha256_bytes(prompt.encode()), f"{label}.prompt_sha256 drift")
    require_str(report, case.get("think_mode"), f"{label}.think_mode")
    report.check(isinstance(case.get("has_tools"), bool), f"{label}.has_tools invalid")
    report.check(isinstance(case.get("prompt_preserves_reasoning"), bool), f"{label}.prompt_preserves_reasoning invalid")
    if name in EXPECTED_SERVER_SEMANTICS:
        expected = EXPECTED_SERVER_SEMANTICS[name]
        report.check(case.get("think_mode") == expected["think_mode"], f"{label}.think_mode drift")
        report.check(case.get("has_tools") == expected["has_tools"], f"{label}.has_tools drift")
        report.check(
            case.get("prompt_preserves_reasoning") == expected["prompt_preserves_reasoning"],
            f"{label}.prompt_preserves_reasoning drift",
        )
    check_tokens(report, case, label)
    report.check(name in REQUIRED_SERVER_CASES or name.startswith("extra_"), f"{label}.name not in required set")


def check_cli_case(report: Report, case_value: Any, label: str) -> None:
    case = require_dict(report, case_value, label)
    name = require_str(report, case.get("name"), f"{label}.name")
    ops = require_list(report, case.get("operations"), f"{label}.operations")
    report.check(len(ops) >= 2, f"{label}.operations too small")
    for idx, op_value in enumerate(ops):
        op = require_dict(report, op_value, f"{label}.operations[{idx}]")
        report.check(op.get("op") in {"begin", "max_effort_prefix", "append_message", "assistant_prefix"}, f"{label}.operations[{idx}].op invalid")
    if name in EXPECTED_CLI_OPS:
        report.check(
            [op.get("op") for op in ops if isinstance(op, dict)] == EXPECTED_CLI_OPS[name],
            f"{label}.operations sequence mismatch",
        )
    check_tokens(report, case, label)
    report.check(name in REQUIRED_CLI_CASES, f"{label}.name not in required set")


def check_dump(path: Path) -> Report:
    report = Report()
    root = require_dict(report, load_json(path), "root")
    report.check(root.get("schema") == "ds4.tokenization.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-server-token-oracle", "source mismatch")
    model = require_dict(report, root.get("model"), "model")
    require_str(report, model.get("path"), "model.path")
    report.check(model.get("size") == EXPECTED_MODEL_SIZE, "model.size mismatch")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "model.sha256 mismatch")

    check_tokenizer(report, root)

    text_cases = require_list(report, root.get("text_cases"), "text_cases")
    report.check(len(text_cases) >= 5, "text_cases coverage too small")
    for idx, case in enumerate(text_cases):
        check_text_like_case(report, case, f"text_cases[{idx}]")

    rendered = require_list(report, root.get("rendered_chat_cases"), "rendered_chat_cases")
    report.check(len(rendered) >= 2, "rendered_chat_cases coverage too small")
    for idx, case in enumerate(rendered):
        check_text_like_case(report, case, f"rendered_chat_cases[{idx}]")

    server_cases = require_list(report, root.get("server_request_cases"), "server_request_cases")
    server_names = {case.get("name") for case in server_cases if isinstance(case, dict)}
    for name in sorted(REQUIRED_SERVER_CASES):
        report.check(name in server_names, f"missing server request case {name}")
    for idx, case in enumerate(server_cases):
        check_server_case(report, case, f"server_request_cases[{idx}]")

    cli_cases = require_list(report, root.get("cli_chat_cases"), "cli_chat_cases")
    cli_names = {case.get("name") for case in cli_cases if isinstance(case, dict)}
    for name in sorted(REQUIRED_CLI_CASES):
        report.check(name in cli_names, f"missing CLI case {name}")
    for idx, case in enumerate(cli_cases):
        check_cli_case(report, case, f"cli_chat_cases[{idx}]")

    return report


def check_manifest(manifest_path: Path, dump_path: Path) -> Report:
    report = Report()
    manifest = require_dict(report, load_json(manifest_path), "manifest")
    report.check(manifest.get("schema") == "ds4.tokenization_baseline.v1", "manifest schema mismatch")
    model = require_dict(report, manifest.get("model"), "manifest.model")
    report.check(model.get("sha256") == EXPECTED_MODEL_SHA256, "manifest model sha256 mismatch")
    report.check(model.get("size_bytes") == EXPECTED_MODEL_SIZE, "manifest model size mismatch")
    current = require_dict(report, manifest.get("dumps", {}).get("current_c"), "manifest.dumps.current_c")
    report.check(current.get("path") == "current-c.json", "manifest current-c path mismatch")
    report.check(current.get("sha256") == sha256_file(dump_path), "manifest current-c sha256 drift")
    report.check(current.get("size_bytes") == dump_path.stat().st_size, "manifest current-c size drift")
    refresh = require_list(report, manifest.get("refresh_commands"), "manifest.refresh_commands")
    report.check(len(refresh) >= 2, "manifest refresh commands missing")
    return report


def run_negative(path: Path) -> Report:
    report = Report()
    original = load_json(path)

    def expect_failure(label: str, mutate: Any) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-token-negative-") as tmp:
            bad = copy.deepcopy(original)
            mutate(bad)
            bad_path = Path(tmp) / f"{label}.json"
            bad_path.write_text(json.dumps(bad, ensure_ascii=False))
            report.check(not check_dump(bad_path).ok, f"negative {label} drift was not detected")

    expect_failure("token-count", lambda obj: obj["tokenizer"].__setitem__("token_count", obj["tokenizer"]["token_count"] + 1))
    expect_failure("token-bytes-hash", lambda obj: obj["tokenizer"].__setitem__("token_bytes_sha256", "0" * 64))
    expect_failure("merge-hash", lambda obj: obj["tokenizer"].__setitem__("merge_pairs_sha256", "1" * 64))
    expect_failure("special-id", lambda obj: obj["tokenizer"]["special_token_at"][0].__setitem__("id", obj["tokenizer"]["special_token_at"][1]["id"]))
    expect_failure("literal-special", lambda obj: obj["tokenizer"]["literal_special_tokens"][0].__setitem__("bytes", "plain"))
    expect_failure("missing-server-case", lambda obj: obj.__setitem__("server_request_cases", obj["server_request_cases"][:-1]))
    expect_failure("server-think-mode", lambda obj: obj["server_request_cases"][6].__setitem__("think_mode", "none"))
    expect_failure("prompt-bytes", lambda obj: obj["server_request_cases"][0].__setitem__("prompt_text", obj["server_request_cases"][0]["prompt_text"] + "x"))
    expect_failure("request-body-hash", lambda obj: obj["server_request_cases"][0].__setitem__("request_body_sha256", "2" * 64))
    expect_failure("token-id", lambda obj: obj["text_cases"][0]["tokens"][0].__setitem__("id", obj["text_cases"][0]["tokens"][0]["id"] + 1))
    expect_failure("token-piece", lambda obj: obj["text_cases"][0]["tokens"][0]["bytes"].append(0))
    expect_failure("cli-operation", lambda obj: obj["cli_chat_cases"][0]["operations"][0].__setitem__("op", "append_message"))
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", nargs="?", type=Path, default=BASELINE_C)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    report = check_dump(args.dump)
    print_report("tokenization schema", report)
    if not report.ok:
        return 1

    manifest_path = args.manifest
    if manifest_path is None and args.dump == BASELINE_C and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest = check_manifest(manifest_path, args.dump)
        print_report("tokenization manifest", manifest)
        if not manifest.ok:
            return 1

    if args.negative_test:
        negative = run_negative(args.dump)
        print_report("tokenization negative tests", negative)
        if not negative.ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
