#!/usr/bin/env python3
"""Validate the M7.2 current-C KV header and policy oracle dump."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.2" / "current-c.json"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "kv" / "m7.2" / "manifest.json"
M05_HEADER_TSV = ROOT / "ds4-parity" / "baselines" / "kv-artifacts" / "m0.5" / "logs" / "kv-header.tsv"

DEFAULT_OPTIONS = {
    "min_tokens": 512,
    "cold_max_tokens": 30000,
    "continued_interval_tokens": 10000,
    "boundary_trim_tokens": 32,
    "boundary_align_tokens": 2048,
}

EXPECTED_REASONS = {
    None: 0,
    "cold": 1,
    "continued": 2,
    "evict": 3,
    "shutdown": 4,
    "agent-system": 5,
    "agent-session": 6,
    "other": 0,
}

EXPECTED_KEY_KINDS = {
    0: "token-text",
    1: "token-text",
    2: "responses-visible",
    4: "thinking-visible",
    3: "responses-visible",
    6: "responses-visible",
}

EXPECTED_STORE_LEN = {
    "below_min": 500,
    "at_min_plus_trim": 544,
    "aligned_after_trim": 2048,
    "larger_aligned_after_trim": 4096,
    "align_zero_uses_trimmed_stable": 968,
}

EXPECTED_CHAT_ANCHOR = {
    "last_user_before_assistant": 3,
    "user_below_min": -1,
    "missing_markers": -1,
    "assistant_first": -1,
    "multiple_users_before_assistant": 4,
    "exact_min_boundary": 2,
    "same_user_and_assistant_id": -1,
}

EXPECTED_CONTINUED = {
    "below_min": 0,
    "unaligned_interval": 0,
    "aligned_interval": 10240,
    "already_stored": 0,
    "disabled_store": 0,
    "align_zero_interval": 10000,
    "no_interval": 0,
}

EXPECTED_FILE_SIZE = {
    "no_budget": (True, 382, 0),
    "under_budget_with_slack": (True, 382, 386),
    "over_budget_with_slack": (False, 382, 386),
    "overflow_text": (False, 0, 0),
}

EXPECTED_PREFIX = {
    "matching_prefix": True,
    "empty_prefix": True,
    "mismatch": False,
    "prefix_longer_than_text": False,
}

EXPECTED_EVICTION = {
    "fresh_hits": 4.0,
    "one_half_life": 2.5,
    "stale_hits_floor": 1.0,
    "zero_timestamp": 1.0,
    "zero_file_size": 0.0,
}

EXPECTED_FIND = {
    "longest_prefix": (
        True,
        "6e02b13abe89b118b6732c1e2011644573b89ec4",
        700,
        16,
        2,
        32768,
    ),
    "reject_quant": (False, None, None, None, None, None),
    "allow_cross_quant_when_config_accepts": (
        True,
        "6e02b13abe89b118b6732c1e2011644573b89ec4",
        700,
        16,
        2,
        32768,
    ),
    "reject_ctx_too_small": (False, None, None, None, None, None),
    "reject_below_min": (False, None, None, None, None, None),
    "longest_with_large_context": (
        True,
        "204345290f64d38104fc16e1c4ac1bedc2b7cf91",
        800,
        22,
        4,
        65536,
    ),
}

M05_FIELDS = [
    "file",
    "quant",
    "reason",
    "reason_name",
    "ext_flags",
    "tokens",
    "hits",
    "ctx",
    "payload_bytes",
    "rendered_text_bytes",
    "trailer_bytes",
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


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_dict(report: Report, value: Any, path: str) -> dict[str, Any]:
    report.check(isinstance(value, dict), f"{path}: expected object")
    return value if isinstance(value, dict) else {}


def require_list(report: Report, value: Any, path: str) -> list[Any]:
    report.check(isinstance(value, list), f"{path}: expected array")
    return value if isinstance(value, list) else []


def require_int(report: Report, value: Any, path: str) -> int | None:
    report.check(isinstance(value, int), f"{path}: expected int")
    return value if isinstance(value, int) else None


def check_hex(report: Report, value: Any, path: str, expected_len: int | None = None) -> bytes:
    report.check(isinstance(value, str), f"{path}: expected hex string")
    if not isinstance(value, str):
        return b""
    report.check(len(value) % 2 == 0, f"{path}: odd hex length")
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        report.check(False, f"{path}: invalid hex")
        return b""
    if expected_len is not None:
        report.check(len(raw) == expected_len, f"{path}: byte length drift")
    return raw


def build_header(inp: dict[str, Any]) -> bytes:
    h = bytearray(48)
    h[0:4] = b"KVC\x01"
    h[4] = int(inp["quant_bits"]) & 0xFF
    h[5] = int(inp["reason"]) & 0xFF
    h[6] = int(inp["ext_flags"]) & 0xFF
    h[8:12] = struct.pack("<I", int(inp["tokens"]))
    h[12:16] = struct.pack("<I", int(inp["hits"]))
    h[16:20] = struct.pack("<I", int(inp["ctx_size"]))
    h[24:32] = struct.pack("<Q", int(inp["created_at"]))
    h[32:40] = struct.pack("<Q", int(inp["last_used"]))
    h[40:48] = struct.pack("<Q", int(inp["payload_bytes"]))
    return bytes(h)


def by_name(report: Report, items: Any, path: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for idx, raw in enumerate(require_list(report, items, path)):
        obj = require_dict(report, raw, f"{path}[{idx}]")
        name = obj.get("name")
        report.check(isinstance(name, str) and bool(name), f"{path}[{idx}].name invalid")
        if isinstance(name, str):
            report.check(name not in result, f"{path}: duplicate case {name}")
            result[name] = obj
    return result


def check_constants(report: Report, root: dict[str, Any]) -> None:
    constants = require_dict(report, root.get("constants"), "constants")
    report.check(constants.get("fixed_header") == 48, "constants.fixed_header drift")
    report.check(constants.get("default_mb") == 4096, "constants.default_mb drift")
    report.check(constants.get("hit_half_life_seconds") == 21600, "constants.hit_half_life_seconds drift")
    flags = require_dict(report, constants.get("ext_flags"), "constants.ext_flags")
    report.check(flags.get("tool_map") == 1, "ext flag tool_map drift")
    report.check(flags.get("responses_visible") == 2, "ext flag responses_visible drift")
    report.check(flags.get("thinking_visible") == 4, "ext flag thinking_visible drift")
    report.check(require_dict(report, root.get("defaults"), "defaults") == DEFAULT_OPTIONS, "defaults drift")


def check_reason_and_key_kind(report: Report, root: dict[str, Any]) -> None:
    reasons: dict[Any, Any] = {}
    for idx, raw in enumerate(require_list(report, root.get("reason_codes"), "reason_codes")):
        obj = require_dict(report, raw, f"reason_codes[{idx}]")
        reasons[obj.get("input")] = obj.get("code")
    report.check(reasons == EXPECTED_REASONS, "reason code mapping drift")

    key_kinds = {}
    for idx, raw in enumerate(require_list(report, root.get("key_kind_cases"), "key_kind_cases")):
        obj = require_dict(report, raw, f"key_kind_cases[{idx}]")
        flags = require_int(report, obj.get("ext_flags"), f"key_kind_cases[{idx}].ext_flags")
        if flags is not None:
            key_kinds[flags] = obj.get("key_kind")
    report.check(key_kinds == EXPECTED_KEY_KINDS, "key kind mapping drift")


def check_little_endian(report: Report, root: dict[str, Any]) -> None:
    for idx, raw in enumerate(require_list(report, root.get("little_endian_cases"), "little_endian_cases")):
        obj = require_dict(report, raw, f"little_endian_cases[{idx}]")
        value = require_int(report, obj.get("value"), f"little_endian_cases[{idx}].value")
        raw_hex = check_hex(report, obj.get("hex"), f"little_endian_cases[{idx}].hex", 4)
        if value is not None and raw_hex:
            report.check(raw_hex == struct.pack("<I", value & 0xFFFFFFFF), f"little_endian_cases[{idx}].hex drift")
            report.check(obj.get("roundtrip") == value, f"little_endian_cases[{idx}].roundtrip drift")


def check_sha_and_paths(report: Report, root: dict[str, Any]) -> None:
    for idx, raw in enumerate(require_list(report, root.get("sha_cases"), "sha_cases")):
        obj = require_dict(report, raw, f"sha_cases[{idx}]")
        text = check_hex(report, obj.get("text_hex"), f"sha_cases[{idx}].text_hex")
        report.check(obj.get("sha1") == hashlib.sha1(text).hexdigest(), f"sha_cases[{idx}].sha1 drift")

    name_cases = require_list(report, root.get("name_cases"), "name_cases")
    report.check(len(name_cases) == 3, "name case coverage drift")
    valid = require_dict(report, name_cases[0], "name_cases[0]")
    report.check(valid.get("valid") is True, "valid name rejected")
    report.check(valid.get("sha") == "abcdef0123456789abcdef0123456789abcdef01", "valid name sha lowercase drift")
    for idx in (1, 2):
        obj = require_dict(report, name_cases[idx], f"name_cases[{idx}]")
        report.check(obj.get("valid") is False, f"name_cases[{idx}] should be invalid")

    path_cases = require_list(report, root.get("path_cases"), "path_cases")
    report.check(len(path_cases) == 3, "path case coverage drift")
    report.check(require_dict(report, path_cases[0], "path_cases[0]").get("joined") == "cache/file.kv", "path join missing slash drift")
    report.check(require_dict(report, path_cases[1], "path_cases[1]").get("joined") == "cache/file.kv", "path join duplicate slash drift")
    sha_case = require_dict(report, path_cases[2], "path_cases[2]")
    sha = sha_case.get("sha")
    report.check(isinstance(sha, str) and sha_case.get("basename") == f"{sha}.kv", "path_for_sha basename drift")


def check_headers(report: Report, root: dict[str, Any]) -> None:
    cases = by_name(report, root.get("header_cases"), "header_cases")
    report.check(set(cases) == {"cold_token_text", "continued_tool_map", "unknown_reason_normalized"}, "header case coverage drift")
    for name, obj in cases.items():
        inp = require_dict(report, obj.get("input"), f"header_cases.{name}.input")
        header = check_hex(report, obj.get("header_hex"), f"header_cases.{name}.header_hex", 48)
        if inp and header:
            report.check(header == build_header(inp), f"header_cases.{name}.header_hex drift")
        tb = check_hex(report, obj.get("text_len_hex"), f"header_cases.{name}.text_len_hex", 4)
        text_bytes = inp.get("text_bytes")
        if isinstance(text_bytes, int) and tb:
            report.check(tb == struct.pack("<I", text_bytes), f"header_cases.{name}.text_len_hex drift")
        report.check(obj.get("read_ok") is True, f"header_cases.{name}.read_ok drift")
        decoded = require_dict(report, obj.get("decoded"), f"header_cases.{name}.decoded")
        expected_reason = inp.get("reason")
        if isinstance(expected_reason, int) and expected_reason > 6:
            expected_reason = 0
        for field in ("quant_bits", "ext_flags", "tokens", "hits", "ctx_size", "created_at", "last_used", "payload_bytes"):
            report.check(decoded.get(field) == inp.get(field), f"header_cases.{name}.{field} drift")
        report.check(decoded.get("reason") == expected_reason, f"header_cases.{name}.reason drift")
        report.check(decoded.get("text_bytes") == inp.get("text_bytes"), f"header_cases.{name}.text_bytes drift")

    invalid = by_name(report, root.get("invalid_header_cases"), "invalid_header_cases")
    report.check(set(invalid) == {"bad_magic", "bad_version", "zero_tokens", "bad_quant"}, "invalid header coverage drift")
    for name, obj in invalid.items():
        check_hex(report, obj.get("header_hex"), f"invalid_header_cases.{name}.header_hex", 48)
        report.check(obj.get("read_ok") is False, f"invalid_header_cases.{name} accepted")


def check_policy_cases(report: Report, root: dict[str, Any]) -> None:
    policy = require_dict(report, root.get("policy_cases"), "policy_cases")

    store_len = by_name(report, policy.get("store_len"), "policy_cases.store_len")
    report.check(set(store_len) == set(EXPECTED_STORE_LEN), "store_len coverage drift")
    for name, expected in EXPECTED_STORE_LEN.items():
        report.check(store_len.get(name, {}).get("store_len") == expected, f"store_len.{name} drift")

    chat = by_name(report, policy.get("chat_anchor"), "policy_cases.chat_anchor")
    report.check(set(chat) == set(EXPECTED_CHAT_ANCHOR), "chat_anchor coverage drift")
    for name, expected in EXPECTED_CHAT_ANCHOR.items():
        report.check(chat.get(name, {}).get("anchor_pos") == expected, f"chat_anchor.{name} drift")

    continued = by_name(report, policy.get("continued_store_target"), "policy_cases.continued_store_target")
    report.check(set(continued) == set(EXPECTED_CONTINUED), "continued_store_target coverage drift")
    for name, expected in EXPECTED_CONTINUED.items():
        report.check(continued.get(name, {}).get("target") == expected, f"continued_store_target.{name} drift")

    sizes = by_name(report, policy.get("file_size_fits"), "policy_cases.file_size_fits")
    report.check(set(sizes) == set(EXPECTED_FILE_SIZE), "file_size_fits coverage drift")
    for name, expected in EXPECTED_FILE_SIZE.items():
        obj = sizes.get(name, {})
        fits, file_bytes, required = expected
        report.check(obj.get("fits") is fits, f"file_size_fits.{name}.fits drift")
        report.check(obj.get("file_bytes") == file_bytes, f"file_size_fits.{name}.file_bytes drift")
        report.check(obj.get("required_bytes") == required, f"file_size_fits.{name}.required_bytes drift")

    prefixes = by_name(report, policy.get("byte_prefix_match"), "policy_cases.byte_prefix_match")
    report.check(set(prefixes) == set(EXPECTED_PREFIX), "byte_prefix_match coverage drift")
    for name, expected in EXPECTED_PREFIX.items():
        report.check(prefixes.get(name, {}).get("matches") is expected, f"byte_prefix_match.{name} drift")

    evictions = by_name(report, policy.get("eviction_score"), "policy_cases.eviction_score")
    report.check(set(EXPECTED_EVICTION) <= set(evictions), "eviction_score coverage drift")
    for name, expected in EXPECTED_EVICTION.items():
        score = evictions.get(name, {}).get("score")
        report.check(isinstance(score, (int, float)), f"eviction_score.{name}.score invalid")
        if isinstance(score, (int, float)):
            report.check(abs(float(score) - expected) <= 1e-9, f"eviction_score.{name}.score drift")
    protected = evictions.get("protected_sha", {})
    score = protected.get("score")
    protected_ok = isinstance(score, (int, float)) and math.isclose(
        float(score),
        1.7976931348623157e308,
        rel_tol=1e-15,
    )
    report.check(protected_ok, "eviction_score.protected_sha drift")

    find = by_name(report, policy.get("find_text_prefix"), "policy_cases.find_text_prefix")
    report.check(set(find) == set(EXPECTED_FIND), "find_text_prefix coverage drift")
    for name, expected in EXPECTED_FIND.items():
        found, sha, tokens, text_bytes, quant, ctx = expected
        obj = find.get(name, {})
        report.check(obj.get("found") is found, f"find_text_prefix.{name}.found drift")
        if found:
            report.check(obj.get("selected_sha") == sha, f"find_text_prefix.{name}.sha drift")
            report.check(obj.get("selected_tokens") == tokens, f"find_text_prefix.{name}.tokens drift")
            report.check(obj.get("selected_text_bytes") == text_bytes, f"find_text_prefix.{name}.text_bytes drift")
            report.check(obj.get("selected_quant_bits") == quant, f"find_text_prefix.{name}.quant drift")
            report.check(obj.get("selected_ctx_size") == ctx, f"find_text_prefix.{name}.ctx drift")


def parse_m05_rows() -> dict[str, dict[str, Any]]:
    with M05_HEADER_TSV.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized: dict[str, Any] = {}
        for field in M05_FIELDS:
            value = row[field]
            if field in {"file", "reason_name"}:
                normalized[field] = value
            else:
                normalized[field] = int(value)
        result[normalized["file"]] = normalized
    return result


def check_m05_fixture(report: Report, root: dict[str, Any]) -> None:
    fixture = require_dict(report, root.get("m0_5_header_fixture"), "m0_5_header_fixture")
    report.check(fixture.get("path") == "ds4-parity/baselines/kv-artifacts/m0.5/logs/kv-header.tsv", "m0_5 fixture path drift")
    expected_rows = require_list(report, fixture.get("expected_rows"), "m0_5_header_fixture.expected_rows")
    actual = parse_m05_rows()
    report.check(len(expected_rows) == 3, "m0_5 expected row count drift")
    report.check(len(actual) == 3, "m0_5 tsv row count drift")
    for idx, raw in enumerate(expected_rows):
        row = require_dict(report, raw, f"m0_5_header_fixture.expected_rows[{idx}]")
        filename = row.get("file")
        report.check(isinstance(filename, str), f"m0_5 row {idx} filename invalid")
        if not isinstance(filename, str):
            continue
        actual_row = actual.get(filename)
        report.check(actual_row is not None, f"m0_5 row {filename} missing from tsv")
        if actual_row is None:
            continue
        for field in M05_FIELDS:
            report.check(row.get(field) == actual_row.get(field), f"m0_5 {filename}.{field} drift")


def check_dump(obj: Any) -> Report:
    report = Report()
    root = require_dict(report, obj, "root")
    report.check(root.get("schema") == "ds4.kv_policy_oracle.v1", "schema mismatch")
    report.check(root.get("source") == "current-c-kvstore-no-model", "source mismatch")
    report.check(root.get("model") == "no model is loaded for this oracle", "model note drift")
    check_constants(report, root)
    check_reason_and_key_kind(report, root)
    check_little_endian(report, root)
    check_sha_and_paths(report, root)
    check_headers(report, root)
    check_policy_cases(report, root)
    check_m05_fixture(report, root)
    return report


def check_manifest(path: Path, artifact: Path) -> Report:
    report = Report()
    manifest = load_json(path)
    root = require_dict(report, manifest, "manifest")
    report.check(root.get("schema") == "ds4.kv_policy_manifest.v1", "manifest schema mismatch")
    artifact_info = require_dict(report, root.get("artifact"), "manifest.artifact")
    report.check(artifact_info.get("path") == "current-c.json", "manifest artifact path drift")
    report.check(artifact_info.get("sha256") == sha256_file(artifact), "manifest artifact sha drift")
    report.check(artifact_info.get("size_bytes") == artifact.stat().st_size, "manifest artifact size drift")
    report.check(root.get("oracle") == "current C ds4_kvstore no-model header and policy helpers", "manifest oracle drift")
    validations = require_list(report, root.get("validation"), "manifest.validation")
    for required in (
        "arch -arm64 make ds4-kv-policy-dump",
        "./ds4-kv-policy-dump ds4-parity/baselines/kv/m7.2/current-c.json",
        "python3 ds4-parity/check_kv_policy_dump.py --negative-test",
    ):
        report.check(required in validations, f"manifest validation missing {required}")
    return report


def run_negative_tests(obj: Any) -> Report:
    report = Report()

    def expect_failure(name: str, mutator: Any) -> None:
        candidate = copy.deepcopy(obj)
        mutator(candidate)
        sub = check_dump(candidate)
        report.check(not sub.ok, f"negative test did not fail: {name}")

    expect_failure(
        "default min token drift",
        lambda c: c["defaults"].__setitem__("min_tokens", 513),
    )
    expect_failure(
        "header byte drift",
        lambda c: c["header_cases"][0].__setitem__(
            "header_hex",
            "00" + c["header_cases"][0]["header_hex"][2:],
        ),
    )
    expect_failure(
        "invalid header accepted",
        lambda c: c["invalid_header_cases"][0].__setitem__("read_ok", True),
    )
    expect_failure(
        "store len drift",
        lambda c: c["policy_cases"]["store_len"][2].__setitem__("store_len", 4096),
    )
    expect_failure(
        "eviction score drift",
        lambda c: c["policy_cases"]["eviction_score"][1].__setitem__("score", 2.0),
    )
    expect_failure(
        "find prefix drift",
        lambda c: c["policy_cases"]["find_text_prefix"][0].__setitem__(
            "selected_tokens",
            600,
        ),
    )
    expect_failure("m0.5 header drift", lambda c: c["m0_5_header_fixture"]["expected_rows"][0].__setitem__("tokens", 551))
    return report


def print_report(label: str, report: Report) -> None:
    if report.ok:
        print(f"{label}: PASS, {report.checks} checks")
    else:
        print(f"{label}: FAIL, {len(report.errors)} errors / {report.checks} checks")
        for err in report.errors[:40]:
            print(f"  - {err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    obj = load_json(args.artifact)
    schema_report = check_dump(obj)
    print_report("kv policy oracle schema", schema_report)
    ok = schema_report.ok

    manifest_path = args.manifest
    if manifest_path is None and args.artifact.resolve() == BASELINE.resolve() and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest_report = check_manifest(manifest_path, args.artifact)
        print_report("kv policy manifest", manifest_report)
        ok = ok and manifest_report.ok

    if args.negative_test:
        negative_report = run_negative_tests(obj)
        print_report("kv policy negative tests", negative_report)
        ok = ok and negative_report.ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
