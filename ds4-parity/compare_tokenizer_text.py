#!/usr/bin/env python3
"""Compare Rust tokenizer output against the M5.2 current-C oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_M52 = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.2" / "current-c.json"
TOKENIZER_GGUF = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.3" / "tokenizer.gguf"
MANIFEST = ROOT / "ds4-parity" / "baselines" / "tokenization" / "m5.3" / "manifest.json"
GGUF_MAGIC = 0x4655_4747
GGUF_STRING = 8
GGUF_ARRAY = 9
TOKEN_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")


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


def token_ids_sha(tokens: list[dict[str, Any]]) -> str:
    blob = bytearray()
    for token in tokens:
        blob.extend(f"{token['id']}\n".encode())
    return hashlib.sha256(blob).hexdigest()


def token_pieces_sha(tokens: list[dict[str, Any]]) -> str:
    blob = bytearray()
    for token in tokens:
        piece = bytes(token["bytes"])
        blob.extend(f"{token['id']}:{len(piece)}:".encode())
        blob.extend(piece)
        blob.extend(b"\n")
    return hashlib.sha256(blob).hexdigest()


def run_rust_dump(tokenizer_path: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ds4-gguf",
            "--bin",
            "ds4-tokenizer-dump",
            "--",
            str(tokenizer_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def compare_dumps(baseline: dict[str, Any], rust: dict[str, Any]) -> Report:
    report = Report()
    report.check(rust.get("schema") == "ds4.rust_tokenizer.v1", "rust schema mismatch")
    base_tok = baseline["tokenizer"]
    rust_tok = rust.get("tokenizer", {})
    for key in ("token_count", "token_bytes_sha256", "merge_count", "merge_pairs_sha256"):
        report.check(rust_tok.get(key) == base_tok.get(key), f"tokenizer.{key} drift")

    base_special = {entry["name"]: entry for entry in base_tok["special_token_at"]}
    rust_special = {entry.get("name"): entry for entry in rust_tok.get("special_token_at", [])}
    report.check(set(rust_special) == set(base_special), "special token names drift")
    for name, expected in base_special.items():
        got = rust_special.get(name, {})
        report.check(got.get("id") == expected["id"], f"special token {name} id drift")
        report.check(got.get("text") == expected["text"], f"special token {name} text drift")

    base_cases = {
        case["name"]: case
        for case in baseline["text_cases"]
        if case.get("mode") == "plain_text"
    }
    rust_cases = {case.get("name"): case for case in rust.get("text_cases", [])}
    report.check(set(rust_cases) == set(base_cases), "plain text case names drift")
    for name, expected in base_cases.items():
        got = rust_cases.get(name, {})
        report.check(got.get("input") == expected["input"], f"{name}.input drift")
        report.check(got.get("token_count") == expected["token_count"], f"{name}.token_count drift")
        got_tokens = got.get("tokens", [])
        expected_tokens = expected["tokens"]
        report.check(
            [token.get("id") for token in got_tokens] == [token["id"] for token in expected_tokens],
            f"{name}.token ids drift",
        )
        report.check(
            [token.get("bytes") for token in got_tokens] == [token["bytes"] for token in expected_tokens],
            f"{name}.token bytes drift",
        )
        if isinstance(got_tokens, list):
            report.check(token_ids_sha(got_tokens) == expected["token_ids_sha256"], f"{name}.token_ids_sha256 drift")
            report.check(
                token_pieces_sha(got_tokens) == expected["token_pieces_sha256"],
                f"{name}.token_pieces_sha256 drift",
            )
    return report


def check_manifest(manifest_path: Path, tokenizer_path: Path) -> Report:
    report = Report()
    manifest = load_json(manifest_path)
    report.check(manifest.get("schema") == "ds4.tokenizer_baseline.v1", "manifest schema mismatch")
    fixture = manifest.get("tokenizer_fixture", {})
    report.check(fixture.get("path") == "tokenizer.gguf", "manifest fixture path mismatch")
    report.check(fixture.get("sha256") == sha256_file(tokenizer_path), "manifest fixture sha256 drift")
    report.check(fixture.get("size_bytes") == tokenizer_path.stat().st_size, "manifest fixture size drift")
    return report


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, pos)[0], pos + 4


def read_u64(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, pos)[0], pos + 8


def read_string(data: bytes, pos: int) -> tuple[bytes, int]:
    n, pos = read_u64(data, pos)
    end = pos + n
    if end > len(data):
        raise ValueError("truncated string")
    return data[pos:end], end


def parse_tokenizer_fixture(path: Path) -> dict[str, list[bytes]]:
    data = path.read_bytes()
    pos = 0
    magic, pos = read_u32(data, pos)
    if magic != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version, pos = read_u32(data, pos)
    if version != 3:
        raise ValueError("unsupported GGUF version")
    tensor_count, pos = read_u64(data, pos)
    if tensor_count != 0:
        raise ValueError("expected tokenizer-only fixture")
    metadata_count, pos = read_u64(data, pos)
    out: dict[str, list[bytes]] = {}
    for _ in range(metadata_count):
        key, pos = read_string(data, pos)
        type_id, pos = read_u32(data, pos)
        if type_id != GGUF_ARRAY:
            raise ValueError("expected array metadata")
        element_type, pos = read_u32(data, pos)
        if element_type != GGUF_STRING:
            raise ValueError("expected string array")
        n, pos = read_u64(data, pos)
        values = []
        for _ in range(n):
            value, pos = read_string(data, pos)
            values.append(value)
        out[key.decode("utf-8")] = values
    return out


def write_string(out: bytearray, value: bytes) -> None:
    out.extend(struct.pack("<Q", len(value)))
    out.extend(value)


def write_tokenizer_fixture(path: Path, arrays: dict[str, list[bytes]]) -> None:
    out = bytearray()
    out.extend(struct.pack("<I", GGUF_MAGIC))
    out.extend(struct.pack("<I", 3))
    out.extend(struct.pack("<Q", 0))
    out.extend(struct.pack("<Q", len(arrays)))
    for key in TOKEN_KEYS:
        if key not in arrays:
            continue
        write_string(out, key.encode("utf-8"))
        out.extend(struct.pack("<I", GGUF_ARRAY))
        out.extend(struct.pack("<I", GGUF_STRING))
        out.extend(struct.pack("<Q", len(arrays[key])))
        for value in arrays[key]:
            write_string(out, value)
    path.write_bytes(out)


def run_negative(tokenizer_path: Path, baseline: dict[str, Any]) -> Report:
    report = Report()
    original = parse_tokenizer_fixture(tokenizer_path)

    def expect_failure(label: str, mutate: Any) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-tokenizer-negative-") as tmp:
            arrays = copy.deepcopy(original)
            mutate(arrays)
            bad_path = Path(tmp) / f"{label}.gguf"
            write_tokenizer_fixture(bad_path, arrays)
            rc, stdout, _stderr = run_rust_dump(bad_path)
            if rc != 0:
                report.check(True, f"negative {label} rejected")
                return
            try:
                rust = json.loads(stdout)
            except json.JSONDecodeError:
                report.check(True, f"negative {label} produced invalid output")
                return
            report.check(not compare_dumps(baseline, rust).ok, f"negative {label} drift was not detected")

    expect_failure("missing-token-table", lambda obj: obj.pop("tokenizer.ggml.tokens"))
    expect_failure("missing-merges", lambda obj: obj.pop("tokenizer.ggml.merges"))
    expect_failure("token-bytes-hash", lambda obj: obj["tokenizer.ggml.tokens"].__setitem__(2, b"changed"))
    expect_failure("merge-hash", lambda obj: obj["tokenizer.ggml.merges"].__setitem__(0, b"changed merge"))
    expect_failure("missing-special", lambda obj: obj["tokenizer.ggml.tokens"].__setitem__(0, b"not-bos"))
    expect_failure("invalid-utf8-token", lambda obj: obj["tokenizer.ggml.tokens"].__setitem__(2, b"\xff"))
    expect_failure(
        "merge-rank",
        lambda obj: obj["tokenizer.ggml.merges"].__setitem__(
            1,
            obj["tokenizer.ggml.merges"][0],
        ),
    )
    return report


def print_report(label: str, report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"{label}: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_GGUF)
    parser.add_argument("--baseline", type=Path, default=BASELINE_M52)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    rc, stdout, stderr = run_rust_dump(args.tokenizer)
    if rc != 0:
        sys.stderr.write(stderr)
        return rc
    rust = json.loads(stdout)
    compare = compare_dumps(baseline, rust)
    print_report("tokenizer text comparison", compare)
    if not compare.ok:
        return 1

    manifest_path = args.manifest
    if manifest_path is None and args.tokenizer == TOKENIZER_GGUF and MANIFEST.exists():
        manifest_path = MANIFEST
    if manifest_path is not None:
        manifest = check_manifest(manifest_path, args.tokenizer)
        print_report("tokenizer manifest", manifest)
        if not manifest.ok:
            return 1

    if args.negative_test:
        negative = run_negative(args.tokenizer, baseline)
        print_report("tokenizer negative tests", negative)
        if not negative.ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
