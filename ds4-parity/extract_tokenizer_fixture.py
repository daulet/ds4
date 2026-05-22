#!/usr/bin/env python3
"""Extract tokenizer token/merge metadata into a minimal GGUF fixture."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = 0x4655_4747
GGUF_STRING = 8
GGUF_ARRAY = 9
TOKEN_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")


def read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF file")
    return data


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_string(f: BinaryIO) -> bytes:
    n = read_u64(f)
    return read_exact(f, n)


def skip_value(f: BinaryIO, type_id: int) -> None:
    if type_id in (0, 1, 7):
        read_exact(f, 1)
    elif type_id in (2, 3):
        read_exact(f, 2)
    elif type_id in (4, 5, 6):
        read_exact(f, 4)
    elif type_id == GGUF_STRING:
        read_string(f)
    elif type_id == GGUF_ARRAY:
        element_type = read_u32(f)
        n = read_u64(f)
        for _ in range(n):
            skip_value(f, element_type)
    elif type_id in (10, 11, 12):
        read_exact(f, 8)
    else:
        raise ValueError(f"unknown GGUF metadata type {type_id}")


def read_string_array_payload(f: BinaryIO) -> tuple[bytes, int]:
    payload = bytearray()
    element_type = read_u32(f)
    payload += struct.pack("<I", element_type)
    n = read_u64(f)
    payload += struct.pack("<Q", n)
    if element_type != GGUF_STRING:
        raise ValueError(f"expected string array, got element type {element_type}")
    for _ in range(n):
        text = read_string(f)
        payload += struct.pack("<Q", len(text))
        payload += text
    return bytes(payload), n


def extract_arrays(path: Path) -> dict[str, tuple[bytes, int]]:
    out: dict[str, tuple[bytes, int]] = {}
    with path.open("rb") as f:
        magic = read_u32(f)
        if magic != GGUF_MAGIC:
            raise ValueError("model is not a GGUF file")
        version = read_u32(f)
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        _tensor_count = read_u64(f)
        metadata_count = read_u64(f)
        for _ in range(metadata_count):
            key = read_string(f).decode("utf-8")
            type_id = read_u32(f)
            if key in TOKEN_KEYS:
                if type_id != GGUF_ARRAY:
                    raise ValueError(f"{key} is not an array")
                out[key] = read_string_array_payload(f)
            else:
                skip_value(f, type_id)
    missing = [key for key in TOKEN_KEYS if key not in out]
    if missing:
        raise ValueError(f"missing tokenizer metadata: {', '.join(missing)}")
    return out


def write_string(out: bytearray, value: bytes) -> None:
    out += struct.pack("<Q", len(value))
    out += value


def write_fixture(path: Path, arrays: dict[str, tuple[bytes, int]]) -> None:
    out = bytearray()
    out += struct.pack("<I", GGUF_MAGIC)
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 0)
    out += struct.pack("<Q", len(TOKEN_KEYS))
    for key in TOKEN_KEYS:
        payload, _count = arrays[key]
        write_string(out, key.encode("utf-8"))
        out += struct.pack("<I", GGUF_ARRAY)
        out += payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    arrays = extract_arrays(args.model)
    write_fixture(args.output, arrays)
    print(f"wrote {args.output}")
    print(f"size {args.output.stat().st_size}")
    print(f"sha256 {sha256_file(args.output)}")
    for key in TOKEN_KEYS:
        print(f"{key} {arrays[key][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
