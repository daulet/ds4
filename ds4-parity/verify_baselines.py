#!/usr/bin/env python3
"""Static verifier for committed DS4 parity baselines.

This verifier intentionally does not rerun model-backed commands.  It checks the
committed Milestone 0 artifact set for byte integrity and basic structured
shape so later parity work has a cheap local guardrail before heavier oracle
runs are wired in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


EXPECTED_OFFICIAL_VEC_SHA256 = (
    "0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac"
)
EXPECTED_PROMESSI_SHA256 = (
    "f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f"
)
EXPECTED_MODEL_SHA256 = (
    "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
)


@dataclass
class Section:
    name: str
    oracle: str
    fixture: str
    comparator: str
    checks: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


class Verifier:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.sections: list[Section] = []

    @property
    def ok(self) -> bool:
        return all(section.ok for section in self.sections)

    def path(self, rel: str | Path) -> Path:
        return self.root / rel

    def add_section(
        self, name: str, oracle: str, fixture: str, comparator: str
    ) -> Section:
        section = Section(name, oracle, fixture, comparator)
        self.sections.append(section)
        return section

    def require_file(self, section: Section, rel: str | Path) -> Path:
        path = self.path(rel)
        section.check(path.is_file(), f"missing file: {rel}")
        return path

    def read_text(self, section: Section, rel: str | Path) -> str:
        path = self.require_file(section, rel)
        if not path.is_file():
            return ""
        try:
            return path.read_text()
        except UnicodeDecodeError as exc:
            section.check(False, f"failed to decode {rel}: {exc}")
            return ""

    def load_json(self, section: Section, rel: str | Path) -> object | None:
        text = self.read_text(section, rel)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            section.check(False, f"invalid JSON in {rel}: {exc}")
            return None

    def sha256_file(self, rel: str | Path) -> str:
        h = hashlib.sha256()
        with self.path(rel).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_artifact_hashes(self, section: Section, rel: str) -> None:
        text = self.read_text(section, rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            section.check(len(parts) >= 2, f"{rel}:{lineno}: malformed hash line")
            if len(parts) < 2:
                continue
            expected, artifact = parts[0], parts[1]
            path = self.path(artifact)
            section.check(path.is_file(), f"{rel}:{lineno}: missing artifact {artifact}")
            if not path.is_file():
                continue
            actual = self.sha256_file(artifact)
            section.check(
                actual == expected,
                f"{rel}:{lineno}: sha256 mismatch for {artifact}: {actual} != {expected}",
            )

    def verify_exit_statuses(
        self, section: Section, expected: dict[str, int], base: str = "ds4-parity/baselines/logs"
    ) -> None:
        for filename, want in sorted(expected.items()):
            rel = f"{base}/{filename}"
            text = self.read_text(section, rel)
            matches = re.findall(r"^exit_status: (\d+)$", text, re.MULTILINE)
            section.check(bool(matches), f"{rel}: missing exit_status")
            if matches:
                got = int(matches[-1])
                section.check(got == want, f"{rel}: exit_status {got}, want {want}")

    def verify_m02_build_logs(self) -> None:
        section = self.add_section(
            "m0.2 build baselines",
            "captured C/CUDA/Metal build and smoke-test logs",
            "ds4-parity/baselines/logs/m0.2-*.log",
            "expected exit statuses and command-log markers",
        )
        expected = {
            "m0.2-make-clean.log": 0,
            "m0.2-make.log": 2,
            "m0.2-arm64-make-clean.log": 0,
            "m0.2-arm64-make.log": 0,
            "m0.2-arm64-make-test.log": 2,
            "m0.2-arm64-ds4-test-server.log": 0,
            "m0.2-arm64-ds4-test-metal-kernels.log": 0,
            "m0.2-arm64-make-clean-before-cpu.log": 0,
            "m0.2-arm64-make-cpu.log": 0,
            "m0.2-arm64-cpu-artifacts.log": 0,
            "m0.2-arm64-make-cuda-regression.log": 0,
            "m0.2-b300-pod-apply.log": 0,
            "m0.2-b300-pod-wait.log": 0,
            "m0.2-b300-source-copy.log": 0,
            "m0.2-b300-env.log": 0,
            "m0.2-b300-make-cuda-generic.log": 0,
            "m0.2-b300-cuda-artifacts.log": 0,
            "m0.2-b300-make-cuda-regression.log": 0,
            "m0.2-b300-make-test.log": 2,
            "m0.2-b300-ds4-test-server.log": 0,
            "m0.2-b300-ds4-test-metal-kernels.log": 0,
        }
        self.verify_exit_statuses(section, expected)
        arm64 = self.read_text(section, "ds4-parity/baselines/logs/m0.2-arm64-cpu-artifacts.log")
        section.check("Mach-O 64-bit executable arm64" in arm64, "arm64 artifact log lacks Mach-O arm64 marker")
        cuda = self.read_text(section, "ds4-parity/baselines/logs/m0.2-b300-cuda-artifacts.log")
        section.check("ELF 64-bit LSB pie executable, x86-64" in cuda, "B300 artifact log lacks x86-64 ELF marker")
        local_fail = self.read_text(section, "ds4-parity/baselines/logs/m0.2-make.log")
        section.check("unsupported option '-mcpu='" in local_fail, "default local make failure marker changed")

    def verify_m03_logprob(self) -> None:
        section = self.add_section(
            "m0.3 official vector baseline",
            "current ./ds4_test --logprob-vectors on B300 q2-imatrix",
            "tests/test-vectors/official.vec and ds4-parity/baselines/logs/m0.3-*",
            "fixture hash, model hash, vector markers, and exit statuses",
        )
        vec_rel = "tests/test-vectors/official.vec"
        self.require_file(section, vec_rel)
        if self.path(vec_rel).is_file():
            section.check(
                self.sha256_file(vec_rel) == EXPECTED_OFFICIAL_VEC_SHA256,
                f"{vec_rel}: official vector sha256 mismatch",
            )
            section.check(self.path(vec_rel).stat().st_size == 1207, f"{vec_rel}: official vector size mismatch")
        expected = {
            "m0.3-b300-source-refresh.log": 0,
            "m0.3-b300-download-wrapper.log": 0,
            "m0.3-b300-model-hash.log": 0,
            "m0.3-b300-make-ds4-test.log": 0,
            "m0.3-b300-logprob-vectors.log": 0,
            "m0.3-official-vec-hash.log": 0,
        }
        self.verify_exit_statuses(section, expected)
        log = self.read_text(section, "ds4-parity/baselines/logs/m0.3-b300-logprob-vectors.log")
        for marker in [
            EXPECTED_MODEL_SHA256,
            "ds4-test: vector short_italian_fact",
            "ds4-test: vector short_code_completion",
            "ds4-test: vector short_reasoning_plain",
            "ds4-test: vector long_memory_archive skipped",
            "ds4-test: vector long_code_audit",
            "logprob-vectors: OK",
        ]:
            section.check(marker in log, f"M0.3 log missing marker: {marker}")

    def verify_m04_server(self) -> None:
        section = self.add_section(
            "m0.4 server trace baselines",
            "current ./ds4-server B300 trace replay",
            "server-fixtures/m0.4 and server-traces/m0.4",
            "artifact hashes, JSON/SSE parsing, replay markers",
        )
        self.verify_artifact_hashes(section, "ds4-parity/baselines/server-traces/m0.4/logs/artifact-sha256.txt")
        for rel in sorted(self.path("ds4-parity/baselines/server-fixtures/m0.4").glob("*.json")):
            self.load_json(section, rel.relative_to(self.root))
        responses = {
            "chat_basic.json": ("baseline ready", "stop", 0, 11),
            "chat_cache_seed.json": ("cache ready", "stop", 0, 39),
            "chat_cache_continuation.json": ("cache continued", "stop", 41, 9),
            "chat_thinking_disabled.json": ("2", "stop", 0, 15),
        }
        for name, (content, finish, cached, written) in responses.items():
            rel = f"ds4-parity/baselines/server-traces/m0.4/responses/{name}"
            obj = self.load_json(section, rel)
            if not isinstance(obj, dict):
                continue
            choice = obj["choices"][0]
            usage = obj["usage"]["prompt_tokens_details"]
            section.check(choice["message"].get("content") == content, f"{rel}: content drift")
            section.check(choice["finish_reason"] == finish, f"{rel}: finish drift")
            section.check(usage["cached_tokens"] == cached, f"{rel}: cached token drift")
            section.check(usage["cache_write_tokens"] == written, f"{rel}: cache write drift")
        tool = self.load_json(section, "ds4-parity/baselines/server-traces/m0.4/responses/chat_tool_call.json")
        if isinstance(tool, dict):
            choice = tool["choices"][0]
            calls = choice["message"].get("tool_calls", [])
            section.check(choice["finish_reason"] == "tool_calls", "chat_tool_call: finish_reason drift")
            section.check(calls and calls[0]["function"]["name"] == "list_files", "chat_tool_call: tool name drift")
            section.check(calls and calls[0]["function"]["arguments"] == '{"path":"."}', "chat_tool_call: arguments drift")
        sse = self.read_text(section, "ds4-parity/baselines/server-traces/m0.4/responses/chat_stream.sse")
        section.check("data: [DONE]" in sse, "chat_stream.sse missing terminal DONE")
        replay = self.read_text(section, "ds4-parity/baselines/server-traces/m0.4/logs/replay.log")
        section.check(replay.count("http_code=200") == 6, "M0.4 replay should contain six HTTP 200 entries")

    def verify_m05_kv(self) -> None:
        section = self.add_section(
            "m0.5 KV restore baselines",
            "current ./ds4-server disk-KV restore behavior",
            "kv-fixtures/m0.5 and kv-artifacts/m0.5",
            "artifact hashes, JSON parsing, KV metadata, rendered text hashes",
        )
        self.verify_artifact_hashes(section, "ds4-parity/baselines/kv-artifacts/m0.5/logs/artifact-sha256.txt")
        for rel in sorted(self.path("ds4-parity/baselines/kv-fixtures/m0.5").glob("*.json")):
            self.load_json(section, rel.relative_to(self.root))
        expected_responses = {
            "seed_miss.json": ("I notice", "length", 550, 0, 550),
            "seed_restore.json": ("I notice", "length", 550, 550, 0),
            "continuation_restore.json": ("kv continued", "stop", 561, 552, 9),
        }
        for name, (content, finish, prompt_tokens, cached, written) in expected_responses.items():
            rel = f"ds4-parity/baselines/kv-artifacts/m0.5/responses/{name}"
            obj = self.load_json(section, rel)
            if not isinstance(obj, dict):
                continue
            choice = obj["choices"][0]
            usage = obj["usage"]
            details = usage["prompt_tokens_details"]
            section.check(choice["message"].get("content") == content, f"{rel}: content drift")
            section.check(choice["finish_reason"] == finish, f"{rel}: finish drift")
            section.check(usage["prompt_tokens"] == prompt_tokens, f"{rel}: prompt token drift")
            section.check(details["cached_tokens"] == cached, f"{rel}: cached token drift")
            section.check(details["cache_write_tokens"] == written, f"{rel}: cache write drift")
        raw_kv = list(self.path("ds4-parity/baselines/kv-artifacts/m0.5").glob("**/*.kv"))
        section.check(not raw_kv, "raw .kv files should not be committed")
        rows = self.read_tsv(section, "ds4-parity/baselines/kv-artifacts/m0.5/logs/kv-header.tsv")
        expected = {
            "0ab2314538b11686a11e296b7f697651fbd17e60.kv": ("cold", 550, 1, 32768, 2520),
            "a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv": ("shutdown", 552, 1, 32768, 2528),
            "4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.kv": ("shutdown", 563, 0, 32768, 2632),
        }
        by_file = {row.get("file", ""): row for row in rows}
        section.check(set(by_file) == set(expected), "kv-header.tsv file set drift")
        for name, (reason, tokens, hits, ctx, text_bytes) in expected.items():
            row = by_file.get(name)
            if not row:
                continue
            section.check(row["magic"] == "KVC", f"{name}: bad magic")
            section.check(row["version"] == "1", f"{name}: bad version")
            section.check(row["quant"] == "2", f"{name}: bad quant")
            section.check(row["reason_name"] == reason, f"{name}: reason drift")
            section.check(int(row["tokens"]) == tokens, f"{name}: token drift")
            section.check(int(row["hits"]) == hits, f"{name}: hit drift")
            section.check(int(row["ctx"]) == ctx, f"{name}: context drift")
            section.check(int(row["rendered_text_bytes"]) == text_bytes, f"{name}: rendered byte drift")
            rendered_rel = f"ds4-parity/baselines/kv-artifacts/m0.5/rendered-text/{name[:-3]}.txt"
            self.require_file(section, rendered_rel)
            if self.path(rendered_rel).is_file():
                section.check(self.sha256_file(rendered_rel) == row["rendered_text_sha256"], f"{name}: rendered text hash drift")
        replay = self.read_text(section, "ds4-parity/baselines/kv-artifacts/m0.5/logs/replay.log")
        for marker in [
            "seed_miss",
            "cached_tokens=0",
            "seed_restore",
            "cached_tokens=550",
            "continuation_restore",
            "cached_tokens=552",
            "cache_write_tokens=9",
        ]:
            section.check(marker in replay, f"M0.5 replay missing marker: {marker}")

    def read_tsv(self, section: Section, rel: str) -> list[dict[str, str]]:
        text = self.read_text(section, rel)
        if not text:
            return []
        rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
        section.check(bool(rows), f"{rel}: no rows")
        return rows

    def verify_m06_bench(self) -> None:
        section = self.add_section(
            "m0.6 benchmark CSV baselines",
            "current ./ds4-bench B300 short and long sweeps",
            "bench/m0.6 CSVs and speed-bench/promessi_sposi.txt",
            "artifact hashes, prompt hash, CSV schema and row contracts",
        )
        self.verify_artifact_hashes(section, "ds4-parity/baselines/bench/m0.6/logs/artifact-sha256.txt")
        prompt_rel = "speed-bench/promessi_sposi.txt"
        self.require_file(section, prompt_rel)
        if self.path(prompt_rel).is_file():
            section.check(self.sha256_file(prompt_rel) == EXPECTED_PROMESSI_SHA256, f"{prompt_rel}: prompt hash drift")
        expected_header = ["ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens", "gen_tps", "kvcache_bytes"]
        csv_expectations = {
            "b300-short.csv": ([2048, 4096, 6144, 8192], [2048, 2048, 2048, 2048]),
            "b300-long.csv": ([16384, 24576, 32768], [16384, 8192, 8192]),
        }
        for name, (ctx_expected, prefill_expected) in csv_expectations.items():
            rel = f"ds4-parity/baselines/bench/m0.6/csv/{name}"
            rows = self.read_csv(section, rel, expected_header)
            if not rows:
                continue
            ctx_values = self.csv_ints(section, name, rows, "ctx_tokens")
            prefill_values = self.csv_ints(section, name, rows, "prefill_tokens")
            gen_values = self.csv_ints(section, name, rows, "gen_tokens")
            cache_values = self.csv_ints(section, name, rows, "kvcache_bytes")
            prefill_tps = self.csv_floats(section, name, rows, "prefill_tps")
            gen_tps = self.csv_floats(section, name, rows, "gen_tps")
            if ctx_values is not None:
                section.check(ctx_values == ctx_expected, f"{name}: ctx frontier drift")
            if prefill_values is not None:
                section.check(prefill_values == prefill_expected, f"{name}: prefill interval drift")
            if gen_values is not None:
                section.check(all(value == 32 for value in gen_values), f"{name}: gen_tokens drift")
            if prefill_tps is not None:
                section.check(all(value > 0 for value in prefill_tps), f"{name}: non-positive prefill_tps")
            if gen_tps is not None:
                section.check(all(value > 0 for value in gen_tps), f"{name}: non-positive gen_tps")
            if cache_values is not None:
                section.check(all(value > 0 for value in cache_values), f"{name}: non-positive kvcache_bytes")
        summary = self.load_json(section, "ds4-parity/baselines/bench/m0.6/logs/csv-summary.json")
        section.check(isinstance(summary, list) and len(summary) == 2, "csv-summary.json should contain two summaries")
        capture = self.read_text(section, "ds4-parity/baselines/bench/m0.6/logs/capture-env.txt")
        for marker in [EXPECTED_MODEL_SHA256, EXPECTED_PROMESSI_SHA256, "NVIDIA B300 SXM6 AC"]:
            section.check(marker in capture, f"M0.6 capture-env missing marker: {marker}")

    def read_csv(self, section: Section, rel: str, expected_header: list[str]) -> list[dict[str, str]]:
        path = self.require_file(section, rel)
        if not path.is_file():
            return []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            section.check(reader.fieldnames == expected_header, f"{rel}: header drift: {reader.fieldnames}")
            rows = list(reader)
        section.check(bool(rows), f"{rel}: no rows")
        return rows

    def csv_ints(
        self, section: Section, csv_name: str, rows: list[dict[str, str]], field: str
    ) -> list[int] | None:
        values: list[int] = []
        for index, row in enumerate(rows, start=2):
            try:
                values.append(int(row[field]))
            except (KeyError, TypeError, ValueError):
                section.check(False, f"{csv_name}:{index}: invalid integer field {field}")
                return None
        return values

    def csv_floats(
        self, section: Section, csv_name: str, rows: list[dict[str, str]], field: str
    ) -> list[float] | None:
        values: list[float] = []
        for index, row in enumerate(rows, start=2):
            try:
                values.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                section.check(False, f"{csv_name}:{index}: invalid float field {field}")
                return None
        return values

    def run(self) -> None:
        self.verify_m02_build_logs()
        self.verify_m03_logprob()
        self.verify_m04_server()
        self.verify_m05_kv()
        self.verify_m06_bench()

    def report_text(self) -> str:
        lines = ["DS4 parity baseline verification", f"root: {self.root}"]
        for section in self.sections:
            status = "PASS" if section.ok else "FAIL"
            lines.extend(
                [
                    f"[{status}] {section.name}",
                    f"  oracle: {section.oracle}",
                    f"  fixture: {section.fixture}",
                    f"  comparator: {section.comparator}",
                    f"  checks: {section.checks}",
                ]
            )
            for error in section.errors:
                lines.append(f"  - {error}")
        passed = sum(1 for section in self.sections if section.ok)
        checks = sum(section.checks for section in self.sections)
        lines.append(
            f"summary: {passed}/{len(self.sections)} sections passed, {checks} checks"
        )
        return "\n".join(lines) + "\n"

    def report_json(self) -> str:
        payload = {
            "root": str(self.root),
            "ok": self.ok,
            "sections": [
                {
                    "name": section.name,
                    "oracle": section.oracle,
                    "fixture": section.fixture,
                    "comparator": section.comparator,
                    "ok": section.ok,
                    "checks": section.checks,
                    "errors": section.errors,
                }
                for section in self.sections
            ],
        }
        return json.dumps(payload, indent=2) + "\n"


def copy_negative_fixture_root(source_root: Path, target_root: Path) -> None:
    shutil.copytree(source_root / "ds4-parity", target_root / "ds4-parity")
    (target_root / "tests" / "test-vectors").mkdir(parents=True)
    shutil.copy2(
        source_root / "tests" / "test-vectors" / "official.vec",
        target_root / "tests" / "test-vectors" / "official.vec",
    )
    (target_root / "speed-bench").mkdir()
    shutil.copy2(
        source_root / "speed-bench" / "promessi_sposi.txt",
        target_root / "speed-bench" / "promessi_sposi.txt",
    )


def run_negative_test(root: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="ds4-parity-negative-") as tmp:
        tmp_root = Path(tmp)
        copy_negative_fixture_root(root, tmp_root)
        drift_file = tmp_root / "ds4-parity/baselines/bench/m0.6/csv/b300-short.csv"
        with drift_file.open("a") as f:
            f.write("# deliberate drift for verifier negative test\n")
        verifier = Verifier(tmp_root)
        verifier.run()
        if verifier.ok:
            print("negative-test: FAIL: verifier did not detect deliberate CSV drift")
            return 1
        print("negative-test: PASS: verifier detected deliberate CSV drift")
        for section in verifier.sections:
            if section.errors:
                print(f"detected-by: {section.name}")
                print(f"first-error: {section.errors[0]}")
                break
        return 0


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to verify (default: parent of ds4-parity/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON report instead of text",
    )
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="copy fixtures to a temp directory, corrupt one, and require verification failure",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.negative_test:
        return run_negative_test(root)
    verifier = Verifier(root)
    verifier.run()
    sys.stdout.write(verifier.report_json() if args.json else verifier.report_text())
    return 0 if verifier.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
