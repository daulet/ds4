#!/usr/bin/env python3
"""Compare DS4 server and KV artifacts against the Milestone 0 baselines.

The default invocation self-compares the committed M0.4 and M0.5 artifacts.  A
candidate directory can be supplied later when fresh oracle runs or Rust output
exist.  Only the volatile fields documented in the baseline manifest are
normalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


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


class Comparator:
    def __init__(
        self,
        root: Path,
        server_baseline: Path,
        server_candidate: Path,
        kv_baseline: Path,
        kv_candidate: Path,
    ) -> None:
        self.root = root.resolve()
        self.server_baseline = self.resolve(server_baseline)
        self.server_candidate = self.resolve(server_candidate)
        self.kv_baseline = self.resolve(kv_baseline)
        self.kv_candidate = self.resolve(kv_candidate)
        self.sections: list[Section] = []

    @property
    def ok(self) -> bool:
        return all(section.ok for section in self.sections)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def add_section(
        self, name: str, oracle: str, fixture: str, comparator: str
    ) -> Section:
        section = Section(name, oracle, fixture, comparator)
        self.sections.append(section)
        return section

    def require_file(self, section: Section, path: Path) -> bool:
        section.check(path.is_file(), f"missing file: {self.rel(path)}")
        return path.is_file()

    def read_text(self, section: Section, path: Path) -> str:
        if not self.require_file(section, path):
            return ""
        try:
            return path.read_text()
        except UnicodeDecodeError as exc:
            section.check(False, f"failed to decode {self.rel(path)}: {exc}")
            return ""

    def read_json(self, section: Section, path: Path) -> object | None:
        text = self.read_text(section, path)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            section.check(False, f"invalid JSON in {self.rel(path)}: {exc}")
            return None

    def compare_equal(self, section: Section, label: str, left: object, right: object) -> None:
        section.check(left == right, f"{label}: drift")

    def compare_server(self) -> None:
        section = self.add_section(
            "m0.4 server normalized comparison",
            "M0.4 current ./ds4-server B300 trace replay",
            "server-traces/m0.4 responses, headers, SSE, replay log, and trace",
            "normalize documented volatile IDs/timestamps/timings, compare behavior",
        )
        self.compare_json_dir(
            section,
            self.server_baseline / "responses",
            self.server_candidate / "responses",
            [
                "chat_basic.json",
                "chat_cache_continuation.json",
                "chat_cache_seed.json",
                "chat_thinking_disabled.json",
                "chat_tool_call.json",
                "models.json",
            ],
        )
        self.compare_sse(
            section,
            self.server_baseline / "responses/chat_stream.sse",
            self.server_candidate / "responses/chat_stream.sse",
        )
        self.compare_text_files(
            section,
            "server headers",
            self.server_baseline / "headers",
            self.server_candidate / "headers",
            [
                "chat_basic.headers.txt",
                "chat_cache_continuation.headers.txt",
                "chat_cache_seed.headers.txt",
                "chat_stream.headers.txt",
                "chat_thinking_disabled.headers.txt",
                "chat_tool_call.headers.txt",
            ],
            normalize_header_text,
        )
        self.compare_text_file(
            section,
            "server replay log",
            self.server_baseline / "logs/replay.log",
            self.server_candidate / "logs/replay.log",
            normalize_replay_log,
        )
        self.compare_text_file(
            section,
            "server trace",
            self.server_baseline / "traces/server.trace",
            self.server_candidate / "traces/server.trace",
            normalize_trace_text,
        )

    def compare_kv(self) -> None:
        section = self.add_section(
            "m0.5 KV normalized comparison",
            "M0.5 current ./ds4-server disk-KV restore replay",
            "kv-artifacts/m0.5 responses, cache decisions, rendered text, and KV metadata",
            "normalize documented timestamps/raw hashes, compare KV behavior",
        )
        self.compare_json_dir(
            section,
            self.kv_baseline / "responses",
            self.kv_candidate / "responses",
            [
                "continuation_restore.json",
                "models-server-a.json",
                "models-server-b.json",
                "models-server-c.json",
                "seed_miss.json",
                "seed_restore.json",
            ],
        )
        self.compare_text_files(
            section,
            "KV headers",
            self.kv_baseline / "headers",
            self.kv_candidate / "headers",
            [
                "continuation_restore.headers.txt",
                "seed_miss.headers.txt",
                "seed_restore.headers.txt",
            ],
            normalize_header_text,
        )
        self.compare_text_file(
            section,
            "KV replay log",
            self.kv_baseline / "logs/replay.log",
            self.kv_candidate / "logs/replay.log",
            normalize_replay_log,
        )
        self.compare_text_file(
            section,
            "KV cache decisions",
            self.kv_baseline / "logs/cache-decisions.txt",
            self.kv_candidate / "logs/cache-decisions.txt",
            normalize_path_text,
        )
        self.compare_text_files(
            section,
            "KV traces",
            self.kv_baseline / "traces",
            self.kv_candidate / "traces",
            ["server-a.trace", "server-b.trace", "server-c.trace"],
            normalize_trace_text,
        )
        self.compare_text_files(
            section,
            "KV rendered text",
            self.kv_baseline / "rendered-text",
            self.kv_candidate / "rendered-text",
            [
                "0ab2314538b11686a11e296b7f697651fbd17e60.txt",
                "4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.txt",
                "a0cac6ff193696ccb5d7e9ae151d7255d39cf161.txt",
            ],
            normalize_lf_text,
        )
        self.compare_text_file(
            section,
            "KV normalized hashes",
            self.kv_baseline / "logs/kv-file-normalized-sha256.txt",
            self.kv_candidate / "logs/kv-file-normalized-sha256.txt",
            normalize_path_text,
        )
        self.compare_kv_header(section)

    def compare_json_dir(
        self,
        section: Section,
        baseline_dir: Path,
        candidate_dir: Path,
        names: list[str],
    ) -> None:
        for name in names:
            left_path = baseline_dir / name
            right_path = candidate_dir / name
            left = self.read_json(section, left_path)
            right = self.read_json(section, right_path)
            if left is None or right is None:
                continue
            self.compare_equal(
                section,
                f"JSON {name}",
                normalize_json(left),
                normalize_json(right),
            )

    def compare_sse(self, section: Section, baseline: Path, candidate: Path) -> None:
        left = self.read_text(section, baseline)
        right = self.read_text(section, candidate)
        if not left or not right:
            return
        self.compare_equal(
            section,
            "SSE chat_stream.sse",
            normalize_sse(left, section, self.rel(baseline)),
            normalize_sse(right, section, self.rel(candidate)),
        )

    def compare_text_files(
        self,
        section: Section,
        label: str,
        baseline_dir: Path,
        candidate_dir: Path,
        names: list[str],
        normalizer: Callable[[str], str],
    ) -> None:
        for name in names:
            self.compare_text_file(
                section,
                f"{label} {name}",
                baseline_dir / name,
                candidate_dir / name,
                normalizer,
            )

    def compare_text_file(
        self,
        section: Section,
        label: str,
        baseline: Path,
        candidate: Path,
        normalizer: Callable[[str], str],
    ) -> None:
        left = self.read_text(section, baseline)
        right = self.read_text(section, candidate)
        if not left or not right:
            return
        self.compare_equal(section, label, normalizer(left), normalizer(right))

    def compare_kv_header(self, section: Section) -> None:
        left = read_kv_header(
            section, self.kv_baseline / "logs/kv-header.tsv", self.rel
        )
        right = read_kv_header(
            section, self.kv_candidate / "logs/kv-header.tsv", self.rel
        )
        self.compare_equal(section, "KV header semantic fields", left, right)

    def run(self) -> None:
        self.compare_server()
        self.compare_kv()

    def report_text(self) -> str:
        lines = ["DS4 server/KV artifact comparison", f"root: {self.root}"]
        lines.append(f"server_baseline: {self.rel(self.server_baseline)}")
        lines.append(f"server_candidate: {self.rel(self.server_candidate)}")
        lines.append(f"kv_baseline: {self.rel(self.kv_baseline)}")
        lines.append(f"kv_candidate: {self.rel(self.kv_candidate)}")
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
            "server_baseline": self.rel(self.server_baseline),
            "server_candidate": self.rel(self.server_candidate),
            "kv_baseline": self.rel(self.kv_baseline),
            "kv_candidate": self.rel(self.kv_candidate),
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


VOLATILE_JSON_KEYS = {"id", "created"}


def normalize_json(value: object) -> object:
    value = deepcopy(value)
    strip_volatile_json(value)
    return value


def strip_volatile_json(value: object) -> None:
    if isinstance(value, dict):
        for key in list(value.keys()):
            if key in VOLATILE_JSON_KEYS:
                value[key] = f"<{key}>"
            else:
                strip_volatile_json(value[key])
    elif isinstance(value, list):
        for item in value:
            strip_volatile_json(item)


def normalize_sse(text: str, section: Section, label: str) -> list[object]:
    events: list[object] = []
    for raw_line in normalize_lf_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("data: "):
            section.check(False, f"{label}: non-data SSE line: {line}")
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            events.append(normalize_json(json.loads(payload)))
        except json.JSONDecodeError as exc:
            section.check(False, f"{label}: invalid SSE JSON: {exc}")
    return events


def normalize_lf_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_header_text(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in normalize_lf_text(text).splitlines() if line.strip()
    )


def normalize_path_text(text: str) -> str:
    text = normalize_lf_text(text)
    text = text.replace("/workspace/ds4/", "")
    return text.strip() + "\n"


def normalize_replay_log(text: str) -> str:
    text = normalize_path_text(text)
    text = re.sub(r"started_utc=\S+", "started_utc=<time>", text)
    text = re.sub(r"finished_utc=\S+", "finished_utc=<time>", text)
    text = re.sub(r"finished_utc:\s+\S+", "finished_utc: <time>", text)
    return text


def normalize_trace_text(text: str) -> str:
    text = normalize_path_text(text)
    text = re.sub(
        r"===== request (\d+) \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} =====",
        r"===== request \1 <time> =====",
        text,
    )
    text = re.sub(r"elapsed_sec: [0-9.]+", "elapsed_sec: <elapsed>", text)
    text = re.sub(r"load=[0-9.]+ ms", "load=<ms> ms", text)
    text = re.sub(r"save=[0-9.]+ ms", "save=<ms> ms", text)
    return text


KV_VOLATILE_COLUMNS = {"sha256", "created_unix", "last_used_unix"}


def read_kv_header(
    section: Section, path: Path, rel: Callable[[Path], str]
) -> list[dict[str, str]]:
    if not path.is_file():
        section.check(False, f"missing file: {rel(path)}")
        return []
    rows = list(csv.DictReader(path.read_text().splitlines(), delimiter="\t"))
    section.check(bool(rows), f"{rel(path)}: no KV header rows")
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append(
            {key: value for key, value in row.items() if key not in KV_VOLATILE_COLUMNS}
        )
    return sorted(normalized, key=lambda row: row.get("file", ""))


def make_comparator(
    root: Path,
    server_candidate: Path | None = None,
    kv_candidate: Path | None = None,
) -> Comparator:
    server = Path("ds4-parity/baselines/server-traces/m0.4")
    kv = Path("ds4-parity/baselines/kv-artifacts/m0.5")
    return Comparator(
        root=root,
        server_baseline=server,
        server_candidate=server_candidate or server,
        kv_baseline=kv,
        kv_candidate=kv_candidate or kv,
    )


def copy_candidate_roots(root: Path, tmp_root: Path) -> tuple[Path, Path]:
    server_src = root / "ds4-parity/baselines/server-traces/m0.4"
    kv_src = root / "ds4-parity/baselines/kv-artifacts/m0.5"
    server_dst = tmp_root / "server-candidate"
    kv_dst = tmp_root / "kv-candidate"
    shutil.copytree(server_src, server_dst)
    shutil.copytree(kv_src, kv_dst)
    return server_dst, kv_dst


def run_negative_test(root: Path) -> int:
    cases: list[tuple[str, Callable[[Path, Path], None]]] = [
        ("finish_reason", corrupt_finish_reason),
        ("cached_tokens", corrupt_cached_tokens),
        ("cache_source", corrupt_cache_source),
        ("kv_reason", corrupt_kv_reason),
        ("rendered_text", corrupt_rendered_text),
    ]
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ds4-server-kv-negative-") as tmp:
        tmp_root = Path(tmp)
        for name, corrupt in cases:
            case_root = tmp_root / name
            case_root.mkdir()
            server_candidate, kv_candidate = copy_candidate_roots(root, case_root)
            corrupt(server_candidate, kv_candidate)
            comparator = make_comparator(root, server_candidate, kv_candidate)
            comparator.run()
            if comparator.ok:
                failures.append(name)
            else:
                first = next(
                    error
                    for section in comparator.sections
                    for error in section.errors
                )
                print(f"negative-test {name}: PASS: {first}")
    if failures:
        print("negative-test: FAIL: drift was not detected for " + ", ".join(failures))
        return 1
    print(f"negative-test: PASS: {len(cases)} drift cases detected")
    return 0


def corrupt_finish_reason(server: Path, kv: Path) -> None:
    del kv
    path = server / "responses/chat_basic.json"
    obj = json.loads(path.read_text())
    obj["choices"][0]["finish_reason"] = "length"
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def corrupt_cached_tokens(server: Path, kv: Path) -> None:
    del server
    path = kv / "responses/seed_restore.json"
    obj = json.loads(path.read_text())
    obj["usage"]["prompt_tokens_details"]["cached_tokens"] = 549
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def corrupt_cache_source(server: Path, kv: Path) -> None:
    del server
    path = kv / "logs/cache-decisions.txt"
    text = path.read_text()
    path.write_text(text.replace("cache_source: disk-text", "cache_source: none", 1))


def corrupt_kv_reason(server: Path, kv: Path) -> None:
    del server
    path = kv / "logs/kv-header.tsv"
    text = path.read_text()
    path.write_text(text.replace("\tcold\t", "\tevict\t", 1))


def corrupt_rendered_text(server: Path, kv: Path) -> None:
    del server
    path = kv / "rendered-text/0ab2314538b11686a11e296b7f697651fbd17e60.txt"
    with path.open("a") as f:
        f.write(" drift")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of ds4-parity/)",
    )
    parser.add_argument(
        "--server-candidate",
        type=Path,
        help="candidate server-traces artifact directory (default: M0.4 baseline)",
    )
    parser.add_argument(
        "--kv-candidate",
        type=Path,
        help="candidate kv-artifacts directory (default: M0.5 baseline)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="copy baseline artifacts, corrupt behavioral fields, and require failures",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.negative_test:
        return run_negative_test(root)
    comparator = make_comparator(root, args.server_candidate, args.kv_candidate)
    comparator.run()
    sys.stdout.write(comparator.report_json() if args.json else comparator.report_text())
    return 0 if comparator.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
