#!/usr/bin/env python3
"""Compare DS4 benchmark CSV artifacts against the M0.6 baseline.

Workload shape is behavioral surface: CSV schema, context frontiers, prefill
intervals, generation-token counts, and KV cache byte counts must match
exactly.  Throughput is performance surface and is compared only after the
candidate metadata confirms the same model, prompt, CUDA backend marker, and
GPU machine class.  Regressions beyond the configured threshold fail the
comparison and are reported as performance regressions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


EXPECTED_HEADER = [
    "ctx_tokens",
    "prefill_tokens",
    "prefill_tps",
    "gen_tokens",
    "gen_tps",
    "kvcache_bytes",
]
CSV_NAMES = ["b300-short.csv", "b300-long.csv"]
DEFAULT_MAX_REGRESSION = 0.10
EXACT_ENV_KEYS = [
    "model_sha256",
    "model_size_bytes",
    "prompt_path",
    "prompt_sha256",
    "prompt_size_bytes",
]
COMMAND_KEYS = ["short_command", "long_command"]


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


@dataclass(frozen=True)
class BenchRow:
    ctx_tokens: int
    prefill_tokens: int
    prefill_tps: float
    gen_tokens: int
    gen_tps: float
    kvcache_bytes: int


class BenchComparator:
    def __init__(
        self,
        root: Path,
        baseline_dir: Path,
        candidate_dir: Path,
        max_regression: float,
    ) -> None:
        self.root = root.resolve()
        self.baseline_dir = self.resolve(baseline_dir)
        self.candidate_dir = self.resolve(candidate_dir)
        self.max_regression = max_regression
        self.sections: list[Section] = []
        self.same_performance_environment = False

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

    def run(self) -> None:
        self.compare_environment()
        self.compare_csvs()
        self.audit_summaries()

    def compare_environment(self) -> None:
        section = self.add_section(
            "m0.6 benchmark environment",
            "M0.6 B300 CUDA benchmark capture metadata",
            "bench/m0.6/logs/capture-env.txt",
            "same model, prompt, CUDA backend marker, and GPU machine class",
        )
        baseline = parse_capture_env(
            section,
            self.baseline_dir / "logs/capture-env.txt",
            self.rel,
        )
        candidate = parse_capture_env(
            section,
            self.candidate_dir / "logs/capture-env.txt",
            self.rel,
        )
        if not baseline or not candidate:
            self.same_performance_environment = False
            return

        for key in EXACT_ENV_KEYS:
            section.check(key in baseline, f"baseline capture-env missing {key}")
            section.check(key in candidate, f"candidate capture-env missing {key}")
            if key in baseline and key in candidate:
                section.check(
                    candidate[key] == baseline[key],
                    f"capture-env {key} drift: {candidate[key]!r} != {baseline[key]!r}",
                )

        baseline_gpu = gpu_machine_class(baseline.get("gpu", ""))
        candidate_gpu = gpu_machine_class(candidate.get("gpu", ""))
        section.check(bool(baseline_gpu), "baseline capture-env missing GPU class")
        section.check(bool(candidate_gpu), "candidate capture-env missing GPU class")
        if baseline_gpu and candidate_gpu:
            section.check(
                candidate_gpu == baseline_gpu,
                f"GPU machine class drift: {candidate_gpu!r} != {baseline_gpu!r}",
            )

        for key in COMMAND_KEYS:
            section.check(key in baseline, f"baseline capture-env missing {key}")
            section.check(key in candidate, f"candidate capture-env missing {key}")
            if key in baseline and key in candidate:
                baseline_cuda = " --cuda" in f" {baseline[key]} "
                candidate_cuda = " --cuda" in f" {candidate[key]} "
                section.check(baseline_cuda, f"baseline {key} lacks --cuda marker")
                section.check(
                    candidate_cuda == baseline_cuda,
                    f"candidate {key} backend marker drift",
                )

        self.same_performance_environment = section.ok

    def compare_csvs(self) -> None:
        section = self.add_section(
            "m0.6 benchmark CSV comparison",
            "M0.6 b300-short.csv and b300-long.csv",
            "bench/m0.6/csv/*.csv",
            (
                "shape fields exact; throughput must stay within "
                f"{self.max_regression:.0%} regression threshold"
            ),
        )
        for name in CSV_NAMES:
            baseline = read_bench_csv(
                section,
                self.baseline_dir / "csv" / name,
                self.rel,
            )
            candidate = read_bench_csv(
                section,
                self.candidate_dir / "csv" / name,
                self.rel,
            )
            self.compare_csv_rows(section, name, baseline, candidate)

    def compare_csv_rows(
        self,
        section: Section,
        name: str,
        baseline: list[BenchRow],
        candidate: list[BenchRow],
    ) -> None:
        section.check(
            len(candidate) == len(baseline),
            f"{name}: row count drift {len(candidate)} != {len(baseline)}",
        )
        for index, (expected, got) in enumerate(zip(baseline, candidate), start=2):
            label = f"{name}:{index}"
            section.check(
                got.ctx_tokens == expected.ctx_tokens,
                f"{label}: ctx_tokens drift {got.ctx_tokens} != {expected.ctx_tokens}",
            )
            section.check(
                got.prefill_tokens == expected.prefill_tokens,
                f"{label}: prefill_tokens drift {got.prefill_tokens} != {expected.prefill_tokens}",
            )
            section.check(
                got.gen_tokens == expected.gen_tokens,
                f"{label}: gen_tokens drift {got.gen_tokens} != {expected.gen_tokens}",
            )
            section.check(
                got.kvcache_bytes == expected.kvcache_bytes,
                f"{label}: kvcache_bytes drift {got.kvcache_bytes} != {expected.kvcache_bytes}",
            )
            section.check(got.prefill_tps > 0, f"{label}: non-positive prefill_tps")
            section.check(got.gen_tps > 0, f"{label}: non-positive gen_tps")
            if self.same_performance_environment:
                self.check_throughput(
                    section,
                    f"{label}: prefill_tps",
                    got.prefill_tps,
                    expected.prefill_tps,
                )
                self.check_throughput(
                    section,
                    f"{label}: gen_tps",
                    got.gen_tps,
                    expected.gen_tps,
                )
            else:
                section.check(
                    False,
                    f"{label}: throughput threshold not valid because environment comparison failed",
                )

    def check_throughput(
        self,
        section: Section,
        label: str,
        got: float,
        expected: float,
    ) -> None:
        floor = expected * (1.0 - self.max_regression)
        section.check(
            got >= floor,
            (
                f"{label}: performance regression {got:g} < {floor:g} "
                f"(baseline {expected:g}, threshold {self.max_regression:.0%})"
            ),
        )

    def audit_summaries(self) -> None:
        section = self.add_section(
            "m0.6 benchmark CSV summary audit",
            "committed csv-summary.json derived from benchmark CSVs",
            "bench/m0.6/logs/csv-summary.json",
            "recompute summary fields from CSV rows",
        )
        self.audit_summary_dir(section, "baseline", self.baseline_dir)
        self.audit_summary_dir(section, "candidate", self.candidate_dir)

    def audit_summary_dir(self, section: Section, label: str, directory: Path) -> None:
        actual = read_summary_json(section, directory / "logs/csv-summary.json", self.rel)
        computed = []
        for name in sorted(CSV_NAMES):
            rows = read_bench_csv(section, directory / "csv" / name, self.rel)
            computed.append(compute_summary(name, rows))
        if actual is None:
            return
        section.check(
            sort_summary(actual) == sort_summary(computed),
            f"{label} csv-summary.json drift",
        )

    def report_text(self) -> str:
        lines = [
            "DS4 benchmark CSV comparison",
            f"root: {self.root}",
            f"baseline_dir: {self.rel(self.baseline_dir)}",
            f"candidate_dir: {self.rel(self.candidate_dir)}",
            f"max_regression: {self.max_regression:.0%}",
        ]
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
            "baseline_dir": self.rel(self.baseline_dir),
            "candidate_dir": self.rel(self.candidate_dir),
            "max_regression": self.max_regression,
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


def parse_capture_env(section: Section, path: Path, rel) -> dict[str, str]:
    if not path.is_file():
        section.check(False, f"missing file: {rel(path)}")
        return {}
    values: dict[str, str] = {}
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            section.check(False, f"{rel(path)}:{lineno}: malformed env line")
            continue
        key, value = line.split("=", 1)
        values[key] = value
    section.check(bool(values), f"{rel(path)}: no capture env entries")
    return values


def gpu_machine_class(gpu: str) -> str:
    return gpu.split(",", 1)[0].strip()


def read_bench_csv(section: Section, path: Path, rel) -> list[BenchRow]:
    if not path.is_file():
        section.check(False, f"missing file: {rel(path)}")
        return []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            section.check(
                reader.fieldnames == EXPECTED_HEADER,
                f"{rel(path)}: header drift: {reader.fieldnames}",
            )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        section.check(False, f"failed to decode {rel(path)}: {exc}")
        return []

    parsed: list[BenchRow] = []
    section.check(bool(rows), f"{rel(path)}: no rows")
    for lineno, row in enumerate(rows, start=2):
        parsed_row = parse_bench_row(section, rel(path), lineno, row)
        if parsed_row is not None:
            parsed.append(parsed_row)
    return parsed


def parse_bench_row(
    section: Section,
    filename: str,
    lineno: int,
    row: dict[str, str],
) -> BenchRow | None:
    ctx_tokens = parse_int_field(section, filename, lineno, row, "ctx_tokens")
    prefill_tokens = parse_int_field(section, filename, lineno, row, "prefill_tokens")
    prefill_tps = parse_float_field(section, filename, lineno, row, "prefill_tps")
    gen_tokens = parse_int_field(section, filename, lineno, row, "gen_tokens")
    gen_tps = parse_float_field(section, filename, lineno, row, "gen_tps")
    kvcache_bytes = parse_int_field(section, filename, lineno, row, "kvcache_bytes")
    if (
        ctx_tokens is None
        or prefill_tokens is None
        or prefill_tps is None
        or gen_tokens is None
        or gen_tps is None
        or kvcache_bytes is None
    ):
        return None
    return BenchRow(
        ctx_tokens=ctx_tokens,
        prefill_tokens=prefill_tokens,
        prefill_tps=prefill_tps,
        gen_tokens=gen_tokens,
        gen_tps=gen_tps,
        kvcache_bytes=kvcache_bytes,
    )


def parse_int_field(
    section: Section,
    filename: str,
    lineno: int,
    row: dict[str, str],
    field: str,
) -> int | None:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        section.check(False, f"{filename}:{lineno}: invalid integer field {field}")
        return None


def parse_float_field(
    section: Section,
    filename: str,
    lineno: int,
    row: dict[str, str],
    field: str,
) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        section.check(False, f"{filename}:{lineno}: invalid float field {field}")
        return None
    section.check(math.isfinite(value), f"{filename}:{lineno}: non-finite field {field}")
    return value


def compute_summary(name: str, rows: list[BenchRow]) -> dict[str, object]:
    return {
        "csv": name,
        "rows": len(rows),
        "ctx_tokens": [row.ctx_tokens for row in rows],
        "prefill_tokens": [row.prefill_tokens for row in rows],
        "gen_tokens": [row.gen_tokens for row in rows],
        "min_prefill_tps": min((row.prefill_tps for row in rows), default=0.0),
        "max_prefill_tps": max((row.prefill_tps for row in rows), default=0.0),
        "min_gen_tps": min((row.gen_tps for row in rows), default=0.0),
        "max_gen_tps": max((row.gen_tps for row in rows), default=0.0),
        "kvcache_bytes": [row.kvcache_bytes for row in rows],
    }


def read_summary_json(
    section: Section, path: Path, rel
) -> list[dict[str, object]] | None:
    if not path.is_file():
        section.check(False, f"missing file: {rel(path)}")
        return None
    try:
        obj = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        section.check(False, f"invalid JSON in {rel(path)}: {exc}")
        return None
    section.check(isinstance(obj, list), f"{rel(path)}: summary should be a list")
    if not isinstance(obj, list):
        return None
    out: list[dict[str, object]] = []
    for index, item in enumerate(obj):
        if not isinstance(item, dict):
            section.check(False, f"{rel(path)} item {index}: should be an object")
            continue
        out.append(item)
    return out


def sort_summary(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(items, key=lambda item: str(item.get("csv", "")))


def make_comparator(
    root: Path,
    candidate_dir: Path | None = None,
    baseline_dir: Path | None = None,
    max_regression: float = DEFAULT_MAX_REGRESSION,
) -> BenchComparator:
    baseline = baseline_dir or Path("ds4-parity/baselines/bench/m0.6")
    return BenchComparator(
        root=root,
        baseline_dir=baseline,
        candidate_dir=candidate_dir or baseline,
        max_regression=max_regression,
    )


def copy_candidate_root(root: Path, target: Path) -> Path:
    src = root / "ds4-parity/baselines/bench/m0.6"
    dst = target / "bench-candidate"
    shutil.copytree(src, dst)
    return dst


def run_negative_test(root: Path, max_regression: float) -> int:
    cases: list[tuple[str, Callable[[Path, float], None]]] = [
        ("schema", corrupt_schema),
        ("frontier", corrupt_frontier),
        ("gen_tokens", corrupt_gen_tokens),
        ("kvcache_bytes", corrupt_kvcache_bytes),
        ("throughput", corrupt_throughput),
    ]
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ds4-bench-negative-") as tmp:
        tmp_root = Path(tmp)
        for name, corrupt in cases:
            case_root = tmp_root / name
            case_root.mkdir()
            candidate = copy_candidate_root(root, case_root)
            corrupt(candidate, max_regression)
            comparator = make_comparator(
                root=root,
                candidate_dir=candidate,
                max_regression=max_regression,
            )
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


def corrupt_schema(candidate: Path, max_regression: float) -> None:
    del max_regression
    path = candidate / "csv/b300-short.csv"
    text = path.read_text()
    path.write_text(text.replace("gen_tps", "decode_tps", 1))


def corrupt_frontier(candidate: Path, max_regression: float) -> None:
    del max_regression
    path = candidate / "csv/b300-short.csv"
    text = path.read_text()
    path.write_text(text.replace("\n4096,2048,", "\n4097,2048,", 1))


def corrupt_gen_tokens(candidate: Path, max_regression: float) -> None:
    del max_regression
    path = candidate / "csv/b300-long.csv"
    text = path.read_text()
    path.write_text(text.replace(",32,35.09,", ",31,35.09,", 1))


def corrupt_kvcache_bytes(candidate: Path, max_regression: float) -> None:
    del max_regression
    path = candidate / "csv/b300-long.csv"
    text = path.read_text()
    path.write_text(text.replace(",475014540", ",475014541", 1))


def corrupt_throughput(candidate: Path, max_regression: float) -> None:
    path = candidate / "csv/b300-short.csv"
    text = path.read_text()
    baseline = 1435.38
    regressed = baseline * (1.0 - max_regression) - 1.0
    path.write_text(text.replace(",1435.38,", f",{regressed:.2f},", 1))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of ds4-parity/)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="baseline bench/m0.6 artifact directory",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="candidate bench artifact directory (default: baseline directory)",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=DEFAULT_MAX_REGRESSION,
        help="maximum accepted throughput regression ratio (default: 0.10)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="copy baseline artifacts, corrupt CSV fields, and require failures",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if (
        args.max_regression < 0
        or args.max_regression >= 1
        or not math.isfinite(args.max_regression)
    ):
        print("--max-regression must be a finite ratio in [0, 1)", file=sys.stderr)
        return 2
    if args.negative_test:
        return run_negative_test(root, args.max_regression)
    comparator = make_comparator(
        root=root,
        candidate_dir=args.candidate_dir,
        baseline_dir=args.baseline_dir,
        max_regression=args.max_regression,
    )
    comparator.run()
    sys.stdout.write(comparator.report_json() if args.json else comparator.report_text())
    return 0 if comparator.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
