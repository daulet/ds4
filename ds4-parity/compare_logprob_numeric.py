#!/usr/bin/env python3
"""Compare DS4 official-vector logprob fixtures and M0.3 run evidence.

The M0.3 B300 log records that the C runner executed
``./ds4_test --logprob-vectors`` successfully, but the runner only prints
per-case pass/fail markers.  The numeric oracle lives in
``tests/test-vectors/official.vec`` and the raw official API JSON that generated
that compact fixture.  This comparator makes that contract explicit for later
Rust output: selected tokens must match exactly, while logprob values may only
drift within the declared tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


EXPECTED_MODEL_SHA256 = (
    "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
)
EXPECTED_MODEL = "deepseek-v4-flash"
DEFAULT_LOGPROB_ABS_TOLERANCE = 4.0
OFFICIAL_FIXTURE_ABS_TOLERANCE = 0.0
DISABLED_CASES = {
    "long_memory_archive": "API/official graph mismatch",
}


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
class TopEntry:
    token_hex: str
    token_bytes: bytes
    logprob: float


@dataclass
class Step:
    index: int
    selected_hex: str
    selected_bytes: bytes
    top_count: int
    top: list[TopEntry] = field(default_factory=list)


@dataclass
class VecCase:
    case_id: str
    ctx: int
    nsteps: int
    prompt_path: str
    steps: list[Step]

    @property
    def generated_bytes(self) -> bytes:
        out = bytearray()
        for step in self.steps:
            out.extend(step.selected_bytes)
        return bytes(out)


class NumericComparator:
    def __init__(
        self,
        root: Path,
        baseline_vec: Path,
        candidate_vec: Path,
        official_dir: Path,
        run_log: Path,
        logprob_abs_tolerance: float,
    ) -> None:
        self.root = root.resolve()
        self.baseline_vec = self.resolve(baseline_vec)
        self.candidate_vec = self.resolve(candidate_vec)
        self.official_dir = self.resolve(official_dir)
        self.run_log = self.resolve(run_log)
        self.logprob_abs_tolerance = logprob_abs_tolerance
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

    def run(self) -> None:
        baseline = self.parse_vec_section(
            "baseline official.vec parse",
            self.baseline_vec,
            "M0.3 compact official-vector fixture",
        )
        candidate = self.parse_vec_section(
            "candidate logprob vector parse",
            self.candidate_vec,
            "candidate official-vector-style output",
        )
        self.compare_candidate_to_baseline(baseline, candidate)
        self.audit_official_json(baseline)
        self.verify_b300_run_log(baseline)

    def parse_vec_section(
        self, name: str, path: Path, oracle: str
    ) -> list[VecCase]:
        section = self.add_section(
            name,
            oracle,
            self.rel(path),
            "parse compact case/step/top logprob fixture",
        )
        return parse_vec(section, path, self.rel)

    def compare_candidate_to_baseline(
        self, baseline: list[VecCase], candidate: list[VecCase]
    ) -> None:
        section = self.add_section(
            "m0.3 candidate vector comparison",
            "M0.3 official.vec selected-token and top-logprob contract",
            f"{self.rel(self.baseline_vec)} vs {self.rel(self.candidate_vec)}",
            (
                "case shape and token bytes exact; top logprobs within "
                f"{self.logprob_abs_tolerance:g} absolute tolerance"
            ),
        )
        baseline_by_id = {case.case_id: case for case in baseline}
        candidate_by_id = {case.case_id: case for case in candidate}
        section.check(
            list(candidate_by_id) == list(baseline_by_id),
            "case order or case id set drift",
        )

        for case_id, expected in baseline_by_id.items():
            got = candidate_by_id.get(case_id)
            section.check(got is not None, f"{case_id}: missing candidate case")
            if got is None:
                continue
            section.check(got.ctx == expected.ctx, f"{case_id}: ctx drift")
            section.check(
                got.prompt_path == expected.prompt_path,
                f"{case_id}: prompt path drift",
            )
            section.check(got.nsteps == expected.nsteps, f"{case_id}: nsteps drift")
            section.check(
                len(got.steps) == len(expected.steps),
                f"{case_id}: parsed step count drift",
            )
            for expected_step, got_step in zip(expected.steps, got.steps):
                label = f"{case_id} step {expected_step.index}"
                section.check(
                    got_step.index == expected_step.index,
                    f"{label}: step index drift",
                )
                section.check(
                    got_step.selected_bytes == expected_step.selected_bytes,
                    f"{label}: selected token drift",
                )
                section.check(
                    got_step.top_count == expected_step.top_count,
                    f"{label}: top-count drift",
                )
                section.check(
                    len(got_step.top) == len(expected_step.top),
                    f"{label}: parsed top entry count drift",
                )
                for top_index, (expected_top, got_top) in enumerate(
                    zip(expected_step.top, got_step.top)
                ):
                    top_label = f"{label} top {top_index}"
                    section.check(
                        got_top.token_bytes == expected_top.token_bytes,
                        f"{top_label}: top token drift",
                    )
                    self.check_float_close(
                        section,
                        top_label,
                        got_top.logprob,
                        expected_top.logprob,
                        self.logprob_abs_tolerance,
                    )

    def audit_official_json(self, baseline: list[VecCase]) -> None:
        section = self.add_section(
            "m0.3 official JSON fixture audit",
            "raw DeepSeek official API top-logprob JSON",
            "tests/test-vectors/official/*.official.json and official.vec",
            "selected bytes exact; compact top logprobs exactly match raw JSON",
        )
        for case in baseline:
            path = self.official_dir / f"{case.case_id}.official.json"
            obj = self.read_json(section, path)
            if not isinstance(obj, dict):
                continue
            section.check(
                obj.get("schema") == "ds4-official-logprobs-v1",
                f"{case.case_id}: schema drift",
            )
            section.check(obj.get("id") == case.case_id, f"{case.case_id}: id drift")
            section.check(
                obj.get("model") == EXPECTED_MODEL,
                f"{case.case_id}: model drift",
            )
            steps = obj.get("steps")
            section.check(
                isinstance(steps, list),
                f"{case.case_id}: official steps should be a list",
            )
            if not isinstance(steps, list):
                continue
            section.check(
                len(steps) == case.nsteps,
                f"{case.case_id}: official step count drift",
            )
            message = obj.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    section.check(
                        case.generated_bytes == content.encode(),
                        f"{case.case_id}: selected byte stream does not match message",
                    )
                else:
                    section.check(False, f"{case.case_id}: message content missing")
            else:
                section.check(False, f"{case.case_id}: message object missing")

            for expected_step, official_step in zip(case.steps, steps):
                self.audit_official_step(section, case.case_id, expected_step, official_step)

    def audit_official_step(
        self,
        section: Section,
        case_id: str,
        expected_step: Step,
        official_step: object,
    ) -> None:
        label = f"{case_id} step {expected_step.index}"
        if not isinstance(official_step, dict):
            section.check(False, f"{label}: official step should be an object")
            return
        section.check(
            official_step.get("step") == expected_step.index,
            f"{label}: official step index drift",
        )
        token = official_step.get("token")
        token_bytes = token_object_bytes(section, token, f"{label} selected")
        section.check(
            token_bytes == expected_step.selected_bytes,
            f"{label}: official selected token drift",
        )
        top_logprobs = official_step.get("top_logprobs")
        section.check(
            isinstance(top_logprobs, list),
            f"{label}: official top_logprobs should be a list",
        )
        if not isinstance(top_logprobs, list):
            return
        section.check(
            len(top_logprobs) >= expected_step.top_count,
            f"{label}: official top_logprobs shorter than compact top-count",
        )
        for top_index, expected_top in enumerate(expected_step.top):
            if top_index >= len(top_logprobs):
                continue
            official_top = top_logprobs[top_index]
            if not isinstance(official_top, dict):
                section.check(False, f"{label} top {top_index}: not an object")
                continue
            official_bytes = token_object_bytes(
                section,
                official_top.get("token"),
                f"{label} top {top_index}",
            )
            section.check(
                official_bytes == expected_top.token_bytes,
                f"{label} top {top_index}: official top token drift",
            )
            official_logprob = official_top.get("logprob")
            if not isinstance(official_logprob, (int, float)):
                section.check(
                    False,
                    f"{label} top {top_index}: official logprob missing",
                )
                continue
            self.check_float_close(
                section,
                f"{label} top {top_index}",
                expected_top.logprob,
                float(official_logprob),
                OFFICIAL_FIXTURE_ABS_TOLERANCE,
            )

    def verify_b300_run_log(self, baseline: list[VecCase]) -> None:
        section = self.add_section(
            "m0.3 B300 run log evidence",
            "captured ./ds4_test --logprob-vectors execution on B300",
            self.rel(self.run_log),
            "run metadata and per-case pass markers",
        )
        text = self.read_text(section, self.run_log)
        if not text:
            return
        markers = [
            "--logprob-vectors",
            "DS4_TEST_VECTOR_FILE=tests/test-vectors/official.vec",
            EXPECTED_MODEL_SHA256,
            "ds4: CUDA backend initialized on NVIDIA B300 SXM6 AC",
            "logprob-vectors: OK",
            "ds4 tests: ok",
            "exit_status: 0",
        ]
        for marker in markers:
            section.check(marker in text, f"M0.3 log missing marker: {marker}")
        forbidden = [
            "selected token mismatch",
            "official top token missing locally",
            "logprob delta too high",
            "TEST FAILED",
        ]
        for marker in forbidden:
            section.check(marker not in text, f"M0.3 log contains failure marker: {marker}")
        for case in baseline:
            if case.case_id in DISABLED_CASES:
                marker = f"ds4-test: vector {case.case_id} skipped ({DISABLED_CASES[case.case_id]})"
            else:
                marker = f"ds4-test: vector {case.case_id}"
            section.check(marker in text, f"M0.3 log missing case marker: {marker}")

    def check_float_close(
        self,
        section: Section,
        label: str,
        got: float,
        expected: float,
        tolerance: float,
    ) -> None:
        if not math.isfinite(got) or not math.isfinite(expected):
            section.check(False, f"{label}: non-finite logprob")
            return
        delta = abs(got - expected)
        section.check(
            delta <= tolerance,
            (
                f"{label}: logprob drift {got:g} vs {expected:g} "
                f"(delta {delta:g}, tolerance {tolerance:g})"
            ),
        )

    def report_text(self) -> str:
        lines = [
            "DS4 logprob/numeric comparison",
            f"root: {self.root}",
            f"baseline_vec: {self.rel(self.baseline_vec)}",
            f"candidate_vec: {self.rel(self.candidate_vec)}",
            f"official_dir: {self.rel(self.official_dir)}",
            f"run_log: {self.rel(self.run_log)}",
            f"logprob_abs_tolerance: {self.logprob_abs_tolerance:g}",
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
            "baseline_vec": self.rel(self.baseline_vec),
            "candidate_vec": self.rel(self.candidate_vec),
            "official_dir": self.rel(self.official_dir),
            "run_log": self.rel(self.run_log),
            "logprob_abs_tolerance": self.logprob_abs_tolerance,
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


def parse_vec(section: Section, path: Path, rel) -> list[VecCase]:
    if not path.is_file():
        section.check(False, f"missing file: {rel(path)}")
        return []
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError as exc:
        section.check(False, f"failed to decode {rel(path)}: {exc}")
        return []

    cases: list[VecCase] = []
    current: VecCase | None = None
    current_step: Step | None = None
    steps_by_index: dict[int, Step] = {}
    seen_case_ids: set[str] = set()

    def finish_case(lineno: int) -> None:
        nonlocal current, current_step, steps_by_index
        if current is None:
            section.check(False, f"{rel(path)}:{lineno}: end outside case")
            return
        ordered = [steps_by_index.get(i) for i in range(current.nsteps)]
        missing = [i for i, step in enumerate(ordered) if step is None]
        section.check(not missing, f"{current.case_id}: missing steps {missing}")
        for step in ordered:
            if step is None:
                continue
            section.check(
                len(step.top) == step.top_count,
                f"{current.case_id} step {step.index}: top-count {len(step.top)}, want {step.top_count}",
            )
        current.steps = [step for step in ordered if step is not None]
        cases.append(current)
        current = None
        current_step = None
        steps_by_index = {}

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "case":
            if current is not None:
                section.check(False, f"{rel(path)}:{lineno}: nested case")
                continue
            if len(parts) != 5:
                section.check(False, f"{rel(path)}:{lineno}: malformed case line")
                continue
            case_id, ctx_text, nsteps_text, prompt_path = parts[1:]
            ctx = parse_int(section, ctx_text, f"{rel(path)}:{lineno}: ctx")
            nsteps = parse_int(section, nsteps_text, f"{rel(path)}:{lineno}: nsteps")
            if ctx is None or nsteps is None:
                continue
            section.check(case_id not in seen_case_ids, f"{case_id}: duplicate case id")
            seen_case_ids.add(case_id)
            section.check(ctx > 0, f"{case_id}: ctx must be positive")
            section.check(0 < nsteps <= 16, f"{case_id}: nsteps out of range")
            current = VecCase(case_id, ctx, nsteps, prompt_path, [])
            current_step = None
            steps_by_index = {}
            continue

        if kind == "step":
            if current is None:
                section.check(False, f"{rel(path)}:{lineno}: step outside case")
                continue
            if len(parts) != 4:
                section.check(False, f"{rel(path)}:{lineno}: malformed step line")
                continue
            index = parse_int(section, parts[1], f"{rel(path)}:{lineno}: step index")
            token_bytes = parse_hex(section, parts[2], f"{rel(path)}:{lineno}: selected")
            top_count = parse_int(section, parts[3], f"{rel(path)}:{lineno}: top-count")
            if index is None or token_bytes is None or top_count is None:
                current_step = None
                continue
            section.check(0 <= index < current.nsteps, f"{current.case_id}: step {index} out of range")
            section.check(index not in steps_by_index, f"{current.case_id}: duplicate step {index}")
            section.check(0 <= top_count <= 32, f"{current.case_id} step {index}: top-count out of range")
            current_step = Step(index, parts[2].lower(), token_bytes, top_count)
            steps_by_index[index] = current_step
            continue

        if kind == "top":
            if current is None or current_step is None:
                section.check(False, f"{rel(path)}:{lineno}: top outside step")
                continue
            if len(parts) != 3:
                section.check(False, f"{rel(path)}:{lineno}: malformed top line")
                continue
            token_bytes = parse_hex(section, parts[1], f"{rel(path)}:{lineno}: top token")
            logprob = parse_float(section, parts[2], f"{rel(path)}:{lineno}: logprob")
            if token_bytes is None or logprob is None:
                continue
            section.check(
                len(current_step.top) < current_step.top_count,
                f"{current.case_id} step {current_step.index}: too many top entries",
            )
            current_step.top.append(TopEntry(parts[1].lower(), token_bytes, logprob))
            continue

        if kind == "end":
            if len(parts) != 1:
                section.check(False, f"{rel(path)}:{lineno}: malformed end line")
                continue
            finish_case(lineno)
            continue

        section.check(False, f"{rel(path)}:{lineno}: unexpected line: {line}")

    if current is not None:
        section.check(False, f"{rel(path)}: unterminated case {current.case_id}")

    section.check(bool(cases), f"{rel(path)}: no vector cases parsed")
    return cases


def parse_int(section: Section, text: str, label: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        section.check(False, f"{label}: invalid integer {text!r}")
        return None


def parse_float(section: Section, text: str, label: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        section.check(False, f"{label}: invalid float {text!r}")
        return None
    section.check(math.isfinite(value), f"{label}: non-finite float")
    return value


def parse_hex(section: Section, text: str, label: str) -> bytes | None:
    if len(text) % 2 != 0:
        section.check(False, f"{label}: odd-length hex token")
        return None
    try:
        return bytes.fromhex(text)
    except ValueError:
        section.check(False, f"{label}: invalid hex token")
        return None


def token_object_bytes(section: Section, token: object, label: str) -> bytes | None:
    if not isinstance(token, dict):
        section.check(False, f"{label}: token should be an object")
        return None
    raw = token.get("bytes")
    if not isinstance(raw, list):
        section.check(False, f"{label}: token bytes should be a list")
        return None
    out = bytearray()
    for index, value in enumerate(raw):
        if not isinstance(value, int) or not 0 <= value <= 255:
            section.check(False, f"{label}: byte {index} out of range")
            return None
        out.append(value)
    return bytes(out)


def make_comparator(
    root: Path,
    baseline_vec: Path | None = None,
    candidate_vec: Path | None = None,
    official_dir: Path | None = None,
    run_log: Path | None = None,
    logprob_abs_tolerance: float = DEFAULT_LOGPROB_ABS_TOLERANCE,
) -> NumericComparator:
    baseline = baseline_vec or Path("tests/test-vectors/official.vec")
    return NumericComparator(
        root=root,
        baseline_vec=baseline,
        candidate_vec=candidate_vec or baseline,
        official_dir=official_dir or Path("tests/test-vectors/official"),
        run_log=run_log or Path("ds4-parity/baselines/logs/m0.3-b300-logprob-vectors.log"),
        logprob_abs_tolerance=logprob_abs_tolerance,
    )


def run_negative_test(root: Path, tolerance: float) -> int:
    cases: list[tuple[str, Callable[[Path, float], None]]] = [
        ("token_drift", corrupt_selected_token),
        ("numeric_drift", corrupt_logprob),
    ]
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ds4-logprob-negative-") as tmp:
        tmp_root = Path(tmp)
        for name, corrupt in cases:
            candidate_vec = tmp_root / f"{name}.vec"
            shutil.copy2(root / "tests/test-vectors/official.vec", candidate_vec)
            corrupt(candidate_vec, tolerance)
            comparator = make_comparator(
                root=root,
                candidate_vec=candidate_vec,
                logprob_abs_tolerance=tolerance,
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


def corrupt_selected_token(path: Path, tolerance: float) -> None:
    del tolerance
    text = path.read_text()
    path.write_text(text.replace("step 0 416461 1", "step 0 416462 1", 1))


def corrupt_logprob(path: Path, tolerance: float) -> None:
    text = path.read_text()
    drift = tolerance + 0.5
    path.write_text(text.replace("top 416461 0", f"top 416461 {drift:g}", 1))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of ds4-parity/)",
    )
    parser.add_argument(
        "--baseline-vec",
        type=Path,
        help="baseline compact vector file (default: tests/test-vectors/official.vec)",
    )
    parser.add_argument(
        "--candidate-vec",
        type=Path,
        help="candidate compact vector file (default: baseline vector)",
    )
    parser.add_argument(
        "--official-dir",
        type=Path,
        help="raw official JSON directory (default: tests/test-vectors/official)",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        help="captured M0.3 run log (default: ds4-parity/baselines/logs/m0.3-b300-logprob-vectors.log)",
    )
    parser.add_argument(
        "--logprob-abs-tol",
        type=float,
        default=DEFAULT_LOGPROB_ABS_TOLERANCE,
        help="absolute logprob tolerance for candidate comparison (default: 4.0)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    parser.add_argument(
        "--negative-test",
        action="store_true",
        help="copy official.vec, corrupt token/numeric fields, and require failures",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.logprob_abs_tol < 0 or not math.isfinite(args.logprob_abs_tol):
        print("--logprob-abs-tol must be a finite non-negative number", file=sys.stderr)
        return 2
    if args.negative_test:
        return run_negative_test(root, args.logprob_abs_tol)
    comparator = make_comparator(
        root=root,
        baseline_vec=args.baseline_vec,
        candidate_vec=args.candidate_vec,
        official_dir=args.official_dir,
        run_log=args.run_log,
        logprob_abs_tolerance=args.logprob_abs_tol,
    )
    comparator.run()
    sys.stdout.write(comparator.report_json() if args.json else comparator.report_text())
    return 0 if comparator.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
