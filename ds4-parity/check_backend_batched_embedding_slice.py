#!/usr/bin/env python3
"""Validate the M13.2 batched embedding replacement slice."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SLICE = ROOT / "ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json"
M13_1_MATRIX = ROOT / "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json"
RUST_MODULE = ROOT / "rust/ds4-gpu/src/replacement_slice.rs"
EMITTER = ROOT / "rust/ds4-gpu/src/bin/ds4-backend-replacement-slice.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_ID = "m13.2-embedding-and-indexer-embed-tokens-hc"
EXPECTED_FIXTURE_ID = "m13.1-embed-tokens-hc"
EXPECTED_OPERATION = "ds4_gpu_embed_tokens_hc_tensor"
EXPECTED_METHOD = "embed_tokens_hc"
EXPECTED_FAMILY = "embedding_and_indexer"
EXPECTED_OUTPUTS = ["after_layer42_hc", "logits"]
EXPECTED_SUPPORTED = ["cuda-b300"]
EXPECTED_UNSUPPORTED = ["cpu", "metal", "runtime-default-route"]
EXPECTED_COMPARATORS = [
    "ds4-parity/compare_prefill_whole_short.py",
    "ds4-parity/compare_prefill_chunked.py",
    "ds4-parity/compare_prefill_resumed.py",
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


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    artifact = load_json(SLICE)
    matrix = load_json(M13_1_MATRIX)
    rust_source = read_text(RUST_MODULE)
    emitter_source = read_text(EMITTER)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, artifact, matrix, rust_source, emitter_source, texts, run_commands=not args.no_commands)
    if args.negative_test:
        run_negative_tests(report, artifact, matrix, rust_source, emitter_source, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    artifact: dict[str, Any],
    matrix: dict[str, Any],
    rust_source: str,
    emitter_source: str,
    texts: dict[str, str],
    *,
    run_commands: bool,
) -> None:
    validate_artifact(report, artifact)
    validate_against_m13_1(report, artifact, matrix)
    validate_rust_module(report, artifact, rust_source, emitter_source)
    if run_commands:
        validate_rust_emitter(report, artifact)
        validate_fail_closed(report, artifact)
        run_dependency_checkers(report)
    validate_static_wiring(report, texts)


def validate_artifact(report: Report, artifact: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_replacement_slice.v1",
        "milestone": "M13.2",
        "status": "batched-embedding-replacement-slice",
        "id": EXPECTED_ID,
        "operation_family": EXPECTED_FAMILY,
        "fixture_id": EXPECTED_FIXTURE_ID,
        "operation": EXPECTED_OPERATION,
        "method": EXPECTED_METHOD,
        "rust_module": "rust/ds4-gpu/src/replacement_slice.rs",
        "facade_replay": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
        "tensor_fixture_manifest": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
        "comparator": "ds4-parity/compare_prefill_whole_short.py",
        "next_required_gate": "M13.3 Indexed Decode Selection Replacement Slice",
        "backend_check": "not-requested",
    }
    for key, value in expected.items():
        report.check(artifact.get(key) == value, f"artifact {key} drift")
    report.check(artifact.get("output_fields") == EXPECTED_OUTPUTS, "output field drift")
    report.check(artifact.get("supported_backends") == EXPECTED_SUPPORTED, "supported backend drift")
    report.check(artifact.get("unsupported_backends") == EXPECTED_UNSUPPORTED, "unsupported backend drift")
    report.check(artifact.get("runtime_route_change") is False, "runtime route overclaim")
    report.check(artifact.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(artifact.get("kernel_replacement") is False, "kernel replacement overclaim")
    for key in ["rust_module", "facade_replay", "tensor_fixture_manifest", "comparator"]:
        value = artifact.get(key)
        report.check(isinstance(value, str) and (ROOT / value).exists(), f"artifact path missing: {key}")


def validate_against_m13_1(report: Report, artifact: dict[str, Any], matrix: dict[str, Any]) -> None:
    report.check(matrix.get("milestone") == "M13.1", "M13.1 matrix milestone drift")
    row = matrix_row(matrix, EXPECTED_OPERATION)
    report.check(row is not None, "M13.1 embed_tokens row missing")
    if row is None:
        return
    report.check(row.get("coverage_level") == "pair-comparator-ready", "M13.1 coverage level drift")
    report.check(row.get("fixture_gap") is False, "M13.1 fixture gap drift")
    report.check(row.get("method") == artifact.get("method"), "M13.1 method drift")
    report.check(row.get("route_candidate_stage") == "M13.2 Batched Embedding Replacement Slice", "M13.1 route stage drift")
    report.check(row.get("comparators") == EXPECTED_COMPARATORS, "M13.1 comparator list drift")
    report.check(row.get("covered_outputs") == artifact.get("output_fields"), "M13.1 output field drift")
    policy = matrix.get("claim_policy", {})
    report.check(policy.get("runtime_route_change") is False, "M13.1 route-change policy drift")
    report.check(policy.get("default_route_replacement_active") is False, "M13.1 default route policy drift")
    report.check(policy.get("removals_allowed") is False, "M13.1 removal policy drift")


def validate_rust_module(
    report: Report,
    artifact: dict[str, Any],
    rust_source: str,
    emitter_source: str,
) -> None:
    for needle in [
        "BATCHED_EMBEDDING_REPLACEMENT_SLICE",
        "BACKEND_REPLACEMENT_SLICES",
        "batched_embedding_replacement_slice",
        "replacement_slice_by_id",
        artifact["status"],
        artifact["id"],
        artifact["operation"],
        artifact["method"],
        "runtime_route_change: false",
        "general_backend_replacement: false",
        "kernel_replacement: false",
    ]:
        report.check(needle in rust_source, f"Rust replacement slice missing {needle}")
    for needle in ["--slice", "replacement_slice_by_id", "spec.status", "usage: ds4-backend-replacement-slice"]:
        report.check(needle in emitter_source, f"Rust emitter missing {needle}")


def validate_rust_emitter(report: Report, artifact: dict[str, Any]) -> None:
    emitted = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-replacement-slice",
            "--quiet",
            "--",
            "--slice",
            "batched-embedding",
        ],
        expected_code=0,
    )
    report.check(emitted == artifact, "Rust batched embedding slice emitter drift")
    default = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-replacement-slice",
            "--quiet",
        ],
        expected_code=0,
    )
    report.check(default.get("milestone") == "M12.4", "default replacement slice drift")
    supported = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-replacement-slice",
            "--quiet",
            "--",
            "--slice",
            "m13.2",
            "--backend",
            "cuda-b300",
        ],
        expected_code=0,
    )
    report.check(supported.get("backend_check") == "supported", "supported backend check drift")
    report.check(supported.get("checked_backend") == "cuda-b300", "supported backend identity drift")


def validate_fail_closed(report: Report, artifact: dict[str, Any]) -> None:
    for backend in artifact.get("unsupported_backends", []):
        emitted = run_json(
            [
                "cargo",
                "run",
                "-p",
                "ds4-gpu",
                "--bin",
                "ds4-backend-replacement-slice",
                "--quiet",
                "--",
                "--slice",
                "m13.2",
                "--backend",
                backend,
            ],
            expected_code=2,
        )
        report.check(emitted.get("schema") == "ds4.backend_replacement_slice.error.v1", f"{backend}: error schema drift")
        report.check(emitted.get("milestone") == "M13.2", f"{backend}: error milestone drift")
        report.check(emitted.get("id") == EXPECTED_ID, f"{backend}: error id drift")
        report.check(emitted.get("backend_check") == "unsupported", f"{backend}: unsupported marker drift")
        report.check(emitted.get("requested_backend") == backend, f"{backend}: requested backend drift")
        report.check(emitted.get("supported_backends") == EXPECTED_SUPPORTED, f"{backend}: supported list drift")
        report.check(emitted.get("unsupported_backends") == EXPECTED_UNSUPPORTED, f"{backend}: unsupported list drift")


def run_dependency_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/check_backend_expansion_decision.py", "--negative-test"],
        ["ds4-parity/check_backend_expansion_matrix.py", "--negative-test"],
        ["ds4-parity/compare_prefill_whole_short.py", "--negative-test"],
        ["ds4-parity/compare_prefill_chunked.py", "--negative-test"],
        ["ds4-parity/compare_prefill_resumed.py", "--negative-test"],
    ]
    for command in commands:
        proc = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        report.check(proc.returncode == 0, f"{command[0]} failed: {proc.stderr or proc.stdout}")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M13.2 Batched embedding replacement slice" in texts["report"], "unified report wiring missing")
    report.check("check_backend_batched_embedding_slice.py" in texts["report"], "report checker path missing")
    report.check("Validate the M13.2 Batched embedding replacement slice" in texts["readme"], "README wiring missing")
    report.check("M13.2: Batched Embedding Replacement Slice" in texts["roadmap"], "roadmap M13.2 missing")
    report.check("M13.3: Indexed Decode Selection Replacement Slice" in texts["roadmap"], "roadmap M13.3 missing")
    report.check("Earlier M13.2 Batched Embedding Replacement Slice" in texts["status"], "status M13.2 previous item missing")
    report.check(
        "Active item: M13" in texts["status"]
        or "Active item: post-M13 roadmap decision" in texts["status"]
        or "Active item: M14" in texts["status"],
        "status M13 active item missing",
    )
    report.check("#### M13.2: Batched Embedding Replacement Slice" in texts["todo"], "TODO M13.2 missing")
    report.check("#### M13.3: Indexed Decode Selection Replacement Slice" in texts["todo"], "TODO M13.3 missing")


def run_negative_tests(
    report: Report,
    artifact: dict[str, Any],
    matrix: dict[str, Any],
    rust_source: str,
    emitter_source: str,
    texts: dict[str, str],
) -> None:
    mutations = [
        ("route overclaim", lambda obj: with_value(obj, "runtime_route_change", True)),
        ("general replacement overclaim", lambda obj: with_value(obj, "general_backend_replacement", True)),
        ("operation drift", lambda obj: with_value(obj, "operation", "ds4_gpu_indexer_topk_tensor")),
        ("output drift", lambda obj: with_value(obj, "output_fields", ["missing_logits"])),
        ("unsupported backend drift", lambda obj: with_value(obj, "unsupported_backends", ["cpu", "metal"])),
        ("wrong next gate", lambda obj: with_value(obj, "next_required_gate", "M13.5")),
        ("wrong matrix comparator", mutate_matrix_comparator),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(
            mutated_report,
            mutate(artifact),
            matrix,
            rust_source,
            emitter_source,
            texts,
            run_commands=False,
        )
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def with_value(artifact: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated[key] = value
    return mutated


def mutate_matrix_comparator(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["comparator"] = "ds4-parity/compare_prefill_chunked.py"
    return mutated


def matrix_row(matrix: dict[str, Any], operation: str) -> dict[str, Any] | None:
    rows = matrix.get("matrix")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("operation") == operation:
            return row
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_json(command: list[str], *, expected_code: int) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != expected_code:
        raise SystemExit(
            f"{' '.join(command)}: expected exit {expected_code}, got {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{' '.join(command)}: invalid JSON output: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{' '.join(command)}: expected JSON object")
    return data


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Batched embedding replacement slice: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
