#!/usr/bin/env python3
"""Validate the M13.3 indexed decode selection replacement slices."""

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
SLICE_SET = ROOT / "ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json"
M13_1_MATRIX = ROOT / "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json"
RUST_MODULE = ROOT / "rust/ds4-gpu/src/replacement_slice.rs"
EMITTER = ROOT / "rust/ds4-gpu/src/bin/ds4-backend-replacement-slice.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_FAMILY = "embedding_and_indexer"
EXPECTED_COMPARATOR = "ds4-parity/compare_decode_long_indexed_attention.py"
EXPECTED_SUPPORTED = ["cuda-b300"]
EXPECTED_UNSUPPORTED = ["cpu", "metal", "runtime-default-route"]
EXPECTED_NEXT_GATE = "M13.4 Batch Indexer Fixture Gap Closure"
EXPECTED_SLICES = [
    {
        "alias": "indexer-score-one",
        "id": "m13.3-embedding-and-indexer-indexer-score-one",
        "fixture_id": "m13.1-indexer-score-one",
        "operation": "ds4_gpu_indexer_score_one_tensor",
        "method": "indexer_score_one",
        "output_fields": ["layer2_indexer_scores"],
    },
    {
        "alias": "indexer-topk",
        "id": "m13.3-embedding-and-indexer-indexer-topk",
        "fixture_id": "m13.1-indexer-topk",
        "operation": "ds4_gpu_indexer_topk_tensor",
        "method": "indexer_topk",
        "output_fields": ["layer2_comp_selected"],
    },
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
    artifact = load_json(SLICE_SET)
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
    validate_set_artifact(report, artifact)
    validate_slice_artifacts(report, artifact)
    validate_against_m13_1(report, artifact, matrix)
    validate_rust_module(report, artifact, rust_source, emitter_source)
    if run_commands:
        validate_rust_emitter(report, artifact)
        validate_fail_closed(report, artifact)
        validate_ambiguous_milestone_rejected(report)
        run_dependency_checkers(report)
    validate_static_wiring(report, texts)


def validate_set_artifact(report: Report, artifact: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_replacement_slice_set.v1",
        "milestone": "M13.3",
        "status": "indexed-decode-selection-replacement-slices",
        "id": "m13.3-embedding-and-indexer-indexed-decode-selection",
        "operation_family": EXPECTED_FAMILY,
        "source_matrix": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
        "comparator": EXPECTED_COMPARATOR,
        "next_required_gate": EXPECTED_NEXT_GATE,
    }
    for key, value in expected.items():
        report.check(artifact.get(key) == value, f"slice-set {key} drift")
    report.check(artifact.get("slice_ids") == [item["id"] for item in EXPECTED_SLICES], "slice id list drift")
    report.check(artifact.get("runtime_route_change") is False, "runtime route overclaim")
    report.check(artifact.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(artifact.get("kernel_replacement") is False, "kernel replacement overclaim")
    report.check(artifact.get("default_route_replacement_active") is False, "default route replacement overclaim")
    for key in ["source_matrix", "comparator"]:
        value = artifact.get(key)
        report.check(isinstance(value, str) and (ROOT / value).exists(), f"slice-set path missing: {key}")


def validate_slice_artifacts(report: Report, artifact: dict[str, Any]) -> None:
    slices = artifact_slices(artifact)
    report.check(len(slices) == 2, "slice-set should contain exactly two slices")
    for expected in EXPECTED_SLICES:
        item = slices.get(expected["id"])
        report.check(item is not None, f"slice missing: {expected['id']}")
        if item is None:
            continue
        fields = {
            "schema": "ds4.backend_replacement_slice.v1",
            "milestone": "M13.3",
            "status": "indexed-decode-selection-replacement-slice",
            "id": expected["id"],
            "operation_family": EXPECTED_FAMILY,
            "fixture_id": expected["fixture_id"],
            "operation": expected["operation"],
            "method": expected["method"],
            "rust_module": "rust/ds4-gpu/src/replacement_slice.rs",
            "facade_replay": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
            "tensor_fixture_manifest": "ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json",
            "comparator": EXPECTED_COMPARATOR,
            "next_required_gate": EXPECTED_NEXT_GATE,
            "backend_check": "not-requested",
        }
        for key, value in fields.items():
            report.check(item.get(key) == value, f"{expected['id']}: {key} drift")
        report.check(item.get("output_fields") == expected["output_fields"], f"{expected['id']}: output drift")
        report.check(item.get("supported_backends") == EXPECTED_SUPPORTED, f"{expected['id']}: supported backend drift")
        report.check(item.get("unsupported_backends") == EXPECTED_UNSUPPORTED, f"{expected['id']}: unsupported backend drift")
        report.check(item.get("runtime_route_change") is False, f"{expected['id']}: runtime route overclaim")
        report.check(item.get("general_backend_replacement") is False, f"{expected['id']}: general replacement overclaim")
        report.check(item.get("kernel_replacement") is False, f"{expected['id']}: kernel replacement overclaim")
        for key in ["rust_module", "facade_replay", "tensor_fixture_manifest", "comparator"]:
            value = item.get(key)
            report.check(isinstance(value, str) and (ROOT / value).exists(), f"{expected['id']}: path missing: {key}")


def validate_against_m13_1(report: Report, artifact: dict[str, Any], matrix: dict[str, Any]) -> None:
    report.check(matrix.get("milestone") == "M13.1", "M13.1 matrix milestone drift")
    policy = matrix.get("claim_policy", {})
    report.check(policy.get("runtime_route_change") is False, "M13.1 route-change policy drift")
    report.check(policy.get("default_route_replacement_active") is False, "M13.1 default-route policy drift")
    report.check(policy.get("general_backend_replacement") is False, "M13.1 general replacement policy drift")
    report.check(policy.get("kernel_replacement") is False, "M13.1 kernel replacement policy drift")
    report.check(policy.get("removals_allowed") is False, "M13.1 removal policy drift")
    for expected in EXPECTED_SLICES:
        row = matrix_row(matrix, expected["operation"])
        item = artifact_slices(artifact).get(expected["id"])
        report.check(row is not None, f"M13.1 row missing: {expected['operation']}")
        report.check(item is not None, f"artifact row missing: {expected['id']}")
        if row is None or item is None:
            continue
        report.check(row.get("coverage_level") == "pair-comparator-ready", f"{expected['operation']}: coverage drift")
        report.check(row.get("fixture_gap") is False, f"{expected['operation']}: fixture gap drift")
        report.check(row.get("rust_safe_wrapper") is True, f"{expected['operation']}: Rust wrapper drift")
        report.check(row.get("method") == item.get("method"), f"{expected['operation']}: method drift")
        report.check(row.get("route_candidate_stage") == "M13.3 Indexed Decode Selection Replacement Slice", f"{expected['operation']}: route stage drift")
        report.check(row.get("comparators") == [EXPECTED_COMPARATOR], f"{expected['operation']}: comparator drift")
        report.check(row.get("covered_outputs") == item.get("output_fields"), f"{expected['operation']}: output drift")
        report.check(row.get("route_status") == "not-route-gated", f"{expected['operation']}: route status drift")
        report.check(row.get("removal_decision") == "retain_current_backend", f"{expected['operation']}: removal decision drift")


def validate_rust_module(
    report: Report,
    artifact: dict[str, Any],
    rust_source: str,
    emitter_source: str,
) -> None:
    for needle in [
        "INDEXER_SCORE_ONE_REPLACEMENT_SLICE",
        "INDEXER_TOPK_REPLACEMENT_SLICE",
        "INDEXED_DECODE_SELECTION_REPLACEMENT_SLICES",
        "indexer_score_one_replacement_slice",
        "indexer_topk_replacement_slice",
        "indexed_decode_selection_replacement_slices",
        "runtime_route_change: false",
        "general_backend_replacement: false",
        "kernel_replacement: false",
    ]:
        report.check(needle in rust_source, f"Rust replacement slice missing {needle}")
    for expected in EXPECTED_SLICES:
        for needle in [
            expected["id"],
            expected["fixture_id"],
            expected["operation"],
            expected["method"],
            expected["output_fields"][0],
            expected["alias"],
        ]:
            report.check(needle in rust_source, f"Rust replacement slice missing {needle}")
    for needle in ["--slice", "replacement_slice_by_id", "spec.status", "usage: ds4-backend-replacement-slice"]:
        report.check(needle in emitter_source, f"Rust emitter missing {needle}")
    report.check("m13.3\" =>" not in rust_source, "ambiguous M13.3 alias should not select a single slice")
    report.check(artifact.get("default_route_replacement_active") is False, "artifact default-route policy drift")


def validate_rust_emitter(report: Report, artifact: dict[str, Any]) -> None:
    slices = artifact_slices(artifact)
    default = run_json(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-backend-replacement-slice", "--quiet"],
        expected_code=0,
    )
    report.check(default.get("milestone") == "M12.4", "default replacement slice drift")
    for expected in EXPECTED_SLICES:
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
                expected["alias"],
            ],
            expected_code=0,
        )
        report.check(emitted == slices.get(expected["id"]), f"{expected['id']}: emitter drift")
        method = run_json(
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
                expected["method"],
            ],
            expected_code=0,
        )
        report.check(method.get("id") == expected["id"], f"{expected['id']}: method selector drift")
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
                expected["alias"],
                "--backend",
                "cuda-b300",
            ],
            expected_code=0,
        )
        report.check(supported.get("backend_check") == "supported", f"{expected['id']}: supported backend check drift")
        report.check(supported.get("checked_backend") == "cuda-b300", f"{expected['id']}: supported backend identity drift")


def validate_fail_closed(report: Report, artifact: dict[str, Any]) -> None:
    slices = artifact_slices(artifact)
    for expected in EXPECTED_SLICES:
        for backend in EXPECTED_UNSUPPORTED:
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
                    expected["alias"],
                    "--backend",
                    backend,
                ],
                expected_code=2,
            )
            spec = slices.get(expected["id"], {})
            report.check(emitted.get("schema") == "ds4.backend_replacement_slice.error.v1", f"{expected['id']} {backend}: error schema drift")
            report.check(emitted.get("milestone") == "M13.3", f"{expected['id']} {backend}: error milestone drift")
            report.check(emitted.get("id") == expected["id"], f"{expected['id']} {backend}: error id drift")
            report.check(emitted.get("backend_check") == "unsupported", f"{expected['id']} {backend}: unsupported marker drift")
            report.check(emitted.get("requested_backend") == backend, f"{expected['id']} {backend}: requested backend drift")
            report.check(emitted.get("supported_backends") == spec.get("supported_backends"), f"{expected['id']} {backend}: supported list drift")
            report.check(emitted.get("unsupported_backends") == spec.get("unsupported_backends"), f"{expected['id']} {backend}: unsupported list drift")


def validate_ambiguous_milestone_rejected(report: Report) -> None:
    proc = run_process(
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
            "m13.3",
        ]
    )
    report.check(proc.returncode == 1, "ambiguous M13.3 selector should fail")
    report.check("unknown replacement slice: m13.3" in proc.stderr, "ambiguous M13.3 error drift")


def run_dependency_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/check_backend_expansion_decision.py", "--negative-test"],
        ["ds4-parity/check_backend_expansion_matrix.py", "--negative-test"],
        ["ds4-parity/check_backend_batched_embedding_slice.py", "--negative-test"],
        ["ds4-parity/compare_decode_long_indexed_attention.py", "--negative-test"],
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
    report.check("M13.3 Indexed decode selection replacement slice" in texts["report"], "unified report wiring missing")
    report.check("check_backend_indexed_decode_slice.py" in texts["report"], "report checker path missing")
    report.check("Validate the M13.3 Indexed decode selection replacement slice" in texts["readme"], "README wiring missing")
    report.check("M13.3: Indexed Decode Selection Replacement Slice" in texts["roadmap"], "roadmap M13.3 missing")
    report.check("M13.4: Batch Indexer Fixture Gap Closure" in texts["roadmap"], "roadmap M13.4 missing")
    report.check("Earlier M13.3 Indexed Decode Selection Replacement Slice" in texts["status"], "status M13.3 previous item missing")
    report.check(
        "Active item: M13" in texts["status"]
        or "Active item: post-M13 roadmap decision" in texts["status"]
        or "Active item: M14" in texts["status"],
        "status M13 active item missing",
    )
    report.check("#### M13.3: Indexed Decode Selection Replacement Slice" in texts["todo"], "TODO M13.3 missing")
    report.check("#### M13.4: Batch Indexer Fixture Gap Closure" in texts["todo"], "TODO M13.4 missing")


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
        ("default route overclaim", lambda obj: with_value(obj, "default_route_replacement_active", True)),
        ("missing slice", drop_last_slice),
        ("operation drift", lambda obj: mutate_slice_value(obj, 0, "operation", "ds4_gpu_indexer_topk_tensor")),
        ("output drift", lambda obj: mutate_slice_value(obj, 1, "output_fields", ["missing_comp_selected"])),
        ("unsupported backend drift", lambda obj: mutate_slice_value(obj, 0, "unsupported_backends", ["cpu", "metal"])),
        ("wrong next gate", lambda obj: mutate_slice_value(obj, 1, "next_required_gate", "M13.5")),
        ("wrong comparator", lambda obj: mutate_slice_value(obj, 0, "comparator", "ds4-parity/compare_decode_first_kernel.py")),
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


def mutate_slice_value(artifact: dict[str, Any], index: int, key: str, value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["slices"][index][key] = value
    return mutated


def drop_last_slice(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["slices"] = mutated["slices"][:-1]
    return mutated


def artifact_slices(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_slices = artifact.get("slices")
    if not isinstance(raw_slices, list):
        return {}
    slices = {}
    for item in raw_slices:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            slices[item["id"]] = item
    return slices


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
    proc = run_process(command)
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


def run_process(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Indexed decode selection replacement slices: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
