#!/usr/bin/env python3
"""Validate the M12.4 first backend replacement slice."""

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
SLICE = ROOT / "ds4-parity/baselines/backend/m12.4/replacement-slice.json"
M12_2_MANIFEST = ROOT / "ds4-parity/baselines/backend/m12.2/manifest.json"
M12_3_REPLAY = ROOT / "ds4-parity/baselines/backend/m12.3/facade-replay.json"
RUST_MODULE = ROOT / "rust/ds4-gpu/src/replacement_slice.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_ID = "m12.4-embedding-and-indexer-embed-token-hc"
EXPECTED_FIXTURE_ID = "first_kernel_embed_token_hc"
EXPECTED_OPERATION = "ds4_gpu_embed_token_hc_tensor"
EXPECTED_METHOD = "embed_token_hc"
EXPECTED_FAMILY = "embedding_and_indexer"
EXPECTED_SUPPORTED = ["cuda-b300"]
EXPECTED_UNSUPPORTED = ["cpu", "metal", "runtime-default-route"]


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
    manifest = load_json(M12_2_MANIFEST)
    replay = load_json(M12_3_REPLAY)
    rust_source = read_text(RUST_MODULE)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, artifact, manifest, replay, rust_source, texts, run_commands=not args.no_commands)
    if args.negative_test:
        run_negative_tests(report, artifact, manifest, replay, rust_source, texts)
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
    manifest: dict[str, Any],
    replay: dict[str, Any],
    rust_source: str,
    texts: dict[str, str],
    *,
    run_commands: bool,
) -> None:
    validate_artifact(report, artifact)
    validate_against_m12_2(report, artifact, manifest, run_comparator=run_commands)
    validate_against_m12_3(report, artifact, replay)
    validate_rust_module(report, artifact, rust_source)
    if run_commands:
        validate_rust_emitter(report, artifact)
        validate_fail_closed(report, artifact)
    validate_static_wiring(report, texts)


def validate_artifact(report: Report, artifact: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_replacement_slice.v1",
        "milestone": "M12.4",
        "status": "first-replacement-slice",
        "id": EXPECTED_ID,
        "operation_family": EXPECTED_FAMILY,
        "fixture_id": EXPECTED_FIXTURE_ID,
        "operation": EXPECTED_OPERATION,
        "method": EXPECTED_METHOD,
        "rust_module": "rust/ds4-gpu/src/replacement_slice.rs",
        "facade_replay": "ds4-parity/baselines/backend/m12.3/facade-replay.json",
        "tensor_fixture_manifest": "ds4-parity/baselines/backend/m12.2/manifest.json",
        "comparator": "ds4-parity/compare_decode_first_kernel_oracle.py",
        "next_required_gate": "M12.5 Runtime Backend Route Gate",
        "backend_check": "not-requested",
    }
    for key, value in expected.items():
        report.check(artifact.get(key) == value, f"artifact {key} drift")
    report.check(artifact.get("output_fields") == ["cur_hc"], "output field drift")
    report.check(artifact.get("supported_backends") == EXPECTED_SUPPORTED, "supported backend drift")
    report.check(artifact.get("unsupported_backends") == EXPECTED_UNSUPPORTED, "unsupported backend drift")
    report.check(artifact.get("runtime_route_change") is False, "runtime route overclaim")
    report.check(artifact.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(artifact.get("kernel_replacement") is False, "kernel replacement overclaim")
    for key in ["rust_module", "facade_replay", "tensor_fixture_manifest", "comparator"]:
        value = artifact.get(key)
        report.check(isinstance(value, str) and (ROOT / value).exists(), f"artifact path missing: {key}")


def validate_against_m12_2(
    report: Report,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    *,
    run_comparator: bool,
) -> None:
    fixture = fixture_by_id(manifest, artifact.get("fixture_id"))
    report.check(fixture is not None, "selected fixture missing from M12.2")
    if fixture is None:
        return
    report.check(fixture.get("operation_family") == artifact.get("operation_family"), "M12.2 family drift")
    report.check(fixture.get("operations") == [artifact.get("operation")], "M12.2 operation drift")
    report.check(fixture.get("comparator") == artifact.get("comparator"), "M12.2 comparator drift")
    report.check(fixture.get("output_fields") == artifact.get("output_fields"), "M12.2 output field drift")
    report.check("cuda-backend" in fixture.get("rerun_command", ""), "M12.2 B300 CUDA rerun command missing")
    if run_comparator:
        run_fixture_comparator(report, fixture)


def validate_against_m12_3(report: Report, artifact: dict[str, Any], replay: dict[str, Any]) -> None:
    replay_entry = replay_by_fixture_id(replay, artifact.get("fixture_id"))
    report.check(replay_entry is not None, "selected fixture missing from M12.3 replay")
    if replay_entry is None:
        return
    report.check(replay_entry.get("operation_family") == artifact.get("operation_family"), "M12.3 family drift")
    report.check(replay_entry.get("comparator") == artifact.get("comparator"), "M12.3 comparator drift")
    report.check(replay_entry.get("output_fields") == artifact.get("output_fields"), "M12.3 output field drift")
    calls = replay_entry.get("calls")
    report.check(isinstance(calls, list) and len(calls) == 1, "M12.3 selected replay call drift")
    if isinstance(calls, list) and calls:
        call = calls[0]
        report.check(call.get("operation") == artifact.get("operation"), "M12.3 operation drift")
        report.check(call.get("method") == artifact.get("method"), "M12.3 method drift")
        report.check(call.get("tensor_args") == ["out_hc"], "M12.3 tensor arg drift")


def validate_rust_module(report: Report, artifact: dict[str, Any], rust_source: str) -> None:
    for needle in [
        "FIRST_BACKEND_REPLACEMENT_SLICE",
        artifact["id"],
        artifact["operation_family"],
        artifact["fixture_id"],
        artifact["operation"],
        artifact["method"],
        "runtime_route_change: false",
        "general_backend_replacement: false",
        "kernel_replacement: false",
        "ensure_supported_backend",
        "UnsupportedReplacementBackend",
    ]:
        report.check(needle in rust_source, f"Rust replacement slice missing {needle}")


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
        ],
        expected_code=0,
    )
    report.check(emitted == artifact, "Rust replacement slice emitter drift")
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
                "--backend",
                backend,
            ],
            expected_code=2,
        )
        report.check(emitted.get("schema") == "ds4.backend_replacement_slice.error.v1", f"{backend}: error schema drift")
        report.check(emitted.get("backend_check") == "unsupported", f"{backend}: unsupported marker drift")
        report.check(emitted.get("requested_backend") == backend, f"{backend}: requested backend drift")
        report.check(emitted.get("supported_backends") == EXPECTED_SUPPORTED, f"{backend}: supported list drift")
        report.check(emitted.get("unsupported_backends") == EXPECTED_UNSUPPORTED, f"{backend}: unsupported list drift")


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M12.4 Backend replacement slice" in texts["report"], "unified report wiring missing")
    report.check("check_backend_replacement_slice.py" in texts["report"], "report checker path missing")
    report.check("Validate the M12.4 Backend replacement slice" in texts["readme"], "README wiring missing")
    report.check("M12.4: First Backend Replacement Slice" in texts["roadmap"], "roadmap M12.4 missing")
    report.check("- Status: complete." in texts["roadmap"], "roadmap complete status missing")
    report.check("#### M12.5: Runtime Backend Route Gate" in texts["roadmap"], "roadmap M12.5 missing")
    report.check(
        "#### M12.6: Backend Replacement Closure And Removal Decision" in texts["roadmap"],
        "roadmap M12.6 missing",
    )
    report.check(
        "Active item: M12.5 Runtime Backend Route Gate" in texts["status"]
        or "Active item: M12.6 Backend Replacement Closure And Removal Decision" in texts["status"]
        or "Active item: post-M12 roadmap decision" in texts["status"],
        "status active item missing",
    )
    report.check("Earlier M12.4 First Backend Replacement Slice" in texts["status"], "status previous item missing")
    report.check("#### M12.4: First Backend Replacement Slice" in texts["todo"], "TODO M12.4 missing")
    report.check("#### M12.5: Runtime Backend Route Gate" in texts["todo"], "TODO M12.5 missing")


def run_negative_tests(
    report: Report,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    replay: dict[str, Any],
    rust_source: str,
    texts: dict[str, str],
) -> None:
    mutations = [
        ("route overclaim", mutate_route_claim),
        ("general replacement overclaim", mutate_general_replacement_claim),
        ("operation drift", mutate_operation),
        ("output drift", mutate_output),
        ("unsupported backend drift", mutate_unsupported),
        ("wrong next gate", mutate_next_gate),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(
            mutated_report,
            mutate(artifact),
            manifest,
            replay,
            rust_source,
            texts,
            run_commands=False,
        )
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_route_claim(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["runtime_route_change"] = True
    return mutated


def mutate_general_replacement_claim(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["general_backend_replacement"] = True
    return mutated


def mutate_operation(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["operation"] = "ds4_gpu_indexer_topk_tensor"
    return mutated


def mutate_output(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["output_fields"] = ["missing_cur_hc"]
    return mutated


def mutate_unsupported(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["unsupported_backends"] = ["cpu", "metal"]
    return mutated


def mutate_next_gate(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["next_required_gate"] = "M12.6 Current Backend Closure Matrix"
    return mutated


def run_fixture_comparator(report: Report, fixture: dict[str, Any]) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            fixture["comparator"],
            "--oracle",
            fixture["oracle"]["path"],
            "--candidate",
            fixture["candidate"]["path"],
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    report.check(proc.returncode == 0, f"fixture comparator failed: {proc.stderr or proc.stdout}")


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


def fixture_by_id(manifest: dict[str, Any], fixture_id: Any) -> dict[str, Any] | None:
    for fixture in manifest.get("fixtures", []):
        if isinstance(fixture, dict) and fixture.get("id") == fixture_id:
            return fixture
    return None


def replay_by_fixture_id(replay: dict[str, Any], fixture_id: Any) -> dict[str, Any] | None:
    for entry in replay.get("replays", []):
        if isinstance(entry, dict) and entry.get("fixture_id") == fixture_id:
            return entry
    return None


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Backend replacement slice: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
