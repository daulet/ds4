#!/usr/bin/env python3
"""Validate the M12.5 runtime backend route gate."""

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
GATE = ROOT / "ds4-parity/baselines/backend/m12.5/runtime-route-gate.json"
M12_4_SLICE = ROOT / "ds4-parity/baselines/backend/m12.4/replacement-slice.json"
M10_9C = ROOT / "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json"
M10_9D = ROOT / "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json"
M10_9E = ROOT / "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json"
M10_9F = ROOT / "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json"
RUST_MODULE = ROOT / "rust/ds4-gpu/src/backend_route_gate.rs"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_ID = "m12.5-runtime-backend-route-gate"
EXPECTED_SLICE_ID = "m12.4-embedding-and-indexer-embed-token-hc"
EXPECTED_OPERATION = "ds4_gpu_embed_token_hc_tensor"
EXPECTED_METHOD = "embed_token_hc"
EXPECTED_FAMILY = "embedding_and_indexer"
EXPECTED_SUPPORTED = ["cuda-b300"]
EXPECTED_UNSUPPORTED = ["cpu", "metal", "runtime-default-route"]
EXPECTED_ARTIFACTS = [
    "ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json",
    "ds4-parity/baselines/graph/m10.9d/runtime-long-context.json",
    "ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json",
    "ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json",
]
EXPECTED_QUALITY_GATES = [
    "official-vectors",
    "long-context",
    "tool-server",
    "same-session-benchmark",
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
    artifact = load_json(GATE)
    slice_artifact = load_json(M12_4_SLICE)
    runtime_artifacts = {
        "M10.9c": load_json(M10_9C),
        "M10.9d": load_json(M10_9D),
        "M10.9e": load_json(M10_9E),
        "M10.9f": load_json(M10_9F),
    }
    rust_source = read_text(RUST_MODULE)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(
        report,
        artifact,
        slice_artifact,
        runtime_artifacts,
        rust_source,
        texts,
        run_commands=not args.no_commands,
    )
    if args.negative_test:
        run_negative_tests(report, artifact, slice_artifact, runtime_artifacts, rust_source, texts)
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
    slice_artifact: dict[str, Any],
    runtime_artifacts: dict[str, dict[str, Any]],
    rust_source: str,
    texts: dict[str, str],
    *,
    run_commands: bool,
) -> None:
    validate_artifact(report, artifact)
    validate_against_m12_4(report, artifact, slice_artifact)
    validate_runtime_artifacts(report, artifact, runtime_artifacts)
    validate_rust_module(report, artifact, rust_source)
    if run_commands:
        validate_rust_emitter(report, artifact)
        validate_route_decisions(report)
        run_runtime_gate_checkers(report)
    validate_static_wiring(report, texts)


def validate_artifact(report: Report, artifact: dict[str, Any]) -> None:
    expected = {
        "schema": "ds4.backend_runtime_route_gate.v1",
        "milestone": "M12.5",
        "status": "runtime-route-gate",
        "id": EXPECTED_ID,
        "route_selector": "--runtime-backend-route",
        "default_route": "current-backend",
        "opt_in_route": "replacement-slice",
        "selected_slice_id": EXPECTED_SLICE_ID,
        "operation_family": EXPECTED_FAMILY,
        "operation": EXPECTED_OPERATION,
        "method": EXPECTED_METHOD,
        "replacement_slice_artifact": "ds4-parity/baselines/backend/m12.4/replacement-slice.json",
        "runtime_graph_route": "graph",
        "graph_backend": "cuda",
        "benchmark_policy": "same-session-current-c-parity",
        "next_required_gate": "M12.6 Backend Replacement Closure And Removal Decision",
        "route_check": "not-requested",
    }
    for key, value in expected.items():
        report.check(artifact.get(key) == value, f"artifact {key} drift")
    report.check(artifact.get("supported_backends") == EXPECTED_SUPPORTED, "supported backend drift")
    report.check(artifact.get("unsupported_backends") == EXPECTED_UNSUPPORTED, "unsupported backend drift")
    report.check(artifact.get("validation_artifacts") == EXPECTED_ARTIFACTS, "validation artifact drift")
    report.check(artifact.get("quality_gates") == EXPECTED_QUALITY_GATES, "quality gate drift")
    report.check(artifact.get("default_route_unchanged") is True, "default route changed")
    report.check(artifact.get("replacement_route_opt_in") is True, "replacement route not opt-in")
    report.check(
        artifact.get("default_route_replacement_active") is False,
        "default route replacement became active",
    )
    report.check(artifact.get("general_backend_replacement") is False, "general backend replacement overclaim")
    report.check(artifact.get("kernel_replacement") is False, "kernel replacement overclaim")
    for key in ["replacement_slice_artifact", "validation_artifacts"]:
        values = artifact.get(key)
        if isinstance(values, str):
            values = [values]
        report.check(isinstance(values, list), f"artifact path field invalid: {key}")
        if isinstance(values, list):
            for value in values:
                report.check(isinstance(value, str) and (ROOT / value).exists(), f"artifact path missing: {value}")


def validate_against_m12_4(
    report: Report,
    artifact: dict[str, Any],
    slice_artifact: dict[str, Any],
) -> None:
    report.check(slice_artifact.get("milestone") == "M12.4", "M12.4 slice milestone drift")
    report.check(slice_artifact.get("id") == artifact.get("selected_slice_id"), "selected slice id drift")
    report.check(slice_artifact.get("operation_family") == artifact.get("operation_family"), "selected family drift")
    report.check(slice_artifact.get("operation") == artifact.get("operation"), "selected operation drift")
    report.check(slice_artifact.get("method") == artifact.get("method"), "selected method drift")
    report.check(slice_artifact.get("supported_backends") == artifact.get("supported_backends"), "slice backend drift")
    report.check(slice_artifact.get("runtime_route_change") is False, "M12.4 must not pre-change routes")
    report.check(
        slice_artifact.get("next_required_gate") == "M12.5 Runtime Backend Route Gate",
        "M12.4 must point to M12.5",
    )


def validate_runtime_artifacts(
    report: Report,
    artifact: dict[str, Any],
    runtime_artifacts: dict[str, dict[str, Any]],
) -> None:
    for milestone in ["M10.9c", "M10.9d", "M10.9e", "M10.9f"]:
        data = runtime_artifacts[milestone]
        report.check(data.get("milestone") == milestone, f"{milestone}: milestone drift")
        report.check(
            data.get("runtime_graph_route") == artifact.get("runtime_graph_route"),
            f"{milestone}: route drift",
        )
        report.check(data.get("backend") == artifact.get("graph_backend"), f"{milestone}: backend drift")
        model = data.get("model")
        report.check(isinstance(model, dict), f"{milestone}: model metadata missing")
        if isinstance(model, dict):
            report.check(model.get("sha256") == model.get("expected_sha256"), f"{milestone}: model sha drift")
            report.check(model.get("bytes") == 86720111488, f"{milestone}: model byte drift")

    official = runtime_artifacts["M10.9c"]
    rust = require_dict(report, official.get("rust"), "M10.9c.rust")
    report.check(rust.get("exit_code") == 0, "M10.9c Rust official-vector exit drift")
    report.check(official.get("top_k") == 20, "M10.9c top-k drift")
    report.check(official.get("logprob_abs_tolerance") == 4.0, "M10.9c tolerance drift")

    long_context = runtime_artifacts["M10.9d"]
    settings = require_dict(report, long_context.get("settings"), "M10.9d.settings")
    report.check(settings.get("ctx") == 100000, "M10.9d context drift")
    report.check(settings.get("max_tokens") == 350, "M10.9d token drift")

    tool_server = runtime_artifacts["M10.9e"]
    request = require_dict(report, tool_server.get("request"), "M10.9e.request")
    controls = require_dict(report, request.get("controls"), "M10.9e.request.controls")
    report.check(controls.get("expected_tool_name") == "list_files", "M10.9e expected tool drift")
    rust_tool = require_dict(report, tool_server.get("rust"), "M10.9e.rust")
    report.check(rust_tool.get("ok") is True, "M10.9e Rust tool/server gate failed")

    benchmark = runtime_artifacts["M10.9f"]
    quality_gates = benchmark.get("quality_gates")
    report.check(isinstance(quality_gates, list), "M10.9f quality gates missing")
    if isinstance(quality_gates, list):
        report.check(len(quality_gates) == 5, "M10.9f quality gate count drift")
        for gate in quality_gates:
            report.check(isinstance(gate, dict) and gate.get("ok") is True, "M10.9f quality gate failed")
    performance = require_dict(report, benchmark.get("performance"), "M10.9f.performance")
    report.check(performance.get("same_session_current_c") == "pass", "same-session current-C policy drift")
    report.check(performance.get("same_session_regressions") == [], "same-session benchmark regression drift")
    report.check(performance.get("m0_6_threshold") == "documented_regression", "M0.6 policy drift")
    claim_boundary = require_dict(report, benchmark.get("claim_boundary"), "M10.9f.claim_boundary")
    report.check(claim_boundary.get("backend_replacement") is False, "M10.9f backend replacement overclaim")


def validate_rust_module(report: Report, artifact: dict[str, Any], rust_source: str) -> None:
    for needle in [
        "FIRST_BACKEND_RUNTIME_ROUTE_GATE",
        "RuntimeBackendRoute",
        "route_decision",
        "default_route_replacement_active: false",
        "replacement_route_opt_in: true",
        "general_backend_replacement: false",
        "kernel_replacement: false",
        artifact["id"],
        artifact["selected_slice_id"],
        artifact["operation"],
        artifact["next_required_gate"],
    ]:
        report.check(needle in rust_source, f"Rust route gate missing {needle}")


def validate_rust_emitter(report: Report, artifact: dict[str, Any]) -> None:
    emitted = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
        ],
        expected_code=0,
    )
    report.check(emitted == artifact, "Rust route gate emitter drift")


def validate_route_decisions(report: Report) -> None:
    replacement = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--route",
            "replacement-slice",
            "--backend",
            "cuda-b300",
        ],
        expected_code=0,
    )
    report.check(replacement.get("route_check") == "supported", "replacement route check drift")
    report.check(replacement.get("checked_route") == "replacement-slice", "replacement route identity drift")
    report.check(replacement.get("checked_backend") == "cuda-b300", "replacement backend identity drift")
    report.check(replacement.get("decision_replacement_active") is True, "replacement route inactive")

    default = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--route",
            "current-backend",
            "--backend",
            "cuda-b300",
        ],
        expected_code=0,
    )
    report.check(default.get("checked_route") == "current-backend", "default route identity drift")
    report.check(default.get("decision_replacement_active") is False, "default route activated replacement")

    unsupported_backend = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--route",
            "replacement-slice",
            "--backend",
            "cpu",
        ],
        expected_code=3,
    )
    report.check(
        unsupported_backend.get("schema") == "ds4.backend_runtime_route_gate.error.v1",
        "unsupported backend schema drift",
    )
    report.check(unsupported_backend.get("route_check") == "unsupported-backend", "unsupported backend marker drift")
    report.check(unsupported_backend.get("requested_backend") == "cpu", "unsupported backend identity drift")

    unsupported_route = run_json(
        [
            "cargo",
            "run",
            "-p",
            "ds4-gpu",
            "--bin",
            "ds4-backend-route-gate",
            "--quiet",
            "--",
            "--route",
            "target-stream",
        ],
        expected_code=2,
    )
    report.check(unsupported_route.get("route_check") == "unsupported-route", "unsupported route marker drift")
    report.check(unsupported_route.get("requested_route") == "target-stream", "unsupported route identity drift")


def run_runtime_gate_checkers(report: Report) -> None:
    commands = [
        ["ds4-parity/run_runtime_graph_official_vectors.py", "--negative-test"],
        ["ds4-parity/run_runtime_graph_long_context.py", "--negative-test"],
        ["ds4-parity/run_tool_call_quality.py", "--negative-test"],
        ["ds4-parity/run_runtime_graph_bench.py", "--negative-test"],
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
    report.check("M12.5 Backend runtime route gate" in texts["report"], "unified report wiring missing")
    report.check("check_backend_runtime_route_gate.py" in texts["report"], "report checker path missing")
    report.check("Validate the M12.5 Backend runtime route gate" in texts["readme"], "README wiring missing")
    report.check("M12.5: Runtime Backend Route Gate" in texts["roadmap"], "roadmap M12.5 missing")
    report.check("- Status: complete." in texts["roadmap"], "roadmap M12.5 complete status missing")
    report.check(
        "#### M12.6: Backend Replacement Closure And Removal Decision" in texts["roadmap"],
        "roadmap M12.6 missing",
    )
    report.check(
        "Active item: M12.6 Backend Replacement Closure And Removal Decision" in texts["status"]
        or "Active item: post-M12 roadmap decision" in texts["status"]
        or "Active item: M13" in texts["status"],
        "status active item missing",
    )
    report.check("check_backend_replacement_closure.py" in texts["report"], "M12.6 report checker missing")
    report.check("Earlier M12.5 Runtime Backend Route Gate" in texts["status"], "status previous item missing")
    report.check("#### M12.5: Runtime Backend Route Gate" in texts["todo"], "TODO M12.5 missing")
    report.check(
        "#### M12.6: Backend Replacement Closure And Removal Decision" in texts["todo"],
        "TODO M12.6 missing",
    )


def run_negative_tests(
    report: Report,
    artifact: dict[str, Any],
    slice_artifact: dict[str, Any],
    runtime_artifacts: dict[str, dict[str, Any]],
    rust_source: str,
    texts: dict[str, str],
) -> None:
    mutations = [
        ("default route active", lambda obj: with_value(obj, "default_route_replacement_active", True)),
        ("replacement not opt-in", lambda obj: with_value(obj, "replacement_route_opt_in", False)),
        ("backend replacement overclaim", lambda obj: with_value(obj, "general_backend_replacement", True)),
        ("operation drift", lambda obj: with_value(obj, "operation", "ds4_gpu_indexer_topk_tensor")),
        ("missing official gate", remove_first_validation_artifact),
        ("benchmark policy drift", lambda obj: with_value(obj, "benchmark_policy", "m0.6-only")),
        ("wrong next gate", lambda obj: with_value(obj, "next_required_gate", "M13")),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(
            mutated_report,
            mutate(artifact),
            slice_artifact,
            runtime_artifacts,
            rust_source,
            texts,
            run_commands=False,
        )
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def with_value(artifact: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated[key] = value
    return mutated


def remove_first_validation_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(artifact)
    mutated["validation_artifacts"] = mutated["validation_artifacts"][1:]
    return mutated


def require_dict(report: Report, obj: Any, label: str) -> dict[str, Any]:
    report.check(isinstance(obj, dict), f"{label} must be object")
    return obj if isinstance(obj, dict) else {}


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
    print(f"Backend runtime route gate: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
