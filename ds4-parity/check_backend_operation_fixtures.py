#!/usr/bin/env python3
"""Validate the M12.2 backend operation tensor fixture bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ds4-parity/baselines/backend/m12.2/manifest.json"
BOUNDARY_INVENTORY = ROOT / "ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json"
SHA_LOG = ROOT / "ds4-parity/baselines/backend/m12.2/captures/sha256sums.txt"
SIZE_LOG = ROOT / "ds4-parity/baselines/backend/m12.2/captures/sizes.txt"
README = ROOT / "ds4-parity/README.md"
REPORT = ROOT / "ds4-parity/run_parity_report.py"
ROADMAP = ROOT / "RUST_PORT_ROADMAP.md"
TODO = ROOT / ".memory/TODO.md"
STATUS = ROOT / ".memory/status.md"

EXPECTED_FAMILIES = [
    "embedding_and_indexer",
    "dense_norm_rope_kv",
    "compressor_attention",
    "routing_moe",
    "hc_output",
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
    manifest = load_json(MANIFEST)
    boundary = load_json(BOUNDARY_INVENTORY)
    texts = {
        "readme": read_text(README),
        "report": read_text(REPORT),
        "roadmap": read_text(ROADMAP),
        "todo": read_text(TODO),
        "status": read_text(STATUS),
    }

    report = Report()
    validate(report, manifest, boundary, texts, run_pair_comparators=not args.no_pair_comparators)
    if args.negative_test:
        run_negative_tests(report, manifest, boundary, texts)
    print_report(report)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    parser.add_argument("--no-pair-comparators", action="store_true")
    return parser.parse_args(list(argv))


def validate(
    report: Report,
    manifest: dict[str, Any],
    boundary: dict[str, Any],
    texts: dict[str, str],
    *,
    run_pair_comparators: bool,
    capture_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    report.check(manifest.get("schema") == "ds4.backend_operation_fixture_bundle.v1", "schema drift")
    report.check(manifest.get("milestone") == "M12.2", "milestone drift")
    report.check(manifest.get("status") == "current-backend-fixture-capture", "status drift")
    report.check(manifest.get("next_stage") == "M12.3", "next stage must be M12.3")
    validate_environment(report, manifest)
    validate_claim_policy(report, manifest)
    validate_comparison_policy(report, manifest)
    validate_fixtures(
        report,
        manifest,
        boundary,
        run_pair_comparators=run_pair_comparators,
        capture_overrides=capture_overrides or {},
    )
    validate_logs(report, manifest)
    validate_static_wiring(report, texts)


def validate_environment(report: Report, manifest: dict[str, Any]) -> None:
    env = manifest.get("capture_environment")
    report.check(isinstance(env, dict), "capture environment missing")
    if not isinstance(env, dict):
        return
    expected = {
        "context": "hou2-prod1",
        "namespace": "default",
        "pod": "ds4-rust-port-b300",
        "workdir": "/workspace/ds4",
        "temp_kubeconfig": "/tmp/ds4-hou2-prod1.kubeconfig",
        "backend": "cuda",
        "model_sha256": "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668",
        "model_bytes": 86720111488,
    }
    for key, value in expected.items():
        report.check(env.get(key) == value, f"capture environment {key} drift")
    source_commit = env.get("source_commit")
    report.check(isinstance(source_commit, str) and len(source_commit) == 40, "source commit missing")


def validate_claim_policy(report: Report, manifest: dict[str, Any]) -> None:
    policy = manifest.get("claim_policy")
    report.check(isinstance(policy, dict), "claim policy missing")
    if not isinstance(policy, dict):
        return
    report.check(policy.get("backend_replacement") == "not_claimed", "backend replacement overclaim")
    report.check(policy.get("runtime_route_change") is False, "runtime route change overclaim")
    report.check(policy.get("raw_tensor_bodies_committed") is False, "raw tensor bodies must not be committed")
    report.check(policy.get("next_required_gate") == "M12.3 Rust Backend Facade Parity Harness", "next gate drift")


def validate_comparison_policy(report: Report, manifest: dict[str, Any]) -> None:
    policy = manifest.get("comparison_policy")
    report.check(isinstance(policy, dict), "comparison policy missing")
    if not isinstance(policy, dict):
        return
    for key in ["digest", "samples", "nonzero_elements", "shape", "dtype", "float_tolerance"]:
        report.check(isinstance(policy.get(key), str) and policy[key], f"comparison policy {key} missing")


def validate_fixtures(
    report: Report,
    manifest: dict[str, Any],
    boundary: dict[str, Any],
    *,
    run_pair_comparators: bool,
    capture_overrides: dict[str, dict[str, Any]],
) -> None:
    fixtures = manifest.get("fixtures")
    report.check(isinstance(fixtures, list) and len(fixtures) == 5, "fixture count drift")
    if not isinstance(fixtures, list):
        return

    families, operations = boundary_sets(boundary)
    seen_ids: set[str] = set()
    got_families: list[str] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            report.check(False, "fixture entry is not an object")
            continue
        fixture_id = fixture.get("id")
        report.check(isinstance(fixture_id, str) and fixture_id, "fixture id missing")
        report.check(fixture_id not in seen_ids, f"duplicate fixture id {fixture_id}")
        if isinstance(fixture_id, str):
            seen_ids.add(fixture_id)

        family = fixture.get("operation_family")
        got_families.append(str(family))
        report.check(family in families, f"{fixture_id}: unknown operation family {family}")
        fixture_ops = fixture.get("operations")
        report.check(isinstance(fixture_ops, list) and fixture_ops, f"{fixture_id}: operations missing")
        if isinstance(fixture_ops, list):
            report.check(
                all(isinstance(op, str) and op in operations for op in fixture_ops),
                f"{fixture_id}: operation outside M12.1 inventory",
            )

        comparator = fixture.get("comparator")
        report.check(isinstance(comparator, str) and comparator.startswith("ds4-parity/"), f"{fixture_id}: comparator missing")
        if isinstance(comparator, str):
            report.check((ROOT / comparator).exists(), f"{fixture_id}: comparator path missing")

        command = fixture.get("rerun_command")
        validate_rerun_command(report, fixture_id, command, comparator)

        oracle = validate_capture_ref(report, fixture_id, fixture.get("oracle"), capture_overrides)
        candidate = validate_capture_ref(report, fixture_id, fixture.get("candidate"), capture_overrides)
        validate_capture_pair(report, fixture, oracle, candidate)
        if run_pair_comparators and isinstance(comparator, str):
            run_pair_comparator(report, fixture_id, comparator, fixture)

    report.check(got_families == EXPECTED_FAMILIES, f"selected family order drift: {got_families!r}")


def validate_rerun_command(report: Report, fixture_id: Any, command: Any, comparator: Any) -> None:
    report.check(isinstance(command, str) and command, f"{fixture_id}: rerun command missing")
    if not isinstance(command, str):
        return
    for needle in [
        "git archive HEAD",
        "kubectl",
        "--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig",
        "--context hou2-prod1",
        "ds4-rust-port-b300",
        "cd /workspace/ds4",
        "CUDA_ARCH=native",
        "--features cuda-backend",
    ]:
        report.check(needle in command, f"{fixture_id}: rerun command missing {needle}")
    if isinstance(comparator, str):
        report.check(comparator in command, f"{fixture_id}: rerun command missing comparator")
    report.check("--runtime-graph graph" not in command, f"{fixture_id}: M12.2 must not change runtime route")


def validate_capture_ref(
    report: Report,
    fixture_id: Any,
    ref: Any,
    capture_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    report.check(isinstance(ref, dict), f"{fixture_id}: capture ref missing")
    if not isinstance(ref, dict):
        return None
    path_text = ref.get("path")
    report.check(isinstance(path_text, str) and path_text, f"{fixture_id}: capture path missing")
    report.check(isinstance(ref.get("schema"), str) and ref["schema"], f"{fixture_id}: schema missing")
    report.check(isinstance(ref.get("sha256"), str) and len(ref["sha256"]) == 64, f"{fixture_id}: sha256 missing")
    report.check(isinstance(ref.get("bytes"), int) and ref["bytes"] > 0, f"{fixture_id}: byte size missing")
    if not isinstance(path_text, str):
        return None
    path = ROOT / path_text
    report.check(path.exists(), f"{fixture_id}: capture file missing {path_text}")
    if path.exists():
        report.check(path.stat().st_size == ref.get("bytes"), f"{fixture_id}: byte size drift for {path_text}")
        report.check(sha256_file(path) == ref.get("sha256"), f"{fixture_id}: sha256 drift for {path_text}")
    obj = capture_overrides.get(path_text)
    if obj is None and path.exists():
        obj = load_json(path)
    if obj is not None:
        report.check(obj.get("schema") == ref.get("schema"), f"{fixture_id}: capture schema drift")
    return obj


def validate_capture_pair(
    report: Report,
    fixture: dict[str, Any],
    oracle: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> None:
    fixture_id = fixture.get("id")
    if oracle is None or candidate is None:
        return
    report.check(oracle.get("case") == fixture.get("case"), f"{fixture_id}: oracle case drift")
    report.check(candidate.get("case") == fixture.get("case"), f"{fixture_id}: candidate case drift")
    if "source" in oracle:
        report.check(oracle.get("source") == "current-c", f"{fixture_id}: oracle source drift")

    fields = fixture.get("output_fields")
    report.check(isinstance(fields, list) and fields, f"{fixture_id}: output fields missing")
    if not isinstance(fields, list):
        return
    oracle_outputs = capture_outputs(oracle)
    candidate_outputs = capture_outputs(candidate)
    report.check(list(oracle_outputs) == fields, f"{fixture_id}: oracle output field order drift")
    report.check(list(candidate_outputs) == fields, f"{fixture_id}: candidate output field order drift")
    for field in fields:
        o = oracle_outputs.get(field)
        c = candidate_outputs.get(field)
        report.check(isinstance(o, dict), f"{fixture_id}: oracle output {field} missing")
        report.check(isinstance(c, dict), f"{fixture_id}: candidate output {field} missing")
        if not isinstance(o, dict) or not isinstance(c, dict):
            continue
        for key in ["fnv1a64", "nonzero_elements", "elements", "bytes"]:
            if key in o or key in c:
                report.check(o.get(key) == c.get(key), f"{fixture_id}: {field} {key} mismatch")
        validate_sample_indices(report, fixture_id, field, o.get("samples"), c.get("samples"))


def capture_outputs(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = obj.get("outputs")
    if isinstance(outputs, dict):
        return {key: value for key, value in outputs.items() if isinstance(value, dict)}
    output = obj.get("output")
    if isinstance(output, dict) and isinstance(output.get("field"), str):
        return {output["field"]: output}
    return {}


def validate_sample_indices(
    report: Report,
    fixture_id: Any,
    field: str,
    oracle_samples: Any,
    candidate_samples: Any,
) -> None:
    if oracle_samples is None and candidate_samples is None:
        return
    report.check(isinstance(oracle_samples, list), f"{fixture_id}: {field} oracle samples missing")
    report.check(isinstance(candidate_samples, list), f"{fixture_id}: {field} candidate samples missing")
    if not isinstance(oracle_samples, list) or not isinstance(candidate_samples, list):
        return
    oracle_indices = [sample.get("index") for sample in oracle_samples if isinstance(sample, dict)]
    candidate_indices = [sample.get("index") for sample in candidate_samples if isinstance(sample, dict)]
    report.check(oracle_indices == candidate_indices, f"{fixture_id}: {field} sample index drift")


def run_pair_comparator(report: Report, fixture_id: Any, comparator: str, fixture: dict[str, Any]) -> None:
    oracle = fixture.get("oracle")
    candidate = fixture.get("candidate")
    if not isinstance(oracle, dict) or not isinstance(candidate, dict):
        return
    command = [
        sys.executable,
        comparator,
        "--oracle",
        str(ROOT / str(oracle.get("path"))),
        "--candidate",
        str(ROOT / str(candidate.get("path"))),
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        report.errors.append(f"{fixture_id}: comparator failed: {proc.stdout.strip()} {proc.stderr.strip()}")
    report.check(proc.returncode == 0, f"{fixture_id}: pair comparator failed")


def validate_logs(report: Report, manifest: dict[str, Any]) -> None:
    sha_entries = parse_two_column_log(report, SHA_LOG)
    size_entries = parse_two_column_log(report, SIZE_LOG, integer_first=True)
    for fixture in manifest.get("fixtures", []):
        if not isinstance(fixture, dict):
            continue
        for ref_key in ["oracle", "candidate"]:
            ref = fixture.get(ref_key)
            if not isinstance(ref, dict):
                continue
            path = ref.get("path")
            report.check(sha_entries.get(path) == ref.get("sha256"), f"{path}: sha log drift")
            report.check(size_entries.get(path) == ref.get("bytes"), f"{path}: size log drift")


def parse_two_column_log(report: Report, path: Path, *, integer_first: bool = False) -> dict[str, Any]:
    report.check(path.exists(), f"{path.relative_to(ROOT)} missing")
    if not path.exists():
        return {}
    entries: dict[str, Any] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.strip().endswith(" total"):
            continue
        parts = line.split()
        report.check(len(parts) == 2, f"{path.relative_to(ROOT)}:{lineno}: malformed line")
        if len(parts) != 2:
            continue
        value, rel = parts
        entries[rel] = int(value) if integer_first else value
    return entries


def validate_static_wiring(report: Report, texts: dict[str, str]) -> None:
    report.check("M12.2 Backend operation tensor fixtures" in texts["readme"], "README M12.2 section missing")
    report.check("M12.2 Backend operation tensor fixtures" in texts["report"], "unified report M12.2 item missing")
    report.check("ds4-parity/check_backend_operation_fixtures.py" in texts["report"], "report command missing checker")
    report.check("M12.2: Operation Tensor Fixture Capture" in texts["roadmap"], "roadmap M12.2 missing")
    report.check(
        "Earlier M12.2 Operation Tensor Fixture Capture" in texts["status"],
        "status missing M12.2 completion",
    )
    report.check(
        "Active item: M12.3 Rust Backend Facade Parity Harness" in texts["status"]
        or "Active item: M12.4 First Backend Replacement Slice" in texts["status"]
        or "Active item: M12.5 Runtime Backend Route Gate" in texts["status"],
        "status active item must advance to M12.3 or later",
    )


def run_negative_tests(
    report: Report,
    manifest: dict[str, Any],
    boundary: dict[str, Any],
    texts: dict[str, str],
) -> None:
    tests: list[tuple[str, dict[str, Any], dict[str, dict[str, Any]]]] = []

    missing_fixture = copy.deepcopy(manifest)
    missing_fixture["fixtures"] = missing_fixture["fixtures"][:-1]
    tests.append(("missing fixture", missing_fixture, {}))

    bad_sha = copy.deepcopy(manifest)
    bad_sha["fixtures"][0]["candidate"]["sha256"] = "0" * 64
    tests.append(("candidate sha drift", bad_sha, {}))

    bad_family = copy.deepcopy(manifest)
    bad_family["fixtures"][0]["operation_family"] = "vulkan"
    tests.append(("unknown family", bad_family, {}))

    bad_command = copy.deepcopy(manifest)
    bad_command["fixtures"][0]["rerun_command"] = bad_command["fixtures"][0]["rerun_command"].replace(
        "--context hou2-prod1",
        "--context removed",
    )
    tests.append(("missing explicit context", bad_command, {}))

    route_claim = copy.deepcopy(manifest)
    route_claim["claim_policy"]["runtime_route_change"] = True
    tests.append(("route change overclaim", route_claim, {}))

    digest_drift = copy.deepcopy(manifest)
    candidate_path = digest_drift["fixtures"][0]["candidate"]["path"]
    candidate = load_json(ROOT / candidate_path)
    candidate["output"]["fnv1a64"] = "0000000000000000"
    tests.append(("candidate output digest drift", digest_drift, {candidate_path: candidate}))

    for name, mutated_manifest, overrides in tests:
        mutated_report = Report()
        validate(
            mutated_report,
            mutated_manifest,
            boundary,
            texts,
            run_pair_comparators=False,
            capture_overrides=overrides,
        )
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def boundary_sets(boundary: dict[str, Any]) -> tuple[set[str], set[str]]:
    families: set[str] = set()
    operations: set[str] = set()
    for family in boundary.get("operation_families", []):
        if not isinstance(family, dict):
            continue
        name = family.get("name")
        if isinstance(name, str):
            families.add(name)
        for operation in family.get("operations", []):
            if isinstance(operation, str):
                operations.add(operation)
    return families, operations


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Backend operation tensor fixtures: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
