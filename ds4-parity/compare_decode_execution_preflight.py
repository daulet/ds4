#!/usr/bin/env python3
"""Validate the Rust decode execution preflight contract for M10.5c4c2b1."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "cargo": ROOT / "rust/ds4-gpu/Cargo.toml",
    "lib": ROOT / "rust/ds4-gpu/src/lib.rs",
    "module": ROOT / "rust/ds4-gpu/src/decode_execution.rs",
    "bin": ROOT / "rust/ds4-gpu/src/bin/ds4-decode-exec-preflight.rs",
    "report": ROOT / "ds4-parity/run_parity_report.py",
    "readme": ROOT / "ds4-parity/README.md",
    "roadmap": ROOT / "RUST_PORT_ROADMAP.md",
    "todo": ROOT / ".memory/TODO.md",
}

SCHEMA = "ds4.decode_execution_preflight.v1"
SCOPE = "model_backed_b300_preflight"
CASE = "short_decode_logits"
MODEL_SHA = "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
REQUIRED_CHECKPOINTS = {
    "short_decode_logits",
    "short_decode_layer2_attn_comp_cache",
    "short_decode_layer2_index_comp_cache",
}
REQUIRED_TENSORS = {
    (None, "logits"),
    (2, "layer_attn_comp_cache"),
    (2, "layer_index_comp_cache"),
}


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
    texts = {name: path.read_text() for name, path in FILES.items()}
    if args.negative_test:
        return run_negative_tests(texts)

    report = Report()
    validate_static(report, texts)
    if args.candidate is not None:
        validate_candidate(report, load_json(args.candidate))

    if report.ok:
        print(f"Rust decode execution preflight contract: {report.checks} checks")
    else:
        print_errors(report.errors)
    return 0 if report.ok else 1


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="B300 preflight JSON emitted by ds4-decode-exec-preflight")
    parser.add_argument("--negative-test", action="store_true")
    return parser.parse_args(list(argv))


def validate_static(report: Report, texts: dict[str, str]) -> None:
    report.check('ds4-gguf = { path = "../ds4-gguf" }' in texts["cargo"], "ds4-gpu must depend on ds4-gguf for model-backed parsing")
    report.check("pub mod decode_execution;" in texts["lib"], "decode_execution module not exported")

    module = texts["module"]
    report.check(SCHEMA in module, "preflight schema missing")
    report.check(SCOPE in module, "preflight scope missing")
    report.check("DECODE_EXECUTION_PREFLIGHT_LAYERS: &[usize] = &[0, 2, 3]" in module, "preflight layer slice drift")
    for checkpoint in sorted(REQUIRED_CHECKPOINTS):
        report.check(checkpoint in module, f"checkpoint target {checkpoint} missing")
    for field in ("logits", "layer_attn_comp_cache", "layer_index_comp_cache"):
        report.check(field in module, f"representative tensor {field} missing")
    report.check("covers_default_decode" in module, "compression coverage helper missing")

    bin_text = texts["bin"]
    for needle, message in (
        ("parse_gguf_allowing_missing_tensor_data", "GGUF prefix parser missing"),
        ("bind_ds4_weights", "DS4 weight binding missing"),
        ("mmap(", "full model mmap missing"),
        ("set_model_fd", "model fd bridge missing"),
        ("set_model_map_range", "model tensor-data range bridge missing"),
        ("cache_model_range", "representative model cache hook missing"),
        ("cache_q8_f16_range", "representative q8/f16 cache hook missing"),
        ("Tensor::allocate", "representative tensor allocation missing"),
        ("PREFLIGHT_CHECKPOINT_TARGETS", "checkpoint target report missing"),
        ("REPRESENTATIVE_TENSORS", "representative tensor plan missing"),
    ):
        report.check(needle in bin_text, message)
    report.check("gguf.tensor_data_offset" in bin_text, "model range must start at tensor_data_offset")
    report.check("SMALL_MODEL_CACHE_LIMIT" in bin_text, "model cache range must stay bounded")
    report.check("Q8_CACHE_LIMIT" in bin_text, "q8 cache range must stay bounded")

    report_text = texts["report"]
    report.check("M10.5c4c2b1 Rust decode execution preflight comparator" in report_text, "unified report comparator missing")
    report.check("M10.5c4c2b1 B300 Rust decode execution preflight rerun" in report_text, "B300 preflight skip missing")
    report.check("--bin ds4-decode-exec-preflight" in report_text, "B300 preflight command missing binary")
    report.check("--candidate /tmp/ds4-c2b1-preflight.json" in report_text, "B300 preflight candidate validation missing")
    report.check(MODEL_SHA in report_text, "B300 preflight command missing model hash")

    report.check("M10.5c4c2b1 Rust decode execution preflight" in texts["readme"], "README entry missing")
    report.check("M10.5c4c2b1: Rust Decode Execution Preflight" in texts["roadmap"], "roadmap split missing")
    report.check("M10.5c4c2b2: Rust One-Token Decode B300 Execution" in texts["roadmap"], "roadmap remainder missing")
    report.check("M10.5c4c2b1: Rust Decode Execution Preflight" in texts["todo"], "TODO split missing")
    report.check("M10.5c4c2b2: Rust One-Token Decode B300 Execution" in texts["todo"], "TODO remainder missing")


def validate_candidate(report: Report, obj: dict[str, Any]) -> None:
    report.check(obj.get("schema") == SCHEMA, "candidate schema drift")
    report.check(obj.get("scope") == SCOPE, "candidate scope drift")
    report.check(obj.get("case") == CASE, "candidate case drift")

    model = obj.get("model")
    report.check(isinstance(model, dict), "candidate model must be object")
    if isinstance(model, dict):
        report.check(model.get("sha256") == MODEL_SHA, "candidate model sha drift")
        report.check(model.get("mapped_size") == 86720111488, "candidate model size drift")
        report.check(model.get("backend_model_size") == model.get("mapped_size"), "candidate backend model size drift")
        report.check(isinstance(model.get("tensor_count"), int) and model["tensor_count"] > 0, "candidate tensor count invalid")
        report.check(isinstance(model.get("tensor_data_offset"), int) and model["tensor_data_offset"] > 0, "candidate tensor_data_offset invalid")
        report.check(model.get("bound_layers") == 43, "candidate bound layer count drift")

    backend = obj.get("backend")
    report.check(isinstance(backend, dict), "candidate backend must be object")
    if isinstance(backend, dict):
        report.check(backend.get("initialized") is True, "candidate backend not initialized")
        report.check(backend.get("set_model_fd") is True, "candidate model fd not set")
        map_range = backend.get("set_model_map_range")
        report.check(isinstance(map_range, dict), "candidate model map range missing")
        if isinstance(map_range, dict) and isinstance(model, dict):
            report.check(map_range.get("offset") == model.get("tensor_data_offset"), "candidate map offset drift")
            report.check(map_range.get("bytes") == model.get("mapped_size") - model.get("tensor_data_offset"), "candidate map byte count drift")

    coverage = obj.get("layer_coverage")
    report.check(isinstance(coverage, dict), "candidate layer coverage missing")
    if isinstance(coverage, dict):
        report.check(coverage.get("dense") is True, "candidate dense coverage missing")
        report.check(coverage.get("ratio4") is True, "candidate ratio4 coverage missing")
        report.check(coverage.get("ratio128") is True, "candidate ratio128 coverage missing")
        report.check(coverage.get("covers_default_decode") is True, "candidate default decode coverage missing")

    checkpoints = obj.get("checkpoint_targets")
    report.check(isinstance(checkpoints, list), "candidate checkpoint targets missing")
    if isinstance(checkpoints, list):
        names = {entry.get("name") for entry in checkpoints if isinstance(entry, dict)}
        report.check(REQUIRED_CHECKPOINTS <= names, "candidate checkpoint target set incomplete")

    tensors = obj.get("representative_tensors")
    report.check(isinstance(tensors, list), "candidate representative tensors missing")
    if isinstance(tensors, list):
        got = {
            (entry.get("layer"), entry.get("field"))
            for entry in tensors
            if isinstance(entry, dict) and entry.get("allocated") is True
        }
        report.check(REQUIRED_TENSORS <= got, "candidate representative tensor set incomplete")

    selected = obj.get("selected_weights")
    report.check(isinstance(selected, list) and len(selected) >= 20, "candidate selected weights too small")
    if isinstance(selected, list):
        roles = {entry.get("role") for entry in selected if isinstance(entry, dict)}
        for role in (
            "base.token_embd",
            "base.output",
            "base.layer.0.attn_norm",
            "base.layer.2.indexer_proj",
            "base.layer.3.attn_compressor_norm",
        ):
            report.check(role in roles, f"candidate selected weight {role} missing")

    cache = obj.get("cache")
    report.check(isinstance(cache, dict), "candidate cache report missing")
    if isinstance(cache, dict):
        report.check(nonempty_list(cache.get("model_ranges")), "candidate model cache ranges missing")
        report.check(nonempty_list(cache.get("q8_f16_ranges")), "candidate q8/f16 cache ranges missing")


def load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read candidate {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"candidate {path}: expected JSON object")
    return obj


def nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def run_negative_tests(texts: dict[str, str]) -> int:
    mutations = [
        ("remove schema", "module", SCHEMA, "ds4.decode_execution_preflight.removed"),
        ("remove layer coverage", "module", "&[0, 2, 3]", "&[0, 2]"),
        ("remove mmap", "bin", "mmap(", "mmap_removed("),
        ("remove model range", "bin", "set_model_map_range", "set_model_map"),
        ("remove tensor allocation", "bin", "Tensor::allocate", "Tensor_removed::new"),
        ("remove b300 candidate check", "report", "--candidate /tmp/ds4-c2b1-preflight.json", ""),
        ("remove roadmap split", "roadmap", "M10.5c4c2b1: Rust Decode Execution Preflight", "M10.5c4c2b1 removed"),
    ]
    failures: list[str] = []
    for label, key, needle, replacement in mutations:
        mutated = copy.deepcopy(texts)
        if needle not in mutated[key]:
            failures.append(f"{label}: mutation needle not found")
            continue
        mutated[key] = mutated[key].replace(needle, replacement)
        report = Report()
        validate_static(report, mutated)
        if report.ok:
            failures.append(f"{label}: validation unexpectedly passed")
    if failures:
        print_errors(failures)
        return 1
    print(f"negative tests passed: {len(mutations)} mutations rejected")
    return 0


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
