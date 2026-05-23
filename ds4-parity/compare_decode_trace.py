#!/usr/bin/env python3
"""Compare the Rust dry-run decode execution trace against the M10.5b oracle."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import compare_decode_backend_facade


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "ds4-parity/baselines/graph/m10.5b/decode-plan-oracle.json"
RUST_FACADE = ROOT / "rust/ds4-gpu/src/decode_backend.rs"
C_SOURCE = ROOT / "ds4.c"
N_LAYER = 43
N_INDEXER_TOP_K = 512


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


def run_rust_trace() -> dict[str, Any]:
    proc = subprocess.run(
        ["cargo", "run", "-p", "ds4-gpu", "--bin", "ds4-decode-trace", "--quiet"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


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


def compress_ratio(layer: int) -> int:
    if layer < 2:
        return 0
    return 4 if layer % 2 == 0 else 128


def compressed_rows_before(pos: int, ratio: int) -> int:
    return 0 if ratio == 0 else pos // ratio


def compressed_rows_after(pos: int, ratio: int) -> int:
    if ratio == 0:
        return 0
    return pos // ratio + (1 if compressed_row_emits(pos, ratio) else 0)


def compressed_row_emits(pos: int, ratio: int) -> bool:
    return ratio != 0 and pos % ratio == ratio - 1


def validate_c_source_anchors(report: Report, source: str) -> None:
    fragments = [
        "DS4_N_INDEXER_TOP_K    = 512,",
        "const bool emit = ((pos + 1u) % ratio) == 0u;",
        "const uint32_t comp_row = g->layer_n_comp[il];",
        "if (ok && emit) g->layer_n_comp[il]++;",
        "const uint32_t index_row = g->layer_n_index_comp[il];",
        "if (ok && emit) g->layer_n_index_comp[il]++;",
        "if (ok && g->layer_n_comp[il] > decode_top_k)",
        "g->layer_n_index_comp[il],\n                                                                DS4_N_INDEXER_HEAD,",
        "if (ok && allow_split_flush && split_after_layers != 0 && il + 1u == split_after_layers)",
        "ok = ds4_gpu_flush_commands() != 0;",
    ]
    for fragment in fragments:
        report.check(fragment in source, f"C decode source anchor missing: {fragment}")


def validate(report: Report, trace: dict[str, Any], oracle: dict[str, Any]) -> None:
    report.check(trace.get("schema") == "ds4.decode_trace.v1", "schema drift")
    report.check(trace.get("scope") == "dry_run", "scope drift")
    validate_c_source_anchors(report, C_SOURCE.read_text())

    expected_cases = {case["name"]: case for case in oracle["cases"]}
    got_cases = {case.get("name"): case for case in trace.get("cases", [])}
    report.check(set(got_cases) == set(expected_cases), "case set drift")

    facade_specs = {
        spec.operation: spec
        for spec in compare_decode_backend_facade.parse_facade_specs(RUST_FACADE.read_text())
    }
    facade_ops_seen: set[str] = set()
    for name, expected in expected_cases.items():
        case = got_cases.get(name)
        if not isinstance(case, dict):
            continue
        validate_case(report, case, expected, oracle, facade_specs, facade_ops_seen)

    report.check(
        facade_ops_seen == set(compare_decode_backend_facade.DEFAULT_DECODE_OPERATIONS),
        f"default facade coverage drift: {sorted(facade_ops_seen)}",
    )


def validate_case(
    report: Report,
    case: dict[str, Any],
    expected: dict[str, Any],
    oracle: dict[str, Any],
    facade_specs: dict[str, Any],
    facade_ops_seen: set[str],
) -> None:
    name = expected["name"]
    events = case.get("events", [])
    summary = case.get("summary", {})
    report.check(isinstance(events, list), f"{name}: events must be list")
    report.check(isinstance(summary, dict), f"{name}: summary must be object")
    if not isinstance(events, list) or not isinstance(summary, dict):
        return

    report.check([event.get("index") for event in events] == list(range(len(events))), f"{name}: event indexes drift")
    report.check(summary.get("events") == len(events), f"{name}: summary event count drift")
    report.check(summary.get("layers") == N_LAYER, f"{name}: layer count drift")
    layer_counts = oracle["layer_counts"]
    report.check(summary.get("dense_layers") == layer_counts["dense"], f"{name}: dense layer count drift")
    report.check(summary.get("ratio4_layers") == layer_counts["ratio4"], f"{name}: ratio4 layer count drift")
    report.check(summary.get("ratio128_layers") == layer_counts["ratio128"], f"{name}: ratio128 layer count drift")
    report.check(summary.get("state_events") == N_LAYER, f"{name}: state event count drift")
    report.check(
        summary.get("compressed_emit_layers")
        == expected["ratio4_emit_layers"] + expected["ratio128_emit_layers"],
        f"{name}: compressed emit layer count drift",
    )
    report.check(
        summary.get("indexed_attention_layers") == expected["ratio4_indexed_layers"],
        f"{name}: indexed attention layer count drift",
    )
    report.check(
        summary.get("split_flushes") == (1 if expected["flush_after_layer"] is not None else 0),
        f"{name}: split flush count drift",
    )
    report.check(
        summary.get("output_head_calls") == (6 if expected["need_logits"] else 0),
        f"{name}: output head call count drift",
    )
    report.check(
        summary.get("read_logits_calls") == (1 if expected["need_logits"] else 0),
        f"{name}: read logits count drift",
    )
    report.check(summary.get("synchronize_on_failure_calls") == 1, f"{name}: synchronize marker drift")

    ops = Counter(event.get("operation") for event in events)
    report.check(ops["ds4_gpu_begin_commands"] == 1, f"{name}: begin command count drift")
    report.check(ops["ds4_gpu_end_commands"] == 1, f"{name}: end command count drift")
    report.check(ops["ds4_gpu_flush_commands"] == summary.get("split_flushes"), f"{name}: flush op count drift")
    report.check(ops["ds4_gpu_tensor_read"] == summary.get("read_logits_calls"), f"{name}: tensor read count drift")
    report.check(ops["ds4_gpu_synchronize"] == 1, f"{name}: synchronize op count drift")
    validate_token_stage_order(report, name, events, expected, oracle)
    validate_embed_and_output_tape(report, name, events, expected)

    stage_events_by_layer: dict[int, list[str]] = {}
    state_events: dict[int, dict[str, Any]] = {}
    layer_stage_set = set(oracle["layer_stage_order"])
    for event in events:
        if event.get("kind") == "facade":
            operation = event.get("operation")
            facade_ops_seen.add(operation)
            spec = facade_specs.get(operation)
            report.check(spec is not None, f"{name}: unexpected facade operation {operation}")
            if spec is not None:
                report.check(event.get("method") == spec.method, f"{name}: method drift for {operation}")
                report.check(event.get("tensor_args") == spec.tensor_args, f"{name}: tensor args drift for {operation}")
        layer = event.get("layer")
        if (
            isinstance(layer, int)
            and event.get("kind") == "stage"
            and event.get("stage") in layer_stage_set
        ):
            stage_events_by_layer.setdefault(layer, []).append(event.get("stage"))
        if isinstance(layer, int) and event.get("kind") == "state":
            state_events[layer] = event

    for layer in range(N_LAYER):
        report.check(
            stage_events_by_layer.get(layer) == oracle["layer_stage_order"],
            f"{name}: layer {layer} stage order drift",
        )
        report.check(layer in state_events, f"{name}: missing layer {layer} state event")

    validate_state_edges(report, name, state_events, expected)


def validate_token_stage_order(
    report: Report,
    name: str,
    events: list[dict[str, Any]],
    expected: dict[str, Any],
    oracle: dict[str, Any],
) -> None:
    layer_stage_set = set(oracle["layer_stage_order"])
    token_stage_events = [
        event
        for event in events
        if event.get("kind") == "stage" and event.get("stage") not in layer_stage_set
    ]
    expected_stages = ["begin_commands", "embed_token_hc", "decode_layers"]
    if expected["flush_after_layer"] is not None:
        expected_stages.append("split_flush")
    if expected["need_logits"]:
        expected_stages.append("output_head")
    expected_stages.append("end_commands")
    if expected["need_logits"]:
        expected_stages.append("read_logits")
    expected_stages.append("synchronize_on_failure")
    report.check(
        [event.get("stage") for event in token_stage_events] == expected_stages,
        f"{name}: token stage order drift",
    )

    for event in token_stage_events:
        if event.get("stage") == "split_flush":
            report.check(event.get("layer") == expected["flush_after_layer"], f"{name}: split flush layer drift")
        else:
            report.check(event.get("layer") is None, f"{name}: token stage {event.get('stage')} should not carry a layer")

    split_stage_events = [
        event for event in events if event.get("kind") == "stage" and event.get("stage") == "split_flush"
    ]
    split_ops = [event for event in events if event.get("operation") == "ds4_gpu_flush_commands"]
    if expected["flush_after_layer"] is None:
        report.check(not split_stage_events, f"{name}: unexpected split flush stage")
        report.check(not split_ops, f"{name}: unexpected split flush op")
        return

    report.check(len(split_stage_events) == 1, f"{name}: split flush stage count drift")
    report.check(len(split_ops) == 1, f"{name}: split flush op count drift")
    if len(split_stage_events) != 1:
        return
    split_event = split_stage_events[0]
    flush_layer = expected["flush_after_layer"]
    layer_events = [
        event
        for event in events
        if event.get("layer") == flush_layer and event.get("stage") != "split_flush"
    ]
    next_layer_events = [
        event
        for event in events
        if event.get("layer") == flush_layer + 1 and event.get("stage") != "split_flush"
    ]
    report.check(layer_events and max(event["index"] for event in layer_events) < split_event["index"],
                 f"{name}: split flush did not follow layer {flush_layer}")
    if flush_layer + 1 < N_LAYER:
        report.check(next_layer_events and split_event["index"] < min(event["index"] for event in next_layer_events),
                     f"{name}: split flush did not precede layer {flush_layer + 1}")
    else:
        report.check(not next_layer_events, f"{name}: unexpected events after final split flush layer")
    if len(split_ops) == 1:
        report.check(split_ops[0].get("layer") == flush_layer, f"{name}: split flush op layer drift")
        report.check(split_ops[0].get("index") == split_event.get("index") + 1, f"{name}: split flush op order drift")


def validate_embed_and_output_tape(
    report: Report,
    name: str,
    events: list[dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    embed_ops = [
        event.get("operation")
        for event in events
        if event.get("kind") == "facade" and event.get("stage") == "embed_token_hc"
    ]
    report.check(embed_ops == ["ds4_gpu_embed_token_hc_tensor"], f"{name}: embed facade tape drift")
    output_ops = [
        event.get("operation")
        for event in events
        if event.get("kind") == "facade" and event.get("stage") == "output_head"
    ]
    expected_output = [
        "ds4_gpu_rms_norm_plain_tensor",
        "ds4_gpu_matmul_f16_tensor",
        "ds4_gpu_output_hc_weights_tensor",
        "ds4_gpu_hc_weighted_sum_tensor",
        "ds4_gpu_rms_norm_weight_tensor",
        "ds4_gpu_matmul_q8_0_tensor",
    ] if expected["need_logits"] else []
    report.check(output_ops == expected_output, f"{name}: output head facade tape drift")


def validate_state_edges(
    report: Report,
    name: str,
    state_events: dict[int, dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    pos = expected["pos"]
    for layer in range(N_LAYER):
        state = state_events.get(layer, {}).get("state", {})
        ratio = compress_ratio(layer)
        report.check(state.get("pos") == pos, f"{name}: layer {layer} pos drift")
        report.check(state.get("raw_row") == expected["raw_row"], f"{name}: layer {layer} raw row drift")
        report.check(state.get("n_raw") == expected["n_raw"], f"{name}: layer {layer} n_raw drift")
        report.check(state.get("raw_start") == expected["raw_start"], f"{name}: layer {layer} raw_start drift")
        if ratio == 0:
            report.check(state.get("compression") == "dense", f"{name}: layer {layer} dense compression drift")
            report.check(state.get("comp_before") is None, f"{name}: layer {layer} dense comp_before drift")
            report.check(state.get("comp_after") is None, f"{name}: layer {layer} dense comp_after drift")
            report.check(state.get("index_before") is None, f"{name}: layer {layer} dense index_before drift")
            report.check(state.get("index_after") is None, f"{name}: layer {layer} dense index_after drift")
            report.check(not state.get("emit_compressed_row"), f"{name}: layer {layer} dense emit drift")
            report.check(not state.get("indexed_attention"), f"{name}: layer {layer} dense indexed attention drift")
            continue

        expected_before = compressed_rows_before(pos, ratio)
        expected_after = compressed_rows_after(pos, ratio)
        expected_emit = compressed_row_emits(pos, ratio)
        expected_indexed = ratio == 4 and expected_after > N_INDEXER_TOP_K
        expected_attention = (
            "ds4_gpu_attention_indexed_mixed_batch_heads_tensor"
            if expected_indexed
            else "ds4_gpu_attention_decode_heads_tensor"
        )
        report.check(state.get("compression") == f"ratio{ratio}", f"{name}: layer {layer} compression drift")
        report.check(state.get("comp_before") == expected_before, f"{name}: layer {layer} comp_before drift")
        report.check(state.get("comp_after") == expected_after, f"{name}: layer {layer} comp_after drift")
        report.check(state.get("emit_compressed_row") == expected_emit, f"{name}: layer {layer} emit cadence drift")
        report.check(state.get("indexed_attention") == expected_indexed, f"{name}: layer {layer} indexed attention drift")
        report.check(state.get("attention_operation") == expected_attention, f"{name}: layer {layer} attention op drift")
        if ratio == 4:
            report.check(state.get("index_before") == expected_before, f"{name}: layer {layer} index_before drift")
            report.check(state.get("index_after") == expected_after, f"{name}: layer {layer} index_after drift")
        else:
            report.check(state.get("index_before") is None, f"{name}: layer {layer} ratio128 index_before drift")
            report.check(state.get("index_after") is None, f"{name}: layer {layer} ratio128 index_after drift")

    dense = state_events.get(0, {}).get("state", {})
    report.check(dense.get("compression") == "dense", f"{name}: dense layer state drift")
    report.check(dense.get("pos") == expected["pos"], f"{name}: dense layer pos drift")
    report.check(dense.get("raw_row") == expected["raw_row"], f"{name}: raw row drift")
    report.check(dense.get("n_raw") == expected["n_raw"], f"{name}: n_raw drift")
    report.check(dense.get("raw_start") == expected["raw_start"], f"{name}: raw_start drift")

    ratio4 = state_events.get(2, {}).get("state", {})
    report.check(ratio4.get("compression") == "ratio4", f"{name}: ratio4 compression drift")
    report.check(ratio4.get("comp_before") == expected["ratio4_comp_before"], f"{name}: ratio4 comp_before drift")
    report.check(ratio4.get("comp_after") == expected["ratio4_comp_after"], f"{name}: ratio4 comp_after drift")
    report.check(ratio4.get("index_before") == expected["ratio4_comp_before"], f"{name}: index comp_before drift")
    report.check(ratio4.get("index_after") == expected["ratio4_comp_after"], f"{name}: index comp_after drift")
    report.check(
        ratio4.get("indexed_attention") == (expected["ratio4_indexed_layers"] != 0),
        f"{name}: ratio4 indexed-attention drift",
    )
    expected_ratio4_attention = (
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor"
        if expected["ratio4_indexed_layers"] != 0
        else "ds4_gpu_attention_decode_heads_tensor"
    )
    report.check(
        ratio4.get("attention_operation") == expected_ratio4_attention,
        f"{name}: ratio4 attention operation drift",
    )

    ratio128 = state_events.get(3, {}).get("state", {})
    report.check(ratio128.get("compression") == "ratio128", f"{name}: ratio128 compression drift")
    report.check(ratio128.get("comp_before") == expected["ratio128_comp_before"], f"{name}: ratio128 comp_before drift")
    report.check(ratio128.get("comp_after") == expected["ratio128_comp_after"], f"{name}: ratio128 comp_after drift")
    report.check(ratio128.get("index_before") is None, f"{name}: ratio128 index_before drift")
    report.check(ratio128.get("index_after") is None, f"{name}: ratio128 index_after drift")
    report.check(not ratio128.get("indexed_attention"), f"{name}: ratio128 indexed-attention drift")


def run_negative_tests(report: Report, trace: dict[str, Any], oracle: dict[str, Any]) -> None:
    mutations = [
        ("summary", mutate_summary),
        ("operation", mutate_operation),
        ("split-flush-layer", mutate_split_flush_layer),
        ("state", mutate_state),
        ("state-emit", mutate_emit_state),
    ]
    for name, mutate in mutations:
        mutated_report = Report()
        validate(mutated_report, mutate(trace), oracle)
        report.check(not mutated_report.ok, f"negative mutation was not detected: {name}")


def mutate_summary(trace: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(trace)
    mutated["cases"][0]["summary"]["split_flushes"] = 0
    return mutated


def mutate_operation(trace: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(trace)
    for event in mutated["cases"][0]["events"]:
        if event.get("operation") == "ds4_gpu_embed_token_hc_tensor":
            event["operation"] = "ds4_gpu_not_a_real_decode_op"
            break
    return mutated


def mutate_split_flush_layer(trace: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(trace)
    for event in mutated["cases"][0]["events"]:
        if event.get("stage") == "split_flush":
            event["layer"] += 1
            break
    return mutated


def mutate_state(trace: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(trace)
    for case in mutated["cases"]:
        if case["name"] != "ratio_emit_boundary":
            continue
        for event in case["events"]:
            if event.get("kind") == "state" and event.get("layer") == 2:
                event["state"]["comp_after"] += 1
                return mutated
    raise AssertionError("ratio_emit_boundary layer 2 state not found")


def mutate_emit_state(trace: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(trace)
    for case in mutated["cases"]:
        if case["name"] != "short_decode_after_prefill":
            continue
        for event in case["events"]:
            if event.get("kind") == "state" and event.get("layer") == 2:
                event["state"]["emit_compressed_row"] = True
                return mutated
    raise AssertionError("short_decode_after_prefill layer 2 state not found")


def print_report(report: Report) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Rust decode trace comparator: {status}, {report.checks} checks")
    for error in report.errors:
        print(f"  - {error}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-test", action="store_true")
    args = parser.parse_args(argv)

    report = Report()
    oracle = load_json(ORACLE)
    trace = run_rust_trace()
    validate(report, trace, oracle)
    if args.negative_test:
        run_negative_tests(report, trace, oracle)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
