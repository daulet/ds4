# DS4 Parity Artifacts

This directory stores oracle captures and future Rust comparison fixtures.

- `baselines/`: current C/CUDA/Metal command captures used as the oracle for
  early port work.
- Future fixture roots should be added by milestone and should include a
  manifest entry that names the oracle, input fixture, comparator, acceptance
  rule, command, environment, output files, and drift policy.

Do not treat a log file as proof by itself. A manifest entry must explain which
behavior the log proves and which requirements remain blocked or deferred.

## Static Verification

Run the local static verifier before changing baseline artifacts:

```sh
python3 ds4-parity/verify_baselines.py
```

The verifier checks committed hashes and structured artifact shapes without
rerunning model-backed commands. Its negative test copies the baseline fixtures
to a temporary directory, corrupts one CSV, and requires verification failure:

```sh
python3 ds4-parity/verify_baselines.py --negative-test
```

Compare the committed server and KV artifacts against themselves with the
documented normalizations:

```sh
python3 ds4-parity/compare_server_kv.py
```

The comparator accepts candidate artifact directories from a fresh run or a
future Rust implementation:

```sh
python3 ds4-parity/compare_server_kv.py \
  --server-candidate /path/to/server-traces/m0.4 \
  --kv-candidate /path/to/kv-artifacts/m0.5
```

Its negative test corrupts behavioral server/KV fields in temporary candidate
copies and requires comparison failure:

```sh
python3 ds4-parity/compare_server_kv.py --negative-test
```

Run the Milestone 9 server/runtime report, which bundles the model-free Rust
server tests, server/KV artifact comparators, and exact B300 refresh skips:

```sh
python3 ds4-parity/run_server_parity_report.py
```

Compare the committed official-vector logprob fixture and M0.3 run evidence:

```sh
python3 ds4-parity/compare_logprob_numeric.py
```

The M0.3 B300 run log is pass/fail evidence from
`./ds4_test --logprob-vectors`; it does not dump runtime logits. The numeric
slice lives in `tests/test-vectors/official.vec` and the raw official JSON under
`tests/test-vectors/official/`. The comparator therefore audits those numeric
fixtures, verifies the B300 run markers, and compares candidate vector files
with exact selected-token matching plus the reported absolute logprob tolerance:

```sh
python3 ds4-parity/compare_logprob_numeric.py \
  --candidate-vec /path/to/official-vector-style-output.vec
```

Its negative test corrupts a temporary candidate selected token and logprob
value and requires comparison failure:

```sh
python3 ds4-parity/compare_logprob_numeric.py --negative-test
```

Compare the committed B300 benchmark CSV baselines:

```sh
python3 ds4-parity/compare_bench_csv.py
```

The benchmark comparator treats workload shape as exact behavioral surface:
schema, context frontiers, prefill intervals, generation-token counts, and
`kvcache_bytes` must match. Throughput is performance surface; it is compared
only after capture metadata confirms the same model, prompt, CUDA backend
marker, and GPU machine class. The default throughput threshold is the M0.6
policy of at most 10% regression:

```sh
python3 ds4-parity/compare_bench_csv.py \
  --candidate-dir /path/to/bench/m0.6 \
  --max-regression 0.10
```

Its negative test corrupts schema, context frontier, generation-token,
`kvcache_bytes`, and throughput fields in temporary candidate copies and
requires comparison failure:

```sh
python3 ds4-parity/compare_bench_csv.py --negative-test
```

Run the unified local parity report:

```sh
python3 ds4-parity/run_parity_report.py
```

The unified report runs local no-model C checks, then runs the committed M1,
M4, M5, M6, M7, M8, M9, and M10 comparator reports. Model-backed B300 oracle
refreshes are skipped by default, but each skip includes the temp-kubeconfig
and explicit-context rerun command needed to reproduce the check. For a
comparator-only report:

```sh
python3 ds4-parity/run_parity_report.py --skip-local-oracles
```

## Runtime Graph Inventory

Validate the Milestone 10.2 current-C graph plan and backend operation
inventory:

```sh
python3 ds4-parity/check_graph_plan_inventory.py
```

The checker compares the committed graph oracle under
`baselines/graph/m10.2/` against `ds4_gpu.h`, `ds4.c` graph tensor fields,
fixed DS4 model constants, compression ratios, context-cap calculations, and
recorded command-buffer boundaries. Plan cases assume
`DS4_METAL_PREFILL_CHUNK` and `DS4_METAL_GRAPH_RAW_CAP` are unset. Its negative
test removes a backend facade assignment, removes a tensor owner, and mutates a
raw-cap plan value:

```sh
python3 ds4-parity/check_graph_plan_inventory.py --negative-test
```

Compare the Rust M10.3 graph-plan and facade inventory against that oracle:

```sh
python3 ds4-parity/compare_graph_plan_rust.py --negative-test
```

The Rust comparator checks `rust/ds4-gpu/src/graph_plan.rs` for matching
backend operation facade targets, graph tensor owner groups, command-boundary
records, and the M10.2 context/MTP plan cases. Its negative test mutates an
operation, a tensor field, a command boundary, and an MTP plan case in memory.

Validate the M10.9a runtime graph closure matrix:

```sh
python3 ds4-parity/check_runtime_graph_closure_matrix.py
python3 ds4-parity/check_runtime_graph_closure_matrix.py --negative-test
```

Refresh the B300 fixture-readiness probe used by M10.9:

```sh
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; \
  target=$(readlink -f /workspace/ds4/ds4flash.gguf); \
  printf "resolved_model=%s\n" "$target"; \
  stat -c "resolved_model_bytes=%s" "$target"; \
  sha256sum tests/test-vectors/official.vec speed-bench/promessi_sposi.txt; \
  test -f ds4-parity/baselines/bench/m0.6/csv/b300-short.csv; \
  test -f ds4-parity/baselines/bench/m0.6/csv/b300-long.csv; \
  python3 -m json.tool \
    ds4-parity/baselines/bench/m0.6/logs/csv-summary.json >/dev/null; \
  printf "m109_fixture_probe=ok\n"'
```

The matrix pins M10.9b through M10.9f to concrete oracles, fixture paths,
artifact locations, rerun commands, claim boundaries, and drift policies. It
forbids reporting full runtime graph parity before the route, official-vector,
long-context, tool/server, and benchmark closure gates are all current.

Validate the M10.9b Rust runtime graph route selector:

```sh
python3 ds4-parity/check_runtime_graph_route_preflight.py
python3 ds4-parity/check_runtime_graph_route_preflight.py --negative-test
```

Refresh the exact route-preflight summary after changing runtime route
selection:

```sh
python3 ds4-parity/check_runtime_graph_route_preflight.py \
  --write-summary \
  ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json \
  --negative-test
```

The M10.9b comparator builds the Rust runtime binaries, records exact
target-stream, disabled-route, invalid-selector, and unsupported graph-route
outcomes, and verifies unsupported graph selection fails before model open,
stream output, checkpoint/cache mutation, or server KVC directory creation.

Validate the M10.9c Runtime graph official-vector gate:

```sh
python3 ds4-parity/run_runtime_graph_official_vectors.py
python3 ds4-parity/run_runtime_graph_official_vectors.py --negative-test
```

Refresh the live B300 Rust runtime official-vector summary:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; CUDA_ARCH=native \
  python3 ds4-parity/run_runtime_graph_official_vectors.py \
    --workdir /workspace/ds4 --model /workspace/ds4/ds4flash.gguf \
    --write-summary /tmp/ds4-m109c-official-vectors.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m109c-official-vectors.json \
  ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json
```

The M10.9c comparator checks the Rust `ds4-runtime-official-vectors-rs`
capture against the M0.3 official-vector contract on B300: route `graph`,
backend `cuda`, q2-imatrix model hash, fixture hash, selected token bytes,
top-logprob shape, official-top presence, M6 logprob tolerance, and the
current-C `long_memory_archive` skip reason.

Validate the M10.9d Runtime graph long-context gate:

```sh
python3 ds4-parity/run_runtime_graph_long_context.py
python3 ds4-parity/run_runtime_graph_long_context.py --negative-test
```

Refresh the live B300 Rust runtime long-context summary:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; CUDA_ARCH=native \
  python3 ds4-parity/run_runtime_graph_long_context.py \
    --workdir /workspace/ds4 --model /workspace/ds4/ds4flash.gguf \
    --write-summary /tmp/ds4-m109d-long-context.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m109d-long-context.json \
  ds4-parity/baselines/graph/m10.9d/runtime-long-context.json
```

The M10.9d comparator checks the Rust `ds4-runtime-long-context-rs`
capture against the current-C `./ds4_test --long-context` pass/fail contract
on B300: route `graph`, backend `cuda`, q2-imatrix model hash, long prompt
hash, context length, deterministic generation settings, cache/KVC accounting,
fact-recall output, no target-stream fallback marker, and retained raw logs.

Validate the M10.9e Runtime graph tool/server gate:

```sh
python3 ds4-parity/run_tool_call_quality.py
python3 ds4-parity/run_tool_call_quality.py --negative-test
```

Refresh the live B300 Rust runtime tool/server summary:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; CUDA_ARCH=native make ds4_test; \
  CUDA_ARCH=native cargo build -p ds4-engine --bin ds4-server-runtime-rs; \
  CUDA_ARCH=native python3 ds4-parity/run_tool_call_quality.py \
    --server-bin target/debug/ds4-server-runtime-rs \
    --model /workspace/ds4/ds4flash.gguf --backend cuda \
    --runtime-graph graph --out-dir /tmp/ds4-m109e-tool-call-quality \
    --ready-timeout 360 --write-summary /tmp/ds4-m109e-tool-server.json \
    --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m109e-tool-server.json \
  ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json
```

The M10.9e comparator checks the Rust `ds4-server-runtime-rs` graph-route
tool-call quality run against current-C `./ds4_test --tool-call-quality` and
M9 server/runtime replay contracts: route `graph`, backend `cuda`, q2-imatrix
model hash, HTTP 200, `tool_calls`, `list_files`, `{"path":"."}`, trace
cache ledger markers, no target-stream fallback marker, and retained raw
request/response/header/trace/stdout/stderr artifacts.

Validate the M10.9f Runtime graph benchmark closure:

```sh
python3 ds4-parity/run_runtime_graph_bench.py
python3 ds4-parity/run_runtime_graph_bench.py --negative-test
```

Refresh the live B300 Rust runtime benchmark closure summary:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; CUDA_ARCH=native \
  python3 ds4-parity/run_runtime_graph_bench.py \
    --workdir /workspace/ds4 --model /workspace/ds4/ds4flash.gguf \
    --output-dir /tmp/ds4-m109f-bench \
    --write-summary /tmp/ds4-m109f-benchmark-closure.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m109f-benchmark-closure.json \
  ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json
```

The M10.9f comparator checks Rust `ds4-runtime-graph-bench-rs` short and
long benchmark CSVs against the M0.6 `ds4-bench` B300 baseline: route
`graph`, backend `cuda`, q2-imatrix model hash, prompt hash, context
frontiers, prefill intervals, generation-token counts, KVC snapshot bytes,
throughput threshold, M10.9a through M10.9e gate status, and the explicit
claim boundary that this closes Milestone 10 without claiming a general
backend replacement.

Validate the M11.1 Agent trace replay oracle:

```sh
arch -arm64 make ds4-agent
./ds4-agent --dump-agent-trace-oracle \
  ds4-parity/baselines/agent/m11.1/current-c.json
python3 ds4-parity/compare_agent_trace_replay.py --negative-test
```

The M11.1 comparator checks the no-model current-C agent replay fixture
against the Rust `ds4-agent-trace-replay-rs` emitter: normalized workspace and
session identifiers, scripted model events, parsed DSML tool sequence,
deterministic tool-output stubs, transcript role boundaries, session
save/list/switch/history/new operations, final visible output, and manifest
hashes. This is the first M11 gate; it defines replay fixtures before claiming
Rust ownership of the live agent loop.

Validate the M11.2 Agent rendered-context replay:

```sh
cargo run --quiet -p ds4-gguf --bin ds4-agent-rendered-context-rs \
  > ds4-parity/baselines/agent/m11.2/rendered-context.json
python3 ds4-parity/compare_agent_rendered_context.py --negative-test
```

The M11.2 comparator replays the M11.1 scripted fixture through Rust prompt
rendering and checks normalized rendered context: system/user/assistant/tool
role boundaries, assistant EOS insertion, raw DSML preservation, tool-result
tags, final visible assistant text, and absence of live session commands in
the prompt. It remains model-free and does not claim Rust ownership of tool
execution or the live agent loop.

Validate the M11.3 Agent deterministic tool/session replay:

```sh
cargo run --quiet -p ds4-gguf --bin ds4-agent-deterministic-replay-rs \
  > ds4-parity/baselines/agent/m11.3/deterministic-replay.json
python3 ds4-parity/compare_agent_deterministic_replay.py --negative-test
```

The M11.3 comparator replays the M11.1 deterministic `list` tool stub and
save/list/switch/history/new session commands against the Rust artifact. It
checks tool output insertion, transcript roles, normalized session ids, command
order and inputs, history text, final command-visible output, manifest hashes,
and the M11.2 rendered-context boundary for the tool result. It remains
model-free and does not claim live Rust agent-loop ownership.

Validate the M11.4 Agent no-model loop smoke:

```sh
cargo run --quiet -p ds4-gguf --bin ds4-agent-loop-smoke-rs \
  > ds4-parity/baselines/agent/m11.4/loop-smoke.json
python3 ds4-parity/compare_agent_loop_smoke.py --negative-test
```

The M11.4 comparator checks the replay-proven Rust loop smoke against M11.1
through M11.3: prompt rendering before and after tool insertion, Rust DSML
parser state, deterministic tool replay, session command order, normalized
session ids, and final visible outputs. It remains no-model and explicitly
defers model-backed manual smoke until the live Rust agent loop is enabled.

Validate the M12.1 Backend boundary inventory:

```sh
python3 ds4-parity/check_backend_boundary_inventory.py --negative-test
```

The M12.1 checker validates
`baselines/backend/m12.1/backend-boundary-inventory.json` against the M10.2
operation-family oracle, current backend ABI sources, M10.5c4c1 CUDA smoke
contract, M10.9 runtime graph/benchmark artifacts, B300 rerun commands, and
the no-removal/no-replacement claim policy. It remains inventory-only and does
not claim Rust-owned backend kernels.

Validate the M12.2 Backend operation tensor fixtures:

```sh
python3 ds4-parity/check_backend_operation_fixtures.py --negative-test
```

The M12.2 checker validates the live B300 fixture bundle under
`baselines/backend/m12.2/`: current-C oracle JSON and Rust facade candidate
JSON for first-kernel embedding, layer-0 QKV/RoPE, layer-0 attention output,
layer-0 FFN/router/MoE, and full output-head/logits. It reuses the existing
pair comparators, checks committed hashes/sizes, and keeps runtime routing and
backend replacement claims unchanged. The current fixture bundle passes with
576 checker assertions.

Validate the M12.3 Backend facade replay harness:

```sh
python3 ds4-parity/check_backend_facade_replay.py --negative-test
```

The M12.3 checker validates `baselines/backend/m12.3/facade-replay.json`
against the M12.2 fixture manifest, the M12.1 backend boundary inventory, the
Rust `DecodeBackend` facade operation table, and the selected candidate source
call order. It checks tensor bindings, synchronized command-batch evidence,
error propagation through `GpuStatus::from_raw(...).into_result()`, delegated
M12.2 output comparators, and the no-route-change/no-backend-replacement claim
policy. The current replay harness passes with 769 checker assertions.

Validate the M12.4 Backend replacement slice:

```sh
python3 ds4-parity/check_backend_replacement_slice.py --negative-test
```

The M12.4 checker validates `baselines/backend/m12.4/replacement-slice.json`
against the M12.2 first-kernel fixture and M12.3 facade replay. It also runs
the Rust `ds4-backend-replacement-slice` descriptor emitter, checks the
selected B300 CUDA backend path, verifies CPU/Metal/default-route selectors
fail closed, and keeps runtime routing, general backend replacement, and kernel
replacement claims false. The current M12.4 fixture passes with 85 checker
assertions.

Validate the M12.5 Backend runtime route gate:

```sh
python3 ds4-parity/check_backend_runtime_route_gate.py --negative-test
```

The M12.5 checker validates
`baselines/backend/m12.5/runtime-route-gate.json` against the M12.4 replacement
slice and the M10.9 B300 runtime graph evidence. It also runs the Rust
`ds4-backend-route-gate` emitter, checks the opt-in replacement route for the
`cuda-b300` backend, verifies default-route and unsupported-route fail-closed
behavior, and preserves the no-general-backend-replacement/no-kernel-replacement
claim boundary. The current M12.5 fixture passes with 135 checker assertions.

Validate the M12.6 Backend replacement closure:

```sh
python3 ds4-parity/check_backend_replacement_closure.py --negative-test
```

The M12.6 checker validates
`baselines/backend/m12.6/backend-replacement-closure.json` against M12.1
through M12.5 backend artifacts and the M10.9 B300 route evidence. It rejects
default-route activation, full-backend replacement claims, kernel replacement
claims, and any C/CUDA/Metal removal decision while only one embedding/indexer
operation has an opt-in route-gated replacement slice. The current M12.6
fixture passes with 147 checker assertions.

Validate the M13.0 Backend expansion decision:

```sh
python3 ds4-parity/check_backend_expansion_decision.py --negative-test
```

The M13.0 checker validates
`baselines/backend/m13.0/backend-expansion-decision.json` against the M12.6
closure matrix, M12.1 backend inventory, M10.2 graph inventory, and existing
M10.5/M10.6 prefill/indexed-attention comparator paths. It chooses to broaden
the existing embedding/indexer route, requires all six remaining M12.6
operations to be assigned to M13.1 through M13.5 work, and keeps removals,
default-route replacement, general backend replacement, and kernel replacement
claims false. The current M13.0 fixture passes with 186 checker assertions.

Validate the M13.1 Backend expansion matrix:

```sh
python3 ds4-parity/check_backend_expansion_matrix.py --negative-test
```

The M13.1 checker validates
`baselines/backend/m13.1/embedding-indexer-expansion-matrix.json` against the
M13.0 decision, M12.6 remaining-operation list, M12.1 backend inventory, and
M10.2 graph inventory. It checks current-C anchors, Rust facade or graph-plan
anchors, comparator paths, route-candidate stages, fixture-gap rows, and the
no-route-change/no-removal claim policy for every remaining embedding/indexer
operation. The current M13.1 fixture passes with 186 checker assertions.

Validate the M13.2 Batched embedding replacement slice:

```sh
python3 ds4-parity/check_backend_batched_embedding_slice.py --negative-test
```

The M13.2 checker validates
`baselines/backend/m13.2/batched-embedding-replacement-slice.json` against the
M13.1 matrix row for `ds4_gpu_embed_tokens_hc_tensor`, the Rust replacement
slice registry, the `ds4-backend-replacement-slice` emitter, and the M10.6
whole/chunked/resumed prefill comparators. It verifies the selected
`cuda-b300` backend path, CPU/Metal/default-route fail-closed behavior, and
keeps runtime routing, general backend replacement, and kernel replacement
claims false. The committed M13.2 fixture passes with 96 checker assertions.

Validate the M13.3 Indexed decode selection replacement slice:

```sh
python3 ds4-parity/check_backend_indexed_decode_slice.py --negative-test
```

The M13.3 checker validates
`baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json`
against the M13.1 matrix rows for `ds4_gpu_indexer_score_one_tensor` and
`ds4_gpu_indexer_topk_tensor`, the Rust replacement slice registry, the
`ds4-backend-replacement-slice` emitter, and the M10.5c4d3 long indexed
attention comparator. It keeps the two slices explicit, rejects ambiguous
`m13.3` selection, verifies the selected `cuda-b300` backend path,
CPU/Metal/default-route fail-closed behavior, and keeps runtime routing,
general backend replacement, and kernel replacement claims false. The committed
M13.3 fixture passes with 195 checker assertions.

Validate the M13.4 Batch indexer fixture gap closure:

```sh
python3 ds4-parity/check_backend_batch_indexer_fixtures.py --negative-test
```

The M13.4 checker validates
`baselines/backend/m13.4/batch-indexer-fixture-bundle.json` against the M13.1
fixture-gap rows for `ds4_gpu_indexer_scores_prefill_tensor`,
`ds4_gpu_indexer_scores_decode_batch_tensor`, and
`ds4_gpu_dsv4_topk_mask_tensor`. The bundle records B300-rerunnable current-C
fixture contracts, current-C/Rust source anchors, debug dump hooks, output
field and dtype contracts, and exact rerun commands. It keeps raw tensor bodies
out of the repository and keeps runtime routing, default-route replacement,
general backend replacement, and kernel replacement claims false. The committed
M13.4 fixture passes with 182 checker assertions.

Validate the M13.5 Expanded embedding/indexer route closure:

```sh
python3 ds4-parity/check_backend_expanded_route_closure.py --negative-test
```

The M13.5 checker validates
`baselines/backend/m13.5/expanded-route-gate.json` and
`baselines/backend/m13.5/expanded-route-closure.json` against the M13.1
through M13.4 embedding/indexer artifacts plus the M10.9 B300 runtime graph
evidence. It verifies the expanded route remains explicit opt-in, the default
route remains `current-backend`, the first M12.4/M13.2/M13.3 operations are
Rust replacement slices, and the M13.4 fixture-only operations remain
current-backend sidecars with removals blocked. The committed M13.5 fixture
passes with 279 checker assertions.

Validate the post-M13 roadmap decision:

```sh
python3 ds4-parity/check_post_m13_roadmap_decision.py --negative-test
```

The post-M13 checker validates
`baselines/roadmap/post-m13/post-m13-roadmap-decision.json` against the M13.0
through M13.5 artifacts and the M10.9 runtime evidence. It records that the
roadmap is complete through M13.5, no next implementation stage is selected,
the default route remains `current-backend`, current-backend sidecars remain
required, and C/GPU backend removals are blocked until a future roadmap starts
from new oracles. The committed post-M13 decision passes with 100 checker
assertions.

Validate the M10.4 current-C graph checkpoint oracle:

```sh
python3 ds4-parity/check_graph_checkpoint_oracle.py --negative-test
```

The checkpoint oracle is captured on B300 under `baselines/graph/m10.4/`.
It records selected device tensor checkpoints for short prefill, one-token
decode, layer-2 compressed KV state, long chunked prefill, and cache
continuation prefill. Exact checkpoints compare SHA256; long-context logits
compare selected f32 samples with the recorded tolerance. The MTP verifier
entry is explicitly skipped when no support MTP model is available in the
capture environment.

Compare the Rust `ds4-gpu-sys` ABI declarations against the M10.2 operation
oracle and `ds4_gpu.h`:

```sh
python3 ds4-parity/compare_gpu_sys_abi.py
```

Run its negative test before committing ABI-surface changes:

```sh
python3 ds4-parity/compare_gpu_sys_abi.py --negative-test
```

The ABI comparator checks that every graph backend primitive recorded in the
M10.2 oracle is declared in Rust with matching return and parameter types. Its
negative test removes one Rust declaration and mutates both Rust-side and
C-header parameter types in memory.

Compare the Rust M10.5b no-execute decode plan against the current-C decode
plan oracle:

```sh
python3 ds4-parity/compare_decode_plan_rust.py
python3 ds4-parity/compare_decode_plan_rust.py --negative-test
```

The decode plan oracle records the default `metal_graph_eval_token_raw_swa`
stage order, split-flush boundary, raw SWA row math, and compressed/indexer
counter transitions for representative first-token, short-prefill,
ratio-boundary, long-indexed, and no-logits cases. The negative test mutates a
stage, raw-start value, and indexed-layer count in memory.

Compare the Rust M10.5c1 structured DS4 weight table against the existing flat
tensor bindings and the C `ds4_weights`/`ds4_layer_weights` field inventory:

```sh
python3 ds4-parity/compare_rust_weight_table.py
python3 ds4-parity/compare_rust_weight_table.py --negative-test
```

The comparator builds a synthetic DS4 GGUF, asks `ds4-gguf-dump` for the
structured weight table, verifies that it flattens back to `bound_tensors`, and
checks base/layer field order plus dense, ratio-4, ratio-128, and hash-layer
presence rules. Its negative test mutates a layer, a field, and a presence bit
in memory.

Compare the Rust M10.5c2 decode graph-state plan against the M10.2 tensor owner
inventory:

```sh
python3 ds4-parity/compare_graph_state_plan.py
python3 ds4-parity/compare_graph_state_plan.py --negative-test
```

The comparator runs `ds4-graph-state-plan`, checks decode owner fields against
the M10.2 oracle, and pins the no-kernel allocation plan: `hc_*` views,
lazy `ffn_out`, directional steering as external input, full-capacity
persistent cache zero-fill obligations, and selected raw/ratio-4/ratio-128
cache byte sizes. Its negative test mutates summary, field, and view data in
memory.

Compare the Rust M10.5c3 decode backend facade against the default fused
one-token decode primitive list:

```sh
python3 ds4-parity/compare_decode_backend_facade.py
python3 ds4-parity/compare_decode_backend_facade.py --negative-test
```

The comparator checks the facade operation table against the M10.2 operation
inventory and M10.5a ABI declarations, verifies tensor argument order from each
safe Rust method signature, and keeps command, read, view, and sync operations
anchored to the existing lifecycle wrappers. Its negative test mutates a
facade entry, tensor argument order, and raw sys call in memory.

Compare the Rust M10.5c4a dry-run decode execution trace against the M10.5b
decode plan oracle:

```sh
python3 ds4-parity/compare_decode_trace.py
python3 ds4-parity/compare_decode_trace.py --negative-test
```

The comparator runs `ds4-decode-trace`, checks the dry-run trace schema, case
set, layer stage order, facade method/tensor-argument coverage, command/read/
sync markers, raw/compressed/indexer cache-counter transitions, and default
decode operation coverage. Its negative test mutates summary, operation, and
state data, including split-flush and emit-cadence fields, in memory.

Compare the Rust M10.5c4b dry-run decode runtime bridge against the graph-state
plan, structured weight table, and decode trace:

```sh
python3 ds4-parity/compare_decode_runtime_bridge.py
python3 ds4-parity/compare_decode_runtime_bridge.py --negative-test
```

The comparator runs `ds4-decode-runtime-bridge`, checks graph-state handle
ownership/storage against M10.5c2, validates initial cache counters, resolves
facade tensor arguments from the M10.5c4a trace, and checks selected dense,
ratio-4, ratio-128, and hash-layer weight roles against the M10.5c1 structured
weight table. Its negative test mutates summary, handle, binding, and
weight-role data in memory.

Compare the M10.5c4c1 Rust CUDA backend smoke contract before running the B300
ABI smoke:

```sh
python3 ds4-parity/compare_b300_rust_backend_smoke.py
python3 ds4-parity/compare_b300_rust_backend_smoke.py --negative-test
```

The comparator checks that `ds4-gpu` exposes a feature-gated `cuda-backend`
Linux build path, that the build script tracks and links the C/CUDA backend
sources and CUDA libraries, that the backend ABI smoke test is enabled for
B300 only under that feature, and that the unified report carries the exact
B300 rerun command.

Compare the M10.5c4c2a Rust decode model-map bridge before running the B300
model-map backend smoke:

```sh
python3 ds4-parity/compare_decode_model_map_bridge.py
python3 ds4-parity/compare_decode_model_map_bridge.py --negative-test
```

The comparator checks that the Rust decode backend exposes safe wrappers for
model map, file descriptor, map-range, and CUDA cache-range backend calls,
that CUDA-only cache wrappers stay Linux-gated, that the B300 test exercises
fd/map/range/cache success and failure paths, and that the unified report
carries the exact B300 rerun command.

Compare the M10.5c4c2b1 Rust decode execution preflight before running the
B300 model-backed preflight binary:

```sh
python3 ds4-parity/compare_decode_execution_preflight.py
python3 ds4-parity/compare_decode_execution_preflight.py --negative-test
```

The comparator checks that the Rust preflight can mmap the real GGUF, parse the
header without copying tensor data, bind DS4 weights, hand the tensor-data range
to the backend, allocate representative M10.4 checkpoint tensors, and exercise
bounded model/Q8 cache hooks. On B300, validate the emitted JSON too:

```sh
python3 ds4-parity/compare_decode_execution_preflight.py \
  --candidate /tmp/ds4-c2b1-preflight.json
```

Compare the M10.5c4c2b2a Rust full decode state allocation before scheduling
the one-token decode facade:

```sh
python3 ds4-parity/compare_decode_state_allocation.py
python3 ds4-parity/compare_decode_state_allocation.py --negative-test
```

The comparator checks that `ds4-decode-state-alloc` walks the M10.5c2
graph-state table, allocates every initially owned tensor for the
`ctx32768_mtp_off` plan, applies zero and negative-infinity initialization,
creates the planned `hc_*` views, and releases the backend through cleanup. On
B300, validate the emitted JSON too:

```sh
python3 ds4-parity/compare_decode_state_allocation.py \
  --candidate /tmp/ds4-c2b2a-state-allocation.json
```

Compare the M10.5c4c2b2b1 Rust first decode kernel before running the full
one-token scheduler:

```sh
python3 ds4-parity/compare_decode_first_kernel.py
python3 ds4-parity/compare_decode_first_kernel.py --negative-test
```

The comparator checks that `ds4-decode-first-kernel` maps the real GGUF,
binds DS4 weights, opens a command batch, calls the Rust `embed_token_hc`
facade with `base.token_embd`, synchronizes, reads `cur_hc`, and cleans up the
backend. On B300, validate the emitted JSON too:

```sh
python3 ds4-parity/compare_decode_first_kernel.py \
  --candidate /tmp/ds4-c2b2b1-first-kernel.json
```

Compare the M10.5c4c2b2b2a Rust first-kernel current-C oracle before adding
more one-token scheduler calls:

```sh
python3 ds4-parity/compare_decode_first_kernel_oracle.py
python3 ds4-parity/compare_decode_first_kernel_oracle.py --negative-test
```

The comparator checks that `ds4-first-kernel-oracle-dump` uses the current C
model loader, DS4 weight binding, `embed_token_f16`, and
`hc_from_plain_embedding` to emit an independent `cur_hc` oracle. On B300,
validate the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_first_kernel_oracle.py \
  --oracle /tmp/ds4-c2b2b2a-first-kernel-oracle.json \
  --candidate /tmp/ds4-c2b2b2a-first-kernel-rust.json
```

Compare the M10.5c4c2b2b2b1 Rust layer-0 attention HC-pre prefix before
adding more one-token scheduler calls:

```sh
python3 ds4-parity/compare_decode_layer0_attn_hc_pre.py
python3 ds4-parity/compare_decode_layer0_attn_hc_pre.py --negative-test
```

The comparator checks that `ds4-layer0-attn-hc-pre-oracle-dump` uses the
current C model loader, DS4 weight binding, model fd/map bridge, GPU embedding,
GPU HC RMS normalization, layer-0 GPU `hc_attn_fn`, and the fused GPU HC
split/weighted-sum/attention-norm kernel to emit independent tensor digests. On
B300, validate the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer0_attn_hc_pre.py \
  --oracle /tmp/ds4-c2b2b2b1-layer0-attn-hc-pre-oracle.json \
  --candidate /tmp/ds4-c2b2b2b1-layer0-attn-hc-pre-rust.json
```

Compare the M10.5c4c2b2b2b2a Rust layer-0 QKV/RoPE prefix before adding
the full one-token scheduler:

```sh
python3 ds4-parity/compare_decode_layer0_qkv_rope.py
python3 ds4-parity/compare_decode_layer0_qkv_rope.py --negative-test
```

The comparator checks that `ds4-layer0-qkv-rope-oracle-dump` emits the
current-C GPU tensor path through HC-pre, Q/KV projection, fused QKV RMS norm,
dense Q projection, head RMS norm, and dense RoPE. On B300, validate the
current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer0_qkv_rope.py \
  --oracle /tmp/ds4-c2b2b2b2a-layer0-qkv-rope-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2a-layer0-qkv-rope-rust.json
```

Compare the M10.5c4c2b2b2b2b1 Rust layer-0 attention-output prefix before
adding the full one-token scheduler:

```sh
python3 ds4-parity/compare_decode_layer0_attn_output.py
python3 ds4-parity/compare_decode_layer0_attn_output.py --negative-test
```

The comparator checks that `ds4-layer0-attn-output-oracle-dump` emits the
current-C GPU tensor path through QKV/RoPE, dense raw-KV store, dense attention
decode, inverse RoPE, low-rank attention output, and final HC expansion. On
B300, validate the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer0_attn_output.py \
  --oracle /tmp/ds4-c2b2b2b2b1-layer0-attn-output-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b1-layer0-attn-output-rust.json
```

Compare the M10.5c4c2b2b2b2b2a Rust layer-0 FFN-output body before
expanding to the full one-token scheduler:

```sh
python3 ds4-parity/compare_decode_layer0_ffn_output.py
python3 ds4-parity/compare_decode_layer0_ffn_output.py --negative-test
```

The comparator checks that `ds4-layer0-ffn-output-oracle-dump` emits the
current-C GPU tensor path through the already-validated attention-output
boundary, FFN HC-pre, router selection, routed MoE, shared expert SwiGLU,
shared down projection, and final FFN HC expansion. On B300, validate the
current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer0_ffn_output.py \
  --oracle /tmp/ds4-c2b2b2b2b2a-layer0-ffn-output-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2a-layer0-ffn-output-rust.json
```

Compare the M10.5c4c2b2b2b2b2b1 Rust layer-0 output-head body before
expanding to the full one-token scheduler:

```sh
python3 ds4-parity/compare_decode_layer0_output_head.py
python3 ds4-parity/compare_decode_layer0_output_head.py --negative-test
```

The comparator checks that `ds4-layer0-output-head-oracle-dump` emits the
current-C GPU tensor path through the production decode-layer encoder and
production output-head encoder, then compares post-FFN HC, output HC collapse,
output embedding norm, and logits readback. On B300, validate the current-C
oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer0_output_head.py \
  --oracle /tmp/ds4-c2b2b2b2b2b1-layer0-output-head-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b1-layer0-output-head-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2a Rust two-layer output-head body before
adding compressed layer-2 cache mutation and the full one-token scheduler:

```sh
python3 ds4-parity/compare_decode_two_layer_output_head.py
python3 ds4-parity/compare_decode_two_layer_output_head.py --negative-test
```

The comparator checks that `ds4-two-layer-output-head-oracle-dump` emits the
current-C GPU tensor path through the production decode-layer encoder for
layers 0 and 1, the production HC buffer swap after each layer, and the
production output-head encoder, then compares both layer HC boundaries, output
HC collapse, output embedding norm, and logits readback. On B300, validate the
current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_two_layer_output_head.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2a-two-layer-output-head-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2a-two-layer-output-head-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b1 Rust layer-2 compressor-state body before
adding compressed attention, FFN, remaining layers, and final logits:

```sh
python3 ds4-parity/compare_decode_layer2_compressor_state.py
python3 ds4-parity/compare_decode_layer2_compressor_state.py --negative-test
```

The comparator checks that `ds4-layer2-compressor-state-oracle-dump` emits the
current-C GPU tensor path through dense layers 0 and 1, the production HC
buffer swap after each dense layer, and layer 2 through raw KV store plus the
ratio-4 attention and indexer compressor-state updates. On B300, validate the
current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer2_compressor_state.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b1-layer2-compressor-state-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b1-layer2-compressor-state-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b2a Rust layer-2 attention-output body before
adding the layer-2 FFN, remaining layers, and final logits:

```sh
python3 ds4-parity/compare_decode_layer2_attn_output.py
python3 ds4-parity/compare_decode_layer2_attn_output.py --negative-test
```

The comparator checks that `ds4-layer2-attn-output-oracle-dump` emits the
current-C GPU tensor path through dense layers 0 and 1, the production HC
buffer swap after each dense layer, and layer 2 through compressor-state
mutation, raw-only attention decode, inverse compressed RoPE, attention output,
and after-attention HC expansion. On B300, validate the current-C oracle and
Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer2_attn_output.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b2a-layer2-attn-output-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b2a-layer2-attn-output-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b2b1 Rust layer-2 FFN-output body before
adding the remaining compressed layers, ratio-128 coverage, the output head,
and final logits:

```sh
python3 ds4-parity/compare_decode_layer2_ffn_output.py
python3 ds4-parity/compare_decode_layer2_ffn_output.py --negative-test
```

The comparator checks that `ds4-layer2-ffn-output-oracle-dump` emits the
current-C GPU tensor path through dense layers 0 and 1, the production HC
buffer swap after each dense layer, and layer 2 through attention-output,
FFN HC-pre, router selection, routed MoE, shared expert, and after-FFN HC
expansion. On B300, validate the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer2_ffn_output.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b2b1-layer2-ffn-output-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b2b1-layer2-ffn-output-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b2b2a Rust layer-3 ratio-128 FFN-output body
before adding the remaining compressed layers, output head, and final logits:

```sh
python3 ds4-parity/compare_decode_layer3_ffn_output.py
python3 ds4-parity/compare_decode_layer3_ffn_output.py --negative-test
```

The comparator checks that `ds4-layer3-ffn-output-oracle-dump` emits the
current-C GPU tensor path through dense layers 0 and 1, the layer-2 ratio-4
FFN-output boundary plus HC swap, and layer 3 through ratio-128 compressor
state, raw-only attention decode, attention output, FFN HC-pre, router
selection, routed MoE, shared expert, and after-FFN HC expansion. On B300,
validate the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer3_ffn_output.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b2b2a-layer3-ffn-output-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b2b2a-layer3-ffn-output-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b2b2b1 Rust layer-4 post-ratio128
ratio-4/indexer FFN-output body before adding the remaining compressed layers,
output head, and final logits:

```sh
python3 ds4-parity/compare_decode_layer4_ffn_output.py
python3 ds4-parity/compare_decode_layer4_ffn_output.py --negative-test
```

The comparator checks that `ds4-layer4-ffn-output-oracle-dump` emits the
current-C GPU tensor path through dense layers 0 and 1, the layer-2 ratio-4
FFN-output boundary plus HC swap, the layer-3 ratio-128 FFN-output boundary
plus HC swap, and layer 4 through ratio-4 attention/indexer compressor state,
raw-only attention decode, attention output, FFN HC-pre, router selection,
routed MoE, shared expert, and after-FFN HC expansion. On B300, validate the
current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_layer4_ffn_output.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b2b2b2a Rust all-layer final-HC body before
adding the output head and final logits:

```sh
python3 ds4-parity/compare_decode_all_layer_final_hc.py
python3 ds4-parity/compare_decode_all_layer_final_hc.py --negative-test
```

The comparator checks that `ds4-all-layer-final-hc-oracle-dump` emits the
current-C GPU tensor path through all 43 production decode-layer encoder calls,
the production HC buffer swap after each layer, final HC checkpoints after
layers 4, 5, and 42, and representative ratio-128 and ratio-4/indexer
compressor state. On B300, validate the current-C oracle and Rust readback
together:

```sh
python3 ds4-parity/compare_decode_all_layer_final_hc.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b2b2b2a-all-layer-final-hc-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b2b2b2a-all-layer-final-hc-rust.json
```

Compare the M10.5c4c2b2b2b2b2b2b2b2b2b Rust full output-head/logits body
after all 43 decode layers:

```sh
python3 ds4-parity/compare_decode_full_output_head.py
python3 ds4-parity/compare_decode_full_output_head.py --negative-test
```

The comparator checks that `ds4-full-output-head-oracle-dump` emits the
current-C GPU tensor path through all 43 production decode-layer encoder calls,
the production HC buffer swap after each layer, final layer-42 HC readback, and
the output-head sequence `rms_norm_plain`, `matmul_f16`,
`output_hc_weights`, `hc_weighted_sum`, `rms_norm_weight`, and vocab
`matmul_q8_0`. On B300, validate the current-C oracle and Rust readback
together:

```sh
python3 ds4-parity/compare_decode_full_output_head.py \
  --oracle /tmp/ds4-c2b2b2b2b2b2b2b2b2b-full-output-head-oracle.json \
  --candidate /tmp/ds4-c2b2b2b2b2b2b2b2b2b-full-output-head-rust.json
```

Compare the M10.5c4d1 Rust short decode-continuation output-head/logits body
after the deterministic token sequence `0..21`:

```sh
python3 ds4-parity/compare_decode_short_continuation_output_head.py
python3 ds4-parity/compare_decode_short_continuation_output_head.py --negative-test
```

The comparator checks that `ds4-short-continuation-output-head-oracle-dump`
emits the current-C GPU continuation path through 22 production
`metal_graph_eval_token_raw_swa` calls, final layer-42 HC readback, output-head
tensors, logits, selected raw-cache rows, and selected ratio-4/ratio-128
compressed cache state. On B300, validate the current-C oracle and Rust
readback together:

```sh
python3 ds4-parity/compare_decode_short_continuation_output_head.py \
  --oracle /tmp/ds4-c4d1-short-continuation-output-head-oracle.json \
  --candidate /tmp/ds4-c4d1-short-continuation-output-head-rust.json
```

Compare the M10.5c4d2 Rust ratio-boundary decode-continuation output-head/logits
body after the deterministic token sequence `0..127`:

```sh
python3 ds4-parity/compare_decode_ratio_boundary_output_head.py
python3 ds4-parity/compare_decode_ratio_boundary_output_head.py --negative-test
```

The comparator checks that `ds4-ratio-boundary-output-head-oracle-dump` emits
the current-C GPU continuation path through 128 production
`metal_graph_eval_token_raw_swa` calls, the final layer-42 HC readback,
output-head tensors, logits, ratio-4 row 31, ratio-128 row 0, and selected
raw/cache state. On B300, validate the current-C oracle and Rust readback
together:

```sh
python3 ds4-parity/compare_decode_ratio_boundary_output_head.py \
  --oracle /tmp/ds4-c4d2-ratio-boundary-output-head-oracle.json \
  --candidate /tmp/ds4-c4d2-ratio-boundary-output-head-rust.json
```

Compare the M10.5c4d3 Rust long indexed-attention decode branch after the
deterministic token sequence `0..2051`:

```sh
python3 ds4-parity/compare_decode_long_indexed_attention.py
python3 ds4-parity/compare_decode_long_indexed_attention.py --negative-test
```

The comparator checks that `ds4-long-indexed-attention-oracle-dump` emits the
current-C GPU continuation path through 2,051 production
`metal_graph_eval_token_raw_swa` warmup calls, then manually runs token 2051
through layer 2 so the strict ratio-4 `DS4_N_INDEXER_TOP_K` threshold is crossed
at the first indexed layer. It records the selected top-k compressed rows,
indexer scores, indexed-attention heads, and final layer-2 outputs. On B300,
validate the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_long_indexed_attention.py \
  --oracle /tmp/ds4-c4d3-long-indexed-attention-oracle.json \
  --candidate /tmp/ds4-c4d3-long-indexed-attention-rust.json
```

Compare the M10.5c4d4 Rust directional-steering decode branch for token `0`,
layer `0`, `dir-steering/out/verbosity.f32`, attention scale `0.5`, and FFN
scale `0.25`:

```sh
python3 ds4-parity/compare_decode_directional_steering.py
python3 ds4-parity/compare_decode_directional_steering.py --negative-test
```

The comparator checks that `ds4-directional-steering-oracle-dump` emits the
current-C GPU steering path around attention output and FFN output projection,
then validates the Rust safe facade readback for layer-0 post-steer attention,
post-steer HC expansion, post-steer FFN, and final logits. On B300, validate
the current-C oracle and Rust readback together:

```sh
python3 ds4-parity/compare_decode_directional_steering.py \
  --oracle /tmp/ds4-c4d4-directional-steering-oracle.json \
  --candidate /tmp/ds4-c4d4-directional-steering-rust.json
```

Compare the M10.6a Rust prefill scheduling plan before executing prefill GPU
kernels:

```sh
python3 ds4-parity/compare_prefill_plan_rust.py
python3 ds4-parity/compare_prefill_plan_rust.py --negative-test
cargo run -p ds4-gpu --bin ds4-prefill-plan --quiet > /tmp/ds4-m106a-prefill-plan.json
python3 ds4-parity/compare_prefill_plan_rust.py \
  --candidate /tmp/ds4-m106a-prefill-plan.json
```

The comparator checks that Rust mirrors the current-C default prefill cap,
whole-vs-chunked routing, resumed-suffix threshold, absolute prefill-cap chunk
alignment, final output batch row, progress points, and layer-batch call counts
for the M10.6a fixtures.

Compare the M10.6b Rust whole-prefill short-prompt execution against the
current-C layer-major prefill oracle:

```sh
python3 ds4-parity/compare_prefill_whole_short.py
python3 ds4-parity/compare_prefill_whole_short.py --negative-test
make ds4-prefill-whole-short-oracle-dump CUDA_ARCH=native
./ds4-prefill-whole-short-oracle-dump --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/short_italian_fact.txt \
  --backend cuda \
  --output /tmp/ds4-m106b-prefill-whole-short-oracle.json
CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend \
  --bin ds4-prefill-whole-short --quiet -- \
  --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/short_italian_fact.txt \
  > /tmp/ds4-m106b-prefill-whole-short-rust.json
python3 ds4-parity/compare_prefill_whole_short.py \
  --oracle /tmp/ds4-m106b-prefill-whole-short-oracle.json \
  --candidate /tmp/ds4-m106b-prefill-whole-short-rust.json
```

The comparator checks that Rust uses the same rendered chat prompt tokens,
2048-row prefill-cap graph plan, 21 active prompt rows, final-row output head,
and layer-major raw/compressed cache counters as current C.

Compare the M10.6c Rust cold chunked-prefill execution against the current-C
chunked oracle. On CUDA, run both sides with `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`
so the MoE down-projection path is deterministic enough for exact digest
comparison:

```sh
python3 ds4-parity/compare_prefill_chunked.py
python3 ds4-parity/compare_prefill_chunked.py --negative-test
export DS4_CUDA_MOE_NO_ATOMIC_DOWN=1
make ds4-prefill-whole-short-oracle-dump CUDA_ARCH=native
./ds4-prefill-whole-short-oracle-dump --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 2052 \
  --backend cuda \
  --output /tmp/ds4-m106c-prefill-chunked-2052-oracle.json
CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend \
  --bin ds4-prefill-whole-short --quiet -- \
  --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 2052 \
  > /tmp/ds4-m106c-prefill-chunked-2052-rust.json
python3 ds4-parity/compare_prefill_chunked.py \
  --oracle /tmp/ds4-m106c-prefill-chunked-2052-oracle.json \
  --candidate /tmp/ds4-m106c-prefill-chunked-2052-rust.json
./ds4-prefill-whole-short-oracle-dump --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --backend cuda \
  --output /tmp/ds4-m106c-prefill-chunked-long-oracle.json
CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend \
  --bin ds4-prefill-whole-short --quiet -- \
  --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  > /tmp/ds4-m106c-prefill-chunked-long-rust.json
python3 ds4-parity/compare_prefill_chunked.py \
  --oracle /tmp/ds4-m106c-prefill-chunked-long-oracle.json \
  --candidate /tmp/ds4-m106c-prefill-chunked-long-rust.json
```

The comparator checks the 2048+4 and 2048+1305 chunk schedules, absolute raw
ring rows, per-layer compressed counters, final-row output head dimensions, and
exact output digests/samples against the current-C chunked oracle.

Compare the M10.6d Rust resumed-suffix prefill execution against the current-C
`ds4_session_sync` oracle. On CUDA, run both sides with
`DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` for exact digest comparison:

```sh
python3 ds4-parity/compare_prefill_resumed.py
python3 ds4-parity/compare_prefill_resumed.py --negative-test
export DS4_CUDA_MOE_NO_ATOMIC_DOWN=1
make ds4-prefill-whole-short-oracle-dump CUDA_ARCH=native
./ds4-prefill-whole-short-oracle-dump --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 512 \
  --resume-prefix-tokens 512 \
  --backend cuda \
  --output /tmp/ds4-m106d-prefill-resumed-cache-oracle.json
CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend \
  --bin ds4-prefill-whole-short --quiet -- \
  --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 512 \
  --resume-prefix-tokens 512 \
  > /tmp/ds4-m106d-prefill-resumed-cache-rust.json
python3 ds4-parity/compare_prefill_resumed.py \
  --oracle /tmp/ds4-m106d-prefill-resumed-cache-oracle.json \
  --candidate /tmp/ds4-m106d-prefill-resumed-cache-rust.json
./ds4-prefill-whole-short-oracle-dump --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 514 \
  --resume-prefix-tokens 512 \
  --backend cuda \
  --output /tmp/ds4-m106d-prefill-resumed-decode-oracle.json
CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend \
  --bin ds4-prefill-whole-short --quiet -- \
  --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 514 \
  --resume-prefix-tokens 512 \
  > /tmp/ds4-m106d-prefill-resumed-decode-rust.json
python3 ds4-parity/compare_prefill_resumed.py \
  --oracle /tmp/ds4-m106d-prefill-resumed-decode-oracle.json \
  --candidate /tmp/ds4-m106d-prefill-resumed-decode-rust.json
./ds4-prefill-whole-short-oracle-dump --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 2337 \
  --resume-prefix-tokens 1537 \
  --backend cuda \
  --output /tmp/ds4-m106d-prefill-resumed-chunked-oracle.json
CUDA_ARCH=native cargo run -p ds4-gpu --features cuda-backend \
  --bin ds4-prefill-whole-short --quiet -- \
  --model /path/to/ds4flash.gguf \
  --prompt tests/test-vectors/prompts/long_memory_archive.txt \
  --limit-tokens 2337 \
  --resume-prefix-tokens 1537 \
  > /tmp/ds4-m106d-prefill-resumed-chunked-rust.json
python3 ds4-parity/compare_prefill_resumed.py \
  --oracle /tmp/ds4-m106d-prefill-resumed-chunked-oracle.json \
  --candidate /tmp/ds4-m106d-prefill-resumed-chunked-rust.json
```

The comparator checks exact-prefix cache-hit, decode-suffix, and resumed
chunked-prefill route decisions, resume threshold handling, extension chunk
boundaries, decode-token counts, raw ring rows, compressed counters, and exact
output digests/samples.

Compare the M10.7a Rust graph-session payload layout plan against the current-C
no-model graph payload oracle:

```sh
python3 ds4-parity/compare_graph_session_payload_plan.py
python3 ds4-parity/compare_graph_session_payload_plan.py --negative-test
arch -arm64 make ds4-session-payload-dump
./ds4-session-payload-dump --graph-plan \
  > /tmp/ds4-m107a-graph-payload-c.json
cargo run -p ds4-gguf --bin ds4-session-payload-dump-rs --quiet -- \
  --graph-plan \
  > /tmp/ds4-m107a-graph-payload-rust.json
python3 ds4-parity/compare_graph_session_payload_plan.py \
  --oracle /tmp/ds4-m107a-graph-payload-c.json \
  --candidate /tmp/ds4-m107a-graph-payload-rust.json
```

The comparator checks the default graph payload header plan, prefill/raw/comp
caps, logical and physical raw ring order, ratio-4 and ratio-128 row counts,
per-section byte totals, sampled per-layer row/state bytes, and final payload
size without loading a model or restoring tensors.

Compare the M10.7b Rust graph-session payload reader/writer helpers against the
current-C no-model graph payload rejection probe:

```sh
python3 ds4-parity/compare_graph_session_payload_rw.py
python3 ds4-parity/compare_graph_session_payload_rw.py --negative-test
arch -arm64 make ds4-session-payload-dump
./ds4-session-payload-dump --graph-probe \
  > /tmp/ds4-m107b-graph-payload-rw-c.json
cargo run -p ds4-gguf --bin ds4-session-payload-dump-rs --quiet -- \
  --graph-probe \
  > /tmp/ds4-m107b-graph-payload-rw-rust.json
python3 ds4-parity/compare_graph_session_payload_rw.py \
  --oracle /tmp/ds4-m107b-graph-payload-rw-c.json \
  --candidate /tmp/ds4-m107b-graph-payload-rw-rust.json
```

The comparator checks byte-identical synthetic graph payload writes by FNV,
parsed raw-ring summaries for short and wrapped payloads, section byte totals,
and the C-compatible rejection codes for truncated, trailing, invalid
compressed/index counts, raw-ring, context, layout, chunk-layout, and comp-cap
boundary cases without restoring tensors.

Compare the M10.7c1 Rust restore payload header plan against the committed
M7.8 B300 current-C restore oracle:

```sh
python3 ds4-parity/compare_restore_payload_header_plan.py
python3 ds4-parity/compare_restore_payload_header_plan.py --negative-test
cargo run -p ds4-gguf --bin ds4-session-payload-dump-rs --quiet -- \
  --restore-header-plan \
  > /tmp/ds4-m107c1-restore-header-rust.json
python3 ds4-parity/compare_restore_payload_header_plan.py \
  --candidate /tmp/ds4-m107c1-restore-header-rust.json
```

The comparator checks M7.8 disk and memory-snapshot restore records for seed and
continuation prompts, including case order, model identity, prompt tokens,
DSV4 header bytes, graph caps, raw-live rows, payload/snapshot byte counts, and
the hash-only raw-body policy. It does not require raw restore bodies or claim
tensor restore behavior.

Compare the M10.7c2 Rust raw graph payload import summary against the committed
M7.8 disk-payload oracle:

```sh
python3 ds4-parity/compare_graph_payload_raw_import.py
python3 ds4-parity/compare_graph_payload_raw_import.py --negative-test
```

Recapture the Rust raw import summary on the B300 pod, where the hash-only raw
payload bodies remain:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; python3 ds4-parity/compare_graph_payload_raw_import.py --live --workdir /workspace/ds4 --write-summary /tmp/ds4-m107c2-raw-import.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m107c2-raw-import.json \
  ds4-parity/baselines/kv/m10.7c2/rust-b300-raw-import.json
```

The comparator records the observed disk seed and continuation raw payload
SHA256 values plus the historical oracle SHA256 values, and exact-gates byte
counts, Rust graph-reader acceptance, parsed header fields, raw-ring mapping,
section byte plan, compressed/index row counts, and hash-only raw-body policy.
The raw body SHA is per-capture metadata because unused or numerically unstable
payload bytes can drift across B300 captures; this check does not restore
tensors into graph memory.

Compare the M10.7c3a Rust raw graph snapshot import summary against the
committed M7.8 memory-snapshot oracle:

```sh
python3 ds4-parity/compare_graph_snapshot_raw_import.py
python3 ds4-parity/compare_graph_snapshot_raw_import.py --negative-test
```

Recapture the memory snapshot bodies and Rust import summary on the B300 pod:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; make ds4-restore-dump CUDA_ARCH=native; mkdir -p ds4-parity/baselines/kv/m7.8/raw; ./ds4-restore-dump --backend cuda -m /workspace/ds4/ds4flash.gguf --model-sha256 efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668 --seed-prompt ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_prompt.txt --seed-assistant ds4-parity/baselines/kv-fixtures/m7.8/restore_seed_assistant.txt --continuation-user ds4-parity/baselines/kv-fixtures/m7.8/restore_continuation_user.txt --payload-dir ds4-parity/baselines/kv/m7.8/raw --snapshot-dir ds4-parity/baselines/kv/m7.8/raw -o /tmp/ds4-m107c3a-current-c-with-snapshots.json; python3 ds4-parity/check_restore_dump.py /tmp/ds4-m107c3a-current-c-with-snapshots.json --negative-test; python3 ds4-parity/compare_graph_snapshot_raw_import.py --live --workdir /workspace/ds4 --write-summary /tmp/ds4-m107c3a-snapshot-raw-import.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m107c3a-snapshot-raw-import.json \
  ds4-parity/baselines/kv/m10.7c3a/rust-b300-snapshot-raw-import.json
```

The comparator records the observed seed and continuation memory snapshot
SHA256 values plus the historical oracle SHA256 values, and exact-gates byte
counts, Rust graph-reader acceptance, parsed header fields, raw-ring mapping,
section byte plan, compressed/index row counts, and hash-only raw-body policy.
It materializes snapshot bodies only on B300; the raw body SHA is per-capture
metadata and the check still does not restore tensors into graph memory.

Compare the M10.7c3b Rust graph restore target plan against the current-C graph
restore order:

```sh
python3 ds4-parity/compare_graph_restore_target_plan.py
python3 ds4-parity/compare_graph_restore_target_plan.py --negative-test
cargo run -p ds4-gguf --bin ds4-session-payload-dump-rs --quiet -- \
  --restore-target-plan \
  > /tmp/ds4-m107c3b-restore-target-rust.json
python3 ds4-parity/compare_graph_restore_target_plan.py \
  --candidate /tmp/ds4-m107c3b-restore-target-rust.json
```

The comparator checks the four M7.8 disk payload and memory snapshot cases over
checkpoint/logit/count-table targets, raw logical-to-physical ring row mapping,
per-layer compressed-cache and state-tensor targets, ratio-4 indexer targets,
and post-restore counter state. It uses parsed metadata only and does not move
bytes into graph tensors.

Compare the M10.7c3c Rust graph restore tensor readback summary against the
current B300 raw-body summaries:

```sh
python3 ds4-parity/compare_graph_restore_readback.py
python3 ds4-parity/compare_graph_restore_readback.py --negative-test
```

Recapture the Rust graph restore readback summary on the B300 pod, where the
hash-only disk payload and memory snapshot bodies remain:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; python3 ds4-parity/compare_graph_restore_readback.py --live --workdir /workspace/ds4 --write-summary /tmp/ds4-m107c3c-restore-readback.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m107c3c-restore-readback.json \
  ds4-parity/baselines/kv/m10.7c3c/rust-b300-restore-readback.json
```

The comparator checks that Rust writes the four M7.8 raw disk/snapshot payloads
into Rust-owned graph tensors, reads the written spans back in C restore order,
and matches source/readback FNVs for checkpoint tokens, logits, count tables,
raw rows, compressed rows, attention/indexer state tensors, sampled layers, and
post-restore counters. It still does not execute decode or claim next-token
behavior.

Compare the M10.7c3d Rust graph restore next-token summary, refreshed for
M10.7d3b frontier projection, against the same-capture current-C restore oracle,
tensor-readback evidence, and restore-frontier contract:

```sh
python3 ds4-parity/compare_graph_restore_next_token.py
python3 ds4-parity/compare_graph_restore_next_token.py --negative-test
```

Recapture the Rust graph restore next-token summary on the B300 pod, where the
hash-only disk payload and memory snapshot bodies remain:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; CUDA_ARCH=native python3 ds4-parity/compare_graph_restore_next_token.py --live --workdir /workspace/ds4 --write-summary /tmp/ds4-m107d3b-restore-next-token.json --negative-test'
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp ds4-rust-port-b300:/tmp/ds4-m107d3b-restore-next-token.json \
  ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json
```

The live comparator first recaptures the current-C restore oracle because raw
payload bodies are per-capture evidence. It then checks the Rust-restored graph
payload state over payload hashes, checkpoint length and FNV, restored-logits
FNV, selected token, top-logprob order and scores, cache source, same-capture
Rust readback evidence, committed M10.7c3c readback evidence, and post-restore
graph counters. For M10.7d3b it also checks each restored payload's loaded
frontier, unaligned current-live skip, next continued-store target,
already-stored boundary skip, and shutdown reason projection against the
M10.7d3 restore-frontier contract.

Validate the M10.7d3a graph restore continued-frontier contract:

```sh
python3 ds4-parity/check_graph_restore_frontier_contract.py
python3 ds4-parity/check_graph_restore_frontier_contract.py --negative-test
```

The checker compares the committed M10.7d3 restore-frontier contract against
the M10.7c3d restored-token evidence and the M7.2 current-C continued-frontier
policy oracle. It pins restored token counts, loaded frontier values,
re-enabled continued-store targets, already-stored skip behavior, and KVC
reason-code references without requiring a B300 recapture.

Validate the M10.7d3c1 post-restore KVC decision contract:

```sh
python3 ds4-parity/check_post_restore_kvc_decision_contract.py
python3 ds4-parity/check_post_restore_kvc_decision_contract.py --negative-test
```

The checker compares the committed post-restore KVC decision contract against
the M10.7d3b restored-frontier projection, the M10.7d2 runtime ledger
contract, the M9.8f5 B300 runtime replay summary, and the M7.4a KVC file
layout oracle. It pins unaligned continued-store skips, re-enabled next
continued targets, already-stored boundary skips, and shutdown-write header
expectations before the B300 KVC file-writing smoke.

Validate the M10.7d3c2 post-restore KVC file smoke summary:

```sh
python3 ds4-parity/compare_post_restore_kvc_smoke.py
python3 ds4-parity/compare_post_restore_kvc_smoke.py --negative-test
```

Refresh the B300 summary from the raw graph payload bodies:

```sh
python3 ds4-parity/compare_post_restore_kvc_smoke.py --live \
  --workdir /workspace/ds4 \
  --output-dir /tmp/ds4-m107d3c2-kvc \
  --write-summary /tmp/ds4-m107d3c2-post-restore-kvc.json \
  --negative-test
```

The comparator checks the Rust KVC wrapper smoke against the M10.7d3c1
decision contract and M10.7d3b same-capture restored graph evidence, including
KVC file names, headers, payload byte counts, payload digests, rendered text
key bytes, skip decisions, restored frontier state, and graph counters.

Validate the M10.8a model-free MTP state-machine contract:

```sh
python3 ds4-parity/check_mtp_state_machine_contract.py
python3 ds4-parity/check_mtp_state_machine_contract.py --negative-test
```

Refresh the B300 MTP support-artifact availability check:

```sh
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; ls -l /workspace/ds4/ds4flash.gguf; \
  readlink -f /workspace/ds4/ds4flash.gguf; \
  test ! -e /workspace/ds4/missing-mtp.gguf; \
  candidates=$(find /workspace/ds4 -maxdepth 3 -type f \
  \( -iname "*mtp*.gguf" -o -iname "*draft*.gguf" \) -print | sort); \
  printf "mtp_candidates=%s\n" "$candidates"; test -z "$candidates"'
```

The checker pins the current-C MTP decision state machine against source
anchors for draft execution, exact N=2 verification, suffix verification,
frontier snapshot/restore/prefix1 commit, and sequential fallback. It also
ties the MTP support-artifact blocker to the existing M8.12b CLI runtime
baseline so later MTP-enabled B300 stages cannot pass as MTP-off silently.

Compare the M10.8b Rust MTP decision planner against the M10.8a contract:

```sh
python3 ds4-parity/compare_mtp_decision_plan.py
python3 ds4-parity/compare_mtp_decision_plan.py --negative-test
```

The comparator runs `ds4-mtp-decision-plan`, compares each model-free Rust
planner row to the M10.8a contract, and pins fail-closed behavior for missing
support, disabled guards, verifier failures, restore/replay paths, and the
sequential safety fallback before any GPU MTP kernels are ported.

Compare the M10.8c Rust MTP draft orchestration plan against current-C draft
anchors:

```sh
python3 ds4-parity/compare_mtp_draft_plan.py
python3 ds4-parity/compare_mtp_draft_plan.py --negative-test
```

The comparator runs `ds4-mtp-draft-plan`, checks the wrapper and recursive
draft-HC paths, pins the draft command sequence, HC input/output roles,
top-id/logits readback roles, `mtp_n_raw` transition, and failure restoration
behavior, and records the live B300 MTP draft smoke as blocked until an MTP
support GGUF is present.

Compare the M10.8d Rust MTP exact-N=2 verifier orchestration plan against
current-C verifier anchors:

```sh
python3 ds4-parity/compare_mtp_decode2_plan.py
python3 ds4-parity/compare_mtp_decode2_plan.py --negative-test
```

The comparator runs `ds4-mtp-decode2-plan`, checks exact target token order,
the `metal_graph_verify_decode2_exact` command sequence, top0/logits0/logits1
readback roles, full-accept versus prefix1 logits source, prefix1 frontier
commit, failure restore behavior, and records the live B300 exact-N=2 smoke as
blocked until an MTP support GGUF is present.

Compare the M10.8e Rust MTP suffix verifier orchestration plan against
current-C suffix verifier anchors:

```sh
python3 ds4-parity/compare_mtp_suffix_plan.py
python3 ds4-parity/compare_mtp_suffix_plan.py --negative-test
```

The comparator runs `ds4-mtp-suffix-plan`, checks row-top sequence semantics,
full-accept last-row logits, prefix1 captured-frontier commit, restore/replay
fallbacks, exact replay debug behavior, suffix verifier failure
restore-or-error handling, and records the live B300 suffix smoke as blocked
until an MTP support GGUF is present.

Compare the M10.8f Rust MTP frontier mutation plan against current-C frontier
anchors:

```sh
python3 ds4-parity/compare_mtp_frontier_plan.py
python3 ds4-parity/compare_mtp_frontier_plan.py --negative-test
```

The comparator runs `ds4-mtp-frontier-plan`, checks snapshot and restore
counter handling, `mtp_n_raw` save/restore, ratio-4 index frontier copies,
prefix1 commit counter rewinds, invisible speculative-row policy, and ties the
plan back to M10.7d3 restored-frontier evidence.

Check the M10.8g1 current-C MTP stream parity contract before composing the
Rust stream outcome planner:

```sh
python3 ds4-parity/check_mtp_stream_parity_contract.py
python3 ds4-parity/check_mtp_stream_parity_contract.py --negative-test
```

The checker pins the end-to-end speculative stream outcomes from
`ds4_session_eval_speculative_argmax`: disabled or missing MTP, first-draft
miss, exact N=2 full/prefix/failure, suffix full/prefix/replay/failure,
sequential fallback, frontier restore/commit, `mtp_n_raw` keep policy, visible
cache/KVC state, and the explicit B300 missing-support blocker.

Compare the M10.8g2 Rust MTP stream outcome planner against the M10.8g1
current-C stream contract:

```sh
python3 ds4-parity/compare_mtp_stream_plan.py
python3 ds4-parity/compare_mtp_stream_plan.py --negative-test
```

The comparator runs `ds4-mtp-stream-plan`, checks final accepted stream deltas,
checkpoint deltas, logits ownership, frontier operations, `mtp_n_raw` keep
policy, cache/KVC visibility, fallback/error state, and verifies every selected
draft, verifier, suffix, and frontier sub-plan ID is present in the Rust plan
sources.

Compare the M10.8g3a Rust MTP runtime guard plan against the M10.8g2
unavailable stream outcomes and runtime source anchors:

```sh
python3 ds4-parity/compare_mtp_runtime_guard.py
python3 ds4-parity/compare_mtp_runtime_guard.py --negative-test
```

The comparator runs `ds4-mtp-runtime-guard-plan`, checks disabled,
first-draft-miss, and missing-support stream semantics against the M10.8g2
planner, and verifies the `EngineOptions`, `ds4-gguf` CLI parser,
one-shot/interactive/server runtime mappings, argmax/session non-MTP surfaces,
current-C speculative dispatch guards, and B300 missing-support artifact
anchors remain present.

Check the M10.8g3b runtime target-stream no-drift contract:

```sh
python3 ds4-parity/compare_mtp_runtime_no_drift.py
python3 ds4-parity/compare_mtp_runtime_no_drift.py --negative-test
```

The comparator ties the M10.8g3a disabled runtime guard rows to the committed
M8.12a current-C one-shot target-stream oracle and the M9.8f5 B300 Rust runtime
server replay summary. It verifies the one-shot target-stream cases remain
MTP-off, the Rust server runtime replay still matches the M0.5 current-C
content and cache/KVC accounting, and the B300 rerun hooks for Rust one-shot
and server runtime replay remain documented.

Check the M10.8g3c B300 missing-support runtime smoke:

```sh
python3 ds4-parity/compare_mtp_runtime_missing_support.py
python3 ds4-parity/compare_mtp_runtime_missing_support.py --negative-test
```

Refresh the live B300 summary:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4 && \
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; \
  export PATH=/tmp/cargo/bin:$PATH CARGO_HOME=/tmp/cargo RUSTUP_HOME=/tmp/rustup; \
  CUDA_ARCH=native cargo build -p ds4-engine --bin ds4-cli-one-shot-rs && \
  CUDA_ARCH=native python3 ds4-parity/compare_mtp_runtime_missing_support.py \
  --live --workdir /workspace/ds4 \
  --candidate-binary target/debug/ds4-cli-one-shot-rs \
  --write-summary /tmp/ds4-m108g3c-missing-support-runtime.json \
  --negative-test' && \
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp \
  ds4-rust-port-b300:/tmp/ds4-m108g3c-missing-support-runtime.json \
  ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json
```

The comparator pins the Rust runtime missing-MTP path to the M10.8g3a
missing-support guard row and the M10.8g1 stream blocker. It checks the B300
support-artifact search, exit code, empty stdout, current-C matching stderr,
blocked-before-stream visibility, zero checkpoint mutation, and no cache/KVC
visibility.

Check the M10.8g4a B300 support-artifact branch decision:

```sh
python3 ds4-parity/compare_mtp_support_branch.py
python3 ds4-parity/compare_mtp_support_branch.py --negative-test
```

Refresh the live B300 branch decision:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4 && \
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; \
  python3 ds4-parity/compare_mtp_support_branch.py \
  --live --workdir /workspace/ds4 \
  --write-summary /tmp/ds4-m108g4a-support-branch-decision.json \
  --negative-test' && \
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp \
  ds4-rust-port-b300:/tmp/ds4-m108g4a-support-branch-decision.json \
  ds4-parity/baselines/graph/m10.8g4a/support-branch-decision.json
```

The branch decision links the M10.8g1 stream blocker and M10.8g3c Rust runtime
blocker to the current B300 support-artifact search. With an empty candidate
list it selects `support_absent_blocker_closure` for M10.8g4b and forbids
reporting the result as either MTP-off success or MTP-enabled parity.

Check the M10.8g4b B300 MTP end-to-end closure:

```sh
python3 ds4-parity/compare_mtp_end_to_end_closure.py
python3 ds4-parity/compare_mtp_end_to_end_closure.py --negative-test
```

Refresh the live B300 end-to-end closure:

```sh
git archive HEAD | kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig \
  --context hou2-prod1 -n default exec -i ds4-rust-port-b300 -- \
  tar -xf - -C /workspace/ds4 && \
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default exec ds4-rust-port-b300 -- sh -lc \
  'set -e; cd /workspace/ds4; \
  python3 ds4-parity/compare_mtp_support_branch.py \
  --live --workdir /workspace/ds4 \
  --write-summary /tmp/ds4-m108g4a-support-branch-decision.json \
  --negative-test && \
  python3 ds4-parity/compare_mtp_end_to_end_closure.py \
  --branch-decision /tmp/ds4-m108g4a-support-branch-decision.json \
  --write-summary /tmp/ds4-m108g4b-end-to-end-closure.json \
  --negative-test' && \
kubectl --kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1 \
  -n default cp \
  ds4-rust-port-b300:/tmp/ds4-m108g4b-end-to-end-closure.json \
  ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json
```

The closure consumes the M10.8g4a branch decision, M10.8g1 stream blocker, and
M10.8g3c runtime blocker. With the support-absent branch selected, the
support-present comparator remains `not_run` due to `support_artifact_absent`,
the result is the explicit `blocked_missing_mtp_model` blocker, and
MTP-enabled current-C versus Rust parity is not claimed.

## Sampling And Logprob Parity

Run the local Milestone 6 report:

```sh
python3 ds4-parity/run_sampling_parity_report.py
```

The report runs the M6.2 fixed-logits C oracle checker, M6.3 Rust sampler
comparator, M6.4 committed session-logits fixture checker, M6.5 Rust
model-logits comparator, M6.6a decode-policy oracle checker, and M6.6b Rust
decode-policy comparator. The B300 model-backed M6.4 recapture is skipped by
default with the exact manifest refresh command.

## KV And Snapshot Parity

Run the local Milestone 7 report:

```sh
python3 ds4-parity/run_kv_parity_report.py
```

The report builds the local no-model C session-payload helper, runs the M7.2
through M7.8 KV/snapshot comparators, emits text or JSON reports, and skips the
model-backed B300 M7.8 restore recapture only with the exact manifest refresh
commands.

Validate the M7.2 no-model current-C KV header and policy oracle:

```sh
python3 ds4-parity/check_kv_policy_dump.py --negative-test
```

The fixture is generated by `./ds4-kv-policy-dump` and covers KVC header bytes,
decoded fields, reason and key-kind mapping, SHA/path helpers, size budgeting,
store-boundary selection, continued-store targets, eviction scoring, byte-prefix
matching, text-prefix lookup, and the committed M0.5 parsed KVC header rows.

Compare the M7.3 Rust no-model KVC header and policy port against that oracle:

```sh
python3 ds4-parity/compare_kv_policy.py --negative-test
```

The comparator runs `ds4-kv-policy-dump-rs`, checks byte-identical header
encoding/decoding, reason and extension flag behavior, SHA/path helpers, policy
decisions, eviction scores, and M0.5 row references, and verifies that targeted
negative mutations fail.

Compare the M7.4a generic KVC full-file Rust writer/reader against the
current-C oracle:

```sh
python3 ds4-parity/compare_kvc_file.py --negative-test
```

The fixture is generated by `./ds4-kvc-file-dump` and covers complete
fixed-header/text/opaque-payload/opaque-trailer bytes, reader metadata,
file-size budget decisions, and malformed header/text/payload/trailer boundary
cases without loading or restoring a model session.

Compare the M7.4b server-owned KVC trailer payload port:

```sh
python3 ds4-parity/compare_kv_trailer.py --negative-test
```

The fixture is generated by `./ds4-kv-trailer-dump` and covers `KTM` tool-map
trailer bytes, text filtering, duplicate-block suppression, multiple IDs for
one DSML block, wanted-ID load filtering, visible-transcript extension flags
without payload bytes, and malformed trailer boundaries.

Validate the M7.5 DSV4 session payload shape oracle:

```sh
python3 ds4-parity/check_session_payload_shape.py --negative-test
```

The structural fixture is generated by `./ds4-session-payload-dump` and covers
the 13-u32 DSV4 header, fixed DS4 layout fields, body section order and size
accounting, and current C load rejection categories. The checker also ties in
the M0.5 B300 payload-size/hash records and exact B300 recapture commands while
keeping raw KV payloads hash-only.

Compare the M7.6 Rust DSV4 session payload reader against the M7.5 oracle:

```sh
python3 ds4-parity/compare_session_payload.py --negative-test
```

The comparator runs `ds4-session-payload-dump-rs`, checks that Rust decodes the
same header fields, fixed layout, size accounting, trailing/truncated body
boundaries, and structural rejection categories as C, and treats the M0.5
payload records as fixture preconditions rather than raw body inputs.

Compare the M7.7 Rust KV replay decisions against committed current-C replay
artifacts:

```sh
python3 ds4-parity/compare_kv_replay.py --negative-test
```

The comparator derives its C oracle from the M0.4 server trace and M0.5 disk-KV
trace artifacts, checks M5 prompt-rendering fixture hashes as preconditions,
runs `ds4-kv-replay-dump-rs`, and compares cache source, cached/write token
counts, disk KV reason/key fields, rendered text hashes, token-prefix records,
DSML tool-call records, and effective prompt suffix bytes.

Validate the M7.8 B300 current-C restore oracle:

```sh
python3 ds4-parity/check_restore_dump.py \
  ds4-parity/baselines/kv/m7.8/current-c.json \
  --manifest ds4-parity/baselines/kv/m7.8/manifest.json \
  --negative-test
```

The artifact is captured from the recorded B300 model and covers disk DSV4
payload restore plus in-memory `ds4_session_snapshot` restore for seed and
continuation prompts. Raw payload/snapshot bodies are larger than 1 MiB and are
not committed; the JSON records hashes, header prefixes, selected tokens, top-20
logprob ordering, score deltas, model/backend identity, fixture hashes, and the
exact B300 refresh commands.

## Server Runtime Quality

Run the Rust server-runtime tool-call quality hook on a B300 model snapshot:

```sh
python3 ds4-parity/run_tool_call_quality.py \
  --server-bin target/debug/ds4-server-runtime-rs \
  --model /workspace/ds4/ds4flash.gguf \
  --backend cuda \
  --out-dir /tmp/ds4-m96d-tool-call-quality \
  --ready-timeout 360
```

The runner mirrors the C `./ds4_test --tool-call-quality` surface at the HTTP
runtime boundary. It launches fast and `--quality` server-runtime cases, sends a
compact OpenAI tool-call request with `temperature=0`, `seed=123`,
`max_tokens=256`, and `stream=false`; `top_k=0`, `top_p=1.0`, and `min_p=0.05`
come from the shared C/Rust OpenAI defaults. The classifier requires a
`list_files` tool call with arguments `{"path":"."}` and finish reason
`tool_calls`. It writes per-case request, response, headers, trace, stdout, and
stderr files plus `summary.json`/`summary.txt` so raw outputs are retained for
failures and drift investigation.

Run the model-free classifier checks locally:

```sh
python3 ds4-parity/run_tool_call_quality.py --self-test
```

## CLI Surface Parity

Validate the M8.2 current-C no-model CLI parse/error oracle:

```sh
python3 ds4-parity/check_cli_parse_dump.py \
  ds4-parity/baselines/cli/m8.2/current-c.json \
  --manifest ds4-parity/baselines/cli/m8.2/manifest.json \
  --negative-test
```

The fixture covers help output, missing option values, unknown options, invalid
numeric and backend values, duplicate prompt sources, removed/deprecated flags,
`--dump-tokens` without a prompt, imatrix option coupling, prompt-file open
errors, and `--perplexity-file` prompt-source rejection. These cases must remain
local and model-free.

Compare the M8.3 Rust parser-only CLI surface against that oracle:

```sh
python3 ds4-parity/compare_cli_parse.py --negative-test
```

The comparator builds `ds4-cli-parse-rs`, runs it against the M8.2 argument
matrix, and compares exit status, stdout/stderr emptiness, stable help anchors,
stderr category anchors, and no-model-load markers.

Validate the M8.4 current-C CLI token/prompt diagnostic oracle:

```sh
python3 ds4-parity/check_cli_token_dump.py \
  ds4-parity/baselines/cli/m8.4/current-c.json \
  --manifest ds4-parity/baselines/cli/m8.4/manifest.json \
  --negative-test
```

The fixture runs `./ds4 --dump-tokens` on B300 with `-p`, `--prompt-file`, a
rendered-chat prompt, `--system`, empty `--system`, `--think`, `--think-max` at
both context thresholds, and `--nothink`. `--dump-tokens` exits before the
normal prompt builder and thinking warning path, so the system/thinking cases
are expected to produce byte-identical token dumps to the base prompt with empty
stderr.

Compare the M8.5 Rust CLI token-dump path against that oracle:

```sh
python3 ds4-parity/compare_cli_token_dump.py --negative-test
```

The comparator builds `ds4-cli-token-dump-rs`, substitutes the small M5.3
tokenizer GGUF for the B300 model path, and requires exact C/Rust stdout bytes,
stderr bytes, exit status, and parsed token IDs for every M8.4 case. The Rust
diagnostic writer intentionally prints raw tokenizer table text, matching
current C `dump_tokens_fp`, rather than decoded token bytes.

## CUDA Rust Ownership

Validate the M14.0 complete CUDA ownership inventory and `cuda-oxide` adoption
contract:

```sh
python3 ds4-parity/check_cuda_rust_ownership_inventory.py --negative-test
```

The inventory source-hashes the current CUDA ABI, Rust FFI/build boundary, and
CUDA implementation; it assigns every exported CUDA function and every unique
CUDA kernel symbol to an M14 Rust ownership stage. It records the inspected
`cuda-oxide` revision and capability evidence, while explicitly keeping the
current CUDA backend as the oracle and blocking default-route promotion or
`ds4_cuda.cu` removal until M14.6 closure.

Validate the M14.1a opt-in `cuda-oxide` host-substrate B300 smoke:

```sh
python3 ds4-parity/check_cuda_oxide_substrate_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-substrate-smoke`: Rust-owned CUDA context/stream setup, device
transfer/readback, zeroed allocation/readback, and managed-buffer lifetime. It
does not claim DS4 compute-kernel ownership, runtime route activation, or CUDA
source removal.

Validate the M14.1b1 bounded model-residency handles B300 smoke:

```sh
python3 ds4-parity/check_model_residency_handles_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-residency-smoke` against a bounded prefix read from the pinned
GGUF: managed advice/prefetch, mapped-host device-pointer creation, and
registered caller-owned host lifetime. It does not claim complete model-map
or range-cache ownership, DS4 kernels, or runtime route activation.

Validate the M14.1b2a Rust-owned mmap/device-range copy B300 smoke:

```sh
python3 ds4-parity/check_model_range_copy_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-range-copy-smoke` against the pinned GGUF: Rust-owned mmap
lifetime, bounds-checked range selection, CUDA device-buffer copy/readback,
and exact cache reuse. It does not claim registered/HMM/direct-I/O strategy
selection, DS4 kernels, or runtime route activation.

Validate the M14.1b2b1 file-staged model-range strategy B300 smoke:

```sh
python3 ds4-parity/check_model_range_strategy_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-range-strategy-smoke` against the pinned GGUF: the Rust cache
selects mmap-sourced and file-staged device-copy strategies independently,
then requires exact readback equality and per-strategy cache reuse. It does
not claim `O_DIRECT`, registered mapped-host ranges, pageable HMM, DS4
kernels, or runtime route activation.

Validate the M14.1b2b2 registered range selection and fallback B300 smoke:

```sh
python3 ds4-parity/check_model_registered_range_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-registered-range-smoke` against the pinned GGUF: an unaligned
request is expanded to a page-aligned read-only registration attempt. On the
captured B300 runtime CUDA returns error `801` (`operation not supported`), so
the strategy must use and reuse the exact mmap-sourced device-copy fallback.
It does not claim successful zero-copy registration on B300, pageable HMM,
`O_DIRECT`, DS4 kernels, or runtime route activation.

Validate the M14.1b2b3a pageable HMM range strategy B300 smoke:

```sh
python3 ds4-parity/check_model_pageable_hmm_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-pageable-hmm-smoke` against the pinned GGUF: an unaligned
request is expanded to a page-aligned pageable-memory window, advised and
prefetched through a borrowed guard, then read back through the direct HMM
pointer. The proof path synchronizes prefetch lifetime and does not claim
`O_DIRECT`, asynchronous production prefetch policy, DS4 kernels, or runtime
route activation.

Validate the M14.1b2b3b1 direct-I/O pinned read selection B300 smoke:

```sh
python3 ds4-parity/check_model_direct_io_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-direct-io-smoke` against the pinned GGUF: an ordinary
unaligned request is staged through an aligned `O_DIRECT` pinned window and
the unaligned file tail takes the buffered fallback; both CUDA readbacks must
match. It does not claim asynchronous staging-ring/event scheduling,
cache-budget policy, persistent disable-after-error state, kernels, or route
activation.

Validate the M14.1b2b3b2 asynchronous staging ring and budget policy B300 smoke:

```sh
python3 ds4-parity/check_model_async_staging_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-async-staging-smoke` against the pinned GGUF: seven direct
chunks rotate through four pinned slots with event-guarded reuse, two
admitted ranges share a bounded device arena, and the next range takes the
budget fallback while admitted bytes read back exactly. Feature tests cover
the direct-I/O disable errno policy; the live smoke does not claim an induced
I/O error, source-page discard/progress behavior, kernels, or route
activation.

Validate the M14.1b2c model-map cache closure B300 smoke:

```sh
python3 ds4-parity/check_model_map_closure_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-model-map-closure-smoke` against the pinned GGUF. It proves
contained-range reuse with exact CUDA readback, Linux source file/mapping
discard advisory calls and keep-pages suppression, explicit non-TTY progress
emission and disabled suppression, and fresh-cache reset state. It does not
claim physical page eviction, default runtime environment/terminal wiring,
DS4 kernel consumption, or route activation.

Validate the M14.1b3a managed-KV and memory-report policy B300 smoke:

```sh
python3 ds4-parity/check_allocation_policy_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-allocation-policy-smoke` using the cuda-oxide context memory query.
It proves managed allocation, current-C managed-KV threshold and reserve
choices, and current-C-shaped memory-report formatting. It does not claim
Q8 converted caches, quality-mode BLAS selection, DS4 kernels, or route
activation.

Validate the M8.6 current-C CLI logprob/perplexity diagnostic oracle:

```sh
python3 ds4-parity/check_cli_diagnostics_dump.py \
  ds4-parity/baselines/cli/m8.6/current-c.json \
  --manifest ds4-parity/baselines/cli/m8.6/manifest.json \
  --negative-test
```

The fixture covers `--dump-logprobs` with inline and prompt-file prompts,
`--logprobs-top-k`, a bad logprob output path, and `--perplexity-file` over a
fixed raw text file. The artifact stores raw stdout/stderr bytes, emitted
logprob JSON bytes plus parsed summaries, perplexity scalar fields, and the M6
score tolerance policy for future Rust comparisons.
