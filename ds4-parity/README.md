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

Validate the M14.1b3b Q8 admission and quality-mode policy B300 smoke:

```sh
python3 ds4-parity/check_q8_quality_policy_smoke.py --negative-test
```

The fixture records a feature-enabled B300 execution of
`ds4-cuda-q8-quality-policy-smoke` using cuda-oxide cuBLAS math-mode control.
It proves current-C Q8/F16 and Q8/F32 admission/failure-disable decisions plus
live TF32/default-math selection. It does not claim converted Q8 buffers or
their failure-time release, dequant kernels, DS4 compute kernels, or route
activation.

Validate the M14.1b4 Rust fill-kernel and command-lifetime B300 smoke:

```sh
python3 ds4-parity/check_fill_command_lifetime_smoke.py --negative-test
```

The fixture records an executable-local `#[cuda_module]` fill kernel run on
B300 after cuda-oxide stopped forcing the unsupported `sm_103` PTX target for
a basic portable kernel. It proves prefix fill, negative-infinity fill,
zero-count/bounds behavior, current-C's no-op begin command, and context-wide
flush/end/synchronize wrappers. It does not claim library artifact retention,
dequant or graph compute kernels, runtime graph integration, or default-route
ownership.

Validate the M14.1c Rust CUDA substrate route-closure gate:

```sh
python3 ds4-parity/check_substrate_route_closure.py --negative-test
```

The closure fixture records that M14.1 exposes only the opt-in Rust CUDA
resource substrate and `fill_f32` command surface to following kernel work.
It corrects the inventory so converted Q8 cache ownership and dequant kernels
remain assigned to M14.3, and rejects default-route promotion or C CUDA
removal.

Validate the M14.2a Rust CUDA add/repeat elementwise kernel smoke:

```sh
python3 ds4-parity/check_elementwise_kernel_smoke.py --negative-test
```

The fixture records executable-local `add_kernel` and `repeat_hc_kernel`
execution on B300. It proves f32 add output, repeated-HC-row output, and
bounded argument rejection while leaving embedding, indexer/top-k, SwiGLU,
directional steering, runtime graph integration, and default-route ownership
unclaimed.

Validate the M14.2b1 Rust CUDA directional-steering kernel smoke:

```sh
python3 ds4-parity/check_directional_steering_kernel_smoke.py --negative-test
```

The fixture records executable-local `directional_steering_project_kernel`
execution on B300 through static shared memory and block synchronization. It
also records the remaining SwiGLU blocker: `f32::exp()` selects cuda-oxide's
libdevice/NVVM path, whose emitted opaque-pointer IR is rejected by CUDA 13.2
`libnvvm`, so SwiGLU ownership is not claimed.

Validate the M14.2b2 Rust CUDA SwiGLU/libdevice kernel smoke:

```sh
python3 ds4-parity/check_swiglu_kernel_smoke.py --negative-test
```

The fixture records executable-local `swiglu_kernel` execution on B300 after
cuda-oxide revision `d4791b7002152af3b7f6b15a48d7f5acd7a63011` repaired
the libdevice route: portable `sm_80` PTX with `__nv_expf` is linked into a
context-targeted `sm_103` cubin. It proves finite and NaN clamp behavior,
unclamped output, SiLU, weight, and shape rejection without claiming
embedding, indexer/top-k, runtime route, or C CUDA removal ownership.

Validate the M14.2c Rust CUDA FP16 embedding kernel-pair smoke:

```sh
python3 ds4-parity/check_embedding_kernel_smoke.py --negative-test
```

The fixture records executable-local `embed_token_hc_kernel` and
`embed_tokens_hc_kernel` execution on B300 through primitive Rust `f16`
loads. It proves hidden-copy replication and batched invalid-token fallback
while leaving model-range cache consumption, indexer/top-k, runtime route,
and C CUDA removal ownership pending.

Validate the M14.2d1 Rust CUDA scalar indexer selection kernel smoke:

```sh
python3 ds4-parity/check_indexer_scalar_kernel_smoke.py --negative-test
```

The fixture records executable-local `indexer_scores_kernel`,
`indexer_topk_kernel`, and `topk_mask_kernel` execution on B300. It proves
scalar fallback scoring, causal masking, stable scalar top-k tie ordering,
and top-k mask output while leaving direct/WMMA score dispatch, specialized
top-k dispatch, runtime route, and C CUDA removal pending.

Validate the M14.2d2a Rust CUDA direct-one indexer score kernel smoke:

```sh
python3 ds4-parity/check_indexer_direct_kernel_smoke.py --negative-test
```

The fixture records executable-local `indexer_score_one_direct_kernel`
execution on B300 through the same four-warp shuffle-down reduction shape as
current C. It proves direct scoring, causal masking, and NaN/negative clamp
behavior while leaving tensor-core scoring, specialized top-k dispatch,
runtime route, and C CUDA removal pending.

Validate the M14.2d2b1 Rust CUDA base tensor-core indexer score kernel smoke:

```sh
python3 ds4-parity/check_indexer_wmma_kernel_smoke.py --negative-test
```

The fixture records executable-local `indexer_scores_wmma_kernel` execution
on B300 through two cuda-oxide `m16n8k16` operations covering the current-C
`16 x 16` output tile. It proves base WMMA scoring, weighted output,
NaN/negative suppression, and causal masking while leaving widened WMMA
dispatch, specialized top-k dispatch, runtime route, and C CUDA removal
pending.

Validate the M14.2d2b2a Rust CUDA WMMA32 tensor-core indexer score kernel smoke:

```sh
python3 ds4-parity/check_indexer_wmma32_kernel_smoke.py --negative-test
```

The fixture records executable-local `indexer_scores_wmma32_kernel`
execution on B300 through two warps covering the current-C `16 x 32`
output tile. It proves WMMA32 scoring, weighted output, NaN/negative
suppression, and causal masking while leaving WMMA64/WMMA128 dispatch,
specialized top-k dispatch, runtime route, and C CUDA removal pending.

Validate the M14.2d2b2b Rust CUDA WMMA64 tensor-core indexer score kernel smoke:

```sh
python3 ds4-parity/check_indexer_wmma64_kernel_smoke.py --negative-test
```

The fixture records executable-local `indexer_scores_wmma64_kernel`
execution on B300 through four warps covering the current-C `16 x 64`
output tile. It proves WMMA64 scoring, weighted output, NaN/negative
suppression, and causal masking while leaving WMMA128 dispatch priority,
specialized top-k dispatch, runtime route, and C CUDA removal pending.

Validate the M14.2d2b2c Rust CUDA WMMA128 and score-dispatch smoke:

```sh
python3 ds4-parity/check_indexer_wmma128_dispatch_smoke.py --negative-test
```

The fixture records executable-local `indexer_scores_wmma128_kernel`
execution on B300 through eight warps covering the current-C `16 x 128`
output tile, plus the Rust selector for current-C's validated-input score
kernel priority order. It leaves specialized top-k dispatch, runtime route,
and C CUDA removal pending.

Validate the M14.2d2c1 Rust CUDA 1024-element bitonic top-k kernel smoke:

```sh
python3 ds4-parity/check_indexer_topk1024_kernel_smoke.py --negative-test
```

The fixture records executable-local `indexer_topk_1024_kernel` execution on
B300 with current-C's descending score and lower-index tie ordering. It
proves the `top_k == 512 && n_comp <= 1024` fast path, including partial
component padding, while leaving larger top-k branches, indexed ascending
sort, runtime route, and C CUDA removal pending.

Validate the M14.2d2c2 Rust CUDA power-of-two top-k kernel smoke:

```sh
python3 ds4-parity/check_indexer_topk_pow2_kernel_smoke.py --negative-test
```

The fixture records executable-local 2048/4096 `u32`-index and 8192
`u16`-index bitonic kernels on B300, preserving descending score and
lower-index tie order. It proves the larger shared-memory kernel behavior
while leaving CUB selection, chunked merging, indexed ascending sort,
runtime route, and C CUDA removal pending.

Validate the M14.2d2c3 Rust CUDA packed-key top-k equivalent smoke:

```sh
python3 ds4-parity/check_indexer_topk_packed_kernel_smoke.py --negative-test
```

The fixture records executable-local packed-key dynamic-shared-memory top-k
execution on B300 after the pinned cuda-oxide host API repair needed to opt
into a 65,536-byte launch. It proves current-C ordered-float and lower-index
key semantics while leaving CUB library ownership, branch selection,
chunk/tree merging, indexed ascending sort, runtime route, and C CUDA removal
pending.

Validate the M14.2d2c4 Rust CUDA chunk/tree top-k kernel smoke:

```sh
python3 ds4-parity/check_indexer_topk_tree_kernel_smoke.py --negative-test
```

The fixture records executable-local 4096-element chunk, tree-merge, and
final-merge kernels on B300 through a contiguous scratch-level plan. It proves
multi-token stride isolation and partial final-chunk behavior while leaving
indexed ascending sort, specialized dispatch policy, runtime route, and
C CUDA removal pending.

Validate the M14.2d2c5 Rust CUDA indexed-sort and top-k dispatch smoke:

```sh
python3 ds4-parity/check_indexer_topk_dispatch_smoke.py --negative-test
```

The fixture records executable-local `indexed_topk_sort_512_asc_kernel`
execution on B300 plus validated-input selectors for the specialized top-k
launch order. It maps the capability-gated CUB positions to the validated
packed-key equivalent without claiming CUB implementation, runtime route, or
C CUDA removal.

Validate the M14.2e Rust CUDA kernel-family closure gate:

```sh
python3 ds4-parity/check_m14_2_kernel_closure.py --negative-test
```

The closure fixture aggregates the M14.2 B300 kernel proofs, reassigns the
routed-MoE-only `zero_kernel` to M14.5, and records the packed-key top-k
semantic equivalent without claiming CUB implementation. It rejects
default-route promotion and C CUDA removal.

Validate the M14.3a Rust CUDA plain and weighted RMS normalization smoke:

```sh
python3 ds4-parity/check_rms_norm_kernel_smoke.py --negative-test
```

The fixture records executable-local `rms_norm_plain_kernel` and
`rms_norm_weight_kernel` execution on B300 using the libdevice-linked
reciprocal RMS scale path. It proves multi-row plain and weighted output,
single-row behavior, and bounds rejection while leaving fused QKV/head norm,
dense projection, Q8 conversion or matmul, runtime graph integration, and
default-route ownership unclaimed.

Validate the M14.3b1 Rust CUDA fused QKV and basic head RMS normalization smoke:

```sh
python3 ds4-parity/check_fused_rms_norm_kernel_smoke.py --negative-test
```

The fixture records executable-local `dsv4_qkv_rms_norm_rows_kernel` and
`head_rms_norm_kernel` execution on B300 using the libdevice-linked
reciprocal RMS scale path. It proves the fused rows-by-two grid with unequal
Q and KV widths and in-place per-head normalization while leaving the
RMS-plus-RoPE-tail kernel, fused-QKV fallback policy, projection, Q8, route,
and removal ownership unclaimed.

Validate the M14.3b2 Rust CUDA head RMS normalization and RoPE-tail smoke:

```sh
python3 ds4-parity/check_head_rms_rope_tail_kernel_smoke.py --negative-test
```

The fixture records executable-local `head_rms_norm_rope_tail_kernel`
execution on B300 through libdevice-linked RMS, YARN, and rotary math. It
proves interpolated rotation, YARN forward rotation, inverse rotation, and
shape rejection while leaving standalone RoPE, projection, Q8, route, and
removal ownership unclaimed.

Validate the M14.3c1 Rust CUDA base F16 and F32 projection smoke:

```sh
python3 ds4-parity/check_dense_projection_kernel_smoke.py --negative-test
```

The fixture records executable-local `matmul_f16_kernel` and
`matmul_f32_kernel` execution on B300 with shared reductions and primitive
F16 weight loads. It proves base multi-token output layout while leaving
serial/ordered/paired F16 variants, cuBLAS dispatch, Q8, route, and removal
ownership unclaimed.

Validate the M14.3c2 Rust CUDA ordered and serial F16 projection smoke:

```sh
python3 ds4-parity/check_ordered_projection_kernel_smoke.py --negative-test
```

The fixture records executable-local `matmul_f16_serial_kernel`,
`matmul_f16_ordered_chunks_kernel`, and
`matmul_f16_pair_ordered_chunks_kernel` execution on B300. It proves
multi-token serial output, ordered chunk reduction, and unequal-width paired
output while leaving cuBLAS dispatch, activation conversion, Q8, route, and
removal ownership unclaimed.

Validate the M14.3c3 Rust CUDA BLAS projection and activation conversion smoke:

```sh
python3 ds4-parity/check_blas_projection_kernel_smoke.py --negative-test
```

The fixture records executable-local `f32_to_f16_kernel` execution and
`cuda-core` F16/F32 BLAS projection on B300. It proves current-C-compatible
dense projection dispatch while leaving Q8, route, and removal ownership
unclaimed.

Validate the M14.3d1 Rust CUDA Q8 conversion kernel smoke:

```sh
python3 ds4-parity/check_q8_conversion_kernel_smoke.py --negative-test
```

The fixture records packed Q8 F16/F32 dequantization and activation
quantization execution on B300. It proves nearest-even rounding and
partial-block padding while leaving Q8 matmul, dispatch, route, and removal
ownership unclaimed.

Validate the M14.3d2 Rust CUDA Q8 matmul kernel smoke:

```sh
python3 ds4-parity/check_q8_matmul_kernel_smoke.py --negative-test
```

The fixture records direct-quantizing, generic prequantized, single-token
warp8, and batched warp8 Q8 matmul execution on B300. It proves packed Q8
scalar integer-dot behavior while leaving DP4A acceleration, pair/HC
expansion, dispatch, route, and removal ownership unclaimed.

Validate the M14.3d3 Rust CUDA Q8 specialized matmul kernel smoke:

```sh
python3 ds4-parity/check_q8_specialized_matmul_kernel_smoke.py --negative-test
```

The fixture records paired unequal-width and HC-expansion Q8 matmul execution
on B300. It proves optional HC block addition while leaving DP4A
acceleration, dispatch, route, and removal ownership unclaimed.

Validate the M14.3d4 Rust CUDA Q8 DP4A and dispatch-policy smoke:

```sh
python3 ds4-parity/check_q8_dp4a_dispatch_smoke.py --negative-test
```

The fixture records signed packed-i8 DP4A execution and current-C-compatible
Q8 dispatch policy on B300. It proves emitted `dp4a.s32.s32` PTX plus the
scalar partial-block fallback while leaving runtime route activation and C
CUDA removal unclaimed.

Validate the M14.4a Rust CUDA standalone RoPE and FP8 KV quantization smoke:

```sh
python3 ds4-parity/check_rope_kv_quantization_kernel_smoke.py --negative-test
```

The fixture records executable-local `rope_tail_kernel` and
`fp8_kv_quantize_kernel` execution on B300 through the libdevice-linked
cuda-oxide path. It proves position-stride and YARN inverse tail rotation,
E4M3FN round-trip quantization across a partial 64-wide prefix chunk, and
unchanged RoPE-tail values while leaving KV storage, compressor, attention,
runtime route, and C CUDA removal unclaimed.

Validate the M14.4b Rust CUDA raw KV storage and indexer QAT smoke:

```sh
python3 ds4-parity/check_raw_kv_indexer_qat_kernel_smoke.py --negative-test
```

The fixture records executable-local `store_raw_kv_batch_kernel` and
`indexer_hadamard_fp4_kernel` execution on B300. It proves FP16-round-tripped
ring storage across distinct wrapped rows and the 128-wide Hadamard plus
E2M1FN activation-simulation round trip. It leaves same-launch overlapping
row ordering, composed FP8 store, compressor, attention, runtime route, and
C CUDA removal unclaimed.

Validate the M14.4c1 Rust CUDA composed KV and compressor-store smoke:

```sh
python3 ds4-parity/check_composed_kv_compressor_store_kernel_smoke.py --negative-test
```

The fixture records executable-local composed FP8 quantization plus raw-store
execution and `compressor_store_kernel`/`compressor_set_rows_kernel` execution
on B300. It proves ratio-4 state-row selection and both F32 and F16 APE reads.
It leaves compressor pooling/shift, wrapper orchestration, attention, runtime
route, and C CUDA removal unclaimed.

Validate the M14.4c2 Rust CUDA compressor pool and ratio-4 shift smoke:

```sh
python3 ds4-parity/check_compressor_pool_shift_kernel_smoke.py --negative-test
```

The fixture records executable-local `compressor_prefill_pool_kernel`,
`compressor_update_pool_kernel`, and `compressor_shift_ratio4_kernel`
execution on B300. It proves general-ratio, ratio-4, ratio-4 replay, and F16
APE pool behavior while leaving update/prefill wrapper orchestration,
attention, runtime route, and C CUDA removal unclaimed.

Validate the M14.4c3a Rust CUDA compressor update orchestration smoke:

```sh
python3 ds4-parity/check_compressor_update_orchestration_smoke.py --negative-test
```

The fixture records executable-local update orchestration through store,
pooling, weighted RMS normalization, YARN RoPE, and ratio-4 shift execution
on B300. It proves ratio-4 non-emission, ratio-4 emission with F16 APE and a
nonzero compressed row, and general-ratio emission while leaving prefill and
replay orchestration, attention, runtime route, and C CUDA removal unclaimed.

Validate the M14.4c3b Rust CUDA compressor prefill orchestration smoke:

```sh
python3 ds4-parity/check_compressor_prefill_orchestration_smoke.py --negative-test
```

The fixture records executable-local general prefill, ratio-4 prefill/replay,
and ratio-4 state-only orchestration on B300. It proves state initialization,
remainder placement, replay output ordering, weighted RMS/RoPE composition,
F16 APE input, and optional FP8 compressed output while leaving attention,
runtime route, and C CUDA removal unclaimed.

Validate the M14.4d1 Rust CUDA single-token mixed attention decode smoke:

```sh
python3 ds4-parity/check_attention_decode_single_mixed_smoke.py --negative-test
```

The fixture records executable-local `ds4_gpu_attention_decode_heads_tensor`
semantics on B300. It proves wrapped raw rows, compressed-row masking,
learned-sink softmax participation, and raw-only output while leaving
batched/window/heads8, prefill/indexed/output-Q8 attention, runtime route,
and C CUDA removal unclaimed.

Validate the M14.4d2 Rust CUDA generic batched mixed attention decode smoke:

```sh
python3 ds4-parity/check_attention_decode_batch_mixed_smoke.py --negative-test
```

The fixture records executable-local generic
`ds4_gpu_attention_decode_raw_batch_heads_tensor` and
`ds4_gpu_attention_decode_mixed_batch_heads_tensor` semantics on B300. It
proves causal raw-window selection, ring wrap, per-token compressed
visibility and masking, learned-sink softmax participation, and raw-only
batched output while leaving heads8-online dispatch,
prefill/indexed/output-Q8 attention, runtime route, and C CUDA removal
unclaimed.

Validate the M14.4d3 Rust CUDA heads8 online attention decode smoke:

```sh
python3 ds4-parity/check_attention_decode_heads8_online_smoke.py --negative-test
```

The fixture records executable-local `attention_decode_mixed_heads8_online_kernel`
semantics and the current-C decode dispatch predicates on B300. It proves
grouped and partial-head-group output, single-all and batched causal-window
behavior, ring wrap, compressed visibility, learned-sink softmax, and both
score-buffer-overflow and window-attention selection while leaving prefill,
indexed, output-Q8 attention, runtime route, and C CUDA removal unclaimed.

Validate the M14.4d4 Rust CUDA generic attention prefill smoke:

```sh
python3 ds4-parity/check_attention_prefill_generic_smoke.py --negative-test
```

The fixture records executable-local `attention_prefill_raw_kernel` and
`attention_prefill_mixed_kernel` semantics on B300. It proves generic raw,
static mixed, and masked mixed output with causal windows, compressed-row
visibility and masking, and learned-sink softmax while leaving static
heads8-online/CUBLAS prefill dispatch, indexed/output-Q8 attention, runtime
route, and C CUDA removal unclaimed.

Validate the M14.4d5 Rust CUDA optimized attention prefill smoke:

```sh
python3 ds4-parity/check_attention_prefill_optimized_smoke.py --negative-test
```

The fixture records executable-local `attention_static_mixed_heads8_online_kernel`
semantics and live `cuda-core` strided-batched SGEMM execution on B300. It
proves static heads8-online output, raw cuBLAS prefill output, masked mixed
cuBLAS prefill output, and current-C branch priority while leaving indexed
and output-Q8 attention, runtime route, and C CUDA removal unclaimed.

Validate the M14.4d6 Rust CUDA generic indexed mixed attention smoke:

```sh
python3 ds4-parity/check_attention_indexed_generic_smoke.py --negative-test
```

The fixture records executable-local `attention_indexed_mixed_kernel`
semantics on B300. It proves ordered and duplicate top-k row handling,
invalid/out-of-visible filtering, ratio-zero all-compressed visibility,
causal wrapped raw rows, and learned-sink softmax while leaving indexed
sort/heads8 dispatch, output-Q8 attention, runtime route, and C CUDA removal
unclaimed.

Validate the M14.4d7 Rust CUDA optimized indexed attention smoke:

```sh
python3 ds4-parity/check_attention_indexed_optimized_smoke.py --negative-test
```

The fixture records integration of the prior indexed ascending-sort policy
with `attention_indexed_mixed_heads8_online_kernel` and execution of
`attention_indexed_mixed_heads8_rb4_kernel` on B300. It proves sorted-online
and filtered/duplicate rb4 output plus current-C dispatch priority while
leaving output-Q8 attention, runtime route, and C CUDA removal unclaimed.

Validate the M14.4d8a Rust CUDA native output-Q8 attention smoke:

```sh
python3 ds4-parity/check_attention_output_q8_native_smoke.py --negative-test
```

The fixture records native Q8 execution for
`ds4_gpu_attention_output_low_q8_tensor` and
`ds4_gpu_attention_output_q8_batch_tensor` on B300. It proves low-only and
batched two-stage grouped-output projections with partial Q8 blocks while
leaving optional F16/cuBLAS A dispatch, runtime route, and C CUDA removal
unclaimed.

Validate the M14.4d8b Rust CUDA cuBLAS output-Q8 attention smoke:

```sh
python3 ds4-parity/check_attention_output_q8_cublas_smoke.py --negative-test
```

The fixture records the optional attention-output-A cuBLAS branch on B300:
F16-rounded grouped-head packing, live grouped projection through the
cuda-oxide safe SGEMM adapter, low-output unpacking, and the current-C branch
predicate. It records the `CUDA_R_16F` GemmEx API difference explicitly and
leaves runtime route activation and C CUDA removal unclaimed.

Validate the M14.5a Rust CUDA scalar router smoke:

```sh
python3 ds4-parity/check_router_scalar_smoke.py --negative-test
```

The fixture records current-C scalar router behavior on B300: probability
transformation, bias-ranked top-6 selection, hash routing with invalid-token
fallback, normalized selected weights, and both single-token and batched
layouts. It leaves parallel/warp routing, routed MoE, hyperconnection,
runtime route, and C CUDA removal unclaimed.

Validate the M14.5b Rust CUDA optimized router smoke:

```sh
python3 ds4-parity/check_router_optimized_smoke.py --negative-test
```

The fixture records current-C optimized router behavior on B300: parallel
shared-memory probability storage, default warp-shuffle top-k selection,
equal-score index ordering, partial four-row blocks, hash fallback, and the
warp/parallel/scalar disable-flag priority. It leaves routed MoE,
hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c1 Rust CUDA routed MoE F32-activation fallback smoke:

```sh
python3 ds4-parity/check_routed_moe_f32_smoke.py --negative-test
```

The fixture records current-C packed-weight routed MoE fallback behavior on
B300: table-indexed IQ2-XXS gate/up decode, Q2_K down decode, weighted
SwiGLU/clamp behavior, negative-expert fallback, expert summation, and both
single-token and batched layouts. It leaves Q8 activation/optimized dispatch,
Q4_K, hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2a Rust CUDA default single-token quantized routed MoE
smoke:

```sh
python3 ds4-parity/check_routed_moe_quantized_single_smoke.py --negative-test
```

The fixture records current-C default single-token IQ2/Q2 quantized routed
MoE behavior on B300: Q8_K input and intermediate activation fields,
LUT-equivalent IQ2-XXS/Q8_K gate/up decode, direct six-expert Q2_K/Q8_K down
output, auxiliary write mode, zero quantization, and negative-expert
fallback. It leaves batched sorted/tiled dispatch, Q4_K, hyperconnection,
runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2b1 Rust CUDA routed MoE sorted-pair metadata smoke:

```sh
python3 ds4-parity/check_routed_moe_sorted_pairs_smoke.py --negative-test
```

The fixture records current-C batched sorted-pair metadata behavior on B300:
device-atomic expert counts, prefix offsets/cursors, grouped scatter with
duplicate pair preservation, and negative-expert bucket-zero semantics. It
leaves sorted projection, expert-tile/atomic-down execution, Q4_K,
hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2b2 Rust CUDA sorted-P2 routed MoE smoke:

```sh
python3 ds4-parity/check_routed_moe_sorted_p2_smoke.py --negative-test
```

The fixture records current-C no-expert-tiles/default-P2 batched IQ2/Q2
quantized routed MoE behavior on B300: sorted metadata consumption, batched
Q8_K input quantization, pair-indexed gate/up and down projection, final
per-token summation, partial row/pair tiles, and negative-expert fallback. It
leaves expert-tile/atomic-down scheduling, Q4_K, hyperconnection, runtime
route, and C CUDA removal unclaimed.

Validate the M14.5c2c1 Rust CUDA routed MoE expert-tile metadata smoke:

```sh
python3 ds4-parity/check_routed_moe_expert_tiles_smoke.py --negative-test
```

The fixture records current-C batched expert-tile descriptor behavior on B300:
default eight-pair and alternate four-pair tile offsets, expert/start
descriptors, partial final tiles, and negative-expert bucket-zero counts. It
leaves tile-local projection, atomic-down/rowspan execution, Q4_K,
hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2c2 Rust CUDA routed MoE tile8 row32 smoke:

```sh
python3 ds4-parity/check_routed_moe_tile8_row32_smoke.py --negative-test
```

The fixture records current-C functional default tile8 row32 expert projection
behavior on B300: multi-tile and partial-tile IQ2-XXS/Q8_K gate/up output,
Q2_K/Q8_K non-atomic down output, and negative-expert bucketing through tile
metadata. It leaves shared-cache specialization, tile4,
atomic-down/tile16/rowspan execution, Q4_K, hyperconnection, runtime route,
and C CUDA removal unclaimed.

Validate the M14.5c2c3 Rust CUDA routed MoE tile4 row32 smoke:

```sh
python3 ds4-parity/check_routed_moe_tile4_row32_smoke.py --negative-test
```

The fixture records current-C functional `DS4_CUDA_MOE_TILE4` row32 expert
projection behavior on B300: three-tile and partial-tile IQ2-XXS/Q8_K
gate/up output, Q2_K/Q8_K non-atomic down output, and negative-expert
bucketing through tile metadata. It leaves shared-cache specialization,
atomic-down/tile16/rowspan execution, Q4_K, hyperconnection, runtime route,
and C CUDA removal unclaimed.

Validate the M14.5c2c4 Rust CUDA routed MoE atomic-down smoke:

```sh
python3 ds4-parity/check_routed_moe_atomic_down_smoke.py --negative-test
```

The fixture records current-C `DS4_CUDA_MOE_ATOMIC_DOWN` behavior on B300:
device-zero initialization followed by token-indexed float atomic accumulation
for both tile8 and tile4 row32 down schedules. It leaves tile16/rowspan
scheduling, shared-cache specialization, Q4_K, hyperconnection, runtime
route, and C CUDA removal unclaimed.

Validate the M14.5c2c5 Rust CUDA routed MoE tile16 row32 smoke:

```sh
python3 ds4-parity/check_routed_moe_tile16_row32_smoke.py --negative-test
```

The fixture records current-C tile16 atomic-down scheduling behavior on B300:
separate tile16 down descriptors, retained tile8 gate metadata, and partial
tile16 token-indexed float atomic accumulation. It leaves gate/down
row2048/rowspan scheduling, shared-cache specialization, Q4_K,
hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2c6 Rust CUDA routed MoE gate-rowspan smoke:

```sh
python3 ds4-parity/check_routed_moe_gate_rowspan_smoke.py --negative-test
```

The fixture records current-C widened tile8 gate scheduling behavior on B300:
functional row512, row1024, and row2048 output equivalence over retained
expert-tile descriptors. It leaves widened down scheduling, shared-cache
specialization, Q4_K, hyperconnection, runtime route, and C CUDA removal
unclaimed.

Validate the M14.5c2c7 Rust CUDA routed MoE down-rowspan smoke:

```sh
python3 ds4-parity/check_routed_moe_down_rowspan_smoke.py --negative-test
```

The fixture records current-C widened tile16 atomic-down scheduling behavior
on B300: functional row512, row1024, and row2048 output equivalence over
retained tile16 descriptors. It leaves shared-cache specialization, Q4_K,
hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2d Rust CUDA single-token Q4_K routed MoE smoke:

```sh
python3 ds4-parity/check_routed_moe_q4_k_single_smoke.py --negative-test
```

The fixture records the current-C single-token type-12 path on B300: packed
Q4_K/Q8_K gate/up and direct six-expert down output with optional auxiliary
writes. It leaves shared-cache expert-tile specialization, hyperconnection,
runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2e Rust CUDA routed MoE shared-cache smoke:

```sh
python3 ds4-parity/check_routed_moe_shared_cache_smoke.py --negative-test
```

The fixture records current-C expert-tile shared-memory behavior on B300:
synchronized cached Q8/IQ2/Q2 inputs and row512, row1024, and row2048
gate/down output equivalence. It leaves generic/sorted qwarp fallback
projection, hyperconnection, runtime route, and C CUDA removal unclaimed.

Validate the M14.5c2f Rust CUDA routed MoE qwarp fallback smoke:

```sh
python3 ds4-parity/check_routed_moe_qwarp_fallback_smoke.py --negative-test
```

The fixture records current-C selectable qwarp fallback behavior on B300:
single-token no-decode-LUT generic projection and batched
no-expert-tiles/no-P2 sorted projection. It leaves hyperconnection, runtime
route, and C CUDA removal unclaimed.

Validate the M14.5d Rust CUDA hyperconnection smoke:

```sh
python3 ds4-parity/check_hyperconnection_smoke.py --negative-test
```

The fixture records current-C hyperconnection behavior on B300: Sinkhorn
split, direct and split-stride residual reduction, plain/add expansion,
fused split-plus-normalization, and output weights. It closes M14.5
operation-family ownership while leaving default-route promotion and C CUDA
removal to M14.6.

Validate the M14.6a production route linkage blocker:

```sh
python3 ds4-parity/check_cuda_route_promotion_gate.py --negative-test
```

The fixture records why route promotion was rejected at M14.6a: the
production Linux Rust build still compiled `ds4_cuda.cu`, the cuda-oxide
crate had no linkable `ds4_gpu_*` ABI backend, and the Rust runtime graph
route was unimplemented. M14.6b owns that backend assembly work.

Validate the M14.6b1 Rust CUDA resource ABI exports:

```sh
python3 ds4-parity/check_cuda_abi_resource_smoke.py --negative-test
```

The fixture records the first linkable Rust `ds4_gpu_*` subset: 16
initialization, tensor/resource, copy, synchronization, and managed-KV
symbols emitted from a cuda-oxide `staticlib` and executed on B300. Compute
exports, production linker selection, and route promotion remain pending.

Validate the M14.6b2a Rust CUDA tensor-fill ABI export:

```sh
python3 ds4-parity/check_cuda_abi_tensor_fill_smoke.py --negative-test
```

The fixture records the first linkable Rust compute symbol:
`ds4_gpu_tensor_fill_f32` is implemented through CUDA's stream-ordered D32
memset primitive, with exact float-bit behavior verified on B300. Remaining
graph compute exports, production linker selection, and route promotion remain
pending.

Validate the M14.6b2b1 Rust CUDA embedded elementwise ABI module:

```sh
python3 ds4-parity/check_cuda_abi_elementwise_smoke.py --negative-test
```

The fixture records the first linkable Rust embedded-kernel compute exports:
`ds4_gpu_add_tensor` and `ds4_gpu_repeat_hc_tensor` execute through a C
consumer of `libds4_cuda.a` on B300, including valid in-place add aliasing.
The static-library consumer must currently retain the embedded artifact object
with `--whole-archive`, and the generated object emits an executable-stack
linker warning; production linker selection and remaining graph exports remain
pending.

Validate the M14.6b2b2a Rust CUDA directional-steering ABI export:

```sh
python3 ds4-parity/check_cuda_abi_directional_steering_smoke.py --negative-test
```

The fixture records `ds4_gpu_directional_steering_project_tensor` running
through the reusable Rust embedded-kernel archive in a C consumer on B300,
including exact in-place projection results and invalid-input rejection. The
static-library retention requirement and generated executable-stack warning
remain open production-link integration work; other graph exports and route
promotion remain pending.

Validate the M14.6b2b2b1 Rust CUDA SwiGLU libdevice ABI export:

```sh
python3 ds4-parity/check_cuda_abi_swiglu_libdevice_smoke.py --negative-test
```

The fixture records `ds4_gpu_swiglu_tensor` running from a C consumer of the
Rust static library on B300. Embedded PTX containing exponential libdevice
references is extracted and linked into an `sm_103` cubin at module load
time, then its temporary link directory is removed; clamped, unclamped, and
output/input alias cases pass. Whole-archive retention, executable-stack
warning removal, and the remaining graph exports remain open before route
promotion.

Validate the M14.6b2b2b2a Rust CUDA plain RMS ABI export:

```sh
python3 ds4-parity/check_cuda_abi_plain_rms_smoke.py --negative-test
```

The fixture records `ds4_gpu_rms_norm_plain_tensor` and
`ds4_gpu_rms_norm_plain_rows_tensor` running from a C consumer of the Rust
static library on B300, including batched rows, in-place aliasing, and the
current-C zero-width result boundary. Weighted RMS remains pending because it
reads weights through the model-map range ABI; whole-archive retention and
the generated executable-stack warning also remain open before route
promotion.

Validate the M14.6b2b2b2b1 Rust CUDA weighted RMS device-copy ABI export:

```sh
python3 ds4-parity/check_cuda_abi_weighted_rms_device_copy_smoke.py --negative-test
```

The fixture records `ds4_gpu_rms_norm_weight_tensor` and
`ds4_gpu_rms_norm_weight_rows_tensor` running from a C consumer of the Rust
static library on B300 through an internal immutable weight-range device-copy
cache. It covers batched rows, in-place aliasing, alternate model offsets,
invalid ranges, and current-C zero-width behavior. Public model-map controls,
whole-archive retention, and the generated executable-stack warning remain
open before route promotion.

Validate the M14.6b2b2b2b2a Rust CUDA basic model-control device-copy ABI
export:

```sh
python3 ds4-parity/check_cuda_abi_model_control_device_copy_smoke.py --negative-test
```

The fixture records the public model-map, fd, map-range, and cache-range
symbols executing from a C consumer of the Rust static library on B300. It
pre-caches a weighted RMS range, switches model pointers, mutates the former
mapping to catch stale retained bytes, and checks zero-byte/invalid range
behavior. This stage provides the caller-map device-copy baseline only;
registered/HMM/prefetch, fd-backed direct-I/O, preload selection, q8/f16
cache policy, whole-archive retention, and the executable-stack warning
remain open.

Validate the M14.6b2b2b2b2b1 Rust CUDA registered-attempt device-copy
fallback ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_registered_fallback_smoke.py --negative-test
```

The fixture combines the live B300 registered-range probe, which reports CUDA
error code 801 and a matching device-copy fallback, with a C-linked public ABI
consumer using a page-aligned model mapping. The Rust cache helper attempts
read-only registration only when the rounded host range remains within the
caller-declared map and otherwise retains the device-copy route. Pageable
HMM/prefetch, fd-backed staging, cross-range registration-disable policy,
preload selection, q8/f16 cache hooks, whole-archive retention, and the
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2a Rust CUDA pageable HMM fallback ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_pageable_hmm_smoke.py --negative-test
```

The fixture combines the live B300 pageable-HMM probe with a C-linked public
ABI consumer that selects the deterministic current-C fallback environment:
chunked model copy selected and explicitly suppressed while HMM exclusions
remain absent. Rust retains and consumes only the advised page-rounded
window, while the public cache call correctly reports that the direct HMM
pointer is not a retained cache admission. Chunked-copy
success/allocation-failure routing, global HMM reads outside that window, fd
staging, q8/f16 cache hooks, whole-archive retention, and the
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b1 Rust CUDA chunk-selected model-copy ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_chunk_selected_copy_smoke.py --negative-test
```

The fixture drives the deterministic successful `DS4_CUDA_COPY_MODEL_CHUNKED`
public route through a C-linked consumer, mutates the host weights after
map-range setup, calls map-range again, and requires weighted RMS to observe
the original copied device image. Preceding whole-map registration
precedence, allocation/transfer-failure HMM fallback, copy-chunk override and
discard/progress effects, unconsumed ranges, fd-backed staging, remaining
cache policy, whole-archive retention, and the executable-stack warning
remain open.

Validate the M14.6b2b2b2b2b2b2a Rust CUDA whole-map registration precedence
ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_whole_registration_precedence_smoke.py --negative-test
```

The fixture combines the live B300 read-only registration probe, which reports
CUDA error code 801, with an aligned C-linked public ABI consumer. Rust
attempts whole-map registration when `DS4_CUDA_COPY_MODEL` is empty, while
the observed B300 rejection continues into the previously validated
chunk-selected copied-image path and preserves the original weighted output
after host mutation. Successful global zero-copy registration, fd-backed
staging, residual failure/cache policy, whole-archive retention, and the
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b1 Rust CUDA buffered fd-backed weight-cache
ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_buffered_fd_cache_smoke.py --negative-test
```

The fixture selects `DS4_CUDA_WEIGHT_CACHE=1` with
`DS4_CUDA_NO_DIRECT_IO=1`, configures the fd before establishing an aligned
host map, and deliberately gives the file and host mapping different weights.
On B300, after the recorded whole-map registration rejection, Rust consumes
the original file weights and retains the cached device bytes even after the
backing file is changed. Direct-I/O reopen/alignment, asynchronous staging,
cache-budget and source-page policy, residual failure handling, whole-archive
retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2a Rust CUDA direct-I/O fd-cache ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_direct_io_fd_cache_smoke.py --negative-test
```

The fixture permits direct I/O while selecting public fd-backed caching. The
C-linked B300 consumer proves fd-sourced weighted output and retained cache
reuse; because that output cannot reveal the read mechanism, the fixture
separately cites the refreshed M14.1b2b3b1 B300 probe that observes aligned
`O_DIRECT` selection, exact readback, and tail buffered fallback. Persistent
direct-read error disablement, asynchronous staging, cache-budget and
source-page policy, residual failure handling, whole-archive retention, and
the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b1 Rust CUDA direct-I/O error-disable ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_direct_io_error_disable_smoke.py --negative-test
```

The fixture records public source ownership of current-C selected-direct-read
disable classes and a B300 feature-policy test, while reusing the preceding
direct-enabled C-linked public consumer as a success regression. No live
public B300 request is claimed to have induced a disabling direct-read error.
Asynchronous staging, cache-budget and source-page policy, residual
model-control handling, whole-archive retention, and the executable-stack
warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2a Rust CUDA direct-I/O async staging ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_direct_io_async_staging_smoke.py --negative-test
```

The fixture selects direct-enabled public fd caching with a 16 MiB staging
chunk and requests five chunks through a C-linked B300 consumer. Public
output proves fd-backed computation and retained cache reuse; source wiring
and the earlier lower-level asynchronous staging baseline establish the
four-slot event-ring contract without claiming that the C-linked output
exposes event counts. Buffered-only asynchronous staging, arena/cache-budget
and source-page/progress policy, residual model-control selection,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b1 Rust CUDA buffered fd async staging ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_buffered_fd_async_staging_smoke.py --negative-test
```

The fixture selects buffered-only public fd caching with
`DS4_CUDA_NO_DIRECT_IO=1`, sets a 16 MiB chunk, and requests five chunks
through a C-linked B300 consumer. Public output proves buffered fd-sourced
computation and retained cache reuse, while the existing lower-level
asynchronous staging baseline remains the event-count proof. Arena/cache
budget, source-page/progress policy, residual model-control selection,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2a Rust CUDA public fd arena
suballocation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_arena_suballocation_smoke.py --negative-test
```

The fixture selects buffered public fd caching with a bounded 256 MiB arena
chunk override and consumes two disjoint cached ranges through a C-linked
B300 consumer. Public output proves fd-sourced results and retained reuse for
both ranges after backing-file mutation; source wiring plus the retained
M14.1b2b3b2 lower-level baseline establish arena ownership without claiming
that the public output exposes arena count or device offsets. Cache-budget
fallback, source-page/progress policy, residual model-control selection,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b1 Rust CUDA public fd cache-budget
fallback ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_cache_budget_smoke.py --negative-test
```

The fixture admits one small buffered fd-backed range under a one GiB cache
limit, then requests a one GiB range whose source pages are inaccessible and
whose file bytes do not exist. A successful C-linked B300 call proves
pre-transfer budget fallback and retained reuse of the admitted range without
claiming public compute through the uncached fallback pointer. Source-page
progress policy, residual model-control selection, whole-archive retention,
and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2a Rust CUDA public fd source-page
and progress ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_source_page_progress_smoke.py --negative-test
```

The fixture performs two ordinary and one suppressed multi-chunk buffered fd
uploads through the C-linked B300 consumer. It interposes the advisory calls
made by the linked Rust static library and captures stderr to prove source
file/mapping discard invocation, non-TTY progress plus reset on model
replacement, and `DS4_CUDA_KEEP_MODEL_PAGES`/verbose suppression. It does not
claim physical page eviction or TTY refresh rendering. Residual model-control
selection, whole-archive retention, and the executable-stack warning remain
open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public cross-range
registration-disable ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_registration_disable_smoke.py --negative-test
```

The fixture interposes `cuMemHostRegister_v2` from a C-linked B300 consumer
with error code 801. It proves that the public range-cache path attempts
whole-map then first-range registration, suppresses a subsequent disjoint
range attempt after the selected failure, and retries after model
replacement resets the gate. Weighted RMS results verify device-copy
fallback output; successful zero-copy registration, remaining failure
selection, whole-archive retention, and the executable-stack warning remain
open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public full-model
copy selection ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_full_model_copy_smoke.py --negative-test
```

The fixture selects nonempty `DS4_CUDA_COPY_MODEL` from a C-linked B300
consumer, interposes host registration, mutates host weights after model-map
setup, and replaces the map. It proves that successful full-model copying
skips registration and retains the copied device image for weighted RMS
reads. Allocation or transfer failure continuation into registration is
source-backed but not forced in live execution; remaining failure selection,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public direct-model
read selection ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_direct_model_read_smoke.py --negative-test
```

The fixture selects nonempty `DS4_CUDA_DIRECT_MODEL` from a C-linked B300
consumer after deterministically rejecting whole-map registration. It proves
that weighted reads observe host mutation without a per-range
registration/copy admission and that `ds4_gpu_cache_model_range` reports this
direct pointer as uncached. The corrected pageable-HMM predecessor proves the
same cache-return boundary for prefetched host reads. Remaining failure
selection, whole-archive retention, and the executable-stack warning remain
open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public default fd
selection ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_default_fd_selection_smoke.py --negative-test
```

The fixture supplies a bound model fd from a C-linked B300 consumer after
deterministically rejecting whole-map registration. It proves that buffered
fd reads do not require `DS4_CUDA_WEIGHT_CACHE`, remain selected under
`DS4_CUDA_WEIGHT_PRELOAD`, and are bypassed by `DS4_CUDA_NO_FD_CACHE` before
fallback range handling. Remaining failure selection, whole-archive
retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 Rust CUDA public fd-budget
fallback cache-result ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_budget_cache_result_smoke.py --negative-test
```

The fixture admits a near-one-GiB buffered fd-backed range through a
C-linked B300 consumer, then forces a small raw fd budget fallback with
divergent host and file weights. It proves that
`ds4_gpu_cache_model_range` reports the fallback as uncached while weighted
RMS consumes the returned host bytes. Arena allocation failure, persistent
cache-full state, `DS4_CUDA_STRICT_WEIGHT_CACHE` continuation, whole-archive
retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a Rust CUDA public fd-arena
failure-selection ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_arena_failure_selection_smoke.py --negative-test
```

The fixture interposes the 256 MiB fd-arena device allocation from a
C-linked B300 consumer. It proves non-strict allocation failure returns an
uncached host pointer, strict failure continues into cached device-copy
fallback, and the model-lifetime cache-full state suppresses a second arena
allocation attempt. Aligned-budget strict routing is source-backed but not
separately forced live; staging allocation/read/copy failure selection,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba Rust CUDA public fd-upload
failure-continuation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_upload_failure_continuation_smoke.py --negative-test
```

The fixture selects the direct fd branch from a C-linked B300 consumer,
injects one asynchronous fd upload copy failure, rejects registration, and
uses divergent fd and host weights to prove that failure continues to cached
device-copy fallback rather than retrying a buffered fd upload. Staged
allocation, fd-read, event, and final synchronization failure observations,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba Rust CUDA public fd stage
pool reuse ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_stage_pool_reuse_smoke.py --negative-test
```

The fixture selects buffered fd caching from a C-linked B300 consumer,
creates one four-slot pinned stage pool, then arms `cuMemAllocHost_v2`
failure before a second disjoint range. File-backed weighted output from the
second range proves the sufficient stage pool is reused without another
pinned allocation or registration fallback. Initial allocation and
pool-growth failure, fd-read, event, and final synchronization observations,
whole-archive retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba Rust CUDA public fd
stage-allocation failure continuation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_stage_allocation_failure_smoke.py --negative-test
```

The fixture selects buffered fd caching from a C-linked B300 consumer,
forces `cuMemAllocHost_v2` failure before two disjoint ranges across a
strict-mode transition, and rejects the first range-registration attempt.
Host-backed cached weighted output from both ranges proves failed stage
allocation continues through device-copy fallback independently of strict
mode; the repeated allocation failure proves it does not latch arena
cache-full state. Fd-read, event, final synchronization, whole-archive
retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba Rust CUDA public fd-read
failure continuation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_read_failure_smoke.py --negative-test
```

The fixture selects buffered fd caching from a C-linked B300 consumer,
injects `EIO` from `pread` only for the configured model fd before two
disjoint ranges across a strict-mode transition, and rejects the first
range-registration attempt. Host-backed cached weighted output from both
ranges proves buffered read failure continues through device-copy fallback
independently of strict mode; two injected reads prove failure does not latch
arena cache-full state. Event, final synchronization, whole-archive
retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba Rust CUDA public fd
event-record failure continuation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_event_record_failure_smoke.py --negative-test
```

The fixture selects buffered fd caching from a C-linked B300 consumer,
forwards `cuEventRecord` through setup, injects event-record failure before
two disjoint ranges across a strict-mode transition, and rejects the first
range-registration attempt. Host-backed cached weighted output from both
ranges proves event-record failure continues through device-copy fallback
independently of strict mode; two injected records prove failure does not
latch arena cache-full state. Event-wait, final synchronization, whole-archive
retention, and the executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba Rust CUDA public fd
event-wait failure continuation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_event_wait_failure_smoke.py --negative-test
```

The fixture selects buffered fd caching from a C-linked B300 consumer,
requests ranges exceeding the four-slot event ring, injects
`cuEventSynchronize` failure on fifth-chunk slot reuse across a strict-mode
transition, and rejects the first range-registration attempt. Host-backed
cached weighted output from both ranges proves event-wait failure continues
through device-copy fallback independently of strict mode. Final
stream-synchronization, whole-archive retention, and the executable-stack
warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba Rust CUDA public fd
final upload synchronization failure continuation ABI:

```sh
python3 ds4-parity/check_cuda_abi_model_control_fd_final_sync_failure_smoke.py --negative-test
```

The fixture selects buffered fd caching from a C-linked B300 consumer,
injects one `cuStreamSynchronize` failure per staged fd request across a
strict-mode transition, and forwards subsequent synchronizations so fallback
completion remains observable. Host-backed cached weighted output from both
ranges proves final synchronization failure continues through device-copy
fallback independently of strict mode. Q8/f16 hooks, graph compute,
whole-archive retention, route promotion, and the executable-stack warning
remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba Rust CUDA public
single-token F16 projection ABI:

```sh
python3 ds4-parity/check_cuda_abi_matmul_f16_single_token_smoke.py --negative-test
```

The C-linked B300 fixture dispatches `ds4_gpu_matmul_f16_tensor` through the
default ordered-chunks, forced base, and forced serial single-token paths,
then mutates host model bytes to verify the cached F16 range remains the
projection weight source. Its recorded rejection of multi-token projection
is a historical boundary now consumed by the multi-token F16 BLAS successor
below.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbba Rust CUDA public
single-token paired F16 projection ABI:

```sh
python3 ds4-parity/check_cuda_abi_matmul_f16_pair_single_token_smoke.py --negative-test
```

The C-linked B300 fixture dispatches `ds4_gpu_matmul_f16_pair_tensor` through
its default paired ordered-chunks path and its no-pair, no-ordered, and
serial independent fallback selections, then mutates host model bytes to
verify both cached F16 weight ranges remain authoritative. Its recorded
rejection of multi-token projection is a historical boundary now consumed by
the multi-token F16 BLAS successor below.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbba Rust CUDA public
single-token F32 projection ABI:

```sh
python3 ds4-parity/check_cuda_abi_matmul_f32_single_token_smoke.py --negative-test
```

The C-linked B300 fixture dispatches `ds4_gpu_matmul_f32_tensor` through its
single-token base kernel, then mutates host model bytes to verify the cached
F32 range remains authoritative. Its recorded rejection of multi-token
projection is a historical boundary now consumed by the successor below.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbba Rust CUDA
public multi-token F32 BLAS projection ABI:

```sh
python3 ds4-parity/check_cuda_abi_matmul_f32_multi_token_blas_smoke.py --negative-test
```

The C-linked B300 fixture first exercises the retained single-token base
kernel, then mutates host model bytes and dispatches two tokens through the
cuda-oxide cuBLAS adapter to prove cached F32 weights remain authoritative
across both branches; it also unsets an initialization-time
`DS4_CUDA_NO_TF32` before BLAS execution to verify the math-mode selection is
retained. Multi-token F16 BLAS, Q8/F16 cache hooks, quality-mode mutation,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba Rust CUDA
public multi-token F16 BLAS projection ABI:

```sh
python3 ds4-parity/check_cuda_abi_matmul_f16_multi_token_blas_smoke.py --negative-test
```

The C-linked B300 fixture establishes the existing one-token and paired F16
paths, mutates host model bytes, and then executes multi-token mixed-precision
projection and paired delegation through the cuda-oxide BLAS adapter. A
non-half-exact second token proves F32-to-F16 activation conversion on the
BLAS route, while `DS4_CUDA_SERIAL_F16_MATMUL` proves the multi-token F32
activation fallback is retained. Q8/F16 cache hooks, quality-mode mutation,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba Rust CUDA
public Q8 cache and quality-controls ABI:

```sh
python3 ds4-parity/check_cuda_abi_q8_quality_controls_smoke.py --negative-test
```

The C-linked B300 fixture preloads eligible packed Q8 ranges into retained
F16 and optional F32 converted buffers, verifies exact preload reuse and
quality-mode cache suppression through device-memory deltas, calls the
public memory-report hook, and uses multi-token F32 BLAS output to prove
quality and `DS4_CUDA_NO_TF32` change effective math selection. Public Q8
matmul consumers, remaining graph compute, whole-archive retention policy,
route promotion, C CUDA removal, and the embedded-object executable-stack
warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba Rust CUDA
public Q8 matmul ABI:

```sh
python3 ds4-parity/check_cuda_abi_q8_matmul_smoke.py --negative-test
```

The C-linked B300 fixture executes `ds4_gpu_matmul_q8_0_tensor` through
default DP4A, the scalar-disable override, native batched and generic
dispatch, and opt-in F16 and F32 expanded BLAS routes. Specialized Q8
pair/HC consumers, remaining graph compute, whole-archive retention policy,
route promotion, C CUDA removal, and the embedded-object executable-stack
warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba Rust CUDA
public hyperconnection expansion ABI:

```sh
python3 ds4-parity/check_cuda_abi_hc_expand_smoke.py --negative-test
```

The C-linked B300 fixture exercises direct post/comb expansion, split
expansion, split-plus-add expansion, and aliased `block_add == block_out`
behavior through the embedded Rust kernel. Fused public Q8 HC consumers,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba Rust CUDA
public fused Q8 hyperconnection consumers ABI:

```sh
python3 ds4-parity/check_cuda_abi_fused_q8_hc_smoke.py --negative-test
```

The C-linked B300 fixture executes the public fused matmul and shared-down
consumers through DP4A and scalar Q8 paths, the disabled-fusion delegation
path, and aliased routed-output addition. The internal Q8 pair consumer,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba Rust CUDA
public shared gate/up Q8 SwiGLU ABI:

```sh
python3 ds4-parity/check_cuda_abi_shared_gate_up_swiglu_q8_smoke.py --negative-test
```

The C-linked B300 fixture executes the public shared gate/up consumer through
its private paired DP4A and scalar routes plus its disabled-pair public
fallback, and checks clamped SwiGLU output and invalid range rejection.
Remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba Rust CUDA
public hyperconnection weighted-sum ABI:

```sh
python3 ds4-parity/check_cuda_abi_hc_weighted_sum_smoke.py --negative-test
```

The C-linked B300 fixture executes direct and split-stride weighted-sum
reductions through the embedded Rust kernel, with noisy unused split entries
to prove stride selection, then checks short-input and zero-shape rejection.
Sinkhorn, fused reductions, remaining graph compute, whole-archive retention
policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba Rust CUDA
public hyperconnection split-Sinkhorn ABI:

```sh
python3 ds4-parity/check_cuda_abi_hc_split_sinkhorn_smoke.py --negative-test
```

The C-linked B300 fixture executes two-row split-Sinkhorn generation through
the embedded Rust kernel, validates alternate cached model parameter ranges
and shorter-output row flooring, then checks invalid model and HC inputs.
Fused split-weighted reductions, remaining graph compute, whole-archive
retention policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba Rust CUDA
public hyperconnection split weighted-sum ABI:

```sh
python3 ds4-parity/check_cuda_abi_hc_split_weighted_sum_smoke.py --negative-test
```

The C-linked B300 fixture executes synchronized split generation and
weighted-sum reduction through one embedded Rust kernel, validates emitted
split values, output-defined row count, and alternate cached model parameter
ranges, then checks invalid tensor and model spans. Normalized fused
reduction, output HC weights, remaining graph compute, whole-archive
retention policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba Rust
CUDA public hyperconnection split weighted-sum norm ABI:

```sh
python3 ds4-parity/check_cuda_abi_hc_split_weighted_sum_norm_smoke.py --negative-test
```

The C-linked B300 fixture executes the one-row fused normalization kernel,
validates alternate cached model parameter ranges, and proves both
environment-disabled and multi-row fallback behavior, then checks invalid
normalization and tensor spans. Output HC weights, remaining graph compute,
whole-archive retention policy, route promotion, C CUDA removal, and the
embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public output hyperconnection weights ABI:

```sh
python3 ds4-parity/check_cuda_abi_output_hc_weights_smoke.py --negative-test
```

The C-linked B300 fixture executes multi-token and single-token
sigmoid-plus-eps output weights through the embedded Rust kernel, validates
alternate cached model parameter ranges, then checks short input, partial
output row, model-range, and zero-hyperconnection rejection. Remaining graph
compute, whole-archive retention policy, route promotion, C CUDA removal, and
the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public embedding hyperconnection ABI:

```sh
python3 ds4-parity/check_cuda_abi_embedding_smoke.py --negative-test
```

The C-linked B300 fixture executes single-token hidden replication and
batched token embedding through the embedded Rust kernels, validates
invalid-token row-zero fallback and alternate cached model ranges, then
checks Rust's guarded rejection of unsafe single-token and short-buffer
requests. Remaining graph compute, whole-archive retention policy, route
promotion, C CUDA removal, and the embedded-object executable-stack warning
remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public head RMS norm ABI:

```sh
python3 ds4-parity/check_cuda_abi_head_rms_norm_smoke.py --negative-test
```

The C-linked B300 fixture executes multi-row in-place head normalization
through the embedded Rust reduction kernel, then checks short-tensor,
zero-dimension, and null rejection. RoPE, remaining graph compute,
whole-archive retention policy, route promotion, C CUDA removal, and the
embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public FP8 KV quantization ABI:

```sh
python3 ds4-parity/check_cuda_abi_fp8_kv_quantize_smoke.py --negative-test
```

The C-linked B300 fixture executes E4M3FN round-trip quantization only over
the non-RoPE prefix through the embedded Rust kernel, proves partial-chunk
handling and preserved empty-prefix/zero-width no-op behavior, then checks
invalid-shape and null rejection. Standalone RoPE, raw KV storage,
compressor, attention, remaining graph compute, whole-archive retention
policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public indexer QAT ABI:

```sh
python3 ds4-parity/check_cuda_abi_indexer_qat_smoke.py --negative-test
```

The C-linked B300 fixture executes the normalized Hadamard plus E2M1FN
block-quantization path over two 128-wide rows through the embedded Rust
kernel, then checks short-tensor, zero-row, wrong-width, and null rejection.
Standalone RoPE, raw KV storage, compressor, attention, routed MoE,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public standalone RoPE ABI:

```sh
python3 ds4-parity/check_cuda_abi_rope_tail_smoke.py --negative-test
```

The C-linked B300 fixture executes interpolated forward and inverse YaRN
rotary-tail transforms through the embedded Rust kernel, proves the
non-RoPE prefix remains unchanged, then checks zero-pair, invalid-shape, and
null rejection. Raw KV storage, compressor, attention, routed MoE, remaining
graph compute, whole-archive retention policy, route promotion, C CUDA
removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public raw KV storage ABI:

```sh
python3 ds4-parity/check_cuda_abi_raw_kv_storage_smoke.py --negative-test
```

The C-linked B300 fixture executes the single-row and batched public raw KV
store wrappers through one embedded Rust kernel, proves FP16 round-trip
storage and `uint32_t` wraparound row selection on distinct destinations, then
checks short-span, zero-grid, and null rejection. Overlapping same-launch
ring writes, composed FP8/raw storage, compressor, attention, routed MoE,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public composed FP8 raw KV storage ABI:

```sh
python3 ds4-parity/check_cuda_abi_composed_kv_fp8_raw_store_smoke.py --negative-test
```

The C-linked B300 fixture executes the public quantize-then-raw-store wrapper
through the two existing embedded Rust kernels, proves FP8 prefix mutation,
untouched rotary tail, FP16 raw-row storage, and `uint32_t` row modulo, then
checks current-C failure ordering by rejecting a short raw destination only
after retaining the FP8 mutation in `kv`. Compressor, attention, routed MoE,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public compressor batch-store ABI:

```sh
python3 ds4-parity/check_cuda_abi_compressor_store_batch_smoke.py --negative-test
```

The C-linked B300 fixture executes the public compressor batch-store wrapper
through one embedded Rust kernel, proves ratio-4 state placement, FP32 and
FP16 APE reads, non-power-of-two `uint32_t` position wrap, and untouched
rows, then checks invalid-range, invalid-shape, overflow, and null rejection.
Compressor update/prefill, attention, routed MoE, remaining graph compute,
whole-archive retention policy, route promotion, C CUDA removal, and the
embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public compressor ratio-4 state-only ABI:

```sh
python3 ds4-parity/check_cuda_abi_compressor_state_ratio4_smoke.py --negative-test
```

The C-linked B300 fixture executes the public ratio-4 state-only wrapper
through one embedded Rust set-rows kernel after stream-ordered state
initialization. It proves FP32 and FP16 APE state placement, initialized
untouched rows, and invalid model-range no-mutation ordering, then checks
invalid-shape, overflow, and null rejection. Compressor update/full prefill,
attention, routed MoE, remaining graph compute, whole-archive retention
policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public compressor ratio-4 replay ABI:

```sh
python3 ds4-parity/check_cuda_abi_compressor_replay_ratio4_smoke.py --negative-test
```

The C-linked B300 fixture executes the public ratio-4 replay wrapper through
one embedded replay-pool kernel composed with weighted RMS, stride-4 RoPE,
optional FP8 processing, and final state rebuild. It proves FP32 and FP16 APE
output, output-before-state ordering, zero-RoPE handling, and validation
rejection. Compressor update/general prefill, attention, routed MoE, remaining
graph compute, whole-archive retention policy, route promotion, C CUDA
removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public compressor update ABI:

```sh
python3 ds4-parity/check_cuda_abi_compressor_update_smoke.py --negative-test
```

The C-linked B300 fixture executes the public update wrapper through embedded
update-pool and ratio-4 shift kernels composed with store, weighted RMS, and
stride-1 RoPE paths. It proves store-only non-emission, ratio-4 and
general-ratio emission, wrapped-position emission, and the emitted
zero-RoPE partial-failure boundary, then checks validation rejection. General
prefill, attention, routed MoE, remaining graph compute, whole-archive
retention policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public general compressor prefill ABI:

```sh
python3 ds4-parity/check_cuda_abi_compressor_prefill_smoke.py --negative-test
```

The C-linked B300 fixture executes the public non-replay prefill wrapper
through one embedded pool kernel composed with state-row placement, weighted
RMS, stride-by-ratio RoPE, and optional FP8 processing. It proves ratio-4
previous/remainder state banks, general-ratio remainder state, `n_comp == 0`
state-only success, zero-RoPE compressed output, wrapped positions, and
validation rejection. Attention, routed MoE, remaining graph compute,
whole-archive retention policy, route promotion, C CUDA removal, and the
embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public single-token attention decode heads ABI:

```sh
python3 ds4-parity/check_cuda_abi_attention_decode_heads_smoke.py --negative-test
```

The C-linked B300 fixture executes the public single-token mixed attention
wrapper through embedded generic and score-cap overflow-online kernels. It
proves masked and raw-only wrapped-ring output, sink softmax, overflow
dispatch and current-C overflow raw visibility, environment/mask rejection,
and validation rejection. Remaining attention, routed MoE, remaining graph
compute, whole-archive retention policy, route promotion, C CUDA removal, and
the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public batched attention decode ABI:

```sh
python3 ds4-parity/check_cuda_abi_attention_decode_batch_smoke.py --negative-test
```

The C-linked B300 fixture executes the public raw and mixed batched attention
wrappers through the generalized embedded decode kernel family. It proves
masked mixed and raw-only causal ring/window output, compressed visibility,
sink softmax, forced overflow-online dispatch, explicit online-window
execution, and rejection controls while preserving the preceding
single-token witness. Prefill, indexed and output-Q8 attention, routed MoE,
remaining graph compute, whole-archive retention policy, route promotion, C
CUDA removal, and the embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public indexed batched attention ABI:

```sh
python3 ds4-parity/check_cuda_abi_attention_indexed_batch_smoke.py --negative-test
```

The C-linked B300 fixture executes the public indexed mixed batched attention
wrapper through embedded sort, generic, online heads8, and rb4 kernels. It
proves filtered top-k and ratio-zero compressed visibility, causal raw
window/ring behavior, sink softmax, sorted-online dispatch, sort-disable,
two-pass rb4 and forced-generic environment gates, and validation rejection.
Prefill and output-Q8 attention, routed MoE, remaining graph compute,
whole-archive retention policy, route promotion, C CUDA removal, and the
embedded-object executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public low-Q8 attention output ABI:

```sh
python3 ds4-parity/check_cuda_abi_attention_output_low_q8_smoke.py --negative-test
```

The C-linked B300 fixture executes the public single-token low-Q8 output
wrapper through the existing embedded Q8 quantizer and the grouped output-A
projection kernel. It proves partial-block native projection, DP4A
environment-gate equivalence, and rejection controls. Batched output-Q8 and
prefill attention, routed MoE, remaining graph compute, whole-archive
retention policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

Validate the M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba Rust
CUDA public batched Q8 attention output ABI:

```sh
python3 ds4-parity/check_cuda_abi_attention_output_q8_batch_smoke.py --negative-test
```

The C-linked B300 fixture executes the public batched Q8 output wrapper
through native grouped output-A, optional F16-rounded safe-SGEMM output-A,
and attention-labeled output-B projection. It proves the cuBLAS selection and
fallback gates, output-B cache-label behavior, and rejection controls.
Prefill attention, routed MoE, remaining graph compute, whole-archive
retention policy, route promotion, C CUDA removal, and the embedded-object
executable-stack warning remain open.

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
