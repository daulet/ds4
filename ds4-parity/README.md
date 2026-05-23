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
