# M0.2 Build Baseline Manifest

## Capture Scope

- Work item: M0.2 Baseline Build Command Capture
- Source oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Capture commit: `004ade62556bcfa7c8950a624619f8492230102b`
- Drift policy: no source behavior changes are introduced by this capture.
- Manifest rule: rerun the listed command in the listed environment and compare
  exit status plus declared artifact/log behavior.

## Local Machine

- Cwd: `/Users/dzhanguzin/dev/personal/ds4`
- Metadata log: `logs/m0.2-local-machine.txt`
- Machine: Apple M4 Pro, macOS Darwin 25.4.0.
- Default shell architecture: `x86_64`.
- Intended local build architecture: `arm64` via `arch -arm64`.
- Default compiler target: `x86_64-apple-darwin25.4.0`.
- Arm64 compiler target: `arm64-apple-darwin25.4.0`.
- Model availability: `ds4flash.gguf` is absent locally.

## Local Command Entries

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| `make clean` | default shell | 0 | `logs/m0.2-make-clean.log` | Clean target removes ignored build products. |
| `make` | default shell | 2 | `logs/m0.2-make.log` | Recorded local Rosetta failure: Apple clang rejects `-mcpu=native` for the default `x86_64-apple-darwin25.4.0` target. |
| `arch -arm64 make clean` | local arm64 | 0 | `logs/m0.2-arm64-make-clean.log` | Clean target succeeds before the local Metal build. |
| `arch -arm64 make` | local arm64 | 0 | `logs/m0.2-arm64-make.log` | Builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent` for the local Metal backend. |
| `arch -arm64 make test` | local arm64, no model | 2 | `logs/m0.2-arm64-make-test.log` | Test binary builds, then the default all-test run stops at missing `ds4flash.gguf`. |
| `arch -arm64 ./ds4_test --server` | local arm64, no model | 0 | `logs/m0.2-arm64-ds4-test-server.log` | Server parser/rendering/cache unit tests pass without a model. |
| `arch -arm64 ./ds4_test --metal-kernels` | local arm64 Metal | 0 | `logs/m0.2-arm64-ds4-test-metal-kernels.log` | Isolated Metal kernel numeric check passes on Apple M4 Pro. |
| `arch -arm64 make clean` | local arm64 | 0 | `logs/m0.2-arm64-make-clean-before-cpu.log` | Clean target succeeds before the CPU build. |
| `arch -arm64 make cpu` | local arm64 | 0 | `logs/m0.2-arm64-make-cpu.log` | Builds CPU-only `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`. |
| `file ds4 ds4-server ds4-bench ds4-eval ds4-agent` | after local CPU build | 0 | `logs/m0.2-arm64-cpu-artifacts.log` | CPU build artifacts are arm64 Mach-O executables. |
| `arch -arm64 make cuda-regression` | local macOS | 0 | `logs/m0.2-arm64-make-cuda-regression.log` | Darwin target records that CUDA regression requires a CUDA build. |

## B300 CUDA Command Entries

- Kubeconfig workflow: temporary per-session copy at
  `/tmp/ds4-hou2-prod1.kubeconfig`; host kubectl context left unchanged.
- Context: `hou2-prod1`
- Namespace: `default`
- Pod: `ds4-rust-port-b300`
- Node: `c1v17-b300n1-nic1`
- GPU: NVIDIA B300 SXM6 AC, UUID `GPU-81f6bd2a-3404-6445-1788-365264243aab`
- Pod environment log: `logs/m0.2-b300-env.log`
- Pod creation logs: `logs/m0.2-b300-pod-apply.log`,
  `logs/m0.2-b300-pod-wait.log`
- Source copy log: `logs/m0.2-b300-source-copy.log`

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| `kubectl apply` for `ds4-rust-port-b300` | `hou2-prod1/default` | 0 | `logs/m0.2-b300-pod-apply.log` | Reusable B300 pod created for this port. |
| `kubectl wait pod/ds4-rust-port-b300 --for=condition=Ready --timeout=10m` | `hou2-prod1/default` | 0 | `logs/m0.2-b300-pod-wait.log` | Pod reached Ready on `c1v17-b300n1-nic1`. |
| `git archive HEAD \| kubectl exec ... tar -xf - -C /workspace/ds4` | local to B300 pod | 0 | `logs/m0.2-b300-source-copy.log` | Capture commit source copied without local uncommitted artifacts. |
| `make cuda-generic` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.2-b300-make-cuda-generic.log` | Builds CUDA `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`. |
| `file ds4 ds4-server ds4-bench ds4-eval ds4-agent` | after B300 CUDA build | 0 | `logs/m0.2-b300-cuda-artifacts.log` | CUDA build artifacts are x86-64 Linux ELF executables. |
| `make cuda-regression` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.2-b300-make-cuda-regression.log` | CUDA backend initializes on NVIDIA B300 SXM6 AC (`sm_103`) and the long-context CUDA regression passes. |
| `make test` | B300 pod, no model | 2 | `logs/m0.2-b300-make-test.log` | Test binary builds, then the default all-test run stops at missing `ds4flash.gguf`. |
| `./ds4_test --server` | B300 pod, no model | 0 | `logs/m0.2-b300-ds4-test-server.log` | Server parser/rendering/cache unit tests pass. |
| `./ds4_test --metal-kernels` | B300 pod CUDA backend | 0 | `logs/m0.2-b300-ds4-test-metal-kernels.log` | Backend tensor kernel check passes through CUDA despite the legacy flag name; stdout/stderr ordering may differ under `kubectl exec`. |

## Blocked Or Deferred Model-Backed Entries

These are not M0.2 failures; they require the model fixture captured in later
Milestone 0 items.

- `./ds4_test --logprob-vectors`
- `./ds4_test --long-context`
- `./ds4_test --tool-call-quality`
- `ds4-bench` short and long prompt CSV captures
- Server request traces that load the model

The exact blocker in both local and B300 environments is absence of
`ds4flash.gguf`. M0.3 must either provide `DS4_TEST_MODEL` or record the model
download/path/hash before claiming model-backed parity coverage.

# M0.3 Official Vector Logprob Baseline

## Capture Scope

- Work item: M0.3 Official Vector Logprob Baseline
- Source oracle commit: `9e35378f7f759fb63d3591641d6e9b65a9f0672b`
- Oracle: current C/CUDA `./ds4_test --logprob-vectors` path.
- Fixture: `tests/test-vectors/official.vec`
- Fixture SHA256: `0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac`
- Fixture size: 1207 bytes.
- Fixture hash log: `logs/m0.3-official-vec-hash.log`
- Comparator: rerun the exact command below and compare exit status plus vector
  case outcome lines.
- Acceptance: command exits 0, `logprob-vectors: OK` appears, selected greedy
  vector tokens match exactly, and the runner's documented
  `long_memory_archive` official-graph mismatch skip remains explicit in the
  log.
- Drift policy: future Rust selected greedy tokens must match exactly; numeric
  logprob slices use the tolerance defined by the later Rust parity harness.

## B300 Model Fixture

- Context: `hou2-prod1`
- Namespace: `default`
- Pod: `ds4-rust-port-b300`
- Node: `c1v17-b300n1-nic1`
- Model target: `q2-imatrix`
- Model path: `/workspace/ds4/ds4flash.gguf`
- Resolved model path:
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Model size: 86,720,111,488 bytes.

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| `shasum -a 256 tests/test-vectors/official.vec && wc -c tests/test-vectors/official.vec` | local repo | 0 | `logs/m0.3-official-vec-hash.log` | Fixture hash and size match the values above. |
| refresh `/workspace/ds4` from `git archive HEAD` | local to B300 pod | 0 | `logs/m0.3-b300-source-refresh.log` | Pod source matches capture commit without local uncommitted artifacts. |
| `DS4_GGUF_DIR=/workspace/ds4/gguf ./download_model.sh q2-imatrix` | B300 pod | 0 | `logs/m0.3-b300-download-wrapper.log`, `logs/m0.3-b300-download-summary.log` | q2-imatrix download completed and `ds4flash.gguf` was linked. |
| inspect `ds4flash.gguf` and `gguf/` | B300 pod | 0 | `logs/m0.3-b300-model-files.log` | Symlink resolves to the q2-imatrix GGUF under `/workspace/ds4/gguf`. |
| `sha256sum $(readlink -f ds4flash.gguf)` | B300 pod | 0 | `logs/m0.3-b300-model-hash.log` | Model hash and byte size match the values above. |
| `make ds4_test` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.3-b300-make-ds4-test.log` | CUDA test binary builds from the capture commit. |
| `DS4_TEST_MODEL=/workspace/ds4/ds4flash.gguf DS4_TEST_VECTOR_FILE=tests/test-vectors/official.vec ./ds4_test --logprob-vectors` | B300 pod, CUDA backend | 0 | `logs/m0.3-b300-logprob-vectors.log` | Official-vector logprob baseline passes on NVIDIA B300 SXM6 AC (`sm_103`). |

# M0.4 Server Trace Baselines

## Capture Scope

- Work item: M0.4 Server Trace Baselines
- Source oracle commit: `3d87577962abeac1ab0d80e9c21d0012bfc53afb`
- Oracle: current C/CUDA `./ds4-server` with human-readable tracing enabled.
- Fixture directory: `server-fixtures/m0.4/`
- Artifact directory: `server-traces/m0.4/`
- Comparator: replay the listed request JSON files against the current server
  command in the table order within one server process, then compare HTTP
  status, response shape, usage/cache fields, and normalized trace events.
- Acceptance: all replay entries return HTTP 200; JSON responses parse; SSE
  response contains normal chat completion chunks and terminal usage; tool-call
  response maps generated DSML back to OpenAI `tool_calls`; thinking-disabled
  response emits visible content without reasoning; cache continuation records a
  live-prefix cache reuse.
- Drift policy: prompt rendering, response object shape, tool-call fields,
  thinking mode, cache source, token-count fields, and finish reasons must match
  unless a later milestone explicitly approves a change.

## Normalization Rules

Future comparators must normalize values that are expected to vary between
server runs:

- Wall-clock timestamps in response `created` fields, trace request headings,
  and server logs.
- Generated OpenAI object IDs and tool call IDs.
- Throughput, elapsed time, model-load timing, and CUDA progress rates in
  `server.log` and trace summaries.
- Header values only if a future server emits time-varying headers. The current
  captured headers contain status, content length/type, and connection close.
- Stored header and SSE artifacts use LF line endings; comparators may
  normalize CRLF response framing and a terminal SSE delimiter blank line.
- Absolute pod-local artifact paths when the repo-relative artifact path is
  unchanged.

The following fields are intentional behavioral surface and should not be
normalized away: HTTP status, endpoint, stream mode, `think_mode`, rendered
prompt structure, generated visible text, tool function name and arguments,
finish reason, `prompt_tokens`, `completion_tokens`, `cached_tokens`,
`cache_write_tokens`, `cache_source`, and trace cache-decision fields.

## B300 Server Fixture

- Context: `hou2-prod1`
- Namespace: `default`
- Pod: `ds4-rust-port-b300`
- Node: `c1v17-b300n1-nic1`
- GPU: NVIDIA B300 SXM6 AC, UUID
  `GPU-81f6bd2a-3404-6445-1788-365264243aab`
- Model path: `/workspace/ds4/ds4flash.gguf`
- Resolved model path:
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Model size: 86,720,111,488 bytes.
- `ds4-server` SHA256:
  `8b03ac65417e5481ff4a0cad44657284e9c567d3c3edf2602e92d7a04f3f87d2`
- Server command log: `server-traces/m0.4/logs/capture-env.txt`
- Server trace: `server-traces/m0.4/traces/server.trace`
- Server stdout/stderr log: `server-traces/m0.4/logs/server.log`
- Replay log: `server-traces/m0.4/logs/replay.log`
- Per-request response headers: `server-traces/m0.4/headers/*.headers.txt`
- Startup models snapshot: `server-traces/m0.4/responses/models.json`
- Disk-KV file listing: `server-traces/m0.4/logs/kv-files.txt`
- Artifact hashes: `server-traces/m0.4/logs/artifact-sha256.txt`
- Artifact byte sizes: `server-traces/m0.4/logs/artifact-sizes.txt`
- Disk KV is configured in this capture, but the prompts intentionally stay
  below the 512-token disk-KV write threshold; `kv-files.txt` is expected to be
  empty for M0.4. Full disk-KV file coverage is deferred to M0.5.
- Replay order is load-bearing: use one freshly started `ds4-server` process
  and run the table in order. `chat_cache_seed` must run immediately before
  `chat_cache_continuation` because the latter validates live-token cache reuse
  from the same process.

| Replay | Endpoint | Fixture | Response | Expected behavioral marker |
| --- | --- | --- | --- | --- |
| `models` | `GET /v1/models` | readiness probe | `server-traces/m0.4/responses/models.json` | Lists `deepseek-v4-flash`, `context_length=32768`, `max_completion_tokens=64`, and supported chat parameters. |
| `chat_basic` | `/v1/chat/completions` | `server-fixtures/m0.4/chat_basic.json` | `server-traces/m0.4/responses/chat_basic.json` | Non-streaming chat returns `baseline ready`, `finish_reason=stop`. |
| `chat_stream` | `/v1/chat/completions` | `server-fixtures/m0.4/chat_stream.json` | `server-traces/m0.4/responses/chat_stream.sse` | SSE stream emits chat deltas plus terminal usage. |
| `chat_tool_call` | `/v1/chat/completions` | `server-fixtures/m0.4/chat_tool_call.json` | `server-traces/m0.4/responses/chat_tool_call.json` | DSML tool output maps to OpenAI `tool_calls` for `list_files` with `{"path":"."}`. |
| `chat_thinking_disabled` | `/v1/chat/completions` | `server-fixtures/m0.4/chat_thinking_disabled.json` | `server-traces/m0.4/responses/chat_thinking_disabled.json` | Explicit `thinking: {"type":"disabled"}` renders `think_mode: none` and visible answer `2`. |
| `chat_cache_seed` | `/v1/chat/completions` | `server-fixtures/m0.4/chat_cache_seed.json` | `server-traces/m0.4/responses/chat_cache_seed.json` | Seed request writes the live prompt suffix and returns `cache ready`. |
| `chat_cache_continuation` | `/v1/chat/completions` | `server-fixtures/m0.4/chat_cache_continuation.json` | `server-traces/m0.4/responses/chat_cache_continuation.json` | Continuation reuses the live prefix: `cache_source=memory-token`, `cached_tokens=41`, `cache_write_tokens=9`. |

## Command Entries

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| refresh `/workspace/ds4` from `git archive HEAD` | local to B300 pod, explicit `--context hou2-prod1` | 0 | `logs/m0.4-b300-source-refresh.log` | Pod source matches the source oracle commit without local uncommitted artifacts. |
| `make clean ds4-server` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.4-b300-make-ds4-server.log` | CUDA server binary is rebuilt from the source oracle commit. |
| `sha256sum ds4-server && file ds4-server` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.4-b300-ds4-server-sha256.log` | Rebuilt server binary hash matches the value recorded above. |
| copy `server-fixtures/m0.4/` into the pod | local to B300 pod, explicit `--context hou2-prod1` | 0 | `logs/m0.4-b300-fixture-copy.log` | All six fixture JSON files are available in the pod workspace. |
| traced replay script | B300 pod, `/workspace/ds4` | 0 | `logs/m0.4-b300-server-replay.log` | All six request replays return HTTP 200 and artifacts are written under `server-traces/m0.4/`. |
| `jq empty` over fixture and response JSON files | local repo | 0 | command output not persisted | All request fixtures and JSON responses parse. |

# M0.5 KV And Snapshot Baselines

## Capture Scope

- Work item: M0.5 KV And Snapshot Baselines
- Source oracle commit: `0623bbb4d97d056a58e208e324216f97abed685e`
- Oracle: current C/CUDA `./ds4-server` disk-KV store and restore path.
- Fixture directory: `kv-fixtures/m0.5/`
- Artifact directory: `kv-artifacts/m0.5/`
- Comparator: replay the listed request JSON files across three fresh
  `ds4-server` lifetimes sharing one disk-KV directory, then compare HTTP
  status, response usage cache fields, normalized trace cache decisions, parsed
  KV headers, rendered cached text, and KV file hashes.
- Acceptance: cold seed returns HTTP 200 and writes a 550-token disk-KV entry;
  same-prompt restart restores 550 tokens from `disk-text`; continuation
  restart restores the 552-token shutdown prefix from `disk-text` and writes
  the 9-token suffix. KV headers parse as version 1 `KVC` files with quant 2,
  context 32768, and the recorded reason/token fields.
- Drift policy: future Rust-written KV files must match the current format and
  rendered text byte-for-byte until a versioned format change is explicitly
  introduced. Header creation and last-used timestamps are normalized by
  zeroing bytes 24 through 39 before hashing.

## Normalization Rules

Future comparators must normalize values that are expected to vary between
runs:

- KV fixed-header `created_unix` and `last_used_unix` fields at bytes 24
  through 39. `kv-file-normalized-sha256.txt` records hashes after this
  zeroing; full raw file hashes are retained in `kv-file-sha256.txt`.
- Wall-clock timestamps in response `created` fields, trace request headings,
  replay logs, and server logs.
- Generated OpenAI object IDs.
- Throughput, elapsed time, model-load timing, CUDA progress rates, and local
  port numbers in server logs and traces.
- Absolute pod-local paths when repo-relative artifact paths are unchanged.

The following fields are behavioral surface and should not be normalized away:
HTTP status, prompt token counts, generated content, finish reason,
`cached_tokens`, `cache_write_tokens`, `cache_source`, `disk_cached_tokens`,
disk-cache filename, KV magic/version/quant/reason/ext flags/token count/hits,
context size, payload byte count, rendered text bytes, and rendered text hash.

## B300 KV Fixture

- Context: `hou2-prod1`
- Namespace: `default`
- Pod: `ds4-rust-port-b300`
- Node: `c1v17-b300n1-nic1`
- GPU: NVIDIA B300 SXM6 AC, UUID
  `GPU-81f6bd2a-3404-6445-1788-365264243aab`
- Model path: `/workspace/ds4/ds4flash.gguf`
- Resolved model path:
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Model size: 86,720,111,488 bytes.
- `ds4-server` SHA256:
  `038f660eaf6fb1d6d1eca53faf543fe800bf10330ce56159fb5541c29ff76608`
- Server command log: `kv-artifacts/m0.5/logs/capture-env.txt`
- Replay log: `kv-artifacts/m0.5/logs/replay.log`
- Cache decision summary: `kv-artifacts/m0.5/logs/cache-decisions.txt`
- Parsed KV metadata: `kv-artifacts/m0.5/logs/kv-header.tsv`
- Raw KV file hashes: `kv-artifacts/m0.5/logs/kv-file-sha256.txt`
- Timestamp-normalized KV file hashes:
  `kv-artifacts/m0.5/logs/kv-file-normalized-sha256.txt`
- Extracted rendered cache text: `kv-artifacts/m0.5/rendered-text/*.txt`
- Per-server traces: `kv-artifacts/m0.5/traces/server-*.trace`
- Per-server stdout/stderr logs: `kv-artifacts/m0.5/logs/server-*.log`
- Artifact hashes: `kv-artifacts/m0.5/logs/artifact-sha256.txt`
- Artifact byte sizes: `kv-artifacts/m0.5/logs/artifact-sizes.txt`
- Raw `.kv` binaries are intentionally not checked in because each generated
  file is roughly 30 MiB. They remain reproducible from the fixture and command
  entries; committed artifacts include full hashes, normalized hashes, parsed
  headers, and extracted rendered text.

| Replay | Server lifetime | Fixture | Response | Expected behavioral marker |
| --- | --- | --- | --- | --- |
| `seed_miss` | `server-a` | `kv-fixtures/m0.5/kv_seed.json` | `kv-artifacts/m0.5/responses/seed_miss.json` | Cold request: `prompt_tokens=550`, `cached_tokens=0`, `cache_write_tokens=550`, trace `cache_source=none`, generated content `I notice`, `finish_reason=length`. |
| `seed_restore` | `server-b` | `kv-fixtures/m0.5/kv_seed.json` | `kv-artifacts/m0.5/responses/seed_restore.json` | Restart restore: `prompt_tokens=550`, `cached_tokens=550`, `cache_write_tokens=0`, trace `cache_source=disk-text`, `disk_cached_tokens=550`, cache file `0ab2314538b11686a11e296b7f697651fbd17e60.kv`. |
| `continuation_restore` | `server-c` | `kv-fixtures/m0.5/kv_continuation.json` | `kv-artifacts/m0.5/responses/continuation_restore.json` | Continuation restore: `prompt_tokens=561`, `cached_tokens=552`, `cache_write_tokens=9`, trace `cache_source=disk-text`, `disk_cached_tokens=552`, cache file `a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv`, generated content `kv continued`. |

## KV Files

| File | Reason | Tokens | Hits | Rendered bytes | Size bytes | Comparator |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `0ab2314538b11686a11e296b7f697651fbd17e60.kv` | `cold` | 550 | 1 | 2520 | 31,529,520 | Full and timestamp-normalized hashes in `kv-file-*.txt`; rendered text in `rendered-text/0ab2314538b11686a11e296b7f697651fbd17e60.txt`. |
| `a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv` | `shutdown` | 552 | 1 | 2528 | 31,583,296 | Full and timestamp-normalized hashes in `kv-file-*.txt`; rendered text in `rendered-text/a0cac6ff193696ccb5d7e9ae151d7255d39cf161.txt`. |
| `4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.kv` | `shutdown` | 563 | 0 | 2632 | 31,690,964 | Full and timestamp-normalized hashes in `kv-file-*.txt`; rendered text in `rendered-text/4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.txt`. |

## Command Entries

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| refresh `/workspace/ds4` from `git archive HEAD` | local to B300 pod, explicit `--context hou2-prod1` | 0 | `logs/m0.5-b300-source-refresh.log` | Pod source matches the source oracle commit without local uncommitted artifacts. |
| copy `kv-fixtures/m0.5/` into the pod | local to B300 pod, explicit `--context hou2-prod1` | 0 | `logs/m0.5-b300-fixture-copy.log` | All fixture files are available in the pod workspace without macOS sidecar files. |
| `make clean ds4-server` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.5-b300-make-ds4-server.log` | CUDA server binary is rebuilt from the source oracle commit. |
| `sha256sum ds4-server && file ds4-server` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.5-b300-ds4-server-sha256.log` | Rebuilt server binary hash matches the value recorded above. |
| three-lifetime disk-KV replay script | B300 pod, `/workspace/ds4` | 0 | `kv-artifacts/m0.5/logs/replay.log` | `seed_miss`, `seed_restore`, and `continuation_restore` all return HTTP 200 with the expected cache markers. |
| parse generated `.kv` headers and rendered text | B300 pod, `/workspace/ds4` | 0 | `kv-artifacts/m0.5/logs/kv-header.tsv` | All generated cache files parse as version 1 `KVC` files with the recorded reason, token, hit, context, payload, rendered-text, and size fields. |
| `jq empty` over fixture and response JSON files | local repo | 0 | command output not persisted | All request fixtures and JSON responses parse. |
| `sha256sum -c kv-artifacts/m0.5/logs/artifact-sha256.txt` | local repo | 0 | command output not persisted | Committed non-raw-KV artifacts match the recorded hashes. |

# M0.6 Benchmark CSV Baselines

## Capture Scope

- Work item: M0.6 Benchmark CSV Baselines
- Source oracle commit: `add2c507f81aa2e363809213771134c282c50bf2`
- Oracle: current C/CUDA `./ds4-bench` throughput benchmark.
- Prompt fixture: `speed-bench/promessi_sposi.txt`
- Prompt SHA256:
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`
- Prompt size: 1,329,139 bytes.
- Artifact directory: `bench/m0.6/`
- Comparator: rerun the listed `ds4-bench` commands on the same B300 class,
  model, backend, prompt, context sweep, and generation-token count; then
  compare CSV schema, context frontiers, prefill interval sizes, generation
  token counts, KV snapshot byte counts, and throughput ratios.
- Acceptance: short-context CSV contains rows at 2048, 4096, 6144, and 8192
  tokens; long-context CSV contains rows at 16384, 24576, and 32768 tokens; all
  rows use 32 greedy generation tokens; CSV headers match
  `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes`;
  benchmark commands exit 0 and artifact hashes verify.
- Drift policy: CSV schema, workload parameters, prompt hash, model hash,
  backend, context frontiers, generation-token counts, and `kvcache_bytes` are
  exact comparison surface. `prefill_tps` and `gen_tps` are performance surface:
  compare only on the same machine class and treat regressions larger than 10%
  as requiring explicit review unless a later milestone defines a different
  threshold.

## B300 Benchmark Fixture

- Context: `hou2-prod1`
- Namespace: `default`
- Pod: `ds4-rust-port-b300`
- Node: `c1v17-b300n1-nic1`
- GPU: NVIDIA B300 SXM6 AC, UUID
  `GPU-81f6bd2a-3404-6445-1788-365264243aab`
- Driver: 580.126.16
- Reported power limit: 1100.00 W
- Model path: `/workspace/ds4/ds4flash.gguf`
- Resolved model path:
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Model size: 86,720,111,488 bytes.
- `ds4-bench` SHA256:
  `380c3299214a25cd5492e9272e5edabcbf047dc7ecd58cc874c05c1f647793d2`
- Capture environment: `bench/m0.6/logs/capture-env.txt`
- CSV summary: `bench/m0.6/logs/csv-summary.json`
- Artifact hashes: `bench/m0.6/logs/artifact-sha256.txt`
- Artifact byte sizes: `bench/m0.6/logs/artifact-sizes.txt`

| CSV | Command | Rows | Expected behavioral marker |
| --- | --- | ---: | --- |
| `bench/m0.6/csv/b300-short.csv` | `./ds4-bench -m /workspace/ds4/ds4flash.gguf --cuda --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 8192 --step-incr 2048 --gen-tokens 32 --csv ds4-parity/baselines/bench/m0.6/csv/b300-short.csv` | 4 | Context frontiers 2048/4096/6144/8192; each row prefills a 2048-token interval, decodes 32 tokens, and records KV snapshot bytes from 52,184,460 to 136,750,476. |
| `bench/m0.6/csv/b300-long.csv` | `./ds4-bench -m /workspace/ds4/ds4flash.gguf --cuda --prompt-file speed-bench/promessi_sposi.txt --ctx-start 16384 --ctx-max 32768 --step-incr 8192 --gen-tokens 32 --csv ds4-parity/baselines/bench/m0.6/csv/b300-long.csv` | 3 | Context frontiers 16384/24576/32768; rows prefill 16384 then two 8192-token intervals, decode 32 tokens, and record KV snapshot bytes from 249,505,164 to 475,014,540. |

## Command Entries

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| refresh `/workspace/ds4` from `git archive HEAD` | local to B300 pod, explicit `--context hou2-prod1` | 0 | `logs/m0.6-b300-source-refresh.log` | Pod source matches the source oracle commit and prompt fixture hash. |
| `make clean ds4-bench` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.6-b300-make-ds4-bench.log` | CUDA benchmark binary is rebuilt from the source oracle commit. |
| short-context `ds4-bench` run | B300 pod, `/workspace/ds4` | 0 | `bench/m0.6/logs/b300-short.stderr.log`, `bench/m0.6/logs/replay.log` | Writes `bench/m0.6/csv/b300-short.csv` with four rows and the expected CSV schema. |
| long-context `ds4-bench` run | B300 pod, `/workspace/ds4` | 0 | `bench/m0.6/logs/b300-long.stderr.log`, `bench/m0.6/logs/replay.log` | Writes `bench/m0.6/csv/b300-long.csv` with three rows and the expected CSV schema. |
| CSV parser and summary check | B300 pod, `/workspace/ds4` | 0 | `bench/m0.6/logs/csv-summary.json` | Both CSVs parse with the expected header, non-empty row sets, and `gen_tokens=32` in every row. |
| local CSV parser check | local repo | 0 | command output not persisted | CSV headers, context rows, prefill intervals, generation-token counts, and positive throughput values match the manifest. |
| `sha256sum -c bench/m0.6/logs/artifact-sha256.txt` | local repo | 0 | command output not persisted | Committed benchmark artifacts match the recorded hashes. |
