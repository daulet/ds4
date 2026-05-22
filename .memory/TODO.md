# DS4 Rust Port TODO

## Active Items

### M0.1: Port Execution Protocol

- Status: done
- Goal: add project-local port protocol, active board, parity status, and
  lessons log; document commit-sized Milestone 0 work items in
  `RUST_PORT_ROADMAP.md`.
- Source evidence needed: `AGENT.md`, `CONTRIBUTING.md`,
  `RUST_PORT_ROADMAP.md`, current git status.
- Oracle: current checked-out C implementation at the starting commit.
- Comparator: no source behavior changes; diff is docs/state-only.
- Validation needed: `git diff --name-only` lists only roadmap and `.memory/`
  files; `git diff --check` reports no whitespace errors.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/`.

### M0.2: Baseline Build Command Capture

- Status: done
- Goal: capture reproducible baseline command lines and logs for `make`,
  `make test`, `make cpu`, and backend-specific regression commands.
- Source evidence needed: `Makefile`, `CONTRIBUTING.md`, local model/backend
  availability, B300 pod availability for GPU-heavy CUDA validation.
- Oracle: current C build products from the checked-out commit.
- Comparator: baseline manifest reruns compare exit status and declared output
  artifacts.
- Validation needed: execute available commands, record logs and blocked remote
  commands with exact rerun instructions.
- Owner path: `ds4-parity/baselines/`, `.memory/status.md`.

### M0.3: Official Vector Logprob Baseline

- Status: done
- Goal: capture the current implementation's output for
  `tests/test-vectors/official.vec`.
- Source evidence needed: `tests/test-vectors/README.md`,
  `tests/test-vectors/official.vec`, `tests/ds4_test.c`, model availability on
  local or B300 environment.
- Oracle: current `./ds4_test --logprob-vectors` path.
- Comparator: baseline manifest entry plus later replay through `ds4-parity`.
- Validation needed: run with exact `DS4_TEST_MODEL` and
  `DS4_TEST_VECTOR_FILE` or record the exact model acquisition blocker.
- Owner path: `ds4-parity/baselines/`, `tests/test-vectors/`.

### M0.4: Server Trace Baselines

- Status: done
- Goal: capture representative current server request behavior.
- Source evidence needed: `ds4_server.c`, `tests/ds4_test.c`, server CLI help,
  request/trace fixtures to be created under `ds4-parity/baselines/`.
- Oracle: current `./ds4-server` binary with trace enabled.
- Comparator: request replay records response JSON or event stream plus trace
  files; later Rust server responses compare after approved normalization.
- Validation needed: build/run current server on B300 with the q2-imatrix model,
  replay fixed request JSON fixtures, and record command, response, trace, model
  identity, and normalization rules.
- Owner path: `ds4-parity/baselines/`, `.memory/status.md`.

### M0.5: KV And Snapshot Baselines

- Status: done
- Goal: capture current KV-cache and session-restore behavior for prompt reuse.
- Source evidence needed: `ds4_server.c`, KV-store helpers and tests in
  `tests/ds4_test.c`, existing M0.4 cache trace, and model-backed B300
  workspace state.
- Oracle: current C KV store and session snapshot implementation.
- Comparator: binary file hashes and trace-normalized cache decisions.
- Validation needed: generate prompt inputs, cache directory, KV files, trace
  output, and manifest entries describing cache hit, cache miss, prefix match,
  and restore behavior.
- Owner path: `ds4-parity/baselines/`, `.memory/status.md`.

### M0.6: Benchmark CSV Baselines

- Status: done
- Goal: capture at least one short-context and one long-context `ds4-bench`
  CSV for the reference backend.
- Source evidence needed: `ds4-bench` CLI behavior, model-backed B300
  workspace state, benchmark prompt fixture, and machine/backend metadata.
- Oracle: current `./ds4-bench` binary on the same machine, model, backend,
  context settings, and power state.
- Comparator: CSV schema and selected throughput fields compared by later
  `ds4-parity` helpers.
- Validation needed: generate benchmark prompt fixtures, run short-context and
  long-context B300 captures, record command lines, CSV outputs, model identity,
  backend, and normalization/performance comparison policy.
- Owner path: `ds4-parity/baselines/bench/`, `.memory/status.md`.

### M1.1: Parity Harness Work Item Breakdown

- Status: done
- Goal: split Milestone 1 into reviewable work items in `RUST_PORT_ROADMAP.md`
  before adding Rust harness code.
- Source evidence needed: Milestone 0 manifest sections and artifact layout,
  existing build/test commands, and the roadmap's Milestone 1 deliverables.
- Oracle: Milestone 0 baseline artifacts.
- Comparator: documentation-only change; no source behavior changes.
- Validation needed: `git diff --check`, review that each proposed Milestone 1
  item has a tangible goal, oracle, comparator, acceptance rule, and validation
  gate.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/status.md`.

### M1.2: Static Baseline Artifact Verifier

- Status: done
- Goal: add a local `ds4-parity` verifier that checks committed Milestone 0
  artifacts without rerunning model-backed commands.
- Source evidence needed: M0 artifact hash files, M0.4/M0.5 JSON and trace
  artifacts, M0.5 KV metadata, M0.6 benchmark CSVs, and the baseline manifest.
- Oracle: committed Milestone 0 artifacts and their hash/shape contracts.
- Comparator: checked-in verifier command that validates hashes, parses
  structured artifacts, and emits a standard report.
- Validation needed: verifier exits 0 on committed baselines; a negative test
  against a temporary changed fixture fails; `git diff --check`.
- Owner path: `ds4-parity/`, `ds4-parity/baselines/`.

### M1.3: Server And KV Normalization Comparators

- Status: done
- Goal: add comparison helpers for M0.4 server traces/responses and M0.5 KV
  restore artifacts.
- Source evidence needed: M0.4 server fixtures, responses, traces, documented
  normalization rules, M0.5 cache decision logs, rendered cached text, and
  parsed `kv-header.tsv`.
- Oracle: M0.4 `server-traces/m0.4/` and M0.5 `kv-artifacts/m0.5/`.
- Comparator: harness command or module comparing a candidate artifact
  directory to the baseline after only documented normalizations.
- Validation needed: self-comparison succeeds; temporary edits to cache source,
  cached token count, finish reason, KV reason, or rendered text fail;
  `git diff --check`.
- Owner path: `ds4-parity/`, `ds4-parity/baselines/`.

### M1.4: Logprob And Numeric Comparator

- Status: done
- Goal: add numeric comparison support for official-vector output and later
  tensor/logit fixtures.
- Source evidence needed: M0.3 official-vector baseline,
  `tests/test-vectors/official.vec`, model identity, and captured B300 logprob
  output.
- Oracle: M0.3 official-vector baseline and current
  `./ds4_test --logprob-vectors` output.
- Comparator: harness logic parsing exact selected-token outcomes and numeric
  slices with explicit tolerances in the report.
- Validation needed: current captured M0.3 output compares cleanly; negative
  tests for token drift and numeric drift fail; `git diff --check`.
- Owner path: `ds4-parity/`, `tests/test-vectors/`.

### M1.5: Benchmark CSV Comparator

- Status: done
- Goal: add comparison support for M0.6 benchmark CSV baselines.
- Source evidence needed: M0.6 `bench/m0.6/csv/b300-short.csv`,
  `bench/m0.6/csv/b300-long.csv`, prompt hash, model hash, B300 machine record,
  and `csv-summary.json`.
- Oracle: M0.6 benchmark CSV baselines captured on the B300 node.
- Comparator: harness logic validating CSV schema, context frontiers, prefill
  intervals, generation-token counts, `kvcache_bytes`, and throughput ratios
  against the documented drift policy.
- Validation needed: committed M0.6 CSVs self-compare cleanly; schema,
  frontier, generation-token, cache-byte, and throughput-threshold edits fail;
  `git diff --check`.
- Owner path: `ds4-parity/`, `ds4-parity/baselines/bench/m0.6/`.

### M1.6: Oracle Runner And Unified Report

- Status: done
- Goal: add a single documented command that can run available current-C oracle
  checks and compare their outputs to Milestone 0 baselines.
- Source evidence needed: current C binaries, Milestone 0 baselines, local
  no-model fixture availability, B300-routed model fixtures, and the baseline
  manifest.
- Oracle: current C binaries plus Milestone 0 baselines.
- Comparator: report-producing `ds4-parity` command that either runs available
  oracle checks locally or marks model/GPU cases with exact B300 rerun commands,
  then compares produced artifacts with the M1.2 through M1.5 helpers.
- Validation needed: local no-model checks run without `ds4flash.gguf`;
  model-backed checks are either executed on the B300 pod or reported as
  skipped with exact reproduction commands; unified report has no unexpected
  baseline drift; `git diff --check`.
- Owner path: `ds4-parity/`, `ds4-parity/baselines/`.

### M2.1: Rust Workspace And FFI Skeleton

- Status: done
- Goal: add Rust workspace and crate skeletons without changing existing C
  binary behavior.
- Source evidence needed: Milestone 2 roadmap, current `Makefile`, existing C
  build/test commands, and `ds4_gpu.h` backend ABI.
- Oracle: current C build and test commands.
- Comparator: existing `make`, `make test`, `make cpu`, plus Rust workspace
  tests that do not require a full model.
- Validation needed: existing C targets still build/test as before; Rust crates
  build and test independently without requiring `ds4flash.gguf`; `git
  diff --check`.
- Owner path: Rust workspace files to be introduced, `Makefile`.

### M3.1: Backend ABI Wrapper Parity

- Status: done
- Goal: port the first safe Rust wrapper slice around `ds4_gpu.h` without
  changing the C backend ABI.
- Source evidence needed: `ds4_gpu.h`, existing C backend implementations,
  `tests/ds4_test.c`, and the M2 Rust workspace crates.
- Oracle: direct calls to the current C backend ABI.
- Comparator: small tensor fixtures for allocation, fill, copy, read/write,
  byte-size queries, command flushing, and error paths.
- Validation needed: Rust wrapper tests match direct C ABI results
  byte-for-byte for simple tensor operations; error cases return equivalent
  failure categories; existing C tests still pass.
- Owner path: `rust/ds4-gpu/`, `rust/ds4-gpu-sys/`.

### M4.1: Model Metadata Work Item Breakdown

- Status: done
- Goal: split Milestone 4 into commit-sized GGUF/model-metadata parity work
  items before adding Rust parser or runtime code.
- Source evidence needed: Milestone 4 roadmap text, current C loader and model
  metadata structures, available local/B300 model fixtures, and existing parity
  harness conventions.
- Oracle: current C loader behavior and any C metadata dump contract selected
  for Milestone 4.
- Comparator: documentation-only work item list that defines the metadata dump
  comparison contract for later executable items.
- Validation needed: `RUST_PORT_ROADMAP.md` and `.memory/` changes only; each
  proposed Milestone 4 item has a tangible goal, oracle, comparator,
  acceptance rule, and validation gate; `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M4.2: C Metadata Dump Oracle

- Status: done
- Goal: add a current-C metadata dump helper that opens GGUF through the
  existing loader and emits deterministic machine-readable metadata without
  running inference.
- Source evidence needed: `model_open`, `parse_metadata`, `parse_tensors`,
  `model_summary`, `config_validate_model`, `weights_bind`,
  `mtp_weights_bind`, the B300 q2-imatrix model identity, and any small GGUF
  fixture needs.
- Oracle: direct current C loader and binding behavior.
- Comparator: a `ds4-parity` checker that parses the dump JSON and validates
  schema, deterministic ordering, selected metadata values, tensor directory
  summaries, type histograms, and bound semantic tensor names.
- Validation needed: helper builds, local small-fixture dump passes schema
  checks, supported-model capture is run on B300 or recorded with an exact
  rerun command, and `git diff --check`.
- Owner path: C dump helper surface, `ds4-parity/`,
  `ds4-parity/baselines/metadata/`, `.memory/status.md`.

### M4.3: Rust GGUF Directory Parser

- Status: done
- Goal: add a `ds4-gguf` parser for GGUF v3 header, metadata descriptors,
  scalar and array value decoding, tensor directory parsing, alignment,
  absolute offsets, and tensor byte sizing.
- Source evidence needed: M4.2 C metadata dump schema, `parse_metadata`,
  `parse_tensors`, `gguf_types`, `tensor_nbytes`, `cursor_*`, and the B300
  q2-imatrix metadata dump evidence.
- Oracle: M4.2 C metadata dumps for the same GGUF files.
- Comparator: Rust dump output compared to the C dump for version, counts,
  key/type/value summaries, tensor names, dims, types, relative and absolute
  offsets, byte sizes, and type histograms.
- Validation needed: Rust parser tests, C-vs-Rust dump comparison, negative
  fixture checks, `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: new `rust/ds4-gguf/` crate or equivalent parser module,
  `ds4-parity/`, `.memory/status.md`.

### M4.4: DS4 Metadata Validation Parity

- Status: done
- Goal: port DS4-specific metadata validation from `config_validate_model`,
  including required key lookup, numeric type coercions, compression ratio
  arrays, SwiGLU clamp arrays, RoPE constants, HC constants, and expert
  routing constants.
- Source evidence needed: `config_validate_model`, M4.2 metadata dump schema,
  M4.3 Rust GGUF metadata value model, supported-model metadata, and generated
  metadata mutation fixtures.
- Oracle: C validation behavior and first-failure messages from the M4.2 dump
  helper.
- Comparator: C and Rust validation runs compared by pass/fail, normalized
  first failing key, expected/got category, and tolerance policy.
- Validation needed: metadata validation comparator, negative fixtures,
  workspace Rust tests, and `git diff --check`.
- Owner path: `rust/ds4-gguf/`, `ds4-parity/`, `.memory/status.md`.

### M4.5: Tensor Binding And Layout Parity

- Status: done
- Goal: port the semantic tensor binding and layout checks from `weights_bind`,
  `mtp_weights_bind`, `weights_validate_layout`, and
  `mtp_weights_validate_layout`.
- Source evidence needed: C bound tensor table and layout validation from the
  M4.2 dump helper, tensor naming rules, optional and conditional tensor rules,
  supported-model tensor directory, and generated tensor mutation fixtures.
- Oracle: C bound tensor table and layout validation from the M4.2 dump helper.
- Comparator: compare bound semantic tensor names to tensor descriptor
  identity, dims, type, offsets, byte size, optional-vs-required status, and
  normalized first failure.
- Validation needed: tensor binding comparator, negative fixtures, workspace
  Rust tests, and `git diff --check`.
- Owner path: `rust/ds4-gguf/`, `ds4-parity/`, `.memory/status.md`.

### M4.6: Metadata Baselines And Unified Report Integration

- Status: done
- Goal: commit supported-model metadata baselines and wire metadata comparison
  into the unified parity report.
- Source evidence needed: M4.2 B300 metadata dump, M4.3 through M4.5 C/Rust
  metadata comparators, recorded B300 model path, size, SHA256, and exact B300
  refresh commands.
- Oracle: current C metadata dump captured on the B300 q2-imatrix model with
  the recorded model path, size, and SHA256.
- Comparator: a metadata comparator that self-compares committed baselines,
  compares candidate C/Rust dumps, and detects scalar, array, tensor shape,
  tensor type, binding, and offset drift.
- Validation needed: metadata comparator self-checks, negative tests, unified
  parity report, any required B300 capture command, and `git diff --check`.
- Owner path: `ds4-parity/baselines/metadata/`, `ds4-parity/`,
  `.memory/status.md`.

### M4.7: Unsupported GGUF Negative Fixtures

- Status: done
- Goal: lock down unsupported and malformed GGUF behavior before Rust metadata
  is used by runtime code.
- Source evidence needed: current C loader and validation failures from the
  M4.2 dump helper, Rust GGUF parser errors, and existing generated metadata
  validation/layout fixtures.
- Oracle: current C loader and validation failures from the M4.2 dump helper.
- Comparator: C and Rust validator runs compared by exit status and normalized
  first error category/key.
- Validation needed: negative fixture generation/comparison, workspace Rust
  tests, and `git diff --check`.
- Owner path: `ds4-parity/`, `rust/ds4-gguf/`, `.memory/status.md`.

### M5.1: Tokenization Work Item Breakdown

- Status: done
- Goal: split Milestone 5 tokenization, prompt rendering, and DSML parity into
  reviewable Rust port work items before adding runtime-facing text code.
- Source evidence needed: current C tokenizer loading/encoding/decoding, chat
  prompt rendering, DSML tool-call parsing/rendering, CLI/server request paths,
  and available Milestone 0 request/vector fixtures.
- Oracle: `RUST_PORT_ROADMAP.md` Milestone 5 plus current C CLI/server
  rendering and token handling.
- Comparator: documentation-only work item list that defines concrete fixtures,
  C dump helpers if needed, candidate Rust APIs, acceptance rules, drift
  policies, review gates, and validation gates.
- Validation needed: `git diff --check`.
- Owner path: `.memory/TODO.md`, `.memory/status.md`,
  `RUST_PORT_ROADMAP.md` source evidence.

### M5.2: C Token And Prompt Dump Oracle

- Status: done
- Goal: add a deterministic current-C oracle that dumps tokenizer vocabulary
  identity, text tokenization, rendered chat prompt bytes, rendered prompt token
  IDs, CLI chat-construction token streams, token pieces, and request fixture
  identity without running inference.
- Source evidence needed: `vocab_load`, `bpe_tokenize_text`,
  `tokenize_rendered_chat_vocab`, `ds4_dump_text_tokenization`,
  `render_chat_prompt_text`, `parse_chat_request`, `ds4_chat_begin`,
  `ds4_chat_append_message`, `ds4_chat_append_assistant_prefix`, CLI
  `build_prompt`, `special_token_at`, `vocab_token_is_literal_special`, M0.3
  official vectors, and M0.4/M0.5 server request fixtures and traces.
- Oracle: current C tokenizer and prompt renderer on the B300 q2-imatrix model,
  using `/workspace/ds4/ds4flash.gguf` and the recorded model SHA256.
- Comparator: schema checker for committed token/prompt dumps, including a
  canonical tokenizer identity section with token count, sha256 over ordered
  token byte strings, merge count, sha256 over ordered merge pairs/ranks, and a
  sorted `special_token_at` name/id table plus a sorted
  `vocab_token_is_literal_special` id/bytes table for literal-special decoding.
  `vocab_load` reads only `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`
  into `ds4_vocab`, so token type/score metadata is intentionally outside this
  tokenizer identity surface. Negative tests cover missing fixture records,
  token-count drift, token-bytes hash drift, merge hash drift, special-token ID
  drift, literal-special decoding table drift, prompt-byte drift, token ID
  drift, and token-piece drift; the CLI fixture family records per-step append
  operations and final token streams without a unified rendered-prompt byte
  field.
- Acceptance: committed fixtures cover plain text, rendered chat specials,
  thinking enabled/disabled/max, system/developer text, tool/function results,
  OpenAI tools, existing M0.4 server requests, and CLI direct-token prompt
  construction variants; tokenizer identity fields are present and hash only
  canonical ordered token bytes and ordered merge records, not capture paths or
  timestamps.
- Drift policy: model path and capture workspace may be normalized; rendered
  server prompt bytes, server token IDs, CLI append operation sequences, CLI
  token streams, token pieces, request fixture names, model identity, token byte
  hash, merge hash, and special-token table are exact. Server trace timestamps,
  request IDs, host/port strings, and process IDs may be normalized; request
  fixture names and sha256 over exact request body bytes as received, with no
  JSON re-serialization, are exact.
- Review gate: ask Claude to review fixture coverage and dump schema against
  tokenizer and prompt-rendering source.
- Validation needed: B300 capture command, schema/negative checks, local build
  of the dump helper, and `git diff --check`.
- Owner path: `ds4.c`, `ds4.h`, a dump helper or `ds4-parity/` capture script,
  `ds4-parity/baselines/tokenization/`, `.memory/status.md`.

### M5.3: Rust Vocabulary Loader And JoyAI BPE

- Status: done
- Goal: port tokenizer metadata loading, byte-level GPT-2 encoding/decoding,
  JoyAI pre-tokenization, BPE merge ranking, and plain text tokenization into
  Rust without prompt rendering yet.
- Source evidence needed: `vocab_load`, `byte_encode`,
  `bpe_emit_piece`, `bpe_tokenize_text`, `ds4_token_text`, tokenizer metadata
  in the M4.6 baseline, M5.2 tokenizer identity fields, and M5.2 text-token
  fixtures.
- Oracle: M5.2 C text-tokenization fixture outputs.
- Comparator: C/Rust text tokenization comparison for token IDs and token
  pieces, plus generated negative fixtures for missing token table, missing
  merges, M5.2 token-bytes hash drift, M5.2 merge hash drift, missing required
  special token, invalid UTF-8 token string, and merge-rank drift.
- Acceptance: Rust plain text token IDs and decoded token pieces match C for
  ASCII, whitespace, numbers split into 1-3 digit groups, punctuation/newline
  groups, code snippets, CJK/kana, accented UTF-8, and byte fallback cases.
  This item owns generic `ds4_token_text` byte/UTF-8 piece decoding for ordinary
  tokenizer IDs; rendered special-token literal decoding is owned by M5.4.
- Drift policy: no token ID or decoded-piece drift; any stricter tokenizer
  metadata rejection must be source-proven and named by fixture.
- Review gate: ask Claude to review JoyAI split rules, byte encode/decode, and
  allocation/error boundaries.
- Validation needed: Rust unit tests, C/Rust tokenizer comparator, negative
  tests, `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: `rust/ds4-gguf/` or a new Rust tokenizer crate/module,
  `ds4-parity/`, `.memory/status.md`.

### M5.4: Rust Rendered Chat Special Tokenization

- Status: done
- Goal: port rendered-chat tokenization, special token recognition, and token
  text decoding for the exact `special_token_at` rendered-control table:
  `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<｜User｜>`,
  `<｜Assistant｜>`, `<think>`, `</think>`, and `｜DSML｜`.
- Source evidence needed: `special_token_at`, `tokenize_span`,
  `tokenize_rendered_chat_vocab`, `vocab_token_is_literal_special`,
  `ds4_token_text`, M5.2 rendered-chat fixtures, and M0.4 traces containing
  rendered prompts and token windows.
- Oracle: M5.2 C rendered-chat fixture outputs.
- Comparator: C/Rust rendered-chat comparison for prompt bytes, token IDs, and
  token pieces, including literal special-looking text inside ordinary user
  content vs trusted rendered control text.
- Acceptance: rendered special markers become the exact C special token IDs;
  ordinary spans still use JoyAI BPE; `ds4_token_text`-equivalent decoding
  preserves literal strings for every M5.2
  `vocab_token_is_literal_special` table entry. Generic byte-fallback and UTF-8
  token-piece decoding remains owned by M5.3.
- Drift policy: no special token, DSML marker, or token text drift.
- Review gate: ask Claude to review special-token scanning order and the trust
  boundary between rendered prompts and user-supplied plain content.
- Validation needed: rendered-chat comparator, Rust tests, `cargo test
  --workspace`, and `git diff --check`.
- Owner path: Rust tokenizer module, `ds4-parity/`, `.memory/status.md`.

### M5.5: Prompt Renderer Parity

- Status: done
- Goal: port `render_chat_prompt_text` and CLI chat construction semantics for
  thinking modes, Think Max prefix, system/developer messages, user/tool/function
  messages, assistant content/reasoning, tool schemas, and assistant prefixes.
- Source evidence needed: `render_chat_prompt_text`, `render_live_tool_tail`,
  `append_tools_prompt_text`, `role_is_system`, `role_is_user_like`,
  `ds4_chat_begin`, `ds4_chat_append_message`,
  `ds4_chat_append_assistant_prefix`, CLI `build_prompt`, and M0.4/M0.5
  request/trace fixtures.
- Oracle: M5.2 C prompt fixture outputs and M0.4 server traces.
- Comparator: C/Rust prompt renderer comparison for rendered bytes and
  rendered token IDs for basic chat, stream chat, thinking disabled, thinking
  high, Think Max, system/developer content, multi-turn assistant history,
  tool/function results, and tool schemas before system text; CLI-path fixtures
  compare the direct `ds4_chat_*` append operation sequence and final token
  stream without requiring a synthetic rendered byte stream.
- Acceptance: server rendered prompt bytes and token IDs match for every
  committed server fixture; CLI direct-token prompt construction matches C token
  streams for the committed CLI fixtures; pending assistant prefixes use
  `<think>` or `</think>` exactly as C does. Tool-schema fixtures cover zero,
  one, and multiple tools; absent `tools`/`functions` fields vs explicit empty
  arrays; duplicate tool names; OpenAI `tools` and legacy `functions` inputs;
  missing and empty descriptions; required and optional parameters; nested JSON
  schema fragments; property ordering; and placement before system/developer
  text.
- Drift policy: no server prompt-byte drift; no server or CLI token ID drift;
  any normalized request metadata must be outside the prompt byte/token
  comparison.
- Review gate: ask Claude to review role handling, thinking-mode branches, and
  fixture coverage against server and CLI source.
- Validation needed: prompt comparator, CLI token-stream comparator, Rust
  tests, existing `./ds4_test --server`, CLI prompt construction fixture run,
  `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust prompt module, `ds4-parity/`, `.memory/status.md`.

### M5.6: DSML Formatting And Parse Parity

- Status: split into M5.6a and M5.6b before implementation
- Goal: the original DSML formatting/generated-message/streaming parser item
  was too broad for one reviewable oracle surface. M5.6a owns server DSML
  formatting plus `parse_generated_message_ex`; M5.6b owns `agent_dsml_parse`
  streaming chunk schedules.
- Drift policy: no implementation behavior changed by this split.

### M5.6a: Server DSML Formatting And Generated-Message Parse Parity

- Status: done
- Goal: port server DSML tool-call formatting, raw sampled DSML replay,
  parameter ordering, string/JSON parameter rendering, delimiter escaping, tool
  result escaping, and `parse_generated_message_ex` boundaries.
- Source evidence needed: `append_dsml_tool_calls_text`,
  `append_dsml_arguments_from_json`, `append_dsml_parameter_text`,
  `append_dsml_json_literal`, `append_tool_result_text`,
  `parse_generated_message_ex`, M0.4 tool-call fixture, and server unit tests
  around DSML escaping/replay and generated-message boundaries.
- Oracle: C server DSML formatting/parsing behavior captured by M5.2 fixtures
  and a focused no-model DSML oracle dump.
- Comparator: C/Rust byte comparison for rendered DSML blocks and parsed tool
  call JSON, including raw sampled DSML replay, schema property order, string
  vs JSON parameters, `</｜DSML｜parameter>` escaping, tool-result escaping, and
  DSML before/after `</think>`.
- Acceptance: exact DSML block bytes match. For `parse_generated_message_ex`,
  parsed tool-call names, ids, arguments, order, finish categories, and final
  assistant content match C for every committed fixture, including no promotion
  of pre-`</think>` or malformed executable tool calls when C rejects them.
- Drift policy: no DSML byte drift; no parser broadening that turns ordinary
  prose into executable tool calls; stricter parser errors require fixtures.
- Review gate: ask Claude to review escaping, parameter ordering, raw replay,
  and generated-message parser boundaries.
- Validation needed: DSML comparator, Rust tests, existing `./ds4_test
  --server`, `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust DSML module, `ds4-parity/`, `.memory/status.md`.

### M5.6b: Agent DSML Streaming Parse Parity

- Status: pending
- Goal: port `agent_dsml_parse` streaming behavior for incremental generated
  DSML, including parser states, buffering, emitted calls, and error/truncated
  boundaries.
- Source evidence needed: `agent_dsml_parser`, `agent_dsml_feed`,
  `agent_dsml_parse`, `agent_dsml_close_tag_at`, `agent_dsml_find_close_tag`,
  M5.6a DSML fixtures, and any existing agent call-site tests that consume the
  parser state.
- Oracle: C `ds4_agent.c` streaming parser behavior captured through a
  deterministic no-model oracle dump or fixture runner.
- Comparator: a chunk-split fixture runner replays generated DSML through the C
  parser surface and Rust port under whole-message, one-byte, marker-prefix,
  escaped-delimiter, parameter-boundary, `</think>`, malformed-tag, and
  truncated-at-EOF schedules for unterminated tool-call, invoke, parameter, and
  think blocks.
- Acceptance: streaming state transitions, emitted tool-call events, buffered
  text, error categories, and final parser state match C for every committed
  schedule.
- Drift policy: no streaming parser broadening; partial or malformed DSML must
  stay buffered or rejected wherever C does.
- Review gate: ask Claude to review chunk coverage, EOF semantics, and parser
  state categories for over-broad matching.
- Validation needed: agent DSML comparator, Rust tests,
  `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust agent DSML module, `ds4-parity/`, `.memory/status.md`.

### M5.7: Request Fixture Integration And Text Parity Report

- Status: pending
- Goal: wire tokenization, rendered prompt, and DSML comparators into a single
  Milestone 5 text parity report that runs locally from committed fixtures and
  records exact B300 refresh commands for model-backed recapture.
- Source evidence needed: M5.2 through M5.6b comparators, M0.4/M0.5 request
  fixture manifests, and current unified parity report conventions.
- Oracle: committed M5 token/prompt/DSML fixtures captured from current C.
- Comparator: a report that runs all local text comparators, summarizes fixture
  counts and first drift paths, and skips only model-backed recapture with exact
  B300 rerun commands.
- Acceptance: local static report passes without the model; failure output names
  fixture, field, expected/got, and rerun command where applicable.
- Drift policy: report normalizes only capture paths/timestamps; rendered
  prompt bytes, token IDs, token pieces, DSML bytes, and parsed tool-call
  structures remain exact.
- Review gate: ask Claude to review report integration and failure output.
- Validation needed: text parity report, unified parity report if wired there,
  py_compile, `cargo test --workspace`, and `git diff --check`.
- Owner path: `ds4-parity/`, `.memory/status.md`.

## Later Items

Add later roadmap items from `RUST_PORT_ROADMAP.md` as each active item
completes.
