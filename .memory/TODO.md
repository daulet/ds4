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

- Status: done
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

- Status: done
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

### M6.1: Sampling And Logprob Work Item Breakdown

- Status: done
- Goal: split Milestone 6 sampling and logprob parity into reviewable work
  items before adding Rust sampler or logits-processing code.
- Source evidence needed: `RUST_PORT_ROADMAP.md` Milestone 6, current C
  sampling and logprob paths, official-vector fixtures, server/CLI decode
  selection behavior, stop-token handling, and existing numeric comparator
  conventions.
- Oracle: current C `ds4_session_top_logprobs`, token logprob, decode
  selection, and stop-handling behavior.
- Comparator: documentation-only work item list that defines fixed logits
  fixtures, official-vector prompt cases, top-logprob ordering checks, selected
  token checks, token-byte checks, numeric tolerance, review gates, and
  validation gates.
- Acceptance: the split gives each sampling/logprob item a tangible oracle,
  fixture, comparator, drift policy, and validation gate before implementation
  begins.
- Drift policy: no implementation behavior changes in the split.
- Review gate: ask Claude to review that the split isolates fixture capture,
  Rust sampler logic, logprob/top-k comparison, stop-token behavior, and report
  integration without mixing oracle surfaces.
- Validation needed: `git diff --check`.
- Owner path: `.memory/TODO.md`, `.memory/status.md`,
  `RUST_PORT_ROADMAP.md` source evidence.

### M6.2: C Fixed-Logits Sampling And Logprob Oracle

- Status: done
- Goal: expose current C sampling and logprob math through a deterministic
  no-model oracle dump over fixed logits arrays.
- Source evidence needed: `sample_argmax`, `sample_rng_next`,
  `sample_top_p_min_p`, `ds4_session_top_logprobs`,
  `ds4_session_token_logprob`, public `ds4.h` session API declarations,
  `ds4_cli.c` CLI default resolution, `ds4_server.c` OpenAI, responses, and
  Anthropic request default resolution plus DSML structural-sampling override,
  `ds4_agent.c` agent default resolution, and existing M0.3/M1.4 logprob
  comparator conventions.
- Oracle: current C sampler, RNG, top-logprob, and token-logprob behavior.
- Comparator: schema checker and negative tests for a deterministic fixed-logits
  C oracle dump.
- Fixture: synthetic logits arrays covering greedy ties, non-finite logits,
  temperature normalization, `top_p` clamping, `top_k` caps, `min_p`
  thresholds, full-vocab sampling, seeded RNG draws, top-logprob ordering, and
  per-token logprob requests. The fixture also includes source-named resolved
  request-surface sampling tuples for CLI defaults, OpenAI chat/responses
  defaults, Anthropic defaults, agent defaults, thinking-mode sampling defaults,
  and deterministic structural DSML sampling defaults, recorded as explicit
  `temperature`, `top_k`, `top_p`, `min_p`, and seed inputs where applicable.
- Acceptance: oracle output is deterministic, local, no-model, and records
  selected token, consumed RNG state, filtered candidate set, logits, logprobs,
  and drift paths.
- Drift policy: no sampler or logprob behavior changes; fixture formatting may
  normalize paths and timestamps only.
- Review gate: ask Claude to review fixture coverage against C sampler and
  logprob source.
- Validation needed: C oracle helper build, schema/negative checks, and
  `git diff --check`.
- Owner path: C oracle dump surface, `ds4-parity/`,
  `ds4-parity/baselines/sampling/`, `.memory/status.md`.

### M6.3: Rust Sampler And Logprob Math

- Status: done
- Goal: port C sampler, RNG, top-logprob, and token-logprob math to Rust
  without depending on model execution.
- Source evidence needed: M6.2 fixed-logits C oracle dump,
  `sample_argmax`, `sample_rng_next`, `sample_top_p_min_p`,
  `ds4_session_top_logprobs`, `ds4_session_token_logprob`, and the M6.2
  checker's source-named request-surface parameter tuples.
- Oracle: the M6.2 fixed-logits C oracle dump.
- Comparator: C/Rust comparison for selected token, RNG state, candidate
  filtering, top-logprob ordering, per-token logprob, and numeric tolerance.
- Fixture: the committed M6.2 synthetic logits fixture set.
- Acceptance: greedy choices and sampled choices match exactly for every seeded
  fixture, including source-named resolved request-surface sampling parameter
  tuples; logprob values match within the explicit M6 numeric tolerance.
- Drift policy: no selection drift; any stricter non-finite handling must be
  source-proven and named by fixture.
- Review gate: ask Claude to review Rust numeric edge cases, candidate
  filtering order, RNG semantics, and allocation behavior.
- Validation needed: Rust tests, sampler comparator with negative tests,
  `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: Rust sampler/logprob module, `ds4-parity/`,
  `.memory/status.md`.

### M6.4: Current-C Session Logits And Logprob Fixture Oracle

- Status: done
- Goal: capture model-backed current-C session logits and top-logprob slices
  for official-vector prompt cases without requiring Rust runtime execution.
- Source evidence needed: `RUST_PORT_ROADMAP.md` M6.4, M0.3 official-vector
  prompt cases, `ds4_session_sync`, `ds4_session_argmax`,
  `ds4_session_top_logprobs`, `ds4_session_token_logprob`, and
  `ds4_session_eval` on the B300 q2-imatrix model.
- Oracle: current C session execution on the recorded B300 model.
- Comparator: schema/hash checker that validates the committed model-backed
  logits fixture and records skipped local refresh with exact B300 rerun
  commands.
- Fixture: official-vector prompt cases, selected continuation tokens, logits
  payloads or logits hashes for each scored step, top-logprob slices,
  token-byte renderings, context settings, backend, model identity, and exact
  B300 refresh commands.
- Skips: `long_memory_archive` remains skipped for the existing API/official
  graph mismatch; `long_code_audit` is skipped because repeated B300 CUDA
  captures produce byte-different long-context logits.
- Acceptance: selected greedy tokens match the existing official-vector
  contract, top-logprob slices are deterministic for the recorded backend, and
  the fixture is small enough to commit or explicitly shards large binary
  payloads with hashes.
- Drift policy: model path and capture workspace may be normalized; logits
  bytes or hashes, selected token IDs/bytes, top-logprob order, token bytes,
  backend, and model hash are exact.
- Review gate: ask Claude to review capture schema, artifact size policy, and
  B300 refresh command fidelity.
- Validation needed: B300 capture or existing captured-state check, schema/hash
  checker with negative tests, skipped local refresh evidence, and
  `git diff --check`.
- Owner path: C logits fixture dump surface, `ds4-parity/baselines/sampling/`,
  `.memory/status.md`.

### M6.5: Rust Fixed-Logits Model-Slice Comparator

- Status: done
- Goal: run Rust sampler and logprob math over the M6.4 captured model logits
  slices and compare token presentation against current C.
- Coverage caveat: M6.4 intentionally omits `long_code_audit` logits because
  repeated B300 CUDA long-context captures drift byte-wise. Add a later
  long-context tolerance oracle or CPU-backed reference before claiming full
  official-vector long-context sampler coverage.
- Source evidence needed: M6.4 current-C session logits and top-logprob
  fixture, M6.3 Rust sampler/logprob module, and the tokenizer identity fixture
  already used by Milestone 5.
- Oracle: M6.4 current-C session logits and top-logprob fixture.
- Fixture: committed M6.4 logits payloads plus the tokenizer identity fixture
  already used by Milestone 5.
- Comparator: Rust fixed-logits dump compared to C selected token,
  top-logprob order, logprob values, token IDs, and token bytes.
- Acceptance: Rust chooses the same greedy token for every model-backed step,
  computes the same top-logprob ordering, and renders token bytes identically.
- Drift policy: no token, ordering, or byte drift; numeric differences must
  stay within the M6 tolerance and report max absolute delta.
- Review gate: ask Claude to review fixture loading, token-byte conversion, and
  tolerance reporting.
- Validation passed: `python3 -m py_compile
  ds4-parity/compare_model_logits.py`, `python3
  ds4-parity/compare_model_logits.py --negative-test` (`2982` comparator
  checks, max abs logit/logprob delta about `5.0e-08`; `6` negative checks),
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf --bin
  ds4-model-logits-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: Rust model-slice comparator, `ds4-parity/`,
  `.memory/status.md`.

### M6.6a: Decode Stop Policy C Oracle Fixtures

- Status: done
- Goal: expose request-surface decode stop policy through deterministic
  no-model C fixtures before any Rust policy port.
- Oracle: current C CLI, server, and agent decode-loop behavior around EOS,
  `max_tokens`, user stop sequences, UTF-8 stream-safe holding, and API finish
  reason mapping.
- Fixture: generated-token/text schedules plus request option records for CLI,
  OpenAI chat/responses, Anthropic, and agent defaults. Tool-call schedules
  cover only the decode-loop finish transition after a complete DSML tool-call
  boundary has already been identified.
- Comparator: schema checker and negative tests for a C oracle dump covering
  finish reason, emitted visible text, held streaming tail, session
  invalidation requirement, stop boundary offsets, and API finish mappings.
- Acceptance: EOS, length, stop sequence, UTF-8 boundary, and complete
  tool-call finish outcomes match C for every fixture.
- Drift policy: no finish-reason or emitted-text drift; policy-only
  normalizations must not hide token/text boundary changes.
- Review gate: ask Claude to review stop sequence coverage and the boundary
  between sampler math, M5 DSML parsing, and API finish semantics.
- Validation passed: `arch -arm64 make ds4-decode-policy-dump`, `./ds4-decode-policy-dump
  ds4-parity/baselines/sampling/m6.6a/current-c.json`, `python3 -m
  py_compile ds4-parity/check_decode_policy_dump.py`, `python3
  ds4-parity/check_decode_policy_dump.py --negative-test` (`969` schema
  checks, `5` manifest checks, `10` negative checks), `arch -arm64 make
  ds4_test`, `./ds4_test --server`, and `git diff --check`.
- Owner path: C decode-policy oracle, `ds4-parity/`,
  `.memory/status.md`.

### M6.6b: Rust Decode Stop Policy Port

- Status: done
- Goal: port the request-surface decode stop policy over no-model
  generated-token/text schedules without implementing Rust CLI/server runtime.
- Source evidence needed: M6.6a current-C decode policy dump, C stop-list and
  UTF-8 helper behavior, tool marker boundary behavior, and API finish mapping
  helpers.
- Oracle: M6.6a C decode stop policy oracle dump at
  `ds4-parity/baselines/sampling/m6.6a/current-c.json`.
- Fixture: committed M6.6a stop-policy schedules and request option records.
- Comparator: C/Rust policy comparison for request records, schedules, finish
  reason, emitted raw/visible text, streamed text, held streaming tail, session
  invalidation requirement, stop boundary offsets, tool boundary flags, and
  API finish mappings.
- Acceptance: Rust policy output matches C for every EOS, length, stop
  sequence, UTF-8 boundary, and complete tool-call fixture.
- Drift policy: no finish-reason, emitted-text, held-tail, session
  invalidation, API mapping, or boundary-offset drift.
- Review gate: ask Claude to review the Rust policy boundary and make sure it
  does not reimplement M5 DSML parsing or require model execution.
- Validation passed: `python3 -m py_compile
  ds4-parity/compare_decode_policy.py`, `python3
  ds4-parity/compare_decode_policy.py --negative-test` (`1059` comparator
  checks, `10` negative checks), `cargo fmt --all -- --check`, `cargo test -p
  ds4-gguf decode_policy`, `cargo test -p ds4-gguf --bin
  ds4-decode-policy-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: Rust decode-policy port, `ds4-parity/`,
  `.memory/status.md`.

### M6.7: Sampling And Logprob Report Integration

- Status: done
- Goal: wire M6 local comparators and B300 refresh records into the parity
  reports.
- Source evidence needed: M6.2 through M6.6b manifests, comparator commands,
  current `run_parity_report.py` and text parity report conventions, and B300
  refresh commands for model-backed M6.4 recapture.
- Oracle: committed M6 fixed-logits, model-backed logits, and decode-policy
  fixtures.
- Fixture: M6.2 through M6.6b manifest entries and refresh commands.
- Comparator: a Milestone 6 report that runs all local sampling/logprob
  comparators, summarizes numeric tolerances and first drift paths, and skips
  only model-backed recapture with exact B300 commands; the unified parity
  report includes that M6 report.
- Acceptance: local report passes without the model, JSON output is machine
  readable, failure output names fixture/field/expected/got, and B300
  refreshes are reproducible from the report.
- Drift policy: report normalizes only capture paths and timestamps.
- Review gate: ask Claude to review report integration and failure output.
- Validation passed: `python3 -m py_compile
  ds4-parity/run_sampling_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_sampling_parity_report.py` (`summary: 6 passed, 1
  skipped, 0 failed`), `python3 ds4-parity/run_sampling_parity_report.py
  --json | python3 -m json.tool`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` (`summary: 7 passed,
  10 skipped, 0 failed`), `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles --json | python3 -m json.tool`, `cargo test
  --workspace`, and `git diff --check`.
- Owner path: `ds4-parity/run_sampling_parity_report.py`,
  `ds4-parity/run_parity_report.py`, `ds4-parity/README.md`,
  `.memory/status.md`.

### M7.1: KV Store Work Item Breakdown

- Status: done
- Goal: split Milestone 7 KV store and snapshot parity into reviewable work
  items before adding Rust persistence code.
- Source evidence needed: `RUST_PORT_ROADMAP.md` Milestone 7,
  `ds4_kvstore.c`, `ds4_kvstore.h`, session snapshot payload code in
  `ds4.c`, M0.5 KV artifacts, server tool-map trailer code, in-memory
  snapshot APIs, and existing report conventions.
- Oracle: current `ds4_kvstore` and session snapshot implementation plus
  committed M0.5 KV/cache artifacts.
- Fixture: M0.5 `kv-artifacts`, current `ds4_kvstore` source, current session
  payload source, in-memory snapshot API surface, trailer-hook source, and
  existing parity-report conventions.
- Comparator: documentation-only work item list that defines header/policy,
  Rust parser, generic full-file round trips, per-extension trailer coverage,
  payload shape, Rust payload reader, request replay, B300 disk and in-memory
  restore, and report integration comparison contracts for later executable
  items. M7.2 is the first open item; M7.5 is independently eligible because
  it depends on session payload code rather than KV header/policy work.
- Acceptance: the split gives each KV/snapshot item a tangible oracle, fixture,
  comparator, drift policy, review gate, and validation gate before
  implementation begins.
- Drift policy: no implementation behavior changes in the split.
- Review gate: ask Claude to review that the split isolates header/policy
  fixtures, Rust format parsing, generic full-file round trips, per-extension
  trailer coverage, on-disk payload structure, in-memory snapshot restore,
  request replay, B300 recapture, and report integration without mixing oracle
  surfaces.
- Validation needed: `git diff --check`.
- Owner path: `.memory/TODO.md`, `.memory/status.md`,
  `RUST_PORT_ROADMAP.md` source evidence.

### M7.2: C KV Header And Policy Oracle

- Status: done
- Goal: expose current C KV-cache header, filename, and policy behavior through
  a deterministic no-model oracle dump.
- Source evidence needed: `ds4_kvstore.c`, `ds4_kvstore.h`, `ds4_server.c`
  tool-map trailer constants where extension flags matter, M0.5 parsed header
  rows, rendered cache text, and cache-decision logs.
- Oracle: current C no-model KV helpers: KVC header layout,
  `ds4_kvstore_fill_header`, `ds4_kvstore_read_header`,
  `ds4_kvstore_read_entry_file`, default options, reason/key-kind mapping,
  SHA/path helpers, store-boundary selection, chat-anchor selection,
  continued-store target selection, file-size budgeting, byte-prefix matching,
  eviction scoring, and text-prefix entry selection. Model/session-bound token
  rendering, live-prefix storage, continued storage, and `try_load_text` are
  out of scope for this no-model item.
- Comparator: schema checker and negative tests for a deterministic C oracle
  dump covering exact header bytes, decoded fields, selected path names,
  policy outputs, and first-failure paths.
- Fixture: synthetic text bytes, token IDs, cache entries, timestamps, file
  sizes, option records, explicit `now` values for eviction scoring, and
  committed M0.5 parsed header rows.
- Acceptance: oracle output is deterministic, local, no-model, and captures
  the current KVC header bytes, field decoding, SHA keying, prefix selection,
  eviction ordering, store target decisions, and boundary edge cases.
- Drift policy: no KV policy or format behavior changes; fixture formatting may
  normalize paths and timestamps only.
- Review gate: ask Claude to review fixture coverage against `ds4_kvstore`
  source and M0.5 artifacts.
- Validation passed: `arch -arm64 make ds4-kv-policy-dump`,
  `./ds4-kv-policy-dump ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m py_compile ds4-parity/check_kv_policy_dump.py`, and
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema
  checks, `11` manifest checks, `7` negative checks). Existing build/test
  surface validation also passed and is listed in `.memory/status.md`.
- Owner path: C oracle dump surface, `ds4-parity/`,
  `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.3: Rust KV Header And Policy Parser

- Status: done
- Goal: port the KVC header parser/writer and no-model KV policy decisions to
  Rust without loading model sessions.
- Source evidence needed: M7.2 current-C KV policy oracle dump,
  `ds4_kvstore.c`, `ds4_kvstore.h`, committed M0.5 header/rendered-text/cache
  artifacts, and existing Rust parity crate conventions.
- Oracle: M7.2 current-C KV header and policy oracle dump at
  `ds4-parity/baselines/kv/m7.2/current-c.json`.
- Fixture: committed M7.2 synthetic dump plus M0.5 header, rendered-text, and
  cache-decision artifacts.
- Comparator: C/Rust comparison for header bytes, decoded fields, reason and
  extension flags, SHA file names, file-size budgeting, prefix selection,
  store-boundary selection, continued-store targets, and eviction ordering.
- Acceptance: Rust parses existing KVC headers, writes byte-identical headers
  for every fixture, and matches C policy decisions for every no-model case.
- Drift policy: no header-byte, keying, selection, or eviction drift; a
  versioned format change is not allowed in this item.
- Review gate: ask Claude to review byte-order handling, integer overflow
  checks, timestamp normalization boundaries, and policy tie-break behavior.
- Validation passed: `python3 -m py_compile
  ds4-parity/compare_kv_policy.py`, `python3
  ds4-parity/compare_kv_policy.py --negative-test` (`KV policy C/Rust
  comparator: PASS, 1488 checks`; negative tests `PASS, 8 checks`), `python3
  ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema checks,
  `11` manifest checks, `7` negative checks), `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-policy-dump-rs`, `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust KV policy module, `ds4-parity/`,
  `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.4a: Generic KVC Full-File Round Trip

- Status: done
- Goal: compare full KVC file construction, generic optional trailer bytes,
  file-size budgeting, and cross-reader acceptance without restoring model
  tensors.
- Source evidence needed: M7.3 Rust KVC header writer and no-model policy
  comparator, `ds4_kvstore.c`, `ds4_kvstore.h`,
  `ds4_kvstore_trailer_hooks`, current C entry-file reader behavior, and M0.5
  KVC artifact records.
- Oracle: current C fixed-header/text/payload file layout,
  `ds4_kvstore_trailer_hooks`, current C entry-file reader behavior, and C
  `ds4_kvstore_file_size_fits` for the produced file sizes.
- Fixture: synthetic cache text, opaque payload bytes, fixed timestamps,
  option records, generic extension-flag combinations, opaque trailer bytes,
  and truncated/corrupted header, text, payload, and trailer data.
- Comparator: C writer versus Rust writer byte comparison for the complete
  KVC file, Rust reader acceptance of C-written files, C reader acceptance of
  Rust-written files, and negative tests for malformed header/text/payload and
  trailer boundaries.
- Acceptance: full files are byte-identical for the fixed-header, text,
  payload, and trailer fixture; C can read Rust-written metadata/trailer files;
  Rust can read C-written metadata/trailer files; malformed files fail at the
  same boundary category; Rust writer output size equals the C policy
  `file_size_fits` budget input for each fixture.
- Drift policy: no KVC full-file byte drift, extension-flag drift,
  trailer-size drift, or cross-reader acceptance drift; opaque payload bytes
  remain uninterpreted in this item.
- Review gate: ask Claude to review generic trailer-hook coverage, full-file
  byte identity, file-size budget cross-checks, and C-reads-Rust/Rust-reads-C
  round-trip evidence.
- Validation passed: `arch -arm64 make ds4-kvc-file-dump`,
  `./ds4-kvc-file-dump ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kvc_file.py`, `python3
  ds4-parity/compare_kvc_file.py --negative-test` (`KVC file C/Rust
  comparator: PASS, 277 checks`; negative tests `PASS, 8 checks`),
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf kvc`,
  `cargo test -p ds4-gguf --bin ds4-kvc-file-dump-rs`,
  `cargo test --workspace`, `arch -arm64 make cpu`, CPU-regenerated
  `./ds4-kvc-file-dump` comparison against the committed M7.4a artifact, and
  `git diff --check`.
- Owner path: C/Rust KVC file fixture helpers, `ds4-parity/`,
  `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.4b: KV Extension Trailer Payload Coverage

- Status: done
- Goal: compare server-owned KVC extension payloads and extension-flag
  semantics separately from the generic full-file round trip.
- Source evidence needed: server tool-map trailer format (`KTM` version 1),
  `DS4_KVSTORE_EXT_TOOL_MAP`, `DS4_KVSTORE_EXT_RESPONSES_VISIBLE`,
  `DS4_KVSTORE_EXT_THINKING_VISIBLE`, current C trailer write/load helper
  behavior, M7.4a generic full-file comparator, and server protocol fixture
  conventions.
- Oracle: server tool-map trailer format (`KTM` version 1),
  `DS4_KVSTORE_EXT_TOOL_MAP`, `DS4_KVSTORE_EXT_RESPONSES_VISIBLE`,
  `DS4_KVSTORE_EXT_THINKING_VISIBLE`, and current C trailer write/load helper
  behavior.
- Fixture: tool-map trailer entries with boundary cases for zero entries,
  multiple entries, UTF-8 bytes, long IDs, long DSML records,
  duplicate-shaped entries, visible-transcript extension flags without payload
  bytes, and truncated/corrupted trailer data.
- Comparator: C/Rust comparison for extension flags, serialized trailer byte
  size, trailer bytes, decoded tool-map entries, visible-transcript flag
  handling, and malformed trailer rejection.
- Acceptance: Rust emits and decodes the same extension flags and tool-map
  trailer bytes as C, visible transcript flags do not imply extra payload
  bytes, and malformed extension data fails at the same boundary category.
- Drift policy: no extension-flag, trailer-byte, decoded-entry, or malformed
  rejection drift.
- Review gate: ask Claude to review per-extension payload coverage and make
  sure server-owned trailer semantics are not hidden inside generic KVC
  parsing.
- Validation passed: `arch -arm64 make ds4-kv-trailer-dump`,
  `./ds4-kv-trailer-dump ds4-parity/baselines/kv/m7.4b/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4b/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kv_trailer.py`, `python3
  ds4-parity/compare_kv_trailer.py --negative-test` (`KV trailer C/Rust
  comparator: PASS, 432 checks`; negative tests `PASS, 8 checks`),
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf tool_map`,
  `cargo test -p ds4-gguf --bin ds4-kv-trailer-dump-rs`,
  `cargo test --workspace`, `arch -arm64 make cpu`, CPU-regenerated
  `./ds4-kv-trailer-dump` comparison against the committed M7.4b artifact,
  and `git diff --check`.
- Owner path: server trailer fixture helpers, Rust KVC extension parser,
  `ds4-parity/`, `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.5: C Session Payload Shape Oracle

- Status: done
- Goal: expose current C session payload structure, size budgeting, and
  on-disk payload-header rejection behavior before any Rust payload reader.
- Source evidence needed: `DS4_SESSION_PAYLOAD_MAGIC`,
  `DS4_SESSION_PAYLOAD_VERSION`, `DS4_SESSION_PAYLOAD_U32_FIELDS`,
  `ds4_session_payload_bytes`, `ds4_session_save_payload`,
  `ds4_session_load_payload`, current fixed model-layout constants, M0.5
  payload-size/hash records, and B300 refresh commands.
- Oracle: current C on-disk DSV4 payload format and load/save rejection
  behavior. In-memory `ds4_session_save_snapshot` and
  `ds4_session_load_snapshot` are excluded from this on-disk payload item.
- Fixture: deterministic no-model structural records for payload constants and
  rejection cases, frozen current-C model-layout constants (`DS4_N_LAYER`,
  `DS4_N_HEAD_DIM`, `DS4_N_INDEXER_HEAD_DIM`, and `DS4_N_VOCAB`), M0.5
  payload-size/hash records, and exact B300 refresh commands for model-backed
  payload captures.
- Comparator: schema/hash checker that validates payload metadata, fixed header
  fields, size calculations, rejection categories, and skipped B300 recapture
  commands.
- Acceptance: the committed fixture records the DSV4 payload contract, payload
  size inputs, rejection cases, and model-backed hash records needed for a Rust
  reader. Raw payloads larger than 1 MiB are represented by hashes plus exact
  recapture commands instead of being committed.
- Drift policy: no payload format or acceptance-rule drift; model path and
  capture workspace may be normalized.
- Review gate: ask Claude to review that payload-shape evidence is sufficient
  for a Rust reader while avoiding a premature Rust runtime/session port.
- Validation: `arch -arm64 make ds4-session-payload-dump`,
  `./ds4-session-payload-dump | python3 -m json.tool`,
  `python3 ds4-parity/check_session_payload_shape.py --write-baseline
  ds4-parity/baselines/kv/m7.5/current-c.json`, `python3 -m json.tool
  ds4-parity/baselines/kv/m7.5/current-c.json`, `python3 -m py_compile
  ds4-parity/check_session_payload_shape.py`, and `python3
  ds4-parity/check_session_payload_shape.py --negative-test`. `arch -arm64
  make cpu` and `git diff --check` also passed.
- Owner path: C payload oracle surface, `ds4-parity/`,
  `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.6: Rust Session Payload Header Reader

- Status: done
- Goal: add a Rust reader for DSV4 session payload headers and structural
  validation without restoring tensors or executing a model.
- Oracle: the M7.5 current-C on-disk session payload shape oracle.
- Fixture: committed M7.5 payload metadata, rejection fixtures, frozen current-C
  model-layout constants, and M0.5 payload-size/hash records.
- Comparator: C/Rust comparison for payload magic, version, field decoding,
  model-layout constant checks, payload-size accounting, trailing-byte
  rejection, and malformed-header rejection.
- Acceptance: Rust decodes current payload metadata, reports the same
  structural rejection categories as C, and never claims tensor restore or
  in-memory snapshot support.
- Drift policy: no payload-header or size-accounting drift; body tensor
  interpretation remains out of scope.
- Review gate: ask Claude to review format-boundary checks, checked arithmetic,
  and the no-runtime-restore scope boundary.
- Validation: `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_session_payload.py`, `python3
  ds4-parity/compare_session_payload.py --negative-test`, `cargo test -p
  ds4-gguf session_payload`, `cargo test -p ds4-gguf --bin
  ds4-session-payload-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: Rust session payload reader, `ds4-parity/`,
  `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.7: KV Replay And Prefix Decision Comparator

- Status: done
- Goal: compare request-level cache hit, cache miss, prefix match, exact DSML
  replay, and effective prompt suffix construction against current C artifacts.
- Oracle: M0.5 KV/cache traces, M0.4 server traces where DSML rendering is
  involved, current `ds4_kvstore_try_load_text`, and current prompt rendering
  behavior already covered by Milestone 5.
- Fixture: committed cache request JSON, rendered cached text, cache-decision
  logs, token prefix records, DSML tool-call records, hashes of the Milestone 5
  prompt-rendering artifacts consumed as opaque fixture inputs, and M7.3 Rust
  KV policy outputs.
- Comparator: C/log-artifact versus Rust comparison for `cache_source`,
  cached-token counts, cache-write-token counts, reason codes, key kind,
  extension flags, rendered text SHA, exact prefix token records, and effective
  prompt suffix bytes. M5 artifact-hash drift is reported as a fixture
  precondition failure, not as KV behavior drift.
- Acceptance: Rust makes the same cache hit/miss and prefix decisions for the
  committed replay cases, including DSML-visible transcript boundaries and text
  suffix construction, using M5 outputs as fixed inputs. M7.7 only uses replay
  prompts already covered by committed M5 artifacts; extending M5 rendering
  coverage is M5 work, not M7.7 work.
- Drift policy: no cache-decision, rendered-text, token-prefix, or suffix-byte
  drift; trace timing and process paths may be normalized.
- Review gate: ask Claude to review replay coverage at the boundary between KV
  policy, opaque Milestone 5 text fixtures, and future server runtime work.
- Validation passed: `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_kv_replay.py`, `python3 -m json.tool
  ds4-parity/baselines/kv/m7.7/current-c.json`, `python3 -m json.tool
  ds4-parity/baselines/kv/m7.7/manifest.json`, `python3
  ds4-parity/compare_kv_replay.py --negative-test` (`KV replay C fixture
  preconditions: PASS, 333 checks`; `KV replay C/Rust comparator: PASS, 273
  checks`; `KV replay Rust policy precondition: PASS, 14 checks`; manifest
  `PASS, 6 checks`; negative tests `PASS, 6 checks`), `cargo test -p
  ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-replay-dump-rs`, `cargo test --workspace`, and `git diff --check`.
- Owner path: KV replay comparator, `ds4-parity/`, Rust KV policy helpers,
  `.memory/status.md`.

### M7.8: B300 Disk KV And In-Memory Snapshot Restore Oracle

- Status: done
- Goal: capture model-backed current-C evidence for both disk KV/session
  payload restore and in-memory `ds4_session_snapshot` restore.
- Source evidence needed: current C server/session save and restore paths on
  the recorded B300 model, `ds4_session_save_payload`,
  `ds4_session_load_payload`, `ds4_session_save_snapshot`,
  `ds4_session_load_snapshot`, `ds4_session_top_logprobs`, and selected-token
  output after restore.
- Oracle: current C server/session save and restore paths on the recorded B300
  model.
- Fixture: fixed cache seed prompts, continuation prompts, cache directory
  setup, selected disk restore points, selected in-memory snapshot points, raw
  KV/payload hash records, snapshot metadata records, top-logprob slices,
  selected tokens, model identity, backend identity, and exact B300 refresh
  commands. Rust in-memory snapshot loading is deferred to Milestone 10 runtime
  ownership; this item captures current-C oracle evidence only.
- Comparator: schema/hash/top-logprob checker that validates committed restore
  records locally and skips recapture only with exact B300 rerun commands.
- Acceptance: restored and uninterrupted current-C sessions have matching
  selected next token and top-logprob token ordering; logit/logprob score
  values match within the M6 model-logits absolute tolerance of `1e-5` from
  `ds4-parity/compare_model_logits.py`. Raw payloads larger than 1 MiB are
  represented by hashes plus exact recapture commands instead of being
  committed.
- Drift policy: model path and capture workspace may be normalized; prompt
  bytes, restore metadata, snapshot metadata, payload hashes, selected tokens,
  top-logprob token IDs/order, and score tolerances are exact.
- Review gate: ask Claude to review B300 command fidelity, artifact-size
  policy, disk-vs-memory snapshot separation, and distribution-comparison
  tolerance.
- Validation passed: B300 `ds4-rust-port-b300` capture built
  `ds4-restore-dump` with `CUDA_ARCH=native`, captured
  `ds4-parity/baselines/kv/m7.8/current-c.json`, and passed
  `python3 ds4-parity/check_restore_dump.py
  ds4-parity/baselines/kv/m7.8/current-c.json --negative-test` on the pod
  (`restore oracle schema: PASS, 1448 checks`; negative tests `PASS, 6
  checks`). Local validation passed for `arch -arm64 make ds4-restore-dump`,
  `python3 -m py_compile ds4-parity/check_restore_dump.py`, manifest generation
  via `--write-manifest`, `python3 ds4-parity/check_restore_dump.py
  ds4-parity/baselines/kv/m7.8/current-c.json --manifest
  ds4-parity/baselines/kv/m7.8/manifest.json --negative-test` (`restore
  oracle schema: PASS, 1448 checks`; manifest `PASS, 13 checks`; negative tests
  `PASS, 6 checks`), `python3 -m json.tool` for `current-c.json` and
  `manifest.json`, `arch -arm64 make cpu`, and `git diff --check`.
- Owner path: B300 restore oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/kv/`, `.memory/status.md`.

### M7.9: KV And Snapshot Report Integration

- Status: done
- Goal: wire M7 local comparators and B300 restore refresh records into the
  parity reports.
- Source evidence needed: committed M7.2 through M7.8 fixtures, manifest
  entries, local comparator commands, M0.5 baseline artifacts, and B300 restore
  recapture records.
- Oracle: committed M7.2 through M7.8 fixtures and refresh commands.
- Fixture: M7 manifest entries, local comparator commands, M0.5 baseline
  artifacts, and B300 restore recapture records.
- Comparator: a Milestone 7 report that runs all local KV/snapshot comparators,
  summarizes first drift paths, and skips only model-backed B300 recapture with
  exact commands; the unified parity report includes that M7 report.
- Acceptance: local report passes without the model, JSON output is machine
  readable, failure output names fixture/field/expected/got, and B300 refreshes
  are reproducible from the report.
- Drift policy: report normalizes only capture paths and timestamps.
- Review gate: ask Claude to review report integration and skipped-B300 command
  fidelity.
- Validation passed: `python3 -m py_compile
  ds4-parity/run_kv_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_kv_parity_report.py` (`summary: 9 passed, 1
  skipped, 0 failed`), `python3 ds4-parity/run_kv_parity_report.py --json |
  python3 -m json.tool >/dev/null`, `python3 ds4-parity/run_parity_report.py`
  (`summary: 13 passed, 5 skipped, 0 failed`), `cargo test --workspace`, and
  `git diff --check`.
- Owner path: parity report integration, `ds4-parity/`, `.memory/status.md`.

### M8.1: CLI Surface Work Item Breakdown

- Status: done
- Goal: split Milestone 8 into commit-sized CLI parity work items before
  adding Rust CLI behavior.
- Source evidence needed: Milestone 8 roadmap text, current `ds4_cli.c`, CLI
  targets in `Makefile`, existing CLI/test-vector fixtures, and current
  parity-report conventions.
- Oracle: current `./ds4` binary and its documented CLI behavior.
- Fixture: help/flag surface, one-shot prompt, prompt-file input, stdin
  transcript, logprob dump mode, thinking controls, output formatting, and
  stderr/error categories to be assigned to later executable milestones.
- Comparator: documentation-only breakdown with a tangible oracle, fixture,
  comparator, acceptance rule, and validation gate per CLI work item.
- Acceptance: `RUST_PORT_ROADMAP.md` and the active board name executable CLI
  milestones that can be reviewed and compared independently; no source or
  build behavior changes.
- Drift policy: no behavior drift; roadmap wording may normalize names and
  fixture paths.
- Review gate: ask Claude to review whether the split makes each CLI milestone
  verifiable and comparable to current C.
- Validation passed: inspected `ds4_cli.c` usage, generation, diagnostics,
  parser, inspect, imatrix, and REPL surfaces; updated
  `RUST_PORT_ROADMAP.md` with M8.2 through M8.16 work items;
  `git diff --check` passed.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/status.md`, `.memory/TODO.md`.

### M8.2: Current-C CLI Parse And Error Oracle

- Status: done
- Goal: capture the no-model CLI argument, help, and early error surface.
- Source evidence needed: current `ds4_cli.c` parser, `usage()` text, `main()`
  early exits, local `ds4` build target, and parse/error branches that exit
  before model loading.
- Oracle: current `./ds4` parser in `ds4_cli.c` before model loading.
- Fixture: `--help`, missing option values, unknown options, invalid numeric
  and float values, invalid backend names, duplicate prompt sources, `--server`,
  removed `--metal-graph-generate`, `--dump-tokens` without a prompt, imatrix
  option coupling, and `--perplexity-file` prompt-source rejection.
- Comparator: schema checker for exit status, stdout/stderr category, help text
  anchors, and exact option names.
- Acceptance: all cases are local and model-free; exit code and stderr category
  match exactly, with help text compared by stable section anchors.
- Drift policy: executable path and compiler diagnostics may be normalized; CLI
  option spelling, exit status, and user-facing error category are exact.
- Review gate: ask Claude to review coverage for parser branches and accidental
  model-loading cases.
- Validation passed: `arch -arm64 make ds4`, baseline generation with
  `python3 ds4-parity/check_cli_parse_dump.py --write-baseline
  ds4-parity/baselines/cli/m8.2/current-c.json --write-manifest
  ds4-parity/baselines/cli/m8.2/manifest.json`, `python3 -m py_compile
  ds4-parity/check_cli_parse_dump.py`, `python3
  ds4-parity/check_cli_parse_dump.py
  ds4-parity/baselines/cli/m8.2/current-c.json --manifest
  ds4-parity/baselines/cli/m8.2/manifest.json --negative-test` (`CLI parse
  oracle: PASS, 369 checks`; manifest `PASS, 11 checks`; negative tests `PASS,
  7 checks`), `python3 -m json.tool` for both M8.2 JSON files, and
  `git diff --check`.
- Owner path: CLI parse/error oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.3: Rust CLI Parse And Error Parity

- Status: done
- Goal: implement Rust CLI parsing for the M8.2 no-model surface.
- Source evidence needed: committed M8.2 current-C CLI parse/error fixture,
  existing Rust workspace layout, current `ds4_cli.c` parser, and CLI binary
  naming/build conventions.
- Oracle: committed M8.2 current-C CLI parse/error fixture.
- Fixture: same argument matrix as M8.2, run against the Rust CLI binary.
- Comparator: C/Rust CLI parser comparator for exit status, stdout/stderr
  category, help anchors, normalized executable names, and option spelling.
- Acceptance: Rust exits with the same status and reports the same category and
  option names without loading a model for early parse failures.
- Drift policy: binary path and usage indentation may be normalized only where
  documented by the comparator.
- Review gate: ask Claude to review parser compatibility and whether any C
  parser branch remains uncovered.
- Validation passed: `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_cli_parse.py`, `cargo test -p ds4-gguf cli_parse`
  (3 parser tests passed), `python3 ds4-parity/compare_cli_parse.py
  --negative-test` (`CLI parse C fixture preconditions: PASS, 224 checks`;
  `CLI parse C/Rust comparator: PASS, 244 checks`; negative tests `PASS, 5
  checks`), `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust CLI parser, CLI parse comparator, `ds4-parity/`,
  `.memory/status.md`.

### M8.4: Current-C CLI Token And Prompt Diagnostic Oracle

- Status: done
- Goal: capture current-C CLI prompt ingestion and token-dump behavior.
- Source evidence used: `ds4_cli.c` prompt-source handling, `--dump-tokens`
  path, thinking controls, prompt rendering/tokenizer milestones, and B300
  model/tokenizer availability.
- Oracle: current `./ds4 --dump-tokens` with the recorded B300 model/tokenizer.
- Fixture: `-p`, `--prompt-file`, rendered-chat prompt passthrough, custom
  system prompt, empty system prompt, and `--think`/`--think-max`/`--nothink`
  cases proving those controls are ignored by the early `--dump-tokens` exit.
- Comparator: schema/hash checker for token IDs, raw stdout token bytes,
  prompt-file byte hashes, empty warning categories, and exact B300 refresh
  commands.
- Acceptance: prompt bytes, raw token stdout, token sequence, and no-warning
  category match the current CLI fixture; system/thinking controls remain
  byte-identical to the base prompt because `--dump-tokens` exits before
  `build_prompt` and `cli_warn_think_max_downgraded`.
- Drift policy: model path and B300 workspace may be normalized; prompt bytes,
  token IDs, token bytes, thinking controls, and warning categories are exact.
- Review gate: ask Claude to review prompt-source and thinking-control coverage
  against `ds4_cli.c`.
- Validation passed: B300 capture on `ds4-rust-port-b300` after `make ds4
  CUDA_ARCH=native`, `python3 ds4-parity/check_cli_token_dump.py
  ds4-parity/baselines/cli/m8.4/current-c.json --manifest
  ds4-parity/baselines/cli/m8.4/manifest.json --negative-test` (`CLI token dump
  oracle: PASS, 306 checks`; manifest `PASS, 18 checks`; negative tests `PASS,
  8 checks`), `python3 -m py_compile ds4-parity/check_cli_token_dump.py`,
  `python3 -m json.tool ds4-parity/baselines/cli/m8.4/current-c.json`, and
  `git diff --check`.
- Owner path: CLI token/prompt oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.5: Rust CLI Token And Prompt Diagnostic Parity

- Status: done
- Goal: implement Rust CLI behavior for `--dump-tokens` and prompt-source
  diagnostics.
- Source evidence used: committed M8.4 current-C token/prompt diagnostic
  fixture, M8.3 Rust CLI parser, Rust tokenizer text path, and `ds4_cli.c`
  dump-token early-exit behavior.
- Oracle: committed M8.4 current-C token/prompt diagnostic fixture.
- Fixture: same prompt-source cases and ignored system/thinking-control cases
  as M8.4, run through the Rust CLI.
- Comparator: C/Rust diagnostic comparator for token IDs, token bytes,
  prompt-file hashes, stdout shape, stderr warning categories, and exit status.
- Acceptance: Rust matches the C CLI for prompt ingestion and dump-token output
  without introducing alternate formatting or applying system/thinking controls
  in the `--dump-tokens` path.
- Drift policy: executable paths and timing-free stderr prefixes may be
  normalized; token and prompt surfaces are exact.
- Review gate: ask Claude to review CLI-to-tokenizer plumbing and normalization
  boundaries.
- Validation passed: `cargo fmt --all -- --check`, `cargo test -p ds4-gguf
  cli_parse`, `cargo test -p ds4-gguf token_text_decodes_gpt2_byte_mapping`,
  `python3 -m py_compile ds4-parity/compare_cli_token_dump.py`, `python3
  ds4-parity/compare_cli_token_dump.py --skip-build --negative-test` (`CLI
  token dump tokenizer fixture: PASS, 3 checks`; C fixture preconditions `PASS,
  166 checks`; C/Rust comparator `PASS, 65 checks`; negative tests `PASS, 5
  checks`), `python3 ds4-parity/compare_cli_parse.py --skip-build
  --negative-test` (`CLI parse C fixture preconditions: PASS, 224 checks`;
  C/Rust comparator `PASS, 244 checks`; negative tests `PASS, 5 checks`),
  `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust CLI token dump path, CLI token comparator, `ds4-parity/`,
  `.memory/status.md`.

### M8.6: Current-C CLI Logprob And Perplexity Oracle

- Status: done
- Goal: capture current-C CLI machine-readable diagnostic outputs that require
  model execution.
- Source evidence used: `ds4_cli.c` `--dump-logprobs` and
  `--perplexity-file` dispatch, M6 score-tolerance policy, prompt-file handling,
  and B300 model/backend availability.
- Oracle: current `./ds4 --dump-logprobs` and `./ds4 --perplexity-file` on the
  recorded B300 model.
- Fixture: fixed short prompt, prompt-file variant, `--logprobs-top-k`, greedy
  token limit, invalid output path category, and a fixed raw-text perplexity
  file.
- Comparator: schema/numeric checker for JSON logprob shape, selected tokens,
  top-logprob ordering, score tolerances from M6, perplexity text fields, file
  hashes, and exact B300 refresh commands.
- Acceptance: selected tokens and top-logprob ordering match exactly; score
  values stay within the M6 model-logits tolerance; perplexity scalar fields
  match within documented numeric tolerances.
- Drift policy: timing/progress stderr and workspace paths may be normalized;
  CLI JSON fields, selected tokens, score tolerances, and output file hashes are
  exact.
- Review gate: ask Claude to review diagnostic output coverage and numeric
  tolerance reuse from M6.
- Validation passed: B300 capture on `ds4-rust-port-b300` after `make ds4
  CUDA_ARCH=native`, `python3 ds4-parity/check_cli_diagnostics_dump.py
  ds4-parity/baselines/cli/m8.6/current-c.json --manifest
  ds4-parity/baselines/cli/m8.6/manifest.json --negative-test` (`CLI
  diagnostics oracle: PASS, 267 checks`; manifest `PASS, 12 checks`; negative
  tests `PASS, 7 checks`), local revalidation with the same PASS counts,
  `python3 -m py_compile ds4-parity/check_cli_diagnostics_dump.py`, `python3
  -m json.tool` for both M8.6 JSON files, and `git diff --check`.
- Owner path: CLI diagnostic oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.7: Rust CLI Logprob And Perplexity Parity

- Status: split into M8.7a and M8.7b before implementation
- Goal: the original Rust CLI logprob/perplexity parity item required
  model/session execution that the Rust tree does not yet expose. M8.7a owns
  the Rust diagnostic runtime boundary prerequisite; M8.7b owns the CLI
  diagnostic output surface that runs on top of it. See `RUST_PORT_ROADMAP.md`
  M8.7/M8.7a/M8.7b.
- Drift policy: no source behavior changed by this split; no replay-only proxy
  is accepted as execution parity.
- Owner path: Rust CLI diagnostic paths, CLI diagnostic comparator,
  `ds4-parity/`, `.memory/status.md`.

### M8.8: Current-C CLI Inspect Output Oracle

- Status: done
- Goal: capture the current-C `--inspect` CLI output surface.
- Source evidence used: `ds4_cli.c` `--inspect` dispatch, `ds4_engine_summary`
  output, model/backend identity, and B300 model availability.
- Oracle: current `./ds4 --inspect` on the recorded B300 model.
- Fixture: model path, backend selection, summary stdout/stderr records, model
  identity, prompt/control-ignored inspect case, exit status, and exact B300
  refresh commands.
- Comparator: schema/hash checker for summary output anchors, model/backend
  identity, exit status, and refresh commands.
- Acceptance: summary output anchors and model identity match current C; no
  generation, REPL, perplexity, or imatrix path is entered.
- Drift policy: workspace paths and volatile memory addresses may be
  normalized; model identity, summary sections, and exit status are exact.
- Review gate: ask Claude to review inspect-output coverage against
  `ds4_engine_summary` dispatch.
- Validation passed: B300 capture on `ds4-rust-port-b300` after `make ds4
  CUDA_ARCH=native`, `python3 ds4-parity/check_cli_inspect_dump.py
  ds4-parity/baselines/cli/m8.8/current-c.json --manifest
  ds4-parity/baselines/cli/m8.8/manifest.json --negative-test` (`CLI inspect
  oracle: PASS, 112 checks`; manifest `PASS, 20 checks`; negative tests `PASS,
  8 checks`), local revalidation with the same PASS counts, `python3 -m
  py_compile ds4-parity/check_cli_inspect_dump.py`, `python3 -m json.tool` for
  both M8.8 JSON files, and `git diff --check`.
- Owner path: CLI inspect oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.9: Rust CLI Inspect Output Parity

- Status: split into M8.9a and M8.9b before implementation
- Goal: the original Rust CLI inspect parity item required an engine-open and
  engine-summary boundary that the Rust tree does not yet expose. M8.9a owns
  the Rust inspect runtime boundary prerequisite; M8.9b owns the CLI inspect
  output surface that runs on top of it. See `RUST_PORT_ROADMAP.md`
  M8.9/M8.9a/M8.9b.
- Drift policy: no source behavior changed by this split; no fake summary or
  artifact replay is accepted as execution parity.
- Owner path: Rust CLI inspect path, CLI inspect comparator, `ds4-parity/`,
  `.memory/status.md`.

### M8.9a: Rust Inspect Runtime Boundary Prerequisite

- Status: done
- Goal: introduce or expose a Rust-accessible engine-open and engine-summary
  boundary sufficient to load the M8.8 model/backend and emit the same summary
  surface without entering generation, REPL, perplexity, or imatrix paths.
- Source evidence used: C `ds4_engine_open`/`ds4_engine_summary`, committed
  M8.8 inspect fixture, Rust FFI/build boundary, Rust `Engine` lifecycle
  wrapper, and B300 CUDA runtime validation.
- Oracle: current C `ds4_engine_open` and `ds4_engine_summary` on the M8.8
  inspect fixture.
- Fixture: the same plain inspect and prompt/control inspect cases captured by
  M8.8.
- Comparator: C/Rust runtime-boundary comparator for exit status, parsed summary
  fields, stdout identity, backend identity anchors, and forbidden-path stderr
  markers.
- Acceptance: Rust can execute the inspect runtime boundary against the same
  model/backend and match current-C summary fields and dispatch exclusivity.
- Drift policy: path, backend progress, and startup-timing normalization only.
- Review gate: ask Claude to review FFI ownership, unsafe boundaries,
  lifecycle/cleanup behavior, and comparator coverage.
- Validation passed: local `cargo fmt --all -- --check`, local `cargo test
  --workspace`, local `python3 -m py_compile
  ds4-parity/compare_cli_inspect_runtime.py`, B300 `cargo test -p ds4-engine`
  using temporary `/tmp/ds4-cargo`/`/tmp/ds4-rustup`, B300 `cargo build -p
  ds4-engine --bin ds4-inspect-runtime-rs`, and B300
  `python3 ds4-parity/compare_cli_inspect_runtime.py
  ds4-parity/baselines/cli/m8.8/current-c.json --candidate-binary
  target/debug/ds4-inspect-runtime-rs --negative-test` (`CLI inspect runtime
  comparator: PASS, 68 checks`; negative tests `PASS, 5 checks`).
- Owner path: Rust inspect runtime boundary, CLI inspect comparator,
  `ds4-parity/`, `.memory/status.md`.

### M8.9b: Rust CLI Inspect Output Surface

- Status: done
- Goal: route Rust CLI `--inspect` handling through the M8.9a runtime boundary.
- Source evidence needed: committed M8.8 inspect fixture, M8.9a runtime
  comparator, and Rust CLI parse/dispatch code.
- Oracle: committed M8.8 current-C inspect fixture plus the M8.9a
  runtime-boundary comparator.
- Fixture: same M8.8 CLI cases.
- Comparator: C/Rust inspect comparator for normalized summary output, model
  identity, backend identity, exit status, and forbidden-path stderr markers.
- Acceptance: Rust enters only the inspect path and matches the committed C
  summary surface within documented normalization, without using the M8.8 JSON
  artifact as a substitute for model execution.
- Drift policy: path, backend progress, and startup-timing normalization only.
- Review gate: ask Claude to review dispatch exclusivity and output
  normalization.
- Validation passed: local `cargo fmt --all -- --check`, local `cargo test
  -p ds4-gguf cli_parse::tests::config_retains_inspect_backend_and_runtime_flags`,
  local `cargo test -p ds4-engine`, local `cargo test --workspace`, local
  `python3 -m py_compile ds4-parity/compare_cli_inspect_runtime.py`, local
  `git diff --check`, B300 `cargo test -p ds4-gguf
  cli_parse::tests::config_retains_inspect_backend_and_runtime_flags`, B300
  `cargo test -p ds4-engine`, B300 `cargo build -p ds4-engine --bin
  ds4-cli-inspect-rs`, and B300
  `python3 ds4-parity/compare_cli_inspect_runtime.py
  ds4-parity/baselines/cli/m8.8/current-c.json --candidate-binary
  target/debug/ds4-cli-inspect-rs --use-case-argv --negative-test` (`CLI
  inspect comparator: PASS, 68 checks`; negative tests `PASS, 5 checks`).
- Owner path: Rust CLI inspect path, CLI inspect comparator, `ds4-parity/`,
  `.memory/status.md`.

### M8.10: Current-C CLI Imatrix Capture Oracle

- Status: split into M8.10a and M8.10b before output-oracle implementation
- Goal: split the original current-C CLI imatrix capture oracle because the
  roadmap assumed B300 could run it, but current C forces `--imatrix-out` to
  the Metal backend and the imatrix collector requires Metal.
- Source evidence needed: `ds4_cli.c` imatrix option parsing,
  `ds4_engine_open`, `ds4_engine_collect_imatrix`, B300 failure proof, and
  local model-host availability.
- Oracle: source evidence plus B300 execution evidence from the recorded model
  host.
- Comparator: roadmap/board review against source and the B300 failure proof.
- Acceptance: no output `.dat` oracle is claimed from the B300 CUDA host; the
  output-hash oracle remains blocked until a valid Metal-capable model host or
  current-C CUDA imatrix support exists.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split avoids overstating
  current-C output coverage and preserves an exact future capture contract.
- Validation passed: B300 proof command showed exit 1, zero stdout bytes,
  `backend=metal`, `Metal backend requested but this build is linked with CUDA,
  not Metal`, and no output `.dat`; local check found no `ds4flash.gguf` or
  imatrix GGUF in the workspace on this 48 GiB host; `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M8.10a: Current-C CLI Imatrix Feasibility Guard

- Status: done
- Goal: capture the current reason the original M8.10 output oracle cannot run
  on the B300 CUDA model host.
- Source evidence needed: `ds4_cli.c`, `ds4.c`, B300 `./ds4
  --imatrix-dataset --imatrix-out` proof, and local model availability check.
- Oracle: current C `./ds4 --imatrix-dataset --imatrix-out` on B300 with a
  tiny fixed dataset and the recorded model path.
- Fixture: tiny one-prompt dataset, `--ctx 64`, `--imatrix-max-prompts 1`,
  `--imatrix-max-tokens 16`, `/workspace/ds4/ds4flash.gguf`, and B300
  `hou2-prod1` pod identity.
- Comparator: status evidence requires exit code 1, zero stdout bytes,
  `backend=metal` in context-buffer stderr, `Metal backend requested but this
  build is linked with CUDA, not Metal`, and no output `.dat`.
- Acceptance: the B300 blocker is recorded exactly enough to prevent an
  invalid `.dat` hash oracle from being committed.
- Drift policy: workspace paths may drift; backend mismatch category, exit
  status, and missing output are exact.
- Review gate: ask Claude to review source and evidence for the Metal-only
  conclusion.
- Validation passed: source inspection; B300 proof command; local model
  availability check; `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M8.10b: Current-C CLI Imatrix Output Oracle

- Status: blocked
- Goal: capture the current-C CLI imatrix execution mode once a valid host is
  available.
- Source evidence needed: current `./ds4 --imatrix-dataset --imatrix-out`
  behavior on a Metal-capable host with the recorded model, fixed imatrix
  dataset, imatrix limit flags, output file metadata, and stderr/progress
  categories.
- Oracle: current `./ds4 --imatrix-dataset --imatrix-out` on the recorded model
  from a Metal-capable host, or on B300 after current C gains CUDA imatrix
  collection support.
- Fixture: fixed imatrix dataset, output `.dat` file hash/size, `--ctx`,
  `--imatrix-max-prompts`, `--imatrix-max-tokens`, backend/model identity,
  progress stderr categories, and exact refresh commands.
- Comparator: schema/hash checker for output file metadata, prompt/token limit
  accounting, exit status, stderr categories, and refresh commands.
- Acceptance: output file hash/size and limit accounting match current C for
  the fixed dataset; invalid coupling remains covered by M8.2.
- Drift policy: timing/progress counters and workspace paths may be normalized;
  output bytes, limit semantics, model identity, and exit status are exact.
- Review gate: ask Claude to review imatrix dataset coverage and limit
  semantics.
- Validation needed: output capture, local checker with negative tests, and
  `git diff --check`.
- Owner path: CLI imatrix oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.11: Rust CLI Imatrix Capture Parity

- Status: blocked on M8.10b
- Goal: implement Rust CLI parity for imatrix capture mode.
- Source evidence needed: committed M8.10b current-C imatrix fixture and Rust
  CLI/runtime support for imatrix capture.
- Oracle: committed M8.10b current-C imatrix fixture.
- Fixture: same dataset, limit, context, backend, and output-path cases as
  M8.10b.
- Comparator: C/Rust imatrix comparator for output file hash/size, limit
  accounting, exit status, and normalized stderr categories.
- Acceptance: Rust writes the same imatrix output bytes for the committed
  dataset and preserves the current C limit semantics.
- Drift policy: timing/progress/path normalization only.
- Review gate: ask Claude to review file-output determinism and limit handling.
- Validation needed: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison when required, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: Rust CLI imatrix path, CLI imatrix comparator, `ds4-parity/`,
  `.memory/status.md`.

### M8.12: Current-C CLI One-Shot Generation Oracle

- Status: split into M8.12a and M8.12b before fixture implementation
- Goal: split the broad current-C one-shot generation oracle into core
  transcript behavior and advanced runtime-control behavior before committing
  fixtures.
- Source evidence needed: `ds4_cli.c` one-shot generation dispatch, current
  `./ds4` on the recorded B300 model, fixed prompt inputs, thinking controls,
  sampling controls, advanced runtime controls, backend/model identity, and
  stderr normalization policy.
- Oracle: current `./ds4` one-shot mode on the recorded B300 model.
- Comparator: roadmap/board review that each successor item has one tangible
  capture surface and one checker contract.
- Acceptance: core generation, prompt-file, thinking-control, seeded sampling,
  and context-size boundary behavior are captured first; MTP, directional
  steering, quality, warm-weight, thread, and backend-option coverage remains a
  separately verifiable oracle item.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split preserves M8.12 coverage
  without hiding advanced flags in the core transcript oracle.
- Validation passed: roadmap/board diff and `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M8.12a: Current-C CLI One-Shot Core Transcript Oracle

- Status: done
- Goal: capture deterministic current-C one-shot CLI generation transcripts for
  the core prompt and thinking-control surface.
- Source evidence needed: `ds4_cli.c` one-shot generation dispatch, current
  `./ds4` on the recorded B300 model, fixed prompt inputs, thinking controls,
  sampling controls, backend/model identity, and stderr normalization policy.
- Oracle: current `./ds4` one-shot mode on the recorded B300 model.
- Fixture: `-p`, `--prompt-file`, greedy generation with fixed token limit,
  seeded non-greedy sampling, `--nothink`, `--think`, `--think-max` downgrade
  warning, explicit `--cuda`, fixed `--ctx`, context-size boundary behavior,
  prompt hashes, seed, backend/model identity, and timing/progress stderr
  normalization rules.
- Comparator: transcript checker for stdout bytes, selected token sequence when
  available from captured text, exit status, stderr categories, prompt hashes,
  seed, backend/model identity, and exact B300 refresh commands.
- Acceptance: deterministic cases match byte-for-byte after documented
  progress/timing normalization; sampled cases are fixed by seed and compared by
  generated stdout bytes plus metadata.
- Drift policy: timing, throughput, terminal color, and absolute paths may be
  normalized; generated bytes, exit status, and stderr categories are exact.
- Review gate: ask Claude to review stdout/stderr normalization and seeded
  sampling determinism.
- Validation passed: B300 capture on `ds4-rust-port-b300` after `make ds4
  CUDA_ARCH=native`, `python3 ds4-parity/check_cli_generation_dump.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --manifest
  ds4-parity/baselines/cli/m8.12a/manifest.json --negative-test` (`CLI
  generation oracle: PASS, 156 checks`; manifest `PASS, 17 checks`; negative
  tests `PASS, 5 checks`), local revalidation with the same PASS counts,
  `python3 -m py_compile ds4-parity/check_cli_generation_dump.py`,
  `python3 -m json.tool` for both M8.12a JSON files, and `git diff --check`.
- Owner path: CLI one-shot oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.12b: Current-C CLI One-Shot Runtime-Control Oracle

- Status: done
- Goal: capture current-C one-shot generation behavior for advanced runtime
  controls after the core transcript oracle is stable.
- Source evidence needed: B300 support-artifact availability for MTP and
  directional steering, current `./ds4` advanced option handling, and M8.12a
  transcript checker extension points.
- Oracle: current `./ds4` one-shot mode on the recorded B300 model plus any
  required support model or steering fixture availability proof.
- Fixture: backend selection, `--mtp`, `--mtp-draft`, `--mtp-margin`,
  `--quality`, `--dir-steering-file`, `--dir-steering-ffn`,
  `--dir-steering-attn`, `--warm-weights`, `-t`/`--threads`, and the same
  timing/progress stderr normalization rules as M8.12a.
- Comparator: transcript checker extensions for startup/runtime-control stderr
  categories, generated stdout bytes, exit status, support-artifact identity,
  and exact B300 refresh commands.
- Acceptance: each advanced flag either has an executed transcript artifact or
  an exact availability blocker; no flag is silently folded into the core
  oracle without evidence.
- Drift policy: timing, throughput, and absolute paths may be normalized;
  generated bytes, exit status, support-artifact identity, and stderr
  categories are exact.
- Review gate: ask Claude to review advanced-option coverage and blocker
  evidence.
- Validation passed: B300 capture on `ds4-rust-port-b300` after `make ds4
  CUDA_ARCH=native`, `python3 ds4-parity/check_cli_runtime_controls_dump.py
  ds4-parity/baselines/cli/m8.12b/current-c.json --manifest
  ds4-parity/baselines/cli/m8.12b/manifest.json --negative-test` (`CLI
  runtime-controls oracle: PASS, 158 checks`; manifest `PASS, 16 checks`;
  negative tests `PASS, 5 checks`), local revalidation with the same PASS
  counts, `python3 -m py_compile
  ds4-parity/check_cli_runtime_controls_dump.py`, `python3 -m json.tool` for
  both M8.12b JSON files, and `git diff --check`.
- Owner path: CLI one-shot oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.13: Rust CLI One-Shot Generation Parity

- Status: split into M8.13a, M8.13b, M8.13c, and M8.13d before implementation
- Goal: split the original Rust CLI one-shot parity item because the Rust
  runtime currently exposes engine open/summary but not prompt encoding,
  generated-token text, argmax generation callbacks, or session sampling.
- Source evidence needed: committed M8.12a/M8.12b current-C transcript
  fixtures, `rust/ds4-engine/src/lib.rs`,
  `rust/ds4-engine/src/bin/ds4-cli-inspect-rs.rs`, `ds4.h`, and `ds4_cli.c`.
- Oracle: committed M8.12a/M8.12b current-C transcript fixtures plus source
  evidence that Rust needs executable runtime boundaries before transcript
  parity can be claimed.
- Comparator: roadmap/board review that successor items introduce runtime
  boundaries before CLI transcript surfaces and do not replay current-C stdout
  as Rust execution.
- Acceptance: the broad item is decomposed into runtime-boundary prerequisites
  and CLI surface parity items.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split avoids overstating Rust
  runtime ownership and preserves M8.12a/M8.12b comparison coverage.
- Validation passed: source inspection, roadmap/board diff, `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M8.13a: Rust Argmax One-Shot Runtime Boundary

- Status: done
- Goal: expose a Rust-accessible one-shot argmax generation boundary over the
  current engine API without implementing the full CLI surface yet.
- Source evidence needed: `ds4.h`, `ds4.c`, `rust/ds4-engine/src/lib.rs`, the
  M8.12a greedy transcript cases, and B300 model/runtime availability.
- Oracle: current C `ds4_encode_chat_prompt`, `ds4_engine_generate_argmax`,
  `ds4_token_text`, `ds4_tokens_free`, and the M8.12a greedy transcript cases.
- Fixture: minimal inline and prompt-file prompts, `--nothink`/`--think`,
  explicit backend/model/context/token limit, and the M8.12a too-small-context
  error category.
- Comparator: C/Rust runtime-boundary comparator for exit status, generated
  stdout bytes, normalized stderr categories, and no artifact replay.
- Acceptance: Rust can open the same model/backend, encode prompt tokens through
  the current engine API, emit generated token bytes through callbacks, and
  match the greedy M8.12a cases it claims.
- Drift policy: timing/progress/path normalization only; generated bytes and
  exit categories exact.
- Review gate: ask Claude to review unsafe callback ownership, token text
  lifetimes, prompt token ownership, and cleanup behavior.
- Validation passed: B300 `cargo build -p ds4-engine --bin
  ds4-argmax-runtime-rs` with `CUDA_ARCH=native`; B300
  `python3 ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-argmax-runtime-rs --negative-test` (`CLI argmax runtime
  comparator: PASS, 109 checks`; negative tests `PASS, 4 checks`); local
  `cargo fmt --all -- --check`; local `cargo test --workspace`; local
  `python3 -m py_compile ds4-parity/compare_cli_argmax_runtime.py`; and
  `git diff --check`.
- Owner path: `rust/ds4-engine/`,
  `ds4-parity/compare_cli_argmax_runtime.py`, `.memory/status.md`.

### M8.13b: Rust Session Sampling Runtime Boundary

- Status: done
- Goal: expose the session-backed Rust runtime boundary needed for seeded
  non-greedy one-shot sampling and future MTP speculation.
- Source evidence needed: `ds4.h`, `ds4.c`, `rust/ds4-engine/src/lib.rs`, the
  M8.12a seeded sampling case, and B300 model/runtime availability.
- Oracle: current C `ds4_session_create`, `ds4_session_sync`,
  `ds4_session_sample`, `ds4_session_eval`, `ds4_token_text`, and the M8.12a
  seeded sampling transcript case.
- Fixture: M8.12a seeded sampling case with seed `12345`, temperature/top-p/min-p
  controls, context/token limit, and prompt identity.
- Comparator: C/Rust runtime-boundary comparator for generated bytes, seed,
  exit status, normalized stderr categories, and no artifact replay.
- Acceptance: Rust can run the seeded session path against the same model/backend
  and match current-C generated bytes for the committed seeded case.
- Drift policy: timing/progress/path normalization only.
- Review gate: ask Claude to review session lifetime, RNG mutation, evaluation
  loop, and error propagation.
- Validation passed: B300 `cargo build -p ds4-engine --bin
  ds4-session-runtime-rs` with `CUDA_ARCH=native`; B300
  `python3 ds4-parity/compare_cli_session_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-session-runtime-rs --negative-test` (`CLI session runtime
  comparator: PASS, 28 checks`; negative tests `PASS, 5 checks`); local
  `cargo fmt --all -- --check`; local `cargo test -p ds4-engine`; local
  `cargo build -p ds4-engine --bin ds4-session-runtime-rs`; local
  `cargo test --workspace`; local `python3 -m py_compile` for M8.13a/M8.13b
  comparators; and `git diff --check`.
- Owner path: `rust/ds4-engine/`,
  `ds4-parity/compare_cli_session_runtime.py`, `.memory/status.md`.

### M8.13c: Rust CLI One-Shot Core Transcript Surface

- Status: done
- Goal: route the Rust CLI one-shot core surface through the M8.13a/M8.13b
  runtime boundaries.
- Source evidence needed: committed M8.12a current-C transcript fixture, Rust
  CLI parse/dispatch code, and M8.13a/M8.13b runtime-boundary comparators.
- Oracle: committed M8.12a current-C transcript fixture plus M8.13a/M8.13b
  runtime-boundary comparators.
- Fixture: same `-p`, `--prompt-file`, `--nothink`, `--think`, `--think-max`,
  seeded sampling, and context-boundary cases as M8.12a.
- Comparator: C/Rust CLI transcript comparator for normalized stderr, stdout
  bytes, exit status, and fixture metadata.
- Acceptance: Rust one-shot mode matches current C for M8.12a without using
  current-C JSON replay as a substitute for execution.
- Drift policy: only documented progress/timing/path normalization.
- Review gate: ask Claude to review CLI orchestration and normalization.
- Validation passed: B300 `cargo build -p ds4-engine --bin
  ds4-cli-one-shot-rs` with `CUDA_ARCH=native`; B300
  `python3 ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-cli-one-shot-rs --negative-test` (`CLI one-shot runtime
  comparator: PASS, 144 checks`; negative tests `PASS, 5 checks`); local
  parser tests; local `cargo build -p ds4-engine --bin ds4-cli-one-shot-rs`;
  local `cargo test --workspace`; local `python3 -m py_compile` for M8.13a
  through M8.13c comparators; and `git diff --check`.
- Owner path: `rust/ds4-gguf/src/cli_parse.rs`, `rust/ds4-engine/`,
  `ds4-parity/compare_cli_one_shot_runtime.py`, `.memory/status.md`.

### M8.13d: Rust CLI One-Shot Runtime-Control Surface

- Status: done
- Goal: extend Rust CLI one-shot parity to the M8.12b runtime-control cases.
- Source evidence needed: committed M8.12b current-C transcript fixture, Rust
  CLI parse/dispatch code, advanced runtime-control plumbing, and support
  artifact availability.
- Oracle: committed M8.12b current-C transcript fixture plus runtime-boundary
  support for the relevant engine options.
- Fixture: `--backend cuda --quality -t 2`, `--warm-weights`,
  `--dir-steering-file`, blocked `--backend metal`, and blocked missing-MTP
  behavior from M8.12b.
- Comparator: C/Rust CLI transcript comparator for generated bytes, support
  artifact identity, availability blockers, exit status, and normalized stderr.
- Acceptance: Rust matches every M8.12b executed or blocked runtime-control case
  it claims, including exact blockers for unavailable support artifacts.
- Drift policy: timing/progress/path normalization only.
- Review gate: ask Claude to review advanced option plumbing and blocker
  evidence.
- Validation passed: local `cargo fmt --all -- --check`; local `cargo test -p
  ds4-gguf cli_parse -- --nocapture`; local `cargo build -p ds4-engine --bin
  ds4-cli-one-shot-rs`; local `cargo test -p ds4-engine`; local full
  `cargo test --workspace`; local `python3 -m py_compile` for M8.13a through
  M8.13d comparators; local `git diff --check`; B300 `cargo build -p
  ds4-engine --bin ds4-cli-one-shot-rs` with `CUDA_ARCH=native`; and B300
  `python3 ds4-parity/compare_cli_runtime_controls_runtime.py
  ds4-parity/baselines/cli/m8.12b/current-c.json --candidate-binary
  target/debug/ds4-cli-one-shot-rs --negative-test` (`CLI runtime-controls
  runtime comparator: PASS, 154 checks`; negative tests `PASS, 6 checks`).
- Owner path: Rust CLI one-shot path, runtime-control transcript comparator,
  `ds4-parity/`, `.memory/status.md`.

### M8.14: Current-C Interactive CLI Transcript Oracle

- Status: done
- Goal: capture scripted current-C interactive CLI behavior.
- Source evidence needed: current `./ds4` REPL implementation, PTY behavior for
  `linenoise`, prompts, Ctrl+C, and command output, and B300 model-backed CLI
  availability.
- Oracle: current `./ds4` REPL using a PTY so prompts, command output, terminal
  control, and interruption behavior are represented.
- Fixture: `/help`, `/think`, `/think-max`, `/nothink`, `/ctx`, `/read`,
  unknown command, `/quit`, empty input, one short model-backed turn, and a
  Ctrl+C interruption case if the PTY harness can make it deterministic.
- Comparator: PTY transcript checker for prompt markers, command responses,
  normalized progress/timing lines, exit status, transcript hashes, and exact
  B300 refresh commands for model-backed cases.
- Acceptance: command-only cases run locally without a model where possible;
  model-backed transcript cases have reproducible B300 commands and stable
  normalization for timing, terminal control, and color.
- Drift policy: terminal color/control sequences, timing, and absolute paths may
  be normalized; prompt markers, command responses, and exit categories are
  exact.
- Review gate: ask Claude to review PTY determinism and command coverage.
- Validation passed: B300 `make ds4 CUDA_ARCH=native`; B300
  `python3 ds4-parity/check_cli_interactive_dump.py
  --write-baseline ds4-parity/baselines/cli/m8.14/current-c.json
  --write-manifest ds4-parity/baselines/cli/m8.14/manifest.json --binary
  ./ds4`; B300 and local
  `python3 ds4-parity/check_cli_interactive_dump.py
  ds4-parity/baselines/cli/m8.14/current-c.json --manifest
  ds4-parity/baselines/cli/m8.14/manifest.json --negative-test` (`CLI
  interactive oracle: PASS, 89 checks`; manifest `PASS, 15 checks`; negative
  tests `PASS, 6 checks`); local `python3 -m json.tool` for both M8.14 JSON
  files; local `python3 -m py_compile ds4-parity/check_cli_interactive_dump.py`;
  and `git diff --check`.
- Owner path: CLI interactive oracle artifacts, `ds4-parity/`,
  `ds4-parity/baselines/cli/`, `.memory/status.md`.

### M8.15: Rust Interactive CLI Transcript Parity

- Status: split into M8.15a, M8.15b, and M8.15c before implementation
- Goal: split Rust interactive CLI parity because the current Rust runtime has
  one-shot generation but not reusable sessions, chat transcript mutation,
  session progress callbacks, or REPL state.
- Source evidence needed: committed M8.14 current-C PTY transcript fixture,
  `ds4_cli.c`, `ds4.h`, and `rust/ds4-engine/src/lib.rs`.
- Oracle: committed M8.14 current-C PTY transcript fixture plus source evidence
  that Rust needs reusable runtime and command-state prerequisites.
- Fixture: M8.14 `command_suite` and `ctrl_c_at_prompt` cases.
- Comparator: roadmap/board review that successor items separate reusable
  runtime ownership, REPL command state, and final PTY transcript parity.
- Acceptance: broad interactive parity is decomposed into runtime-boundary,
  command-state, and PTY-surface items before source behavior changes.
- Drift policy: no source behavior changes; this is scope control.
- Review gate: ask Claude to review whether the split preserves all M8.14
  transcript coverage and does not hide runtime gaps.
- Validation needed: source inspection, roadmap/board diff, and
  `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M8.15a: Rust Reusable Interactive Session Boundary

- Status: done
- Goal: expose the Rust runtime primitives needed for interactive turns without
  building the REPL surface yet.
- Source evidence needed: `ds4.h` chat/session APIs, `ds4_cli.c` interactive
  turn logic, `rust/ds4-engine/src/lib.rs`, and the M8.14 model-backed turns.
- Oracle: current C `ds4_chat_begin`, `ds4_chat_append_message`,
  `ds4_chat_append_assistant_prefix`, `ds4_chat_append_max_effort_prefix`,
  `ds4_tokens_push`, reusable `ds4_session_*` APIs, and the M8.14 `/read` plus
  direct-prompt generated bytes.
- Fixture: B300 model, context 128, one generated token, `--temp 0`,
  `--nothink`, the M8.14 `/read` fixture prompt, and the direct prompt `Answer
  with one short noun: glacier.` after the first turn.
- Comparator: C/Rust runtime-boundary comparator for generated bytes, reusable
  session position/room handling, transcript token append/eos behavior,
  normalized progress/timing categories, and no PTY line-editing assumptions.
- Acceptance: Rust can maintain the same token transcript across two
  interactive turns, reuse one session, emit the same generated bytes as M8.14,
  and recreate/invalidate the session on context reset.
- Drift policy: timing/progress/path normalization only; generated bytes and
  exit categories are exact.
- Review gate: ask Claude to review unsafe session ownership, progress callback
  lifetimes, token transcript mutation, and cleanup.
- Validation passed: local `cargo fmt --all -- --check`; local `cargo build -p
  ds4-engine --bin ds4-interactive-runtime-rs`; local `cargo test -p
  ds4-engine`; local full `cargo test --workspace`; local `python3 -m
  py_compile ds4-parity/compare_cli_interactive_runtime.py
  ds4-parity/check_cli_interactive_dump.py`; local `git diff --check`; B300
  `cargo build -p ds4-engine --bin ds4-interactive-runtime-rs` with
  `CUDA_ARCH=native`; and B300 `python3
  ds4-parity/compare_cli_interactive_runtime.py
  ds4-parity/baselines/cli/m8.14/current-c.json --candidate-binary
  target/debug/ds4-interactive-runtime-rs --negative-test` (`CLI interactive
  runtime comparator: PASS, 19 checks`; negative tests `PASS, 4 checks`).
- Owner path: `rust/ds4-engine/`, `ds4-parity/`, `.memory/status.md`.

### M8.15b: Rust REPL Command State Surface

- Status: done
- Goal: implement the Rust command-state layer for the M8.14 interactive
  commands before claiming PTY transcript parity.
- Source evidence needed: `ds4_cli.c:run_repl`, M8.14 command transcript, and
  Rust CLI parser state.
- Oracle: current C command handling in `ds4_cli.c:run_repl` and the committed
  M8.14 command transcript.
- Fixture: empty input, `/help`, `/think`, `/think-max`, `/nothink`, `/ctx
  128`, `/read FILE`, unknown command, `/quit`, and Ctrl+C-at-prompt state.
- Comparator: command-state tests and/or a non-model harness for response text,
  thinking-mode transitions, context updates, read-file errors, quit/exit
  handling, and Ctrl+C-at-prompt recovery.
- Acceptance: Rust command handling produces the same response categories and
  state transitions that M8.14 records, with model-backed turn execution still
  delegated to M8.15a.
- Drift policy: command response text and state transitions are exact; terminal
  redraw is deferred to M8.15c.
- Review gate: ask Claude to review command routing and state-machine edge
  cases.
- Validation passed: local `cargo fmt --all -- --check`; local `cargo test -p
  ds4-engine interactive_cli -- --nocapture` (5 REPL command tests); local
  `cargo test -p ds4-engine` (11 tests); local full `cargo test --workspace`;
  and local `git diff --check`.
- Owner path: Rust CLI interactive state, `.memory/status.md`.

### M8.15c: Rust Interactive PTY Transcript Surface

- Status: done
- Goal: wire the Rust no-prompt CLI path into an interactive PTY surface and
  compare it to the M8.14 current-C transcript.
- Source evidence needed: M8.14 PTY oracle, M8.15a reusable runtime, and M8.15b
  command-state implementation.
- Oracle: committed M8.14 PTY transcript fixture plus M8.15a/M8.15b behavior.
- Fixture: same `command_suite` and `ctrl_c_at_prompt` cases as M8.14.
- Comparator: C/Rust PTY transcript comparator for prompt markers, command
  responses, normalized timing/progress, generated bytes, exit status,
  transcript hashes, and interruption behavior.
- Acceptance: Rust REPL command handling, prompt display, file-read behavior,
  thinking-mode switching, context reset, model-backed turns, and
  Ctrl+C-at-prompt behavior match the C CLI within documented normalization.
- Drift policy: terminal escape/redraw normalization only where the comparator
  records it.
- Review gate: ask Claude to review interactive state handling, PTY
  determinism, and transcript normalization.
- Validation passed: local `cargo fmt --all -- --check`; local `cargo build -p
  ds4-engine --bin ds4-cli-interactive-rs`; local `cargo test -p
  ds4-engine`; local full `cargo test --workspace`; local `python3 -m
  py_compile ds4-parity/compare_cli_interactive_pty.py
  ds4-parity/check_cli_interactive_dump.py`; local `git diff --check`; B300
  `cargo build -p ds4-engine --bin ds4-cli-interactive-rs` with
  `CUDA_ARCH=native`; and B300 `python3
  ds4-parity/compare_cli_interactive_pty.py
  ds4-parity/baselines/cli/m8.14/current-c.json --candidate-binary
  target/debug/ds4-cli-interactive-rs --write-candidate
  /tmp/ds4-m8.15c-rust-pty.json --negative-test` (`CLI interactive PTY
  comparator: PASS, 59 checks`; negative tests `PASS, 4 checks`).
- Owner path: Rust CLI interactive path, PTY comparator, `ds4-parity/`,
  `.memory/status.md`.

### M8.16: CLI Parity Report Integration

- Status: done
- Goal: wire M8 CLI comparators and B300 refresh records into parity reports.
- Source evidence needed: M8.2 through M8.15 fixtures, local comparator
  commands, B300 recapture records, and PTY transcript records.
- Oracle: committed M8.2 through M8.15 fixtures and refresh commands.
- Fixture: M8 manifest entries, local comparator commands, B300 recapture
  records, and PTY transcript records.
- Comparator: a Milestone 8 report that runs all local CLI comparators,
  summarizes first drift paths, and skips model-backed B300 refresh/runtime
  checks with exact commands; the unified parity report includes that M8 report.
- Acceptance: local report passes without the model, JSON output is
  machine-readable, failures name fixture/field/expected/got where underlying
  comparators provide it, and B300 refreshes are reproducible from the report.
- Drift policy: report normalizes only capture paths and timestamps.
- Review gate: ask Claude to review report integration and skipped-B300 command
  fidelity.
- Validation passed: local `python3 -m py_compile
  ds4-parity/run_cli_parity_report.py ds4-parity/run_parity_report.py`; local
  `python3 ds4-parity/run_cli_parity_report.py` (`summary: 9 passed, 13
  skipped, 0 failed`); local `python3 ds4-parity/run_cli_parity_report.py
  --json` plus `python3 -m json.tool`; local `python3
  ds4-parity/run_parity_report.py` (`summary: 14 passed, 5 skipped, 0
  failed`); local `cargo test --workspace`; and local `git diff --check`.
- Owner path: `ds4-parity/`, `.memory/status.md`.

### M9.1: Server Surface Work Item Breakdown

- Status: done
- Goal: split Milestone 9 into commit-sized Rust server parity work items
  before adding Rust server source behavior.
- Source evidence needed: `RUST_PORT_ROADMAP.md` Milestone 9, `ds4_server.c`,
  `tests/ds4_test.c` server tests, M0.4 server fixtures/traces, and M0.5 KV
  restore fixtures.
- Oracle: current `./ds4-server` behavior captured in M0.4/M0.5 plus
  `./ds4_test --server`.
- Fixture: fixed request JSON fixtures, server trace outputs, KV/cache
  fixtures, and streaming/non-streaming response cases.
- Comparator: documentation-only breakdown that assigns concrete oracle,
  fixture, comparator, acceptance, drift policy, validation, and owner paths to
  each server parity work item.
- Acceptance: the next server implementation items are small enough to validate
  independently and include request/response, streaming, trace, cache, and
  tool-call coverage without mixing unrelated server behavior.
- Drift policy: documentation-only; no source behavior changes.
- Review gate: ask Claude to review item boundaries and missing server
  behavioral surfaces.
- Validation passed: source inspection of `ds4_server.c`, `tests/ds4_test.c`,
  M0.4 server fixtures/traces, M0.5 KV fixtures/artifacts, and
  `compare_server_kv.py`; roadmap/board diff; and `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M9.2: Server Request Parse And Prompt Render Surface

- Status: done via M9.2a, M9.2b, and M9.2c
- Goal: port the model-free request parsing and prompt-rendering surface needed
  by the server before adding an HTTP listener.
- Oracle: C `parse_chat_request`, `parse_anthropic_request`,
  `parse_responses_request`, `render_chat_prompt_text`, thinking-control tests,
  stop-list tests, context-length error shape, and `ds4_server_unit_tests_run`.
- Fixture: M0.4 request JSON plus server unit vectors for OpenAI, Responses,
  Anthropic, tool schemas, reasoning/thinking controls, stop lists, context
  limits, CORS policy inputs, and prompt-rendered text.
- Comparator: Rust unit tests and/or dump helper comparing normalized request
  fields, rendered prompt text, stream flags, tool schema ordering, stop lists,
  thinking mode, max-token/context decisions, and protocol error categories.
- Acceptance: Rust produces the same request semantics and prompt text for the
  covered unit vectors without opening sockets or loading the model.
- Drift policy: exact for semantic fields and prompt bytes; stable-category
  comparison for path/limit-bearing error text.
- Review gate: ask Claude to review parser coverage for OpenAI chat, Responses,
  Anthropic, thinking controls, stop lists, and context-limit errors.
- Validation passed: targeted parser tests, full `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check` across child items.
- Owner path: Rust server parser modules, `ds4-parity/`, `.memory/status.md`.

### M9.2a: OpenAI Chat Request Core Parse And Render

- Status: done
- Goal: port the model-free OpenAI `/v1/chat/completions` core request parser
  and prompt renderer, excluding tool-call payloads and alternate protocols.
- Oracle: C `parse_chat_request`, `render_chat_prompt_text`, request default
  tests, thinking-control tests, stop-list tests, context-limit error tests, and
  M0.4 non-tool OpenAI request fixtures.
- Fixture: M0.4 `chat_basic`, `chat_stream`, `chat_thinking_disabled`,
  `chat_cache_seed`, and `chat_cache_continuation` request JSON plus unit
  vectors for defaults, stream options, stop lists, and thinking controls.
- Comparator: Rust unit tests and/or dump helper comparing normalized request
  fields, rendered prompt bytes, stream flags, generation options, thinking
  mode, stop lists, max-token/context decisions, and error categories.
- Acceptance: Rust matches C parser/render semantics for non-tool OpenAI chat
  requests without opening sockets or loading the model.
- Drift policy: exact for semantic fields and prompt bytes; stable-category
  comparison for path/limit-bearing error text.
- Review gate: ask Claude to review OpenAI field coverage, default values,
  stream option handling, thinking mode mapping, and context-limit errors.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat`, full
  `cargo test --workspace`, `python3 ds4-parity/run_cli_parity_report.py`,
  `git diff --check`, staged diff check, and Claude review PASS.
- Owner path: Rust server parser modules, `ds4-parity/`, `.memory/status.md`.

### M9.2b: OpenAI Tool Schema And DSML Prompt Render Surface

- Status: done
- Goal: port model-free OpenAI tool schema parsing and DSML prompt rendering
  without implementing model-backed tool-call generation.
- Oracle: C `parse_tools_value`, `openai_function_schema_from_tool`,
  `append_tools_prompt_text`, `append_dsml_tool_calls_text`, tool schema order
  tests, DSML parser tests, and M0.4 `chat_tool_call` request/trace prompt.
- Fixture: M0.4 `chat_tool_call.json`, M0.4 tool trace prompt segment, and unit
  vectors for schema property order, DSML argument ordering, malformed tool-call
  recovery, partial tool-call holds, and loose nested parameters.
- Comparator: Rust unit tests/dump helper comparing tool schema normalization,
  rendered tool prompt bytes, DSML call text, executable-tool boundary
  categories, and recoverable parse categories.
- Acceptance: Rust model-free tool parsing and prompt rendering match current C
  for OpenAI tool requests before server response generation is ported.
- Drift policy: exact for schema names, argument order, prompt bytes, and DSML
  text; random call IDs are out of scope for this parser-only item.
- Review gate: ask Claude to review schema ordering, DSML state-machine edges,
  malformed/recoverable tool parsing, and prompt placement before system text.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat`, full
  `cargo test --workspace`, `python3 ds4-parity/run_cli_parity_report.py`,
  `git diff --check`, staged diff check, and Claude review PASS.
- Owner path: Rust server tool/DSML parser modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.2c: Responses And Anthropic Request Parse Surface

- Status: done via M9.2c1, M9.2c2, and M9.2c3
- Goal: port model-free Responses and Anthropic request parsing/rendering inputs
  while leaving response/event emission for M9.7.
- Oracle: C `parse_responses_request`, `parse_anthropic_request`,
  `parse_responses_input`, `parse_anthropic_messages`, protocol system/tool
  validation tests, and live-tail requirement tests.
- Fixture: unit vectors for Responses namespace/tool_search schemas, reasoning
  inputs, function_call outputs, tool outputs, Anthropic content blocks, private
  system filtering, tool use/results, and live-tail validation.
- Comparator: Rust protocol unit tests/dump helper comparing normalized request
  fields, rendered prompt/live-tail text, tool-output validation categories,
  reasoning requirements, usage-relevant flags, and stream flags.
- Acceptance: Rust matches current C request semantics for Responses and
  Anthropic without opening sockets, emitting protocol responses, or loading the
  model. M9.2c parser tests use no-op/stub server state for validation paths;
  KV/tool-memory replay side effects from `kv_cache_restore_tool_memory_for_messages`
  and `tool_memory_attach_to_messages` are explicitly deferred to M9.8.
- Drift policy: exact for semantic fields, validation categories, prompt bytes,
  and live-tail text; random IDs and response timestamps are out of scope here.
- Review gate: ask Claude to review protocol-specific state, reasoning replay
  requirements, namespace tool schema restoration, and Anthropic tool-result ID
  validation.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check` across child items.
- Owner path: Rust server protocol parser modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.2c1: Responses Core Input And Reasoning Parse Surface

- Status: done
- Goal: port model-free Responses API core request parsing for `input`,
  `instructions`, scalar generation controls, reasoning effort/summary flags,
  durable-state rejection, and prompt rendering, excluding tool-output
  live-tail validation and tool schemas loaded from input tool-search results.
- Oracle: C `parse_responses_request`, `parse_responses_reasoning`,
  string/array `input` handling, `instructions` system prepend, model alias
  thinking fallbacks, and `previous_response_id`/`conversation` rejection
  branches.
- Fixture: unit vectors for bare string input, message input arrays,
  instructions prepend, `reasoning.effort`/`reasoning.summary`, tool-choice
  unsupported categories, and durable-state non-null errors.
- Comparator: Rust unit tests comparing normalized request fields, rendered
  prompt bytes, stream flags, reasoning summary emit flag, thinking mode,
  generation controls, and stable error categories.
- Acceptance: Rust matches current C Responses core request semantics without
  live tool-state validation, protocol response emission, sockets, or model
  loading. Top-level `tools` can participate in prompt rendering here, but
  tool schemas loaded from input items and the final combined-schema merge are
  completed in M9.2c2.
- Drift policy: exact for prompt bytes and semantic fields; stable-category
  comparison for durable-state and unsupported tool-choice error strings.
- Review gate: ask Claude to review Responses core field coverage, reasoning
  controls, instructions ordering, durable-state rejection, and prompt bytes.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c2: Responses Tool Output And Live-Tail Parse Surface

- Status: split into M9.2c2a, M9.2c2b, and M9.2c2c before implementation
- Goal: split the Responses tool-output/function-call item into reviewable
  parser, schema-loading, and live-continuation work before implementation.
- Oracle: M9.2c2a through M9.2c2c below, each tied to the current C
  `parse_responses_input`, tool-schema helpers, and live continuation helpers.
- Fixture: documentation-only split assigning function/tool-call input vectors,
  namespace/tool-search schema vectors, and live-tail validation vectors to
  separate commits.
- Comparator: roadmap and active-board review that every child item has an
  oracle, fixture, comparator, acceptance rule, drift policy, review gate, and
  validation gate.
- Acceptance: the original M9.2c2 scope is fully covered by child items before
  any source behavior changes.
- Drift policy: documentation-only; no source behavior drift.
- Review gate: ask Claude to review boundary completeness and whether any C
  branch is unassigned.
- Validation needed: roadmap/board diff and `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c2a: Responses Function Call And Tool Output Input Surface

- Status: done
- Goal: port model-free Responses input items that become chat tool-call
  history or tool-result history: `function_call`, `custom_tool_call`,
  hosted-tool calls, `function_call_output`, custom/hosted tool outputs, call
  IDs, pending-reasoning merge rules, and DSML prompt rendering.
- Oracle: C `parse_responses_input` branches for `function_call`,
  `custom_tool_call`, `local_shell_call`, `web_search_call`,
  `tool_search_call`, `image_generation_call`, `function_call_output`,
  `custom_tool_call_output`, hosted tool outputs,
  `chat_msg_add_tool_call_id`, and `render_chat_prompt_text`.
- Fixture: unit vectors for assistant text plus split tool-call item merging,
  function-call JSON arguments, custom tool free-text input, hosted tool action
  payloads, output/result body selection, trailing tool-result prompt tails,
  duplicate call-id preservation, and pending reasoning before tool outputs.
- Comparator: Rust unit tests comparing normalized `ChatMessage`/`ToolCall`
  fields, tool-result call IDs, prompt bytes, DSML argument ordering, and
  reasoning attachment.
- Acceptance: Rust parses and renders Responses tool-call/tool-output input
  history like C without loading dynamic tool schemas or validating live server
  state.
- Drift policy: exact for message roles, call IDs, tool names, argument JSON
  spelling after existing Rust minification, prompt bytes, and reasoning
  placement.
- Review gate: ask Claude to review function/custom/hosted call parsing,
  merge-with-previous-assistant behavior, call-id preservation, and prompt
  rendering.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c2b: Responses Tool Search And Namespace Schema Loading

- Status: done
- Goal: port Responses dynamic tool schema parsing: top-level `tool_search`,
  namespace tool groups, tool-search-output `tools` loading, combined
  top-level plus loaded schemas, and namespace/wire-name metadata.
- Oracle: C `responses_special_schema_from_tool`,
  `responses_namespace_function_schema_from_tool`,
  `append_responses_namespace_tool_schemas`, `parse_tools_value`,
  `tool_schema_orders_add_json_wire`, and the `tool_search_output` loading
  branch in `parse_responses_input`.
- Fixture: unit vectors for hosted `tool_search`, function named
  `tool_search`, namespace schema flattening, namespace `wire_name` metadata,
  tool-search output loading, malformed tool-search tool lists, and combined
  top-level plus loaded schema prompt text.
- Comparator: Rust unit tests comparing raw schema lines, `ToolSchemaOrder`
  metadata, prompt schema placement, loaded-schema append order, and malformed
  dynamic-tool rejection.
- Acceptance: Rust matches C schema loading and namespace restoration for
  Responses parser inputs while response-output emission remains outside this
  parser milestone.
- Drift policy: exact for schema names, namespace prefixes, wire names,
  property order, prompt schema line order, and malformed-schema categories.
- Review gate: ask Claude to review namespace flattening, hosted tool-search
  distinction, loaded-schema ordering, and malformed dynamic-tool handling.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c2c: Responses Live Tail Validation Surface

- Status: done
- Goal: port model-free Responses live continuation validation outputs:
  missing call-id errors, `requires_live_tool_state`,
  `requires_live_reasoning`, live call-id collection, and visible live suffix
  rendering for trailing tool results.
- Oracle: C `responses_validate_tool_outputs`,
  `responses_prepare_live_continuation`, `responses_find_prior_call_msg`,
  `chat_msg_collect_tool_call_ids`, `render_live_tool_tail`, and related C unit
  vectors.
- Fixture: unit vectors for tool-output-only missing state, live-known
  tool-output-only continuation, stateless replay with and without prior
  reasoning, non-thinking replay, assistant-call anchor plus trailing tool
  outputs, and tool-result suffix escaping.
- Comparator: Rust unit tests using an explicit no-op/live stub state to
  compare validation category, requirement flags, collected call IDs, and
  live-tail prompt bytes.
- Acceptance: Rust reports the same Responses live-continuation parser state as
  C without touching real server KV/tool-memory side effects, which remain
  assigned to M9.8.
- Drift policy: exact for error strings, live/reasoning requirement flags,
  call-id ordering/deduplication, and live-tail bytes.
- Review gate: ask Claude to review missing-state rejection, live-state flag
  assignment, reasoning replay requirements, and live-tail construction.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c3: Anthropic Message And Tool Result Parse Surface

- Status: done via M9.2c3a, M9.2c3b, and M9.2c3c
- Goal: split the Anthropic parser surface into reviewable core-message,
  tool-history/schema, and live-continuation work before implementation.
- Oracle: M9.2c3a through M9.2c3c below, each tied to the current C
  `parse_anthropic_*` and live validation helpers.
- Fixture: documentation-only split assigning system/message/control vectors,
  tool schema/tool history vectors, and live-tail validation vectors to
  separate commits.
- Comparator: roadmap and active-board review that every child item has an
  oracle, fixture, comparator, acceptance rule, drift policy, review gate, and
  validation gate.
- Acceptance: the original M9.2c3 scope is fully covered by child items before
  any source behavior changes.
- Drift policy: documentation-only; no source behavior drift.
- Review gate: ask Claude to review boundary completeness and whether any
  Anthropic C branch is unassigned.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check` across child items.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c3a: Anthropic Core Message And Control Parse Surface

- Status: done
- Goal: port model-free Anthropic core request parsing for `messages`,
  `system`, string/text content blocks, private system filtering, scalar
  generation controls, stop sequences, stream flag, `thinking`,
  `output_config.effort`, bare `reasoning_effort`, model alias fallbacks, and
  prompt rendering without tools.
- Oracle: C `parse_anthropic_request`, `parse_anthropic_messages`,
  `parse_anthropic_content`, `parse_anthropic_content_block` text branches,
  `parse_anthropic_system`, `parse_anthropic_system_object`,
  `parse_output_config_effort`, `parse_thinking_control_value`, `parse_stop`,
  and model alias thinking logic.
- Fixture: unit vectors for missing messages, string and block system prompts,
  private system blocks, string content, text content arrays, scalar controls,
  stop sequences, stream flag, thinking enabled/disabled,
  `output_config.effort`, bare `reasoning_effort`, and prompt bytes.
- Comparator: Rust unit tests comparing normalized request fields, rendered
  prompt bytes, stop lists, thinking mode, generation controls, stream flag,
  and stable error categories.
- Acceptance: Rust matches current C Anthropic core request semantics without
  tool schemas, tool history, live state, sockets, or model loading.
- Drift policy: exact for semantic fields and prompt bytes; stable-category
  comparison for missing/invalid request errors.
- Review gate: ask Claude to review system block filtering, text-content
  parsing, stop/thinking controls, effort precedence, and prompt bytes.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c3b: Anthropic Tool Schema And Tool History Parse Surface

- Status: done
- Goal: port model-free Anthropic tool schemas, `tool_choice.type`,
  assistant `tool_use` content blocks, user `tool_result` blocks, tool-use IDs,
  tool result prompt rendering, and DSML request-history rendering.
- Oracle: C `parse_anthropic_request` tool/tool_choice branches,
  `parse_tools_value`, `parse_anthropic_content_block` `tool_use` and
  `tool_result` branches, `chat_msg_add_tool_call_id`, and
  `render_chat_prompt_text`.
- Fixture: unit vectors for direct Anthropic tool schemas, OpenAI-compatible
  wrapped tools when accepted by shared parsing, tool_choice auto/any/none,
  assistant tool_use blocks with ordered object input, user tool_result blocks
  with string and content-array bodies, delimiter escaping, and prompt bytes.
- Comparator: Rust unit tests comparing tool schema lines, `ToolSchemaOrder`
  property order, parsed `ToolCall` fields and IDs, tool-result call IDs,
  rendered DSML, prompt bytes, and tool-choice suppression.
- Acceptance: Rust matches current C Anthropic tool schema and visible tool
  history semantics without live tool-result validation or protocol response
  emission.
- Drift policy: exact for schema/property order, tool IDs, tool/result body
  text, prompt bytes, and DSML rendering.
- Review gate: ask Claude to review tool_use input parsing, tool_result body
  handling, tool_choice behavior, call-id preservation, and prompt rendering.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.2c3c: Anthropic Live Tool Result Validation Surface

- Status: done
- Goal: port model-free Anthropic live continuation validation outputs:
  missing `tool_use_id` errors, live-state requirement flags, live tool-use ID
  collection, and visible live suffix rendering for trailing tool results.
- Oracle: C `anthropic_validate_tool_results`,
  `anthropic_prepare_live_continuation`, `anthropic_msg_is_tool_result_tail`,
  `responses_find_prior_call_msg`, `chat_msg_collect_tool_call_ids`,
  `render_live_tool_tail`, and related C unit vectors.
- Fixture: unit vectors for tool-result-only missing state, live-known
  tool-result-only continuation, stateless replay with prior assistant
  `tool_use`, role/order edge cases where `tool_use` appears before `role`,
  and live suffix text.
- Comparator: Rust unit tests using an explicit no-op/live stub state to
  compare validation category, requirement flags, collected tool-use IDs, and
  live-tail prompt bytes.
- Acceptance: Rust reports the same Anthropic live-continuation parser state as
  C without touching real server KV/tool-memory side effects, which remain
  assigned to M9.8.
- Drift policy: exact for error strings, live-state requirement flags,
  tool-use ID ordering/deduplication, and live-tail bytes.
- Review gate: ask Claude to review missing-state rejection, live-state flag
  assignment, prior tool_use detection, and live-tail construction.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server protocol parser modules, `.memory/status.md`.

### M9.3: Rust HTTP Skeleton And Model Metadata Endpoints

- Status: split into M9.3a, M9.3b, and M9.3c before implementation
- Goal: add a Rust server binary with HTTP framing, request routing, CORS
  behavior, `/v1/models`, and no-generation error paths.
- Oracle: current `ds4-server` socket behavior, M0.4 `models.json`,
  CORS/preflight unit tests, malformed HTTP/body handling, and server CLI flags.
- Fixture: M0.4 `models.json` plus local HTTP fixtures for OPTIONS, disabled
  CORS, bad routes, bad methods, bad JSON, missing model, and context limits.
- Comparator: local HTTP replay comparing status lines, headers, JSON bodies,
  CORS headers, and deterministic model metadata without a model load.
- Acceptance: Rust can start, answer `/v1/models`, reject unsupported requests
  with the same protocol shape, and pass local no-model HTTP replay.
- Drift policy: exact status/header/body fields except volatile date-like
  headers if introduced.
- Review gate: ask Claude to review socket lifetime, request framing,
  route/error coverage, and CORS header parity.
- Validation needed: roadmap/board diff and `git diff --check`.
- Owner path: Rust server binary/modules, `ds4-parity/`, `.memory/status.md`.

### M9.3a: HTTP Framing And CORS Response Surface

- Status: done
- Goal: port the byte-level HTTP request parser and response formatting helpers
  without adding a listener or model-backed request execution.
- Oracle: C `read_http_request`, `header_end`, `content_length`,
  `http_response`, `http_error`, `append_cors_headers`, and the CORS HTTP unit
  tests.
- Fixture: Rust unit vectors for request-line parsing, query stripping,
  content-length body extraction, malformed/incomplete requests, 200/204/400
  response bytes, CORS enabled/disabled headers, and JSON error body shape.
- Comparator: exact byte comparison for status line, header order,
  `Content-Length`, optional `Content-Type`, CORS headers, connection close,
  and response body.
- Acceptance: Rust can parse complete in-memory HTTP requests and format C-like
  HTTP responses/errors deterministically without opening sockets.
- Drift policy: exact bytes for supported helper output; unsupported malformed
  request cases compare by stable reject/accept category.
- Review gate: ask Claude to review header parsing bounds, content-length
  handling, query stripping, CORS header parity, and response byte order.
- Validation passed: targeted `cargo test -p ds4-gguf server_http -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  and `git diff --check`.
- Owner path: Rust server HTTP helper modules, `.memory/status.md`.

### M9.3b: Model Metadata And Route Dispatch Surface

- Status: done
- Goal: port no-model route dispatch for OPTIONS, `/v1/models`,
  `/v1/models/deepseek-v4-flash`, and unknown endpoints on top of the M9.3a
  HTTP helpers.
- Oracle: C `client_main` route branches for OPTIONS and model GET routes,
  `send_models`, `send_model`, `append_model_json_values`, and unknown endpoint
  `http_error` behavior.
- Fixture: M0.4 `models.json` plus synthetic HTTP requests for OPTIONS,
  CORS on/off, model list, single model, query-string variants, bad routes, and
  wrong methods.
- Comparator: route-handler tests comparing response status, headers, body
  bytes, CORS behavior, and deterministic model metadata for configured
  context/default-token values.
- Acceptance: Rust route handling can answer model metadata and preflight
  requests and reject unknown routes without a live socket or model load.
- Drift policy: exact for model JSON fields and route response bytes.
- Review gate: ask Claude to review route precedence, model metadata constants,
  CORS propagation, and unknown endpoint behavior.
- Validation passed: targeted `cargo test -p ds4-gguf server_http -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  and `git diff --check`.
- Owner path: Rust server HTTP route modules, `.memory/status.md`.

### M9.3c: No-Model Server Binary And Negative HTTP Replay

- Status: split into M9.3c1 and M9.3c2 before implementation
- Goal: preserve the original no-model server scope while separating
  route/parser error mapping from socket/process replay.
- Oracle: C `client_main`, current Rust M9.3a/M9.3b HTTP helpers, C
  `listen_on`, `configure_client_socket`, and relevant server CLI flags.
- Fixture: no source behavior change in this item.
- Comparator: documentation-only diff review; no Rust implementation changes.
- Acceptance: the split names one in-memory dispatch item and one socket replay
  item, each with a distinct oracle, fixture, comparator, and validation gate.
- Drift policy: no implementation or fixture drift allowed.
- Review gate: ask Claude to review that the split preserves the original
  M9.3c acceptance criteria without moving generation into either sub-item.
- Validation needed: `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/status.md`.

### M9.3c1: No-Model Generation Error Dispatcher

- Status: done
- Goal: add an in-memory no-model dispatcher that wires M9.3a/M9.3b helpers to
  generation-route parse/error handling without opening sockets or running
  generation.
- Oracle: C `client_main` negative paths for bad HTTP, bad JSON, unknown
  endpoint, missing messages/input/model/prompt, unsupported durable state,
  unsupported tool choice, and context-length errors.
- Fixture: local no-model HTTP replay cases for malformed HTTP, OPTIONS,
  model routes, bad routes, bad generation JSON, missing request fields,
  unsupported durable state, unsupported tool choice, and context-limit errors.
- Comparator: in-process Rust tests comparing status lines, headers, JSON
  bodies, CORS headers, and parser error text for each negative route.
- Acceptance: Rust can answer model metadata/preflight and return C-shaped
  negative responses for no-model generation requests without a live socket or
  model load.
- Drift policy: exact response bytes for covered negative paths.
- Review gate: ask Claude to review route-to-parser error mapping, unsupported
  generation handling, context-limit response body, and CORS propagation.
- Validation passed: targeted `cargo test -p ds4-gguf server_no_model -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  and `git diff --check`.
- Owner path: Rust server HTTP route modules, `.memory/status.md`.

### M9.3c2: No-Model Server Binary And Socket Replay

- Status: done
- Goal: add a Rust `ds4-server-rs` binary that binds/listens, exposes server
  CLI startup flags needed by M9.3, wires M9.3c1 dispatch to accepted sockets,
  and shuts down deterministically in local replay tests.
- Oracle: C `listen_on`, `configure_client_socket`, `parse_options`, startup
  defaults for `--host`, `--port`, `--ctx`, `--tokens`, and `--cors`, plus the
  M9.3c1 in-memory dispatcher.
- Fixture: local socket replay cases for malformed HTTP, OPTIONS, model routes,
  bad routes, bad generation JSON, missing request fields, unsupported durable
  state, unsupported tool choice, and context-limit errors.
- Comparator: local process/socket replay comparing status lines, headers, JSON
  bodies, CORS headers, CLI flag effects, and deterministic process shutdown.
- Acceptance: `ds4-server-rs` can start locally, answer model metadata, and
  return C-shaped negative responses for no-model generation requests through a
  real TCP socket.
- Drift policy: exact response bytes except process IDs, port choices, and any
  volatile timing fields if introduced.
- Review gate: ask Claude to review socket lifetime, read limits, blocking
  behavior, CLI flag parsing, shutdown behavior, and comparator coverage.
- Validation passed: local no-model HTTP comparator
  `cargo test -p ds4-gguf --test no_model_server -- --nocapture`,
  targeted Rust server tests
  `cargo test -p ds4-gguf --bin ds4-server-rs -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  and `git diff --check`.
- Owner path: Rust server binary/modules, `ds4-parity/`, `.memory/status.md`.

### M9.4: Non-Streaming Chat Completion Runtime

- Status: split into M9.4a, M9.4b, M9.4c, and M9.4d before implementation
- Goal: implement model-backed Rust `/v1/chat/completions` non-streaming
  generation for the M0.4 non-streaming OpenAI cases.
- Oracle: M0.4 `chat_basic`, `chat_thinking_disabled`, `chat_cache_seed`, and
  `chat_cache_continuation` current-C responses/traces.
- Fixture: M0.4 request JSON, response JSON, headers, and trace segments for
  non-streaming chat without tool-call output.
- Comparator: B300 request replay comparing normalized response JSON, usage
  fields, finish reasons, generated bytes, headers, and trace prompt/cache
  fields.
- Acceptance: Rust non-streaming responses match current C for the covered
  deterministic prompts and do not regress M8 CLI runtime comparators.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates only;
  generated text, finish reason, usage counts, and cache fields are exact.
- Review gate: ask Claude to review server/session ownership, request-to-runtime
  option mapping, response JSON shape, usage accounting, and trace fields.
- Validation needed: B300 non-streaming comparator with negative tests, local
  parser tests, `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust server runtime path, `ds4-parity/`, `.memory/status.md`.

### M9.4a: Model-Backed Server Runtime Boundary

- Status: done
- Goal: give the Rust server a model-backed runtime boundary that can load
  `ds4flash.gguf`, own the engine/session lifetime from the server binary, and
  keep M9.3 no-model route/error behavior available while generation remains
  explicitly scoped to later M9.4 items.
- Oracle: C `parse_options`, `client_main`, `listen_on`,
  `configure_client_socket`, Rust `ds4-engine` model-open/session APIs, M9.3
  socket replay behavior, and M0.4 B300 server startup logs.
- Fixture: M9.3 no-model socket fixtures, M0.4 `models.json`, bad-generation
  requests, B300 model path/SHA records, and current-C server startup command.
- Comparator: local no-model replay must still pass through the server
  boundary, and a B300 smoke command must start the Rust server with `-m
  /workspace/ds4/ds4flash.gguf`, answer `OPTIONS`/model metadata/error routes,
  and reject valid chat generation with the planned-not-implemented response.
- Acceptance: model loading and socket ownership live in the Rust runtime
  crate without introducing a dependency cycle, CLI startup flags match the
  C server subset needed by M0.4, tokenizer-backed context-limit checks are
  available when a model is loaded, and valid non-streaming generation is not
  silently handled by the no-model path.
- Drift policy: startup timing and log rates may differ; model identity,
  bind behavior, HTTP error bodies, model metadata, and prompt-token counts are
  exact.
- Review gate: ask Claude to review crate ownership, engine lifetime, CLI flag
  parity, prompt-token counting, socket shutdown behavior, and no-model
  regression coverage.
- Validation passed: local no-model socket replay
  `cargo test -p ds4-gguf server_no_model -- --nocapture`, targeted runtime
  tests `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  full `cargo test --workspace`, B300 model-load smoke replay from
  `/workspace/ds4-m94a` using `/workspace/ds4/ds4flash.gguf`, B300 targeted
  runtime tests, `cargo fmt --all -- --check`, and `git diff --check`.
- Owner path: `rust/ds4-engine`, Rust server binary/modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.4b: OpenAI Non-Streaming Response And Usage Builder

- Status: done
- Goal: add pure Rust helpers for non-streaming OpenAI chat-completion response
  JSON, HTTP headers, usage accounting, finish reasons, and trace-ready
  request metadata without running the model.
- Oracle: C `openai_completion_json`, `append_openai_usage_json`,
  context-length helpers, M0.4 `chat_basic`/`chat_thinking_disabled` response
  JSON, and M0.4 non-streaming headers.
- Fixture: normalized M0.4 non-streaming response JSON for `chat_basic` and
  `chat_thinking_disabled`, header files, usage/cache detail vectors, and
  parser output from M9.2.
- Comparator: local unit tests compare normalized JSON and headers while
  allowing only IDs and timestamps to be injected; cache-read/cache-write usage
  values are supplied as explicit inputs.
- Acceptance: the builder emits C-shaped non-streaming chat JSON and HTTP
  responses for fixed generated text, usage counts, finish reasons, model
  names, cache details, and error cases, with no model runtime or cache policy
  embedded in the formatting layer.
- Drift policy: IDs and timestamps are injected or normalized only; JSON field
  names, ordering, finish reasons, usage counts, model names, and headers are
  exact.
- Review gate: ask Claude to review JSON shape, usage accounting,
  cache-detail clamping, header parity, and separation from runtime/cache
  policy.
- Validation passed: targeted response-builder tests
  `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server formatting modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.4c: No-Cache Non-Streaming Chat Generation Replay

- Status: done
- Goal: route model-backed non-streaming `/v1/chat/completions` requests
  through the Rust engine for the no-cache M0.4 cases, returning the M9.4b
  response surface for `chat_basic` and `chat_thinking_disabled`.
- Oracle: M0.4 `chat_basic` and `chat_thinking_disabled` request/response
  JSON, headers, trace segments, generated text, usage counts, and C
  request-to-runtime option mapping.
- Fixture: M0.4 `chat_basic.json`, `chat_thinking_disabled.json`,
  corresponding response/header/trace files, B300 model path/SHA records, and
  local parser vectors for thinking-disabled aliases.
- Comparator: B300 replay against the Rust server compares normalized response
  JSON, headers, generated bytes, finish reason, prompt/completion/total token
  counts, sampling options, and trace-rendered prompt fields.
- Acceptance: Rust matches current C for the two no-cache non-streaming
  OpenAI chat fixtures, keeps unsupported tools/streaming out of this path,
  and does not regress existing M8 CLI model-backed comparators.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates
  only; generated text, finish reason, prompt text, prompt tokens,
  completion tokens, and usage fields are exact.
- Review gate: ask Claude to review request-to-runtime option mapping,
  prompt-token accounting, thinking-disabled behavior, response assembly,
  unsupported-route fallbacks, and B300 comparator normalization.
- Validation passed: targeted runtime tests
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  full `cargo test --workspace`, B300 no-cache normalized replay for
  `chat_basic` and `chat_thinking_disabled` from `/workspace/ds4-m94c` using
  `/workspace/ds4/ds4flash.gguf`, B300 trace prompt/generated-field checks,
  `cargo fmt --all -- --check`, and `git diff --check`.
- Owner path: Rust server runtime path, `ds4-parity/`, `.memory/status.md`.

### M9.4d: Memory-Token Cache Seed And Continuation Replay

- Status: done
- Goal: add the in-memory server session/cache behavior needed for M0.4
  `chat_cache_seed` and `chat_cache_continuation`, including cache read/write
  usage fields and trace cache decisions.
- Oracle: M0.4 `chat_cache_seed` and `chat_cache_continuation`
  request/response JSON, headers, trace cache sections, server log cache
  messages, and C live-session common-prefix behavior.
- Fixture: M0.4 cache seed/continuation request JSON, response/header/trace
  files, server log cache lines, and B300 replay command records.
- Comparator: B300 sequential replay against one Rust server process compares
  normalized responses, generated bytes, finish reasons, usage/cache detail
  fields, trace prompt/cache sections, and live cache-source decisions.
- Acceptance: Rust preserves the live session across the seed and continuation
  requests, reports the same cached and cache-write token counts as current C,
  produces matching generated text and usage, and leaves disk KV/tool-memory
  behavior to M9.8.
- Drift policy: normalize IDs, timestamps, paths, startup timing, and token
  rates only; cache source, cached token counts, cache-write counts,
  generated text, finish reason, and prompt trace fields are exact.
- Review gate: ask Claude to review session reuse, common-prefix calculation,
  cache usage accounting, trace cache fields, request ordering assumptions, and
  M9.8 boundary separation.
- Validation passed: targeted runtime tests
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  full `cargo test --workspace`, B300 sequential cache replay for
  `chat_cache_seed` and `chat_cache_continuation` from `/workspace/ds4-m94d`
  using `/workspace/ds4/ds4flash.gguf`, B300 trace cache-field checks,
  `cargo fmt --all -- --check`, and `git diff --check`.
- Owner path: Rust server runtime/cache path, `ds4-parity/`,
  `.memory/status.md`.

### M9.5: Streaming Chat Completion SSE Surface

- Status: split into M9.5a and M9.5b before implementation
- Goal: implement Rust streaming `/v1/chat/completions` SSE framing and usage
  reporting for the M0.4 stream case.
- Oracle: M0.4 `chat_stream.sse`, stream headers, stream trace, and unit tests
  for UTF-8/stop-list streaming holds.
- Fixture: M0.4 `chat_stream.json`, `chat_stream.sse`, headers, trace segment,
  and unit vectors for partial UTF-8 and stop-text handling.
- Comparator: B300 SSE replay comparing event order, delta payloads, finish
  chunk, usage chunk, headers, generated bytes, and normalized timing/progress.
- Acceptance: Rust emits the same SSE event sequence and usage semantics for
  the deterministic stream fixture and preserves model-visible text boundaries.
- Drift policy: normalize IDs, timestamps, token rates, and chunk timing only;
  event names/order and text deltas are exact.
- Review gate: ask Claude to review SSE flush behavior, partial UTF-8 handling,
  stop-list trimming, client disconnect paths, and usage chunk parity.
- Validation needed: B300 streaming comparator with negative tests,
  decode-policy streaming hold tests, `cargo test --workspace`, and
  `git diff --check`.
- Owner path: Rust server streaming path, `ds4-parity/`, `.memory/status.md`.

### M9.5a: OpenAI Chat SSE Formatter And Header Builder

- Status: done
- Goal: add pure Rust helpers for OpenAI chat-completion SSE headers, role
  chunk, content delta chunks, final finish chunk, optional usage chunk, and
  `[DONE]` terminator without running the model.
- Oracle: C `sse_headers`, `sse_chunk`, `sse_usage_chunk`, `sse_done`,
  `append_openai_usage_json`, M0.4 `chat_stream.sse`, and M0.4 stream headers.
- Fixture: M0.4 `chat_stream.sse`, `chat_stream.headers.txt`, injected
  ID/timestamp values, the observed stream deltas `stream` and ` baseline`,
  finish reason `stop`, and usage/cache counts `11/2/13` with cache `0/11`.
- Comparator: local unit tests compare exact SSE body and headers for injected
  ID/timestamp/delta/usage inputs, plus negative checks for usage omission and
  JSON escaping.
- Acceptance: the formatter emits the current-C byte sequence for the M0.4
  stream fixture with no model runtime, socket lifetime, cache policy, or token
  decoding embedded in the formatting layer.
- Drift policy: IDs and timestamps are injected or normalized only; event
  order, JSON field order, content deltas, finish reason, usage fields,
  headers, blank lines, and `[DONE]` are exact.
- Review gate: ask Claude to review SSE JSON/header shape, usage accounting,
  escaping, event ordering, and separation from runtime/cache policy.
- Validation passed: targeted SSE formatter tests
  `cargo test -p ds4-gguf server_response -- --nocapture`, decode-policy
  streaming hold tests `cargo test -p ds4-gguf decode_policy -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server formatting modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.5b: Model-Backed Streaming Chat Replay

- Status: done
- Goal: route supported streaming `/v1/chat/completions` requests through the
  Rust server runtime for the M0.4 `chat_stream` fixture, preserving per-token
  content deltas and usage reporting through the M9.5a formatter.
- Oracle: M0.4 `chat_stream.json`, `chat_stream.sse`, stream headers, stream
  trace segment, C token-delta behavior, existing decode-policy tests for
  UTF-8/stop-list streaming holds, and B300 model-backed replay.
- Fixture: M0.4 stream request/response/header/trace files, B300 model
  path/SHA records, request-to-runtime sampling controls, and generated token
  deltas `stream` and ` baseline`.
- Comparator: B300 replay against one Rust server process compares normalized
  SSE body, headers, generated bytes, finish reason, usage/cache detail fields,
  trace prompt/cache fields, and per-token content delta boundaries.
- Acceptance: Rust matches current C for the deterministic OpenAI stream
  fixture, keeps tools/thinking/stop-list requests outside this path until
  their roadmap items, and leaves true disk-KV/tool-memory behavior to M9.8.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates
  only; SSE event order, content delta boundaries, finish reason, usage fields,
  prompt text, prompt tokens, and cache fields are exact.
- Review gate: ask Claude to review server streaming routing, token chunk
  capture, SSE flush/write behavior, unsupported-route boundaries,
  decode-policy coverage, usage accounting, and B300 comparator normalization.
- Validation passed: targeted runtime tests
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted SSE formatter tests `cargo test -p ds4-gguf server_response -- --nocapture`,
  decode-policy streaming hold tests
  `cargo test -p ds4-gguf decode_policy -- --nocapture`, full
  `cargo test --workspace`, B300 streaming replay for `chat_stream` from
  `/workspace/ds4-m95b` using `/workspace/ds4/ds4flash.gguf`, B300 trace
  streaming-field checks, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: Rust server runtime path, `ds4-parity/`, `.memory/status.md`.

### M9.6: Tool-Call And DSML Server Surface

- Status: split
- Goal: split the remaining OpenAI tool-call server work into response
  formatting, non-streaming model-backed replay, streaming tool-call deltas,
  and quality-regression wiring.
- Source evidence needed: M5.6 server DSML parser/formatter, M9.2b OpenAI tool
  schema prompt rendering, M0.4 `chat_tool_call` fixtures/traces, M9.5 SSE
  formatter, and `test_tool_call_quality`.
- Oracle: documentation-only split preserving the existing M9.6 oracle set
  while assigning one comparator and review gate per implementation surface.
- Comparator: roadmap/TODO/status diff shows no behavior change and assigns
  exact oracle, fixture, comparator, acceptance, and drift policy per sub-item.
- Acceptance: M9.6a-d are documented before implementation starts, with M9.6a
  active first because it is pure response JSON and unblocks runtime replay.
- Drift policy: docs-only split must not relax exact tool names, arguments,
  finish reasons, prompt text, trace records, or SSE bytes.
- Review gate: ask Claude to review whether the split isolates independent
  oracle surfaces and avoids reimplementing already completed M5.6/M9.2b work.
- Validation needed: `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M9.6a: OpenAI Tool-Call Response Formatter

- Status: done
- Goal: add a pure formatter for final OpenAI chat responses whose assistant
  message contains parsed `tool_calls`, including optional reasoning content
  and usage details.
- Source evidence needed: M0.4 `chat_tool_call` response JSON, M5.6
  generated-message parser unit vectors, and existing M9.4 response/usage
  formatting tests.
- Oracle: exact M0.4 `chat_tool_call` response shape with injected
  outer ID/timestamp/call IDs and parser-produced `DsmlJsonCall` values.
- Fixture: M0.4 `chat_tool_call` response plus model-free vectors for one
  call, multiple calls, explicit IDs, generated IDs, ordered arguments, and
  escaped tool names/arguments.
- Comparator: exact JSON byte comparison with normalized outer IDs,
  timestamps, and generated call IDs.
- Acceptance: formatter emits C-compatible `tool_calls` arrays, empty
  assistant content, `finish_reason:"tool_calls"`, stable argument strings,
  and unchanged cache usage fields without touching model execution.
- Drift policy: response object field order, tool-call field order, argument
  strings, finish reason, and usage fields are exact; injected IDs/timestamps
  are the only normalized values.
- Review gate: ask Claude to review response JSON field order, call-ID
  injection, escaping, and separation from runtime DSML parsing.
- Validation passed: targeted response formatter tests
  `cargo test -p ds4-gguf server_response -- --nocapture`, DSML parser tests
  `cargo test -p ds4-gguf dsml -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`; Claude review returned no blockers.
- Owner path: `rust/ds4-gguf/src/server_response.rs`,
  `rust/ds4-gguf/src/dsml.rs`, `.memory/status.md`.

### M9.6b: Model-Backed Tool-Call Replay

- Status: done
- Goal: route supported non-streaming OpenAI tool requests through the Rust
  server runtime, parse generated DSML with the existing generated-message
  parser, and emit the M9.6a tool-call response shape.
- Source evidence needed: M0.4 `chat_tool_call` request/response/trace,
  M9.2b prompt-render tests, M5.6 parser tests, and B300 model-backed replay.
- Oracle: deterministic M0.4 `chat_tool_call` replay on B300.
- Fixture: M0.4 `chat_tool_call.json`, response JSON, trace segment with
  rendered tool prompt, generated DSML, prompt/completion token counts, and
  `tool_call[0]` trace record.
- Comparator: B300 replay compares normalized response ID/timestamp/call ID,
  exact tool name, arguments, finish reason, usage/cache fields, rendered
  prompt, generated DSML parse result, and trace tool records.
- Acceptance: the Rust server no longer rejects supported non-streaming tool
  chat requests, matches the deterministic C fixture, and still rejects
  streaming tool-call requests until M9.6c.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates
  only; prompt text, tool schema order, generated DSML parse output, finish
  reason, and trace records are exact.
- Review gate: ask Claude to review runtime routing, parser finish mapping,
  trace records, unsupported-route boundaries, and B300 comparator
  normalization.
- Validation passed: B300 tool-call comparator for `chat_tool_call` using the
  raw M0.4 trace request as prompt oracle, B300 targeted runtime tests
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  local targeted runtime tests with the same command, targeted response tests
  `cargo test -p ds4-gguf server_response -- --nocapture`, targeted DSML
  parser tests `cargo test -p ds4-gguf dsml -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`; Claude review returned no blockers.
- Owner path: Rust server runtime path, `ds4-parity/`, `.memory/status.md`.

### M9.6c: Streaming Tool-Call Deltas

- Status: split
- Goal: split OpenAI streaming tool-call work into pure SSE event formatting,
  incremental DSML-to-delta translation, and model-backed runtime replay.
- Source evidence needed: C tool-call streaming unit tests,
  `sse_chat_tool_call_start_delta`, `sse_chat_tool_call_args_delta_n`,
  `openai_sse_stream_update`, `openai_tool_stream_update`, M5.6 DSML parser
  vectors, M9.5a SSE formatting, and M9.6b model-backed tool replay.
- Oracle: documentation-only split preserving the M9.6c oracle set while
  assigning byte formatting, parser state, and runtime replay to child items.
- Comparator: roadmap/TODO/status diff shows no behavior change and assigns
  exact oracle, fixture, comparator, acceptance, and drift policy per sub-item.
- Acceptance: M9.6c1-c3 are documented before implementation starts, with
  M9.6c1 active first because byte-stable SSE formatting unblocks the parser
  and runtime replay.
- Drift policy: docs-only split must not relax event order, argument
  fragments, finish reasons, prompt text, generated DSML, trace records, usage
  fields, or `[DONE]` bytes.
- Review gate: ask Claude to review whether the split isolates independent
  oracle surfaces and preserves the original streaming tool-call parity
  requirements.
- Validation needed: `git diff --check`.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M9.6c1: Tool-Call SSE Event Formatter

- Status: complete
- Goal: add pure OpenAI chat SSE helpers for streamed `tool_calls` deltas:
  role chunk, tool-call start delta, argument-fragment deltas, full-call
  fallback delta, finish chunk, optional usage chunk, and `[DONE]`.
- Source evidence needed: C helpers `sse_chat_tool_call_start_delta`,
  `sse_chat_tool_call_args_delta_n`, `append_tool_call_deltas_json`,
  `openai_sse_finish_live`, and M9.5a non-tool SSE formatting behavior.
- Oracle: exact C-shaped SSE bytes for model-free tool-call streaming event
  cases with injected IDs/timestamps.
- Fixture: one call, multiple calls, explicit IDs, generated IDs, escaped
  names, argument fragments containing JSON escapes, full-call fallback deltas,
  optional usage, and stream headers.
- Comparator: exact SSE byte comparison with injected chat ID/timestamp/call
  IDs and pre-split argument fragments.
- Acceptance: formatter emits C-compatible event order and JSON field order for
  tool-call deltas without owning DSML parsing or model runtime routing.
- Drift policy: event names, object field order, blank lines, usage fields, and
  `[DONE]` bytes are exact; injected IDs and timestamps are the only normalized
  values.
- Review gate: ask Claude to review SSE byte shape, field ordering, generated
  ID fallback, escaping, usage chunk placement, and separation from parser and
  runtime policy.
- Validation passed: targeted `cargo test -p ds4-gguf server_response -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: `rust/ds4-gguf/src/server_response.rs`, `.memory/status.md`.

### M9.6c2: Incremental DSML Tool-Call Stream Translator

- Status: complete
- Goal: translate incremental generated DSML bytes into the M9.6c1 OpenAI
  tool-call start/argument delta events while holding incomplete tags,
  parameter close sentinels, DSML entities, and UTF-8 tails.
- Source evidence needed: C `openai_sse_stream_update`,
  `openai_tool_stream_update`, `tool_param_value_stream_safe_len`, partial-tool
  unit tests, and M5.6 generated-message parser behavior for completed DSML.
- Oracle: model-free C-compatible event streams for partial DSML chunk
  schedules.
- Fixture: chunk schedules for one call, multiple calls, partial invoke tags,
  partial parameter tags, partial `</｜DSML｜parameter>` sentinels, partial
  `&lt;` entities, split UTF-8, raw JSON arguments, malformed/recoverable
  tails, and complete fallback blocks.
- Comparator: model-free event-by-event SSE comparison for every chunk
  schedule, including held bytes and emitted argument fragments.
- Acceptance: streamed argument fragments match C for partial DSML schedules,
  completed calls still parse to the same final `DsmlJsonCall` values, and
  non-tool streaming behavior from M9.5 remains unchanged.
- Drift policy: normalize generated call IDs only; emitted/held argument
  bytes, event order, finish reason, and malformed-tail behavior are exact.
- Review gate: ask Claude to review state transitions, partial-hold logic,
  UTF-8/entity boundaries, malformed tail recovery, and non-tool streaming
  regression coverage.
- Validation passed: targeted `cargo test -p ds4-gguf server_response -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- Owner path: `rust/ds4-gguf/src/server_response.rs`,
  `rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs`, `.memory/status.md`.

### M9.6c3: Model-Backed Streaming Tool-Call Replay

- Status: complete
- Goal: route supported streaming OpenAI tool chat requests through the Rust
  server runtime, feed per-token bytes through the M9.6c2 translator, and emit
  the M9.6c1 SSE response shape.
- Source evidence needed: B300 model-backed replay of the M0.4
  `chat_tool_call` request with `"stream":true`, C streaming unit-test event
  order, M9.6b prompt/trace behavior, and M9.5b non-tool streaming replay
  behavior.
- Oracle: B300 streaming replay for deterministic tool-call generation.
- Fixture: raw M0.4 trace request with streaming enabled, normalized SSE body,
  stream headers, generated DSML, trace fields, usage/cache fields, and B300
  model identity.
- Comparator: B300 replay compares normalized chat ID, timestamp, and
  generated call ID while checking exact tool name, argument fragments, finish
  reason, usage/cache fields, rendered prompt, generated DSML, trace records,
  and `[DONE]` bytes.
- Acceptance: Rust streams deterministic tool-call output through the HTTP
  runtime, preserves M9.6b non-streaming behavior, and keeps thinking/stop-list
  requests outside this path until their own roadmap items.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates
  only; event order, argument fragments, finish reason, usage fields, prompt
  text, generated DSML, and trace tool records are exact.
- Review gate: ask Claude to review runtime routing, write/flush behavior,
  translator integration, unsupported-route boundaries, trace fields, and B300
  comparator normalization.
- Validation passed: B300 model-backed streaming tool-call replay on
  `hou2-prod1` pod `ds4-rust-port-b300` with snapshot `/workspace/ds4-m96c3`
  and model `/workspace/ds4/ds4flash.gguf`; SSE parsed as role, tool start for
  `list_files`, argument fragments reassembling to `{"path":"."}`, finish
  `tool_calls`, usage, and `[DONE]`; trace recorded streaming/tools enabled,
  DSML start/end, `generated_tokens: 42`, and parsed tool call arguments.
  Local checks passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Owner path: `rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs`,
  `ds4-parity/`, `.memory/status.md`.

### M9.6d: Tool-Call Quality Parity Hook

- Status: complete
- Goal: connect the Rust server/runtime tool-call path to the existing
  tool-call quality regression surface after non-streaming and streaming
  behavior is in place.
- Source evidence needed: `test_tool_call_quality`, current C quality
  thresholds/logs, M0.4 tool-call trace fixtures, and the M9.6b/M9.6c
  response comparators.
- Oracle: current C tool-call quality pass/fail categories and raw artifacts.
- Fixture: quality-run command lines, model identity, seed/sampling controls,
  expected tool-call success categories, and preserved raw outputs for
  failures.
- Comparator: Rust quality run reports the same pass/fail categories and
  stores raw tool-call outputs for any drift investigation.
- Acceptance: `./ds4_test --tool-call-quality` has a Rust-runtime path or a
  documented equivalent runner with exact model/seed controls and artifact
  capture.
- Drift policy: no threshold changes without recording C and Rust raw outputs;
  tool names, argument strings, and failure categories remain comparable.
- Review gate: ask Claude to review quality-run wiring, artifact preservation,
  and whether the comparator can distinguish runtime regressions from model
  nondeterminism.
- Checks passed: added `ds4-parity/run_tool_call_quality.py` as the documented
  Rust-runtime equivalent runner for `./ds4_test --tool-call-quality`; B300
  run on `hou2-prod1` pod `ds4-rust-port-b300` with snapshot
  `/workspace/ds4-m96d` and model `/workspace/ds4/ds4flash.gguf` passed both
  fast and exact/`--quality` cases with category `ok`, HTTP 200, tool
  `list_files`, arguments `{"path":"."}`, and finish `tool_calls`; artifacts
  are under `/tmp/ds4-m96d-tool-call-quality`. Local checks passed for
  `python3 -m py_compile ds4-parity/run_tool_call_quality.py`,
  `python3 ds4-parity/run_tool_call_quality.py --self-test`,
  `ruff format --check ds4-parity/run_tool_call_quality.py`,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Owner path: `tests/`, Rust server runtime path, `ds4-parity/`,
  `.memory/status.md`.

### M9.7: Responses And Anthropic Protocol Surface

- Status: done via M9.7a and M9.7b
- Goal: port Responses and Anthropic request/response/stream protocol surfaces
  that share the server request and tool-memory core.
- Oracle: C Responses/Anthropic parsers, live-tail/tool-output validation
  tests, streaming event builders, and usage reporting tests in `ds4_server.c`.
- Fixture: unit vectors for Responses namespace/tool_search schemas, Responses
  reasoning/tool outputs, Anthropic content blocks/system filtering, Anthropic
  tool use/results, live stream deltas, and cache usage fields.
- Comparator: model-free protocol dump/tests comparing normalized request
  semantics, response/event JSON, usage fields, live-tail requirements, and
  tool-output validation categories.
- Acceptance: Rust matches current C protocol semantics for Responses and
  Anthropic without needing the HTTP runtime to be model-backed.
- Drift policy: normalize random IDs and timestamps; event names/order, usage
  fields, validation categories, and rendered prompt/live-tail text are exact.
- Review gate: ask Claude to review protocol live state, reasoning replay,
  namespace schema restoration, and Anthropic tool-result ID validation.
- Validation needed: protocol unit/comparator tests, `cargo test --workspace`,
  and `git diff --check`.
- Owner path: Rust server protocol modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.7a: Responses And Anthropic Final Response Formatters

- Status: done
- Goal: port model-free non-streaming Responses and Anthropic response body and
  HTTP formatting for assistant text, reasoning, tool calls, finish mapping,
  and cache usage fields.
- Oracle: C `responses_final_response`, `anthropic_final_response`,
  `responses_append_function_call_item`, `append_anthropic_content`,
  `append_responses_usage_json`, `append_anthropic_usage_json`, and associated
  server unit vectors.
- Fixture: unit vectors with assistant text, reasoning summaries, empty and
  reasoning-only Anthropic content, function calls, namespace-restored
  Responses calls, Responses `tool_search_call` output, plain functions named
  `tool_search`, finish reasons, and cache read/write usage details.
- Comparator: Rust formatter tests compare exact JSON/HTTP bodies after
  injecting deterministic IDs and timestamps for fields that C randomizes.
- Acceptance: Rust exposes final Responses and Anthropic response formatters
  that match C protocol semantics without opening sockets, loading a model, or
  routing through the runtime.
- Drift policy: random IDs and timestamps are injected in tests; item/event
  names, finish/status mappings, usage fields, namespace restoration,
  `tool_search` discrimination, and empty-content behavior are exact.
- Review gate: ask Claude to review formatter parity against C helpers,
  especially cache usage clamping, Responses namespace restoration,
  `tool_search` discrimination, and Anthropic empty content.
- Validation passed: targeted `cargo test -p ds4-gguf server_response -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and non-interactive Claude review with no blockers.
- Owner path: `rust/ds4-gguf/src/server_response.rs`,
  `rust/ds4-gguf/src/lib.rs`, `.memory/status.md`.

### M9.7b: Responses And Anthropic Streaming Event Builders

- Status: done
- Goal: port model-free Responses and Anthropic SSE event/body builders for
  reasoning deltas, output text deltas, tool-call lifecycle events, terminal
  usage events, and Anthropic content-block deltas.
- Oracle: C `responses_sse_*`, `anthropic_sse_*`, `responses_sse_completed`,
  `anthropic_sse_finish_live`, and streaming server unit vectors for live
  reasoning, text, tool use, cache usage, and terminal status.
- Fixture: unit vectors for Responses created/output_item/content_part/
  function_call events, Anthropic message/content_block/delta events, partial
  DSML tool arguments, reasoning close behavior, terminal length/error/tool
  statuses, and cache usage fields.
- Comparator: model-free Rust SSE formatter tests comparing event order, event
  names, JSON payload fields, sequence numbers where applicable, and terminal
  bodies after deterministic ID/timestamp injection.
- Acceptance: Rust can format Responses and Anthropic streaming protocol events
  equivalent to C without needing model-backed runtime integration.
- Drift policy: event ordering, lifecycle names, status fields, usage fields,
  and tool argument JSON are exact; random IDs and timestamps are injected.
- Review gate: ask Claude to review lifecycle ordering and hidden reasoning
  replay semantics before runtime integration.
- Validation passed: targeted `cargo test -p ds4-gguf server_response -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, NUL scan, and non-interactive Claude review with no
  blockers.
- Owner path: `rust/ds4-gguf/src/server_response.rs`,
  `rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs`, `.memory/status.md`.

### M9.8a: Server Cache/KV/Tool-Memory Work Item Split

- Status: done
- Goal: split the broad server cache, KV restore, continued-frontier, eviction,
  and tool-memory work into implementation stages with separate oracles.
- Oracle: M9.8 source map against `ds4_server.c` helpers around
  `tool_memory_attach_to_messages`, live continuation state, KV policy helpers,
  KV tool-map trailer restore, and request-path cache accounting.
- Fixture: roadmap/TODO state only.
- Comparator: review the resulting M9.8b-M9.8f stages for one tangible
  behavior boundary, explicit oracle, comparator, acceptance, drift policy, and
  validation gate per stage.
- Acceptance: M9.8 has no catch-all implementation item; each remaining slice
  can be reviewed, validated, committed, and pushed independently.
- Drift policy: docs/state-only; no runtime behavior changes.
- Review gate: ask Claude to confirm the split preserves coverage and does not
  hide model-backed validation under model-free stages.
- Validation passed: `git diff --check`, docs inspection, and non-interactive
  Claude review with no blockers.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M9.8b: Tool-Memory Replay Core

- Status: done
- Goal: port exact sampled DSML tool-call memory replay for OpenAI/Responses
  and Anthropic histories before prompt rendering.
- Oracle: C tests `test_tool_memory_replays_sampled_dsml`,
  `test_anthropic_tool_memory_replays_sampled_dsml`, and
  `test_tool_memory_max_ids_prunes_oldest`, plus
  `tool_memory_attach_to_messages`.
- Fixture: generated/canonical tool-call histories whose JSON argument order
  differs from sampled DSML, duplicate DSML blocks shared by multiple ids, and
  max-entry pruning vectors.
- Comparator: model-free Rust tests comparing replay stats, raw DSML prompt
  bytes, call-id lookup behavior, canonical fallback, and pruning order.
- Acceptance: Rust prompt rendering can replay exact sampled DSML by call id
  when memory is available and falls back to canonical rendering only when ids
  are missing or span distinct sampled DSML blocks.
- Drift policy: sampled DSML bytes and parameter order are exact; random tool
  ids are injected; memory size accounting only needs to preserve pruning
  outcomes.
- Review gate: ask Claude to review raw-vs-canonical DSML selection and missing
  id stats.
- Validation passed: targeted `cargo test -p ds4-gguf tool_memory -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, NUL scan over touched Rust files, and non-interactive
  Claude review with no blockers.
- Validation gate: targeted tool-memory tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.
- Owner path: Rust server chat/prompt/tool-memory modules and
  `.memory/status.md`.

### M9.8c: Live Continuation And Visible-Prefix State

- Status: done
- Goal: port server live continuation matching for Responses, Anthropic, and
  hidden-reasoning visible-prefix replay.
- Oracle: C tests `test_anthropic_live_tail_renders_tool_results_only`,
  `test_responses_live_tail_renders_tool_outputs_only`, and source helpers
  `responses_live_continuation_prompt`, `anthropic_live_continuation_prompt`,
  `responses_live_visible_prefix_prompt`, and
  `thinking_live_visible_prefix_prompt`.
- Fixture: tool-result-only follow-up requests, visible replay that omits
  hidden reasoning, and mismatched live token/call-id cases.
- Comparator: model-free Rust tests comparing live suffix text, required live
  state flags, visible-prefix prompt construction, call-id matching, and
  mismatch fallbacks.
- Acceptance: Rust request parsing and runtime state can distinguish direct live
  protocol continuation from replay/prefix matching and can produce C-shaped
  continuation prompts.
- Drift policy: rendered suffix/prefix bytes and call-id matching are exact;
  live token frontiers are injected in tests.
- Review gate: ask Claude to review direct-continuation versus prefix-replay
  separation.
- Validation passed: targeted `cargo test -p ds4-gguf server_chat -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, NUL scan over touched Rust files, and non-interactive
  Claude review with no blockers.
- Validation gate: targeted live-continuation tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.
- Owner path: Rust server chat/runtime state modules and `.memory/status.md`.

### M9.8d: Disk-KV Policy Completion

- Status: done
- Goal: complete Rust disk-KV policy parity for store boundaries, continued
  checkpoints, file-size budgeting, lookup, and eviction.
- Oracle: C tests for `kv_cache_store_len`, `kv_cache_chat_anchor_pos`,
  `kv_cache_continued_store_target`, cold-store suppression/restoration,
  `kv_cache_file_size_fits`, `kv_cache_find_text_prefix`, and eviction scoring,
  plus `ds4-parity/check_kv_policy_dump.py`.
- Fixture: M7.2 KV policy oracle, M0.5 KV header rows, synthetic text-prefix
  entries, protected SHA sets, and budget-edge cases.
- Comparator: Rust model-free tests and/or parity script output comparing
  reason codes, key kinds, trimmed boundaries, continued targets, fit decisions,
  longest text-prefix selection, protected entries, and eviction score order.
- Acceptance: Rust KV policy decisions match current C without requiring CUDA
  or model execution.
- Drift policy: reason codes, ext flags, token counts, and eviction order are
  exact; timestamps and paths are normalized.
- Review gate: ask Claude to review boundary trimming, budget math, and
  eviction protection.
- Validation passed: targeted `cargo test -p ds4-gguf kv_policy -- --nocapture`,
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test`, `python3
  ds4-parity/compare_kv_policy.py --negative-test`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, NUL scan
  over touched Rust files, and non-interactive Claude review with no blockers.
- Validation gate: targeted KV policy tests, `python3
  ds4-parity/check_kv_policy_dump.py --negative-test`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: `rust/ds4-gguf/src/kv_policy.rs`, `ds4-parity/`,
  `.memory/status.md`.

### M9.8e: KV Tool-Map Trailer Restore

- Status: done
- Goal: port KV tool-map trailer serialization/decoding and restore exact DSML
  tool memory before prompt rendering.
- Oracle: C test `test_kv_tool_map_restores_before_prompt_render`,
  `kv_tool_map_write`, `kv_tool_map_load_from_pos`, and
  `kv_cache_restore_tool_memory_for_messages`.
- Fixture: KVC files with `KV_EXT_TOOL_MAP`, wanted call-id filters, malformed
  tool-map trailers, and assistant histories that would otherwise render
  canonical JSON argument order.
- Comparator: Rust tests comparing decoded trailer entries, count/length error
  behavior, wanted-id filtering, memory source stats, and prompt bytes after
  restore.
- Acceptance: Rust restores tool memory from disk before prompt rendering so
  cached histories preserve sampled DSML bytes across process restarts.
- Drift policy: trailer binary layout, id limits, DSML length limits, and prompt
  bytes are exact; unreadable or malformed trailers stop at C-equivalent partial
  restore points.
- Review gate: ask Claude to review binary bounds checks and restore-before-
  render ordering.
- Validation passed: targeted `cargo test -p ds4-gguf tool_memory --
  --nocapture`, targeted `cargo test -p ds4-gguf tool_map -- --nocapture`,
  `python3 ds4-parity/compare_kv_trailer.py --negative-test`, `cargo test -p
  ds4-gguf --bin ds4-kv-trailer-dump-rs`, full `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, NUL scan over touched Rust
  files, and non-interactive Claude review with no blockers.
- Validation gate: targeted KV tool-map tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.
- Owner path: `rust/ds4-gguf/src/kv_policy.rs`, Rust tool-memory module,
  `.memory/status.md`.

### M9.8f1: Runtime Cache/KV Integration Split

- Status: done
- Goal: split the broad runtime cache/KV request-path work into reviewable
  stages with one oracle and comparator boundary per commit.
- Oracle: M9.8f source map against `ds4_server.c` helpers for request-path
  cache loading, session payload restore, continued store, tool-memory replay,
  cache usage fields, and B300 replay closure.
- Fixture: existing M0.4 cache seed/continuation fixtures, M0.5 KV replay
  artifacts, current Rust runtime live-token cache tests, and M9.8b-M9.8e Rust
  model-free helpers.
- Comparator: review the resulting M9.8f2-M9.8f5 stages for independent
  acceptance criteria and no remaining catch-all runtime cache item.
- Acceptance: each runtime cache slice below is independently implementable,
  reviewable, and measurable against current-C behavior before B300 closure.
- Drift policy: this split changes only roadmap/status docs; behavior stays
  unchanged.
- Review gate: ask Claude to review the split for missing runtime cache
  responsibilities.
- Validation passed: `git diff --check` and non-interactive Claude review with
  no blockers.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M9.8f2: Runtime Cache Configuration And Trace Contract

- Status: done
- Goal: add the Rust runtime server cache configuration/state and trace
  contract needed for disk cache decisions without yet loading or writing KVC
  payloads.
- Oracle: C server CLI/config fields for KV cache enablement, directory,
  budget, cross-quant policy, continued-store options, and trace cache-decision
  fields.
- Fixture: existing Rust `ds4-server-runtime-rs` CLI tests, M0.4 cache trace
  sections, and M7.2/M7.7 cache-decision oracle rows.
- Comparator: model-free tests comparing parsed config, default cache policy,
  trace field names, cache-source strings, and usage accounting inputs.
- Acceptance: runtime state can represent C-equivalent cache policy and emit
  cache decision traces for none/memory-token/disk-text cases without changing
  model execution.
- Drift policy: CLI spelling, default values, trace keys, and cache-source
  strings are exact; paths and timestamps remain normalized.
- Review gate: ask Claude to review runtime surface compatibility and trace
  contract stability.
- Validation needed: targeted runtime cache-surface tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Validation passed: targeted `cargo test -p ds4-engine --bin
  ds4-server-runtime-rs -- --nocapture`, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, NUL scan
  over touched Rust files, and non-interactive Claude review with no blockers.
- Owner path: `rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs`,
  `rust/ds4-gguf/src/kv_policy.rs`, `.memory/status.md`.

### M9.8f3: Runtime Disk-KV Lookup And Payload Restore

- Status: done
- Goal: wire text-prefix KVC lookup, session payload restore, effective prompt
  suffix tokenization, and tool-map trailer restore into the runtime request
  path before generation.
- Oracle: C `kv_cache_try_load_text`,
  `ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix`,
  `ds4_session_load_payload`, and
  `kv_cache_restore_tool_memory_for_messages`.
- Fixture: M0.5 seed restore/continuation KV artifacts, synthetic KVC files
  with tool-map trailers, and request histories that require exact DSML replay.
- Comparator: model-free disk lookup/load tests plus B300 smoke replay for one
  cache hit, one miss, and one tool-memory restored prompt.
- Acceptance: Rust can restore disk KV payloads selected by rendered text
  prefix, tokenize only the visible suffix, and report disk cache/read stats
  without losing exact DSML tool calls.
- Drift policy: cache key bytes, loaded token counts, suffix prompt bytes,
  tool replay stats, and load failure reasons are exact.
- Review gate: ask Claude to review unsafe session-payload FFI and suffix
  prompt construction.
- Validation needed: targeted disk-restore tests, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, B300 restore smoke if local
  model is unavailable, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and non-interactive Claude review with no blockers.
- Validation passed: targeted `cargo test -p ds4-engine --bin
  ds4-server-runtime-rs -- --nocapture`, `cargo build -p ds4-engine --bin
  ds4-server-runtime-rs`, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, B300 smoke on
  `ds4-rust-port-b300` covering one C-seeded disk hit, one miss, and one
  disk-restored tool-map prompt, full `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, NUL scan over touched files, and two
  non-interactive Claude reviews with no blockers after the live-session disk
  restore guard fix.
- Owner path: Rust engine session FFI, runtime server cache path,
  `rust/ds4-gguf/src/tool_memory.rs`, `.memory/status.md`.

### M9.8f4: Runtime KV Store, Continued Frontier, And Eviction

- Status: done
- Goal: write cold/continued/shutdown KVC checkpoints from the runtime server,
  preserve continued-frontier suppression/restoration, write tool-map trailers,
  and evict without deleting the just-written protected checkpoint.
- Oracle: C `kv_cache_store_current`, `kv_cache_maybe_store_continued`,
  `kv_cache_suppress_continued_store`, `kv_cache_store_live_prefix_text`, and
  `ds4_kvstore_evict`.
- Fixture: synthetic stores with budget edges, existing compatible files,
  visible-transcript keys, and M0.5 KV header/rendered-text artifacts.
- Comparator: file header/trailer byte checks, store/skip/evict trace checks,
  and M7.2 policy comparator reuse for boundary decisions.
- Acceptance: Rust writes KVC files with matching headers, rendered text keys,
  payload sizes, tool-map trailers, continued-frontier updates, and eviction
  outcomes.
- Drift policy: reason codes, extension flags, rendered key bytes, store token
  counts, trailer bytes, and protected eviction behavior are exact.
- Review gate: ask Claude to review store failure rollback and eviction
  protection.
- Validation needed: targeted store/evict tests, `python3
  ds4-parity/compare_kv_policy.py --negative-test`, `python3
  ds4-parity/compare_kvc_file.py --negative-test`, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.
- Validation passed: targeted `cargo test -p ds4-engine --bin
  ds4-server-runtime-rs -- --nocapture`, targeted `cargo test -p ds4-gguf
  tool_memory::tests:: -- --nocapture`, `python3
  ds4-parity/compare_kv_policy.py --negative-test`, `python3
  ds4-parity/compare_kvc_file.py --negative-test`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, B300 smoke
  on `ds4-rust-port-b300` proving cold KVC write, graceful shutdown KVC write,
  fresh-process disk-text restore, and protected eviction under a 64 MiB
  budget, and non-interactive Claude review with no blockers.
- Owner path: runtime server cache path, Rust KVC helpers, `.memory/status.md`.

### M9.8f5: Runtime Cache/KV Replay Comparator Closure

- Status: done
- Goal: close M9.8 runtime cache behavior against M0.4/M0.5 current-C
  artifacts and B300 model-backed replay.
- Oracle: M0.4 cache seed/continuation responses and traces, M0.5 seed
  miss/restore and continuation restore artifacts, KV headers, rendered text,
  cache decision logs, and `compare_server_kv.py`.
- Fixture: B300 `/workspace/ds4/ds4flash.gguf`, committed server request JSON,
  response/header/trace artifacts, and normalized KV metadata.
- Comparator: B300 replay comparing normalized responses, headers, generated
  bytes, trace cache decisions, KV metadata, rendered text, disk/text cache
  source, cached token counts, cache write tokens, and eviction/store decisions.
- Acceptance: Rust server cache behavior matches current-C artifacts end to
  end and M9.8 can hand off to the server parity report item.
- Drift policy: normalize paths, timestamps, raw KV payload hashes, and random
  IDs only; cache source, token counts, rendered text, KV headers, and eviction
  outcomes are exact.
- Review gate: ask Claude to review comparator coverage and B300 command
  fidelity.
- Validation needed: B300 KV/server comparator with negative tests, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, `python3
  ds4-parity/compare_server_kv.py`, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Validation passed: exact M0.5 three-lifetime Rust runtime replay on
  `ds4-rust-port-b300` matched current-C seed miss, seed restore,
  continuation restore, trace cache decisions, and KVC header rows; evidence is
  recorded in `ds4-parity/baselines/kv/m9.8f5/runtime-rust-b300-summary.json`.
  Also passed `python3 ds4-parity/compare_server_kv.py`, `python3
  ds4-parity/compare_server_kv.py --negative-test`, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, JSON
  validation, NUL scan over touched files, and non-interactive Claude review
  with no blockers.
- Owner path: Rust server runtime/cache/KV modules, `ds4-parity/`,
  `.memory/status.md`.

### M9.9: Server Parity Report Integration

- Status: done
- Goal: add a Milestone 9 server parity report and wire it into the unified
  parity report.
- Oracle: M9.2 through M9.8 comparators, M0.4/M0.5 refresh commands, and
  `./ds4_test --server` through the Rust server path.
- Fixture: committed server request/response/trace/KV artifacts and B300 rerun
  records for model-backed server comparisons.
- Comparator: local report that runs model-free server parser/protocol tests,
  summarizes server/KV artifact comparisons, and records B300 server replay
  commands as exact skips when the model is unavailable locally.
- Acceptance: `python3 ds4-parity/run_server_parity_report.py` passes locally,
  the unified report includes it, JSON output is machine-readable, and B300
  replay commands are sufficient to refresh the model-backed artifacts.
- Drift policy: report output normalizes only paths and timestamps.
- Review gate: ask Claude to review report coverage and B300 command fidelity.
- Validation needed: server report, unified report, `./ds4_test --server`
  through the Rust path or documented B300 skip, `cargo test --workspace`, and
  `git diff --check`.
- Validation passed: `python3
  ds4-parity/check_runtime_kv_replay_summary.py`, direct pass-count parser
  smoke, `python3 ds4-parity/run_server_parity_report.py`, `python3
  ds4-parity/run_server_parity_report.py --json`, `python3
  ds4-parity/run_parity_report.py`, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, Python syntax checks, NUL scan over
  touched files, generated helper cleanup, and non-interactive Claude review
  with no blockers.
- Owner path: `ds4-parity/`, Rust server report paths, `.memory/status.md`.

### M10.1: Runtime Graph Work Item Breakdown

- Status: done
- Goal: split Milestone 10 runtime graph orchestration parity into comparable
  implementation and oracle-capture work items before moving graph ownership.
- Oracle: current C `ds4_gpu_graph` allocation/execution paths in `ds4.c`,
  backend primitives in `ds4_gpu.h`, existing graph diagnostics, and the broad
  Milestone 10 acceptance gates in `RUST_PORT_ROADMAP.md`.
- Fixture: roadmap/TODO state plus source evidence for graph allocation,
  decode, prefill, compressed KV, session payload, MTP, and benchmark closure.
- Comparator: documentation-only review that each M10.2+ item has an explicit
  oracle, fixture, comparator, acceptance rule, drift policy, and validation
  gate.
- Acceptance: no M10 implementation item remains catch-all; graph shape,
  backend coverage, tensor oracle capture, decode, prefill, KV/session state,
  MTP, and end-to-end closure can be reviewed and validated independently.
- Drift policy: no runtime behavior changes; this item only defines measurable
  boundaries and required evidence.
- Review gate: ask Claude to review whether the split is comparable and avoids
  unmeasurable graph-port steps.
- Validation needed: roadmap/TODO diff inspection, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Validation passed: roadmap/TODO diff inspection, `git diff --check`, NUL scan
  over touched files, and non-interactive Claude review with no blockers.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/TODO.md`,
  `.memory/status.md`.

### M10.2: Backend Operation Inventory And Graph Plan Oracle

- Status: done
- Goal: capture a current-C oracle for the graph tensor plan, backend primitive
  surface, and command-buffer boundaries that Rust must preserve.
- Oracle: `ds4_gpu.h`, C call sites under `metal_graph_alloc_raw_cap`,
  `metal_graph_encode_layer_attention_batch`,
  `metal_graph_encode_layer_ffn_batch`, `metal_graph_eval_token_raw_swa`,
  `metal_graph_prefill_chunked_range`, and MTP verifier helpers.
- Fixture: graph plans for short, 2k, and 32k context settings; MTP
  disabled/enabled where model files are available; grouped `ds4_gpu.h`
  operation inventory.
- Comparator: machine-readable checker comparing graph plan and operation
  inventory against source-derived expectations and failing on missing backend
  primitives or unassigned graph tensors.
- Acceptance: every graph-used `ds4_gpu.h` primitive has a named Rust
  trait/facade target, every persistent/work tensor family has an owner group,
  and command-buffer boundaries are recorded before Rust scheduling starts.
- Drift policy: operation names, tensor families, context caps, compression
  ratios, command boundaries, and MTP enablement are exact; pointer addresses,
  timings, and allocation addresses are ignored.
- Review gate: ask Claude to review inventory completeness against
  `ds4_gpu.h` and graph call sites.
- Validation needed: oracle checker with negative fixture, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Validation passed: `python3 ds4-parity/check_graph_plan_inventory.py`,
  `python3 ds4-parity/check_graph_plan_inventory.py --negative-test`, JSON
  report smoke, Python syntax check, unified parity report comparator row,
  `git diff --check`, NUL scan over touched files, and non-interactive Claude
  review with no blockers.
- Owner path: graph plan/oracle files under `ds4-parity/`, `ds4_gpu.h`
  inventory references, `.memory/status.md`.

### M10.3: Rust Backend Trait And Graph Plan Surface

- Status: done
- Goal: add Rust graph-plan data structures and backend trait/facade coverage
  for the full M10.2 operation inventory without executing a model graph yet.
- Oracle: M10.2 graph plan and backend operation inventory.
- Fixture: same context/MTP plan cases as M10.2 plus synthetic missing-op and
  tensor-size mismatch cases.
- Comparator: Rust tests and/or parity script comparing serialized Rust graph
  plans, tensor families, capacities, and trait coverage against the C oracle.
- Acceptance: Rust can name and size the graph state C allocates, exposes a
  backend facade for every required primitive, and fails closed when the C
  inventory gains an unassigned primitive.
- Drift policy: tensor names, capacities, compression caps, raw-window caps,
  and operation names are exact; backend implementation remains FFI-backed.
- Review gate: ask Claude to review trait completeness and backend-specific
  semantics.
- Validation needed: targeted Rust graph-plan tests, inventory comparator,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Validation passed: `cargo test -p ds4-gpu graph_plan --lib`, `python3
  ds4-parity/compare_graph_plan_rust.py`, `python3
  ds4-parity/compare_graph_plan_rust.py --negative-test`, Python syntax check,
  unified parity report comparator row, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, NUL scan over touched files, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph/backend facade modules, graph plan comparator,
  `.memory/status.md`.

### M10.4: Current-C Intermediate Tensor Checkpoint Oracle

- Status: done
- Goal: add current-C checkpoint capture for selected graph tensors at decode,
  prefill, compressed-KV, output-head, and MTP verification boundaries.
- Oracle: C graph execution through `metal_graph_eval_token_raw_swa`,
  `metal_graph_prefill_chunked_range`, `metal_graph_prefill_layer_major`, and
  MTP verifier paths using the same GPU backend and model.
- Fixture: official-vector prompt slices, a chunked long-context prompt slice,
  cache continuation prompt slice, and MTP-enabled two-token draft case when
  the support model is available.
- Comparator: tensor checkpoint manifest and checker that compares shape,
  dtype, row selection, counter values, hashes, and selected f32 tolerances
  against fresh captures.
- Acceptance: C can recapture deterministic enough intermediate checkpoints
  for the next Rust decode/prefill stages, and nondeterministic rows are
  excluded or explicitly marked hash-only/skip with evidence.
- Drift policy: tensor shape, dtype, row index, layer, stage, and cache counters
  are exact; f32 values use per-stage tolerances; timings and absolute paths
  are normalized.
- Review gate: ask Claude to review checkpoint coverage and nondeterminism
  policy.
- Validation passed: B300 checkpoint capture on `ds4-rust-port-b300`, checker
  self-compare, negative mutation test, JSON syntax checks, unified parity
  report comparator row, local dump-target build, Python syntax check, `git
  diff --check`, NUL scan over touched files, and non-interactive Claude
  review with no blockers.
- Owner path: C checkpoint oracle helper, `ds4-parity/`, B300 artifacts,
  `.memory/status.md`.

### M10.5a: Rust GPU Sys ABI Surface For Graph Primitives

- Status: done
- Goal: expose the complete M10.2 graph backend primitive surface in
  `ds4-gpu-sys` without scheduling model execution yet.
- Oracle: M10.2 operation inventory plus the exact current `ds4_gpu.h`
  signatures.
- Fixture: all 81 backend operations recorded in
  `baselines/graph/m10.2/graph-plan-inventory.json`.
- Comparator: `compare_gpu_sys_abi.py` parses `ds4_gpu.h` and
  `rust/ds4-gpu-sys/src/lib.rs`, then compares every Rust declaration's return
  and parameter ABI types against the C signature.
- Acceptance: every oracle operation has a Rust sys declaration with matching
  return type and parameter type sequence, and synthetic missing/type-drift
  mutations fail closed.
- Drift policy: operation names, return ABI types, and parameter ABI types are
  exact; no backend execution or tensor values are compared in this item.
- Review gate: ask Claude to review the unsafe ABI surface and static
  comparator.
- Validation passed: ABI comparator with negative test, Python syntax check,
  `cargo test --workspace`, unified comparator-only parity report, `cargo fmt
  --all -- --check`, `git diff --check`, NUL scan over touched files, and
  non-interactive Claude review with no blockers.
- Owner path: `rust/ds4-gpu-sys/`, `ds4-parity/`, `.memory/status.md`.

### M10.5b: Rust Decode Call-Order And State Plan

- Status: done
- Goal: model one-token raw-SWA decode scheduling in Rust as an executable
  plan/trace before launching backend kernels.
- Oracle: C `metal_graph_eval_token_raw_swa`,
  `metal_graph_encode_token_raw_swa`, `metal_graph_encode_decode_layer`, and
  M10.4 decode checkpoint metadata.
- Fixture: short first-token and continuation-token cases covering dense,
  ratio-4 compressed/indexer, and ratio-128 compressed layers.
- Comparator: Rust plan trace versus C-derived call order, command boundaries,
  raw/compressed/indexer cache counter transitions, and tensor owner plan.
- Acceptance: Rust emits the exact decode layer/head/output scheduling plan and
  cache counter transitions needed for the B300 execution comparator, without
  mutating backend state.
- Drift policy: operation order, layer classification, token position, cache
  counters, and command boundaries are exact; tensor values remain out of
  scope.
- Review gate: ask Claude to review decode ordering and cache-state modeling.
- Validation passed: targeted Rust decode-plan tests, plan comparator with
  negative test, JSON syntax check, Python syntax check, unified
  comparator-only parity report, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, NUL scan over touched files, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph decode modules, backend facade, B300 comparator,
  `.memory/status.md`.

### M10.5c1: Rust Structured Decode Weight Table

- Status: done
- Goal: add a structured Rust DS4 base/layer weight table equivalent to the C
  `ds4_weights` and `ds4_layer_weights` pointer layout, without launching graph
  kernels.
- Oracle: C `ds4_weights`/`ds4_layer_weights` field inventory and the existing
  Rust `bind_ds4_tensors` flat role bindings.
- Fixture: synthetic DS4 GGUF tensor directory from the M4 tensor-binding
  comparator, covering dense, ratio-4, ratio-128, optional bias, and hash-layer
  fields.
- Comparator: `compare_rust_weight_table.py` runs `ds4-gguf-dump`, checks the
  structured table against C field order, and verifies that it flattens exactly
  back to `bound_tensors`.
- Acceptance: every C base/layer field has a typed Rust table slot, absence is
  preserved for dense/indexer/hash optional fields, and synthetic layer/field
  mutations fail closed.
- Drift policy: field names, order, and present/absent optional semantics are
  exact; tensor values and backend execution remain out of scope.
- Review gate: ask Claude to review weight ownership shape and comparator
  coverage.
- Validation passed: targeted Rust tests, weight-table comparator with negative
  test, unified comparator-only parity report, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Owner path: `rust/ds4-gguf/`, `ds4-parity/`, `.memory/status.md`.

### M10.5c2: Rust Decode Graph Tensor State

- Status: done
- Goal: add Rust graph tensor state allocation/zero-fill scaffolding for
  one-token decode, without issuing decode kernels yet.
- Oracle: M10.2 graph tensor owner inventory and M10.4 checkpoint note that
  cache tensors hash full allocated capacity.
- Fixture: M10.3 graph plan cases for raw/compressed/indexer cache sizes and
  representative ctx/prompt combinations.
- Comparator: Rust tensor-state plan versus M10.2 owner inventory and M10.3
  graph-plan byte-size formulas.
- Acceptance: Rust state names every decode tensor owner, allocates the same
  sizes, and records zero-fill obligations for unused cache capacity.
- Drift policy: tensor names, owner groups, allocated byte sizes, and zero-fill
  requirements are exact; kernel values remain out of scope.
- Review gate: ask Claude to review tensor ownership, lifetimes, and zero-fill
  obligations.
- Validation passed: targeted Rust tests, tensor-state comparator with negative
  test, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and non-interactive Claude review with no blockers.
- Owner path: Rust graph tensor state modules, `ds4-parity/`,
  `.memory/status.md`.

### M10.5c3: Rust Decode Backend Facade

- Status: completed
- Goal: add safe Rust facade methods for the subset of M10.5a backend
  primitives used by default fused one-token decode, without owning the full
  decode schedule yet.
- Oracle: M10.2 operation inventory, M10.5a ABI comparator, and C
  `metal_graph_encode_decode_layer` default fusion branches with graph-reference
  environment flags unset.
- Fixture: decode primitive groups for embedding, QKV/norm, raw/compressed KV
  store, attention, router/MoE, hyper-connection, output head, command
  begin/flush/end, read, and synchronize-on-failure.
- Comparator: Rust facade coverage versus required decode primitive list and
  ABI type checks.
- Acceptance: every default decode primitive has one safe Rust wrapper with the
  same tensor argument order as the C call site and no unwrapped raw ABI calls
  leak into scheduler code.
- Drift policy: operation names and tensor argument order are exact; no GPU
  numeric execution is compared in this item.
- Review gate: ask Claude to review unsafe FFI encapsulation and tensor
  argument ordering.
- Validation passed: facade comparator with negative test, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: `rust/ds4-gpu/`, `rust/ds4-gpu-sys/`, `ds4-parity/`,
  `.memory/status.md`.

### M10.5c4: Rust Single-Token Decode Graph Execution

- Status: split
- Goal: move one-token decode scheduling for the target model into Rust while
  calling existing backend primitives through the M10.5c3 facade.
- Oracle: M10.4 decode checkpoints and the M10.5b call-order plan.
- Fixture: official-vector first-token and continuation-token cases with raw
  SWA, ratio-4 compressed/indexer layers, ratio-128 compressed layers, and
  directional-steering disabled/enabled cases if available.
- Comparator: Rust-vs-C intermediate tensor diffs, logits diffs, raw/compressed
  cache counter comparisons, and command-boundary trace comparison.
- Acceptance: Rust decode produces matching logits and selected intermediate
  tensors for one token, updates raw/compressed/indexer counters like C, and
  preserves command-buffer boundaries required by the backend.
- Drift policy: scheduling order, cache counters, token position, and command
  boundaries are exact; f32 tensor values follow M10.4 tolerances.
- Review gate: ask Claude to review decode ordering, cache mutation, and unsafe
  backend calls.
- Validation needed: targeted decode comparator on B300, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph decode modules, backend facade, B300 comparator,
  `.memory/status.md`.

### M10.5c4a: Rust Decode Execution Trace Oracle

- Status: completed
- Goal: add a Rust dry-run decode execution trace that expands the M10.5b
  token/layer plan into facade method calls and cache-counter transitions,
  without calling backend kernels.
- Oracle: M10.5b decode plan, M10.5c3 facade operation table, and current-C
  default branches in `metal_graph_encode_decode_layer` and
  `metal_graph_encode_output_head`.
- Fixture: first-token, short-prefill decode, ratio-4 emit, long indexed
  decode, and no-logits/no-split cases from the M10.5b oracle.
- Comparator: JSON trace comparator for stage order, facade method names,
  tensor argument roles, raw/compressed/indexer counter deltas, split flush,
  read, and synchronize-on-failure markers.
- Acceptance: the Rust trace proves the full default one-token call tape and
  state-counter mutations before any FFI execution is introduced.
- Drift policy: method names, stage order, tensor roles, counters, and command
  boundaries are exact; no tensor values or GPU execution are compared.
- Review gate: ask Claude to review trace completeness against C default
  branches and M10.5c3 facade coverage.
- Validation passed: trace comparator with negative test, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph decode trace modules, `ds4-parity/`,
  `.memory/status.md`.

### M10.5c4b: Rust Decode Runtime State Bridge

- Status: completed
- Goal: instantiate the decode graph runtime state from M10.5c2 tensor plans
  and M10.5c1 structured weights, still without launching kernels.
- Oracle: M10.5c1 structured weight table, M10.5c2 graph-state allocation
  plan, and M10.5c4a trace tensor roles.
- Fixture: `ctx32768_mtp_off` state plus dense, ratio-4, ratio-128, and
  hash-layer weight-presence slices.
- Comparator: runtime-state dump comparing allocated/view/lazy/external tensor
  roles, per-layer cache counters, required weight offsets/types, and facade
  call inputs against the dry-run trace.
- Acceptance: every tensor and weight role needed by the default decode trace
  resolves to a Rust-owned runtime handle or explicit external input before
  backend execution.
- Drift policy: tensor names, weight roles, offsets/types, allocation sizes,
  and initial counters are exact; backend kernel values remain out of scope.
- Review gate: ask Claude to review lifetime ownership, optional tensor
  handling, and weight-role mapping.
- Validation passed: runtime-state comparator with negative test, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph runtime state bridge, `ds4-parity/`,
  `.memory/status.md`.

### M10.5c4c1: Rust CUDA Backend Linkage And B300 ABI Smoke

- Status: completed
- Goal: make the Rust GPU crate link the C/CUDA backend on Linux when explicitly
  requested and prove the safe tensor wrappers can execute on the B300 pod.
- Oracle: M10.5a GPU ABI declarations, the existing C CUDA build flags in the
  Makefile, and the safe-vs-direct backend ABI smoke test.
- Fixture: B300 `ds4-rust-port-b300` pod, `CUDA_ARCH=native`, and a feature-gated
  `ds4-gpu --features cuda-backend` build so non-CUDA Linux builds remain
  no-link by default.
- Comparator: static smoke-contract checker plus B300 `backend_abi` Rust test
  exercising initialize, tensor allocation, write/read/fill/view/copy,
  command flush/end, synchronize, and failure paths through the CUDA backend.
- Acceptance: B300 Rust can compile/link `ds4.c` and `ds4_cuda.cu`, execute the
  backend ABI smoke test through CUDA, and preserve the existing macOS backend
  ABI coverage without requiring CUDA for ordinary Linux builds.
- Drift policy: C/CUDA compiler/link flags follow the Makefile unless a future
  validated B300 run requires a concrete override; runtime outputs are exact
  byte comparisons against direct C ABI calls.
- Review gate: ask Claude to review feature gating, CUDA link flags, and backend
  test containment.
- Validation passed: static smoke-contract comparator with negative test, B300
  `cargo test -p ds4-gpu --features cuda-backend --test backend_abi`, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust GPU build/test wiring, B300 comparator,
  `.memory/status.md`.

### M10.5c4c2a: Rust Decode Model-Map Backend Bridge

- Status: completed
- Goal: expose the model-map backend calls Rust decode execution needs before
  passing real GGUF weight pointers to CUDA kernels.
- Oracle: M10.2 `ModelMapBackend` operation inventory, M10.5a sys ABI
  declarations, the C CUDA model-map/cache behavior, and the M10.5c4c1 B300
  Rust CUDA backend smoke.
- Fixture: tiny model bytes plus file descriptor in the backend ABI smoke, with
  CUDA cache-range calls exercised on B300 under `--features cuda-backend`.
- Comparator: static model-map bridge checker plus B300 `model_map_abi` Rust
  test covering fd, full map, map range, cache range, q8/f16 cache hook, and
  out-of-range failure paths.
- Acceptance: Rust has safe status-returning wrappers for model map, model fd,
  model map range, CUDA model cache range, and CUDA q8/f16 cache range; CUDA-only
  cache wrappers remain Linux-gated; B300 executes the wrappers before the
  scheduler starts using real weight offsets.
- Drift policy: wrapper names and ABI calls are exact; cache residency behavior
  follows the C backend, while the smoke fixture uses tiny deterministic bytes
  instead of the full GGUF model.
- Review gate: ask Claude to review model-map lifetime, cfg gating, and CUDA
  cache wrapper containment.
- Validation passed: model-map bridge comparator with negative test, B300 `cargo
  test -p ds4-gpu --features cuda-backend --test model_map_abi`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust decode backend model-map wrappers, B300 comparator,
  `.memory/status.md`.

### M10.5c4c2b: Rust One-Token Decode B300 Execution

- Status: active
- Goal: execute the default one-token decode trace through the M10.5c3 facade
  on B300 and capture Rust checkpoints for the M10.4 decode cases.
- Oracle: M10.4 decode checkpoints, the M10.5c4a trace for exact call order
  and counter transitions, the M10.5c4b runtime state bridge, and the
  M10.5c4c1 B300 Rust CUDA backend smoke plus M10.5c4c2a model-map bridge.
- Fixture: official-vector first-token and continuation-token layer-coverage
  cases covering raw SWA, ratio-4 compressed/indexer layers, and ratio-128
  compressed layers. Continuation-state reuse is deferred to M10.5c4d.
- Comparator: B300 Rust-vs-C intermediate tensor hashes/samples, logits diff,
  raw/compressed/indexer counter comparison, and command-boundary trace diff.
- Acceptance: Rust one-token decode matches the selected M10.4 tensor
  checkpoints and logits within recorded tolerances while preserving cache
  counters and command boundaries.
- Drift policy: scheduling order, cache counters, token position, and command
  boundaries are exact; f32 tensor values follow M10.4 tolerances.
- Review gate: ask Claude to review decode ordering, cache mutation, and
  unsafe backend-call containment.
- Validation needed: targeted decode comparator on B300, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.

### M10.5c4d: Decode Continuation And Optional Steering Closure

- Status: pending
- Goal: close one-token decode coverage for continuation-state and optional
  directional-steering cases after default B300 execution is passing.
- Oracle: M10.4 continuation checkpoints, C directional-steering decode
  branches, and M10.5c4c2b execution results.
- Fixture: continuation-token decode after prefill, long indexed decode, and
  directional-steering enabled cases when support vectors are available.
- Comparator: continuation Rust-vs-C tensor/logit/counter diffs plus optional
  steering-specific tensor diffs or an explicit unavailable-fixture skip.
- Acceptance: continuation decode stays comparable to C, and steering coverage
  is either validated or explicitly skipped with the missing fixture recorded.
- Drift policy: continuation counters and command boundaries are exact; f32
  values follow M10.4 tolerances; optional steering skip text is exact.
- Review gate: ask Claude to review continuation state reuse, optional tensor
  ownership, and skip conditions.
- Validation needed: continuation/steering comparator on B300 or exact skip,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.

### M10.6: Rust Layer-Major Prefill And Chunking

- Status: pending
- Goal: move layer-major and chunked prefill scheduling into Rust while keeping
  backend operations FFI-backed.
- Oracle: M10.4 prefill checkpoints plus C
  `metal_graph_prefill_layer_major` and `metal_graph_prefill_chunked_range`.
- Fixture: short whole-prefill prompt, boundary-crossing resume suffix, 2k+
  chunked prompt, and long-context prompt slice.
- Comparator: Rust-vs-C prefill tensor checkpoints, final logits, raw ring
  physical/logical row mapping, compressed row counters, and progress/chunk
  boundary traces.
- Acceptance: Rust prefill matches C for whole, chunked, and resumed suffix
  paths, including output-row selection and cache state after the final chunk.
- Drift policy: chunk boundaries, raw ring mapping, compressed counters, and
  selected logits are exact within M10.4 tolerances; progress timestamps are
  normalized.
- Review gate: ask Claude to review chunk boundary and resume-prefix handling.
- Validation needed: targeted prefill comparator on B300, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph prefill modules, B300 comparator,
  `.memory/status.md`.

### M10.7: Rust Graph Session State And Payload Parity

- Status: pending
- Goal: make Rust own graph session state needed by cache snapshots, disk KV
  payloads, restore, and continued-frontier decisions.
- Oracle: C session payload save/load paths, M7 session-payload fixtures, M0.5
  KV artifacts, M9.8 runtime cache behavior, and graph state counters from
  M10.5/M10.6.
- Fixture: short and long checkpoints, raw-ring wrap cases, ratio-4 and
  ratio-128 compressed states, restored disk KVC payloads, and continued-store
  frontiers.
- Comparator: session payload byte/field comparator, disk-KV replay comparator,
  Rust-vs-C cache counter checks, and B300 restore smoke.
- Acceptance: Rust graph state can save, restore, and continue sessions with
  C-compatible payload bytes and M9.8 cache accounting while Rust owns decode
  and prefill scheduling.
- Drift policy: payload layout, counter fields, cache source, cached token
  counts, and store/restore decisions are exact; raw payload hashes remain
  normalized where existing policy requires.
- Review gate: ask Claude to review payload compatibility and cache/frontier
  invariants.
- Validation needed: session payload comparator, KV replay comparator, B300
  restore smoke, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and non-interactive Claude review with no blockers.
- Owner path: Rust graph session state, session payload/KV comparators,
  `.memory/status.md`.

### M10.8: Rust MTP Draft And Verifier Orchestration

- Status: pending
- Goal: move MTP draft, exact N=2 verifier, prefix-1 commit, and speculative
  frontier restore/rollback orchestration into Rust.
- Oracle: C `metal_graph_eval_mtp_draft`,
  `metal_graph_verify_decode2_exact`, `spec_frontier_snapshot`,
  `spec_frontier_restore`, and `spec_frontier_commit_prefix1`.
- Fixture: MTP disabled, first-draft miss, one-token accept, two-token accept,
  verifier failure, prefix1 commit, and rollback cases on the B300 support
  model path when available.
- Comparator: accepted-token sequence comparison, frontier counter/tensor
  checkpoint comparison, logits/top-id comparison, and visible output parity
  for speculative argmax.
- Acceptance: Rust speculative decode never changes the target output stream,
  commits only verified prefixes, and restores graph state exactly on misses or
  verifier failures.
- Drift policy: accepted token sequence, frontier counters, rollback state, and
  target logits are exact within M10.4 tolerances; probe logs and timings are
  normalized.
- Review gate: ask Claude to review state rollback and target-stream safety.
- Validation needed: B300 MTP comparator, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust MTP graph orchestration, B300 MTP comparator,
  `.memory/status.md`.

### M10.9: Runtime Graph End-To-End And Benchmark Closure

- Status: pending
- Goal: close Milestone 10 by routing the Rust runtime path through Rust-owned
  graph scheduling and comparing end-to-end quality, long-context behavior, and
  throughput.
- Oracle: current C graph runtime on the same model/backend and committed M0.3,
  M0.6, M9 server/cache artifacts.
- Fixture: official vector logprob cases, long-context fact-recall prompt,
  tool-call-quality prompts, M9 server requests, and short/long `ds4-bench`
  benchmark prompts.
- Comparator: Rust-runtime `ds4_test --logprob-vectors`,
  `ds4_test --long-context`, `ds4_test --tool-call-quality`, server replay
  comparators, and same-backend `ds4-bench` CSV comparator.
- Acceptance: Rust graph runtime passes official-vector, long-context, and
  tool-call-quality gates through the Rust path; server parity remains green;
  benchmark regression beyond the agreed threshold is documented before merge.
- Drift policy: behavioral outputs and cache accounting are exact; benchmark
  throughput uses the M0.6 threshold policy and same model/backend/machine
  identity.
- Review gate: ask Claude to review end-to-end coverage and benchmark
  comparability.
- Validation needed: B300 Rust-runtime end-to-end suite, server parity report,
  benchmark CSV comparator, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: Rust runtime graph path, parity reports, benchmark artifacts,
  `.memory/status.md`.

## Later Items

Add later roadmap items from `RUST_PORT_ROADMAP.md` as each active item
completes.
