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

- Status: pending
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
- Validation needed: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison run, `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust CLI one-shot path, CLI transcript comparator,
  `ds4-parity/`, `.memory/status.md`.

### M8.13d: Rust CLI One-Shot Runtime-Control Surface

- Status: pending after M8.13c
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
- Validation needed: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison run, `cargo test --workspace`, and `git diff --check`.
- Owner path: Rust CLI one-shot path, runtime-control transcript comparator,
  `ds4-parity/`, `.memory/status.md`.

## Later Items

Add later roadmap items from `RUST_PORT_ROADMAP.md` as each active item
completes.
