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

### M10.5c4c2b1: Rust Decode Execution Preflight

- Status: completed
- Goal: prove the real-model Rust decode execution inputs are usable on B300
  before launching the full one-token scheduler.
- Oracle: M10.4 short decode checkpoint targets, the M10.5c4b runtime-state
  bridge, the M10.5c4c2a model-map bridge, and C startup behavior that maps
  only the GGUF tensor-data range before decode.
- Fixture: `/workspace/ds4/ds4flash.gguf` on B300 plus representative dense,
  ratio-4, and ratio-128 layers `[0, 2, 3]`, including the layer-2 compressed
  cache tensors used by the M10.4 short-decode oracle.
- Comparator: static contract plus optional B300 JSON validator that checks
  mmap-backed GGUF header parsing, DS4 weight binding, fd/map-range handoff,
  bounded representative model/Q8 cache hooks, representative tensor
  allocation, and checkpoint-target coverage.
- Acceptance: Rust emits `ds4.decode_execution_preflight.v1` from B300 after
  mapping the real GGUF tensor-data range, binding 43 layers, allocating the
  representative checkpoint tensors, and validating at least one model cache
  range plus one Q8/F16 cache hook.
- Drift policy: model path may vary only by rerun command; model SHA, model
  size, tensor-data offset semantics, selected checkpoint names, selected
  layers, and cache hook presence are exact.
- Review gate: ask Claude to review mmap lifetime, bounded cache selection,
  and backend cleanup ordering.
- Validation passed: preflight comparator with negative test, B300 preflight
  binary plus candidate JSON validation, `cargo test -p ds4-gpu
  decode_execution --lib`, `cargo check -p ds4-gpu --bin
  ds4-decode-exec-preflight`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: Rust decode execution preflight module/bin, B300 comparator,
  `.memory/status.md`.

### M10.5c4c2b2a: Rust Full Decode State Allocation

- Status: completed
- Goal: allocate and initialize the full M10.5c2 decode graph-state surface on
  B300 before scheduling the first numeric decode kernels.
- Oracle: M10.5c2 graph-state allocation plan for `ctx32768_mtp_off`, including
  the 349 logical instances, 272 initially owned allocations, 806,175,248 owned
  bytes, three `hc_*` views, lazy `ffn_out`, external directional steering, and
  exact zero/negative-infinity fill counts.
- Fixture: `ds4-decode-state-alloc` under the CUDA backend on B300, using the
  default 32768-token context/prompt plan without model tensor values.
- Comparator: static allocation contract plus optional B300 JSON validator that
  checks summary counts, largest allocations, view extents, fill kinds, and
  backend cleanup.
- Acceptance: Rust emits `ds4.decode_state_allocation.v1` after allocating all
  initially owned graph-state tensors, applying required fills, creating the
  planned views, and cleaning up the backend.
- Drift policy: allocation counts, byte totals, view offsets/extents, lazy and
  external counts, and fill counts are exact; backend logging and GPU identity
  may vary by B300 host.
- Review gate: ask Claude to review allocation ownership, view lifetimes, fill
  semantics, and cleanup on error paths.
- Validation passed: state-allocation comparator with negative test, B300
  allocation binary plus candidate JSON validation, `cargo check -p ds4-gpu
  --bin ds4-decode-state-alloc`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: Rust decode state allocation binary, B300 comparator,
  `.memory/status.md`.

### M10.5c4c2b2b1: Rust First Decode Kernel Execution

- Status: completed
- Goal: cross from allocation/preflight into one real model-backed decode
  kernel on B300 before scheduling the full one-token tape.
- Oracle: the M10.5c4c2b1 real-GGUF map/bind path, M10.5c4c2b2a `cur_hc`
  allocation shape, the M10.5c3 `embed_token_hc` facade wrapper, and C decode
  startup behavior that launches kernels inside command batches.
- Fixture: `/workspace/ds4/ds4flash.gguf` on B300, token `0`,
  `base.token_embd`, and the `cur_hc` graph-state tensor.
- Comparator: static first-kernel contract plus optional B300 JSON validator
  checking model identity, command-batch execution, token-embedding offset,
  `cur_hc` shape, nonzero readback, and pinned sample values.
- Acceptance: Rust emits `ds4.decode_first_kernel.v1` after mapping the real
  GGUF, launching `embed_token_hc` through the safe facade, synchronizing,
  reading back `cur_hc`, and cleaning up the backend.
- Drift policy: model size, tensor-data offset, token embedding offset/size,
  token id, output shape, and selected samples are exact; backend stderr logs
  and B300 host identity may vary.
- Review gate: ask Claude to review command-batch lifecycle, model-map
  lifetime, tensor readback, and cleanup ordering.
- Validation passed: first-kernel comparator with negative test, B300
  first-kernel binary plus candidate JSON validation, `cargo check -p ds4-gpu
  --bin ds4-decode-first-kernel`, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: Rust decode first-kernel binary, B300 comparator,
  `.memory/status.md`.

### M10.5c4c2b2b2a: Rust First-Kernel Current-C Oracle Comparator

- Status: completed
- Goal: compare the B300 Rust `embed_token_hc` readback against an
  independently emitted current-C oracle before adding more scheduler calls.
- Oracle: current C model startup semantics for `token_embd.weight`, including
  `model_open`, `config_validate_model`, `weights_bind`, `embed_token_f16`, and
  `hc_from_plain_embedding`.
- Fixture: `/workspace/ds4/ds4flash.gguf` on B300, token `0`,
  `base.token_embd`, and the `cur_hc` graph-state tensor read back from Rust.
- Comparator: B300 paired JSON comparator that checks current-C oracle and Rust
  candidate model identity, token embedding offset/size, output shape, full
  `cur_hc` FNV digest, nonzero count, and selected f32 samples.
- Acceptance: `ds4-first-kernel-oracle-dump` emits
  `ds4.first_kernel_oracle.v1`, `ds4-decode-first-kernel` emits
  `ds4.decode_first_kernel.v1`, and the paired comparator accepts their
  `cur_hc` output on B300.
- Drift policy: model size, tensor-data offset, token id, token embedding
  offset/size, output shape, digest, nonzero count, and selected samples are
  exact; output file paths and backend stderr may vary by rerun.
- Review gate: ask Claude to review the oracle helper's use of current-C
  helpers, Rust digest emission, and paired comparator failure modes.
- Validation passed: oracle comparator with negative test, B300 current-C
  oracle plus Rust candidate paired validation, `make
  ds4-first-kernel-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-first-kernel`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: current-C first-kernel oracle helper, Rust first-kernel JSON,
  paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b1: Rust Layer-0 Attention HC-Pre B300 Execution

- Status: completed
- Goal: execute the first post-embedding layer-0 attention HC-pre prefix
  through the M10.5c3 facade on B300 before taking ownership of the full
  one-token scheduler.
- Oracle: the current-C `model_open`, `config_validate_model`, `weights_bind`,
  `ds4_gpu_set_model_fd`, `ds4_gpu_set_model_map_range`,
  `ds4_gpu_embed_token_hc_tensor`, `ds4_gpu_rms_norm_plain_tensor`,
  `ds4_gpu_matmul_f16_tensor`, and
  `ds4_gpu_hc_split_weighted_sum_norm_tensor` path for token `0` and layer
  `0`.
- Fixture: B300 `ds4flash.gguf`, token `0`, layer `0`, `cur_hc`, `flat_hc`,
  `hc_mix`, `hc_split`, `attn_cur`, and `attn_norm`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for `cur_hc`, `flat_hc`, `hc_mix`, `hc_split`,
  `attn_cur`, and `attn_norm`; current-C SHA256 digests are captured for
  evidence, and selected f32 samples compare within the existing `1e-6`
  tolerance.
- Acceptance: Rust launches `embed_token_hc`, `rms_norm_plain`, `matmul_f16`,
  and `hc_split_weighted_sum_norm` in one command batch, synchronizes, and
  matches the current-C layer-0 HC-pre oracle on B300.
- Drift policy: token, layer, model offsets, tensor byte sizes, and FNV digests
  are exact; selected f32 sample text may vary by JSON formatting but numeric
  values must stay within `1e-6`.
- Review gate: ask Claude to review C oracle helper sequencing, Rust facade
  call ordering, tensor byte/readback sizes, and paired comparator failure
  modes.
- Validation passed: layer-0 HC-pre comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-attn-hc-pre-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-attn-hc-pre`, c2b1 first-kernel rerun, c2b2b2a
  current-C oracle rerun, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: current-C layer-0 HC-pre oracle helper, Rust layer-0 HC-pre JSON,
  paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b2: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2a and M10.5c4c2b2b2b2b so the next commit has a
  tensor-level execution oracle before cache mutation and attention scheduling
  enter the same diff.

### M10.5c4c2b2b2b2a: Rust Layer-0 QKV RoPE B300 Execution

- Status: completed
- Goal: execute the layer-0 Q/KV projection and RoPE prefix after the
  M10.5c4c2b2b2b1 HC-pre boundary on B300.
- Oracle: the current-C GPU tensor path for token `0`, layer `0`: model
  fd/map bridge, embedding, HC RMS/matmul/split/norm, `ds4_gpu_matmul_q8_0`
  for `attn_q_a` and `attn_kv`, fused `ds4_gpu_dsv4_qkv_rms_norm_rows`,
  `ds4_gpu_matmul_q8_0` for `attn_q_b`, `ds4_gpu_head_rms_norm`, and
  `ds4_gpu_rope_tail` for dense layer-0 `q` and `kv`.
- Fixture: B300 `ds4flash.gguf`, token `0`, layer `0`, position `0`,
  `attn_norm`, `qr`, `kv_raw`, `qr_norm`, final RoPE `q`, and final RoPE
  `kv`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for the six tensors, current-C SHA256 evidence, and
  selected f32 samples within the existing `1e-6` tolerance.
- Acceptance: Rust launches the HC-pre prefix plus `matmul_q8_0`,
  `dsv4_qkv_rms_norm_rows`, `head_rms_norm`, and `rope_tail` through the safe
  facade in one command batch, synchronizes, and matches the current-C GPU
  oracle on B300.
- Drift policy: token, layer, position, dense RoPE constants, model offsets,
  tensor byte sizes, and FNV digests are exact; selected f32 sample text may
  vary by JSON formatting but numeric values must stay within `1e-6`.
- Review gate: ask Claude to review Q/KV operation ordering, fused norm
  arguments, RoPE constants, tensor sizes, and comparator failure modes.
- Validation passed: QKV/RoPE comparator with negative test, B300 current-C
  oracle plus Rust candidate paired validation, `make
  ds4-layer0-qkv-rope-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-qkv-rope`, c2b2b2b1 layer-0 HC-pre rerun, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: current-C layer-0 QKV/RoPE oracle helper, Rust layer-0 QKV/RoPE
  JSON, paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b2b: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b1 and M10.5c4c2b2b2b2b2 so cache mutation and
  layer-0 attention output get an exact tensor oracle before the remaining
  all-layer scheduler and logits path are introduced.

### M10.5c4c2b2b2b2b1: Rust Layer-0 Attention Output B300 Execution

- Status: done
- Goal: execute the dense layer-0 raw KV store, attention, inverse RoPE,
  attention output projection, and HC expansion after the M10.5c4c2b2b2b2a
  QKV/RoPE boundary on B300.
- Oracle: the current-C GPU tensor path for token `0`, layer `0`, position
  `0`: HC-pre, QKV/RoPE, `ds4_gpu_kv_fp8_store_raw_tensor`,
  `ds4_gpu_attention_decode_heads_tensor`, inverse `ds4_gpu_rope_tail`,
  `ds4_gpu_attention_output_low_q8_tensor`, and
  `ds4_gpu_matmul_q8_0_hc_expand_tensor` with dense-layer `raw_cap`,
  `raw_row`, `n_raw`, and `raw_start` pinned.
- Fixture: B300 `ds4flash.gguf`, token `0`, layer `0`, position `0`, dense
  raw SWA cache row `0`, post-store `kv`, raw cache row, final inverse-RoPE
  `heads`, `attn_low`, `attn_out`, and `after_attn_hc`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for the six tensors, current-C SHA256 evidence,
  dense cache counter fields, selected f32 samples within `1e-6`, and pinned
  layer-0 attention-output weight metadata.
- Acceptance: Rust launches the QKV/RoPE prefix plus raw KV store,
  attention decode, inverse RoPE, low-rank attention output, and HC expansion
  through the safe facade in one command batch, synchronizes, and matches the
  current-C GPU oracle on B300.
- Drift policy: token, layer, position, raw cache counters, dense RoPE
  constants, model offsets, tensor byte sizes, and FNV digests are exact;
  selected f32 sample text may vary by JSON formatting but numeric values must
  stay within `1e-6`.
- Review gate: ask Claude to review cache-row/counter handling, attention
  arguments, inverse RoPE constants, output projection dimensions, HC expansion
  inputs, and comparator failure modes.
- Validation passed: layer-0 attention-output comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-attn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-attn-output`, c2b2b2b2a layer-0 QKV/RoPE rerun, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: current-C layer-0 attention-output oracle helper, Rust layer-0
  attention-output JSON, paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b2b2: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2a and M10.5c4c2b2b2b2b2b so the layer-0 FFN
  body gets an exact tensor oracle before the remaining all-layer scheduler,
  cache-compression transitions, and logits path are introduced.

### M10.5c4c2b2b2b2b2a: Rust Layer-0 FFN Output B300 Execution

- Status: done
- Goal: execute the layer-0 FFN body after the M10.5c4c2b2b2b2b1 attention
  output boundary on B300.
- Oracle: the current-C GPU tensor path for token `0`, layer `0`, position
  `0`: HC-pre, QKV/RoPE, dense raw KV store, dense attention, inverse RoPE,
  attention output, FFN HC-pre, FFN norm, router logits/select, routed MoE,
  shared expert SwiGLU, shared down projection, and final FFN HC expansion.
- Fixture: B300 `ds4flash.gguf`, token `0`, layer `0`, position `0`, dense
  raw SWA cache row `0`, post-attention `after_attn_hc`, FFN tensors
  `ffn_cur`, `ffn_norm`, `router_logits`, `router_probs`,
  `router_selected`, `router_weights`, `routed_out`, `shared_mid`,
  `shared_out`, and `after_ffn_hc`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for the FFN tensors, current-C SHA256 evidence,
  router selected IDs, selected f32 samples within `1e-6`, and pinned
  layer-0 FFN weight metadata.
- Acceptance: Rust launches the layer-0 attention-output prefix plus FFN
  HC-pre, router, routed MoE, shared expert, shared down, and HC expansion
  through the safe facade in one command batch, synchronizes, and matches the
  current-C GPU oracle on B300.
- Drift policy: token, layer, position, raw cache counters, model offsets,
  tensor byte sizes, router-selected IDs, and FNV digests are exact; selected
  f32 sample text may vary by JSON formatting but numeric values must stay
  within `1e-6`.
- Review gate: ask Claude to review FFN HC-pre ordering, router arguments,
  routed expert dimensions and byte strides, shared SwiGLU/down dimensions,
  HC expansion inputs, and comparator failure modes.
- Validation needed: layer-0 FFN-output comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-ffn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-ffn-output`, c2b2b2b2b1 layer-0 attention-output rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Owner path: current-C layer-0 FFN-output oracle helper, Rust layer-0
  FFN-output JSON, paired comparator, `.memory/status.md`.
- Validation: B300 paired FFN-output validator passed 819 checks before
  pinning and 885 checks after pinning exact FFN digests, router metadata, and
  weight metadata; B300 c2b2b2b2b1 attention-output predecessor rerun passed
  493 checks; local `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` passed 30/20/0, `cargo test --workspace` passed,
  `cargo fmt --all -- --check` passed, `git diff --check` passed, and
  non-interactive Claude review returned no blockers.

### M10.5c4c2b2b2b2b2b: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b1 and M10.5c4c2b2b2b2b2b2 so the
  output-head/logits kernels are independently compared before the remaining
  all-layer scheduler, cache-compression transitions, and final decode trace
  are introduced together.

### M10.5c4c2b2b2b2b2b1: Rust Layer-0 Output Head B300 Execution

- Status: done
- Goal: execute the output-head path on the validated layer-0 FFN output on
  B300.
- Oracle: the current-C GPU path for token `0`, layer `0`, position `0`
  through the production decode-layer encoder, followed by the output HC
  collapse, output norm, and vocab projection kernels.
- Fixture: B300 `ds4flash.gguf`, token `0`, layer `0`, position `0`,
  post-layer `after_ffn_hc`, output-head tensors `output_pre`,
  `output_weights`, `output_embd`, `output_norm`, and `logits`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for the output-head tensors, current-C SHA256
  evidence, selected f32 samples within `1e-6`, and pinned output-head weight
  metadata.
- Acceptance: Rust launches the validated layer-0 decode path plus
  `rms_norm_plain`, output HC preprojection, `output_hc_weights`,
  `hc_weighted_sum`, output RMS norm, and vocab projection through the safe
  facade in one command batch, synchronizes, and matches the current-C GPU
  oracle on B300.
- Drift policy: token, layer, position, raw cache counters, model offsets,
  output tensor byte sizes, and FNV digests are exact; selected f32 sample text
  may vary by JSON formatting but numeric values must stay within `1e-6`.
- Review gate: ask Claude to review output-head ordering, HC collapse inputs,
  output weight dimensions, vocab projection dimensions, and comparator
  failure modes.
- Validation passed: layer-0 output-head comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-output-head-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-output-head`, c2b2b2b2b2a layer-0 FFN-output rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Owner path: current-C layer-0 output-head oracle helper, Rust layer-0
  output-head JSON, paired comparator, `.memory/status.md`.
- Evidence: B300 output-head paired validation passed 399 pinned checks with
  `after_ffn_hc=3d49316c93ce351f`, `output_pre=67cd67e9413ba488`,
  `output_weights=b7b3f62be8581476`, `output_embd=0b0d4f86243397e3`,
  `output_norm=24029c3b5c92306e`, and `logits=27d2e668424d8d9f`;
  predecessor c2b2b2b2b2a FFN-output B300 rerun passed 885 checks.

### M10.5c4c2b2b2b2b2b2: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2a and M10.5c4c2b2b2b2b2b2b so the
  dense layer-loop buffer swap is compared before compressed layer-2 cache
  mutation and all-layer scheduling are introduced.

### M10.5c4c2b2b2b2b2b2a: Rust Two Dense-Layer Output Head B300 Execution

- Status: done
- Goal: execute two dense decode layers followed by the output-head path on
  B300 after the single-layer FFN output and output-head kernels are
  independently compared.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0` and
  `1` through the production decode-layer encoder with the production
  `cur_hc`/`after_ffn_hc` swap after each layer, followed by the production
  output-head encoder.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0` and
  `1`, `after_layer0_hc`, `after_layer1_hc`, output-head tensors
  `output_pre`, `output_weights`, `output_embd`, `output_norm`, and `logits`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests, current-C SHA256 evidence, selected f32 samples
  within `1e-6`, pinned output-head weight metadata, and exact layer/raw-cache
  operation counters.
- Acceptance: Rust launches layer `0`, swaps the HC buffers, launches layer `1`
  from the swapped current HC, runs output-head from the layer-1 HC through the
  safe facade in one command batch, synchronizes, and matches the current-C GPU
  oracle on B300.
- Drift policy: token, position, layer count, raw cache rows, output tensor byte
  sizes, and FNV digests are exact; selected f32 sample text may vary by JSON
  formatting but numeric values must stay within `1e-6`.
- Review gate: ask Claude to review dense layer-loop ordering, buffer swap
  semantics, layer-specific raw-cache use, output-head input selection, and
  comparator failure modes.
- Validation needed: two-dense-layer output-head comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-two-layer-output-head-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-two-layer-output-head`, c2b2b2b2b2b1 layer-0 output-head rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Owner path: current-C two-layer output-head oracle helper, Rust two-layer
  output-head JSON, paired comparator, `.memory/status.md`.
- Evidence: B300 two-dense-layer output-head paired validation passed 446
  pinned checks with `after_layer0_hc=3d49316c93ce351f`,
  `after_layer1_hc=f764d7067de5c945`, `output_pre=ebc1b8ccc088d27a`,
  `output_weights=e20bda6aca5453b2`, `output_embd=b5d1377b7c179886`,
  `output_norm=2ce848a4cc2363db`, and `logits=14dbbac3cd6ed7a8`;
  predecessor c2b2b2b2b2b1 layer-0 output-head B300 rerun passed 399 checks;
  local unified report passed 32/22/0, workspace tests passed, fmt/diff checks
  passed, and the touched-file NUL scan passed across 11 files.

### M10.5c4c2b2b2b2b2b2b: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2b1 and M10.5c4c2b2b2b2b2b2b2 so the
  first ratio-4 compressed/indexer state mutation is compared before compressed
  attention, all-layer scheduling, and final logits are introduced together.

### M10.5c4c2b2b2b2b2b2b1: Rust Layer-2 Ratio-4 Compressor State B300 Execution

- Status: done
- Goal: execute dense layers `0` and `1`, then execute layer `2` through the
  ratio-4 attention and indexer compressor state updates on B300 without yet
  taking ownership of compressed attention or the layer-2 FFN/output-head tail.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0` and
  `1` through production decode-layer plus HC swaps, followed by production
  layer `2` decode. The oracle reads the layer-2 raw cache row, attention
  compressor frontier state, indexer compressor frontier state, and counters
  after production layer `2`; later layer-2 stages do not mutate those frontier
  tensors.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0` and
  `1`, ratio-4 layer `2`, `raw_row=0`, `n_raw=1`, `emit_compressed_row=false`,
  `layer_n_comp[2]=0`, and `layer_n_index_comp[2]=0`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for `after_layer1_hc`, `layer2_raw_cache_row`,
  `layer2_attn_state_kv`, `layer2_attn_state_score`,
  `layer2_index_state_kv`, and `layer2_index_state_score`, plus pinned
  compressor/indexer weight metadata and exact ratio/counter fields.
- Acceptance: Rust launches the validated dense layer `0` and `1` prefix,
  swaps HC buffers after each dense layer, launches layer `2` Q/KV/RoPE,
  stores raw KV, executes attention and indexer `matmul_f16_pair` plus
  `compressor_update` through the safe facade, initializes frontier state the
  same way as current C, and matches the current-C GPU oracle on B300.
- Drift policy: layer id, ratio, emit cadence, row counters, tensor byte sizes,
  and FNV digests are exact; selected f32 samples may differ only by JSON
  formatting and must stay within `1e-6`.
- Review gate: ask Claude to review layer-2 input HC selection, compressor
  state initialization, ratio-4/indexer dimensions, no-emit counter semantics,
  and comparator failure modes.
- Validation needed: layer-2 compressor-state comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-layer2-compressor-state-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-compressor-state`, c2b2b2b2b2b2a two-layer output-head
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Owner path: current-C layer-2 compressor-state oracle helper, Rust layer-2
  compressor-state JSON, paired comparator, `.memory/status.md`.
- Evidence: B300 layer-2 compressor-state paired validation passed 589 pinned
  checks with `after_layer1_hc=f764d7067de5c945`,
  `layer2_raw_cache_row=51f0a2971a59c6da`,
  `layer2_attn_state_kv=57544afc0dfa6bcf`,
  `layer2_attn_state_score=38d2d40c6f170ab6`,
  `layer2_index_state_kv=2a44d6b140b6ef0b`, and
  `layer2_index_state_score=b8da053681327aec`; predecessor
  c2b2b2b2b2b2a two-layer output-head B300 rerun passed 446 checks; local
  unified report passed 33/23/0, workspace tests passed, fmt/diff checks
  passed, and the touched-file NUL scan passed.

### M10.5c4c2b2b2b2b2b2b2: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2b2a and
  M10.5c4c2b2b2b2b2b2b2b so the layer-2 attention-output boundary is compared
  after the first compressor-state mutation, before the remaining FFN,
  all-layer scheduler, and final logits are introduced together.

### M10.5c4c2b2b2b2b2b2b2a: Rust Layer-2 Attention Output B300 Execution

- Status: done
- Goal: execute dense layers `0` and `1`, then execute layer `2` through
  raw-only attention decode, inverse compressed RoPE, low-rank attention output
  projection, and HC expansion on B300 without yet taking the layer-2 FFN,
  remaining layers, or final logits.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0` and
  `1` through production decode-layer plus HC swaps, followed by production
  layer `2` decode. The oracle reads the layer-2 raw cache row, compressor
  frontier state, attention heads after inverse compressed RoPE, attention
  low/output tensors, and `after_attn_hc` after production layer `2`.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0` and
  `1`, ratio-4 layer `2`, `raw_row=0`, `n_raw=1`, `raw_start=0`,
  `n_comp=0`, no indexed compressed rows, `emit_compressed_row=false`,
  `layer_n_comp[2]=0`, and `layer_n_index_comp[2]=0`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for `after_layer1_hc`, layer-2 raw/cache state,
  `layer2_heads`, `layer2_attn_low`, `layer2_attn_out`, and
  `layer2_after_attn_hc`, plus pinned attention-output weight metadata and
  exact raw/compressed counter fields.
- Acceptance: Rust launches the validated dense layer `0`/`1` prefix and
  layer-2 compressor-state prefix, then runs raw-only `attention_decode_heads`,
  inverse compressed `rope_tail`, `attention_output_low_q8`, and
  `matmul_q8_0_hc_expand` through the safe facade, matching the current-C GPU
  oracle on B300.
- Drift policy: layer id, ratio, `n_raw`, `raw_start`, `n_comp`, selected row
  count, tensor byte sizes, and FNV digests are exact; selected f32 samples may
  differ only by JSON formatting and must stay within `1e-6` relative or
  absolute tolerance.
- Review gate: ask Claude to review the no-compressed-row attention branch,
  inverse compressed RoPE settings, attention-output dimensions, and HC
  expansion input selection.
- Validation needed: layer-2 attention-output comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-layer2-attn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-attn-output`, c2b2b2b2b2b2b1 layer-2 compressor-state
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Owner path: current-C layer-2 attention-output oracle helper, Rust layer-2
  attention-output JSON, paired comparator, `.memory/status.md`.
- Evidence: B300 layer-2 attention-output paired validation passed 815 pinned
  checks with `after_layer1_hc=f764d7067de5c945`,
  `layer2_raw_cache_row=51f0a2971a59c6da`,
  `layer2_attn_state_kv=57544afc0dfa6bcf`,
  `layer2_attn_state_score=38d2d40c6f170ab6`,
  `layer2_index_state_kv=2a44d6b140b6ef0b`,
  `layer2_index_state_score=b8da053681327aec`,
  `layer2_heads=241a32d72fe7885b`,
  `layer2_attn_low=6d33e52dbc93ed09`,
  `layer2_attn_out=c5a61256ab424d80`, and
  `layer2_after_attn_hc=9c038ab7c95176b4`. B300 artifact SHA256 values were
  `oracle=728fa6b858f9ff6669424eac7691b65d1ffe9d78e9c5f7cbe85c412cc5ce80a7`
  and
  `rust=8b561a5eca4874bb4ba6bbf5bc83080d761f294a7cce839435c5815ff2db9ca3`;
  predecessor c2b2b2b2b2b2b1 layer-2 compressor-state B300 rerun passed 589
  checks; local unified report passed 34/24/0, workspace tests passed,
  fmt/diff checks passed, and the touched-file NUL scan passed.

### M10.5c4c2b2b2b2b2b2b2b: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2b2b1 and
  M10.5c4c2b2b2b2b2b2b2b2 so the layer-2 FFN-output boundary is compared
  after the layer-2 raw-only attention-output boundary, before the remaining
  layers, ratio-128 coverage, output head, and final logits are introduced
  together.

### M10.5c4c2b2b2b2b2b2b2b1: Rust Layer-2 FFN Output B300 Execution

- Status: complete
- Goal: execute dense layers `0` and `1`, then execute layer `2` through the
  full FFN tail and final layer-2 HC expansion on B300. This slice extends the
  validated layer-2 attention-output boundary without yet taking the remaining
  40 layers, output head, or final logits.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0` and
  `1` through production decode-layer plus HC swaps, followed by production
  layer `2` decode. The oracle reads the layer-2 attention-output tensors,
  layer-2 FFN HC-pre/norm/router/MoE/shared-expert tensors, and final
  `after_ffn_hc` after production layer `2`.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0` and
  `1`, ratio-4 layer `2`, `raw_row=0`, `n_raw=1`, `raw_start=0`,
  `n_comp=0`, no indexed compressed rows, `emit_compressed_row=false`,
  `layer_n_comp[2]=0`, and `layer_n_index_comp[2]=0`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for the layer-2 attention-output boundary plus
  `layer2_ffn_cur`, `layer2_ffn_norm`, router outputs, routed/shared expert
  tensors, and `layer2_after_ffn_hc`, plus pinned layer-2 FFN weight metadata
  and exact raw/compressed counter fields.
- Acceptance: Rust launches the validated dense layer `0`/`1` prefix and
  layer-2 attention-output prefix, then runs `hc_split_weighted_sum_norm`,
  `router_select`, `routed_moe_one`, `shared_gate_up_swiglu_q8_0`, and
  `shared_down_hc_expand_q8_0` for layer `2` through the safe facade, matching
  the current-C GPU oracle on B300.
- Drift policy: layer id, ratio, `n_raw`, `raw_start`, `n_comp`, selected row
  count, tensor byte sizes, selected expert ids, and FNV digests are exact;
  selected f32 samples may differ only by JSON formatting and must stay within
  `1e-6` relative or absolute tolerance.
- Review gate: ask Claude to review layer-2 FFN residual/split selection,
  router hash/bias options, routed expert byte-strides, shared expert
  dimensions, and HC expansion input selection.
- Validation passed: layer-2 FFN-output comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation with 1,383 pinned
  checks, local `make ds4-layer2-ffn-output-oracle-dump`, `cargo check -p
  ds4-gpu --bin ds4-decode-layer2-ffn-output`, c2b2b2b2b2b2b2a layer-2
  attention-output rerun with 815 checks, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Owner path: current-C layer-2 FFN-output oracle helper, Rust layer-2
  FFN-output JSON, paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b2b2b2b2b2: Rust One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2b2b2a and
  M10.5c4c2b2b2b2b2b2b2b2b so the first ratio-128 compressed layer is
  compared after the layer-2 ratio-4 FFN-output boundary, before the repeated
  remaining layers, output head, and final logits are introduced.

### M10.5c4c2b2b2b2b2b2b2b2a: Rust Layer-3 Ratio-128 FFN Output B300 Execution

- Status: completed
- Goal: execute dense layers `0` and `1`, layer `2` through its validated
  ratio-4 FFN-output boundary, then execute layer `3` as the first ratio-128
  compressed layer through the full FFN tail and final layer-3 HC expansion on
  B300. This slice extends coverage to the ratio-128 compressor path without
  yet taking the remaining repeated layers, output head, or final logits.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0` and
  `1` through production decode-layer plus HC swaps, layer `2` through
  production ratio-4 decode plus HC swap, followed by production layer `3`
  ratio-128 decode. The oracle reads `after_layer2_hc`, the layer-3 raw cache
  row, layer-3 attention compressor state, layer-3 attention output tensors,
  layer-3 FFN HC-pre/norm/router/MoE/shared-expert tensors, and final
  `layer3_after_ffn_hc`.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0`
  and `1`, ratio-4 layer `2`, ratio-128 layer `3`, `raw_row=0`, `n_raw=1`,
  `raw_start=0`, `n_comp=0`, no selected compressed rows,
  `emit_compressed_row=false`, and `layer_n_comp[3]=0`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for `after_layer2_hc`, layer-3 attention compressor
  state, layer-3 attention/FFN tensors, router outputs, routed/shared expert
  tensors, and `layer3_after_ffn_hc`, plus pinned layer-3 attention/FFN weight
  metadata and exact raw/compressed counter fields.
- Acceptance: Rust launches the validated dense layer `0`/`1` prefix and
  layer-2 FFN-output prefix, swaps the layer-2 HC output into `cur_hc`, then
  runs layer `3` ratio-128 `matmul_f16_pair`, `compressor_update`,
  raw-only `attention_decode_heads`, attention output HC expansion,
  `hc_split_weighted_sum_norm`, `router_select`, `routed_moe_one`,
  `shared_gate_up_swiglu_q8_0`, and `shared_down_hc_expand_q8_0` through the
  safe facade, matching the current-C GPU oracle on B300.
- Drift policy: layer id, compression ratio, `n_raw`, `raw_start`, `n_comp`,
  tensor byte sizes, selected expert ids, and FNV digests are exact; selected
  f32 samples may differ only by JSON formatting and must stay within `1e-6`
  relative or absolute tolerance.
- Review gate: ask Claude to review layer-3 ratio-128 counter selection,
  compressor state sizing, absence of indexer state, router hash/bias options,
  routed expert byte-strides, shared expert dimensions, and HC expansion input
  selection.
- Validation passed: layer-3 FFN-output comparator with negative test, local
  comparator self-check, paired `/tmp/ds4-layer3-oracle.json` vs
  `/tmp/ds4-layer3-rust.json` validation with 1,261 checks, B300 current-C
  oracle plus Rust candidate paired validation with
  `oracle=932b669ec3b4fdee0369b745968f92dc7ebc3c97e9b063b012bd380118dde9df`
  and
  `rust=3153600948c0e41e4b2fa01075eb8f0d1d2824435a46b2b4d365569b70ef1797`,
  local `arch -arm64 make ds4-layer3-ffn-output-oracle-dump`, local `cargo
  check -p ds4-gpu --bin ds4-decode-layer3-ffn-output`, B300
  c2b2b2b2b2b2b2b1 layer-2 FFN-output rerun with 1,383 checks, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with `NO BLOCKERS`.
- Owner path: current-C layer-3 FFN-output oracle helper, Rust layer-3
  FFN-output JSON, paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b2b2b2b2b2b: Rust Remaining One-Token Decode B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2b2b2b1 and
  M10.5c4c2b2b2b2b2b2b2b2b2 so the first post-ratio128 ratio-4/indexer
  layer is compared before the remaining repeated layers, output head, and
  final logits are introduced.

### M10.5c4c2b2b2b2b2b2b2b2b1: Rust Layer-4 Post-Ratio128 Ratio-4 FFN Output B300 Execution

- Status: completed
- Goal: execute dense layers `0` and `1`, ratio-4 layer `2`, ratio-128 layer
  `3`, then execute layer `4` as the first ratio-4/indexer layer after a
  ratio-128 predecessor through the full FFN tail and final layer-4 HC
  expansion on B300. This isolates the ratio128-to-ratio4 transition before
  the remaining repeated layers and final logits.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0`
  through `3` via production decode-layer execution plus HC swaps, followed by
  production layer `4` ratio-4 decode without the final HC swap. The oracle
  reads `after_layer3_hc`, layer-4 raw cache row, attention compressor state,
  indexer compressor state, attention output tensors, FFN HC-pre/norm/router,
  routed/shared expert tensors, and final `layer4_after_ffn_hc`.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0`
  and `1`, ratio-4 layer `2`, ratio-128 layer `3`, ratio-4 layer `4`,
  `raw_row=0`, `n_raw=1`, `raw_start=0`, `n_comp=0`, no selected compressed
  rows, `emit_compressed_row=false`, and `layer_n_comp[4]=0`.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for `after_layer3_hc`, layer-4 attention/indexer
  compressor state, layer-4 attention/FFN tensors, router outputs,
  routed/shared expert tensors, and `layer4_after_ffn_hc`, plus pinned
  layer-4 attention/FFN weight metadata and exact raw/compressed/indexer
  counter fields.
- Acceptance: Rust launches the validated dense layer `0`/`1`, layer-2
  FFN-output, and layer-3 FFN-output prefix, swaps the layer-3 HC output into
  `cur_hc`, then runs layer `4` ratio-4 `matmul_f16_pair`,
  attention/indexer `compressor_update`, raw-only `attention_decode_heads`,
  attention output HC expansion, `hc_split_weighted_sum_norm`,
  `router_select`, `routed_moe_one`, `shared_gate_up_swiglu_q8_0`, and
  `shared_down_hc_expand_q8_0` through the safe facade, matching the
  current-C GPU oracle on B300.
- Drift policy: layer id, compression ratio, `n_raw`, `raw_start`, `n_comp`,
  `layer_n_index_comp`, tensor byte sizes, selected expert ids, and FNV
  digests are exact; selected f32 samples may differ only by JSON formatting
  and must stay within `1e-6` relative or absolute tolerance.
- Review gate: ask Claude to review the layer-3-to-layer-4 HC swap, ratio-4
  counter selection, attention/indexer compressor state sizing, router
  hash/bias options, routed expert byte-strides, shared expert dimensions, and
  HC expansion input selection.
- Validation passed: layer-4 FFN-output comparator self-check and negative
  test, paired `/tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-oracle.json`
  vs `/tmp/ds4-c2b2b2b2b2b2b2b2b1-layer4-ffn-output-rust.json` validation
  with 1,383 pinned checks, B300 current-C oracle plus Rust CUDA candidate
  validation with 1,209 checks before pinning, B300 predecessor
  c2b2b2b2b2b2b2b2a layer-3 ratio-128 FFN-output rerun with 1,261 checks,
  local `arch -arm64 make ds4-layer4-ffn-output-oracle-dump`, and local
  `cargo check -p ds4-gpu --bin ds4-decode-layer4-ffn-output`. Final local
  workspace gates and Claude review are recorded in `.memory/status.md`.
- Owner path: current-C layer-4 FFN-output oracle helper, Rust layer-4
  FFN-output JSON, paired comparator, `.memory/status.md`.

### M10.5c4c2b2b2b2b2b2b2b2b2: Rust Remaining Layer Loop And Logits B300 Execution

- Status: split
- Split into M10.5c4c2b2b2b2b2b2b2b2b2a and
  M10.5c4c2b2b2b2b2b2b2b2b2b so the repeated all-layer decode loop reaches a
  comparable final HC boundary before the output-head and final logits are
  attached.

### M10.5c4c2b2b2b2b2b2b2b2b2a: Rust All-Layer Final HC B300 Execution

- Status: done
- Goal: execute token `0` at position `0` through all 43 decode layers on
  B300 using the Rust safe facade, stopping after the production layer-42
  `cur_hc`/`after_ffn_hc` swap and before the output-head kernels. This
  proves the repeated layer scheduler, alternating ratio-4/indexer and
  ratio-128 compressed layers, and per-layer raw/cache state mutation without
  mixing in vocab projection drift.
- Oracle: the current-C GPU path for token `0`, position `0`, layers `0`
  through `42` via production `metal_graph_encode_decode_layer` execution and
  HC swaps after every layer. The oracle reads layer-loop HC checkpoints after
  layers `4`, `5`, and `42`, raw-cache rows for layers `5` and `42`, and
  attention/indexer compressor state for the first newly covered ratio-128
  layer and the final ratio-4/indexer layer.
- Fixture: B300 `ds4flash.gguf`, token `0`, position `0`, dense layers `0`
  and `1`, ratio-4 layers `2,4,...,42`, ratio-128 layers `3,5,...,41`,
  `raw_row=0`, `n_raw=1`, `raw_start=0`, no emitted compressed rows, and
  zero compressed/indexer counters for all compressed layers at this position.
- Comparator: B300 paired current-C oracle vs Rust candidate JSON with exact
  full-buffer FNV digests for the selected HC, raw-cache, and compressor-state
  checkpoints, exact layer/count metadata, and tolerant JSON float samples.
- Acceptance: Rust executes every layer through the same safe facade calls used
  by the layer-4 comparator, swaps the HC buffers after each layer, preserves
  zero compressed/indexer counters for position `0`, and matches the current-C
  final layer-42 HC and selected cache/state checkpoints on B300.
- Drift policy: layer count, compression schedule, cache counters, raw row,
  `n_raw`, `raw_start`, tensor byte sizes, and FNV digests are exact; selected
  f32 samples may differ only by JSON formatting and must stay within `1e-6`
  relative or absolute tolerance.
- Review gate: ask Claude to review the generic all-layer decode helper,
  per-layer cache/state ownership, compression schedule, HC swap loop, and
  checkpoint readback before commit.
- Validation passed: all-layer final-HC comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-all-layer-final-hc-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-all-layer-final-hc`, c2b2b2b2b2b2b2b2b1 layer-4 FFN-output
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, local unified report with B300 rerun command coverage, and
  non-interactive Claude review with `NO BLOCKERS`.
- Owner path: current-C all-layer final-HC oracle helper, Rust all-layer
  final-HC JSON, paired comparator, `.memory/status.md`.
- Evidence: B300 all-layer final-HC paired validation passed 730 pinned checks
  with `after_layer4_hc=b19322ec84d84935`,
  `after_layer5_hc=b9c9026559412805`,
  `after_layer42_hc=cbd17b425564f63f`,
  `layer5_raw_cache_row=8f2606992a7f1a18`,
  `layer5_attn_state_kv=8c17d55c4b8e6de9`,
  `layer5_attn_state_score=292852343a4b4512`,
  `layer42_raw_cache_row=029806013304ca31`,
  `layer42_attn_state_kv=42a2f55a8dc3403b`,
  `layer42_attn_state_score=5b0a233b9c74b3ee`,
  `layer42_index_state_kv=2f5aefc0f5ed2728`, and
  `layer42_index_state_score=6a5003b30aad1406`; predecessor layer-4
  FFN-output B300 rerun passed 1,209 checks; local
  `run_parity_report.py --skip-local-oracles` passed 38/28/0; Claude review
  returned `NO BLOCKERS`.

### M10.5c4c2b2b2b2b2b2b2b2b2b: Rust Full One-Token Output Head And Logits B300 Execution

- Status: done
- Goal: execute the default one-token decode trace through the M10.5c3 facade
  on B300 after the layer-4 post-ratio128 ratio-4 FFN-output boundary, the
  layer-3 ratio-128 FFN-output boundary, the layer-2 FFN-output boundary, the
  layer-2 raw-only attention-output boundary, the first ratio-4
  compressor/indexer state mutation, two dense decode layers, layer-0 FFN
  output, and output-head kernels are independently compared.
- Oracle: M10.4 decode checkpoints, the M10.5c4a trace for exact call order
  and counter transitions, the M10.5c4b runtime state bridge, and the
  M10.5c4c1 B300 Rust CUDA backend smoke plus M10.5c4c2a model-map bridge,
  M10.5c4c2b1 execution preflight, M10.5c4c2b2a full state allocation,
  M10.5c4c2b2b1 first-kernel execution, M10.5c4c2b2b2a current-C first-kernel
  oracle comparator, M10.5c4c2b2b2b1 layer-0 HC-pre comparator, and
  M10.5c4c2b2b2b2a layer-0 QKV/RoPE comparator, and
  M10.5c4c2b2b2b2b1 layer-0 attention-output comparator, and
  M10.5c4c2b2b2b2b2a layer-0 FFN-output comparator, and
  M10.5c4c2b2b2b2b2b1 layer-0 output-head comparator, and
  M10.5c4c2b2b2b2b2b2a two-dense-layer output-head comparator,
  M10.5c4c2b2b2b2b2b2b1 layer-2 ratio-4 compressor-state comparator, and
  M10.5c4c2b2b2b2b2b2b2a layer-2 attention-output comparator, and
  M10.5c4c2b2b2b2b2b2b2b1 layer-2 FFN-output comparator,
  M10.5c4c2b2b2b2b2b2b2b2a layer-3 ratio-128 FFN-output comparator, and
  M10.5c4c2b2b2b2b2b2b2b2b1 layer-4 post-ratio128 ratio-4 FFN-output
  comparator, and M10.5c4c2b2b2b2b2b2b2b2b2a all-layer final-HC
  comparator.
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
- Validation passed: full output-head comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, c2b2b2b2b2b2a
  two-layer output-head B300 predecessor rerun, c2b2b2b2b2b2b2b2b2a
  all-layer final-HC B300 predecessor rerun, `make
  ds4-full-output-head-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-full-output-head`, local unified report with B300 rerun command
  coverage, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, touched-file NUL scan, and non-interactive Claude review with
  `NO BLOCKERS`.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.
- Evidence: B300 full output-head paired validation passed 440 pinned checks
  with `after_layer42_hc=cbd17b425564f63f`,
  `output_pre=91ea6aeb7a0a0d9f`,
  `output_weights=323062ce53dc6f9c`,
  `output_embd=8788c46e4f0a1f30`,
  `output_norm=185c73c1de55a942`, and
  `logits=432eef0524ced3ad`; predecessor two-layer output-head B300 rerun
  passed 446 checks; predecessor all-layer final-HC B300 rerun passed 730
  checks; local `run_parity_report.py --skip-local-oracles` passed 39/29/0.

### M10.5c4d1: Rust Short Decode-Continuation Output-Head B300 Execution

- Status: completed
- Goal: execute a short decode-seeded continuation sequence through the Rust
  safe-facade scheduler, ending at token position `21` with real reused raw and
  ratio-4 compressed decode state before layer-major Rust prefill exists.
- Oracle: current-C GPU `metal_graph_eval_token_raw_swa` over the same
  deterministic 22-token sequence, production HC swaps, compressed-row
  emission, and final output-head logits.
- Fixture: B300 `ds4flash.gguf`, token sequence `0..21`, context `32768`,
  final position `21`, no directional steering, no MTP, and default split
  flush after layer `3`.
- Comparator: paired Rust-vs-C final tensor hashes/samples for layer-42 HC,
  output-head tensors, logits, selected raw-cache rows, and selected
  compressed-cache state/counter metadata.
- Acceptance: Rust matches the current-C final logits and selected cache/state
  checkpoints for the 22-token continuation sequence, with `layer2` and
  `layer42` ratio-4 counters at `5` and ratio-128 counters at `0`.
- Drift policy: token sequence, final position, split flush layer, cache
  counters, raw ring rows, and output tensor digests are exact; JSON float
  samples use the M10.5 tolerant-sample policy.
- Review gate: ask Claude to review continuation state reuse, compressed-row
  emission, counter increments, and command flush boundaries.
- Validation passed: short continuation comparator with negative test, paired
  local artifact validation with 798 pinned checks, B300 current-C oracle plus
  Rust CUDA candidate validation with 766 checks, B300 predecessor full
  output-head rerun with 430 checks, local `arch -arm64 make
  ds4-short-continuation-output-head-oracle-dump`, local `cargo check -p
  ds4-gpu --bin ds4-decode-short-continuation-output-head`, local unified
  report with B300 rerun command coverage, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, touched-file NUL scan,
  non-interactive Claude review with `NO BLOCKERS`, and pinned artifact SHA256
  `oracle=7c53400cef52a6f73aa8fea06ec4f64298d045bd0776397cc6f3030bbdf38429`
  and
  `rust=f00f23abc84474e7d00a8958ebb0c4f055889a384744b004954cfdfa9eb651a6`.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.
- Evidence: B300 short continuation paired validation matched `sequence_len=22`,
  `final_position=21`, `raw_row=21`, `raw_start=0`, `n_raw=22`,
  `layer2_n_comp=5`, `layer2_n_index_comp=5`, `layer5_n_comp=0`,
  `layer42_n_comp=5`, and `layer42_n_index_comp=5`. Full-buffer FNV digests
  are `after_layer42_hc=40e22a11d8ca9178`,
  `output_pre=642c2b6d18b62c67`,
  `output_weights=9592e0f3a26737e1`,
  `output_embd=e57d3ebe8ed8c63c`,
  `output_norm=1615bc086702b3b8`,
  `logits=fcc73408cecb8073`,
  `layer2_raw_cache_row=3befca08431b15ed`,
  `layer2_attn_comp_row4=061fb5b8eabae3db`,
  `layer2_index_comp_row4=a8afc0bf90381f52`,
  `layer5_attn_state_kv=2c574c58aad15bc1`,
  `layer5_attn_state_score=71948016152ae1de`,
  `layer42_raw_cache_row=998292db4c5534e7`,
  `layer42_attn_comp_row4=24844d05b88a2c04`,
  `layer42_index_comp_row4=c7e7a2f46c2aa3b2`,
  `layer42_attn_state_kv=cf3576176ae9d092`, and
  `layer42_index_state_kv=06ac626b7530144e`; local
  `run_parity_report.py --skip-local-oracles` passed 40/30/0.

### M10.5c4d2: Rust Ratio-Boundary Continuation Coverage

- Status: completed
- Goal: extend decode-continuation execution to a ratio boundary where ratio-4
  and ratio-128 compressed rows are emitted by the final token.
- Oracle: current-C GPU `metal_graph_eval_token_raw_swa` over the same
  deterministic token sequence and final output-head path.
- Fixture: B300 `ds4flash.gguf`, deterministic token sequence ending at
  position `127`, no directional steering, and no MTP.
- Comparator: paired Rust-vs-C final tensor hashes/samples, emitted compressed
  row digests, and ratio-4/ratio-128 counter metadata.
- Acceptance: Rust matches current-C through final logits while updating both
  ratio-4 and ratio-128 compressed caches at the boundary.
- Drift policy: token sequence, boundary position, cache counters, row
  selection, and output tensor digests are exact; JSON float samples use the
  M10.5 tolerant-sample policy.
- Review gate: ask Claude to review ratio-boundary counter transitions and row
  quantization.
- Validation passed: ratio-boundary comparator with negative test, paired local
  artifact validation with 865 pinned checks, B300 current-C oracle plus Rust
  CUDA candidate validation with 829 checks, local `arch -arm64 make
  ds4-ratio-boundary-output-head-oracle-dump`, local `cargo check -p ds4-gpu
  --bin ds4-decode-ratio-boundary-output-head`, local unified report with B300
  rerun command coverage, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, touched-file NUL scan, non-interactive Claude
  review with `NO BLOCKERS`, and pinned artifact SHA256
  `oracle=b8e813b11312f931a4bb786d297661d933588bba78da7bad78a147653c2c58c7`
  and
  `rust=72fbe9424cecf96d6710d0a1adc43563a5d7af0360ec804628e9224bec081449`.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.
- Evidence: B300 ratio-boundary paired validation matched `sequence_len=128`,
  `final_position=127`, `raw_row=127`, `raw_start=0`, `n_raw=128`,
  `emit_compressed_row=1`, `layer2_n_comp=32`,
  `layer2_n_index_comp=32`, `layer5_n_comp=1`, `layer42_n_comp=32`,
  and `layer42_n_index_comp=32`. Full-buffer FNV digests are
  `after_layer42_hc=12f1089ad3297673`,
  `output_pre=71f7d1ca0703e093`,
  `output_weights=3e646960d299fca0`,
  `output_embd=3f0d9c27cf78b430`,
  `output_norm=a1baf22acb3476dc`,
  `logits=c67eab1a566286ae`,
  `layer2_raw_cache_row=cfc54c8671abaa5a`,
  `layer2_attn_comp_row31=72353245d1b57607`,
  `layer2_index_comp_row31=63be8943c4bf8cd2`,
  `layer5_raw_cache_row=082429f33ac1c8df`,
  `layer5_attn_comp_row0=e65ab25c4927545f`,
  `layer5_attn_state_kv=49fb25b3760e6207`,
  `layer5_attn_state_score=3e158062911a288e`,
  `layer42_raw_cache_row=3346c7f9ebeed46e`,
  `layer42_attn_comp_row31=6b9b38fa19457e18`,
  `layer42_index_comp_row31=2a0d37865baff695`,
  `layer42_attn_state_kv=0aa0087d1d1dcd79`, and
  `layer42_index_state_kv=1e0df1e98d453bcd`; local
  `run_parity_report.py --skip-local-oracles` passed 41/31/0.

### M10.5c4d3: Rust Long Indexed-Continuation Attention Coverage

- Status: complete
- Goal: cover the long-context ratio-4 indexed-attention branch without
  requiring Rust layer-major prefill ownership.
- Oracle: current-C GPU decode-layer execution for the selected long indexed
  continuation state and indexed mixed-attention branch.
- Fixture: B300 `ds4flash.gguf`, a deterministic long decode state whose
  ratio-4 compressed row count exceeds `DS4_N_INDEXER_TOP_K`.
- Comparator: Rust-vs-C indexed-attention tensor hashes/samples, top-k selected
  rows, raw ring metadata, compressed counters, and final selected layer output.
- Acceptance: Rust calls the indexed-attention backend with the same selected
  compressed rows and matches the current-C selected tensor checkpoints.
- Drift policy: indexed threshold, top-k rows, counter values, raw ring
  metadata, and tensor digests are exact; JSON float samples use the M10.5
  tolerant-sample policy.
- Review gate: ask Claude to review indexed attention state seeding, top-k row
  ownership, and row-selection comparator coverage.
- Validation passed: static and negative long indexed-attention comparator
  checks, local `arch -arm64 make ds4-long-indexed-attention-oracle-dump`,
  local `cargo check -p ds4-gpu --bin ds4-decode-long-indexed-attention`, B300
  current-C oracle plus Rust CUDA candidate validation with 644 checks, pinned
  local artifact validation with 666 checks, C4d2 artifact cross-check, local
  unified report with B300 rerun command coverage and 42/32/0 pass/skip/fail,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  touched-file NUL scan, non-interactive Claude review with `NO BLOCKERS`, and
  artifact SHA256
  `oracle=26aab1234b7ca7527dd2aa10f522ffce199d147187e8ec05e86ed504b79b9eed`
  and
  `rust=3406e9f746471cad0f2cbfe4f23297d2438f857b21140bf068e6386422eb4f1d`.
- Evidence: deterministic tokens `0..2051` warmed up tokens `0..2050` through
  production current-C decode, stopped token `2051` after layer `2`, crossed
  the strict ratio-4 indexed-attention threshold with `layer2_n_comp=513` and
  `layer2_n_index_comp=513`, matched `layer2_comp_selected=96be5e90e07d5fe3`,
  `layer2_heads=152cefad5f4521d0`,
  `layer2_attn_out=d31399afb15f9523`,
  `layer2_after_attn_hc=ce72c471b910e3e4`,
  `layer2_raw_cache_row=1eccdd715c4f26b1`,
  `layer2_attn_comp_row512=25b13ef81b3cc643`, and
  `layer2_index_comp_row512=8bf040cdf84597fb`. The CUDA single-token indexed
  fallback now fills selected compressed rows deterministically in top-k order.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.

### M10.5c4d4: Rust Directional-Steering Decode Coverage

- Status: done
- Goal: validate directional-steering decode execution in the Rust decode
  facade now that continuation state is comparable.
- Oracle: C `metal_graph_encode_decode_layer` directional-steering branches
  around attention output and FFN output projection, followed by the current-C
  output head.
- Fixture: B300 `ds4flash.gguf`, token `0`, layer `0`, and
  `dir-steering/out/verbosity.f32` with attention scale `0.5` and FFN scale
  `0.25`.
- Comparator: Rust-vs-C steering tensor hashes/samples for layer-0 post-steer
  attention output, post-steer attention HC expansion, post-steer FFN output,
  post-steer FFN HC expansion, and final logits.
- Acceptance: steering-enabled Rust decode matches current-C for the selected
  fixture; an exact skip is allowed only if the B300 steering artifact or a
  named safe facade operation becomes unavailable before implementation.
- Drift policy: steering file path, file hash, attn/FFN scales, layer index,
  output tensor set, and skip text are exact.
- Review gate: ask Claude to review optional tensor ownership, steering
  projection placement, and skip conditions.
- Validation passed: B300 current-C oracle plus Rust CUDA candidate validation
  with 469 pinned checks; local static comparator passed with 33 checks;
  negative tests rejected 13 mutations; `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, touched-file NUL scan, and
  non-interactive Claude review passed.
- Owner path: Rust decode execution modules, B300 comparator,
  `.memory/status.md`.

### M10.6a: Rust Prefill Scheduling Plan

- Status: done
- Goal: move the C prefill routing and chunk-boundary plan into Rust without
  executing GPU kernels.
- Oracle: C `ds4_default_prefill_cap_for_prompt`,
  `metal_graph_prefill_layer_major`, `metal_graph_prefill_chunked_range`, and
  `metal_graph_resume_prefill_min_tokens`.
- Fixture: cold 22-token whole prefill, cold 2048-token boundary whole prefill,
  cold 2052-token chunked prefill, resumed suffix from token `1537` for `800`
  tokens, short resumed suffix below the prefill threshold, and exact-prefix
  cache hit.
- Comparator: `ds4-parity/compare_prefill_plan_rust.py --negative-test` plus
  `ds4-prefill-plan` JSON candidate validation.
- Acceptance: Rust plan matches current-C route, prefill cap, raw cap, chunk
  starts/sizes, first chunk, final output batch row, progress points, and
  layer-batch call count for every fixture.
- Drift policy: env override behavior is out of scope for this default-plan
  slice; default cap/chunk/resume constants and boundary math are exact.
- Review gate: ask Claude to review chunk boundary and resume-prefix handling.
- Validation gate: comparator and negative tests, `cargo test -p ds4-gpu
  prefill_plan`, `cargo run -p ds4-gpu --bin ds4-prefill-plan`, `cargo fmt
  --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Validation passed: six static cases, six chunks, and six progress points
  matched the embedded oracle; JSON candidate validation passed; negative
  tests rejected 10 mutations; `run_parity_report.py --skip-local-oracles`
  reported 44 passed, 33 skipped, and 0 failed; `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, touched-file NUL scan,
  and non-interactive Claude review passed.
- Owner path: `rust/ds4-gpu/src/prefill_plan.rs`, `ds4-parity/`,
  `.memory/status.md`.

### M10.6b: Rust Whole-Prefill Short Execution

- Status: done
- Goal: execute a short whole-prompt layer-major prefill through the Rust safe
  facade.
- Oracle: C `metal_graph_prefill_layer_major` on B300 for a short prompt that
  fits in one prefill cap.
- Fixture: deterministic short prompt tokens under the default prefill cap.
- Comparator: Rust-vs-C prefill tensor checkpoints and final logits for the
  short whole-prefill path.
- Acceptance: Rust whole-prefill matches current-C checkpoints, final logits,
  output row, and raw/compressed counters within M10.4 tolerances.
- Drift policy: prompt tokens, output row, raw/cache counters, and selected
  tensor digest set are exact.
- Review gate: ask Claude to review layer-major command ordering and output-row
  selection.
- Validation passed: B300 `ds4-rust-port-b300` current-C oracle vs Rust
  candidate comparator passed 780 checks for
  `short_italian_fact_whole_prefill`; local static comparator passed 60
  checks; negative tests rejected 15 mutations; unified parity report with
  local oracles skipped reported 45 passed, 34 skipped, and 0 failed;
  `arch -arm64 make ds4_prefill_whole_short_oracle_dump_cpu.o`;
  `arch -arm64 make ds4-prefill-whole-short-oracle-dump`; full
  `cargo test --workspace`; `cargo fmt --all -- --check`; `git diff --check`;
  and non-interactive Claude review passed with no blockers.
- Owner path: `ds4_prefill_whole_short_oracle_dump.c`,
  `rust/ds4-gpu/src/bin/ds4-prefill-whole-short.rs`,
  `rust/ds4-gpu/src/decode_backend.rs`, `ds4-parity/`.

### M10.6c: Rust Cold Chunked-Prefill Execution

- Status: done
- Goal: execute cold chunked prefill through Rust for prompts larger than the
  default prefill cap.
- Oracle: C `metal_graph_prefill_chunked` and
  `metal_graph_prefill_chunked_range` with `start=0`.
- Fixture: a 2052-token cap-crossing prompt and a longer context slice that
  emits multiple chunk boundaries.
- Comparator: Rust-vs-live-current-C chunk boundary trace, progress-equivalent
  chunk endpoints, final logits, raw ring rows, compressed counters, and output
  digests/samples captured with deterministic CUDA MoE down projection
  (`DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` on B300).
- Acceptance: Rust cold chunked prefill matches current-C chunk schedule and
  final cache/logit state.
- Drift policy: chunk starts/sizes, final batch row, progress positions, raw
  ring mapping, compressed counters, and output digests are exact against the
  same-run current-C oracle; optimized CUDA MoE atomic-down output is treated as
  nondeterministic and excluded from exact comparison.
- Review gate: ask Claude to review chunk loop state and final-row handling.
- Validation passed: B300 `ds4-rust-port-b300` current-C oracle vs Rust
  candidate comparator passed 400 checks for the 2052-token cap-crossing prompt
  and 400 checks for the full long memory archive prompt with
  `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`; local static comparator passed 30 checks;
  negative tests rejected 5 chunked mutations and 15 whole-prefill mutations;
  unified parity report with local oracles skipped reported 46 passed, 35
  skipped, and 0 failed; `arch -arm64 make
  ds4-prefill-whole-short-oracle-dump`; `cargo check -p ds4-gpu --bin
  ds4-prefill-whole-short`; full `cargo test --workspace`; `cargo fmt --all --
  --check`; `git diff --check`; and non-interactive Claude review with no
  blockers.
- Owner path: `ds4.c`, `ds4_prefill_whole_short_oracle_dump.c`,
  `rust/ds4-gpu/src/bin/ds4-prefill-whole-short.rs`,
  `rust/ds4-gpu/src/decode_backend.rs`, `ds4-parity/`.

### M10.6d: Rust Resumed-Suffix Prefill Execution

- Status: done
- Goal: execute checkpoint extension through Rust, choosing decode for short
  suffixes and chunked prefill for longer suffixes.
- Oracle: C session checkpoint path around
  `metal_graph_resume_prefill_min_tokens` and
  `metal_graph_prefill_chunked_range`.
- Fixture: `long_memory_archive` exact-prefix cache hit at 512 tokens, short
  512-to-514 decode suffix below the default resume-prefill threshold, and
  boundary-crossing `long_memory_archive_1537_to_2337` resumed suffix with
  chunks `(1537,511)` and `(2048,289)`.
- Comparator: Rust-vs-C route decision, resume threshold, extension chunk
  starts/sizes, decode-token count, final logits, checkpoint length, raw ring
  rows, and compressed counters.
- Acceptance: Rust resumed suffix behavior matches C for cache hit, decode
  fallback, and chunked extension cases.
- Drift policy: checkpoint-prefix matching, suffix threshold, chunk alignment,
  and cache state are exact; progress timestamps are normalized.
- Review gate: ask Claude to review resume-prefix and cache-frontier handling.
- Validation passed: B300 `ds4-rust-port-b300` current-C oracle vs Rust
  candidate comparator passed 425 checks for the cache-hit fixture, 425 checks
  for the short decode-suffix fixture, and 425 checks for the resumed-chunked
  fixture with `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`; local static comparator passed
  27 checks; negative tests rejected 6 resumed-prefill mutations, 5
  chunked-prefill mutations, and 15 whole-prefill mutations; unified parity
  report with local oracles skipped reported 47 passed, 36 skipped, and 0
  failed; `arch -arm64 make ds4-prefill-whole-short-oracle-dump`; `cargo
  check -p ds4-gpu --bin ds4-prefill-whole-short`; full `cargo test
  --workspace`; `cargo fmt --all -- --check`; `python3 -m py_compile
  ds4-parity/compare_prefill_resumed.py ds4-parity/run_parity_report.py`;
  `git diff --check`; and non-interactive Claude review with `NO BLOCKERS`.
- Owner path: `ds4.c`, `ds4.h`,
  `ds4_prefill_whole_short_oracle_dump.c`,
  `rust/ds4-gpu/src/bin/ds4-prefill-whole-short.rs`,
  `ds4-parity/compare_prefill_resumed.py`,
  `ds4-parity/run_parity_report.py`, `ds4-parity/README.md`,
  `.memory/status.md`.

### M10.7a: Rust Graph Session Payload Layout Plan

- Status: done
- Goal: make Rust own the graph-session payload byte/row/count plan before
  writing or restoring live tensors.
- Oracle: current C graph session payload sizing and row-order rules in
  `ds4_session_payload_bytes`, `ds4_session_save_payload`, and
  `ds4_session_load_payload`, exposed through a no-model graph layout dump.
- Fixture: default `ctx=32768` graph payload plans for
  `short_checkpoint_tokens3`, `continued_frontier_tokens924`,
  `prefill_cap_cross_tokens2052`, `raw_ring_wrap_tokens2305`, and
  `near_context_tokens32767`.
- Comparator: C-vs-Rust graph payload plan comparator covering header fields,
  prefill/raw/comp caps, raw live rows, logical raw row order, physical raw
  ring rows, ratio-4 and ratio-128 compressed/indexed row counts, per-section
  byte totals, and final payload bytes.
- Acceptance: Rust emits the same graph payload layout plan as C for every
  fixture without loading a model or claiming tensor restore support.
- Drift policy: layout fields, row counts, ring mapping, section byte totals,
  and payload bytes are exact; graph env overrides remain out of scope for this
  default-layout slice.
- Review gate: ask Claude to review payload byte accounting and raw-ring
  ordering before commit.
- Validation passed: `arch -arm64 make ds4-session-payload-dump`; C and Rust
  `--graph-plan` JSON parse checks; `python3
  ds4-parity/compare_graph_session_payload_plan.py` passed 901 checks; `python3
  ds4-parity/compare_graph_session_payload_plan.py --negative-test` passed 901
  checks and rejected 7 mutations; `cargo test -p ds4-gguf session_payload`;
  `python3 -m py_compile ds4-parity/compare_graph_session_payload_plan.py
  ds4-parity/run_parity_report.py`; `git diff --check`; `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` reported 48 passed, 36
  skipped, and 0 failed; `cargo test --workspace`; `cargo fmt --all --
  --check`; and non-interactive Claude review with `NO BLOCKERS`.
- Owner path: `ds4.c`, `ds4.h`, `ds4_session_payload_dump.c`,
  `rust/ds4-gguf/src/session_payload.rs`,
  `rust/ds4-gguf/src/bin/ds4-session-payload-dump-rs.rs`, `ds4-parity/`,
  `.memory/status.md`.

### M10.7b: Rust Graph Session Payload Reader And Writer

- Status: done
- Goal: add Rust-owned graph session payload header/body parsing and write-plan
  helpers for live graph state snapshots.
- Oracle: M10.7a layout plan, M7.5 session payload structural fixtures, and
  current C load rejection categories.
- Fixture: synthetic graph payload bodies with valid, truncated, trailing,
  invalid compressed count, invalid index count, raw-ring mismatch, and
  context/layout mismatch cases.
- Comparator: Rust reader/writer round-trip plus C rejection-code comparator
  for the same payload bytes.
- Acceptance: Rust validates and serializes C-compatible graph payload bytes
  without restoring GPU tensors yet.
- Drift policy: payload bytes, section order, error classes, and count bounds
  are exact.
- Review gate: ask Claude to review binary bounds checks and error mapping.
- Validation passed: `arch -arm64 make ds4-session-payload-dump`; C and Rust
  `--graph-probe` JSON parse checks; `python3
  ds4-parity/compare_graph_session_payload_rw.py` passed 375 checks; `python3
  ds4-parity/compare_graph_session_payload_rw.py --negative-test` passed 375
  checks and rejected 7 mutations; `cargo test -p ds4-gguf session_payload`;
  `python3 -m py_compile ds4-parity/compare_graph_session_payload_rw.py
  ds4-parity/run_parity_report.py`; `git diff --check`; `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` reported 49 passed, 36
  skipped, and 0 failed; `cargo test --workspace`; `cargo fmt --all --
  --check`; non-interactive Claude review with `NO BLOCKERS`; and focused
  post-review C build/comparator/Rust test reruns after the style-nit cleanup.
- Owner path: Rust graph session payload reader/writer, C rejection probe,
  `ds4-parity/`, `.memory/status.md`.

### M10.7c: Rust Disk KV Payload Restore Smoke

- Status: split into M10.7c1-M10.7c3 before implementation; M10.7c1 and
  M10.7c2 done; M10.7c3 split into M10.7c3a-M10.7c3d before tensor restore;
  M10.7c3a, M10.7c3b, M10.7c3c, and M10.7c3d done; M10.7d active.
- Goal: advance disk and memory restore parity in slices: committed restore
  metadata first, raw B300 payload bytes second, and tensor restore behavior
  third.
- Oracle: current C M7.8 restore oracle and M0.5 disk KVC artifacts on B300.
- Acceptance: each subitem has its own oracle, fixture, comparator, and
  validation gate before any tensor-restore claims.
- Owner path: Rust graph restore runtime, B300 restore comparator,
  `ds4-parity/`, `.memory/status.md`.

### M10.7c1: Rust Restore Payload Header Contract

- Status: done
- Goal: prove Rust graph payload planning matches the committed M7.8 restore
  oracle headers and payload byte counts before loading raw restore bodies.
- Oracle: `ds4-parity/baselines/kv/m7.8/current-c.json` header prefixes,
  payload/snapshot byte counts, prompt-token counts, and fixture identity.
- Fixture: disk and in-memory restore records for seed and continuation prompts
  on `/workspace/ds4/ds4flash.gguf`.
- Comparator: current-C restore oracle vs Rust restore-header plan over case
  order, kind, prompt tokens, header prefix bytes, graph caps, raw-live rows,
  payload/snapshot byte counts, and hash-only body policy.
- Acceptance: Rust emits the same DSV4 graph payload header and byte budget as
  the M7.8 current-C restore records without reading raw payload bodies or
  claiming tensor restore.
- Drift policy: header bytes, prompt-token counts, payload byte counts, model
  identity, and raw-body hash-only policy are exact.
- Review gate: ask Claude to review that this remains a header/size contract
  and does not overclaim restore behavior.
- Validation passed: Rust restore-header JSON emission and JSON parse;
  `python3 ds4-parity/compare_restore_payload_header_plan.py` passed 127
  checks; `python3 ds4-parity/compare_restore_payload_header_plan.py
  --negative-test` passed 127 checks and rejected 7 mutations; targeted Rust
  test `restore_header_contract_matches_m78_payload_sizes`;
  `python3 -m py_compile ds4-parity/compare_restore_payload_header_plan.py
  ds4-parity/run_parity_report.py`; `git diff --check`; `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` reported 50 passed, 36
  skipped, and 0 failed; `cargo test --workspace`; `cargo fmt --all --
  --check`; and non-interactive Claude review with `NO BLOCKERS`.
- Owner path: Rust restore header plan, M7.8 restore comparator, `ds4-parity/`,
  `.memory/status.md`.

### M10.7c2: Rust Disk KV Payload Byte Import Smoke

- Status: done
- Goal: on B300, feed the raw C-written disk KVC restore payload bytes into the
  Rust graph payload reader and prove Rust accepts the bytes with the recorded
  hashes and section plan.
- Oracle: M7.8 disk payload raw files and `payload_sha256` records on B300.
- Fixture: seed and continuation disk restore payload bodies in the M7.8 raw
  artifact location on `/workspace/ds4`.
- Comparator: B300 Rust payload-reader smoke over observed and historical
  payload SHA256 metadata, header fields, payload length, section byte plan,
  compressed/index counts, and rejection of mutated summaries.
- Acceptance: Rust can import the same raw disk payload bytes as C at the reader
  level, without restoring tensors into graph memory yet.
- Drift policy: lengths, header fields, count tables, section offsets, and
  Rust-reader acceptance are exact; raw payload hashes are per-capture metadata
  because B300 restore bodies can drift while preserving restore behavior.
- Review gate: ask Claude to review raw-byte bounds checks and B300 evidence.
- Validation passed: B300 live raw import with summary writeback passed 104
  checks and rejected 7 mutations; `python3
  ds4-parity/compare_graph_payload_raw_import.py --negative-test` passed 100
  checks and rejected 7 mutations; `python3 -m py_compile
  ds4-parity/compare_graph_payload_raw_import.py
  ds4-parity/run_parity_report.py`; `cargo test -p ds4-gguf
  session_payload`; `git diff --check`; `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` reported 51 passed, 37
  skipped, and 0 failed; `cargo test --workspace`; `cargo fmt --all --
  --check`; and non-interactive Claude review with `NO BLOCKERS`.
- Owner path: Rust graph payload byte reader, B300 raw payload comparator,
  `.memory/status.md`.

### M10.7c3: Rust Graph Tensor Restore Next-Token Smoke

- Status: split into M10.7c3a-M10.7c3d before implementation; M10.7c3a,
  M10.7c3b, M10.7c3c, and M10.7c3d done.
- Goal: advance graph restore from raw memory snapshot availability, to restore
  target mapping, to tensor readback, and finally to next-token behavior.
- Oracle: current C M7.8 restore oracle on B300.
- Acceptance: each subitem has a concrete raw-body or restore-state comparator
  before claiming next-token parity.
- Owner path: current-C restore dumper, Rust graph payload reader, Rust graph
  restore runtime, B300 restore comparators, `.memory/status.md`.

### M10.7c3a: Rust Memory Snapshot Raw Body Import Smoke

- Status: done
- Goal: materialize C-written in-memory snapshot bodies on B300 and feed them to
  the Rust graph payload reader with the recorded snapshot hashes and section
  plan.
- Oracle: M7.8 `snapshot_seed` and `snapshot_continuation` records in the
  current-C B300 restore oracle.
- Fixture: raw memory snapshot bodies emitted by `ds4-restore-dump
  --snapshot-dir` into `/workspace/ds4/ds4-parity/baselines/kv/m7.8/raw`.
- Comparator: B300 Rust snapshot-reader smoke over observed and historical
  snapshot SHA256 metadata, header fields, snapshot length, section byte plan,
  compressed/index counts, and rejection of mutated summaries.
- Acceptance: Rust can import the same raw memory snapshot bytes as C at the
  reader level, without restoring tensors into graph memory yet.
- Drift policy: lengths, header fields, count tables, section offsets, and
  Rust-reader acceptance are exact; snapshot body hashes are per-capture
  metadata because B300 restore bodies can drift while preserving restore
  behavior.
- Review gate: ask Claude to review snapshot body materialization, raw-byte
  bounds checks, and B300 evidence.
- Validation passed: B300 raw disk import rerun under the corrected per-capture
  hash policy passed 108 checks and rejected 9 mutations; B300 raw snapshot
  materialization/import passed 110 checks and rejected 9 mutations; local
  `python3 ds4-parity/compare_graph_payload_raw_import.py --negative-test`
  passed 104 checks and rejected 9 mutations; local `python3
  ds4-parity/compare_graph_snapshot_raw_import.py --negative-test` passed 104
  checks and rejected 9 mutations; `python3 -m py_compile
  ds4-parity/compare_graph_payload_raw_import.py
  ds4-parity/compare_graph_snapshot_raw_import.py ds4-parity/run_parity_report.py
  ds4-parity/check_restore_dump.py`; `cargo test -p ds4-gguf
  session_payload`; `git diff --check`; `arch -arm64 make ds4-restore-dump`;
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` reported 52
  passed, 38 skipped, and 0 failed; `cargo fmt --all -- --check`; `cargo test
  --workspace`; and non-interactive Claude review with no blockers.
- Owner path: current-C restore dumper, Rust graph payload byte reader, B300 raw
  snapshot comparator, `.memory/status.md`.

### M10.7c3b: Rust Graph Restore Target Mapping Contract

- Status: done
- Goal: map every parsed disk payload and memory snapshot section to the Rust
  graph restore destination and counter update without moving tensor bytes yet.
- Oracle: current C `ds4_session_load_payload` graph restore order and the M7.8
  raw disk/snapshot payload summaries.
- Fixture: seed and continuation disk payload and memory snapshot section plans.
- Comparator: Rust restore-target plan vs C restore order over raw logical to
  physical ring positions, per-layer compressed/index sections, checkpoint
  token/logit sections, and graph counter writes.
- Acceptance: every restore byte section has one destination and every graph
  counter update has a documented source before GPU writes are introduced.
- Drift policy: section order, destination names, per-layer coverage, and counter
  values are exact.
- Review gate: ask Claude to review that the mapping matches C graph restore
  semantics and does not claim execution behavior.
- Validation passed: `python3
  ds4-parity/compare_graph_restore_target_plan.py --negative-test` passed 6012
  checks and rejected 8 mutations; explicit candidate-file comparison with
  `cargo run -p ds4-gguf --bin ds4-session-payload-dump-rs --quiet --
  --restore-target-plan` passed 6012 checks; `python3 -m py_compile
  ds4-parity/compare_graph_restore_target_plan.py ds4-parity/run_parity_report.py`;
  `cargo test -p ds4-gguf session_payload`; `cargo fmt --all -- --check`; `git
  diff --check`; `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
  reported 53 passed, 38 skipped, and 0 failed; `cargo test --workspace`; and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph restore target plan, `ds4-parity/`,
  `.memory/status.md`.

### M10.7c3c: Rust Graph Tensor Restore Readback Smoke

- Status: done
- Goal: write C disk payload and memory snapshot bytes into Rust-owned graph
  tensor allocations on B300 and read back deterministic section hashes.
- Oracle: M10.7c3b restore-target plan plus M7.8 raw body hashes.
- Fixture: seed and continuation disk payload and memory snapshot raw bodies.
- Comparator: B300 Rust restore readback summary over restored checkpoint
  tokens, logits hash, raw-cache section hashes, compressed-cache section hashes,
  state-tensor hashes, and graph counters.
- Acceptance: Rust writes each restore section to the expected tensor destination
  and readback hashes match the imported raw bytes before decode execution.
- Drift policy: payload hashes, tensor section hashes, checkpoint length, and
  graph counters are exact.
- Review gate: ask Claude to review tensor write/readback bounds and B300
  evidence.
- Validation passed: B300 live
  `python3 ds4-parity/compare_graph_restore_readback.py --live --workdir
  /workspace/ds4 --write-summary /tmp/ds4-m107c3c-restore-readback.json
  --negative-test` passed 1365 checks and rejected 8 mutations; local `python3
  ds4-parity/compare_graph_restore_readback.py --negative-test` passed 1365
  checks and rejected 8 mutations; `python3 -m py_compile
  ds4-parity/compare_graph_restore_readback.py ds4-parity/run_parity_report.py`;
  `cargo test -p ds4-gpu --bin ds4-graph-restore-readback`; `cargo check -p
  ds4-gpu --bin ds4-graph-restore-readback`; `cargo fmt --all -- --check`;
  `git diff --check`; `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` reported 54 passed, 39 skipped, and 0 failed; `cargo
  test --workspace`; and non-interactive Claude review with no blockers.
- Owner path: Rust graph restore runtime, B300 restore readback comparator,
  `.memory/status.md`.

### M10.7c3d: Rust Graph Tensor Restore Next-Token Smoke

- Status: done
- Goal: restore C-written disk and memory snapshot payloads into Rust graph
  session state on B300 and prove next-token behavior matches current C.
- Oracle: current C M7.8 restore oracle on B300 plus M10.7c3c tensor readback
  evidence.
- Fixture: seed disk payload restore, continuation disk payload restore, and
  in-memory snapshot restore on `/workspace/ds4/ds4flash.gguf`.
- Comparator: B300 Rust-vs-current-C restore comparator over payload hashes,
  checkpoint tokens, selected token, top-logprob order, cache source, and graph
  counters.
- Acceptance: Rust-restored sessions produce the same next-token state as the C
  restore oracle for the committed fixtures.
- Drift policy: payload body hashes, restored checkpoint length, selected token,
  top-logprob order, cache source, and graph counters are exact. Raw body
  SHA256 values are per-capture metadata, so exact top-logprob scores compare
  against the same-capture current-C restore oracle.
- Review gate: ask Claude to review restore invariants and B300 evidence.
- Validation passed: B300 live `python3
  ds4-parity/compare_graph_restore_next_token.py --live --workdir
  /workspace/ds4 --write-summary /tmp/ds4-m107c3d-restore-next-token.json
  --negative-test` passed 4030 checks and rejected 11 mutations; local
  `python3 ds4-parity/compare_graph_restore_next_token.py --negative-test`
  passed 4030 checks and rejected 11 mutations; `python3 -m py_compile
  ds4-parity/compare_graph_restore_next_token.py ds4-parity/run_parity_report.py`;
  `cargo check -p ds4-gpu --bin ds4-graph-restore-next-token`; `cargo test -p
  ds4-gpu --bin ds4-graph-restore-next-token`; `cargo fmt --all -- --check`;
  `git diff --check`; `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` reported 55 passed, 40 skipped, and 0 failed; `cargo
  test --workspace`; and non-interactive Claude review with no blockers.
- Owner path: Rust graph restore runtime, B300 restore comparator,
  `.memory/status.md`.

### M10.7d1: Continued-Frontier Policy Transition Matrix

- Status: done
- Goal: make the Rust continued-frontier policy matrix cover C target
  selection, note-store updates, cold-store suppression, failed-store restore,
  already-stored skips, disabled policy, and reset-after-miss behavior before
  touching graph tensors.
- Oracle: current C `kv_cache_continued_store_target`,
  `kv_cache_suppress_continued_store`,
  `kv_cache_restore_suppressed_continued`, M7.2 policy baseline rows, and the
  M9.8f4 runtime store invariants.
- Fixture: no-model synthetic cases for fresh frontier, below-min, unaligned,
  already stored, disabled, align-zero, no-interval, suppressed cold frontier,
  restore-on-failure, ignore-non-suppressed restore, and reset-after-cache-miss.
- Comparator: Rust-vs-C continued target, old frontier, new frontier,
  suppression result, restore result, and reset state.
- Acceptance: Rust policy helpers and runtime cache state expose the same
  frontier transitions as current C without requiring a model or B300.
- Drift policy: target tokens, old/new frontier tokens, skip reasons encoded by
  case names, and reset/suppression/restore transitions are exact.
- Review gate: ask Claude to review the policy matrix for missing C
  transition cases.
- Validation passed: `python3 ds4-parity/check_kv_policy_dump.py
  --negative-test` with 521 schema checks, 11 manifest checks, and 8 negative
  checks; `python3 ds4-parity/compare_kv_policy.py --negative-test` with 1725
  comparator checks and 9 negative checks; `python3
  ds4-parity/compare_kv_replay.py --negative-test`; `python3
  ds4-parity/run_kv_parity_report.py` with 9 passed, 1 skipped, and 0 failed;
  `python3 ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped,
  and 0 failed; targeted Rust tests for continued-store policy and runtime
  reset; Python syntax checks; `cargo test --workspace`; `cargo fmt --all --
  --check`; `git diff --check`; `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 55 passed, 40 skipped, and 0 failed; and
  non-interactive Claude review with no blockers.
- Owner path: Rust KV policy dump/comparator, Rust runtime cache-state tests,
  `.memory/status.md`.

### M10.7d2a: Runtime Continued-Frontier Ledger Contract

- Status: done
- Goal: add a model-free Rust runtime cache ledger that records the
  continued-frontier decisions made around cache misses, memory hits, disk
  restores, cold-store suppression, note-store updates, failed-store restore,
  and decode-time continued-store attempts without changing graph tensors.
- Oracle: current C `generate_job` ordering around
  `kv_cache_store_current`, `kv_cache_suppress_continued_store`,
  `kv_cache_restore_suppressed_continued`, `kv_cache_maybe_store_continued`,
  and the M10.7d1 policy transition matrix.
- Fixture: synthetic runtime cache probes and store outcomes for fresh miss
  reset, memory-token hit, disk restore hit, cold prefix store success,
  cold prefix store failure, full-prompt cold store, continued-store hit/skip,
  and tool-call decode suppression.
- Comparator: event order, cache source, cached tokens, cache write tokens,
  disk cached tokens, continued frontier before/after, store reason, store
  success, and rollback result.
- Acceptance: Rust runtime state exposes the same decision sequence current C
  uses, with no model or B300 dependency.
- Drift policy: event names/order, token counts, reason names, success flags,
  and frontier before/after values are exact.
- Review gate: ask Claude to review runtime event ordering and rollback
  invariants.
- Validation passed: focused runtime ledger tests, KV policy comparator, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  skip-local unified parity report, and non-interactive Claude review with no
  blockers.
- Evidence: Rust runtime cache state now records per-request cache decision and
  continued-frontier events for reset, suppression, restore, note, live-prefix
  store, current-store, and continued-store attempts without requiring a model
  or B300.
- Owner path: Rust runtime cache/store ledger, runtime tests,
  `.memory/status.md`.

### M10.7d2b: Runtime KV Replay Checker Closure

- Status: done
- Goal: extend the committed runtime KV replay checker/artifact contract so
  M0.5 seed miss, seed restore, continuation restore, and memory-token
  continuation validate continued-frontier ledger fields in addition to cache
  source and token counts.
- Oracle: M9.8f5 B300 runtime replay summary, M7.7 KV replay comparator, M0.5
  current-C artifacts, and the M10.7d2a ledger contract.
- Fixture: committed M9.8f5 summary, M0.5 KV fixtures, and a model-free
  memory-token continuation trace.
- Comparator: summary schema/checker fields for cache source, cached/write
  tokens, disk cached tokens, KVC reason fields, ledger event order, and
  continued frontier before/after.
- Acceptance: checked-in replay evidence fails if runtime cache accounting or
  continued-frontier ledger semantics drift.
- Drift policy: timestamps, paths, and raw payload hashes are normalized; cache
  source, token counts, reason fields, event order, and frontier transitions
  are exact.
- Review gate: ask Claude to review checker coverage and artifact semantics.
- Validation passed: runtime KV replay checker with negative tests, KV replay
  comparator, targeted Rust runtime tests, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, skip-local unified parity report,
  and non-interactive Claude review with no blockers.
- Evidence: `check_runtime_kv_replay_summary.py` now validates the M9.8f5 B300
  replay summary plus a model-free M10.7d2 ledger contract covering seed miss,
  seed restore, continuation restore, and memory-token continuation event
  order/frontier transitions; `run_server_parity_report.py` runs the checker
  with negative tests.
- Owner path: runtime KV replay checker/summary contract,
  `.memory/status.md`.

### M10.7d2c: Runtime Continued-Store B300 Replay Refresh

- Status: done
- Goal: refresh the model-backed B300 Rust runtime replay so real seed miss,
  seed restore, continuation restore, and continued-store decisions include the
  M10.7d2 ledger evidence.
- Oracle: current C runtime cache/store ordering, M9.8f5 replay commands, M0.5
  current-C artifacts, and M10.7d2b checker contract.
- Fixture: B300 `/workspace/ds4/ds4flash.gguf`, M0.5 seed and continuation
  request fixtures, runtime traces, KVC headers, and ledger summary fields.
- Comparator: B300 replay summary comparing response content, cache source,
  cached/write tokens, disk tokens, KVC reasons, ledger events, and frontier
  transitions.
- Acceptance: real Rust runtime replay matches current-C store/restore
  decisions and produces checked-in ledger evidence.
- Drift policy: generated text, cache source, token counts, reason fields,
  event order, and frontier transitions are exact; paths, timestamps, and raw
  payload hashes are normalized.
- Review gate: ask Claude to review B300 command fidelity and replay coverage.
- Validation passed: B300 runtime replay, checker negative tests, KV replay
  comparator, server parity report, JSON/Python syntax checks, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  skip-local unified parity report, and non-interactive Claude review with no
  blockers.
- Evidence: runtime traces now include a stable runtime cache ledger section,
  the M9.8f5 B300 summary records checked `ledger_cases` plus raw trace event
  counts/names, and the checker validates six negative mutations. The live
  B300 M0.5 replay passed after using a 20-second startup wait for CUDA model
  cache initialization.
- Owner path: B300 runtime replay summary/checker, `.memory/status.md`.

### M10.7d3a: Graph Restore Frontier Contract

- Status: done
- Goal: define a model-free restore/frontier contract that maps restored graph
  payload token counts onto continued-frontier decisions before any new B300
  graph/KVC smoke is trusted.
- Oracle: M10.7c3d same-capture restore cases, M9.8f4 continued-store runtime
  behavior, M10.7d1 continued-frontier transition matrix, and current C KVC
  reason/header fields.
- Fixture: committed M10.7c3d restore summary, M7.2 continued-frontier policy
  matrix, M9.8f4 cold/continued/shutdown store evidence, and M0.5 KVC header
  rows.
- Comparator: a restore-frontier contract checker covering restored token
  counts, loaded frontier values, re-enabled continued-store targets,
  already-stored skip cases, and KVC reason expectations.
- Acceptance: the checked-in contract fails if restored graph token counts,
  continued-frontier transitions, or expected post-restore reason fields drift.
- Drift policy: restored token counts, frontier before/after values, event
  order, KVC reason names, and skip/write decisions are exact; paths and raw
  payload hashes remain normalized under the M10.7c3d same-capture policy.
- Review gate: ask Claude to review contract coverage and oracle selection.
- Validation passed: contract checker with negative tests, KV policy
  comparator, graph restore next-token comparator, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Evidence: added
  `ds4-parity/baselines/kv/m10.7d3/restore-frontier-contract.json` and
  `ds4-parity/check_graph_restore_frontier_contract.py`. The checker validates
  restored graph token counts, loaded frontier values, re-enabled continued
  targets, already-stored skip behavior, and KVC reason references against
  M10.7c3d, M7.2, and M0.5 artifacts with seven negative mutations, and is
  wired into the unified parity report. Non-interactive Claude review returned
  no blockers.
- Owner path: restore-frontier contract/checker, `.memory/status.md`.

### M10.7d3b: B300 Restored-Graph Frontier Projection

- Status: done
- Goal: extend the B300 Rust graph restore smoke so each restored payload emits
  continued-frontier projection evidence derived from the actual restored graph
  token counts.
- Oracle: M10.7d3a contract, M10.7c3d same-capture current-C restore oracle,
  and Rust restored-payload readback evidence.
- Fixture: B300 raw graph payload and memory snapshot bodies for
  `disk_seed_payload`, `snapshot_seed`, `disk_continuation_payload`, and
  `snapshot_continuation`, plus the M10.7c3d current-C restore capture.
- Comparator: B300 summary fields for restored token count, loaded frontier,
  next continued target, already-stored skip, selected token/top-logprob
  evidence, and graph counters.
- Acceptance: Rust-restored graph payloads report the same frontier projection
  decisions as the M10.7d3a contract while preserving M10.7c3d next-token
  evidence.
- Drift policy: restored token counts, selected token/top-logprob evidence,
  frontier projection fields, and graph counters are exact within the
  same-capture policy.
- Review gate: ask Claude to review B300 command fidelity and restored-payload
  frontier invariants.
- Validation needed: B300 live graph restore projection capture, comparator
  with negative tests, targeted Rust tests, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Evidence: `ds4-graph-restore-next-token` now emits
  `frontier_projection` for each restored payload, including loaded frontier,
  unaligned current-live skip, next continued target, already-stored boundary
  skip, and shutdown reason projection. The B300 live comparator refreshed
  `ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json` and
  passed 4177 checks plus 12 negative mutations; full workspace, formatting,
  diff, unified parity, and non-interactive Claude validation also passed.
- Owner path: Rust graph restore smoke, B300 comparator, `.memory/status.md`.

### M10.7d3c: Post-Restore KVC Write/Skip B300 Smoke

- Status: split into M10.7d3c1 and M10.7d3c2
- Goal: prove a restored graph checkpoint followed by save/skip decisions
  writes or skips KVC checkpoints with current-C-compatible reason fields and
  continued-frontier state.
- Oracle: M10.7d3b restored-graph frontier projection, M9.8f4 runtime KVC
  store behavior, current C graph restore, and current C KVC reason/header
  fields.
- Fixture: B300 fresh miss, exact-prefix disk restore, continuation restore,
  restored-frontier re-enable, already-stored frontier skip, and shutdown or
  continued KVC write after restore.
- Comparator: restored checkpoint/logits evidence, cache source, cached/write
  token counts, continued-frontier state, KVC reason fields, and graph
  counters.
- Acceptance: Rust-restored graph sessions continue with the same cache
  store/skip decisions as current C after restore.
- Drift policy: restored token counts, frontier tokens, reason fields, graph
  counters, and write/skip decisions are exact; paths, timestamps, and raw
  payload hashes are normalized.
- Review gate: ask Claude to review B300 command fidelity and post-restore KVC
  invariants.
- Validation needed: B300 post-restore KVC smoke, runtime KV replay
  comparator, graph restore projection comparator, targeted Rust tests, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Owner path: Rust graph restore/runtime cache integration, B300 comparator,
  `.memory/status.md`.

### M10.7d3c1: Post-Restore KVC Decision Contract

- Status: done
- Goal: define a model-free post-restore KVC decision contract before trusting
  a B300 file-writing smoke.
- Oracle: M10.7d3b restored-graph frontier projection, M10.7d2 runtime ledger
  decisions, M0.5 KVC header rows, and the M7.4a KVC file layout oracle.
- Fixture: restored token counts and frontier projections for the four
  M10.7d3b graph restore cases, M9.8f5 runtime ledger events, M0.5 KVC header
  rows, and KVC reason-code constants.
- Comparator: a model-free checker covering unaligned post-restore continued
  skips, re-enabled next continued targets, already-stored boundary skips, and
  shutdown-write header expectations for restored graph payloads.
- Acceptance: the contract fails if restored token counts, KVC reason names,
  reason codes, header fields, write/skip decisions, or continued-frontier
  transitions drift.
- Drift policy: restored token counts, frontier values, KVC reason fields,
  header fields, and skip/write decisions are exact; path and raw body hashes
  remain normalized under the M10.7c3d same-capture policy.
- Review gate: ask Claude to review oracle coverage and whether the contract
  is strong enough to gate the B300 smoke.
- Validation needed: contract checker with negative tests, graph restore
  projection comparator, runtime KV replay checker, KVC file comparator,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.
- Evidence: added
  `ds4-parity/baselines/kv/m10.7d3/post-restore-kvc-decision-contract.json`
  and `ds4-parity/check_post_restore_kvc_decision_contract.py`. The checker
  passed 4 post-restore cases, 3 runtime references, and 8 negative mutations,
  and is wired into the unified parity report.
- Owner path: post-restore KVC contract/checker, `.memory/status.md`.

### M10.7d3c2: B300 Restored Payload KVC File Smoke

- Status: done
- Goal: run a B300 Rust smoke that wraps restored graph payload bodies in KVC
  files and records the matching post-restore skip decisions.
- Oracle: M10.7d3c1 post-restore KVC decision contract, M10.7d3b
  same-capture restored graph evidence, and current C KVC file/header behavior.
- Fixture: B300 raw graph payload and memory snapshot bodies for
  `disk_seed_payload`, `snapshot_seed`, `disk_continuation_payload`, and
  `snapshot_continuation`, plus rendered cache-text keys derived from the same
  current-C restore capture.
- Comparator: B300 summary fields for KVC file names, headers, payload byte
  counts, payload digests, rendered text byte counts, skip decisions, restored
  frontier state, and graph counters.
- Acceptance: each restored graph payload can produce a current-C-compatible
  shutdown KVC file, while unaligned and already-stored continued-store probes
  skip exactly as the contract predicts.
- Drift policy: KVC header fields, reason names/codes, token counts, payload
  bytes, text bytes, skip/write decisions, and graph counters are exact; raw
  body hashes remain same-capture evidence.
- Review gate: ask Claude to review B300 command fidelity and KVC
  header/payload invariants.
- Validation needed: B300 post-restore KVC file smoke, comparator with negative
  tests, targeted Rust tests, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added `ds4-post-restore-kvc-smoke`,
  `ds4-parity/compare_post_restore_kvc_smoke.py`, and
  `ds4-parity/baselines/kv/m10.7d3/rust-b300-post-restore-kvc.json`. The live
  B300 run passed 536 comparator checks and seven negative mutations while
  proving the four restored graph payload bodies can be wrapped in shutdown
  KVC files whose headers, file sizes, payload digests, rendered text keys,
  skip decisions, restored frontiers, and graph counters match the M10.7d3c1
  contract and M10.7d3b evidence. Non-interactive Claude review returned
  `NO BLOCKERS`.
- Owner path: Rust graph restore KVC smoke, B300 comparator,
  `.memory/status.md`.

### M10.8: Rust MTP Draft And Verifier Orchestration

- Status: split into M10.8a through M10.8g
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
- Evidence: split before implementation into M10.8a through M10.8g. The final
  split gives MTP availability/contract, model-free planner, MTP draft kernel
  orchestration, exact N=2 verifier orchestration, suffix/microbatch verifier
  orchestration, speculative frontier mutation, and end-to-end stream parity
  separate validation surfaces. Non-interactive Claude review returned
  `NO BLOCKERS` after the missing draft/suffix stages were added.
- Owner path: Rust MTP graph orchestration, B300 MTP comparator,
  `.memory/status.md`.

### M10.8a: MTP State Machine Contract And Availability Check

- Status: done
- Goal: capture the current-C speculative decode decision contract and B300
  MTP support-artifact availability before any Rust MTP execution is trusted.
- Oracle: `ds4_engine_has_mtp`, `ds4_engine_mtp_draft_tokens`,
  `ds4_session_decode_speculative`, `metal_graph_eval_mtp_draft`,
  `metal_graph_verify_decode2_exact`, `metal_graph_verify_suffix_tops`,
  `spec_frontier_snapshot`, `spec_frontier_restore`, and
  `spec_frontier_commit_prefix1`.
- Fixture: MTP disabled, missing MTP support model, first-draft miss, exact
  N=2 two-token accept, exact N=2 one-token prefix1 accept, verifier failure
  rollback, microbatch full accept, microbatch prefix1 accept, replay fallback,
  and sequential safety fallback cases.
- Comparator: model-free JSON contract plus checker that pins guard conditions,
  verifier path selection, checkpoint mutations, accepted-token counts, logits
  source, frontier snapshot/restore/commit calls, and B300 support-artifact
  availability/blocker state.
- Acceptance: every later M10.8 implementation case has a named current-C
  oracle row and exact expected state transition; if no MTP GGUF is present on
  B300, the enabled live smoke remains explicitly blocked instead of silently
  degrading to MTP-off behavior.
- Drift policy: guard flags, environment-variable gates, accepted counts,
  fallback names, checkpoint length transitions, and frontier operations are
  exact; timings and probe log durations are ignored.
- Review gate: ask Claude to review contract coverage against current-C MTP
  control flow and B300 availability evidence.
- Validation needed: contract checker with negative tests, Python/JSON syntax,
  `cargo fmt --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added
  `ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json` and
  `ds4-parity/check_mtp_state_machine_contract.py`. The checker pins 12
  current-C MTP decision rows, 10 source anchors, M10.2 command boundaries,
  M8.12b missing-MTP support-artifact evidence, and seven negative mutations.
  A live B300 availability check confirmed `/workspace/ds4/missing-mtp.gguf`
  is absent and no `*mtp*.gguf` or `*draft*.gguf` candidate exists under
  `/workspace/ds4` at depth 3. Validation passed the checker negative tests,
  JSON/Python syntax, `cargo fmt --all -- --check`, `git diff --check`, the
  live B300 availability command, unified parity with 59 passed, 42 skipped,
  and 0 failed, and non-interactive Claude review with `NO BLOCKERS`.
- Owner path: MTP contract baseline/checker, parity report,
  `.memory/status.md`.

### M10.8b: Rust MTP Decision Planner

- Status: done
- Goal: add Rust-owned model-free planning for MTP verifier selection,
  accepted-prefix decisions, fallback routing, and logits/frontier ownership
  without executing GPU kernels.
- Oracle: M10.8a state-machine contract.
- Fixture: the M10.8a decision rows, including disabled/missing-support rows
  and verifier success/failure rows.
- Comparator: Rust planner JSON compared exactly to the M10.8a contract for
  selected verifier path, snapshot requirement, commit/restore action,
  accepted token count, checkpoint mutation, and logits source.
- Acceptance: Rust can fail closed for unavailable MTP, choose the same C
  verifier/fallback path for each row, and describe state mutations without
  touching target-stream logits or graph tensors.
- Drift policy: planner enums, operation order, accepted counts, and
  fail-closed errors are exact; runtime timing and log text are ignored.
- Review gate: ask Claude to review target-stream safety and fail-closed
  behavior.
- Validation needed: Rust planner tests, comparator with negative tests,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  unified parity report, and non-interactive Claude review with no blockers.
- Evidence: added `rust/ds4-gpu/src/mtp_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-decision-plan.rs`, and
  `ds4-parity/compare_mtp_decision_plan.py`. The Rust planner emits the 12
  model-free M10.8a decision rows with explicit accepted suffix, frontier
  action, logits source, fallback, and fail-closed fields before GPU kernels
  are involved. Validation passed targeted Rust planner tests, the comparator
  with 12 cases, 194 checks, and 6 negative mutations, JSON output parsing,
  Python syntax, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, unified parity with 60 passed, 42 skipped, and 0 failed, and
  non-interactive Claude review with `NO BLOCKERS`.
- Owner path: Rust MTP planner, comparator, `.memory/status.md`.

### M10.8c: Rust MTP Draft Kernel Orchestration Smoke

- Status: done
- Goal: move Rust-owned orchestration for the MTP draft graph path while
  preserving the current-C draft top-id/logits contract.
- Oracle: `metal_graph_eval_mtp_draft`,
  `metal_graph_eval_mtp_draft_from_hc`, M10.2 command boundaries, and the
  M10.8a draft rows.
- Fixture: first-draft path, repeated draft-from-HC path, MTP raw-window cap
  behavior, and draft failure/fallback probes on B300 when an MTP support
  model is available; otherwise the fixture is a documented support-artifact
  blocker.
- Comparator: command-boundary trace, `mtp_n_raw` transition, draft top-id,
  logits role, previous/output HC tensor role, and fail-closed error
  comparison against the current-C oracle.
- Acceptance: Rust draft orchestration produces the same draft top-id/logits
  role and `mtp_n_raw` transition as current C, and any draft failure leaves
  target decode state eligible for the same fallback path.
- Drift policy: draft path name, command boundaries, tensor roles,
  `mtp_n_raw`, top-id, and fallback classification are exact; float logits use
  M10.4 tolerances when captured.
- Review gate: ask Claude to review draft graph state isolation from the
  target stream.
- Validation needed: B300 MTP draft smoke or explicit MTP-model blocker,
  comparator with negative tests, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Evidence: added `rust/ds4-gpu/src/mtp_draft_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-draft-plan.rs`, and
  `ds4-parity/compare_mtp_draft_plan.py`. The Rust draft plan pins first and
  recursive MTP draft HC roles, command steps, readbacks, raw-frontier
  transition, failure restoration, and the explicit missing-MTP live blocker.
  Validation passed targeted Rust draft-plan tests, the comparator with 5
  cases, 118 checks, and 6 negative mutations, JSON output parsing, Python
  syntax, the live B300 missing-MTP blocker command, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, unified parity with 61
  passed, 42 skipped, and 0 failed, and non-interactive Claude review with
  `NO BLOCKERS`.
- Owner path: Rust MTP draft orchestration, B300 comparator,
  `.memory/status.md`.

### M10.8d: Rust Exact N=2 Verifier Orchestration Smoke

- Status: done
- Goal: move the exact N=2 verifier command orchestration into Rust while
  preserving the target decode kernel order and row-logits contract.
- Oracle: `metal_graph_verify_decode2_exact`, M10.2 command boundaries, M10.4
  checkpoint policy, and the M10.8a exact-N=2 rows.
- Fixture: exact N=2 full accept, exact N=2 prefix1 accept, and exact N=2
  verifier failure/rollback probes on B300 when an MTP support model is
  available; otherwise the fixture is a documented support-artifact blocker.
- Comparator: command-boundary trace, top0/logits0/logits1 metadata, accepted
  sequence, checkpoint length, and frontier action comparison against the
  current-C oracle.
- Acceptance: Rust exact-N=2 orchestration produces the same accepted prefix
  and logits source as current C, and restores the pre-verifier frontier on
  failure.
- Drift policy: verifier path name, command boundaries, accepted counts,
  logits row role, checkpoint length, and restore behavior are exact; float
  logits use M10.4 tolerances when captured.
- Review gate: ask Claude to review exact target-stream preservation.
- Validation needed: B300 exact-N=2 smoke or explicit MTP-model blocker,
  comparator with negative tests, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Evidence: added `rust/ds4-gpu/src/mtp_decode2_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-decode2-plan.rs`, and
  `ds4-parity/compare_mtp_decode2_plan.py`. The Rust exact-N=2 verifier plan
  pins target token order, decode-layer command steps, prefix1 frontier capture,
  top0/logits0/logits1 readbacks, full-accept and prefix1 logits source,
  failure restore, and the explicit missing-MTP live blocker. Validation passed
  targeted Rust decode2-plan tests, the comparator with 4 cases, 148 checks,
  and 7 negative mutations, JSON output parsing, Python syntax, the live B300
  missing-MTP blocker command, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, unified parity with 62 passed, 42 skipped, and
  0 failed, and non-interactive Claude review with `NO BLOCKERS`.
- Owner path: Rust exact-N=2 orchestration, B300 comparator,
  `.memory/status.md`.

### M10.8e: Rust Suffix Verifier Orchestration Smoke

- Status: done
- Goal: move Rust-owned orchestration for the suffix/microbatch verifier path
  while preserving current-C full-accept, prefix1-accept, replay, and rollback
  decisions.
- Oracle: `metal_graph_verify_suffix_tops`, `metal_graph_read_spec_logits_row`,
  the microbatch branch in `ds4_session_decode_speculative`, and the M10.8a
  suffix-verifier rows.
- Fixture: microbatch full accept, microbatch prefix1 accept with captured
  prefix state, replay fallback, exact-replay debug fallback, and verifier
  failure rollback probes on B300 when an MTP support model is available;
  otherwise the fixture is a documented support-artifact blocker.
- Comparator: row top-id sequence, accepted prefix length, logits-row role,
  checkpoint length transition, snapshot requirement, prefix1/replay/restore
  action, and failure classification.
- Acceptance: Rust suffix verifier orchestration accepts the same prefix as
  current C, reads logits from the same row role, and restores or commits the
  same frontier action before returning to the target stream.
- Drift policy: verifier path, row-top order, accepted count, checkpoint
  length, logits-row role, and frontier action are exact; float logits use
  M10.4 tolerances when captured.
- Review gate: ask Claude to review suffix verifier rollback and prefix1
  safety.
- Validation needed: B300 suffix-verifier smoke or explicit MTP-model blocker,
  comparator with negative tests, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Evidence: added `rust/ds4-gpu/src/mtp_suffix_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-suffix-plan.rs`, and
  `ds4-parity/compare_mtp_suffix_plan.py`. The Rust suffix verifier plan pins
  full accept, prefix1 accept, restore/replay, exact replay debug, failure
  restore-or-error behavior, row-top/readback roles, and the explicit
  missing-MTP live blocker. Validation passed targeted Rust suffix-plan tests,
  the comparator with 6 cases, 179 checks, and 8 negative mutations, JSON output
  parsing, Python syntax, the live B300 missing-MTP blocker command, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, unified
  parity with 63 passed, 42 skipped, and 0 failed, and non-interactive Claude
  review with `NO BLOCKERS`.
- Owner path: Rust suffix-verifier orchestration, B300 comparator,
  `.memory/status.md`.

### M10.8f: Rust Spec Frontier Snapshot Restore And Prefix1 Commit

- Status: done
- Goal: move speculative frontier snapshot, restore, and prefix1 commit state
  mutation into Rust-owned orchestration.
- Oracle: `spec_frontier_snapshot`, `spec_frontier_restore`,
  `spec_frontier_commit_prefix1`, and the M10.7d3 restored-frontier/KVC
  evidence for counter semantics.
- Fixture: compressed-layer frontier counters, ratio-4 index counters,
  `mtp_n_raw`, prefix1 captured frontiers, full restore, and one-token commit
  probes.
- Comparator: counter/tensor-copy plan comparison and B300 state digest
  comparison for snapshot, restore, and prefix1 commit.
- Acceptance: Rust can restore exactly to the pre-verifier frontier after a
  miss and can commit exactly one verified token without exposing speculative
  row counters as live target state.
- Drift policy: frontier counters, layer sets, ratio-specific index handling,
  `mtp_n_raw`, and state digest hashes are exact; invisible append-only garbage
  rows remain allowed only where the C contract allows them.
- Review gate: ask Claude to review rollback and prefix1 commit safety.
- Validation needed: B300 frontier mutation smoke, comparator with negative
  tests, targeted Rust tests, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Owner path: Rust frontier mutation orchestration, B300 comparator,
  `.memory/status.md`.
- Evidence: added `rust/ds4-gpu/src/mtp_frontier_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-frontier-plan.rs`, and
  `ds4-parity/compare_mtp_frontier_plan.py`, a Rust model-free frontier
  mutation plan that pins snapshot, restore, prefix1 commit, ratio-4 index
  handling, `mtp_n_raw` save/restore, invisible speculative-row policy, and
  the B300 missing-MTP live blocker against current-C anchors and M10.7d3
  restored-frontier evidence. Validation passed targeted Rust frontier-plan
  tests, the comparator with 8 cases, 145 checks, and 8 negative mutations,
  JSON output parsing, Python syntax, the live B300 missing-MTP blocker
  command, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and unified parity with 64 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.

### M10.8g: Rust MTP End-To-End Stream Parity

- Status: done
- Goal: integrate Rust MTP draft/verifier/frontier orchestration into the
  runtime path and compare accepted tokens against current C.
- Oracle: current-C speculative decode on the same B300 model/support-model
  pair, plus M10.8a through M10.8f contracts.
- Fixture: MTP disabled, missing support model, first-draft miss, one-token
  accept, two-token accept, verifier failure, prefix1 commit, rollback, and
  sequential fallback cases.
- Comparator: accepted token sequence, visible target output stream, final
  checkpoint length, logits/top-id parity, frontier state digest, and cache/KVC
  accounting.
- Acceptance: Rust MTP never changes the non-speculative target stream,
  commits only verified prefixes, and restores graph state exactly on misses
  or verifier failures.
- Drift policy: accepted sequence, visible output, checkpoint/frontier state,
  and cache accounting are exact; probe logs and timings are normalized.
- Review gate: ask Claude to review end-to-end target-stream safety.
- Validation needed: B300 MTP comparator or explicit support-artifact blocker,
  server/runtime parity checks, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
- Evidence: split before implementation into M10.8g1 through M10.8g4 so the
  stream-level contract, Rust outcome planner, runtime no-drift guard, and live
  B300 support-model comparator can be validated independently. Validation
  passed the live B300 support-artifact blocker command, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and unified
  parity with 64 passed, 42 skipped, and 0 failed. Non-interactive Claude
  review returned `NO BLOCKERS`.
- Evidence: M10.8g closed through M10.8g4b with the explicit support-artifact
  blocker because no B300 MTP support GGUF is present. The final closure
  records `support_absent_blocker_closure`, support-present comparator
  `not_run` due to `support_artifact_absent`, `blocked_missing_mtp_model`,
  empty `mtp_candidates=`, next stage `M10.9`, and no MTP-off or MTP-enabled
  parity claim.
- Owner path: Rust runtime MTP integration, B300 comparator,
  `.memory/status.md`.

### M10.8g1: MTP Stream Parity Contract And Blocker

- Status: done
- Goal: capture the final current-C speculative stream contract and live B300
  support-model blocker before any Rust end-to-end MTP stream path is trusted.
- Oracle: `ds4_session_eval_speculative_argmax`, `ds4_session_eval_internal`,
  M10.8a through M10.8f contracts, and the B300 support-artifact search.
- Fixture: MTP disabled, missing support model, first-draft miss, margin-skip
  one-token accept, exact N=2 full accept, exact N=2 prefix1 accept, exact N=2
  verifier failure, suffix full accept, suffix prefix1 accept, suffix
  restore/replay accept, suffix verifier failure, and sequential fallback.
- Comparator: JSON contract plus checker that pins accepted-token deltas,
  checkpoint length deltas, logits source, frontier snapshot/restore/commit,
  `mtp_n_raw` keep policy, target-stream invariants, cache/KVC visibility, and
  explicit support-artifact blocker state.
- Acceptance: every later M10.8g stream case has a named current-C oracle row
  and exact final stream state; absent B300 MTP support remains an explicit
  blocker instead of silently becoming an MTP-off pass.
- Drift policy: accepted tokens, checkpoint length, logits source, frontier
  counters, `mtp_n_raw`, cache/KVC accounting, and blocker status are exact;
  probe logs and timings are normalized.
- Review gate: ask Claude to review contract coverage against current-C
  end-to-end target-stream safety.
- Validation needed: contract checker with negative tests, Python/JSON syntax,
  B300 support-artifact blocker command, `cargo fmt --all -- --check`, `git
  diff --check`, unified parity report, and non-interactive Claude review with
  no blockers.
- Owner path: M10.8g stream contract, B300 support-artifact check,
  `.memory/status.md`.
- Evidence: added
  `ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json` and
  `ds4-parity/check_mtp_stream_parity_contract.py`, a stream-level current-C
  contract checker for disabled/missing MTP, first-draft miss, exact N=2
  full/prefix/failure, suffix full/prefix/replay/failure, sequential fallback,
  frontier restore/commit, `mtp_n_raw` keep policy, visible cache/KVC state,
  and the B300 missing-MTP blocker. Validation passed JSON syntax, Python
  syntax, the checker with 12 cases, 368 checks, and 8 negative mutations, the
  live B300 missing-MTP blocker command, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, and unified parity with 65 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.

### M10.8g2: Rust MTP Stream Outcome Planner

- Status: done
- Goal: compose the Rust draft, verifier, suffix, and frontier plans into a
  Rust-owned stream outcome planner without executing GPU kernels.
- Oracle: M10.8g1 stream contract plus M10.8b through M10.8f Rust plan outputs.
- Fixture: the M10.8g1 stream rows, including disabled/missing-support rows,
  first-draft miss, prefix1 commit, rollback, replay, and sequential fallback.
- Comparator: Rust JSON plan compared exactly to the stream contract for
  accepted tokens, checkpoint mutation, logits source, frontier operation,
  `mtp_n_raw` keep policy, cache/KVC visibility, and fallback/error state.
- Acceptance: Rust can fail closed for unavailable MTP and predict the same
  final stream state as current C for every model-free MTP outcome row.
- Drift policy: row order, selected sub-plan IDs, stream mutations, frontier
  operations, and blocker names are exact; timings remain ignored.
- Review gate: ask Claude to review planner composition against the M10.8
  sub-plan contracts.
- Validation needed: comparator with negative tests, targeted Rust tests, JSON
  parsing, Python syntax, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
- Owner path: Rust MTP stream planner, comparator, `.memory/status.md`.
- Evidence: added `rust/ds4-gpu/src/mtp_stream_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-stream-plan.rs`, and
  `ds4-parity/compare_mtp_stream_plan.py`, a Rust model-free stream outcome
  planner that composes M10.8b through M10.8f subplans against the M10.8g1
  stream contract. Validation passed targeted Rust stream-plan tests, JSON
  output parsing, Python syntax, the comparator with 12 cases, 369 checks, and
  8 negative mutations, the live B300 missing-MTP blocker command, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and unified
  parity with 66 passed, 42 skipped, and 0 failed. Non-interactive Claude
  review returned `NO BLOCKERS`.

### M10.8g3: Rust Runtime Guard And Target-Stream No-Drift Smoke

- Status: done
- Goal: wire the Rust runtime surface through the MTP stream guard for disabled
  and missing-support cases while proving the non-speculative target stream is
  unchanged.
- Oracle: current-C one-token target decode, Rust runtime non-spec output, and
  the M10.8g2 unavailable-MTP stream outcomes.
- Fixture: MTP off, `--mtp` pointing at a missing support model, first-token
  eval with no valid draft, server/runtime request replay, and cache/KVC ledger
  probes.
- Comparator: accepted output text, accepted token sequence, final checkpoint
  length, logits/top-id source, runtime cache/KVC accounting, and explicit
  blocker/error/fallback state.
- Acceptance: unavailable or disabled MTP cannot alter the visible target
  stream, checkpoint, logits ownership, or cache/KVC accounting.
- Drift policy: stream bytes, token IDs, checkpoint length, cache/KVC counts,
  and blocker text are exact; request IDs, timing, and probe logs are
  normalized.
- Review gate: ask Claude to review no-drift runtime guard safety.
- Validation needed: server/runtime parity checks, targeted Rust tests, missing
  support smoke, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, unified parity report, and non-interactive Claude review with
  no blockers.
- Owner path: Rust runtime MTP guard, server/runtime comparators,
  `.memory/status.md`.
- Split: this stage is split into M10.8g3a through M10.8g3c before
  implementation so static guard wiring, no-drift runtime comparison, and live
  missing-support smoke stay independently reviewable.
- Evidence: split M10.8g3 into M10.8g3a Rust runtime MTP guard contract and
  static wiring, M10.8g3b runtime target-stream no-drift comparator, and
  M10.8g3c B300 missing-support runtime smoke. Validation passed the live B300
  missing-support artifact check with empty `mtp_candidates=`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and unified
  parity with 66 passed, 42 skipped, and 0 failed. Non-interactive Claude
  review returned `NO BLOCKERS`.

#### M10.8g3a: Rust Runtime MTP Guard Contract And Static Wiring

- Status: done
- Goal: define the Rust-owned runtime MTP guard contract for disabled,
  first-draft-miss, and missing-support paths without launching model-backed
  runtime generation.
- Oracle: M10.8g2 unavailable stream outcomes, Rust `EngineOptions` defaults,
  Rust CLI/server MTP flag parsing, and current-C runtime guard anchors.
- Fixture: MTP off, `--mtp` path configured with no support artifact,
  first-token eval with no valid draft, one-shot/interactive/server runtime
  option mapping, and the argmax/session runtime surfaces that intentionally
  remain non-MTP.
- Comparator: Rust JSON guard plan checked against M10.8g2 stream rows plus
  static source anchors for `EngineOptions`, `ds4-gguf` CLI parsing, runtime
  binaries, and current-C speculative dispatch guards.
- Acceptance: disabled and unavailable MTP paths are classified before any
  speculative stream mutation and expose only target-stream state or the
  explicit missing-support blocker.
- Drift policy: surface names, source anchors, selected M10.8g2 rows, stream
  deltas, checkpoint deltas, cache/KVC visibility, fallback/error names, and
  blocker text are exact.
- Review gate: ask Claude to review static runtime guard coverage and fail-closed
  semantics.
- Validation needed: comparator with negative tests, targeted Rust tests, JSON
  parsing, Python syntax, live B300 missing-support blocker check, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, unified
  parity report, and non-interactive Claude review with no blockers.
- Owner path: Rust runtime MTP guard plan, comparator, `.memory/status.md`.
- Evidence: added `rust/ds4-gpu/src/mtp_runtime_guard_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-runtime-guard-plan.rs`, and
  `ds4-parity/compare_mtp_runtime_guard.py`, a Rust model-free runtime guard
  plan that ties `EngineOptions`, `ds4-gguf` MTP CLI parsing,
  one-shot/interactive/server runtime mappings, argmax/session non-MTP
  surfaces, current-C speculative dispatch guards, and the B300 missing-support
  artifact check to the M10.8g2 disabled, first-draft-miss, and
  missing-support stream outcomes. Validation passed targeted Rust guard tests,
  JSON output parsing, Python syntax, the comparator with 7 cases, 292 checks,
  and 7 negative mutations, the live B300 missing-support artifact check with
  empty `mtp_candidates=`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and unified parity with 67 passed, 42 skipped,
  and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.

#### M10.8g3b: Runtime Target-Stream No-Drift Comparator

- Status: done
- Goal: compare real Rust runtime no-MTP output against current-C target-stream
  output for the disabled guard path.
- Oracle: current-C one-token target decode, existing Rust runtime non-spec
  output, and the M10.8g3a disabled-MTP guard row.
- Fixture: one-shot runtime, server runtime request replay, MTP off, short
  first-token generation, and cache/KVC ledger probes.
- Comparator: accepted text bytes, accepted token IDs, checkpoint length,
  logits/top-id ownership, cache/KVC accounting, and guard state.
- Acceptance: enabling the runtime guard with MTP unavailable or disabled cannot
  change target-stream bytes, token IDs, checkpoint length, logits ownership, or
  cache/KVC accounting.
- Drift policy: target-stream bytes, token IDs, checkpoint length, cache/KVC
  counts, and guard state are exact; request IDs, timing, and logs are
  normalized.
- Review gate: ask Claude to review target-stream no-drift runtime comparison.
- Evidence: added `ds4-parity/compare_mtp_runtime_no_drift.py` and wired it
  into `ds4-parity/run_parity_report.py` and `ds4-parity/README.md`. The
  comparator ties the M10.8g3a disabled runtime guard rows to the M8.12a
  current-C one-shot target-stream oracle and the M9.8f5 B300 Rust runtime
  replay summary, checking 3 CLI no-MTP target-stream cases, 3 server no-MTP
  replay cases, cache/KVC ledger probes, guard linkage, and static runtime
  report hooks.
- Evidence: validation passed Python syntax, `python3
  ds4-parity/compare_mtp_runtime_no_drift.py --negative-test` with 3 CLI
  cases, 3 server cases, 180 checks, and 6 negative mutations, live B300
  one-shot no-MTP runtime comparator with 144 checks and 5 negative checks,
  `python3 ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped,
  and 0 failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 68 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- Owner path: runtime no-drift comparator, server/runtime fixtures,
  `.memory/status.md`.

#### M10.8g3c: B300 Missing-Support Runtime Smoke

- Status: done
- Goal: run the missing-support runtime path on B300 and keep the support-model
  absence explicit without claiming MTP-enabled parity.
- Oracle: M10.8g3a missing-support guard row, B300 support-artifact search, and
  current runtime missing-MTP behavior.
- Fixture: B300 `ds4-rust-port-b300`, `/workspace/ds4/ds4flash.gguf`,
  absent `/workspace/ds4/missing-mtp.gguf`, and candidate search for
  `*mtp*.gguf` or `*draft*.gguf`.
- Comparator: missing-support blocker text, stream visibility, checkpoint
  mutation, cache/KVC visibility, and recorded candidate search output.
- Acceptance: the B300 runtime path fails closed before speculative stream
  mutation and records `mtp_candidates=` as empty until a support GGUF exists.
- Drift policy: blocker text and candidate search output are exact; pod names,
  request IDs, and timing are normalized.
- Review gate: ask Claude to review live missing-support smoke scope.
- Evidence: added `ds4-parity/compare_mtp_runtime_missing_support.py`,
  `ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json`,
  README instructions, unified report wiring, and an exact B300 rerun hook. The
  comparator ties the Rust B300 missing-MTP runtime result to the M10.8g3a
  missing-support guard row, the M10.8g1 stream blocker, and the M8.12b
  current-C missing-MTP runtime case.
- Evidence: live B300 smoke passed with 118 checks and 7 negative mutations,
  recording `/workspace/ds4/ds4flash.gguf` at 86,720,111,488 bytes, absent
  `/workspace/ds4/missing-mtp.gguf`, empty `mtp_candidates=`, exit code 1,
  empty stdout, stderr SHA
  `826268e476a14b68cf733c113b9a8517c9c3209988de7dbb3bbd98e7f64f444a`,
  `blocked_before_stream` visibility, checkpoint delta 0, and no cache/KVC
  visibility. Local validation passed Python syntax, the comparator with 118
  checks and 7 negative mutations, targeted Rust guard test
  `missing_support_blocks_before_stream_mutation`, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 69 passed, 43
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- Owner path: B300 missing-support runtime smoke, comparator,
  `.memory/status.md`.

### M10.8g4: B300 Support-Model End-To-End Comparator

- Status: done
- Goal: close M10.8g with a same-B300 current-C versus Rust speculative stream
  comparator when an MTP support GGUF is available, or keep the blocker
  explicitly recorded when it is not.
- Oracle: current-C speculative decode on the same target GGUF, support GGUF,
  backend, prompt, draft depth, margin, and environment gates.
- Fixture: first-draft miss, one-token accept, two-token accept, suffix full
  accept, suffix replay accept, verifier failure, prefix1 commit, rollback,
  sequential fallback, EOS, and cache/KVC continuation probes.
- Comparator: accepted token sequence, visible output stream, final checkpoint
  length, logits/top-id parity, frontier state digest, `mtp_n_raw`, cache/KVC
  accounting, and explicit blocker output when the support GGUF is absent.
- Acceptance: Rust MTP never changes the non-speculative target stream,
  commits only verified prefixes, restores graph state exactly on misses or
  verifier failures, and has a reproducible rerun command for support-artifact
  availability.
- Drift policy: stream output, accepted tokens, checkpoint/frontier state,
  logits/top-id, `mtp_n_raw`, cache/KVC accounting, model identity, and support
  artifact identity are exact; timings and probe logs are normalized.
- Review gate: ask Claude to review end-to-end comparator coverage and blocker
  semantics.
- Validation needed: B300 MTP comparator or explicit support-artifact blocker,
  server/runtime parity checks, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
- Owner path: B300 MTP stream comparator, support-artifact blocker,
  `.memory/status.md`.
- Split: M10.8g4 is split into M10.8g4a support-artifact branch decision and
  M10.8g4b final support comparator or explicit blocker closure so the
  currently missing support-model path does not get mixed with MTP-enabled
  parity claims.
- Evidence: split validation passed the live B300 support-artifact probe with
  `/workspace/ds4/ds4flash.gguf`, absent `/workspace/ds4/missing-mtp.gguf`, and
  empty `mtp_candidates=`, plus `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 69 passed, 43
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- Evidence: M10.8g4b closed this parent through the support-absent branch. The
  final closure artifact records the target model identity, absent
  `/workspace/ds4/missing-mtp.gguf`, empty support candidates, not-run
  support-present comparator, explicit `blocked_missing_mtp_model`, next stage
  `M10.9`, and the claim policy forbidding `MTP-off pass` and `MTP-enabled
  parity`.

#### M10.8g4a: B300 Support-Artifact Branch Decision

- Status: done
- Goal: decide whether M10.8g4 can run the same-B300 support-model comparator
  or must close with the explicit support-artifact blocker.
- Oracle: M10.8g1 B300 support-artifact search, M10.8g3c Rust runtime
  missing-support summary, and the B300 target-model identity.
- Fixture: B300 `ds4-rust-port-b300`, `/workspace/ds4/ds4flash.gguf`,
  `/workspace/ds4/missing-mtp.gguf`, and `*mtp*.gguf` or `*draft*.gguf`
  candidate search at max depth 3.
- Comparator: support candidate list, target model identity, missing-support
  runtime summary linkage, and selected M10.8g4 branch.
- Acceptance: if the support candidate list is empty, M10.8g4b must use the
  explicit blocker closure and must not claim MTP-enabled parity; if support
  appears, M10.8g4b must run a same-B300 current-C versus Rust comparator.
- Drift policy: candidate paths, model identity, branch selection, and blocker
  linkage are exact; command logs and timings are normalized.
- Review gate: ask Claude to review branch selection and overclaim prevention.
- Validation needed: live B300 support-artifact check, comparator with negative
  tests, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, unified parity report, and non-interactive Claude review with no
  blockers.
- Evidence: added `ds4-parity/compare_mtp_support_branch.py`,
  `ds4-parity/baselines/graph/m10.8g4a/support-branch-decision.json`, README
  instructions, unified report wiring, and an exact B300 rerun hook. The branch
  decision links the M10.8g1 stream blocker and M10.8g3c Rust runtime blocker
  to the current B300 support-artifact search.
- Evidence: live B300 branch capture passed with 48 checks and 6 negative
  mutations, recording `/workspace/ds4/ds4flash.gguf` at 86,720,111,488 bytes,
  absent `/workspace/ds4/missing-mtp.gguf`, empty `mtp_candidates=`, selected
  branch `support_absent_blocker_closure`, next stage `M10.8g4b`, and a claim
  policy forbidding `MTP-off pass` and `MTP-enabled parity`. Local validation
  passed Python syntax, the comparator with 48 checks and 6 negative mutations,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 70
  passed, 44 skipped, and 0 failed. Non-interactive Claude review returned
  `NO BLOCKERS`.
- Owner path: support-artifact branch selector, B300 blocker summaries,
  `.memory/status.md`.

#### M10.8g4b: B300 End-To-End Blocker Or Support Comparator Closure

- Status: done
- Goal: close M10.8g with either a same-B300 current-C versus Rust speculative
  stream comparator or the explicit support-artifact blocker selected by
  M10.8g4a.
- Oracle: current-C speculative stream behavior when a support GGUF exists,
  otherwise M10.8g1 stream blocker plus M10.8g3c Rust runtime blocker summary.
- Fixture: support-present cases for first-draft miss, one-token accept,
  two-token accept, suffix full/replay accept, verifier failure, prefix1
  commit, rollback, sequential fallback, EOS, and cache/KVC continuation; or
  the support-absent blocker fixture from M10.8g4a.
- Comparator: accepted stream, checkpoint/frontier state, logits/top-id,
  `mtp_n_raw`, cache/KVC accounting, model/support identity, and explicit
  blocker output when support is absent.
- Acceptance: support-present runs compare current C and Rust exactly; support
  absent records a final blocker without passing as MTP-off or MTP-enabled
  parity.
- Drift policy: stream state, model/support identity, and blocker output are
  exact; timings and probe logs are normalized.
- Review gate: ask Claude to review final closure semantics.
- Validation needed: B300 support comparator or blocker comparator with negative
  tests, server/runtime parity checks, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Owner path: M10.8g final comparator or blocker closure, `.memory/status.md`.
- Evidence: added `ds4-parity/compare_mtp_end_to_end_closure.py`,
  `ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json`, README
  instructions, unified report wiring, and an exact B300 rerun hook. The
  closure consumes the M10.8g4a support-branch decision, M10.8g1 stream
  blocker, and M10.8g3c Rust runtime blocker.
- Evidence: live B300 closure validation passed with 58 checks and 7 negative
  mutations after refreshing the M10.8g4a branch decision. The artifact records
  `support_absent_blocker_closure`, support-present comparator `not_run` due to
  `support_artifact_absent`, `/workspace/ds4/ds4flash.gguf` at 86,720,111,488
  bytes, absent `/workspace/ds4/missing-mtp.gguf`, empty `mtp_candidates=`,
  `blocked_before_stream` visibility, checkpoint delta 0, no cache/KVC
  visibility, `blocked_missing_mtp_model`, next stage `M10.9`, and no
  MTP-enabled parity claim. Local validation passed Python syntax, the
  comparator with 58 checks and 7 negative mutations, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 71 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.

### M10.9: Runtime Graph End-To-End And Benchmark Closure

- Status: split before implementation into M10.9a through M10.9f
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
- Split: M10.9 is split into M10.9a through M10.9f before implementation so
  runtime-route activation, official-vector quality, long-context quality,
  tool/server quality, benchmark comparison, and final closure remain
  independently comparable.
- Evidence: split M10.9 into M10.9a closure matrix, M10.9b route switch,
  M10.9c official-vector gate, M10.9d long-context gate, M10.9e tool/server
  gate, and M10.9f benchmark/final closure before implementation. Validation
  passed the B300 fixture-readiness probe for the resolved model path, model
  size 86,720,111,488 bytes, `official.vec` SHA
  `0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac`,
  benchmark prompt SHA
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`,
  existing M0.6 benchmark CSV fixtures, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 71 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.

#### M10.9a: Runtime Graph Closure Matrix And Rerun Contract

- Status: done
- Goal: define the exact M10.9 artifact matrix, runtime selectors, B300 rerun
  commands, and drift policy before enabling any Rust graph runtime path.
- Oracle: M0.3 official-vector baseline, M0.6 benchmark baseline, M9
  server/runtime replay, M10.2 through M10.8 graph and MTP contracts, and the
  live B300 model/prompt fixture inventory.
- Fixture: `/workspace/ds4/ds4flash.gguf`, `tests/test-vectors/official.vec`,
  long-context and tool-call-quality prompts, M9 server fixtures, M0.6
  `ds4-bench` CSVs, and the exact Rust runtime binaries/options to be compared.
- Comparator: closure-matrix checker that verifies every M10.9b through
  M10.9f gate has a named oracle, command, artifact path, comparator, rerun
  command, acceptance rule, and drift policy.
- Acceptance: no later M10.9 item can claim runtime graph parity without a
  concrete artifact and comparator; model-backed B300 checks have exact
  temp-kubeconfig reruns and local comparator fallbacks where possible.
- Drift policy: stage IDs, command lines, fixture paths, model identity,
  artifact locations, and claim boundaries are exact; timings remain
  non-semantic until benchmark comparison.
- Review gate: ask Claude to review the closure matrix for missing gates and
  overbroad acceptance.
- Validation needed: matrix checker with negative tests, B300 fixture-readiness
  probe, server/runtime parity report, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Owner path: M10.9 closure matrix, B300 rerun contract, `.memory/status.md`.
- Evidence: added
  `ds4-parity/baselines/graph/m10.9a/runtime-graph-closure-matrix.json`,
  `ds4-parity/check_runtime_graph_closure_matrix.py`, README instructions,
  unified report wiring, and an exact B300 fixture-readiness rerun hook. The
  matrix pins M10.9b through M10.9f to concrete oracles, fixture paths,
  artifact paths, rerun commands, comparators, acceptance rules, drift
  policies, and claim boundaries.
- Evidence: validation passed Python syntax, `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test` with 118
  checks and 8 negative mutations, the live B300 fixture-readiness probe with
  resolved model size 86,720,111,488 bytes, `official.vec` SHA
  `0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac`,
  benchmark prompt SHA
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`,
  existing M0.6 benchmark CSV fixtures, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 72 passed, 46 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.

#### M10.9b: Rust Runtime Graph Route Switch And Preflight

- Status: complete
- Goal: add the Rust runtime graph route behind an explicit selector and prove
  unsupported configurations fail closed before changing visible generation.
- Oracle: current Rust runtime target-stream path, M10.5 through M10.8 graph
  scheduling contracts, and current C graph runtime option behavior.
- Fixture: one-shot, interactive, and server runtime options; CUDA and non-CUDA
  backend selectors; disabled graph route; missing model; unsupported graph
  route; and cache/KVC state preflight cases.
- Comparator: Rust route-preflight summary checked against source anchors and
  runtime no-drift cases for selected output, checkpoint, cache/KVC accounting,
  and explicit fail-closed errors.
- Acceptance: graph route selection is explicit, default behavior is unchanged,
  and unsupported graph-runtime cases fail before stream or cache mutation.
- Drift policy: option names, fail-closed categories, output bytes, checkpoint
  deltas, and cache/KVC counters are exact; logs and timings are normalized.
- Review gate: ask Claude to review route selection and fail-closed behavior.
- Evidence: added shared Rust `RuntimeGraphRoute` selector support and wired
  `--runtime-graph`/`--runtime-graph-route` through one-shot, interactive, and
  server runtime binaries. Added
  `ds4-parity/check_runtime_graph_route_preflight.py` and
  `ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json` for
  exact target-stream, disabled-route, invalid-selector, CUDA/non-CUDA
  unsupported graph-route, missing-model, and server KVC preflight outcomes.
  Non-server unsupported graph selection exits 99 before model open, stream
  output, or checkpoint/cache mutation. Server CUDA graph selection now reaches
  the missing-model path without stream output or server KVC directory
  creation; target-stream and `off` keep the existing missing-model behavior.
- Validation: `python3 ds4-parity/check_runtime_graph_route_preflight.py
  --negative-test` passed with 274 checks and 8 negative mutations, `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test` remained
  green with 118 checks and 8 negative mutations after status advanced to
  M10.9c, targeted Rust route/parser/server tests passed, `python3
  ds4-parity/run_server_parity_report.py` passed with 10 passed, 3 skipped,
  and 0 failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` passed with 73 passed, 46 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- Owner path: Rust runtime graph route selector, route-preflight comparator,
  `.memory/status.md`.

#### M10.9c: B300 Official-Vector Rust Runtime Gate

- Status: complete
- Goal: run official-vector logprob cases through the Rust graph runtime path on
  B300 and compare against the current-C M0.3 baseline.
- Oracle: current-C `./ds4_test --logprob-vectors` B300 baseline, M6 numeric
  tolerance policy, and the Rust runtime graph route selected in M10.9b.
- Fixture: `/workspace/ds4/ds4flash.gguf`,
  `tests/test-vectors/official.vec`, CUDA backend, deterministic generation
  settings, and captured Rust runtime vector output.
- Comparator: selected-token exact comparison plus numeric logit/logprob
  tolerance comparison for official-vector rows, with raw Rust logs retained.
- Acceptance: Rust runtime selected greedy tokens match the current-C baseline;
  numeric scores stay within the existing M6 tolerance; skipped current-C rows
  remain explicit.
- Drift policy: selected token IDs, fixture hash, model hash, backend, and skip
  reasons are exact; score tolerances follow the M6 policy.
- Review gate: ask Claude to review official-vector comparison scope and skip
  handling.
- Evidence: added `ds4-runtime-official-vectors-rs`, exposed Rust session
  argmax/top-logprob/eval APIs, added
  `ds4-parity/run_runtime_graph_official_vectors.py`, and captured
  `ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json` from the
  B300 pod. The artifact records route `graph`, backend `cuda`, q2-imatrix
  model hash, `official.vec` hash, raw Rust stdout/stderr, selected-token
  matches, top-logprob rows, official-top deltas, and the current-C
  `long_memory_archive` skip reason.
- Validation: live B300 Rust runtime official-vector capture passed with 1,958
  checks, max official-logprob delta 0.678254604, and 8 negative mutations.
  Local comparator, closure matrix, route preflight, Rust binary tests,
  `cargo test --workspace`, server parity report, formatter check, diff check,
  and unified parity report passed; non-interactive Claude review returned `NO
  BLOCKERS`. Exact command evidence is recorded in `.memory/status.md`.
- Owner path: B300 official-vector Rust runtime artifact, comparator,
  `.memory/status.md`.

#### M10.9d: B300 Long-Context Rust Runtime Gate

- Status: complete
- Goal: run the long-context quality gate through the Rust graph runtime path
  and keep nondeterministic score surfaces out of exact comparisons.
- Oracle: current-C `./ds4_test --long-context` baseline behavior, M7/M10
  long-context graph checkpoint contracts, and the M10.9c official-vector
  runtime evidence.
- Fixture: long-context fact-recall prompt, B300 target model, CUDA backend,
  graph route selector, and retained Rust stdout/stderr/log artifacts.
- Comparator: pass/fail classification, selected behavioral markers, context
  length, cache/KVC accounting, and graph-route evidence with raw logs retained
  for score drift investigation.
- Acceptance: Rust graph runtime completes the long-context gate without
  falling back to the C host route; any known nondeterministic score surface is
  explicitly classified instead of compared byte-for-byte.
- Drift policy: command line, model/backend identity, context length, pass/fail
  markers, fallback state, and cache/KVC accounting are exact; floating score
  surfaces use documented tolerance or explicit nondeterminism labels.
- Review gate: ask Claude to review long-context gate semantics.
- Evidence: added `ds4-runtime-long-context-rs`, added
  `ds4-parity/run_runtime_graph_long_context.py`, wired the comparator into the
  unified report and README, and captured
  `ds4-parity/baselines/graph/m10.9d/runtime-long-context.json` from the B300
  pod. The artifact records route `graph`, backend `cuda`, q2-imatrix model
  hash, prompt hash, current-C long-context stdout/stderr, raw Rust
  stdout/stderr, 30,474 prompt tokens, 76 completion tokens, `stop`, exact
  fact-recall output, and cache/KVC write accounting equal to the prompt token
  count.
- Validation: live B300 Rust runtime long-context capture passed with 126
  checks and 8 negative mutations. Local comparator, Rust binary check/tests,
  Python syntax checks, `cargo test --workspace`, server parity report,
  formatter check, diff check, unified parity report, and non-interactive
  Claude review passed. Exact command evidence is recorded in
  `.memory/status.md`.
- Owner path: B300 long-context Rust runtime artifact, comparator,
  `.memory/status.md`.

#### M10.9e: Tool-Call Quality And Server Replay Rust Runtime Gate

- Status: complete
- Goal: prove Rust graph runtime preserves tool-call quality and server/cache
  request behavior on B300.
- Oracle: current-C `./ds4_test --tool-call-quality`, M9 server/runtime replay,
  and the existing `run_tool_call_quality.py` classifier.
- Fixture: B300 Rust server runtime binary, OpenAI tool-call request fixture,
  M9 server/cache request fixtures, trace output, cache/KVC directories, and
  retained raw responses.
- Comparator: tool-call classifier summary, HTTP response/status comparison,
  trace/cache ledger checks, and no-fallback graph-route evidence.
- Acceptance: tool-call classification passes through the Rust graph runtime
  path, server/runtime parity remains green, and cache/KVC behavior matches the
  M9 replay contracts.
- Drift policy: HTTP status, response schema, tool name/arguments, trace/cache
  ledger markers, and fallback state are exact; request IDs and timings are
  normalized.
- Review gate: ask Claude to review tool/server runtime gate coverage.
- Evidence: extended `ds4-parity/run_tool_call_quality.py` into a
  self-contained Rust graph tool/server artifact comparator, wired it into the
  unified report and README, and captured
  `ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json` from the B300
  pod. The artifact records route `graph`, backend `cuda`, q2-imatrix model
  hash, current-C `./ds4_test --tool-call-quality` stdout/stderr, raw Rust
  request/response/trace/log blobs for fast and exact quality cases, HTTP 200,
  finish `tool_calls`, tool `list_files`, arguments `{"path":"."}`, and
  trace/cache ledger markers.
- Validation: live B300 Rust server runtime tool-call capture passed with 167
  checks and 8 negative mutations. Local comparator, route preflight, closure
  matrix, Rust server-runtime tests, Python syntax checks, workspace tests,
  server parity report, formatter check, diff check, unified parity report, and
  non-interactive Claude review passed. Exact command evidence is recorded in
  `.memory/status.md`.
- Owner path: tool-call quality runner, server/runtime replay artifacts,
  `.memory/status.md`.

#### M10.9f: Benchmark Comparator And Milestone 10 Closure

- Status: complete
- Goal: compare Rust graph runtime benchmark CSVs against the M0.6 same-B300
  current-C benchmark baseline and close Milestone 10.
- Oracle: M0.6 `ds4-bench` short/long CSV baseline, M10.9c through M10.9e
  quality gates, and same B300 model/backend identity.
- Fixture: short and long `speed-bench/promessi_sposi.txt` sweeps,
  `/workspace/ds4/ds4flash.gguf`, CUDA backend, Rust graph runtime benchmark
  CSVs, and capture metadata.
- Comparator: existing benchmark CSV comparator extended or configured for Rust
  candidate CSVs, plus final closure checker that verifies all M10.9 gates are
  current.
- Acceptance: benchmark workload shape matches exactly, throughput regression
  beyond the agreed threshold is documented, all M10.9 quality gates are green,
  and Milestone 10 closure does not claim unsupported backend replacement.
- Drift policy: CSV schema, prompt hash, model hash, backend, context
  frontiers, generation-token counts, `kvcache_bytes`, and gate statuses are
  exact; throughput uses the M0.6 regression threshold.
- Review gate: ask Claude to review benchmark comparability and final Milestone
  10 closure claims.
- Evidence: added `ds4-runtime-graph-bench-rs`, exposed Rust session snapshot
  and EOS-excluding argmax helpers needed to mirror `ds4-bench`, added
  `ds4-parity/run_runtime_graph_bench.py`, wired the comparator into the
  unified report and README, and captured
  `ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json` from the
  B300 pod. The artifact records route `graph`, backend `cuda`, q2-imatrix
  model hash, prompt hash, short/long benchmark CSV rows, exact context
  frontiers, prefill intervals, generation-token counts, KVC snapshot bytes,
  and M10.9a through M10.9e gate status.
- Validation: live B300 Rust benchmark closure passed with 349 checks and 8
  negative mutations. The artifact documents 7 older M0.6 decode throughput
  threshold misses, reproduces the same drift with same-session current-C
  `ds4-bench`, and verifies Rust stays within the same-session current-C
  threshold. Local comparator, workspace tests, server parity report,
  formatter check, diff check, unified parity report, and non-interactive
  Claude review passed. Exact command evidence is recorded in
  `.memory/status.md`.
- Owner path: Rust benchmark artifacts, final M10 closure checker,
  `.memory/status.md`.

#### M11: Agent Trace Replay

- Status: split before implementation into M11.1 through M11.4
- Goal: port the integrated coding agent only after runtime and server parity
  are stable.
- Oracle: current `ds4-agent` traces and deterministic replay fixtures.
- Fixture: scripted agent session fixtures plus deterministic tool-output
  replay or tool execution stubs.
- Comparator: agent trace replay with normalized timestamps, paths, and command
  duration fields.
- Acceptance: tool-call sequence, rendered context, session switching, and
  final visible outputs match fixture expectations; live manual sessions remain
  a final smoke test, not the primary comparator.
- Split:
  - M11.1 Agent Trace Replay Oracle And Fixture Contract.
  - M11.2 Rust Agent Rendered Context Replay.
  - M11.3 Deterministic Tool Stub And Session Command Replay.
  - M11.4 Rust Agent Loop And Manual Smoke.
- Validation: each substage must add or extend replay fixtures before claiming
  live Rust agent-loop behavior.

#### M11.1: Agent Trace Replay Oracle And Fixture Contract

- Status: complete.
- Goal: establish the no-model current-C replay fixture before porting the live
  Rust agent loop.
- Oracle: `./ds4-agent --dump-agent-trace-oracle`.
- Fixture: `ds4-parity/baselines/agent/m11.1/current-c.json` plus
  `manifest.json`.
- Comparator: `ds4-parity/compare_agent_trace_replay.py --negative-test`.
- Acceptance: Rust `ds4-agent-trace-replay-rs` emits the same normalized replay
  fixture; parsed DSML tool sequence, deterministic tool stubs, transcript
  roles, session save/list/switch/history/new operations, final visible output,
  and manifest hash checks pass.
- Evidence:
  - Added current-C `--dump-agent-trace-oracle` no-model dump path.
  - Added Rust `ds4-agent-trace-replay-rs` emitter.
  - Added unified parity report item `M11.1 Agent trace replay oracle`.
- Validation passed:
  - `arch -arm64 make ds4-agent`
  - `./ds4-agent --dump-agent-trace-oracle ds4-parity/baselines/agent/m11.1/current-c.json`
  - `python3 ds4-parity/compare_agent_trace_replay.py --negative-test`
    (`225 checks`)

#### M11.2: Rust Agent Rendered Context Replay

- Status: complete.
- Goal: replay the scripted M11.1 events into Rust prompt/context rendering
  before adding deterministic tool execution or the live agent loop.
- Oracle: M11.1 current-C replay fixture and existing prompt/DSML contracts.
- Fixture: M11.1 `single_tool_round` and `session_switching_commands` events.
- Comparator: `ds4-parity/compare_agent_rendered_context.py --negative-test`.
- Acceptance: system/user/assistant/tool/assistant boundaries, assistant EOS
  insertion, final visible output, and no-live-model behavior match fixture
  expectations.
- Evidence:
  - Added Rust `ds4-agent-rendered-context-rs` artifact emitter.
  - Added `ds4-parity/baselines/agent/m11.2/rendered-context.json` and
    `manifest.json`.
  - Added unified parity report item `M11.2 Agent rendered-context replay`.
- Validation passed:
  - `cargo run --quiet -p ds4-gguf --bin ds4-agent-rendered-context-rs >
    ds4-parity/baselines/agent/m11.2/rendered-context.json`
  - `python3 ds4-parity/compare_agent_rendered_context.py --negative-test`
    (`178 checks`)

#### M11.3: Deterministic Tool Stub And Session Command Replay

- Status: complete.
- Goal: replay deterministic tool stubs and session commands from the M11.1
  fixture without live model sampling.
- Oracle: M11.1 current-C replay fixture and M11.2 rendered-context artifact.
- Fixture: `single_tool_round` tool stub plus `session_switching_commands`.
- Comparator: `ds4-parity/compare_agent_deterministic_replay.py
  --negative-test`.
- Acceptance: tool output insertion, session operation order, normalized
  session ids, history text, and final visible command output match fixture
  expectations.
- Evidence:
  - Added Rust `ds4-agent-deterministic-replay-rs` artifact emitter.
  - Added `ds4-parity/baselines/agent/m11.3/deterministic-replay.json` and
    `manifest.json`.
  - Added unified parity report item `M11.3 Agent deterministic tool/session
    replay`.
- Validation passed:
  - `cargo run --quiet -p ds4-gguf --bin ds4-agent-deterministic-replay-rs >
    ds4-parity/baselines/agent/m11.3/deterministic-replay.json`
  - `python3 ds4-parity/compare_agent_deterministic_replay.py
    --negative-test` (`230 checks`)

#### M11.4: Rust Agent Loop And Manual Smoke

- Status: complete.
- Goal: wire the replay-proven Rust prompt, tool, and session paths into a live
  Rust agent loop while keeping manual sessions as smoke validation.
- Oracle: M11.1 through M11.3 replay fixtures plus current-C no-model command
  behavior.
- Fixture: scripted no-model loop smoke covering one tool round and
  save/list/switch/history/new commands.
- Comparator: Rust no-model agent-loop smoke with normalized visible output and
  transcript/session state checks.
- Acceptance: the Rust loop consumes scripted model events, invokes
  deterministic tool replay, applies session commands, and records visible
  output without model sampling before any manual smoke claim.
- Evidence:
  - Added Rust `ds4-agent-loop-smoke-rs` no-model smoke emitter.
  - Added `ds4-parity/baselines/agent/m11.4/loop-smoke.json` and
    `manifest.json`.
  - Added unified parity report item `M11.4 Agent no-model loop smoke`.
- Validation passed:
  - `cargo run --quiet -p ds4-gguf --bin ds4-agent-loop-smoke-rs >
    ds4-parity/baselines/agent/m11.4/loop-smoke.json`
  - `python3 ds4-parity/compare_agent_loop_smoke.py --negative-test`
    (`223 checks`)

#### M12: Backend Replacement Parity Split Planning

- Status: complete.
- Goal: split backend replacement into replay-comparable milestones before any
  CUDA/Metal backend ownership changes.
- Oracle: current backend implementation plus M10 runtime graph and benchmark
  fixtures.
- Fixture: operation-level tensor fixtures, official-vector fixtures,
  long-context fixtures, and `ds4-bench` CSVs.
- Comparator: per-stage tensor/runtime/benchmark comparators with explicit
  current-C or B300 oracles.
- Acceptance: each M12 substage has an oracle, fixture, comparator,
  acceptance criteria, and drift policy before implementation begins.
- Evidence:
  - Split M12 into M12.1 through M12.6 with explicit oracle, fixture,
    comparator, acceptance, and drift-policy criteria.

#### M12.1: Backend Boundary Inventory And Claim Matrix

- Status: complete.
- Goal: inventory current backend operation families and claim boundaries before
  any replacement implementation.
- Oracle: current backend headers/build wiring, Rust FFI wrappers, M10.5c4c1,
  M10.9 runtime closure, and B300 benchmark artifacts.
- Fixture: backend-boundary inventory JSON.
- Comparator: inventory checker for ownership states, fixture sources, rerun
  commands, and no-removal/no-replacement overclaims.
- Acceptance: every backend operation family has owner state, fixture source,
  comparator path, and drift policy.
- Drift policy: refresh inventory whenever C/CUDA/Metal signatures, Rust FFI
  wrappers, or route selectors change.
- Evidence:
  - Added `ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json`.
  - Added `ds4-parity/check_backend_boundary_inventory.py --negative-test`.
  - Added unified parity report item `M12.1 Backend boundary inventory`.
- Validation passed:
  - `python3 -m py_compile ds4-parity/check_backend_boundary_inventory.py
    ds4-parity/check_runtime_graph_closure_matrix.py
    ds4-parity/run_parity_report.py`
  - `python3 -m json.tool
    ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json`
  - `python3 ds4-parity/check_backend_boundary_inventory.py --negative-test`
  - `python3 ds4-parity/check_runtime_graph_closure_matrix.py
    --negative-test`
  - `cargo fmt --all -- --check`
  - `git diff --check`
  - `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 82
    passed, 50 skipped, 0 failed.

#### M12.2: Operation Tensor Fixture Capture

- Status: complete.
- Goal: capture operation-level tensor inputs/outputs for the first backend
  replacement families without changing runtime routing.
- Oracle: current backend outputs on the same backend and model class used by
  the target replacement slice.
- Fixture: tensor fixture bundle with operation name, shape, dtype, backend
  marker, model hash, prompt/vector hash, output hash, and numeric tolerance.
- Comparator: tensor fixture checker with exact SHA checks for byte-identical
  buffers and documented tolerances for f32/f16 output comparisons.
- Acceptance: each selected operation has current-backend fixture coverage,
  negative tests for shape/dtype/hash drift, and a rerun command for the same
  hardware class.
- Drift policy: fixture drift is accepted only with a refreshed current-backend
  oracle, recorded model/backend identity, and comparator tolerance rationale.
- Evidence:
  - Captured live B300 current-C oracle and Rust facade candidate JSON under
    `ds4-parity/baselines/backend/m12.2/captures/`.
  - Added `ds4-parity/baselines/backend/m12.2/manifest.json`.
  - Added `ds4-parity/check_backend_operation_fixtures.py --negative-test`.
  - Added unified parity report item `M12.2 Backend operation tensor fixtures`.
- Validation passed:
  - Live B300 first-kernel comparator with 103 checks.
  - Live B300 layer-0 QKV/RoPE comparator with 426 checks.
  - Live B300 layer-0 attention-output comparator with 493 checks.
  - Live B300 layer-0 FFN-output comparator with 885 checks.
  - Live B300 full output-head comparator with 440 checks.
  - `python3 ds4-parity/check_backend_operation_fixtures.py --negative-test`
    with 576 checks.
  - `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 83
    passed, 50 skipped, 0 failed.

#### M12.3: Rust Backend Facade Parity Harness

- Status: complete.
- Goal: route selected backend operations through a Rust-owned facade while
  still allowing the current backend implementation to serve as the oracle.
- Oracle: M12.2 tensor fixtures plus existing M10.5 backend ABI/facade
  comparators.
- Fixture: Rust facade replay artifact showing operation order, tensor binding,
  synchronization points, and error propagation for the selected families.
- Comparator: facade replay comparator against the M12.2 tensor fixture bundle
  and current ABI/facade contracts.
- Acceptance: Rust facade calls bind the same tensors in the same order,
  preserve current error categories, and produce fixture-matching outputs for
  the selected operations.
- Drift policy: any facade signature or operation-order change must update the
  replay artifact and keep old/current backend comparison available until the
  route gate passes.
- Evidence:
  - Added `ds4-parity/baselines/backend/m12.3/facade-replay.json`.
  - Added `ds4-parity/check_backend_facade_replay.py --negative-test`.
  - Added unified parity report item `M12.3 Backend facade replay harness`.
  - Preserved no backend replacement and no runtime route change claims.
- Validation passed:
  - `python3 ds4-parity/check_backend_facade_replay.py --negative-test`
    with 769 checks.
  - `python3 ds4-parity/check_backend_operation_fixtures.py --negative-test`
    with 576 checks after accepting M12.4 as the next active item.
  - `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 84
    passed, 50 skipped, 0 failed.

#### M12.4: First Backend Replacement Slice

- Status: complete.
- Goal: replace one bounded backend operation family in Rust or a Rust-owned
  backend module while leaving broader runtime routing unchanged.
- Oracle: M12.2 current-backend tensor fixtures and M12.3 facade replay.
- Fixture: replacement-slice summary with selected operation family, supported
  backend/platform, unsupported paths, tensor output comparisons, and explicit
  non-goals.
- Comparator: replacement-slice comparator that checks operation outputs,
  unsupported-path failures, and the claim boundary.
- Acceptance: the selected operation family matches fixture outputs within
  tolerance, fails closed for unsupported backends, and does not claim general
  backend replacement.
- Drift policy: replacement output drift requires a same-hardware rerun of the
  current-backend oracle and a comparator update that explains the tolerance or
  expected numeric change.
- Evidence:
  - Added Rust-owned slice descriptor
    `rust/ds4-gpu/src/replacement_slice.rs`.
  - Added descriptor emitter
    `rust/ds4-gpu/src/bin/ds4-backend-replacement-slice.rs`.
  - Added `ds4-parity/baselines/backend/m12.4/replacement-slice.json`.
  - Added `ds4-parity/check_backend_replacement_slice.py --negative-test`.
  - Selected the bounded `embedding_and_indexer` /
    `ds4_gpu_embed_token_hc_tensor` slice against the M12.2
    `first_kernel_embed_token_hc` fixture and M12.3 facade replay.
  - CPU, Metal, and default-route backend selectors fail closed before any
    runtime route change.
  - `python3 ds4-parity/check_backend_replacement_slice.py --negative-test`
    passed with 85 checks.
  - `cargo test -p ds4-gpu replacement_slice`, `cargo fmt --all -- --check`,
    `git diff --check`, and `python3 ds4-parity/run_parity_report.py
    --skip-local-oracles` passed with 85 passed, 50 skipped, 0 failed.

#### M12.5: Runtime Backend Route Gate

- Status: complete.
- Goal: expose the replacement slice through an explicit runtime route and
  validate end-to-end behavior without replacing the default backend.
- Oracle: current default route plus M10.9 official-vector, long-context,
  tool/server, and benchmark gates.
- Fixture: runtime route artifact recording route selector, backend identity,
  official-vector results, long-context results, quality gates, and benchmark
  deltas.
- Comparator: runtime route comparator that checks output parity, quality-gate
  parity, benchmark deltas, and route/preflight behavior.
- Acceptance: replacement route passes official-vector and long-context gates,
  preserves tool/server quality, records benchmark comparison on the same
  machine class, and remains opt-in.
- Drift policy: route behavior drift requires preserving current default-route
  evidence and rerunning the M10.9 closure gates.
- Evidence:
  - Added Rust-owned runtime backend route gate descriptor
    `rust/ds4-gpu/src/backend_route_gate.rs`.
  - Added descriptor emitter
    `rust/ds4-gpu/src/bin/ds4-backend-route-gate.rs`.
  - Added `ds4-parity/baselines/backend/m12.5/runtime-route-gate.json`.
  - Added `ds4-parity/check_backend_runtime_route_gate.py --negative-test`.
  - The explicit opt-in route is `replacement-slice` through
    `--runtime-backend-route`; the default route remains `current-backend` and
    does not activate the replacement slice.
  - The gate selects the M12.4 `embedding_and_indexer` /
    `ds4_gpu_embed_token_hc_tensor` slice for `cuda-b300`, rejects CPU/Metal
    and runtime-default-route selectors, and keeps general backend replacement
    plus kernel replacement claims false.
  - The checker ties route validation to the existing M10.9 B300 graph-route
    official-vector, long-context, tool/server, and same-session benchmark
    artifacts.
  - `python3 ds4-parity/check_backend_runtime_route_gate.py --negative-test`
    passed with 135 checks.
  - `cargo test -p ds4-gpu backend_route_gate`, `cargo fmt --all -- --check`,
    `git diff --check`, and `python3 ds4-parity/run_parity_report.py
    --skip-local-oracles` passed with 86 passed, 50 skipped, 0 failed.

#### M12.6: Backend Replacement Closure And Removal Decision

- Status: complete.
- Goal: decide whether any C/CUDA/Metal backend code can be removed, retained
  as a sidecar, or kept as an oracle after replacement routes pass.
- Oracle: M12.1 through M12.5 artifacts plus current removal criteria.
- Fixture: closure matrix listing backend families, replacement status,
  remaining sidecars, oracle coverage, runtime route status, and removal
  decision.
- Comparator: closure checker that rejects removal when any operation family,
  runtime route, benchmark, or platform-specific regression gate is missing.
- Acceptance: every replacement claim is backed by operation fixtures,
  end-to-end route gates, benchmark evidence, and explicit retained-sidecar
  decisions.
- Drift policy: closure drift must keep prior oracle artifacts available until
  the replacement route has equivalent or better coverage on each supported
  backend class.
- Evidence:
  - Added `ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json`.
  - Added `ds4-parity/check_backend_replacement_closure.py --negative-test`.
  - The closure matrix records all M12.1 backend operation families, the M12.2
    tensor fixture families, M12.3 facade replay coverage, the single M12.4
    route-gated replacement operation, and M12.5 opt-in route status.
  - Removal decision: retain current C/CUDA/Metal backend code as both sidecar
    and oracle. No removals are allowed in M12 because only
    `ds4_gpu_embed_token_hc_tensor` has opt-in route-gated replacement
    coverage, the default route remains current-backend, and current backend
    artifacts are still active references.
  - `python3 ds4-parity/check_backend_replacement_closure.py --negative-test`
    passed with 147 checks.
  - `cargo fmt --all -- --check`, `git diff --check`, and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` passed with 87
    passed, 50 skipped, 0 failed.

## Later Items

### M13: Backend Replacement Expansion

- Status: split before implementation into M13.0 through M13.5.
- Goal: broaden the only existing route-gated backend family before considering
  any unrelated backend-family replacement or removal.
- Oracle: M12.6 closure matrix, M12.1 backend inventory, M10.2 graph
  operation inventory, and existing M10.5/M10.6 current-C execution
  comparators.
- Fixture: M13 decision and per-operation embedding/indexer expansion
  artifacts.
- Comparator: M13 decision, fixture-matrix, replacement-slice, route-gate, and
  closure checkers.
- Acceptance: every remaining embedding/indexer operation is assigned to a
  covered fixture or explicit gap stage; default route stays current-backend;
  removals stay blocked.
- Drift policy: refresh the M13 matrix whenever embedding/indexer operation
  signatures, fixture coverage, or route selectors change.

#### M13.0: Backend Expansion Decision

- Status: complete.
- Goal: choose the post-M12 backend replacement direction and split it into
  reviewable, oracle-comparable M13 stages.
- Oracle: M12.6 closure matrix, M12.1 operation inventory, M10.2 graph
  operation inventory, and existing prefill/indexed-attention comparator
  coverage.
- Fixture:
  `ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json`.
- Comparator: `ds4-parity/check_backend_expansion_decision.py
  --negative-test`.
- Acceptance: M13 chooses to broaden the existing `embedding_and_indexer`
  route, maps all six remaining M12.6 operations to M13 work, and keeps
  removals/default-route replacement/general backend replacement/kernel
  replacement claims false.
- Drift policy: if M12.6 closure or existing comparator coverage changes,
  refresh this decision before starting M13.1.
- Validation passed:
  - `python3 ds4-parity/check_backend_expansion_decision.py --negative-test`
    with 186 checks.
  - Python syntax, JSON formatting, the M12.6 closure checker, `cargo fmt
    --all -- --check`, `git diff --check`, and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` with 88 passed, 50
    skipped, 0 failed.

#### M13.1: Embedding/Indexer Expansion Fixture Matrix

- Status: complete.
- Goal: create the operation-by-operation fixture matrix for all six remaining
  `embedding_and_indexer` operations.
- Oracle: M13.0 decision, M12.6 remaining-operation list, M12.1 inventory,
  M10.2 graph inventory, and existing prefill/indexed-attention comparators.
- Fixture:
  `ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json`.
- Comparator: `ds4-parity/check_backend_expansion_matrix.py --negative-test`.
- Acceptance: covered operations reference executable comparators; gap
  operations remain blocked; no route/default-route/removal claim changes.
- Drift policy: operation-list drift requires refreshing current-backend
  inventory and preserving the M13.0 decision artifact.
- Evidence:
  - Added
    `ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json`.
  - Added `ds4-parity/check_backend_expansion_matrix.py --negative-test`.
  - Classified the six remaining operations into pair-comparator-ready and
    fixture-gap rows without changing route/default-route/removal claims.
- Validation passed:
  - `python3 ds4-parity/check_backend_expansion_matrix.py --negative-test`
    with 186 checks.
  - Python syntax, JSON formatting, `cargo fmt --all -- --check`, `git diff
    --check`, and `python3 ds4-parity/run_parity_report.py
    --skip-local-oracles` with 89 passed, 50 skipped, 0 failed.

#### M13.2: Batched Embedding Replacement Slice

- Status: complete.
- Goal: add an opt-in replacement slice for
  `ds4_gpu_embed_tokens_hc_tensor`.
- Oracle: current-C whole/chunked/resumed prefill output comparators.
- Fixture:
  `ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json`.
- Comparator: `ds4-parity/check_backend_batched_embedding_slice.py
  --negative-test` plus M10.6 prefill pair comparators.
- Acceptance: batched embedding output fields match current-C fixtures and
  unsupported backends fail closed without changing the default route.
- Drift policy: output drift requires a same-B300 current-C oracle refresh.
- Evidence:
  - Added the M13.2 Rust replacement slice descriptor and explicit emitter
    `--slice` selection.
  - Added
    `ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json`.
  - Added `ds4-parity/check_backend_batched_embedding_slice.py
    --negative-test`.
  - Kept runtime route, general backend replacement, and kernel replacement
    claims false.
  - Validation passed:
    `python3 ds4-parity/check_backend_batched_embedding_slice.py
    --negative-test` (96 checks), `cargo test -p ds4-gpu replacement_slice`
    (4 tests), `python3 ds4-parity/check_backend_replacement_slice.py
    --negative-test` (85 checks), and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
    (90 passed, 50 skipped, 0 failed).

#### M13.3: Indexed Decode Selection Replacement Slice

- Status: complete.
- Goal: add opt-in replacement slices for
  `ds4_gpu_indexer_score_one_tensor` and `ds4_gpu_indexer_topk_tensor`.
- Oracle: current-C long indexed-attention decode comparator.
- Fixture:
  `ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json`.
- Comparator: `ds4-parity/check_backend_indexed_decode_slice.py
  --negative-test` plus the long indexed-attention pair comparator.
- Acceptance: indexer score and selected-row outputs match current-C fixtures
  and unsupported backends fail closed without changing the default route.
- Drift policy: score or selected-row drift requires a same-B300 long
  indexed-attention oracle refresh.
- Evidence:
  - Added explicit Rust replacement slice descriptors for
    `ds4_gpu_indexer_score_one_tensor` and `ds4_gpu_indexer_topk_tensor`.
  - Added the M13.3 slice-set fixture and
    `ds4-parity/check_backend_indexed_decode_slice.py --negative-test`.
  - The slices use the M13.1 pair-comparator-ready rows, require explicit
    per-slice selection, fail closed for CPU, Metal, and runtime-default-route,
    and keep runtime route, general backend replacement, and kernel replacement
    claims false.
  - Validation passed:
    `python3 ds4-parity/check_backend_indexed_decode_slice.py
    --negative-test` (195 checks), `cargo test -p ds4-gpu replacement_slice`
    (6 tests), `python3 ds4-parity/check_backend_replacement_slice.py
    --negative-test` (85 checks), `python3
    ds4-parity/check_backend_batched_embedding_slice.py --negative-test` (96
    checks), and `python3 ds4-parity/run_parity_report.py
    --skip-local-oracles` (91 passed, 50 skipped, 0 failed).

#### M13.4: Batch Indexer Fixture Gap Closure

- Status: complete.
- Goal: close fixture gaps for `ds4_gpu_indexer_scores_prefill_tensor`,
  `ds4_gpu_indexer_scores_decode_batch_tensor`, and
  `ds4_gpu_dsv4_topk_mask_tensor`.
- Oracle: current-C prefill and batch-decode indexer paths on B300.
- Fixture:
  `ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json`.
- Comparator: `ds4-parity/check_backend_batch_indexer_fixtures.py
  --negative-test`.
- Acceptance: each batch-only indexer operation has current-C fixture coverage
  before it can join an expanded replacement route.
- Drift policy: fixture drift requires a same-hardware current-C oracle refresh
  and tolerance rationale.
- Evidence:
  - Added the M13.4 batch indexer fixture bundle for
    `ds4_gpu_indexer_scores_prefill_tensor`,
    `ds4_gpu_indexer_scores_decode_batch_tensor`, and
    `ds4_gpu_dsv4_topk_mask_tensor`.
  - Added `ds4-parity/check_backend_batch_indexer_fixtures.py
    --negative-test`.
  - Added the missing current-C debug dump hook for `comp_mask` after
    `ds4_gpu_dsv4_topk_mask_tensor`.
  - The bundle records B300-rerunnable current-C fixture contracts, exact source
    anchors, output/dtype contracts, and rerun commands while keeping raw tensor
    bodies out of the repository and route/default-route/general/backend/kernel
    replacement claims false.
  - Validation passed:
    `python3 ds4-parity/check_backend_batch_indexer_fixtures.py
    --negative-test` (182 checks), `arch -arm64 make
    ds4-prefill-whole-short-oracle-dump`, B300 fixture probe
    (`m134_fixture_probe=ok`), and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
    (92 passed, 50 skipped, 0 failed).

#### M13.5: Embedding/Indexer Route Gate And Closure

- Status: complete.
- Goal: expose the expanded embedding/indexer route through an opt-in gate and
  record retained-sidecar closure.
- Oracle: M13.1 through M13.4 artifacts plus M10.9 official-vector,
  long-context, tool/server, and benchmark gates.
- Fixture:
  `ds4-parity/baselines/backend/m13.5/expanded-route-gate.json` and
  `ds4-parity/baselines/backend/m13.5/expanded-route-closure.json`.
- Comparator: `ds4-parity/check_backend_expanded_route_closure.py
  --negative-test`.
- Acceptance: expanded route stays opt-in, default route stays
  current-backend, and removal is rejected unless every operation in the family
  is covered.
- Drift policy: route drift keeps all prior current-backend oracles until
  default-route parity is proven.
- Evidence:
  - Added the M13.5 expanded route gate and closure matrix for the full
    embedding/indexer operation family.
  - The closure matrix records `ds4_gpu_embed_token_hc_tensor`,
    `ds4_gpu_embed_tokens_hc_tensor`, `ds4_gpu_indexer_score_one_tensor`, and
    `ds4_gpu_indexer_topk_tensor` as opt-in Rust replacement slices.
  - The closure matrix keeps `ds4_gpu_indexer_scores_prefill_tensor`,
    `ds4_gpu_indexer_scores_decode_batch_tensor`, and
    `ds4_gpu_dsv4_topk_mask_tensor` on retained current-backend sidecars.
  - Default route activation, general backend replacement, kernel replacement,
    and removals remain blocked until a post-M13 decision.
  - Validation passed: `python3
    ds4-parity/check_backend_expanded_route_closure.py --negative-test` (279
    checks), `cargo test -p ds4-gpu backend_route_gate` (4 tests), `python3
    ds4-parity/check_backend_runtime_route_gate.py --negative-test` (135
    checks), `python3 ds4-parity/check_backend_batch_indexer_fixtures.py
    --negative-test` (182 checks), and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` (93 passed, 50
    skipped, 0 failed).

### Post-M13 Roadmap Decision

- Status: complete.
- Goal: close the active post-M13 roadmap decision gate without selecting an
  unsupported removal or default-route-promotion stage.
- Oracle: M13.0 through M13.5 artifacts plus the M10.9 runtime graph evidence.
- Fixture:
  `ds4-parity/baselines/roadmap/post-m13/post-m13-roadmap-decision.json`.
- Comparator: `ds4-parity/check_post_m13_roadmap_decision.py
  --negative-test`.
- Acceptance: all M13 stages are recorded complete, no next implementation
  stage is selected, the default route remains current-backend, retained
  current-backend sidecars block C/GPU removals, and open decisions are deferred
  to a future roadmap.
- Drift policy: any future scope must start by adding a new roadmap section with
  its own oracle, fixture, comparator, acceptance, and validation plan.
- Evidence:
  - Added the post-M13 roadmap decision artifact and checker.
  - The decision records that no next implementation stage is selected, no C/GPU
    removal is allowed, and future work requires a new roadmap with new oracles.
  - Validation passed: `python3
    ds4-parity/check_post_m13_roadmap_decision.py --negative-test` (100
    checks), `python3 ds4-parity/check_backend_expanded_route_closure.py
    --negative-test` (279 checks), and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` (94 passed, 50
    skipped, 0 failed).

### M14: Rust CUDA Ownership Via cuda-oxide

- Status: split before implementation into M14.0 through M14.6.
- Goal: replace all resource management and kernel logic currently owned by
  `ds4_cuda.cu` with Rust code using the verified `cuda-oxide` substrate.
- Oracle: current-C CUDA execution plus the existing M10 through M13 parity
  artifacts.
- Comparator: source-hashed ownership inventory followed by B300 stage
  comparators and end-to-end closure gates.
- Acceptance: every CUDA export and unique CUDA kernel is assigned to an
  executable Rust-ownership stage; no route promotion or C CUDA removal occurs
  until all stages close.
- Drift policy: refresh M14.0 whenever the CUDA/FFI/build ownership surface or
  pinned `cuda-oxide` source revision changes.

#### M14.0: CUDA Rust Ownership Inventory And Adoption Contract

- Status: done
- Goal: freeze the complete CUDA-to-Rust ownership surface before CUDA runtime
  implementation changes.
- Source evidence needed: `ds4_cuda.cu`, `ds4_gpu.h`,
  `rust/ds4-gpu-sys/src/lib.rs`, `rust/ds4-gpu/build.rs`, and verified
  `cuda-oxide` source at
  `0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200`.
- Oracle: the current exported CUDA and kernel symbol sets plus the current
  Rust FFI/CUDA build wiring.
- Fixture:
  `ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json`.
- Comparator: `ds4-parity/check_cuda_rust_ownership_inventory.py
  --negative-test`.
- Validation needed: the checker accepts the inventory, rejects missing
  ownership assignments and removal overclaims, and the unified report accepts
  the new comparator.
- Owner path: `RUST_PORT_ROADMAP.md`, `.memory/`,
  `ds4-parity/baselines/backend/m14.0/`, `ds4-parity/`.

#### M14.1: cuda-oxide Substrate And Tensor Residency

- Status: split before implementation into M14.1a through M14.1c.
- Goal: introduce an opt-in Rust CUDA substrate path backed by `cuda-oxide`
  for context/stream ownership, tensor allocation/copy/fill, synchronization,
  model-map residency/cache, and memory reporting.
- Source evidence needed: M14.0 ownership inventory, current
  `ds4_cuda.cu` lifecycle/cache implementation, `rust/ds4-gpu` facade, and
  `cuda-oxide` `cuda-core` residency/launch/BLAS contracts.
- Oracle: current-C CUDA tensor/resource behavior on B300.
- Comparator: tensor allocation/copy/read-write and model residency smoke
  fixtures executed through both current-C and opt-in Rust CUDA paths.
- Validation needed: Rust CUDA feature compiles on B300 with the pinned
  `cuda-oxide` revision, the opt-in substrate fixture compares cleanly, and
  current default route remains unchanged.
- Owner path: Rust CUDA backend modules, Cargo/build integration,
  `ds4-parity/baselines/backend/m14.1/`, `.memory/`.

##### M14.1a: Host Substrate Buffer Roundtrip

- Status: done
- Goal: add a feature-gated Rust CUDA crate pinned to the verified
  `cuda-oxide` fork revision and prove context/stream ownership, device
  transfer, and managed-buffer lifetime on B300.
- Source evidence needed: M14.0 inventory, `cuda-core` context/stream,
  `DeviceBuffer`, and `ManagedBuffer` APIs, and the live B300 CUDA/toolchain
  availability check.
- Oracle: current-C initialization, allocation/write/read/synchronize, and
  managed-allocation behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json`.
- Comparator: `ds4-parity/check_cuda_oxide_substrate_smoke.py
  --negative-test` plus B300 execution of the Rust smoke binary.
- Validation needed: local default-feature workspace tests remain buildable;
  feature-enabled CUDA build/run succeeds on B300 or records the exact
  toolchain blocker; routing remains unchanged.
- Owner path: `rust/ds4-cuda/`, workspace Cargo files, `ds4-parity/`,
  `.memory/`.
- Evidence: feature-gated `rust/ds4-cuda` pins
  `cuda-oxide` revision `0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200`;
  B300 execution on `NVIDIA B300 SXM6 AC` passed device roundtrip,
  zeroed-buffer roundtrip, and managed-buffer lifetime checks without claiming
  kernel or route ownership. Local workspace tests, the 53-check M14.1a
  checker, the 124-check M14.0 inventory checker, and the unified report (96
  passed, 50 skipped, 0 failed) passed. Non-interactive Claude review was
  blocked by missing local CLI login; adversarial self-review found no
  material feature-boundary, dependency-revision, evidence-scope, or route
  ownership issue.

##### M14.1b: Model Residency And Command Lifetime

- Status: split before implementation into M14.1b1 through M14.1b4.
- Goal: move model residency, caching/policy, and command-lifetime ownership
  onto the Rust substrate in separately executable cuts.
- Oracle: current-C model-backed B300 resource behavior.
- Comparator: stage-specific B300 resource fixtures and closure checks.
- Validation needed: each cut passes its B300 model-backed comparison and
  preserves the default route.
- Owner path: Rust CUDA substrate and M14.1 artifacts.

###### M14.1b1: Bounded Model Residency Handles

- Status: done
- Goal: prove cuda-oxide managed advice/prefetch, mapped-host device pointer,
  and registered caller-owned host lifetime on a bounded window read from the
  real B300 model.
- Oracle: current-C model prefetch and host-registration intent, without
  claiming model-range cache selection or graph consumption.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json`.
- Comparator: `ds4-parity/check_model_residency_handles_smoke.py
  --negative-test` plus B300 execution of the Rust smoke binary.
- Validation needed: feature-enabled B300 run over the pinned model window;
  local default-feature workspace tests and unified report remain passing; no
  complete-model-map, kernel, or route claim.
- Owner path: `rust/ds4-cuda/`, `ds4-parity/`, `.memory/`.
- Evidence: the feature-enabled B300 smoke used a 4096-byte prefix of
  `/workspace/ds4/ds4flash.gguf` and passed managed advice/prefetch,
  mapped-host device-pointer, and registered-host lifetime checks on
  `NVIDIA B300 SXM6 AC`, while explicitly reporting no complete-model-map,
  kernel, or route ownership. A live SHA256 refresh confirmed
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`.
  Local workspace tests, the 64-check M14.1b1 gate, prior M14.1a/M14.0
  gates, and unified parity (97 passed, 50 skipped, 0 failed) passed;
  non-interactive Claude was unavailable due missing CLI login, and
  self-review closed the hash-evidence gap.

###### M14.1b2: Model Map And Range Cache Policy

- Status: split before implementation into M14.1b2a through M14.1b2c.
- Goal: port model-map and model-range cache strategies as bounded,
  executable Rust-owned cuts with current-C retained as oracle.

####### M14.1b2a: Owned Mmap Device Range Copy

- Status: done
- Goal: own model file/mmap lifetime in Rust and prove bounds-checked,
  byte-exact cached CUDA device-range copy plus reuse on B300.
- Oracle: current-C model fd/map/range-cache intent, limited to device-copy
  cache behavior.
- Fixture: `ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json`.
- Comparator: `ds4-parity/check_model_range_copy_smoke.py --negative-test`
  plus B300 Rust smoke.
- Validation needed: pinned-model B300 exact readback and cache reuse; no
  direct-I/O, HMM, registered-zero-copy, Q8, kernel, or route claim.
- Evidence: the feature-enabled B300 smoke mmaped the pinned GGUF, rejected
  an out-of-bounds range, copied/read back one 4096-byte CUDA device range,
  and reused exactly one cache entry on `NVIDIA B300 SXM6 AC`; strategy,
  kernel, and route ownership remain false. Feature compilation first exposed
  an invalid `Debug` derive, and self-review then found a rejected-null
  `mmap` cleanup leak; both were fixed before the final B300 rerun. Local
  workspace tests, the 64-check M14.1b2a gate, prior gates, and unified
  parity (98 passed, 50 skipped, 0 failed) passed; non-interactive Claude was
  unavailable due missing CLI login.

####### M14.1b2b: Model Range Strategy Parity

- Status: split before implementation into M14.1b2b1 through M14.1b2b3.
- Goal: port the independently verifiable model-range strategy branches with
  current-C retained as the comparison source.

######## M14.1b2b1: File-Staged Range Strategy

- Status: done
- Goal: prove explicit mmap-copy versus file-staged-copy selection and
  byte-exact selected-range reuse on B300.
- Oracle: current-C device-copy and `cuda_model_range_ptr_from_fd` branches,
  excluding O_DIRECT, registration, HMM, staging-ring, and cache-budget
  policy.
- Validation needed: Rust B300 readbacks from both selected strategies match
  exactly and reuse their cache entries without kernel or route claims.
- Fixture: `ds4-parity/baselines/backend/m14.1b2b1/model-range-strategy-smoke.json`.
- Evidence: the B300 strategy smoke cached and reused a 4096-byte range
  independently through mmap-source and file-staged-source uploads, and both
  device readbacks exactly matched the pinned GGUF bytes. Registered mapped
  ranges, pageable HMM, O_DIRECT/staging policy, kernels, and route activation
  remain unclaimed. A live model SHA256 refresh matched the recorded GGUF;
  workspace tests, the 73-check M14.1b2b1 gate, retained M14 gates, and
  unified parity (99 passed, 50 skipped, 0 failed) passed. Non-interactive
  Claude produced no review result before termination; self-review found no
  material issue.

######## M14.1b2b2: Registered Range Strategy

- Status: done
- Goal: add page-aligned read-only mapped registration selection and the
  mmap-sourced device-copy fallback used when CUDA registration fails.
- Oracle: current-C page-aligned `cudaHostRegisterMapped |
  cudaHostRegisterReadOnly` attempt and its `cudaMemcpy` fallback after a
  registration failure; the file-descriptor strategy remains M14.1b2b1.
- Fixture: `ds4-parity/baselines/backend/m14.1b2b2/model-registered-range-smoke.json`.
- Evidence: cuda-oxide revision `b938480882f208045bc36ecf29da1ec5531d55ba`
  adds the immutable read-only registration handle. The B300 smoke expanded
  requested range `13..4109` to page-aligned range `0..8192`, observed CUDA
  error `801` (`operation not supported`) from read-only registration, then
  read back the exact 4096 requested bytes through the mmap-copy fallback and
  reused its cache entry. This is an explicit fallback proof, not a claim
  that zero-copy registration succeeds on B300 or that current-C's
  cross-range registration-disable state is already ported. Validation
  passed: local workspace tests and formatting, the M14.1b2b2 checker,
  retained M14 gates, `git diff --check`, B300 feature tests and predecessor
  smoke, and unified parity (100 passed, 50 skipped, 0 failed).
  Non-interactive Claude review was unavailable because the local CLI
  reported `Not logged in`; self-review corrected the fallback source from
  file staging to the current-C mmap copy branch before final validation.

######## M14.1b2b3a: Pageable HMM Range Strategy

- Status: done
- Goal: add page-aligned pageable-memory advice/prefetch and direct-pointer
  readback for a bounded mmap-backed model range without combining it with
  file-staging policy.
- Oracle: current-C `cuda_model_prefetch_range` and `g_model_hmm_direct`
  selection in `cuda_model_range_ptr`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b3a/model-pageable-hmm-smoke.json`.
- Evidence: corrected cuda-oxide revision
  `361300ea643688eea87eaa215d9a62a5e74a30e6` provides an immutable
  pageable-HMM handle with asynchronous prefetch kept unsafe until stream
  completion; the DS4 proof wrapper synchronizes before returning. On B300,
  requested range `13..4109` expanded to `0..8192`, pageable-memory access
  was available with host page tables disabled, both advice calls and
  prefetch succeeded, and direct HMM readback matched the exact requested
  4096 bytes. O_DIRECT/staging, asynchronous production policy, kernels, and
  route activation remain unclaimed. Validation passed local workspace tests
  and formatting, the 73-check M14.1b2b3a comparator, retained M14
  comparators, `git diff --check`, B300 feature tests and predecessor smoke,
  and unified parity (101 passed, 50 skipped, 0 failed). Non-interactive
  Claude review was unavailable because the CLI reported `Not logged in`;
  self-review caught and fixed the cuda-oxide asynchronous borrowed-prefetch
  safety defect before this revision was pinned.

######## M14.1b2b3b: Direct-I/O Staging Policy

- Status: split into M14.1b2b3b1 and M14.1b2b3b2 before implementation
- Goal: add direct-I/O read selection and asynchronous pinned staging as
  separate current-C policy slices.

######### M14.1b2b3b1: Direct-I/O Pinned Read Selection

- Status: done
- Goal: add Linux `O_DIRECT` aligned read selection and buffered fallback
  through synchronized pinned host-to-device upload.
- Oracle: current-C `ds4_gpu_set_model_fd`, `cuda_model_stage_read`, and the
  selected transfer portion of `cuda_model_range_ptr_from_fd`, excluding
  event-ring overlap and arena/budget state.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b3b1/model-direct-io-smoke.json`.
- Evidence: the B300 smoke read requested range `13..4109` through an
  `O_DIRECT` aligned pinned window `0..8192` at alignment `4096`, uploaded
  the requested bytes through CUDA, and read them back exactly. Because the
  pinned model length is not direct-I/O aligned, its final 13-byte request
  exercised the buffered fallback and also read back exactly. Asynchronous
  ring/event scheduling, cache-budget and persistent disable-after-error
  policy, kernels, and route activation remain unclaimed.
- Validation: local workspace tests, formatting and diff checks, the
  79-check new comparator, retained M14 comparators, B300 feature-enabled
  crate tests and predecessor HMM smoke, and unified parity (102 passed, 50
  skipped, 0 failed) passed. Non-interactive Claude review was unavailable
  because the CLI reported `Not logged in`; adversarial self-review found no
  lifetime, alignment, fallback, or bounded-claim defect.

######### M14.1b2b3b2: Asynchronous Staging Ring And Budget Policy

- Status: done
- Goal: add multi-buffer event-driven overlap, direct-I/O error suppression,
  and range-cache arena/budget behavior.
- Oracle: current-C `cuda_model_stage_pool_alloc`, `cuda_model_stage_read`,
  `cuda_model_arena_alloc`, and `cuda_model_range_ptr_from_fd`, excluding
  source-page discard/progress side effects and compute-kernel consumption.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json`.
- Evidence: the B300 smoke uploaded seven direct-I/O chunks through a
  four-slot pinned ring with two event-guarded slot reuses, admitted two
  ranges into one 32,768-byte arena totaling 28,672 bytes, selected budget
  fallback for the next byte, rejected a new arena whose aligned reservation
  exceeded its remaining raw-byte budget, and read back both admitted ranges
  exactly. Intermediate upload errors drain already-enqueued copies before
  staging-slot state is cleared. The feature-enabled crate test covers the current-C direct-I/O
  disable-after-selected-error errno set while the live smoke explicitly
  does not claim an induced I/O error. Source-page discard/progress policy,
  kernels, and route activation remain unclaimed.
- Validation: local workspace tests, formatting and diff checks, the
  96-check new comparator, retained M14 comparators, B300 feature-enabled
  crate tests and predecessor direct-I/O smoke, and unified parity (103
  passed, 50 skipped, 0 failed) passed. Local feature compilation requires
  CUDA headers unavailable on this host; B300 supplied that gate.
  Non-interactive Claude review was unavailable because the CLI reported
  `Not logged in`; adversarial self-review fixed error-path draining and
  aligned new-arena admission defects before recording no remaining
  lifetime, policy, or bounded-claim issue.

####### M14.1b2c: Model Map Cache Closure

- Status: done
- Goal: close model-map/range-cache ownership, remaining source-page
  discard/progress policy, and retained-current-C route evidence before
  allocation/quality work.
- Oracle: current-C `cuda_model_range_ptr`, `cuda_model_drop_file_pages`,
  `cuda_model_discard_source_pages`, `cuda_model_load_progress_note`, and
  `cuda_model_range_release_all`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json`.
- Evidence: the B300 smoke cached 8,192 bytes and served a 257-byte interior
  range exactly without another upload; its two staging chunks issued two
  file discard calls totaling 8,192 bytes and two page-aligned mapping
  discard calls totaling 16,384 bytes. A retention-policy cache issued no
  discard calls. Explicit non-TTY progress emitted the initial current-C
  message once for three notes, disabled progress emitted nothing, and a
  fresh cache began with empty state. Physical eviction, runtime environment
  or terminal selection wiring, kernels, and route activation remain
  unclaimed.
- Validation: local workspace tests, formatting and diff checks, the 84-check
  new comparator, retained M14 comparators, B300 feature-enabled crate tests
  and predecessor asynchronous-staging smoke, and unified parity (104 passed,
  50 skipped, 0 failed) passed. Local feature compilation requires CUDA
  headers unavailable on this host; B300 supplied that gate. Non-interactive
  Claude review was unavailable because the CLI reported `Not logged in`;
  adversarial self-review fixed a progress-threshold overflow edge before
  recording no remaining pointer, advisory-claim, progress, lifetime, or
  bounded-claim issue.

###### M14.1b3: Allocation And Quality Policy

- Status: split before implementation into M14.1b3a and M14.1b3b.
- Goal: port managed-KV, Q8/F16 cache, quality-mode, and memory-report policy
  without adding compute kernels.

####### M14.1b3a: Managed KV And Memory Report Policy

- Status: done
- Goal: port managed-tensor allocation proof, managed-KV selection, and
  memory-report formatting without claiming converted-weight caches or BLAS
  math-mode behavior.
- Oracle: current-C `ds4_gpu_tensor_alloc_managed`,
  `cuda_managed_kv_reserve_bytes`, `ds4_gpu_should_use_managed_kv_cache`, and
  `ds4_gpu_print_memory_report`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b3a/allocation-policy-smoke.json`.
- Evidence: cuda-oxide revision `0ec61156a7c5d65802402898b7a197bfff266d31`
  adds `CudaContext::memory_info()`. The B300 smoke queried valid live
  capacity, allocated managed memory, matched current-C report formatting,
  and exercised empty KV, forced managed KV, unavailable-query, sufficient
  capacity, reserve-pressure, and context-exceeds-free decisions. Q8 caches,
  quality mode, kernels, and route activation remain unclaimed.
- Validation: local workspace tests, formatting and diff checks, the 64-check
  comparator and retained M14 comparators, full B300 `cuda-core` tests, B300
  feature-enabled `ds4-cuda` tests, predecessor model-map closure smoke, and
  unified parity (105 passed, 50 skipped, 0 failed) passed. Non-interactive
  Claude review timed out without a completed result; adversarial self-review
  found no threshold, reserve, transient-capacity, report-format,
  dependency-pin, or bounded-claim issue.

####### M14.1b3b: Q8 Cache And Quality Policy

- Status: done
- Goal: port Q8 cache admission/failure-disable policy and quality-mode BLAS
  selection without claiming converted buffers or changing the default
  runtime route.
- Oracle: current-C `cuda_q8_f16_cache_reserve_bytes`,
  `cuda_q8_f16_cache_allowed`, `cuda_q8_f16_preload_allowed`,
  `cuda_q8_f16_cache_has_budget`,
  `cuda_q8_f16_cache_disable_after_failure`,
  `cuda_q8_f32_cache_allowed`, and `ds4_gpu_set_quality`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b3b/q8-quality-policy-smoke.json`.
- Evidence: cuda-oxide revision `aabe10dc4fa0086375104458909e222d1ac1cfe3`
  adds typed `Blas::set_math_mode(BlasMathMode)` and passed its B300
  `cublas-sys` plus full `cuda-core` tests. Rust Q8 policy covers F16
  eligibility/preload/reserve/budget/failure-disable behavior and F32
  optional-preload selection. The B300 DS4 smoke applied TF32 fast mode and
  both default-math paths through live cuBLAS. Converted Q8 buffers and their
  failure-time synchronization/release, dequant kernels assigned to M14.3,
  compute kernels, and route activation remain unclaimed.
- Validation: local workspace tests, formatter and diff checks, the 71-check
  comparator and retained M14 comparators, B300 `cublas-sys` and full
  `cuda-core` tests, B300 feature-enabled `ds4-cuda` tests, and unified
  parity (106 passed, 50 skipped, 0 failed) passed. Non-interactive Claude
  review timed out without a completed result; adversarial self-review fixed
  the reserve-equality boundary test and narrowed failure ownership to
  disable-state policy before recording no remaining bounded-claim defect.

###### M14.1b4: Fill Kernel And Command Lifetime

- Status: done
- Goal: prove an executable-local Rust `fill_f32` CUDA kernel and current-C
  command completion semantics on the opt-in substrate.
- Oracle: current-C `ds4_gpu_tensor_fill_f32`, `fill_f32_kernel`,
  `ds4_gpu_flush_commands`, `ds4_gpu_end_commands`, and
  `ds4_gpu_synchronize`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b4/fill-command-lifetime-smoke.json`.
- Evidence: cuda-oxide tool revision
  `981e3244a107d84d807cfb087793269c477cc764` fixes B300 `cargo oxide run`
  target selection by retaining the backend-selected portable `sm_80` target
  for a basic kernel instead of forcing rejected `sm_103` PTX. The B300
  `ds4-cuda-fill-lifetime-smoke` execution proved prefix and
  negative-infinity fills, zero-count and bounds behavior, current-C's no-op
  begin command, and context-wide flush/end/synchronize wrappers through a
  Rust `#[kernel]`. This does not
  claim reusable library embedded-kernel linkage, dequant or graph compute
  kernels, runtime graph integration, or route activation.
- Validation: local workspace tests, formatter and diff checks, the
  fill/command-lifetime
  comparator and retained M14 checks, B300 feature-enabled `ds4-cuda` tests,
  B300 `cargo-oxide` tests and live kernel execution, and unified parity
  (107 passed, 50 skipped, 0 failed) passed. Non-interactive Claude review
  timed out without a completed result; adversarial self-review found no
  remaining ownership, fill-semantic, synchronization, or evidence defect.

##### M14.1c: Substrate Route Closure Gate

- Status: done
- Goal: close M14.1 so M14.2 kernels can depend on Rust-owned resource
  behavior while keeping C CUDA as the retained oracle.
- Oracle: M14.0 and M14.1a/M14.1b artifacts.
- Fixture:
  `ds4-parity/baselines/backend/m14.1c/substrate-route-closure.json`.
- Comparator: `ds4-parity/check_substrate_route_closure.py --negative-test`
  and B300 feature-test plus fill-kernel rerun contract.
- Evidence: corrected the inventory so `ds4_gpu_cache_q8_f16_range` and both
  dequant kernels remain M14.3 work; M14.1 owns `fill_f32_kernel` and an
  explicit Rust no-op `begin_commands` command facade. Default-route
  promotion and C CUDA removal remain rejected.
- Validation: local formatting, diff and workspace tests passed; the updated
  81-check M14.1b4 comparator and 139-check closure comparator passed; B300
  feature-enabled `ds4-cuda` tests passed with 21 tests and live cargo-oxide
  fill execution reported `begin_is_noop:true`; unified parity passed with
  108 passed, 50 skipped, and 0 failed. Non-interactive Claude review timed
  out without a completed result; adversarial self-review found no remaining
  ownership-boundary or route-claim defect.
- Owner path: M14.1 artifacts and route policy.

#### M14.2: Embedding Indexer And Elementwise Kernels

- Status: done through M14.2e; M14.2b is
  further split after B300 exposed a separate libdevice/NVVM SwiGLU blocker;
  M14.2d is split into scalar fallback proof and optimized dispatch ownership;
  M14.2d2 is split into direct-one, tensor-core score, and specialized top-k
  slices; M14.2d2b is split into base-tile and widened multi-warp score
  ownership because those paths have separate launch and validation evidence;
  M14.2d2b2 is split into WMMA32, WMMA64, and WMMA128/priority slices.
- Stage split: M14.2a Add And Repeat Elementwise Kernels; M14.2b1
  Directional Steering Projection Kernel; M14.2b2 SwiGLU Libdevice Path;
  M14.2c Embedding Kernel Pair; M14.2d1 Scalar Indexer Selection Kernels;
  M14.2d2a Direct-One Indexer Score Kernel; M14.2d2b1 Base Tensor-Core
  Indexer Score Kernel; M14.2d2b2a WMMA32 Tensor-Core Indexer Score Kernel;
  M14.2d2b2b WMMA64 Tensor-Core Indexer Score Kernel; M14.2d2b2c WMMA128
  Tensor-Core Indexer Score Kernel And Dispatch Priority; M14.2d2c1
  1024 Bitonic Top-K Kernel; M14.2d2c2 Power-Of-Two Top-K Kernels;
  M14.2d2c3 CUB-Or-Equivalent Top-K Branch; M14.2d2c4 Chunked And
  Tree-Merge Top-K Kernels; M14.2d2c5 Indexed Ascending Top-K Sort And
  Dispatch Policy; M14.2e Kernel Closure Gate.

##### M14.2a: Add And Repeat Elementwise Kernels

- Status: done
- Goal: port `add_kernel` and `repeat_hc_kernel` through an executable-local
  Rust cuda-oxide smoke while keeping the route opt-in.
- Oracle: current-C add/repeat kernels and exported tensor wrappers.
- Fixture:
  `ds4-parity/baselines/backend/m14.2a/elementwise-kernel-smoke.json`.
- Comparator: `ds4-parity/check_elementwise_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added Rust `add_kernel` and `repeat_hc_kernel` with current-C
  256-thread launch geometry; B300 feature-enabled `ds4-cuda` tests passed
  with 22 tests and live cargo-oxide execution selected `sm_80` and proved
  add output, repeat-HC output, add bounds rejection, and repeat-shape
  rejection. Local formatter, diff, and workspace tests passed; the 69-check
  comparator passed and unified parity passed with 109 passed, 50 skipped,
  and 0 failed. Non-interactive Claude review timed out without a completed
  result; adversarial self-review fixed the `repeat_hc` wrapper to preserve
  current-C's 64-bit shape product before B300 execution. Embedding,
  indexer/top-k, SwiGLU, directional steering, route activation, and removal
  remain unclaimed.

##### M14.2b1: Directional Steering Projection Kernel

- Status: done
- Goal: port in-place directional steering projection with current-C-shaped
  shared-memory reduction without claiming SwiGLU or route activation.
- Fixture:
  `ds4-parity/baselines/backend/m14.2b1/directional-steering-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_directional_steering_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added Rust `directional_steering_project_kernel` using
  `SharedArray<f32, 256>`, `thread::sync_threads()`, and in-place row
  projection. B300 feature-enabled `ds4-cuda` tests passed with 23 tests and
  live cargo-oxide execution selected `sm_80` and proved directional output
  and shape rejection. A combined SwiGLU experiment exposed that `f32::exp()`
  selects NVVM IR whose opaque-pointer function signature CUDA 13.2
  `libnvvm` rejects with `parse expected type`; SwiGLU remains unclaimed.
  Local formatter, diff, and workspace tests passed; the 71-check directional
  comparator and unified parity report passed with 110 passed, 50 skipped,
  and 0 failed. Non-interactive Claude review timed out without a completed
  result; self-review retained the in-place row-ownership/synchronization
  proof and explicit unowned SwiGLU blocker.

##### M14.2b2: SwiGLU Libdevice Path

- Status: done
- Goal: make cuda-oxide execute current-C-shaped SwiGLU math on B300 after
  repairing the blocked libdevice/NVVM path.
- Fixture:
  `ds4-parity/baselines/backend/m14.2b2/swiglu-kernel-smoke.json`.
- Comparator: `ds4-parity/check_swiglu_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: pushed cuda-oxide revision
  `d4791b7002152af3b7f6b15a48d7f5acd7a63011`, which converts the
  failed opaque-pointer NVVM path into portable PTX plus context-targeted
  libdevice cubin loading. The Rust `swiglu_kernel` preserves current-C
  finite and NaN clamps, unclamped behavior, SiLU exponential, output weight,
  and bounds behavior through a typed launch wrapped around
  `cuda_host::ltoir`. B300 feature-enabled
  `ds4-cuda` tests passed with 24 tests; live execution emitted portable
  `sm_80` PTX, generated `ds4_cuda_swiglu_smoke.sm_103.cubin`, and proved
  clamped output, unclamped output, and invalid-shape rejection with no
  compile or link target overrides. Embedding, indexer/top-k, route
  activation, and C CUDA removal remain unclaimed.
- Validation: local workspace tests, formatter/diff checks, the 73-check
  SwiGLU comparator, and unified parity passed with 116 passed, 45 skipped,
  and 0 failed. Non-interactive Claude review timed out without a completed
  result; adversarial self-review caught a NaN clamp divergence caused by
  optimized-away float comparisons and fixed it with explicit IEEE-754 bit
  classification before final B300 validation.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2c: Embedding Kernel Pair

- Status: done
- Goal: port primitive-FP16 single-token hidden-copy and batched embedding
  loads through executable-local Rust cuda-oxide kernels without claiming
  model-map routing.
- Oracle: current-C `embed_token_hc_kernel`, `embed_tokens_hc_kernel`, and
  their exported tensor wrappers.
- Fixture:
  `ds4-parity/baselines/backend/m14.2c/embedding-kernel-smoke.json`.
- Comparator: `ds4-parity/check_embedding_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added Rust `embed_token_hc_kernel` and
  `embed_tokens_hc_kernel` using primitive `f16` widened to `f32`, with
  repeated hidden-copy output and current-C batch invalid-token fallback to
  row zero. The Rust single-token helper additionally rejects invalid token
  rows before device launch. B300 feature-enabled `ds4-cuda` tests passed
  with 25 tests and live execution selected portable `sm_80` PTX and proved
  single-token output, batched fallback output, and host-side rejection on
  `NVIDIA B300 SXM6 AC`. Local formatting, diff, and workspace tests passed;
  the 69-check embedding comparator and unified parity passed with 117
  passed, 45 skipped, and 0 failed. Non-interactive Claude review timed out
  without a completed result; adversarial self-review confirmed the
  valid-call/batch-fallback match and documented single-token safety
  strengthening.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d1: Scalar Indexer Selection Kernels

- Status: done
- Goal: port scalar fallback scoring, selection, and mask kernels without
  claiming optimized score or top-k launch branches.
- Oracle: current-C `indexer_scores_kernel`, `indexer_topk_kernel`,
  `topk_mask_kernel`, and fallback launch sites.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d1/indexer-scalar-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_scalar_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Evidence: added executable-local Rust `indexer_scores_kernel`,
  `indexer_topk_kernel`, and `topk_mask_kernel`; B300 feature-enabled
  `ds4-cuda` tests passed with 26 tests and live cargo-oxide execution selected
  portable `sm_80` PTX and proved score output, causal masking, scalar top-k
  output and tie ordering, top-k mask output, and invalid-shape rejection on
  `NVIDIA B300 SXM6 AC`. Direct-one/WMMA scoring, specialized top-k
  dispatch, route activation, and C CUDA removal remain unclaimed. Local
  formatter/diff checks, workspace tests, the 73-check scalar indexer
  comparator, and unified parity passed with 118 passed, 45 skipped, and 0
  failed. Non-interactive Claude review produced no completed result before
  its timeout; adversarial self-review confirmed the scalar `fmaxf` and
  stable-tie semantics plus the optimized-dispatch non-claim, and aligned
  mask launch sizing with current C before the final B300 rerun.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d2: Optimized Indexer And Top-K Dispatch

- Status: split before implementation into M14.2d2a through M14.2d2c
- Goal: port or explicitly close current-C direct/WMMA score and specialized
  top-k dispatch ownership after scalar fallback proof.
- Oracle: direct-one and WMMA score kernels plus power-of-two, CUB, chunked,
  merged, and tree top-k launch branches.

##### M14.2d2a: Direct-One Indexer Score Kernel

- Status: done
- Goal: port current-C's fixed-shape direct-one score kernel with its
  four-warp shuffle reduction before tensor-core and top-k work.
- Oracle: `indexer_score_one_direct_kernel` and its fixed-shape launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2a/indexer-direct-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_direct_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Evidence: added executable-local Rust `indexer_score_one_direct_kernel`
  using 128-thread geometry, four-warp `warp::shuffle_down_f32` reduction,
  positive-score weighting, causal masking, and host bounds rejection. B300
  feature-enabled `ds4-cuda` tests passed with 27 tests and live cargo-oxide
  execution emitted portable `sm_80` PTX and proved direct output, causal
  masking, NaN/negative clamp behavior, and invalid-shape rejection on
  `NVIDIA B300 SXM6 AC`. WMMA score dispatch, specialized top-k dispatch,
  route activation, and C CUDA removal remain unclaimed. Local
  formatter/diff checks, workspace tests, the 66-check direct indexer
  comparator, and unified parity passed with 119 passed, 45 skipped, and 0
  failed. Non-interactive Claude review produced no completed result before
  its timeout; adversarial self-review confirmed the fixed lane/head
  grouping, shuffle-down reduction, explicit NaN clamp, and non-claims.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d2b: Tensor-Core Indexer Score Kernels

- Status: done through M14.2d2b1 and M14.2d2b2a through M14.2d2b2c
- Goal: port the 16/32/64/128-component WMMA score branches through
  cuda-oxide warp-scoped MMA.

##### M14.2d2b1: Base Tensor-Core Indexer Score Kernel

- Status: done
- Goal: port current-C's 16-component WMMA score tile through cuda-oxide's
  warp-scoped `m16n8k16` intrinsic surface.
- Oracle: `indexer_scores_wmma_kernel` and its 32-thread launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b1/indexer-wmma-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Evidence: added executable-local Rust `indexer_scores_wmma_kernel` using
  native `f16` shared staging, two `mma_m16n8k16_f32_f16` calls per
  `16 x 16` current-C tile, positive-score weighting, scaling, causal
  masking, and host bounds rejection. B300 feature-enabled `ds4-cuda` tests
  passed with 28 tests and live cargo-oxide execution emitted portable
  `sm_80` PTX and proved base output, both eight-column MMA halves,
  per-token weighting, NaN/negative suppression, causal masking, and
  invalid-shape rejection on `NVIDIA B300 SXM6 AC`. A first
  device compile identified cuda-oxide's unsupported generic `u32::min` drop
  glue path; explicit scalar comparisons retained semantics and produced the
  successful live run. Widened WMMA dispatch, specialized top-k dispatch,
  route activation, and C CUDA removal remain unclaimed. Local workspace
  tests, formatter/diff checks, the 72-check base WMMA comparator, and
  unified parity passed with 120 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review produced no completed result before its
  timeout; adversarial self-review expanded the fixture to prove weighted
  output and NaN/negative suppression before the final B300 execution.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d2b2: Widened Tensor-Core Indexer Score Dispatch

- Status: done through M14.2d2b2a through M14.2d2b2c
- Goal: port current-C's 32/64/128-component WMMA score branches and
  dispatch priority after the base tile proof.

##### M14.2d2b2a: WMMA32 Tensor-Core Indexer Score Kernel

- Status: done
- Goal: port current-C's two-warp, 32-component tensor-core score branch
  without claiming larger WMMA dispatch.
- Oracle: `indexer_scores_wmma32_kernel` and its 64-thread launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b2a/indexer-wmma32-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma32_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Evidence: added executable-local Rust `indexer_scores_wmma32_kernel` using
  two warps, native `f16` staging, and cuda-oxide `m16n8k16` MMA calls.
  B300 feature-enabled `ds4-cuda` tests passed with 29 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved WMMA32
  output over two 32-component blocks, two-warp tile mapping, per-token
  weighting, NaN/negative suppression, causal masking, and invalid-shape
  rejection on `NVIDIA B300 SXM6 AC`. WMMA64/WMMA128 dispatch, specialized
  top-k dispatch, route activation, and C CUDA removal remain unclaimed.
  Local formatter/diff checks, workspace tests, the 73-check WMMA32
  comparator, and unified parity passed with 121 passed, 45 skipped, and 0
  failed. Non-interactive Claude review produced no completed result before
  its timeout; adversarial self-review compared two-warp staging,
  accumulator scatter, causal early exit, and explicit `fmaxf` semantics
  against current C.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d2b2b: WMMA64 Tensor-Core Indexer Score Kernel

- Status: done
- Goal: port current-C's four-warp, 64-component tensor-core score branch.
- Oracle: `indexer_scores_wmma64_kernel` and its 128-thread launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b2b/indexer-wmma64-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma64_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Evidence: added executable-local Rust `indexer_scores_wmma64_kernel` using
  four warps, native `f16` staging, and cuda-oxide `m16n8k16` MMA calls.
  B300 feature-enabled `ds4-cuda` tests passed with 30 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved WMMA64
  output over two 64-component blocks, four-warp tile mapping, per-token
  weighting, NaN/negative suppression, causal masking, and invalid-shape
  rejection on `NVIDIA B300 SXM6 AC`. WMMA128 priority, specialized top-k
  dispatch, route activation, and C CUDA removal remain unclaimed. Local
  formatter/diff checks, workspace tests, the 73-check WMMA64 comparator,
  and unified parity passed with 122 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review produced no completed result before its
  timeout; adversarial self-review compared four-warp column ownership,
  accumulator scatter, causal early exit, and explicit `fmaxf` semantics
  against current C.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d2b2c: WMMA128 Tensor-Core Indexer Score Kernel And Dispatch Priority

- Status: done
- Goal: port current-C's eight-warp, 128-component score branch and final
  widened-WMMA priority contract.
- Oracle: `indexer_scores_wmma128_kernel`, its 256-thread launch branch, and
  validated-input `indexer_scores_launch` score-kernel priority.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b2c/indexer-wmma128-dispatch-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma128_dispatch_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Evidence: added executable-local Rust `indexer_scores_wmma128_kernel`
  using eight warps, native `f16` staging, and cuda-oxide `m16n8k16` MMA
  calls; added pure Rust `select_indexer_score_kernel` for the validated
  direct-one, WMMA128/64/32/base, and scalar ordering contract. B300
  feature-enabled `ds4-cuda` tests passed with 32 tests and live cargo-oxide
  execution emitted portable `sm_80` PTX and proved WMMA128 output over two
  128-component blocks, eight-warp tile mapping, per-token weighting,
  NaN/negative suppression, causal masking, invalid-shape rejection, and
  score-dispatch ordering on `NVIDIA B300 SXM6 AC`. Specialized top-k,
  route activation, and C CUDA removal remain unclaimed. Local formatter/diff
  checks, workspace tests, the 84-check WMMA128/dispatch comparator, and
  unified parity passed with 123 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review produced no completed result before its
  timeout; adversarial self-review added direct-one-disabled and
  global-WMMA-disabled selector checks before the final B300 rerun.
- Owner path: Rust cuda-oxide kernel smoke and current-C operation oracle.

##### M14.2d2c: Specialized Top-K Kernels

- Status: split before implementation into M14.2d2c1 through M14.2d2c5
- Goal: port specialized top-k sort, chunk, merge, tree, and indexed-sort
  kernels.

##### M14.2d2c1: 1024 Bitonic Top-K Kernel

- Status: done
- Goal: port the current-C shared-memory `indexer_topk_1024_kernel` used for
  validated `top_k == 512` and `n_comp <= 1024` calls.
- Oracle: `indexer_topk_1024_kernel` and its guarded
  `ds4_gpu_indexer_topk_tensor` launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c1/indexer-topk1024-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk1024_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Acceptance: Rust owns only bounded 1024-element top-k sorting and shape
  rejection; larger top-k branches, indexed ascending sort, runtime route,
  and C CUDA removal remain unclaimed.
- Evidence: added executable-local Rust `indexer_topk_1024_kernel` mirroring
  the current-C 1024-thread shared-memory bitonic network and lower-index tie
  order. B300 feature-enabled `ds4-cuda` tests passed with 33 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved full-width
  output, partial-width sentinel exclusion, stable tie ordering, and
  invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Larger top-k dispatch,
  indexed ascending sort, runtime route, and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 69-check comparator, and
  unified parity passed with 124 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained only bounded kernel-shape and ordering
  ownership.

##### M14.2d2c2: Power-Of-Two Top-K Kernels

- Status: done
- Goal: port 2048/4096 and 8192 power-of-two shared-memory top-k branches.
- Oracle: `indexer_topk_pow2_kernel<2048>`,
  `indexer_topk_pow2_kernel<4096>`, and fallback
  `indexer_topk_pow2_u16_kernel<8192>`.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c2/indexer-topk-pow2-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_pow2_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Evidence: added executable-local Rust 2048/4096 `u32`-index and 8192
  `u16`-index bitonic kernels preserving current-C descending order and
  lower-index ties. B300 feature-enabled tests passed with 34 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved each kernel
  output and sentinel exclusion. CUB selection, chunked merging, indexed
  sort, runtime route, and C CUDA removal remain unclaimed. Local formatting,
  diff, workspace tests, the 76-check comparator, and unified parity passed
  with 125 passed, 45 skipped, and 0 failed. Non-interactive Claude review
  timed out without a completed result; adversarial self-review retained the
  CUB dispatch non-claim.

##### M14.2d2c3: CUB-Or-Equivalent Top-K Branch

- Status: done
- Goal: port or explicitly close current-C's CUB radix-sort optimization.
- Oracle: current-C `topk_float_ordered_key`, `topk_pack_key`,
  `indexer_topk_8192_cub_kernel`, and its dynamic-shared-memory opt-in.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c3/indexer-topk-packed-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_packed_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Evidence: added executable-local Rust
  `indexer_topk_8192_packed_key_equivalent_kernel`, preserving current-C
  ordered-float and lower-index packed-key semantics with sentinel exclusion
  through a dynamic-shared-memory bitonic equivalent. Its initial B300 launch
  failed with `DriverError(1, "invalid argument")` until pinned cuda-oxide
  revision `e9c0d677104751179985098f02212ff044d3ec22` added
  `CudaFunction::set_max_dynamic_shared_memory_size` for the required
  65,536-byte launch. B300 feature-enabled tests passed with 35 tests and
  live cargo-oxide execution proved 4096- and 6000-component output,
  positive-NaN ordering, tie order, sentinel exclusion, and shape rejection
  on `NVIDIA B300 SXM6 AC`. CUB library ownership, dispatch selection,
  chunk/tree merge, indexed sort, runtime route, and C CUDA removal remain
  unclaimed. Local formatting, diff, workspace tests, the 80-check
  comparator, and unified parity passed with 126 passed, 45 skipped, and
  0 failed. Non-interactive Claude review timed out without a completed
  result; adversarial self-review retained the CUB and dispatch non-claims.

##### M14.2d2c4: Chunked And Tree-Merge Top-K Kernels

- Status: done
- Goal: port chunk candidate, intermediate tree merge, and final merge
  kernels together with their scratch layout.
- Oracle: current-C `indexer_topk_chunk_pow2_kernel<4096>`,
  `indexer_topk_tree_merge_pow2_kernel<4096>`,
  `indexer_topk_merge_pow2_kernel<4096>`, and its scratch allocation loop.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c4/indexer-topk-tree-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_tree_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Evidence: added executable-local Rust chunk, intermediate tree-merge, and
  final-merge kernels plus current-C-shaped contiguous scratch level
  calculation. B300 feature-enabled tests passed with 36 tests and live
  cargo-oxide execution proved a two-token, ten-chunk case with one
  intermediate level, per-token stride isolation, partial final-chunk
  sentinel exclusion, and a 12,288-element scratch plan. Specialized top-k
  dispatch policy, indexed ascending sort, runtime route, and C CUDA removal
  remain unclaimed. Local formatting, diff, workspace tests, the 81-check
  comparator, and unified parity passed with 127 passed, 45 skipped, and 0
  failed. Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained the dispatch and indexed-sort non-claims.

##### M14.2d2c5: Indexed Ascending Top-K Sort And Dispatch Policy

- Status: done
- Goal: port indexed attention's ascending 512-element sort and close
  specialized top-k dispatch ordering.
- Oracle: current-C `indexed_topk_sort_512_asc_kernel`, the indexed-attention
  multi-token gate, and `ds4_gpu_indexer_topk_tensor` branch order.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c5/indexer-topk-dispatch-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_dispatch_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Evidence: added executable-local Rust ascending 512-index sort and
  validated-input top-k branch selectors, using the proven packed-key
  equivalent for capability-gated CUB positions without claiming CUB
  implementation. B300 feature-enabled tests passed with 38 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved two sorted
  rows, sort-gate behavior, packed-key-equivalent selection, fallback branch
  order, and invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Runtime route
  activation and C CUDA removal remain unclaimed. Local formatting, diff,
  workspace tests, the 81-check comparator, and unified parity passed with
  128 passed, 45 skipped, and 0 failed. Non-interactive Claude review timed
  out without a completed result; adversarial self-review added a direct
  `DS4_CUDA_NO_TOPK8192` fall-through assertion before the final B300 rerun.

##### M14.2e: M14.2 Kernel Closure Gate

- Status: done
- Goal: close the M14.2 operation-family kernel ownership ledger while
  retaining runtime route activation and C CUDA removal as later work.
- Oracle: M14.0 inventory, all M14.2 stage artifacts, and current-C
  routed-MoE ownership of `zero_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.2e/kernel-ownership-closure.json`.
- Comparator: `ds4-parity/check_m14_2_kernel_closure.py --negative-test`
  plus the retained M14.2d2c5 B300 rerun contract.
- Evidence: aggregated all fifteen M14.2 B300 proof artifacts and corrected
  the inventory so routed-MoE-only `zero_kernel` is M14.5 work. The closure
  records the packed-key top-k implementation as a semantic equivalent
  without claiming CUB ownership; default-route promotion and C CUDA removal
  remain rejected. Local formatting, diff, workspace tests, the 156-check
  closure comparator, and unified parity passed with 129 passed, 45 skipped,
  and 0 failed. A non-interactive Claude review attempt timed out without a
  completed result; self-review caught the `zero_kernel` correction.

#### M14.3: Dense Projection Quantization And Norm Kernels

- Status: done through M14.3d4
- Goal: port dense projection, Q8 conversion, and normalization kernels
  through bounded Rust CUDA slices with current C retained as the oracle.

##### M14.3a: Plain And Weighted RMS Norm Kernels

- Status: done
- Goal: prove standalone plain and weighted RMS normalization reductions
  through opt-in Rust CUDA kernels.
- Oracle: current-C `rms_norm_plain_kernel`, `rms_norm_weight_kernel`, and
  their tensor launch surfaces.
- Fixture:
  `ds4-parity/baselines/backend/m14.3a/rms-norm-kernel-smoke.json`.
- Comparator: `ds4-parity/check_rms_norm_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust kernels for plain and weighted
  per-row normalization. B300 feature-enabled tests passed with 39 tests and
  live cargo-oxide emitted portable `sm_80` PTX while proving multi-row
  plain, multi-row weighted, single-row, and invalid-shape behavior on
  `NVIDIA B300 SXM6 AC`. Fused QKV/head normalization, dense projection, Q8
  kernels, route activation, and C CUDA removal remain unclaimed. Local
  formatting, diff, workspace tests, the 72-check comparator, and unified
  parity passed with 130 passed, 45 skipped, and 0 failed. Non-interactive
  Claude review timed out without a completed result; the parity run caught
  and corrected M14.2e active-stage wiring before commit.

##### M14.3b: Fused QKV And Head RMS Norm Kernels

- Status: done through M14.3b1 and M14.3b2, which keep basic fused QKV/head
  RMS normalization distinct from head RMS plus YARN/RoPE tail math
- Goal: port fused QKV and head RMS normalization without claiming the
  remaining projection or Q8 operation family.

##### M14.3b1: Fused QKV And Basic Head RMS Norm Kernels

- Status: done
- Goal: prove fused Q/KV weighted RMS normalization and basic in-place head
  RMS normalization through opt-in Rust CUDA kernels.
- Oracle: current-C `dsv4_qkv_rms_norm_rows_kernel`,
  `head_rms_norm_kernel`, and their tensor launch surfaces.
- Fixture:
  `ds4-parity/baselines/backend/m14.3b1/fused-rms-norm-kernel-smoke.json`.
- Comparator: `ds4-parity/check_fused_rms_norm_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust fused QKV and basic head RMS kernels.
  B300 feature-enabled tests passed with 40 tests and live cargo-oxide
  emitted portable `sm_80` PTX while proving asymmetric Q/KV widths and
  in-place head normalization on `NVIDIA B300 SXM6 AC`. Head RMS plus
  RoPE-tail fusion, fused-QKV fallback policy, projection, Q8 kernels, route
  activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 73-check comparator, and
  unified parity passed with 131 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result; live
  compilation found and corrected the in-place `DisjointSlice` read through
  a mutable device pointer before the successful B300 rerun.

##### M14.3b2: Head RMS Norm Rope Tail Kernel

- Status: done
- Goal: port the combined head RMS normalization and YARN/RoPE tail kernel
  without claiming the remaining projection or Q8 operation family.
- Oracle: current-C `head_rms_norm_rope_tail_kernel`,
  `rope_yarn_ramp_dev`, and `ds4_gpu_head_rms_norm_rope_tail_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3b2/head-rms-rope-tail-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_head_rms_rope_tail_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust `head_rms_norm_rope_tail_kernel`
  with current-C per-head reduction, tail-only rotary math, YARN ramp
  mixing, and inverse sign handling. B300 feature-enabled tests passed with
  41 tests and live cargo-oxide emitted portable `sm_80` PTX through
  libdevice while proving interpolated, YARN forward, inverse, and
  invalid-shape behavior on `NVIDIA B300 SXM6 AC`. Standalone RoPE, dense
  projection, Q8 kernels, route activation, and C CUDA removal remain
  unclaimed.
  Local formatting, diff, workspace tests, the 74-check comparator, and
  unified parity passed with 132 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained standalone-RoPE, projection, Q8, and
  route non-claims.

##### M14.3c: Dense F16 And F32 Projection Kernels

- Status: split into M14.3c1 base reductions, M14.3c2 ordered, paired, and
  serial F16 kernels, and M14.3c3 BLAS dispatch and activation conversion
  before Q8 ownership is claimed
- Goal: port dense F16/F32 projection execution as a separately comparable
  slice before claiming Q8 conversion or quantized matmul ownership.

##### M14.3c1: Base F16 And F32 Projection Kernels

- Status: done
- Goal: prove direct base-reduction projection kernels with primitive F16
  weight loads and F32 weights without claiming wrapper dispatch policy.
- Oracle: current-C `matmul_f16_kernel` and `matmul_f32_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3c1/dense-projection-kernel-smoke.json`.
- Comparator: `ds4-parity/check_dense_projection_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust `matmul_f16_kernel` and
  `matmul_f32_kernel` with current-C 256-thread shared reductions and
  primitive F16 weight widening. B300 feature-enabled tests passed with 42
  tests and live cargo-oxide emitted portable `sm_80` PTX while proving F16,
  F32, multi-token stride, and invalid-shape behavior on
  `NVIDIA B300 SXM6 AC`. Ordered/paired/serial F16 variants, cuBLAS
  dispatch, Q8 kernels, route activation, and C CUDA removal remain
  unclaimed.
  Local formatting, diff, workspace tests, the 74-check comparator, and
  unified parity passed with 133 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained ordered/paired/serial, cuBLAS, Q8, and
  route non-claims.

##### M14.3c2: Ordered Paired And Serial F16 Projection Kernels

- Status: done
- Goal: port the remaining non-cuBLAS F16 projection kernels without claiming
  their environment-controlled or cuBLAS selection policy.
- Oracle: current-C `matmul_f16_serial_kernel`,
  `matmul_f16_ordered_chunks_kernel`, and
  `matmul_f16_pair_ordered_chunks_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3c2/ordered-projection-kernel-smoke.json`.
- Comparator: `ds4-parity/check_ordered_projection_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust serial, ordered-chunk, and paired
  ordered-chunk F16 kernels with primitive F16 loads and fixed 32-way
  accumulation order. B300 feature-enabled tests passed with 43 tests and
  live cargo-oxide emitted portable `sm_80` PTX while proving serial
  multi-token, ordered-chunk, unequal-width paired, and invalid-shape
  behavior on `NVIDIA B300 SXM6 AC`. Device compilation exposed unsupported
  `usize::min` lowering; replacing it with current-C's explicit upper-bound
  clamp retained the kernel contract. cuBLAS dispatch, activation conversion,
  Q8 kernels, route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 74-check comparator, and
  unified parity passed with 134 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained cuBLAS, activation-conversion, Q8, and
  route non-claims.

##### M14.3c3: F16 And F32 BLAS Dispatch And Activation Conversion

- Status: done
- Goal: port the remaining dense F16/F32 dispatch and activation-conversion
  behavior without claiming Q8 execution or default-route integration.
- Oracle: current-C `f32_to_f16_kernel`, `ds4_gpu_matmul_f16_tensor`,
  `ds4_gpu_matmul_f16_pair_tensor`, and `ds4_gpu_matmul_f32_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3c3/blas-projection-kernel-smoke.json`.
- Comparator: `ds4-parity/check_blas_projection_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: advanced `cuda-oxide` to provide mixed-precision cuBLAS and
  DS4-layout projection wrappers, added an executable-local Rust
  `f32_to_f16_kernel`, and encoded current-C F16/F32 and pair dispatch
  priorities. B300 feature-enabled tests passed with 45 tests and live
  cargo-oxide emitted portable `sm_80` PTX while proving activation
  conversion, mixed F16/F32 BLAS, F32 BLAS, dispatch priority, pair
  dispatch, and invalid-shape behavior on `NVIDIA B300 SXM6 AC`. Q8
  kernels, route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 79-check comparator, and
  unified parity passed with 135 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review corrected stale predecessor current-pin and
  successor-stage assertions before closure.

##### M14.3d1: Q8 Dequantization And Activation Quantization Kernels

- Status: done
- Goal: port `dequant_q8_0_to_f16_kernel`,
  `dequant_q8_0_to_f32_kernel`, and `quantize_q8_0_f32_kernel` without
  claiming Q8 matmul dispatch or route integration.
- Oracle: current-C packed Q8 dequantization and `lrintf`-based activation
  quantization kernels.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d1/q8-conversion-kernel-smoke.json`.
- Comparator: `ds4-parity/check_q8_conversion_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust packed Q8 F16/F32 dequantization and
  32-lane activation quantization kernels. B300 feature-enabled tests passed
  with 46 tests and live cargo-oxide emitted portable `sm_80` PTX through
  libdevice while proving ties-to-even rounding, partial-block padding, and
  invalid-shape behavior on `NVIDIA B300 SXM6 AC`. Quantized matmul, Q8
  dispatch, route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 75-check comparator, and
  unified parity passed with 136 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained quantized-matmul, Q8-dispatch, route, and
  removal non-claims.

##### M14.3d2: Base And Prequantized Q8 Matmul Kernels

- Status: done
- Goal: port base and prequantized Q8 matmul execution before claiming paired
  or HC-expansion kernels and final Q8 dispatch policy.
- Oracle: current-C `matmul_q8_0_kernel`, `matmul_q8_0_preq_kernel`,
  `matmul_q8_0_preq_warp8_kernel`, and
  `matmul_q8_0_preq_batch_warp8_kernel` scalar integer-dot behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d2/q8-matmul-kernel-smoke.json`.
- Comparator: `ds4-parity/check_q8_matmul_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust direct-quantizing, generic
  prequantized, single-token warp8, and batched warp8 Q8 matmul kernels.
  B300 feature-enabled tests passed with 47 tests and live cargo-oxide
  emitted portable `sm_80` PTX through libdevice while proving matching
  partial-block outputs and invalid-shape rejection on `NVIDIA B300 SXM6 AC`.
  DP4A acceleration, pair/HC-expansion kernels, Q8 dispatch, route
  activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 81-check comparator, and
  unified parity passed with 137 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained DP4A, pair/HC-expansion, dispatch, route,
  and removal non-claims.

##### M14.3d3: Paired And HC-Expansion Q8 Matmul Kernels

- Status: done
- Goal: port paired and HC-expansion Q8 matmul kernels before claiming DP4A
  acceleration, final Q8 dispatch policy, or runtime route ownership.
- Oracle: current-C `matmul_q8_0_pair_preq_warp8_kernel` and
  `matmul_q8_0_hc_expand_preq_warp8_kernel` behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d3/q8-specialized-matmul-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_q8_specialized_matmul_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust paired unequal-width and HC-expansion
  Q8 matmul kernels. B300 feature-enabled tests passed with 48 tests and live
  cargo-oxide emitted portable `sm_80` PTX while proving HC block output,
  optional block-add behavior, partial blocks, and invalid-shape rejection on
  `NVIDIA B300 SXM6 AC`. DP4A acceleration, Q8 dispatch, route activation,
  and C CUDA removal remain unclaimed.
  Local formatting, diff, workspace tests, the 71-check comparator, and
  unified parity passed with 138 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review retained DP4A, dispatch, route, and removal
  non-claims.

##### M14.3d4: Q8 DP4A Acceleration And Dispatch Policy

- Status: done
- Goal: add the remaining DP4A accelerated integer-dot path and Q8 dispatch
  policy before any runtime route or C-removal claim.
- Oracle: current-C `dot_i8x32_dp4a`, `dot_i8_block`,
  `cuda_q8_use_dp4a`, and `cuda_matmul_q8_0_tensor_labeled` path order.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d4/q8-dp4a-dispatch-smoke.json`.
- Comparator: `ds4-parity/check_q8_dp4a_dispatch_smoke.py --negative-test`
  plus live B300 cargo-oxide execution and PTX assembly validation.
- Evidence: added cuda-oxide signed `dp4a_i8` support with LLVM-18-compatible
  `dp4a.s32.s32` inline PTX lowering and an executable-local Rust accelerated
  Q8 matmul proof. Added current-C-compatible `select_q8_matmul_path` and
  `q8_dp4a_enabled` policy. B300 feature-enabled tests passed with 50 tests;
  live cargo-oxide emitted portable `sm_80` PTX, `ptxas` accepted its DP4A
  instruction, and output matched for both full accelerated blocks and
  partial scalar tails on `NVIDIA B300 SXM6 AC`. Runtime graph integration,
  route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the 64-check comparator, retained
  M14 checks, and unified parity passed with 134 passed, 50 skipped, and
  0 failed.

#### M14.4: RoPE KV Compressor And Attention Kernels

- Status: done; M14.5 is active
- Goal: port the current-C RoPE, KV quantization/storage, compressor, and
  attention operation family through bounded Rust CUDA slices.

##### M14.4a: Standalone RoPE Tail And FP8 KV Quantization Kernels

- Status: done
- Goal: port standalone `rope_tail_kernel` and `fp8_kv_quantize_kernel`
  behavior before claiming KV storage, compressor, or attention execution.
- Oracle: current-C `rope_tail_kernel`, `fp8_kv_quantize_kernel`,
  `ds4_gpu_rope_tail_tensor`, and `ds4_gpu_dsv4_fp8_kv_quantize_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4a/rope-kv-quantization-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_rope_kv_quantization_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust standalone RoPE-tail and FP8 KV
  quantization kernels with position stride, YARN inverse rotation, E4M3FN
  non-RoPE-prefix round trip, partial 64-wide chunk handling, and preserved
  RoPE tail. On B300 pod `ds4-rust-port-b300`, feature-enabled tests passed
  with 51 tests and live cargo-oxide execution emitted portable `sm_80` PTX
  with libdevice linkage on `NVIDIA B300 SXM6 AC`. KV storage, compressor,
  attention, runtime route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the 66-check comparator, retained
  M14 checks, and unified parity passed with 135 passed, 50 skipped, and
  0 failed.

##### M14.4b: Raw KV Storage And Indexer QAT Kernels

- Status: done
- Goal: port `store_raw_kv_batch_kernel` and `indexer_hadamard_fp4_kernel`
  behavior before claiming compressor or attention execution.
- Oracle: current-C `store_raw_kv_batch_kernel`,
  `indexer_hadamard_fp4_kernel`, `ds4_gpu_store_raw_kv_tensor`,
  `ds4_gpu_store_raw_kv_batch_tensor`, and `ds4_gpu_dsv4_indexer_qat_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4b/raw-kv-indexer-qat-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_raw_kv_indexer_qat_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust direct raw-KV ring storage with
  current-C FP16 round trip and indexer QAT with 128-wide Hadamard plus
  four-block E2M1FN activation simulation. B300 feature-enabled tests passed
  with 52 tests and live cargo-oxide execution emitted portable `sm_80` PTX
  with libdevice linkage on `NVIDIA B300 SXM6 AC`. Ring wrap is proved only
  across distinct destination rows; same-launch overlapping row writes,
  composed FP8 storage, compressor, attention, runtime route activation, and
  C CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the 68-check comparator, retained
  M14 checks, and unified parity passed with 136 passed, 50 skipped, and
  0 failed.

##### M14.4c: Composed KV Storage And Compressor Kernels

- Status: done through M14.4c3b
- Goal: port composed KV storage and bounded compressor kernels before
  claiming attention execution.

###### M14.4c1: Composed FP8 Raw Storage And Compressor Row Stores

- Status: done
- Goal: port `ds4_gpu_kv_fp8_store_raw_tensor`, `compressor_store_kernel`,
  and `compressor_set_rows_kernel` before claiming compressor pooling.
- Oracle: current-C `ds4_gpu_kv_fp8_store_raw_tensor`,
  `compressor_store_kernel`, `compressor_set_rows_kernel`, and
  `model_scalar_dev`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4c1/composed-kv-compressor-store-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_composed_kv_compressor_store_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust composed FP8 quantization plus
  raw-store execution and ratio-4 compressor store/set-row execution. B300
  feature-enabled tests passed with 53 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX with libdevice linkage and matched composed
  storage, ratio-4 row geometry, and F32/F16 APE outputs on
  `NVIDIA B300 SXM6 AC`. Compressor pooling/shift, wrapper orchestration,
  normalization/RoPE composition, attention, runtime route activation, and C
  CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the 67-check comparator, retained
  M14 checks, and unified parity passed with 137 passed, 50 skipped, and
  0 failed.

###### M14.4c2: Compressor Pooling And Ratio-4 Shift Kernels

- Status: done
- Goal: port `compressor_prefill_pool_kernel`,
  `compressor_update_pool_kernel`, and `compressor_shift_ratio4_kernel`
  before claiming update/prefill wrapper orchestration.
- Oracle: current-C `compressor_prefill_pool_kernel`,
  `compressor_update_pool_kernel`, and `compressor_shift_ratio4_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4c2/compressor-pool-shift-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_compressor_pool_shift_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust compressor prefill-pool, update-pool,
  and ratio-4 shift execution with F16 APE input coverage. B300
  feature-enabled tests passed with 54 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX with libdevice linkage and matched
  general-ratio, ratio-4/replay, update-pool, and state-shift outputs on
  `NVIDIA B300 SXM6 AC`. Update/prefill wrapper orchestration,
  normalization/RoPE/FP8 composition, attention, runtime route activation,
  and C CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the 68-check comparator, retained
  M14 checks, and unified parity passed with 138 passed, 50 skipped, and
  0 failed.

###### M14.4c3a: Compressor Update Orchestration

- Status: done
- Goal: compose owned compressor row storage, update pooling, weighted RMS
  normalization, RoPE, and ratio-4 shift into the current-C update surface
  before claiming prefill/replay orchestration.
- Oracle: current-C `ds4_gpu_compressor_update_tensor`,
  `compressor_store_kernel`, `compressor_update_pool_kernel`,
  `rms_norm_weight_kernel`, `rope_tail_kernel`, and
  `compressor_shift_ratio4_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4c3a/compressor-update-orchestration-smoke.json`.
- Comparator:
  `ds4-parity/check_compressor_update_orchestration_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust compressor update orchestration with
  a nonzero compressed-row offset, F16 APE, weighted RMS normalization, YARN
  RoPE, and ratio-4 post-emission state shift coverage. B300 feature-enabled
  tests passed with 55 tests; live cargo-oxide execution emitted portable
  `sm_80` PTX with libdevice linkage and matched non-emitting, ratio-4
  emitting, and general-ratio emitting outputs on `NVIDIA B300 SXM6 AC`.
  Prefill/replay orchestration, FP8 compressed-cache composition, attention,
  runtime route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the c3a comparator, retained M14
  checks, and unified parity passed with 139 passed, 50 skipped, and
  0 failed.

###### M14.4c3b: Compressor Prefill And Replay Orchestration

- Status: done
- Goal: compose owned compressor kernels with normalization, RoPE, and
  optional FP8 operations into the current-C prefill and replay surfaces
  before claiming attention execution.
- Oracle: current-C `ds4_gpu_compressor_prefill_tensor`,
  `ds4_gpu_compressor_prefill_ratio4_replay_tensor`,
  `ds4_gpu_compressor_prefill_state_ratio4_tensor`,
  `compressor_set_rows_kernel`, `compressor_prefill_pool_kernel`,
  `rms_norm_weight_kernel`, `rope_tail_kernel`, and
  `fp8_kv_quantize_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4c3b/compressor-prefill-orchestration-smoke.json`.
- Comparator:
  `ds4-parity/check_compressor_prefill_orchestration_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust state initialization, set-row
  placement, pooled compressed output, weighted RMS, YARN RoPE, and optional
  FP8 processing across general prefill, ratio-4 prefill/replay, and ratio-4
  state-only cases. B300 feature-enabled tests passed with 56 tests; live
  cargo-oxide execution emitted portable `sm_80` PTX with libdevice linkage
  and matched state placement, replay ordering, and optional FP8 outputs on
  `NVIDIA B300 SXM6 AC`. Attention, runtime route activation, and C CUDA
  removal remain unclaimed. Local formatting, diff, library tests, the c3b
  comparator, retained M14 checks, and unified parity passed with 140 passed,
  50 skipped, and 0 failed.

##### M14.4d: Attention Kernels

- Status: done through M14.4d8b
- Goal: port current-C attention decode, prefill, indexed, and output-Q8
  device behavior after compressor surfaces are proved.

###### M14.4d1: Single-Token Mixed Attention Decode Surface

- Status: done
- Goal: port the exported single-token mixed attention decode behavior before
  batched/window/heads8, prefill/indexed, or output-Q8 ownership.
- Oracle: current-C `ds4_gpu_attention_decode_heads_tensor` and the
  `single_all` branch of `attention_decode_mixed_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d1/attention-decode-single-mixed-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_decode_single_mixed_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust single-token mixed-attention
  execution using the current-C visible-row and sink-softmax semantics. B300
  feature-enabled tests passed with 57 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX with libdevice linkage and matched
  masked/unmasked compressed rows, wrapped raw rows, sink softmax, and
  raw-only outputs on `NVIDIA B300 SXM6 AC`. Batched/window/heads8 decode,
  prefill/indexed/output-Q8 attention, runtime route activation, and C CUDA
  removal remain unclaimed. Local formatting, diff, library tests, the d1
  comparator, retained M14 checks, and unified parity passed with 141 passed,
  50 skipped, and 0 failed.

###### M14.4d2: Generic Batched Mixed Attention Decode Surfaces

- Status: done
- Goal: port the generic batched raw and mixed decode surfaces before claiming
  optimized heads8-online, prefill, indexed, or output-Q8 attention.
- Oracle: current-C `attention_decode_batch_launch`,
  `ds4_gpu_attention_decode_raw_batch_heads_tensor`,
  `ds4_gpu_attention_decode_mixed_batch_heads_tensor`, and the generic
  `attention_decode_mixed_kernel` launch.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d2/attention-decode-batch-mixed-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_decode_batch_mixed_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust generic batched mixed-attention
  execution using the current-C causal-window, compressed-visibility, and
  sink-softmax semantics. B300 feature-enabled tests passed with 58 tests;
  live cargo-oxide execution emitted portable `sm_80` PTX with libdevice
  linkage and matched mixed/raw batched output, wrapped raw rows, per-token
  compressed masks, and learned sink contribution on `NVIDIA B300 SXM6 AC`.
  Heads8-online dispatch, prefill/indexed/output-Q8 attention, runtime route
  activation, and C CUDA removal remain unclaimed. Local formatting, diff,
  library tests, the 63-check d2 comparator, retained M14 checks, and unified
  parity passed with 142 passed, 50 skipped, and 0 failed.

###### M14.4d3: Heads8 Online Attention Decode Kernels

- Status: done
- Goal: port optimized heads8-online decode and its dispatch selection before
  prefill, indexed, or output-Q8 attention ownership.
- Oracle: current-C `attention_decode_mixed_heads8_online_kernel`,
  `attention_decode_batch_launch`, `cuda_attention_score_buffer_fits`, and
  decode window-attention environment/quality dispatch conditions.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d3/attention-decode-heads8-online-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_decode_heads8_online_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust grouped-head online decode execution
  with partial final head group, batched causal-window/ring-row coverage,
  single-token all-compressed coverage, and `select_attention_decode_path`
  dispatch-policy matching. B300 feature-enabled tests passed with 60 tests;
  live cargo-oxide execution emitted portable `sm_80` PTX with libdevice
  linkage and matched online decode output on `NVIDIA B300 SXM6 AC`.
  Raw/mixed prefill, indexed, output-Q8 attention, runtime route activation,
  and C CUDA removal remain unclaimed. Local formatting, diff, library tests,
  the 64-check d3 comparator, retained M14 checks, and unified parity passed
  with 143 passed, 50 skipped, and 0 failed.

###### M14.4d4: Generic Raw And Mixed Attention Prefill Kernels

- Status: done
- Goal: port generic raw and mixed prefill kernels before claiming optimized
  static-online/CUBLAS prefill, indexed, or output-Q8 attention.
- Oracle: current-C `attention_prefill_raw_kernel`,
  `attention_prefill_mixed_kernel`, `ds4_gpu_attention_prefill_raw_heads_tensor`,
  `attention_prefill_mixed_launch`,
  `ds4_gpu_attention_prefill_static_mixed_heads_tensor`, and
  `ds4_gpu_attention_prefill_masked_mixed_heads_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d4/attention-prefill-generic-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_prefill_generic_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust generic raw and mixed prefill kernels
  with static and masked compressed-row cases. B300 feature-enabled tests
  passed with 61 tests; live cargo-oxide execution emitted portable `sm_80`
  PTX with libdevice linkage and matched generic raw/static/masked mixed
  output on `NVIDIA B300 SXM6 AC`. Static heads8-online/CUBLAS prefill
  dispatch, indexed/output-Q8 attention, runtime route activation, and C CUDA
  removal remain unclaimed. Local formatting, diff, library tests, the
  68-check d4 comparator, retained attention checks, and unified parity
  passed with 149 passed, 45 skipped, and 0 failed.

###### M14.4d5: Static Heads8 Online And CUBLAS Attention Prefill Dispatch

- Status: done
- Goal: port optimized static heads8-online and CUBLAS attention prefill
  dispatch before claiming indexed or output-Q8 attention.
- Oracle: current-C `attention_static_mixed_heads8_online_kernel`,
  `attention_prefill_raw_softmax_kernel`, `attention_prefill_mixed_softmax_kernel`,
  `attention_prefill_pack_mixed_kv_kernel`,
  `attention_prefill_unpack_heads_kernel`,
  `ds4_gpu_attention_prefill_raw_heads_tensor`, and
  `attention_prefill_mixed_launch`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d5/attention-prefill-optimized-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_prefill_optimized_smoke.py --negative-test`
  plus live B300 cargo-oxide and cuBLAS execution.
- Evidence: added executable-local grouped-head online prefill,
  current-C-equivalent softmax/pack/unpack kernels, row-major cuBLAS adapter
  kernels, and dispatch-policy matching. B300 feature-enabled tests passed
  with 63 tests; live cargo-oxide execution emitted portable `sm_80` PTX
  with libdevice linkage and matched static online, raw cuBLAS, and masked
  mixed cuBLAS outputs on `NVIDIA B300 SXM6 AC`. Indexed/output-Q8 attention,
  runtime route activation, and C CUDA removal remain unclaimed. Local
  formatting, diff, library tests, the 70-check d5 comparator, retained
  attention checks, and unified parity passed with 150 passed, 45 skipped,
  and 0 failed.

###### M14.4d6: Generic Indexed Mixed Attention Surface

- Status: done
- Goal: port the generic indexed mixed-attention surface before optimized
  indexed heads8/sort dispatch or output-Q8 attention ownership.
- Oracle: current-C `attention_indexed_mixed_kernel` and
  `ds4_gpu_attention_indexed_mixed_batch_heads_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d6/attention-indexed-generic-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_indexed_generic_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust generic indexed mixed attention
  execution covering ordered/duplicate top-k rows, invalid/out-of-visible
  filtering, `ratio == 0` all-compressed visibility, causal raw windows,
  raw-ring wrapping, and learned-sink softmax. B300 feature-enabled tests
  passed with 64 tests; live cargo-oxide execution emitted portable `sm_80`
  PTX with libdevice linkage and matched output on `NVIDIA B300 SXM6 AC`.
  Indexed sort/heads8 dispatch, output-Q8 attention, runtime route activation,
  and C CUDA removal remain unclaimed. Local formatting, diff, library tests,
  the 64-check d6 comparator, retained attention checks, and unified parity
  passed with 146 passed, 50 skipped, and 0 failed.

###### M14.4d7: Optimized Indexed Sort And Heads8 Attention Kernels

- Status: done
- Goal: port indexed top-k sorting and optimized heads8 dispatch before
  output-Q8 attention ownership.
- Oracle: current-C `indexed_topk_sort_512_asc_kernel`,
  `attention_indexed_mixed_heads8_online_kernel`,
  `attention_indexed_mixed_heads8_rb4_kernel`, and
  `ds4_gpu_attention_indexed_mixed_batch_heads_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d7/attention-indexed-optimized-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_indexed_optimized_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local sorted-topk-to-online indexed attention
  execution, rb4 filtered/duplicate indexed execution, and current-C branch
  policy. B300 feature-enabled tests passed with 66 tests; live cargo-oxide
  execution emitted portable `sm_80` PTX with libdevice linkage and matched
  both optimized indexed outputs on `NVIDIA B300 SXM6 AC`. Output-Q8
  attention, runtime route activation, and C CUDA removal remain unclaimed.
  Local formatting, diff, library tests, the 68-check d7 comparator, retained
  attention checks, and unified parity passed with 147 passed, 50 skipped,
  and 0 failed.

###### M14.4d8: Output Q8 Attention Projection Surfaces

- Status: done after splitting into M14.4d8a and M14.4d8b
- Goal: port native Q8 output surfaces before the optional F16/cuBLAS A
  projection optimization.

####### M14.4d8a: Native Q8 Attention Output Projection Surfaces

- Status: done
- Goal: port native-Q8 low-output and batched two-stage output projection
  behavior before claiming the optional cuBLAS A path.
- Oracle: current-C `quantize_q8_0_f32_kernel`,
  `grouped_q8_0_a_preq_warp8_kernel`,
  `matmul_q8_0_preq_batch_warp8_kernel`,
  `ds4_gpu_attention_output_low_q8_tensor`, and
  `ds4_gpu_attention_output_q8_batch_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d8a/attention-output-q8-native-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_output_q8_native_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Evidence: added executable-local Rust grouped A and B output projection
  orchestration over Q8 prequantized inputs, covering single low output,
  batched low/output, and partial blocks. B300 feature-enabled tests passed
  with 67 tests; live cargo-oxide execution emitted portable `sm_80` PTX
  with libdevice linkage and matched native Q8 output on
  `NVIDIA B300 SXM6 AC`. F16/cuBLAS A dispatch, runtime route activation,
  and C CUDA removal remain unclaimed. Local formatting, diff, library tests,
  the 63-check d8a comparator, retained attention checks, and unified parity
  passed with 148 passed, 50 skipped, and 0 failed.

####### M14.4d8b: CUBLAS Attention Output A Dispatch

- Status: done
- Goal: port the optional F16/cuBLAS attention-output-A acceleration and
  branch policy before attention-family closure.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d8b/attention-output-q8-cublas-smoke.json`
- Comparator:
  `ds4-parity/check_attention_output_q8_cublas_smoke.py --negative-test`
- Evidence: executable-local Rust F16 grouped-head packing and low-output
  unpacking are composed with live cuda-oxide SGEMM over F16-rounded inputs
  for the current-C optional cuBLAS A branch. B300 feature-enabled tests
  passed with 69 tests; live cargo-oxide execution emitted portable `sm_80`
  PTX with libdevice linkage and matched packed, grouped, and unpacked output
  on `NVIDIA B300 SXM6 AC`. The safe cuda-oxide API exposes SGEMM rather than
  current-C's `CUDA_R_16F` GemmEx entry point; runtime route activation and C
  CUDA removal remain unclaimed. Local formatting, diff, library tests, the
  66-check d8b comparator, retained attention checks, and unified parity
  passed with 149 passed, 50 skipped, and 0 failed.

#### M14.5: Router MoE And Hyperconnection Kernels

- Status: done through M14.5d
- Goal: port the remaining current-C router, routed-MoE, shared-expert, and
  hyperconnection CUDA surfaces after attention-family closure.

##### M14.5a: Scalar Router Selection Surfaces

- Status: done
- Goal: port scalar single-token and batched router selection before optimized
  dispatch and routed expert execution.
- Fixture: `ds4-parity/baselines/backend/m14.5a/router-scalar-smoke.json`
- Comparator: `ds4-parity/check_router_scalar_smoke.py --negative-test`
- Evidence: executable-local Rust scalar router selection covers current-C
  softplus probabilities, bias-ranked top-6 output, hash routing with
  invalid-token fallback, normalized selected weights, and single/batched
  layout. B300 feature-enabled tests passed with 70 tests; live cargo-oxide
  execution emitted portable `sm_80` PTX through libdevice and matched scalar
  router outputs on `NVIDIA B300 SXM6 AC`. Parallel/warp dispatch, routed
  MoE, hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the 63-check M14.5a
  comparator, retained M14 checks, and unified parity passed with 150 passed,
  50 skipped, and 0 failed.

##### M14.5b: Parallel And Warp Router Dispatch

- Status: done
- Goal: port optimized parallel/warp router-selection kernels and current-C
  dispatch priority before routed MoE execution.
- Fixture: `ds4-parity/baselines/backend/m14.5b/router-optimized-smoke.json`
- Comparator: `ds4-parity/check_router_optimized_smoke.py --negative-test`
- Evidence: executable-local Rust parallel shared-memory and warp-shuffle
  router kernels cover optimized selection, deterministic tie ordering,
  partial four-row blocks, hash fallback, and current-C optimized dispatch
  priority. B300 feature-enabled tests passed with 72 tests; live cargo-oxide
  execution emitted portable `sm_80` PTX through libdevice and matched
  optimized router outputs on `NVIDIA B300 SXM6 AC`. Routed MoE,
  hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the 70-check M14.5b
  comparator, retained M14 checks, and unified parity passed with 151 passed,
  50 skipped, and 0 failed.

##### M14.5c1: Packed F32-Activation Routed MoE Fallback

- Status: done
- Goal: port the current-C packed IQ2-XXS gate/up and Q2_K down
  F32-activation fallback before optimized routed dispatch.
- Fixture: `ds4-parity/baselines/backend/m14.5c1/routed-moe-f32-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_f32_smoke.py --negative-test`
- Evidence: executable-local Rust routed-MoE fallback kernels cover packed
  IQ2-XXS gate/up decode, packed Q2_K down decode, weighted SwiGLU/clamp,
  negative-expert fallback, expert summation, and single/batched layout.
  B300 feature-enabled tests passed with 73 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX through libdevice and matched routed fallback
  outputs on `NVIDIA B300 SXM6 AC`. Q8 activation/optimized dispatch, Q4_K,
  hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the M14.5c1 comparator,
  retained M14 checks, and unified parity passed with 152 passed, 50 skipped,
  and 0 failed.

##### M14.5c2a: Default Single-Token Quantized Routed MoE Dispatch

- Status: done
- Goal: port the current-C default single-token IQ2-XXS/Q2_K quantized
  routed-MoE compute path after packed fallback closure.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2a/routed-moe-quantized-single-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_quantized_single_smoke.py --negative-test`
- Evidence: executable-local Rust quantized routed-MoE kernels cover Q8_K
  input/intermediate activation fields, LUT-equivalent IQ2-XXS/Q8_K gate/up
  decode, Q2_K/Q8_K direct six-expert down output, optional auxiliary writes,
  zero quantization, and negative-expert fallback. B300 feature-enabled tests
  passed with 74 tests; live cargo-oxide execution emitted portable `sm_80`
  PTX through libdevice and matched default single-token results on
  `NVIDIA B300 SXM6 AC`. Batched sorted/tiled dispatch, Q4_K,
  hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the M14.5c2a comparator,
  retained M14 checks, and unified parity passed with 153 passed, 50 skipped,
  and 0 failed.

##### M14.5c2b1: Batched Sorted-Pair Metadata

- Status: done
- Goal: port the current-C pair histogram, prefix-offset, and scatter
  metadata kernels before sorted quantized projection.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2b1/routed-moe-sorted-pairs-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_sorted_pairs_smoke.py --negative-test`
- Evidence: executable-local Rust device-atomic metadata kernels cover
  expert histogram, prefix offsets/cursors, grouped scatter, duplicate pair
  preservation, and negative-expert expert-zero bucketing without assuming
  ordering within an equal-expert atomic region. B300 feature-enabled tests
  passed with 75 tests; live cargo-oxide execution emitted portable `sm_80`
  PTX and matched metadata behavior on `NVIDIA B300 SXM6 AC`. Sorted
  projection, expert-tile/atomic-down execution, Q4_K, hyperconnection,
  runtime route activation, and C CUDA removal remain unclaimed. Local
  formatting, diff, library tests, the M14.5c2b1 comparator, retained M14
  checks, and unified parity passed with 154 passed, 50 skipped, and 0 failed.

##### M14.5c2b2: Sorted-Pair P2 Quantized Projection

- Status: done
- Goal: port no-expert-tiles/default-P2 batched IQ2-XXS/Q2_K gate/down
  projection over sorted metadata before expert-tile and atomic-down variants.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2b2/routed-moe-sorted-p2-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_sorted_p2_smoke.py --negative-test`
- Evidence: executable-local Rust sorted P2 kernels compose atomic sorted-pair
  metadata, Q8_K quantization, IQ2-XXS/Q8_K gate/up projection, Q2_K/Q8_K
  down projection, and token summation for two-token batched input. B300
  feature-enabled tests passed with 76 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX through libdevice and matched sorted P2 output
  behavior on `NVIDIA B300 SXM6 AC`. Expert-tile or atomic-down scheduling,
  Q4_K, hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the M14.5c2b2 comparator,
  retained M14 checks, and unified parity passed with 155 passed, 50 skipped,
  and 0 failed.

##### M14.5c2c1: Expert-Tile Descriptor Metadata

- Status: done
- Goal: port current-C expert-tile offset and descriptor construction before
  tile-local gate/down projection.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c1/routed-moe-expert-tiles-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_expert_tiles_smoke.py --negative-test`
- Evidence: executable-local Rust tile offset and descriptor kernels cover
  default eight-pair and alternate four-pair expert grouping, partial final
  tiles, and negative-expert bucket-zero counts. B300 feature-enabled tests
  passed with 77 tests; live cargo-oxide execution emitted portable `sm_80`
  PTX and matched descriptor behavior on `NVIDIA B300 SXM6 AC`. Tile-local
  projection, atomic-down/rowspan execution, Q4_K, hyperconnection, runtime
  route activation, and C CUDA removal remain unclaimed. Local formatting,
  diff, library tests, the M14.5c2c1 comparator, retained M14 checks, and
  unified parity passed with 156 passed, 50 skipped, and 0 failed.

##### M14.5c2c2: Default Tile8 Row32 Projection

- Status: done
- Goal: port current-C default eight-pair row32 expert-tile gate/down
  projection before atomic-down and wider-row scheduling variants.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c2/routed-moe-tile8-row32-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_tile8_row32_smoke.py --negative-test`
- Evidence: executable-local Rust functional tile8 row32 kernels cover
  IQ2-XXS/Q8_K gate/up and Q2_K/Q8_K non-atomic down projection over
  expert-tile descriptors, including multi-tile and partial-tile groups.
  B300 feature-enabled tests passed with 78 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX through libdevice and matched outputs on
  `NVIDIA B300 SXM6 AC`. Shared-cache specialization, tile4,
  atomic-down/tile16/rowspan execution, Q4_K, hyperconnection, runtime route
  activation, and C CUDA removal remain unclaimed. Local formatting, diff,
  library tests, the M14.5c2c2 comparator, retained M14 checks, and unified
  parity passed with 157 passed, 50 skipped, and 0 failed.

##### M14.5c2c3: Tile4 Row32 Projection

- Status: done
- Goal: port optional four-pair row32 expert-tile gate/down projection before
  atomic-down, tile16, rowspan, and shared-cache optimization boundaries.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c3/routed-moe-tile4-row32-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_tile4_row32_smoke.py --negative-test`
- Evidence: executable-local Rust `DS4_CUDA_MOE_TILE4`-selected functional
  tile4 row32 kernels cover IQ2-XXS/Q8_K gate/up and Q2_K/Q8_K non-atomic
  down projection over expert-tile descriptors, including a three-tile
  same-expert group and partial tiles. B300 feature-enabled tests passed with
  79 tests; live cargo-oxide execution emitted portable `sm_80` PTX through
  libdevice and matched outputs on `NVIDIA B300 SXM6 AC`. Shared-cache
  specialization, atomic-down/tile16/rowspan execution, Q4_K,
  hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the M14.5c2c3
  comparator, retained M14 checks, and unified parity passed with 158 passed,
  50 skipped, and 0 failed.

##### M14.5c2c4: Atomic Expert-Tile Down Output

- Status: done
- Goal: port token-indexed atomic accumulation for expert-tile down projection
  before tile16 and widened-row scheduling variants.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c4/routed-moe-atomic-down-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_atomic_down_smoke.py --negative-test`
- Evidence: executable-local Rust `zero_kernel` and row32
  `DeviceAtomicF32::fetch_add` branches cover token-indexed atomic down
  accumulation for both tile8 and tile4 descriptor schedules. B300
  feature-enabled tests passed with 80 tests; live cargo-oxide execution
  emitted portable `sm_80` PTX and matched both atomic outputs on
  `NVIDIA B300 SXM6 AC`. Tile16/rowspan dispatch, shared-cache
  specialization, Q4_K, hyperconnection, runtime route activation, and C
  CUDA removal remain unclaimed. Local formatting, diff, library tests, the
  M14.5c2c4 comparator, retained M14 checks, and unified parity passed with
  159 passed, 50 skipped, and 0 failed.

##### M14.5c2c5: Tile16 Row32 Atomic Down

- Status: done
- Goal: port the high-token tile16 row32 atomic down projection selection
  before widened-row and shared-cache specialization boundaries.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c5/routed-moe-tile16-row32-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_tile16_row32_smoke.py --negative-test`
- Evidence: executable-local Rust tile16 row32 atomic-down projection uses
  separately built tile16 descriptors while retaining tile8 gate metadata and
  covers a partial tile16 group. B300 feature-enabled tests passed with 81
  tests; live cargo-oxide execution emitted portable `sm_80` PTX through
  libdevice and matched atomic output on `NVIDIA B300 SXM6 AC`. Gate/down
  row2048/rowspan dispatch, shared-cache specialization, Q4_K,
  hyperconnection, runtime route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, library tests, the M14.5c2c5
  comparator, retained M14 checks, and unified parity passed with 160 passed,
  50 skipped, and 0 failed.

##### M14.5c2c6: Gate Tile8 Rowspan Projection

- Status: done
- Goal: port the tile8 widened-row gate/up scheduling variants before
  widened-row atomic down and shared-cache specialization closure.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c6/routed-moe-gate-rowspan-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_gate_rowspan_smoke.py --negative-test`
- Evidence: executable-local Rust parameterized tile8 gate-rowspan projection
  covers the row512, row1024, and row2048 current-C scheduling spans over
  retained tile descriptors. B300 feature-enabled tests passed with 82 tests;
  live cargo-oxide execution found seven kernels, emitted portable `sm_80`
  PTX through libdevice, and matched all three gate outputs on
  `NVIDIA B300 SXM6 AC`. Widened down scheduling, shared-cache
  specialization, Q4_K, hyperconnection, runtime route activation, and C CUDA
  removal remain unclaimed. Local formatting, diff, library tests, the
  M14.5c2c6 comparator, retained routed-MoE checks, and unified parity passed
  with 161 passed, 50 skipped, and 0 failed.

##### M14.5c2c7: Down Tile16 Rowspan Projection

- Status: done
- Goal: port the tile16 widened-row atomic down scheduling variants after
  row32 atomic down and widened gate scheduling are established.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c7/routed-moe-down-rowspan-smoke.json`
- Comparator: `ds4-parity/check_routed_moe_down_rowspan_smoke.py --negative-test`
- Evidence: executable-local Rust parameterized tile16 atomic down-rowspan
  projection covers the row512, row1024, and row2048 current-C scheduling
  spans over retained tile16 descriptors. B300 feature-enabled tests passed
  with 83 tests; live cargo-oxide execution found eight kernels, emitted
  portable `sm_80` PTX through libdevice, lowered the device float atomic
  operation, and matched all three down outputs on `NVIDIA B300 SXM6 AC`.
  Shared-cache specialization, Q4_K, hyperconnection, runtime route
  activation, and C CUDA removal remain unclaimed. Local formatting, diff,
  library tests, the M14.5c2c7 comparator, retained routed-MoE checks, and
  unified parity passed with 162 passed, 50 skipped, and 0 failed.

##### M14.5c2d: Single-Token Q4_K Routed MoE

- Status: done
- Goal: port the current-C single-token Q4_K gate/up and direct down-sum
  branch after the IQ2/Q2 tiled routed-MoE family is functionally covered.
- Evidence:
  - Added opt-in `DS4_CUDA_MOE_Q4_K=1` kernels for packed Q4_K/Q8_K dot,
    single-token gate/up, and direct sum-six down in
    `rust/ds4-cuda/src/bin/routed_moe_quantized_single_smoke.rs`.
  - Captured B300 evidence in
    `ds4-parity/baselines/backend/m14.5c2d/routed-moe-q4-k-single-smoke.json`
    with comparator
    `ds4-parity/check_routed_moe_q4_k_single_smoke.py --negative-test`.
  - Live B300 execution emitted portable `sm_80` PTX through libdevice,
    matched Q4_K output against the Rust oracle, and passed feature tests
    with 84 tests. The local CUDA-feature build is blocked because
    `/usr/local/cuda/include/cuda.h` is absent on this Mac. Shared-cache
    specialization, hyperconnection, runtime route activation, and C CUDA
    removal remain unclaimed. Local formatting, diff and library tests, the
    M14.5c2d comparator, retained selector run, and unified parity passed with
    168 passed, 45 skipped, and 0 failed.

##### M14.5c2e: Shared-Cache Expert-Tile Projection

- Status: done
- Goal: port current-C expert-tile shared-memory staging for Q8 input blocks
  and IQ2 lookup/sign tables before beginning hyperconnection kernels.
- Evidence:
  - Added opt-in `DS4_CUDA_MOE_SHARED_CACHE=1` cached row-span gate/up and
    tile16 atomic-down kernels in
    `rust/ds4-cuda/src/bin/routed_moe_tile8_row32_smoke.rs`, with
    synchronized `SharedArray` staging of Q8 values, fixture IQ2 tables, and
    Q2 block sums.
  - Captured B300 evidence in
    `ds4-parity/baselines/backend/m14.5c2e/routed-moe-shared-cache-smoke.json`
    with comparator
    `ds4-parity/check_routed_moe_shared_cache_smoke.py --negative-test`.
  - Live B300 execution emitted portable `sm_80` PTX through libdevice,
    matched row512/row1024/row2048 gate and down outputs, and passed feature
    tests with 85 tests. Generic/sorted qwarp fallback projection,
    hyperconnection, runtime route activation, and C CUDA removal remain
    unclaimed. Local formatting, diff and library tests, the M14.5c2e
    comparator, retained selector checks, and unified parity passed with 169
    passed, 45 skipped, and 0 failed.

##### M14.5c2f: Generic And Sorted Qwarp Quantized Routed MoE

- Status: done
- Goal: port current-C quantized fallback projections selected when
  decode-LUT or sorted-P2/expert-tile scheduling is disabled.
- Evidence:
  - Added opt-in `DS4_CUDA_MOE_QWARP_FALLBACK=1` generic single-token and
    sorted no-P2 kernels in
    `rust/ds4-cuda/src/bin/routed_moe_sorted_p2_smoke.rs`.
  - Captured B300 evidence in
    `ds4-parity/baselines/backend/m14.5c2f/routed-moe-qwarp-fallback-smoke.json`
    with comparator
    `ds4-parity/check_routed_moe_qwarp_fallback_smoke.py --negative-test`.
  - Live B300 execution emitted portable `sm_80` PTX through libdevice,
    matched generic and sorted qwarp gate/down/summed output, and passed
    feature tests with 86 tests. The local CUDA-feature build is blocked
    because `/usr/local/cuda/include/cuda.h` is absent on this Mac.
    Hyperconnection, runtime route activation, and C CUDA removal remain
    unclaimed. Local formatting, diff and library tests, the M14.5c2f
    comparator, retained selector checks, and unified parity passed with 170
    passed, 45 skipped, and 0 failed.

##### M14.5d: Hyperconnection Split And Expansion Kernels

- Status: done
- Goal: port current-C hyperconnection split, weighted-sum, expansion, and
  output-weight kernel surfaces after routed-MoE compute closure.
- Fixture: `ds4-parity/baselines/backend/m14.5d/hyperconnection-smoke.json`
- Comparator: `ds4-parity/check_hyperconnection_smoke.py --negative-test`.
- Evidence: added executable-local Rust hyperconnection split, weighted-sum,
  expansion, fused split/reduction, fused normalization, and output-weight
  kernels. Live B300 cargo-oxide execution found six kernels and eight device
  functions, emitted portable `sm_80` PTX through libdevice, linked a
  `239188`-byte LTOIR container, and matched deterministic split, direct and
  split-stride weighted sum, add/plain expansion, fused normalized output,
  and output HC-weight results on `NVIDIA B300 SXM6 AC`. B300 feature tests
  passed with 87 tests. The local CUDA-feature build is blocked because
  `/usr/local/cuda/include/cuda.h` is absent on this Mac. M14.5 operation
  families are complete on the opt-in Rust path; runtime route activation and
  C CUDA removal remain unclaimed. Local formatting, diff and library tests,
  the 77-check M14.5d comparator, retained M14.5 comparators, and unified
  parity passed with 171 passed, 45 skipped, and 0 failed.

#### M14.6: CUDA Route Promotion And C CUDA Removal Gate

- Status: active; split into M14.6a and M14.6b after production linkage
  inspection
- Goal: determine whether all validated Rust CUDA operation families can be
  integrated into the default runtime route and whether `ds4_cuda.cu`
  linkage can be removed.
- Oracle: the M14.0 inventory, all M14 validation artifacts, and retained
  current-C end-to-end/quality/benchmark gates.
- Acceptance: only promote the default route or remove C CUDA after exported
  API coverage and same-B300 end-to-end comparisons pass; otherwise record
  the precise blocker and retain the C route.

##### M14.6a: Production Route Linkage Blocker

- Status: done
- Goal: determine whether the validated cuda-oxide operations are already
  reachable through the production Rust runtime route.
- Fixture:
  `ds4-parity/baselines/backend/m14.6a/production-route-blocker.json`
- Comparator: `ds4-parity/check_cuda_route_promotion_gate.py --negative-test`.
- Evidence: Linux `rust/ds4-gpu/build.rs` still compiles and archives
  `ds4_cuda.cu`; `rust/ds4-gpu` does not depend on `rust/ds4-cuda`;
  `rust/ds4-cuda` exposes executable-local CUDA proof modules rather than a
  linkable `ds4_gpu_*` implementation; and `rust/ds4-engine` still rejects
  `--runtime-graph graph`. Default-route promotion and C CUDA removal are
  therefore blocked rather than overclaimed. Validation: 86 local
  `ds4-cuda` library tests passed, 88 B300 feature tests passed, the
  production-linkage checker passed with 56 checks, and the unified parity
  report passed with 172 passes, 45 skips, and no failures.

##### M14.6b: Rust CUDA ABI Backend Assembly

- Status: active; split into M14.6b1 and M14.6b2
- Goal: assemble the validated cuda-oxide operations behind the production
  `ds4_gpu_*` ABI and use that linkable backend for same-B300 route
  comparison before any C CUDA removal.

##### M14.6b1: Rust CUDA Resource ABI Exports

- Status: done
- Goal: export the resource and command subset of `ds4_gpu_*` from a
  linkable Rust cuda-oxide static library.
- Fixture: `ds4-parity/baselines/backend/m14.6b1/abi-resource-smoke.json`
- Comparator: `ds4-parity/check_cuda_abi_resource_smoke.py --negative-test`.
- Evidence: `rust/ds4-cuda/src/abi.rs` exports 16 initialization, tensor,
  transfer, synchronization, and managed-KV symbols. B300 execution on
  `NVIDIA B300 SXM6 AC` passed the resource ABI smoke and static-library
  symbol inspection; 87 local library tests and 89 B300 backend-feature
  tests passed; the resource ABI checker passed with 77 checks; and the
  unified parity report passed with 173 passes, 45 skips, and no failures.
  Compute ABI ownership and production linker promotion remain explicitly
  false.

##### M14.6b2: Rust CUDA Compute ABI Assembly

- Status: active; split into M14.6b2a and M14.6b2b
- Goal: consolidate validated kernel families into reusable modules and
  export the compute ABI before any production linker or route promotion.

##### M14.6b2a: Rust CUDA Tensor Fill ABI Export

- Status: done
- Goal: export `ds4_gpu_tensor_fill_f32` from the linkable Rust static
  library without requiring embedded kernel retention in a downstream binary.
- Fixture: `ds4-parity/baselines/backend/m14.6b2a/abi-tensor-fill-smoke.json`
- Comparator: `ds4-parity/check_cuda_abi_tensor_fill_smoke.py --negative-test`.
- Evidence: `rust/ds4-cuda/src/abi.rs` now implements the first compute ABI
  symbol through stream-ordered `cuda_core::sys::cuMemsetD32Async` using the
  supplied float's exact bits. B300 execution on `NVIDIA B300 SXM6 AC`
  passed prefix, tensor-view, managed, signed-zero, negative-infinity,
  zero-count, bounds, and null-input checks; `nm` confirmed 17 static-library
  symbols; 88 local library tests and 90 B300 backend-feature tests passed;
  the tensor-fill ABI checker passed with 91 checks; and the unified parity
  report passed with 174 passes, 45 skips, and no failures. Local CUDA-feature
  compilation remains unavailable because `/usr/local/cuda/include/cuda.h`
  is absent. Graph compute ABI ownership, production linker promotion, and
  default-route promotion remain explicitly false. The required
  non-interactive Claude review was invoked with the changed-file, oracle,
  comparator, and validation evidence bundle, but timed out after 60 seconds
  without a completed result; adversarial self-review added managed-tensor
  coverage and documented the raw-driver write-bounds invariant.

##### M14.6b2b: Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b1 and M14.6b2b2
- Goal: export the remaining validated graph compute symbols from reusable
  Rust-owned modules before any production linker or route promotion.

##### M14.6b2b1: Rust CUDA Elementwise ABI Module

- Status: done
- Goal: export `ds4_gpu_add_tensor` and `ds4_gpu_repeat_hc_tensor` through
  reusable embedded Rust CUDA kernels while preserving current-C in-place add
  behavior through the C ABI.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b1/abi-elementwise-smoke.json`
- Comparator: `ds4-parity/check_cuda_abi_elementwise_smoke.py --negative-test`.
- Evidence: `rust/ds4-cuda/src/abi_kernels.rs` defines library-owned
  `abi_add_kernel` and `abi_repeat_hc_kernel` names disjoint from retained
  executable smokes, and `abi.rs` exports the two new ABI
  symbols using raw launch parameters so `out == a` remains valid. A C
  executable linked from `libds4_cuda.a` on `NVIDIA B300 SXM6 AC` passes
  add, aliasing, repeat, invalid-shape, and null checks; `nm` confirms 19
  exported `ds4_gpu_*` symbols. The embedded artifact currently requires
  `--whole-archive` retention and produces a missing `.note.GNU-stack`
  linker warning. Validation: 89 local library tests and 91 B300 kernel-
  feature tests pass; the elementwise ABI checker and unified parity report
  pass with 175 passed, 45 skipped, and no failures. Remaining graph compute
  ABI and production-route promotion are not claimed. The required
  non-interactive Claude review timed out after 60 seconds without a completed
  result; adversarial self-review fixed the embedded module-name mismatch and
  duplicate kernel-symbol collision while retaining raw alias-preserving
  launches.

##### M14.6b2b2: Remaining Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b2a and M14.6b2b2b
- Goal: export the remaining graph compute ABI symbols and resolve embedded
  artifact production-link integration before selecting a Rust CUDA route.

##### M14.6b2b2a: Directional Steering ABI Export

- Status: done
- Goal: export `ds4_gpu_directional_steering_project_tensor` through the
  reusable embedded Rust CUDA ABI module with current-C in-place behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2a/abi-directional-steering-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_directional_steering_smoke.py --negative-test`.
- Evidence: `abi_directional_steering_project_kernel` is uniquely named in
  `rust/ds4-cuda/src/abi_kernels.rs`, and the ABI wrapper validates shape
  bounds before raw in-place launch. A C-linked static-library smoke on
  `NVIDIA B300 SXM6 AC` passes exact projection output, zero-scale,
  undersized-direction, and null-input checks; `nm` confirms 20 exports.
  Validation includes 90 local library tests and 92 B300 kernel-feature tests.
  Whole-archive retention and the generated embedded object's
  `.note.GNU-stack` warning remain production integration work. Remaining
  graph compute ABI and route promotion are not claimed. The directional ABI
  checker passes with 77 checks, and unified parity passes with 176 passed,
  45 skipped, and no failures. The required non-interactive Claude review
  timed out after 60 seconds without a completed result; adversarial
  self-review retained overflow-safe bounds and verified raw launch ordering
  through the C-linked B300 execution.

##### M14.6b2b2b: Remaining Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b2b1 and M14.6b2b2b2
- Goal: export the remaining graph compute ABI symbols and resolve embedded
  artifact production-link integration before selecting a Rust CUDA route.

##### M14.6b2b2b1: SwiGLU Libdevice ABI Export

- Status: done
- Goal: export `ds4_gpu_swiglu_tensor` through an embedded Rust CUDA module
  whose PTX can be linked with libdevice from a C-linked static-library
  consumer.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b1/abi-swiglu-libdevice-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_swiglu_libdevice_smoke.py --negative-test`.
- Evidence: `abi_swiglu_kernel` is loaded from embedded PTX through
  `cuda_host::ltoir::build_cubin_from_ptx_with_libdevice` when `__nv_*`
  references are present. The B300 C-linked static-library smoke passes
  clamped, unclamped, output/input alias, invalid-shape, zero-count, and
  null-input checks; it emits and removes a process-local linked `sm_103`
  cubin from embedded `sm_80` PTX, and `nm` confirms 21 exports. Local
  library tests pass with 91 tests;
  B300 release-feature tests pass with 93 tests. Non-release cuda-oxide
  feature codegen rejects reusable library `std::f32::<impl f32>::exp`
  before libdevice lowering and remains recorded as a backend constraint. Whole-archive
  retention and `.note.GNU-stack` remain open before route promotion. The
  SwiGLU ABI checker passes with 96 checks and unified parity passes with 177
  passed, 45 skipped, and no failures. The required non-interactive Claude
  review timed out after 60 seconds without a completed result; adversarial
  self-review removed successful runtime link artifacts after module load and
  retained the non-release codegen constraint explicitly.

##### M14.6b2b2b2: Remaining Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b2b2a and M14.6b2b2b2b because plain
  RMS normalization is tensor-only, while weighted RMS and downstream
  normalization APIs require the still-unowned `model_map` range boundary.
- Goal: export the remaining graph compute ABI symbols and resolve embedded
  artifact production-link integration before selecting a Rust CUDA route.

##### M14.6b2b2b2a: Plain RMS Norm ABI Export

- Status: done
- Goal: export `ds4_gpu_rms_norm_plain_tensor` and
  `ds4_gpu_rms_norm_plain_rows_tensor` from the reusable embedded Rust CUDA
  module without claiming weighted/model-backed normalization.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2a/abi-plain-rms-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_plain_rms_smoke.py --negative-test`.
- Evidence: `abi_rms_norm_plain_kernel` executes through a C-linked
  `libds4_cuda.a` consumer on B300 and passes single-row, batched-row,
  in-place alias, undersized-output, zero-row, current-C zero-width, and null
  checks; `nm` confirms 23 exports. Local library tests pass with 92 tests and
  B300 release-feature tests pass with 94 tests. Weighted RMS remains blocked
  on Rust ownership of the model-map range boundary. Whole-archive retention,
  the `.note.GNU-stack` warning, and the shared-module non-release SwiGLU
  codegen blocker remain open before route promotion. The plain-RMS ABI
  checker passes with 94 checks and unified parity passes with 178 passed, 45
  skipped, and no failures. The required non-interactive Claude review timed
  out after 60 seconds without a completed result; self-review preserved the
  current-C zero-width boundary and kept model-backed RMS pending.

##### M14.6b2b2b2b: Weighted RMS And Model-Backed ABI Assembly

- Status: active; split into M14.6b2b2b2b1 and M14.6b2b2b2b2 because the
  weighted RMS calls contain their model range arguments, while full runtime
  replacement still needs public model-map control and preload policy.
- Goal: export model-backed graph-compute symbols while assembling the public
  model-map control ABI required by a complete route.

##### M14.6b2b2b2b1: Weighted RMS Device-Copy ABI Export

- Status: done
- Goal: export `ds4_gpu_rms_norm_weight_tensor` and
  `ds4_gpu_rms_norm_weight_rows_tensor` through a retained device-copy
  weight-range cache without claiming public model-map controls.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b1/abi-weighted-rms-device-copy-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_weighted_rms_device_copy_smoke.py --negative-test`.
- Evidence: the Rust ABI validates each requested model weight range, copies
  immutable bytes into cached device storage with a completed initial upload,
  and releases those ranges at cleanup; `abi_rms_norm_weight_kernel` then
  executes from a C-linked static-library consumer on B300. Single-row,
  batched-row, in-place alias, alternate-offset, invalid-range, zero-row,
  current-C zero-width, and null-model cases pass, and `nm` confirms 25
  exports. Local library tests pass with 93 tests and B300 release-feature
  tests pass with 95 tests. The weighted ABI checker passes with 102 checks,
  and unified parity passes with 179 passed, 45 skipped, and no failures.
  The required non-interactive Claude adversarial review timed out after 60
  seconds without a completed result; self-review kept public residency
  controls out of this leaf. Public model-map control exports, whole-archive
  retention, the `.note.GNU-stack` warning, and the shared-module non-release
  SwiGLU codegen blocker remain pending.

##### M14.6b2b2b2b2: Public Model-Map Control ABI Assembly

- Status: active; split through M14.6b2b2b2b2a, M14.6b2b2b2b2b1,
  M14.6b2b2b2b2b2a, M14.6b2b2b2b2b2b1, and M14.6b2b2b2b2b2b2 because
  baseline public linkage, bounded registered-range fallback, deterministic
  pageable HMM fallback, and successful chunk-selected copy can be validated
  separately from fd-backed and residual policy.
- Goal: export model-map, file-descriptor, map-range, and cache-range
  controls without weakening the current-C residency-policy contract.

##### M14.6b2b2b2b2a: Basic Model-Control Device-Copy ABI Export

- Status: done
- Goal: export `ds4_gpu_set_model_map`, `ds4_gpu_set_model_fd`,
  `ds4_gpu_set_model_map_range`, and `ds4_gpu_cache_model_range` through
  caller-map device-copy caching without claiming optimized residency policy.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2a/abi-model-control-device-copy-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_device_copy_smoke.py --negative-test`.
- Evidence: the Rust ABI now resets retained copied model ranges on a new
  model mapping and exposes the baseline cache hook; on B300 a C-linked
  static-library consumer pre-caches weighted RMS data, switches mappings,
  mutates and rereads the former mapping to prove stale cache release, and
  passes zero-byte and invalid-range/map behavior with 29 exported symbols.
  Local library tests pass with 94 tests and B300 release-feature tests pass
  with 96 tests. The basic model-control checker passes with 104 checks, and
  unified parity passes with 180 passed, 45 skipped, and no failures.
  The required non-interactive Claude adversarial review timed out after 60
  seconds without a completed result; self-review fixed a model-map
  replacement/cache cleanup lock-order inversion. Fd-backed staging,
  registered/HMM/prefetch, direct-I/O, preload/copy environment selection,
  q8/f16 cache hooks, whole-archive retention, and the `.note.GNU-stack`
  warning remain pending.

##### M14.6b2b2b2b2b1: Registered Attempt And Device-Copy Fallback ABI

- Status: done
- Goal: connect page-bounded read-only registered caller-map attempts and
  explicit device-copy fallback to the public cache-range ABI without claiming
  pageable HMM, fd-backed staging, graph-compute closure, or route promotion.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b1/abi-model-control-registered-fallback-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_registered_fallback_smoke.py --negative-test`.
- Evidence: `rust/ds4-cuda/src/abi.rs` retains either a read-only registered
  guard or a device-copy buffer and attempts registration only for a
  page-rounded source wholly inside the declared raw model map. On B300 the
  existing registration probe reports CUDA error code 801 and matching
  device-copy fallback, while a C-linked public ABI consumer executes a
  page-aligned cached-range weighted RMS call with matching output and 29
  unchanged exports. Local default library tests pass with 95 tests; local
  feature-gated compilation is blocked by the absent CUDA header; B300
  release-feature tests pass with 97 tests. The registered-fallback checker
  passes with 94 checks, and unified parity passes with 181 passed, 45
  skipped, and no failures. The required non-interactive Claude review timed
  out after 60 seconds without completed findings;
  self-review kept synchronized lifetime release and left cross-range
  registration-disable, pageable HMM/prefetch, fd staging, preload/copy
  selection, q8/f16 cache hooks, whole-archive retention, and the
  `.note.GNU-stack` warning pending.

##### M14.6b2b2b2b2b2: Pageable HMM And Fd-Backed Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2a and M14.6b2b2b2b2b2b
  because the deterministic pageable HMM fallback subset can be tested
  without claiming chunked full-model copy or fd staging.
- Goal: connect pageable HMM/prefetch, chunked full-model copy, fd-backed
  direct-I/O staging, registration-disable, preload/copy selection, and
  remaining cache-policy branches to the public model-control ABI without
  claiming remaining graph compute or route promotion.

###### M14.6b2b2b2b2b2a: Pageable HMM Fallback ABI

- Status: done
- Goal: connect the deterministic current-C pageable HMM fallback subset to
  public map-range/direct-read calls while leaving chunk-copy and fd policy
  pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2a/abi-model-control-pageable-hmm-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_pageable_hmm_smoke.py --negative-test`.
- Evidence: the Rust ABI retains a page-bounded pageable host guard only for
  deterministic current-C fallback selection (`DS4_CUDA_COPY_MODEL_CHUNKED`
  plus `DS4_CUDA_NO_MODEL_COPY` or `DS4_CUDA_DIRECT_MODEL`) and routes
  matching model reads through that prefetched window without reporting a
  cache admission. On B300 the existing
  HMM probe confirms pageable advice/prefetch and exact direct readback; a
  C-linked public consumer selects the fallback environment, observes an
  uncached direct-pointer result, and matches weighted RMS output with 29
  unchanged exports. Local default library tests
  pass with 96 tests and B300 release-feature tests pass with 98 tests. The
  pageable-HMM checker passes with 107 checks after the cache-result
  correction, and unified parity at the original leaf passes with 182
  passed, 45 skipped, and no failures. The required non-interactive Claude
  review timed out after 60 seconds without completed findings; self-review
  fixed the consumption-time
  `DS4_CUDA_WEIGHT_CACHE`/`DS4_CUDA_WEIGHT_PRELOAD` exclusion guard.
  Chunked-copy success/allocation-failure policy, global HMM reads outside the
  advised window, fd staging, registration-disable, q8/f16 cache hooks,
  whole-archive retention, and the `.note.GNU-stack` warning remain pending.

###### M14.6b2b2b2b2b2b: Registration And Fd-Backed Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b1,
  M14.6b2b2b2b2b2b2a, M14.6b2b2b2b2b2b2b1, and
  M14.6b2b2b2b2b2b2b2a, and M14.6b2b2b2b2b2b2b2b because deterministic
  successful chunk-selected copying, whole-map registration precedence,
  buffered fd caching, and synchronous direct-I/O fd caching are
  independently testable from residual failure/cache policy.
- Goal: connect chunked full-model copy/failure routing, fd-backed direct-I/O
  staging, registration-disable, preload/copy selection, and remaining
  cache-policy branches without claiming graph-compute closure or route
  promotion.

####### M14.6b2b2b2b2b2b1: Chunk-Selected Model Copy ABI

- Status: done
- Goal: connect deterministic successful `DS4_CUDA_COPY_MODEL_CHUNKED`
  public map-range behavior through a retained copied device image while
  leaving copy failure and fd policy pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b1/abi-model-control-chunk-selected-copy-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_chunk_selected_copy_smoke.py --negative-test`.
- Evidence: Rust retains a copied device image for the unsuppressed
  chunk-selected route, fills the public consumed prefix through bounded
  pinned transfers, and serves model-backed weighted RMS from that image
  before pageable or per-range caching. The C-linked B300 consumer mutates
  host weights after the initial map-range call, calls map-range a second
  time, and still matches the original weighted output, proving same-map
  reuse. Local library tests pass with 97 tests and B300 release-feature
  tests pass with 99 tests; the static library retains 29 exports. The
  chunk-selected copy checker passes with 92 checks, and unified parity
  passes with 183 passed, 45 skipped, and no failures. Self-review found and
  fixed the repeated-map recopy mismatch against current C. The required
  non-interactive Claude review timed out after 60 seconds without completed
  findings. Whole-map registration precedence, allocation/transfer-failure
  fallback to HMM, copy-chunk override and discard/progress side effects,
  unconsumed model ranges, fd staging, registration-disable, q8/f16 cache
  hooks, whole-archive retention, and the `.note.GNU-stack` warning remain
  pending.

####### M14.6b2b2b2b2b2b2a: Whole-Map Registration Precedence ABI

- Status: done
- Goal: connect whole-map read-only registration attempt and
  registered-pointer precedence to public Rust model control while keeping
  fd-backed policy and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2a/abi-model-control-whole-registration-precedence-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_whole_registration_precedence_smoke.py --negative-test`.
- Evidence: Rust retains a whole-map read-only registration guard and gives
  its device pointer precedence ahead of copied, pageable, and per-range
  model storage. Selection matches current C because empty
  `DS4_CUDA_COPY_MODEL` still attempts registration. The dedicated B300
  registration probe reports CUDA error code 801; an aligned C-linked
  consumer then continues into retained chunk-selected copied weights and
  matches original weighted output after host mutation. Local library tests
  pass with 98 tests, B300 release-feature tests pass with 100 tests, and
  the static library retains 29 exports. The whole-map registration
  precedence checker passes with 93 checks, and unified parity passes with
  184 passed, 45 skipped, and no failures. Successful registered zero-copy
  is implemented but not live-observed on this B300 device. The required
  non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`; self-review fixed empty-string
  `DS4_CUDA_COPY_MODEL` selection to match current C. Fd-backed staging,
  residual failure/cache policy, graph compute closure, whole-archive
  retention, route promotion, and the `.note.GNU-stack` warning remain
  pending.

####### M14.6b2b2b2b2b2b2b1: Buffered Fd-Backed Weight Cache ABI

- Status: done
- Goal: connect the deterministic buffered fd-backed public weight-cache
  subset while leaving direct-I/O, asynchronous/budget, residual policy, and
  route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b1/abi-model-control-buffered-fd-cache-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_buffered_fd_cache_smoke.py --negative-test`.
- Evidence: Rust binds an fd configured before the next model map and, only
  for `DS4_CUDA_WEIGHT_CACHE=1` plus `DS4_CUDA_NO_DIRECT_IO=1`, uploads a
  buffered `pread` range into pinned-backed retained device storage before
  page-bounded registration or caller-map fallback. An aligned C-linked B300
  consumer uses different fd and host-map weights and rewrites the file
  after caching; weighted RMS still observes the original fd-backed bytes.
  Local library tests pass with 99 tests, B300 release-feature tests pass
  with 101 tests, and the static library retains 29 exports. The buffered fd
  checker passes with 96 checks, and unified parity passes with 185 passed,
  45 skipped, and no failures. Self-review corrected fd-vs-range-registration
  priority and interrupted-`pread` retry behavior to match current C.
  The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Direct-I/O
  reopen/alignment, asynchronous staging, cache budget, source-page
  advice/progress, residual failure/cache policy, graph compute closure,
  whole-archive retention, route promotion, and the `.note.GNU-stack`
  warning remain pending.

####### M14.6b2b2b2b2b2b2b2: Direct-I/O And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b because retained direct-fd reopen plus aligned
  synchronous read/fallback behavior is independently testable from
  persistent error-disable, asynchronous staging, budget, and residual
  cache policy.
- Goal: connect public fd-backed direct-I/O staging, asynchronous/budget
  policy, chunk-copy failure routing, and residual model-control policy
  without claiming graph compute closure or route promotion.

######## M14.6b2b2b2b2b2b2b2a: Direct-I/O Fd Cache ABI

- Status: done
- Goal: connect the public retained `O_DIRECT` fd reopen, aligned pinned
  read, and buffered fallback subset while leaving persistent direct-read
  error disablement, asynchronous staging, budgets, residual failure/cache
  policy, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2a/abi-model-control-direct-io-fd-cache-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_io_fd_cache_smoke.py --negative-test`.
- Evidence: Rust reopens the configured public model fd through
  `/proc/self/fd/<fd>` with `O_DIRECT` on Linux when direct I/O is
  permitted, retains file-size/alignment state, performs aligned pinned
  reads with buffered fallback, and consumes the resulting device range
  before per-range registration or caller-map fallback. The C-linked B300
  consumer observes fd-backed weighted RMS and retained cache reuse with
  `DS4_CUDA_NO_DIRECT_IO` unset; it does not claim to directly observe
  `O_DIRECT` selection. A refreshed M14.1b2b3b1 B300 direct-read probe
  reports `direct_io_selected=true`, 4096-byte alignment, aligned read
  offset/size `0`/`8192`, exact readback, and tail buffered fallback.
  Local library tests pass with 100 tests, B300 release-feature tests pass
  with 102 tests, and the static library retains 29 exports. The direct-I/O
  fd-cache checker passes with 111 checks and unified parity passes with 186
  passed, 45 skipped, and no failures. The required non-interactive Claude
  review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed
  findings; self-review kept public output separate from lower-level direct
  selection evidence. Persistent direct-read error disablement,
  asynchronous staging, cache-budget and source-page/progress policy,
  residual failure/cache policy, graph compute closure, whole-archive
  retention, route promotion, and the `.note.GNU-stack` warning remain
  pending.

######## M14.6b2b2b2b2b2b2b2b: Direct-I/O Residual Failure And Cache Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2 because the current-C direct-read disable
  transition and errno classes can be proved independently from public
  asynchronous staging, arena/budget, and remaining cache selection policy.
- Goal: connect persistent direct-read error disablement, asynchronous/budget
  policy, chunk-copy failure routing, and residual model-control policy
  without claiming graph compute closure or route promotion.

######### M14.6b2b2b2b2b2b2b2b1: Direct-I/O Error Disable ABI

- Status: done
- Goal: connect current-C disable-after-selected-direct-read-error state to
  the public Rust fd-cache path without claiming a live induced public error,
  asynchronous staging, budgets, remaining cache policy, or route promotion.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b1/abi-model-control-direct-io-error-disable-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_io_error_disable_smoke.py --negative-test`.
- Evidence: Rust applies current-C direct-read disabling errno classes
  (`EINVAL`, `EFAULT`, `ENOTSUP`, `EOPNOTSUPP`), drops its retained direct
  fd state, resets direct alignment, and enters buffered fd fallback. This
  branch is not reliably inducible through the public B300 filesystem
  harness: the B300 feature tests execute the public error-class policy
  check, while the previously linked direct-enabled fd-cache consumer is
  rerun as a successful-regression gate. Local library tests pass with 101
  tests, B300 release-feature tests pass with 104 tests, and the static
  library retains 29 exports. The error-disable checker passes with 88
  checks and unified parity passes with 187 passed, 45 skipped, and no
  failures. The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings; self-review
  retained the explicit non-claim for live public error induction and
  asynchronous/budget policy. Asynchronous staging, arena/cache-budget and
  source-page/progress policy, chunk-copy failure selection, graph compute
  closure, whole-archive retention, route promotion, and the
  `.note.GNU-stack` warning remain pending.

######### M14.6b2b2b2b2b2b2b2b2: Public Async Staging And Residual Cache Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b because direct-enabled public asynchronous
  staging is independently testable from buffered-only staging, arena/cache
  budget, source-page/progress, and residual selection policy.
- Goal: connect public asynchronous/budget staging, chunk-copy failure
  routing, and residual model-control selection/cache policy without claiming
  graph compute closure or route promotion.

########## M14.6b2b2b2b2b2b2b2b2a: Public Direct-I/O Async Staging ABI

- Status: done
- Goal: connect direct-enabled public fd-cache requests to four-slot
  asynchronous staging and the current-C model-copy chunk-size clamp while
  leaving buffered-only async, budgets, and residual policy pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2a/abi-model-control-direct-io-async-staging-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_io_async_staging_smoke.py --negative-test`.
- Evidence: Rust uses four pinned direct-enabled fd staging slots with event
  waits before reuse and final synchronization. The C-linked B300 consumer
  sets a 16 MiB chunk override, requests five chunks, observes fd-backed
  weighted output, and reuses retained device bytes after file mutation; it
  does not claim public event-count observation. The existing lower-level
  async B300 baseline records four slots, seven events, and two reuse waits.
  Local tests pass with 102 tests, B300 release-feature tests pass with 106
  tests, and the static library retains 29 exports. The checker passes with
  103 checks and unified parity passes with 188 passed, 45 skipped, and no
  failures. The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  Buffered-only async, arena/cache-budget and source-page/progress policy,
  residual selection, whole-archive retention, route promotion, and the
  `.note.GNU-stack` warning remain pending.

########## M14.6b2b2b2b2b2b2b2b2b: Residual Fd Cache And Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2 because buffered-only public asynchronous
  staging is independently testable from arena/cache budget,
  source-page/progress, and residual selection policy.
- Goal: connect buffered-only asynchronous staging, arena/cache-budget and
  source-page/progress policy, chunk-copy failure routing, and residual
  model-control selection/cache policy without claiming graph compute closure
  or route promotion.

########### M14.6b2b2b2b2b2b2b2b2b1: Public Buffered Fd Async Staging ABI

- Status: done
- Goal: connect buffered-only public fd-cache requests to the shared
  four-slot asynchronous uploader while leaving arena/budget and residual
  policy pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b1/abi-model-control-buffered-fd-async-staging-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_buffered_fd_async_staging_smoke.py --negative-test`.
- Evidence: Rust routes `DS4_CUDA_NO_DIRECT_IO=1` fd-cache reads through the
  existing four-slot event-backed uploader. A C-linked B300 consumer sets a
  16 MiB chunk override, requests five buffered chunks, observes fd-backed
  weighted output, and retains cached device bytes after file mutation; the
  direct-enabled five-chunk public consumer also passes after refactoring.
  Local tests pass with 103 tests, B300 release-feature tests pass with 107
  tests, and the static library retains 29 exports. The checker passes with
  103 checks and unified parity passes with 189 passed, 45 skipped, and no
  failures. The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  Arena/cache-budget and source-page/progress policy, residual selection,
  whole-archive retention, route promotion, and the `.note.GNU-stack`
  warning remain pending.

########### M14.6b2b2b2b2b2b2b2b2b2: Arena Budget And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b because public fd arena suballocation/lifetime
  is independently testable from cache-budget fallback, source-page/progress,
  and residual selection policy.
- Goal: connect arena/cache-budget and source-page/progress policy,
  chunk-copy failure routing, and residual model-control selection/cache
  policy without claiming graph compute closure or route promotion.

############ M14.6b2b2b2b2b2b2b2b2b2a: Public Fd Arena Suballocation ABI

- Status: done
- Goal: connect public fd-cache ranges to retained arena suballocation and
  synchronized arena reset while leaving cache-budget fallback,
  source-page/progress, residual selection, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-arena-suballocation-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_arena_suballocation_smoke.py --negative-test`.
- Evidence: Rust retains Linux fd-cache destinations in `ABI_MODEL_ARENAS`,
  applies the current-C arena chunk override clamp/growth rule and
  256-byte-aligned suballocation, and clears arena lifetime after
  synchronization. A C-linked B300 consumer selects a 256 MiB bounded arena
  allocation, admits two disjoint buffered fd ranges, observes fd-backed
  weighted output over divergent host bytes, and reuses both retained ranges
  after backing-file mutation. Local tests pass with 104 tests, B300
  release-feature tests pass with 109 tests, and the static library retains
  29 exports. The lower-level M14.1b2b3b2 B300 baseline remains the
  internal one-arena/two-range proof. The new checker passes with 104 checks,
  and unified parity passes with 190 passed, 45 skipped, and no failures.
  The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Cache-budget fallback,
  source-page/progress policy, residual selection, whole-archive retention,
  route promotion, and the `.note.GNU-stack` warning remain pending.

############ M14.6b2b2b2b2b2b2b2b2b2b: Cache Budget And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2 because public fd cache-budget fallback is
  independently testable from source-page/progress and residual selection
  policy.
- Goal: connect public cache-budget fallback, source-page/progress policy,
  chunk-copy failure routing, and residual model-control selection/cache
  policy without claiming graph compute closure or route promotion.

############# M14.6b2b2b2b2b2b2b2b2b2b1: Public Fd Cache Budget Fallback ABI

- Status: done
- Goal: connect public fd-cache limit admission and uncached budget fallback
  pointer resolution while leaving source-page/progress, residual selection,
  and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-cache-budget-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_cache_budget_smoke.py --negative-test`.
- Evidence: Rust parses the current-C GiB cache-limit policy, retains admitted
  fd byte accounting with arena state, rejects over-budget requests before
  staging/source construction, and returns an uncached direct model pointer
  for that operation. A C-linked B300 consumer admits and computes one small
  fd range, then returns successfully from repeated rejected 1 GiB requests
  whose source pages are inaccessible and whose file bytes are absent. It
  deliberately does not consume the returned host fallback pointer in a
  kernel. Local tests pass with 105 tests, B300 release-feature tests pass
  with 111 tests, and the static library retains 29 exports. The prior public
  fd-arena, buffered asynchronous staging, and direct-I/O asynchronous staging
  C-linked consumers also pass against that budget-aware static library.
  The public budget checker passes 110 checks, and the default unified report
  passes with 191 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Source-page discard/progress, residual
  selection, whole-archive retention, route promotion, and the
  `.note.GNU-stack` warning remain pending.

############# M14.6b2b2b2b2b2b2b2b2b2b2: Source-Page Progress And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b because public fd source-page/progress
  behavior is independently observable from remaining selection policy.
- Goal: connect source-page/progress policy, residual model-control
  selection/cache behavior, and remaining failure routing without claiming
  graph compute closure or route promotion.

############## M14.6b2b2b2b2b2b2b2b2b2b2a: Public Fd Source-Page And Progress ABI

- Status: done
- Goal: connect source-file/mapping discard advice and model-load progress
  behavior to public fd-cached uploads while leaving residual model-control
  selection and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-source-page-progress-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_source_page_progress_smoke.py --negative-test`.
- Evidence: Rust now applies source-file and source-mapping discard advice
  after public fd upload chunks, emits/suppresses current-C progress behavior,
  and resets progress on synchronized model replacement or cleanup. A
  C-linked B300 consumer interposes both POSIX advice calls and captures
  stderr across two ordinary multi-chunk uploads and one suppressed upload,
  observing advice invocation, progress reset, and keep-pages/verbose
  suppression without claiming physical eviction or TTY refresh rendering.
  Local tests pass with 106 tests, B300 release-feature tests pass with 112
  tests, and the static library retains 29 exports. The prior public budget,
  fd-arena, buffered asynchronous staging, and direct-I/O asynchronous staging
  linked consumers pass against the new static library. The public
  page/progress checker passes 122 checks, and the default unified report
  passes with 192 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Residual model-control selection, whole-archive
  retention, route promotion, and the `.note.GNU-stack` warning remain
  pending.

############## M14.6b2b2b2b2b2b2b2b2b2b2b: Residual Model-Control Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2b2 because cross-range registration disablement
  and reset are independently observable from remaining failure selection.
- Goal: connect residual model-control selection/cache behavior and remaining
  failure routing without claiming graph compute closure or route promotion.

############### M14.6b2b2b2b2b2b2b2b2b2b2b1: Public Cross-Range Registration Disable ABI

- Status: done
- Goal: preserve current-C per-range read-only registration disablement after
  selected errors and reset it on public model replacement while leaving
  remaining failure selection and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-registration-disable-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_registration_disable_smoke.py --negative-test`.
- Evidence: Rust now retains a model-lifetime registration gate, disables it
  only after `CUDA_ERROR_NOT_SUPPORTED` or `CUDA_ERROR_INVALID_VALUE`, and
  resets it on public map replacement or cleanup. A C-linked B300 consumer
  interposes `cuMemHostRegister_v2` with error code 801, observes two
  attempts before the first range disables retries, no attempt for a second
  disjoint range, and two new attempts after model replacement. Weighted RMS
  continues through device-copy fallback without claiming successful
  zero-copy registration. Local tests pass with 107 tests, B300
  release-feature tests pass with 114 tests, and the static library retains
  29 exports. The prior public registered-fallback, whole-map-registration,
  fd-budget, and fd source-page/progress linked consumers pass against the
  new static library. The public registration-disable checker passes 102
  checks, and the default unified report passes with 193 passed, 45 skipped,
  and 0 failed. The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Remaining
  failure selection, whole-archive retention, route promotion, and the
  `.note.GNU-stack` warning remain pending.

############### M14.6b2b2b2b2b2b2b2b2b2b2b2: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b because successful nonempty full-model copy
  selection is independently observable from remaining failure selection.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

################ M14.6b2b2b2b2b2b2b2b2b2b2b2a: Public Full-Model Copy Selection ABI

- Status: done
- Goal: preserve current-C successful nonempty `DS4_CUDA_COPY_MODEL`
  selection, copied-image retention, and copy-failure continuation wiring
  while leaving live copy-failure observation and remaining selection pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-full-model-copy-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_full_model_copy_smoke.py --negative-test`.
- Evidence: Rust now routes nonempty full-model copy selection through a
  retained whole-map device image before registration, with failed-copy
  continuation to registration preserved in source. A C-linked B300 consumer
  selects full copy, interposes registration, mutates host weights after map
  setup, replaces the map, and observes retained copied-image weighted output
  with zero registration calls. Local tests pass with 108 tests, B300
  release-feature tests pass with 115 tests, and the static library retains
  29 exports. The chunk-selected-copy, whole-map-registration,
  registration-disable, fd-budget, and fd source-page/progress linked
  consumers pass against the full-copy-aware static library; the two fd-only
  fixtures leave full-model copy unselected to match current-C precedence.
  The public full-copy checker passes 94 checks, and the default unified
  report passes with 194 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Live forced copy failure, remaining failure
  selection, whole-archive retention, route promotion, and the
  `.note.GNU-stack` warning remain pending.

################ M14.6b2b2b2b2b2b2b2b2b2b2b2b: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2 because nonempty direct-model read
  selection and its uncached public cache-result boundary are independently
  observable from remaining failure selection.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

################# M14.6b2b2b2b2b2b2b2b2b2b2b2b1: Public Direct-Model Read Selection ABI

- Status: done
- Goal: preserve current-C nonempty `DS4_CUDA_DIRECT_MODEL` host-read
  selection and its uncached public cache result while leaving remaining
  failure selection and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-direct-model-read-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_model_read_smoke.py --negative-test`.
- Evidence: Rust now resolves nonempty direct-model reads from the caller
  mapping before per-range staging after global copied/registered state and
  reports only actual retained storage from public cache calls. A C-linked
  B300 consumer rejects whole-map registration with error code 801, observes
  no range admission or registration retry, mutates host weights, and
  verifies weighted RMS reads the changed bytes. The corrected pageable-HMM
  predecessor proves its prefetched direct pointer is likewise not reported
  as cached. Local tests pass with 109 tests, B300 release-feature tests pass
  with 116 tests, and the static library retains 29 exports. The preceding
  full-model-copy linked consumer passes against the new static library.
  The public direct-model read checker passes 88 checks, and the default
  unified report passes with 195 passed, 45 skipped, and 0 failed. The
  required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Remaining
  failure selection, whole-archive retention, route promotion, and the
  `.note.GNU-stack` warning remain pending.

################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b because public bound-fd selection without
  the weight-cache flag and its preload/disable boundaries are independently
  observable from remaining failure selection.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2a: Public Default Fd Selection ABI

- Status: done
- Goal: preserve current-C bound-fd selection when `DS4_CUDA_WEIGHT_CACHE` is
  absent, retain fd selection under `DS4_CUDA_WEIGHT_PRELOAD`, and honor
  `DS4_CUDA_NO_FD_CACHE` bypass while leaving remaining failure selection and
  route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-default-fd-selection-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_default_fd_selection_smoke.py --negative-test`.
- Evidence: Rust now selects a configured model fd whenever fd caching is not
  disabled and direct-host bypass is not selected, without requiring
  `DS4_CUDA_WEIGHT_CACHE`. A C-linked B300 consumer rejects whole-map
  registration with error code 801, observes file-backed weighted output
  with the weight-cache flag absent and under preload, then observes fallback
  host-weight output and a range-registration attempt with
  `DS4_CUDA_NO_FD_CACHE`. Local tests pass with 110 tests, B300
  release-feature tests pass with 117 tests, and the static library retains
  29 exports. The preceding buffered-fd, direct-model, pageable-HMM, and
  full-model-copy linked consumers pass against the new static library.
  The public default-fd selection checker passes 91 checks, and the default
  unified report passes with 196 passed, 45 skipped, and 0 failed. The
  required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  Remaining failure selection, whole-archive retention, route promotion, and
  the `.note.GNU-stack` warning remain pending.

################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2 because the public uncached return value
  and compute use of fd-budget host fallback are independently observable
  from arena-allocation and strict-cache failure selection.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1: Public Fd Budget Fallback Cache-Result ABI

- Status: done
- Goal: preserve current-C uncached public cache-result behavior for raw fd
  byte-budget fallback and observe weighted compute through the returned host
  pointer while leaving arena/strict-cache failure selection pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-budget-cache-result-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_budget_cache_result_smoke.py --negative-test`.
- Evidence: Rust now reports retained cache membership after public range
  resolution and releases the retained-range mutex before callback execution.
  A C-linked B300 consumer admits a near-one-GiB fd-backed range, triggers a
  small over-budget fallback with divergent host/file weights, observes an
  uncached public result, and verifies weighted RMS consumes host bytes. The
  corrected preceding fd-budget consumer now reports its oversized fallback
  as uncached; default-fd, direct-model, pageable-HMM, and full-model-copy
  consumers pass against the rebuilt archive. Local tests pass with 111
  tests, B300 release-feature tests pass with 118 tests, and the static
  library retains 29 exports. The public fd-budget cache-result checker passes
  101 checks, and the default unified report passes with 197 passed, 45
  skipped, and 0 failed. The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Arena
  allocation failure, persistent cache-full state,
  `DS4_CUDA_STRICT_WEIGHT_CACHE` continuation, route promotion, and the
  `.note.GNU-stack` warning remain pending.

################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2b because fd arena allocation failure,
  strict continuation, and persistent cache-full behavior are independently
  observable from remaining staging/read/copy failure selection.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

#################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a: Public Fd Arena Failure Selection ABI

- Status: done
- Goal: preserve current-C fd arena allocation-failure routing, including
  non-strict uncached host fallback, strict continuation into cached fallback
  handling, and persistent no-retry state after allocation failure.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-arena-failure-selection-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_arena_failure_selection_smoke.py --negative-test`.
- Evidence: Rust now separates raw fd-budget fallback from arena failure,
  stores the model-lifetime cache-full state, routes arena failure through
  `DS4_CUDA_STRICT_WEIGHT_CACHE`, and allocates uninitialized fd-arena bytes
  before staged writes. A C-linked B300 consumer interposes the 256 MiB
  arena allocation with out-of-memory, proves uncached host-weight output in
  non-strict mode, proves cached device-copy output in strict mode, and
  proves no allocation retry for a second range. Local tests pass with 112
  tests, B300 release-feature tests pass with 119 tests, and the static
  library retains 29 exports. Successful fd-arena suballocation, fd-budget
  cache-result, default-fd, and registration-disable linked consumers pass
  against the rebuilt archive. The focused comparator passes with 107 checks
  and the default unified parity report passes with 198 passed, 45 skipped,
  and 0 failed. The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  Aligned-budget strict routing is source-backed but not separately forced
  live; staging allocation/read/copy failure selection, route promotion, and
  the `.note.GNU-stack` warning remain pending.

#################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2b: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bb because failed selected fd uploads
  must continue into fallback handling without retrying through a buffered
  fd upload.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

##################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba: Public Fd Upload Failure Continuation ABI

- Status: done
- Goal: preserve current-C one-attempt fd failure routing so a failed
  selected fd upload continues into registration or device-copy fallback
  without launching a second buffered fd upload.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba/abi-model-control-fd-upload-failure-continuation-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_upload_failure_continuation_smoke.py --negative-test`.
- Evidence: Rust now selects exactly one configured fd upload branch before
  matching its result, so an attempted direct-fd upload failure falls through
  to registration/device-copy handling once. A C-linked B300 consumer selects
  that branch, injects one `cuMemcpyHtoDAsync_v2` failure after tensor setup,
  rejects range registration, and proves cached device-copy consumes and
  retains original host bytes rather than retrying divergent fd bytes. Local
  tests pass with 113 tests, B300 release-feature tests pass with 120 tests,
  and the static library retains 29 exports. Fd-arena failure, fd-budget
  cache-result, default-fd, direct-I/O asynchronous-staging, and
  registration-disable linked consumers pass against the rebuilt archive.
  The focused comparator passes with 102 checks and the default unified
  parity report passes with 199 passed, 45 skipped, and 0 failed. The
  required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Staged
  allocation, fd-read, event-record, event-wait, and final synchronization
  failure observations, route promotion, and the `.note.GNU-stack` warning
  remain pending.

##################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbb because the public Rust fd path must
  retain the four-slot staging pool across cached ranges before remaining
  failure observations can be isolated.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

###################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba: Public Fd Stage Pool Reuse ABI

- Status: done
- Goal: preserve current-C public fd stage-pool lifetime so successful range
  uploads retain and reuse four pinned staging slots until cleanup.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba/abi-model-control-fd-stage-pool-reuse-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_stage_pool_reuse_smoke.py --negative-test`.
- Evidence: Rust now keeps a Linux-only public fd staging pool independently
  from device arenas, reuses sufficient slots for later range uploads and
  model-map replacement, and releases it at public cleanup. A C-linked B300
  consumer selects buffered fd caching, establishes one four-slot pool, arms
  `cuMemAllocHost_v2` failure before a second disjoint range, and proves
  file-backed weighted output without another allocation or registration
  fallback. Local tests pass with 114 tests, B300 release-feature tests pass
  with 121 tests, and the static library retains 29 exports. Fd-upload
  failure continuation, fd-arena failure, fd-budget cache-result, default-fd,
  direct-I/O asynchronous-staging, and registration-disable linked consumers
  pass against the rebuilt archive. The focused comparator passes with 105
  checks, and the default unified parity report passes with 200
  passed, 45 skipped, and 0 failed. The required non-interactive Claude
  review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed
  findings. Initial stage allocation and pool-growth failure, fd-read,
  event, and final synchronization observations, route promotion, and the
  `.note.GNU-stack` warning remain pending.

###################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbb because stage allocation failure
  continuation can be live-observed independently from fd read, event, and
  final synchronization failure paths.
- Goal: connect remaining model-control failure selection without claiming
  graph compute closure or route promotion.

####################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba: Public Fd Stage Allocation Failure Continuation ABI

- Status: done
- Goal: preserve current-C public fd stage-allocation failure continuation so
  failed pinned staging retries on later ranges and continues through cached
  host fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba/abi-model-control-fd-stage-allocation-failure-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_stage_allocation_failure_smoke.py --negative-test`.
- Evidence: Existing Rust stage-pool allocation propagation already maps
  failed `pinned_zeroed` into registration/device-copy fallback without
  consulting strict fd-cache mode, so no routing code change is required. A
  C-linked B300 consumer selects buffered fd caching, forces
  `cuMemAllocHost_v2` failure before two disjoint ranges across a strict-mode
  transition, rejects the first range-registration attempt, and proves both
  ranges retain host-backed cached output rather than fd bytes. The second
  allocation attempt proves no arena cache-full latch; one range-registration
  attempt preserves the prior registration-disable boundary. Local tests pass
  with 115 tests, B300 release-feature tests pass with 122 tests, and the
  static library retains 29 exports. Stage-pool reuse, fd-upload failure
  continuation, fd-arena failure, fd-budget cache-result, default-fd,
  direct-I/O asynchronous-staging, and registration-disable linked consumers
  pass against the rebuilt archive. The focused comparator and the default
  unified parity report pass with 201 passed, 45 skipped, and 0 failed. The
  required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Fd-read,
  event, and final synchronization observations, route promotion, and the
  `.note.GNU-stack` warning remain pending.

####################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbb because buffered fd-read failure
  continuation can be forced independently from event and final
  synchronization failure paths.
- Goal: connect remaining fd-read, event, and final synchronization failure
  selection without claiming graph compute closure or route promotion.

######################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba: Public Fd Read Failure Continuation ABI

- Status: done
- Goal: preserve current-C public buffered fd-read failure continuation so
  failed staged reads retry on later ranges and continue through cached host
  fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba/abi-model-control-fd-read-failure-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_read_failure_smoke.py --negative-test`.
- Evidence: Existing Rust buffered-read propagation already maps failed
  `pread` into registration/device-copy fallback without consulting strict
  fd-cache mode, so no routing code change is required. A C-linked B300
  consumer selects buffered fd caching, injects `EIO` from `pread` only for
  the configured model fd before two disjoint ranges across a strict-mode
  transition, rejects the first range-registration attempt, and proves both
  ranges retain host-backed cached output rather than fd bytes. Two read
  failures prove retry without arena cache-full latching; one
  range-registration attempt preserves the prior registration-disable
  boundary. Local tests pass with 116 tests, B300 release-feature tests pass
  with 123 tests, and the static library retains 29 exports.
  Stage-allocation failure, stage-pool reuse, fd-upload failure continuation,
  fd-arena failure, fd-budget cache-result, default-fd, direct-I/O
  asynchronous-staging, and registration-disable linked consumers pass
  against the rebuilt archive. The focused comparator and default unified
  parity report pass with 202 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Event and final synchronization observations,
  route promotion, and the `.note.GNU-stack` warning remain pending.

######################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbb because event-record failure
  continuation can be forced independently from event-wait and final
  stream-synchronization failure paths.
- Goal: connect remaining event and final synchronization failure selection
  without claiming graph compute closure or route promotion.

######################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba: Public Fd Event Record Failure Continuation ABI

- Status: done
- Goal: preserve current-C public fd event-record failure continuation so
  failed staging records retry on later ranges and continue through cached
  host fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba/abi-model-control-fd-event-record-failure-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_event_record_failure_smoke.py --negative-test`.
- Evidence: Existing Rust event-record propagation already maps failed
  `backend.record_event()` into registration/device-copy fallback without
  consulting strict fd-cache mode, and cuda-oxide maps that call to
  `cuEventRecord`, so no routing code change is required. A C-linked B300
  consumer selects buffered fd caching, forwards `cuEventRecord` through
  setup, injects record failure before two disjoint ranges across a
  strict-mode transition, rejects the first range-registration attempt, and
  proves both ranges retain host-backed cached output rather than fd bytes.
  Two event-record failures prove retry without arena cache-full latching; one
  range-registration attempt preserves the prior registration-disable
  boundary. Local tests pass with 117 tests, B300 release-feature tests pass
  with 124 tests, and the static library retains 29 exports. Fd-read failure,
  stage-allocation failure, stage-pool reuse, fd-upload failure continuation,
  fd-arena failure, fd-budget cache-result, default-fd, direct-I/O
  asynchronous-staging, and registration-disable linked consumers pass
  against the rebuilt archive. The focused comparator and default unified
  parity report pass with 203 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Event-wait and final synchronization
  observations, route promotion, and the `.note.GNU-stack` warning remain
  pending.

######################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbb because event-wait failure
  continuation can be forced independently from final
  stream-synchronization failure.
- Goal: connect remaining event-wait and final synchronization failure
  selection without claiming graph compute closure or route promotion.

########################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba: Public Fd Event Wait Failure Continuation ABI

- Status: done
- Goal: preserve current-C public fd event-wait failure continuation so failed
  staging-slot reuse retries on later ranges and continues through cached host
  fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba/abi-model-control-fd-event-wait-failure-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_event_wait_failure_smoke.py --negative-test`.
- Evidence: Existing Rust event-wait propagation already maps failed
  `event.synchronize()` into registration/device-copy fallback without
  consulting strict fd-cache mode, so no routing code change is required. A
  C-linked B300 consumer selects buffered fd caching, requests two ranges
  exceeding the four-slot event ring, injects `cuEventSynchronize` failure on
  fifth-chunk slot reuse across a strict-mode transition, rejects the first
  range-registration attempt, and proves both ranges retain host-backed
  cached output rather than fd bytes. Local tests pass with 118 tests, B300
  release-feature tests pass with 125 tests, and the static library retains
  29 exports. Event-record failure, fd-read failure, stage-allocation
  failure, stage-pool reuse, fd-upload failure continuation, fd-arena
  failure, fd-budget cache-result, default-fd, direct-I/O
  asynchronous-staging, and registration-disable linked consumers pass
  against the rebuilt archive. The focused comparator and default unified
  parity report pass with 204 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Final stream-synchronization observation, route
  promotion, and the `.note.GNU-stack` warning remain pending.

########################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbb because final
  stream-synchronization failure can be forced independently from remaining
  graph-compute and route-promotion work.
- Goal: connect remaining final stream-synchronization failure selection
  without claiming graph compute closure or route promotion.

########################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba: Public Fd Final Sync Failure Continuation ABI

- Status: done
- Goal: preserve current-C public fd final upload synchronization failure
  continuation so failed completed staging attempts retry on later ranges and
  continue through cached host fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba/abi-model-control-fd-final-sync-failure-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_final_sync_failure_smoke.py --negative-test`.
- Evidence: Existing Rust final synchronization handling already maps failed
  `backend.synchronize()` into registration/device-copy fallback without
  consulting strict fd-cache mode, so no routing code change is required. A
  C-linked B300 consumer selects buffered fd caching, injects one
  `cuStreamSynchronize` failure per staged fd attempt across a strict-mode
  transition, forwards subsequent synchronization so fallback can complete,
  rejects the first range-registration attempt, and proves both ranges retain
  host-backed cached output rather than fd bytes. Local tests pass with 119
  tests, B300 release-feature tests pass with 126 tests, and the static
  library retains 29 exports. Event-wait failure, event-record failure,
  fd-read failure, stage-allocation failure, stage-pool reuse, fd-upload
  failure continuation, fd-arena failure, fd-budget cache-result, default-fd,
  direct-I/O asynchronous-staging, and registration-disable linked consumers
  pass against the rebuilt archive. The focused comparator and default unified
  parity report pass with 205 passed, 45 skipped, and 0 failed. The required
  non-interactive Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
  without completed findings. Q8/f16 hooks, remaining graph compute, route
  promotion, and the `.note.GNU-stack` warning remain pending.

########################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbb because the public
  single-token F16 projection ABI can be linked and observed independently
  from multi-token BLAS, paired/Q8 compute, and production route promotion.
- Goal: connect remaining q8/f16 hooks, graph compute, whole-archive
  retention, and production route-promotion work without claiming C CUDA
  removal before those gates pass.

############################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba: Public Single-Token F16 Projection ABI

- Status: done
- Goal: Rust-own the public single-token F16 dense-projection ABI through the
  current-C base, ordered-chunks, and serial dispatch boundary without
  claiming multi-token BLAS, paired projection, Q8 cache, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba/abi-matmul-f16-single-token-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f16_single_token_smoke.py --negative-test`.
- Evidence: Rust now exports `ds4_gpu_matmul_f16_tensor` for `n_tok == 1`,
  resolves F16 weights through the cached model-range path, and launches
  embedded base, ordered-chunks, or serial kernels. A C-linked B300 consumer
  verifies all three selections, observes cached weights after host-map
  mutation, and rejects unowned multi-token BLAS plus invalid ranges. Local
  tests pass with 120 tests, B300 release-feature tests pass with 127 tests,
  and the static library exposes 30 Rust ABI symbols. Fourteen predecessor
  C-linked consumers pass against the rebuilt archive; the generated embedded
  object still produces its `.note.GNU-stack` executable-stack warning on
  each relink. The focused comparator and default unified parity report pass
  with 206 passed, 45 skipped, and 0 failed. The required non-interactive
  Claude review returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed
  findings. Multi-token/paired F16, Q8/F16 cache hooks, remaining graph
  compute, whole-archive retention policy, route promotion, and C CUDA
  removal remain pending.

############################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbb because public
  single-token paired F16 projection can be proved independently from
  multi-token BLAS, Q8/F16 cache hooks, and route promotion.
- Goal: connect multi-token/paired F16 projection, q8/f16 cache hooks,
  remaining graph compute, whole-archive retention policy, and production
  route-promotion work without claiming C CUDA removal before those gates
  pass.

############################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbba: Public Single-Token Paired F16 Projection ABI

- Status: done
- Goal: Rust-own public single-token paired F16 projection through the
  current-C paired ordered-chunks dispatch and its independent fallback
  selections without claiming multi-token BLAS, Q8/F16 cache, or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbba/abi-matmul-f16-pair-single-token-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f16_pair_single_token_smoke.py --negative-test`.
- Evidence: Rust exports `ds4_gpu_matmul_f16_pair_tensor` for `n_tok == 1`
  through a new embedded paired ordered-chunks kernel and reuses the owned
  single-token F16 ABI for forced independent fallbacks. A C-linked B300
  consumer verifies default paired, no-pair, no-ordered, and serial fallback
  output; cached dual weights survive host mutation, while multi-token and
  invalid ranges reject. Local tests pass with 121 tests, B300
  release-feature tests pass with 128 tests, the static library exposes 31
  Rust ABI symbols, and fifteen predecessor C-linked consumers pass against
  the rebuilt archive; each predecessor relink retains the known
  embedded-object executable-stack warning. The focused comparator and
  default unified report pass with 207 passed, 45 skipped, and 0 failed.
  The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Multi-token
  BLAS, Q8/F16 cache hooks, remaining graph compute,
  whole-archive retention policy, route promotion, C CUDA removal, and the
  embedded-object executable-stack warning remain pending.

############################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbb because public
  single-token F32 projection can be proved independently from multi-token
  BLAS, Q8/F16 cache hooks, and route promotion.
- Goal: connect multi-token F16/F32 BLAS projection, q8/f16 cache hooks,
  remaining graph compute, whole-archive retention policy, and production
  route-promotion work without claiming C CUDA removal before those gates
  pass.

############################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbba: Public Single-Token F32 Projection ABI

- Status: done
- Goal: Rust-own public single-token F32 projection through the current-C
  base-kernel dispatch without claiming multi-token BLAS, Q8/F16 cache, or
  route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbba/abi-matmul-f32-single-token-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f32_single_token_smoke.py --negative-test`.
- Evidence: Rust exports `ds4_gpu_matmul_f32_tensor` for `n_tok == 1`
  through a new embedded current-C-equivalent base reduction kernel and
  resolves cached F32 model weights before launch. A C-linked B300 consumer
  verified output and cached weights after host mutation while rejecting the
  then-unowned multi-token BLAS boundary; that negative observation is
  historical after the successor F32 BLAS leaf below. Local tests pass with 122 tests, B300
  release-feature tests pass with 129 tests, the static library exposes 32
  Rust ABI symbols, and sixteen predecessor C-linked consumers pass against
  the rebuilt archive; each predecessor relink retains the known
  embedded-object executable-stack warning. The focused comparator and
  default unified report pass with 208 passed, 45 skipped, and 0 failed.
  The required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Multi-token
  F16/F32 BLAS, Q8/F16 cache hooks, remaining graph compute, whole-archive
  retention policy, route promotion, C CUDA removal, and the embedded-object
  executable-stack warning remain pending.

############################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbb because public
  multi-token F32 BLAS projection can be proved independently from
  multi-token F16 BLAS, Q8/F16 cache hooks, and route promotion.
- Goal: connect multi-token F16/F32 BLAS projection, q8/f16 cache hooks,
  remaining graph compute, whole-archive retention policy, and production
  route-promotion work without claiming C CUDA removal before those gates
  pass.

############################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbba: Public Multi-Token F32 BLAS Projection ABI

- Status: done
- Goal: Rust-own public multi-token F32 projection through the current-C
  `cublasSgemm` boundary without claiming F16 BLAS, Q8/F16 cache,
  quality-mode mutation, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbba/abi-matmul-f32-multi-token-blas-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f32_multi_token_blas_smoke.py --negative-test`.
- Evidence: Rust retains the one-token embedded F32 base kernel and
  dispatches `n_tok > 1` through the cuda-oxide `project_f32` cuBLAS
  adapter with current-C initialization-time default TF32 versus
  `DS4_CUDA_NO_TF32` selection.
  A C-linked B300 consumer verifies one-token base output and two-token BLAS
  output after host-map mutation and after unsetting an initialization-time
  `DS4_CUDA_NO_TF32`, proving cached F32 weights and math selection remain
  authoritative across the widened public symbol. Local tests pass with 123
  tests, B300 release-feature tests pass with 130 tests, the static library
  remains at 32 Rust ABI symbols, and sixteen unaffected predecessor
  C-linked consumers pass against the rebuilt archive; each predecessor
  relink retains the known embedded-object executable-stack warning. The
  historical single-token F32 negative witness is not rerun because this
  successor owns its former rejected boundary. The focused comparator and
  default unified parity report pass with 209 passed, 45 skipped, and 0
  failed. The
  required non-interactive Claude review returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Multi-token F16 BLAS, Q8/F16
  cache hooks, quality-mode mutation, remaining graph compute, whole-archive
  retention policy, route promotion, C CUDA removal, and the
  embedded-object executable-stack warning remain pending.

############################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbb because public
  multi-token F16 BLAS projection and paired delegation have one bounded
  public ABI comparator independently from Q8/F16 caching and route work.
- Goal: connect multi-token F16 BLAS projection, q8/f16 cache hooks,
  quality-mode mutation, remaining graph compute, whole-archive retention
  policy, and production route-promotion work without claiming C CUDA
  removal before those gates pass.

################################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba: Public Multi-Token F16 BLAS Projection ABI

- Status: done
- Goal: Rust-own public multi-token F16 projection and paired delegation
  through current-C-compatible F32-to-F16 activation conversion and
  `cublasGemmEx` without claiming Q8/F16 cache, quality-mode mutation, or
  route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba/abi-matmul-f16-multi-token-blas-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f16_multi_token_blas_smoke.py --negative-test`.
- Evidence: Rust retains synchronously allocated reusable F16 activation scratch storage through
  queued BLAS work, promotes `abi_f32_to_f16_kernel` into the reusable ABI
  module, and sends multi-token direct and paired-delegated F16 projection
  through cuda-oxide `project_f16_f32`, while a serial override retains the
  F32 activation fallback. A C-linked B300 consumer proves cached F16 weight
  authority after host mutation and observes F32-to-F16 activation rounding.
  Local library tests pass with 124 tests; B300 release-feature tests pass
  with 131 tests; the static library remains at 32 exports; fifteen
  unaffected predecessor linked consumers pass, each retaining the known
  executable-stack warning. Earlier single-token F16 rejection witnesses are
  historical after this successor. The focused comparator and default
  unified parity report pass with 210 passed, 45 skipped, and 0 failed.
  Pre-implementation and final pass-end Claude review attempts each returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Q8/F16 cache
  hooks, quality mutation, remaining graph compute, whole-archive retention,
  route promotion, C CUDA removal, and the warning remain pending.

################################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbb because public Q8
  preload, memory-report, and quality controls have a bounded linked
  comparator independently from Q8 matmul and route work.
- Goal: connect q8/f16 cache hooks, quality-mode mutation, remaining graph
  compute, whole-archive retention policy, and production route-promotion
  work without claiming C CUDA removal before those gates pass.

################################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba: Public Q8 Cache And Quality Controls ABI

- Status: done
- Goal: Rust-own public Q8 converted-cache preload, memory reporting, and
  quality-mode BLAS mutation through current-C-compatible policy without
  claiming Q8 matmul or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba/abi-q8-quality-controls-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_q8_quality_controls_smoke.py --negative-test`.
- Evidence: Rust now exports `ds4_gpu_cache_q8_f16_range`,
  `ds4_gpu_print_memory_report`, and `ds4_gpu_set_quality`, retaining
  ABI-owned converted F16/F32 Q8 buffers and using embedded
  `abi_dequant_q8_0_to_f16_kernel` and `abi_dequant_q8_0_to_f32_kernel`.
  Existing multi-token dense BLAS projections consume mutable effective math
  selection after quality changes. A C-linked B300 consumer proves live
  converted-buffer allocation/reuse, quality suppression and re-enable
  allocation, optional F32 preload, callable memory reporting, and distinct
  TF32 versus default-math outputs. Local library tests pass with 125 tests;
  B300 release-feature tests pass with 132 tests; the static library exposes
  35 Rust ABI symbols; sixteen predecessor linked consumers pass with the
  known executable-stack warning. The focused comparator and default
  unified parity report pass with 211 passed, 45 skipped, and 0 failed.
  Pre-implementation and final pass-end non-interactive Claude review
  attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
  completed findings. Q8 matmul compute,
  remaining graph compute, whole-archive retention, route promotion, C CUDA
  removal, and the warning remain pending.

################################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbb because the base
  public Q8 matmul consumer has a bounded linked comparator independently
  from specialized pair/HC graph consumers and route work.
- Goal: connect public Q8 matmul consumers, remaining graph compute,
  whole-archive retention policy, and production route-promotion work
  without claiming C CUDA removal before those gates pass.

################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba: Public Q8 Matmul ABI

- Status: done
- Goal: Rust-own `ds4_gpu_matmul_q8_0_tensor` through current-C-compatible
  expanded BLAS and native prequantized dispatch without claiming Q8
  pair/HC consumers or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba/abi-q8-matmul-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_q8_matmul_smoke.py --negative-test`.
- Evidence: Rust consumes retained Q8 F32/F16 converted ranges for
  multi-token BLAS, retains ABI-owned quantized activation scratch for the
  native routes, and embeds current-C-compatible quantize, warp8,
  batch-warp8, generic, DP4A, and scalar-fallback kernels. A C-linked B300
  consumer proves default DP4A and `DS4_CUDA_NO_Q8_DP4A`, batch-warp and
  `DS4_CUDA_NO_Q8_BATCH_WARP`, opt-in F16 and F32 expanded BLAS, and invalid
  range rejection. Local library tests pass with 126 tests; B300
  release-feature tests pass with 133 tests; the static library exposes 36
  Rust ABI symbols; all 36 preceding linked ABI consumers pass against the
  rebuilt archive with the known executable-stack warning. All 40 CUDA ABI
  comparators pass, and the unified parity report passes with 212 passed, 45
  skipped, and 0 failed. Pre-implementation and final pass-end
  non-interactive Claude review attempts each returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Specialized
  Q8 pair/HC consumers, remaining graph compute, whole-archive retention,
  route promotion, C CUDA removal, and the warning remain pending.

################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbb because the public
  HC expansion kernel family is a bounded prerequisite for fused public Q8
  HC consumers.
- Goal: connect public HC expansion, fused public Q8 HC consumers, remaining graph
  compute, whole-archive retention policy, and production route-promotion
  work without claiming C CUDA removal before those gates pass.

################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba: Public Hyperconnection Expansion ABI

- Status: done
- Goal: Rust-own `ds4_gpu_hc_expand_tensor`,
  `ds4_gpu_hc_expand_split_tensor`, and
  `ds4_gpu_hc_expand_add_split_tensor` through a current-C-compatible
  embedded expansion kernel without claiming fused Q8 HC consumers or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba/abi-hc-expand-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_hc_expand_smoke.py --negative-test`.
- Evidence: Rust now exports the direct, split, and split-plus-add HC
  expansion public ABI and embeds one stride-aware kernel. A C-linked B300
  consumer proves direct and split results, normal and aliased block-add
  behavior, and invalid-input rejection. Local library tests pass with 127
  tests; B300 release-feature tests pass with 134 tests; the static library
  exposes 39 Rust ABI symbols; 37 preceding linked ABI consumers pass against
  the rebuilt archive with the known embedded-object executable-stack warning.
  All 41 CUDA ABI comparators pass, and the unified parity report passes with
  213 passed, 45 skipped, and 0 failed. Pre-implementation and final
  pass-end non-interactive Claude review attempts each returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings. Fused public
  Q8 HC consumers, remaining graph compute, whole-archive retention, route
  promotion, C CUDA removal, and the warning remain pending.

################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbb because the public
  fused Q8 HC consumers share a bounded kernel and fallback contract
  independently from internal pair consumers and route work.
- Goal: connect fused public Q8 HC consumers, remaining graph compute,
  whole-archive retention policy, and production route-promotion work
  without claiming C CUDA removal before those gates pass.

#################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba: Public Fused Q8 Hyperconnection Consumers ABI

- Status: done
- Goal: Rust-own `ds4_gpu_matmul_q8_0_hc_expand_tensor` and
  `ds4_gpu_shared_down_hc_expand_q8_0_tensor` through the fused Q8 HC
  expansion kernel and existing disabled-fusion fallback without claiming
  internal pair consumers or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba/abi-fused-q8-hc-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_fused_q8_hc_smoke.py --negative-test`.
- Evidence: Rust exports both public fused consumers, embeds a fused
  prequantized Q8 HC kernel with DP4A/scalar selection and explicit aliased
  add handling, and retains current-C disabled-fusion delegation through the
  existing public Q8 and HC exports. A C-linked B300 witness proves fused
  DP4A/scalar output, both fallback routes, aliased shared-down add, and
  invalid-shape rejection. Local tests pass with 128 tests; B300 feature
  tests pass with 135 tests; the static library exposes 41 symbols; all 38
  preceding linked ABI consumers pass against the rebuilt archive with the
  known embedded-object executable-stack warning. All 42 CUDA ABI
  comparators pass, and the unified parity report passes with 214 passed, 45
  skipped, and 0 failed. The pre-implementation non-interactive Claude
  review attempt returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`; final pass-end
  Claude review returned `NO BLOCKERS`. The internal Q8 pair consumer,
  remaining graph compute, whole-archive/route promotion, C CUDA removal,
  and the warning remain pending.

#################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbb because the
  public shared gate/up Q8 SwiGLU consumer has a bounded paired-kernel and
  disabled-pair fallback contract independently from later graph work.
- Goal: connect internal pair and remaining graph compute,
  whole-archive retention policy, and production route-promotion work
  without claiming C CUDA removal before those gates pass.

##################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba: Public Shared Gate Up SwiGLU Q8 ABI

- Status: done
- Goal: Rust-own `ds4_gpu_shared_gate_up_swiglu_q8_0_tensor` through a
  private paired-Q8 implementation kernel and the public disabled-pair
  fallback without exporting `ds4_gpu_matmul_q8_0_pair_tensor` or claiming
  route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba/abi-shared-gate-up-swiglu-q8-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_shared_gate_up_swiglu_q8_smoke.py --negative-test`.
- Evidence: Rust exports the public wrapper, embeds the paired
  prequantized-Q8 kernel with DP4A/scalar selection, and preserves
  `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR` delegation through existing Q8
  matmul and SwiGLU exports. A C-linked B300 witness proves paired DP4A and
  scalar output, disabled-pair fallback, clamped SwiGLU output, and invalid
  range rejection. Local tests pass with 129 tests; B300 feature tests pass
  with 136 tests; the static library exposes 42 symbols; all 39 preceding
  linked ABI consumers pass against the rebuilt archive with the known
  embedded-object executable-stack warning. All 43 CUDA ABI comparators
  pass, and the unified parity report passes with 215 passed, 45 skipped,
  and 0 failed. The pre-implementation non-interactive Claude review attempt
  returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`; final pass-end Claude review
  returned `NO BLOCKERS`. Remaining graph compute, whole-archive/route
  promotion, C CUDA removal, and the warning remain pending.

##################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbb because direct and
  split-stride hyperconnection weighted-sum reductions form a bounded public
  ABI leaf independently from Sinkhorn and later fused reductions.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

###################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba: Public Hyperconnection Weighted Sum ABI

- Status: done
- Goal: Rust-own `ds4_gpu_hc_weighted_sum_tensor` and
  `ds4_gpu_hc_weighted_sum_split_tensor` through one stride-aware embedded
  reduction kernel without claiming Sinkhorn, fused reductions, or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba/abi-hc-weighted-sum-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_hc_weighted_sum_smoke.py --negative-test`.
- Evidence: Rust exports both weighted-sum wrappers through
  `abi_hc_weighted_sum_kernel`, retaining valid current-C output-token
  derivation while validating accessed residual and split spans. A C-linked
  B300 witness proves direct and split-stride output plus short-input and
  zero-shape rejection. Local tests pass with 130 tests; B300 feature tests
  pass with 137 tests; the static library exposes 44 symbols; all 40
  preceding linked ABI consumers pass against the rebuilt archive with the
  known embedded-object executable-stack warning. All 44 CUDA ABI comparators
  pass, and the unified parity report passes with 216 passed, 45 skipped,
  and 0 failed. The pre-implementation and final pass-end non-interactive
  Claude review attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  Sinkhorn, fused reductions, remaining graph compute, whole-archive/route
  promotion, C CUDA removal, and the warning remain pending.

###################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbb because public
  split-Sinkhorn generation is a bounded model-backed ABI leaf independently
  from fused split-weighted reductions.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

####################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba: Public Hyperconnection Split Sinkhorn ABI

- Status: done
- Goal: Rust-own `ds4_gpu_hc_split_sinkhorn_tensor` through its four-branch
  mixer transform and model-backed scale/base inputs without claiming fused
  reductions or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba/abi-hc-split-sinkhorn-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_hc_split_sinkhorn_smoke.py --negative-test`.
- Evidence: Rust exports the split-Sinkhorn wrapper through
  `abi_hc_split_sinkhorn_kernel` and private `abi_hc4_split_one` math,
  resolving scale/base parameters through cached model ranges while retaining
  current-C full-row flooring. A C-linked B300 witness proves two-row output,
  shorter-output flooring, alternate parameter ranges, and invalid-input
  rejection. Local tests pass with 131 tests; B300 feature tests pass with
  138 tests; the static library exposes 45 symbols; all 41 preceding linked
  ABI consumers pass against the rebuilt archive with the known embedded
  object executable-stack warning. All 45 CUDA ABI comparators pass, and the
  unified parity report passes with 217 passed, 45 skipped, and 0 failed. The
  pre-implementation and final pass-end non-interactive Claude review attempts
  each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`. Fused split-weighted
  reductions, remaining graph compute,
  whole-archive/route promotion, C CUDA removal, and the warning remain
  pending.

####################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbb because the
  public fused split-weighted reduction is a bounded ABI leaf independently
  from normalized reduction and output-head HC weights.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

######################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba: Public Hyperconnection Split Weighted Sum ABI

- Status: done
- Goal: Rust-own `ds4_gpu_hc_split_weighted_sum_tensor` through one
  synchronized embedded split-and-reduction kernel without claiming the
  normalized fused consumer, output-head HC weights, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba/abi-hc-split-weighted-sum-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_hc_split_weighted_sum_smoke.py --negative-test`.
- Evidence: Rust exports the fused wrapper through
  `abi_hc_split_weighted_sum_fused_kernel`, retaining output-defined full-row
  launch behavior while validating all accessed input spans and cached
  scale/base model ranges. A C-linked B300 witness proves fused output,
  emitted split values, alternate parameter ranges, output-defined row count,
  and invalid-input rejection. Local tests pass with 132 tests; B300 feature
  tests pass with 139 tests; the static library exposes 46 symbols; all 42
  preceding linked ABI consumers pass against the rebuilt archive with the
  known embedded-object executable-stack warning. All 46 CUDA ABI comparators
  pass, and the unified parity report passes with 218 passed, 45 skipped,
  and 0 failed. The pre-implementation and final pass-end non-interactive
  Claude review attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  Normalized fused reduction, output HC weights, remaining graph compute,
  whole-archive/route promotion, C CUDA removal, and the warning remain
  pending.

######################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbb because the
  normalized fused HC consumer and its fallback policy form a bounded public
  ABI leaf independently from output-head HC weights.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

######################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba: Public Hyperconnection Split Weighted Sum Norm ABI

- Status: done
- Goal: Rust-own `ds4_gpu_hc_split_weighted_sum_norm_tensor` through its
  one-row fused kernel and current-C disabled-or-multi-row fallback without
  claiming output-head HC weights or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba/abi-hc-split-weighted-sum-norm-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_hc_split_weighted_sum_norm_smoke.py --negative-test`.
- Evidence: Rust exports the normalized fused wrapper through
  `abi_hc_split_weighted_sum_norm_fused_kernel`, preserves the current-C
  disabled-or-multi-row fallback through existing public exports, and
  validates the fused model ranges. A C-linked B300 witness proves one-row
  split/output/norm, multi-row first-normalized-row fallback,
  disabled-fusion fallback, alternate parameter ranges, and invalid-input
  rejection. Local tests pass with 133 tests; B300 feature tests pass with
  140 tests; the static library exposes 47 symbols; all 43 preceding linked
  ABI consumers pass against the rebuilt archive with the known embedded
  object executable-stack warning. All 47 CUDA ABI comparators pass, and the
  unified parity report passes with 219 passed, 45 skipped, and 0 failed. The
  pre-implementation and final pass-end non-interactive Claude review
  attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`. Output HC
  weights, remaining graph compute, whole-archive/route promotion, C CUDA
  removal, and the warning remain pending.

######################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbb because public
  output-head hyperconnection weight generation is a bounded ABI leaf
  independently from remaining graph compute and route promotion.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

########################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba: Public Output Hyperconnection Weights ABI

- Status: done
- Goal: Rust-own `ds4_gpu_output_hc_weights_tensor` through its public
  sigmoid-plus-eps output-weight kernel without claiming remaining graph
  compute or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba/abi-output-hc-weights-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_output_hc_weights_smoke.py --negative-test`.
- Evidence: Rust exports the output-weight wrapper through
  `abi_output_hc_weights_kernel`, deriving complete output rows and resolving
  scale/base inputs through cached model ranges. A C-linked B300 witness
  proves multi-token and single-token row-derived output, alternate parameter
  ranges, and invalid-input rejection. Local tests pass with 134 tests; B300
  feature tests pass with 141 tests; the static library exposes 48 symbols;
  all 44 preceding linked ABI consumers pass against the rebuilt archive with
  the known embedded-object executable-stack warning. All 48 CUDA ABI
  comparators pass, and the unified parity report passes with 220 passed, 45
  skipped, and 0 failed. The pre-implementation and final pass-end
  non-interactive Claude review attempts each returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`. Rust rejects output token counts above
  `u32::MAX` instead of retaining current-C launch-argument narrowing.
  Remaining graph compute,
  whole-archive/route promotion, C CUDA removal, and the warning remain
  pending.

########################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbb because the
  public single-token and batched embedding wrappers form a bounded ABI leaf
  independently from remaining graph compute and route promotion.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

########################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba: Public Embedding Hyperconnection ABI

- Status: done
- Goal: Rust-own `ds4_gpu_embed_token_hc_tensor` and
  `ds4_gpu_embed_tokens_hc_tensor` through their public FP16 embedding
  kernels without claiming remaining graph compute or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba/abi-embedding-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_embedding_smoke.py --negative-test`.
- Evidence: Rust exports both embedding wrappers through embedded kernels,
  validating public tensor spans and consuming cached FP16 model rows. A
  C-linked B300 witness proves single-token replication, batched invalid-ID
  fallback to row zero, alternate model ranges, and invalid-input rejection.
  Local tests pass with 135 tests; B300 feature tests pass with 142 tests;
  the static library exposes 50 symbols; all 45 preceding linked ABI
  consumers pass against the rebuilt archive with the known executable-stack
  warning. All 49 CUDA ABI comparators pass, and the unified parity report
  passes with 221 passed, 45 skipped, and 0 failed. Rust rejects
  out-of-vocabulary single-token calls, short
  single-token output, and zero-dimensional or overflowing launches instead
  of retaining current-C unchecked or undefined behavior. The
  pre-implementation and final pass-end non-interactive Claude review
  attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`. Remaining graph
  compute, whole-archive/route promotion, C CUDA
  removal, and the warning remain pending.

########################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbb because
  public in-place head RMS normalization is independently comparable before
  RoPE, KV, attention, compressor, and route work.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

############################################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbba: Public Head RMS Norm ABI

- Status: done
- Goal: Rust-own `ds4_gpu_head_rms_norm_tensor` through its public in-place
  reduction kernel without claiming RoPE, remaining graph compute, or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbba/abi-head-rms-norm-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_head_rms_norm_smoke.py --negative-test`.
- Evidence: Rust exports the in-place public head RMS wrapper through an
  embedded 256-thread reduction kernel. A C-linked B300 witness proves
  multi-row normalization, short-tensor rejection, zero-dimension
  rejection, and null rejection. Local tests pass with 136 tests; B300
  feature tests pass with 143 tests; the static library exposes 51 symbols;
  all 46 preceding linked ABI consumers pass against the rebuilt archive
  with the known executable-stack warning. Rust rejects zero-dimensional and
  oversized launch grids rather than retaining current-C undefined launch
  behavior. All 50 CUDA ABI comparators pass, and the unified parity report
  passes with 222 passed, 45 skipped, and 0 failed. The pre-implementation
  and final pass-end non-interactive Claude review attempts each returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.

############################################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbb because
  public FP8 KV prefix quantization is independently comparable before
  standalone RoPE, KV storage, compressor, attention, and route work.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

############################################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbba: Public FP8 KV Quantization ABI

- Status: done
- Goal: Rust-own `ds4_gpu_dsv4_fp8_kv_quantize_tensor` through its public
  in-place E4M3FN prefix quantization kernel without claiming standalone
  RoPE, raw KV storage, compressor, attention, remaining graph compute, or
  route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbba/abi-fp8-kv-quantize-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_fp8_kv_quantize_smoke.py --negative-test`.
- Evidence: Rust exports the in-place FP8 KV wrapper through an embedded
  64-thread E4M3FN prefix kernel. A C-linked B300 witness proves prefix
  output, partial-chunk handling, untouched RoPE tail, empty-prefix and
  zero-width no-op behavior, and invalid-input rejection. Local tests pass
  with 137 tests; B300 feature tests pass with 144 tests; the static library
  exposes 52 symbols; all 47 preceding linked ABI consumers pass against the
  rebuilt archive with the known executable-stack warning. Rust rejects
  zero-token launches before current-C's invalid zero-grid submission while
  preserving no-op dimensions. All 51 CUDA ABI comparators pass, and the
  unified parity report passes with 223 passed, 45 skipped, and 0 failed.
  The pre-implementation and final pass-end non-interactive Claude review
  attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.

############################################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbb
  because public indexer Hadamard/FP4 QAT is independently comparable before
  standalone RoPE, KV storage, compressor, attention, routed MoE, and route
  work.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

############################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbba: Public Indexer QAT ABI

- Status: done
- Goal: Rust-own `ds4_gpu_dsv4_indexer_qat_tensor` through its public
  in-place normalized Hadamard plus E2M1FN block-quantization kernel without
  claiming standalone RoPE, raw KV storage, compressor, attention, routed
  MoE, remaining graph compute, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbba/abi-indexer-qat-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_indexer_qat_smoke.py --negative-test`.
- Evidence: Rust exports the in-place public indexer QAT wrapper through an
  embedded 128-thread normalized Hadamard plus E2M1FN kernel. A C-linked
  B300 witness proves two-row transformed output, per-32-value FP4 scaling,
  short-tensor rejection, invalid-shape rejection, and null rejection. Local
  tests pass with 138 tests; B300 feature tests pass with 145 tests; the
  static library exposes 53 symbols; all 48 preceding linked ABI consumers
  pass against the rebuilt archive with the known executable-stack warning.
  All 52 CUDA ABI comparators pass, and the unified parity report passes with
  224 passed, 45 skipped, and 0 failed.
  The pre-implementation and final pass-end non-interactive Claude review
  attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.

############################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbb
  because public standalone RoPE is independently comparable before raw KV
  storage, compressor, attention, routed MoE, and route work.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

############################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbba: Public Standalone RoPE ABI

- Status: done
- Goal: Rust-own `ds4_gpu_rope_tail_tensor` through its public unit-stride
  rotary-tail kernel without claiming raw KV storage, compressor, attention,
  routed MoE, remaining graph compute, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbba/abi-rope-tail-smoke.json`
- Comparator:
  `ds4-parity/check_cuda_abi_rope_tail_smoke.py --negative-test`.
- Evidence: Rust exports the public standalone RoPE wrapper through an
  embedded rotary-tail kernel preserving valid interpolation, inverse, and
  YaRN behavior. A C-linked B300 witness proves interpolated forward output,
  inverse YaRN output, untouched non-RoPE prefixes, zero-pair rejection,
  invalid-shape rejection, and null rejection. Local tests pass with 139
  tests; B300 feature tests pass with 146 tests; the static library exposes
  54 symbols and embeds 32 kernels; all 49 preceding linked ABI consumers
  pass against the rebuilt archive with the known executable-stack warning.
  All 53 CUDA ABI comparators pass, and the unified parity report passes with
  225 passed, 45 skipped, and 0 failed. Rust preserves current-C widened row
  addressing and rejects zero-grid and overflowing pair construction rather
  than retaining current-C invalid launch behavior. The pre-implementation
  and final pass-end non-interactive Claude review attempts each returned
  `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.

############################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.
