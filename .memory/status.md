# DS4 Rust Port Status

- Date: 2026-05-22 UTC
- Branch: `main`
- Starting oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Active item: M7.4b KV Extension Trailer Payload Coverage
- Last validated source commit: M7.4a generic KVC full-file round trip in this
  commit; prior pushed source commit
  `eb331100047ca97a845d879d4eb4d4c6828ab9cf`
- Active debugging ledger: none
- B300 context: `hou2-prod1`
- B300 namespace: `default`
- B300 pod: `ds4-rust-port-b300`
- B300 node: `c1v17-b300n1-nic1`
- B300 temp kubeconfig: `/tmp/ds4-hou2-prod1.kubeconfig` for this local
  session; regenerate a temp copy in future sessions instead of treating this
  path as durable, and pass `--context hou2-prod1` explicitly because the temp
  kubeconfig can contain other contexts.
- Known local validation constraint: `ds4flash.gguf` is not present in the
  workspace, so model-backed tests and benchmark baselines need a model path or
  remote B300 execution.
- B300 model path: `/workspace/ds4/ds4flash.gguf`
- B300 model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- B300 model size: 86,720,111,488 bytes.

## Last Evidence

- `git status --short` was clean before M0.1 edits.
- `AGENT.md`, `CONTRIBUTING.md`, and `RUST_PORT_ROADMAP.md` were read before
  creating the protocol.
- M0.1 validation passed with `git diff --name-only` and `git diff --check`.
- M0.1 Claude review returned PASS before commit.
- M0.2 local arm64 validation captured `arch -arm64 make` exit 0,
  `arch -arm64 make cpu` exit 0, `./ds4_test --server` exit 0, and
  `./ds4_test --metal-kernels` exit 0.
- M0.2 local default `make` and model-backed `make test` failures are recorded
  in `ds4-parity/baselines/manifest.md` with exact logs.
- M0.2 B300 validation captured `make cuda-generic` exit 0,
  `make cuda-regression` exit 0, `./ds4_test --server` exit 0, and
  `./ds4_test --metal-kernels` exit 0 on `ds4-rust-port-b300`.
- M0.3 B300 validation downloaded q2-imatrix, recorded model hash/size, built
  `ds4_test`, and captured `./ds4_test --logprob-vectors` exit 0 with
  `logprob-vectors: OK`.
- M0.4 B300 validation refreshed source commit
  `3d87577962abeac1ab0d80e9c21d0012bfc53afb`, built `ds4-server`, and replayed
  six server fixtures from `ds4-parity/baselines/server-fixtures/m0.4/` with
  HTTP 200 for all requests.
- M0.4 artifacts live under `ds4-parity/baselines/server-traces/m0.4/`; the
  final trace records non-streaming chat, SSE chat, DSML-to-OpenAI tool calls,
  explicit thinking-disabled chat, and cache continuation with
  `cache_source=memory-token`, `cached_tokens=41`, `cache_write_tokens=9`.
- M0.5 B300 validation refreshed source commit
  `0623bbb4d97d056a58e208e324216f97abed685e`, built `ds4-server`, and replayed
  three disk-KV server lifetimes from
  `ds4-parity/baselines/kv-fixtures/m0.5/` with HTTP 200 for all requests.
- M0.5 artifacts live under `ds4-parity/baselines/kv-artifacts/m0.5/`; the
  replay records a cold 550-token cache write, a fresh-process 550-token
  `disk-text` restore, and a fresh-process continuation restore of the
  552-token shutdown prefix with a 9-token suffix write.
- M0.5 raw `.kv` files are not checked in; committed comparator artifacts
  include full raw hashes, timestamp-normalized hashes, parsed KVC headers, and
  extracted rendered cache text.
- M0.6 B300 validation refreshed source commit
  `add2c507f81aa2e363809213771134c282c50bf2`, built `ds4-bench`, and captured
  short-context and long-context CSV baselines using
  `speed-bench/promessi_sposi.txt` with SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- M0.6 artifacts live under `ds4-parity/baselines/bench/m0.6/`; the short CSV
  covers 2048 through 8192 tokens and the long CSV covers 16384 through 32768
  tokens, both with 32 greedy generation tokens per frontier.
- M1.1 documented the Milestone 1 implementation work items in
  `RUST_PORT_ROADMAP.md`; the next executable item is a static verifier for the
  committed Milestone 0 artifacts.
- M1.2 added `python3 ds4-parity/verify_baselines.py`, which verifies M0.2
  through M0.6 artifact families locally without rerunning model-backed
  commands. Its negative test corrupts a copied benchmark CSV and requires the
  verifier to detect the drift.
- M1.3 added `python3 ds4-parity/compare_server_kv.py`, which self-compares
  committed M0.4 server and M0.5 KV artifacts with only documented
  normalizations. Its negative test covers finish reason, cached token count,
  cache source, KV reason, and rendered text drift.
- M1.4 added `python3 ds4-parity/compare_logprob_numeric.py`, which parses the
  compact official-vector fixture, audits it against raw official API JSON,
  verifies the M0.3 B300 pass markers, and compares candidate vector files with
  exact selected tokens plus a reported 4.0 absolute logprob tolerance. Its
  negative test covers selected-token drift and numeric drift outside tolerance.
- M1.5 added `python3 ds4-parity/compare_bench_csv.py`, which self-compares
  committed M0.6 benchmark CSV artifacts, validates capture metadata for
  threshold use, requires exact workload shape and KV byte counts, and applies
  the documented 10% throughput regression threshold. Its negative test covers
  schema, context frontier, generation-token, cache-byte, and throughput drift.
- M1.6 added `python3 ds4-parity/run_parity_report.py`, which runs local
  no-model C oracles, invokes the M1.2 through M1.5 comparator commands, and
  reports skipped B300/model-backed oracle refreshes with explicit
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1` rerun
  commands. The local report passed with 9 executed items and 4 B300 refreshes
  skipped by design.
- M2.1 added a Rust workspace with `ds4-gpu-sys` and `ds4-gpu`, seeded core
  tensor/command/model-map FFI declarations, added smoke-only safe status
  wrappers, and wired `make rust-test`. Validation passed for `cargo fmt`,
  `cargo test --workspace`, `make rust-test`, sequential `arch -arm64 make`,
  sequential `arch -arm64 make cpu`, and the unified parity report.
- M3.1 added safe Rust `Tensor`, `TensorView`, and `CommandBatch` wrappers over
  the existing `ds4_gpu.h` tensor/command ABI without changing the C ABI. The
  macOS `ds4-gpu` build script compiles the current `ds4.c` and `ds4_metal.m`
  backend objects into a test archive so Rust tests call the real Metal
  implementation rather than a mock.
- M3.1 Rust ABI parity validation passed with
  `cargo test -p ds4-gpu safe_tensor_wrapper_matches_direct_c_abi -- --nocapture`;
  the test compares safe-wrapper and direct-C paths for allocation, byte-size
  queries, write/read, fill, view writes, command-batched copy, flush/end,
  synchronize, and out-of-bounds write/copy failures.
- M3.1 full validation passed for `cargo fmt --all -- --check`,
  `cargo test --workspace`, `make rust-test`, `git diff --check`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make cpu`, and
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed.
- M4.1 split Milestone 4 into concrete GGUF/model-metadata work items after
  reading `ds4.c` loader, summary, metadata validation, base tensor binding,
  and MTP tensor binding surfaces. The next executable item is the current-C
  metadata dump oracle.
- M4.2 added `./ds4-metadata-dump`, which opens the model through the current C
  GGUF loader, runs `config_validate_model` and `weights_bind`, and emits
  deterministic `ds4.metadata.v1` JSON with selected metadata values, tensor
  type histograms, all tensor descriptors, and bound semantic tensor roles.
- M4.2 added `python3 ds4-parity/check_metadata_dump.py`, whose schema checker
  validates the dump and whose negative test detects tensor-count drift, a
  missing required bound role, and a missing required metadata key.
- M4.2 B300 validation copied the M4.2 source files into
  `/workspace/ds4`, built with `make clean ds4-metadata-dump CUDA_ARCH=native`,
  dumped `/workspace/ds4/ds4flash.gguf`, and passed
  `python3 ds4-parity/check_metadata_dump.py /tmp/ds4-metadata.json --negative-test`.
  The generated B300 dump had 633,297 bytes, SHA256
  `39ad79574b19421e2c470a055376258b9415eb1f429188426cfd2860688a2a2f`,
  1,328 tensors, and 1,511 bound tensor roles.
- M4.2 local validation passed for `arch -arm64 make ds4-metadata-dump`,
  `./ds4-metadata-dump --help`, local schema/negative checks against the copied
  B300 dump, `python3 -m py_compile ds4-parity/check_metadata_dump.py`,
  sequential `arch -arm64 make clean`, sequential `arch -arm64 make`,
  sequential `arch -arm64 make clean`, sequential `arch -arm64 make cpu`,
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed,
  `cargo test --workspace`, and `git diff --check`.
- M4.3 added a dependency-free `ds4-gguf` Rust crate and `ds4-gguf-dump` CLI
  that parse GGUF v3 metadata and tensor directory records, compute C-equivalent
  tensor byte sizes and aligned absolute offsets, and emit the same
  `ds4.metadata.v1` directory surface as the C metadata dump.
- M4.3 added `./ds4-metadata-dump --directory-only` so local synthetic GGUF
  fixtures can compare the current C GGUF directory parser against Rust without
  requiring the full DS4 model or semantic tensor binding.
- M4.3 added `python3 ds4-parity/compare_gguf_directory.py`, whose synthetic
  fixture covers scalar metadata, array metadata, non-power-of-two
  `general.alignment=48`, F32 byte sizing, Q8_0 block byte sizing, relative and
  absolute offsets, C-compatible float metadata formatting, unsupported scalar
  metadata emission as `null`, and C/Rust rejection of corrupted magic,
  truncated metadata, truncated tensor data, and tensor offset overflow.
- M4.3 B300 check confirmed the pod does not currently have `rustc` or `cargo`,
  so this item used local synthetic C-vs-Rust directory comparison instead of a
  B300 Rust run. Real supported-model Rust comparison remains deferred to the
  roadmap item that provides Rust on the model host or transfers feasible dump
  artifacts.
- M4.3 local validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests,
  `python3 ds4-parity/compare_gguf_directory.py --negative-test` with 14
  checks, `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_gguf_directory.py ds4-parity/check_metadata_dump.py`,
  local schema/negative checks against the copied B300 M4.2 dump,
  `cargo test --workspace`, `git diff --check`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make cpu`, and
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed.
- M4.4 added `./ds4-metadata-dump --validate-config-only`, which runs current C
  `config_validate_model` after GGUF parsing but skips tensor binding, making
  local metadata-only validation fixtures possible without the full model
  tensor table.
- M4.4 added `validate_ds4_metadata` in `ds4-gguf`, matching C required-key
  behavior, `u64` and `f32` coercion rules, optional expert group defaults,
  fixed DeepSeek4 constants, compression-ratio arrays, SwiGLU clamp arrays,
  RoPE constants, HC constants, RMS epsilon, expert weight scale, and expert
  weight normalization.
- M4.4 added `python3 ds4-parity/compare_metadata_validation.py`, whose
  synthetic fixtures compare C and Rust pass/fail behavior and normalized first
  failures for baseline metadata, C-compatible numeric coercions, missing keys,
  wrong scalar types, wrong scalar values, short arrays, negative compression
  ratios, wrong compression ratios, float tolerance failures, non-integer
  `u64` inputs, non-float `f32` inputs, and boolean drift.
- M4.4 focused validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests, `python3
  ds4-parity/compare_metadata_validation.py --negative-test` with 41 checks,
  and `python3 -m py_compile ds4-parity/compare_metadata_validation.py`.
- M4.5 added `./ds4-metadata-dump --validate-layout-only`, which runs current C
  metadata validation plus base/MTP tensor binding and layout validation from
  GGUF directories while skipping tensor payload range checks for synthetic
  local fixtures.
- M4.5 added Rust base and MTP tensor binding/layout validation in `ds4-gguf`,
  including required, optional, compression-ratio-dependent, hash-layer-only,
  plain F16/F32 MTP, routed expert quant-type, routed gate/up type equality,
  and fixed tensor dimension rules.
- M4.5 added `python3 ds4-parity/compare_tensor_bindings.py`, whose synthetic
  fixtures compare C and Rust layout dumps for base plus MTP bindings and
  negative cases for missing required tensors, wrong types, wrong dimensions,
  optional tensor type drift, routed expert type drift, routed gate/up type
  mismatch, missing compressor/indexer tensors, MTP plain-type rejection, and
  missing required MTP tensors.
- M4.5 focused validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests, `python3
  ds4-parity/compare_tensor_bindings.py --negative-test` with 33 checks, and
  `python3 -m py_compile ds4-parity/compare_tensor_bindings.py`.
- M4.6 recaptured the supported-model metadata baseline on B300
  `ds4-rust-port-b300` in `hou2-prod1` after refreshing `ds4.c`, `ds4.h`, and
  `ds4_metadata_dump.c` from source commit
  `58bad019226499d5b294340093f77c70b7250b79`.
- M4.6 committed `ds4-parity/baselines/metadata/m4.6/current-c.json` for
  `/workspace/ds4/ds4flash.gguf`, whose resolved path is
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
  model size is 86,720,111,488 bytes, model SHA256 is
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  dump size is 633,297 bytes, and dump SHA256 is
  `39ad79574b19421e2c470a055376258b9415eb1f429188426cfd2860688a2a2f`.
- M4.6 added `python3 ds4-parity/compare_metadata_baseline.py --negative-test`,
  which schema-checks the committed baseline, verifies manifest artifact hashes,
  normalizes model paths/source for candidate comparisons, and detects scalar
  metadata, array metadata, tensor shape, tensor type, binding, and offset
  drift.
- M4.6 wired the metadata baseline comparator into
  `python3 ds4-parity/run_parity_report.py` and added a B300 skip entry with
  exact source-refresh, capture, hash, schema-check, and copy-back commands.
- M4.7 added `python3 ds4-parity/compare_gguf_failures.py`, a generated
  malformed-GGUF matrix that compares C and Rust rejection status plus
  normalized first-error categories for invalid magic, unsupported version,
  truncated metadata, unknown metadata type, bad tensor dimension, out-of-file
  tensor data, tensor offset overflow, missing required metadata, wrong
  metadata type, bad metadata array length, and unsupported DS4 tensor type.
- M4.7 validation passed for `arch -arm64 make ds4-metadata-dump`,
  `python3 ds4-parity/compare_gguf_failures.py` with 55 checks,
  `python3 ds4-parity/compare_gguf_failures.py --list-cases`, M4.3 through
  M4.5 comparators (`compare_gguf_directory.py --negative-test`,
  `compare_metadata_validation.py --negative-test`, and
  `compare_tensor_bindings.py --negative-test`), `python3 -m py_compile` for
  all involved comparators, and `cargo test --workspace`.
- M5.1 split Milestone 5 into M5.2 through M5.7 after reading tokenizer source
  (`vocab_load`, JoyAI `bpe_tokenize_text`, rendered-chat special tokenization,
  `ds4_token_text`, and `ds4_dump_text_tokenization`), CLI prompt paths
  (`--dump-tokens`, `build_prompt`, and REPL append functions), server prompt
  and API paths (`parse_chat_request`, `render_chat_prompt_text`,
  `render_live_tool_tail`, and DSML formatting/parsing helpers), the agent DSML
  streaming parser, and existing M0.3/M0.4/M0.5 fixtures.
- M5.1 validation passed for `git diff --check` and non-interactive Claude
  review after tightening tokenizer identity, server-vs-CLI prompt oracles,
  token decoding ownership, DSML chunk/EOF parser schedules, tool-schema
  fixture variants, and request body hashing; final Claude review returned
  `NO BLOCKERS`.
- M5.2 added current-C tokenizer and prompt oracle dumping through
  `./ds4-server --dump-token-oracle`, with tokenizer identity hashing in
  `ds4_engine_dump_tokenizer_identity_json`, shared `ds4_sha256_hex`, raw
  request-body hashing, server prompt/token fixtures, and CLI `ds4_chat_*`
  operation/token-stream fixtures. The dump mode opens the model through the
  existing engine path but exits before session/listener/worker startup, and
  advisory token text emission preserves valid UTF-8 while escaping invalid
  raw bytes so future byte-fallback fixtures still produce valid JSON.
- M5.2 committed
  `ds4-parity/baselines/tokenization/m5.2/current-c.json` captured on B300
  `ds4-rust-port-b300` from `/workspace/ds4/ds4flash.gguf`; dump size is
  124,497 bytes and dump SHA256 is
  `b0689f47abe63750ab3191772d5661d5f0f433e954bcfd0de6a0e55a747489e9`.
  The tokenizer identity records 129,280 tokens, token-bytes SHA256
  `c92251fc634ff01cc6767d2e3ce1d368e72b5f02b647983d4410eb0c46693fa3`,
  127,741 merge records, merge-pairs SHA256
  `8100a9693dc10b8aad79abbe20b172545ff5e1e6051e0705cc91e73b88e3751f`,
  the seven rendered-control specials, and 863 literal-special tokens.
- M5.2 B300 validation passed after copying the changed source/checker into
  `/workspace/ds4`, building with `make clean ds4-server CUDA_ARCH=native`,
  dumping the oracle from the q2-imatrix model, and running
  `python3 ds4-parity/check_tokenization_dump.py
  /tmp/ds4-tokenization-m5.2-current-c.json --negative-test`; the final B300
  checker reported `tokenization schema: PASS, 13409 checks` and
  `tokenization negative tests: PASS, 12 checks`.
- M5.2 local validation passed for
  `python3 ds4-parity/check_tokenization_dump.py
  ds4-parity/baselines/tokenization/m5.2/current-c.json --manifest
  ds4-parity/baselines/tokenization/m5.2/manifest.json --negative-test`,
  with `tokenization schema: PASS, 13409 checks`, `tokenization manifest:
  PASS, 11 checks`, and `tokenization negative tests: PASS, 12 checks`;
  `python3 -m py_compile ds4-parity/check_tokenization_dump.py`,
  `./ds4_test --server`, `git diff --check`, `arch -arm64 make ds4-server`,
  `cargo test --workspace`, and `arch -arm64 make cpu` also passed.
- M5.2 Claude review returned `NO BLOCKERS`; after hardening invalid UTF-8
  token text escaping and checker pinning for exact special/server semantics,
  the follow-up Claude review also returned `NO BLOCKERS`.
- M5.3 added `Ds4Tokenizer` to `ds4-gguf`, loading
  `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` from GGUF metadata,
  computing the same canonical token/merge SHA256 identity as C, validating
  required DS4 special token IDs, porting JoyAI plain-text pre-tokenization and
  byte-level BPE merge ranking, and decoding ordinary token pieces through the
  GPT-2 byte mapping used by `ds4_token_text`.
- M5.3 added `ds4-tokenizer-dump` for fixed plain-text cases and
  `python3 ds4-parity/compare_tokenizer_text.py`, which compares Rust token
  IDs and decoded token-piece bytes against the M5.2 current-C `text_cases`.
  Its negative tests cover missing token table, missing merges, token-bytes
  hash drift, merge hash drift, missing required special token, invalid UTF-8
  token strings, and merge-rank drift.
- M5.3 B300 extraction copied `ds4-parity/extract_tokenizer_fixture.py` to
  `ds4-rust-port-b300` and wrote
  `/tmp/ds4-tokenization-m5.3/tokenizer.gguf` from
  `/workspace/ds4/ds4flash.gguf`. The committed tokenizer-only GGUF fixture has
  129,280 tokens, 127,741 merges, size 4,722,720 bytes, and SHA256
  `b1e0d128bde9ea996fee335c9662e93707d2a68decaeb47a8dc5fb902bdbb025`.
- M5.3 local validation passed for `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 8 tests,
  `python3 -m py_compile ds4-parity/extract_tokenizer_fixture.py
  ds4-parity/compare_tokenizer_text.py`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 51 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 7 checks`.
- M5.3 Claude review returned `NO BLOCKERS` after checking the Rust tokenizer
  against the C byte encoding, JoyAI split rules, BPE merge loop, token text
  decoding, and comparator scope.
- M5.4 added Rust rendered-chat tokenization over the exact C
  `special_token_at` rendered-control table. `tokenize_rendered_chat` scans
  trusted rendered prompt bytes for BOS, EOS, User, Assistant, `<think>`,
  `</think>`, and `｜DSML｜`, emits those special token IDs, and tokenizes
  intervening spans through the existing JoyAI BPE path; plain `tokenize_text`
  remains separate so special-looking user text is not trusted as control text.
- M5.4 extended `ds4-tokenizer-dump` and
  `python3 ds4-parity/compare_tokenizer_text.py` to compare the M5.2
  `rendered_chat_cases` exactly for rendered prompt bytes, token IDs, and
  decoded token-piece bytes. Negative checks now include rendered special-token
  ID drift and rendered ordinary-piece drift.
- M5.4 local validation passed for `cargo fmt --all -- --check`,
  `cargo test --workspace` with 9 `ds4-gguf` tests,
  `python3 -m py_compile ds4-parity/compare_tokenizer_text.py`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 71 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 9 checks`.
- M5.4 Claude review returned `NO BLOCKERS`; after adding Rust dump `mode`
  fields and pinning them in the comparator, the follow-up Claude review also
  returned `NO BLOCKERS`.
- M5.5 added a Rust prompt renderer matching C `render_chat_prompt_text` for
  the committed M5.2 server prompt cases: tool schemas before system text,
  system/developer aggregation, user/tool/function message handling, assistant
  history turns, thinking disabled/high/max prefixes, DSML tool-call rendering,
  escaped tool-result closing tags, and pending assistant prefixes.
- M5.5 added direct Rust CLI token construction for the M5.2 `ds4_chat_*`
  operation fixtures, covering begin, Think Max prefix append, system/developer
  direct text, user/tool/function messages, assistant content, and assistant
  prefixes for high/max/none thinking modes.
- M5.5 extended `ds4-tokenizer-dump` and
  `python3 ds4-parity/compare_tokenizer_text.py` to compare every M5.2
  `server_request_cases` prompt byte string, rendered token IDs, decoded token
  pieces, CLI operation sequence, and CLI token stream. Negative checks now
  include server prompt-byte drift, server token-ID drift, CLI operation drift,
  and CLI token-piece drift.
- M5.5 local validation passed for `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 11 tests, `cargo test --workspace`,
  `./ds4_test --server`, `python3 -m py_compile
  ds4-parity/compare_tokenizer_text.py`, `git diff --check`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 154 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 13 checks`.
- M5.5 Claude review returned `NO BLOCKERS` after checking Rust prompt
  rendering against C role handling, thinking branches, DSML/tool-result
  escaping, CLI token construction, and comparator coverage.
- M5.6 was split into M5.6a and M5.6b before implementation because server
  generated-message DSML parsing and agent incremental DSML streaming have
  different oracle surfaces and comparator shapes. M5.6a owns server DSML
  formatting plus `parse_generated_message_ex`; M5.6b owns `agent_dsml_parse`
  chunk schedules and streaming state/event parity.
- M5.6 split validation passed for docs-only `git diff --name-only`, `git
  diff --check`, and non-interactive Claude review. Claude returned
  `NO BLOCKERS`.
- M5.6a added `./ds4-server --dump-dsml-oracle`, a no-model current-C DSML
  oracle covering rendered tool-call blocks, raw sampled DSML replay, JSON and
  string parameters, sentinel escaping, tool-result escaping,
  `parse_generated_message_ex`, and recoverable response parsing. The committed
  baseline lives at `ds4-parity/baselines/dsml/m5.6a/current-c.json` with size
  17,016 bytes and SHA256
  `3f20b4869a2035deab709e3299de91ccf151f46fa3524a8b389814ebbf880442`.
- M5.6a added Rust DSML formatting/parsing in `ds4_gguf::dsml`, routed the Rust
  prompt renderer's DSML and tool-result escaping through that module, and added
  `ds4-dsml-dump` plus `python3 ds4-parity/compare_dsml.py`.
- M5.6a validation passed for `arch -arm64 make ds4-server`,
  `./ds4-server --dump-dsml-oracle /tmp/ds4-dsml-final-c.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/compare_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6a/manifest.json --negative-test` with
  `DSML comparison: PASS, 410 checks`, `python3 -m py_compile
  ds4-parity/compare_dsml.py`, `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 14 tests, `cargo test --workspace`,
  `./ds4_test --server`, `python3 ds4-parity/compare_tokenizer_text.py
  --manifest ds4-parity/baselines/tokenization/m5.3/manifest.json
  --negative-test`, and `git diff --check`.
- M5.6a Claude review returned `NO BLOCKERS` after checking the Rust DSML
  parser/formatter against C tool-start ordering, raw block boundaries,
  sentinel escaping, entity escaping, JSON minification, response recovery,
  raw DSML replay, prompt-renderer routing, and comparator coverage.
- M5.6a implementation commit:
  `aaab1818710384e1c0b754c94f63dbf408ddb724`.
- M5.6b added `./ds4-agent --dump-agent-dsml-oracle`, a no-model current-C
  oracle for the agent incremental DSML parser. The fixture records whole,
  one-byte, marker-prefix, and parameter-boundary schedules where applicable,
  including raw/search buffer hex, parser states, current call, completed calls,
  parameter state, and error text after each chunk.
- M5.6b added Rust `ds4_gguf::agent_dsml`, `ds4-agent-dsml-dump`, and
  `python3 ds4-parity/compare_agent_dsml.py`. The committed C baseline lives at
  `ds4-parity/baselines/dsml/m5.6b/current-c.json` with size 887,559 bytes and
  SHA256
  `0b0f21728b0f5230dcbae5d3d2a99e272347ecdeac04fa57ca07ec00b9f00618`.
- M5.6b validation passed for `arch -arm64 make ds4-agent`,
  `./ds4-agent --dump-agent-dsml-oracle /tmp/agent-dsml-final-c.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/compare_agent_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6b/manifest.json --negative-test` with
  `agent DSML comparison: PASS, 37873 checks`, `python3 -m py_compile
  ds4-parity/compare_agent_dsml.py ds4-parity/compare_dsml.py`,
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf` with 16 tests,
  `cargo test --workspace`, `python3 ds4-parity/compare_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6a/manifest.json --negative-test`,
  `./ds4_test --server`, and `git diff --check`.
- M5.6b Claude review returned `NO BLOCKERS` after checking byte-vs-UTF-8
  behavior, mid-chunk done/error accumulation, close-tag variants, search-tail
  behavior, raw buffer accumulation, current/completed call transitions,
  fixture coverage, and no-model oracle startup.
- M5.6b implementation commit:
  `d6bade1d5bde4c72280bed0395322d85dfc30d5e`.
- M5.7 added `python3 ds4-parity/run_text_parity_report.py`, which runs the
  M5.2 token/prompt schema checker, M5.3-M5.5 Rust tokenizer/prompt
  comparator, M5.6a server DSML comparator, and M5.6b agent DSML comparator
  from committed fixtures without requiring the model locally.
- M5.7 records model-backed refreshes as skipped report items using exact
  `refresh_commands` from the M5.2 and M5.3 manifests, preserving the
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1` B300
  command path for future recapture.
- M5.7 wired the text report into
  `python3 ds4-parity/run_parity_report.py`, so the unified parity report now
  includes Milestone 5 text parity alongside earlier baseline comparators.
- M5.7 validation passed for `python3 -m py_compile
  ds4-parity/run_text_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_text_parity_report.py` with `summary: 4 passed, 2
  skipped, 0 failed`, JSON mode output with `ok: true`,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with
  `summary: 6 passed, 10 skipped, 0 failed`, `cargo test --workspace`, and
  `git diff --check`.
- M5.7 Claude review returned `NO BLOCKERS` after checking report
  integration, failure output, B300 refresh command fidelity, JSON/text output
  shape, status/TODO consistency, and accidental local model dependencies.
- M5.7 implementation commit:
  `3223f6e3a09f066873c5b8afc1b855adabad068d`.
- M6.1 split Milestone 6 into M6.2 through M6.7, including M6.6a/M6.6b,
  after reading the public sampling/logprob APIs in `ds4.h`, current C sampler
  and logprob math
  (`sample_argmax`, `sample_rng_next`, `sample_top_p_min_p`,
  `ds4_session_top_logprobs`, and `ds4_session_token_logprob`), CLI
  `--dump-logprobs` and perplexity surfaces, server decode stop handling,
  agent sampling, M0.3 official-vector tests, and M1.4 numeric comparator
  conventions.
- M6.1 defined separate oracle surfaces for no-model fixed-logits sampler math,
  Rust sampler/logprob math, B300 current-C session logits capture, Rust
  fixed-logits model-slice comparison, C decode stop policy fixtures, Rust
  decode stop policy comparison, and report integration.
- M6.1 validation passed for `git diff --check`; Claude review returned
  `NO BLOCKERS` after tightening M6.2 fixture ownership for source-named
  request-surface sampling tuples and splitting decode stop policy into M6.6a
  C oracle fixtures plus M6.6b Rust policy comparison.
- M6.1 implementation commit:
  `4d401ecf2a2f13e214927ab8ec05dc931d5e796e`.
- M6.2 added `./ds4-sampling-dump`, a no-model current-C fixed-logits sampler
  and logprob oracle that records selected token, actual sampler selection,
  consumed RNG state, effective sampling parameters, filtered candidate sets,
  input logits, top-logprob slices, token-logprob requests, and source-named
  request-surface sampling tuples.
- M6.2 committed
  `ds4-parity/baselines/sampling/m6.2/current-c.json` with size 16,556 bytes
  and SHA256
  `f3740560d562960ed3960f7aa07f50793b7b4338a31114b67f827ee9706493e0`.
- M6.2 routes oracle trace fields through the same helper used by
  `ds4_session_sample`, and request-surface sampling tuples now resolve through
  shared `ds4_sampling_params_*` helpers used by server, CLI, and agent
  defaults.
- M6.2 added `python3 ds4-parity/check_sampling_dump.py`, whose schema checker
  validates coverage for greedy ties, non-finite logits, temperature
  normalization, `top_p` clamping, `top_k` caps, `min_p` thresholds,
  full-vocab sampling, seeded RNG draws, top-logprob ordering, token-logprob
  requests, and request-surface parameter tuples. Its negative tests catch
  selected-token drift, missing request cases, candidate-list drift,
  top-logprob ordering drift, token-logprob schema drift, and manifest hash
  drift.
- M6.2 validation passed for `arch -arm64 make ds4-sampling-dump`,
  `./ds4-sampling-dump /tmp/ds4-sampling-m6.2-refresh.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/check_sampling_dump.py
  ds4-parity/baselines/sampling/m6.2/current-c.json --manifest
  ds4-parity/baselines/sampling/m6.2/manifest.json --negative-test` with
  `sampling schema: PASS, 1243 checks`, `sampling manifest: PASS, 7 checks`,
  and `sampling negative tests: PASS, 6 checks`, `python3 -m py_compile
  ds4-parity/check_sampling_dump.py`, `arch -arm64 make ds4_test`,
  `./ds4_test --server`, `arch -arm64 make cpu`, CPU
  `./ds4-sampling-dump` compared byte-for-byte with the committed baseline, and
  `git diff --check`.
- M6.2 Claude review returned `NO BLOCKERS` after checking sampler helper
  sharing, RNG bookkeeping, candidate ordering, request-surface helper
  plumbing, fake-session logprob safety, manifest checks, and negative-test
  coverage. Non-blocking notes: `matches_actual` now compares two calls through
  the same helper, and the schema checker is mostly shape/coverage while
  byte-for-byte baseline comparison carries M6.2 drift detection.
- M6.2 implementation commit:
  `b1b637978779700fb6ce7250e67eaa3eb23c19c6`.
- M6.3 added Rust no-model sampler/logprob math in `ds4_gguf::sampling`,
  including argmax, xorshift RNG, top-p/min-p/top-k filtering, full-vocab
  sampling, top-logprob slices, token-logprob scoring, and shared sampling
  parameter defaults.
- M6.3 added `cargo run --quiet -p ds4-gguf --bin ds4-sampling-dump-rs`, which
  emits the same fixed-logits case set as the M6.2 C oracle with selected
  tokens, RNG states, filtered candidates, and logprob scores.
- M6.3 added `python3 ds4-parity/compare_sampling.py --negative-test`, whose
  C/Rust comparator enforces exact selected token, RNG state, candidate IDs,
  candidate counts, request case coverage, top-logprob order, and token-logprob
  request shape, with `1e-5` ordinary absolute float tolerance and `1e-6`
  relative tolerance for large sentinel values. Negative tests catch selected
  token drift, RNG drift, candidate-list drift, logprob drift, and request
  coverage drift.
- M6.3 validation passed for `cargo test -p ds4-gguf sampling --quiet` with 3
  sampling tests passing, `python3 -m py_compile
  ds4-parity/compare_sampling.py`, `python3 ds4-parity/compare_sampling.py
  --negative-test --write-rust-dump /tmp/ds4-sampling-rust-from-comparator.json`
  with `sampling C/Rust comparator: PASS, 3241 checks` and `sampling C/Rust
  negative tests: PASS, 5 checks`, `cargo fmt --all -- --check`,
  `cargo test --workspace` with all workspace tests passing, and
  `git diff --check`.
- M6.3 Claude review returned `NO BLOCKERS` after checking Rust numeric edge
  cases, RNG semantics, candidate filtering order, top-logprob tie order,
  non-finite handling, request fixture coverage, and comparator negative tests.
  Non-blocking notes: top-p/full-vocab tied-logit fixture coverage is latent,
  Rust faithfully recomputes full-vocab weights during roulette like C, and
  greedy mode intentionally leaves effective params unclamped to match C.
- M6.3 implementation commit:
  `fea2ea3de57a260474d349d2536527bf2c16927a`.
- M6.4 added `./ds4-logits-dump`, a current-C model-backed oracle helper that
  runs official-vector prompts through `ds4_session_sync`,
  `ds4_session_argmax`, `ds4_session_top_logprobs`,
  `ds4_session_token_logprob`, and `ds4_session_eval`, then records selected
  tokens, token bytes, top-logprob slices, official-top deltas, and per-step
  full-logits SHA256s. The helper requires a 64-character lowercase
  `--model-sha256` and verifies the actual model file via `sha256sum` or
  `shasum -a 256` before opening the engine.
- M6.4 exposes `ds4_session_logits_data` so the dump helper can write a
  contiguous f32 logits blob without moving model execution into the helper.
- M6.4 captured B300 current-C artifacts on `ds4-rust-port-b300` in
  `hou2-prod1/default` after refreshing source into `/workspace/ds4` and
  building `make ds4-logits-dump CUDA_ARCH=native`. Capture command wrote
  `ds4-parity/baselines/sampling/m6.4/current-c.json` with size 19,535 bytes
  and SHA256
  `5343e5aa855305ca2092943e155a359db50a28216d44927d450d2e0cce82efd0`,
  plus `ds4-parity/baselines/sampling/m6.4/logits.f32le` with size
  4,654,080 bytes and SHA256
  `972636c24ff63534d3a7fb7b1360e78786dee0bdd111f1fde813aa758e1f1928`.
- M6.4 fixture contains 9 scored steps across `short_italian_fact`,
  `short_code_completion`, and `short_reasoning_plain`. `long_memory_archive`
  remains explicitly skipped for the existing API/official-graph mismatch, and
  `long_code_audit` is explicitly skipped because repeated B300 CUDA captures
  produced byte-different long-context logits even with deterministic-kernel
  probes.
- M6.4 added `python3 ds4-parity/check_session_logits_dump.py`, whose schema
  and hash checker validates model/backend identity, case coverage, prompt
  hashes, selected-token matches, top-logprob shape, selected/top scores
  recomputed from the f32le logits blob, official-top local matches and delta
  tolerances, contiguous per-step logits ranges, per-step logits SHA256s,
  whole-blob manifest SHA256 plus n_vocab/step counts, and exact
  temp-kubeconfig/context refresh commands.
- M6.4 validation passed for `arch -arm64 make ds4-logits-dump`,
  `python3 -m py_compile ds4-parity/check_session_logits_dump.py`, B300
  `make ds4-logits-dump CUDA_ARCH=native`, B300 capture with
  `./ds4-logits-dump --backend cuda -m /workspace/ds4/ds4flash.gguf -v
  tests/test-vectors/official.vec -o
  ds4-parity/baselines/sampling/m6.4/current-c.json -l
  ds4-parity/baselines/sampling/m6.4/logits.f32le --model-sha256
  efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  local and B300 checker runs with `session logits schema: PASS, 2356 checks`,
  `session logits manifest: PASS, 20 checks`, and `session logits negative
  tests: PASS, 11 checks`, `python3 ds4-parity/compare_logprob_numeric.py`
  with `summary: 5/5 sections passed, 528 checks`, `arch -arm64 make cpu`,
  and `git diff --check`.
- M6.5 adds `ds4-model-logits-dump-rs`, which reads the committed M6.4
  `logits.f32le` blob as contiguous f32 vocab slices, loads the M5.3
  tokenizer GGUF, runs Rust `sample_argmax` and `top_logprobs`, and emits a
  flat per-slice JSON dump with selected token IDs, rendered token bytes, and
  top-logprob scores.
- M6.5 adds `python3 ds4-parity/compare_model_logits.py`, which maps those
  flat Rust slices back to the M6.4 current-C case/step records and compares
  selected token, selected bytes, expected bytes, logits offsets, top-logprob
  ordering, top token IDs, token bytes, logits, and logprobs.
- M6.5 validation passed with `python3 -m py_compile
  ds4-parity/compare_model_logits.py`, `python3
  ds4-parity/compare_model_logits.py --negative-test` (`model logits C/Rust
  comparator: PASS, 2982 checks, max_abs_logit_delta=5.00000041e-08,
  max_abs_logprob_delta=5.00000006e-08`; negative tests `PASS, 6 checks`),
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf --bin
  ds4-model-logits-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M6.6a adds `./ds4-decode-policy-dump`, a no-model current-C decode stop
  policy oracle. The helper includes the `ds4_server.c` test surface so the
  fixture uses the real C stop-list, UTF-8 stream-hold, DSML marker, generated
  message parse, Anthropic stop-reason, and Responses status mapping helpers.
- M6.6a fixture `ds4-parity/baselines/sampling/m6.6a/current-c.json` covers
  CLI EOS/length, server OpenAI EOS/length/user-stop/stream-stop-tail/
  streaming-stop-hit/partial-UTF-8/stop-at-mid-UTF-8-boundary/
  tool-call-boundary, server Responses length mapping, server Anthropic tool
  mapping, and agent EOS/length defaults. The artifact is 17,000 bytes with
  SHA256
  `9d11d90a12e1ee4d16ac1d4aa8c971efe775a86202004db91aff8d452081a2b5`.
- M6.6a adds `python3 ds4-parity/check_decode_policy_dump.py`, whose schema
  and negative checks validate case coverage, request option records,
  generated text schedules, finish reason, visible bytes, streamed bytes,
  held-tail bytes, session invalidation, stop boundary offsets, tool-call
  boundary flags, and API finish mappings.
- M6.6a validation passed with `arch -arm64 make ds4-decode-policy-dump`,
  `./ds4-decode-policy-dump ds4-parity/baselines/sampling/m6.6a/current-c.json`,
  `python3 -m py_compile ds4-parity/check_decode_policy_dump.py`, `python3
  ds4-parity/check_decode_policy_dump.py --negative-test` (`decode policy
  schema: PASS, 969 checks`; manifest `PASS, 5 checks`; negative tests `PASS,
  10 checks`), `arch -arm64 make ds4_test`, `./ds4_test --server`, and
  `git diff --check`.
- M6.6b adds the Rust byte-oriented decode stop policy in
  `rust/ds4-gguf/src/decode_policy.rs` plus `ds4-decode-policy-dump-rs`.
  It mirrors the M6.6a generated-token schedules without introducing a Rust
  CLI/server runtime or reimplementing M5 DSML parsing; the tool case only
  observes complete tool-call marker boundaries.
- M6.6b adds `python3 ds4-parity/compare_decode_policy.py`, which runs the
  Rust dump and compares request records, schedules, finish reason, raw and
  visible bytes, streamed bytes, held tails, session invalidation, stop
  boundaries, tool-boundary flags, API finish mappings, and per-step streaming
  metadata against the committed M6.6a C oracle.
- M6.6b validation passed with `python3 -m py_compile
  ds4-parity/compare_decode_policy.py`, `python3
  ds4-parity/compare_decode_policy.py --negative-test` (`decode policy C/Rust
  comparator: PASS, 1059 checks`; negative tests `PASS, 10 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf decode_policy`, `cargo test
  -p ds4-gguf --bin ds4-decode-policy-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M6.7 adds `python3 ds4-parity/run_sampling_parity_report.py`, which runs the
  M6.2 current-C sampler/logprob checker, M6.3 Rust sampler comparator, M6.4
  committed session-logits fixture checker, M6.5 Rust model-logits comparator,
  M6.6a current-C decode policy checker, and M6.6b Rust decode-policy
  comparator.
- M6.7 records the model-backed M6.4 B300 session-logits recapture as a
  skipped report item using the exact `refresh_commands` from
  `ds4-parity/baselines/sampling/m6.4/manifest.json`; no other M6 local
  comparator is skipped by the M6 report.
- M6.7 wires the sampling/logprob report into
  `python3 ds4-parity/run_parity_report.py`. Validation passed with `python3
  -m py_compile ds4-parity/run_sampling_parity_report.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_sampling_parity_report.py` (`summary: 6 passed, 1 skipped, 0
  failed`), `python3 ds4-parity/run_sampling_parity_report.py --json |
  python3 -m json.tool`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` (`summary: 7 passed, 10 skipped, 0 failed`), `python3
  ds4-parity/run_parity_report.py --skip-local-oracles --json | python3 -m
  json.tool`, `cargo test --workspace`, and `git diff --check`.
- M7.1 split Milestone 7 into C KV header/policy oracle, Rust KV
  parser/policy, generic full-file round-trip coverage, per-extension trailer
  coverage, C on-disk session payload shape oracle, Rust payload header reader,
  KV replay/prefix decision comparator, B300 disk-KV and in-memory snapshot
  restore oracle, and report integration items. The first executable item is
  the no-model C KV header and policy oracle; the C on-disk session payload
  shape oracle is independently eligible because it depends on session payload
  code rather than KV header/policy work.
- M7.2 added `./ds4-kv-policy-dump`, a deterministic no-model current-C
  oracle for KVC header bytes, decoded fields, reason/key-kind mapping,
  little-endian helpers, SHA/path helpers, size budgeting, store-boundary
  selection, chat-anchor selection, continued-store targets, byte-prefix
  matching, eviction scoring with explicit `now`, text-prefix lookup, and M0.5
  parsed header fixture references.
- M7.2 added `python3 ds4-parity/check_kv_policy_dump.py`, whose schema,
  manifest, and negative checks validate the C oracle dump and the committed
  M0.5 `kv-header.tsv` row references.
- M7.2 local validation passed for `arch -arm64 make ds4-kv-policy-dump`,
  `./ds4-kv-policy-dump ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m py_compile ds4-parity/check_kv_policy_dump.py`, and
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema
  checks, `11` manifest checks, `7` negative checks), `arch -arm64 make`,
  `arch -arm64 make cpu`, deterministic CPU-regenerated dump comparison
  against the committed M7.2 artifact, `arch -arm64 make ds4_test`,
  `./ds4_test --server`, and `git diff --check`.
- M7.3 adds `rust/ds4-gguf/src/kv_policy.rs` for no-model KVC header
  parsing/writing, reason/key-kind helpers, SHA/path helpers, file-size
  budgeting, store-boundary selection, chat-anchor selection,
  continued-store target selection, byte-prefix matching, eviction scoring,
  and text-prefix entry selection.
- M7.3 adds `ds4-kv-policy-dump-rs`, which emits the same deterministic
  synthetic no-model policy fixture as the M7.2 C oracle with a Rust schema and
  source label.
- M7.3 adds `python3 ds4-parity/compare_kv_policy.py`, which runs the Rust
  dump and recursively compares it to the committed M7.2 C oracle while
  allowing only the schema/source labels to differ. It checks header bytes,
  decoded fields, reason and extension flags, SHA/path helpers, policy
  decisions, eviction scores, text-prefix selections, and M0.5 header rows.
- M7.3 local validation passed for `python3 -m py_compile
  ds4-parity/compare_kv_policy.py`, `python3
  ds4-parity/compare_kv_policy.py --negative-test` (`KV policy C/Rust
  comparator: PASS, 1488 checks`; negative tests `PASS, 8 checks`), `python3
  ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema checks,
  `11` manifest checks, `7` negative checks), `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-policy-dump-rs`, and `cargo test --workspace`.
- M7.4a adds `./ds4-kvc-file-dump`, a deterministic no-model current-C oracle
  for complete generic KVC file bytes: fixed header, text length, rendered-text
  bytes, opaque payload bytes, and opaque trailer bytes.
- M7.4a fixture `ds4-parity/baselines/kv/m7.4a/current-c.json` covers
  no-trailer, opaque trailer, visible-transcript flag without payload, empty
  text with trailer, no-budget/fitting-budget/over-budget size decisions, and
  malformed header/text/payload/trailer boundary records. The artifact is
  6,445 bytes with SHA256
  `ff37ba4a359b10d66199928a1936b10ec0adc43a17ceb7ba49c0ad3e02c8b7d7`.
- M7.4a adds Rust generic KVC full-file helpers in
  `rust/ds4-gguf/src/kv_policy.rs`; the reader keeps payload and trailer bytes
  opaque and treats all bytes after fixed header, text, and declared payload as
  generic trailer data.
- M7.4a adds `python3 ds4-parity/compare_kvc_file.py`, which runs
  `ds4-kvc-file-dump-rs` and compares complete file hex, read metadata,
  file-size budget records, malformed case outcomes, and trailer-size records
  against the committed C oracle.
- M7.4a local validation passed for `arch -arm64 make ds4-kvc-file-dump`,
  `./ds4-kvc-file-dump ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kvc_file.py`, `python3
  ds4-parity/compare_kvc_file.py --negative-test` (`KVC file C/Rust
  comparator: PASS, 277 checks`; negative tests `PASS, 8 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf kvc`, `cargo test -p
  ds4-gguf --bin ds4-kvc-file-dump-rs`, `cargo test --workspace`, `arch
  -arm64 make cpu`, CPU-regenerated `./ds4-kvc-file-dump` comparison against
  the committed M7.4a artifact, and `git diff --check`.
