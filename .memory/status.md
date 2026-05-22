# DS4 Rust Port Status

- Date: 2026-05-22 UTC
- Branch: `main`
- Starting oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Active item: M4.6 Metadata Baselines And Unified Report Integration
- Last validated source commit: `da12244d15fde251196e3fcc4faa4a7552693ae0`
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
