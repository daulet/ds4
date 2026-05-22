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

- Status: in-progress
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

## Later Items

Add later Milestone 1 items from `RUST_PORT_ROADMAP.md` as each active item
completes.
