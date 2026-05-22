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

- Status: in-progress
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

## Later Items

Add later post-Milestone 0 items from `RUST_PORT_ROADMAP.md` when M0.6
completes.
