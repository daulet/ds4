# DS4 Rust Port Status

- Date: 2026-05-22 UTC
- Branch: `main`
- Starting oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Active item: M3.1 Backend ABI Wrapper Parity
- Last validated source commit: `8f11bac65319ae6d429a6ce3d2c225a46ff73f31`
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
