# Rust Port Roadmap

This roadmap describes a comparison-first Rust port of DwarfStar 4. The current
C/CUDA/Metal implementation remains the correctness and performance oracle until
Rust has matching coverage and validation evidence.

The port should take the long route on purpose: build the oracle and comparator
for each boundary before moving that boundary into Rust. A milestone is complete
only when Rust can be compared against the existing implementation with fixed
fixtures and an explicit acceptance rule.

## Principles

- Keep behavior compatible with the current binaries at every stage.
- Add a comparator before porting a component.
- Move one ownership boundary at a time.
- Keep backend APIs narrow and tensor-resident so CUDA, Metal, and future Rust
  GPU implementations can be swapped behind the same runtime surface.
- Treat current C behavior as the reference unless a milestone explicitly
  defines a new format or semantic change.
- Preserve benchmark comparability by using the same model, backend, prompt,
  context, power state, and machine when measuring speed.

## Comparison Contract

Every milestone should define these fields before implementation starts:

- Oracle: the current C path, current backend ABI, official vector, server
  trace, benchmark CSV, or another fixed reference.
- Fixture: exact model, prompt, request JSON, token sequence, KV file, tensor
  input, or trace replay input.
- Comparator: the Rust command or test that compares Rust output to the oracle.
- Acceptance: byte equality, exact token equality, numeric tolerance, response
  shape match, or speed threshold.
- Drift policy: whether Rust must match the old behavior exactly or whether a
  deliberate format/version change is allowed.

If a phase cannot fill in those fields, it should be split into a smaller phase.

## Execution Work Items

Each work item must be small enough to review independently and must name the
execution-behavior oracle before implementation starts. A commit may contain one
or more work items only when they share the same oracle and validation command.

Required fields for every work item:

- Goal: the tangible behavior, artifact, or ownership boundary completed by the
  commit.
- Oracle: the exact C binary, C ABI path, captured artifact, fixture, benchmark,
  or reference command used as truth.
- Comparator: the command, script, or manual check that compares Rust or port
  artifacts to the oracle.
- Acceptance: the equality rule, tolerance, response shape rule, or documented
  no-behavior-change condition.
- Review gate: a non-interactive Claude review prompt that states the commit
  goal, diff summary, oracle, comparator, and validation evidence.
- Validation gate: commands that must pass before commit and before push.

### Milestone 0 Work Items

Milestone 0 is split because it creates the long-lived oracle and fixture
contract used by later milestones. These items must land in order.

#### M0.1: Port Execution Protocol

- Goal: add project-local port protocol, active board, parity status, and
  lessons log; document these work-item rules in this roadmap.
- Oracle: current checked-out C implementation at the starting commit.
- Fixture: repository state, `AGENT.md`, `CONTRIBUTING.md`, and this roadmap.
- Comparator: no source behavior changes; `git diff --name-only` lists only
  roadmap and `.memory/` files, and `git diff --check` reports no whitespace
  errors.
- Acceptance: protocol names the validation ladder, reviewer policy, commit
  policy, debugging ledger policy, and B300 validation rule; active board names
  the next executable item.
- Drift policy: no implementation or fixture-format drift allowed.

#### M0.2: Baseline Build Command Capture

- Goal: capture reproducible baseline command lines and logs for `make`,
  `make test`, `make cpu`, and backend-specific regression commands available
  on the target machine.
- Oracle: current C build products from the checked-out commit.
- Fixture: machine/backend record, compiler versions, command environment, and
  captured build/test logs under `ds4-parity/baselines/`.
- Comparator: rerun baseline commands and compare exit status plus declared
  output artifacts against the recorded manifest.
- Acceptance: each command has a manifest entry with exact command, cwd,
  commit, environment overrides, exit status, output log path, and whether it is
  runnable locally or requires the B300 pod.
- Drift policy: no Rust behavior exists yet; this records the C oracle exactly.

#### M0.3: Official Vector Logprob Baseline

- Goal: capture the current implementation's output for
  `tests/test-vectors/official.vec`.
- Oracle: current `./ds4_test --logprob-vectors` path.
- Fixture: official vector file, model path or model hash, backend, and
  captured stdout/stderr.
- Comparator: baseline manifest entry plus later replay through `ds4-parity`.
- Acceptance: either the run passes and records the exact artifact, or the
  missing model/backend requirement is recorded as blocked with the exact
  command needed on the B300 pod.
- Drift policy: future Rust must match selected greedy tokens exactly and
  numeric logprob output within the fixture tolerance.

#### M0.4: Server Trace Baselines

- Goal: capture representative current server request behavior.
- Oracle: current `./ds4-server` binary with trace enabled.
- Fixture: fixed request JSON files for non-streaming chat, streaming chat,
  tool call rendering, thinking controls, and cache-related requests.
- Comparator: request replay records response JSON or event stream plus trace
  files; later Rust server responses compare after approved normalization.
- Acceptance: request fixtures, command lines, response artifacts, trace paths,
  model identity, backend, and normalization rules are recorded.
- Drift policy: response shape, prompt rendering, tool-call records, and cache
  decisions must match unless a later milestone explicitly documents a change.

#### M0.5: KV And Snapshot Baselines

- Goal: capture current KV-cache and session-restore behavior for prompt reuse.
- Oracle: current C KV store and session snapshot implementation.
- Fixture: prompt inputs, cache directory, generated KV files, trace output, and
  manifest entries describing cache hit, cache miss, prefix match, and restore.
- Comparator: binary file hashes and trace-normalized cache decisions.
- Acceptance: artifacts can be regenerated from the manifest and compared
  byte-for-byte where timestamps or paths are not part of the format.
- Drift policy: Rust-written KV files must remain byte-identical until a
  versioned format change is explicitly introduced.

#### M0.6: Benchmark CSV Baselines

- Goal: capture at least one short-context and one long-context `ds4-bench`
  CSV for the reference backend.
- Oracle: current `./ds4-bench` binary on the same machine, model, backend,
  context settings, and power state.
- Fixture: prompt file, model identity, backend, command line, machine record,
  and CSV output under `ds4-parity/baselines/bench/`.
- Comparator: CSV schema and selected throughput fields compared by later
  `ds4-parity` helpers.
- Acceptance: CSV artifacts and machine/backend metadata are recorded; any
  unavailable local run is routed to the B300 pod for CUDA validation.
- Drift policy: speed comparisons must use the same fixture and machine class;
  regressions beyond an agreed threshold require explicit documentation.

## Proposed Workspace Shape

Introduce a Rust workspace beside the current C implementation:

- `ds4-core`: model constants, metadata, tokenizer-facing data structures,
  prompt rendering, sampling, tensor descriptors, and shared error types.
- `ds4-gguf`: GGUF parsing, tensor index construction, metadata validation,
  and model-file layout checks.
- `ds4-runtime`: session state, KV state abstractions, graph execution
  orchestration, layer scheduling, and backend dispatch.
- `ds4-gpu-sys`: temporary unsafe FFI bindings to the existing `ds4_gpu.h`
  backend ABI.
- `ds4-gpu`: safe Rust wrappers over `ds4-gpu-sys` and the backend trait used
  by `ds4-runtime`.
- `ds4-parity`: comparison fixtures and oracle runners used during the port.
- `ds4-cli`: Rust replacement for the interactive CLI.
- `ds4-server`: Rust replacement for the HTTP server.
- `ds4-agent`: Rust replacement for the integrated coding agent.

This layout can be adjusted as the port progresses. The important boundary is
that unsafe backend calls are isolated and the runtime sees a safe backend
interface.

## Backend Strategy

The first Rust backend should be an FFI-backed adapter over the existing GPU
API. It should call the current CUDA and Metal implementations through a narrow
Rust wrapper and preserve the current tensor-resident contract.

The Rust runtime should define a backend trait around DS4 operations rather
than around generic CUDA or Metal concepts. Expected operation families include:

- tensor allocation, copy, fill, and lifetime management,
- dense projections and quantized matmuls,
- RMS norm, Q/KV norm, RoPE, and KV rounding,
- raw and compressed KV writes,
- compressor update and prefill paths,
- raw, mixed, indexed, and masked attention,
- router selection and routed MoE execution,
- hyper-connection reductions and expansions.

The future CUDA rewrite can then replace only the backend implementation. Until
the cuda-oxide gaps are closed, CUDA should remain reachable through the current
backend or through targeted FFI sidecars.

## Milestone 0: Baseline Oracle Capture

Before any Rust port work, capture the current implementation's observable
behavior.

Deliverables:

- Baseline commands for `make`, `make test`, `make cpu`, and backend-specific
  regression suites.
- Baseline logprob-vector output using `tests/test-vectors/official.vec`.
- Baseline server traces for representative request JSON fixtures.
- Baseline KV-cache files for representative prompt reuse and session restore.
- Baseline `ds4-bench` CSVs for at least one short and one long-context prompt.
- A `ds4-parity` fixture directory layout for all future comparison artifacts.

Oracle:

- Current checked-out C/CUDA/Metal implementation.

Acceptance:

- Baseline commands are reproducible on the target machine.
- Fixtures record exact command lines, model path or model hash, backend,
  environment variables, and output files.

## Milestone 1: Parity Harness

Build the Rust comparison harness before porting logic.

Deliverables:

- `ds4-parity` runner that can invoke the current C binaries or C ABI helpers.
- Snapshot tests for text fixtures, JSON fixtures, metadata dumps, and binary
  payloads.
- Numeric comparison helpers for logits, probabilities, tensors, and benchmark
  CSVs.
- A standard report format that says which oracle, fixture, and tolerance were
  used.

Oracle:

- The baseline artifacts from Milestone 0.

Acceptance:

- The harness can compare the current implementation against its own captured
  baseline and report no unexpected drift.
- Numeric comparisons use explicit tolerances; text and binary comparisons are
  byte-exact unless a fixture says otherwise.

### Milestone 1 Work Items

Milestone 1 must land as harness-only commits. It should not introduce Rust
runtime behavior or change existing C binaries.

#### M1.1: Parity Harness Work Item Breakdown

- Goal: split Milestone 1 into reviewable work items before adding harness
  code.
- Oracle: Milestone 0 baseline artifacts and this roadmap's Milestone 1
  deliverables.
- Fixture: `ds4-parity/baselines/manifest.md`, the committed Milestone 0
  artifact directories, and current build/test command records.
- Comparator: documentation-only review; no source behavior changes.
- Acceptance: each Milestone 1 implementation item names a tangible goal,
  oracle, fixture, comparator, acceptance rule, drift policy, review gate, and
  validation gate.
- Drift policy: no implementation or fixture-format drift allowed.
- Review gate: ask Claude to review the roadmap diff for missing or oversized
  Milestone 1 work items.
- Validation gate: `git diff --check`.

#### M1.2: Static Baseline Artifact Verifier

- Goal: add a local `ds4-parity` verifier that checks committed Milestone 0
  artifacts without rerunning model-backed commands.
- Oracle: committed Milestone 0 artifact hash files, JSON responses, CSV files,
  parsed KV metadata, and the baseline manifest.
- Fixture: `ds4-parity/baselines/**/artifact-sha256.txt`,
  M0.4/M0.5 response JSON, M0.5 KV metadata, and M0.6 benchmark CSVs.
- Comparator: a checked-in command such as
  `python3 ds4-parity/verify_baselines.py` that validates hashes, parses JSON
  and CSV artifacts, and emits a standard report naming oracle, fixture,
  comparator, and result for each section.
- Acceptance: the verifier exits 0 against the committed baseline artifacts,
  reports every Milestone 0 artifact family, and fails when a copied fixture is
  deliberately changed in a temporary test directory.
- Drift policy: exact bytes for hash-listed artifacts; structured parsers may
  normalize only the timestamp/path fields already documented in
  `ds4-parity/baselines/manifest.md`.
- Review gate: ask Claude to review the verifier diff, report format, and
  validation evidence for uncovered Milestone 0 artifacts.
- Validation gate: run the verifier, run its negative test, and run
  `git diff --check`.

#### M1.3: Server And KV Normalization Comparators

- Goal: add comparison helpers for M0.4 server traces/responses and M0.5 KV
  restore artifacts.
- Oracle: M0.4 `server-traces/m0.4/` and M0.5 `kv-artifacts/m0.5/`.
- Fixture: M0.4 server request fixtures, response JSON/SSE/header artifacts,
  trace files, M0.5 cache decision logs, rendered cached text, and parsed
  `kv-header.tsv`.
- Comparator: harness subcommands or modules that compare a candidate artifact
  directory to the baseline after applying only the documented normalizations.
- Acceptance: self-comparison of committed M0.4 and M0.5 artifacts reports no
  drift; temporary candidate edits to cache source, cached token count, finish
  reason, KV reason, or rendered text produce failures.
- Drift policy: response shape, prompt rendering, tool-call mapping, cache
  source, cache token counts, KV header semantic fields, and rendered text are
  exact behavioral surface.
- Review gate: ask Claude to review the normalization allow-list and negative
  test coverage for over-normalization.
- Validation gate: run self-comparison, run negative tests, and run
  `git diff --check`.

#### M1.4: Logprob And Numeric Comparator

- Goal: add numeric comparison support for official-vector output and later
  tensor/logit fixtures.
- Oracle: M0.3 official-vector baseline and the current
  `./ds4_test --logprob-vectors` command on the recorded model/backend.
- Fixture: `tests/test-vectors/official.vec`, model identity from M0.3, and the
  captured M0.3 B300 logprob output.
- Comparator: harness logic that parses selected greedy-token outcomes exactly
  and compares numeric slices with an explicit tolerance recorded in the report.
- Acceptance: current captured M0.3 output compares cleanly to its baseline;
  exact-token mismatches fail; numeric mismatches outside the declared tolerance
  fail.
- Drift policy: selected greedy tokens are exact; numeric logprob/tensor values
  may use only the tolerance declared by the fixture or report.
- Review gate: ask Claude to review tolerance choices and exact-token failure
  coverage.
- Validation gate: run the numeric comparator, run negative tests for token and
  numeric drift, and run `git diff --check`.

#### M1.5: Benchmark CSV Comparator

- Goal: add comparison support for M0.6 benchmark CSV baselines.
- Oracle: M0.6 `bench/m0.6/csv/b300-short.csv` and
  `bench/m0.6/csv/b300-long.csv`.
- Fixture: the M0.6 prompt hash, model hash, B300 machine record, CSV files,
  and `csv-summary.json`.
- Comparator: harness logic that validates CSV schema, context frontiers,
  prefill intervals, generation-token counts, `kvcache_bytes`, and throughput
  ratios against the M0.6 drift policy.
- Acceptance: committed M0.6 CSVs self-compare cleanly; schema, frontier,
  generation-token, or cache-byte edits fail; throughput values below the
  documented threshold are reported as performance regressions.
- Drift policy: workload shape and `kvcache_bytes` are exact; throughput is
  compared only on the same machine class and uses the M0.6 threshold unless a
  later milestone changes it.
- Review gate: ask Claude to review threshold handling and failure reporting.
- Validation gate: run CSV self-comparison, run negative tests, and run
  `git diff --check`.

#### M1.6: Oracle Runner And Unified Report

- Goal: add a single documented command that can run available current-C
  oracle checks and compare their outputs to Milestone 0 baselines.
- Oracle: current C binaries plus Milestone 0 baselines.
- Fixture: local non-model fixtures, B300-routed model fixtures, and the
  baseline manifest.
- Comparator: a report-producing `ds4-parity` command that either runs the
  available oracle locally or marks a model/GPU case with the exact B300 command
  required, then compares produced artifacts with the M1.2 through M1.5 helpers.
- Acceptance: local no-model checks run without `ds4flash.gguf`; model-backed
  checks are either executed on the B300 pod or reported as skipped with exact
  reproduction commands; the unified report has no unexpected drift against the
  current baseline.
- Drift policy: no source behavior changes; runner skips are allowed only when
  the report names the missing model/backend requirement and exact rerun
  command.
- Review gate: ask Claude to review runner skip semantics, B300 command
  routing, and whether report results are actionable.
- Validation gate: run the unified report locally, run any available B300
  oracle checks needed for the commit, and run `git diff --check`.

## Milestone 2: Rust Workspace and FFI Skeleton

Add Rust build structure without moving behavior.

Deliverables:

- Rust workspace and crate skeletons.
- Build integration that leaves existing C targets unchanged.
- `ds4-gpu-sys` declarations for the existing backend ABI.
- Empty or smoke-test-only safe wrappers in `ds4-gpu`.

Oracle:

- Current C build and test commands.

Comparator:

- Existing `make`, `make test`, and `make cpu`.
- Rust workspace tests that do not require a full model.

Acceptance:

- No behavior change in existing binaries.
- Rust crates build and test independently.

## Milestone 3: Backend ABI Wrapper Parity

Port only the unsafe boundary around `ds4_gpu.h`.

Deliverables:

- Safe Rust tensor handle with ownership, byte-size checks, and drop behavior.
- Safe wrappers for allocation, read/write, copy/fill, command flush, and tensor
  byte queries.
- Direct C-vs-Rust wrapper tests that call the same backend operation through
  both paths on the same inputs.

Oracle:

- Direct calls to the current C backend ABI.

Comparator:

- Small tensor fixtures for allocation, fill, copy, read/write, and error paths.

Acceptance:

- Rust wrapper results match direct C ABI results byte-for-byte for simple
  tensor operations.
- Error cases return equivalent failure categories.
- Existing C tests still pass.

## Milestone 4: Model Metadata and GGUF Parity

Move GGUF metadata parsing and DS4 model validation into Rust.

Deliverables:

- Rust metadata dump format.
- C metadata dump helper, if the current loader does not already expose one.
- Tensor index, quantization mix, MTP state, and required-constant validation in
  Rust.

Oracle:

- Current C loader on supported and intentionally unsupported GGUF files.

Comparator:

- Metadata dump comparison for each fixture model.

Acceptance:

- Supported models produce equivalent metadata and tensor indexes.
- Unsupported models fail at the same validation boundary or with a stricter
  Rust error that is documented in the fixture.

## Milestone 5: Tokenization, Prompt Rendering, and DSML Parity

Port text handling before runtime execution.

Deliverables:

- Rust prompt renderer for thinking, non-thinking, latest reminder, tools, and
  developer/internal roles.
- Rust DSML tool-call parsing and tool-result rendering.
- C oracle fixtures that dump rendered prompts, token IDs, and exact DSML blocks
  for representative conversations.

Oracle:

- Current C CLI/server rendering and token handling.

Comparator:

- Request JSON fixtures, prompt text fixtures, tool-call fixtures, and expected
  token ID streams.

Acceptance:

- Rendered prompt bytes match.
- Token ID streams match.
- Exact DSML replay cases match byte-for-byte.
- Deterministic fallback DSML rendering matches fixture output.

## Milestone 6: Sampling and Logprob Parity

Port logits post-processing and token selection.

Deliverables:

- Rust sampler and logits processors.
- Fixed logits fixtures captured from the current implementation.
- Comparison of top-k logprob slices, selected token, stop handling, and token
  byte conversion.

Oracle:

- Current `ds4_session_top_logprobs`, token logprob, and decode selection
  behavior.

Comparator:

- Fixed logits arrays and official-vector prompt cases.

Acceptance:

- Selected greedy tokens match exactly.
- Top-logprob ordering and token bytes match.
- Logprob values match within an explicit tolerance for floating-point output.

## Milestone 7: KV Store and Snapshot Parity

Port disk KV and in-memory snapshot handling with format comparison first.

Deliverables:

- Rust parser and writer for current KV-cache headers and payload boundaries.
- Fixtures for cache hit, cache miss, prefix match, eviction, exact DSML replay,
  and snapshot restore.
- A versioning policy if Rust must change any on-disk structure.

Oracle:

- Current `ds4_kvstore` and session snapshot implementation.

Comparator:

- Binary KV files, rendered cached text, token prefix fixtures, and restore
  traces.

Acceptance:

- Existing KV files parse in Rust.
- Rust-written KV files are byte-identical to C unless the milestone explicitly
  introduces a versioned format.
- Cache hit/miss decisions match for the same prompt and options.
- Restored sessions produce the same next-token distribution within tolerance.

## Milestone 8: CLI Surface Parity

Port the CLI after core text, sampling, and persistence pieces are comparable.

Deliverables:

- Rust CLI with matching flags and output modes.
- Golden CLI transcript fixtures for one-shot prompt, prompt file, logprob dump,
  thinking controls, and error handling.

Oracle:

- Current `./ds4` binary.

Comparator:

- CLI fixture runner that executes C and Rust binaries with the same arguments.

Acceptance:

- Exit status and stderr category match.
- Machine-readable outputs match byte-for-byte where applicable.
- Interactive behavior is covered by scripted stdin/stdout transcripts.

## Milestone 9: Server Surface Parity

Port the HTTP server with request/trace comparison.

Deliverables:

- Rust server with matching endpoints, request parsing, streaming shape, thinking
  controls, trace fields, and tool-call behavior.
- Fixed request JSON fixtures.
- Trace normalization rules for timestamps and performance counters.

Oracle:

- Current `./ds4-server` binary.

Comparator:

- Request replay that sends the same fixture to C and Rust servers.

Acceptance:

- Non-streaming JSON response shape matches.
- Streaming event sequence matches after normalizing timestamps and generated
  token timing.
- Trace-rendered prompts, cache decisions, and tool-call records match.
- Existing `./ds4_test --server` passes through the Rust server path.

## Milestone 10: Runtime Graph Orchestration Parity

Move layer scheduling and graph execution from C into Rust while still using the
FFI-backed GPU backend.

Deliverables:

- Rust representation of the DS4 layer graph.
- Rust scheduling for prefill, decode, compressed KV maintenance, MTP, and
  backend dispatch.
- Backend trait coverage for every operation currently called through
  `ds4_gpu.h`.
- Intermediate tensor checkpoints for selected layers and stages.

Oracle:

- Current C graph orchestration using the same GPU backend.

Comparator:

- Intermediate tensor diffs, official-vector cases, long-context prompts, and
  benchmark-frontier snapshots.

Acceptance:

- Intermediate tensors match within per-stage tolerances.
- `./ds4_test --logprob-vectors` passes through the Rust runtime path.
- `./ds4_test --long-context` passes through the Rust runtime path.
- `./ds4_test --tool-call-quality` passes through the Rust runtime path.
- `ds4-bench` throughput is compared against the same backend and model quant;
  any regression beyond the agreed threshold is documented before merge.

## Milestone 11: Agent Trace Replay

Port the integrated coding agent only after runtime and server parity are
stable.

Deliverables:

- Scripted agent session fixtures.
- Tool execution stubs or deterministic tool-output replay.
- Rust agent loop and session switching behavior.

Oracle:

- Current `ds4-agent` traces and deterministic replay fixtures.

Comparator:

- Agent trace replay with normalized timestamps, paths, and command duration
  fields.

Acceptance:

- Tool-call sequence, rendered context, session switching, and final visible
  outputs match fixture expectations.
- Live manual sessions remain a final smoke test, not the primary comparator.

## Milestone 12: Backend Replacement Parity

Only after Rust owns the host runtime should GPU backend replacement begin.

CUDA work should track the cuda-oxide capability roadmap. Until the relevant
gaps are closed, acceptable strategies include keeping the existing CUDA backend,
using raw CUDA/cuBLAS bindings from Rust, or preserving targeted CUDA C++
sidecars for specialized kernels.

Oracle:

- Old backend implementation behind the Rust runtime.

Comparator:

- Operation-level tensor fixtures, official-vector fixtures, long-context
  fixtures, and `ds4-bench` CSVs.

Acceptance:

- Operation outputs match within documented tolerances.
- End-to-end official-vector and long-context tests pass.
- Backend-specific regression suites such as `make cuda-regression` and
  `./ds4_test --metal-kernels` pass where applicable.
- Speed comparison is recorded for the same machine, backend target, model
  quant, and prompt sweep.

## Removal Criteria for C Host Code

C host code should only be removed after Rust owns the equivalent behavior and
the following are true:

- The Rust path can run the default CLI and server flows.
- Official-vector and server tests pass through the Rust path.
- Long-context and tool-call quality tests pass on at least one production
  backend.
- The old code path is no longer needed as a reference for active port work.
- Documentation and build commands clearly describe the Rust entry points.

GPU backend C/CUDA/Objective-C files may remain longer than C host logic. They
serve a different role: hardware-specific execution, not application
orchestration.

## Open Decisions

- Whether the Rust CLI/server should initially call into a C `ds4_engine` shim
  or wait until graph orchestration is ported.
- How much of the current CPU reference path should be preserved in Rust.
- Whether GGUF tooling should join the same workspace or stay as separate C and
  Python utilities until the runtime port stabilizes.
- How to version KV persistence if Rust needs to change the on-disk structure.
- Which intermediate tensor checks are worth keeping as permanent diagnostics.
- What numeric tolerances should be used per backend and per operation family.
- What speed regression threshold is acceptable for each backend milestone.
