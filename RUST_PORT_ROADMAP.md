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

### Milestone 4 Work Items

Milestone 4 must keep metadata parsing separate from runtime execution. The
first Rust parser should compare against a stable C dump before it feeds any
generation path.

#### M4.1: Model Metadata Work Item Breakdown

- Goal: split Milestone 4 into reviewable GGUF/model-metadata parity work
  items before adding parser code.
- Oracle: the Milestone 4 roadmap, `ds4.c` GGUF loader, metadata validators,
  and tensor binding code.
- Fixture: source evidence from `model_open`, `parse_metadata`,
  `parse_tensors`, `config_validate_model`, `weights_bind`, and
  `mtp_weights_bind`.
- Comparator: documentation-only review; no source behavior changes.
- Acceptance: each Milestone 4 implementation item names a tangible goal,
  oracle, fixture, comparator, acceptance rule, drift policy, review gate, and
  validation gate.
- Drift policy: no implementation or fixture-format drift allowed.
- Review gate: ask Claude to review the roadmap split for oversized or missing
  metadata parity work.
- Validation gate: `git diff --check`.

#### M4.2: C Metadata Dump Oracle

- Goal: add a current-C metadata dump helper that opens GGUF through the
  existing loader and emits deterministic machine-readable metadata without
  running inference.
- Oracle: current `model_open`, `parse_metadata`, `parse_tensors`,
  `config_validate_model`, `weights_bind`, and, when an MTP path is supplied,
  `mtp_weights_bind`.
- Fixture: the B300 q2-imatrix model identity from `.memory/status.md`, plus
  any small synthetic GGUF fixtures needed to exercise loader failures.
- Comparator: a `ds4-parity` checker that parses the C dump JSON and validates
  schema, deterministic ordering, selected required metadata values, tensor
  directory summaries, tensor type histograms, and bound semantic tensor names.
- Acceptance: the dump includes GGUF version, metadata count, tensor count,
  alignment, tensor data offset, model size, selected required scalars and
  arrays, tensor type histogram, all tensor descriptors, and the bound base/MTP
  tensor table needed by later Rust comparisons.
- Drift policy: the dump is an oracle format; schema changes must be
  append-only or update the comparator and fixtures in the same commit.
- Review gate: ask Claude to review the dump schema against the C loader and
  binding code for missing semantic fields.
- Validation gate: build the helper, run it on any local small fixtures, route
  supported-model capture to B300 when needed, run the schema checker, and run
  `git diff --check`.

#### M4.3: Rust GGUF Directory Parser

- Goal: add a `ds4-gguf` parser for GGUF v3 header, metadata descriptors,
  scalar/array value decoding, tensor directory parsing, alignment, absolute
  offsets, and tensor byte sizing.
- Oracle: M4.2 C metadata dumps for the same GGUF files.
- Fixture: committed small GGUF fixtures for header/directory coverage and the
  supported-model dump captured by M4.2.
- Comparator: Rust dump output compared to the C dump for version, counts,
  key/type/value summaries, tensor names, dims, types, relative/absolute
  offsets, byte sizes, and type histograms.
- Acceptance: supported fixtures match exactly; malformed header, truncated
  metadata, unsupported type, offset overflow, and out-of-file tensor cases
  fail with the same normalized category as the C loader.
- Drift policy: this item does not port DS4-specific semantic validation; it
  only matches the generic GGUF directory surface already parsed by C.
- Review gate: ask Claude to review parser bounds checks, overflow handling,
  value decoding, and C/Rust dump comparison coverage.
- Validation gate: run Rust parser tests, C-vs-Rust dump comparison, negative
  fixture checks, `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.

#### M4.4: DS4 Metadata Validation Parity

- Goal: port DS4-specific metadata validation from `config_validate_model`,
  including required key lookup, numeric type coercions, compression ratio
  arrays, SwiGLU clamp arrays, RoPE constants, HC constants, and expert
  routing constants.
- Oracle: C validation behavior and first-failure messages from the M4.2 dump
  helper.
- Fixture: supported-model metadata plus generated mutations for missing keys,
  wrong value types, wrong scalar values, short arrays, negative compression
  ratios, and float values outside the C tolerance.
- Comparator: C and Rust validation runs compared by pass/fail, normalized
  first failing key, expected/got category, and tolerance policy.
- Acceptance: supported metadata passes; every negative fixture fails at the
  same semantic boundary or has a documented stricter Rust rejection.
- Drift policy: Rust may not relax required metadata keys or numeric
  tolerances; stricter failures must name the fixture and reason.
- Review gate: ask Claude to review the key list, coercion rules, array
  handling, and tolerance choices against `ds4.c`.
- Validation gate: run metadata validation comparator, negative tests,
  workspace Rust tests, and `git diff --check`.

#### M4.5: Tensor Binding And Layout Parity

- Goal: port the semantic tensor binding and layout checks from `weights_bind`,
  `mtp_weights_bind`, `weights_validate_layout`, and
  `mtp_weights_validate_layout`.
- Oracle: C bound tensor table and layout validation from the M4.2 dump helper.
- Fixture: supported model tensor directory, optional MTP model dump when
  available, and generated mutations for missing tensors, wrong dims, wrong
  quant types, bad routed expert types, mismatched gate/up expert types, and
  out-of-range offsets.
- Comparator: compare bound semantic tensor names to tensor descriptor
  identity, dims, type, offsets, byte size, optional-vs-required status, and
  normalized first failure.
- Acceptance: all base model layer bindings match; hash-layer-only tensors,
  compression-ratio-dependent tensors, optional `exp_probs_b`, routed expert
  quant types, and MTP-required tensors follow the C rules.
- Drift policy: optional and required tensor semantics must not drift; any MTP
  fixture gap must be recorded as blocked rather than silently skipped.
- Review gate: ask Claude to review the generated tensor name matrix and
  optional/conditional binding rules against `ds4.c`.
- Validation gate: run tensor binding comparator, negative tests, workspace
  Rust tests, and `git diff --check`.

#### M4.6: Metadata Baselines And Unified Report Integration

- Goal: commit supported-model metadata baselines and wire metadata comparison
  into the unified parity report.
- Oracle: current C metadata dump captured on the B300 q2-imatrix model with
  the recorded model path, size, and SHA256.
- Fixture: `ds4-parity/baselines/metadata/m4.6/` JSON dumps, schema metadata,
  artifact hashes, model identity, and rerun commands.
- Comparator: a metadata comparator that self-compares committed baselines,
  compares candidate C/Rust dumps, and detects scalar, array, tensor shape,
  tensor type, binding, and offset drift.
- Acceptance: local static comparison passes without the model; the unified
  report includes metadata comparison and records exact B300 refresh commands
  for model-backed recapture.
- Drift policy: model path, timestamps, and temporary workspace paths may be
  normalized; semantic metadata, tensor descriptors, and binding tables are
  exact.
- Review gate: ask Claude to review baseline size, normalization rules, and
  report integration for actionable failure output.
- Validation gate: run metadata comparator self-checks, negative tests,
  unified parity report, any required B300 capture command, and
  `git diff --check`.

#### M4.7: Unsupported GGUF Negative Fixtures

- Goal: lock down unsupported and malformed GGUF behavior before Rust metadata
  is used by runtime code.
- Oracle: current C loader and validation failures from the M4.2 dump helper.
- Fixture: small generated GGUF fixtures for invalid magic, unsupported
  version, truncation, missing required metadata, wrong metadata type, bad
  array length, unsupported tensor type, bad tensor dimension, and tensor data
  outside the file.
- Comparator: C and Rust validator runs compared by exit status and normalized
  first error category/key.
- Acceptance: every negative fixture fails before runtime execution; Rust
  matches the C failure boundary or documents a stricter rejection in the
  fixture manifest.
- Drift policy: negative fixtures must not be weakened to pass; any accepted
  behavior change requires an explicit fixture and comparator update.
- Review gate: ask Claude to review fixture construction and normalized error
  categories for over-broad matching.
- Validation gate: run negative fixture generation/comparison, workspace Rust
  tests, and `git diff --check`.

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

### Milestone 5 Work Item Adjustment

The original DSML item spans two parser surfaces with different oracle shapes:
server generated-message parsing and agent incremental streaming parsing. Split
the DSML work before implementation so each commit has one focused comparator.

#### M5.6a: Server DSML Formatting And Generated-Message Parse Parity

- Goal: port server DSML tool-call formatting, raw sampled DSML replay,
  parameter ordering, string/JSON parameter rendering, delimiter escaping, tool
  result escaping, and `parse_generated_message_ex` boundaries.
- Oracle: current `ds4-server` DSML formatter/parser helpers, exposed through a
  deterministic no-model DSML oracle dump.
- Fixture: focused DSML formatting and generated-message inputs covering
  canonical and short markers, raw replay, schema property order, string vs JSON
  parameters, escaped `</｜DSML｜parameter>`, tool results, malformed invokes,
  and DSML before/after `</think>`.
- Comparator: C/Rust byte comparison for rendered DSML blocks and parsed
  generated-message JSON.
- Acceptance: exact DSML block bytes match; parsed server tool-call names, ids,
  argument JSON, ordering, reasoning/content split, and accepted/rejected
  generated-message boundaries match C.
- Drift policy: no DSML byte drift and no parser broadening that turns ordinary
  prose or pre-`</think>` text into executable tool calls.
- Review gate: ask Claude to review escaping, parameter ordering, raw replay,
  and generated-message parser boundaries.
- Validation gate: run the DSML oracle checker/comparator, Rust tests,
  existing `./ds4_test --server`, `cargo test --workspace`, and
  `git diff --check`.

#### M5.6b: Agent DSML Streaming Parse Parity

- Goal: port `agent_dsml_parse` streaming behavior for incremental generated
  DSML, including parser states, buffering, emitted calls, and error/truncated
  boundaries.
- Oracle: current `ds4_agent.c` streaming parser behavior exposed through a
  deterministic no-model oracle dump or fixture runner.
- Fixture: the M5.6a DSML generated-message cases plus chunk schedules for
  whole-message, one-byte, marker-prefix, escaped-delimiter,
  parameter-boundary, `</think>`, malformed tag, and truncated-at-EOF inputs for
  unterminated tool-call, invoke, parameter, and think blocks.
- Comparator: C/Rust state-transition and event comparison for every fixture
  and chunk schedule.
- Acceptance: streaming transitions, emitted tool-call events, buffered text,
  error categories, and final parser state match C for every committed schedule.
- Drift policy: no streaming parser broadening; partial or malformed DSML must
  stay buffered or rejected wherever C does.
- Review gate: ask Claude to review chunk coverage, EOF semantics, and parser
  state categories for over-broad matching.
- Validation gate: run the agent DSML comparator, Rust tests,
  `cargo test --workspace`, and `git diff --check`.

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

### Milestone 6 Work Item Adjustment

Milestone 6 spans pure sampler math, model-backed logits capture, token-byte
presentation, and request-surface stop policy. Split it before implementation
so no commit mixes no-model fixed-logit fixtures with B300 model recapture or
server/CLI finish semantics.

#### M6.2: C Fixed-Logits Sampling And Logprob Oracle

- Goal: expose current C sampling and logprob math through a deterministic
  no-model oracle dump over fixed logits arrays.
- Oracle: `sample_argmax`, `sample_rng_next`, `sample_top_p_min_p`,
  `ds4_session_top_logprobs`, and `ds4_session_token_logprob` behavior in the
  current C implementation.
- Fixture: synthetic logits arrays covering greedy ties, non-finite logits,
  temperature normalization, `top_p` clamping, `top_k` caps, `min_p`
  thresholds, full-vocab sampling, seeded RNG draws, top-logprob ordering, and
  per-token logprob requests. The fixture also includes source-named resolved
  request-surface sampling tuples for CLI defaults, OpenAI chat/responses
  defaults, Anthropic defaults, agent defaults, thinking-mode sampling defaults,
  and deterministic structural DSML sampling defaults, recorded as explicit
  `temperature`, `top_k`, `top_p`, `min_p`, and seed inputs where applicable.
- Comparator: schema checker and negative tests for the C oracle dump.
- Acceptance: oracle output is deterministic, local, no-model, and records
  selected token, consumed RNG state, filtered candidate set, logits,
  logprobs, and first-failure paths for drift.
- Drift policy: no sampler or logprob behavior changes; fixture formatting may
  normalize paths and timestamps only.
- Review gate: ask Claude to review fixture coverage against C sampler and
  logprob source.
- Validation gate: build the C oracle helper, run schema and negative checks,
  and run `git diff --check`.

#### M6.3: Rust Sampler And Logprob Math

- Goal: port C sampler, RNG, top-logprob, and token-logprob math to Rust
  without depending on model execution.
- Oracle: the M6.2 fixed-logits C oracle dump.
- Fixture: the committed M6.2 synthetic logits fixture set.
- Comparator: C/Rust comparison for selected token, RNG state, candidate
  filtering, top-logprob ordering, per-token logprob, and numeric tolerance.
- Acceptance: greedy choices and sampled choices match exactly for every seeded
  fixture, including direct parameter combinations used by request surfaces
  after they resolve defaults, thinking-mode sampling defaults, and
  deterministic structural DSML sampling to explicit `temperature`, `top_k`,
  `top_p`, and `min_p` values; logprob values match within the explicit M6
  numeric tolerance.
- Drift policy: no selection drift; any stricter non-finite handling must be
  source-proven and named by fixture.
- Review gate: ask Claude to review Rust numeric edge cases, candidate
  filtering order, RNG semantics, and allocation behavior.
- Validation gate: Rust tests, sampler comparator with negative tests,
  `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.

#### M6.4: Current-C Session Logits And Logprob Fixture Oracle

- Goal: capture model-backed current-C session logits and top-logprob slices
  for official-vector prompt cases without requiring Rust runtime execution.
- Oracle: current C `ds4_session_sync`, `ds4_session_argmax`,
  `ds4_session_top_logprobs`, `ds4_session_token_logprob`, and
  `ds4_session_eval` on the B300 q2-imatrix model.
- Fixture: official-vector prompt cases, selected continuation tokens, full
  logits or hashed logits payloads for each scored step, top-logprob slices,
  token-byte renderings, context settings, backend, model identity, and exact
  B300 refresh commands.
- Comparator: schema/hash checker that validates the committed model-backed
  logits fixture and records skipped local refresh with B300 rerun commands.
- Acceptance: selected greedy tokens match the existing official-vector
  contract, top-logprob slices are deterministic for the recorded backend, and
  the fixture is small enough to commit or explicitly shards large binary
  payloads with hashes.
- Drift policy: model path and capture workspace may be normalized; logits
  bytes or hashes, selected token IDs/bytes, top-logprob order, token bytes,
  backend, and model hash are exact.
- Review gate: ask Claude to review capture schema, artifact size policy, and
  B300 refresh command fidelity.
- Validation gate: B300 capture or exact blocked rerun command, local
  schema/hash checks, existing M0.3 logprob comparator, and
  `git diff --check`.

#### M6.5: Rust Fixed-Logits Model-Slice Comparator

- Goal: run Rust sampler and logprob math over the M6.4 captured model logits
  slices and compare token presentation against current C.
- Oracle: M6.4 current-C session logits and top-logprob fixture.
- Fixture: committed M6.4 logits payloads plus the tokenizer identity fixture
  already used by Milestone 5.
- Comparator: Rust fixed-logits dump compared to C selected token, top-logprob
  order, logprob values, token IDs, and token bytes.
- Acceptance: Rust chooses the same greedy token for every model-backed step,
  computes the same top-logprob ordering, and renders token bytes identically.
- Drift policy: no token, ordering, or byte drift; numeric differences must
  stay within the M6 tolerance and report max absolute delta.
- Review gate: ask Claude to review fixture loading, token-byte conversion, and
  tolerance reporting.
- Validation gate: model-slice comparator with negative tests, Rust tests,
  `cargo test --workspace`, and `git diff --check`.

#### M6.6a: Decode Stop Policy C Oracle Fixtures

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
  invalidation requirement, and stop boundary offsets.
- Acceptance: EOS, length, stop sequence, UTF-8 boundary, and complete
  tool-call finish outcomes match C for every fixture.
- Drift policy: no finish-reason or emitted-text drift; policy-only
  normalizations must not hide token/text boundary changes.
- Distinct from M5.6: M5.6 owns DSML byte formatting and parser state; this
  item owns only the decode loop decision once parser/tracker state reports a
  complete tool-call boundary.
- Review gate: ask Claude to review stop sequence coverage and the boundary
  between sampler math, M5 DSML parsing, and API finish semantics.
- Validation gate: C policy oracle checker with negative tests, existing
  server tests, and `git diff --check`.

#### M6.6b: Rust Decode Stop Policy Port

- Goal: port the request-surface decode stop policy over no-model
  generated-token/text schedules without implementing Rust CLI/server runtime.
- Oracle: the M6.6a C decode stop policy oracle dump.
- Fixture: committed M6.6a stop-policy schedules and request option records.
- Comparator: C/Rust policy comparison for finish reason, emitted visible text,
  held streaming tail, session invalidation requirement, and stop boundary
  offsets.
- Acceptance: Rust policy output matches C for every EOS, length, stop
  sequence, UTF-8 boundary, and complete tool-call fixture.
- Drift policy: no finish-reason, emitted-text, held-tail, or boundary-offset
  drift.
- Review gate: ask Claude to review the Rust policy boundary and make sure it
  does not reimplement M5 DSML parsing or require model execution.
- Validation gate: policy comparator with negative tests, Rust tests,
  `cargo test --workspace`, and `git diff --check`.

#### M6.7: Sampling And Logprob Report Integration

- Goal: wire M6 local comparators and B300 refresh records into the parity
  reports.
- Oracle: committed M6 fixed-logits, model-backed logits, and decode-policy
  fixtures.
- Fixture: M6.2 through M6.6b manifest entries and refresh commands.
- Comparator: a Milestone 6 report that runs all local sampling/logprob
  comparators, summarizes numeric tolerances and first drift paths, and skips
  only model-backed recapture with exact B300 commands; the unified parity
  report includes that M6 report.
- Acceptance: local report passes without the model, JSON output is machine
  readable, failure output names fixture/field/expected/got, and B300 refreshes
  are reproducible from the report.
- Drift policy: report normalizes only capture paths and timestamps.
- Review gate: ask Claude to review report integration and failure output.
- Validation gate: M6 report, unified parity report, `py_compile`,
  `cargo test --workspace`, and `git diff --check`.

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

### Milestone 7 Work Item Adjustment

Milestone 7 spans on-disk cache file structure, cache policy decisions,
session-payload boundaries, in-memory session snapshots, request-level replay,
and model-backed restore validation. Split it before implementation so no
commit mixes local no-model KV format fixtures with B300 snapshot/logit
recapture.

#### M7.1: KV Store Work Item Breakdown

- Goal: split Milestone 7 into reviewable KV, session-payload, replay, and
  report work items before adding Rust persistence code.
- Oracle: current `ds4_kvstore` and session snapshot implementation, plus the
  committed M0.5 KV/cache artifacts.
- Fixture: M0.5 `kv-artifacts`, current `ds4_kvstore` source, current session
  payload source, trailer-hook source, in-memory snapshot API surface, and
  existing parity-report conventions.
- Comparator: documentation-only review; no source behavior changes.
- Acceptance: each Milestone 7 item names a tangible goal, oracle, fixture,
  comparator, acceptance rule, drift policy, review gate, and validation gate.
- Drift policy: no implementation or fixture-format drift allowed.
- Review gate: ask Claude to review that the split isolates header/policy
  fixtures, Rust format parsing, generic full-file round trips, per-extension
  trailer coverage, on-disk payload structure, in-memory snapshot restore,
  request replay, B300 recapture, and report integration without mixing oracle
  surfaces.
- Validation gate: `git diff --check`.

#### M7.2: C KV Header And Policy Oracle

- Goal: expose current C KV-cache header, filename, and policy behavior through
  a deterministic no-model oracle dump.
- Oracle: `ds4_kvstore` no-model helpers in the current C implementation:
  fixed header layout, `ds4_kvstore_fill_header`,
  `ds4_kvstore_read_header`, `ds4_kvstore_read_entry_file`,
  `ds4_kvstore_default_options`, `ds4_kvstore_reason_code`,
  `ds4_kvstore_key_kind`, `ds4_kvstore_store_len`,
  `ds4_kvstore_chat_anchor_pos`, `ds4_kvstore_continued_store_target`,
  `ds4_kvstore_file_size_fits`, `ds4_kvstore_entry_eviction_score`,
  `ds4_kvstore_find_text_prefix`, `ds4_kvstore_byte_prefix_match`,
  SHA/path helpers, little-endian helpers, reason encoding, and extension flag
  encoding. Model/session-bound helpers such as token rendering,
  `store_live_prefix`, `maybe_store_continued`, and `try_load_text` are out of
  scope for this no-model oracle.
- Fixture: synthetic text bytes, token IDs, cache entries, timestamps, file
  sizes, option records, explicit `now` values for eviction scoring, and
  committed M0.5 parsed header rows. Entry `created_at` and `last_used` are
  fixture inputs, not capture-time values.
- Comparator: schema checker and negative tests for the C oracle dump,
  including exact header bytes, decoded fields, selected path names, policy
  outputs, and first-failure paths.
- Acceptance: oracle output is deterministic, local, no-model, and captures
  the current KVC header bytes, field decoding, SHA keying, prefix selection,
  eviction ordering, store target decisions, and boundary edge cases.
- Drift policy: no KV policy or format behavior changes; fixture formatting may
  normalize paths and timestamps only.
- Review gate: ask Claude to review fixture coverage against `ds4_kvstore`
  source and M0.5 artifacts.
- Validation gate: build the C oracle helper, run schema and negative checks,
  run the existing server/KV unit surface, and run `git diff --check`.

#### M7.3: Rust KV Header And Policy Parser

- Goal: port the KVC header parser/writer and no-model KV policy decisions to
  Rust without loading model sessions.
- Oracle: the M7.2 current-C KV header and policy oracle dump.
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
- Validation gate: KV comparator with negative tests, Rust unit tests,
  `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.

#### M7.4a: Generic KVC Full-File Round Trip

- Goal: compare full KVC file construction, generic optional trailer bytes,
  file-size budgeting, and cross-reader acceptance without restoring model
  tensors.
- Prerequisite: M7.3 Rust KVC header writer and no-model policy comparator.
- Oracle: current C fixed-header/text/payload file layout,
  `ds4_kvstore_trailer_hooks`, current C entry-file reader behavior, and C
  `ds4_kvstore_file_size_fits` for the produced file sizes.
- Fixture: synthetic cache text, opaque payload bytes, fixed timestamps, option
  records, generic extension-flag combinations, opaque trailer bytes, and
  truncated/corrupted header, text, payload, and trailer data.
- Comparator: C writer versus Rust writer byte comparison for the complete KVC
  file, Rust reader acceptance of C-written files, C reader acceptance of
  Rust-written files, and negative tests for malformed header/text/payload and
  trailer boundaries.
- Acceptance: full files are byte-identical for the fixed-header, text,
  payload, and trailer fixture; C can read Rust-written metadata/trailer files;
  Rust can read C-written metadata/trailer files; malformed files fail at the
  same boundary category; Rust writer output size equals the C policy
  `file_size_fits` budget input for each fixture.
- Drift policy: no KVC full-file byte drift, extension-flag drift, trailer-size
  drift, or cross-reader acceptance drift; opaque payload bytes remain
  uninterpreted in this item.
- Review gate: ask Claude to review generic trailer-hook coverage, full-file
  byte identity, file-size budget cross-checks, and C-reads-Rust/Rust-reads-C
  round-trip evidence.
- Validation gate: full-file comparator with negative tests, C helper build,
  Rust tests, `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.

#### M7.4b: KV Extension Trailer Payload Coverage

- Goal: compare server-owned KVC extension payloads and extension-flag
  semantics separately from the generic full-file round trip.
- Prerequisite: M7.4a generic full-file round-trip comparator.
- Oracle: server tool-map trailer format (`KTM` version 1),
  `DS4_KVSTORE_EXT_TOOL_MAP`, `DS4_KVSTORE_EXT_RESPONSES_VISIBLE`,
  `DS4_KVSTORE_EXT_THINKING_VISIBLE`, and current C trailer write/load helper
  behavior.
- Fixture: tool-map trailer entries with boundary cases for zero entries,
  multiple entries, UTF-8 bytes, long IDs, long DSML records, duplicate-shaped
  entries, visible-transcript extension flags without payload bytes, and
  truncated/corrupted trailer data.
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
- Validation gate: extension trailer comparator with negative tests, C helper
  build, Rust tests, `cargo fmt --all -- --check`, `cargo test --workspace`,
  and `git diff --check`.

#### M7.5: C Session Payload Shape Oracle

- Goal: expose current C session snapshot payload structure, size budgeting,
  and on-disk payload-header rejection behavior before any Rust payload reader.
- Oracle: `DS4_SESSION_PAYLOAD_MAGIC`, session payload version and fields,
  `ds4_session_payload_bytes`, `ds4_session_save_payload`,
  and `ds4_session_load_payload` behavior in the current C implementation.
  In-memory `ds4_session_save_snapshot` and `ds4_session_load_snapshot` have no
  on-disk magic or byte layout and are deliberately excluded from this item.
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
- Validation gate: C payload oracle helper or exact B300 blocked command,
  schema/hash checker with negative tests, and `git diff --check`.

#### M7.6: Rust Session Payload Header Reader

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
- Validation gate: payload comparator with negative tests, Rust tests,
  `cargo fmt --all -- --check`, `cargo test --workspace`, and
  `git diff --check`.

#### M7.7: KV Replay And Prefix Decision Comparator

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
- Validation gate: replay comparator with negative tests, Milestone 5 fixture
  hash precondition check, `cargo test --workspace`, and `git diff --check`.

#### M7.8: B300 Disk KV And In-Memory Snapshot Restore Oracle

- Goal: capture model-backed current-C evidence for both disk KV/session
  payload restore and in-memory `ds4_session_snapshot` restore.
- Oracle: current C server/session save and restore paths on the recorded B300
  model, `ds4_session_save_payload`, `ds4_session_load_payload`,
  `ds4_session_save_snapshot`, `ds4_session_load_snapshot`,
  `ds4_session_top_logprobs`, and selected-token output after restore.
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
- Validation gate: B300 capture or exact skipped recapture command, local
  schema/hash checker with negative tests, and `git diff --check`.

#### M7.9: KV And Snapshot Report Integration

- Goal: wire M7 local comparators and B300 restore refresh records into the
  parity reports.
- Oracle: committed M7.2 through M7.8 fixtures and refresh commands.
- Fixture: M7 manifest entries, local comparator commands, M0.5 baseline
  artifacts, and B300 restore recapture records.
- Comparator: a Milestone 7 report that runs all local KV/snapshot comparators,
  summarizes first drift paths, and skips only model-backed B300 recapture with
  exact commands; the unified parity report includes that M7 report.
- Acceptance: local report passes without the model, JSON output is machine
  readable, failure output names fixture/field/expected/got, and B300
  refreshes are reproducible from the report.
- Drift policy: report normalizes only capture paths and timestamps.
- Review gate: ask Claude to review report integration and skipped-B300
  command fidelity.
- Validation gate: M7 report, unified parity report, `py_compile`,
  `cargo test --workspace`, and `git diff --check`.

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

Work items:

#### M8.1: CLI Surface Work Item Breakdown

- Goal: split Milestone 8 into reviewable CLI parity work items before adding
  Rust CLI behavior.
- Oracle: current `ds4_cli.c` option parser, invocation modes, diagnostics, and
  REPL command surface.
- Fixture: roadmap text plus current CLI source evidence for prompt sources,
  backend flags, sampling flags, thinking controls, diagnostics, one-shot
  generation, and interactive commands.
- Comparator: documentation-only diff that assigns an oracle, fixture,
  comparator, acceptance rule, and validation gate to each executable CLI item.
- Acceptance: each later M8 item is small enough to validate independently and
  names whether it needs a local no-model fixture, B300 model capture, PTY
  transcript, or Rust implementation comparator.
- Drift policy: no source behavior changes.
- Review gate: ask Claude to review that the split is verifiable and comparable
  to the current C CLI.
- Validation gate: docs/state-only diff inspection and `git diff --check`.
- Scope note: `--head-test`, `--first-token-test`, `--metal-graph-test`,
  `--metal-graph-full-test`, and `--metal-graph-prompt-test` are internal
  graph/runtime diagnostics whose success-path parity belongs with Milestone 10
  runtime graph orchestration. M8.2 captures their parser/help surface and
  removed-flag errors; M8 does not claim Rust success-path parity for those
  debug modes.

#### M8.2: Current-C CLI Parse And Error Oracle

- Goal: capture the no-model CLI argument, help, and early error surface.
- Oracle: current `./ds4` parser in `ds4_cli.c` before model loading.
- Fixture: `--help`, missing option values, unknown options, invalid numeric
  and float values, invalid backend names, duplicate prompt sources,
  `--server`, removed `--metal-graph-generate`, `--dump-tokens` without a
  prompt, imatrix option coupling, and `--perplexity-file` prompt-source
  rejection.
- Comparator: schema checker for exit status, stdout/stderr category, help text
  anchors, and exact option names.
- Acceptance: all cases are local and model-free; exit code and stderr category
  match exactly, with help text compared by stable section anchors.
- Drift policy: executable path and compiler diagnostics may be normalized; CLI
  option spelling, exit status, and user-facing error category are exact.
- Review gate: ask Claude to review coverage for parser branches and accidental
  model-loading cases.
- Validation gate: local C capture, schema checker with negative tests, and
  `git diff --check`.

#### M8.3: Rust CLI Parse And Error Parity

- Goal: implement Rust CLI parsing for the M8.2 no-model surface.
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
- Validation gate: Rust CLI parser tests, comparator with negative tests,
  `cargo test --workspace`, and `git diff --check`.

#### M8.4: Current-C CLI Token And Prompt Diagnostic Oracle

- Goal: capture current-C CLI prompt ingestion and token-dump behavior.
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
- Review gate: ask Claude to review prompt-source and thinking-control
  coverage against `ds4_cli.c`.
- Validation gate: B300 capture or exact skipped recapture command, local
  checker with negative tests, and `git diff --check`.

#### M8.5: Rust CLI Token And Prompt Diagnostic Parity

- Goal: implement Rust CLI behavior for `--dump-tokens` and prompt-source
  diagnostics.
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
- Validation gate: comparator with negative tests, targeted Rust CLI tests,
  `cargo test --workspace`, and `git diff --check`.

#### M8.6: Current-C CLI Logprob And Perplexity Oracle

- Goal: capture current-C CLI machine-readable diagnostic outputs that require
  model execution.
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
- Validation gate: B300 capture or exact skipped recapture command, local
  checker with negative tests, and `git diff --check`.

#### M8.7: Rust CLI Logprob And Perplexity Parity Split

- Goal: split the original Rust CLI logprob/perplexity parity item because it
  requires model/session execution that the Rust tree does not yet expose.
- Oracle: repository evidence that Rust currently has tokenizer, prompt,
  fixed-logits sampler/logprob, model-logits replay, and GPU tensor wrappers,
  but no `ds4_engine`/`ds4_session` execution boundary.
- Comparator: roadmap/board review against `rust/ds4-gguf/src/sampling.rs`,
  `rust/ds4-gguf/src/bin/ds4-model-logits-dump-rs.rs`,
  `rust/ds4-gpu/src/lib.rs`, and `rust/ds4-gpu-sys/src/lib.rs`.
- Acceptance: the original model-backed parity claim is not implemented by a
  replay-only proxy; it is decomposed into separately verifiable prerequisites
  and remains blocked until a Rust runtime/session path exists.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split avoids overstating Rust
  model-backed parity and keeps each successor item comparable to current C.
- Validation gate: inspect the cited Rust paths, `git diff --check`, and
  review.

#### M8.7a: Rust Diagnostic Runtime Boundary Prerequisite

- Goal: introduce or expose a Rust-accessible model/session execution boundary
  sufficient to run one prompt prefill, query next-token logits/top-logprobs,
  evaluate a selected token, and score teacher-forced tokens. A thin FFI-backed
  boundary over current C is acceptable for this milestone if the comparator can
  later hold a Rust-owned implementation to the same contract.
- Oracle: current C `ds4_engine_open`, `ds4_session_create`,
  `ds4_session_sync`, `ds4_session_top_logprobs`, `ds4_session_argmax`,
  `ds4_session_token_logprob`, and `ds4_session_eval` on the M8.6 fixture
  prompts.
- Fixture: a minimal prompt/token sequence that can prove prefill, one decode
  step, top-logprob order, and teacher-forced token scoring without running the
  full M8.6 CLI surface.
- Comparator: C/Rust runtime-boundary comparator for selected token,
  top-logprob IDs/order, token-logprob result, and M6 numeric tolerances.
- Acceptance: Rust can execute the minimal diagnostic session path against the
  same model/backend and match current-C selected IDs/order with M6 score
  tolerance.
- Drift policy: test-driver paths, stderr, and timing may be normalized; token
  IDs/order and numeric tolerance policy are exact.
- Review gate: ask Claude to review runtime ownership, unsafe/FFI boundaries,
  and parity comparator coverage.
- Validation gate: B300 runtime comparator with negative tests, targeted Rust
  tests, `cargo test --workspace`, and `git diff --check`.

#### M8.7b: Rust CLI Diagnostic Output Surface

- Goal: route the runtime-boundary outputs through Rust CLI diagnostic
  formatting for `--dump-logprobs` and `--perplexity-file`.
- Oracle: committed M8.6 current-C CLI diagnostic fixture plus the M8.7a
  runtime-boundary comparator.
- Fixture: same M8.6 CLI cases, run through Rust once M8.7a exists.
- Comparator: C/Rust comparator for JSON shape, selected tokens, top-logprob
  ordering, score tolerances, perplexity text fields, stderr categories, and
  error exits.
- Acceptance: Rust emits the same machine-readable diagnostic surface and
  preserves M6 numeric tolerance policy without using current-C JSON replay as
  a substitute for model execution.
- Drift policy: path and progress-stderr normalization only.
- Review gate: ask Claude to review model-backed CLI diagnostic parity and
  failure categories.
- Validation gate: comparator with negative tests, targeted Rust tests, B300
  refresh if required by the comparator, `cargo test --workspace`, and
  `git diff --check`.

#### M8.8: Current-C CLI Inspect Output Oracle

- Goal: capture the current-C `--inspect` CLI output surface.
- Oracle: current `./ds4 --inspect` on the recorded B300 model.
- Fixture: model path, backend selection, summary stdout/stderr records, model
  identity, exit status, and exact B300 refresh commands.
- Comparator: schema/hash checker for summary output anchors, model/backend
  identity, exit status, and refresh commands.
- Acceptance: summary output anchors and model identity match current C; no
  generation, REPL, perplexity, or imatrix path is entered.
- Drift policy: workspace paths and volatile memory addresses may be
  normalized; model identity, summary sections, and exit status are exact.
- Review gate: ask Claude to review inspect-output coverage against
  `ds4_engine_summary` dispatch.
- Validation gate: B300 capture or exact skipped recapture command, local
  checker with negative tests, and `git diff --check`.

#### M8.9: Rust CLI Inspect Output Parity Split

- Goal: split the original Rust CLI `--inspect` parity item because the Rust
  tree currently recognizes the option in parser code but has no engine-open or
  engine-summary boundary that can produce the committed M8.8 model summary.
- Oracle: repository evidence that `rust/ds4-gguf/src/cli_parse.rs` accepts
  `--inspect` only as a parse-through flag while `parse_cli` still returns the
  parser-only model-backed-path stub, and `rust/ds4-gpu*` expose tensor/cache
  primitives but no `ds4_engine_open`/`ds4_engine_summary` equivalent.
- Comparator: roadmap/board review against `rust/ds4-gguf/src/cli_parse.rs`,
  `rust/ds4-gpu/src/lib.rs`, `rust/ds4-gpu-sys/src/lib.rs`, and the committed
  M8.8 current-C inspect fixture.
- Acceptance: the original inspect parity claim is not implemented by a fake
  summary or artifact replay; it is decomposed into a runtime-boundary
  prerequisite and the CLI parity surface that runs on top of it.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split keeps inspect parity
  verifiable against the M8.8 current-C oracle without overstating Rust runtime
  support.
- Validation gate: inspect the cited Rust paths, `git diff --check`, and
  review.

#### M8.9a: Rust Inspect Runtime Boundary Prerequisite

- Goal: introduce or expose a Rust-accessible engine-open and engine-summary
  boundary sufficient to load the M8.8 model/backend and emit the same summary
  surface without entering generation, REPL, perplexity, or imatrix paths. A
  thin FFI-backed boundary over current C is acceptable for this milestone if a
  future Rust-owned implementation can be held to the same comparator.
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
- Validation gate: B300 runtime comparator with negative tests, targeted Rust
  tests, `cargo test --workspace`, and `git diff --check`.

#### M8.9b: Rust CLI Inspect Output Surface

- Goal: route Rust CLI `--inspect` handling through the M8.9a runtime boundary.
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
- Validation gate: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison when required, `cargo test --workspace`, and
  `git diff --check`.

#### M8.10: Current-C CLI Imatrix Capture Oracle Split

- Goal: split the original current-C CLI imatrix capture oracle because the
  roadmap assumed the B300 model host could execute it, but the current C
  imatrix collector is Metal-only and `--imatrix-out` forces the CLI backend to
  Metal before collection.
- Oracle: source evidence in `ds4_cli.c` and `ds4.c`, plus B300 execution
  evidence from the recorded model host.
- Comparator: roadmap/board review against `ds4_cli.c` option parsing,
  `ds4_engine_open`, and `ds4_engine_collect_imatrix`.
- Acceptance: the original output-hash oracle is not claimed from a B300 CUDA
  run; it is decomposed into a captured feasibility guard and a blocked
  output-oracle item that requires either a Metal-capable host with the
  recorded model or current-C CUDA imatrix support.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split avoids overstating
  current-C output coverage and preserves an exact future capture contract.
- Validation gate: B300 forced-Metal failure proof, local model-host
  availability check, `git diff --check`, and review.

#### M8.10a: Current-C CLI Imatrix Feasibility Guard

- Goal: capture the current reason the original M8.10 output oracle cannot run
  on the B300 CUDA model host.
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
- Validation gate: B300 proof command, local model availability check, and
  `git diff --check`.

#### M8.10b: Current-C CLI Imatrix Output Oracle

- Goal: capture the current-C CLI imatrix execution mode once a valid host is
  available.
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
- Validation gate: output capture, checker with negative tests, and
  `git diff --check`.

#### M8.11: Rust CLI Imatrix Capture Parity

- Goal: implement Rust CLI parity for imatrix capture mode.
- Oracle: committed M8.10b current-C imatrix fixture.
- Fixture: same dataset, limit, context, backend, and output-path cases as
  M8.10b.
- Comparator: C/Rust imatrix comparator for output file hash/size, limit
  accounting, exit status, and normalized stderr categories.
- Acceptance: Rust writes the same imatrix output bytes for the committed
  dataset and preserves the current C limit semantics.
- Drift policy: timing/progress/path normalization only.
- Review gate: ask Claude to review file-output determinism and limit handling.
- Validation gate: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison when required, `cargo test --workspace`, and
  `git diff --check`.

#### M8.12: Current-C CLI One-Shot Generation Oracle Split

- Goal: split the broad current-C one-shot generation oracle into core
  transcript behavior and advanced runtime-control behavior before committing
  fixtures.
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
- Validation gate: roadmap/board diff, `git diff --check`, and review.

#### M8.12a: Current-C CLI One-Shot Core Transcript Oracle

- Goal: capture deterministic current-C one-shot CLI generation transcripts for
  the core prompt and thinking-control surface.
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
- Validation gate: B300 capture or exact skipped recapture command, local
  transcript checker with negative tests, and `git diff --check`.

#### M8.12b: Current-C CLI One-Shot Runtime-Control Oracle

- Goal: capture current-C one-shot generation behavior for advanced runtime
  controls after the core transcript oracle is stable.
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
- Validation gate: B300 capture or exact blocked recapture command, local
  transcript checker with negative tests, and `git diff --check`.

#### M8.13: Rust CLI One-Shot Generation Parity Split

- Goal: split the original Rust CLI one-shot parity item because the Rust
  runtime currently exposes engine open/summary but not prompt encoding,
  generated-token text, argmax generation callbacks, or session sampling.
- Oracle: committed M8.12a/M8.12b current-C transcript fixtures plus source
  evidence from `rust/ds4-engine/src/lib.rs`, `ds4.h`, and `ds4_cli.c`.
- Comparator: roadmap/board review that successor items introduce executable
  Rust runtime boundaries before claiming CLI transcript parity.
- Acceptance: no Rust one-shot transcript is produced by replaying current-C
  stdout; the item is decomposed into runtime-boundary prerequisites and CLI
  surface parity items.
- Drift policy: no source behavior changes; this is roadmap scope control.
- Review gate: ask Claude to review whether the split avoids overstating Rust
  runtime ownership and preserves M8.12a/M8.12b comparison coverage.
- Validation gate: source inspection, roadmap/board diff, `git diff --check`,
  and review.

#### M8.13a: Rust Argmax One-Shot Runtime Boundary

- Goal: expose a Rust-accessible one-shot argmax generation boundary over the
  current engine API without implementing the full CLI surface yet.
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
- Validation gate: B300 runtime comparator with negative tests, targeted Rust
  tests, `cargo test --workspace`, and `git diff --check`.

#### M8.13b: Rust Session Sampling Runtime Boundary

- Goal: expose the session-backed Rust runtime boundary needed for seeded
  non-greedy one-shot sampling and future MTP speculation.
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
- Validation gate: B300 runtime comparator with negative tests, targeted Rust
  tests, `cargo test --workspace`, and `git diff --check`.

#### M8.13c: Rust CLI One-Shot Core Transcript Surface

- Goal: route the Rust CLI one-shot core surface through the M8.13a/M8.13b
  runtime boundaries.
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
- Validation gate: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison run, `cargo test --workspace`, and `git diff --check`.

#### M8.13d: Rust CLI One-Shot Runtime-Control Surface

- Goal: extend Rust CLI one-shot parity to the M8.12b runtime-control cases.
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
- Validation gate: comparator with negative tests, targeted Rust CLI tests,
  B300 comparison run, `cargo test --workspace`, and `git diff --check`.

#### M8.14: Current-C Interactive CLI Transcript Oracle

- Goal: capture scripted current-C interactive CLI behavior.
- Oracle: current `./ds4` REPL using a PTY so `linenoise`, prompts, Ctrl+C, and
  command output are represented.
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
- Validation gate: local or B300 PTY capture, checker with negative tests, and
  `git diff --check`.

#### M8.15: Rust Interactive CLI Transcript Parity

- Goal: split Rust interactive CLI parity before implementation because the
  current Rust runtime exposes one-shot generation but not reusable sessions,
  chat transcript mutation, session progress callbacks, or REPL state.
- Oracle: committed M8.14 current-C PTY transcript fixture plus source evidence
  from `ds4_cli.c`, `ds4.h`, and `rust/ds4-engine/src/lib.rs`.
- Fixture: M8.14 `command_suite` and `ctrl_c_at_prompt` cases.
- Comparator: roadmap/board review that successor items separate reusable
  runtime ownership, REPL command state, and final PTY transcript parity.
- Acceptance: broad interactive parity is decomposed into runtime-boundary,
  command-state, and PTY-surface items before source behavior changes.
- Drift policy: no source behavior changes; this is scope control.
- Review gate: ask Claude to review whether the split preserves all M8.14
  transcript coverage and does not hide runtime gaps.
- Validation gate: source inspection, roadmap/board diff, and `git diff --check`.

#### M8.15a: Rust Reusable Interactive Session Boundary

- Goal: expose the Rust runtime primitives needed for interactive turns without
  building the REPL surface yet.
- Oracle: current C `ds4_chat_begin`, `ds4_chat_append_message`,
  `ds4_chat_append_assistant_prefix`, `ds4_chat_append_max_effort_prefix`,
  `ds4_tokens_push`, reusable `ds4_session_*` APIs, and the M8.14 model-backed
  `/read` plus direct-prompt turn outputs.
- Fixture: B300 `/workspace/ds4/ds4flash.gguf`, context 128, one generated
  token, `--temp 0`, `--nothink`, the M8.14 `/read` fixture prompt, and the
  direct prompt `Answer with one short noun: glacier.` after the first turn.
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
- Validation gate: B300 runtime comparator with negative tests, targeted Rust
  tests, `cargo test --workspace`, and `git diff --check`.

#### M8.15b: Rust REPL Command State Surface

- Goal: implement the Rust command-state layer for the M8.14 interactive
  commands before claiming PTY transcript parity.
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
- Validation gate: targeted Rust command tests, `cargo test --workspace`, and
  `git diff --check`.

#### M8.15c: Rust Interactive PTY Transcript Surface

- Goal: wire the Rust no-prompt CLI path into an interactive PTY surface and
  compare it to the M8.14 current-C transcript.
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
- Validation gate: PTY comparator with negative tests, B300 comparison run,
  targeted Rust CLI tests, `cargo test --workspace`, and `git diff --check`.

#### M8.16: CLI Parity Report Integration

- Goal: wire M8 CLI comparators and B300 refresh records into parity reports.
- Oracle: committed M8.2 through M8.15 fixtures and refresh commands.
- Fixture: M8 manifest entries, local comparator commands, B300 recapture
  records, and PTY transcript records.
- Comparator: a Milestone 8 report that runs all local CLI comparators,
  summarizes first drift paths, and skips only model-backed B300 recaptures with
  exact commands; the unified parity report includes that M8 report.
- Acceptance: local report passes without the model, JSON output is
  machine-readable, failures name fixture/field/expected/got where underlying
  comparators provide it, and B300 refreshes are reproducible from the report.
- Drift policy: report normalizes only capture paths and timestamps.
- Review gate: ask Claude to review report integration and skipped-B300 command
  fidelity.
- Validation gate: M8 report, unified parity report, `py_compile`,
  `cargo test --workspace`, and `git diff --check`.

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

#### M9.1: Server Surface Work Item Breakdown

- Goal: split Milestone 9 into commit-sized Rust server parity work items before
  changing server behavior.
- Oracle: `ds4_server.c`, `tests/ds4_test.c --server`, M0.4 server replay
  artifacts, and M0.5 KV restore artifacts.
- Fixture: existing M0.4 request JSON fixtures, M0.4 response/header/SSE/trace
  artifacts, M0.5 KV fixtures/artifacts, and the in-C server unit vectors.
- Comparator: documentation-only breakdown assigning oracle, fixture,
  comparator, acceptance, drift policy, validation, and owner paths to each
  server parity item.
- Acceptance: the next implementation items cover request parsing, HTTP
  framing, non-streaming output, streaming output, tool protocols, cache/KV
  behavior, and report integration without mixing unrelated behavior.
- Drift policy: documentation-only; no source behavior changes.
- Review gate: ask Claude to review item boundaries and missing server
  behavioral surfaces.
- Validation gate: roadmap/board diff and `git diff --check`.

#### M9.2: Server Request Parse And Prompt Render Surface

- Goal: port the model-free request parsing and prompt-rendering surface needed
  by the server before adding an HTTP listener.
- Oracle: `parse_chat_request`, `parse_anthropic_request`,
  `parse_responses_request`, `render_chat_prompt_text`, thinking-control tests,
  stop-list tests, context-length error shape, and `ds4_server_unit_tests_run`.
- Fixture: M0.4 request JSON, server unit vectors for OpenAI/Responses/
  Anthropic messages, tool schemas, reasoning/thinking controls, stop lists,
  context limits, CORS policy inputs, and prompt-rendered text.
- Comparator: Rust unit tests and/or a dump helper comparing normalized request
  fields, rendered prompt text, stream flags, tool schema ordering, stop lists,
  thinking mode, max-token/context decisions, and protocol error categories.
- Acceptance: Rust produces the same request semantics and prompt text for the
  covered unit vectors without opening sockets or loading the model.
- Drift policy: exact for semantic fields and prompt bytes; error text may be
  compared by stable category where C text embeds paths or limits.
- Review gate: ask Claude to review parser coverage for OpenAI chat, Responses,
  Anthropic, thinking controls, stop lists, and context-limit errors.
- Validation gate: targeted Rust parser/render tests, `cargo test --workspace`,
  `python3 ds4-parity/run_cli_parity_report.py`, and `git diff --check`.

#### M9.2a: OpenAI Chat Request Core Parse And Render

- Goal: port the model-free OpenAI `/v1/chat/completions` core request parser
  and prompt renderer, excluding tool-call payloads and alternate protocols.
- Oracle: `parse_chat_request`, `render_chat_prompt_text`, request default
  tests, thinking-control tests, stop-list tests, context-limit error tests,
  and M0.4 non-tool OpenAI request fixtures.
- Fixture: M0.4 `chat_basic`, `chat_stream`, `chat_thinking_disabled`,
  `chat_cache_seed`, and `chat_cache_continuation` request JSON plus unit
  vectors for defaults, stream options, stop lists, and thinking controls.
- Comparator: Rust unit tests and/or a dump helper comparing normalized request
  fields, rendered prompt bytes, stream flags, generation options, thinking
  mode, stop lists, max-token/context decisions, and error categories.
- Acceptance: Rust matches the C parser/render semantics for non-tool OpenAI
  chat requests without opening sockets or loading the model.
- Drift policy: exact for semantic fields and prompt bytes; stable-category
  comparison for path/limit-bearing error text.
- Review gate: ask Claude to review OpenAI field coverage, default values,
  stream option handling, thinking mode mapping, and context-limit errors.
- Validation gate: targeted Rust parser/render tests, `cargo test --workspace`,
  `python3 ds4-parity/run_cli_parity_report.py`, and `git diff --check`.

#### M9.2b: OpenAI Tool Schema And DSML Prompt Render Surface

- Goal: port the model-free OpenAI tool schema parsing and DSML prompt-rendering
  surface without implementing model-backed tool-call generation.
- Oracle: `parse_tools_value`, `openai_function_schema_from_tool`,
  `append_tools_prompt_text`, `append_dsml_tool_calls_text`, tool schema order
  tests, DSML parser tests, and M0.4 `chat_tool_call` request/trace prompt.
- Fixture: M0.4 `chat_tool_call.json`, M0.4 tool trace prompt segment, and
  unit vectors for schema property order, DSML argument ordering, malformed
  tool-call recovery, partial tool-call holds, and loose nested parameters.
- Comparator: Rust unit tests/dump helper comparing tool schema normalization,
  rendered tool prompt bytes, DSML call text, executable-tool boundary
  categories, and recoverable parse categories.
- Acceptance: Rust model-free tool parsing and prompt rendering match current C
  for OpenAI tool requests before any server response generation is ported.
- Drift policy: exact for schema names, argument order, prompt bytes, and DSML
  text; random call IDs are out of scope for this parser-only item.
- Review gate: ask Claude to review schema ordering, DSML state-machine edges,
  malformed/recoverable tool parsing, and prompt placement before system text.
- Validation gate: targeted Rust tool/DSML tests, `cargo test --workspace`,
  `python3 ds4-parity/run_cli_parity_report.py`, and `git diff --check`.

#### M9.2c: Responses And Anthropic Request Parse Surface

- Goal: port the model-free Responses and Anthropic request parsing/rendering
  inputs while leaving response/event emission for M9.7.
- Oracle: `parse_responses_request`, `parse_anthropic_request`,
  `parse_responses_input`, `parse_anthropic_messages`, protocol system/tool
  validation tests, and live-tail requirement tests.
- Fixture: unit vectors for Responses namespace/tool_search schemas, reasoning
  inputs, function_call outputs, tool outputs, Anthropic content blocks,
  private system filtering, tool use/results, and live-tail validation.
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
- Validation gate: source inspection, roadmap/board diff, and
  `git diff --check`.

#### M9.2c1: Responses Core Input And Reasoning Parse Surface

- Goal: port model-free Responses API core request parsing for `input`,
  `instructions`, scalar generation controls, reasoning effort/summary flags,
  durable-state rejection, and prompt rendering, excluding tool-output
  live-tail validation and tool schemas loaded from input tool-search results.
- Oracle: `parse_responses_request`, `parse_responses_reasoning`, string/array
  `input` handling, `instructions` system prepend, model-alias thinking
  fallbacks, and `previous_response_id`/`conversation` rejection branches.
- Fixture: unit vectors for bare string input, message input arrays,
  instructions prepend, `reasoning.effort`/`reasoning.summary`, unsupported
  tool-choice categories, and durable-state non-null errors.
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
- Validation gate: targeted Rust Responses core parser tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.2c2: Responses Tool Output And Live-Tail Parse Surface

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
- Validation gate: roadmap/board diff and `git diff --check`.

#### M9.2c2a: Responses Function Call And Tool Output Input Surface

- Goal: port model-free Responses input items that become chat tool-call
  history or tool-result history: `function_call`, `custom_tool_call`,
  hosted-tool calls, `function_call_output`, custom/hosted tool outputs, call
  IDs, pending-reasoning merge rules, and DSML prompt rendering.
- Oracle: `parse_responses_input` branches for `function_call`,
  `custom_tool_call`, `local_shell_call`, `web_search_call`,
  `tool_search_call`, `image_generation_call`,
  `function_call_output`, `custom_tool_call_output`, hosted tool outputs,
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
- Validation gate: targeted Rust Responses function/tool input tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.2c2b: Responses Tool Search And Namespace Schema Loading

- Goal: port Responses dynamic tool schema parsing: top-level
  `tool_search`, namespace tool groups, tool-search-output `tools` loading,
  combined top-level plus loaded schemas, and namespace/wire-name metadata.
- Oracle: `responses_special_schema_from_tool`,
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
- Validation gate: targeted Rust Responses schema-loading tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.2c2c: Responses Live Tail Validation Surface

- Goal: port model-free Responses live continuation validation outputs:
  missing call-id errors, `requires_live_tool_state`,
  `requires_live_reasoning`, live call-id collection, and visible live suffix
  rendering for trailing tool results.
- Oracle: `responses_validate_tool_outputs`,
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
- Validation gate: targeted Rust Responses live-tail tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.2c3: Anthropic Message And Tool Result Parse Surface

- Goal: port model-free Anthropic request parsing for system/content blocks,
  tools, tool choice, stop sequences, thinking controls, tool-use/tool-result
  messages, and live-tail validation.
- Oracle: `parse_anthropic_request`, `parse_anthropic_system`,
  `parse_anthropic_system_object`, `parse_anthropic_content`,
  `parse_anthropic_messages`, `anthropic_validate_tool_results`, and
  `anthropic_prepare_live_continuation`.
- Fixture: unit vectors for string and block system prompts, text content
  arrays, tool_use/tool_result blocks, private system filtering,
  `tool_choice.type`, `stop_sequences`, `output_config.effort`, bare
  `reasoning_effort`, missing live tool state, and live tool-result suffix
  rendering.
- Comparator: Rust unit tests comparing normalized request fields, rendered
  prompt/live-tail bytes, stop lists, thinking mode, tool schemas, validation
  categories, and Anthropic live-state requirement flags.
- Acceptance: Rust matches current C Anthropic request semantics without
  emitting Anthropic protocol events, sockets, or model loading. Tests use
  no-op server state for live validation; actual KV/tool-memory replay side
  effects remain assigned to M9.8.
- Drift policy: exact for prompt/live-tail bytes and semantic fields;
  stable-category comparison for live-state validation errors.
- Review gate: ask Claude to review Anthropic block parsing, private system
  filtering, tool-result ID validation, and live-tail construction.
- Validation gate: targeted Rust Anthropic parser tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.3: Rust HTTP Skeleton And Model Metadata Endpoints

- Goal: add a Rust server binary with HTTP framing, request routing, CORS
  behavior, `/v1/models`, and no-generation error paths.
- Oracle: current `ds4-server` socket behavior, `models.json` from M0.4,
  CORS/preflight unit tests, malformed HTTP/body handling, and server CLI
  startup flags.
- Fixture: M0.4 `models.json` plus focused local HTTP fixtures for OPTIONS,
  disabled CORS, bad routes, bad methods, bad JSON, missing model, and context
  limit errors.
- Comparator: local HTTP replay comparing status lines, headers, JSON bodies,
  CORS headers, and deterministic model metadata without requiring a model load.
- Acceptance: the Rust server can start, answer `/v1/models`, reject unsupported
  requests with the same protocol shape, and pass local no-model HTTP replay.
- Drift policy: exact status/header/body fields except volatile date-like
  headers, which are ignored if introduced.
- Review gate: ask Claude to review socket lifetime, request framing,
  route/error coverage, and CORS header parity.
- Validation gate: local HTTP comparator with negative tests,
  `cargo test --workspace`, `python3 ds4-parity/run_parity_report.py`, and
  `git diff --check`.

#### M9.4: Non-Streaming Chat Completion Runtime

- Goal: implement model-backed Rust `/v1/chat/completions` non-streaming
  generation for the M0.4 non-streaming OpenAI cases.
- Oracle: M0.4 `chat_basic`, `chat_thinking_disabled`, `chat_cache_seed`, and
  `chat_cache_continuation` current-C responses/traces.
- Fixture: M0.4 request JSON, response JSON, headers, and trace segments for
  non-streaming chat without tool-call output.
- Comparator: B300 request replay against the Rust server comparing normalized
  response JSON, usage fields, finish reasons, generated bytes, headers, and
  trace-rendered prompt/cache fields.
- Acceptance: Rust non-streaming responses match the current C oracle for the
  covered deterministic prompts and do not regress existing M8 CLI runtime
  comparators.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates only;
  generated text, finish reason, usage counts, and cache fields are exact.
- Review gate: ask Claude to review server/session ownership, request-to-runtime
  option mapping, response JSON shape, usage accounting, and trace fields.
- Validation gate: B300 non-streaming comparator with negative tests, local
  parser tests, `cargo test --workspace`, and `git diff --check`.

#### M9.5: Streaming Chat Completion SSE Surface

- Goal: implement Rust streaming `/v1/chat/completions` SSE framing and usage
  reporting for the M0.4 stream case.
- Oracle: M0.4 `chat_stream.sse`, stream headers, stream trace, and unit tests
  for UTF-8/stop-list streaming holds.
- Fixture: M0.4 `chat_stream.json`, `chat_stream.sse`, headers, trace segment,
  and unit vectors for partial UTF-8 and stop-text handling.
- Comparator: B300 SSE replay comparing event order, delta payloads, finish
  chunk, usage chunk, headers, generated bytes, and normalized timing/progress.
- Acceptance: Rust emits the same SSE event sequence and usage semantics for the
  deterministic stream fixture and preserves model-visible text boundaries.
- Drift policy: normalize IDs, timestamps, token rates, and chunk timing only;
  event names/order and text deltas are exact.
- Review gate: ask Claude to review SSE flush behavior, partial UTF-8 handling,
  stop-list trimming, client disconnect paths, and usage chunk parity.
- Validation gate: B300 streaming comparator with negative tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.6: Tool-Call And DSML Server Surface

- Goal: port OpenAI tool schema parsing, DSML prompt rendering, tool-call
  extraction, and tool-call response JSON for the server path.
- Oracle: M0.4 `chat_tool_call`, `test_tool_call_quality`, DSML/tool parser
  unit tests, tool schema order tests, and tool-call streaming unit tests.
- Fixture: M0.4 tool-call request/response/trace plus unit vectors for OpenAI
  tools, DSML argument ordering, loose nested parameters, partial tool-call
  streaming, multiple calls, and raw argument/entity holds.
- Comparator: model-free parser/dump tests for tool schemas and DSML plus B300
  replay for deterministic tool-call generation and response shape.
- Acceptance: Rust produces the same tool prompt, detects the same executable
  tool-call boundaries, emits matching tool-call JSON, and preserves argument
  ordering/canonicalization.
- Drift policy: normalize generated call IDs where random; tool names,
  arguments, finish reasons, prompt text, and trace tool records are exact.
- Review gate: ask Claude to review DSML state-machine parity, schema ordering,
  random-ID normalization, and malformed/recoverable tool parse paths.
- Validation gate: model-free tool parser tests, B300 tool-call comparator with
  negative tests, `cargo test --workspace`, and `git diff --check`.

#### M9.7: Responses And Anthropic Protocol Surface

- Goal: port the Responses and Anthropic request/response/stream protocol
  surfaces that share the server request and tool-memory core.
- Oracle: `parse_responses_request`, `parse_anthropic_request`, Responses and
  Anthropic live-tail/tool-output validation tests, streaming event builders,
  and usage reporting tests in `ds4_server.c`.
- Fixture: server unit vectors for Responses namespace/tool_search schemas,
  Responses reasoning/tool outputs, Anthropic content blocks/system filtering,
  Anthropic tool use/results, live stream deltas, and cache usage fields.
- Comparator: model-free protocol dump/tests comparing normalized request
  semantics, response/event JSON, usage fields, live-tail requirements, and
  tool-output validation categories.
- Acceptance: Rust matches current C protocol semantics for Responses and
  Anthropic without needing the HTTP runtime to be model-backed.
- Drift policy: normalize random IDs and timestamps; event names/order, usage
  fields, validation categories, and rendered prompt/live-tail text are exact.
- Review gate: ask Claude to review protocol-specific live state, reasoning
  replay requirements, namespace tool schema restoration, and Anthropic
  tool-result ID validation.
- Validation gate: protocol unit/comparator tests, `cargo test --workspace`,
  and `git diff --check`.

#### M9.8: Server Cache, KV Restore, And Tool Memory

- Goal: port server cache decisions, disk-KV restore, continued-frontier logic,
  eviction policy, and tool-memory replay.
- Oracle: M0.4 cache continuation, M0.5 KV replay artifacts, KV/cache unit
  tests, tool-memory replay tests, and `compare_server_kv.py`.
- Fixture: M0.4 cache seed/continuation artifacts, M0.5 seed miss/restore and
  continuation restore artifacts, KV headers, rendered text, cache decision
  logs, and tool-memory unit vectors.
- Comparator: B300 replay comparing normalized responses, headers, trace cache
  decisions, KV metadata, rendered text, disk/text cache source, cached token
  counts, and eviction decisions.
- Acceptance: Rust server cache behavior matches M0.4/M0.5 current-C artifacts
  and preserves tool-memory replay semantics for future tool-result requests.
- Drift policy: normalize paths, timestamps, raw KV hashes, and random IDs only;
  cache source, token counts, rendered text, KV headers, and eviction outcomes
  are exact.
- Review gate: ask Claude to review cache-key construction, live vs disk tool
  memory, boundary trimming, KV budget checks, and eviction protection.
- Validation gate: B300 KV/server comparator with negative tests,
  `python3 ds4-parity/compare_server_kv.py`, `cargo test --workspace`, and
  `git diff --check`.

#### M9.9: Server Parity Report Integration

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
- Validation gate: server report, unified report, `./ds4_test --server` through
  the Rust path or documented B300 skip, `cargo test --workspace`, and
  `git diff --check`.

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
