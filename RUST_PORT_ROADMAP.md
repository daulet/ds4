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
