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
- Validation gate: roadmap/board diff and `git diff --check`.

#### M9.2c3a: Anthropic Core Message And Control Parse Surface

- Goal: port model-free Anthropic core request parsing for `messages`,
  `system`, string/text content blocks, private system filtering, scalar
  generation controls, stop sequences, stream flag, `thinking`,
  `output_config.effort`, bare `reasoning_effort`, model alias fallbacks, and
  prompt rendering without tools.
- Oracle: `parse_anthropic_request`, `parse_anthropic_messages`,
  `parse_anthropic_content`, `parse_anthropic_content_block` text branches,
  `parse_anthropic_system`, `parse_anthropic_system_object`,
  `parse_output_config_effort`, `parse_thinking_control_value`, `parse_stop`,
  and model alias thinking logic.
- Fixture: unit vectors for missing messages, string and block system prompts,
  private system blocks, string content, text content arrays, scalar controls,
  stop sequences, stream flag, thinking enabled/disabled, `output_config.effort`,
  bare `reasoning_effort`, and prompt bytes.
- Comparator: Rust unit tests comparing normalized request fields, rendered
  prompt bytes, stop lists, thinking mode, generation controls, stream flag,
  and stable error categories.
- Acceptance: Rust matches current C Anthropic core request semantics without
  tool schemas, tool history, live state, sockets, or model loading.
- Drift policy: exact for semantic fields and prompt bytes; stable-category
  comparison for missing/invalid request errors.
- Review gate: ask Claude to review system block filtering, text-content
  parsing, stop/thinking controls, effort precedence, and prompt bytes.
- Validation gate: targeted Rust Anthropic core parser tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.2c3b: Anthropic Tool Schema And Tool History Parse Surface

- Goal: port model-free Anthropic tool schemas, `tool_choice.type`,
  assistant `tool_use` content blocks, user `tool_result` blocks, tool-use IDs,
  tool result prompt rendering, and DSML request-history rendering.
- Oracle: `parse_anthropic_request` tool/tool_choice branches,
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
- Validation gate: targeted Rust Anthropic tool parser tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.2c3c: Anthropic Live Tool Result Validation Surface

- Goal: port model-free Anthropic live continuation validation outputs:
  missing `tool_use_id` errors, live-state requirement flags, live tool-use ID
  collection, and visible live suffix rendering for trailing tool results.
- Oracle: `anthropic_validate_tool_results`,
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
- Validation gate: targeted Rust Anthropic live-tail tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.3: Rust HTTP Skeleton And Model Metadata Endpoints

- Status: split into M9.3a, M9.3b, and M9.3c before implementation.
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

#### M9.3a: HTTP Framing And CORS Response Surface

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
- Validation gate: targeted Rust HTTP helper tests, `cargo test --workspace`,
  and `git diff --check`.

#### M9.3b: Model Metadata And Route Dispatch Surface

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
- Validation gate: targeted Rust route tests, `cargo test --workspace`, and
  `git diff --check`.

#### M9.3c: No-Model Server Binary And Negative HTTP Replay

- Status: split into M9.3c1 and M9.3c2 before implementation.
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
- Validation gate: `git diff --check`.

#### M9.3c1: No-Model Generation Error Dispatcher

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
- Validation gate: targeted Rust dispatcher tests, `cargo test --workspace`,
  and `git diff --check`.

#### M9.3c2: No-Model Server Binary And Socket Replay

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
- Validation gate: local no-model HTTP comparator, targeted Rust server tests,
  `cargo test --workspace`, and `git diff --check`.

#### M9.4: Non-Streaming Chat Completion Runtime

- Status: split into M9.4a, M9.4b, M9.4c, and M9.4d before
  implementation.
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

#### M9.4a: Model-Backed Server Runtime Boundary

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
- Validation gate: local no-model socket replay, B300 model-load smoke replay,
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.

#### M9.4b: OpenAI Non-Streaming Response And Usage Builder

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
- Validation gate: targeted response-builder tests,
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.

#### M9.4c: No-Cache Non-Streaming Chat Generation Replay

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
- Validation gate: B300 no-cache non-streaming comparator, local parser/error
  tests, existing M8 runtime comparator or documented B300 skip,
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.

#### M9.4d: Memory-Token Cache Seed And Continuation Replay

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
- Validation gate: B300 sequential cache replay comparator, local
  response-builder/cache-accounting tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`.

#### M9.5: Streaming Chat Completion SSE Surface

- Status: split into M9.5a and M9.5b before implementation.
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

#### M9.5a: OpenAI Chat SSE Formatter And Header Builder

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
- Validation gate: targeted SSE formatter tests, decode-policy streaming hold
  tests, `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.

#### M9.5b: Model-Backed Streaming Chat Replay

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
  fixture, keeps tools/thinking/stop-list requests outside this path until their
  roadmap items, and leaves true disk-KV/tool-memory behavior to M9.8.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates
  only; SSE event order, content delta boundaries, finish reason, usage fields,
  prompt text, prompt tokens, and cache fields are exact.
- Review gate: ask Claude to review server streaming routing, token chunk
  capture, SSE flush/write behavior, unsupported-route boundaries,
  decode-policy coverage, usage accounting, and B300 comparator normalization.
- Validation gate: B300 streaming comparator for `chat_stream`, targeted
  runtime tests, decode-policy streaming hold tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`.

#### M9.6: Tool-Call And DSML Server Surface

M9.6 spans several already separated oracle surfaces. M5.6 owns generated
DSML parsing/formatting and M9.2b owns model-free OpenAI tool schema prompt
rendering, so the remaining server work is split before implementation.

#### M9.6a: OpenAI Tool-Call Response Formatter

- Goal: add a pure formatter for final OpenAI chat responses whose assistant
  message contains parsed `tool_calls`, including optional reasoning content
  and usage details.
- Oracle: M0.4 `chat_tool_call` response JSON, M5.6 generated-message parser
  unit vectors, and existing M9.4 response/usage formatting tests.
- Fixture: M0.4 `chat_tool_call` response plus model-free vectors for one
  call, multiple calls, explicit IDs, generated IDs, ordered arguments, and
  escaped tool names/arguments.
- Comparator: exact JSON byte comparison with injected ID/timestamp/call IDs
  and existing parser-produced `DsmlJsonCall` values.
- Acceptance: formatter emits C-compatible `tool_calls` arrays, empty assistant
  content, `finish_reason:"tool_calls"`, stable argument strings, and unchanged
  cache usage fields without touching model execution.
- Drift policy: response object field order, tool-call field order, argument
  strings, finish reason, and usage fields are exact; injected outer IDs,
  timestamps, and generated call IDs are the only normalized values.
- Review gate: ask Claude to review response JSON field order, call-ID
  injection, escaping, and separation from runtime DSML parsing.
- Validation gate: targeted response formatter tests, DSML parser tests,
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.

#### M9.6b: Model-Backed Tool-Call Replay

- Goal: route supported non-streaming OpenAI tool requests through the Rust
  server runtime, parse generated DSML with the existing generated-message
  parser, and emit the M9.6a tool-call response shape.
- Oracle: M0.4 `chat_tool_call` request/response/trace, M9.2b prompt-render
  tests, M5.6 parser tests, and B300 model-backed replay.
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
- Validation gate: B300 tool-call comparator for `chat_tool_call`, targeted
  runtime tests, targeted response/parser tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`.

#### M9.6c: Streaming Tool-Call Deltas

M9.6c spans three surfaces: byte-level SSE event builders, a stateful DSML
stream translator, and model-backed runtime replay. Split it before
implementation so each commit has one oracle and comparator.

#### M9.6c1: Tool-Call SSE Event Formatter

- Goal: add pure OpenAI chat SSE helpers for streamed `tool_calls` deltas:
  role chunk, tool-call start delta, argument-fragment deltas, full-call
  fallback delta, finish chunk, optional usage chunk, and `[DONE]`.
- Oracle: C helpers `sse_chat_tool_call_start_delta`,
  `sse_chat_tool_call_args_delta_n`, `append_tool_call_deltas_json`,
  `openai_sse_finish_live`, and M9.5a non-tool SSE formatting behavior.
- Fixture: model-free vectors for one call, multiple calls, explicit IDs,
  generated IDs, escaped names, argument fragments containing JSON escapes,
  full-call fallback deltas, optional usage, and stream headers.
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
- Validation gate: targeted SSE formatter tests, existing response formatter
  tests, `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.

#### M9.6c2: Incremental DSML Tool-Call Stream Translator

- Goal: translate incremental generated DSML bytes into the M9.6c1 OpenAI
  tool-call start/argument delta events while holding incomplete tags,
  parameter close sentinels, DSML entities, and UTF-8 tails.
- Oracle: C `openai_sse_stream_update`, `openai_tool_stream_update`, related
  `tool_param_value_stream_safe_len` and partial-tool unit tests, plus M5.6
  generated-message parser behavior for completed DSML.
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
- Validation gate: targeted stream-translator tests, existing SSE formatter
  tests, DSML parser tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`.

#### M9.6c3: Model-Backed Streaming Tool-Call Replay

- Goal: route supported streaming OpenAI tool chat requests through the Rust
  server runtime, feed per-token bytes through the M9.6c2 translator, and emit
  the M9.6c1 SSE response shape.
- Oracle: B300 model-backed replay of the M0.4 `chat_tool_call` request with
  `"stream":true`, C streaming unit-test event order, M9.6b prompt/trace
  behavior, and M9.5b non-tool streaming replay behavior.
- Fixture: raw M0.4 trace request with streaming enabled, normalized SSE body,
  stream headers, generated DSML, trace fields, usage/cache fields, and B300
  model identity.
- Comparator: B300 replay compares normalized chat ID, timestamp, and
  generated call ID while checking exact tool name, argument fragments,
  finish reason, usage/cache fields, rendered prompt, generated DSML, trace
  records, and `[DONE]` bytes.
- Acceptance: Rust streams deterministic tool-call output through the HTTP
  runtime, preserves M9.6b non-streaming behavior, and keeps thinking/stop-list
  requests outside this path until their own roadmap items.
- Drift policy: normalize IDs, timestamps, startup timing, and token rates
  only; event order, argument fragments, finish reason, usage fields, prompt
  text, generated DSML, and trace tool records are exact.
- Review gate: ask Claude to review runtime routing, write/flush behavior,
  translator integration, unsupported-route boundaries, trace fields, and B300
  comparator normalization.
- Validation gate: B300 streaming tool-call comparator, targeted runtime tests,
  targeted stream-translator/SSE tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`.

#### M9.6d: Tool-Call Quality Parity Hook

- Goal: connect the Rust server/runtime tool-call path to the existing
  tool-call quality regression surface after non-streaming and streaming
  behavior is in place.
- Oracle: `test_tool_call_quality`, current C quality thresholds/logs, M0.4
  tool-call trace fixtures, and the M9.6b/M9.6c response comparators.
- Fixture: quality-run command lines, model identity, seed/sampling controls,
  expected tool-call success categories, and preserved raw outputs for failures.
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
- Validation gate: B300 quality run or documented blocker with exact command,
  targeted tool-call tests, `cargo test --workspace`, and `git diff --check`.

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

#### M9.7a: Responses And Anthropic Final Response Formatters

- Goal: port model-free non-streaming Responses and Anthropic response body and
  HTTP formatting for assistant text, reasoning, tool calls, finish mapping,
  and cache usage fields.
- Oracle: C `responses_final_response`, `anthropic_final_response`,
  `responses_append_function_call_item`, `append_anthropic_content`,
  `append_responses_usage_json`, `append_anthropic_usage_json`, and the
  associated server unit vectors for usage, namespace, `tool_search`, thinking,
  and stop-reason mapping.
- Fixture: unit vectors with assistant text, reasoning summaries, empty
  Anthropic content, reasoning-only Anthropic content, normal function calls,
  namespace-restored Responses calls, Responses `tool_search_call` output,
  plain functions named `tool_search`, finish reasons, and cache read/write
  usage details.
- Comparator: Rust formatter tests compare exact JSON/HTTP bodies after
  injecting deterministic IDs and timestamps for fields that C randomizes.
- Acceptance: Rust exposes final Responses and Anthropic response formatters
  that match C protocol semantics without opening sockets, loading a model, or
  routing through the runtime.
- Drift policy: random IDs and timestamps are injected in tests; item/event
  names, finish/status mappings, usage fields, namespace restoration,
  `tool_search` discrimination, and empty-content behavior are exact.
- Review gate: ask Claude to review the formatter surface against C protocol
  helpers, especially cache usage clamping, Responses namespace restoration,
  `tool_search` discrimination, and Anthropic empty content.
- Validation gate: targeted Rust server-response tests,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

#### M9.7b: Responses And Anthropic Streaming Event Builders

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
- Validation gate: targeted streaming protocol tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.

#### M9.8a: Server Cache/KV/Tool-Memory Work Item Split

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
- Validation gate: `git diff --check`, docs inspection, and non-interactive
  Claude review with no blockers.

#### M9.8b: Tool-Memory Replay Core

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
- Validation gate: targeted tool-memory tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.

#### M9.8c: Live Continuation And Visible-Prefix State

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
- Validation gate: targeted live-continuation tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.

#### M9.8d: Disk-KV Policy Completion

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
- Validation gate: targeted KV policy tests,
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test`,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

#### M9.8e: KV Tool-Map Trailer Restore

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
- Validation gate: targeted KV tool-map tests, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.

#### M9.8f1: Runtime Cache/KV Integration Split

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
- Validation gate: `git diff --check` and non-interactive Claude review with
  no blockers.

#### M9.8f2: Runtime Cache Configuration And Trace Contract

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
- Validation gate: targeted runtime cache-surface tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

#### M9.8f3: Runtime Disk-KV Lookup And Payload Restore

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
- Validation gate: targeted disk-restore tests, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, B300 restore smoke if local
  model is unavailable, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and non-interactive Claude review with no blockers.

#### M9.8f4: Runtime KV Store, Continued Frontier, And Eviction

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
- Validation gate: targeted store/evict tests, `python3
  ds4-parity/compare_kv_policy.py --negative-test`, `python3
  ds4-parity/compare_kvc_file.py --negative-test`, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.

#### M9.8f5: Runtime Cache/KV Replay Comparator Closure

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
- Validation gate: B300 KV/server comparator with negative tests, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, `python3
  ds4-parity/compare_server_kv.py`, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.

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

### Milestone 10 Work Item Adjustment

#### M10.1: Runtime Graph Work Item Breakdown

- Goal: split Milestone 10 runtime graph orchestration parity into comparable
  implementation and oracle-capture work items before moving graph ownership.
- Oracle: current C `ds4_gpu_graph` allocation/execution paths in `ds4.c`,
  backend primitives in `ds4_gpu.h`, existing graph diagnostics, and the broad
  Milestone 10 acceptance gates above.
- Fixture: roadmap/TODO state plus source evidence for graph allocation,
  decode, prefill, compressed KV, session payload, MTP, and benchmark closure.
- Comparator: documentation-only review that each M10.2+ item has an explicit
  oracle, fixture, comparator, acceptance rule, drift policy, and validation
  gate.
- Acceptance: no M10 implementation item remains catch-all; graph shape,
  backend coverage, tensor oracle capture, decode, prefill, KV/session state,
  MTP, and end-to-end closure can be reviewed and validated independently.
- Drift policy: docs/state-only; no runtime behavior changes.
- Review gate: ask Claude to review whether the split is comparable and avoids
  unmeasurable graph-port steps.
- Validation gate: roadmap/TODO diff inspection, `git diff --check`, and
  non-interactive Claude review with no blockers.

#### M10.2: Backend Operation Inventory And Graph Plan Oracle

- Goal: capture a current-C oracle for the graph tensor plan, backend primitive
  surface, and command-buffer boundaries that Rust must preserve.
- Oracle: `ds4_gpu.h`, C call sites under `metal_graph_alloc_raw_cap`,
  `metal_graph_encode_layer_attention_batch`,
  `metal_graph_encode_layer_ffn_batch`, `metal_graph_eval_token_raw_swa`,
  `metal_graph_prefill_chunked_range`, and MTP verifier helpers.
- Fixture: graph plans for at least short, 2k, and 32k context settings; MTP
  disabled/enabled where model files are available; `ds4_gpu.h` operation
  inventory grouped by tensor allocation, model mapping, command buffers,
  embeddings, attention, compressor, MoE, HC/output, and routing.
- Comparator: machine-readable checker comparing the captured graph plan and
  operation inventory against source-derived expectations and failing on
  missing backend primitives or unassigned graph tensors.
- Acceptance: every `ds4_gpu.h` primitive used by the graph has a named Rust
  trait/facade target, every persistent/work tensor family has an owner group,
  and command-buffer begin/end/synchronize boundaries are recorded before Rust
  scheduling starts.
- Drift policy: operation names, tensor families, context caps, compression
  ratios, command boundaries, and MTP enablement are exact; pointer addresses,
  timings, and allocation addresses are ignored.
- Review gate: ask Claude to review inventory completeness against
  `ds4_gpu.h` and graph call sites.
- Validation gate: oracle checker with negative fixture, `git diff --check`,
  and non-interactive Claude review with no blockers.

#### M10.3: Rust Backend Trait And Graph Plan Surface

- Goal: add Rust graph-plan data structures and backend trait/facade coverage
  for the full M10.2 operation inventory without executing a model graph yet.
- Oracle: M10.2 graph plan and backend operation inventory.
- Fixture: same context/MTP plan cases as M10.2 plus synthetic missing-op and
  tensor-size mismatch cases.
- Comparator: Rust tests and/or a parity script comparing serialized Rust graph
  plans, tensor families, capacities, and trait coverage against the C oracle.
- Acceptance: Rust can name and size the graph state C allocates, exposes a
  backend facade for every required primitive, and fails closed when the C
  inventory gains an unassigned primitive.
- Drift policy: tensor names, capacities, compression caps, raw-window caps,
  and operation names are exact; backend implementation remains FFI-backed.
- Review gate: ask Claude to review trait completeness and whether the facade
  hides backend-specific semantics.
- Validation gate: targeted Rust graph-plan tests, inventory comparator,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

#### M10.4: Current-C Intermediate Tensor Checkpoint Oracle

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
  for the next Rust decode/prefill stages, and nondeterministic long-context
  rows are either excluded or explicitly marked hash-only/skip with evidence.
- Drift policy: tensor shape, dtype, row index, layer, stage, and cache counters
  are exact; f32 values use per-stage tolerances; timings and absolute paths are
  normalized.
- Review gate: ask Claude to review checkpoint coverage and nondeterminism
  policy.
- Validation gate: B300 checkpoint capture/checker, negative mutation test,
  `git diff --check`, and non-interactive Claude review with no blockers.

#### M10.5a: Rust GPU Sys ABI Surface For Graph Primitives

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
- Validation gate: ABI comparator with negative test, Python syntax check,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

#### M10.5b: Rust Decode Call-Order And State Plan

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
  counters, and command boundaries are exact; tensor values remain out of scope.
- Review gate: ask Claude to review decode ordering and cache-state modeling.
- Validation gate: targeted Rust tests, plan comparator with negative test,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

#### M10.5c1: Rust Structured Decode Weight Table

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
- Validation gate: targeted Rust tests, weight-table comparator with negative
  test, unified comparator-only parity report, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.

#### M10.5c2: Rust Decode Graph Tensor State

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
- Validation gate: targeted Rust tests, tensor-state comparator with negative
  test, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and non-interactive Claude review with no blockers.

#### M10.5c3: Rust Decode Backend Facade

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
- Validation gate: facade comparator with negative test, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

#### M10.5c4: Rust Single-Token Decode Graph Execution

- Goal: move one-token decode scheduling for the target model into Rust while
  calling the existing backend primitives through the M10.5c3 facade.
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
- Validation gate: targeted decode comparator on B300, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

Split implementation into reviewable, comparable slices before executing GPU
kernels:

##### M10.5c4a: Rust Decode Execution Trace Oracle

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
- Validation gate: trace comparator with negative test, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4b: Rust Decode Runtime State Bridge

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
- Validation gate: runtime-state comparator with negative test, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4c1: Rust CUDA Backend Linkage And B300 ABI Smoke

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
- Validation gate: static smoke-contract comparator with negative test, B300
  `cargo test -p ds4-gpu --features cuda-backend --test backend_abi`, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4c2a: Rust Decode Model-Map Backend Bridge

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
- Validation gate: model-map bridge comparator with negative test, B300 `cargo
  test -p ds4-gpu --features cuda-backend --test model_map_abi`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4c2b1: Rust Decode Execution Preflight

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
- Validation gate: preflight comparator with negative test, B300 preflight
  binary plus candidate JSON validation, `cargo test -p ds4-gpu
  decode_execution --lib`, `cargo check -p ds4-gpu --bin
  ds4-decode-exec-preflight`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.

##### M10.5c4c2b2a: Rust Full Decode State Allocation

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
- Validation gate: state-allocation comparator with negative test, B300
  allocation binary plus candidate JSON validation, `cargo check -p ds4-gpu
  --bin ds4-decode-state-alloc`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.

##### M10.5c4c2b2b1: Rust First Decode Kernel Execution

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
- Validation gate: first-kernel comparator with negative test, B300
  first-kernel binary plus candidate JSON validation, `cargo check -p ds4-gpu
  --bin ds4-decode-first-kernel`, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.

##### M10.5c4c2b2b2a: Rust First-Kernel Current-C Oracle Comparator

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
- Validation gate: oracle comparator with negative test, B300 current-C oracle
  plus Rust candidate paired validation, `make ds4-first-kernel-oracle-dump`,
  `cargo check -p ds4-gpu --bin ds4-decode-first-kernel`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b1: Rust Layer-0 Attention HC-Pre B300 Execution

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
- Validation gate: layer-0 HC-pre comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-attn-hc-pre-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-attn-hc-pre`, c2b1 first-kernel rerun, c2b2b2a
  current-C oracle rerun, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.

##### M10.5c4c2b2b2b2: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2a and M10.5c4c2b2b2b2b so the next commit has a
  tensor-level execution oracle before cache mutation and attention scheduling
  enter the same diff.

##### M10.5c4c2b2b2b2a: Rust Layer-0 QKV RoPE B300 Execution

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
- Validation gate: QKV/RoPE comparator with negative test, B300 current-C
  oracle plus Rust candidate paired validation, `make
  ds4-layer0-qkv-rope-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-qkv-rope`, c2b2b2b1 layer-0 HC-pre rerun, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b1 and M10.5c4c2b2b2b2b2 so cache mutation and
  layer-0 attention output get an exact tensor oracle before the remaining
  all-layer scheduler and logits path are introduced.

##### M10.5c4c2b2b2b2b1: Rust Layer-0 Attention Output B300 Execution

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
- Validation gate: layer-0 attention-output comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-attn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-attn-output`, c2b2b2b2a layer-0 QKV/RoPE rerun, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2a and M10.5c4c2b2b2b2b2b so the layer-0 FFN
  body gets an exact tensor oracle before the remaining all-layer scheduler,
  cache-compression transitions, and logits path are introduced.

##### M10.5c4c2b2b2b2b2a: Rust Layer-0 FFN Output B300 Execution

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
- Validation gate: layer-0 FFN-output comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-ffn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-ffn-output`, c2b2b2b2b1 layer-0 attention-output rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2b: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b1 and M10.5c4c2b2b2b2b2b2 so the
  output-head/logits kernels are independently compared before the remaining
  all-layer scheduler, cache-compression transitions, and final decode trace
  are introduced together.

##### M10.5c4c2b2b2b2b2b1: Rust Layer-0 Output Head B300 Execution

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
- Validation gate: layer-0 output-head comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer0-output-head-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-output-head`, c2b2b2b2b2a layer-0 FFN-output rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2b2: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b2a and M10.5c4c2b2b2b2b2b2b so the
  dense layer-loop buffer swap is compared before compressed layer-2 cache
  mutation and all-layer scheduling are introduced.

##### M10.5c4c2b2b2b2b2b2a: Rust Two Dense-Layer Output Head B300 Execution

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
- Validation gate: two-dense-layer output-head comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-two-layer-output-head-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-two-layer-output-head`, c2b2b2b2b2b1 layer-0 output-head rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2b2b: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b2b1 and M10.5c4c2b2b2b2b2b2b2 so the
  first ratio-4 compressed/indexer state mutation is compared before compressed
  attention, all-layer scheduling, and final logits are introduced together.

##### M10.5c4c2b2b2b2b2b2b1: Rust Layer-2 Ratio-4 Compressor State B300 Execution

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
- Validation gate: layer-2 compressor-state comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer2-compressor-state-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-compressor-state`, c2b2b2b2b2b2a two-layer output-head
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2b2b2: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b2b2a and
  M10.5c4c2b2b2b2b2b2b2b so the layer-2 attention-output boundary is compared
  after the first compressor-state mutation, before the remaining FFN,
  all-layer scheduler, and final logits are introduced together.

##### M10.5c4c2b2b2b2b2b2b2a: Rust Layer-2 Attention Output B300 Execution

- Goal: execute dense layers `0` and `1`, then execute layer `2` through
  raw-only attention decode, inverse compressed RoPE, low-rank attention output
  projection, and HC expansion on B300. This slice extends the validated
  layer-2 compressor-state mutation without yet taking the layer-2 FFN,
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
- Validation gate: layer-2 attention-output comparator with negative test,
  B300 current-C oracle plus Rust candidate paired validation, `make
  ds4-layer2-attn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-attn-output`, c2b2b2b2b2b2b1 layer-2 compressor-state
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2b2b2b: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b2b2b1 and
  M10.5c4c2b2b2b2b2b2b2b2 so the layer-2 FFN-output boundary is compared
  after the layer-2 raw-only attention-output boundary, before the remaining
  layers, ratio-128 coverage, output head, and final logits are introduced
  together.

##### M10.5c4c2b2b2b2b2b2b2b1: Rust Layer-2 FFN Output B300 Execution

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
- Validation gate: layer-2 FFN-output comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer2-ffn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-ffn-output`, c2b2b2b2b2b2b2a layer-2 attention-output
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.

##### M10.5c4c2b2b2b2b2b2b2b2: Rust One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b2b2b2a and
  M10.5c4c2b2b2b2b2b2b2b2b so the first ratio-128 compressed layer is
  compared after the layer-2 ratio-4 FFN-output boundary, before the repeated
  remaining layers, output head, and final logits are introduced.

##### M10.5c4c2b2b2b2b2b2b2b2a: Rust Layer-3 Ratio-128 FFN Output B300 Execution

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
- Validation gate: layer-3 FFN-output comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-layer3-ffn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer3-ffn-output`, c2b2b2b2b2b2b2b1 layer-2 FFN-output rerun,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Validation result: completed on 2026-05-23 with local comparator
  self-check, negative test, paired artifact comparator with 1,261 checks,
  B300 current-C oracle plus Rust candidate paired validation, B300
  c2b2b2b2b2b2b2b1 layer-2 FFN-output rerun with 1,383 checks, local
  `arch -arm64 make ds4-layer3-ffn-output-oracle-dump`, `cargo check -p
  ds4-gpu --bin ds4-decode-layer3-ffn-output`, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive
  Claude review with `NO BLOCKERS`.

##### M10.5c4c2b2b2b2b2b2b2b2b: Rust Remaining One-Token Decode B300 Execution

- Split into M10.5c4c2b2b2b2b2b2b2b2b1 and
  M10.5c4c2b2b2b2b2b2b2b2b2 so the first post-ratio128 ratio-4/indexer
  layer is compared before the remaining repeated layers, output head, and
  final logits are introduced.

##### M10.5c4c2b2b2b2b2b2b2b2b1: Rust Layer-4 Post-Ratio128 Ratio-4 FFN Output B300 Execution

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
- Validation gate: layer-4 FFN-output comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation,
  `make ds4-layer4-ffn-output-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-layer4-ffn-output`, c2b2b2b2b2b2b2b2a layer-3 ratio-128
  FFN-output rerun, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and non-interactive Claude review with no blockers.
- Evidence: the B300 paired layer-4 post-ratio128 ratio-4/indexer
  FFN-output validator passed 1,383 pinned checks after recording
  `after_layer3_hc=734775286457caef`,
  `layer4_attn_state_kv=154dccb1209e67d0`,
  `layer4_index_state_kv=87580106eb9c5c3d`,
  `layer4_router_selected=ec6043f1523b2257`, and
  `layer4_after_ffn_hc=b19322ec84d84935`; the predecessor layer-3
  ratio-128 FFN-output B300 rerun still passed 1,261 checks.

##### M10.5c4c2b2b2b2b2b2b2b2b2: Rust Remaining Layer Loop And Logits B300 Execution

- Split into M10.5c4c2b2b2b2b2b2b2b2b2a and
  M10.5c4c2b2b2b2b2b2b2b2b2b so the repeated all-layer decode loop reaches a
  comparable final HC boundary before the output-head and final logits are
  attached.

##### M10.5c4c2b2b2b2b2b2b2b2b2a: Rust All-Layer Final HC B300 Execution

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
- Validation evidence: all-layer final-HC comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, `make
  ds4-all-layer-final-hc-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-all-layer-final-hc`, c2b2b2b2b2b2b2b2b1 layer-4 FFN-output
  rerun, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, local unified parity report with B300 rerun command coverage, and
  730 pinned B300 checks with final-HC digest
  `after_layer42_hc=cbd17b425564f63f`.

##### M10.5c4c2b2b2b2b2b2b2b2b2b: Rust Full One-Token Output Head And Logits B300 Execution

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
- Validation gate: targeted decode comparator on B300, c2b1 preflight rerun,
  c2b2a state allocation rerun, c2b2b1 first-kernel rerun, c2b2b2a current-C
  oracle rerun, c2b2b2b1 layer-0 HC-pre rerun, c2b2b2b2a layer-0 QKV/RoPE
  rerun, c2b2b2b2b1 layer-0 attention-output rerun, c2b2b2b2b2a layer-0
  FFN-output rerun, c2b2b2b2b2b1 layer-0 output-head rerun,
  c2b2b2b2b2b2a two-dense-layer output-head rerun,
  c2b2b2b2b2b2b1 layer-2 compressor-state rerun,
  c2b2b2b2b2b2b2a layer-2 attention-output rerun,
  c2b2b2b2b2b2b2b1 layer-2 FFN-output rerun,
  c2b2b2b2b2b2b2b2a layer-3 ratio-128 FFN-output rerun,
  c2b2b2b2b2b2b2b2b1 layer-4 post-ratio128 ratio-4 FFN-output rerun, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Validation evidence: full output-head comparator with negative test, B300
  current-C oracle plus Rust candidate paired validation, c2b2b2b2b2b2a
  two-layer output-head B300 predecessor rerun, c2b2b2b2b2b2b2b2b2a
  all-layer final-HC B300 predecessor rerun, `make
  ds4-full-output-head-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-decode-full-output-head`, local unified parity report with B300 rerun
  command coverage, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, touched-file NUL scan, non-interactive Claude review
  with `NO BLOCKERS`, and 440 pinned B300 checks with logits digest
  `logits=432eef0524ced3ad`.

##### M10.5c4d1: Rust Short Decode-Continuation Output-Head B300 Execution

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
- Validation evidence: short continuation comparator with negative test, paired
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
- Evidence: B300 short continuation paired validation matched `sequence_len=22`,
  `final_position=21`, `raw_row=21`, `raw_start=0`, `n_raw=22`,
  `layer2_n_comp=5`, `layer2_n_index_comp=5`, `layer5_n_comp=0`,
  `layer42_n_comp=5`, and `layer42_n_index_comp=5`; full-buffer FNV digests
  include `after_layer42_hc=40e22a11d8ca9178`,
  `logits=fcc73408cecb8073`,
  `layer2_attn_comp_row4=061fb5b8eabae3db`,
  `layer42_attn_comp_row4=24844d05b88a2c04`, and
  `layer42_index_state_kv=06ac626b7530144e`.

##### M10.5c4d2: Rust Ratio-Boundary Continuation Coverage

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
- Validation evidence: ratio-boundary comparator with negative test, paired
  local artifact validation with 865 pinned checks, B300 current-C oracle plus
  Rust CUDA candidate validation with 829 checks, local `arch -arm64 make
  ds4-ratio-boundary-output-head-oracle-dump`, local `cargo check -p ds4-gpu
  --bin ds4-decode-ratio-boundary-output-head`, local unified report with B300
  rerun command coverage, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, touched-file NUL scan, non-interactive Claude
  review with `NO BLOCKERS`, and pinned artifact SHA256
  `oracle=b8e813b11312f931a4bb786d297661d933588bba78da7bad78a147653c2c58c7`
  and
  `rust=72fbe9424cecf96d6710d0a1adc43563a5d7af0360ec804628e9224bec081449`.
- Evidence: B300 ratio-boundary paired validation matched `sequence_len=128`,
  `final_position=127`, `raw_row=127`, `raw_start=0`, `n_raw=128`,
  `emit_compressed_row=1`, `layer2_n_comp=32`,
  `layer2_n_index_comp=32`, `layer5_n_comp=1`, `layer42_n_comp=32`, and
  `layer42_n_index_comp=32`; full-buffer FNV digests include
  `after_layer42_hc=12f1089ad3297673`,
  `logits=c67eab1a566286ae`,
  `layer2_attn_comp_row31=72353245d1b57607`,
  `layer5_attn_comp_row0=e65ab25c4927545f`, and
  `layer42_index_state_kv=1e0df1e98d453bcd`.

##### M10.5c4d3: Rust Long Indexed-Continuation Attention Coverage

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
- Validation gate: indexed-continuation comparator on B300, targeted Rust
  checks, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Validation evidence: static and negative long indexed-attention comparator
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
- Evidence: the fixture warms deterministic tokens `0..2050` through
  production current-C decode and stops token `2051` after layer `2`, matching
  `sequence_len=2052`, `raw_row=2051`, `raw_start=1924`, `n_raw=128`,
  `layer2_n_comp=513`, `layer2_n_index_comp=513`,
  `layer2_comp_selected=96be5e90e07d5fe3`,
  `layer2_heads=152cefad5f4521d0`,
  `layer2_attn_out=d31399afb15f9523`,
  `layer2_after_attn_hc=ce72c471b910e3e4`,
  `layer2_attn_comp_row512=25b13ef81b3cc643`, and
  `layer2_index_comp_row512=8bf040cdf84597fb`. The CUDA single-token
  indexed-attention fallback now fills selected compressed rows
  deterministically in top-k order.

##### M10.5c4d4: Rust Directional-Steering Decode Coverage

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
- Validation gate: steering comparator on B300 or exact skip, targeted Rust
  checks, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Evidence: implemented with B300 `ds4flash.gguf` and
  `dir-steering/out/verbosity.f32`; pinned comparator passed 469 C-vs-Rust
  checks and rejected 13 negative-test mutations.

#### M10.6: Rust Layer-Major Prefill And Chunking

M10.6 is split because whole prefill, cold chunked prefill, and resumed suffix
prefill each have different C routing and failure boundaries.

##### M10.6a: Rust Prefill Scheduling Plan

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
- Evidence: implemented with six Rust scheduling oracle cases covering whole,
  chunked, resumed suffix, short decode fallback, and cache-hit routes;
  comparator passed six cases, six chunks, six progress points, candidate JSON
  validation, and 10 rejected negative-test mutations.

##### M10.6b: Rust Whole-Prefill Short Execution

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
- Validation gate: targeted B300 comparator, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Evidence: implemented a current-C whole-prefill oracle helper and Rust
  `ds4-prefill-whole-short` candidate over the short Italian prompt; B300
  current-C vs Rust comparator passed 780 pinned checks over final logits,
  output row, layer-2/layer-42 raw and compressed rows, layer-5/layer-42
  compressor states, prompt token count, and cache counters.

##### M10.6c: Rust Cold Chunked-Prefill Execution

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
- Validation gate: targeted B300 comparator, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Evidence: implemented cold chunked prefill in the Rust
  `ds4-prefill-whole-short` candidate and same-run current-C oracle coverage
  for 2052-token and full long prompt fixtures. B300 current-C vs Rust
  comparator passed 400 checks for each fixture with deterministic CUDA MoE
  down projection enabled, covering chunk schedule, output rows, raw-ring rows,
  compressed/index counters, output digests, and sampled logits.

##### M10.6d: Rust Resumed-Suffix Prefill Execution

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
- Validation gate: targeted B300 comparator, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Evidence: implemented resumed-prefix execution in the Rust
  `ds4-prefill-whole-short` candidate and same-run current-C oracle coverage
  for cache-hit, short decode-suffix, and resumed-chunked fixtures. B300
  current-C vs Rust comparator passed 425 checks for each fixture with
  deterministic CUDA MoE down projection enabled, covering route choice, resume
  threshold, decode-token count, resumed chunk starts/sizes, checkpoint length,
  raw-ring rows, compressed/index counters, output digests, and sampled logits.

#### M10.7: Rust Graph Session State And Payload Parity

M10.7 crosses payload byte layout, graph tensor state, disk restore, and
continued-frontier policy, so it is split before implementation. Each subitem
keeps current C as the execution-behavior oracle and advances one save/restore
ownership boundary.

##### M10.7a: Rust Graph Session Payload Layout Plan

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
- Validation gate: comparator and negative tests, C helper build, targeted
  Rust tests, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and non-interactive Claude review with no blockers.
- Evidence: implemented the current-C graph payload layout oracle and Rust
  layout planner/dumper for the five default `ctx=32768` fixtures. The
  comparator passed 901 C-vs-Rust checks across schema/scope, constants, body
  order, graph caps, raw logical/physical row mapping, ratio-4 and ratio-128
  row counts, per-section bytes, sampled layer bytes, and final payload bytes;
  negative tests rejected 7 layout mutations. Full local validation passed the
  unified parity report with 48 passed, 36 skipped, and 0 failed, full
  `cargo test --workspace`, format/diff checks, and non-interactive Claude
  review with `NO BLOCKERS`.

##### M10.7b: Rust Graph Session Payload Reader And Writer

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
- Validation gate: payload reader/writer tests, comparator and negative tests,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Evidence: implemented Rust graph payload runtime validation, reader summaries,
  and writer wrappers plus `--graph-probe` dumpers on both C and Rust sides. The
  comparator passed 375 checks across runtime constants, byte FNVs, payload
  sizes, parsed raw-ring summaries, section byte sums, and rejection codes for
  valid short, valid wrapped, truncated, trailing, invalid count, raw-ring,
  context, layout, chunk-layout, and comp-cap cases; negative tests rejected 7
  mutations. Full local validation passed the unified parity report with 49
  passed, 36 skipped, and 0 failed, full `cargo test --workspace`, format/diff
  checks, and non-interactive Claude review with `NO BLOCKERS`.

##### M10.7c: Rust Disk KV Payload Restore Smoke

M10.7c crosses committed restore metadata, raw B300 payload bytes, and actual
graph tensor restore, so it is split before implementation. Each subitem keeps
the M7.8 restore oracle as the current-C behavior source while advancing one
restore boundary.

###### M10.7c1: Rust Restore Payload Header Contract

- Goal: prove Rust graph payload planning matches the committed M7.8 restore
  oracle headers and payload byte counts before loading raw restore bodies.
- Oracle: `ds4-parity/baselines/kv/m7.8/current-c.json` header prefixes,
  payload/snapshot byte counts, prompt-token counts, and fixture identity.
- Fixture: disk and in-memory restore records for seed and continuation
  prompts on `/workspace/ds4/ds4flash.gguf`.
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
- Validation gate: restore-header comparator and negative tests, targeted Rust
  tests, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Evidence: implemented Rust `--restore-header-plan` output and
  `ds4-parity/compare_restore_payload_header_plan.py` against the committed
  M7.8 current-C restore oracle. The comparator passed 127 checks over model
  identity, case order/count, hash-only policy, exact DSV4 header bytes,
  payload/snapshot byte budgets, graph caps, raw-live rows, and ratio row
  counts; negative tests rejected 7 mutations. Full local validation passed
  the unified parity report with 50 passed, 36 skipped, and 0 failed, full
  `cargo test --workspace`, format/diff checks, and non-interactive Claude
  review with `NO BLOCKERS`.

###### M10.7c2: Rust Disk KV Payload Byte Import Smoke

- Goal: on B300, feed the raw C-written disk KVC restore payload bytes into the
  Rust graph payload reader and prove Rust accepts the bytes with the recorded
  hashes and section plan.
- Oracle: M7.8 disk payload raw files and `payload_sha256` records on B300.
- Fixture: seed and continuation disk restore payload bodies in the M7.8 raw
  artifact location on `/workspace/ds4`.
- Comparator: B300 Rust payload-reader smoke over observed and historical
  payload SHA256 metadata, header fields, payload length, section byte plan,
  compressed/index counts, and rejection of mutated summaries.
- Acceptance: Rust can import the same raw disk payload bytes as C at the
  reader level, without restoring tensors into graph memory yet.
- Drift policy: lengths, header fields, count tables, section offsets, and
  Rust-reader acceptance are exact; raw payload hashes are recorded as
  per-capture metadata because B300 restore bodies can drift while preserving
  restore behavior.
- Review gate: ask Claude to review raw-byte bounds checks and B300 evidence.
- Validation gate: B300 raw payload reader smoke, local comparator negative
  tests, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Evidence: implemented Rust `--graph-file-probe <id:path>` and
  `ds4-parity/compare_graph_payload_raw_import.py`, then ran the comparator on
  the B300 pod against the M7.8 hash-only raw disk payload bodies. The live run
  passed 104 checks over disk case order, payload SHA256 metadata and byte
  counts, Rust reader acceptance, raw-ring positions, section byte totals,
  ratio row counts, and hash-only policy; negative tests rejected 7 mutations.
  M10.7c3a later narrowed raw-body SHA256 to per-capture metadata after B300
  reruns proved restore bodies are byte-unstable even when C self-restore
  passes. The committed
  summary `ds4-parity/baselines/kv/m10.7c2/rust-b300-raw-import.json` contains
  only hashes/FNVs/parsed metadata, not raw payload bodies. Full local
  validation passed the unified parity report with 51 passed, 37 skipped, and
  0 failed, full `cargo test --workspace`, format/diff checks, and
  non-interactive Claude review with `NO BLOCKERS`.

###### M10.7c3: Rust Graph Tensor Restore Next-Token Smoke

- Status: split into M10.7c3a-M10.7c3d before implementation; M10.7c3a,
  M10.7c3b, M10.7c3c, and M10.7c3d done.
- Goal: advance graph restore from raw memory snapshot availability, to restore
  target mapping, to tensor readback, and finally to next-token behavior.
- Oracle: current C M7.8 restore oracle on B300.
- Acceptance: each subitem has a concrete raw-body or restore-state comparator
  before claiming next-token parity.
- Owner path: current-C restore dumper, Rust graph payload reader, Rust graph
  restore runtime, B300 restore comparators, `ds4-parity/`, `.memory/status.md`.

###### M10.7c3a: Rust Memory Snapshot Raw Body Import Smoke

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
  Rust-reader acceptance are exact; snapshot body hashes are recorded as
  per-capture metadata because B300 restore bodies can drift while preserving
  restore behavior.
- Review gate: ask Claude to review snapshot body materialization, raw-byte
  bounds checks, and B300 evidence.
- Validation gate: B300 raw snapshot reader smoke, local comparator negative
  tests, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Evidence: added `ds4-restore-dump --snapshot-dir`, B300
  `ds4-parity/baselines/kv/m10.7c3a/rust-b300-snapshot-raw-import.json`, and
  `ds4-parity/compare_graph_snapshot_raw_import.py`. The B300 raw disk import
  rerun under the corrected per-capture hash policy passed 108 checks and
  rejected 9 mutations; the B300 raw snapshot materialization/import passed 110
  checks and rejected 9 mutations. Local validation passed both raw import
  comparators with 104 checks and 9 negative mutations each, Python syntax
  checks, `cargo test -p ds4-gguf session_payload`, `git diff --check`, `arch
  -arm64 make ds4-restore-dump`, the unified parity report with 52 passed, 38
  skipped, and 0 failed, `cargo fmt --all -- --check`, `cargo test
  --workspace`, and non-interactive Claude review with no blockers.

###### M10.7c3b: Rust Graph Restore Target Mapping Contract

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
- Validation gate: restore-target comparator, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Evidence: added `ds4-session-payload-dump-rs --restore-target-plan` and
  `ds4-parity/compare_graph_restore_target_plan.py`. The comparator covers the
  four M7.8 disk/snapshot cases over checkpoint/logit/count-table targets, raw
  logical-to-physical ring rows, per-layer attention compressed-cache and state
  targets, ratio-4 indexer targets, and post-restore counters
  (`layer_n_comp`, `layer_n_index_comp`, `checkpoint_valid`,
  `mtp_draft_valid`, and `mtp_n_raw`). Validation passed the restore-target
  comparator with 6012 checks and 8 negative mutations, an explicit
  candidate-file comparison with 6012 checks, Python syntax checks, `cargo test
  -p ds4-gguf session_payload`, `cargo fmt --all -- --check`, `git diff
  --check`, the unified parity report with 53 passed, 38 skipped, and 0 failed,
  `cargo test --workspace`, and non-interactive Claude review with no blockers.

###### M10.7c3c: Rust Graph Tensor Restore Readback Smoke

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
- Validation gate: B300 tensor readback smoke, restore-target comparator,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Evidence: added `ds4-graph-restore-readback`,
  `ds4-parity/compare_graph_restore_readback.py`, and
  `ds4-parity/baselines/kv/m10.7c3c/rust-b300-restore-readback.json`. The B300
  live smoke writes the four M7.8 disk/snapshot raw bodies into Rust-owned graph
  tensors and reads back checkpoint tokens, logits, count tables, raw rows,
  attention compressed rows, attention state tensors, ratio-4 indexer rows and
  state tensors, sampled layer sections, and post-restore counters. Validation
  passed the B300 live readback comparator with 1365 checks and 8 negative
  mutations, local readback comparator with 1365 checks and 8 negative
  mutations, Python syntax checks, `cargo test -p ds4-gpu --bin
  ds4-graph-restore-readback`, `cargo check -p ds4-gpu --bin
  ds4-graph-restore-readback`, `cargo fmt --all -- --check`, `git diff
  --check`, the unified parity report with 54 passed, 39 skipped, and 0
  failed, `cargo test --workspace`, and non-interactive Claude review with no
  blockers.

###### M10.7c3d: Rust Graph Tensor Restore Next-Token Smoke

- Goal: restore C-written disk and memory snapshot payloads into Rust graph
  session state on B300 and prove next-token behavior matches current C.
- Oracle: current C M7.8 restore oracle on B300 plus M10.7c3c tensor readback
  evidence.
- Fixture: seed disk payload restore, continuation disk payload restore, and
  in-memory snapshot restore on `/workspace/ds4/ds4flash.gguf`.
- Comparator: B300 Rust-vs-current-C restore comparator over payload hashes,
  checkpoint tokens, selected token, top-logprob order, cache source, and graph
  counters.
- Acceptance: Rust-restored sessions produce the same next-token state as the
  C restore oracle for the committed fixtures.
- Drift policy: payload body hashes, restored checkpoint length, selected
  token, top-logprob order, cache source, and graph counters are exact. Raw
  body SHA256 values are per-capture metadata, so exact top-logprob scores
  compare against the same-capture current-C restore oracle.
- Review gate: ask Claude to review restore invariants and B300 evidence.
- Validation gate: B300 restore smoke, session payload comparator, KV replay
  comparator, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and non-interactive Claude review with no blockers.
- Evidence: added `ds4-graph-restore-next-token`,
  `ds4-parity/compare_graph_restore_next_token.py`, and
  `ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json`. The
  B300 live comparator recaptures current C restore output and raw payload
  bodies, runs same-capture Rust tensor readback, restores the same payloads
  into Rust graph state, and checks restored checkpoint/logits FNVs, selected
  token, top-logprob order and scores, cache source, and post-restore graph
  counters. Validation passed the B300 live comparator with 4030 checks and 11
  negative mutations, local `python3
  ds4-parity/compare_graph_restore_next_token.py --negative-test` with 4030
  checks and 11 negative mutations, Python syntax checks, `cargo check -p
  ds4-gpu --bin ds4-graph-restore-next-token`, `cargo test -p ds4-gpu --bin
  ds4-graph-restore-next-token`, `cargo fmt --all -- --check`, `git diff
  --check`, the unified parity report with 55 passed, 40 skipped, and 0
  failed, `cargo test --workspace`, and non-interactive Claude review with no
  blockers.

##### M10.7d1: Continued-Frontier Policy Transition Matrix

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
- Validation gate: policy comparator with negative tests, targeted Rust tests,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- Evidence: added C/Rust `continued_frontier_transitions` oracle rows for
  note-store growth, lower-note skip, fresh-frontier suppression/restore,
  already-stored suppression skip, unaligned suppression skip,
  mismatch-restore ignore, reset-after-miss, and disk-restore loaded frontier
  state. Refreshed M7.2 and M7.7 fixtures/manifests so replay preconditions
  track the new M7.2 policy artifact. Rust adds `reset_continued_frontier`,
  focused policy tests, and a runtime cache-state reset test. Validation passed
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test` with 521 schema
  checks, 11 manifest checks, and 8 negative checks; `python3
  ds4-parity/compare_kv_policy.py --negative-test` with 1725 comparator checks
  and 9 negative checks; `python3 ds4-parity/compare_kv_replay.py
  --negative-test`; `python3 ds4-parity/run_kv_parity_report.py` with 9
  passed, 1 skipped, and 0 failed; `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed; targeted Rust tests for continued-store policy and runtime reset;
  `cargo test --workspace`; `cargo fmt --all -- --check`; `git diff --check`;
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 55
  passed, 40 skipped, and 0 failed; and non-interactive Claude review with no
  blockers.

##### M10.7d2a: Runtime Continued-Frontier Ledger Contract

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
- Validation gate: targeted Rust runtime tests, KV policy comparator, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Evidence: Rust runtime cache state now records per-request cache decision and
  continued-frontier events for reset, suppression, restore, note, live-prefix
  store, current-store, and continued-store attempts. Validation passed focused
  runtime ledger tests, KV policy comparator, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, skip-local unified parity report,
  and non-interactive Claude review with no blockers.

##### M10.7d2b: Runtime KV Replay Checker Closure

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
- Validation gate: runtime KV replay checker with negative tests, KV replay
  comparator, targeted Rust runtime tests, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and non-interactive Claude review
  with no blockers.
- Evidence: `check_runtime_kv_replay_summary.py` now validates the M9.8f5 B300
  replay summary plus a model-free M10.7d2 ledger contract covering seed miss,
  seed restore, continuation restore, and memory-token continuation event
  order/frontier transitions. `run_server_parity_report.py` runs the checker
  with negative tests, and validation passed the runtime checker, KV replay
  comparator, server parity report, workspace tests, formatting, diff check, and
  skip-local unified parity report, plus non-interactive Claude review with no
  blockers.

##### M10.7d2c: Runtime Continued-Store B300 Replay Refresh

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
- Validation gate: B300 runtime replay, checker negative tests, KV replay
  comparator, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and non-interactive Claude review with no blockers.
- Evidence: `ds4-server-runtime-rs` now writes a stable runtime cache ledger
  section in each trace, the M9.8f5 B300 summary records checked ledger cases
  plus raw trace event counts/names for seed miss, seed restore, and
  continuation restore, and the checker validates the summary and M10.7d2
  contract with six negative mutations. The live B300 M0.5 replay passed after
  using a 20-second startup wait for CUDA model cache initialization, and local
  validation passed checker negative tests, KV replay comparator, server parity
  report, workspace tests, formatting, diff check, and skip-local unified
  parity report, plus non-interactive Claude review with no blockers.

##### M10.7d3: Graph Restore Continued-Frontier B300 Smoke

- Status: split before implementation into M10.7d3a, M10.7d3b, and M10.7d3c
  so the graph-restore frontier work remains comparable at each step.
  M10.7d3c is further split into M10.7d3c1 and M10.7d3c2 so the KVC
  write/skip contract is proven before the B300 file-writing smoke.

##### M10.7d3a: Graph Restore Frontier Contract

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
- Validation gate: contract checker with negative tests, KV policy comparator,
  graph restore next-token comparator, `cargo fmt --all -- --check`, `git diff
  --check`, and non-interactive Claude review with no blockers.
- Evidence: added
  `ds4-parity/baselines/kv/m10.7d3/restore-frontier-contract.json` and
  `ds4-parity/check_graph_restore_frontier_contract.py`. The checker validates
  the contract against M10.7c3d restored-token evidence, M7.2 current-C
  continued-frontier policy, and M0.5 KVC reason/header rows with seven
  negative mutations; it is wired into `run_parity_report.py` and documented in
  `ds4-parity/README.md`. Validation passed the contract checker, KV policy
  comparator, graph restore next-token comparator, formatting, diff check, and
  skip-local unified parity report, plus non-interactive Claude review with no
  blockers.

##### M10.7d3b: B300 Restored-Graph Frontier Projection

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
- Validation gate: B300 live graph restore projection capture, comparator with
  negative tests, targeted Rust tests, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, and non-interactive Claude review with
  no blockers.
- Evidence: `ds4-graph-restore-next-token` now emits a
  `frontier_projection` object for each restored B300 graph payload. The
  refreshed M10.7c3d summary records restored frontiers 550/561, unaligned
  current-live skip target 0, next continued target 10240, and
  already-stored boundary target 0 for the four disk/snapshot restore cases.
  `compare_graph_restore_next_token.py` validates those fields against the
  M10.7d3a contract while preserving same-capture next-token/readback checks;
  the exact-tree B300 live run passed 4177 checks and 12 negative mutations,
  and full local validation plus non-interactive Claude review passed with no
  blockers.

##### M10.7d3c: Post-Restore KVC Write/Skip B300 Smoke

- Status: split before implementation into M10.7d3c1 and M10.7d3c2.
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
  token counts, continued-frontier state, KVC reason fields, and graph counters.
- Acceptance: Rust-restored graph sessions continue with the same cache
  store/skip decisions as current C after restore.
- Drift policy: restored token counts, frontier tokens, reason fields, graph
  counters, and write/skip decisions are exact; paths, timestamps, and raw
  payload hashes are normalized.
- Review gate: ask Claude to review B300 command fidelity and post-restore KVC
  invariants.
- Validation gate: B300 post-restore KVC smoke, runtime KV replay comparator,
  graph restore projection comparator, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.

##### M10.7d3c1: Post-Restore KVC Decision Contract

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
- Validation gate: contract checker with negative tests, graph restore
  projection comparator, runtime KV replay checker, KVC file comparator,
  `cargo fmt --all -- --check`, `git diff --check`, and non-interactive Claude
  review with no blockers.
- Evidence: added
  `ds4-parity/baselines/kv/m10.7d3/post-restore-kvc-decision-contract.json`
  and `ds4-parity/check_post_restore_kvc_decision_contract.py`. The checker
  validates the four restored graph cases against M10.7d3b frontier
  projections, the M10.7d3a frontier contract, M9.8f5 runtime replay
  skip/write evidence, the M10.7d2 runtime ledger contract, and the M7.4a KVC
  file layout oracle. It passed eight negative mutations covering token,
  payload, reason, skip, runtime, projection, and KVC-layout drift.

##### M10.7d3c2: B300 Restored Payload KVC File Smoke

- Goal: run a B300 Rust smoke that wraps restored graph payload bodies in KVC
  files and records the matching post-restore skip decisions.
- Oracle: M10.7d3c1 post-restore KVC decision contract, M10.7d3b
  same-capture restored graph evidence, and current C KVC file/header behavior.
- Fixture: B300 raw graph payload and memory snapshot bodies for
  `disk_seed_payload`, `snapshot_seed`, `disk_continuation_payload`, and
  `snapshot_continuation`, plus rendered cache-text keys derived from the
  same current-C restore capture.
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
- Validation gate: B300 post-restore KVC file smoke, comparator with negative
  tests, targeted Rust tests, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added `ds4-post-restore-kvc-smoke`,
  `ds4-parity/compare_post_restore_kvc_smoke.py`, and
  `ds4-parity/baselines/kv/m10.7d3/rust-b300-post-restore-kvc.json`. The live
  `hou2-prod1` B300 run wrapped the four restored raw graph payload bodies in
  deterministic shutdown KVC files, recorded rendered text key metadata,
  restored frontier decisions, graph counters, KVC headers, file sizes, and
  payload digests, then passed 536 exact checks plus seven negative mutations.

#### M10.8: Rust MTP Draft And Verifier Orchestration

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
- Validation gate: B300 MTP comparator, targeted Rust tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- Status: split before implementation into M10.8a through M10.8g so MTP
  availability, state-machine decisions, verifier orchestration, frontier
  mutation, and end-to-end stream parity can be reviewed independently.

##### M10.8a: MTP State Machine Contract And Availability Check

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
- Validation gate: contract checker with negative tests, Python/JSON syntax,
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

##### M10.8b: Rust MTP Decision Planner

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
- Validation gate: Rust planner tests, comparator with negative tests,
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

##### M10.8c: Rust MTP Draft Kernel Orchestration Smoke

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
- Validation gate: B300 MTP draft smoke or explicit MTP-model blocker,
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

##### M10.8d: Rust Exact N=2 Verifier Orchestration Smoke

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
- Validation gate: B300 exact-N=2 smoke or explicit MTP-model blocker,
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

##### M10.8e: Rust Suffix Verifier Orchestration Smoke

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
- Validation gate: B300 suffix-verifier smoke or explicit MTP-model blocker,
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

##### M10.8f: Rust Spec Frontier Snapshot Restore And Prefix1 Commit

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
- Validation gate: B300 frontier mutation smoke, comparator with negative
  tests, targeted Rust tests, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Evidence: added a Rust model-free frontier mutation plan and comparator that
  pin snapshot, restore, prefix1 commit, ratio-4 index handling, `mtp_n_raw`
  save/restore, and invisible speculative-row policy against current-C anchors
  and M10.7d3 restored-frontier evidence. Validation passed targeted Rust
  frontier-plan tests, the comparator with 8 cases, 145 checks, and 8 negative
  mutations, JSON output parsing, Python syntax, the live B300 missing-MTP
  blocker command, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and unified parity with 64 passed, 42 skipped, and 0
  failed. Non-interactive Claude review returned `NO BLOCKERS`.

##### M10.8g: Rust MTP End-To-End Stream Parity

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
- Validation gate: B300 MTP comparator or explicit support-artifact blocker,
  server/runtime parity checks, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
- Status: done.
- Evidence: split before implementation after inspecting
  `ds4_session_eval_speculative_argmax` and the completed M10.8a through
  M10.8f contracts. Validation passed the live B300 support-artifact blocker
  command, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and unified parity with 64 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- Evidence: M10.8g closes with the M10.8g4b explicit support-artifact blocker
  closure because no B300 MTP support GGUF is present. The final closure
  records `support_absent_blocker_closure`, `support_present_comparator`
  `not_run` due to `support_artifact_absent`, `blocked_missing_mtp_model`,
  empty `mtp_candidates=`, and no MTP-off or MTP-enabled parity claim.

##### M10.8g1: MTP Stream Parity Contract And Blocker

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
- Validation gate: contract checker with negative tests, Python/JSON syntax,
  B300 support-artifact blocker command, `cargo fmt --all -- --check`, `git
  diff --check`, unified parity report, and non-interactive Claude review with
  no blockers.
- Evidence: added
  `ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json` and
  `ds4-parity/check_mtp_stream_parity_contract.py`, a stream-level contract
  checker that links 12 end-to-end speculative outcomes to the M10.8a decision
  rows and current-C `ds4_session_eval_speculative_argmax` anchors. Validation
  passed JSON syntax, Python syntax, the checker with 12 cases, 368 checks, and
  8 negative mutations, the live B300 missing-MTP blocker command, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and unified
  parity with 65 passed, 42 skipped, and 0 failed. Non-interactive Claude
  review returned `NO BLOCKERS`.

##### M10.8g2: Rust MTP Stream Outcome Planner

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
- Validation gate: comparator with negative tests, targeted Rust tests, JSON
  parsing, Python syntax, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
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

##### M10.8g3: Rust Runtime Guard And Target-Stream No-Drift Smoke

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
- Validation gate: server/runtime parity checks, targeted Rust tests, missing
  support smoke, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, unified parity report, and non-interactive Claude review with
  no blockers.
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

##### M10.8g3a: Rust Runtime MTP Guard Contract And Static Wiring

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
- Validation gate: comparator with negative tests, targeted Rust tests, JSON
  parsing, Python syntax, live B300 missing-support blocker check, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, unified
  parity report, and non-interactive Claude review with no blockers.
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

##### M10.8g3b: Runtime Target-Stream No-Drift Comparator

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
- Validation gate: runtime comparator with negative tests, server/runtime parity
  checks, targeted Rust tests, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
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

##### M10.8g3c: B300 Missing-Support Runtime Smoke

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
- Validation gate: live B300 missing-support smoke, comparator with negative
  tests, targeted Rust tests, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
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

##### M10.8g4: B300 Support-Model End-To-End Comparator

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
- Validation gate: B300 MTP comparator or explicit support-artifact blocker,
  server/runtime parity checks, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
- Status: done.
- Split: this stage was split into M10.8g4a and M10.8g4b before
  implementation so the B300 support-artifact branch decision stayed separate
  from the final support-model comparator or explicit blocker closure.
- Evidence: split M10.8g4 into M10.8g4a B300 support-artifact branch decision
  and M10.8g4b final support comparator or explicit blocker closure. The live
  B300 support-artifact probe still recorded `/workspace/ds4/ds4flash.gguf`,
  absent `/workspace/ds4/missing-mtp.gguf`, and empty `mtp_candidates=`.
  Validation passed `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 69 passed, 43 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- Evidence: M10.8g4b closed this parent through the support-absent branch. The
  final B300 closure artifact records the target model identity, absent
  `/workspace/ds4/missing-mtp.gguf`, empty support candidates, not-run
  support-present comparator, explicit `blocked_missing_mtp_model` result, and
  the claim policy forbidding `MTP-off pass` and `MTP-enabled parity`.

##### M10.8g4a: B300 Support-Artifact Branch Decision

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
- Validation gate: live B300 support-artifact check, comparator with negative
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

##### M10.8g4b: B300 End-To-End Blocker Or Support Comparator Closure

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
- Validation gate: B300 support comparator or blocker comparator with negative
  tests, server/runtime parity checks, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added `ds4-parity/compare_mtp_end_to_end_closure.py`,
  `ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json`, README
  instructions, unified report wiring, and an exact B300 rerun hook. The
  closure consumes the M10.8g4a support-branch decision, M10.8g1 stream
  blocker, and M10.8g3c Rust runtime blocker.
- Evidence: live B300 closure validation passed with 58 checks and 7 negative
  mutations after refreshing the M10.8g4a branch decision. The artifact records
  `support_absent_blocker_closure`, `support_present_comparator` `not_run` due
  to `support_artifact_absent`, `/workspace/ds4/ds4flash.gguf` at
  86,720,111,488 bytes, absent `/workspace/ds4/missing-mtp.gguf`, empty
  `mtp_candidates=`, `blocked_before_stream` visibility, checkpoint delta 0,
  no cache/KVC visibility, `blocked_missing_mtp_model`, next stage `M10.9`,
  and no MTP-enabled parity claim. Local validation passed Python syntax, the
  comparator with 58 checks and 7 negative mutations, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 71 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.

#### M10.9: Runtime Graph End-To-End And Benchmark Closure

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
- Validation gate: B300 Rust-runtime end-to-end suite, server parity report,
  benchmark CSV comparator, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and non-interactive Claude review with no
  blockers.
- Status: split before implementation into M10.9a through M10.9f so runtime
  route activation, official-vector quality, long-context quality, tool/server
  quality, benchmark comparison, and final closure remain independently
  reviewable.
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

##### M10.9a: Runtime Graph Closure Matrix And Rerun Contract

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
- Validation gate: matrix checker with negative tests, B300 fixture-readiness
  probe, server/runtime parity report, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
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

##### M10.9b: Rust Runtime Graph Route Switch And Preflight

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
- Validation gate: targeted Rust tests, route-preflight comparator with
  negative tests, server/runtime parity report, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added a shared Rust `RuntimeGraphRoute` selector, wired
  `--runtime-graph`/`--runtime-graph-route` through one-shot, interactive, and
  server runtime binaries, and recorded
  `ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json`.
  The route-preflight artifact covers target-stream, disabled-route, invalid
  selector, CUDA/non-CUDA unsupported graph-route, missing-model, and server
  KVC preflight cases. Unsupported graph selection exits 99 before model open,
  stream output, checkpoint/cache mutation, or server KVC directory creation;
  target-stream and `off` keep the existing missing-model behavior.
- Validation evidence: `python3
  ds4-parity/check_runtime_graph_route_preflight.py --negative-test` passed
  with 274 checks and 8 negative mutations, `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test` stayed
  green with 118 checks and 8 negative mutations after status advanced to
  M10.9c, targeted Rust route/parser/server tests passed,
  `python3 ds4-parity/run_server_parity_report.py` passed with 10 passed, 3
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` passed with 73 passed,
  46 skipped, and 0 failed. Non-interactive Claude review returned `NO
  BLOCKERS`.

##### M10.9c: B300 Official-Vector Rust Runtime Gate

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
- Validation gate: live B300 Rust runtime official-vector run, comparator with
  negative tests, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, unified parity report, and non-interactive Claude review with
  no blockers.
- Evidence: added the Rust `ds4-runtime-official-vectors-rs` capture binary
  and `ds4-parity/run_runtime_graph_official_vectors.py`; captured
  `ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json` on
  B300. The artifact records route `graph`, backend `cuda`, exact q2-imatrix
  model hash, `official.vec` hash, raw Rust stdout/stderr, selected-token
  matches, top-logprob rows, official-top deltas, and the current-C
  `long_memory_archive` skip reason.
- Validation evidence: live B300 Rust runtime official-vector capture passed
  with 1,958 checks, max official-logprob delta 0.678254604, and 8 negative
  mutations. Local comparator, closure matrix, route preflight, Rust binary
  tests, `cargo test --workspace`, server parity report, formatter check, diff
  check, and unified parity report passed; non-interactive Claude review
  returned `NO BLOCKERS`. Exact command evidence is recorded in
  `.memory/status.md`.

##### M10.9d: B300 Long-Context Rust Runtime Gate

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
- Validation gate: live B300 Rust runtime long-context run, comparator with
  negative tests, server/runtime parity report, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added the Rust `ds4-runtime-long-context-rs` capture binary and
  `ds4-parity/run_runtime_graph_long_context.py`; captured
  `ds4-parity/baselines/graph/m10.9d/runtime-long-context.json` on B300. The
  artifact records route `graph`, backend `cuda`, exact q2-imatrix model hash,
  long-context prompt hash, current-C `./ds4_test --long-context` stdout/stderr,
  raw Rust stdout/stderr, 30,474 prompt tokens, 76 completion tokens, `stop`,
  exact fact-recall output, and cache/KVC write accounting equal to the prompt
  token count.
- Validation evidence: live B300 Rust runtime long-context capture passed with
  126 checks and 8 negative mutations. Local comparator, Rust binary
  check/tests, Python syntax checks, `cargo test --workspace`, server parity
  report, formatter check, diff check, unified parity report, and
  non-interactive Claude review passed. Exact command evidence is recorded in
  `.memory/status.md`.

##### M10.9e: Tool-Call Quality And Server Replay Rust Runtime Gate

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
- Validation gate: live B300 tool-call quality run, server/runtime parity
  report, comparator negative tests, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, unified parity report, and non-interactive
  Claude review with no blockers.
- Evidence: extended `ds4-parity/run_tool_call_quality.py` into a
  self-contained Rust graph tool/server artifact comparator, wired it into the
  unified report and README, and captured
  `ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json` on B300. The
  artifact records route `graph`, backend `cuda`, exact q2-imatrix model hash,
  current-C `./ds4_test --tool-call-quality` stdout/stderr, raw Rust
  request/response/trace/log blobs for fast and exact quality cases, HTTP 200,
  finish `tool_calls`, tool `list_files`, arguments `{"path":"."}`, and
  trace/cache ledger markers.
- Validation evidence: live B300 Rust server runtime tool-call capture passed
  with 167 checks and 8 negative mutations. Local comparator, route preflight,
  closure matrix, Rust server-runtime tests, Python syntax checks, workspace
  tests, server parity report, formatter check, diff check, unified parity
  report, and non-interactive Claude review passed. Exact command evidence is
  recorded in `.memory/status.md`.

##### M10.9f: Benchmark Comparator And Milestone 10 Closure

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
- Validation gate: live B300 Rust benchmark run, benchmark comparator with
  negative tests, final closure checker, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, unified parity report, and
  non-interactive Claude review with no blockers.
- Evidence: added `ds4-runtime-graph-bench-rs`, exposed Rust session snapshot
  and EOS-excluding argmax helpers needed to mirror `ds4-bench`, added
  `ds4-parity/run_runtime_graph_bench.py`, wired the comparator into the
  unified report and README, and captured
  `ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json` on B300.
  The artifact records route `graph`, backend `cuda`, exact q2-imatrix model
  hash, prompt hash, short/long benchmark CSV rows, exact context frontiers,
  prefill intervals, generation-token counts, KVC snapshot bytes, and
  M10.9a through M10.9e gate status.
- Validation evidence: live B300 Rust benchmark closure passed with 349 checks
  and 8 negative mutations. The artifact documents 7 older M0.6 decode
  throughput threshold misses, reproduces the same drift with same-session
  current-C `ds4-bench`, and verifies Rust stays within the same-session
  current-C threshold. Local comparator, workspace tests, server parity report,
  formatter check, diff check, unified parity report, and non-interactive
  Claude review passed. Exact command evidence is recorded in
  `.memory/status.md`.

## Milestone 11: Agent Trace Replay

Port the integrated coding agent only after runtime and server parity are
stable.

Status: split before implementation into M11.1 through M11.4 so the broad
agent-port work remains replay-comparable:

- M11.1: Agent Trace Replay Oracle And Fixture Contract.
- M11.2: Rust Agent Rendered Context Replay.
- M11.3: Deterministic Tool Stub And Session Command Replay.
- M11.4: Rust Agent Loop And Manual Smoke.

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

### M11 Split

M11 is split before implementation because the integrated agent combines live
model sampling, DSML parsing, tool execution, terminal UI, context compaction,
and session persistence. Each substage must add a fixture and comparator before
claiming more live behavior:

- M11.1 records the no-model current-C replay oracle and validates that Rust
  can emit the same normalized fixture.
- M11.2 compares rendered agent context from the scripted fixture without
  executing tools.
- M11.3 replays deterministic tool stubs and session commands.
- M11.4 wires the Rust live agent loop and keeps manual sessions as a final
  smoke only after the replay gates pass.

#### M11.1: Agent Trace Replay Oracle And Fixture Contract

- Status: complete.
- Oracle: current-C `./ds4-agent --dump-agent-trace-oracle` no-model replay
  dump.
- Fixture: `ds4-parity/baselines/agent/m11.1/current-c.json` with normalized
  workspace/session markers, one scripted DSML tool round, deterministic tool
  output, and save/list/switch/history/new session-command flow.
- Comparator: `ds4-parity/compare_agent_trace_replay.py --negative-test`.
- Acceptance: Rust `ds4-agent-trace-replay-rs` emits the same oracle JSON,
  parsed DSML tool sequence matches `expected.tool_sequence`, transcript roles
  match fixture expectations, session command order is normalized, and manifest
  size/SHA checks pass.
- Validation evidence:
  - `arch -arm64 make ds4-agent` passed locally.
  - `./ds4-agent --dump-agent-trace-oracle ds4-parity/baselines/agent/m11.1/current-c.json` regenerated the baseline.
  - `python3 ds4-parity/compare_agent_trace_replay.py --negative-test` passed
    with 225 checks.
  - Unified parity wiring includes `M11.1 Agent trace replay oracle`.

#### M11.2: Rust Agent Rendered Context Replay

- Status: complete.
- Goal: replay the M11.1 scripted events into the Rust prompt/context renderer
  and compare normalized rendered-message boundaries before tool execution is
  ported.
- Oracle: M11.1 current-C replay fixture plus existing chat/DSML prompt
  rendering contracts.
- Fixture: M11.1 scripted model events and expected transcript roles.
- Comparator: `ds4-parity/compare_agent_rendered_context.py --negative-test`.
- Acceptance: rendered system/user/assistant/tool/assistant boundaries,
  assistant EOS insertion, and visible final output match the fixture.
- Validation evidence:
  - `cargo run --quiet -p ds4-gguf --bin ds4-agent-rendered-context-rs >
    ds4-parity/baselines/agent/m11.2/rendered-context.json` regenerated the
    rendered-context artifact.
  - `python3 ds4-parity/compare_agent_rendered_context.py --negative-test`
    passed with 178 checks.
  - Unified parity wiring includes `M11.2 Agent rendered-context replay`.

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
- Validation evidence:
  - `cargo run --quiet -p ds4-gguf --bin ds4-agent-deterministic-replay-rs >
    ds4-parity/baselines/agent/m11.3/deterministic-replay.json` regenerated
    the deterministic replay artifact.
  - `python3 ds4-parity/compare_agent_deterministic_replay.py
    --negative-test` passed with 230 checks.
  - Unified parity wiring includes `M11.3 Agent deterministic tool/session
    replay`.

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
- Validation evidence:
  - `cargo run --quiet -p ds4-gguf --bin ds4-agent-loop-smoke-rs >
    ds4-parity/baselines/agent/m11.4/loop-smoke.json` regenerated the no-model
    loop-smoke artifact.
  - `python3 ds4-parity/compare_agent_loop_smoke.py --negative-test` passed
    with 223 checks.
  - Unified parity wiring includes `M11.4 Agent no-model loop smoke`.

## Milestone 12: Backend Replacement Parity

Status: split before implementation into M12.1 through M12.6.

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

### M12 Split

M12 is split before implementation because backend replacement spans the
operation ABI, tensor fixtures, backend facade ownership, backend-specific
kernel execution, runtime route selection, and final removal decisions:

- M12.1: Backend Boundary Inventory And Claim Matrix.
- M12.2: Operation Tensor Fixture Capture.
- M12.3: Rust Backend Facade Parity Harness.
- M12.4: First Backend Replacement Slice.
- M12.5: Runtime Backend Route Gate.
- M12.6: Backend Replacement Closure And Removal Decision.

#### M12.1: Backend Boundary Inventory And Claim Matrix

- Status: complete.
- Goal: inventory the current backend boundary and define which pieces remain
  C/CUDA/Metal sidecars versus Rust-owned behavior before any replacement
  claim.
- Oracle: current `ds4_gpu.h`, backend build/link scripts, Rust GPU FFI
  wrappers, M10.5c4c1 CUDA smoke contract, M10.9 runtime graph closure matrix,
  and committed B300 benchmark artifacts.
- Fixture: backend-boundary inventory JSON covering operation families,
  ownership state, required platform, model requirement, and claim boundary.
- Comparator: inventory checker that fails on missing operation families,
  unsupported backend overclaims, missing B300 rerun commands, or removal
  claims before replacement gates exist.
- Acceptance: every backend operation family has a named owner state
  (`current-c`, `ffi-wrapped`, `rust-planned`, or `rust-owned`), a fixture
  source, a comparator path, and a no-removal claim boundary.
- Drift policy: when C/CUDA/Metal signatures, Rust FFI wrappers, or runtime
  route selectors change, refresh the inventory and rerun the checker before
  implementing an M12 replacement slice.
- Evidence:
  - Added `ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json`.
  - Added `ds4-parity/check_backend_boundary_inventory.py --negative-test`.
  - Unified parity wiring includes `M12.1 Backend boundary inventory`.
- Validation:
  - `python3 -m py_compile ds4-parity/check_backend_boundary_inventory.py
    ds4-parity/check_runtime_graph_closure_matrix.py
    ds4-parity/run_parity_report.py` passed.
  - `python3 -m json.tool
    ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json`
    passed.
  - `python3 ds4-parity/check_backend_boundary_inventory.py --negative-test`
    passed.
  - `python3 ds4-parity/check_runtime_graph_closure_matrix.py
    --negative-test`, `cargo fmt --all -- --check`, `git diff --check`, and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles` passed
    with 82 passed, 50 skipped, 0 failed.

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
  - Unified parity wiring includes `M12.2 Backend operation tensor fixtures`.
- Validation:
  - Live B300 capture passed existing pair comparators for first-kernel
    embedding, layer-0 QKV/RoPE, layer-0 attention output, layer-0
    FFN/router/MoE, and full output-head/logits.
  - `python3 ds4-parity/check_backend_operation_fixtures.py --negative-test`
    passed with 576 checks.
  - `python3 ds4-parity/run_parity_report.py --skip-local-oracles` passed
    with 83 passed, 50 skipped, 0 failed.

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
  - Unified parity wiring includes `M12.3 Backend facade replay harness`.
  - The replay artifact maps each selected M12.2 fixture to ordered
    `DecodeBackend` calls, tensor bindings, synchronized candidate evidence,
    and current facade error propagation without changing runtime routes or
    claiming backend replacement.
- Validation:
  - `python3 ds4-parity/check_backend_facade_replay.py --negative-test`
    passed with 769 checks.
  - `python3 ds4-parity/check_backend_operation_fixtures.py --negative-test`
    passed with 576 checks after allowing M12.4 as the next active item.
  - `python3 ds4-parity/check_runtime_graph_closure_matrix.py
    --negative-test`, `cargo fmt --all -- --check`, `git diff --check`, and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles` passed
    with 84 passed, 50 skipped, 0 failed.

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
  evidence and recording whether the change is route plumbing, numeric kernel
  drift, or benchmark variance.
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

## Milestone 13: Backend Replacement Expansion

Status: split before implementation into M13.0 through M13.5.

M12.6 closed the first backend replacement pass with no removals allowed. M13
keeps the current C/CUDA/Metal backend as sidecar and oracle, then broadens the
only existing route-gated family before considering a separate backend family.
The chosen path is `embedding_and_indexer` because M12.4/M12.5 already route
`ds4_gpu_embed_token_hc_tensor` through an opt-in replacement slice, while
M12.6 lists six remaining operations in the same family.

Oracle:

- M12.1 backend boundary inventory, M12.6 closure matrix, M10.2 graph
  operation inventory, and the M10.5/M10.6 current-C execution comparators for
  prefill and long indexed-attention.

Comparator:

- M13 decision and fixture-matrix checkers, per-operation current-C tensor
  comparators, and the M10.9 end-to-end runtime gates once route expansion is
  attempted.

Acceptance:

- Every remaining embedding/indexer operation is assigned to a covered fixture
  or explicit fixture-gap stage before any route expansion.
- Expanded replacement claims remain opt-in and fail closed for unsupported
  backends.
- The default route remains `current-backend`.
- C/CUDA/Metal backend removal remains rejected until all operations in the
  family and runtime gates are covered.

Drift policy:

- Any embedding/indexer operation signature, fixture, or route-selector change
  must refresh the M13 matrix and preserve the M12 current-backend artifacts as
  oracles.

### M13 Split

- M13.0: Backend Expansion Decision.
- M13.1: Embedding/Indexer Expansion Fixture Matrix.
- M13.2: Batched Embedding Replacement Slice.
- M13.3: Indexed Decode Selection Replacement Slice.
- M13.4: Batch Indexer Fixture Gap Closure.
- M13.5: Embedding/Indexer Route Gate And Closure.

#### M13.0: Backend Expansion Decision

- Status: complete.
- Goal: choose the post-M12 replacement direction and split it into
  replay-comparable M13 stages before any new route implementation.
- Oracle: M12.6 closure matrix, M12.1 operation inventory, M10.2 graph
  operation inventory, and existing M10.5/M10.6 comparator coverage.
- Fixture: `ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json`.
- Comparator: `ds4-parity/check_backend_expansion_decision.py
  --negative-test`.
- Acceptance: M13 chooses to broaden the existing `embedding_and_indexer`
  route, maps all six remaining M12.6 operations to M13 work, and keeps
  removals/default-route replacement/general backend replacement/kernel
  replacement claims false.
- Drift policy: if M12.6 closure, M12.1 inventory, graph operation inventory,
  or existing prefill/indexed-attention comparator coverage changes, refresh
  the decision artifact before starting M13.1.
- Evidence:
  - Added
    `ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json`.
  - Added `ds4-parity/check_backend_expansion_decision.py --negative-test`.
  - The decision selects the existing `embedding_and_indexer` route for M13
    broadening, assigns all six remaining M12.6 operations to M13.1 through
    M13.5, and keeps removals plus backend-replacement overclaims false.
  - `python3 ds4-parity/check_backend_expansion_decision.py --negative-test`
    passed with 186 checks.
  - Python syntax, JSON formatting, the M12.6 closure checker, `cargo fmt
    --all -- --check`, `git diff --check`, and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` passed with 88
    passed, 50 skipped, 0 failed.

#### M13.1: Embedding/Indexer Expansion Fixture Matrix

- Status: complete.
- Goal: turn the M13.0 decision into an operation-by-operation matrix that
  identifies existing fixture coverage and fixture gaps for every remaining
  `embedding_and_indexer` operation.
- Oracle: M13.0 decision, M12.6 remaining-operation list, M12.1 inventory,
  M10.2 graph inventory, and existing prefill/indexed-attention comparators.
- Fixture:
  `ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json`.
- Comparator: `ds4-parity/check_backend_expansion_matrix.py --negative-test`.
- Acceptance: all six remaining operations are present, covered operations
  reference an executable comparator, gap operations stay explicitly blocked,
  and no route, default-route, or removal claim changes.
- Drift policy: operation-list drift must be accepted only by refreshing the
  current C/CUDA/Metal inventory and preserving the previous M13.0 decision as
  a historical oracle.
- Evidence:
  - Added
    `ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json`.
  - Added `ds4-parity/check_backend_expansion_matrix.py --negative-test`.
  - The matrix classifies `ds4_gpu_embed_tokens_hc_tensor`,
    `ds4_gpu_indexer_score_one_tensor`, and `ds4_gpu_indexer_topk_tensor` as
    pair-comparator-ready; keeps `ds4_gpu_indexer_scores_prefill_tensor`,
    `ds4_gpu_indexer_scores_decode_batch_tensor`, and
    `ds4_gpu_dsv4_topk_mask_tensor` as fixture-gap operations for M13.4; and
    keeps all route/default-route/removal claims unchanged.
  - `python3 ds4-parity/check_backend_expansion_matrix.py --negative-test`
    passed with 186 checks.
  - Python syntax, JSON formatting, `cargo fmt --all -- --check`, `git diff
    --check`, and `python3 ds4-parity/run_parity_report.py
    --skip-local-oracles` passed with 89 passed, 50 skipped, 0 failed.

#### M13.2: Batched Embedding Replacement Slice

- Status: complete.
- Goal: add an opt-in replacement slice for
  `ds4_gpu_embed_tokens_hc_tensor`.
- Oracle: current-C whole/chunked/resumed prefill output comparators.
- Fixture:
  `ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json`.
- Comparator: `ds4-parity/check_backend_batched_embedding_slice.py
  --negative-test` plus M10.6 prefill pair comparators.
- Acceptance: batched embedding output fields match current-C fixtures within
  documented tolerance, unsupported backends fail closed, and default runtime
  routing is unchanged.
- Drift policy: any output drift requires a same-B300 current-C oracle refresh
  and comparator rationale before the slice can remain active.
- Evidence:
  - Added the M13.2 Rust replacement slice descriptor to
    `rust/ds4-gpu/src/replacement_slice.rs`.
  - Extended `ds4-backend-replacement-slice` with explicit `--slice`
    selection while preserving the M12.4 default output.
  - Added
    `ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json`.
  - Added `ds4-parity/check_backend_batched_embedding_slice.py
    --negative-test`.
  - The slice selects `ds4_gpu_embed_tokens_hc_tensor`, uses M13.1
    pair-comparator-ready prefill coverage, fails closed for CPU, Metal, and
    runtime-default-route, and keeps runtime route, general backend
    replacement, and kernel replacement claims false.
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
- Acceptance: indexer score and selected-row outputs match current-C fixtures,
  unsupported backends fail closed, and default runtime routing is unchanged.
- Drift policy: score or selected-row drift requires a same-B300 long
  indexed-attention oracle refresh.
- Evidence:
  - Added explicit Rust replacement slice descriptors for
    `ds4_gpu_indexer_score_one_tensor` and `ds4_gpu_indexer_topk_tensor`.
  - Added the M13.3 slice-set fixture and
    `ds4-parity/check_backend_indexed_decode_slice.py --negative-test`.
  - The slices use the M13.1 pair-comparator-ready rows and the M10.5c4d3 long
    indexed-attention comparator, require explicit per-slice selection, fail
    closed for CPU, Metal, and runtime-default-route, and keep runtime route,
    general backend replacement, and kernel replacement claims false.
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
  and explicit tolerance rationale.
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
- Acceptance: the expanded route remains opt-in, the default route remains
  `current-backend`, and removal is rejected unless every operation in the
  family is covered.
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

## Post-M13 Roadmap Decision

- Status: complete.
- Goal: close the active `post-M13 roadmap decision` gate without selecting an
  unsupported removal or default-route-promotion stage.
- Oracle: M13.0 through M13.5 artifacts plus the M10.9 runtime graph evidence.
- Fixture:
  `ds4-parity/baselines/roadmap/post-m13/post-m13-roadmap-decision.json`.
- Comparator: `ds4-parity/check_post_m13_roadmap_decision.py
  --negative-test`.
- Acceptance: all M13 stages are recorded complete, no next implementation
  stage is selected, the default route remains `current-backend`, retained
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

## Milestone 14: Rust CUDA Ownership Via cuda-oxide

Status: split before implementation into M14.0 through M14.6. M14.0 is
complete; M14.1 is the active implementation stage.

The post-M13 decision deferred further CUDA scope until a new oracle-backed
roadmap existed. The new scope is complete CUDA ownership transfer: replace
`ds4_cuda.cu` with Rust CUDA resource management and kernels implemented using
the verified `cuda-oxide` substrate, while retaining current-C CUDA execution
as the comparison oracle until the removal gate passes.

Source boundary:

- `ds4_gpu.h` declares 81 public GPU ABI functions and
  `rust/ds4-gpu-sys/src/lib.rs` mirrors all 81 through FFI.
- `ds4_cuda.cu` still implements that ABI, exports two additional internal
  helpers, and contains 113 unique CUDA kernel symbols.
- `rust/ds4-gpu/build.rs` still compiles `ds4_cuda.cu` for Linux CUDA builds
  and links `cudart` and `cublas`; existing Rust replacement slices therefore
  do not yet constitute Rust CUDA kernel ownership.
- `cuda-oxide` `main` at
  `0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200` provides Rust CUDA
  compilation/launch, resource and residency wrappers, cuBLAS wrappers,
  low-precision types, and deterministic selection primitives. It is a
  substrate for this rewrite, not evidence that DS4-specific kernels are
  already ported.

Oracle:

- Current `ds4_cuda.cu` output and lifetime behavior, with M10 through M13
  graph, quality, route, and benchmark artifacts retained through removal.

Comparator:

- A source-hashed ownership inventory for all CUDA exports and kernel symbols,
  followed by stage-specific B300 tensor and end-to-end comparisons.

Acceptance:

- Every CUDA export and kernel has exactly one Rust-ownership stage.
- Each ownership stage must prove its assigned execution on B300 before the
  next stage can depend on it.
- Default CUDA route promotion and removal of `ds4_cuda.cu` remain forbidden
  until M14.6 passes all end-to-end gates.

Drift policy:

- Any change in `ds4_cuda.cu`, `ds4_gpu.h`, the Rust GPU FFI/build wiring, or
  the assigned CUDA export/kernel sets requires refreshing M14.0 before
  claiming further ownership.
- Any numeric or performance drift requires a same-B300 current-C comparison
  with explicit tolerance or performance rationale.

Stage split:

- M14.0: CUDA Rust Ownership Inventory And Adoption Contract.
- M14.1: cuda-oxide Substrate And Tensor Residency.
- M14.2: Embedding Indexer And Elementwise Kernels.
- M14.3: Dense Projection Quantization And Norm Kernels.
- M14.4: RoPE KV Compressor And Attention Kernels.
- M14.5: Router MoE And Hyperconnection Kernels.
- M14.6: CUDA Route Promotion And C CUDA Removal Gate.

#### M14.0: CUDA Rust Ownership Inventory And Adoption Contract

- Status: complete.
- Goal: freeze the complete CUDA-to-Rust ownership surface and map it onto
  executable `cuda-oxide` adoption stages before changing CUDA runtime
  ownership.
- Oracle: current `ds4_cuda.cu`, `ds4_gpu.h`, Rust FFI/build wiring, and
  inspected `cuda-oxide` `main` revision.
- Fixture:
  `ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json`.
- Comparator: `ds4-parity/check_cuda_rust_ownership_inventory.py
  --negative-test`.
- Acceptance: all 83 CUDA-exported functions and all 113 unique CUDA kernel
  symbols are assigned exactly once; the current Rust FFI/CUDA build boundary
  is recorded; removal and default-route claims remain false; M14.1 is
  selected as the first implementation stage.
- Drift policy: source hash, exported symbol, kernel symbol, or cuda-oxide
  revision drift requires an explicit inventory refresh before continuing.
- Evidence:
  - Added the M14.0 inventory fixture and checker.
  - The inventory records the verified `cuda-oxide` revision and its Rust
    resource, launch, BLAS, low-precision, and selection capability evidence.
  - `python3 ds4-parity/check_cuda_rust_ownership_inventory.py
    --negative-test` passed with 124 checks.
  - Python syntax, JSON formatting, `git diff --check`, the post-M13 decision
    checker, and `python3 ds4-parity/run_parity_report.py
    --skip-local-oracles` passed with 95 passed, 50 skipped, 0 failed.
  - Non-interactive Claude review was unavailable because the local CLI
    reported `Not logged in`; adversarial self-review checked symbol
    extraction, one-stage assignment, source-hash refresh behavior, and
    historical-checker compatibility with no material finding.

#### M14.1: cuda-oxide Substrate And Tensor Residency

- Status: split before implementation into M14.1a through M14.1c.
- Goal: replace the CUDA resource/lifetime boundary with a Rust-owned
  `cuda-oxide` path before any DS4 compute kernel consumes it.
- Oracle: current `ds4_cuda.cu` tensor/resource lifetime and model-residency
  behavior plus the M14.0 assigned ownership list.
- Comparator: stage-specific B300 executable smokes and tensor/resource
  fixtures; no graph route changes until this stage closes.
- Acceptance: each substage is opt-in and proves concrete Rust-owned CUDA
  behavior while `ds4_cuda.cu` remains the default and comparison oracle.
- Drift policy: if M14.0 source hashes or assigned resource exports change,
  refresh M14.0 before continuing this split.

##### M14.1a: Host Substrate Buffer Roundtrip

- Status: complete.
- Goal: add a feature-gated Rust CUDA crate pinned to the verified
  `cuda-oxide` revision and prove context/stream ownership plus device and
  managed-buffer roundtrip on B300.
- Oracle: `ds4_gpu_init`, tensor allocation/write/read, synchronization, and
  managed allocation behavior in current `ds4_cuda.cu`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json`.
- Comparator: `ds4-parity/check_cuda_oxide_substrate_smoke.py
  --negative-test` plus an executable Rust smoke binary on B300.
- Acceptance: an opt-in `cuda-oxide-backend` feature uses the pinned fork,
  allocates/transfers CUDA device data and owns a managed buffer through Rust,
  and default graph/runtime routing remains unchanged.
- Drift policy: CUDA/toolchain/substrate output drift requires a new B300
  capture; this stage does not claim arbitrary fill, model cache, kernel, or
  route ownership.
- Evidence:
  - Added the feature-gated `rust/ds4-cuda` crate pinned to `cuda-oxide`
    revision `0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200`, with a
    Rust-owned `CudaContext`, `CudaStream`, `DeviceBuffer`, and
    `ManagedBuffer` substrate surface.
  - Added
    `ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json` and
    `ds4-parity/check_cuda_oxide_substrate_smoke.py --negative-test`.
  - On B300 pod `ds4-rust-port-b300` at node `c1v17-b300n1-nic1`, after
    provisioning pod-local `nightly-2026-04-03` plus `libclang-dev`, the
    feature-enabled Rust smoke executed on `NVIDIA B300 SXM6 AC` and passed
    device roundtrip, zeroed-buffer roundtrip, and managed-buffer lifetime
    checks while reporting no kernel or route ownership.
  - Validation passed: `cargo test --workspace`, `cargo fmt --all -- --check`,
    `python3 ds4-parity/check_cuda_oxide_substrate_smoke.py --negative-test`
    (53 checks), `python3 ds4-parity/check_cuda_rust_ownership_inventory.py
    --negative-test` (124 checks), `git diff --check`, and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` (96 passed, 50
    skipped, 0 failed).
  - Non-interactive Claude review could not run because the local CLI reported
    `Not logged in`; adversarial self-review found no material issue in the
    opt-in feature boundary, immutable dependency revision, limited B300
    ownership claim, or retained current-C oracle/default route. LLVM 21
    remains a prerequisite for later cuda-oxide kernel-compilation stages,
    not for this host-substrate smoke.

##### M14.1b: Model Residency And Command Lifetime

- Status: split before implementation into M14.1b1 through M14.1b4.
- Goal: move model residency, caching/policy, and command-lifetime ownership
  onto the Rust substrate in separately executable cuts.
- Oracle: current-C model-backed B300 resource behavior.
- Comparator: stage-specific B300 resource fixtures and closure checks.
- Acceptance: each substage proves only its named Rust-owned resource
  behavior without changing the default runtime route.
- Drift policy: any model-residency or command-order drift requires refreshed
  B300 evidence and a current-C comparison.

###### M14.1b1: Bounded Model Residency Handles

- Status: complete.
- Goal: prove cuda-oxide managed advice/prefetch, mapped-host device address,
  and registered caller-owned host lifetime on a bounded window read from the
  real B300 model.
- Oracle: current `cuda_model_prefetch_range` and
  `ds4_gpu_set_model_map` resource intent, limited to host residency handles
  rather than cache selection or graph consumption.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json`.
- Comparator: `ds4-parity/check_model_residency_handles_smoke.py
  --negative-test` plus executable Rust B300 smoke over a bounded GGUF window.
- Acceptance: the opt-in Rust crate owns a managed prefetch/advice sequence,
  a mapped-host allocation, and a registered caller-owned range with
  device-visible pointers; it does not claim the complete model map, DS4
  kernels, or route ownership.
- Drift policy: model identity, model-window size, cuda-oxide revision, or
  live CUDA/toolchain drift requires a new B300 capture.
- Evidence:
  - Extended `rust/ds4-cuda` with managed read-mostly/preferred-device
    advice and prefetch, mapped-host allocation, and registered-host range
    guards; the scope contract keeps complete-model-map, kernel, and route
    ownership false.
  - Added
    `ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json`
    and `ds4-parity/check_model_residency_handles_smoke.py --negative-test`.
  - On B300 pod `ds4-rust-port-b300`, the feature-enabled Rust smoke read a
    4096-byte prefix of the pinned 86,720,111,488-byte GGUF and passed
    managed advice/prefetch, mapped device-pointer, and registered
    host-pointer checks on `NVIDIA B300 SXM6 AC`; a live `sha256sum` refresh
    confirmed model SHA256
    `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`.
  - Validation passed: `cargo test --workspace`, `cargo fmt --all -- --check`,
    `python3 ds4-parity/check_model_residency_handles_smoke.py
    --negative-test` (64 checks),
    `python3 ds4-parity/check_cuda_oxide_substrate_smoke.py --negative-test`
    (53 checks), `python3 ds4-parity/check_cuda_rust_ownership_inventory.py
    --negative-test` (124 checks), `git diff --check`, and `python3
    ds4-parity/run_parity_report.py --skip-local-oracles` (97 passed, 50
    skipped, 0 failed).
  - Non-interactive Claude review could not run because the local CLI reported
    `Not logged in`; adversarial self-review found and closed the missing
    current model-SHA confirmation, then found no material lifetime,
    synchronization, bounded-claim, or default-route issue.

###### M14.1b2: Model Map And Range Cache Policy

- Status: split before implementation into M14.1b2a through M14.1b2c.
- Goal: port model file mapping and range-cache strategies onto the Rust
  substrate without conflating an executable bounded cache with every
  current-C policy branch.

####### M14.1b2a: Owned Mmap Device Range Copy

- Status: complete.
- Goal: own model file/map lifetime in Rust and prove a bounded,
  bounds-checked model range can be copied to a cached CUDA device buffer and
  reused with byte-exact readback.
- Oracle: current-C `ds4_gpu_set_model_fd`, `ds4_gpu_set_model_map_range`,
  and `ds4_gpu_cache_model_range` intent, limited to the explicit device-copy
  range-cache branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json`.
- Comparator: `ds4-parity/check_model_range_copy_smoke.py --negative-test`
  plus executable Rust B300 smoke using the pinned GGUF.
- Acceptance: an opt-in Rust `mmap` owner and CUDA device range cache pass
  bounds, exact-readback, and reuse checks without claiming HMM, registered
  zero-copy, direct-I/O, Q8 conversion, kernel, or route ownership.
- Evidence:
  - Added Rust-owned `MappedModelFile` and `ModelRangeCache` types; copied
    ranges synchronize before entering the reusable cache and retain
    fail-closed bounds checking.
  - Added
    `ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json` and
    `ds4-parity/check_model_range_copy_smoke.py --negative-test`.
  - On B300 pod `ds4-rust-port-b300`, after removing an invalid diagnostic
    `Debug` derive rejected by feature compilation, the feature-enabled smoke
    mmaped the pinned GGUF and passed 4096-byte bounds rejection, CUDA
    copy/readback, and cache-reuse checks on `NVIDIA B300 SXM6 AC`.
  - Adversarial self-review found and fixed a rejected-null-address `mmap`
    cleanup leak; the B300 smoke rerun passed after that cleanup-only fix.
  - Validation passed: `cargo test --workspace`, `cargo fmt --all -- --check`,
    `python3 ds4-parity/check_model_range_copy_smoke.py --negative-test`
    (64 checks), prior M14.1b1/M14.1a/M14.0 gates, `git diff --check`, and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles` (98 passed,
    50 skipped, 0 failed).
  - Non-interactive Claude review could not run because the local CLI reported
    `Not logged in`; post-fix self-review found no material mmap/cache
    lifetime, synchronization, bounded-claim, or default-route issue.

####### M14.1b2b: Model Range Strategy Parity

- Status: split before implementation into M14.1b2b1 through M14.1b2b3.
- Goal: implement and compare the model-range strategy branches that are
  separable from model-cache closure, retaining current-C as the oracle.

######## M14.1b2b1: File-Staged Range Strategy

- Status: complete.
- Goal: implement explicit Rust strategy dispatch between mmap-sourced device
  copy and file-descriptor staged device copy, and compare their selected-range
  bytes on B300.
- Oracle: the device-copy and file-descriptor staging branches of current-C
  `cuda_model_range_ptr` and `cuda_model_range_ptr_from_fd`, excluding
  asynchronous staging-ring, cache-budget, and direct-I/O policy.
- Acceptance: the opt-in Rust path can cache and reuse the same bounded model
  range from either mmap or `pread`-style file staging, and both device
  readbacks are byte-exact; it does not claim O_DIRECT, registered-map, HMM,
  DS4-kernel, or runtime-route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b1/model-range-strategy-smoke.json`.
- Comparator: `ds4-parity/check_model_range_strategy_smoke.py --negative-test`
  plus executable Rust B300 smoke using the pinned GGUF.
- Evidence:
  - Added explicit `ModelRangeStrategy::{MmapDeviceCopy,
    FileStagedDeviceCopy}` selection and strategy-keyed range-cache entries;
    file staging reads a bounds-checked selected range through the Rust-owned
    file descriptor before uploading through `cuda-oxide`.
  - On B300 pod `ds4-rust-port-b300`, the feature-enabled strategy smoke
    independently cached and reused a 4096-byte prefix through both strategies
    and produced byte-exact matching readbacks on `NVIDIA B300 SXM6 AC`.
    A live SHA256 refresh confirmed the pinned GGUF identity
    `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`.
  - The stage explicitly leaves registered mapped-host ranges, pageable HMM
    advice/prefetch, O_DIRECT and asynchronous staging policy, compute kernels,
    and runtime route activation unclaimed.
  - Validation passed: `cargo test --workspace`, `cargo fmt --all -- --check`,
    `python3 ds4-parity/check_model_range_strategy_smoke.py --negative-test`
    (73 checks), retained M14.1b2a/M14.1b1/M14.1a/M14.0 gates,
    `git diff --check`, and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles` (99 passed,
    50 skipped, 0 failed).
  - Non-interactive Claude review produced no result and was terminated after
    it failed to complete; adversarial self-review found no material
    strategy-keying, source-lifetime, synchronization, bounded-claim, or
    default-route issue.

######## M14.1b2b2: Registered Range Strategy

- Status: complete.
- Goal: port page-aligned read-only mapped host registration and the
  mmap-sourced device-copy fallback taken when CUDA registration fails.
- Oracle: the page-aligned `cudaHostRegister(... cudaHostRegisterMapped |
  cudaHostRegisterReadOnly)` branch of current-C `cuda_model_range_ptr` and
  its post-registration `cudaMemcpy` fallback; file-descriptor staging
  remains the separately validated M14.1b2b1 branch.
- Acceptance: an unaligned selected range produces an aligned registration
  window; the Rust strategy either retains a live immutable registration
  guard or records the CUDA unsupported result and reuses an exact mmap-copy
  fallback cache entry without claiming B300 zero-copy support or current-C's
  cross-range suppression of later registration attempts.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b2/model-registered-range-smoke.json`.
- Comparator:
  `ds4-parity/check_model_registered_range_smoke.py --negative-test` plus
  executable Rust B300 smoke using the pinned GGUF.
- Evidence:
  - Pinned DS4 to cuda-oxide revision
    `b938480882f208045bc36ecf29da1ec5531d55ba`, which exposes an immutable
    read-only registered-host guard and propagates unsupported CUDA errors.
  - On B300 pod `ds4-rust-port-b300`, the feature-enabled smoke expanded
    requested range `13..4109` to registered range `0..8192`; CUDA returned
    error `801` (`operation not supported`), and the mmap-sourced device-copy
    fallback read back the exact requested 4096 bytes and reused its cache
    entry.
  - Successful zero-copy registration on B300, pageable HMM, O_DIRECT and
    asynchronous staging policy, cross-range unsupported-registration
    suppression, compute kernels, and runtime route activation remain
    unclaimed.
  - Validation passed: `cargo test --workspace`, `cargo fmt --all -- --check`,
    `python3 ds4-parity/check_model_registered_range_smoke.py --negative-test`
    (68 checks), retained M14 gates, `git diff --check`, B300 feature tests
    and predecessor smoke, and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
    (100 passed, 50 skipped, 0 failed).
  - Non-interactive Claude review could not run because the local CLI reported
    `Not logged in`; adversarial self-review corrected the current-C fallback
    source and retained cross-range registration suppression as an explicit
    later-stage non-claim.

######## M14.1b2b3a: Pageable HMM Range Strategy

- Status: complete.
- Goal: port page-aligned pageable-memory advice and prefetch for the mmap
  model range and prove the direct HMM pointer remains byte-exact.
- Oracle: current-C `cuda_model_prefetch_range` and the
  `g_model_hmm_direct` branch of `cuda_model_range_ptr`, excluding file
  staging and direct-I/O behavior.
- Acceptance: an unaligned selected range expands to a page-aligned pageable
  HMM window; Rust applies read-mostly and preferred-device advice, completes
  prefetch through a borrowed guard, and reads the exact requested bytes
  through its direct pointer without claiming production asynchronous policy,
  O_DIRECT, kernels, or runtime route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b3a/model-pageable-hmm-smoke.json`.
- Comparator:
  `ds4-parity/check_model_pageable_hmm_smoke.py --negative-test` plus
  executable Rust B300 smoke using the pinned GGUF.
- Evidence:
  - Pinned DS4 to corrected cuda-oxide revision
    `361300ea643688eea87eaa215d9a62a5e74a30e6`, whose borrowed pageable
    host handle requires unsafe asynchronous prefetch lifetime management;
    DS4 exposes a synchronized safe proof wrapper for this stage.
  - On B300 pod `ds4-rust-port-b300`, the feature-enabled smoke expanded
    requested range `13..4109` to pageable range `0..8192`; the device
    reported pageable-memory access with host page-table access disabled,
    accepted both advice calls and prefetch, and read back the exact
    requested bytes through the HMM direct pointer.
  - The stage explicitly leaves asynchronous production prefetch policy,
    O_DIRECT and pinned staging policy, compute kernels, and runtime route
    activation unclaimed.
  - Validation passed: `cargo test --workspace`, `cargo fmt --all -- --check`,
    `python3 ds4-parity/check_model_pageable_hmm_smoke.py --negative-test`
    (73 checks), retained M14 comparators, `git diff --check`, B300
    feature-enabled tests and predecessor smoke, and
    `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
    (101 passed, 50 skipped, 0 failed).
  - Non-interactive Claude review could not run because the local CLI reported
    `Not logged in`; adversarial self-review found and corrected the
    asynchronous borrowed-prefetch safety defect in cuda-oxide before DS4
    pinned the HMM handle.

######## M14.1b2b3b: Direct-I/O Staging Policy

- Status: split before implementation into M14.1b2b3b1 and M14.1b2b3b2.
- Goal: port file-descriptor direct-I/O and pinned staging behavior through
  separately measurable read-selection and asynchronous scheduling slices.

######### M14.1b2b3b1: Direct-I/O Pinned Read Selection

- Status: complete.
- Goal: port `O_DIRECT` open/aligned-read selection, buffered fallback, and
  synchronized pinned host-to-device upload for a bounded model range.
- Oracle: current-C `ds4_gpu_set_model_fd`, `cuda_model_stage_read`, and the
  selected upload in `cuda_model_range_ptr_from_fd`, excluding its
  multi-buffer event ring, persistent direct-I/O disable state, and arena
  budget.
- Acceptance: a normal unaligned range is read from an aligned direct-I/O
  pinned window and uploads byte-exactly; a final partial-file range selects
  the current-C-style buffered fallback and also uploads byte-exactly without
  claiming asynchronous overlap or cache-budget policy.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b3b1/model-direct-io-smoke.json`.
- Comparator:
  `ds4-parity/check_model_direct_io_smoke.py --negative-test` plus executable
  Rust B300 smoke using the pinned GGUF.
- Evidence:
  - Added a Rust pinned-staging read selection API that attempts Linux
    `O_DIRECT` through the model file descriptor, aligns its CUDA-pinned
    source window, and synchronizes the selected device upload before the
    staging allocation drops.
  - On B300 pod `ds4-rust-port-b300`, requested range `13..4109` selected an
    aligned direct read `0..8192` at alignment `4096`, and exact device
    readback passed. The model's final 13 bytes selected the buffered fallback
    because an aligned direct read would extend past the non-aligned file end,
    and that exact readback passed as well.
  - The stage explicitly leaves asynchronous staging-ring/event scheduling,
    cache-budget policy, persistent disable-after-error state, compute
    kernels, and runtime route activation unclaimed.
  - Validation passed local workspace tests, formatting and diff checks, the
    79-check direct-I/O comparator, retained M14 comparators, B300
    feature-enabled crate tests and predecessor HMM smoke, and unified parity
    with 102 passed, 50 skipped, and 0 failed. Non-interactive Claude review
    was unavailable because the local CLI reported `Not logged in`; adversarial
    self-review found no lifetime, alignment, fallback, or bounded-claim
    defect.

######### M14.1b2b3b2: Asynchronous Staging Ring And Budget Policy

- Status: complete.
- Goal: port the multi-buffer event-driven upload ring, direct-I/O
  disable-after-error state, and range-cache arena/budget decisions.
- Oracle: current-C `cuda_model_stage_pool_alloc`, `cuda_model_stage_read`,
  `cuda_model_arena_alloc`, and `cuda_model_range_ptr_from_fd`, excluding
  source-page discard/progress side effects and compute-kernel consumption.
- Acceptance: a multi-chunk model range uses four CUDA-event-guarded pinned
  slots with observable reuse, admitted ranges share a bounded device arena,
  both raw and aligned new-arena admission gates select budget fallback when
  exhausted, and exact device readback passes. The direct-I/O error-disable
  errno policy must be tested without claiming a live B300 induced error.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json`.
- Comparator:
  `ds4-parity/check_model_async_staging_smoke.py --negative-test` plus
  executable Rust B300 smoke using the pinned GGUF.
- Evidence:
  - Added an opt-in Rust `AsyncPinnedRangeCache` owning a four-slot pinned
    stage ring; each slot is refilled only after synchronizing its recorded
    CUDA event, and intermediate upload errors drain previously enqueued
    copies before slot state is cleared.
  - On B300 pod `ds4-rust-port-b300`, seven direct-I/O chunks used four slots
    with two reuse waits, two cached ranges shared one 32,768-byte arena, and
    the next byte selected budget fallback after 28,672 admitted bytes; a
    separate boundary probe also rejected a new arena whose aligned
    reservation exceeded the remaining raw-byte budget. Both admitted range
    readbacks matched exactly.
  - The B300 feature test validates the current-C direct-I/O disable errno
    classes; the live smoke explicitly records that it did not induce a
    direct-read failure. Source-page discard/progress output, compute kernels,
    and runtime route activation remain unclaimed.
  - Validation passed through local workspace tests, formatter and diff
    checks, the 96-check comparator and retained M14 checks, B300
    feature-enabled crate tests plus predecessor direct-I/O smoke, and unified
    parity with 103 passed, 50 skipped, and 0 failed. Local feature compilation
    is unavailable without CUDA headers; B300 supplied that gate.
    Non-interactive Claude review was unavailable because the local CLI
    reported `Not logged in`; adversarial self-review found and fixed
    error-path draining and aligned new-arena budget admission defects, then
    found no remaining lifetime, policy, or bounded-claim defect.

####### M14.1b2c: Model Map Cache Closure

- Status: complete.
- Goal: close the model-map/range-cache assignment, including remaining
  source-page discard/progress policy and retained-current-C route evidence.
- Oracle: current-C `cuda_model_range_ptr`, `cuda_model_drop_file_pages`,
  `cuda_model_discard_source_pages`, `cuda_model_load_progress_note`, and
  `cuda_model_range_release_all`.
- Acceptance: an opt-in Rust cache must reuse an interior cached range with
  exact CUDA readback, issue Linux file/mapping discard advisory calls after
  staged chunks unless explicitly retained, emit the explicit non-TTY
  progress form unless disabled, and start new cache state cleanly after
  prior cache lifetime ends. DS4 kernel consumption and default-route
  activation remain rejected.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json`.
- Comparator:
  `ds4-parity/check_model_map_closure_smoke.py --negative-test` plus
  executable Rust B300 smoke using the pinned GGUF.
- Evidence:
  - Added explicit `ModelLoadProgressMode` and source-page retention policy
    to `AsyncPinnedRangeCache`, containing-range readback/reuse, and
    Linux `posix_fadvise`/`posix_madvise` call accounting.
  - On B300 pod `ds4-rust-port-b300`, an 8,192-byte admitted range served a
    257-byte interior readback exactly without another upload. Its two staged
    chunks issued two file discard calls totaling 8,192 bytes and two
    page-aligned mapping discard calls totaling 16,384 bytes, while a
    retained-pages cache suppressed both advisory classes.
  - The explicit non-TTY progress policy emitted the current-C initial
    message once for three progress notes; a disabled-progress cache emitted
    no message, and a new cache began with empty range/progress state.
    Physical page eviction, default runtime environment/TTY wiring, DS4
    kernels, and runtime route activation remain unclaimed.
  - Validation passed through local workspace tests, formatter and diff
    checks, the 84-check comparator and retained M14 checks, B300
    feature-enabled crate tests plus predecessor asynchronous-staging smoke,
    and unified parity with 104 passed, 50 skipped, and 0 failed. Local
    feature compilation requires CUDA headers unavailable on this host; B300
    supplied that gate. Non-interactive Claude review was unavailable because
    the local CLI reported `Not logged in`; adversarial self-review fixed a
    progress-threshold overflow edge before finding no remaining pointer,
    advisory-claim, progress, lifetime, or bounded-claim defect.

###### M14.1b3: Allocation And Quality Policy

- Status: split before implementation into M14.1b3a and M14.1b3b.
- Goal: port managed-KV selection, Q8/F16 range-cache policy, quality mode,
  and memory-report behavior without porting compute kernels.

####### M14.1b3a: Managed KV And Memory Report Policy

- Status: complete.
- Goal: own managed-tensor allocation proof, managed-KV selection policy, and
  CUDA memory-report formatting through the opt-in Rust substrate.
- Oracle: current-C `ds4_gpu_tensor_alloc_managed`,
  `cuda_managed_kv_reserve_bytes`, `ds4_gpu_should_use_managed_kv_cache`, and
  `ds4_gpu_print_memory_report`.
- Acceptance: Rust must reproduce empty/huge/small-context/query-failure and
  capacity-pressure managed-KV choices, query live device capacity through
  cuda-oxide, allocate managed storage on B300, and format the memory report
  without claiming Q8 cache, quality-mode, kernel, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b3a/allocation-policy-smoke.json`.
- Comparator:
  `ds4-parity/check_allocation_policy_smoke.py --negative-test` plus
  executable Rust B300 smoke using pinned cuda-oxide revision
  `0ec61156a7c5d65802402898b7a197bfff266d31`.
- Evidence:
  - Added the reusable `CudaContext::memory_info()` cuda-oxide API, validated
    by the full `cuda-core` B300 test suite, and consumed it through
    `CudaOxideSubstrate::memory_capacity`.
  - The Rust policy reproduces the current-C 8 GiB thresholds and clamped
    quarter-capacity reserve rule; deterministic cases covered empty KV,
    forced managed KV, query failure, sufficient capacity, reserve pressure,
    and context exceeding free device memory.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed;
    the live smoke queried valid free/total device capacity, allocated managed
    memory, and emitted the current-C-shaped memory-report prefix. Q8 cache,
    quality-mode, compute kernels, and default-route ownership remain false.
  - Validation passed through local workspace tests, formatter and diff
    checks, the 64-check comparator and retained M14 checks, full B300
    `cuda-core` tests, B300 feature-enabled `ds4-cuda` tests, and the retained
    model-map closure smoke. Unified parity passed with 105 passed, 50
    skipped, and 0 failed. Non-interactive Claude review timed out without a
    completed result; adversarial self-review found no threshold, reserve,
    transient-capacity, report-format, dependency-pin, or bounded-claim
    defect.

####### M14.1b3b: Q8 Cache And Quality Policy

- Status: complete.
- Goal: port Q8/F16 and Q8/F32 admission/failure-disable policy plus quality-mode
  BLAS math selection without promoting the runtime route or claiming the
  dequant kernels assigned to M14.3.
- Oracle: current-C `cuda_q8_f16_cache_reserve_bytes`,
  `cuda_q8_f16_cache_allowed`, `cuda_q8_f16_preload_allowed`,
  `cuda_q8_f16_cache_has_budget`,
  `cuda_q8_f16_cache_disable_after_failure`,
  `cuda_q8_f32_cache_allowed`, and `ds4_gpu_set_quality`.
- Acceptance: Rust reproduces Q8/F16 label, preload, reserve, budget, and
  failure-disable state policy; reproduces Q8/F32 optional-preload selection; and
  applies TF32 versus default cuBLAS math policy through cuda-oxide on B300.
  Converted Q8 device buffers and their failure-time synchronization/release,
  dequant kernels, DS4 compute kernels, and default-route activation remain
  rejected.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b3b/q8-quality-policy-smoke.json`.
- Comparator:
  `ds4-parity/check_q8_quality_policy_smoke.py --negative-test` plus
  executable Rust B300 smoke using pinned cuda-oxide revision
  `aabe10dc4fa0086375104458909e222d1ac1cfe3`.
- Evidence:
  - Added typed cuda-oxide `Blas::set_math_mode(BlasMathMode)` over the
    header-verified `cublasSetMathMode` ABI and validated both math
    selections with full B300 `cuda-core` tests.
  - The Rust Q8 host policy covers F16 eligibility, attention-output preload
    suppression, reserve and below-reserve rejection, notice-once/failure
    disable state, and F32 optional preload selection. Exact reserve
    equality remains admissible, matching current C.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests and
    `ds4-cuda-q8-quality-policy-smoke` passed. The smoke applied fast TF32,
    quality-mode default math, and `NO_TF32` default math through live
    cuBLAS. Converted buffers and their failure-time release, dequant kernels,
    compute kernels, and the default route remain unclaimed.
  - Validation passed through local workspace tests, formatter and diff
    checks, the 71-check comparator and retained M14 checks, B300
    `cublas-sys` and full `cuda-core` tests, B300 feature-enabled `ds4-cuda`
    tests, and unified parity with 106 passed, 50 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review corrected the reserve-equality boundary test and
    narrowed failure ownership to disable-state policy before finding no
    remaining policy, ABI, or bounded-claim defect.

###### M14.1b4: Fill Kernel And Command Lifetime

- Status: complete.
- Goal: prove `ds4_gpu_tensor_fill_f32` and current-C command synchronization
  semantics through an opt-in executable-local Rust CUDA kernel on B300.
- Oracle: current-C `ds4_gpu_tensor_fill_f32`, `fill_f32_kernel`,
  `ds4_gpu_flush_commands`, `ds4_gpu_end_commands`, and
  `ds4_gpu_synchronize`.
- Fixture:
  `ds4-parity/baselines/backend/m14.1b4/fill-command-lifetime-smoke.json`.
- Comparator: `ds4-parity/check_fill_command_lifetime_smoke.py
  --negative-test` plus executable Rust B300 kernel smoke.
- Acceptance: the opt-in executable compiles and launches Rust `fill_f32`,
  preserves prefix/zero-count/bounds behavior, and exposes context-wide
  completion wrappers; dequant kernels, graph kernels, runtime graph
  integration, and the default route remain unclaimed.
- Evidence:
  - Added the `cuda-oxide-kernels` feature and executable-local
    `ds4-cuda-fill-lifetime-smoke` with a Rust `#[kernel] fill_f32` using the
    current-C 256-thread launch shape and explicit count boundary.
  - Pushed cuda-oxide tool fix
    `981e3244a107d84d807cfb087793269c477cc764`: `cargo oxide run` no longer
    raises a portable basic kernel from backend-selected `sm_80` to local
    `sm_103`, which had emitted invalid `.version 6.0 / .target sm_103` PTX
    and failed B300 JIT loading with CUDA error 218.
  - On B300 pod `ds4-rust-port-b300`, `cargo test -p cargo-oxide` passed and
    the kernel smoke executed on `NVIDIA B300 SXM6 AC` with backend-selected
    `sm_80`, proving prefix fill, negative-infinity fill, zero-count no-op,
    bounds rejection, current-C's no-op begin command, and context-wide
    flush/end/synchronize behavior.
  - This is an executable-local kernel proof. Library embedded-module
    retention, dequant kernels, graph compute kernels, runtime graph
    integration, and default-route ownership remain unclaimed.
  - Validation passed through local workspace tests, formatter and diff
    checks, the fill/command-lifetime comparator and retained M14 checks, B300
    feature-enabled `ds4-cuda` tests, B300 `cargo-oxide` tests and kernel
    execution, and unified parity with 107 passed, 50 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review verified the executable-local ownership boundary,
    prefix-count behavior, and context-wide synchronization mapping before
    finding no remaining bounded-claim defect.

##### M14.1c: Substrate Route Closure Gate

- Status: done.
- Goal: close resource ownership and expose the Rust CUDA substrate only to
  the following kernel stages, without promoting the default runtime route.
- Oracle: M14.1a and M14.1b artifacts plus M14.0 claim policy.
- Fixture:
  `ds4-parity/baselines/backend/m14.1c/substrate-route-closure.json`.
- Comparator: `ds4-parity/check_substrate_route_closure.py --negative-test`
  plus the retained B300 feature-test and fill-kernel rerun contract.
- Acceptance: M14.2 may consume the Rust substrate; CUDA C removal and
  default-route promotion remain rejected.
- Drift policy: retained current-C CUDA oracles remain required through M14.6.
- Evidence:
  - Corrected the M14.0 ownership inventory after M14.1b3b established that
    `ds4_gpu_cache_q8_f16_range`, `dequant_q8_0_to_f16_kernel`, and
    `dequant_q8_0_to_f32_kernel` remain M14.3 work, while M14.1 owns only
    `fill_f32_kernel` among CUDA kernels.
  - Added the explicit Rust no-op `begin_commands` facade and live fill smoke
    invocation so the current-C command surface is complete before M14.2
    consumes the substrate.
  - Route promotion and `ds4_cuda.cu` removal remain rejected; all M14.1
    behavior remains opt-in and current C remains the oracle.
  - Validation passed through local formatting, diff, and workspace tests;
    the updated 81-check M14.1b4 comparator; the 139-check closure
    comparator; B300 feature-enabled `ds4-cuda` tests with 21 passing tests;
    live B300 cargo-oxide `fill_f32` execution including
    `begin_is_noop:true`; and unified parity with 108 passed, 50 skipped, and
    0 failed.
  - Non-interactive Claude review produced no completed result before its
    timeout; adversarial self-review confirmed that the reassigned Q8/dequant
    symbols remain M14.3-only, the begin command maps only the current-C
    no-op, and closure makes no route or removal claim.

#### M14.2: Embedding Indexer And Elementwise Kernels

- Status: done through M14.2e; M14.2b
  split into M14.2b1 and M14.2b2 after B300 exposed an independent
  libdevice/NVVM blocker for SwiGLU; M14.2d split into M14.2d1 and M14.2d2
  because scalar selection and optimized dispatch have distinct ownership
  claims; M14.2d2 split into M14.2d2a through M14.2d2c because direct-one
  warp reduction, tensor-core scoring, and specialized top-k selection use
  separate cuda-oxide surfaces; M14.2d2b split into M14.2d2b1 and
  M14.2d2b2 because cuda-oxide's `m16n8k16` surface proves the base
  16-component tile independently from the widened multi-warp dispatch;
  M14.2d2b2 split into M14.2d2b2a through M14.2d2b2c because the 32,
  64, and 128-component branches and final priority wiring need separate
  multi-warp evidence; M14.2d2c split into M14.2d2c1 through M14.2d2c5
  because each specialized top-k family has an independent CUDA launch and
  storage contract.
- Goal: port the M14.2 operation family through bounded Rust CUDA kernel
  slices while retaining current-C oracles and the opt-in-only route.
- Stage split:
  - M14.2a: Add And Repeat Elementwise Kernels.
  - M14.2b1: Directional Steering Projection Kernel.
  - M14.2b2: SwiGLU Libdevice Path.
  - M14.2c: Embedding Kernel Pair.
  - M14.2d1: Scalar Indexer Selection Kernels.
  - M14.2d2a: Direct-One Indexer Score Kernel.
  - M14.2d2b1: Base Tensor-Core Indexer Score Kernel.
  - M14.2d2b2a: WMMA32 Tensor-Core Indexer Score Kernel.
  - M14.2d2b2b: WMMA64 Tensor-Core Indexer Score Kernel.
  - M14.2d2b2c: WMMA128 Tensor-Core Indexer Score Kernel And Dispatch Priority.
  - M14.2d2c1: 1024 Bitonic Top-K Kernel.
  - M14.2d2c2: Power-Of-Two Top-K Kernels.
  - M14.2d2c3: CUB-Or-Equivalent Top-K Branch.
  - M14.2d2c4: Chunked And Tree-Merge Top-K Kernels.
  - M14.2d2c5: Indexed Ascending Top-K Sort And Dispatch Policy.
  - M14.2e: M14.2 Kernel Closure Gate.

##### M14.2a: Add And Repeat Elementwise Kernels

- Status: done.
- Goal: port the bounded f32 `add_kernel` and `repeat_hc_kernel` operations
  through executable-local Rust cuda-oxide kernels before the more complex
  M14.2 families.
- Oracle: current-C `add_kernel`, `ds4_gpu_add_tensor`, `repeat_hc_kernel`,
  and `ds4_gpu_repeat_hc_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.2a/elementwise-kernel-smoke.json`.
- Comparator: `ds4-parity/check_elementwise_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust kernels match bounded add and repeated-row outputs and
  reject invalid host-side arguments; embedding, indexer/top-k, SwiGLU,
  directional steering, route activation, and C CUDA removal remain
  unclaimed.
- Evidence:
  - Added executable-local Rust `add_kernel` and `repeat_hc_kernel` kernels
    with current-C-shaped 256-thread launch geometry and safe disjoint output
    writes.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 22 tests and cargo-oxide executed both kernels on
    `NVIDIA B300 SXM6 AC` using portable `sm_80`, proving add output,
    repeated-HC-row output, add bounds rejection, and repeat-shape rejection.
  - Local formatter, diff, and workspace tests passed; the 69-check
    comparator passed and unified parity passed with 109 passed, 50 skipped,
    and 0 failed.
  - Non-interactive Claude review timed out without a completed result;
    adversarial self-review corrected an initial `repeat_hc` wrapper narrowing
    by preserving current-C's 64-bit shape product before B300 execution and
    found no remaining defect within this bounded claim.
  - This stage remains opt-in; it does not claim any model-backed embedding,
    selection, nonlinear/reduction, route, or removal ownership.

##### M14.2b1: Directional Steering Projection Kernel

- Status: done.
- Goal: port current-C's in-place directional projection and shared-memory
  row reduction before introducing model-backed embedding or indexer/top-k.
- Oracle: current-C `directional_steering_project_kernel` and
  `ds4_gpu_directional_steering_project_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.2b1/directional-steering-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_directional_steering_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns only bounded directional projection; SwiGLU,
  model-backed families, selection families, route activation, and removal
  remain pending.
- Evidence:
  - Added executable-local Rust `directional_steering_project_kernel` with
    one block per row, `SharedArray<f32, 256>` reduction storage,
    `thread::sync_threads()` barriers, and in-place row updates matching the
    current-C operation shape.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 23 tests and cargo-oxide executed the directional projection on
    `NVIDIA B300 SXM6 AC` using portable `sm_80`, proving projected output
    and invalid-shape rejection.
  - Local formatter, diff, and workspace tests passed; the 71-check
    directional comparator passed and unified parity passed with 110 passed,
    50 skipped, and 0 failed.
  - Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained the in-place row-ownership and
    synchronization proof and the explicit unowned SwiGLU blocker.
  - A combined SwiGLU attempt first exposed unsupported device `f32::min`/
    `f32::max` lowering and, after finite clamp comparisons removed that
    issue, exposed the blocking path: `f32::exp()` emits `__nv_expf`, then
    CUDA 13.2 `libnvvm` rejects cuda-oxide's opaque-pointer NVVM IR with
    `parse expected type`. SwiGLU ownership is not claimed by this stage.

##### M14.2b2: SwiGLU Libdevice Path

- Status: done.
- Goal: repair cuda-oxide's blocked executable libdevice path and port
  current-C `swiglu_kernel` and `ds4_gpu_swiglu_tensor`.
- Oracle: current-C `swiglu_kernel` clamp, SiLU exponential, output weighting,
  and `ds4_gpu_swiglu_tensor` argument/launch contract.
- Fixture:
  `ds4-parity/baselines/backend/m14.2b2/swiglu-kernel-smoke.json`.
- Comparator: `ds4-parity/check_swiglu_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Acceptance: Rust owns bounded SwiGLU and retains the previously established
  directional projection ownership; embedding, indexer/top-k, route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Pushed cuda-oxide revision
    `d4791b7002152af3b7f6b15a48d7f5acd7a63011`, which emits portable PTX
    for device `__nv_*` calls, links NVIDIA libdevice into a cubin targeted
    to the executing CUDA context, and keys linked artifacts by architecture.
  - Added executable-local Rust `swiglu_kernel` with current-C-shaped finite
    and NaN clamps, unclamped behavior, SiLU exponential, weighting,
    256-thread geometry, and host buffer validation. It uses
    `cuda_host::ltoir` to load the libdevice-linked PTX while retaining typed
    kernel launches.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 24 tests. Live cargo-oxide execution emitted portable `sm_80` PTX,
    generated `ds4_cuda_swiglu_smoke.sm_103.cubin`, and proved clamped,
    unclamped, and invalid-shape behavior on `NVIDIA B300 SXM6 AC` without
    target overrides.
  - Local workspace tests, formatter/diff checks, the 73-check SwiGLU
    comparator, and unified parity passed with 116 passed, 45 skipped, and
    0 failed. Non-interactive Claude review timed out without a completed
    result; adversarial self-review found and fixed NaN clamp handling by
    replacing optimized-away float comparisons with explicit IEEE-754 bit
    classification before the recorded B300 pass.
  - This stage remains opt-in; it does not claim embedding/model-range,
    indexer/top-k, runtime route, or C CUDA removal ownership.

##### M14.2c: Embedding Kernel Pair

- Status: done.
- Goal: port current-C's FP16 single-token and batched embedding loads through
  executable-local Rust cuda-oxide kernels without coupling the proof to
  model-range routing.
- Oracle: current-C `embed_token_hc_kernel`, `embed_tokens_hc_kernel`,
  `ds4_gpu_embed_token_hc_tensor`, and `ds4_gpu_embed_tokens_hc_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.2c/embedding-kernel-smoke.json`.
- Comparator: `ds4-parity/check_embedding_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns the bounded FP16 embedding kernel pair and host-side
  shape safety; model-range consumption, indexer/top-k, route activation, and
  C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `embed_token_hc_kernel` and
    `embed_tokens_hc_kernel` using primitive `f16` loads widened to `f32`,
    256-thread launch geometry, repeated hidden-copy rows, and the current-C
    batch rule mapping negative or out-of-vocabulary tokens to row zero.
  - The single-token Rust helper rejects an out-of-vocabulary token instead
    of admitting current-C's unchecked device row read; this strengthens
    invalid-input safety without changing valid-call output.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 25 tests and live cargo-oxide execution emitted portable `sm_80`
    PTX and proved single-token output, batched fallback output, and bounds
    rejection on `NVIDIA B300 SXM6 AC`.
  - Local workspace tests and formatter/diff checks passed; the 69-check
    embedding comparator and unified parity passed with 117 passed, 45
    skipped, and 0 failed. A first device build corrected the test fixture
    from a `half`-crate constructor to primitive `f16::from_bits` values
    before the recorded B300 pass.
  - Non-interactive Claude review produced no completed result before its
    timeout; adversarial self-review confirmed that valid-call outputs and
    batch fallback match current C, and that the stricter single-token
    invalid-input rejection is recorded rather than overclaimed as parity.
  - This stage remains opt-in; it does not claim model-range consumption,
    indexer/top-k, runtime route, or C CUDA removal ownership.

##### M14.2d1: Scalar Indexer Selection Kernels

- Status: done.
- Goal: port the scalar fallback indexer scoring, top-k selection, and mask
  operations before claiming direct/WMMA or specialized top-k dispatch.
- Oracle: current-C `indexer_scores_kernel`, `indexer_topk_kernel`,
  `topk_mask_kernel`, and their fallback launch sites.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d1/indexer-scalar-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_scalar_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Acceptance: Rust owns only bounded scalar indexer score, scalar fallback
  top-k, and top-k mask kernels; direct-one/WMMA scoring, specialized
  power-of-two/CUB/chunked top-k dispatch, route activation, and C CUDA
  removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_scores_kernel`,
    `indexer_topk_kernel`, and `topk_mask_kernel` with current-C-shaped
    positive-score reduction, causal negative-infinity masking, stable
    earlier-index top-k ties, and selected-row mask semantics.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 26 tests and live cargo-oxide execution emitted portable `sm_80`
    PTX and proved noncausal scoring, causal masking, top-k output, tie
    ordering, mask output, and invalid-shape rejection on
    `NVIDIA B300 SXM6 AC`.
  - Local workspace tests, formatter/diff checks, the 73-check scalar
    indexer comparator, and unified parity passed with 118 passed, 45
    skipped, and 0 failed. Non-interactive Claude review produced no
    completed result before its timeout; adversarial self-review confirmed
    `fmaxf` NaN/negative handling, stable scalar top-k ties, and the bounded
    optimized-dispatch non-claim, and aligned mask launch sizing with the C
    wrapper's maximum-work calculation before the final B300 rerun.
  - This stage remains opt-in; it does not claim optimized indexer/top-k
    dispatch, runtime route, or C CUDA removal ownership.

##### M14.2d2: Optimized Indexer And Top-K Dispatch

- Status: split before implementation into M14.2d2a through M14.2d2c.
- Goal: port or explicitly close ownership of current-C direct/WMMA score
  selection and specialized top-k dispatch after scalar fallback is proven.
- Oracle: current-C direct-one and WMMA score kernels plus specialized
  power-of-two, CUB, chunked, merged, and tree top-k launch branches.

##### M14.2d2a: Direct-One Indexer Score Kernel

- Status: done.
- Goal: port current-C's fixed-shape direct-one score kernel with the same
  four-warp shuffle reduction before tensor-core or specialized top-k work.
- Oracle: current-C `indexer_score_one_direct_kernel` and the
  `n_tokens == 1`, `head_dim == 128`, `n_head == 64` launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2a/indexer-direct-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_direct_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Acceptance: Rust owns only the direct-one fixed-shape kernel and its
  host-side bounds validation; WMMA score branches, specialized top-k
  dispatch, route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_score_one_direct_kernel` with
    current-C-shaped 128-thread geometry, four-warps-per-head-group
    reduction, `warp::shuffle_down_f32` accumulation, positive-score
    weighting, and causal negative-infinity masking.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 27 tests and live cargo-oxide execution lowered the warp shuffle,
    emitted portable `sm_80` PTX, and proved direct output, causal masking,
    NaN/negative clamp behavior, and invalid-shape rejection on
    `NVIDIA B300 SXM6 AC`.
  - Local workspace tests, formatter/diff checks, the 66-check direct
    indexer comparator, and unified parity passed with 119 passed, 45
    skipped, and 0 failed. Non-interactive Claude review produced no
    completed result before its timeout; adversarial self-review confirmed
    lane/head grouping, shuffle-down reduction, explicit NaN clamp handling,
    and the WMMA/top-k non-claims.
  - This stage remains opt-in; it does not claim WMMA scoring, specialized
    top-k dispatch, runtime route, or C CUDA removal ownership.

##### M14.2d2b: Tensor-Core Indexer Score Kernels

- Status: done through M14.2d2b1 and M14.2d2b2a through M14.2d2b2c.
- Goal: port the 16/32/64/128-component WMMA score branches through
  cuda-oxide's warp-scoped MMA surface.

##### M14.2d2b1: Base Tensor-Core Indexer Score Kernel

- Status: done.
- Goal: port current-C's 16-component `indexer_scores_wmma_kernel` through
  cuda-oxide's `m16n8k16` warp-scoped MMA surface before widened dispatch.
- Oracle: current-C `indexer_scores_wmma_kernel` and the final
  `indexer_scores_wmma_kernel<<<grid, 32>>>` branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b1/indexer-wmma-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Acceptance: Rust owns only the base 16-component WMMA score tile and
  host-side bounds validation; widened 32/64/128-component dispatch,
  specialized top-k, route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_scores_wmma_kernel` using native
    `f16` shared staging and two cuda-oxide `mma_m16n8k16_f32_f16` calls to
    cover current-C's `16 x 16` output tile, followed by positive-score
    weighting, scaling, and causal negative-infinity masking.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 28 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved base WMMA output, both eight-column MMA halves,
    per-token weighting, NaN/negative suppression, causal masking, and
    invalid-shape rejection on `NVIDIA B300 SXM6 AC`.
  - A first device compile identified unsupported device drop glue induced
    by generic `u32::min`; replacing it with explicit scalar comparisons
    kept the semantics and allowed PTX emission and execution.
  - Local workspace tests, formatter/diff checks, the 72-check base WMMA
    comparator, and unified parity passed with 120 passed, 45 skipped, and
    0 failed. Non-interactive Claude review produced no completed result
    before its timeout; adversarial self-review extended the live fixture to
    prove weighted output and NaN/negative suppression before the final B300
    run.
  - This stage remains opt-in; it does not claim widened WMMA scoring,
    specialized top-k dispatch, runtime route, or C CUDA removal ownership.

##### M14.2d2b2: Widened Tensor-Core Indexer Score Dispatch

- Status: done through M14.2d2b2a through M14.2d2b2c.
- Goal: port current-C's 32/64/128-component multi-warp WMMA branches and
  their dispatch priority after the base tile is proven.

##### M14.2d2b2a: WMMA32 Tensor-Core Indexer Score Kernel

- Status: done.
- Goal: port current-C's 32-component two-warp WMMA score branch without
  claiming the larger multi-warp branches or dispatch priority.
- Oracle: current-C `indexer_scores_wmma32_kernel` and its
  `indexer_scores_wmma32_kernel<<<grid, 64>>>` launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b2a/indexer-wmma32-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma32_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Acceptance: Rust owns only the 32-component two-warp WMMA score tile and
  host-side bounds validation; WMMA64/WMMA128 dispatch, specialized top-k,
  route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_scores_wmma32_kernel` using two
    warps, native `f16` shared staging, and two cuda-oxide
    `mma_m16n8k16_f32_f16` accumulators per warp to cover current-C's
    `16 x 32` output tile.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 29 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved WMMA32 output across two 32-component blocks,
    two-warp tile mapping, per-token weighting, NaN/negative suppression,
    causal masking, and invalid-shape rejection on `NVIDIA B300 SXM6 AC`.
  - Local workspace tests, formatter/diff checks, the 73-check WMMA32
    comparator, and unified parity passed with 121 passed, 45 skipped, and
    0 failed. Non-interactive Claude review produced no completed result
    before its timeout; adversarial self-review compared the two-warp
    staging, accumulator scatter, causal early exit, and explicit `fmaxf`
    equivalent against current C.
  - This stage remains opt-in; it does not claim WMMA64/WMMA128 score
    dispatch, specialized top-k dispatch, runtime route, or C CUDA removal
    ownership.

##### M14.2d2b2b: WMMA64 Tensor-Core Indexer Score Kernel

- Status: done.
- Goal: port current-C's 64-component four-warp WMMA score branch after the
  two-warp mapping is proven.
- Oracle: current-C `indexer_scores_wmma64_kernel` and its
  `indexer_scores_wmma64_kernel<<<grid, 128>>>` launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b2b/indexer-wmma64-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma64_kernel_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Acceptance: Rust owns only the 64-component four-warp WMMA score tile and
  host-side bounds validation; WMMA128 priority, specialized top-k, route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_scores_wmma64_kernel` using four
    warps, native `f16` shared staging, and two cuda-oxide
    `mma_m16n8k16_f32_f16` accumulators per warp to cover current-C's
    `16 x 64` output tile.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 30 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved WMMA64 output across two 64-component blocks,
    four-warp tile mapping, per-token weighting, NaN/negative suppression,
    causal masking, and invalid-shape rejection on `NVIDIA B300 SXM6 AC`.
  - Local workspace tests, formatter/diff checks, the 73-check WMMA64
    comparator, and unified parity passed with 122 passed, 45 skipped, and
    0 failed. Non-interactive Claude review produced no completed result
    before its timeout; adversarial self-review compared four-warp column
    ownership, accumulator scatter, causal early exit, and explicit `fmaxf`
    semantics against current C.
  - This stage remains opt-in; it does not claim WMMA128 priority,
    specialized top-k dispatch, runtime route, or C CUDA removal ownership.

##### M14.2d2b2c: WMMA128 Tensor-Core Indexer Score Kernel And Dispatch Priority

- Status: done.
- Goal: port current-C's 128-component eight-warp WMMA score branch and close
  the 128/64/32/base priority selection contract.
- Oracle: current-C `indexer_scores_wmma128_kernel`, its
  `indexer_scores_wmma128_kernel<<<grid, 256>>>` launch branch, and the
  `indexer_scores_launch` direct-one/WMMA/scalar priority chain.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2b2c/indexer-wmma128-dispatch-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_wmma128_dispatch_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Acceptance: Rust owns the 128-component eight-warp score tile and the
  validated-input score-kernel selection policy; specialized top-k, runtime
  route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_scores_wmma128_kernel` using eight
    warps, native `f16` shared staging, and two cuda-oxide
    `mma_m16n8k16_f32_f16` accumulators per warp to cover current-C's
    `16 x 128` output tile.
  - Added `select_indexer_score_kernel`, an opt-in policy projection of the
    current-C validated-input launch chain: direct-one first, then
    WMMA128/64/32/base under quality and disable gates, otherwise scalar.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 32 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved WMMA128 output across two 128-component blocks,
    eight-warp tile mapping, per-token weighting, NaN/negative suppression,
    causal masking, invalid-shape rejection, and score dispatch ordering on
    `NVIDIA B300 SXM6 AC`.
  - Local workspace tests, formatter/diff checks, the 84-check
    WMMA128/dispatch comparator, and unified parity passed with 123 passed,
    45 skipped, and 0 failed. Non-interactive Claude review produced no
    completed result before its timeout; adversarial self-review added
    direct-one-disabled and global-WMMA-disabled selector checks before the
    final B300 rerun.
  - This stage remains opt-in; it does not claim specialized top-k dispatch,
    runtime route activation, or C CUDA removal ownership.

##### M14.2d2c: Specialized Top-K Kernels

- Status: split before implementation into M14.2d2c1 through M14.2d2c5.
- Goal: port the shared-memory, CUB-equivalent, chunked, merge, tree-merge,
  and indexed ascending-sort top-k kernels.

##### M14.2d2c1: 1024 Bitonic Top-K Kernel

- Status: done.
- Goal: port current-C's `top_k == 512 && n_comp <= 1024` shared-memory
  bitonic fast path without claiming larger selection branches.
- Oracle: current-C `indexer_topk_1024_kernel` and its guarded
  `ds4_gpu_indexer_topk_tensor` launch branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c1/indexer-topk1024-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk1024_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Acceptance: Rust owns only executable-local 1024-element top-k sorting and
  host-side shape safety; larger top-k dispatch, indexed ascending sort,
  route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_topk_1024_kernel` with the current-C
    1024-thread shared-memory bitonic network, descending score order, and
    lower-index tie break.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 33 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved full-width output, partial-width sentinel
    exclusion, tie ordering, and invalid-shape rejection on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 69-check comparator, and
    unified parity passed with 124 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review confirmed that the Rust scope owns only the
    guarded kernel shape and current-C ordering, not dispatch selection.
  - This stage remains opt-in; it does not claim larger top-k branches,
    indexed ascending sort, runtime route activation, or C CUDA removal.

##### M14.2d2c2: Power-Of-Two Top-K Kernels

- Status: done.
- Goal: port the 2048/4096 and 8192 shared-memory power-of-two selection
  branches after the 1024 shape is proven.
- Oracle: current-C `indexer_topk_pow2_kernel<2048>`,
  `indexer_topk_pow2_kernel<4096>`, and fallback
  `indexer_topk_pow2_u16_kernel<8192>` implementations.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c2/indexer-topk-pow2-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_pow2_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Acceptance: Rust owns direct execution of the power-of-two kernels and
  their ordering/storage semantics; CUB selection, chunked merging, indexed
  ascending sort, route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexer_topk_pow2_2048_kernel`,
    `indexer_topk_pow2_4096_kernel`, and
    `indexer_topk_pow2_u16_8192_kernel`, preserving current-C shared-memory
    index width and lower-index tie order.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 34 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved output and sentinel exclusion for all three
    kernels on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 76-check comparator, and
    unified parity passed with 125 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained the explicit non-claim for CUB branch
    selection.
  - This stage does not claim that the 8192 fallback wins current-C's CUB
    selection on B300; CUB policy is isolated in M14.2d2c3.

##### M14.2d2c3: CUB-Or-Equivalent Top-K Branch

- Status: done.
- Goal: port or explicitly close current-C's dynamic-shared-memory CUB
  selection optimization with equivalent validated ordering.
- Oracle: current-C `topk_float_ordered_key`, `topk_pack_key`, and
  `indexer_topk_8192_cub_kernel` plus its dynamic-shared-memory opt-in
  launch prerequisite.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c3/indexer-topk-packed-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_packed_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns an executable packed-key top-k equivalent and its
  65,536-byte dynamic-shared-memory setup; CUB internals, branch selection,
  chunk/tree merge, indexed ascending sort, route activation, and C CUDA
  removal remain pending.
- Evidence:
  - Added executable-local Rust
    `indexer_topk_8192_packed_key_equivalent_kernel`, preserving current-C's
    ordered float-bit key, lower-index tie key, and negative-infinity
    sentinel behavior through a dynamic-shared-memory bitonic equivalent.
  - The initial B300 launch failed with `DriverError(1, "invalid argument")`
    because current-C's `cudaFuncSetAttribute` prerequisite was missing from
    the Rust host layer. Pinned cuda-oxide revision
    `e9c0d677104751179985098f02212ff044d3ec22` adds
    `CudaFunction::set_max_dynamic_shared_memory_size`.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 35 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved 4096- and 6000-component output, positive-NaN
    ordering, tie order, sentinel exclusion, dynamic shared-memory setup,
    and invalid-shape rejection on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 80-check comparator, and
    unified parity passed with 126 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained the explicit CUB implementation and
    dispatch-selection non-claims.
  - This stage remains opt-in; it does not claim CUB library implementation
    or specialized top-k dispatch selection.

##### M14.2d2c4: Chunked And Tree-Merge Top-K Kernels

- Status: done.
- Goal: port the large-component chunk candidate, intermediate tree merge,
  and final merge kernels plus their scratch layout.
- Oracle: current-C `indexer_topk_chunk_pow2_kernel<4096>`,
  `indexer_topk_tree_merge_pow2_kernel<4096>`,
  `indexer_topk_merge_pow2_kernel<4096>`, and its
  `DS4_CUDA_TOPK_MERGE_GROUP` scratch allocation loop.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c4/indexer-topk-tree-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_tree_kernel_smoke.py --negative-test` plus
  live B300 cargo-oxide execution.
- Acceptance: Rust owns direct chunk/tree/final merge execution and
  contiguous scratch-level calculations; indexed ascending sort, specialized
  dispatch policy, route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust 4096-element chunk, intermediate tree-merge,
    and final-merge kernels using current-C descending score and lower-index
    tie semantics.
  - The host smoke represents current-C's one scratch allocation with
    explicit non-overlapping offsets: for two tokens and ten chunks it uses
    levels `{offset: 0, stride: 5120}` and
    `{offset: 10240, stride: 1024}` for 12,288 elements total.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 36 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved chunk, tree, and final outputs, token-stride
    isolation, partial final-chunk sentinel exclusion, and invalid-shape
    rejection on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 81-check comparator, and
    unified parity passed with 127 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained the dispatch and indexed-sort
    non-claims.

##### M14.2d2c5: Indexed Ascending Top-K Sort And Dispatch Policy

- Status: done.
- Goal: port indexed attention's ascending 512-element sort and close the
  validated-input specialized top-k dispatch ordering.
- Oracle: current-C `indexed_topk_sort_512_asc_kernel`, its indexed-attention
  multi-token gate, and `ds4_gpu_indexer_topk_tensor` branch order.
- Fixture:
  `ds4-parity/baselines/backend/m14.2d2c5/indexer-topk-dispatch-smoke.json`.
- Comparator:
  `ds4-parity/check_indexer_topk_dispatch_smoke.py --negative-test` plus live
  B300 cargo-oxide execution.
- Acceptance: Rust owns the opt-in ascending index sort kernel and
  validated-input specialized top-k branch policy, selecting its proven
  packed-key equivalent rather than claiming CUB implementation; runtime route
  activation and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `indexed_topk_sort_512_asc_kernel` mirroring
    current-C's 512-thread ascending bitonic network and the multi-token sort
    gate.
  - Added `select_indexer_topk_kernel`, preserving current-C's
    1024/2048/4096/8192/chunk/tree/scalar ordering and mapping the
    capability-gated CUB path to the validated packed-key equivalent.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
    passed with 38 tests. Live cargo-oxide execution emitted portable
    `sm_80` PTX and proved two ascending rows, sort-gate behavior, packed
    equivalent selection, fallback ordering, and invalid-shape rejection on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 81-check comparator, and
    unified parity passed with 128 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review added a direct `DS4_CUDA_NO_TOPK8192`
    fall-through assertion before the final B300 rerun.

##### M14.2e: M14.2 Kernel Closure Gate

- Status: done.
- Goal: close the M14.2 kernel ownership ledger without claiming runtime
  route activation or C CUDA removal.
- Oracle: M14.0 inventory, all M14.2 stage artifacts, and current-C
  routed-MoE ownership of `zero_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.2e/kernel-ownership-closure.json`.
- Comparator: `ds4-parity/check_m14_2_kernel_closure.py --negative-test`
  plus the retained M14.2d2c5 B300 rerun contract.
- Acceptance: M14.3 may consume the Rust-owned opt-in kernel family;
  default-route promotion and `ds4_cuda.cu` removal remain rejected.
- Evidence:
  - Aggregated all fifteen validated M14.2 B300 artifacts into one closure
    ledger while preserving each stage's opt-in and retained-current-C
    constraints.
  - Corrected the ownership inventory by assigning `zero_kernel` to M14.5,
    because current C invokes it only inside routed MoE atomic-down
    execution.
  - Recorded that Rust's packed-key-equivalent top-k branch preserves the
    current-C selection semantics without claiming the CUB implementation.
  - Passed local formatting, diff, workspace tests, the 156-check closure
    comparator, and unified parity with 129 passed, 45 skipped, and 0 failed.
    A non-interactive Claude review attempt timed out without a completed
    result; adversarial self-review found the corrected `zero_kernel`
    assignment before closure.

#### M14.3: Dense Projection Quantization And Norm Kernels

- Status: done through M14.3d4.
- Goal: port the M14.3 dense projection, Q8 conversion, and normalization
  operation family through bounded Rust CUDA slices.

##### M14.3a: Plain And Weighted RMS Norm Kernels

- Status: done.
- Goal: prove the standalone plain and weighted RMS normalization row
  reductions through opt-in Rust CUDA kernels.
- Oracle: current-C `rms_norm_plain_kernel`, `rms_norm_weight_kernel`, and
  their one-row and multi-row tensor launch surfaces.
- Fixture:
  `ds4-parity/baselines/backend/m14.3a/rms-norm-kernel-smoke.json`.
- Comparator: `ds4-parity/check_rms_norm_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: fused QKV/head normalization, projection, Q8 kernels, runtime
  route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust kernels using current-C's 256-thread
    per-row shared-memory reduction and a libdevice-linked reciprocal RMS
    scale for plain and weighted output.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 39 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    and proved multi-row plain, multi-row weighted, single-row, and
    invalid-shape behavior on `NVIDIA B300 SXM6 AC`.
  - Passed local formatting, diff, workspace tests, the 72-check comparator,
    and unified parity with 130 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result; the
    parity run caught and corrected the M14.2e active-stage wiring before
    commit.

##### M14.3b: Fused QKV And Head RMS Norm Kernels

- Status: done through M14.3b1 and M14.3b2, which separate fused QKV/basic
  head RMS normalization from head RMS plus YARN/RoPE tail math.
- Goal: port fused QKV and head RMS normalization behavior without claiming
  the remaining projection or Q8 operation family.

##### M14.3b1: Fused QKV And Basic Head RMS Norm Kernels

- Status: done.
- Goal: prove fused Q/KV weighted RMS normalization and basic in-place head
  RMS normalization through opt-in Rust CUDA kernels.
- Oracle: current-C `dsv4_qkv_rms_norm_rows_kernel`,
  `head_rms_norm_kernel`, and their tensor launch surfaces.
- Fixture:
  `ds4-parity/baselines/backend/m14.3b1/fused-rms-norm-kernel-smoke.json`.
- Comparator: `ds4-parity/check_fused_rms_norm_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: head RMS plus RoPE-tail fusion, the fused-QKV fallback policy,
  projection, Q8 kernels, runtime route activation, and C CUDA removal remain
  pending.
- Evidence:
  - Added executable-local Rust fused QKV and head RMS kernels using the
    current-C row reduction and libdevice-linked reciprocal scale.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 40 tests. Live cargo-oxide execution emitted portable `sm_80` PTX,
    handled asymmetric Q and KV widths, and proved in-place head
    normalization on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 73-check comparator, and
    unified parity passed with 131 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result; live
    compilation found and corrected the in-place `DisjointSlice` read through
    a mutable device pointer before the successful B300 rerun.

##### M14.3b2: Head RMS Norm Rope Tail Kernel

- Status: done.
- Goal: port the combined head RMS normalization and YARN/RoPE tail kernel
  without claiming remaining projection or Q8 ownership.
- Oracle: current-C `head_rms_norm_rope_tail_kernel`,
  `rope_yarn_ramp_dev`, and `ds4_gpu_head_rms_norm_rope_tail_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3b2/head-rms-rope-tail-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_head_rms_rope_tail_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: standalone RoPE, dense projection, Q8 kernels, runtime route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `head_rms_norm_rope_tail_kernel` with
    current-C per-head reduction, tail-only rotary math, YARN ramp mixing,
    and inverse sign handling.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 41 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and proved interpolated, YARN forward, inverse, and
    invalid-shape behavior on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 74-check comparator, and
    unified parity passed with 132 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained standalone-RoPE, projection, Q8, and
    route non-claims.

##### M14.3c: Dense F16 And F32 Projection Kernels

- Status: split into M14.3c1 base reductions, M14.3c2 ordered, paired, and
  serial F16 kernels, and M14.3c3 BLAS dispatch and activation conversion
  before Q8 ownership is claimed.
- Goal: port dense F16/F32 projection execution as a separately comparable
  slice before claiming Q8 conversion or quantized matmul ownership.

##### M14.3c1: Base F16 And F32 Projection Kernels

- Status: done.
- Goal: prove direct base-reduction projection kernels with primitive F16
  weight loads and F32 weights without claiming wrapper dispatch policy.
- Oracle: current-C `matmul_f16_kernel` and `matmul_f32_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3c1/dense-projection-kernel-smoke.json`.
- Comparator: `ds4-parity/check_dense_projection_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: ordered/paired/serial F16 variants, cuBLAS selection and
  activation conversion, Q8 kernels, runtime route activation, and C CUDA
  removal remain pending.
- Evidence:
  - Added executable-local Rust `matmul_f16_kernel` and `matmul_f32_kernel`
    with current-C 256-thread shared reductions and primitive F16 weight
    widening.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 42 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    and proved F16, F32, multi-token stride, and invalid-shape behavior on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 74-check comparator, and
    unified parity passed with 133 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained ordered/paired/serial, cuBLAS, Q8, and
    route non-claims.

##### M14.3c2: Ordered Paired And Serial F16 Projection Kernels

- Status: done.
- Goal: port the remaining non-cuBLAS F16 projection kernels without claiming
  their environment-controlled or cuBLAS selection policy.
- Oracle: current-C `matmul_f16_serial_kernel`,
  `matmul_f16_ordered_chunks_kernel`, and
  `matmul_f16_pair_ordered_chunks_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3c2/ordered-projection-kernel-smoke.json`.
- Comparator: `ds4-parity/check_ordered_projection_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: F16/F32 cuBLAS selection and activation conversion, Q8 kernels,
  runtime route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust serial, ordered-chunk, and paired
    ordered-chunk F16 kernels using primitive F16 loads and current-C's fixed
    32-way accumulation order.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 43 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    and proved serial multi-token, ordered-chunk, unequal-width paired, and
    invalid-shape behavior on `NVIDIA B300 SXM6 AC`.
  - Device compilation exposed unsupported `usize::min` lowering; replacing
    it with current-C's explicit upper-bound clamp retained the kernel
    contract.
  - Local formatting, diff, workspace tests, the 74-check comparator, and
    unified parity passed with 134 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained cuBLAS, activation-conversion, Q8, and
    route non-claims.

##### M14.3c3: F16 And F32 BLAS Dispatch And Activation Conversion

- Status: done.
- Goal: port the remaining dense F16/F32 dispatch and activation-conversion
  behavior without claiming Q8 execution or default-route integration.
- Oracle: current-C `f32_to_f16_kernel`, `ds4_gpu_matmul_f16_tensor`,
  `ds4_gpu_matmul_f16_pair_tensor`, and `ds4_gpu_matmul_f32_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.3c3/blas-projection-kernel-smoke.json`.
- Comparator: `ds4-parity/check_blas_projection_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Q8 kernels, runtime route activation, and C CUDA removal remain
  pending.
- Evidence:
  - Advanced `cuda-oxide` to provide mixed-precision cuBLAS and DS4-layout
    projection wrappers, added an executable-local Rust `f32_to_f16_kernel`,
    and encoded current-C F16/F32 and pair dispatch priorities.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 45 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    and proved activation conversion, mixed F16/F32 BLAS, F32 BLAS,
    dispatch priority, pair dispatch, and invalid-shape behavior on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 79-check comparator, and
    unified parity passed with 135 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review corrected stale predecessor current-pin and
    successor-stage assertions before closure.

##### M14.3d1: Q8 Dequantization And Activation Quantization Kernels

- Status: done.
- Goal: port `dequant_q8_0_to_f16_kernel`,
  `dequant_q8_0_to_f32_kernel`, and `quantize_q8_0_f32_kernel` without
  claiming Q8 matmul dispatch or route integration.
- Oracle: current-C packed Q8 dequantization and `lrintf`-based activation
  quantization kernels.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d1/q8-conversion-kernel-smoke.json`.
- Comparator: `ds4-parity/check_q8_conversion_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: quantized matmul kernels, Q8 dispatch, runtime route activation,
  and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `dequant_q8_0_to_f16_kernel`,
    `dequant_q8_0_to_f32_kernel`, and `quantize_q8_0_f32_kernel` while
    preserving current-C's 34-byte packed Q8 blocks, ties-to-even rounding,
    and partial-block zero padding.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 46 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and proved packed F16/F32 dequantization, activation
    quantization, ties-to-even results, partial-block padding, and
    invalid-shape behavior on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 75-check comparator, and
    unified parity passed with 136 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained quantized-matmul, Q8-dispatch, route,
    and removal non-claims.

##### M14.3d2: Base And Prequantized Q8 Matmul Kernels

- Status: done.
- Goal: port base and prequantized Q8 matmul execution before claiming paired
  or HC-expansion kernels and final Q8 dispatch policy.
- Oracle: current-C `matmul_q8_0_kernel`, `matmul_q8_0_preq_kernel`,
  `matmul_q8_0_preq_warp8_kernel`, and
  `matmul_q8_0_preq_batch_warp8_kernel` scalar integer-dot behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d2/q8-matmul-kernel-smoke.json`.
- Comparator: `ds4-parity/check_q8_matmul_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: DP4A acceleration, paired and HC-expansion kernels, final Q8
  dispatch, runtime route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `matmul_q8_0_kernel`,
    `matmul_q8_0_preq_kernel`, `matmul_q8_0_preq_warp8_kernel`, and
    `matmul_q8_0_preq_batch_warp8_kernel` with packed Q8 scalar integer-dot
    execution.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 47 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and proved direct quantizing, generic prequantized,
    single-token warp8, batched warp8, partial-block, and invalid-shape
    behavior on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 81-check comparator, and
    unified parity passed with 137 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained DP4A, pair/HC-expansion, dispatch,
    route, and removal non-claims.

##### M14.3d3: Paired And HC-Expansion Q8 Matmul Kernels

- Status: done.
- Goal: port paired and HC-expansion Q8 matmul kernels before claiming DP4A
  acceleration, final Q8 dispatch policy, or runtime route ownership.
- Oracle: current-C `matmul_q8_0_pair_preq_warp8_kernel` and
  `matmul_q8_0_hc_expand_preq_warp8_kernel` behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d3/q8-specialized-matmul-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_q8_specialized_matmul_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: DP4A acceleration, Q8 dispatch, runtime route activation, and C
  CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `matmul_q8_0_pair_preq_warp8_kernel` and
    `matmul_q8_0_hc_expand_preq_warp8_kernel` with packed Q8 scalar
    integer-dot execution and optional HC block addition.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 48 tests. Live cargo-oxide execution emitted portable `sm_80` PTX
    and proved unequal-width paired output, HC block output, HC-expansion
    output with and without block addition, partial blocks, and invalid-shape
    behavior on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, workspace tests, the 71-check comparator, and
    unified parity passed with 138 passed, 45 skipped, and 0 failed.
    Non-interactive Claude review timed out without a completed result;
    adversarial self-review retained DP4A, dispatch, route, and removal
    non-claims.

##### M14.3d4: Q8 DP4A Acceleration And Dispatch Policy

- Status: done.
- Goal: add the remaining DP4A accelerated integer-dot path and Q8 dispatch
  policy before any runtime route or C-removal claim.
- Oracle: current-C `dot_i8x32_dp4a`, `dot_i8_block`,
  `cuda_q8_use_dp4a`, and `cuda_matmul_q8_0_tensor_labeled` path order.
- Fixture:
  `ds4-parity/baselines/backend/m14.3d4/q8-dp4a-dispatch-smoke.json`.
- Comparator: `ds4-parity/check_q8_dp4a_dispatch_smoke.py --negative-test`
  plus live B300 cargo-oxide execution and PTX assembly validation.
- Acceptance: Rust owns opt-in DP4A acceleration and Q8 dispatch policy;
  runtime graph integration, default route activation, and C CUDA removal
  remain pending.
- Evidence:
  - Added cuda-oxide signed `dp4a_i8` support and lowered it through
    `dp4a.s32.s32` inline PTX because the validated LLVM 18.1.3 backend does
    not lower the newer NVVM intrinsic form.
  - Added executable-local Rust accelerated Q8 matmul proof plus
    `select_q8_matmul_path` and `q8_dp4a_enabled`, matching current-C
    BLAS/warp8/batch-warp/generic order and disable policy.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 50 tests. Live cargo-oxide execution emitted portable `sm_80` PTX,
    `ptxas` accepted its `dp4a.s32.s32` instruction, and the smoke proved
    matching full-block DP4A and partial-block scalar outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 64-check comparator, retained
    M14 checks, and unified parity passed with 134 passed, 50 skipped, and
    0 failed.

#### M14.4: RoPE KV Compressor And Attention Kernels

- Status: done; M14.5 is active.
- Goal: port the current-C RoPE, KV quantization/storage, compressor, and
  attention operation family through bounded Rust CUDA slices.

##### M14.4a: Standalone RoPE Tail And FP8 KV Quantization Kernels

- Status: done.
- Goal: port standalone `rope_tail_kernel` and `fp8_kv_quantize_kernel`
  behavior before claiming KV storage, compressor, or attention execution.
- Oracle: current-C `rope_tail_kernel`, `fp8_kv_quantize_kernel`,
  `ds4_gpu_rope_tail_tensor`, and `ds4_gpu_dsv4_fp8_kv_quantize_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4a/rope-kv-quantization-kernel-smoke.json`.
- Comparator:
  `ds4-parity/check_rope_kv_quantization_kernel_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in standalone RoPE tail and FP8 KV quantization;
  KV storage, compressor, attention, runtime graph integration, default route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `rope_tail_kernel` with position stride,
    YARN interpolation, attention scaling, and inverse tail rotation.
  - Added executable-local Rust `fp8_kv_quantize_kernel` with E4M3FN
    dequantized round trip for only the non-RoPE prefix, including a partial
    trailing 64-wide chunk.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 51 tests and live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage. The smoke proved matching tail rotation and KV
    prefix quantization while preserving the RoPE tail on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 66-check comparator, retained
    M14 checks, and unified parity passed with 135 passed, 50 skipped, and
    0 failed.

##### M14.4b: Raw KV Storage And Indexer QAT Kernels

- Status: done.
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
- Acceptance: Rust owns opt-in direct raw-KV storage and indexer QAT
  execution. Same-launch overlapping ring-row write ordering, composed FP8
  storage, compressor, attention, runtime graph integration, default route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust `store_raw_kv_batch_kernel` with current-C
    FP16 round-trip storage and deterministic ring wrap across distinct
    destination rows.
  - Added executable-local Rust `indexer_hadamard_fp4_kernel` with 128-wide
    Hadamard rotation and four-block E2M1FN activation simulation.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 52 tests and live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matching raw-storage/indexer outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 68-check comparator, retained
    M14 checks, and unified parity passed with 136 passed, 50 skipped, and
    0 failed.

##### M14.4c: Composed KV Storage And Compressor Kernels

- Status: done through M14.4c3b.
- Goal: port composed KV storage and bounded compressor kernels before
  claiming attention execution.

###### M14.4c1: Composed FP8 Raw Storage And Compressor Row Stores

- Status: done.
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
- Acceptance: Rust owns opt-in composed FP8 raw storage and compressor
  row-store kernels with both accepted APE scalar types. Compressor
  pooling/shift, wrapper orchestration, normalization/RoPE composition,
  attention, runtime graph integration, default route activation, and C CUDA
  removal remain pending.
- Evidence:
  - Added executable-local Rust composed FP8 quantization plus raw-store
    execution and ratio-4 compressor store/set-row execution.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 53 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched composed storage, ratio-4 row
    geometry, and F32/F16 APE outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 67-check comparator, retained
    M14 checks, and unified parity passed with 137 passed, 50 skipped, and
    0 failed.

###### M14.4c2: Compressor Pooling And Ratio-4 Shift Kernels

- Status: done.
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
- Acceptance: Rust owns opt-in compressor pooling and ratio-4 shift device
  kernels across general-ratio, ratio-4, and replay cases. Update/prefill
  wrapper orchestration, normalization/RoPE/FP8 composition, attention,
  runtime graph integration, default route activation, and C CUDA removal
  remain pending.
- Evidence:
  - Added executable-local Rust compressor prefill-pool, update-pool, and
    ratio-4 shift execution with F16 APE input coverage.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 54 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched general-ratio, ratio-4/replay,
    update-pool, and state-shift outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 68-check comparator, retained
    M14 checks, and unified parity passed with 138 passed, 50 skipped, and
    0 failed.

###### M14.4c3a: Compressor Update Orchestration

- Status: done.
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
- Acceptance: Rust owns opt-in compressor update orchestration across
  non-emitting store-only, ratio-4 emitting, and general-ratio emitting
  cases. Prefill/replay orchestration, FP8 compressed-cache composition,
  attention, runtime graph integration, default route activation, and C CUDA
  removal remain pending.
- Evidence:
  - Added executable-local Rust compressor update orchestration with a
    nonzero compressed-row offset, F16 APE, weighted RMS normalization, YARN
    RoPE, and ratio-4 post-emission state shift coverage.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 55 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched non-emitting, ratio-4 emitting, and
    general-ratio emitting outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the c3a comparator, retained M14
    checks, and unified parity passed with 139 passed, 50 skipped, and
    0 failed.

###### M14.4c3b: Compressor Prefill And Replay Orchestration

- Status: done.
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
- Acceptance: Rust owns opt-in compressor prefill, ratio-4 replay, and
  ratio-4 state-only orchestration including optional FP8 compressed output.
  Attention, runtime graph integration, default route activation, and C CUDA
  removal remain pending.
- Evidence:
  - Added executable-local Rust state initialization, set-row placement,
    pooled compressed output, weighted RMS, YARN RoPE, and optional FP8
    processing across general prefill, ratio-4 prefill/replay, and
    ratio-4 state-only cases.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 56 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched state placement, replay ordering, and
    optional FP8 outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the c3b comparator, retained M14
    checks, and unified parity passed with 140 passed, 50 skipped, and
    0 failed.

##### M14.4d: Attention Kernels

- Status: done through M14.4d8b.
- Goal: port current-C attention decode, prefill, indexed, and output-Q8
  device behavior after compressor surfaces are proved.

###### M14.4d1: Single-Token Mixed Attention Decode Surface

- Status: done.
- Goal: port the exported single-token mixed attention decode behavior before
  batched/window/heads8, prefill/indexed, or output-Q8 ownership.
- Oracle: current-C `ds4_gpu_attention_decode_heads_tensor` and the
  `single_all` branch of `attention_decode_mixed_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d1/attention-decode-single-mixed-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_decode_single_mixed_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in single-token decode with raw-ring,
  compressed-row masking, learned-sink softmax, and raw-only output cases.
  Batched/window/heads8 decode, prefill/indexed/output-Q8 attention, runtime
  graph integration, default route activation, and C CUDA removal remain
  pending.
- Evidence:
  - Added executable-local Rust single-token mixed-attention execution using
    the current-C visible-row and sink-softmax semantics.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 57 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched masked/unmasked compressed rows,
    wrapped raw rows, sink softmax, and raw-only outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the d1 comparator, retained M14
    checks, and unified parity passed with 141 passed, 50 skipped, and
    0 failed.

###### M14.4d2: Generic Batched Mixed Attention Decode Surfaces

- Status: done.
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
- Acceptance: Rust owns opt-in generic batched raw/mixed decode with causal
  window selection, raw-ring wrap, compressed visibility/masking, learned
  sink softmax, and raw-only batched output. Heads8-online dispatch,
  prefill/indexed/output-Q8 attention, runtime graph integration, default
  route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust generic batched mixed-attention execution
    using current-C causal-window, compressed-visibility, and sink-softmax
    semantics.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 58 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched mixed/raw batched output, wrapped raw
    rows, per-token compressed masks, and learned sink contribution on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 63-check d2 comparator,
    retained M14 checks, and unified parity passed with 142 passed, 50
    skipped, and 0 failed.

###### M14.4d3: Heads8 Online Attention Decode Kernels

- Status: done.
- Goal: port optimized heads8-online decode and its dispatch selection before
  prefill, indexed, or output-Q8 attention ownership.
- Oracle: current-C `attention_decode_mixed_heads8_online_kernel`,
  `attention_decode_batch_launch`, `cuda_attention_score_buffer_fits`, and
  the decode window-attention environment/quality dispatch conditions.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d3/attention-decode-heads8-online-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_decode_heads8_online_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in decode heads8-online output semantics and the
  current-C decode online dispatch predicates for score-buffer overflow and
  enabled multi-token window attention. Raw/mixed prefill, indexed, and
  output-Q8 attention, runtime graph integration, default route activation,
  and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust grouped-head online decode execution with a
    partial final head group, batched causal-window/ring-row coverage, and
    single-token all-compressed coverage.
  - Added `select_attention_decode_path`, matching current-C overflow,
    mask, head-dimension, disabled-window, forced-window, token-threshold,
    and quality-mode selection behavior.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 60 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched online decode outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 64-check d3 comparator,
    retained M14 checks, and unified parity passed with 143 passed, 50
    skipped, and 0 failed.

###### M14.4d4: Generic Raw And Mixed Attention Prefill Kernels

- Status: done.
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
- Acceptance: Rust owns opt-in generic raw, static mixed, and masked mixed
  prefill output semantics with causal raw windows, compressed visibility and
  masking, and learned-sink softmax. Static heads8-online/CUBLAS prefill
  dispatch, indexed/output-Q8 attention, runtime graph integration, default
  route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust generic raw and mixed prefill kernels with
    static and masked compressed-row cases.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 61 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched generic raw/static/masked mixed output
    on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 68-check d4 comparator,
    retained attention checks, and unified parity passed with 149 passed, 45
    skipped, and 0 failed.

###### M14.4d5: Static Heads8 Online And CUBLAS Attention Prefill Dispatch

- Status: done.
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
- Acceptance: Rust owns opt-in static heads8-online prefill output semantics,
  live raw and masked-mixed cuBLAS prefill pipelines, and current-C online /
  cuBLAS / generic branch selection. Indexed/output-Q8 attention, runtime
  graph integration, default route activation, and C CUDA removal remain
  pending.
- Evidence:
  - Added executable-local grouped-head online prefill, current-C-equivalent
    softmax/pack/unpack kernels, and row-major adapter kernels required to
    execute the safe `cuda-core` strided-batched SGEMM API.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 63 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched static online, raw cuBLAS, and masked
    mixed cuBLAS outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 70-check d5 comparator,
    retained attention checks, and unified parity passed with 150 passed, 45
    skipped, and 0 failed.

###### M14.4d6: Generic Indexed Mixed Attention Surface

- Status: done.
- Goal: port the generic indexed mixed-attention surface before optimized
  indexed heads8/sort dispatch or output-Q8 attention ownership.
- Oracle: current-C `attention_indexed_mixed_kernel` and
  `ds4_gpu_attention_indexed_mixed_batch_heads_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d6/attention-indexed-generic-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_indexed_generic_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in generic indexed mixed attention output
  semantics including ordered/duplicate top-k filtering, ratio-zero all-row
  visibility, causal raw windows, wrapped raw rows, and learned-sink softmax.
  Indexed sort/heads8 dispatch, output-Q8 attention, runtime graph
  integration, default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust generic indexed mixed attention kernel with
    selected compressed-row order/duplicate preservation and ratio-zero
    visibility coverage.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 64 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched indexed output on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 64-check d6 comparator,
    retained attention checks, and unified parity passed with 146 passed, 50
    skipped, and 0 failed.

###### M14.4d7: Optimized Indexed Sort And Heads8 Attention Kernels

- Status: done.
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
- Acceptance: Rust consumes the already owned indexed-sort policy and owns
  opt-in indexed heads8-online/rb4 output semantics and branch selection.
  Output-Q8 attention, runtime graph integration, default route activation,
  and C CUDA removal remain pending.
- Evidence:
  - Added executable-local sorted-topk-to-online indexed attention execution,
    rb4 filtered/duplicate indexed execution, and current-C-equivalent online,
    two-pass, and generic branch policy.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 66 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched both optimized indexed outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 68-check d7 comparator,
    retained attention checks, and unified parity passed with 147 passed, 50
    skipped, and 0 failed.

###### M14.4d8: Output Q8 Attention Projection Surfaces

- Status: done after splitting into M14.4d8a and M14.4d8b.
- Goal: port native Q8 output surfaces before the optional F16/cuBLAS A
  projection optimization.

####### M14.4d8a: Native Q8 Attention Output Projection Surfaces

- Status: done.
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
- Acceptance: Rust consumes already owned Q8 conversion/matmul kernels and
  owns opt-in native low-output and batched two-stage attention projection
  semantics. F16/cuBLAS attention-output-A dispatch, runtime graph
  integration, default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust grouped A projection and B projection
    orchestration over Q8 prequantized inputs, covering single low output,
    batched low/output, and partial blocks.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 67 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched native Q8 output on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 63-check d8a comparator,
    retained attention checks, and unified parity passed with 148 passed, 50
    skipped, and 0 failed.

####### M14.4d8b: CUBLAS Attention Output A Dispatch

- Status: done.
- Goal: port the optional F16/cuBLAS attention-output-A acceleration and
  branch policy before attention-family closure.
- Oracle: current-C `attention_pack_group_heads_f16_kernel`,
  `attention_unpack_group_low_kernel`, and the
  `ds4_gpu_attention_output_q8_batch_tensor` cuBLAS A branch.
- Fixture:
  `ds4-parity/baselines/backend/m14.4d8b/attention-output-q8-cublas-smoke.json`.
- Comparator:
  `ds4-parity/check_attention_output_q8_cublas_smoke.py --negative-test`
  plus live B300 cargo-oxide and cuBLAS execution.
- Acceptance: Rust owns opt-in attention-output-A cuBLAS branch selection,
  F16-rounded grouped-head pack/unpack layout, and live grouped-A cuBLAS
  output through cuda-oxide's safe SGEMM adapter. The public cuda-oxide
  wrapper does not expose current-C's `CUDA_R_16F` GemmEx entry point, so
  exact F16 GemmEx invocation is recorded as an adapter difference rather
  than overclaimed. Runtime graph integration, default route activation, and
  C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust F16 pack/unpack kernels and grouped-A cuBLAS
    pipeline with current-C-equivalent quality, readiness, minimum-token,
    disable-flag, and expanded-weight branch policy.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 69 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    with libdevice linkage and matched packed, grouped, and unpacked output
    on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 66-check d8b comparator,
    retained attention checks, and unified parity passed with 149 passed, 50
    skipped, and 0 failed.

#### M14.5: Router MoE And Hyperconnection Kernels

- Status: done through M14.5d.
- Goal: port the remaining current-C router, routed-MoE, shared-expert, and
  hyperconnection CUDA surfaces after attention-family closure.

##### M14.5a: Scalar Router Selection Surfaces

- Status: done.
- Goal: port the scalar single-token and batched router-selection behavior
  before optimized router dispatch and routed expert execution.
- Oracle: current-C `router_select_kernel`,
  `ds4_gpu_router_select_tensor`, and `ds4_gpu_router_select_batch_tensor`.
- Fixture: `ds4-parity/baselines/backend/m14.5a/router-scalar-smoke.json`.
- Comparator: `ds4-parity/check_router_scalar_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in scalar router probability transformation,
  bias-ranked top-6 output, hash routing with invalid-token fallback, scaled
  selected weights, and single/batched layout. Parallel/warp selection,
  routed MoE, hyperconnection, runtime graph integration, default route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust scalar router selection for current-C softplus
    probability, bias ordering, hash selection, and normalized-weight
    behavior.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 70 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched scalar router outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 63-check M14.5a comparator,
    retained M14 checks, and unified parity passed with 150 passed, 50
    skipped, and 0 failed.

##### M14.5b: Parallel And Warp Router Dispatch

- Status: done.
- Goal: port optimized parallel/warp router-selection kernels and current-C
  dispatch priority before routed MoE execution.
- Oracle: current-C `router_select_parallel_kernel`,
  `router_select_warp_topk_kernel`, and the optimized launch selection in
  `ds4_gpu_router_select_tensor` and `ds4_gpu_router_select_batch_tensor`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5b/router-optimized-smoke.json`.
- Comparator: `ds4-parity/check_router_optimized_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in optimized router probability storage,
  parallel shared-memory selection, warp-shuffle top-k ordering including
  equal-score index ordering, partial four-row warp blocks, hash fallback,
  and current-C warp/parallel/scalar dispatch priority. Routed MoE,
  hyperconnection, runtime graph integration, default route activation, and C
  CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust parallel and warp router kernels and public
    optimized path selection matching the current-C disable-flag ordering.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 72 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched shared-memory and warp-shuffle router
    outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the 70-check M14.5b comparator,
    retained M14 checks, and unified parity passed with 151 passed, 50
    skipped, and 0 failed.

##### M14.5c1: Packed F32-Activation Routed MoE Fallback

- Status: done.
- Goal: port the current-C routed-MoE fallback which decodes packed IQ2-XXS
  gate/up weights and packed Q2_K down weights against F32 activations.
- Oracle: current-C `dev_iq2_xxs_dot_f32`, `dev_q2_K_dot_f32`,
  `moe_gate_up_mid_f32_kernel`, `moe_down_f32_kernel`, and `moe_sum_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c1/routed-moe-f32-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_f32_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in packed IQ2-XXS and Q2_K fallback decode,
  weighted SwiGLU/clamp behavior, negative-expert fallback, expert summation,
  and single-token and batched fallback layout. Q8 activation quantization,
  optimized routed-MoE dispatch, Q4_K execution, hyperconnection, runtime
  graph integration, default route activation, and C CUDA removal remain
  pending.
- Evidence:
  - Added executable-local Rust packed-weight routed-MoE fallback kernels with
    table-indexed IQ2-XXS gate/up decode, Q2_K down decode, shared reductions,
    weighted activation, and expert summation.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 73 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched batch, single-token, clamp, and
    negative-expert routed-MoE results on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c1 comparator, retained
    M14 checks, and unified parity passed with 152 passed, 50 skipped, and 0
    failed.

##### M14.5c2a: Default Single-Token Quantized Routed MoE Dispatch

- Status: done.
- Goal: port the current-C default single-token IQ2-XXS/Q2_K quantized
  routed-MoE compute path after the exact packed fallback is closed.
- Oracle: current-C `q8_K_quantize_kernel`,
  `dev_dot_iq2_xxs_q8_K_block`, `dev_dot_q2_K_q8_K_block`,
  `moe_gate_up_mid_decode_lut_qwarp32_kernel`, and
  `moe_down_sum6_qwarp32_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2a/routed-moe-quantized-single-smoke.json`.
- Comparator:
  `ds4-parity/check_routed_moe_quantized_single_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in Q8_K activation quantization, LUT-equivalent
  IQ2-XXS/Q8_K gate/up decode, Q2_K/Q8_K direct six-expert down output, the
  default single-token IQ2/Q2 compute branch, optional gate/up auxiliary
  writes, and negative-expert fallback. Batched sorted/tiled dispatch, Q4_K,
  hyperconnection, runtime graph integration, default route activation, and C
  CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust quantized routed-MoE kernels for Q8_K input
    and intermediate activation fields, quarter-warp gate/up execution, and
    direct sum-six down output.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 74 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched default single-token output, auxiliary
    writes, negative-expert behavior, and zero-input quantization on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2a comparator, retained
    M14 checks, and unified parity passed with 153 passed, 50 skipped, and 0
    failed.

##### M14.5c2b1: Batched Sorted-Pair Metadata

- Status: done.
- Goal: port the current-C batched routed-MoE pair histogram, prefix-offset,
  and scatter metadata kernels before sorted projection execution.
- Oracle: current-C `moe_count_sorted_pairs_kernel`,
  `moe_prefix_sorted_pairs_kernel`, and `moe_scatter_sorted_pairs_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2b1/routed-moe-sorted-pairs-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_sorted_pairs_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in atomic expert histogram, prefix offsets/cursors,
  scatter grouping, duplicate pair preservation, and negative-expert
  expert-zero bucketing. Sorted projection, expert-tile/atomic-down execution,
  Q4_K, hyperconnection, runtime graph integration, default route activation,
  and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust device-atomic metadata kernels matching the
    current-C count, prefix, and scatter stages without imposing ordering
    inside equal-expert atomic regions.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 75 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    and matched histogram, prefix, grouped-scatter, duplicate, and
    negative-expert metadata behavior on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2b1 comparator, retained
    M14 checks, and unified parity passed with 154 passed, 50 skipped, and 0
    failed.

##### M14.5c2b2: Sorted-Pair P2 Quantized Projection

- Status: done.
- Goal: port the no-expert-tiles/default-P2 batched IQ2-XXS/Q2_K gate/down
  projection kernels over the sorted metadata before expert-tile and
  atomic-down scheduling variants.
- Oracle: current-C `use_p2_sorted` dispatch,
  `moe_gate_up_mid_sorted_p2_qwarp32_kernel`,
  `moe_down_sorted_p2_qwarp32_kernel`, and `moe_sum_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2b2/routed-moe-sorted-p2-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_sorted_p2_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in no-expert-tiles/default-P2 batched IQ2/Q2
  quantized routed-MoE projection over sorted metadata, including pair-indexed
  weighted activation/down output and final per-token summation. Expert-tile
  or atomic-down scheduling, Q4_K, hyperconnection, runtime graph integration,
  default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust sorted P2 kernels composed with the already
    ported metadata, Q8_K quantization, and sum surfaces; the fixture covers
    two tokens, duplicate experts, negative-expert fallback, and partial rows.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 76 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched sorted P2 gate/up, down, and summed output
    behavior on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2b2 comparator, retained
    M14 checks, and unified parity passed with 155 passed, 50 skipped, and 0
    failed.

##### M14.5c2c1: Expert-Tile Descriptor Metadata

- Status: done.
- Goal: port the current-C expert-tile offset and descriptor construction
  before any tile-local gate/down projection is claimed.
- Oracle: current-C `moe_build_expert_tile_offsets_kernel`,
  `moe_build_expert_tiles_kernel`, and `expert_tile_m` selection.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c1/routed-moe-expert-tiles-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_expert_tiles_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in tile descriptor generation for default
  eight-pair and alternate four-pair expert grouping, including partial tiles
  and negative-expert bucket-zero counts. Tile-local projection,
  atomic-down/rowspan dispatch, Q4_K, hyperconnection, runtime graph
  integration, default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust tile offset and descriptor kernels composed
    with the sorted-pair expert-count surface for both block sizes.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 77 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    and matched tile totals, expert identifiers, starts, partial tiles, and
    expert-zero fallback on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c1 comparator, retained
    M14 checks, and unified parity passed with 156 passed, 50 skipped, and 0
    failed.

##### M14.5c2c2: Default Tile8 Row32 Projection

- Status: done.
- Goal: port current-C default eight-pair row32 expert-tile gate/down
  projection after the descriptor metadata is closed, leaving atomic-down
  and wider-row variants for subsequent evidence boundaries.
- Oracle: current-C `moe_gate_up_mid_expert_tile8_row32_kernel`,
  `moe_down_expert_tile8_row32_kernel`, and default `expert_tile_m == 8u`
  dispatch.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c2/routed-moe-tile8-row32-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_tile8_row32_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in functional tile8 row32 gate/up and non-atomic
  down projection over validated expert-tile descriptors, including multi-tile
  and partial-tile groups. Shared-cache specialization, tile4,
  atomic-down/tile16/rowspan dispatch, Q4_K, hyperconnection, runtime graph
  integration, default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust tile8 row32 projection kernels over
    prequantized Q8_K inputs, covering same-expert multi-tile execution and
    pair-indexed output behavior.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 78 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched tile8 gate/up and down outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c2 comparator, retained
    M14 checks, and unified parity passed with 157 passed, 50 skipped, and 0
    failed.

##### M14.5c2c3: Tile4 Row32 Projection

- Status: done.
- Goal: port the optional four-pair row32 expert-tile gate/down projection
  path before atomic-down, tile16, rowspan, and shared-cache optimization
  boundaries.
- Oracle: current-C `moe_gate_up_mid_expert_tile4_row32_kernel`,
  `moe_down_expert_tile4_row32_kernel`, and `DS4_CUDA_MOE_TILE4` selection.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c3/routed-moe-tile4-row32-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_tile4_row32_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in functional tile4 row32 gate/up and non-atomic
  down projection over validated expert-tile descriptors, including a
  three-tile same-expert group and partial tile. Shared-cache specialization,
  atomic-down/tile16/rowspan dispatch, Q4_K, hyperconnection, runtime graph
  integration, default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added `DS4_CUDA_MOE_TILE4`-selected executable-local Rust tile4 row32
    projection kernels over the shared functional row32 proof harness.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 79 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched tile4 gate/up and down outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c3 comparator, retained
    M14 checks, and unified parity passed with 158 passed, 50 skipped, and 0
    failed.

##### M14.5c2c4: Atomic Expert-Tile Down Output

- Status: done.
- Goal: port token-indexed atomic accumulation for expert-tile down projection
  before tile16 and widened-row scheduling variants.
- Oracle: current-C `zero_kernel`, `use_atomic_down` dispatch, and atomic
  branches of `moe_down_expert_tile8_row32_kernel` and
  `moe_down_expert_tile4_row32_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c4/routed-moe-atomic-down-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_atomic_down_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in device-zero initialization and token-indexed
  `DeviceAtomicF32` down accumulation for tile8 and tile4 row32 schedules.
  Tile16/rowspan dispatch, shared-cache specialization, Q4_K, hyperconnection,
  runtime graph integration, default route activation, and C CUDA removal
  remain pending.
- Evidence:
  - Added executable-local Rust `zero_kernel` and atomic row32 down branches;
    one atomic execution validates both tile8 and tile4 scheduling against
    token-summed reference output.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 80 tests; live cargo-oxide execution emitted portable `sm_80` PTX,
    lowered `DeviceAtomicF32::fetch_add`, and matched atomic outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c4 comparator, retained
    M14 checks, and unified parity passed with 159 passed, 50 skipped, and 0
    failed.

##### M14.5c2c5: Tile16 Row32 Atomic Down

- Status: done.
- Goal: port the high-token tile16 row32 atomic down projection selection
  before widened-row and shared-cache specialization boundaries.
- Oracle: current-C `use_down_tile16` dispatch, separate tile16 descriptor
  construction, and `moe_down_expert_tile16_row32_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c5/routed-moe-tile16-row32-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_tile16_row32_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in tile16 row32 atomic-down scheduling with
  retained tile8 gate metadata and partial tile16 behavior. Gate/down
  row2048/rowspan dispatch, shared-cache specialization, Q4_K,
  hyperconnection, runtime graph integration, default route activation, and
  C CUDA removal remain pending.
- Evidence:
  - Added executable-local Rust tile16 row32 atomic down projection selected
    over separately built tile16 descriptors while gate/up remains tile8.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 81 tests; live cargo-oxide execution emitted portable `sm_80` PTX
    through libdevice and matched partial tile16 atomic output on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c5 comparator, retained
    M14 checks, and unified parity passed with 160 passed, 50 skipped, and 0
    failed.

##### M14.5c2c6: Gate Tile8 Rowspan Projection

- Status: done.
- Goal: port the tile8 widened-row gate/up scheduling variants before
  widened-row atomic down and shared-cache specialization closure.
- Oracle: current-C `moe_gate_up_mid_expert_tile8_rowspan_kernel<512>`,
  `moe_gate_up_mid_expert_tile8_rowspan_kernel<1024>`, and
  `moe_gate_up_mid_expert_tile8_row2048_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c6/routed-moe-gate-rowspan-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_gate_rowspan_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in functional tile8 gate projection for the
  row512, row1024, and row2048 scheduling variants over retained tile
  descriptors, including partial row spans. Widened down scheduling,
  shared-cache specialization, Q4_K, hyperconnection, runtime graph
  integration, default route activation, and C CUDA removal remain pending.
- Evidence:
  - Added an executable-local Rust parameterized tile8 gate-rowspan kernel
    selected by the smoke harness for all three current-C scheduling spans.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 82 tests; live cargo-oxide execution found seven kernels, emitted
    portable `sm_80` PTX through libdevice, and matched all three gate-span
    outputs on `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c6 comparator, retained
    routed-MoE checks, and unified parity passed with 161 passed, 50 skipped,
    and 0 failed.

##### M14.5c2c7: Down Tile16 Rowspan Projection

- Status: done.
- Goal: port the tile16 widened-row atomic down scheduling variants after
  row32 atomic down and widened gate scheduling are established.
- Oracle: current-C `moe_down_expert_tile16_rowspan_kernel<512>`,
  `moe_down_expert_tile16_rowspan_kernel<1024>`, and
  `moe_down_expert_tile16_row2048_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2c7/routed-moe-down-rowspan-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_down_rowspan_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in functional tile16 atomic down projection for
  row512, row1024, and row2048 scheduling variants over retained tile16
  descriptors, including partial row spans. Shared-cache specialization,
  Q4_K, hyperconnection, runtime graph integration, default route activation,
  and C CUDA removal remain pending.
- Evidence:
  - Added an executable-local Rust parameterized tile16 atomic down-rowspan
    kernel selected by the smoke harness for all three current-C spans.
  - On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
    with 83 tests; live cargo-oxide execution found eight kernels, emitted
    portable `sm_80` PTX through libdevice, lowered the device float atomic
    operation, and matched all three down-span outputs on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff, library tests, the M14.5c2c7 comparator, retained
    routed-MoE checks, and unified parity passed with 162 passed, 50 skipped,
    and 0 failed.

##### M14.5c2d: Single-Token Q4_K Routed MoE

- Status: done.
- Goal: port the current-C single-token Q4_K gate/up and direct down-sum
  branch after the IQ2/Q2 tiled routed-MoE family is functionally covered.
- Evidence:
  - Added the opt-in `DS4_CUDA_MOE_Q4_K=1` route to the existing
    executable-local Rust quantized single-token smoke binary, with Q4_K/Q8_K
    packed dot, gate/up, and direct sum-six down kernels.
  - The current-C oracle remains `dev_q4_K_get_scale_min`,
    `dev_dot_q4_K_q8_K_block`,
    `moe_gate_up_mid_decode_q4K_qwarp32_kernel`, and
    `moe_down_q4K_sum6_qwarp32_kernel`.
  - The fixture and comparator are
    `ds4-parity/baselines/backend/m14.5c2d/routed-moe-q4-k-single-smoke.json`
    and `ds4-parity/check_routed_moe_q4_k_single_smoke.py --negative-test`.
  - B300 feature tests passed with 84 tests; live cargo-oxide execution found
    five kernels, emitted portable `sm_80` PTX through libdevice, and matched
    Q4_K gate/up plus direct down output on `NVIDIA B300 SXM6 AC`.
  - The local CUDA-kernel feature build is unavailable on this Mac because
    `/usr/local/cuda/include/cuda.h` is absent; B300 is the execution proof.
  - Shared-cache expert-tile specialization, hyperconnection, runtime graph
    integration, default route activation, and C CUDA removal remain
    unclaimed.
  - Local formatting, diff and library tests, the M14.5c2d comparator, the
    retained selector run, and unified parity passed with 168 passed, 45
    skipped, and 0 failed.

##### M14.5c2e: Shared-Cache Expert-Tile Projection

- Status: done.
- Goal: port the current-C expert-tile gate/down specialization that stages
  Q8 blocks and IQ2 lookup tables in shared memory before starting
  hyperconnection kernels.
- Evidence:
  - Added opt-in `DS4_CUDA_MOE_SHARED_CACHE=1` cached row-span kernels in the
    existing expert-tile smoke binary, composed with gate-rowspan and tile16
    atomic-down selectors.
  - The Rust kernels stage Q8 scale/value data, fixture IQ2 lookup/sign data,
    and Q2 block sums in `SharedArray` storage with block synchronization
    before dot products.
  - The fixture and comparator are
    `ds4-parity/baselines/backend/m14.5c2e/routed-moe-shared-cache-smoke.json`
    and `ds4-parity/check_routed_moe_shared_cache_smoke.py --negative-test`.
  - B300 feature tests passed with 85 tests; live cargo-oxide execution found
    ten kernels, emitted portable `sm_80` PTX through libdevice, and matched
    shared-cache gate/down output for row spans `512`, `1024`, and `2048` on
    `NVIDIA B300 SXM6 AC`.
  - Local formatting, diff and library tests, the M14.5c2e comparator, the
    retained selector checks, and unified parity passed with 169 passed, 45
    skipped, and 0 failed.
  - Generic/sorted qwarp fallback projection, hyperconnection, runtime graph
    integration, default route activation, and C CUDA removal remain
    unclaimed.

##### M14.5c2f: Generic And Sorted Qwarp Quantized Routed MoE

- Status: done.
- Goal: port the selectable current-C quantized fallback projections used
  when decode-LUT or sorted-P2/expert-tile scheduling is disabled.
- Oracle: current-C `moe_gate_up_mid_qwarp32_kernel`,
  `moe_down_qwarp32_kernel`, `moe_gate_up_mid_sorted_qwarp32_kernel`,
  `moe_down_sorted_qwarp32_kernel`, and their selector branches.
- Fixture:
  `ds4-parity/baselines/backend/m14.5c2f/routed-moe-qwarp-fallback-smoke.json`.
- Comparator: `ds4-parity/check_routed_moe_qwarp_fallback_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns opt-in single-token no-decode-LUT qwarp gate/up
  with generic non-direct down/sum and batched no-expert-tiles/no-P2 sorted
  qwarp gate/down/sum execution over already validated Q8 and sorted-pair
  surfaces. Hyperconnection, runtime graph integration, default route
  activation, and C CUDA removal remain pending.
- Evidence:
  - Added `DS4_CUDA_MOE_QWARP_FALLBACK=1`-selected generic and sorted qwarp
    kernels in the existing sorted-P2 executable-local Rust smoke binary.
  - Live B300 cargo-oxide execution found eleven kernels, emitted portable
    `sm_80` PTX through libdevice, and matched generic single-token and
    sorted no-P2 gate/down/summed output on `NVIDIA B300 SXM6 AC`.
  - B300 feature tests passed with 86 tests. The local CUDA-feature build is
    unavailable on this Mac because `/usr/local/cuda/include/cuda.h` is
    absent; B300 is the execution proof.
  - Local formatting, diff and library tests, the M14.5c2f comparator, the
    retained selector checks, and unified parity passed with 170 passed, 45
    skipped, and 0 failed.
  - Hyperconnection, runtime graph integration, default route activation, and
    C CUDA removal remain unclaimed.

##### M14.5d: Hyperconnection Split And Expansion Kernels

- Status: done.
- Goal: port the remaining current-C hyperconnection split, weighted-sum,
  expansion, and output-weight kernel surfaces after routed-MoE compute
  closure.
- Oracle: current-C `hc_split_sinkhorn_kernel`, `hc_weighted_sum_kernel`,
  `hc_expand_kernel`, `hc_split_weighted_sum_fused_kernel`,
  `hc_split_weighted_sum_norm_fused_kernel`, and
  `output_hc_weights_kernel`.
- Fixture:
  `ds4-parity/baselines/backend/m14.5d/hyperconnection-smoke.json`.
- Comparator: `ds4-parity/check_hyperconnection_smoke.py --negative-test`
  plus live B300 cargo-oxide execution.
- Acceptance: Rust owns the opt-in M14.5 hyperconnection kernel family,
  including synchronized fused split/reduction/normalization behavior and
  plain/add expansion. Runtime graph integration, default-route activation,
  and C CUDA removal remain pending in M14.6.
- Evidence:
  - Added executable-local Rust hyperconnection split, weighted-sum,
    expansion, fused split/reduction, fused normalization, and output-weight
    kernels with deterministic host reference calculations.
  - Live B300 cargo-oxide execution found six kernels and eight total device
    functions, emitted portable `sm_80` PTX through libdevice, linked a
    `239188`-byte LTOIR container, and matched all split, reduction,
    expansion, normalization, and output-weight outputs on
    `NVIDIA B300 SXM6 AC`.
  - B300 feature tests passed with 87 tests. The local CUDA-feature build is
    unavailable on this Mac because `/usr/local/cuda/include/cuda.h` is
    absent; B300 is the execution proof.
  - M14.5 is now complete as an opt-in operation-family port. Default route
    promotion and `ds4_cuda.cu` removal are not claimed.
  - Local formatting, diff and library tests, the 77-check M14.5d
    comparator, retained M14.5 comparators, and unified parity passed with
    171 passed, 45 skipped, and 0 failed.

#### M14.6: CUDA Route Promotion And C CUDA Removal Gate

- Status: active; split into M14.6a and M14.6b after production linkage
  inspection showed that operation proof does not yet provide a Rust ABI
  backend.
- Goal: determine whether the Rust CUDA backend can replace the current-C
  default CUDA path now that all assigned operation families have opt-in Rust
  execution proofs.
- Oracle: all M14 fixtures, the M14.0 ownership inventory, and retained
  current-C official-vector, long-context, tool/server, and benchmark gates.
- Comparator: same-B300 end-to-end Rust CUDA route versus retained current-C
  route, including required quality and throughput gates.
- Acceptance: promote the Rust CUDA default route and remove `ds4_cuda.cu`
  linkage only if exported-function coverage and end-to-end gates pass;
  otherwise record the exact blocker and retain current C.

##### M14.6a: Production Route Linkage Blocker

- Status: done.
- Goal: test whether the validated cuda-oxide kernels are already reachable
  through the production Rust CUDA runtime before attempting route promotion.
- Oracle: `rust/ds4-gpu/build.rs`, `rust/ds4-gpu-sys/src/lib.rs`,
  `rust/ds4-cuda`, and `rust/ds4-engine/src/lib.rs`.
- Fixture:
  `ds4-parity/baselines/backend/m14.6a/production-route-blocker.json`.
- Comparator: `ds4-parity/check_cuda_route_promotion_gate.py --negative-test`.
- Acceptance: fail closed when production still compiles `ds4_cuda.cu`, the
  Rust cuda-oxide crate does not implement the `ds4_gpu_*` ABI, or the Rust
  runtime graph route remains unavailable.
- Evidence:
  - Confirmed that Linux `ds4-gpu` still compiles and archives
    `ds4_cuda.o`, exposes the 81-function GPU ABI through FFI, and has no
    dependency on `ds4-cuda`.
  - Confirmed that `ds4-cuda` currently packages kernels in executable-local
    modules and does not export a linkable `ds4_gpu_*` implementation, while
    `ds4-engine` still rejects `--runtime-graph graph`.
  - The gate records default-route promotion and C CUDA removal as blocked,
    with Rust ABI backend assembly required next.
  - `cargo test --locked -p ds4-cuda --lib` passed with 86 tests locally;
    B300 `cargo test --locked -p ds4-cuda --features cuda-oxide-backend --
    --nocapture` passed with 88 tests; the production-linkage checker passed
    with 56 checks; and the unified parity report passed with 172 passes,
    45 skips, and no failures.

##### M14.6b: Rust CUDA ABI Backend Assembly

- Status: active; split into M14.6b1 and M14.6b2 so linkable resource
  exports are validated before compute export assembly.
- Goal: consolidate the validated cuda-oxide substrate and kernel families
  behind a linkable Rust implementation of the production `ds4_gpu_*` ABI,
  then compare the resulting route end to end before removing current C.
- Acceptance: the production Linux CUDA build can select a Rust-owned ABI
  backend without compiling `ds4_cuda.cu`, and same-B300 output, quality,
  long-context, server, and benchmark gates pass before default promotion.

##### M14.6b1: Rust CUDA Resource ABI Exports

- Status: done.
- Goal: make the validated cuda-oxide resource substrate linkable through
  production-shaped `ds4_gpu_*` C ABI symbols without overclaiming compute
  execution or route ownership.
- Fixture: `ds4-parity/baselines/backend/m14.6b1/abi-resource-smoke.json`.
- Comparator: `ds4-parity/check_cuda_abi_resource_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda` now emits an opt-in Rust `staticlib` and exports 16
    initialization, tensor/resource, synchronization, and managed-KV policy
    symbols from `rust/ds4-cuda/src/abi.rs`.
  - On `ds4-rust-port-b300` (`NVIDIA B300 SXM6 AC`), the resource ABI smoke
    passed device allocation, managed allocation, view, copy, synchronization,
    and invalid-input checks; `nm` confirmed all 16 exported symbols in
    `target/debug/libds4_cuda.a`.
  - Local `cargo test --locked -p ds4-cuda --lib` passed with 87 tests and
    B300 `cargo test --locked -p ds4-cuda --features cuda-oxide-backend --
    --nocapture` passed with 89 tests.
  - `check_cuda_abi_resource_smoke.py --negative-test` passed with 77 checks,
    and the unified parity report passed with 173 passes, 45 skips, and no
    failures.
  - This stage does not export tensor-fill or compute operations and does not
    alter the production linker or default route; `ds4_cuda.cu` remains
    required.

##### M14.6b2: Rust CUDA Compute ABI Assembly

- Status: active; split into M14.6b2a and M14.6b2b so the first linkable
  compute operation is validated independently from graph-kernel ABI assembly.
- Goal: move validated cuda-oxide kernel families into reusable modules and
  export the remaining production operation ABI needed before switching the
  Linux CUDA link away from `ds4_cuda.cu`.

##### M14.6b2a: Rust CUDA Tensor Fill ABI Export

- Status: done.
- Goal: export `ds4_gpu_tensor_fill_f32` from the linkable Rust ABI with
  current-C validation and IEEE-754 bit-fill behavior, without depending on
  executable-local embedded PTX.
- Fixture: `ds4-parity/baselines/backend/m14.6b2a/abi-tensor-fill-smoke.json`.
- Comparator: `ds4-parity/check_cuda_abi_tensor_fill_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now implements `ds4_gpu_tensor_fill_f32`
    through stream-ordered `cuda_core::sys::cuMemsetD32Async` and
    `value.to_bits()`, preserving arbitrary float bit patterns without
    embedding a new library-owned kernel module.
  - On `ds4-rust-port-b300` (`NVIDIA B300 SXM6 AC`), the ABI smoke passed
    prefix fill, tensor-view offset fill, managed fill, signed-zero bit
    preservation, negative-infinity fill, zero-count, bounds, and null-input
    checks; `nm` confirmed 17 exported symbols in `target/debug/libds4_cuda.a`.
  - Local `cargo test --locked -p ds4-cuda --lib` passed with 88 tests and
    B300 `cargo test --locked -p ds4-cuda --features cuda-oxide-backend --
    --nocapture` passed with 90 tests.
  - `check_cuda_abi_tensor_fill_smoke.py --negative-test` passed with 91
    checks, and the unified parity report passed with 174 passes, 45 skips,
    and no failures. The local CUDA-feature build is unavailable on this Mac
    because `/usr/local/cuda/include/cuda.h` is absent; B300 is the execution
    proof.
  - The required non-interactive Claude review was invoked with the
    changed-file, oracle, comparator, and validation evidence bundle, but
    timed out after 60 seconds without a completed result; adversarial
    self-review added managed-tensor coverage and documented the raw-driver
    write-bounds invariant.
  - Graph compute ABI symbols, the production linker switch, and default-route
    promotion remain pending.

##### M14.6b2b: Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b1 and M14.6b2b2 after the first
  reusable embedded-kernel ABI exports required a separately testable static
  library retention contract.
- Goal: export the remaining validated graph compute operation ABI from
  reusable Rust-owned CUDA modules before any production linker or route
  promotion.

##### M14.6b2b1: Rust CUDA Elementwise ABI Module

- Status: done.
- Goal: export `ds4_gpu_add_tensor` and `ds4_gpu_repeat_hc_tensor` from a
  reusable Rust-owned embedded CUDA module while retaining current-C
  input/output aliasing behavior at the C ABI boundary.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b1/abi-elementwise-smoke.json`.
- Comparator: `ds4-parity/check_cuda_abi_elementwise_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi_kernels.rs` now owns cuda-oxide
    `abi_add_kernel` and `abi_repeat_hc_kernel`, with names disjoint from
    executable-local smoke kernels, loaded from the static library's embedded
    `ds4-cuda` module; `abi.rs` submits raw driver launch parameters so valid
    `out == a` addition remains supported.
  - A C consumer fixture links the Rust static library on
    `ds4-rust-port-b300` (`NVIDIA B300 SXM6 AC`) and passes add, in-place
    add, repeated hyperconnection-row output, invalid-range, and null-input
    checks; `nm` confirms 19 `ds4_gpu_*` exports.
  - The static-library embedded artifact has no retaining symbol, so this
    proof links it with `--whole-archive`; the generated embedded object also
    emits a missing `.note.GNU-stack` linker warning that remains a required
    production-integration cleanup before route promotion.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 89 tests and
    B300 feature tests pass with 91 tests. The elementwise ABI comparator and
    unified parity report pass with 175 passed, 45 skipped, and no failures.
  - The required non-interactive Claude review was invoked with the source,
    oracle, comparator, B300 proof, and known-linker-risk bundle, but timed
    out after 60 seconds without a completed result. Adversarial self-review
    fixed the embedded module-name mismatch and duplicate kernel-symbol
    collision, and retained raw launches to preserve valid add aliasing.
  - Remaining graph compute ABI symbols, production linker selection, and
    default-route promotion remain pending.

##### M14.6b2b2: Remaining Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b2a and M14.6b2b2b so the first
  in-place reduction kernel export is proved independently from the remaining
  graph compute ABI.
- Goal: export the remaining validated graph compute symbols from reusable
  Rust-owned modules, and address embedded-artifact production-link
  integration before any route promotion.

##### M14.6b2b2a: Directional Steering ABI Export

- Status: done.
- Goal: export `ds4_gpu_directional_steering_project_tensor` from the
  reusable embedded Rust CUDA ABI module with current-C in-place behavior.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2a/abi-directional-steering-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_directional_steering_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi_kernels.rs` adds uniquely named
    `abi_directional_steering_project_kernel` with the current-C block
    reduction and in-place update; `abi.rs` validates dimensions and submits
    the raw kernel parameter surface.
  - A C consumer of `libds4_cuda.a` on `ds4-rust-port-b300`
    (`NVIDIA B300 SXM6 AC`) passes exact two-row projection output plus
    zero-scale, undersized-direction, and null-input rejection. `nm` confirms
    20 `ds4_gpu_*` exports.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 90 tests and
    B300 feature tests pass with 92 tests. Whole-archive retention and the
    generated embedded object's `.note.GNU-stack` warning remain production
    integration requirements.
  - `check_cuda_abi_directional_steering_smoke.py --negative-test` passes
    with 77 checks, and unified parity passes with 176 passed, 45 skipped,
    and no failures.
  - The required non-interactive Claude review was invoked with the source,
    oracle, comparator, B300 proof, and known-linker-risk bundle, but timed
    out after 60 seconds without a completed result. Adversarial self-review
    retained the overflow-safe bounds checks and verified the raw parameter
    boundary through the passing C-linked executable.
  - Remaining graph compute exports, production linker selection, and
    default-route promotion remain pending.

##### M14.6b2b2b: Remaining Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b2b1 and M14.6b2b2b2 because the first
  nonlinear ABI export requires a reusable embedded-libdevice loading path
  independently from the remaining graph compute ABI.
- Goal: export the remaining graph compute ABI symbols and resolve embedded
  artifact production-link integration before selecting a Rust CUDA route.

##### M14.6b2b2b1: SwiGLU Libdevice ABI Export

- Status: done.
- Goal: export `ds4_gpu_swiglu_tensor` through the reusable embedded Rust
  CUDA module while linking the embedded PTX against libdevice at load time.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b1/abi-swiglu-libdevice-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_swiglu_libdevice_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi_kernels.rs` adds uniquely named
    `abi_swiglu_kernel` and loads embedded PTX through
    `cuda_host::ltoir::build_cubin_from_ptx_with_libdevice` when the bundle
    contains `__nv_*` references; `abi.rs` preserves the current-C
    output/input alias boundary through raw launch parameters.
  - A C consumer of `libds4_cuda.a` on `ds4-rust-port-b300`
    (`NVIDIA B300 SXM6 AC`) passes clamped, unclamped, aliasing,
    invalid-shape, zero-count, and null-input checks. It creates a linked
    process-local `/tmp/ds4-cuda-abi-<pid>/ds4-cuda-abi.sm_103.cubin`,
    removes the extracted/link artifacts after module load, and `nm`
    confirms 21 `ds4_gpu_*` exports.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 91 tests and
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 93 tests. The non-release
    cuda-oxide feature test rejects reusable library
    `std::f32::<impl f32>::exp` device code before libdevice lowering; this
    is recorded as an outstanding
    backend codegen constraint, not omitted.
  - Whole-archive retention and the generated embedded object's
    `.note.GNU-stack` warning remain production integration requirements.
  - `check_cuda_abi_swiglu_libdevice_smoke.py --negative-test` passes with
    96 checks, and unified parity passes with 177 passed, 45 skipped, and no
    failures.
  - The required non-interactive Claude review was invoked with the source,
    oracle, comparator, B300 proof, and known-linker/codegen-risk bundle, but
    timed out after 60 seconds without a completed result. Adversarial
    self-review removed successful runtime link artifacts after module load
    and kept the non-release cuda-oxide codegen failure explicit.
  - Remaining graph compute exports, production linker selection, and
    default-route promotion remain pending.

##### M14.6b2b2b2: Remaining Rust CUDA Kernel ABI Assembly

- Status: active; split into M14.6b2b2b2a and M14.6b2b2b2b because plain
  RMS normalization uses only tensor inputs while weighted RMS and subsequent
  normalization surfaces read weights through the still-unowned
  `model_map`/`cuda_model_range_ptr` ABI boundary.
- Goal: export the remaining graph compute ABI symbols and resolve embedded
  artifact production-link integration before selecting a Rust CUDA route.

##### M14.6b2b2b2a: Plain RMS Norm ABI Export

- Status: done.
- Goal: export `ds4_gpu_rms_norm_plain_tensor` and
  `ds4_gpu_rms_norm_plain_rows_tensor` through the reusable embedded Rust
  CUDA module without claiming model-backed normalization.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2a/abi-plain-rms-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_plain_rms_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi_kernels.rs` adds
    `abi_rms_norm_plain_kernel`; `abi.rs` exports the single-row and batched
    plain-RMS wrappers while preserving output/input aliasing and the
    current-C zero-width success boundary.
  - A C consumer of `libds4_cuda.a` on `ds4-rust-port-b300`
    (`NVIDIA B300 SXM6 AC`) passes single-row, two-row, aliasing,
    undersized-output, zero-row, zero-width, and null-input checks; `nm`
    confirms 23 exported `ds4_gpu_*` symbols.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 92 tests and
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 94 tests.
  - The shared embedded module still carries the preceding non-release
    SwiGLU codegen blocker, whole-archive retention requirement, and generated
    `.note.GNU-stack` linker warning; weighted RMS remains pending on
    model-map ABI ownership rather than being overclaimed here.
  - `check_cuda_abi_plain_rms_smoke.py --negative-test` passes with 94
    checks, and unified parity passes with 178 passed, 45 skipped, and no
    failures.
  - The required non-interactive Claude adversarial review was invoked with
    the source, oracle, comparator, B300 proof, and open boundary/risk bundle,
    but timed out after 60 seconds without a completed result. Adversarial
    self-review preserved the current-C zero-width result boundary and kept
    weighted/model-backed RMS outside this leaf.

##### M14.6b2b2b2b: Weighted RMS And Model-Backed ABI Assembly

- Status: active; split into M14.6b2b2b2b1 and M14.6b2b2b2b2 because the
  weighted RMS calls carry their own model range arguments and can execute
  through a bounded immutable device-copy cache without first claiming the
  full public model-map control and preload policy surface.
- Goal: export model-backed graph-compute symbols while incrementally
  assembling the public model-map control ABI required by a complete route.

##### M14.6b2b2b2b1: Weighted RMS Device-Copy ABI Export

- Status: done.
- Goal: export `ds4_gpu_rms_norm_weight_tensor` and
  `ds4_gpu_rms_norm_weight_rows_tensor` through an internal cached
  device-copy weight-range path without claiming the public model-map
  controls.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b1/abi-weighted-rms-device-copy-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_weighted_rms_device_copy_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` validates the per-call host model range,
    copies immutable requested weight bytes to retained `DeviceBuffer`
    storage, synchronizes the initial upload, and frees cached ranges at
    cleanup; it does not export `ds4_gpu_set_model_map` or related controls.
  - `rust/ds4-cuda/src/abi_kernels.rs` adds
    `abi_rms_norm_weight_kernel`. A C consumer of `libds4_cuda.a` on
    `ds4-rust-port-b300` (`NVIDIA B300 SXM6 AC`) passes single-row,
    two-row, output/input alias, alternate-offset, invalid-range, zero-row,
    current-C zero-width, and null-model checks; `nm` confirms 25 exported
    `ds4_gpu_*` symbols.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 93 tests and
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 95 tests.
  - `check_cuda_abi_weighted_rms_device_copy_smoke.py --negative-test`
    passes with 102 checks, and unified parity passes with 179 passed, 45
    skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the source, C oracle, comparators, B300 proof, and open-risk boundary, but
    timed out after 60 seconds without a completed result. Adversarial
    self-review retained only immutable weight-range device-copy ownership
    and left all public model-map residency controls outside this leaf.
  - Public model-map fd/preload/cache policy, whole-archive artifact
    retention, the generated `.note.GNU-stack` linker warning, and the shared
    module's non-release SwiGLU codegen blocker remain open before route
    promotion.

##### M14.6b2b2b2b2: Public Model-Map Control ABI Assembly

- Status: active; split through M14.6b2b2b2b2a, M14.6b2b2b2b2b1,
  M14.6b2b2b2b2b2a, M14.6b2b2b2b2b2b1, and M14.6b2b2b2b2b2b2
  because baseline public linkage, bounded registered-range fallback,
  deterministic pageable HMM fallback, and successful chunk-selected copy
  can be validated separately from fd-backed and residual policy.
- Goal: export the model-map, file-descriptor, map-range, and cache-range
  control surfaces needed to substitute Rust beneath model-backed runtime
  callers without reducing the current-C residency policy contract.

##### M14.6b2b2b2b2a: Basic Model-Control Device-Copy ABI Export

- Status: done.
- Goal: export `ds4_gpu_set_model_map`, `ds4_gpu_set_model_fd`,
  `ds4_gpu_set_model_map_range`, and `ds4_gpu_cache_model_range` through the
  caller-owned model-map device-copy cache while retaining the optimized
  residency policy branches as explicit follow-on work.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2a/abi-model-control-device-copy-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_device_copy_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` promotes the retained model-range device-copy
    cache to the public control surface. Replacing the active model mapping
    synchronizes submitted work and clears retained cached ranges before
    recording the new map; zero-byte cache requests retain current-C no-op
    success and invalid nonzero ranges fail closed.
  - A C consumer of `libds4_cuda.a` on `ds4-rust-port-b300` (`NVIDIA B300
    SXM6 AC`) pre-caches a weight range through
    `ds4_gpu_cache_model_range`, consumes it through weighted RMS, switches
    model mappings, mutates the old mapping to prove stale cached bytes were
    released, and passes invalid-map/range checks; `nm` confirms 29 exported
    `ds4_gpu_*` symbols.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 94 tests and
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 96 tests.
  - `check_cuda_abi_model_control_device_copy_smoke.py --negative-test`
    passes with 104 checks, and unified parity passes with 180 passed, 45
    skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the source, C oracle, comparator, B300 proof, and open policy boundary,
    but timed out after 60 seconds without a completed result. Adversarial
    self-review found and fixed a control-state/cache cleanup lock-order
    inversion by making model-map replacement acquire the backend first.
  - `ds4_gpu_set_model_fd` is present for ABI linkage but does not yet select
    fd-backed staging. Registered map/HMM/prefetch, direct-I/O, environment
    preload/copy selection, q8/f16 cache hooks, whole-archive retention, and
    the generated `.note.GNU-stack` linker warning remain open.

##### M14.6b2b2b2b2b1: Registered Attempt And Device-Copy Fallback ABI

- Status: done.
- Goal: connect a safely bounded read-only registered caller-map attempt and
  explicit device-copy fallback to the public cached-range ABI without
  claiming pageable HMM, fd-backed staging, remaining graph compute, or route
  promotion.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b1/abi-model-control-registered-fallback-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_registered_fallback_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now retains either a
    `ReadOnlyRegisteredHostMemory` guard with its adjusted requested device
    pointer or the previous `DeviceBuffer` fallback. It only forms the
    page-rounded registered slice when that slice is wholly inside the
    caller-declared model mapping; otherwise it uses device copy.
  - The live registered-range probe on `ds4-rust-port-b300` (`NVIDIA B300
    SXM6 AC`) observes `read_only_registration_error_code=801`,
    `mmap_device_copy_fallback=true`, and `fallback_readback_matches=true`.
    A C-linked consumer then passes a page-aligned model mapping through
    `ds4_gpu_cache_model_range`, consumes it through weighted RMS with matching
    output, and `nm` confirms the 29 exported `ds4_gpu_*` symbols are unchanged.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 95 tests; local
    feature-gated compilation remains blocked by the missing local CUDA header
    at `/usr/local/cuda/include/cuda.h`. B300 `cargo test --locked --release
    -p ds4-cuda --features cuda-oxide-kernels --lib` passes with 97 tests.
  - `check_cuda_abi_model_control_registered_fallback_smoke.py
    --negative-test` passes with 94 checks, and unified parity passes with 181
    passed, 45 skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the source, C oracle, comparator, B300 probe, and public C-linked proof,
    but timed out after 60 seconds without completed findings. Self-review
    retained synchronization before registration release and kept pageable
    HMM, fd staging, and cross-range registration-disable policy outside this
    leaf.
  - Current C disables later registration attempts after selected unsupported
    errors; that optimization, pageable HMM/prefetch, fd-backed direct-I/O,
    preload/copy environment selection, q8/f16 cache hooks, whole-archive
    retention, and the generated `.note.GNU-stack` linker warning remain open.

##### M14.6b2b2b2b2b2: Pageable HMM And Fd-Backed Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2a and M14.6b2b2b2b2b2b
  because deterministic pageable HMM fallback can be validated without
  claiming chunked full-model copy or fd-backed staging.
- Goal: connect the independently validated pageable HMM/prefetch, chunked
  full-model copy, fd-backed direct-I/O staging, registration-disable,
  preload/copy selection, and cache-policy branches to the public control ABI
  without prematurely promoting the runtime route.

###### M14.6b2b2b2b2b2a: Pageable HMM Fallback ABI

- Status: done.
- Goal: connect the deterministic current-C pageable HMM fallback subset to
  the public map-range and direct model-read ABI while retaining chunked
  full-model copy and fd-backed policy as follow-on work.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2a/abi-model-control-pageable-hmm-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_pageable_hmm_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains a page-bounded
    `ReadOnlyPageableHostMemory` guard after the exact deterministic
    current-C fallback selection: chunked copy selected and explicitly
    suppressed by `DS4_CUDA_NO_MODEL_COPY` or `DS4_CUDA_DIRECT_MODEL`, with
    HMM exclusion variables absent. Model-range consumption consults that
    prefetched window before registered or device-copy caching, without
    reporting the prefetched direct pointer as a cache admission.
  - The independent B300 pageable-HMM probe observes pageable access,
    read-mostly/preferred-device advice, successful prefetch, and exact
    direct-pointer readback. A C-linked public consumer selects the fallback
    environment, observes that the direct pointer is not reported as cached,
    consumes the prefetched window through weighted RMS with matching output,
    and preserves the 29-symbol Rust ABI export set.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 96 tests; B300
    `cargo test --locked --release -p ds4-cuda --features cuda-oxide-kernels
    --lib` passes with 98 tests and the release static library rebuilds.
  - `check_cuda_abi_model_control_pageable_hmm_smoke.py --negative-test`
    passes with 107 checks after the cache-result correction, and unified
    parity at the original leaf passes with 182 passed, 45
    skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the HMM subset boundary, C oracle, comparator, and B300 evidence, but
    timed out after 60 seconds without completed findings. Self-review fixed
    a consumption-time policy mismatch by rechecking
    `DS4_CUDA_WEIGHT_CACHE` and `DS4_CUDA_WEIGHT_PRELOAD` before reusing an
    existing prefetched direct window, matching current C.
  - Chunked full-model copy success/allocation-failure handling, global HMM
    direct reads outside the advised window, fd-backed staging,
    registration-disable policy, q8/f16 cache hooks, whole-archive
    retention, and the generated `.note.GNU-stack` warning remain open.

###### M14.6b2b2b2b2b2b: Registration And Fd-Backed Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b1,
  M14.6b2b2b2b2b2b2a, M14.6b2b2b2b2b2b2b1, and
  M14.6b2b2b2b2b2b2b2a, and M14.6b2b2b2b2b2b2b2b because deterministic
  successful chunk-selected copying, whole-map registration precedence,
  buffered fd caching, and synchronous direct-I/O fd caching are separately
  testable from residual failure/cache policy.
- Goal: connect chunked full-model copy/failure routing, fd-backed direct-I/O
  staging, registration-disable, preload/copy selection, and remaining
  cache-policy branches to the public model-control ABI without claiming
  remaining graph compute or route promotion.

####### M14.6b2b2b2b2b2b1: Chunk-Selected Model Copy ABI

- Status: done.
- Goal: connect the deterministic successful `DS4_CUDA_COPY_MODEL_CHUNKED`
  public map-range route through a retained copied device image without
  claiming copy-failure routing or fd-backed policy.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b1/abi-model-control-chunk-selected-copy-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_chunk_selected_copy_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains an `AbiCopiedModel` device image when
    chunk-selected copy is enabled without the current-C suppression
    variables. It copies only the public header plus consumed tensor-data
    window using bounded pinned transfers, and consults that device image
    before pageable or per-range caching.
  - The C-linked B300 consumer mutates its host weights after the initial
    map-range call, invokes map-range again, and still observes the original
    weighted-RMS result. This proves same-map calls reuse the retained copied
    image and do not silently recopy changed host bytes.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 97 tests; B300
    `cargo test --locked --release -p ds4-cuda --features cuda-oxide-kernels
    --lib` passes with 99 tests, the static library rebuilds, and the Rust
    export set remains 29 symbols.
  - `check_cuda_abi_model_control_chunk_selected_copy_smoke.py --negative-test`
    passes with 92 checks, and unified parity passes with 183 passed, 45
    skipped, and no failures.
  - Self-review found and fixed a same-map repeat mismatch: current C retains
    `g_model_device_owned`, while the initial Rust draft would have recopied
    mutated host bytes on a second map-range call.
  - The required non-interactive Claude adversarial review was invoked with
    the copy-selection boundary, C oracle, comparator, and B300 evidence, but
    timed out after 60 seconds without completed findings.
  - Preceding whole-map registration precedence, allocation/transfer-failure
    fallback into HMM, model-copy chunk override and discard/progress side
    effects, unconsumed model ranges, fd-backed staging,
    registration-disable policy, q8/f16 cache hooks, whole-archive retention,
    and the generated `.note.GNU-stack` warning remain open.

####### M14.6b2b2b2b2b2b2a: Whole-Map Registration Precedence ABI

- Status: done.
- Goal: connect current-C whole-map read-only registration attempt and
  registered-pointer precedence to public Rust model control, while
  validating the B300 rejected-registration continuation into chunk-selected
  copied weights without claiming fd-backed policy or route promotion.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2a/abi-model-control-whole-registration-precedence-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_whole_registration_precedence_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains an `AbiRegisteredModel`
    `ReadOnlyRegisteredHostMemory` guard until synchronized replacement or
    cleanup and resolves model ranges from a global registered pointer ahead
    of copied, pageable, and per-range storage.
  - Selection matches current C: a nonempty `DS4_CUDA_COPY_MODEL` value
    suppresses registration, while an empty value still attempts
    registration before chunk-selected range handling.
  - The dedicated B300 registered-range probe reports read-only registration
    rejected with CUDA error code 801. An aligned C-linked public consumer
    selects the empty-copy-model registration attempt plus chunk-selected
    fallback and preserves original weighted-RMS output after host mutation.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 98 tests; B300
    `cargo test --locked --release -p ds4-cuda --features cuda-oxide-kernels
    --lib` passes with 100 tests, the static library rebuilds, and the Rust
    export set remains 29 symbols.
  - `check_cuda_abi_model_control_whole_registration_precedence_smoke.py
    --negative-test` passes with 93 checks, and unified parity passes with
    184 passed, 45 skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the registration-precedence boundary, current-C oracle, comparator, and
    B300 evidence, but returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings. Self-review corrected nonempty-vs-empty
    `DS4_CUDA_COPY_MODEL` selection to match current C.
  - Successful global registered zero-copy is implemented from the current-C
    oracle but is not live-observed on this B300 device. Fd-backed staging,
    residual model-copy failure/cache policy, remaining graph compute,
    whole-archive retention, route promotion, and the generated
    `.note.GNU-stack` warning remain open.

####### M14.6b2b2b2b2b2b2b1: Buffered Fd-Backed Weight Cache ABI

- Status: done.
- Goal: connect the deterministic buffered fd-backed public weight-cache
  subset while leaving direct-I/O reopen, asynchronous staging, budget,
  residual failure/cache policy, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b1/abi-model-control-buffered-fd-cache-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_buffered_fd_cache_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` binds a configured fd to the current or next
    model map and selects a buffered `pread` into CUDA-pinned storage only
    for the bounded `DS4_CUDA_WEIGHT_CACHE=1` plus
    `DS4_CUDA_NO_DIRECT_IO=1` environment. The resulting device range is
    retained ahead of page-bounded registration and caller-map device-copy
    fallback.
  - The aligned C-linked B300 consumer configures the fd before the host
    mapping, gives the file and mapping different weights, and rewrites the
    file after the initial cache call. Weighted RMS continues to observe the
    original fd-backed cached bytes, proving fd precedence and cache reuse.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 99 tests; B300
    `cargo test --locked --release -p ds4-cuda --features cuda-oxide-kernels
    --lib` passes with 101 tests, the static library rebuilds, and the Rust
    export set remains 29 symbols.
  - `check_cuda_abi_model_control_buffered_fd_cache_smoke.py --negative-test`
    passes with 96 checks, and unified parity passes with 185 passed, 45
    skipped, and no failures.
  - Self-review moved fd upload before constructing the per-range registered
    candidate and made interrupted buffered reads retry `EINTR`, matching
    current-C fd-before-range-map and `cuda_pread_full` behavior.
  - The required non-interactive Claude adversarial review was invoked with
    the buffered-fd boundary, current-C oracle, comparator, and B300
    evidence, but returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings.
  - Direct-I/O fd reopen/aligned reads, asynchronous staging, range-arena
    budget, source-page discard/progress, residual model-copy failure/cache
    policy, q8/f16 hooks, remaining graph compute, whole-archive retention,
    route promotion, and the generated `.note.GNU-stack` warning remain
    open.

####### M14.6b2b2b2b2b2b2b2: Direct-I/O And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b because retained direct-fd reopen plus aligned
  synchronous read/fallback behavior is independently testable from
  persistent error-disable, asynchronous staging, budget, and residual
  cache policy.
- Goal: connect public fd-backed direct-I/O staging, asynchronous/budget
  policy, chunk-copy failure routing, and residual model-control
  selection/cache policy without claiming remaining graph compute or route
  promotion.

######## M14.6b2b2b2b2b2b2b2a: Direct-I/O Fd Cache ABI

- Status: done.
- Goal: connect the public retained `O_DIRECT` fd reopen, aligned pinned
  read, and buffered fallback subset while leaving persistent direct-read
  error disablement, asynchronous staging, budgets, residual failure/cache
  policy, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2a/abi-model-control-direct-io-fd-cache-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_io_fd_cache_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` reopens the configured public model fd
    through `/proc/self/fd/<fd>` with `O_DIRECT` on Linux while direct I/O
    is permitted, retains the alignment/file-size state, performs aligned
    pinned reads for in-file windows, and uses buffered fd reads when a
    direct read cannot supply the request. The retained fd-cached device
    range still precedes per-range registration and caller-map copy.
  - The C-linked B300 consumer observes fd-backed weighted RMS and retained
    cache reuse with `DS4_CUDA_NO_DIRECT_IO` unset. The public output does
    not expose whether the successful fd read was direct; the refreshed
    M14.1b2b3b1 B300 probe supplies direct-I/O evidence, reporting
    `direct_io_selected=true`, 4096-byte alignment, aligned read
    offset/size `0`/`8192`, exact readback, and tail buffered fallback.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 100 tests;
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 102 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - `check_cuda_abi_model_control_direct_io_fd_cache_smoke.py
    --negative-test` passes with 111 checks, and unified parity passes with
    186 passed, 45 skipped, and no failures.
  - Persistent direct-read error disablement, asynchronous staging, cache
    budget/source-page/progress policy, residual failure/cache policy,
    q8/f16 hooks, remaining graph compute, whole-archive retention, route
    promotion, and the generated `.note.GNU-stack` warning remain open.
  - The required non-interactive Claude adversarial review was invoked with
    the public direct-I/O fd-cache boundary, current-C oracle, comparator,
    and B300 evidence, but returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`
    without completed findings. Self-review kept the public linked output
    separate from the lower-level direct-read selection evidence.

######## M14.6b2b2b2b2b2b2b2b: Direct-I/O Residual Failure And Cache Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2 because the current-C direct-read disable
  transition and errno classes can be proved independently from public
  asynchronous staging, arena/budget, and remaining cache selection policy.
- Goal: connect persistent direct-read error disablement, asynchronous/budget
  policy, chunk-copy failure routing, and residual model-control
  selection/cache policy without claiming remaining graph compute or route
  promotion.

######### M14.6b2b2b2b2b2b2b2b1: Direct-I/O Error Disable ABI

- Status: done.
- Goal: connect current-C disable-after-selected-direct-read-error state to
  the public Rust fd-cache path without claiming a live induced public error,
  asynchronous staging, budgets, remaining cache policy, or route promotion.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b1/abi-model-control-direct-io-error-disable-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_io_error_disable_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now applies the current-C disabling errno
    classes (`EINVAL`, `EFAULT`, `ENOTSUP`, `EOPNOTSUPP`) to a selected
    direct read: it drops the retained direct file, resets direct alignment,
    and continues through the existing buffered fd fallback.
  - This branch is not reliably inducible through the public B300 filesystem
    harness. The B300 feature test executes the public error-class policy
    check, while the previously linked public direct-enabled fd-cache
    consumer is rerun as a successful-regression gate.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 101 tests;
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 104 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - `check_cuda_abi_model_control_direct_io_error_disable_smoke.py
    --negative-test` passes with 88 checks, and unified parity passes with
    187 passed, 45 skipped, and no failures.
  - Asynchronous staging, arena/cache-budget and source-page/progress
    policy, chunk-copy failure selection, q8/f16 hooks, remaining graph
    compute, whole-archive retention, route promotion, and the generated
    `.note.GNU-stack` warning remain open.
  - The required non-interactive Claude adversarial review was invoked with
    the public direct-read error-disable boundary, current-C oracle,
    comparator, and B300 evidence, but returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
    Self-review retained the explicit non-claim for live public error
    induction and for asynchronous/budget policy.

######### M14.6b2b2b2b2b2b2b2b2: Public Async Staging And Residual Cache Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b because direct-enabled public asynchronous
  staging is independently testable from buffered-only staging, arena/cache
  budget, source-page/progress, and residual selection policy.
- Goal: connect public asynchronous/budget staging, chunk-copy failure
  routing, and residual model-control selection/cache policy without claiming
  remaining graph compute or route promotion.

########## M14.6b2b2b2b2b2b2b2b2a: Public Direct-I/O Async Staging ABI

- Status: done.
- Goal: connect the direct-enabled public fd-cache route to the current-C
  four-slot asynchronous staging loop and model-copy chunk-size clamp while
  leaving buffered-only staging, arena/budget, source-page/progress, residual
  selection, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2a/abi-model-control-direct-io-async-staging-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_io_async_staging_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now parses the current-C
    `DS4_CUDA_MODEL_COPY_CHUNK_MB` clamp, allocates four CUDA-pinned staging
    slots for the direct-enabled public fd-cache path, waits on a prior slot
    event before reuse, records each queued transfer, and synchronizes queued
    work before the retained device range can be consumed.
  - A C-linked B300 consumer selects direct-enabled fd caching with a 16 MiB
    chunk override, requests five chunks, then consumes a small weighted-RMS
    subrange. It observes fd-backed output and retained cached-device reuse
    after rewriting the file. This output intentionally does not claim to
    observe event counts or `O_DIRECT` selection.
  - The independent lower-level M14.1b2b3b2 B300 baseline retains the live
    event-ring observation for the same substrate operations: four staging
    slots, seven queued chunks/events, and two reuse waits.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 102 tests;
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 106 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - `check_cuda_abi_model_control_direct_io_async_staging_smoke.py
    --negative-test` passes with 103 checks, and unified parity passes with
    188 passed, 45 skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the public direct-I/O asynchronous-staging boundary, current-C oracle,
    comparator, and B300 evidence, but returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Buffered-only public asynchronous staging, arena/cache-budget and
    source-page/progress policy, chunk-copy failure selection, q8/f16 hooks,
    remaining graph compute, whole-archive retention, route promotion, and
    the generated `.note.GNU-stack` warning remain open.

########## M14.6b2b2b2b2b2b2b2b2b: Residual Fd Cache And Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2 because buffered-only public asynchronous
  staging is independently testable from arena/cache budget,
  source-page/progress, and residual model-control selection policy.
- Goal: connect buffered-only public asynchronous staging, arena/cache-budget
  and source-page/progress policy, chunk-copy failure routing, and remaining
  model-control selection/cache policy without claiming remaining graph
  compute or route promotion.

########### M14.6b2b2b2b2b2b2b2b2b1: Public Buffered Fd Async Staging ABI

- Status: done.
- Goal: connect the buffered-only public fd-cache route to the shared
  four-slot asynchronous staging loop while leaving arena/budget,
  source-page/progress, residual selection, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b1/abi-model-control-buffered-fd-async-staging-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_buffered_fd_async_staging_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` factors the four-slot event-backed fd uploader
    and routes `DS4_CUDA_WEIGHT_CACHE=1` with `DS4_CUDA_NO_DIRECT_IO=1`
    through buffered `pread` chunks in that shared upload loop.
  - A C-linked B300 consumer sets a 16 MiB chunk override, requests five
    buffered chunks, consumes a small weighted-RMS subrange from fd-sourced
    bytes, and preserves the retained device result after rewriting the
    backing file. The preceding direct-enabled five-chunk public consumer is
    rerun successfully after the shared-uploader refactor.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 103 tests;
    B300 `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 107 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - `check_cuda_abi_model_control_buffered_fd_async_staging_smoke.py
    --negative-test` passes with 103 checks, and unified parity passes with
    189 passed, 45 skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the public buffered fd asynchronous-staging boundary, current-C oracle,
    comparator, and B300 evidence, but returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Arena/cache-budget and source-page/progress policy, chunk-copy failure
    selection, q8/f16 hooks, remaining graph compute, whole-archive
    retention, route promotion, and the generated `.note.GNU-stack` warning
    remain open.

########### M14.6b2b2b2b2b2b2b2b2b2: Arena Budget And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b because public fd arena suballocation and
  lifetime are independently testable from cache-budget fallback,
  source-page/progress, and residual selection policy.
- Goal: connect arena/cache-budget and source-page/progress policy,
  chunk-copy failure routing, and residual model-control selection/cache
  policy without claiming remaining graph compute or route promotion.

############ M14.6b2b2b2b2b2b2b2b2b2a: Public Fd Arena Suballocation ABI

- Status: done.
- Goal: connect public fd-cache ranges to retained arena suballocation and
  synchronized arena reset while leaving cache-budget fallback,
  source-page/progress, residual selection, and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-arena-suballocation-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_arena_suballocation_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains Linux public fd-cache allocations in
    `ABI_MODEL_ARENAS`, parses the current-C
    `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` clamp/growth rule, suballocates
    256-byte-aligned destinations for the asynchronous uploader, and clears
    arenas only after synchronization during model replacement or cleanup.
  - A C-linked B300 consumer selects buffered fd caching with a 256 MiB arena
    override, admits two disjoint ranges, observes each fd-backed weighted
    result over divergent host bytes, and preserves both retained results
    after rewriting the backing file. Its output intentionally does not claim
    internal arena count or device offsets.
  - The existing M14.1b2b3b2 B300 lower-level baseline remains the internal
    arena proof: it observes one device arena serving two admitted ranges.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 104 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 109 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - `check_cuda_abi_model_control_fd_arena_suballocation_smoke.py
    --negative-test` passes with 104 checks, and unified parity passes with
    190 passed, 45 skipped, and no failures.
  - The required non-interactive Claude adversarial review was invoked with
    the public fd-arena boundary, current-C oracle, comparator, and B300
    evidence, but returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings.
  - Cache-budget fallback, source-page/progress policy, residual model-control
    selection, q8/f16 hooks, remaining graph compute, whole-archive
    retention, route promotion, and the generated `.note.GNU-stack` warning
    remain open.

############ M14.6b2b2b2b2b2b2b2b2b2b: Cache Budget And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2 because public fd cache-budget fallback is
  independently testable from source-page/progress and residual selection
  policy.
- Goal: connect public cache-budget fallback, source-page/progress policy,
  chunk-copy failure routing, and residual model-control selection/cache
  policy without claiming remaining graph compute or route promotion.

############# M14.6b2b2b2b2b2b2b2b2b2b1: Public Fd Cache Budget Fallback ABI

- Status: done.
- Goal: connect public fd-cache limit admission and uncached budget fallback
  pointer resolution while leaving source-page/progress, residual selection,
  and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-cache-budget-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_cache_budget_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now retains admitted fd range byte accounting
    with the arena owner, parses the current-C
    `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` GiB/unlimited policy, rejects both raw
    and aligned over-budget reservations before staging, and resolves a
    rejected request to an uncached direct model pointer before constructing
    a host source slice.
  - A C-linked B300 consumer maps only the admitted page as readable, gives
    its fd only that page, admits and consumes a small fd-backed range, then
    successfully rejects repeated 1 GiB cache requests under a 1 GiB limit.
    Success proves no transfer from inaccessible rejected pages; the fixture
    intentionally does not claim compute execution through the returned host
    fallback pointer.
  - The existing M14.1b2b3b2 B300 lower-level baseline remains the
    not-cached fallback proof under an exact small byte budget.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 105 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 111 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - The preceding public fd-arena, buffered asynchronous staging, and
    direct-I/O asynchronous staging C-linked B300 consumers rerun successfully
    against the budget-aware static library.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_cache_budget_smoke.py
    --negative-test` passes with 110 checks, and the default unified parity
    report passes with 191 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Source-page/progress policy, public compute through the returned host
    fallback pointer, residual model-control selection, q8/f16 hooks,
    remaining graph compute, whole-archive retention, route promotion, and
    the generated `.note.GNU-stack` warning remain open.

############# M14.6b2b2b2b2b2b2b2b2b2b2: Source-Page Progress And Residual Model-Control Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b because public fd source-page/progress
  behavior is independently observable from remaining selection policy.
- Goal: connect source-page/progress policy, residual model-control
  selection/cache behavior, and remaining failure routing without claiming
  remaining graph compute or route promotion.

############## M14.6b2b2b2b2b2b2b2b2b2b2a: Public Fd Source-Page And Progress ABI

- Status: done.
- Goal: connect source-file/mapping discard advice and model-load progress
  behavior to public fd-cached uploads while leaving residual model-control
  selection and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-source-page-progress-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_source_page_progress_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now applies file and source-mapping discard
    advice after each enqueued public fd chunk, maintains current-C progress
    emission/suppression state, and resets that state with synchronized public
    model replacement or cleanup.
  - A C-linked B300 consumer interposes both POSIX advice calls and captures
    stderr around two ordinary uploads plus one suppressed upload. It observes
    two advice calls for each ordinary multi-chunk range, an initial non-TTY
    progress line after each model replacement, and no advice/progress when
    `DS4_CUDA_KEEP_MODEL_PAGES` plus verbose mode are selected.
  - The consumer observes advisory invocation rather than physical page
    eviction and does not claim TTY refresh rendering.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 106 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 112 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - The preceding public budget, fd-arena, buffered asynchronous staging, and
    direct-I/O asynchronous staging C-linked B300 consumers rerun successfully
    against the source-page/progress-aware static library.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_source_page_progress_smoke.py
    --negative-test` passes with 122 checks, and the default unified parity
    report passes with 192 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Residual model-control selection, public fallback compute, q8/f16 hooks,
    remaining graph compute, whole-archive retention, route promotion, and
    the generated `.note.GNU-stack` warning remain open.

############## M14.6b2b2b2b2b2b2b2b2b2b2b: Residual Model-Control Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2b2 because cross-range registration disablement
  and reset are independently observable from remaining failure selection.
- Goal: connect residual model-control selection/cache behavior and remaining
  failure routing without claiming remaining graph compute or route promotion.

############### M14.6b2b2b2b2b2b2b2b2b2b2b1: Public Cross-Range Registration Disable ABI

- Status: done.
- Goal: preserve current-C per-range read-only registration disablement after
  selected errors and reset it on public model replacement while leaving
  remaining failure selection and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-registration-disable-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_registration_disable_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains a model-lifetime registration gate,
    disables it only after `CUDA_ERROR_NOT_SUPPORTED` or
    `CUDA_ERROR_INVALID_VALUE`, and re-enables it during public map
    replacement or cleanup.
  - A C-linked B300 consumer interposes `cuMemHostRegister_v2` with error
    code 801. It observes one whole-map and one first-range attempt, no
    attempt for a second disjoint range after disablement, and two new
    attempts after replacing the active map.
  - The consumer verifies weighted RMS results through the device-copy
    fallback; it does not claim a successful zero-copy registration path.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 107 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 114 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - The preceding public registered-fallback, whole-map-registration,
    fd-budget, and fd source-page/progress C-linked B300 consumers rerun
    successfully against the registration-disable-aware static library.
  - `python3
    ds4-parity/check_cuda_abi_model_control_registration_disable_smoke.py
    --negative-test` passes with 102 checks, and the default unified parity
    report passes with 193 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Remaining failure selection, q8/f16 hooks, remaining graph compute,
    whole-archive retention, route promotion, and the generated
    `.note.GNU-stack` warning remain open.

############### M14.6b2b2b2b2b2b2b2b2b2b2b2: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b because successful nonempty full-model copy
  selection is independently observable from remaining failure selection.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

################ M14.6b2b2b2b2b2b2b2b2b2b2b2a: Public Full-Model Copy Selection ABI

- Status: done.
- Goal: preserve current-C successful nonempty `DS4_CUDA_COPY_MODEL`
  selection, copied-image retention, and copy-failure continuation wiring
  while leaving live copy-failure observation and remaining selection pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-full-model-copy-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_full_model_copy_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now routes a nonempty full-model copy
    selection through a whole-map copied device image before attempting
    registration, while a failed copy continues to whole-map registration.
  - A C-linked B300 consumer selects full copy, interposes registration,
    mutates host weights after setup, replaces the model map, and verifies
    weighted RMS reads from each retained copied image with zero registration
    calls.
  - Full-copy allocation or transfer failure continuation is source-backed
    but not forced in the live consumer.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 108 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 115 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - The preceding chunk-selected-copy, whole-map-registration,
    registration-disable, fd-budget, and fd source-page/progress C-linked
    B300 consumers pass against the full-copy-aware static library. The two
    fd-only fixtures now leave full-model copy unselected to match current-C
    precedence.
  - `python3
    ds4-parity/check_cuda_abi_model_control_full_model_copy_smoke.py
    --negative-test` passes with 94 checks, and the default unified parity
    report passes with 194 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Remaining failure selection, q8/f16 hooks, remaining graph compute,
    whole-archive retention, route promotion, and the generated
    `.note.GNU-stack` warning remain open.

################ M14.6b2b2b2b2b2b2b2b2b2b2b2b: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2 because nonempty direct-model read
  selection and the uncached cache-result boundary are independently
  observable from remaining failure selection.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

################# M14.6b2b2b2b2b2b2b2b2b2b2b2b1: Public Direct-Model Read Selection ABI

- Status: done.
- Goal: preserve current-C nonempty `DS4_CUDA_DIRECT_MODEL` host-read
  selection and its uncached public cache result while leaving remaining
  failure selection and route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-direct-model-read-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_direct_model_read_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now resolves nonempty direct-model reads from
    the caller mapping after global copied/registered state but before
    pageable or per-range staging, while public cache calls report only
    actual retained storage.
  - A C-linked B300 consumer interposes whole-map registration with error
    code 801, selects direct-model reads, observes no range-cache admission
    or registration retry, mutates host weights, and verifies that weighted
    RMS observes the changed bytes.
  - The corrected pageable-HMM predecessor consumer verifies that its
    prefetched direct pointer remains usable for weighted RMS but is not
    reported as a cache admission.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 109 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 116 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - The preceding full-model-copy C-linked B300 consumer reruns successfully
    against the direct-model-aware static library.
  - `python3
    ds4-parity/check_cuda_abi_model_control_direct_model_read_smoke.py
    --negative-test` passes with 88 checks, and the default unified parity
    report passes with 195 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Remaining failure selection, q8/f16 hooks, remaining graph compute,
    whole-archive retention, route promotion, and the generated
    `.note.GNU-stack` warning remain open.

################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b because public bound-fd selection without
  the weight-cache flag and its preload/disable boundaries are independently
  observable from remaining failure selection.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2a: Public Default Fd Selection ABI

- Status: done.
- Goal: preserve current-C bound-fd selection when `DS4_CUDA_WEIGHT_CACHE` is
  absent, retain fd selection under `DS4_CUDA_WEIGHT_PRELOAD`, and honor
  `DS4_CUDA_NO_FD_CACHE` bypass while leaving remaining failure selection and
  route promotion pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-default-fd-selection-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_default_fd_selection_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now selects a configured model fd whenever fd
    caching is not disabled and direct-host bypass is not selected, without
    incorrectly requiring `DS4_CUDA_WEIGHT_CACHE`.
  - A C-linked B300 consumer interposes whole-map registration with error
    code 801 and verifies file-backed weighted RMS output with the
    weight-cache flag absent and with preload set, then verifies
    `DS4_CUDA_NO_FD_CACHE` produces fallback host-weight output with the
    expected range-registration attempt.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 110 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 117 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - The preceding buffered-fd, direct-model, pageable-HMM, and
    full-model-copy C-linked B300 consumers rerun successfully against the
    default-fd-aware static library.
  - `python3
    ds4-parity/check_cuda_abi_model_control_default_fd_selection_smoke.py
    --negative-test` passes with 91 checks, and the default unified parity
    report passes with 196 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Remaining failure selection, q8/f16 hooks, remaining graph compute,
    whole-archive retention, route promotion, and the generated
    `.note.GNU-stack` warning remain open.

################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1 and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2 because the public uncached return value
  and compute use of fd-budget host fallback are independently observable
  from arena-allocation and strict-cache failure selection.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b1: Public Fd Budget Fallback Cache-Result ABI

- Status: done.
- Goal: preserve current-C uncached public cache-result behavior for raw fd
  byte-budget fallback and observe weighted compute through the returned host
  pointer while leaving arena/strict-cache failure selection pending.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b1/abi-model-control-fd-budget-cache-result-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_budget_cache_result_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now makes `ds4_gpu_cache_model_range` report
    retained cache membership after range resolution, so an fd-budget host
    fallback returns uncached, and releases the retained-range mutex before
    invoking callbacks.
  - A C-linked B300 consumer admits a near-one-GiB fd-backed range, triggers
    a small over-budget fallback with divergent host/file weights, observes
    an uncached public result, and verifies weighted RMS consumes host bytes.
  - The corrected preceding fd-budget consumer now requires oversized
    fallback requests to report uncached; default-fd, direct-model,
    pageable-HMM, and full-model-copy linked consumers pass against the
    rebuilt archive.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 111 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 118 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_budget_cache_result_smoke.py
    --negative-test` passes with 101 checks, and the default unified parity
    report passes with 197 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Arena allocation failure, persistent cache-full state,
    `DS4_CUDA_STRICT_WEIGHT_CACHE` continuation, route promotion, and the
    generated `.note.GNU-stack` warning remain open.

################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2b because fd arena allocation failure,
  strict continuation, and persistent cache-full behavior are independently
  observable from remaining staging/read/copy failure selection.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

#################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a: Public Fd Arena Failure Selection ABI

- Status: done.
- Goal: preserve current-C fd arena allocation-failure routing, including
  non-strict uncached host fallback, strict continuation into cached fallback
  handling, and persistent no-retry state after allocation failure.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2a/abi-model-control-fd-arena-failure-selection-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_arena_failure_selection_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now separates raw fd-budget fallback from
    arena failure, stores the model-lifetime cache-full state, routes arena
    failure through `DS4_CUDA_STRICT_WEIGHT_CACHE`, and checks aligned arena
    budget after reuse of an existing arena.
  - `rust/ds4-cuda/src/substrate.rs` exposes uninitialized byte allocation
    for fd arenas, so this branch corresponds to device allocation rather
    than an unrelated zero-fill operation.
  - A C-linked B300 consumer interposes the 256 MiB Rust fd-arena allocation
    with out-of-memory, proves uncached host-weight output in non-strict mode,
    proves cached device-copy output in strict mode, and proves that a second
    range does not retry allocation after the latched failure.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 112 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 119 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Successful fd-arena suballocation, fd-budget cache-result, default-fd,
    and registration-disable C-linked B300 consumers pass against the rebuilt
    archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_arena_failure_selection_smoke.py
    --negative-test` passes with 107 checks, and the default unified parity
    report passes with 198 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Aligned-budget strict routing is source-backed but not independently
    forced live; staging allocation/read/copy failure selection, route
    promotion, and the generated `.note.GNU-stack` warning remain open.

#################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2b: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bb because a failed selected direct-fd
  upload must continue to fallback handling without retrying through a
  buffered fd upload.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

##################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba: Public Fd Upload Failure Continuation ABI

- Status: done.
- Goal: preserve current-C one-attempt fd failure routing so a failed
  selected fd upload continues into registration or device-copy fallback
  without launching a second buffered fd upload.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2ba/abi-model-control-fd-upload-failure-continuation-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_upload_failure_continuation_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now selects exactly one configured fd upload
    branch before matching its result. The direct-I/O branch already owns its
    internal buffered read fallback, so an attempted upload failure falls
    through to registration/device-copy handling once.
  - A C-linked B300 consumer selects the direct fd branch, injects one
    `cuMemcpyHtoDAsync_v2` failure after tensor setup, rejects range
    registration, and distinguishes host-byte device-copy fallback from an
    incorrect retry using divergent fd bytes.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 113 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 120 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Fd-arena failure, fd-budget cache-result, default-fd, direct-I/O
    asynchronous-staging, and registration-disable C-linked B300 consumers
    pass against the rebuilt archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_upload_failure_continuation_smoke.py
    --negative-test` passes with 102 checks, and the default unified parity
    report passes with 199 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Staged allocation, fd-read, event-record, event-wait, and final
    synchronization failure observations are not independently forced live;
    route promotion and the generated `.note.GNU-stack` warning remain open.

##################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbb because the public Rust fd path must
  retain the four-slot staging pool across cached ranges before remaining
  failure observations can be isolated.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

###################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba: Public Fd Stage Pool Reuse ABI

- Status: done.
- Goal: preserve current-C public fd stage-pool lifetime so successful range
  uploads retain and reuse four pinned staging slots until cleanup.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bba/abi-model-control-fd-stage-pool-reuse-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_stage_pool_reuse_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` now owns a Linux-only public stage pool
    independently from fd arenas, keeps sufficient slots across later range
    uploads and model-map replacement, and releases them during public
    cleanup.
  - A C-linked B300 consumer selects buffered fd caching, establishes one
    four-slot pool, then arms `cuMemAllocHost_v2` failure before caching a
    second disjoint range. File-backed output from the second range proves
    no fresh stage allocation or registration fallback occurs after reuse.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 114 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 121 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Fd-upload failure continuation, fd-arena failure, fd-budget cache-result,
    default-fd, direct-I/O asynchronous-staging, and registration-disable
    C-linked B300 consumers pass against the rebuilt archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_stage_pool_reuse_smoke.py
    --negative-test` passes with 105 checks, and the default unified parity
    report passes with 200 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Initial stage allocation and pool-growth failure, fd-read, event, and
    final synchronization observations remain separate; route promotion and
    the generated `.note.GNU-stack` warning remain open.

###################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbb because stage allocation failure
  continuation can be live-observed independently from fd read, event, and
  final synchronization failure paths.
- Goal: connect remaining model-control failure selection without claiming
  remaining graph compute or route promotion.

####################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba: Public Fd Stage Allocation Failure Continuation ABI

- Status: done.
- Goal: preserve current-C public fd stage-allocation failure continuation so
  failed pinned staging retries on later ranges and continues through cached
  host fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbba/abi-model-control-fd-stage-allocation-failure-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_stage_allocation_failure_smoke.py --negative-test`.
- Evidence:
  - Existing `rust/ds4-cuda/src/abi.rs` propagation already maps failed
    `pinned_zeroed` stage allocation to the public registration/device-copy
    fallback path without consulting `DS4_CUDA_STRICT_WEIGHT_CACHE`; this leaf
    records that boundary without changing routing code.
  - A C-linked B300 consumer selects buffered fd caching, forces
    `cuMemAllocHost_v2` failure before two disjoint ranges separated by a
    strict-mode transition, rejects the first range-registration attempt, and
    proves both ranges retain host-backed cached output rather than fd bytes.
    The second allocation attempt also proves staging failure does not latch
    fd arena cache-full state, while the single registration attempt preserves
    the previously owned range-registration-disable policy.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 115 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 122 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Stage-pool reuse, fd-upload failure continuation, fd-arena failure,
    fd-budget cache-result, default-fd, direct-I/O asynchronous-staging, and
    registration-disable C-linked B300 consumers pass against the rebuilt
    archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_stage_allocation_failure_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    201 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Fd-read, event, and final synchronization failure observations remain
    separate; route promotion and the generated `.note.GNU-stack` warning
    remain open.

####################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbb because buffered fd-read failure
  continuation can be forced independently from event and final
  synchronization failure paths.
- Goal: connect remaining fd-read, event, and final synchronization failure
  selection without claiming remaining graph compute or route promotion.

######################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba: Public Fd Read Failure Continuation ABI

- Status: done.
- Goal: preserve current-C public buffered fd-read failure continuation so
  failed staged reads retry on later ranges and continue through cached host
  fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbba/abi-model-control-fd-read-failure-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_read_failure_smoke.py --negative-test`.
- Evidence:
  - Existing `rust/ds4-cuda/src/abi.rs` read propagation already maps failed
    buffered `pread` into public registration/device-copy fallback without
    consulting `DS4_CUDA_STRICT_WEIGHT_CACHE`; this leaf records that
    boundary without changing routing code.
  - A C-linked B300 consumer selects buffered fd caching, injects `EIO` from
    `pread` only for the configured model fd before two disjoint ranges
    separated by a strict-mode transition, rejects the first
    range-registration attempt, and proves both ranges retain host-backed
    cached output rather than fd bytes. Two observed reads prove failure is
    retried without latching fd arena cache-full state; one registration
    attempt preserves the previously owned registration-disable policy.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 116 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 123 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Stage-allocation failure, stage-pool reuse, fd-upload failure
    continuation, fd-arena failure, fd-budget cache-result, default-fd,
    direct-I/O asynchronous-staging, and registration-disable C-linked B300
    consumers pass against the rebuilt archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_read_failure_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    202 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Event and final synchronization failure observations remain separate;
    route promotion and the generated `.note.GNU-stack` warning remain open.

######################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba and
  M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbb because event-record failure
  continuation can be forced independently from event-wait and final
  stream-synchronization failure paths.
- Goal: connect remaining event and final synchronization failure selection
  without claiming remaining graph compute or route promotion.

######################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba: Public Fd Event Record Failure Continuation ABI

- Status: done.
- Goal: preserve current-C public fd event-record failure continuation so
  failed staging records retry on later ranges and continue through cached
  host fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbba/abi-model-control-fd-event-record-failure-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_event_record_failure_smoke.py --negative-test`.
- Evidence:
  - Existing `rust/ds4-cuda/src/abi.rs` event-record propagation already maps
    failed `backend.record_event()` through the selected fd upload into public
    registration/device-copy fallback without consulting
    `DS4_CUDA_STRICT_WEIGHT_CACHE`; cuda-oxide maps that call to
    `cuEventRecord`, so this leaf records the boundary without changing routing
    code.
  - A C-linked B300 consumer selects buffered fd caching, forwards
    `cuEventRecord` through setup, then injects record failure before two
    disjoint ranges separated by a strict-mode transition, rejects the first
    range-registration attempt, and proves both ranges retain host-backed
    cached output rather than fd bytes. Two observed records prove failure is
    retried without latching fd arena cache-full state; one registration
    attempt preserves the previously owned registration-disable policy.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 117 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 124 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Fd-read failure, stage-allocation failure, stage-pool reuse, fd-upload
    failure continuation, fd-arena failure, fd-budget cache-result, default-fd,
    direct-I/O asynchronous-staging, and registration-disable C-linked B300
    consumers pass against the rebuilt archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_event_record_failure_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    203 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Event-wait and final synchronization failure observations remain separate;
    route promotion and the generated `.note.GNU-stack` warning remain open.

######################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbb because event-wait
  failure continuation can be forced independently from final
  stream-synchronization failure.
- Goal: connect remaining event-wait and final synchronization failure
  selection without claiming remaining graph compute or route promotion.

########################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba: Public Fd Event Wait Failure Continuation ABI

- Status: done.
- Goal: preserve current-C public fd event-wait failure continuation so failed
  staging-slot reuse retries on later ranges and continues through cached host
  fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbba/abi-model-control-fd-event-wait-failure-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_event_wait_failure_smoke.py --negative-test`.
- Evidence:
  - Existing `rust/ds4-cuda/src/abi.rs` event-wait propagation already maps
    failed `event.synchronize()` through selected fd upload into public
    registration/device-copy fallback without consulting
    `DS4_CUDA_STRICT_WEIGHT_CACHE`; this leaf records the boundary without
    changing routing code.
  - A C-linked B300 consumer selects buffered fd caching, requests two ranges
    exceeding the four-slot event ring, injects `cuEventSynchronize` failure
    on fifth-chunk slot reuse across a strict-mode transition, rejects the
    first range-registration attempt, and proves both ranges retain
    host-backed cached output rather than fd bytes.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 118 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 125 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Event-record failure, fd-read failure, stage-allocation failure,
    stage-pool reuse, fd-upload failure continuation, fd-arena failure,
    fd-budget cache-result, default-fd, direct-I/O asynchronous-staging, and
    registration-disable C-linked B300 consumers pass against the rebuilt
    archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_event_wait_failure_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    204 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Final stream-synchronization failure observation remains separate; route
    promotion and the generated `.note.GNU-stack` warning remain open.

########################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbb: Remaining Residual Failure Selection Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbb because final
  stream-synchronization failure can be forced independently from remaining
  graph-compute and route-promotion work.
- Goal: connect remaining final stream-synchronization failure selection
  without claiming remaining graph compute or route promotion.

########################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba: Public Fd Final Sync Failure Continuation ABI

- Status: done.
- Goal: preserve current-C public fd final upload synchronization failure
  continuation so failed completed staging attempts retry on later ranges and
  continue through cached host fallback independently of strict fd-cache mode.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbba/abi-model-control-fd-final-sync-failure-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_model_control_fd_final_sync_failure_smoke.py --negative-test`.
- Evidence:
  - Existing `rust/ds4-cuda/src/abi.rs` final synchronization handling already
    maps failed `backend.synchronize()` through selected fd upload into public
    registration/device-copy fallback without consulting
    `DS4_CUDA_STRICT_WEIGHT_CACHE`; this leaf records the boundary without
    changing routing code.
  - A C-linked B300 consumer selects buffered fd caching, injects one
    `cuStreamSynchronize` failure per staged fd attempt across a strict-mode
    transition, forwards subsequent synchronization so device-copy fallback
    can complete, rejects the first range-registration attempt, and proves
    both ranges retain host-backed cached output rather than fd bytes.
  - Local `cargo test --locked -p ds4-cuda --lib` passes with 119 tests; B300
    `cargo test --locked --release -p ds4-cuda --features
    cuda-oxide-kernels --lib` passes with 126 tests, the static library
    rebuilds, and the Rust export set remains 29 symbols.
  - Event-wait failure, event-record failure, fd-read failure,
    stage-allocation failure, stage-pool reuse, fd-upload failure
    continuation, fd-arena failure, fd-budget cache-result, default-fd,
    direct-I/O asynchronous-staging, and registration-disable C-linked B300
    consumers pass against the rebuilt archive.
  - `python3
    ds4-parity/check_cuda_abi_model_control_fd_final_sync_failure_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    205 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Q8/f16 hooks, remaining graph compute, route promotion, and the generated
    `.note.GNU-stack` warning remain open.

########################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbb because the public
  single-token F16 projection ABI can be linked and observed independently
  from multi-token BLAS, paired/Q8 compute, and production route promotion.
- Goal: connect remaining q8/f16 hooks, graph compute, whole-archive
  retention, and production route-promotion work without claiming C CUDA
  removal before those gates pass.

############################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba: Public Single-Token F16 Projection ABI

- Status: done.
- Goal: Rust-own the public single-token F16 dense-projection ABI through the
  current-C base, ordered-chunks, and serial dispatch boundary without
  claiming multi-token BLAS, paired projection, Q8 cache, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbba/abi-matmul-f16-single-token-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f16_single_token_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports `ds4_gpu_matmul_f16_tensor` for
    `n_tok == 1`, resolves F16 weights through the cached model-range path,
    and delegates to embedded base, ordered-chunks, or serial kernels.
  - A C-linked B300 consumer verifies default ordered-chunks, forced base,
    and forced serial output, proves cached F16 weights survive host-map
    mutation, and rejects unowned multi-token BLAS plus invalid ranges.
  - Local library tests pass with 120 tests; B300 release-feature tests pass
    with 127 tests; the static library now exposes 30 Rust ABI symbols.
  - Fourteen predecessor C-linked consumers pass against the rebuilt archive;
    each relink continues to emit the known generated embedded-object
    `.note.GNU-stack` executable-stack warning.
  - `python3 ds4-parity/check_cuda_abi_matmul_f16_single_token_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    206 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Multi-token/paired F16 projection, Q8/F16 cache hooks, remaining graph
    compute, whole-archive retention policy, route promotion, and C CUDA
    removal remain open.

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

- Status: done.
- Goal: Rust-own public single-token paired F16 projection through the
  current-C paired ordered-chunks dispatch and its independent fallback
  selections without claiming multi-token BLAS, Q8/F16 cache, or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbba/abi-matmul-f16-pair-single-token-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f16_pair_single_token_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports `ds4_gpu_matmul_f16_pair_tensor` for
    `n_tok == 1`, selecting a new embedded paired ordered-chunks kernel by
    default and delegating forced fallbacks through the owned single-token
    F16 projection ABI.
  - A C-linked B300 consumer verifies default paired output, no-pair,
    no-ordered, and serial independent fallback output; both cached F16
    weights survive host-map mutation, while multi-token paired projection
    and invalid ranges are rejected.
  - Local library tests pass with 121 tests; B300 release-feature tests pass
    with 128 tests; the static library now exposes 31 Rust ABI symbols, and
    fifteen predecessor C-linked consumers, including the single-token
    projection leaf, pass against the rebuilt archive. Each predecessor
    relink retains the known embedded-object executable-stack warning.
  - `python3 ds4-parity/check_cuda_abi_matmul_f16_pair_single_token_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    207 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Multi-token F16 BLAS projection, Q8/F16 cache hooks, remaining graph
    compute, whole-archive retention policy, route promotion, C CUDA removal,
    and the generated embedded-object executable-stack warning remain open.

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

- Status: done.
- Goal: Rust-own public single-token F32 projection through the current-C
  base-kernel dispatch without claiming multi-token BLAS, Q8/F16 cache, or
  route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbba/abi-matmul-f32-single-token-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f32_single_token_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports `ds4_gpu_matmul_f32_tensor` for
    `n_tok == 1`, resolves cached F32 model weights, and launches a new
    embedded current-C-equivalent base reduction kernel.
  - A C-linked B300 consumer verified single-token F32 output and cached
    F32 weights after host-map mutation while rejecting the then-unowned
    multi-token BLAS boundary; that negative observation is historical after
    the successor F32 BLAS leaf below.
  - Local library tests pass with 122 tests; B300 release-feature tests pass
    with 129 tests; the static library exposes 32 Rust ABI symbols, and
    sixteen predecessor C-linked consumers pass against the rebuilt archive,
    retaining the known embedded-object executable-stack warning.
  - `python3 ds4-parity/check_cuda_abi_matmul_f32_single_token_smoke.py
    --negative-test` passes, and the default unified parity report passes with
    208 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Multi-token F16/F32 BLAS projection, Q8/F16 cache hooks, remaining graph
    compute, whole-archive retention policy, route promotion, C CUDA removal,
    and the generated embedded-object executable-stack warning remain open.

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

- Status: done.
- Goal: Rust-own public multi-token F32 projection through the current-C
  `cublasSgemm` boundary without claiming F16 BLAS, Q8/F16 cache,
  quality-mode mutation, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbba/abi-matmul-f32-multi-token-blas-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f32_multi_token_blas_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains the one-token embedded F32 base
    kernel and dispatches `n_tok > 1` through the cuda-oxide `project_f32`
    cuBLAS adapter, with the current-C initialization-time default TF32
    versus `DS4_CUDA_NO_TF32` math selection.
  - A C-linked B300 consumer verifies both one-token base output and
    two-token BLAS output after host-map mutation and after unsetting an
    initialization-time `DS4_CUDA_NO_TF32`, proving the cached F32 model
    range and math selection remain authoritative across the widened public
    symbol.
  - Local library tests pass with 123 tests; B300 release-feature tests pass
    with 130 tests; the static library remains at 32 Rust ABI symbols, and
    sixteen unaffected predecessor C-linked consumers pass against the
    rebuilt archive, retaining the known embedded-object executable-stack
    warning. The historical single-token F32 negative witness is not rerun
    because its rejected boundary is now owned by this successor.
  - The focused comparator passes, and the default unified parity report
    passes with 209 passed, 45 skipped, and 0 failed.
  - The required non-interactive Claude review returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without completed findings.
  - Multi-token F16 BLAS projection, Q8/F16 cache hooks, quality-mode
    mutation, remaining graph compute, whole-archive retention policy, route
    promotion, C CUDA removal, and the generated embedded-object
    executable-stack warning remain open.

############################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbb because public
  multi-token F16 BLAS projection and paired delegation can be proved
  independently from Q8/F16 cache hooks, quality mutation, and route
  promotion.
- Goal: connect multi-token F16 BLAS projection, q8/f16 cache hooks,
  quality-mode mutation, remaining graph compute, whole-archive retention
  policy, and production route-promotion work without claiming C CUDA
  removal before those gates pass.

################################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba: Public Multi-Token F16 BLAS Projection ABI

- Status: done.
- Goal: Rust-own public multi-token F16 projection and paired delegation
  through the current-C F32-to-F16 activation conversion plus
  `cublasGemmEx` boundary without claiming Q8/F16 cache, quality-mode
  mutation, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbba/abi-matmul-f16-multi-token-blas-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_matmul_f16_multi_token_blas_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains a synchronously allocated ABI-owned
    F16 activation scratch buffer, synchronizes before replacing it, and dispatches multi-token
    F16 projection through the cuda-oxide `project_f16_f32` adapter while
    preserving the current-C serial fallback. `ds4_gpu_matmul_f16_pair_tensor`
    now permits multi-token independent delegation through that same owner.
  - `rust/ds4-cuda/src/abi_kernels.rs` exposes the reusable
    `abi_f32_to_f16_kernel` required by the staticlib path; this promotes the
    previously validated executable-local conversion proof into public ABI
    execution.
  - A C-linked B300 consumer proves one-token and paired predecessors,
    multi-token F16 BLAS output, paired multi-token delegation, cached F16
    weight authority after host-map mutation, half-rounding of a
    non-representable F32 activation on the BLAS route, and retained serial
    multi-token F32 activation behavior.
  - Local library tests pass with 124 tests; B300 release-feature tests pass
    with 131 tests; the static library remains at 32 Rust ABI symbols, and
    fifteen unaffected predecessor C-linked consumers pass against the
    rebuilt archive, retaining the known embedded-object executable-stack
    warning. The earlier single-token F16 rejection witnesses are historical
    after this successor.
  - The focused comparator passes, and the default unified parity report
    passes with 210 passed, 45 skipped, and 0 failed.
  - Pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings.
  - Q8/F16 cache hooks, quality-mode mutation, remaining graph compute,
    whole-archive retention policy, route promotion, C CUDA removal, and the
    generated embedded-object executable-stack warning remain open.

################################ M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbb because the public
  Q8 preload, memory-report, and quality controls can be proved independently
  from Q8 matmul consumers and remaining graph route work.
- Goal: connect q8/f16 cache hooks, quality-mode mutation, remaining graph
  compute, whole-archive retention policy, and production route-promotion
  work without claiming C CUDA removal before those gates pass.

################################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba: Public Q8 Cache And Quality Controls ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_cache_q8_f16_range`,
  `ds4_gpu_print_memory_report`, and `ds4_gpu_set_quality` through the
  current-C Q8 converted-preload and mutable cuBLAS math-selection boundary
  without claiming Q8 matmul or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbba/abi-q8-quality-controls-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_q8_quality_controls_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` retains ABI-owned Q8 F16 and optional F32
    converted buffers, applies existing Q8 admission policy to the public
    preload hook, exports the memory diagnostic, and records a mutable
    quality/default-math state consumed by existing dense BLAS projections.
  - `rust/ds4-cuda/src/abi_kernels.rs` promotes
    `abi_dequant_q8_0_to_f16_kernel` and `abi_dequant_q8_0_to_f32_kernel`
    into the embedded public ABI module.
  - A C-linked B300 consumer observes F16 preload allocation and exact-cache
    reuse, quality suppression and re-enable allocation, optional F32
    preload allocation, callable memory reporting, and quality versus
    `DS4_CUDA_NO_TF32` selection through multi-token F32 BLAS output.
  - Local library tests pass with 125 tests; B300 release-feature tests pass
    with 132 tests; the static library exposes 35 Rust ABI symbols, and
    sixteen predecessor C-linked consumers pass against the rebuilt archive,
    retaining the known embedded-object executable-stack warning.
  - The focused comparator passes, and the default unified parity report
    passes with 211 passed, 45 skipped, and 0 failed.
  - Pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings.
  - Q8 matmul compute exports, remaining graph compute, whole-archive
    retention policy, route promotion, C CUDA removal, and the generated
    embedded-object executable-stack warning remain open.

################################# M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbb because the base
  public Q8 matmul consumer can be compared independently from specialized
  pair/HC graph consumers and route work.
- Goal: connect public Q8 matmul consumers, remaining graph compute,
  whole-archive retention policy, and production route-promotion work
  without claiming C CUDA removal before those gates pass.

################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba: Public Q8 Matmul ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_matmul_q8_0_tensor` through the current-C Q8
  expanded BLAS and native prequantized dispatch boundary without claiming
  specialized Q8 consumers or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbba/abi-q8-matmul-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_q8_matmul_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the public Q8 matmul entry point,
    consumes retained F32/F16 converted buffers for multi-token BLAS, and
    retains ABI-owned quantized activation scratch across native Q8 launches.
  - `rust/ds4-cuda/src/abi_kernels.rs` promotes activation quantization and
    native warp8, batch-warp8, and generic matmul kernels, with CUDA-Oxide
    DP4A for full blocks and scalar fallback selected by
    `DS4_CUDA_NO_Q8_DP4A`.
  - A C-linked B300 consumer proves default DP4A and its scalar-disable
    override, batch-warp and generic override dispatch, F16 and F32 expanded
    BLAS opt-in routes, and invalid-range rejection.
  - Local library tests pass with 126 tests; B300 release-feature tests pass
    with 133 tests; the static library exposes 36 Rust ABI symbols; all 36
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 40 CUDA ABI comparators pass, and the unified parity report passes
    with 212 passed, 45 skipped, and 0 failed.
  - Pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings.
  - Specialized public Q8 pair/HC consumers, remaining graph compute,
    whole-archive retention policy, route promotion, C CUDA removal, and the
    generated embedded-object executable-stack warning remain open.

################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbb because the public
  HC expansion kernel family is a bounded prerequisite for fused public Q8
  HC consumers.
- Goal: connect public HC expansion, fused public Q8 HC consumers, remaining graph
  compute, whole-archive retention policy, and production route-promotion
  work without claiming C CUDA removal before those gates pass.

################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba: Public Hyperconnection Expansion ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_hc_expand_tensor`,
  `ds4_gpu_hc_expand_split_tensor`, and
  `ds4_gpu_hc_expand_add_split_tensor` through one current-C-compatible
  embedded expansion kernel without claiming fused Q8 HC consumers or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbba/abi-hc-expand-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_hc_expand_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the three public HC expansion entry
    points and validates direct and split-strided source spans before launch.
  - `rust/ds4-cuda/src/abi_kernels.rs` promotes a stride-aware
    `abi_hc_expand_kernel` covering direct, split, and split-plus-add calls.
  - A C-linked B300 consumer proves direct and split output parity,
    split-plus-add behavior, aliased `block_add == block_out` behavior, and
    invalid-input rejection.
  - Local library tests pass with 127 tests; B300 release-feature tests pass
    with 134 tests; the static library exposes 39 Rust ABI symbols; 37
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 41 CUDA ABI comparators pass, and the unified parity report passes
    with 213 passed, 45 skipped, and 0 failed.
  - Pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S` without
    completed findings.
  - Fused public Q8 HC consumers, remaining graph compute, whole-archive
    retention policy, route promotion, C CUDA removal, and the generated
    embedded-object executable-stack warning remain open.

################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbb because the two
  public fused Q8 HC consumers share a bounded current-C kernel and fallback
  contract independently from internal pair consumers and route work.
- Goal: connect fused public Q8 HC consumers, remaining graph compute,
  whole-archive retention policy, and production route-promotion work
  without claiming C CUDA removal before those gates pass.

#################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba: Public Fused Q8 Hyperconnection Consumers ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_matmul_q8_0_hc_expand_tensor` and
  `ds4_gpu_shared_down_hc_expand_q8_0_tensor` through their fused Q8 HC
  expansion path and existing disabled-fusion fallback without claiming
  internal pair consumers or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbba/abi-fused-q8-hc-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_fused_q8_hc_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports both public fused consumers, reuses
    retained activation quantization, and delegates the disabled-fusion
    selection through the already-owned Q8 matmul and HC expansion ABI.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_matmul_q8_0_hc_expand_preq_warp8_kernel`, retaining DP4A/scalar
    selection and explicit add semantics when output aliases routed input.
  - A C-linked B300 consumer proves DP4A and scalar fused results, both
    disabled-fusion fallback routes, shared-down add including aliased add,
    and invalid-shape rejection.
  - Local library tests pass with 128 tests; B300 release-feature tests pass
    with 135 tests; the static library exposes 41 Rust ABI symbols; all 38
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 42 CUDA ABI comparators pass, and the unified parity report passes
    with 214 passed, 45 skipped, and 0 failed.
  - The pre-implementation non-interactive Claude review attempt returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`; final pass-end Claude review returned
    `NO BLOCKERS`.
  - The internal Q8 pair consumer, remaining graph compute, whole-archive
    retention policy, route promotion, C CUDA removal, and the generated
    embedded-object executable-stack warning remain open.

#################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbb because the
  public shared gate/up Q8 SwiGLU consumer has a bounded paired-kernel and
  disabled-pair fallback contract independently from later graph work.
- Goal: connect internal pair and remaining graph compute,
  whole-archive retention policy, and production route-promotion work
  without claiming C CUDA removal before those gates pass.

##################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba: Public Shared Gate Up SwiGLU Q8 ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_shared_gate_up_swiglu_q8_0_tensor` through its
  private paired-Q8 acceleration and public disabled-pair fallback without
  exporting `ds4_gpu_matmul_q8_0_pair_tensor` or claiming route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbba/abi-shared-gate-up-swiglu-q8-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_shared_gate_up_swiglu_q8_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the public shared gate/up consumer,
    reuses retained activation quantization, and delegates
    `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR` through the already-owned public
    Q8 matmul and SwiGLU exports.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_matmul_q8_0_pair_preq_warp8_kernel` as a private implementation
    kernel with retained DP4A/scalar dot selection.
  - A C-linked B300 consumer proves paired DP4A and scalar output,
    disabled-pair fallback, clamped SwiGLU output, and invalid-range
    rejection.
  - Local library tests pass with 129 tests; B300 release-feature tests pass
    with 136 tests; the static library exposes 42 Rust ABI symbols; all 39
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 43 CUDA ABI comparators pass, and the unified parity report passes
    with 215 passed, 45 skipped, and 0 failed.
  - The pre-implementation non-interactive Claude review attempt returned
    `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`; final pass-end Claude review returned
    `NO BLOCKERS`.
  - Remaining graph compute, whole-archive retention policy, route
    promotion, C CUDA removal, and the generated embedded-object
    executable-stack warning remain open.

##################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbb because direct and
  split-stride hyperconnection weighted-sum reductions form a bounded public
  ABI leaf independently from Sinkhorn and later fused reductions.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

###################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba: Public Hyperconnection Weighted Sum ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_hc_weighted_sum_tensor` and
  `ds4_gpu_hc_weighted_sum_split_tensor` through one stride-aware embedded
  reduction kernel without claiming Sinkhorn, fused reductions, or route
  ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbba/abi-hc-weighted-sum-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_hc_weighted_sum_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the direct and split public wrappers,
    retains current-C token derivation for valid output tensors, and rejects
    residual or split inputs shorter than their accessed token span.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_hc_weighted_sum_kernel` with `u64` flattened indexing and direct or
    split-prefix stride selection.
  - A C-linked B300 consumer proves direct weighted output, split-stride
    output with noisy unused split entries, and short-input and zero-shape
    rejection.
  - Local library tests pass with 130 tests; B300 release-feature tests pass
    with 137 tests; the static library exposes 44 Rust ABI symbols; all 40
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 44 CUDA ABI comparators pass, and the unified parity report passes
    with 216 passed, 45 skipped, and 0 failed.
  - The pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  - Sinkhorn, fused split reductions, remaining graph compute, whole-archive
    retention policy, route promotion, C CUDA removal, and the generated
    embedded-object executable-stack warning remain open.

###################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbb because public
  split-Sinkhorn generation is a bounded model-backed ABI leaf independently
  from fused split-weighted reductions.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

####################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba: Public Hyperconnection Split Sinkhorn ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_hc_split_sinkhorn_tensor` through its four-branch
  mixer transform and model-backed scale/base inputs without claiming fused
  reductions or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbba/abi-hc-split-sinkhorn-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_hc_split_sinkhorn_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the split-Sinkhorn wrapper and resolves
    its three scale floats and 24 base floats through the established cached
    model-range path while retaining current-C full-row flooring.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_hc_split_sinkhorn_kernel` and private `abi_hc4_split_one` math using
    libdevice sigmoid and Sinkhorn iterations.
  - A C-linked B300 consumer proves two-row Sinkhorn output, shorter-output
    row flooring, alternate model parameter ranges, and invalid-input
    rejection.
  - Local library tests pass with 131 tests; B300 release-feature tests pass
    with 138 tests; the static library exposes 45 Rust ABI symbols; all 41
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 45 CUDA ABI comparators pass, and the unified parity report passes
    with 217 passed, 45 skipped, and 0 failed.
  - The pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  - Fused split-weighted reductions, remaining graph compute, whole-archive
    retention policy, route promotion, C CUDA removal, and the generated
    embedded-object executable-stack warning remain open.

####################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbb because the
  public fused split-weighted reduction is a bounded ABI leaf independently
  from normalized reduction and output-head HC weights.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

######################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba: Public Hyperconnection Split Weighted Sum ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_hc_split_weighted_sum_tensor` through one
  synchronized embedded split-and-reduction kernel without claiming the
  normalized fused consumer, output-head HC weights, or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbba/abi-hc-split-weighted-sum-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_hc_split_weighted_sum_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the fused wrapper, resolves its three
    scale floats and 24 base floats through the established cached
    model-range path, and rejects incomplete output rows or short input
    spans.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_hc_split_weighted_sum_fused_kernel`, reusing private
    `abi_hc4_split_one` math and a block synchronization boundary before
    reducing residual hyperconnections.
  - A C-linked B300 consumer proves fused output, emitted split values,
    alternate model parameter ranges, output-defined row count, and
    invalid-input rejection.
  - Local library tests pass with 132 tests; B300 release-feature tests pass
    with 139 tests; the static library exposes 46 Rust ABI symbols; all 42
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 46 CUDA ABI comparators pass, and the unified parity report passes
    with 218 passed, 45 skipped, and 0 failed.
  - The pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  - Normalized fused reduction, output HC weights, remaining graph compute,
    whole-archive retention policy, route promotion, C CUDA removal, and the
    generated embedded-object executable-stack warning remain open.

######################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbb because the
  normalized fused HC consumer and its fallback policy form a bounded public
  ABI leaf independently from output-head HC weights.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

######################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba: Public Hyperconnection Split Weighted Sum Norm ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_hc_split_weighted_sum_norm_tensor` through its
  one-row fused kernel and current-C disabled-or-multi-row fallback without
  claiming output-head HC weights or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbba/abi-hc-split-weighted-sum-norm-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_hc_split_weighted_sum_norm_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the normalized wrapper, retaining
    current-C environment and row-count dispatch while resolving all fused
    parameter spans through the established cached model-range path.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_hc_split_weighted_sum_norm_fused_kernel`, reusing private split math
    and performing its weighted RMS reduction behind thread-block barriers.
  - A C-linked B300 consumer proves the one-row fused result, the disabled
    and multi-row fallback behavior, alternate model parameter ranges, and
    invalid-input rejection.
  - Local library tests pass with 133 tests; B300 release-feature tests pass
    with 140 tests; the static library exposes 47 Rust ABI symbols; all 43
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 47 CUDA ABI comparators pass, and the unified parity report passes
    with 219 passed, 45 skipped, and 0 failed.
  - The pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  - Output HC weights, remaining graph compute, whole-archive retention
    policy, route promotion, C CUDA removal, and the generated embedded-object
    executable-stack warning remain open.

######################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbb because public
  output-head hyperconnection weight generation is a bounded ABI leaf
  independently from remaining graph compute and route promotion.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

########################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba: Public Output Hyperconnection Weights ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_output_hc_weights_tensor` through its public
  sigmoid-plus-eps output-weight kernel without claiming remaining graph
  compute or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbba/abi-output-hc-weights-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_output_hc_weights_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports the public output-weight wrapper,
    deriving complete `n_hc` rows and resolving scale/base parameter spans
    through the established cached model-range path.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds
    `abi_output_hc_weights_kernel`, computing current-C
    `sigmoid(pre * scale + base) + eps` output weights.
  - A C-linked B300 consumer proves multi-token and single-token output,
    alternate model parameter ranges, and invalid-input rejection.
  - Local library tests pass with 134 tests; B300 release-feature tests pass
    with 141 tests; the static library exposes 48 Rust ABI symbols; all 44
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 48 CUDA ABI comparators pass, and the unified parity report passes
    with 220 passed, 45 skipped, and 0 failed.
  - The pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  - Rust rejects output token counts above `u32::MAX` instead of retaining
    current-C launch-argument narrowing for oversized tensors.
  - Remaining graph compute, whole-archive retention policy, route promotion,
    C CUDA removal, and the generated embedded-object executable-stack warning
    remain open.

########################################## M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active; split into M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba
  and M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbb because the
  public single-token and batched embedding wrappers are a bounded ABI leaf
  independently from remaining graph compute and route promotion.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

########################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba: Public Embedding Hyperconnection ABI

- Status: done.
- Goal: Rust-own `ds4_gpu_embed_token_hc_tensor` and
  `ds4_gpu_embed_tokens_hc_tensor` through their public FP16 embedding
  kernels without claiming remaining graph compute or route ownership.
- Fixture:
  `ds4-parity/baselines/backend/m14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbba/abi-embedding-smoke.json`.
- Comparator:
  `ds4-parity/check_cuda_abi_embedding_smoke.py --negative-test`.
- Evidence:
  - `rust/ds4-cuda/src/abi.rs` exports both embedding wrappers, validates the
    accessed tensor spans, and resolves FP16 embedding rows through the
    established cached model-range path.
  - `rust/ds4-cuda/src/abi_kernels.rs` embeds single-token replication and
    batched token kernels; the batched path retains current-C invalid-ID
    fallback to embedding row zero.
  - A C-linked B300 consumer proves single-token replication, batched invalid
    token fallback, alternate model ranges, and invalid-input rejection.
  - Local library tests pass with 135 tests; B300 release-feature tests pass
    with 142 tests; the static library exposes 50 Rust ABI symbols; all 45
    preceding linked ABI consumers pass against the rebuilt archive with the
    known generated embedded-object executable-stack warning.
  - All 49 CUDA ABI comparators pass, and the unified parity report passes
    with 221 passed, 45 skipped, and 0 failed.
  - Rust rejects out-of-vocabulary single tokens, short single outputs, and
    zero-dimensional or overflowing launches rather than retaining
    current-C unchecked or undefined behavior.
  - The pre-implementation and final pass-end non-interactive Claude review
    attempts each returned `CLAUDE_REVIEW_TIMEOUT_AFTER_60S`.
  - Remaining graph compute, whole-archive retention policy, route promotion,
    C CUDA removal, and the generated embedded-object executable-stack warning
    remain open.

########################################### M14.6b2b2b2b2b2b2b2b2b2b2b2b2b2bbbbbbbbbbbbbbbbbbbbbbbb: Remaining Graph Compute And Route Promotion Policy

- Status: active.
- Goal: connect remaining graph compute, whole-archive retention policy, and
  production route-promotion work without claiming C CUDA removal before
  those gates pass.

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
