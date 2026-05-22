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
