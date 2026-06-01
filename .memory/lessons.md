# DS4 Rust Port Lessons

Record only non-obvious findings discovered through trial and error that are not
available directly from the repo.

## 2026-06-01: Explicit Global Loads Need Phase-Scoped Retention

- Symptom: forcing typed `ld.global` source reads in cached gate/up and down
  improved gate/up but made down materially slower in the same B300 profile.
- Root cause: fixing the address-space form is not a generally beneficial
  transformation across these two different hot kernels; the down instruction
  schedule or pressure response outweighs its nominal load-form improvement.
- Permanent rule: introduce CUDA address-space intrinsics only on a
  phase-attributed path with adjacent measurement. Retain the gate/up `u16` and
  `u32` loads here, and reject the matching down edit until down has its own
  measured repair.

## 2026-05-30: Staticlib Embedded Kernels Need Final-Link Retention And Package Module Names

- Symptom: library-owned `add_kernel` and `repeat_hc_kernel` compiled into
  `libds4_cuda.a`, but a downstream Rust binary exposed no embedded modules;
  a C-linked archive consumer then found module `ds4-cuda` while lookup of
  `ds4_cuda` returned `ModuleNotFound`; once loaded, the full feature test
  matrix rejected duplicate entry-point symbols shared with the older
  executable-local smoke.
- Root cause: cuda-oxide emits the embedded PTX bundle as a no-symbol archive
  object named for the Cargo package, so dependency linking does not propagate
  that artifact and conventional static-library extraction does not retain it.
- Permanent rule: load embedded library modules by Cargo package name, prefix
  reusable ABI kernel entry points distinctly from executable smokes, and
  explicitly retain the archive's embedded object at the final link boundary
  until production integration supplies a more selective retaining symbol.
  Also track the generated object's missing `.note.GNU-stack` warning before
  production route promotion.

## 2026-05-30: Embedded Kernel Proofs Need Final-Binary Ownership And Portable PTX Targets

- Symptom: an M14.1b4 `#[cuda_module]` placed in `ds4-cuda` library code
  compiled but the smoke executable reported `ModuleNotFound`; after moving
  the module into the binary, B300 loading failed with CUDA error 218.
- Root cause: cuda-oxide embeds non-generic PTX in the crate being linked as
  the final executable for this path, while `cargo oxide run` also overrode
  a basic backend-selected target with local B300 `sm_103`; `/usr/bin/llc`
  emitted `.version 6.0 / .target sm_103`, which CUDA 13.2 rejects.
- Permanent rule: put executable-owned embedded kernel modules in the final
  binary until cross-crate artifact retention is explicitly implemented, and
  let basic kernels use the backend-selected portable target on `sm_80`-or-
  newer GPUs. cuda-oxide revision
  `981e3244a107d84d807cfb087793269c477cc764` enforces the target rule.

## 2026-05-25: Stage-Progression Checkers Need Future Closure Markers

- Symptom: after M13.5 moved `.memory/status.md` to the post-M13 decision, the
  unified report failed only in older static-wiring checks that accepted
  `Active item: M13` but not the closure marker.
- Root cause: earlier milestone checkers encoded the next active milestone as a
  bounded string list, so later roadmap closure states can make already-passed
  historical artifacts look stale.
- Permanent rule: milestone static-wiring checks should accept durable closure
  markers for later roadmap phases when the artifact itself remains unchanged.
  Do not weaken artifact or comparator assertions; only broaden status
  progression strings that prove the board advanced.

## 2026-05-24: Field-Adding Comparator Mutations Need Missing-Path Failures

- Symptom: the first M10.7d3b `compare_graph_restore_next_token.py
  --negative-test` crashed with `KeyError: 'frontier_projection'` before the
  B300 summary had been refreshed with the new field.
- Root cause: the negative-test mutator assumed every mutation path already
  existed in the candidate summary, but field-adding milestones intentionally
  validate against a stale committed fixture until the live capture refreshes
  it.
- Permanent rule: comparator mutation helpers should convert missing mutation
  paths into ordinary negative-test failures instead of uncaught exceptions.
  That keeps stale-fixture failures readable while still requiring the new
  field to exist after the fixture is refreshed.

## 2026-05-24: B300 Runtime Replays Need Startup Slack After Source Refresh

- Symptom: the first M0.5 runtime replay refresh failed with HTTP code `000`
  because `curl` ran before `ds4-server-runtime-rs` was listening.
- Root cause: after rebuilding the Rust runtime on the B300 pod, CUDA model
  cache startup took about 13 seconds before the server bound the port.
- Permanent rule: B300 runtime replay commands should wait at least 20 seconds,
  or use an explicit readiness probe, before sending replay requests after a
  fresh build or source sync.

## 2026-05-24: M0.5 Seed Cold Stores Do Not Suppress Continued Frontier

- Symptom: the live B300 ledger showed `suppress_continued_store` failed for
  the 550-token M0.5 seed miss, and the frontier advanced only when the cold
  write called `note_store`.
- Root cause: `--kv-cache-continued-interval-tokens 0` still aligns continued
  store targets to the configured boundary, so 550 prompt tokens is a valid
  cold-store length but not a continued-store boundary.
- Permanent rule: M0.5 seed-miss replay oracles should expect cold-store
  success plus `note_store` frontier growth, not pre-cold continued-frontier
  suppression.

## 2026-05-24: B300 Restore Payload Bodies Are Capture Metadata

- Symptom: rerunning `ds4-restore-dump` on B300, including a rerun from the M7.8
  capture source commit, produced different disk payload and memory snapshot
  SHA256 values while `check_restore_dump.py` still passed C source-vs-restored
  selected-token/top-logprob self-checks.
- Root cause: the raw restore bodies include bytes that are not stable enough to
  be treated as a cross-capture oracle, even when the restore behavior and DSV4
  layout contract are valid.
- Permanent rule: raw restore-body SHA256/FNV values should be recorded as
  per-capture evidence. Parity gates should keep byte counts, DSV4 headers,
  section layout, count tables, Rust reader acceptance, and behavior comparators
  exact instead of failing solely on raw body SHA drift.

## 2026-05-24: Comparator Scripts Need A Live Entrypoint Check

- Symptom: `compare_decode_directional_steering.py --negative-test` initially
  exited 0 without printing any checks or failures.
- Root cause: the script defined `main()` but did not call it under
  `if __name__ == "__main__"`, so Python syntax checks and shell exit status
  alone did not prove that the comparator ran.
- Permanent rule: new parity comparators must be run at least once in the
  static/no-args mode and once with `--negative-test`, and the validation log
  should record the printed check or mutation count, not just exit code.

## 2026-05-22: Local Apple Silicon Builds Need Explicit arm64 In This Shell

- Symptom: plain `make` failed before compiling DS4 because Apple clang rejected
  `-mcpu=native` for target `x86_64-apple-darwin25.4.0`.
- Root cause: the Codex shell reports `uname -m` as `x86_64` on an Apple M4 Pro,
  while `arch -arm64 cc --version` selects the intended
  `arm64-apple-darwin25.4.0` compiler target.
- Source evidence: `Makefile` sets `NATIVE_CPU_FLAG ?= -mcpu=native` on Darwin;
  `ds4-parity/baselines/logs/m0.2-make.log` captures the failure and
  `ds4-parity/baselines/logs/m0.2-arm64-make.log` captures the successful arm64
  build.
- Permanent rule: local macOS baseline captures from this environment should run
  build/test commands through `arch -arm64 ...`, and should use `make clean`
  between Metal and CPU builds because both targets write the same binary names.

## 2026-05-22: B300 Kubectl Commands Need Explicit hou2-prod1 Context

- Symptom: a pod lookup without `--context hou2-prod1` reported NotFound even
  though the reusable B300 pod was still running.
- Root cause: `/tmp/ds4-hou2-prod1.kubeconfig` contains multiple contexts and
  its current context can be `mc1-lab1`; using only `--kubeconfig` is not enough
  to select the intended cluster.
- Permanent rule: every B300 command for this port should pass both
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig` and `--context hou2-prod1`.

## 2026-05-22: Avoid Tiny Disk-KV Prefixes In Server Trace Captures

- Symptom: M0.4 server replay returned HTTP 500 with `cuda prefill failed` for
  an 11-token non-thinking chat prompt when the server ran with
  `--kv-cache-min-tokens 1`.
- Root cause: that setting forced the disk-KV cold-prefix path on a tiny prompt;
  the same request passed with no disk KV and with disk KV using the default
  512-token minimum.
- Permanent rule: M0.4 server trace captures should use the server's normal
  context and default-scale disk-KV thresholds. M0.5 should test disk-KV files
  with prompt sizes that naturally satisfy the production thresholds rather than
  forcing `--kv-cache-min-tokens 1`.

## 2026-05-22: M0.5 KV Fixtures Should Commit Metadata, Not Raw KV Files

- Symptom: a 160-word synthetic cache prompt tokenized below the 512-token
  disk-KV threshold and produced no cache files, while a 900-word prompt
  produced 52 to 74 MiB cache files.
- Root cause: the useful M0.5 prompt needs to sit just above the production
  disk-KV threshold; 270 `cacheNNN` words tokenized to 550 prompt tokens and
  generated reproducible cache behavior with roughly 30 MiB `.kv` files.
- Permanent rule: M0.5 commits fixture JSON, rendered cached text, parsed KVC
  headers, full hashes, and timestamp-normalized hashes, but leaves raw `.kv`
  binaries out of git unless a later milestone explicitly needs binary fixtures
  checked in.

## 2026-05-22: Avoid `status` As A zsh Wrapper Variable

- Symptom: an M0.6 source-refresh wrapper exited before copying source and
  logged `zsh: read-only variable: status`.
- Root cause: zsh reserves `status`; assigning to it fails even though the same
  pattern would work with an ordinary shell variable name.
- Permanent rule: local capture wrappers should use `rc` or another neutral
  variable for command exit codes.

## 2026-05-22: Verifier Negative Tests Should Fail As Reports, Not Crashes

- Symptom: the first M1.2 negative test corrupted a benchmark CSV and the
  verifier raised a Python `ValueError` while parsing the bad row.
- Root cause: the normal path assumed typed CSV fields after header validation;
  corrupt candidate artifacts need to be reported as comparison failures, not
  uncaught parser exceptions.
- Permanent rule: parity verifier parsers should convert malformed structured
  artifacts into section errors so negative tests prove failure reporting as
  well as drift detection.

## 2026-05-23: Pair Exact Tensor Digests With Tolerant JSON Float Samples

- Symptom: the first B300 Rust-vs-current-C first-kernel comparator failed even
  though the full `cur_hc` FNV digest matched exactly.
- Root cause: Rust and C emitted equivalent sample floats with different JSON
  decimal spelling.
- Permanent rule: paired tensor comparators should keep full-buffer digests
  exact, but compare selected JSON float samples numerically within the
  milestone tolerance.

## 2026-05-23: Execution-Behavior Oracles Need The Same GPU Math Path

- Symptom: the first layer-0 HC-pre B300 comparator matched `cur_hc`, but exact
  FNV digests diverged for `flat_hc`, `hc_mix`, `hc_split`, `attn_cur`, and
  `attn_norm`.
- Root cause: the initial C oracle used CPU helpers for RMS norm, F16 matvec,
  HC split, weighted sum, and attention norm, while the Rust candidate used the
  current C CUDA tensor ABI.
- Permanent rule: when acceptance asks for exact execution-behavior digests,
  the current-C oracle must drive the same GPU tensor functions as the Rust
  facade. CPU helpers remain useful semantic references, but not exact digest
  oracles for CUDA math.

## 2026-05-22: M0.3 Runtime Log Is Pass/Fail Evidence, Not A Numeric Dump

- Symptom: M1.4 needed a numeric comparator, but
  `m0.3-b300-logprob-vectors.log` only recorded per-case pass markers and
  `logprob-vectors: OK`.
- Root cause: `./ds4_test --logprob-vectors` enforces exact selected-token
  matching and a hardcoded 4.0 logprob tolerance internally, but it does not
  print the runtime top-logprob values unless a failure occurs.
- Permanent rule: compare logprob numeric fixtures through
  `tests/test-vectors/official.vec` and the raw official JSON, and treat the
  M0.3 B300 log as evidence that the C runtime enforced that fixture on the
  recorded model/backend.

## 2026-05-22: Fast-Math Can Break Non-Finite Oracle JSON Helpers

- Symptom: the first M6.2 sampling oracle dump emitted bare `inf`/`-inf`,
  which made the JSON invalid even though the helper intended to quote
  non-finite float values.
- Root cause: local C builds use `-ffast-math`; under those flags, ordinary
  non-finite checks around `float` parameters can be optimized under finite
  assumptions. The sampler's `isfinite` behavior is still the current C oracle,
  but the JSON serialization helper needed to classify raw float bits without
  those optimizations.
- Permanent rule: no-model oracle dumpers that need to serialize non-finite
  float fixtures should use raw bit fixtures plus an unoptimized bit-classifier
  helper for JSON output, then run `python3 -m json.tool` or an equivalent
  parser before committing the baseline.

## 2026-05-22: B300 Cannot Capture Current-C Imatrix Output Today

- Symptom: the M8.10 current-C imatrix capture attempt on the B300 pod exited
  before writing `/tmp/m8.10-imatrix.dat`.
- Root cause: `--imatrix-out` forces the CLI backend to Metal, while the B300
  build is CUDA-linked. The observed stderr was `backend=metal` followed by
  `ds4: Metal backend requested but this build is linked with CUDA, not Metal`.
- Permanent rule: do not try to refresh imatrix `.dat` output baselines on the
  B300 CUDA pod until current C supports CUDA imatrix collection. Use a
  Metal-capable host with the recorded model, or keep M8.10b/M8.11 blocked.

## 2026-05-22: B300 Has Steering Fixture But No MTP GGUF

- Symptom: M8.12b could execute directional steering but had to record MTP as a
  missing-support-artifact blocker.
- Root cause: the B300 workspace contains the tracked
  `dir-steering/out/verbosity.f32` file, but no MTP GGUF under `/workspace/ds4`.
- Permanent rule: runtime-control refreshes can use the committed verbosity
  steering vector on B300, but MTP transcript coverage needs a support-model
  provisioning step before it can become an executed oracle case.

## 2026-05-24: Split MTP Work By Current-C Execution Boundary

- Symptom: the first M10.8 split isolated exact N=2 verification and frontier
  mutation, but left MTP draft orchestration and suffix/microbatch verifier
  orchestration to the final end-to-end item.
- Root cause: the split followed broad behavior categories instead of every
  current-C execution boundary called by `ds4_session_decode_speculative`.
- Permanent rule: speculative decode port stages must give MTP draft,
  exact-N=2 verifier, suffix verifier, frontier mutation, and end-to-end
  stream parity their own validation surfaces before integration.

## 2026-05-30: Embedded Libdevice Kernels Need A Staticlib Load Boundary

- Symptom: the reusable Rust ABI module could load add and steering PTX
  directly, but adding SwiGLU requires resolving `__nv_expf` before a C
  consumer can execute the static library.
- Root cause: `cuda-core` embeds portable PTX in the final executable, while
  `cuda-host` links libdevice only from a filesystem PTX path; ordinary
  embedded loading cannot resolve libdevice references.
- Permanent rule: for reusable nonlinear ABI kernels, extract the embedded
  PTX and pass it through the cuda-host libdevice cubin builder before module
  load, then remove successful process-local link artifacts. Record
  whole-archive retention and non-release codegen behavior as production
  integration requirements.

## 2026-05-31: Do Not Port CUDA `__vsub4` As A Three-Operand PTX Video Op

- Symptom: a cuda-oxide packed byte-subtraction probe built Rust PTX with
  `vsub4.u32.u32.u32`, but the B300 assembler rejected all `16` hot-kernel
  sites with argument mismatch errors.
- Root cause: the legacy PTX `vsub4` video instruction requires its selector
  operand, while CUDA 13.2 lowers `__vsub4` for `compute_80` to scalar
  emulation rather than emitting that video instruction.
- Permanent rule: keep the retained packed sign transform for portable Rust
  gate/up code; do not pursue a `vsub4` intrinsic unless a valid, measured
  target-specific lowering is first proven by assembly and end-to-end tests.

## 2026-05-31: Check Emitted Calls Even For `#[inline(always)]` CUDA Helpers

- Symptom: changing the retained inlined quarter-warp reduction loop into
  three explicit shuffles caused cuda-oxide to emit out-of-line caller sites
  again despite retaining `#[inline(always)]`.
- Root cause: the CUDA code-generation decision changes with the expanded
  helper body; source-level inlining annotations alone do not prove hot-kernel
  PTX shape.
- Permanent rule: for Rust CUDA helper scheduling repairs, record both the
  helper PTX and each hot caller's call/shuffle/spill counts before attributing
  a performance change to inlining.

## 2026-06-01: Match Typed Q8 Staging Width Before Tuning Consumers Further

- Symptom: after aligned Q8 shared loads and reduction repairs, cached Rust
  gate/up and down still trailed current C while staging the same Q8_K blocks
  through byte copies.
- Root cause: current C copies typed `cuda_block_q8_K` elements into shared
  memory, while Rust retained byte-granularity staging even after its hot
  consumers used aligned word loads.
- Permanent rule: when a CUDA oracle stages naturally aligned block structs,
  match both the shared consumption width and the staging width before
  pursuing arithmetic approximations; retain based on repeated phase totals
  when one-shot end-to-end throughput is noisy.

## 2026-06-01: Expand Fixed Reductions At The Hot Site When Helper Inlining Drifts

- Symptom: the branch-free fixed-order quarter-warp helper improved Rust CUDA
  phases but cuda-oxide emitted hot gate/up and down reduction calls while
  current C kept the shuffle sequence in-kernel.
- Root cause: a fixed-order device helper did not retain caller inlining even
  with `#[inline(always)]`; macro expansion controls the hot-site PTX shape.
- Permanent rule: when helper-level fixed ordering is useful but inlining
  regresses, test a narrowly scoped macro-expanded hot site and verify call,
  shuffle, and spill counts before retaining it.

## 2026-06-01: Align Cached Q8 Values Rather Than External Q8 Blocks

- Symptom: after aligned staging and in-kernel reduction repair, cached Rust
  gate/up and down still emitted `128` and `256` shared `u32` Q8 input loads.
- Root cause: Q8 values start four bytes into an external `292`-byte block, so
  adjacent value words are not naturally eight-byte aligned in a packed cache.
- Permanent rule: keep the external Q8 ABI unchanged, but when a private CUDA
  scratch representation needs paired loads, add explicit slot padding and
  validate PTX load width, spill counts, full vectors, and phase timing.

## 2026-06-01: Carry Private Q8 Alignment Through Staging

- Symptom: after aligning cached Q8 consumers, gate/up and down still trailed
  current C while writing every private cached block as `73` separate words.
- Root cause: the padded private slot aligns the `72`-word Q8 tail for paired
  writes as well as paired reads, but staging had retained word-width stores.
- Permanent rule: after introducing a private padded cache layout, measure
  aligned producer width as well as consumer width while keeping the external
  ABI and full-vector correctness fixed.

## 2026-06-01: Isolate Metadata Loads From Packed Payload Loads

- Symptom: cached down remained slower than current C after gate global-load
  repair, but converting packed Q2 data words to explicit global `u32` loads
  made down substantially slower despite a cleaner-looking PTX load shape.
- Root cause: address-space lowering benefits are not uniform within the Q2
  block; the two aligned halfword metadata reads improve modestly, while the
  repeated packed payload-word conversion expands into a losing schedule on
  B300.
- Permanent rule: isolate metadata, payload, and staging load changes as
  separate probes and retain only order-reversed phase wins with unchanged
  arithmetic and full-vector correctness.

## 2026-06-01: Pair Q2 Scale Bytes Without Globalizing Q2 Payloads

- Symptom: explicit halfword loads for the two Q2 metadata fields helped
  cached down modestly, but repeated low-nibble scale bytes still followed
  the scalar byte-load path.
- Root cause: scale bytes are naturally consumed in adjacent pairs and are
  aligned within every 84-byte Q2 block, so keeping them scalar leaves a
  measurable read/decoding overhead without requiring payload changes.
- Permanent rule: once an aligned narrow load is validated, apply it to
  natural metadata/control pairs separately from quantized payload words and
  confirm the result in both benchmark orders.
