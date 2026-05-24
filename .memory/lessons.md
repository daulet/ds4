# DS4 Rust Port Lessons

Record only non-obvious findings discovered through trial and error that are not
available directly from the repo.

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
