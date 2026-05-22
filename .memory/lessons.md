# DS4 Rust Port Lessons

Record only non-obvious findings discovered through trial and error that are not
available directly from the repo.

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
