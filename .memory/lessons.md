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
