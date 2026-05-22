# M0.2 Build Baseline Manifest

## Capture Scope

- Work item: M0.2 Baseline Build Command Capture
- Source oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Capture commit: `004ade62556bcfa7c8950a624619f8492230102b`
- Drift policy: no source behavior changes are introduced by this capture.
- Manifest rule: rerun the listed command in the listed environment and compare
  exit status plus declared artifact/log behavior.

## Local Machine

- Cwd: `/Users/dzhanguzin/dev/personal/ds4`
- Metadata log: `logs/m0.2-local-machine.txt`
- Machine: Apple M4 Pro, macOS Darwin 25.4.0.
- Default shell architecture: `x86_64`.
- Intended local build architecture: `arm64` via `arch -arm64`.
- Default compiler target: `x86_64-apple-darwin25.4.0`.
- Arm64 compiler target: `arm64-apple-darwin25.4.0`.
- Model availability: `ds4flash.gguf` is absent locally.

## Local Command Entries

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| `make clean` | default shell | 0 | `logs/m0.2-make-clean.log` | Clean target removes ignored build products. |
| `make` | default shell | 2 | `logs/m0.2-make.log` | Recorded local Rosetta failure: Apple clang rejects `-mcpu=native` for the default `x86_64-apple-darwin25.4.0` target. |
| `arch -arm64 make clean` | local arm64 | 0 | `logs/m0.2-arm64-make-clean.log` | Clean target succeeds before the local Metal build. |
| `arch -arm64 make` | local arm64 | 0 | `logs/m0.2-arm64-make.log` | Builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent` for the local Metal backend. |
| `arch -arm64 make test` | local arm64, no model | 2 | `logs/m0.2-arm64-make-test.log` | Test binary builds, then the default all-test run stops at missing `ds4flash.gguf`. |
| `arch -arm64 ./ds4_test --server` | local arm64, no model | 0 | `logs/m0.2-arm64-ds4-test-server.log` | Server parser/rendering/cache unit tests pass without a model. |
| `arch -arm64 ./ds4_test --metal-kernels` | local arm64 Metal | 0 | `logs/m0.2-arm64-ds4-test-metal-kernels.log` | Isolated Metal kernel numeric check passes on Apple M4 Pro. |
| `arch -arm64 make clean` | local arm64 | 0 | `logs/m0.2-arm64-make-clean-before-cpu.log` | Clean target succeeds before the CPU build. |
| `arch -arm64 make cpu` | local arm64 | 0 | `logs/m0.2-arm64-make-cpu.log` | Builds CPU-only `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`. |
| `file ds4 ds4-server ds4-bench ds4-eval ds4-agent` | after local CPU build | 0 | `logs/m0.2-arm64-cpu-artifacts.log` | CPU build artifacts are arm64 Mach-O executables. |
| `arch -arm64 make cuda-regression` | local macOS | 0 | `logs/m0.2-arm64-make-cuda-regression.log` | Darwin target records that CUDA regression requires a CUDA build. |

## B300 CUDA Command Entries

- Kubeconfig workflow: temporary per-session copy at
  `/tmp/ds4-hou2-prod1.kubeconfig`; host kubectl context left unchanged.
- Context: `hou2-prod1`
- Namespace: `default`
- Pod: `ds4-rust-port-b300`
- Node: `c1v17-b300n1-nic1`
- GPU: NVIDIA B300 SXM6 AC, UUID `GPU-81f6bd2a-3404-6445-1788-365264243aab`
- Pod environment log: `logs/m0.2-b300-env.log`
- Pod creation logs: `logs/m0.2-b300-pod-apply.log`,
  `logs/m0.2-b300-pod-wait.log`
- Source copy log: `logs/m0.2-b300-source-copy.log`

| Command | Environment | Exit | Log | Acceptance |
| --- | --- | ---: | --- | --- |
| `kubectl apply` for `ds4-rust-port-b300` | `hou2-prod1/default` | 0 | `logs/m0.2-b300-pod-apply.log` | Reusable B300 pod created for this port. |
| `kubectl wait pod/ds4-rust-port-b300 --for=condition=Ready --timeout=10m` | `hou2-prod1/default` | 0 | `logs/m0.2-b300-pod-wait.log` | Pod reached Ready on `c1v17-b300n1-nic1`. |
| `git archive HEAD \| kubectl exec ... tar -xf - -C /workspace/ds4` | local to B300 pod | 0 | `logs/m0.2-b300-source-copy.log` | Capture commit source copied without local uncommitted artifacts. |
| `make cuda-generic` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.2-b300-make-cuda-generic.log` | Builds CUDA `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`. |
| `file ds4 ds4-server ds4-bench ds4-eval ds4-agent` | after B300 CUDA build | 0 | `logs/m0.2-b300-cuda-artifacts.log` | CUDA build artifacts are x86-64 Linux ELF executables. |
| `make cuda-regression` | B300 pod, `/workspace/ds4` | 0 | `logs/m0.2-b300-make-cuda-regression.log` | CUDA backend initializes on NVIDIA B300 SXM6 AC (`sm_103`) and the long-context CUDA regression passes. |
| `make test` | B300 pod, no model | 2 | `logs/m0.2-b300-make-test.log` | Test binary builds, then the default all-test run stops at missing `ds4flash.gguf`. |
| `./ds4_test --server` | B300 pod, no model | 0 | `logs/m0.2-b300-ds4-test-server.log` | Server parser/rendering/cache unit tests pass. |
| `./ds4_test --metal-kernels` | B300 pod CUDA backend | 0 | `logs/m0.2-b300-ds4-test-metal-kernels.log` | Backend tensor kernel check passes through CUDA despite the legacy flag name; stdout/stderr ordering may differ under `kubectl exec`. |

## Blocked Or Deferred Model-Backed Entries

These are not M0.2 failures; they require the model fixture captured in later
Milestone 0 items.

- `./ds4_test --logprob-vectors`
- `./ds4_test --long-context`
- `./ds4_test --tool-call-quality`
- `ds4-bench` short and long prompt CSV captures
- Server request traces that load the model

The exact blocker in both local and B300 environments is absence of
`ds4flash.gguf`. M0.3 must either provide `DS4_TEST_MODEL` or record the model
download/path/hash before claiming model-backed parity coverage.
