# DS4 Rust Port Status

- Date: 2026-05-30 UTC
- Branch: `main`
- Starting oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Active item: M14.3b2 Head RMS Norm Rope Tail Kernel
- M14.1 cuda-oxide Substrate And Tensor Residency is split into M14.1a through
  M14.1c before implementation; M14.1b is further split into M14.1b1 through
  M14.1b4 because bounded residency handles, model-cache policy, allocation
  policy, and kernel command lifetime have distinct evidence boundaries.
  M14.1b2 is further split into M14.1b2a through M14.1b2c because device
  range-copy proof does not establish registered/HMM/direct-I/O strategy
  parity or model-cache closure. M14.1b2b is further split into M14.1b2b1
  through M14.1b2b3b2 because file-staged device copy, registered fallback,
  pageable HMM, direct-I/O read selection, and asynchronous staging policy
  have distinct API and live-evidence boundaries. M14.1b3 is split into
  M14.1b3a and M14.1b3b because managed-KV/memory-report policy only needs a
  memory-capacity query, while Q8 caches and quality mode need BLAS or kernel
  ownership.
- M14.2 Embedding Indexer And Elementwise Kernels is split into M14.2a
  through M14.2e because standalone elementwise, nonlinear/reduction,
  model-backed embedding, indexer/top-k, and closure work need separate live
  CUDA evidence boundaries. M14.2b is further split into M14.2b1 and
  M14.2b2 because B300 proved the directional projection PTX path while
  exposing a separate libdevice/NVVM executable blocker for SwiGLU. M14.2d
  is further split into M14.2d1 and M14.2d2 because scalar fallback
  selection and optimized dispatch have distinct ownership boundaries.
  M14.2d2 is further split into M14.2d2a through M14.2d2c because direct-one
  warp reduction, tensor-core score kernels, and specialized top-k kernels
  depend on distinct cuda-oxide primitives. M14.2d2b is further split into
  M14.2d2b1 and M14.2d2b2 because a base `16 x 16` score tile maps to two
  cuda-oxide MMA operations independently from widened multi-warp dispatch.
  M14.2d2b2 is further split into M14.2d2b2a through M14.2d2b2c because
  the 32, 64, and 128-component multi-warp kernels and final priority wiring
  need bounded live evidence. M14.2d2c is further split into M14.2d2c1
  through M14.2d2c5 because 1024 bitonic selection, larger power-of-two
  selection, CUB-or-equivalent selection, chunk/tree merge, and indexed
  ascending sort have separate CUDA launch and storage contracts.
- M14.3 Dense Projection Quantization And Norm Kernels is split beginning
  with M14.3a and M14.3b because standalone plain/weighted RMS normalization
  can be proved independently from fused QKV/head normalization, projection,
  and Q8 kernel families. M14.3b is further split into M14.3b1 and M14.3b2
  because basic fused QKV/head RMS normalization is independent from the
  combined head-normalization and YARN/RoPE tail path.
- Last validated source before the active item: M14.3b1 Fused QKV And Basic Head RMS Norm Kernels.
- Earlier M14.3a Plain And Weighted RMS Norm Kernels.
- Earlier M14.2e M14.2 Kernel Closure Gate.
- Earlier M14.2d2c5 Indexed Ascending Top-K Sort And Dispatch Policy.
- Earlier M14.2d2c4 Chunked And Tree-Merge Top-K Kernels.
- Earlier M14.2d2c3 CUB-Or-Equivalent Top-K Branch.
- Earlier M14.2d2c2 Power-Of-Two Top-K Kernels.
- Earlier M14.2d2c1 1024 Bitonic Top-K Kernel.
- Earlier M14.2d2b2c WMMA128 Tensor-Core Indexer Score Kernel And Dispatch Priority.
- Earlier M14.2d2b2b WMMA64 Tensor-Core Indexer Score Kernel.
- Earlier M14.2d2b2a WMMA32 Tensor-Core Indexer Score Kernel.
- Earlier M14.2d2b1 Base Tensor-Core Indexer Score Kernel.
- Earlier M14.2d2a Direct-One Indexer Score Kernel.
- Earlier M14.2d1 Scalar Indexer Selection Kernels.
- Earlier M14.2c Embedding Kernel Pair.
- Earlier M14.2b2 SwiGLU Libdevice Path.
- Earlier M14.2b1 Directional Steering Projection Kernel.
- Earlier M14.2a Add And Repeat Elementwise Kernels.
- Earlier M14.1c Substrate Route Closure Gate.
- Earlier M14.1b4 Fill Kernel And Command Lifetime.
- Earlier M14.1b3b Q8 Cache And Quality Policy.
- Earlier M14.1b3a Managed KV And Memory Report Policy.
- Earlier M14.1b2c Model Map Cache Closure.
- Earlier M14.1b2b3b2 Asynchronous Staging Ring And Budget Policy.
- Earlier M14.1b2b3b1 Direct-I/O Pinned Read Selection.
- Earlier M14.1b2b3a Pageable HMM Range Strategy.
- Earlier M14.1b2b2 Registered Range Strategy.
- Earlier M14.1b2b1 File-Staged Range Strategy.
- Earlier M14.1b2a Owned Mmap Device Range Copy.
- Earlier M14.1b1 Bounded Model Residency Handles.
- Earlier M14.1a Host Substrate Buffer Roundtrip.
- Earlier M14.0 CUDA Rust Ownership Inventory And Adoption Contract.
- Earlier post-M13 roadmap decision.
- Earlier M13.5 Embedding/Indexer Route Gate And Closure.
- Earlier M13.4 Batch Indexer Fixture Gap Closure.
- Earlier M13.3 Indexed Decode Selection Replacement Slice.
- Earlier M13.2 Batched Embedding Replacement Slice.
- Earlier M13.1 Embedding/Indexer Expansion Fixture Matrix.
- Earlier M13.0 Backend Expansion Decision.
- Earlier M12.6 Backend Replacement Closure And Removal Decision.
- Earlier M12.5 Runtime Backend Route Gate.
- Earlier M12.4 First Backend Replacement Slice.
- Earlier M12.3 Rust Backend Facade Parity Harness.
- Earlier M12.2 Operation Tensor Fixture Capture.
- Earlier M12.1 Backend Boundary Inventory And Claim Matrix.
- Earlier M12 Backend Replacement Parity split into M12.1 through M12.6 before
  implementation.
- Earlier M11 Agent Trace Replay split into M11.1 through M11.4 before
  implementation.
- Earlier M11.4 Rust Agent Loop And Manual Smoke.
- Earlier M11.3 Deterministic Tool Stub And Session Command Replay.
- Earlier M11.2 Rust Agent Rendered Context Replay.
- Earlier M11.1 Agent Trace Replay Oracle And Fixture Contract.
- Earlier M10.9f Benchmark Comparator And Milestone 10 Closure.
- Earlier M10.9e Tool-Call Quality And Server Replay Rust Runtime Gate.
- Earlier M10.9d B300 Long-Context Rust Runtime Gate.
- Earlier M10.9c B300 Official-Vector Rust Runtime Gate.
- Earlier M10.9b Rust Runtime Graph Route Switch And Preflight.
- Earlier M10.9a Runtime Graph Closure Matrix And Rerun Contract.
- Earlier M10.9 Runtime Graph End-To-End And Benchmark Closure split into
  M10.9a through M10.9f before implementation.
- Earlier M10.8g4b B300 End-To-End Blocker Or Support Comparator Closure.
- Earlier M10.8g4b B300 End-To-End Blocker Or Support Comparator Closure.
- Earlier M10.8g4a B300 Support-Artifact Branch Decision.
- Earlier M10.8g4 B300 Support-Model End-To-End Comparator.
- Earlier M10.8g Rust MTP End-To-End Stream Parity.
- Earlier M10.8g3c B300 Missing-Support Runtime Smoke.
- Earlier M10.8g3b Runtime Target-Stream No-Drift Comparator.
- Earlier M10.8g3a Rust Runtime MTP Guard Contract And Static Wiring.
- Earlier M10.8g3 Rust Runtime Guard And Target-Stream No-Drift Smoke split
  into M10.8g3a through M10.8g3c before implementation.
- Earlier M10.8g2 Rust MTP Stream Outcome Planner.
- Earlier M10.8g1 MTP Stream Parity
  Contract And Blocker.
- Earlier M10.8g Rust MTP End-To-End Stream Parity split into M10.8g1 through
  M10.8g4 before implementation.
- Earlier M10.8f Rust Spec Frontier Snapshot Restore And Prefix1 Commit.
- Earlier M10.8e Rust Suffix Verifier Orchestration Smoke.
- Earlier M10.8d Rust Exact N=2 Verifier Orchestration Smoke.
- M10.8 Rust MTP Draft And Verifier Orchestration is split into M10.8a through
  M10.8g before implementation. M10.8g is split into M10.8g1 through M10.8g4
  before runtime stream implementation.
- Earlier M10.8c Rust MTP Draft Kernel Orchestration Smoke.
- Earlier M10.8b Rust MTP Decision Planner.
- Earlier M10.8a MTP State Machine Contract And Availability Check.
- M10.7d3 Graph Restore Continued-Frontier B300 Smoke is split into M10.7d3a
  through M10.7d3c before graph/KVC smoke claims.
- M10.7d3c Post-Restore KVC Write/Skip B300 Smoke is split into M10.7d3c1
  model-free post-restore KVC decision contract and M10.7d3c2 B300 restored
  payload KVC file smoke.
- Earlier M10.7d3c2 B300 Restored Payload KVC File Smoke.
- Earlier M10.7d3c1 Post-Restore KVC Decision Contract.
- Earlier M10.7d3b B300 Restored-Graph Frontier Projection.
- Earlier M10.7d3a Graph Restore Frontier Contract.
- Earlier M10.7d2c Runtime Continued-Store B300 Replay Refresh.
- Earlier M10.7d2b Runtime KV Replay Checker Closure.
- Earlier M10.7d2a Runtime Continued-Frontier Ledger Contract.
- Earlier M10.7d1 Continued-Frontier Policy Transition Matrix.
- M10.7c3 Rust Graph Tensor Restore Next-Token Smoke is split into M10.7c3a
  through M10.7c3d before tensor restore or next-token claims.
- Earlier M10.7c3d Rust Graph Tensor Restore Next-Token Smoke.
- Earlier M10.7c3c Rust Graph Tensor Restore Readback Smoke.
- Earlier M10.7c3b Rust Graph Restore Target Mapping Contract.
- Earlier M10.7c3a Rust Memory Snapshot Raw Body Import Smoke.
- Earlier M10.7c2 Rust Disk KV Payload Byte Import Smoke.
- Earlier M10.7c1 Rust Restore Payload Header Contract.
- Earlier M10.7b Rust Graph Session Payload Reader And Writer.
- Earlier M10.7a Rust Graph Session Payload Layout Plan.
- Earlier M10.5c4d2 Rust Ratio-Boundary
  Continuation Coverage.
- Earlier M10.5c4c2b2b2b2b2b2b2b2b2b validation remains recorded below.
- Active debugging ledger: none
- B300 context: `hou2-prod1`
- B300 namespace: `default`
- B300 pod: `ds4-rust-port-b300`
- B300 node: `c1v17-b300n1-nic1`
- B300 temp kubeconfig: `/tmp/ds4-hou2-prod1.kubeconfig` for this local
  session; regenerate a temp copy in future sessions instead of treating this
  path as durable, and pass `--context hou2-prod1` explicitly because the temp
  kubeconfig can contain other contexts.
- Known local validation constraint: `ds4flash.gguf` is not present in the
  workspace, so model-backed tests and benchmark baselines need a model path or
  remote B300 execution.
- B300 model path: `/workspace/ds4/ds4flash.gguf`
- B300 model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- B300 model size: 86,720,111,488 bytes.

## Last Evidence

- M14.3b1 Fused QKV And Basic Head RMS Norm Kernels adds executable-local
  Rust cuda-oxide `dsv4_qkv_rms_norm_rows_kernel` and
  `head_rms_norm_kernel`. On B300 pod `ds4-rust-port-b300`, feature-enabled
  `ds4-cuda` tests passed with 40 tests and live cargo-oxide execution
  emitted portable `sm_80` PTX and proved asymmetric Q/KV row widths plus
  in-place head normalization on `NVIDIA B300 SXM6 AC`. Its fixture and
  checker are
  `ds4-parity/baselines/backend/m14.3b1/fused-rms-norm-kernel-smoke.json`
  and `ds4-parity/check_fused_rms_norm_kernel_smoke.py --negative-test`.
  Head RMS plus RoPE-tail fusion, the environment-controlled fused-QKV
  fallback policy, projection, Q8 kernels, route activation, and C CUDA
  removal remain unclaimed.
  Local formatting, diff, workspace tests, the 73-check comparator, and
  unified parity passed with 131 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result; live
  compilation found and corrected the in-place `DisjointSlice` read through
  a mutable device pointer before the successful B300 rerun.
- M14.3a Plain And Weighted RMS Norm Kernels adds executable-local Rust
  cuda-oxide `rms_norm_plain_kernel` and `rms_norm_weight_kernel` using the
  current-C 256-thread row reduction and a libdevice-linked reciprocal RMS
  scale. On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
  passed with 39 tests and live cargo-oxide execution emitted portable
  `sm_80` PTX and proved multi-row plain, multi-row weighted, single-row, and
  invalid-shape behavior on `NVIDIA B300 SXM6 AC`. Its fixture and checker
  are `ds4-parity/baselines/backend/m14.3a/rms-norm-kernel-smoke.json` and
  `ds4-parity/check_rms_norm_kernel_smoke.py --negative-test`. Fused QKV/head
  normalization, dense projection, Q8 kernels, route activation, and C CUDA
  removal remain unclaimed.
  Local formatting, diff, workspace tests, the 72-check comparator, and
  unified parity passed with 130 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result; the
  parity run exposed and corrected the completed M14.2e checker's active-stage
  assertion before commit.
- M14.2e M14.2 Kernel Closure Gate aggregates all fifteen M14.2 B300 proof
  artifacts and records that the Rust kernel family is available to later
  operation stages only on the existing opt-in path. Its inventory audit
  reassigns `zero_kernel` to M14.5 because current C launches it only in the
  routed-MoE atomic-down branch, and retains the packed-key branch as an
  equivalent implementation without claiming `cub::BlockRadixSort`. Its
  fixture and checker are
  `ds4-parity/baselines/backend/m14.2e/kernel-ownership-closure.json` and
  `ds4-parity/check_m14_2_kernel_closure.py --negative-test`. Default-route
  promotion and C CUDA removal remain rejected.
  Local formatting, diff, workspace tests, the 156-check closure comparator,
  and unified parity passed with 129 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review timed out without a completed result;
  adversarial self-review caught and corrected the routed-MoE-only
  `zero_kernel` assignment before closure.
- M14.2d2c5 Indexed Ascending Top-K Sort And Dispatch Policy adds
  executable-local Rust cuda-oxide `indexed_topk_sort_512_asc_kernel` and
  validated-input Rust selectors for current-C's ascending-sort gate and
  specialized top-k launch priority. The policy selects the already validated
  packed-key equivalent when its dynamic-shared launch is available, without
  claiming CUB library implementation. On B300 pod `ds4-rust-port-b300`,
  feature-enabled `ds4-cuda` tests passed with 38 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved two sorted
  rows, the multi-token sort gate, packed-key-equivalent selection, and
  fallback branch order on `NVIDIA B300 SXM6 AC`. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2c5/indexer-topk-dispatch-smoke.json`
  and `ds4-parity/check_indexer_topk_dispatch_smoke.py --negative-test`.
  Runtime route activation and C CUDA removal remain unclaimed. Local
  formatting, diff, workspace tests, the 81-check comparator, and unified
  parity passed with 128 passed, 45 skipped, and 0 failed. Non-interactive
  Claude review timed out without a completed result; adversarial self-review
  added a direct `DS4_CUDA_NO_TOPK8192` fall-through assertion before the
  final B300 rerun.
- M14.2d2c4 Chunked And Tree-Merge Top-K Kernels adds executable-local Rust
  cuda-oxide `indexer_topk_chunk_pow2_4096_kernel`,
  `indexer_topk_tree_merge_pow2_4096_kernel`, and
  `indexer_topk_merge_pow2_4096_kernel`. Its host smoke models current-C's
  single contiguous scratch allocation with explicit non-overlapping level
  offsets and per-token strides. On B300 pod `ds4-rust-port-b300`,
  feature-enabled `ds4-cuda` tests passed with 36 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved a
  two-token, ten-chunk case with one intermediate tree level, a partial final
  chunk, and a 12,288-element scratch plan on `NVIDIA B300 SXM6 AC`. Its
  fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2c4/indexer-topk-tree-kernel-smoke.json`
  and `ds4-parity/check_indexer_topk_tree_kernel_smoke.py --negative-test`.
  Specialized top-k dispatch policy, indexed ascending sort, runtime route
  activation, and C CUDA removal remain unclaimed. Local formatting, diff,
  workspace tests, the 81-check comparator, and unified parity passed with
  127 passed, 45 skipped, and 0 failed. Non-interactive Claude review timed
  out without a completed result; adversarial self-review retained the
  dispatch and route non-claims.
- M14.2d2c3 CUB-Or-Equivalent Top-K Branch adds executable-local Rust
  cuda-oxide `indexer_topk_8192_packed_key_equivalent_kernel`, retaining
  current-C's ordered-float packed key, lower-index tie order, and sentinel
  exclusion while using a dynamic-shared-memory bitonic equivalent instead
  of claiming `cub::BlockRadixSort` ownership. The first B300 launch failed
  with `DriverError(1, "invalid argument")` because the Rust host layer had
  not issued current-C's large dynamic-shared-memory opt-in; cuda-oxide
  revision `e9c0d677104751179985098f02212ff044d3ec22` adds
  `CudaFunction::set_max_dynamic_shared_memory_size`, after which the
  65,536-byte launch passed. On B300 pod `ds4-rust-port-b300`,
  feature-enabled `ds4-cuda` tests passed with 35 tests and live
  cargo-oxide execution emitted portable `sm_80` PTX and proved 4096- and
  6000-component output, positive-NaN ordering, tie order, sentinel
  exclusion, and invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Its
  fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2c3/indexer-topk-packed-kernel-smoke.json`
  and `ds4-parity/check_indexer_topk_packed_kernel_smoke.py --negative-test`.
  This stage does not claim CUB library implementation, specialized top-k
  dispatch policy, chunked merging, indexed ascending sort, runtime route
  activation, or C CUDA removal. Local formatting, diff, workspace tests, the
  80-check comparator, and unified parity passed with 126 passed, 45 skipped,
  and 0 failed. Non-interactive Claude review timed out without a completed
  result; adversarial self-review retained the CUB and dispatch non-claims.
- M14.2d2c2 Power-Of-Two Top-K Kernels adds executable-local Rust cuda-oxide
  `indexer_topk_pow2_2048_kernel`, `indexer_topk_pow2_4096_kernel`, and
  `indexer_topk_pow2_u16_8192_kernel`, preserving current-C shared-memory
  index width, descending order, and lower-index tie breaking. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 34
  tests and live cargo-oxide execution emitted portable `sm_80` PTX and
  proved each kernel output and sentinel exclusion on `NVIDIA B300 SXM6 AC`.
  Its fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2c2/indexer-topk-pow2-kernel-smoke.json`
  and `ds4-parity/check_indexer_topk_pow2_kernel_smoke.py --negative-test`.
  The stage does not claim B300 branch selection against the current-C CUB
  optimization; CUB policy, chunked merging, indexed ascending sort, runtime
  route activation, and C CUDA removal remain unclaimed. Local formatting,
  diff, workspace tests, the 76-check comparator, and unified parity passed
  with 125 passed, 45 skipped, and 0 failed. Non-interactive Claude review
  timed out without a completed result; adversarial self-review retained the
  CUB dispatch non-claim.
- M14.2d2c1 1024 Bitonic Top-K Kernel adds executable-local Rust cuda-oxide
  `indexer_topk_1024_kernel` with the current-C 1024-thread shared-memory
  bitonic network, descending score order, and lower-index tie breaking.
  On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
  with 33 tests and live cargo-oxide execution emitted portable `sm_80` PTX
  and proved full-width output, partial-width sentinel exclusion, stable tie
  ordering, and invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Its
  fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2c1/indexer-topk1024-kernel-smoke.json`
  and `ds4-parity/check_indexer_topk1024_kernel_smoke.py --negative-test`.
  Larger top-k dispatch, indexed ascending sort, runtime route activation,
  and C CUDA removal remain unclaimed. Local formatting, diff, workspace
  tests, the 69-check comparator, and unified parity passed with 124 passed,
  45 skipped, and 0 failed. Non-interactive Claude review timed out without
  a completed result; adversarial self-review retained only bounded
  kernel-shape and ordering ownership.
- M14.2d2b2c WMMA128 Tensor-Core Indexer Score Kernel And Dispatch Priority
  adds executable-local Rust cuda-oxide `indexer_scores_wmma128_kernel`
  with eight warps, native `f16` shared staging, and cuda-oxide
  `mma_m16n8k16_f32_f16` operations covering current-C's `16 x 128` score
  tile. It also adds pure Rust `select_indexer_score_kernel` matching the
  validated-input direct-one, WMMA128/64/32/base, and scalar fallback order.
  On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed
  with 32 tests and live cargo-oxide execution emitted portable `sm_80` PTX
  and proved WMMA128 output over two component blocks, eight-warp tile
  mapping, per-token weighting, NaN/negative suppression, causal masking,
  invalid-shape rejection, and score-dispatch ordering on
  `NVIDIA B300 SXM6 AC`. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2b2c/indexer-wmma128-dispatch-smoke.json`
  and `ds4-parity/check_indexer_wmma128_dispatch_smoke.py --negative-test`.
  Specialized top-k dispatch, runtime route activation, and C CUDA removal
  remain unclaimed. Local formatter/diff checks, workspace tests, the
  84-check WMMA128/dispatch comparator, and unified parity passed with 123
  passed, 45 skipped, and 0 failed. Non-interactive Claude review produced
  no completed result before its timeout; adversarial self-review added
  direct-one-disabled and global-WMMA-disabled selector checks before the
  final B300 rerun.
- M14.2d2b2b WMMA64 Tensor-Core Indexer Score Kernel adds executable-local
  Rust cuda-oxide `indexer_scores_wmma64_kernel` with four warps, native
  `f16` shared staging, and cuda-oxide `mma_m16n8k16_f32_f16` operations
  covering current-C's `16 x 64` score tile. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 30
  tests and live cargo-oxide execution emitted portable `sm_80` PTX and
  proved WMMA64 output over two component blocks, four-warp tile mapping,
  per-token weighting, NaN/negative suppression, causal masking, and
  invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Its fixture and checker
  are `ds4-parity/baselines/backend/m14.2d2b2b/indexer-wmma64-kernel-smoke.json`
  and `ds4-parity/check_indexer_wmma64_kernel_smoke.py --negative-test`.
  WMMA128 priority, specialized top-k dispatch, runtime route activation,
  and C CUDA removal remain unclaimed. Local formatter/diff checks, workspace
  tests, the 73-check WMMA64 comparator, and unified parity passed with 122
  passed, 45 skipped, and 0 failed. Non-interactive Claude review produced
  no completed result before its timeout; adversarial self-review compared
  four-warp column ownership, accumulator scatter, causal early exit, and
  explicit `fmaxf` semantics against current C.
- M14.2d2b2a WMMA32 Tensor-Core Indexer Score Kernel adds executable-local
  Rust cuda-oxide `indexer_scores_wmma32_kernel` with two warps, native
  `f16` shared staging, and cuda-oxide `mma_m16n8k16_f32_f16` operations
  covering current-C's `16 x 32` score tile. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 29
  tests and live cargo-oxide execution emitted portable `sm_80` PTX and
  proved WMMA32 output over two component blocks, two-warp tile mapping,
  per-token weighting, NaN/negative suppression, causal masking, and
  invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Its fixture and checker
  are `ds4-parity/baselines/backend/m14.2d2b2a/indexer-wmma32-kernel-smoke.json`
  and `ds4-parity/check_indexer_wmma32_kernel_smoke.py --negative-test`.
  WMMA64/WMMA128 dispatch, specialized top-k dispatch, runtime route
  activation, and C CUDA removal remain unclaimed. Local formatter/diff
  checks, workspace tests, the 73-check WMMA32 comparator, and unified
  parity passed with 121 passed, 45 skipped, and 0 failed. Non-interactive
  Claude review produced no completed result before its timeout; adversarial
  self-review compared two-warp staging, accumulator scatter, causal early
  exit, and explicit `fmaxf` semantics against current C.
- M14.2d2b1 Base Tensor-Core Indexer Score Kernel adds executable-local Rust
  cuda-oxide `indexer_scores_wmma_kernel` with native `f16` shared staging
  and two `mma_m16n8k16_f32_f16` calls covering current-C's `16 x 16`
  score tile. On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda`
  tests passed with 28 tests and live cargo-oxide execution emitted portable
  `sm_80` PTX and proved base WMMA output, both eight-column MMA halves,
  per-token weighting, NaN/negative suppression, causal masking, and
  invalid-shape rejection on `NVIDIA B300 SXM6 AC`.
  Its fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2b1/indexer-wmma-kernel-smoke.json`
  and `ds4-parity/check_indexer_wmma_kernel_smoke.py --negative-test`.
  A first device compile exposed unsupported generic `u32::min` drop glue;
  explicit scalar comparisons preserved behavior and enabled the successful
  PTX build. Widened WMMA dispatch, specialized top-k dispatch, runtime
  route activation, and C CUDA removal remain unclaimed. Local workspace
  tests, formatter/diff checks, the 72-check base WMMA comparator, and
  unified parity passed with 120 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review produced no completed result before its
  timeout; adversarial self-review expanded the live fixture to prove
  weighted output and NaN/negative suppression before final B300 execution.
- M14.2d2a Direct-One Indexer Score Kernel adds executable-local Rust
  cuda-oxide `indexer_score_one_direct_kernel` with current-C-shaped
  128-thread geometry, four-warp `warp::shuffle_down_f32` reduction,
  positive-score weighting, and causal negative-infinity masking. On B300
  pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 27
  tests and live cargo-oxide execution lowered the warp shuffle, emitted
  portable `sm_80` PTX, and proved direct output, causal masking,
  NaN/negative clamp behavior, and invalid-shape rejection on
  `NVIDIA B300 SXM6 AC`. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.2d2a/indexer-direct-kernel-smoke.json`
  and `ds4-parity/check_indexer_direct_kernel_smoke.py --negative-test`.
  WMMA score dispatch, specialized top-k dispatch, runtime route activation,
  and C CUDA removal remain unclaimed. Local formatter/diff checks, workspace
  tests, the 66-check direct indexer comparator, and unified parity passed
  with 119 passed, 45 skipped, and 0 failed. Non-interactive Claude review
  produced no completed result before its timeout; adversarial self-review
  confirmed fixed lane/head grouping, shuffle-down reduction, explicit NaN
  clamp behavior, and the WMMA/top-k non-claims.
- M14.2d1 Scalar Indexer Selection Kernels adds executable-local Rust
  cuda-oxide `indexer_scores_kernel`, `indexer_topk_kernel`, and
  `topk_mask_kernel` paths matching current-C scalar fallback reduction,
  positive-score weighting, causal negative-infinity masking, stable
  earlier-index top-k ties, and selected-row mask values. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 26
  tests and live cargo-oxide execution emitted portable `sm_80` PTX and
  proved score output, causal masking, top-k output and tie ordering, mask
  output, and invalid-shape rejection on `NVIDIA B300 SXM6 AC`. Its fixture
  and checker are
  `ds4-parity/baselines/backend/m14.2d1/indexer-scalar-kernel-smoke.json`
  and `ds4-parity/check_indexer_scalar_kernel_smoke.py --negative-test`.
  Direct-one/WMMA scoring, specialized top-k dispatch, runtime route
  activation, and C CUDA removal remain unclaimed. Local formatter/diff
  checks, workspace tests, the 73-check scalar indexer comparator, and
  unified parity passed with 118 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review produced no completed result before its
  timeout; adversarial self-review confirmed scalar `fmaxf` handling, stable
  strict-greater top-k ties, and the explicit optimized-dispatch non-claim,
  and aligned mask launch sizing with current C before the final B300 rerun.
- M14.2c Embedding Kernel Pair adds executable-local Rust cuda-oxide
  `embed_token_hc_kernel` and `embed_tokens_hc_kernel` paths using primitive
  `f16` loads widened to `f32`. It proves repeated hidden-copy rows and the
  current-C batch rule that maps negative or out-of-vocabulary tokens to row
  zero. The Rust single-token helper rejects an invalid token before launch,
  strengthening current-C's unchecked invalid-input edge. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 25
  tests and live cargo-oxide execution emitted portable `sm_80` PTX and
  proved single-token output, batch fallback output, shape rejection, and
  invalid single-token rejection on `NVIDIA B300 SXM6 AC`. Its fixture and
  checker are
  `ds4-parity/baselines/backend/m14.2c/embedding-kernel-smoke.json` and
  `ds4-parity/check_embedding_kernel_smoke.py --negative-test`. Model-range
  consumption, indexer/top-k, route activation, and C CUDA removal remain
  unclaimed. Local formatting, diff, and workspace tests passed; the
  69-check embedding comparator and unified parity passed with 117 passed,
  45 skipped, and 0 failed. Non-interactive Claude review timed out without
  a completed result; adversarial self-review confirmed valid-call and
  batch-fallback behavior and the explicit single-token safety strengthening.
- M14.2b2 SwiGLU Libdevice Path adds the executable-local Rust cuda-oxide
  `swiglu_kernel` path with current-C-shaped finite and NaN clamps, unclamped
  behavior, SiLU exponential, output weighting, and host-side shape
  rejection. It pins cuda-oxide
  revision `d4791b7002152af3b7f6b15a48d7f5acd7a63011`, which repairs
  the observed `__nv_expf` failure by emitting portable PTX and linking
  libdevice into a cubin targeted to the executing CUDA context. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 24
  tests; live cargo-oxide execution emitted portable `sm_80` PTX, generated
  `ds4_cuda_swiglu_smoke.sm_103.cubin`, and proved matching clamped and
  unclamped output plus invalid-shape rejection without target overrides. Its
  fixture and checker
  are `ds4-parity/baselines/backend/m14.2b2/swiglu-kernel-smoke.json` and
  `ds4-parity/check_swiglu_kernel_smoke.py --negative-test`. Directional
  projection ownership is retained; embedding, indexer/top-k, route
  activation, and C CUDA removal remain unclaimed. Local workspace tests,
  formatter/diff checks, the 73-check SwiGLU comparator, and unified parity
  passed with 116 passed, 45 skipped, and 0 failed. Non-interactive Claude
  review timed out without a completed result; adversarial self-review found
  and fixed NaN clamp handling using explicit IEEE-754 bit classification
  before the final B300 run.
- M14.2b1 Directional Steering Projection Kernel introduces the
  executable-local Rust cuda-oxide `directional_steering_project_kernel`
  path with one block per row, `SharedArray<f32, 256>` reduction storage,
  block synchronization, and in-place projection. On B300 pod
  `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests passed with 23
  tests and live cargo-oxide execution selected portable `sm_80` while
  proving directional output and invalid-shape rejection. Its fixture and
  checker are
  `ds4-parity/baselines/backend/m14.2b1/directional-steering-kernel-smoke.json`
  and `ds4-parity/check_directional_steering_kernel_smoke.py --negative-test`.
  A combined SwiGLU attempt recorded the remaining blocker: `f32::exp()`
  emits `__nv_expf`, selects cuda-oxide NVVM IR output, and CUDA 13.2
  `libnvvm` rejects its opaque-pointer function signature with
  `parse expected type`. SwiGLU, model-backed kernels, route activation, and
  C CUDA removal remain unclaimed. Local formatter, diff, and workspace tests
  passed; the 71-check directional comparator and unified parity report
  passed with 110 passed, 50 skipped, and 0 failed. Non-interactive Claude
  review timed out without a completed result; self-review retained the
  in-place row-ownership/synchronization proof and explicit unowned SwiGLU
  blocker.
- M14.2a Add And Repeat Elementwise Kernels introduces the executable-local
  Rust cuda-oxide `add_kernel` and `repeat_hc_kernel` smoke path with
  current-C-shaped 256-thread launch geometry and safe disjoint output
  writes. On B300 pod `ds4-rust-port-b300`, feature-enabled `ds4-cuda` tests
  passed with 22 tests and live cargo-oxide execution selected portable
  `sm_80` while proving add output, repeated-HC-row output, add bounds
  rejection, and repeat-shape rejection. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.2a/elementwise-kernel-smoke.json` and
  `ds4-parity/check_elementwise_kernel_smoke.py --negative-test`. Embedding,
  indexer/top-k, SwiGLU, directional steering, route activation, and C CUDA
  removal remain unclaimed. Local formatter, diff, and workspace tests
  passed; the 69-check comparator passed and unified parity passed with 109
  passed, 50 skipped, and 0 failed. Non-interactive Claude review timed out
  without a completed result; adversarial self-review corrected the
  `repeat_hc` wrapper to preserve current-C's 64-bit shape product before the
  recorded B300 execution.
- M14.1c Substrate Route Closure Gate closes the opt-in Rust resource
  substrate boundary for later kernel stages. It corrects the M14.0
  inventory after the Q8 policy milestone proved that
  `ds4_gpu_cache_q8_f16_range`, `dequant_q8_0_to_f16_kernel`, and
  `dequant_q8_0_to_f32_kernel` remain M14.3 ownership rather than M14.1;
  M14.1 owns only `fill_f32_kernel` among CUDA kernels. The Rust substrate
  now also exposes current-C's no-op `begin_commands` operation and the B300
  fill smoke invokes it before command completion. Its fixture and checker
  are `ds4-parity/baselines/backend/m14.1c/substrate-route-closure.json` and
  `ds4-parity/check_substrate_route_closure.py --negative-test`. This closure
  does not activate the default route or allow `ds4_cuda.cu` removal. Local
  formatter, diff, and workspace tests passed; the updated 81-check M14.1b4
  comparator and 139-check closure comparator passed; B300 feature-enabled
  tests passed with 21 tests and the live cargo-oxide fill execution reported
  `begin_is_noop:true`; unified parity passed with 108 passed, 50 skipped,
  and 0 failed. Non-interactive Claude review timed out without a completed
  result; adversarial self-review confirmed the deferred M14.3 ownership,
  no-op command mapping, and no route or removal overclaim.
- M14.1b4 Fill Kernel And Command Lifetime adds the opt-in
  `cuda-oxide-kernels` feature and executable-local
  `ds4-cuda-fill-lifetime-smoke`. Its Rust `#[kernel] fill_f32` uses the
  current-C 256-thread launch shape with explicit count semantics, while the
  substrate now exposes current-C's no-op begin and context-wide
  flush/end/synchronize completion wrappers. Initial B300 execution proved
  two toolchain defects before the
  final success: library-only embedded modules were not retained in the
  executable, and `cargo oxide run` forced `sm_103` while `/usr/bin/llc`
  emitted invalid `.version 6.0 / .target sm_103` PTX, producing CUDA JIT
  error 218. The cuda-oxide tool fix is
  `981e3244a107d84d807cfb087793269c477cc764`; with that revision the B300
  run selected portable `sm_80` and proved prefix fill, negative-infinity
  fill, zero-count no-op, bounds rejection, and context-wide completion on
  `NVIDIA B300 SXM6 AC`. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b4/fill-command-lifetime-smoke.json`
  and `ds4-parity/check_fill_command_lifetime_smoke.py --negative-test`.
  This remains an executable-local opt-in proof: library embedded-kernel
  retention, dequant kernels, graph compute kernels, runtime graph
  integration, and default-route ownership remain unclaimed. Local workspace
  tests, formatter/diff checks, the 77-check comparator and retained M14
  checks, B300 feature-enabled `ds4-cuda` tests, B300 `cargo-oxide` tests and
  kernel execution, and unified parity (107 passed, 50 skipped, 0 failed)
  passed. Non-interactive Claude review timed out without a completed result;
  adversarial self-review found no remaining ownership, fill-semantic,
  synchronization, or evidence defect.
- M14.1b3b Q8 Cache And Quality Policy pins cuda-oxide revision
  `aabe10dc4fa0086375104458909e222d1ac1cfe3`, which adds typed
  `Blas::set_math_mode(BlasMathMode)` over the header-verified
  `cublasSetMathMode` ABI. The Rust host policy reproduces current-C Q8/F16
  eligibility, attention-output preload suppression, reserve/budget and
  failure-disable transitions, Q8/F32 optional-preload selection, and
  TF32/default-math quality decisions. On B300 pod `ds4-rust-port-b300`,
  `cublas-sys`, full `cuda-core`, and feature-enabled `ds4-cuda` tests
  passed; the live smoke applied TF32 fast mode and both default-math paths
  through cuBLAS. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b3b/q8-quality-policy-smoke.json` and
  `ds4-parity/check_q8_quality_policy_smoke.py --negative-test`. Converted
  Q8 buffers and their failure-time synchronization/release, dequant kernels
  assigned to M14.3, DS4 compute kernels, and default-route ownership remain
  unclaimed. Local workspace tests, formatter/diff checks, the 71-check
  comparator and retained M14 checks, B300 `cublas-sys` and full
  `cuda-core` tests, B300 feature-enabled `ds4-cuda` tests, and unified
  parity (106 passed, 50 skipped, 0 failed) passed. Non-interactive Claude
  review timed out without a completed result; adversarial self-review fixed
  the reserve-equality boundary test and narrowed failure ownership to
  disable-state policy before finding no remaining policy, ABI, or
  bounded-claim defect.
- M14.1b3a Managed KV And Memory Report Policy pins cuda-oxide revision
  `0ec61156a7c5d65802402898b7a197bfff266d31`, which adds the reusable
  `CudaContext::memory_info()` API. The opt-in Rust policy reproduces
  current-C managed-KV thresholds and reserve selection and exact memory
  report formatting. On B300 pod `ds4-rust-port-b300`, full `cuda-core`
  tests and feature-enabled `ds4-cuda` tests passed; the live smoke queried
  valid device capacity, allocated managed memory, and exercised empty,
  forced-managed, query-failure, sufficient-capacity, reserve-pressure, and
  context-exceeds-free choices. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b3a/allocation-policy-smoke.json` and
  `ds4-parity/check_allocation_policy_smoke.py --negative-test`. Q8 caches,
  quality-mode BLAS selection, kernels, and default-route ownership remain
  unclaimed. Local workspace tests, formatter/diff checks, the 64-check
  comparator and retained M14 checks, full B300 `cuda-core` tests, B300
  feature-enabled `ds4-cuda` tests, predecessor model-map closure smoke, and
  unified parity (105 passed, 50 skipped, 0 failed) passed. Non-interactive
  Claude review timed out without a completed result; adversarial self-review
  found no threshold, reserve, transient-capacity, report-format,
  dependency-pin, or bounded-claim issue.
- M14.1b2c Model Map Cache Closure extends the opt-in Rust range cache with
  contained-range reuse, Linux source-page discard advisory policy, explicit
  non-TTY progress emission, and cache-lifetime reset evidence. On B300 pod
  `ds4-rust-port-b300`, an 8,192-byte range served a 257-byte interior
  readback exactly without re-upload; two chunks issued two file advice calls
  totaling 8,192 bytes and two page-aligned mapping advice calls totaling
  16,384 bytes. A keep-pages cache suppressed advice, disabled progress
  emitted no message, and a fresh cache began empty. Its fixture and checker
  are `ds4-parity/baselines/backend/m14.1b2c/model-map-closure-smoke.json`
  and `ds4-parity/check_model_map_closure_smoke.py --negative-test`.
  Physical eviction, runtime environment/terminal selection wiring, DS4
  kernels, and default-route ownership remain unclaimed.
  Local workspace tests, formatter/diff checks, the 84-check comparator and
  retained M14 checks, B300 feature tests and predecessor async staging smoke,
  and unified parity (104 passed, 50 skipped, 0 failed) passed. Local feature
  compilation cannot run without CUDA headers; B300 supplied that gate.
  Non-interactive Claude review was unavailable because the CLI reported
  `Not logged in`; adversarial self-review fixed a progress-threshold
  overflow edge before finding no remaining pointer, advisory-claim,
  progress, lifetime, or bounded-claim issue.
- M14.1b2b3b2 Asynchronous Staging Ring And Budget Policy adds an opt-in
  Rust four-slot pinned upload ring with CUDA-event-guarded refill, persistent
  direct-I/O disable-after-selected-error policy, and bounded device-arena
  admission. On B300 pod `ds4-rust-port-b300`, seven direct-I/O chunks used
  four slots with two reuse waits, two ranges shared one 32,768-byte arena
  totaling 28,672 bytes, the next byte selected budget fallback, and exact
  readback passed for both admitted ranges. A boundary probe rejected a new
  arena whose aligned reservation exceeded remaining budget, and intermediate
  upload errors drain queued copies before staging-slot state clears. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b2b3b2/model-async-staging-smoke.json`
  and `ds4-parity/check_model_async_staging_smoke.py --negative-test`.
  Feature-enabled B300 tests validate the direct-I/O disable errno policy;
  the live smoke does not claim an induced I/O error. Source-page
  discard/progress behavior, DS4 kernels, and default-route ownership remain
  false. Local workspace tests, formatter/diff checks, the 96-check
  comparator and retained M14 checks, B300 feature tests and predecessor
  direct-I/O smoke, and unified parity (103 passed, 50 skipped, 0 failed)
  passed. Local feature compilation cannot run without CUDA headers; B300
  supplied that gate. Non-interactive Claude review was unavailable because
  the CLI reported `Not logged in`; adversarial self-review fixed error-path
  draining and aligned new-arena admission defects before finding no remaining
  lifetime, policy, or bounded-claim issue.
- M14.1b2b3b1 Direct-I/O Pinned Read Selection adds a Rust
  `O_DIRECT`-or-buffered pinned stage and synchronized selected-range device
  upload. On B300 pod `ds4-rust-port-b300`, requested range `13..4109` read
  from aligned direct-I/O window `0..8192` at alignment `4096`, then matched
  exact CUDA readback. The model's final 13 bytes took the buffered fallback
  because its file end is not direct-I/O aligned, and that exact CUDA
  readback also matched. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b2b3b1/model-direct-io-smoke.json` and
  `ds4-parity/check_model_direct_io_smoke.py --negative-test`.
  Asynchronous staging-ring/event scheduling, cache-budget policy, persistent
  disable-after-error state, DS4 kernels, and default-route ownership remain
  false. Validation passed local workspace tests, formatting and diff checks,
  the 79-check new comparator, retained M14 comparators, B300 feature-enabled
  crate tests and predecessor HMM smoke, and unified parity with 102 passed,
  50 skipped, and 0 failed. Non-interactive Claude review was unavailable
  because the CLI reported `Not logged in`; adversarial self-review found no
  lifetime, alignment, fallback, or bounded-claim defect.
- M14.1b2b3a Pageable HMM Range Strategy pins corrected cuda-oxide revision
  `361300ea643688eea87eaa215d9a62a5e74a30e6`, after integration review
  changed borrowed asynchronous pageable prefetch from a safe API to an
  explicitly unsafe operation requiring stream-completion lifetime. DS4 wraps
  that operation in a synchronized opt-in proof path. On B300 pod
  `ds4-rust-port-b300`, requested range `13..4109` expanded to pageable range
  `0..8192`; CUDA reported pageable-memory access with host page-table access
  disabled, accepted read-mostly and preferred-device advice plus prefetch,
  and direct HMM readback matched the exact requested bytes. Its fixture and
  checker are
  `ds4-parity/baselines/backend/m14.1b2b3a/model-pageable-hmm-smoke.json`
  and `ds4-parity/check_model_pageable_hmm_smoke.py --negative-test`.
  Asynchronous production prefetch policy, O_DIRECT/pinned staging, DS4
  kernels, and default-route ownership remain false. Validation passed
  workspace tests and formatting, the 73-check new comparator, retained M14
  comparators, B300 feature tests and predecessor smoke, and unified parity
  with 101 passed, 50 skipped, and 0 failed. Non-interactive Claude review
  was unavailable because the CLI reported `Not logged in`; self-review found
  and fixed the cuda-oxide asynchronous borrowed-prefetch safety defect
  before DS4 pinned the dependency.
- M14.1b2b2 Registered Range Strategy pins cuda-oxide revision
  `b938480882f208045bc36ecf29da1ec5531d55ba` and adds page-aligned
  read-only mapped-host registration selection with an explicit mmap-sourced
  device-copy fallback. On B300 pod `ds4-rust-port-b300`, requested range
  `13..4109` expanded to registration range `0..8192`; CUDA returned error
  `801` (`operation not supported`) for the read-only registration attempt,
  and the fallback copied/read back the exact requested 4096 bytes and reused
  its cache entry. Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b2b2/model-registered-range-smoke.json`
  and `ds4-parity/check_model_registered_range_smoke.py --negative-test`.
  Successful B300 zero-copy registration, current-C cross-range suppression
  after unsupported registration, pageable HMM, O_DIRECT/asynchronous staging
  and cache-budget policy, DS4 kernels, and default-route ownership remain
  false. Validation passed local workspace tests and formatting, B300 feature
  tests and the prior strategy smoke, retained M14 comparator gates, and
  unified parity with 100 passed, 50 skipped, and 0 failed. Non-interactive
  Claude review was unavailable because the CLI reported `Not logged in`;
  self-review corrected the fallback source to the current-C mmap copy branch.
- M14.1b2b1 File-Staged Range Strategy adds explicit
  `ModelRangeStrategy::{MmapDeviceCopy, FileStagedDeviceCopy}` dispatch and
  strategy-keyed range cache entries under the opt-in `ds4-cuda` feature. On
  B300 pod `ds4-rust-port-b300`, the smoke independently cached and reused the
  same 4096-byte pinned-model prefix from mmap and file-descriptor positional
  reads, then obtained matching device readbacks on `NVIDIA B300 SXM6 AC`.
  Its fixture and checker are
  `ds4-parity/baselines/backend/m14.1b2b1/model-range-strategy-smoke.json`
  and `ds4-parity/check_model_range_strategy_smoke.py --negative-test`.
  Registered mapped-host range fallback, pageable HMM, O_DIRECT/asynchronous
  staging and cache-budget policy, DS4 kernels, and default-route ownership
  remain false. A live SHA256 refresh matched the recorded GGUF identity.
  Validation passed `cargo test --workspace`, `cargo fmt --all -- --check`,
  the 73-check M14.1b2b1 checker, retained M14 gates, `git diff --check`, and
  unified parity with 99 passed, 50 skipped, and 0 failed. Non-interactive
  Claude review failed to complete and was terminated without a result;
  adversarial self-review found no material strategy-keying, source-lifetime,
  synchronization, bounded-claim, or default-route issue.
- M14.1b2a Owned Mmap Device Range Copy adds Rust-owned `MappedModelFile`
  lifetime and `ModelRangeCache` device-copy/reuse behavior under the opt-in
  `ds4-cuda` feature. On B300 pod `ds4-rust-port-b300`, the feature compile
  first rejected an unnecessary `Debug` derive on `DeviceBuffer<u8>`; after
  removing that derive, the smoke mmaped pinned model
  `/workspace/ds4/ds4flash.gguf`, rejected an invalid range, copied/read back
  a 4096-byte device range exactly, and reused the cached entry on
  `NVIDIA B300 SXM6 AC`. The fixture and checker are
  `ds4-parity/baselines/backend/m14.1b2a/model-range-copy-smoke.json` and
  `ds4-parity/check_model_range_copy_smoke.py --negative-test`; registered,
  HMM, direct-I/O, Q8, kernel, and route strategy ownership remain false.
  Adversarial self-review found and fixed a rejected-null-address `mmap`
  cleanup leak, and the B300 smoke rerun passed after that fix. Validation
  passed `cargo test --workspace`, `cargo fmt --all -- --check`, the M14.1b2a
  checker with 64 checks, retained predecessor gates, `git diff --check`, and
  unified parity with 98 passed, 50 skipped, and 0 failed. Non-interactive
  Claude review could not run because the local CLI reported `Not logged in`;
  post-fix self-review found no material mmap/cache lifetime,
  synchronization, bounded-claim, or default-route issue.
- M14.1b1 Bounded Model Residency Handles extends the opt-in `rust/ds4-cuda`
  crate with managed read-mostly/preferred-device advice and prefetch,
  mapped-host allocation, and registered-host range guards. Its B300 smoke
  reads a 4096-byte prefix from pinned model
  `/workspace/ds4/ds4flash.gguf` (86,720,111,488 bytes, SHA256
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`)
  and passed managed advice/prefetch, mapped device-pointer, and registered
  host-pointer checks on `NVIDIA B300 SXM6 AC`. The fixture and checker are
  `ds4-parity/baselines/backend/m14.1b1/model-residency-handles-smoke.json`
  and `ds4-parity/check_model_residency_handles_smoke.py --negative-test`;
  complete-model-map, kernel, and route ownership remain false. A live
  `sha256sum` refresh confirmed the model identity above after adversarial
  self-review identified that the smoke output itself records size but not
  hash. Validation passed `cargo test --workspace`, `cargo fmt --all --
  --check`, the M14.1b1 checker with 64 checks, the retained M14.1a/M14.0
  gates, `git diff --check`, and unified parity with 97 passed, 50 skipped,
  and 0 failed. Non-interactive Claude review could not run because the local
  CLI reported `Not logged in`; post-fix self-review found no material
  lifetime, synchronization, bounded-claim, or default-route issue.
- M14.1a Host Substrate Buffer Roundtrip adds the opt-in `rust/ds4-cuda`
  crate, pinned `cuda-core` dependency from `cuda-oxide` revision
  `0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200`, executable
  `ds4-cuda-substrate-smoke`, fixture
  `ds4-parity/baselines/backend/m14.1a/cuda-oxide-substrate-smoke.json`, and
  checker `ds4-parity/check_cuda_oxide_substrate_smoke.py --negative-test`.
  The B300 pod was confirmed running on node `c1v17-b300n1-nic1`; its
  pod-local build required explicit `CARGO_HOME=/tmp/cargo`,
  `RUSTUP_HOME=/tmp/rustup`, `nightly-2026-04-03`, and installed
  `libclang-dev`. The feature-enabled smoke then executed on
  `NVIDIA B300 SXM6 AC`, passing device roundtrip, zeroed-buffer roundtrip,
  and managed-buffer lifetime checks while reporting
  `owns_ds4_kernels=false` and `changes_default_route=false`. Validation
  passed: `cargo test --workspace`, `cargo fmt --all -- --check`, `python3
  ds4-parity/check_cuda_oxide_substrate_smoke.py --negative-test` (53 checks),
  `python3 ds4-parity/check_cuda_rust_ownership_inventory.py --negative-test`
  (124 checks), `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` (96 passed, 50
  skipped, 0 failed). Non-interactive Claude review could not run because the
  local CLI reported `Not logged in`; adversarial self-review found no
  material issue in the opt-in feature boundary, immutable dependency
  revision, limited B300 ownership claim, or retained current-C
  oracle/default route. LLVM 21 remains a prerequisite for later cuda-oxide
  kernel-compilation stages, not for this host-substrate smoke.
- M14.0 adds
  `ds4-parity/baselines/backend/m14.0/cuda-rust-ownership-inventory.json` and
  `ds4-parity/check_cuda_rust_ownership_inventory.py --negative-test`. The
  inventory records 81 public CUDA ABI functions mirrored by Rust FFI, two
  additional CUDA-only exported helpers, 113 unique CUDA kernel symbols, and
  the verified `cuda-oxide` `main` revision
  `0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200`. All functions and kernels are
  assigned to M14.1 through M14.5; default-route promotion and
  `ds4_cuda.cu` removal remain blocked until M14.6. Validation passed:
  `python3 ds4-parity/check_cuda_rust_ownership_inventory.py --negative-test`
  (124 checks), `python3 ds4-parity/check_post_m13_roadmap_decision.py
  --negative-test` (99 checks), and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` (95 passed, 50 skipped, 0 failed). Non-interactive
  Claude review could not run because the local CLI reported `Not logged in`;
  adversarial self-review found no material issue in the inventory extraction,
  unique stage assignment, source-hash drift guard, or successor-status
  compatibility changes.
- Post-M13 roadmap decision adds
  `ds4-parity/baselines/roadmap/post-m13/post-m13-roadmap-decision.json` and
  `ds4-parity/check_post_m13_roadmap_decision.py --negative-test`. It records
  that `RUST_PORT_ROADMAP.md` is complete through M13.5, no next implementation
  stage is selected, the default route remains `current-backend`, current-backend
  sidecars from M13.5 still block C/GPU backend removals, and the open decisions
  are deferred to a future roadmap that must start from new oracles. Validation
  passed: `python3 ds4-parity/check_post_m13_roadmap_decision.py
  --negative-test` (100 checks), `python3
  ds4-parity/check_backend_expanded_route_closure.py --negative-test` (279
  checks), and `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
  (94 passed, 50 skipped, 0 failed).
- M13.5 adds
  `ds4-parity/baselines/backend/m13.5/expanded-route-gate.json`,
  `ds4-parity/baselines/backend/m13.5/expanded-route-closure.json`, and
  `ds4-parity/check_backend_expanded_route_closure.py --negative-test` for the
  expanded embedding/indexer route. The closure matrix records
  `ds4_gpu_embed_token_hc_tensor`, `ds4_gpu_embed_tokens_hc_tensor`,
  `ds4_gpu_indexer_score_one_tensor`, and `ds4_gpu_indexer_topk_tensor` as
  opt-in Rust replacement slices, keeps
  `ds4_gpu_indexer_scores_prefill_tensor`,
  `ds4_gpu_indexer_scores_decode_batch_tensor`, and
  `ds4_gpu_dsv4_topk_mask_tensor` on retained current-backend sidecars, and
  blocks default-route activation, general backend replacement, kernel
  replacement, and removals until a post-M13 decision. Validation passed:
  `python3 ds4-parity/check_backend_expanded_route_closure.py
  --negative-test` (279 checks), `cargo test -p ds4-gpu backend_route_gate` (4
  tests), `python3 ds4-parity/check_backend_runtime_route_gate.py
  --negative-test` (135 checks), `python3
  ds4-parity/check_backend_batch_indexer_fixtures.py --negative-test` (182
  checks), and `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
  (93 passed, 50 skipped, 0 failed).
- M13.4 adds
  `ds4-parity/baselines/backend/m13.4/batch-indexer-fixture-bundle.json` and
  `ds4-parity/check_backend_batch_indexer_fixtures.py --negative-test` for the
  three M13.1 fixture-gap operations:
  `ds4_gpu_indexer_scores_prefill_tensor`,
  `ds4_gpu_indexer_scores_decode_batch_tensor`, and
  `ds4_gpu_dsv4_topk_mask_tensor`. It also adds the missing current-C debug dump
  hook for `comp_mask` after `ds4_gpu_dsv4_topk_mask_tensor`. The bundle records
  B300-rerunnable current-C fixture contracts, exact source anchors,
  output/dtype contracts, and rerun commands while keeping raw tensor bodies out
  of the repository and route/default-route/general/backend/kernel replacement
  claims false. Validation passed: `python3
  ds4-parity/check_backend_batch_indexer_fixtures.py --negative-test` (182
  checks), `arch -arm64 make ds4-prefill-whole-short-oracle-dump`, B300 fixture
  probe (`m134_fixture_probe=ok`), and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` (92 passed, 50 skipped,
  0 failed).
- M13.3 adds explicit Rust replacement slice descriptors for
  `ds4_gpu_indexer_score_one_tensor` and `ds4_gpu_indexer_topk_tensor`,
  `ds4-parity/baselines/backend/m13.3/indexed-decode-selection-replacement-slices.json`,
  and `ds4-parity/check_backend_indexed_decode_slice.py --negative-test`. The
  two slices use the M13.1 pair-comparator-ready rows and the M10.5c4d3 long
  indexed-attention comparator, require explicit per-slice selection, keep CPU,
  Metal, and runtime-default-route fail-closed, and leave runtime route,
  default-route, general backend replacement, and kernel replacement claims
  false. Validation passed: `python3
  ds4-parity/check_backend_indexed_decode_slice.py --negative-test` (195
  checks), `cargo test -p ds4-gpu replacement_slice` (6 tests), `python3
  ds4-parity/check_backend_replacement_slice.py --negative-test` (85 checks),
  `python3 ds4-parity/check_backend_batched_embedding_slice.py --negative-test`
  (96 checks), and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` (91 passed, 50 skipped, 0 failed).
- M13.2 adds the Rust replacement slice descriptor for
  `ds4_gpu_embed_tokens_hc_tensor`,
  `ds4-parity/baselines/backend/m13.2/batched-embedding-replacement-slice.json`,
  and `ds4-parity/check_backend_batched_embedding_slice.py --negative-test`.
  The `ds4-backend-replacement-slice` emitter now accepts explicit `--slice`
  selection while preserving the M12.4 default output. The M13.2 slice uses the
  M13.1 pair-comparator-ready row for batched embedding, keeps CPU, Metal, and
  runtime-default-route fail-closed, and leaves runtime route, default-route,
  general backend replacement, and kernel replacement claims false. Validation
  passed: `python3 ds4-parity/check_backend_batched_embedding_slice.py
  --negative-test` (96 checks), `cargo test -p ds4-gpu replacement_slice` (4
  tests), `python3 ds4-parity/check_backend_replacement_slice.py
  --negative-test` (85 checks), and
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` (90 passed, 50
  skipped, 0 failed).
- M13.1 adds
  `ds4-parity/baselines/backend/m13.1/embedding-indexer-expansion-matrix.json`
  and `ds4-parity/check_backend_expansion_matrix.py --negative-test`. The
  matrix compares the six remaining M12.6 `embedding_and_indexer` operations
  against the M13.0 decision, M12.1 inventory, M12.6 closure, M10.2 graph
  inventory, current-C anchors, Rust facade or graph-plan anchors, and existing
  comparator paths. It marks `ds4_gpu_embed_tokens_hc_tensor`,
  `ds4_gpu_indexer_score_one_tensor`, and `ds4_gpu_indexer_topk_tensor` as
  pair-comparator-ready while keeping `ds4_gpu_indexer_scores_prefill_tensor`,
  `ds4_gpu_indexer_scores_decode_batch_tensor`, and
  `ds4_gpu_dsv4_topk_mask_tensor` as fixture-gap operations. Route,
  default-route, removal, general backend replacement, and kernel replacement
  claims remain unchanged. Validation passed `python3
  ds4-parity/check_backend_expansion_matrix.py --negative-test` with 186
  checks, Python syntax, JSON formatting, `cargo fmt --all -- --check`, `git
  diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 89 passed, 50 skipped, 0 failed. M13.2 Batched
  Embedding Replacement Slice is active.
- M13.0 adds
  `ds4-parity/baselines/backend/m13.0/backend-expansion-decision.json` and
  `ds4-parity/check_backend_expansion_decision.py --negative-test`. The
  decision chooses to broaden the existing `embedding_and_indexer` route rather
  than start a new backend family because M12.6 left one opt-in
  `ds4_gpu_embed_token_hc_tensor` replacement and six remaining operations in
  that same family. The M13 split covers M13.1 fixture-matrix closure, M13.2
  batched embedding replacement, M13.3 indexed decode selection replacement,
  M13.4 batch indexer fixture-gap closure, and M13.5 route-gate/closure. It
  keeps removals, default-route replacement, general backend replacement, and
  kernel replacement claims false. Validation passed `python3
  ds4-parity/check_backend_expansion_decision.py --negative-test` with 186
  checks, Python syntax, JSON formatting, the M12.6 closure checker, `cargo fmt
  --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 88 passed, 50
  skipped, 0 failed. M13.1 Embedding/Indexer Expansion Fixture Matrix is
  active.
- M12.6 adds
  `ds4-parity/baselines/backend/m12.6/backend-replacement-closure.json` and
  `ds4-parity/check_backend_replacement_closure.py --negative-test`. The
  closure matrix records all M12.1 backend operation families, the M12.2 tensor
  fixture families, M12.3 facade replay coverage, the single M12.4 route-gated
  replacement operation, and M12.5 opt-in route status. Removal decision:
  retain current C/CUDA/Metal backend code as both sidecar and oracle. No
  removals are allowed in M12 because only `ds4_gpu_embed_token_hc_tensor` has
  opt-in route-gated replacement coverage, the default route remains
  current-backend, and current backend artifacts are still active references.
  Validation passed Python syntax, JSON formatting, the M12.6 checker with 147
  checks and negative tests, the M12.1, M12.2, M12.3, M12.4, M12.5, and runtime
  graph closure checkers with negative tests, `cargo fmt --all -- --check`,
  `git diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 87 passed, 50 skipped, 0 failed. Post-M12 roadmap
  decision is active.
- M12.5 adds the Rust-owned runtime backend route gate descriptor
  `rust/ds4-gpu/src/backend_route_gate.rs`, the descriptor emitter
  `rust/ds4-gpu/src/bin/ds4-backend-route-gate.rs`,
  `ds4-parity/baselines/backend/m12.5/runtime-route-gate.json`, and
  `ds4-parity/check_backend_runtime_route_gate.py --negative-test`. The
  explicit opt-in route is `replacement-slice` through
  `--runtime-backend-route`; the default route remains `current-backend` and
  does not activate the replacement slice. The gate selects the M12.4
  `embedding_and_indexer` / `ds4_gpu_embed_token_hc_tensor` slice for
  `cuda-b300`, rejects CPU/Metal and runtime-default-route selectors, and keeps
  general backend replacement plus kernel replacement claims false. The checker
  ties route validation to the existing M10.9 B300 graph-route official-vector,
  long-context, tool/server, and same-session benchmark artifacts. Validation
  passed Python syntax, JSON formatting, the M12.5 checker with 135 checks and
  negative tests, the M12.1, M12.2, M12.3, M12.4, and runtime graph closure
  checkers with negative tests, `cargo fmt --all -- --check`, `cargo test -p
  ds4-gpu backend_route_gate`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 86 passed, 50
  skipped, 0 failed. M12.6
  Backend Replacement Closure And Removal Decision is active.
- M12.4 adds the Rust-owned slice descriptor
  `rust/ds4-gpu/src/replacement_slice.rs`, the descriptor emitter
  `rust/ds4-gpu/src/bin/ds4-backend-replacement-slice.rs`,
  `ds4-parity/baselines/backend/m12.4/replacement-slice.json`, and
  `ds4-parity/check_backend_replacement_slice.py --negative-test`. The first
  bounded slice is `embedding_and_indexer` / `ds4_gpu_embed_token_hc_tensor`
  against the M12.2 `first_kernel_embed_token_hc` fixture and M12.3 facade
  replay. CPU, Metal, and default-route backend selectors fail closed before
  runtime route changes, and general backend replacement plus kernel
  replacement claims remain false. Validation passed Python syntax, JSON
  formatting, the M12.4 checker with 85 checks and negative tests, the M12.1,
  M12.2, M12.3, and runtime graph closure checkers with negative tests, `cargo
  fmt --all -- --check`, `cargo test -p ds4-gpu replacement_slice`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 85 passed, 50 skipped, 0 failed. M12.5 Runtime
  Backend Route Gate is active.
- M12.3 adds `ds4-parity/baselines/backend/m12.3/facade-replay.json` and
  `ds4-parity/check_backend_facade_replay.py --negative-test`, wiring the
  checker into the unified parity report and README. The replay artifact maps
  each selected M12.2 fixture to ordered `DecodeBackend` calls, tensor
  bindings, synchronized candidate evidence, and current facade error
  propagation while preserving no backend replacement and no runtime route
  change claims. Validation passed Python syntax, JSON formatting, the M12.3
  checker with 769 checks and negative tests, the M12.2 checker with 576
  checks and negative tests, the runtime graph closure matrix with negative
  tests, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 84 passed, 50
  skipped, 0 failed. M12.4 First Backend Replacement Slice became active.
- M12.2 captured live B300 current-C oracle JSON and Rust facade candidate JSON
  under `ds4-parity/baselines/backend/m12.2/captures/`, adds
  `ds4-parity/baselines/backend/m12.2/manifest.json`, and adds
  `ds4-parity/check_backend_operation_fixtures.py --negative-test`. The bundle
  covers first-kernel embedding, layer-0 QKV/RoPE, layer-0 attention output,
  layer-0 FFN/router/MoE, and full output-head/logits without changing runtime
  routes or claiming backend replacement. Live B300 capture passed the existing
  pair comparators with 103, 426, 493, 885, and 440 checks respectively.
  Validation passed Python syntax, JSON formatting, the M12.2 checker with 576
  checks and negative tests, the runtime graph closure matrix with negative
  tests, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 83 passed, 50
  skipped, 0 failed. M12.3 Rust Backend Facade Parity Harness became active.
- M12.1 adds
  `ds4-parity/baselines/backend/m12.1/backend-boundary-inventory.json` and
  `ds4-parity/check_backend_boundary_inventory.py --negative-test`, wiring the
  checker into the unified parity report and README. The inventory ties each
  M10.2 backend operation family to owner state, platform requirement, model
  requirement, fixture source, comparator path, drift policy, B300 rerun
  commands, and a no-removal/no-replacement claim policy. Validation passed
  Python syntax, JSON formatting, the M12.1 checker with negative tests, the
  runtime graph closure matrix with negative tests, `cargo fmt --all
  -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 82 passed, 50
  skipped, 0 failed. M12.2 Operation Tensor Fixture Capture is active.
- M12 Backend Replacement Parity is split before implementation into M12.1
  through M12.6. Each substage now records Oracle, Fixture, Comparator,
  Acceptance, and Drift policy criteria before any backend ownership changes.
  M12.1 Backend Boundary Inventory And Claim Matrix is active.
- M11.4 adds the Rust `ds4-agent-loop-smoke-rs` no-model smoke emitter,
  records `ds4-parity/baselines/agent/m11.4/loop-smoke.json`, adds
  `ds4-parity/compare_agent_loop_smoke.py --negative-test`, and wires the
  comparator into the unified parity report and README. The artifact parses the
  scripted DSML with the Rust agent parser, inserts the deterministic tool
  result, applies save/list/switch/history/new session commands, normalizes
  session ids, and explicitly defers model-backed manual smoke.
- M11.4 validation passed `cargo run --quiet -p ds4-gguf --bin
  ds4-agent-loop-smoke-rs >
  ds4-parity/baselines/agent/m11.4/loop-smoke.json` and `python3
  ds4-parity/compare_agent_loop_smoke.py --negative-test` with 223 checks.
  Follow-up gates also passed: Python syntax checks; `cargo fmt --all
  -- --check`; `cargo test --workspace`; `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test`; `python3
  ds4-parity/run_server_parity_report.py`; and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 81 passed, 50
  skipped, 0 failed.
- M11.3 adds the Rust `ds4-agent-deterministic-replay-rs` artifact emitter,
  records `ds4-parity/baselines/agent/m11.3/deterministic-replay.json`, adds
  `ds4-parity/compare_agent_deterministic_replay.py --negative-test`, and
  wires the comparator into the unified parity report and README. The artifact
  replays the M11.1 deterministic `list` tool stub and
  save/list/switch/history/new command effects, cross-checks the M11.2 rendered
  tool-result boundary, and keeps `live_execution` and `model_sampling` false.
- M11.3 validation passed `cargo run --quiet -p ds4-gguf --bin
  ds4-agent-deterministic-replay-rs >
  ds4-parity/baselines/agent/m11.3/deterministic-replay.json` and `python3
  ds4-parity/compare_agent_deterministic_replay.py --negative-test` with 230
  checks.
  Follow-up gates also passed: Python syntax checks; `cargo fmt --all
  -- --check`; `cargo test --workspace`; `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test`; `python3
  ds4-parity/run_server_parity_report.py`; and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 80 passed, 50
  skipped, 0 failed.
- M11.2 adds the Rust `ds4-agent-rendered-context-rs` artifact emitter,
  records `ds4-parity/baselines/agent/m11.2/rendered-context.json`, adds
  `ds4-parity/compare_agent_rendered_context.py --negative-test`, and wires
  the comparator into the unified parity report and README. The artifact
  replays M11.1 scripted cases through Rust prompt rendering and records
  normalized prompt text, role boundaries, marker counts, raw DSML preservation,
  tool-result tags, and final visible assistant text without live model
  sampling.
- M11.2 validation passed `cargo run --quiet -p ds4-gguf --bin
  ds4-agent-rendered-context-rs >
  ds4-parity/baselines/agent/m11.2/rendered-context.json` and `python3
  ds4-parity/compare_agent_rendered_context.py --negative-test` with 178
  checks.
- M11.1 adds a no-model current-C `./ds4-agent --dump-agent-trace-oracle`
  replay oracle, records
  `ds4-parity/baselines/agent/m11.1/current-c.json`, adds the Rust
  `ds4-agent-trace-replay-rs` emitter, and wires
  `ds4-parity/compare_agent_trace_replay.py --negative-test` into the unified
  parity report and README. The fixture normalizes workspace/session markers,
  pins a scripted DSML `list` tool round with deterministic tool output, and
  pins save/list/switch/history/new session-command flow before any live Rust
  agent-loop claim.
- M11.1 validation passed `arch -arm64 make ds4-agent`;
  `./ds4-agent --dump-agent-trace-oracle
  ds4-parity/baselines/agent/m11.1/current-c.json`; and `python3
  ds4-parity/compare_agent_trace_replay.py --negative-test` with 225 checks.
  Follow-up gates also passed: Python syntax checks; `cargo test -p ds4-gguf
  agent_trace_replay`; `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test`; `cargo
  fmt --all -- --check`; `cargo test --workspace`; `python3
  ds4-parity/run_server_parity_report.py`; and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 78 passed, 50
  skipped, 0 failed.
- M10.9f adds the Rust `ds4-runtime-graph-bench-rs` benchmark capture binary,
  exposes Rust session snapshot and EOS-excluding argmax helpers needed to
  mirror `ds4-bench`, adds `ds4-parity/run_runtime_graph_bench.py`, wires the
  comparator into the unified report and README, and records
  `ds4-parity/baselines/graph/m10.9f/runtime-benchmark-closure.json` from the
  B300 pod. The artifact records route `graph`, backend `cuda`, q2-imatrix
  model hash
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  prompt hash
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`,
  exact short/long benchmark CSV workload shape, KVC snapshot bytes matching
  M0.6, M10.9a through M10.9e gate status, and a claim boundary that closes
  Milestone 10 without claiming general backend replacement.
- M10.9f validation passed the live B300 Rust benchmark closure with 349
  checks and 8 negative mutations. The artifact documents 7 older M0.6 decode
  throughput threshold misses, reproduces the same drift with same-session
  current-C `ds4-bench`, and verifies Rust stays within the same-session
  current-C threshold; `python3 ds4-parity/run_runtime_graph_bench.py
  --negative-test`; Python syntax checks; `cargo test -p ds4-engine --bin
  ds4-runtime-graph-bench-rs`; `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test`; `cargo
  test --workspace`; `python3 ds4-parity/run_server_parity_report.py`; `cargo
  fmt --all -- --check`; `git diff --check`; and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`. Non-interactive
  Claude review returned `NO BLOCKERS`.
- M10.9e extends `ds4-parity/run_tool_call_quality.py` into a self-contained
  Rust graph tool/server artifact comparator, wires it into the unified report
  and README, transitions `ds4-server-runtime-rs --runtime-graph graph` to run
  on CUDA/Metal while still rejecting CPU, refreshes the route-preflight
  artifact for that server-specific graph behavior, and records
  `ds4-parity/baselines/graph/m10.9e/runtime-tool-server.json` from the B300
  pod. The artifact records route `graph`, backend `cuda`, q2-imatrix model
  hash
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  current-C `./ds4_test --tool-call-quality` stdout/stderr, raw Rust
  request/response/trace/log blobs for fast and exact quality cases, HTTP 200,
  finish `tool_calls`, tool `list_files`, arguments `{"path":"."}`, and
  trace/cache ledger markers.
- M10.9e validation passed the live B300 Rust server runtime tool-call capture
  with 167 checks and 8 negative mutations; `python3
  ds4-parity/run_tool_call_quality.py --negative-test`; `python3
  ds4-parity/check_runtime_graph_route_preflight.py --negative-test`; `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test`; Python
  syntax checks; `cargo test -p ds4-engine --bin ds4-server-runtime-rs`;
  `cargo test --workspace`; `python3 ds4-parity/run_server_parity_report.py`;
  `cargo fmt --all -- --check`; `git diff --check`; and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`. Non-interactive
  Claude review returned `NO BLOCKERS`.
- M10.9d adds the Rust `ds4-runtime-long-context-rs` capture binary, adds
  `ds4-parity/run_runtime_graph_long_context.py`, wires the comparator into the
  unified report and README, and records
  `ds4-parity/baselines/graph/m10.9d/runtime-long-context.json` from the B300
  pod. The artifact records route `graph`, backend `cuda`, q2-imatrix model
  hash
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  long-context prompt hash
  `29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666`,
  current-C long-context stdout/stderr, raw Rust stdout/stderr, 30,474 prompt
  tokens, 76 completion tokens, `stop`, exact fact-recall output, and
  cache/KVC write accounting equal to the prompt token count.
- M10.9d validation passed the live B300 Rust runtime long-context capture with
  126 checks and 8 negative mutations; `python3
  ds4-parity/run_runtime_graph_long_context.py --negative-test`; `cargo check
  -p ds4-engine --bin ds4-runtime-long-context-rs`; `cargo test -p
  ds4-engine --bin ds4-runtime-long-context-rs`; Python syntax checks; `cargo
  test --workspace`; `python3 ds4-parity/run_server_parity_report.py` with 10
  passed, 3 skipped, and 0 failed; `cargo fmt --all -- --check`; `git diff
  --check`; and `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
  with 75 passed, 48 skipped, and 0 failed. Non-interactive Claude review
  returned `NO BLOCKERS`.
- M10.9c adds the Rust `ds4-runtime-official-vectors-rs` capture binary,
  exposes Rust session argmax/top-logprob/eval APIs, adds
  `ds4-parity/run_runtime_graph_official_vectors.py`, wires the comparator into
  the unified report, and records
  `ds4-parity/baselines/graph/m10.9c/runtime-official-vectors.json` from the
  B300 pod. The artifact records route `graph`, backend `cuda`, q2-imatrix
  model hash
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  `official.vec` hash
  `0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac`,
  raw Rust stdout/stderr, selected-token matches, top-logprob rows,
  official-top deltas, and the current-C `long_memory_archive` skip reason.
- M10.9c validation passed the live B300 Rust runtime official-vector capture
  with 1,958 checks, max official-logprob delta 0.678254604, and 8 negative
  mutations; `python3 ds4-parity/run_runtime_graph_official_vectors.py
  --negative-test`; `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test`; `python3
  ds4-parity/check_runtime_graph_route_preflight.py --negative-test`; Python
  syntax checks; `cargo test -p ds4-engine --bin
  ds4-runtime-official-vectors-rs`; `cargo test --workspace`; `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed; `cargo fmt --all -- --check`; `git diff --check`; and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 74 passed, 47
  skipped, and 0 failed. Non-interactive Claude review returned `NO
  BLOCKERS`.
- M10.9b adds shared Rust `RuntimeGraphRoute` selector support, wires
  `--runtime-graph`/`--runtime-graph-route` through one-shot, interactive, and
  server runtime binaries, and records
  `ds4-parity/baselines/graph/m10.9b/runtime-graph-route-preflight.json`.
  The route-preflight artifact covers target-stream, disabled-route,
  invalid-selector, CUDA/non-CUDA unsupported graph-route, missing-model, and
  server KVC preflight cases. Non-server unsupported graph selection exits 99
  before model open, stream output, or checkpoint/cache mutation. Server CUDA
  graph selection now reaches the missing-model path without stream output or
  server KVC directory creation; target-stream and `off` keep the existing
  missing-model behavior.
- M10.9b validation passed Python syntax checks, `python3
  ds4-parity/check_runtime_graph_route_preflight.py --negative-test` with 274
  checks and 8 negative mutations, `python3
  ds4-parity/check_runtime_graph_closure_matrix.py --negative-test` with 118
  checks and 8 negative mutations after status advanced to M10.9c, targeted
  Rust route/parser/server tests, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 73 passed, 46 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.9a adds
  `ds4-parity/baselines/graph/m10.9a/runtime-graph-closure-matrix.json`,
  `ds4-parity/check_runtime_graph_closure_matrix.py`, README instructions,
  unified report wiring, and an exact B300 fixture-readiness rerun hook. The
  matrix pins M10.9b through M10.9f to concrete oracles, fixture paths,
  artifact paths, rerun commands, comparators, acceptance rules, drift
  policies, and claim boundaries.
- M10.9a validation passed Python syntax, `python3
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
- M10.9 split before implementation into M10.9a closure matrix, M10.9b route
  switch, M10.9c official-vector gate, M10.9d long-context gate, M10.9e
  tool/server gate, and M10.9f benchmark/final closure.
- M10.9 split validation passed the B300 fixture-readiness probe for resolved
  model path
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
  model size 86,720,111,488 bytes, `official.vec` SHA
  `0223bbe1eaa3b626be87849df389af91c3f3f6e6b0d4436baf2dbb6ed624b1ac`,
  benchmark prompt SHA
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`,
  existing M0.6 benchmark CSV fixtures, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 71 passed, 45 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g4b adds `ds4-parity/compare_mtp_end_to_end_closure.py`,
  `ds4-parity/baselines/graph/m10.8g4b/end-to-end-closure.json`, README
  instructions, unified report wiring, and an exact B300 rerun hook. The
  closure consumes the M10.8g4a support-branch decision, M10.8g1 stream
  blocker, and M10.8g3c Rust runtime blocker.
- M10.8g4b live B300 closure validation passed with 58 checks and 7 negative
  mutations after refreshing the M10.8g4a branch decision. The artifact records
  `support_absent_blocker_closure`, support-present comparator `not_run` due to
  `support_artifact_absent`, `/workspace/ds4/ds4flash.gguf` at 86,720,111,488
  bytes, absent `/workspace/ds4/missing-mtp.gguf`, empty `mtp_candidates=`,
  `blocked_before_stream` visibility, checkpoint delta 0, no cache/KVC
  visibility, `blocked_missing_mtp_model`, next stage `M10.9`, and no
  MTP-enabled parity claim. Local validation passed Python syntax, `python3
  ds4-parity/compare_mtp_end_to_end_closure.py --negative-test` with 58 checks
  and 7 negative mutations, `python3 ds4-parity/run_server_parity_report.py`
  with 10 passed, 3 skipped, and 0 failed, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 71 passed, 45
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g4 and M10.8g are closed through the explicit support-artifact blocker;
  MTP-enabled current-C versus Rust stream parity is not claimed until a B300
  MTP support GGUF exists.
- M10.8g4a adds `ds4-parity/compare_mtp_support_branch.py`,
  `ds4-parity/baselines/graph/m10.8g4a/support-branch-decision.json`, README
  instructions, unified report wiring, and an exact B300 rerun hook. The branch
  decision links the M10.8g1 stream blocker and M10.8g3c Rust runtime blocker
  to the current B300 support-artifact search.
- M10.8g4a live B300 branch capture passed with 48 checks and 6 negative
  mutations, recording `/workspace/ds4/ds4flash.gguf` at 86,720,111,488 bytes,
  absent `/workspace/ds4/missing-mtp.gguf`, empty `mtp_candidates=`, selected
  branch `support_absent_blocker_closure`, next stage `M10.8g4b`, and a claim
  policy forbidding `MTP-off pass` and `MTP-enabled parity`. Local validation
  passed Python syntax, `python3 ds4-parity/compare_mtp_support_branch.py
  --negative-test` with 48 checks and 6 negative mutations, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 70 passed, 44
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g4 split before implementation into M10.8g4a B300 support-artifact
  branch decision and M10.8g4b final support comparator or explicit blocker
  closure, so the currently missing support-model path stays separate from
  MTP-enabled parity claims.
- M10.8g4 split validation passed the live B300 support-artifact probe with
  `/workspace/ds4/ds4flash.gguf`, absent `/workspace/ds4/missing-mtp.gguf`, and
  empty `mtp_candidates=`, plus `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 69 passed, 43
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g3c adds `ds4-parity/compare_mtp_runtime_missing_support.py`,
  `ds4-parity/baselines/graph/m10.8g3c/rust-b300-missing-support-runtime.json`,
  README instructions, unified report wiring, and an exact B300 rerun hook. The
  comparator ties the Rust B300 missing-MTP runtime result to the M10.8g3a
  missing-support guard row, the M10.8g1 stream blocker, and the M8.12b
  current-C missing-MTP runtime case.
- M10.8g3c live B300 smoke passed with 118 checks and 7 negative mutations,
  recording `/workspace/ds4/ds4flash.gguf` at 86,720,111,488 bytes, absent
  `/workspace/ds4/missing-mtp.gguf`, empty `mtp_candidates=`, exit code 1,
  empty stdout, stderr SHA
  `826268e476a14b68cf733c113b9a8517c9c3209988de7dbb3bbd98e7f64f444a`,
  `blocked_before_stream` visibility, checkpoint delta 0, and no cache/KVC
  visibility. Local validation passed Python syntax, `python3
  ds4-parity/compare_mtp_runtime_missing_support.py --negative-test` with 118
  checks and 7 negative mutations, targeted Rust guard test
  `missing_support_blocks_before_stream_mutation`, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 69 passed, 43
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g3b adds `ds4-parity/compare_mtp_runtime_no_drift.py` and wires it into
  `ds4-parity/run_parity_report.py` and `ds4-parity/README.md`. The comparator
  ties the M10.8g3a disabled runtime guard rows to the M8.12a current-C
  one-shot target-stream oracle and the M9.8f5 B300 Rust runtime replay
  summary, checking 3 CLI no-MTP target-stream cases, 3 server no-MTP replay
  cases, cache/KVC ledger probes, guard linkage, and static runtime report
  hooks.
- M10.8g3b validation passed Python syntax, `python3
  ds4-parity/compare_mtp_runtime_no_drift.py --negative-test` with 3 CLI
  cases, 3 server cases, 180 checks, and 6 negative mutations, live B300
  one-shot no-MTP runtime comparator with 144 checks and 5 negative checks,
  `python3 ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped,
  and 0 failed, `cargo test --workspace`, `cargo fmt --all -- --check`, `git
  diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 68 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g3a adds `rust/ds4-gpu/src/mtp_runtime_guard_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-runtime-guard-plan.rs`, and
  `ds4-parity/compare_mtp_runtime_guard.py`, a Rust model-free runtime guard
  plan that ties `EngineOptions`, `ds4-gguf` MTP CLI parsing,
  one-shot/interactive/server runtime mappings, argmax/session non-MTP
  surfaces, current-C speculative dispatch guards, and the B300 missing-support
  artifact check to the M10.8g2 disabled, first-draft-miss, and
  missing-support stream outcomes.
- M10.8g3a validation passed targeted Rust guard tests, JSON output parsing,
  Python syntax, `python3 ds4-parity/compare_mtp_runtime_guard.py
  --negative-test` with 7 cases, 292 checks, and 7 negative mutations, the live
  B300 missing-support artifact check with empty `mtp_candidates=`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 67 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g3 is split before implementation into M10.8g3a Rust runtime MTP guard
  contract and static wiring, M10.8g3b runtime target-stream no-drift
  comparator, and M10.8g3c B300 missing-support runtime smoke. The split keeps
  disabled-MTP, missing-support, first-draft-miss, server/runtime request
  replay, cache/KVC ledger, and target-stream no-drift coverage separate before
  the support-model end-to-end comparator.
- M10.8g3 split validation passed the live B300 missing-support artifact check
  with empty `mtp_candidates=`, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 66 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g2 adds `rust/ds4-gpu/src/mtp_stream_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-stream-plan.rs`, and
  `ds4-parity/compare_mtp_stream_plan.py`, a Rust model-free stream outcome
  planner that composes M10.8b through M10.8f subplans against the M10.8g1
  stream contract. It pins final accepted stream deltas, checkpoint deltas,
  logits ownership, selected subplan IDs, frontier operations, `mtp_n_raw`
  keep policy, cache/KVC visibility, fallback/error state, and missing-MTP
  blocker semantics for 12 stream rows.
- M10.8g2 validation passed targeted Rust stream-plan tests, JSON output
  parsing, Python syntax, `python3 ds4-parity/compare_mtp_stream_plan.py
  --negative-test` with 12 cases, 369 checks, and 8 negative mutations, the
  live B300 missing-MTP blocker command, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 66 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g1 adds
  `ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json` and
  `ds4-parity/check_mtp_stream_parity_contract.py`, a stream-level current-C
  contract checker that links 12 end-to-end speculative outcomes to M10.8a
  decision rows and `ds4_session_eval_speculative_argmax` anchors: disabled or
  missing MTP, first-draft miss, exact N=2 full/prefix/failure, suffix
  full/prefix/replay/failure, sequential fallback, frontier restore/commit,
  `mtp_n_raw` keep policy, visible cache/KVC state, and the B300 missing-MTP
  blocker.
- M10.8g1 validation passed JSON syntax, Python syntax, `python3
  ds4-parity/check_mtp_stream_parity_contract.py --negative-test` with 12
  cases, 368 checks, and 8 negative mutations, the live B300 missing-MTP
  blocker command, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 65 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8g is split before implementation into M10.8g1 stream parity contract
  and blocker, M10.8g2 Rust MTP stream outcome planner, M10.8g3 Rust runtime
  guard and target-stream no-drift smoke, and M10.8g4 B300 support-model
  end-to-end comparator. The split follows the current-C
  `ds4_session_eval_speculative_argmax` stream outcomes: disabled/missing MTP,
  first-draft miss, exact N=2 full/prefix/failure, suffix full/prefix/replay,
  verifier failure, sequential fallback, frontier restore/commit, and
  `mtp_n_raw` keep policy. Validation passed the live B300 support-artifact
  blocker command, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 64 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8f adds `rust/ds4-gpu/src/mtp_frontier_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-frontier-plan.rs`, and
  `ds4-parity/compare_mtp_frontier_plan.py`, a Rust model-free frontier
  mutation plan that pins snapshot, restore, prefix1 commit, ratio-4 index
  handling, `mtp_n_raw` save/restore, invisible speculative-row policy, and
  the explicit B300 missing-MTP live blocker against current-C anchors and
  M10.7d3 restored-frontier evidence.
- M10.8f validation passed `cargo test -p ds4-gpu mtp_frontier_plan`,
  `python3 ds4-parity/compare_mtp_frontier_plan.py --negative-test` with 8
  cases, 145 checks, and 8 negative mutations, JSON output parsing via `cargo
  run -p ds4-gpu --bin ds4-mtp-frontier-plan --quiet | python3 -m json.tool`,
  Python syntax, the live B300 missing-MTP blocker command, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 64 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8e adds `rust/ds4-gpu/src/mtp_suffix_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-suffix-plan.rs`, and
  `ds4-parity/compare_mtp_suffix_plan.py`, a Rust model-free MTP suffix
  verifier orchestration plan that pins full accept, prefix1 accept,
  restore/replay, exact replay debug, failure restore-or-error behavior,
  row-top/readback roles, and the explicit B300 missing-MTP live blocker.
- M10.8e validation passed `cargo test -p ds4-gpu mtp_suffix_plan`, `python3
  ds4-parity/compare_mtp_suffix_plan.py --negative-test` with 6 cases, 179
  checks, and 8 negative mutations, JSON output parsing via `cargo run -p
  ds4-gpu --bin ds4-mtp-suffix-plan --quiet | python3 -m json.tool`, Python
  syntax, the live B300 missing-MTP blocker command, `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 63 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8d adds `rust/ds4-gpu/src/mtp_decode2_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-decode2-plan.rs`, and
  `ds4-parity/compare_mtp_decode2_plan.py`, a Rust model-free exact-N=2 MTP
  verifier orchestration plan that pins target token order, decode-layer command
  steps, prefix1 frontier capture, top0/logits0/logits1 readback roles,
  full-accept and prefix1 logits source, failure restore behavior, and the
  explicit B300 missing-MTP live blocker.
- M10.8d validation passed `cargo test -p ds4-gpu mtp_decode2_plan`,
  `python3 ds4-parity/compare_mtp_decode2_plan.py --negative-test` with 4
  cases, 148 checks, and 7 negative mutations, JSON output parsing via `cargo
  run -p ds4-gpu --bin ds4-mtp-decode2-plan --quiet | python3 -m json.tool`,
  Python syntax, the live B300 missing-MTP blocker command, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 62 passed, 42
  skipped, and 0 failed. Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8c adds `rust/ds4-gpu/src/mtp_draft_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-draft-plan.rs`, and
  `ds4-parity/compare_mtp_draft_plan.py`, a Rust model-free MTP draft
  orchestration plan that pins first-draft and recursive draft-HC roles,
  command steps, top-id/logits readback roles, `mtp_n_raw` transition, failure
  restoration, and the explicit B300 missing-MTP live blocker.
- M10.8c validation passed `cargo test -p ds4-gpu mtp_draft_plan`,
  `python3 ds4-parity/compare_mtp_draft_plan.py --negative-test` with 5
  cases, 118 checks, and 6 negative mutations, JSON output parsing via
  `cargo run -p ds4-gpu --bin ds4-mtp-draft-plan --quiet | python3 -m
  json.tool`, Python syntax, the live B300 missing-MTP blocker command,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 61
  passed, 42 skipped, and 0 failed. Non-interactive Claude review returned
  `NO BLOCKERS`.
- M10.8b adds `rust/ds4-gpu/src/mtp_plan.rs`,
  `rust/ds4-gpu/src/bin/ds4-mtp-decision-plan.rs`, and
  `ds4-parity/compare_mtp_decision_plan.py`, a Rust model-free MTP decision
  planner that emits the same 12 accepted-prefix/frontier/logits/fallback rows
  as the M10.8a current-C contract before any GPU MTP kernels are ported.
- M10.8b validation passed `cargo test -p ds4-gpu mtp_plan`,
  `python3 ds4-parity/compare_mtp_decision_plan.py --negative-test` with 12
  cases, 194 checks, and 6 negative mutations, JSON output parsing via
  `cargo run -p ds4-gpu --bin ds4-mtp-decision-plan --quiet | python3 -m
  json.tool`, Python syntax, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 60 passed, 42 skipped, and 0 failed.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.8a adds
  `ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json` and
  `ds4-parity/check_mtp_state_machine_contract.py`, a model-free current-C MTP
  state-machine contract. It pins 12 decision rows across missing support,
  MTP-disabled guard, first-draft miss, margin skip, exact N=2 verification,
  suffix/microbatch verification, rollback/error handling, and sequential
  fallback.
- M10.8a ties B300 MTP support-artifact availability to the M8.12b CLI runtime
  baseline and a live `hou2-prod1` probe: `/workspace/ds4/missing-mtp.gguf` is
  absent and no `*mtp*.gguf` or `*draft*.gguf` candidates exist under
  `/workspace/ds4` at depth 3, so later MTP-enabled live stages remain
  explicitly blocked until a support GGUF is available.
- M10.8a validation passed `python3
  ds4-parity/check_mtp_state_machine_contract.py --negative-test`, JSON
  syntax, Python syntax for the checker and unified report, `cargo fmt --all
  -- --check`, `git diff --check`, the live B300 availability command, and
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 59
  passed, 42 skipped, and 0 failed. Non-interactive Claude review returned
  `NO BLOCKERS` after a hung first review was killed and a narrower blocker
  review completed.
- M10.8 is split before implementation into M10.8a state-machine contract and
  B300 MTP availability check, M10.8b Rust model-free decision planner,
  M10.8c Rust MTP draft kernel orchestration smoke, M10.8d exact-N=2 verifier
  orchestration smoke, M10.8e suffix verifier orchestration smoke, M10.8f Rust
  speculative frontier snapshot/restore/prefix1 commit, and M10.8g end-to-end
  stream parity. The split follows current-C boundaries in
  `ds4_session_decode_speculative`, `metal_graph_eval_mtp_draft`,
  `metal_graph_verify_decode2_exact`, `metal_graph_verify_suffix_tops`,
  `spec_frontier_snapshot`, `spec_frontier_restore`, and
  `spec_frontier_commit_prefix1`; non-interactive Claude review returned
  `NO BLOCKERS` after a first pass caught and the final split fixed missing
  dedicated MTP draft and suffix-verifier stages.
- M10.7d3c2 adds `ds4-post-restore-kvc-smoke`,
  `ds4-parity/compare_post_restore_kvc_smoke.py`, and
  `ds4-parity/baselines/kv/m10.7d3/rust-b300-post-restore-kvc.json`, a live
  B300 smoke plus offline comparator for wrapping restored graph payload bodies
  in deterministic shutdown KVC files.
- M10.7d3c2 live B300 validation on `hou2-prod1` ran
  `python3 ds4-parity/compare_post_restore_kvc_smoke.py --live --workdir
  /workspace/ds4 --output-dir /tmp/ds4-m107d3c2-kvc --write-summary
  /tmp/ds4-m107d3c2-post-restore-kvc.json --negative-test`; it passed 536
  exact checks and seven negative mutations across `disk_seed_payload`,
  `snapshot_seed`, `disk_continuation_payload`, and `snapshot_continuation`.
- M10.7d3c2 pins KVC file names, deterministic shutdown headers, file sizes,
  rendered text key bytes, payload FNV/SHA256 digests, restored token/frontier
  decisions, and graph counters against the M10.7d3c1 contract and M10.7d3b
  same-capture restored graph evidence.
- M10.7d3c2 non-interactive Claude review returned `NO BLOCKERS` after
  checking B300 command fidelity, KVC header/payload invariants, comparator
  coverage, and evidence consistency.
- M10.7d3c1 adds
  `ds4-parity/baselines/kv/m10.7d3/post-restore-kvc-decision-contract.json`
  and `ds4-parity/check_post_restore_kvc_decision_contract.py`, a model-free
  contract that ties the four restored graph payload cases to KVC
  write/skip expectations before the B300 file-writing smoke.
- M10.7d3c1 validates unaligned post-restore continued-store skips,
  re-enabled next continued targets, already-stored boundary skips, and
  shutdown-write header expectations against M10.7d3b restored-frontier
  projection, the M10.7d3a frontier contract, M9.8f5 runtime replay evidence,
  the M10.7d2 runtime ledger contract, and the M7.4a KVC file layout oracle.
- M10.7d3c1 validation passed `python3
  ds4-parity/check_post_restore_kvc_decision_contract.py --negative-test` with
  4 post-restore cases, 3 runtime references, and 8 negative mutations, plus
  graph restore projection, frontier contract, runtime KV replay, KVC file,
  Python syntax, JSON syntax, `cargo fmt --all -- --check`, `git diff
  --check`, and `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 57 passed, 40 skipped, and 0 failed after
  fail-loud negative-test target lookup hardening. Non-interactive Claude
  review reported no blockers.
- M10.7d3c is split before implementation into M10.7d3c1 model-free
  post-restore KVC decision contract and M10.7d3c2 B300 restored payload KVC
  file smoke, so the write/skip decision matrix is proven before any new B300
  KVC file-writing evidence is trusted.
- M10.7d3b extends `ds4-graph-restore-next-token` so each restored B300 graph
  payload emits `frontier_projection` evidence derived from the restored token
  count. The summary now records loaded frontier, unaligned current-live skip,
  next continued-store target, already-stored boundary skip, and shutdown
  reason projection for `disk_seed_payload`, `snapshot_seed`,
  `disk_continuation_payload`, and `snapshot_continuation`.
- M10.7d3b refreshes
  `ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json` from a
  live `hou2-prod1` B300 run. The four cases loaded restored frontiers 550,
  550, 561, and 561; current-live target stayed 0; next continued target was
  10240; and already-stored boundary target stayed 0.
- M10.7d3b validation passed the exact-tree live B300 graph restore projection
  comparator with 4177 checks and 12 negative mutations, local `python3
  ds4-parity/compare_graph_restore_next_token.py --negative-test`,
  `python3 ds4-parity/check_graph_restore_frontier_contract.py
  --negative-test`, `python3 ds4-parity/compare_kv_policy.py
  --negative-test`, targeted Rust frontier-projection tests, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 56 passed, 40
  skipped, and 0 failed, plus non-interactive Claude review with `NO
  BLOCKERS`.
- M10.7d3a adds
  `ds4-parity/baselines/kv/m10.7d3/restore-frontier-contract.json`, a
  model-free graph-restore continued-frontier contract that maps the four
  M10.7c3d restored graph payload cases onto restored token counts, loaded
  frontier values, re-enabled continued-store targets, already-stored skip
  behavior, and KVC reason-code references.
- M10.7d3a adds `ds4-parity/check_graph_restore_frontier_contract.py`, which
  validates the contract against the M10.7c3d restored-token summary, the M7.2
  current-C continued-frontier policy matrix, and the M0.5 KVC header rows.
  The checker has seven mutation-based negative tests and is wired into
  `ds4-parity/run_parity_report.py`.
- M10.7d3a validation passed `python3
  ds4-parity/check_graph_restore_frontier_contract.py --negative-test`,
  `python3 ds4-parity/compare_kv_policy.py --negative-test`, `python3
  ds4-parity/compare_graph_restore_next_token.py --negative-test`, Python
  syntax and JSON checks, `cargo fmt --all -- --check`, `git diff --check`,
  and `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 56
  passed, 40 skipped, and 0 failed, plus `cargo test --workspace` and
  non-interactive Claude review with `NO BLOCKERS`.
- M10.7d2c exposes the Rust runtime cache ledger in per-request traces under
  `--- runtime cache ledger ---`, including cache source, reason, token counts,
  cached/write/disk token counts, continued-frontier before/after values, and
  success fields. The runtime snapshots those events before writing each trace
  and keeps cache policy decisions unchanged.
- M10.7d2c refreshes the M9.8f5 B300 runtime replay summary with checked
  `ledger_cases` for seed miss, seed restore, and continuation restore. The
  checker now validates those summary ledger cases, raw trace event
  counts/names, the M10.7d2 contract, and six negative mutations.
- M10.7d2c B300 replay ran on `ds4-rust-port-b300` in `hou2-prod1` after
  syncing `HEAD` plus the runtime trace patch into `/workspace/ds4`. The first
  replay showed a 13-second CUDA startup before listening, so the rerun used a
  20-second startup wait and passed. The M0.5 replay reproduced seed miss,
  seed restore, and continuation restore responses plus the three expected KV
  files: `0ab2314538b11686a11e296b7f697651fbd17e60.kv`,
  `4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.kv`, and
  `a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv`.
- M10.7d2c validation passed B300 runtime replay, `python3
  ds4-parity/check_runtime_kv_replay_summary.py --negative-test`, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `python3 ds4-parity/compare_kv_replay.py --negative-test`,
  JSON/Python syntax checks, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, and `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 55 passed, 40
  skipped, and 0 failed, plus non-interactive Claude review with `NO
  BLOCKERS`.
- M10.7d2b adds
  `ds4-parity/baselines/kv/m10.7d2/runtime-ledger-contract.json`, a
  model-free ledger contract for M0.5 seed miss, seed restore, continuation
  restore, and M0.4 memory-token continuation. The contract pins cache source,
  cached/write/disk token counts, KVC cold-write reason metadata, ledger event
  order, and continued-frontier before/after transitions.
- M10.7d2b extends `ds4-parity/check_runtime_kv_replay_summary.py` to validate
  the M9.8f5 B300 replay summary together with the M10.7d2 ledger contract and
  adds five mutation-based negative checks. `run_server_parity_report.py` now
  runs that checker with `--negative-test`.
- M10.7d2b validation passed `python3
  ds4-parity/check_runtime_kv_replay_summary.py --negative-test`, `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed, `python3 ds4-parity/compare_kv_replay.py --negative-test`,
  `python3 -m py_compile ds4-parity/check_runtime_kv_replay_summary.py
  ds4-parity/run_server_parity_report.py`, `cargo test --workspace`, `cargo
  fmt --all -- --check`, `git diff --check`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 55 passed, 40
  skipped, and 0 failed, and non-interactive Claude review with `NO BLOCKERS`.
- M10.7d2a adds a private Rust runtime cache ledger that is cleared per chat
  request and records cache decisions, reset-after-miss, cold-store
  suppression, frontier note/restore, live-prefix store attempts, shutdown or
  eviction stores, and decode-time continued-store attempts without changing
  cache policy decisions.
- M10.7d2a exposes `ServerSession::position()` for runtime cache accounting and
  adds model-free tests for reset ledger ordering, suppress/restore/cache
  decision events, and failed suppression without frontier mutation.
- M10.7d2a validation passed focused runtime ledger tests, `python3
  ds4-parity/compare_kv_policy.py --negative-test` with 1725 comparator checks
  and 9 negative checks, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, `python3
  ds4-parity/compare_graph_restore_next_token.py --negative-test` after
  restoring the exact M10.7c3d status marker, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 55 passed, 40
  skipped, and 0 failed, and non-interactive Claude review with `NO BLOCKERS`.
- M10.7d1 adds a C/Rust `continued_frontier_transitions` policy matrix covering
  note-store growth, lower-note skip, fresh-frontier suppression/restore,
  already-stored suppression skip, unaligned suppression skip,
  mismatch-restore ignore, reset-after-miss, and disk-restore loaded frontier
  state. The M7.2 policy oracle and M7.7 replay precondition fixtures were
  refreshed to track the new C policy artifact.
- M10.7d1 adds Rust `reset_continued_frontier`, extends continued-store policy
  tests for reset and disk-loaded frontiers, and adds a
  `RuntimeCacheState` reset-after-miss test so the runtime cache wrapper
  exposes the same reset state as current C before M10.7d2 runtime replay work.
- M10.7d1 validation passed `python3
  ds4-parity/check_kv_policy_dump.py --negative-test` with 521 schema checks,
  11 manifest checks, and 8 negative checks; `python3
  ds4-parity/compare_kv_policy.py --negative-test` with 1725 comparator checks
  and 9 negative checks; `python3 ds4-parity/compare_kv_replay.py
  --negative-test`; `python3 ds4-parity/run_kv_parity_report.py` with 9
  passed, 1 skipped, and 0 failed; `python3
  ds4-parity/run_server_parity_report.py` with 10 passed, 3 skipped, and 0
  failed; targeted Rust tests for continued-store policy and runtime reset;
  Python syntax checks; `cargo test --workspace`; `cargo fmt --all --
  --check`; `git diff --check`; `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 55 passed, 40 skipped, and 0 failed; and
  non-interactive Claude review with `NO BLOCKERS`.
- M10.7d2 is split before implementation into M10.7d2a model-free runtime
  continued-frontier ledger contract, M10.7d2b runtime KV replay checker
  closure, and M10.7d2c B300 runtime replay refresh. All three M10.7d2
  subitems are done; M10.7d3 is active and owns the graph-restore
  continued-frontier B300 smoke.
- M10.7d3 is split before implementation into M10.7d3a model-free graph
  restore frontier contract, M10.7d3b B300 restored-graph frontier projection,
  and M10.7d3c post-restore KVC write/skip B300 smoke. M10.7d3a and M10.7d3b
  are done; M10.7d3c is active.
- M10.7c3d adds `ds4-graph-restore-next-token`, a B300 Rust GPU smoke that
  restores the four C-written disk payload and memory snapshot bodies into
  Rust graph state, computes selected token and top-logprob slices from the
  restored session logits, and reports restored checkpoint/logits FNVs, cache
  source, same-capture readback evidence, and post-restore graph counters.
- M10.7c3d adds `ds4-parity/compare_graph_restore_next_token.py` and
  `ds4-parity/baselines/kv/m10.7c3d/rust-b300-restore-next-token.json`. The
  live comparator recaptures the current-C restore oracle first because raw
  payload bodies are per-capture evidence, then compares Rust against that same
  capture while also validating the committed M10.7c3c readback summary.
- M10.7c3d validation passed B300 live restore next-token comparison with 4030
  checks and 11 negative mutations, local `python3
  ds4-parity/compare_graph_restore_next_token.py --negative-test` with 4030
  checks and 11 negative mutations, Python syntax checks, `cargo check -p
  ds4-gpu --bin ds4-graph-restore-next-token`, `cargo test -p ds4-gpu --bin
  ds4-graph-restore-next-token`, `cargo fmt --all -- --check`, `git diff
  --check`, `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
  with 55 passed, 40 skipped, and 0 failed, `cargo test --workspace`, and
  non-interactive Claude review with `NO BLOCKERS`.
- M10.7d is split before implementation into M10.7d1 continued-frontier policy
  transition matrix, M10.7d2 runtime continued-store replay decisions, and
  M10.7d3 graph restore continued-frontier B300 smoke. M10.7d1 closed the
  policy/reset/suppression state machine before runtime or graph restore
  changes.
- M10.7c3c adds `ds4-graph-restore-readback`, a B300 Rust GPU smoke that reads
  the four M7.8 disk payload and memory snapshot raw bodies, writes graph
  sections into Rust-owned `ds4_gpu::Tensor` allocations, and reads the written
  spans back before any decode execution or next-token claim.
- M10.7c3c adds `ds4-parity/compare_graph_restore_readback.py` and
  `ds4-parity/baselines/kv/m10.7c3c/rust-b300-restore-readback.json`. The
  summary records hash-only B300 evidence for checkpoint tokens, logits, count
  tables, raw rows, attention compressed rows, attention state tensors, ratio-4
  indexer rows and state tensors, sampled layer sections, and post-restore
  counters while leaving raw bodies on B300.
- M10.7c3c validation passed B300 live restore readback with 1365 checks and 8
  negative mutations, local `python3
  ds4-parity/compare_graph_restore_readback.py --negative-test` with 1365
  checks and 8 negative mutations, Python syntax checks, `cargo test -p
  ds4-gpu --bin ds4-graph-restore-readback`, `cargo check -p ds4-gpu --bin
  ds4-graph-restore-readback`, `cargo fmt --all -- --check`, `git diff
  --check`, the unified parity report with 54 passed, 39 skipped, and 0 failed,
  `cargo test --workspace`, and non-interactive Claude review with
  `NO BLOCKERS`.
- M10.7c3b adds `ds4-session-payload-dump-rs --restore-target-plan`, a
  no-tensor-write Rust graph restore target plan for the four M7.8 disk payload
  and memory snapshot cases.
- M10.7c3b adds `ds4-parity/compare_graph_restore_target_plan.py`, which
  independently reconstructs the expected C graph restore destinations and
  compares checkpoint/logit/count-table targets, raw logical-to-physical ring
  rows, per-layer attention compressed-cache and state targets, ratio-4 indexer
  targets, and post-restore counters (`layer_n_comp`, `layer_n_index_comp`,
  `checkpoint_valid`, `mtp_draft_valid`, and `mtp_n_raw`).
- M10.7c3b validation passed `python3
  ds4-parity/compare_graph_restore_target_plan.py --negative-test` with 6012
  checks and 8 negative mutations, an explicit candidate-file comparison with
  6012 checks, Python syntax checks, `cargo test -p ds4-gguf session_payload`,
  `cargo fmt --all -- --check`, `git diff --check`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 53 passed, 38
  skipped, and 0 failed, `cargo test --workspace`, and non-interactive Claude
  review with `NO BLOCKERS`.
- M10.7c3a adds `ds4-restore-dump --snapshot-dir`, which writes the
  `ds4_session_save_snapshot` memory payload bytes to B300 raw files while
  leaving the existing restore JSON format compatible with the M7.8 checker.
- M10.7c3a adds
  `ds4-parity/baselines/kv/m10.7c3a/rust-b300-snapshot-raw-import.json` and
  `ds4-parity/compare_graph_snapshot_raw_import.py`. The summary records only
  metadata for `snapshot_seed` and `snapshot_continuation`: observed snapshot
  SHA256, historical oracle SHA256, byte counts, FNVs, Rust reader acceptance,
  parsed graph layout, and B300 source context. Raw snapshot bodies remain on
  B300 and are not committed.
- M10.7c3a discovered that B300 restore bodies are byte-unstable across
  recaptures: rerunning current HEAD and the M7.8 capture source both produced
  different raw disk/snapshot SHA256 values while C source-vs-restored
  self-checks passed. `.memory/lessons.md` now records the permanent policy:
  raw restore-body SHA256/FNV values are per-capture evidence, while exact gates
  stay on byte counts, DSV4 headers, section layout, count tables, Rust reader
  acceptance, and behavior comparators.
- M10.7c3a validation passed B300 raw disk import under the corrected
  per-capture hash policy with 108 checks and 9 negative mutations, B300 raw
  snapshot materialization/import with 110 checks and 9 negative mutations,
  local raw payload and snapshot import comparators with 104 checks and 9
  negative mutations each, Python syntax checks, `cargo test -p ds4-gguf
  session_payload`, `git diff --check`, `arch -arm64 make ds4-restore-dump`,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 52
  passed, 38 skipped, and 0 failed, `cargo fmt --all -- --check`, `cargo test
  --workspace`, and non-interactive Claude review with `NO BLOCKERS`.
- M10.7c2 adds `ds4-session-payload-dump-rs --graph-file-probe <id:path>`,
  which reads C-written graph payload bytes from disk and runs the Rust
  `read_graph_payload` parser over the actual file contents. The probe reports
  file byte length, FNV, C-compatible acceptance/rejection code, parsed raw-ring
  positions, section byte totals, and compressed/index row counts without
  restoring tensors into graph memory.
- M10.7c2 adds the hash-only B300 summary
  `ds4-parity/baselines/kv/m10.7c2/rust-b300-raw-import.json` for the M7.8
  `disk_seed_payload` and `disk_continuation_payload` raw bodies. The summary
  records only metadata: observed payload SHA256, historical oracle SHA256, byte
  counts, FNVs, Rust reader acceptance, parsed graph layout, and B300 source
  context; the raw payload bodies remain on the B300 workspace and are not
  committed. M10.7c3a discovered B300 restore bodies are byte-unstable across
  recaptures, so raw-body SHA256 is capture evidence rather than an exact drift
  gate.
- M10.7c2 adds `ds4-parity/compare_graph_payload_raw_import.py`, which compares
  the B300 Rust summary to `ds4-parity/baselines/kv/m7.8/current-c.json` over
  disk case order, observed and historical payload SHA256 metadata, byte counts,
  exact Rust reader acceptance, decoded DSV4 header-derived raw ring positions,
  ratio-4 and ratio-128 row counts, all section byte totals, hash-only policy,
  and the exact B300 rerun command. The live B300 run passed 104 checks and
  negative tests originally rejected 7 mutations; M10.7c3a extends the
  comparator to reject raw-hash policy drift without treating unstable raw-body
  bytes as an oracle failure.
- M10.7c2 validation passed B300 live raw import with summary writeback,
  `python3 ds4-parity/compare_graph_payload_raw_import.py --negative-test`,
  `python3 -m py_compile ds4-parity/compare_graph_payload_raw_import.py
  ds4-parity/run_parity_report.py`, `cargo test -p ds4-gguf session_payload`,
  `git diff --check`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 51 passed, 37 skipped, and 0 failed, `cargo test
  --workspace`, `cargo fmt --all -- --check`, and non-interactive Claude review
  with `NO BLOCKERS`.
- M10.7c1 adds `ds4-session-payload-dump-rs --restore-header-plan`, a
  hash-only Rust restore payload header plan over the committed M7.8 current-C
  restore oracle. It emits the four seed/continuation disk-payload and
  memory-snapshot cases, model identity, raw-body policy, DSV4 header bytes,
  payload byte counts, graph caps, raw-live rows, and ratio row counts without
  reading raw restore bodies or claiming tensor restore.
- M10.7c1 adds `ds4-parity/compare_restore_payload_header_plan.py`, which
  normalizes `ds4-parity/baselines/kv/m7.8/current-c.json` into the same
  restore-header shape and compares schema/source/oracle path, model path and
  SHA, case order and count, kind, prompt case, prompt token count, exact
  header bytes, payload/snapshot byte budgets, graph fields, and hash-only
  policy. The comparator passed 127 checks and negative tests rejected 7
  mutations.
- M10.7c1 validation passed Rust restore-header JSON emission and JSON parse,
  `python3 ds4-parity/compare_restore_payload_header_plan.py`, `python3
  ds4-parity/compare_restore_payload_header_plan.py --negative-test`, targeted
  Rust test `restore_header_contract_matches_m78_payload_sizes`,
  `python3 -m py_compile ds4-parity/compare_restore_payload_header_plan.py
  ds4-parity/run_parity_report.py`, `git diff --check`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 50 passed, 36
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all --
  --check`, and non-interactive Claude review with `NO BLOCKERS`.
- M10.7b adds a graph-specific Rust payload runtime, `read_graph_payload`,
  `append_graph_payload_plan`, parsed section summaries, and
  `ds4-session-payload-dump-rs --graph-probe`. The reader/writer slice validates
  graph headers, counts, section lengths, raw-ring mapping, trailing bytes, and
  C-compatible rejection categories without restoring tensor contents.
- M10.7b adds `ds4-session-payload-dump --graph-probe` and
  `ds4_dump_graph_session_payload_probe_json`, a no-model C graph payload probe
  that mirrors the C graph load checks for the same synthetic bytes before GPU
  tensor restore. The fixtures cover valid short, valid raw-ring wrap,
  truncated, trailing, invalid compressed count, invalid index count,
  raw-ring mismatch, context-fit, layout, chunk-layout, and comp-cap cases.
- M10.7b adds `ds4-parity/compare_graph_session_payload_rw.py`, which compares
  C and Rust graph payload read/write probe reports across runtime constants,
  byte FNVs, payload byte counts, parsed raw-ring summaries, section byte sums,
  and rejection codes. The comparator passed 375 checks and negative tests
  rejected 7 mutations.
- M10.7b validation passed `arch -arm64 make ds4-session-payload-dump`, C and
  Rust `--graph-probe` JSON parse checks, `python3
  ds4-parity/compare_graph_session_payload_rw.py`, `python3
  ds4-parity/compare_graph_session_payload_rw.py --negative-test`, `cargo test
  -p ds4-gguf session_payload`, `python3 -m py_compile
  ds4-parity/compare_graph_session_payload_rw.py ds4-parity/run_parity_report.py`,
  `git diff --check`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 49 passed, 36 skipped, and 0 failed, `cargo test
  --workspace`, `cargo fmt --all -- --check`, non-interactive Claude review
  with `NO BLOCKERS`, and post-review focused C build/comparator/Rust test
  reruns after addressing the non-blocking style nit.
- M10.7 splits broad graph session state/payload parity into M10.7a layout
  planning, M10.7b payload reader/writer, M10.7c disk KV restore smoke, and
  M10.7d continued-frontier save/restore policy so each slice has a concrete
  oracle, fixture, comparator, and acceptance boundary.
- M10.7a adds `ds4-session-payload-dump --graph-plan`, the C
  `ds4_dump_graph_session_payload_plan_json` oracle, Rust
  `graph_payload_plan`, and `ds4-session-payload-dump-rs --graph-plan`. The
  no-model fixture covers `ctx=32768` graph payload plans for
  `short_checkpoint_tokens3`, `continued_frontier_tokens924`,
  `prefill_cap_cross_tokens2052`, `raw_ring_wrap_tokens2305`, and
  `near_context_tokens32767`.
- M10.7a adds `ds4-parity/compare_graph_session_payload_plan.py`, which
  compares the C and Rust graph payload layout plans across schema/scope,
  constants, body order, prefill/raw/comp caps, raw logical and physical row
  mapping, ratio-4 and ratio-128 compressed/indexed row counts, section byte
  totals, sampled layer bytes, and final payload bytes. The comparator passed
  901 checks and negative tests rejected 7 layout mutations.
- M10.7a validation passed `arch -arm64 make ds4-session-payload-dump`, C and
  Rust `--graph-plan` JSON parse checks, `python3
  ds4-parity/compare_graph_session_payload_plan.py`, `python3
  ds4-parity/compare_graph_session_payload_plan.py --negative-test`, `cargo
  test -p ds4-gguf session_payload`, `python3 -m py_compile
  ds4-parity/compare_graph_session_payload_plan.py
  ds4-parity/run_parity_report.py`, `git diff --check`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 48 passed, 36
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all --
  --check`, and non-interactive Claude review with `NO BLOCKERS`.
- M10.6a splits M10.6 into `M10.6a` scheduling-plan parity, `M10.6b`
  short whole-prefill execution, `M10.6c` cold chunked-prefill execution, and
  `M10.6d` resumed-suffix execution so each slice has a concrete oracle,
  comparator, and acceptance boundary.
- M10.6a adds `rust/ds4-gpu/src/prefill_plan.rs` and
  `ds4-prefill-plan`, which emit `ds4.prefill_plan.v1` for the current-C
  default scheduling policy. The Rust plan mirrors default
  `ds4_default_prefill_cap_for_prompt`, `metal_graph_prefill_layer_major`,
  `metal_graph_prefill_chunked_range`, and
  `metal_graph_resume_prefill_min_tokens` behavior for six fixtures covering
  cold whole prefill, the 2048-token cap boundary, a 2052-token cold chunked
  prompt, resumed suffix alignment from token `1537`, short resumed decode
  fallback, and exact-prefix cache hit.
- M10.6a adds `ds4-parity/compare_prefill_plan_rust.py`, which validates six
  cases, six chunks, six progress points, the route, prefill cap, raw cap,
  chunk cap, first chunk, chunk starts/sizes, final output batch row, output
  absolute position, progress points, and layer-batch call counts. Candidate
  JSON validation passed against `/tmp/ds4-m106a-prefill-plan.json`, and the
  negative tests rejected 10 mutations.
- M10.6a validation passed `cargo test -p ds4-gpu prefill_plan`, `python3 -m
  py_compile ds4-parity/compare_prefill_plan_rust.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/compare_prefill_plan_rust.py`, `python3
  ds4-parity/compare_prefill_plan_rust.py --negative-test`, `cargo run -p
  ds4-gpu --bin ds4-prefill-plan --quiet >
  /tmp/ds4-m106a-prefill-plan.json`, `python3 -m json.tool
  /tmp/ds4-m106a-prefill-plan.json`, `python3
  ds4-parity/compare_prefill_plan_rust.py --candidate
  /tmp/ds4-m106a-prefill-plan.json`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 44 passed, 33
  skipped, and 0 failed, `cargo fmt --all -- --check`, `git diff --check`,
  touched-file NUL scan, `cargo test --workspace`, and non-interactive Claude
  review with `NO BLOCKERS`.
- M10.5c4d4 adds `ds4-directional-steering-oracle-dump` and
  `ds4_dump_directional_steering_decode_oracle_json`, which emit
  `ds4.directional_steering_decode_oracle.v1` for B300 `ds4flash.gguf`, token
  `0`, layer `0`, `dir-steering/out/verbosity.f32`, attention scale `0.5`,
  and FFN scale `0.25`. The current-C oracle loads the steering file through
  `metal_graph_load_directional_steering`, captures layer-0 post-steer
  `attn_out`, post-steer attention HC expansion, post-steer `ffn_out`,
  post-steer FFN HC expansion, final layer-42 HC, output-head tensors, and
  final logits.
- M10.5c4d4 extends `ds4-decode-full-output-head` with optional
  `--dir-steering-file`, `--dir-steering-attn`, and `--dir-steering-ffn` flags.
  The Rust candidate emits `ds4.decode_directional_steering.v1` and uses safe
  facade wrappers for `attention_output_q8_batch`,
  `directional_steering_project`, `add`, and `hc_expand_split` so the
  directional-steering attention and FFN branches match current-C placement.
- The B300 paired directional-steering validator passed 469 pinned checks with
  steering file FNV `960514fa6e7884ca`. Full-buffer FNV digests are
  `layer0_attn_out=68356dba6c067ffa`,
  `layer0_after_attn_hc=f1c47bcde7bdec38`,
  `layer0_ffn_out=7c8abeae9af7cc84`,
  `layer0_after_ffn_hc=db94a9015d610f1b`,
  `after_layer42_hc=7b8a60690319eff8`,
  `output_pre=5b6b7ffd274f62b2`,
  `output_weights=42a754df67d85acf`,
  `output_embd=c1e3490b198cf968`,
  `output_norm=53be42a180587d23`, and
  `logits=8caf00d359fba4f1`.
- M10.5c4d4 validation passed local `python3
  ds4-parity/compare_decode_directional_steering.py` with 33 checks, local and
  B300 `python3 ds4-parity/compare_decode_directional_steering.py
  --negative-test` with 13 rejected mutations, B300 current-C oracle plus Rust
  CUDA candidate validation with 469 checks, copied local artifact validation
  with 469 checks, local `arch -arm64 make
  ds4-directional-steering-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-full-output-head`, `python3 -m py_compile
  ds4-parity/compare_decode_directional_steering.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 43 passed, 33
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, touched-file NUL scan, and non-interactive
  Claude review with `NO BLOCKERS`. Artifact SHA256:
  `oracle=0f558a7d6cd9f24cf60b7f93b63479ecf3e3f7aeabe261608f657a2eb9bff3c1`
  and
  `rust=c27189cc8a719fc9431a8c2b02f33a95d5c44b04c329c44e48fa925c30f538f7`.
- M10.5c4d3 adds `ds4-long-indexed-attention-oracle-dump` and
  `ds4_dump_long_indexed_attention_oracle_json`, which emit
  `ds4.long_indexed_attention_oracle.v1` for deterministic tokens `0..2051`.
  The current-C oracle warms up tokens `0..2050` through production
  `metal_graph_eval_token_raw_swa`, then manually executes token `2051`
  through layer `2` so the first ratio-4 indexed-attention branch crosses the
  strict `DS4_N_INDEXER_TOP_K` threshold without requiring layer-major prefill.
- M10.5c4d3 adds `ds4-decode-long-indexed-attention`, which executes the same
  2,051 full-token Rust CUDA decode warmup, stops token `2051` after layer `2`,
  calls the indexed mixed-attention backend with the same top-k rows, and emits
  layer-2 HC, heads, attention output, indexer, selected-row, raw-cache, and
  compressed-row checkpoints.
- M10.5c4d3 makes the CUDA single-token indexed-attention fallback fill
  `comp_rows` deterministically in top-k order instead of racing through an
  atomic counter; this removed process-to-process FNV drift while preserving the
  same selected rows.
- The B300 paired long indexed-attention validator passed 644 checks before
  digest pinning and 666 pinned checks locally against copied artifacts. It
  matched `sequence_len=2052`, `final_position=2051`, `full_decode_tokens=2051`,
  `final_decoded_layers=3`, `total_decode_layer_calls=88196`,
  `raw_row=2051`, `raw_start=1924`, `n_raw=128`, `layer2_n_comp=513`,
  `layer2_n_index_comp=513`, and `layer2_final_comp_row=512`. Full-buffer FNV
  digests are `after_layer2_hc=8e3d1c2ef4ac4e1f`,
  `layer2_heads=152cefad5f4521d0`,
  `layer2_attn_out=d31399afb15f9523`,
  `layer2_after_attn_hc=ce72c471b910e3e4`,
  `layer2_indexer_q=e18d30079195cac8`,
  `layer2_indexer_weights=67d5b94599c46b4e`,
  `layer2_indexer_scores=1e190e89087c4d93`,
  `layer2_comp_selected=96be5e90e07d5fe3`,
  `layer2_raw_cache_row=1eccdd715c4f26b1`,
  `layer2_attn_comp_row512=25b13ef81b3cc643`, and
  `layer2_index_comp_row512=8bf040cdf84597fb`.
- M10.5c4d3 validation passed static and negative long indexed-attention
  comparator checks, local `arch -arm64 make
  ds4-long-indexed-attention-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-long-indexed-attention`, B300 current-C oracle plus Rust CUDA
  candidate validation with 644 checks, pinned local artifact validation with
  666 checks, C4d2 artifact cross-check, `python3 -m py_compile
  ds4-parity/compare_decode_long_indexed_attention.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 42 passed, 32
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, touched-file NUL scan, non-interactive Claude
  review with `NO BLOCKERS`, and artifact SHA256
  `oracle=26aab1234b7ca7527dd2aa10f522ffce199d147187e8ec05e86ed504b79b9eed`
  and
  `rust=3406e9f746471cad0f2cbfe4f23297d2438f857b21140bf068e6386422eb4f1d`.
- M10.5c4d2 adds `ds4-ratio-boundary-output-head-oracle-dump` and
  `ds4_dump_ratio_boundary_output_head_oracle_json`, which emit
  `ds4.ratio_boundary_output_head_oracle.v1` for the deterministic token
  sequence `0..127`, final position `127`, production current-C
  `metal_graph_eval_token_raw_swa`, final layer-42 HC, output-head tensors,
  ratio-4 row `31`, ratio-128 row `0`, and selected raw/cache state.
- M10.5c4d2 adds `ds4-decode-ratio-boundary-output-head`, which maps the real
  GGUF on B300, executes 128 tokens through all 43 Rust safe-facade decode
  layers, flushes after layer `3`, emits and quantizes ratio-4 and ratio-128
  compressed rows at the final token, stays below the indexed-attention
  threshold, and runs the final output-head/logits kernels.
- The B300 paired ratio-boundary output-head validator passed 829 checks
  before digest pinning and 865 pinned checks locally against copied artifacts.
  It matched `sequence_len=128`, `final_position=127`, `raw_row=127`,
  `raw_start=0`, `n_raw=128`, `emit_compressed_row=1`,
  `layer2_n_comp=32`, `layer2_n_index_comp=32`, `layer5_n_comp=1`,
  `layer42_n_comp=32`, and `layer42_n_index_comp=32`. Full-buffer FNV
  digests are `after_layer42_hc=12f1089ad3297673`,
  `output_pre=71f7d1ca0703e093`,
  `output_weights=3e646960d299fca0`,
  `output_embd=3f0d9c27cf78b430`,
  `output_norm=a1baf22acb3476dc`,
  `logits=c67eab1a566286ae`,
  `layer2_raw_cache_row=cfc54c8671abaa5a`,
  `layer2_attn_comp_row31=72353245d1b57607`,
  `layer2_index_comp_row31=63be8943c4bf8cd2`,
  `layer5_raw_cache_row=082429f33ac1c8df`,
  `layer5_attn_comp_row0=e65ab25c4927545f`,
  `layer5_attn_state_kv=49fb25b3760e6207`,
  `layer5_attn_state_score=3e158062911a288e`,
  `layer42_raw_cache_row=3346c7f9ebeed46e`,
  `layer42_attn_comp_row31=6b9b38fa19457e18`,
  `layer42_index_comp_row31=2a0d37865baff695`,
  `layer42_attn_state_kv=0aa0087d1d1dcd79`, and
  `layer42_index_state_kv=1e0df1e98d453bcd`.
- M10.5c4d2 validation passed `python3
  ds4-parity/compare_decode_ratio_boundary_output_head.py --negative-test`,
  paired local artifact validation with 865 checks, local `arch -arm64 make
  ds4-ratio-boundary-output-head-oracle-dump`, local `cargo check -p ds4-gpu
  --bin ds4-decode-ratio-boundary-output-head`, B300 current-C oracle plus
  Rust CUDA candidate validation with 829 checks, `python3 -m py_compile
  ds4-parity/compare_decode_ratio_boundary_output_head.py
  ds4-parity/run_parity_report.py`, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, touched-file NUL scan,
  non-interactive Claude review with `NO BLOCKERS`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 41 passed, 31
  skipped, and 0 failed, and artifact SHA256
  `oracle=b8e813b11312f931a4bb786d297661d933588bba78da7bad78a147653c2c58c7`
  and
  `rust=72fbe9424cecf96d6710d0a1adc43563a5d7af0360ec804628e9224bec081449`.
- M10.5c4d1 adds `ds4-short-continuation-output-head-oracle-dump` and
  `ds4_dump_short_continuation_output_head_oracle_json`, which emit
  `ds4.short_continuation_output_head_oracle.v1` for the deterministic
  token sequence `0..21`, final position `21`, production current-C
  `metal_graph_eval_token_raw_swa`, final layer-42 HC, output-head tensors,
  selected raw-cache rows, and selected ratio-4/ratio-128 compressed cache
  state.
- M10.5c4d1 adds `ds4-decode-short-continuation-output-head`, which maps the
  real GGUF on B300, executes 22 tokens through all 43 Rust safe-facade decode
  layers, flushes after layer `3`, emits and quantizes ratio-4 compressed rows,
  preserves the indexed-attention branch for a later long-context milestone,
  and runs the final output-head/logits kernels.
- The B300 paired short continuation output-head validator passed 766 checks
  before digest pinning and 798 pinned checks locally against copied artifacts.
  It matched `sequence_len=22`, `final_position=21`, `raw_row=21`,
  `raw_start=0`, `n_raw=22`, `layer2_n_comp=5`,
  `layer2_n_index_comp=5`, `layer5_n_comp=0`, `layer42_n_comp=5`, and
  `layer42_n_index_comp=5`. Full-buffer FNV digests are
  `after_layer42_hc=40e22a11d8ca9178`,
  `output_pre=642c2b6d18b62c67`,
  `output_weights=9592e0f3a26737e1`,
  `output_embd=e57d3ebe8ed8c63c`,
  `output_norm=1615bc086702b3b8`,
  `logits=fcc73408cecb8073`,
  `layer2_raw_cache_row=3befca08431b15ed`,
  `layer2_attn_comp_row4=061fb5b8eabae3db`,
  `layer2_index_comp_row4=a8afc0bf90381f52`,
  `layer5_attn_state_kv=2c574c58aad15bc1`,
  `layer5_attn_state_score=71948016152ae1de`,
  `layer42_raw_cache_row=998292db4c5534e7`,
  `layer42_attn_comp_row4=24844d05b88a2c04`,
  `layer42_index_comp_row4=c7e7a2f46c2aa3b2`,
  `layer42_attn_state_kv=cf3576176ae9d092`, and
  `layer42_index_state_kv=06ac626b7530144e`.
- M10.5c4d1 validation passed `python3
  ds4-parity/compare_decode_short_continuation_output_head.py --negative-test`,
  paired local artifact validation with 798 checks, local `arch -arm64 make
  ds4-short-continuation-output-head-oracle-dump`, local `cargo check -p
  ds4-gpu --bin ds4-decode-short-continuation-output-head`, B300 current-C
  oracle plus Rust CUDA candidate validation with 766 checks, B300 predecessor
  full output-head rerun with 430 checks, `python3 -m py_compile
  ds4-parity/compare_decode_short_continuation_output_head.py
  ds4-parity/run_parity_report.py`, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, touched-file NUL scan,
  non-interactive Claude review with `NO BLOCKERS`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 40 passed, 30
  skipped, and 0 failed, and artifact SHA256
  `oracle=7c53400cef52a6f73aa8fea06ec4f64298d045bd0776397cc6f3030bbdf38429`
  and
  `rust=f00f23abc84474e7d00a8958ebb0c4f055889a384744b004954cfdfa9eb651a6`.
- M10.5c4c2b2b2b2b2b2b2b2b2b adds
  `ds4-full-output-head-oracle-dump` and
  `ds4_dump_full_output_head_oracle_json`, which emit
  `ds4.full_output_head_oracle.v1` for token `0`, position `0`, all 43
  production current-C decode layers, HC swaps after every layer, final
  layer-42 HC, output-head HC pre/weights/embedding/norm tensors, and final
  vocab logits.
- M10.5c4c2b2b2b2b2b2b2b2b2b adds
  `ds4-decode-full-output-head`, which reuses the all-layer Rust safe-facade
  scheduler, runs output-head `rms_norm_plain`, `matmul_f16`,
  `output_hc_weights`, `hc_weighted_sum`, `rms_norm_weight`, and
  `matmul_q8_0`, and emits pinned tensor digests for the final HC and logits.
- The B300 paired full output-head validator passed 440 pinned checks with
  `decoded_layers=43`, `dense_layers=2`, `ratio4_layers=21`,
  `ratio128_layers=20`, `raw_cap=2304`, `raw_window=128`,
  `layer5_n_comp=0`, `layer42_n_comp=0`, and `layer42_n_index_comp=0`.
  Full-buffer FNV digests are
  `after_layer42_hc=cbd17b425564f63f`,
  `output_pre=91ea6aeb7a0a0d9f`,
  `output_weights=323062ce53dc6f9c`,
  `output_embd=8788c46e4f0a1f30`,
  `output_norm=185c73c1de55a942`, and
  `logits=432eef0524ced3ad`.
- M10.5c4c2b2b2b2b2b2b2b2b2b validation passed `python3
  ds4-parity/compare_decode_full_output_head.py --negative-test`, static
  comparator validation, paired local artifact validation with 440 checks,
  local `arch -arm64 make ds4-full-output-head-oracle-dump`, local
  `cargo check -p ds4-gpu --bin ds4-decode-full-output-head`, B300 current-C
  oracle plus Rust CUDA candidate validation with 440 checks, B300
  c2b2b2b2b2b2a two-layer output-head predecessor rerun with 446 checks, B300
  c2b2b2b2b2b2b2b2b2a all-layer final-HC predecessor rerun with 730 checks,
  `python3 -m py_compile ds4-parity/compare_decode_full_output_head.py
  ds4-parity/run_parity_report.py`, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, touched-file NUL scan,
  non-interactive Claude review with `NO BLOCKERS`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 39 passed, 29
  skipped, and 0 failed, and artifact SHA256
  `oracle=0377c0a6a45e8b7ee1d52d2b78e850cda41f8d4180a7ccec18d282ede5484d3b`
  and
  `rust=31fb1985b493ac60e9491fde9280f3e027b882f056991efc67995d8bbd40b988`.
- M10.5c4c2b2b2b2b2b2b2b2b2a adds
  `ds4-all-layer-final-hc-oracle-dump` and
  `ds4_dump_all_layer_final_hc_oracle_json`, which emit
  `ds4.all_layer_final_hc_oracle.v1` for token `0`, position `0`, all 43
  production current-C decode layers, HC swaps after every layer, HC
  checkpoints after layers `4`, `5`, and `42`, raw-cache rows for layers `5`
  and `42`, ratio-128 layer-5 attention compressor state, and ratio-4
  layer-42 attention/indexer compressor state.
- M10.5c4c2b2b2b2b2b2b2b2b2a adds
  `ds4-decode-all-layer-final-hc`, which maps the real GGUF on B300, allocates
  per-layer raw/cache/state tensors, executes all layers through the safe Rust
  facade using the DS4 compression schedule, preserves zero compressed/indexer
  counters for position `0`, swaps `cur_hc`/`after_ffn_hc` after each layer,
  and stops before the output-head kernels.
- The B300 paired all-layer final-HC validator passed 730 pinned checks with
  `decoded_layers=43`, `dense_layers=2`, `ratio4_layers=21`,
  `ratio128_layers=20`, `raw_cap=2304`, `raw_window=128`,
  `layer5_comp_cap=258`, `layer42_comp_cap=8194`, and zero compressed/indexer
  counters. Full-buffer FNV digests are
  `after_layer4_hc=b19322ec84d84935`,
  `after_layer5_hc=b9c9026559412805`,
  `after_layer42_hc=cbd17b425564f63f`,
  `layer5_raw_cache_row=8f2606992a7f1a18`,
  `layer5_attn_state_kv=8c17d55c4b8e6de9`,
  `layer5_attn_state_score=292852343a4b4512`,
  `layer42_raw_cache_row=029806013304ca31`,
  `layer42_attn_state_kv=42a2f55a8dc3403b`,
  `layer42_attn_state_score=5b0a233b9c74b3ee`,
  `layer42_index_state_kv=2f5aefc0f5ed2728`, and
  `layer42_index_state_score=6a5003b30aad1406`.
- M10.5c4c2b2b2b2b2b2b2b2b2a validation passed `python3
  ds4-parity/compare_decode_all_layer_final_hc.py --negative-test`, `python3
  ds4-parity/compare_decode_all_layer_final_hc.py`, paired local artifact
  validation with 730 checks, local `arch -arm64 make
  ds4-all-layer-final-hc-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-all-layer-final-hc`, B300 current-C oracle plus Rust CUDA
  candidate validation with 730 checks, B300 c2b2b2b2b2b2b2b2b1 layer-4
  FFN-output predecessor rerun with 1,209 checks, `python3 -m py_compile
  ds4-parity/compare_decode_all_layer_final_hc.py ds4-parity/run_parity_report.py`,
  `cargo test --workspace`, `cargo fmt --all -- --check`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 38 passed, 28
  skipped, and 0 failed, `git diff --check`, touched-file NUL scan,
  non-interactive Claude review with `NO BLOCKERS`, and
  artifact SHA256
  `oracle=a86956a394942dae56c446f4c51f37790f50b3b249c01c8fa932aac1efba9fb3`
  and
  `rust=e3e124043cd2d829c23bc8196971b0fa498fd63feca8b66473919bf3313807c1`.
- M10.5c4c2b2b2b2b2b2b2b2b1 adds
  `ds4-layer4-ffn-output-oracle-dump` and
  `ds4_dump_layer4_ffn_output_oracle_json`, which emit
  `ds4.layer4_ffn_output_oracle.v1` for token `0`, position `0`, dense
  layers `0` and `1` through production current-C decode-layer execution and
  HC swaps, layer `2` through production ratio-4 decode plus HC swap, layer
  `3` through production ratio-128 decode plus HC swap, then layer `4`
  through the first post-ratio128 ratio-4/indexer layer without the final HC
  swap.
- M10.5c4c2b2b2b2b2b2b2b2b1 adds
  `ds4-decode-layer4-ffn-output`, which maps the real GGUF on B300, runs the
  validated dense layer `0`/`1`, layer-2 FFN-output, and layer-3
  FFN-output prefix, swaps the layer-3 HC output into `cur_hc`, then runs
  layer `4` ratio-4 attention and indexer `matmul_f16_pair`,
  `compressor_update`, raw-only `attention_decode_heads`, attention output HC
  expansion, router selection, routed MoE, shared expert, and final FFN HC
  expansion through the safe Rust facade.
- The B300 paired layer-4 post-ratio128 ratio-4/indexer FFN-output validator
  passed 1,383 pinned checks with `compression_ratio=4`,
  `compressor_coefficient=2`, `layer_comp_cap=8194`,
  `attn_state_dim=8192`, `index_state_dim=2048`, router bias enabled, and
  router hash disabled. Full-buffer FNV digests are
  `after_layer3_hc=734775286457caef`,
  `layer4_raw_cache_row=773d10a59842c20a`,
  `layer4_attn_state_kv=154dccb1209e67d0`,
  `layer4_attn_state_score=c244ef562e602ca4`,
  `layer4_index_state_kv=87580106eb9c5c3d`,
  `layer4_index_state_score=959468ecfe1ed8b7`,
  `layer4_heads=1d3d9836714526a6`,
  `layer4_attn_low=bfec0203a220e7dd`,
  `layer4_attn_out=51b41d3c5b2a78e6`,
  `layer4_after_attn_hc=9d92af0dabbe42af`,
  `layer4_ffn_cur=bd6bb8a0b394edbb`,
  `layer4_ffn_norm=2dc5bde361206e1a`,
  `layer4_router_logits=ca0e6d89162fe6eb`,
  `layer4_router_probs=9e9f6f56421fbcd3`,
  `layer4_router_selected=ec6043f1523b2257`,
  `layer4_router_weights=7f9ddb23390151b2`,
  `layer4_routed_mid=27995a19a136f82a`,
  `layer4_routed_out=1bc6d01dc8139185`,
  `layer4_shared_mid=6df189c1c09015f0`,
  `layer4_shared_out=abefadb626b0180b`, and
  `layer4_after_ffn_hc=b19322ec84d84935`.
- M10.5c4c2b2b2b2b2b2b2b2b1 validation passed `python3
  ds4-parity/compare_decode_layer4_ffn_output.py --negative-test`, `python3
  ds4-parity/compare_decode_layer4_ffn_output.py`, paired local artifact
  validation with 1,383 checks, local `arch -arm64 make
  ds4-layer4-ffn-output-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer4-ffn-output`, B300 current-C oracle plus Rust CUDA
  candidate validation with 1,209 checks before pinning, B300 c2b2b2b2b2b2b2b2a
  layer-3 ratio-128 FFN-output rerun with 1,261 checks, `python3 -m
  py_compile ds4-parity/compare_decode_layer4_ffn_output.py
  ds4-parity/run_parity_report.py`, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with 37 passed, 27 skipped, and 0 failed, `git diff
  --check`, touched-file NUL scan, non-interactive Claude review with
  `NO BLOCKERS`, and artifact SHA256
  `oracle=71ca50531844a0b062aa2f44773c8247fd38a3ebfe92877c9ed0bf8269c2f7eb`
  and
  `rust=fe75f0554993151a5dfff569d8517b6b70acb4a42a7c710a661d4180ac2cdd28`.
- M10.5c4c2b2b2b2b2b2b2b2b2 splits the remaining layer-loop/logits item into
  M10.5c4c2b2b2b2b2b2b2b2b2a all-layer final-HC execution and
  M10.5c4c2b2b2b2b2b2b2b2b2b full one-token output-head/logits execution. The
  split keeps the next commit comparable at the post-layer-42 HC boundary
  before final vocab projection is attached.
- M10.5c4c2b2b2b2b2b2b2b2 splits the remaining one-token scheduler item into
  a layer-3 ratio-128 FFN-output execution bridge and the next full
  all-layer/output-head/logits slice M10.5c4c2b2b2b2b2b2b2b2b. The split keeps
  the next commit comparable at the first ratio-128 compressed layer boundary
  before the repeated remaining layers and final logits are introduced.
- M10.5c4c2b2b2b2b2b2b2b2b splits the remaining one-token scheduler item into
  a layer-4 post-ratio128 ratio-4/indexer FFN-output execution bridge and the
  next remaining layer-loop/logits slice M10.5c4c2b2b2b2b2b2b2b2b2. The split
  keeps the current commit comparable at the first ratio128-to-ratio4
  transition before repeated layer-loop and output-head/logits closure.
- M10.5c4c2b2b2b2b2b2b2b2a adds
  `ds4-layer3-ffn-output-oracle-dump` and
  `ds4_dump_layer3_ffn_output_oracle_json`, which emit
  `ds4.layer3_ffn_output_oracle.v1` for token `0`, position `0`, dense
  layers `0` and `1` through production current-C decode-layer execution and
  HC swaps, layer `2` through production ratio-4 decode plus HC swap, then
  layer `3` through the first ratio-128 compressed layer with no indexer state.
- M10.5c4c2b2b2b2b2b2b2b2a adds
  `ds4-decode-layer3-ffn-output`, which maps the real GGUF on B300, runs the
  validated dense layer `0`/`1` and layer-2 FFN-output prefix, swaps the
  layer-2 HC output into `cur_hc`, then runs layer `3` ratio-128
  `matmul_f16_pair`, `compressor_update`, raw-only `attention_decode_heads`,
  attention output HC expansion, router selection, routed MoE, shared expert,
  and final FFN HC expansion through the safe Rust facade.
- The B300 paired layer-3 ratio-128 FFN-output validator passed 1,261 pinned
  checks with `compression_ratio=128`, `compressor_coefficient=1`,
  `has_indexer=false`, `attn_state_dim=65536`, router bias enabled, and router
  hash disabled. Full-buffer FNV digests are
  `after_layer2_hc=26babcdeac41b377`,
  `layer3_raw_cache_row=d20115e20ce6b227`,
  `layer3_attn_state_kv=cff54fc174994e78`,
  `layer3_attn_state_score=cf72b4d7a2540261`,
  `layer3_heads=2873c18505f20162`,
  `layer3_attn_low=56cc1933165cb906`,
  `layer3_attn_out=77d0e59ceb43ba15`,
  `layer3_after_attn_hc=e97f76051b8b0abc`,
  `layer3_ffn_cur=ac1b6cf1006a1af6`,
  `layer3_ffn_norm=df3fa50fae679d17`,
  `layer3_router_logits=4fc7b21579345fe0`,
  `layer3_router_probs=84d328bf4409ed88`,
  `layer3_router_selected=75eb5975465f7fea`,
  `layer3_router_weights=8d1d5f3cd181558d`,
  `layer3_routed_mid=349df5a076aa05b5`,
  `layer3_routed_out=006bf65de34acf9e`,
  `layer3_shared_mid=cf125b8f97a3ea96`,
  `layer3_shared_out=f8a526b184ed7bcb`, and
  `layer3_after_ffn_hc=734775286457caef`.
- M10.5c4c2b2b2b2b2b2b2b2a validation passed `python3
  ds4-parity/compare_decode_layer3_ffn_output.py --negative-test`, `python3
  ds4-parity/compare_decode_layer3_ffn_output.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer3_ffn_output.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer3-ffn-output-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer3-ffn-output`, B300 current-C oracle plus Rust candidate
  paired validation with artifact SHA256
  `oracle=932b669ec3b4fdee0369b745968f92dc7ebc3c97e9b063b012bd380118dde9df`
  and
  `rust=3153600948c0e41e4b2fa01075eb8f0d1d2824435a46b2b4d365569b70ef1797`,
  B300 c2b2b2b2b2b2b2b1 layer-2 FFN-output rerun with 1,383 checks, local
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with `NO BLOCKERS`.
- M10.5c4c2b2b2b2b2b2b2b splits the remaining one-token scheduler item into a
  layer-2 FFN-output execution bridge and the next full scheduler/logits slice
  M10.5c4c2b2b2b2b2b2b2b2. The split keeps the next commit comparable at the
  layer-2 FFN HC expansion boundary before the remaining layers, ratio-128
  coverage, output head, and final logits are introduced together.
- M10.5c4c2b2b2b2b2b2b2b1 adds
  `ds4-layer2-ffn-output-oracle-dump` and
  `ds4_dump_layer2_ffn_output_oracle_json`, which emit
  `ds4.layer2_ffn_output_oracle.v1` for token `0`, position `0`, dense
  layers `0` and `1` through the production current-C decode-layer encoder and
  HC swaps, then layer `2` through production attention-output, FFN HC-pre,
  router selection, routed MoE, shared expert, and final `after_ffn_hc`.
- M10.5c4c2b2b2b2b2b2b2b1 adds
  `ds4-decode-layer2-ffn-output`, which maps the real GGUF on B300, binds DS4
  weights, launches the validated dense layer `0`/`1` and layer-2
  attention-output prefix, then runs `hc_split_weighted_sum_norm`,
  `router_select`, `routed_moe_one`, `shared_gate_up_swiglu_q8_0`, and
  `shared_down_hc_expand_q8_0` through the safe Rust facade.
- The B300 paired layer-2 FFN-output validator passed 1,383 pinned checks with
  full-buffer FNV digests `layer2_ffn_cur=d0becc7729c8b33d`,
  `layer2_ffn_norm=ead9d19c71277f8a`,
  `layer2_router_logits=89b254c2cac1245a`,
  `layer2_router_probs=23b3dac5b0b03386`,
  `layer2_router_selected=cadcd78086393cff`,
  `layer2_router_weights=fa578aa92d03d83d`,
  `layer2_routed_mid=16dd75f68757ccb7`,
  `layer2_routed_out=4ffcffcb3c9d6daf`,
  `layer2_shared_mid=e717a53c9497794b`,
  `layer2_shared_out=3e7b4f4e70d9b893`, and
  `layer2_after_ffn_hc=26babcdeac41b377`, while preserving the previously
  pinned layer-2 attention-output boundary digests. Pinned layer-2 FFN weights
  include `layer2_hc_ffn_fn=(79456856864,786432,1,f16)`,
  `layer2_ffn_gate_tid2eid=(11539264,3102720,26,i32)`,
  `layer2_ffn_gate_exps=(3638520640,553648128,16,iq2_xxs)`,
  `layer2_ffn_up_exps=(4896811840,553648128,16,iq2_xxs)`,
  `layer2_ffn_down_exps=(4192168768,704643072,10,q2_k)`, and
  `layer2_ffn_down_shexp=(79438228032,8912896,8,q8_0)`.
- M10.5c4c2b2b2b2b2b2b2b1 validation passed `python3
  ds4-parity/compare_decode_layer2_ffn_output.py --negative-test`, `python3
  ds4-parity/compare_decode_layer2_ffn_output.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer2_ffn_output.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer2-ffn-output-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-ffn-output`, B300 current-C oracle plus Rust candidate
  paired validation with artifact SHA256
  `oracle=63ab2f77418ec2ea933dfb223f056b2ffeabd6857b8aa86f0137187cdcb36242`
  and
  `rust=ba67b7cd2abfcad3ddbcaa0c48895cbc1176f729128703748fbb1a231be33503`,
  B300 c2b2b2b2b2b2b2a layer-2 attention-output rerun with 815 checks, local
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 35
  passed, 25 skipped, and 0 failed, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, touched-file NUL scan, and
  non-interactive Claude review with `NO BLOCKERS`.
- M10.5c4c2b2b2b2b2b2b2a adds
  `ds4-layer2-attn-output-oracle-dump` and
  `ds4_dump_layer2_attn_output_oracle_json`, which emit
  `ds4.layer2_attn_output_oracle.v1` for token `0`, position `0`, dense
  layers `0` and `1` through the production current-C decode-layer encoder and
  HC swaps, then layer `2` through the production decode-layer path while
  reading the layer-2 raw cache row, compressor frontier state, attention heads
  after inverse compressed RoPE, attention low/output tensors, and
  `after_attn_hc`.
- M10.5c4c2b2b2b2b2b2b2a adds
  `ds4-decode-layer2-attn-output`, which maps the real GGUF on B300, binds DS4
  weights, launches the validated dense layer `0`/`1` prefix, swaps HC buffers
  after each dense layer, executes layer `2` Q/KV/RoPE, raw KV store, attention
  and indexer compressor-state mutation, then runs raw-only
  `attention_decode_heads`, inverse compressed `rope_tail`,
  `attention_output_low_q8`, and `matmul_q8_0_hc_expand` through the safe Rust
  facade.
- The B300 paired layer-2 attention-output validator passed 815 pinned checks
  with full-buffer FNV digests `after_layer1_hc=f764d7067de5c945`,
  `layer2_raw_cache_row=51f0a2971a59c6da`,
  `layer2_attn_state_kv=57544afc0dfa6bcf`,
  `layer2_attn_state_score=38d2d40c6f170ab6`,
  `layer2_index_state_kv=2a44d6b140b6ef0b`,
  `layer2_index_state_score=b8da053681327aec`,
  `layer2_heads=241a32d72fe7885b`,
  `layer2_attn_low=6d33e52dbc93ed09`,
  `layer2_attn_out=c5a61256ab424d80`, and
  `layer2_after_attn_hc=9c038ab7c95176b4`. Pinned layer-2 attention-output
  weights include `layer2_attn_sinks=(79275269952,256,0,f32)`,
  `layer2_attn_output_a=(79315790400,35651584,8,q8_0)`, and
  `layer2_attn_output_b=(79351441984,35651584,8,q8_0)`.
- M10.5c4c2b2b2b2b2b2b2a validation passed `python3
  ds4-parity/compare_decode_layer2_attn_output.py --negative-test`, `python3
  ds4-parity/compare_decode_layer2_attn_output.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer2_attn_output.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer2-attn-output-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer2-attn-output`, B300 current-C oracle plus Rust candidate
  paired validation with artifact SHA256
  `oracle=728fa6b858f9ff6669424eac7691b65d1ffe9d78e9c5f7cbe85c412cc5ce80a7`
  and
  `rust=8b561a5eca4874bb4ba6bbf5bc83080d761f294a7cce839435c5815ff2db9ca3`,
  B300 c2b2b2b2b2b2b1 layer-2 compressor-state rerun with 589 checks, local
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with 34
  passed, 24 skipped, and 0 failed, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, and a touched-file NUL scan.
- M10.5c4c2b2b2b2b2b2b2 splits the remaining one-token scheduler item into a
  layer-2 attention-output execution bridge and the next full scheduler/logits
  slice M10.5c4c2b2b2b2b2b2b2b. The split keeps the next commit comparable at
  the raw-only layer-2 attention, inverse compressed RoPE, attention projection,
  and HC expansion boundary before the layer-2 FFN and remaining scheduler are
  introduced together.
- M10.5c4c2b2b2b2b2b2b1 adds
  `ds4-layer2-compressor-state-oracle-dump` and
  `ds4_dump_layer2_compressor_state_oracle_json`, which emit
  `ds4.layer2_compressor_state_oracle.v1` for token `0`, position `0`, dense
  layers `0` and `1` through the production current-C decode-layer encoder and
  HC swaps, then layer `2` through raw KV store plus ratio-4 attention/indexer
  compressor-state mutation.
- M10.5c4c2b2b2b2b2b2b1 adds
  `ds4-decode-layer2-compressor-state`, which maps the real GGUF on B300,
  binds DS4 weights, launches the validated dense layer `0`/`1` prefix, swaps
  HC buffers after each dense layer, then executes layer `2` Q/KV/RoPE, raw KV
  store, attention `matmul_f16_pair`, attention `compressor_update`, indexer
  `matmul_f16_pair`, and indexer `compressor_update` through the safe Rust
  facade.
- The B300 paired layer-2 compressor-state validator passed 589 pinned checks
  with full-buffer FNV digests `after_layer1_hc=f764d7067de5c945`,
  `layer2_raw_cache_row=51f0a2971a59c6da`,
  `layer2_attn_state_kv=57544afc0dfa6bcf`,
  `layer2_attn_state_score=38d2d40c6f170ab6`,
  `layer2_index_state_kv=2a44d6b140b6ef0b`, and
  `layer2_index_state_score=b8da053681327aec`. Pinned layer-2 compressor
  weights include `layer2_attn_compressor_kv=(79283669056,8388608,1,f16)`,
  `layer2_attn_compressor_gate=(79275280448,8388608,1,f16)`,
  `layer2_indexer_compressor_kv=(79294157376,2097152,1,f16)`, and
  `layer2_indexer_compressor_gate=(79292060224,2097152,1,f16)`.
- M10.5c4c2b2b2b2b2b2b1 validation passed `python3
  ds4-parity/compare_decode_layer2_compressor_state.py --negative-test`,
  `python3 ds4-parity/compare_decode_layer2_compressor_state.py`, `python3 -m
  py_compile ds4-parity/compare_decode_layer2_compressor_state.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer2-compressor-state-oracle-dump`, local `cargo check -p ds4-gpu
  --bin ds4-decode-layer2-compressor-state`, B300 current-C oracle plus Rust
  candidate paired validation, pinned B300 artifact rerun, B300 c2b2b2b2b2b2a
  two-layer output-head rerun with 446 checks, local `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 33 passed, 23
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, and a touched-file NUL scan.
- M10.5c4c2b2b2b2b2b2b splits the remaining one-token scheduler item into a
  first ratio-4 compressor/indexer state mutation bridge and the next
  compressed-attention/all-layer scheduler slice M10.5c4c2b2b2b2b2b2b2. The
  split keeps the next commit comparable at the first layer-2 persistent cache
  mutation boundary before compressed attention and final logits are introduced.
- M10.5c4c2b2b2b2b2b2a adds `ds4-two-layer-output-head-oracle-dump` and
  `ds4_dump_two_layer_output_head_oracle_json`, which emit
  `ds4.two_layer_output_head_oracle.v1` for token `0`, position `0`, layers
  `0` and `1` through the production current-C decode-layer encoder, the
  production `cur_hc`/`after_ffn_hc` swap after each layer, and the production
  output-head encoder.
- M10.5c4c2b2b2b2b2b2a adds `ds4-decode-two-layer-output-head`, which maps the
  real GGUF on B300, binds DS4 weights, launches dense layer 0, swaps the HC
  buffers, launches dense layer 1 with its own raw cache, swaps again, runs the
  output-head path from the layer-1 HC through the safe Rust facade in one
  command batch, synchronizes, and reads back both layer HC boundaries plus
  output-head tensors and logits.
- The B300 paired two-dense-layer output-head validator passed 446 pinned checks
  with full-buffer FNV digests `after_layer0_hc=3d49316c93ce351f`,
  `after_layer1_hc=f764d7067de5c945`, `output_pre=ebc1b8ccc088d27a`,
  `output_weights=e20bda6aca5453b2`, `output_embd=b5d1377b7c179886`,
  `output_norm=2ce848a4cc2363db`, and `logits=14dbbac3cd6ed7a8`. Pinned
  output-head weights are `token_embd=(77928033088,1059061760,1,f16)`,
  `output_hc_fn=(86157337440,131072,1,f16)`,
  `output_hc_scale=(86157468512,4,0,f32)`,
  `output_hc_base=(86157337408,16,0,f32)`,
  `output_norm=(86720095104,16384,0,f32)`, and
  `output=(86157468544,562626560,8,q8_0)`.
- M10.5c4c2b2b2b2b2b2a validation passed `python3
  ds4-parity/compare_decode_two_layer_output_head.py --negative-test`,
  `python3 ds4-parity/compare_decode_two_layer_output_head.py`, `python3 -m
  py_compile ds4-parity/compare_decode_two_layer_output_head.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-two-layer-output-head-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-two-layer-output-head`, B300 current-C oracle plus Rust candidate
  paired validation, pinned B300 artifact rerun, and B300 c2b2b2b2b2b1
  layer-0 output-head rerun with 399 checks, local `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 32 passed, 22
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and a touched-file NUL scan across 11 files.
- M10.5c4c2b2b2b2b2b2 splits the remaining one-token scheduler item into a
  two-dense-layer output-head execution bridge and the next compressed/all-layer
  scheduler slice M10.5c4c2b2b2b2b2b2b. The split keeps the next commit
  comparable at the production `cur_hc`/`after_ffn_hc` buffer-swap boundary
  before layer-2 compressed-cache mutation is introduced.
- M10.5c4c2b2b2b2b2b1 adds `ds4-layer0-output-head-oracle-dump` and
  `ds4_dump_layer0_output_head_oracle_json`, which emit
  `ds4.layer0_output_head_oracle.v1` for token `0`, layer `0`, position `0`
  using the current-C model loader, config validation, weight binding, model
  fd/map bridge, the production `metal_graph_encode_decode_layer` for layer 0,
  and the production `metal_graph_encode_output_head` path.
- M10.5c4c2b2b2b2b2b1 adds `ds4-decode-layer0-output-head`, which maps the real
  GGUF on B300, binds DS4 weights, launches the validated layer-0 prefix plus
  `rms_norm_plain`, output HC preprojection, `output_hc_weights`,
  `hc_weighted_sum`, output RMS norm, and vocab projection through the safe Rust
  facade in one command batch, synchronizes, and reads back `after_ffn_hc`,
  `output_pre`, `output_weights`, `output_embd`, `output_norm`, and `logits`.
- The B300 paired layer-0 output-head validator passed 399 pinned checks with
  full-buffer FNV digests `after_ffn_hc=3d49316c93ce351f`,
  `output_pre=67cd67e9413ba488`, `output_weights=b7b3f62be8581476`,
  `output_embd=0b0d4f86243397e3`, `output_norm=24029c3b5c92306e`, and
  `logits=27d2e668424d8d9f`. Pinned output-head weights are
  `output_hc_fn=(86157337440,131072,1,f16)`,
  `output_hc_scale=(86157468512,4,0,f32)`,
  `output_hc_base=(86157337408,16,0,f32)`,
  `output_norm=(86720095104,16384,0,f32)`, and
  `output=(86157468544,562626560,8,q8_0)`.
- M10.5c4c2b2b2b2b2b1 validation passed `python3
  ds4-parity/compare_decode_layer0_output_head.py --negative-test`, `python3
  ds4-parity/compare_decode_layer0_output_head.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer0_output_head.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer0-output-head-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-output-head`, B300 current-C oracle plus Rust candidate
  paired validation, pinned B300 artifact rerun, and B300 c2b2b2b2b2a
  layer-0 FFN-output rerun with 885 checks, local `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with 31 passed, 21
  skipped, and 0 failed, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, a touched-file NUL scan, and non-interactive Claude
  review with no blockers.
- M10.5c4c2b2b2b2b2b splits the remaining one-token scheduler item into a
  layer-0 output-head execution bridge and the next full scheduler/cache/logits
  slice M10.5c4c2b2b2b2b2b2. The split keeps the next commit comparable at the
  HC collapse and vocab-projection boundary before all 43 layers and cache
  transitions are introduced together.
- M10.5c4c2b2b2b2b2a adds `ds4-layer0-ffn-output-oracle-dump` and
  `ds4_dump_layer0_ffn_output_oracle_json`, which emit
  `ds4.layer0_ffn_output_oracle.v1` for token `0`, layer `0`, position `0`
  using the current-C model loader, config validation, weight binding, model
  fd/map bridge, the validated attention-output GPU prefix, FFN HC-pre, router
  selection, routed MoE, shared expert SwiGLU, shared down projection, and
  final FFN HC expansion.
- M10.5c4c2b2b2b2b2a adds `ds4-decode-layer0-ffn-output`, which maps the real
  GGUF on B300, binds DS4 weights, launches the attention-output prefix plus
  `hc_split_weighted_sum_norm`, `router_select`, `routed_moe_one`,
  `shared_gate_up_swiglu_q8_0`, and `shared_down_hc_expand_q8_0` through the
  safe Rust facade in one command batch, synchronizes, and reads back
  `after_attn_hc`, `ffn_cur`, `ffn_norm`, `router_logits`, `router_probs`,
  `router_selected`, `router_weights`, `routed_mid`, `routed_out`,
  `shared_mid`, `shared_out`, and `after_ffn_hc`.
- The B300 paired layer-0 FFN-output validator passed 819 checks before
  pinning and 885 checks after pinning exact weight metadata, router metadata,
  and full-buffer FNV digests: `after_attn_hc=ad09657ac6584898`,
  `ffn_cur=6a4fadf124b872b9`, `ffn_norm=51f4215200d2855c`,
  `router_logits=ea0d089c828257f3`, `router_probs=8435f2b23e429e02`,
  `router_selected=6028192a0e6c3c3e`,
  `router_weights=0a7ff588f5caa574`, `routed_mid=a51a0c8b6f39b89a`,
  `routed_out=507a5d29b2e806e9`, `shared_mid=8fb3b60df337c136`,
  `shared_out=3f90851fbe0be24c`, and `after_ffn_hc=3d49316c93ce351f`.
- M10.5c4c2b2b2b2b2a predecessor validation reran the B300
  c2b2b2b2b1 layer-0 attention-output paired check and passed 493 checks with
  the pinned `kv`, `raw_cache_row`, `heads`, `attn_low`, `attn_out`, and
  `after_attn_hc` digests unchanged.
- M10.5c4c2b2b2b2b2a validation passed `python3
  ds4-parity/compare_decode_layer0_ffn_output.py --negative-test`, `python3
  ds4-parity/compare_decode_layer0_ffn_output.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer0_ffn_output.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer0-ffn-output-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-ffn-output`, B300 current-C oracle plus Rust candidate
  paired validation, pinned B300 artifact rerun, B300 c2b2b2b2b1
  layer-0 attention-output rerun, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and
  non-interactive Claude review with no blockers.
- M10.5c4c2b2b2b2b2 splits the remaining one-token scheduler item into a
  layer-0 FFN-output execution bridge and the next full scheduler/logits slice
  M10.5c4c2b2b2b2b2b. The split keeps the next commit comparable at a tensor
  boundary before all 43 layers, compressed-cache transitions, and logits are
  introduced together.
- M10.5c4c2b2b2b2b1 splits the remaining one-token scheduler item into a
  layer-0 attention-output execution bridge and the next active full scheduler
  slice M10.5c4c2b2b2b2b2.
- M10.5c4c2b2b2b2b1 adds `ds4-layer0-attn-output-oracle-dump` and
  `ds4_dump_layer0_attn_output_oracle_json`, which emit
  `ds4.layer0_attn_output_oracle.v1` for token `0`, layer `0`, position `0`
  using the current-C model loader, config validation, weight binding, model
  fd/map bridge, GPU HC-pre prefix, QKV/RoPE, dense raw KV store, dense
  attention decode, inverse RoPE, low-rank attention output, and HC expansion.
- M10.5c4c2b2b2b2b1 adds `ds4-decode-layer0-attn-output`, which maps the real
  GGUF on B300, binds DS4 weights, launches the QKV/RoPE prefix plus
  `kv_fp8_store_raw`, `attention_decode_heads`, inverse `rope_tail`,
  `attention_output_low_q8`, and `matmul_q8_0_hc_expand` through the safe Rust
  facade in one command batch, synchronizes, and reads back `kv`,
  `raw_cache_row`, `heads`, `attn_low`, `attn_out`, and `after_attn_hc`.
- The B300 paired layer-0 attention-output validator passed 429 checks before
  pinning and 493 checks after pinning exact weight metadata, with exact FNV
  digests: `kv=92463977ae7f1b2e`,
  `raw_cache_row=bc56a173f5dd62cf`, `heads=4676767a2ee68e0c`,
  `attn_low=22e1bbf2f9236b99`, `attn_out=21c920ca48b6c7c3`, and
  `after_attn_hc=ad09657ac6584898`.
- M10.5c4c2b2b2b2b1 validation passed `python3
  ds4-parity/compare_decode_layer0_attn_output.py --negative-test`, `python3
  ds4-parity/compare_decode_layer0_attn_output.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer0_attn_output.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer0-attn-output-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-attn-output`, B300 current-C oracle plus Rust candidate
  paired validation, pinned B300 artifact rerun, B300 c2b2b2b2a layer-0
  QKV/RoPE rerun, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles`, `cargo test --workspace`, `cargo fmt --all
  -- --check`, `git diff --check`, touched-file NUL scan, and
  non-interactive Claude review with no blockers.
- M10.5c4c2b2b2b2a split the remaining one-token scheduler item into a
  layer-0 QKV/RoPE execution bridge and the next active full scheduler slice
  M10.5c4c2b2b2b2b.
- M10.5c4c2b2b2b2a adds `ds4-layer0-qkv-rope-oracle-dump` and
  `ds4_dump_layer0_qkv_rope_oracle_json`, which emit
  `ds4.layer0_qkv_rope_oracle.v1` for token `0`, layer `0`, position `0`
  using the current-C model loader, config validation, weight binding, model
  fd/map bridge, GPU HC-pre prefix, Q/KV projection, fused QKV RMS norm, dense
  Q projection, head RMS norm, and RoPE for dense `q` and `kv`.
- M10.5c4c2b2b2b2a adds `ds4-decode-layer0-qkv-rope`, which maps the real GGUF
  on B300, binds DS4 weights, launches the HC-pre prefix plus
  `matmul_q8_0`, `dsv4_qkv_rms_norm_rows`, `head_rms_norm`, and `rope_tail`
  through the safe Rust facade in one command batch, synchronizes, and reads
  back `attn_norm`, `qr`, `kv_raw`, `qr_norm`, `q`, and `kv`.
- The B300 paired QKV/RoPE validator passed 426 checks with exact FNV digests:
  `attn_norm=24e0d5fc736b2ace`, `qr=a04804371dd4a6ae`,
  `kv_raw=a1c0a2aae4fedc0e`, `qr_norm=059a86592fc239a9`,
  `q=a4af0bca4d611025`, and `kv=c18013c1fe8e0391`.
- M10.5c4c2b2b2b2a validation passed `python3
  ds4-parity/compare_decode_layer0_qkv_rope.py --negative-test`, `python3
  ds4-parity/compare_decode_layer0_qkv_rope.py`, `python3 -m py_compile
  ds4-parity/compare_decode_layer0_qkv_rope.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-layer0-qkv-rope-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-layer0-qkv-rope`, B300 current-C oracle plus Rust candidate
  paired validation, B300 c2b2b2b1 layer-0 HC-pre rerun, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`, and `cargo test
  --workspace`.
- M10.5c4c2b2b2b1 splits the remaining one-token scheduler item into a
  layer-0 attention HC-pre execution bridge and the next active full
  scheduler slice M10.5c4c2b2b2b2.
- M10.5c4c2b2b2b1 adds `ds4-layer0-attn-hc-pre-oracle-dump` and
  `ds4_dump_layer0_attn_hc_pre_oracle_json`, which emit
  `ds4.layer0_attn_hc_pre_oracle.v1` for token `0`, layer `0`, using the
  current-C model loader, config validation, weight binding, model fd/map
  bridge, and GPU tensor ABI calls for embedding, HC RMS norm, F16 matmul, and
  fused HC split/weighted-sum/attention-norm.
- M10.5c4c2b2b2b1 adds `ds4-decode-layer0-attn-hc-pre`, which maps the real
  GGUF on B300, binds DS4 weights, launches `embed_token_hc`,
  `rms_norm_plain`, `matmul_f16`, and `hc_split_weighted_sum_norm` through the
  safe Rust facade in one command batch, synchronizes, and reads back `cur_hc`,
  `flat_hc`, `hc_mix`, `hc_split`, `attn_cur`, and `attn_norm`.
- The B300 paired validator passed 346 checks with exact FNV digests:
  `cur_hc=f76512db41f80c4d`, `flat_hc=5abe5cafeb9fd15d`,
  `hc_mix=ea50bbe93ae96ca4`, `hc_split=d0f0c7dc02340820`,
  `attn_cur=110f29cd4090669f`, and `attn_norm=24e0d5fc736b2ace`.
- M10.5c4c2b2b2b1 validation passed `python3
  ds4-parity/compare_decode_layer0_attn_hc_pre.py --negative-test`, local
  `arch -arm64 make ds4-layer0-attn-hc-pre-oracle-dump`, local `cargo check
  -p ds4-gpu --bin ds4-decode-layer0-attn-hc-pre`, B300 current-C oracle plus
  Rust candidate paired validation, B300 c2b1 first-kernel rerun, B300
  c2b2b2a first-kernel current-C oracle rerun, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, and `git diff --check`.
- M10.5c4c2b2b2a splits the remaining one-token scheduler item before adding
  more Rust decode calls because M10.5c4c2b2b1 compared Rust first-kernel
  readback only against static pinned values. The new slice must compare the
  same B300 Rust `cur_hc` readback against a current-C oracle emitted through
  `model_open`, `config_validate_model`, `weights_bind`, `embed_token_f16`, and
  `hc_from_plain_embedding`.
- M10.5c4c2b2b2a adds `ds4-first-kernel-oracle-dump` and
  `ds4_dump_first_kernel_oracle_json`, which emit
  `ds4.first_kernel_oracle.v1` for token `0` using the current-C model loader,
  config validation, weight binding, F16 token embedding load, and HC
  broadcast.
- M10.5c4c2b2b2a adds a full-buffer FNV digest to the Rust
  `ds4.decode_first_kernel.v1` readback and
  `ds4-parity/compare_decode_first_kernel_oracle.py`, which pairs the current-C
  oracle and Rust candidate on B300. The paired B300 rerun passed 103 checks
  with exact `cur_hc` FNV `f76512db41f80c4d`, current-C SHA256
  `46e80d78c0dc648b773c230b37a4afd1446d7a3cc39f3a43a01a07d4ecf40dca`,
  16,384 nonzero f32 elements, and matching selected samples within the
  existing `1e-6` sample tolerance.
- M10.5c4c2b2b2a validation passed `python3
  ds4-parity/compare_decode_first_kernel_oracle.py --negative-test`, `python3
  ds4-parity/compare_decode_first_kernel.py --negative-test`, `python3 -m
  py_compile ds4-parity/compare_decode_first_kernel.py
  ds4-parity/compare_decode_first_kernel_oracle.py
  ds4-parity/run_parity_report.py`, local `arch -arm64 make
  ds4-first-kernel-oracle-dump`, local `cargo check -p ds4-gpu --bin
  ds4-decode-first-kernel`, B300 current-C oracle plus Rust candidate paired
  validation, B300 first-kernel candidate validation, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, touched-file
  NUL scan, and non-interactive Claude review with no blockers.
- M10.5c4c2b2b1 splits the remaining one-token scheduler item into a first
  real-kernel execution bridge and the next active full scheduler slice
  M10.5c4c2b2b2a.
- M10.5c4c2b2b1 adds `ds4-decode-first-kernel`, which maps the real GGUF on
  B300, binds DS4 weights, sets the model fd/range, allocates `cur_hc`, opens a
  command batch, launches `embed_token_hc` through the safe Rust facade for
  token `0`, synchronizes, reads back `cur_hc`, and releases the backend.
- The B300 first-kernel run emitted `ds4.decode_first_kernel.v1` and the
  candidate validator passed 64 checks. Evidence: model size 86,720,111,488
  bytes, tensor-data offset 5,333,824, 1,328 tensors, 43 bound layers,
  `base.token_embd` offset 77,928,033,088, `cur_hc` 65,536 bytes, 16,384
  nonzero f32 elements, and pinned samples at indices 0, 1, 8192, 16382, and
  16383.
- M10.5c4c2b2b1 adds `ds4-parity/compare_decode_first_kernel.py`, which checks
  the static contract and optionally validates the B300 JSON candidate. It is
  wired into the unified parity report as `M10.5c4c2b2b1 Rust first decode
  kernel comparator` with an exact B300 rerun command.
- M10.5c4c2b2b1 validation passed `cargo check -p ds4-gpu --bin
  ds4-decode-first-kernel`, `python3
  ds4-parity/compare_decode_first_kernel.py --negative-test`, `python3 -m
  py_compile ds4-parity/compare_decode_first_kernel.py
  ds4-parity/run_parity_report.py`, B300 `CUDA_ARCH=native cargo run -p
  ds4-gpu --features cuda-backend --bin ds4-decode-first-kernel --quiet --
  --model /workspace/ds4/ds4flash.gguf` followed by candidate validation,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, and `git diff --check`.
- M10.5c4c2b2a splits the remaining numeric B300 execution item into a full
  decode-state allocation bridge and the remaining one-token scheduler work.
- M10.5c4c2b2a adds `ds4-decode-state-alloc`, which walks the M10.5c2
  graph-state table for `ctx32768_mtp_off`, allocates every initially owned
  tensor, applies zero and negative-infinity fills, creates the planned
  `hc_pre`, `hc_post`, and `hc_comb` views, reports the allocation surface, and
  releases the backend with cleanup.
- The B300 state-allocation run emitted `ds4.decode_state_allocation.v1` and
  the candidate validator passed 63 checks. Evidence: 349 logical instances,
  272 initially owned allocations, 806,175,248 owned bytes, three views, one
  lazy owned tensor, one external input, 105 zero-full-capacity fills, 62
  zero-state fills, and 62 negative-infinity fills. Largest required
  allocations included `comp_mask`, `indexer_scores`, and layer-2
  `layer_attn_comp_cache`.
- M10.5c4c2b2a adds `ds4-parity/compare_decode_state_allocation.py`, which
  checks the static contract and optionally validates the B300 JSON candidate.
  It is wired into the unified parity report as `M10.5c4c2b2a Rust full decode
  state allocation comparator` with an exact B300 rerun command.
- M10.5c4c2b2a validation passed `cargo check -p ds4-gpu --bin
  ds4-decode-state-alloc`, `python3
  ds4-parity/compare_decode_state_allocation.py --negative-test`, `python3
  -m py_compile ds4-parity/compare_decode_state_allocation.py
  ds4-parity/run_parity_report.py`, B300 `CUDA_ARCH=native cargo run -p
  ds4-gpu --features cuda-backend --bin ds4-decode-state-alloc --quiet`
  followed by candidate validation, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles`, `cargo test --workspace`, `cargo fmt --all --
  --check`, and `git diff --check`.
- M10.5c4c2b1 splits the original B300 one-token execution item into a
  model-backed preflight and the remaining numeric decode slice. After the
  M10.5c4c2b2a state-allocation bridge and M10.5c4c2b2b1 first-kernel bridge,
  the active M10.5c4c2b2b2a slice must first anchor that first-kernel readback
  to a current-C oracle before the remaining one-token facade schedule compares
  M10.4 tensors/logits.
- M10.5c4c2b1 adds `rust/ds4-gpu/src/decode_execution.rs` and
  `ds4-decode-exec-preflight`, which mmap the real GGUF on B300, parse the
  GGUF header without copying tensor data, bind DS4 weights, set the model fd
  and tensor-data map range, allocate representative M10.4 checkpoint tensors,
  and exercise bounded model-range plus Q8/F16 cache hooks.
- The B300 preflight emitted `ds4.decode_execution_preflight.v1` and the
  candidate validator passed 69 checks. Evidence: model size
  86,720,111,488 bytes, tensor count 1,328, tensor-data offset 5,333,824,
  bound layers 43, selected layers `[0, 2, 3]`, representative tensors
  `cur_hc`, `logits`, `layer_raw_cache`, `layer_attn_comp_cache`, and
  `layer_index_comp_cache`, and cache hooks over 22 model ranges plus one
  Q8/F16 range.
- M10.5c4c2b1 adds `ds4-parity/compare_decode_execution_preflight.py`, which
  checks the static contract and optionally validates the B300 JSON candidate.
  It is wired into the unified parity report as `M10.5c4c2b1 Rust decode
  execution preflight comparator` with an exact B300 rerun command.
- M10.5c4c2b1 validation passed `cargo test -p ds4-gpu decode_execution
  --lib`, `cargo check -p ds4-gpu --bin ds4-decode-exec-preflight`, `python3
  ds4-parity/compare_decode_execution_preflight.py --negative-test`, `python3
  -m py_compile ds4-parity/compare_decode_execution_preflight.py
  ds4-parity/run_parity_report.py`, B300 `CUDA_ARCH=native cargo run -p
  ds4-gpu --features cuda-backend --bin ds4-decode-exec-preflight --quiet --
  --model /workspace/ds4/ds4flash.gguf --model-sha256
  efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
  followed by candidate validation, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles`, `cargo test --workspace`, `cargo fmt --all --
  --check`, and `git diff --check`.
- M10.5c4c2a adds safe Rust decode-backend wrappers for the model-map backend:
  `set_model_map`, `set_model_fd`, `set_model_map_range`,
  `cache_model_range`, and `cache_q8_f16_range`, with CUDA-only cache wrappers
  gated to Linux. Rust now rejects invalid model ranges before FFI because the
  CUDA backend accepts some invalid map ranges unless chunked-copy mode is
  active.
- M10.5c4c2a adds a B300-only `model_map_abi` smoke test covering fd, full map,
  map range, CUDA model cache, optional q8/f16 cache hook, and invalid-range
  failure paths against tiny deterministic model bytes.
- M10.5c4c2a adds `ds4-parity/compare_decode_model_map_bridge.py`, which checks
  sys ABI coverage, safe wrapper status bridges, Linux cfg containment, test
  coverage, and the unified B300 rerun command. It is wired into the unified
  parity report as `M10.5c4c2a Rust decode model-map bridge comparator`.
- M10.5c4c2a validation passed `python3
  ds4-parity/compare_decode_model_map_bridge.py --negative-test`, `python3 -m
  py_compile ds4-parity/compare_decode_model_map_bridge.py
  ds4-parity/run_parity_report.py`, B300 `CUDA_ARCH=native cargo test -p
  ds4-gpu --features cuda-backend --test model_map_abi -- --nocapture`, local
  `cargo test -p ds4-gpu model_map --lib`, local `cargo test -p ds4-gpu
  --features cuda-backend --test model_map_abi -- --nocapture`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, and `git diff --check`.
- M10.5c4c1 adds a feature-gated `ds4-gpu` `cuda-backend` build path that links
  `ds4.c` and `ds4_cuda.cu` into the Rust GPU crate on Linux only when the
  feature is requested, while preserving the existing macOS backend link path
  and keeping ordinary non-CUDA Linux builds no-link by default.
- M10.5c4c1 broadens the backend ABI smoke test to B300 Linux under
  `--features cuda-backend`, fixes the compiler helper so macOS `arch`
  wrapping is never used on Linux, and records the exact B300 source-refresh
  and smoke command in the unified parity report.
- M10.5c4c1 adds `ds4-parity/compare_b300_rust_backend_smoke.py`, which checks
  CUDA feature gating, C/CUDA source tracking, CUDA library links, Linux test
  cfg containment, and the unified B300 rerun command. It is wired into the
  unified parity report as `M10.5c4c1 Rust CUDA backend smoke contract`.
- M10.5c4c1 validation passed `python3
  ds4-parity/compare_b300_rust_backend_smoke.py --negative-test`, `python3 -m
  py_compile ds4-parity/compare_b300_rust_backend_smoke.py
  ds4-parity/run_parity_report.py`, B300 `CUDA_ARCH=native cargo test -p
  ds4-gpu --features cuda-backend --test backend_abi -- --nocapture`, local
  `cargo test -p ds4-gpu --features cuda-backend --test backend_abi --
  --nocapture`, `python3 ds4-parity/run_parity_report.py --skip-local-oracles`,
  `cargo test --workspace`, `cargo fmt --all -- --check`, and `git diff
  --check`.
- M10.5c4b adds `rust/ds4-gpu/src/decode_runtime.rs` and
  `ds4-decode-runtime-bridge`, a no-execute runtime-state bridge for
  `ctx32768_mtp_off` that resolves M10.5c2 graph-state handles, initial
  per-layer cache counters, M10.5c4a facade tensor arguments, and selected
  M10.5c1 weight-role slices before GPU execution.
- M10.5c4b records dense, ratio-4, ratio-128, and hash-layer weight presence
  expectations for base/layer roles, keeps routed MoE expert tensors explicitly
  in the weight source path, and preserves graph-state owned/view/lazy/external
  ownership without allocating backend tensors.
- M10.5c4b adds `ds4-parity/compare_decode_runtime_bridge.py`, which compares
  bridge handles to `ds4-graph-state-plan`, validates initial counters, checks
  every M10.5c4a trace facade tensor argument has a state or weight source, and
  verifies selected weight roles against the M10.5c1 structured weight table.
  It is wired into the unified parity report as `M10.5c4b Rust decode runtime
  bridge comparator`.
- M10.5c4b validation passed `cargo test -p ds4-gpu decode_runtime --lib`,
  `cargo run -p ds4-gpu --bin ds4-decode-runtime-bridge --quiet | python3 -m
  json.tool`, `python3 ds4-parity/compare_decode_runtime_bridge.py
  --negative-test`, `python3 -m py_compile
  ds4-parity/compare_decode_runtime_bridge.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and NUL scan
  over touched files.
- M10.5c4a adds `rust/ds4-gpu/src/decode_trace.rs` and
  `ds4-decode-trace`, a no-FFI dry-run execution trace that expands every
  M10.5b decode-plan case into default M10.5c3 facade calls, existing
  command/read/sync wrappers, per-layer stage markers, and per-layer
  raw/compressed/indexer cache-counter state events.
- M10.5c4a records default fused decode behavior before backend execution:
  dense layers use `attention_decode_heads`, ratio-4 layers switch to
  `attention_indexed_mixed_batch_heads` after the strict `> 512` indexer
  threshold, ratio-4/ratio-128 emit cases update compressed counters, split
  flush remains a token-level stage attached after layer 3, and no-logits
  cases omit output-head and read events.
- M10.5c4a adds `ds4-parity/compare_decode_trace.py`, which runs
  `ds4-decode-trace`, checks schema/cases, layer stage order, facade
  method/tensor-argument coverage from M10.5c3, command/read/sync markers,
  raw/compressed/indexer state transitions, and fails closed on summary,
  operation, split-flush, and state mutations. It is wired into the unified
  parity report as `M10.5c4a Rust decode trace comparator`.
- M10.5c4a validation passed `cargo test -p ds4-gpu decode_trace --lib`,
  `cargo run -p ds4-gpu --bin ds4-decode-trace --quiet -- --case
  ratio_emit_boundary | python3 -m json.tool`, `python3
  ds4-parity/compare_decode_trace.py --negative-test`, `python3 -m py_compile
  ds4-parity/compare_decode_trace.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and NUL scan
  over touched files. Non-interactive Claude review returned no blockers.
- M10.5c4 was split before implementation because the original item spans
  dry-run scheduling, runtime tensor/weight bridge construction, B300 numeric
  execution, continuation-state validation, and optional directional-steering
  coverage. M10.5c4c1 is the completed B300 Rust CUDA backend linkage and ABI
  smoke prerequisite for numeric one-token execution. M10.5c4c2a is the
  completed model-map backend bridge needed before real GGUF weight offsets are
  passed to CUDA kernels; the active next slice is M10.5c4c2b.
- M10.5c4b is the runtime-state bridge from M10.5c1/M10.5c2 weights and
  tensor plans. M10.5c4c was further split because the B300 pod had no default
  Rust toolchain and `ds4-gpu` only linked the C backend on macOS. M10.5c4c1
  creates the feature-gated Linux CUDA backend link/smoke path; M10.5c4c2 is
  the first B300 one-token execution/checkpoint slice, and M10.5c4d closes
  continuation and optional directional-steering coverage or records exact
  unavailable-fixture skips.
- M10.5c3 adds `rust/ds4-gpu/src/decode_backend.rs`, a safe facade over the
  default fused one-token decode backend primitives. The facade uses
  `TensorRef`/`TensorMut` handles from `Tensor` and `TensorView`, keeps raw
  `sys::ds4_gpu_*` calls behind the facade/lifecycle modules, and covers the
  C default decode operation list for embedding, QKV/norm, KV store,
  compressor/indexer, attention, router/MoE, hyper-connection, output head,
  command/read/view, and synchronize-on-failure.
- M10.5c3 adds `ModelMap` and `DecodeBackend` wrappers for model-map pointer
  and size threading. Optional compressed attention tensors are checked before
  FFI so nonzero compressed counts or mask flags cannot pass null pointers to
  `ds4_gpu_attention_decode_heads_tensor`.
- M10.5c3 adds `ds4-parity/compare_decode_backend_facade.py`, which verifies
  the facade operation table against the M10.2 operation oracle and M10.5a ABI
  declarations, checks tensor argument order from Rust method signatures,
  anchors existing command/read/view/sync wrappers, and fails closed on missing
  facade entry, tensor-order drift, and missing raw sys-call mutations. It is
  wired into the unified parity report as
  `M10.5c3 Rust decode backend facade comparator`.
- M10.5c3 validation passed `cargo test -p ds4-gpu decode_backend --lib`,
  `python3 ds4-parity/compare_decode_backend_facade.py --negative-test`,
  `python3 -m py_compile ds4-parity/compare_decode_backend_facade.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with `18 passed, 10
  skipped, 0 failed`, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, and NUL scan over touched files. Non-interactive Claude
  review returned no blockers; a follow-up review after the optional null
  guard also returned no blockers.
- M10.5c2 adds the no-execute decode graph tensor state plan in
  `rust/ds4-gpu/src/graph_state.rs`. It covers the decode-scope owner groups
  from M10.2: `GraphDecodeState`, `GraphPersistentKvState`,
  `GraphLayerWorkState`, and `GraphOptionalControlState`, while explicitly
  excluding speculative, MTP, and prefill owner groups for later work.
- M10.5c2 records C allocation-shape details needed before kernel execution:
  `hc_pre`/`hc_post`/`hc_comb` are views into `hc_split`, `ffn_out` is lazy
  optional, directional steering is an external input for this slice, raw and
  compressed persistent caches carry full-capacity zero-fill obligations for
  future checkpoint hashes, and compressor state tensors carry zero or
  negative-infinity initialization obligations.
- M10.5c2 adds `ds4-graph-state-plan`, a JSON dump for the graph-state plan,
  plus `ds4-parity/compare_graph_state_plan.py`. The comparator checks the
  `ctx32768_mtp_off` case against the M10.2 owner oracle, verifies excluded
  owners, summary counts, view geometry, lazy/external storage, selected
  raw/ratio-4/ratio-128 cache byte sizes, and fails closed on summary, field,
  and view mutations. It is wired into the unified parity report as
  `M10.5c2 Rust graph state comparator`.
- M10.5c2 validation passed `cargo test -p ds4-gpu graph_state --lib`,
  `python3 ds4-parity/compare_graph_state_plan.py --negative-test`,
  `python3 -m py_compile ds4-parity/compare_graph_state_plan.py
  ds4-parity/run_parity_report.py`, `cargo run -p ds4-gpu --bin
  ds4-graph-state-plan --quiet -- --case ctx32768_mtp_off | python3 -m
  json.tool`, `python3 ds4-parity/run_parity_report.py --skip-local-oracles`
  with `17 passed, 10 skipped, 0 failed`, `cargo test --workspace`, `cargo fmt
  --all -- --check`, `git diff --check`, and NUL scan over touched files.
  Non-interactive Claude review returned `NO BLOCKERS`.
- M10.5c1 adds structured Rust DS4 decode weight bindings in
  `rust/ds4-gguf/src/lib.rs`: `Ds4Weights` for base model weights and
  `Ds4LayerWeights` for every C `ds4_layer_weights` field. Required fields are
  stored as `TensorInfo`; dense/compressor/indexer/hash/optional-bias fields
  preserve `None` where the existing flat binding marks them absent.
- M10.5c1 changes `ds4-gguf-dump --validate-ds4-layout` to construct the
  structured base weight table first, emit a `weight_table` JSON section, and
  flatten that table back to the existing `bound_tensors` output. MTP bindings
  still use the existing flat MTP path and remain out of scope for this decode
  slice.
- M10.5c1 adds `ds4-parity/compare_rust_weight_table.py`, which builds the
  synthetic DS4 GGUF tensor directory used by the tensor-binding comparator,
  runs `ds4-gguf-dump`, checks base/layer field order against C
  `ds4_weights`/`ds4_layer_weights`, verifies the structured table flattens
  exactly to `bound_tensors`, and fails closed on removed-layer, removed-field,
  and presence-bit mutations. It is wired into the unified parity report as
  `M10.5c1 Rust structured weight table comparator`.
- M10.5c1 validation passed `cargo test -p ds4-gguf ds4_weight_table --lib`,
  `python3 ds4-parity/compare_rust_weight_table.py`, `python3
  ds4-parity/compare_rust_weight_table.py --negative-test`, `python3 -m
  py_compile ds4-parity/compare_rust_weight_table.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with `16 passed, 10
  skipped, 0 failed`, `arch -arm64 make ds4-metadata-dump` followed by
  `python3 ds4-parity/compare_tensor_bindings.py --negative-test`, `cargo
  test --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and NUL
  scan over touched files. Non-interactive Claude review returned
  `NO BLOCKERS`.
- M10.5b adds the no-execute Rust decode plan in
  `rust/ds4-gpu/src/decode_plan.rs`. It mirrors the default
  `metal_graph_eval_token_raw_swa` scheduling surface without backend calls:
  token-level stage order, layer profile stage order, one begin/end command
  pair, default split flush after layer 3, post-end logits read policy, raw SWA
  row/span/start math, DS4 layer compression counts, ratio-4 and ratio-128
  compressed-row counter transitions, and the strict `> 512` ratio-4 indexed
  attention threshold.
- M10.5b commits the current-C oracle at
  `ds4-parity/baselines/graph/m10.5b/decode-plan-oracle.json`, covering
  first-token, short-prefill decode, ratio-boundary emission, long indexed
  decode, and no-logits/no-split cases under the same default fusion and
  directional-steering-disabled assumptions used by M10.4.
- M10.5b adds `ds4-parity/compare_decode_plan_rust.py`, which compares the
  Rust source constants against the JSON oracle and fails closed on in-memory
  stage-order, raw-start, and indexed-layer-count mutations. It is wired into
  the unified parity report as `M10.5b Rust decode plan comparator`.
- M10.5b validation passed `cargo test -p ds4-gpu decode_plan --lib`,
  `python3 ds4-parity/compare_decode_plan_rust.py`, `python3
  ds4-parity/compare_decode_plan_rust.py --negative-test`, `python3 -m
  json.tool ds4-parity/baselines/graph/m10.5b/decode-plan-oracle.json`,
  `python3 -m py_compile ds4-parity/compare_decode_plan_rust.py
  ds4-parity/run_parity_report.py`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` with `15 passed, 10 skipped, 0 failed`, `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, NUL scan over
  touched files, and non-interactive Claude review with `NO BLOCKERS`.
- M10.5a splits the broad one-token decode scheduler step into reviewable
  ABI-surface, call-order/state-plan, and backend-execution stages before
  M10.6 prefill. The completed ABI slice exposes all 81 M10.2 graph backend
  primitives in `rust/ds4-gpu-sys/src/lib.rs`, adding 57 declarations beyond
  the existing lifecycle/tensor/command/model-map surface.
- M10.5a adds `ds4-parity/compare_gpu_sys_abi.py`, which compares the M10.2
  operation oracle and current `ds4_gpu.h` signatures against
  `ds4-gpu-sys`. The comparator checks every required operation's return type
  and parameter ABI type sequence, and its negative test catches a missing Rust
  declaration plus Rust-side and C-header parameter type drift.
- M10.5a is wired into the unified parity report as
  `M10.5a Rust GPU sys ABI comparator` and documented in
  `ds4-parity/README.md`.
- M10.5a validation passed `python3 ds4-parity/compare_gpu_sys_abi.py`,
  `python3 ds4-parity/compare_gpu_sys_abi.py --negative-test`,
  `python3 -m py_compile ds4-parity/compare_gpu_sys_abi.py
  ds4-parity/run_parity_report.py`, `cargo test --workspace`, `python3
  ds4-parity/run_parity_report.py --skip-local-oracles` with `14 passed, 10
  skipped, 0 failed`, `cargo fmt --all -- --check`, `git diff --check`, NUL
  scan over touched files, and non-interactive Claude review with
  `NO BLOCKERS`.
- M10.4 adds `ds4-graph-checkpoint-dump` plus
  `ds4_dump_graph_checkpoint_oracle_json`, a current-C graph checkpoint dump
  path that exercises the normal C graph backend without changing production
  scheduling. The B300 baseline is committed at
  `ds4-parity/baselines/graph/m10.4/current-c.json` with manifest
  `ds4-parity/baselines/graph/m10.4/manifest.json`.
- M10.4 B300 capture used pod `ds4-rust-port-b300` on node
  `c1v17-b300n1-nic1`, backend `cuda`, ctx `32768`, model
  `/workspace/ds4/ds4flash.gguf` SHA256
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  short prompt `short_italian_fact` with 21 tokens, and long prompt
  `long_memory_archive` with 3353 tokens.
- M10.4 records 10 checkpoints: short layer-major prefill logits, one-token
  decode logits, layer-2 attention/index compressed KV after decode, long
  chunked prefill logits and layer-2 compressed KV, and cache-continuation
  resumed prefill logits and layer-2 compressed KV. Eight checkpoints compare
  exact SHA256; two long-context logits checkpoints compare selected f32
  samples with recorded tolerance. MTP verifier capture is explicitly skipped
  because no support MTP model was provided in the B300 capture environment.
- M10.4 validation passed `arch -arm64 make ds4-graph-checkpoint-dump`, B300
  `make ds4-graph-checkpoint-dump CUDA_ARCH=native`, B300 capture plus
  `python3 ds4-parity/check_graph_checkpoint_oracle.py ... --negative-test`,
  copied artifact JSON syntax checks, local
  `python3 ds4-parity/check_graph_checkpoint_oracle.py --negative-test`, and
  the unified parity report comparator row. Non-interactive Claude review
  returned `NO BLOCKERS`.
- M10.3 adds the model-execution-neutral Rust graph surface in
  `rust/ds4-gpu/src/graph_plan.rs`: fixed DS4 graph constants, context/raw
  cap/compression plan math, the four M10.2 plan cases, backend facade targets
  for all 81 `ds4_gpu.h` operations, owner entries for all 113
  `ds4_gpu_graph` tensor fields, byte-size formulas for graph tensor classes,
  and command-boundary records for the 15 M10.2 boundaries.
- M10.3 adds `ds4-parity/compare_graph_plan_rust.py`, which compares the Rust
  graph inventory against the M10.2 JSON oracle and fails closed on synthetic
  missing-operation, missing-tensor, missing-command-boundary, and missing-plan
  mutations. It is wired into `ds4-parity/run_parity_report.py` and documented
  in `ds4-parity/README.md`.
- M10.3 validation passed targeted `cargo test -p ds4-gpu graph_plan --lib`,
  `python3 ds4-parity/compare_graph_plan_rust.py`, `python3
  ds4-parity/compare_graph_plan_rust.py --negative-test`, Python syntax check,
  the unified parity report comparator row, full `cargo test --workspace`,
  `cargo fmt --all -- --check`, `git diff --check`, NUL scan over touched
  files, and non-interactive Claude review with `NO BLOCKERS`.
- M10.2 adds `ds4-parity/check_graph_plan_inventory.py` and the committed
  source-derived oracle
  `ds4-parity/baselines/graph/m10.2/graph-plan-inventory.json`. The checker
  compares the oracle against `ds4_gpu.h`, `ds4.c` graph tensor fields, fixed
  DS4 model constants, compression-ratio source, context-cap calculations, and
  command-buffer boundary functions.
- M10.2 records 81 `ds4_gpu.h` operations with named Rust facade targets, 113
  `ds4_gpu_graph` tensor fields with owner groups, graph plan cases for
  ctx=128, ctx=2048, and ctx=32768 with MTP off/on, and 15 command-boundary
  records covering decode, chunked/layer-major prefill, MTP draft/verifier,
  profile split helpers, and speculative frontier copy/restore/commit.
- M10.2 validation passed `python3
  ds4-parity/check_graph_plan_inventory.py`, `python3
  ds4-parity/check_graph_plan_inventory.py --negative-test`, JSON report smoke,
  Python syntax checks, the unified parity report comparator row, `git
  diff --check`, NUL scan over touched files, and non-interactive Claude review
  with `NO BLOCKERS`.
- M10.1 split broad runtime graph orchestration into comparator-first work
  items M10.2 through M10.9: backend operation inventory/graph plan oracle,
  Rust backend trait and graph-plan surface, current-C intermediate tensor
  checkpoint oracle, Rust single-token decode scheduling, Rust layer-major and
  chunked prefill, Rust graph session state and payload parity, Rust MTP
  draft/verifier orchestration, and end-to-end/benchmark closure.
- M10.1 source review tied the split to current C graph paths:
  `ds4_gpu_graph`, `metal_graph_alloc_raw_cap`,
  `metal_graph_encode_layer_attention_batch`,
  `metal_graph_encode_layer_ffn_batch`, `metal_graph_eval_token_raw_swa`,
  `metal_graph_prefill_layer_major`, `metal_graph_prefill_chunked_range`,
  `metal_graph_verify_decode2_exact`, `metal_graph_eval_mtp_draft`, and
  speculative frontier snapshot/restore/commit helpers, plus backend primitive
  groups in `ds4_gpu.h`.
- M10.1 validation passed roadmap/TODO diff inspection, `git diff --check`, NUL
  scan over touched files, and non-interactive Claude review with
  `NO BLOCKERS`. Claude confirmed all named source paths exist, dependency
  ordering is measurable, and no M10 graph responsibility is unassigned.
- M9.9 adds `ds4-parity/run_server_parity_report.py`, wires it into
  `ds4-parity/run_parity_report.py`, documents it in `ds4-parity/README.md`,
  and adds `ds4-parity/check_runtime_kv_replay_summary.py` so the M9.8f5 B300
  runtime replay summary is checked structurally rather than only as JSON.
- M9.9 local report coverage includes Rust runtime route/cache tests,
  `server_chat`, `server_response`, `server_http`, `server_no_model`, the
  `no_model_server` socket replay, `compare_server_kv.py`, the
  `compare_server_kv.py --negative-test` mutation checks,
  `compare_kv_replay.py --negative-test`, and the structural M9.8f5 KV replay
  summary checker. Model-backed B300 rows remain SKIP, but include explicit
  temp-kubeconfig/context rerun commands for M0.4 server replay, M0.5
  three-lifetime KV replay, and `ds4_test --server`.
- M9.9 guards filtered cargo-test rows against false positives by parsing the
  `test result: ok. N passed` line and failing any model-free row that matches
  zero tests or has no parseable pass count. The B300 M0.4/M0.5 rerun commands
  install cleanup traps after spawning servers and avoid top-level `|| true`
  fallthrough around replay failures.
- M9.9 validation passed `python3
  ds4-parity/check_runtime_kv_replay_summary.py`, a direct pass-count parser
  smoke, `python3 ds4-parity/run_server_parity_report.py` with `10 passed, 3
  skipped, 0 failed`, `python3 ds4-parity/run_server_parity_report.py --json`
  with `ok=true`, `python3 ds4-parity/run_parity_report.py` with `15 passed, 5
  skipped, 0 failed`, `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, Python syntax checks, NUL scan over touched files,
  cleanup of generated `ds4-session-payload-dump`, and non-interactive Claude
  review with `NO BLOCKERS`.
- M9.8f5 closes the M9.8 runtime cache/KV path against the committed M0.5
  current-C artifacts with an exact three-lifetime Rust runtime replay on B300.
  Evidence is recorded in
  `ds4-parity/baselines/kv/m9.8f5/runtime-rust-b300-summary.json`.
- M9.8f5 B300 replay used the Rust runtime binary at commit
  `b3078f6c0b3389073050dd450cda6ec7325c146b`, pod
  `ds4-rust-port-b300`, model
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
  and the M0.5 flags: `--ctx 32768`, `--tokens 16`, disk budget `512 MiB`,
  min tokens `512`, cold max `30000`, and continued interval `0`.
- M9.8f5 replay matched the M0.5 current-C cache behavior: `seed_miss`
  produced content `I notice`, `cached_tokens=0`, `cache_write_tokens=550`,
  and `cache_source=none`; `seed_restore` restored
  `0ab2314538b11686a11e296b7f697651fbd17e60.kv` with
  `cached_tokens=550`, `cache_write_tokens=0`, and
  `cache_source=disk-text`; `continuation_restore` restored
  `a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv` with
  `cached_tokens=552`, `cache_write_tokens=9`, `cache_source=disk-text`, and
  generated `kv continued`.
- M9.8f5 replay matched the M0.5 KVC header rows exactly for file names,
  reasons, token counts, hit counts, context length, payload bytes, rendered
  text bytes, and file sizes for `0ab231...` cold, `a0cac...` shutdown, and
  `4f149...` shutdown checkpoints.
- M9.8f5 local validation passed `python3 ds4-parity/compare_server_kv.py`,
  `python3 ds4-parity/compare_server_kv.py --negative-test`, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, JSON validation for the
  M9.8f5 replay summary, full `cargo test --workspace`, `cargo fmt --all --
  --check`, `git diff --check`, NUL scan over touched files, and
  non-interactive Claude review with `NO BLOCKERS`.
- M9.8f4 adds the runtime store side of disk KV: Rust exposes C
  `ds4_kvstore_store_live_prefix`, `ds4_kvstore_maybe_store_continued`,
  continued-frontier note/suppress/restore helpers, C chat-anchor/store-length
  helpers, and explicit prompt-sync/decode split points through `ds4-engine`.
- M9.8f4 wires the Rust OpenAI Chat runtime to match C's request path:
  live cache misses persist the current checkpoint before disk replacement,
  cold prompts sync/store either the stable chat anchor or full prompt,
  continued checkpoints are attempted after prefill and during decode until a
  tool-call DSML block begins, and SIGINT/SIGTERM now drive a graceful server
  return so shutdown checkpoints are written.
- M9.8f4 writes tool-map trailers for stored KVC files by preserving sampled raw
  DSML from generated tool calls, mapping assigned OpenAI tool-call ids into
  `ToolMemory`, and feeding `write_tool_map_trailer` through C trailer hooks
  with server-compatible block ordering.
- M9.8f4 B300 smoke used temp kubeconfig
  `/tmp/ds4-hou2-prod1.kubeconfig`, pod `ds4-rust-port-b300`, model
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
  and temp artifacts under `/tmp/ds4-m98f4-smoke` plus
  `/tmp/ds4-m98f4-evict`. The first request wrote a cold KVC
  `1a9fd13c3ac5bebd1f3203ca09207324c526659b.kv` with reason `1`, 923 tokens,
  quant `2`, 36,651,000 payload bytes, and 3,030 rendered text bytes; graceful
  shutdown wrote a reason `4` KVC with 924 tokens; a fresh server restored the
  cold KVC with `cache_source: disk-text` and `disk_cached_tokens: 923`.
- M9.8f4 protected-eviction B300 smoke used a 64 MiB budget and left exactly
  one just-written shutdown KVC after termination:
  `0793c50a7cf4d0586e220083df58efd9e89b6184.kv`, reason `4`, 1,103 tokens,
  39,111,880 payload bytes, proving eviction did not delete the protected
  checkpoint.
- M9.8f4 validation passed targeted `cargo test -p ds4-engine --bin
  ds4-server-runtime-rs -- --nocapture`, targeted `cargo test -p ds4-gguf
  tool_memory::tests:: -- --nocapture`, `python3
  ds4-parity/compare_kv_policy.py --negative-test`, `python3
  ds4-parity/compare_kvc_file.py --negative-test`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, and the B300
  smokes above. Non-interactive Claude review returned `NO BLOCKERS`.
- M9.8f3 links `ds4_kvstore.c` into `ds4-engine` and adds Rust FFI for
  `ds4_kvstore_open`, `ds4_kvstore_try_load_text`, and
  `ds4_kvstore_load_result_free`, preserving the C `ds4_kvstore`,
  `ds4_kvstore_entry`, options, trailer-hooks, and load-result layouts.
- M9.8f3 wires OpenAI Chat runtime disk restore before generation for empty
  sessions: Rust opens the C KV store, probes live cache state, restores a
  rendered-text prefix KVC payload with C session-load logic, and generates
  from the C-built effective prompt containing the exact loaded token prefix
  plus a newly tokenized visible suffix.
- M9.8f3 intentionally guards disk restore to empty Rust sessions until the
  store milestone lands, because C persists the current live checkpoint before
  replacing it with an older disk snapshot. Claude initially flagged the
  missing live-checkpoint persist as a blocker; the final reviewed code skips
  disk replacement when `live_tokens_before > 0`.
- M9.8f3 restores disk KVC tool-map trailers before prompt rendering, reuses
  the existing `ToolMemory` exact sampled-DSML path, and fixes OpenAI Chat
  parsing so assistant `tool_calls[].id` and tool-message `tool_call_id` survive
  into `ChatMessage`.
- M9.8f3 B300 smoke used temp kubeconfig
  `/tmp/ds4-hou2-prod1.kubeconfig`, pod `ds4-rust-port-b300`, model
  `/workspace/ds4/ds4flash.gguf`, and temp artifacts under
  `/tmp/ds4-m98f3-smoke`. The smoke passed after building
  `target/debug/ds4-server-runtime-rs` on the pod with a temporary Rust toolchain
  in `/tmp`: one C-seeded Rust disk hit restored 550 tokens from
  `/tmp/ds4-m98f3-smoke/kv/0ab2314538b11686a11e296b7f697651fbd17e60.kv`, one
  unrelated Rust request missed with `cache_source: none`, and one synthetic KVC
  tool-map request reported `tool_replay: mem=0 disk=1 canonical=0
  missing_ids=0` with `pwd sampled` in the rendered prompt and no canonical
  fallback in that rendered prompt.
- M9.8f3 validation passed targeted `cargo test -p ds4-engine --bin
  ds4-server-runtime-rs -- --nocapture`, `cargo build -p ds4-engine --bin
  ds4-server-runtime-rs`, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, NUL scan
  over touched files, the B300 smoke above, and two non-interactive Claude
  reviews with `NO BLOCKERS`.
- M9.8f2 adds the Rust runtime server cache configuration surface for C's
  disk-KV flags: cache directory, disk budget, min/cold/continued/boundary
  policy options, cross-quant rejection, exact DSML tool replay disablement,
  and tool-memory ID limit.
- M9.8f2 keeps this stage model-execution-neutral: runtime trace decisions
  report none/memory-token/disk-text cache contract rows, including
  `tool_replay`, `disk_cached_tokens`, and optional `disk_cache_file`, but disk
  KVC lookup/loading/writing remains split into M9.8f3 and M9.8f4.
- M9.8f2 validation passed targeted `cargo test -p ds4-engine --bin
  ds4-server-runtime-rs -- --nocapture`, `python3
  ds4-parity/compare_kv_replay.py --negative-test`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, NUL scan
  over touched Rust files, and non-interactive Claude review with no blockers.
- M9.8f1 split the broad runtime cache/KV request-path item into four
  implementation stages plus a B300 comparator-closure stage:
  M9.8f2 runtime cache configuration and trace contract, M9.8f3 disk-KV lookup
  and payload restore, M9.8f4 KV store/continued frontier/eviction, and M9.8f5
  end-to-end replay closure.
- M9.8f1 source-mapped the remaining work to C helpers for
  `kv_cache_try_load_text`, session payload restore, exact-prefix suffix prompt
  construction, `kv_cache_store_current`, continued-store policy, tool-map
  replay, and B300 artifact comparison.
- M9.8f1 validation passed `git diff --check` and non-interactive Claude review
  with no blockers.
- M9.8e adds Rust restore plumbing for KVC tool-map trailers: it collects
  wanted tool-call ids from assistant calls and tool-result ids, decodes a KVC
  trailer only when `EXT_TOOL_MAP` is set, restores matching entries into
  `ToolMemory` as disk-sourced DSML, and keeps C-equivalent partial entries
  from malformed trailers.
- M9.8e tests cover wanted-id filtering, disk replay stats, restore-before-
  prompt-render ordering, canonical fallback for missing ids, and partial
  restore from a truncated second trailer entry.
- M9.8e revalidated the existing C/Rust trailer comparator:
  `python3 ds4-parity/compare_kv_trailer.py --negative-test` passed 432
  positive checks and 8 negative checks.
- M9.8e validation passed targeted `cargo test -p ds4-gguf tool_memory --
  --nocapture`, targeted `cargo test -p ds4-gguf tool_map -- --nocapture`,
  `cargo test -p ds4-gguf --bin ds4-kv-trailer-dump-rs`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `git diff --check`, NUL scan
  over touched Rust files, and non-interactive Claude review with no blockers.
- M9.8d adds Rust policy helpers for C's continued-checkpoint bookkeeping:
  `note_store`, `suppress_continued_store`, and
  `restore_suppressed_continued`. These preserve the C rule that a cold store
  on a continued frontier can suppress the duplicate continued write and then
  restore the prior frontier if the cold write fails.
- M9.8d tests cover note-only-increases behavior, duplicate continued-boundary
  suppression, restored frontier re-enabling, no-op suppression for already
  stored or unaligned frontiers, and no-op restore for non-suppressed states.
- M9.8d also revalidated the existing no-model KV policy comparator against
  the current-C M7.2 oracle: `check_kv_policy_dump.py --negative-test` and
  `compare_kv_policy.py --negative-test` both passed.
- M9.8d validation passed targeted `cargo test -p ds4-gguf kv_policy --
  --nocapture`, full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, NUL scan over touched Rust files, and non-interactive
  Claude review with no blockers.
- M9.8c extends Rust live continuation state with live-token frontiers,
  Responses visible transcript keys, Anthropic call-id frontiers, and
  tool-less thinking visible transcript keys.
- M9.8c adds model-free continuation planners for C cache sources
  `responses-visible`, `responses-tool-output`, `anthropic-tool-output`, and
  `thinking-visible`. Direct tool-output plans require exact call-id set and
  live frontier matches; visible-prefix plans require a strict byte-prefix
  extension before returning the suffix that must be tokenized after the live
  sampled prefix.
- M9.8c tests cover same-id-set matching independent of order, frontier
  mismatch rejection, Responses visible replay with and without matched IDs,
  parsed Responses/Anthropic tool-output suffix planning, and thinking-visible
  rejection for non-chat or Responses requests.
- M9.8c validation passed targeted `cargo test -p ds4-gguf server_chat --
  --nocapture`, full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, NUL scan over touched Rust files, and non-interactive
  Claude review with no blockers.
- M9.8b adds a Rust `ToolMemory` core that stores exact sampled DSML blocks by
  tool-call id, tracks RAM versus disk source for replay stats, upgrades disk
  entries to RAM on matching sampled replay, and prunes least-recently-used ids
  under configured entry/byte limits.
- M9.8b `ToolMemory::attach_to_messages` mirrors C
  `tool_memory_attach_to_messages`: it only attaches raw DSML when every tool
  call id in a message resolves to the same sampled DSML block, otherwise it
  records canonical fallback and missing-id counts without mutating the message.
- M9.8b prompt-rendering tests cover OpenAI/Responses sampled DSML replay,
  Anthropic sampled DSML replay, missing-id canonical fallback, split-block
  canonical fallback, disk source stats, LRU pruning after lookup touch, and
  disk-to-RAM source upgrade without duplicate entries.
- M9.8b validation passed targeted `cargo test -p ds4-gguf tool_memory --
  --nocapture`, full `cargo test --workspace`, `cargo fmt --all -- --check`,
  `git diff --check`, NUL scan over touched Rust files, and non-interactive
  Claude review with no blockers.
- M9.8a splits the broad server cache, KV restore, continued-frontier, eviction,
  and tool-memory item into separately reviewable stages: tool-memory replay
  core (M9.8b), live continuation/visible-prefix state (M9.8c), disk-KV policy
  completion (M9.8d), KV tool-map trailer restore (M9.8e), and runtime
  cache/KV replay integration with B300 validation (M9.8f).
- M9.8a is docs/state-only and keeps model-backed B300 replay validation in the
  runtime integration slice instead of hiding it under model-free policy or
  formatter work.
- M9.8a validation passed `git diff --check`, docs inspection, and
  non-interactive Claude review with no blockers.
- M9.7b adds exported model-free SSE formatters and HTTP wrappers for
  Responses and Anthropic protocol streams:
  `format_responses_stream_sse`, `format_responses_stream_http`,
  `format_anthropic_message_stream_sse`, and
  `format_anthropic_message_stream_http`, plus `ResponsesStreamResponse`.
- M9.7b Responses streaming emits C-shaped `data:` events with monotonic
  `sequence_number` insertion after each `type` field, covering
  `response.created`, reasoning summary lifecycle events, message content-part
  lifecycle events, function-call argument delta/done events, tool-search
  output items that skip argument events, and terminal
  `response.completed`/`response.incomplete`/`response.failed` payloads.
- M9.7b Responses streaming keeps C's reasoning replay rule: a reasoning item
  that did not close naturally is marked `incomplete` both in
  `response.output_item.done` and in the terminal response output even when the
  response-level finish maps to `completed`.
- M9.7b Anthropic streaming emits `event:`-framed message/content-block
  lifecycles for thinking, text, tool-use input JSON deltas, terminal
  `message_delta`, and `message_stop`; thinking blocks include the message ID
  as the signature delta, tool-use deltas carry normalized JSON fragments, and
  reasoning-only streams add the empty text block required by the C path.
- M9.7b local validation passed targeted `cargo test -p ds4-gguf
  server_response -- --nocapture` with 30 server-response tests, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and a NUL scan over the touched Rust files; this slice is model-free, so B300
  execution was not required.
- M9.7b non-interactive Claude review returned `NO BLOCKERS` for sequence
  numbers, event order, reasoning incomplete semantics, function/tool-search
  argument events, Anthropic content-block lifecycle, usage fields, and exported
  API shape.
- M9.7a adds exported Rust final-response formatters for Responses and
  Anthropic non-streaming protocol bodies plus HTTP wrappers:
  `format_responses_final_response_json`,
  `format_responses_final_response_http`, `format_anthropic_message_json`, and
  `format_anthropic_message_http`.
- M9.7a Responses formatting mirrors C `responses_final_response`: injected
  response/reasoning/message/tool IDs and timestamps for deterministic tests,
  `completed`/`incomplete`/`failed` response status mapping, item status
  mapping, `max_tokens` incomplete details, server-error body, output text,
  optional reasoning summary emission, function-call output items, and cache
  usage fields with clamped `cached_tokens` and `cache_write_tokens`.
- M9.7a Responses tool output preserves C's protocol-specific tool identity
  rules: namespace tool calls emit the original wire name plus `namespace`,
  hosted `tool_search` calls emit `tool_search_call` with object arguments and
  `execution:"client"`, while plain user functions named `tool_search` remain
  `function_call` when their `ToolSchemaOrder` is not marked as a hosted
  Responses tool search.
- M9.7a Anthropic formatting mirrors C `anthropic_final_response` and
  `append_anthropic_content`: thinking blocks precede text/tool blocks,
  generated tool-use IDs default to `toolu_<message-id>_<index>`, tool
  arguments are normalized object JSON, empty content still emits an empty text
  block, reasoning-only content emits thinking followed by empty text, and
  `tool_calls`/`length`/other finishes map to `tool_use`/`max_tokens`/
  `end_turn`.
- M9.7a local validation passed targeted `cargo test -p ds4-gguf
  server_response -- --nocapture`, full `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`; this slice is
  model-free, so B300 execution was not required.
- M9.7a non-interactive Claude review returned `NO BLOCKERS`; the only
  non-blocking note was that Responses `output_tokens_details.reasoning_tokens`
  remains hard-coded to `0`, matching C `append_responses_usage_json`.
- M9.6d added `ds4-parity/run_tool_call_quality.py`, a documented Rust
  server-runtime equivalent runner for the C `./ds4_test --tool-call-quality`
  surface.
- M9.6d runner launches separate fast and exact/`--quality`
  `ds4-server-runtime-rs` cases with distinct ports, model path, backend,
  context size, trace path, and command list recorded in `summary.json`.
- M9.6d runner sends a compact OpenAI tool-call request with `temperature=0`,
  `seed=123`, `max_tokens=256`, and `stream=false`; `summary.json` records the
  shared C/Rust OpenAI defaults `top_k=0`, `top_p=1.0`, and `min_p=0.05`.
- M9.6d classifier reports structural categories for HTTP errors, malformed
  JSON, missing choice/message/tool/function payloads, wrong tool name, invalid
  or wrong arguments, wrong finish reason, and `ok`, and self-tests exercise
  every category plus the wire request seed control.
- M9.6d preserves per-case request, response, headers, trace, stdout, and
  stderr files under the output directory, and writes `summary.json` plus
  `summary.txt` for pass/fail category comparison.
- M9.6d local checks passed for `python3 -m py_compile
  ds4-parity/run_tool_call_quality.py`, `python3
  ds4-parity/run_tool_call_quality.py --self-test`, `ruff format --check
  ds4-parity/run_tool_call_quality.py`, full `cargo test --workspace`,
  `cargo fmt --all -- --check`, and `git diff --check`.
- M9.6d B300 run used pod `ds4-rust-port-b300` in `hou2-prod1`, snapshot
  `/workspace/ds4-m96d`, and model `/workspace/ds4/ds4flash.gguf`; the command
  was `python3 ds4-parity/run_tool_call_quality.py --server-bin
  target/debug/ds4-server-runtime-rs --model /workspace/ds4/ds4flash.gguf
  --backend cuda --out-dir /tmp/ds4-m96d-tool-call-quality --ready-timeout
  360`.
- M9.6d B300 `summary.json` reported fast and exact cases both passed with
  category `ok`, HTTP 200, tool `list_files`, arguments `{"path":"."}`, and
  finish `tool_calls`; artifacts remain under
  `/tmp/ds4-m96d-tool-call-quality/fast` and
  `/tmp/ds4-m96d-tool-call-quality/exact`.
- M9.6d B300 fast and exact traces both recorded `stream: 0`, `tools: 1`,
  `max_tokens: 256`, `temperature: 0.000`, `top_k: 0`, `top_p: 1.000`,
  `min_p: 0.050`, `seed: 123`, `generated_tokens: 42`, DSML start/end,
  finish `tool_calls`, and parsed tool call `list_files` with arguments
  `{"path": "."}`.
- M9.6d B300 runtime processes were stopped after each run; `pgrep -af
  ds4-server-runtime-rs` found no leftover runtime process beyond the check
  shell.
- M9.6d Claude review initially caught incomplete classifier self-test
  coverage; after adding all missing category cases and rerunning local and
  B300 checks, the final Claude review returned no blockers.
- M9.6c3 routes supported streaming OpenAI tool chat requests through the Rust
  server runtime instead of rejecting them, while keeping non-tool streaming on
  the existing `OpenAiChatStream` path and preserving thinking/stop-list
  unsupported boundaries.
- M9.6c3 feeds runtime `token_texts` through
  `OpenAiToolCallStreamTranslator`, falls back to full parsed tool-call deltas
  when no live events are emitted, emits parsed assistant content before tool
  deltas when present, and uses `format_openai_chat_tool_stream_http` for role,
  tool start, argument fragments, finish, optional usage, and `[DONE]`.
- M9.6c3 extends the tool-stream formatter with a `Content` event and reuses
  the existing content chunk JSON helper so non-tool streaming remains
  byte-compatible with the prior M9.5 path.
- M9.6c3 local checks passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.6c3 B300 checks used pod `ds4-rust-port-b300` in `hou2-prod1`, snapshot
  `/workspace/ds4-m96c3`, and model `/workspace/ds4/ds4flash.gguf`; targeted
  runtime and formatter tests passed in the pod, then
  `cargo build -p ds4-engine --bin ds4-server-runtime-rs` produced the live
  runtime binary.
- M9.6c3 B300 live replay used the raw M9.6b `chat_tool_call` request with
  `"stream":true` and `include_usage`; SSE artifacts
  `/tmp/ds4-m96c3-chat_tool_call_stream.sse` and
  `/tmp/ds4-m96c3-chat_tool_call_stream.headers` parsed as role assistant,
  tool start for `list_files`, argument fragments reassembling to
  `{"path":"."}`, finish `tool_calls`, usage `394/42/436` with cache `0/394`,
  and `[DONE]`.
- M9.6c3 B300 trace `/tmp/ds4-m96c3-server.trace` recorded `stream: 1`,
  `tools: 1`, `stream_include_usage: 1`, `prompt_tokens: 394`,
  `generated_tokens: 42`, `finish: tool_calls`, `dsml_start: 1`,
  `dsml_end: 1`, and parsed `tool_call[0]` name `list_files` with arguments
  `{"path": "."}`; stderr/stdout artifacts remain at
  `/tmp/ds4-m96c3-server.stderr` and `/tmp/ds4-m96c3-server.stdout`.
- M9.6c3 B300 runtime process was stopped after replay; `pgrep -af
  ds4-server-runtime-rs` found no leftover runtime process beyond the check
  shell.
- M9.6c3 Claude review returned no blockers after checking runtime routing,
  unsupported-route boundaries, SSE shape and ordering, fallback IDs,
  content-before-tool ordering, translator input fallback, unit coverage, and
  the B300 replay evidence.
- M9.6c2 added a model-free `OpenAiToolCallStreamTranslator` that consumes
  generated DSML byte chunks and emits owned M9.6c1 tool-call stream events
  without model/runtime routing or SSE framing policy.
- M9.6c2 mirrors the C OpenAI tool-stream state machine across search,
  between-invokes, between-params, and param-value states, including generated
  deterministic stream call IDs from an injected prefix and cached call IDs for
  final parsed-call alignment.
- M9.6c2 preserves raw JSON parameter fragments, JSON-escapes string
  parameter fragments after DSML entity unescape, emits parameter commas and
  string close quotes at the same boundaries as C, and holds partial
  invoke/parameter tags, partial parameter-close sentinels, partial DSML
  entities, and split UTF-8 tails.
- M9.6c2 exact event tests cover canonical and plain DSML syntaxes, string
  entity holds, raw JSON argument preservation, comma insertion, partial invoke
  tags, partial parameter-close sentinels, split UTF-8, done/error state
  probes, and unchanged existing M9.6c1/M9.5 formatter tests.
- M9.6c2 local validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.6c2 Claude review returned no blockers after checking state transitions,
  partial tag and tail holds, entity/UTF-8 boundaries, raw JSON preservation,
  string escaping/unescaping, deterministic stream IDs, malformed-tail behavior,
  public API suitability for M9.6c3, and absence of runtime/model policy.
- M9.6c2 did not need B300 validation because it is still model-free; B300
  replay remains owned by M9.6c3.
- M9.6c1 added model-free OpenAI chat tool-stream SSE helpers in
  `ds4_gguf::server_response` for role chunks, live tool-call start deltas,
  argument-fragment deltas, full-call fallback deltas, finish chunks, optional
  usage chunks, stream headers, and `[DONE]`.
- M9.6c1 keeps parser/runtime policy out of the formatter by accepting explicit
  stream events and parsed fallback `DsmlJsonCall` values; full-call fallback
  deltas reuse the M9.6a argument normalizer and `{chat_id}_tool_{index}`
  generated-ID shape.
- M9.6c1 exact byte tests cover escaped tool names, argument fragments
  containing JSON escapes and newlines, generated and explicit full-call IDs,
  invalid-argument `{}` fallback, usage placement, finish `tool_calls`, stream
  headers, and unchanged existing M9.5 SSE responses.
- M9.6c1 local validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.6c1 did not need B300 validation because it is a pure model-free SSE
  formatter; model-backed streaming replay remains owned by M9.6c3.
- M9.6c1 Claude review returned no blockers after checking exact C-shaped
  start/argument/full-call/finish SSE byte shapes, field ordering, generated-ID
  fallback, argument normalization/escaping, usage and `[DONE]` placement,
  unchanged M9.5 stream output, and the absence of parser/runtime policy in
  the formatter API.
- M9.6c was split before implementation because byte-level SSE event
  formatting, incremental DSML-to-delta translation, and model-backed runtime
  replay have distinct oracle/comparator surfaces.
- M9.6c1 is active first because byte-stable tool-call SSE event helpers
  unblock both the stateful streaming translator and the model-backed runtime
  replay without mixing parser or runtime policy into the formatter commit.
- M9.6c docs-only split validation passed for `git diff --check`.
- M9.6c docs-only split Claude review returned no blockers after checking
  tangible goals, oracle/fixture/comparator/acceptance/drift coverage per
  child item, and preservation of event order, argument fragments, finish
  reasons, prompt text, generated DSML, trace records, usage fields, and
  `[DONE]` bytes.
- M9.6b routes supported non-streaming OpenAI tool chat requests through the
  Rust server runtime instead of rejecting them, while keeping streaming tool
  calls rejected for M9.6c and preserving the thinking/stop-list unsupported
  boundaries.
- M9.6b parses completed generated DSML with the existing M5.6
  `parse_generated_message_for_response` path, maps parsed calls to
  `finish_reason:"tool_calls"`, assigns deterministic OpenAI `call_` IDs for
  normalized comparison, and emits the M9.6a tool-call response formatter.
- M9.6b extends runtime traces with parsed-message `dsml_start`, `dsml_end`,
  parsed content/reasoning when present, and `tool_call[n]` records containing
  ID, name, and parser-produced argument JSON.
- M9.6b B300 validation used `/workspace/ds4-m96b` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed in the pod.
  The server replay on port `18200` normalized only outer response ID,
  timestamp, and generated tool-call ID while matching M0.4 `chat_tool_call`
  headers/body semantics, tool name `list_files`, arguments `{"path":"."}`,
  finish `tool_calls`, usage `394/42/436`, cache `0/394`, generated DSML, and
  trace fields.
- The M9.6b prompt comparator used the raw request JSON extracted from the
  M0.4 trace rather than the pretty-printed committed
  `chat_tool_call.json` fixture, because the C/Rust prompt renderer preserves
  raw tool-schema whitespace. The pretty fixture still produced the same
  normalized response, but not byte-identical prompt text.
- M9.6b artifacts remain in the pod at
  `/tmp/ds4-m96b-chat_tool_call.request.json`,
  `/tmp/ds4-m96b-chat_tool_call.json`,
  `/tmp/ds4-m96b-chat_tool_call.headers`,
  `/tmp/ds4-m96b-server.trace`, and `/tmp/ds4-m96b-server.stderr`; the server
  process was stopped and `pgrep` showed no lingering `ds4-server-runtime-rs`
  process beyond the check shell.
- M9.6b local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_response -- --nocapture`, targeted
  `cargo test -p ds4-gguf dsml -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.6b Claude review returned no blockers after checking runtime routing,
  parser finish mapping, deterministic call-ID fallback, response/usage
  preservation, trace records, unsupported-route boundaries, and the
  raw-trace-request prompt comparator.
- M9.6a added pure OpenAI chat tool-call response formatting in
  `ds4_gguf::server_response`, reusing the existing chat completion response
  struct while adding explicit tool-call JSON/HTTP helpers.
- M9.6a normalizes parser-produced tool-call argument objects through the DSML
  JSON argument parser before escaping them as OpenAI `function.arguments`
  strings, matching C's `append_json_object_string` behavior and falling back
  to `{}` for invalid argument JSON.
- M9.6a tests compare the exact M0.4 `chat_tool_call` JSON body and HTTP
  headers, generated call IDs (`{chat_id}_tool_{index}`), explicit call IDs,
  multiple-call ordering, escaped names, normalized argument strings, and
  invalid-argument fallback without touching model execution or runtime
  routing.
- M9.6a local validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, targeted
  `cargo test -p ds4-gguf dsml -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.6a Claude review returned no blockers after checking response field
  order, argument normalization/escaping, call-ID fallback semantics,
  usage/cache preservation, and the absence of runtime/model routing changes.
- M9.6 was split before implementation because the remaining server tool-call
  work spans independent oracle surfaces: pure final-response JSON formatting,
  model-backed non-streaming replay, streaming tool-call deltas, and
  tool-quality regression wiring.
- M9.6a is active first because M5.6 already owns generated DSML parsing and
  M9.2b already owns model-free tool schema prompt rendering; a pure
  `tool_calls` response formatter is the smallest remaining prerequisite for
  the runtime replay.
- M9.6 split validation passed for `git diff --check`; implementation
  validation remains owned by M9.6a and later sub-items.
- M9.5 was split before implementation because pure SSE byte formatting and
  model-backed per-token streaming replay have distinct oracles, comparators,
  and validation gates.
- M9.5a added pure OpenAI chat SSE helpers for headers,
  role/content/final/usage chunks, optional usage omission, JSON escaping, and
  `[DONE]` formatting with injected IDs/timestamps and fixed deltas.
- M9.5a formatter tests compare exact M0.4 `chat_stream.sse` bytes and stream
  headers; the committed fixture ends with one newline after `[DONE]`, and the
  formatter follows that oracle byte-for-byte.
- M9.5a validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, decode-policy
  streaming hold tests `cargo test -p ds4-gguf decode_policy -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.5b added model-backed streaming routing for supported OpenAI chat
  requests, captured raw per-token text chunks from `ServerSession`, and fed
  those chunks through the M9.5a formatter for SSE responses.
- M9.5b converts token chunks into SSE deltas through the existing
  `utf8_stream_safe_len` helper, preserving ordinary token boundaries while
  holding split multi-byte UTF-8 bytes until they are safe to emit.
- M9.5b keeps tools, thinking, and stop-list requests outside the streaming
  path while preserving the existing non-streaming response behavior.
- M9.5b local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_response -- --nocapture`,
  decode-policy streaming hold tests
  `cargo test -p ds4-gguf decode_policy -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.5b B300 validation used `/workspace/ds4-m95b` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed, and the
  server replay on port `18198` normalized only IDs/timestamps while matching
  M0.4 `chat_stream` SSE headers/body, deltas `stream` and ` baseline`, finish
  `stop`, usage `11/2/13`, cache `0/11`, and one newline after `[DONE]`.
- M9.5b B300 trace validation checked `stream: 1`, `prompt_tokens: 11`,
  `stream_include_usage: 1`, `cache_source: none`, `generated_tokens: 2`, and
  final content `stream baseline`. Artifacts remain in the pod at
  `/tmp/ds4-m95b-server.trace`, `/tmp/ds4-m95b-server.stderr`, and
  `/tmp/ds4-m95b-chat_stream.*`; the server process was stopped by the
  validation script.
- M9.4d added a reusable `ServerSession` path for model-backed server
  generation so `/v1/chat/completions` requests in one Rust server process can
  reuse the live token prefix from prior completions.
- M9.4d reports server-generation cache read/write counts,
  `live_tokens_before`, and `live_prompt_common` from the Rust session, then
  forwards those counts into OpenAI usage details and `--trace` cache-decision
  fields.
- M9.4d local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4d B300 validation used `/workspace/ds4-m94d` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed, and a single
  server replay on port `18197` normalized only IDs/timestamps while matching
  `chat_cache_seed` content `cache ready`, finish `stop`, usage `39/2/41`,
  cache `0/39`, and `chat_cache_continuation` content `cache continued`,
  finish `stop`, usage `50/2/52`, cache `41/9`.
- M9.4d B300 trace validation checked rendered prompts, `cache_source: none`
  for the seed, `cache_source: memory-token`, `cached_tokens: 41`,
  `live_tokens_before: 41`, `live_prompt_common: 41`,
  `memory_token_reusable: 1`, generated token counts, and final content for
  the continuation. Artifacts remain in the pod at
  `/tmp/ds4-m94d-server.trace`, `/tmp/ds4-m94d-server.stderr`, and
  `/tmp/ds4-m94d-*.json`; the server process was stopped by the validation
  script.
- M9.4c added server-generation runtime support in `ds4-engine` that syncs a
  rendered prompt into a fresh session, samples raw token text without the CLI
  trailing-newline printer, returns prompt/completion token counts, and reports
  `stop`, `length`, or `error` finish reasons.
- M9.4c routes model-backed OpenAI `/v1/chat/completions` requests through the
  runtime only for the no-cache M0.4 surface: non-streaming, no tools,
  non-thinking, no stop-list requests. Streaming, tools, thinking, and stops
  remain explicit 503 boundaries for later M9 items.
- M9.4c uses the M9.4b response builder for successful chat responses and
  writes a `--trace` file with request metadata, no-cache decision fields,
  rendered prompt, generated text, finish reason, and generated-token count.
- M9.4c local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4c B300 validation used `/workspace/ds4-m94c` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed, and the
  server replay on port `18196` normalized only IDs/timestamps while matching
  `chat_basic` content `baseline ready`, finish `stop`, usage `11/3/14`, and
  `chat_thinking_disabled` content `2`, finish `stop`, usage `15/1/16`.
- M9.4c B300 trace validation checked rendered prompts, `cache_source: none`,
  prompt token counts, generated token counts, and final content for both
  no-cache fixtures. Artifacts remain in the pod at
  `/tmp/ds4-m94c-server.trace`, `/tmp/ds4-m94c-server.stderr`, and
  `/tmp/ds4-m94c-*.json`; the server process was stopped by the validation
  script.
- M9.4b added `rust/ds4-gguf/src/server_response.rs` with pure formatting
  helpers for OpenAI non-streaming chat-completion JSON, HTTP response
  wrapping, usage details, finish reasons, optional reasoning content, and
  C-compatible cache read/write clamping.
- M9.4b response-builder tests compare exact M0.4 `chat_basic`,
  `chat_thinking_disabled`, `chat_cache_seed`, and `chat_cache_continuation`
  JSON bodies using injected IDs/timestamps and explicit usage/cache counts.
- M9.4b header tests compare the `chat_basic` `Content-Length`,
  `Content-Type`, and `Connection: close` header surface through the existing
  C-shaped HTTP formatter; escaping tests cover visible content and optional
  reasoning content.
- M9.4b validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4a added `ds4-server-runtime-rs` under `rust/ds4-engine` so the
  model-backed server boundary lives in the runtime crate and depends on
  `ds4-gguf` helpers without creating a dependency cycle.
- M9.4a parses the C server startup subset for model path, backend selection,
  MTP options, threads, directional steering, warm/quality flags, host, port,
  context length, default tokens, and CORS.
- M9.4a opens `Engine`, creates a session, uses `Engine::encode_chat_prompt`
  for tokenizer-backed prompt-token counts, preserves M9.3 no-model
  route/error behavior, and returns a distinct 503 JSON error for valid
  generation while model-backed chat generation remains unimplemented.
- M9.4a local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_no_model -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4a B300 validation used a copied source snapshot at `/workspace/ds4-m94a`
  and model `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed and
  the server smoke on port `18194` loaded CUDA, answered `/v1/models` with
  `context_length=16`, returned `missing messages` for bad chat JSON,
  returned a tokenizer-backed completion context error with
  `n_prompt_tokens=29` and `n_ctx=16`, and rejected valid chat generation with
  `model-backed chat generation is not implemented yet`.
- M9.4a B300 smoke artifacts remain in the pod at
  `/tmp/ds4-m94a-server.stderr` and `/tmp/ds4-m94a-*.out`; the server process
  was stopped by the validation script.
- M9.4 was split before implementation because model-backed server ownership,
  non-streaming response formatting, no-cache B300 generation replay, and
  memory-token cache continuation have distinct oracles and validation gates.
- M9.4a now owns the model-backed Rust server runtime boundary, engine/session
  lifetime, model-load smoke replay, tokenizer-backed prompt-token counting,
  and preservation of M9.3 no-model route/error behavior.
- M9.4b now owns pure OpenAI non-streaming chat response/usage/header
  formatting, with IDs and timestamps injected or normalized outside the
  builder.
- M9.4c now owns B300 no-cache non-streaming `/v1/chat/completions` replay for
  `chat_basic` and `chat_thinking_disabled`.
- M9.4d now owns live memory-token cache seed/continuation replay for
  `chat_cache_seed` and `chat_cache_continuation`, leaving disk KV/tool-memory
  behavior to M9.8.
- M9.3c2 added the `ds4-server-rs` binary with model-free `--host`, `--port`,
  `--ctx`, `--tokens`, and `--cors` startup flags plus C-style localhost
  binding behavior.
- M9.3c2 wires accepted TCP sockets through the M9.3c1 no-model dispatcher,
  reads until a complete request or stable parser failure, writes C-shaped
  HTTP responses, and closes the client socket.
- M9.3c2 added real process/socket replay coverage for OPTIONS, model list,
  unknown route, malformed HTTP, missing messages, unsupported durable state,
  unsupported tool choice, no-model context-limit response, and valid
  generation rejection through 503.
- M9.3c2 validation passed for local no-model HTTP comparator
  `cargo test -p ds4-gguf --test no_model_server -- --nocapture`, targeted
  binary tests `cargo test -p ds4-gguf --bin ds4-server-rs -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3c1 added `rust/ds4-gguf/src/server_no_model.rs` as a socket-free
  dispatcher that combines M9.3a/M9.3b HTTP helpers with generation-route
  parse/error handling.
- M9.3c1 added Rust completion request parsing for `/v1/completions` negative
  and no-model dispatch paths, including missing-prompt handling and C-shaped
  completion prompt rendering.
- M9.3c1 added API-specific context-length response bodies for OpenAI chat,
  Responses, completions, and Anthropic messages, with injectable prompt-token
  counting for future tokenizer-backed or replay-specific checks.
- M9.3c1 route tests cover bad HTTP, preflight, model route reuse, bad JSON,
  missing `messages`/`input`/`prompt`, unsupported durable state, unsupported
  tool choice, per-API context-length bodies, CORS propagation, and valid
  generation rejection through a 503 no-model JSON error.
- M9.3c1 validation passed for targeted
  `cargo test -p ds4-gguf server_no_model -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3c was split before implementation because in-memory generation-route
  parser/error dispatch and socket/process replay have distinct oracles,
  fixture families, and validation gates.
- M9.3c1 now owns the no-model dispatcher that combines M9.3a/M9.3b helpers
  with generation-route parse/error mapping for bad HTTP, bad JSON, missing
  messages/input/model/prompt, unsupported durable state/tool choice, and
  context length errors without sockets or model loading.
- M9.3c2 now owns the `ds4-server-rs` binary, CLI flag parsing, TCP bind/listen
  loop, accepted-socket dispatch, local replay comparator, and deterministic
  shutdown behavior.
- M9.3b added a no-model HTTP route dispatch surface over the M9.3a in-memory
  parser/formatter helpers.
- M9.3b covers C `client_main` behavior for `OPTIONS`, `GET /v1/models`,
  `GET /v1/models/deepseek-v4-flash`, unknown endpoint rejection, and bad HTTP
  request rejection without opening sockets or loading a model.
- M9.3b added deterministic model metadata JSON matching C
  `append_model_json_values`, including `deepseek-v4-flash`, created timestamp
  `1767225600`, owner `ds4.c`, provider context length, capped
  `max_completion_tokens`, and supported parameter ordering.
- M9.3b route tests compare exact response bytes for preflight, model list,
  single-model, bad HTTP, unknown endpoint, wrong method, CORS propagation, and
  parser-level query stripping; the model-list body is also compared with the
  captured M0.4 `models.json` fixture.
- M9.3b validation passed for targeted
  `cargo test -p ds4-gguf server_http -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3a added `rust/ds4-gguf/src/server_http.rs` with in-memory HTTP request
  parsing plus C-shaped response/error formatting and exported the helper
  surface from `ds4_gguf`.
- M9.3a covers C-style CRLF and LF-only header terminators, query stripping,
  case-insensitive `Content-Length`, exact body slicing, malformed/incomplete
  request categories, 200 JSON response bytes, 204 no-content preflight bytes,
  opt-in CORS header order, and JSON error body escaping.
- M9.3a validation passed for targeted
  `cargo test -p ds4-gguf server_http -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3 was split before implementation because byte-level HTTP
  framing/CORS, no-model route/model metadata dispatch, and socket/process
  replay have distinct C oracles, fixture families, and validation gates.
- M9.3a now owns in-memory HTTP request parsing, response/error byte
  formatting, CORS header parity, query stripping, content-length handling, and
  malformed request categories without opening sockets.
- M9.3b now owns OPTIONS and model metadata route dispatch on top of M9.3a,
  including `/v1/models`, `/v1/models/deepseek-v4-flash`, unknown endpoint
  behavior, and deterministic model JSON.
- M9.3c now owns the local `ds4-server-rs` no-model binary, socket loop,
  startup flags, no-generation negative replay, and local HTTP comparator.
- M9.2c3c added exported Rust `AnthropicLiveState` plus
  `parse_anthropic_core_request_with_live_state` so parser tests can exercise
  live-known Anthropic tool-result continuations without touching server KV or
  tool-memory side effects.
- M9.2c3c validates Anthropic `tool_result.tool_use_id` like C: missing
  tool-result-only state returns the exact Anthropic continuation error,
  live-known IDs set `anthropic_requires_live_tool_state`, and replayed prior
  assistant `tool_use` blocks avoid the live-state requirement.
- M9.2c3c collects trailing Anthropic tool-result IDs, renders the visible
  live suffix from EOS through user tool results to the next assistant prefix,
  and ignores appended system messages when locating the final tool-result
  tail.
- M9.2c3c validation covered missing live state, live-known tool-result-only
  continuation, replayed prior `tool_use` with `content` before `role`, exact
  error text, delimiter escaping, collected IDs, and live-tail prompt bytes.
- M9.2c3c validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2 request parse and prompt-render surface is now complete enough for the
  roadmap to move to M9.3 HTTP skeleton work.
- M9.2c3b extended Rust Anthropic request parsing for top-level `tools`,
  `tool_choice.type`, active tool-schema prompting, assistant `tool_use`
  blocks, user `tool_result` blocks, tool-use IDs, and DSML request-history
  rendering.
- M9.2c3b preserves raw nested `input` JSON for Anthropic `tool_use` blocks so
  DSML parameter rendering keeps object-field order and numeric spelling such
  as `2.0`, while still using stable fallback `arguments` text for non-object
  input.
- M9.2c3b validation covered direct Anthropic schemas, OpenAI-compatible
  wrapped tools, `tool_choice.type` auto/none behavior, Anthropic string
  `tool_choice` skip behavior, role-after-content `tool_use`, content-array
  `tool_result`, call-id preservation, delimiter escaping, and exact prompt
  bytes for visible tool history.
- M9.2c3b validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c3a added exported Rust `AnthropicRequest` and
  `parse_anthropic_core_request` for model-free Anthropic core request
  parsing.
- M9.2c3a covers `messages`, `system` string/array/object parsing, private
  `x-anthropic-*` system filtering, text and thinking content blocks, scalar
  generation controls, stop sequences, stream flag, `thinking`,
  `output_config.effort`, bare `reasoning_effort`, model alias fallbacks, and
  prompt rendering without tools.
- M9.2c3a validation covered missing messages, invalid messages,
  system-private filtering and newline joining, content arrays, thinking block
  rendering, effort precedence, disabled thinking, invalid effort rejection,
  and prompt bytes.
- M9.2c3a validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c3 was split before implementation because Anthropic core
  message/system/control parsing, tool schema/tool history parsing, and live
  tool-result continuation validation have distinct C branches, fixture
  families, and validation comparators.
- M9.2c3a now owns Anthropic `messages`, `system`, text content, private system
  filtering, scalar controls, stop sequences, stream flag, thinking/effort
  controls, model alias fallbacks, and non-tool prompt rendering.
- M9.2c3b now owns Anthropic tool schemas, `tool_choice.type`, assistant
  `tool_use`, user `tool_result`, tool-use IDs, tool-result prompt rendering,
  and DSML request-history rendering.
- M9.2c3c now owns Anthropic missing/live `tool_use_id` validation, live-state
  requirement flags, live tool-use ID collection, and live suffix rendering.
- M9.2c2c added a Rust `ResponsesLiveState` stub plus request metadata for
  `responses_requires_live_tool_state`, `responses_requires_live_reasoning`,
  `responses_live_call_ids`, and `responses_live_suffix_text`.
- M9.2c2c added model-free validation matching C
  `responses_validate_tool_outputs`: tool-output-only requests without live or
  replayed prior call IDs now return the same stable error string, live-known
  tool-output-only requests set `requires_live_tool_state`, and stateless
  thinking-mode replays without prior reasoning set `requires_live_reasoning`.
- M9.2c2c added a Rust live-tool-tail renderer matching C
  `render_live_tool_tail`, including the leading EOS, tool-result delimiter
  escaping, and next assistant prefix for thinking and non-thinking modes.
- M9.2c2c validation covered missing live state, live-known tool-output-only
  continuations, replayed prior tool calls with and without reasoning,
  non-thinking replay, call-id collection, and suffix text excluding prior tool
  call DSML.
- M9.2c2c validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c2b extended `ToolSchemaOrder` with namespace, wire-name, and hosted
  `tool_search` metadata needed by Responses schema parsing.
- M9.2c2b extended Rust `parse_tools_value` to match C handling for top-level
  hosted `tool_search`, normal functions named `tool_search`, namespace tool
  flattening, property order capture, and dynamic `tool_search_output.tools`
  schema loading.
- M9.2c2b combines top-level tool schemas before dynamically loaded schemas in
  prompt text while preserving parser field-order replacement semantics in
  `tool_orders`.
- M9.2c2b validation covered hosted `tool_search` distinction, namespace
  prompt-name flattening with namespace/wire metadata, dynamic schema loading
  after top-level schemas, and malformed dynamic tool-list rejection.
- M9.2c2b validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c2a extended the Rust prompt data model with stable tool-call IDs on
  assistant calls and tool result messages without changing existing prompt
  rendering for non-tool histories.
- M9.2c2a extended Responses `input` parsing to preserve raw item fields where
  C preserves raw JSON, parse `function_call`, `custom_tool_call`,
  `local_shell_call`, `web_search_call`, `tool_search_call`,
  `image_generation_call`, function/custom outputs, and hosted tool outputs
  into `ChatMessage`/`ToolCall` history.
- M9.2c2a validation covered assistant text plus split tool-call merging,
  reasoning attachment, DSML argument ordering and numeric spelling,
  custom-tool free-text fallback, hosted tool names/actions, output/result/tool
  body selection, tool-result delimiter escaping, and call-id preservation.
- M9.2c2a validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c2 was split before implementation because Responses tool-call input
  parsing, dynamic tool schema loading, and live continuation validation have
  distinct C branches, fixture families, and validation comparators.
- M9.2c2a now owns function/custom/hosted call inputs, tool output inputs, call
  IDs, pending-reasoning merge rules, and DSML prompt rendering.
- M9.2c2b now owns hosted `tool_search`, namespace schema flattening,
  tool-search-output dynamic schema loading, combined schema ordering, and
  namespace/wire-name metadata.
- M9.2c2c now owns missing/live call-id validation, live-state requirement
  flags, reasoning replay requirement flags, call-id collection, and live-tail
  suffix text.
- M9.2c1 added the exported Rust `ResponsesRequest` and
  `parse_responses_core_request` surface for model-free Responses API core
  request parsing.
- M9.2c1 covers bare string and array `input`, `instructions` system prepend,
  scalar generation controls, `reasoning.effort`, reasoning summary opt-in,
  model-alias thinking fallbacks, prompt rendering, top-level tool prompt
  participation, durable-state rejection for `previous_response_id` and
  `conversation`, unsupported `tool_choice` categories, and strict text-content
  shape checks.
- M9.2c1 validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c was split before implementation because Responses core input/reasoning,
  Responses tool-output/live-tail validation, and Anthropic content/tool-result
  parsing have distinct oracle surfaces, fixtures, validation categories, and
  prompt/live-tail comparators.
- M9.2c split review initially found boundary gaps; the split now explicitly
  defers KV/tool-memory replay side effects to M9.8, makes M9.2c2 depend on
  M9.2c1 for combined Responses tool schemas, names `parse_anthropic_system`
  and `parse_anthropic_system_object`, and includes Anthropic bare
  `reasoning_effort` coverage.
- M9.2c1 now owns Responses core `input`, `instructions`, scalar generation
  controls, reasoning effort/summary flags, durable-state rejection, and prompt
  rendering without live tool-state validation.
- M9.2c2 now owns Responses function/tool output inputs, tool-search schema
  loading, namespace tool schema restoration, and live-tail validation.
- M9.2c3 now owns Anthropic content/system blocks, tools/tool_choice,
  stop/thinking controls, tool-use/tool-result messages, and live-tail
  validation.
- M9.2b extended `rust/ds4-gguf/src/server_chat.rs` to parse OpenAI `tools`,
  `tool_choice`, assistant `tool_calls`, tool schema property order, tool role
  prompt rendering, and DSML request-history rendering without loading a model
  or opening sockets.
- M9.2b exact prompt-byte coverage uses the raw request body captured in the
  M0.4 `chat_tool_call` trace because C preserves raw tool-schema whitespace in
  the prompt; the pretty committed JSON fixture is still used for semantic
  parser checks.
- M9.2b validation covered schema order, `tool_choice: "none"` suppression,
  DSML argument ordering, raw JSON argument minification, numeric spelling
  preservation, tool-result delimiter escaping, and reasoning preservation when
  tools are active.
- M9.2a added a dependency-free Rust OpenAI chat request parser in
  `rust/ds4-gguf/src/server_chat.rs`, exported through `ds4-gguf`, for
  model-free `/v1/chat/completions` core fields.
- M9.2a parser coverage includes the M0.4 non-tool OpenAI fixtures
  `chat_basic`, `chat_stream`, `chat_thinking_disabled`, `chat_cache_seed`,
  and `chat_cache_continuation`, matching rendered prompt bytes, stream flags,
  sampling defaults/options, seeds, max-token fields, thinking mode, stop lists,
  error categories, and OpenAI context-length error body shape.
- M9.2a Claude review returned PASS on the material parser parity checks after
  reviewing defaults, duplicate-field order, stream options, thinking/reasoning
  mapping, stop lists, context-length helper shape, prompt bytes, and fixture
  coverage.
- M9.2a intentionally excludes tool schema/tool-call payload handling and
  alternate protocols; M9.2b owns OpenAI tool schema plus DSML prompt rendering,
  while M9.2c owns Responses and Anthropic request inputs.
- `git status --short` was clean before M0.1 edits.
- `AGENT.md`, `CONTRIBUTING.md`, and `RUST_PORT_ROADMAP.md` were read before
  creating the protocol.
- M0.1 validation passed with `git diff --name-only` and `git diff --check`.
- M0.1 Claude review returned PASS before commit.
- M0.2 local arm64 validation captured `arch -arm64 make` exit 0,
  `arch -arm64 make cpu` exit 0, `./ds4_test --server` exit 0, and
  `./ds4_test --metal-kernels` exit 0.
- M0.2 local default `make` and model-backed `make test` failures are recorded
  in `ds4-parity/baselines/manifest.md` with exact logs.
- M0.2 B300 validation captured `make cuda-generic` exit 0,
  `make cuda-regression` exit 0, `./ds4_test --server` exit 0, and
  `./ds4_test --metal-kernels` exit 0 on `ds4-rust-port-b300`.
- M0.3 B300 validation downloaded q2-imatrix, recorded model hash/size, built
  `ds4_test`, and captured `./ds4_test --logprob-vectors` exit 0 with
  `logprob-vectors: OK`.
- M0.4 B300 validation refreshed source commit
  `3d87577962abeac1ab0d80e9c21d0012bfc53afb`, built `ds4-server`, and replayed
  six server fixtures from `ds4-parity/baselines/server-fixtures/m0.4/` with
  HTTP 200 for all requests.
- M0.4 artifacts live under `ds4-parity/baselines/server-traces/m0.4/`; the
  final trace records non-streaming chat, SSE chat, DSML-to-OpenAI tool calls,
  explicit thinking-disabled chat, and cache continuation with
  `cache_source=memory-token`, `cached_tokens=41`, `cache_write_tokens=9`.
- M0.5 B300 validation refreshed source commit
  `0623bbb4d97d056a58e208e324216f97abed685e`, built `ds4-server`, and replayed
  three disk-KV server lifetimes from
  `ds4-parity/baselines/kv-fixtures/m0.5/` with HTTP 200 for all requests.
- M0.5 artifacts live under `ds4-parity/baselines/kv-artifacts/m0.5/`; the
  replay records a cold 550-token cache write, a fresh-process 550-token
  `disk-text` restore, and a fresh-process continuation restore of the
  552-token shutdown prefix with a 9-token suffix write.
- M0.5 raw `.kv` files are not checked in; committed comparator artifacts
  include full raw hashes, timestamp-normalized hashes, parsed KVC headers, and
  extracted rendered cache text.
- M0.6 B300 validation refreshed source commit
  `add2c507f81aa2e363809213771134c282c50bf2`, built `ds4-bench`, and captured
  short-context and long-context CSV baselines using
  `speed-bench/promessi_sposi.txt` with SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- M0.6 artifacts live under `ds4-parity/baselines/bench/m0.6/`; the short CSV
  covers 2048 through 8192 tokens and the long CSV covers 16384 through 32768
  tokens, both with 32 greedy generation tokens per frontier.
- M1.1 documented the Milestone 1 implementation work items in
  `RUST_PORT_ROADMAP.md`; the next executable item is a static verifier for the
  committed Milestone 0 artifacts.
- M1.2 added `python3 ds4-parity/verify_baselines.py`, which verifies M0.2
  through M0.6 artifact families locally without rerunning model-backed
  commands. Its negative test corrupts a copied benchmark CSV and requires the
  verifier to detect the drift.
- M1.3 added `python3 ds4-parity/compare_server_kv.py`, which self-compares
  committed M0.4 server and M0.5 KV artifacts with only documented
  normalizations. Its negative test covers finish reason, cached token count,
  cache source, KV reason, and rendered text drift.
- M1.4 added `python3 ds4-parity/compare_logprob_numeric.py`, which parses the
  compact official-vector fixture, audits it against raw official API JSON,
  verifies the M0.3 B300 pass markers, and compares candidate vector files with
  exact selected tokens plus a reported 4.0 absolute logprob tolerance. Its
  negative test covers selected-token drift and numeric drift outside tolerance.
- M1.5 added `python3 ds4-parity/compare_bench_csv.py`, which self-compares
  committed M0.6 benchmark CSV artifacts, validates capture metadata for
  threshold use, requires exact workload shape and KV byte counts, and applies
  the documented 10% throughput regression threshold. Its negative test covers
  schema, context frontier, generation-token, cache-byte, and throughput drift.
- M1.6 added `python3 ds4-parity/run_parity_report.py`, which runs local
  no-model C oracles, invokes the M1.2 through M1.5 comparator commands, and
  reports skipped B300/model-backed oracle refreshes with explicit
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1` rerun
  commands. The local report passed with 9 executed items and 4 B300 refreshes
  skipped by design.
- M2.1 added a Rust workspace with `ds4-gpu-sys` and `ds4-gpu`, seeded core
  tensor/command/model-map FFI declarations, added smoke-only safe status
  wrappers, and wired `make rust-test`. Validation passed for `cargo fmt`,
  `cargo test --workspace`, `make rust-test`, sequential `arch -arm64 make`,
  sequential `arch -arm64 make cpu`, and the unified parity report.
- M3.1 added safe Rust `Tensor`, `TensorView`, and `CommandBatch` wrappers over
  the existing `ds4_gpu.h` tensor/command ABI without changing the C ABI. The
  macOS `ds4-gpu` build script compiles the current `ds4.c` and `ds4_metal.m`
  backend objects into a test archive so Rust tests call the real Metal
  implementation rather than a mock.
- M3.1 Rust ABI parity validation passed with
  `cargo test -p ds4-gpu safe_tensor_wrapper_matches_direct_c_abi -- --nocapture`;
  the test compares safe-wrapper and direct-C paths for allocation, byte-size
  queries, write/read, fill, view writes, command-batched copy, flush/end,
  synchronize, and out-of-bounds write/copy failures.
- M3.1 full validation passed for `cargo fmt --all -- --check`,
  `cargo test --workspace`, `make rust-test`, `git diff --check`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make cpu`, and
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed.
- M4.1 split Milestone 4 into concrete GGUF/model-metadata work items after
  reading `ds4.c` loader, summary, metadata validation, base tensor binding,
  and MTP tensor binding surfaces. The next executable item is the current-C
  metadata dump oracle.
- M4.2 added `./ds4-metadata-dump`, which opens the model through the current C
  GGUF loader, runs `config_validate_model` and `weights_bind`, and emits
  deterministic `ds4.metadata.v1` JSON with selected metadata values, tensor
  type histograms, all tensor descriptors, and bound semantic tensor roles.
- M4.2 added `python3 ds4-parity/check_metadata_dump.py`, whose schema checker
  validates the dump and whose negative test detects tensor-count drift, a
  missing required bound role, and a missing required metadata key.
- M4.2 B300 validation copied the M4.2 source files into
  `/workspace/ds4`, built with `make clean ds4-metadata-dump CUDA_ARCH=native`,
  dumped `/workspace/ds4/ds4flash.gguf`, and passed
  `python3 ds4-parity/check_metadata_dump.py /tmp/ds4-metadata.json --negative-test`.
  The generated B300 dump had 633,297 bytes, SHA256
  `39ad79574b19421e2c470a055376258b9415eb1f429188426cfd2860688a2a2f`,
  1,328 tensors, and 1,511 bound tensor roles.
- M4.2 local validation passed for `arch -arm64 make ds4-metadata-dump`,
  `./ds4-metadata-dump --help`, local schema/negative checks against the copied
  B300 dump, `python3 -m py_compile ds4-parity/check_metadata_dump.py`,
  sequential `arch -arm64 make clean`, sequential `arch -arm64 make`,
  sequential `arch -arm64 make clean`, sequential `arch -arm64 make cpu`,
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed,
  `cargo test --workspace`, and `git diff --check`.
- M4.3 added a dependency-free `ds4-gguf` Rust crate and `ds4-gguf-dump` CLI
  that parse GGUF v3 metadata and tensor directory records, compute C-equivalent
  tensor byte sizes and aligned absolute offsets, and emit the same
  `ds4.metadata.v1` directory surface as the C metadata dump.
- M4.3 added `./ds4-metadata-dump --directory-only` so local synthetic GGUF
  fixtures can compare the current C GGUF directory parser against Rust without
  requiring the full DS4 model or semantic tensor binding.
- M4.3 added `python3 ds4-parity/compare_gguf_directory.py`, whose synthetic
  fixture covers scalar metadata, array metadata, non-power-of-two
  `general.alignment=48`, F32 byte sizing, Q8_0 block byte sizing, relative and
  absolute offsets, C-compatible float metadata formatting, unsupported scalar
  metadata emission as `null`, and C/Rust rejection of corrupted magic,
  truncated metadata, truncated tensor data, and tensor offset overflow.
- M4.3 B300 check confirmed the pod does not currently have `rustc` or `cargo`,
  so this item used local synthetic C-vs-Rust directory comparison instead of a
  B300 Rust run. Real supported-model Rust comparison remains deferred to the
  roadmap item that provides Rust on the model host or transfers feasible dump
  artifacts.
- M4.3 local validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests,
  `python3 ds4-parity/compare_gguf_directory.py --negative-test` with 14
  checks, `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_gguf_directory.py ds4-parity/check_metadata_dump.py`,
  local schema/negative checks against the copied B300 M4.2 dump,
  `cargo test --workspace`, `git diff --check`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make cpu`, and
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed.
- M4.4 added `./ds4-metadata-dump --validate-config-only`, which runs current C
  `config_validate_model` after GGUF parsing but skips tensor binding, making
  local metadata-only validation fixtures possible without the full model
  tensor table.
- M4.4 added `validate_ds4_metadata` in `ds4-gguf`, matching C required-key
  behavior, `u64` and `f32` coercion rules, optional expert group defaults,
  fixed DeepSeek4 constants, compression-ratio arrays, SwiGLU clamp arrays,
  RoPE constants, HC constants, RMS epsilon, expert weight scale, and expert
  weight normalization.
- M4.4 added `python3 ds4-parity/compare_metadata_validation.py`, whose
  synthetic fixtures compare C and Rust pass/fail behavior and normalized first
  failures for baseline metadata, C-compatible numeric coercions, missing keys,
  wrong scalar types, wrong scalar values, short arrays, negative compression
  ratios, wrong compression ratios, float tolerance failures, non-integer
  `u64` inputs, non-float `f32` inputs, and boolean drift.
- M4.4 focused validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests, `python3
  ds4-parity/compare_metadata_validation.py --negative-test` with 41 checks,
  and `python3 -m py_compile ds4-parity/compare_metadata_validation.py`.
- M4.5 added `./ds4-metadata-dump --validate-layout-only`, which runs current C
  metadata validation plus base/MTP tensor binding and layout validation from
  GGUF directories while skipping tensor payload range checks for synthetic
  local fixtures.
- M4.5 added Rust base and MTP tensor binding/layout validation in `ds4-gguf`,
  including required, optional, compression-ratio-dependent, hash-layer-only,
  plain F16/F32 MTP, routed expert quant-type, routed gate/up type equality,
  and fixed tensor dimension rules.
- M4.5 added `python3 ds4-parity/compare_tensor_bindings.py`, whose synthetic
  fixtures compare C and Rust layout dumps for base plus MTP bindings and
  negative cases for missing required tensors, wrong types, wrong dimensions,
  optional tensor type drift, routed expert type drift, routed gate/up type
  mismatch, missing compressor/indexer tensors, MTP plain-type rejection, and
  missing required MTP tensors.
- M4.5 focused validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests, `python3
  ds4-parity/compare_tensor_bindings.py --negative-test` with 33 checks, and
  `python3 -m py_compile ds4-parity/compare_tensor_bindings.py`.
- M4.6 recaptured the supported-model metadata baseline on B300
  `ds4-rust-port-b300` in `hou2-prod1` after refreshing `ds4.c`, `ds4.h`, and
  `ds4_metadata_dump.c` from source commit
  `58bad019226499d5b294340093f77c70b7250b79`.
- M4.6 committed `ds4-parity/baselines/metadata/m4.6/current-c.json` for
  `/workspace/ds4/ds4flash.gguf`, whose resolved path is
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
  model size is 86,720,111,488 bytes, model SHA256 is
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  dump size is 633,297 bytes, and dump SHA256 is
  `39ad79574b19421e2c470a055376258b9415eb1f429188426cfd2860688a2a2f`.
- M4.6 added `python3 ds4-parity/compare_metadata_baseline.py --negative-test`,
  which schema-checks the committed baseline, verifies manifest artifact hashes,
  normalizes model paths/source for candidate comparisons, and detects scalar
  metadata, array metadata, tensor shape, tensor type, binding, and offset
  drift.
- M4.6 wired the metadata baseline comparator into
  `python3 ds4-parity/run_parity_report.py` and added a B300 skip entry with
  exact source-refresh, capture, hash, schema-check, and copy-back commands.
- M4.7 added `python3 ds4-parity/compare_gguf_failures.py`, a generated
  malformed-GGUF matrix that compares C and Rust rejection status plus
  normalized first-error categories for invalid magic, unsupported version,
  truncated metadata, unknown metadata type, bad tensor dimension, out-of-file
  tensor data, tensor offset overflow, missing required metadata, wrong
  metadata type, bad metadata array length, and unsupported DS4 tensor type.
- M4.7 validation passed for `arch -arm64 make ds4-metadata-dump`,
  `python3 ds4-parity/compare_gguf_failures.py` with 55 checks,
  `python3 ds4-parity/compare_gguf_failures.py --list-cases`, M4.3 through
  M4.5 comparators (`compare_gguf_directory.py --negative-test`,
  `compare_metadata_validation.py --negative-test`, and
  `compare_tensor_bindings.py --negative-test`), `python3 -m py_compile` for
  all involved comparators, and `cargo test --workspace`.
- M5.1 split Milestone 5 into M5.2 through M5.7 after reading tokenizer source
  (`vocab_load`, JoyAI `bpe_tokenize_text`, rendered-chat special tokenization,
  `ds4_token_text`, and `ds4_dump_text_tokenization`), CLI prompt paths
  (`--dump-tokens`, `build_prompt`, and REPL append functions), server prompt
  and API paths (`parse_chat_request`, `render_chat_prompt_text`,
  `render_live_tool_tail`, and DSML formatting/parsing helpers), the agent DSML
  streaming parser, and existing M0.3/M0.4/M0.5 fixtures.
- M5.1 validation passed for `git diff --check` and non-interactive Claude
  review after tightening tokenizer identity, server-vs-CLI prompt oracles,
  token decoding ownership, DSML chunk/EOF parser schedules, tool-schema
  fixture variants, and request body hashing; final Claude review returned
  `NO BLOCKERS`.
- M5.2 added current-C tokenizer and prompt oracle dumping through
  `./ds4-server --dump-token-oracle`, with tokenizer identity hashing in
  `ds4_engine_dump_tokenizer_identity_json`, shared `ds4_sha256_hex`, raw
  request-body hashing, server prompt/token fixtures, and CLI `ds4_chat_*`
  operation/token-stream fixtures. The dump mode opens the model through the
  existing engine path but exits before session/listener/worker startup, and
  advisory token text emission preserves valid UTF-8 while escaping invalid
  raw bytes so future byte-fallback fixtures still produce valid JSON.
- M5.2 committed
  `ds4-parity/baselines/tokenization/m5.2/current-c.json` captured on B300
  `ds4-rust-port-b300` from `/workspace/ds4/ds4flash.gguf`; dump size is
  124,497 bytes and dump SHA256 is
  `b0689f47abe63750ab3191772d5661d5f0f433e954bcfd0de6a0e55a747489e9`.
  The tokenizer identity records 129,280 tokens, token-bytes SHA256
  `c92251fc634ff01cc6767d2e3ce1d368e72b5f02b647983d4410eb0c46693fa3`,
  127,741 merge records, merge-pairs SHA256
  `8100a9693dc10b8aad79abbe20b172545ff5e1e6051e0705cc91e73b88e3751f`,
  the seven rendered-control specials, and 863 literal-special tokens.
- M5.2 B300 validation passed after copying the changed source/checker into
  `/workspace/ds4`, building with `make clean ds4-server CUDA_ARCH=native`,
  dumping the oracle from the q2-imatrix model, and running
  `python3 ds4-parity/check_tokenization_dump.py
  /tmp/ds4-tokenization-m5.2-current-c.json --negative-test`; the final B300
  checker reported `tokenization schema: PASS, 13409 checks` and
  `tokenization negative tests: PASS, 12 checks`.
- M5.2 local validation passed for
  `python3 ds4-parity/check_tokenization_dump.py
  ds4-parity/baselines/tokenization/m5.2/current-c.json --manifest
  ds4-parity/baselines/tokenization/m5.2/manifest.json --negative-test`,
  with `tokenization schema: PASS, 13409 checks`, `tokenization manifest:
  PASS, 11 checks`, and `tokenization negative tests: PASS, 12 checks`;
  `python3 -m py_compile ds4-parity/check_tokenization_dump.py`,
  `./ds4_test --server`, `git diff --check`, `arch -arm64 make ds4-server`,
  `cargo test --workspace`, and `arch -arm64 make cpu` also passed.
- M5.2 Claude review returned `NO BLOCKERS`; after hardening invalid UTF-8
  token text escaping and checker pinning for exact special/server semantics,
  the follow-up Claude review also returned `NO BLOCKERS`.
- M5.3 added `Ds4Tokenizer` to `ds4-gguf`, loading
  `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` from GGUF metadata,
  computing the same canonical token/merge SHA256 identity as C, validating
  required DS4 special token IDs, porting JoyAI plain-text pre-tokenization and
  byte-level BPE merge ranking, and decoding ordinary token pieces through the
  GPT-2 byte mapping used by `ds4_token_text`.
- M5.3 added `ds4-tokenizer-dump` for fixed plain-text cases and
  `python3 ds4-parity/compare_tokenizer_text.py`, which compares Rust token
  IDs and decoded token-piece bytes against the M5.2 current-C `text_cases`.
  Its negative tests cover missing token table, missing merges, token-bytes
  hash drift, merge hash drift, missing required special token, invalid UTF-8
  token strings, and merge-rank drift.
- M5.3 B300 extraction copied `ds4-parity/extract_tokenizer_fixture.py` to
  `ds4-rust-port-b300` and wrote
  `/tmp/ds4-tokenization-m5.3/tokenizer.gguf` from
  `/workspace/ds4/ds4flash.gguf`. The committed tokenizer-only GGUF fixture has
  129,280 tokens, 127,741 merges, size 4,722,720 bytes, and SHA256
  `b1e0d128bde9ea996fee335c9662e93707d2a68decaeb47a8dc5fb902bdbb025`.
- M5.3 local validation passed for `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 8 tests,
  `python3 -m py_compile ds4-parity/extract_tokenizer_fixture.py
  ds4-parity/compare_tokenizer_text.py`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 51 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 7 checks`.
- M5.3 Claude review returned `NO BLOCKERS` after checking the Rust tokenizer
  against the C byte encoding, JoyAI split rules, BPE merge loop, token text
  decoding, and comparator scope.
- M5.4 added Rust rendered-chat tokenization over the exact C
  `special_token_at` rendered-control table. `tokenize_rendered_chat` scans
  trusted rendered prompt bytes for BOS, EOS, User, Assistant, `<think>`,
  `</think>`, and `｜DSML｜`, emits those special token IDs, and tokenizes
  intervening spans through the existing JoyAI BPE path; plain `tokenize_text`
  remains separate so special-looking user text is not trusted as control text.
- M5.4 extended `ds4-tokenizer-dump` and
  `python3 ds4-parity/compare_tokenizer_text.py` to compare the M5.2
  `rendered_chat_cases` exactly for rendered prompt bytes, token IDs, and
  decoded token-piece bytes. Negative checks now include rendered special-token
  ID drift and rendered ordinary-piece drift.
- M5.4 local validation passed for `cargo fmt --all -- --check`,
  `cargo test --workspace` with 9 `ds4-gguf` tests,
  `python3 -m py_compile ds4-parity/compare_tokenizer_text.py`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 71 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 9 checks`.
- M5.4 Claude review returned `NO BLOCKERS`; after adding Rust dump `mode`
  fields and pinning them in the comparator, the follow-up Claude review also
  returned `NO BLOCKERS`.
- M5.5 added a Rust prompt renderer matching C `render_chat_prompt_text` for
  the committed M5.2 server prompt cases: tool schemas before system text,
  system/developer aggregation, user/tool/function message handling, assistant
  history turns, thinking disabled/high/max prefixes, DSML tool-call rendering,
  escaped tool-result closing tags, and pending assistant prefixes.
- M5.5 added direct Rust CLI token construction for the M5.2 `ds4_chat_*`
  operation fixtures, covering begin, Think Max prefix append, system/developer
  direct text, user/tool/function messages, assistant content, and assistant
  prefixes for high/max/none thinking modes.
- M5.5 extended `ds4-tokenizer-dump` and
  `python3 ds4-parity/compare_tokenizer_text.py` to compare every M5.2
  `server_request_cases` prompt byte string, rendered token IDs, decoded token
  pieces, CLI operation sequence, and CLI token stream. Negative checks now
  include server prompt-byte drift, server token-ID drift, CLI operation drift,
  and CLI token-piece drift.
- M5.5 local validation passed for `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 11 tests, `cargo test --workspace`,
  `./ds4_test --server`, `python3 -m py_compile
  ds4-parity/compare_tokenizer_text.py`, `git diff --check`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 154 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 13 checks`.
- M5.5 Claude review returned `NO BLOCKERS` after checking Rust prompt
  rendering against C role handling, thinking branches, DSML/tool-result
  escaping, CLI token construction, and comparator coverage.
- M5.6 was split into M5.6a and M5.6b before implementation because server
  generated-message DSML parsing and agent incremental DSML streaming have
  different oracle surfaces and comparator shapes. M5.6a owns server DSML
  formatting plus `parse_generated_message_ex`; M5.6b owns `agent_dsml_parse`
  chunk schedules and streaming state/event parity.
- M5.6 split validation passed for docs-only `git diff --name-only`, `git
  diff --check`, and non-interactive Claude review. Claude returned
  `NO BLOCKERS`.
- M5.6a added `./ds4-server --dump-dsml-oracle`, a no-model current-C DSML
  oracle covering rendered tool-call blocks, raw sampled DSML replay, JSON and
  string parameters, sentinel escaping, tool-result escaping,
  `parse_generated_message_ex`, and recoverable response parsing. The committed
  baseline lives at `ds4-parity/baselines/dsml/m5.6a/current-c.json` with size
  17,016 bytes and SHA256
  `3f20b4869a2035deab709e3299de91ccf151f46fa3524a8b389814ebbf880442`.
- M5.6a added Rust DSML formatting/parsing in `ds4_gguf::dsml`, routed the Rust
  prompt renderer's DSML and tool-result escaping through that module, and added
  `ds4-dsml-dump` plus `python3 ds4-parity/compare_dsml.py`.
- M5.6a validation passed for `arch -arm64 make ds4-server`,
  `./ds4-server --dump-dsml-oracle /tmp/ds4-dsml-final-c.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/compare_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6a/manifest.json --negative-test` with
  `DSML comparison: PASS, 410 checks`, `python3 -m py_compile
  ds4-parity/compare_dsml.py`, `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 14 tests, `cargo test --workspace`,
  `./ds4_test --server`, `python3 ds4-parity/compare_tokenizer_text.py
  --manifest ds4-parity/baselines/tokenization/m5.3/manifest.json
  --negative-test`, and `git diff --check`.
- M5.6a Claude review returned `NO BLOCKERS` after checking the Rust DSML
  parser/formatter against C tool-start ordering, raw block boundaries,
  sentinel escaping, entity escaping, JSON minification, response recovery,
  raw DSML replay, prompt-renderer routing, and comparator coverage.
- M5.6a implementation commit:
  `aaab1818710384e1c0b754c94f63dbf408ddb724`.
- M5.6b added `./ds4-agent --dump-agent-dsml-oracle`, a no-model current-C
  oracle for the agent incremental DSML parser. The fixture records whole,
  one-byte, marker-prefix, and parameter-boundary schedules where applicable,
  including raw/search buffer hex, parser states, current call, completed calls,
  parameter state, and error text after each chunk.
- M5.6b added Rust `ds4_gguf::agent_dsml`, `ds4-agent-dsml-dump`, and
  `python3 ds4-parity/compare_agent_dsml.py`. The committed C baseline lives at
  `ds4-parity/baselines/dsml/m5.6b/current-c.json` with size 887,559 bytes and
  SHA256
  `0b0f21728b0f5230dcbae5d3d2a99e272347ecdeac04fa57ca07ec00b9f00618`.
- M5.6b validation passed for `arch -arm64 make ds4-agent`,
  `./ds4-agent --dump-agent-dsml-oracle /tmp/agent-dsml-final-c.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/compare_agent_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6b/manifest.json --negative-test` with
  `agent DSML comparison: PASS, 37873 checks`, `python3 -m py_compile
  ds4-parity/compare_agent_dsml.py ds4-parity/compare_dsml.py`,
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf` with 16 tests,
  `cargo test --workspace`, `python3 ds4-parity/compare_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6a/manifest.json --negative-test`,
  `./ds4_test --server`, and `git diff --check`.
- M5.6b Claude review returned `NO BLOCKERS` after checking byte-vs-UTF-8
  behavior, mid-chunk done/error accumulation, close-tag variants, search-tail
  behavior, raw buffer accumulation, current/completed call transitions,
  fixture coverage, and no-model oracle startup.
- M5.6b implementation commit:
  `d6bade1d5bde4c72280bed0395322d85dfc30d5e`.
- M5.7 added `python3 ds4-parity/run_text_parity_report.py`, which runs the
  M5.2 token/prompt schema checker, M5.3-M5.5 Rust tokenizer/prompt
  comparator, M5.6a server DSML comparator, and M5.6b agent DSML comparator
  from committed fixtures without requiring the model locally.
- M5.7 records model-backed refreshes as skipped report items using exact
  `refresh_commands` from the M5.2 and M5.3 manifests, preserving the
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1` B300
  command path for future recapture.
- M5.7 wired the text report into
  `python3 ds4-parity/run_parity_report.py`, so the unified parity report now
  includes Milestone 5 text parity alongside earlier baseline comparators.
- M5.7 validation passed for `python3 -m py_compile
  ds4-parity/run_text_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_text_parity_report.py` with `summary: 4 passed, 2
  skipped, 0 failed`, JSON mode output with `ok: true`,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with
  `summary: 6 passed, 10 skipped, 0 failed`, `cargo test --workspace`, and
  `git diff --check`.
- M5.7 Claude review returned `NO BLOCKERS` after checking report
  integration, failure output, B300 refresh command fidelity, JSON/text output
  shape, status/TODO consistency, and accidental local model dependencies.
- M5.7 implementation commit:
  `3223f6e3a09f066873c5b8afc1b855adabad068d`.
- M6.1 split Milestone 6 into M6.2 through M6.7, including M6.6a/M6.6b,
  after reading the public sampling/logprob APIs in `ds4.h`, current C sampler
  and logprob math
  (`sample_argmax`, `sample_rng_next`, `sample_top_p_min_p`,
  `ds4_session_top_logprobs`, and `ds4_session_token_logprob`), CLI
  `--dump-logprobs` and perplexity surfaces, server decode stop handling,
  agent sampling, M0.3 official-vector tests, and M1.4 numeric comparator
  conventions.
- M6.1 defined separate oracle surfaces for no-model fixed-logits sampler math,
  Rust sampler/logprob math, B300 current-C session logits capture, Rust
  fixed-logits model-slice comparison, C decode stop policy fixtures, Rust
  decode stop policy comparison, and report integration.
- M6.1 validation passed for `git diff --check`; Claude review returned
  `NO BLOCKERS` after tightening M6.2 fixture ownership for source-named
  request-surface sampling tuples and splitting decode stop policy into M6.6a
  C oracle fixtures plus M6.6b Rust policy comparison.
- M6.1 implementation commit:
  `4d401ecf2a2f13e214927ab8ec05dc931d5e796e`.
- M6.2 added `./ds4-sampling-dump`, a no-model current-C fixed-logits sampler
  and logprob oracle that records selected token, actual sampler selection,
  consumed RNG state, effective sampling parameters, filtered candidate sets,
  input logits, top-logprob slices, token-logprob requests, and source-named
  request-surface sampling tuples.
- M6.2 committed
  `ds4-parity/baselines/sampling/m6.2/current-c.json` with size 16,556 bytes
  and SHA256
  `f3740560d562960ed3960f7aa07f50793b7b4338a31114b67f827ee9706493e0`.
- M6.2 routes oracle trace fields through the same helper used by
  `ds4_session_sample`, and request-surface sampling tuples now resolve through
  shared `ds4_sampling_params_*` helpers used by server, CLI, and agent
  defaults.
- M6.2 added `python3 ds4-parity/check_sampling_dump.py`, whose schema checker
  validates coverage for greedy ties, non-finite logits, temperature
  normalization, `top_p` clamping, `top_k` caps, `min_p` thresholds,
  full-vocab sampling, seeded RNG draws, top-logprob ordering, token-logprob
  requests, and request-surface parameter tuples. Its negative tests catch
  selected-token drift, missing request cases, candidate-list drift,
  top-logprob ordering drift, token-logprob schema drift, and manifest hash
  drift.
- M6.2 validation passed for `arch -arm64 make ds4-sampling-dump`,
  `./ds4-sampling-dump /tmp/ds4-sampling-m6.2-refresh.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/check_sampling_dump.py
  ds4-parity/baselines/sampling/m6.2/current-c.json --manifest
  ds4-parity/baselines/sampling/m6.2/manifest.json --negative-test` with
  `sampling schema: PASS, 1243 checks`, `sampling manifest: PASS, 7 checks`,
  and `sampling negative tests: PASS, 6 checks`, `python3 -m py_compile
  ds4-parity/check_sampling_dump.py`, `arch -arm64 make ds4_test`,
  `./ds4_test --server`, `arch -arm64 make cpu`, CPU
  `./ds4-sampling-dump` compared byte-for-byte with the committed baseline, and
  `git diff --check`.
- M6.2 Claude review returned `NO BLOCKERS` after checking sampler helper
  sharing, RNG bookkeeping, candidate ordering, request-surface helper
  plumbing, fake-session logprob safety, manifest checks, and negative-test
  coverage. Non-blocking notes: `matches_actual` now compares two calls through
  the same helper, and the schema checker is mostly shape/coverage while
  byte-for-byte baseline comparison carries M6.2 drift detection.
- M6.2 implementation commit:
  `b1b637978779700fb6ce7250e67eaa3eb23c19c6`.
- M6.3 added Rust no-model sampler/logprob math in `ds4_gguf::sampling`,
  including argmax, xorshift RNG, top-p/min-p/top-k filtering, full-vocab
  sampling, top-logprob slices, token-logprob scoring, and shared sampling
  parameter defaults.
- M6.3 added `cargo run --quiet -p ds4-gguf --bin ds4-sampling-dump-rs`, which
  emits the same fixed-logits case set as the M6.2 C oracle with selected
  tokens, RNG states, filtered candidates, and logprob scores.
- M6.3 added `python3 ds4-parity/compare_sampling.py --negative-test`, whose
  C/Rust comparator enforces exact selected token, RNG state, candidate IDs,
  candidate counts, request case coverage, top-logprob order, and token-logprob
  request shape, with `1e-5` ordinary absolute float tolerance and `1e-6`
  relative tolerance for large sentinel values. Negative tests catch selected
  token drift, RNG drift, candidate-list drift, logprob drift, and request
  coverage drift.
- M6.3 validation passed for `cargo test -p ds4-gguf sampling --quiet` with 3
  sampling tests passing, `python3 -m py_compile
  ds4-parity/compare_sampling.py`, `python3 ds4-parity/compare_sampling.py
  --negative-test --write-rust-dump /tmp/ds4-sampling-rust-from-comparator.json`
  with `sampling C/Rust comparator: PASS, 3241 checks` and `sampling C/Rust
  negative tests: PASS, 5 checks`, `cargo fmt --all -- --check`,
  `cargo test --workspace` with all workspace tests passing, and
  `git diff --check`.
- M6.3 Claude review returned `NO BLOCKERS` after checking Rust numeric edge
  cases, RNG semantics, candidate filtering order, top-logprob tie order,
  non-finite handling, request fixture coverage, and comparator negative tests.
  Non-blocking notes: top-p/full-vocab tied-logit fixture coverage is latent,
  Rust faithfully recomputes full-vocab weights during roulette like C, and
  greedy mode intentionally leaves effective params unclamped to match C.
- M6.3 implementation commit:
  `fea2ea3de57a260474d349d2536527bf2c16927a`.
- M6.4 added `./ds4-logits-dump`, a current-C model-backed oracle helper that
  runs official-vector prompts through `ds4_session_sync`,
  `ds4_session_argmax`, `ds4_session_top_logprobs`,
  `ds4_session_token_logprob`, and `ds4_session_eval`, then records selected
  tokens, token bytes, top-logprob slices, official-top deltas, and per-step
  full-logits SHA256s. The helper requires a 64-character lowercase
  `--model-sha256` and verifies the actual model file via `sha256sum` or
  `shasum -a 256` before opening the engine.
- M6.4 exposes `ds4_session_logits_data` so the dump helper can write a
  contiguous f32 logits blob without moving model execution into the helper.
- M6.4 captured B300 current-C artifacts on `ds4-rust-port-b300` in
  `hou2-prod1/default` after refreshing source into `/workspace/ds4` and
  building `make ds4-logits-dump CUDA_ARCH=native`. Capture command wrote
  `ds4-parity/baselines/sampling/m6.4/current-c.json` with size 19,535 bytes
  and SHA256
  `5343e5aa855305ca2092943e155a359db50a28216d44927d450d2e0cce82efd0`,
  plus `ds4-parity/baselines/sampling/m6.4/logits.f32le` with size
  4,654,080 bytes and SHA256
  `972636c24ff63534d3a7fb7b1360e78786dee0bdd111f1fde813aa758e1f1928`.
- M6.4 fixture contains 9 scored steps across `short_italian_fact`,
  `short_code_completion`, and `short_reasoning_plain`. `long_memory_archive`
  remains explicitly skipped for the existing API/official-graph mismatch, and
  `long_code_audit` is explicitly skipped because repeated B300 CUDA captures
  produced byte-different long-context logits even with deterministic-kernel
  probes.
- M6.4 added `python3 ds4-parity/check_session_logits_dump.py`, whose schema
  and hash checker validates model/backend identity, case coverage, prompt
  hashes, selected-token matches, top-logprob shape, selected/top scores
  recomputed from the f32le logits blob, official-top local matches and delta
  tolerances, contiguous per-step logits ranges, per-step logits SHA256s,
  whole-blob manifest SHA256 plus n_vocab/step counts, and exact
  temp-kubeconfig/context refresh commands.
- M6.4 validation passed for `arch -arm64 make ds4-logits-dump`,
  `python3 -m py_compile ds4-parity/check_session_logits_dump.py`, B300
  `make ds4-logits-dump CUDA_ARCH=native`, B300 capture with
  `./ds4-logits-dump --backend cuda -m /workspace/ds4/ds4flash.gguf -v
  tests/test-vectors/official.vec -o
  ds4-parity/baselines/sampling/m6.4/current-c.json -l
  ds4-parity/baselines/sampling/m6.4/logits.f32le --model-sha256
  efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  local and B300 checker runs with `session logits schema: PASS, 2356 checks`,
  `session logits manifest: PASS, 20 checks`, and `session logits negative
  tests: PASS, 11 checks`, `python3 ds4-parity/compare_logprob_numeric.py`
  with `summary: 5/5 sections passed, 528 checks`, `arch -arm64 make cpu`,
  and `git diff --check`.
- M6.5 adds `ds4-model-logits-dump-rs`, which reads the committed M6.4
  `logits.f32le` blob as contiguous f32 vocab slices, loads the M5.3
  tokenizer GGUF, runs Rust `sample_argmax` and `top_logprobs`, and emits a
  flat per-slice JSON dump with selected token IDs, rendered token bytes, and
  top-logprob scores.
- M6.5 adds `python3 ds4-parity/compare_model_logits.py`, which maps those
  flat Rust slices back to the M6.4 current-C case/step records and compares
  selected token, selected bytes, expected bytes, logits offsets, top-logprob
  ordering, top token IDs, token bytes, logits, and logprobs.
- M6.5 validation passed with `python3 -m py_compile
  ds4-parity/compare_model_logits.py`, `python3
  ds4-parity/compare_model_logits.py --negative-test` (`model logits C/Rust
  comparator: PASS, 2982 checks, max_abs_logit_delta=5.00000041e-08,
  max_abs_logprob_delta=5.00000006e-08`; negative tests `PASS, 6 checks`),
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf --bin
  ds4-model-logits-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M6.6a adds `./ds4-decode-policy-dump`, a no-model current-C decode stop
  policy oracle. The helper includes the `ds4_server.c` test surface so the
  fixture uses the real C stop-list, UTF-8 stream-hold, DSML marker, generated
  message parse, Anthropic stop-reason, and Responses status mapping helpers.
- M6.6a fixture `ds4-parity/baselines/sampling/m6.6a/current-c.json` covers
  CLI EOS/length, server OpenAI EOS/length/user-stop/stream-stop-tail/
  streaming-stop-hit/partial-UTF-8/stop-at-mid-UTF-8-boundary/
  tool-call-boundary, server Responses length mapping, server Anthropic tool
  mapping, and agent EOS/length defaults. The artifact is 17,000 bytes with
  SHA256
  `9d11d90a12e1ee4d16ac1d4aa8c971efe775a86202004db91aff8d452081a2b5`.
- M6.6a adds `python3 ds4-parity/check_decode_policy_dump.py`, whose schema
  and negative checks validate case coverage, request option records,
  generated text schedules, finish reason, visible bytes, streamed bytes,
  held-tail bytes, session invalidation, stop boundary offsets, tool-call
  boundary flags, and API finish mappings.
- M6.6a validation passed with `arch -arm64 make ds4-decode-policy-dump`,
  `./ds4-decode-policy-dump ds4-parity/baselines/sampling/m6.6a/current-c.json`,
  `python3 -m py_compile ds4-parity/check_decode_policy_dump.py`, `python3
  ds4-parity/check_decode_policy_dump.py --negative-test` (`decode policy
  schema: PASS, 969 checks`; manifest `PASS, 5 checks`; negative tests `PASS,
  10 checks`), `arch -arm64 make ds4_test`, `./ds4_test --server`, and
  `git diff --check`.
- M6.6b adds the Rust byte-oriented decode stop policy in
  `rust/ds4-gguf/src/decode_policy.rs` plus `ds4-decode-policy-dump-rs`.
  It mirrors the M6.6a generated-token schedules without introducing a Rust
  CLI/server runtime or reimplementing M5 DSML parsing; the tool case only
  observes complete tool-call marker boundaries.
- M6.6b adds `python3 ds4-parity/compare_decode_policy.py`, which runs the
  Rust dump and compares request records, schedules, finish reason, raw and
  visible bytes, streamed bytes, held tails, session invalidation, stop
  boundaries, tool-boundary flags, API finish mappings, and per-step streaming
  metadata against the committed M6.6a C oracle.
- M6.6b validation passed with `python3 -m py_compile
  ds4-parity/compare_decode_policy.py`, `python3
  ds4-parity/compare_decode_policy.py --negative-test` (`decode policy C/Rust
  comparator: PASS, 1059 checks`; negative tests `PASS, 10 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf decode_policy`, `cargo test
  -p ds4-gguf --bin ds4-decode-policy-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M6.7 adds `python3 ds4-parity/run_sampling_parity_report.py`, which runs the
  M6.2 current-C sampler/logprob checker, M6.3 Rust sampler comparator, M6.4
  committed session-logits fixture checker, M6.5 Rust model-logits comparator,
  M6.6a current-C decode policy checker, and M6.6b Rust decode-policy
  comparator.
- M6.7 records the model-backed M6.4 B300 session-logits recapture as a
  skipped report item using the exact `refresh_commands` from
  `ds4-parity/baselines/sampling/m6.4/manifest.json`; no other M6 local
  comparator is skipped by the M6 report.
- M6.7 wires the sampling/logprob report into
  `python3 ds4-parity/run_parity_report.py`. Validation passed with `python3
  -m py_compile ds4-parity/run_sampling_parity_report.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_sampling_parity_report.py` (`summary: 6 passed, 1 skipped, 0
  failed`), `python3 ds4-parity/run_sampling_parity_report.py --json |
  python3 -m json.tool`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` (`summary: 7 passed, 10 skipped, 0 failed`), `python3
  ds4-parity/run_parity_report.py --skip-local-oracles --json | python3 -m
  json.tool`, `cargo test --workspace`, and `git diff --check`.
- M7.1 split Milestone 7 into C KV header/policy oracle, Rust KV
  parser/policy, generic full-file round-trip coverage, per-extension trailer
  coverage, C on-disk session payload shape oracle, Rust payload header reader,
  KV replay/prefix decision comparator, B300 disk-KV and in-memory snapshot
  restore oracle, and report integration items. The first executable item is
  the no-model C KV header and policy oracle; the C on-disk session payload
  shape oracle is independently eligible because it depends on session payload
  code rather than KV header/policy work.
- M7.2 added `./ds4-kv-policy-dump`, a deterministic no-model current-C
  oracle for KVC header bytes, decoded fields, reason/key-kind mapping,
  little-endian helpers, SHA/path helpers, size budgeting, store-boundary
  selection, chat-anchor selection, continued-store targets, byte-prefix
  matching, eviction scoring with explicit `now`, text-prefix lookup, and M0.5
  parsed header fixture references.
- M7.2 added `python3 ds4-parity/check_kv_policy_dump.py`, whose schema,
  manifest, and negative checks validate the C oracle dump and the committed
  M0.5 `kv-header.tsv` row references.
- M7.2 local validation passed for `arch -arm64 make ds4-kv-policy-dump`,
  `./ds4-kv-policy-dump ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m py_compile ds4-parity/check_kv_policy_dump.py`, and
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema
  checks, `11` manifest checks, `7` negative checks), `arch -arm64 make`,
  `arch -arm64 make cpu`, deterministic CPU-regenerated dump comparison
  against the committed M7.2 artifact, `arch -arm64 make ds4_test`,
  `./ds4_test --server`, and `git diff --check`.
- M7.3 adds `rust/ds4-gguf/src/kv_policy.rs` for no-model KVC header
  parsing/writing, reason/key-kind helpers, SHA/path helpers, file-size
  budgeting, store-boundary selection, chat-anchor selection,
  continued-store target selection, byte-prefix matching, eviction scoring,
  and text-prefix entry selection.
- M7.3 adds `ds4-kv-policy-dump-rs`, which emits the same deterministic
  synthetic no-model policy fixture as the M7.2 C oracle with a Rust schema and
  source label.
- M7.3 adds `python3 ds4-parity/compare_kv_policy.py`, which runs the Rust
  dump and recursively compares it to the committed M7.2 C oracle while
  allowing only the schema/source labels to differ. It checks header bytes,
  decoded fields, reason and extension flags, SHA/path helpers, policy
  decisions, eviction scores, text-prefix selections, and M0.5 header rows.
- M7.3 local validation passed for `python3 -m py_compile
  ds4-parity/compare_kv_policy.py`, `python3
  ds4-parity/compare_kv_policy.py --negative-test` (`KV policy C/Rust
  comparator: PASS, 1488 checks`; negative tests `PASS, 8 checks`), `python3
  ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema checks,
  `11` manifest checks, `7` negative checks), `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-policy-dump-rs`, and `cargo test --workspace`.
- M7.4a adds `./ds4-kvc-file-dump`, a deterministic no-model current-C oracle
  for complete generic KVC file bytes: fixed header, text length, rendered-text
  bytes, opaque payload bytes, and opaque trailer bytes.
- M7.4a fixture `ds4-parity/baselines/kv/m7.4a/current-c.json` covers
  no-trailer, opaque trailer, visible-transcript flag without payload, empty
  text with trailer, no-budget/fitting-budget/over-budget size decisions, and
  malformed header/text/payload/trailer boundary records. The artifact is
  6,445 bytes with SHA256
  `ff37ba4a359b10d66199928a1936b10ec0adc43a17ceb7ba49c0ad3e02c8b7d7`.
- M7.4a adds Rust generic KVC full-file helpers in
  `rust/ds4-gguf/src/kv_policy.rs`; the reader keeps payload and trailer bytes
  opaque and treats all bytes after fixed header, text, and declared payload as
  generic trailer data.
- M7.4a adds `python3 ds4-parity/compare_kvc_file.py`, which runs
  `ds4-kvc-file-dump-rs` and compares complete file hex, read metadata,
  file-size budget records, malformed case outcomes, and trailer-size records
  against the committed C oracle.
- M7.4a local validation passed for `arch -arm64 make ds4-kvc-file-dump`,
  `./ds4-kvc-file-dump ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kvc_file.py`, `python3
  ds4-parity/compare_kvc_file.py --negative-test` (`KVC file C/Rust
  comparator: PASS, 277 checks`; negative tests `PASS, 8 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf kvc`, `cargo test -p
  ds4-gguf --bin ds4-kvc-file-dump-rs`, `cargo test --workspace`, `arch
  -arm64 make cpu`, CPU-regenerated `./ds4-kvc-file-dump` comparison against
  the committed M7.4a artifact, and `git diff --check`.
- M7.4b adds `./ds4-kv-trailer-dump`, a deterministic no-model current-C
  oracle for server-owned KVC trailer payloads using the real
  `kv_tool_map_serialized_size`, `kv_tool_map_write`, and
  `kv_tool_map_load_from_pos` helpers from `ds4_server.c`.
- M7.4b fixture `ds4-parity/baselines/kv/m7.4b/current-c.json` covers empty
  tool-map output, single-block output, text filtering, duplicate-block
  suppression, multiple IDs for one DSML block, UTF-8 bytes with a long ID,
  disabled exact replay, visible-transcript extension flags without payload
  bytes, and malformed trailer load/decode boundaries. The artifact is 13,232
  bytes with SHA256
  `c5f73f2ea0f712e5fa1f2ee57666e1907304324d5334f398356c90ca40401d73`.
- M7.4b adds Rust tool-map trailer helpers in
  `rust/ds4-gguf/src/kv_policy.rs`; the writer scans DSML tool-call blocks,
  suppresses duplicate blocks, mirrors C's reverse insertion order for
  multiple IDs on one block, and the reader preserves partial decoded entries
  on malformed trailers.
- M7.4b adds `python3 ds4-parity/compare_kv_trailer.py`, which runs
  `ds4-kv-trailer-dump-rs` and compares trailer bytes, decoded entries,
  load-count behavior, wanted-ID filtering, extension flag records, and
  malformed trailer categories against the committed C oracle.
- M7.4b local validation passed for `arch -arm64 make ds4-kv-trailer-dump`,
  `./ds4-kv-trailer-dump ds4-parity/baselines/kv/m7.4b/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4b/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kv_trailer.py`, `python3
  ds4-parity/compare_kv_trailer.py --negative-test` (`KV trailer C/Rust
  comparator: PASS, 432 checks`; negative tests `PASS, 8 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf tool_map`, `cargo test -p
  ds4-gguf --bin ds4-kv-trailer-dump-rs`, `cargo test --workspace`, `arch
  -arm64 make cpu`, CPU-regenerated `./ds4-kv-trailer-dump` comparison against
  the committed M7.4b artifact, and `git diff --check`.
- M7.5 adds `./ds4-session-payload-dump`, a deterministic no-model current-C
  oracle for the DSV4 session payload shape using the real payload constants,
  fixed DS4 model layout constants, size formula, and `ds4_session_load_payload`
  rejection behavior on synthetic CPU fixtures.
- M7.5 fixture `ds4-parity/baselines/kv/m7.5/current-c.json` records the
  13-u32 DSV4 header, little-endian magic bytes, field order, body section
  order, synthetic payload byte accounting, header rejection categories,
  body/trailing/truncated/row-count rejection categories, and M0.5 B300
  model-backed payload size/hash records. The artifact is 19,774 bytes with
  SHA256 `479d05d7274fde43ea5a2676895637639113534ee3f7bbb2723d032756b10806`.
- M7.5 adds `python3 ds4-parity/check_session_payload_shape.py`, which reruns
  the C payload dump, compares it to the committed fixture, verifies the M0.5
  payload records against committed logs and hashes, and checks that exact B300
  recapture commands preserve the temp kubeconfig, explicit context, pod, and
  model path.
- M7.5 local validation passed for `arch -arm64 make ds4-session-payload-dump`,
  `./ds4-session-payload-dump | python3 -m json.tool`, baseline generation via
  `python3 ds4-parity/check_session_payload_shape.py --write-baseline
  ds4-parity/baselines/kv/m7.5/current-c.json`, `python3 -m json.tool
  ds4-parity/baselines/kv/m7.5/current-c.json`, `python3 -m py_compile
  ds4-parity/check_session_payload_shape.py`, and `python3
  ds4-parity/check_session_payload_shape.py --negative-test` (`Session payload
  shape oracle: PASS, 552 checks`; negative tests `PASS, 8 checks`), `arch
  -arm64 make cpu`, and `git diff --check`.
- M7.6 adds `rust/ds4-gguf/src/session_payload.rs`, a no-runtime-restore Rust
  reader for DSV4 payload headers and structural body boundaries. It mirrors
  C's combined bad-magic/bad-version `unsupported-version` behavior, fixed DS4
  layout checks, CPU layout/cap checks, row-count validation, truncated body
  rejection, and trailing-payload rejection.
- M7.6 adds `ds4-session-payload-dump-rs`, which emits the Rust structural
  surface for the same synthetic M7.5 cases without loading a model or claiming
  tensor/session restore support.
- M7.6 adds `python3 ds4-parity/compare_session_payload.py`, which compares
  Rust output to the M7.5 current-C structural oracle and checks the M0.5
  payload/hash/B300-command records as fixture preconditions.
- M7.6 local validation passed for `cargo fmt --all -- --check`, `python3 -m
  py_compile ds4-parity/compare_session_payload.py`, `python3
  ds4-parity/compare_session_payload.py --negative-test` (`Session payload
  C/Rust structural comparator: PASS, 347 checks`; M0.5 fixture contract
  `PASS, 17 checks`; negative tests `PASS, 8 checks`), `cargo test -p
  ds4-gguf session_payload`, `cargo test -p ds4-gguf --bin
  ds4-session-payload-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M7.7 adds Rust cache-replay helpers in `rust/ds4-gguf/src/kv_policy.rs`
  for live-prefix reuse, disk-text restore accounting, cache write token
  calculation, and byte-prefix effective prompt suffix construction.
- M7.7 adds `ds4-kv-replay-dump-rs`, a no-model Rust replay fixture for the
  committed M0.5 cold/disk restore cases and M0.4 DSML/cache-continuation
  cases.
- M7.7 adds `ds4-parity/baselines/kv/m7.7/current-c.json`, derived from the
  committed M0.4/M0.5 traces and responses. It records M5 prompt-rendering
  artifact hashes as fixture preconditions, six replay cases, disk-cache
  reason/key/rendered-text fields, token-window hashes for mismatch cases,
  DSML tool-call records, and effective prompt suffix byte hex for disk and
  memory-prefix restores.
- M7.7 adds `python3 ds4-parity/compare_kv_replay.py`, which regenerates the
  C replay oracle from committed artifacts, fails M5 hash drift as a
  precondition, compares Rust replay output, and checks the M7.3 Rust policy
  dump's M0.5 KVC header rows.
- M7.7 local validation passed for `cargo fmt --all -- --check`, `python3 -m
  py_compile ds4-parity/compare_kv_replay.py`, JSON validation for
  `ds4-parity/baselines/kv/m7.7/current-c.json` and `manifest.json`, `python3
  ds4-parity/compare_kv_replay.py --negative-test` (`KV replay C fixture
  preconditions: PASS, 333 checks`; `KV replay C/Rust comparator: PASS, 273
  checks`; `KV replay Rust policy precondition: PASS, 14 checks`; manifest
  `PASS, 6 checks`; negative tests `PASS, 6 checks`), `cargo test -p
  ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-replay-dump-rs`, `cargo test --workspace`, and `git diff --check`.
- M7.8 adds `./ds4-restore-dump`, a current-C model-backed restore oracle
  helper for the recorded B300 model. It captures disk DSV4 payload restore and
  in-memory `ds4_session_snapshot` restore for seed and continuation prompts,
  recording selected tokens, top-20 logprob slices, max score deltas, payload
  sizes, payload/snapshot hashes, header prefixes, fixture hashes, backend
  identity, and raw-payload non-commit policy.
- M7.8 B300 validation on `ds4-rust-port-b300` in `hou2-prod1` refreshed the
  uncommitted M7.8 source delta into `/workspace/ds4`, built
  `ds4-restore-dump` with `CUDA_ARCH=native`, opened
  `/workspace/ds4/ds4flash.gguf` with SHA256
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`, and
  captured `ds4-parity/baselines/kv/m7.8/current-c.json`.
- M7.8 committed `current-c.json` is 15,715 bytes with SHA256
  `5a50459507e7750179f187a1ea177ac8f0f44c8e2c41ea6a08ee922e861e7574`;
  raw restore bodies remain uncommitted on the capture workspace and are
  represented by hashes plus exact B300 refresh commands in
  `ds4-parity/baselines/kv/m7.8/manifest.json`.
- M7.8 validation passed on B300 for `python3
  ds4-parity/check_restore_dump.py
  ds4-parity/baselines/kv/m7.8/current-c.json --negative-test` (`restore
  oracle schema: PASS, 1448 checks`; negative tests `PASS, 6 checks`).
- M7.8 local validation passed for `arch -arm64 make ds4-restore-dump`,
  `python3 -m py_compile ds4-parity/check_restore_dump.py`, manifest generation
  with `--write-manifest`, `python3 ds4-parity/check_restore_dump.py
  ds4-parity/baselines/kv/m7.8/current-c.json --manifest
  ds4-parity/baselines/kv/m7.8/manifest.json --negative-test` (`restore oracle
  schema: PASS, 1448 checks`; manifest `PASS, 13 checks`; negative tests
  `PASS, 6 checks`), `python3 -m json.tool` for both committed M7.8 JSON files,
  `arch -arm64 make cpu`, and `git diff --check`.
- M7.9 adds `python3 ds4-parity/run_kv_parity_report.py`, which builds the
  local no-model `ds4-session-payload-dump` helper, runs M7.2 through M7.8
  KV/snapshot comparator commands, emits text or machine-readable JSON, and
  skips only the model-backed M7.8 B300 restore recapture with the manifest
  refresh commands.
- M7.9 wires the Milestone 7 report into
  `python3 ds4-parity/run_parity_report.py` as the `M7.9 KV/snapshot parity
  report` comparator item, so the unified report now covers M1, M4, M5, M6, and
  M7 comparator families.
- M7.9 validation passed for `python3 -m py_compile
  ds4-parity/run_kv_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_kv_parity_report.py` (`summary: 9 passed, 1 skipped,
  0 failed`), `python3 ds4-parity/run_kv_parity_report.py --json | python3 -m
  json.tool >/dev/null`, `python3 ds4-parity/run_parity_report.py` (`summary:
  13 passed, 5 skipped, 0 failed`), `cargo test --workspace`, and
  `git diff --check`.
- M8.1 split Milestone 8 into CLI parity work items after inspecting
  `ds4_cli.c` usage text, parser branches, prompt building, one-shot
  generation, `--dump-tokens`, `--dump-logprobs`, `--perplexity-file`,
  `--inspect`, imatrix capture, debug/test mode flags, thinking controls, and
  REPL command handling.
- M8.1 added roadmap items M8.2 through M8.16, covering current-C parse/error
  oracle, Rust parse/error parity, token/prompt diagnostics, logprob/perplexity
  diagnostics, inspect output, imatrix capture, one-shot generation,
  interactive PTY transcripts, and CLI report integration. The next executable
  item is M8.2 current-C CLI parse/error oracle.
- M8.1 validation passed for docs/state-only diff inspection and
  `git diff --check`.
- M8.2 adds `python3 ds4-parity/check_cli_parse_dump.py`, a local no-model
  current-C CLI parser/error oracle checker. It captures 20 cases for `--help`,
  missing option values, unknown options, invalid numeric/float/backend values,
  duplicate prompt sources, missing prompt files, `--server`,
  `--metal-graph-generate`, `--dump-tokens` without a prompt, imatrix option
  coupling, and `--perplexity-file` prompt-source rejection.
- M8.2 committed `ds4-parity/baselines/cli/m8.2/current-c.json` is 23,626 bytes
  with SHA256 `d395a55e92957b84deb4cb43d4b70c5a2e78bac363b7d11be1200f1d3601fa22`.
  The checker stores stdout/stderr bytes and hashes while asserting stable help
  anchors, error categories, exact option names, exit status, and no model-load
  markers.
- M8.2 validation passed for `arch -arm64 make ds4`, baseline generation with
  `python3 ds4-parity/check_cli_parse_dump.py --write-baseline
  ds4-parity/baselines/cli/m8.2/current-c.json --write-manifest
  ds4-parity/baselines/cli/m8.2/manifest.json`, `python3 -m py_compile
  ds4-parity/check_cli_parse_dump.py`, `python3
  ds4-parity/check_cli_parse_dump.py
  ds4-parity/baselines/cli/m8.2/current-c.json --manifest
  ds4-parity/baselines/cli/m8.2/manifest.json --negative-test` (`CLI parse
  oracle: PASS, 369 checks`; manifest `PASS, 11 checks`; negative tests `PASS,
  7 checks`), `python3 -m json.tool` for both M8.2 JSON files, and
  `git diff --check`.
- M8.3 adds `rust/ds4-gguf/src/cli_parse.rs` and `ds4-cli-parse-rs`, a
  parser-only Rust CLI surface for the committed M8.2 no-model argument matrix.
  It emits the same exit categories for help, parser errors, removed/deprecated
  flags, imatrix/perplexity coupling, and `--dump-tokens` without a prompt, and
  deliberately exits with an unsupported parser-only status if a model-backed
  path is reached outside the M8.3 fixture.
- M8.3 adds `python3 ds4-parity/compare_cli_parse.py`, which builds
  `ds4-cli-parse-rs`, validates the M8.2 C fixture preconditions, and compares
  Rust exit status, stdout/stderr emptiness, stable help anchors, stderr
  category anchors, and no-model-load markers.
- M8.3 validation passed for `cargo fmt --all -- --check`, `python3 -m
  py_compile ds4-parity/compare_cli_parse.py`, `cargo test -p ds4-gguf
  cli_parse` (3 parser tests passed), `python3
  ds4-parity/compare_cli_parse.py --negative-test` (`CLI parse C fixture
  preconditions: PASS, 224 checks`; `CLI parse C/Rust comparator: PASS, 244
  checks`; negative tests `PASS, 5 checks`), `cargo test --workspace`, and
  `git diff --check`.
- M8.4 adds `python3 ds4-parity/check_cli_token_dump.py`, B300 fixtures under
  `ds4-parity/baselines/cli-fixtures/m8.4/`, and the committed current-C
  `--dump-tokens` oracle at `ds4-parity/baselines/cli/m8.4/current-c.json`.
  The checker captures raw stdout/stderr bytes as base64 plus hashes, parses
  the first-line token ID list, records prompt bytes and prompt-file hashes, and
  asserts that `--system`, empty `--system`, `--think`, both `--think-max`
  context thresholds, and `--nothink` are ignored by the early dump-token exit.
- M8.4 committed `ds4-parity/baselines/cli/m8.4/current-c.json` is 18,870 bytes
  with SHA256 `87d427fb88563c15a07e859618fd585c6cb847bc77add556da89edf504bfb51c`.
- M8.4 B300 validation passed after refreshing `/workspace/ds4` on
  `ds4-rust-port-b300`, building `make ds4 CUDA_ARCH=native`, capturing the
  baseline, and running `python3 ds4-parity/check_cli_token_dump.py
  ds4-parity/baselines/cli/m8.4/current-c.json --manifest
  ds4-parity/baselines/cli/m8.4/manifest.json --negative-test` (`CLI token dump
  oracle: PASS, 306 checks`; manifest `PASS, 18 checks`; negative tests `PASS,
  8 checks`).
- M8.5 refactors `rust/ds4-gguf/src/cli_parse.rs` to expose the parsed model
  path, prompt text, and `--dump-tokens` flag while preserving the M8.3
  parser-only surface. It adds `ds4-cli-token-dump-rs`, which loads the M5.3
  tokenizer GGUF, tokenizes the raw prompt with `tokenize_rendered_chat`, and
  writes the C diagnostic format.
- M8.5 adds `Ds4Tokenizer::token_text_bytes` for diagnostics that need raw
  tokenizer table bytes. The C `dump_tokens_fp` output uses those raw token text
  bytes (`Ġtoken` style), not decoded token bytes (` token` style); existing
  decoded `token_bytes` behavior is unchanged.
- M8.5 adds `python3 ds4-parity/compare_cli_token_dump.py`, which validates the
  M8.4 C fixture, checks the M5.3 tokenizer fixture hash, substitutes that
  tokenizer fixture for the B300 model path, and compares Rust/C exit status,
  stdout bytes, stderr bytes, and token IDs exactly.
- M8.5 validation passed for `cargo fmt --all -- --check`, `cargo test -p
  ds4-gguf cli_parse` (4 parser tests passed), `cargo test -p ds4-gguf
  token_text_decodes_gpt2_byte_mapping` (1 tokenizer diagnostic test passed),
  `python3 -m py_compile ds4-parity/compare_cli_token_dump.py`, `python3
  ds4-parity/compare_cli_token_dump.py --skip-build --negative-test` (`CLI
  token dump tokenizer fixture: PASS, 3 checks`; C fixture preconditions `PASS,
  166 checks`; C/Rust comparator `PASS, 65 checks`; negative tests `PASS, 5
  checks`), `python3 ds4-parity/compare_cli_parse.py --skip-build
  --negative-test` (`CLI parse C fixture preconditions: PASS, 224 checks`;
  C/Rust comparator `PASS, 244 checks`; negative tests `PASS, 5 checks`), and
  `cargo test --workspace`.
- M8.6 adds `python3 ds4-parity/check_cli_diagnostics_dump.py`, fixtures under
  `ds4-parity/baselines/cli-fixtures/m8.6/`, and the current-C diagnostic
  artifact at `ds4-parity/baselines/cli/m8.6/current-c.json`.
- M8.6 captures four B300 CLI diagnostic cases: inline `--dump-logprobs`
  (`top_k=3`, 2 generated steps, selected IDs 2581 and 1309), prompt-file
  `--dump-logprobs` (`top_k=5`, 1 generated step, selected ID 2581), a bad
  logprob output path error, and `--perplexity-file` (`tokens=69`, `scored=4`,
  `nll=0.158310216`, `avg_nll=0.039577554`, `ppl=1.040371181`).
- M8.6 committed `ds4-parity/baselines/cli/m8.6/current-c.json` is 16,161 bytes
  with SHA256 `838646513c85069db6ecc34ae5b8729257ecd89e7a6b28002e5e6e4f3edc429c`.
- M8.6 B300 validation passed after refreshing `/workspace/ds4` on
  `ds4-rust-port-b300`, building `make ds4 CUDA_ARCH=native`, capturing the
  baseline, and running `python3 ds4-parity/check_cli_diagnostics_dump.py
  ds4-parity/baselines/cli/m8.6/current-c.json --manifest
  ds4-parity/baselines/cli/m8.6/manifest.json --negative-test` (`CLI
  diagnostics oracle: PASS, 267 checks`; manifest `PASS, 12 checks`; negative
  tests `PASS, 7 checks`). Local revalidation of the copied artifact reported
  the same PASS counts; `python3 -m py_compile
  ds4-parity/check_cli_diagnostics_dump.py`, `python3 -m json.tool` for both
  M8.6 JSON files, and `git diff --check` also passed.
- M8.7 is not implemented as model-backed parity because the Rust tree does not
  yet expose a model/session execution boundary. Current Rust evidence is
  tokenizer/prompt/fixed-logits support in `rust/ds4-gguf`, a model-logits
  replay binary over captured logits, and low-level GPU tensor wrappers in
  `rust/ds4-gpu`; there is no Rust `ds4_engine`/`ds4_session` equivalent that
  can run the M8.6 CLI prompts.
- M8.7 has been split in `RUST_PORT_ROADMAP.md` into a runtime-boundary
  prerequisite and the actual CLI diagnostic output parity item. The original
  M8.7 is blocked until that runtime-boundary prerequisite exists; skipping to
  M8.8 avoids claiming replay-only artifact handling as execution parity.
- M8.8 adds `ds4-parity/check_cli_inspect_dump.py` and the current-C inspect
  artifact at `ds4-parity/baselines/cli/m8.8/current-c.json`.
- M8.8 captures two B300 CLI inspect cases: plain `--cuda --inspect` and an
  inspect case with prompt/context/generation controls that must keep the same
  summary stdout and avoid context-buffer, think-max, generation, perplexity,
  or imatrix logs.
- M8.8 committed `ds4-parity/baselines/cli/m8.8/current-c.json` is 9,087 bytes
  with SHA256 `613a03b604204831a04d5c74be15f3c4ecdf33f990aef43fcb3e92e6fe894ca1`.
- M8.8 B300 validation passed after refreshing `/workspace/ds4` on
  `ds4-rust-port-b300`, building `make ds4 CUDA_ARCH=native`, capturing the
  baseline, and running `python3 ds4-parity/check_cli_inspect_dump.py
  ds4-parity/baselines/cli/m8.8/current-c.json --manifest
  ds4-parity/baselines/cli/m8.8/manifest.json --negative-test` (`CLI inspect
  oracle: PASS, 112 checks`; manifest `PASS, 20 checks`; negative tests `PASS,
  8 checks`). Local revalidation of the copied artifact reported the same PASS
  counts; `python3 -m py_compile ds4-parity/check_cli_inspect_dump.py`,
  `python3 -m json.tool` for both M8.8 JSON files, and `git diff --check` also
  passed.
- M8.9 is not implemented as inspect parity because the Rust tree only accepts
  `--inspect` as a recognized parser option and still returns the parser-only
  model-backed-path stub from `parse_cli`; there is no Rust engine-open or
  engine-summary boundary.
- M8.9 has been split in `RUST_PORT_ROADMAP.md` into an inspect runtime-boundary
  prerequisite and the actual CLI inspect output surface item. The original
  M8.9 is blocked until the runtime-boundary prerequisite exists.
- M8.9a adds `rust/ds4-engine`, a Rust `Engine` wrapper over
  `ds4_engine_open`/`ds4_engine_summary`/`ds4_engine_close`, the
  `ds4-inspect-runtime-rs` runtime binary, and
  `ds4-parity/compare_cli_inspect_runtime.py`.
- M8.9a local validation passed for `cargo fmt --all -- --check`, `cargo test
  --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_inspect_runtime.py`, and `git diff --check`.
- M8.9a B300 validation used a temporary Rust 1.95.0 toolchain under
  `/tmp/ds4-cargo` and `/tmp/ds4-rustup`, built the CUDA-backed Rust binary,
  and passed `python3 ds4-parity/compare_cli_inspect_runtime.py
  ds4-parity/baselines/cli/m8.8/current-c.json --candidate-binary
  target/debug/ds4-inspect-runtime-rs --negative-test` (`CLI inspect runtime
  comparator: PASS, 68 checks`; negative tests `PASS, 5 checks`).
- M8.9b adds `ds4-cli-inspect-rs`, which routes parsed Rust CLI `--inspect`
  configuration through the M8.9a `Engine` boundary instead of replaying the
  M8.8 JSON artifact.
- M8.9b extends `CliConfig` so the Rust parser preserves inspect dispatch,
  backend selection, `--warm-weights`, and `--quality` for runtime-boundary
  consumers while keeping non-inspect model-backed paths stubbed.
- M8.9b updates `ds4-parity/compare_cli_inspect_runtime.py` with
  `--use-case-argv`, so the comparator can run the exact committed M8.8 CLI
  argv through the Rust binary, including prompt/control flags that current C
  ignores in inspect mode.
- M8.9b local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-gguf cli_parse::tests::config_retains_inspect_backend_and_runtime_flags`,
  `cargo test -p ds4-engine`, `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_inspect_runtime.py`, and `git diff --check`.
- M8.9b B300 validation used the temporary Rust 1.95.0 toolchain under
  `/tmp/ds4-cargo` and `/tmp/ds4-rustup`, built `ds4-cli-inspect-rs`, and
  passed `python3 ds4-parity/compare_cli_inspect_runtime.py
  ds4-parity/baselines/cli/m8.8/current-c.json --candidate-binary
  target/debug/ds4-cli-inspect-rs --use-case-argv --negative-test` (`CLI
  inspect comparator: PASS, 68 checks`; negative tests `PASS, 5 checks`).
- M8.10 is not implemented as an output-hash oracle because current C forces
  `--imatrix-out` to the Metal backend in `ds4_cli.c`, and
  `ds4_engine_collect_imatrix` requires `DS4_BACKEND_METAL` plus `metal_ready`.
  The B300 model host is a CUDA build, so it cannot write a valid imatrix
  `.dat` artifact today.
- M8.10 has been split in `RUST_PORT_ROADMAP.md` into M8.10a, the completed
  feasibility guard, and M8.10b, the blocked current-C imatrix output oracle.
- M8.10a B300 proof refreshed `/workspace/ds4` to
  `bfd96275d077e33970d368a92a99963451e3384d`, built `make ds4
  CUDA_ARCH=native`, wrote a tiny `/tmp/m8.10-imatrix-dataset.txt`, and ran
  `./ds4 -m /workspace/ds4/ds4flash.gguf --ctx 64 --imatrix-dataset
  /tmp/m8.10-imatrix-dataset.txt --imatrix-out /tmp/m8.10-imatrix.dat
  --imatrix-max-prompts 1 --imatrix-max-tokens 16`.
- M8.10a B300 proof returned exit 1, stdout bytes 0, stderr
  `ds4: context buffers 22.51 MiB (ctx=64, backend=metal, prefill_chunk=64,
  raw_kv_rows=256, compressed_kv_rows=18)` followed by
  `ds4: Metal backend requested but this build is linked with CUDA, not Metal`,
  and no `/tmp/m8.10-imatrix.dat` output file.
- M8.10a local availability check found no `ds4flash.gguf` or imatrix GGUF in
  the workspace on this `x86_64` host with `51539607552` bytes of RAM, so a
  local Metal capture of the recorded q2-imatrix model is not currently
  feasible.
- M8.11 is blocked because it requires the committed M8.10b current-C imatrix
  output fixture. It should not be implemented against the M8.10a failure proof
  or a synthetic `.dat` substitute.
- M8.12 is the next runnable roadmap item because it captures current-C
  one-shot generation transcripts on the B300 CUDA model host and does not
  depend on the blocked imatrix output oracle.
- M8.12 has been split in `RUST_PORT_ROADMAP.md` into M8.12a core prompt,
  thinking-control, seeded-sampling, and context transcript capture, followed by
  M8.12b advanced runtime-control coverage for MTP, directional steering,
  quality, warm-weights, threads, and backend-option behavior.
- M8.12a adds `ds4-parity/check_cli_generation_dump.py`, fixture
  `ds4-parity/baselines/cli-fixtures/m8.12a/prompt_file.txt`, and the
  current-C transcript artifact at
  `ds4-parity/baselines/cli/m8.12a/current-c.json`.
- M8.12a committed current-C artifact size is 25,651 bytes with SHA256
  `d56ab4566471731abdb55769b23dceb9c9e42ad1d40ffc18bbb201a343861628`.
- M8.12a captures five B300 CLI one-shot cases: greedy inline `--nothink`,
  prompt-file `--think`, low-context `--think-max` downgrade warning, seeded
  non-greedy sampling with seed `12345`, and a too-small-context error case.
- M8.12a case stdout hashes are: `greedy_inline_nothink`
  `862550215bb33a4e6f591f4c1c52fcd03dc98022f1acad87ce807f7d58b8c03c`,
  `prompt_file_think`
  `e566cf8e60978ac10a300c2503a68e03edd6d162fb11e2496057d57346660af0`,
  `think_max_downgrade`
  `c36d4b240ac10dcf300ba8d9d5aafc33957b8c1976c7f5ae2b26e654701e1b74`,
  `seeded_sampling_nothink`
  `c72869b348ae66d5a5267de18ed40cd032dc13a908bc32ad10d30f4d1b550c39`,
  and `ctx_too_small`
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- M8.12a B300 validation passed after copying the uncommitted checker/fixture to
  `/workspace/ds4`, running the capture/checker command, and then copying
  `current-c.json` and `manifest.json` back. Local revalidation passed
  `python3 ds4-parity/check_cli_generation_dump.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --manifest
  ds4-parity/baselines/cli/m8.12a/manifest.json --negative-test` with oracle
  `PASS, 156 checks`, manifest `PASS, 17 checks`, and negative tests `PASS, 5
  checks`.
- M8.12b adds `ds4-parity/check_cli_runtime_controls_dump.py` and the
  current-C runtime-control transcript artifact at
  `ds4-parity/baselines/cli/m8.12b/current-c.json`.
- M8.12b committed current-C artifact size is 22,815 bytes with SHA256
  `b9e5aca6f745ce846d1daceed820f5c6e3aa06d80f87674b24017df285fe221e`.
- M8.12b captures five B300 CLI runtime-control cases:
  `--backend cuda --quality -t 2`, `--warm-weights`, directional steering via
  `dir-steering/out/verbosity.f32`, blocked `--backend metal` on the CUDA
  build, and blocked `--mtp /workspace/ds4/missing-mtp.gguf`.
- M8.12b records the directional steering support artifact
  `dir-steering/out/verbosity.f32` as 704,512 bytes with SHA256
  `6414573b7d88822e16e6fe5972386ef2f1e51fc8502fe5849c4a611afad50cdd`,
  and records that no MTP GGUF is present in the B300 workspace.
- M8.12b B300 validation passed after refreshing `/workspace/ds4` to
  `dad2b7d95cb0ad8fcdce6044dac860ae9bf68a44`, copying the uncommitted
  checker, rebuilding `ds4`, running the capture/checker command, and copying
  `current-c.json` and `manifest.json` back. Local revalidation passed
  `python3 ds4-parity/check_cli_runtime_controls_dump.py
  ds4-parity/baselines/cli/m8.12b/current-c.json --manifest
  ds4-parity/baselines/cli/m8.12b/manifest.json --negative-test` with oracle
  `PASS, 158 checks`, manifest `PASS, 16 checks`, and negative tests `PASS, 5
  checks`.
- M8.13 has been split in `RUST_PORT_ROADMAP.md` into M8.13a argmax one-shot
  runtime boundary, M8.13b session sampling runtime boundary, M8.13c core CLI
  transcript surface, and M8.13d runtime-control CLI transcript surface.
- M8.13 source inspection found that `rust/ds4-engine/src/lib.rs` currently
  wraps `ds4_engine_open`, `ds4_engine_summary`, and `ds4_engine_close`, but
  does not expose prompt encoding, generated-token text, argmax generation, or
  session sampling.
- M8.13 source inspection found that
  `rust/ds4-engine/src/bin/ds4-cli-inspect-rs.rs` still exits 99 for the
  non-inspect model-backed path, so it cannot produce M8.12a/M8.12b one-shot
  transcripts.
- M8.13 source inspection found the needed current-C runtime APIs in `ds4.h`:
  `ds4_encode_chat_prompt`, `ds4_tokenize_rendered_chat`,
  `ds4_engine_generate_argmax`, `ds4_token_text`, `ds4_tokens_free`,
  `ds4_session_create`, `ds4_session_sync`, `ds4_session_sample`, and
  `ds4_session_eval`.
- M8.13a adds safe Rust ownership for `ds4_tokens`, `ThinkMode`, C context
  memory estimates, prompt encoding through `ds4_encode_chat_prompt` or
  `ds4_tokenize_rendered_chat`, and argmax generation through
  `ds4_engine_generate_argmax` with Rust callbacks that convert generated token
  IDs through `ds4_token_text` and free the C-allocated pieces.
- M8.13a adds `ds4-argmax-runtime-rs`, a narrow runtime-boundary binary for
  greedy one-shot generation. It accepts the M8.12a greedy/error argv surface,
  logs context memory and Think Max downgrade warnings, but rejects nonzero
  `--temp` so seeded sampling remains in M8.13b.
- M8.13a adds `ds4-parity/compare_cli_argmax_runtime.py`, which runs
  `target/debug/ds4-argmax-runtime-rs` against the M8.12a current-C
  `greedy_inline_nothink`, `prompt_file_think`, `think_max_downgrade`, and
  `ctx_too_small` cases. It excludes `seeded_sampling_nothink` for M8.13b.
- M8.13a B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  overlaying the M8.13a files on the pushed M8.13 split commit, building with
  `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-argmax-runtime-rs`, and running
  `python3 ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-argmax-runtime-rs --negative-test`.
- M8.13a B300 comparator reported `CLI argmax runtime comparator: PASS, 109
  checks` and `CLI argmax runtime negative tests: PASS, 4 checks`.
- M8.13a local validation passed for `cargo fmt --all -- --check`, focused
  `cargo test -p ds4-engine token_printer -- --nocapture`, `cargo build -p
  ds4-engine --bin ds4-argmax-runtime-rs`, full `cargo test --workspace`,
  `python3 -m py_compile ds4-parity/compare_cli_argmax_runtime.py`, and
  `git diff --check`.
- M8.13b adds `SamplingOptions` and a Rust session-backed generation path that
  calls current-C `ds4_session_create`, `ds4_session_sync`,
  `ds4_session_sample`, `ds4_session_eval`, `ds4_session_ctx`,
  `ds4_session_pos`, and `ds4_token_eos`. It preserves Rust ownership of the
  generated stdout buffer and frees the C session with `ds4_session_free`.
- M8.13b adds `ds4-session-runtime-rs`, a narrow seeded non-greedy runtime
  binary. It accepts the M8.12a seeded sampling argv surface, requires `--seed`,
  and rejects `--temp 0` so greedy generation remains owned by M8.13a.
- M8.13b adds `ds4-parity/compare_cli_session_runtime.py`, which runs
  `target/debug/ds4-session-runtime-rs` against the M8.12a
  `seeded_sampling_nothink` current-C case with seed `12345`; it does not cover
  the M8.12a greedy cases already owned by M8.13a.
- M8.13b B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  overlaying the M8.13b files on the pushed M8.13a commit, building with
  `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-session-runtime-rs`, and running
  `python3 ds4-parity/compare_cli_session_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-session-runtime-rs --negative-test`.
- M8.13b B300 comparator reported `CLI session runtime comparator: PASS, 28
  checks` and `CLI session runtime negative tests: PASS, 5 checks`.
- M8.13b local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-engine`, `cargo build -p ds4-engine --bin ds4-session-runtime-rs`,
  full `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_session_runtime.py
  ds4-parity/compare_cli_argmax_runtime.py`, and `git diff --check`.
- M8.13c extends `rust/ds4-gguf/src/cli_parse.rs` so `CliConfig` retains the
  core one-shot generation surface: system prompt, context, token limit,
  temperature, top-p, min-p, optional seed, and thinking mode.
- M8.13c adds `ds4-cli-one-shot-rs`, which parses the exact M8.12a argv through
  the shared Rust CLI parser, routes `--temp 0` cases through the M8.13a argmax
  boundary, routes the seeded non-greedy case through the M8.13b session
  boundary, and rejects non-generation modes outside this milestone.
- M8.13c adds `ds4-parity/compare_cli_one_shot_runtime.py`, which runs
  `target/debug/ds4-cli-one-shot-rs` against all five M8.12a current-C cases:
  `greedy_inline_nothink`, `prompt_file_think`, `think_max_downgrade`,
  `seeded_sampling_nothink`, and `ctx_too_small`.
- M8.13c B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  overlaying the M8.13c files on the pushed M8.13b commit, building with
  `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-cli-one-shot-rs`, and running
  `python3 ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-cli-one-shot-rs --negative-test`.
- M8.13c B300 comparator reported `CLI one-shot runtime comparator: PASS, 144
  checks` and `CLI one-shot runtime negative tests: PASS, 5 checks`.
- M8.13c local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-gguf cli_parse -- --nocapture`, `cargo build -p ds4-engine --bin
  ds4-cli-one-shot-rs`, `cargo test -p ds4-engine`, full
  `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/compare_cli_session_runtime.py`, and `git diff --check`.
- M8.13d extends `EngineOptions` and `ds4-cli-one-shot-rs` so Rust one-shot
  generation passes through M8.12b runtime controls: optional MTP path, thread
  count, MTP draft tokens and margin, directional steering file and scales,
  warm weights, and quality.
- M8.13d extends `rust/ds4-gguf/src/cli_parse.rs` so the shared Rust CLI parser
  retains `--mtp`, `--mtp-draft`, `--mtp-margin`, `-t`/`--threads`,
  `--dir-steering-file`, `--dir-steering-ffn`, and `--dir-steering-attn`; it
  also preserves the current-C default of `--dir-steering-file` without an
  explicit scale implying FFN scale `1.0`.
- M8.13d makes `ds4-cli-one-shot-rs` return the C-side blocked startup exit
  path for `ds4_engine_open` failures, avoiding an extra Rust stderr wrapper for
  the M8.12b blocked `--backend metal` and missing-MTP cases.
- M8.13d adds `ds4-parity/compare_cli_runtime_controls_runtime.py`, which runs
  `target/debug/ds4-cli-one-shot-rs` against all five M8.12b current-C cases:
  `backend_name_cuda_quality_threads`, `warm_weights`, `directional_steering`,
  `backend_metal_error`, and `mtp_missing_model`.
- M8.13d B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  copying the changed files one by one to `/workspace/ds4`, verifying SHA256
  matches, building with `CARGO_HOME=/tmp/ds4-cargo
  RUSTUP_HOME=/tmp/ds4-rustup PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native
  cargo build -p ds4-engine --bin ds4-cli-one-shot-rs`, and running
  `python3 ds4-parity/compare_cli_runtime_controls_runtime.py
  ds4-parity/baselines/cli/m8.12b/current-c.json --candidate-binary
  target/debug/ds4-cli-one-shot-rs --negative-test`.
- M8.13d B300 comparator reported `CLI runtime-controls runtime comparator:
  PASS, 154 checks` and `CLI runtime-controls runtime negative tests: PASS, 6
  checks`.
- M8.13d local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-gguf cli_parse -- --nocapture`, `cargo build -p ds4-engine --bin
  ds4-cli-one-shot-rs`, `cargo test -p ds4-engine`, full
  `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_runtime_controls_runtime.py
  ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/compare_cli_session_runtime.py`, and `git diff --check`.
- M8.14 adds `ds4-parity/check_cli_interactive_dump.py`, a current-C PTY
  capture/checker for interactive CLI transcripts. It sets an explicit PTY
  window size so `linenoise` does not block on cursor-position probing and sends
  carriage returns to match terminal Enter behavior.
- M8.14 adds fixture
  `ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt` with SHA256
  `418e80cf0af232690c1cdb12b0ca015953f0f14d24c2c8ba40052464387b49b3`.
- M8.14 captures two current-C B300 PTY cases in
  `ds4-parity/baselines/cli/m8.14/current-c.json`: `command_suite` and
  `ctrl_c_at_prompt`. The command suite covers empty input, `/help`, `/think`,
  `/think-max`, `/nothink`, `/ctx 128`, `/read`, an unknown command, one direct
  model-backed prompt, and `/quit`; the Ctrl+C case covers deterministic
  Ctrl+C behavior at the prompt.
- M8.14 committed current-C artifact size is 34,582 bytes with SHA256
  `223939b68a2791d79b2b7bac207e1e2a89db71f3073d0e4ab885a34e08c65a9f`.
  The manifest size is 2,472 bytes with SHA256
  `d63181e48f0c381308401692c60672a3d6bf1dfe5e142036df728ee100cd40f4`.
- M8.14 B300 validation passed after copying the checker and fixture to
  `/workspace/ds4`, rebuilding `ds4` with `make ds4 CUDA_ARCH=native`, running
  the PTY capture/checker, and copying `current-c.json` and `manifest.json`
  back from the pod.
- M8.14 B300 and local revalidation passed
  `python3 ds4-parity/check_cli_interactive_dump.py
  ds4-parity/baselines/cli/m8.14/current-c.json --manifest
  ds4-parity/baselines/cli/m8.14/manifest.json --negative-test` with oracle
  `PASS, 89 checks`, manifest `PASS, 15 checks`, and negative tests `PASS, 6
  checks`.
- M8.14 local validation also passed `python3 -m json.tool` for both M8.14 JSON
  files, `python3 -m py_compile ds4-parity/check_cli_interactive_dump.py`, and
  `git diff --check`.
- M8.15 was split before implementation because the current Rust runtime has
  one-shot generation but not reusable sessions, chat transcript mutation,
  session progress callbacks, or REPL command state.
- M8.15 source inspection found the needed current-C APIs in `ds4.h`:
  `ds4_chat_begin`, `ds4_chat_append_message`,
  `ds4_chat_append_assistant_prefix`, `ds4_chat_append_max_effort_prefix`,
  `ds4_tokens_push`, reusable `ds4_session_*` APIs,
  `ds4_session_set_progress`, `ds4_session_common_prefix`,
  `ds4_session_invalidate`, `ds4_session_pos`, and `ds4_session_ctx`.
- M8.15 source inspection found that `rust/ds4-engine/src/lib.rs` currently
  exposes one-shot prompt encoding plus argmax/session generation helpers, but
  `Session` and `TokenPrinter` are private and there is no public reusable
  chat transcript/session boundary for interactive turns.
- M8.15 has been split in `RUST_PORT_ROADMAP.md` into M8.15a reusable
  interactive session boundary, M8.15b REPL command-state surface, and M8.15c
  final interactive PTY transcript surface.
- M8.15a extends `rust/ds4-engine/src/lib.rs` with a borrowed `ChatSession`
  wrapper over the current C chat/session APIs, including chat transcript
  creation, user/assistant append, reusable session creation/reset,
  session-sync progress callbacks, token append/eos, session position/context
  handling, and two-turn generation.
- M8.15a adds `ds4-interactive-runtime-rs`, a narrow non-PTY runtime-boundary
  binary that simulates the M8.14 model-backed `/read` turn followed by the
  direct prompt `Answer with one short noun: glacier.` and emits explicit
  `read`/`direct` turn blocks.
- M8.15a adds `ds4-parity/compare_cli_interactive_runtime.py`, which extracts
  the M8.14 generated turn bytes from the committed PTY transcript and compares
  them against `target/debug/ds4-interactive-runtime-rs` while also checking
  runtime stderr anchors and forbidden unsupported paths.
- M8.15a B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  copying the changed files to `/workspace/ds4`, verifying SHA256 matches,
  building with `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-interactive-runtime-rs`, and running
  `python3 ds4-parity/compare_cli_interactive_runtime.py
  ds4-parity/baselines/cli/m8.14/current-c.json --candidate-binary
  target/debug/ds4-interactive-runtime-rs --negative-test`.
- M8.15a B300 comparator reported `CLI interactive runtime comparator: PASS, 19
  checks` and `CLI interactive runtime negative tests: PASS, 4 checks`.
- M8.15a local validation passed for `cargo fmt --all -- --check`, `cargo build
  -p ds4-engine --bin ds4-interactive-runtime-rs`, `cargo test -p
  ds4-engine`, full `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_interactive_runtime.py
  ds4-parity/check_cli_interactive_dump.py`, and `git diff --check`.
- M8.15b adds `rust/ds4-engine/src/interactive_cli.rs`, a model-free REPL
  command-state surface covering empty input, `/help`, `/think`, `/think-max`,
  `/nothink`, `/ctx`, `/read`, unknown slash commands, `/quit`, `/exit`,
  normal prompt dispatch, and Ctrl+C-at-prompt recovery.
- M8.15b exports the REPL command-state module from `ds4-engine` while leaving
  PTY line editing and model-backed turn execution for M8.15c/M8.15a.
- M8.15b command matching uses command-boundary checks so short or extended
  slash commands such as `/c`, `/ctxx`, `/rea`, and `/readx` route to the C
  CLI unknown-command category instead of matching or panicking.
- M8.15b local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-engine interactive_cli -- --nocapture` (5 REPL command tests), `cargo
  test -p ds4-engine` (11 tests), full `cargo test --workspace`, and `git diff
  --check`.
- M8.15c adds `ds4-cli-interactive-rs`, a no-prompt Rust REPL binary that uses
  the shared CLI parser, M8.15b `ReplState`, and M8.15a `ChatSession` to cover
  the M8.14 `command_suite` and `ctrl_c_at_prompt` PTY cases.
- M8.15c adds `ChatSession::run_turn_to_writer` so PTY generation writes model
  bytes before the timing line, matching the current C merged stdout/stderr
  transcript order while preserving the existing buffered `run_turn` API.
- M8.15c adds `ds4-parity/compare_cli_interactive_pty.py`, which reuses the
  M8.14 PTY driver and explicitly normalizes linenoise redraw-only frames while
  comparing committed prompts, command responses, generated bytes, timing and
  progress categories, exit status, normalized transcript hashes, and
  Ctrl+C-at-prompt recovery.
- M8.15c B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  copying the changed files to `/workspace/ds4`, building with `CARGO_HOME=/tmp/ds4-cargo
  RUSTUP_HOME=/tmp/ds4-rustup PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native
  cargo build -p ds4-engine --bin ds4-cli-interactive-rs`, and running
  `python3 ds4-parity/compare_cli_interactive_pty.py
  ds4-parity/baselines/cli/m8.14/current-c.json --candidate-binary
  target/debug/ds4-cli-interactive-rs --write-candidate
  /tmp/ds4-m8.15c-rust-pty.json --negative-test`.
- M8.15c B300 comparator reported `CLI interactive PTY comparator: PASS, 59
  checks` and `CLI interactive PTY negative tests: PASS, 4 checks`.
- M8.15c local validation passed for `cargo fmt --all -- --check`, `cargo build
  -p ds4-engine --bin ds4-cli-interactive-rs`, `cargo test -p ds4-engine`, full
  `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_interactive_pty.py
  ds4-parity/check_cli_interactive_dump.py`, and `git diff --check`.
- M8.16 adds `ds4-parity/run_cli_parity_report.py`, which executes local M8 CLI
  artifact validators/comparators and records model-backed B300 current-C
  refreshes plus Rust runtime/PTY checks as skipped items with exact rerun
  commands.
- M8.16 wires the CLI report into `ds4-parity/run_parity_report.py` as the
  `M8.16 CLI parity report` comparator item.
- M8.16 local CLI report validation passed with `summary: 9 passed, 13 skipped,
  0 failed`; JSON output from `python3 ds4-parity/run_cli_parity_report.py
  --json` parsed with `python3 -m json.tool`.
- M8.16 unified report validation passed with `summary: 14 passed, 5 skipped,
  0 failed`, including the nested `M8.16 CLI parity report`.
- M8.16 local validation also passed `python3 -m py_compile
  ds4-parity/run_cli_parity_report.py ds4-parity/run_parity_report.py`, full
  `cargo test --workspace`, and `git diff --check`.
- M9.1 split Milestone 9 into concrete server parity items in
  `RUST_PORT_ROADMAP.md` and `.memory/TODO.md`: request parse/render, HTTP
  skeleton and model metadata, non-streaming chat runtime, streaming SSE,
  OpenAI tool/DSML server surface, Responses/Anthropic protocols, cache/KV/tool
  memory, and server report integration.
- M9.1 source inspection found that `ds4_server.c` server scope includes
  OpenAI chat, Responses, Anthropic, streaming deltas, CORS/preflight behavior,
  thinking controls, stop lists, DSML/tool parsing, live tool-tail validation,
  usage accounting, cache/KV restore, tool-memory replay, and eviction policy.
- M9.1 baseline inspection confirmed M0.4 covers `models`, non-streaming chat,
  streaming chat, tool calls, thinking-disabled chat, and memory-token cache
  continuation, while M0.5 covers disk-KV seed miss/restore and continuation
  restore with KV headers, rendered text, traces, and cache decisions.
- M9.1 validation passed with source/fixture inspection, roadmap/board diff, and
  `git diff --check`.
- M9.2 was split before implementation because the full request parse/render
  surface spans OpenAI core chat, OpenAI tool/DSML prompt rendering, and
  Responses/Anthropic protocol inputs.
- M9.2a is the next executable item and is intentionally limited to model-free
  OpenAI `/v1/chat/completions` core fields and prompt rendering, excluding
  tool-call payloads and alternate protocols.
- M9.2b covers OpenAI tool schema parsing and DSML prompt rendering, while M9.2c
  covers Responses and Anthropic request parsing/rendering inputs; later M9.6
  and M9.7 remain responsible for model-backed tool/protocol response behavior.
- M9.2 split validation passed with roadmap/board diff and `git diff --check`.
- M10.6b adds a current-C whole-prefill short-prompt oracle helper,
  `ds4-prefill-whole-short`, the Rust prefill safe-facade operation wrappers,
  and `compare_prefill_whole_short.py` with pinned B300 output digests for
  `short_italian_fact_whole_prefill`.
- M10.6b B300 validation used pod `ds4-rust-port-b300` in `hou2-prod1` with
  `/workspace/ds4/ds4flash.gguf`; current-C
  `metal_graph_prefill_layer_major` and Rust whole-prefill candidate matched in
  `compare_prefill_whole_short.py --oracle
  /tmp/ds4-m106b-prefill-whole-short-oracle.json --candidate
  /tmp/ds4-m106b-prefill-whole-short-rust.json`, reporting 780 checks.
- M10.6b local validation passed static comparator 60 checks, negative tests
  rejected 15 mutations, unified report `--skip-local-oracles` reported 45
  passed, 34 skipped, and 0 failed, `arch -arm64` C helper builds, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, `git diff --check`,
  and non-interactive Claude review with no blockers.
- M10.6c extends the current-C prefill oracle helper and Rust
  `ds4-prefill-whole-short` candidate to cold chunked prompts, including a
  2052-token cap-crossing case and the full long memory archive prompt. The
  paired current-C and Rust reports use a same-run C oracle rather than the
  older M10.4 baseline because optimized CUDA MoE atomic-down output is
  nondeterministic.
- M10.6c adds `ds4-parity/compare_prefill_chunked.py`, which validates static
  structure, live current-C oracle parity, chunk starts/sizes, output absolute
  positions and local rows, raw ring rows/spans, compressed/index counters,
  output digests, and sampled logits for both chunked fixtures.
- M10.6c B300 validation used pod `ds4-rust-port-b300` in `hou2-prod1` with
  `/workspace/ds4/ds4flash.gguf` and `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`;
  current-C oracle vs Rust candidate comparator passed 400 checks for the
  2052-token fixture and 400 checks for the full long prompt.
- M10.6c local validation passed static chunked comparator 30 checks, chunked
  negative tests rejected 5 mutations, whole-prefill negative tests rejected 15
  mutations, unified parity report with local oracles skipped reported 46
  passed, 35 skipped, and 0 failed, `arch -arm64 make
  ds4-prefill-whole-short-oracle-dump`, `cargo check -p ds4-gpu --bin
  ds4-prefill-whole-short`, full `cargo test --workspace`, `cargo fmt --all --
  --check`, and `git diff --check`.
- M10.6c non-interactive Claude review returned no blockers. The review noted a
  dead future-context risk where continued aligned non-Ratio4 chunks could have
  used the zero-prefix compressor prefill path; the final Rust path restricts
  the aligned replay branch to Ratio4 and keeps continued non-Ratio4 chunks on
  per-token compressor updates.
- M10.6d extends the current-C prefill oracle helper and Rust
  `ds4-prefill-whole-short` candidate to resumed-prefix execution. The covered
  routes are a 512-token exact-prefix cache hit, a 512-to-514 short suffix that
  falls below `metal_graph_resume_prefill_min_tokens` and decodes token by
  token, and a 1537-to-2337 resumed suffix with chunk starts/sizes
  `(1537,511)` and `(2048,289)`.
- M10.6d adds `ds4-parity/compare_prefill_resumed.py`, which validates static
  structure, live current-C oracle parity, route decisions, resume threshold,
  decode-token count, resumed chunk boundaries, checkpoint length, final output
  rows, raw ring rows/spans, compressed/index counters, output digests, and
  sampled logits for cache-hit, decode-suffix, and resumed-chunked fixtures.
- M10.6d B300 validation used pod `ds4-rust-port-b300` in `hou2-prod1` with
  `/workspace/ds4/ds4flash.gguf` and `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1`;
  current-C oracle vs Rust candidate comparator passed 425 checks for the
  cache-hit fixture, 425 checks for the short decode-suffix fixture, and 425
  checks for the resumed-chunked fixture.
- M10.6d local validation passed static resumed comparator 27 checks, negative
  tests rejected 6 mutations, whole-prefill negative tests rejected 15
  mutations, chunked-prefill negative tests rejected 5 mutations, unified
  parity report with local oracles skipped reported 47 passed, 36 skipped, and
  0 failed, `arch -arm64 make ds4-prefill-whole-short-oracle-dump`, `cargo
  check -p ds4-gpu --bin ds4-prefill-whole-short`, full `cargo test
  --workspace`, `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_prefill_resumed.py ds4-parity/run_parity_report.py`, and
  `git diff --check`.
- M10.6d non-interactive Claude review returned `NO BLOCKERS`. The review
  questioned resumed chunk `output_row` handling for mid-block suffixes; the
  final code documents that chunked prefill writes each chunk into dense local
  batch rows while absolute token positions drive cache/raw-ring addressing.
