# DS4 Parity Artifacts

This directory stores oracle captures and future Rust comparison fixtures.

- `baselines/`: current C/CUDA/Metal command captures used as the oracle for
  early port work.
- Future fixture roots should be added by milestone and should include a
  manifest entry that names the oracle, input fixture, comparator, acceptance
  rule, command, environment, output files, and drift policy.

Do not treat a log file as proof by itself. A manifest entry must explain which
behavior the log proves and which requirements remain blocked or deferred.

## Static Verification

Run the local static verifier before changing baseline artifacts:

```sh
python3 ds4-parity/verify_baselines.py
```

The verifier checks committed hashes and structured artifact shapes without
rerunning model-backed commands. Its negative test copies the baseline fixtures
to a temporary directory, corrupts one CSV, and requires verification failure:

```sh
python3 ds4-parity/verify_baselines.py --negative-test
```

Compare the committed server and KV artifacts against themselves with the
documented normalizations:

```sh
python3 ds4-parity/compare_server_kv.py
```

The comparator accepts candidate artifact directories from a fresh run or a
future Rust implementation:

```sh
python3 ds4-parity/compare_server_kv.py \
  --server-candidate /path/to/server-traces/m0.4 \
  --kv-candidate /path/to/kv-artifacts/m0.5
```

Its negative test corrupts behavioral server/KV fields in temporary candidate
copies and requires comparison failure:

```sh
python3 ds4-parity/compare_server_kv.py --negative-test
```

Compare the committed official-vector logprob fixture and M0.3 run evidence:

```sh
python3 ds4-parity/compare_logprob_numeric.py
```

The M0.3 B300 run log is pass/fail evidence from
`./ds4_test --logprob-vectors`; it does not dump runtime logits. The numeric
slice lives in `tests/test-vectors/official.vec` and the raw official JSON under
`tests/test-vectors/official/`. The comparator therefore audits those numeric
fixtures, verifies the B300 run markers, and compares candidate vector files
with exact selected-token matching plus the reported absolute logprob tolerance:

```sh
python3 ds4-parity/compare_logprob_numeric.py \
  --candidate-vec /path/to/official-vector-style-output.vec
```

Its negative test corrupts a temporary candidate selected token and logprob
value and requires comparison failure:

```sh
python3 ds4-parity/compare_logprob_numeric.py --negative-test
```

Compare the committed B300 benchmark CSV baselines:

```sh
python3 ds4-parity/compare_bench_csv.py
```

The benchmark comparator treats workload shape as exact behavioral surface:
schema, context frontiers, prefill intervals, generation-token counts, and
`kvcache_bytes` must match. Throughput is performance surface; it is compared
only after capture metadata confirms the same model, prompt, CUDA backend
marker, and GPU machine class. The default throughput threshold is the M0.6
policy of at most 10% regression:

```sh
python3 ds4-parity/compare_bench_csv.py \
  --candidate-dir /path/to/bench/m0.6 \
  --max-regression 0.10
```

Its negative test corrupts schema, context frontier, generation-token,
`kvcache_bytes`, and throughput fields in temporary candidate copies and
requires comparison failure:

```sh
python3 ds4-parity/compare_bench_csv.py --negative-test
```

Run the unified local parity report:

```sh
python3 ds4-parity/run_parity_report.py
```

The unified report runs local no-model C checks, then runs the committed M1,
M4, M5, and M6 comparator reports. Model-backed B300 oracle refreshes are
skipped by default, but each skip includes the temp-kubeconfig and
explicit-context rerun command needed to reproduce the check. For a
comparator-only report:

```sh
python3 ds4-parity/run_parity_report.py --skip-local-oracles
```

## Sampling And Logprob Parity

Run the local Milestone 6 report:

```sh
python3 ds4-parity/run_sampling_parity_report.py
```

The report runs the M6.2 fixed-logits C oracle checker, M6.3 Rust sampler
comparator, M6.4 committed session-logits fixture checker, M6.5 Rust
model-logits comparator, M6.6a decode-policy oracle checker, and M6.6b Rust
decode-policy comparator. The B300 model-backed M6.4 recapture is skipped by
default with the exact manifest refresh command.
