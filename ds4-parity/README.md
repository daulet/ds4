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
