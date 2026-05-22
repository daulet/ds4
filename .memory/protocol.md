# DS4 Rust Port Protocol

## Mission

Port DS4 host/runtime behavior to Rust one ownership boundary at a time while
the current C/CUDA/Metal implementation remains the execution-behavior oracle.

## Active Board Selection

Use `.memory/TODO.md` as the active board. Select the first open item whose
preconditions are satisfied, preferring the largest item that still has one
tangible goal, one oracle, and one comparator. If an item cannot name those
three fields, split it in `RUST_PORT_ROADMAP.md` before implementation.

## Required Work Item Shape

Every implementation or artifact commit must state:

- Goal: the behavior, artifact, or ownership boundary completed.
- Oracle: the C binary, C ABI path, captured artifact, fixture, or benchmark
  used as truth.
- Comparator: the command or script that compares new output to the oracle.
- Acceptance: exact equality, tolerance, response shape rule, or explicit
  no-behavior-change condition.
- Validation: commands run before commit.

## Reviewer Policy

Each commit requires a non-interactive Claude review before committing. The
prompt must state the commit goal, changed files, oracle, comparator, validation
evidence, and ask for an adversarial principal-engineer review. Fix critical or
material findings, then rerun the relevant validation and review if the fix
changes the reviewed behavior.

Use `ultrathink` in Claude prompts. Source, tests, and logs override reviewer
claims.

## Validation Ladder

Use the narrowest gate that proves the item while iterating, then run the
item's required gate before commit.

- Docs/protocol-only: inspect the diff and ensure no production source or build
  behavior changed.
- C build surface: `make` for the default local backend, `make cpu` for
  CPU-only portability, and `make test` when a model/backend is available.
- Rust workspace surface: `cargo fmt --all --check`, targeted `cargo check`,
  targeted `cargo test`, then workspace `cargo test` once a workspace exists.
- CUDA/GPU or expensive validation: run on a reused B300 GPU pod in
  `hou2-prod1`; record the chosen namespace and pod identity in
  `.memory/status.md` before the first run, then record command, commit, model
  identity, and logs for each validation.
- Benchmarks: compare the same model, backend, prompt, context settings,
  machine class, and power/thermal state.

Read logs before marking an item done.

## Commit And Push Policy

Commit only after the item passes validation and review. Stage only files that
belong to the item. Push each reviewed commit to `origin main` unless the branch
changes or the user redirects.

## Debugging Ledger

When debugging a failing behavior, create or update `.memory/debugging-ledger.md`
with the current hypothesis, exact commands, observed outputs, source evidence,
and next check. After compaction, read the ledger before continuing. When the
debugging goal is achieved, delete or shrink the ledger so only durable lessons
remain in `.memory/lessons.md`.

There is no active debugging ledger when no failure investigation is open.

## Lessons

Record only non-obvious findings discovered through trial and error that are not
fetchable directly from the repo. Use `.memory/lessons.md`.

## Incident Stops And Forbidden Actions

- Do not run large CPU inference on macOS.
- Do not run multiple huge model processes concurrently.
- Do not mutate shared GPU or cluster resources until the target B300 pod,
  namespace, and command have been confirmed.
- Do not relax comparisons, skip assertions, or weaken tests to pass a gate.
- Do not introduce permanent semantic variants behind flags.

## Loop Continuation

After each reviewed and pushed commit, update `.memory/status.md` and
`.memory/TODO.md`, then continue with the next open roadmap item unless blocked
by missing model/backend access, a shared-resource incident stop, or user
redirection.
