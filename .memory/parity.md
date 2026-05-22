# DS4 Rust Port Parity State

## Current Oracle

- Implementation: current C/CUDA/Metal DS4 code.
- Starting commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Primary docs: `AGENT.md`, `CONTRIBUTING.md`, `RUST_PORT_ROADMAP.md`

## Baseline Artifact Root

Future baseline artifacts should live under `ds4-parity/baselines/` with a
manifest that records command, cwd, commit, model identity, backend, environment
overrides, output files, and acceptance rule.

## Drift Policy

No Rust port milestone may intentionally drift from the C oracle unless the
milestone documents the format or semantic change, updates the comparator, and
records the acceptance rule before implementation.

The operating rules for drift, review, validation, and commit policy live in
`.memory/protocol.md`.
