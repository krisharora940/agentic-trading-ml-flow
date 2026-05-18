---
name: ml4t-multiple-testing
description: "Use when correcting for many trials, many factors, or repeated hypothesis search."
---

# Multiple Testing

Use this skill when the task needs ML4T-style guidance to stress-test whether an apparent edge is real, stable, and tradable.

## Workflow

1. Define the benchmark, sample split, and failure criteria first.
2. Use purging, embargoes, or walk-forward structure when labels overlap.
3. Inspect distribution shifts, concentration, and regime dependence.
4. Treat p-values as weak evidence unless paired with economic intuition and execution realism.

## Guardrails

- Do not assume bar close data was tradable at bar open.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/diagnostic/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
