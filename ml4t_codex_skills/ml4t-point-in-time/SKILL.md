---
name: ml4t-point-in-time
description: "Use when enforcing point-in-time correctness across datasets, features, and joins."
---

# Point In Time

Use this skill when the task needs ML4T-style guidance to structure the research loop so model iteration does not corrupt evidence.

## Workflow

1. Work from hypothesis to dataset to validation to simulator in that order.
2. Freeze evaluation slices before tuning.
3. Promote ideas only after they survive diagnostics and implementation friction.
4. Log decisions, discarded ideas, and reasons for failure.

## Guardrails

- Do not assume bar close data was tradable at bar open.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/about/
- https://ml4trading.io/docs/
- https://ml4trading.io/docs/diagnostic/user-guide/workflows/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
