---
name: ml4t-guardrail-execution-realism
description: "Use when pressure-testing fill assumptions, costs, latency, and liquidity realism."
---

# Guardrail Execution Realism

Use this skill when the task needs ML4T-style guidance to prevent false edge from leakage, timing errors, or unrealistic execution assumptions.

## Workflow

1. Trace what was knowable at decision time for every feature and label.
2. Separate research convenience from live execution reality.
3. Bias toward rejecting an idea if timing cannot be proved clean.
4. Document assumptions that would make the result invalid in production.

## Guardrails

- Treat unclear timing as a blocking issue, not a documentation gap.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/about/
- https://ml4trading.io/docs/backtest/
- https://ml4trading.io/docs/diagnostic/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
