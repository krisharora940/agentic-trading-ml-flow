---
name: ml4t-rolling-sharpe
description: "Use when measuring time-varying Sharpe behavior and degradation across regimes."
---

# Rolling Sharpe

Use this skill when the task needs ML4T-style guidance to frame the trading-ML problem correctly before modeling.

## Workflow

1. Define the prediction target, decision timestamp, and execution timestamp.
2. State the market assumption behind the edge and what regime can break it.
3. Separate signal discovery from position sizing, risk, and execution.
4. Write down the minimum falsification test before adding complexity.

## Guardrails

- Do not assume bar close data was tradable at bar open.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/about/
- https://ml4trading.io/docs/diagnostic/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
