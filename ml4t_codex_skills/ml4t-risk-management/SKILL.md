---
name: ml4t-risk-management
description: "Use when defining stop logic, exposure limits, drawdown controls, or kill-switch behavior."
---

# Risk Management

Use this skill when the task needs ML4T-style guidance to convert forecasts into positions with explicit risk and capital constraints.

## Workflow

1. Separate alpha forecast quality from the allocation rule.
2. Define exposure, leverage, concentration, and liquidity limits.
3. Model turnover penalties and portfolio drift explicitly.
4. Review how sizing behaves in volatility spikes and correlation breaks.

## Guardrails

- Do not let sizing logic hide weak alpha quality.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/backtest/
- https://ml4trading.io/docs/live/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
