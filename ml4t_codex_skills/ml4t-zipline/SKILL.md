---
name: ml4t-zipline
description: "Use when working with Zipline-style pipelines, calendars, and execution simulation."
---

# Zipline

Use this skill when the task needs ML4T-style guidance to simulate execution honestly enough to reject fragile strategies.

## Workflow

1. Specify order timing, fill assumptions, fees, spread, and slippage.
2. Use point-in-time features and realistic portfolio state updates.
3. Measure turnover, concentration, drawdown path, and capacity limits.
4. Prefer a simple falsifiable simulator over a rich but opaque one.

## Guardrails

- Do not assume bar close data was tradable at bar open.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/backtest/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
