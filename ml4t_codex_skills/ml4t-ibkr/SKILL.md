---
name: ml4t-ibkr
description: "Use when integrating Interactive Brokers workflows for live or paper trading."
---

# Ibkr

Use this skill when the task needs ML4T-style guidance to ship a trading model with monitoring, rollback, and operational discipline.

## Workflow

1. Define the live decision loop, latency budget, and failure policy.
2. Monitor inputs, outputs, fills, risk, and model drift separately.
3. Make rollbacks simple and independent of model retraining.
4. Assume outages, stale data, and broker edge cases will happen.

## Guardrails

- Do not ship a live path without kill-switches, stale-data checks, and rollback.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/live/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
