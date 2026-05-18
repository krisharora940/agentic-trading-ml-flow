---
name: ml4t-reddit-ohlc
description: "Use when sourcing or reconstructing OHLC-style market data from Reddit discussions, posts, or community signals."
---

# Reddit Ohlc

Use this skill when the task needs ML4T-style guidance to acquire market data with explicit timestamp semantics and survivorship controls.

## Workflow

1. Identify source, timezone, session rules, and instrument universe.
2. Track when each field became knowable, not just the event timestamp.
3. Preserve raw data and build deterministic cleaning steps.
4. Verify splits, dividends, delistings, and symbol mapping before modeling.

## Guardrails

- Do not mix adjusted and unadjusted fields without stating why.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/data/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
