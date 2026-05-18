---
name: ml4t-postgres
description: "Use when storing market data, features, labels, or model artifacts in PostgreSQL."
---

# Postgres

Use this skill when the task needs ML4T-style guidance to build repeatable research and trading plumbing around the model.

## Workflow

1. Make data, features, labels, and model artifacts versionable.
2. Keep pipelines deterministic and restartable.
3. Expose audit fields for timestamps, training windows, and feature lineage.
4. Automate health checks before promoting anything toward live use.

## Guardrails

- Do not assume bar close data was tradable at bar open.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/data/
- https://ml4trading.io/docs/engineer/
- https://ml4trading.io/docs/live/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
