---
name: ml4t-feature-importance
description: "Use when ranking, pruning, or diagnosing trading features with robust importance methods."
---

# Feature Importance

Use this skill when the task needs ML4T-style guidance to turn raw market structure into leakage-safe predictive features.

## Workflow

1. Anchor every feature to the information set available at decision time.
2. Prefer simple, stable transforms before expressive feature factories.
3. Check missingness, outliers, and scaling behavior by regime.
4. Measure whether the feature survives friction, delays, and cross-validation.

## Guardrails

- Do not assume bar close data was tradable at bar open.
- Do not fit scalers, encoders, or selectors on the full sample.
- Call out missing timestamp provenance before trusting performance.
- Prefer rejecting a result to rationalizing unrealistic assumptions.

## Public ML4T references

- https://ml4trading.io/docs/engineer/

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
