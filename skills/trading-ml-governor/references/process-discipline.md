# Process Discipline

This reference captures the project philosophy behind the Trading ML Governor skill.

## Why process discipline matters

Durable trading performance depends more on a disciplined, adaptive workflow than on model sophistication. Market shockwaves from 2020 through 2025 broke assumptions calibrated to the prior decade. The practical lesson is that process failures usually dominate algorithmic failures.

Use this vocabulary precisely:

- Structural break: a meaningful change in the data-generating process.
- Regime: a persistent environment with different return, volatility, liquidity, or correlation behavior.
- Drift: gradual change in feature, label, or model relationships.
- Online detection: monitoring procedures that flag possible change while the system is running.

Anchor the project stance in adaptive-market thinking: edge is conditional, temporary, and shaped by participant behavior.

## The ML4T workflow

Treat the lifecycle as two linked layers:

1. Data infrastructure foundation.
2. Iterative strategy research loop.

The foundation enforces point-in-time correctness, reproducibility, data lineage, and honest simulation assumptions. The research loop moves from scoping through features, models, backtests, and deployment.

These items must be fixed before research begins:

- Decision-time correctness.
- Universe rules.
- Label definitions.
- Cost-model class.
- Evaluation protocol.

Everything else can iterate, but only inside the evidence boundary.

## Evidence boundary

The evidence boundary separates exploration from confirmation. Credibility comes from:

- Counting the search.
- Freezing specs before confirmation.
- Reserving untouched holdout data for final evaluation.

Once confirmation evidence is viewed, redesign must stop or the result drops back into exploration.

## Causal inference and generative AI

Causal inference and generative AI amplify the upside of good process and the downside of bad process.

Two research entry points:

- Prediction-first.
- Mechanism-first.

Two deliverable types:

- Tradable signals.
- Measurement deliverables such as premia estimates, exposures, or risk attribution.

Causal inference is a diagnostic discipline tool, not a universal requirement. Generative AI should speed up ideation, coding, and documentation, but its outputs are never privileged. They must pass leakage, timing, tradability, and validation checks like any other artifact.

## Market regimes

Regime detection is a risk lens, not a guaranteed return-timing tool.

Useful outcomes:

- Map environments to position sizing.
- Adjust exposure caps.
- Tighten monitoring thresholds.
- Explain where the strategy is fragile.

Avoid:

- Look-ahead labels created ex post.
- Over-interpretation of unsupervised clusters.
- Pretending regime definitions were available live when they were not.

## Independent vs institutional

Institutional setups partially mitigate process failures through dedicated infrastructure and controls. Independent researchers must replace that with documented checkpoints and reusable systems.

Common independent failure modes:

- Goalpost drift.
- Assumption stacking.
- Flexibility without search accounting.
- Discovering untradability too late.

The solo advantage is tighter iteration and access to capacity-constrained niches. The highest-leverage investment is reusable infrastructure that makes disciplined experimentation cheap and repeatable.
