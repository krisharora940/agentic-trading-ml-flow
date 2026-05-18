---
name: trading-ml-governor
description: "Use when managing or building a trading ML project end-to-end with disciplined process, evidence boundaries, point-in-time correctness, regime awareness, and production governance. Ideal for solo systematic trading research, agent orchestration, and preventing ad hoc model iteration."
---

# Trading ML Governor

Use this skill as the default operating system for a trading ML project. Its job is to keep research disciplined, force decision-time correctness, account for search, and move work through explicit gates from idea to live monitoring.

This skill does not replace domain skills. It decides what must happen first, what evidence is required before progressing, and which `ml4t-*` skills to invoke next.

## Core stance

- Durable edge comes from process discipline more than model novelty.
- Non-stationarity is the default. Assume drift, regime shifts, and structural breaks are real until evidence says otherwise.
- Research flexibility is dangerous unless the search is counted and holdouts stay untouched.
- Anything that cannot survive timestamp, cost, and tradability scrutiny is not a result.
- Model-family escalation is earned. Linear baselines come first, GBMs are the default flexible workhorse, and deep learning only earns budget when structure or diagnostics justify it.
- Label design, horizon choice, and prediction-to-decision translation often matter more than architecture choice.

## Operating mode

When activated, always do these things first:

1. State the project object: instrument universe, timeframe, decision timestamp, execution timestamp, holding period, target behavior, and deployment destination.
2. State the current phase: foundation, exploration, confirmation, implementation, paper-trading, or live.
3. State the evidence boundary: what data is still allowed for exploration, what is frozen, and what holdout remains untouched.
4. List the blocking unknowns before proposing new experiments.

If any of the above are missing, stop speculative modeling and close the definition gap first.

## Workflow

### 1. Foundation gate

Before research starts, fix these project rules:

- Universe construction and survival rules.
- Data sources, adjustments, revisions, and timestamp provenance.
- Point-in-time join policy.
- Label definition and event horizon.
- Execution model class: next-bar, intrabar, event-driven, or queue-aware.
- Cost model class: commissions, spread, slippage, latency, borrow, and liquidity assumptions.
- Validation protocol: walk-forward, purging, embargo, and final holdout.

Use these skills when needed:

- `ml4t-point-in-time`
- `ml4t-data-leakage`
- `ml4t-guardrail-lookahead-bias`
- `ml4t-guardrail-point-in-time`
- `ml4t-walk-forward`
- `ml4t-purging`

Do not allow feature ideation until this gate is at least minimally specified.

### 2. Scoping gate

Define the research question in falsifiable terms:

- What edge is being claimed.
- Why it should exist economically or behaviorally.
- What observable would invalidate it.
- What minimal baseline must be beaten after costs.

Choose one of two entry modes:

- Prediction-first: forecast a target, then test economic usefulness.
- Mechanism-first: start from a causal or structural story, then derive measurable predictions.

Deliverable types:

- Tradable signal.
- Measurement deliverable such as regime map, risk attribution, or premia estimate.

### 3. Exploration loop

Exploration is allowed to be iterative, but not unbounded.

For every experiment:

1. Log the hypothesis.
2. Log the data slice and allowed information set.
3. Log the feature family and why it should matter.
4. Log the model class and why it is proportionate to the sample.
5. Log the exact evaluation slice touched.
6. Log whether this counts as a new search attempt.

The aim is not to maximize backtest fitness. The aim is to cheaply reject weak ideas without contaminating confirmation evidence.

Model ladder for exploration:

- Start with regularized linear baselines for calibration, sign discipline, and turnover awareness.
- Promote to GBMs when nonlinear interactions are plausible and the sample is still fundamentally tabular.
- Only promote to sequence models when temporal structure is the point of the problem, not because the linear or GBM result disappointed.
- Treat latent factor models as a separate objective family for dimensionality reduction, attribution, or pricing-error structure.
- Treat causal ML as a mechanism test or heterogeneity lens, not a mandatory path for every signal.

Useful skills during exploration:

- `ml4t-ohlcv-features`
- `ml4t-zscore`
- `ml4t-fracdiff`
- `ml4t-feature-importance`
- `ml4t-clustering`
- `ml4t-triple-barrier`
- `ml4t-meta-labeling`
- `ml4t-strategy-workflow`
- `ml4t-model-validation`
- `ml4t-factor-research`

### 4. Confirmation gate

An idea only reaches confirmation if it already survives:

- Point-in-time audit.
- Leakage audit.
- Cost realism screen.
- Regime sensitivity check.
- Concentration and turnover review.

Confirmation requires frozen specs:

- Fixed features.
- Fixed label logic.
- Fixed model family.
- Fixed sizing rule.
- Fixed cost assumptions.
- Fixed evaluation script.

Then run the reserved validation path and final holdout without redesigning the strategy midstream.

Confirmation diagnostics must answer:

- Does the signal survive walk-forward validation, not just one favorable split?
- Is apparent improvement coming from labels or from the model family?
- Does importance remain sign-consistent and economically plausible across refits?
- Is uncertainty or calibration stable enough to support thresholded decisions?
- Does the signal survive the translation from IC or AUC to realized trading quality after turnover and cost?

Useful skills:

- `ml4t-cpcv`
- `ml4t-backtest-overfitting`
- `ml4t-multiple-testing`
- `ml4t-uniqueness`
- `ml4t-sfi`
- `ml4t-rolling-sharpe`

### 5. Regime and decay layer

Treat regimes as a risk lens, not a magic timing engine.

Required questions:

- Which regime definition is available at decision time?
- Does the signal rely on one environment?
- How does volatility, drawdown, turnover, and hit rate change by regime?
- What risk action changes when regime confidence changes?

Allowed regime actions:

- Position scaling.
- Exposure caps.
- Strategy throttling.
- Monitoring thresholds.

Not allowed:

- Ex-post storytelling from clusters with no live mapping.
- Return claims based on labels that were only obvious after the fact.

### 6. Tradability gate

Before any promotion beyond research, check:

- Order timing vs. bar construction.
- Fill realism.
- Spread sensitivity.
- Slippage sensitivity.
- Capacity and liquidity.
- Failure under delayed or missing data.
- Breadth and turnover.
- Cost-survival range, not one optimistic cost point.
- Prediction-to-profit translation, not just raw signal metrics.

Useful skills:

- `ml4t-slippage`
- `ml4t-vectorbt`
- `ml4t-backtrader`
- `ml4t-backtesting-py`
- `ml4t-position-sizing`
- `ml4t-risk-management`

### 7. Production gate

Promotion requires:

- Reproducible training pipeline.
- Versioned datasets and model artifacts.
- Explicit promotion and rollback policy.
- Paper-trading or shadow mode plan.
- Monitoring for inputs, outputs, fills, pnl, risk, and drift.

Useful skills:

- `ml4t-model-registry`
- `ml4t-live-monitoring`
- `ml4t-drift-retraining`
- `ml4t-fastapi`
- `ml4t-postgres`
- `ml4t-ibkr`

## Checkpoints

Use these checkpoints to keep the project in check:

1. Scoping checkpoint: edge statement, target, horizon, and falsification test exist.
2. Data integrity checkpoint: timestamp provenance, adjustment policy, and point-in-time joins are documented.
3. Signal robustness checkpoint: feature logic survives leakage and regime review.
4. Tradability checkpoint: costs and execution do not erase the edge.
5. Monitoring checkpoint: live failure modes and rollback path are specified.
6. Translation checkpoint: signal quality, breadth, turnover, cost survival, and uncertainty all support the claimed trading objective.

If a checkpoint fails, move backward intentionally. Do not patch around it with more model complexity.

## Guardrails

- Do not let model sophistication outrun timestamp discipline.
- Do not let exploration touch the final holdout.
- Do not keep tuning after seeing confirmation evidence.
- Do not confuse regime clustering with causality.
- Do not treat generative AI output as evidence. Treat it as candidate material subject to the same audits as any other artifact.
- Do not stack assumptions silently. Write them down where they can be challenged.
- Do not continue if untradability is discovered late. Re-scope immediately.
- Do not escalate from linear to GBM to deep learning without proving the simpler layer is structurally insufficient.
- Do not accept a model-family win that disappears after cost translation or uncertainty review.
- Do not treat feature importance as stable unless it persists across walk-forward windows and refits.

## Response pattern

When using this skill, structure work in this order:

1. Phase and objective.
2. What is already fixed.
3. What is still exploratory.
4. What evidence is required next.
5. Which `ml4t-*` skill(s) should be applied now.
6. What would invalidate the current path.

Keep the project moving, but never by relaxing validation discipline.

## References

Read [references/process-discipline.md](references/process-discipline.md) when you need the conceptual frame and language behind this workflow.
Read [references/model-and-strategy-workflow.md](references/model-and-strategy-workflow.md) when choosing model families, validation structure, and the signal-to-strategy translation path.
