# Part 3 And Part 4 For The CTO

This reference extracts the technical design implications of ML4T Part 3 and Part 4 for a production-minded research stack.

## 1. The bottleneck is not orchestration

LangGraph can keep iterating forever. That is not the scarce resource.

The real bottlenecks are:

- evidence quality
- validation structure
- protocol completeness
- cost realism
- risk translation
- reproducible comparison across model families

An institutional-grade system is not defined by more agents. It is defined by whether the same infrastructure can answer, cleanly and repeatedly:

- what changed
- why it changed
- whether the change survived sequential validation
- whether it still survives after costs and risk overlays

## 2. Model plumbing must support family comparison without code drift

Part 3 covers multiple model families with different strengths, but the workflow implication is straightforward:

- keep labels and features reusable
- keep evaluation reusable
- swap model families through a common contract

Required shared contract:

- `fit(train_frame, label_spec, feature_spec)`
- `predict(test_frame) -> score frame`
- `calibrate(validation_frame) -> calibrated score frame or calibration artifact`
- `translate(score frame, policy) -> signal frame`

Without this, apparent model improvements are often infrastructure differences in disguise.

## 3. The system must separate four layers

The stack should keep these layers independently versioned:

1. signal model
2. decision policy
3. allocation/risk overlay
4. execution/cost model

This is the CTO reading of Part 4. A predictive gain can vanish in translation if:

- thresholding is poor
- turnover explodes
- costs dominate
- sizing amplifies path risk

If these layers are entangled, the team cannot tell where the edge was lost.

## 4. A backtest is an executable protocol

Part 4 makes the protocol idea non-optional.

The CTO must ensure every backtest run encodes:

- decision timestamp
- execution timestamp
- order type assumption
- rebalancing cadence
- sizing policy
- constraints
- benchmark
- cost sensitivity

This should be stored as data, not hidden in notebook code.

## 5. Non-ML baselines are infrastructure, not pedagogy

The baseline strategy is not just educational.

It provides:

- a regression test for the simulator
- a reference for turnover and cost realism
- a fallback if ML translation collapses
- a benchmark for deciding whether model complexity paid for itself

Institutional-grade systems keep baselines runnable at all times.

## 6. Validation infrastructure has to reflect dependent samples

The CTO needs first-class support for:

- walk-forward splits
- purging and embargo
- CPCV-style variants where useful
- search-aware result accounting

This should be encoded in config and artifacts, not left to ad hoc notebook choices.

If labels overlap, the system should expose that overlap and make purging state visible in every validation run.

## 7. Cost and risk are not add-ons

Part 4 makes this explicit: strategy quality is inseparable from transaction costs and risk overlays.

The system therefore needs:

- instrument-specific cost models
- sensitivity sweeps over spread/slippage assumptions
- turnover reporting
- drawdown and recovery reporting
- exposure decomposition hooks

The CTO should treat “model metrics only” outputs as incomplete objects.

## 8. Diagnostics should decompose failure mode

The stack should distinguish:

- prediction decay
- translation decay
- execution decay
- data-quality failure
- protocol mismatch

That is the key technical barrier to institutional quality. Without failure decomposition, the loop becomes blind tuning.

## 9. The practical design target

The right technical target is not “an autonomous agent that keeps editing itself.”

It is:

- a versioned research controller
- a shared model/score/signal/backtest contract
- protocol-complete backtests
- validation artifacts that survive scrutiny
- cost and risk translation baked into every comparison
- promotion gates that reject improvements which do not survive integrated evaluation
