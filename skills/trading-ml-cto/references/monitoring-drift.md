# Monitoring And Drift

Use this reference when designing live monitoring, diagnosing decay, or updating models.

## Failure Classes

Technical failure:

- Same intended inputs produce different outputs.
- Backtest/live pipeline diverges.
- Data, feature, order, or broker state is wrong.

Statistical failure:

- Implementation is correct.
- Predictive relationship has decayed.
- Causes may include overfitting, look-ahead bias discovered late, regime change, or alpha crowding.

Do not treat one as the other.

## Monitoring Stack

Data integrity:

- Freshness.
- Missing fields.
- Duplicate bars.
- Bad timestamps.
- Corporate action or adjustment mismatch.

Performance:

- Rolling Sharpe.
- IC.
- Hit rate.
- Drawdown.
- Turnover.
- Live-to-backtest realization ratio.

Execution quality:

- Slippage.
- Spread cost.
- Fill ratio.
- Latency.
- Reject rate.

Use trailing windows and tiered thresholds: watch, warning, critical. Each threshold needs a response action.

## Drift Diagnostics

Data drift:

- PSI.
- K-S tests.
- Feature distribution monitoring.

Feature drift:

- SHAP value monitoring.
- Feature importance shift.

Concept drift:

- ADWIN.
- DDM.
- Prediction error streams.

Diagnostic table:

- Drift without decay: model may be robust.
- Decay without detected drift: monitoring coverage is incomplete.
- Drift with decay: retraining on recent data may be warranted.
- No drift and no decay: continue monitoring.

## Safe Model Updates

Use scheduled and triggered retraining, but promote only after evidence.

Required:

- Shadow mode challenger.
- Capital-capped A/B test when appropriate.
- Gradual allocation increase.
- Tested rollback.
- Deflated Sharpe or bootstrap comparison.
- Multiple-testing correction when challengers compete.

Reject improvements below a practical Sharpe delta, often around 0.2 to 0.3, unless there is a clear portfolio-level reason.
