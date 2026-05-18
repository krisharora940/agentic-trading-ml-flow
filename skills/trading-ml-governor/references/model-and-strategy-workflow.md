# Part 3 And Part 4 Takeaways

This reference distills the ML4T third-edition model and strategy lessons that matter most for an autonomous trading research workflow.

## 1. Model choice is downstream of task design

Part 3 frames the first modeling decision as: what prediction problem actually maps to a tradable action?

- Classification is often better than raw-return regression when the economic decision is discrete.
- Label preprocessing, horizon choice, and winsorization can move results more than model-family changes.
- Probability calibration matters because thresholds, ranks, and weighted decisions all depend on score meaning, not just score ordering.

For the governor this means:

- Ask whether the target is aligned with the action before changing the model.
- Treat label redesign as a first-class research move, not an afterthought.
- Require a documented translation from prediction output to order decision.

## 2. Use a model-family ladder, not architecture wandering

The chapter sequence implies a practical model ladder:

- Regularized linear models for baseline, calibration, and sign discipline.
- GBMs as the default flexible tabular model.
- Sequence models only when temporal structure is central and simpler formulations leave evidence that they are underfitting.
- Latent-factor models when the objective is attribution, compression, or pricing structure.
- Causal ML when mechanism or treatment heterogeneity is the research question.

For the governor this means:

- A stronger model only earns budget after the weaker family has been fairly validated.
- “Different” is not a reason to promote a model family.
- Every escalation needs a stated structural reason.

## 3. Validation must mirror search and dependence

Part 3 and the diagnostic workflows emphasize that financial samples are dependent, overlap is common, and single-split wins are weak evidence.

Required structure:

- Walk-forward or rolling-origin evaluation by default.
- Purging and embargo when labels overlap.
- Search-aware inference once repeated tuning begins.
- Stability checks across folds, not just aggregate means.

For the governor this means:

- No result is confirmation-grade without sequential validation.
- Overlap structure must be measured, not assumed away.
- Trial counting is part of the result, not separate paperwork.

## 4. Interpretability is a diagnostic, not decoration

Part 3 treats SHAP and related diagnostics as tools for checking:

- sign consistency
- magnitude plausibility
- stability across refits
- regime-conditional behavior

For the governor this means:

- Importance is only useful if it persists.
- A feature that “wins” one fit but changes sign or disappears across windows is a warning.
- Explanations should pressure-test signal plausibility, not just summarize it.

## 5. Backtesting is falsification

Part 4 reframes backtesting from “does it work?” to “what would disprove it?”

Every backtest must fully specify:

- signal timing
- execution timing
- rebalancing cadence
- sizing rule
- order type assumption
- cost model
- constraints and benchmark

For the governor this means:

- Reject results with underspecified protocol.
- Require cost sensitivity ranges, not one assumed spread/slippage point.
- Keep a non-ML baseline so ML is competing with something interpretable.

## 6. Prediction quality is not trading quality

Part 4 repeatedly separates:

- signal quality
- portfolio translation
- cost survival
- stability over time

IC, AUC, or accuracy never settle the case alone.

For the governor this means:

- Add a translation checkpoint between model validation and promotion.
- Ask whether breadth, turnover, and cost convert the signal into a viable strategy.
- Treat holdout disappointment as a classification problem: prediction decay, translation decay, execution decay, or data issue.

## 7. The practical default is robustness over peak

The synthesis chapter makes the strongest cross-case-study point: robust GBM-like pipelines often beat flashier alternatives after full implementation friction.

For the governor this means:

- Prefer stable, repeatable gains over peak validation metrics.
- Favor 5-10 hypothesis-driven iterations over architecture churn.
- Escalate to confirmation only when the integrated pipeline survives data, validation, cost, and risk scrutiny together.
