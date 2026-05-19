# Trading ML V4

This repository is the research and production substrate for a BNR 1m trading ML system.

## Objective

Build a disciplined ML4T-style workflow for a BNR strategy centered on the `09:30:00` to `09:30:59` America/New_York price zone. The first goal is not live deployment. The first goal is an honest research factory that can define candidate setups, label them, train a classifier, backtest them, audit them, and say whether the edge exists.

## Stage 1 Scope

Stage 1 established the project substrate:

- Repo structure.
- Canonical project state schema.
- Global config system.
- Skill registry.
- Experiment registry format.
- Databento data manifest format.
- Structured run logging.
- Basic tests.

## Current State

The repo has moved beyond Stage 1:

- Stage 2 (BNR research engine) is implemented in `src/trading_ml/` and exercised by the `tools/` runners.
- Stage 3 (LangGraph + DeepAgents orchestration) is implemented via `src/trading_ml/langgraph_integration.py` and `src/trading_ml/deepagents_integration.py` (requires the optional agents dependencies).

## Non-Negotiable Guardrails

- No feature may use information unavailable at decision time.
- Exploration, validation, and holdout periods stay separated.
- Every experiment must be logged with hypothesis, config, data slice, result, and decision.
- Backtest results are invalid until timestamp, cost, and slippage assumptions are explicit.
- Strategy logic must remain portable across research and later paper/live modes.
- Human examples may guide labeling and review, but they do not bypass validation discipline.
- Model changes after seeing holdout results send the work back to exploration.

## Initial Layout

- `configs/`: global configuration, evidence boundary, and skill mapping.
- `data/`: manifests and future raw or processed datasets.
- `experiments/`: registry and structured run logs.
- `models/`: model artifacts and registries.
- `notebooks/`: exploratory notebooks kept outside the production pipeline.
- `reports/`: validation and audit outputs.
- `src/trading_ml/`: project code.
- `tests/`: basic test harness for schemas, config, and logging.

## Immediate Next Phases

- Stage 4: backtest-to-live technical parity and production controls.
