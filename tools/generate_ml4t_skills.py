from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path("/Users/radhikaarora/Documents/Trading ML V4/ml4t_codex_skills")


CATEGORY_GUIDES = {
    "Concepts": {
        "focus": "frame the trading-ML problem correctly before modeling",
        "workflow": [
            "Define the prediction target, decision timestamp, and execution timestamp.",
            "State the market assumption behind the edge and what regime can break it.",
            "Separate signal discovery from position sizing, risk, and execution.",
            "Write down the minimum falsification test before adding complexity.",
        ],
        "docs": [
            "https://ml4trading.io/about/",
            "https://ml4trading.io/docs/diagnostic/",
        ],
    },
    "Data Acquisition": {
        "focus": "acquire market data with explicit timestamp semantics and survivorship controls",
        "workflow": [
            "Identify source, timezone, session rules, and instrument universe.",
            "Track when each field became knowable, not just the event timestamp.",
            "Preserve raw data and build deterministic cleaning steps.",
            "Verify splits, dividends, delistings, and symbol mapping before modeling.",
        ],
        "docs": [
            "https://ml4trading.io/docs/data/",
        ],
    },
    "Feature Engineering": {
        "focus": "turn raw market structure into leakage-safe predictive features",
        "workflow": [
            "Anchor every feature to the information set available at decision time.",
            "Prefer simple, stable transforms before expressive feature factories.",
            "Check missingness, outliers, and scaling behavior by regime.",
            "Measure whether the feature survives friction, delays, and cross-validation.",
        ],
        "docs": [
            "https://ml4trading.io/docs/engineer/",
        ],
    },
    "Evaluation & Validation": {
        "focus": "stress-test whether an apparent edge is real, stable, and tradable",
        "workflow": [
            "Define the benchmark, sample split, and failure criteria first.",
            "Use purging, embargoes, or walk-forward structure when labels overlap.",
            "Inspect distribution shifts, concentration, and regime dependence.",
            "Treat p-values as weak evidence unless paired with economic intuition and execution realism.",
        ],
        "docs": [
            "https://ml4trading.io/docs/diagnostic/",
        ],
    },
    "Backtesting": {
        "focus": "simulate execution honestly enough to reject fragile strategies",
        "workflow": [
            "Specify order timing, fill assumptions, fees, spread, and slippage.",
            "Use point-in-time features and realistic portfolio state updates.",
            "Measure turnover, concentration, drawdown path, and capacity limits.",
            "Prefer a simple falsifiable simulator over a rich but opaque one.",
        ],
        "docs": [
            "https://ml4trading.io/docs/backtest/",
        ],
    },
    "Portfolio Management": {
        "focus": "convert forecasts into positions with explicit risk and capital constraints",
        "workflow": [
            "Separate alpha forecast quality from the allocation rule.",
            "Define exposure, leverage, concentration, and liquidity limits.",
            "Model turnover penalties and portfolio drift explicitly.",
            "Review how sizing behaves in volatility spikes and correlation breaks.",
        ],
        "docs": [
            "https://ml4trading.io/docs/backtest/",
            "https://ml4trading.io/docs/live/",
        ],
    },
    "Infrastructure": {
        "focus": "build repeatable research and trading plumbing around the model",
        "workflow": [
            "Make data, features, labels, and model artifacts versionable.",
            "Keep pipelines deterministic and restartable.",
            "Expose audit fields for timestamps, training windows, and feature lineage.",
            "Automate health checks before promoting anything toward live use.",
        ],
        "docs": [
            "https://ml4trading.io/docs/data/",
            "https://ml4trading.io/docs/engineer/",
            "https://ml4trading.io/docs/live/",
        ],
    },
    "Workflows": {
        "focus": "structure the research loop so model iteration does not corrupt evidence",
        "workflow": [
            "Work from hypothesis to dataset to validation to simulator in that order.",
            "Freeze evaluation slices before tuning.",
            "Promote ideas only after they survive diagnostics and implementation friction.",
            "Log decisions, discarded ideas, and reasons for failure.",
        ],
        "docs": [
            "https://ml4trading.io/about/",
            "https://ml4trading.io/docs/",
            "https://ml4trading.io/docs/diagnostic/user-guide/workflows/",
        ],
    },
    "Production": {
        "focus": "ship a trading model with monitoring, rollback, and operational discipline",
        "workflow": [
            "Define the live decision loop, latency budget, and failure policy.",
            "Monitor inputs, outputs, fills, risk, and model drift separately.",
            "Make rollbacks simple and independent of model retraining.",
            "Assume outages, stale data, and broker edge cases will happen.",
        ],
        "docs": [
            "https://ml4trading.io/docs/live/",
        ],
    },
    "Guardrails": {
        "focus": "prevent false edge from leakage, timing errors, or unrealistic execution assumptions",
        "workflow": [
            "Trace what was knowable at decision time for every feature and label.",
            "Separate research convenience from live execution reality.",
            "Bias toward rejecting an idea if timing cannot be proved clean.",
            "Document assumptions that would make the result invalid in production.",
        ],
        "docs": [
            "https://ml4trading.io/about/",
            "https://ml4trading.io/docs/backtest/",
            "https://ml4trading.io/docs/diagnostic/",
        ],
    },
}


SKILLS = [
    ("Concepts", "ml4t-information-ratio", "Use when evaluating strategy quality with information ratio, tracking error, and risk-adjusted active returns."),
    ("Concepts", "ml4t-tcs", "Use when reasoning about the true cost of skill, including friction, uncertainty, and signal decay in trading ML."),
    ("Concepts", "ml4t-parkinson-volatility", "Use when estimating volatility from high-low ranges instead of close-to-close returns."),
    ("Concepts", "ml4t-rolling-sharpe", "Use when measuring time-varying Sharpe behavior and degradation across regimes."),
    ("Concepts", "ml4t-pnl-labeling", "Use when defining labels directly from realized PnL, execution rules, or trade outcomes."),
    ("Concepts", "ml4t-concept-drift", "Use when detecting regime change, model staleness, or feature-target relationship drift."),
    ("Concepts", "ml4t-f1-threshold", "Use when tuning classification thresholds where class balance and trade economics matter more than raw accuracy."),
    ("Concepts", "ml4t-hyperparameter-validation", "Use when validating hyperparameters without contaminating holdout evidence."),
    ("Concepts", "ml4t-neutralization", "Use when removing market, sector, beta, or other common exposures from features or signals."),
    ("Data Acquisition", "ml4t-reddit-ohlc", "Use when sourcing or reconstructing OHLC-style market data from Reddit discussions, posts, or community signals."),
    ("Data Acquisition", "ml4t-whalewisdom", "Use when using WhaleWisdom or 13F-style holdings data in a point-in-time research workflow."),
    ("Data Acquisition", "ml4t-sec-filings", "Use when ingesting SEC filings, fundamentals, or event timestamps for trading models."),
    ("Data Acquisition", "ml4t-quote-data", "Use when handling bid, ask, spread, or quote-derived microstructure data."),
    ("Data Acquisition", "ml4t-tiingo", "Use when working with Tiingo market data, adjusted prices, or corporate actions."),
    ("Data Acquisition", "ml4t-binance", "Use when acquiring Binance spot or derivatives data for crypto trading research."),
    ("Data Acquisition", "ml4t-databento", "Use when using Databento datasets, schemas, and timestamped market data feeds."),
    ("Data Acquisition", "ml4t-polygon", "Use when sourcing equities or options data from Polygon with timestamp and adjustment hygiene."),
    ("Feature Engineering", "ml4t-fracdiff", "Use when applying fractional differencing to preserve memory while reducing non-stationarity."),
    ("Feature Engineering", "ml4t-fte", "Use when building feature transformation pipelines for trading data with strict fit/transform separation."),
    ("Feature Engineering", "ml4t-ohlcv-features", "Use when engineering indicators or statistical features from OHLCV bars."),
    ("Feature Engineering", "ml4t-zscore", "Use when standardizing features with rolling or cross-sectional z-score transforms."),
    ("Feature Engineering", "ml4t-kelly-fraction", "Use when deriving Kelly-style sizing features or decision rules from estimated edge and odds."),
    ("Feature Engineering", "ml4t-feature-importance", "Use when ranking, pruning, or diagnosing trading features with robust importance methods."),
    ("Feature Engineering", "ml4t-clustering", "Use when clustering assets, regimes, or feature states to simplify model structure."),
    ("Feature Engineering", "ml4t-volatility-targeting", "Use when scaling features, forecasts, or positions to a volatility target."),
    ("Evaluation & Validation", "ml4t-backtest-overfitting", "Use when estimating whether backtest performance is overstated by search, selection, or repeated tuning."),
    ("Evaluation & Validation", "ml4t-multiple-testing", "Use when correcting for many trials, many factors, or repeated hypothesis search."),
    ("Evaluation & Validation", "ml4t-walk-forward", "Use when validating models with sequential walk-forward or rolling-origin evaluation."),
    ("Evaluation & Validation", "ml4t-psi", "Use when using population stability index to detect feature drift or train-live mismatch."),
    ("Evaluation & Validation", "ml4t-cpcv", "Use when applying combinatorial purged cross-validation to overlapping trading samples."),
    ("Evaluation & Validation", "ml4t-cv", "Use when choosing cross-validation structure for time series or panel-style trading data."),
    ("Evaluation & Validation", "ml4t-uniqueness", "Use when quantifying sample uniqueness under overlapping labels or event windows."),
    ("Evaluation & Validation", "ml4t-sfi", "Use when using single-feature importance to test whether a feature contributes standalone signal."),
    ("Backtesting", "ml4t-vectorbt", "Use when prototyping vectorized backtests with explicit assumptions around fees, fills, and timing."),
    ("Backtesting", "ml4t-backtrader", "Use when building event-driven backtests in Backtrader and auditing execution assumptions."),
    ("Backtesting", "ml4t-zipline", "Use when working with Zipline-style pipelines, calendars, and execution simulation."),
    ("Backtesting", "ml4t-backtesting-py", "Use when using backtesting.py for fast strategy iteration with realistic parameter discipline."),
    ("Backtesting", "ml4t-slippage", "Use when modeling slippage, spread, queue risk, or impact in a trading simulator."),
    ("Portfolio Management", "ml4t-hrp", "Use when allocating capital with hierarchical risk parity or correlation-aware diversification."),
    ("Portfolio Management", "ml4t-position-sizing", "Use when turning model scores into position sizes under risk and turnover constraints."),
    ("Portfolio Management", "ml4t-risk-management", "Use when defining stop logic, exposure limits, drawdown controls, or kill-switch behavior."),
    ("Infrastructure", "ml4t-fastapi", "Use when exposing research or trading services through FastAPI with operational safeguards."),
    ("Infrastructure", "ml4t-postgres", "Use when storing market data, features, labels, or model artifacts in PostgreSQL."),
    ("Infrastructure", "ml4t-kafka", "Use when designing streaming market-data or event pipelines with Kafka."),
    ("Infrastructure", "ml4t-airflow", "Use when orchestrating scheduled data, training, or backtest workflows with Airflow."),
    ("Workflows", "ml4t-triple-barrier", "Use when labeling events with triple-barrier logic and timestamp-aware exits."),
    ("Workflows", "ml4t-meta-labeling", "Use when stacking a second model on top of primary signals to decide whether to act."),
    ("Workflows", "ml4t-ensemble", "Use when combining multiple models or signals into a more stable trading decision."),
    ("Workflows", "ml4t-point-in-time", "Use when enforcing point-in-time correctness across datasets, features, and joins."),
    ("Workflows", "ml4t-purging", "Use when purging overlapping samples and embargoing adjacent periods during validation."),
    ("Workflows", "ml4t-sbt", "Use when running sequential bootstrap techniques for dependent financial samples."),
    ("Workflows", "ml4t-data-leakage", "Use when auditing a pipeline for leakage across features, labels, joins, scaling, or validation."),
    ("Workflows", "ml4t-case-study-development", "Use when turning a trading idea into a reproducible end-to-end case-study research build."),
    ("Workflows", "ml4t-factor-research", "Use when researching, validating, and pressure-testing factor-style signals before strategy integration."),
    ("Workflows", "ml4t-model-validation", "Use when validating a trading ML model with sequential splits, overlap controls, and economic thresholds."),
    ("Workflows", "ml4t-production-readiness", "Use when deciding whether a research result is operationally and statistically ready to advance toward paper or live trading."),
    ("Workflows", "ml4t-strategy-workflow", "Use when managing the full ML4T strategy workflow from scoping through feature design, validation, backtest, and promotion."),
    ("Production", "ml4t-ibkr", "Use when integrating Interactive Brokers workflows for live or paper trading."),
    ("Production", "ml4t-live-monitoring", "Use when monitoring live predictions, fills, pnl, risk, or model health."),
    ("Production", "ml4t-model-registry", "Use when versioning, promoting, rolling back, or governing trained trading models."),
    ("Production", "ml4t-drift-retraining", "Use when defining retraining triggers, drift thresholds, and promotion criteria."),
    ("Guardrails", "ml4t-guardrail-lookahead-bias", "Use when checking that no feature, label, or execution rule leaks future information."),
    ("Guardrails", "ml4t-guardrail-point-in-time", "Use when validating point-in-time joins, timestamps, revisions, and knowledge dates."),
    ("Guardrails", "ml4t-guardrail-execution-realism", "Use when pressure-testing fill assumptions, costs, latency, and liquidity realism."),
    ("Guardrails", "ml4t-guardrail-research-discipline", "Use when enforcing holdout integrity, limited tuning loops, and evidence-preserving research workflow."),
]


def render_skill(category: str, name: str, description: str) -> str:
    guide = CATEGORY_GUIDES[category]
    title = name.replace("ml4t-", "").replace("-", " ").title()
    workflow = "\n".join(f"{i}. {step}" for i, step in enumerate(guide["workflow"], start=1))
    docs = "\n".join(f"- {url}" for url in guide["docs"])
    guardrails = [
        "Do not assume bar close data was tradable at bar open.",
        "Do not fit scalers, encoders, or selectors on the full sample.",
        "Call out missing timestamp provenance before trusting performance.",
        "Prefer rejecting a result to rationalizing unrealistic assumptions.",
    ]
    if category == "Production":
        guardrails[0] = "Do not ship a live path without kill-switches, stale-data checks, and rollback."
    elif category == "Data Acquisition":
        guardrails[0] = "Do not mix adjusted and unadjusted fields without stating why."
    elif category == "Portfolio Management":
        guardrails[0] = "Do not let sizing logic hide weak alpha quality."
    elif category == "Guardrails":
        guardrails[0] = "Treat unclear timing as a blocking issue, not a documentation gap."
    guardrail_lines = "\n".join(f"- {line}" for line in guardrails)

    body = f"""---
name: {name}
description: "{description}"
---

# {title}

Use this skill when the task needs ML4T-style guidance to {guide["focus"]}.

## Workflow

{workflow}

## Guardrails

{guardrail_lines}

## Public ML4T references

{docs}

Note: this Codex skill pack is an adaptation of the public ML4T skill catalog and docs, not a copy of gated skill bodies.
"""
    return textwrap.dedent(body)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for category, name, description in SKILLS:
        skill_dir = ROOT / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(render_skill(category, name, description))


if __name__ == "__main__":
    main()
