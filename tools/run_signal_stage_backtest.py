from __future__ import annotations

import json
import random
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.stage2_modeling import score_model_split
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_splits import build_walk_forward_splits


def build_stitched_predictions(result: dict, model_family: str) -> list[dict]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("signal-stage backtest requires pandas") from exc
    features = pd.DataFrame(result["features_records"])
    labels = pd.DataFrame(result["labels_records"])
    merged = features.merge(labels, on="candidate_id", how="inner")
    feature_cols = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label", "outcome", "entry_time", "exit_time", "entry_price", "stop_price", "target_price", "exit_price", "bars_held", "mfe", "mae", "pnl_r"}
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    folds_input, _ = build_walk_forward_splits(merged)
    stitched = []
    for train, test, fold_meta in folds_input[:2]:
        if train.empty or test.empty or train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        scored = score_model_split(train, test, model_family=model_family, feature_cols=feature_cols)
        pred = scored["prediction_frame"].copy()
        pred["fold"] = fold_meta.fold
        stitched.extend(pred.to_dict(orient="records"))
    return stitched


def plumbing_random_test(records: list[dict], threshold: float) -> dict:
    sample = [dict(row) for row in records]
    probs = [row["probability"] for row in sample]
    random.Random(7).shuffle(probs)
    for row, prob in zip(sample, probs, strict=True):
        row["probability"] = prob
    return run_event_driven_policy_backtest(sample, threshold=threshold)


def main() -> None:
    state = build_agent_loop_state()
    config = dict(state["stage2_config"])
    config["target_multiple"] = 1.5
    threshold = 0.45
    result = run_stage2_research_engine(Stage2Config(**config))
    families = ["linear_baseline", "gbm"]
    rows = []
    for family in families:
        stitched = build_stitched_predictions(result, family)
        execution = run_event_driven_policy_backtest(stitched, threshold=threshold)
        utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
        random_execution = plumbing_random_test(stitched, threshold)
        random_total = float(random_execution.get("total_pnl_r", 0.0) or 0.0) if random_execution.get("status") == "complete" else None
        total = float(execution.get("total_pnl_r", 0.0) or 0.0) if execution.get("status") == "complete" else None
        rows.append(
            {
                "model_family": family,
                "threshold": threshold,
                "status": execution.get("status"),
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": total,
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "win_rate": float(execution.get("win_rate", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "utility_score": utility.get("score"),
                "random_plumbing_total_pnl_r": random_total,
                "ic_to_sharpe_proxy": (total / abs(float(execution.get("max_drawdown_r", 1.0) or 1.0))) if total is not None else None,
            }
        )
    payload = {
        "source": "signal_stage_backtest_30s_mnq",
        "label_target_multiple": 1.5,
        "signal_threshold": threshold,
        "rows": rows,
    }
    output = Path("reports/signal_stage_backtest.json")
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
