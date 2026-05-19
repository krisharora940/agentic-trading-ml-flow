from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.config import load_bnr_config
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_audit import build_validation_audit


def main() -> None:
    state = build_agent_loop_state()
    bnr_config = load_bnr_config()
    label_space = bnr_config["label_search_v1"]["space"]
    thresholds = [float(x) for x in bnr_config["translation_contract"]["threshold_grid"]]
    output = Path("reports/label_threshold_matrix.json")

    rows: list[dict] = []
    for horizon_bars, stop_multiple, target_multiple in product(
        label_space["horizon_bars"],
        label_space["stop_multiple"],
        label_space["target_multiple"],
    ):
        config = dict(state["stage2_config"])
        config["horizon_bars"] = int(horizon_bars)
        config["stop_multiple"] = float(stop_multiple)
        config["target_multiple"] = float(target_multiple)
        result = run_stage2_research_engine(Stage2Config(**config))
        validation = build_validation_audit(result, {})
        stitched = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []))
        for threshold in thresholds:
            execution = run_event_driven_policy_backtest(stitched, threshold=threshold)
            utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
            rows.append(
                {
                    "feature_family": config["feature_family"],
                    "horizon_bars": int(horizon_bars),
                    "stop_multiple": float(stop_multiple),
                    "target_multiple": float(target_multiple),
                    "threshold": float(threshold),
                    "candidate_count": int(result.get("candidate_count", 0)),
                    "walk_forward_status": validation.get("walk_forward", {}).get("status"),
                    "walk_forward_mean_roc_auc": validation.get("walk_forward", {}).get("mean_roc_auc"),
                    "signal_count": int(execution.get("signal_count", 0) or 0),
                    "trade_count": int(execution.get("trade_count", 0) or 0),
                    "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                    "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                    "win_rate": float(execution.get("win_rate", 0.0) or 0.0),
                    "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                    "utility_score": utility.get("score"),
                }
            )
        output.write_text(
            json.dumps(
                {
                    "source": "governed_label_threshold_matrix",
                    "partial_rows": rows,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["utility_score"] or float("-inf")),
            row["total_pnl_r"],
            row["avg_trade_r"],
        ),
        reverse=True,
    )
    payload = {
        "source": "governed_label_threshold_matrix",
        "ranked_rows": ranked,
        "best_row": ranked[0] if ranked else None,
    }
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
