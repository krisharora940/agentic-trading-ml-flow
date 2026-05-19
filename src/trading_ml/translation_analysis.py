from __future__ import annotations

from typing import Any

from trading_ml.config import load_bnr_config
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.utility_analysis import compute_execution_utility


def build_translation_analysis(
    stage2_result: dict[str, Any],
    prediction_records: list[dict[str, Any]] | None = None,
    *,
    sizing_policy: str | None = None,
    regime_throttle_policy: str | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    records = list(prediction_records or [])
    if not records:
        model_summary = dict(stage2_result.get("model_summary", {}))
        records = list(model_summary.get("prediction_records", []))
    if not records:
        return {"status": "pending", "reason": "missing_prediction_records"}

    frame = pd.DataFrame(records)
    if frame.empty or "probability" not in frame or "pnl_r" not in frame:
        return {"status": "pending", "reason": "invalid_prediction_records"}

    bnr_config = load_bnr_config()
    thresholds = list(bnr_config.get("translation_contract", {}).get("threshold_grid", [0.45, 0.5, 0.55, 0.6, 0.65]))
    sizing_name = sizing_policy or str(bnr_config.get("frozen_benchmark", {}).get("sizing_policy", "binary_threshold_v1"))
    throttle_name = regime_throttle_policy or str(bnr_config.get("frozen_benchmark", {}).get("regime_throttle_policy", "none"))
    rows: list[dict[str, Any]] = []
    execution_ready = {"direction", "entry_time", "exit_time", "entry_price", "exit_price", "stop_price", "session_date"}.issubset(frame.columns)
    for threshold in thresholds:
        if execution_ready:
            execution = run_event_driven_policy_backtest(
                frame.to_dict(orient="records"),
                threshold=float(threshold),
                sizing_policy=sizing_name,
                regime_throttle_policy=throttle_name,
            )
            utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
            rows.append(
                {
                    "threshold": float(threshold),
                    "trade_count": int(execution.get("trade_count", 0) or 0),
                    "avg_pnl_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                    "win_rate": float(execution.get("win_rate", 0.0) or 0.0),
                    "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                    "avg_size_multiplier": float(execution.get("avg_size_multiplier", 0.0) or 0.0),
                    "throttled_signals": int(execution.get("throttled_signals", 0) or 0),
                    "utility_score": utility.get("score"),
                }
            )
        else:
            selected = frame[frame["probability"] >= threshold].copy()
            trade_count = int(len(selected))
            avg_pnl_r = float(selected["pnl_r"].mean()) if trade_count else 0.0
            win_rate = float((selected["label"] == 1).mean()) if trade_count else 0.0
            rows.append(
                {
                    "threshold": float(threshold),
                    "trade_count": trade_count,
                    "avg_pnl_r": avg_pnl_r,
                    "win_rate": win_rate,
                    "total_pnl_r": float(selected["pnl_r"].sum()) if trade_count else 0.0,
                    "utility_score": None,
                }
            )

    best = max(rows, key=lambda item: ((float(item["utility_score"]) if item.get("utility_score") is not None else float("-inf")), item["total_pnl_r"], item["trade_count"]))
    status = "pass" if best["trade_count"] > 0 and best["total_pnl_r"] > 0 else "fail"
    return {
        "status": status,
        "sizing_policy": sizing_name,
        "regime_throttle_policy": throttle_name,
        "rows": rows,
        "best_threshold": best,
    }
