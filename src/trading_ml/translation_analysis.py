from __future__ import annotations

from typing import Any

from trading_ml.config import load_bnr_config


def build_translation_analysis(stage2_result: dict[str, Any], prediction_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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

    thresholds = list(load_bnr_config().get("translation_contract", {}).get("threshold_grid", [0.45, 0.5, 0.55, 0.6, 0.65]))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = frame[frame["probability"] >= threshold].copy()
        trade_count = int(len(selected))
        avg_pnl_r = float(selected["pnl_r"].mean()) if trade_count else 0.0
        win_rate = float((selected["label"] == 1).mean()) if trade_count else 0.0
        rows.append(
            {
                "threshold": threshold,
                "trade_count": trade_count,
                "avg_pnl_r": avg_pnl_r,
                "win_rate": win_rate,
                "total_pnl_r": float(selected["pnl_r"].sum()) if trade_count else 0.0,
            }
        )

    best = max(rows, key=lambda item: (item["avg_pnl_r"], item["trade_count"]))
    status = "pass" if best["trade_count"] > 0 and best["avg_pnl_r"] > 0 else "fail"
    return {
        "status": status,
        "rows": rows,
        "best_threshold": best,
    }
