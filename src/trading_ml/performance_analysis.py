from __future__ import annotations

from typing import Any


def build_stitched_performance_summary(
    prediction_records: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    if not prediction_records:
        return {"status": "pending", "reason": "missing_prediction_records"}

    frame = pd.DataFrame(prediction_records)
    if frame.empty or "probability" not in frame or "pnl_r" not in frame:
        return {"status": "pending", "reason": "invalid_prediction_records"}

    frame["entry_time"] = pd.to_datetime(frame["entry_time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["entry_time"]).sort_values("entry_time").reset_index(drop=True)
    trades = frame[frame["probability"] >= float(threshold)].copy()
    if trades.empty:
        return {
            "status": "pending",
            "reason": "no_trades_at_threshold",
            "threshold": float(threshold),
            "stitched_prediction_count": int(len(frame)),
        }

    trades["cum_pnl_r"] = trades["pnl_r"].cumsum()
    trades["cum_peak_r"] = trades["cum_pnl_r"].cummax()
    trades["drawdown_r"] = trades["cum_pnl_r"] - trades["cum_peak_r"]

    session_groups = trades.groupby("session_date", dropna=False)["pnl_r"]
    session_rows = [
        {
            "session_date": str(session_date),
            "trade_count": int(series.count()),
            "session_pnl_r": float(series.sum()),
            "avg_trade_r": float(series.mean()),
        }
        for session_date, series in session_groups
    ]

    equity_rows = [
        {
            "candidate_id": str(row["candidate_id"]),
            "entry_time": row["entry_time"].isoformat(),
            "session_date": str(row["session_date"]),
            "probability": float(row["probability"]),
            "pnl_r": float(row["pnl_r"]),
            "cum_pnl_r": float(row["cum_pnl_r"]),
            "drawdown_r": float(row["drawdown_r"]),
        }
        for _, row in trades.iterrows()
    ]

    total_pnl_r = float(trades["pnl_r"].sum())
    avg_trade_r = float(trades["pnl_r"].mean())
    win_rate = float((trades["label"] == 1).mean()) if "label" in trades else 0.0
    max_drawdown_r = float(trades["drawdown_r"].min())
    positive_sessions = sum(1 for row in session_rows if row["session_pnl_r"] > 0)
    negative_sessions = sum(1 for row in session_rows if row["session_pnl_r"] < 0)

    return {
        "status": "complete",
        "threshold": float(threshold),
        "stitched_prediction_count": int(len(frame)),
        "trade_count": int(len(trades)),
        "total_pnl_r": total_pnl_r,
        "avg_trade_r": avg_trade_r,
        "win_rate": win_rate,
        "max_drawdown_r": max_drawdown_r,
        "session_count": int(len(session_rows)),
        "positive_sessions": positive_sessions,
        "negative_sessions": negative_sessions,
        "session_rows": session_rows,
        "equity_curve": equity_rows,
    }
