from __future__ import annotations

from typing import Any

from trading_ml.config import load_global_config
from trading_ml.translation_policy import allow_signal_for_regime, compute_position_size, compute_regime_size_multiplier


def run_event_driven_policy_backtest(
    prediction_records: list[dict[str, Any]],
    *,
    threshold: float,
    slippage_profile: str | None = None,
    sizing_policy: str | None = None,
    regime_throttle_policy: str | None = None,
    regime_size_policy: str | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    if not prediction_records:
        return {"status": "pending", "reason": "missing_prediction_records"}

    frame = pd.DataFrame(prediction_records)
    required = {"candidate_id", "direction", "probability", "entry_time", "exit_time", "entry_price", "exit_price", "stop_price", "pnl_r", "session_date"}
    if frame.empty or not required.issubset(frame.columns):
        return {"status": "pending", "reason": "invalid_prediction_records"}

    frame["entry_time"] = pd.to_datetime(frame["entry_time"], errors="coerce", utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["entry_time", "exit_time", "entry_price", "exit_price", "stop_price"]).sort_values("entry_time").reset_index(drop=True)
    frame = frame[frame["probability"] >= float(threshold)].copy()
    if frame.empty:
        return {"status": "pending", "reason": "no_trades_at_threshold", "threshold": float(threshold)}

    costs = load_global_config()
    slippage_cfg = dict(costs.get("slippage", {}))
    profile = slippage_profile or str(slippage_cfg.get("profile", "base"))
    slippage_mode = str(slippage_cfg.get("model", "ticks"))
    tick_size = float(slippage_cfg.get("tick_size", 0.25) or 0.25)
    ticks_per_side = _ticks_per_side(slippage_cfg, profile)

    active_exit_time = None
    executed: list[dict[str, Any]] = []
    skipped_overlaps = 0
    throttled_signals = 0
    cum_pnl_r = 0.0
    peak_pnl_r = 0.0

    for row in frame.to_dict(orient="records"):
        if not allow_signal_for_regime(row, policy_name=regime_throttle_policy):
            throttled_signals += 1
            continue

        entry_time = row["entry_time"]
        exit_time = row["exit_time"]
        if active_exit_time is not None and entry_time < active_exit_time:
            skipped_overlaps += 1
            continue

        direction = str(row["direction"])
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        stop_price = float(row["stop_price"])
        risk_points = abs(entry_price - stop_price)
        if risk_points <= 0:
            continue

        size_multiplier = compute_position_size(
            float(row.get("probability", 0.0) or 0.0),
            threshold=float(threshold),
            policy_name=sizing_policy,
        )
        regime_multiplier = compute_regime_size_multiplier(row, policy_name=regime_size_policy)
        final_size_multiplier = size_multiplier * regime_multiplier
        if final_size_multiplier <= 0:
            continue

        filled_entry = _apply_slippage(
            entry_price,
            direction,
            slippage_mode=slippage_mode,
            tick_size=tick_size,
            ticks_per_side=ticks_per_side,
            is_entry=True,
        )
        filled_exit = _apply_slippage(
            exit_price,
            direction,
            slippage_mode=slippage_mode,
            tick_size=tick_size,
            ticks_per_side=ticks_per_side,
            is_entry=False,
        )
        pnl_points = (filled_exit - filled_entry) if direction == "long" else (filled_entry - filled_exit)
        gross_pnl_r = pnl_points / risk_points
        pnl_r = gross_pnl_r * final_size_multiplier

        cum_pnl_r += pnl_r
        peak_pnl_r = max(peak_pnl_r, cum_pnl_r)
        drawdown_r = cum_pnl_r - peak_pnl_r

        executed.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "session_date": str(row["session_date"]),
                "direction": direction,
                "probability": float(row["probability"]),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "filled_entry_price": filled_entry,
                "filled_exit_price": filled_exit,
                "gross_label_pnl_r": float(row["pnl_r"]),
                "gross_executed_pnl_r": gross_pnl_r,
                "executed_pnl_r": pnl_r,
                "base_size_multiplier": size_multiplier,
                "regime_size_multiplier": regime_multiplier,
                "size_multiplier": final_size_multiplier,
                "cum_pnl_r": cum_pnl_r,
                "drawdown_r": drawdown_r,
                "bars_held": int(row.get("bars_held", 0) or 0),
            }
        )
        active_exit_time = exit_time

    if not executed:
        return {
            "status": "pending",
            "reason": "no_executed_trades",
            "threshold": float(threshold),
            "overlap_skips": skipped_overlaps,
            "throttled_signals": throttled_signals,
        }

    executed_frame = pd.DataFrame(executed)
    session_rows = [
        {
            "session_date": str(session_date),
            "trade_count": int(len(group)),
            "session_pnl_r": float(group["executed_pnl_r"].sum()),
            "avg_trade_r": float(group["executed_pnl_r"].mean()),
        }
        for session_date, group in executed_frame.groupby("session_date", dropna=False)
    ]

    return {
        "status": "complete",
        "threshold": float(threshold),
        "signal_count": int(len(frame)),
        "throttled_signals": throttled_signals,
        "trade_count": int(len(executed)),
        "overlap_skips": skipped_overlaps,
        "total_pnl_r": float(executed_frame["executed_pnl_r"].sum()),
        "avg_trade_r": float(executed_frame["executed_pnl_r"].mean()),
        "avg_size_multiplier": float(executed_frame["size_multiplier"].mean()),
        "avg_base_size_multiplier": float(executed_frame["base_size_multiplier"].mean()),
        "avg_regime_size_multiplier": float(executed_frame["regime_size_multiplier"].mean()),
        "win_rate": float((executed_frame["executed_pnl_r"] > 0).mean()),
        "max_drawdown_r": float(executed_frame["drawdown_r"].min()),
        "session_count": int(len(session_rows)),
        "positive_sessions": sum(1 for row in session_rows if row["session_pnl_r"] > 0),
        "negative_sessions": sum(1 for row in session_rows if row["session_pnl_r"] < 0),
        "fill_assumptions": {
            "model": slippage_mode,
            "profile": profile,
            "tick_size": tick_size,
            "ticks_per_side": ticks_per_side,
            "order_type": "market_on_signal_close_proxy",
            "single_position": True,
            "sizing_policy": sizing_policy or "binary_threshold_v1",
            "regime_throttle_policy": regime_throttle_policy or "none",
            "regime_size_policy": regime_size_policy or "none",
        },
        "session_rows": session_rows,
        "equity_curve": executed,
    }


def _apply_slippage(
    price: float,
    direction: str,
    *,
    slippage_mode: str,
    tick_size: float,
    ticks_per_side: float,
    is_entry: bool,
) -> float:
    if slippage_mode == "ticks":
        adj = tick_size * ticks_per_side
    else:
        adj = price * (ticks_per_side / 10_000.0)
    if direction == "long":
        return price + adj if is_entry else price - adj
    return price - adj if is_entry else price + adj


def _ticks_per_side(slippage_cfg: dict[str, Any], profile: str) -> float:
    if profile == "stressed":
        return float(slippage_cfg.get("stressed_ticks_per_side", 6.0) or 6.0)
    return float(slippage_cfg.get("base_ticks_per_side", 3.0) or 3.0)
