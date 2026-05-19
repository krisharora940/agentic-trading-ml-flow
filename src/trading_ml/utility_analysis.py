from __future__ import annotations

from typing import Any

from trading_ml.config import load_bnr_config


def compute_execution_utility(summary: dict[str, Any]) -> dict[str, Any]:
    utility_cfg = dict(load_bnr_config().get("utility_contract", {}))
    total_pnl_weight = float(utility_cfg.get("total_pnl_weight", 1.0) or 1.0)
    avg_trade_weight = float(utility_cfg.get("avg_trade_weight", 0.5) or 0.5)
    max_drawdown_penalty = float(utility_cfg.get("max_drawdown_penalty", 0.75) or 0.75)
    trade_count_bonus = float(utility_cfg.get("trade_count_bonus", 0.05) or 0.05)
    min_trade_count = int(utility_cfg.get("min_trade_count", 4) or 4)

    trade_count = int(summary.get("trade_count", 0) or 0)
    total_pnl_r = float(summary.get("total_pnl_r", 0.0) or 0.0)
    avg_trade_r = float(summary.get("avg_trade_r", 0.0) or 0.0)
    max_drawdown_r = abs(float(summary.get("max_drawdown_r", 0.0) or 0.0))
    count_credit = min(trade_count, min_trade_count)

    score = (
        total_pnl_weight * total_pnl_r
        + avg_trade_weight * avg_trade_r
        - max_drawdown_penalty * max_drawdown_r
        + trade_count_bonus * count_credit
    )
    return {
        "score": score,
        "weights": {
            "total_pnl_weight": total_pnl_weight,
            "avg_trade_weight": avg_trade_weight,
            "max_drawdown_penalty": max_drawdown_penalty,
            "trade_count_bonus": trade_count_bonus,
            "min_trade_count": min_trade_count,
        },
    }
