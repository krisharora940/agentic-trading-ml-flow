from __future__ import annotations

from typing import Any


def build_regime_features(
    session: Any, pre_trigger: Any, candidate: Any
) -> dict[str, float]:
    if pre_trigger.empty:
        return _default_features()

    closes = pre_trigger["close"].astype(float)
    returns = closes.pct_change().dropna()
    last_10 = returns.tail(10)
    last_30 = returns.tail(30)
    range_10 = pre_trigger.tail(10)
    range_30 = pre_trigger.tail(30)

    vol_10 = float(last_10.std()) if len(last_10) > 1 else 0.0
    vol_30 = float(last_30.std()) if len(last_30) > 1 else 0.0
    trend_10 = _net_return(pre_trigger.tail(10))
    trend_30 = _net_return(pre_trigger.tail(30))
    chop_10 = _chop_ratio(last_10)
    chop_30 = _chop_ratio(last_30)
    range_ratio_10 = _price_range(range_10) / max(candidate.zone.width, 0.25)
    range_ratio_30 = _price_range(range_30) / max(candidate.zone.width, 0.25)
    vol_ratio = vol_10 / vol_30 if vol_30 > 0 else 0.0
    trend_strength_10 = trend_10 / vol_10 if vol_10 > 0 else 0.0
    trend_strength_30 = trend_30 / vol_30 if vol_30 > 0 else 0.0

    return {
        "reg_vol_10": vol_10,
        "reg_vol_30": vol_30,
        "reg_vol_ratio": vol_ratio,
        "reg_trend_10": trend_10,
        "reg_trend_30": trend_30,
        "reg_trend_strength_10": trend_strength_10,
        "reg_trend_strength_30": trend_strength_30,
        "reg_chop_10": chop_10,
        "reg_chop_30": chop_30,
        "reg_range_ratio_10": range_ratio_10,
        "reg_range_ratio_30": range_ratio_30,
        "reg_pretrigger_bar_count": float(len(pre_trigger)),
        "reg_high_vol_state": 1.0 if vol_ratio > 1.1 else 0.0,
        "reg_trending_state": (
            1.0 if abs(trend_strength_10) > 1.0 and chop_10 < 8.0 else 0.0
        ),
    }


def _default_features() -> dict[str, float]:
    return {
        "reg_vol_10": 0.0,
        "reg_vol_30": 0.0,
        "reg_vol_ratio": 0.0,
        "reg_trend_10": 0.0,
        "reg_trend_30": 0.0,
        "reg_trend_strength_10": 0.0,
        "reg_trend_strength_30": 0.0,
        "reg_chop_10": 0.0,
        "reg_chop_30": 0.0,
        "reg_range_ratio_10": 0.0,
        "reg_range_ratio_30": 0.0,
        "reg_pretrigger_bar_count": 0.0,
        "reg_high_vol_state": 0.0,
        "reg_trending_state": 0.0,
    }


def _net_return(frame: Any) -> float:
    if frame.empty:
        return 0.0
    first_open = float(frame.iloc[0]["open"])
    last_close = float(frame.iloc[-1]["close"])
    return (last_close - first_open) / first_open if first_open else 0.0


def _price_range(frame: Any) -> float:
    if frame.empty:
        return 0.0
    return float(frame["high"].max() - frame["low"].min())


def _chop_ratio(returns: Any) -> float:
    if len(returns) == 0:
        return 0.0
    abs_sum = float(returns.abs().sum())
    net = abs(float(returns.sum()))
    return abs_sum / max(net, 1e-9)
