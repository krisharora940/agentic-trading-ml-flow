from __future__ import annotations

import os
from typing import Any

from trading_ml.config import load_bnr_config


def load_engineer_feature_config() -> dict[str, Any]:
    config = load_bnr_config()
    engineer = dict(config.get("engineer_features", {}))
    engineer.setdefault("backend", "hybrid")
    engineer.setdefault("enabled", True)
    engineer.setdefault("disable_numba_jit", True)
    engineer.setdefault("features", ["rsi", "atr", "ema", "mfi", "stddev", "choppiness_index"])
    return engineer


def compute_engineer_features(bars: Any, *, features: list[str] | None = None) -> Any | None:
    config = load_engineer_feature_config()
    if not config.get("enabled", True):
        return None

    feature_list = list(features or config.get("features", []))
    if not feature_list:
        return None

    if config.get("disable_numba_jit", True):
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    try:
        import polars as pl
        from ml4t.engineer import compute_features
    except Exception:
        return None

    pd = _require_pandas()
    frame = bars.reset_index().copy()
    timestamp_col = frame.columns[0]
    frame = frame.rename(columns={timestamp_col: "ts_event"})
    if "symbol" not in frame.columns:
        frame["symbol"] = "UNKNOWN"
    if "source_timeframe" not in frame.columns:
        frame["source_timeframe"] = "unknown"
    if "ts_event_utc" not in frame.columns:
        frame["ts_event_utc"] = pd.to_datetime(frame["ts_event"], utc=True)

    base_columns = set(frame.columns)
    featured = compute_features(pl.from_pandas(frame), feature_list).to_pandas()
    extra_columns = [column for column in featured.columns if column not in base_columns]
    if not extra_columns:
        return None

    result = featured[["ts_event", *extra_columns]].copy()
    result["ts_event"] = pd.to_datetime(result["ts_event"])
    return result.set_index("ts_event").sort_index()


def engineer_feature_snapshot(feature_frame: Any | None, cutoff: Any) -> dict[str, float]:
    if feature_frame is None or feature_frame.empty:
        return {}

    snapshot = feature_frame[feature_frame.index < cutoff].tail(1)
    if snapshot.empty:
        return {}

    row = snapshot.iloc[0]
    values: dict[str, float] = {}
    for column, value in row.items():
        if _is_numeric(value):
            values[f"eng_{column}"] = float(value)
    return values


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("ml4t-engineer adapter requires pandas.") from exc
    return pd
