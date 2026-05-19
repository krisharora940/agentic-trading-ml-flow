from __future__ import annotations

from datetime import timedelta
from typing import Any

from trading_ml.config import load_global_config
from trading_ml.diagnostic_adapter import build_diagnostic_walk_forward_splits
from trading_ml.validation_types import WalkForwardFold


def build_walk_forward_splits(merged: Any) -> tuple[list[tuple[Any, Any, WalkForwardFold]], dict[str, Any]]:
    diagnostic_result = build_diagnostic_walk_forward_splits(merged)
    if diagnostic_result is not None:
        return diagnostic_result

    pd = _require_pandas()
    global_config = load_global_config()
    validation = dict(global_config.get("validation", {}))
    timeframe_seconds = 30
    if "entry_time" in merged.columns:
        times = pd.to_datetime(merged["entry_time"], errors="coerce", utc=True)
        diffs = times.sort_values().diff().dropna()
        if not diffs.empty:
            timeframe_seconds = max(int(diffs.min().total_seconds()), 30)

    min_train_sessions = int(validation.get("min_train_sessions", 10) or 10)
    test_sessions = int(validation.get("test_sessions", 5) or 5)
    step_sessions = int(validation.get("step_sessions", test_sessions) or test_sessions)
    embargo_bars = int(validation.get("embargo_bars", 2) or 0)
    unique_sessions = sorted(merged["session_date"].dropna().astype(str).unique().tolist())

    merged = merged.copy()
    merged["entry_time"] = pd.to_datetime(merged["entry_time"], errors="coerce", utc=True)
    merged["exit_time"] = pd.to_datetime(merged["exit_time"], errors="coerce", utc=True)
    folds: list[tuple[Any, Any, WalkForwardFold]] = []
    for fold_idx, start in enumerate(range(min_train_sessions, len(unique_sessions) - test_sessions + 1, step_sessions), start=1):
        train_sessions_list = unique_sessions[:start]
        test_sessions_list = unique_sessions[start : start + test_sessions]
        train = merged[merged["session_date"].astype(str).isin(train_sessions_list)].copy()
        test = merged[merged["session_date"].astype(str).isin(test_sessions_list)].copy()
        if train.empty or test.empty:
            continue
        test_start = test["entry_time"].min()
        embargo_delta = timedelta(seconds=timeframe_seconds * embargo_bars)
        purge_mask = train["exit_time"] >= test_start
        embargo_mask = train["exit_time"] >= (test_start - embargo_delta)
        purged_rows = int(purge_mask.sum())
        embargo_rows = int((embargo_mask & ~purge_mask).sum())
        train = train[~embargo_mask].copy()
        fold = WalkForwardFold(
            fold=fold_idx,
            train_sessions=train_sessions_list,
            test_sessions=test_sessions_list,
            train_rows=int(len(train)),
            test_rows=int(len(test)),
            purged_rows=purged_rows,
            embargo_rows=embargo_rows,
        )
        folds.append((train, test, fold))

    metadata = {
        "backend": "custom",
        "min_train_sessions": min_train_sessions,
        "test_sessions": test_sessions,
        "step_sessions": step_sessions,
        "embargo_bars": embargo_bars,
        "session_count": len(unique_sessions),
    }
    return folds, metadata


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Validation splits require pandas.") from exc
    return pd
