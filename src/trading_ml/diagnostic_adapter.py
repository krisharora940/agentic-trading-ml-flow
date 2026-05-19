from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trading_ml.config import load_global_config
from trading_ml.validation_types import WalkForwardFold


def prepare_diagnostic_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    mpl_dir = Path("/private/tmp/mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def diagnostic_available() -> bool:
    prepare_diagnostic_runtime()
    try:
        import ml4t.diagnostic  # noqa: F401
    except Exception:
        return False
    return True


def build_diagnostic_walk_forward_splits(merged: Any) -> tuple[list[tuple[Any, Any, WalkForwardFold]], dict[str, Any]] | None:
    if not diagnostic_available():
        return None

    from ml4t.diagnostic.splitters import WalkForwardCV, WalkForwardConfig

    pd = _require_pandas()
    global_config = load_global_config()
    validation = dict(global_config.get("validation", {}))
    session_count = len(sorted(merged["session_date"].dropna().astype(str).unique().tolist()))
    min_train_sessions = int(validation.get("min_train_sessions", 10) or 10)
    test_sessions = int(validation.get("test_sessions", 5) or 5)
    step_sessions = int(validation.get("step_sessions", test_sessions) or test_sessions)
    embargo_bars = int(validation.get("embargo_bars", 0) or 0)
    n_splits = max((session_count - min_train_sessions) // max(step_sessions, 1), 1)

    frame = merged.copy().sort_values("entry_time").reset_index(drop=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], errors="coerce", utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], errors="coerce", utc=True)
    config = WalkForwardConfig(
        n_splits=n_splits,
        train_size=min_train_sessions,
        test_size=test_sessions,
        step_size=step_sessions,
        label_horizon=embargo_bars,
        session_col="session_date",
        timestamp_col="entry_time",
        calendar_id="CME_Equity",
    )
    cv = WalkForwardCV(config=config)

    folds: list[tuple[Any, Any, WalkForwardFold]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(frame), start=1):
        train = frame.iloc[list(train_idx)].copy()
        test = frame.iloc[list(test_idx)].copy()
        if train.empty or test.empty:
            continue
        folds.append(
            (
                train,
                test,
                WalkForwardFold(
                    fold=fold_idx,
                    train_sessions=sorted(train["session_date"].astype(str).unique().tolist()),
                    test_sessions=sorted(test["session_date"].astype(str).unique().tolist()),
                    train_rows=int(len(train)),
                    test_rows=int(len(test)),
                    purged_rows=0,
                    embargo_rows=0,
                ),
            )
        )

    metadata = {
        "backend": "ml4t_diagnostic",
        "min_train_sessions": min_train_sessions,
        "test_sessions": test_sessions,
        "step_sessions": step_sessions,
        "embargo_bars": embargo_bars,
        "session_count": session_count,
    }
    return folds, metadata


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Diagnostic adapter requires pandas.") from exc
    return pd
