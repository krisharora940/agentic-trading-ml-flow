from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

from trading_ml.config import load_bnr_config, load_global_config
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_modeling import score_model_split


def build_cpcv_audit(stage2_result: dict[str, Any]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    features = pd.DataFrame(stage2_result.get("features_records", []))
    labels = pd.DataFrame(stage2_result.get("labels_records", []))
    if features.empty or labels.empty:
        return {"status": "pending", "reason": "missing_artifacts"}

    merged = features.merge(labels, on="candidate_id", how="inner")
    if merged.empty or merged["label"].nunique() < 2:
        return {"status": "pending", "reason": "insufficient_labels"}

    merged = merged.sort_values("entry_time").reset_index(drop=True)
    session_dates = sorted(merged["session_date"].dropna().astype(str).unique().tolist())
    if len(session_dates) < 40:
        return {"status": "pending", "reason": "insufficient_sessions", "session_count": len(session_dates)}

    global_config = load_global_config()
    bnr_config = load_bnr_config()
    validation_cfg = dict(global_config.get("validation", {}))
    threshold = float(bnr_config.get("frozen_benchmark", {}).get("threshold", 0.45) or 0.45)
    n_groups = min(8, max(6, len(session_dates) // 20))
    n_test_groups = 2
    max_combinations = 20
    embargo_bars = int(validation_cfg.get("embargo_bars", 0) or 0)
    label_horizon = int(stage2_result.get("config", {}).get("horizon_bars", 20) or 20)
    feature_cols = _feature_columns(merged)
    if not feature_cols:
        return {"status": "pending", "reason": "no_feature_columns"}

    groups = _partition_sessions(session_dates, n_groups)
    group_index = {session: idx for idx, bucket in enumerate(groups) for session in bucket}
    total_paths = comb(n_groups, n_test_groups)
    selected_combinations = list(combinations(range(n_groups), n_test_groups))[:max_combinations]
    rows: list[dict[str, Any]] = []
    negative_paths = 0

    model_family = str(stage2_result.get("config", {}).get("model_family", "linear_baseline"))
    for fold_id, test_groups in enumerate(selected_combinations, start=1):
        test_sessions = sorted(session for idx in test_groups for session in groups[idx])
        test = merged[merged["session_date"].astype(str).isin(test_sessions)].copy()
        if test.empty or test["label"].nunique() < 2:
            continue
        test_min_idx = min(test.index)
        test_max_idx = max(test.index)
        purge_start = max(test_min_idx - label_horizon, 0)
        embargo_end = min(test_max_idx + embargo_bars, len(merged) - 1)
        train = merged[(merged.index < purge_start) | (merged.index > embargo_end)].copy()
        train = train[~train["session_date"].astype(str).isin(test_sessions)]
        if train.empty or train["label"].nunique() < 2:
            continue

        scored = score_model_split(train, test, model_family=model_family, feature_cols=feature_cols)
        prediction_frame = scored["prediction_frame"].copy()
        execution = run_event_driven_policy_backtest(prediction_frame.to_dict(orient="records"), threshold=threshold)
        total_pnl_r = float(execution.get("total_pnl_r", 0.0) or 0.0)
        if total_pnl_r <= 0:
            negative_paths += 1
        rows.append(
            {
                "fold": fold_id,
                "test_groups": list(test_groups),
                "test_sessions": test_sessions,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "roc_auc": float(scored["metrics"].get("roc_auc", 0.0) or 0.0),
                "precision": float(scored["metrics"].get("precision", 0.0) or 0.0),
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": total_pnl_r,
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
            }
        )

    if not rows:
        return {"status": "pending", "reason": "no_valid_cpcv_paths", "total_paths": total_paths}

    pbo = negative_paths / len(rows)
    mean_pnl = sum(row["total_pnl_r"] for row in rows) / len(rows)
    mean_roc_auc = sum(row["roc_auc"] for row in rows) / len(rows)
    status = "pass" if pbo <= 0.50 and mean_pnl > 0 else "fail"
    return {
        "status": status,
        "backend": "local_cpcv",
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "label_horizon": label_horizon,
        "embargo_bars": embargo_bars,
        "total_paths": total_paths,
        "evaluated_paths": len(rows),
        "pbo": pbo,
        "mean_total_pnl_r": mean_pnl,
        "mean_roc_auc": mean_roc_auc,
        "paths": rows,
    }


def _partition_sessions(session_dates: list[str], n_groups: int) -> list[list[str]]:
    size = max(len(session_dates) // n_groups, 1)
    groups: list[list[str]] = []
    for idx in range(n_groups):
        start = idx * size
        end = (idx + 1) * size if idx < n_groups - 1 else len(session_dates)
        bucket = session_dates[start:end]
        if bucket:
            groups.append(bucket)
    return groups


def _feature_columns(merged: Any) -> list[str]:
    import pandas as pd

    exclude = {
        "candidate_id",
        "session_date",
        "label",
        "outcome",
        "entry_time",
        "exit_time",
        "entry_price",
        "stop_price",
        "target_price",
        "exit_price",
        "bars_held",
        "mfe",
        "mae",
        "pnl_r",
    }
    return [
        col
        for col in merged.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(merged[col]) and not merged[col].isna().all()
    ]
