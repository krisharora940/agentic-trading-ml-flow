from __future__ import annotations

from itertools import combinations
from math import comb
import json
from pathlib import Path
import random
from typing import Any

from trading_ml.config import load_bnr_config, load_global_config
from trading_ml.deflated_sharpe_analysis import compute_sharpe_ratio
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_modeling import score_model_split


def build_cpcv_audit(
    stage2_result: dict[str, Any], artifact_context: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    session_dates = sorted(
        merged["session_date"].dropna().astype(str).unique().tolist()
    )
    if len(session_dates) < 40:
        return {
            "status": "pending",
            "reason": "insufficient_sessions",
            "session_count": len(session_dates),
        }

    global_config = load_global_config()
    bnr_config = load_bnr_config()
    validation_cfg = dict(global_config.get("validation", {}))
    threshold = float(
        bnr_config.get("frozen_benchmark", {}).get("threshold", 0.45) or 0.45
    )
    n_groups = min(8, max(6, len(session_dates) // 20))
    n_test_groups = 2
    max_combinations = 20
    embargo_bars = int(validation_cfg.get("embargo_bars", 0) or 0)
    label_horizon = int(stage2_result.get("config", {}).get("horizon_bars", 20) or 20)
    feature_cols = _feature_columns(merged)
    if not feature_cols:
        return {"status": "pending", "reason": "no_feature_columns"}

    groups = _partition_sessions(session_dates, n_groups)
    total_paths = comb(n_groups, n_test_groups)
    selected_combinations = list(combinations(range(n_groups), n_test_groups))
    rng = random.Random(42)
    rng.shuffle(selected_combinations)
    selected_combinations = selected_combinations[:max_combinations]
    rows: list[dict[str, Any]] = []
    negative_paths = 0

    model_family = str(
        stage2_result.get("config", {}).get("model_family", "linear_baseline")
    )
    run_id = str((artifact_context or {}).get("run_id", "") or "")
    artifact_root = _artifact_root(run_id) if run_id else None

    for fold_id, test_groups in enumerate(selected_combinations, start=1):
        test_sessions = sorted(
            session for idx in test_groups for session in groups[idx]
        )
        test = merged[merged["session_date"].astype(str).isin(test_sessions)].copy()
        if test.empty or test["label"].nunique() < 2:
            continue
        test_min_idx = min(test.index)
        test_max_idx = max(test.index)
        purge_start = max(test_min_idx - label_horizon, 0)
        embargo_end = min(test_max_idx + embargo_bars, len(merged) - 1)
        train = merged[
            (merged.index < purge_start) | (merged.index > embargo_end)
        ].copy()
        train = train[~train["session_date"].astype(str).isin(test_sessions)]
        if train.empty or train["label"].nunique() < 2:
            continue

        scored = score_model_split(
            train, test, model_family=model_family, feature_cols=feature_cols
        )
        prediction_frame = scored["prediction_frame"].copy()
        execution = run_event_driven_policy_backtest(
            prediction_frame.to_dict(orient="records"), threshold=threshold
        )
        total_pnl_r = float(execution.get("total_pnl_r", 0.0) or 0.0)
        path_returns = [
            float(row.get("executed_pnl_r", 0.0) or 0.0)
            for row in execution.get("equity_curve", [])
        ]
        sharpe_r = compute_sharpe_ratio(path_returns)
        attribution, persisted_rows = _path_attribution(prediction_frame, execution)
        rows_artifact = (
            _persist_cpcv_path_rows(artifact_root, fold_id, persisted_rows)
            if artifact_root is not None
            else None
        )
        if total_pnl_r <= 0:
            negative_paths += 1
        rows.append(
            {
                "path_id": f"cpcv_{fold_id:03d}",
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
                "win_rate": float(execution.get("win_rate", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "sharpe_r": sharpe_r,
                "rows_artifact": (
                    str(rows_artifact) if rows_artifact is not None else None
                ),
                **attribution,
            }
        )

    if not rows:
        return {
            "status": "pending",
            "reason": "no_valid_cpcv_paths",
            "total_paths": total_paths,
        }

    pbo = negative_paths / len(rows)
    mean_pnl = sum(row["total_pnl_r"] for row in rows) / len(rows)
    sorted_pnls = sorted(row["total_pnl_r"] for row in rows)
    median_pnl = sorted_pnls[len(rows) // 2]
    min_path_pnl = min(row["total_pnl_r"] for row in rows)
    path_positive_rate = sum(1 for row in rows if row["total_pnl_r"] > 0) / len(rows)
    mean_roc_auc = sum(row["roc_auc"] for row in rows) / len(rows)
    distribution = _path_distribution(sorted_pnls)
    ranked_rows = sorted(rows, key=lambda row: row["total_pnl_r"])
    status = (
        "pass"
        if pbo <= 0.25
        and mean_pnl > 0
        and median_pnl > 0
        and path_positive_rate >= 0.60
        and min_path_pnl > -5.0
        else "fail"
    )
    return {
        "status": status,
        "backend": "local_cpcv",
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "label_horizon": label_horizon,
        "embargo_bars": embargo_bars,
        "total_paths": total_paths,
        "evaluated_paths": len(rows),
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "pbo": pbo,
        "mean_total_pnl_r": mean_pnl,
        "median_total_pnl_r": median_pnl,
        "min_path_pnl_r": min_path_pnl,
        "path_positive_rate": path_positive_rate,
        "distribution": distribution,
        "worst_paths": ranked_rows[:3],
        "best_paths": list(reversed(ranked_rows[-3:])),
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
        if col not in exclude
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]


def _path_distribution(sorted_pnls: list[float]) -> dict[str, float]:
    def pct(p: float) -> float:
        if not sorted_pnls:
            return 0.0
        idx = min(len(sorted_pnls) - 1, max(0, int(round((len(sorted_pnls) - 1) * p))))
        return float(sorted_pnls[idx])

    negative_tail = [value for value in sorted_pnls if value < 0]
    return {
        "p10_total_pnl_r": pct(0.10),
        "p25_total_pnl_r": pct(0.25),
        "p75_total_pnl_r": pct(0.75),
        "p90_total_pnl_r": pct(0.90),
        "negative_tail_contribution_r": float(sum(negative_tail)),
    }


def _path_attribution(
    prediction_frame: Any, execution: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import pandas as pd
    except ImportError:
        return {}

    executed = pd.DataFrame(execution.get("equity_curve", []) or [])
    if executed.empty:
        return {
            "largest_loss_cluster_r": 0.0,
            "subtype_breakdown": [],
            "time_of_day_breakdown": [],
            "volatility_regime_breakdown": [],
            "trend_regime_breakdown": [],
            "threshold_distribution": [],
            "path_calibration": {},
        }, []

    frame = prediction_frame.copy()
    executed_ids = set(executed["candidate_id"].astype(str).tolist())
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame = frame[frame["candidate_id"].isin(executed_ids)].copy()
    merged = frame.merge(
        executed[["candidate_id", "executed_pnl_r"]],
        on="candidate_id",
        how="left",
    )
    if "entry_time" in merged.columns:
        merged["entry_time"] = pd.to_datetime(
            merged["entry_time"], errors="coerce", utc=True
        )
        merged["time_bucket"] = merged["entry_time"].dt.strftime("%H:%M")
    else:
        merged["time_bucket"] = "unknown"

    persisted_rows = merged.to_dict(orient="records")
    return {
        "largest_loss_cluster_r": _largest_loss_cluster(executed),
        "subtype_breakdown": _simple_group(merged, "setup_subtype"),
        "time_of_day_breakdown": _simple_group(merged, "time_bucket"),
        "volatility_regime_breakdown": _simple_group(merged, "reg_high_vol_state"),
        "trend_regime_breakdown": _simple_group(merged, "reg_trending_state"),
        "threshold_distribution": _probability_bins(merged),
        "path_calibration": _calibration_rows(merged),
    }, persisted_rows


def _largest_loss_cluster(executed: Any) -> float:
    running = 0.0
    worst = 0.0
    for pnl in executed["executed_pnl_r"].astype(float).tolist():
        if pnl < 0:
            running += pnl
            worst = min(worst, running)
        else:
            running = 0.0
    return float(worst)


def _simple_group(frame: Any, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    rows = []
    for key, group in frame.groupby(column, dropna=False):
        rows.append(
            {
                "key": str(key),
                "trade_count": int(len(group)),
                "total_pnl_r": (
                    float(group["executed_pnl_r"].sum())
                    if "executed_pnl_r" in group
                    else 0.0
                ),
                "avg_trade_r": (
                    float(group["executed_pnl_r"].mean())
                    if "executed_pnl_r" in group
                    else 0.0
                ),
                "win_rate": (
                    float((group["executed_pnl_r"] > 0).mean())
                    if "executed_pnl_r" in group
                    else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda row: row["trade_count"], reverse=True)


def _probability_bins(frame: Any) -> list[dict[str, Any]]:
    if "probability" not in frame.columns or frame.empty:
        return []
    bins = [(0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]
    rows = []
    for low, high in bins:
        group = frame[(frame["probability"] >= low) & (frame["probability"] < high)]
        rows.append(
            {
                "bucket": f"[{low:.2f},{high:.2f})",
                "trade_count": int(len(group)),
                "total_pnl_r": (
                    float(group["executed_pnl_r"].sum()) if len(group) else 0.0
                ),
                "avg_trade_r": (
                    float(group["executed_pnl_r"].mean()) if len(group) else 0.0
                ),
            }
        )
    return rows


def _calibration_rows(frame: Any) -> dict[str, Any]:
    import pandas as pd

    if (
        "probability" not in frame.columns
        or "label" not in frame.columns
        or frame.empty
    ):
        return {}
    ranked = frame.sort_values("probability").reset_index(drop=True).copy()
    bucket_count = min(5, max(2, int(len(ranked) ** 0.5)))
    ranked["bucket"] = pd.qcut(
        ranked.index, q=bucket_count, labels=False, duplicates="drop"
    )
    rows = []
    for bucket, group in ranked.groupby("bucket"):
        rows.append(
            {
                "bucket": int(bucket),
                "count": int(len(group)),
                "probability_mean": float(group["probability"].mean()),
                "hit_rate": float(group["label"].mean()),
            }
        )
    return {"rows": rows}


def _artifact_root(run_id: str) -> Path:
    root = REPORTS_DIR / "runs" / run_id / "cpcv"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _persist_cpcv_path_rows(
    root: Path, fold_id: int, rows: list[dict[str, Any]]
) -> Path:
    path = root / f"path_cpcv_{fold_id:03d}_rows.json"
    path.write_text(json.dumps(rows, default=str, indent=2), encoding="utf-8")
    return path
