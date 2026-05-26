from __future__ import annotations

from typing import Any


def build_failure_map(
    stage2_result: dict[str, Any],
    stitched_prediction_records: list[dict[str, Any]],
    executed_summary: dict[str, Any],
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    features = pd.DataFrame(stage2_result.get("features_records", []))
    labels = pd.DataFrame(stage2_result.get("labels_records", []))
    stitched = pd.DataFrame(stitched_prediction_records or [])
    executed = pd.DataFrame(executed_summary.get("equity_curve", []) or [])

    if features.empty or labels.empty or stitched.empty:
        return {"status": "pending", "reason": "missing_artifacts"}

    feature_columns = [
        "candidate_id",
        "setup_subtype",
        "reg_high_vol_state",
        "reg_trending_state",
        "trigger_seconds_after_open",
        "break_body_fraction",
        "break_efficiency_ratio",
        "reclaim_close_location",
        "reclaim_failure_count",
    ]
    available_feature_columns = [
        column for column in feature_columns if column in features.columns
    ]
    feature_frame = features[available_feature_columns].copy()
    for column in feature_columns:
        if column not in feature_frame.columns:
            feature_frame[column] = 0.0 if column != "setup_subtype" else "unknown"

    base = stitched.merge(
        labels[["candidate_id", "label", "outcome", "pnl_r", "bars_held"]],
        on="candidate_id",
        how="left",
        suffixes=("", "_label"),
    ).merge(
        feature_frame[feature_columns],
        on="candidate_id",
        how="left",
    )
    for column in feature_columns[1:]:
        if column not in base.columns:
            left = f"{column}_x"
            right = f"{column}_y"
            if left in base.columns and right in base.columns:
                base[column] = base[left].where(base[left].notna(), base[right])
            elif left in base.columns:
                base[column] = base[left]
            elif right in base.columns:
                base[column] = base[right]
    if "setup_subtype" not in base.columns:
        base["setup_subtype"] = "unknown"

    base["session_month"] = base["session_date"].astype(str).str.slice(0, 7)
    base["regime_bucket"] = base.apply(
        lambda row: (
            f"hv{int(row.get('reg_high_vol_state', 0) or 0)}_tr{int(row.get('reg_trending_state', 0) or 0)}"
        ),
        axis=1,
    )

    executed_ids = (
        set(executed["candidate_id"].tolist())
        if not executed.empty and "candidate_id" in executed
        else set()
    )
    base["executed"] = base["candidate_id"].isin(executed_ids)

    return {
        "status": "complete",
        "sample": {
            "candidate_predictions": int(len(base)),
            "executed_trades": int(base["executed"].sum()),
        },
        "by_month": _group_rows(base, "session_month"),
        "by_subtype": _group_rows(base, "setup_subtype"),
        "by_regime": _group_rows(base, "regime_bucket"),
        "executed_failures_by_subtype": _executed_failures(base),
    }


def _group_rows(frame: Any, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(column, dropna=False)
    for key, group in grouped:
        rows.append(
            {
                column: str(key),
                "rows": int(len(group)),
                "positive_rate": (
                    float(group["label"].mean()) if "label" in group else 0.0
                ),
                "avg_pnl_r": float(group["pnl_r"].mean()) if "pnl_r" in group else 0.0,
                "avg_probability": (
                    float(group["probability"].mean())
                    if "probability" in group
                    else 0.0
                ),
                "executed_share": (
                    float(group["executed"].mean()) if "executed" in group else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda row: row["rows"], reverse=True)


def _executed_failures(frame: Any) -> list[dict[str, Any]]:
    executed = frame[frame["executed"]].copy()
    if executed.empty:
        return []
    failures = executed[executed["pnl_r"] <= 0].groupby("setup_subtype", dropna=False)
    rows: list[dict[str, Any]] = []
    for subtype, group in failures:
        rows.append(
            {
                "setup_subtype": str(subtype),
                "rows": int(len(group)),
                "avg_pnl_r": float(group["pnl_r"].mean()),
                "avg_break_efficiency_ratio": (
                    float(group["break_efficiency_ratio"].mean())
                    if "break_efficiency_ratio" in group
                    else 0.0
                ),
                "avg_reclaim_close_location": (
                    float(group["reclaim_close_location"].mean())
                    if "reclaim_close_location" in group
                    else 0.0
                ),
                "avg_reclaim_failure_count": (
                    float(group["reclaim_failure_count"].mean())
                    if "reclaim_failure_count" in group
                    else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda row: row["rows"], reverse=True)
