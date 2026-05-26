from __future__ import annotations

from typing import Any


def build_feature_validation(
    features: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels)
    if features_df.empty:
        return {"status": "pending", "reason": "empty_features"}
    if labels_df.empty or not {"candidate_id", "label"} <= set(labels_df.columns):
        return {"status": "pending", "reason": "empty_labels"}
    merged = features_df.merge(
        labels_df[["candidate_id", "label"]], on="candidate_id", how="inner"
    )
    if merged.empty:
        return {"status": "pending", "reason": "empty_merged_dataset"}

    numeric_cols = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label"}
        and pd.api.types.is_numeric_dtype(merged[col])
    ]
    rows = []
    for col in numeric_cols:
        series = merged[col]
        missing_rate = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))
        corr = series.corr(merged["label"])
        rows.append(
            {
                "feature": col,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "label_corr": float(corr) if corr == corr else None,
                "is_constant": unique_count <= 1,
            }
        )

    failed = [row for row in rows if row["is_constant"] or row["missing_rate"] > 0.25]
    return {
        "status": "complete",
        "feature_count": len(rows),
        "failed_count": len(failed),
        "failed_features": failed[:15],
        "top_abs_label_corr": sorted(
            [row for row in rows if row["label_corr"] is not None],
            key=lambda row: abs(float(row["label_corr"])),
            reverse=True,
        )[:15],
    }
