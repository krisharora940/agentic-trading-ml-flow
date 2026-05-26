from __future__ import annotations

from typing import Any


def build_feature_diagnostics(stage2_result: dict[str, Any]) -> dict[str, Any]:
    try:
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    features = list(stage2_result.get("features_records", []))
    labels = list(stage2_result.get("labels_records", []))
    if not features or not labels:
        return {"status": "pending", "reason": "missing_artifacts"}

    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels)
    merged = features_df.merge(
        labels_df[["candidate_id", "label"]], on="candidate_id", how="inner"
    )
    if merged.empty or merged["label"].nunique() < 2:
        return {"status": "pending", "reason": "insufficient_class_diversity"}

    merged = merged.sort_values("session_date").reset_index(drop=True)
    split = max(1, int(len(merged) * 0.7))
    train = merged.iloc[:split]
    test = merged.iloc[split:]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        return {"status": "pending", "reason": "insufficient_holdout_diversity"}

    numeric_features = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label"}
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    scored: list[dict[str, Any]] = []
    for column in numeric_features:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(train[[column]], train["label"])
        probabilities = model.predict_proba(test[[column]])[:, 1]
        scored.append(
            {
                "feature": column,
                "single_feature_roc_auc": float(
                    roc_auc_score(test["label"], probabilities)
                ),
            }
        )

    ranked = sorted(
        scored, key=lambda item: item["single_feature_roc_auc"], reverse=True
    )
    return {
        "status": "complete",
        "feature_count": len(numeric_features),
        "top_features": ranked[:8],
        "weak_features": ranked[-5:],
    }
