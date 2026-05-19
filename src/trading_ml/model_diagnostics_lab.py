from __future__ import annotations

from typing import Any


def build_model_diagnostics_lab(
    prediction_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]] | None = None,
    labels_records: list[dict[str, Any]] | None = None,
    *,
    model_family: str = "linear_baseline",
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    frame = pd.DataFrame(prediction_records)
    if frame.empty or "probability" not in frame or "label" not in frame:
        return {"status": "pending", "reason": "invalid_prediction_records"}

    bucket_count = min(5, max(2, int(len(frame) ** 0.5)))
    ranked = frame.sort_values("probability").reset_index(drop=True)
    ranked["bucket"] = pd.qcut(ranked.index, q=bucket_count, labels=False, duplicates="drop")

    bucket_rows = []
    for bucket, group in ranked.groupby("bucket"):
        bucket_rows.append(
            {
                "bucket": int(bucket),
                "count": int(len(group)),
                "probability_mean": float(group["probability"].mean()),
                "positive_rate": float(group["label"].mean()),
                "avg_pnl_r": float(group["pnl_r"].mean()) if "pnl_r" in group else 0.0,
            }
        )

    monotonic = _is_monotonic(bucket_rows)
    uncertainty = {
        "probability_mean": float(frame["probability"].mean()),
        "probability_std": float(frame["probability"].std(ddof=0)) if len(frame) > 1 else 0.0,
        "high_confidence_share": float((frame["probability"] >= 0.7).mean()),
        "low_confidence_share": float((frame["probability"] <= 0.3).mean()),
    }
    calibration = _build_calibration_review(frame)
    return {
        "status": "complete",
        "bucket_rows": bucket_rows,
        "bucket_monotonicity": monotonic,
        "uncertainty_review": uncertainty,
        "calibration_review": calibration,
        "shap_analysis": _build_shap_analysis(feature_records or [], labels_records or [], model_family=model_family),
    }


def _is_monotonic(bucket_rows: list[dict[str, Any]]) -> bool:
    if len(bucket_rows) < 2:
        return False
    positives = [row["positive_rate"] for row in sorted(bucket_rows, key=lambda row: row["probability_mean"])]
    return all(later >= earlier for earlier, later in zip(positives, positives[1:], strict=False))


def _build_calibration_review(frame: Any) -> dict[str, Any]:
    bucket_count = min(10, max(3, int(len(frame) ** 0.5)))
    ranked = frame.sort_values("probability").reset_index(drop=True).copy()
    ranked["calibration_bucket"] = ranked.index
    ranked["calibration_bucket"] = ranked["calibration_bucket"].apply(
        lambda idx: min(bucket_count - 1, int(idx * bucket_count / max(len(ranked), 1)))
    )
    rows = []
    ece = 0.0
    uncertainty = float(frame["label"].mean()) * (1.0 - float(frame["label"].mean()))
    reliability = 0.0
    resolution = 0.0
    for bucket, group in ranked.groupby("calibration_bucket"):
        count = int(len(group))
        if count <= 0:
            continue
        prob_mean = float(group["probability"].mean())
        hit_rate = float(group["label"].mean())
        weight = count / max(len(ranked), 1)
        gap = abs(prob_mean - hit_rate)
        ece += weight * gap
        reliability += weight * ((prob_mean - hit_rate) ** 2)
        resolution += weight * ((hit_rate - float(frame["label"].mean())) ** 2)
        rows.append(
            {
                "bucket": int(bucket),
                "count": count,
                "probability_mean": prob_mean,
                "hit_rate": hit_rate,
                "abs_gap": gap,
            }
        )
    status = "pass" if ece <= 0.08 else "fail"
    return {
        "status": status,
        "ece": ece,
        "brier_decomposition": {
            "uncertainty": uncertainty,
            "reliability": reliability,
            "resolution": resolution,
        },
        "reliability_rows": rows,
    }


def _build_shap_analysis(
    feature_records: list[dict[str, Any]],
    labels_records: list[dict[str, Any]],
    *,
    model_family: str,
) -> dict[str, Any]:
    try:
        import pandas as pd
        import shap
        from sklearn.impute import SimpleImputer
    except ImportError:
        return {"status": "pending", "reason": "missing_optional_dependencies"}

    from trading_ml.stage2_modeling import build_classifier

    if not feature_records or not labels_records:
        return {"status": "pending", "reason": "missing_artifacts"}

    features = pd.DataFrame(feature_records)
    labels = pd.DataFrame(labels_records)
    merged = features.merge(labels, on="candidate_id", how="inner").sort_values("session_date").reset_index(drop=True)
    if len(merged) < 20 or merged["label"].nunique() < 2:
        return {"status": "pending", "reason": "insufficient_rows"}

    feature_cols = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label", "outcome", "entry_time", "exit_time", "entry_price", "stop_price", "target_price", "exit_price", "bars_held", "mfe", "mae", "pnl_r"}
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    if not feature_cols:
        return {"status": "pending", "reason": "no_feature_columns"}

    split = max(1, int(len(merged) * 0.7))
    train = merged.iloc[:split].copy()
    test = merged.iloc[split:].copy()
    if test.empty:
        return {"status": "pending", "reason": "empty_test_split"}

    model_name, model = build_classifier(model_family)
    model.fit(train[feature_cols], train["label"])

    imputer = SimpleImputer(strategy="median")
    background = imputer.fit_transform(train[feature_cols])
    sample = imputer.transform(test[feature_cols].head(25))
    feature_names = feature_cols

    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1]
    try:
        explainer = shap.Explainer(predict_fn, background[:25], feature_names=feature_names)
        shap_values = explainer(sample)
        values = getattr(shap_values, "values", None)
        if values is None:
            return {"status": "pending", "reason": "invalid_shap_output"}
        abs_mean = abs(values).mean(axis=0)
        ranked = sorted(
            [{"feature": feature_names[idx], "mean_abs_shap": float(score)} for idx, score in enumerate(abs_mean)],
            key=lambda row: row["mean_abs_shap"],
            reverse=True,
        )
        worst = test.nsmallest(min(5, len(test)), "pnl_r").copy()
        worst_sample = imputer.transform(worst[feature_cols])
        worst_shap = explainer(worst_sample)
        worst_rows = []
        for row_idx, (_, trade) in enumerate(worst.iterrows()):
            contributions = worst_shap.values[row_idx]
            top_idx = sorted(range(len(feature_names)), key=lambda idx: abs(contributions[idx]), reverse=True)[:3]
            worst_rows.append(
                {
                    "candidate_id": str(trade["candidate_id"]),
                    "session_date": str(trade["session_date"]),
                    "pnl_r": float(trade.get("pnl_r", 0.0) or 0.0),
                    "top_contributors": [
                        {"feature": feature_names[idx], "shap_value": float(contributions[idx])} for idx in top_idx
                    ],
                }
            )
        return {
            "status": "complete",
            "backend": "shap",
            "model_name": model_name,
            "top_features": ranked[:10],
            "worst_trade_explanations": worst_rows,
        }
    except Exception as exc:
        return {
            "status": "pending",
            "reason": "shap_runtime_failed",
            "detail": str(exc),
        }
