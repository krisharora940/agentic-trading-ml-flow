from __future__ import annotations

from typing import Any

from trading_ml.cpcv_analysis import build_cpcv_audit
from trading_ml.config import load_global_config
from trading_ml.validation_splits import build_walk_forward_splits


def build_validation_audit(stage2_result: dict[str, Any], search_results: dict[str, Any]) -> dict[str, Any]:
    features = list(stage2_result.get("features_records", []))
    labels = list(stage2_result.get("labels_records", []))
    walk_forward = _walk_forward_check(features, labels)
    cpcv = build_cpcv_audit(stage2_result)
    purging = _purging_check(labels)
    multiple_testing = _multiple_testing_check(search_results)
    overfitting = _overfitting_check(walk_forward, cpcv, multiple_testing)
    return {
        "walk_forward": walk_forward,
        "cpcv": cpcv,
        "purging": purging,
        "multiple_testing": multiple_testing,
        "overfitting": overfitting,
    }


def _walk_forward_check(features: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    if not features or not labels:
        return {"status": "pending", "reason": "missing_artifacts"}

    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels)
    merged = features_df.merge(labels_df, on="candidate_id", how="inner")
    if merged.empty:
        return {"status": "pending", "reason": "empty_merged_dataset"}

    folds_input, metadata = build_walk_forward_splits(merged)
    if not folds_input:
        return {"status": "pending", "reason": "insufficient_sessions", **metadata}

    folds: list[dict[str, Any]] = []
    stitched_predictions: list[dict[str, Any]] = []
    feature_cols = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label", "outcome", "entry_time", "exit_time", "entry_price", "stop_price", "target_price", "exit_price", "bars_held", "mfe", "mae", "pnl_r"}
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    for train, test, fold_meta in folds_input:
        if train.empty or test.empty or train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(train[feature_cols], train["label"])
        probabilities = model.predict_proba(test[feature_cols])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        prediction_frame = test[
            [
                "candidate_id",
                "session_date",
                "direction",
                "setup_subtype",
                "label",
                "pnl_r",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "stop_price",
                "target_price",
                "bars_held",
            ]
        ].copy()
        prediction_frame["probability"] = probabilities
        prediction_frame["prediction"] = predictions
        prediction_frame["fold"] = fold_meta.fold
        stitched_predictions.extend(prediction_frame.to_dict(orient="records"))
        folds.append(
            {
                **fold_meta.to_dict(),
                "roc_auc": float(roc_auc_score(test["label"], probabilities)),
                "precision": float(precision_score(test["label"], predictions, zero_division=0)),
            }
        )

    if len(folds) < 2:
        return {"status": "pending", "reason": "insufficient_valid_folds", "fold_count": len(folds), **metadata}

    mean_roc_auc = sum(item["roc_auc"] for item in folds) / len(folds)
    status = "pass" if mean_roc_auc >= 0.55 else "fail"
    return {
        "status": status,
        "fold_count": len(folds),
        "mean_roc_auc": mean_roc_auc,
        **metadata,
        "folds": folds,
        "stitched_prediction_records": stitched_predictions,
    }


def _purging_check(labels: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    if not labels:
        return {"status": "pending", "reason": "missing_labels"}

    labels_df = pd.DataFrame(labels)
    if labels_df.empty or "entry_time" not in labels_df or "exit_time" not in labels_df:
        return {"status": "pending", "reason": "missing_time_bounds"}

    global_config = load_global_config()
    purging_bars = int(global_config["validation"].get("purging_bars", 0))
    labels_df["entry_time"] = pd.to_datetime(labels_df["entry_time"], errors="coerce", utc=True)
    labels_df["exit_time"] = pd.to_datetime(labels_df["exit_time"], errors="coerce", utc=True)
    labels_df = labels_df.dropna(subset=["entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    if len(labels_df) < 2:
        return {"status": "pass", "overlap_ratio": 0.0, "overlapping_pairs": 0}

    overlapping_pairs = 0
    for idx in range(1, len(labels_df)):
        if labels_df.iloc[idx]["entry_time"] < labels_df.iloc[idx - 1]["exit_time"]:
            overlapping_pairs += 1
    overlap_ratio = overlapping_pairs / max(len(labels_df) - 1, 1)
    if overlapping_pairs == 0:
        return {"status": "pass", "overlap_ratio": overlap_ratio, "overlapping_pairs": overlapping_pairs}
    if purging_bars > 0:
        return {"status": "pass", "overlap_ratio": overlap_ratio, "overlapping_pairs": overlapping_pairs, "purging_bars": purging_bars}
    return {
        "status": "fail",
        "overlap_ratio": overlap_ratio,
        "overlapping_pairs": overlapping_pairs,
        "purging_bars": purging_bars,
    }


def _multiple_testing_check(search_results: dict[str, Any]) -> dict[str, Any]:
    trial_count = int(search_results.get("trial_count", 0) or 0)
    accepted = dict(search_results.get("accepted_trial", {}) or {})
    if trial_count <= 0:
        return {"status": "pending", "reason": "no_trials"}
    if not accepted:
        return {"status": "fail", "trial_count": trial_count, "reason": "no_trial_cleared_controller"}

    net_delta = float(accepted.get("net_delta_vs_baseline", 0.0) or 0.0)
    roc_delta = float(accepted.get("roc_auc_delta_vs_baseline", 0.0) or 0.0)
    penalty = 0.01 * (trial_count ** 0.5)
    adjusted_net_delta = net_delta - penalty
    status = "pass" if adjusted_net_delta > 0 and roc_delta >= 0 else "fail"
    return {
        "status": status,
        "trial_count": trial_count,
        "accepted_trial_id": accepted.get("trial_id"),
        "net_delta_vs_baseline": net_delta,
        "roc_auc_delta_vs_baseline": roc_delta,
        "complexity_penalty": penalty,
        "adjusted_net_delta": adjusted_net_delta,
    }


def _overfitting_check(walk_forward: dict[str, Any], cpcv: dict[str, Any], multiple_testing: dict[str, Any]) -> str:
    if walk_forward.get("status") == "fail" or cpcv.get("status") == "fail" or multiple_testing.get("status") == "fail":
        return "fail"
    if walk_forward.get("status") == "pass" and cpcv.get("status") == "pass" and multiple_testing.get("status") == "pass":
        return "pass"
    return "pending"
