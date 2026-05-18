from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ModelRunSummary:
    model_name: str
    model_family: str
    rows_train: int
    rows_test: int
    positive_rate_train: float
    positive_rate_test: float
    avg_entry_price: float
    avg_risk_points: float
    metrics: dict[str, float]
    prediction_records: list[dict[str, Any]]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_baseline_classifier(features: Any, labels: Any, *, model_family: str = "linear_baseline") -> ModelRunSummary:
    pd, _np = _require_pandas_numpy()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Stage 2 modeling requires scikit-learn.") from exc

    merged = features.merge(labels, on="candidate_id", how="inner")
    if len(merged) < 10 or merged["label"].nunique() < 2:
        return ModelRunSummary(
            model_name="logistic_regression",
            model_family=model_family,
            rows_train=0,
            rows_test=0,
            positive_rate_train=0.0,
            positive_rate_test=float(merged["label"].mean()) if len(merged) else 0.0,
            avg_entry_price=float(labels["entry_price"].mean()) if len(labels) else 0.0,
            avg_risk_points=float((labels["entry_price"] - labels["stop_price"]).abs().mean()) if len(labels) else 0.0,
            metrics={},
            prediction_records=[],
            status="insufficient_class_diversity",
        )

    merged = merged.sort_values("session_date").reset_index(drop=True)
    split = max(1, int(len(merged) * 0.7))
    train = merged.iloc[:split]
    test = merged.iloc[split:]
    feature_cols = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label", "entry_price", "stop_price", "target_price", "exit_price", "bars_held", "mfe", "mae", "pnl_r"} and pd.api.types.is_numeric_dtype(merged[col])
    ]
    if model_family == "gbm":
        model_name = "hist_gradient_boosting"
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200),
        )
    else:
        model_name = "logistic_regression"
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    model.fit(train[feature_cols], train["label"])
    probabilities = model.predict_proba(test[feature_cols])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "precision": float(precision_score(test["label"], predictions, zero_division=0)),
        "recall": float(recall_score(test["label"], predictions, zero_division=0)),
        "brier": float(brier_score_loss(test["label"], probabilities)),
    }
    if test["label"].nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(test["label"], probabilities))

    prediction_records = test[["candidate_id", "session_date", "label", "pnl_r"]].copy()
    prediction_records["probability"] = probabilities
    prediction_records["prediction"] = predictions

    return ModelRunSummary(
        model_name=model_name,
        model_family=model_family,
        rows_train=int(len(train)),
        rows_test=int(len(test)),
        positive_rate_train=float(train["label"].mean()),
        positive_rate_test=float(test["label"].mean()),
        avg_entry_price=float(merged["entry_price"].mean()),
        avg_risk_points=float((merged["entry_price"] - merged["stop_price"]).abs().mean()),
        metrics=metrics,
        prediction_records=prediction_records.to_dict(orient="records"),
        status="fit",
    )


def _require_pandas_numpy():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Stage 2 modeling requires pandas and numpy.") from exc
    return pd, np
