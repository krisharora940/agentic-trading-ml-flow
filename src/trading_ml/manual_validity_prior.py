from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_ml.stage2_bnr import CandidateSetup


MANUAL_SEQUENCE_INDEX_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "manual_labels"
    / "trades_20260526092550_sequence_index.csv"
)
MATCH_TOLERANCE_SECONDS = 90


@dataclass(slots=True)
class ManualValiditySummary:
    status: str
    manual_example_count: int = 0
    matched_example_count: int = 0
    matched_candidate_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    model_status: str = "not_run"
    artifact_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "manual_example_count": self.manual_example_count,
            "matched_example_count": self.matched_example_count,
            "matched_candidate_count": self.matched_candidate_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "model_status": self.model_status,
            "artifact_path": self.artifact_path,
        }


def augment_features_with_manual_validity_prior(
    features: Any, candidates: list[CandidateSetup]
) -> tuple[Any, ManualValiditySummary]:
    pd = _require_pandas()
    if features.empty:
        return features, ManualValiditySummary(status="missing_features")
    manual = _load_manual_examples()
    if manual is None or manual.empty:
        return (
            _with_default_prior_columns(features),
            ManualValiditySummary(status="missing_manual_examples"),
        )

    matched = _match_manual_examples_to_candidates(manual, candidates)
    augmented = features.copy()
    augmented["manual_validity_probability"] = 0.5
    augmented["manual_high_quality_probability"] = 0.0
    augmented["manual_example_matched"] = 0.0

    if matched.empty:
        return (
            augmented,
            ManualValiditySummary(
                status="no_candidate_matches",
                manual_example_count=int(len(manual)),
                artifact_path=str(MANUAL_SEQUENCE_INDEX_PATH),
            ),
        )

    training = augmented.merge(matched, on="candidate_id", how="inner")
    positive_count = int(training["manual_validity_label"].sum())
    negative_count = int(len(training) - positive_count)
    summary = ManualValiditySummary(
        status="available",
        manual_example_count=int(len(manual)),
        matched_example_count=int(len(matched)),
        matched_candidate_count=int(training["candidate_id"].nunique()),
        positive_count=positive_count,
        negative_count=negative_count,
        artifact_path=str(MANUAL_SEQUENCE_INDEX_PATH),
    )
    if len(training) < 20 or training["manual_validity_label"].nunique() < 2:
        summary.model_status = "insufficient_class_diversity"
        augmented.loc[
            augmented["candidate_id"].isin(training["candidate_id"]),
            "manual_example_matched",
        ] = 1.0
        return augmented, summary

    feature_cols = _numeric_feature_columns(training)
    valid_probs = _fit_predict_probabilities(
        training[feature_cols],
        training["manual_validity_label"],
    )
    high_quality_probs = _fit_predict_probabilities(
        training[feature_cols],
        training["manual_high_quality_label"],
    )
    final_valid = _fit_final_model(
        training[feature_cols], training["manual_validity_label"]
    )
    final_high_quality = _fit_final_model(
        training[feature_cols], training["manual_high_quality_label"]
    )
    if final_valid is None or final_high_quality is None:
        summary.model_status = "fit_failed"
        return augmented, summary

    all_feature_cols = [
        col
        for col in feature_cols
        if col in augmented.columns and pd.api.types.is_numeric_dtype(augmented[col])
    ]
    augmented["manual_validity_probability"] = final_valid.predict_proba(
        augmented[all_feature_cols]
    )[:, 1]
    augmented["manual_high_quality_probability"] = final_high_quality.predict_proba(
        augmented[all_feature_cols]
    )[:, 1]
    augmented.loc[
        augmented["candidate_id"].isin(training["candidate_id"]),
        "manual_example_matched",
    ] = 1.0
    augmented.loc[
        augmented["candidate_id"].isin(training["candidate_id"]),
        "manual_validity_probability",
    ] = valid_probs
    augmented.loc[
        augmented["candidate_id"].isin(training["candidate_id"]),
        "manual_high_quality_probability",
    ] = high_quality_probs
    summary.model_status = "fit"
    return augmented, summary


def _with_default_prior_columns(features: Any) -> Any:
    augmented = features.copy()
    augmented["manual_validity_probability"] = 0.5
    augmented["manual_high_quality_probability"] = 0.0
    augmented["manual_example_matched"] = 0.0
    return augmented


def _load_manual_examples() -> Any | None:
    pd = _require_pandas()
    if not MANUAL_SEQUENCE_INDEX_PATH.exists():
        return None
    frame = pd.read_csv(MANUAL_SEQUENCE_INDEX_PATH)
    if frame.empty:
        return frame
    frame["entry_anchor_timestamp"] = pd.to_datetime(
        frame["entry_anchor_timestamp"], errors="coerce", utc=True
    ).dt.tz_convert("America/New_York")
    frame = frame.dropna(subset=["entry_anchor_timestamp"]).copy()
    frame["direction"] = frame["side"].astype(str).str.lower()
    frame["manual_validity_label"] = (
        frame["setup_valid_normalized"].astype(str).str.lower().eq("yes").astype(int)
    )
    frame["manual_high_quality_label"] = (
        frame["quality_bucket"].astype(str).eq("high_quality").astype(int)
    )
    frame["session_date"] = frame["open_date"].astype(str)
    return frame


def _match_manual_examples_to_candidates(
    manual: Any, candidates: list[CandidateSetup]
) -> Any:
    pd = _require_pandas()
    if not candidates:
        return pd.DataFrame()
    candidate_rows = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "session_date": candidate.session_date,
                "direction": candidate.direction,
                "decision_time": pd.Timestamp(candidate.decision_time),
            }
            for candidate in candidates
        ]
    )
    candidate_rows["decision_time"] = pd.to_datetime(
        candidate_rows["decision_time"], errors="coerce", utc=True
    ).dt.tz_convert("America/New_York")
    matched_rows: list[dict[str, Any]] = []
    used_candidate_ids: set[str] = set()
    for _, row in manual.sort_values("entry_anchor_timestamp").iterrows():
        same_slice = candidate_rows[
            (candidate_rows["session_date"] == row["session_date"])
            & (candidate_rows["direction"] == row["direction"])
            & (~candidate_rows["candidate_id"].isin(used_candidate_ids))
        ].copy()
        if same_slice.empty:
            continue
        same_slice["delta_seconds"] = (
            (same_slice["decision_time"] - row["entry_anchor_timestamp"])
            .abs()
            .dt.total_seconds()
        )
        best = same_slice.sort_values("delta_seconds").iloc[0]
        if float(best["delta_seconds"]) > MATCH_TOLERANCE_SECONDS:
            continue
        used_candidate_ids.add(str(best["candidate_id"]))
        matched_rows.append(
            {
                "candidate_id": str(best["candidate_id"]),
                "manual_validity_label": int(row["manual_validity_label"]),
                "manual_high_quality_label": int(row["manual_high_quality_label"]),
            }
        )
    return pd.DataFrame(matched_rows)


def _numeric_feature_columns(frame: Any) -> list[str]:
    pd = _require_pandas()
    excluded = {
        "candidate_id",
        "session_date",
        "setup_subtype",
        "manual_validity_label",
        "manual_high_quality_label",
    }
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _fit_predict_probabilities(features: Any, target: Any) -> list[float]:
    if len(features) == 0 or target.nunique() < 2:
        return [0.5 for _ in range(len(features))]
    model = _fit_final_model(features, target)
    if model is None:
        return [0.5 for _ in range(len(features))]
    return list(model.predict_proba(features)[:, 1])


def _fit_final_model(features: Any, target: Any) -> Any | None:
    if len(features) == 0 or target.nunique() < 2:
        return None
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(features, target)
    return model


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("manual validity prior requires pandas") from exc
    return pd
