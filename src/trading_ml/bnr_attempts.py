from __future__ import annotations

from dataclasses import asdict
from typing import Any

from trading_ml.schemas import BNRAttempt


def build_bnr_attempts(
    stage2_result: dict[str, Any],
    stitched_prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        return []

    features = pd.DataFrame(stage2_result.get("features_records", []))
    labels = pd.DataFrame(stage2_result.get("labels_records", []))
    stitched = pd.DataFrame(stitched_prediction_records or [])
    if features.empty or labels.empty or stitched.empty:
        return []

    feature_columns = [
        "candidate_id",
        "session_date",
        "direction",
        "setup_subtype",
        "trigger_seconds_after_open",
        "break_efficiency_ratio",
        "reclaim_close_location",
        "reclaim_failure_count",
        "deepest_zone_retrace_fraction",
        "post_reclaim_close_strength",
        "reg_high_vol_state",
        "reg_trending_state",
    ]
    available = [column for column in feature_columns if column in features.columns]
    merged = (
        stitched.merge(
            labels[["candidate_id", "label", "outcome", "pnl_r", "bars_held"]],
            on="candidate_id",
            how="left",
            suffixes=("", "_label"),
        )
        .merge(features[available], on="candidate_id", how="left", suffixes=("", "_feature"))
        .copy()
    )
    if merged.empty:
        return []

    attempts: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        probability = _safe_float(row.get("probability"))
        pnl_r = _safe_float(row.get("pnl_r"))
        label = _safe_int(row.get("label"))
        prediction = _safe_int(row.get("prediction"))
        trigger_seconds = _safe_float(row.get("trigger_seconds_after_open"))
        attempt = BNRAttempt(
            attempt_id=f"ATT-{row.get('candidate_id')}",
            candidate_id=str(row.get("candidate_id")),
            session_date=str(row.get("session_date") or row.get("session_date_feature") or ""),
            direction=str(row.get("direction", "unknown")),
            setup_subtype=str(row.get("setup_subtype", "unknown")),
            time_bucket=_time_bucket(trigger_seconds),
            probability_bucket=_probability_bucket(probability),
            executed=bool(prediction == 1),
            label=label,
            prediction=prediction,
            probability=probability,
            pnl_r=pnl_r,
            outcome=str(row.get("outcome", "unknown")),
            failure_reason=_failure_reason(row),
            path_class=_path_class(row),
            features={
                "trigger_seconds_after_open": trigger_seconds,
                "break_efficiency_ratio": _safe_float(row.get("break_efficiency_ratio")),
                "reclaim_close_location": _safe_float(row.get("reclaim_close_location")),
                "reclaim_failure_count": _safe_int(row.get("reclaim_failure_count")),
                "deepest_zone_retrace_fraction": _safe_float(row.get("deepest_zone_retrace_fraction")),
                "post_reclaim_close_strength": _safe_float(row.get("post_reclaim_close_strength")),
                "reg_high_vol_state": _safe_int(row.get("reg_high_vol_state")),
                "reg_trending_state": _safe_int(row.get("reg_trending_state")),
            },
        )
        attempts.append(asdict(attempt))
    return attempts


def _time_bucket(trigger_seconds: float | None) -> str:
    if trigger_seconds is None:
        return "unknown"
    if trigger_seconds <= 300:
        return "early_open"
    if trigger_seconds <= 900:
        return "mid_open"
    return "late_open"


def _probability_bucket(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    if probability < 0.4:
        return "low_confidence"
    if probability < 0.6:
        return "mid_confidence"
    return "high_confidence"


def _path_class(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome", "unknown"))
    pnl_r = _safe_float(row.get("pnl_r"))
    bars_held = _safe_int(row.get("bars_held"))
    if outcome == "target":
        return "runner" if bars_held is None or bars_held <= 6 else "delayed_runner"
    if outcome == "timeout":
        return "chop"
    if pnl_r is not None and pnl_r <= 0:
        return "failure"
    return "unclear"


def _failure_reason(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome", "unknown"))
    retrace = _safe_float(row.get("deepest_zone_retrace_fraction")) or 0.0
    continuation = _safe_float(row.get("post_reclaim_close_strength")) or 0.0
    reclaims = _safe_int(row.get("reclaim_failure_count")) or 0
    if outcome == "timeout":
        return "no_follow_through"
    if outcome == "ambiguous_stop_first":
        return "same_bar_conflict"
    if retrace > 0.75:
        return "deep_retrace_failure"
    if continuation < 0.2:
        return "weak_continuation"
    if reclaims == 0:
        return "no_reclaim_edge"
    return "stop_before_target"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
