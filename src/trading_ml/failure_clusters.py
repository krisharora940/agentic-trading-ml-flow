from __future__ import annotations

from dataclasses import asdict
from typing import Any

from trading_ml.schemas import FailureCluster


def build_failure_clusters(attempts: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        return []

    frame = pd.DataFrame(attempts)
    if frame.empty or "failure_reason" not in frame.columns:
        return []
    if "features" in frame.columns:
        feature_frame = pd.json_normalize(frame["features"]).add_prefix("feature_")
        frame = pd.concat([frame.drop(columns=["features"]).reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)

    failed = frame[(frame["label"] == 0) | (frame["pnl_r"].fillna(0.0) <= 0)].copy()
    if failed.empty:
        return []

    clusters: list[dict[str, Any]] = []
    grouped = failed.groupby(["failure_reason", "setup_state", "environment_state", "time_bucket"], dropna=False)
    for (failure_reason, setup_state, environment_state, time_bucket), group in grouped:
        if len(group) < 2:
            continue
        dominant_subtype = _mode(group.get("setup_subtype"))
        recommendation = _recommendation(str(failure_reason))
        cluster = FailureCluster(
            cluster_id=f"FC-{failure_reason}-{setup_state}-{environment_state}-{time_bucket}",
            family=str(failure_reason),
            rows=int(len(group)),
            avg_pnl_r=float(group["pnl_r"].mean()) if "pnl_r" in group else 0.0,
            avg_probability=float(group["probability"].mean()) if "probability" in group else 0.0,
            dominant_subtype=dominant_subtype,
            dominant_setup_state=str(setup_state),
            dominant_environment_state=str(environment_state),
            dominant_time_bucket=str(time_bucket),
            recommended_family=recommendation["family"],
            recommended_focus=recommendation["focus"],
            evidence={
                "avg_break_efficiency_ratio": _mean(group, "feature_break_efficiency_ratio"),
                "avg_reclaim_close_location": _mean(group, "feature_reclaim_close_location"),
                "avg_deepest_zone_retrace_fraction": _mean(group, "feature_deepest_zone_retrace_fraction"),
                "avg_post_reclaim_close_strength": _mean(group, "feature_post_reclaim_close_strength"),
                "probability_bucket_mode": _mode(group.get("probability_bucket")),
                "path_class_mode": _mode(group.get("path_class")),
                "setup_state_mode": _mode(group.get("setup_state")),
                "environment_state_mode": _mode(group.get("environment_state")),
            },
        )
        clusters.append(asdict(cluster))

    clusters.sort(key=lambda row: (-int(row["rows"]), float(row["avg_pnl_r"])))
    return clusters[:limit]


def _recommendation(failure_reason: str) -> dict[str, Any]:
    mapping = {
        "no_follow_through": {
            "family": "exit_behavior_research",
            "focus": ["post_reclaim_displacement", "runner_vs_chop_path", "scratch_timing"],
        },
        "weak_continuation": {
            "family": "feature",
            "focus": ["confirmation_strength", "followthrough_efficiency", "auction_clarity"],
        },
        "deep_retrace_failure": {
            "family": "setup",
            "focus": ["retrace_depth_tolerance", "reclaim_quality", "environment_eligibility"],
        },
        "no_reclaim_edge": {
            "family": "candidate_universe_expansion",
            "focus": ["attempt_definition", "reclaim_timing", "candidate_deduplication"],
        },
    }
    return mapping.get(
        failure_reason,
        {
            "family": "feature",
            "focus": ["attempt_state_classification", "confirmation_quality"],
        },
    )


def _mode(series: Any) -> str:
    if series is None:
        return "unknown"
    mode = series.mode(dropna=True)
    if mode.empty:
        return "unknown"
    return str(mode.iloc[0])


def _mean(frame: Any, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame[column].mean())
