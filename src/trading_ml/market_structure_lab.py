from __future__ import annotations

from typing import Any


def build_market_structure_lab(
    candidates: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    candidate_frame = pd.DataFrame(candidates)
    label_frame = pd.DataFrame(labels)
    if candidate_frame.empty or label_frame.empty:
        return {"status": "pending", "reason": "empty_inputs"}

    candidate_frame["trace"] = candidate_frame["trace"].apply(lambda value: value or {})
    trace_frame = pd.json_normalize(candidate_frame["trace"])
    merged = pd.concat(
        [
            candidate_frame.drop(columns=["trace"]).reset_index(drop=True),
            trace_frame.reset_index(drop=True),
        ],
        axis=1,
    ).merge(
        label_frame[
            ["candidate_id", "label", "outcome", "pnl_r", "bars_held", "mfe", "mae"]
        ],
        on="candidate_id",
        how="inner",
    )
    if merged.empty:
        return {"status": "pending", "reason": "empty_merged_dataset"}

    merged["structure_family"] = merged.apply(_structure_family, axis=1)
    merged["failure_reason"] = merged.apply(_failure_reason, axis=1)

    family_rows = []
    for family, group in merged.groupby("structure_family", dropna=False):
        family_rows.append(
            {
                "family": str(family),
                "count": int(len(group)),
                "positive_rate": float(group["label"].mean()),
                "avg_pnl_r": float(group["pnl_r"].mean()),
                "reclaim_mean": (
                    float(group["reclaim_count"].mean())
                    if "reclaim_count" in group
                    else 0.0
                ),
                "retrace_mean": (
                    float(group["deepest_zone_retrace_fraction"].mean())
                    if "deepest_zone_retrace_fraction" in group
                    else 0.0
                ),
                "continuation_strength_mean": (
                    float(group["post_reclaim_close_strength"].mean())
                    if "post_reclaim_close_strength" in group
                    else 0.0
                ),
            }
        )

    failure_rows = []
    failed = merged[merged["label"] == 0]
    for reason, group in failed.groupby("failure_reason", dropna=False):
        failure_rows.append(
            {
                "failure_reason": str(reason),
                "count": int(len(group)),
                "avg_pnl_r": float(group["pnl_r"].mean()),
                "common_family": (
                    str(group["structure_family"].mode().iloc[0])
                    if not group["structure_family"].mode().empty
                    else "unknown"
                ),
            }
        )

    top_failures = sorted(failure_rows, key=lambda row: row["count"], reverse=True)[:8]
    return {
        "status": "complete",
        "candidate_count": int(len(merged)),
        "structure_families": sorted(
            family_rows, key=lambda row: row["avg_pnl_r"], reverse=True
        ),
        "failure_taxonomy": top_failures,
        "market_structure_questions": _market_structure_questions(
            top_failures, family_rows
        ),
    }


def _structure_family(row: Any) -> str:
    close_confirmed = float(row.get("first_break_close_confirmed", 0.0) or 0.0) > 0
    retrace = float(row.get("deepest_zone_retrace_fraction", 0.0) or 0.0)
    reclaims = int(row.get("reclaim_count", 0) or 0)
    continuation = float(row.get("post_reclaim_close_strength", 0.0) or 0.0)

    break_quality = "close_break" if close_confirmed else "wick_break"
    retrace_quality = "shallow" if retrace <= 0.35 else "deep"
    reclaim_quality = "multi_reclaim" if reclaims >= 2 else "single_reclaim"
    continuation_quality = "strong_cont" if continuation >= 0.5 else "weak_cont"
    return f"{break_quality}:{retrace_quality}:{reclaim_quality}:{continuation_quality}"


def _failure_reason(row: Any) -> str:
    outcome = str(row.get("outcome", "unknown"))
    retrace = float(row.get("deepest_zone_retrace_fraction", 0.0) or 0.0)
    continuation = float(row.get("post_reclaim_close_strength", 0.0) or 0.0)
    reclaims = int(row.get("reclaim_count", 0) or 0)

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


def _market_structure_questions(
    failure_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]
) -> list[str]:
    questions: list[str] = []
    if failure_rows:
        top = failure_rows[0]["failure_reason"]
        questions.append(
            f"How can the setup filter reduce {top} without collapsing trade count?"
        )
    if family_rows:
        best = family_rows[0]["family"]
        questions.append(
            f"Which pre-open and reclaim conditions best explain strength in {best}?"
        )
    return questions
