from __future__ import annotations

from typing import Any


def classify_candidate_subtype(candidate: Any) -> str:
    trace = dict(getattr(candidate, "trace", {}) or {})
    close_confirmed = float(trace.get("first_break_close_confirmed", 0.0) or 0.0) > 0
    wick_only = float(trace.get("first_break_wick_only", 0.0) or 0.0) > 0
    retrace = float(trace.get("deepest_zone_retrace_fraction", 0.0) or 0.0)
    continuation = float(trace.get("post_reclaim_close_strength", 0.0) or 0.0)
    displacement = float(trace.get("continuation_displacement_ratio", 0.0) or 0.0)
    reclaims = int(getattr(candidate, "reclaim_count", 0) or 0)

    if close_confirmed and retrace <= 0.35 and continuation >= 0.5:
        return "clean_break_continuation"
    if retrace >= 0.65:
        return "deep_retrace_repair"
    if wick_only and reclaims >= 1:
        return "wick_reclaim_repair"
    if continuation < 0.2 or displacement < 0.35:
        return "weak_follow_through"
    return "balanced_reclaim_continuation"


def filter_candidates_by_subtype(candidates: list[Any], subtype: str) -> list[Any]:
    if subtype in {"", "all", "all_subtypes"}:
        return candidates
    return [candidate for candidate in candidates if classify_candidate_subtype(candidate) == subtype]


def list_bnr_subtypes() -> list[str]:
    return [
        "all_subtypes",
        "clean_break_continuation",
        "balanced_reclaim_continuation",
        "deep_retrace_repair",
        "wick_reclaim_repair",
        "weak_follow_through",
    ]
