from __future__ import annotations

from typing import Any


FEATURE_CATALOG: dict[str, list[str]] = {
    "opening_auction": [
        "opening_gap_bps",
        "opening_zone_width_pct_atr",
        "opening_zone_body_to_range",
        "opening_zone_volume_percentile",
        "pre_930_drift",
        "pre_930_volatility",
    ],
    "break_quality": [
        "break_close_distance_to_zone",
        "break_body_fraction",
        "break_speed_bars",
        "break_volume_surge",
        "break_range_expansion",
        "break_efficiency_ratio",
    ],
    "pivot_reclaim": [
        "pivot_symmetry",
        "pivot_overlap_ratio",
        "reclaim_latency_bars",
        "reclaim_close_location",
        "reclaim_body_strength",
        "reclaim_failure_count",
    ],
    "continuation_quality": [
        "first_pullback_shallowness",
        "continuation_bar_sequence_strength",
        "post_entry_range_expansion",
        "distance_to_session_extrema",
        "impulse_to_retrace_ratio",
        "time_under_water_bars",
    ],
    "time_of_day": [
        "minutes_since_open",
        "seconds_since_break",
        "seconds_since_reclaim",
        "entry_clock_bucket",
        "subsession_bucket",
    ],
    "context_structure": [
        "prior_day_range_position",
        "overnight_range_position",
        "inside_outside_day_flag",
        "opening_drive_alignment",
        "relative_location_to_vwap",
        "distance_to_prior_high_low",
    ],
}


KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "reclaim": ("pivot_reclaim", "continuation_quality"),
    "pivot": ("pivot_reclaim",),
    "opening": ("opening_auction", "time_of_day"),
    "open": ("opening_auction", "time_of_day"),
    "break": ("break_quality",),
    "continuation": ("continuation_quality",),
    "trend": ("context_structure", "continuation_quality"),
    "vwap": ("context_structure",),
    "time": ("time_of_day",),
    "volume": ("opening_auction", "break_quality"),
}


def build_strategy_intake(strategy_notes: str, bnr_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    notes = (strategy_notes or "").strip()
    lower = notes.lower()
    selected_groups: list[str] = []
    for keyword, groups in KEYWORD_MAP.items():
        if keyword in lower:
            for group in groups:
                if group not in selected_groups:
                    selected_groups.append(group)
    if not selected_groups:
        selected_groups = ["opening_auction", "break_quality", "pivot_reclaim", "continuation_quality", "time_of_day"]

    feature_backlog = {group: FEATURE_CATALOG[group] for group in selected_groups}
    research_questions = _build_research_questions(lower)
    return {
        "status": "complete" if notes else "seeded_from_default_bnr",
        "strategy_notes": notes,
        "selected_feature_groups": selected_groups,
        "feature_backlog": feature_backlog,
        "research_questions": research_questions,
        "next_feature_labs": _next_feature_labs(selected_groups),
        "bnr_setup_name": (bnr_spec or {}).get("setup", {}).get("name", "BNR"),
    }


def _build_research_questions(lower: str) -> list[str]:
    questions = [
        "Which opening-auction conditions separate strong BNR days from noisy repair days?",
        "What break and reclaim geometry most strongly predicts continuation quality?",
        "Which time-of-day delays weaken the setup even when the pattern still forms?",
    ]
    if "vwap" in lower:
        questions.append("How does VWAP location change reclaim quality and continuation odds?")
    if "volume" in lower:
        questions.append("Does volume expansion at the break or reclaim improve utility after costs?")
    return questions


def _next_feature_labs(selected_groups: list[str]) -> list[str]:
    mapping = {
        "opening_auction": "auction_context_lab",
        "break_quality": "break_quality_lab",
        "pivot_reclaim": "reclaim_microstructure_lab",
        "continuation_quality": "continuation_followthrough_lab",
        "time_of_day": "timing_decay_lab",
        "context_structure": "session_structure_lab",
    }
    return [mapping[group] for group in selected_groups if group in mapping]
