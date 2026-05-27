from __future__ import annotations

from typing import Any


FAMILY_COLUMNS: dict[str, list[str]] = {
    "bnr_geometry": [
        "direction_long",
        "zone_width",
        "zone_width_bps",
        "zone_midpoint",
        "entry_distance_from_midpoint",
        "entry_distance_from_zone_high",
        "entry_distance_from_zone_low",
        "first_break_wick_only",
        "first_break_close_confirmed",
        "first_break_wick_excess_points",
        "first_break_close_excess_points",
        "continuation_displacement_ratio",
        "break_close_distance_to_zone",
        "break_body_fraction",
        "break_speed_bars",
        "break_volume_surge",
        "break_range_expansion",
        "break_efficiency_ratio",
    ],
    "pivot_reclaim": [
        "pivot_duration_bars",
        "continuation_delay_bars",
        "reentry_count",
        "reclaim_count",
        "deepest_zone_retrace_fraction",
        "post_reclaim_close_strength",
        "opposite_boundary_close_violation",
        "pivot_symmetry",
        "pivot_overlap_ratio",
        "reclaim_latency_bars",
        "reclaim_close_location",
        "reclaim_body_strength",
        "reclaim_failure_count",
    ],
    "pre_trigger_context": [
        "trigger_seconds_after_open",
        "pre_trigger_return",
        "pre_trigger_range",
        "pre_trigger_volume",
        "prior_close_gap",
    ],
    "sequence_structure": [
        "retrace_at_entry",
        "pivot_flem_dist",
        "time_since_pivot_sec",
        "body_last",
        "body_sum",
        "body_mean",
        "in_dir_ratio",
        "max_in_dir_run",
        "bars_since_pivot",
        "zone_over_range",
        "pivot_over_range",
        "dist_to_extrema_atr",
        "zone_to_extrema_atr",
    ],
    "regime_features": [
        "reg_vol_10",
        "reg_vol_30",
        "reg_vol_ratio",
        "reg_trend_10",
        "reg_trend_30",
        "reg_trend_strength_10",
        "reg_trend_strength_30",
        "reg_chop_10",
        "reg_chop_30",
        "reg_range_ratio_10",
        "reg_range_ratio_30",
        "reg_pretrigger_bar_count",
        "reg_high_vol_state",
        "reg_trending_state",
    ],
    "manual_quality_prior": [
        "manual_validity_probability",
        "manual_high_quality_probability",
        "manual_example_matched",
    ],
}


def list_feature_families() -> list[str]:
    return [
        "all_features",
        "bnr_geometry",
        "pivot_reclaim",
        "pre_trigger_context",
        "regime_features",
        "engineer_indicators",
        "bnr_core",
        "bnr_plus_context",
        "bnr_plus_engineer",
        "context_plus_reclaim",
        "context_plus_geometry",
        "reclaim_plus_engineer",
        "context_plus_regime",
        "reclaim_plus_regime",
    ]


def apply_feature_family(features: Any, family: str) -> Any:
    if family in {"", "all", "all_features"}:
        return features

    keep = {"candidate_id", "session_date", "setup_subtype"}
    if family == "engineer_indicators":
        keep.update(
            column for column in features.columns if str(column).startswith("eng_")
        )
    elif family == "bnr_core":
        keep.update(FAMILY_COLUMNS["bnr_geometry"])
        keep.update(FAMILY_COLUMNS["pivot_reclaim"])
    elif family == "bnr_plus_context":
        keep.update(FAMILY_COLUMNS["bnr_geometry"])
        keep.update(FAMILY_COLUMNS["pivot_reclaim"])
        keep.update(FAMILY_COLUMNS["pre_trigger_context"])
        keep.update(FAMILY_COLUMNS["sequence_structure"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
    elif family == "bnr_plus_engineer":
        keep.update(FAMILY_COLUMNS["bnr_geometry"])
        keep.update(FAMILY_COLUMNS["pivot_reclaim"])
        keep.update(FAMILY_COLUMNS["sequence_structure"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
        keep.update(
            column for column in features.columns if str(column).startswith("eng_")
        )
    elif family == "context_plus_reclaim":
        keep.update(FAMILY_COLUMNS["pre_trigger_context"])
        keep.update(FAMILY_COLUMNS["pivot_reclaim"])
        keep.update(FAMILY_COLUMNS["sequence_structure"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
    elif family == "context_plus_geometry":
        keep.update(FAMILY_COLUMNS["pre_trigger_context"])
        keep.update(FAMILY_COLUMNS["bnr_geometry"])
        keep.update(FAMILY_COLUMNS["sequence_structure"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
    elif family == "reclaim_plus_engineer":
        keep.update(FAMILY_COLUMNS["pivot_reclaim"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
        keep.update(
            column for column in features.columns if str(column).startswith("eng_")
        )
    elif family == "context_plus_regime":
        keep.update(FAMILY_COLUMNS["pre_trigger_context"])
        keep.update(FAMILY_COLUMNS["sequence_structure"])
        keep.update(FAMILY_COLUMNS["regime_features"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
    elif family == "reclaim_plus_regime":
        keep.update(FAMILY_COLUMNS["pivot_reclaim"])
        keep.update(FAMILY_COLUMNS["sequence_structure"])
        keep.update(FAMILY_COLUMNS["regime_features"])
        keep.update(FAMILY_COLUMNS["manual_quality_prior"])
    else:
        keep.update(FAMILY_COLUMNS.get(family, []))

    selected = [column for column in features.columns if column in keep]
    if not selected:
        return features
    return features[selected].copy()
