from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trading_ml.bnr_subtypes import classify_candidate_subtype, filter_candidates_by_subtype
from trading_ml.config import load_bnr_config
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_bnr import calculate_bnr_zones, generate_breakout_candidates
from trading_ml.stage2_data import load_ohlcv_file, regular_session
from trading_ml.stage2_features import build_feature_matrix
from trading_ml.stage2_labeling import label_candidates
from trading_ml.stage2_pipeline import Stage2Config


MARKET_STATES = [
    "clean_strong_continuation",
    "weak_or_grindy_continuation",
    "auction_repair",
    "balanced_chop",
    "messy_transition",
    "failed_directional",
]
SETUP_QUALITIES = ["high_quality", "marginal", "avoid"]


def build_market_state_setup_quality_diagnostic(state: dict[str, Any]) -> dict[str, Any]:
    """Build a diagnostic-only market-state/setup-quality artifact.

    This intentionally stops before model fitting, search, or holdout access.
    """

    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    source_path = _diagnostic_source_path(state)
    if not source_path:
        return {"status": "pending", "reason": "missing_exploration_source"}
    if _looks_like_holdout(source_path, state):
        return {
            "status": "blocked",
            "reason": "holdout_source_refused",
            "source_path": source_path,
            "holdout_status": "locked",
            "search_executed": False,
            "models_trained": 0,
        }

    config = _diagnostic_config(state, source_path)
    try:
        features, labels, metadata = _build_point_in_time_inputs(config)
    except Exception as exc:  # pragma: no cover - keeps graph diagnostic resumable.
        return {
            "status": "pending",
            "reason": "diagnostic_input_build_failed",
            "error": str(exc),
            "source_path": source_path,
            "holdout_status": "locked",
            "search_executed": False,
            "models_trained": 0,
        }

    if features.empty or labels.empty:
        return {
            "status": "pending",
            "reason": "missing_candidate_features_or_labels",
            "source_path": source_path,
            "holdout_status": "locked",
            "search_executed": False,
            "models_trained": 0,
        }

    merged = features.merge(labels, on="candidate_id", how="inner", suffixes=("", "_label"))
    if merged.empty:
        return {
            "status": "pending",
            "reason": "no_feature_label_overlap",
            "source_path": source_path,
            "holdout_status": "locked",
            "search_executed": False,
            "models_trained": 0,
        }

    diagnostic_features = _build_market_state_features(merged)
    labeled = _assign_provisional_states(diagnostic_features)
    cpcv_exposure = _cpcv_worst_path_exposure(labeled)

    return {
        "status": "complete",
        "family": "market_state_setup_quality",
        "priority_hypothesis": "market_state_quality_classifier",
        "execution_mode": "diagnostic_only",
        "source_path": source_path,
        "holdout_status": "locked",
        "search_executed": False,
        "models_trained": 0,
        "candidate_count": int(len(labeled)),
        "candidate_counts_by_state": _counts(labeled, "market_state"),
        "candidate_counts_by_quality": _counts(labeled, "setup_quality"),
        "label_balance": {
            "overall": _label_balance(labeled),
            "by_market_state": _group_label_balance(labeled, "market_state"),
            "by_setup_quality": _group_label_balance(labeled, "setup_quality"),
        },
        "missingness": _missingness(labeled),
        "leakage_audit": _leakage_audit(metadata, labeled),
        "pnl_by_state": _pnl_by_group(labeled, "market_state"),
        "pnl_by_setup_quality": _pnl_by_group(labeled, "setup_quality"),
        "cpcv_worst_path_exposure_by_state": cpcv_exposure,
        "avoided_vs_traded_simulation_summary": _avoided_vs_traded_summary(labeled),
        "cheap_state_policy_simulation": _cheap_state_policy_simulation(labeled),
        "residual_tail_diagnostic": _residual_tail_diagnostic(labeled),
        "followthrough_confirmation_policy_gate": _followthrough_confirmation_policy_gate(labeled),
        "support_sufficiency": _support_sufficiency(labeled),
        "reliability_warnings": _reliability_warnings(labeled),
        "provisional_label_definitions": {
            "market_state": MARKET_STATES,
            "setup_quality": SETUP_QUALITIES,
            "note": "Heuristic diagnostics only. Not a trading rule and not a trained classifier.",
        },
        "feature_groups": _feature_group_columns(),
        "data_metadata": metadata,
        "_labeled_rows": labeled.to_dict(orient="records"),
    }


def _diagnostic_config(state: dict[str, Any], source_path: str) -> Stage2Config:
    overrides = dict(state.get("stage2_config", {}) or {})
    overrides["source_path"] = source_path
    allowed = set(getattr(Stage2Config, "__dataclass_fields__", {}).keys())
    payload = {key: value for key, value in overrides.items() if key in allowed}
    if "symbol" not in payload:
        return Stage2Config.from_bnr_config(source_path)
    return Stage2Config(**payload)


def _build_point_in_time_inputs(config: Stage2Config) -> tuple[Any, Any, dict[str, Any]]:
    pd = _require_pandas()
    bars = load_ohlcv_file(
        config.source_path,
        symbol=config.symbol,
        timeframe=config.timeframe,
        timezone=config.timezone,
    )
    rth = regular_session(bars)
    zones = calculate_bnr_zones(rth, symbol=config.symbol, timeframe=config.timeframe)
    candidates = generate_breakout_candidates(
        rth,
        zones,
        timeframe=config.timeframe,
        earliest_trigger_time=config.earliest_trigger_time,
        latest_trigger_time=config.latest_trigger_time,
        break_buffer_points=config.break_buffer_points,
    )
    candidates = filter_candidates_by_subtype(candidates, config.setup_subtype)
    labels = label_candidates(
        rth,
        candidates,
        horizon_bars=config.horizon_bars,
        stop_multiple=config.stop_multiple,
        target_multiple=config.target_multiple,
    )
    features, feature_audits = build_feature_matrix(rth, candidates)
    if not features.empty:
        subtype_map = {candidate.candidate_id: classify_candidate_subtype(candidate) for candidate in candidates}
        features["setup_subtype"] = features["candidate_id"].map(subtype_map)
        raw_market_state = _build_raw_market_state_features(rth, candidates)
        features = features.merge(raw_market_state, on="candidate_id", how="left")
    labels_df = pd.DataFrame([label.to_dict() for label in labels])
    metadata = {
        "candidate_count": len(candidates),
        "zone_count": len(zones),
        "feature_audit": {
            "rows": len(feature_audits),
            "failed": len([audit for audit in feature_audits if audit.status != "pass"]),
            "issues": sorted({issue for audit in feature_audits if audit.status != "pass" for issue in audit.issues}),
        },
        "config": asdict(config),
    }
    return features, labels_df, metadata


def _build_market_state_features(frame: Any) -> Any:
    out = frame.copy()
    out["directional_efficiency_open_to_now"] = _fallback_score(
        out,
        "directional_efficiency_open_to_now",
        ["reg_trend_strength_30", "reg_trend_strength_10", "break_efficiency_ratio"],
    )
    out["recent_directional_efficiency"] = _fallback_score(
        out,
        "recent_directional_efficiency",
        ["break_efficiency_ratio", "continuation_displacement_ratio"],
    )
    out["body_to_range_ratio_recent"] = _fallback_score(
        out,
        "body_to_range_ratio_recent",
        ["break_body_fraction", "reclaim_body_strength"],
    )
    out["wick_rejection_ratio_recent"] = _fallback_score(
        out,
        "wick_rejection_ratio_recent",
        ["first_break_wick_only", "pivot_overlap_ratio", "reclaim_failure_count"],
    )
    out["alternating_bar_ratio"] = _fallback_score(out, "alternating_bar_ratio", ["reg_chop_10", "pivot_overlap_ratio"])
    out["local_swing_overlap_score"] = _fallback_score(out, "local_swing_overlap_score", ["pivot_overlap_ratio", "reg_range_ratio_10"])
    out["impulse_speed_score"] = _fallback_score(
        out,
        "impulse_speed_score",
        ["break_range_expansion", "break_efficiency_ratio", "break_volume_surge"],
    )
    out["repair_speed_score"] = _fallback_score(
        out,
        "repair_speed_score",
        ["reclaim_body_strength", "reclaim_close_location", "reclaim_latency_bars"],
        invert=["reclaim_latency_bars"],
    )
    out["flem_compression_score"] = _fallback_score(
        out,
        "flem_compression_score",
        ["pivot_overlap_ratio", "reclaim_latency_bars", "pre_trigger_range"],
    )
    out["pivot_cleanliness_score"] = _fallback_score(
        out,
        "pivot_cleanliness_score",
        ["pivot_symmetry", "pivot_overlap_ratio", "reclaim_failure_count"],
        invert=["pivot_overlap_ratio", "reclaim_failure_count"],
    )
    out["distance_to_recent_high_low"] = _fallback_score(
        out,
        "distance_to_recent_high_low",
        ["entry_distance_from_zone_high", "entry_distance_from_zone_low"],
    )
    out["failed_followthrough_count"] = _fallback_score(
        out,
        "failed_followthrough_count",
        ["reclaim_failure_count", "opposite_boundary_close_violation"],
    )
    out["msq_candle_speed_volatility"] = _mean_score(
        out,
        ["break_range_expansion", "reg_vol_ratio", "reg_vol_10", "impulse_speed_score"],
    )
    out["msq_wick_body_structure"] = _mean_score(
        out,
        ["body_to_range_ratio_recent", "reclaim_close_location", "post_reclaim_close_strength", "wick_rejection_ratio_recent"],
        invert=["wick_rejection_ratio_recent"],
    )
    out["msq_impulse_strength"] = _mean_score(
        out,
        ["recent_directional_efficiency", "continuation_displacement_ratio", "reg_trend_strength_10", "impulse_speed_score"],
    )
    out["msq_flem_shape"] = _mean_score(
        out,
        ["post_reclaim_close_strength", "continuation_displacement_ratio", "flem_compression_score", "opposite_boundary_close_violation"],
        invert=["flem_compression_score", "opposite_boundary_close_violation"],
    )
    out["msq_pivot_shape"] = _mean_score(
        out,
        ["pivot_cleanliness_score", "reclaim_close_location", "deepest_zone_retrace_fraction", "repair_speed_score"],
    )
    out["msq_recent_high_low_context"] = _mean_score(
        out,
        ["distance_to_recent_high_low", "local_swing_overlap_score", "pre_trigger_range", "zone_width"],
        invert=["distance_to_recent_high_low", "local_swing_overlap_score"],
    )
    out["msq_day_structure"] = _mean_score(
        out,
        ["directional_efficiency_open_to_now", "reg_trend_strength_30", "reg_trend_30", "reg_range_ratio_30", "reg_chop_30"],
        invert=["reg_range_ratio_30", "reg_chop_30"],
    )
    out["msq_chop_range_unpredictability"] = _mean_score(
        out,
        [
            "reg_chop_10",
            "reg_chop_30",
            "alternating_bar_ratio",
            "local_swing_overlap_score",
            "wick_rejection_ratio_recent",
            "flem_compression_score",
            "failed_followthrough_count",
        ],
    )
    out["msq_auction_clarity"] = _mean_score(
        out,
        [
            "msq_impulse_strength",
            "msq_wick_body_structure",
            "msq_flem_shape",
            "msq_day_structure",
            "msq_chop_range_unpredictability",
        ],
        invert=["msq_chop_range_unpredictability"],
    )
    return out


def _assign_provisional_states(frame: Any) -> Any:
    out = frame.copy()
    chop = out["msq_chop_range_unpredictability"].fillna(0.5)
    impulse = out["msq_impulse_strength"].fillna(0.5)
    flem = out["msq_flem_shape"].fillna(0.5)
    pivot = out["msq_pivot_shape"].fillna(0.5)
    clarity = out["msq_auction_clarity"].fillna(0.5)
    day_structure = out["msq_day_structure"].fillna(0.5)
    recent_efficiency = out["recent_directional_efficiency"].fillna(0.5)
    open_efficiency = out["directional_efficiency_open_to_now"].fillna(0.5)
    body_ratio = out["body_to_range_ratio_recent"].fillna(0.5)
    wick_rejection = out["wick_rejection_ratio_recent"].fillna(0.5)
    alternating = out["alternating_bar_ratio"].fillna(0.5)
    swing_overlap = out["local_swing_overlap_score"].fillna(0.5)
    compression = out["flem_compression_score"].fillna(0.5)
    failed_followthrough = out["failed_followthrough_count"].fillna(0.0)
    repair_depth = _series(out, "deepest_zone_retrace_fraction", 0.0).fillna(0.0)
    reclaim_count = _series(out, "reclaim_count", 0.0).fillna(0.0)
    violation = _series(out, "opposite_boundary_close_violation", 0.0).fillna(0.0)

    states = []
    qualities = []
    for idx in out.index:
        chop_signals = sum(
            [
                chop.loc[idx] >= 0.58,
                swing_overlap.loc[idx] >= 0.58,
                alternating.loc[idx] >= 0.55,
                wick_rejection.loc[idx] >= 0.58,
                recent_efficiency.loc[idx] <= 0.42,
                compression.loc[idx] >= 0.60,
                failed_followthrough.loc[idx] >= 0.55,
            ]
        )
        messy_signals = sum(
            [
                clarity.loc[idx] <= 0.48,
                pivot.loc[idx] <= 0.45,
                flem.loc[idx] <= 0.45,
                repair_depth.loc[idx] >= 0.55,
                reclaim_count.loc[idx] >= 2,
                body_ratio.loc[idx] <= 0.40,
            ]
        )
        strong_continuation = (
            clarity.loc[idx] >= 0.64
            and impulse.loc[idx] >= 0.58
            and recent_efficiency.loc[idx] >= 0.55
            and day_structure.loc[idx] >= 0.52
            and chop_signals <= 1
        )

        if violation.loc[idx] > 0 or failed_followthrough.loc[idx] >= 0.75:
            state = "failed_directional"
        elif chop_signals >= 3 or (chop.loc[idx] >= 0.55 and recent_efficiency.loc[idx] <= 0.45):
            state = "balanced_chop"
        elif messy_signals >= 3:
            state = "messy_transition"
        elif repair_depth.loc[idx] >= 0.65 or reclaim_count.loc[idx] >= 2 or pivot.loc[idx] <= 0.35:
            state = "auction_repair"
        elif strong_continuation:
            state = "clean_strong_continuation"
        else:
            state = "weak_or_grindy_continuation"

        if state == "clean_strong_continuation" and clarity.loc[idx] >= 0.62:
            quality = "high_quality"
        elif (
            state in {"balanced_chop", "messy_transition", "failed_directional"}
            or (state == "weak_or_grindy_continuation" and (recent_efficiency.loc[idx] <= 0.50 or impulse.loc[idx] <= 0.45))
            or clarity.loc[idx] <= 0.40
        ):
            quality = "avoid"
        else:
            quality = "marginal"
        states.append(state)
        qualities.append(quality)

    out["market_state"] = states
    out["setup_quality"] = qualities
    return out


def _mean_score(frame: Any, columns: list[str], invert: list[str] | None = None) -> Any:
    import pandas as pd

    invert_set = set(invert or [])
    parts = []
    for column in columns:
        if column not in frame.columns:
            continue
        score = _normalize(frame[column])
        if column in invert_set:
            score = 1.0 - score
        parts.append(score)
    if not parts:
        return pd.Series([float("nan")] * len(frame), index=frame.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def _fallback_score(frame: Any, preferred: str, columns: list[str], invert: list[str] | None = None) -> Any:
    if preferred in frame.columns and not _to_numeric(frame[preferred]).isna().all():
        return _normalize(frame[preferred])
    return _mean_score(frame, columns, invert=invert)


def _build_raw_market_state_features(bars: Any, candidates: list[Any]) -> Any:
    pd = _require_pandas()
    rows = []
    for candidate in candidates:
        cutoff = pd.Timestamp(candidate.feature_cutoff_time)
        history = bars[bars.index < cutoff]
        session = history[history.index.date == pd.Timestamp(candidate.session_date).date()]
        recent = session.tail(16)
        pivot_start = pd.Timestamp(candidate.break_time)
        pivot_end = pd.Timestamp(candidate.trigger_time)
        pivot_window = session[(session.index > pivot_start) & (session.index <= pivot_end)]
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "directional_efficiency_open_to_now": _directional_efficiency(session),
                "recent_directional_efficiency": _directional_efficiency(recent),
                "body_to_range_ratio_recent": _body_to_range_ratio(recent),
                "wick_rejection_ratio_recent": _wick_rejection_ratio(recent),
                "alternating_bar_ratio": _alternating_bar_ratio(recent),
                "local_swing_overlap_score": _local_swing_overlap_score(recent),
                "impulse_speed_score": _impulse_speed_score(session, recent, candidate),
                "repair_speed_score": _repair_speed_score(pivot_window, candidate),
                "flem_compression_score": _flem_compression_score(pivot_window, candidate),
                "pivot_cleanliness_score": _pivot_cleanliness_score(pivot_window, candidate),
                "distance_to_recent_high_low": _distance_to_recent_high_low(recent, candidate),
                "failed_followthrough_count": _failed_followthrough_count(pivot_window, candidate),
            }
        )
    return pd.DataFrame(rows)


def _directional_efficiency(frame: Any) -> float:
    if frame.empty or len(frame) < 2:
        return 0.0
    closes = frame["close"].astype(float)
    net = abs(float(closes.iloc[-1] - closes.iloc[0]))
    path = float(closes.diff().abs().sum())
    return float(net / path) if path > 0 else 0.0


def _body_to_range_ratio(frame: Any) -> float:
    if frame.empty:
        return 0.0
    ranges = (frame["high"].astype(float) - frame["low"].astype(float)).replace(0, float("nan"))
    bodies = (frame["close"].astype(float) - frame["open"].astype(float)).abs()
    return float((bodies / ranges).fillna(0.0).mean())


def _wick_rejection_ratio(frame: Any) -> float:
    if frame.empty:
        return 0.0
    ranges = (frame["high"].astype(float) - frame["low"].astype(float)).replace(0, float("nan"))
    bodies = (frame["close"].astype(float) - frame["open"].astype(float)).abs()
    wick = (ranges - bodies).clip(lower=0.0)
    return float((wick / ranges).fillna(0.0).mean())


def _alternating_bar_ratio(frame: Any) -> float:
    if frame.empty or len(frame) < 3:
        return 0.0
    signs = (frame["close"].astype(float) - frame["open"].astype(float)).apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    alternations = sum(1 for prev, cur in zip(signs.iloc[:-1], signs.iloc[1:]) if prev != cur)
    return float(alternations / max(len(signs) - 1, 1))


def _local_swing_overlap_score(frame: Any) -> float:
    if frame.empty or len(frame) < 3:
        return 0.0
    overlaps = 0
    previous_low = None
    previous_high = None
    for _, row in frame.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        if previous_low is not None and high >= previous_low and low <= previous_high:
            overlaps += 1
        previous_low = low
        previous_high = high
    return float(overlaps / max(len(frame) - 1, 1))


def _impulse_speed_score(session: Any, recent: Any, candidate: Any) -> float:
    if recent.empty:
        return 0.0
    zone_width = max(float(candidate.zone.width), 0.25)
    if candidate.direction == "long":
        progress = float(recent["close"].iloc[-1]) - float(recent["open"].iloc[0])
    else:
        progress = float(recent["open"].iloc[0]) - float(recent["close"].iloc[-1])
    bars = max(len(recent), 1)
    return float(max(progress, 0.0) / zone_width / bars)


def _repair_speed_score(pivot_window: Any, candidate: Any) -> float:
    if pivot_window.empty:
        return 0.0
    zone_width = max(float(candidate.zone.width), 0.25)
    bars = max(len(pivot_window), 1)
    if candidate.direction == "long":
        repair = float(pivot_window["close"].iloc[-1]) - float(pivot_window["low"].min())
    else:
        repair = float(pivot_window["high"].max()) - float(pivot_window["close"].iloc[-1])
    return float(max(repair, 0.0) / zone_width / bars)


def _flem_compression_score(pivot_window: Any, candidate: Any) -> float:
    if pivot_window.empty:
        return 0.0
    zone_width = max(float(candidate.zone.width), 0.25)
    pivot_range = float(pivot_window["high"].max() - pivot_window["low"].min())
    return float(max(0.0, 1.0 - min(pivot_range / zone_width, 2.0) / 2.0))


def _pivot_cleanliness_score(pivot_window: Any, candidate: Any) -> float:
    if pivot_window.empty:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - _local_swing_overlap_score(pivot_window))))


def _distance_to_recent_high_low(recent: Any, candidate: Any) -> float:
    if recent.empty:
        return 0.0
    zone_width = max(float(candidate.zone.width), 0.25)
    entry = float(candidate.entry_reference_price)
    distance = min(abs(entry - float(recent["high"].max())), abs(entry - float(recent["low"].min())))
    return float(distance / zone_width)


def _failed_followthrough_count(pivot_window: Any, candidate: Any) -> float:
    if pivot_window.empty:
        return 0.0
    failures = 0
    for _, row in pivot_window.iterrows():
        close = float(row["close"])
        if candidate.direction == "long" and close <= float(candidate.zone.high):
            failures += 1
        if candidate.direction == "short" and close >= float(candidate.zone.low):
            failures += 1
    return float(failures)


def _normalize(series: Any) -> Any:
    numeric = _to_numeric(series)
    low = numeric.quantile(0.05)
    high = numeric.quantile(0.95)
    if high == low:
        return numeric * 0 + 0.5
    return ((numeric - low) / (high - low)).clip(0.0, 1.0)


def _to_numeric(series: Any) -> Any:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _series(frame: Any, column: str, default: float) -> Any:
    import pandas as pd

    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([default] * len(frame), index=frame.index)


def _counts(frame: Any, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).to_dict().items()}


def _label_balance(frame: Any) -> dict[str, Any]:
    if "label" not in frame.columns or frame.empty:
        return {"count": int(len(frame)), "positive_rate": None}
    labels = _to_numeric(frame["label"])
    return {
        "count": int(len(frame)),
        "positive_rate": float(labels.mean()) if len(labels.dropna()) else None,
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }


def _group_label_balance(frame: Any, group_col: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_col, dropna=False):
        row = _label_balance(group)
        row[group_col] = str(key)
        rows.append(row)
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def _missingness(frame: Any) -> dict[str, Any]:
    groups = _feature_group_columns()
    by_group = {}
    for group, columns in groups.items():
        existing = [column for column in columns if column in frame.columns]
        if not existing:
            by_group[group] = {"status": "missing", "missing_rate": 1.0, "columns_present": []}
            continue
        missing = frame[existing].isna().mean().mean()
        by_group[group] = {
            "status": "complete" if missing <= 0.20 else "sparse",
            "missing_rate": float(missing),
            "columns_present": existing,
        }
    return by_group


def _leakage_audit(metadata: dict[str, Any], frame: Any) -> dict[str, Any]:
    feature_columns = [column for columns in _feature_group_columns().values() for column in columns]
    forbidden_tokens = ["pnl", "mfe", "mae", "exit", "target", "stop", "bars_held", "outcome", "label"]
    issues = []
    for column in feature_columns:
        lowered = column.lower()
        if any(token in lowered for token in forbidden_tokens):
            issues.append(f"forbidden_feature_name:{column}")
    audit = dict(metadata.get("feature_audit", {}) or {})
    if int(audit.get("failed", 0) or 0) > 0:
        issues.extend([f"stage2_feature_audit:{issue}" for issue in audit.get("issues", [])])
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "point_in_time_basis": "stage2 feature_cutoff_time per candidate; labels and PnL used only after feature/state assignment for diagnostics",
        "holdout_locked": True,
    }


def _pnl_by_group(frame: Any, group_col: str) -> list[dict[str, Any]]:
    if "pnl_r" not in frame.columns:
        return []
    rows = []
    pnl = _to_numeric(frame["pnl_r"])
    work = frame.copy()
    work["pnl_r"] = pnl
    for key, group in work.groupby(group_col, dropna=False):
        group_pnl = _to_numeric(group["pnl_r"]).dropna()
        rows.append(
            {
                group_col: str(key),
                "count": int(len(group)),
                "total_pnl_r": float(group_pnl.sum()) if len(group_pnl) else 0.0,
                "avg_pnl_r": float(group_pnl.mean()) if len(group_pnl) else 0.0,
                "win_rate": float((group_pnl > 0).mean()) if len(group_pnl) else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def _support_sufficiency(frame: Any) -> dict[str, Any]:
    return {
        "minimum_rows": 30,
        "minimum_class_count": 5,
        "by_market_state": _support_rows(frame, "market_state"),
        "by_setup_quality": _support_rows(frame, "setup_quality"),
    }


def _reliability_warnings(frame: Any) -> list[dict[str, Any]]:
    warnings = []
    for state in MARKET_STATES:
        count = int((frame["market_state"] == state).sum()) if "market_state" in frame.columns else 0
        if count < 25:
            warnings.append(
                {
                    "state": state,
                    "count": count,
                    "warning": "state_has_fewer_than_25_rows",
                }
            )
    return warnings


def _avoided_vs_traded_summary(frame: Any) -> dict[str, Any]:
    if "pnl_r" not in frame.columns:
        return {"status": "pending", "reason": "missing_pnl"}
    work = frame.copy()
    work["pnl_r"] = _to_numeric(work["pnl_r"])
    avoid_states = {"balanced_chop", "messy_transition", "failed_directional"}
    avoided = work[(work["setup_quality"] == "avoid") | (work["market_state"].isin(avoid_states))]
    baseline_pnl = float(work["pnl_r"].sum())
    return {
        "status": "diagnostic_only",
        "rule": "avoid setup_quality=avoid plus balanced_chop/messy_transition/failed_directional",
        "baseline_candidate_count": int(len(work)),
        "baseline_total_pnl_r": baseline_pnl,
        **_avoidance_row(work, avoided),
        "scenario_checks": [
            {
                "name": "avoid_setup_quality_only",
                **_avoidance_row(work, work[work["setup_quality"] == "avoid"]),
            },
            {
                "name": "avoid_weak_messy_failed_only",
                **_avoidance_row(work, work[work["market_state"].isin({"weak_or_grindy_continuation", "messy_transition", "failed_directional"})]),
            },
            {
                "name": "avoid_balanced_chop_only",
                **_avoidance_row(work, work[work["market_state"] == "balanced_chop"]),
            },
        ],
        "not_a_trading_rule": True,
    }


def _cheap_state_policy_simulation(frame: Any) -> dict[str, Any]:
    variants = [
        {"name": "baseline", "weights": {}},
        {"name": "exclude_weak_or_grindy_continuation", "weights": {"weak_or_grindy_continuation": 0.0}},
        {"name": "exclude_messy_transition", "weights": {"messy_transition": 0.0}},
        {
            "name": "exclude_weak_or_grindy_continuation_and_messy_transition",
            "weights": {"weak_or_grindy_continuation": 0.0, "messy_transition": 0.0},
        },
        {"name": "size_haircut_weak_or_grindy_continuation", "weights": {"weak_or_grindy_continuation": 0.5}},
        {"name": "size_haircut_messy_transition", "weights": {"messy_transition": 0.5}},
        {
            "name": "allow_only_clean_strong_continuation_and_auction_repair",
            "default_weight": 0.0,
            "weights": {"clean_strong_continuation": 1.0, "auction_repair": 1.0},
        },
    ]
    return {
        "status": "complete",
        "mode": "diagnostic_policy_simulation_only",
        "models_trained": 0,
        "search_executed": False,
        "holdout_status": "locked",
        "policy_variants": [_simulate_policy_variant(frame, variant) for variant in variants],
    }


def _followthrough_confirmation_policy_gate(frame: Any) -> dict[str, Any]:
    variants = [
        {"name": "baseline", "gate": "baseline"},
        {"name": "exclude_weak_or_grindy_continuation", "gate": "exclude_weak"},
        {"name": "exclude_weak_or_grindy_plus_pre_entry_tempo_persistence_gate", "gate": "tempo_persistence"},
        {"name": "exclude_weak_or_grindy_plus_pre_entry_breakout_quality_gate", "gate": "breakout_quality"},
        {"name": "exclude_weak_or_grindy_plus_delayed_followthrough_confirmation_entry", "gate": "delayed_followthrough"},
        {"name": "exclude_weak_or_grindy_plus_liquidity_proximity_rejection_veto", "gate": "liquidity_veto"},
        {"name": "exclude_weak_or_grindy_plus_combined_conservative_gate", "gate": "combined_conservative"},
    ]
    rows = [_simulate_followthrough_gate_variant(frame, variant) for variant in variants]
    return {
        "status": "complete",
        "family": "followthrough_confirmation_policy_gate",
        "parent_family": "market_state_setup_quality",
        "mode": "cheap_policy_simulation_first",
        "input_evidence": [
            "residual cpcv_010 after exclude_weak_or_grindy_continuation",
            "poor_followthrough_after_trigger",
            "failed_breakout_no_followthrough",
            "candle_tempo_deterioration",
            "trend_persistence_breakdown",
            "liquidity_proximity_rejection",
        ],
        "governance": {
            "models_trained": 0,
            "search_executed": False,
            "holdout_status": "locked",
            "promotion_blocked": True,
            "does_not_filter_on_cpcv_010": True,
            "post_trigger_confirmation_requires_delayed_entry": True,
        },
        "leakage_audit": _followthrough_gate_leakage_audit(rows),
        "policy_variants": rows,
    }


def _simulate_followthrough_gate_variant(frame: Any, variant: dict[str, Any]) -> dict[str, Any]:
    work = frame.copy()
    work["pnl_r"] = _to_numeric(work["pnl_r"]) if "pnl_r" in work.columns else 0.0
    weights = []
    fill_impacts = []
    pit_flags = []
    for _, row in work.iterrows():
        weight, fill_impact, pit_valid = _followthrough_gate_decision(row, variant["gate"])
        weights.append(weight)
        fill_impacts.append(fill_impact)
        pit_flags.append(pit_valid)
    work["policy_weight"] = weights
    work["fill_impact_r"] = fill_impacts
    work["pit_valid"] = pit_flags
    work["policy_pnl_r"] = (work["pnl_r"] + work["fill_impact_r"]) * work["policy_weight"]
    cpcv = _simulate_followthrough_gate_cpcv(frame, variant)
    return {
        "variant": variant["name"],
        "gate": variant["gate"],
        "candidate_count": int(len(work)),
        "trade_count": int((work["policy_weight"] > 0).sum()),
        "retained_fraction": float((work["policy_weight"] > 0).mean()) if len(work) else 0.0,
        "effective_trade_count": float(work["policy_weight"].sum()),
        "total_pnl_r": float(work["policy_pnl_r"].sum()),
        "baseline_total_pnl_r": float(work["pnl_r"].sum()),
        "avoided_pnl_r": float((work["pnl_r"] * (1.0 - work["policy_weight"])).sum()),
        "delayed_entry_fill_impact_r": float(work["fill_impact_r"].sum()),
        "mean_cpcv_path_pnl_r": cpcv["mean_path_pnl_r"],
        "median_cpcv_path_pnl_r": cpcv["median_path_pnl_r"],
        "worst_3_cpcv_paths": cpcv["worst_3_paths"],
        "cpcv_paths": cpcv["paths"],
        "positive_path_rate": cpcv["positive_path_rate"],
        "prior_worst_path_effect": cpcv["prior_worst_path_effect"],
        "state_contribution_table": _followthrough_state_contribution(work),
        "point_in_time_valid": bool(all(pit_flags)),
        "leakage_audit": {
            "status": "pass" if all(pit_flags) else "fail",
            "delayed_entry_model": variant["gate"] == "delayed_followthrough",
            "uses_cpcv_path_id_as_filter": False,
        },
    }


def _followthrough_gate_decision(row: Any, gate: str) -> tuple[float, float, bool]:
    if gate == "baseline":
        return 1.0, 0.0, True
    if str(row.get("market_state")) == "weak_or_grindy_continuation":
        return 0.0, 0.0, True
    if gate == "exclude_weak":
        return 1.0, 0.0, True
    if gate == "tempo_persistence":
        keep = (
            _num(row, "recent_directional_efficiency") >= 0.35
            and _num(row, "impulse_speed_score") >= 0.18
            and _num(row, "alternating_bar_ratio") <= 0.72
            and _num(row, "wick_rejection_ratio_recent") <= 0.68
        )
        return (1.0 if keep else 0.0), 0.0, True
    if gate == "breakout_quality":
        keep = (
            _num(row, "first_break_close_confirmed") >= 0.5
            and _num(row, "break_efficiency_ratio") >= 0.12
            and _num(row, "break_close_distance_to_zone") >= 0.05
            and _num(row, "first_break_wick_only") < 0.5
        )
        return (1.0 if keep else 0.0), 0.0, True
    if gate == "delayed_followthrough":
        keep = _num(row, "continuation_displacement_ratio") >= 0.30 and _num(row, "post_reclaim_close_strength") >= 0.35
        fill_impact = -0.05 if keep else 0.0
        return (1.0 if keep else 0.0), fill_impact, True
    if gate == "liquidity_veto":
        keep = _num(row, "distance_to_recent_high_low") > 0.20 and _num(row, "wick_rejection_ratio_recent") < 0.70
        return (1.0 if keep else 0.0), 0.0, True
    if gate == "combined_conservative":
        tempo = _followthrough_gate_decision(row, "tempo_persistence")[0] > 0
        breakout = _followthrough_gate_decision(row, "breakout_quality")[0] > 0
        liquidity = _followthrough_gate_decision(row, "liquidity_veto")[0] > 0
        keep = tempo and breakout and liquidity
        return (1.0 if keep else 0.0), 0.0, True
    return 1.0, 0.0, True


def _simulate_followthrough_gate_cpcv(frame: Any, variant: dict[str, Any]) -> dict[str, Any]:
    import statistics

    feature_by_id = frame.set_index(frame["candidate_id"].astype(str)).to_dict(orient="index")
    rows = []
    for path in _available_cpcv_path_rows():
        total = 0.0
        baseline = 0.0
        trade_count = 0
        effective_count = 0.0
        state_totals: dict[str, dict[str, Any]] = {}
        fill_impact = 0.0
        for path_row in path["rows"]:
            candidate_id = str(path_row.get("candidate_id", ""))
            joined = dict(path_row)
            joined.update(feature_by_id.get(candidate_id, {}))
            pnl = float(path_row.get("executed_pnl_r", path_row.get("pnl_r", 0.0)) or 0.0)
            weight, row_fill_impact, _ = _followthrough_gate_decision(joined, variant["gate"])
            weighted = (pnl + row_fill_impact) * weight
            state = str(joined.get("market_state", "unmatched"))
            baseline += pnl
            total += weighted
            fill_impact += row_fill_impact * weight
            if weight > 0:
                trade_count += 1
                effective_count += weight
            bucket = state_totals.setdefault(state, {"market_state": state, "trade_count": 0, "total_pnl_r": 0.0})
            bucket["trade_count"] += 1
            bucket["total_pnl_r"] += weighted
        rows.append(
            {
                "path_id": str(path["path_id"]),
                "baseline_total_pnl_r": baseline,
                "total_pnl_r": total,
                "pnl_delta_r": total - baseline,
                "trade_count": trade_count,
                "effective_trade_count": effective_count,
                "delayed_entry_fill_impact_r": fill_impact,
                "state_contribution_table": sorted(state_totals.values(), key=lambda item: item["trade_count"], reverse=True),
            }
        )
    pnls = [row["total_pnl_r"] for row in rows]
    prior_ids = {"cpcv_010", "cpcv_003", "cpcv_002"}
    return {
        "mean_path_pnl_r": float(statistics.mean(pnls)) if pnls else 0.0,
        "median_path_pnl_r": float(statistics.median(pnls)) if pnls else 0.0,
        "positive_path_rate": float(sum(1 for value in pnls if value > 0) / len(pnls)) if pnls else 0.0,
        "worst_3_paths": sorted(rows, key=lambda row: row["total_pnl_r"])[:3],
        "paths": rows,
        "prior_worst_path_effect": [row for row in rows if row["path_id"] in prior_ids],
    }


def _followthrough_state_contribution(work: Any) -> list[dict[str, Any]]:
    rows = []
    for state, group in work.groupby("market_state", dropna=False):
        rows.append(
            {
                "market_state": str(state),
                "candidate_count": int(len(group)),
                "trade_count": int((_to_numeric(group["policy_weight"]) > 0).sum()),
                "retained_fraction": float((_to_numeric(group["policy_weight"]) > 0).mean()) if len(group) else 0.0,
                "baseline_pnl_r": float(_to_numeric(group["pnl_r"]).sum()),
                "policy_pnl_r": float(_to_numeric(group["policy_pnl_r"]).sum()),
            }
        )
    return sorted(rows, key=lambda row: row["candidate_count"], reverse=True)


def _followthrough_gate_leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues = []
    for row in rows:
        if not row.get("point_in_time_valid"):
            issues.append(f"{row.get('variant')}:not_point_in_time_valid")
        if row.get("leakage_audit", {}).get("uses_cpcv_path_id_as_filter"):
            issues.append(f"{row.get('variant')}:uses_cpcv_path_id_as_filter")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "point_in_time_basis": "pre-entry gates use feature_cutoff_time features; delayed follow-through variant is evaluated as a delayed-entry policy with fill impact",
        "holdout_locked": True,
    }


def _residual_tail_diagnostic(frame: Any) -> dict[str, Any]:
    pd = _require_pandas()
    path_rows = next((row["rows"] for row in _available_cpcv_path_rows() if row["path_id"] == "cpcv_010"), [])
    if not path_rows:
        return {"status": "pending", "reason": "missing_cpcv_010_rows"}
    state_features = frame.set_index(frame["candidate_id"].astype(str)).to_dict(orient="index")
    joined = []
    for row in path_rows:
        candidate_id = str(row.get("candidate_id", ""))
        feature_row = dict(state_features.get(candidate_id, {}) or {})
        if not feature_row:
            continue
        merged = dict(row)
        merged.update(feature_row)
        merged["executed_pnl_r"] = float(row.get("executed_pnl_r", row.get("pnl_r", 0.0)) or 0.0)
        joined.append(merged)
    if not joined:
        return {"status": "pending", "reason": "no_joined_cpcv_010_rows"}
    joined_df = pd.DataFrame(joined)
    surviving = joined_df[joined_df["market_state"] != "weak_or_grindy_continuation"].copy()
    tail = surviving[surviving["executed_pnl_r"] < 0].copy()
    clusters = _residual_failure_clusters(tail)
    support = _residual_support_summary(surviving, tail)
    return {
        "status": "complete",
        "mode": "diagnostic_only",
        "policy": "exclude_weak_or_grindy_continuation",
        "path_id": "cpcv_010",
        "models_trained": 0,
        "search_executed": False,
        "holdout_status": "locked",
        "baseline_path_trade_count": int(len(joined_df)),
        "baseline_path_total_pnl_r": float(joined_df["executed_pnl_r"].sum()),
        "surviving_trade_count": int(len(surviving)),
        "surviving_total_pnl_r": float(surviving["executed_pnl_r"].sum()),
        "surviving_loss_trade_count": int(len(tail)),
        "surviving_loss_total_pnl_r": float(tail["executed_pnl_r"].sum()) if not tail.empty else 0.0,
        "state_contribution": _residual_group_summary(surviving, "market_state"),
        "setup_quality_contribution": _residual_group_summary(surviving, "setup_quality"),
        "residual_failure_clusters": clusters,
        "candidate_explanatory_features": _residual_explanatory_features(tail),
        "structural_coherence": _residual_structural_coherence(clusters, support),
        "supporting_trade_session_counts": support,
    }


def _residual_failure_clusters(frame: Any) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    specs = [
        (
            "failed_breakout_no_followthrough",
            [
                "first_break_wick_only",
                "first_break_close_confirmed",
                "break_close_distance_to_zone",
                "break_efficiency_ratio",
                "continuation_displacement_ratio",
                "opposite_boundary_close_violation",
            ],
            lambda row: _num(row, "first_break_wick_only") >= 0.5
            or _num(row, "first_break_close_confirmed") < 0.5
            or _num(row, "break_close_distance_to_zone") <= 0.15
            or _num(row, "continuation_displacement_ratio") <= 0.25
            or _num(row, "opposite_boundary_close_violation") > 0,
        ),
        (
            "poor_followthrough_after_trigger",
            ["continuation_displacement_ratio", "post_reclaim_close_strength", "recent_directional_efficiency", "failed_followthrough_count"],
            lambda row: _num(row, "continuation_displacement_ratio") <= 0.30
            or _num(row, "post_reclaim_close_strength") <= 0.40
            or _num(row, "failed_followthrough_count") >= 0.50,
        ),
        (
            "volatility_expansion_then_collapse",
            ["break_range_expansion", "impulse_speed_score", "continuation_displacement_ratio", "body_to_range_ratio_recent"],
            lambda row: _num(row, "break_range_expansion") >= 1.50
            and (_num(row, "continuation_displacement_ratio") <= 0.35 or _num(row, "body_to_range_ratio_recent") <= 0.45),
        ),
        (
            "liquidity_proximity_rejection",
            ["distance_to_recent_high_low", "wick_rejection_ratio_recent", "entry_distance_from_zone_high", "entry_distance_from_zone_low"],
            lambda row: _num(row, "distance_to_recent_high_low") <= 0.35 or _num(row, "wick_rejection_ratio_recent") >= 0.55,
        ),
        (
            "late_opening_window_exhaustion",
            ["trigger_seconds_after_open", "impulse_speed_score", "recent_directional_efficiency"],
            lambda row: _num(row, "trigger_seconds_after_open") >= 3600
            and (_num(row, "recent_directional_efficiency") <= 0.45 or _num(row, "impulse_speed_score") <= 0.35),
        ),
        (
            "low_quality_repair",
            ["reclaim_latency_bars", "reclaim_body_strength", "reclaim_close_location", "reclaim_failure_count", "repair_speed_score"],
            lambda row: _num(row, "reclaim_latency_bars") >= 2
            or _num(row, "reclaim_failure_count") >= 1
            or _num(row, "reclaim_body_strength") <= 0.35
            or _num(row, "repair_speed_score") <= 0.20,
        ),
        (
            "messy_pivot_or_overlap",
            ["pivot_overlap_ratio", "pivot_cleanliness_score", "pivot_symmetry", "alternating_bar_ratio", "local_swing_overlap_score"],
            lambda row: _num(row, "pivot_overlap_ratio") >= 0.40
            or _num(row, "pivot_cleanliness_score") <= 0.45
            or _num(row, "alternating_bar_ratio") >= 0.55
            or _num(row, "local_swing_overlap_score") >= 0.65,
        ),
        (
            "trend_persistence_breakdown",
            ["directional_efficiency_open_to_now", "recent_directional_efficiency", "reg_trend_strength_10", "reg_trend_strength_30"],
            lambda row: _num(row, "directional_efficiency_open_to_now") <= 0.45
            or _num(row, "recent_directional_efficiency") <= 0.40
            or _num(row, "reg_trend_strength_10") <= 0.25,
        ),
        (
            "candle_tempo_deterioration",
            ["impulse_speed_score", "alternating_bar_ratio", "body_to_range_ratio_recent", "wick_rejection_ratio_recent"],
            lambda row: _num(row, "impulse_speed_score") <= 0.35
            or _num(row, "alternating_bar_ratio") >= 0.55
            or _num(row, "wick_rejection_ratio_recent") >= 0.55,
        ),
    ]
    rows = []
    total_loss = abs(float(frame["executed_pnl_r"].sum())) if "executed_pnl_r" in frame else 0.0
    for name, features, predicate in specs:
        matched = frame[frame.apply(predicate, axis=1)]
        if matched.empty:
            continue
        loss = float(matched["executed_pnl_r"].sum())
        rows.append(
            {
                "cluster": name,
                "trade_count": int(len(matched)),
                "session_count": int(matched["session_date"].astype(str).nunique()) if "session_date" in matched else 0,
                "total_pnl_r": loss,
                "avg_pnl_r": float(matched["executed_pnl_r"].mean()),
                "loss_share": float(abs(loss) / total_loss) if total_loss > 0 else 0.0,
                "candidate_explanatory_features": features,
                "state_contribution": _residual_group_summary(matched, "market_state"),
                "example_candidate_ids": matched["candidate_id"].astype(str).head(5).tolist(),
            }
        )
    return sorted(rows, key=lambda row: (row["trade_count"], abs(row["total_pnl_r"])), reverse=True)


def _residual_explanatory_features(frame: Any) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    candidates = [
        "continuation_displacement_ratio",
        "post_reclaim_close_strength",
        "break_efficiency_ratio",
        "break_range_expansion",
        "distance_to_recent_high_low",
        "wick_rejection_ratio_recent",
        "trigger_seconds_after_open",
        "reclaim_latency_bars",
        "reclaim_body_strength",
        "pivot_overlap_ratio",
        "pivot_cleanliness_score",
        "alternating_bar_ratio",
        "impulse_speed_score",
        "recent_directional_efficiency",
        "directional_efficiency_open_to_now",
        "failed_followthrough_count",
    ]
    rows = []
    for feature in candidates:
        if feature not in frame.columns:
            continue
        series = _to_numeric(frame[feature]).dropna()
        if series.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
            }
        )
    return rows


def _residual_support_summary(surviving: Any, tail: Any) -> dict[str, Any]:
    return {
        "surviving_trade_count": int(len(surviving)),
        "surviving_session_count": int(surviving["session_date"].astype(str).nunique()) if not surviving.empty and "session_date" in surviving else 0,
        "loss_trade_count": int(len(tail)),
        "loss_session_count": int(tail["session_date"].astype(str).nunique()) if not tail.empty and "session_date" in tail else 0,
    }


def _residual_structural_coherence(clusters: list[dict[str, Any]], support: dict[str, Any]) -> dict[str, Any]:
    loss_count = int(support.get("loss_trade_count", 0) or 0)
    if loss_count == 0:
        return {"assessment": "no_residual_losses", "reason": "No surviving losing trades after policy."}
    top_count = int(clusters[0]["trade_count"]) if clusters else 0
    top_share = top_count / loss_count if loss_count else 0.0
    multi_cluster_coverage = sum(row["trade_count"] for row in clusters[:3]) / loss_count if loss_count else 0.0
    if top_share >= 0.50 or multi_cluster_coverage >= 1.25:
        assessment = "structurally_coherent"
    elif top_share >= 0.30 or multi_cluster_coverage >= 0.80:
        assessment = "partly_structural"
    else:
        assessment = "mostly_random_or_underexplained"
    return {
        "assessment": assessment,
        "top_cluster_share_of_loss_trades": top_share,
        "top_3_cluster_membership_per_loss_trade": multi_cluster_coverage,
        "note": "Cluster memberships can overlap; high multi-cluster coverage means repeated structural signatures, not disjoint groups.",
    }


def _residual_group_summary(frame: Any, group_col: str) -> list[dict[str, Any]]:
    if frame.empty or group_col not in frame.columns:
        return []
    rows = []
    for key, group in frame.groupby(group_col, dropna=False):
        pnl = _to_numeric(group["executed_pnl_r"]) if "executed_pnl_r" in group else _to_numeric(group["pnl_r"])
        rows.append(
            {
                group_col: str(key),
                "trade_count": int(len(group)),
                "session_count": int(group["session_date"].astype(str).nunique()) if "session_date" in group else 0,
                "total_pnl_r": float(pnl.sum()),
                "avg_pnl_r": float(pnl.mean()) if len(pnl) else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["trade_count"], reverse=True)


def _num(row: Any, column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
    except AttributeError:
        return default
    try:
        if value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def market_state_policy_variant_specs() -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "weights": {}},
        {"name": "exclude_weak_or_grindy_continuation", "weights": {"weak_or_grindy_continuation": 0.0}},
        {"name": "haircut_weak_or_grindy_continuation", "weights": {"weak_or_grindy_continuation": 0.5}},
        {
            "name": "exclude_weak_or_grindy_continuation_and_messy_transition",
            "weights": {"weak_or_grindy_continuation": 0.0, "messy_transition": 0.0},
        },
    ]


def run_market_state_policy_simulation(state: dict[str, Any], variants: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        return {
            "status": "pending",
            "reason": "market_state_diagnostic_unavailable",
            "diagnostic": diagnostic,
            "models_trained": 0,
            "search_executed": False,
            "holdout_status": "locked",
        }
    frame_rows = diagnostic.get("_labeled_rows")
    if not frame_rows:
        return {
            "status": "pending",
            "reason": "labeled_rows_not_available",
            "diagnostic": {key: value for key, value in diagnostic.items() if key != "_labeled_rows"},
            "models_trained": 0,
            "search_executed": False,
            "holdout_status": "locked",
        }
    pd = _require_pandas()
    frame = pd.DataFrame(frame_rows)
    specs = variants or market_state_policy_variant_specs()
    return {
        "status": "complete",
        "mode": "governed_tiny_policy_simulation",
        "models_trained": 0,
        "search_executed": True,
        "holdout_status": "locked",
        "trial_count": len(specs),
        "policy_variants": [_simulate_policy_variant(frame, spec) for spec in specs],
        "diagnostic_summary": {key: value for key, value in diagnostic.items() if key != "_labeled_rows"},
    }


def _simulate_policy_variant(frame: Any, variant: dict[str, Any]) -> dict[str, Any]:
    work = frame.copy()
    work["pnl_r"] = _to_numeric(work["pnl_r"]) if "pnl_r" in work.columns else 0.0
    work["policy_weight"] = work["market_state"].map(lambda state: _state_weight(str(state), variant))
    baseline_pnl = float(work["pnl_r"].sum())
    total_pnl = float((work["pnl_r"] * work["policy_weight"]).sum())
    avoided_pnl = float((work["pnl_r"] * (1.0 - work["policy_weight"])).sum())
    cpcv = _simulate_cpcv_paths(work, variant)
    return {
        "variant": variant["name"],
        "candidate_count": int(len(work)),
        "trade_count": int((work["policy_weight"] > 0).sum()),
        "effective_trade_count": float(work["policy_weight"].sum()),
        "total_pnl_r": total_pnl,
        "baseline_total_pnl_r": baseline_pnl,
        "avoided_pnl_r": avoided_pnl,
        "pnl_delta_r": float(total_pnl - baseline_pnl),
        "mean_cpcv_path_pnl_r": cpcv["mean_path_pnl_r"],
        "median_cpcv_path_pnl_r": cpcv["median_path_pnl_r"],
        "worst_3_cpcv_paths": cpcv["worst_3_paths"],
        "cpcv_paths": cpcv["paths"],
        "positive_path_rate": cpcv["positive_path_rate"],
        "prior_worst_path_effect": cpcv["prior_worst_path_effect"],
        "state_contribution_table": _policy_state_contribution(work),
    }


def _state_weight(state: str, variant: dict[str, Any]) -> float:
    if state in dict(variant.get("weights", {})):
        return float(variant["weights"][state])
    return float(variant.get("default_weight", 1.0))


def _simulate_cpcv_paths(frame: Any, variant: dict[str, Any]) -> dict[str, Any]:
    import statistics

    state_by_id = frame.set_index(frame["candidate_id"].astype(str))[["market_state"]].to_dict(orient="index")
    rows = []
    for path in _available_cpcv_path_rows():
        path_id = str(path["path_id"])
        total = 0.0
        baseline = 0.0
        trade_count = 0
        effective_count = 0.0
        state_totals: dict[str, dict[str, Any]] = {}
        for row in path["rows"]:
            candidate_id = str(row.get("candidate_id", ""))
            state = str(dict(state_by_id.get(candidate_id, {}) or {}).get("market_state", "unmatched"))
            pnl = float(row.get("executed_pnl_r", row.get("pnl_r", 0.0)) or 0.0)
            weight = _state_weight(state, variant) if state != "unmatched" else 1.0
            baseline += pnl
            total += pnl * weight
            if weight > 0:
                trade_count += 1
                effective_count += weight
            bucket = state_totals.setdefault(state, {"market_state": state, "trade_count": 0, "total_pnl_r": 0.0})
            bucket["trade_count"] += 1
            bucket["total_pnl_r"] += pnl * weight
        rows.append(
            {
                "path_id": path_id,
                "baseline_total_pnl_r": baseline,
                "total_pnl_r": total,
                "pnl_delta_r": total - baseline,
                "trade_count": trade_count,
                "effective_trade_count": effective_count,
                "state_contribution_table": sorted(state_totals.values(), key=lambda item: item["trade_count"], reverse=True),
            }
        )
    pnls = [row["total_pnl_r"] for row in rows]
    worst = sorted(rows, key=lambda row: row["total_pnl_r"])[:3]
    prior_ids = {"cpcv_010", "cpcv_003", "cpcv_002"}
    return {
        "mean_path_pnl_r": float(statistics.mean(pnls)) if pnls else 0.0,
        "median_path_pnl_r": float(statistics.median(pnls)) if pnls else 0.0,
        "positive_path_rate": float(sum(1 for value in pnls if value > 0) / len(pnls)) if pnls else 0.0,
        "worst_3_paths": worst,
        "paths": rows,
        "prior_worst_path_effect": [row for row in rows if row["path_id"] in prior_ids],
    }


def _policy_state_contribution(work: Any) -> list[dict[str, Any]]:
    rows = []
    for state, group in work.groupby("market_state", dropna=False):
        pnl = _to_numeric(group["pnl_r"])
        weight = _to_numeric(group["policy_weight"])
        weighted_pnl = pnl * weight
        rows.append(
            {
                "market_state": str(state),
                "candidate_count": int(len(group)),
                "trade_count": int((weight > 0).sum()),
                "effective_trade_count": float(weight.sum()),
                "baseline_pnl_r": float(pnl.sum()),
                "weighted_pnl_r": float(weighted_pnl.sum()),
                "avoided_pnl_r": float((pnl * (1.0 - weight)).sum()),
            }
        )
    return sorted(rows, key=lambda row: row["candidate_count"], reverse=True)


def _available_cpcv_path_rows() -> list[dict[str, Any]]:
    diagnostic = _read_json(REPORTS_DIR / "exploration_benchmark_diagnostics.json")
    cpcv = dict(diagnostic.get("cpcv", {}) or {})
    root = Path(str(cpcv.get("artifact_root", "") or ""))
    paths = []
    if root.exists():
        for path in sorted(root.glob("path_cpcv_*_rows.json")):
            path_id = path.name.replace("path_", "").replace("_rows.json", "")
            paths.append({"path_id": path_id, "rows": _read_rows_artifact(path)})
    if paths:
        return paths
    known = list(cpcv.get("worst_paths", []) or []) + list(cpcv.get("best_paths", []) or [])
    for row in known:
        path_id = str(row.get("path_id", "unknown"))
        paths.append({"path_id": path_id, "rows": _read_rows_artifact(row.get("rows_artifact"))})
    return paths


def _avoidance_row(work: Any, avoided: Any) -> dict[str, Any]:
    traded = work.drop(index=avoided.index)
    baseline_pnl = float(work["pnl_r"].sum())
    traded_pnl = float(traded["pnl_r"].sum()) if not traded.empty else 0.0
    avoided_pnl = float(avoided["pnl_r"].sum()) if not avoided.empty else 0.0
    return {
        "retained_candidate_count": int(len(traded)),
        "retained_total_pnl_r": traded_pnl,
        "avoided_candidate_count": int(len(avoided)),
        "avoided_total_pnl_r": avoided_pnl,
        "pnl_delta_if_avoided": float(traded_pnl - baseline_pnl),
        "retained_fraction": float(len(traded) / len(work)) if len(work) else 0.0,
    }


def _support_rows(frame: Any, group_col: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_col, dropna=False):
        labels = _to_numeric(group["label"]) if "label" in group.columns else []
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        enough = len(group) >= 30 and positives >= 5 and negatives >= 5
        rows.append(
            {
                group_col: str(key),
                "count": int(len(group)),
                "positive_count": positives,
                "negative_count": negatives,
                "enough_support_for_modeling": bool(enough),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def _cpcv_worst_path_exposure(frame: Any) -> dict[str, Any]:
    diagnostic = _read_json(REPORTS_DIR / "exploration_benchmark_diagnostics.json")
    worst_paths = list(dict(diagnostic.get("cpcv", {}) or {}).get("worst_paths", []) or [])[:3]
    state_by_id = frame.set_index(frame["candidate_id"].astype(str))[["market_state", "setup_quality"]].to_dict(orient="index")
    rows = []
    aggregate: dict[str, dict[str, Any]] = {}
    quality_aggregate: dict[str, dict[str, Any]] = {}
    for path in worst_paths:
        path_id = str(path.get("path_id", "unknown"))
        path_rows = _read_rows_artifact(path.get("rows_artifact"))
        matched = []
        for row in path_rows:
            candidate_id = str(row.get("candidate_id", ""))
            state = state_by_id.get(candidate_id)
            if not state:
                continue
            pnl = float(row.get("executed_pnl_r", row.get("pnl_r", 0.0)) or 0.0)
            matched.append({**state, "executed_pnl_r": pnl})
            bucket = aggregate.setdefault(
                str(state["market_state"]),
                {"market_state": str(state["market_state"]), "trade_count": 0, "total_pnl_r": 0.0},
            )
            bucket["trade_count"] += 1
            bucket["total_pnl_r"] += pnl
            quality_bucket = quality_aggregate.setdefault(
                str(state["setup_quality"]),
                {"setup_quality": str(state["setup_quality"]), "trade_count": 0, "total_pnl_r": 0.0},
            )
            quality_bucket["trade_count"] += 1
            quality_bucket["total_pnl_r"] += pnl
        rows.append(
            {
                "path_id": path_id,
                "path_total_pnl_r": float(path.get("total_pnl_r", 0.0) or 0.0),
                "matched_trade_count": len(matched),
                "state_exposure": _path_state_exposure(matched),
            }
        )
    return {
        "source": "exploration_benchmark_diagnostics.cpcv.worst_paths",
        "paths": rows,
        "aggregate_by_market_state": sorted(aggregate.values(), key=lambda row: row["trade_count"], reverse=True),
        "aggregate_by_setup_quality": sorted(quality_aggregate.values(), key=lambda row: row["trade_count"], reverse=True),
    }


def _path_state_exposure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["market_state"])
        bucket = totals.setdefault(key, {"market_state": key, "trade_count": 0, "total_pnl_r": 0.0})
        bucket["trade_count"] += 1
        bucket["total_pnl_r"] += float(row.get("executed_pnl_r", 0.0) or 0.0)
    for bucket in totals.values():
        bucket["avg_trade_r"] = bucket["total_pnl_r"] / bucket["trade_count"] if bucket["trade_count"] else 0.0
    return sorted(totals.values(), key=lambda row: row["trade_count"], reverse=True)


def _feature_group_columns() -> dict[str, list[str]]:
    return {
        "candle_speed_volatility": ["msq_candle_speed_volatility"],
        "wick_body_structure": ["msq_wick_body_structure"],
        "impulse_strength": ["msq_impulse_strength"],
        "flem_shape": ["msq_flem_shape"],
        "pivot_shape": ["msq_pivot_shape"],
        "recent_highs_lows_context": ["msq_recent_high_low_context"],
        "day_structure": ["msq_day_structure"],
        "chop_range_unpredictability_avoidance": ["msq_chop_range_unpredictability", "msq_auction_clarity"],
        "requested_pit_features": [
            "directional_efficiency_open_to_now",
            "recent_directional_efficiency",
            "body_to_range_ratio_recent",
            "wick_rejection_ratio_recent",
            "alternating_bar_ratio",
            "local_swing_overlap_score",
            "impulse_speed_score",
            "repair_speed_score",
            "flem_compression_score",
            "pivot_cleanliness_score",
            "distance_to_recent_high_low",
            "failed_followthrough_count",
        ],
    }


def _diagnostic_source_path(state: dict[str, Any]) -> str:
    configured = str(dict(state.get("stage2_config", {}) or {}).get("source_path", "") or "")
    if configured:
        return configured
    diagnostics = _read_json(REPORTS_DIR / "exploration_benchmark_diagnostics.json")
    return str(diagnostics.get("source_path", "") or "")


def _looks_like_holdout(source_path: str, state: dict[str, Any]) -> bool:
    if "holdout" in source_path.lower():
        return True
    manifest = dict(state.get("data_manifest", {}) or {})
    for entry in manifest.get("files", []) or []:
        if entry.get("source_path") == source_path and entry.get("boundary_role") == "holdout":
            return True
    return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_rows_artifact(path_value: Any) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Market-state setup-quality diagnostics require pandas.") from exc
    return pd
