from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from trading_ml.engineer_adapter import compute_engineer_features, engineer_feature_snapshot, load_engineer_feature_config
from trading_ml.stage2_regime_features import build_regime_features
from trading_ml.stage2_bnr import CandidateSetup


@dataclass(slots=True)
class FeatureAudit:
    candidate_id: str
    status: str
    latest_feature_timestamp: str | None
    feature_cutoff_time: str
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_feature_matrix(bars: Any, candidates: list[CandidateSetup]) -> tuple[Any, list[FeatureAudit]]:
    pd = _require_pandas()
    engineer_config = load_engineer_feature_config()
    engineer_frame = None
    if engineer_config.get("backend", "hybrid") in {"hybrid", "ml4t_engineer"}:
        engineer_frame = compute_engineer_features(bars, features=list(engineer_config.get("features", [])))
    rows: list[dict[str, Any]] = []
    audits: list[FeatureAudit] = []
    for candidate in candidates:
        cutoff = pd.Timestamp(candidate.feature_cutoff_time)
        history = bars[bars.index < cutoff]
        latest = history.index.max() if not history.empty else None
        issues: list[str] = []
        if latest is not None and latest >= cutoff:
            issues.append("feature_timestamp_after_cutoff")
        session = history[history.index.date == pd.Timestamp(candidate.session_date).date()]
        prior = history[history.index < session.index.min()] if not session.empty else history.iloc[:0]
        trigger_start = pd.Timestamp(candidate.trigger_time)
        pre_trigger = session[session.index <= trigger_start]
        break_stats = _break_context(session, candidate)
        break_lab = _break_quality_lab(session, pre_trigger, candidate, break_stats)
        reclaim_lab = _reclaim_microstructure_lab(session, pre_trigger, candidate, break_stats)
        row = {
            "candidate_id": candidate.candidate_id,
            "session_date": candidate.session_date,
            "direction_long": 1 if candidate.direction == "long" else 0,
            "zone_width": candidate.zone.width,
            "zone_width_bps": candidate.zone.width_bps,
            "zone_midpoint": candidate.zone.midpoint,
            "entry_distance_from_midpoint": candidate.entry_reference_price - candidate.zone.midpoint,
            "entry_distance_from_zone_high": candidate.entry_reference_price - candidate.zone.high,
            "entry_distance_from_zone_low": candidate.entry_reference_price - candidate.zone.low,
            "trigger_seconds_after_open": _seconds_after_open(trigger_start),
            "pre_trigger_return": _return_from_first_to_last(pre_trigger),
            "pre_trigger_range": _range(pre_trigger),
            "pre_trigger_volume": float(pre_trigger["volume"].sum()) if not pre_trigger.empty else 0.0,
            "prior_close_gap": _prior_close_gap(prior, session),
            "first_break_wick_only": break_stats["first_break_wick_only"],
            "first_break_close_confirmed": break_stats["first_break_close_confirmed"],
            "first_break_wick_excess_points": break_stats["first_break_wick_excess_points"],
            "first_break_close_excess_points": break_stats["first_break_close_excess_points"],
            "pivot_duration_bars": break_stats["pivot_duration_bars"],
            "continuation_delay_bars": break_stats["continuation_delay_bars"],
            "reentry_count": break_stats["reentry_count"],
            "reclaim_count": break_stats["reclaim_count"],
            "deepest_zone_retrace_fraction": break_stats["deepest_zone_retrace_fraction"],
            "continuation_displacement_ratio": break_stats["continuation_displacement_ratio"],
            "post_reclaim_close_strength": break_stats["post_reclaim_close_strength"],
            "opposite_boundary_close_violation": break_stats["opposite_boundary_close_violation"],
        }
        row.update(break_lab)
        row.update(reclaim_lab)
        row.update(build_regime_features(session, pre_trigger, candidate))
        if engineer_config.get("backend", "hybrid") != "bnr_only":
            row.update(engineer_feature_snapshot(engineer_frame, cutoff))
        rows.append(row)
        audits.append(
            FeatureAudit(
                candidate_id=candidate.candidate_id,
                status="pass" if not issues else "fail",
                latest_feature_timestamp=latest.isoformat() if latest is not None else None,
                feature_cutoff_time=candidate.feature_cutoff_time,
                issues=issues,
            )
        )
    return pd.DataFrame(rows), audits


def _seconds_after_open(ts: Any) -> int:
    return int((ts.hour * 3600 + ts.minute * 60 + ts.second) - (9 * 3600 + 30 * 60))


def _return_from_first_to_last(df: Any) -> float:
    if df.empty:
        return 0.0
    first = float(df.iloc[0]["open"])
    last = float(df.iloc[-1]["close"])
    return (last - first) / first if first else 0.0


def _range(df: Any) -> float:
    if df.empty:
        return 0.0
    return float(df["high"].max() - df["low"].min())


def _prior_close_gap(prior: Any, session: Any) -> float:
    if prior.empty or session.empty:
        return 0.0
    prior_close = float(prior.iloc[-1]["close"])
    session_open = float(session.iloc[0]["open"])
    return (session_open - prior_close) / prior_close if prior_close else 0.0


def _break_context(session: Any, candidate: CandidateSetup) -> dict[str, float]:
    default = {
        "first_break_wick_only": 0.0,
        "first_break_close_confirmed": 0.0,
        "first_break_wick_excess_points": 0.0,
        "first_break_close_excess_points": 0.0,
        "pivot_duration_bars": 0.0,
        "continuation_delay_bars": 0.0,
        "reentry_count": 0.0,
        "reclaim_count": 0.0,
        "deepest_zone_retrace_fraction": 0.0,
        "continuation_displacement_ratio": 0.0,
        "post_reclaim_close_strength": 0.0,
        "opposite_boundary_close_violation": 0.0,
    }
    trace = candidate.trace or {}
    trace_keys = set(default)
    if trace_keys.issubset(trace):
        return {key: float(trace.get(key, default[key])) for key in default}

    pd = _require_pandas()
    trigger_time = pd.Timestamp(candidate.trigger_time)
    pre_trigger = session[session.index <= trigger_time]
    first_break = _find_first_break(pre_trigger, candidate)
    if first_break is None:
        return default

    break_ts, break_row = first_break
    post_break = pre_trigger[pre_trigger.index >= break_ts]
    after_break = pre_trigger[pre_trigger.index > break_ts]
    zone_width = candidate.zone.width if candidate.zone.width else 1.0

    if candidate.direction == "long":
        wick_excess = max(float(break_row["high"]) - candidate.zone.high, 0.0)
        close_excess = max(float(break_row["close"]) - candidate.zone.high, 0.0)
        wick_only = 1.0 if wick_excess > 0.0 and close_excess <= 0.0 else 0.0
        close_confirmed = 1.0 if close_excess > 0.0 else 0.0
        reentry_mask = after_break["low"] <= candidate.zone.high
        reclaim_mask = (after_break["low"] <= candidate.zone.high) & (after_break["close"] > candidate.zone.high)
        deepest_retrace = (
            max(candidate.zone.high - float(after_break["low"].min()), 0.0) / zone_width if not after_break.empty else 0.0
        )
        opposite_violation = 1.0 if (after_break["close"] < candidate.zone.low).any() else 0.0
        post_reclaim_close_strength = (
            max(float(after_break["close"].max()) - candidate.zone.high, 0.0) / zone_width if not after_break.empty else 0.0
        )
    else:
        wick_excess = max(candidate.zone.low - float(break_row["low"]), 0.0)
        close_excess = max(candidate.zone.low - float(break_row["close"]), 0.0)
        wick_only = 1.0 if wick_excess > 0.0 and close_excess <= 0.0 else 0.0
        close_confirmed = 1.0 if close_excess > 0.0 else 0.0
        reentry_mask = after_break["high"] >= candidate.zone.low
        reclaim_mask = (after_break["high"] >= candidate.zone.low) & (after_break["close"] < candidate.zone.low)
        deepest_retrace = (
            max(float(after_break["high"].max()) - candidate.zone.low, 0.0) / zone_width if not after_break.empty else 0.0
        )
        opposite_violation = 1.0 if (after_break["close"] > candidate.zone.high).any() else 0.0
        post_reclaim_close_strength = (
            max(candidate.zone.low - float(after_break["close"].min()), 0.0) / zone_width if not after_break.empty else 0.0
        )

    pivot_duration = max(len(post_break) - 1, 0)
    continuation_delay = max(len(pre_trigger[pre_trigger.index < trigger_time]) - len(pre_trigger[pre_trigger.index < break_ts]), 0)
    continuation_displacement_ratio = abs(candidate.entry_reference_price - float(break_row["close"])) / zone_width

    default.update(
        {
            "first_break_wick_only": wick_only,
            "first_break_close_confirmed": close_confirmed,
            "first_break_wick_excess_points": wick_excess,
            "first_break_close_excess_points": close_excess,
            "pivot_duration_bars": float(pivot_duration),
            "continuation_delay_bars": float(continuation_delay),
            "reentry_count": float(int(reentry_mask.sum())) if not after_break.empty else 0.0,
            "reclaim_count": float(int(reclaim_mask.sum())) if not after_break.empty else 0.0,
            "deepest_zone_retrace_fraction": deepest_retrace,
            "continuation_displacement_ratio": continuation_displacement_ratio,
            "post_reclaim_close_strength": post_reclaim_close_strength,
            "opposite_boundary_close_violation": opposite_violation,
        }
    )
    return default


def _break_quality_lab(session: Any, pre_trigger: Any, candidate: CandidateSetup, break_stats: dict[str, float]) -> dict[str, float]:
    pd = _require_pandas()
    default = {
        "break_close_distance_to_zone": 0.0,
        "break_body_fraction": 0.0,
        "break_speed_bars": 0.0,
        "break_volume_surge": 0.0,
        "break_range_expansion": 0.0,
        "break_efficiency_ratio": 0.0,
    }
    first_break = _find_first_break(pre_trigger, candidate)
    if first_break is None:
        return default

    break_ts, break_row = first_break
    zone_width = max(candidate.zone.width, 0.25)
    break_open = float(break_row["open"])
    break_close = float(break_row["close"])
    break_high = float(break_row["high"])
    break_low = float(break_row["low"])
    break_range = max(break_high - break_low, 1e-9)
    break_body = abs(break_close - break_open)
    pre_break = session[session.index < break_ts]
    recent = pre_break.tail(6)
    recent_range = _range(recent) / max(len(recent), 1) if not recent.empty else 0.0
    recent_vol_mean = float(recent["volume"].mean()) if not recent.empty else 0.0
    session_open = pd.Timestamp.combine(pd.Timestamp(candidate.session_date).date(), pd.Timestamp("09:31:00").time()).tz_localize(session.index.tz)
    speed_bars = float(max(len(session[(session.index >= session_open) & (session.index <= break_ts)]) - 1, 0))

    if candidate.direction == "long":
        close_distance = (break_close - candidate.zone.high) / zone_width
        directional_move = break_close - candidate.zone.high
    else:
        close_distance = (candidate.zone.low - break_close) / zone_width
        directional_move = candidate.zone.low - break_close

    return {
        "break_close_distance_to_zone": float(max(close_distance, 0.0)),
        "break_body_fraction": float(break_body / break_range),
        "break_speed_bars": speed_bars,
        "break_volume_surge": float(float(break_row["volume"]) / recent_vol_mean) if recent_vol_mean > 0 else 0.0,
        "break_range_expansion": float((break_range / recent_range) if recent_range > 0 else 0.0),
        "break_efficiency_ratio": float(max(directional_move, 0.0) / break_range),
    }


def _reclaim_microstructure_lab(session: Any, pre_trigger: Any, candidate: CandidateSetup, break_stats: dict[str, float]) -> dict[str, float]:
    pd = _require_pandas()
    default = {
        "pivot_symmetry": 0.0,
        "pivot_overlap_ratio": 0.0,
        "reclaim_latency_bars": 0.0,
        "reclaim_close_location": 0.0,
        "reclaim_body_strength": 0.0,
        "reclaim_failure_count": 0.0,
    }
    if pre_trigger.empty:
        return default

    break_ts = pd.Timestamp(candidate.break_time)
    trigger_ts = pd.Timestamp(candidate.trigger_time)
    pivot_window = pre_trigger[(pre_trigger.index > break_ts) & (pre_trigger.index <= trigger_ts)]
    if pivot_window.empty:
        return default

    zone_width = max(candidate.zone.width, 0.25)
    highs = pivot_window["high"].astype(float)
    lows = pivot_window["low"].astype(float)
    closes = pivot_window["close"].astype(float)
    opens = pivot_window["open"].astype(float)
    overlap_count = 0
    previous_low = None
    previous_high = None
    reclaim_failures = 0
    reclaim_bar = pivot_window.iloc[-1]

    for _, row in pivot_window.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        if previous_low is not None and high >= previous_low and low <= previous_high:
            overlap_count += 1
        previous_low = low
        previous_high = high
        if candidate.direction == "long" and close < candidate.zone.high:
            reclaim_failures += 1
        if candidate.direction == "short" and close > candidate.zone.low:
            reclaim_failures += 1

    pivot_range = float(highs.max() - lows.min())
    pivot_symmetry = abs(candidate.entry_reference_price - candidate.pivot_price) / zone_width
    latency_bars = float(max(len(pivot_window) - 1, 0))
    reclaim_range = max(float(reclaim_bar["high"]) - float(reclaim_bar["low"]), 1e-9)
    reclaim_body = abs(float(reclaim_bar["close"]) - float(reclaim_bar["open"]))

    if candidate.direction == "long":
        close_location = (float(reclaim_bar["close"]) - float(reclaim_bar["low"])) / reclaim_range
    else:
        close_location = (float(reclaim_bar["high"]) - float(reclaim_bar["close"])) / reclaim_range

    return {
        "pivot_symmetry": float(pivot_symmetry),
        "pivot_overlap_ratio": float(overlap_count / max(len(pivot_window), 1)),
        "reclaim_latency_bars": latency_bars,
        "reclaim_close_location": float(close_location),
        "reclaim_body_strength": float(reclaim_body / reclaim_range),
        "reclaim_failure_count": float(reclaim_failures),
    }


def _find_first_break(pre_trigger: Any, candidate: CandidateSetup) -> tuple[Any, Any] | None:
    if candidate.break_time:
        break_ts = _require_pandas().Timestamp(candidate.break_time)
        if break_ts in pre_trigger.index:
            return break_ts, pre_trigger.loc[break_ts]
    for ts, row in pre_trigger.iterrows():
        if candidate.direction == "long" and float(row["high"]) > candidate.zone.high:
            return ts, row
        if candidate.direction == "short" and float(row["low"]) < candidate.zone.low:
            return ts, row
    return None


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Stage 2 requires pandas.") from exc
    return pd
