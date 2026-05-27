from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Literal


Direction = Literal["long", "short"]


@dataclass(slots=True)
class BNRZone:
    symbol: str
    session_date: str
    zone_start: str
    zone_end: str
    decision_available_at: str
    high: float
    low: float
    midpoint: float
    width: float
    width_bps: float
    source_timeframe: str
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateSetup:
    candidate_id: str
    symbol: str
    session_date: str
    zone: BNRZone
    break_time: str
    break_decision_time: str
    trigger_time: str
    decision_time: str
    direction: Direction
    setup_type: str
    entry_reference_price: float
    invalidation_reference_price: float
    pivot_price: float
    pivot_time: str
    flem_price: float
    flem_time: str
    reentry_count: int
    reclaim_count: int
    feature_cutoff_time: str
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["zone"] = self.zone.to_dict()
        return data


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Stage 2 requires pandas. Install with `python3 -m pip install pandas pyarrow`."
        ) from exc
    return pd


def calculate_bnr_zones(
    bars: Any,
    *,
    symbol: str = "MNQ",
    timeframe: str = "30s",
    zone_start: str = "09:30:00",
    zone_end: str = "09:30:59",
    decision_time: str = "09:31:00",
) -> list[BNRZone]:
    pd = require_pandas()
    zones: list[BNRZone] = []
    for session_date in sorted({idx.date() for idx in bars.index}):
        session = bars[bars.index.date == session_date]
        zone_bars = session.between_time(zone_start, zone_end, inclusive="both")
        if zone_bars.empty:
            continue
        high = float(zone_bars["high"].max())
        low = float(zone_bars["low"].min())
        midpoint = (high + low) / 2.0
        width = high - low
        decision_at = pd.Timestamp.combine(
            session_date, pd.Timestamp(decision_time).time()
        ).tz_localize(bars.index.tz)
        flags: list[str] = []
        if width <= 0:
            flags.append("non_positive_zone_width")
        if len(zone_bars) < (2 if timeframe == "30s" else 1):
            flags.append("partial_opening_zone")
        zones.append(
            BNRZone(
                symbol=symbol,
                session_date=session_date.isoformat(),
                zone_start=zone_bars.index.min().isoformat(),
                zone_end=zone_bars.index.max().isoformat(),
                decision_available_at=decision_at.isoformat(),
                high=high,
                low=low,
                midpoint=midpoint,
                width=width,
                width_bps=(width / midpoint) * 10_000 if midpoint else 0.0,
                source_timeframe=timeframe,
                quality_flags=flags,
            )
        )
    return zones


def _bar_completion_time(index: Any, timeframe: str) -> Any:
    seconds = 30 if timeframe == "30s" else 60
    return index + timedelta(seconds=seconds)


def _is_opposite_pullback_bar(row: Any, direction: Direction, zone: BNRZone) -> bool:
    open_price = float(row["open"])
    close_price = float(row["close"])
    in_zone = float(row["low"]) <= zone.high and float(row["high"]) >= zone.low
    if not in_zone:
        return False
    if direction == "long":
        return close_price < open_price
    return close_price > open_price


def generate_breakout_candidates(
    bars: Any,
    zones: list[BNRZone],
    *,
    timeframe: str = "30s",
    earliest_trigger_time: str = "09:32:00",
    latest_trigger_time: str = "11:00:00",
    break_buffer_points: float = 0.0,
    max_candidates_per_direction: int = 0,
    candidate_engine: str = "canonical",
) -> list[CandidateSetup]:
    if candidate_engine == "event_driven_v1":
        return _generate_event_driven_breakout_candidates(
            bars,
            zones,
            timeframe=timeframe,
            earliest_trigger_time=earliest_trigger_time,
            latest_trigger_time=latest_trigger_time,
            break_buffer_points=break_buffer_points,
            max_candidates_per_direction=max_candidates_per_direction,
        )

    pd = require_pandas()
    candidates: list[CandidateSetup] = []
    for zone in zones:
        session_date = pd.Timestamp(zone.session_date).date()
        day_bars = bars[bars.index.date == session_date].sort_index()
        if day_bars.empty:
            continue
        one_minute = _build_one_minute_bars(day_bars)
        counts = {"long": 0, "short": 0}
        for break_ts, break_row in one_minute.iterrows():
            break_close_time = break_ts + timedelta(minutes=1)
            if break_close_time < pd.Timestamp(zone.decision_available_at):
                continue
            if break_close_time.time() > pd.Timestamp(latest_trigger_time).time():
                continue

            if (
                not _direction_limit_reached(
                    counts["long"], max_candidates_per_direction
                )
                and float(break_row["close"]) > zone.high + break_buffer_points
            ):
                candidate = _candidate_from_break_state(
                    zone=zone,
                    day_bars=day_bars,
                    one_minute=one_minute,
                    break_ts=break_ts,
                    break_row=break_row,
                    direction="long",
                    timeframe=timeframe,
                    earliest_trigger_time=earliest_trigger_time,
                    latest_trigger_time=latest_trigger_time,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    counts["long"] += 1

            if (
                not _direction_limit_reached(
                    counts["short"], max_candidates_per_direction
                )
                and float(break_row["close"]) < zone.low - break_buffer_points
            ):
                candidate = _candidate_from_break_state(
                    zone=zone,
                    day_bars=day_bars,
                    one_minute=one_minute,
                    break_ts=break_ts,
                    break_row=break_row,
                    direction="short",
                    timeframe=timeframe,
                    earliest_trigger_time=earliest_trigger_time,
                    latest_trigger_time=latest_trigger_time,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    counts["short"] += 1

            if _all_direction_limits_reached(counts, max_candidates_per_direction):
                break
    return candidates


def _generate_event_driven_breakout_candidates(
    bars: Any,
    zones: list[BNRZone],
    *,
    timeframe: str,
    earliest_trigger_time: str,
    latest_trigger_time: str,
    break_buffer_points: float,
    max_candidates_per_direction: int,
) -> list[CandidateSetup]:
    pd = require_pandas()
    candidates: list[CandidateSetup] = []
    for zone in zones:
        session_date = pd.Timestamp(zone.session_date).date()
        day_bars = bars[bars.index.date == session_date].sort_index()
        if day_bars.empty:
            continue

        one_minute = _build_one_minute_bars(day_bars)
        earliest_entry_dt = pd.Timestamp.combine(
            session_date, pd.Timestamp(earliest_trigger_time).time()
        ).tz_localize(day_bars.index.tz)
        latest_entry_dt = pd.Timestamp.combine(
            session_date, pd.Timestamp(latest_trigger_time).time()
        ).tz_localize(day_bars.index.tz)
        events = _build_event_stream(day_bars, one_minute)
        counts = {"long": 0, "short": 0}
        break_sequence = 0
        attempt_sequence = {"long": 0, "short": 0}

        direction: Direction | None = None
        break_ts: Any = None
        break_row: Any = None
        break_decision_time: Any = None
        flem_price: float | None = None
        flem_time: Any = None
        pivot_price: float | None = None
        pivot_time: Any = None
        reentry_count = 0
        reclaim_count = 0
        pullback_bar_seen = False
        last_pullback_bar_time: Any = None
        above_boundary_prev = False
        below_boundary_prev = False

        def reset_break_state() -> None:
            nonlocal direction
            nonlocal break_ts
            nonlocal break_row
            nonlocal break_decision_time
            nonlocal flem_price
            nonlocal flem_time
            reset_attempt_state()
            direction = None
            break_ts = None
            break_row = None
            break_decision_time = None
            flem_price = None
            flem_time = None

        def reset_attempt_state() -> None:
            nonlocal pivot_price
            nonlocal pivot_time
            nonlocal reentry_count
            nonlocal reclaim_count
            nonlocal pullback_bar_seen
            nonlocal last_pullback_bar_time
            nonlocal above_boundary_prev
            nonlocal below_boundary_prev
            pivot_price = None
            pivot_time = None
            reentry_count = 0
            reclaim_count = 0
            pullback_bar_seen = False
            last_pullback_bar_time = None
            above_boundary_prev = False
            below_boundary_prev = False

        def start_break(
            active_direction: Direction, ts: Any, row: Any, event_time: Any
        ) -> None:
            nonlocal direction
            nonlocal break_ts
            nonlocal break_row
            nonlocal break_decision_time
            nonlocal flem_price
            nonlocal flem_time
            nonlocal above_boundary_prev
            nonlocal below_boundary_prev
            nonlocal break_sequence
            reset_break_state()
            direction = active_direction
            break_ts = ts
            break_row = row
            break_decision_time = event_time
            break_sequence += 1
            flem_price = float(
                row["high"] if active_direction == "long" else row["low"]
            )
            flem_time = event_time
            above_boundary_prev = float(row["close"]) > zone.high
            below_boundary_prev = float(row["close"]) < zone.low

        for _, event in events.iterrows():
            event_time = event["event_time"]
            ts = event["timestamp"]
            if event_time < pd.Timestamp(zone.decision_available_at):
                continue

            if event["kind"] == "1m":
                close = float(event["close"])
                if close > zone.high + break_buffer_points:
                    if direction != "long" and not _direction_limit_reached(
                        counts["long"], max_candidates_per_direction
                    ):
                        start_break("long", ts, event, event_time)
                    continue
                if close < zone.low - break_buffer_points:
                    if direction != "short" and not _direction_limit_reached(
                        counts["short"], max_candidates_per_direction
                    ):
                        start_break("short", ts, event, event_time)
                    continue

            if (
                direction is None
                or break_decision_time is None
                or event["kind"] != timeframe
                or event_time <= break_decision_time
            ):
                continue

            if event_time > latest_entry_dt:
                reset_break_state()
                continue

            close = float(event["close"])
            high = float(event["high"])
            low = float(event["low"])

            if direction == "long" and close < zone.low:
                reset_break_state()
                continue
            if direction == "short" and close > zone.high:
                reset_break_state()
                continue

            if direction == "long":
                if flem_price is None or high >= flem_price:
                    flem_price = high
                    flem_time = event_time
            else:
                if flem_price is None or low <= flem_price:
                    flem_price = low
                    flem_time = event_time

            is_pullback_bar = _is_opposite_pullback_bar(event, direction, zone)
            if is_pullback_bar:
                pullback_bar_seen = True
                reentry_count += 1
                last_pullback_bar_time = event_time

            if pullback_bar_seen:
                if direction == "long":
                    if pivot_price is None or low < pivot_price:
                        pivot_price = low
                        pivot_time = event_time
                else:
                    if pivot_price is None or high > pivot_price:
                        pivot_price = high
                        pivot_time = event_time

            if direction == "long":
                reclaimed = (
                    pullback_bar_seen
                    and last_pullback_bar_time is not None
                    and event_time > last_pullback_bar_time
                    and close > zone.high
                )
                if reclaimed and not above_boundary_prev:
                    reclaim_count += 1
                    above_boundary_prev = True
                    if (
                        event_time >= earliest_entry_dt
                        and pivot_price is not None
                        and not _direction_limit_reached(
                            counts["long"], max_candidates_per_direction
                        )
                    ):
                        attempt_sequence["long"] += 1
                        candidates.append(
                            _finalize_candidate(
                                zone=zone,
                                break_ts=break_ts,
                                break_row=break_row,
                                entry_ts=ts,
                                entry_row=event,
                                direction=direction,
                                timeframe=timeframe,
                                pivot_price=pivot_price,
                                pivot_time=pivot_time,
                                flem_price=float(flem_price),
                                flem_time=flem_time,
                                reentry_count=reentry_count,
                                reclaim_count=reclaim_count,
                                candidate_engine="event_driven_v1",
                                break_sequence=break_sequence,
                                attempt_sequence=attempt_sequence["long"],
                            )
                        )
                        counts["long"] += 1
                        reset_attempt_state()
                        above_boundary_prev = close > zone.high
                        continue
                else:
                    above_boundary_prev = close > zone.high
            else:
                reclaimed = (
                    pullback_bar_seen
                    and last_pullback_bar_time is not None
                    and event_time > last_pullback_bar_time
                    and close < zone.low
                )
                if reclaimed and not below_boundary_prev:
                    reclaim_count += 1
                    below_boundary_prev = True
                    if (
                        event_time >= earliest_entry_dt
                        and pivot_price is not None
                        and not _direction_limit_reached(
                            counts["short"], max_candidates_per_direction
                        )
                    ):
                        attempt_sequence["short"] += 1
                        candidates.append(
                            _finalize_candidate(
                                zone=zone,
                                break_ts=break_ts,
                                break_row=break_row,
                                entry_ts=ts,
                                entry_row=event,
                                direction=direction,
                                timeframe=timeframe,
                                pivot_price=pivot_price,
                                pivot_time=pivot_time,
                                flem_price=float(flem_price),
                                flem_time=flem_time,
                                reentry_count=reentry_count,
                                reclaim_count=reclaim_count,
                                candidate_engine="event_driven_v1",
                                break_sequence=break_sequence,
                                attempt_sequence=attempt_sequence["short"],
                            )
                        )
                        counts["short"] += 1
                        reset_attempt_state()
                        below_boundary_prev = close < zone.low
                        continue
                else:
                    below_boundary_prev = close < zone.low

            if _all_direction_limits_reached(counts, max_candidates_per_direction):
                break
    return candidates


def _direction_limit_reached(count: int, max_candidates_per_direction: int) -> bool:
    return int(max_candidates_per_direction) > 0 and count >= int(
        max_candidates_per_direction
    )


def _all_direction_limits_reached(
    counts: dict[str, int], max_candidates_per_direction: int
) -> bool:
    return int(max_candidates_per_direction) > 0 and all(
        value >= int(max_candidates_per_direction) for value in counts.values()
    )


def _build_one_minute_bars(day_bars: Any) -> Any:
    agg = (
        day_bars.resample("1min", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    return agg


def _build_event_stream(day_bars: Any, one_minute: Any) -> Any:
    pd = require_pandas()
    ev_1m = one_minute.copy()
    ev_1m["timestamp"] = ev_1m.index
    ev_1m["event_time"] = ev_1m.index + timedelta(minutes=1)
    ev_1m["kind"] = "1m"
    ev_30s = day_bars.copy()
    ev_30s["timestamp"] = ev_30s.index
    ev_30s["event_time"] = ev_30s.index + timedelta(seconds=30)
    ev_30s["kind"] = "30s"
    events = pd.concat(
        [
            ev_1m[
                [
                    "timestamp",
                    "event_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "kind",
                ]
            ],
            ev_30s[
                [
                    "timestamp",
                    "event_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "kind",
                ]
            ],
        ],
        ignore_index=True,
    )
    events["kind_order"] = events["kind"].map({"1m": 0, "30s": 1}).fillna(2)
    return events.sort_values(["event_time", "kind_order", "timestamp"]).reset_index(
        drop=True
    )


def _candidate_from_break_state(
    *,
    zone: BNRZone,
    day_bars: Any,
    one_minute: Any,
    break_ts: Any,
    break_row: Any,
    direction: Direction,
    timeframe: str,
    earliest_trigger_time: str,
    latest_trigger_time: str,
) -> CandidateSetup | None:
    pd = require_pandas()
    break_decision_time = break_ts + timedelta(minutes=1)
    earliest_entry_dt = pd.Timestamp.combine(
        pd.Timestamp(zone.session_date).date(),
        pd.Timestamp(earliest_trigger_time).time(),
    ).tz_localize(day_bars.index.tz)
    latest_entry_dt = pd.Timestamp.combine(
        pd.Timestamp(zone.session_date).date(), pd.Timestamp(latest_trigger_time).time()
    ).tz_localize(day_bars.index.tz)
    post_break_bars = day_bars[
        day_bars.index.map(lambda idx: _bar_completion_time(idx, timeframe))
        > break_decision_time
    ]
    if post_break_bars.empty:
        return None

    flem_price = float(break_row["high"] if direction == "long" else break_row["low"])
    flem_time = break_decision_time
    pivot_price: float | None = None
    pivot_time: Any = None
    reentry_count = 0
    reclaim_count = 0
    pullback_bar_seen = False
    pullback_pivot_locked = False
    last_pullback_bar_time: Any = None
    above_boundary_prev = float(break_row["close"]) > zone.high
    below_boundary_prev = float(break_row["close"]) < zone.low

    invalidation_minutes = one_minute[one_minute.index >= break_ts]

    for ts, row in post_break_bars.iterrows():
        bar_close_time = _bar_completion_time(ts, timeframe)
        if bar_close_time > latest_entry_dt:
            break

        if direction == "long":
            flem_candidate = float(row["high"])
            if flem_candidate >= flem_price:
                flem_price = flem_candidate
                flem_time = bar_close_time
        else:
            flem_candidate = float(row["low"])
            if flem_candidate <= flem_price:
                flem_price = flem_candidate
                flem_time = bar_close_time

        if _invalidated_before_entry(
            invalidation_minutes, zone, direction, bar_close_time
        ):
            return None

        is_pullback_bar = _is_opposite_pullback_bar(row, direction, zone)
        if is_pullback_bar:
            pullback_bar_seen = True
            reentry_count += 1
            last_pullback_bar_time = bar_close_time

        if pullback_bar_seen:
            if direction == "long":
                if pivot_price is None or float(row["low"]) < pivot_price:
                    pivot_price = float(row["low"])
                    pivot_time = bar_close_time
            else:
                if pivot_price is None or float(row["high"]) > pivot_price:
                    pivot_price = float(row["high"])
                    pivot_time = bar_close_time
            pullback_pivot_locked = pivot_price is not None

        if direction == "long":
            reclaimed = (
                pullback_bar_seen
                and pullback_pivot_locked
                and last_pullback_bar_time is not None
                and bar_close_time > last_pullback_bar_time
                and float(row["close"]) > zone.high
            )
            if reclaimed and not above_boundary_prev:
                reclaim_count += 1
            above_boundary_prev = float(row["close"]) > zone.high
            if (
                reclaimed
                and bar_close_time >= earliest_entry_dt
                and pivot_price is not None
            ):
                return _finalize_candidate(
                    zone=zone,
                    break_ts=break_ts,
                    break_row=break_row,
                    entry_ts=ts,
                    entry_row=row,
                    direction=direction,
                    timeframe=timeframe,
                    pivot_price=pivot_price,
                    pivot_time=pivot_time,
                    flem_price=flem_price,
                    flem_time=flem_time,
                    reentry_count=reentry_count,
                    reclaim_count=reclaim_count,
                )
        else:
            reclaimed = (
                pullback_bar_seen
                and pullback_pivot_locked
                and last_pullback_bar_time is not None
                and bar_close_time > last_pullback_bar_time
                and float(row["close"]) < zone.low
            )
            if reclaimed and not below_boundary_prev:
                reclaim_count += 1
            below_boundary_prev = float(row["close"]) < zone.low
            if (
                reclaimed
                and bar_close_time >= earliest_entry_dt
                and pivot_price is not None
            ):
                return _finalize_candidate(
                    zone=zone,
                    break_ts=break_ts,
                    break_row=break_row,
                    entry_ts=ts,
                    entry_row=row,
                    direction=direction,
                    timeframe=timeframe,
                    pivot_price=pivot_price,
                    pivot_time=pivot_time,
                    flem_price=flem_price,
                    flem_time=flem_time,
                    reentry_count=reentry_count,
                    reclaim_count=reclaim_count,
                )
    return None


def _invalidated_before_entry(
    invalidation_minutes: Any, zone: BNRZone, direction: Direction, bar_close_time: Any
) -> bool:
    eligible = invalidation_minutes[
        (invalidation_minutes.index + timedelta(minutes=1)) <= bar_close_time
    ]
    if eligible.empty:
        return False
    if direction == "long":
        return bool((eligible["close"] < zone.low).any())
    return bool((eligible["close"] > zone.high).any())


def _finalize_candidate(
    *,
    zone: BNRZone,
    break_ts: Any,
    break_row: Any,
    entry_ts: Any,
    entry_row: Any,
    direction: Direction,
    timeframe: str,
    pivot_price: float,
    pivot_time: Any,
    flem_price: float,
    flem_time: Any,
    reentry_count: int,
    reclaim_count: int,
    candidate_engine: str = "canonical",
    break_sequence: int = 1,
    attempt_sequence: int = 1,
) -> CandidateSetup:
    break_decision_time = break_ts + timedelta(minutes=1)
    decision_time = _bar_completion_time(entry_ts, timeframe)
    setup_type = f"break_reentry_reclaim_{direction}"
    candidate_id = (
        f"{zone.symbol}-{zone.session_date}-{setup_type}-{entry_ts.isoformat()}"
    )

    if direction == "long":
        first_break_wick_excess = max(float(break_row["high"]) - zone.high, 0.0)
        first_break_close_excess = max(float(break_row["close"]) - zone.high, 0.0)
        first_break_wick_only = (
            first_break_wick_excess > 0 and first_break_close_excess <= 0
        )
        deepest_zone_retrace_fraction = (
            max(zone.high - pivot_price, 0.0) / zone.width if zone.width else 0.0
        )
        continuation_displacement_ratio = (
            max(float(entry_row["close"]) - pivot_price, 0.0) / zone.width
            if zone.width
            else 0.0
        )
        post_reclaim_close_strength = (
            max(float(entry_row["close"]) - zone.high, 0.0) / zone.width
            if zone.width
            else 0.0
        )
    else:
        first_break_wick_excess = max(zone.low - float(break_row["low"]), 0.0)
        first_break_close_excess = max(zone.low - float(break_row["close"]), 0.0)
        first_break_wick_only = (
            first_break_wick_excess > 0 and first_break_close_excess <= 0
        )
        deepest_zone_retrace_fraction = (
            max(pivot_price - zone.low, 0.0) / zone.width if zone.width else 0.0
        )
        continuation_displacement_ratio = (
            max(pivot_price - float(entry_row["close"]), 0.0) / zone.width
            if zone.width
            else 0.0
        )
        post_reclaim_close_strength = (
            max(zone.low - float(entry_row["close"]), 0.0) / zone.width
            if zone.width
            else 0.0
        )

    return CandidateSetup(
        candidate_id=candidate_id,
        symbol=zone.symbol,
        session_date=zone.session_date,
        zone=zone,
        break_time=break_ts.isoformat(),
        break_decision_time=break_decision_time.isoformat(),
        trigger_time=entry_ts.isoformat(),
        decision_time=decision_time.isoformat(),
        direction=direction,
        setup_type=setup_type,
        entry_reference_price=float(entry_row["close"]),
        invalidation_reference_price=float(
            zone.low if direction == "long" else zone.high
        ),
        pivot_price=float(pivot_price),
        pivot_time=pivot_time.isoformat(),
        flem_price=float(flem_price),
        flem_time=flem_time.isoformat(),
        reentry_count=reentry_count,
        reclaim_count=reclaim_count,
        feature_cutoff_time=decision_time.isoformat(),
        trace={
            "break_bar_start": break_ts.isoformat(),
            "break_bar_close_time": break_decision_time.isoformat(),
            "break_close": float(break_row["close"]),
            "break_high": float(break_row["high"]),
            "break_low": float(break_row["low"]),
            "entry_bar_start": entry_ts.isoformat(),
            "entry_bar_close_time": decision_time.isoformat(),
            "entry_close": float(entry_row["close"]),
            "zone_high": zone.high,
            "zone_low": zone.low,
            "first_break_wick_only": float(first_break_wick_only),
            "first_break_close_confirmed": float(first_break_close_excess > 0),
            "first_break_wick_excess_points": float(first_break_wick_excess),
            "first_break_close_excess_points": float(first_break_close_excess),
            "pivot_duration_bars": float(max(reentry_count + reclaim_count - 1, 0)),
            "continuation_delay_bars": float(
                max(int((decision_time - break_decision_time).total_seconds() // 30), 0)
            ),
            "deepest_zone_retrace_fraction": float(deepest_zone_retrace_fraction),
            "continuation_displacement_ratio": float(continuation_displacement_ratio),
            "post_reclaim_close_strength": float(post_reclaim_close_strength),
            "opposite_boundary_close_violation": 0.0,
            "candidate_engine": candidate_engine,
            "break_sequence": float(break_sequence),
            "attempt_sequence": float(attempt_sequence),
        },
    )
