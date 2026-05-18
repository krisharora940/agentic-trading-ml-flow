from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from trading_ml.stage2_bnr import CandidateSetup


LabelValue = Literal[0, 1]


@dataclass(slots=True)
class TradeLabel:
    candidate_id: str
    label: LabelValue
    outcome: str
    entry_time: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: str | None
    exit_price: float | None
    bars_held: int
    mfe: float
    mae: float
    pnl_r: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label_candidates(
    bars: Any,
    candidates: list[CandidateSetup],
    *,
    horizon_bars: int = 20,
    stop_multiple: float = 1.0,
    target_multiple: float = 1.5,
) -> list[TradeLabel]:
    labels: list[TradeLabel] = []
    for candidate in candidates:
        labels.append(
            label_candidate(
                bars,
                candidate,
                horizon_bars=horizon_bars,
                stop_multiple=stop_multiple,
                target_multiple=target_multiple,
            )
        )
    return labels


def label_candidate(
    bars: Any,
    candidate: CandidateSetup,
    *,
    horizon_bars: int,
    stop_multiple: float,
    target_multiple: float,
) -> TradeLabel:
    pd = _require_pandas()
    decision_time = pd.Timestamp(candidate.decision_time)
    future = bars[bars.index >= decision_time].head(horizon_bars)
    entry = candidate.entry_reference_price
    # V1 target: classify whether the trade reaches +target_r before -stop_r within the fixed horizon.
    risk = max(candidate.zone.width * stop_multiple, 0.25)
    if candidate.direction == "long":
        stop = entry - risk
        target = entry + candidate.zone.width * target_multiple
    else:
        stop = entry + risk
        target = entry - candidate.zone.width * target_multiple

    mfe = 0.0
    mae = 0.0
    exit_time = None
    exit_price = None
    outcome = "timeout"
    label: LabelValue = 0
    bars_held = 0

    for i, (ts, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high"])
        low = float(row["low"])
        if candidate.direction == "long":
            mfe = max(mfe, high - entry)
            mae = min(mae, low - entry)
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            mfe = max(mfe, entry - low)
            mae = min(mae, entry - high)
            stop_hit = high >= stop
            target_hit = low <= target

        bars_held = i
        if stop_hit and target_hit:
            outcome = "ambiguous_stop_first"
            exit_time = ts.isoformat()
            exit_price = stop
            label = 0
            break
        if stop_hit:
            outcome = "stop"
            exit_time = ts.isoformat()
            exit_price = stop
            label = 0
            break
        if target_hit:
            outcome = "target"
            exit_time = ts.isoformat()
            exit_price = target
            label = 1
            break

    if exit_price is None and not future.empty:
        last_ts = future.index[-1]
        last_close = float(future.iloc[-1]["close"])
        exit_time = last_ts.isoformat()
        exit_price = last_close

    pnl = _directional_pnl(candidate, entry, exit_price if exit_price is not None else entry)
    pnl_r = pnl / risk if risk else 0.0
    return TradeLabel(
        candidate_id=candidate.candidate_id,
        label=label,
        outcome=outcome,
        entry_time=decision_time.isoformat(),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        exit_time=exit_time,
        exit_price=exit_price,
        bars_held=bars_held,
        mfe=mfe,
        mae=mae,
        pnl_r=pnl_r,
    )


def _directional_pnl(candidate: CandidateSetup, entry: float, exit_price: float) -> float:
    if candidate.direction == "long":
        return exit_price - entry
    return entry - exit_price


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Stage 2 requires pandas.") from exc
    return pd
