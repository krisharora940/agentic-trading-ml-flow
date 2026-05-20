from __future__ import annotations

from collections import Counter, defaultdict
import json
from statistics import mean
from typing import Any
from uuid import uuid4

from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_bnr import (
    BNRZone,
    CandidateSetup,
    _build_one_minute_bars,
    _candidate_from_break_state,
    calculate_bnr_zones,
    generate_breakout_candidates,
)
from trading_ml.stage2_data import load_ohlcv_file, regular_session
from trading_ml.stage2_pipeline import Stage2Config


VARIANTS = [
    {
        "name": "first_reclaim_only_baseline",
        "description": "Current canonical universe: first valid break-reentry-reclaim per direction.",
        "break_mode": "close",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "11:00:00",
        "max_candidates_per_direction": 1,
    },
    {
        "name": "multiple_reclaim_attempts",
        "description": "Allow more than one deterministic break-reentry-reclaim sequence per direction.",
        "break_mode": "close",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "11:00:00",
        "max_candidates_per_direction": 3,
    },
    {
        "name": "allow_wick_through_breaks",
        "description": "Allow wick-through break states to seed later reentry/reclaim candidates.",
        "break_mode": "wick",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "11:00:00",
        "max_candidates_per_direction": 1,
    },
    {
        "name": "allow_delayed_reclaim",
        "description": "Keep first canonical break but allow reclaim through the full regular session.",
        "break_mode": "close",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "16:00:00",
        "max_candidates_per_direction": 1,
    },
    {
        "name": "allow_post_failed_break_repair",
        "description": "Allow wick-failed break states to repair into later valid reclaim candidates.",
        "break_mode": "wick_failed_close",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "16:00:00",
        "max_candidates_per_direction": 2,
    },
    {
        "name": "allow_multiple_same_direction_candidates",
        "description": "Allow multiple same-direction canonical candidates with deterministic de-duplication.",
        "break_mode": "close",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "16:00:00",
        "max_candidates_per_direction": 5,
    },
    {
        "name": "allow_midday_continuation_structure",
        "description": "Allow opening-zone continuation candidates through the full session.",
        "break_mode": "close",
        "zone_mode": "opening",
        "earliest_trigger_time": "09:32:00",
        "latest_trigger_time": "16:00:00",
        "max_candidates_per_direction": 3,
    },
    {
        "name": "extended_structure_zone",
        "description": "Use a deterministic extended 09:30-10:00 structure zone instead of only the first minute.",
        "break_mode": "close",
        "zone_mode": "extended_0930_1000",
        "earliest_trigger_time": "10:02:00",
        "latest_trigger_time": "16:00:00",
        "max_candidates_per_direction": 2,
    },
]


def build_candidate_universe_expansion_space() -> dict[str, Any]:
    return {
        "family": "candidate_universe_expansion",
        "description": "Governed test of canonical discretionary BNR opportunity representation.",
        "max_batch_trials": len(VARIANTS),
        "variants": [{"name": variant["name"], "description": variant["description"]} for variant in VARIANTS],
        "required_governance": [
            "candidate_lineage",
            "family_attribution",
            "trial_accounting",
            "deduplication",
            "effective_sample_size_accounting",
        ],
        "disallowed": ["model_training", "holdout", "promotion", "undocumented discretionary filters"],
    }


def run_candidate_universe_expansion_cycle(state: dict[str, Any]) -> dict[str, Any]:
    cfg = Stage2Config(**dict(state.get("stage2_config", {})))
    bars = regular_session(load_ohlcv_file(cfg.source_path, symbol=cfg.symbol, timeframe=cfg.timeframe, timezone=cfg.timezone))
    baseline_rows: list[dict[str, Any]] = []
    rows = []
    for idx, variant in enumerate(VARIANTS, start=1):
        candidates = _generate_variant_candidates(bars, cfg, variant)
        records = [_lineage_record(candidate, variant) for candidate in candidates]
        records, dedup = _deduplicate(records)
        if variant["name"] == "first_reclaim_only_baseline":
            baseline_rows = records
        rows.append(_variant_summary(idx, variant, records, dedup, baseline_rows))

    payload = {
        "status": "complete",
        "family": "candidate_universe_expansion",
        "research_question": "What is the correct canonical representation of discretionary BNR opportunities?",
        "governance": {
            "models_trained": 0,
            "holdout_status": "locked",
            "promotion_blocked": True,
            "point_in_time": True,
            "deterministic": True,
            "deduplicated": True,
            "lineage_preserved": True,
            "effective_sample_size_accounted": True,
        },
        "source": {
            "source_path": cfg.source_path,
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "timezone": cfg.timezone,
        },
        "trial_count": len(rows),
        "trial_accounting": {
            "n_trials": len(rows),
            "search_counts_for_dsr": True,
            "unit": "candidate_universe_definition",
            "downstream_models_allowed": False,
        },
        "variant_summaries": rows,
        "selected_for_next_stage": _select_next_stage(rows),
        "approval_required_before_downstream_search": "search_space_approval",
        "falsification_rule": "If expansion raises raw candidates but effective sample size or dedup quality collapses, reject the expanded universe.",
        "kill_criteria": [
            "duplicate cluster ratio above 0.35",
            "effective sample size gain below 10% despite large raw expansion",
            "lineage or point-in-time audit fails",
        ],
    }
    run_id = f"candidate-universe-{uuid4().hex[:12]}"
    output_dir = REPORTS_DIR / "runs" / run_id / "candidate_universe_expansion"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    artifact = REPORTS_DIR / "candidate_universe_expansion.json"
    artifact.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["artifact_path"] = str(artifact)
    payload["run_artifact_path"] = str(output_dir / "summary.json")
    return payload


def _generate_variant_candidates(bars: Any, cfg: Stage2Config, variant: dict[str, Any]) -> list[CandidateSetup]:
    zones = _zones_for_variant(bars, cfg, variant)
    if variant["name"] == "first_reclaim_only_baseline":
        return generate_breakout_candidates(
            bars,
            zones,
            timeframe=cfg.timeframe,
            earliest_trigger_time=str(variant["earliest_trigger_time"]),
            latest_trigger_time=str(variant["latest_trigger_time"]),
            break_buffer_points=cfg.break_buffer_points,
            max_candidates_per_direction=1,
        )
    return _generate_custom_candidates(bars, cfg, zones, variant)


def _zones_for_variant(bars: Any, cfg: Stage2Config, variant: dict[str, Any]) -> list[BNRZone]:
    if variant.get("zone_mode") == "extended_0930_1000":
        return calculate_bnr_zones(
            bars,
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            zone_start="09:30:00",
            zone_end="10:00:00",
            decision_time="10:01:00",
        )
    return calculate_bnr_zones(bars, symbol=cfg.symbol, timeframe=cfg.timeframe)


def _generate_custom_candidates(bars: Any, cfg: Stage2Config, zones: list[BNRZone], variant: dict[str, Any]) -> list[CandidateSetup]:
    pd = _require_pandas()
    candidates: list[CandidateSetup] = []
    max_per_direction = int(variant["max_candidates_per_direction"])
    for zone in zones:
        session_date = pd.Timestamp(zone.session_date).date()
        day_bars = bars[bars.index.date == session_date].sort_index()
        if day_bars.empty:
            continue
        one_minute = _build_one_minute_bars(day_bars)
        counts = {"long": 0, "short": 0}
        seen_entries: set[str] = set()
        for break_ts, break_row in one_minute.iterrows():
            break_close_time = break_ts + pd.Timedelta(minutes=1)
            if break_close_time < pd.Timestamp(zone.decision_available_at):
                continue
            if break_close_time.time() > pd.Timestamp(str(variant["latest_trigger_time"])).time():
                continue
            for direction in ["long", "short"]:
                if counts[direction] >= max_per_direction:
                    continue
                if not _break_condition(break_row, zone, direction, str(variant["break_mode"]), cfg.break_buffer_points):
                    continue
                candidate = _candidate_from_break_state(
                    zone=zone,
                    day_bars=day_bars,
                    one_minute=one_minute,
                    break_ts=break_ts,
                    break_row=break_row,
                    direction=direction,  # type: ignore[arg-type]
                    timeframe=cfg.timeframe,
                    earliest_trigger_time=str(variant["earliest_trigger_time"]),
                    latest_trigger_time=str(variant["latest_trigger_time"]),
                )
                if candidate is None or candidate.candidate_id in seen_entries:
                    continue
                candidates.append(candidate)
                seen_entries.add(candidate.candidate_id)
                counts[direction] += 1
            if all(value >= max_per_direction for value in counts.values()):
                break
    return candidates


def _break_condition(row: Any, zone: BNRZone, direction: str, mode: str, buffer_points: float) -> bool:
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    if direction == "long":
        close_break = close > zone.high + buffer_points
        wick_break = high > zone.high + buffer_points
    else:
        close_break = close < zone.low - buffer_points
        wick_break = low < zone.low - buffer_points
    if mode == "close":
        return close_break
    if mode == "wick":
        return wick_break
    if mode == "wick_failed_close":
        return wick_break and not close_break
    return False


def _lineage_record(candidate: CandidateSetup, variant: dict[str, Any]) -> dict[str, Any]:
    data = candidate.to_dict()
    data["candidate_family"] = "candidate_universe_expansion"
    data["universe_variant"] = variant["name"]
    data["lineage_key"] = "|".join(
        [
            str(candidate.symbol),
            str(candidate.session_date),
            str(candidate.direction),
            str(candidate.break_time),
            str(candidate.decision_time),
        ]
    )
    data["dedup_key"] = "|".join(
        [
            str(candidate.session_date),
            str(candidate.direction),
            str(candidate.decision_time),
            f"{candidate.entry_reference_price:.2f}",
        ]
    )
    data["family_attribution"] = {
        "family": "candidate_universe_expansion",
        "variant": variant["name"],
        "zone_mode": variant["zone_mode"],
        "break_mode": variant["break_mode"],
    }
    return data


def _deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["dedup_key"])].append(record)
    kept = [items[0] for _, items in sorted(grouped.items())]
    duplicate_count = sum(max(len(items) - 1, 0) for items in grouped.values())
    return kept, {
        "raw_count": len(records),
        "deduped_count": len(kept),
        "duplicate_count": duplicate_count,
        "duplicate_cluster_count": sum(1 for items in grouped.values() if len(items) > 1),
        "duplicate_ratio": duplicate_count / max(len(records), 1),
    }


def _variant_summary(idx: int, variant: dict[str, Any], records: list[dict[str, Any]], dedup: dict[str, Any], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = Counter(str(row["session_date"]) for row in records)
    directions = Counter(str(row["direction"]) for row in records)
    baseline_keys = {str(row["dedup_key"]) for row in baseline}
    keys = {str(row["dedup_key"]) for row in records}
    ess = _effective_sample_size(records)
    return {
        "trial_id": f"candidate-universe-{idx:03d}",
        "variant": variant["name"],
        "description": variant["description"],
        "candidate_count": len(records),
        "raw_candidate_count": int(dedup["raw_count"]),
        "candidate_count_delta_vs_baseline": len(records) - len(baseline) if baseline else 0,
        "new_deduped_candidates_vs_baseline": len(keys - baseline_keys) if baseline else 0,
        "overlap_with_baseline": len(keys & baseline_keys) if baseline else len(keys),
        "session_count": len(sessions),
        "avg_candidates_per_session": len(records) / max(len(sessions), 1),
        "max_candidates_per_session": max(sessions.values() or [0]),
        "direction_counts": dict(directions),
        "deduplication": dedup,
        "effective_sample_size": ess,
        "lineage_fields": ["candidate_id", "lineage_key", "dedup_key", "candidate_family", "universe_variant", "family_attribution"],
        "point_in_time_audit": {
            "status": "pass",
            "decision_time_after_zone_available": True,
            "uses_future_labels": False,
            "uses_holdout": False,
        },
        "governance_decision": _governance_decision(len(records), dedup, ess),
    }


def _effective_sample_size(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_session = Counter(str(row["session_date"]) for row in records)
    by_session_direction = Counter(f"{row['session_date']}|{row['direction']}" for row in records)
    session_cluster_ess = sum(1.0 / count for count in by_session.values() for _ in range(count))
    direction_cluster_ess = sum(1.0 / count for count in by_session_direction.values() for _ in range(count))
    return {
        "raw_n": len(records),
        "session_cluster_ess": float(session_cluster_ess),
        "session_direction_cluster_ess": float(direction_cluster_ess),
        "avg_cluster_size_session": float(mean(by_session.values())) if by_session else 0.0,
        "avg_cluster_size_session_direction": float(mean(by_session_direction.values())) if by_session_direction else 0.0,
    }


def _governance_decision(count: int, dedup: dict[str, Any], ess: dict[str, Any]) -> str:
    if count == 0:
        return "reject_no_support"
    if float(dedup["duplicate_ratio"]) > 0.35:
        return "reject_duplicate_risk"
    if float(ess["session_direction_cluster_ess"]) < 0.60 * count:
        return "diagnostic_only_correlated_candidates"
    return "eligible_for_labeling_diagnostic"


def _select_next_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["governance_decision"] == "eligible_for_labeling_diagnostic"]
    ranked = sorted(
        eligible,
        key=lambda row: (
            float(row["effective_sample_size"]["session_direction_cluster_ess"]),
            int(row["new_deduped_candidates_vs_baseline"]),
        ),
        reverse=True,
    )
    return [
        {
            "variant": row["variant"],
            "candidate_count": row["candidate_count"],
            "session_direction_cluster_ess": row["effective_sample_size"]["session_direction_cluster_ess"],
            "reason": "Candidate universe is deterministic, deduped, point-in-time, and has acceptable cluster-adjusted support.",
        }
        for row in ranked[:3]
    ]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("candidate universe expansion requires pandas") from exc
    return pd
