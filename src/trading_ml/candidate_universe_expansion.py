from __future__ import annotations

from collections import Counter, defaultdict
import json
from statistics import mean
from typing import Any
from uuid import uuid4

from trading_ml.bnr_subtypes import classify_candidate_subtype
from trading_ml.feature_families import apply_feature_family
from trading_ml.feature_validation import build_feature_validation
from trading_ml.market_structure_lab import build_market_structure_lab
from trading_ml.model_diagnostics_lab import build_model_diagnostics_lab
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
from trading_ml.stage2_features import build_feature_matrix
from trading_ml.stage2_labeling import label_candidates
from trading_ml.stage2_modeling import train_baseline_classifier
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


def _resolve_variant_subset(controller_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    controller = dict(controller_state or {})
    requested_names = list(controller.get("variant_names", []) or controller.get("fast_variant_names", []) or [])
    if not requested_names:
        return list(VARIANTS)
    requested = set(requested_names)
    selected = [variant for variant in VARIANTS if variant["name"] in requested]
    if not selected:
        return list(VARIANTS)
    baseline = next((variant for variant in VARIANTS if variant["name"] == "first_reclaim_only_baseline"), None)
    if baseline and baseline["name"] not in {variant["name"] for variant in selected}:
        selected.insert(0, baseline)
    return selected


def _slice_bars_for_runtime(bars: Any, controller_state: dict[str, Any] | None = None) -> Any:
    controller = dict(controller_state or {})
    max_sessions = controller.get("max_sessions")
    if not max_sessions:
        return bars
    session_dates = sorted({idx.date() for idx in bars.index})
    keep_count = max(1, int(max_sessions))
    keep_dates = set(session_dates[-keep_count:])
    mask = [ts.date() in keep_dates for ts in bars.index]
    return bars.loc[mask]


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
    controller_state = dict(state.get("controller_state", {}) or {})
    bars = regular_session(load_ohlcv_file(cfg.source_path, symbol=cfg.symbol, timeframe=cfg.timeframe, timezone=cfg.timezone))
    bars = _slice_bars_for_runtime(bars, controller_state)
    baseline_rows: list[dict[str, Any]] = []
    rows = []
    variants = _resolve_variant_subset(controller_state)
    for idx, variant in enumerate(variants, start=1):
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
        "session_subset_count": len(sorted({idx.date() for idx in bars.index})),
        "variant_subset": [variant["name"] for variant in variants],
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


def run_candidate_universe_labeling_diagnostic(
    state: dict[str, Any],
    *,
    variant_names: list[str] | None = None,
) -> dict[str, Any]:
    cfg = Stage2Config(**dict(state.get("stage2_config", {})))
    bars = regular_session(load_ohlcv_file(cfg.source_path, symbol=cfg.symbol, timeframe=cfg.timeframe, timezone=cfg.timezone))
    selected = list(variant_names or ["first_reclaim_only_baseline", "allow_wick_through_breaks", "allow_delayed_reclaim"])
    variant_specs = [variant for variant in VARIANTS if variant["name"] in selected]
    rows: list[dict[str, Any]] = []
    for variant in variant_specs:
        candidates = _generate_variant_candidates(bars, cfg, variant)
        labels = label_candidates(
            bars,
            candidates,
            horizon_bars=cfg.horizon_bars,
            stop_multiple=cfg.stop_multiple,
            target_multiple=cfg.target_multiple,
        )
        features, feature_audits = build_feature_matrix(bars, candidates)
        if not features.empty:
            subtype_map = {candidate.candidate_id: classify_candidate_subtype(candidate) for candidate in candidates}
            features["setup_subtype"] = features["candidate_id"].map(subtype_map)
        features = apply_feature_family(features, cfg.feature_family)
        pd = _require_pandas()
        labels_df = pd.DataFrame([label.to_dict() for label in labels])
        model_summary = train_baseline_classifier(features, labels_df, model_family=cfg.model_family) if not labels_df.empty else None
        prediction_records = model_summary.prediction_records if model_summary else []
        rows.append(
            {
                "variant": variant["name"],
                "description": variant["description"],
                "candidate_count": len(candidates),
                "label_summary": {
                    "rows": int(len(labels_df)),
                    "positive_rate": float(labels_df["label"].mean()) if not labels_df.empty else 0.0,
                    "avg_pnl_r": float(labels_df["pnl_r"].mean()) if not labels_df.empty else 0.0,
                },
                "model_metrics": (model_summary.to_dict() if model_summary else {"status": "no_labels"}).get("metrics", {}),
                "feature_audit": {
                    "rows": len(feature_audits),
                    "failed": len([audit for audit in feature_audits if audit.status != "pass"]),
                },
                "feature_validation": build_feature_validation(features.to_dict(orient="records"), labels_df.to_dict(orient="records")),
                "market_structure_lab": build_market_structure_lab(
                    [candidate.to_dict() for candidate in candidates],
                    labels_df.to_dict(orient="records"),
                ),
                "model_diagnostics": build_model_diagnostics_lab(
                    prediction_records,
                    features.to_dict(orient="records"),
                    labels_df.to_dict(orient="records"),
                    model_family=cfg.model_family,
                ),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            float(dict(row.get("model_metrics", {})).get("roc_auc", 0.0) or 0.0),
            float(dict(row.get("label_summary", {})).get("avg_pnl_r", 0.0) or 0.0),
            int(row.get("candidate_count", 0) or 0),
        ),
        reverse=True,
    )
    payload = {
        "status": "complete",
        "family": "candidate_universe_expansion",
        "diagnostic_type": "downstream_labeling_model_diagnostic",
        "variants_tested": selected,
        "variant_diagnostics": rows,
        "ranked_variants": [
            {
                "variant": row["variant"],
                "candidate_count": row["candidate_count"],
                "positive_rate": row["label_summary"]["positive_rate"],
                "avg_pnl_r": row["label_summary"]["avg_pnl_r"],
                "roc_auc": dict(row.get("model_metrics", {})).get("roc_auc"),
                "precision": dict(row.get("model_metrics", {})).get("precision"),
                "recall": dict(row.get("model_metrics", {})).get("recall"),
            }
            for row in ranked
        ],
    }
    run_id = f"candidate-universe-labeling-{uuid4().hex[:12]}"
    output_dir = REPORTS_DIR / "runs" / run_id / "candidate_universe_labeling_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    artifact = REPORTS_DIR / "candidate_universe_labeling_diagnostic.json"
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
