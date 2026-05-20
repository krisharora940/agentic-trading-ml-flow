from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import uuid4

from trading_ml.exit_engine import (
    EXIT_VARIANTS,
    STRUCTURE_PARTIAL_REPLAY_VARIANTS,
    _attach_dsr_psr,
    _evaluate_exit_policy,
    _frozen_v1_entries,
)
from trading_ml.market_state_quality import _diagnostic_config, build_market_state_setup_quality_diagnostic
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_data import load_ohlcv_file, regular_session


EXIT_BEHAVIOR_FAMILIES = [
    "scratch_exits",
    "structure_trailing",
    "partial_profit_runner",
    "time_stop",
    "break_even_policy",
    "volatility_based_exit",
    "liquidity_rejection_exit",
]


def build_exit_behavior_research_space() -> dict[str, Any]:
    return {
        "family": "exit_behavior_research",
        "description": "Governed trade-path behavior research for frozen market_state_setup_quality_v1 entries.",
        "max_batch_trials": len(EXIT_VARIANTS),
        "stages": [
            "trade_path_diagnostics",
            "exit_behavior_archetypes",
            "candidate_exit_families",
            "bounded_replay_existing_entries",
            "shortlist_for_full_validation",
        ],
        "candidate_exit_families": EXIT_BEHAVIOR_FAMILIES,
        "bounded_replay_variants": list(EXIT_VARIANTS),
        "disallowed_knobs": ["entry_filter", "setup_filter", "model_training", "holdout", "path_specific_exit"],
    }


def run_exit_behavior_research_cycle(state: dict[str, Any]) -> dict[str, Any]:
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        return {"status": "pending", "reason": "diagnostic_unavailable", "diagnostic": diagnostic}

    config = _diagnostic_config(state, str(diagnostic["source_path"]))
    bars = regular_session(
        load_ohlcv_file(
            config.source_path,
            symbol=config.symbol,
            timeframe=config.timeframe,
            timezone=config.timezone,
        )
    )
    entries = _frozen_v1_entries(diagnostic)
    path_rows = [_trade_path_diagnostics(row, bars, config.horizon_bars) for row in entries]
    archetypes = _archetype_summary(path_rows)
    exit_families = _candidate_exit_families(archetypes)
    replay_rows = [_evaluate_exit_policy(name, entries, bars, config.horizon_bars) for name in EXIT_VARIANTS]
    _attach_dsr_psr(replay_rows)
    baseline = next((row for row in replay_rows if row["variant"] == "fixed_1_5r_tp_sl_baseline"), replay_rows[0])
    for row in replay_rows:
        row["exit_family"] = _variant_family(row["variant"])
        row["governance_decision"] = _replay_decision(row, baseline)
    ranked = sorted(
        replay_rows,
        key=lambda row: (
            row["governance_decision"] == "shortlist_for_full_validation",
            float(row.get("mean_cpcv_path_pnl_r", 0.0) or 0.0),
            float(row.get("total_pnl_r", 0.0) or 0.0),
        ),
        reverse=True,
    )
    shortlisted = [row for row in ranked if row["governance_decision"] == "shortlist_for_full_validation"]
    payload = {
        "status": "complete",
        "family": "exit_behavior_research",
        "candidate": "market_state_setup_quality_v1",
        "entry_definition": "exclude_weak_or_grindy_continuation + pre_entry_breakout_quality_gate",
        "governance": {
            "entries_frozen": True,
            "setup_filters_frozen": True,
            "models_trained": 0,
            "holdout_status": "locked",
            "no_path_specific_exits": True,
            "point_in_time_required": True,
            "promotion_blocked": True,
            "search_budget": {
                "max_trials": len(EXIT_VARIANTS),
                "max_model_trains": 0,
                "max_holdout_runs": 0,
                "runtime": "bounded_existing_artifact_replay",
            },
        },
        "stage_1_trade_path_diagnostics": _path_diagnostic_summary(path_rows),
        "stage_2_archetypes": archetypes,
        "stage_3_candidate_exit_families": exit_families,
        "stage_4_bounded_replay": [_strip(row) for row in ranked],
        "stage_5_full_validation_shortlist": [_strip(row) for row in shortlisted],
        "full_validation_requirements": [
            "walk_forward",
            "purged_cpcv",
            "dsr_psr_with_real_trial_count",
            "translation",
            "drawdown_analysis",
            "leakage_audit",
        ],
        "failure_attribution": _failure_attribution(path_rows, ranked),
        "rationale": {
            "why_this_family": "Current blocker is exit-path robustness for frozen high-quality BNR continuation entries.",
            "why_not_entry_changes": "Entry set is frozen to avoid reopening the exhausted benchmark/search layer.",
            "why_not_tp_sl_optimization_first": "Path behavior must explain failure modes before direct TP/SL tuning is allowed.",
        },
    }
    run_id = f"exit-behavior-{uuid4().hex[:12]}"
    output_dir = REPORTS_DIR / "runs" / run_id / "exit_behavior_research"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    artifact = REPORTS_DIR / "exit_behavior_research_market_state_v1.json"
    artifact.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["artifact_path"] = str(artifact)
    payload["run_artifact_path"] = str(output_dir / "summary.json")
    return payload


def run_structure_partial_exit_replay_cycle(state: dict[str, Any]) -> dict[str, Any]:
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        return {"status": "pending", "reason": "diagnostic_unavailable", "diagnostic": diagnostic}

    config = _diagnostic_config(state, str(diagnostic["source_path"]))
    bars = regular_session(
        load_ohlcv_file(
            config.source_path,
            symbol=config.symbol,
            timeframe=config.timeframe,
            timezone=config.timezone,
        )
    )
    entries = _frozen_v1_entries(diagnostic)
    rows = [_evaluate_exit_policy(name, entries, bars, config.horizon_bars) for name in STRUCTURE_PARTIAL_REPLAY_VARIANTS]
    _attach_dsr_psr(rows)
    baseline = next(row for row in rows if row["variant"] == "fixed_1_5r_tp_sl_baseline")
    baseline_tail = _right_tail_r(baseline["trade_results"])
    for row in rows:
        row["exit_family"] = _variant_family(row["variant"])
        row["right_tail_p90_r"] = _right_tail_r(row["trade_results"])
        row["gate_results"] = _structure_partial_gates(row, baseline, baseline_tail)
        row["decision"] = "advance_candidate" if all(row["gate_results"].values()) else "reject"

    ranked = sorted(
        rows,
        key=lambda row: (
            row["decision"] == "advance_candidate",
            float(row.get("right_tail_p90_r", 0.0) or 0.0),
            float(row.get("mean_cpcv_path_pnl_r", 0.0) or 0.0),
            float(row.get("total_pnl_r", 0.0) or 0.0),
        ),
        reverse=True,
    )
    accepted = next((row for row in ranked if row["decision"] == "advance_candidate"), None)
    payload = {
        "status": "complete",
        "family": "exit_behavior_research",
        "subfamily": "partial_profit_runner_and_structure_trailing",
        "candidate": "market_state_setup_quality_v1",
        "entry_definition": "exclude_weak_or_grindy_continuation + pre_entry_breakout_quality_gate",
        "governance": {
            "entries_frozen": True,
            "trade_count_required": len(entries),
            "setup_filters_frozen": True,
            "models_trained": 0,
            "holdout_status": "locked",
            "no_path_specific_exits": True,
            "point_in_time_required": True,
            "search_budget": {
                "max_trials": len(STRUCTURE_PARTIAL_REPLAY_VARIANTS),
                "allowed_families": ["partial_profit_runner", "structure_trailing"],
                "disallowed_knobs": ["entry changes", "setup filters", "model training", "holdout"],
            },
        },
        "strict_requirements": {
            "lower_drawdown_than_baseline": True,
            "better_right_tail_than_baseline": True,
            "cpcv_pbo_max": 0.25,
            "worst_path_min_r": -5.0,
            "dsr_psr": "pass",
            "trade_count": len(entries),
        },
        "trial_count": len(rows),
        "ranked_trials": [_strip(row) for row in ranked],
        "best_trial": _strip(ranked[0]) if ranked else None,
        "accepted_trial": _strip(accepted) if accepted else None,
        "batch_decision": "accept" if accepted else "revise",
    }
    run_id = f"exit-structure-partial-{uuid4().hex[:12]}"
    output_dir = REPORTS_DIR / "runs" / run_id / "exit_behavior_research"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    artifact = REPORTS_DIR / "exit_behavior_structure_partial_replay_market_state_v1.json"
    artifact.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["artifact_path"] = str(artifact)
    payload["run_artifact_path"] = str(output_dir / "summary.json")
    return payload


def _trade_path_diagnostics(row: dict[str, Any], bars: Any, horizon_bars: int) -> dict[str, Any]:
    pd = _require_pandas()
    entry_time = pd.Timestamp(row["entry_time"])
    future = bars[bars.index >= entry_time].head(horizon_bars)
    direction = str(row["direction"])
    entry = float(row["entry_price"])
    stop = float(row["stop_price"])
    risk = max(abs(entry - stop), 0.25)
    favorable: list[float] = []
    adverse: list[float] = []
    closes: list[float] = []
    ranges: list[float] = []
    bodies: list[float] = []
    signs: list[int] = []
    for _, bar in future.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        open_ = float(bar["open"])
        close = float(bar["close"])
        favorable.append((high - entry) / risk if direction == "long" else (entry - low) / risk)
        adverse.append((low - entry) / risk if direction == "long" else (entry - high) / risk)
        closes.append(close)
        ranges.append(max(high - low, 0.0))
        bodies.append(abs(close - open_))
        signs.append(1 if close > open_ else -1 if close < open_ else 0)

    mfe_r = max(favorable) if favorable else 0.0
    mae_r = min(adverse) if adverse else 0.0
    time_to_mfe = favorable.index(mfe_r) + 1 if favorable else 0
    time_to_mae = adverse.index(mae_r) + 1 if adverse else 0
    pnl_r = float(row.get("pnl_r", 0.0) or 0.0)
    first_window = max(1, min(6, len(closes)))
    first_closes = closes[:first_window]
    favorable_closes = [
        close for close in first_closes if (close > entry if direction == "long" else close < entry)
    ]
    follow_through = len(favorable_closes) / first_window if first_window else 0.0
    first_body = mean(bodies[:3]) if len(bodies) >= 3 else (mean(bodies) if bodies else 0.0)
    late_body = mean(bodies[-3:]) if len(bodies) >= 3 else first_body
    tempo_decay = 1.0 - min(late_body / max(first_body, 1e-9), 1.0) if bodies else 0.0
    first_range = mean(ranges[:3]) if len(ranges) >= 3 else (mean(ranges) if ranges else 0.0)
    late_range = mean(ranges[-3:]) if len(ranges) >= 3 else first_range
    volatility_collapse = 1.0 - min(late_range / max(first_range, 1e-9), 1.0) if ranges else 0.0
    giveback = max(0.0, mfe_r - pnl_r)
    reversal = giveback / max(mfe_r, 1e-9) if mfe_r > 0 else 0.0
    alternating = sum(1 for left, right in zip(signs[:-1], signs[1:]) if left and right and left != right)
    alternating_ratio = alternating / max(len(signs) - 1, 1) if signs else 0.0
    liquidity_rejection = float(row.get("distance_to_recent_high_low", 1.0) or 1.0) <= 0.25 and float(row.get("wick_rejection_ratio_recent", 0.0) or 0.0) >= 0.55
    continuation_quality = max(0.0, min(1.0, (follow_through + max(mfe_r, 0.0) / 1.5 + (1.0 - reversal)) / 3.0))
    taxonomy = _outcome_taxonomy(
        mfe_r=mfe_r,
        mae_r=mae_r,
        pnl_r=pnl_r,
        time_to_mfe=time_to_mfe,
        time_to_mae=time_to_mae,
        follow_through=follow_through,
        tempo_decay=tempo_decay,
        volatility_collapse=volatility_collapse,
        reversal=reversal,
        alternating_ratio=alternating_ratio,
        liquidity_rejection=liquidity_rejection,
    )
    return {
        "candidate_id": str(row["candidate_id"]),
        "session_date": str(row["session_date"]),
        "market_state": str(row.get("market_state", "")),
        "setup_quality": str(row.get("setup_quality", "")),
        "direction": direction,
        "pnl_r": pnl_r,
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "time_to_mfe_bars": int(time_to_mfe),
        "time_to_mae_bars": int(time_to_mae),
        "follow_through_persistence": float(follow_through),
        "tempo_decay": float(tempo_decay),
        "volatility_collapse": float(volatility_collapse),
        "reversal_behavior": float(reversal),
        "liquidity_rejection": bool(liquidity_rejection),
        "continuation_quality_after_entry": float(continuation_quality),
        "alternating_bar_ratio_path": float(alternating_ratio),
        "trade_outcome_taxonomy": taxonomy,
    }


def _outcome_taxonomy(**metrics: Any) -> str:
    mfe = float(metrics["mfe_r"])
    mae = float(metrics["mae_r"])
    pnl = float(metrics["pnl_r"])
    if mae <= -0.75 and int(metrics["time_to_mae"]) <= 3 and mfe < 0.35:
        return "fast_failure"
    if mfe >= 1.5 and pnl > 0.5:
        return "runner"
    if mfe >= 1.0 and float(metrics["follow_through"]) >= 0.55 and pnl > 0:
        return "strong_continuation"
    if mfe >= 0.75 and pnl <= 0 and float(metrics["reversal"]) >= 0.70:
        return "late_reversal"
    if mfe < 0.45 and mae <= -0.50 and bool(metrics["liquidity_rejection"]):
        return "fake_breakout"
    if pnl < 0 and abs(mae) < 0.75 and float(metrics["time_to_mae"]) > 3:
        return "slow_bleed"
    if float(metrics["alternating_ratio"]) >= 0.45 or float(metrics["tempo_decay"]) >= 0.55 or float(metrics["volatility_collapse"]) >= 0.55:
        return "chop_decay"
    return "strong_continuation" if pnl > 0 else "slow_bleed"


def _archetype_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trade_outcome_taxonomy"])].append(row)
    summary = []
    for archetype, group in grouped.items():
        pnls = [float(row["pnl_r"]) for row in group]
        summary.append(
            {
                "archetype": archetype,
                "trade_count": len(group),
                "session_count": len({row["session_date"] for row in group}),
                "total_pnl_r": float(sum(pnls)),
                "avg_pnl_r": float(mean(pnls)) if pnls else 0.0,
                "median_pnl_r": float(median(pnls)) if pnls else 0.0,
                "avg_mfe_r": float(mean(float(row["mfe_r"]) for row in group)),
                "avg_mae_r": float(mean(float(row["mae_r"]) for row in group)),
                "avg_follow_through_persistence": float(mean(float(row["follow_through_persistence"]) for row in group)),
                "avg_tempo_decay": float(mean(float(row["tempo_decay"]) for row in group)),
                "avg_volatility_collapse": float(mean(float(row["volatility_collapse"]) for row in group)),
                "liquidity_rejection_count": sum(1 for row in group if row["liquidity_rejection"]),
            }
        )
    return sorted(summary, key=lambda row: (row["total_pnl_r"], -row["trade_count"]))


def _path_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(row["pnl_r"]) for row in rows]
    return {
        "trade_count": len(rows),
        "session_count": len({row["session_date"] for row in rows}),
        "total_pnl_r": float(sum(pnls)),
        "avg_mfe_r": float(mean(float(row["mfe_r"]) for row in rows)) if rows else 0.0,
        "avg_mae_r": float(mean(float(row["mae_r"]) for row in rows)) if rows else 0.0,
        "avg_time_to_mfe_bars": float(mean(float(row["time_to_mfe_bars"]) for row in rows)) if rows else 0.0,
        "avg_time_to_mae_bars": float(mean(float(row["time_to_mae_bars"]) for row in rows)) if rows else 0.0,
        "avg_follow_through_persistence": float(mean(float(row["follow_through_persistence"]) for row in rows)) if rows else 0.0,
        "avg_tempo_decay": float(mean(float(row["tempo_decay"]) for row in rows)) if rows else 0.0,
        "avg_volatility_collapse": float(mean(float(row["volatility_collapse"]) for row in rows)) if rows else 0.0,
        "taxonomy_counts": dict(Counter(str(row["trade_outcome_taxonomy"]) for row in rows)),
    }


def _candidate_exit_families(archetypes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["archetype"]: row for row in archetypes}
    specs = [
        ("scratch_exits", ["fast_failure", "fake_breakout"], "Early failure/fakeout support."),
        ("structure_trailing", ["late_reversal", "runner"], "Protects favorable path giveback while allowing continuation."),
        ("partial_profit_runner", ["runner", "late_reversal"], "Tests whether runners pay for late reversals."),
        ("time_stop", ["slow_bleed", "chop_decay"], "Targets low-progress holding periods."),
        ("break_even_policy", ["late_reversal"], "Targets post-MFE giveback without changing entries."),
        ("volatility_based_exit", ["chop_decay", "slow_bleed"], "Targets post-entry volatility collapse and decay."),
        ("liquidity_rejection_exit", ["fake_breakout"], "Targets rejection near recent high/low context."),
    ]
    rows = []
    for family, triggers, reason in specs:
        support = sum(int(by_name.get(name, {}).get("trade_count", 0) or 0) for name in triggers)
        loss = sum(float(by_name.get(name, {}).get("total_pnl_r", 0.0) or 0.0) for name in triggers)
        rows.append(
            {
                "family": family,
                "diagnostic_support_trades": support,
                "supported_archetypes": triggers,
                "supported_archetype_pnl_r": loss,
                "rationale": reason,
                "initial_status": "eligible_for_bounded_replay" if support >= 10 else "diagnostic_only_low_support",
            }
        )
    return rows


def _replay_decision(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    pbo = float(row.get("pbo", 1.0) or 1.0)
    dd = float(row.get("max_drawdown_r", 0.0) or 0.0)
    mean_path = float(row.get("mean_cpcv_path_pnl_r", 0.0) or 0.0)
    worst = min(float(path.get("total_pnl_r", 0.0) or 0.0) for path in row.get("worst_3_cpcv_paths", []) or [{"total_pnl_r": 0.0}])
    if (
        row.get("dsr_psr", {}).get("status") == "pass"
        and pbo <= min(0.25, float(baseline.get("pbo", 1.0) or 1.0))
        and dd > float(baseline.get("max_drawdown_r", -999.0) or -999.0)
        and mean_path > 0
        and worst > -5.0
    ):
        return "shortlist_for_full_validation"
    return "reject_or_diagnostic_only"


def _structure_partial_gates(row: dict[str, Any], baseline: dict[str, Any], baseline_tail: float) -> dict[str, bool]:
    worst_path = min(
        float(path.get("total_pnl_r", 0.0) or 0.0)
        for path in row.get("worst_3_cpcv_paths", []) or [{"total_pnl_r": 0.0}]
    )
    return {
        "lower_drawdown": float(row.get("max_drawdown_r", 0.0) or 0.0) > float(baseline.get("max_drawdown_r", 0.0) or 0.0),
        "better_right_tail": float(row.get("right_tail_p90_r", 0.0) or 0.0) > baseline_tail,
        "cpcv_pbo_lte_0_25": float(row.get("pbo", 1.0) or 1.0) <= 0.25,
        "worst_path_gt_minus_5r": worst_path > -5.0,
        "dsr_psr_pass": row.get("dsr_psr", {}).get("status") == "pass",
        "trade_count_212": int(row.get("trade_count", 0) or 0) == 212,
        "holdout_locked": True,
        "models_trained_zero": True,
    }


def _right_tail_r(trades: list[dict[str, Any]]) -> float:
    pnls = sorted(float(row.get("pnl_r", 0.0) or 0.0) for row in trades)
    if not pnls:
        return 0.0
    idx = min(len(pnls) - 1, int(round(0.90 * (len(pnls) - 1))))
    return float(pnls[idx])


def _variant_family(variant: str) -> str:
    if "scratch" in variant:
        return "scratch_exits"
    if "partial" in variant or "runner" in variant:
        return "partial_profit_runner"
    if "trailing" in variant or "structure_trail" in variant:
        return "structure_trailing"
    if "time_stop" in variant:
        return "time_stop"
    if "breakeven" in variant:
        return "break_even_policy"
    if "tempo" in variant:
        return "volatility_based_exit"
    if "recent_high_low" in variant:
        return "liquidity_rejection_exit"
    return "fixed_tp_sl_baseline"


def _failure_attribution(path_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    losing = [row for row in path_rows if float(row["pnl_r"]) <= 0]
    taxonomy_counts = Counter(str(row["trade_outcome_taxonomy"]) for row in losing)
    rejected = [row for row in replay_rows if row.get("governance_decision") != "shortlist_for_full_validation"]
    return {
        "dominant_losing_archetypes": dict(taxonomy_counts.most_common(5)),
        "rejected_families": [
            {
                "variant": row["variant"],
                "exit_family": row["exit_family"],
                "reason": "Failed shortlist gates for PBO, drawdown, DSR/PSR, or CPCV tail.",
            }
            for row in rejected
        ],
        "known_risks": [
            "Exit improvements may be path-shape-specific unless validated out of sample.",
            "Small frozen entry sample limits archetype reliability.",
            "Exit replay uses existing bar data and must not become holdout-driven tuning.",
        ],
    }


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "trade_results"}


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("exit behavior research requires pandas") from exc
    return pd
