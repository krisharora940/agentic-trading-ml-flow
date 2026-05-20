from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_ml.paths import REPORTS_DIR


def build_setup_redesign_plan(state: dict[str, Any]) -> dict[str, Any]:
    cpcv = _read_json(REPORTS_DIR / "cpcv_failure_attribution.json")
    large_sample = _read_json(REPORTS_DIR / "large_sample_failure_map.json")
    market_structure = _latest_market_structure_lab()
    plan = dict(state.get("next_step_plan", {}) or {})
    rationale = dict(plan.get("rationale", {}) or {})

    return {
        "status": "ready_for_setup_redesign",
        "benchmark_status": "exhausted_or_structurally_fragile",
        "parked_benchmark": "current_bnr_benchmark_definition",
        "reason": plan.get("why_selected") or "Current benchmark failed hard exhaustion criteria.",
        "evidence_used": {
            "persistent_tail_paths": rationale.get("persistent_tail_paths", []),
            "families_failed": rationale.get("families_failed", []),
            "cpcv_failure_summary": cpcv.get("failure_summary", {}),
            "dominant_failure_axes": cpcv.get("dominant_failure_axes", {}),
            "market_structure_lab": market_structure,
            "large_sample_failure_map": large_sample.get("failure_map", {}),
        },
        "redesign_principles": [
            "model BNR as semi-discretionary market-structure interpretation, not pure deterministic geometry",
            "represent trend quality, volatility behavior, candle speed, sequence cleanliness, liquidity context, and chop state",
            "separate continuation from auction-repair behavior",
            "avoid exact clock buckets as primary filters",
            "do not use model, threshold, or policy-meta changes as substitutes for setup definition",
            "keep holdout locked",
        ],
        "research_focus": {
            "old_abstraction": "static BNR event geometry",
            "new_abstraction": "BNR geometry plus evolving intraday auction state and setup-quality interpretation",
            "primary_questions": [
                "How do we represent clean versus ugly price-action sequences?",
                "How do we identify choppy or unpredictable auctions before entry?",
                "How do trend quality, candle speed, volatility expansion, and local liquidity structure change BNR expectancy?",
                "Which state features explain high-confidence losses without becoming post-hoc exact filters?",
            ],
        },
        "new_research_family": _market_state_quality_research_family(),
        "latent_feature_families": _latent_feature_families(),
        "candidate_setup_hypotheses": _candidate_hypotheses(cpcv, large_sample, market_structure),
        "bounded_search_budget": {
            "max_trials": 4,
            "max_cycles": 1,
            "max_full_validations": 1,
            "max_cpcv_runs": 1,
            "allowed_knobs": [
                "market-state feature family",
                "setup-quality feature family",
                "auction-state classifier",
                "continuation versus repair ontology",
            ],
            "disallowed_knobs": [
                "holdout data",
                "exact clock filters",
                "model-family escalation",
                "threshold-only rescue",
            ],
        },
        "falsification_rule": (
            "If redesigned setup ontology does not change the repeated CPCV worst-path signature "
            "or collapses trade count below the sample floor, park this BNR family."
        ),
        "approval_checkpoint": "setup_redesign_mandate_approval",
        "next_graph_mode": "planning_only_until_setup_mandate_approved",
    }


def _candidate_hypotheses(cpcv: dict[str, Any], large_sample: dict[str, Any], market_structure: dict[str, Any]) -> list[dict[str, Any]]:
    axes = dict(cpcv.get("dominant_failure_axes", {}) or {})
    failures = list(market_structure.get("failure_taxonomy", []) or [])
    by_subtype = list(dict(large_sample.get("failure_map", {}) or {}).get("by_subtype", []) or [])
    return [
        {
            "name": "market_state_quality_classifier",
            "rationale": "The discretionary edge depends on continuous auction interpretation; encode trend, volatility, chop, speed, and sequence cleanliness before another geometry change.",
            "evidence": {
                "persistent_tail": axes,
                "market_structure_questions": market_structure.get("market_structure_questions", []),
            },
            "feature_families": [
                "trend_quality",
                "volatility_expansion_contraction",
                "candle_speed_momentum_texture",
                "sequence_cleanliness",
                "local_liquidity_structure",
                "intraday_auction_state",
                "chop_avoidance",
            ],
            "kill_rule": "Kill if state features do not change high-confidence loss attribution or repeated CPCV worst paths.",
        },
        {
            "name": "bnr_continuation_only",
            "rationale": "No-follow-through dominates failure taxonomy; require explicit continuation evidence before entry.",
            "evidence": {
                "top_failure": failures[0] if failures else {},
                "probability_tail": axes.get("probability_bucket", {}),
            },
            "kill_rule": "Kill if trade count falls below budget floor or CPCV tail signature is unchanged.",
        },
        {
            "name": "bnr_repair_split",
            "rationale": "Deep retrace repair behaves like a different auction state and should not share one benchmark with continuation.",
            "evidence": {
                "cpcv_subtype_axis": axes.get("subtype", {}),
                "large_sample_subtypes": by_subtype,
            },
            "kill_rule": "Kill if repair split improves local PnL but repeats the same CPCV worst paths.",
        },
        {
            "name": "bnr_no_follow_through_filter",
            "rationale": "The setup needs a structural no-trade condition, not a model confidence override.",
            "evidence": {
                "market_structure_questions": market_structure.get("market_structure_questions", []),
            },
            "kill_rule": "Kill if filter reduces breadth without improving mean CPCV path PnL.",
        },
    ]


def _market_state_quality_research_family() -> dict[str, Any]:
    return {
        "family": "market_state_setup_quality",
        "priority_hypothesis": "market_state_quality_classifier",
        "objective": (
            "Represent the discretionary BNR filter as market-state and setup-quality classification "
            "from market open through candidate decision time."
        ),
        "not_a_trading_rule": True,
        "holdout_status": "locked",
        "candidate_features": {
            "candle_speed_volatility": [
                "bar_velocity_points_per_30s",
                "rolling_velocity_zscore",
                "range_expansion_rate",
                "pre_trigger_compression_ratio",
                "post_open_volatility_state",
            ],
            "wick_body_structure": [
                "body_to_range_ratio_sequence",
                "upper_lower_wick_pressure",
                "close_location_persistence",
                "rejection_wick_follow_through",
            ],
            "impulse_strength": [
                "consecutive_directional_closes",
                "impulse_efficiency",
                "volatility_adjusted_displacement",
                "acceleration_deceleration_slope",
            ],
            "flem_shape": [
                "flem_drive_speed",
                "flem_overlap_ratio",
                "flem_close_quality",
                "flem_stall_after_break",
            ],
            "pivot_shape": [
                "pivot_depth_vs_zone",
                "pivot_duration",
                "pivot_symmetry",
                "repair_speed",
                "failed_repair_count",
            ],
            "recent_highs_lows_liquidity": [
                "distance_to_recent_swing_high_low",
                "recent_swing_sweep_flag",
                "range_boundary_proximity",
                "local_liquidity_rejection_context",
            ],
            "day_structure": [
                "opening_drive_state",
                "trend_day_probability_proxy",
                "balance_area_width",
                "prior_session_range_location",
                "vwap_relation_if_available",
            ],
            "chop_avoidance": [
                "overlap_ratio",
                "alternating_close_count",
                "failed_directional_attempts",
                "range_contraction_persistence",
                "auction_clarity_score",
            ],
        },
        "candidate_labels": {
            "market_state_label": [
                "trend_continuation",
                "balanced_chop",
                "volatility_expansion",
                "volatility_contraction",
                "failed_directional_auction",
            ],
            "setup_quality_label": [
                "clean_tradeable_bnr",
                "marginal_bnr",
                "ugly_no_trade_bnr",
                "repair_state_bnr",
            ],
            "label_sources": [
                "pre-entry price action only",
                "fold-local outcome attribution",
                "optional human-reviewed annotation set for calibration, not holdout selection",
            ],
        },
        "quality_states": [
            {
                "state": "clean_continuation",
                "description": "Directional auction with efficient impulse, clean pullback/repair, and low chop.",
            },
            {
                "state": "auction_repair",
                "description": "Deep or complex repair that may need a separate ontology from continuation.",
            },
            {
                "state": "balanced_chop",
                "description": "Overlapping, low-clarity auction where technically valid BNRs should usually be avoided.",
            },
            {
                "state": "failed_directional",
                "description": "Break or reclaim sequence lacks follow-through and creates high-confidence false positives.",
            },
        ],
        "validation_plan": {
            "first_pass": "feature audit and fold-local market-state classifier on exploration data only",
            "required_gates": [
                "point-in-time feature construction",
                "walk-forward pass",
                "purged CPCV with changed worst-path signature",
                "DSR/PSR with actual n_trials",
                "calibration review by market-state bucket",
                "trade count does not collapse below budget floor",
            ],
            "diagnostics": [
                "high-confidence loss attribution by quality state",
                "CPCV worst-path comparison versus parked benchmark",
                "state distribution stability across months",
                "feature importance sign and stability review",
            ],
        },
        "falsification_rules": [
            "Falsify if quality states do not separate high-confidence losses from winners out of sample.",
            "Falsify if CPCV worst paths remain cpcv_010/cpcv_003/cpcv_002 after state features are included.",
            "Falsify if improved robustness comes mostly from lower trade count rather than better state discrimination.",
            "Falsify if labels require post-entry information or discretionary leakage.",
        ],
        "bounded_search_budget": {
            "max_trials": 4,
            "max_cycles": 1,
            "max_runtime_seconds": 1800,
            "max_full_validations": 1,
            "max_cpcv_runs": 1,
            "max_model_trains": 8,
            "allowed_knobs": [
                "market-state feature group",
                "setup-quality label taxonomy",
                "quality-state classifier target",
                "fold-local calibration method",
            ],
            "disallowed_knobs": [
                "holdout data",
                "exact clock filters",
                "post-entry features",
                "model escalation before linear/GBM baselines justify it",
                "threshold-only rescue",
                "reopening parked BNR benchmark geometry",
            ],
        },
    }


def _latent_feature_families() -> list[dict[str, Any]]:
    return [
        {
            "family": "trend_quality",
            "examples": ["directional persistence", "slope stability", "higher-high/lower-low alignment", "trend versus balance state"],
            "purpose": "avoid BNRs forming inside poor directional structure",
        },
        {
            "family": "volatility_expansion_contraction",
            "examples": ["pre-trigger compression", "post-break expansion", "range expansion rate", "volatility-adjusted displacement"],
            "purpose": "distinguish energized breaks from slow/noisy auctions",
        },
        {
            "family": "candle_speed_momentum_texture",
            "examples": ["distance traveled per bar", "body-to-wick persistence", "acceleration/deceleration", "consecutive directional closes"],
            "purpose": "represent auction urgency and momentum texture instead of static ATR only",
        },
        {
            "family": "sequence_cleanliness",
            "examples": ["overlap/chop ratio", "failed continuation count", "stalling after reclaim", "repair speed"],
            "purpose": "separate clean price-action sequences from ugly technically valid BNRs",
        },
        {
            "family": "local_liquidity_structure",
            "examples": ["distance to recent swing high/low", "local range boundary proximity", "sweep/rejection context"],
            "purpose": "capture trapped-participant and range-boundary context",
        },
        {
            "family": "intraday_auction_state",
            "examples": ["opening drive", "balanced auction", "trend day", "mean-reverting chop", "failed directional condition"],
            "purpose": "avoid firing during choppy or unpredictable day states",
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latest_market_structure_lab() -> dict[str, Any]:
    runs_root = REPORTS_DIR / "runs"
    if not runs_root.exists():
        return {}
    for artifact_dir in sorted(runs_root.glob("*/node_artifacts"), key=lambda path: path.stat().st_mtime, reverse=True):
        audit_files = sorted(artifact_dir.glob("*_audit_agent_*.json"))
        for audit_path in reversed(audit_files):
            try:
                payload = json.loads(audit_path.read_text(encoding="utf-8")).get("payload", {})
            except json.JSONDecodeError:
                continue
            market_structure = dict(payload.get("market_structure_lab", {}) or {})
            if market_structure.get("status") == "complete":
                return market_structure
    return {}
