from __future__ import annotations

import json
from typing import Any

from trading_ml.config import load_bnr_config, load_research_program_config
from trading_ml.paths import REPORTS_DIR
from trading_ml.research_os import build_research_director_plan
from trading_ml.registry import REGISTRY_PATH


RESEARCH_FAMILIES = [
    "setup",
    "model",
    "feature",
    "threshold",
    "label",
    "sample_expansion",
    "subtype",
    "policy_gate",
    "policy_meta",
    "tail_path_cleanup",
    "exit_behavior_research",
    "candidate_universe_expansion",
    "validation_window",
    "holdout_confirmation",
]

FAMILY_ACTIONS = {
    "setup": "run_setup_search_cycle",
    "model": "run_model_family_cycle",
    "feature": "run_market_structure_feature_cycle",
    "threshold": "run_threshold_cycle",
    "label": "run_label_policy_cycle",
    "sample_expansion": "run_sample_expansion_cycle",
    "subtype": "run_subtype_cycle",
    "policy_gate": "run_policy_gate_cycle",
    "policy_meta": "run_policy_meta_cycle",
    "tail_path_cleanup": "run_tail_path_cleanup_cycle",
    "exit_behavior_research": "run_exit_behavior_research_cycle",
    "candidate_universe_expansion": "run_candidate_universe_expansion_cycle",
    "validation_window": "advance_to_reserved_validation_window",
    "holdout_confirmation": "advance_to_holdout_confirmation",
}

FAMILY_RISK_LEVEL = {
    "setup": "medium",
    "model": "medium",
    "feature": "medium",
    "threshold": "high",
    "label": "medium",
    "sample_expansion": "medium",
    "subtype": "medium",
    "policy_gate": "high",
    "policy_meta": "high",
    "tail_path_cleanup": "high",
    "exit_behavior_research": "medium",
    "candidate_universe_expansion": "medium",
    "validation_window": "low",
    "holdout_confirmation": "low",
}

DEFAULT_SEARCH_BUDGETS = {
    "setup": {
        "max_trials": 12,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["horizon_bars", "break_buffer_points"],
        "disallowed_knobs": ["holdout data", "post-hoc threshold edits"],
    },
    "model": {
        "max_trials": 2,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["model_family"],
        "disallowed_knobs": ["label edits", "holdout data"],
    },
    "feature": {
        "max_trials": 8,
        "max_cycles": 3,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["feature_family"],
        "disallowed_knobs": ["holdout data", "trial-specific thresholds"],
    },
    "threshold": {
        "max_trials": 5,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["decision_threshold"],
        "disallowed_knobs": ["feature edits", "label edits", "holdout data"],
    },
    "label": {
        "max_trials": 8,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["horizon_bars", "stop_multiple", "target_multiple"],
        "disallowed_knobs": ["holdout data", "model escalation"],
    },
    "sample_expansion": {
        "max_trials": 4,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["latest_trigger_time", "break_buffer_points"],
        "disallowed_knobs": [
            "holdout data",
            "model escalation",
            "threshold-only edits",
        ],
    },
    "subtype": {
        "max_trials": 6,
        "max_cycles": 2,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["setup_subtype"],
        "disallowed_knobs": ["exact clock filters", "holdout data"],
    },
    "policy_gate": {
        "max_trials": 4,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["break_quality_policy"],
        "disallowed_knobs": ["exact clock filters", "holdout data"],
    },
    "policy_meta": {
        "max_trials": 4,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["reclaim_meta_policy"],
        "disallowed_knobs": ["exact clock filters", "holdout data"],
    },
    "tail_path_cleanup": {
        "max_trials": 4,
        "max_cycles": 1,
        "max_runtime": "bounded_by_batch",
        "allowed_knobs": ["structural tail-path BNR cleanup policies"],
        "disallowed_knobs": ["model changes", "threshold-only search", "holdout data"],
    },
    "exit_behavior_research": {
        "max_trials": 8,
        "max_cycles": 1,
        "max_runtime": "bounded_existing_artifact_replay",
        "allowed_knobs": ["point-in-time exit behavior families"],
        "disallowed_knobs": [
            "entry changes",
            "model training",
            "holdout data",
            "path-specific exits",
        ],
    },
    "candidate_universe_expansion": {
        "max_trials": 8,
        "max_cycles": 1,
        "max_runtime": "bounded_universe_diagnostic",
        "allowed_knobs": ["canonical candidate universe definitions"],
        "disallowed_knobs": [
            "model training",
            "holdout data",
            "undocumented discretionary filters",
        ],
    },
    "validation_window": {
        "max_trials": 1,
        "max_cycles": 1,
        "max_runtime": "single_reserved_run",
        "allowed_knobs": ["boundary source"],
        "disallowed_knobs": ["spec edits"],
    },
    "holdout_confirmation": {
        "max_trials": 1,
        "max_cycles": 1,
        "max_runtime": "single_holdout_run",
        "allowed_knobs": ["holdout source"],
        "disallowed_knobs": ["any benchmark edits"],
    },
}

FALSIFICATION_RULES = {
    "setup": "If setup variants do not improve walk-forward utility without weakening CPCV, kill setup search for this benchmark.",
    "model": "If model changes do not improve calibration and CPCV while preserving utility, return to the frozen baseline family.",
    "feature": "If feature-family variants do not improve walk-forward and utility versus the frozen benchmark, abandon this feature line.",
    "threshold": "If threshold variants only move trade count or fail to clear the losing confidence bucket, abandon threshold work.",
    "label": "If label variants improve apparent utility but degrade calibration, CPCV, or mean path PnL, reject label redesign.",
    "sample_expansion": "If sample expansion does not improve CPCV tail stability without destroying mean path PnL, reject the current BNR sample geometry.",
    "subtype": "If subtype variants do not improve CPCV tail without reducing median/mean path PnL, abandon subtype cleanup.",
    "policy_gate": "If policy gates reduce tail loss by removing breadth or lowering mean path PnL, reject the gate.",
    "policy_meta": "If meta-policy filters reduce tail loss but binary utility or CPCV breadth deteriorates, reject the filter.",
    "tail_path_cleanup": "If structural cleanup does not improve the same worst CPCV paths without collapsing trade count, kill the current BNR benchmark line.",
    "exit_behavior_research": "If trade-path archetypes do not produce an exit family that improves PBO/drawdown without weakening CPCV tail, park exit behavior research for this candidate.",
    "candidate_universe_expansion": "If expanded candidate definitions increase raw count without cluster-adjusted effective sample support, reject universe expansion.",
    "validation_window": "If the frozen benchmark fails reserved validation, return to diagnostics without holdout access.",
    "holdout_confirmation": "If the frozen benchmark fails holdout confirmation, reject promotion and freeze further edits.",
}

KILL_CRITERIA = {
    "setup": ["kill after 3 CPCV failures with no walk-forward or utility improvement"],
    "model": [
        "kill model escalation while CPCV is failing",
        "kill if simpler baseline has equal or better utility",
    ],
    "feature": ["kill feature line after 2 cycles with no utility or CPCV improvement"],
    "threshold": [
        "kill if losses are already concentrated in the high-confidence bucket",
        "kill if threshold lift is only lower breadth",
    ],
    "label": ["kill if calibration or mean path PnL deteriorates"],
    "sample_expansion": [
        "kill if expanded sample keeps the same CPCV tail paths",
        "kill if added breadth materially lowers mean path PnL",
    ],
    "subtype": [
        "kill if tail improves but mean path PnL deteriorates",
        "kill if subtype support is sample-limited",
    ],
    "policy_gate": [
        "kill if gate is tied to a narrow clock artifact",
        "kill if breadth falls below benchmark tolerance",
    ],
    "policy_meta": [
        "kill if binary utility exceeds sized/filtered utility across cycles"
    ],
    "tail_path_cleanup": [
        "kill current BNR benchmark if the same CPCV tail paths persist after structural cleanup",
        "kill cleanup policy if trade count falls below 70% of benchmark",
    ],
    "exit_behavior_research": [
        "kill if bounded exit replay cannot improve PBO and drawdown together",
        "kill if improvement is driven by path-specific or non-point-in-time behavior",
    ],
    "candidate_universe_expansion": [
        "kill if lineage, deduplication, or point-in-time audit fails",
        "kill if effective sample size does not improve despite raw expansion",
    ],
    "validation_window": ["kill promotion path if reserved validation fails"],
    "holdout_confirmation": ["kill promotion if any hard gate fails on holdout"],
}


def build_program_state() -> dict[str, Any]:
    config = load_research_program_config()["program"]
    return {
        "name": config["name"],
        "mandate": config["mandate"],
        "primary_objective": config["primary_objective"],
        "promotion_standard": config["promotion_standard"],
        "principles": list(config["principles"]["items"]),
        "market_structure_questions": list(
            config["market_structure_questions"]["items"]
        ),
        "sample_floor": dict(config.get("sample_floor", {})),
        "workstreams": {
            name: {
                "required_artifacts": list(spec["required_artifacts"]),
                "status": "pending",
                "missing_artifacts": list(spec["required_artifacts"]),
            }
            for name, spec in config["workstreams"].items()
        },
        "program_gaps": [],
        "priority_mandates": [],
        "institutional_status": "pending",
        "next_step_plan": {},
    }


def evaluate_program_state(state: dict[str, Any]) -> dict[str, Any]:
    program = dict(state.get("program_state", {}) or build_program_state())
    workstreams = dict(program.get("workstreams", {}))
    present = _present_artifacts(state)
    program_gaps: list[str] = []

    for name, spec in workstreams.items():
        required = list(spec.get("required_artifacts", []))
        missing = [artifact for artifact in required if artifact not in present]
        spec["missing_artifacts"] = missing
        spec["status"] = "ready" if not missing else "partial"
        if missing:
            program_gaps.append(f"{name}: missing {', '.join(missing)}")

    priority_mandates = _priority_mandates(state, program_gaps)
    next_step_plan = _next_step_plan(state, program_gaps)
    next_step_plan = build_research_director_plan(state, next_step_plan)
    program["workstreams"] = workstreams
    program["program_gaps"] = program_gaps
    program["priority_mandates"] = priority_mandates
    program["next_step_plan"] = next_step_plan
    program["institutional_status"] = (
        "ready_for_confirmation" if not program_gaps else "research_os_incomplete"
    )
    return program


def _present_artifacts(state: dict[str, Any]) -> set[str]:
    stage2 = dict(state.get("stage2_result", {}))
    audit = dict(state.get("audit_summary", {}))
    translation = dict(state.get("translation_summary", {}))
    feature_diagnostics = dict(state.get("feature_diagnostics", {}))
    technical_review = dict(state.get("technical_review", {}))
    present: set[str] = set()

    if state.get("data_manifest_loaded"):
        present.add("manifest")
    if state.get("stage2_config", {}).get("source_path"):
        present.add("opening_hours_cache")
        present.add("session_policy")
        present.add("timestamp_policy")
    if state.get("bnr_spec"):
        present.update({"setup_ontology", "market_structure_playbook"})
    if stage2.get("label_summary"):
        present.update({"label_registry", "objective_matrix"})
    if state.get("label_spec"):
        present.add("ambiguity_policy")
    if state.get("feature_spec"):
        present.add("feature_family_registry")
        present.add("market_structure_pack")
    if stage2.get("feature_validation") or feature_diagnostics:
        present.add("feature_validation")
    if stage2.get("market_structure_lab", {}).get("status") == "complete":
        present.add("failure_taxonomy")
    if any(
        "regime" in name for name in feature_diagnostics.get("family_scores", {}).keys()
    ):
        present.add("regime_pack")
    if state.get("model_spec"):
        present.add("baseline_ladder")
    if stage2.get("model_summary", {}).get("metrics"):
        present.add("calibration_checks")
        present.add("uncertainty_review")
    if stage2.get("model_diagnostics", {}).get("bucket_monotonicity") is not None:
        present.add("bucket_monotonicity")
    walk_forward = audit.get("walk_forward")
    if walk_forward not in (None, "pending"):
        present.add("walk_forward")
    purging = audit.get("purging")
    if purging not in (None, "pending"):
        present.add("purging")
    multiple_testing = dict(audit.get("multiple_testing", {}))
    if multiple_testing and multiple_testing.get("status") not in (None, "pending"):
        present.add("multiple_testing")
    if technical_review.get("score_to_signal_contract_required") is True or translation:
        present.add("signal_to_order_contract")
    if state.get("backtest_summary"):
        present.add("event_driven_backtest")
        present.add("risk_budget")
        present.add("position_sizing_policy")
    if state.get("controller_state", {}).get("benchmark_name"):
        present.add("registry")
        present.add("promotion_policy")
        present.add("shadow_mode_plan")
        present.add("monitoring_plan")
    random_signal_plumbing = audit.get("random_signal_plumbing")
    if isinstance(random_signal_plumbing, dict):
        if random_signal_plumbing.get("status") not in (None, "pending"):
            present.add("random_signal_plumbing")
    elif random_signal_plumbing not in (None, "pending"):
        present.add("random_signal_plumbing")
    if state.get("stage2_config", {}).get("symbol") == "MNQ":
        present.add("slippage_model")
        present.add("overlap_policy")
    return present


def _priority_mandates(state: dict[str, Any], program_gaps: list[str]) -> list[str]:
    mandates: list[str] = []
    audit = dict(state.get("audit_summary", {}))
    translation = dict(state.get("translation_summary", {}))
    sample_floor = dict((state.get("program_state", {}) or {}).get("sample_floor", {}))
    stage2 = dict(state.get("stage2_result", {}))
    sessions = int(stage2.get("data_quality", {}).get("sessions", 0) or 0)
    min_sessions = int(sample_floor.get("min_exploration_sessions", 0) or 0)
    if "thesis_lab: missing failure_taxonomy" in program_gaps:
        mandates.append(
            "Build a BNR failure taxonomy from false positives, false negatives, and no-trade cases."
        )
    if "model_lab: missing bucket_monotonicity" in program_gaps:
        mandates.append(
            "Add prediction bucket monotonicity and score-decile diagnostics before more model escalation."
        )
    if "validation_lab: missing random_signal_plumbing" in program_gaps:
        mandates.append(
            "Keep a plumbing-control backtest in the standard workflow for every promoted benchmark."
        )
    if "portfolio_lab: missing position_sizing_policy" in program_gaps:
        mandates.append(
            "Formalize score-to-size translation before calling the system institutional-grade."
        )
    if "production_lab: missing shadow_mode_plan" in program_gaps:
        mandates.append(
            "Define shadow-mode and rollback requirements now, before widening autonomous search."
        )
    if audit.get("walk_forward") == "fail":
        mandates.append(
            "Treat market-structure and label quality as the next research lever, not threshold churn."
        )
    if min_sessions and sessions and sessions < min_sessions:
        mandates.append(
            f"Do not trust exploratory winners below the {min_sessions}-session sample floor."
        )
    if translation.get("status") == "pass":
        mandates.append(
            "Preserve execution realism and utility scoring as hard gates in every cycle."
        )
    return mandates


def _next_step_plan(state: dict[str, Any], program_gaps: list[str]) -> dict[str, Any]:
    stage2 = dict(state.get("stage2_result", {}))
    audit = dict(state.get("audit_summary", {}))
    translation = dict(state.get("translation_summary", {}))
    controller = dict(state.get("controller_state", {}))
    bnr_spec = dict(state.get("bnr_spec", {}))
    frozen = dict(bnr_spec.get("frozen_benchmark", {}))
    benchmark = {
        "feature_family": frozen.get("feature_family"),
        "model_family": frozen.get("model_family"),
        "threshold": frozen.get("threshold"),
        "sizing_policy": frozen.get("sizing_policy"),
        "regime_throttle_policy": frozen.get("regime_throttle_policy"),
        "policy_gate": frozen.get("policy_gate"),
        "policy_meta": frozen.get("policy_meta"),
    }
    if not state.get("stage2_config", {}).get("source_path"):
        return {
            "status": "blocked",
            "lane": "data_lab",
            "action": "load_boundary_scoped_source",
            "reason": "No exploration-scoped source is active.",
            "controller_override": {},
            "stage2_overrides": {},
            "success_criteria": ["boundary-scoped source selected"],
        }
    if "validation_lab: missing random_signal_plumbing" in program_gaps:
        return {
            "status": "ready",
            "lane": "validation_lab",
            "action": "maintain_random_signal_plumbing",
            "reason": "Validation protocol still needs plumbing control artifacts.",
            "controller_override": {},
            "stage2_overrides": {},
            "success_criteria": ["random signal baseline remains worse than benchmark"],
        }

    walk_forward = dict(audit.get("walk_forward", {}))
    cpcv = dict(audit.get("cpcv", {}))
    deflated_sharpe = dict(audit.get("deflated_sharpe", {}))
    model_diagnostics = dict(stage2.get("model_diagnostics", {}))
    shap = dict(model_diagnostics.get("shap_analysis", {}))
    calibration = dict(model_diagnostics.get("calibration_review", {}))
    top_features = [
        row.get("feature") for row in shap.get("top_features", []) if row.get("feature")
    ]
    utility_status = translation.get("status")
    best_translation = dict(translation.get("best_threshold", {}))
    utility_gap = best_translation.get("utility_gap_vs_binary")

    if walk_forward.get("status") == "pass" and cpcv.get("status") == "fail":
        attribution = _load_cpcv_failure_attribution()
        family_decision = _select_next_research_family(
            state=state,
            attribution=attribution,
            top_features=top_features,
            calibration=calibration,
            market_structure_lab=dict(stage2.get("market_structure_lab", {})),
            current_blocker="cpcv_tail_path_fragility",
        )
        return {
            "status": "ready",
            "lane": "robustness_rebuild",
            "action": family_decision["action"],
            "reason": "Walk-forward passes but CPCV fails; improve robustness before model escalation.",
            "controller_override": {
                "active_family": family_decision["selected_family"]
            },
            "stage2_overrides": {
                "feature_family": benchmark.get("feature_family")
                or state.get("stage2_config", {}).get("feature_family")
            },
            "current_blocker": "cpcv_tail_path_fragility",
            "selected_family": family_decision["selected_family"],
            "candidate_families": family_decision["candidate_families"],
            "family_scores": family_decision["family_scores"],
            "diagnostic_evidence_used": family_decision["evidence_used"],
            "evidence_used": family_decision["evidence_used"],
            "evidence_not_used": family_decision["evidence_not_used"],
            "rejected_alternatives": family_decision["rejected_alternatives"],
            "families_rejected": family_decision["families_rejected"],
            "why_selected": family_decision["why_selected"],
            "why_rejected": family_decision["why_rejected"],
            "known_risks": family_decision["known_risks"],
            "falsification_rule": family_decision["falsification_rule"],
            "kill_criteria": family_decision["kill_criteria"],
            "search_budget": family_decision["search_budget"],
            "approval_required": family_decision["approval_required"],
            "multi_cycle_memory": family_decision["multi_cycle_memory"],
            "rationale": family_decision["rationale"],
            "benchmark_status": family_decision.get("benchmark_status", "active"),
            "recommended_next_action": family_decision.get("recommended_next_action"),
            "benchmark_contract": benchmark,
            "success_criteria": [
                "cpcv status = pass",
                "mean_total_pnl_r > 0",
                "walk_forward stays pass",
            ],
        }
    if (
        walk_forward.get("status") == "pass"
        and cpcv.get("status") == "pass"
        and deflated_sharpe.get("status") == "fail"
    ):
        return {
            "status": "ready",
            "lane": "research_cycle",
            "action": "reduce_search_complexity_and_retest",
            "reason": "Validation is acceptable, but DSR says observed performance is not yet above multiple-testing inflation.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {
                "feature_family": benchmark.get("feature_family")
                or state.get("stage2_config", {}).get("feature_family")
            },
            "benchmark_contract": benchmark,
            "success_criteria": [
                "deflated sharpe probability >= 0.95",
                "walk_forward stays pass",
                "cpcv stays pass",
            ],
        }
    if (
        utility_status == "pass"
        and calibration.get("status") == "fail"
        and benchmark.get("sizing_policy") not in {None, "binary_threshold_v1"}
    ):
        return {
            "status": "ready",
            "lane": "translation_lab",
            "action": "fallback_to_conservative_sizing",
            "reason": "Utility passes, but calibration is weak for probability-driven sizing.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {
                "feature_family": benchmark.get("feature_family")
                or state.get("stage2_config", {}).get("feature_family")
            },
            "benchmark_contract": benchmark,
            "success_criteria": [
                "binary or tiered sizing matches or beats current utility",
                "walk_forward stays pass",
                "cpcv stays pass",
            ],
        }
    if walk_forward.get("status") == "fail":
        return {
            "status": "ready",
            "lane": "feature_lab",
            "action": "run_market_structure_feature_cycle",
            "reason": "Generalization is weak; improve the setup representation before more policy or model churn.",
            "controller_override": {"active_family": "feature"},
            "stage2_overrides": {
                "feature_family": benchmark.get("feature_family")
                or state.get("stage2_config", {}).get("feature_family")
            },
            "benchmark_contract": benchmark,
            "success_criteria": [
                "walk_forward status = pass",
                "utility improves versus frozen benchmark",
            ],
        }
    if (
        walk_forward.get("status") == "pass"
        and cpcv.get("status") == "pass"
        and utility_status == "pass"
    ):
        return {
            "status": "ready",
            "lane": "validation_lab",
            "action": "advance_to_reserved_validation_window",
            "reason": "Exploration benchmark is robust enough to test on untouched validation data.",
            "controller_override": {"active_family": "validation_window"},
            "stage2_overrides": {},
            "benchmark_contract": benchmark,
            "success_criteria": [
                "validation utility positive",
                "validation CPCV pass or not applicable",
                "no benchmark edits during validation",
            ],
        }
    if (
        walk_forward.get("status") == "pass"
        and cpcv.get("status") == "pass"
        and utility_status == "fail"
    ):
        return {
            "status": "ready",
            "lane": "translation_lab",
            "action": "run_translation_policy_cycle",
            "reason": "Robustness is acceptable, but score-to-trade translation is weak.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {
                "feature_family": benchmark.get("feature_family")
                or state.get("stage2_config", {}).get("feature_family")
            },
            "benchmark_contract": benchmark,
            "success_criteria": [
                "utility improves versus frozen benchmark",
                "walk_forward stays pass",
                "cpcv stays pass",
            ],
        }
    if utility_gap is not None and float(utility_gap) > 3.0:
        return {
            "status": "ready",
            "lane": "translation_lab",
            "action": "audit_sizing_lift",
            "reason": "Sized utility materially exceeds binary utility; confirm the gain is not mostly sizing-driven.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {
                "feature_family": benchmark.get("feature_family")
                or state.get("stage2_config", {}).get("feature_family")
            },
            "benchmark_contract": benchmark,
            "success_criteria": [
                "sizing lift is stable",
                "binary utility remains positive",
                "calibration is acceptable for probability-based sizing",
            ],
        }
    return {
        "status": "ready",
        "lane": "research_cycle",
        "action": "continue_governed_benchmark_iteration",
        "reason": "No higher-priority protocol failure detected.",
        "controller_override": controller,
        "stage2_overrides": {},
        "benchmark_contract": benchmark,
        "success_criteria": [
            "improvement beats frozen benchmark on utility and robustness"
        ],
    }


def _load_cpcv_failure_attribution() -> dict[str, Any]:
    path = REPORTS_DIR / "cpcv_failure_attribution.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _select_next_research_family(
    *,
    state: dict[str, Any],
    attribution: dict[str, Any],
    top_features: list[str],
    calibration: dict[str, Any],
    market_structure_lab: dict[str, Any],
    current_blocker: str,
) -> dict[str, Any]:
    evidence = _build_family_evidence(
        attribution=attribution,
        top_features=top_features,
        calibration=calibration,
        market_structure_lab=market_structure_lab,
        current_blocker=current_blocker,
    )
    memory = _build_multi_cycle_memory(state)
    persistent_tail = dict(memory.get("persistent_tail_failure", {}) or {})
    if persistent_tail.get("status") == "active":
        evidence["persistent_tail_failure"] = True
        evidence["persistent_tail_paths"] = persistent_tail.get("path_ids", [])
        evidence["families_already_failed"] = persistent_tail.get("families", [])
        evidence["tail_cleanup_failed"] = bool(
            persistent_tail.get("tail_cleanup_failed")
        )
        evidence["label_geometry_failed"] = bool(
            persistent_tail.get("label_geometry_failed")
        )
        evidence["sample_expansion_failed"] = bool(
            persistent_tail.get("sample_expansion_failed")
        )
        evidence["used"]["persistent_tail_failure"] = persistent_tail
        evidence["not_used"].append(
            "families that already failed the same CPCV tail paths"
        )
    if _benchmark_exhaustion_triggered(persistent_tail):
        return _benchmark_exhaustion_decision(
            evidence=evidence, memory=memory, persistent_tail=persistent_tail
        )
    family_scores = [
        _score_family(family, evidence, memory) for family in RESEARCH_FAMILIES
    ]
    ranked = sorted(
        family_scores,
        key=lambda row: (row["risk_adjusted_score"], row["diagnostic_alignment"]),
        reverse=True,
    )
    exhausted = set(memory.get("exhausted_families", []) or [])
    if evidence.get("tail_cleanup_failed"):
        exhausted.discard("label")
        exhausted.discard("sample_expansion")
    selected = next(
        (row for row in ranked if row["family"] not in exhausted), ranked[0]
    )
    selected_family = str(selected["family"])
    rejected = [
        {
            "family": row["family"],
            "reason": _rejection_reason(row, selected_family),
            "risk_adjusted_score": row["risk_adjusted_score"],
        }
        for row in ranked
        if str(row["family"]) != selected_family
    ]
    return {
        "selected_family": selected_family,
        "action": FAMILY_ACTIONS[selected_family],
        "candidate_families": [
            {
                "family": row["family"],
                "expected_value": _score_band(float(row["expected_upside"])),
                "risk": FAMILY_RISK_LEVEL[str(row["family"])],
                "reason": row["reason"],
                "risk_adjusted_score": row["risk_adjusted_score"],
            }
            for row in ranked
        ],
        "family_scores": ranked,
        "evidence_used": evidence["used"],
        "evidence_not_used": evidence["not_used"],
        "rejected_alternatives": rejected,
        "families_rejected": [str(row["family"]) for row in ranked[1:]],
        "why_selected": selected["reason"],
        "why_rejected": {
            str(row["family"]): _rejection_reason(row, selected_family)
            for row in ranked[1:]
        },
        "known_risks": _known_risks(selected_family, evidence),
        "falsification_rule": FALSIFICATION_RULES[selected_family],
        "kill_criteria": KILL_CRITERIA[selected_family],
        "search_budget": _budget_for_family(selected_family),
        "approval_required": _approval_required(selected_family),
        "multi_cycle_memory": memory,
        "rationale": selected["reason"],
    }


def _build_family_evidence(
    *,
    attribution: dict[str, Any],
    top_features: list[str],
    calibration: dict[str, Any],
    market_structure_lab: dict[str, Any],
    current_blocker: str,
) -> dict[str, Any]:
    dominant = dict(attribution.get("dominant_failure_axes", {}) or {})
    subtype = dict(dominant.get("subtype") or {})
    time_of_day = dict(dominant.get("time_of_day") or {})
    probability_bucket = dict(dominant.get("probability_bucket") or {})
    used = {
        "current_blocker": current_blocker,
        "cpcv_attribution_status": attribution.get("status", "missing"),
        "failure_type": attribution.get("failure_summary", {}).get("failure_type"),
        "subtype": subtype,
        "time_of_day": time_of_day,
        "probability_bucket": probability_bucket,
        "calibration_status": calibration.get("status"),
        "top_features": top_features[:5],
        "market_structure_status": market_structure_lab.get("status"),
    }
    not_used = [
        "holdout results",
        "exact clock bucket as a direct gate",
        "post-hoc family preference not backed by diagnostics",
    ]
    return {
        "used": used,
        "not_used": not_used,
        "current_blocker": current_blocker,
        "subtype_support": int(subtype.get("trade_count", 0) or 0),
        "subtype_key": subtype.get("key"),
        "time_bucket_is_exact": bool(time_of_day.get("key"))
        and ":" in str(time_of_day.get("key")),
        "high_confidence_loss": probability_bucket.get("key") == "[0.65,1.00]"
        and float(probability_bucket.get("total_pnl_r", 0.0) or 0.0) < 0,
        "calibration_failed": calibration.get("status") == "fail",
        "reclaim_features_active": any(
            name in top_features
            for name in [
                "reclaim_body_strength",
                "post_reclaim_close_strength",
                "reclaim_close_location",
            ]
        ),
        "break_features_active": any(
            name in top_features
            for name in [
                "break_close_distance_to_zone",
                "break_efficiency_ratio",
                "first_break_close_excess_points",
            ]
        ),
    }


def _score_family(
    family: str, evidence: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    scores = {
        "evidence_support": 1.0,
        "expected_upside": 1.0,
        "overfit_risk": 2.0,
        "implementation_cost": 2.0,
        "diagnostic_alignment": 1.0,
        "prior_failure_count": float(memory.get("failure_counts", {}).get(family, 0)),
        "promotion_relevance": 1.0,
    }
    reason = "Baseline option retained for completeness."

    if family in {"model", "holdout_confirmation"}:
        scores["overfit_risk"] = 5.0 if family == "model" else 1.0
        scores["diagnostic_alignment"] = 0.0
        scores["expected_upside"] = 0.0
        reason = "CPCV failure blocks model escalation and reserved-data progression."
    elif family == "subtype":
        support = int(evidence.get("subtype_support", 0) or 0)
        scores["evidence_support"] = 4.0 if support >= 10 else 2.0
        scores["expected_upside"] = 3.0 if support >= 10 else 2.0
        scores["diagnostic_alignment"] = 5.0 if support >= 10 else 2.0
        scores["overfit_risk"] = 3.0 if support >= 10 else 5.0
        scores["promotion_relevance"] = 4.0
        reason = "Worst-path losses are concentrated by setup subtype, so test structural subtype hypotheses before narrower gates."
    elif family == "validation_window":
        if evidence.get("current_blocker") == "cpcv_tail_path_fragility":
            scores["evidence_support"] = 0.0
            scores["expected_upside"] = 0.0
            scores["diagnostic_alignment"] = 0.0
            scores["promotion_relevance"] = 0.0
            scores["overfit_risk"] = 4.0
            reason = "CPCV failure blocks reserved validation-window use until robustness improves."
        else:
            scores["evidence_support"] = 2.0
            scores["expected_upside"] = 3.0
            scores["diagnostic_alignment"] = 2.0
            reason = "Validation slicing may explain path fragility once hard robustness gates are no longer vetoing."
    elif family == "policy_gate":
        active = bool(evidence.get("break_features_active"))
        scores["evidence_support"] = 3.0 if active else 1.5
        scores["expected_upside"] = 3.0
        scores["diagnostic_alignment"] = 3.0 if active else 1.0
        scores["overfit_risk"] = 5.0 if evidence.get("time_bucket_is_exact") else 4.0
        scores["promotion_relevance"] = 3.0
        reason = "Break-quality gates may reduce the tail, but exact-bucket attribution makes overfit risk high."
    elif family == "policy_meta":
        active = bool(evidence.get("reclaim_features_active"))
        scores["evidence_support"] = 3.0 if active else 1.5
        scores["expected_upside"] = 3.0
        scores["diagnostic_alignment"] = 4.0 if active else 1.0
        scores["overfit_risk"] = 4.0
        scores["promotion_relevance"] = 3.0
        reason = "Reclaim-quality features point to a policy filter, but it must not hide weak breadth."
        if evidence.get("persistent_tail_failure"):
            scores["expected_upside"] = 0.0
            scores["diagnostic_alignment"] = 0.0
            scores["overfit_risk"] = 5.0
            reason = "Persistent CPCV tail paths already survived policy-meta work; do not repeat broad meta filtering."
    elif family == "tail_path_cleanup":
        persistent = bool(evidence.get("persistent_tail_failure"))
        cleanup_failed = bool(evidence.get("tail_cleanup_failed"))
        scores["evidence_support"] = (
            0.0 if cleanup_failed else (5.0 if persistent else 2.0)
        )
        scores["expected_upside"] = (
            0.0 if cleanup_failed else (4.0 if persistent else 2.0)
        )
        scores["diagnostic_alignment"] = (
            0.0 if cleanup_failed else (5.0 if persistent else 2.0)
        )
        scores["overfit_risk"] = 5.0 if cleanup_failed else 4.0
        scores["implementation_cost"] = 2.0
        scores["promotion_relevance"] = 0.0 if cleanup_failed else 5.0
        reason = (
            "Structural tail-path cleanup already failed; move to label geometry/sample expansion instead of repeating it."
            if cleanup_failed
            else (
                "The same worst CPCV paths persist across families; run structural tail-path cleanup against those exact paths."
                if persistent
                else "Tail-path cleanup is reserved for repeated CPCV path failures."
            )
        )
    elif family == "threshold":
        high_conf_loss = bool(evidence.get("high_confidence_loss"))
        calibration_failed = bool(evidence.get("calibration_failed"))
        scores["evidence_support"] = (
            3.0 if high_conf_loss and calibration_failed else 1.0
        )
        scores["expected_upside"] = 2.0 if calibration_failed else 1.0
        scores["diagnostic_alignment"] = 3.0 if calibration_failed else 1.0
        scores["overfit_risk"] = 5.0
        scores["implementation_cost"] = 1.0
        scores["promotion_relevance"] = 2.0
        reason = "Threshold-only work is weak because losses occur in the high-confidence bucket unless calibration itself fails."
    elif family == "label":
        calibration_failed = bool(evidence.get("calibration_failed"))
        cleanup_failed = bool(evidence.get("tail_cleanup_failed"))
        label_failed = bool(evidence.get("label_geometry_failed"))
        scores["evidence_support"] = (
            0.0
            if label_failed
            else (4.0 if cleanup_failed else (3.0 if calibration_failed else 1.0))
        )
        scores["expected_upside"] = (
            0.0
            if label_failed
            else (4.0 if cleanup_failed else (3.0 if calibration_failed else 1.0))
        )
        scores["diagnostic_alignment"] = (
            0.0
            if label_failed
            else (5.0 if cleanup_failed else (4.0 if calibration_failed else 1.0))
        )
        scores["overfit_risk"] = 5.0 if label_failed else 3.0
        scores["promotion_relevance"] = (
            0.0 if label_failed else (4.0 if cleanup_failed else 3.0)
        )
        reason = (
            "Label geometry already failed the same CPCV tail; move to sample expansion."
            if label_failed
            else (
                "Tail cleanup failed to rescue repeated CPCV paths; change label geometry or sample breadth before another policy filter."
                if cleanup_failed
                else "Label redesign is justified only when high-confidence losses pair with failed calibration."
            )
        )
    elif family == "sample_expansion":
        label_failed = bool(evidence.get("label_geometry_failed"))
        persistent = bool(evidence.get("persistent_tail_failure"))
        expansion_failed = bool(evidence.get("sample_expansion_failed"))
        scores["evidence_support"] = (
            0.0
            if expansion_failed
            else (5.0 if label_failed else (1.0 if persistent else 0.5))
        )
        scores["expected_upside"] = (
            0.0
            if expansion_failed
            else (4.0 if label_failed else (1.0 if persistent else 0.5))
        )
        scores["diagnostic_alignment"] = (
            0.0
            if expansion_failed
            else (5.0 if label_failed else (1.0 if persistent else 0.5))
        )
        scores["overfit_risk"] = 5.0 if expansion_failed else 3.0
        scores["implementation_cost"] = 3.0
        scores["promotion_relevance"] = (
            0.0 if expansion_failed else (5.0 if label_failed else 1.0)
        )
        reason = (
            "Sample expansion already failed the same CPCV tail; do not spend more compute widening this benchmark."
            if expansion_failed
            else (
                "Tail cleanup and label geometry failed the same CPCV paths; test whether BNR needs wider sample geometry before killing the line."
                if label_failed
                else "Sample expansion is reserved for persistent tail failures after structural and label geometry attempts."
            )
        )
    elif family == "feature":
        scores["evidence_support"] = 2.0
        scores["expected_upside"] = 3.0
        scores["diagnostic_alignment"] = 2.0
        scores["overfit_risk"] = 3.0
        scores["promotion_relevance"] = 3.0
        reason = "Broader structural features are the fallback when attribution is incomplete or too narrow."
    elif family == "setup":
        scores["evidence_support"] = 1.5
        scores["expected_upside"] = 2.0
        scores["diagnostic_alignment"] = 1.0
        scores["overfit_risk"] = 4.0
        scores["implementation_cost"] = 4.0
        reason = "Setup search is expensive and should wait unless failures point to the setup definition itself."

    risk_adjusted = (
        scores["evidence_support"]
        + scores["expected_upside"]
        + scores["diagnostic_alignment"]
        + scores["promotion_relevance"]
        - scores["overfit_risk"]
        - (0.5 * scores["implementation_cost"])
        - (1.5 * scores["prior_failure_count"])
    )
    return {
        "family": family,
        **scores,
        "risk_adjusted_score": round(risk_adjusted, 3),
        "reason": reason,
    }


def _benchmark_exhaustion_triggered(persistent_tail: dict[str, Any]) -> bool:
    if persistent_tail.get("status") != "active":
        return False
    families = set(persistent_tail.get("families", []) or [])
    if len(families) < 3:
        return False
    if not bool(persistent_tail.get("dsr_failed_after_real_trials")):
        return False
    if bool(persistent_tail.get("any_hard_gate_passed")):
        return False
    return True


def _benchmark_exhaustion_decision(
    *,
    evidence: dict[str, Any],
    memory: dict[str, Any],
    persistent_tail: dict[str, Any],
) -> dict[str, Any]:
    families = sorted(set(persistent_tail.get("families", []) or []))
    reason = (
        "Same top 3 CPCV worst paths persisted across 3+ distinct families, DSR stayed failed with real n_trials, "
        "and no accepted trial passed hard gates."
    )
    rejected = [
        {
            "family": family,
            "reason": "Benchmark exhaustion rule supersedes further family search.",
            "risk_adjusted_score": None,
        }
        for family in RESEARCH_FAMILIES
    ]
    return {
        "selected_family": None,
        "action": "park_bnr_benchmark_definition",
        "candidate_families": [],
        "family_scores": [],
        "evidence_used": evidence["used"],
        "evidence_not_used": evidence["not_used"],
        "rejected_alternatives": rejected,
        "families_rejected": list(RESEARCH_FAMILIES),
        "why_selected": reason,
        "why_rejected": {
            family: "Benchmark is exhausted or structurally fragile under the hard kill rule."
            for family in RESEARCH_FAMILIES
        },
        "known_risks": [
            "continuing this exact benchmark definition likely burns compute on a structurally fragile tail"
        ],
        "falsification_rule": "Do not resume this benchmark definition unless the setup definition or market-structure hypothesis changes materially.",
        "kill_criteria": [
            "same top 3 CPCV worst paths across 3+ families",
            "DSR failed after real n_trials",
            "no accepted trial passed hard CPCV/DSR gates",
        ],
        "search_budget": {
            "max_trials": 0,
            "max_cycles": 0,
            "allowed_knobs": [],
            "disallowed_knobs": ["all knobs for this benchmark definition"],
        },
        "approval_required": None,
        "multi_cycle_memory": memory,
        "rationale": {
            "benchmark_status": "exhausted_or_structurally_fragile",
            "persistent_tail_paths": persistent_tail.get("path_ids", []),
            "families_failed": families,
            "run_ids": persistent_tail.get("run_ids", []),
        },
        "benchmark_status": "exhausted_or_structurally_fragile",
        "recommended_next_action": "park current BNR benchmark; open market_structure/setup redesign only if the deterministic spec changes materially",
    }


def _build_multi_cycle_memory(state: dict[str, Any]) -> dict[str, Any]:
    killed = set(state.get("killed_families", []) or [])
    failure_counts: dict[str, int] = {}
    accepted: list[str] = []
    rejected: list[str] = []
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines()[-250:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            result = dict(record.get("result", {}) or {})
            accounting = dict(result.get("trial_accounting", {}) or {})
            family = accounting.get("family") or _family_from_experiment_id(
                str(record.get("experiment_id", ""))
            )
            if family not in RESEARCH_FAMILIES:
                continue
            decision = str(record.get("decision", ""))
            if decision == "accept":
                accepted.append(str(family))
            elif decision == "reject":
                rejected.append(str(family))
                failure_counts[str(family)] = failure_counts.get(str(family), 0) + 1
    exhausted = sorted(
        {family for family, count in failure_counts.items() if count >= 3} | killed
    )
    persistent_tail = _persistent_tail_failure_from_artifacts()
    return {
        "failure_counts": failure_counts,
        "accepted_families": sorted(set(accepted)),
        "recent_rejected_families": sorted(set(rejected)),
        "killed_families": sorted(killed),
        "exhausted_families": exhausted,
        "persistent_tail_failure": persistent_tail,
    }


def _persistent_tail_failure_from_artifacts() -> dict[str, Any]:
    runs_root = REPORTS_DIR / "runs"
    if not runs_root.exists():
        return {"status": "inactive", "reason": "missing_runs"}
    observations: list[dict[str, Any]] = []
    for artifact_dir in sorted(
        runs_root.glob("*/node_artifacts"), key=lambda path: path.stat().st_mtime
    )[-40:]:
        search_files = sorted(artifact_dir.glob("*_search_controller_agent_*.json"))
        audit_files = sorted(artifact_dir.glob("*_audit_agent_*.json"))
        if not search_files or not audit_files:
            continue
        try:
            search = json.loads(search_files[-1].read_text(encoding="utf-8")).get(
                "payload", {}
            )
            audit = json.loads(audit_files[-1].read_text(encoding="utf-8")).get(
                "payload", {}
            )
        except json.JSONDecodeError:
            continue
        cpcv = dict(audit.get("cpcv", {}) or {})
        if cpcv.get("status") != "fail":
            continue
        dsr = dict(audit.get("deflated_sharpe", {}) or {})
        path_ids = [
            str(row.get("path_id"))
            for row in cpcv.get("worst_paths", [])[:3]
            if row.get("path_id")
        ]
        if not path_ids:
            continue
        family = str(
            search.get("executed_research_family")
            or search.get("search_results", {}).get("family")
            or "unknown"
        )
        observations.append(
            {
                "family": family,
                "batch_decision": str(
                    search.get("search_results", {}).get("batch_decision") or ""
                ),
                "accepted_trial": bool(
                    search.get("search_results", {}).get("accepted_trial")
                ),
                "dsr_status": str(dsr.get("status") or ""),
                "dsr_n_trials": int(dsr.get("n_trials", 0) or 0),
                "hard_gate_passed": cpcv.get("status") == "pass"
                and dsr.get("status") == "pass",
                "path_ids": path_ids,
                "signature": "|".join(path_ids),
                "run_id": artifact_dir.parent.name,
            }
        )
    signatures: dict[str, dict[str, Any]] = {}
    for obs in observations:
        bucket = signatures.setdefault(
            obs["signature"],
            {"families": set(), "runs": [], "path_ids": obs["path_ids"]},
        )
        bucket["families"].add(obs["family"])
        bucket["runs"].append(obs["run_id"])
    for signature, bucket in signatures.items():
        families = sorted(bucket["families"])
        if len(families) >= 2:
            tail_cleanup_failed = any(
                obs["family"] == "tail_path_cleanup"
                and obs.get("batch_decision") != "accept"
                for obs in observations
                if obs.get("signature") == signature
            )
            label_geometry_failed = any(
                obs["family"] == "label"
                for obs in observations
                if obs.get("signature") == signature
            )
            sample_expansion_failed = any(
                obs["family"] == "sample_expansion"
                for obs in observations
                if obs.get("signature") == signature
            )
            matching = [
                obs for obs in observations if obs.get("signature") == signature
            ]
            dsr_failed_after_real_trials = any(
                obs.get("dsr_status") == "fail"
                and int(obs.get("dsr_n_trials", 0) or 0) > 1
                for obs in matching
            )
            any_hard_gate_passed = any(
                bool(obs.get("hard_gate_passed")) for obs in matching
            )
            return {
                "status": "active",
                "signature": signature,
                "path_ids": bucket["path_ids"],
                "families": families,
                "run_ids": bucket["runs"][-5:],
                "tail_cleanup_failed": tail_cleanup_failed,
                "label_geometry_failed": label_geometry_failed,
                "sample_expansion_failed": sample_expansion_failed,
                "dsr_failed_after_real_trials": dsr_failed_after_real_trials,
                "any_hard_gate_passed": any_hard_gate_passed,
                "blocked_families": [
                    "model",
                    "threshold",
                    "policy_meta",
                    "validation_window",
                    "holdout_confirmation",
                ],
                "allowed_families": [
                    "tail_path_cleanup",
                    "subtype",
                    "label",
                    "sample_expansion",
                    "policy_gate",
                ],
            }
    return {
        "status": "inactive",
        "reason": "no_repeated_tail_signature",
        "observations": observations[-5:],
    }


def _family_from_experiment_id(experiment_id: str) -> str | None:
    for family in sorted(RESEARCH_FAMILIES, key=len, reverse=True):
        if (
            f"-{family}-" in experiment_id
            or f"-{family.replace('_', '-')}-" in experiment_id
        ):
            return family
    return None


def _budget_for_family(family: str) -> dict[str, Any]:
    budget = dict(DEFAULT_SEARCH_BUDGETS[family])
    try:
        config = load_bnr_config()
    except Exception:
        config = {}
    config_key = f"{family}_search_v1"
    if family == "setup":
        config_key = "search_v1"
    if family == "tail_path_cleanup":
        config_key = "tail_path_cleanup_search_v1"
    configured = dict(config.get(config_key, {}) or {})
    if configured.get("max_batch_trials"):
        budget["max_trials"] = int(configured["max_batch_trials"])
    budget["minimum_evidence_required"] = [
        "walk_forward pass",
        "purging pass",
        "CPCV improves or passes",
        "DSR/PSR uses full batch trial count",
    ]
    return budget


def _approval_required(family: str) -> str | None:
    if family in {
        "setup",
        "model",
        "feature",
        "threshold",
        "label",
        "sample_expansion",
        "subtype",
        "policy_gate",
        "policy_meta",
        "tail_path_cleanup",
    }:
        return "search_space_approval"
    if family == "holdout_confirmation":
        return "holdout_unlock_approval"
    if family == "validation_window":
        return "frozen_spec_approval"
    return None


def _known_risks(family: str, evidence: dict[str, Any]) -> list[str]:
    risks = [f"{FAMILY_RISK_LEVEL[family]} overfit risk"]
    if evidence.get("time_bucket_is_exact") and family in {
        "policy_gate",
        "policy_meta",
        "subtype",
    }:
        risks.append("exact time bucket may be path-composition noise")
    if evidence.get("subtype_support", 0) < 20 and family == "subtype":
        risks.append("subtype support may be sample-limited")
    if family == "threshold" and evidence.get("high_confidence_loss"):
        risks.append("high-confidence loss bucket limits threshold-only upside")
    if family == "tail_path_cleanup":
        risks.append(
            "cleanup must improve exact failed CPCV paths, not just headline utility"
        )
    if family == "sample_expansion":
        risks.append(
            "expanded breadth may add lower-quality setups and reduce mean path PnL"
        )
    return risks


def _score_band(value: float) -> str:
    if value >= 4.0:
        return "high"
    if value >= 2.5:
        return "medium"
    return "low"


def _rejection_reason(row: dict[str, Any], selected_family: str) -> str:
    if row["prior_failure_count"] >= 3:
        return "Family is exhausted by prior failed cycles."
    if row["diagnostic_alignment"] <= 1:
        return f"Lower diagnostic alignment than {selected_family}."
    if row["overfit_risk"] >= 5:
        return "Overfit risk is too high for the current evidence."
    return f"Lower risk-adjusted score than {selected_family}."
